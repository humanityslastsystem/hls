"""The in-build reference service of an agent build (``AgentProducerSpec.reference_service``): the
task's pinned vLLM reference, asleep on the build sandbox's GPUs, which the agent may ask to score
token sequences over a unix socket (``hls.reference.service``).

It is one harness role of the agent_build stage, a container on the task's measured baseline
image, which carries the reference's vLLM, and the stage's one shared GPU step: it runs on the GPUs
the sandbox holds (``share_gpus_of``), in a step of its own, outside the sandbox's seal. Its memory
is oversubscribed on the sandbox's (``share_memory_of``): its step's limit is its own declared
memory, the stage's budget does not count it, and the job's memory limit bounds the two together.
The sandbox's memory and the stage's are those of a build without it. The stage starts it once the sandbox has
reported, before the agent's clock starts, and stops it as soon as the agent's part ends, right
after the sandbox; ``StageContext.close`` is the backstop.

What the sandbox gets of it is the socket's directory, bound read-only at
``service_harness.SOCKET_DIR``, and ``service_harness.SOCKET_ENV_VAR`` naming the socket: no image,
no weights path of the service's, no script, no store path and no secret. The directory lies in the
sandbox role's own directory and holds nothing but the socket the service binds. The service's
container gets the weights the agent's data already copied to the node, read-only, the socket's
directory, writable, and its one script from the plan's checkout, read-only.

It never decides how the build ends: a service that cannot start, fails to load, does not become
ready within ``service_harness.READY_TIMEOUT_SEC``, ends, or cannot be reached from the sandbox is
recorded (``build.reference_service_unreachable``, ``build.reference_service_stopped``) and the
build goes on without it, graded as any build is. A service that was short of memory for a load
answers the agent with HTTP 507 and stays up, unloaded, for the agent to load again
(``build.reference_service_unloaded``); it holds no engine then, so nothing ends it. Its own ending is never part of a stage fault: the
stage reports the role itself and judges only the sandbox's. A service that is not ready is stopped
at once, since a half-built engine holds most of every card the agent works on; its socket's
directory stays bound, so a connection to the path the brief names then fails. Every wait on it is
bounded: each ``/status`` read by ``STATUS_TIMEOUT_SEC``, the probe from the sandbox by
``PROBE_TIMEOUT_SEC``, the start by the scheduler's own wait for a step, the stop by the stage's
grace."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict

from hls.agent import exec_protocol
from hls.provision.mounts import OFFLINE_ENV, WEIGHTS_TARGET
from hls.reference import service_harness as rs
from hls.spec.resources import ResourceShape
from hls.stages.engine_cache import ENGINE_CACHE_GIB
from hls.store.base import check_segment
from hls.substrate.cgroup import read_keyed
from hls.substrate.container import ContainerSpec, Mount, PinnedImage
from hls.substrate.faults import SharedMemoryEvidence, StepMemory
from hls.substrate.memory_watch import MemorySample, read_peak

if TYPE_CHECKING:
    from hls.agent.environment import SandboxEnvironment
    from hls.pipeline.stage import StageContext
    from hls.substrate.base import RoleHandle
    from hls.substrate.faults import RoleOutcome

log = logging.getLogger(__name__)

#: The service's role.
SERVICE = "reference-service"
#: The socket's directory, under the sandbox role's own directory, and the socket in it.
SOCKET_DIR_NAME = "reference"
SERVICE_SOCKET = f"{SOCKET_DIR_NAME}/{Path(rs.SOCKET_PATH).name}"
#: Seconds between the stage's reads of a starting service's ``/status``.
POLL_SEC = 5.0
#: Seconds the probe of the service from inside the sandbox may take. Generous: the probe tells a
#: reachable service from an unreachable one, and a deadline tight enough to trip on a loaded
#: sandbox would report the two alike.
PROBE_TIMEOUT_SEC = 120
#: The service role's CPUs: the verify role's (``hls.pipeline.pipelines.REFERENCE_SHAPE``), which
#: runs the same engine. Its step shares the sandbox's cores, so they are not counted against the
#: stage's.
SERVICE_CPUS = 8
#: Host memory the service holds, as a multiple of the checkpoint's bytes on disk. Sleep level 1
#: copies every rank's weights into pinned host memory, inside the role's memory: 1.06 times the
#: checkpoint's bytes for agentic_coding's at tensor parallel 8, the excess being layers every rank
#: holds whole. The rest of the factor covers the engine's worker processes.
HOST_MEMORY_FACTOR = 1.3
#: The job's batch step, where the runner runs, as the runner's watch observes it beside the service.
BATCH = "batch"
#: Characters of a failure's reason an event keeps.
_REASON_CHARS = 2000
#: Bytes at the end of the service's stderr a failure quotes from.
_TAIL_BYTES = 4096


class ReferenceServiceParams(BaseModel):
    """What the agent_build stage runs its reference service with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The image the service runs on, by its name in this instance's ``images.json``: the task's
    #: measured baseline image, which carries the reference's vLLM.
    image: Annotated[str, AfterValidator(check_segment)]
    #: The service role's shape (``service_shape``).
    shape: ResourceShape


def service_memory_gib(weights_bytes: int) -> int:
    """The service's memory, in GiB, for a checkpoint of *weights_bytes* on disk:
    ``HOST_MEMORY_FACTOR`` times it, rounded up."""
    return math.ceil(weights_bytes * HOST_MEMORY_FACTOR / 2**30)


def service_shape(sandbox: ResourceShape, weights_bytes: int) -> ResourceShape:
    """The service role's shape beside a sandbox of shape *sandbox*, for a checkpoint of
    *weights_bytes* on disk: every GPU of the sandbox, which it shares, ``SERVICE_CPUS`` (at most
    the sandbox's), its memory (``service_memory_gib``), and the room of an engine's built products
    in disk (``ENGINE_CACHE_GIB``), which its workdir holds. Raise ValueError for a sandbox with no
    GPU."""
    if sandbox.gpus < 1 or sandbox.gpu_kind is None:
        raise ValueError("the reference service runs on the sandbox's GPUs, and the sandbox has none")
    return ResourceShape(gpus=sandbox.gpus, gpu_kind=sandbox.gpu_kind, cpus=min(SERVICE_CPUS, sandbox.cpus),
                         memory_gib=service_memory_gib(weights_bytes), disk_gib=ENGINE_CACHE_GIB,
                         time_sec=sandbox.time_sec)


def shape_problems(service: ResourceShape, sandbox: ResourceShape) -> list[str]:
    """Why a service of shape *service* cannot run beside a sandbox of shape *sandbox*: it must
    share every GPU of the sandbox, of the same kind, and take no more CPUs than the sandbox has."""
    problems = []
    if service.gpus < 1 or service.gpus != sandbox.gpus or service.gpu_kind != sandbox.gpu_kind:
        problems.append(f"the reference service shares every GPU of the sandbox ({sandbox.gpus} "
                        f"{sandbox.gpu_kind}); it asks for {service.gpus} {service.gpu_kind}")
    if service.cpus > sandbox.cpus:
        problems.append(f"the reference service's {service.cpus} CPUs run on the sandbox's {sandbox.cpus}")
    return problems


class _NotReady(Exception):
    """The service will not become ready, and why."""


class BuildService:
    """The reference service of one agent_build attempt: its socket's directory, its role, the
    watch beside the agent and its stop."""

    def __init__(self, ctx: StageContext, params: ReferenceServiceParams, engine: rs.ServiceEngine,
                 sandbox_root: Path, emit: Callable[..., None]) -> None:
        #: The attempt's context.
        self.ctx = ctx
        #: The service's image and shape.
        self.params = params
        #: The engine it builds: the task's pinned vLLM reference's.
        self.engine = engine
        #: The socket's host directory, in the sandbox role's own directory.
        self.socket_dir = sandbox_root / SOCKET_DIR_NAME
        #: The socket, as the runner reaches it.
        self.host_socket = sandbox_root / SERVICE_SOCKET
        #: Emits the build's events.
        self.emit = emit
        #: The role, once started; None before, and when it never was.
        self.handle: RoleHandle | None = None
        #: When the role reported, by ``time.monotonic``; the ready timeout counts from it.
        self.launched_at: float | None = None
        #: When the role reported, as POSIX time, which an OOM's time is read against.
        self.launched_wall: float | None = None
        #: The job's own ``oom`` count when the role reported; None when it could not be read.
        self.job_ooms_at_start: int | None = None
        #: The ``/status`` reply that showed it ready; None until then.
        self.ready: dict[str, Any] | None = None
        #: Why it will never be ready; None while it may be, or is.
        self.failure: str | None = None
        #: How the role ended, once the stage reported it.
        self.outcome: RoleOutcome | None = None

    # the sandbox's side

    def mount(self) -> Mount:
        """The sandbox's one mount of the service: the socket's directory, read-only."""
        return Mount(self.socket_dir, rs.SOCKET_DIR)

    def env(self) -> dict[str, str]:
        """The sandbox's one variable of the service: the socket's path."""
        return {rs.SOCKET_ENV_VAR: rs.SOCKET_PATH}

    def prepare(self) -> None:
        """Make the socket's directory, 0700, before the sandbox starts, so its bind has a source
        whatever becomes of the service."""
        self.socket_dir.mkdir(mode=0o700)
        self.socket_dir.chmod(0o700)

    # the role

    def start(self, sandbox: RoleHandle, image: PinnedImage, data: list[Mount], checkout: Path,
              shape: ResourceShape) -> None:
        """Start the service's role on the GPUs of *sandbox*, which has reported, on the pinned
        baseline *image*, with the weights mount of the agent's *data*, its script from *checkout*,
        for *shape*. A role whose step did not start or report is recorded and the build goes
        on."""
        ctx = self.ctx
        [weights] = [m for m in data if m.target == WEIGHTS_TARGET]
        workdir = ctx.role_dir(SERVICE) / "wd"
        workdir.mkdir()
        mounts = (Mount(weights.source, WEIGHTS_TARGET), Mount(self.socket_dir, rs.SOCKET_DIR, read_only=False),
                  Mount(checkout / rs.SCRIPT_REL, rs.SCRIPT_DEST))
        spec = ContainerSpec(image=image, command=tuple(self.engine.argv(weights=WEIGHTS_TARGET)), workdir=workdir,
                             mounts=mounts, gpus=True)
        env = {**OFFLINE_ENV, "HLS_WEIGHTS_DIR": WEIGHTS_TARGET}
        self.handle = ctx.start_container(SERVICE, spec, shape, owner="harness", env=env, share_gpus_of=sandbox,
                                          share_memory_of=sandbox)
        report = ctx.report(SERVICE)
        if report is None or report.mismatches:
            self.failure = ("its step did not start or report its grant" if report is None
                            else f"its grant differs from its shape: {'; '.join(report.mismatches)}")
            self._unready("start", self.failure)
            return
        self.launched_at, self.launched_wall = time.monotonic(), time.time()
        watch = ctx.watch
        if watch is not None:
            ooms = watch.job_ooms()
            self.job_ooms_at_start = ooms.own if ooms is not None else None
            if watch.job_dir is not None:
                # The runner's memory is evidence when the job's limit is reached with the service in
                # it (``memory_evidence``); its batch step is observed, never killed.
                watch.observe(BATCH, watch.job_dir / "step_batch", role="runner")
        self.emit("build.reference_service_started", socket=rs.SOCKET_PATH, host_socket=str(self.host_socket))

    def memory_evidence(self, oom_at: float | None, sandbox_step: str | None) -> SharedMemoryEvidence:
        """What the service, the runner and the sandbox (whose step is *sandbox_step*) held, as the
        build rule reads it for a sandbox whose last OOM was at *oom_at*
        (``hls.substrate.faults.classify_shared_build_end``). Read while the service still runs,
        before the stage stops it; whatever cannot be read is None."""
        ctx = self.ctx
        watch = ctx.watch
        report = ctx.report(SERVICE) if self.handle is not None else None
        running = (report is not None and self.handle is not None and self.handle.poll() is None
                   and self.launched_wall is not None and oom_at is not None and self.launched_wall <= oom_at)
        step_dir = Path(report.cgroup_dir).parent if report is not None else None
        try:
            ooms = read_keyed(step_dir / "memory.events").get("oom") if step_dir is not None else None
        except (OSError, ValueError):
            ooms = None
        breach = watch is not None and report is not None and watch.breach(report.step) is not None
        job = watch.job_ooms() if watch is not None else None
        runner = None
        if watch is not None and watch.job_dir is not None and watch.last(BATCH) is not None:
            runner = _step_memory(watch.last(BATCH), watch.highest(BATCH), read_peak(watch.job_dir / "step_batch"))
        sandbox_report = ctx.report("sandbox")
        return SharedMemoryEvidence(
            service_running=running, service_limit_bytes=self.params.shape.memory_gib * 2**30, service_ooms=ooms,
            service_breach=breach,
            service=_step_memory(*(_samples(watch, report.step) if report is not None else (None, None)),
                                 read_peak(step_dir) if step_dir is not None else None),
            runner_reserve_bytes=ctx.plan.settings.run.runner_memory_gib * 2**30, runner=runner,
            sandbox=_step_memory(*(_samples(watch, sandbox_step) if sandbox_step is not None else (None, None)),
                                 read_peak(Path(sandbox_report.cgroup_dir).parent) if sandbox_report else None),
            job_ooms_at_start=self.job_ooms_at_start, job_ooms_at_end=job.own if job is not None else None)

    def tick(self) -> None:
        """On each of the stage's checks while the agent runs: stop a service that will never be
        ready, and report one that was ready and whose role ended on its own, once. Until it is
        ready the watch tells why it is not (``watch``)."""
        if self.handle is None or self.outcome is not None:
            return
        if self.failure is not None or (self.ready is not None and self.handle.poll() is not None):
            self.stop()

    def stop(self) -> None:
        """Stop the role as planned, or report its own end if it ended first, and record how it
        ended. Idempotent; a role never started has nothing to stop."""
        if self.handle is None or self.outcome is not None:
            return
        self.outcome = self.ctx.stop_role(SERVICE)
        self.emit("build.reference_service_stopped", ending=self.outcome.ending, exit_code=self.outcome.exit_code)

    # the watch

    async def watch(self, env: SandboxEnvironment) -> None:
        """Beside the running agent, for the whole of its part: read ``/status`` every
        ``POLL_SEC``; probe the service from inside the sandbox, over the socket the agent uses,
        the first time it is built; record each time it reports unloaded, memory having been short
        for a load (``build.reference_service_unloaded``); and end, through ``tick``, a service that
        reports failed, whose role ends before it was ever built, or that stays starting for
        ``READY_TIMEOUT_SEC`` at a stretch, counted from its role's report or from the first read
        that saw it start again. A read that cannot connect before it was ever built is still
        starting: the socket is there only once the service binds it. Never fatal; cancelled when
        the agent's part ends, and a service never built nor unloaded by then is recorded as not
        ready."""
        if self.handle is None or self.failure is not None or self.launched_at is None:
            return
        starting_since: float | None = self.launched_at
        last = "no reply yet"
        unloaded = seen_unloaded = False
        try:
            while True:
                status: dict[str, Any] | None = None
                try:
                    status = await asyncio.to_thread(rs.get_status, str(self.host_socket))
                except (OSError, ValueError) as e:
                    last = f"{type(e).__name__}: {e}"
                state = status.get("state") if status is not None else None
                if state is not None:
                    last = f"state {state!r}"
                if status is not None and state in rs.READY_STATES:
                    starting_since, unloaded = None, False
                    if self.ready is None:
                        self.ready = status
                        await self._probe(env, status)
                elif status is not None and state == rs.UNLOADED_STATE:
                    starting_since = None
                    if not unloaded:
                        unloaded = seen_unloaded = True
                        self.emit("build.reference_service_unloaded", memory_error=status.get("memory_error"))
                elif status is not None and state == rs.FAILED_STATE:
                    raise _NotReady(f"the reference service reported failed: {status.get('error')}")
                elif state == rs.STARTING_STATE or (state is None and self.ready is None):
                    unloaded = False
                    now = time.monotonic()
                    starting_since = now if starting_since is None else starting_since
                    if now - starting_since >= rs.READY_TIMEOUT_SEC:
                        raise _NotReady(f"the reference service was not ready within {rs.READY_TIMEOUT_SEC:.0f} s "
                                        f"(last: {last})")
                code = self.handle.poll()
                if code is not None:
                    if self.ready is None:
                        raise _NotReady(f"the reference service's role ended with status {code} before it was "
                                        f"ready (last: {last})")
                    return
                await asyncio.sleep(POLL_SEC)
        except asyncio.CancelledError:
            if self.ready is None and self.failure is None and not seen_unloaded:
                self._unready("start", "the agent's part ended before the service was ready")
            raise
        except _NotReady as e:
            self.failure = str(e)
            self._unready("start" if self.ready is None else "run", self.failure)

    async def _probe(self, env: SandboxEnvironment, status: dict[str, Any]) -> None:
        """Ask the service for its ``/status`` from inside the sandbox, and record the answer with
        the host's *status* and the startup time, or that the sandbox cannot reach it."""
        startup = round(time.monotonic() - (self.launched_at or time.monotonic()), 1)
        try:
            res = await asyncio.wait_for(
                env.exec(rs.probe_command(python=exec_protocol.SANDBOX_PYTHON), timeout_sec=PROBE_TIMEOUT_SEC),
                timeout=PROBE_TIMEOUT_SEC)
            if res.return_code != 0:
                raise RuntimeError(f"probe exited {res.return_code}: {(res.stderr or res.stdout or '').strip()[-500:]}")
            seen = rs.read_probe(res.stdout or "")
        except Exception as e:  # noqa: BLE001 -- the probe only records whether the sandbox reaches it
            self._unready("probe", f"{type(e).__name__}: {e}", startup_sec=startup)
            return
        self.emit("build.reference_service_probed", service_status=status, startup_sec=startup,
                  sandbox_state=str(seen.get("state"))[:64])

    def _unready(self, where: str, reason: str, *, startup_sec: float | None = None) -> None:
        """Record that the service the brief promises is not there for the agent, with why and
        the end of what it said, and log it."""
        said = self._tail() if where == "start" else None
        text = self.ctx.scrubber.scrub(reason + (f"; its output ends: {said}" if said else ""))[:_REASON_CHARS]
        if startup_sec is None and self.launched_at is not None:
            startup_sec = round(time.monotonic() - self.launched_at, 1)
        self.emit("build.reference_service_unreachable", where=where, reason=text, startup_sec=startup_sec)
        log.error("the reference service the brief promises is not reachable (%s): %s; the agent builds without it",
                  where, text[:300])

    def _tail(self) -> str | None:
        """The last lines of the service's stderr, joined; None when it wrote none."""
        path = self.ctx.attempt_dir / "logs" / f"{SERVICE}.err"
        try:
            with open(path, "rb") as f:
                f.seek(max(0, f.seek(0, os.SEEK_END) - _TAIL_BYTES))
                lines = f.read().decode(errors="replace").strip().splitlines()
        except OSError:
            return None
        return " | ".join(lines[-3:]) or None


def _samples(watch, step: str) -> tuple[MemorySample | None, MemorySample | None]:
    """The watch's last and highest samples of *step*; None for each it has not."""
    if watch is None:
        return None, None
    return watch.last(step), watch.highest(step)


def _step_memory(last: MemorySample | None, highest: MemorySample | None, peak: int | None) -> StepMemory:
    return StepMemory(last_bytes=last.footprint if last is not None else None,
                      highest_bytes=highest.footprint if highest is not None else None, peak_bytes=peak)
