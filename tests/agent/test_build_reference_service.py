"""The agent_build stage's in-build reference service (``AgentProducerSpec.reference_service``,
``hls.stages.reference_service``), end to end on one machine as ``tests.agent.test_stage`` runs a
build: the real service script runs in the fake container on a stand-in vLLM
(``tests.agent.fake_vllm``), beside the fake sandbox, on the sandbox's GPUs.

Off, nothing changes. On, the service is started once the sandbox has reported, on the GPUs the
sandbox holds, on the task's baseline image; the sandbox gets its socket's directory read-only and the
variable naming the socket, and nothing else of it; the agent runs while it starts; it is probed
from inside the sandbox once it is ready, and stopped right after the sandbox, before collection,
on every path. A service that cannot start, fails, ends, is never ready or cannot be reached is
recorded and the build is graded as any build is."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from hls.agent import environment
from hls.agent.environment import SandboxEnvironment
from hls.agent.workspace import SandboxLayout
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.checkout import CodeState
from hls.pipeline.plan import Toolchain
from hls.pipeline.planner import PlanError, build_plan
from hls.pipeline.stage import StageContext, StageFault
from hls.provision.model import Provisioned, WeightsRef, weights_key
from hls.recipes.render import recipe_key as def_key
from hls.reference import service_harness as rs
from hls.spec.resources import ResourceShape, concurrent
from hls.stages import reference_service as service_mod
from hls.stages.agent_build import AgentBuildStage
from hls.stages.images import KEPT_DISK_GIB
from hls.store.cache import ContentCache, copy_disk_bytes, entry_sha256
from hls.store.lock import Holder
from hls.store.posix import PosixStore
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.base import NodeSize
from hls.substrate.memory_watch import MemoryWatch
from hls.store.layout import InstanceLayout
from hls.substrate.secrets import Scrubber, SecretsFile
from tests.agent import fake_sandbox
from tests.agent import test_stage as ts
from tests.agent.test_stage import (DATA_GIB, IMAGE_KEY, IMAGE_NAME, RUNNER_GIB, SECRETS, SNAPSHOT,
                                    MappedEnvironment, _events, _manifest, _plan, _record, _script)
from tests.pipeline.fakes import LocalLauncher
from tests.store.entries import populate

HERE = Path(__file__).resolve().parent
TASK_MODULE = "tests.agent.build_task_service"
BASELINE_KEY, BASELINE_NAME = "baseline-0123456789abcdef", "baseline"
#: The sandbox of these builds: the test sandbox's, with a GPU for the service to share.
SANDBOX = ts.SANDBOX.model_copy(update={"gpus": 1, "gpu_kind": "H200"})
#: The service's role: the sandbox's GPU, one CPU, 2 GiB and a GiB of disk.
SERVICE_SHAPE = ResourceShape(gpus=1, gpu_kind="H200", cpus=1, memory_gib=2, disk_gib=1, time_sec=600)
SERVICE = {"image": BASELINE_NAME, "shape": SERVICE_SHAPE.model_dump()}
ROLES = concurrent([SANDBOX, SNAPSHOT])
#: The stage of a build without the service, and of one with it.
STAGE_OFF = ROLES.model_copy(update={"cpus": SANDBOX.cpus, "memory_gib": ROLES.memory_gib + RUNNER_GIB,
                                     "disk_gib": ROLES.disk_gib + KEPT_DISK_GIB + DATA_GIB,
                                     "time_sec": ts.STAGE_TIME})
#: The service shares the sandbox's memory: its stage asks for the same memory, and the disk of the
#: service's image's tree and workdir more.
STAGE_ON = STAGE_OFF.model_copy(update={"disk_gib": STAGE_OFF.disk_gib + KEPT_DISK_GIB + SERVICE_SHAPE.disk_gib})
VOCAB = 1000
GIB = 2**30
#: The build tests' fixtures: the fake bins, and every context closed once a test ends.
sandbox_bins = ts.sandbox_bins
closed_contexts = ts.closed_contexts


class ServiceMappedEnvironment(MappedEnvironment):
    """``MappedEnvironment`` with the service's socket directory rewritten too, as the sandbox's
    mount namespace binds it."""

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        MappedEnvironment.SENT.append((command, dict(env or {})))
        MappedEnvironment.TIMEOUTS.append((command, timeout_sec))
        layout = SandboxLayout(self.socket.parent.parent)
        paths = {m.target: str(m.source) for m in layout.mounts()} | {
            "/tmp": str(layout.workdir / "tmp"), rs.SOCKET_DIR: str(layout.root / service_mod.SOCKET_DIR_NAME)}
        return await SandboxEnvironment.exec(
            self, fake_sandbox.mapped(paths, command), cwd=fake_sandbox.mapped(paths, cwd) if cwd else None,
            env={k: fake_sandbox.mapped(paths, v) for k, v in env.items()} if env else None, timeout_sec=timeout_sec,
            user=user)


@pytest.fixture
def service_bins(sandbox_bins, tmp_path_factory, monkeypatch):
    """The fake sandbox's bins, and a ``python3`` in the fake containers' PATH that runs the
    service on the stand-in vLLM, by the plan file this returns with the CLIs' control directory.
    The service is read every 0.05 s, and probed from a sandbox that binds its socket."""
    base = tmp_path_factory.mktemp("svc")
    plan = base / "vllm.json"
    plan.write_text(json.dumps({"record": str(base / "engine.json")}))
    bin_dir = base / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "python3"
    wrapper.write_text(f"#!/bin/sh\nFAKE_VLLM_PLAN={plan} PYTHONPATH={HERE / 'fake_vllm'} "
                       f"exec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)
    import os

    monkeypatch.setenv("FAKE_SANDBOX_PATH", f"{bin_dir}:{os.environ['FAKE_SANDBOX_PATH']}")
    monkeypatch.setattr(service_mod, "POLL_SEC", 0.05)
    monkeypatch.setattr(environment, "SandboxEnvironment", ServiceMappedEnvironment)
    return sandbox_bins, plan


def _vllm(plan: Path, **knobs) -> None:
    plan.write_text(json.dumps({**json.loads(plan.read_text()), **knobs}))


def _params(*, service: bool = True, **changes) -> dict:
    base = {"task": {"module": TASK_MODULE}, "sandbox_shape": SANDBOX.model_dump(),
            **({"reference_service": SERVICE} if service else {})}
    return ts._params(**{**base, **changes})


def _ctx(tmp_path: Path, params: dict, *, shape: ResourceShape = STAGE_ON,
         launcher: LocalLauncher | None = None, watch: MemoryWatch | None = None,
         run: dict | None = None) -> StageContext:
    """A context for an attempt of *params* on *shape*, as ``test_stage._ctx`` makes one, with the
    baseline image in this instance's ``images.json`` and weights whose config names a vocabulary,
    under the run settings *run* changes."""
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="hls-"))
    ts._SCRATCHES.append(scratch)
    plan = _plan(tmp_path, params, shape=shape, run=run)
    store = PosixStore(plan.store_root)
    cache = ContentCache(store)
    holder = Holder.here("test")
    images = {}
    # Two images of different bytes: the agent cap, and the baseline image the service runs on.
    files = {IMAGE_NAME: fake_sandbox.image_bytes(), BASELINE_NAME: json.dumps({"image": BASELINE_NAME}).encode()}
    for name, key in ((IMAGE_NAME, IMAGE_KEY), (BASELINE_NAME, BASELINE_KEY)):
        entry = populate(cache, "images", key, lambda d, data=files[name]: (d / "image.sif").write_bytes(data),
                         holder=holder).entry
        sif = next(f.sha256 for f in entry.manifest.files if f.path == "image.sif")
        images[name] = {"key": key, "entry_sha256": entry_sha256(entry), "sif_sha256": sif}
    images_key = plan.layout.attempt_file("images", 1, "outputs/images.json")
    store.write_bytes(images_key, json.dumps(images).encode())
    key = weights_key("org/model", "c" * 40)
    weights = populate(cache, "weights", key,
                       lambda d: (d / "config.json").write_text(json.dumps({"vocab_size": VOCAB})), holder=holder).entry
    provisioned = Provisioned(weights=WeightsRef(repo="org/model", revision="c" * 40, key=key,
                                                 entry_sha256=entry_sha256(weights),
                                                 disk_bytes=copy_disk_bytes(weights)), rows=None, tools=[])
    prov_key = plan.layout.attempt_file("provision", 1, "outputs/provisioned.json")
    store.write_bytes(prov_key, provisioned.model_dump_json().encode())
    inputs = {plan.layout.manifest("provision"): _manifest(plan, store, "provision", {"provisioned": prov_key}),
              plan.layout.manifest("images"): _manifest(plan, store, "images", {"images": images_key})}
    stage = plan.stage("agent_build")
    granted = {var: SECRETS[name] for var, name in stage.grants.items()}
    events = EventLog(store.path(plan.layout.attempt_file("agent_build", 1, "events.jsonl")))
    ctx = StageContext(store=store, plan=plan, stage=stage, attempt=1,
                       launcher=(launcher or _launcher()).use_plan(plan),
                       watch=watch, events=events, inputs=inputs, deadline=time.time() + 300, secrets=granted,
                       scrubber=Scrubber(SECRETS), cache=cache, scratch=scratch, job_id="100",
                       runtime=ApptainerRuntime(), images=ApptainerImages(scratch))
    ts._CONTEXTS.append(ctx)
    return ctx


def _launcher(**knobs) -> LocalLauncher:
    """The local launcher on a node of one build of the stage's size, so the commands' ceiling is
    one this host's kernel can set (``tests.support.node``)."""
    return LocalLauncher(gpus=1, node=NodeSize(gpus=SANDBOX.gpus, cpus=SANDBOX.cpus), **knobs)


def _kinds(ctx: StageContext, prefix: str = "build.reference_service") -> list[str]:
    return [e["kind"] for e in _events(ctx) if e["kind"].startswith(prefix)]


def _service_events(ctx: StageContext) -> list[dict]:
    return [e for e in _events(ctx) if e["kind"].startswith("build.reference_service")]


def _role_event(ctx: StageContext, kind: str, role: str) -> dict:
    [event] = [e for e in _events(ctx) if e["kind"] == kind and e["role"] == role]
    return event


def _index(ctx: StageContext, pred) -> int:
    return next(i for i, e in enumerate(_events(ctx)) if pred(e))


def _end_when(ctx_box: list, control: Path, kinds: tuple[str, ...], *, give_up_sec: float = 60.0) -> threading.Thread:
    """A thread that moves the runner's clock past the hard limit once the attempt's events hold
    one of *kinds*, or after *give_up_sec* regardless, so an agent that hangs ends its part then."""
    def run() -> None:
        deadline = time.monotonic() + give_up_sec
        while time.monotonic() < deadline:
            if ctx_box:
                path = ctx_box[0].attempt_dir / "events.jsonl"
                if path.exists() and any(e["kind"] in kinds for e in read_events(path)):
                    break
            time.sleep(0.05)
        (control / "offset").write_text("1200")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _hanging_build(tmp_path: Path, control: Path, until: tuple[str, ...], **ctx_kw) -> tuple[StageContext, dict]:
    """Run a build whose agent commits and hangs until the events hold one of *until*; its part
    then ends at the hard limit."""
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "hang": True}])
    ctx = _ctx(tmp_path, _params(), **ctx_kw)
    _end_when([ctx], control, until)
    return ctx, AgentBuildStage().run(ctx)


def _sandbox_record(control_records: Path) -> dict:
    return fake_sandbox.records(control_records)["sandbox"]


def _records_dir() -> Path:
    import os

    return Path(os.environ["FAKE_SANDBOX_RECORDS"])


# ---------------------------------------------------------------------------
# Off: nothing changes
# ---------------------------------------------------------------------------


def test_off_starts_no_service_binds_nothing_and_records_nothing(sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "advance": 900}])
    ctx = ts._ctx(tmp_path, ts._params())
    outputs = AgentBuildStage().run(ctx)
    assert "source_bundle" in outputs
    assert not any(e.get("role") == service_mod.SERVICE for e in _events(ctx))
    assert _kinds(ctx) == []
    sandbox = _sandbox_record(_records_dir())
    assert not any(rs.SOCKET_DIR in b for b in sandbox["binds"])
    assert rs.SOCKET_ENV_VAR not in sandbox["env"]
    assert rs.SOCKET_ENV_VAR not in json.loads(ctx.store.read_bytes(outputs["agent_env"]))["container"]
    assert not (ctx.role_dir("sandbox") / service_mod.SOCKET_DIR_NAME).exists()


def test_a_producer_without_the_service_plans_none(tmp_path):
    """The params of a build without the service carry no field of it, so its plan is as it was."""
    ctx = ts._ctx(tmp_path, ts._params())
    assert "reference_service" not in ctx.stage.params


# ---------------------------------------------------------------------------
# On: placement, isolation, readiness, stop
# ---------------------------------------------------------------------------


def test_the_service_runs_beside_the_sandbox_on_its_gpus_and_is_stopped_before_collection(service_bins, tmp_path):
    control, plan = service_bins
    ctx, outputs = _hanging_build(tmp_path, control, ("build.reference_service_probed",
                                                      "build.reference_service_unreachable"))
    assert "source_bundle" in outputs and _record(ctx, outputs).ending == "hard_limit"
    # No OOM at the job's limit: the record holds no shared-memory evidence.
    assert _record(ctx, outputs).shared_memory is None
    assert _kinds(ctx) == ["build.reference_service_started", "build.reference_service_probed",
                           "build.reference_service_stopped"]
    started, probed, stopped = _service_events(ctx)
    assert started["socket"] == "/run/hls-reference/ref.sock"
    assert started["host_socket"] == str(ctx.role_dir("sandbox") / "reference" / "ref.sock")
    assert probed["service_status"]["state"] == "asleep" and probed["sandbox_state"] == "asleep"
    assert probed["service_status"]["idle_sleep_sec"] == 120
    assert stopped["ending"] == "planned_stop"
    # Its own step, a harness role sharing the sandbox's GPUs, started once the sandbox reported.
    sandbox_started = _role_event(ctx, "role_started", "sandbox")
    service_started = _role_event(ctx, "role_started", service_mod.SERVICE)
    assert service_started["owner"] == "harness" and service_started["gpus"] == sandbox_started["gpus"]
    assert service_started["step"] != sandbox_started["step"]
    assert _index(ctx, lambda e: e is not None and e.get("role") == "sandbox" and e["kind"] == "role_started") < \
        _index(ctx, lambda e: e.get("role") == service_mod.SERVICE and e["kind"] == "role_started")
    # Stopped right after the sandbox, before the commit point ran.
    events = _events(ctx)
    order = [(e["kind"], e.get("role")) for e in events if e["kind"] == "role_ended"]
    assert order.index(("role_ended", "sandbox")) < order.index(("role_ended", service_mod.SERVICE)) < \
        order.index(("role_ended", "commit"))
    # The engine the grader builds, with sleep mode and the report width the brief states.
    engine = json.loads((plan.parent / "engine.json").read_text())
    assert engine["tensor_parallel_size"] == 1 and engine["enable_prefix_caching"] is False
    assert engine["enable_sleep_mode"] is True and engine["max_logprobs"] == 21
    assert engine["gpu_memory_utilization"] == 0.9 and engine["dtype"] == "auto"
    assert ctx.running_roles() == []


def test_the_sandbox_gets_the_socket_and_nothing_else_of_the_service(service_bins, tmp_path):
    control, _ = service_bins
    ctx, outputs = _hanging_build(tmp_path, control, ("build.reference_service_probed",
                                                      "build.reference_service_unreachable"))
    records = fake_sandbox.records(_records_dir())
    sandbox, service = records["sandbox"], records[service_mod.SERVICE]
    socket_dir = str(ctx.role_dir("sandbox") / "reference")
    # The sandbox: the socket's directory, read-only at the path the brief names, and the variable.
    service_binds = [b for b in sandbox["binds"] if b.split(":")[1] == rs.SOCKET_DIR]
    assert service_binds == [f"{socket_dir}:{rs.SOCKET_DIR}:ro"]
    assert sandbox["env"][rs.SOCKET_ENV_VAR] == rs.SOCKET_PATH
    # Nothing else of the service: not its image, its script, its role's directory or its own mounts.
    # The runtime binds the host's resolver files into every container at their own paths.
    service_sources = {b.split(":")[0] for b in service["binds"] if not b.startswith("/etc/")}
    sandbox_sources = {b.split(":")[0] for b in sandbox["binds"] if not b.startswith("/etc/")}
    assert service["image"] != sandbox["image"]
    assert not any(s.startswith(service["image"]) for s in sandbox_sources)
    assert str(ts.REPO / rs.SCRIPT_REL) not in sandbox_sources
    assert not any(s.startswith(str(ctx.role_dir(service_mod.SERVICE))) for s in sandbox_sources)
    assert service_sources & sandbox_sources == {socket_dir} | ({b.split(":")[0] for b in sandbox["binds"]
                                                                 if b.split(":")[1] == "/data/weights"})
    assert not any(str(ctx.store.root) in s for s in sandbox_sources)
    named = [k for k, v in sandbox["env"].items()
             if rs.SOCKET_DIR in v or service["image"] in v or str(ts.REPO) in v or str(ctx.store.root) in v]
    assert named == [rs.SOCKET_ENV_VAR]
    # The directory holds the socket alone while the service runs, and the agent's record names it.
    assert json.loads(ctx.store.read_bytes(outputs["agent_env"]))["container"][rs.SOCKET_ENV_VAR] == "plain"
    # The service: the weights read-only, the socket's directory writable, its script read-only.
    assert sorted(b.split(":", 1)[1] for b in service["binds"] if not b.startswith("/etc/")) == sorted([
        "/data/weights:ro", f"{rs.SOCKET_DIR}", f"{rs.SCRIPT_DEST}:ro"])
    assert f"{ts.REPO / rs.SCRIPT_REL}:{rs.SCRIPT_DEST}:ro" in service["binds"]
    assert service["nv"] is True and sandbox["nv"] is True
    assert service["command"][:2] == ["python3", rs.SCRIPT_DEST]
    assert service["env"] == {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HLS_WEIGHTS_DIR": "/data/weights"}
    # Its step is no contestant's and is not sealed.
    assert _role_event(ctx, "role_started", service_mod.SERVICE)["owner"] == "harness"


def test_the_agent_runs_while_the_service_starts(service_bins, tmp_path):
    """The engine build takes minutes and the agent's budget is running: the agent's clock starts
    while the service still loads, and the probe follows its readiness, inside the agent's part."""
    control, plan = service_bins
    _vllm(plan, load_sec=3)
    ctx, outputs = _hanging_build(tmp_path, control, ("build.reference_service_probed",
                                                      "build.reference_service_unreachable"))
    started = _index(ctx, lambda e: e["kind"] == "build.started")
    probed = _index(ctx, lambda e: e["kind"] == "build.reference_service_probed")
    assert _index(ctx, lambda e: e["kind"] == "build.reference_service_started") < started < probed
    assert _service_events(ctx)[1]["startup_sec"] >= 3
    assert "source_bundle" in outputs


def test_an_agent_part_that_ends_first_records_the_service_as_not_ready(service_bins, tmp_path):
    control, plan = service_bins
    _vllm(plan, load="hang")
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    assert "source_bundle" in outputs
    assert _kinds(ctx) == ["build.reference_service_started", "build.reference_service_unreachable",
                           "build.reference_service_stopped"]
    unready = _service_events(ctx)[1]
    assert unready["where"] == "start" and "ended before the service was ready" in unready["reason"]
    assert _service_events(ctx)[2]["ending"] == "planned_stop"
    assert ctx.running_roles() == []


@pytest.mark.parametrize("load, needle", [
    ("fail", "reported failed: engine load failed: RuntimeError('fake vllm: no such architecture in the registry')"),
    ("exit", "role ended with status"),
])
def test_a_service_that_fails_is_recorded_stopped_at_once_and_the_build_is_graded(service_bins, tmp_path, load,
                                                                                    needle):
    control, plan = service_bins
    _vllm(plan, load=load)
    ctx, outputs = _hanging_build(tmp_path, control, ("build.reference_service_stopped",))
    assert "source_bundle" in outputs and _record(ctx, outputs).fault is None
    assert _kinds(ctx) == ["build.reference_service_started", "build.reference_service_unreachable",
                           "build.reference_service_stopped"]
    unready = _service_events(ctx)[1]
    assert unready["where"] == "start" and needle in unready["reason"]
    # Stopped while the agent still ran, before the agent's part ended.
    stopped = _index(ctx, lambda e: e["kind"] == "build.reference_service_stopped")
    assert stopped < _index(ctx, lambda e: e["kind"] == "role_ended" and e["role"] == "sandbox")
    assert _role_event(ctx, "role_ended", "sandbox")["ending"] == "planned_stop"


@pytest.mark.parametrize("knobs", [{"gpu_free_gib": 50}, {"load": "oom"}], ids=["cards-short", "engine-oom"])
def test_a_service_short_of_memory_for_its_load_stays_up_unloaded_and_is_recorded(service_bins, tmp_path, knobs):
    """Memory short for the load is an answer to the agent, not a failure: the service holds no
    engine, answers 507 saying which memory and by how much, and waits for a wake to load it
    again. The stage records it and lets it run; it is stopped with the sandbox. The cards short of
    vLLM's fraction are found before the engine is built; the engine's own out-of-memory error is
    caught where it raises it."""
    control, plan = service_bins
    _vllm(plan, **knobs)
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "hang": True}])
    ctx = _ctx(tmp_path, _params())
    _end_when([ctx], control, ("build.reference_service_unloaded",))
    outputs = AgentBuildStage().run(ctx)
    assert "source_bundle" in outputs and _record(ctx, outputs).fault is None
    assert _kinds(ctx) == ["build.reference_service_started", "build.reference_service_unloaded",
                           "build.reference_service_stopped"]
    unloaded, stopped = _service_events(ctx)[1:]
    short = unloaded["memory_error"]
    assert (short["memory"], short["operation"]) == ("gpu", "load")
    if "gpu_free_gib" in knobs:
        assert short["gpu_free_bytes"] == [50 * 2**30] and short["gpu_need_bytes"] == [126 * 2**30]
    else:
        assert "Tried to allocate 20.00 GiB" in short["error"]
    assert stopped["ending"] == "planned_stop"
    assert _index(ctx, lambda e: e["kind"] == "build.reference_service_stopped") > \
        _index(ctx, lambda e: e["kind"] == "role_ended" and e["role"] == "sandbox")


def test_a_service_that_is_never_ready_is_ended_and_the_build_goes_on(service_bins, tmp_path, monkeypatch):
    """A half-built engine holds most of every card the agent works on, so a service that is not
    ready in time is stopped, not left starting."""
    control, plan = service_bins
    _vllm(plan, load="hang")
    monkeypatch.setattr(rs, "READY_TIMEOUT_SEC", 1.0)
    ctx, outputs = _hanging_build(tmp_path, control, ("build.reference_service_stopped",))
    assert "source_bundle" in outputs
    unready = _service_events(ctx)[1]
    assert unready["where"] == "start" and "not ready within 1 s" in unready["reason"]
    assert "'starting'" in unready["reason"]
    assert _service_events(ctx)[2] == {**_service_events(ctx)[2], "ending": "planned_stop"}
    assert _index(ctx, lambda e: e["kind"] == "build.reference_service_stopped") < \
        _index(ctx, lambda e: e["kind"] == "role_ended" and e["role"] == "sandbox")


def test_a_service_the_sandbox_cannot_reach_is_recorded_and_the_agent_still_runs(service_bins, tmp_path,
                                                                                monkeypatch):
    control, _ = service_bins
    # A sandbox whose mount namespace lacks the socket's directory.
    monkeypatch.setattr(environment, "SandboxEnvironment", MappedEnvironment)
    ctx, outputs = _hanging_build(tmp_path, control, ("build.reference_service_unreachable",
                                                      "build.reference_service_probed"))
    assert "source_bundle" in outputs
    assert _kinds(ctx) == ["build.reference_service_started", "build.reference_service_unreachable",
                           "build.reference_service_stopped"]
    unready = _service_events(ctx)[1]
    assert unready["where"] == "probe" and "FileNotFoundError" in unready["reason"]


def test_a_probe_that_hangs_is_cut_at_its_bound_and_the_build_goes_on(service_bins, tmp_path, monkeypatch):
    control, _ = service_bins
    monkeypatch.setattr(rs, "probe_command", lambda **kw: "sleep 60")
    monkeypatch.setattr(service_mod, "PROBE_TIMEOUT_SEC", 1)
    t0 = time.monotonic()
    ctx, outputs = _hanging_build(tmp_path, control, ("build.reference_service_unreachable",
                                                      "build.reference_service_probed"))
    assert "source_bundle" in outputs and time.monotonic() - t0 < 50
    unready = _service_events(ctx)[1]
    assert unready["where"] == "probe"


def test_a_sandbox_that_does_not_start_cleanly_starts_no_service(service_bins, tmp_path):
    control, _ = service_bins
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params(), launcher=_launcher(mismatch={"sandbox"}))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert e.value.fault.fault_class == "infra"
    assert not any(ev.get("role") == service_mod.SERVICE for ev in _events(ctx))
    assert _kinds(ctx) == ["build.reference_service_unreachable"]


def test_a_service_whose_step_does_not_start_cleanly_is_recorded_and_the_build_goes_on(service_bins, tmp_path):
    control, _ = service_bins
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params(), launcher=_launcher(mismatch={service_mod.SERVICE}))
    outputs = AgentBuildStage().run(ctx)
    assert "source_bundle" in outputs
    assert _kinds(ctx) == ["build.reference_service_unreachable", "build.reference_service_stopped"]
    unready = _service_events(ctx)[0]
    assert unready["where"] == "start" and "grant differs" in unready["reason"]
    # The sandbox still has the path the brief names.
    assert any(b.endswith(f":{rs.SOCKET_DIR}:ro") for b in _sandbox_record(_records_dir())["binds"])


def test_a_service_that_dies_after_it_was_ready_is_recorded_when_seen(service_bins, tmp_path):
    """Nothing in the build depends on the service: its end is recorded as soon as the stage sees
    it, and the agent's part goes on."""
    control, _ = service_bins
    ctx_box: list[StageContext] = []
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "hang": True}])
    ctx = _ctx(tmp_path, _params())
    ctx_box.append(ctx)

    def kill_after_probe() -> None:
        deadline = time.monotonic() + 60
        path = ctx.attempt_dir / "events.jsonl"
        while time.monotonic() < deadline:
            if path.exists() and any(e["kind"] == "build.reference_service_probed" for e in read_events(path)):
                ctx._roles[service_mod.SERVICE].kill()
                break
            time.sleep(0.05)

    threading.Thread(target=kill_after_probe, daemon=True).start()
    _end_when(ctx_box, control, ("build.reference_service_stopped",))
    outputs = AgentBuildStage().run(ctx)
    assert "source_bundle" in outputs and _record(ctx, outputs).ending == "hard_limit"
    assert _kinds(ctx) == ["build.reference_service_started", "build.reference_service_probed",
                           "build.reference_service_stopped"]
    assert _service_events(ctx)[2]["ending"] != "planned_stop"
    assert _index(ctx, lambda e: e["kind"] == "build.reference_service_stopped") < \
        _index(ctx, lambda e: e["kind"] == "role_ended" and e["role"] == "sandbox")


@ts.NOT_ROOT
def test_a_preflight_only_build_stops_the_service_with_the_sandbox(service_bins, tmp_path):
    control, _ = service_bins
    _script(control, window=1_000_000, preflight={}, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="preflight"))
    outputs = AgentBuildStage().run(ctx)
    assert "probe" in outputs and _record(ctx, outputs).ending == "preflight"
    assert _kinds(ctx)[0] == "build.reference_service_started"
    assert _kinds(ctx)[-1] == "build.reference_service_stopped"
    assert ctx.running_roles() == []


# ---------------------------------------------------------------------------
# Refusals, before any role
# ---------------------------------------------------------------------------


def test_a_producer_that_asks_for_the_service_with_a_plan_that_launches_none_is_refused(sandbox_bins, tmp_path):
    ctx = _ctx(tmp_path, _params(service=False), shape=STAGE_OFF)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert e.value.fault.fault_class == "infra" and not e.value.fault.retryable
    assert "asks for reference service" in e.value.fault.reason and "launches none" in e.value.fault.reason
    assert not any(ev["kind"] == "role_started" for ev in _events(ctx))


def test_a_plan_that_launches_the_service_for_a_producer_without_it_is_refused(sandbox_bins, tmp_path,
                                                                             monkeypatch):
    from tests.agent import build_task, build_task_service

    monkeypatch.setattr(build_task_service, "ARTIFACT", build_task.ARTIFACT)
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert e.value.fault.fault_class == "infra" and not e.value.fault.retryable
    assert "runs no reference service" in e.value.fault.reason and "launches one" in e.value.fault.reason
    assert not any(ev["kind"] == "role_started" for ev in _events(ctx))


def test_a_task_without_exactly_one_vllm_reference_is_refused_at_plan_time(tmp_path):
    """The service serves the one engine that grades the task; the build task grades against an
    HF reference."""
    with pytest.raises(PlanError, match="VLLMReference"):
        _plan(tmp_path, _params(task={"module": "tests.agent.build_task"}), shape=STAGE_ON)


def test_an_engine_that_does_not_span_the_shared_gpus_is_refused_at_plan_time(tmp_path, monkeypatch):
    from tests.agent import build_task_service

    [verifier] = build_task_service.TASK.verifiers
    wide = verifier.model_copy(update={"reference": verifier.reference.model_copy(update={"tensor_parallel_size": 2})})
    monkeypatch.setattr(build_task_service, "TASK", build_task_service.TASK.model_copy(update={"verifiers": [wide]}))
    with pytest.raises(PlanError, match="engine spans 2 GPUs, and it shares the sandbox's 1"):
        _plan(tmp_path, _params(), shape=STAGE_ON)


@pytest.mark.parametrize("change, needle", [
    ({"gpus": 2}, "shares every GPU of the sandbox"),
    ({"gpu_kind": "B200"}, "shares every GPU of the sandbox"),
    ({"cpus": 3}, "run on the sandbox's 2"),
])
def test_a_service_shape_that_does_not_fit_the_sandbox_is_refused_at_plan_time(tmp_path, change, needle):
    params = _params(reference_service={"image": BASELINE_NAME,
                                        "shape": SERVICE_SHAPE.model_copy(update=change).model_dump()})
    with pytest.raises(PlanError, match=needle):
        _plan(tmp_path, params, shape=STAGE_ON)


def test_the_stage_holds_the_services_disk_and_none_of_its_memory(tmp_path):
    """The service's memory is oversubscribed on the sandbox's: a stage that asks for it on top is
    refused as one that lacks the service's disk is."""
    with pytest.raises(PlanError, match="reference service's image's tree"):
        _plan(tmp_path, _params(), shape=STAGE_OFF)
    with pytest.raises(PlanError, match="reference service's image's tree"):
        _plan(tmp_path, _params(), shape=STAGE_ON.model_copy(
            update={"memory_gib": STAGE_ON.memory_gib + SERVICE_SHAPE.memory_gib}))
    _plan(tmp_path, _params(), shape=STAGE_ON)


def test_a_service_whose_memory_alone_does_not_fit_the_stage_is_refused_at_plan_time(tmp_path):
    """Its step's limit is its own memory, which the job's memory must hold by itself beside the
    runner."""
    params = _params(reference_service={"image": BASELINE_NAME, "shape": SERVICE_SHAPE.model_copy(
        update={"memory_gib": STAGE_ON.memory_gib - RUNNER_GIB + 1}).model_dump()})
    with pytest.raises(PlanError, match="the reference service's .* GiB and the runner's 4 GiB"):
        _plan(tmp_path, params, shape=STAGE_ON)


def test_the_service_is_one_more_extraction_and_one_more_role():
    from hls.stages.agent_build import AgentBuildParams

    on = AgentBuildParams.model_validate(_params())
    off = AgentBuildParams.model_validate(_params(service=False))
    assert AgentBuildStage.max_roles(on) == AgentBuildStage.max_roles(off) + 2


# ---------------------------------------------------------------------------
# Whose end it is when the job's memory runs out beside the service
# ---------------------------------------------------------------------------


def _memory_files(directory: Path, *, gib: float, oom: int = 0, oom_kill: int = 0) -> None:
    """Fake memory accounting for a cgroup: a footprint of *gib* GiB, ``memory.peak`` a GiB above
    it, and the ``oom`` and ``oom_kill`` counts given."""
    directory.mkdir(parents=True, exist_ok=True)
    size = int(gib * GIB)
    (directory / "memory.current").write_text(f"{size}\n")
    (directory / "memory.swap.current").write_text("0\n")
    (directory / "memory.stat").write_text(f"anon {size}\nfile 0\nshmem 0\n")
    (directory / "memory.events").write_text(f"oom {oom}\noom_kill {oom_kill}\n")
    (directory / "memory.peak").write_text(f"{size + GIB}\n")


def _job_files(job: Path, *, own: int) -> None:
    job.mkdir(parents=True, exist_ok=True)
    (job / "memory.events.local").write_text(f"oom {own}\noom_kill 0\n")
    (job / "memory.events").write_text(f"oom {own}\noom_kill {own}\n")


def _kill_exec_server(ctx: StageContext) -> None:
    """SIGKILL the sandbox's exec server, as the kernel's OOM killer would end it: the sandbox ends."""
    import os
    import signal

    server = str(ctx.role_dir("sandbox") / "exec")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        args = cmd.split(b"\0")
        # The server itself, not the seal and the runtime wrapping it, whose arguments name it too.
        if len(args) > 2 and args[1:3] == [b"-I", f"{server}/exec_server.py".encode()]:
            os.kill(int(proc.name), signal.SIGKILL)


@pytest.fixture
def job_watch(tmp_path, monkeypatch):
    """The runner's memory watch over a fake job cgroup, the kernel enforcing (no kill by the
    runner), the sandbox's split sampler planted with the harness at 1 GiB of its 4 GiB reserve.
    Yields the watch and the job's cgroup."""
    from tests.agent.test_stage import _PlantedSampler

    monkeypatch.setattr(ts.stage_module, "SplitSampler", _PlantedSampler)
    job = tmp_path / "cg" / "job_100"
    _job_files(job, own=0)
    watch = MemoryWatch(enforce=False, interval_sec=0.05, job_dir=job)
    watch.start()
    yield watch, job
    watch.stop()


def _job_oom_build(tmp_path: Path, control: Path, watch: MemoryWatch, job: Path, *, after: str,
                   service_ooms: int = 0, runner_gib: float | None = 1.0,
                   run: dict | None = None) -> StageContext:
    """A build whose job's memory runs out once the events hold *after*: the job's own ``oom``
    count rises, the kernel kills in the sandbox at the job's limit (its own count unchanged) and
    its exec server dies. The service's step shows *service_ooms* of its own; the runner's batch
    step holds *runner_gib* GiB, or cannot be read with None. *run* changes the run settings."""
    if runner_gib is not None:
        _memory_files(job / "step_batch", gib=runner_gib)
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "hang": True}])
    ctx = _ctx(tmp_path, _params(), watch=watch, launcher=_launcher(steps_under=job), run=run)

    def exhaust() -> None:
        deadline = time.monotonic() + 60
        path = ctx.attempt_dir / "events.jsonl"
        while time.monotonic() < deadline:
            if path.exists() and any(e["kind"] == after for e in read_events(path)):
                break
            time.sleep(0.05)
        service = ctx.report(service_mod.SERVICE)
        if service is not None:
            _memory_files(Path(service.cgroup_dir), gib=1.5)
            _memory_files(Path(service.cgroup_dir).parent, gib=1.5, oom=service_ooms)
        sandbox = ctx.report("sandbox")
        _memory_files(Path(sandbox.cgroup_dir), gib=12, oom_kill=1)
        _memory_files(Path(sandbox.cgroup_dir).parent, gib=12, oom_kill=1)
        _job_files(job, own=1)
        time.sleep(0.3)  # the watch samples the steps as they now stand
        _kill_exec_server(ctx)

    threading.Thread(target=exhaust, daemon=True).start()
    return ctx


def test_the_jobs_memory_run_out_while_the_service_and_runner_are_within_theirs_is_the_sessions(
        service_bins, tmp_path, job_watch):
    """Case 1: the service within its own limit and the runner within its reserve, the job's limit
    reached is the session's overuse by arithmetic. The build ends as a sandbox out of its memory
    does, graded on its last commit, and the record holds what the rule read."""
    control, _ = service_bins
    watch, job = job_watch
    ctx = _job_oom_build(tmp_path, control, watch, job, after="build.reference_service_probed")
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert record.ending == "contestant_oom" and record.fault.fault_class == "contestant"
    assert "shares with the in-build reference service" in record.fault.reason
    assert "source_bundle" in outputs and record.grade_after_sandbox_oom is True
    shared = record.shared_memory
    assert shared is not None and shared.service_running and shared.service_ooms == 0
    assert not shared.service_breach and shared.service_limit_bytes == 2 * GIB
    assert shared.runner is not None and shared.runner.highest_bytes == GIB and shared.runner.peak_bytes == 2 * GIB
    assert shared.runner_reserve_bytes == RUNNER_GIB * GIB
    assert shared.sandbox.peak_bytes == 13 * GIB and shared.service.highest_bytes == int(1.5 * GIB)
    assert (shared.job_ooms_at_start, shared.job_ooms_at_end) == (0, 1)
    assert record.fault.evidence["sandbox.shared.service_running"] == "True"


def test_the_sessions_end_on_the_jobs_memory_scores_zero_under_a_plan_that_does_not_grade_it(
        service_bins, tmp_path, job_watch):
    """Case 1 under ``run.grade_after_sandbox_oom`` false: the session's end follows the plan's
    rule as an end at the sandbox's own limit does, the contestant's zero with nothing graded."""
    control, _ = service_bins
    watch, job = job_watch
    ctx = _job_oom_build(tmp_path, control, watch, job, after="build.reference_service_probed",
                         run={"grade_after_sandbox_oom": False})
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False)
    assert ctx.contestant_fault() == fault
    assert fault.evidence["grade_after_sandbox_oom"] == "false"
    assert "shares with the in-build reference service" in fault.reason and "nothing graded" in fault.reason
    assert not [s.name for s in ctx.launcher.launched if s.name.startswith("commit")]


def test_the_jobs_memory_run_out_while_the_service_is_over_its_own_limit_is_the_harnesss(
        service_bins, tmp_path, job_watch):
    """Case 2: the service reached its own limit, so the excess may be its own: infra, retried."""
    control, _ = service_bins
    watch, job = job_watch
    ctx = _job_oom_build(tmp_path, control, watch, job, after="build.reference_service_probed", service_ooms=1)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "the reference service reached its own limit" in fault.reason
    assert fault.evidence["sandbox.shared.service_ooms"] == "1"


def test_the_jobs_memory_run_out_with_no_service_running_is_the_harnesss_as_ever(
        service_bins, tmp_path, job_watch):
    """Case 3: the service had ended before the OOM; the build rule's verdict on a kernel kill at
    the job's limit, infra, stands."""
    control, plan = service_bins
    _vllm(plan, load="fail")
    watch, job = job_watch
    ctx = _job_oom_build(tmp_path, control, watch, job, after="build.reference_service_stopped")
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "the job's own memory limit was reached" in fault.reason
    assert fault.evidence["sandbox.shared.service_running"] == "False"


@pytest.mark.parametrize("runner_gib, needle", [(None, "the runner's memory could not be read"),
                                                (5.0, "against its 4294967296 byte reserve")],
                         ids=["unreadable", "over-reserve"])
def test_a_runner_that_cannot_be_read_or_is_over_its_reserve_leaves_the_end_the_harnesss(
        service_bins, tmp_path, job_watch, runner_gib, needle):
    """A contestant is never charged on an assumption: without the runner's footprint, or with it
    over its reserve, the job's limit reached is infra."""
    control, _ = service_bins
    watch, job = job_watch
    ctx = _job_oom_build(tmp_path, control, watch, job, after="build.reference_service_probed",
                         runner_gib=runner_gib)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert needle in e.value.fault.reason


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def _agentic_build(reference_service: bool, *, allocation_gib: int | None = None,
                   catalog_key: str = "agentic_coding") -> dict:
    """The images and agent_build stages the planner makes for agentic_coding's build, with its
    producer's flag as given, for a checkpoint of 305 GiB, in its build allocation or one of
    *allocation_gib*."""
    from hls.artifact.artifact import BuildAllocation
    from hls.pipeline.pipelines import build_stages
    from hls.settings import RunSettings
    from hls.taskref import TaskRef
    from tasks.agentic_coding import task as agentic

    producer = agentic.ARTIFACT.producer.model_copy(update={"reference_service": reference_service})
    if allocation_gib is not None:
        allocation = BuildAllocation(cpus=producer.build_allocation.cpus, memory_gib=allocation_gib)
        producer = producer.model_copy(update={"build_allocation": allocation})
    key = weights_key("org/agentic", "d" * 40)
    provisioned = Provisioned(weights=WeightsRef(repo="org/agentic", revision="d" * 40, key=key, entry_sha256="e" * 64,
                                                 disk_bytes=305 * 2**30), rows=None, tools=[])
    decls = build_stages(task_ref=TaskRef(module="tasks.agentic_coding.task"), task=agentic.TASK, producer=producer,
                         arm=ts.OPUS, arm_change=None, catalog_key=catalog_key, runner_memory_gib=RUNNER_GIB,
                         seal_memory_gib=0, collect_allowance_sec=RunSettings().collect_allowance_sec,
                         held=lambda keys: set(keys), recipe_key=def_key, provisioned=provisioned)
    return {d.name: d for d in decls}


def test_the_planner_gives_agentic_codings_build_the_baseline_image_and_the_services_role():
    """The service runs on the task's measured baseline image, and its memory is oversubscribed on
    the sandbox's: the sandbox and the stage keep the memory of a build without it."""
    from hls.recipes import catalog
    from hls.stages.agent_build import AgentBuildParams
    from hls.stages.engine_cache import ENGINE_CACHE_GIB
    from tasks.agentic_coding import task as agentic

    baseline = catalog.task_images("agentic_coding").measured["baseline"]
    on, off = _agentic_build(True), _agentic_build(False)
    assert baseline in [i["name"] for i in on["images"].params["images"]]
    assert baseline not in [i["name"] for i in off["images"].params["images"]]
    build_on, build_off = on["agent_build"], off["agent_build"]
    assert "reference_service" not in build_off.params
    assert build_on.params["sandbox_shape"] == build_off.params["sandbox_shape"]
    sandbox = ResourceShape.model_validate(build_on.params["sandbox_shape"])
    assert sandbox.memory_gib == agentic.BUILD_ALLOCATION.memory_gib - RUNNER_GIB - 7
    assert build_on.params["reference_service"] == {
        "image": baseline,
        "shape": ResourceShape(gpus=8, gpu_kind="H200", cpus=8, memory_gib=397, disk_gib=ENGINE_CACHE_GIB,
                               time_sec=sandbox.time_sec).model_dump()}
    # The stage's memory is the build allocation's either way; the service's image's tree and its
    # workdir take disk of their own; its GPUs, CPUs and memory are the sandbox's.
    assert build_on.shape.memory_gib == build_off.shape.memory_gib == agentic.BUILD_ALLOCATION.memory_gib
    assert build_on.shape == build_off.shape.model_copy(update={
        "disk_gib": build_off.shape.disk_gib + KEPT_DISK_GIB + ENGINE_CACHE_GIB})
    # The stage's own check accepts the plan: the grader's engine spans the 8 GPUs it shares.
    AgentBuildStage.check(build_on.shape, AgentBuildParams.model_validate(build_on.params),
                          runner_memory_gib=RUNNER_GIB)


def test_no_other_stage_and_no_sbatch_argument_moves_with_the_service(tmp_path):
    """Planned whole, a build with the service differs from one without only in the agent_build
    stage's disk and parameters and the images stage's list: every other stage's shape is the same,
    and every allocation asks for the same memory, CPUs and GPUs. Its time grows by the launch
    allowance (``site.step_launch_allowance_sec``) of the roles the service adds: the images stage's
    four for the baseline image and four for the vendor image it builds on, and the build's two,
    the image's extraction and the service; every other sbatch argument is the same, and none asks
    for disk."""
    from hls.pipeline.pipelines import KNOWN_STAGE_NAMES
    from hls.settings import Settings
    from hls.substrate.slurm.resources import sbatch_args
    from tests.provision.warmer_config import task_sandbox_stores

    # The site names the sandbox stores the build's provision stage makes what the task needs in.
    stores = {name: str(tmp_path / "sandbox-stores" / name) for name in task_sandbox_stores("tasks.agentic_coding.task")}
    table = ts._settings_table(tmp_path, site={"secrets_path": str(tmp_path / "secrets" / "secrets.toml")})
    settings = Settings.model_validate({**table, "sandbox_stores": stores})
    plans = {}
    for on in (True, False):
        decls = list(_agentic_build(on).values())
        plan = build_plan(decls=decls, task="agentic_coding", pipeline="build",
                          layout=InstanceLayout.rollout("agentic_coding", "r1"), settings=settings,
                          code=CodeState(checkout=Path("/c/hls"), hls_commit="a" * 40, harbor_commit="b" * 40,
                                         dirty_diff_sha256=None),
                          toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                          job_name="hls-build", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=("agentic_coding",),
                          now=1000.0, secrets=SecretsFile(secrets=SECRETS))
        plans[on] = plan
    on, off = plans[True], plans[False]
    assert [s.name for s in on.stages] == [s.name for s in off.stages]
    for a, b in zip(on.stages, off.stages):
        if a.name == "agent_build":
            assert a.shape == b.shape.model_copy(update={"disk_gib": a.shape.disk_gib}), a.name
        else:
            assert a.shape == b.shape, a.name
            if a.name != "images":
                assert a.params == b.params, a.name
    assert len(on.allocations) == len(off.allocations)
    site = ts._settings(tmp_path).site
    for a, b in zip(on.allocations, off.allocations):
        # Four roles of the images stage for each of the baseline image and the vendor image under
        # it, which no other image of a build needs; two of the build's for the service.
        launches = (2 * 4 + 2) * site.step_launch_allowance_sec if "agent_build" in a.stages else 0
        assert a.request.time_sec == b.request.time_sec + launches
        same = {"disk_gib": 0, "time_sec": 0}
        assert a.request.model_copy(update=same) == b.request.model_copy(update=same)

        def args(alloc) -> list[str]:
            return [x for x in sbatch_args(alloc.request, site, job_name="j", output=Path("/o"),
                                           partition=alloc.placement.partition, qos=alloc.placement.qos,
                                           socket_tasks=alloc.placement.socket_tasks)
                    if not x.startswith("--time=")]

        assert args(a) == args(b)
        assert a.placement.qos == b.placement.qos


def test_a_run_that_grades_its_build_still_pins_the_services_image():
    """``hls run`` replaces the build's images stage with one for both halves; the service's
    image stays in it, since the build pins it from this instance's images."""
    from hls.pipeline.pipelines import rollout_stages
    from hls.recipes import catalog
    from hls.settings import RunSettings
    from hls.taskref import TaskRef
    from tasks.agentic_coding import task as agentic

    key = weights_key("org/agentic", "d" * 40)
    provisioned = Provisioned(weights=WeightsRef(repo="org/agentic", revision="d" * 40, key=key, entry_sha256="e" * 64,
                                                 disk_bytes=305 * 2**30), rows=None, tools=[])
    producer = agentic.ARTIFACT.producer
    decls = rollout_stages(task_ref=TaskRef(module="tasks.agentic_coding.task"), task=agentic.TASK, producer=producer,
                           serve_command=[], arm=ts.OPUS, arm_change=None, catalog_key="agentic_coding",
                           run=RunSettings(), seal_memory_gib=0, allocation=producer.evaluation_allocation,
                           held=lambda keys: set(keys), recipe_key=def_key, provisioned=provisioned)
    stages = {d.name: d for d in decls}
    pinned = [i["name"] for i in stages["images"].params["images"]]
    service = stages["agent_build"].params["reference_service"]["image"]
    assert service == catalog.task_images("agentic_coding").measured["baseline"] and service in pinned


def test_a_task_without_a_measured_baseline_image_cannot_run_the_service():
    with pytest.raises(ValueError, match="runs on the task's measured baseline image, and its module declares none"):
        _agentic_build(True, catalog_key="sample_task")
