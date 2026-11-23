"""L1 -- SandboxToolRunner: the harness-side shell for a coding agent.

``spec.tools`` ships the in-process runner, whose tools are Python callables cheap
enough to run in-band. A coding agent's tool is not that: it is a shell, it edits a
repository, and two trajectories sharing one checkout would write over each other. So
this runner gives every trajectory its own container, executes one command per turn in
it, and tears it down when the trajectory ends.

The trust boundary is the reason it lives on the harness's side (design Sec.9). The
artifact serves one reply per HTTP call and never runs a command; the framework parses
the command out of the reply and dispatches it here. The wall time of every command is
therefore real and counts toward the trajectory's makespan, which a contestant cannot
shorten by claiming a test suite ran instantly.

Backend-agnostic by construction. The runner holds a factory that turns a
``LaunchSpec`` into a harbor ``Environment``, and harbor's environments all
expose the same ``exec`` primitive, so the same runner drives Docker on a workstation
and Apptainer on a Slurm cluster. The factory is injected rather than imported, which
is also what lets the tests drive it without a container runtime.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Union

from hls.spec.tools import DEVICE, HostPath, StoreEntry, ToolRunner, ToolSpec

#: The tool a coding agent is given. One command per call, run in the session's
#: checkout. Named for what it is, because the name reaches the model in the prompt.
BASH_TOOL = ToolSpec(
    name="bash",
    description=(
        "Run one bash command in the repository checkout and return its combined "
        "output and exit code. Each call runs in a fresh subshell, so directory and "
        "environment changes do not persist between calls."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run."},
            "timeout_sec": {"type": "integer",
                            "description": "Seconds before the command is killed."},
        },
        "required": ["command"],
    },
)

#: What a command that exceeded its wall-clock budget reports. A shell convention
#: (128 + SIGKILL), so a model that knows shells reads it without being told.
TIMEOUT_EXIT_CODE = 137

#: Seconds between checks that a starting container's exec server answers.
SERVER_READY_POLL_SEC = 0.05

#: What a command reports when the sandbox itself failed rather than the command. The
#: distinction matters in triage: a run of these is an infrastructure fault, while a
#: run of ordinary non-zero exits is the model failing to make progress.
SANDBOX_ERROR_EXIT_CODE = 125


#: A launch setting that may differ from one trajectory to the next. A plain string is
#: the same for every session; a callable is handed the session's context and returns
#: that session's value. Used for the image and the working directory, because a
#: benchmark can ship one environment per task rather than one for the whole corpus.
SessionValue = Union[str, Callable[[dict[str, Any]], str]]

#: The same, for a setting that is a flag rather than a name. Separate because a
#: corpus can mix tasks that were written to be solved offline with tasks that expect
#: a network, and the sandbox has to follow each task's own declaration rather than
#: pick one answer for the whole run.
SessionFlag = Union[bool, Callable[[dict[str, Any]], bool]]

#: A resource budget that may be one figure for the run or one per session. None means
#: the backend's own default, which is not the same as zero and stays None.
SessionSize = Union[int, None, Callable[[dict[str, Any]], Union[int, None]]]

#: The same, for a setting that is a list of directories. Used for the read-only state
#: a session resumes from, which most sessions do not have and which differs per task
#: for the ones that do.
SessionDirs = Union[
    Sequence[str], Callable[[dict[str, Any]], Sequence[str]]]


def _resolve(value: Any, context: dict[str, Any]) -> str:
    """One session's value for a setting that may vary per session."""
    return value(context) if callable(value) else value


def _resolve_flag(value: Any, context: dict[str, Any]) -> bool:
    """One session's value for a flag that may vary per session."""
    return bool(value(context)) if callable(value) else bool(value)


def _resolve_size(value: Any, context: dict[str, Any]) -> int | None:
    """One session's figure for a resource budget that may vary per session.

    None passes through rather than becoming zero: a backend reads None as "no limit
    I impose" and zero as a limit of nothing."""
    resolved = value(context) if callable(value) else value
    return None if resolved is None else int(resolved)


def _resolve_dirs(value: Any, context: dict[str, Any]) -> tuple[str, ...]:
    """One session's directories for a setting that may vary per session."""
    return tuple(value(context) if callable(value) else value)


@dataclass
class _Session:
    """One trajectory's container, and the lock that serializes commands into it.

    The lock is not contention control. The loop is strictly one command per turn, so
    it should never be held twice; it is there so that a framework bug that fired two
    commands at once would serialize rather than interleave two ``exec`` calls on one
    environment.

    ``workdir`` is read back off the spec the container was launched from rather than
    off the runner, so the directory commands run in is the one the container actually
    got even when it varies per session.

    ``overlay_dir`` is the host directory holding everything the container wrote, or
    None when the backend used its own tmpfs. Kept here so closing the session can
    reclaim it without recomputing a path the launch already chose."""

    env: Any
    workdir: str
    overlay_dir: str | None = None
    #: The spec this container was launched from, kept so it can be launched again.
    #: The same spec and not a fresh one: the overlay upper lives on the host at the
    #: path this pool chose for the session, so a replacement built from this spec comes
    #: back with everything the trajectory wrote, while one built from a new spec
    #: would silently start it over in an empty checkout.
    spec: Any = None
    #: The session context, for re-running the task's setup command on a restart.
    context: dict = field(default_factory=dict)
    #: Containers this session has had replaced under it. Reported on every command
    #: so the diagnostics can tell a session that lost its sandbox three times from
    #: one that never did; they are not the same measurement.
    restarts: int = 0
    #: Why the container was replaced during the command now running, as
    #: ``Type: message``, or "" when this command replaced nothing. Read and cleared
    #: by ``run`` when it stamps the result, so it belongs to the turn that restarted
    #: rather than to every turn after it. Ends in ``[overlay recreated empty]`` when
    #: the overlay was gone and the replacement started without what the trajectory
    #: had written.
    restart_cause: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SandboxToolRunner(ToolRunner):
    """A bash tool backed by one container per trajectory.

    ``env_factory`` receives a ``LaunchSpec`` and returns a harbor environment,
    constructed but not yet started; this runner starts it and stops it again around
    the trajectory, so a factory only has to choose a backend. ``setup_command`` is run
    once inside the started container before the first turn, and is where a task stages
    its repository snapshot into ``workdir``. Both are supplied by the task, so this
    class carries no knowledge of what is being edited.

    Sessions are opened and closed by the framework around each trajectory
    (``ToolRunner.open_session`` / ``close_session``). A command for a session that
    was never opened is an error rather than an implicit open: an implicit one would
    silently give a trajectory a container with no repository in it, and the model
    would spend its whole budget in an empty directory.

    ``image`` and ``workdir`` each take either a string, shared by every trajectory, or
    a callable handed the session context. A benchmark that ships one environment per
    task needs both: the image because the task's tools and data are baked into it, and
    the working directory because it is whatever that image was built around, and
    starting a container somewhere it has no such directory fails the launch.

    ``allow_internet`` takes either a flag for the whole run or a callable handed the
    session context, so a corpus assembled from two benchmarks can give each task the
    network its own definition asks for. A backend that cannot seal refuses the launch
    rather than quietly running the task open, so a task that declares itself offline
    either gets that or fails loudly.

    ``cpus`` and ``memory_mb`` take the same two shapes, for the same reason one step
    further in. A corpus whose tasks declare their own environments spans a wide range
    -- an eightfold spread in CPUs is ordinary -- and one figure for all of them has to
    be either the largest, which is what bounds how many trajectories run at once, or
    smaller, which starves the largest task. Starving it is not a short trajectory: the
    container is OOM-killed mid-command and the runtime's error comes back as ordinary
    command output, so the model reads a harness failure as its own command failing.
    None leaves the budget to the backend.

    ``host_paths`` names what the launch has to reach inside the driver's container --
    the directory the images were staged in, and whatever device the runtime mounts them
    through. The driver is itself in a container, so nothing on the node is visible to
    it unless it was bound in, and a path left out here fails at the first turn of the
    first trajectory rather than at deploy. Each entry says whether it is bound
    read-only, and every sandbox sees these binds as the driver has them, so only what
    the drive writes is left writable (``HostPath``). The container that runs the driver
    carries the runtime and harbor: it is the task's harness image.
    """

    def __init__(
        self,
        *,
        image: SessionValue,
        env_factory: Callable[[Any], Any],
        workdir: SessionValue = "/testbed",
        setup_command: Callable[[dict[str, Any]], str] | None = None,
        cpus: SessionSize = None,
        memory_mb: SessionSize = None,
        allow_internet: SessionFlag = False,
        lower_overlays: SessionDirs = (),
        reclaim_overlays: bool = True,
        max_sandbox_restarts: int = 0,
        scratch_workdir: bool = True,
        mounts: list[dict] | None = None,
        host_paths: tuple[HostPath, ...] = (),
        overlay_root: str | None = None,
        env: dict[str, str] | None = None,
        default_timeout_sec: int = 300,
        setup_timeout_sec: int = 600,
        open_timeout_sec: float = 900.0,
        output_limit_chars: int = 1_000_000,
        server_python: str | None = None,
        row_entries: Callable[[dict[str, Any]], Sequence[tuple[str, str, str]]] | None = None,
        place_rows: Callable[[Sequence[dict[str, Any]], Mapping[str, Path]], list[str]] | None = None,
    ) -> None:
        self.image = image
        #: What a corpus row needs beside its image, as (container path, "file" or "dir", what it
        #: is), which ``store_entries`` turns into store entries; None when a row needs only its
        #: image.
        self.row_entries = row_entries
        #: What puts the corpus rows' ``row_entries`` into the site's stores from the task's own
        #: files, handed the rows and the stores by name to their directories; one line per entry,
        #: and StoreRefused for one that does not match what the task pins. None when nothing is
        #: placed from the task.
        self.place_rows = place_rows
        self.workdir = workdir
        self._host_paths = tuple(host_paths)
        self.overlay_root = overlay_root
        #: This pool's own directory under ``overlay_root``; see ``overlay_dir``.
        self._pool = _pool_token()
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.allow_internet = allow_internet
        self.lower_overlays = lower_overlays
        # Off only for an offline pass whose output is the overlay itself; see
        # ``close_session``. On for every graded run, or a drive leaks 30 GiB a
        # session and the filesystem fills with nobody watching.
        self.reclaim_overlays = reclaim_overlays
        # Replacement containers a session may have before its faults are handed to
        # the model instead. Zero keeps the documented behaviour, which is right for a
        # one-off blip; above zero is for an exec server that dies with nothing to bring
        # it back, when the trajectory would spend hundreds of turns talking to a
        # container that executes nothing. Bounded because rebuilding one that cannot
        # survive spends a drive's wall clock and produces no turns, which is worse than
        # ending the session.
        self.max_sandbox_restarts = max_sandbox_restarts
        self.scratch_workdir = scratch_workdir
        self.mounts = list(mounts or [])
        self.env = dict(env or {})
        self.default_timeout_sec = default_timeout_sec
        self.setup_timeout_sec = setup_timeout_sec
        #: How long a container may take to start before the session is given up on.
        #:
        #: Opening a session does two things and only one of them was bounded. The
        #: setup command has had ``setup_timeout_sec`` all along; starting the
        #: container had nothing, which is the expensive half and, for a task that
        #: sets no setup command, the only half that runs.
        #:
        #: Generous: this is not a latency budget and a slow node should not lose a
        #: session. It is there so that a start which never answers cannot hold an
        #: admission slot for the rest of the drive.
        self.open_timeout_sec = open_timeout_sec
        # A ceiling on what one command hands back, so a runaway `find /` cannot pull
        # a gigabyte of stdout through the driver. The loop elides again, far smaller,
        # for what the model sees; this one only bounds memory.
        self.output_limit_chars = output_limit_chars
        #: The interpreter the exec server runs on inside a session's container when the
        #: image's own lacks its packages, at the path a ``mounts`` entry binds it; None
        #: for images that all carry their own.
        self.server_python = server_python
        self._env_factory = env_factory
        self._setup_command = setup_command
        self._sessions: dict[str, _Session] = {}

    def describe_tools(self) -> list[ToolSpec]:
        return [BASH_TOOL]

    @property
    def host_paths(self) -> tuple[HostPath, ...]:
        return self._host_paths

    def _located(self, path: str, kind: str, what: str) -> StoreEntry | None:
        """*path*, as the driver's container sees it, as an entry of the store whose host path
        holds it; None for a path in no store."""
        for hp in self._host_paths:
            if hp.store != DEVICE and (path == hp.path or path.startswith(hp.path.rstrip("/") + "/")):
                return StoreEntry(store=hp.store, path=path[len(hp.path):].lstrip("/"), kind=kind, what=what)
        return None

    def _in_driver(self, path: str) -> str:
        """*path* inside a sandbox, as the driver's container sees it: through the runner's bind
        that carries it, to the bind's source; *path* itself when no bind carries it."""
        for m in self.mounts:
            target, source = str(m.get("target", "")), str(m.get("source", ""))
            if target and (path == target or path.startswith(target.rstrip("/") + "/")):
                return source + path[len(target):]
        return path

    def _runtime_entry(self) -> StoreEntry | None:
        """The exec server's interpreter (``server_python``) as a store entry; None when the
        runner names none or no store holds it."""
        if self.server_python is None:
            return None
        return self._located(self._in_driver(self.server_python), "file", "the exec server's interpreter")

    def store_entries(self, rows: Sequence[Any]) -> tuple[StoreEntry, ...]:
        """What the sandbox stores must hold before a drive of *rows* starts: each store's
        directory, the exec server's interpreter (``server_python``), each row's image and what
        ``row_entries`` names for it, each once. A path that lies in no store is not listed."""
        found: list[StoreEntry | None] = [StoreEntry(store=hp.store, path="", kind="dir", what="its directory")
                                          for hp in self._host_paths if hp.store != DEVICE]
        found.append(self._runtime_entry())
        for row in rows:
            context = dict(row)
            image = _resolve(self.image, context)
            found.append(self._located(image, "file", f"the image {image.rsplit('/', 1)[-1]}"))
            for path, kind, what in (self.row_entries(context) if self.row_entries is not None else ()):
                found.append(self._located(path, kind, what))
        return tuple(dict.fromkeys(e for e in found if e is not None))

    def provision_stores(self, stores: Mapping[str, Path], rows: Sequence[Any] = ()) -> list[str]:
        """Make what this runner needs in the site's stores, *stores* by name to their
        directories, before any drive of the corpus *rows*: each store's directory, the overlay
        root (``overlay_root``), the exec server's runtime where a store holds its interpreter
        (``hls.provision.sandbox_runtime``), and what ``place_rows`` places for the rows. Return
        one line per step. Raise KeyError for a store *stores* does not name, RuntimeStageError
        when the runtime cannot be staged, and StoreRefused for a row's entry that does not match
        what the task pins."""
        done = []
        for hp in self._host_paths:
            if hp.store != DEVICE:
                stores[hp.store].mkdir(parents=True, exist_ok=True)
                done.append(f"{hp.store}: its directory")
        overlay = self._located(self.overlay_root, "dir", "the overlay root") if self.overlay_root else None
        if overlay is not None and overlay.path:
            (stores[overlay.store] / overlay.path).mkdir(parents=True, exist_ok=True)
            done.append(f"{overlay.store}: the overlay root {overlay.path}")
        runtime = self._runtime_entry()
        if runtime is not None:
            from hls.provision import sandbox_runtime

            # The interpreter is <tree>/bin/python3, so the tree is two levels above it.
            tree = stores[runtime.store] / Path(runtime.path).parent.parent
            done.append(f"{runtime.store}: the exec server's runtime, {sandbox_runtime.stage(tree)}")
        if self.place_rows is not None:
            done += self.place_rows([dict(row) for row in rows], stores)
        return done

    def overlay_dir(self, session_id: str) -> str | None:
        """Where this session's container keeps what it writes, or None for the
        backend's default.

        One directory per session, inside one directory per pool: an overlay's upper
        layer is the container's whole writable filesystem, and two sessions sharing one
        would write over each other. The session id alone does not keep them apart. It
        is derived from the task, so every drive of the corpus opens the same ones, and
        drives on different nodes can share one root on a cluster filesystem, where the
        first of two concurrent drives to close a task's session would delete the
        directory under the other."""
        if not self.overlay_root:
            return None
        import os

        return os.path.join(self.overlay_root, self._pool, session_id)

    def lower_overlay_dirs(self, context: dict[str, Any]) -> tuple[str, ...]:
        """Directories this session's container starts from, stacked read-only under
        its own writable layer, or empty for one that starts from its image alone.

        For a session that resumes work an earlier container did: what that container
        wrote is mounted back, deletions included. Read-only, so concurrent sessions
        resuming the same state share one directory and none of them writes it."""
        return _resolve_dirs(self.lower_overlays, context)

    def launch_spec(self, session_id: str = "", **context: Any):
        """The ``LaunchSpec`` one session's container is started from.

        Imported here rather than at module scope so a harness that never runs a
        sandboxed task does not import harbor to define a task."""
        from harbor.environments.launch import LaunchSpec

        return LaunchSpec(
            image=_resolve(self.image, context),
            env=dict(self.env),
            mounts=list(self.mounts),
            cpus=_resolve_size(self.cpus, context),
            memory_mb=_resolve_size(self.memory_mb, context),
            allow_internet=_resolve_flag(self.allow_internet, context),
            workdir=_resolve(self.workdir, context),
            scratch_workdir=self.scratch_workdir,
            overlay_dir=self.overlay_dir(session_id),
            lower_overlay_dirs=self.lower_overlay_dirs(context),
            server_python=self.server_python,
            # The launch runs from its staging directory, so its relative paths resolve
            # there and not in whatever directory the driver happens to be in.
            start_in_staging_dir=True,
            # A container start is on a trajectory's clock: a coarse poll would charge
            # every session up to a second of waiting for a server that was already up.
            ready_poll_sec=SERVER_READY_POLL_SEC,
        )

    async def open_session(self, session_id: str, **context: Any) -> None:
        """Start this trajectory's container and stage its repository into it.

        Re-opening a live session is a no-op rather than a second container: a
        framework that retried a trajectory would otherwise leak the first one, and
        the leak would only show up as a node running out of memory hours later.

        Both halves are bounded: the container's start by ``open_timeout_sec`` and the
        setup command by ``setup_timeout_sec``. Exceeding either gives the container
        and its overlay back rather than holding an admission slot, and the first
        raises ``SandboxNotStarted``, which fails the drive as the harness's own fault
        rather than scoring it a session short."""
        if session_id in self._sessions:
            return
        spec = self.launch_spec(session_id=session_id, **context)
        overlay = getattr(spec, "overlay_dir", None)
        if overlay:
            # The backend names this directory but does not create it, and a launch
            # against a path that does not exist fails.
            import os

            os.makedirs(overlay, exist_ok=True)
        env = await _maybe_await(self._env_factory(spec))
        try:
            # force_build=False: the image is prebuilt, and a task's sandbox image is
            # not a thing this runner knows how to build.
            #
            # Bounded, because a launch that never answers would otherwise do nothing
            # at all: a session stuck here has not ended, so it would hold its
            # admission slot and contribute no traffic while the drive ran out its
            # wall clock around it. Past the bound it fails the drive, as a launch
            # that raises does.
            await asyncio.wait_for(
                _maybe_await(env.start(force_build=False)), self.open_timeout_sec)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            await _stop_quietly(env)
            self._reclaim(overlay)
            # Not SandboxGone, which drops one session: a session the drive asked for
            # never had a sandbox, and a run scored one session short for the
            # harness's own reason is a population nobody chose.
            from hls.core.faults import SandboxNotStarted

            raise SandboxNotStarted(
                f"sandbox not started: session={session_id!r} did not start within "
                f"{self.open_timeout_sec:g}s; the container was released"
            ) from exc
        except BaseException:
            # The session is not registered yet, so close_session cannot reach this
            # container; release it here or it outlives the trajectory.
            await _stop_quietly(env)
            self._reclaim(overlay)
            raise
        self._sessions[session_id] = _Session(env=env, workdir=spec.workdir,
                                              overlay_dir=overlay, spec=spec,
                                              context=dict(context))
        if self._setup_command is None:
            return
        try:
            command = self._setup_command(dict(context))
            result = await env.exec(command, cwd=spec.workdir,
                                    timeout_sec=self.setup_timeout_sec)
            if _return_code(result) != 0:
                # Fail the trajectory rather than run it against an empty directory:
                # every turn after this would measure the model's reaction to a
                # missing repository, which is not what the task poses.
                raise RuntimeError(
                    f"sandbox setup failed for session {session_id!r} "
                    f"(exit {_return_code(result)}): {_combined(result)[:2000]}")
        except BaseException:
            # Staging is the last thing open_session does, so a session that gets here
            # never opened; release its container rather than wait for a close that
            # the framework has no reason to make.
            await self.close_session(session_id)
            raise

    async def close_session(self, session_id: str) -> None:
        """Stop the container ``open_session`` started for this trajectory.

        A session that was never opened is not an error: a trajectory can fail before
        its first turn, and the framework closes it either way.

        An overlay directory goes with the container. What it holds is whatever the
        trajectory wrote, which a graded run never reads again -- the gate scores the
        served tokens, not the work -- and trajectories that each keep theirs fill
        the filesystem they are all on. Removed after the container
        stops, so nothing is pulled out from under a process still writing to it.

        Unless ``reclaim_overlays`` is off, which is for the one caller whose output
        *is* that directory: an offline pass that captures what a container wrote so a
        later session can start from it. Reclaiming there would delete the artifact
        the pass exists to produce, and the pass would report success."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        await _stop_quietly(session.env)
        if self.reclaim_overlays:
            self._reclaim(session.overlay_dir)

    def _reclaim(self, overlay: str | None) -> None:
        """Delete a session's overlay directory, and this pool's directory with its
        last session. Nothing outside this pool is touched: the session directory is
        under this pool's own, and ``rmdir`` refuses a pool directory that still holds
        another session."""
        if not overlay:
            return
        import os

        _remove_quietly(overlay)
        try:
            os.rmdir(os.path.dirname(overlay))
        except OSError:
            pass

    def restarts(self, session_id: str) -> int:
        """Containers this session has had replaced under it."""
        session = self._sessions.get(session_id)
        return session.restarts if session else 0

    async def _restart(self, session_id: str, session: _Session) -> bool:
        """Replace this session's container, in place, from the spec it was launched
        from. True when the trajectory can carry on.

        Nothing of the trajectory is lost that was not already lost: the loop runs
        every command in a fresh subshell, so no working directory, environment or
        process survives a command anyway, and the overlay the replacement mounts is
        the same one. What does not come back is anything the container held only in
        memory or in a tmpfs -- ``/dev/shm`` above all, which is where some agents do
        most of their work.

        Returns False rather than raising if the replacement cannot start. The caller
        then reports the fault to the model as it did before restarts existed, so a
        node out of disk ends a trajectory on its own stopping rules rather than on an
        exception.

        An overlay directory that is gone is made again, empty, because a launch
        against a missing one fails. What the trajectory wrote is lost either way, and
        the restart cause says so rather than passing for a restart that came back
        intact."""
        await _stop_quietly(session.env)
        try:
            overlay = session.overlay_dir
            if overlay:
                import os

                if not os.path.isdir(overlay):
                    os.makedirs(overlay, exist_ok=True)
                    session.restart_cause = (
                        f"{session.restart_cause} [overlay recreated empty]".lstrip())
            env = await _maybe_await(self._env_factory(session.spec))
            await _maybe_await(env.start(force_build=False))
        # Not a cancellation: a session cut while its container is being replaced ends
        # as cut, not as a sandbox that could not be replaced.
        except Exception:
            return False
        session.env = env
        session.restarts += 1
        if self._setup_command is None:
            return True
        try:
            result = await env.exec(self._setup_command(dict(session.context)),
                                    cwd=session.workdir,
                                    timeout_sec=self.setup_timeout_sec)
            # A replacement that could not stage its repository is no better than the
            # container it replaced, so it is not offered to the trajectory.
            return _return_code(result) == 0
        except Exception:
            return False

    async def run(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one command in this session's container.

        Every result carries ``restarted``, the containers this session has had
        replaced under it so far. It is stamped here, once, rather than by each return
        inside ``_run``, because that is the difference between a guarantee and a
        convention: a reader told the field is always present has given up the one
        signal that would catch a writer who forgot, so the promise has to be kept
        where it is made.

        A return that left the field out would not show in the reported value: the
        reader's own default turns the absent key into a literal zero.

        The count is the session's, not the command's: it does not reset, and a reader
        comparing consecutive turns can treat any fall as a reporting gap rather than
        an event.

        ``restart_cause`` is stamped the same way and is the opposite kind of
        quantity: the exception that provoked a replacement during *this* command, or
        "" when this command provoked none. A session that restarts and recovers would
        otherwise record that a replacement happened and nothing about what it was
        replacing, which is exactly the case a healthy run produces."""
        if tool_name != BASH_TOOL.name:
            raise KeyError(f"unknown tool {tool_name!r}; have {BASH_TOOL.name!r}")
        session_id = args.get("session_id")
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(
                f"no open sandbox for session {session_id!r}; the framework must call "
                f"open_session before the trajectory's first command")
        async with session.lock:
            out = await self._run(session_id, session, args)
            # Read under the same lock the command ran under. Outside it, a second
            # command on this session could restart between these two lines and this
            # result would report a replacement that happened after it finished.
            out["restarted"] = session.restarts
            # Cleared as it is read, because the cause belongs to the command that
            # provoked it. The tally is cumulative and should be; a cause carried
            # forward would have every later turn claiming a fault that happened once.
            out["restart_cause"] = session.restart_cause
            session.restart_cause = ""
        return out

    async def _run(self, session_id: str, session: "_Session",
                   args: dict[str, Any]) -> dict[str, Any]:
        """One command's own outcome, without the session's restart tally.

        Split from ``run`` so that every path out of here is stamped by one line there.
        Returns from this method deliberately do not set ``restarted``. The caller
        holds the session lock."""
        command = args["command"]
        timeout = int(args.get("timeout_sec") or self.default_timeout_sec)
        try:
            result = await session.env.exec(command, cwd=session.workdir,
                                            timeout_sec=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return {"output": f"Command timed out after {timeout} seconds.",
                    "exit_code": TIMEOUT_EXIT_CODE, "timed_out": True}
        except Exception as exc:
            # The sandbox broke, not the command. A trajectory is the traffic this
            # benchmark exists to generate, so the first move is to get a container
            # back rather than to tell the model its command failed.
            #
            # Recorded before anything is attempted, so it survives the path where the
            # replacement works. Otherwise `exc` would reach a record only when the
            # budget ran out or restarts were off, and a run whose restarts all succeed
            # would never say what any of them was for.
            if self.max_sandbox_restarts:
                session.restart_cause = _one_line(
                    f"{type(exc).__name__}: {exc}")
            result = None
            for _ in range(max(0, self.max_sandbox_restarts
                               - session.restarts)):
                if not await self._restart(session_id, session):
                    break
                try:
                    result = await session.env.exec(
                        command, cwd=session.workdir, timeout_sec=timeout)
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    return {"output": f"Command timed out after {timeout} "
                                      f"seconds.",
                            "exit_code": TIMEOUT_EXIT_CODE, "timed_out": True}
                except Exception:
                    result = None
            if result is None and self.max_sandbox_restarts:
                # A caller that asked for restarts and watched them run out has a
                # sandbox that stopped working, not an unlucky command, so this is
                # ours and it propagates: the trajectory driver drops the session
                # rather than score a failed request against a contestant whose
                # server never touched the exec channel. The message is the record
                # -- the driver writes it into the trace as the drop's reason -- so
                # it carries what this layer knows, which is containers. Turn counts
                # reach the trace through the session's own final event instead; the
                # runner does not know about turns and should not learn.
                from hls.core.faults import SANDBOX_GONE, SandboxGone

                # ``restarts`` is what was spent, which is what the same field
                # means in the message a task's loop writes; the budget it was
                # spent against is its own field rather than a second format for
                # the first one.
                raise SandboxGone(
                    f"{SANDBOX_GONE} session={session_id!r} "
                    f"restarts={session.restarts} "
                    f"restart_budget={self.max_sandbox_restarts} "
                    f"last_error={_one_line(f'{type(exc).__name__}: {exc}')}"
                ) from exc
            if result is None:
                # Restarts were not asked for. Reported to the model as a failed
                # command so the trajectory continues and the run is scored on what
                # the server served, and flagged so triage can tell the two apart.
                # A caller that would rather end the trajectory on a run of these
                # counts the flag itself and raises SandboxGone, which drops the
                # session the same way exhausting a budget above does.
                return {"output": f"Sandbox error: {type(exc).__name__}: {exc}",
                        "exit_code": SANDBOX_ERROR_EXIT_CODE, "timed_out": False,
                        "sandbox_error": True}
        return {"output": _combined(result)[:self.output_limit_chars],
                "exit_code": _return_code(result), "timed_out": False}


#: Environment variable naming the backend a sandbox is launched on, overriding the
#: probe below. Set on the driver's image by a task that knows its substrate.
SANDBOX_BACKEND_ENV = "HLS_SANDBOX_BACKEND"


def default_env_factory(spec: Any) -> Any:
    """Build an environment for ``spec`` on whichever container runtime this host has.

    Construction only. ``SandboxToolRunner.open_session`` starts what a factory hands
    back and stops it again at the end of the trajectory, so a task replacing this
    factory chooses a backend and nothing else.

    The task declares its sandbox before it knows where it will run, and the driver
    that eventually runs the loop is the only thing that knows -- a workstation has
    Docker, a Slurm node has Apptainer and no daemon. So the choice is made here, at
    the first ``open_session``, from ``SANDBOX_BACKEND_ENV`` when a task pinned it and
    otherwise from which runtime is on PATH.

    Apptainer is probed first: a login node may have both, and on a cluster the
    daemonless one is the one that will actually work from inside a job."""
    import os
    import shutil

    backend = os.environ.get(SANDBOX_BACKEND_ENV, "").strip().lower()
    if not backend:
        if shutil.which("apptainer") or shutil.which("singularity"):
            backend = "apptainer"
        elif shutil.which("docker"):
            backend = "docker"
        else:
            raise RuntimeError(
                "no container runtime for the sandbox: neither apptainer/singularity "
                f"nor docker is on PATH. Set {SANDBOX_BACKEND_ENV} if one is "
                "installed somewhere this probe does not look.")
    if backend in ("apptainer", "singularity"):
        from harbor.environments.singularity.singularity import SingularityEnvironment
        return SingularityEnvironment.launch(spec)
    if backend == "docker":
        from harbor.environments.docker.docker import DockerEnvironment
        return DockerEnvironment.launch(spec)
    raise RuntimeError(
        f"{SANDBOX_BACKEND_ENV}={backend!r} names no sandbox backend; "
        f"expected 'apptainer' or 'docker'")


def _pool_token() -> str:
    """A name for one pool's directory under the overlay root. The random part keeps
    pools apart; the host and pid are there so a listing of the root says which
    process each directory belongs to."""
    import os
    import socket
    import uuid

    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _remove_quietly(path: str | None) -> None:
    """Delete a session's overlay directory, swallowing whatever removal raises.

    Same reason as ``_stop_quietly``: by the time this runs the trajectory is over and
    its measurement is recorded, so raising here would turn leaked disk into a failed
    run."""
    if not path:
        return
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


async def _stop_quietly(env: Any) -> None:
    """Release a container, swallowing whatever teardown raises.

    By the time this runs the trajectory is over and its measurement is recorded, so
    raising here would turn a leaked container into a failed run. The substrate reaps
    what is left when the job ends."""
    try:
        await _maybe_await(env.stop(delete=True))
    except Exception:
        pass


async def _maybe_await(value: Any) -> Any:
    """The value, awaited when the backend returned a coroutine.

    harbor's ``launch`` is a sync classmethod while ``stop`` is async, and a test's
    fake is whichever is easier to write, so both shapes are accepted."""
    if asyncio.iscoroutine(value):
        return await value
    return value


#: Characters of an underlying error kept in a fault message. A transport library's
#: exception can carry a whole response body, and the message is recorded in one field
#: of the run record beside everything else the fault names.
_ERROR_CHARS = 200


def _one_line(text: str) -> str:
    """``text`` with every run of whitespace collapsed to a single space, truncated.

    Fault messages are recorded as one line, and the text being quoted into them comes
    from somewhere else's exception, which may be a paragraph."""
    flat = " ".join(str(text).split())
    return flat[:_ERROR_CHARS]


def _return_code(result: Any) -> int:
    """The exit status of an ``ExecResult``, defaulting to success when a backend
    omitted it."""
    return int(getattr(result, "return_code", 0) or 0)


def _combined(result: Any) -> str:
    """stdout and stderr as the shell would have shown them.

    Merged rather than reported separately because the model is reading a terminal,
    where a compiler's warnings and its output are interleaved in one stream. Order is
    stdout then stderr, which is the best a backend that captured them separately
    allows."""
    out = getattr(result, "stdout", None) or ""
    err = getattr(result, "stderr", None) or ""
    if out and err:
        return f"{out}\n{err}" if not out.endswith("\n") else f"{out}{err}"
    return out or err


__all__ = ["BASH_TOOL", "SANDBOX_BACKEND_ENV", "SANDBOX_ERROR_EXIT_CODE",
           "TIMEOUT_EXIT_CODE", "SandboxToolRunner", "SessionDirs", "SessionFlag",
           "SessionValue", "default_env_factory"]
