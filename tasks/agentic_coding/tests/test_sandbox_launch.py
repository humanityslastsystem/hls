"""A session's sandbox launch, as harbor's Singularity backend renders it from the task's tool
runner: every path the launch reads or writes in the driver's container lies under a path the
drive binds there from the site's sandbox stores (``TOOL_RUNNER.host_paths``), and only the
overlay root, which the launch writes, is bound writable. The launch runs from its staging
directory, names the exec server's interpreter the runtime store holds, polls a starting
server quickly, and isolates the network of a row that declares itself offline.

Rendered through harbor's own ``start``, with its process and its control channel stood in for
as harbor's unit tests stand them in, so no container runs."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path, PurePosixPath

import pytest

from harbor.environments.singularity import singularity as sg
from harbor.environments.singularity.singularity import SingularityEnvironment
from hls.spec.sandbox import SANDBOX_BACKEND_ENV, SERVER_READY_POLL_SEC, default_env_factory
from hls.spec.tools import DEVICE, HostPath
from tasks.agentic_coding import prefix as prefix_format
from tasks.agentic_coding import datasets as corpus
from tasks.agentic_coding import task as task_mod
from tasks.agentic_coding.datasets import (ALLOW_INTERNET_FIELD, CPUS_FIELD, MEMORY_MB_FIELD, PREFIX_DIGEST_FIELD,
                                           TASK_NAME_FIELD, WORKDIR_FIELD)
from hls.provision.sandbox_runtime import RUNTIME_TAG
from tasks.agentic_coding.task import (SANDBOX_IMAGE_DIR, SANDBOX_OVERLAY_DIR, SANDBOX_PREFIX_DIR,
                                       SANDBOX_RUNTIME_DIR, SANDBOX_RUNTIME_MOUNT, TOOL_RUNNER)

#: A row that declares itself offline, and one that resumes a pre-rolled prefix with a network.
OFFLINE = {TASK_NAME_FIELD: "fix-permissions", WORKDIR_FIELD: "/app", ALLOW_INTERNET_FIELD: False,
           CPUS_FIELD: 2, MEMORY_MB_FIELD: 4096}
RESUMED = {TASK_NAME_FIELD: "git-to-zig", WORKDIR_FIELD: "/app", ALLOW_INTERNET_FIELD: True,
           CPUS_FIELD: 4, MEMORY_MB_FIELD: 8192, PREFIX_DIGEST_FIELD: "e" * 64}


class _Stdout:
    async def readline(self):
        return b""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _Proc:
    """A launch process that stays alive; its pid is this test's own, a harmless target."""

    def __init__(self):
        self.returncode = None
        self.pid = os.getpid()
        self.stdout = _Stdout()

    async def wait(self):
        await asyncio.sleep(5)


class _Ready:
    status_code = 200


class _Client:
    """A control channel that answers at once."""

    async def get(self, url):
        return _Ready()

    async def aclose(self):
        pass


class _State:
    """A pre-rolled state directory that exists, at the path the task's runner composes."""

    def __init__(self, path: Path):
        self.path = path

    def is_dir(self) -> bool:
        return True

    def __str__(self) -> str:
        return str(self.path)


@pytest.fixture
def staged_prefixes(monkeypatch):
    """Every prefix's state is staged, so a resumed row mounts its state."""
    monkeypatch.setattr(task_mod, "prefix_format", type("Prefixes", (), {
        "state_path": staticmethod(lambda d, name: _State(prefix_format.state_path(d, name)))}))


def _launch(monkeypatch, context: dict) -> tuple[object, SingularityEnvironment, list[str], dict]:
    """The launch spec of one session of *context*, the environment the task's factory builds
    from it on apptainer, and the command and spawn options its start runs."""
    monkeypatch.setenv(SANDBOX_BACKEND_ENV, "apptainer")
    spec = TOOL_RUNNER.launch_spec("s0", **context)
    env = default_env_factory(spec)
    assert isinstance(env, SingularityEnvironment)
    launches: list[tuple[tuple, dict]] = []

    async def spawn(*args, **kwargs):
        launches.append((args, kwargs))
        return _Proc()

    async def no_teardown(self):
        """No container was started."""

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(sg.httpx, "AsyncClient", lambda **kw: _Client())
    monkeypatch.setattr(SingularityEnvironment, "_record_container_init", lambda self: None)
    monkeypatch.setattr(SingularityEnvironment, "_terminate_sandbox", no_teardown)
    try:
        asyncio.run(env.start(force_build=False))
    finally:
        if env._staging_dir is not None:
            shutil.rmtree(env._staging_dir, ignore_errors=True)
    [(argv, kwargs)] = launches
    return spec, env, list(argv), kwargs


def _bound(path: str) -> HostPath:
    """The host path of the driver's binds that *path* lies under."""
    [held] = [p for p in TOOL_RUNNER.host_paths
              if p.store != DEVICE and PurePosixPath(path).is_relative_to(p.path)]
    return held


def _overlays(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--overlay"]


def _binds(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-B"]


def test_the_driver_is_bound_each_store_the_launch_names_and_the_fuse_device():
    assert {(p.path, p.read_only, p.store) for p in TOOL_RUNNER.host_paths} == {
        (SANDBOX_IMAGE_DIR, True, task_mod.IMAGE_STORE), (SANDBOX_OVERLAY_DIR, False, task_mod.OVERLAY_STORE),
        (SANDBOX_PREFIX_DIR, True, task_mod.PREFIX_STORE), (SANDBOX_RUNTIME_DIR, True, task_mod.RUNTIME_STORE),
        ("/dev/fuse", False, DEVICE)}
    assert TOOL_RUNNER.overlay_root == SANDBOX_OVERLAY_DIR


@pytest.mark.parametrize("context", [OFFLINE, RESUMED], ids=["offline", "resumed"])
def test_every_path_the_launch_names_lies_under_a_bind_writable_only_where_it_writes(monkeypatch, staged_prefixes,
                                                                                    context):
    spec, _, _, _ = _launch(monkeypatch, context)
    reads = [spec.image, *spec.lower_overlay_dirs, *(m["source"] for m in spec.mounts)]
    assert all(_bound(path).read_only for path in reads), reads
    assert not _bound(spec.overlay_dir).read_only
    assert _bound(spec.image).path == SANDBOX_IMAGE_DIR and _bound(spec.overlay_dir).path == SANDBOX_OVERLAY_DIR
    assert [_bound(m["source"]).path for m in spec.mounts] == [SANDBOX_RUNTIME_DIR]


def test_a_resumed_session_starts_from_its_state_in_the_prefix_store(monkeypatch, staged_prefixes):
    spec, _, argv, _ = _launch(monkeypatch, RESUMED)
    state = str(prefix_format.state_path(SANDBOX_PREFIX_DIR, "git-to-zig"))
    assert spec.lower_overlay_dirs == (state,)
    # The state read-only under the session's own writable layer, which comes last.
    assert _overlays(argv) == [f"{state}:ro", spec.overlay_dir]


def test_a_cold_session_starts_from_its_image_alone(monkeypatch):
    spec, _, argv, _ = _launch(monkeypatch, {k: v for k, v in RESUMED.items() if k != PREFIX_DIGEST_FIELD})
    assert spec.lower_overlay_dirs == ()
    assert _overlays(argv) == [spec.overlay_dir]


@pytest.mark.parametrize("context", [OFFLINE, RESUMED], ids=["offline", "resumed"])
def test_the_launch_sets_what_the_runner_must_set(monkeypatch, staged_prefixes, context):
    spec, env, argv, kwargs = _launch(monkeypatch, context)
    name = context[TASK_NAME_FIELD]
    assert spec.image == f"{SANDBOX_IMAGE_DIR}/{name}.sif" and argv[argv.index(spec.image) - 1] != "-B"
    # One writable layer per session, in this pool's directory of the overlay root, in place
    # of the runtime's capped session tmpfs; no scratch over the image's working directory.
    assert PurePosixPath(spec.overlay_dir).parent.parent == PurePosixPath(SANDBOX_OVERLAY_DIR)
    assert PurePosixPath(spec.overlay_dir).name == "s0"
    assert "--writable-tmpfs" not in argv and "--scratch" not in argv and spec.scratch_workdir is False
    # The exec server's interpreter, from the runtime store, read-only at the path harbor tries.
    assert spec.server_python == f"{SANDBOX_RUNTIME_MOUNT}/bin/python3"
    assert argv[argv.index("--env") + 1] == f"HARBOR_SERVER_PYTHON={SANDBOX_RUNTIME_MOUNT}/bin/python3"
    assert f"{SANDBOX_RUNTIME_DIR}/{RUNTIME_TAG}:{SANDBOX_RUNTIME_MOUNT}:ro" in _binds(argv)
    # The launch runs from its staging directory and polls a starting server every 50 ms.
    assert kwargs["cwd"] == env._staging_dir
    assert spec.start_in_staging_dir is True and spec.ready_poll_sec == SERVER_READY_POLL_SEC == 0.05
    assert (spec.cpus, spec.memory_mb) == (context[CPUS_FIELD], context[MEMORY_MB_FIELD])
    assert argv[argv.index("--pwd") + 1] == "/app"


def test_a_session_of_the_corpus_launches_with_the_flat_cpus_and_memory_whatever_its_row_declares():
    # The session's context is its seed's metadata, and the seed carries neither field of a
    # row's declaration, so the launch does not follow the row.
    row = {TASK_NAME_FIELD: "live-database-cutover", corpus.INSTRUCTION_FIELD: "do it", WORKDIR_FIELD: "/app",
           ALLOW_INTERNET_FIELD: True, CPUS_FIELD: 16, MEMORY_MB_FIELD: 32768, corpus.STORAGE_MB_FIELD: 30720}
    [item] = corpus._seed_items(row, 0, "test", "terminal-bench")
    context = dict(item.turns[0].metadata)
    assert CPUS_FIELD not in context and MEMORY_MB_FIELD not in context
    spec = TOOL_RUNNER.launch_spec(item.request_id, **context)
    assert (spec.cpus, spec.memory_mb) == (corpus.SANDBOX_CPUS, corpus.SANDBOX_MEMORY_MB) == (4, 16384)


def test_an_offline_row_gets_no_network_and_an_online_one_keeps_it(monkeypatch, staged_prefixes):
    spec, _, argv, _ = _launch(monkeypatch, OFFLINE)
    assert spec.allow_internet is False
    assert argv[argv.index("--net"):argv.index("--net") + 3] == ["--net", "--network", "none"]
    assert argv[argv.index("--uds") + 1] == "/staging/control.sock"
    spec, _, argv, _ = _launch(monkeypatch, RESUMED)
    assert spec.allow_internet is True and "--net" not in argv and "--port" in argv


def test_the_singularity_backend_takes_every_field_the_runner_sets(monkeypatch, staged_prefixes):
    # A backend that cannot honour a field refuses the spec when it is built; Singularity
    # declares overlay directories and network isolation.
    _, env, _, _ = _launch(monkeypatch, RESUMED)
    assert env.capabilities.overlay_dirs is True and env.capabilities.disable_internet is True
