from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

import hls.pipeline.stage as stage_module
from hls.agent.memory import ReadyPids, SplitSample
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.plan import Plan
from hls.pipeline.planner import StageDecl
from hls.pipeline.stage import (
    MARKER_SEARCH_BYTES,
    Stage,
    StageContext,
    StageFault,
    container_strings,
    register,
    stage_fault,
)
from hls.spec.resources import ResourceShape
from hls.store.cache import CacheEntry, ContentCache, copies_root, entry_sha256
from hls.store.digest import sha256_bytes
from hls.store.manifest import OutputRef
from hls.store.lock import GenerationLock, Holder
from hls.store.posix import PosixStore
from hls.substrate.apptainer.argv import HOST_FILES, START_MARKER, ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.stages.noop import NoopParams
from hls.substrate.base import FaultRecord, JobState, RoleHandle, RoleSpec
from hls.substrate.container import ContainerSpec, Mount, PinnedImage
from hls.substrate import log_cap
from hls.substrate.faults import classify_role, infra_fault
from hls.substrate.memory_watch import MemorySample, MemoryWatch
from hls.substrate.nproc import DEFAULT_NPROC_MAX, CeilingRule
from hls.substrate.role_exec import ENV_FD_PATH, dropped
from hls.substrate.secret_forms import SCRUB_CHUNK_BYTES
from hls.substrate.secrets import Scrubber
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan
from tests.store.entries import entry_reads, populate, recording_reads
from tests.substrate import fake_apptainer
from tests.substrate.sandboxes import IMAGE_SHA256, image_file

#: The stand-in for the cache's role program whose copies land other bytes than they read.
FLIPPING = Path(__file__).parents[1] / "store" / "flipping_cache_role.py"
#: The stand-in for it whose report names another digest for one file than it checked.
MISREPORTING = Path(__file__).parents[1] / "store" / "misreporting_cache_role.py"
#: A stand-in for it that sleeps until it is killed, or exits at once with ``$HANGING_ROLE_EXIT``.
HANGING = Path(__file__).with_name("hanging_role.py")

SHAPE = ResourceShape(cpus=2, memory_gib=2, disk_gib=1, time_sec=60)
SECRETS = {"HF_TOKEN": "hf-token-0123456789", "OTHER_TOKEN": "other-token-0123456789"}
#: A value of the secrets file that this stage was not granted, such as an agent's credential.
RELAY = "relay-token-0123456789"
#: How long after a command's own first instant the stage may first see it started: one pass of
#: its wait, and half a second for a loaded machine.
SEEN_WITHIN_SEC = stage_module._POLL_SEC + 0.5


class RecordingLauncher(LocalLauncher):
    """A local launcher that notes each role's env file as it was handed over."""

    def __init__(self, **knobs) -> None:
        super().__init__(gpus=2, **knobs)
        self.env_files: dict[str, tuple[int, str]] = {}

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        env_file = spec.env_file if spec.container is None else spec.container.env_file
        if env_file is not None:
            mode = stat.S_IMODE(env_file.stat().st_mode)
            self.env_files[spec.name] = (mode, env_file.read_text())
        return super().launch(spec, share_gpus_of=share_gpus_of)


class SkippingLauncher(LocalLauncher):
    """A local launcher whose roles never take their env file, like a step that died first."""

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        bare = RoleSpec(name=spec.name, argv=spec.argv, shape=spec.shape, owner=spec.owner,
                        log_dir=spec.log_dir)
        return super().launch(bare, share_gpus_of=share_gpus_of)


class TmpLauncher(LocalLauncher):
    """A local launcher that, like the Slurm one, gives each role a TMPDIR in its role
    directory, and puts a file there."""

    def __init__(self, scratch: Path) -> None:
        super().__init__(gpus=2)
        self.scratch = scratch

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        tmp = self.scratch / "roles" / spec.name / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "in-use").write_text("x")
        return super().launch(spec, share_gpus_of=share_gpus_of)


class RaisingLauncher(LocalLauncher):
    """A local launcher that cannot start anything."""

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        raise RuntimeError("srun is gone")


class BlockingLauncher(LocalLauncher):
    """A local launcher that holds the launch of one named role until it is released, so an
    off-thread start of it stays in flight."""

    def __init__(self, role: str) -> None:
        super().__init__(gpus=2)
        self.role = role
        self.reached = threading.Event()
        self.release = threading.Event()

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        if spec.name == self.role:
            self.reached.set()
            assert self.release.wait(30), "the test never released the blocked launch"
        return super().launch(spec, share_gpus_of=share_gpus_of)


class _BreachedWatch:
    """A memory watch that says every step was killed for memory, and knows no job cgroup."""

    #: No job cgroup, so the job's OOM counts are unknown.
    job_dir = None

    def watch(self, step, cgroup_dir, limit_bytes, kill, *, role) -> None:
        pass

    def job_ooms(self) -> None:
        return None

    def last(self, step) -> MemorySample | None:
        return None

    def breach(self, step) -> MemorySample:
        return MemorySample(at=0.0, current=3 * 2**30, swap_current=0, reclaimable=0, oom_kills=0)

    def watch_disk(self, step, paths, limit_bytes, kill, *, role, cgroup_dir=None, ceiling=None,
                   harness_pids=None) -> None:
        pass

    def unwatch_disk(self, step) -> None:
        pass

    def disk_breach(self, step) -> None:
        return None

    def pids_hits(self, step) -> None:
        return None

    def pool(self, step) -> None:
        return None

    def pool_peak(self, step) -> None:
        return None

    def disk_at_end(self, step) -> None:
        return None


def _context(plan: Plan, launcher: LocalLauncher, scratch: Path, *, secrets: dict[str, str] | None = None,
             scrubber: Scrubber | None = None, watch: MemoryWatch | _BreachedWatch | None = None,
             deadline_in: float = 60.0) -> StageContext:
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("noop_a", 1, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage("noop_a"), attempt=1, launcher=launcher.use_plan(plan),
                        watch=watch, events=events, inputs={}, deadline=time.time() + deadline_in,
                        secrets=secrets or {}, scrubber=scrubber, cache=ContentCache(store), scratch=scratch,
                        job_id="100", runtime=ApptainerRuntime(), images=ApptainerImages(scratch))


def _ctx(tmp_path: Path, launcher: LocalLauncher, *, site: dict | None = None, **kw) -> tuple[Plan, StageContext]:
    plan = noop_plan(tmp_path, site=site)
    return plan, _context(plan, launcher, tmp_path / "node" / "hls-100", **kw)


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _first_then_sleep(first: Path) -> tuple[str, ...]:
    """A command that writes the time it started to *first*, then sleeps a minute."""
    return tuple(_python(f"import time; open({str(first)!r}, 'w').write(repr(time.time())); time.sleep(60)"))


def _dump_env(out: Path) -> list[str]:
    """A command that writes its environment, less what any process has, to *out*."""
    return _python("import json, os, sys; skip = {'PATH', 'HOME', 'LC_CTYPE', 'PWD', 'SHLVL', '_'}; "
                   "env = {k: v for k, v in os.environ.items() if k not in skip}; "
                   "json.dump(env, open(sys.argv[1], 'w'))") + [str(out)]


def _err(ctx: StageContext, role: str) -> str:
    return (ctx.attempt_dir / "logs" / f"{role}.err").read_text()


def _pinned(ctx: StageContext) -> PinnedImage:
    """A small image file extracted ahead in the context's scratch, pinned by the context."""
    assert ctx.scratch is not None
    return ctx.pin_file(image_file(ctx.scratch), sha256=IMAGE_SHA256)


def _quick(ctx: StageContext, name: str = "srv", *, command: tuple[str, ...] = ("true",), **kw) -> ContainerSpec:
    """A container for role *name* on an image the context pinned, whose workdir lies in its own
    role directory."""
    return ContainerSpec(image=_pinned(ctx), command=command, workdir=ctx.role_dir(name) / "wd", **kw)


def _entry(ctx: StageContext, key: str = "org+m@" + "a" * 40) -> Path:
    """Publish a weights entry holding ``w.bin`` and return the copy ``ctx.materialize`` made."""
    return _copy(ctx, "weights", key, {"w.bin": b"w"})


def _copy(ctx: StageContext, kind: str, key: str, files: dict[str, bytes]) -> Path:
    """Publish entry *kind*/*key* holding *files* and return the copy ``ctx.materialize`` made."""
    entry = _entry_of(ctx, kind, key, files)
    return ctx.materialize(kind, key, entry_sha256(entry))


def _entry_of(ctx: StageContext, kind: str, key: str, files: dict[str, bytes]) -> CacheEntry:
    """Publish entry *kind*/*key* holding *files*."""

    def fill(d: Path) -> None:
        for rel, data in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(data)

    return populate(ctx.cache, kind, key, fill, holder=Holder.here("test")).entry


def _own_roles(launcher: LocalLauncher) -> list[str]:
    """The roles a test started itself: all but the cache's copies of the entries it mounts."""
    return [s.name for s in launcher.launched if s.argv[1:3] != ("-m", "hls.store.cache_role")]


def _apptainer_script(bin_dir: Path, body: str) -> None:
    """Replace the fake apptainer with a shell script that runs *body*."""
    path = bin_dir / "apptainer"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


@pytest.fixture
def apptainer_on_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    fake_apptainer.install(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    monkeypatch.delenv("FAKE_APPTAINER_FAIL", raising=False)
    return bin_dir


def test_a_roles_secrets_reach_it_in_a_private_file_that_is_gone_once_it_starts(tmp_path):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS)
    out = tmp_path / "env.json"
    ctx.start_role("dl", _python("import json, os, sys; json.dump({k: os.environ.get(k) for k in "
                                 "('HF_TOKEN', 'OTHER_TOKEN')}, open(sys.argv[1], 'w'))") + [str(out)],
                   SHAPE, secret_env=["HF_TOKEN"], owner="harness")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0
    mode, text = launcher.env_files["dl"]
    assert mode == 0o600 and "HF_TOKEN=" in text and "OTHER_TOKEN" not in text
    spec = launcher.launched[0]
    assert spec.env_file == ctx.role_dir("dl") / "role.env" and not spec.env_fd and spec.env == {}
    assert not spec.env_file.exists()
    assert json.loads(out.read_text()) == {"HF_TOKEN": SECRETS["HF_TOKEN"], "OTHER_TOKEN": None}
    assert stat.S_IMODE(ctx.role_dir("dl").stat().st_mode) == 0o700


def test_a_roles_start_records_where_its_cpus_lie(tmp_path):
    # Eight CPUs given four on each socket, as a Slurm grant can give them: the step
    # wrapper reports them as split (hls.substrate.role_exec.placement).
    split = ([16, 17, 18, 19, 76, 77, 80, 81], [0, 1], True)
    launcher = RecordingLauncher(placement={"srv": split})
    _, ctx = _ctx(tmp_path, launcher)
    ctx.start_role("srv", ["true"], SHAPE, owner="contestant")
    ctx.wait_roles()
    ctx.start_role("drv", ["true"], SHAPE, owner="harness")
    ctx.wait_roles()
    started = {e["role"]: e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_started"}
    assert (started["srv"]["cpus"], started["srv"]["numa_nodes"], started["srv"]["numa_split"]) == split
    assert (started["drv"]["cpus"], started["drv"]["numa_nodes"], started["drv"]["numa_split"]) == ([], [], False)


def test_a_secret_the_stage_was_not_granted_is_refused(tmp_path):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets={"HF_TOKEN": SECRETS["HF_TOKEN"]})
    with pytest.raises(ValueError, match="was not granted .*OTHER_TOKEN"):
        ctx.start_role("dl", _python("pass"), SHAPE, secret_env=["OTHER_TOKEN"], owner="harness")
    assert launcher.launched == [] and not (ctx.role_dir("dl") / "role.env").exists()


@pytest.mark.parametrize("where", ["env", "argv"])
@pytest.mark.parametrize("value", [SECRETS["HF_TOKEN"], f"Bearer {SECRETS['HF_TOKEN']}", RELAY])
def test_a_secret_value_outside_secret_env_is_refused(tmp_path, value, where):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS, scrubber=Scrubber({"relay": RELAY}))
    env, argv = {"PLAIN": "1"}, _python("pass")
    if where == "env":
        env["AUTH"] = value
    else:
        argv.append(value)
    named = "env AUTH" if where == "env" else "argv[3]"
    with pytest.raises(ValueError, match=re.escape(f"role dl: {named} holds a secret")) as caught:
        ctx.start_role("dl", argv, SHAPE, env=env, owner="harness")
    assert value not in str(caught.value)
    assert launcher.launched == []


def test_a_contestant_role_gets_no_stage_grant(tmp_path):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS)
    with pytest.raises(ValueError, match="contestant role contest: .*host-only"):
        ctx.start_role("contest", _python("pass"), SHAPE, owner="contestant", secret_env=["HF_TOKEN"])
    assert launcher.launched == [] and not (ctx.role_dir("contest") / "role.env").exists()


def test_a_stale_file_at_the_env_file_path_is_replaced(tmp_path):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS)
    (ctx.role_dir("dl") / "role.env").write_text("STALE='1'\n")
    ctx.start_role("dl", _python("pass"), SHAPE, secret_env=["HF_TOKEN"], owner="harness")
    ctx.wait_roles()
    _, text = launcher.env_files["dl"]
    assert "STALE" not in text and "HF_TOKEN=" in text


@pytest.mark.parametrize("name", ["../x", "a/b", ".."])
def test_a_role_name_that_is_not_one_path_segment_is_refused_before_anything_is_written(tmp_path, name):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS)
    with pytest.raises(ValueError, match="role name"):
        ctx.role_dir(name)
    with pytest.raises(ValueError, match="role name"):
        ctx.start_role(name, _python("pass"), SHAPE, secret_env=["HF_TOKEN"], owner="harness")
    assert launcher.launched == [] and not (tmp_path / "node" / "hls-100" / "x").exists()
    assert not list((tmp_path / "node").rglob("role.env"))


def _fresh(role: Path) -> list[Path]:
    """What a fresh role directory holds: its step's tmp and its log cappers' error files."""
    return sorted([role / "tmp", *(log_cap.errors_file(role, suffix) for suffix in (".out", ".err"))])


def test_a_role_directory_left_by_an_earlier_stage_is_emptied_before_this_stage_uses_it(tmp_path):
    plan = noop_plan(tmp_path)
    scratch = tmp_path / "node" / "hls-100"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kept").write_text("k")
    earlier = _context(plan, RecordingLauncher(), scratch)
    left = earlier.role_dir("srv")
    (left / "left-by-noop_a").write_text("x")
    (left / "link").symlink_to(outside)
    (left / "sub").mkdir()
    (left / "sub" / "deep").write_text("y")
    later = _context(plan.model_copy(), RecordingLauncher(), scratch)
    own = later.role_dir("srv")
    # Fresh: only the empty tmp directory its step will use and its log cappers' empty error files.
    assert own == left and sorted(own.iterdir()) == _fresh(own) and list((own / "tmp").iterdir()) == []
    assert stat.S_IMODE(own.stat().st_mode) == 0o700
    assert (outside / "kept").read_text() == "k"
    (own / "made-here").write_text("z")
    assert later.role_dir("srv") == own and (own / "made-here").exists()


def test_a_role_directory_that_is_a_link_is_replaced_by_a_directory(tmp_path):
    plan = noop_plan(tmp_path)
    scratch = tmp_path / "node" / "hls-100"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kept").write_text("k")
    (scratch / "roles").mkdir(parents=True)
    (scratch / "roles" / "srv").symlink_to(outside)
    own = _context(plan, RecordingLauncher(), scratch).role_dir("srv")
    assert not own.is_symlink() and own.is_dir() and sorted(own.iterdir()) == _fresh(own)
    assert (outside / "kept").read_text() == "k"


def test_a_started_roles_directory_is_never_emptied_under_it(tmp_path):
    scratch = tmp_path / "node" / "hls-100"
    _, ctx = _ctx(tmp_path, TmpLauncher(scratch))
    ctx.start_role("dl", _python("import time; time.sleep(0.5)"), SHAPE, owner="harness")
    assert (ctx.role_dir("dl") / "tmp" / "in-use").exists()
    ctx.wait_roles()
    ctx.close()
    assert list((scratch / "roles").iterdir()) == []


def test_close_removes_the_role_directories_the_context_made(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    (ctx.role_dir("srv") / "data").write_text("x")
    ctx.start_role("dl", _python("pass"), SHAPE, owner="harness")
    ctx.close()
    assert list((tmp_path / "node" / "hls-100" / "roles").iterdir()) == []


def test_a_container_gets_exactly_its_env_file_through_descriptor_3(tmp_path, apptainer_on_path, monkeypatch):
    # --cleanenv would pass these from the step's environment; the step removes them first.
    for name in [k for k in os.environ if dropped(k, container=True)]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS)
    out = ctx.role_dir("srv") / "out"
    out.mkdir()
    spec = ContainerSpec(image=_pinned(ctx), command=tuple(_dump_env(out / "env.json")),
                         workdir=ctx.role_dir("srv") / "wd", mounts=(Mount(out, "/out", read_only=False),))
    plain = {"PLAIN": "1", "FD3_VALUE": "fd3-plain-0123456789"}
    ctx.start_container("srv", spec, SHAPE, env=plain, owner="harness")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0, _err(ctx, "srv")
    role = launcher.launched[0]
    assert role.argv == tuple(ApptainerRuntime().argv(spec, env_file=ENV_FD_PATH))
    assert role.env_fd and role.env == {} and role.start_marker == START_MARKER
    assert role.env_file == ctx.role_dir("srv") / "container.env" and not role.env_file.exists()
    mode, text = launcher.env_files["srv"]
    assert mode == 0o600 and "HF_TOKEN" not in text
    assert json.loads((out / "env.json").read_text()) == plain
    assert START_MARKER in _err(ctx, "srv")
    [started] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_started"]
    assert started["stripped_env"] == ["HTTPS_PROXY", "TERM"]


@pytest.mark.parametrize("owner", ["harness", "contestant"])
def test_apptainers_own_variables_from_a_task_prolog_reach_neither_apptainer_nor_the_container(
        tmp_path, apptainer_on_path, monkeypatch, owner):
    # A site's TaskProlog exports variables into every step after the launcher built its
    # environment. APPTAINER_BIND would add a bind past --no-mount bind-paths, out of sight of
    # the mount checks, and APPTAINERENV_X would set X inside the container.
    hidden = tmp_path / "shared"
    hidden.mkdir()
    prolog = {"APPTAINER_BIND": str(hidden), "APPTAINER_BINDPATH": str(hidden), "SINGULARITY_BIND": str(hidden),
              "APPTAINER_CACHEDIR": str(tmp_path / "cache"), "APPTAINERENV_LEAKED": "from-the-prolog",
              "SINGULARITYENV_ALSO_LEAKED": "from-the-prolog"}
    record = tmp_path / "apptainer.json"
    monkeypatch.setenv("FAKE_APPTAINER_RECORD", str(record))
    for name in [k for k in os.environ if dropped(k, container=True)]:
        monkeypatch.delenv(name)
    _, ctx = _ctx(tmp_path, RecordingLauncher(task_prolog=prolog))
    out = ctx.role_dir("srv") / "out"
    out.mkdir()
    spec = _quick(ctx, command=tuple(_dump_env(out / "env.json")), mounts=(Mount(out, "/out", read_only=False),))
    ctx.start_container("srv", spec, SHAPE, env={"PLAIN": "1"}, owner=owner)
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0, _err(ctx, "srv")
    seen = fake_apptainer.read_record(record)
    assert seen["runtime_env_names"] == [] and seen["binds"] == [*(f"{f}:{f}:ro" for f in HOST_FILES), f"{out}:/out"]
    assert json.loads((out / "env.json").read_text()) == {"PLAIN": "1"}
    [started] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_started"]
    assert started["stripped_env"] == sorted(prolog)


@pytest.mark.parametrize("where, value", [
    (where, value) for where in ("env", "command", "mount")
    for value in (SECRETS["HF_TOKEN"], f"Bearer {SECRETS['HF_TOKEN']}", RELAY)
    # A mount path with a space never reaches the launcher: the runtime refuses it.
    if not (where == "mount" and " " in value)])
def test_a_stage_grant_never_reaches_a_container(tmp_path, value, where):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS, scrubber=Scrubber({"relay": RELAY}))
    env = {"HF_TOKEN": value} if where == "env" else {}
    command = ("echo", value) if where == "command" else ("true",)
    mounts = (Mount(ctx.role_dir("srv") / value, "/data"),) if where == "mount" else ()
    spec = _quick(ctx, command=command, mounts=mounts)
    strings = container_strings(spec)
    named = "env HF_TOKEN" if where == "env" else f"spec[{next(i for i, a in enumerate(strings) if value in a)}]"
    message = f"harness container srv: {named} holds a secret; the secrets granted to a stage are host-only"
    with pytest.raises(ValueError, match=re.escape(message)) as caught:
        ctx.start_container("srv", spec, SHAPE, env=env, owner="harness")
    assert value not in str(caught.value)
    assert launcher.launched == [] and not (ctx.role_dir("srv") / "container.env").exists()


#: A route's base URL, a secret of the file, and its host.
BASE_URL = "https://relay.example/v1/anthropic"


def test_a_container_argument_naming_a_url_secrets_host_alone_is_not_refused(tmp_path, apptainer_on_path):
    """The host of a base URL is scrubbed from records but gives no access: a seal's policy
    argument names the route's host, and the container starts."""
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS, scrubber=Scrubber({"relay_url": BASE_URL}))
    spec = _quick(ctx, command=("echo", '{"domains": {"mode": "allow", "entries": ["relay.example"]}}'))
    ctx.start_container("srv", spec, SHAPE, owner="harness")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and [r.name for r in launcher.launched][-1] == "srv"


@pytest.mark.parametrize("where", ["env", "command"])
def test_a_container_argument_holding_the_full_base_url_is_refused(tmp_path, where):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS, scrubber=Scrubber({"relay_url": BASE_URL}))
    env = {"BASE": BASE_URL} if where == "env" else {}
    spec = _quick(ctx, command=("curl", f"{BASE_URL}/messages") if where == "command" else ("true",))
    strings = container_strings(spec)
    named = "env BASE" if where == "env" else f"spec[{next(i for i, a in enumerate(strings) if BASE_URL in a)}]"
    with pytest.raises(ValueError, match=re.escape(f"harness container srv: {named} holds a secret")) as caught:
        ctx.start_container("srv", spec, SHAPE, env=env, owner="harness")
    assert BASE_URL not in str(caught.value) and "relay.example" not in str(caught.value)
    assert launcher.launched == []


def test_a_host_roles_logs_are_scrubbed_once_it_ends_also_across_a_chunk_boundary(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher(), secrets=SECRETS)
    # The value straddles the scrubber's chunk boundary in both files.
    code = (f"import os, sys; head = 'x' * ({SCRUB_CHUNK_BYTES} - 5); value = os.environ['HF_TOKEN']; "
            "sys.stdout.write(head + value + '\\n'); sys.stderr.write(head + value + '\\n')")
    ctx.start_role("dl", _python(code), SHAPE, secret_env=["HF_TOKEN"], owner="harness")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0
    for suffix in (".out", ".err"):
        data = (ctx.attempt_dir / "logs" / f"dl{suffix}").read_bytes()
        assert data == b"x" * (SCRUB_CHUNK_BYTES - 5) + b"<redacted:HF_TOKEN>\n"


def _store_files_holding(root: Path, value: str) -> list[Path]:
    """Every file under *root* whose bytes hold *value*."""
    return [p for p in root.rglob("*") if p.is_file() and not p.is_symlink() and value.encode() in p.read_bytes()]


def test_a_role_given_secrets_logs_outside_the_store_until_it_is_reported(tmp_path):
    # A runner SIGKILLed between the role's end and its report never reports it; the store must
    # then hold nothing the role printed unscrubbed.
    plan, ctx = _ctx(tmp_path, RecordingLauncher(), secrets=SECRETS)
    token = SECRETS["HF_TOKEN"]
    code = "import os, sys; v = os.environ['HF_TOKEN']; print('out', v); print('err', v, file=sys.stderr)"
    handle = ctx.start_role("dl", _python(code), SHAPE, secret_env=["HF_TOKEN"], owner="harness")
    assert handle.wait(30) == 0
    local = ctx.role_dir("dl") / "logs"
    assert token in (local / "dl.out").read_text() and stat.S_IMODE(ctx.role_dir("dl").stat().st_mode) == 0o700
    assert _store_files_holding(plan.store_root, token) == []
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0
    logs = ctx.attempt_dir / "logs"
    assert (logs / "dl.out").read_text() == "out <redacted:HF_TOKEN>\n"
    assert (logs / "dl.err").read_text() == "err <redacted:HF_TOKEN>\n"
    assert (logs / "dl.report.json").exists() and _store_files_holding(plan.store_root, token) == []


def test_a_role_given_secrets_with_scrub_logs_logs_into_the_store_and_is_scrubbed_there(tmp_path):
    """A role that starts after a contestant may have filled the node's disk logs where every
    other role does, its report among its logs, so its launch needs no new space on that disk;
    its logs are scrubbed in place when it is reported."""
    plan, ctx = _ctx(tmp_path, RecordingLauncher(), secrets=SECRETS)
    token = SECRETS["HF_TOKEN"]
    code = "import os, sys; v = os.environ['HF_TOKEN']; print('out', v); print('err', v, file=sys.stderr)"
    handle = ctx.start_role("dl", _python(code), SHAPE, secret_env=["HF_TOKEN"], owner="harness", scrub_logs=True)
    assert handle.wait(30) == 0
    assert handle.spec.log_dir == ctx.attempt_dir / "logs"
    assert not (ctx.role_dir("dl") / "logs").exists()
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0
    logs = ctx.attempt_dir / "logs"
    assert (logs / "dl.out").read_text() == "out <redacted:HF_TOKEN>\n"
    assert (logs / "dl.err").read_text() == "err <redacted:HF_TOKEN>\n"
    assert (logs / "dl.report.json").exists() and _store_files_holding(plan.store_root, token) == []


def test_a_role_directory_holds_the_log_cappers_error_files_before_its_role_launches(tmp_path):
    """The launcher's log cappers append their own errors to files in the role's directory; they
    are made with it, so a role made ready before a contestant starts opens them at launch with
    no new space on the node's disk."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    role = ctx.role_dir("dl")
    for suffix in (".out", ".err"):
        errors = log_cap.errors_file(role, suffix)
        assert errors.parent == role and errors.is_file() and errors.stat().st_size == 0


def test_a_role_without_secrets_logs_straight_into_the_store(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher(), secrets=SECRETS)
    handle = ctx.start_role("plain", _python("print('hello')"), SHAPE, owner="harness")
    assert handle.wait(30) == 0
    assert (ctx.attempt_dir / "logs" / "plain.out").read_text() == "hello\n"
    ctx.wait_roles()


class _StepGoneWhileScrubbing(Scrubber):
    """A scrubber whose first scrub of a log removes the step cgroup *step*, as a step's cgroup
    goes while a long log is scrubbed."""

    def __init__(self, values: dict[str, str], step: Path) -> None:
        super().__init__(values)
        self.step = step

    def scrub_file(self, path: Path, **kwargs) -> None:
        if self.step.exists():
            shutil.rmtree(self.step)
        super().scrub_file(path, **kwargs)


def test_the_jobs_oom_counts_are_read_before_a_roles_logs_are_scrubbed(tmp_path):
    # The step's own count says its own limit was not reached. Read after the scrub, with the step
    # gone, the rise below the job would be put down to this role instead.
    job = _job(tmp_path)
    watch = MemoryWatch(enforce=False, job_dir=job)
    _, ctx = _ctx(tmp_path, _OomLauncher(job), watch=watch, secrets=SECRETS,
                  scrubber=_StepGoneWhileScrubbing(SECRETS, job / "step_1"))
    handle = ctx.start_role("dl", _python("import sys; print('x'); sys.exit(137)"), SHAPE,
                            secret_env=["HF_TOKEN"], owner="harness")
    assert handle.wait(30) == 137
    (job / "step_1" / "memory.events").write_text("oom 0\noom_kill 1\n")
    (job / "memory.events").write_text("oom 1\noom_kill 0\n")
    [outcome] = ctx.wait_roles()
    assert not (job / "step_1").exists() and outcome.own_oom is False


def test_a_role_directory_deeper_than_the_interpreter_recurses_is_removed_at_close(tmp_path):
    # A contestant writes what it likes into its own directory; closing the stage still removes it.
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    p = ctx.role_dir("srv")
    for _ in range(sys.getrecursionlimit() + 100):
        p = p / "a"
        p.mkdir()
    ctx.close()
    assert list((tmp_path / "node" / "hls-100" / "roles").iterdir()) == []


def _duplicate(ctx: StageContext) -> dict:
    ctx.start_role("srv", _python("pass"), SHAPE, owner="harness")
    return {}


def _stopping(ctx: StageContext) -> dict:
    ctx.stop_roles()
    return {}


def _over_budget(ctx: StageContext) -> dict:
    return {"shape": ResourceShape(cpus=64, memory_gib=2, time_sec=60)}


def _bad_share_target(ctx: StageContext) -> dict:
    gpu = ResourceShape(gpus=1, gpu_kind="H200", cpus=1, memory_gib=1, time_sec=60)
    target = ctx.start_role("t", _python("pass"), gpu, owner="harness")
    return {"shape": gpu, "share_gpus_of": target}


@pytest.mark.parametrize("setup, error, match", [
    (_duplicate, ValueError, "role srv already started"),
    (_stopping, StageFault, "the stage is stopping"),
    (_over_budget, StageFault, "the roles need 64 CPUs"),
    (_bad_share_target, StageFault, "cannot share the GPUs of t"),
], ids=["duplicate", "stopping", "over_budget", "bad_share_target"])
def test_a_container_is_admitted_like_any_role(tmp_path, setup, error, match):
    launcher = RecordingLauncher(mismatch=("t",))
    _, ctx = _ctx(tmp_path, launcher)
    kw = {"shape": SHAPE, "owner": "harness", **setup(ctx)}
    started = len(launcher.launched)
    with pytest.raises(error, match=match):
        ctx.start_container("srv", _quick(ctx), kw.pop("shape"), **kw)
    assert len(launcher.launched) == started and not (ctx.role_dir("srv") / "container.env").exists()


def test_an_off_thread_start_in_flight_refuses_every_start_pin_hold_and_close(tmp_path, apptainer_on_path):
    """``start_off_thread`` starts a role's step in a worker thread; while it is in flight the
    stage's thread may start no other role, pin no image, copy, seed or publish no cache entry,
    release no image, hold no deadline and close nothing, so the two threads never admit against
    the same resources or grow the deadline at once. Each is refused by its own guard, named by
    what was asked, before it touches anything. The worker's own start_container is let through,
    and once the stage collects the future it may do all of those again."""
    launcher = BlockingLauncher("snap")
    _, ctx = _ctx(tmp_path, launcher)
    image = _pinned(ctx)
    snap = ContainerSpec(image=image, command=("true",), workdir=ctx.role_dir("snap") / "wd")
    other = ContainerSpec(image=image, command=("true",), workdir=ctx.role_dir("other") / "wd")
    entry = _entry_of(ctx, "weights", "org+m@" + "a" * 40, {"w.bin": b"w"})
    # Copy and pin once before the window, so the calls in it take the memoized fast path and only
    # the pin's and the copy's own guards, not a role start's, refuse them.
    ctx.materialize("weights", entry.key, entry_sha256(entry))
    future = ctx.start_off_thread("snap", lambda: ctx.start_container("snap", snap, SHAPE, owner="harness",
                                                                      overlap=True))
    assert launcher.reached.wait(10), "the off-thread start never reached the launcher"
    launched = len(launcher.launched)
    off = "off the stage's thread"
    with pytest.raises(RuntimeError, match=off):
        ctx.start_role("other", ("true",), SHAPE, owner="harness")
    with pytest.raises(RuntimeError, match=off):
        ctx.start_container("other", other, SHAPE, owner="harness")
    with pytest.raises(RuntimeError, match=off):
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    with pytest.raises(RuntimeError, match=off):
        ctx.pin_file(image_file(ctx.scratch), sha256=IMAGE_SHA256)
    with pytest.raises(RuntimeError, match=off):
        ctx.node_image("images/x", IMAGE_SHA256)
    seeded = ctx.role_dir("other") / "seeded"
    with pytest.raises(RuntimeError, match=f"cache entry weights/{re.escape(entry.key)} is seeded while .*{off}"):
        ctx.seed("weights", entry.key, seeded, expect=entry_sha256(entry))
    built = ctx.role_dir("other") / "built"
    built.mkdir(parents=True)
    with pytest.raises(RuntimeError, match=f"cache entry engine-cache/k is published while .*{off}"):
        ctx.publish("engine-cache", "k", built)
    with pytest.raises(RuntimeError, match=f"image {IMAGE_SHA256} is released while .*{off}"):
        ctx.release(image)
    assert not seeded.exists() and ctx.cache.lookup("engine-cache", "k") is None and Path(image.ref).is_dir()
    with pytest.raises(RuntimeError, match=off):
        ctx.hold_until(ctx.clock() + 10_000)
    with pytest.raises(RuntimeError, match=off):
        ctx.close()
    # None of the refused calls reached the launcher; the snapshot's own start is still blocked.
    assert len(launcher.launched) == launched
    launcher.release.set()
    handle = future.result(30)
    assert handle.spec.name == "snap" and "snap" in ctx.running_roles()
    # Collected: the stage's thread may hold its deadline again and wait for the role.
    assert ctx.hold_until(ctx.clock() + 10_000) is True
    assert [o.ending for o in ctx.wait_roles()] == ["exited"]


@pytest.mark.parametrize("where", ["store", "attempt", "logs", "checkout", "outside", "workdir", "scratch",
                                   "roles", "other_role", "job_copies"])
def test_a_contestant_container_sees_only_its_own_role_directory(tmp_path, where):
    launcher = RecordingLauncher()
    plan, ctx = _ctx(tmp_path, launcher)
    own = ctx.role_dir("srv")
    _entry(ctx)
    source = {"store": plan.store_root / "cache" / "x", "attempt": ctx.attempt_dir,
              "logs": ctx.attempt_dir / "logs", "checkout": plan.code.checkout / "tasks",
              "outside": tmp_path / "elsewhere", "workdir": own / "data", "scratch": tmp_path / "node" / "hls-100",
              "roles": tmp_path / "node" / "hls-100" / "roles", "other_role": ctx.role_dir("dl"),
              "job_copies": tmp_path / "node" / "hls-100" / "100"}[where]
    workdir = plan.store_root / "wd" if where == "workdir" else own / "wd"
    spec = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=workdir,
                         mounts=(Mount(source, "/data"),))
    with pytest.raises(ValueError, match="contestant container srv: .* not in the role's own directory"):
        ctx.start_container("srv", spec, SHAPE, owner="contestant")
    assert _own_roles(launcher) == []


def test_a_contestant_container_may_mount_paths_in_its_own_role_directory(tmp_path, apptainer_on_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    data = ctx.role_dir("srv") / "data"
    data.mkdir()
    ctx.start_container("srv", _quick(ctx, mounts=(Mount(data, "/data"), Mount(ctx.role_dir("srv"), "/own"))),
                        SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0, _err(ctx, "srv")


@pytest.mark.parametrize("part", ["entry", "file"])
def test_a_contestant_container_may_mount_a_cache_entry_the_stage_copied_read_only(tmp_path, apptainer_on_path,
                                                                                   part):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    weights = _entry(ctx)
    source = weights if part == "entry" else weights / "w.bin"
    ctx.start_container("srv", _quick(ctx, mounts=(Mount(source, "/data/weights"),)), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0, _err(ctx, "srv")


def test_a_cache_copy_under_a_linked_node_scratch_may_still_be_mounted_read_only(tmp_path, apptainer_on_path):
    (tmp_path / "real-node").mkdir()
    (tmp_path / "node-link").symlink_to(tmp_path / "real-node")
    ctx = _context(noop_plan(tmp_path), RecordingLauncher(), tmp_path / "node-link" / "hls-100")
    weights = _entry(ctx)
    assert weights.is_relative_to(tmp_path / "node-link")
    ctx.start_container("srv", _quick(ctx, mounts=(Mount(weights, "/data/weights"),)), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0, _err(ctx, "srv")


@pytest.mark.parametrize("owner", ["harness", "contestant"])
@pytest.mark.parametrize("part", ["entry", "file"])
def test_no_container_mounts_a_cache_entry_copy_writable(tmp_path, owner, part):
    # A write into a copy would reach a later stage of the job, which reuses the copy.
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    weights = _entry(ctx)
    source = weights if part == "entry" else weights / "w.bin"
    message = f"mount /data/weights source {source} is in a cache entry copy, which no container may write"
    with pytest.raises(ValueError, match=re.escape(message)):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(source, "/data/weights", read_only=False),)), SHAPE,
                            owner=owner)
    assert _own_roles(launcher) == []


def test_no_container_mounts_writable_a_directory_that_holds_the_copies(tmp_path):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    _entry(ctx)
    message = (f"mount /scratch source {ctx.scratch} holds the cache entry copies, which no container may "
               "write")
    with pytest.raises(ValueError, match=re.escape(message)):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(ctx.scratch, "/scratch", read_only=False),)), SHAPE,
                            owner="harness")
    assert _own_roles(launcher) == []


@pytest.mark.parametrize("owner", ["harness", "contestant"])
def test_a_link_to_a_cache_entry_copy_is_judged_by_where_it_leads_before_it_is_mounted_writable(tmp_path, owner):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    link = ctx.role_dir("srv") / "weights"
    link.symlink_to(_entry(ctx))
    with pytest.raises(ValueError, match=re.escape(f"mount /data/weights source {link} is in a cache entry copy")):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(link, "/data/weights", read_only=False),)), SHAPE,
                            owner=owner)
    assert _own_roles(launcher) == []


def test_a_copy_under_a_linked_node_scratch_is_never_mounted_writable(tmp_path):
    (tmp_path / "real-node").mkdir()
    (tmp_path / "node-link").symlink_to(tmp_path / "real-node")
    launcher = RecordingLauncher()
    ctx = _context(noop_plan(tmp_path), launcher, tmp_path / "node-link" / "hls-100")
    weights = _entry(ctx)
    with pytest.raises(ValueError, match=re.escape(f"{weights} is in a cache entry copy")):
        ctx.start_container("drv", _quick(ctx, "drv", mounts=(Mount(weights, "/data/weights", read_only=False),)),
                            SHAPE, owner="harness")
    assert _own_roles(launcher) == []


def test_a_harness_containers_workdir_is_never_in_a_cache_entry_copy(tmp_path):
    # The workdir backs the container's /tmp, /var/tmp and $HOME, so it is written.
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    weights = _entry(ctx)
    spec = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=weights / "wd")
    with pytest.raises(ValueError, match=re.escape(f"workdir {weights / 'wd'} is in a cache entry copy")):
        ctx.start_container("drv", spec, SHAPE, owner="harness")
    assert _own_roles(launcher) == []


def test_a_harness_container_may_not_write_a_copy_another_stage_made(tmp_path):
    plan = noop_plan(tmp_path)
    scratch = tmp_path / "node" / "hls-100"
    weights = _entry(_context(plan, RecordingLauncher(), scratch))
    launcher = RecordingLauncher()
    ctx = _context(plan, launcher, scratch)
    with pytest.raises(ValueError, match=re.escape(f"{weights} is in a cache entry copy")):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(weights, "/data/weights", read_only=False),)), SHAPE,
                            owner="harness")
    assert _own_roles(launcher) == []


def test_a_contestant_containers_workdir_is_never_in_a_cache_entry_copy(tmp_path):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    weights = _entry(ctx)
    spec = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=weights / "wd")
    with pytest.raises(ValueError, match=re.escape(f"workdir {weights / 'wd'} is not in the role's own directory")):
        ctx.start_container("srv", spec, SHAPE, owner="contestant")
    assert _own_roles(launcher) == []


def test_a_contestant_container_may_not_mount_a_copy_another_stage_made(tmp_path):
    launcher = RecordingLauncher()
    plan, earlier = _ctx(tmp_path, launcher)
    weights = _entry(earlier)
    ctx = _context(plan, launcher, tmp_path / "node" / "hls-100")
    with pytest.raises(ValueError, match="not in the role's own directory"):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(weights, "/data/weights"),)), SHAPE, owner="contestant")
    assert _own_roles(launcher) == []


ROWS_KEY = "rows-0123456789abcdef"
# Two tenants, one of them named "dev", so a rule that looks for "dev" anywhere in the path is
# caught out by tenants/dev/test.
ROWS_FILES = {"ids.json": b'{"dev": ["1"], "test": ["2"]}', "tenants/t0/dev/rows.json": b"[1]",
              "tenants/t0/dev/prompt_prefix.txt": b"P", "tenants/t0/test/rows.json": b"[2]",
              "tenants/dev/dev/rows.json": b"[3]", "tenants/dev/test/rows.json": b"[4]"}
FULL_KEY, DEV_KEY = "tiny-v1-full", "tiny-v1-dev-0123456789ab"
ROWS_REFUSAL = "where a contestant may see only tenants/<id>/dev"


@pytest.mark.parametrize("rel", ["", "ids.json", "tenants", "tenants/t0", "tenants/t0/test",
                                 "tenants/t0/test/rows.json", "tenants/dev", "tenants/dev/test",
                                 "tenants/dev/test/rows.json"])
def test_a_contestant_container_never_mounts_the_test_rows_or_the_row_ids(tmp_path, rel):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    rows = _copy(ctx, "rows", ROWS_KEY, ROWS_FILES)
    source = rows / rel if rel else rows
    with pytest.raises(ValueError, match=re.escape(f"mount /data/x source {source} is in rows entry {ROWS_KEY}, "
                                                   + ROWS_REFUSAL)):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(source, "/data/x"),)), SHAPE, owner="contestant")
    assert _own_roles(launcher) == []


def test_a_link_inside_a_copy_is_judged_by_where_it_leads(tmp_path):
    """A mount source is judged after following links, so a link under a tenant's dev rows that
    leads to its test rows is refused like the test rows themselves."""
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    rows = _copy(ctx, "rows", ROWS_KEY, ROWS_FILES)
    link = rows / "tenants" / "t0" / "dev" / "more"
    link.symlink_to(rows / "tenants" / "t0" / "test")
    with pytest.raises(ValueError, match=re.escape(f"mount /data/x source {link} is in rows entry {ROWS_KEY}, "
                                                   + ROWS_REFUSAL)):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(link, "/data/x"),)), SHAPE, owner="contestant")
    assert _own_roles(launcher) == []


@pytest.mark.parametrize("key", [FULL_KEY, "corpus-dev-search-v1-full"])
def test_a_contestant_container_never_mounts_a_full_tool_bundle(tmp_path, key):
    """Also when the bundle's name holds "-dev-": the key's suffix says which variant it is."""
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    full = _copy(ctx, "tools", key, {"manifest.json": b"{}"})
    for source in (full, full / "manifest.json"):
        with pytest.raises(ValueError, match=re.escape(f"is in tools entry {key}, which is not a dev bundle")):
            ctx.start_container("srv", _quick(ctx, mounts=(Mount(source, "/data/tools/tiny"),)), SHAPE,
                                owner="contestant")
    assert _own_roles(launcher) == []


def test_every_role_says_whose_it_is(tmp_path):
    """A role started without its owner would default to one side of the wall, so there is no
    default: both ways of starting a role refuse a call that does not name the owner."""
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    with pytest.raises(TypeError, match="owner"):
        ctx.start_role("dl", _python("pass"), SHAPE)
    with pytest.raises(TypeError, match="owner"):
        ctx.start_container("srv", _quick(ctx), SHAPE)
    assert _own_roles(launcher) == []


def test_a_contestant_container_never_mounts_an_entry_of_a_kind_the_wall_does_not_know(tmp_path):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    other = _copy(ctx, "misc", "img-1", {"a.sif": b"x"})
    with pytest.raises(ValueError, match=re.escape("is in misc entry img-1, a kind no contestant may mount")):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(other, "/data/x"),)), SHAPE, owner="contestant")
    assert _own_roles(launcher) == []


def test_a_contestant_container_may_mount_the_dev_rows_and_the_dev_bundle(tmp_path, apptainer_on_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    rows = _copy(ctx, "rows", ROWS_KEY, ROWS_FILES)
    dev = _copy(ctx, "tools", DEV_KEY, {"manifest.json": b"{}"})
    mounts = (Mount(rows / "tenants" / "t0" / "dev", "/data/tenants/t0/dev"),
              Mount(rows / "tenants" / "dev" / "dev", "/data/tenants/dev/dev"),
              Mount(rows / "tenants" / "t0" / "dev" / "rows.json", "/data/one.json"), Mount(dev, "/data/tools/tiny"),
              Mount(_entry(ctx), "/data/weights"))
    ctx.start_container("srv", _quick(ctx, mounts=mounts), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0, _err(ctx, "srv")


def test_a_harness_container_may_mount_a_whole_rows_entry_and_a_full_bundle(tmp_path, apptainer_on_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    rows = _copy(ctx, "rows", ROWS_KEY, ROWS_FILES)
    full = _copy(ctx, "tools", FULL_KEY, {"manifest.json": b"{}"})
    ctx.start_container("drv", _quick(ctx, "drv", mounts=(Mount(rows, "/data/rows"), Mount(full, "/data/tools/t"))),
                        SHAPE, owner="harness")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0, _err(ctx, "drv")


@pytest.mark.parametrize("target, label", [("store", "the store"), ("logs", "the log directory")])
def test_a_mount_source_that_links_to_the_store_or_the_logs_is_refused(tmp_path, target, label):
    launcher = RecordingLauncher()
    plan, ctx = _ctx(tmp_path, launcher)
    logs = ctx.attempt_dir / "logs"
    logs.mkdir(parents=True)
    link = ctx.role_dir("srv") / "data"
    link.symlink_to(plan.store_root if target == "store" else logs)
    with pytest.raises(ValueError, match=re.escape(f"mount /data source {link} is in {label}")):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(link, "/data"),)), SHAPE, owner="contestant")
    assert _own_roles(launcher) == []


@pytest.mark.parametrize("linked, label", [("attempt", "the attempt directory"), ("logs", "the log directory")])
def test_a_contestant_never_sees_the_attempt_or_log_directory_wherever_a_link_puts_it(tmp_path, linked, label):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    own = ctx.role_dir("srv")
    (own / "linked").mkdir()
    place = ctx.attempt_dir if linked == "attempt" else ctx.attempt_dir / "logs"
    place.parent.mkdir(parents=True, exist_ok=True)
    place.symlink_to(own / "linked")
    with pytest.raises(ValueError, match=re.escape(f"mount /own source {own} holds {label}")):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(own, "/own"),)), SHAPE, owner="contestant")
    assert _own_roles(launcher) == []


@pytest.mark.parametrize("inside", ["store", "checkout"])
def test_a_scratch_inside_the_store_or_checkout_still_keeps_them_from_a_contestant(tmp_path, inside):
    launcher = RecordingLauncher()
    plan = noop_plan(tmp_path)
    checkout = tmp_path / "co"
    plan = plan.model_copy(update={"code": plan.code.model_copy(update={"checkout": checkout})})
    scratch = (plan.store_root if inside == "store" else checkout) / "node" / "hls-100"
    ctx = _context(plan, launcher, scratch)
    spec = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=ctx.role_dir("srv") / "wd")
    with pytest.raises(ValueError, match=f"contestant .* the {inside}"):
        ctx.start_container("srv", spec, SHAPE, owner="contestant")
    assert _own_roles(launcher) == []


@pytest.mark.parametrize("owner", ["harness", "contestant"])
@pytest.mark.parametrize("where, holds", [("root", "the store"), ("store_parent", "the store"),
                                          ("checkout_parent", "the checkout")])
def test_no_container_mounts_a_directory_that_holds_the_store_or_the_checkout(tmp_path, owner, where, holds):
    launcher = RecordingLauncher()
    plan, ctx = _ctx(tmp_path, launcher)
    source = {"root": Path("/"), "store_parent": plan.store_root.parent,
              "checkout_parent": plan.code.checkout.parent}[where]
    message = f"{owner} container srv: mount /data source {source} holds {holds}"
    with pytest.raises(ValueError, match=re.escape(message)):
        ctx.start_container("srv", _quick(ctx, mounts=(Mount(source, "/data"),)), SHAPE, owner=owner)
    assert _own_roles(launcher) == []


def test_a_harness_container_may_mount_the_store(tmp_path, apptainer_on_path):
    launcher = RecordingLauncher()
    plan, ctx = _ctx(tmp_path, launcher)
    plan.store_root.mkdir(parents=True, exist_ok=True)
    spec = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=ctx.role_dir("ref") / "wd",
                         mounts=(Mount(plan.store_root, "/store"),))
    ctx.start_container("ref", spec, SHAPE, owner="harness")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0


def test_a_runtime_failure_before_the_marker_is_infra_even_for_a_contestant(tmp_path, apptainer_on_path,
                                                                            monkeypatch):
    monkeypatch.setenv("FAKE_APPTAINER_FAIL", "1")
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    spec = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "runtime_failed" and outcome.exit_code == 255
    assert outcome.detail.startswith("FATAL")
    fault = classify_role(outcome)
    assert fault is not None and fault.fault_class == "infra" and fault.retryable


def test_a_contestant_command_that_fails_after_the_marker_is_its_result(tmp_path, apptainer_on_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    spec = ContainerSpec(image=_pinned(ctx), command=tuple(_python("import sys; sys.exit(1)")),
                         workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 1
    assert classify_role(outcome) is None


def test_a_runtime_that_hangs_before_the_marker_is_runtime_failed_not_time_limit(tmp_path, apptainer_on_path):
    _apptainer_script(apptainer_on_path, "exec sleep 30")
    _, ctx = _ctx(tmp_path, RecordingLauncher(), deadline_in=1.0)
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.detail) == ("runtime_failed", "no start marker on stderr")


def test_a_containers_command_limit_counts_from_when_its_start_marker_was_seen(tmp_path, apptainer_on_path):
    """A runtime that takes longer to start the container than the command may run costs the
    command none of its time: the command is stopped only once it has run that long, and ends
    ``time_limit``, its own end, never the runtime's failure."""
    gate = tmp_path / "runtime-starts"
    _apptainer_script(apptainer_on_path, f"while [ ! -e {gate} ]; do sleep 0.05; done\n"
                                         f"exec {sys.executable} {fake_apptainer.__file__} \"$@\"")
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    first = tmp_path / "command-started"
    ctx.start_container("srv", _quick(ctx, command=_first_then_sleep(first)), SHAPE, owner="contestant")
    # Held twice as long as the command may run, counted from the step's start.
    release_at = ctx.clock() + 1.0

    def release() -> None:
        while ctx.clock() < release_at:
            time.sleep(0.05)
        gate.touch()

    releaser = threading.Thread(target=release)
    releaser.start()
    [outcome] = ctx.wait_roles("srv", command_sec=0.5)
    stopped = ctx.clock()
    releaser.join()
    began = ctx.command_started_at("srv")
    assert outcome.ending == "time_limit", outcome
    assert began is not None and release_at <= began <= float(first.read_text()) + SEEN_WITHIN_SEC
    assert stopped >= began + 0.5


def test_a_container_whose_command_never_starts_is_held_to_the_deadline_alone(tmp_path, apptainer_on_path):
    """No limit of a command's own stops a container before its command started, so a runtime
    that hangs there is stopped at the stage's deadline and ends ``runtime_failed``."""
    _apptainer_script(apptainer_on_path, "exec sleep 30")
    _, ctx = _ctx(tmp_path, RecordingLauncher(), deadline_in=1.5)
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles("srv", command_sec=0.1)
    assert ctx.clock() >= ctx.deadline
    assert (outcome.ending, outcome.detail) == ("runtime_failed", "no start marker on stderr")
    assert ctx.command_started_at("srv") is None


@pytest.mark.parametrize("runtime", ["held", "never-starts", "fails"])
def test_a_wait_for_a_command_to_start_ends_at_its_marker_its_end_or_the_deadline(tmp_path, apptainer_on_path,
                                                                                   runtime):
    """It returns when the marker was seen, however long the runtime took, within a pass of the
    command's own start, and None once the role ended without one or the deadline passed: only
    the deadline bounds a runtime's start. The held runtime is released by the stage's own look at
    the clock past the hold, once a look found no marker, so the next look is a pass later."""
    gate, first = tmp_path / "runtime-starts", tmp_path / "command-started"
    body = {"held": f"while [ ! -e {gate} ]; do sleep 0.05; done\n"
                    f"exec {sys.executable} {fake_apptainer.__file__} \"$@\"",
            "never-starts": "exec sleep 30", "fails": "exit 255"}[runtime]
    _apptainer_script(apptainer_on_path, body)
    _, ctx = _ctx(tmp_path, RecordingLauncher(), deadline_in=1.5 if runtime == "never-starts" else 60.0)
    ctx.start_container("srv", _quick(ctx, command=_first_then_sleep(first)), SHAPE, owner="contestant")
    real = ctx.clock
    release_at = real() + 0.5

    def clock() -> float:
        now = real()
        if runtime == "held" and now >= release_at and not gate.exists():
            gate.touch()
            give_up = real() + 60
            while not ctx.command_started("srv"):
                assert real() < give_up, "the command never started"
                time.sleep(0.01)
        return now

    ctx.clock = clock
    began = ctx.wait_command_started("srv")
    if runtime == "held":
        assert began is not None and began == ctx.command_started_at("srv")
        assert release_at <= began <= float(first.read_text()) + SEEN_WITHIN_SEC
    else:
        assert began is None and (ctx.clock() >= ctx.deadline) is (runtime == "never-starts")
        # A runtime that never starts is still running when the wait gives up at the deadline.
        assert ctx.running_roles() == (["srv"] if runtime == "never-starts" else [])
    ctx.stop_role("srv")


def test_a_command_no_look_found_started_before_its_role_was_reported_has_no_start(tmp_path, apptainer_on_path):
    """A start first seen once the role was reported would come after its end: there is none."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    handle = ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    assert handle.wait(30) == 0
    [outcome] = ctx.wait_roles("srv", command_sec=60)
    assert (outcome.ending, outcome.exit_code) == ("exited", 0) and ctx.command_started("srv")
    assert ctx.command_started_at("srv") is None


def test_a_host_roles_command_starts_with_its_step(tmp_path):
    """A role with no start marker runs its command from its step's start."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    ctx.start_role("dl", _python("pass"), SHAPE, owner="harness")
    assert ctx.command_started_at("dl") == ctx.role_started_at("dl") is not None


@pytest.mark.parametrize("fails, ending", [(True, "runtime_failed"), (False, "memory_limit")])
def test_a_memory_breach_counts_only_once_the_command_started(tmp_path, apptainer_on_path, monkeypatch, fails,
                                                              ending):
    if fails:
        monkeypatch.setenv("FAKE_APPTAINER_FAIL", "1")
    _, ctx = _ctx(tmp_path, RecordingLauncher(), watch=_BreachedWatch())
    ctx.start_container("srv", _quick(ctx, command=tuple(_python("import sys; sys.exit(1)"))), SHAPE,
                        owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == ending


class _OomLauncher(LocalLauncher):
    """A local launcher whose role steps end OUT_OF_MEMORY in accounting once they exit
    nonzero, with their commands left as they are, laid out as Slurm steps under *job*."""

    def __init__(self, job: Path) -> None:
        super().__init__(gpus=2, steps_under=job)

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        handle = super().launch(spec, share_gpus_of=share_gpus_of)
        handle.step_state = JobState.OUT_OF_MEMORY
        return handle


def _job(tmp_path: Path) -> Path:
    """A fake ``job_<id>`` cgroup whose own limit and the limits below it were never reached."""
    job = tmp_path / "job_1"
    (job / "step_batch").mkdir(parents=True)
    for name in ("memory.events.local", "memory.events", "step_batch/memory.events"):
        (job / name).write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
    return job


@pytest.mark.parametrize("limit, fault_class, retryable", [("own", "contestant", False), ("job", "infra", True)])
def test_a_contestant_container_killed_at_its_own_limit_is_the_contestants_fault(tmp_path, apptainer_on_path, limit,
                                                                                fault_class, retryable):
    # The container runs inside its step, so the wrapper's report names the step's cgroup and the
    # step's own count says whose limit the kernel reached, as for a host role.
    job = _job(tmp_path)
    watch = MemoryWatch(enforce=False, job_dir=job)
    launcher = _OomLauncher(job)
    _, ctx = _ctx(tmp_path, launcher, watch=watch)
    ctx.start_container("srv", _quick(ctx, command=tuple(_python("import sys; sys.exit(137)"))), SHAPE,
                        owner="contestant")
    assert ctx.report("srv").cgroup_dir == str(job / "step_1" / "user")
    assert launcher.launched[0].owner == "contestant" and launcher.launched[0].start_marker == START_MARKER
    assert ctx._roles["srv"].wait(30) == 137
    (job / "step_1" / "memory.events").write_text(f"oom {int(limit == 'own')}\noom_kill 1\n")
    if limit == "job":
        (job / "memory.events.local").write_text("oom 1\noom_kill 0\n")
        (job / "memory.events").write_text("oom 1\noom_kill 0\n")
    else:
        (job / "memory.events").write_text("oom 1\noom_kill 0\n")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.oom_kills, outcome.own_oom, outcome.job_oom) == (
        "exited", 1, limit == "own", limit == "job")
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == (fault_class, retryable)


def test_a_runtime_that_exits_0_without_the_marker_is_an_exit(tmp_path, apptainer_on_path):
    _apptainer_script(apptainer_on_path, "exit 0")
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("exited", 0)


def test_a_container_whose_runtime_failed_during_a_shutdown_is_stopped(tmp_path, apptainer_on_path, monkeypatch):
    monkeypatch.setenv("FAKE_APPTAINER_FAIL", "1")
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    handle = ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    assert handle.wait(30) == 255
    ctx.stop_roles()
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "stopped"


def test_a_container_refused_at_its_grant_check_is_a_grant_mismatch(tmp_path, apptainer_on_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher(mismatch=("srv",)))
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("grant_mismatch", 97)


def test_the_runtimes_fatal_line_is_scrubbed_before_it_is_recorded(tmp_path, apptainer_on_path):
    token = SECRETS["HF_TOKEN"]
    _apptainer_script(apptainer_on_path, f"printf 'FATAL:   could not open image /images/%s.sif\\n' '{token}' >&2; "
                                         "exit 255")
    _, ctx = _ctx(tmp_path, RecordingLauncher(), scrubber=Scrubber({"hf_token": token}))
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "runtime_failed"
    assert outcome.detail == "FATAL:   could not open image /images/<redacted:hf_token>.sif"
    assert token not in (ctx.attempt_dir / "events.jsonl").read_text()


def test_the_marker_counts_only_as_a_whole_line(tmp_path, apptainer_on_path):
    _apptainer_script(apptainer_on_path, f"printf 'FATAL:   while starting {START_MARKER}: no image\\n' >&2; exit 255")
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.detail) == ("runtime_failed", f"FATAL:   while starting {START_MARKER}: no image")


@pytest.mark.parametrize("warning_bytes, ending", [(MARKER_SEARCH_BYTES - 1024, "exited"),
                                                   (MARKER_SEARCH_BYTES + 1024, "runtime_failed")],
                         ids=["inside_the_window", "past_the_window"])
def test_the_marker_is_searched_for_in_the_first_mib_of_stderr(tmp_path, apptainer_on_path, warning_bytes, ending):
    warnings = apptainer_on_path / "warnings.txt"
    line = b"WARNING: " + b"w" * 1014 + b"\n"
    warnings.write_bytes(line * (warning_bytes // len(line)))
    assert warnings.stat().st_size == warning_bytes
    _apptainer_script(apptainer_on_path, f"cat {warnings} >&2\nexec {sys.executable} {fake_apptainer.__file__} \"$@\"")
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    ctx.start_container("srv", _quick(ctx, command=tuple(_python("import sys; sys.exit(1)"))), SHAPE,
                        owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == (ending, 1)


def test_a_leftover_env_file_is_removed_once_the_roles_end(tmp_path):
    _, ctx = _ctx(tmp_path, SkippingLauncher(), secrets=SECRETS)
    ctx.start_role("dl", _python("pass"), SHAPE, secret_env=["HF_TOKEN"], owner="harness")
    assert (ctx.role_dir("dl") / "role.env").exists()
    ctx.wait_roles()
    assert not (ctx.role_dir("dl") / "role.env").exists()


def test_an_env_file_is_removed_when_the_launch_fails(tmp_path):
    _, ctx = _ctx(tmp_path, RaisingLauncher(), secrets=SECRETS)
    with pytest.raises(RuntimeError, match="srun is gone"):
        ctx.start_role("dl", _python("pass"), SHAPE, secret_env=["HF_TOKEN"], owner="harness")
    assert not (ctx.role_dir("dl") / "role.env").exists()


def test_materialize_copies_the_entry_in_a_harness_role_and_records_its_digest(tmp_path):
    plan, ctx = _ctx(tmp_path, RecordingLauncher())
    entry = _published(ctx, "org+m@" + "a" * 40)
    path = ctx.materialize("weights", entry.key, entry_sha256(entry))
    assert path == tmp_path / "node" / "hls-100" / "100" / "weights" / entry.key
    assert (path / "w.bin").read_bytes() == b"w"
    assert ctx.cache_entries == {f"weights/{entry.key}": entry_sha256(entry)}
    [role] = ctx.launcher.launched
    stage = plan.stage("noop_a").shape
    assert role.name == f"copy-weights-{entry_sha256(entry)[:12]}" and role.owner == "harness"
    # It runs alone: the stage's CPUs and its memory less the runner's, no GPU and no disk.
    assert role.shape == stage.model_copy(update={"gpus": 0, "gpu_kind": None, "disk_gib": 0,
                                                  "memory_gib": stage.memory_gib - 4})
    report = ctx.scratch / "roles" / role.name / "digests.json"
    assert role.argv[:4] == (sys.executable, "-m", "hls.store.cache_role", "copy")
    assert role.argv[4:] == ("--entry", str(entry.path), "--expect", entry_sha256(entry), "--dest",
                             role.argv[9], "--digests-out", str(report))
    assert Path(role.argv[9]).parent == path.parent and Path(role.argv[9]).name.startswith(".tmp-")


def test_a_stage_copies_an_entry_once_however_many_of_its_roles_mount_it(tmp_path, monkeypatch):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    entry = _published(ctx)
    asked = []
    real = ctx.cache.materialize
    monkeypatch.setattr(ctx.cache, "materialize", lambda *a, **kw: asked.append(a) or real(*a, **kw))
    first = ctx.materialize("weights", entry.key, entry_sha256(entry))
    assert ctx.materialize("weights", entry.key, entry_sha256(entry)) == first and len(asked) == 1
    assert len(ctx.launcher.launched) == 1
    # Another digest for the same key is never the copy this stage holds.
    with pytest.raises(StageFault, match="producer recorded"):
        ctx.materialize("weights", entry.key, "0" * 64)
    assert ctx.cache_entries == {f"weights/{entry.key}": entry_sha256(entry)}


def test_a_later_stage_of_the_job_reuses_the_copy_with_no_role_until_its_stamp_cannot_be_trusted(tmp_path):
    plan = noop_plan(tmp_path)
    scratch = tmp_path / "node" / "hls-100"
    first = _context(plan, RecordingLauncher(), scratch)
    entry = _published(first)
    path = first.materialize("weights", entry.key, entry_sha256(entry))
    first.close()
    later = _context(plan, RecordingLauncher(), scratch)
    assert later.materialize("weights", entry.key, entry_sha256(entry)) == path
    assert later.launcher.launched == []
    # A new file with the same bytes and times: the stamp cannot be trusted, the copy still matches.
    st = os.lstat(path / "w.bin")
    (path / "w.bin.twin").write_bytes((path / "w.bin").read_bytes())
    os.utime(path / "w.bin.twin", ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(path / "w.bin.twin", path / "w.bin")
    third = _context(plan, RecordingLauncher(), scratch)
    assert third.materialize("weights", entry.key, entry_sha256(entry)) == path
    [role] = third.launcher.launched
    assert role.name == f"verify-weights-{entry_sha256(entry)[:12]}" and role.owner == "harness"
    assert role.argv[3:8] == ("verify", "--copy", str(path), "--expect", entry_sha256(entry))
    assert third.cache_entries == {f"weights/{entry.key}": entry_sha256(entry)}


def test_a_copy_an_earlier_stage_verified_and_something_wrote_into_since_is_not_mounted_again(tmp_path):
    plan = noop_plan(tmp_path)
    scratch = tmp_path / "node" / "hls-100"
    first = _context(plan, RecordingLauncher(), scratch)
    entry = _published(first)
    path = first.materialize("weights", entry.key, entry_sha256(entry))
    (path / "w.bin").write_bytes(b"written by a harness role")
    later = _context(plan, RecordingLauncher(), scratch)
    with pytest.raises(StageFault, match="the node-local copy of cache entry .* changed since it was "
                                         "verified: changed w.bin") as caught:
        later.materialize("weights", entry.key, entry_sha256(entry))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and later.cache_entries == {}
    assert fault.evidence["entry"] == f"weights/{entry.key}"


def test_a_copy_an_earlier_stage_verified_and_something_touched_since_is_not_mounted_again(tmp_path):
    plan = noop_plan(tmp_path)
    scratch = tmp_path / "node" / "hls-100"
    first = _context(plan, RecordingLauncher(), scratch)
    entry = _published(first)
    path = first.materialize("weights", entry.key, entry_sha256(entry))
    recorded = entry.manifest.files[0].mtime_ns
    os.utime(path / "w.bin", ns=(recorded, recorded + 1))
    later = _context(plan, RecordingLauncher(), scratch)
    with pytest.raises(StageFault, match="the node-local copy of cache entry .* changed since it was "
                                         f"verified: w.bin has mtime {recorded + 1} ns, not the recorded "
                                         f"{recorded}") as caught:
        later.materialize("weights", entry.key, entry_sha256(entry))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and later.cache_entries == {}


def test_a_missing_cache_entry_is_a_retryable_infra_fault(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    with pytest.raises(StageFault, match="cache entry weights/nope is not published") as caught:
        ctx.materialize("weights", "nope", "0" * 64)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert ctx.cache_entries == {} and ctx.launcher.launched == []


def test_an_entry_published_again_under_its_key_is_not_the_one_its_producer_recorded(tmp_path):
    """A consumer pins each entry to the ENTRY.json digest its producer recorded, so an entry
    removed and filled again under the same key is refused, not silently used."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    key = "org+m@" + "c" * 40
    recorded = entry_sha256(_published(ctx, key))
    shutil.rmtree(ctx.store.root / "cache" / "weights" / key)
    refilled = entry_sha256(_published(ctx, key, data=b"w2"))
    message = f"cache entry weights/{key} has ENTRY.json digest {refilled}, not the {recorded} its producer recorded"
    with pytest.raises(StageFault, match=re.escape(message)) as caught:
        ctx.materialize("weights", key, recorded)
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert (fault.evidence["expected"], fault.evidence["found"]) == (recorded, refilled)
    assert ctx.cache_entries == {} and not (tmp_path / "node" / "hls-100" / "100" / "weights" / key).exists()
    with pytest.raises(ValueError):
        ctx.materialize("weights", key, "sha256:" + refilled)


def _published(ctx: StageContext, key: str = "org+m@" + "b" * 40, *, data: bytes = b"w") -> CacheEntry:
    return _entry_of(ctx, "weights", key, {"w.bin": data})


def test_a_corrupt_entry_manifest_is_an_infra_fault_a_retry_would_repeat(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    entry = _published(ctx)
    (entry.path / "ENTRY.json").write_text("{not json")
    with pytest.raises(StageFault, match=f"cache entry weights/{re.escape(entry.key)} is corrupt") as caught:
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable, fault.evidence["entry"]) == ("infra", False, f"weights/{entry.key}")
    assert ctx.cache_entries == {}


@pytest.mark.parametrize("damage", [lambda p: p.write_bytes(b"changed"), lambda p: p.write_bytes(b"x")],
                         ids=["resized", "same size"])
def test_an_entry_whose_published_files_changed_is_an_infra_fault_a_retry_would_repeat(tmp_path, damage):
    """The copy role refuses the entry in the store, which a retry would read again."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    entry = _published(ctx)
    damage(entry.path / "w.bin")
    with pytest.raises(StageFault, match=f"cache entry weights/{re.escape(entry.key)} is corrupt: changed "
                                         "w.bin") as caught:
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    fault = caught.value.fault
    assert not fault.retryable and fault.evidence["entry"] == f"weights/{entry.key}"
    assert fault.evidence["role"] == f"copy-weights-{entry_sha256(entry)[:12]}"
    assert ctx.cache_entries == {} and list((ctx.scratch / "100" / "weights").iterdir()) == []


def test_a_bad_node_local_copy_of_a_sound_entry_is_a_retryable_infra_fault(tmp_path):
    """A copy that lands other bytes than the entry holds went wrong on the way, and a retry
    copies the entry anew: the role fails without refusing the entry."""
    launcher = LocalLauncher(gpus=2, replace_modules={"hls.store.cache_role": FLIPPING})
    _, ctx = _ctx(tmp_path, launcher)
    entry = _published(ctx)
    with pytest.raises(StageFault, match=f"cache entry weights/{re.escape(entry.key)} could not be copied to "
                                         "the node: harness role copy-weights-") as caught:
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.evidence["entry"] == f"weights/{entry.key}"
    assert ctx.cache_entries == {} and list((ctx.scratch / "100" / "weights").iterdir()) == []


def test_a_manifest_copy_gone_wrong_on_the_way_is_a_retryable_infra_fault(tmp_path, monkeypatch):
    """The copy of ENTRY.json lands other bytes while the store's is sound: a retry copies it again."""
    monkeypatch.setenv("FLIP_ONLY", "ENTRY.json")
    launcher = LocalLauncher(gpus=2, replace_modules={"hls.store.cache_role": FLIPPING})
    _, ctx = _ctx(tmp_path, launcher)
    entry = _published(ctx)
    with pytest.raises(StageFault, match=f"cache entry weights/{re.escape(entry.key)} could not be copied to "
                                         "the node: harness role copy-weights-.* status 75") as caught:
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    assert caught.value.fault.retryable and ctx.cache_entries == {}


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
@pytest.mark.parametrize("flip_only", [None, "ENTRY.json"], ids=["data file", "manifest"])
def test_a_failed_reread_of_the_store_after_a_bad_copy_is_retried_not_refused(tmp_path, monkeypatch,
                                                                               flip_only):
    """A read error during the role's second read of the store's file says nothing about the
    entry: it is the role's own failure, retried, and never a refusal of the entry."""
    monkeypatch.setenv("HIDE_SOURCE", "1")
    if flip_only is not None:
        monkeypatch.setenv("FLIP_ONLY", flip_only)
    launcher = LocalLauncher(gpus=2, replace_modules={"hls.store.cache_role": FLIPPING})
    _, ctx = _ctx(tmp_path, launcher)
    entry = _published(ctx)
    try:
        with pytest.raises(StageFault, match="could not be copied to the node: harness role copy-weights-.* "
                                             "status 1") as caught:
            ctx.materialize("weights", entry.key, entry_sha256(entry))
    finally:
        for p in entry.path.rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)
    assert caught.value.fault.retryable and "corrupt" not in caught.value.fault.reason


def test_an_entry_manifest_that_changed_after_the_lookup_is_refused_and_not_retried(tmp_path, monkeypatch):
    """The store's ENTRY.json is not the one the stage pinned when the copy role reads it: the
    role reads it again, finds the store's other, and refuses the entry."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    entry = _published(ctx)
    real = ctx._published

    def then_rewritten(*args):
        found = real(*args)
        (found.path / "ENTRY.json").write_text((found.path / "ENTRY.json").read_text() + "\n")
        return found

    monkeypatch.setattr(ctx, "_published", then_rewritten)
    with pytest.raises(StageFault, match=f"cache entry weights/{re.escape(entry.key)} is corrupt: the "
                                         "entry's ENTRY.json has sha256 [0-9a-f]+, not the "
                                         f"{entry_sha256(entry)} its producer recorded") as caught:
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    assert not caught.value.fault.retryable


def test_a_copy_whose_roles_report_does_not_match_the_manifest_is_a_retryable_infra_fault(tmp_path):
    launcher = LocalLauncher(gpus=2, replace_modules={"hls.store.cache_role": MISREPORTING})
    _, ctx = _ctx(tmp_path, launcher)
    entry = _published(ctx)
    with pytest.raises(StageFault, match=f"the node-local copy of cache entry weights/{re.escape(entry.key)} "
                                         "differs from the entry: .*changed w.bin") as caught:
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable, fault.evidence["entry"]) == ("infra", True, f"weights/{entry.key}")
    assert ctx.cache_entries == {} and list((ctx.scratch / "100" / "weights").iterdir()) == []


def test_a_copy_whose_role_reports_no_digests_is_a_retryable_infra_fault(tmp_path, monkeypatch):
    monkeypatch.setenv("HANGING_ROLE_EXIT", "0")
    launcher = LocalLauncher(gpus=2, replace_modules={"hls.store.cache_role": HANGING})
    _, ctx = _ctx(tmp_path, launcher)
    entry = _published(ctx)
    with pytest.raises(StageFault, match="reported no digests") as caught:
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    assert caught.value.fault.retryable and ctx.cache_entries == {}


def test_a_copy_still_running_at_the_deadline_is_killed(tmp_path):
    """The runner waits for the copy as for any role, so the stage's deadline and a stop still
    apply while a large entry is copied."""
    launcher = LocalLauncher(gpus=2, replace_modules={"hls.store.cache_role": HANGING})
    _, ctx = _ctx(tmp_path, launcher, deadline_in=3.0)
    entry = _published(ctx)
    started = time.monotonic()
    with pytest.raises(StageFault, match="ended time_limit") as caught:
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    assert time.monotonic() - started < 30 and caught.value.fault.retryable


def test_a_copy_is_stopped_when_the_allocation_stops(tmp_path):
    launcher = LocalLauncher(gpus=2, replace_modules={"hls.store.cache_role": HANGING})
    _, ctx = _ctx(tmp_path, launcher)
    entry = _published(ctx)
    timer = threading.Timer(1.0, ctx.stop_roles)
    timer.start()
    try:
        with pytest.raises(StageFault, match="stopped") as caught:
            ctx.materialize("weights", entry.key, entry_sha256(entry))
    finally:
        timer.cancel()
    assert caught.value.fault.retryable and ctx.cache_entries == {}


def test_an_entry_is_copied_only_before_the_stages_roles_start(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    entry = _published(ctx)
    ctx.start_role("long", _python("import time; time.sleep(30)"), SHAPE, owner="harness")
    try:
        with pytest.raises(RuntimeError, match="have not been waited for"):
            ctx.materialize("weights", entry.key, entry_sha256(entry))
        assert [s.name for s in ctx.launcher.launched] == ["long"]
        assert list((ctx.scratch / "100" / "weights").iterdir()) == []
    finally:
        ctx.close()


@pytest.mark.parametrize("code", [3, 0])
def test_an_entry_is_not_copied_while_a_role_that_ended_is_not_yet_waited_for(tmp_path, code):
    """The copy's wait would report every role not yet reported, and take a role's outcome, a
    fault or a success, away from the stage that started it: the copy waits until the stage has
    waited for its roles."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    entry = _published(ctx)
    handle = ctx.start_role("quick", _python(f"import sys; sys.exit({code})"), SHAPE, owner="harness")
    handle.wait(30)
    assert ctx.running_roles() == []
    with pytest.raises(RuntimeError, match=r"roles \['quick'\] have not been waited for"):
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    assert [s.name for s in ctx.launcher.launched] == ["quick"]
    [outcome] = ctx.wait_roles()
    assert (outcome.name, outcome.exit_code) == ("quick", code)
    if code == 0:
        assert (ctx.materialize("weights", entry.key, entry_sha256(entry)) / "w.bin").read_bytes() == b"w"
    ctx.close()


def test_an_entry_is_not_copied_once_the_stage_is_stopping(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    entry = _published(ctx)
    ctx.stopping = True
    with pytest.raises(StageFault, match="stopping") as caught:
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    assert caught.value.fault.retryable and ctx.launcher.launched == []


def test_a_stage_whose_memory_leaves_the_copy_none_is_refused(tmp_path):
    plan = noop_plan(tmp_path)
    # A runner that kept the whole of the stage's 6 GiB; the planner refuses such a plan.
    run = plan.settings.run.model_copy(update={"runner_memory_gib": 6})
    plan = plan.model_copy(update={"settings": plan.settings.model_copy(update={"run": run})})
    ctx = _context(plan, RecordingLauncher(), tmp_path / "node" / "hls-100")
    entry = _published(ctx)
    with pytest.raises(StageFault, match="leaves role copy-weights-.* none") as caught:
        ctx.materialize("weights", entry.key, entry_sha256(entry))
    assert not caught.value.fault.retryable and ctx.launcher.launched == []


def test_the_runner_reads_no_file_of_a_weights_rows_or_tools_entry_or_of_its_copy(tmp_path):
    """Copies, reuses and checks of copies of every kind move their bytes through roles; the
    stage's own process reads each entry's manifest and each copy's stamp only."""
    plan = noop_plan(tmp_path)
    scratch = tmp_path / "node" / "hls-100"
    first = _context(plan, RecordingLauncher(), scratch)
    entries = {
        "weights": _entry_of(first, "weights", "org+m@" + "d" * 40, {"model.safetensors": b"\x01" * 4096}),
        "rows": _entry_of(first, "rows", "rows-" + "0" * 16,
                          {"tenants/t0/dev/rows.json": b"[]", "ids.json": b'{"dev": [], "test": []}'}),
        "tools": _entry_of(first, "tools", "bundle-" + "1" * 8 + "-full", {"tool.bin": b"t"}),
    }
    with recording_reads() as reads:
        copies = {kind: first.materialize(kind, e.key, entry_sha256(e)) for kind, e in entries.items()}
        first.close()
        later = _context(plan, RecordingLauncher(), scratch)
        assert {k: later.materialize(k, e.key, entry_sha256(e)) for k, e in entries.items()} == copies
        for copy in copies.values():
            stamp = json.loads((copy / ".verified").read_text())
            (copy / ".verified").write_text(json.dumps({**stamp, "job_id": "another"}))
        third = _context(plan, RecordingLauncher(), scratch)
        assert {k: third.materialize(k, e.key, entry_sha256(e)) for k, e in entries.items()} == copies
    assert entry_reads(reads, [first.store.root / "cache", copies_root(scratch, "100")]) == []
    assert [s.name.split("-")[0] for s in first.launcher.launched] == ["copy"] * 3
    assert later.launcher.launched == []
    assert [s.name.split("-")[0] for s in third.launcher.launched] == ["verify"] * 3


def test_an_output_the_stage_writes_is_recorded_by_the_bytes_it_wrote(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    key = ctx.write_output("summary.json", b'{"a": 1}')
    with recording_reads() as reads:
        ref = ctx.output_ref(key)
    assert ref == OutputRef(key=key, sha256=sha256_bytes(b'{"a": 1}'), size=8) and reads == []
    # Written over since, by a role, say: the size it has now is not the one written.
    ctx.store.write_bytes(key, b'{"a": 12}')
    with pytest.raises(StageFault, match="summary.json has 9 bytes, not the 8 its writer gave") as caught:
        ctx.output_ref(key)
    assert caught.value.fault.retryable


def _role_writing_output(ctx: StageContext, name: str, data: bytes, *, report: bool = True,
                         file: str | None = None) -> str:
    """Start harness role *name*, which writes *data* to output *file* (``<name>.bin`` by default)
    and, with *report*, reports it as a role does, keyed by its path under the outputs, wait for
    it, and return the output's name."""
    file = file or f"{name}.bin"
    out, report_path, root = ctx.output_path(file), ctx.outputs_report(name), ctx.outputs_dir
    code = ("import sys; from pathlib import Path; from hls.store.manifest import report_outputs; "
            f"Path(sys.argv[1]).write_bytes({data!r}); "
            + ("report_outputs(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[1]))" if report
               else "pass"))
    ctx.start_role(name, _python(code) + [str(out), str(report_path), str(root)], SHAPE, owner="harness")
    assert [o.exit_code for o in ctx.wait_roles()] == [0]
    return file


def test_an_output_a_role_writes_is_recorded_by_the_digest_it_reported_and_never_read(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    file = _role_writing_output(ctx, "writer", b"a large output")
    with recording_reads() as reads:
        key = ctx.reported_output(file, "writer")
        ref = ctx.output_ref(key)
    assert key == ctx.output_key(file) and reads == [str(ctx.outputs_report("writer"))]
    assert ref == OutputRef(key=key, sha256=sha256_bytes(b"a large output"), size=14)


def test_outputs_under_directories_are_reported_by_their_paths_under_the_outputs(tmp_path):
    """An output may lie in a directory, such as a trace's ``trace/END.json``, and two outputs
    may share a base name: each is reported and recorded by its own path."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    first = _role_writing_output(ctx, "driver", b"{}", file="trace/END.json")
    second = _role_writing_output(ctx, "other", b'{"a": 1}', file="verdict/END.json")
    refs = [ctx.output_ref(ctx.reported_output(first, "driver")),
            ctx.output_ref(ctx.reported_output(second, "other"))]
    assert [(r.key, r.sha256, r.size) for r in refs] == [
        (ctx.output_key("trace/END.json"), sha256_bytes(b"{}"), 2),
        (ctx.output_key("verdict/END.json"), sha256_bytes(b'{"a": 1}'), 8)]


@pytest.mark.parametrize("when", ["after the stage took the report", "before the stage took the report"])
def test_a_role_output_whose_size_changed_since_its_report_is_a_retryable_infra_fault(tmp_path, when):
    """The report carries each output's size, which publish holds the file to: a metadata check,
    so only a same-size rewrite goes unseen without a read. The size is the role's, whenever the
    file changed."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    file = _role_writing_output(ctx, "writer", b"what the role wrote")
    if when.startswith("before"):
        ctx.store.path(ctx.output_key(file)).write_bytes(b"something else, of another length")
    key = ctx.reported_output(file, "writer")
    ctx.store.path(key).write_bytes(b"something else, of another length")
    with pytest.raises(StageFault, match="writer.bin has 33 bytes, not the 19 its writer gave") as caught:
        ctx.output_ref(key)
    assert caught.value.fault.retryable


def test_a_report_counts_only_from_a_role_the_stage_has_waited_for(tmp_path):
    """A role still running could write its output again after it reported it."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    out, report = ctx.output_path("w.bin"), ctx.outputs_report("w")
    code = ("import sys, time; from pathlib import Path; from hls.store.manifest import report_outputs; "
            "Path(sys.argv[1]).write_bytes(b'first'); "
            "report_outputs(Path(sys.argv[2]), Path(sys.argv[1]).parent, Path(sys.argv[1])); time.sleep(30)")
    ctx.start_role("w", _python(code) + [str(out), str(report)], SHAPE, owner="harness")
    deadline = time.time() + 20
    while not report.exists() and time.time() < deadline:
        time.sleep(0.05)
    try:
        with pytest.raises(RuntimeError, match=r"role w has not been waited for"):
            ctx.reported_output("w.bin", "w")
    finally:
        ctx.close()


def test_an_output_whose_role_reported_no_digest_is_a_retryable_infra_fault(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    file = _role_writing_output(ctx, "writer", b"x", report=False)
    with pytest.raises(StageFault, match="role writer reported no digest of output writer.bin") as caught:
        ctx.reported_output(file, "writer")
    assert caught.value.fault.retryable
    with pytest.raises(ValueError, match="no role nobody"):
        ctx.reported_output(file, "nobody")
    # A report that is no map of records is no report.
    ctx.outputs_report("writer").write_text('{"writer.bin": "' + "a" * 64 + '"}')
    with pytest.raises(StageFault, match="role writer reported no digest of output writer.bin"):
        ctx.reported_output(file, "writer")


def test_an_output_no_writer_reported_is_never_published(tmp_path):
    """A stage names an output that a role wrote but whose digest nobody recorded: the runner
    would otherwise have to read it."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    file = _role_writing_output(ctx, "writer", b"x", report=False)
    with pytest.raises(StageFault, match="no writer reported the sha256 of output") as caught:
        ctx.output_ref(ctx.output_key(file))
    assert not caught.value.fault.retryable


def test_a_contestant_containers_report_of_an_output_is_never_taken(tmp_path, apptainer_on_path):
    """A contestant's code chooses what it writes, its own reports among them."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    ctx.wait_roles()
    with pytest.raises(ValueError, match="contestant container srv"):
        ctx.reported_output("srv.bin", "srv")


def test_recorded_images_are_kept_by_role(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    ctx.record_image("artifact", "b" * 64)
    assert ctx.image_digests == {"artifact": "b" * 64}
    with pytest.raises(ValueError):
        ctx.record_image("artifact", "sha256:" + "b" * 64)


# The build sandbox: a contestant container started with a harness reserve.

GIB = 2**30
_SLEEP = tuple(_python("import time; time.sleep(30)"))


class _Planted:
    """Stands in for ``SplitSampler``: made for a step's cgroup, it holds the samples a test
    planted in place of reading /proc, and notes when it was started and stopped."""

    def __init__(self) -> None:
        self.planted: list[SplitSample] = []
        self.cgroups: list[Path] = []
        self.calls: list[str] = []
        self.ready_sources: list[object] = []

    def __call__(self, cgroup_dir: Path, *, ready_source: object = None) -> _Planted:
        self.cgroups.append(cgroup_dir)
        self.ready_sources.append(ready_source)
        return self

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def samples(self) -> list[SplitSample]:
        return list(self.planted)


class _Killer:
    """Stands in for ``ContestantKiller``: made for a step's cgroup, it ends no process and says
    it did while ``ends`` is true, and notes each call and its close."""

    def __init__(self, *, ends: bool = True) -> None:
        self.ends = ends
        self.cgroups: list[Path] = []
        self.ended: list[float] = []
        self.closed = False
        self.ready_sources: list[object] = []

    def __call__(self, cgroup_dir: Path, *, ready_source: object = None) -> _Killer:
        self.cgroups.append(cgroup_dir)
        self.ready_sources.append(ready_source)
        return self

    def end_one(self) -> bool:
        self.ended.append(time.time())
        return self.ends

    def exiting(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


def _over(cgroup: Path) -> None:
    """Fake memory files that put the step at 3 GiB, past SHAPE's 2."""
    cgroup.mkdir(parents=True, exist_ok=True)
    (cgroup / "memory.current").write_text(f"{3 * GIB}\n")
    (cgroup / "memory.swap.current").write_text("0\n")
    (cgroup / "memory.stat").write_text(f"anon {3 * GIB}\nfile 0\nshmem 0\n")
    (cgroup / "memory.events").write_text("oom 0\noom_kill 0\n")


def _started(ctx: StageContext, name: str) -> None:
    """Wait until role *name*'s command has started: a kill before its marker reads as the runtime's."""
    err = ctx.attempt_dir / "logs" / f"{name}.err"
    deadline = time.time() + 30
    while START_MARKER not in (err.read_text() if err.exists() else "") and time.time() < deadline:
        time.sleep(0.05)


def test_a_sandbox_breach_ends_one_contestant_process_and_the_sandbox_lives_on(tmp_path, apptainer_on_path,
                                                                               monkeypatch):
    killer = _Killer()
    monkeypatch.setattr(stage_module, "ContestantKiller", killer)
    watch = MemoryWatch(enforce=True)
    _, ctx = _ctx(tmp_path, RecordingLauncher(cgroup_root=tmp_path / "cg"), watch=watch)
    handle = ctx.start_container("sandbox", _quick(ctx, "sandbox", command=_SLEEP), SHAPE, owner="contestant",
                                 harness_reserve_gib=1)
    _over(tmp_path / "cg" / "cg-sandbox")
    _started(ctx, "sandbox")
    watch.sample_once()
    assert killer.cgroups == [tmp_path / "cg" / "cg-sandbox"] and len(killer.ended) == 1 and handle.poll() is None
    # Stopped with the rest of the stage, as at a shutdown: infra, whatever it survived before.
    [outcome] = ctx.close()
    # A single kill never ends the sandbox, so it is no OOM of its end, only a record.
    assert (outcome.ending, outcome.split.oom_by, outcome.split.reserve_bytes) == ("stopped", None, GIB)
    assert (len(outcome.split.process_kills), outcome.split.process_kill_count, killer.closed) == (1, 1, True)
    assert classify_role(outcome).fault_class == "infra" and ctx.contestant_fault() is None


@pytest.mark.parametrize("harness, fault_class", [(GIB, "contestant"), (GIB + 1, "infra")],
                         ids=["harness-within-reserve", "harness-over-reserve"])
def test_a_sandbox_killed_at_its_limit_is_judged_by_its_split_wherever_its_fault_is_read(
        tmp_path, apptainer_on_path, monkeypatch, harness, fault_class):
    sampler = _Planted()
    monkeypatch.setattr(stage_module, "ContestantKiller", _Killer(ends=False))
    monkeypatch.setattr(stage_module, "SplitSampler", sampler)
    watch = MemoryWatch(enforce=True)
    _, ctx = _ctx(tmp_path, RecordingLauncher(cgroup_root=tmp_path / "cg"), watch=watch)
    ctx.start_container("sandbox", _quick(ctx, "sandbox", command=_SLEEP), SHAPE, owner="contestant",
                        harness_reserve_gib=1)
    _over(tmp_path / "cg" / "cg-sandbox")
    _started(ctx, "sandbox")
    sampler.planted.append(SplitSample(at=time.time() - 1, total=3 * GIB, tmpfs=0, harness=harness,
                                       contestant=2 * GIB - 1, oom_kills=0))
    watch.sample_once()
    [outcome] = ctx.wait_roles()
    assert sampler.cgroups == [tmp_path / "cg" / "cg-sandbox"] and sampler.calls == ["start", "stop"]
    assert (outcome.ending, outcome.split.oom_by, outcome.split.harness_bytes) == ("memory_limit", "runner", harness)
    assert classify_role(outcome).fault_class == fault_class
    assert stage_fault([outcome]).fault_class == fault_class
    decided = ctx.contestant_fault()
    assert (decided is not None and decided.fault_class) == (fault_class == "contestant" and "contestant")
    [event] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"]
    assert (event["split"]["oom_by"], event["split"]["harness_bytes"]) == ("runner", harness)


def test_a_split_container_hands_its_ready_source_to_the_sampler_and_the_killer(
        tmp_path, apptainer_on_path, monkeypatch):
    # The build sandbox measures the contestant's PID namespace depth from the pids the exec
    # server's ready line names; the context must hand that source to both the split sampler and
    # the killer, so neither falls back to a depth it did not measure.
    sampler, killer = _Planted(), _Killer()
    monkeypatch.setattr(stage_module, "SplitSampler", sampler)
    monkeypatch.setattr(stage_module, "ContestantKiller", killer)
    _, ctx = _ctx(tmp_path, RecordingLauncher(cgroup_root=tmp_path / "cg"), watch=MemoryWatch(enforce=True))

    def source() -> ReadyPids:
        return ReadyPids(spawner_pid=15, reaper_pid=17)

    ctx.start_container("sandbox", _quick(ctx, "sandbox", command=tuple(_python("pass"))), SHAPE,
                        owner="contestant", harness_reserve_gib=1, ready_source=source)
    ctx.wait_roles()
    assert sampler.ready_sources == [source] and killer.ready_sources == [source]


@pytest.mark.parametrize("limit, fault_class", [("own", "contestant"), ("job", "infra")])
def test_a_sandbox_the_kernel_oom_killed_is_the_contestants_only_at_its_own_limit(tmp_path, apptainer_on_path,
                                                                                  monkeypatch, limit, fault_class):
    sampler = _Planted()
    monkeypatch.setattr(stage_module, "SplitSampler", sampler)
    job = _job(tmp_path)
    _, ctx = _ctx(tmp_path, _OomLauncher(job), watch=MemoryWatch(enforce=False, job_dir=job))
    ctx.start_container("sandbox", _quick(ctx, "sandbox", command=tuple(_python("import sys; sys.exit(137)"))),
                        SHAPE, owner="contestant", harness_reserve_gib=1)
    sampler.planted.append(SplitSample(at=time.time(), total=GIB, tmpfs=0, harness=GIB // 2, contestant=GIB // 2,
                                       oom_kills=0))
    assert ctx._roles["sandbox"].wait(30) == 137
    # The kill came after the last sample, and only the step's own count says whose limit it was.
    (job / "step_1" / "memory.events").write_text(f"oom {int(limit == 'own')}\noom_kill 1\n")
    if limit == "job":
        (job / "memory.events.local").write_text("oom 1\noom_kill 0\n")
    (job / "memory.events").write_text("oom 1\noom_kill 0\n")
    [outcome] = ctx.wait_roles()
    assert (outcome.own_oom, outcome.job_oom, outcome.split.oom_by) == (limit == "own", limit == "job", "kernel")
    assert outcome.split.oom_at == outcome.split.ended_at and outcome.split.sample_at == sampler.planted[0].at
    assert classify_role(outcome).fault_class == fault_class


def test_a_sandbox_whose_contestant_stays_over_its_limit_loses_its_step_as_the_contestants_fault(
        tmp_path, apptainer_on_path, monkeypatch):
    # Each process the watch ends is replaced at once, so the footprint never drops.
    sampler = _Planted()
    monkeypatch.setattr(stage_module, "ContestantKiller", _Killer())
    monkeypatch.setattr(stage_module, "SplitSampler", sampler)
    watch = MemoryWatch(enforce=True, repeat_grace_sec=0.0)
    _, ctx = _ctx(tmp_path, RecordingLauncher(cgroup_root=tmp_path / "cg"), watch=watch)
    handle = ctx.start_container("sandbox", _quick(ctx, "sandbox", command=_SLEEP), SHAPE, owner="contestant",
                                 harness_reserve_gib=1)
    _over(tmp_path / "cg" / "cg-sandbox")
    _started(ctx, "sandbox")
    sampler.planted.append(SplitSample(at=time.time() - 1, total=3 * GIB, tmpfs=0, harness=GIB,
                                       contestant=2 * GIB - 1, oom_kills=0))
    for _ in range(3):
        watch.sample_once()
    assert handle.poll() is None
    watch.sample_once()
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.split.oom_by, len(outcome.split.process_kills)) == ("memory_limit", "runner", 3)
    assert outcome.split.step_kill_reason.startswith("3 of its processes were ended since")
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == ("contestant", False)
    assert fault.evidence["step_kill_reason"] == outcome.split.step_kill_reason
    assert ctx.contestant_fault().fault_class == "contestant"
    [event] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"]
    assert len(event["split"]["process_kills"]) == 3 and event["split"]["step_kill_reason"].startswith("3 of")


def _exit_on(go: Path, code: int) -> tuple[str, ...]:
    """A command that waits for the file *go*, then exits *code*."""
    return tuple(_python(f"import os, sys, time\nwhile not os.path.exists({str(go)!r}):\n    time.sleep(0.05)\n"
                         f"sys.exit({code})"))


def test_a_single_kill_never_makes_a_later_end_of_the_sandbox_the_contestants(tmp_path, apptainer_on_path,
                                                                              monkeypatch):
    # A test server is ended at a breach and the build goes on; then the exec
    # server exits 1 for a reason of its own, with no OOM of the sandbox's.
    sampler, killer = _Planted(), _Killer()
    monkeypatch.setattr(stage_module, "SplitSampler", sampler)
    monkeypatch.setattr(stage_module, "ContestantKiller", killer)
    watch, go, cgroup = MemoryWatch(enforce=True), tmp_path / "go", tmp_path / "cg" / "cg-sandbox"
    _, ctx = _ctx(tmp_path, RecordingLauncher(cgroup_root=tmp_path / "cg"), watch=watch)
    ctx.start_container("sandbox", _quick(ctx, "sandbox", command=_exit_on(go, 1)), SHAPE, owner="contestant",
                        harness_reserve_gib=1)
    _over(cgroup)
    _started(ctx, "sandbox")
    sampler.planted.append(SplitSample(at=time.time() - 1, total=3 * GIB, tmpfs=0, harness=GIB // 2,
                                       contestant=2 * GIB, oom_kills=0))
    watch.sample_once()
    (cgroup / "memory.current").write_text(f"{GIB}\n")
    watch.sample_once()
    assert len(killer.ended) == 1
    go.touch()
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code, outcome.split.oom_at, outcome.split.process_kill_count) == (
        "exited", 1, None, 1)
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "no OOM recorded" in fault.reason


def test_a_single_kill_after_a_kernel_oom_at_the_jobs_limit_leaves_the_end_infra(tmp_path, apptainer_on_path,
                                                                                 monkeypatch):
    sampler, killer = _Planted(), _Killer()
    monkeypatch.setattr(stage_module, "SplitSampler", sampler)
    monkeypatch.setattr(stage_module, "ContestantKiller", killer)
    job, go = _job(tmp_path), tmp_path / "go"
    watch = MemoryWatch(enforce=True, job_dir=job)
    _, ctx = _ctx(tmp_path, _OomLauncher(job), watch=watch)
    ctx.start_container("sandbox", _quick(ctx, "sandbox", command=_exit_on(go, 137)), SHAPE, owner="contestant",
                        harness_reserve_gib=1)
    _started(ctx, "sandbox")
    now = time.time()
    # The kernel kills at the job's limit, seen 2 s ago; the runner then ends a process of the step.
    sampler.planted += [SplitSample(at=now - 5, total=GIB, tmpfs=0, harness=GIB // 2, contestant=GIB // 2,
                                    oom_kills=0),
                        SplitSample(at=now - 2, total=GIB, tmpfs=0, harness=GIB // 2, contestant=GIB // 2,
                                    oom_kills=1)]
    (job / "memory.events.local").write_text("oom 1\noom_kill 0\n")
    (job / "memory.events").write_text("oom 1\noom_kill 0\n")
    (job / "step_1" / "memory.events").write_text("oom 0\noom_kill 1\n")
    _over(job / "step_1" / "user")
    watch.sample_once()
    assert len(killer.ended) == 1
    go.touch()
    [outcome] = ctx.wait_roles()
    assert (outcome.own_oom, outcome.job_oom, outcome.split.oom_by, outcome.split.oom_at) == (
        False, True, "kernel", now - 2)
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == ("infra", True)


def test_a_sandbox_that_ends_by_itself_is_infra_where_another_container_has_a_result(tmp_path, apptainer_on_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    [srv] = ctx.wait_roles()
    ctx.start_container("sandbox", _quick(ctx, "sandbox"), SHAPE, owner="contestant", harness_reserve_gib=1)
    [sandbox] = ctx.wait_roles()
    assert (srv.ending, srv.exit_code, srv.split, classify_role(srv)) == ("exited", 0, None, None)
    assert (sandbox.ending, sandbox.exit_code, sandbox.split.oom_at) == ("exited", 0, None)
    fault = classify_role(sandbox)
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "no OOM recorded" in fault.reason


@pytest.mark.parametrize("owner, reserve", [("harness", 1), ("contestant", 0), ("contestant", 2)])
def test_a_harness_reserve_is_for_a_contestant_container_and_below_its_memory(tmp_path, owner, reserve):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    with pytest.raises(ValueError, match="harness reserve"):
        ctx.start_container("sandbox", _quick(ctx, "sandbox"), SHAPE, owner=owner, harness_reserve_gib=reserve)
    assert launcher.launched == []


def test_a_host_roles_secret_env_file_can_be_written_before_it_starts(tmp_path):
    """A stage whose contestant may fill the node's disk writes a later host role's env file
    before the contestant starts (``prepare_role_env``); starting the role takes that file, and
    refuses other secrets and logs that would go to the node's disk (no *scrub_logs*)."""
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS)
    ctx.prepare_role_env("dl", ["HF_TOKEN"])
    env_file = ctx.role_dir("dl") / "role.env"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="already started or prepared"):
        ctx.prepare_role_env("dl", ["HF_TOKEN"])
    with pytest.raises(ValueError, match="not the ones its env file was prepared with"):
        ctx.start_role("dl", ["true"], SHAPE, secret_env=["OTHER_TOKEN"], owner="harness", scrub_logs=True)
    with pytest.raises(ValueError, match="scrub_logs"):
        ctx.start_role("dl", ["true"], SHAPE, secret_env=["HF_TOKEN"], owner="harness")
    assert not (ctx.role_dir("dl") / "logs").exists()
    with pytest.raises(ValueError, match=r"was not granted \['RELAY_TOKEN'\]"):
        ctx.prepare_role_env("other", ["RELAY_TOKEN"])
    out = tmp_path / "env.json"
    ctx.start_role("dl", _python("import json, os, sys; json.dump(os.environ.get('HF_TOKEN'), open(sys.argv[1], 'w'))")
                   + [str(out)], SHAPE, secret_env=["HF_TOKEN"], owner="harness", scrub_logs=True)
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0
    assert json.loads(out.read_text()) == SECRETS["HF_TOKEN"]
    assert launcher.env_files["dl"][0] == 0o600 and not env_file.exists()


def test_only_an_infra_fault_never_retried_is_declared_past_a_stages_point_of_no_retry(tmp_path):
    """``declare_no_retry`` keeps the fault an end of the stage is from then on, for the runner's
    SIGTERM; a fault that would be retried, or a contestant's, is refused."""
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    assert ctx.no_retry_fault() is None
    with pytest.raises(ValueError, match="retried"):
        ctx.declare_no_retry(infra_fault("x", retryable=True))
    with pytest.raises(ValueError, match="infra"):
        ctx.declare_no_retry(FaultRecord(fault_class="contestant", reason="x", retryable=False))
    fault = infra_fault("the build failed after its commit point", retryable=False)
    ctx.declare_no_retry(fault)
    assert ctx.no_retry_fault() == fault


@register
class _DecidingStage(Stage):
    """A stage that decides its contestant faults itself, as the agent build does."""

    kind: ClassVar[str] = "deciding_stand_in"
    params_model: ClassVar[type[BaseModel]] = NoopParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()
    decides_contestant_faults: ClassVar[bool] = True

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        pass

    @classmethod
    def max_roles(cls, params: BaseModel) -> int:
        """Its tests start roles on its context; it starts none."""
        return 0

    def run(self, ctx: StageContext) -> dict[str, str]:
        return {}


def test_a_stage_declares_only_a_contestant_fault_and_only_when_it_decides_them(tmp_path, apptainer_on_path):
    """``declare_contestant_fault`` names the zero of a stage that decides its contestant faults
    itself, which ``contestant_fault`` then reads, never its roles' endings. It refuses an infra
    fault, and a stage that leaves that reading to its roles."""
    plan = noop_plan(tmp_path, decls=[StageDecl("noop_a", "deciding_stand_in", SHAPE.model_copy(
        update={"memory_gib": 8}), {"gpu_roles": 0, "cpu_role": True})])
    ctx = _context(plan, RecordingLauncher(), tmp_path / "node" / "hls-100", watch=_BreachedWatch())
    # A contestant container the watch killed over its memory: by its roles, the contestant's fault.
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    ctx.wait_roles()
    assert stage_fault(list(ctx._outcomes.values())).fault_class == "contestant"
    assert ctx.contestant_fault() is None
    with pytest.raises(ValueError, match="is not a contestant fault"):
        ctx.declare_contestant_fault(infra_fault("a test", retryable=True))
    assert ctx.contestant_fault() is None
    zero = FaultRecord(fault_class="contestant", reason="a test", retryable=False)
    assert ctx.declare_contestant_fault(zero) == zero and ctx.contestant_fault() == zero
    _, other = _ctx(tmp_path / "other", RecordingLauncher())
    with pytest.raises(ValueError, match="does not decide its contestant faults"):
        other.declare_contestant_fault(zero)
    assert other.contestant_fault() is None


def test_a_containers_command_ceiling_leaves_its_steps_limit_and_is_judged_at_its_end(tmp_path, apptainer_on_path,
                                                                                      monkeypatch):
    """A container whose own harness gives its commands the process ceiling (the build sandbox's
    spawner) keeps its step's limit: the step wrapper lowers nothing, so the harness in it can
    still fork once the commands reach the ceiling. The context counts the uid's tasks against
    the ceiling when the role is reported, as for a step it lowers (``starvation_fault``)."""
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    ceiling = ctx.process_ceiling().ceiling
    monkeypatch.setattr(stage_module, "uid_tasks", lambda uid: ceiling - 50)
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant", command_nproc=ceiling)
    [outcome] = ctx.wait_roles()
    [spec] = [s for s in launcher.launched if s.name == "srv"]
    assert spec.nproc is None
    assert (outcome.pool_at_end.others_tasks, outcome.pool_at_end.rule.ceiling) == (ceiling - 50, ceiling)
    [ended] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"
               and e["role"] == "srv"]
    assert ended["ceiling"] == ceiling


def test_a_uid_already_at_a_containers_command_ceiling_refuses_its_start_as_infra(tmp_path, apptainer_on_path,
                                                                                  monkeypatch):
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    ceiling = ctx.process_ceiling().ceiling
    monkeypatch.setattr(stage_module, "uid_tasks", lambda uid: ceiling)
    with pytest.raises(StageFault) as e:
        ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant", command_nproc=ceiling)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert f"{ceiling} tasks" in fault.reason and f"ceiling of {ceiling}" in fault.reason
    assert [s.name for s in launcher.launched if s.name == "srv"] == []
    # The commands' ceiling is the stage's, and a harness container has none.
    with pytest.raises(ValueError, match="is not the stage's"):
        ctx.start_container("other", _quick(ctx, "other"), SHAPE, owner="contestant", command_nproc=ceiling - 1)
    with pytest.raises(ValueError, match="for a contestant's container"):
        ctx.start_container("tool", _quick(ctx, "tool"), SHAPE, owner="harness", command_nproc=ceiling)


def test_a_command_ceiling_is_counted_as_the_site_says_and_a_uid_at_it_starts_where_it_counts_per_namespace(
        tmp_path, apptainer_on_path, monkeypatch):
    # The spawner sets the commands' ceiling inside a namespace it made with the runner's limit:
    # where the kernel counts that namespace's own tasks, a uid already at the ceiling leaves the
    # commands every fork, and only one at the runner's hard limit refuses the start.
    monkeypatch.setattr(stage_module, "hard_limit", lambda: 3_000_000)
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher, site={"nproc_accounting": "user_namespace"})
    made = ctx.process_ceiling()
    monkeypatch.setattr(stage_module, "uid_tasks", lambda uid: made.ceiling)
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant", command_nproc=made.ceiling)
    [outcome] = ctx.wait_roles()
    assert outcome.pool_at_end.rule == CeilingRule(ceiling=made.ceiling, accounting="user_namespace",
                                                   parent_limit=3_000_000)
    assert outcome.pool_at_end.node_tasks > 0 and classify_role(outcome) is None
    [ended] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"]
    assert (ended["accounting"], ended["ceiling"]) == ("user_namespace", made.ceiling) and ended["node_tasks"] > 0
    monkeypatch.setattr(stage_module, "uid_tasks", lambda uid: 3_000_000)
    with pytest.raises(StageFault) as e:
        ctx.start_container("again", _quick(ctx, "again"), SHAPE, owner="contestant", command_nproc=made.ceiling)
    assert "at or above the runner's hard limit of 3000000" in e.value.fault.reason
    assert e.value.fault.evidence["accounting"] == "user_namespace"


def test_a_steps_own_ceiling_counts_every_task_of_the_uid_whatever_the_site_says(tmp_path, apptainer_on_path):
    # Set before any namespace is made, a sealed server's under pasta's among them: each namespace
    # records it as the limit of its parent level, so the uid's every task counts on any kernel.
    _, ctx = _ctx(tmp_path, RecordingLauncher(), site={"nproc_accounting": "user_namespace"})
    ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.pool_at_end.rule == CeilingRule(ceiling=ctx.process_ceiling().ceiling, accounting="uid",
                                                   parent_limit=None)


def test_a_count_of_the_uids_tasks_that_fails_neither_refuses_a_launch_nor_lets_a_failing_end_be_scored(
        tmp_path, apptainer_on_path, monkeypatch, caplog):
    # Before the launch the count only spares a start whose commands could fork nothing, so
    # uncounted the role starts. At the report the count is what rules starvation out, so
    # uncounted the role's failing end is infra, retried, never a crashed report and never the
    # contestant's.
    def boom(uid):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(stage_module, "uid_tasks", boom)
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    ceiling = ctx.process_ceiling().ceiling
    with caplog.at_level("ERROR", logger="hls.pipeline.stage"):
        ctx.start_container("srv", _quick(ctx, command=("false",)), SHAPE, owner="contestant",
                            command_nproc=ceiling)
        [outcome] = ctx.wait_roles()
    assert [s.name for s in launcher.launched] == ["srv"]
    assert (outcome.ending, outcome.exit_code) == ("exited", 1)
    assert (outcome.pool_at_end.others_tasks, outcome.pool_at_end.rule.ceiling) == (None, ceiling)
    fault = stage_fault([outcome])
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "could not be counted" in fault.reason
    assert fault.evidence["srv.others_tasks"] == "uncounted"
    assert sum("could not be counted" in r.getMessage() for r in caplog.records) == 2
    # The event says the count failed, which no pool, a role with no ceiling's, never reads as.
    [ended] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"]
    assert (ended["others_tasks"], ended["ceiling"]) == ("uncounted", ceiling) and ended["node_tasks"] > 0
    _, plain = _ctx(tmp_path / "plain", RecordingLauncher())
    plain.start_container("tool", _quick(plain, "tool"), SHAPE, owner="harness")
    plain.wait_roles()
    [tool] = [e for e in read_events(plain.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"]
    assert (tool["others_tasks"], tool["node_tasks"], tool["ceiling"]) == (None, None, None)


def test_every_contestant_launch_runs_under_the_stages_process_ceiling(tmp_path, apptainer_on_path):
    """The context reads the stage's ceiling once and gives it to every contestant launch, a
    container's step and a host role alike; no stage passes it, so none can start one without
    it. Harness launches keep the runner's limit."""
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    for start in (lambda: ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant"),
                  lambda: ctx.start_role("host", ["true"], SHAPE, owner="contestant"),
                  lambda: ctx.start_container("tool", _quick(ctx, "tool"), SHAPE, owner="harness"),
                  lambda: ctx.start_role("helper", ["true"], SHAPE, owner="harness")):
        start()
        ctx.wait_roles()                             # one at a time: the stage holds one role's memory
    ceiling = ctx.process_ceiling()
    launched = {s.name: s for s in launcher.launched}
    for name in ("srv", "host"):
        assert (launched[name].owner, launched[name].nproc) == ("contestant", ceiling.ceiling), name
    for name in ("tool", "helper"):
        assert (launched[name].owner, launched[name].nproc) == ("harness", None), name
    assert ctx.plan.settings.site.nproc_max is None and ceiling.ceiling <= DEFAULT_NPROC_MAX
    [event] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "process_ceiling"]
    assert event["ceiling"] == ceiling.ceiling


def test_a_stage_whose_ceiling_is_under_its_floor_starts_no_contestant_launch(tmp_path, apptainer_on_path,
                                                                             monkeypatch):
    # 20,000 tasks less 8 evaluations' reserves of 1,024 leaves 11,808, under the floor of 8 x 4,096.
    monkeypatch.setattr(stage_module, "hard_limit", lambda: 20_000)
    launcher = RecordingLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    for start in (lambda: ctx.start_container("srv", _quick(ctx), SHAPE, owner="contestant"),
                  lambda: ctx.start_role("host", ["true"], SHAPE, owner="contestant")):
        with pytest.raises(StageFault) as e:
            start()
        assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
        assert "can run on this node" in e.value.fault.reason
    assert [s for s in launcher.launched if s.owner == "contestant"] == []
    ctx.start_role("helper", ["true"], SHAPE, owner="harness")
    ctx.wait_roles()


def test_a_containers_logs_can_reach_the_attempt_only_scrubbed(tmp_path, apptainer_on_path):
    """A container whose logs may quote a secret's host, as the build sandbox's exec server
    quotes the start of each command, has them scrubbed in the attempt once it is reported."""
    scrubber = Scrubber({"relay_url": "https://relay.example/v1"})
    _, ctx = _ctx(tmp_path, RecordingLauncher(), scrubber=scrubber)
    spec = _quick(ctx, command=("sh", "-c", "echo dialled relay.example; echo relay.example >&2"))
    ctx.start_container("srv", spec, SHAPE, owner="harness", scrub_logs=True)
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited"
    for suffix in (".out", ".err"):
        text = (ctx.attempt_dir / "logs" / f"srv{suffix}").read_text()
        assert "relay.example" not in text and "<redacted:relay_url.host>" in text


# ---- seeding a role's writable copy, and publishing a tree a role wrote ---------------------

#: An engine cache entry's key in these tests.
CACHE_KEY = "H200-0123456789abcdef-fedcba9876543210"


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def test_seed_copies_the_published_entry_writable_in_a_harness_role_and_records_its_digest(tmp_path):
    plan, ctx = _ctx(tmp_path, RecordingLauncher())
    entry = _entry_of(ctx, "engine-cache", CACHE_KEY, {"vllm/graph.bin": b"g", "flashinfer/k.so": b"k"})
    dest = ctx.role_dir("capture") / "engine-cache"
    assert ctx.seed("engine-cache", CACHE_KEY, dest) == entry_sha256(entry)
    assert (dest / "vllm" / "graph.bin").read_bytes() == b"g" and (dest / "flashinfer" / "k.so").read_bytes() == b"k"
    assert not (dest / ".verified").exists()
    (dest / "vllm" / "graph.bin").write_bytes(b"recompiled")
    assert ctx.cache_entries == {f"engine-cache/{CACHE_KEY}": entry_sha256(entry)}
    [role] = ctx.launcher.launched
    stage = plan.stage("noop_a").shape
    assert role.name == f"copy-engine-cache-{entry_sha256(entry)[:12]}" and role.owner == "harness"
    assert role.shape == stage.model_copy(update={"gpus": 0, "gpu_kind": None, "disk_gib": 0,
                                                  "memory_gib": stage.memory_gib - 4})
    assert role.argv[3:9] == ("copy", "--entry", str(entry.path), "--expect", entry_sha256(entry), "--dest")


def test_seed_of_an_entry_nobody_published_makes_nothing(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    dest = ctx.role_dir("capture") / "engine-cache"
    assert ctx.seed("engine-cache", CACHE_KEY, dest) is None
    assert not dest.exists() and ctx.launcher.launched == [] and ctx.cache_entries == {}


@pytest.mark.parametrize("damage, says", [
    (lambda e: (e.path / "ENTRY.json").write_text("{not json"), "is corrupt: .* is not a valid entry manifest"),
    (lambda e: (e.path / "k.so").write_bytes(b"other"), "is corrupt: changed k.so"),
], ids=["manifest", "file"])
def test_seed_refuses_a_corrupt_entry_and_a_retry_would_too(tmp_path, damage, says):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    damage(_entry_of(ctx, "engine-cache", CACHE_KEY, {"k.so": b"k"}))
    dest = ctx.role_dir("capture") / "engine-cache"
    with pytest.raises(StageFault, match=f"cache entry engine-cache/{CACHE_KEY} {says}") as caught:
        ctx.seed("engine-cache", CACHE_KEY, dest)
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable, fault.evidence["entry"]) == ("infra", False,
                                                                           f"engine-cache/{CACHE_KEY}")
    assert not dest.exists() and ctx.cache_entries == {}


def test_publish_fills_the_entry_from_the_tree_in_a_harness_role(tmp_path):
    plan, ctx = _ctx(tmp_path, RecordingLauncher())
    files = {"vllm/graph.bin": b"g", "flashinfer/k.so": b"k"}
    source = _tree(ctx.role_dir("capture") / "engine-cache", files)
    got = ctx.publish("engine-cache", CACHE_KEY, source)
    assert got.published and got.entry == ctx.cache.lookup("engine-cache", CACHE_KEY)
    assert {f.path: f.sha256 for f in got.entry.manifest.files} == {p: sha256_bytes(b) for p, b in files.items()}
    assert got.entry.manifest.source == {"stage": "noop_a", "instance": plan.instance}
    [role] = ctx.launcher.launched
    assert role.name == "fill-engine-cache" and role.owner == "harness"
    assert role.argv[3:7] == ("fill", "--source", str(source), "--dest")
    # The tree it was filled from is the role's still.
    assert sorted(str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()) == sorted(files)


def test_a_published_tree_is_seeded_in_a_later_job_with_the_mtimes_its_role_left(tmp_path):
    """An engine's JIT decides by file times whether its kernels are built: the warm stage's
    publish and a capture stage's seed, each by its own role, keep every file's time."""
    plan, ctx = _ctx(tmp_path, RecordingLauncher())
    mtimes = {"flashinfer/fused_moe_90/kernel.so": 1_700_000_300_123_456_789,
              "flashinfer/fused_moe_90/kernel.cuda.o": 1_700_000_200_123_456_789,
              "flashinfer/fused_moe_90/kernel.cu": 1_700_000_100_123_456_789}
    source = _tree(ctx.role_dir("warm") / "engine-cache", {p: p.encode() for p in mtimes})
    for rel, mtime_ns in mtimes.items():
        os.utime(source / rel, ns=(mtime_ns, mtime_ns))
    published = ctx.publish("engine-cache", CACHE_KEY, source).entry
    assert {f.path: f.mtime_ns for f in published.manifest.files} == mtimes
    later = _context(plan, RecordingLauncher(), tmp_path / "node" / "hls-200")
    dest = later.role_dir("capture") / "engine-cache"
    assert later.seed("engine-cache", CACHE_KEY, dest, expect=entry_sha256(published)) == entry_sha256(published)
    assert {p: os.lstat(dest / p).st_mtime_ns for p in mtimes} == mtimes


def test_publish_of_an_entry_published_already_runs_no_role_and_keeps_that_entry(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    there = _entry_of(ctx, "engine-cache", CACHE_KEY, {"k.so": b"first"})
    got = ctx.publish("engine-cache", CACHE_KEY, _tree(tmp_path / "mine", {"k.so": b"mine"}))
    assert not got.published and entry_sha256(got.entry) == entry_sha256(there)
    assert ctx.launcher.launched == []


def test_two_stages_that_publish_one_entry_at_once_publish_it_once(tmp_path):
    plan = noop_plan(tmp_path)
    contexts = [_context(plan, RecordingLauncher(), tmp_path / "node" / f"hls-{n}") for n in (100, 200)]
    sources = [_tree(tmp_path / f"cache-{n}", {"k.so": f"compiled on {n}".encode()}) for n in (100, 200)]
    start, got = threading.Barrier(2), [None, None]

    def publish(i: int) -> None:
        start.wait()
        got[i] = contexts[i].publish("engine-cache", CACHE_KEY, sources[i])

    threads = [threading.Thread(target=publish, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert sorted(g.published for g in got) == [False, True]
    assert entry_sha256(got[0].entry) == entry_sha256(got[1].entry)
    winner = next(i for i in (0, 1) if got[i].published)
    assert (got[winner].entry.path / "k.so").read_bytes() == (sources[winner] / "k.so").read_bytes()


def test_a_tree_the_cache_would_not_publish_is_refused_and_nothing_is_published(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher())
    source = _tree(tmp_path / "cache", {"k.so": b"k"})
    (source / "link").symlink_to(source / "k.so")
    with pytest.raises(StageFault, match=f"the tree for cache entry engine-cache/{CACHE_KEY} was refused: .*link "
                                         "is a symlink") as caught:
        ctx.publish("engine-cache", CACHE_KEY, source)
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", False)
    assert ctx.cache.lookup("engine-cache", CACHE_KEY) is None


def test_a_publish_that_waits_past_the_deadline_for_another_filler_is_retried(tmp_path):
    _, ctx = _ctx(tmp_path, RecordingLauncher(), deadline_in=0.5)
    held = GenerationLock(ctx.store, f"cache-locks/engine-cache/{CACHE_KEY}", ttl_sec=120).try_acquire(
        Holder.here("another stage"))
    with pytest.raises(StageFault, match=f"engine-cache/{CACHE_KEY}: still filled by another stage") as caught:
        ctx.publish("engine-cache", CACHE_KEY, _tree(tmp_path / "cache", {"k.so": b"k"}))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and ctx.launcher.launched == []
    held.release()



def test_a_role_logs_cut_by_its_capper_hold_no_secret_cut_through(tmp_path):
    """A role given a secret prints it where its log's capper cuts: once on the head's last bytes,
    once on the tail's first. The published log holds neither piece."""
    launcher = RecordingLauncher(log_keep_bytes=4096)
    _, ctx = _ctx(tmp_path, launcher, secrets=SECRETS)
    value = SECRETS["HF_TOKEN"]
    code = ("import os, sys; v = os.environ['HF_TOKEN']; "
            "sys.stderr.write('x' * 4091 + v + 'm' * 10000 + v + 'z' * 4091 + '\\n')")
    ctx.start_role("dl", _python(code), SHAPE, secret_env=["HF_TOKEN"], owner="harness")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0
    data = (ctx.attempt_dir / "logs" / "dl.err").read_bytes()
    head, line, tail = data.partition(b"\n[hls: ")
    assert head == b"x" * 4091 and tail.split(b"]\n", 1)[1] == b"z" * 4091 + b"\n"
    assert value[:3].encode() not in data and value[-3:].encode() not in data
