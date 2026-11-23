from __future__ import annotations

import dataclasses
import json
import re
import socket
import stat
import sys
import time
from pathlib import Path

import pytest

import hls.pipeline.stage as stage_module
from hls.egress.denied_packages import DENIED_PACKAGES
from hls.egress.policy import NetworkPolicy, SealPolicy
from hls.pipeline.events import EventLog
from hls.pipeline.stage import StageContext, StageFault
from hls.spec.resources import ResourceShape
from hls.store.posix import PosixStore
from hls.substrate.apptainer import egress
from hls.substrate import seal as seal_mod
from hls.substrate.apptainer.argv import SEALED_DROP_CAPS, ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.seal import UPSTREAM_PROXY_VAR, PayloadEnding, client_env, ingress_port, payload_ending
from hls.substrate.base import JobState, RoleHandle, RoleSpec
from hls.substrate.container import ContainerSpec, Mount, SealPlan
from hls.substrate.faults import classify_role
from hls.substrate.memory_watch import MemoryWatch
from hls.substrate.nproc import node_evaluations
from hls.substrate.role_exec import ENV_FD
from tests.pipeline.fakes import NODE, LocalLauncher
from tests.pipeline.plans import noop_plan
from hls.substrate import ucounts
from tests.substrate import fake_apptainer, fake_pasta
from tests.substrate.sandboxes import IMAGE_SHA256, image_file

SHAPE = ResourceShape(cpus=2, memory_gib=2, disk_gib=1, time_sec=60)
#: A grading node's own per-uid limits, for the seals that serve, so no test depends on its host's.
GRADING_NODE = {u.name: {"max_inotify_instances": 128, "max_inotify_watches": 1_048_576}.get(u.name, 4_124_579)
                for u in ucounts.UCOUNTS}
POLICY = SealPolicy(network=NetworkPolicy.model_validate(
    {"domains": {"mode": "allow", "entries": []}, "packages": {"mode": "deny", "entries": list(DENIED_PACKAGES)}}))
INDEX = SealPlan(policy=POLICY, index=True, deny_list=tuple(DENIED_PACKAGES))
#: A seal that serves: a graded server's, or its probe's.
SERVED = SealPlan(policy=POLICY, index=False, ingress_port=8000)


class RecordingLauncher(LocalLauncher):
    """A local launcher that keeps each role's spec and env file as they were handed over."""

    def __init__(self, **knobs) -> None:
        super().__init__(gpus=2, **knobs)
        self.env_files: dict[str, tuple[int, str]] = {}

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        env_file = spec.env_file if spec.container is None else spec.container.env_file
        if env_file is not None:
            self.env_files[spec.name] = (stat.S_IMODE(env_file.stat().st_mode), env_file.read_text())
        return super().launch(spec, share_gpus_of=share_gpus_of)


@pytest.fixture
def seal_bin(tmp_path, monkeypatch):
    """Fake apptainer, pasta, iptables and interpreter on PATH; returns their directory. The
    fake interceptor listens in the test's own network namespace, so each test gets free
    ports for it rather than the seal's fixed ones, which another test's could hold."""
    with socket.socket() as transparent, socket.socket() as index:
        transparent.bind(("127.0.0.1", 0))
        index.bind(("127.0.0.1", 0))
        monkeypatch.setattr(seal_mod, "EGRESS_TRANSPARENT_PORT", transparent.getsockname()[1])
        monkeypatch.setattr(seal_mod, "EGRESS_INDEX_PORT", index.getsockname()[1])
    bin_dir = tmp_path / "bin"
    fake_apptainer.install(bin_dir)
    fake_pasta.install(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    # The seal resolves its firewall tools from here, not from the system's /usr/sbin.
    monkeypatch.setenv("HLS_SEAL_TOOL_PATH", str(bin_dir))
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(tmp_path / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(tmp_path / "pasta.log"))
    monkeypatch.setenv("FAKE_PYTHON_COUNTER", str(tmp_path / "counter"))
    monkeypatch.setenv("FAKE_APPTAINER_RECORD", str(tmp_path / "apptainer.json"))
    monkeypatch.delenv("FAKE_APPTAINER_FAIL", raising=False)
    return bin_dir


class OomLauncher(RecordingLauncher):
    """A recording launcher whose role steps end OUT_OF_MEMORY in accounting, with their commands
    left as they are, laid out as Slurm steps under *job*."""

    def __init__(self, job: Path) -> None:
        super().__init__(steps_under=job)

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        handle = super().launch(spec, share_gpus_of=share_gpus_of)
        handle.step_state = JobState.OUT_OF_MEMORY
        return handle


def _ctx(tmp_path: Path, bin_dir: Path, launcher: LocalLauncher, plan_lines: list[str], *,
         watch: MemoryWatch | None = None, **site: str) -> StageContext:
    (tmp_path / "pasta.plan").write_text("".join(f"{line}\n" for line in plan_lines))
    plan = noop_plan(tmp_path, site={"pasta_path": str(bin_dir / "pasta"), **site})
    plan = plan.model_copy(update={"toolchain": plan.toolchain.model_copy(update={"python": bin_dir / "python"})})
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("noop_a", 1, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage("noop_a"), attempt=1, launcher=launcher.use_plan(plan),
                        watch=watch, events=events, inputs={}, deadline=time.time() + 60,
                        scratch=tmp_path / "node" / "hls-100", job_id="100", runtime=ApptainerRuntime(),
                        images=ApptainerImages(tmp_path / "node" / "hls-100"))


def _spec(ctx: StageContext, tmp_path: Path, seal: SealPlan, *code: str, mounts: tuple[Mount, ...] = ()) -> ContainerSpec:
    assert ctx.scratch is not None
    return ContainerSpec(image=ctx.pin_file(image_file(ctx.scratch), sha256=IMAGE_SHA256),
                         command=(sys.executable, "-c", "; ".join(code or ("pass",))),
                         workdir=ctx.role_dir("srv") / "wd", mounts=mounts, seal=seal)


def test_a_seal_that_fails_before_the_container_starts_is_infra_even_for_a_contestant(tmp_path, seal_bin):
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["fail:1"])
    ctx.start_container("srv", _spec(ctx, tmp_path, INDEX), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "runtime_failed" and outcome.exit_code == 1
    fault = classify_role(outcome)
    assert fault is not None and fault.fault_class == "infra" and fault.retryable
    assert not (tmp_path / "apptainer.json").exists()
    assert ctx.seal_status("srv") is None


def test_a_sealed_contestant_that_fails_after_the_marker_is_its_result(tmp_path, seal_bin, monkeypatch):
    monkeypatch.setenv("FAKE_PROXY_ERRORS", "2")
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["pass"])
    ctx.start_container("srv", _spec(ctx, tmp_path, INDEX, "import sys", "sys.exit(1)"), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 1
    assert classify_role(outcome) is None
    status = ctx.seal_status("srv")
    assert status is not None and (status.total, status.mid_stream, status.oversized, status.policy_refused,
                                   status.fetches_refused) == (2, 0, (), (), 0)


@pytest.mark.parametrize("seal", [INDEX, SERVED], ids=["pip-step", "served-container"])
def test_a_container_that_fills_the_disk_once_started_still_ends_by_its_own_record(tmp_path, seal_bin, seal,
                                                                                  monkeypatch):
    """A container can fill the node's disk, where its seal's files are, once it started. Its
    ending and its interceptor's status were made at full size before it started and are only
    written over afterwards, so its own failure still reads as its exit, never as a seal that
    recorded nothing. The container here leaves the seal's directory unable to take a new file
    or a rename, as a full disk would refuse the new blocks those need."""
    monkeypatch.setattr(stage_module, "node_limits", lambda: GRADING_NODE)
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["run"])
    state = ctx.role_dir("srv") / "seal"
    code = ("import os, sys, time", f"os.chmod({str(state)!r}, 0o500)", "time.sleep(1)", "sys.exit(3)")
    held = ctx.grading_ucounts(SHAPE, None) if seal.ingress_port is not None else None
    ctx.start_container("srv", _spec(ctx, tmp_path, seal, *code), SHAPE, owner="contestant", seal_ucounts=held)
    try:
        [outcome] = ctx.wait_roles()
    finally:
        state.chmod(0o700)
    assert (outcome.ending, outcome.exit_code) == ("exited", 3) and classify_role(outcome) is None
    status = ctx.seal_status("srv")
    assert status is not None and status.heartbeat >= (state / "started").stat().st_mtime


def _slow_failure(ctx: StageContext, tmp_path: Path) -> ContainerSpec:
    return _spec(ctx, tmp_path, INDEX, "import sys, time", "time.sleep(1.5)", "sys.exit(1)")


def test_a_payload_that_outlives_its_interceptor_ends_runtime_failed(tmp_path, seal_bin, monkeypatch):
    # The interceptor ends after the start marker; the payload's failure may be the missing index.
    monkeypatch.setenv("FAKE_INTERCEPTOR", "quit_after:0.2")
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["run"])
    ctx.start_container("srv", _slow_failure(ctx, tmp_path), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("runtime_failed", 1), _err(ctx)
    assert "interceptor" in (outcome.detail or "")
    fault = classify_role(outcome)
    assert fault is not None and fault.fault_class == "infra" and fault.retryable


def test_a_payload_that_outlives_its_pasta_ends_runtime_failed(tmp_path, seal_bin):
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["die_after:0.2"])
    ctx.start_container("srv", _slow_failure(ctx, tmp_path), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("runtime_failed", 137), _err(ctx)
    fault = classify_role(outcome)
    assert fault is not None and fault.fault_class == "infra" and fault.retryable
    files = ctx.seal_files("srv")
    assert files is not None
    # The block outlives pasta; wait for it so nothing of the test runs on.
    pending = Path(f"{files.ended}.pending")
    for _ in range(300):
        if pending.exists():
            break
        time.sleep(0.05)
    assert pending.exists()


@pytest.mark.parametrize("limit, fault_class, retryable", [("own", "contestant", False), ("job", "infra", True),
                                                          ("unknown", "infra", True)])
def test_a_step_at_its_own_memory_limit_is_the_contestants_fault_whatever_the_seal_lost(tmp_path, seal_bin,
                                                                                    monkeypatch, limit,
                                                                                    fault_class, retryable):
    """The own-limit rule comes before the seal's record: when the kernel kills
    the interceptor because the step reached its own limit, the contestant's memory still decided
    it. When the job's limit was reached, or the counts cannot say whose, it stays infra."""
    job = tmp_path / "job_1"
    (job / "step_batch").mkdir(parents=True)
    for name in ("memory.events.local", "memory.events", "step_batch/memory.events"):
        (job / name).write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
    monkeypatch.setenv("FAKE_INTERCEPTOR", "quit_after:0.2")
    watch = MemoryWatch(enforce=False, job_dir=job) if limit != "unknown" else None
    ctx = _ctx(tmp_path, seal_bin, OomLauncher(job), ["run"], watch=watch)
    ctx.start_container("srv", _slow_failure(ctx, tmp_path), SHAPE, owner="contestant")
    assert ctx._roles["srv"].wait(30) == 1
    assert payload_ending(ctx.seal_files("srv")) == PayloadEnding(exit_code=1, interceptor_alive=False)
    (job / "step_1" / "memory.events").write_text(f"oom {int(limit == 'own')}\noom_kill 1\n")
    (job / "memory.events").write_text("oom 1\noom_kill 0\n")
    if limit == "job":
        (job / "memory.events.local").write_text("oom 1\noom_kill 0\n")
    [outcome] = ctx.wait_roles()
    fault = classify_role(outcome)
    assert fault is not None and (fault.fault_class, fault.retryable) == (fault_class, retryable), outcome
    if limit == "own":
        assert (outcome.ending, outcome.own_oom, outcome.job_oom) == ("exited", True, False)
    else:
        # Not certainly the contestant's memory, so the seal's record decides: the interceptor died.
        assert outcome.ending == "runtime_failed"


def test_a_seal_that_exits_otherwise_than_its_payload_ends_runtime_failed(tmp_path, seal_bin):
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["pass_exit:2"])
    ctx.start_container("srv", _spec(ctx, tmp_path, INDEX, "import sys", "sys.exit(1)"), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("runtime_failed", 2), _err(ctx)
    assert outcome.detail == "the seal exited 2, its payload 1"


def test_a_failure_reason_comes_from_the_attempt_that_failed(tmp_path, seal_bin, monkeypatch):
    # The first attempt's interceptor never binds; the second attempt's apptainer fails.
    monkeypatch.setenv("FAKE_INTERCEPTOR", "die_first")
    monkeypatch.setenv("FAKE_INTERCEPTOR_STARTS", str(tmp_path / "starts"))
    monkeypatch.setenv("FAKE_APPTAINER_FAIL", "1")
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["run", "run"])
    ctx.start_container("srv", _spec(ctx, tmp_path, INDEX), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("runtime_failed", 255), _err(ctx)
    assert outcome.detail == "FATAL:   fake apptainer failure requested"


def _err(ctx: StageContext) -> str:
    return (ctx.attempt_dir / "logs" / "srv.err").read_text()


def test_a_sealed_container_gets_the_index_and_its_step_gets_the_seal_files(tmp_path, seal_bin):
    launcher = RecordingLauncher()
    ctx = _ctx(tmp_path, seal_bin, launcher, ["pass"])
    ctx.start_container("srv", _spec(ctx, tmp_path, INDEX), SHAPE, env={"PLAIN": "1"}, owner="harness")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0
    role = launcher.launched[0]
    assert role.argv[:2] == ("bash", "-c") and role.argv[3] == "hls-seal"
    files = ctx.seal_files("srv")
    assert files is not None and files.status.parent == ctx.role_dir("srv") / "seal"
    # An index on an air-gapped policy: the seal refuses the container's own DNS. Every sealed
    # container that does not serve, a harness one sealed as a contestant's will be too, gets the
    # role's limits on the uid's per-namespace counts, the stage's one reading of them, for its
    # seal to write.
    limits = ucounts.ucounts_env({c.name: c.role for c in ctx.ucount_limits()})
    assert limits[ucounts.SEAL_UCOUNTS_VAR]
    assert role.env == {"HLS_PASTA": str(seal_bin / "pasta"), "HLS_SEAL_STARTED_FILE": str(files.started),
                        "HLS_PROXY_STATUS_FILE": str(files.status), "HLS_SEAL_ENDED_FILE": str(files.ended),
                        "HLS_SEAL_ENV_FD": str(ENV_FD), egress.CLOSED_DNS_VAR: "closed",
                        egress.SEAL_TOOL_PATH_VAR: str(seal_bin), **limits}
    mode, text = launcher.env_files["srv"]
    assert mode == 0o600 and "PIP_INDEX_URL=" in text and "PLAIN=" in text
    record = fake_apptainer.read_record(tmp_path / "apptainer.json")
    assert record["drop_caps"] == SEALED_DROP_CAPS
    assert {"PIP_INDEX_URL", "UV_INDEX_URL", "NO_PROXY", "PLAIN"} <= set(record["env_names"])
    assert not any(name.startswith("HLS_") for name in record["env_names"])
    assert stat.S_IMODE(files.status.parent.stat().st_mode) == 0o700
    # Apptainer ran inside the namespace pasta made, never on the host's network.
    assert record["in_pasta"] is True


def test_the_seal_step_names_the_firewall_tools_dir_only_when_the_runner_has_one(tmp_path, seal_bin,
                                                                                 monkeypatch):
    """HLS_SEAL_TOOL_PATH reaches the sealed step only from the runner's own environment: with it
    unset (production), the step carries none and the seal looks in the system sbin dirs."""
    launcher = RecordingLauncher()
    ctx = _ctx(tmp_path, seal_bin, launcher, ["pass"])
    ctx.start_container("srv", _spec(ctx, tmp_path, INDEX), SHAPE, owner="harness")
    ctx.wait_roles()
    assert launcher.launched[0].env[egress.SEAL_TOOL_PATH_VAR] == str(seal_bin)
    monkeypatch.delenv(egress.SEAL_TOOL_PATH_VAR)
    other = RecordingLauncher()
    (tmp_path / "unset").mkdir()
    # Without the variable the seal would resolve the system's own iptables; the fake pasta fails
    # before its child runs, so no firewall tool is ever executed, and only the step's declared
    # variables are checked here, not the role's ending.
    ctx2 = _ctx(tmp_path / "unset", seal_bin, other, ["fail:1"])
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(tmp_path / "unset" / "pasta.plan"))
    ctx2.start_container("srv", _spec(ctx2, tmp_path / "unset", INDEX), SHAPE, owner="harness")
    ctx2.wait_roles()
    assert egress.SEAL_TOOL_PATH_VAR not in other.launched[0].env


def test_a_served_container_s_port_is_read_once_its_seal_started(tmp_path, seal_bin, monkeypatch):
    monkeypatch.setattr(stage_module, "node_limits", lambda: GRADING_NODE)
    launcher = RecordingLauncher()
    ctx = _ctx(tmp_path, seal_bin, launcher, ["pass"])
    ctx.start_container("srv", _spec(ctx, tmp_path, SERVED), SHAPE, owner="contestant",
                        seal_ucounts=ctx.grading_ucounts(SHAPE, None))
    ctx.wait_roles()
    files = ctx.seal_files("srv")
    assert files is not None and launcher.launched[0].env["HLS_INGRESS_PORT_FILE"] == str(files.port)
    assert ingress_port(files) == 40001


def test_a_seal_that_serves_writes_a_graded_servers_limits_and_no_other_seal_takes_them(tmp_path, seal_bin,
                                                                                        monkeypatch):
    """A seal that serves, a graded server's or its probe's, writes the limits the server is held
    to (``grading_ucounts``), which the stage must give it; a seal that does not serve writes the
    stage's own share of the node and is given none. A start that breaks either is refused before
    anything is launched."""
    node = {**GRADING_NODE, "max_inotify_instances": 256}
    monkeypatch.setattr(stage_module, "node_limits", lambda: node)
    launcher = RecordingLauncher()
    ctx = _ctx(tmp_path, seal_bin, launcher, ["pass", "pass"])
    held = ctx.grading_ucounts(SHAPE, None)
    pip = dataclasses.replace(_spec(ctx, tmp_path, INDEX), workdir=ctx.role_dir("pip") / "wd")
    with pytest.raises(ValueError, match="a seal that serves writes the limits a graded server is held to"):
        ctx.start_container("srv", _spec(ctx, tmp_path, SERVED), SHAPE, owner="contestant")
    with pytest.raises(ValueError, match="only a seal that serves is given a graded server's limits"):
        ctx.start_container("pip", pip, SHAPE, owner="contestant", seal_ucounts=held)
    assert launcher.launched == []
    ctx.start_container("srv", _spec(ctx, tmp_path, SERVED), SHAPE, owner="contestant", seal_ucounts=held)
    ctx.wait_roles()
    ctx.start_container("pip", pip, SHAPE, owner="contestant")
    ctx.wait_roles()
    served, pip = launcher.launched
    # The served seal writes the site's declared share, its 128 instances among the servers of the
    # served shape the fake launcher's node holds; the pip seal this node's 256 among the
    # evaluations of the stage's size.
    servers = node_evaluations(NODE, gpus=SHAPE.gpus, cpus=SHAPE.cpus)
    stages = node_evaluations(NODE, gpus=ctx.stage.shape.gpus, cpus=ctx.stage.shape.cpus)
    assert 128 // servers != 256 // stages
    assert served.env[ucounts.SEAL_UCOUNTS_VAR] == ucounts.ucounts_env(held)[ucounts.SEAL_UCOUNTS_VAR]
    assert f"max_inotify_instances={128 // servers}" in served.env[ucounts.SEAL_UCOUNTS_VAR].split()
    assert f"max_inotify_instances={256 // stages}" in pip.env[ucounts.SEAL_UCOUNTS_VAR].split()


def test_a_sealed_container_may_not_see_its_seal_files(tmp_path, seal_bin):
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), [])
    for exposed in (ctx.role_dir("srv"), ctx.role_dir("srv") / "seal"):
        with pytest.raises(ValueError, match="seal's files"):
            ctx.start_container("srv", _spec(ctx, tmp_path, INDEX, mounts=(Mount(exposed, "/x"),)), SHAPE,
                                owner="harness")
    assert not (ctx.role_dir("srv") / "seal").exists() and not (ctx.role_dir("srv") / "container.env").exists()


@pytest.mark.parametrize("how", ["writable file inside", "link into it", "link to a directory holding it",
                                 "workdir holding it", "workdir inside it"])
def test_no_path_of_a_sealed_container_reaches_its_seal_files(tmp_path, seal_bin, how):
    """Each path is judged by where its links lead, and none may hold the seal's directory or
    lie inside it: through a mount of status.json a contestant could write its own count."""
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), [])
    role, seal = ctx.role_dir("srv"), ctx.role_dir("srv") / "seal"
    workdir, mounts = role / "wd", ()
    if how == "writable file inside":
        mounts = (Mount(seal / "status.json", "/x", read_only=False),)
    elif how == "link into it":
        (role / "evidence").symlink_to(seal / "status.json")
        mounts = (Mount(role / "evidence", "/x"),)
    elif how == "link to a directory holding it":
        (role / "out").mkdir()
        (role / "out" / "up").symlink_to(role)
        mounts = (Mount(role / "out" / "up", "/x"),)
    elif how == "workdir holding it":
        workdir = role
    else:
        workdir = seal / "wd"
    assert ctx.scratch is not None
    spec = ContainerSpec(image=ctx.pin_file(image_file(ctx.scratch), sha256=IMAGE_SHA256), command=("true",),
                         workdir=workdir, mounts=mounts, seal=INDEX)
    with pytest.raises(ValueError, match="seal's files"):
        ctx.start_container("srv", spec, SHAPE, owner="contestant")
    assert not seal.exists() and not (role / "container.env").exists()


def test_the_seals_client_variables_win_over_a_stages_own(tmp_path, seal_bin):
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["pass"])
    dump = ctx.role_dir("srv") / "env.json"
    spec = _spec(ctx, tmp_path, INDEX, "import json, os", f"json.dump(dict(os.environ), open({str(dump)!r}, 'w'))")
    ctx.start_container("srv", spec, SHAPE, env={"PIP_INDEX_URL": "http://elsewhere/simple/", "NO_PROXY": "",
                                                 "PLAIN": "1"}, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0
    env = json.loads(dump.read_text())
    assert env["PIP_INDEX_URL"] == client_env(INDEX)["PIP_INDEX_URL"] and env["NO_PROXY"] == "*"
    assert env["PLAIN"] == "1"


def test_seal_files_of_an_earlier_attempt_are_cleared_before_launch(tmp_path, seal_bin):
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["fail:1"])
    old = ctx.role_dir("srv") / "seal"
    old.mkdir(mode=0o700)
    (old / "status.json").write_text('{"upstream_errors": 5}')
    (old / "started").touch()
    ctx.start_container("srv", _spec(ctx, tmp_path, INDEX), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "runtime_failed"
    assert ctx.seal_status("srv") is None and not (old / "started").exists()


def test_the_sites_upstream_proxy_reaches_the_seal_under_its_own_name_and_never_the_container(tmp_path, seal_bin):
    launcher = RecordingLauncher()
    ctx = _ctx(tmp_path, seal_bin, launcher, ["pass"], egress_upstream_proxy="http://proxy.site:3128")
    ctx.start_container("srv", _spec(ctx, tmp_path, INDEX), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0
    assert launcher.launched[0].env[UPSTREAM_PROXY_VAR] == "http://proxy.site:3128"
    _, text = launcher.env_files["srv"]
    assert UPSTREAM_PROXY_VAR not in text and "proxy.site" not in text
    assert UPSTREAM_PROXY_VAR not in fake_apptainer.read_record(tmp_path / "apptainer.json")["env_names"]


def test_neither_the_seal_nor_apptainer_sees_what_a_site_prolog_adds(tmp_path, seal_bin, monkeypatch):
    """A site's task prolog adds BASH_ENV and apptainer variables to every step. BASH_ENV would
    make the seal's bash read a file first; an apptainer option variable would change the
    container. The step wrapper removes both before the seal starts."""
    sourced = tmp_path / "sourced"
    startup = tmp_path / "startup.sh"
    startup.write_text(f"touch {sourced}\n")
    monkeypatch.setenv("BASH_ENV", str(startup))
    monkeypatch.setenv("APPTAINER_BIND", "/shared")
    monkeypatch.setenv("APPTAINERENV_PLANTED", "1")
    monkeypatch.setenv("APPTAINER_CACHEDIR", str(tmp_path / "cache"))
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["run"])
    ctx.start_container("srv", _spec(ctx, tmp_path, INDEX), SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0
    assert not sourced.exists()
    record = fake_apptainer.read_record(tmp_path / "apptainer.json")
    # A container step loses even apptainer's cache directory, as it loses every variable apptainer reads.
    assert record["in_pasta"] is True and record["runtime_env_names"] == []
    assert "PLANTED" not in record["env_names"]


class _GatewayRuntime(ApptainerRuntime):
    """Apptainer's runtime as a stage reads it, but with a seal that holds memory beside its role, as
    a Docker site's gateway container does."""

    seal_memory_gib = 2


@pytest.mark.parametrize("room", [True, False])
def test_a_sealed_role_counts_its_seals_memory_in_the_stage_budget(tmp_path, seal_bin, room):
    """A sealed container fits a stage that holds its seal's memory beside it and the runner's,
    and is refused, before anything starts, by one that does not."""
    ctx = _ctx(tmp_path, seal_bin, RecordingLauncher(), ["pass"])
    ctx.runtime = _GatewayRuntime()
    runner = ctx.plan.settings.run.runner_memory_gib
    fits = SHAPE.memory_gib + _GatewayRuntime.seal_memory_gib + runner
    ctx.stage = ctx.stage.model_copy(update={"shape": ctx.stage.shape.model_copy(
        update={"memory_gib": fits if room else fits - 1})})
    spec = _spec(ctx, tmp_path, INDEX)
    if room:
        ctx.start_container("srv", spec, SHAPE, owner="harness")
        [outcome] = ctx.wait_roles()
        assert outcome.ending == "exited" and outcome.exit_code == 0
        return
    with pytest.raises(StageFault, match=re.escape(
            f"role srv (its seal holds 2 GiB beside it): the roles need {SHAPE.memory_gib + 2} GiB and the runner "
            f"{runner} GiB; the stage declares {fits - 1}")):
        ctx.start_container("srv", spec, SHAPE, owner="harness")
    assert "srv" not in [r.name for r in ctx.launcher.launched]
