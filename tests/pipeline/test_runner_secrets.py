from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

import pytest

import hls.substrate.runner as runner_mod
from hls.pipeline.events import read_events
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.planner import StageDecl, build_plan
from hls.pipeline.stage import STAGES, StageContext
from hls.spec.resources import ResourceShape
from hls.store.cache import ContentCache, entry_sha256
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.base import FaultRecord, RoleFailed
from hls.substrate.runner import EXIT_FAILED, EXIT_PUBLISHED, EXIT_TERMINATED
from hls.substrate.secrets import load_secrets
from tests.pipeline import probe_stage  # noqa: F401  (registers the probe stage)
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import CODE, noop_plan, settings
from tests.pipeline.test_runner import _runner, _status, _store, _sub
from tests.store.entries import entry_reads, populate, recording_reads

TOKEN = "hf-token-0123456789"
#: A secret of the file that no stage is granted.
OTHER = "other-token-0123456789"
SECRETS = f"""\
[secrets]
hf_token = "{TOKEN}"
other = "{OTHER}"

[grants]
"stage.probe".HF_TOKEN = "hf_token"
"""
SHAPE = ResourceShape(gpus=2, gpu_kind="H200", cpus=8, memory_gib=12, time_sec=600)


def _secrets_file(tmp_path: Path, text: str = SECRETS, mode: int = 0o600) -> Path:
    path = tmp_path / "site" / "secrets.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


def _contexts(monkeypatch) -> list[StageContext]:
    """Every stage context the runner makes from now on, in order."""
    made: list[StageContext] = []

    class Recorded(StageContext):
        def __init__(self, **kw) -> None:
            super().__init__(**kw)
            made.append(self)

    monkeypatch.setattr(runner_mod, "StageContext", Recorded)
    return made


def _probe_plan(tmp_path: Path, secrets_path: Path, *, first: bool = False, **params: object) -> Plan:
    """A plan whose one allocation runs the probe stage, after a stage "first" with no grants
    when *first* is set."""
    decls = [StageDecl("first", "probe", SHAPE)] if first else []
    decls.append(StageDecl("probe", "probe", SHAPE, params))
    plan = build_plan(decls=decls, task="noop", pipeline="probe", layout=InstanceLayout.rollout("noop", "p1"),
                      settings=settings(tmp_path, site={"secrets_path": str(secrets_path)}), code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-probe", known_stage_names=KNOWN_STAGE_NAMES | {"probe", "first"},
                      known_tasks=(), now=1000.0, secrets=load_secrets(secrets_path))
    return _store(plan, _sub(plan))


def test_a_stage_gets_its_grants_and_its_entries_and_images_reach_the_manifest(tmp_path):
    path = _secrets_file(tmp_path)
    key = "org+m@" + "a" * 40
    store = PosixStore(tmp_path / "store")
    entry = populate(ContentCache(store), "weights", key, lambda d: (d / "w").write_bytes(b"w"),
                     holder=Holder.here("test")).entry
    plan = _probe_plan(tmp_path, path, entry=f"weights/{key}@{entry_sha256(entry)}", image="c" * 64)
    assert plan.stage("probe").grants == {"HF_TOKEN": "hf_token"} and plan.store_root == store.root
    assert _runner(plan).run() == EXIT_PUBLISHED
    manifest, _ = read_manifest(store, plan.layout, "probe")
    assert manifest.provenance.cache_entries == {f"weights/{key}": entry_sha256(entry)}
    assert manifest.provenance.image_digests == {"probe": "c" * 64}
    seen = json.loads(store.read_bytes(manifest.outputs["seen"].key))
    digest = hashlib.sha256(TOKEN.encode()).hexdigest()
    assert seen == {"names": ["HF_TOKEN"], "digests": {"HF_TOKEN": digest}, "scrubbed": True,
                    "entry_path": str(tmp_path / "node" / "hls-100" / "100" / "weights" / key)}
    assert TOKEN not in (plan.plan_dir / f"rev-{plan.revision:08d}.json").read_text()


def test_the_runner_publishes_a_role_written_output_by_its_report_and_never_reads_it(tmp_path):
    """At publish the runner records each output by the digest its writer reported and reads
    only its size: a role's output, such as a packaged image, never passes through it."""
    plan = _probe_plan(tmp_path, _secrets_file(tmp_path), role_output="a large role output")
    store = PosixStore(plan.store_root)
    with recording_reads() as reads:
        assert _runner(plan).run() == EXIT_PUBLISHED
    manifest, _ = read_manifest(store, plan.layout, "probe")
    written = manifest.outputs["written"]
    assert written.sha256 == hashlib.sha256(b"a large role output").hexdigest() and written.size == 19
    assert written.key == plan.layout.attempt_file("probe", 1, "outputs/written.bin")
    assert entry_reads(reads, [store.path(written.key)]) == []
    seen = manifest.outputs["seen"]
    assert seen.sha256 == hashlib.sha256(store.read_bytes(seen.key)).hexdigest()


def test_a_stage_without_grants_gets_no_secrets_but_the_scrubber_knows_every_value_of_the_file(tmp_path,
                                                                                               monkeypatch):
    plan = _probe_plan(tmp_path, _secrets_file(tmp_path), first=True)
    assert plan.stage("first").grants == {}
    contexts = _contexts(monkeypatch)
    assert _runner(plan).run() == EXIT_PUBLISHED
    first, probe = contexts
    assert (first.stage.name, dict(first.secrets), dict(probe.secrets)) == ("first", {}, {"HF_TOKEN": TOKEN})
    for ctx in contexts:
        assert ctx.scrubber.scrub(f"{TOKEN} {OTHER}") == "<redacted:hf_token> <redacted:other>"
    store = PosixStore(plan.store_root)
    manifest, _ = read_manifest(store, plan.layout, "first")
    assert manifest.provenance.cache_entries == {} and manifest.provenance.image_digests == {}


def test_every_fault_record_the_runner_writes_is_scrubbed(tmp_path):
    plan = _probe_plan(tmp_path, _secrets_file(tmp_path), leak=True)
    assert _runner(plan).run() == EXIT_FAILED
    store = PosixStore(plan.store_root)
    attempt = store.read_bytes(plan.layout.attempt_file("probe", 1, "status.json")).decode()
    allocation = store.read_bytes(plan.layout.allocation_status(0, 1)).decode()
    events = store.read_bytes(plan.layout.submission_file(0, 1, "events.jsonl")).decode()
    assert "stage_ended" in events
    fault = _status(plan).fault
    assert "<redacted:hf_token>" in fault.reason and "<redacted:hf_token>" in fault.evidence["message"]
    for record in (attempt, allocation, events):
        assert TOKEN not in record and "<redacted:hf_token>" in record


def test_the_attempt_record_a_sigterm_writes_first_is_scrubbed(tmp_path, monkeypatch):
    # The SIGTERM writes the attempt's failed record with the fault the stage decided, and only
    # the stage's own thread writes it again later; a SIGKILL in between leaves the first.
    plan = _probe_plan(tmp_path, _secrets_file(tmp_path))
    runner = _runner(plan)
    store = PosixStore(plan.store_root)
    early: list[tuple[str, str]] = []

    def run(self, ctx):
        ctx.contestant_fault = lambda: FaultRecord(fault_class="contestant", reason=f"the server printed {TOKEN}",
                                                   retryable=False, evidence={"line": f"{OTHER} at the end"})
        runner.terminate()
        runner._shutdown.join(5)
        early.append((store.read_bytes(plan.layout.attempt_file("probe", 1, "status.json")).decode(),
                      store.read_bytes(plan.layout.allocation_status(0, 1)).decode()))
        return {}

    monkeypatch.setattr(STAGES["probe"], "run", run)
    assert runner.run() == EXIT_TERMINATED
    [(attempt, allocation)] = early
    assert '"state": "failed"' in attempt and "<redacted:hf_token>" in attempt and "<redacted:other>" in attempt
    for record in (attempt, allocation):
        assert TOKEN not in record and OTHER not in record


class _LeakyReportLauncher(LocalLauncher):
    """Starts roles whose grant report fails with a message that quotes a secret."""

    def launch(self, spec, *, share_gpus_of=None):
        handle = super().launch(spec, share_gpus_of=share_gpus_of)

        def report(timeout_sec):
            raise RoleFailed(f"the step printed {TOKEN} and {OTHER}")

        handle.report = report
        return handle


def test_every_event_the_runner_and_its_stages_write_is_scrubbed(tmp_path):
    # role_ended carries the role's detail beside the per-step OOM counts; the attempt's log
    # and the submission's log are both written through the runner's scrubber.
    plan = _probe_plan(tmp_path, _secrets_file(tmp_path), role=True)
    runner = _runner(plan, launcher=_LeakyReportLauncher(gpus=2))
    assert runner.run() == EXIT_FAILED
    store = PosixStore(plan.store_root)
    attempt = read_events(store.path(plan.layout.attempt_file("probe", 1, "events.jsonl")))
    [ended] = [e for e in attempt if e["kind"] == "role_ended"]
    assert ended["detail"] == "the step printed <redacted:hf_token> and <redacted:other>"
    assert {"job_ooms_at_start", "job_ooms", "step_ooms", "own_oom", "job_oom"} <= set(ended)
    [failed] = [e for e in attempt if e["kind"] == "role_launch_failed"]
    assert failed["error"] == ended["detail"]
    # The submission's own log goes through the same scrubber, which the run loaded.
    runner.events.emit("probe", text=f"{TOKEN} {OTHER}")
    submission = read_events(store.path(plan.layout.submission_file(0, 1, "events.jsonl")))
    assert submission[-1]["text"] == "<redacted:hf_token> <redacted:other>"
    for log in (plan.layout.attempt_file("probe", 1, "events.jsonl"), plan.layout.submission_file(0, 1, "events.jsonl")):
        text = store.read_bytes(log).decode()
        assert TOKEN not in text and OTHER not in text


def test_a_fault_of_the_runner_itself_is_scrubbed_in_the_allocation_status(tmp_path, monkeypatch):
    plan = _probe_plan(tmp_path, _secrets_file(tmp_path))

    def broken_watch(**kw):
        raise RuntimeError(f"cannot watch with {TOKEN}")

    monkeypatch.setattr(runner_mod, "MemoryWatch", broken_watch)
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert "<redacted:hf_token>" in fault.reason and "<redacted:hf_token>" in fault.evidence["message"]
    assert TOKEN not in PosixStore(plan.store_root).read_bytes(plan.layout.allocation_status(0, 1)).decode()


def test_the_runners_log_is_scrubbed_tracebacks_and_warnings_included(tmp_path, monkeypatch):
    plan = _probe_plan(tmp_path, _secrets_file(tmp_path))

    def broken_watch(**kw):
        raise RuntimeError(f"cannot watch with {TOKEN}")

    monkeypatch.setattr(runner_mod, "MemoryWatch", broken_watch)
    runner = _runner(plan)
    stream = io.StringIO()
    root = logging.getLogger()
    level = root.level
    # The scrubber is looked up when a record is formatted: the runner replaces it after the
    # handler is installed, when it reads the secrets file.
    handler = runner_mod.configure_logging(lambda: runner.scrubber, stream)
    try:
        assert runner.run() == EXIT_FAILED
        warnings.warn(f"a warning that quotes {OTHER}", stacklevel=1)
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
        logging.captureWarnings(False)
    text = stream.getvalue()
    assert "the runner failed" in text and "Traceback" in text and "RuntimeError: cannot watch with" in text
    assert "<redacted:hf_token>" in text and "<redacted:other>" in text
    assert TOKEN not in text and OTHER not in text


def test_an_unusable_secrets_file_is_found_before_a_pinned_input_that_may_still_appear(tmp_path):
    path = _secrets_file(tmp_path)
    pinned = {InstanceLayout.rollout("noop", "other").manifest("package"): "d" * 64}
    plan = build_plan(decls=[StageDecl("probe", "probe", SHAPE, inputs=pinned)], task="noop", pipeline="probe",
                      layout=InstanceLayout.rollout("noop", "p1"),
                      settings=settings(tmp_path, site={"secrets_path": str(path)}), code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-probe", known_stage_names=KNOWN_STAGE_NAMES | {"probe"},
                      known_tasks=(), now=1000.0, secrets=load_secrets(path))
    plan = _store(plan, _sub(plan))
    path.chmod(0o640)
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert "secrets file" in fault.reason and not fault.retryable


def test_a_secrets_file_others_can_read_is_refused_and_never_retried(tmp_path):
    path = _secrets_file(tmp_path)
    plan = _probe_plan(tmp_path, path)
    path.chmod(0o640)
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert fault.fault_class == "infra" and not fault.retryable and "secrets file" in fault.reason
    assert TOKEN not in fault.reason
    assert PosixStore(plan.store_root).list_children(plan.layout.stage("probe")) == []


def test_a_token_pasted_as_a_grants_right_hand_side_never_reaches_the_status(tmp_path):
    # The file changes after planning: the token now stands where its secret's name belongs.
    path = _secrets_file(tmp_path)
    plan = _probe_plan(tmp_path, path)
    _secrets_file(tmp_path, SECRETS.replace('HF_TOKEN = "hf_token"', f'HF_TOKEN = "{TOKEN}"'))
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert fault.fault_class == "infra" and not fault.retryable and "grant stage.probe.HF_TOKEN" in fault.reason
    store = PosixStore(plan.store_root)
    for key in (plan.layout.allocation_status(0, 1), plan.layout.submission_file(0, 1, "events.jsonl")):
        assert TOKEN not in store.read_bytes(key).decode()


@pytest.mark.parametrize("tail", ["x = " + "[" * 2000 + "]" * 2000 + "\n", "x = " + "1" * 5000 + "\n"],
                         ids=["nested", "digits"])
def test_a_secrets_file_the_parser_cannot_take_is_refused_and_never_retried(tmp_path, tail):
    path = _secrets_file(tmp_path)
    plan = _probe_plan(tmp_path, path)
    _secrets_file(tmp_path, SECRETS + tail)
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert fault.fault_class == "infra" and not fault.retryable and "not valid TOML" in fault.reason
    assert TOKEN not in PosixStore(plan.store_root).read_bytes(plan.layout.allocation_status(0, 1)).decode()


@pytest.mark.parametrize("grants", [True, False])
def test_a_secrets_path_that_resolves_into_the_store_on_the_node_is_refused_and_never_read(tmp_path, monkeypatch,
                                                                                          grants):
    # The path was outside the store when the plan was made; the link it goes through leads into
    # the store where the runner resolves it.
    (tmp_path / "site").mkdir()
    (tmp_path / "via").symlink_to(tmp_path / "site")
    _secrets_file(tmp_path)
    linked = tmp_path / "via" / "secrets.toml"
    plan = _probe_plan(tmp_path, linked) if grants else _store(
        noop_plan(tmp_path, site={"secrets_path": str(linked)}), _sub(noop_plan(tmp_path)))
    inside = PosixStore(plan.store_root).root / "site"
    shutil.move(tmp_path / "site", inside)
    (tmp_path / "via").unlink()
    (tmp_path / "via").symlink_to(inside)
    reads = []
    monkeypatch.setattr(runner_mod, "load_secrets", lambda p: reads.append(p) or load_secrets(p))
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert fault.fault_class == "infra" and not fault.retryable
    assert f"site.secrets_path {linked} is in the store" in fault.reason and reads == []


def test_a_grant_of_a_secret_the_file_no_longer_holds_is_refused_before_any_stage_runs(tmp_path):
    path = _secrets_file(tmp_path)
    plan = _probe_plan(tmp_path, path, first=True)
    _secrets_file(tmp_path, '[secrets]\nother = "other-token-0123456789"\n')
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert not fault.retryable and "probe" in fault.reason and "hf_token" in fault.reason
    store = PosixStore(plan.store_root)
    assert store.list_children(plan.layout.stage("first")) == []
    assert store.list_children(plan.layout.stage("probe")) == []


def test_the_secrets_file_is_not_read_when_no_stage_has_grants(tmp_path):
    path = _secrets_file(tmp_path, "this is not toml", mode=0o644)
    plan = noop_plan(tmp_path, site={"secrets_path": str(path)})
    assert all(not s.grants for s in plan.stages)
    plan = _store(plan, _sub(plan))
    assert _runner(plan).run() == EXIT_PUBLISHED
    assert os.stat(path).st_mode & 0o777 == 0o644


def test_the_secrets_file_is_not_read_when_the_stages_with_grants_are_published(tmp_path, monkeypatch):
    path = _secrets_file(tmp_path)
    plan = _probe_plan(tmp_path, path)
    assert _runner(plan).run() == EXIT_PUBLISHED
    reads = []
    monkeypatch.setattr(runner_mod, "load_secrets", lambda p: reads.append(p) or load_secrets(p))
    plan = _store(plan, _sub(plan).model_copy(update={"status": "absent"}), _sub(plan, 2, job_id="101"))
    assert _runner(plan, n=2, job_id="101").run() == EXIT_PUBLISHED
    assert reads == [] and _status(plan, n=2).fault is None
