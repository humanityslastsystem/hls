"""Finalize's secret scan over what an agent build stored: the agent's route grants are
contestant-facing, so a value of theirs that the agent printed quarantines the files it is in and
the instance is scored; the transcript, which the agent wrote, is read as stored
(``AgentBuildStage.raw_outputs``). The source bundle and the snapshots finalize does not read
(``AgentBuildStage.unread_files``): the source scan's record of the bundle's objects covers the
bundle, and a container no record covers is quarantined whole."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import pytest

from hls.agent.commit import PARTIAL_DIR
from hls.arms import ArmRecord, Route
from hls.pipeline.plan import Plan
from hls.pipeline.stage import StageFault
from hls.settings import RunSettings
from hls.spec.resources import ResourceShape
from hls.stages.agent_build import AgentBuildStage
from hls.stages.finalize import FOUND_FILE, SCAN, FinalizeStage, contestant_facing, read_scan_report
from hls.stages.score import ScoreStage
from hls.store.posix import PosixStore
from tests.finalize.published import (ROLLOUT, TREE_SCAN, context, make_plan, publish, publish_instance,
                                      run_and_publish, with_stages, write)
from tests.trace.traces import drive_trace, passing_verdict
from tests.unit.brief_task import PIP_LIMITS

#: A relay key the agent build's route grants: a secret the agent holds.
RELAY = "relay-key-0123456789abcdef"
#: A secret no stage hands to a contestant.
SECRET = "hf-token-0123456789abcdef"
REGISTRY = {"hf_token": SECRET, "relay": RELAY}
ROW = f"runs/fin/index/{ROLLOUT}.json"
ARM = ArmRecord(name="k3", harness="claude-code", model="k3", context_window=200_000, output_limit=64_000,
                effort="max", route="kimi", route_table=Route(credential="relay", auth="bearer"))
SANDBOX = ResourceShape(cpus=2, memory_gib=16, disk_gib=8, time_sec=600)
HARNESS = ResourceShape(cpus=1, memory_gib=2, time_sec=600)
PARAMS = {"task": {"module": "fin_goodput.task"}, "arm": ARM.model_dump(), "sandbox_image": "agent-cap",
          "sandbox_held": None,
          "sandbox_shape": SANDBOX.model_dump(), "served_shape": SANDBOX.model_dump(),
          "snapshot_shape": HARNESS.model_dump(),
          "commit_shape": HARNESS.model_copy(update={"memory_gib": 4}).model_dump(),
          "extract_shape": HARNESS.model_copy(update={"memory_gib": 4}).model_dump(), "data_copies": {},
          "hard_sec": 1100, "staging_sec": 0,
          "collect_allowance_sec": RunSettings.model_fields["collect_allowance_sec"].default,
          "network": {"domains": {"mode": "allow", "entries": []}, "packages": {"mode": "deny", "entries": ["vllm"]}},
          "package_pip": PIP_LIMITS.model_dump()}


def _with_agent_build(plan: Plan) -> Plan:
    """*plan* with this instance's agent_build stage first, granted its arm's route."""
    build = plan.stage("score").model_copy(update={"name": "agent_build", "kind": AgentBuildStage.kind,
                                                   "params": PARAMS, "grants": {"ANTHROPIC_AUTH_TOKEN": "relay"}})
    return with_stages(plan, build)


def _built(tmp_path, checkout) -> tuple[Plan, PosixStore]:
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    trace = drive_trace()
    publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace))
    run_and_publish(store, plan, "score", ScoreStage)
    return _with_agent_build(plan), store


def _finalize(plan: Plan, store: PosixStore):
    ctx = context(store, plan, "finalize", secrets=REGISTRY)
    return ctx, FinalizeStage().run(ctx)


def _quarantined(store: PosixStore, outputs: dict[str, str]) -> dict:
    return json.loads(store.read_bytes(outputs["quarantine"]))


def test_the_agents_route_grants_are_marked_contestant_facing():
    plan = _with_agent_build(make_plan(Path("/x"), Path("/c"), "fin_goodput.task"))
    assert contestant_facing(plan, REGISTRY) == {"relay"}
    assert AgentBuildStage.runs_contestant_code and AgentBuildStage.grants_reach_contestant


def _read(store: PosixStore, ctx) -> dict[str, str]:
    """Each file finalize's scan read, by its store key, with the digest of its bytes."""
    report = read_scan_report(ctx.role_dir(SCAN) / FOUND_FILE, store.root)
    assert report is not None
    return report.read


def test_a_bundle_no_scan_read_is_quarantined_whole_unread_and_the_build_is_scored(tmp_path, checkout):
    """Finalize does not read the bundle: one of our own secrets in its stored bytes, which would
    be the harness's leak in a file it read, is not looked for there. No record of the source scan
    read the bundle's objects, so it is quarantined whole, with no secret named, kept out of any
    export, and the instance is scored."""
    plan, store = _built(tmp_path, checkout)
    bundle = write(store, plan, "agent_build", "source.bundle", b"# v2 git bundle\n" + SECRET.encode())
    ctx, outputs = _finalize(plan, store)
    assert _quarantined(store, outputs) == {"files": {bundle: []}}
    assert bundle not in _read(store, ctx)
    row = json.loads(store.read_bytes(ROW))
    assert (row["status"], row["quarantined"]) == ("ok", [bundle])


def test_a_bundle_the_source_scan_read_is_not_read_again(tmp_path, checkout):
    """The source scan's record of the bundle's objects covers the bundle: finalize reads none of
    its bytes, so one of our own secrets in them is the object scan's to find (its record here found
    none), and the bundle is neither read here nor quarantined. The transcript beside it is read."""
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    trace = drive_trace()
    publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace),
                     bundle=b"# v2 git bundle\n" + SECRET.encode(), tree_scan=TREE_SCAN)
    run_and_publish(store, plan, "score", ScoreStage)
    plan = _with_agent_build(plan)
    log = write(store, plan, "agent_build", "transcript/agent.log", b"nothing secret here\n")
    ctx, outputs = _finalize(plan, store)
    bundle = plan.layout.attempt_file("agent_build", 1, "outputs/source.bundle")
    assert "quarantine" not in outputs
    read = _read(store, ctx)
    assert bundle not in read and log in read


def test_a_partial_bundle_a_killed_commit_point_left_is_quarantined_and_never_read(tmp_path, checkout):
    """A commit point killed outright, and its stage with it, leaves its partial bundle under the
    outputs' ``PARTIAL_DIR``: history no record names and no scan reads inside. Finalize does not
    read it either, so one of our own secrets in it is not looked for, and it is quarantined whole,
    as a container the attempt left unpublished."""
    plan, store = _built(tmp_path, checkout)
    partial = _plant(store, plan, f"{PARTIAL_DIR}/source.bundle-7", b"PACK" + SECRET.encode() + bytes(4 * 2**20))
    ctx, outputs = _finalize(plan, store)
    assert _quarantined(store, outputs) == {"files": {partial: []}}
    assert partial not in _read(store, ctx)


def test_a_snapshot_is_quarantined_whole_and_never_read(tmp_path, checkout):
    """A snapshot, left only when the commit point bundled nothing, is a pack no scan reads inside:
    finalize does not read it either, so one of our own secrets in it is not looked for, and it is
    quarantined whole, listed by its path, and never exported."""
    plan, store = _built(tmp_path, checkout)
    snapshot = plan.layout.attempt_file("agent_build", 1, "snapshots/main-1.bundle")
    store.write_bytes(snapshot, b"# v2 git bundle\n" + SECRET.encode() + bytes(4 * 2**20))
    ctx, outputs = _finalize(plan, store)
    assert _quarantined(store, outputs) == {"files": {snapshot: []}}
    assert snapshot not in _read(store, ctx)
    assert json.loads(store.read_bytes(ROW))["quarantined"] == [snapshot]


def _git(cwd: Path, *args: str) -> None:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(cwd), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@x", "GIT_COMMITTER_NAME": "a",
           "GIT_COMMITTER_EMAIL": "a@x"}
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


def test_a_key_the_agent_committed_is_found_by_no_read_of_finalizes(tmp_path, checkout):
    """Finalize reads none of the bundle, and a real git bundle deflates every object anyway. What
    finds a key the agent committed is the source_scan stage's read of every object of the bundle
    and of its tree (``package-scan.json``); a bundle no published scan read, as here, is kept out of
    any export as unscanned, with no secret named."""
    plan, store = _built(tmp_path, checkout)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "server.py").write_text(f"KEY = {RELAY!r}\n")
    _git(repo, "add", "server.py")
    _git(repo, "commit", "-q", "-m", "v1")
    _git(repo, "bundle", "create", str(tmp_path / "source.bundle"), "main")
    data = (tmp_path / "source.bundle").read_bytes()
    assert RELAY.encode() not in data
    bundle = write(store, plan, "agent_build", "source.bundle", data)
    _, outputs = _finalize(plan, store)
    assert _quarantined(store, outputs) == {"files": {bundle: []}}
    assert json.loads(store.read_bytes(ROW))["quarantined"] == [bundle]


def test_a_transcript_file_starting_with_gzip_magic_does_not_fail_the_scan(tmp_path, checkout):
    """The agent wrote its logs, gzip's magic or a cut stream among them: the transcript is read
    as stored, so the scan finds a value after the magic and never fails on the stream."""
    plan, store = _built(tmp_path, checkout)
    cut = write(store, plan, "agent_build", "transcript/agent.log", gzip.compress(b"x" * 64, mtime=0)[:12] +
                RELAY.encode())
    magic = write(store, plan, "agent_build", "transcript/sessions/s.jsonl", b"\x1f\x8b" + RELAY.encode())
    _, outputs = _finalize(plan, store)
    assert _quarantined(store, outputs) == {"files": {cut: ["relay"], magic: ["relay"]}}


def test_a_key_no_agent_holds_in_the_agents_files_refuses_the_instance(tmp_path, checkout):
    """Any other secret in a file of the build is the harness's own leak."""
    plan, store = _built(tmp_path, checkout)
    leaked = write(store, plan, "agent_build", "transcript/agent.log", SECRET.encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == f"the harness leaked secret values into published files: {leaked} (hf_token)"


def test_the_agents_files_are_read_as_stored_never_decompressed(tmp_path, checkout):
    """A transcript file that is a whole gzip stream is read as the bytes it is, as finalize reads
    every file a stage declares raw: a contestant chooses those bytes, and reading them decompressed
    would let it choose what the scan spends on them. A bundle of the same bytes is not read."""
    plan, store = _built(tmp_path, checkout)
    write(store, plan, "agent_build", "transcript/agent.log.gz", gzip.compress(RELAY.encode(), mtime=0))
    bundle = write(store, plan, "agent_build", "source.bundle", gzip.compress(RELAY.encode(), mtime=0))
    index = write(store, plan, "agent_build", "transcript/INDEX.json", b'{"files": [], "skipped": []}')
    publish(store, plan, "agent_build", {"source_bundle": bundle, "transcript": index})
    _, outputs = _finalize(plan, store)
    # Nothing found as stored; the bundle no scan read is kept out as unscanned.
    assert _quarantined(store, outputs) == {"files": {bundle: []}}


def _plant(store: PosixStore, plan: Plan, name: str, data: bytes) -> str:
    """Write *data* into the agent build's ``outputs/<name>`` as its transcript role writes the
    copy, straight to the file system, whatever the store's keys allow; return its key."""
    key = f"{plan.layout.attempt_file('agent_build', 1, 'outputs')}/{name}"
    path = store.path(plan.layout.attempt_file("agent_build", 1, "outputs")) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return key


@pytest.mark.parametrize("name", [f"transcript/logs/{RELAY}.txt", f"transcript/logs/{RELAY}/notes.txt",
                                  f"transcript/logs/.{RELAY}", f"transcript/logs/.hidden/{RELAY}"],
                         ids=["file", "directory", "dot-file", "under-a-dot-directory"])
def test_a_key_in_a_transcript_files_path_quarantines_it_and_the_path_is_never_shown(tmp_path, checkout, name):
    """The agent names its log files, and the transcript keeps their names: a value in a file's
    path, its own name or a directory's, a dot-file's among them, is found as a value in its bytes
    is, and quarantines the file under the same rule. Every record that names the file names it
    with the value replaced."""
    plan, store = _built(tmp_path, checkout)
    key = _plant(store, plan, name, b"nothing secret here\n")
    _, outputs = _finalize(plan, store)
    quarantined = _quarantined(store, outputs)["files"]
    shown = key.replace(RELAY, "<redacted:relay>")
    assert quarantined[shown] == ["relay"]
    assert all(RELAY not in path for path in quarantined)
    row = store.read_bytes(ROW).decode()
    assert RELAY not in row and shown in json.loads(row)["quarantined"]
    assert RELAY not in store.read_bytes(outputs["quarantine"]).decode()


def test_a_key_no_agent_holds_in_a_file_path_refuses_the_instance_naming_it_scrubbed(tmp_path, checkout):
    plan, store = _built(tmp_path, checkout)
    leaked = _plant(store, plan, f"transcript/logs/{SECRET}.txt", b"plain\n")
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False) and SECRET not in fault.reason
    assert leaked.replace(SECRET, "<redacted:hf_token>") in fault.reason
