"""The export stage: what of each instance leaves and what stays behind, the scan and finalize's
judgment applied instance by instance, the copy and its checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from hls.pipeline.stage import StageFault
from hls.stages.finalize import FinalizeStage
from hls.stages.score import ScoreStage
from hls.substrate.base import FaultRecord
from hls.stages import export as export_stage
from hls.stages.export import (CLAIM_FILE, EXPORT_FILE, ClaimRecord, CopiedFile, ExportRecord, ExportStage,
                               InstanceExport, claim_destination, make_partial, release_claim)
from hls.stages.finalize import OUTSIDE_BUNDLE
from hls.stages.package import AGENT_BUILD_STAGE
from hls.store.export_copy import Made, open_directory
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE, PackageScanArea, PackageScanFinding
from tests.export.instances import (BOTH, BUNDLE, RELAY, SECRET, SOURCE_SCAN, agent_built, destination, ended,
                                    export_plan, facing, failed, finalized, graded_package, package_warmer,
                                    run_export, scan_record, with_package)
from tests.finalize.published import (ROLLOUT, SIF_SHA, context, echoed_trace, make_plan, publish,
                                      publish_instance, run_and_publish, write)
from tests.finalize.test_agent_build_scan import ARM
from tests.pipeline.fakes import LocalLauncher
from tests.trace.traces import drive_trace, passing_verdict

FAKE_COPY = Path(__file__).with_name("fake_copy.py")
RACING_COPY = Path(__file__).with_name("fake_copy_then_race.py")
#: The fixture rollout's index row.
ROW = f"runs/fin/index/{ROLLOUT}.json"
WORKING = "a working record of the orchestrators or the runners"
UNSCANNED_WHY = "quarantined: unscanned: no package scan covers it"


def _record(dest: Path) -> ExportRecord:
    return ExportRecord.model_validate_json((dest / EXPORT_FILE).read_bytes())


def _only(record: ExportRecord) -> InstanceExport:
    [done] = record.instances
    return done


def _behind(done: InstanceExport) -> dict[str, str]:
    return {f.path: f.why for f in done.left_behind}


def _copied(done: InstanceExport) -> dict[str, CopiedFile]:
    return {f.path: f for f in done.copied}


def _nothing_beside(dest: Path) -> bool:
    """Whether the destination's directory holds nothing: no destination and no partial copy."""
    return [p.name for p in dest.parent.iterdir()] == []


def _leak(tmp_path, checkout, store, instances, dest, **kwargs) -> StageFault:
    with pytest.raises(StageFault) as e:
        run_export(store, export_plan(tmp_path, checkout, instances, dest, **kwargs))
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert _nothing_beside(dest)
    return fault


def test_an_export_copies_what_finalize_published(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    root = plan.instance
    store.write_bytes(f"{root}/allocations/0/submissions/1/slurm-7.out", b"runner log\n")
    store.write_bytes(f"{root}/orchestrator/gen-00000001/events.jsonl", b"{}\n")
    dest = destination(tmp_path)
    ctx, outputs = run_export(store, export_plan(tmp_path, checkout, [root], dest))
    assert outputs == {"export": ctx.output_key(EXPORT_FILE)}
    assert store.read_bytes(outputs["export"]) == (dest / EXPORT_FILE).read_bytes()
    record = _record(dest)
    assert (record.export, record.hls_commit, record.secrets_searched) == (ctx.plan.instance, "a" * 40,
                                                                           ["hf_token", "relay"])
    assert record.oracle is None
    done = _only(record)
    assert (done.instance, done.outcome, done.finalized) == (root, "succeeded", True)
    copied = _copied(done)
    for stage in ("provision", "images", "package", "deploy_drive", "verify", "score", "finalize"):
        manifest, _ = read_manifest(store, plan.layout, stage)
        assert f"{root}/stages/{stage}/MANIFEST.json" in copied
        for ref in manifest.outputs.values():
            assert copied[ref.key].sha256 == ref.sha256
    assert f"{root}/plan/rev-00000001.json" in copied and ROW in copied
    for f in done.copied:
        assert hashlib.sha256((dest / f.path).read_bytes()).hexdigest() == f.sha256
        assert (dest / f.path).read_bytes() == store.read_bytes(f.path)
    assert _behind(done) == {f"{root}/allocations/0/submissions/1/slurm-7.out": WORKING,
                             f"{root}/orchestrator/gen-00000001/events.jsonl": WORKING}
    assert not (dest / root / "allocations").exists()
    # The copy's own directory became the destination.
    assert [p.name for p in dest.parent.iterdir()] == [dest.name]


def test_an_earlier_plan_revision_stays_behind(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    later = plan.model_copy(update={"revision": 2})
    store.create(plan.layout.plan_revision(2), later.model_dump_json().encode())
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    done = _only(_record(dest))
    first, second = plan.layout.plan_revision(1), plan.layout.plan_revision(2)
    assert _behind(done)[first] == "an earlier revision of the plan" and not (dest / first).exists()
    assert second in _copied(done)


def test_only_the_attempt_each_stage_counted_leaves(tmp_path, checkout):
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    stale = plan.layout.attempt_file("score", 1, "outputs/score.json")
    store.write_bytes(stale, b'{"value": 0}')
    status = plan.layout.attempt_file("score", 1, "status.json")
    store.write_bytes(status, b"{}")
    plan, store = finalized(tmp_path, checkout, plan=plan, score_attempt=2)
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    done = _only(_record(dest))
    why = "attempt 1 of score, which the instance did not count"
    assert _behind(done)[stale] == why and _behind(done)[status] == why
    assert plan.layout.attempt_file("score", 2, "outputs/score.json") in _copied(done)
    assert not (dest / stale).exists()


def test_an_export_leaves_quarantined_files_behind(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout, plan=facing(make_plan(tmp_path, checkout, "fin_goodput.task")),
                            trace=echoed_trace(RELAY))
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    done = _only(_record(dest))
    assert _behind(done)[chunk] == "quarantined: holds relay"
    assert chunk not in _copied(done) and not (dest / chunk).exists()
    # The quarantine record names paths and secrets' names, and leaves.
    quarantine = read_manifest(store, plan.layout, "finalize")[0].outputs["quarantine"].key
    assert quarantine in _copied(done)
    assert RELAY.encode() not in (dest / EXPORT_FILE).read_bytes()


def test_what_finalize_quarantined_stays_behind_though_the_secret_changed_since(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout, plan=facing(make_plan(tmp_path, checkout, "fin_goodput.task")),
                            trace=echoed_trace(RELAY))
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    dest = destination(tmp_path)
    rotated = {"hf_token": SECRET, "relay": "relay-key-rotated-9876543210"}
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest), registry=rotated)
    assert _behind(_only(_record(dest)))[chunk] == "quarantined: holds relay"
    assert not (dest / chunk).exists()


def _named_after_the_route_key(tmp_path, checkout, registry=None):
    """Export a finalized rollout whose agent holds the route and whose drive wrote a file named
    after the route key and holding it, before finalize, which quarantined it under a scrubbed
    key; return the file's key, that scrubbed key and the destination."""
    plan = facing(make_plan(tmp_path, checkout, "fin_goodput.task"))
    key = plan.layout.attempt_file("deploy_drive", 1, f"logs/note-{RELAY}.txt")
    PosixStore(plan.store_root).write_bytes(key, f"the key is {RELAY}\n".encode())
    plan, store = finalized(tmp_path, checkout, plan=plan)
    record = read_manifest(store, plan.layout, "finalize")[0].outputs["quarantine"].key
    recorded = json.loads(store.read_bytes(record))["files"]
    scrubbed = f"{plan.layout.attempt('deploy_drive', 1)}/logs/note-<redacted:relay>.txt"
    assert recorded[scrubbed] == ["relay"] and RELAY not in json.dumps(recorded)
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest), registry=registry)
    return key, scrubbed, dest


def test_a_contestant_file_named_after_the_route_key_quarantined_by_finalize_stays_behind(tmp_path, checkout):
    key, scrubbed, dest = _named_after_the_route_key(tmp_path, checkout)
    done = _only(_record(dest))
    assert _behind(done)[scrubbed] == "quarantined: holds relay; its path holds relay"
    assert not (dest / key).exists() and RELAY.encode() not in (dest / EXPORT_FILE).read_bytes()


def test_a_key_finalize_recorded_scrubbed_stays_behind_after_the_route_key_rotated(tmp_path, checkout):
    # The export's own scan and path check do not know the old value: only the record matches,
    # its redaction standing for the value, and the file is shown by the record's key.
    rotated = {"hf_token": SECRET, "relay": "relay-key-rotated-9876543210"}
    key, scrubbed, dest = _named_after_the_route_key(tmp_path, checkout, registry=rotated)
    done = _only(_record(dest))
    assert _behind(done)[scrubbed] == "quarantined: holds relay"
    assert not (dest / key).exists() and RELAY.encode() not in (dest / EXPORT_FILE).read_bytes()


def test_a_route_key_in_a_contestant_files_path_leaves_it_behind(tmp_path, checkout):
    # Finalize never saw the file, and its bytes hold no value: the scan finds the value in its
    # path, and the export's own check of what it would publish says so.
    plan, store = finalized(tmp_path, checkout, plan=facing(make_plan(tmp_path, checkout, "fin_goodput.task")))
    key = plan.layout.attempt_file("deploy_drive", 1, f"logs/name-{RELAY}.txt")
    store.write_bytes(key, b"nothing secret here\n")
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    shown = f"{plan.layout.attempt('deploy_drive', 1)}/logs/name-<redacted:relay>.txt"
    assert _behind(_only(_record(dest)))[shown] == "quarantined: holds relay; its path holds relay"
    assert not (dest / key).exists() and RELAY.encode() not in (dest / EXPORT_FILE).read_bytes()


@pytest.mark.parametrize("stage, value, name", [("deploy_drive", SECRET, "hf_token"), ("score", RELAY, "relay")],
                         ids=["no-contestant-holds-it", "harness-file"])
def test_a_path_holding_a_secret_no_contestant_file_may_hold_refuses(tmp_path, checkout, stage, value, name):
    plan, store = finalized(tmp_path, checkout, plan=facing(make_plan(tmp_path, checkout, "fin_goodput.task")))
    store.write_bytes(plan.layout.attempt_file(stage, 1, f"name-{value}.txt"), b"nothing secret here\n")
    fault = _leak(tmp_path, checkout, store, [plan.instance], destination(tmp_path))
    assert fault.reason.endswith(f"({name})") and value not in fault.reason


def test_a_secret_in_an_instances_own_root_refuses(tmp_path, checkout):
    # The scan finds forms in paths below what it lists; the instance's own root, which names every
    # directory the export would make, lies above them and is found in the plan, which holds it.
    plan, store = package_warmer(tmp_path, checkout, findings=[], warmer=f"w-{SECRET}")
    fault = _leak(tmp_path, checkout, store, [plan.instance], destination(tmp_path))
    assert fault.reason.startswith("the harness leaked secret values into published files: ")
    assert "<redacted:hf_token>" in fault.reason and SECRET not in fault.reason


def _agent_transcript(tmp_path, checkout):
    """A finalized rollout whose agent stage published a transcript copy."""
    plan = facing(make_plan(tmp_path, checkout, "fin_goodput.task"))
    store = PosixStore(plan.store_root)
    publish(store, plan, "agent", {"transcript": write(store, plan, "agent", "transcript/INDEX.json", b"{}")})
    return finalized(tmp_path, checkout, plan=plan)


def test_a_dot_named_file_leaves_after_the_scan_and_a_store_temporary_file_stays(tmp_path, checkout):
    plan, store = _agent_transcript(tmp_path, checkout)
    # Dot-named files cannot be store keys (check_segment), so they are written as files.
    (store.path(plan.layout.attempt("agent", 1)) / "outputs" / "transcript" / ".env").write_text("EDITOR=vi\n")
    temp = store.path(plan.layout.attempt("score", 1)) / "outputs" / f".tmp-score.json-{'0' * 32}"
    temp.write_bytes(b"half written")
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    done = _only(_record(dest))
    env = f"{plan.layout.attempt('agent', 1)}/outputs/transcript/.env"
    assert env in _copied(done) and (dest / env).read_text() == "EDITOR=vi\n"
    tmp = temp.relative_to(store.root).as_posix()
    assert _behind(done)[tmp] == "a temporary file of the store" and not (dest / tmp).exists()


def test_a_dot_named_file_holding_the_route_key_is_quarantined(tmp_path, checkout):
    # The scan reads dot-named files, so a contestant's .env is judged as any file.
    plan, store = _agent_transcript(tmp_path, checkout)
    env = store.path(plan.layout.attempt("agent", 1)) / "outputs" / "transcript" / ".env"
    env.write_text(f"KEY={RELAY}\n")
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    key = env.relative_to(store.root).as_posix()
    assert _behind(_only(_record(dest)))[key] == "quarantined: holds relay"
    assert not (dest / key).exists()


def test_a_link_in_an_instance_is_never_followed(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    link = store.path(plan.layout.attempt("score", 1)) / "outputs" / "elsewhere"
    link.symlink_to(tmp_path)
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    key = link.relative_to(store.root).as_posix()
    assert _behind(_only(_record(dest)))[key] == "not a regular file: no export follows a link"
    assert not (dest / key).exists() and not (dest / key).is_symlink()


def test_an_export_of_a_failed_instance_scans_it_first(tmp_path, checkout):
    plan, store = failed(tmp_path, checkout, log=f"server says {RELAY}\n")
    assert json.loads(store.read_bytes(ROW))["scanned"] is False
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    done = _only(_record(dest))
    assert (done.outcome, done.finalized) == ("failed", False)
    log = plan.layout.attempt_file("deploy_drive", 1, "logs/artifact.err")
    assert _behind(done)[log] == "quarantined: holds relay"
    copied = _copied(done)
    # The failed stage's attempt is the evidence of the zero, and leaves but for what was quarantined.
    assert plan.layout.attempt_file("deploy_drive", 1, "logs/driver.err") in copied
    assert {f"{plan.instance}/failed/score.json", f"{plan.instance}/failed/rollout.json", ROW} <= set(copied)
    # No scan read inside its image, and no finalize recorded it: the export's own check keeps it in.
    assert _behind(done)[plan.layout.attempt_file("package", 1, "outputs/submission.sif")] == UNSCANNED_WHY


def test_a_failed_instances_leak_of_ours_refuses(tmp_path, checkout):
    plan, store = failed(tmp_path, checkout, log=f"server says {SECRET}\n")
    fault = _leak(tmp_path, checkout, store, [plan.instance], destination(tmp_path))
    log = plan.layout.attempt_file("deploy_drive", 1, "logs/artifact.err")
    assert fault.reason == f"the harness leaked secret values into published files: {log} (hf_token)"


def test_an_export_refuses_our_leak(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    events = plan.layout.attempt_file("score", 1, "events.jsonl")
    store.write_bytes(events, store.read_bytes(events) + f'{{"note": "{SECRET}"}}\n'.encode())
    dest = destination(tmp_path)
    fault = _leak(tmp_path, checkout, store, [plan.instance], dest)
    assert fault.reason == f"the harness leaked secret values into published files: {events} (hf_token)"
    assert SECRET not in fault.reason


def test_the_index_row_is_scanned_too(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    store.write_bytes(ROW, store.read_bytes(ROW).replace(b'"status"', f'"note": "{SECRET}", "status"'.encode()))
    fault = _leak(tmp_path, checkout, store, [plan.instance], destination(tmp_path))
    assert fault.reason == f"the harness leaked secret values into published files: {ROW} (hf_token)"


def test_a_facing_secret_in_a_harness_file_is_still_our_leak(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout, plan=facing(make_plan(tmp_path, checkout, "fin_goodput.task")))
    events = plan.layout.attempt_file("score", 1, "events.jsonl")
    store.write_bytes(events, store.read_bytes(events) + f'{{"note": "{RELAY}"}}\n'.encode())
    fault = _leak(tmp_path, checkout, store, [plan.instance], destination(tmp_path))
    assert fault.reason == f"the harness leaked secret values into published files: {events} (relay)"


def test_a_gradings_facing_secrets_are_its_own_and_its_files_its_own(tmp_path, checkout):
    # The rollout hands no route to anyone; its grading's agent holds the route, and a value the
    # grading's drive echoed lies under the rollout's root, in the grading's.
    rollout, store = finalized(tmp_path, checkout)
    grading_plan = facing(make_plan(tmp_path, checkout, "fin_goodput.task", kind="grading"))
    grading, _ = finalized(tmp_path, checkout, plan=grading_plan, trace=echoed_trace(RELAY))
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [rollout.instance, grading.instance], dest))
    record = _record(dest)
    assert [d.instance for d in record.instances] == [rollout.instance, grading.instance]
    first, second = record.instances
    chunk = grading.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    assert _behind(second)[chunk] == "quarantined: holds relay"
    assert not any(why.startswith("quarantined") for why in _behind(first).values())
    # The rollout's own export leaves its gradings to their own.
    assert not any(f.path.startswith(f"{rollout.instance}/gradings/") for f in first.copied)
    assert f"runs/fin/index/{ROLLOUT}+{grading.instance.rsplit('/', 1)[1]}.json" in _copied(second)


def test_a_gradings_built_arm_route_is_facing_in_its_files(tmp_path, checkout):
    # The grading hands no route to anyone; the agent of the arm that built its package held it.
    plan, store = graded_package(tmp_path, checkout, built_arm=ARM)
    trace = echoed_trace(RELAY)
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace),
                     drive_inputs=plan.stage("deploy_drive").inputs)
    run_and_publish(store, plan, "score", ScoreStage)
    run_and_publish(store, plan, "finalize", FinalizeStage, secrets=BOTH)
    plan = ended(store, plan)
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    assert _behind(_only(_record(dest)))[chunk] == "quarantined: holds relay"


@pytest.mark.parametrize("finished", [True, False], ids=["finalized", "failed"])
def test_a_grading_naming_another_arm_than_the_one_that_built_its_package_is_refused(tmp_path, checkout,
                                                                                     finished):
    other = ARM.model_copy(update={"name": "arm-a"})
    plan, store = graded_package(tmp_path, checkout, built_arm=other)
    if finished:
        trace = drive_trace()
        publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace),
                         drive_inputs=plan.stage("deploy_drive").inputs)
        plan = ended(store, plan)
    else:
        # The drive's contestant fault ended it before anything of it published.
        store.write_bytes(plan.layout.attempt_file("deploy_drive", 1, "logs/artifact.err"), b"crashed\n")
        fault = FaultRecord(fault_class="contestant", retryable=False, reason="the server exited with status 1")
        plan = ended(store, plan, status="failed", fault=fault)
    with pytest.raises(StageFault) as e:
        run_export(store, export_plan(tmp_path, checkout, [plan.instance], destination(tmp_path)))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "built_arm arm-a is not the arm k3 that built the graded package" in e.value.fault.reason


def test_an_export_judges_a_package_warmers_scan_with_no_facing_secret(tmp_path, checkout):
    plan, store = package_warmer(tmp_path, checkout,
                                 findings=[PackageScanFinding(path="/workspace/server.py", secret="relay")])
    scan = plan.layout.attempt_file(SOURCE_SCAN, 1, f"outputs/{PACKAGE_SCAN_FILE}")
    fault = _leak(tmp_path, checkout, store, [plan.instance], destination(tmp_path))
    assert fault.reason == ("the harness leaked secret values into published files: "
                            f"{scan}::/workspace/server.py (relay)")


def test_a_package_scan_that_does_not_say_which_forms_it_searched_is_refused(tmp_path, checkout):
    plan, store = package_warmer(tmp_path, checkout, findings=[], tree_scan=b'{"secrets": ["hf_token", "relay"], "read": ["bundle", "tree"], "findings": [], "areas": []}')
    scan = plan.layout.attempt_file(SOURCE_SCAN, 1, f"outputs/{PACKAGE_SCAN_FILE}")
    fault = _leak(tmp_path, checkout, store, [plan.instance], destination(tmp_path))
    assert fault.reason.startswith(f"{scan} is not a package scan ({PACKAGE_SCAN_FILE})")


def test_a_clean_package_warmer_leaves_with_its_submission_image(tmp_path, checkout):
    plan, store = package_warmer(tmp_path, checkout, findings=[])
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    done = _only(_record(dest))
    sif = plan.layout.attempt_file("package", 1, "outputs/submission.sif")
    assert _copied(done)[sif].sha256 == SIF_SHA
    assert (done.outcome, done.finalized) == ("succeeded", False)


def _quarantine(store: PosixStore, plan) -> dict:
    """The instance's finalize ``quarantine.json``, as written."""
    return json.loads(store.read_bytes(read_manifest(store, plan.layout, "finalize")[0].outputs["quarantine"].key))


def _build_keys(plan) -> tuple[str, str]:
    """The fixture build's submission image and source bundle, by their store keys."""
    return (plan.layout.attempt_file("package", 1, "outputs/submission.sif"),
            plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/source.bundle"))


def test_a_package_scan_finding_leaves_the_submission_image_and_the_bundle_behind(tmp_path, checkout):
    plan = agent_built(make_plan(tmp_path, checkout, "fin_goodput.task"))
    store = PosixStore(plan.store_root)
    snapshot = plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "snapshots/main-000001.bundle")
    store.write_bytes(snapshot, b"PACK snapshot")
    found = scan_record(BOTH, ["bundle", "tree"], [PackageScanFinding(path="/workspace/server.py", secret="relay")])
    plan, store = finalized(tmp_path, checkout, plan=plan, bundle=BUNDLE, tree_scan=found)
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    done = _only(_record(dest))
    behind = _behind(done)
    sif, bundle = _build_keys(plan)
    where = "a container of the submission, where the package's scan found relay in /workspace/server.py"
    assert behind[sif] == behind[bundle] == where
    # No scan reads a snapshot: finalize recorded it unscanned.
    assert behind[snapshot] == f"{UNSCANNED_WHY}; {where}"
    for key in (sif, bundle, snapshot):
        assert not (dest / key).exists()
    assert plan.layout.attempt_file(SOURCE_SCAN, 1, f"outputs/{PACKAGE_SCAN_FILE}") in _copied(done)


def test_a_secret_past_the_scans_bound_leaves_the_area_the_image_and_the_bundle_behind(tmp_path, checkout):
    over = scan_record(BOTH, ["bundle", "tree"], areas=[PackageScanArea(secret="relay", area="tree", count=3000)])
    plan, store = finalized(tmp_path, checkout, plan=agent_built(make_plan(tmp_path, checkout, "fin_goodput.task")),
                            bundle=BUNDLE, tree_scan=over)
    scan = plan.layout.attempt_file(SOURCE_SCAN, 1, f"outputs/{PACKAGE_SCAN_FILE}")
    sif, bundle = _build_keys(plan)
    assert _quarantine(store, plan) == {"files": {f"{scan}::/workspace/": ["relay"], sif: ["relay"],
                                                  bundle: ["relay"]}}
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    behind = _behind(_only(_record(dest)))
    clause = ("a container of the submission, where the package's scan found relay in every file of /workspace/ "
              "(over_cap)")
    for key in (sif, bundle):
        assert behind[key] == f"quarantined: holds relay; {clause}"
        assert not (dest / key).exists()


def test_a_value_in_the_objects_of_a_bundle_outside_the_store_leaves_every_container_behind(tmp_path, checkout):
    # The store holds no bundle the scan read, as when hls package --bundle names one: a value found
    # in its objects is named by the record, and judged like any finding, secret first.
    found = scan_record(BOTH, ["bundle", "tree"], areas=[PackageScanArea(secret="relay", area="bundle", count=1)])
    plan, store = finalized(tmp_path, checkout, plan=with_package(facing(make_plan(tmp_path, checkout,
                                                                                    "fin_goodput.task"))),
                            tree_scan=found)
    scan = plan.layout.attempt_file(SOURCE_SCAN, 1, f"outputs/{PACKAGE_SCAN_FILE}")
    assert _quarantine(store, plan) == {"files": {f"{scan}::{OUTSIDE_BUNDLE}": ["relay"]}}
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    sif, _ = _build_keys(plan)
    assert _behind(_only(_record(dest)))[sif] == (
        f"a container of the submission, where the package's scan found relay in {OUTSIDE_BUNDLE}")
    assert not (dest / sif).exists()


def test_an_unpublished_agent_build_output_stays_behind_and_the_transcripts_published_files_leave(tmp_path,
                                                                                                checkout):
    plan = agent_built(make_plan(tmp_path, checkout, "fin_goodput.task"))
    store = PosixStore(plan.store_root)
    # A commit point killed outright leaves git's lock of a partial bundle beside the outputs; the
    # store takes no key that starts with a dot, so it is written as a file.
    outputs = plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs")
    store.path(outputs).mkdir(parents=True, exist_ok=True)
    (store.path(outputs) / ".tmp-source.bundle-9.lock").write_bytes(b"PACK half of the history")
    lock = f"{outputs}/.tmp-source.bundle-9.lock"
    log = write(store, plan, AGENT_BUILD_STAGE, "transcript/logs/agent.log", b"plain\n")
    index = ("transcript/INDEX.json", b'{"files": [], "skipped": []}')
    plan, store = finalized(tmp_path, checkout, plan=plan, bundle=BUNDLE, build_extra={"transcript": index})
    assert _quarantine(store, plan) == {"files": {lock: []}}
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    done = _only(_record(dest))
    assert _behind(done)[lock] == UNSCANNED_WHY and not (dest / lock).exists()
    # The transcript is published whole through its index, so its files are judged and leave.
    assert {plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/transcript/INDEX.json"), log} <= set(_copied(done))


def test_a_build_with_no_published_scan_leaves_its_image_and_bundle_behind(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout, plan=agent_built(make_plan(tmp_path, checkout, "fin_goodput.task")),
                            bundle=BUNDLE, tree_scan=None, install_scan=None)
    sif, bundle = _build_keys(plan)
    # Finalize quarantined both with no secret named, and scored the instance: an unscanned file is no leak.
    assert _quarantine(store, plan) == {"files": {sif: [], bundle: []}}
    assert json.loads(store.read_bytes(ROW))["status"] == "ok"
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    behind = _behind(_only(_record(dest)))
    for key in (sif, bundle):
        assert behind[key] == UNSCANNED_WHY
        assert not (dest / key).exists()


def test_a_covered_image_and_bundle_leave_until_the_registry_holds_a_secret_the_scans_did_not_search(
        tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout, plan=agent_built(make_plan(tmp_path, checkout, "fin_goodput.task")),
                            bundle=BUNDLE)
    assert "quarantine" not in read_manifest(store, plan.layout, "finalize")[0].outputs
    sif, bundle = _build_keys(plan)
    covered = destination(tmp_path, "covered")
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], covered))
    copied = _copied(_only(_record(covered)))
    assert copied[sif].sha256 == SIF_SHA and bundle in copied
    # A secret added since: neither scan searched for it.
    grown = {**BOTH, "new_token": "new-token-0123456789abcdef"}
    later = destination(tmp_path, "later")
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], later, export_id="20261004T000001Z-0000dd"),
               registry=grown)
    behind = _behind(_only(_record(later)))
    assert behind[sif] == behind[bundle] == UNSCANNED_WHY
    assert not (later / sif).exists() and not (later / bundle).exists()


def test_oracle_answers_leave_only_when_asked_and_scanned(tmp_path, checkout):
    # The rollout's agent holds the route, which still counts for nothing in the answers area.
    plan, store = finalized(tmp_path, checkout, plan=facing(make_plan(tmp_path, checkout, "fin_goodput.task")))
    answer = "oracle/fin/0123abcd/w1.1.json"
    store.write_bytes(answer, b'{"answers": []}\n')
    without = destination(tmp_path, "without")
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], without))
    assert _record(without).oracle is None and not (without / answer).exists()
    asked = destination(tmp_path, "asked")
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], asked, oracle=True,
                                  export_id="20261004T000001Z-0000dd"))
    data = store.read_bytes(answer)
    assert _record(asked).oracle == [CopiedFile(path=answer, sha256=hashlib.sha256(data).hexdigest(), size=len(data))]
    assert (asked / answer).read_bytes() == store.read_bytes(answer)
    # Even a route an agent of the task was handed is our leak in the answers area.
    store.write_bytes(answer, f'{{"answers": ["{RELAY}"]}}\n'.encode())
    leaked = destination(tmp_path, "leaked")
    with pytest.raises(StageFault) as e:
        run_export(store, export_plan(tmp_path, checkout, [plan.instance], leaked, oracle=True,
                                      export_id="20261004T000002Z-0000ee"))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {answer} (relay)"
    assert not leaked.exists()


def test_oracle_answers_that_are_not_in_the_store_are_refused(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    with pytest.raises(StageFault) as e:
        run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest, oracle=True))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == "fin has no reference answers in the store (oracle/fin)"
    assert _nothing_beside(dest)


def test_a_store_file_that_is_not_what_its_manifest_recorded_refuses(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    key = read_manifest(store, plan.layout, "score")[0].outputs["score"].key
    data = store.read_bytes(key)
    store.write_bytes(key, data[:-1] + (b"x" if data[-1:] != b"x" else b"y"))
    dest = destination(tmp_path)
    with pytest.raises(StageFault) as e:
        run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"the store's {key} holds other bytes than its manifest recorded"
    # The copy, which holds bytes no scan may have read, is removed.
    assert _nothing_beside(dest)


def test_a_copy_role_that_reports_fewer_files_is_retried(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    with pytest.raises(StageFault) as e:
        run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest),
                   launcher=LocalLauncher(gpus=0, replace_modules={"hls.store.export_copy": FAKE_COPY}))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert e.value.fault.reason.startswith("the copy role reported 0 of the ")
    assert _nothing_beside(dest)


def test_a_directory_where_the_attempt_copies_to_is_refused_and_left_as_it_is(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    partial = dest.with_name(f".{dest.name}.{export.instance.rsplit('/', 1)[1]}.1")
    partial.mkdir()
    (partial / "someone-elses.txt").write_text("kept\n")
    with pytest.raises(StageFault) as e:
        run_export(store, export)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"{partial}, where this attempt copies to, exists already"
    assert (partial / "someone-elses.txt").read_text() == "kept\n" and not dest.exists()


def _raced(tmp_path, checkout, race: str):
    """Export a finalized rollout while another writer acts on the destination the export claimed,
    once the copy is done (``fake_copy_then_race.py``); return the fault, the destination and the
    directory the copy went to."""
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    launcher = LocalLauncher(gpus=0, replace_modules={"hls.store.export_copy": RACING_COPY},
                             task_prolog={"RACE": race, "DESTINATION": str(dest)})
    with pytest.raises(StageFault) as e:
        run_export(store, export, launcher=launcher)
    return e.value.fault, dest, dest.with_name(f".{dest.name}.{export.instance.rsplit('/', 1)[1]}.1")


def test_a_claim_another_writer_filled_is_never_replaced_and_keeps_its_files(tmp_path, checkout):
    fault, dest, partial = _raced(tmp_path, checkout, "fill")
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason.startswith(f"another writer put files in the destination {dest}")
    # Their file stays, the export's copy is gone, and the claim, now not empty, is kept.
    assert [p.name for p in dest.iterdir()] == ["theirs.txt"] and not partial.exists()


def test_a_claim_another_writer_replaced_with_a_file_is_never_replaced(tmp_path, checkout):
    fault, dest, partial = _raced(tmp_path, checkout, "file")
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason.startswith(f"another writer replaced the destination {dest} with a file")
    assert dest.read_text() == "another writer's file\n" and not partial.exists()


def test_a_failed_export_leaves_a_directory_another_writer_put_in_place_of_its_claim(tmp_path, checkout):
    # The export removes only the empty claim it made itself.
    fault, dest, partial = _raced(tmp_path, checkout, "swap")
    assert fault.retryable is True
    assert dest.is_dir() and list(dest.iterdir()) == [] and not partial.exists()


@pytest.fixture
def held(tmp_path):
    """A descriptor of tmp_path, as the stage holds the destination's directory."""
    fd = open_directory(tmp_path)
    yield fd
    os.close(fd)


@pytest.mark.parametrize("taken", ["directory", "file", "link"])
def test_a_claim_is_refused_where_anything_is(tmp_path, held, taken):
    dest = tmp_path / "dest"
    if taken == "directory":
        dest.mkdir()
    elif taken == "file":
        dest.write_text("x")
    else:
        dest.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(FileExistsError):
        claim_destination(held, "dest")
    claimed = claim_destination(held, "free")
    made = os.lstat(tmp_path / "free")
    assert (tmp_path / "free").is_dir() and claimed == Made(inode=made.st_ino, ctime_ns=made.st_ctime_ns)


@pytest.mark.parametrize("case", ["untouched", "another-change-time", "filled", "a-file", "gone"])
def test_a_claim_is_removed_only_while_it_is_the_exports_own_and_empty(tmp_path, held, case):
    dest = tmp_path / "dest"
    claim = claim_destination(held, "dest")
    if case == "another-change-time":
        # A directory with the claim's inode, which a filesystem may give a new one, and a later
        # change time.
        claim = claim.model_copy(update={"ctime_ns": claim.ctime_ns - 1})
    elif case == "filled":
        # Filled within the claim's own change time, as a coarse clock allows.
        (dest / "theirs.txt").write_text("kept\n")
        made = os.lstat(dest)
        claim = Made(inode=made.st_ino, ctime_ns=made.st_ctime_ns)
    elif case == "a-file":
        dest.rmdir()
        dest.write_text("kept\n")
    elif case == "gone":
        dest.rmdir()
    assert release_claim(held, "dest", claim) is (case == "untouched")
    assert dest.exists() is (case not in ("untouched", "gone"))


def test_a_route_key_the_package_scan_found_still_quarantines_after_the_site_retired_it(tmp_path, checkout):
    # The agent baked its route key into its tree; the site retired the route since. The plan says
    # the agent held it, so the finding is the contestant's, and the containers stay behind.
    plan = agent_built(make_plan(tmp_path, checkout, "fin_goodput.task"))
    tree = scan_record(BOTH, ["bundle", "tree"], [PackageScanFinding(path="/workspace/server.py", secret="relay")])
    plan, store = finalized(tmp_path, checkout, plan=plan, bundle=BUNDLE, tree_scan=tree)
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest), registry={"hf_token": SECRET})
    sif, bundle = _build_keys(plan)
    where = "a container of the submission, where the package's scan found relay in /workspace/server.py"
    assert _behind(_only(_record(dest)))[sif] == where and not (dest / sif).exists()


def test_a_file_changed_between_the_scan_and_the_copy_is_never_exported(tmp_path, checkout, monkeypatch):
    plan, store = finalized(tmp_path, checkout)
    log = plan.layout.attempt_file("deploy_drive", 1, "logs/driver.err")
    scanned = export_stage.run_scan

    def scan_then_write(ctx, listed, raw, **kwargs):
        report = scanned(ctx, listed, raw, **kwargs)
        store.write_bytes(log, store.read_bytes(log) + SECRET.encode())
        return report

    monkeypatch.setattr(export_stage, "run_scan", scan_then_write)
    dest = destination(tmp_path)
    with pytest.raises(StageFault) as e:
        run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert e.value.fault.reason == f"the store's {log} is not the file the scan read: it changed since"
    assert _nothing_beside(dest)


def test_a_file_the_scan_did_not_read_is_never_exported(tmp_path, checkout, monkeypatch):
    plan, store = finalized(tmp_path, checkout)
    log = plan.layout.attempt_file("deploy_drive", 1, "logs/driver.err")
    scanned = export_stage.run_scan

    def scan_without_the_log(ctx, listed, raw, **kwargs):
        report = scanned(ctx, listed, raw, **kwargs)
        return report._replace(read={key: sha256 for key, sha256 in report.read.items() if key != log})

    monkeypatch.setattr(export_stage, "run_scan", scan_without_the_log)
    dest = destination(tmp_path)
    with pytest.raises(StageFault) as e:
        run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"the scan did not read the store's {log}, which the export would copy"
    assert _nothing_beside(dest)


def test_the_attempt_records_the_directories_it_made_before_it_copies(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    record = store.path(export.layout.attempt_file("export", 1, CLAIM_FILE))
    launcher = LocalLauncher(gpus=0, replace_modules={"hls.store.export_copy": RACING_COPY},
                             task_prolog={"RACE": "recorded", "DESTINATION": str(dest), "CLAIM_RECORD": str(record)})
    run_export(store, export, launcher=launcher)
    assert _only(_record(dest)).instance == plan.instance
    assert sorted(p.name for p in dest.parent.iterdir()) == [dest.name]


def test_finalizes_record_of_an_unscanned_container_holds_after_the_registry_shrank(tmp_path, checkout):
    # Finalize ran where the site held a secret the package scans never searched; the export runs
    # after it left the file, when the scans cover the registry again.
    more = {**BOTH, "extra_key": "extra-key-0123456789abcdef"}
    plan = agent_built(make_plan(tmp_path, checkout, "fin_goodput.task"))
    plan, store = finalized(tmp_path, checkout, plan=plan, bundle=BUNDLE, registry=more)
    sif, bundle = _build_keys(plan)
    assert _quarantine(store, plan) == {"files": {sif: [], bundle: []}}
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest), registry=BOTH)
    behind = _behind(_only(_record(dest)))
    assert behind[sif] == behind[bundle] == UNSCANNED_WHY


@pytest.mark.parametrize("race, theirs", [("swap-partial", ["theirs.txt"]), ("swap-partial-empty", [])],
                         ids=["with-a-file", "empty"])
def test_a_failed_export_leaves_a_directory_another_writer_put_in_place_of_its_copy(tmp_path, checkout, race,
                                                                                    theirs):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    partial = dest.with_name(f".{dest.name}.{export.instance.rsplit('/', 1)[1]}.1")
    launcher = LocalLauncher(gpus=0, replace_modules={"hls.store.export_copy": RACING_COPY},
                             task_prolog={"RACE": race, "DESTINATION": str(dest)})
    with pytest.raises(StageFault):
        run_export(store, export, launcher=launcher)
    assert [p.name for p in partial.iterdir()] == theirs and not dest.exists()


def _crashed_attempt(store, export, dest) -> Path:
    """What attempt 1 of *export* leaves when it died between its claim and its rename: the claim,
    and the directory it made for its copy with a file copied, each recorded in the attempt; return
    that directory."""
    partial = dest.with_name(f".{dest.name}.{export.instance.rsplit('/', 1)[1]}.1")
    parent = open_directory(dest.parent)
    try:
        claim = claim_destination(parent, dest.name)
        made = make_partial(parent, partial.name, dest.name)
        made.close()
    finally:
        os.close(parent)
    (partial / dest.name / "half-copied.txt").write_text("copied before the attempt died\n")
    record = ClaimRecord(claim=claim, partial=made.made)
    store.write_bytes(export.layout.attempt_file("export", 1, CLAIM_FILE), record.model_dump_json().encode())
    return partial


def test_a_retry_takes_back_the_empty_claim_its_export_made_and_removes_its_copy(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    stale = _crashed_attempt(store, export, dest)
    ctx = context(store, export, "export", attempt=2, secrets=BOTH)
    ExportStage().run(ctx)
    assert _only(_record(dest)).instance == plan.instance and not stale.exists()
    assert sorted(p.name for p in dest.parent.iterdir()) == [dest.name]


def test_a_retry_never_takes_a_destination_another_writer_put_in_place_of_its_claim(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    stale = _crashed_attempt(store, export, dest)
    dest.rmdir()
    (dest.parent / "elsewhere").mkdir()
    dest.mkdir()
    ctx = context(store, export, "export", attempt=2, secrets=BOTH)
    with pytest.raises(StageFault) as e:
        ExportStage().run(ctx)
    assert (e.value.fault.reason, e.value.fault.retryable) == (f"the destination {dest} exists", False)
    # Its own earlier copy is removed all the same.
    assert dest.is_dir() and list(dest.iterdir()) == [] and not stale.exists()


def test_a_retry_never_takes_back_a_claim_another_writer_filled(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    _crashed_attempt(store, export, dest)
    (dest / "theirs.txt").write_text("kept\n")
    # Filled within the claim's own change time, as a coarse clock allows.
    key = export.layout.attempt_file("export", 1, CLAIM_FILE)
    made = os.lstat(dest)
    record = ClaimRecord.model_validate_json(store.read_bytes(key)).model_copy(
        update={"claim": Made(inode=made.st_ino, ctime_ns=made.st_ctime_ns)})
    store.write_bytes(key, record.model_dump_json().encode())
    with pytest.raises(StageFault) as e:
        ExportStage().run(context(store, export, "export", attempt=2, secrets=BOTH))
    assert e.value.fault.reason == f"the destination {dest} exists"
    assert [p.name for p in dest.iterdir()] == ["theirs.txt"]


def _events(store, export, attempt: int, kind: str) -> list[dict]:
    """The events of *kind* that attempt *attempt* of *export* recorded."""
    lines = store.read_bytes(export.layout.attempt_file("export", attempt, "events.jsonl")).decode().splitlines()
    return [event for event in map(json.loads, lines) if event["kind"] == kind]


def test_a_retry_names_an_earlier_copy_it_cannot_tell_from_another_writers_for_what_it_is(tmp_path, checkout):
    # Attempt 1 died past its rename: the destination is its complete copy, and the directory it
    # made for the copy is empty, its change time moved by the rename.
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    stale = _crashed_attempt(store, export, dest)
    os.rename(stale / dest.name, dest)
    ctx = context(store, export, "export", attempt=2, secrets=BOTH)
    with pytest.raises(StageFault) as e:
        ExportStage().run(ctx)
    assert e.value.fault.reason == f"the destination {dest} exists"
    assert stale.is_dir() and list(stale.iterdir()) == []
    [left] = _events(store, export, 2, "export_partial_left")
    assert (left["path"], left["why"]) == (str(stale), "not the directory its attempt recorded: another writer's, "
                                                       "or the attempt's own after its copy was renamed onto the "
                                                       "destination")


def test_a_destination_whose_directory_is_reached_through_a_link_is_refused_before_anything_is_read(tmp_path,
                                                                                                   checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    (tmp_path / "real").mkdir()
    dest.parent.rmdir()
    dest.parent.symlink_to(tmp_path / "real")
    ctx = context(store, export, "export", secrets=BOTH)
    with pytest.raises(StageFault) as e:
        ExportStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason.startswith(f"the destination's directory {dest.parent} cannot be opened without "
                                           "following a link: ")
    assert ctx.report("secret-scan") is None and list((tmp_path / "real").iterdir()) == []


def test_a_retry_leaves_a_directory_another_writer_put_in_place_of_an_earlier_copy(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    stale = _crashed_attempt(store, export, dest)
    shutil.rmtree(stale)
    (dest.parent / "elsewhere").mkdir()
    stale.mkdir()
    (stale / "theirs.txt").write_text("kept\n")
    ExportStage().run(context(store, export, "export", attempt=2, secrets=BOTH))
    assert _only(_record(dest)).instance == plan.instance
    assert [p.name for p in stale.iterdir()] == ["theirs.txt"]


def test_an_existing_destination_is_refused_before_anything_is_read(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    dest.mkdir()
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    ctx = context(store, export, "export", secrets=BOTH)
    with pytest.raises(StageFault) as e:
        ExportStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"the destination {dest} exists"
    assert ctx.report("secret-scan") is None


def test_an_instance_still_running_is_refused(tmp_path, checkout):
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    ended(store, plan, status="running")
    with pytest.raises(StageFault) as e:
        run_export(store, export_plan(tmp_path, checkout, [plan.instance], destination(tmp_path)))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"instance {plan.instance} has not ended"


def test_the_export_refuses_to_run_without_the_registry(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    export = export_plan(tmp_path, checkout, [plan.instance], destination(tmp_path))
    with pytest.raises(StageFault) as e:
        ExportStage().run(context(store, export, "export", secrets=BOTH, registry=False))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)


def test_the_export_refuses_a_registry_that_holds_no_secret(tmp_path, checkout):
    plan, store = finalized(tmp_path, checkout)
    dest = destination(tmp_path)
    export = export_plan(tmp_path, checkout, [plan.instance], dest)
    with pytest.raises(StageFault) as e:
        ExportStage().run(context(store, export, "export", secrets={}))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason.startswith("the secrets file holds no secret")
    assert _nothing_beside(dest)
