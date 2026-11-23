"""Finalize judges a URL secret's host, one of the secret's searchable forms, by
the rule it shares with the export: a host in a contestant's file quarantines the file and never
refuses or changes the score; one in a file the harness alone wrote is the harness's leak. A
package scan's record covers a container only when it searched every form."""

from __future__ import annotations

import json

import pytest

from hls.pipeline.stage import StageFault
from hls.stages.finalize import FinalizeStage
from hls.stages.package import AGENT_BUILD_STAGE
from hls.stages.score import ScoreStage
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.secret_scan import PackageScanFinding
from tests.export.instances import BOTH, BUNDLE, MODULE, agent_built, graded_package, scan_record
from tests.finalize.test_agent_build_scan import ARM
from tests.finalize.published import ROLLOUT, context, make_plan, publish, publish_instance, run_and_publish
from tests.trace.traces import drive_trace, passing_verdict

#: A route's base URL and its host.
RELAY_URL = "https://relay-0123456789abcdef.example.net/v1"
HOST = "relay-0123456789abcdef.example.net"
#: The fixture's registry with the base URL beside the other two secrets.
WITH_URL = {**BOTH, "relay_url": RELAY_URL}
ROW = f"runs/fin/index/{ROLLOUT}.json"


def _scored(tmp_path, checkout, *, plan=None, tree=None, install=None, bundle=None):
    """A served instance of the fixture task, scored, ready for finalize on the site of
    ``WITH_URL``; its package scans searched every form of every secret unless *tree* or
    *install* say otherwise."""
    plan = plan or make_plan(tmp_path, checkout, MODULE)
    store = PosixStore(plan.store_root)
    trace = drive_trace()
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace), bundle=bundle,
                     tree_scan=tree or scan_record(WITH_URL, ["bundle", "tree"]),
                     install_scan=install or scan_record(WITH_URL, ["install"]))
    run_and_publish(store, plan, "score", ScoreStage)
    return plan, store


def _finalize(store, plan):
    ctx = context(store, plan, "finalize", secrets=WITH_URL)
    outputs = FinalizeStage().run(ctx)
    publish(store, plan, "finalize", outputs)
    return outputs


def _quarantined(store, outputs) -> dict:
    return json.loads(store.read_bytes(outputs["quarantine"]))["files"] if "quarantine" in outputs else {}


@pytest.mark.parametrize("spelled", [HOST, HOST.upper()], ids=["lower", "upper"])
def test_a_host_in_a_contestants_file_is_quarantined_and_the_score_is_unchanged(tmp_path, checkout, spelled):
    clean_plan, clean_store = _scored(tmp_path / "clean", checkout)
    _finalize(clean_store, clean_plan)
    plan, store = _scored(tmp_path / "host", checkout)
    log = plan.layout.attempt_file("deploy_drive", 1, "logs/artifact.err")
    store.write_bytes(log, f"dialled {spelled}\n".encode())
    outputs = _finalize(store, plan)
    assert _quarantined(store, outputs) == {log: ["relay_url.host"]}
    clean_row = json.loads(clean_store.read_bytes(ROW))
    row = json.loads(store.read_bytes(ROW))
    assert (row["status"], row["score"], row["gate_passed"]) == (clean_row["status"], clean_row["score"],
                                                                 clean_row["gate_passed"])
    assert row["quarantined"] == [log]


def test_a_host_in_a_harness_file_is_our_leak(tmp_path, checkout):
    plan, store = _scored(tmp_path, checkout)
    events = plan.layout.attempt_file("score", 1, "events.jsonl")
    store.write_bytes(events, store.read_bytes(events) + f'{{"note": "{HOST}"}}\n'.encode())
    with pytest.raises(StageFault) as e:
        _finalize(store, plan)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == (f"the harness leaked secret values into published files: {events} "
                                    "(the host of relay_url)")
    assert not store.exists(ROW)


def _containers(plan) -> tuple[str, str]:
    return (plan.layout.attempt_file("package", 1, "outputs/submission.sif"),
            plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/source.bundle"))


@pytest.mark.parametrize("forms, covered", [(("value", "host"), True), (("value",), False)],
                         ids=["every-form", "values-only"])
def test_a_record_covers_its_containers_only_when_it_searched_every_form(tmp_path, checkout, forms, covered):
    plan = agent_built(make_plan(tmp_path, checkout, MODULE))
    plan, store = _scored(tmp_path, checkout, plan=plan, bundle=BUNDLE,
                          tree=scan_record(WITH_URL, ["bundle", "tree"], forms=forms),
                          install=scan_record(WITH_URL, ["install"], forms=forms))
    outputs = _finalize(store, plan)
    sif, bundle = _containers(plan)
    assert _quarantined(store, outputs) == ({} if covered else {sif: [], bundle: []})
    assert read_manifest(store, plan.layout, "finalize") is not None


def test_a_package_scan_that_does_not_say_which_forms_it_searched_is_refused(tmp_path, checkout):
    plan, store = _scored(tmp_path, checkout, tree=b'{"secrets": ["hf_token", "relay"], "read": ["bundle", "tree"], "findings": [], "areas": []}')
    with pytest.raises(StageFault) as e:
        _finalize(store, plan)
    scan = plan.layout.attempt_file("source_scan", 1, "outputs/package-scan.json")
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason.startswith(f"{scan} is not a package scan (package-scan.json)")
    assert not store.exists(ROW)


def test_a_package_graded_after_the_site_retired_its_route_quarantines_what_its_scan_found(tmp_path, checkout):
    # The rollout's agent, of ARM, baked its route key into its tree; the grading names ARM, whose
    # route the site has retired since.
    tree = scan_record(BOTH, ["bundle", "tree"], [PackageScanFinding(path="/workspace/server.py", secret="relay")])
    plan, store = graded_package(tmp_path, checkout, built_arm=ARM, tree_scan=tree)
    trace = drive_trace()
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace),
                     drive_inputs=plan.stage("deploy_drive").inputs)
    run_and_publish(store, plan, "score", ScoreStage)
    ctx = context(store, plan, "finalize", secrets={"hf_token": BOTH["hf_token"]})
    outputs = FinalizeStage().run(ctx)
    scan = f"runs/fin/{ROLLOUT}/stages/source_scan/attempts/1/outputs/package-scan.json"
    assert _quarantined(store, outputs)[f"{scan}::/workspace/server.py"] == ["relay"]
