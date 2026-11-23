"""A URL secret's host is one of its searchable forms: the export finds it in
every file and path it would publish, in any case, stored or gzip-compressed, and judges it by
finalize's rule. A host in a contestant's file leaves the file behind and never refuses; one in a
harness file or the answers area is the harness's leak."""

from __future__ import annotations

import gzip

import pytest

from hls.pipeline.stage import StageFault
from hls.stages.export import EXPORT_FILE, ExportRecord, InstanceExport
from hls.stages.package import AGENT_BUILD_STAGE
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE, PackageScanFinding
from tests.export.instances import (BOTH, BUNDLE, SOURCE_SCAN, agent_built, destination, export_plan, finalized,
                                    run_export, scan_record)
from tests.finalize.published import echoed_trace, make_plan

#: A route's base URL, and its host, which grants no access but may name a private relay.
RELAY_URL = "https://relay-0123456789abcdef.example.net/v1"
HOST = "relay-0123456789abcdef.example.net"
#: The fixture's registry with the base URL beside the other two secrets.
WITH_URL = {**BOTH, "relay_url": RELAY_URL}
UNSCANNED_WHY = "quarantined: unscanned: no package scan covers it"


def _only(dest) -> InstanceExport:
    [done] = ExportRecord.model_validate_json((dest / EXPORT_FILE).read_bytes()).instances
    return done


def _behind(done: InstanceExport) -> dict[str, str]:
    return {f.path: f.why for f in done.left_behind}


def _finalized(tmp_path, checkout, **kwargs):
    """A rollout finalized on the site of ``WITH_URL``, whose package scans searched every form of
    each of its secrets."""
    tree = scan_record(WITH_URL, ["bundle", "tree"])
    install = scan_record(WITH_URL, ["install"])
    return finalized(tmp_path, checkout, registry=WITH_URL, tree_scan=tree, install_scan=install, **kwargs)


@pytest.mark.parametrize("spelled", [HOST, HOST.upper(), "Relay-0123456789ABCDEF.Example.Net"],
                         ids=["lower", "upper", "mixed"])
def test_a_host_in_a_contestants_file_leaves_it_behind_in_any_case(tmp_path, checkout, spelled):
    plan, store = _finalized(tmp_path, checkout)
    log = plan.layout.attempt_file("deploy_drive", 1, "logs/artifact.err")
    store.write_bytes(log, f"dialled {spelled} on start\n".encode())
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest), registry=WITH_URL)
    done = _only(dest)
    assert _behind(done)[log] == "quarantined: holds the host of relay_url"
    assert log not in {f.path for f in done.copied} and not (dest / log).exists()
    assert HOST.encode() not in (dest / EXPORT_FILE).read_bytes().lower()


def test_a_host_a_server_echoed_into_its_compressed_trace_leaves_it_behind(tmp_path, checkout):
    plan, store = _finalized(tmp_path, checkout, trace=echoed_trace(f"see https://{HOST.upper()}/x"))
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    assert HOST.encode() in gzip.decompress(store.read_bytes(chunk)).lower()
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest), registry=WITH_URL)
    assert _behind(_only(dest))[chunk] == "quarantined: holds the host of relay_url"
    assert not (dest / chunk).exists()


def test_a_host_in_a_contestants_files_path_leaves_it_behind(tmp_path, checkout):
    plan, store = _finalized(tmp_path, checkout)
    key = plan.layout.attempt_file("deploy_drive", 1, f"logs/to-{HOST.upper()}.txt")
    store.write_bytes(key, b"nothing to see\n")
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest), registry=WITH_URL)
    shown = f"{plan.layout.attempt('deploy_drive', 1)}/logs/to-<redacted:relay_url.host>.txt"
    assert _behind(_only(dest))[shown] == ("quarantined: holds the host of relay_url; its path holds the host of "
                                           "relay_url")
    assert not (dest / key).exists()


def _refused(tmp_path, checkout, store, plan, **kwargs) -> str:
    dest = destination(tmp_path)
    with pytest.raises(StageFault) as e:
        run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest, **kwargs), registry=WITH_URL)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert [p.name for p in dest.parent.iterdir()] == []
    assert HOST not in e.value.fault.reason.lower()
    return e.value.fault.reason


def test_a_host_in_a_harness_file_is_our_leak(tmp_path, checkout):
    plan, store = _finalized(tmp_path, checkout)
    events = plan.layout.attempt_file("score", 1, "events.jsonl")
    store.write_bytes(events, store.read_bytes(events) + f'{{"note": "{HOST.upper()}"}}\n'.encode())
    reason = _refused(tmp_path, checkout, store, plan)
    assert reason == f"the harness leaked secret values into published files: {events} (the host of relay_url)"


def test_a_host_in_the_answers_area_is_our_leak(tmp_path, checkout):
    plan, store = _finalized(tmp_path, checkout)
    answer = "oracle/fin/0123abcd/w1.1.json"
    store.write_bytes(answer, f'{{"answers": ["{HOST}"]}}\n'.encode())
    reason = _refused(tmp_path, checkout, store, plan, oracle=True)
    assert reason == f"the harness leaked secret values into published files: {answer} (the host of relay_url)"


def test_a_host_a_package_scan_found_leaves_the_containers_behind_and_never_refuses(tmp_path, checkout):
    # No contestant of the rollout holds the base URL, yet a host is no leak: the image and the
    # bundle the tree went into stay behind.
    found = scan_record(WITH_URL, ["bundle", "tree"],
                        [PackageScanFinding(path="/workspace/client.py", secret="relay_url.host")])
    plan, store = finalized(tmp_path, checkout, plan=agent_built(make_plan(tmp_path, checkout, "fin_goodput.task")),
                            registry=WITH_URL, bundle=BUNDLE, tree_scan=found,
                            install_scan=scan_record(WITH_URL, ["install"]))
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest), registry=WITH_URL)
    behind = _behind(_only(dest))
    where = "a container of the submission, where the package's scan found the host of relay_url in /workspace/client.py"
    sif = plan.layout.attempt_file("package", 1, "outputs/submission.sif")
    bundle = plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/source.bundle")
    assert behind[sif] == behind[bundle] == where
    assert plan.layout.attempt_file(SOURCE_SCAN, 1, f"outputs/{PACKAGE_SCAN_FILE}") not in behind


@pytest.mark.parametrize("forms, covered", [(("value", "host"), True), (("value",), False)],
                         ids=["every-form", "values-only"])
def test_a_container_leaves_only_when_its_scans_searched_every_form(tmp_path, checkout, forms, covered):
    tree = scan_record(WITH_URL, ["bundle", "tree"], forms=forms)
    install = scan_record(WITH_URL, ["install"], forms=forms)
    plan, store = finalized(tmp_path, checkout, plan=agent_built(make_plan(tmp_path, checkout, "fin_goodput.task")),
                            registry=WITH_URL, bundle=BUNDLE, tree_scan=tree, install_scan=install)
    dest = destination(tmp_path)
    run_export(store, export_plan(tmp_path, checkout, [plan.instance], dest), registry=WITH_URL)
    done = _only(dest)
    sif = plan.layout.attempt_file("package", 1, "outputs/submission.sif")
    bundle = plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/source.bundle")
    copied = {f.path for f in done.copied}
    if covered:
        assert {sif, bundle} <= copied
    else:
        assert _behind(done)[sif] == _behind(done)[bundle] == UNSCANNED_WHY
        assert not (dest / sif).exists() and not (dest / bundle).exists()
