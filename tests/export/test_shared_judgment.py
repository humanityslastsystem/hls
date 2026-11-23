"""Finalize's judgment as module functions, which the finalize and export stages share:
whose file a key is, the rule secret first, the scan report's reader and the package scans'."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from hls.pipeline.plan import Plan
from hls.pipeline.stage import StageFault
from hls.stages.agent_build import AgentBuildStage
from hls.stages.export import RecordMatcher
from hls.stages.finalize import (OUTSIDE_BUNDLE, PACKAGE_SCAN_SEP, check_built_arm, contestant_facing,
                                 contestant_file, instance_containers, judge,
                                 leak_fault, package_scan_findings, read_scan_report, scan_lists, shown_key,
                                 ScanReport,
                                 unscanned_containers)
from hls.stages.package import AGENT_BUILD_STAGE, SOURCE_BUNDLE_OUTPUT, PackageStage
from hls.stages.source_scan import SourceScanStage
from hls.stages.verify import VerifyStage
from hls.store.digest import sha256_bytes
from hls.store.manifest import Manifest
from hls.store.posix import PosixStore
from hls.substrate.secret_forms import FORMS, form_text
from hls.substrate.secret_scan import (CUT_PREFIX, FOUND_PREFIX, PACKAGE_SCAN_FILE, READ_PREFIX, UNREAD_PREFIX,
                                       PackageScan, PackageScanArea, PackageScanFinding)
from hls.substrate.secrets import Scrubber
from tests.finalize.published import make_plan, publish, with_params, with_stages, write
from tests.finalize.stand_in_stage import AgentStandInStage
from tests.finalize.test_agent_build_scan import ARM

#: The stage that scans the source bundle's objects and its checked-out tree.
SOURCE_SCAN = "source_scan"
#: The names of the secrets the site holds in these tests.
NAMES = ["hf_token", "relay"]


def test_only_a_facing_secret_in_a_contestants_file_is_quarantined():
    findings = [("a", "relay", True), ("b", "relay", False), ("c", "hf_token", True)]
    ours, quarantined = judge(findings, frozenset({"relay"}))
    assert ours == [("b", "relay"), ("c", "hf_token")]
    assert quarantined == {"a": ["relay"], "b": ["relay"], "c": ["hf_token"]}
    assert judge([("a", "relay", True), ("a", "other", True)], frozenset({"relay", "other"})) == (
        [], {"a": ["other", "relay"]})


def test_a_host_is_quarantined_in_a_contestants_file_whoever_could_hold_it_and_ours_elsewhere():
    # A host grants no access and may be public: no facing set decides it, only whose file it is.
    findings = [("a", "relay_url.host", True), ("b", "relay_url.host", False), ("c", "relay_url", True)]
    ours, quarantined = judge(findings, frozenset())
    assert ours == [("b", "relay_url.host"), ("c", "relay_url")]
    assert quarantined == {"a": ["relay_url.host"], "b": ["relay_url.host"], "c": ["relay_url"]}
    assert judge([("a", "relay_url", True), ("a", "relay_url.host", True)], frozenset({"relay_url"})) == (
        [], {"a": ["relay_url", "relay_url.host"]})


def test_a_record_says_a_form_by_its_secret():
    assert (form_text("relay"), form_text("relay_url.host")) == ("relay", "the host of relay_url")
    with pytest.raises(ValueError):
        form_text("relay.port")


def test_the_leaks_fault_names_ten_files_and_counts_the_rest():
    ours = [(f"k{i:02d}", "hf_token") for i in range(12)]
    fault = leak_fault(ours, Scrubber({}))
    assert (fault.fault.fault_class, fault.fault.retryable) == ("infra", False)
    assert fault.fault.reason.startswith("the harness leaked secret values into published files: k00 (hf_token)")
    assert "k09 (hf_token)" in fault.fault.reason and "k10" not in fault.fault.reason
    assert fault.fault.reason.endswith(" and 2 more")


def test_the_leaks_fault_is_scrubbed():
    fault = leak_fault([("runs/t/r/name-hf-value.txt", "hf_token")], Scrubber({"hf_token": "hf-value"}))
    assert "hf-value" not in fault.fault.reason and "<redacted:hf_token>" in fault.fault.reason


def test_a_contestants_file_is_one_of_a_stage_that_runs_contestant_code_or_a_declared_output():
    plan = make_plan(Path("/x"), Path("/c"), "fin_goodput.task")
    attempt = f"{plan.instance}/stages/{{}}/attempts/1/{{}}"
    assert contestant_file(plan, attempt.format("deploy_drive", "logs/driver.err"))
    assert contestant_file(plan, attempt.format("score", "outputs/score.json"))
    assert not contestant_file(plan, attempt.format("score", "events.jsonl"))
    assert not contestant_file(plan, f"{plan.instance}/plan/rev-00000001.json")
    assert not contestant_file(plan, "oracle/fin/key/w.1.json")


def test_a_harness_roles_log_is_a_contestants_file_only_where_its_kind_says_the_role_echoes_one():
    """In a stage that runs no contestant code, only the stdout and stderr of the harness roles its
    kind names (``Stage.contestant_log_roles``) hold a contestant's bytes: verify's role, which
    quotes the server's replies, and source_scan's checkout, which lists the agent's tree. Any
    other harness role's log, score's or provision's, is the harness's own, where a secret is its
    leak."""
    plan = make_plan(Path("/x"), Path("/c"), "fin_goodput.task")
    attempt = f"{plan.instance}/stages/{{}}/attempts/1/{{}}"
    assert (VerifyStage.contestant_log_roles, SourceScanStage.contestant_log_roles) == (("verify",), ("checkout",))
    for stage, role in (("verify", "verify"), ("source_scan", "checkout")):
        assert contestant_file(plan, attempt.format(stage, f"logs/{role}.err"))
        assert contestant_file(plan, attempt.format(stage, f"logs/{role}.out"))
        assert not contestant_file(plan, attempt.format(stage, f"logs/{role}.report.json"))
    assert not contestant_file(plan, attempt.format("source_scan", "logs/scan.err"))
    assert not contestant_file(plan, attempt.format("score", "logs/score.err"))
    assert not contestant_file(plan, attempt.format("score", "logs/score.out"))
    assert not contestant_file(plan, attempt.format("provision", "logs/rows.err"))


def test_the_scan_lists_leave_a_rollouts_gradings_to_them(tmp_path):
    plan = make_plan(tmp_path, Path("/c"), "fin_goodput.task")
    store = PosixStore(plan.store_root)
    store.write_bytes(f"{plan.instance}/plan/rev-00000001.json", b"{}")
    store.write_bytes(f"{plan.instance}/gradings/g-1/plan/rev-00000001.json", b"{}")
    listed, _ = scan_lists(store, plan, [], skip_gradings=True)
    assert listed == [store.path(f"{plan.instance}/plan")]
    listed, _ = scan_lists(store, plan, [], skip_gradings=False)
    assert listed == [store.path(f"{plan.instance}/gradings"), store.path(f"{plan.instance}/plan")]


def test_the_scan_report_names_files_by_their_keys(tmp_path):
    root = tmp_path / "store"
    report = tmp_path / "found.jsonl"
    report.write_text(f'{READ_PREFIX}{{"file": "{root}/runs/t/r/a.json", "sha256": "{"a" * 64}"}}\n'
                      f'{FOUND_PREFIX}{{"file": "{root}/runs/t/r/a.json", "secret": "relay"}}\n'
                      f'{READ_PREFIX}{{"file": "{root}/runs/t/r/b.gz", "sha256": "{"b" * 64}"}}\n'
                      f'{CUT_PREFIX}{{"file": "{root}/runs/t/r/b.gz"}}\n'
                      f'{FOUND_PREFIX}{{"file": "/elsewhere/x", "secret": "hf_token"}}\n'
                      f'{UNREAD_PREFIX}{{"file": "{root}/runs/t/r/link", "kind": "link"}}\n')
    assert read_scan_report(report, root) == ScanReport(
        found=[("runs/t/r/a.json", "relay"), ("/elsewhere/x", "hf_token")], cut=["runs/t/r/b.gz"],
        read={"runs/t/r/a.json": "a" * 64, "runs/t/r/b.gz": "b" * 64}, unread=["runs/t/r/link"])
    report.write_text(f'{READ_PREFIX}{{"file": "{root}/a"}}\n')
    assert read_scan_report(report, root) is None
    report.write_text(f"{FOUND_PREFIX}not json\n")
    assert read_scan_report(report, root) is None
    assert read_scan_report(tmp_path / "missing.jsonl", root) is None


def test_a_package_scans_findings_are_named_by_its_record_and_the_path(tmp_path):
    plan = make_plan(tmp_path, Path("/c"), "fin_goodput.task")
    store = PosixStore(plan.store_root)
    scan = PackageScan(secrets=["relay"], read=["tree"], forms=list(FORMS),
                       findings=[PackageScanFinding(path="/workspace/k.txt", secret="relay")])
    key = write(store, plan, "package", PACKAGE_SCAN_FILE, scan.model_dump_json().encode())
    manifest = publish(store, plan, "package", {"scan": key})
    assert package_scan_findings(store, [manifest]) == [(f"{key}{PACKAGE_SCAN_SEP}/workspace/k.txt", "relay")]
    store.write_bytes(key, b"{}")
    with pytest.raises(StageFault) as e:
        package_scan_findings(store, [manifest])
    assert e.value.fault.retryable is False
    assert e.value.fault.reason.startswith(f"{key} does not match its manifest")


def _pin(store: PosixStore, plan: Plan, stage: str) -> dict[str, str]:
    key = plan.layout.manifest(stage)
    return {key: sha256_bytes(store.read_bytes(key))}


def _scan(read: list[str], *, secrets: tuple[str, ...] = tuple(NAMES), forms: tuple[str, ...] = FORMS,
          areas: list[PackageScanArea] | None = None) -> PackageScan:
    return PackageScan(secrets=list(secrets), read=read, findings=[], areas=areas or [], forms=list(forms))


def _plan(tmp_path: Path) -> Plan:
    """The fixture plan naming the agent_build and package stages by their own kinds, which declare
    their containers (``Stage.compressed_files``, ``Stage.unpublished_containers``)."""
    plan = make_plan(tmp_path, Path("/c"), "fin_goodput.task")
    model = plan.stage("score")
    build = model.model_copy(update={"name": AGENT_BUILD_STAGE, "kind": AgentBuildStage.kind, "params": {}})
    package = model.model_copy(update={"name": "package", "kind": PackageStage.kind, "params": {}})
    return with_stages(plan, build, package)


def _build(store: PosixStore, plan: Plan, *, source: PackageScan | None, install: PackageScan | None,
           built: bool = True) -> tuple[list[Manifest], dict[str, str | None]]:
    """Publish *plan*'s build: with *built*, an agent_build stage's source bundle; unless None, a
    source_scan stage's record *source*, which read that bundle; and a package stage's submission
    image with, unless None, its record *install*, the package having read the source_scan stage
    when there is one. Return the manifests, and the keys of the two records, the image and the
    bundle."""
    manifests: list[Manifest] = []
    keys: dict[str, str | None] = {"bundle": None, "source": None, "install": None}
    if built:
        keys["bundle"] = write(store, plan, AGENT_BUILD_STAGE, "source.bundle", b"PACK history")
        manifests.append(publish(store, plan, AGENT_BUILD_STAGE, {SOURCE_BUNDLE_OUTPUT: keys["bundle"]}))
    pins = None
    if source is not None:
        keys["source"] = write(store, plan, SOURCE_SCAN, PACKAGE_SCAN_FILE, source.model_dump_json().encode())
        manifests.append(publish(store, plan, SOURCE_SCAN, {"package_scan": keys["source"]},
                                 inputs=_pin(store, plan, AGENT_BUILD_STAGE) if built else None))
        pins = _pin(store, plan, SOURCE_SCAN)
    outputs = {"submission": write(store, plan, "package", "submission.sif", b"image")}
    if install is not None:
        keys["install"] = outputs["package_scan"] = write(store, plan, "package", PACKAGE_SCAN_FILE,
                                                          install.model_dump_json().encode())
    manifests.append(publish(store, plan, "package", outputs, inputs=pins))
    keys["image"] = outputs["submission"]
    return manifests, keys


@pytest.mark.parametrize("area", ["tree", "bundle", "install", "outside"])
def test_a_secret_named_only_by_its_area_stands_found_in_everything_that_holds_it(tmp_path, area):
    plan = _plan(tmp_path)
    store = PosixStore(plan.store_root)
    past = [PackageScanArea(secret="relay", area="bundle" if area == "outside" else area, count=3000)]
    manifests, keys = _build(store, plan, source=_scan(["bundle", "tree"], areas=past if area != "install" else None),
                             install=_scan(["install"], areas=past if area == "install" else None),
                             built=area != "outside")
    held = {"tree": [f"{keys['source']}{PACKAGE_SCAN_SEP}/workspace/", keys["bundle"], keys["image"]],
            "bundle": [keys["bundle"]],
            "install": [f"{keys['install']}{PACKAGE_SCAN_SEP}/opt/venv/", keys["image"]],
            # No bundle the scan read is in the store: its objects are named by the record.
            "outside": [f"{keys['source']}{PACKAGE_SCAN_SEP}{OUTSIDE_BUNDLE}"]}[area]
    assert package_scan_findings(store, manifests) == [(key, "relay") for key in held]


def test_a_manifest_a_scan_read_that_is_not_the_one_it_pinned_refuses(tmp_path):
    plan = _plan(tmp_path)
    store = PosixStore(plan.store_root)
    past = [PackageScanArea(secret="relay", area="tree", count=3000)]
    manifests, _ = _build(store, plan, source=_scan(["bundle", "tree"], areas=past), install=_scan(["install"]))
    key = plan.layout.manifest(AGENT_BUILD_STAGE)
    store.write_bytes(key, store.read_bytes(key).replace(b"source.bundle", b"source.bundlf"))
    with pytest.raises(StageFault, match=f"is not the manifest {SOURCE_SCAN} read") as e:
        package_scan_findings(store, manifests)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)


def test_the_bundle_is_read_by_the_scan_of_its_objects_and_the_image_by_both_scans(tmp_path):
    plan = _plan(tmp_path)
    store = PosixStore(plan.store_root)
    manifests, keys = _build(store, plan, source=_scan(["bundle", "tree"]), install=_scan(["install"]))
    assert unscanned_containers(store, plan, manifests, NAMES) == []
    # A record that searched fewer secrets than the site holds reads nothing for it.
    assert set(unscanned_containers(store, plan, manifests, [*NAMES, "new_token"])) == {keys["image"], keys["bundle"]}


@pytest.mark.parametrize("source, install, unscanned", [
    (_scan(["bundle"]), _scan(["install"]), {"image"}),
    (_scan(["bundle", "tree"]), None, {"image"}),
    (_scan(["bundle", "tree"]), _scan(["install"], secrets=("relay",)), {"image"}),
    (_scan(["bundle", "tree"], secrets=("relay",)), _scan(["install"]), {"image", "bundle"}),
    (None, _scan(["install"]), {"image", "bundle"}),
    (None, None, {"image", "bundle"}),
], ids=["tree-unread", "no-install-scan", "install-searched-fewer", "source-searched-fewer", "no-source-scan",
        "no-scan"])
def test_a_container_is_read_only_by_every_scan_that_reads_inside_it(tmp_path, source, install, unscanned):
    plan = _plan(tmp_path)
    store = PosixStore(plan.store_root)
    manifests, keys = _build(store, plan, source=source, install=install)
    assert set(unscanned_containers(store, plan, manifests, NAMES)) == {keys[name] for name in unscanned}


def test_a_record_of_no_secret_reads_nothing(tmp_path):
    # Where the site's file held no secret, the stages write records that read no area.
    plan = _plan(tmp_path)
    store = PosixStore(plan.store_root)
    manifests, keys = _build(store, plan, source=_scan([], secrets=()), install=_scan([], secrets=()))
    assert set(unscanned_containers(store, plan, manifests, [])) == {keys["image"], keys["bundle"]}


def test_an_instances_containers_are_its_image_its_bundle_its_snapshots_and_what_it_left_unpublished(tmp_path):
    plan = _plan(tmp_path)
    store = PosixStore(plan.store_root)
    manifests, keys = _build(store, plan, source=None, install=None)
    snapshot = plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "snapshots/main-000001.bundle")
    store.write_bytes(snapshot, b"PACK snapshot")
    left = write(store, plan, AGENT_BUILD_STAGE, "left.bundle", b"PACK an unpublished bundle")
    assert sorted(instance_containers(store, plan, manifests)) == sorted([keys["bundle"], snapshot, left,
                                                                          keys["image"]])


def test_a_key_that_is_not_utf8_is_shown_with_its_escape():
    assert shown_key("runs/t/\udcff.log") == "runs/t/\\xff.log"


def test_a_name_granted_to_a_contestant_stays_facing_after_the_site_retires_it():
    # A record written while the site held the route names it; the plan's grant still says the
    # contestant held it, whatever the secrets file holds today.
    plan = make_plan(Path("/x"), Path("/c"), "fin_goodput.task")
    agent = plan.stages[0].model_copy(update={"name": "agent", "kind": AgentStandInStage.kind, "params": {},
                                              "after": [], "inputs": {}, "grants": {"RELAY_KEY": "relay"}})
    granted = plan.model_copy(update={"stages": [agent, *plan.stages]})
    assert contestant_facing(granted, {"hf_token": "hf-token-0123456789abcdef"}) == {"relay"}
    assert contestant_facing(plan, {"hf_token": "hf-token-0123456789abcdef"}, ARM) == {"relay"}
    # A name with the granted value is facing too; one the plan grants nothing is not.
    shared = {"relay": "relay-key-0123456789abcdef", "relay_too": "relay-key-0123456789abcdef",
              "hf_token": "hf-token-0123456789abcdef"}
    assert contestant_facing(granted, shared) == {"relay", "relay_too"}


@pytest.mark.parametrize("key, text, matches", [
    ("runs/t/r/a.txt", "runs/t/r/a.txt", True),
    ("runs/t/r/a.txt", "runs/t/r/a.txtx", False),
    ("runs/t/<redacted:k>/a.txt", "runs/t/anything/a.txt", True),
    ("runs/t/<redacted:k>/a.txt", "runs/t//a.txt", False),
    ("<redacted:k>a", "xa", True),
    ("<redacted:k>a", "a", False),
    ("a<redacted:k>", "ab", True),
    ("a<redacted:k>", "a", False),
    ("a<redacted:k>b<redacted:j.host>c", "a1b2c", True),
    ("a<redacted:k>b<redacted:j.host>c", "abbc", False),
    ("a<redacted:k>b<redacted:j.host>c", "a1bb2c", True),
    ("a<redacted:k>bc<redacted:j>bc", "axbcybc", True),
    ("a<redacted:k>bc<redacted:j>bc", "axbcbc", False),
    ("x<redacted:k>x", "xx", False),
    ("x<redacted:k>x", "xyx", True),
    ("a<redacted:k>b", "zab", False),
    ("a<redacted:k>b", "abz", False),
], ids=lambda value: str(value))
def test_a_record_key_stands_for_the_keys_its_redactions_cover_with_at_least_one_character(key, text, matches):
    assert RecordMatcher(key).matches(text) is matches
    # As a pattern each redaction would be: ``.+``, whole.
    pattern = re.compile(".+".join(re.escape(p) for p in re.split(r"<redacted:[^<>]+>", key)), re.DOTALL)
    assert bool(pattern.fullmatch(text)) is matches


def test_matching_a_record_key_of_many_redactions_is_linear_in_the_key_tested(tmp_path):
    # A contestant can name files after the route key; the record's redactions stand for any
    # text, and a sibling built to defeat a backtracking match must still be answered at once.
    key = "runs/t/r/stages/deploy_drive/attempts/1/logs/" + "a<redacted:relay>" * 12 + "z"
    siblings = ["runs/t/r/stages/deploy_drive/attempts/1/logs/" + "a" * n + "b" for n in (50, 500, 5000)]
    matcher = RecordMatcher(key)
    started = time.monotonic()
    assert not any(matcher.matches(text) for text in siblings * 20)
    assert matcher.matches("runs/t/r/stages/deploy_drive/attempts/1/logs/" + "a1" * 12 + "z")
    assert time.monotonic() - started < 0.5


def test_a_rebuild_whose_drive_published_while_its_package_did_not_is_refused(tmp_path):
    plan = with_params(make_plan(tmp_path, Path("/c"), "fin_goodput.task", kind="grading",
                                 artifact={"kind": "package"}), "finalize", built_arm=ARM.model_dump(mode="json"))
    store = PosixStore(plan.store_root)
    drive = publish(store, plan, "deploy_drive", {"drive": write(store, plan, "deploy_drive", "drive.json", b"{}")})
    with pytest.raises(StageFault) as e:
        check_built_arm(store, plan, [drive])
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == "the drive served the package package made, which published no manifest"
    # Before the drive published, the rebuild packaged nothing that was graded.
    check_built_arm(store, plan, [])
