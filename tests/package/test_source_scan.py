"""The source_scan stage: every secret value in a submission's source bundle, read while it is
plain, published in its own manifest before the package stage runs any of the contestant's code.
The checkout role reads every object of its clone of the bundle, its whole history, messages and
tags among them, before the tree's limits decide anything; the tree scan reads the checked-out
tree; ``package-scan.json`` names what they found, never a value."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import pytest

from hls.agent import commit
from hls.package.checkout import CHECKOUT_MEMORY_GIB, CHECKOUT_PEAK_GIB, CHECKOUT_SEC, SourceRecord
from hls.package.scan import FINDINGS_PER_SECRET
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.planner import StageDecl
from hls.pipeline.stage import StageContext, StageFault
from hls.spec.resources import ResourceShape
from hls.pipeline.pipelines import package_shape, source_scan_shape
from hls.stages.package import AGENT_BUILD_STAGE
from hls.stages.provision import role_shape
from hls.stages.source_scan import MIN_TIME_SEC, REST_SEC, SourceScanParams, SourceScanStage, min_disk_gib
from hls.store.cache import ContentCache
from hls.store.digest import sha256_file
from hls.store.manifest import output_ref
from hls.store.posix import PosixStore
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE, PackageScan, PackageScanArea, PackageScanFinding
from hls.substrate.secrets import Scrubber
from tests.package.bundles import GOOD, PYPROJECT, REFUSED, make_bundle, make_history_bundle, make_refused_bundle, stored
from tests.package.test_stage import COPY, INSTANCE, ROLLOUT, SECRET, _agent_build, _left_nothing, _manifest, _pinned_build
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan

#: A secret no stage hands to a contestant, beside the relay's.
TOKEN = "hf-token-0123456789abcdef"
REGISTRY = {"hf_token": TOKEN, "relay": SECRET}
#: The writable disk cap of the build whose bundle the stage scans, in GiB: the least there is.
CAP_GIB = 1
SHAPE = ResourceShape(cpus=2, memory_gib=CHECKOUT_MEMORY_GIB + 4, disk_gib=min_disk_gib(CAP_GIB),
                      time_sec=MIN_TIME_SEC)


def _ctx(tmp_path: Path, *, bundle: Path | None, registry: dict[str, str] | None = REGISTRY,
         launcher: LocalLauncher | None = None, stated: str | None = None,
         build_manifest: str | None = None) -> StageContext:
    """A source_scan stage's context. A named *bundle* is copied into the instance, as ``hls
    package`` copies it, and comes with the digest its operator states, *stated*, by default its
    own. The runner hands the stage every value of the secrets file, *registry*; None for a runner
    that handed it none. *build_manifest* names another instance's agent_build manifest, which the
    stage scans instead of its own."""
    params = {"bundle": None if bundle is None else stored(tmp_path / "store", INSTANCE.source_bundle, bundle),
              "bundle_sha256": None if bundle is None else stated or sha256_file(bundle)[0],
              "build_manifest": build_manifest, "disk_cap_gib": CAP_GIB}
    plan = noop_plan(tmp_path, decls=[StageDecl("source_scan", "source_scan", SHAPE, params)])
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("source_scan", 1, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage("source_scan"), attempt=1,
                        launcher=launcher or LocalLauncher(), watch=None, events=events, inputs={},
                        deadline=time.time() + 120, scrubber=Scrubber(REGISTRY), cache=ContentCache(store),
                        secrets=registry or {}, secret_registry=registry is not None,
                        scratch=tmp_path / "node" / "hls-100", job_id="100")


def _scan(ctx: StageContext, outputs: dict[str, str]) -> PackageScan:
    return PackageScan.model_validate_json(ctx.store.read_bytes(outputs["package_scan"]))


def _started(ctx: StageContext) -> list[str]:
    return [e["role"] for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_started"]


def test_the_tree_and_every_object_of_the_bundle_are_scanned_and_published(tmp_path):
    """A key the agent committed and removed is only in the bundle's history; a token in a file of
    the tree is in the tree and in an object. Both are found, by the checkout's read of every object
    and by the tree's scan, and the record names their secrets, never their values."""
    bundle = make_history_bundle(tmp_path, [({**GOOD, ".env": f"RELAY={SECRET}\n"}, "first"),
                                            ({".env": None, "notes.txt": TOKEN}, "second")])
    ctx = _ctx(tmp_path, bundle=bundle)
    outputs = SourceScanStage().run(ctx)
    assert sorted(outputs) == ["package_scan", "source"]
    record = _scan(ctx, outputs)
    assert record == PackageScan(
        secrets=["hf_token", "relay"], forms=["value", "host"], read=["bundle", "tree"],
        findings=[PackageScanFinding(path="/workspace/notes.txt", secret="hf_token")],
        areas=[PackageScanArea(secret="hf_token", area="bundle", count=1),
               PackageScanArea(secret="relay", area="bundle", count=1)])
    data = ctx.store.read_bytes(outputs["package_scan"])
    assert SECRET.encode() not in data and TOKEN.encode() not in data
    assert outputs["package_scan"].endswith(f"/outputs/{PACKAGE_SCAN_FILE}")
    # Two harness roles, each given the values in its env file only, never on its command line.
    assert _started(ctx) == ["checkout", "package-scan"]
    for spec in ctx.launcher.launched:
        assert spec.owner == "harness" and spec.env_file is not None
        assert SECRET not in " ".join(spec.argv) and TOKEN not in " ".join(spec.argv)
    assert ctx.output_ref(outputs["package_scan"]).sha256 == sha256_file(ctx.store.path(outputs["package_scan"]))[0]


def test_an_over_limit_tree_is_scanned_through_its_objects(tmp_path):
    """A tree holding a symlink is never checked out, and the package will publish invalid_artifact;
    the bundle's objects were read before the limits decided, so the key is found all the same."""
    bundle = make_history_bundle(tmp_path, [({**GOOD, ".env": f"RELAY={SECRET}\n"}, "submission")],
                                 links={"weights": "/data/weights"})
    ctx = _ctx(tmp_path, bundle=bundle)
    outputs = SourceScanStage().run(ctx)
    record = _scan(ctx, outputs)
    assert (record.read, record.findings) == (["bundle"], [])
    assert record.areas == [PackageScanArea(secret="relay", area="bundle", count=1)]
    assert _started(ctx) == ["checkout"]


@pytest.mark.parametrize("pyproject", [PYPROJECT.replace('"httpx>=0.27"', '"vllm"'), '[project]\nversion = "1"\n'],
                         ids=["a-denied-package", "no-project-name"])
def test_a_tree_the_package_will_refuse_is_scanned_all_the_same(tmp_path, pyproject):
    """The package refuses such a tree before any container runs (``tree_problem``); the scan runs
    in a stage of its own, before the package, so it reads the tree whatever the package decides."""
    bundle = make_bundle(tmp_path, {**GOOD, "pyproject.toml": pyproject, ".env": f"RELAY={SECRET}\n"})
    ctx = _ctx(tmp_path, bundle=bundle)
    record = _scan(ctx, SourceScanStage().run(ctx))
    assert record.findings == [PackageScanFinding(path="/workspace/.env", secret="relay")]


def test_a_bundle_without_main_is_not_scanned_and_publishes_no_scan(tmp_path):
    """Nothing is cloned from a bundle without main, so no record claims the bundle was read, and
    finalize keeps it out of any export as unscanned."""
    ctx = _ctx(tmp_path, bundle=make_bundle(tmp_path, GOOD, branch="dev"))
    outputs = SourceScanStage().run(ctx)
    assert sorted(outputs) == ["source"]
    [event] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "source_scan_unscanned"]
    assert "no_main" in event["reason"]


def test_the_scan_reads_this_instances_agent_build_bundle(tmp_path):
    bundle = make_bundle(tmp_path, {**GOOD, ".env": SECRET})
    ctx = _ctx(tmp_path, bundle=None)
    _agent_build(ctx, bundle)
    record = _scan(ctx, SourceScanStage().run(ctx))
    assert record.findings == [PackageScanFinding(path="/workspace/.env", secret="relay")]
    [checkout] = [s for s in ctx.launcher.launched if s.name == "checkout"]
    assert checkout.argv[checkout.argv.index("--size") + 1] == str(bundle.stat().st_size)
    # The checkout holds the bundle to the build's writable disk cap.
    assert checkout.argv[checkout.argv.index("--max-bytes") + 1] == str(CAP_GIB * 2**30)


def test_a_bundle_past_the_builds_disk_cap_is_never_read_or_scanned(tmp_path):
    """The cap the stage holds a bundle to is the build's writable disk cap, the one the commit
    point held the repository to: a recorded bundle past it is refused by its size alone, unread,
    and no record claims it was scanned."""
    ctx = _ctx(tmp_path, bundle=None)
    key = ctx.plan.layout.attempt_file("agent_build", 1, "outputs/source.bundle")
    ctx.store.write_bytes(key, b"")
    with open(ctx.store.path(key), "wb") as f:
        f.truncate(CAP_GIB * 2**30 + 1)
    ctx.store.path(key).chmod(0o200)
    ctx.inputs[ctx.plan.layout.manifest("agent_build")] = _manifest(
        ctx, "agent_build", {"source_bundle": output_ref(ctx.store, key, "0" * 64)})
    outputs = SourceScanStage().run(ctx)
    assert sorted(outputs) == ["source"]
    record = SourceRecord.model_validate_json(ctx.store.read_bytes(outputs["source"]))
    assert record.status == "over_limit" and record.reason == (
        f"the source bundle holds {CAP_GIB * 2**30 + 1} bytes, more than the {CAP_GIB * 2**30} allowed")


def test_the_rest_of_the_stage_is_the_trees_listing_and_check_and_twice_its_checkout_and_scan():
    """Beside the clone and its object scan the stage lists the tree and checks the objects it
    names, each within its own time, then checks it out and scans its files; at the tree's bounds
    (200,000 paths of up to 1,008 bytes in 199,020 directories, 4 GiB of files of three-symbol
    random bytes) the last two take 52.94 s and 54.1 s at their slowest on a build node, and the
    stage allows twice that."""
    assert REST_SEC == commit.TREE_LIST_SEC + commit.KIND_CHECK_SEC + math.ceil(2 * (52.94 + 54.1)) == 504


@pytest.mark.parametrize("cap, disk", [(240, 248), (180, 188), (CAP_GIB, 9)])
def test_the_stage_declares_the_disk_of_a_checkout_at_its_builds_cap_and_a_fixed_time(cap, disk):
    """Its disk is the checkout's at the cap: its clone's pack, as large as the bundle since the
    bundle is not compressed, the tree and git's own files, with no copy of the bundle, which is
    streamed from the store. Its time is the same at any cap: an hour for the checkout's clone and
    object scan (``hls.package.checkout.CHECKOUT_SEC``), whose git runs to the stage's deadline,
    and the rest (``REST_SEC``)."""
    assert min_disk_gib(cap) == cap + 4 + 4 == disk
    assert MIN_TIME_SEC == CHECKOUT_SEC + REST_SEC == 3600 + 504
    shape = source_scan_shape(cap, runner_memory_gib=4)
    assert shape == ResourceShape(cpus=2, memory_gib=CHECKOUT_MEMORY_GIB + 4, disk_gib=disk, time_sec=MIN_TIME_SEC)
    params = SourceScanParams(disk_cap_gib=cap)
    SourceScanStage.check(shape, params, runner_memory_gib=4)
    with pytest.raises(ValueError, match=f"source_scan needs at least {disk} GiB of node-local disk"):
        SourceScanStage.check(shape.model_copy(update={"disk_gib": disk - 1}), params, runner_memory_gib=4)
    with pytest.raises(ValueError, match=re.escape(
            "source_scan needs at least 4104 s of time, the checkout's 3600 s for the clone and the scan of its "
            "objects and 504 s for the tree's listing, checkout and scan; the stage declares 4103")):
        SourceScanStage.check(shape.model_copy(update={"time_sec": MIN_TIME_SEC - 1}), params, runner_memory_gib=4)


@pytest.mark.parametrize("runner", [4, 6])
def test_the_checkout_role_gets_its_measured_peak_and_a_quarter_in_both_stages_that_run_it(runner):
    """The checkout's worst peak at the commit point's bounds, 8.444 GiB (three 4 GiB
    bases, each with a 512 MiB delta: git holds two bases and the object it rebuilds), and a quarter more, rounded up: 11 GiB, which source_scan's and the package's
    roles get beside the runner's reserve, and which each stage refuses to run without."""
    assert CHECKOUT_PEAK_GIB == 8.444 and CHECKOUT_MEMORY_GIB == math.ceil(8.444 * 1.25) == 11
    for cap in (180, 240):
        shape = source_scan_shape(cap, runner_memory_gib=runner)
        assert shape.memory_gib == CHECKOUT_MEMORY_GIB + runner
        assert role_shape(shape, runner).memory_gib == CHECKOUT_MEMORY_GIB
        assert role_shape(package_shape(cap), runner).memory_gib >= CHECKOUT_MEMORY_GIB
    params = SourceScanParams(disk_cap_gib=CAP_GIB)
    SourceScanStage.check(SHAPE, params, runner_memory_gib=4)
    with pytest.raises(ValueError, match=f"source_scan's checkout needs {CHECKOUT_MEMORY_GIB} GiB of memory"):
        SourceScanStage.check(SHAPE.model_copy(update={"memory_gib": CHECKOUT_MEMORY_GIB + 3}), params,
                              runner_memory_gib=4)


def test_the_scan_reads_another_instances_pinned_agent_build_bundle(tmp_path):
    """A grading that packages a rollout's source again scans that rollout's bundle, pinned by its
    agent_build manifest, at the grading's checkout; this instance's own is not read."""
    ctx = _ctx(tmp_path, bundle=None, build_manifest=ROLLOUT.manifest(AGENT_BUILD_STAGE))
    _agent_build(ctx, make_bundle(tmp_path / "own", GOOD))
    pinned = make_bundle(tmp_path / "pinned", {**GOOD, ".env": SECRET})
    _pinned_build(ctx, bundle=pinned)
    record = _scan(ctx, SourceScanStage().run(ctx))
    assert record.findings == [PackageScanFinding(path="/workspace/.env", secret="relay")]
    [checkout] = [s for s in ctx.launcher.launched if s.name == "checkout"]
    assert checkout.argv[checkout.argv.index("--size") + 1] == str(pinned.stat().st_size)


def test_a_pinned_agent_build_that_left_nothing_leaves_nothing_to_scan(tmp_path):
    ctx = _ctx(tmp_path, bundle=None, build_manifest=ROLLOUT.manifest(AGENT_BUILD_STAGE))
    _agent_build(ctx, make_bundle(tmp_path, GOOD))
    _pinned_build(ctx, nothing=True)
    assert SourceScanStage().run(ctx) == {} and ctx.launcher.launched == []


def test_a_named_bundle_and_a_pinned_agent_build_cannot_both_be_scanned():
    with pytest.raises(ValueError, match="a named bundle or an agent_build manifest's, not both"):
        SourceScanStage.params_model.model_validate({"bundle": COPY, "bundle_sha256": "ab" * 32,
                                                     "build_manifest": ROLLOUT.manifest(AGENT_BUILD_STAGE),
                                                     "disk_cap_gib": CAP_GIB})


def test_an_agent_build_that_left_nothing_to_package_leaves_nothing_to_scan(tmp_path):
    ctx = _ctx(tmp_path, bundle=None)
    _left_nothing(ctx)
    assert SourceScanStage().run(ctx) == {} and ctx.launcher.launched == []


def test_a_site_with_no_secret_records_that_nothing_was_read(tmp_path):
    """With no secret to find, nothing is read for one: the record says so, and covers nothing on a
    site that holds a secret."""
    ctx = _ctx(tmp_path, bundle=make_bundle(tmp_path, GOOD), registry={})
    outputs = SourceScanStage().run(ctx)
    assert _scan(ctx, outputs) == PackageScan(secrets=[], forms=["value", "host"], findings=[], read=[])
    assert _started(ctx) == ["checkout"]
    [checkout] = ctx.launcher.launched
    assert "--names" not in checkout.argv and checkout.env_file is None


def test_the_stage_refuses_to_run_without_the_secrets_registry(tmp_path):
    ctx = _ctx(tmp_path, bundle=make_bundle(tmp_path, GOOD), registry=None)
    with pytest.raises(StageFault, match="only with the secrets registry") as caught:
        SourceScanStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert ctx.launcher.launched == []


def test_a_bundle_that_is_not_the_stated_one_is_refused_and_never_retried(tmp_path):
    ctx = _ctx(tmp_path, bundle=make_bundle(tmp_path, GOOD), stated="ab" * 32)
    with pytest.raises(StageFault, match="is not the sha256 stated for it") as caught:
        SourceScanStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable


@pytest.mark.parametrize("refused", sorted(REFUSED))
def test_a_bundle_the_checkout_refuses_is_the_submissions_and_no_fault(tmp_path, refused):
    """The commit point writes its bundle once and reads nothing back, so its bundle may be one git
    does not read back, git having bundled from a store whose objects lied, or one whose tree
    names a tree as a file. The source scan is the first to read it: the checkout proves every
    byte it gives git has the digest the agent_build manifest records, so git's verdict is the
    submission's, ``source.json`` says over_limit with the verdict, nothing is checked out, and
    the stage raises no fault: the package stage publishes the submission's zero. A bundle that
    does not read back has no object read, so no scan record claims it (finalize keeps it out of
    any export as unscanned); a tree naming a tree as a file is refused once every object is
    scanned."""
    ctx = _ctx(tmp_path, bundle=None)
    _agent_build(ctx, make_refused_bundle(tmp_path, refused))
    outputs = SourceScanStage().run(ctx)
    record = SourceRecord.model_validate_json(ctx.store.read_bytes(outputs["source"]))
    assert (record.status, record.files) == ("over_limit", [])
    assert re.fullmatch(REFUSED[refused], record.reason or ""), record.reason
    unscanned = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "source_scan_unscanned"]
    if refused == "kind":
        assert sorted(outputs) == ["package_scan", "source"] and unscanned == []
    else:
        assert sorted(outputs) == ["source"] and len(unscanned) == 1 and "over_limit" in unscanned[0]["reason"]
    assert _started(ctx) == ["checkout"] and ctx.contestant_fault() is None


@pytest.mark.parametrize("change", ["a-byte-changed", "a-byte-added"])
def test_a_bundle_the_store_changed_after_the_commit_point_wrote_it_is_infra_and_never_retried(tmp_path, change):
    """The commit point never reads its bundle back once written, so a store that changes the
    bundle afterwards is first seen here: the checkout refuses a file that is not the size or the
    digest the agent_build manifest records, before any verdict on its content, and the stage
    raises infra that is not retried, since a retry would read the same file. The workspace is gone
    by then, so nothing can bundle it again; nothing is charged to the contestant. The fault names
    the checkout role and the file's own size or digest."""
    ctx = _ctx(tmp_path, bundle=None)
    key = _agent_build(ctx, make_bundle(tmp_path, GOOD))
    data = ctx.store.read_bytes(key)
    ctx.store.write_bytes(key, bytes([data[0] ^ 1]) + data[1:] if change == "a-byte-changed" else data + b"x")
    with pytest.raises(StageFault) as caught:
        SourceScanStage().run(ctx)
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.evidence["role"] == "checkout"
    found = "has sha256 [0-9a-f]{64}, not" if change == "a-byte-changed" else f"holds {len(data) + 1} bytes, not the"
    assert re.search(f"the source bundle does not match its manifest: file .* {found}", fault.reason)
    assert ctx.contestant_fault() is None and not ctx.output_path("source.json").exists()


def test_a_read_of_the_bundle_that_went_wrong_on_the_way_is_infra_and_retried(tmp_path):
    """A read that went wrong on the way, the file where it lies still the recorded bundle, is the
    read's failure: the stage raises infra that is retried, and nothing is charged to the
    contestant."""
    stub = tmp_path / "misread.py"
    stub.write_text("import sys\n"
                    "from hls.package import checkout\n"
                    "real = checkout._chunks\n"
                    "def chunks(fd, path):\n"
                    "    for at, chunk in enumerate(real(fd, path)):\n"
                    "        yield bytes([chunk[0] ^ 1]) + chunk[1:] if at == 0 else chunk\n"
                    "checkout._chunks = chunks\n"
                    "sys.exit(checkout.main())\n")
    ctx = _ctx(tmp_path, bundle=None, launcher=LocalLauncher(replace_modules={"hls.package.checkout": stub}))
    _agent_build(ctx, make_bundle(tmp_path, GOOD))
    with pytest.raises(StageFault) as caught:
        SourceScanStage().run(ctx)
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert ctx.contestant_fault() is None and not ctx.output_path("source.json").exists()


def test_the_stage_runs_no_contestant_code_and_says_what_it_needs():
    assert not SourceScanStage.runs_contestant_code
    assert SourceScanStage.needs_secret_registry and SourceScanStage.max_roles(None) == 2
    # The paths it records of the tree are the contestant's names.
    assert SourceScanStage.contestant_outputs == ("source.json",)
    with pytest.raises(ValueError, match="needs at least"):
        SourceScanStage.check(SHAPE.model_copy(update={"disk_gib": SHAPE.disk_gib - 1}),
                              SourceScanParams(disk_cap_gib=CAP_GIB), runner_memory_gib=4)


def test_a_key_in_more_files_than_the_bound_is_recorded_by_its_area_and_count(tmp_path):
    """The tree's scan lists a secret's first files up to its bound and records the rest by area and
    count, so the record stays small however many files the contestant wrote the key into."""
    files = {**GOOD, **{f"keys/k{i:04}.txt": SECRET for i in range(FINDINGS_PER_SECRET + 5)}}
    ctx = _ctx(tmp_path, bundle=make_bundle(tmp_path, files))
    record = _scan(ctx, SourceScanStage().run(ctx))
    assert len(record.findings) == FINDINGS_PER_SECRET and record.findings[0].path == "/workspace/keys/k0000.txt"
    # The files are one object, as git stores equal bytes once.
    assert record.areas == [PackageScanArea(secret="relay", area="bundle", count=1),
                            PackageScanArea(secret="relay", area="tree", count=FINDINGS_PER_SECRET + 5)]
