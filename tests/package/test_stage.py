from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hls.agent.commit import NoSubmission
from hls.egress.denied_packages import DENIED_PACKAGES
from hls.egress.policy import AccessList, NetworkPolicy, SealPolicy
from hls.egress.proxy import UPSTREAM_FETCHES_MAX
from hls.package.checkout import CHECKOUT_MEMORY_GIB, CHECKOUT_SEC, TREE_MAX_BYTES
from hls.package.integrity import IntegrityRecord
from hls.package.offline import INSTALL_MAX_GIB
from hls.package.result import PackageResult
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.pipelines import NOOP_TASK, package_shape
from hls.pipeline.planner import StageDecl
from hls.pipeline import stage as pipeline_stage
from hls.pipeline.stage import StageContext, StageFault
from hls.recipes.model import CONSTRAINTS
from hls.settings import RunSettings
from hls.spec.resources import GrantedResources, ResourceShape
from hls.stages import package as package_stage
from hls.stages.images import BUILD_DISK_GIB, KEPT_DISK_GIB
from hls.stages.package import (
    AGENT_BUILD_STAGE,
    IMAGES_OUTPUT,
    NO_SUBMISSION_OUTPUT,
    PYPROJECT_MAX_BYTES,
    REPORT_MAX_BYTES,
    REPORT_MAX_DEPTH,
    REPORT_REASON_CHARS,
    SERVED_UCOUNTS_OUTPUT,
    SOURCE_BUNDLE_OUTPUT,
    WORK_DISK_GIB,
    ForeignReport,
    PackageStage,
    PipLimits,
    pip_limits,
    build_verdict,
    contestant_verdict,
    index_outage,
    pyproject_problem,
    read_report,
    report_pins,
    wheels_problem,
)
from hls.store.cache import ContentCache, copies_root, entry_sha256
from hls.store.digest import sha256_bytes, sha256_file
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import (
    Manifest,
    OutputRef,
    Provenance,
    Resources,
    Times,
    output_ref,
    read_outputs_report,
)
from hls.store.posix import PosixStore
from hls.substrate import seal as seal_mod
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer import images as apptainer_images
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.apptainer.egress import CLOSED_DNS_VAR
from hls.substrate.seal import IndexStatus, payload_ended_at
from hls.substrate.base import FaultRecord
from hls.substrate.faults import RoleOutcome
from hls.substrate.sandbox import IMAGE_FILE
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE, PackageScan, PackageScanFinding
from hls.substrate.ucounts import ServedUcounts, served_limits
from hls.substrate.secrets import Scrubber
from tests.package import fake_tools
from tests.package.bundles import (GOOD, PYPROJECT, REFUSED, SERVER, SYMLINKS, make_bundle, make_refused_bundle,
                                   make_tree_bundle, stored)
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan
from tests.store.entries import entry_reads, populate, recording_reads
from tests.substrate import fake_pasta

#: The writable disk cap of the build whose bundle the stage packages, in GiB: the least there is.
CAP_GIB = 1
#: The instance of every stage's plan (``noop_plan``), into which a named bundle is copied.
INSTANCE = InstanceLayout.rollout(NOOP_TASK, "r1")
#: A plan's record of a named bundle's copy.
COPY = {"key": INSTANCE.source_bundle, "size": 1}
SHAPE = ResourceShape(cpus=4, memory_gib=CHECKOUT_MEMORY_GIB + 4, disk_gib=package_stage.min_disk_gib(CAP_GIB),
                      time_sec=10 * 3600)
#: What each pip step of a stage of ``SHAPE`` gets under the default runner reserve, as the plan
#: states it to the stage and the build's brief.
PIP = pip_limits(SHAPE, RunSettings().runner_memory_gib)
BASE = "s-a-tf-5-13-1"
BASE_KEY = "s-a-tf-5-13-1-0123456789abcdef"
SECRET = "relay-secret-0123456789"
POLICY = NetworkPolicy(domains=AccessList(mode="allow"), packages=AccessList(mode="deny", entries=list(DENIED_PACKAGES)))
REPORT = {"version": "1", "install": [
    {"metadata": {"name": "subm", "version": "0.1"}, "download_info": {"url": "@workspace"}, "is_direct": True},
    {"metadata": {"name": "httpx", "version": "0.27.0"},
     "download_info": {"url": "https://files.pythonhosted.org/p/httpx-0.27.0-py3-none-any.whl"}, "is_direct": False},
]}
SUBMISSION_WHEEL = "subm-0.1-py3-none-any.whl"
#: How long after a command's own first instant the stage may first see it started: one pass of
#: its wait, and half a second for a loaded machine.
SEEN_WITHIN_SEC = pipeline_stage._POLL_SEC + 0.5
#: Errors a read of what the submission produced can end with, each with whether it is the
#: submission's: only its own permissions are; the rest are the node's.
ERRNO_SIDES = [(errno.EACCES, True), (errno.EPERM, True), (errno.EIO, False), (errno.ESTALE, False),
               (errno.ENOSPC, False)]
ERRNO_IDS = ["EACCES", "EPERM", "EIO", "ESTALE", "ENOSPC"]


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """The fake apptainer (exec, build and extraction), pip and pasta on PATH; returns (bin dir,
    plan file)."""
    bin_dir, plan = tmp_path / "bin", tmp_path / "tools.json"
    fake_pasta.install(bin_dir)
    fake_tools.install(bin_dir, plan)
    plan.write_text(json.dumps({"dry_run": "ok", "wheel": "ok", "build": "ok", "report": REPORT,
                                "record": str(tmp_path / "tools.log")}))
    (tmp_path / "pasta.plan").write_text("pass\n" * 4)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    # The seal resolves its firewall tools from here, not from the system's /usr/sbin.
    monkeypatch.setenv("HLS_SEAL_TOOL_PATH", str(bin_dir))
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(tmp_path / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(tmp_path / "pasta.log"))
    monkeypatch.setenv("FAKE_PYTHON_COUNTER", str(tmp_path / "counter"))
    for knob in ("FAKE_APPTAINER_FAIL", "FAKE_EXTRACT_FAIL", "FAKE_PROXY_ERRORS", "FAKE_PROXY_MID_STREAM",
                 "FAKE_PROXY_OVERSIZED", "FAKE_PROXY_REFUSED", "FAKE_PROXY_HEARTBEAT_AGE"):
        monkeypatch.delenv(knob, raising=False)
    return bin_dir, plan


def _set(plan: Path, **changes) -> None:
    plan.write_text(json.dumps({**json.loads(plan.read_text()), **changes}))


def _manifest(ctx: StageContext, stage: str, outputs: dict[str, OutputRef]) -> Manifest:
    return Manifest(instance=ctx.plan.instance, stage=stage, attempt=1, outputs=outputs, inputs={},
                    provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                                          image_digests={}, config_sha256="d" * 64),
                    resources=Resources(declared=SHAPE, granted=GrantedResources(gpus=0, cpus=4, memory_gib=8.0,
                                                                                 disk_free_gib=100.0)),
                    times=Times(submitted_at=1.0, allocation_started_at=2.0, started_at=2.5, ended_at=3.0))


def _images(ctx: StageContext, record: dict | None = None,
            tree: dict[str, list] = fake_tools.BASE_TREE) -> Manifest:
    """This instance's images stage, as it publishes ``images.json`` after filling the base with a
    tree image of *tree*."""
    assert ctx.cache is not None
    found = populate(ctx.cache, "images", BASE_KEY,
                     lambda d: (d / IMAGE_FILE).write_bytes(fake_tools.tree_image(tree)),
                     holder=Holder.here("test")).entry
    sif = next(f.sha256 for f in found.manifest.files if f.path == IMAGE_FILE)
    if record is None:
        record = {BASE: {"key": BASE_KEY, "entry_sha256": entry_sha256(found), "sif_sha256": sif}}
    key = ctx.plan.layout.attempt_file("images", 1, "outputs/images.json")
    data = json.dumps(record).encode()
    ctx.store.write_bytes(key, data)
    return _manifest(ctx, "images", {IMAGES_OUTPUT: output_ref(ctx.store, key, sha256_bytes(data))})


#: The figures a build's brief stated for its graded server, which it records beside the brief.
SERVED = ServedUcounts(limits=served_limits({"grading_node_namespaces": 1_048_576,
                                             "grading_node_inotify_instances": 128,
                                             "grading_node_inotify_watches": 1_048_576}, 8))


def _served(ctx: StageContext) -> dict:
    """This instance's agent_build's record of the figures its brief stated (``SERVED``), as an
    output of its manifest."""
    key = ctx.plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/served_ucounts.json")
    data = SERVED.model_dump_json().encode()
    ctx.store.write_bytes(key, data)
    return {SERVED_UCOUNTS_OUTPUT: output_ref(ctx.store, key, sha256_bytes(data))}


def _agent_build(ctx: StageContext, bundle: Path, *, served: bool = True) -> str:
    """This instance's agent_build, as it publishes *bundle*'s bytes and, with *served*, its
    record of the figures its brief stated: the bundle output's key."""
    key = ctx.plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/source.bundle")
    data = bundle.read_bytes()
    ctx.store.write_bytes(key, data)
    ctx.inputs[ctx.plan.layout.manifest(AGENT_BUILD_STAGE)] = _manifest(
        ctx, AGENT_BUILD_STAGE, {SOURCE_BUNDLE_OUTPUT: output_ref(ctx.store, key, sha256_bytes(data)),
                                 **(_served(ctx) if served else {})})
    return key


#: The site's secrets file as the runner hands it to the stage, which needs the whole registry.
REGISTRY = {"relay": SECRET}


def _ctx(tmp_path: Path, bin_dir: Path, *, bundle: Path | None, network: NetworkPolicy = POLICY,
         images: dict | None = None, launcher: LocalLauncher | None = None,
         bundle_sha256: str | None = None, registry: dict[str, str] | None = REGISTRY,
         build_manifest: str | None = None, constraints: str | None = CONSTRAINTS,
         base_tree: dict[str, list] = fake_tools.BASE_TREE) -> StageContext:
    """A package stage's context. A named *bundle* is copied into the instance, as ``hls package``
    copies it, and comes with the digest the operator states, *bundle_sha256*, by default the
    bundle's own. The runner hands the stage every value of the
    secrets file, *registry*; None for a runner that handed it none. *build_manifest* names
    another instance's agent_build manifest, which the stage packages instead of its own. The
    submission base's file is a tree image of *base_tree*, and the plan gives *constraints* as
    its constraints file."""
    if bundle is not None and bundle_sha256 is None:
        bundle_sha256 = sha256_file(bundle)[0]
    params = {"task": "multitenant_slo", "submission_base": BASE, "constraints": constraints, "pip": PIP.model_dump(),
              "network": network.model_dump(mode="json"), "gpu_kind": "H200",
              "bundle": None if bundle is None else stored(tmp_path / "store", INSTANCE.source_bundle, bundle),
              "bundle_sha256": bundle_sha256,
              "build_manifest": build_manifest, "disk_cap_gib": CAP_GIB}
    plan = noop_plan(tmp_path, decls=[StageDecl("package", "package", SHAPE, params)],
                     site={"pasta_path": str(bin_dir / "pasta")})
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("package", 1, "events.jsonl")))
    ctx = StageContext(store=store, plan=plan, stage=plan.stage("package"), attempt=1,
                       launcher=(launcher or LocalLauncher()).use_plan(plan), watch=None, events=events, inputs={},
                       deadline=time.time() + 120, scrubber=Scrubber({"relay": SECRET}), cache=ContentCache(store),
                       secrets=registry or {}, secret_registry=registry is not None,
                       scratch=tmp_path / "node" / "hls-100", job_id="100", runtime=ApptainerRuntime(),
                       images=ApptainerImages(tmp_path / "node" / "hls-100"))
    ctx.inputs = {plan.layout.manifest("images"): _images(ctx, images, base_tree)}
    return ctx


def _result(ctx: StageContext, outputs: dict[str, str]) -> PackageResult:
    return PackageResult.model_validate_json(ctx.store.read_bytes(outputs["package"]))


def _started(ctx: StageContext) -> list[str]:
    roles = [e["role"] for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_started"]
    return ["extract" if r.startswith("extract-") else r for r in roles]


def test_a_good_bundle_builds_a_submission_image_and_records_its_digest(tmp_path, tools):
    bin_dir, plan = tools
    bundle = make_bundle(tmp_path, GOOD)
    ctx = _ctx(tmp_path, bin_dir, bundle=bundle)
    outputs = PackageStage().run(ctx)
    result = _result(ctx, outputs)
    image = ctx.store.path(outputs["submission"])
    built = ctx.role_dir("build") / "submission.sif"
    assert result.status == "built" and result.image == outputs["submission"]
    assert result.sif_sha256 == sha256_file(image)[0] == sha256_file(built)[0] == ctx.image_digests["submission"]
    assert result.bundle_sha256 == sha256_file(bundle)[0] and len(result.commit or "") == 40
    assert result.wheels == ["httpx==0.27.0"]
    # The base is extracted before any other role, the new image before its check; each pip
    # step's copy and temporary files go once it has ended, and the wheels are read whole
    # before the build installs them.
    assert _started(ctx) == ["extract", "checkout", "dry-run-copy", "dry-run", "dry-run-clean", "wheel-copy",
                             "wheel", "wheel-clean", "wheel-check", "build", "extract", "integrity", "install-scan",
                             "publish"]
    # Contestant code runs only in the two pip containers; everything else is ours, and the
    # runner copied and removed no tree: each is a role of its own.
    owners = {spec.name: spec.owner for spec in ctx.launcher.launched if not spec.name.startswith("extract-")}
    assert owners == {"checkout": "harness", "dry-run-copy": "harness", "dry-run": "contestant", "dry-run-clean": "harness", "wheel-copy": "harness",
                      "wheel": "contestant", "wheel-clean": "harness", "wheel-check": "harness", "build": "harness",
                      "integrity": "harness", "install-scan": "harness", "publish": "harness"}
    check = next(s for s in ctx.launcher.launched if s.name == "wheel-check")
    wheels = str(ctx.role_dir("wheel") / "wheels")
    assert check.argv[1:5] == ("-m", "hls.package.wheel_check", "--wheels", wheels)
    for step in ("dry-run", "wheel"):
        assert not (ctx.role_dir(step) / "workspace").exists() and not (ctx.role_dir(step) / "build-tmp").exists()
    assert (ctx.role_dir("wheel") / "wheels" / SUBMISSION_WHEEL).is_file()
    # The pip steps' seals refuse their own DNS: the producer's policy reaches no host.
    assert {s.name: s.env.get(CLOSED_DNS_VAR) for s in ctx.launcher.launched if s.owner == "contestant"} == {
        "dry-run": "closed", "wheel": "closed"}
    # Nothing of the harness runs in a pip step's container, so it holds no capability.
    assert {s.name: s.argv[s.argv.index("--drop-caps") + 1] for s in ctx.launcher.launched
            if s.owner == "contestant"} == {"dry-run": "ALL", "wheel": "ALL"}
    copies = {spec.name: list(spec.argv) for spec in ctx.launcher.launched if spec.name.endswith("-copy")}
    tree = str(ctx.role_dir("checkout") / "tree")
    assert copies == {f"{step}-copy": ["cp", "-a", "--", tree, str(ctx.role_dir(step) / "workspace")]
                      for step in ("dry-run", "wheel")}
    assert list(ctx.cache_entries) == [f"images/{BASE_KEY}"]
    record = IntegrityRecord.model_validate_json(ctx.store.read_bytes(outputs["integrity"]))
    assert record.differences == 0 and "usr" in record.compared
    log = fake_tools.records(tmp_path / "tools.log")
    assert log[1]["pins"] == ["httpx==0.27.0"]
    definition = log[2]["def"]
    assert "--only-binary :all: --find-links /wheels subm ||" in definition and "pip install ." not in definition
    # The install applies the base's constraints file, which is checked to be there first.
    assert f"[ -f {CONSTRAINTS} ]" in definition and f"PIP_CONSTRAINT={CONSTRAINTS} " in definition
    # The build bootstraps from its own checked copy of the base's file, which it then removes.
    assert definition.splitlines()[1] == f"From: {ctx.role_dir('build') / IMAGE_FILE}"
    assert not (ctx.role_dir("build") / IMAGE_FILE).exists()
    # The image is built from the checkout's own tree, never a copy the build hooks could change.
    assert f"    {ctx.role_dir('checkout') / 'tree'} /srv-stage" in definition
    assert log[2]["env"]["APPTAINER_TMPDIR"] == str(ctx.role_dir("build") / "build-tmp")
    assert json.loads(ctx.store.read_bytes(outputs["install_report"]))["install"][1]["metadata"]["name"] == "httpx"
    # Every output reaches the manifest by its writer's digest, the bytes on disk: a role's own
    # report for what it wrote, the stage's for its small records.
    assert {name: ctx.output_ref(key).sha256 for name, key in outputs.items()} == {
        name: sha256_file(ctx.store.path(key))[0] for name, key in outputs.items()}
    writers = {"source": "checkout", "integrity": "integrity", "submission": "publish",
               "package_scan": "install-scan"}
    for name, role in writers.items():
        reported = read_outputs_report(ctx.outputs_report(role))
        assert reported is not None and list(reported) == [Path(outputs[name]).name], role
    # Both contestant steps ran sealed, each on its own copy of the tree in its own role
    # directory, and saw the run's GPU arch and the role's CPUs.
    # Neither forwards a port either way, nor maps its gateway's address to the host's loopback.
    calls = fake_pasta.calls(tmp_path / "pasta.log")
    assert len(calls) == 2 and all(fake_pasta.forwards(flags) == fake_pasta.NO_FORWARDS for flags in calls)
    assert all("--no-map-gw" in flags for flags in calls)
    assert [entry["workspace"] for entry in log[:2]] == [str(ctx.role_dir(step) / "workspace")
                                                        for step in ("dry-run", "wheel")]
    assert {"TORCH_CUDA_ARCH_LIST", "MAX_JOBS", "PIP_INDEX_URL"} <= set(log[0]["env"])
    assert log[0]["build_env"] == {"MAX_JOBS": str(SHAPE.cpus), "TORCH_CUDA_ARCH_LIST": "9.0"}
    # Every role got the stage's shape less the runner's 4 GiB.
    assert {spec.shape.memory_gib for spec in ctx.launcher.launched} == {SHAPE.memory_gib - 4} == {CHECKOUT_MEMORY_GIB}


def test_a_wheel_whose_data_lands_in_the_venv_leaves_the_system_paths_alone(tmp_path, tools):
    """A wheel carrying ``.data/data/bin/sh`` installs it as ``/opt/venv/bin/sh``: the base's
    ``/bin/sh``, which prints the container's start marker, is untouched, and the image builds."""
    bin_dir, plan = tools
    _set(plan, wheel_entries={"subm-0.1.data/data/bin/sh": "#!evil\n"},
         image_changes={"opt/venv/bin/sh": ["file", "#!evil\n", 0o755]})
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    outputs = PackageStage().run(ctx)
    assert _result(ctx, outputs).status == "built"
    assert IntegrityRecord.model_validate_json(ctx.store.read_bytes(outputs["integrity"])).differences == 0


def test_a_wheel_that_changed_a_system_path_makes_the_submission_invalid(tmp_path, tools):
    """A wheel whose entry lands on ``/usr/bin/sh`` and an image whose ``/usr/bin/sh`` changed:
    the change is certainly the submission's, so the image is refused as invalid and not
    published."""
    bin_dir, plan = tools
    _set(plan, wheel_entries={"subm-0.1.data/data/../../../usr/bin/sh": "#!evil\n"},
         image_changes={"usr/bin/sh": ["file", "#!evil\n", 0o755]})
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    outputs = PackageStage().run(ctx)
    result = _result(ctx, outputs)
    assert result.status == "invalid_artifact" and "submission" not in outputs
    assert f"['{SUBMISSION_WHEEL}']" in (result.reason or "") and "/usr/bin/sh" in (result.reason or "")
    record = IntegrityRecord.model_validate_json(ctx.store.read_bytes(outputs["integrity"]))
    assert [(d.path, d.wheels) for d in record.listed] == [("/usr/bin/sh", [SUBMISSION_WHEEL])]
    assert "submission" not in ctx.image_digests and _started(ctx)[-1] == "integrity"
    # The invalid result publishes with every output by its writer's digest.
    assert set(outputs) == {"source", "install_report", "integrity", "package"}
    assert "install-scan" not in _started(ctx)
    assert {name: ctx.output_ref(key).sha256 for name, key in outputs.items()} == {
        name: sha256_file(ctx.store.path(key))[0] for name, key in outputs.items()}


def test_a_changed_system_path_only_a_wheel_pip_left_out_names_is_not_the_submissions(tmp_path, tools):
    """Only the wheels pip installed wrote anything: a change where only a wheel it left out
    would have written is not certainly the submission's."""
    bin_dir, plan = tools
    _set(plan, wheel_entries={"subm-0.1.data/data/../../../usr/bin/sh": "#!evil\n"},
         image_changes={"usr/bin/sh": ["file", "#!evil\n", 0o755]}, not_installed=[SUBMISSION_WHEEL])
    with pytest.raises(StageFault, match="1 of them not accounted for") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD)))
    assert caught.value.fault.fault_class == "infra"


@pytest.mark.parametrize("changes", [
    {"etc/passwd": ["file", "evil\n", 0o644]},
    {"usr/bin/sh": ["file", "#!evil\n", 0o755], "etc/passwd": ["file", "evil\n", 0o644]},
], ids=["harness-side", "partly-unaccounted"])
def test_a_difference_the_wheels_do_not_account_for_is_infra_and_publishes_nothing(tmp_path, tools, changes):
    bin_dir, plan = tools
    _set(plan, wheel_entries={"subm-0.1.data/data/../../../usr/bin/sh": "#!evil\n"}, image_changes=changes)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    with pytest.raises(StageFault, match="not accounted for by installing .*: .*/etc/passwd") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert not ctx.store.exists(ctx.output_key("package.json"))
    assert not ctx.store.exists(ctx.output_key("submission.sif"))


def test_an_integrity_check_that_fails_itself_is_a_retryable_fault(tmp_path, tools):
    bin_dir, _ = tools
    crash = tmp_path / "crash.py"
    crash.write_text("import sys\nsys.exit(1)\n")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD),
               launcher=LocalLauncher(replace_modules={"hls.package.integrity": crash}))
    with pytest.raises(StageFault, match="harness role integrity ended exited with status 1") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable


def test_the_bundle_can_come_from_this_instances_agent_build(tmp_path, tools):
    bin_dir, _ = tools
    bundle = make_bundle(tmp_path, GOOD)
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    _agent_build(ctx, bundle)
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "built" and result.bundle_sha256 == sha256_file(bundle)[0]
    # The package carries the figures the build's brief stated for its graded server, which its
    # grading's seal writes.
    assert result.served_ucounts == SERVED
    # The checkout role read the bundle once where it lies, streaming it into git as it hashed it,
    # and checked it against the manifest's digest; no copy was made, and this process read none
    # of it.
    checkout = next(s for s in ctx.launcher.launched if s.name == "checkout")
    assert checkout.argv[-6:] == ("--size", str(bundle.stat().st_size), "--expect", sha256_file(bundle)[0],
                                  "--max-bytes", str(_CAP))
    assert "--copy" not in checkout.argv


#: The rollout another instance's tests pin the agent_build manifest of, as a grading that packages
#: a rollout's source again pins it.
ROLLOUT = InstanceLayout("runs/multitenant_slo/r-built")


def _pinned_build(ctx: StageContext, *, bundle: Path | None = None, nothing: bool = False) -> None:
    """Put the rollout's published agent_build manifest among *ctx*'s inputs, as the runner hands a
    stage a manifest its plan pins: with *bundle*'s bytes as its source bundle, or with a
    no_submission record (*nothing*)."""
    outputs = {}
    if bundle is not None:
        key = ROLLOUT.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/source.bundle")
        ctx.store.write_bytes(key, bundle.read_bytes())
        outputs[SOURCE_BUNDLE_OUTPUT] = output_ref(ctx.store, key, sha256_bytes(bundle.read_bytes()))
    if nothing:
        key = ROLLOUT.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/no_submission.json")
        data = NoSubmission("no_main", NO_MAIN, 1.0).to_json().encode()
        ctx.store.write_bytes(key, data)
        outputs[NO_SUBMISSION_OUTPUT] = output_ref(ctx.store, key, sha256_bytes(data))
    ctx.inputs[ROLLOUT.manifest(AGENT_BUILD_STAGE)] = _manifest(ctx, AGENT_BUILD_STAGE, outputs).model_copy(
        update={"instance": ROLLOUT.root})


def test_the_bundle_can_come_from_another_instances_pinned_agent_build(tmp_path, tools):
    """A grading that packages a rollout's source again pins the rollout's agent_build manifest:
    the stage packages that manifest's bundle, never one of its own instance."""
    bin_dir, _ = tools
    pinned, own = make_bundle(tmp_path / "pinned", GOOD), make_bundle(tmp_path / "own", {**GOOD, "x.txt": "own"})
    ctx = _ctx(tmp_path, bin_dir, bundle=None, build_manifest=ROLLOUT.manifest(AGENT_BUILD_STAGE))
    _agent_build(ctx, own)
    _pinned_build(ctx, bundle=pinned)
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "built" and result.bundle_sha256 == sha256_file(pinned)[0] != sha256_file(own)[0]


def test_a_pinned_agent_build_that_left_nothing_is_packaged_as_nothing(tmp_path, tools):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None, build_manifest=ROLLOUT.manifest(AGENT_BUILD_STAGE))
    _agent_build(ctx, make_bundle(tmp_path, GOOD))
    _pinned_build(ctx, nothing=True)
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and "the build left nothing to package" in result.reason


def test_a_pinned_agent_build_missing_from_the_inputs_is_refused_not_replaced_by_this_instances(tmp_path, tools):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None, build_manifest=ROLLOUT.manifest(AGENT_BUILD_STAGE))
    _agent_build(ctx, make_bundle(tmp_path, GOOD))
    with pytest.raises(StageFault, match=f"the agent_build manifest {ROLLOUT.manifest(AGENT_BUILD_STAGE)} the plan "
                                         "pins is not among the stage's inputs") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable


def test_a_pinned_manifest_of_another_stage_is_refused(tmp_path, tools):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None, build_manifest=ROLLOUT.manifest("package"))
    ctx.inputs[ROLLOUT.manifest("package")] = _manifest(ctx, "package", {}).model_copy(
        update={"instance": ROLLOUT.root})
    with pytest.raises(StageFault, match="is the manifest of stage package, not of an agent_build") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable


def test_a_bundle_that_differs_from_its_manifest_is_refused(tmp_path, tools):
    """The checkout role reads the bundle once, before git does, and refuses one whose digest is
    not the manifest's: a retry would read the same file."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    key = _agent_build(ctx, make_bundle(tmp_path, GOOD))
    data = ctx.store.read_bytes(key)
    ctx.store.write_bytes(key, bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(StageFault, match="does not match its manifest: file .* has sha256") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert _started(ctx) == ["extract", "checkout"] and not (ctx.role_dir("checkout") / "tree").exists()


def test_a_bundle_whose_size_is_not_its_manifests_is_refused_before_it_is_read(tmp_path, tools):
    """The store's file is not the size the agent_build manifest records: the store's fault,
    infra that no retry mends, decided before the role reads a byte of it."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    key = _agent_build(ctx, make_bundle(tmp_path, GOOD))
    recorded = ctx.store.size(key)
    with open(ctx.store.path(key), "ab") as f:
        f.write(b"x")
    ctx.store.path(key).chmod(0o200)
    with pytest.raises(StageFault, match=f"does not match its manifest: file .* holds {recorded + 1} bytes, not the "
                                         f"{recorded} its record names") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert _started(ctx) == ["extract", "checkout"] and not ctx.output_path("source.json").exists()


def test_no_bundle_at_all_is_an_infra_fault(tmp_path, tools):
    bin_dir, _ = tools
    with pytest.raises(StageFault, match="no source bundle") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=None))
    assert not caught.value.fault.retryable


def test_a_named_bundle_is_read_from_its_copy_and_checked_against_the_digest_the_operator_stated(tmp_path, tools):
    """``hls package --bundle B --sha256 D`` copied B into the instance: the checkout role reads
    the copy once where it lies in the store, never B, checks its size against the plan's before
    any read, streams it into git as it hashes it and checks it against D, as it checks an
    agent_build bundle against its manifest, before any verdict; package.json records D."""
    bin_dir, _ = tools
    bundle = make_bundle(tmp_path, GOOD)
    digest = sha256_file(bundle)[0]
    ctx = _ctx(tmp_path, bin_dir, bundle=bundle, bundle_sha256=digest)
    bundle.unlink()
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "built" and result.bundle_sha256 == digest
    checkout = next(s for s in ctx.launcher.launched if s.name == "checkout")
    copy = ctx.store.path(INSTANCE.source_bundle)
    assert checkout.argv[checkout.argv.index("--bundle") + 1] == str(copy)
    assert checkout.argv[-6:] == ("--size", str(copy.stat().st_size), "--expect", digest, "--max-bytes", str(_CAP))


@pytest.mark.parametrize("change", ["a-byte", "its-size"])
def test_a_copy_changed_after_planning_is_refused_by_the_role(tmp_path, tools, change):
    """The store's copy is not the bytes the plan records, whether it differs in a byte or in its
    size: the checkout role refuses it before git reads it (``DIGEST_MISMATCH_EXIT``), and a
    retry would read the same file, so the fault is infra and not retried, with nothing checked
    out."""
    bin_dir, _ = tools
    bundle = make_bundle(tmp_path, GOOD)
    ctx = _ctx(tmp_path, bin_dir, bundle=bundle)
    copy = ctx.store.path(INSTANCE.source_bundle)
    data = copy.read_bytes()
    copy.write_bytes(data[:-1] + bytes([data[-1] ^ 1]) if change == "a-byte" else data + b"\0")
    said = (f"file {copy} has sha256 {sha256_file(copy)[0]}, not {sha256_file(bundle)[0]}" if change == "a-byte"
            else f"file {copy} holds {len(data) + 1} bytes, not the {len(data)} its record names")
    with pytest.raises(StageFault, match=f"the source bundle is not the sha256 stated for it: {said}") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert _started(ctx) == ["extract", "checkout"]
    assert not (ctx.role_dir("checkout") / "tree").exists()
    assert not ctx.output_path("source.json").exists()


@pytest.mark.parametrize("params, match", [
    ({"bundle": COPY}, "a named bundle needs the sha256 its operator states"),
    ({"bundle_sha256": "ab" * 32}, "bundle_sha256 is the digest of a named bundle"),
    ({"bundle": COPY, "bundle_sha256": "AB" * 32}, "not a bare hex SHA-256"),
    ({"bundle": COPY, "bundle_sha256": "ab" * 32, "build_manifest": ROLLOUT.manifest(AGENT_BUILD_STAGE)},
     "a named bundle or an agent_build manifest's, not both"),
    ({"bundle": "/x.bundle", "bundle_sha256": "ab" * 32}, "valid dictionary or instance of StoredBundle"),
    ({"bundle": {"key": "/x.bundle", "size": 1}, "bundle_sha256": "ab" * 32}, "key"),
    ({"bundle": {"key": "runs/t/x/source.bundle", "size": -1}, "bundle_sha256": "ab" * 32},
     "greater than or equal to 0"),
], ids=["bundle-without-digest", "digest-without-bundle", "malformed-digest", "bundle-and-manifest",
        "an-operator-path", "not-a-store-key", "a-negative-size"])
def test_a_named_bundle_comes_with_its_digest_and_only_it_has_one(params, match):
    with pytest.raises(ValueError, match=match):
        PackageStage.params_model.model_validate(
            {"task": "t", "submission_base": BASE, "constraints": CONSTRAINTS, "pip": PIP.model_dump(),
             "network": POLICY.model_dump(mode="json"), "disk_cap_gib": CAP_GIB, **params})


@pytest.mark.parametrize("images", [
    {},
    {BASE: {"key": "a/b", "entry_sha256": "e" * 64}},
    {BASE: {"key": BASE_KEY}},
], ids=["absent", "bad-key", "no-digest"])
def test_an_images_record_that_does_not_name_the_base_is_refused(tmp_path, tools, images):
    bin_dir, _ = tools
    with pytest.raises(StageFault, match=f"names no image {BASE} with a key and an entry digest") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD), images=images))
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable


def test_an_images_record_that_differs_from_its_manifest_or_is_missing_is_refused(tmp_path, tools):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    ctx.store.write_bytes(ctx.plan.layout.attempt_file("images", 1, "outputs/images.json"), b"{}")
    with pytest.raises(StageFault, match="does not match its manifest") as caught:
        PackageStage().run(ctx)
    assert not caught.value.fault.retryable
    again = tmp_path / "again"
    ctx = _ctx(again, bin_dir, bundle=make_bundle(again, GOOD))
    ctx.inputs = {}
    with pytest.raises(StageFault, match="no images output of this instance's images stage"):
        PackageStage().run(ctx)


def test_the_base_is_pinned_by_the_entry_digest_the_images_stage_recorded(tmp_path, tools):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD),
               images={BASE: {"key": BASE_KEY, "entry_sha256": "e" * 64}})
    with pytest.raises(StageFault, match="has ENTRY.json digest") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert _started(ctx) == []


@pytest.mark.parametrize("files, links, branch, reason", [
    (GOOD, None, "dev", "no commit on main"),
    ({"pyproject.toml": PYPROJECT}, None, "main", "missing required file(s): server.py"),
    ({**GOOD, "pyproject.toml": PYPROJECT.replace('"httpx>=0.27"', '"vllm"')}, None, "main",
     "not permitted by policy: ['vllm']"),
    ({**GOOD, "pyproject.toml": '[project]\nversion = "1"\n'}, None, "main", "[project].name"),
    ({**GOOD, "pyproject.toml": "[project\n"}, None, "main", "pyproject.toml"),
    (GOOD, {"weights": "/etc/passwd"}, "main",
     "the submission is past the package's limits: the tree tracks a symlink, which a submission may not hold: "
     "'weights'"),
])
def test_a_submission_that_breaks_the_contract_is_invalid_before_any_container(tmp_path, tools, files, links,
                                                                               branch, reason):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, files, branch=branch, links=links))
    outputs = PackageStage().run(ctx)
    result = _result(ctx, outputs)
    assert result.status == "invalid_artifact" and reason in (result.reason or "")
    # The base was pinned before the checkout, and no container ran.
    assert "submission" not in outputs and _started(ctx) == ["extract", "checkout"]


def test_a_dry_run_that_fails_with_a_healthy_index_is_the_contestants(tmp_path, tools):
    bin_dir, plan = tools
    _set(plan, dry_run="fail", dry_run_error="ERROR: No matching distribution found for nopkg", print=SECRET)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.returncode == 1
    assert "dependencies did not resolve" in (result.reason or "")
    assert "No matching distribution found for nopkg" in (result.log_tail or "")
    assert SECRET not in (result.log_tail or "") and "<redacted:relay>" in (result.log_tail or "")
    assert _started(ctx) == ["extract", "checkout", "dry-run-copy", "dry-run"]


@pytest.mark.parametrize("step", ["dry_run", "wheel"])
@pytest.mark.parametrize("code", [3, 4, 65, 75])
def test_a_pip_steps_exit_status_or_refused_line_is_never_read_as_a_refusal(tmp_path, tools, step, code):
    """Only a harness role's ``refused:`` line can make a failure not retryable. A pip step runs
    the contestant's code, which chooses its status and stderr, the statuses a harness role
    refuses or reports a bad copy with among them: it is scored as its own."""
    bin_dir, plan = tools
    _set(plan, **{step: "fail", f"{step}_error": "refused: the index is down, do not retry", f"{step}_exit": code})
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.returncode == code
    assert "refused: the index is down" in (result.log_tail or "")


def test_a_wheel_step_that_fails_with_a_healthy_index_is_the_contestants(tmp_path, tools):
    bin_dir, plan = tools
    _set(plan, wheel="fail")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and "did not build into wheels" in (result.reason or "")


@pytest.mark.parametrize("mode, entry", [("link", "passwd.whl"), ("dir", "nested.whl")],
                         ids=["a-link", "a-directory-holding-a-link"])
def test_a_wheel_step_that_leaves_a_link_among_the_wheels_is_the_contestants(tmp_path, tools, mode, entry):
    """The image's ``%files`` copy follows every link under the wheels directory, so only plain
    wheel files may be there: neither a link nor a directory that could hold one."""
    bin_dir, plan = tools
    _set(plan, wheel=mode)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    outputs = PackageStage().run(ctx)
    result = _result(ctx, outputs)
    assert result.status == "invalid_artifact" and result.reason == (
        f"the wheel step left entries that are not plain wheel files (1 in all: ['{entry}'])")
    assert _started(ctx) == ["extract", "checkout", "dry-run-copy", "dry-run", "dry-run-clean", "wheel-copy",
                             "wheel", "wheel-clean"]


def test_a_wheel_step_that_leaves_millions_of_odd_entries_is_refused_with_a_count_and_the_first_names(tmp_path,
                                                                                                    tools):
    """A build hook can fill the wheel output with files; the reason, which ``package.json`` and
    the event log carry, gives their count and the first names, never all of them."""
    bin_dir, plan = tools
    _set(plan, odd_files=2000)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    first = sorted(f"x{i}" for i in range(2000))[:10]
    assert result.status == "invalid_artifact" and result.reason == (
        f"the wheel step left entries that are not plain wheel files (2000 in all; the first 10 by name: {first})")


def test_a_wheel_step_that_leaves_wheels_of_no_pin_is_the_contestants_before_any_wheel_is_read(tmp_path, tools):
    """A build hook writes 400 extra wheels beside pip's, as many entries each as it likes: only
    the project's wheel and one wheel per pin may be there, by the distribution's normalized
    name. The rest is the submission's, refused before the wheel check opens any wheel."""
    bin_dir, plan = tools
    extras = [f"w{k}-0.1-py3-none-any.whl" for k in range(400)]
    _set(plan, extra_wheels=extras)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.reason == (
        "the wheel step left wheels that are neither the project's nor a pinned dependency's (400 in all; the first "
        f"10 by name: {sorted(extras)[:10]})")
    assert _started(ctx)[-1] == "wheel-clean"


@pytest.mark.parametrize("extra", ["httpx-0.28.0-py3-none-any.whl", "HTTPX-0.27.0-1-py3-none-any.whl"],
                         ids=["another-version", "another-spelling"])
def test_a_wheel_step_that_leaves_two_wheels_of_one_distribution_is_the_contestants(tmp_path, tools, extra):
    bin_dir, plan = tools
    _set(plan, extra_wheels=[extra])
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    both = sorted([extra, "httpx-0.27.0-py3-none-any.whl"])
    assert result.status == "invalid_artifact" and result.reason == (
        f"the wheel step left more than one wheel of a distribution (2 in all: {both})")
    assert _started(ctx)[-1] == "wheel-clean"


def test_a_wheel_step_that_leaves_no_wheel_of_the_project_is_the_contestants(tmp_path, tools):
    bin_dir, plan = tools
    _set(plan, no_project_wheel=True)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.reason == "the wheel step left no wheel of the project subm"
    assert _started(ctx)[-1] == "wheel-clean"


@pytest.mark.parametrize("names", [
    ["my_server-1.0-py3-none-any.whl", "foo_bar-1.0-1-cp312-cp312-linux_x86_64.whl", "baz-2-py3-none-any.whl"],
    ["My.Server-1.0-py3-none-any.whl", "Foo.Bar-1.0-py3-none-any.whl"],
], ids=["each-pin", "spelled-otherwise"])
def test_the_wheels_of_the_project_and_its_pins_pass_by_their_normalized_names(tmp_path, names):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in names:
        (wheels / name).write_bytes(b"")
    assert wheels_problem(wheels, "My-Server", ["foo-bar==1.0", "baz @ https://example.org/baz-2-py3-none-any.whl"]) \
        is None


@pytest.mark.parametrize("stray", ["foo.whl", "my_server-py3-none-any.whl", "-1.0-py3-none-any.whl",
                                   "my_server-1.0-a-b-c-d-py3-none-any.whl"],
                         ids=["no-fields", "four-fields", "no-name", "eight-fields"])
def test_a_file_whose_name_is_no_wheels_of_a_pin_is_refused(tmp_path, stray):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in ("my_server-1.0-py3-none-any.whl", stray):
        (wheels / name).write_bytes(b"")
    assert wheels_problem(wheels, "my-server", []) == (
        f"the wheel step left wheels that are neither the project's nor a pinned dependency's (1 in all: ['{stray}'])")


def test_each_name_in_a_reason_is_cut_at_200_characters(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / ("y" * 250)).write_bytes(b"")
    assert wheels_problem(wheels, "subm", []) == (
        f"the wheel step left entries that are not plain wheel files (1 in all: ['{'y' * 200}'])")


def test_a_wheel_step_that_leaves_a_source_distribution_among_the_wheels_is_the_contestants(tmp_path, tools):
    """The offline install would build it in ``%post``, outside the seal: only wheels are let through."""
    bin_dir, plan = tools
    _set(plan, wheel="sdist")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    outputs = PackageStage().run(ctx)
    result = _result(ctx, outputs)
    assert result.status == "invalid_artifact" and "['evil-1.0.tar.gz']" in (result.reason or "")
    assert _started(ctx)[-1] == "wheel-clean"


@pytest.mark.parametrize("errors, mid_stream", [(2, 2), (1, 0), (1, 1)],
                         ids=["twice", "once-before-a-stream", "once-mid-stream"])
def test_a_wheel_step_that_fails_while_the_index_was_out_is_infra(tmp_path, tools, monkeypatch, errors, mid_stream):
    """Every failure the index counts is its upstream's: one mid-stream counts only when the
    step was reading the file promptly, which no stall of the step's can bring about."""
    bin_dir, plan = tools
    _set(plan, wheel="fail")
    monkeypatch.setenv("FAKE_PROXY_ERRORS", str(errors))
    monkeypatch.setenv("FAKE_PROXY_MID_STREAM", str(mid_stream))
    with pytest.raises(StageFault, match=f"package index failed upstream {errors} times, {mid_stream} of them "
                                         "mid-stream") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD)))
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable


@pytest.mark.parametrize("step", ["dry_run", "wheel"])
def test_a_page_the_index_refused_for_its_size_is_named_in_the_invalid_package(tmp_path, tools, monkeypatch, step):
    """A page's size is PyPI's doing, but a dependency the index will not serve fails the package
    all the same, as the submission's: the reason names the page, what was read of it and the
    cap, so the brief can state the limit."""
    bin_dir, plan = tools
    _set(plan, **{step: "fail"})
    refused = {"page": "https://pypi.org/simple/huge/", "read": 64 * 2**20 + 65536, "cap": 64 * 2**20}
    monkeypatch.setenv("FAKE_PROXY_OVERSIZED", json.dumps([refused]))
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact"
    assert ("the package index refused the simple page https://pypi.org/simple/huge/ for its size: more than its "
            "cap of 67108864 bytes (67174400 read before it stopped)") in (result.reason or "")


@pytest.mark.parametrize("step", ["dry_run", "wheel"])
def test_a_build_hook_that_never_ends_is_the_submissions_at_the_shared_limit(tmp_path, tools, monkeypatch, step):
    """A pip step still running when the time both steps may take together is spent, while its
    package index counted no upstream failure, spent that time in the submission's own build
    hooks: the package is invalid, and nothing retries it."""
    bin_dir, plan = tools
    monkeypatch.setattr(package_stage, "PIP_STEPS_SEC", 2)
    _set(plan, **{step: "hang"})
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    started = time.time()
    result = _result(ctx, PackageStage().run(ctx))
    name = step.replace("_", "-")
    assert result.status == "invalid_artifact" and result.reason == (
        f"package step {name} did not end within the 2 s both pip steps may take together while its package index "
        "counted no upstream failure")
    assert time.time() - started < 60 and _started(ctx)[-1] == name


def test_a_dry_run_that_takes_three_hours_leaves_the_wheel_step_one():
    """The two steps share one budget: the dry run may take all of it, and the wheel step gets
    what the dry run left, counted in the commands' own time. A command first seen started
    after it ended spends none of it, and the budget never runs below nothing."""
    budget = package_stage.PipBudget(4 * 3600)
    assert budget.left == 4 * 3600
    budget.spend(1000.0, 1000.0 + 3 * 3600)
    assert budget.left == 3600
    budget.spend(20_000.0, 19_999.5)
    assert budget.left == 3600
    budget.spend(20_000.0, 20_000.0 + 2 * 3600)
    assert budget.left == 0


def test_the_wheel_step_is_stopped_once_the_dry_run_has_spent_most_of_the_budget(tmp_path, tools, monkeypatch):
    """At four seconds for both steps, a dry run of three seconds leaves the wheel step about
    one, counted in the commands' own time: from when the stage saw a step's command start to
    when its payload ended, by the seal's record. A hanging build hook is stopped once its
    command has had what was left, and the package is invalid."""
    bin_dir, plan = tools
    monkeypatch.setattr(package_stage, "PIP_STEPS_SEC", 4)
    firsts = {step: tmp_path / f"{step}-started" for step in ("dry_run", "wheel")}
    _set(plan, dry_run_sleep=3, wheel="hang", dry_run_started=str(firsts["dry_run"]),
         wheel_started=str(firsts["wheel"]))
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    waits: dict[str, tuple[float | None, float]] = {}
    real = ctx.wait_roles

    def wait_roles(*names, command_sec=None):
        outcomes = real(*names, command_sec=command_sec)
        for o in outcomes:
            waits[o.name] = (command_sec, ctx.clock())
        return outcomes

    monkeypatch.setattr(ctx, "wait_roles", wait_roles)
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.reason == (
        "package step wheel did not end within the 4 s both pip steps may take together while its package index "
        "counted no upstream failure")
    (dry_limit, _), (wheel_limit, wheel_stopped) = waits["dry-run"], waits["wheel"]
    dry_files = ctx.seal_files("dry-run")
    dry_ended = None if dry_files is None else payload_ended_at(dry_files)
    dry_began, wheel_began = ctx.command_started_at("dry-run"), ctx.command_started_at("wheel")
    assert dry_ended is not None and dry_began is not None and wheel_began is not None
    dry_ran = dry_ended - dry_began
    assert dry_limit == 4 and 2 < dry_ran < 4
    assert wheel_limit == pytest.approx(4 - dry_ran) and wheel_stopped >= wheel_began + wheel_limit
    # Each start was seen within a pass of the command's own first instant.
    assert dry_began - float(firsts["dry_run"].read_text()) <= SEEN_WITHIN_SEC
    assert wheel_began - float(firsts["wheel"].read_text()) <= SEEN_WITHIN_SEC


def test_a_wheel_step_whose_runtime_starts_late_still_gets_what_the_dry_run_left(tmp_path, tools,
                                                                                 monkeypatch):
    """The pip steps' budget is their commands' own time: a runtime that takes longer to start
    the wheel step's container than the dry run left of the budget costs the step none of it.
    The build hook that hangs is stopped once its command has had what was left, and the
    package is invalid; a stop before the command started would have read as the runtime's
    failure, which is infra."""
    bin_dir, plan = tools
    monkeypatch.setattr(package_stage, "PIP_STEPS_SEC", 3)
    gate, first = tmp_path / "wheel-runtime-starts", tmp_path / "wheel-started"
    _set(plan, dry_run_sleep=1, wheel="hang", wheel_hold=str(gate), wheel_started=str(first))
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    real = ctx.wait_roles
    released: list[float] = []

    def release(at: float) -> None:
        while ctx.clock() < at:
            time.sleep(0.05)
        gate.touch()

    def wait_roles(*names, **limits):
        if "wheel" in ctx.running_roles():
            # Held until the whole budget has passed since the step's start, so a budget counted
            # from there would have run out before its command started.
            released.append(ctx.clock() + package_stage.PIP_STEPS_SEC)
            threading.Thread(target=release, args=(released[0],), daemon=True).start()
        return real(*names, **limits)

    monkeypatch.setattr(ctx, "wait_roles", wait_roles)
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.reason == (
        "package step wheel did not end within the 3 s both pip steps may take together while its package index "
        "counted no upstream failure")
    began = ctx.command_started_at("wheel")
    assert began is not None and released[0] <= began <= float(first.read_text()) + SEEN_WITHIN_SEC


def test_a_wheel_step_the_deadline_cuts_after_the_dry_run_spent_most_of_the_budget_is_infra(tmp_path, tools,
                                                                                            monkeypatch):
    """The dry run spent most of the budget, and the stage's deadline stops the wheel step before
    its command has had what was left: the step did not have all of the budget left to it, which
    is infra, even though what the budget still holds once the cut step is charged would fit
    before the deadline."""
    bin_dir, plan = tools
    monkeypatch.setattr(package_stage, "PIP_STEPS_SEC", 10)
    _set(plan, dry_run_sleep=6, wheel="hang")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    real = ctx.wait_roles
    limits: dict[str, float | None] = {}

    def wait_roles(*names, command_sec=None):
        if "wheel" in ctx.running_roles():
            ctx.deadline = ctx.clock() + 3.5
        outcomes = real(*names, command_sec=command_sec)
        for o in outcomes:
            limits[o.name] = command_sec
        return outcomes

    monkeypatch.setattr(ctx, "wait_roles", wait_roles)
    with pytest.raises(StageFault, match="package step wheel was cut by the stage's deadline before the pip steps' "
                                         "limit") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.evidence["ending"] == "time_limit"
    began, left = ctx.command_started_at("wheel"), limits["wheel"]
    assert began is not None and left is not None and left < 10 / 2 and ctx.deadline < began + left


def test_a_wheel_step_the_deadline_stops_as_its_command_starts_is_infra(tmp_path, tools, monkeypatch):
    """A step the stage's deadline stopped before the stage saw its command start had none of the
    budget counted: it was cut by the deadline, which is infra, even when its start marker came
    in time for the stop to read it, so the step ends ``time_limit`` and not ``runtime_failed``."""
    bin_dir, plan = tools
    gate = tmp_path / "wheel-runtime-starts"
    _set(plan, wheel="hang", wheel_hold=str(gate))
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    real_clock, real_wait = ctx.clock, ctx.wait_roles
    cut: list[float] = []

    def clock() -> float:
        now = real_clock()
        if cut and now >= cut[0] and not gate.exists():
            # The first look at the deadline: the runtime starts the command before it returns.
            gate.touch()
            give_up = real_clock() + 60
            while not ctx.command_started("wheel"):
                assert real_clock() < give_up, "the wheel step's command never started"
                time.sleep(0.05)
        return now

    def wait_roles(*names, **limits):
        if "wheel" in ctx.running_roles():
            ctx.deadline = real_clock() + 1
            cut.append(ctx.deadline)
        return real_wait(*names, **limits)

    ctx.clock = clock
    monkeypatch.setattr(ctx, "wait_roles", wait_roles)
    with pytest.raises(StageFault, match="package step wheel was cut by the stage's deadline before the pip steps' "
                                         "limit") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.evidence["ending"] == "time_limit"
    assert gate.exists() and ctx.command_started_at("wheel") is None


@pytest.mark.parametrize("step", ["dry_run", "wheel"])
def test_a_step_at_the_shared_limit_while_its_interceptor_stopped_reporting_is_infra(tmp_path, tools, monkeypatch,
                                                                                    step):
    """An interceptor that hangs counts nothing, and every request of the step then waits: the
    step reaching its limit is not certainly the submission's when the interceptor's heartbeat
    is more than a minute old. The step is stopped at its limit, so the seal records no ending
    and the heartbeat is judged when the stage saw the step end."""
    bin_dir, plan = tools
    monkeypatch.setattr(package_stage, "PIP_STEPS_SEC", 5)
    monkeypatch.setenv("FAKE_PROXY_HEARTBEAT_AGE", "120")
    _set(plan, **{step: "hang"})
    name = step.replace("_", "-")
    with pytest.raises(StageFault, match=f"package step {name} did not end within the 5 s both pip steps may take "
                                         "together while its package index's interceptor had not reported for "
                                         "1[0-9][0-9] s") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD)))
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable


def test_a_failed_step_while_its_interceptor_stopped_reporting_is_infra(tmp_path, tools, monkeypatch):
    """Requests to a hung interceptor time out, and pip fails on them: not the submission's."""
    bin_dir, plan = tools
    monkeypatch.setenv("FAKE_PROXY_HEARTBEAT_AGE", "120")
    _set(plan, wheel="fail")
    with pytest.raises(StageFault, match="package step wheel failed with status 1 while its package index's "
                                         "interceptor had not reported for") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD)))
    assert caught.value.fault.fault_class == "infra"


def test_a_heartbeat_under_a_minute_old_leaves_the_failure_the_submissions(tmp_path, tools, monkeypatch):
    bin_dir, plan = tools
    monkeypatch.setenv("FAKE_PROXY_HEARTBEAT_AGE", "45")
    _set(plan, wheel="fail")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    assert _result(ctx, PackageStage().run(ctx)).status == "invalid_artifact"


@pytest.mark.parametrize("age", [10, 61], ids=["fresh-at-the-payloads-end", "stale-at-the-payloads-end"])
def test_the_heartbeat_is_judged_when_the_payload_ended_not_when_the_stage_saw_it(tmp_path, tools, monkeypatch, age):
    """The seal stops the interceptor as the payload ends, and writes the payload's ending then.
    A slow controller reports the step 90 s later: a heartbeat 10 s older than the payload's
    end still leaves the failure the submission's, and one 61 s older makes it infra."""
    bin_dir, plan = tools
    monkeypatch.setenv("FAKE_PROXY_HEARTBEAT_AGE", str(age))
    _set(plan, wheel="fail")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    ctx.deadline = time.time() + 600
    late = [0.0]
    ctx.clock = lambda: time.time() + late[0]
    wait_roles = ctx.wait_roles

    def slow_controller(*names, command_sec=None):
        outcomes = wait_roles(*names, command_sec=command_sec)
        if any(o.name == "wheel" for o in outcomes):
            late[0] = 90.0
        return outcomes

    monkeypatch.setattr(ctx, "wait_roles", slow_controller)
    if age < package_stage.HEARTBEAT_STALE_SEC:
        assert _result(ctx, PackageStage().run(ctx)).status == "invalid_artifact"
    else:
        with pytest.raises(StageFault, match="package step wheel failed with status 1 while its package index's "
                                             "interceptor had not reported for 6[1-9] s") as caught:
            PackageStage().run(ctx)
        assert caught.value.fault.fault_class == "infra"
    assert late == [90.0]
    files = ctx.seal_files("wheel")
    assert files is not None and payload_ended_at(files) is not None


@pytest.mark.parametrize("step", ["dry_run", "wheel"])
def test_a_flood_past_the_fetch_cap_fails_as_the_submissions_and_is_named(tmp_path, tools, monkeypatch, step):
    """The interceptor refused the fetches past its cap and counted no upstream failure, so the
    step's failure is the submission's; the reason says the index refused them, so the brief can
    state the cap."""
    bin_dir, plan = tools
    monkeypatch.setenv("FAKE_PROXY_REFUSED", "3")
    _set(plan, **{step: "fail"})
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact"
    assert (result.reason or "").endswith(
        f"; the package index refused 3 fetches from its upstream past the {UPSTREAM_FETCHES_MAX} one step may make")


def test_a_step_at_the_shared_limit_while_the_index_failed_upstream_is_infra(tmp_path, tools, monkeypatch):
    bin_dir, plan = tools
    monkeypatch.setattr(package_stage, "PIP_STEPS_SEC", 2)
    monkeypatch.setenv("FAKE_PROXY_ERRORS", "1")
    _set(plan, dry_run="hang")
    with pytest.raises(StageFault, match="package step dry-run did not end within the 2 s both pip steps may take "
                                         "together while the package index failed upstream 1 times") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD)))
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable


def test_a_step_the_stages_deadline_cut_before_the_shared_limit_is_infra(tmp_path, tools, monkeypatch):
    """A step that did not get all the budget left to it is not certainly too slow on its own
    account."""
    bin_dir, plan = tools
    monkeypatch.setattr(package_stage, "PIP_STEPS_SEC", 600)
    _set(plan, dry_run="hang")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    ctx.deadline = time.time() + 8
    with pytest.raises(StageFault, match="package step dry-run was cut by the stage's deadline before the pip steps' "
                                         "limit") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra"


@pytest.mark.parametrize("cap", [240, 180, CAP_GIB])
def test_both_pip_steps_may_run_four_hours_together_and_the_stage_holds_them(cap):
    """Four hours is 4.4 times 3,247 s, what one pip step doing the work of both takes on a slow
    index; the stage's time holds them with the hour the checkout of the bundle is given
    (``hls.package.checkout.CHECKOUT_SEC``), whose git runs to the stage's deadline, and two hours
    for the rest of the package: the same at any cap."""
    assert package_stage.PIP_STEPS_SEC == 4 * 3600 >= max(4 * 3247, 30 * 60)
    assert package_stage.MIN_TIME_SEC == package_stage.PIP_STEPS_SEC + CHECKOUT_SEC + package_stage.REST_SEC == (
        4 * 3600 + 3600 + 2 * 3600)
    assert package_shape(cap).time_sec == package_stage.MIN_TIME_SEC
    params = PackageStage.params_model.model_validate(
        {"task": "t", "submission_base": BASE, "constraints": CONSTRAINTS, "pip": PIP.model_dump(), "network": POLICY.model_dump(mode="json"),
         "disk_cap_gib": cap})
    shape = package_shape(cap).model_copy(update={"cpus": 4})
    PackageStage.check(shape, params, runner_memory_gib=4)
    with pytest.raises(ValueError, match=re.escape(
            "package needs at least 25200 s of time, the 14400 s the pip steps may take together, the checkout's "
            "3600 s and the rest of the package's 7200 s; the stage declares 25199")):
        PackageStage.check(shape.model_copy(update={"time_sec": package_stage.MIN_TIME_SEC - 1}), params,
                           runner_memory_gib=4)


def test_a_dry_run_that_fills_the_disk_once_started_still_fails_as_the_submissions(tmp_path, tools, monkeypatch):
    """The step's seal makes its ending and its interceptor's status at full size before the
    payload starts and only writes over them afterwards, so a build hook that fills the node's
    disk, where they are, leaves both whole: the step's own failure is the submission's, never
    a seal that recorded nothing. The real seal script runs here, around the fake interceptor on
    free ports."""
    bin_dir, plan = tools
    with socket.socket() as transparent, socket.socket() as index:
        transparent.bind(("127.0.0.1", 0))
        index.bind(("127.0.0.1", 0))
        monkeypatch.setattr(seal_mod, "EGRESS_TRANSPARENT_PORT", transparent.getsockname()[1])
        monkeypatch.setattr(seal_mod, "EGRESS_INDEX_PORT", index.getsockname()[1])
    (tmp_path / "pasta.plan").write_text("run\n")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    ctx.plan = ctx.plan.model_copy(update={"toolchain": ctx.plan.toolchain.model_copy(
        update={"python": bin_dir / "python"})})
    ctx.launcher.use_plan(ctx.plan)
    seal = ctx.role_dir("dry-run") / "seal"
    _set(plan, dry_run="fail", seal_full=str(seal))
    try:
        result = _result(ctx, PackageStage().run(ctx))
    finally:
        if seal.exists():
            seal.chmod(0o700)
    assert result.status == "invalid_artifact" and (result.reason or "").startswith(
        "the submission's dependencies did not resolve against the submission base")
    assert [line.strip() for line in (tmp_path / "pasta.plan").read_text().splitlines()] == []


def test_a_failed_step_while_the_index_counted_nothing_is_the_submissions(tmp_path, tools, monkeypatch):
    bin_dir, plan = tools
    _set(plan, wheel="fail")
    monkeypatch.setenv("FAKE_PROXY_ERRORS", "0")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    assert _result(ctx, PackageStage().run(ctx)).status == "invalid_artifact"


def test_a_runtime_failure_before_the_container_starts_is_infra(tmp_path, tools, monkeypatch):
    bin_dir, _ = tools
    monkeypatch.setenv("FAKE_APPTAINER_FAIL", "1")
    with pytest.raises(StageFault, match="dry-run: runtime failed") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD)))
    assert caught.value.fault.fault_class == "infra"


@pytest.mark.parametrize("fault_class, retryable", [("contestant", False), ("infra", True)])
def test_a_fault_the_roles_decided_for_a_pip_step_is_raised_as_it_is(tmp_path, tools, monkeypatch, fault_class,
                                                                     retryable):
    """A pip step's own fault, such as one killed for memory, is the roles' to decide
    (``stage_fault``): a contestant fault when its own limit was certainly reached, infra
    otherwise. The stage raises the decision as it is and nothing of its own."""
    bin_dir, _ = tools
    decided = FaultRecord(fault_class=fault_class, reason="contestant role wheel ran out of memory",
                          retryable=retryable, evidence={})
    real = package_stage.stage_fault
    monkeypatch.setattr(package_stage, "stage_fault",
                        lambda outcomes: decided if [o.name for o in outcomes] == ["wheel"] else real(outcomes))
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    with pytest.raises(StageFault) as caught:
        PackageStage().run(ctx)
    assert caught.value.fault is decided
    assert not ctx.store.exists(ctx.output_key("package.json"))


def test_an_offline_build_that_fails_before_the_install_is_infra(tmp_path, tools):
    bin_dir, plan = tools
    _set(plan, build="fail-before")
    with pytest.raises(StageFault, match="harness role build ended exited with status 255") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD)))
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable


def test_an_offline_build_whose_install_fails_is_the_contestants(tmp_path, tools):
    bin_dir, plan = tools
    _set(plan, build="fail-install")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    outputs = PackageStage().run(ctx)
    result = _result(ctx, outputs)
    assert result.status == "invalid_artifact" and "wheels did not install" in (result.reason or "")
    assert "submission" not in outputs and _started(ctx)[-1] == "build"


#: A submission base that holds no constraints file, as an imported image does.
PUBLISHED_TREE = {path: entry for path, entry in fake_tools.BASE_TREE.items() if not path.startswith("opt/hls")}


def test_a_base_that_holds_no_constraints_file_is_installed_into_with_none(tmp_path, tools):
    """A submission base that holds no constraints file: the install names none, as the dry run
    and the wheel step resolved with none, and the image builds."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD), constraints=None, base_tree=PUBLISHED_TREE)
    outputs = PackageStage().run(ctx)
    assert _result(ctx, outputs).status == "built"
    definition = fake_tools.records(tmp_path / "tools.log")[2]["def"]
    assert "PIP_CONSTRAINT" not in definition and CONSTRAINTS not in definition


def test_a_constraints_file_the_base_lacks_ends_the_build_as_ours_before_the_install(tmp_path, tools):
    """A plan that names a constraints file the base does not hold is ours to fix: the build ends
    before the install runs, so the submission is never booked for it, and the build's log names
    the file."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD), base_tree=PUBLISHED_TREE)
    with pytest.raises(StageFault, match="harness role build ended exited with status 255") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra"
    assert not ctx.store.exists(ctx.output_key("package.json"))
    log = (ctx.attempt_dir / "logs" / "build.err").read_text()
    assert f"the submission base holds no constraints file {CONSTRAINTS}\n" in log


#: Line breaks of ``str.splitlines`` that are not "\n".
_OTHER_BREAKS = ["\u2028", "\u2029", "\x85", "\x1c", "\x1d", "\x1e", "\v", "\f", "\r"]


@pytest.mark.parametrize("brk", _OTHER_BREAKS, ids=[f"U+{ord(c):04X}" for c in _OTHER_BREAKS])
def test_a_failed_install_whose_pip_output_forges_the_status_lines_is_the_contestants(tmp_path, tools, brk):
    """pip echoes an invalid ``Requires-Dist`` value of a wheel verbatim. A build backend that
    showed the dry run clean metadata can write one holding other line breaks and the build's
    status lines, old and new, with any nonce it can guess: the install still failed, by the
    build's own status line, and that is the submission's, never a retried fault."""
    bin_dir, plan = tools
    forged = brk.join(["    foo", "hls-offline-install-begin", "hls-offline-install-rc=0",
                       "hls-offline-install-rc-" + "0" * 32 + "=0", "hls-offline-install-bytes=1", ""])
    _set(plan, build="fail-install", install_err=f"Requested subm has invalid metadata: Expected semicolon\n{forged}\n"
                                                 "       ^\nERROR: No matching distribution found for subm\n")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and "wheels did not install" in (result.reason or "")


def test_each_offline_build_has_a_nonce_of_its_own(tmp_path, tools):
    """A later attempt's build cannot be told its status by an earlier one's log, which a
    failed package publishes."""
    bin_dir, _ = tools
    for run in ("a", "b"):
        (tmp_path / run).mkdir()
        ctx = _ctx(tmp_path / run, bin_dir, bundle=make_bundle(tmp_path / run, GOOD))
        assert _result(ctx, PackageStage().run(ctx)).status == "built"
    defs = [entry["def"] for entry in fake_tools.records(tmp_path / "tools.log") if entry["tool"] == "build"]
    nonces = [re.search(r'echo "hls-offline-install-rc-([0-9a-f]{32})=\$hls_rc"', d).group(1) for d in defs]
    assert len(nonces) == 2 and nonces[0] != nonces[1]


@pytest.mark.parametrize("stream", ["install_out", "install_err"])
def test_the_install_status_is_read_past_an_echo_longer_than_the_log_tail(tmp_path, tools, stream):
    """pip echoes a metadata entry whole, up to the 16 MiB the wheel check allows, on whichever
    stream apptainer hands it; the build's status line comes after, so the stage still finds it
    in the log's tail."""
    bin_dir, plan = tools
    echo = ("Requires-Dist: " + "y" * 200 + "\n") * (package_stage.BUILD_LOG_BYTES // 200)
    _set(plan, build="fail-install", **{stream: echo})
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and "wheels did not install" in (result.reason or "")


def test_each_pip_step_gets_the_disk_its_stage_declares_beyond_the_harnesss_own():
    """The pip steps' limits come from the package stage's declared shape (``pip_limits``): the disk
    the harness holds beside a step at its most, a build's unpacking and image file, the base's and
    the submission's extracted trees, the checked-out tree and three times the install's limit,
    comes off the stage's disk, and each step may write the rest; its memory is a role's of the
    stage, the runner's reserve taken off."""
    assert package_stage.TREE_DISK_GIB == -(-TREE_MAX_BYTES // 2**30)
    assert package_stage.HARNESS_DISK_GIB == (BUILD_DISK_GIB + 2 * KEPT_DISK_GIB + package_stage.TREE_DISK_GIB
                                              + 3 * INSTALL_MAX_GIB)
    assert package_stage.HARNESS_DISK_GIB == package_stage.BUILD_STEPS_DISK_GIB - package_stage.PIP_DISK_FLOOR_GIB
    # The least a stage may declare leaves a step room for its copy of the tree at the tree's limit
    # and wheels that declare the most the install may add.
    assert package_stage.PIP_DISK_FLOOR_GIB == package_stage.TREE_DISK_GIB + INSTALL_MAX_GIB
    assert WORK_DISK_GIB == package_stage.TREE_DISK_GIB + package_stage.PIP_DISK_FLOOR_GIB
    assert pip_limits(SHAPE, 4) == PipLimits(disk_gib=package_stage.PIP_DISK_FLOOR_GIB, memory_gib=CHECKOUT_MEMORY_GIB)
    # A stage that declares more gives every gibibyte of it to the steps.
    bigger = SHAPE.model_copy(update={"disk_gib": SHAPE.disk_gib + 30, "memory_gib": 40})
    assert pip_limits(bigger, 4) == PipLimits(disk_gib=package_stage.PIP_DISK_FLOOR_GIB + 30, memory_gib=36)
    # A package planned at a larger cap declares its checkout's disk, which is the steps' once the
    # clone has gone.
    planned = package_shape(180)
    assert pip_limits(planned, 4).disk_gib == planned.disk_gib - package_stage.HARNESS_DISK_GIB


def test_a_stage_that_cannot_give_the_stated_pip_limits_is_refused_when_planned():
    """The plan states each pip step's disk and memory to the stage and to the build's brief; a
    stage whose declared shape (after the site's resources table) gives less is refused, so a step
    is never held to less than the brief said."""
    params = PackageStage.params_model.model_validate(
        {"task": "t", "submission_base": BASE, "constraints": CONSTRAINTS, "pip": PIP.model_dump(), "network": POLICY.model_dump(mode="json"),
         "disk_cap_gib": CAP_GIB})
    PackageStage.check(SHAPE, params, runner_memory_gib=4)
    # More room than stated is the stage's to keep: the steps are held to what the brief says.
    PackageStage.check(SHAPE.model_copy(update={"disk_gib": SHAPE.disk_gib + 8}), params, runner_memory_gib=4)
    for more in ({"disk_gib": PIP.disk_gib + 1}, {"memory_gib": PIP.memory_gib + 1}):
        stated = params.model_copy(update={"pip": PIP.model_copy(update=more)})
        with pytest.raises(ValueError, match="each pip step"):
            PackageStage.check(SHAPE, stated, runner_memory_gib=4)


def test_a_stage_whose_roles_cannot_hold_the_checkout_is_refused_when_planned():
    """The checkout runs as one of the stage's roles, which get the stage's memory less the
    runner's: under the checkout's measured peak and a quarter (``CHECKOUT_MEMORY_GIB``) the stage
    is refused before it runs."""
    small = SHAPE.model_copy(update={"memory_gib": CHECKOUT_MEMORY_GIB + 3})
    params = PackageStage.params_model.model_validate(
        {"task": "t", "submission_base": BASE, "constraints": CONSTRAINTS, "pip": pip_limits(small, 4).model_dump(),
         "network": POLICY.model_dump(mode="json"), "disk_cap_gib": CAP_GIB})
    with pytest.raises(ValueError, match=f"package's checkout needs {CHECKOUT_MEMORY_GIB} GiB of memory"):
        PackageStage.check(small, params, runner_memory_gib=4)


def test_package_takes_no_gpu_room_to_build_and_memory_beside_the_runners():
    params = PackageStage.params_model.model_validate(
        {"task": "t", "submission_base": BASE, "constraints": CONSTRAINTS, "pip": PIP.model_dump(), "network": POLICY.model_dump(mode="json"),
         "disk_cap_gib": CAP_GIB})
    # A build and its image file, two trees, the work, and three times the installed-size limit.
    assert package_stage.min_disk_gib(CAP_GIB) == package_stage.BUILD_STEPS_DISK_GIB == (
        BUILD_DISK_GIB + 2 * KEPT_DISK_GIB + WORK_DISK_GIB + 3 * INSTALL_MAX_GIB)
    PackageStage.check(SHAPE, params, runner_memory_gib=4)
    with pytest.raises(ValueError, match="no GPU"):
        PackageStage.check(SHAPE.model_copy(update={"gpus": 1, "gpu_kind": "H200"}), params, runner_memory_gib=4)
    with pytest.raises(ValueError, match="disk"):
        PackageStage.check(SHAPE.model_copy(update={"disk_gib": SHAPE.disk_gib - 1}), params, runner_memory_gib=4)
    with pytest.raises(ValueError, match="leaves its roles none"):
        PackageStage.check(SHAPE, params, runner_memory_gib=SHAPE.memory_gib)
    with pytest.raises(ValueError):
        PackageStage.params_model.model_validate({"task": "t", "submission_base": "a/b", "network": {},
                                                  "disk_cap_gib": CAP_GIB})
    with pytest.raises(ValueError, match="must be an absolute path in the image"):
        PackageStage.params_model.model_validate({"task": "t", "submission_base": BASE, "pip": PIP.model_dump(),
                                                  "constraints": "opt/hls/constraints.txt",
                                                  "network": POLICY.model_dump(mode="json"), "disk_cap_gib": CAP_GIB})


def _outcome(ending: str, code: int | None, oom: int = 0) -> RoleOutcome:
    return RoleOutcome("wheel", "contestant", code, ending, oom)  # type: ignore[arg-type]


#: When the steps of the verdict table ended.
_ENDED = 1_000_000.0


def _errors(total: int, mid_stream: int = 0, *, age: float = 5.0) -> IndexStatus:
    """The index's count, its interceptor's heartbeat *age* seconds older than the step's end."""
    return IndexStatus(total=total, mid_stream=mid_stream, oversized=(), policy_refused=(), fetches_refused=0,
                       heartbeat=_ENDED - age)


@pytest.mark.parametrize("outcome, errors, whole_limit, verdict", [
    (_outcome("exited", 0), _errors(0), True, "ok"),
    (_outcome("exited", 0), _errors(3), True, "ok"),
    (_outcome("exited", 0), _errors(0, age=600), True, "ok"),
    (_outcome("exited", 1), _errors(0), True, "invalid"),
    (_outcome("exited", 1), _errors(0, age=60), True, "invalid"),
    (_outcome("exited", 1), _errors(0, age=61), True, "infra"),
    (_outcome("exited", 1), _errors(1, 1), True, "infra"),
    (_outcome("exited", 1), _errors(1), True, "infra"),
    (_outcome("exited", 1), _errors(2, 2), True, "infra"),
    (_outcome("exited", 1), None, True, "infra"),
    (_outcome("exited", 137, oom=1), _errors(0), True, "infra"),
    (_outcome("time_limit", None), _errors(0), True, "invalid"),
    (_outcome("time_limit", None), _errors(0, age=60), True, "invalid"),
    (_outcome("time_limit", None), _errors(0, age=61), True, "infra"),
    (_outcome("time_limit", None), _errors(1, 1), True, "infra"),
    (_outcome("time_limit", None), None, True, "infra"),
    (_outcome("time_limit", None), _errors(0), False, "infra"),
    (_outcome("stopped", None), _errors(0), True, "infra"),
    (_outcome("scheduler_ended", 1), _errors(0), True, "infra"),
], ids=["ok", "ok-despite-errors", "ok-despite-a-stale-heartbeat", "failed", "failed-heartbeat-a-minute-old",
        "failed-heartbeat-stale", "failed-mid-stream-error", "failed-error", "failed-errors", "failed-no-count", "oom",
        "own-limit", "own-limit-heartbeat-a-minute-old", "own-limit-heartbeat-stale", "own-limit-error",
        "own-limit-no-count", "deadline-first", "stopped", "scheduler-ended"])
def test_whose_failure_a_contestant_build_step_is(outcome, errors, whole_limit, verdict):
    """A step that failed, or reached the pip steps' limit having had the whole of it, is the
    submission's when its index counted no upstream failure and its interceptor reported within
    the last minute; every other ending but its own exit is ours."""
    assert contestant_verdict(outcome, upstream=errors, whole_limit=whole_limit, ended_at=_ENDED) == verdict


@pytest.mark.parametrize("total, mid_stream, outage", [(0, 0, False), (1, 1, True), (1, 0, True), (2, 2, True),
                                                       (2, 1, True)])
def test_the_index_is_out_after_any_failure_its_upstream_counted(total, mid_stream, outage):
    assert index_outage(_errors(total, mid_stream)) is outage


def test_a_copy_that_is_not_the_built_image_is_not_published(tmp_path, tools, monkeypatch):
    """The publish role reports the digest of the bytes it wrote; one that is not the digest the
    image was pinned by is a copy gone wrong, which a retry copies again."""
    bin_dir, _ = tools
    cut = tmp_path / "cut.py"
    cut.write_text("import sys\nfrom pathlib import Path\nfrom hls.store.manifest import report_outputs\n"
                   "a = sys.argv\nout, report = Path(a[a.index('--out') + 1]), Path(a[a.index('--outputs-out') + 1])\n"
                   "out.write_bytes(b'x')\nreport_outputs(report, out.parent, out)\n")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    real = ctx.start_role

    def start_role(name, argv, shape, **kw):
        if name == "publish":
            argv = [sys.executable, str(cut), *argv]
        return real(name, argv, shape, **kw)

    monkeypatch.setattr(ctx, "start_role", start_role)
    with pytest.raises(StageFault, match="has sha256 .*, not the .* it was pinned by") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert hashlib.sha256(b"x").hexdigest() not in ctx.image_digests.values()


def test_a_tree_past_the_checkouts_limits_is_invalid_before_any_copy_or_container(tmp_path, tools):
    """A tree nested deeper than Python can recurse never reaches the runner's stack: git's
    listing refuses it in the checkout role, and the stage publishes the refusal."""
    bin_dir, _ = tools
    deep = "/".join(["d"] * (sys.getrecursionlimit() + 100)) + "/f"
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, {**GOOD, deep: "x"}))
    outputs = PackageStage().run(ctx)
    result = _result(ctx, outputs)
    assert result.status == "invalid_artifact" and "past the package's limits" in (result.reason or "")
    assert "a path longer than the 1024 bytes allowed" in (result.reason or "") and len(result.commit or "") == 40
    assert _started(ctx) == ["extract", "checkout"] and not (ctx.role_dir("checkout") / "tree").exists()


@pytest.mark.parametrize("refused", sorted(REFUSED))
def test_a_bundle_the_checkout_refuses_on_its_content_is_invalid_before_any_copy_or_container(tmp_path, tools,
                                                                                              refused):
    """The commit point writes its bundle once and reads nothing back, so the bundle agent_build
    publishes may be one git does not read back, git having bundled from a store whose objects
    lied, or one whose tree names a tree as a file. The checkout proves every byte it gives git
    has the digest the manifest records, so git's verdict is the submission's: an invalid package,
    the verdict in its reason, published before any copy or container, and never a fault."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    _agent_build(ctx, make_refused_bundle(tmp_path, refused))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and len(result.commit or "") == 40
    assert re.fullmatch(f"the submission is past the package's limits: {REFUSED[refused]}", result.reason or ""), \
        result.reason
    assert _started(ctx) == ["extract", "checkout"] and not (ctx.role_dir("checkout") / "tree").exists()
    assert ctx.contestant_fault() is None


def test_a_tree_holding_a_path_git_refuses_is_invalid_before_any_copy_or_container(tmp_path, tools):
    """git's plumbing can commit a .git directory, which git's checkout refuses to write; the
    listing finds it, so the refusal is the contestant's rather than a failed role."""
    bin_dir, _ = tools
    bundle = make_tree_bundle(tmp_path, {"pyproject.toml": False, "server.py": False,
                                         ".git/hooks/post-checkout": False})
    ctx = _ctx(tmp_path, bin_dir, bundle=bundle)
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.reason == (
        "the submission is past the package's limits: the tree tracks a path git refuses to check out: "
        "'.git/hooks/post-checkout'")
    assert _started(ctx) == ["extract", "checkout"] and not (ctx.role_dir("checkout") / "tree").exists()


#: The most bytes a bundle may hold: the build's writable disk cap.
_CAP = CAP_GIB * 2**30
_OVER_LIMIT = (f"the submission is past the package's limits: the source bundle holds {_CAP + 1} bytes, "
               f"more than the {_CAP} allowed")


def _past_the_limit(path: Path) -> Path:
    """A file one byte past the build's writable disk cap at *path*, all zeros and sparse."""
    with open(path, "wb") as f:
        f.truncate(_CAP + 1)
    return path


@pytest.mark.parametrize("cap, disk", [(240, 288), (180, 228), (CAP_GIB, 224)])
def test_the_stage_declares_the_disk_of_its_checkout_at_the_builds_cap(cap, disk):
    """The checkout runs once the submission base is extracted and before anything else: the base's
    tree, and the checkout's disk at the build's writable disk cap, its clone's pack, as large as
    the bundle since the bundle is not compressed, the tree and git's own files, with no copy of
    the bundle (``hls.package.checkout.checkout_disk_gib``). The stage needs that or the build
    steps' disk (``BUILD_STEPS_DISK_GIB``), whichever is more."""
    assert package_stage.min_disk_gib(cap) == max(package_stage.BUILD_STEPS_DISK_GIB,
                                                  KEPT_DISK_GIB + cap + 4 + 4) == disk
    assert package_shape(cap).disk_gib == disk
    shape = package_shape(cap).model_copy(update={"cpus": 4})
    params = PackageStage.params_model.model_validate(
        {"task": "t", "submission_base": BASE, "constraints": CONSTRAINTS, "pip": pip_limits(shape, 4).model_dump(),
         "network": POLICY.model_dump(mode="json"), "disk_cap_gib": cap})
    PackageStage.check(shape, params, runner_memory_gib=4)
    with pytest.raises(ValueError, match=f"package needs at least {disk} GiB of node-local disk"):
        PackageStage.check(shape.model_copy(update={"disk_gib": disk - 1}), params, runner_memory_gib=4)


def test_the_checkout_holds_the_bundle_to_the_builds_cap(tmp_path, tools):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    assert _result(ctx, PackageStage().run(ctx)).status == "built"
    [checkout] = [s for s in ctx.launcher.launched if s.name == "checkout"]
    assert checkout.argv[checkout.argv.index("--max-bytes") + 1] == str(_CAP)


def test_an_agent_build_bundle_past_the_byte_limit_is_invalid_by_its_size_alone_unread(tmp_path, tools):
    """Its manifest records its size, so a store file of that size is the contestant's bundle,
    and past the limit it is refused by its size alone: the role reads none of it."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    key = ctx.plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/source.bundle")
    ctx.store.write_bytes(key, b"")
    _past_the_limit(ctx.store.path(key)).chmod(0o200)
    ctx.inputs[ctx.plan.layout.manifest(AGENT_BUILD_STAGE)] = _manifest(
        ctx, AGENT_BUILD_STAGE, {SOURCE_BUNDLE_OUTPUT: output_ref(ctx.store, key, "0" * 64), **_served(ctx)})
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.bundle_sha256 is None and result.commit is None
    assert result.reason == _OVER_LIMIT
    assert _started(ctx) == ["extract", "checkout"] and not (ctx.role_dir("checkout") / "source.bundle").exists()


@pytest.mark.parametrize("over, status", [(0, "built"), (1, "invalid_artifact")], ids=["at-the-limit", "one-byte-past"])
def test_a_pyproject_past_its_size_is_invalid_before_this_process_parses_it(tmp_path, tools, over, status):
    bin_dir, _ = tools
    line = "# padding\n"
    pad = PYPROJECT_MAX_BYTES + over - len(PYPROJECT.encode()) - len(line)
    pyproject = PYPROJECT + "#" + "x" * (pad - 2) + "\n" + line
    assert len(pyproject.encode()) == PYPROJECT_MAX_BYTES + over
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, {**GOOD, "pyproject.toml": pyproject}))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == status
    if over:
        assert result.reason == (f"pyproject.toml holds {PYPROJECT_MAX_BYTES + 1} bytes, more than the "
                                 f"{PYPROJECT_MAX_BYTES} allowed")
        assert _started(ctx) == ["extract", "checkout"]


@pytest.mark.parametrize("files, links, reason", [
    ({"server.py": SERVER}, {"pyproject.toml": "/etc/hostname"},
     "the submission is past the package's limits: the tree tracks a symlink, which a submission may not hold: "
     "'pyproject.toml'"),
    ({"server.py": SERVER, "conf/pyproject.toml": PYPROJECT}, {"pyproject.toml": "conf/pyproject.toml"},
     "the submission is past the package's limits: the tree tracks a symlink, which a submission may not hold: "
     "'pyproject.toml'"),
    ({"server.py": SERVER, "pyproject.toml/x": ""}, None, "pyproject.toml is not a regular file"),
], ids=["a-symlink-to-a-host-file", "a-symlink-in-the-tree", "a-directory"])
def test_a_pyproject_that_is_not_a_regular_file_is_invalid_and_never_followed(tmp_path, tools, files, links,
                                                                               reason):
    """This process reads pyproject.toml, so anything there but a regular file is the
    submission's, refused before anything opens it: a link, to the host or within the tree, is
    refused with the tree, never followed."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, files, links=links))
    with recording_reads() as reads:
        result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.reason == reason
    assert _started(ctx) == ["extract", "checkout"]
    assert not [r for r in reads if "pyproject.toml" in r]


def test_a_pyproject_that_is_not_utf8_is_the_submissions_not_a_crash(tmp_path, tools):
    """The prescan reads pyproject.toml as text; one that is not UTF-8 is the submission's,
    never an error of this process, which would be retried."""
    bin_dir, _ = tools
    bundle = make_bundle(tmp_path, {"server.py": SERVER, "pyproject.toml": PYPROJECT})
    repo = tmp_path / "bad"
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    # The bundle has no HEAD, so the clone names its branch; with no configured default it
    # would otherwise make no local main.
    subprocess.run(["git", "clone", "-q", "-b", "main", str(bundle), str(repo)], check=True, capture_output=True, env=env)
    (repo / "pyproject.toml").write_bytes(PYPROJECT.encode() + b"# \xff\n")
    for argv in (["add", "-A"], ["-c", "user.name=a", "-c", "user.email=a@x", "commit", "-q", "-m", "x"],
                 ["bundle", "create", str(tmp_path / "bad.bundle"), "main"]):
        subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True, env=env)
    ctx = _ctx(tmp_path, bin_dir, bundle=tmp_path / "bad.bundle")
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.reason == "pyproject.toml is not UTF-8"
    assert _started(ctx) == ["extract", "checkout"]


@pytest.mark.parametrize("value, words", [
    ("[" * 1000 + "]" * 1000, "RecursionError: maximum recursion depth exceeded"),
    ("1" * 5000, "ValueError: Exceeds the limit (4300 digits)"),
], ids=["nested-1000-deep", "an-integer-of-5000-digits"])
def test_a_pyproject_the_parser_cannot_hold_is_the_submissions_not_a_crash(tmp_path, tools, value, words):
    """TOML nested deeper than the parser recurses, or holding an integer Python will not
    convert, is the submission's like any other pyproject that does not parse: refused before any
    container, never an error of this process, which would be retried."""
    bin_dir, _ = tools
    pyproject = f"{PYPROJECT}x = {value}\n"
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, {**GOOD, "pyproject.toml": pyproject}))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact"
    assert (result.reason or "").startswith(f"pyproject.toml cannot be parsed: {words}")
    assert _started(ctx) == ["extract", "checkout"]


@pytest.mark.parametrize("make, reason", [
    (lambda p: os.mkfifo(p), "pyproject.toml is not a regular file"),
    (lambda p: p.mkdir(), "pyproject.toml is not a regular file"),
    (lambda p: os.symlink("/etc/hostname", p), "pyproject.toml is a symlink"),
    (lambda p: p.write_bytes(b"x" * (PYPROJECT_MAX_BYTES + 1)),
     f"pyproject.toml holds {PYPROJECT_MAX_BYTES + 1} bytes, more than the {PYPROJECT_MAX_BYTES} allowed"),
    (lambda p: p.write_bytes(b"x" * PYPROJECT_MAX_BYTES), None),
    (lambda p: None, None),
], ids=["a-named-pipe", "a-directory", "a-symlink", "past-its-size", "at-its-size", "missing"])
def test_what_is_at_pyproject_is_judged_from_its_metadata_alone(tmp_path, make, reason):
    """A named pipe cannot come from a git tree, but the check judges any path by ``lstat``
    alone, so nothing waits on it or follows it; a missing file is the contract's to report."""
    make(tmp_path / "pyproject.toml")
    with recording_reads() as reads:
        assert pyproject_problem(tmp_path) == reason
    assert reads == []


@pytest.mark.parametrize("code, submissions", ERRNO_SIDES, ids=ERRNO_IDS)
def test_a_pyproject_lstat_error_is_the_submissions_only_for_its_permissions(tmp_path, monkeypatch, code,
                                                                             submissions):
    """EACCES and EPERM judging pyproject.toml come of permissions in the submission's tree;
    any other error is the node's, raised for the stage to report as infra."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PYPROJECT)
    real = os.lstat

    def lstat(path, *args, **kwargs):
        if Path(path) == pyproject:
            raise OSError(code, os.strerror(code), str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(package_stage.os, "lstat", lstat)
    if submissions:
        assert pyproject_problem(tmp_path) == (f"pyproject.toml cannot be read: [Errno {code}] "
                                               f"{os.strerror(code)}: '{pyproject}'")
    else:
        with pytest.raises(OSError, match=rf"\[Errno {code}\]"):
            pyproject_problem(tmp_path)


def test_a_pyproject_in_a_directory_this_process_cannot_search_is_the_submissions(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "pyproject.toml").write_text(PYPROJECT)
    tree.chmod(0o600)
    try:
        reason = pyproject_problem(tree) or ""
        assert reason.startswith("pyproject.toml cannot be read: [Errno 13] Permission denied")
    finally:
        tree.chmod(0o755)


@pytest.mark.parametrize("code, submissions", ERRNO_SIDES, ids=ERRNO_IDS)
def test_a_pyproject_read_error_in_the_stage_is_the_submissions_only_for_its_permissions(tmp_path, tools,
                                                                                         monkeypatch, code,
                                                                                         submissions):
    """The stage publishes a pyproject.toml its permissions keep out as the submission's; any
    other error reading it is a retryable fault of the node, with its errno in the reason."""
    bin_dir, _ = tools
    real = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.name == "pyproject.toml":
            raise OSError(code, os.strerror(code), str(self))
        return real(self, *args, **kwargs)

    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    monkeypatch.setattr(Path, "read_text", read_text)
    if submissions:
        result = _result(ctx, PackageStage().run(ctx))
        assert result.status == "invalid_artifact"
        assert (result.reason or "").startswith(f"pyproject.toml cannot be read: [Errno {code}]")
    else:
        node = rf"^the tree's pyproject.toml cannot be read: \[Errno {code}\]"
        with pytest.raises(StageFault, match=node) as caught:
            PackageStage().run(ctx)
        assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
        assert caught.value.fault.evidence["errno"] == str(code)
    assert _started(ctx) == ["extract", "checkout"]


_UNREADABLE = "the dry run's report.json cannot be read: "
_UNUSABLE = "the dry run's report.json cannot be used: "
_TOO_DEEP = _UNUSABLE + f"JsonDepthError: nested deeper than {REPORT_MAX_DEPTH} levels"


@pytest.mark.parametrize("change, reason", [
    ({"report_pad": REPORT_MAX_BYTES}, f"holds more than the {REPORT_MAX_BYTES} bytes allowed"),
    ({"report_link": "/etc/hostname"}, "the dry run's report.json is a symlink"),
    ({"report_fifo": True}, "the dry run's report.json is not a regular file"),
    ({"report_dir": True}, "the dry run's report.json is not a regular file"),
    ({"report_locked": True}, _UNREADABLE + "[Errno 13] Permission denied"),
    ({"report_dir_locked": True}, _UNREADABLE + "[Errno 13] Permission denied"),
    ({"report_missing": True}, "the dry run wrote no report.json"),
    ({"report_raw": "{"}, _UNUSABLE + "JSONDecodeError: Expecting property name enclosed in double quotes"),
    ({"report_raw": '{"install": 3}'}, _UNUSABLE + "ValueError: not a pip installation report: "),
    ({"report_raw": "[" * REPORT_MAX_DEPTH + "]" * REPORT_MAX_DEPTH},
     _UNUSABLE + "ValueError: not a pip installation report: "),
    ({"report_raw": "[" * (REPORT_MAX_DEPTH + 1) + "]" * (REPORT_MAX_DEPTH + 1)}, _TOO_DEEP),
    ({"report_raw": "[" * 100_000 + "]" * 100_000}, _TOO_DEEP),
    ({"report_raw": "1" * 5000}, _UNUSABLE + "ValueError: Exceeds the limit (4300 digits)"),
    ({"report_raw": '{"install": "\udcff"}'},
     _UNUSABLE + "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff"),
    # json.loads of bytes would read UTF-16, where U+4122 is the bytes of '"A': the brackets after
    # it would be inside a string to the depth count and nested to the parser.
    ({"report_raw": ('["\u4122", ' + "[" * 100 + "]" * 100 + "]").encode("utf-16-le").decode("ascii")},
     _UNUSABLE + "JSONDecodeError: "),
], ids=["past-its-size", "a-symlink-to-a-host-file", "a-named-pipe", "a-directory", "unreadable",
        "in-a-locked-directory", "missing", "not-json", "not-pips-shape", "nested-to-its-depth",
        "nested-one-past-its-depth", "nested-100000-deep", "an-integer-of-5000-digits", "not-utf8",
        "utf16-hiding-its-nesting"])
def test_a_dry_run_report_pip_would_not_write_is_the_submissions(tmp_path, tools, change, reason):
    """The step's container wrote the report, and pip writes a plain, readable UTF-8 file far
    smaller than ``REPORT_MAX_BYTES`` and nested far less than ``REPORT_MAX_DEPTH`` that parses as
    its installation report: anything else the step left is the submission's doing, whatever way
    it fails, permissions that keep this process out among them. This process opens it once,
    following no link and never waiting on a pipe, and reads at most one byte past the limit."""
    bin_dir, plan = tools
    _set(plan, **change)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    try:
        outputs = PackageStage().run(ctx)
    finally:
        (ctx.role_dir("dry-run") / "build-tmp").chmod(0o755)
    result = _result(ctx, outputs)
    assert result.status == "invalid_artifact" and reason in (result.reason or "")
    assert result.returncode == 0 and "install_report" not in outputs and _started(ctx)[-1] == "dry-run"


def _bytes_read() -> int:
    """The bytes this process has read through system calls so far (``rchar``)."""
    return int(next(line.split()[1] for line in Path("/proc/self/io").read_text().splitlines()
                    if line.startswith("rchar:")))


def test_the_dry_run_report_may_hold_eight_mib():
    """pip writes a report of a few hundred KB; 8 MiB is far above that, and bounds what parsing
    the worst report a build hook can write costs the runner to about 400 MB, on every interpreter
    since ``REPORT_MAX_DEPTH`` keeps a deeper one from the parser (``REPORT_MAX_BYTES``)."""
    assert REPORT_MAX_BYTES == 8 * 2**20


def test_the_dry_run_report_may_nest_32_deep():
    """pip's report nests six deep, to an archive's hashes in an item's ``download_info``; 32
    leaves room for a newer pip."""
    assert REPORT_MAX_DEPTH == 32


def test_the_dry_run_report_is_read_up_to_its_size_and_no_further(tmp_path):
    at, past, gone = tmp_path / "at.json", tmp_path / "past.json", tmp_path / "gone.json"
    for path, size in ((at, REPORT_MAX_BYTES), (past, 4 * REPORT_MAX_BYTES)):
        with open(path, "wb") as f:
            f.truncate(size)
    assert len(read_report(at)) == REPORT_MAX_BYTES
    before = _bytes_read()
    with pytest.raises(ForeignReport, match=f"holds more than the {REPORT_MAX_BYTES} bytes allowed"):
        read_report(past)
    # A report far past the limit is read one byte past it, never whole.
    assert REPORT_MAX_BYTES < _bytes_read() - before < 2 * REPORT_MAX_BYTES
    with pytest.raises(ForeignReport, match="^the dry run wrote no report.json$"):
        read_report(gone)


class _FailingFile:
    """A file object whose every read fails with errno *code*."""

    def __init__(self, code: int) -> None:
        self.code = code

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass

    def read(self, size: int = -1) -> bytes:
        raise OSError(self.code, os.strerror(self.code))


@pytest.mark.parametrize("call", ["lstat", "open", "fstat", "read"])
@pytest.mark.parametrize("code, submissions", ERRNO_SIDES, ids=ERRNO_IDS)
def test_a_report_read_error_is_the_submissions_only_for_its_permissions(tmp_path, monkeypatch, call, code,
                                                                         submissions):
    """At each step of reading the report, EACCES and EPERM come of the permissions the step's
    container left, the submission's doing (``ForeignReport``); any other error is the node's,
    raised for the stage to report as infra. A descriptor once open is closed either way."""
    report = tmp_path / "report.json"
    report.write_text("{}")
    real = {name: getattr(os, name) for name in ("lstat", "open", "fstat", "close", "fdopen")}
    opened, closed = [], []

    def fail(*args):
        raise OSError(code, os.strerror(code))

    def lstat(path, *args, **kwargs):
        return fail() if call == "lstat" and Path(path) == report else real["lstat"](path, *args, **kwargs)

    def open_(path, *args, **kwargs):
        if call == "open" and Path(path) == report:
            fail()
        opened.append(real["open"](path, *args, **kwargs))
        return opened[-1]

    def fstat(fd):
        return fail() if call == "fstat" and fd in opened else real["fstat"](fd)

    def fdopen(fd, *args, **kwargs):
        return _FailingFile(code) if call == "read" and fd in opened else real["fdopen"](fd, *args, **kwargs)

    def close(fd):
        closed.append(fd)
        real["close"](fd)

    fakes = {"lstat": lstat, "open": open_, "fstat": fstat, "fdopen": fdopen, "close": close}
    for name, fake in fakes.items():
        monkeypatch.setattr(package_stage.os, name, fake)
    with pytest.raises(ForeignReport if submissions else OSError, match=rf"\[Errno {code}\]") as caught:
        read_report(report)
    assert isinstance(caught.value, ForeignReport) is submissions
    assert closed == opened and len(opened) == (call in ("fstat", "read"))


def test_a_dry_run_report_too_big_to_parse_in_memory_is_the_submissions(monkeypatch):
    """Only the step's container wrote the report, so running out of memory parsing it is the
    submission's failure too."""
    def loads(data):
        raise MemoryError

    monkeypatch.setattr(package_stage.json, "loads", loads)
    with pytest.raises(ForeignReport, match="^the dry run's report.json cannot be used: MemoryError$"):
        report_pins(b"{}")


def test_a_dry_run_report_nested_past_its_depth_is_refused_before_it_is_parsed(monkeypatch):
    """The deepest report the size allows, 4,194,304 levels, is the submission's without reaching
    the parser, whose reach and cost past the bound depend on the interpreter and its stack: from
    Python 3.14 it parses this one under an unlimited stack, in about 960 MB."""
    def loads(data):
        raise AssertionError("the report was parsed")

    monkeypatch.setattr(package_stage.json, "loads", loads)
    half = REPORT_MAX_BYTES // 2
    with pytest.raises(ForeignReport) as caught:
        report_pins(b"[" * half + b"]" * half)
    assert str(caught.value) == _TOO_DEEP


def test_a_dry_run_report_within_its_depth_parses_on_the_least_stack_a_thread_may_have():
    """A report as deep as ``REPORT_MAX_DEPTH`` allows parses in a thread of 64 KiB, about the
    least stack an interpreter lets a thread have (52 KiB on 3.14), so no report the bound admits
    can run the parser out of stack."""
    data = b"[" * REPORT_MAX_DEPTH + b"]" * REPORT_MAX_DEPTH
    raised: list[Exception] = []

    def parse() -> None:
        try:
            report_pins(data)
        except Exception as e:
            raised.append(e)

    previous = threading.stack_size(64 * 2**10)
    try:
        thread = threading.Thread(target=parse)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    assert len(raised) == 1 and isinstance(raised[0], ForeignReport), raised
    assert str(raised[0]).startswith(_UNUSABLE + "ValueError: not a pip installation report: ")


def test_a_recursion_error_parsing_a_report_within_its_depth_is_not_the_submissions(monkeypatch):
    """Within ``REPORT_MAX_DEPTH`` the report cannot be what runs the parser out of stack (previous
    test), so a RecursionError from the parse is this process's own and raised as it is."""
    def loads(data):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(package_stage.json, "loads", loads)
    with pytest.raises(RecursionError):
        report_pins(b"{}")


def test_why_a_dry_run_report_cannot_be_used_is_kept_short():
    """A pin is as long as the report lets it be; the reason keeps the start of why."""
    item = {"metadata": {"name": "x" * 100_000 + " y", "version": "1"},
            "download_info": {"url": "https://h/x.whl"}, "is_direct": False}
    with pytest.raises(ForeignReport) as caught:
        report_pins(json.dumps({"install": [item]}).encode())
    reason = str(caught.value)
    assert reason.startswith(_UNUSABLE + "ValueError: not a pin of one requirement: 'xxx")
    assert len(reason) == len(_UNUSABLE) + REPORT_REASON_CHARS


def test_a_package_builds_on_a_node_with_exactly_the_stages_disk(tmp_path, tools, monkeypatch):
    """Each extracted tree is KEPT_DISK_GIB by its apparent size, and the node has exactly the
    disk the stage declares. Every role's grant is checked as the step wrapper checks it."""
    bin_dir, _ = tools
    monkeypatch.setenv("FAKE_TREE_GIB", str(KEPT_DISK_GIB))
    launcher = LocalLauncher(node_disk_gib=SHAPE.disk_gib, scratch=tmp_path / "node" / "hls-100")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD), launcher=launcher)
    assert _result(ctx, PackageStage().run(ctx)).status == "built"
    declared = {"extract" if s.name.startswith("extract-") else s.name: s.shape.disk_gib for s in launcher.launched}
    assert declared == {"checkout": 0, "extract": 0, "dry-run-copy": 0, "dry-run": PIP.disk_gib, "dry-run-clean": 0,
                        "wheel-copy": 0, "wheel": PIP.disk_gib, "wheel-clean": 0, "wheel-check": 0,
                        "build": 0, "integrity": 0, "publish": 0, "install-scan": 0}
    # Each pip step holds the memory the plan states for it, as the brief does.
    assert {s.name: s.shape.memory_gib for s in launcher.launched if s.name in ("dry-run", "wheel")} == {
        "dry-run": PIP.memory_gib, "wheel": PIP.memory_gib}
    free = [(h.spec.name, h.report(1.0).granted.disk_free_gib) for h in launcher.handles]
    by_role = dict(free)
    extracts = [f for name, f in free if name.startswith("extract-")]
    # The base's tree took its share before the checkout; the build still had room for its
    # unpacking, and the submission's extraction for its tree.
    assert extracts[0] - by_role["checkout"] >= KEPT_DISK_GIB
    assert by_role["build"] >= BUILD_DISK_GIB
    assert extracts[-1] >= KEPT_DISK_GIB


def test_the_dry_runs_writes_are_gone_before_the_wheel_step_starts(tmp_path, tools, monkeypatch):
    """The two pip steps' caps never coexist: a dry run that wrote nearly its whole cap leaves
    the wheel step the free disk it had itself, on a node with exactly the stage's disk."""
    bin_dir, plan = tools
    _set(plan, dry_run_fill_gib=PIP.disk_gib - 1)
    monkeypatch.setenv("FAKE_TREE_GIB", str(KEPT_DISK_GIB))
    launcher = LocalLauncher(node_disk_gib=SHAPE.disk_gib, scratch=tmp_path / "node" / "hls-100")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD), launcher=launcher)
    assert _result(ctx, PackageStage().run(ctx)).status == "built"
    by_role = {h.spec.name: h.report(1.0).granted.disk_free_gib for h in launcher.handles}
    assert by_role["dry-run-clean"] <= by_role["dry-run"] - (PIP.disk_gib - 2)
    assert by_role["wheel"] >= by_role["dry-run"] - 0.01


def test_a_step_that_locked_its_directories_is_still_cleaned(tmp_path, tools):
    """A build hook may leave directories its owner can neither read nor enter; the clean gives
    the permissions back first, so a step's own writes never fail the harness."""
    bin_dir, plan = tools
    _set(plan, dry_run_locked=True)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    assert _result(ctx, PackageStage().run(ctx)).status == "built"
    assert not (ctx.role_dir("dry-run") / "build-tmp").exists()


@pytest.mark.parametrize("locked", ["dir", "file"], ids=["the-wheels-directory", "a-wheel"])
def test_wheels_the_wheel_step_locked_are_given_back_before_anything_reads_them(tmp_path, tools, locked):
    """The wheel step may take every permission off the wheels directory it writes, or off a
    wheel there. Its clean role gives the owner every permission back on what the stage keeps,
    before this process lists the wheels and the build copies them, so the step's own writes
    never fail the harness."""
    bin_dir, plan = tools
    _set(plan, wheels_locked=locked)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    assert _result(ctx, PackageStage().run(ctx)).status == "built"
    wheels = ctx.role_dir("wheel") / "wheels"
    assert os.access(wheels, os.R_OK | os.W_OK | os.X_OK)
    assert os.access(wheels / SUBMISSION_WHEEL, os.R_OK | os.W_OK)
    clean = next(s for s in ctx.launcher.launched if s.name == "wheel-clean")
    assert str(wheels) in clean.argv and _started(ctx).index("wheel-clean") < _started(ctx).index("build")


def test_giving_the_wheels_their_permissions_back_never_follows_a_link(tmp_path, tools):
    """A link the wheel step leaves among the wheels may lead to a file the harness owns; the
    clean role changes the wheels alone, never what a link leads to."""
    bin_dir, plan = tools
    harness_file = tmp_path / "harness-file"
    harness_file.write_text("x")
    harness_file.chmod(0o400)
    _set(plan, wheel="link", wheel_link_target=str(harness_file))
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and "['passwd.whl']" in (result.reason or "")
    assert stat.S_IMODE(harness_file.stat().st_mode) == 0o400


def test_wheels_this_process_still_cannot_list_are_the_submissions(tmp_path, tools, monkeypatch):
    """Only the wheel step wrote the wheels directory, so should it stay unreadable after the
    clean role, listing it fails on the submission's account, never as an error of this
    process, which would be retried."""
    bin_dir, plan = tools
    _set(plan, wheels_locked="dir")
    # A clean role that gives back the permissions only on what it removes.
    keeps_nothing = 'shift "$(($1 + 1))"; chmod -R u+rwX -- "$@" && rm -rf -- "$@"'
    monkeypatch.setattr(package_stage, "CLEAN_SHELL", keeps_nothing)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    wheels = ctx.role_dir("wheel") / "wheels"
    try:
        result = _result(ctx, PackageStage().run(ctx))
    finally:
        if wheels.exists():
            wheels.chmod(0o755)
    assert result.status == "invalid_artifact" and result.returncode is None
    denied = f"[Errno 13] Permission denied: '{wheels}'"
    assert result.reason == f"the wheel step's output cannot be read: {denied}"
    assert _started(ctx)[-1] == "wheel-clean"


def test_a_wheel_that_unpacks_to_more_than_it_declares_is_the_contestants_before_the_build(tmp_path, tools):
    """A wheel step can leave a decompression bomb, which pip in the offline build would read:
    the wheel check refuses it, naming the wheel and its entry, before the build starts."""
    bin_dir, plan = tools
    _set(plan, wheel="lying")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    outputs = PackageStage().run(ctx)
    result = _result(ctx, outputs)
    assert result.status == "invalid_artifact" and result.reason == (
        "the entry 'bomb/__init__.py' of the wheel httpx-0.27.0-py3-none-any.whl unpacks to more than the 100 "
        "bytes it declares")
    assert _started(ctx)[-2:] == ["wheel-clean", "wheel-check"] and "submission" not in outputs


@pytest.mark.parametrize("code", [errno.EIO, errno.ESTALE, errno.ENOSPC], ids=["EIO", "ESTALE", "ENOSPC"])
def test_a_node_error_in_the_wheel_check_is_retryable_infra(tmp_path, tools, code):
    """The check records an error of the node reading a wheel with its errno; the stage raises
    it as a retryable fault and builds nothing."""
    bin_dir, _ = tools
    script = tmp_path / "node_error.py"
    script.write_text("import sys\nfrom pathlib import Path\nfrom hls.package.wheel_check import WheelCheck\n"
                      "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
                      f"out.write_text(WheelCheck(node_error='the wheel x.whl: [Errno {code}] boom', "
                      f"node_errno={code}).model_dump_json())\n")
    launcher = LocalLauncher(replace_modules={"hls.package.wheel_check": script})
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD), launcher=launcher)
    with pytest.raises(StageFault, match=rf"^the wheel check could not read the wheels: the wheel x.whl: "
                                         rf"\[Errno {code}\] boom$") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert caught.value.fault.evidence["errno"] == str(code) and _started(ctx)[-1] == "wheel-check"


def test_a_wheel_check_that_fails_itself_is_a_retryable_fault(tmp_path, tools):
    bin_dir, _ = tools
    crash = tmp_path / "crash.py"
    crash.write_text("import sys\nsys.exit(1)\n")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD),
               launcher=LocalLauncher(replace_modules={"hls.package.wheel_check": crash}))
    with pytest.raises(StageFault, match="harness role wheel-check ended exited with status 1") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable


@pytest.mark.parametrize("code, submissions", ERRNO_SIDES, ids=ERRNO_IDS)
def test_a_wheels_listing_error_is_the_submissions_only_for_its_permissions(tmp_path, monkeypatch, code,
                                                                            submissions):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    real = os.scandir

    def scandir(path):
        if Path(path) == wheels:
            raise OSError(code, os.strerror(code), str(path))
        return real(path)

    monkeypatch.setattr(package_stage.os, "scandir", scandir)
    if submissions:
        assert wheels_problem(wheels, "subm", []) == (f"the wheel step's output cannot be read: [Errno {code}] "
                                                          f"{os.strerror(code)}: '{wheels}'")
    else:
        with pytest.raises(OSError, match=rf"\[Errno {code}\]"):
            wheels_problem(wheels, "subm", [])


@pytest.mark.parametrize("site", ["report", "wheels"])
@pytest.mark.parametrize("code", [errno.EIO, errno.ESTALE, errno.ENOSPC], ids=["EIO", "ESTALE", "ENOSPC"])
def test_a_node_error_reading_what_a_step_wrote_is_retryable_infra(tmp_path, tools, monkeypatch, site, code):
    """An error reading the dry run's report or the wheels that the submission's permissions
    do not explain is the node's: a retryable fault, with its errno in the reason."""
    bin_dir, _ = tools
    path = {"report": "/r/report.json", "wheels": "/r/wheels"}[site]

    def fail(*args):
        raise OSError(code, os.strerror(code), path)

    monkeypatch.setattr(package_stage, {"report": "read_report", "wheels": "wheels_problem"}[site], fail)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    what = {"report": "the dry run's report.json", "wheels": "the wheel step's output"}[site]
    with pytest.raises(StageFault, match=rf"^{what} cannot be read: \[Errno {code}\]") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert caught.value.fault.evidence["errno"] == str(code)
    assert not ctx.store.exists(ctx.output_key("package.json"))


def test_the_base_is_pinned_before_any_other_role_and_no_cache_entry_is_copied(tmp_path, tools):
    """A pin starts a role and waits for it, so the stage pins its base first; the submission's
    own image can only be pinned once built, when no other role runs. The build reads the base's
    file from its entry itself: the stage copies no cache entry to the node."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    assert _result(ctx, PackageStage().run(ctx)).status == "built"
    events = [e for e in read_events(ctx.attempt_dir / "events.jsonl")
              if e["kind"] in ("image_extracted", "role_started", "role_ended")]
    marks = [e["kind"] if e["kind"] == "image_extracted" else f"{e['kind']}:{e['role'].split('-')[0]}"
             for e in events]
    assert marks[:4] == ["role_started:extract", "role_ended:extract", "image_extracted", "role_started:checkout"]
    second = marks.index("image_extracted", 3)
    assert marks[second - 3:second] == ["role_ended:build", "role_started:extract", "role_ended:extract"]
    assert ctx.cache_entries == {f"images/{BASE_KEY}": json.loads(ctx.store.read_bytes(
        ctx.plan.layout.attempt_file("images", 1, "outputs/images.json")))[BASE]["entry_sha256"]}
    assert ctx.scratch is not None and not copies_root(ctx.scratch, ctx.job_id).exists()


def test_a_base_file_that_is_not_the_one_its_entry_records_is_infra_never_retried(tmp_path, tools, monkeypatch):
    """The build copies the base's file from the store and checks the copy before it builds; a
    copy with another digest would be read again by a retry."""
    bin_dir, _ = tools
    real = apptainer_images.build_argv

    def build_argv(def_file, out, *, digest, parent, apptainer):
        return real(def_file, out, digest=digest, parent=(parent[0], "0" * 64, parent[2]), apptainer=apptainer)

    monkeypatch.setattr(apptainer_images, "build_argv", build_argv)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    with pytest.raises(StageFault, match="refused|has sha256") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert "has sha256" in caught.value.fault.reason and _started(ctx)[-1] == "build"


@pytest.mark.parametrize("network", [
    NetworkPolicy(),
    NetworkPolicy(domains=AccessList(mode="allow"), packages=AccessList(mode="allow", entries=["httpx"])),
    NetworkPolicy(domains=AccessList(mode="allow"), packages=AccessList(mode="deny", entries=[])),
], ids=["open", "allow-rule", "empty-deny-rule"])
def test_a_network_without_a_package_deny_rule_is_refused_when_planned(network):
    params = PackageStage.params_model.model_validate(
        {"task": "t", "submission_base": BASE, "constraints": CONSTRAINTS, "pip": PIP.model_dump(),
         "network": network.model_dump(mode="json"), "disk_cap_gib": CAP_GIB})
    with pytest.raises(ValueError, match="has no package-deny rule"):
        PackageStage.check(SHAPE, params, runner_memory_gib=4)


#: A producer's own deny list of 12 serving frameworks and kernels, with no vllm or flashinfer.
TWELVE = ["sglang", "lmdeploy", "tensorrt-llm", "text-generation-inference", "ctranslate2", "flash-attn", "xformers",
          "exllamav2", "auto-gptq", "autoawq", "llama-cpp-python", "deepspeed"]


def test_a_producer_with_its_own_deny_list_packages_behind_a_seal_that_holds_it(tmp_path, tools, monkeypatch):
    """Not every producer declares the harness list: one may deny its own 12 names. Its pip
    steps' seal carries exactly those."""
    bin_dir, _ = tools
    seals = []
    real = package_stage.SealPlan

    def sealplan(**kw):
        seals.append(kw["deny_list"])
        return real(**kw)

    monkeypatch.setattr(package_stage, "SealPlan", sealplan)
    network = NetworkPolicy(domains=AccessList(mode="allow"), packages=AccessList(mode="deny", entries=TWELVE))
    PackageStage.check(SHAPE, PackageStage.params_model.model_validate(
        {"task": "t", "submission_base": BASE, "constraints": CONSTRAINTS, "pip": PIP.model_dump(), "network": network.model_dump(mode="json"),
         "disk_cap_gib": CAP_GIB}), runner_memory_gib=4)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD), network=network)
    assert _result(ctx, PackageStage().run(ctx)).status == "built"
    assert seals == [tuple(TWELVE)] * 2


@pytest.mark.parametrize("added, status", [(24 * 2**30, "built"), (24 * 2**30 + 1, "invalid_artifact")],
                         ids=["at-the-limit", "one-byte-past"])
def test_an_install_that_adds_more_than_the_limit_to_the_venv_is_the_contestants(tmp_path, tools, added, status):
    """A wheel unpacks to more than it ships; the install may add 24 GiB to /opt/venv, and a
    submission past that is refused before its image is extracted anywhere."""
    bin_dir, plan = tools
    _set(plan, installed_bytes=added)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == status
    if status == "invalid_artifact":
        assert result.reason == (f"the submission's install adds {added} bytes to /opt/venv, more than the 24 GiB "
                                 "allowed")
        assert _started(ctx)[-1] == "build" and "submission" not in ctx.image_digests


def test_a_build_that_reports_no_installed_size_is_infra(tmp_path, tools):
    bin_dir, plan = tools
    _set(plan, installed_bytes="unknown")
    with pytest.raises(StageFault, match="reported no size") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD)))
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable


@pytest.mark.parametrize("files, links, first", SYMLINKS.values(), ids=SYMLINKS)
def test_a_tree_with_any_symlink_is_invalid_before_any_copy_or_container(tmp_path, tools, files, links, first):
    """The image's ``%files`` copy follows every link, so a submission may hold none: the
    checkout refuses the tree from git's listing, and nothing of it is written, copied or built.
    The chain's ``t`` would lead to a file beside the checked-out tree."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, {**GOOD, **files}, links=links))
    (ctx.role_dir("checkout") / "secret").write_text("HOST-SECRET\n")
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "invalid_artifact" and result.reason == (
        f"the submission is past the package's limits: the tree tracks a symlink, which a submission may not hold: "
        f"{first!r}")
    assert _started(ctx) == ["extract", "checkout"] and not (ctx.role_dir("checkout") / "tree").exists()


def test_a_copy_that_fails_is_a_retryable_harness_fault_never_the_contestants(tmp_path, tools):
    """Within the checkout's limits a copy cannot fail on the tree's account, so a copy that
    fails is the harness's: retried, and no container ran."""
    bin_dir, _ = tools
    cp = bin_dir / "cp"
    cp.write_text("#!/bin/sh\ncase \"$*\" in *workspace*) echo 'cp: No space left on device' >&2; exit 1 ;; esac\n"
                  "exec /bin/cp \"$@\"\n")
    cp.chmod(0o755)
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    with pytest.raises(StageFault, match="dry-run-copy") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert _started(ctx) == ["extract", "checkout", "dry-run-copy"]



_NONCE = "0123456789abcdef" * 2
_INSTALLED = f"hls-offline-install-rc-{_NONCE}=0\n"
_ADDED = f"hls-offline-install-bytes-{_NONCE}="
_LIMIT = 24 * 2**30


@pytest.mark.parametrize("ending, code, oom, log, verdict", [
    ("exited", 0, 0, _INSTALLED + f"{_ADDED}12\n", "ok"),
    ("exited", 1, 0, f"hls-offline-install-rc-{_NONCE}=1\n", "invalid"),
    ("exited", 255, 0, _INSTALLED + f"{_ADDED}{_LIMIT + 1}\n", "invalid"),
    ("exited", 0, 0, _INSTALLED + f"{_ADDED}{_LIMIT + 1}\n", "invalid"),
    ("exited", 255, 0, _INSTALLED + f"{_ADDED}{_LIMIT}\n", "infra"),
    ("exited", 255, 0, "FATAL: no space left on device\n", "infra"),
    ("exited", 255, 1, f"hls-offline-install-rc-{_NONCE}=1\n", "infra"),
    ("time_limit", None, 0, _INSTALLED + f"{_ADDED}{_LIMIT + 1}\n", "infra"),
    ("exited", 1, 0, f"hls-offline-install-rc-{'f' * 32}=1\n", "infra"),
], ids=["built", "install-failed", "stopped-past-the-limit", "built-past-the-limit", "at-the-limit-but-failed",
        "before-the-install", "killed-for-memory", "cut-by-the-deadline", "another-nonce"])
def test_whose_failure_the_offline_build_is(ending, code, oom, log, verdict):
    """The submission's when its install failed or added more than the limit, which stops the
    build: whatever the build's own status then; ours for any other failure, and for any ending
    but the build's own exit. Only lines carrying the build's own nonce are its status."""
    outcome = RoleOutcome("build", "harness", code, ending, oom)  # type: ignore[arg-type]
    assert build_verdict(outcome, log, _NONCE) == verdict


def test_a_failing_step_whose_seal_ended_otherwise_than_its_payload_is_infra(tmp_path, tools):
    """A pip step's exit is its payload's only by the seal's own record of that ending, with
    the interceptor alive and the same status (``_seal_failure``): a seal that exited with
    another status after its payload failed is the harness's, never the submission's."""
    bin_dir, plan = tools
    _set(plan, dry_run="fail")
    (tmp_path / "pasta.plan").write_text("pass_exit:2\n")
    with pytest.raises(StageFault, match="dry-run: runtime failed") as caught:
        PackageStage().run(_ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD)))
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert caught.value.fault.evidence["dry-run.detail"] == "the seal exited 2, its payload 1"


def _corrupting_cp(bin_dir: Path, pattern: str) -> None:
    """A ``cp`` on PATH that appends a byte to a copy whose destination matches the shell
    pattern *pattern*, as a copy gone wrong on the way would land."""
    cp = bin_dir / "cp"
    cp.write_text(f'#!/bin/sh\n/bin/cp "$@" || exit\nfor last; do :; done\ncase "$last" in {pattern}) '
                  'printf x >> "$last";; esac\n')
    cp.chmod(0o755)


def test_a_base_copy_that_went_wrong_on_the_way_is_retried(tmp_path, tools):
    """The build's copy of the base's file differs while the store's file still has the digest
    its entry records: the copy went wrong on the way, the role says so with its own status
    (``COPY_FAILED_EXIT``), and a retry copies again."""
    bin_dir, _ = tools
    _corrupting_cp(bin_dir, f"*/build/{IMAGE_FILE}")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    with pytest.raises(StageFault, match="harness role build ended exited with status 75") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert not (ctx.role_dir("build") / IMAGE_FILE).exists() and _started(ctx)[-1] == "build"


def test_a_bundle_copy_the_plan_names_that_is_not_there_is_infra_never_retried(tmp_path, tools):
    """A retry would look for the same file, so nothing starts: not even the base's pin."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    ctx.store.path(INSTANCE.source_bundle).unlink()
    with pytest.raises(StageFault, match=f"the copy {INSTANCE.source_bundle} of the source bundle does not "
                                         "exist") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert _started(ctx) == []


def test_a_twelve_name_producer_lets_its_submission_declare_what_its_list_allows(tmp_path, tools):
    """The deny rule is the producer's own: under a producer's own twelve names a
    submission that declares vllm passes the prescan, its steps' seal carries those twelve, and
    under the harness list the same submission is refused before any container."""
    bin_dir, _ = tools
    network = NetworkPolicy(domains=AccessList(mode="allow"), packages=AccessList(mode="deny", entries=TWELVE))
    vllm = {**GOOD, "pyproject.toml": PYPROJECT.replace('"httpx>=0.27"', '"vllm"')}
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, vllm), network=network)
    assert _result(ctx, PackageStage().run(ctx)).status == "built"
    policies = [SealPolicy.model_validate_json(s.argv[s.argv.index("hls-seal") + 5]) for s in ctx.launcher.launched
                if s.owner == "contestant"]
    assert [p.network.packages for p in policies] == [AccessList(mode="deny", entries=list(TWELVE))] * 2
    again = tmp_path / "shared"
    ctx = _ctx(again, bin_dir, bundle=make_bundle(again, vllm))
    assert "not permitted by policy: ['vllm']" in (_result(ctx, PackageStage().run(ctx)).reason or "")


def test_the_runner_opens_no_image_file_and_reads_no_bundle_or_entry(tmp_path, tools):
    """Every image file, the bundle and every cache entry file are read by roles alone: the
    stage's own process reads only the small records it decides on."""
    bin_dir, _ = tools
    bundle = make_bundle(tmp_path, GOOD)
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    _agent_build(ctx, bundle)
    with recording_reads() as reads:
        outputs = PackageStage().run(ctx)
        refs = [ctx.output_ref(key) for key in outputs.values()]
        ctx.close()
    assert _result(ctx, outputs).status == "built" and len(refs) == 6
    assert [r for r in reads if ".sif" in r or ".bundle" in r] == []
    assert ctx.scratch is not None
    roles = ctx.scratch / "roles"
    roots = [ctx.store.root / "cache", ctx.scratch / "sandboxes", roles / "wheel" / "wheels"]
    assert entry_reads(reads, roots) == []
    # Of the build's outputs it reads only its record of the figures the brief stated.
    build = ctx.store.path(ctx.plan.layout.attempt(AGENT_BUILD_STAGE, 1))
    assert {Path(r).name for r in reads if Path(r).is_relative_to(build)} == {"served_ucounts.json"}
    # Of what the roles wrote it reads the records it decides on, and the build's digest report.
    assert sorted({Path(r).name for r in reads if Path(r).is_relative_to(ctx.outputs_dir)}) == [
        "integrity.json", "source.json"]
    assert sorted({Path(r).name for r in reads if Path(r).is_relative_to(roles / "build")}) == ["sha256"]
    checked = sorted({Path(r).name for r in reads if Path(r).is_relative_to(roles / "wheel-check")})
    assert checked == ["check.json"]


# --- an agent_build that left nothing to package --------------------------------------------------

NO_MAIN = ("the workspace's repository has no commit on main (its branches are dev); only the latest commit on "
           "main is evaluated")


def _left_nothing(ctx: StageContext, *, data: bytes | None = None, bundle: Path | None = None,
                  tamper: bool = False) -> None:
    """Add this instance's agent_build manifest to *ctx*'s inputs, with a no_submission output
    holding *data* (a no_main record by default), each output by its writer's digest, and a
    source bundle too when *bundle* is given."""
    key = ctx.plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/no_submission.json")
    data = data if data is not None else NoSubmission("no_main", NO_MAIN, 1.0).to_json().encode()
    ctx.store.write_bytes(key, data)
    outputs = {NO_SUBMISSION_OUTPUT: output_ref(ctx.store, key, sha256_bytes(data)), **_served(ctx)}
    if bundle is not None:
        bundle_key = ctx.plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/source.bundle")
        bundle_data = bundle.read_bytes()
        ctx.store.write_bytes(bundle_key, bundle_data)
        outputs[SOURCE_BUNDLE_OUTPUT] = output_ref(ctx.store, bundle_key, sha256_bytes(bundle_data))
    ctx.inputs[ctx.plan.layout.manifest(AGENT_BUILD_STAGE)] = _manifest(ctx, AGENT_BUILD_STAGE, outputs)
    if tamper:
        ctx.store.write_bytes(key, b"changed")


def test_a_build_that_left_nothing_to_package_is_invalid_before_any_role(tmp_path, tools):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    _left_nothing(ctx)
    outputs = PackageStage().run(ctx)
    result = _result(ctx, outputs)
    assert (result.status, result.reason) == ("invalid_artifact", f"the build left nothing to package: {NO_MAIN}")
    assert (result.bundle_sha256, result.commit, result.image) == (None, None, None)
    assert result.served_ucounts == SERVED
    assert ctx.launcher.launched == [] and _started(ctx) == [] and ctx.cache_entries == {}
    assert set(outputs) == {"package"}


def test_the_reason_is_scrubbed_before_it_is_published(tmp_path, tools):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    _left_nothing(ctx, data=NoSubmission("unreadable", f"git said {SECRET}", 1.0).to_json().encode())
    result = _result(ctx, PackageStage().run(ctx))
    assert SECRET not in (result.reason or "") and "<redacted:relay>" in (result.reason or "")


@pytest.mark.parametrize("case, match", [
    ("both", "both a source bundle and no_submission"),
    ("tampered", "does not match its manifest"),
    ("not-a-record", "not a no_submission record"),
])
def test_a_no_submission_output_that_contradicts_its_build_is_refused(tmp_path, tools, case, match):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    if case == "both":
        _left_nothing(ctx, bundle=make_bundle(tmp_path, GOOD))
    else:
        _left_nothing(ctx, data=b"{}" if case == "not-a-record" else None, tamper=case == "tampered")
    with pytest.raises(StageFault, match=match) as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert ctx.launcher.launched == []


def test_an_agent_build_with_neither_output_keeps_the_refusal(tmp_path, tools):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    ctx.inputs[ctx.plan.layout.manifest(AGENT_BUILD_STAGE)] = _manifest(ctx, AGENT_BUILD_STAGE, {})
    with pytest.raises(StageFault, match="no source bundle") as caught:
        PackageStage().run(ctx)
    assert not caught.value.fault.retryable


def test_a_named_bundle_is_built_whatever_agent_build_published(tmp_path, tools):
    bin_dir, _ = tools
    bundle = make_bundle(tmp_path, GOOD)
    ctx = _ctx(tmp_path, bin_dir, bundle=bundle)
    _left_nothing(ctx)
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "built" and result.bundle_sha256 == sha256_file(bundle)[0]


def test_the_pip_steps_run_under_the_stages_process_ceiling_which_the_result_records(tmp_path, tools):
    """The submission's build backends are its code: both pip steps run under the stage's
    process ceiling, as every contestant container does, and package.json records it. The
    harness roles keep the runner's limit."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "built" and result.process_ceiling == ctx.process_ceiling()
    ended = {e["role"]: e["ceiling"] for e in read_events(ctx.attempt_dir / "events.jsonl")
             if e["kind"] == "role_ended"}
    assert ended["dry-run"] == ended["wheel"] == result.process_ceiling.ceiling
    assert ended["checkout"] is None and ended["dry-run-copy"] is None


@pytest.mark.parametrize("ending", ["failed", "time_limit"])
def test_a_failing_pip_step_the_uids_other_tasks_starved_is_infra_not_the_submissions(
        tmp_path, tools, monkeypatch, ending):
    """Once the step is gone, the uid's other tasks alone fill its pool: its forks may have failed
    for what others started. Infra, retried, before any verdict, whether it exited failing or
    was cut at the pip steps' limit."""
    bin_dir, plan = tools
    monkeypatch.setattr(package_stage, "PIP_STEPS_SEC", 2)
    _set(plan, dry_run="fail" if ending == "failed" else "hang")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    ceiling = ctx.process_ceiling().ceiling
    monkeypatch.setattr("hls.pipeline.stage.uid_tasks", lambda uid: ceiling)
    with pytest.raises(StageFault) as caught:
        PackageStage().run(ctx)
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "starved" in fault.reason


@pytest.mark.parametrize("ending", ["failed", "time_limit"])
def test_a_failing_pip_step_whose_uids_tasks_cannot_be_counted_is_infra_not_the_submissions(
        tmp_path, tools, monkeypatch, ending):
    """Once the step is gone the uid's tasks could not be counted, so nothing rules out that the
    others filled its pool: infra, retried, before any verdict, whether it exited failing or was
    cut at the pip steps' limit."""
    bin_dir, plan = tools
    monkeypatch.setattr(package_stage, "PIP_STEPS_SEC", 2)
    _set(plan, dry_run="fail" if ending == "failed" else "hang")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))

    def uncountable(uid):
        raise OSError(errno.EMFILE, "Too many open files", "/proc")

    monkeypatch.setattr("hls.pipeline.stage.uid_tasks", uncountable)
    with pytest.raises(StageFault) as caught:
        PackageStage().run(ctx)
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "could not be counted" in fault.reason
    assert [v for k, v in fault.evidence.items() if k.endswith("others_tasks")] == ["uncounted"]


def test_a_built_package_names_its_bundle():
    with pytest.raises(ValueError, match="names the bundle"):
        PackageResult(task="t", status="built", bundle_sha256=None, commit="f" * 40, image="k",
                      sif_sha256="e" * 64, served_ucounts=None)


# ---------------------------------------------------------------------------
# The scan of what the install added, which the image compresses
# ---------------------------------------------------------------------------


#: The venv's site-packages, from an image's root.
SITE = "opt/venv/lib/python3.12/site-packages"


def _scan(ctx: StageContext, outputs: dict[str, str]) -> PackageScan:
    return PackageScan.model_validate_json(ctx.store.read_bytes(outputs["package_scan"]))


def test_a_key_a_wheel_installed_is_found_among_the_files_the_install_added(tmp_path, tools):
    """The submission image is squashfs, compressed: once its system paths are the base's, the stage
    scans every file of its venv the base does not hold as it is, the wheels' entries and the
    bytecode pip compiled, and no other file of the image. The tree and the bundle are the
    source_scan stage's, before any of the contestant's code ran."""
    bin_dir, plan = tools
    _set(plan, wheel_entries={"subm/key.py": f"KEY = {SECRET!r}\n"},
         image_changes={f"{SITE}/subm/key.py": ["file", f"KEY = {SECRET!r}\n", 0o644],
                        f"{SITE}/subm/__pycache__": ["dir", 0o755],
                        f"{SITE}/subm/__pycache__/key.cpython-312.pyc": ["file", f"\x00{SECRET}", 0o644],
                        "opt/hls/notes.txt": ["file", SECRET, 0o644]})
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, {**GOOD, ".env": SECRET}))
    outputs = PackageStage().run(ctx)
    assert _result(ctx, outputs).status == "built"
    # The findings come in the order of their paths.
    assert _scan(ctx, outputs) == PackageScan(
        secrets=["relay"], forms=["value", "host"], read=["install"],
        findings=[PackageScanFinding(path=f"/{SITE}/subm/__pycache__/key.cpython-312.pyc", secret="relay"),
                  PackageScanFinding(path=f"/{SITE}/subm/key.py", secret="relay")])
    started = _started(ctx)
    assert started[started.index("integrity") + 1:] == ["install-scan", "publish"]
    [role] = [spec for spec in ctx.launcher.launched if spec.name == "install-scan"]
    assert role.owner == "harness" and role.env_file is not None and SECRET not in " ".join(role.argv)
    assert role.argv[1:5] == ("-m", "hls.package.scan", "--names", "relay") and "--tree" not in role.argv


def test_package_scan_json_holds_paths_and_secret_names_never_a_value(tmp_path, tools):
    bin_dir, plan = tools
    _set(plan, image_changes={f"{SITE}/subm/__init__.py": ["file", f"KEY = {SECRET!r}\n", 0o644]})
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD))
    outputs = PackageStage().run(ctx)
    data = ctx.store.read_bytes(outputs["package_scan"])
    assert SECRET.encode() not in data
    # Read back through finalize's own model of the file.
    assert PackageScan.model_validate_json(data).findings == [
        PackageScanFinding(path=f"/{SITE}/subm/__init__.py", secret="relay")]
    assert outputs["package_scan"].endswith(f"/outputs/{PACKAGE_SCAN_FILE}")
    reported = read_outputs_report(ctx.outputs_report("install-scan"))
    assert reported is not None and reported[PACKAGE_SCAN_FILE].sha256 == ctx.output_ref(outputs["package_scan"]).sha256


@pytest.mark.parametrize("plan_changes", [{"dry_run": "fail"}, {}], ids=["invalid", "nothing-checked-out"])
def test_a_package_that_builds_no_image_scans_nothing(tmp_path, tools, plan_changes):
    """Without an image there is nothing of the install's to scan; the bundle is the source_scan
    stage's."""
    bin_dir, plan = tools
    _set(plan, **plan_changes)
    branch = "main" if plan_changes else "dev"
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, {**GOOD, ".env": SECRET}, branch=branch))
    outputs = PackageStage().run(ctx)
    assert _result(ctx, outputs).status == "invalid_artifact" and "package_scan" not in outputs
    assert "install-scan" not in _started(ctx)


def test_a_site_with_no_secret_records_that_nothing_was_read_and_runs_no_scan_role(tmp_path, tools):
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD), registry={})
    outputs = PackageStage().run(ctx)
    assert _scan(ctx, outputs) == PackageScan(secrets=[], forms=["value", "host"], findings=[], read=[])
    assert "install-scan" not in _started(ctx)
    assert _result(ctx, outputs).status == "built"


def test_the_stage_refuses_to_run_without_the_secrets_registry(tmp_path, tools):
    """Without every value of the secrets file the stage could not scan what the image hides from
    finalize, so it refuses to run at all, before any role."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD), registry=None)
    with pytest.raises(StageFault, match="only with the secrets registry") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert ctx.launcher.launched == []
    assert PackageStage.needs_secret_registry is True


def test_a_scan_of_the_install_that_fails_is_a_retryable_fault_and_publishes_no_image(tmp_path, tools):
    bin_dir, _ = tools
    crash = tmp_path / "crash.py"
    crash.write_text("import sys\nsys.exit(1)\n")
    ctx = _ctx(tmp_path, bin_dir, bundle=make_bundle(tmp_path, GOOD),
               launcher=LocalLauncher(replace_modules={"hls.package.scan": crash}))
    with pytest.raises(StageFault, match="harness role install-scan ended exited with status 1") as caught:
        PackageStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert not ctx.store.exists(ctx.output_key("package.json")) and "publish" not in _started(ctx)


@pytest.mark.parametrize("plant", ["missing", "changed", "not-a-record"])
def test_a_build_whose_record_of_its_servers_figures_is_missing_or_changed_is_infra_not_retried(tmp_path, tools,
                                                                                                plant):
    """The figures a build's brief stated are the harness's own record of that build: one that
    is missing, is not what the build's manifest records, or does not parse fails the package as
    infra that a retry would not change, before any role."""
    bin_dir, _ = tools
    ctx = _ctx(tmp_path, bin_dir, bundle=None)
    _agent_build(ctx, make_bundle(tmp_path, GOOD), served=plant != "missing")
    key = ctx.plan.layout.attempt_file(AGENT_BUILD_STAGE, 1, "outputs/served_ucounts.json")
    if plant == "changed":
        ctx.store.write_bytes(key, ctx.store.read_bytes(key).replace(b"128", b"256"))
    elif plant == "not-a-record":
        manifest = ctx.inputs[ctx.plan.layout.manifest(AGENT_BUILD_STAGE)]
        ctx.store.write_bytes(key, b"{}")
        ctx.inputs[ctx.plan.layout.manifest(AGENT_BUILD_STAGE)] = manifest.model_copy(update={"outputs": {
            **manifest.outputs, SERVED_UCOUNTS_OUTPUT: output_ref(ctx.store, key, sha256_bytes(b"{}"))}})
    with pytest.raises(StageFault, match="served_ucounts") as caught:
        PackageStage().run(ctx)
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", False)
    assert ctx.launcher.launched == []


def test_a_named_bundle_carries_no_figures_since_no_brief_stated_any_for_it(tmp_path, tools):
    """An operator's named bundle was not built by this instance's agent, so no brief stated its
    server's figures; its grading computes them from the site's declaration."""
    bin_dir, _ = tools
    bundle = make_bundle(tmp_path, GOOD)
    ctx = _ctx(tmp_path, bin_dir, bundle=bundle)
    result = _result(ctx, PackageStage().run(ctx))
    assert result.status == "built" and result.served_ucounts is None
