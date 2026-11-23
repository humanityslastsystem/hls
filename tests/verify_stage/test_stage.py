from __future__ import annotations

import dataclasses
import gzip
import importlib
import json
import platform
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hls.core.trace import Trace
from hls.pipeline.checkout import CodeState
from hls.pipeline.events import EventLog
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, NOOP_TASK
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.planner import PlanError, StageDecl, build_plan
from hls.pipeline.stage import StageContext, StageFault
from hls.provision.model import Provisioned, RowsRef, ToolRef, WeightsRef, rows_key, tool_key, weights_key
from hls.provision.mounts import copies_gib
from hls.spec.resources import GrantedResources, ResourceShape
from hls.stages.images import KEPT_DISK_GIB
from hls.stages.verify import (VerifyParams, VerifyStage, _Verify, answers_writer, multi_engine_problem,
                               reference_engines, verifier_gpus, writer_instance)
from hls.store.cache import ContentCache, copy_disk_bytes, entry_sha256
from hls.store.digest import sha256_file
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import Manifest, Provenance, Resources, Times, output_ref, read_outputs_report
from hls.store.posix import PosixStore
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.container import ContainerSpec, Mount
from hls.substrate.faults import RoleOutcome
from hls.substrate.nproc import ProcessCeiling
from hls.substrate.sandbox import sandbox_paths, tree_problem
from hls.substrate.secrets import Scrubber
from hls.trace.store import TraceEnd, read_end, write_trace
from hls.core.payloads import Envelope, TextPayload
from hls.drive.record import ArtifactRecord, DriveRecord, DriverRecord, SealProbeRecord
from hls.drive.stream import TraceStreamRecord
from hls.taskref import TaskRef, load_task
from hls.verify.answers import AnswerEntry, StoreAnswers, answer_key, prompt_digest
from hls.verify.role import VerifyRecord
from tests.drive import fake_container
from tests.drive.copy_order import CallOrder
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import settings
from tests.store.entries import populate
from tests.verify_stage.conftest import measured_pins, default_calibration
from tests.verify_stage.fixture import gate_trace, single_trace

#: The verify role's shape; the stage declares the runner's memory and the image's disk beside it.
SHAPE = ResourceShape(gpus=1, gpu_kind="B200", cpus=4, memory_gib=16, time_sec=600)
#: What a served drive's record says of its trace writer: it kept up and did not fail.
KEPT_UP = TraceStreamRecord(backlog_peak=1, lag_max_sec=0.01, space_wait_max_sec=0.0, failure=None)
CPU = ResourceShape(cpus=2, memory_gib=4, time_sec=600)
#: The memory the test plans leave to the runner (``run.runner_memory_gib``'s default).
RUNNER_GIB = 4
IMAGE_KEY = "reference-hf-0123456789abcdef"
REPO_ID, REVISION, ROWS = "org/model", "c" * 40, rows_key("0123456789abcdef")
#: A tool bundle's two variants: the reference mounts the full one, as the driver does.
TOOL_KEYS = {"full": tool_key("tiny", "v1", "full"), "dev": tool_key("tiny", "v1", "dev", "0123456789ab")}
#: The stage's outputs for the gate task's graded trace.
GATE_OUTPUTS = ["replies.0.TokenFaithfulnessVerifier", "replies.1.MetricVerifier", "replies.3.MetricVerifier",
                "verdict", "verify"]


#: The copies of the fixture's data on the node, as the planner declares them (``data_copies``): the
#: weights and the full tool bundle, a MiB each, more than each copy takes.
DATA_COPIES = {f"weights/{weights_key(REPO_ID, REVISION)}": 2**20, f"tools/{TOOL_KEYS['full']}": 2**20}
#: GiB they take on the node.
DATA_GIB = copies_gib(DATA_COPIES)


def _stage(role: ResourceShape) -> ResourceShape:
    """The stage's shape for a role of shape *role*."""
    return role.model_copy(update={"memory_gib": role.memory_gib + RUNNER_GIB,
                                   "disk_gib": role.disk_gib + KEPT_DISK_GIB + DATA_GIB})


@pytest.fixture
def container(tmp_path, monkeypatch) -> Path:
    """The fake container runtime on PATH, before this interpreter's directory; returns the
    directory of its records."""
    bin_dir = tmp_path / "bin"
    fake_container.install(bin_dir)
    venv = Path(sys.executable).parent
    monkeypatch.setenv("PATH", f"{bin_dir}:{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CONTAINER_PATH", f"{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CONTAINER_RECORDS", str(tmp_path / "records"))
    return tmp_path / "records"


def _params(module: str, shape: ResourceShape = SHAPE, data_copies: dict[str, int] = DATA_COPIES) -> dict:
    return {"task": {"module": module}, "reference_image": "reference", "reference_shape": shape.model_dump(),
            "data_copies": data_copies}


def _plan(tmp_path: Path, checkout: Path, module: str, shape: ResourceShape = SHAPE,
          stage_shape: ResourceShape | None = None, data_copies: dict[str, int] = DATA_COPIES) -> Plan:
    stage = stage_shape or _stage(shape).model_copy(
        update={"disk_gib": shape.disk_gib + KEPT_DISK_GIB + copies_gib(data_copies)})
    decl = StageDecl("verify", "verify", stage, _params(module, shape, data_copies))
    return build_plan(decls=[decl], task=NOOP_TASK,
                      pipeline="noop2", layout=InstanceLayout.rollout(NOOP_TASK, "r1"), settings=settings(tmp_path),
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-verify", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=1000.0,
                      measured=measured_pins(checkout, module))


def _manifest(plan: Plan, store: PosixStore, stage: str, outputs: dict[str, str]) -> Manifest:
    """A published manifest of *stage* of this instance, recording each output by name."""
    refs = {name: output_ref(store, key, sha256_file(store.path(key))[0]) for name, key in outputs.items()}
    return Manifest(instance=plan.instance, stage=stage, attempt=1, outputs=refs, inputs={},
                    provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                                          image_digests={}, config_sha256="d" * 64),
                    resources=Resources(declared=SHAPE, granted=GrantedResources(gpus=1, gpu_kind="B200", cpus=4,
                                                                                 memory_gib=16.0, disk_free_gib=100.0)),
                    times=Times(submitted_at=1.0, allocation_started_at=2.0, started_at=2.5, ended_at=3.0))


def _drive_record(outcome: str, end: TraceEnd | None) -> bytes:
    served, ran = outcome == "served", outcome != "invalid_artifact"
    probe = SealProbeRecord(refused_host="hls-seal-probe.invalid", refusal_sec=0.1, ingress_sec=0.1,
                            icmp_sec=0.1, ucounts_sec=0.0, caps_sec=0.0, inbound_tcp=20001, inbound_udp=20002,
                            inbound_addresses=("127.0.0.1", "10.0.0.5"),
                            inbound_sec=0.1, sec=0.5)
    artifact = ArtifactRecord(image_sha256="e" * 64, ending=None if served else "exited",
                              exit_code=None if served else 1, command_started_after_sec=0.5,
                              ready_after_command_sec=1.0 if served else None, engine_cache=None)
    return DriveRecord(
        outcome=outcome, reason=None if served else f"the drive ended {outcome}", seal_probe=probe if ran else None,
        process_ceiling=ProcessCeiling(hard=None, reserve=1024, evaluations=8, evaluations_unknown=None,
                                       uncapped=None, nproc_max=65_536, ceiling=65_536,
                                       floor=8 * 4096) if ran else None,
        artifact=artifact if ran else None,
        driver=DriverRecord(ending="exited", exit_code=0, drive_timeout_sec=600, trace_stream=KEPT_UP)
        if served else None,
        url="http://127.0.0.1:8000" if served else None, drive_started_at=1.0 if served else None,
        drive_ended_at=2.0 if served else None, trace=end).model_dump_json().encode()


def _ctx(tmp_path: Path, checkout: Path, module: str, *, outcome: str = "served", trace: Trace | None = None,
         shape: ResourceShape = SHAPE, drive_json: bytes | None = None, stage_shape: ResourceShape | None = None,
         images: dict | None = None, provisioned_json: bytes | None = None, attempt: int = 1,
         data_copies: dict[str, int] = DATA_COPIES) -> StageContext:
    """Verify attempt *attempt* over a published provision, images stage and a drive that ended
    *outcome*, with *trace* when it served; *drive_json* replaces the drive's record, *images*
    the images stage's record, *provisioned_json* the provision stage's, and *stage_shape* the
    stage's shape, which is *shape*'s otherwise, the plan declaring *data_copies*, the copies of its
    data."""
    plan = _plan(tmp_path, checkout, module, shape, stage_shape, data_copies)
    store = PosixStore(plan.store_root)
    cache = ContentCache(store)
    holder = Holder.here("test")
    image = populate(cache, "images", IMAGE_KEY, lambda d: fake_container.make_image(
        d / "image.sif", {".singularity.d/labels.json": (b"{}", 0o644)}), holder=holder).entry
    key = weights_key(REPO_ID, REVISION)
    weights = populate(cache, "weights", key, lambda d: (d / "config.json").write_text("{}"), holder=holder).entry
    rows = populate(cache, "rows", ROWS, lambda d: (d / "ids.json").write_text("[]"), holder=holder).entry
    tools = []
    for variant, tkey in TOOL_KEYS.items():
        entry = populate(cache, "tools", tkey, lambda d, v=variant: (d / "bundle.txt").write_text(v),
                         holder=holder).entry
        tools.append(ToolRef(name="tiny", identity="v1", variant=variant,
                             dev_digest=None if variant == "full" else "0123456789ab", key=tkey,
                             entry_sha256=entry_sha256(entry), disk_bytes=copy_disk_bytes(entry)))
    provisioned = Provisioned(weights=WeightsRef(repo=REPO_ID, revision=REVISION, key=key,
                                                 entry_sha256=entry_sha256(weights), disk_bytes=copy_disk_bytes(weights)),
                              rows=RowsRef(key=ROWS, entry_sha256=entry_sha256(rows), disk_bytes=copy_disk_bytes(rows), tenants=["t0"]), tools=tools)
    prov_key = plan.layout.attempt_file("provision", 1, "outputs/provisioned.json")
    store.write_bytes(prov_key, provisioned_json if provisioned_json is not None
                      else provisioned.model_dump_json().encode())
    if images is None:
        images = {"reference": {"key": IMAGE_KEY, "entry_sha256": entry_sha256(image), "sif_sha256": _sif_of(image)}}
    images_key = plan.layout.attempt_file("images", 1, "outputs/images.json")
    store.write_bytes(images_key, json.dumps(images).encode())
    # The drive stage's outputs as it publishes them: its record, and the trace's END.json.
    outputs, end = {}, None
    if outcome == "served":
        trace_dir = store.path(plan.layout.attempt_file("deploy_drive", 1, "outputs/trace"))
        end = write_trace(trace if trace is not None else gate_trace(), trace_dir, ending="drained",
                          driver={"class": "t"}).end
        outputs["trace"] = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/END.json")
    drive_key = plan.layout.attempt_file("deploy_drive", 1, "outputs/drive.json")
    store.write_bytes(drive_key, drive_json if drive_json is not None else _drive_record(outcome, end))
    outputs["drive"] = drive_key
    inputs = {plan.layout.manifest("provision"): _manifest(plan, store, "provision", {"provisioned": prov_key}),
              plan.layout.manifest("images"): _manifest(plan, store, "images", {"images": images_key}),
              plan.layout.manifest("deploy_drive"): _manifest(plan, store, "deploy_drive", outputs)}
    events = EventLog(store.path(plan.layout.attempt_file("verify", attempt, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage("verify"), attempt=attempt,
                        launcher=LocalLauncher(gpus=2).use_plan(plan),
                        watch=None, events=events, inputs=inputs, deadline=time.time() + 120, scrubber=Scrubber({}),
                        cache=cache, scratch=tmp_path / "node" / "hls-100", job_id="100", runtime=ApptainerRuntime(),
                        images=ApptainerImages(tmp_path / "node" / "hls-100"))


def _sif_of(entry) -> str:
    return next(f.sha256 for f in entry.manifest.files if f.path == "image.sif")


def _trace_dir(ctx: StageContext) -> Path:
    return ctx.store.path(ctx.plan.layout.attempt_file("deploy_drive", 1, "outputs/trace"))


def _verify_spec(ctx: StageContext):
    [spec] = [s for s in ctx.launcher.launched if s.name == "verify"]
    return spec


def _fault(ctx: StageContext) -> StageFault:
    with pytest.raises(StageFault) as e:
        VerifyStage().run(ctx)
    assert all(h.poll() is not None for h in ctx.launcher.handles), "a role outlived the stage"
    return e.value


@pytest.mark.parametrize("instance, stage", [("warmers/multimodal-asr/20261005T120000Z-ab12cd", "oracle_fill"),
                                             ("runs/t/r1/gradings/g-1", "verify")])
def test_an_answers_writer_names_the_instance_whose_attempt_it_is(instance, stage):
    ctx = SimpleNamespace(plan=SimpleNamespace(instance=instance), stage=SimpleNamespace(name=stage), attempt=2)
    assert writer_instance(answers_writer(ctx)) == instance


@pytest.mark.parametrize("writer", ["g1", "g1+verify", "warmers+t+w1+oracle_fill+x"])
def test_a_writer_no_stage_names_names_no_instance(writer):
    assert writer_instance(writer) is None


def test_a_served_drive_is_verified_in_a_harness_container_on_the_reference_image(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    outputs = VerifyStage().run(ctx)
    assert sorted(outputs) == GATE_OUTPUTS
    assert outputs["verdict"] == ctx.output_key("verdict.json")
    verdict = json.loads(ctx.store.read_bytes(outputs["verdict"]))
    assert verdict["aggregate_passed"] is False and verdict["failure_mode"] == "correctness"
    record = VerifyRecord.model_validate_json(ctx.store.read_bytes(outputs["verify"]))
    assert [(e.role, e.engine) for e in record.engines] == [("reference", "FakeTeacherEngine")]
    assert record.libraries["python"] == platform.python_version()
    spec = _verify_spec(ctx)
    assert (spec.owner, spec.shape) == ("harness", SHAPE)
    # The image was extracted by a harness role and pinned to the digest images.json recorded.
    assert [s.owner for s in ctx.launcher.launched if s.name.startswith("extract-")] == ["harness"]
    sha = _sif_of(ctx.cache.lookup("images", IMAGE_KEY))
    assert ctx.image_digests == {"verify": sha}
    seen = fake_container.records(container)["verify"]
    root, manifest = sandbox_paths(ctx.scratch, sha)
    assert seen["image"] == str(root) and seen["nv"] is True
    assert not {"--writable", "--writable-tmpfs", "--overlay"} & set(spec.argv)
    assert tree_problem(root, manifest, sha)[0] is None  # the role ran from the tree and left it as extracted
    weights = ctx.materialize("weights", weights_key(REPO_ID, REVISION),
                              ctx.cache_entries[f"weights/{weights_key(REPO_ID, REVISION)}"])
    tools = ctx.materialize("tools", TOOL_KEYS["full"], ctx.cache_entries[f"tools/{TOOL_KEYS['full']}"])
    area = ctx.store.path(f"oracle/{ctx.plan.task}")
    assert set(seen["binds"]) >= {f"{ctx.plan.code.checkout}:/opt/hls-src:ro", f"{_trace_dir(ctx)}:/hls/trace:ro",
                                  f"{ctx.outputs_dir}:/hls/out", f"{ctx.outputs_report('verify')}:/hls/outputs.json",
                                  f"{area}:/hls/answers", f"{weights}:/data/weights:ro", f"{tools}:/data/tools/tiny:ro",
                                  f"{ctx.role_dir('verify') / 'measured'}:/hls/measured:ro"}
    # The role loads a copy of the calibration the plan pinned, the default file's bytes.
    copy = ctx.role_dir("verify") / "measured" / "calibration.json"
    assert copy.read_bytes() == default_calibration(checkout, "vstage_gate.task").read_bytes()
    # The reference mounts the weights and the full tool bundle: no rows, no dev bundle.
    assert set(ctx.cache_entries) == {f"images/{IMAGE_KEY}", f"weights/{weights_key(REPO_ID, REVISION)}",
                                      f"tools/{TOOL_KEYS['full']}"}
    # The extraction, a copy of each of the two entries and the role: the most the stage declares
    # one attempt starts, which sizes finalize's time.
    assert len(ctx.launcher.launched) == VerifyStage.max_roles(ctx.params) == 4
    # The answers are keyed by the digest of the image file the role runs, never by its cache key.
    assert seen["command"] == ["python", "-m", "hls.verify.role", "--task-ref", "/hls/task-ref.json",
                               "--trace", "/hls/trace", "--out", "/hls/out", "--outputs-out", "/hls/outputs.json",
                               "--eval-id", "terminal", "--answers", "/hls/answers", "--writer",
                               f"{ctx.plan.instance.replace('/', '+')}+verify+1", "--gpu-kind", "B200",
                               "--reference-image", sha, "--calibration", "/hls/measured/calibration.json"]
    env = seen["env"]
    assert {"PYTHONPATH", "TORCH_CUDA_ARCH_LIST", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HLS_WEIGHTS_DIR"} <= set(env)
    assert (env["PYTHONPATH"], env["TORCH_CUDA_ARCH_LIST"]) == ("/opt/hls-src/src:/opt/hls-src", "10.0")
    # No judge model is provisioned or mounted.
    assert "HLS_WEIGHTS_MAP" not in env and not any(":/data/judge" in b for b in seen["binds"])
    assert ctx.contestant_fault() is None
    assert all(h.poll() is not None for h in ctx.launcher.handles)


def test_every_output_of_the_verify_role_is_recorded_by_its_report(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    outputs = VerifyStage().run(ctx)
    report = read_outputs_report(ctx.outputs_report("verify"))
    assert report is not None
    # Every file the role wrote is an output, and the manifest records each by the role's own
    # report of it, which is the file's digest.
    assert sorted(report) == sorted(p.name for p in ctx.outputs_dir.iterdir())
    assert sorted(ctx.store.path(key).name for key in outputs.values()) == sorted(report)
    for key in outputs.values():
        name = ctx.store.path(key).name
        assert (ctx.output_ref(key).sha256, ctx.output_ref(key).size) == (report[name].sha256, report[name].size)
        assert report[name].sha256 == sha256_file(ctx.store.path(key))[0]


class _WritesWithoutReport(ApptainerRuntime):
    """Runs each container's command as a program that writes the role's two files and reports
    none of them."""

    def argv(self, spec, *, env_file, seal_python=None):
        script = ("import pathlib, sys; out, report = map(pathlib.Path, sys.argv[1:]); "
                  "[(out / f).write_text('{}') for f in ('verdict.json', 'verify.json')]; report.write_text('{}')")
        command = ("python", "-c", script, "/hls/out", "/hls/outputs.json")
        return super().argv(dataclasses.replace(spec, command=command), env_file=env_file, seal_python=seal_python)


def test_a_verify_role_that_reports_none_of_its_outputs_is_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    ctx.launcher.runtime = _WritesWithoutReport()
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "role verify reported no digest of output verdict.json"


def test_the_role_runs_for_the_stages_own_time(tmp_path, checkout, container):
    # reference_shape declares the stage's default time; a site's [resources] time is the role's too.
    longer = _stage(SHAPE).model_copy(update={"time_sec": 900})
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task", stage_shape=longer)
    VerifyStage().run(ctx)
    assert _verify_spec(ctx).shape == SHAPE.model_copy(update={"time_sec": 900})


@pytest.mark.parametrize("outcome", ["invalid_artifact", "crashed", "not_ready"])
def test_a_drive_that_did_not_serve_publishes_a_failing_verdict_and_starts_no_role(tmp_path, checkout, container,
                                                                                   outcome):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task", outcome=outcome)
    outputs = VerifyStage().run(ctx)
    assert sorted(outputs) == ["verdict", "verify"]
    assert json.loads(ctx.store.read_bytes(outputs["verdict"])) == {
        "aggregate_passed": False, "failure_mode": outcome, "per_event": {},
        "diagnostics": {"reason": f"the drive ended {outcome}"}, "per_verifier_results": []}
    record = VerifyRecord.model_validate_json(ctx.store.read_bytes(outputs["verify"]))
    # No verification ran, so no engine and no library was used.
    assert (record.engines, record.libraries) == ([], {})
    # The stage wrote both files itself, so the manifest records the bytes it wrote.
    for key in outputs.values():
        assert ctx.output_ref(key).sha256 == sha256_file(ctx.store.path(key))[0]
    assert ctx.launcher.launched == [] and ctx.cache_entries == {} and ctx.image_digests == {}


def test_a_deterministic_refusal_is_infra_that_is_not_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_drift.task")
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason.startswith("verify refused: CalibrationError: reference stack drifted from calibration")
    assert not ctx.store.path(ctx.output_key("verdict.json")).exists()


def test_a_trace_chunk_that_does_not_match_its_end_fails_the_role_and_is_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    chunk = _trace_dir(ctx) / read_end(_trace_dir(ctx)).chunks[0].name
    chunk.write_bytes(gzip.compress(gzip.decompress(chunk.read_bytes()).replace(b'"B"', b'"C"')))
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "harness role verify ended exited with status 1"
    assert "TraceIncomplete" in (ctx.attempt_dir / "logs" / "verify.err").read_text()
    assert not ctx.store.path(ctx.output_key("verdict.json")).exists()


class _BrokenRuntime(ApptainerRuntime):
    """Extracts as apptainer does, but runs each container with a runtime that fails before
    the container starts, as apptainer does when it cannot set one up."""

    def __init__(self, broken: Path) -> None:
        super().__init__()
        #: The failing program.
        self.broken = broken

    def argv(self, spec, *, env_file, seal_python=None):
        return [str(self.broken), *super().argv(spec, env_file=env_file, seal_python=seal_python)[1:]]


def test_a_container_that_never_started_is_retried(tmp_path, checkout, container):
    broken = tmp_path / "bin" / "broken-apptainer"
    broken.write_text("#!/bin/sh\necho 'FATAL:   could not set up the container' >&2\nexit 255\n")
    broken.chmod(0o755)
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    ctx.launcher.runtime = _BrokenRuntime(broken)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.evidence["verify.ending"] == "runtime_failed"


class _SilentRole(ApptainerRuntime):
    """Runs each container's command as a program that exits 0 and writes nothing."""

    def argv(self, spec, *, env_file, seal_python=None):
        return super().argv(dataclasses.replace(spec, command=("python", "-c", "pass")), env_file=env_file,
                            seal_python=seal_python)


def test_a_role_that_exits_0_without_its_files_is_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    ctx.launcher.runtime = _SilentRole()
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "the verify role exited 0 without writing ['verdict.json', 'verify.json']"


class _CraftedRefusal(ApptainerRuntime):
    """Runs each container's command as a program that logs what a reply could hold, a
    ``refused:`` line among it, and exits 3 without the role's own refusal line."""

    def argv(self, spec, *, env_file, seal_python=None):
        script = ("import sys; print('reply 7: refused: the server will not answer', file=sys.stderr); "
                  "print('refused: the task is refused as it is declared', file=sys.stderr); sys.exit(3)")
        return super().argv(dataclasses.replace(spec, command=("python", "-c", script)), env_file=env_file,
                            seal_python=seal_python)


def test_an_exit_3_without_the_roles_own_refusal_line_is_retried(tmp_path, checkout, container):
    # The role logs what it reads, replies a contestant's server wrote among it: only the role's
    # own refusal line makes exit 3 a refusal.
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    ctx.launcher.runtime = _CraftedRefusal()
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "harness role verify ended exited with status 3"


def test_the_reference_image_and_its_data_are_copied_before_the_role_starts(tmp_path, checkout, container):
    # A copy or an extraction runs alone.
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    order = CallOrder(ctx)
    VerifyStage().run(ctx)
    assert order.staged() == ["node_image", "materialize", "materialize"]
    assert order.while_running() == [] and order.roles_before_staged() == []
    assert [name for kind, name, _ in order.calls if kind == "role"] == ["verify"]


@pytest.mark.parametrize("ending, exit_code, oom_kills", [("memory_limit", 137, 1), ("exited", 137, 1),
                                                          ("exited", 3, 1), ("time_limit", None, 0),
                                                          ("exited", 1, 0), ("time_limit", 3, 0)],
                         ids=["memory-limit", "oom-kill", "refusal-status-after-an-oom", "time-limit",
                              "another-status", "refusal-status-after-the-deadline"])
def test_every_fault_of_the_harness_role_is_infra_and_retried(tmp_path, checkout, container, ending, exit_code,
                                                              oom_kills):
    # The one role is the harness's: no ending of it is ever the contestant's fault, an OOM
    # included, and only a clean exit 3 is a refusal: not another status, nor status 3 from a
    # role the deadline ended. The role's own refusal line is on its stderr, so the ending alone
    # decides.
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    (ctx.attempt_dir / "logs").mkdir(parents=True, exist_ok=True)
    (ctx.attempt_dir / "logs" / "verify.err").write_text("hls-verify-refused: CalibrationError: stale\n")
    outcome = RoleOutcome("verify", "harness", exit_code, ending, oom_kills, own_oom=True, job_oom=False)
    with pytest.raises(StageFault) as e:
        _Verify(ctx).raise_fault(outcome)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)


def test_the_roles_last_refusal_line_is_its_reason_scrubbed(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    ctx.scrubber = Scrubber({"TOKEN": "s3cr3t-value"})
    (ctx.attempt_dir / "logs").mkdir(parents=True, exist_ok=True)
    (ctx.attempt_dir / "logs" / "verify.err").write_text(
        "hls-verify-refused: logged from a reply\nnoise\nhls-verify-refused: CalibrationError: s3cr3t-value stale\n")
    with pytest.raises(StageFault) as e:
        _Verify(ctx).raise_fault(RoleOutcome("verify", "harness", 3, "exited", 0))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason.startswith("verify refused: CalibrationError: ")
    assert "s3cr3t-value" not in e.value.fault.reason and e.value.fault.reason.endswith(" stale")


def test_an_end_file_other_than_the_published_one_is_refused_before_any_role(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    end = _trace_dir(ctx) / "END.json"
    end.write_text(end.read_text() + "\n")
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert "does not match its manifest" in fault.reason
    assert ctx.launcher.launched == []


@pytest.mark.parametrize("images, words", [
    ({}, "does not name image reference's key and entry digest"),
    ({"reference": {"key": IMAGE_KEY}}, "does not name image reference's key and entry digest"),
    ({"reference": {"key": "../escape", "entry_sha256": "0" * 64}}, "does not name image reference's key"),
    ({"reference": {"key": IMAGE_KEY, "entry_sha256": "not-a-digest"}}, "does not name image reference's key"),
    ({"reference": {"key": IMAGE_KEY, "entry_sha256": 7}}, "does not name image reference's key and entry digest"),
    ({"reference": {"key": IMAGE_KEY, "entry_sha256": "0" * 64}}, "0" * 64),
], ids=["no-reference", "no-digest", "bad-key", "bad-digest", "digest-not-text", "other-digest"])
def test_a_reference_image_images_json_does_not_pin_is_refused_before_the_role(tmp_path, checkout, container, images,
                                                                             words):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task", images=images)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert words in fault.reason
    assert ctx.launcher.launched == [] and ctx.image_digests == {}


def test_the_reference_runs_without_gpus_when_no_verifier_has_an_engine(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_metric.task", trace=single_trace(), shape=CPU)
    outputs = VerifyStage().run(ctx)
    assert json.loads(ctx.store.read_bytes(outputs["verdict"]))["aggregate_passed"] is True
    seen = fake_container.records(container)["verify"]
    assert seen["nv"] is False and "TORCH_CUDA_ARCH_LIST" not in seen["env"]


@pytest.mark.parametrize("where, words", [("moved", "the stage was planned without input manifest "),
                                          ("relabelled", "is the deploy_drive manifest of runs/")],
                         ids=["another-instances-key", "another-instance-under-this-key"])
def test_only_this_instances_drive_is_read(tmp_path, checkout, container, where, words):
    ctx = _ctx(tmp_path, checkout, "vstage_metric.task", trace=single_trace(), shape=CPU)
    own = ctx.plan.layout.manifest("deploy_drive")
    other = InstanceLayout.rollout(NOOP_TASK, "r2")
    theirs = ctx.inputs.pop(own).model_copy(update={"instance": other.root})
    ctx.inputs[other.manifest("deploy_drive") if where == "moved" else own] = theirs
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert words in fault.reason
    assert ctx.launcher.launched == []


@pytest.mark.parametrize("stage, output", [("deploy_drive", "drive"), ("deploy_drive", "trace"),
                                           ("images", "images"), ("provision", "provisioned")])
def test_an_input_manifest_without_the_output_the_stage_reads_is_refused(tmp_path, checkout, container, stage,
                                                                         output):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    key = ctx.plan.layout.manifest(stage)
    manifest = ctx.inputs[key]
    ctx.inputs[key] = manifest.model_copy(update={"outputs": {k: v for k, v in manifest.outputs.items() if k != output}})
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert f"published no {output}" in fault.reason
    assert [s for s in ctx.launcher.launched if not s.name.startswith("extract-")] == []


def test_a_trace_output_that_is_not_the_traces_end_record_is_refused(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    key = ctx.plan.layout.manifest("deploy_drive")
    manifest = ctx.inputs[key]
    chunk = ctx.plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    trace = output_ref(ctx.store, chunk, sha256_file(ctx.store.path(chunk))[0])
    ctx.inputs[key] = manifest.model_copy(update={"outputs": {**manifest.outputs, "trace": trace}})
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert "is not its attempt's outputs/trace/END.json" in fault.reason
    assert ctx.launcher.launched == []


@pytest.mark.parametrize("provisioned, words", [
    (b'{"weights": 7, "rows": null, "tools": []}', "outputs/provisioned.json is not a provision record: "),
    (Provisioned(weights=None, rows=None, tools=[]).model_dump_json().encode(),
     "the reference's data cannot be mounted: role reference mounts the task model's weights, which were not "
     "provisioned"),
], ids=["not-a-provision-record", "no-weights"])
def test_a_reference_whose_data_cannot_be_mounted_is_refused_before_the_role(tmp_path, checkout, container,
                                                                            provisioned, words):
    # Another attempt reads the same provision record, so neither is retried.
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task", provisioned_json=provisioned)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert words in fault.reason
    assert [s.name for s in ctx.launcher.launched if not s.name.startswith("extract-")] == []


def test_a_drive_json_that_is_not_a_drive_record_is_refused(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task", outcome="crashed",
               drive_json=json.dumps({"outcome": "exploded", "reason": "x"}).encode())
    fault = _fault(ctx).fault
    assert (fault.retryable, ctx.launcher.launched) == (False, [])
    assert fault.reason.startswith("drive.json is not a drive record")


def _check(module: str, shape: ResourceShape, stage_shape: ResourceShape | None = None) -> None:
    params = VerifyParams(task=TaskRef(module=module), reference_image="reference", reference_shape=shape,
                          data_copies=DATA_COPIES)
    VerifyStage.check(stage_shape or _stage(shape), params, runner_memory_gib=RUNNER_GIB)


def test_a_multi_engine_verifier_with_an_unplaceable_engine_is_refused(checkout):
    # The fake generator and forcer do not place themselves, so each would take every GPU the
    # role sees: however many it has, they cannot share it.
    for gpus in (1, 2, 8):
        with pytest.raises(ValueError, match=r"^vstage_pair.task: PairVerifier loads 2 reference engines at "
                                             "once, and these take every GPU the role sees, so they cannot "
                                             "share it: FakeForcer, FakeGenerator$"):
            _check("vstage_pair.task", SHAPE.model_copy(update={"gpus": gpus}))


def test_a_multi_engine_verifier_is_accepted_on_a_role_with_a_gpu_each(checkout):
    _check("vstage_pair_placed.task", SHAPE.model_copy(update={"gpus": 2}))
    _check("vstage_pair_placed.task", SHAPE.model_copy(update={"gpus": 8}))
    with pytest.raises(ValueError, match=r"^vstage_pair_placed.task: PairVerifier loads 2 reference engines "
                                         "at once, on 2 GPUs in all; reference_shape has 1$"):
        _check("vstage_pair_placed.task", SHAPE)


def test_verifier_gpus_sums_a_verifiers_engines(checkout):
    gpus = {m: verifier_gpus(load_task(TaskRef(module=f"{m}.task")))
            for m in ("vstage_pair_placed", "vstage_pair", "vstage_wide", "vstage_gate", "vstage_metric")}
    # Two engines of one GPU each are live together; one engine over two GPUs needs two; a token
    # gate beside a letter match needs the gate's one; a letter match alone needs none.
    assert gpus == {"vstage_pair_placed": 2, "vstage_pair": 2, "vstage_wide": 2, "vstage_gate": 1,
                    "vstage_metric": 0}


def test_the_shape_must_hold_the_widest_engine(checkout):
    # One engine over two GPUs is one engine: it needs two GPUs, and is not refused as several.
    _check("vstage_wide.task", SHAPE.model_copy(update={"gpus": 2}))
    with pytest.raises(ValueError, match="a reference engine of vstage_wide.task asks for 2 GPUs; reference_shape "
                                         "has 1"):
        _check("vstage_wide.task", SHAPE)
    _check("vstage_gate.task", SHAPE)
    _check("vstage_metric.task", CPU)
    with pytest.raises(ValueError, match="asks for 1 GPUs; reference_shape has 0"):
        _check("vstage_gate.task", CPU)


def test_the_engines_gpu_kind_and_the_stage_shape_are_checked(checkout):
    with pytest.raises(ValueError, match=r"ask for \['B200'\] GPUs; reference_shape has H100"):
        _check("vstage_gate.task", SHAPE.model_copy(update={"gpu_kind": "H100"}))
    stage = _stage(SHAPE)
    for other in (stage.model_copy(update={"cpus": 8}), stage.model_copy(update={"disk_gib": 10}),
                  SHAPE, stage.model_copy(update={"memory_gib": SHAPE.memory_gib}),
                  stage.model_copy(update={"disk_gib": SHAPE.disk_gib}),
                  stage.model_copy(update={"disk_gib": SHAPE.disk_gib + KEPT_DISK_GIB})):
        with pytest.raises(ValueError, match="its resources must be reference_shape's with 4 GiB more memory for the "
                                             "runner, 40 GiB more disk for the image it extracts and 1 GiB for the "
                                             "copies of its data"):
            _check("vstage_gate.task", SHAPE, other)
    # A site may give the stage another time, never other resources.
    _check("vstage_gate.task", SHAPE, stage.model_copy(update={"time_sec": 7200}))


def test_copies_of_the_references_data_the_plan_made_no_room_for_are_refused(tmp_path, checkout, container):
    """The plan declares the copies of the reference's data, which the stage's disk holds
    (``data_copies``, by ``copied_entries``); a copy it did not declare, or one that takes more,
    an entry published again since the plan, is infra not retried, before any role or copy."""
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task", data_copies={})
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == ("the reference's data cannot be mounted: the plan made no room on the node for a copy "
                            f"of {', '.join(DATA_COPIES)}: plan the instance again")
    assert [s.name for s in ctx.launcher.launched if not s.name.startswith("extract-")] == []


def test_a_task_that_does_not_load_is_refused_when_planned(checkout):
    with pytest.raises(ValueError, match=r"task vstage_absent.task does not load: ModuleNotFoundError"):
        _check("vstage_absent.task", SHAPE)
    with pytest.raises(ValueError, match="task tests.verify_stage.fixture does not load: ValueError: .*declares no "
                                         "TASK"):
        _check("tests.verify_stage.fixture", SHAPE)


def test_the_planner_refuses_what_the_verify_role_cannot_run(tmp_path, checkout):
    with pytest.raises(PlanError, match="stage verify: vstage_pair.task: PairVerifier loads 2 reference "
                                        "engines at once, and these take every GPU the role sees"):
        _plan(tmp_path, checkout, "vstage_pair.task", SHAPE.model_copy(update={"gpus": 2}))
    with pytest.raises(PlanError, match="stage verify: .*asks for 2 GPUs; reference_shape has 1"):
        _plan(tmp_path, checkout, "vstage_wide.task")
    with pytest.raises(PlanError, match="stage verify: task vstage_absent.task does not load"):
        _plan(tmp_path, checkout, "vstage_absent.task")


#: The graded module of each published task.
PUBLISHED_MODULES = ("tasks.multitenant_slo.task", "tasks.tab_completion.task", "tasks.multimodal_tts.task",
                     "tasks.multimodal_asr.task", "tasks.multimodal_omni.task",
                     "tasks.deepresearch.task")


@pytest.mark.parametrize("module", PUBLISHED_MODULES)
def test_no_published_task_holds_two_engines_at_once(module):
    # Lifting the refusal changes nothing for them: each verifier loads at most one engine of one
    # GPU, and the role they verify on keeps its one GPU.
    task = importlib.import_module(module).TASK
    assert all(len(engines) <= 1 for _, engines in reference_engines(task))
    assert verifier_gpus(task) == 1
    assert multi_engine_problem(task, 1) is None


# ---------------------------------------------------------------------------
# Reference answers on the store
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attempt", [1, 2])
def test_each_attempt_writes_answers_under_its_own_name_into_the_tasks_area(tmp_path, checkout, container, attempt):
    ctx = _ctx(tmp_path, checkout, "vstage_cached.task", trace=single_trace(), attempt=attempt)
    outputs = VerifyStage().run(ctx)
    record = VerifyRecord.model_validate_json(ctx.store.read_bytes(outputs["verify"]))
    [key] = record.answers_written
    writer = f"{ctx.plan.instance.replace('/', '+')}+verify+{attempt}"
    task = load_task(TaskRef(module="vstage_cached.task"))
    [verifier] = task.effective_tenants()[0].verifiers
    sha = _sif_of(ctx.cache.lookup("images", IMAGE_KEY))
    assert key == f"{answer_key(verifier.reference.fingerprint(task), 'B200', sha)}/{writer}.000001.json"
    assert ctx.store.exists(f"oracle/{ctx.plan.task}/{key}")
    # The area lies outside the instance, so no output of the stage is an answers file.
    assert all(k.startswith(f"{ctx.plan.instance}/") for k in outputs.values())


CACHED_WEIGHTS = "vstage_cachedweights.task"


def _bank(ctx: StageContext, events) -> str:
    """The fake generator's answer to each of *events*' prompts, banked in the task's answers
    area under the key the verify role reads, as the oracle fill banks them; the file's key."""
    task = load_task(TaskRef(module=CACHED_WEIGHTS))
    [verifier] = task.effective_tenants()[0].verifiers
    sha = _sif_of(ctx.cache.lookup("images", IMAGE_KEY))
    answers = StoreAnswers(PosixStore(ctx.store.path(f"oracle/{ctx.plan.task}")), gpu_kind="B200", image=sha,
                           writer="fill")
    reply = Envelope(request_id="x", contents=[TextPayload(text="B")]).model_dump(mode="json")
    answers.bank(verifier.reference.fingerprint(task),
                 [AnswerEntry(request_id=ev.request_id, digest=prompt_digest(ev.input), answer=reply) for ev in events])
    [key] = answers.written
    return key


def _weights_mounted(ctx: StageContext, seen: dict) -> bool:
    """Whether the stage copied the task model's weights and the role mounts them."""
    copied = f"weights/{weights_key(REPO_ID, REVISION)}" in ctx.cache_entries
    bound = any(b.split(":")[1] == "/data/weights" for b in seen["binds"])
    assert copied == bound == ("HLS_WEIGHTS_DIR" in seen["env"])
    return copied


def test_a_verify_whose_every_answer_is_in_the_store_copies_no_weights_and_loads_no_engine(tmp_path, checkout,
                                                                                          container):
    trace = single_trace()
    ctx = _ctx(tmp_path, checkout, CACHED_WEIGHTS, trace=trace)
    filled = _bank(ctx, trace.events)
    outputs = VerifyStage().run(ctx)
    assert json.loads(ctx.store.read_bytes(outputs["verdict"]))["aggregate_passed"] is True
    record = VerifyRecord.model_validate_json(ctx.store.read_bytes(outputs["verify"]))
    assert record.engines == [] and record.answers_written == {}
    assert list(record.answers_read) == [filled]
    assert not _weights_mounted(ctx, fake_container.records(container)["verify"])
    # The extraction, the tool bundle's copy and the role: no weights copy.
    assert [s.name.split("-")[0] for s in ctx.launcher.launched] == ["extract", "copy", "verify"]


def test_a_partly_answered_verify_copies_the_weights_and_loads_the_engine(tmp_path, checkout, container):
    trace = single_trace()
    ctx = _ctx(tmp_path, checkout, CACHED_WEIGHTS, trace=trace)
    _bank(ctx, trace.events[:1])
    outputs = VerifyStage().run(ctx)
    assert json.loads(ctx.store.read_bytes(outputs["verdict"]))["aggregate_passed"] is True
    record = VerifyRecord.model_validate_json(ctx.store.read_bytes(outputs["verify"]))
    # The engine loaded, which it can only do with the weights mounted, and answered the miss.
    assert [(e.role, e.engine) for e in record.engines] == [("reference", "WeightsGenerator")]
    assert len(record.answers_written) == 1
    assert _weights_mounted(ctx, fake_container.records(container)["verify"])


def test_an_empty_area_copies_the_weights(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, CACHED_WEIGHTS, trace=single_trace())
    VerifyStage().run(ctx)
    assert _weights_mounted(ctx, fake_container.records(container)["verify"])


def test_a_reference_that_does_not_cache_copies_the_weights_without_reading_the_trace(tmp_path, checkout, container,
                                                                                     monkeypatch):
    ctx = _ctx(tmp_path, checkout, "vstage_gate.task")
    monkeypatch.setattr("hls.stages.verify.iter_events", lambda *_: pytest.fail("the trace was read"))
    VerifyStage().run(ctx)
    assert _weights_mounted(ctx, fake_container.records(container)["verify"])


def test_a_task_whose_verifiers_have_no_engine_copies_no_weights(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_metric.task", trace=single_trace(), shape=CPU)
    VerifyStage().run(ctx)
    assert not _weights_mounted(ctx, fake_container.records(container)["verify"])


def test_answers_that_vanish_after_the_decision_fail_the_engines_load_and_the_attempt_is_retried(tmp_path, checkout,
                                                                                              container):
    """The area only grows (writers create files and never change or remove one), so every answer
    the stage found is there for the role. Only a storage fault can hide one in between; the role
    then deploys its engine, which finds no weights and fails to load: infra, retried, as any
    engine that fails to load is. The retry decides again from what it reads."""
    trace = single_trace()
    ctx = _ctx(tmp_path, checkout, CACHED_WEIGHTS, trace=trace)
    filled = ctx.store.path(f"oracle/{ctx.plan.task}/{_bank(ctx, trace.events)}")
    start = ctx.start_container

    def hide_then_start(name, *args, **kwargs):
        if name == "verify":
            filled.unlink()
        return start(name, *args, **kwargs)

    ctx.start_container = hide_then_start
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "WeightsGenerator failed to load" in (ctx.attempt_dir / "logs" / "verify.err").read_text()


def test_a_caching_reference_on_a_multi_turn_workload_is_refused_when_planned(tmp_path, checkout):
    """From turn two a prompt carries the server's earlier replies, so a caching reference would
    keep answers conditioned on a contestant's text in the answers area, which no scan reads."""
    words = (r"vstage_cachedchat.task: tenant t0's workload runs 2.5 turns per session, so its prompts carry the "
             r"server's replies from turn two, and AskVerifier caches the answers of its reference "
             r"GeneratorReference \(cache=True\)")
    with pytest.raises(ValueError, match=words):
        _check("vstage_cachedchat.task", SHAPE)
    with pytest.raises(PlanError, match="stage verify: " + words):
        _plan(tmp_path, checkout, "vstage_cachedchat.task")
    _check("vstage_cached.task", SHAPE)


def test_a_caching_reference_on_an_agentic_task_is_refused_when_planned(tmp_path, checkout):
    """An agentic task's driver runs one trajectory per row, each call carrying the server's
    earlier replies, though its workload declares one turn per session."""
    task = load_task(TaskRef(module="vstage_cachedagent.task"))
    assert task.agent is not None and task.effective_tenants()[0].workload.mean_turns_per_session == 1
    words = (r"vstage_cachedagent.task: the task declares an agent, so tenant t0's trajectories send calls that "
             r"carry the server's earlier replies whatever its workload declares, and AskVerifier caches the "
             r"answers of its reference GeneratorReference \(cache=True\)")
    with pytest.raises(ValueError, match=words):
        _check("vstage_cachedagent.task", SHAPE)
    with pytest.raises(PlanError, match="stage verify: " + words):
        _plan(tmp_path, checkout, "vstage_cachedagent.task")


def test_a_role_without_gpus_keys_its_answers_by_cpu(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_metric.task", trace=single_trace(), shape=CPU)
    VerifyStage().run(ctx)
    command = fake_container.records(container)["verify"]["command"]
    assert command[command.index("--gpu-kind") + 1] == "cpu"


def test_a_contestant_container_may_not_mount_the_answers_area(tmp_path, checkout, container):
    """The answers are the reference's answers to the withheld test split: only the verify role,
    a harness container, mounts them. A contestant container mounts nothing in the store, not
    even read-only."""
    ctx = _ctx(tmp_path, checkout, "vstage_metric.task", trace=single_trace(), shape=CPU)
    area = ctx.store.path(f"oracle/{ctx.plan.task}")
    area.mkdir(parents=True)
    spec = ContainerSpec(image=_Verify(ctx).image(), command=("true",), workdir=ctx.role_dir("srv") / "wd",
                         mounts=(Mount(area, "/hls/answers"),))
    with pytest.raises(ValueError, match=re.escape(f"mount /hls/answers source {area} is in the store")):
        ctx.start_container("srv", spec, CPU.model_copy(update={"disk_gib": 1}), owner="contestant")
    assert [s.name for s in ctx.launcher.launched if not s.name.startswith("extract-")] == []


def test_the_refusal_search_reads_as_far_back_as_the_score_stage(tmp_path, monkeypatch):
    """Both stages search the end of their role's stderr for its refusal over one bound, the score
    stage's (``hls.stages.score.REASON_SEARCH_BYTES``)."""
    from hls.stages import score, verify

    err = tmp_path / "verify.err"
    err.write_text(f"{verify.REFUSED_PREFIX}early\n" + "x" * 100 + "\n")
    assert verify._reason(err) == "early"
    monkeypatch.setattr(score, "REASON_SEARCH_BYTES", 50)
    assert verify._reason(err) is None
