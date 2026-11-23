"""The oracle_fill stage on the local launcher, with the deploy_drive tests' fake container: the
fill role runs as a program inside a fake container on the pinned reference image, with the
test rows, the reference's data and the task's answers area mounted."""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import pytest

from hls.measured import DEV_REFERENCE_FILE, pinned_key, store_key
from hls.taskref import TaskRef, blob_id, load_task
from hls.verify.dev_reference import read_dev_reference
from hls.pipeline.checkout import CodeState
from hls.pipeline.events import EventLog
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, NOOP_TASK
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.planner import PlanError, StageDecl, build_plan
from hls.pipeline.stage import StageContext, StageFault
from hls.provision.model import Provisioned, RowsRef, ToolRef, WeightsRef, rows_key, tool_key, weights_key
from hls.provision.mounts import copies_gib
from hls.spec.resources import ResourceShape
from hls.stages.images import KEPT_DISK_GIB
from hls.stages.oracle_fill import OracleFillParams, OracleFillStage, _Fill
from hls.store.cache import ContentCache, copy_disk_bytes, entry_sha256
from hls.store.digest import sha256_file
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_outputs_report
from hls.store.posix import PosixStore
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.faults import RoleOutcome
from hls.substrate.sandbox import sandbox_paths
from hls.substrate.secrets import Scrubber
from hls.verify.answers import answer_key, questions_name
from hls.verify.fill import FillRecord, question_scopes, question_shape
from tests.drive import fake_container
from tests.drive.copy_order import CallOrder
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import settings
from tests.store.entries import populate
from tests.verify_stage.test_stage import _manifest, _sif_of

#: The fill role's shape; the stage declares the runner's memory and the image's disk beside it.
SHAPE = ResourceShape(gpus=1, gpu_kind="B200", cpus=4, memory_gib=16, time_sec=600)
#: The memory the test plans leave to the runner (``run.runner_memory_gib``'s default).
RUNNER_GIB = 4
IMAGE_KEY = "reference-hf-0123456789abcdef"
REPO_ID, REVISION = "org/model", "c" * 40
ROWS = rows_key("0123456789abcdef")
TEST_ROWS = [{"q": "1"}, {"q": "2"}, {"q": "3"}]
#: A tool bundle's two variants: the fill mounts the full one, as the reference does.
TOOL_KEYS = {"full": tool_key("tiny", "v1", "full"), "dev": tool_key("tiny", "v1", "dev", "0123456789ab")}


#: The copies of the fixture's data on the node, as the planner declares them (``data_copies``): the
#: weights, the rows and the full tool bundle, a MiB each, more than each copy takes.
DATA_COPIES = {f"weights/{weights_key(REPO_ID, REVISION)}": 2**20, f"rows/{ROWS}": 2**20,
               f"tools/{TOOL_KEYS['full']}": 2**20}
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


def _params(module: str, shape: ResourceShape = SHAPE, chunk: int = 2,
            data_copies: dict[str, int] = DATA_COPIES, dev_reference: bool = False) -> dict:
    return {"task": {"module": module}, "reference_image": "reference", "reference_shape": shape.model_dump(),
            "chunk": chunk, "data_copies": data_copies, "dev_reference": dev_reference}


def _plan(tmp_path: Path, checkout: Path, module: str, shape: ResourceShape = SHAPE,
          stage_shape: ResourceShape | None = None, data_copies: dict[str, int] = DATA_COPIES,
          dev_reference: bool = False) -> Plan:
    stage = stage_shape or _stage(shape).model_copy(
        update={"disk_gib": shape.disk_gib + KEPT_DISK_GIB + copies_gib(data_copies)})
    decl = StageDecl("oracle_fill", "oracle_fill", stage, _params(module, shape, data_copies=data_copies,
                                                                  dev_reference=dev_reference))
    return build_plan(decls=[decl], task=NOOP_TASK, pipeline="oracle_fill",
                      layout=InstanceLayout.warmer(NOOP_TASK, "w1"), settings=settings(tmp_path),
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-oracle-fill", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=1000.0)


def _ctx(tmp_path: Path, checkout: Path, module: str, *, rows: list[dict] = TEST_ROWS,
         stage_shape: ResourceShape | None = None, images: dict | None = None,
         provisioned_json: bytes | None = None, data_copies: dict[str, int] = DATA_COPIES,
         dev_reference: bool = False) -> StageContext:
    """The fill attempt's context over a published provision and images stage: the reference
    image, the weights, the rows and a tool bundle in the cache. *images* replaces the images
    stage's record, *provisioned_json* the provision stage's, and *stage_shape* the stage's
    shape; the plan declares *data_copies*, the copies of its data."""
    plan = _plan(tmp_path, checkout, module, stage_shape=stage_shape, data_copies=data_copies,
                 dev_reference=dev_reference)
    store = PosixStore(plan.store_root)
    cache = ContentCache(store)
    holder = Holder.here("test")
    image = populate(cache, "images", IMAGE_KEY, lambda d: fake_container.make_image(
        d / "image.sif", {".singularity.d/labels.json": (b"{}", 0o644)}), holder=holder).entry
    key = weights_key(REPO_ID, REVISION)
    weights = populate(cache, "weights", key, lambda d: (d / "config.json").write_text("{}"), holder=holder).entry

    def fill_rows(d: Path) -> None:
        for split, data in (("dev", [{"q": "0"}]), ("test", rows)):
            (d / "tenants" / "t0" / split).mkdir(parents=True)
            (d / "tenants" / "t0" / split / "rows.json").write_text(json.dumps(data))

    entry = populate(cache, "rows", ROWS, fill_rows, holder=holder).entry
    tools = []
    for variant, tkey in TOOL_KEYS.items():
        bundle = populate(cache, "tools", tkey, lambda d, v=variant: (d / "bundle.txt").write_text(v),
                          holder=holder).entry
        tools.append(ToolRef(name="tiny", identity="v1", variant=variant,
                             dev_digest=None if variant == "full" else "0123456789ab", key=tkey,
                             entry_sha256=entry_sha256(bundle), disk_bytes=copy_disk_bytes(bundle)))
    provisioned = Provisioned(weights=WeightsRef(repo=REPO_ID, revision=REVISION, key=key,
                                                 entry_sha256=entry_sha256(weights), disk_bytes=copy_disk_bytes(weights)),
                              rows=RowsRef(key=ROWS, entry_sha256=entry_sha256(entry), disk_bytes=copy_disk_bytes(entry), tenants=["t0"]), tools=tools)
    prov_key = plan.layout.attempt_file("provision", 1, "outputs/provisioned.json")
    store.write_bytes(prov_key, provisioned_json if provisioned_json is not None
                      else provisioned.model_dump_json().encode())
    if images is None:
        images = {"reference": {"key": IMAGE_KEY, "entry_sha256": entry_sha256(image), "sif_sha256": _sif_of(image)}}
    images_key = plan.layout.attempt_file("images", 1, "outputs/images.json")
    store.write_bytes(images_key, json.dumps(images).encode())
    inputs = {plan.layout.manifest("provision"): _manifest(plan, store, "provision", {"provisioned": prov_key}),
              plan.layout.manifest("images"): _manifest(plan, store, "images", {"images": images_key})}
    events = EventLog(store.path(plan.layout.attempt_file("oracle_fill", 1, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage("oracle_fill"), attempt=1,
                        launcher=LocalLauncher(gpus=2).use_plan(plan), watch=None, events=events, inputs=inputs,
                        deadline=time.time() + 120, scrubber=Scrubber({}), cache=cache,
                        scratch=tmp_path / "node" / "hls-100", job_id="100", runtime=ApptainerRuntime(),
                        images=ApptainerImages(tmp_path / "node" / "hls-100"))


def _fill_spec(ctx: StageContext):
    [spec] = [s for s in ctx.launcher.launched if s.name == "oracle-fill"]
    return spec


def _fault(ctx: StageContext) -> StageFault:
    with pytest.raises(StageFault) as e:
        OracleFillStage().run(ctx)
    assert all(h.poll() is not None for h in ctx.launcher.handles), "a role outlived the stage"
    return e.value


def test_a_fill_runs_on_the_pinned_reference_image_with_the_test_rows_and_the_answers_area(tmp_path, checkout,
                                                                                          container):
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    outputs = OracleFillStage().run(ctx)
    assert outputs == {"fill": ctx.output_key("fill.json")}
    record = FillRecord.model_validate_json(ctx.store.read_bytes(outputs["fill"]))
    [ref] = record.references
    assert (ref.prompts, ref.answered, ref.served) == (3, 3, 0) and len(record.answers_written) == 2
    spec = _fill_spec(ctx)
    assert (spec.owner, spec.shape) == ("harness", SHAPE)
    # The image was extracted by a harness role and pinned to the digest images.json recorded.
    assert [s.owner for s in ctx.launcher.launched if s.name.startswith("extract-")] == ["harness"]
    sha = _sif_of(ctx.cache.lookup("images", IMAGE_KEY))
    assert ctx.image_digests == {"oracle-fill": sha}
    seen = fake_container.records(container)["oracle-fill"]
    root, _ = sandbox_paths(ctx.scratch, sha)
    assert seen["image"] == str(root) and seen["nv"] is True
    rows = ctx.materialize("rows", ROWS, ctx.cache_entries[f"rows/{ROWS}"])
    key = weights_key(REPO_ID, REVISION)
    weights = ctx.materialize("weights", key, ctx.cache_entries[f"weights/{key}"])
    tools = ctx.materialize("tools", TOOL_KEYS["full"], ctx.cache_entries[f"tools/{TOOL_KEYS['full']}"])
    area = ctx.store.path(f"oracle/{ctx.plan.task}")
    assert set(seen["binds"]) >= {f"{ctx.plan.code.checkout}:/opt/hls-src:ro", f"{weights}:/data/weights:ro",
                                  f"{rows}/tenants/t0/test:/data/tenants/t0/test:ro",
                                  f"{rows}/tenants/t0/dev:/data/tenants/t0/dev:ro",
                                  f"{tools}:/data/tools/tiny:ro", f"{area}:/hls/answers",
                                  f"{ctx.outputs_dir}:/hls/out",
                                  f"{ctx.outputs_report('oracle-fill')}:/hls/outputs.json"}
    # The weights, the rows and the full tool bundle are copied; never the dev bundle.
    assert set(ctx.cache_entries) == {f"images/{IMAGE_KEY}", f"weights/{key}", f"rows/{ROWS}",
                                      f"tools/{TOOL_KEYS['full']}"}
    # The answers are keyed by the digest of the image file the role runs, as a grading's verify keys them.
    writer = f"{ctx.plan.instance.replace('/', '+')}+oracle_fill+1"
    assert seen["command"] == ["python", "-m", "hls.verify.fill", "--task-ref", "/hls/task-ref.json",
                               "--rows", "/data", "--rows-key", ROWS, "--answers", "/hls/answers", "--writer", writer,
                               "--gpu-kind", "B200", "--reference-image", sha, "--out", "/hls/out/fill.json",
                               "--outputs-out", "/hls/outputs.json", "--chunk", "2"]
    env = seen["env"]
    assert (env["PYTHONPATH"], env["TORCH_CUDA_ARCH_LIST"], env["HF_HUB_OFFLINE"]) == (
        "/opt/hls-src/src:/opt/hls-src", "10.0", "1")
    assert all(k.split("/")[1].startswith(f"{writer}.") for k in record.answers_written)
    # The question set, written last, names the rows entry the provision record names.
    [questions] = record.questions
    assert questions.split("/")[1].startswith(f"questions-{ROWS}-")
    assert all(h.poll() is not None for h in ctx.launcher.handles)


def test_another_question_set_under_the_fills_name_is_a_refusal_that_is_not_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    task = load_task(TaskRef(module="vstage_fill.task"))
    [(fingerprint, scope)] = question_scopes(task).items()
    sha = _sif_of(ctx.cache.lookup("images", IMAGE_KEY))
    name = f"{answer_key(fingerprint, 'B200', sha)}/{questions_name(ROWS, question_shape(task, scope))}"
    ctx.store.write_bytes(f"oracle/{ctx.plan.task}/{name}", b"{}")
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason.startswith("oracle fill refused: QuestionSetConflict: ")
    assert f"/{name} holds another question set" in fault.reason
    assert not ctx.store.exists(ctx.output_key("fill.json"))


def test_a_provision_record_with_no_rows_is_infra_that_is_not_retried(tmp_path, checkout, container):
    # Nothing names the test rows the question set would stand for.
    key = weights_key(REPO_ID, REVISION)
    provisioned = Provisioned(weights=WeightsRef(repo=REPO_ID, revision=REVISION, key=key, entry_sha256="0" * 64,
                                                 disk_bytes=4096),
                              rows=None, tools=[])
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task", provisioned_json=provisioned.model_dump_json().encode())
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason.endswith("names no rows entry, so there are no test rows to fill")
    assert not any(s.name == "oracle-fill" for s in ctx.launcher.launched)


def test_a_fill_that_produces_the_dev_reference_publishes_it_and_writes_the_tasks_measured_file(
        tmp_path, checkout, container):
    """With ``dev_reference``, the role answers the dev rows too; the stage publishes the record
    the role reports and writes it into the store's measured area, where runs find it."""
    ctx = _ctx(tmp_path, checkout, "vstage_devref.task", dev_reference=True)
    outputs = OracleFillStage().run(ctx)
    assert sorted(outputs) == ["dev_reference", "fill"]
    data = ctx.store.read_bytes(outputs["dev_reference"])
    assert ctx.store.read_bytes(store_key(ctx.plan.task, DEV_REFERENCE_FILE)) == data
    assert ctx.store.read_bytes(pinned_key(ctx.plan.task, DEV_REFERENCE_FILE, blob_id(data))) == data
    record, _ = read_dev_reference(ctx.store.path(outputs["dev_reference"]), load_task(TaskRef(module="vstage_devref.task")))
    rows = json.loads(record.tenants["t0"].text)["rows"]
    assert len(rows) == 1 and all(row == {"transcript": "B", "accuracy": 1.0} for row in rows.values())


def test_the_fill_roles_record_is_recorded_by_its_report(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    outputs = OracleFillStage().run(ctx)
    report = read_outputs_report(ctx.outputs_report("oracle-fill"))
    assert report is not None and sorted(report) == ["fill.json"]
    # The manifest records the output by the role's own report of it, which is the file's digest.
    ref = ctx.output_ref(outputs["fill"])
    assert (ref.sha256, ref.size) == (report["fill.json"].sha256, report["fill.json"].size)
    assert report["fill.json"].sha256 == sha256_file(ctx.store.path(outputs["fill"]))[0]


class _WritesWithoutReport(ApptainerRuntime):
    """Runs each container's command as a program that writes the role's record and reports
    nothing."""

    def argv(self, spec, *, env_file, seal_python=None):
        script = ("import pathlib, sys; out, report = map(pathlib.Path, sys.argv[1:]); "
                  "(out / 'fill.json').write_text('{}'); report.write_text('{}')")
        command = ("python", "-c", script, "/hls/out", "/hls/outputs.json")
        return super().argv(dataclasses.replace(spec, command=command), env_file=env_file, seal_python=seal_python)


def test_a_fill_role_that_does_not_report_its_record_is_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    ctx.launcher.runtime = _WritesWithoutReport()
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "role oracle-fill reported no digest of output fill.json"


def test_the_reference_image_and_its_data_are_copied_before_the_role_starts(tmp_path, checkout, container):
    # A copy or an extraction runs alone: the image, the weights, the test
    # rows and the full tool bundle.
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    order = CallOrder(ctx)
    OracleFillStage().run(ctx)
    assert order.staged() == ["node_image", "materialize", "materialize", "materialize"]
    assert order.while_running() == [] and order.roles_before_staged() == []
    assert [name for kind, name, _ in order.calls if kind == "role"] == ["oracle-fill"]


def test_the_role_runs_for_the_stages_own_time(tmp_path, checkout, container):
    longer = _stage(SHAPE).model_copy(update={"time_sec": 900})
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task", stage_shape=longer)
    OracleFillStage().run(ctx)
    assert _fill_spec(ctx).shape == SHAPE.model_copy(update={"time_sec": 900})


def test_a_deterministic_refusal_is_infra_that_is_not_retried(tmp_path, checkout, container):
    rows = [{"q": "1", "ctx": "c", "qs": ["a", "b"], "as": ["x", "y"]}]
    ctx = _ctx(tmp_path, checkout, "vstage_multi.task", rows=rows)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason.startswith("oracle fill refused: FillRefused: t0/1 has 2 turns")
    assert not ctx.store.exists(ctx.output_key("fill.json"))


def test_the_roles_last_refusal_line_is_its_reason_scrubbed(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    ctx.scrubber = Scrubber({"TOKEN": "s3cr3t-value"})
    (ctx.attempt_dir / "logs").mkdir(parents=True, exist_ok=True)
    (ctx.attempt_dir / "logs" / "oracle-fill.err").write_text(
        "hls-verify-refused: an earlier line\nnoise\nhls-verify-refused: FillRefused: s3cr3t-value turns\n")
    with pytest.raises(StageFault) as e:
        _Fill(ctx).raise_fault(RoleOutcome("oracle-fill", "harness", 3, "exited", 0))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason.startswith("oracle fill refused: FillRefused: ")
    assert "s3cr3t-value" not in e.value.fault.reason and e.value.fault.reason.endswith(" turns")


def test_any_other_failure_of_the_role_is_retried_and_keeps_its_finished_chunks(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_flaky.task")  # dies on its second chunk
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "harness role oracle-fill ended exited with status 1"
    assert "the flaky generator died" in (ctx.attempt_dir / "logs" / "oracle-fill.err").read_text()
    assert not ctx.store.exists(ctx.output_key("fill.json"))
    # The chunk it finished is kept for the retry.
    assert len(list(ctx.store.path(f"oracle/{ctx.plan.task}").rglob("*.json"))) == 1


def test_answers_that_cannot_be_written_end_the_role_as_infra_that_is_retried(tmp_path, checkout, container):
    # The area another user made without write access to it, a full quota: the retry asks only
    # for the rows still missing.
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    area = ctx.store.path(f"oracle/{ctx.plan.task}")
    area.mkdir(parents=True)
    area.chmod(0o555)
    try:
        fault = _fault(ctx).fault
    finally:
        area.chmod(0o755)
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "harness role oracle-fill ended exited with status 1"
    assert "could not write reference answers" in (ctx.attempt_dir / "logs" / "oracle-fill.err").read_text()
    assert not ctx.store.exists(ctx.output_key("fill.json"))


class _SilentRole(ApptainerRuntime):
    """Runs each container's command as a program that exits 0 and writes nothing."""

    def argv(self, spec, *, env_file, seal_python=None):
        return super().argv(dataclasses.replace(spec, command=("python", "-c", "pass")), env_file=env_file,
                            seal_python=seal_python)


def test_a_role_that_exits_0_without_its_record_is_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    ctx.launcher.runtime = _SilentRole()
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "the fill role exited 0 without writing fill.json"


@pytest.mark.parametrize("ending, exit_code, oom_kills", [("memory_limit", 137, 1), ("exited", 137, 1),
                                                          ("exited", 3, 1), ("time_limit", None, 0),
                                                          ("exited", 1, 0), ("time_limit", 3, 0)],
                         ids=["memory-limit", "oom-kill", "refusal-status-after-an-oom", "time-limit",
                              "another-status", "refusal-status-after-the-deadline"])
def test_every_fault_of_the_harness_role_is_infra_and_retried(tmp_path, checkout, container, ending, exit_code,
                                                              oom_kills):
    # The one role is the harness's: no ending of it is ever the contestant's fault, an OOM
    # included, and only a clean exit 3 is a refusal, whatever its stderr says.
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    (ctx.attempt_dir / "logs").mkdir(parents=True, exist_ok=True)
    (ctx.attempt_dir / "logs" / "oracle-fill.err").write_text("hls-verify-refused: FillRefused: no\n")
    outcome = RoleOutcome("oracle-fill", "harness", exit_code, ending, oom_kills, own_oom=True, job_oom=False)
    with pytest.raises(StageFault) as e:
        _Fill(ctx).raise_fault(outcome)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)


@pytest.mark.parametrize("images, words", [
    ({}, "does not name image reference's key and entry digest"),
    ({"reference": {"key": IMAGE_KEY}}, "does not name image reference's key and entry digest"),
    ({"reference": {"key": "../escape", "entry_sha256": "0" * 64}}, "does not name image reference's key"),
    ({"reference": {"key": IMAGE_KEY, "entry_sha256": "not-a-digest"}}, "does not name image reference's key"),
    ({"reference": {"key": IMAGE_KEY, "entry_sha256": "0" * 64}}, "0" * 64),
], ids=["no-reference", "no-digest", "bad-key", "bad-digest", "other-digest"])
def test_a_reference_image_images_json_does_not_pin_is_refused_before_the_role(tmp_path, checkout, container, images,
                                                                             words):
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task", images=images)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert words in fault.reason
    assert [s for s in ctx.launcher.launched if not s.name.startswith("extract-")] == []
    assert ctx.image_digests == {}


@pytest.mark.parametrize("stage", ["images", "provision"])
def test_a_fill_without_its_input_manifest_is_refused_before_any_role(tmp_path, checkout, container, stage):
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    key = ctx.plan.layout.manifest(stage)
    del ctx.inputs[key]
    fault = _fault(ctx).fault
    assert (fault.retryable, ctx.launcher.launched) == (False, [])
    assert fault.reason == f"stage oracle_fill reads {key}, which is not among its inputs"


def test_an_images_json_other_than_the_published_one_is_refused_before_any_role(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task")
    images = ctx.store.path(ctx.plan.layout.attempt_file("images", 1, "outputs/images.json"))
    images.write_text(images.read_text() + "\n")
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert "does not match its manifest" in fault.reason
    assert ctx.launcher.launched == []


@pytest.mark.parametrize("provisioned, words", [
    (b'{"weights": 7, "rows": null, "tools": []}', "outputs/provisioned.json is not a provision record: "),
    (Provisioned(weights=None, rows=RowsRef(key=ROWS, entry_sha256="0" * 64, tenants=["t0"], disk_bytes=4096),
                 tools=[]).model_dump_json().encode(),
     "the fill's data cannot be mounted: role oracle mounts the task model's weights, which were not provisioned"),
], ids=["not-a-provision-record", "no-weights"])
def test_a_fill_whose_data_cannot_be_mounted_is_refused_before_the_role(tmp_path, checkout, container, provisioned,
                                                                        words):
    # Another attempt reads the same provision record, so neither is retried.
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task", provisioned_json=provisioned)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert words in fault.reason
    assert [s.name for s in ctx.launcher.launched if not s.name.startswith("extract-")] == []


def _check(module: str, shape: ResourceShape = SHAPE, stage_shape: ResourceShape | None = None) -> None:
    params = OracleFillParams(task=TaskRef(module=module), reference_image="reference", reference_shape=shape,
                              data_copies=DATA_COPIES)
    OracleFillStage.check(stage_shape or _stage(shape), params, runner_memory_gib=RUNNER_GIB)


def test_copies_of_the_fills_data_the_plan_made_no_room_for_are_refused(tmp_path, checkout, container):
    """As the verify stage's: the plan declares the copies of the fill's data, which its disk
    holds (``data_copies``), and a copy it did not declare is infra not retried, before any role
    or copy."""
    ctx = _ctx(tmp_path, checkout, "vstage_fill.task", data_copies={})
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == ("the fill's data cannot be mounted: the plan made no room on the node for a copy of "
                            f"{', '.join(DATA_COPIES)}: plan the instance again")
    assert [s.name for s in ctx.launcher.launched if not s.name.startswith("extract-")] == []


def test_check_refuses_a_task_with_nothing_to_fill(checkout):
    with pytest.raises(ValueError, match="vstage_uncached.task: no reference caches its answers to a corpus question"):
        _check("vstage_uncached.task")
    with pytest.raises(ValueError, match="vstage_gate.task: no reference caches"):
        _check("vstage_gate.task")


def test_check_applies_the_verify_stages_shape_rules(checkout):
    stage = _stage(SHAPE)
    _check("vstage_fill.task", stage_shape=stage.model_copy(update={"time_sec": 86400}))
    for other in (stage.model_copy(update={"cpus": 8}), SHAPE, stage.model_copy(update={"memory_gib": 16}),
                  stage.model_copy(update={"disk_gib": 0}),
                  stage.model_copy(update={"disk_gib": SHAPE.disk_gib + KEPT_DISK_GIB})):
        with pytest.raises(ValueError, match="the oracle_fill stage runs one role, so its resources must be "
                                             "reference_shape's with 4 GiB more memory for the runner, 40 GiB more "
                                             "disk for the image it extracts and 1 GiB for the copies of its data"):
            _check("vstage_fill.task", stage_shape=other)
    cpu = ResourceShape(cpus=4, memory_gib=16, time_sec=600)
    with pytest.raises(ValueError, match="a reference engine of vstage_fill.task asks for 1 GPUs; reference_shape "
                                         "has 0"):
        _check("vstage_fill.task", shape=cpu)
    h200 = SHAPE.model_copy(update={"gpu_kind": "H200"})
    with pytest.raises(ValueError, match=r"ask for \['B200'\] GPUs; reference_shape has H200"):
        _check("vstage_fill.task", shape=h200)


def test_check_refuses_a_caching_reference_on_a_multi_turn_workload_as_the_verify_stage_does(tmp_path, checkout):
    # From turn two a prompt carries the server's earlier replies: the same refusal as the
    # verify stage's (``multi_turn_caching``). The role's refusal of a multi-turn row is the
    # second guard (``test_a_deterministic_refusal_is_infra_that_is_not_retried``).
    words = (r"vstage_cachedchat.task: tenant t0's workload runs 2.5 turns per session, so its prompts carry the "
             r"server's replies from turn two, and AskVerifier caches the answers of its reference "
             r"GeneratorReference \(cache=True\)")
    with pytest.raises(ValueError, match=words):
        _check("vstage_cachedchat.task")
    with pytest.raises(PlanError, match="stage oracle_fill: " + words):
        _plan(tmp_path, checkout, "vstage_cachedchat.task")
    _check("vstage_cached.task")


def test_check_refuses_a_caching_reference_on_an_agentic_task_as_the_verify_stage_does(tmp_path, checkout):
    words = (r"vstage_cachedagent.task: the task declares an agent, so tenant t0's trajectories send calls that "
             r"carry the server's earlier replies whatever its workload declares, and AskVerifier caches the "
             r"answers of its reference GeneratorReference \(cache=True\)")
    with pytest.raises(ValueError, match=words):
        _check("vstage_cachedagent.task")
    with pytest.raises(PlanError, match="stage oracle_fill: " + words):
        _plan(tmp_path, checkout, "vstage_cachedagent.task")


def test_check_accepts_a_caching_tenant_beside_a_multi_turn_one_that_caches_nothing(checkout):
    _check("vstage_fill_mixed.task")


def test_check_refuses_a_task_that_does_not_load(checkout):
    with pytest.raises(ValueError, match=r"task vstage_absent.task does not load: ModuleNotFoundError"):
        _check("vstage_absent.task")


def test_the_planner_refuses_what_the_fill_cannot_run(tmp_path, checkout):
    with pytest.raises(PlanError, match="stage oracle_fill: vstage_uncached.task: no reference caches"):
        _plan(tmp_path, checkout, "vstage_uncached.task")
