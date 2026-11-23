"""``hls oracle fill`` then ``hls grade`` on one machine, through the in-process Slurm of
``tests/pipeline/test_end_to_end_local.py`` whose node has exactly the disk each allocation asks
for: the fill computes the caching reference's answers to the test split into the store, and
the grading's verify serves every one of them, looked up by the prompts the real driver sent,
without deploying the reference. A fill and a grading under the same task overrides provision
and ask about the same rows; a fill without them covers no grading that has them."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from hls.pipeline import commands
from hls.pipeline.plan import PlanStore
from hls.pipeline.planner import PlanError
from hls.pipeline.preflight import check_oracle_coverage
from hls.provision.model import Provisioned
from hls.provision.mounts import copies_gib
from hls.store.layout import InstanceLayout
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.taskref import TaskRef, load_task
from hls.verify.fill import FillRecord
from hls.substrate.slurm.scheduler import submit_argv
from tests.drive import fake_container
from tests.drive.test_stage import SERVER, _server_image
from tests.evaluate import fixture
from tests.evaluate import test_grade_end_to_end_local as whole_path
from tests.evaluate.test_grade_end_to_end_local import _cache_images, _grade, _output, _slurm
from tests.evaluate.warmer import publish_package
from tests.pipeline.hls_cli import flags, run_hls
from tests.pipeline.test_end_to_end_local import LocalSlurm
from tests.provision.warmer_config import SECRETS
from tests.recipes.warmer_site import local_grading_node, site_config

#: The grading tests' fake runtime, seal and hub.
bins = whole_path.bins


def test_a_fill_answers_the_split_and_every_later_grading_serves_it(tmp_path, login, in_catalog, bins):
    store = PosixStore(tmp_path / "store")
    baseline = fake_container.make_image(tmp_path / "baseline.sif", {"workspace/server.py": (SERVER.read_bytes(),
                                                                                             0o644)}).read_bytes()
    in_catalog(roles=("baseline",))
    _cache_images(store, baseline)
    config = site_config(tmp_path, site=f'pasta_path = "{bins / "pasta"}"\n{local_grading_node()}',
                         secrets=SECRETS)
    # The plan sizes the fill's copies of the task's data by what the cache holds.
    whole_path.provision(tmp_path, config, "ev_cached.task")

    slurm = _slurm(tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FAKE_CONTAINER_RECORDS", str(tmp_path / "records-fill"))
        code, lines = _fill_command(slurm, config, {})
    assert code == 0, lines
    fill_instance = lines[0].removeprefix("instance ")
    assert fill_instance.startswith(f"warmers/{fixture.TASK_ID}/")
    # A fill records no score: its last line is how it ended.
    assert lines[-1] == f"{fill_instance}: succeeded: every stage published"
    [job] = slurm.jobs.values()
    assert {"--partition=gpu-a", "--gpus=b200:1", "--job-name=hls-oracle-fill-test"} <= set(submit_argv(job.request))
    # The node had exactly the allocation's disk: the fill's tree and the copies of its data, since
    # the cache held every image and the images stage holds none.
    plan = PlanStore(store, InstanceLayout(fill_instance)).latest()
    assert plan.stage("images").shape.disk_gib == 0
    data = copies_gib(plan.stage("oracle_fill").params["data_copies"])
    assert plan.allocations[0].request.disk_gib == plan.stage("oracle_fill").shape.disk_gib == 40 + data
    record = FillRecord.model_validate_json(_output(store, fill_instance, "oracle_fill", "fill"))
    [ref] = record.references
    # The test split's four rows in chunks of three: two calls, two files.
    assert (ref.engine, ref.prompts, ref.answered, ref.served) == ("FakeGenerator", 4, 4, 0)
    assert len(record.answers_written) == 2
    filled = fake_container.records(tmp_path / "records-fill")["oracle-fill"]
    assert any(b.endswith(":/data/tenants/t0/test:ro") for b in filled["binds"])
    # A later grading's plan-time check finds every answer from the login node, reading no row:
    # the question set the fill recorded last, under the rows key the provision stage published,
    # which the login node resolves from the task's declaration alone.
    assert len(record.questions) == 1
    provisioned = Provisioned.model_validate_json(_output(store, fill_instance, "provision", "provisioned"))
    manifest, _ = read_manifest(store, InstanceLayout(fill_instance), "oracle_fill")
    task = importlib.import_module("ev_cached.task").TASK
    coverage = check_oracle_coverage(store, task_ref=TaskRef(module="ev_cached.task"), task_id=fixture.TASK_ID,
                                     task=task, gpu_kind="B200",
                                     reference_sha256=manifest.provenance.image_digests["oracle-fill"])
    assert (coverage.rows, coverage.prompts, coverage.files) == (provisioned.rows.key, 4, 2)

    package = publish_package(store, _server_image(tmp_path / "submission.sif", "serve").read_bytes())
    code, again = _grade(tmp_path, bins, _slurm(tmp_path), package=package, records="records-grade",
                         task_module="ev_cached.task")
    assert code == 0, again
    grading = next(line for line in again if line.startswith("instance ")).removeprefix("instance ")
    # The fixture task verifies on its built layer, so the grading's plan could only warn.
    assert again[0].startswith(f"warning: {fixture.TASK_ID} verifies on a built image")
    assert grading.startswith(f"{package}/gradings/")
    verify = json.loads(_output(store, grading, "verify", "verify"))
    assert verify["engines"] == [], "every answer was filled, so the reference was never deployed"
    assert verify["answers_read"] == record.answers_written and verify["answers_written"] == {}
    score = json.loads(_output(store, grading, "score", "score"))
    assert score["gate_passed"] and score["value"] > 0
    # The fill and the verify ran on one image and silicon, so they share one answers directory.
    assert {k.split("/")[0] for k in record.answers_written} == {k.split("/")[0] for k in verify["answers_read"]}


def _fill_command(slurm: LocalSlurm, config: Path, overrides: dict[str, str]) -> tuple[int, list[str]]:
    """``hls oracle fill ev_cached.task --chunk 3`` with *overrides*: the exit status and what it
    printed."""
    return run_hls(slurm, ["oracle", "fill", "ev_cached.task", *flags(overrides), "--chunk", "3",
                           "--job-name", "hls-oracle-fill-test", "--config", str(config)])


#: An override that shrinks the test split from four rows to three: another rows entry, and other
#: questions for the caching reference.
SPLIT = {"task.data_source.split.test.value": "3"}


def _fill(tmp_path, overrides: dict[str, str], config) -> str:
    """``hls provision`` then ``hls oracle fill ev_cached.task``, each with *overrides*, run to its
    end; the fill's instance."""
    whole_path.provision(tmp_path, config, "ev_cached.task", overrides)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FAKE_CONTAINER_RECORDS", str(tmp_path / f"records-fill-{len(overrides)}"))
        code, lines = _fill_command(_slurm(tmp_path), config, overrides)
    assert code == 0, lines
    return lines[0].removeprefix("instance ")


def test_a_grading_with_task_overrides_is_covered_only_by_a_fill_with_the_same_ones(tmp_path, login, in_catalog,
                                                                                    bins):
    """An override of the test split changes the rows a grading provisions and the questions its
    verify asks. A fill with the same override provisions and answers those rows, and the
    grading's plan-time check finds them; a fill without it answered the bare task's rows, under
    another key, and the check refuses, naming the fill to run."""
    store = PosixStore(tmp_path / "store")
    baseline = fake_container.make_image(tmp_path / "baseline.sif", {"workspace/server.py": (SERVER.read_bytes(),
                                                                                             0o644)}).read_bytes()
    in_catalog(roles=("baseline",))
    _cache_images(store, baseline)
    config = site_config(tmp_path, site=f'pasta_path = "{bins / "pasta"}"', secrets=SECRETS)
    ref = TaskRef(module="ev_cached.task", overrides={"task.data_source.split.test.value": 3})
    task = load_task(ref)

    bare = _fill(tmp_path, {}, config)
    manifest, _ = read_manifest(store, InstanceLayout(bare), "oracle_fill")
    image = manifest.provenance.image_digests["oracle-fill"]
    with pytest.raises(PlanError, match=r"Run hls oracle fill ev_cached\.task "
                                        r"--task\.data_source\.split\.test\.value 3 first$"):
        check_oracle_coverage(store, task_ref=ref, task_id=fixture.TASK_ID, task=task, gpu_kind="B200",
                              reference_sha256=image)

    filled = _fill(tmp_path, SPLIT, config)
    [asked] = FillRecord.model_validate_json(_output(store, filled, "oracle_fill", "fill")).references
    assert asked.prompts == 3
    provisioned = Provisioned.model_validate_json(_output(store, filled, "provision", "provisioned"))
    coverage = check_oracle_coverage(store, task_ref=ref, task_id=fixture.TASK_ID, task=task, gpu_kind="B200",
                                     reference_sha256=image)
    assert (coverage.rows, coverage.prompts) == (provisioned.rows.key, 3)
    assert coverage.rows != Provisioned.model_validate_json(_output(store, bare, "provision", "provisioned")).rows.key
    # The grading plans with the override, and its provision stage provisions the overridden task.
    # Its verify runs on the task's built image, whose file is not known before the images stage,
    # so the plan warns rather than checks.
    lines: list[str] = []
    plan = commands.plan_grade(config=config, overrides=SPLIT, task_module="ev_cached.task", measured="baseline",
                               job_name="hls-grade-test", echo=lines.append)
    assert plan.stage("provision").params["task"] == ref.model_dump(mode="json")
    assert any(line.startswith(f"warning: {fixture.TASK_ID} verifies on a built image") for line in lines)

