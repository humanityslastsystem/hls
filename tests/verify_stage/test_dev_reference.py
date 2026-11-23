"""A task's dev reference (``hls.verify.dev_reference``): the oracle fill produces it from the dev
rows, the record it writes fits the task it was produced for and no other, and every role that
sees the dev split finds its text beside the dev rows, while the rows entry holds none of it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hls.measured import DEV_REFERENCE_FILE, defaults_path
from hls.provision.model import Provisioned, RowsRef
from hls.provision.mounts import copied_entries, data_mounts, dev_view
from hls.provision.rows import tenant_mount_path
from hls.store.cache import subtree_disk_bytes
from hls.store.digest import sha256_file
from hls.store.manifest import read_outputs_report
from hls.taskref import TaskRef, load_task
from hls.verify.dev_reference import (DevReference, DevReferenceError, _document, _graders, _layout, declaring_tenants,
                                      dev_reference_fingerprint, produce,
                                      read_dev_reference)
from hls.verify import ContainsMatchMetric
from hls.verify.fill import FILL_FILE
from hls.verify.inprocess import EngineHost
from tests.verify_stage.fixture import dev_reference_task
from tests.verify_stage.test_fill import ROWS, _fill, _rows

MODULE = "vstage_devref.task"


def _dev_rows(root: Path, rows: list[dict] = ROWS) -> Path:
    """A data root holding tenant t0's dev rows, and its test rows, as the oracle role sees them."""
    _rows(root, rows)
    dev = root / "tenants" / "t0" / "dev"
    dev.mkdir(parents=True)
    (dev / "rows.json").write_text(json.dumps(rows))
    return root


async def _produced(task, rows: Path) -> DevReference:
    async with EngineHost() as host:
        return await produce(task, MODULE, rows, host=host)


def test_the_fill_produces_a_dev_reference_of_each_answer_scored_as_the_gate_scores_it(tmp_path):
    task = dev_reference_task()
    record = asyncio.run(_produced(task, _dev_rows(tmp_path / "data")))
    assert (record.task, record.dev_fingerprint) == (MODULE, dev_reference_fingerprint(task))
    [(tenant, published)] = record.tenants.items()
    assert (tenant, published.name) == ("t0", "oracle_dev.json")
    doc = json.loads(published.text)
    # The fake generator answers "B", every row's target, so the reference scores 1 on each.
    assert list(doc) == ["what", "metric", "oracle_mean_accuracy", "n", "normalised_example", "rows"]
    assert doc["n"] == len(ROWS) and doc["oracle_mean_accuracy"] == 1.0
    assert len(doc["rows"]) == len(ROWS)
    assert all(row == {"transcript": "B", "accuracy": 1.0} for row in doc["rows"].values())
    assert doc["normalised_example"] == {"raw": "B", "normalised": "b"}


def test_a_dev_reference_graded_by_a_metric_with_no_published_document_is_refused_before_any_engine_runs(tmp_path):
    """A dev reference has one document for each grading it is published for; a tenant graded by
    any other metric is refused, before an engine is asked anything."""
    task = dev_reference_task()
    other = task.model_copy(update={"verifiers": [task.verifiers[0].model_copy(update={"metric": ContainsMatchMetric()})]})

    class Unasked:
        async def acquire(self, engine):
            raise AssertionError("an engine was asked though no document can be published")

    with pytest.raises(DevReferenceError, match=r"graded by \['ContainsMatchMetric'\], which has no published"):
        asyncio.run(produce(other, MODULE, _dev_rows(tmp_path / "data"), host=Unasked()))


def test_a_dev_reference_fits_only_the_dev_rows_reference_and_metric_it_was_produced_on(tmp_path):
    task = dev_reference_task()
    record = asyncio.run(_produced(task, _dev_rows(tmp_path / "data")))
    path = tmp_path / DEV_REFERENCE_FILE
    path.write_text(record.model_dump_json(indent=1))
    assert read_dev_reference(path, task)[0] == record
    other = task.model_copy(update={"data_source": task.data_source.model_copy(
        update={"split": task.data_source.split.model_copy(update={"dev": task.data_source.split.test})})})
    with pytest.raises(DevReferenceError, match="stale dev reference"):
        read_dev_reference(path, other)
    renamed = record.model_copy(update={"tenants": {"t0": record.tenants["t0"].model_copy(update={"name": "x.json"})}})
    path.write_text(renamed.model_dump_json())
    with pytest.raises(DevReferenceError, match="not each declaring tenant's file"):
        read_dev_reference(path, task)


def test_the_fill_role_writes_and_reports_the_dev_reference_beside_its_record(checkout, tmp_path):
    out = tmp_path / "out" / FILL_FILE
    done = _fill(checkout, MODULE, _dev_rows(tmp_path / "data"), tmp_path / "area", out,
                 "--dev-out", str(out.parent / DEV_REFERENCE_FILE))
    assert done.returncode == 0, done.stderr
    report = read_outputs_report(out.parent / "outputs.json")
    assert report is not None and sorted(report) == [DEV_REFERENCE_FILE, FILL_FILE]
    assert (report[DEV_REFERENCE_FILE].sha256, report[DEV_REFERENCE_FILE].size) == sha256_file(
        out.parent / DEV_REFERENCE_FILE)
    task = load_task(TaskRef(module=MODULE))
    record, _ = read_dev_reference(out.parent / DEV_REFERENCE_FILE, task)
    assert record.task == MODULE


def test_a_task_that_declares_no_dev_reference_cannot_produce_one(checkout, tmp_path):
    out = tmp_path / "out" / FILL_FILE
    done = _fill(checkout, "vstage_fill.task", _dev_rows(tmp_path / "data"), tmp_path / "area", out,
                 "--dev-out", str(out.parent / DEV_REFERENCE_FILE))
    assert done.returncode == 3 and "declares no measured dev reference" in done.stderr


def test_a_role_that_sees_the_dev_split_finds_the_dev_reference_beside_the_rows(tmp_path):
    """The rows entry's copy holds the rows alone; a role that sees the dev split mounts a view that
    copies them, with the dev reference's text beside them, and leaves the rows copy as it was; a
    role that does not see it mounts the copy as it is."""
    copy = tmp_path / "copy"
    for split in ("dev", "test"):
        (copy / "tenants" / "t0" / split).mkdir(parents=True)
        (copy / "tenants" / "t0" / split / "rows.json").write_text(f"{split} rows")
    rows = RowsRef(key="rows-0123456789abcdef", entry_sha256="e" * 64, tenants=["t0"], disk_bytes=10,
                   dev_disk_bytes={"t0": 3 * 4096})
    provisioned = Provisioned(weights=None, rows=rows, tools=[])
    files = {"t0": {"oracle_dev.json": "the reference's answers"}}
    # The stage declares each role's view beside the copies: the dev split's copy and the file in
    # whole blocks, with a block for its name.
    declared = copied_entries(("artifact", "driver", "reference"), provisioned, weights=False, dev_files=files)
    assert declared == {f"rows/{rows.key}": 10, "dev-view/artifact/t0": 5 * 4096, "dev-view/driver/t0": 5 * 4096}
    with pytest.raises(ValueError, match="the plan made no room on the node for a copy of dev-view/artifact/t0"):
        data_mounts("artifact", provisioned, lambda kind, key, digest: pytest.fail("copied before the check"),
                    declared={f"rows/{rows.key}": 10}, weights=False, dev_files=files, views=tmp_path / "views")
    unsized = Provisioned(weights=None, rows=rows.model_copy(update={"dev_disk_bytes": {}}), tools=[])
    with pytest.raises(ValueError, match="states no size of tenant t0's dev split"):
        copied_entries(("artifact",), unsized, weights=False, dev_files=files)
    mounts, _ = data_mounts("artifact", provisioned, lambda kind, key, digest: copy, declared=declared,
                            weights=False, dev_files=files, views=tmp_path / "views")
    [dev] = mounts
    assert dev.target == tenant_mount_path("t0", "dev") and dev.read_only
    assert sorted(p.name for p in dev.source.iterdir()) == ["oracle_dev.json", "rows.json"]
    assert (dev.source / "oracle_dev.json").read_text() == "the reference's answers"
    source = copy / "tenants" / "t0" / "dev" / "rows.json"
    assert (dev.source / "rows.json").read_text() == "dev rows"
    assert source.stat().st_nlink == 1 and (dev.source / "rows.json").stat().st_ino != source.stat().st_ino
    assert not (copy / "tenants" / "t0" / "dev" / "oracle_dev.json").exists()
    driver, _ = data_mounts("driver", provisioned, lambda kind, key, digest: copy, declared=declared,
                            weights=False, dev_files=files, views=tmp_path / "driver-views")
    assert [m.source for m in driver] == [tmp_path / "driver-views" / "t0", copy / "tenants" / "t0" / "test"]
    reference, _ = data_mounts("reference", provisioned, lambda kind, key, digest: copy, declared=declared,
                               weights=False, dev_files=files, views=tmp_path / "reference-views")
    assert reference == [] and not (tmp_path / "reference-views").exists()
    with pytest.raises(ValueError, match="need a directory for their views"):
        data_mounts("artifact", provisioned, lambda kind, key, digest: copy, declared=declared, weights=False,
                    dev_files=files)


def test_the_provision_record_sizes_a_copy_of_each_tenants_dev_split():
    """What a view copies is counted as a whole entry's copy is: each file in whole blocks, and a
    block for the name of every file and directory under the split, the split's own included."""
    files = [SimpleNamespace(path=path, size=size) for path, size in (
        ("tenants/t0/dev/rows.json", 5000), ("tenants/t0/dev/audio/a.flac", 1), ("tenants/t0/test/rows.json", 10**9),
        ("tenants/t1/dev/rows.json", 10), ("ids.json", 10))]
    entry = SimpleNamespace(manifest=SimpleNamespace(files=files))
    # rows.json: 2 blocks, a.flac: 1, three names, and the dev and audio directories.
    assert subtree_disk_bytes(entry, "tenants/t0/dev") == 4096 * (2 + 1 + 2 + 2)
    assert subtree_disk_bytes(entry, "tenants/t1/dev") == 4096 * (1 + 1 + 1)
    assert subtree_disk_bytes(entry, "tenants/t2/dev") == 4096


def test_a_dev_reference_never_replaces_a_file_the_dev_rows_hold(tmp_path):
    rows = tmp_path / "dev"
    rows.mkdir()
    (rows / "oracle_dev.json").write_text("in the rows")
    with pytest.raises(ValueError, match="already hold"):
        dev_view(rows, tmp_path / "view", {"oracle_dev.json": "the reference's"})


@pytest.mark.parametrize("module", ["tasks.multimodal_asr.task", "tasks.multimodal_omni.task"])
def test_each_shipped_default_dev_reference_fits_its_task(module):
    ref = TaskRef(module=module)
    record, _ = read_dev_reference(defaults_path(ref, DEV_REFERENCE_FILE), load_task(ref))
    [published] = record.tenants.values()
    assert published.name == "oracle_dev.json" and json.loads(published.text)["n"] == 64


@pytest.mark.parametrize("module", ["tasks.multimodal_asr.task", "tasks.multimodal_omni.task"])
def test_a_produced_dev_reference_has_the_shape_of_the_shipped_default_byte_for_byte(module):
    """The document the fill publishes for a tenant graded by ``WERMetric`` or by the
    ``MixedModalityMatchMetric`` arms is the one the task's brief describes: laid out from the
    default's own rows and means, it is the default's text, byte for byte."""
    ref = TaskRef(module=module)
    task = load_task(ref)
    record, _ = read_dev_reference(defaults_path(ref, DEV_REFERENCE_FILE), task)
    text = record.tenants["t0"].text
    doc = json.loads(text)
    if "oracle_mean_accuracy" in doc:
        scored = [(rid, "0", row["transcript"], row["accuracy"]) for rid, row in doc["rows"].items()]
        values = {"0": doc["oracle_mean_accuracy"]}
    else:
        scored = [(rid, row["arm"], row["answer"], row["score"]) for rid, row in doc["rows"].items()]
        values = {"image": doc["oracle_mean_image"], "audio": doc["oracle_mean_audio"]}
    [tenant] = declaring_tenants(task)
    produced = _document(_layout(task.id, tenant.id, _graders(tenant)), scored, values, module.split(".")[1])
    assert json.dumps(produced, indent=1) + "\n" == text
    # What the brief says the file carries: the reference's answer and score for every dev row,
    # the arm of each and a mean per arm (multimodal_omni), the mean and a worked normalisation
    # example (multimodal_asr).
    assert produced["n"] == len(produced["rows"]) == 64
    if "oracle_mean_accuracy" in produced:
        assert all(set(row) == {"transcript", "accuracy"} for row in produced["rows"].values())
        assert isinstance(produced["oracle_mean_accuracy"], float)
        assert set(produced["normalised_example"]) == {"raw", "normalised"}
    else:
        arms = {row["arm"] for row in produced["rows"].values()}
        assert arms == {"image", "audio"} and all(set(row) == {"arm", "answer", "score"}
                                                  for row in produced["rows"].values())
        assert all(isinstance(produced[f"oracle_mean_{arm}"], float) for arm in arms)
        assert sum(produced["n_by_arm"].values()) == 64
