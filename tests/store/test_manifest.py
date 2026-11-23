from __future__ import annotations

import hashlib
import os

import pytest
from pydantic import ValidationError

import hls.store.manifest as manifest_mod
from hls.spec.resources import GrantedResources, ResourceShape
from hls.store.base import NotFound
from hls.store.layout import InstanceLayout
from hls.store.manifest import (AlreadyPublished, InputMismatch, Manifest, OutputRef, Provenance, Resources,
                                ReportedOutput, Times, canonical_json, output_ref, publish_manifest,
                                read_manifest, read_outputs_report, report_outputs, report_written_outputs,
                                require_inputs)
from hls.store.posix import PosixStore
from tests.store.entries import recording_reads

LAY = InstanceLayout.rollout("t", "r")
SHAPE = ResourceShape(gpus=1, gpu_kind="H200", cpus=8, memory_gib=180, time_sec=3600)
GRANT = GrantedResources(gpus=1, gpu_kind="H200", cpus=8, memory_gib=180.0, disk_free_gib=900.0)


PROVENANCE = dict(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                  image_digests={"build": "c" * 64}, config_sha256="d" * 64)


def _manifest(stage: str, outputs: dict[str, OutputRef], inputs: dict[str, str] | None = None,
              layout: InstanceLayout = LAY) -> Manifest:
    return Manifest(
        instance=layout.root, stage=stage, attempt=1, outputs=outputs, inputs=inputs or {},
        provenance=Provenance(**PROVENANCE),
        resources=Resources(declared=SHAPE, granted=GRANT),
        times=Times(submitted_at=1.0, allocation_started_at=2.0, started_at=2.5, ended_at=3.0))


def test_canonical_json_ignores_construction_order():
    a = _manifest("package", {"x": OutputRef(key="k1", sha256="1" * 64, size=1),
                              "y": OutputRef(key="k2", sha256="2" * 64, size=2)})
    b = _manifest("package", {"y": OutputRef(key="k2", sha256="2" * 64, size=2),
                              "x": OutputRef(key="k1", sha256="1" * 64, size=1)})
    assert canonical_json(a) == canonical_json(b)


def test_publish_once_and_read_back(tmp_path):
    store = PosixStore(tmp_path)
    m = _manifest("package", {})
    digest = publish_manifest(store, LAY, m)
    got, got_digest = read_manifest(store, LAY, "package")
    assert got == m and got_digest == digest
    with pytest.raises(AlreadyPublished):
        publish_manifest(store, LAY, m)


def test_read_missing_is_none(tmp_path):
    assert read_manifest(PosixStore(tmp_path), LAY, "verify") is None


def test_output_ref_records_the_writers_digest_and_reads_only_the_objects_size(tmp_path):
    """The writer of an output reported its digest; the reference reads nothing of the bytes."""
    store = PosixStore(tmp_path)
    key = "runs/t/r/stages/package/attempts/1/outputs/image.sif"
    data = b"a large image"
    store.write_bytes(key, data)
    reported = hashlib.sha256(data).hexdigest()
    with recording_reads() as reads:
        ref = output_ref(store, key, reported)
    assert ref == OutputRef(key=key, sha256=reported, size=len(data)) and reads == []
    with pytest.raises(NotFound):
        output_ref(store, "runs/t/r/stages/package/attempts/1/outputs/absent", reported)
    with pytest.raises(ValidationError):
        output_ref(store, key, "sha256:" + reported)


def test_a_roles_output_report_keys_each_file_by_its_path_under_the_outputs_with_its_size(tmp_path):
    outputs = tmp_path / "outputs"
    (outputs / "trace").mkdir(parents=True)
    (outputs / "trace" / "END.json").write_bytes(b"{}")
    (outputs / "END.json").write_bytes(b"[1]")
    report = tmp_path / "role" / "outputs.json"
    report.parent.mkdir()
    report_outputs(report, outputs, outputs / "trace" / "END.json", outputs / "END.json")
    assert read_outputs_report(report) == {
        "trace/END.json": ReportedOutput(sha256=hashlib.sha256(b"{}").hexdigest(), size=2),
        "END.json": ReportedOutput(sha256=hashlib.sha256(b"[1]").hexdigest(), size=3)}
    with pytest.raises(ValueError):
        report_outputs(report, outputs / "trace", outputs / "END.json")
    for bad in ("", "[]", '{"a": "' + "a" * 64 + '"}', '{"a": {"sha256": "x", "size": 1}}',
                '{"a": {"sha256": "' + "a" * 64 + '", "size": -1}}'):
        report.write_text(bad)
        assert read_outputs_report(report) is None
    assert read_outputs_report(tmp_path / "absent.json") is None


def test_written_outputs_are_keyed_by_their_path_under_the_outputs_and_none_outside(tmp_path, monkeypatch):
    """A role that hashed what it wrote as it wrote it reports those records as they are, keyed
    as ``report_outputs`` keys them, and opens none of the files; a file outside the outputs is
    refused before anything is reported."""
    outputs = tmp_path / "outputs"
    end = ReportedOutput(sha256="a" * 64, size=7)
    chunk = ReportedOutput(sha256="b" * 64, size=2**34)
    report = tmp_path / "role" / "outputs.json"
    report.parent.mkdir()

    def refuse(path, *a, **k):
        raise AssertionError(f"{path} was read back")

    monkeypatch.setattr(manifest_mod, "hash_file", refuse)
    report_written_outputs(report, outputs, {outputs / "trace" / "END.json": end,
                                             outputs / "trace" / "chunk-000000.jsonl.gz": chunk})
    assert read_outputs_report(report) == {"trace/END.json": end, "trace/chunk-000000.jsonl.gz": chunk}
    report.unlink()
    for outside in (tmp_path / "END.json", outputs / ".." / "END.json", outputs):
        with pytest.raises(ValueError):
            report_written_outputs(report, outputs, {outputs / "trace" / "END.json": end, outside: end})
        with pytest.raises(ValueError):
            report_outputs(report, outputs, outside)
        assert not report.exists()


def test_a_report_given_its_disk_beforehand_is_written_over_in_place(tmp_path, monkeypatch):
    """A stage gives a role's report its disk before a contestant can fill the node's. The role
    writes over those bytes and cuts the rest; it never empties the file first, which would hand
    its blocks to whoever takes them next."""
    outputs = tmp_path / "outputs"
    report = tmp_path / "outputs.json"
    report.write_bytes(b" " * 2**16)
    inode = report.stat().st_ino
    flags: list[int] = []
    real_open = os.open

    def recording(path, flag, *args, **kwargs):
        if os.fspath(path) == os.fspath(report):
            flags.append(flag)
        return real_open(path, flag, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording)
    end = ReportedOutput(sha256="a" * 64, size=7)
    report_written_outputs(report, outputs, {outputs / "trace" / "END.json": end})
    monkeypatch.undo()
    assert len(flags) == 1 and not flags[0] & os.O_TRUNC
    assert report.stat().st_ino == inode and report.stat().st_size < 2**16
    assert read_outputs_report(report) == {"trace/END.json": end}


def test_times_record_allocation_start_apart_from_stage_start():
    times = _manifest("package", {}).times
    assert times.allocation_started_at == 2.0 and times.started_at == 2.5


def test_require_inputs(tmp_path):
    store = PosixStore(tmp_path)
    d_build = publish_manifest(store, LAY, _manifest("agent_build", {}))
    got = require_inputs(store, {LAY.manifest("agent_build"): d_build})
    assert set(got) == {LAY.manifest("agent_build")}
    with pytest.raises(InputMismatch, match=r"runs/t/r/stages/package/MANIFEST\.json: not published"):
        require_inputs(store, {LAY.manifest("package"): "0" * 64})
    with pytest.raises(InputMismatch, match=r"runs/t/r/stages/agent_build/MANIFEST\.json: digest"):
        require_inputs(store, {LAY.manifest("agent_build"): "f" * 64})


def test_grading_pins_rollout_manifests_by_key(tmp_path):
    store = PosixStore(tmp_path)
    grading = LAY.grading("g1")
    d_build = publish_manifest(store, LAY, _manifest("agent_build", {}))
    d_rollout_pkg = publish_manifest(store, LAY, _manifest("package", {}))
    # A rebuild gives the grading its own package, beside the rollout's.
    d_grading_pkg = publish_manifest(store, grading, _manifest("package", {}, layout=grading))
    assert d_rollout_pkg != d_grading_pkg
    pins = {LAY.manifest("agent_build"): d_build, grading.manifest("package"): d_grading_pkg}
    got = require_inputs(store, pins)
    assert got[LAY.manifest("agent_build")].instance == LAY.root
    assert got[grading.manifest("package")].instance == grading.root
    publish_manifest(store, grading, _manifest("verify", {}, inputs=pins, layout=grading))
    assert read_manifest(store, grading, "verify")[0].inputs == pins
    with pytest.raises(InputMismatch, match="digest"):
        require_inputs(store, {LAY.manifest("package"): d_grading_pkg})


def test_corrupt_input_manifest_is_an_input_mismatch(tmp_path):
    store = PosixStore(tmp_path)
    key = LAY.manifest("agent_build")
    garbage = b"\x00not a manifest"
    store.create(key, garbage)
    with pytest.raises(InputMismatch, match=r"agent_build/MANIFEST\.json: not a valid manifest"):
        require_inputs(store, {key: hashlib.sha256(garbage).hexdigest()})
    with pytest.raises(InputMismatch, match=r"agent_build/MANIFEST\.json: digest"):
        require_inputs(store, {key: "e" * 64})


def test_require_inputs_reports_every_problem_at_once(tmp_path):
    store = PosixStore(tmp_path)
    d_build = publish_manifest(store, LAY, _manifest("agent_build", {}))
    garbage = b"{}"
    store.create(LAY.manifest("package"), garbage)
    pins = {
        LAY.manifest("agent_build"): "f" * 64,
        LAY.manifest("package"): hashlib.sha256(garbage).hexdigest(),
        LAY.manifest("verify"): "0" * 64,
    }
    with pytest.raises(InputMismatch) as caught:
        require_inputs(store, pins)
    msg = str(caught.value)
    assert f"{LAY.manifest('agent_build')}: digest {d_build} is not the planned {'f' * 64}" in msg
    assert f"{LAY.manifest('package')}: not a valid manifest" in msg
    assert f"{LAY.manifest('verify')}: not published" in msg


def test_directory_at_an_input_key_is_not_published(tmp_path):
    (tmp_path / LAY.manifest("agent_build")).mkdir(parents=True)
    with pytest.raises(InputMismatch, match="not published"):
        require_inputs(PosixStore(tmp_path), {LAY.manifest("agent_build"): "0" * 64})


@pytest.mark.parametrize(
    "key",
    [
        "agent_build",
        "runs/t/r/stages/agent_build",
        "runs/t/r/stages/agent_build/plan.json",
        "runs/t/r/agent_build/MANIFEST.json",
        "stages/agent_build/MANIFEST.json",
        "runs/t/r/stages/Agent-Build/MANIFEST.json",
        "runs/t/r/stages/agent_build/MANIFEST.json\n",
        "runs/../r/stages/agent_build/MANIFEST.json",
    ],
)
def test_input_keys_must_name_stage_manifests(tmp_path, key):
    with pytest.raises(ValueError):
        require_inputs(PosixStore(tmp_path), {key: "0" * 64})
    with pytest.raises(ValidationError):
        _manifest("verify", {}, inputs={key: "0" * 64})


@pytest.mark.parametrize("bad", ["sha256:" + "0" * 64, "0" * 64 + "\n", "0" * 63])
def test_require_inputs_rejects_a_malformed_planned_digest(tmp_path, bad):
    with pytest.raises(ValueError):
        require_inputs(PosixStore(tmp_path), {LAY.manifest("agent_build"): bad})


PREFIXED = "sha256:" + "c" * 64


@pytest.mark.parametrize(
    "build",
    [
        lambda: OutputRef(key="k", sha256=PREFIXED, size=1),
        lambda: OutputRef(key="k", sha256="c" * 64 + "\n", size=1),
        lambda: Provenance(**{**PROVENANCE, "image_digests": {"build": PREFIXED}}),
        lambda: Provenance(**{**PROVENANCE, "config_sha256": PREFIXED}),
        lambda: Provenance(**{**PROVENANCE, "dirty_diff_sha256": PREFIXED}),
        lambda: Provenance(**{**PROVENANCE, "cache_entries": {"weights/m": PREFIXED}}),
        lambda: _manifest("verify", {}, inputs={LAY.manifest("package"): PREFIXED}),
    ],
    ids=["output", "output_newline", "image", "config", "dirty_diff", "cache_entry", "input"],
)
def test_digests_are_bare_hex(build):
    with pytest.raises(ValidationError):
        build()


def test_manifest_stage_and_instance_must_match_layout(tmp_path):
    store = PosixStore(tmp_path)
    other = InstanceLayout.rollout("t", "other")
    with pytest.raises(ValueError):
        publish_manifest(store, other, _manifest("package", {}))


def test_canonical_json_refuses_nan():
    m = _manifest("package", {})
    bad = m.model_copy(update={"times": m.times.model_copy(update={"ended_at": float("nan")})})
    with pytest.raises(ValueError):
        canonical_json(bad)


def test_require_inputs_refuses_a_manifest_stored_under_another_key(tmp_path):
    store = PosixStore(tmp_path)
    publish_manifest(store, LAY, _manifest("package", {}))
    raw = store.read_bytes(LAY.manifest("package"))
    moved = LAY.grading("g").manifest("package")
    store.create(moved, raw)
    with pytest.raises(InputMismatch, match=r"holds the manifest of runs/t/r/stages/package/MANIFEST\.json"):
        require_inputs(store, {moved: hashlib.sha256(raw).hexdigest()})


def test_manifest_naming_an_invalid_instance_or_stage_is_an_input_problem(tmp_path):
    store = PosixStore(tmp_path)
    base = _manifest("package", {})
    bad_stage = base.model_copy(update={"stage": "Bad Stage"})
    bad_instance = base.model_copy(update={"instance": ""})
    pins = {}
    for stage, manifest in (("package", bad_stage), ("verify", bad_instance)):
        raw = canonical_json(manifest)
        store.create(LAY.manifest(stage), raw)
        pins[LAY.manifest(stage)] = hashlib.sha256(raw).hexdigest()
    pins[LAY.manifest("agent_build")] = "0" * 64
    with pytest.raises(InputMismatch) as caught:
        require_inputs(store, pins)
    msg = str(caught.value)
    assert f"{LAY.manifest('package')}: manifest names an invalid instance or stage: " in msg
    assert f"{LAY.manifest('verify')}: manifest names an invalid instance or stage: " in msg
    assert f"{LAY.manifest('agent_build')}: not published" in msg


def test_provenance_records_the_cache_entries_a_stage_used():
    assert Provenance(**PROVENANCE).cache_entries == {}
    prov = Provenance(**{**PROVENANCE, "cache_entries": {"weights/org+m@" + "a" * 40: "e" * 64}})
    assert Provenance.model_validate_json(prov.model_dump_json()) == prov
