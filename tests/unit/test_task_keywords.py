"""A keyword that is not a field of a spec model is refused, including in the files read
back as records and in a copy's update. Every committed baseline credit and calibration loads
under its strict model and holds exactly what that model stores, as its writer writes it: its
fields, and no value the loader computes."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from hls.baseline_credit import BaselineCredit, baseline_credit_bytes
from hls.core.model import AcceleratorSpec, Hardware, InterconnectSpec, ModelSpec
from hls.core.poly import Spec, dump_spec
from hls.spec.datasource import Count, SplitSpec
from hls.spec.workload import ShortOutputCap
from hls.verify.calibration import (CALIBRATION_FILE, CALIBRATION_KINDS, Calibration, CalibrationError,
                                    ReportCalibration, read_calibration)

REPO = Path(__file__).resolve().parents[2]
#: Every task module of the repository, ``tasks.<dir>.task*``.
TASK_MODULES = sorted(".".join(p.relative_to(REPO).with_suffix("").parts) for p in REPO.glob("tasks/*/task*.py"))
#: Every default record of the repository's tasks, as ``<dir>/defaults/<file>``.
RECORDS = sorted(p.relative_to(REPO / "tasks").as_posix() for name in ("baseline.json", CALIBRATION_FILE)
                 for p in REPO.glob(f"tasks/*/defaults/{name}"))


def _subclasses(base: type) -> set[type]:
    found, pending = set(), [base]
    while pending:
        for cls in pending.pop().__subclasses__():
            if cls not in found:
                found.add(cls)
                pending.append(cls)
    return found


def _model(record: str, raw: dict) -> type[Spec]:
    return BaselineCredit if record.endswith("/baseline.json") else CALIBRATION_KINDS[raw["kind"]]


def _models_in(obj: object, found: set[type[BaseModel]], seen: set[int]) -> None:
    """Add to *found* the class of every pydantic model reachable from *obj* through model
    fields, dicts and sequences."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, BaseModel):
        found.add(type(obj))
        for name in type(obj).model_fields:
            _models_in(getattr(obj, name, None), found, seen)
    elif isinstance(obj, dict):
        for value in obj.values():
            _models_in(value, found, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            _models_in(value, found, seen)


def test_a_task_keyword_that_is_not_a_field_is_refused():
    # max_tokens is a method of the cap, not a field: the cap would have stayed at its default 16.
    with pytest.raises(ValidationError, match="max_tokens"):
        ShortOutputCap(max_tokens=8192)
    assert ShortOutputCap(cap_tokens=8192).max_tokens() == 8192


@pytest.mark.parametrize("module", TASK_MODULES)
def test_every_task_module_imports_with_unknown_keywords_refused(module):
    assert importlib.import_module(module).TASK is not None


def test_every_spec_model_refuses_unknown_keywords():
    for module in TASK_MODULES:
        importlib.import_module(module)
    models = {c for c in _subclasses(Spec) if c.__module__.startswith(("hls.", "tasks."))}
    assert {BaselineCredit, Calibration, ReportCalibration} <= models
    assert sorted(f"{c.__module__}.{c.__qualname__}" for c in models if c.model_config.get("extra") != "forbid") == []


def test_every_model_in_a_task_tree_refuses_unknown_keywords():
    # Not only Spec: every pydantic model a task module's TASK or ARTIFACT holds.
    found: set[type[BaseModel]] = set()
    for module in TASK_MODULES:
        loaded = importlib.import_module(module)
        for name in ("TASK", "ARTIFACT"):
            _models_in(getattr(loaded, name, None), found, set())
    assert {ModelSpec, Hardware, AcceleratorSpec} <= found
    assert sorted(f"{c.__module__}.{c.__qualname__}" for c in found if c.model_config.get("extra") != "forbid") == []


@pytest.mark.parametrize("cls, kwargs", [
    (ModelSpec, {"base_checkpoint": "m", "arch": "TX", "revison": "a" * 40}),
    (ModelSpec, {"base_checkpoint": "m", "arch": "TX", "chat_template_kwarg": {"enable_thinking": True}}),
    (Hardware, {"accelerator": AcceleratorSpec(kind="H200", count=1), "memory_gib": 1024}),
    (Hardware, {"accelerator": AcceleratorSpec(kind="H200", count=1), "cpus": 64}),
    (AcceleratorSpec, {"kind": "H200", "count": 1, "memory_gib_per_device": 141}),
    (InterconnectSpec, {"kind": "NVLink", "lanes": 18}),
])
def test_a_model_or_hardware_keyword_that_is_not_a_field_is_refused(cls, kwargs):
    # Each would have built with the field it meant at its default: revision "main", no
    # template switch, a brief stating 0 GiB of memory and 0 CPUs.
    keyword = next(k for k in kwargs if k not in cls.model_fields)
    with pytest.raises(ValidationError, match=keyword):
        cls(**kwargs)


def test_a_model_spec_reads_back_its_fingerprint_and_refuses_another():
    model = ModelSpec(base_checkpoint="m", arch="TX", revision="a" * 40,
                      chat_template_kwargs={"enable_thinking": True})
    dumped = model.model_dump(mode="json")
    assert dumped["fingerprint"] == model.fingerprint
    assert ModelSpec.model_validate(dumped) == model
    with pytest.raises(ValidationError, match=re.escape(
            f"fingerprint is 'm@main#TX?', but this model's is {model.fingerprint!r}")):
        ModelSpec.model_validate({**dumped, "fingerprint": "m@main#TX?"})


def test_a_copy_refuses_an_update_key_that_is_not_a_field():
    with pytest.raises(ValidationError, match="max_tokens"):
        ShortOutputCap(cap_tokens=8).model_copy(update={"max_tokens": 16})
    task = importlib.import_module("tasks.multitenant_slo.task").TASK
    with pytest.raises(ValidationError, match="idd"):
        task.model_copy(update={"idd": "x"})
    with pytest.raises(ValidationError, match="revison"):
        task.model.model_copy(update={"revison": "b" * 40})


def test_a_copy_validates_the_values_it_updates():
    cap = ShortOutputCap(cap_tokens=8)
    with pytest.raises(ValidationError, match="cap_tokens"):
        cap.model_copy(update={"cap_tokens": "many"})
    task = importlib.import_module("tasks.multitenant_slo.task").TASK
    with pytest.raises(ValidationError, match="id"):
        task.model_copy(update={"id": 123})
    assert cap.model_copy(update={"cap_tokens": 16}) == ShortOutputCap(cap_tokens=16)
    assert task.model_copy() == task and task.model_copy(deep=True) == task


def test_a_family_member_reads_back_its_kind_and_refuses_another():
    # An override writes a family-valued field as a whole table, kind included.
    split = SplitSpec(dev=Count(value=1), test=Count(value=3))
    assert SplitSpec.model_validate(dump_spec(split)) == split
    with pytest.raises(ValidationError, match="kind is 'Fraction', but this is a Count"):
        Count.model_validate({"kind": "Fraction", "value": 3})


@pytest.mark.parametrize("record", RECORDS)
def test_every_default_record_loads_under_its_strict_model_and_holds_what_it_stores(record):
    text = (REPO / "tasks" / record).read_text()
    raw = json.loads(text)
    model = _model(record, raw)
    assert not model.model_computed_fields
    # The bytes the writers (baseline_credit_bytes, write_calibration) write.
    parsed = model.model_validate(raw)
    if isinstance(parsed, BaselineCredit):
        assert baseline_credit_bytes(parsed).decode() == text
    else:
        assert json.dumps(parsed.model_dump(mode="json"), indent=2) == text


@pytest.mark.parametrize("record", RECORDS)
def test_a_default_record_with_a_key_its_model_lacks_is_refused(record):
    raw = json.loads((REPO / "tasks" / record).read_text())
    model = _model(record, raw)
    nested = "campaign" if model is BaselineCredit else "stack"
    for edited in ({**raw, "since_removed": 1}, {**raw, nested: {**raw[nested], "since_removed": 1}}):
        with pytest.raises(ValidationError, match="since_removed"):
            model.model_validate(edited)


def test_a_baseline_credit_file_carrying_a_summary_or_a_retired_key_is_refused():
    raw = json.loads((REPO / "tasks" / "multitenant_slo" / "defaults" / "baseline.json").read_text())
    credit = BaselineCredit.model_validate(raw)
    for name in ("log_credit_mean", "log_credit_sd", "log_credit_se", "restamp_note"):
        with pytest.raises(ValidationError, match=name):
            BaselineCredit.model_validate({**raw, name: getattr(credit, name, "carried over by hand")})
    drives = [{**raw["drives"][0], "open_ratio": 0.5}, *raw["drives"][1:]]
    with pytest.raises(ValidationError, match="open_ratio"):
        BaselineCredit.model_validate({**raw, "drives": drives})


def test_a_calibration_rate_whose_spread_is_not_its_servings_is_refused():
    raw = json.loads((REPO / "tasks" / "multitenant_slo" / "defaults" / CALIBRATION_FILE).read_text())
    engine = sorted(raw["rate_side"])[0]
    rate = raw["rate_side"][engine]
    edited = {**raw, "rate_side": {**raw["rate_side"], engine: {**rate, "sd": rate["sd"] * 1.01}}}
    with pytest.raises(CalibrationError, match="EngineRate sd="):
        Calibration.model_validate(edited)


def test_read_calibration_refuses_a_missing_file_naming_it(tmp_path):
    assert CALIBRATION_FILE == "calibration.json"
    task = importlib.import_module("tasks.multitenant_slo.task").TASK
    missing = tmp_path / CALIBRATION_FILE
    with pytest.raises(CalibrationError, match=re.escape(f"no calibration at {missing}")):
        read_calibration(missing, task)[0]
    assert read_calibration(REPO / "tasks" / "multitenant_slo" / "defaults" / CALIBRATION_FILE, task)[0].workload_fingerprint == (
        json.loads((REPO / "tasks" / "multitenant_slo" / "defaults" / CALIBRATION_FILE).read_text())["workload_fingerprint"])


def test_read_calibration_refuses_a_file_without_its_kind_tag(tmp_path):
    raw = json.loads((REPO / "tasks" / "multitenant_slo" / "defaults" / CALIBRATION_FILE).read_text())
    del raw["kind"]
    (tmp_path / CALIBRATION_FILE).write_text(json.dumps(raw))
    with pytest.raises(CalibrationError, match="carries no kind tag"):
        read_calibration(tmp_path / CALIBRATION_FILE, importlib.import_module("tasks.multitenant_slo.task").TASK)[0]


def test_the_default_records_are_found():
    # The two record tests above run once per record, so an empty list would pass them silently.
    assert len(RECORDS) >= 10
