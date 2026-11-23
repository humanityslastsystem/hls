"""Tests for Tenant model + TaskSpec XOR authoring + effective_tenants().

Single-tenant form normalizes to one Tenant(id="t0"); multi-tenant form uses
tenants verbatim. The XOR is validated: providing both forms or neither raises
CompatibilityError. effective_tenants() is a plain method -- provenance dumps for
single-tenant tasks carry no 'tenants' key.
"""

import pytest

from hls.core.decode import Greedy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.reference.references import VLLMReference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.verify import LetterMatchMetric, MetricVerifier, TokenFaithfulnessVerifier, WiringError
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import CompatibilityError, TaskSpec
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, Uncapped, Workload)
from hls.prompt import JinjaPromptTemplate
from hls.spec.tenant import Tenant


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _contract():
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
    )


def _model():
    return ModelSpec(base_checkpoint="fake/model", arch="TX")


def _hardware():
    return Hardware(accelerator=AcceleratorSpec(kind="B200", count=1))


def _scorer():
    return GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))


def _dataset():
    return HFDataset(
        path="cais/mmlu",
        subset="abstract_algebra",
        revision="main",
        source_split="test",
        prompt=JinjaPromptTemplate(template="{{ question }}"),
        target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"),
    )


def _data_source():
    return DataSource(
        dataset=_dataset(),
        split=SplitSpec(dev=Count(value=5), test=Count(value=10)),
    )


def _workload():
    return Workload(
        arrival=AllAtOnce(),
        inter_turn_delay=NoDelay(),
        output_length_cap=Uncapped(),
        slo=NoSLO(),
    )


def _verifiers():
    return [MetricVerifier(metric=LetterMatchMetric(), min_value=0.0)]


def _base_kwargs(**overrides):
    """Minimal valid single-tenant TaskSpec kwargs."""
    kw = dict(
        id="T",
        model=_model(),
        hardware=_hardware(),
        contract=_contract(),
        workload=_workload(),
        verifiers=_verifiers(),
        scorer=_scorer(),
    )
    kw.update(overrides)
    return kw


# ---------------------------------------------------------------------------
# Single-tenant form tests
# ---------------------------------------------------------------------------

def test_single_tenant_effective_tenants_is_one_t0():
    """Single-tenant form: effective_tenants() returns exactly one Tenant with id 't0'."""
    ts = TaskSpec(**_base_kwargs())
    tenants = ts.effective_tenants()
    assert len(tenants) == 1
    assert tenants[0].id == "t0"


def test_single_tenant_carries_workload_and_verifiers():
    """Single-tenant form: the synthesized tenant carries the same workload + verifiers."""
    wl = _workload()
    vs = _verifiers()
    ts = TaskSpec(**_base_kwargs(workload=wl, verifiers=vs))
    t = ts.effective_tenants()[0]
    assert t.workload == wl
    assert t.verifiers == vs


def test_single_tenant_carries_data_source():
    """Single-tenant form with data_source: tenant.data_source matches."""
    ds = _data_source()
    ts = TaskSpec(**_base_kwargs(data_source=ds))
    assert ts.effective_tenants()[0].data_source == ds


def test_single_tenant_no_data_source():
    """Single-tenant form allows omitting data_source; tenant.data_source is None."""
    ts = TaskSpec(**_base_kwargs())   # no data_source
    assert ts.effective_tenants()[0].data_source is None


def test_single_tenant_provenance_has_tenants_key():
    """Unified shape: single-tenant dumps under tenants:[{id:"t0", ...}] too, so the
    viewer reads every run the same way (was: no 'tenants' key, byte-identical)."""
    ts = TaskSpec(**_base_kwargs())
    rec = ts.model_dump(mode="json", serialize_as_any=True)
    assert "tenants" in rec
    assert [t["id"] for t in rec["tenants"]] == ["t0"]


def test_single_tenant_provenance_has_no_top_level_triple():
    """Unified shape: config lives ONLY under tenants[] -- the top-level
    data_source/workload/verifiers triple is dropped from the dump."""
    ts = TaskSpec(**_base_kwargs())
    rec = ts.model_dump(mode="json", serialize_as_any=True)
    assert "workload" not in rec and "verifiers" not in rec and "data_source" not in rec
    # ...and the workload landed inside the synthesized t0 tenant instead.
    assert "workload" in rec["tenants"][0]


def test_single_tenant_authoring_forms_dump_identically():
    """The two single-tenant authoring forms -- top-level triple vs an explicit
    tenants=[Tenant(id="t0", ...)] -- now serialize to IDENTICAL dumps."""
    raw = TaskSpec(**_base_kwargs())
    explicit = TaskSpec(**_base_kwargs(
        workload=None, verifiers=None,
        tenants=[Tenant(id="t0", workload=_workload(), verifiers=_verifiers())],
    ))
    assert (raw.model_dump(mode="json", serialize_as_any=True)
            == explicit.model_dump(mode="json", serialize_as_any=True))


def test_python_mode_dump_keeps_tenants_python_typed():
    """mode is threaded: model_dump() (python) keeps tenant leaves python-typed,
    not json-coerced -- here the verifier's `accepts`/contract frozensets etc. stay
    python. Concretely, python-mode tenants equal a direct python-mode tenant dump."""
    ts = TaskSpec(**_base_kwargs())
    rec = ts.model_dump()                                   # python mode
    direct = [t.model_dump(serialize_as_any=True) for t in ts.effective_tenants()]
    assert rec["tenants"] == direct


# ---------------------------------------------------------------------------
# Multi-tenant form tests
# ---------------------------------------------------------------------------

def _tenant(tid: str) -> Tenant:
    return Tenant(id=tid, workload=_workload(), verifiers=_verifiers())


def test_multi_tenant_effective_tenants_verbatim():
    """Multi-tenant form: effective_tenants() returns the tenants list verbatim."""
    t0 = _tenant("corp-a")
    t1 = _tenant("corp-b")
    ts = TaskSpec(**_base_kwargs(
        workload=None, verifiers=None,
        tenants=[t0, t1],
    ))
    result = ts.effective_tenants()
    assert len(result) == 2
    assert result[0].id == "corp-a"
    assert result[1].id == "corp-b"


def test_multi_tenant_order_preserved():
    """Multi-tenant form: order of tenants in effective_tenants() matches the input list."""
    ids = ["z", "a", "m"]
    tenants = [_tenant(tid) for tid in ids]
    ts = TaskSpec(**_base_kwargs(workload=None, verifiers=None, tenants=tenants))
    assert [t.id for t in ts.effective_tenants()] == ids


def test_multi_tenant_provenance_has_tenants_array():
    """Multi-tenant dump: 'tenants' key is present and contains the tenant ids."""
    t0 = _tenant("corp-a")
    t1 = _tenant("corp-b")
    ts = TaskSpec(**_base_kwargs(
        workload=None, verifiers=None,
        tenants=[t0, t1],
    ))
    rec = ts.model_dump(mode="json", serialize_as_any=True)
    assert "tenants" in rec
    assert [t["id"] for t in rec["tenants"]] == ["corp-a", "corp-b"]


# ---------------------------------------------------------------------------
# XOR validation: both forms provided -> CompatibilityError
# ---------------------------------------------------------------------------

def test_both_forms_raises_compatibility_error():
    """Providing both top-level workload AND tenants raises CompatibilityError."""
    with pytest.raises(CompatibilityError):
        TaskSpec(**_base_kwargs(
            workload=_workload(),
            verifiers=_verifiers(),
            tenants=[_tenant("x")],
        ))


def test_tenants_with_top_level_data_source_raises():
    """Providing both tenants AND top-level data_source raises CompatibilityError."""
    with pytest.raises(CompatibilityError):
        TaskSpec(**_base_kwargs(
            workload=None,
            verifiers=None,
            data_source=_data_source(),
            tenants=[_tenant("x")],
        ))


# ---------------------------------------------------------------------------
# Single-tenant missing required fields
# ---------------------------------------------------------------------------

def test_single_form_missing_workload_raises():
    """Single-tenant form without workload raises CompatibilityError."""
    with pytest.raises(CompatibilityError):
        TaskSpec(**_base_kwargs(workload=None))


def test_single_form_missing_verifiers_raises():
    """Single-tenant form without verifiers raises CompatibilityError."""
    with pytest.raises(CompatibilityError):
        TaskSpec(**_base_kwargs(verifiers=None))


# ---------------------------------------------------------------------------
# Multi-tenant structural constraints
# ---------------------------------------------------------------------------

def test_empty_tenants_raises():
    """Empty tenants list raises CompatibilityError."""
    with pytest.raises(CompatibilityError):
        TaskSpec(**_base_kwargs(workload=None, verifiers=None, tenants=[]))


def test_duplicate_tenant_ids_raises():
    """Duplicate tenant ids raise CompatibilityError."""
    with pytest.raises(CompatibilityError):
        TaskSpec(**_base_kwargs(
            workload=None, verifiers=None,
            tenants=[_tenant("x"), _tenant("x")],
        ))


# ---------------------------------------------------------------------------
# Per-tenant wiring: check_wiring runs against the shared contract per tenant
# ---------------------------------------------------------------------------

def test_multi_tenant_mis_wired_verifier_raises_wiring_error():
    """A multi-tenant TaskSpec where one tenant's verifier is mis-wired against
    the SHARED contract fails at construction. TokenFaithfulnessVerifier requires
    TeacherForce; VLLMReference is a Generate-only oracle (no TeacherForce), so
    the per-tenant check_wiring loop must raise for the "bad" tenant."""
    good = _tenant("ok")
    bad = Tenant(
        id="mis-wired",
        workload=_workload(),
        verifiers=[TokenFaithfulnessVerifier(reference=VLLMReference(decode=Greedy()))],
    )
    with pytest.raises(WiringError):
        TaskSpec(**_base_kwargs(
            workload=None, verifiers=None,
            tenants=[good, bad],
        ))


# ---------------------------------------------------------------------------
# Shared single-default predicate (driver + provisioning use the SAME helper)
# ---------------------------------------------------------------------------

