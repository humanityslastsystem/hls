"""Unit wiring test for tasks.multitenant_slo.task.

Confirms the TASK spec builds with 2+ distinct tenants, each with a non-None
data_source and at least one verifier, and that per-tenant SLOs are all distinct
(the multi-SLO story is real).
"""
from __future__ import annotations


def test_task_builds_with_tenants():
    from tasks.multitenant_slo.task import TASK

    tids = [t.id for t in TASK.effective_tenants()]
    assert len(tids) >= 2 and len(set(tids)) == len(tids), (
        f"expected 2+ unique tenant ids; got {tids}"
    )
    slos = {t.id: t.workload.slo for t in TASK.effective_tenants()}
    assert len(slos) == len(tids), "slos dict must have one entry per tenant"


def test_each_tenant_has_data_source_and_verifiers():
    from tasks.multitenant_slo.task import TASK

    for t in TASK.effective_tenants():
        assert t.data_source is not None, (
            f"tenant {t.id!r} has no data_source"
        )
        assert len(t.verifiers) >= 1, (
            f"tenant {t.id!r} has no verifiers"
        )


def test_tenant_slos_are_distinct():
    """Each tenant must carry a different StreamingLatencySLO so the multi-SLO story is real."""
    from tasks.multitenant_slo.task import TASK

    slo_tuples = []
    for t in TASK.effective_tenants():
        slo = t.workload.slo
        # Collect (ttft_ms, tpot_ms) pairs; they must not all be identical.
        slo_tuples.append((slo.ttft_ms, slo.tpot_ms))
    assert len(set(slo_tuples)) > 1, (
        f"all tenants share the same SLO parameters {slo_tuples!r}; "
        "each tenant must have a distinct SLO"
    )


def test_task_id_and_metadata():
    from tasks.multitenant_slo.task import TASK

    assert TASK.id == "multitenant_slo"
    assert TASK.metadata.get("theme") == "multitenant-slo"


def test_contract_is_http_sse_with_token_ids():
    """Contract must be http-sse (streaming SLO + token-level checks); token ids
    are always emitted on the wire contract now, so no protocol flag is needed."""
    from tasks.multitenant_slo.task import CONTRACT

    assert CONTRACT.transport == "http-sse", (
        f"expected http-sse transport; got {CONTRACT.transport!r}"
    )
