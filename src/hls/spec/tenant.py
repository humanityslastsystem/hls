"""Tenant -- one self-contained slice of a multi-tenant task (the config plane).

A Tenant bundles the data, traffic pattern + SLO, and correctness gates for a
single virtual tenant sharing a deployed artifact.  For exactly one tenant,
prefer the top-level single-tenant form (data_source / workload / verifiers on
the TaskSpec): TaskSpec.effective_tenants() synthesizes a Tenant(id="t0") from
that triple, so the task need not author a Tenant directly.  TaskSpec.tenants is
primarily for 2+ tenants, though a one-element list is also accepted.
"""
from __future__ import annotations

import re

from pydantic import field_validator

from hls.core.poly import Spec
from hls.core.seeds import DEFAULT_TENANT_ID
from hls.spec.datasource import DataSource
from hls.spec.workload import Workload
from hls.verify.base import Verifier

__all__ = ["Tenant", "DEFAULT_TENANT_ID"]

# Tenant.id names the directory its rows lie in, tenants/<id>, in the provision stage's
# cache entry and under every role's /data (hls.provision.rows.tenant_subpath). The charset
# keeps the id one plain path component, and "." / ".." (which the charset alone still
# admits) are refused, so no id reaches outside tenants/.
_TENANT_ID_RE = re.compile(r"[A-Za-z0-9._-]+")


class Tenant(Spec):
    """One self-contained slice of a task: its data, its traffic + SLO, and how
    its responses are judged.  A tenant spawns many runtime sessions; it does not
    store them.  Single-tenant tasks normalize to exactly one Tenant (id="t0")."""

    id: str
    data_source: "DataSource | None" = None
    workload: Workload
    verifiers: list[Verifier]

    @field_validator("id")
    @classmethod
    def _safe_id(cls, v: str) -> str:
        if not _TENANT_ID_RE.fullmatch(v):
            raise ValueError(
                f"Tenant.id {v!r} must be non-empty and match [A-Za-z0-9._-]+: it names the "
                "tenants/<id> directory of the tenant's rows, which must stay one path component")
        if v in (".", ".."):
            raise ValueError(
                f"Tenant.id {v!r} is not allowed: it traverses the tenants/<id> directory")
        return v
