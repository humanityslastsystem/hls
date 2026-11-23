"""Seed derivation helpers (the single source for per-tenant seeding).

A multi-tenant run needs a distinct, reproducible seed per tenant: the driver uses
it to draw each tenant's arrival schedule and think times. Dev/test membership is
not a draw (rows are assigned by a hash of their own id) and takes no seed.
"""
from __future__ import annotations

import zlib

#: The id of the synthesized single/default tenant. A single-tenant task normalizes
#: to exactly one ``Tenant(id=DEFAULT_TENANT_ID)`` (TaskSpec.effective_tenants), and
#: this is the ONE id whose ``tenant_seed`` is the unmodified base seed (below).
#: The single source for the "t0" literal; re-exported from ``hls.spec.tenant``.
DEFAULT_TENANT_ID = "t0"


def tenant_seed(base_seed: int, tenant_id: str) -> int:
    """Derive a deterministic, process-stable per-tenant seed.

    Reduces to exactly ``base_seed`` when ``tenant_id == "t0"`` (the single-default-
    tenant case), so a single-tenant task's arrival schedule is drawn from the task
    seed itself.  For any other id, combines the base seed with a stable crc32 of the
    id so each tenant gets a distinct (but reproducible) seed.

    Uses ``zlib.crc32`` rather than the builtin ``hash()`` because Python randomizes
    ``hash()`` per process via ``PYTHONHASHSEED``; crc32 is deterministic across runs,
    which reproducible arrival sampling requires.
    """
    if tenant_id == DEFAULT_TENANT_ID:
        return base_seed
    return (base_seed * 2654435761 + zlib.crc32(tenant_id.encode())) & 0xFFFFFFFF
