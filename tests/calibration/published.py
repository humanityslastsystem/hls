"""A calibration instance in a store, as the orchestrator and its stages leave it: the plan
created under a lease, and stages published with their outputs."""

from __future__ import annotations

from collections.abc import Mapping

from hls.pipeline.plan import Plan, PlanStore
from hls.store.lock import Holder, instance_lease
from hls.store.manifest import publish_manifest
from hls.store.posix import PosixStore
from tests.verify_stage.test_stage import _manifest


def create(store: PosixStore, plan: Plan) -> Plan:
    """*plan* written as its instance's first revision, which is returned."""
    handle = instance_lease(store, plan.layout, ttl_sec=30).try_acquire(Holder.here("test"))
    return PlanStore(store, plan.layout).create(plan, handle)


def publish(store: PosixStore, plan: Plan, stage: str, outputs: Mapping[str, tuple[str, bytes]]) -> dict[str, str]:
    """Publish *stage* of *plan*'s instance with *outputs*, each an output's name to its path
    under the first attempt's ``outputs/`` and its bytes; return each output's key."""
    keys = {}
    for name, (path, data) in outputs.items():
        keys[name] = plan.layout.attempt_file(stage, 1, f"outputs/{path}")
        store.write_bytes(keys[name], data)
    publish_manifest(store, plan.layout, _manifest(plan, store, stage, keys))
    return keys
