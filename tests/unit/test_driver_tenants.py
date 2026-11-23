"""Tests for Driver driving over effective_tenants.

Two scenarios:
  - Single-tenant drive: one synthesized Tenant("t0").
  - Multi-tenant drive: two tenants share one shared t0 and stamp distinct
    tenant_ids; all sessions arrive on the same wall-clock anchor (no per-tenant
    clock reset).

Also tests the open-loop arrival clock via the driver.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hls.core.payloads import Envelope, TextPayload
from hls.core.seeds import tenant_seed
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.spec.datasets import HFDataset, Split
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.drivers import Driver
from hls.spec.tenant import Tenant
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Poisson, Uncapped, Workload
from hls.prompt import JinjaPromptTemplate
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _wl():
    return Workload(
        arrival=AllAtOnce(),
        inter_turn_delay=NoDelay(),
        output_length_cap=Uncapped(),
        slo=NoSLO(),
    )


def _wl_poisson():
    """A workload whose arrival schedule actually depends on the seed (unlike the
    AllAtOnce default), so a determinism test exercises the RNG path."""
    return Workload(
        arrival=Poisson(rate_per_sec=2.0),
        inter_turn_delay=NoDelay(),
        output_length_cap=Uncapped(),
        slo=NoSLO(),
    )


def _hf_dataset():
    return HFDataset(
        path="stub",
        source_split="test",
        id_column="q",
        prompt=JinjaPromptTemplate(template="{{ q }}"),
        target=JinjaPromptTemplate(template="{{ q }}"),
    )


def _write_rows(mount_dir: Path, split: str, rows: list[dict]) -> None:
    """Write rows.json for a split under mount_dir/<split>/rows.json."""
    split_dir = mount_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "rows.json").write_text(json.dumps(rows))


class _EchoClient:
    """Stand-in ArtifactClient. Echoes input back with text='A'."""
    async def submit(self, env: Envelope, *, attempt=None) -> ArtifactResponse:
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id,
            contents=[TextPayload(text="A")]))


# ---------------------------------------------------------------------------
# tenant_seed (the shared helper in hls.core.seeds, used by both driver + provisioning)
# ---------------------------------------------------------------------------

def test_tenant_seed_t0_equals_base_seed():
    """tenant_seed(seed, 't0') must equal seed."""
    for seed in (0, 1, 5, 42, 99):
        assert tenant_seed(seed, "t0") == seed, (
            f"tenant_seed({seed!r}, 't0') must return {seed!r}")


def test_tenant_seed_other_id_differs():
    """tenant_seed with a non-t0 id produces a different value than the raw seed."""
    seed = 7
    assert tenant_seed(seed, "corp-a") != seed
    assert tenant_seed(seed, "corp-a") == tenant_seed(seed, "corp-a")  # deterministic
    assert tenant_seed(seed, "corp-a") != tenant_seed(seed, "corp-b")


# ---------------------------------------------------------------------------
# Single-tenant drive
# ---------------------------------------------------------------------------

def _redirect_items_at(monkeypatch, tmp_path):
    """Point the container mount root at ``tmp_path/data``, so every tenant's rows are
    read from ``tmp_path/data/tenants/<id>/<split>`` instead of the real mount."""
    import hls.spec.datasets as _ds_mod
    monkeypatch.setattr(_ds_mod, "MOUNT_ROOT", tmp_path / "data")


def test_single_tenant_drive_stamps_t0(monkeypatch, tmp_path):
    """Single-tenant drive (tenants=[Tenant('t0', ...)]) stamps tenant_id='t0' on all events."""
    _redirect_items_at(monkeypatch, tmp_path)
    _write_rows(tmp_path / "data" / "tenants" / "t0", "test", [{"q": "A"}, {"q": "B"}])

    ds = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=2)))
    tenant = Tenant(id="t0", data_source=ds, workload=_wl(), verifiers=[])

    eng = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    tenants = [tenant]

    events: list[TraceEvent] = []
    asyncio.run(eng._drive(_EchoClient(), events.append, Split.TEST, tenants=tenants, seed=0))

    assert len(events) == 2
    for ev in events:
        assert ev.input.metadata["tenant_id"] == "t0", (
            f"expected tenant_id='t0', got {ev.input.metadata.get('tenant_id')!r}")
    # Each event carries the tenant, session and turn it belongs to, and nothing of adapters.
    for ev in events:
        assert {"tenant_id", "session_id", "turn_index"} <= set(ev.input.metadata)
        assert "adapter_id" not in ev.input.metadata


def test_single_tenant_drive_stamps_no_adapter(monkeypatch, tmp_path):
    """A single-tenant drive stamps no adapter key on its events: a task serves the base model."""
    _redirect_items_at(monkeypatch, tmp_path)
    _write_rows(tmp_path / "data" / "tenants" / "t0", "test", [{"q": "X"}])

    ds = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=1)))
    tenant = Tenant(id="t0", data_source=ds, workload=_wl(), verifiers=[])

    eng = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    tenants = [tenant]

    events: list[TraceEvent] = []
    asyncio.run(eng._drive(_EchoClient(), events.append, Split.TEST, tenants=tenants, seed=0))

    assert "adapter_id" not in events[0].input.metadata


# ---------------------------------------------------------------------------
# Multi-tenant drive: shared t0, distinct tenant_ids, per-tenant mount path
# ---------------------------------------------------------------------------

def test_two_tenants_stamp_distinct_tenant_ids(monkeypatch, tmp_path):
    """Two tenants produce events stamped with their respective tenant_ids."""
    mount_a = tmp_path / "data" / "tenants" / "corp-a"
    mount_b = tmp_path / "data" / "tenants" / "corp-b"
    _write_rows(mount_a, "test", [{"q": "QA1"}, {"q": "QA2"}])
    _write_rows(mount_b, "test", [{"q": "QB1"}])

    ds_a = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=2)))
    ds_b = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=1)))

    t_a = Tenant(id="corp-a", data_source=ds_a, workload=_wl(), verifiers=[])
    t_b = Tenant(id="corp-b", data_source=ds_b, workload=_wl(), verifiers=[])

    eng = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    tenants = [t_a, t_b]

    _redirect_items_at(monkeypatch, tmp_path)
    events: list[TraceEvent] = []
    asyncio.run(eng._drive(_EchoClient(), events.append, Split.TEST, tenants=tenants, seed=0))

    tenant_ids = {ev.input.metadata["tenant_id"] for ev in events}
    assert tenant_ids == {"corp-a", "corp-b"}, (
        f"expected {{corp-a, corp-b}}, got {tenant_ids}")
    corp_a_events = [ev for ev in events if ev.input.metadata["tenant_id"] == "corp-a"]
    corp_b_events = [ev for ev in events if ev.input.metadata["tenant_id"] == "corp-b"]
    assert len(corp_a_events) == 2, f"corp-a should have 2 events; got {len(corp_a_events)}"
    assert len(corp_b_events) == 1, f"corp-b should have 1 event; got {len(corp_b_events)}"


def test_multi_tenant_named_t0_reads_tenants_subpath(monkeypatch, tmp_path):
    """In a MULTI-tenant task, a tenant literally named "t0" must read from
    /data/tenants/t0/<split>, NOT /data/<split>.

    Provisioning mounts EVERY tenant -- including one named "t0" -- under
    /data/tenants/<id>, and the driver must read where provisioning wrote.

    Here we write "t0"'s rows ONLY under /data/tenants/t0/test (its real mount) and
    DO NOT create /data/test at all. If the driver chose /data for this tenant,
    items() would FileNotFoundError. Passing proves the driver read from the
    tenants/t0 subpath.
    """
    mount_t0 = tmp_path / "data" / "tenants" / "t0"
    mount_a = tmp_path / "data" / "tenants" / "corp-a"
    _write_rows(mount_t0, "test", [{"q": "T0Q1"}, {"q": "T0Q2"}])
    _write_rows(mount_a, "test", [{"q": "AQ1"}])
    # Intentionally NO /data/test written: the only "t0" rows live under tenants/t0.

    ds_t0 = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=2)))
    ds_a = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=1)))

    t_t0 = Tenant(id="t0", data_source=ds_t0, workload=_wl(), verifiers=[])
    t_a = Tenant(id="corp-a", data_source=ds_a, workload=_wl(), verifiers=[])

    eng = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    tenants = [t_t0, t_a]   # every tenant reads tenants/<id>, "t0" included

    _redirect_items_at(monkeypatch, tmp_path)
    events: list[TraceEvent] = []
    asyncio.run(eng._drive(_EchoClient(), events.append, Split.TEST, tenants=tenants, seed=0))

    tenant_ids = {ev.input.metadata["tenant_id"] for ev in events}
    assert tenant_ids == {"t0", "corp-a"}, (
        f"expected {{t0, corp-a}}, got {tenant_ids}")
    t0_events = [ev for ev in events if ev.input.metadata["tenant_id"] == "t0"]
    assert len(t0_events) == 2, (
        f"t0 should have 2 events read from tenants/t0/test; got {len(t0_events)}")
    # The rendered prompt proves the data came from tenants/t0 (T0Q*), not /data.
    t0_prompts = {ev.input.first_of("text").text for ev in t0_events}
    assert t0_prompts == {"T0Q1", "T0Q2"}, (
        f"t0 events must carry the tenants/t0 rows; got {t0_prompts}")


def test_two_tenants_share_single_t0(monkeypatch, tmp_path):
    """All sessions from both tenants schedule against ONE shared t0 (no per-tenant clock reset).

    Proof: AllAtOnce schedules t0+0 for all sessions. With a shared t0, all sessions
    have t_planned_send within a small epsilon of each other. If per-tenant clock
    resets were happening, the second tenant's t_planned_send would be ~epsilon larger,
    but we can't observe that directly -- instead we verify that all t_planned_send
    values are non-None (for turn 0) and are within a 1-second window (all fired
    from the same anchor).
    """
    mount_a = tmp_path / "data" / "tenants" / "corp-a"
    mount_b = tmp_path / "data" / "tenants" / "corp-b"
    _write_rows(mount_a, "test", [{"q": "A"}])
    _write_rows(mount_b, "test", [{"q": "B"}])

    ds_a = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=1)))
    ds_b = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=1)))

    t_a = Tenant(id="corp-a", data_source=ds_a, workload=_wl(), verifiers=[])
    t_b = Tenant(id="corp-b", data_source=ds_b, workload=_wl(), verifiers=[])

    eng = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    tenants = [t_a, t_b]

    _redirect_items_at(monkeypatch, tmp_path)
    events: list[TraceEvent] = []
    asyncio.run(eng._drive(_EchoClient(), events.append, Split.TEST, tenants=tenants, seed=0))

    # All events are turn-0 (single-turn items), so all have t_planned_send set.
    turn0_events = [ev for ev in events if ev.turn_index == 0]
    assert len(turn0_events) == 2
    planned = [ev.t_planned_send for ev in turn0_events]
    assert all(p is not None for p in planned), "turn-0 events must have t_planned_send"
    # All planned sends should be within a 1s window (shared t0, AllAtOnce, near-instant test).
    assert max(planned) - min(planned) < 1.0, (
        f"planned sends span > 1s; per-tenant clock reset suspected: {planned}")


# ---------------------------------------------------------------------------
# The whole multi-tenant firing pattern is a pure function of the seed.
# ---------------------------------------------------------------------------

def test_multi_tenant_firing_pattern_is_deterministic_under_seed(monkeypatch, tmp_path):
    """The full multi-tenant firing pattern is reproducible from the seed alone.

    Drives a 2-tenant spec with STOCHASTIC Poisson arrivals (so the schedule really
    depends on the seed) and captures every session's PLANNED send, keyed by
    (tenant_id, request_id). The two tenants' rows carry the same ids, so their
    request_ids collide -- the tenant_id is what disambiguates, which is
    exactly the cross-tenant identity the rest of the pipeline relies on.

    The wall clock is pinned (harness_now -> constant) and the open-loop sleep is a
    no-op, so each planned send is anchor + schedule-offset with the anchor fixed:
    two same-seed runs must be byte-identical, a different seed must differ, and the
    test neither sleeps for the Poisson gaps nor depends on real time.
    """
    mount_a = tmp_path / "data" / "tenants" / "corp-a"
    mount_b = tmp_path / "data" / "tenants" / "corp-b"
    _write_rows(mount_a, "test", [{"q": f"Q{i}"} for i in range(4)])
    _write_rows(mount_b, "test", [{"q": f"Q{i}"} for i in range(3)])

    ds_a = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=4)))
    ds_b = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=3)))
    t_a = Tenant(id="corp-a", data_source=ds_a, workload=_wl_poisson(), verifiers=[])
    t_b = Tenant(id="corp-b", data_source=ds_b, workload=_wl_poisson(), verifiers=[])

    _redirect_items_at(monkeypatch, tmp_path)
    monkeypatch.setattr("hls.spec.drivers.harness_now", lambda: 1000.0)

    async def _instant(self, _t0: float, stop=None) -> None:
        return None

    monkeypatch.setattr(Driver, "_sleep_until", _instant)  # skip the open-loop wait

    def _run(seed: int) -> dict[tuple[str, str], float]:
        events: list[TraceEvent] = []
        driver = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
        asyncio.run(driver._drive(_EchoClient(), events.append, Split.TEST, tenants=[t_a, t_b], seed=seed))
        return {
            (ev.input.metadata["tenant_id"], ev.request_id): ev.t_planned_send
            for ev in events if ev.turn_index == 0
        }

    pattern_1 = _run(20250622)
    pattern_2 = _run(20250622)
    pattern_other = _run(20250623)

    # Sanity: 4 + 3 sessions, both tenants present, and a real (non-degenerate) spread.
    assert len(pattern_1) == 7
    assert sum(1 for tid, _ in pattern_1 if tid == "corp-a") == 4
    assert sum(1 for tid, _ in pattern_1 if tid == "corp-b") == 3
    spread = sorted(pattern_1.values())
    assert spread[0] != spread[-1], "Poisson schedule should spread sessions over time"

    # Same seed -> byte-identical firing pattern; a different seed -> a different one.
    assert pattern_1 == pattern_2, "same seed must reproduce the exact firing pattern"
    assert pattern_1 != pattern_other, "a different seed must change the firing pattern"


# ---------------------------------------------------------------------------
# Tenant with data_source=None produces no sessions (graceful skip)
# ---------------------------------------------------------------------------

def test_tenant_without_data_source_produces_no_sessions(monkeypatch, tmp_path):
    """A tenant with data_source=None produces no sessions -- skipped gracefully."""
    mount = tmp_path / "data" / "tenants" / "corp-a"
    _write_rows(mount, "test", [{"q": "Q1"}])

    ds_a = DataSource(dataset=_hf_dataset(), split=SplitSpec(dev=Count(value=1), test=Count(value=1)))
    t_a = Tenant(id="corp-a", data_source=ds_a, workload=_wl(), verifiers=[])
    t_b = Tenant(id="corp-b", data_source=None, workload=_wl(), verifiers=[])

    eng = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    tenants = [t_a, t_b]

    _redirect_items_at(monkeypatch, tmp_path)
    events: list[TraceEvent] = []
    asyncio.run(eng._drive(_EchoClient(), events.append, Split.TEST, tenants=tenants, seed=0))

    tenant_ids = {ev.input.metadata["tenant_id"] for ev in events}
    assert tenant_ids == {"corp-a"}, (
        f"corp-b has no data_source; only corp-a events expected; got {tenant_ids}")
