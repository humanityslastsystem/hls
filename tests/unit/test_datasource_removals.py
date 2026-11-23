"""The removal list: ``DataSource.removed_rows``.

A task removes from its served workload the rows its calibration's own serving stack
ran to the output cap, named by each row's own id. The list is part of the workload
identity, since a different request set is a different calibration, and the drive
plan drops the listed sessions before the arrival schedule is drawn, so the workload
a drive offers is the sessions that remain; a listed row takes its whole session.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.prompt import JinjaPromptTemplate
from hls.spec.datasets import HFDataset, Split, WildChatDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.drivers import (Driver, load_mounted_sessions, plan_sessions,
                              remove_listed_rows)
from hls.spec.tenant import Tenant
from hls.spec.workload import (AllAtOnce, GammaInterTurn, NoDelay, NoSLO, Poisson,
                               Uncapped, Workload)
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC


def _dataset() -> HFDataset:
    return HFDataset(path="cais/mmlu", subset="abstract_algebra", revision="main",
                     source_split="test",
                     prompt=JinjaPromptTemplate(template="{{ question }}"),
                     target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))


def _source(removed: tuple[str, ...] = ()) -> DataSource:
    return DataSource(dataset=_dataset(),
                      split=SplitSpec(dev=Count(value=1), test=Count(value=4)),
                      removed_rows=removed)


def _chat_source(removed: tuple[str, ...] = (), n: int = 6) -> DataSource:
    return DataSource(dataset=WildChatDataset(revision="r", language="English", max_turns=6),
                      split=SplitSpec(dev=Count(value=1), test=Count(value=n)),
                      removed_rows=removed)


def _chat_rows(tmp_path: Path, n: int = 6, turns: int = 3, tenant: str = "t0") -> Path:
    """Write ``tenant``'s rows where the drive loop reads them, and return the mount
    root they sit under."""
    rows = [{"conversation_hash": f"c{i}", "user_turns": ["a", "b", "c"][:turns],
             "assistant_turns": ["x", "y", "z"][:turns]} for i in range(n)]
    for split in ("dev", "test"):
        split_dir = tmp_path / "tenants" / tenant / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "rows.json").write_text(json.dumps(rows))
    return tmp_path


def _workload(trim: bool = False) -> Workload:
    """Drawn arrivals and think-times, so a schedule actually depends on the session
    count and the trim has something to cut."""
    return Workload(arrival=Poisson(rate_per_sec=2.0),
                    inter_turn_delay=GammaInterTurn(shape=1.0, scale=1.0, max_gap=3.0),
                    output_length_cap=Uncapped(), slo=NoSLO(), trim_rampdown=trim)


class _EchoClient:
    """Answers every request with one token, instantly."""

    async def submit(self, env, *, attempt=None):
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id, contents=[TextPayload(text="r", tokens=[1])]))


def test_removed_rows_must_be_sorted_and_unique():
    """One form for the list, so the fingerprint has one value for one set."""
    with pytest.raises(ValueError, match="sorted"):
        _source(("2", "0"))
    with pytest.raises(ValueError, match="repeats"):
        _source(("0", "0"))


def test_the_removal_list_is_part_of_the_workload_identity():
    """A different request set is a different calibration: the list moves the
    fingerprint, and so does changing it."""
    from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
    from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
    from hls.spec.contracts import RequestResponseContract
    from hls.spec.taskspec import TaskSpec
    from hls.spec.wire import OpenAIChatProtocol
    from hls.verify import LetterMatchMetric, MetricVerifier
    from hls.verify.calibration import workload_fingerprint

    def task(removed):
        return TaskSpec(
            id="T", model=ModelSpec(base_checkpoint="m", arch="TX"),
            hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
            contract=RequestResponseContract(
                path="/v1/chat/completions", input_modalities=frozenset({"text"}),
                output_modalities=frozenset({"text"}), protocol=OpenAIChatProtocol()),
            workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                              output_length_cap=Uncapped(), slo=NoSLO()),
            data_source=_source(removed),
            verifiers=[MetricVerifier(metric=LetterMatchMetric())],
            scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))

    none, one, two = (workload_fingerprint(task(r)) for r in ((), ("0",), ("0", "1")))
    assert len({none, one, two}) == 3


def test_the_loaders_return_the_whole_split(tmp_path, monkeypatch):
    """The removal is not a load-time filter: the rows come back whole and in order,
    so a tool keying earlier captures against the split can still name
    a row the list has since removed."""
    import hls.spec.datasets as ds_mod

    monkeypatch.setattr(ds_mod, "MOUNT_ROOT", _chat_rows(tmp_path))
    tenant = Tenant(id="t0", data_source=_chat_source(("c1",)),
                    workload=_workload(), verifiers=[])
    (_t, items), = load_mounted_sessions([tenant], Split.TEST)
    assert [i.request_id for i in items] == [f"c{i}" for i in range(6)]


def test_listed_rows_leave_whole_and_the_rest_keep_their_order(tmp_path, monkeypatch):
    """A listed row takes its whole session, the others keep their order, a listed id
    that is not in the split is a no-op, and a tenant left with nothing drops out."""
    import hls.spec.datasets as ds_mod

    monkeypatch.setattr(ds_mod, "MOUNT_ROOT", _chat_rows(tmp_path))
    tenant = Tenant(id="t0", data_source=_chat_source(("c0", "c4", "not-in-split")),
                    workload=_workload(), verifiers=[])
    loaded = load_mounted_sessions([tenant], Split.TEST)
    (_t, kept), = remove_listed_rows(loaded)
    assert [i.request_id for i in kept] == ["c1", "c2", "c3", "c5"]
    assert all(len(i.turns) == 3 for i in kept)
    emptied = Tenant(id="t0", data_source=_chat_source(tuple(f"c{i}" for i in range(6))),
                     workload=_workload(), verifiers=[])
    assert remove_listed_rows(load_mounted_sessions([emptied], Split.TEST)) == []


def test_the_schedule_is_drawn_over_the_sessions_that_remain(tmp_path, monkeypatch):
    """The workload a drive offers is the post-removal list: the schedule and the
    ramp-down budget are drawn for exactly those sessions, keyed by their index in
    that list, so a removed row leaves nothing behind in the schedule."""
    import hls.spec.datasets as ds_mod

    monkeypatch.setattr(ds_mod, "MOUNT_ROOT", _chat_rows(tmp_path))

    def plan(source):
        tenant = Tenant(id="t0", data_source=source, workload=_workload(trim=True),
                        verifiers=[])
        loaded = remove_listed_rows(load_mounted_sessions([tenant], Split.TEST))
        return plan_sessions(loaded, drive_seed=7)

    (bare, _), (with_list, budget) = plan(_chat_source()), plan(_chat_source(("c0", "c4")))
    (_t, items, starts), = with_list
    (_t, bare_items, bare_starts), = bare
    assert [i.request_id for i in items] == ["c1", "c2", "c3", "c5"]
    assert len(starts) == 4 and len(bare_starts) == 6
    # Four sessions drawn from the same seed: the same schedule as any four-session
    # draw, i.e. the first four arrivals of the six-session one.
    assert list(starts) == list(bare_starts[:4])
    assert set(budget) <= {("t0", s) for s in range(4)}


def test_the_drive_serves_only_the_remaining_sessions(tmp_path, monkeypatch):
    """End to end through the driver: a listed session is never sent, and every
    remaining session is served with all its turns."""
    import hls.spec.datasets as ds_mod

    monkeypatch.setattr(ds_mod, "MOUNT_ROOT", _chat_rows(tmp_path))

    def drive(source):
        tenant = Tenant(id="t0", data_source=source, workload=_workload(),
                        verifiers=[])
        events: list[TraceEvent] = []
        driver = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
        asyncio.run(driver._drive(_EchoClient(), events.append, Split.TEST, tenants=[tenant], seed=7))
        turns: dict[str, int] = {}
        for ev in events:
            turns[ev.session_id] = turns.get(ev.session_id, 0) + 1
        return turns

    assert drive(_chat_source()) == {f"c{i}": 3 for i in range(6)}
    assert drive(_chat_source(("c2",))) == {f"c{i}": 3 for i in range(6) if i != 2}
