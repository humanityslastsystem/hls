import asyncio

from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.drivers import Driver
from hls.spec.tenant import Tenant
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, ShortOutputCap, Workload)
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from tests.support.inline_dataset import InlineMultiTurnDataset


def _workload():
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=ShortOutputCap(cap_tokens=8),
                    slo=NoSLO())


def _split(name):
    from hls.spec.datasets import Split
    return Split(name)


class _EchoClient:
    base_url = "http://x"

    async def submit(self, env, *, attempt=None):
        from hls.core.trace import ArtifactResponse
        from hls.core.payloads import Envelope, TextPayload
        return ArtifactResponse(output=Envelope(request_id=env.request_id,
                                                contents=[TextPayload(text="A", tokens=[1])]))


def _drive(ds) -> list:
    """Drive a single Tenant("t0") over *ds* and return its events. The SplitSpec is a
    placeholder (ignored at drive time for in-memory datasets)."""
    tenants = [Tenant(
        id="t0",
        data_source=DataSource(dataset=ds, split=SplitSpec(dev=Count(value=1), test=Count(value=2))),
        workload=_workload(),
        verifiers=[],
    )]
    events = []
    driver = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    asyncio.run(driver._drive(_EchoClient(), events.append, _split("dev"), tenants=tenants, seed=0))
    return events


def test_two_level_loop_emits_event_per_turn_with_planned_send():
    ds = InlineMultiTurnDataset(name="t", conversations=[("c0", [("u1", "g1"), ("u2", "g2")])])
    events = _drive(ds)
    assert len(events) == 2                       # one event per turn
    assert events[0].session_id == "c0" and events[0].turn_index == 0
    # AllAtOnce (offset 0) -> planned send is the run-start anchor, in the ABSOLUTE
    # monotonic frame (same as t_send), so scheduler slack is small + non-negative.
    ps = events[0].t_planned_send
    assert ps is not None and 0.0 <= events[0].t_send - ps < 1.0


def test_followup_turn_has_no_planned_send_but_stitches_history():
    ds = InlineMultiTurnDataset(name="t", conversations=[("c0", [("u1", "g1"), ("u2", "g2")])])
    events = _drive(ds)
    # Turn 1 is a reactive follow-up: no open-loop planned-send.
    assert events[1].turn_index == 1 and events[1].t_planned_send is None
    # History stitching: the second request carries the prior user+assistant turns.
    msgs = events[1].input.metadata["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "u1" and msgs[2]["content"] == "u2"


def test_per_turn_ground_truth_and_output_cap_stamped():
    ds = InlineMultiTurnDataset(name="t", conversations=[("c0", [("u1", "g1"), ("u2", "g2")])])
    events = _drive(ds)
    assert events[0].ground_truth == "g1" and events[1].ground_truth == "g2"
    # ShortOutputCap(cap_tokens=8) rides into every request's metadata.
    assert events[0].input.metadata["max_tokens"] == 8


def test_build_engine_carries_config():
    eng = Driver(request_timeout_sec=12.0)
    assert isinstance(eng, Driver)
    assert eng.request_timeout_sec == 12.0
    assert not {"tenants", "contract", "seed"} & set(vars(eng))
