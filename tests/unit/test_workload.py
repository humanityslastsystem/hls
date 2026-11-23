from hls.spec.workload import (
    AllAtOnce, NoDelay, Uncapped, NoSLO, Workload,
)
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import Trace, TraceEvent


def test_all_at_once_schedule_is_zero_spaced():
    sched = AllAtOnce().schedule(5, seed=0)
    assert sched == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_tenancy_and_adapter_mix_not_in_model_fields():
    assert "tenancy" not in Workload.model_fields
    assert "adapter_mix" not in Workload.model_fields


def test_no_slo_apply_always_passes_with_full_credit():
    """NoSLO never gates and credits every request: correctness (the verdict)
    alone bounds goodput. Latency-gating tasks use StreamingLatencySLO."""
    slo = NoSLO()
    env = Envelope(request_id="r1", contents=[TextPayload(text="x")])
    ev = TraceEvent(request_id="r1", input=env, output=env, t_send=0.0, t_recv=1.0)
    trace = Trace(events=[ev])
    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                        output_length_cap=Uncapped(), slo=slo)
    result = slo.apply(trace, workload)
    assert result.aggregate_passed is True
    assert result.per_event_credit["r1"] == 1.0


def test_identity_subclasses_describe():
    assert NoDelay().describe() == "no inter-turn delay"


def test_prompt_fields_omits_tenancy_and_adapter_mix():
    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                        output_length_cap=Uncapped(), slo=NoSLO())
    fields = workload.prompt_fields()
    assert "tenancy" not in fields
    assert "adapter_mix" not in fields
    assert set(fields) == {"sessions", "requests", "request_rate", "think_time",
                           "output_cap", "slo"}
    # An offline batch of single-turn sessions has no rate and no turn to wait
    # before, so the brief states the arrival alone.
    assert fields["sessions"] == "all sessions dispatched at once (offline batch)"
    assert fields["requests"] is None
    assert fields["request_rate"] is None
    assert fields["think_time"] is None


def test_prompt_fields_state_sessions_and_requests_separately():
    """The arrival process schedules sessions; the brief converts to requests with the
    declared turns per session and says so in both units, so a contestant sizing for
    load reads the request rate it will meet rather than the session rate."""
    from hls.spec.workload import Gamma, GammaInterTurn
    workload = Workload(arrival=Gamma(shape=1.0, scale=11.968969),
                        mean_turns_per_session=2.3914,
                        inter_turn_delay=GammaInterTurn(shape=0.4057, scale=735.8009,
                                                        max_gap=1800.0),
                        output_length_cap=Uncapped(), slo=NoSLO())
    fields = workload.prompt_fields()
    assert fields["sessions"] == ("gamma arrivals (shape=1, scale=11.969), "
                                  "0.08355 sessions/s")
    assert fields["requests"] == "2.391 turns/session, 0.1998 requests/s"
    assert abs(fields["request_rate"] - 0.1998) < 1e-4
    assert fields["think_time"].startswith("gamma think-time")


def test_single_turn_tenant_states_one_turn_per_session():
    from hls.spec.workload import Poisson
    workload = Workload(arrival=Poisson(rate_per_sec=2.0), inter_turn_delay=NoDelay(),
                        output_length_cap=Uncapped(), slo=NoSLO())
    fields = workload.prompt_fields()
    assert fields["sessions"] == "poisson arrivals, 2 sessions/s"
    assert fields["requests"] == "1 turn/session, 2 requests/s"
    assert fields["think_time"] is None
