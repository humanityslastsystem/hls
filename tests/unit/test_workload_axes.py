from hls.core.poly import dump_spec
from hls.spec.workload import (
    AllAtOnce, ArrivalProcess, InterTurnDelay, NoDelay,
    OutputLengthCap, Uncapped, Workload,
)


def _wl():
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=Uncapped(), slo=_slo())


def test_arrival_all_at_once_schedule():
    offs = AllAtOnce().schedule(5, seed=1)
    assert offs == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_inter_turn_no_delay():
    assert NoDelay().delay(3, seed=(11, 0)) == 0.0


def test_uncapped_returns_default_cap():
    # Uncapped is not None: it resolves to the 1M default ceiling so the artifact
    # and the reference oracle always decode under the same, reproducible budget.
    from hls.spec.workload import DEFAULT_MAX_TOKENS
    assert Uncapped().max_tokens() == DEFAULT_MAX_TOKENS == 1_000_000


def test_workload_dumps_with_kinds():
    rec = dump_spec(_wl())
    assert "seed" not in rec   # seed lives on TaskSpec.seed, not on Workload
    assert rec["arrival"]["kind"] == "AllAtOnce"
    assert rec["inter_turn_delay"]["kind"] == "NoDelay"
    assert "tenancy" not in rec
    assert "adapter_mix" not in rec


def test_axis_abcs_present():
    assert issubclass(AllAtOnce, ArrivalProcess)
    assert issubclass(NoDelay, InterTurnDelay)
    assert issubclass(Uncapped, OutputLengthCap)


def _slo():
    from hls.spec.workload import NoSLO
    return NoSLO()
