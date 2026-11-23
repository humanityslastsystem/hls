"""Tests for hls.baseline_credit -- the measured constant a speedup-scored task
subtracts, and the refusals that keep a run from grading against the wrong one.

The scorer under test is the real ``SloLatencyScorer`` (it grades against a baseline
and takes an injected ``baseline_log_credit``), so the artifact's mean is checked
against the same number the scorer would divide by. Measuring the drives is the credit
pipeline's (``tests/credit``); here the model, the gate against a task and the pool.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.core.poly import dump_spec
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.prompt import JinjaPromptTemplate
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit, SloLatencyScorer
from hls.spec.baseline import Baseline
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.resources import BaselineHardware
from hls.spec.workload import (NoDelay, Poisson, StreamingLatencySLO,
                               Uncapped, Workload)
from hls.verify import LetterMatchMetric, MetricVerifier

from hls.baseline_credit import (BaselineCredit, BaselineCreditError, BaselineDrive, BaselineRecord,
                                 CampaignIdentity, baseline_credit_bytes, credit_from_drives, credit_stats,
                                 credited, load_record, read_baseline_credit, spread_said)
from tests.support.credit import write_credit
from hls.core.verdict import passthrough_verdict

#: The module path a fixture campaign records itself under.
_MODULE = "tests.unit.test_baseline_credit"


# ---- task + trace construction -------------------------------------------

def _task(*, gpu_kind: str = "B200", rate: float = 2.0, ttft_ms: float = 100.0,
          scorer=None,
          baseline: Baseline | None = Baseline(image="hls-baseline",
                                               serve_command=["--flag", "1"]),
          ) -> TaskSpec:
    """A single-tenant task whose scorer needs a baseline credit. ``rate`` and
    ``ttft_ms`` are the two knobs the LoadRecord gate keys on but the workload
    fingerprint deliberately ignores."""
    dataset = HFDataset(path="cais/mmlu", subset="abstract_algebra", revision="main",
                        source_split="test",
                        prompt=JinjaPromptTemplate(template="{{ question }}"),
                        target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))
    return TaskSpec(
        id="T", model=ModelSpec(base_checkpoint="m", arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind=gpu_kind, count=1)),
        contract=RequestResponseContract(
            path="/v1/chat/completions", transport="http-sse",
            input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
            protocol=OpenAIChatProtocol()),
        workload=Workload(arrival=Poisson(rate_per_sec=rate), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(),
                          slo=StreamingLatencySLO(ttft_ms=ttft_ms, tpot_ms=50.0)),
        data_source=DataSource(dataset=dataset,
                               split=SplitSpec(dev=Count(value=5), test=Count(value=10))),
        verifiers=[MetricVerifier(metric=LetterMatchMetric(), min_value=0.5)],
        scorer=scorer if scorer is not None else SloLatencyScorer(),
        baseline=baseline,
    )


def _trace_at(ratios: list[float]) -> Trace:
    """A one-second drive whose requests land at exactly the given SLO-normalized
    latencies against a 100ms TTFT budget (single packet, so TPOT is vacuous)."""
    events = []
    for i, r in enumerate(ratios):
        t = 0.1 * r
        inp = Envelope(request_id=f"r{i}", contents=[TextPayload(text="q")],
                       metadata={"tenant_id": "t0"})
        out = Envelope(request_id=f"r{i}", contents=[TextPayload(text="a")])
        events.append(TraceEvent(
            request_id=f"r{i}", input=inp,
            response=ArtifactResponse(output=out, t_first_packet=t, packet_timestamps=[t]),
            t_send=0.0, t_recv=1.0))
    return Trace(events=events)


def _drives(credits: list[float]) -> list[BaselineDrive]:
    """One drive per credit, each starting a minute after the last so their order is
    unambiguous. The fixture task's SLO is request-grain, so its ten requests are ten
    units."""
    return [BaselineDrive(started_at=f"2026-09-05T00:{1 + i:02d}:00+00:00",
                          duration_sec=60.0, n_requests=10, n_units=10,
                          mean_log_credit=c)
            for i, c in enumerate(credits)]


def _campaign(task: TaskSpec, **overrides) -> CampaignIdentity:
    """The identity a campaign on ``task`` would record."""
    from hls.verify.calibration import workload_fingerprint

    fields = dict(
        task=_MODULE,
        workload_fingerprint=workload_fingerprint(task),
        load=load_record(task),
        gpu_kind=task.hardware.accelerator.kind,
        baseline=BaselineRecord(image="hls-baseline", serve_command=["--flag", "1"]))
    fields.update(overrides)
    return CampaignIdentity(**fields)


def _credit(task: TaskSpec, credits: list[float] | None = None, **overrides) -> BaselineCredit:
    """A schema-valid BaselineCredit for ``task``; ``overrides`` replaces any field
    afterwards. The three summaries are computed from the drives, so they are not
    settable here or anywhere else."""
    fields = dict(campaign=_campaign(task),
                  drives=_drives([0.9, 1.1] if credits is None else credits),
                  created_at="2026-09-05T00:10:00+00:00")
    fields.update(overrides)
    return BaselineCredit(**fields)




# ---- credit_stats ---------------------------------------------------------

def test_credit_stats_are_the_mean_sample_sd_and_standard_error():
    mean, sd, se = credit_stats([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert sd == pytest.approx(1.0)                 # sample sd, not population (which is 0.816)
    assert se == pytest.approx(1.0 / math.sqrt(3))


def test_credit_stats_of_a_single_drive_records_no_spread():
    # One drive has no spread between drives, so its sd and standard error are no number at all.
    assert credit_stats([0.5]) == (0.5, None, None)


def test_credit_stats_of_nothing_is_an_error():
    with pytest.raises(BaselineCreditError):
        credit_stats([])


# ---- model validators -----------------------------------------------------

def test_a_valid_credit_carries_the_mean_of_its_drives():
    bc = _credit(_task(), [0.4, 0.6, 0.8])
    assert bc.log_credit_mean == pytest.approx(0.6)
    assert bc.log_credit_se == pytest.approx(bc.log_credit_sd / math.sqrt(3))


@pytest.mark.parametrize("field", ["log_credit_mean", "log_credit_sd", "log_credit_se"])
def test_a_summary_written_into_the_file_is_refused(tmp_path, field):
    """B multiplies every contestant's score by exp(-B), so an edited mean would be a
    silent global rescale of the leaderboard, and the spread is what a reader judges the
    mean by. All three are computed from the drives on every read and never written, so
    a file that carries one was edited by hand, and it is refused."""
    task = _task()
    path = write_credit(tmp_path, _credit(task, [0.4, 0.6]))
    written = json.loads(path.read_text())
    assert field not in written
    assert read_baseline_credit(tmp_path / "baseline.json", task)[0].log_credit_mean == pytest.approx(0.5)
    path.write_text(json.dumps({**written, field: 0.0}))

    with pytest.raises(BaselineCreditError, match=field):
        read_baseline_credit(tmp_path / "baseline.json", task)[0]


def test_a_single_drive_campaign_is_a_credit_with_no_spread():
    """One drive is a credit: its B is that drive's, and it states that it has no spread between
    drives instead of a number."""
    bc = _credit(_task(), [0.5])
    assert bc.log_credit_mean == 0.5
    assert (bc.log_credit_sd, bc.log_credit_se) == (None, None)
    assert spread_said(bc) == "1 drive, so no spread between drives"
    assert spread_said(_credit(_task(), [0.4, 0.6])).endswith(", 2 drives")


def test_a_campaign_of_no_drives_is_refused():
    with pytest.raises(BaselineCreditError) as exc:
        _credit(_task(), [])
    assert "at least 1 drive" in str(exc.value)


# ---- load_record ----------------------------------------------------------

def test_load_record_captures_what_the_workload_fingerprint_leaves_out():
    """Arrival, think-time and SLO are all excluded from the calibration fingerprint
    and all move B, so the credit's own identity has to carry them."""
    rec = load_record(_task(rate=2.0, ttft_ms=100.0))
    assert rec.split == "test"
    assert [t.id for t in rec.tenants] == ["t0"]
    assert rec.tenants[0].arrival == {"kind": "Poisson", "rate_per_sec": 2.0}
    assert rec.tenants[0].inter_turn_delay == {"kind": "NoDelay"}
    assert rec.tenants[0].slo["ttft_ms"] == 100.0


def test_load_record_differs_where_the_workload_fingerprint_does_not():
    from hls.verify.calibration import workload_fingerprint

    slow, fast = _task(rate=2.0), _task(rate=8.0)
    assert workload_fingerprint(slow) == workload_fingerprint(fast)
    assert load_record(slow) != load_record(fast)


# ---- read_baseline_credit -------------------------------------------------

def test_load_refuses_a_missing_file(tmp_path):
    with pytest.raises(BaselineCreditError) as exc:
        read_baseline_credit(tmp_path / "baseline.json", _task())[0]
    assert "no baseline credit at" in str(exc.value)
    assert "hls baseline-credit" in str(exc.value)


def test_load_refuses_a_malformed_file(tmp_path):
    (tmp_path / "baseline.json").write_text("{not json")
    with pytest.raises(BaselineCreditError) as exc:
        read_baseline_credit(tmp_path / "baseline.json", _task())[0]
    assert "not a valid baseline credit file" in str(exc.value)


def test_load_accepts_a_matching_file(tmp_path):
    task = _task()
    bc = _credit(task, [0.4, 0.6])
    write_credit(tmp_path, bc)
    assert read_baseline_credit(tmp_path / "baseline.json", task)[0] == bc


def test_load_refuses_a_credit_measured_on_other_prompts(tmp_path):
    """A workload change (here the test split size) invalidates B: the baseline was
    driven with prompts the graded run does not serve."""
    task = _task()
    write_credit(tmp_path, _credit(task))
    other = task.model_copy(update={"data_source": task.data_source.model_copy(
        update={"split": SplitSpec(dev=Count(value=5), test=Count(value=99))})})
    with pytest.raises(BaselineCreditError) as exc:
        read_baseline_credit(tmp_path / "baseline.json", other)[0]
    assert "stale baseline credit" in str(exc.value)


def test_load_refuses_a_credit_measured_at_another_arrival_rate(tmp_path):
    """The rate is invisible to the workload fingerprint, so only the load record
    catches it -- and B moves with load, since load sets the queueing the reference
    meets."""
    write_credit(tmp_path, _credit(_task(rate=2.0)))
    with pytest.raises(BaselineCreditError) as exc:
        read_baseline_credit(tmp_path / "baseline.json", _task(rate=8.0))[0]
    assert "different load" in str(exc.value)


def test_load_refuses_a_credit_measured_against_another_slo(tmp_path):
    """The SLO is the yardstick each latency is divided by, so re-pinning a budget
    changes every credit including B's. The refusal names the tenant and the field
    that moved rather than printing both load records whole, which for a four-tenant
    task is a wall of JSON to diff by eye."""
    write_credit(tmp_path, _credit(_task(ttft_ms=100.0)))
    with pytest.raises(BaselineCreditError) as exc:
        read_baseline_credit(tmp_path / "baseline.json", _task(ttft_ms=250.0))[0]
    message = str(exc.value)
    assert "different load" in message
    assert "t0.slo" in message
    assert "100.0" in message and "250.0" in message
    assert "t0.arrival" not in message          # what did not change is not printed


def test_load_refuses_a_credit_measured_on_another_gpu(tmp_path):
    write_credit(tmp_path, _credit(_task(gpu_kind="B200")))
    with pytest.raises(BaselineCreditError) as exc:
        read_baseline_credit(tmp_path / "baseline.json", _task(gpu_kind="H200"))[0]
    assert "gpu_kind" in str(exc.value)


def test_load_refuses_a_credit_measured_on_a_retuned_reference_server(tmp_path):
    """B is a measurement of one server at one operating point, so a serve flag the
    task has since changed invalidates it. Without this refusal a retuned baseline
    keeps the old B and silently rescales every contestant's score."""
    task = _task()
    write_credit(tmp_path, _credit(task))
    retuned = task.model_copy(update={"baseline": task.baseline.model_copy(
        update={"serve_command": ["--flag", "2"]})})
    with pytest.raises(BaselineCreditError) as exc:
        read_baseline_credit(tmp_path / "baseline.json", retuned)[0]
    message = str(exc.value)
    assert "different reference server" in message
    assert "serve_command" in message and '"2"' in message
    assert "image" not in message               # what did not change is not printed


def test_load_refuses_a_credit_measured_against_another_reference_image(tmp_path):
    task = _task()
    write_credit(tmp_path, _credit(task))
    other = task.model_copy(update={"baseline": task.baseline.model_copy(
        update={"image": "hls-baseline-v2"})})
    with pytest.raises(BaselineCreditError) as exc:
        read_baseline_credit(tmp_path / "baseline.json", other)[0]
    assert "image:" in str(exc.value) and "hls-baseline-v2" in str(exc.value)


def test_load_refuses_a_credit_for_a_task_that_declares_no_reference_server(tmp_path):
    """A speedup-scored task with no Baseline has nothing the credit could be a
    measurement of, so the artifact cannot be held to it."""
    task = _task()
    write_credit(tmp_path, _credit(task))
    with pytest.raises(BaselineCreditError) as exc:
        read_baseline_credit(tmp_path / "baseline.json", _task(baseline=None))[0]
    assert "declares no baseline" in str(exc.value)


# ---- credited -------------------------------------------------------------

def test_inject_is_a_no_op_for_a_scorer_that_grades_against_no_baseline(tmp_path):
    """A goodput-scored task must not even look for baseline.json, so it runs fine in
    a folder that has none."""
    task = _task(scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))
    assert credited(task, tmp_path / "baseline.json")[0] is task
    assert credited(task, None)[0] is task


def test_inject_threads_the_measured_mean_into_the_scorer(tmp_path):
    task = _task()
    assert task.scorer.baseline_log_credit is None
    write_credit(tmp_path, _credit(task, [0.4, 0.6]))
    injected = credited(task, tmp_path / "baseline.json")[0]
    assert injected.scorer.baseline_log_credit == pytest.approx(0.5)
    assert task.scorer.baseline_log_credit is None      # the input tree is untouched


def test_inject_refuses_without_a_file():
    with pytest.raises(BaselineCreditError) as exc:
        credited(_task(), None)[0]
    assert "no baseline.json was given" in str(exc.value)


def test_an_injected_score_is_the_speedup_over_the_baseline(tmp_path):
    """The end of the chain: a run whose requests land exactly where the baseline's
    did scores 1.0, and one twice as fast scores 2.0."""
    task = _task()
    write_credit(tmp_path, _credit(task, [math.log(2.0), math.log(2.0)]))
    scorer = credited(task, tmp_path / "baseline.json")[0].scorer
    tenants = task.effective_tenants()
    for ratio, want in ((0.5, 1.0), (0.25, 2.0)):
        trace = _trace_at([ratio] * 4)
        score = asyncio.run(scorer.score(tenants, trace, passthrough_verdict(trace),
                                         task.hardware))
        assert score.value == pytest.approx(want)





































# ---- credit_from_drives ------------------------------------------------





def test_folding_ignores_how_many_requests_the_same_units_took(tmp_path):
    """At session grain a trajectory runs for as many turns as the server's own replies
    call for, so three drives of the same 100 sessions come to request counts 9% apart.
    They are three drives of one workload: the sessions are what the draw fixed."""
    task = _task()
    a, b, c = _drives([0.2, 0.4, 0.6])
    a = a.model_copy(update={"n_units": 100, "n_requests": 1362})
    b = b.model_copy(update={"n_units": 100, "n_requests": 1408})
    c = c.model_copy(update={"n_units": 100, "n_requests": 1486})

    bc = credit_from_drives(_campaign(task), [a, b, c])
    assert [d.n_units for d in bc.drives] == [100, 100, 100]
    assert [d.n_requests for d in bc.drives] == [1362, 1408, 1486]


def test_folding_refuses_a_drive_that_credited_fewer_units(tmp_path):
    """The set of units is fixed by the workload draw, so a drive short of its siblings
    lost one. Averaging it in would move B for a reason the file does not record."""
    task = _task()
    short, full = _drives([0.2, 0.4])
    short = short.model_copy(update={"n_units": 94, "n_requests": 1362})
    full = full.model_copy(update={"n_units": 100, "n_requests": 1408})

    with pytest.raises(BaselineCreditError) as exc:
        credit_from_drives(_campaign(task), [short, full])
    message = str(exc.value)
    assert "too different numbers of units" in message
    assert str(short.started_at) in message          # names which drives disagree
    assert "94 units over 1362 requests" in message  # and both counts of each


def test_folding_refuses_a_drive_that_records_no_unit_count(tmp_path):
    """Every drive the credit role measures carries its unit count, so a drive arriving
    without one was hand-built and there is nothing to check it against its siblings."""
    task = _task()
    a, b = _drives([0.2, 0.4])
    a = a.model_copy(update={"n_units": None})

    with pytest.raises(BaselineCreditError) as exc:
        credit_from_drives(_campaign(task), [a, b])
    assert "record no unit count" in str(exc.value)


































def test_pooling_no_drives_at_all_is_refused_on_the_documented_type():
    """The CLI catches only BaselineCreditError, so the empty campaign must land there
    rather than on the bare ValueError the request-count comparison would raise."""
    with pytest.raises(BaselineCreditError) as exc:
        credit_from_drives(_campaign(_task()), [])
    assert "at least 1" in str(exc.value)


def test_a_campaign_of_one_drive_pools_into_a_credit_with_no_spread():
    bc = credit_from_drives(_campaign(_task()), _drives([0.5]))
    assert (bc.log_credit_mean, bc.log_credit_sd, bc.log_credit_se) == (0.5, None, None)







# ---- serialization --------------------------------------------------------

def test_the_written_file_is_plain_json(tmp_path):
    """baseline.json is read by hand and diffed in review, so it must be plain JSON
    with no Python-only floats."""
    bc = _credit(_task(), [0.4, 0.6])
    path = write_credit(tmp_path, bc)
    assert path.name == "baseline.json"
    assert path.read_bytes() == baseline_credit_bytes(bc)
    on_disk = json.loads(path.read_text())
    # A credit that records no hardware is written without the key.
    assert on_disk == {key: value for key, value in dump_spec(bc).items() if key != "hardware"}
    assert on_disk["campaign"]["load"]["tenants"][0]["arrival"]["kind"] == "Poisson"


def test_the_shipped_credits_record_the_gpu_model_alone_and_write_back_byte_for_byte():
    """Each shipped credit records the GPU model its drives ran on, as the
    grant check reads it, every other field blank, and writes back to its own bytes."""
    paths = sorted((Path(__file__).resolve().parents[2] / "tasks").glob("*/defaults/baseline.json"))
    assert paths
    for path in paths:
        data = path.read_bytes()
        credit = BaselineCredit.model_validate_json(data)
        assert credit.hardware == [BaselineHardware(gpu_model="NVIDIA H200")], path
        assert baseline_credit_bytes(credit) == data, path
        assert b'"hardware": [\n    {\n      "gpu_model": "NVIDIA H200"\n    }\n  ]' in data, path
