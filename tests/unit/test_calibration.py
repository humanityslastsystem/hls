"""Tests for hls.verify.calibration.

The calibration-requiring verifier under test is the real
``TokenFaithfulnessVerifier`` (``needs_calibration = True``): it owns a ``reference``
field, carries an injectable ``calibration`` field, and has a verifier-owned
``rate_multiplier`` field that must NOT affect the workload fingerprint. Those are
exactly the surfaces this module's fingerprint/load/inject logic keys off.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hls.core.decode import Greedy, Sample
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.prompt import JinjaPromptTemplate
from hls.reference.references import HFReference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.workload import (AllAtOnce, Gamma, NoDelay, NoSLO, ShortOutputCap,
                               StreamingLatencySLO, Uncapped, Workload)
from hls.verify import LetterMatchMetric, MetricVerifier, TokenFaithfulnessVerifier, Verifier
from hls.verify.calibration import (Calibration, CalibrationError, PreflightResult, ReportCalibration,
                                    UPPER_Z, calibrated, read_calibration, workload_fingerprint,
                                    workload_payload)
from tests.unit.calibration_fixtures import make_calibration, make_report_calibration
import hls.verify.calibration as calibration_module


# ---- TaskSpec construction helpers (mirrors tests/unit/test_taskspec.py) ----

def _model(checkpoint: str = "m") -> ModelSpec:
    return ModelSpec(base_checkpoint=checkpoint, arch="TX")


def _hardware(kind: str = "B200") -> Hardware:
    return Hardware(accelerator=AcceleratorSpec(kind=kind, count=1))


def _contract(path: str = "/v1/chat/completions",
              transport: str = "http") -> RequestResponseContract:
    return RequestResponseContract(
        path=path,
        transport=transport,
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        protocol=OpenAIChatProtocol())


def _dataset(subset: str = "abstract_algebra") -> HFDataset:
    return HFDataset(path="cais/mmlu", subset=subset, revision="main", source_split="test",
                     prompt=JinjaPromptTemplate(template="{{ question }}"),
                     target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))


def _data_source(subset: str = "abstract_algebra", test: int = 10) -> DataSource:
    return DataSource(dataset=_dataset(subset),
                      split=SplitSpec(dev=Count(value=5), test=Count(value=test)))


def _workload(cap=None, arrival=None, slo=None) -> Workload:
    return Workload(arrival=arrival or AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=cap or Uncapped(), slo=slo or NoSLO())


def _oracle() -> HFReference:
    return HFReference(decode=Greedy())


def _base_kwargs() -> dict:
    """A needs_calibration TaskSpec (single verifier: ``TokenFaithfulnessVerifier``)."""
    return dict(
        id="T", model=_model(), hardware=_hardware(), contract=_contract(),
        workload=_workload(), data_source=_data_source(),
        verifiers=[TokenFaithfulnessVerifier(reference=_oracle())],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))


def _agent(*, max_turns: int = 3, max_concurrent_llm: int | None = None):
    """The agent seam a task declares: one framework, one tool, a turn budget."""
    from hls.spec.agent import AgentSpec
    from hls.spec.tools import LocalCallableToolRunner, ToolSpec
    from tests.unit.test_trajectory_driver import _TwoCallFramework

    runner = LocalCallableToolRunner(
        specs=[ToolSpec(name="lookup", description="d", json_schema={})],
        callables={"lookup": lambda q: {"text": q}})
    return AgentSpec(framework=_TwoCallFramework(), tool_runner=runner,
                     max_turns=max_turns, max_concurrent_llm=max_concurrent_llm)


# ---- workload_fingerprint ---------------------------------------------------

def test_fingerprint_stable_across_reconstruction():
    ts1 = TaskSpec(**_base_kwargs())
    ts2 = TaskSpec(**_base_kwargs())
    assert workload_fingerprint(ts1) == workload_fingerprint(ts2)


def test_fingerprint_changes_with_seed():
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["seed"] = 1
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) != workload_fingerprint(ts2)


def test_fingerprint_changes_with_dataset():
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["data_source"] = _data_source(subset="anatomy")
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) != workload_fingerprint(ts2)


def test_fingerprint_changes_with_split_sizes():
    """The split counts choose WHICH rows get provisioned, so they change the prompts
    the reference teacher-forces -- a calibration measured on other rows is stale."""
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["data_source"] = _data_source(test=11)
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) != workload_fingerprint(ts2)


def test_fingerprint_changes_with_output_cap():
    """The cap bounds every continuation the reference scores, so it is
    reference-visible -- the one workload field inside the fingerprint."""
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["workload"] = _workload(cap=ShortOutputCap(cap_tokens=16))
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) != workload_fingerprint(ts2)


def test_fingerprint_changes_with_reference_config():
    """The pinned reference IS the instrument the noise floor was measured with; any
    change to its config invalidates the measurement."""
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["verifiers"] = [TokenFaithfulnessVerifier(
        reference=HFReference(decode=Sample(temperature=0.7)))]
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) != workload_fingerprint(ts2)


def _chunked_task(query_chunk: int) -> TaskSpec:
    """The base task with its oracle's teacher-forcing chunk set, and nothing else
    changed."""
    kwargs = _base_kwargs()
    kwargs["verifiers"] = [TokenFaithfulnessVerifier(
        reference=HFReference(decode=Greedy(), query_chunk=query_chunk))]
    return TaskSpec(**kwargs)


def test_fingerprint_changes_with_the_teacher_forcing_chunk():
    """The chunk the reference tiles its forward at is part of the instrument: the math
    kernel tiles differently at each width, and a handful of gaps in a long reply move
    by about a nat between 4,096 and 512 -- enough to carry one across a threshold. Two
    tasks that differ in nothing else are two instruments."""
    assert (workload_fingerprint(_chunked_task(512))
            != workload_fingerprint(_chunked_task(4096)))


def test_fingerprint_unchanged_with_arrival_rate():
    """The arrival process stays out of the identity: re-pinning a task's load -- a
    baseline search rescaling TOTAL_RATE into the Gamma scale -- does not stale its
    calibration. Recalibrating after a load change is the task author's call."""
    arrival = Gamma(shape=0.5, scale=2.0)
    repinned = Gamma(shape=0.5, scale=0.5)          # the same shape at four times the rate
    assert repinned.mean_rate() == 4.0 * arrival.mean_rate()
    kwargs1 = _base_kwargs()
    kwargs1["workload"] = _workload(arrival=arrival)
    kwargs2 = _base_kwargs()
    kwargs2["workload"] = _workload(arrival=repinned)
    assert workload_fingerprint(TaskSpec(**kwargs1)) == workload_fingerprint(TaskSpec(**kwargs2))


def test_fingerprint_unchanged_with_slo():
    """The SLO is a scoring-time latency objective; it changes no teacher-forced
    token, so tightening it must not stale an existing calibration."""
    kwargs1 = _base_kwargs()
    kwargs1["contract"] = _contract(transport="http-sse")      # StreamingLatencySLO's wiring
    kwargs2 = dict(kwargs1)
    kwargs1["workload"] = _workload(slo=NoSLO())
    kwargs2["workload"] = _workload(slo=StreamingLatencySLO(ttft_ms=250.0, tpot_ms=30.0))
    assert workload_fingerprint(TaskSpec(**kwargs1)) == workload_fingerprint(TaskSpec(**kwargs2))


def test_fingerprint_covers_the_declared_context_window():
    """A framework finalizes trajectories against the server's context window, so
    the window decides which turns exist. The payload names it on every task, None for a
    task that declares none."""
    plain = TaskSpec(**_base_kwargs())
    assert workload_payload(plain)["max_context_len"] is None

    windowed = TaskSpec(**_base_kwargs(), max_context_len=131072)
    assert workload_payload(windowed)["max_context_len"] == 131072
    assert workload_fingerprint(windowed) != workload_fingerprint(plain)


def test_fingerprint_covers_the_agent_seam_but_not_its_timing():
    """An agentic task's prompts are whatever its framework builds, so the framework,
    its settings, the tools it offers and the turns it may take are all part of the
    workload identity. How many of those calls run at once is timing, and stays out
    like the arrival process does."""
    plain = TaskSpec(**_base_kwargs())
    assert workload_payload(plain)["agent"] is None

    agentic = TaskSpec(**_base_kwargs(), agent=_agent())
    payload = workload_payload(agentic)
    assert payload["agent"]["max_turns"] == 3
    assert payload["agent"]["tools"] == [{"name": "lookup", "description": "d",
                                          "json_schema": {}}]
    assert payload["agent"]["framework"].endswith("_TwoCallFramework")

    more_turns = TaskSpec(**_base_kwargs(), agent=_agent(max_turns=4))
    more_parallel = TaskSpec(**_base_kwargs(), agent=_agent(max_concurrent_llm=99))
    assert workload_fingerprint(more_turns) != workload_fingerprint(agentic)
    assert workload_fingerprint(more_parallel) == workload_fingerprint(agentic)


def test_fingerprint_changes_with_the_assets_identity():
    """A tool bundle decides every tool result the framework feeds back into a prompt,
    so a different corpus or index is a different population."""
    from dataclasses import replace

    from tests.unit.test_tool_assets import FakeAssets

    class _Other(FakeAssets):
        def identity(self):
            return "fake-v2"

    plain = TaskSpec(**_base_kwargs(), agent=_agent())
    v1 = TaskSpec(**_base_kwargs(), agent=replace(_agent(), assets=FakeAssets()))
    v2 = TaskSpec(**_base_kwargs(), agent=replace(_agent(), assets=_Other()))
    assert workload_fingerprint(plain) != workload_fingerprint(v1)
    assert workload_fingerprint(plain) != workload_fingerprint(v2)
    assert workload_fingerprint(v1) != workload_fingerprint(v2)


def test_default_traffic_config_is_the_frameworks_public_scalars():
    """A framework declares its knobs as fields, so the default needs no per-framework
    code: the public scalars are the settings, and a private or non-scalar attribute
    (a client, a cache) is not."""
    from dataclasses import dataclass, field

    from hls.spec.agent import AgentFramework

    @dataclass
    class _F(AgentFramework):
        turns_before_final: int = 8
        note: str = "x"
        _private: int = 1
        seen: list = field(default_factory=list)

        async def run_trajectory(self, *a, **k): ...

    assert _F().traffic_config() == {"note": "x", "turns_before_final": 8}


def test_deepresearch_framework_settings_reach_the_fingerprint():
    """The real agentic task's framework settings are in its identity, so retuning one
    of them stales the calibration measured under the old one."""
    from tasks.deepresearch.task import TASK

    cfg = workload_payload(TASK)["agent"]["framework_config"]
    assert cfg["finalize_after_search_turns"] == 80


def test_workload_payload_carries_exactly_the_reference_visible_fields():
    """The payload's field list is the fingerprint's contract, so the set is pinned: every
    task carries every key, a declared-nothing one as None."""
    payload = workload_payload(TaskSpec(**_base_kwargs()))
    assert set(payload) == {"model", "contract", "seed", "tenants", "agent", "max_context_len"}
    assert set(payload["tenants"][0]) == {"id", "data_source", "output_length_cap",
                                          "history_char_budget", "trim_rampdown",
                                          "references"}


def test_fingerprint_changes_with_the_history_window():
    """The client-side context window decides how much accumulated history each
    follow-up turn carries, so it changes WHICH prompt the reference is teacher-forced
    on -- a population change, not a timing one."""
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["workload"] = kwargs2["workload"].model_copy(
        update={"history_char_budget": 8000})
    assert workload_fingerprint(ts1) != workload_fingerprint(TaskSpec(**kwargs2))


def test_fingerprint_changes_with_rampdown_trimming():
    """Ramp-down trimming drops follow-up turns at DRAW time, so it changes which
    turns exist to be served at all."""
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["workload"] = kwargs2["workload"].model_copy(update={"trim_rampdown": True})
    assert workload_fingerprint(ts1) != workload_fingerprint(TaskSpec(**kwargs2))


def test_fingerprint_changes_with_model():
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["model"] = _model(checkpoint="other-checkpoint")
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) != workload_fingerprint(ts2)


def test_fingerprint_changes_with_contract():
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["contract"] = _contract(path="/v2/chat/completions")
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) != workload_fingerprint(ts2)


def test_fingerprint_unchanged_with_id():
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["id"] = "T-other"
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) == workload_fingerprint(ts2)


def test_fingerprint_unchanged_with_metadata():
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["metadata"] = {"theme": "other"}
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) == workload_fingerprint(ts2)


def test_fingerprint_unchanged_with_scorer():
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["scorer"] = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=99.0))
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) == workload_fingerprint(ts2)


def test_fingerprint_unchanged_with_min_tokens():
    """min_tokens lives on the verifier itself, not on `.reference` -- the
    fingerprint only ever dumps `v.reference`, so it must not move."""
    ts1 = TaskSpec(**_base_kwargs())
    kwargs2 = _base_kwargs()
    kwargs2["verifiers"] = [TokenFaithfulnessVerifier(reference=_oracle(), rate_multiplier=3.0)]
    ts2 = TaskSpec(**kwargs2)
    assert workload_fingerprint(ts1) == workload_fingerprint(ts2)


# ---- Calibration model ------------------------------------------------------

def _preflight(passed: bool = True) -> PreflightResult:
    return PreflightResult(passed=passed, flip_rate_vs_fp32=0.001, max_gap_vs_fp32=0.05,
                           n_tokens=1000)


def _calibration(task: TaskSpec, **overrides) -> Calibration:
    return make_calibration(task, **overrides)


def test_calibration_rejects_degenerate_q_max():
    """A zero ceiling would fail any trace carrying a single mismatch, which defeats
    the point of calibrating a noise floor."""
    ts = TaskSpec(**_base_kwargs())
    with pytest.raises(CalibrationError, match="degenerate"):
        _calibration(ts, q_max=0.0, rate_side={
            "hf_pinned": make_calibration(ts).rate_side["hf_pinned"].model_copy(
                update={"mean": 0.0, "sd": 0.0, "rates": [0.0, 0.0, 0.0]})})


def test_device_matches_kind_is_a_word_token_match():
    """Kind matching is by whole token: marketing variants of one kind match, other
    kinds and superstring names never do."""
    from hls.verify.calibration import device_matches_kind

    assert device_matches_kind("NVIDIA B200", "B200")
    assert device_matches_kind("B200", "B200")
    assert device_matches_kind("NVIDIA H200 NVL", "H200")
    assert not device_matches_kind("NVIDIA B200", "H200")
    assert not device_matches_kind("NVIDIA B2000", "B200")
    assert not device_matches_kind("", "B200")


def test_q_max_is_locked_to_the_worst_engine_mean():
    """q_max is the field the deployed rate rule runs on, so it is validator-locked to
    the pool's own draws: a hand-edit that widens the gate while the file's own rates
    say otherwise is refused."""
    ts = TaskSpec(**_base_kwargs())
    with pytest.raises(CalibrationError, match="contradicts its own servings"):
        _calibration(ts, q_max=0.5)


def test_engine_rate_is_locked_to_its_own_draws():
    """The per-engine mean is likewise recomputable from the stored per-draw rates."""
    ts = TaskSpec(**_base_kwargs())
    cal = make_calibration(ts)
    tampered = cal.rate_side["hf_pinned"].model_dump()
    tampered["mean"] = 0.5
    with pytest.raises(CalibrationError, match="contradicts its own servings"):
        _calibration(ts, rate_side={"hf_pinned": tampered})


def test_engine_rate_refuses_a_draw_count_that_disagrees_with_its_lists():
    ts = TaskSpec(**_base_kwargs())
    tampered = make_calibration(ts).rate_side["hf_pinned"].model_dump()
    tampered["draws"] = 9
    with pytest.raises(CalibrationError, match="servings but carries"):
        _calibration(ts, rate_side={"hf_pinned": tampered})


def test_curve_level_parameters_are_locked_to_their_event_counts():
    """lambda_hat is events over exposure and lambda_up is the exact Garwood limit for
    that count, so neither can be loosened without the count that justifies it."""
    ts = TaskSpec(**_base_kwargs())
    cal = make_calibration(ts)
    for field, match in (("lambda_hat", "contradicts events/exposure"),
                         ("lambda_up", "not the Garwood")):
        payload = cal.model_dump(mode="json")
        payload["curve_side"]["engines"]["hf_pinned"]["levels"][0][field] *= 3
        with pytest.raises(CalibrationError, match=match):
            Calibration.model_validate(payload)


def test_curve_refuses_event_counts_that_grow_with_severity():
    """gap >= c is nested, so a level cannot carry more events than a gentler one."""
    ts = TaskSpec(**_base_kwargs())
    with pytest.raises(CalibrationError, match="non-increasing"):
        _calibration(ts, engines={"hf_pinned": {"events": [1, 9, 0, 0, 0, 0, 0]}})


def test_the_rung_family_covers_every_rule_that_can_reject():
    """The Bonferroni split covers every rung that can reject a clean trace -- the gap
    levels plus the rate rule -- so it is read off the grid rather than stored beside
    it, where the two could disagree."""
    ts = TaskSpec(**_base_kwargs())
    cal = make_calibration(ts)
    assert cal.curve_side.n_rungs == len(cal.curve_side.levels_nats) + 1
    assert cal.delta_per_rung == cal.tolerances.delta / cal.curve_side.n_rungs


def test_both_rules_must_cover_the_same_pool():
    """An engine measured on one rule but not the other is half-measured, and the gate
    would still grade against it."""
    ts = TaskSpec(**_base_kwargs())
    cal = make_calibration(ts, engines={"hf_pinned": {}, "vllm": {}})
    payload = cal.model_dump(mode="json")
    del payload["curve_side"]["engines"]["vllm"]
    with pytest.raises(CalibrationError, match="same pool"):
        Calibration.model_validate(payload)


def test_a_calibration_file_carries_no_advisory_text(tmp_path):
    """What the censuses flag is reported when the calibration is measured, not stored in
    it: the model has no field for it, and a file that carries one is refused naming the
    key and the fix."""
    ts = TaskSpec(**_base_kwargs())
    for key in ("trust_warnings", "recurrence_warnings"):
        doc = _calibration(ts).model_dump(mode="json")
        doc[key] = {"hf_pinned": "outside the trusted band"}
        path = tmp_path / "calibration.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(CalibrationError) as exc:
            read_calibration(path, ts)
        assert key in str(exc.value) and "hls calibrate" in str(exc.value)


def test_calibration_rejects_tampered_workload_copy():
    """The persisted workload copy must hash to workload_fingerprint -- an audit
    copy edited independently of the gate is refused at construction."""
    ts = TaskSpec(**_base_kwargs())
    with pytest.raises(CalibrationError, match="workload copy"):
        _calibration(ts, workload={"model": "hand-edited"})


def test_curve_thresholds_grade_against_the_more_permissive_engine():
    """The ceiling is what SOME clean stack produces, not what all of them do, so a
    contestant is never failed for behaving like one of the pool's own members."""
    ts = TaskSpec(**_base_kwargs())
    cal = make_calibration(ts, engines={
        "quiet": {"events": [1, 0, 0, 0, 0, 0, 0]},
        "noisy": {"events": [40, 5, 0, 0, 0, 0, 0]}})
    rows = cal.thresholds(1_000_000)
    assert [r["engine"] for r in rows[:2]] == ["noisy", "noisy"]
    for row in rows:
        assert row["k"] == max(row["k_per_engine"].values())


# ---- read_calibration --------------------------------------------------------

#: What every loader refusal of a missing or unusable calibration tells the operator to do.
_FIX = "with `hls calibrate <module>`"


def _names_the_fix(error: CalibrationError) -> None:
    """*error* sends the operator to the stage-based command, which writes into the store; no
    refusal asks for a commit."""
    assert _FIX in str(error) and "uv run" not in str(error) and "commit" not in str(error), str(error)


def test_read_calibration_missing_file_names_the_fix(tmp_path):
    ts = TaskSpec(**_base_kwargs())
    with pytest.raises(CalibrationError) as exc:
        read_calibration(tmp_path / "calibration.json", ts)[0]
    _names_the_fix(exc.value)


def test_read_calibration_fingerprint_mismatch_names_the_fix(tmp_path):
    ts = TaskSpec(**_base_kwargs())
    cal = _calibration(ts)
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    kwargs2 = _base_kwargs()
    kwargs2["data_source"] = _data_source(subset="anatomy")  # changes the workload fingerprint
    ts2 = TaskSpec(**kwargs2)
    with pytest.raises(CalibrationError) as exc:
        read_calibration(tmp_path / "calibration.json", ts2)[0]
    _names_the_fix(exc.value)


def test_read_calibration_refuses_a_chunk_it_was_not_measured_at(tmp_path):
    """A calibration measured at one teacher-forcing chunk cannot grade a task
    declaring another: its ceilings describe the instrument it ran on. With the chunk
    outside the fingerprint the same file loads clean against either task, and the gate
    grades on ceilings taken on a different instrument."""
    measured = _chunked_task(4096)
    cal = _calibration(measured)
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    assert read_calibration(tmp_path / "calibration.json", measured)[0] == cal
    with pytest.raises(CalibrationError) as exc:
        read_calibration(tmp_path / "calibration.json", _chunked_task(512))[0]
    assert "stale" in str(exc.value)


def test_read_calibration_gpu_kind_mismatch_names_the_fix(tmp_path):
    ts = TaskSpec(**_base_kwargs())
    cal = _calibration(ts, gpu_kind="H100")
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    with pytest.raises(CalibrationError) as exc:
        read_calibration(tmp_path / "calibration.json", ts)[0]
    _names_the_fix(exc.value)
    assert "on the target GPU" in str(exc.value)


def test_read_calibration_preflight_failed_names_the_fix(tmp_path):
    ts = TaskSpec(**_base_kwargs())
    cal = _calibration(ts, preflight=_preflight(passed=False))
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    with pytest.raises(CalibrationError) as exc:
        read_calibration(tmp_path / "calibration.json", ts)[0]
    _names_the_fix(exc.value)


def test_read_calibration_valid_round_trip(tmp_path):
    ts = TaskSpec(**_base_kwargs())
    cal = _calibration(ts)
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    loaded = read_calibration(tmp_path / "calibration.json", ts)[0]
    assert loaded == cal


def test_read_calibration_malformed_json_raises_calibration_error(tmp_path):
    """A syntactically-broken calibration.json must not escape as a raw pydantic
    ValidationError -- every failure path is a CalibrationError naming the file
    and the `hls calibrate` fix."""
    ts = TaskSpec(**_base_kwargs())
    path = tmp_path / "calibration.json"
    path.write_text("{not valid json")
    with pytest.raises(CalibrationError) as exc:
        read_calibration(tmp_path / "calibration.json", ts)[0]
    _names_the_fix(exc.value)
    assert str(path) in str(exc.value)


def test_read_calibration_schema_invalid_json_raises_calibration_error(tmp_path):
    """Valid JSON that doesn't satisfy the Calibration schema (missing required
    fields) is also a CalibrationError, not a raw pydantic ValidationError."""
    ts = TaskSpec(**_base_kwargs())
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"not": "a calibration"}))
    with pytest.raises(CalibrationError) as exc:
        read_calibration(tmp_path / "calibration.json", ts)[0]
    _names_the_fix(exc.value)
    assert str(path) in str(exc.value)


def test_a_calibration_records_every_provenance_field(tmp_path):
    """The serving seeds and the spread confidence factor are not optional: a file that
    leaves one out does not fit the model and is refused naming the file and the fix,
    with no notion of an older or newer file format."""
    ts = TaskSpec(**_base_kwargs())
    path = tmp_path / "calibration.json"
    for section, field in (("draw", "serving_seeds"), ("sizing", "c_m")):
        doc = _calibration(ts).model_dump(mode="json")
        del doc[section][field]
        path.write_text(json.dumps(doc))
        with pytest.raises(CalibrationError) as exc:
            read_calibration(tmp_path / "calibration.json", ts)[0]
        assert field in str(exc.value) and "hls calibrate" in str(exc.value)
        assert "schema" not in str(exc.value) and "version" not in str(exc.value)


def test_serving_seeds_must_cover_every_serving():
    """One seed per serving per engine: a short record cannot re-drive what it
    claims to."""
    ts = TaskSpec(**_base_kwargs())
    base = _calibration(ts)
    with pytest.raises(CalibrationError, match="serving seeds"):
        _calibration(ts, draw=base.draw.model_copy(
            update={"serving_seeds": {"hf_pinned": (11,)}}))


# ---- calibrated ---------------------------------------------------------------

def test_calibrated_untouched_for_metric_only_task():
    kwargs = _base_kwargs()
    kwargs["verifiers"] = [MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)]
    ts = TaskSpec(**kwargs)
    result = calibrated(ts, None)[0]
    assert result is ts


def test_calibrated_raises_when_no_file_is_given():
    ts = TaskSpec(**_base_kwargs())
    with pytest.raises(CalibrationError, match="no calibration.json was given"):
        calibrated(ts, None)[0]


def test_calibrated_injects_object_returned_by_load(monkeypatch):
    """Identity check: the exact Calibration object read_calibration returns must
    flow, unmutated, into the rebuilt verifier (the model_copy chain never copies
    the Calibration itself, just threads the reference through)."""
    ts = TaskSpec(**_base_kwargs())
    cal = _calibration(ts)
    monkeypatch.setattr(calibration_module, "read_calibration",
                        lambda path, task, kind: (cal, b"{}"))
    ts2 = calibrated(ts, object())[0]
    injected = ts2.effective_tenants()[0].verifiers[0].calibration
    assert injected is cal
    # never mutates the original
    assert ts.effective_tenants()[0].verifiers[0].calibration is None


def test_calibrated_injects_for_needs_calibration_task_round_trip(tmp_path):
    ts = TaskSpec(**_base_kwargs())
    cal = _calibration(ts)
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    ts2 = calibrated(ts, tmp_path / "calibration.json")[0]
    injected = ts2.effective_tenants()[0].verifiers[0].calibration
    assert injected == cal


def test_fingerprint_unchanged_by_verifier_tolerances():
    """The tolerance multipliers live on the verifier and describe how the measured
    ceilings are applied, not what traffic the reference sees -- so they must not move
    the workload fingerprint."""
    ts1 = TaskSpec(**_base_kwargs())
    kwargs = _base_kwargs()
    kwargs["verifiers"] = [TokenFaithfulnessVerifier(
        reference=_oracle(), rate_multiplier=3.0, curve_multiplier=2.0)]
    assert workload_fingerprint(ts1) == workload_fingerprint(TaskSpec(**kwargs))


# ---- Verifier.needs_calibration base default ---------------------------------

def test_verifier_needs_calibration_defaults_false():
    assert Verifier.needs_calibration is False
    assert MetricVerifier.needs_calibration is False


def test_token_faithfulness_needs_calibration_true():
    assert TokenFaithfulnessVerifier.needs_calibration is True


# ---- the two calibration artifacts -------------------------------------------
#
# A task has one calibration.json, and the two gates measure different
# things, so the file carries a ``kind`` tag and the loader validates it into the
# artifact the task's verifiers can actually read.


def _report_task_kwargs() -> dict:
    """A task whose single verifier is the sampling gate, so it needs the report
    artifact."""
    from hls.verify import ReportFaithfulnessVerifier
    kwargs = _base_kwargs()
    kwargs["contract"] = _contract().model_copy(
        update={"protocol": OpenAIChatProtocol(return_logprobs=True, top_logprobs=21)})
    kwargs["verifiers"] = [ReportFaithfulnessVerifier(
        reference=HFReference(decode=Sample(temperature=1.0)))]
    return kwargs


def test_the_flips_artifact_is_the_default_kind():
    """A file written before the tag existed carries none, and it is the token-level
    gate's artifact -- the only one there was."""
    ts = TaskSpec(**_base_kwargs())
    assert _calibration(ts).kind == "flips"
    assert calibration_module.calibration_kind(ts) == "flips"


def test_a_sampling_task_needs_the_report_artifact():
    ts = TaskSpec(**_report_task_kwargs())
    assert calibration_module.calibration_kind(ts) == "report"


def test_a_task_with_no_calibrated_verifier_needs_neither():
    kwargs = _base_kwargs()
    kwargs["verifiers"] = [MetricVerifier(metric=LetterMatchMetric())]
    assert calibration_module.calibration_kind(TaskSpec(**kwargs)) is None


def test_verifiers_needing_different_artifacts_are_refused():
    """A task has one calibration file, so there is no resolution: whichever were written,
    one verifier would grade on numbers that are not about its gate."""
    from hls.verify import ReportFaithfulnessVerifier
    kwargs = _report_task_kwargs()
    kwargs["verifiers"] = [
        ReportFaithfulnessVerifier(reference=HFReference(decode=Sample())),
        TokenFaithfulnessVerifier(reference=_oracle())]
    with pytest.raises(CalibrationError, match="different"):
        calibration_module.calibration_kind(TaskSpec(**kwargs))


def test_read_calibration_round_trips_the_report_artifact(tmp_path):
    ts = TaskSpec(**_report_task_kwargs())
    cal = make_report_calibration(ts)
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    assert read_calibration(tmp_path / "calibration.json", ts, "report")[0] == cal


def test_read_calibration_refuses_the_other_kind(tmp_path):
    """Naming which gate the file calibrates is what keeps the refusal readable: the
    wrong artifact would otherwise fail as a heap of missing fields."""
    ts = TaskSpec(**_report_task_kwargs())
    cal = make_calibration(TaskSpec(**_base_kwargs()))
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    with pytest.raises(CalibrationError, match="calibrates the 'flips' gate"):
        read_calibration(tmp_path / "calibration.json", ts, "report")[0]


def test_the_other_kind_refusal_points_at_the_task_the_file_belongs_to(tmp_path):
    """The refusal names the module the file was measured for, and the command that measures
    this task's own, which writes it into the store."""
    ts = TaskSpec(**_report_task_kwargs())
    cal = make_calibration(TaskSpec(**_base_kwargs()))
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    with pytest.raises(CalibrationError) as exc:
        read_calibration(tmp_path / "calibration.json", ts, "report")[0]
    assert cal.task in str(exc.value)
    assert "which writes it into the store" in str(exc.value) and "refuses" not in str(exc.value)


def test_a_report_calibration_short_of_its_precision_names_the_servings_to_raise(tmp_path):
    ts = TaskSpec(**_report_task_kwargs())
    cal = make_report_calibration(ts, engines={"vllm": {"mean_kl": 0.02, "se_kl": 0.01}})
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    with pytest.raises(CalibrationError, match="failed its precision check on vllm") as exc:
        read_calibration(tmp_path / "calibration.json", ts, "report")[0]
    assert "a larger `--servings-per-engine`" in str(exc.value) and "uv run" not in str(exc.value)
    assert "commit" not in str(exc.value)


def test_read_calibration_refuses_a_report_file_for_the_token_gate(tmp_path):
    ts = TaskSpec(**_base_kwargs())
    cal = make_report_calibration(ts)
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    with pytest.raises(CalibrationError, match="calibrates the 'report' gate"):
        read_calibration(tmp_path / "calibration.json", ts)[0]


def test_read_calibration_checks_the_report_artifacts_fingerprint(tmp_path):
    ts = TaskSpec(**_report_task_kwargs())
    cal = make_report_calibration(ts, workload={"probe": "other"},
                                  workload_fingerprint=hashlib.sha256(
                                      json.dumps({"probe": "other"},
                                                 sort_keys=True).encode()).hexdigest())
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    with pytest.raises(CalibrationError, match="stale calibration"):
        read_calibration(tmp_path / "calibration.json", ts, "report")[0]


def test_read_calibration_checks_the_report_artifacts_gpu_kind(tmp_path):
    ts = TaskSpec(**_report_task_kwargs())
    cal = make_report_calibration(ts, gpu_kind="H200")
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    with pytest.raises(CalibrationError, match="gpu_kind"):
        read_calibration(tmp_path / "calibration.json", ts, "report")[0]


def test_calibrated_threads_the_report_artifact(tmp_path):
    ts = TaskSpec(**_report_task_kwargs())
    cal = make_report_calibration(ts)
    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    injected = calibrated(ts, tmp_path / "calibration.json")[0]
    assert injected.effective_tenants()[0].verifiers[0].calibration == cal
    assert ts.effective_tenants()[0].verifiers[0].calibration is None


def test_report_kappa_is_locked_to_the_worst_engine_upper_limit():
    """The ceiling covers every trusted stack rather than the average of them, and a
    hand-edit that loosens it is unrepresentable."""
    with pytest.raises(CalibrationError, match="kappa"):
        make_report_calibration(kappa=0.5)
    cal = make_report_calibration(engines={"vllm": {"mean_kl": 0.02, "se_kl": 0.001},
                                           "sglang": {"mean_kl": 0.05, "se_kl": 0.002}})
    assert cal.kappa == pytest.approx(0.05 + UPPER_Z * 0.002)


def test_report_engine_upper_limit_is_locked_to_its_own_mean_and_error():
    """An upper limit that is not the engine's own mean plus its own error is a claim
    the servings do not support, whichever direction it was edited in."""
    doc = make_report_calibration().model_dump()
    doc["engines"]["vllm"]["upper"] = doc["engines"]["vllm"]["mean_kl"]
    doc["kappa"] = doc["engines"]["vllm"]["upper"]
    with pytest.raises(CalibrationError, match="records upper"):
        ReportCalibration(**doc)


def test_report_calibration_rejects_a_degenerate_ceiling():
    with pytest.raises(CalibrationError, match="degenerate"):
        make_report_calibration(engines={"vllm": {"mean_kl": 0.0, "se_kl": 0.0}})


def test_report_outside_ceiling_is_locked_to_the_worst_engine_upper_limit():
    """The served-outside ceiling is built the way kappa is: each engine's rate plus
    its one-sided margin, the pool's highest, and a hand-edit is unrepresentable."""
    with pytest.raises(CalibrationError, match="outside_ceiling"):
        make_report_calibration(outside_ceiling=0.5)
    cal = make_report_calibration(
        engines={"vllm": {"outside_rate": 0.001, "se_outside": 0.0001},
                 "sglang": {"outside_rate": 0.003, "se_outside": 0.0002}})
    assert cal.outside_ceiling == pytest.approx(0.003 + UPPER_Z * 0.0002)


def test_report_engine_outside_upper_is_locked_to_its_own_rate_and_error():
    doc = make_report_calibration().model_dump()
    doc["engines"]["vllm"]["outside_upper"] = doc["engines"]["vllm"]["outside_rate"]
    doc["outside_ceiling"] = doc["engines"]["vllm"]["outside_upper"]
    with pytest.raises(CalibrationError, match="records outside_upper"):
        ReportCalibration(**doc)


def test_report_calibration_rejects_a_degenerate_outside_ceiling():
    """A pool that never served outside its own atoms did not measure a sampler, and a
    0 ceiling would fail a faithful trace at its first tail token."""
    with pytest.raises(CalibrationError, match="served-outside ceiling"):
        make_report_calibration(engines={"vllm": {"outside_rate": 0.0,
                                                  "se_outside": 0.0}})


def test_a_report_file_without_the_outside_ceiling_is_refused(tmp_path):
    """A file written before the served-outside check existed carries no ceiling for
    it, and there is no reading of the file that supplies one."""
    ts = TaskSpec(**_report_task_kwargs())
    doc = make_report_calibration(ts).model_dump(mode="json")
    doc.pop("outside_ceiling")
    (tmp_path / "calibration.json").write_text(json.dumps(doc))
    with pytest.raises(CalibrationError, match="not a valid calibration file"):
        read_calibration(tmp_path / "calibration.json", ts, "report")[0]


def test_report_calibration_rejects_a_tampered_workload_copy():
    with pytest.raises(CalibrationError, match="workload copy"):
        make_report_calibration(workload={"probe": "edited"})


def test_a_report_file_without_the_end_of_turn_section_is_refused(tmp_path):
    """The section carries the threshold the end-of-turn count reads, so a file without
    it cannot grade one. There is no reading of an older file that recovers it, and the
    refusal says so by naming the command that writes a new one."""
    ts = TaskSpec(**_report_task_kwargs())
    doc = make_report_calibration(ts).model_dump(mode="json")
    doc.pop("end_of_turn")
    (tmp_path / "calibration.json").write_text(json.dumps(doc))
    with pytest.raises(CalibrationError, match="not a valid calibration file"):
        read_calibration(tmp_path / "calibration.json", ts, "report")[0]


def test_the_end_of_turn_threshold_must_be_one_of_the_published_candidates():
    """The value is stated in the brief a contestant builds against, so it is one of a
    handful of round numbers rather than whatever a search happened to land on."""
    from tests.unit.calibration_fixtures import end_of_turn_check
    with pytest.raises(CalibrationError, match="published thresholds"):
        end_of_turn_check(["vllm"], eps=0.07)
