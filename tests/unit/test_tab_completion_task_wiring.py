"""Wiring test for tasks.tab_completion.task: every arithmetic constant is
re-derived, the workload fingerprint is pinned, the serving envelope and the
terminator set are checked against the real checkpoint where it is cached."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from hls.core.capabilities import StackReport, TeacherForce
from hls.reference.references import HFReference
from hls.spec.datasets import FimSessionDataset, SafimDataset
from hls.spec.datasource import Count
from hls.spec.wire import OpenAICompletionProtocol
from hls.spec.workload import GammaInterTurn, NoDelay, Poisson, StreamingLatencySLO
from hls.verify.verifiers import TokenFaithfulnessVerifier
from tests.unit.brief_task import render_brief

MODEL_ID = "JetBrains/Mellum2-12B-A2.5B-Base"
#: The checkpoint's own sentinel ids, which the served prompts must tokenize to.
_FIM_IDS = {"<fim_prefix>": 8, "<fim_suffix>": 10, "<fim_middle>": 9}
_PORTIONS = {"commit_add": 0.25, "commit_edit": 0.25, "algo_block": 0.25, "algo_control": 0.25}
_STEPS = {"commit_add": 5.4, "commit_edit": 5.3, "algo_block": 1.0, "algo_control": 1.0}
_TEST_COUNTS = {"commit_add": 1241, "commit_edit": 1264, "algo_block": 6700, "algo_control": 6700}
_DEV = 1280
_TENANTS = {  # cap, (ttft_ms, tpot_ms)
    "commit_add": (256, (600.0, 10.0)), "commit_edit": (256, (600.0, 10.0)),
    "algo_block": (256, (250.0, 10.0)), "algo_control": (256, (200.0, 10.0))}
_DATASETS = {"commit_add": (FimSessionDataset, "add"), "commit_edit": (FimSessionDataset, "edit"),
             "algo_block": (SafimDataset, "block"), "algo_control": (SafimDataset, "control")}


def _tenants():
    from tasks.tab_completion.task import TASK
    return {t.id: t for t in TASK.effective_tenants()}


def _total_rate():
    from tasks.tab_completion.task import TOTAL_RATE
    return TOTAL_RATE


@pytest.fixture(scope="module")
def tokenizer():
    """The real checkpoint's tokenizer, from the local HF cache only; skipped when its
    ``tokenizer.json`` is not cached (the CI runner has no cache). The cache is asked
    first: with the file absent, the load falls back to converting a slow tokenizer,
    whose error depends on the host's packages (protobuf's ImportError on the runner).

    Loaded through the class ``tokenizer_config.json`` names rather than
    ``AutoTokenizer``, which resolves the model config first and so needs a
    transformers new enough to register this architecture."""
    transformers = pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    if not isinstance(hub.try_to_load_from_cache(MODEL_ID, "tokenizer.json"), str):
        pytest.skip(f"{MODEL_ID}'s tokenizer.json is not in the local HF cache")
    return transformers.PreTrainedTokenizerFast.from_pretrained(MODEL_ID, local_files_only=True)


@pytest.fixture(scope="module")
def checkpoint_config():
    """The checkpoint's ``config.json`` as plain JSON, from the local HF cache only.
    Read as a file for the same reason the tokenizer is loaded directly."""
    hub = pytest.importorskip("huggingface_hub")
    try:
        path = hub.hf_hub_download(MODEL_ID, "config.json", local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL_ID} is not in the local HF cache")
    return json.loads(Path(path).read_text())


def test_four_tenants_in_declared_order():
    assert list(_tenants()) == ["commit_add", "commit_edit", "algo_block", "algo_control"]


def test_task_validates_and_serializes():
    from hls.core.poly import dump_spec
    from tasks.tab_completion.task import TASK
    rec = dump_spec(TASK)
    assert rec["id"] == "tab_completion" and len(rec["tenants"]) == 4
    assert dump_spec(TASK) == rec


@pytest.mark.parametrize("name", list(_TENANTS))
def test_datasets_are_the_pinned_fim_corpora(name):
    cls, subset = _DATASETS[name]
    ds = _tenants()[name].data_source
    assert isinstance(ds.dataset, cls) and ds.dataset.subset == subset
    assert ds.dataset.revision != "main"
    assert ds.split == type(ds.split)(dev=Count(value=_DEV), test=Count(value=_TEST_COUNTS[name]))
    assert ds.dev_dataset is None and ds.removed_rows == ()
    assert "fim_prompt.txt" in ds.dev_reference


def test_commit_tenants_are_eight_step_sessions_and_safim_is_single_shot():
    from tasks.tab_completion.task import SESSION_STEPS
    t = _tenants()
    assert SESSION_STEPS == 8
    for name in ("commit_add", "commit_edit"):
        assert t[name].data_source.dataset.max_steps == SESSION_STEPS
        assert isinstance(t[name].workload.inter_turn_delay, GammaInterTurn)
        assert t[name].workload.trim_rampdown is True
    for name in ("algo_block", "algo_control"):
        assert isinstance(t[name].workload.inter_turn_delay, NoDelay)


def test_the_dev_reference_states_the_same_step_count_the_workload_draws():
    """The cut count is inside the workload fingerprint, so a drift between what the
    contestant is told and what is served would be silent."""
    from tasks.tab_completion.task import SESSION_STEPS
    text = _tenants()["commit_add"].data_source.dev_reference["fim_prompt.txt"]
    assert f"up to {SESSION_STEPS} requests" in text


def test_typing_pace_median_is_ten_seconds():
    d = _tenants()["commit_add"].workload.inter_turn_delay
    xs = [d.delay(1, seed=(s, 0)) for s in range(20_000)]
    assert statistics.median(xs) == pytest.approx(10.0, rel=0.05)
    assert max(xs) <= d.max_gap


@pytest.mark.parametrize("name", list(_TENANTS))
def test_session_rate_is_the_portion_over_steps(name):
    a = _tenants()[name].workload.arrival
    assert isinstance(a, Poisson)
    assert a.rate_per_sec == pytest.approx(_PORTIONS[name] * _total_rate() / _STEPS[name], rel=1e-6)


def test_request_rates_partition_the_total():
    total = sum(t.workload.arrival.rate_per_sec * _STEPS[n] for n, t in _tenants().items())
    assert total == pytest.approx(_total_rate(), rel=1e-6)


@pytest.mark.parametrize("name", list(_TENANTS))
def test_declared_turns_are_the_ones_the_session_rate_was_derived_from(name):
    """The brief multiplies each tenant's session rate by ``mean_turns_per_session``
    to state its request rate, so the declared factor must be the one ``_arrival``
    divided by; otherwise the brief states a load the driver does not offer."""
    w = _tenants()[name].workload
    assert w.mean_turns_per_session == _STEPS[name]
    assert w.prompt_fields()["request_rate"] == pytest.approx(
        _PORTIONS[name] * _total_rate(), rel=1e-6)


def test_brief_states_the_workload_in_sessions_and_requests():
    """A commit tenant's session rate is 5.4x below its request rate; the brief must
    print both, and never call a session a request."""
    from tasks.tab_completion.task import ARTIFACT, TASK, TOTAL_RATE
    text, _ = render_brief(TASK, ARTIFACT.brief)
    workload = text.split("\n## Workload\n", 1)[1].split("\n## Network Contract", 1)[0]
    assert f"Offered load over all\ntenants: {TOTAL_RATE:.4g} requests/s" in workload
    commit = workload.split("### Tenant commit_add", 1)[1].split("### Tenant", 1)[0]
    assert "- Sessions: poisson arrivals, 1.139 sessions/s" in commit
    assert "- Requests: 5.4 turns/session, 6.153 requests/s" in commit
    assert "- Between turns: gamma think-time" in commit
    algo = workload.split("### Tenant algo_block", 1)[1].split("### Tenant", 1)[0]
    assert "- Requests: 1 turn/session, 6.153 requests/s" in algo
    assert "Between turns" not in algo
    assert "Requests arrive" not in text
    assert "Sessions arrive open-loop" in text


@pytest.mark.parametrize("name", list(_TENANTS))
def test_counts_realize_equal_request_shares(name):
    requests = {n: _TEST_COUNTS[n] * _STEPS[n] for n in _TEST_COUNTS}
    assert requests[name] / sum(requests.values()) == pytest.approx(0.25, abs=0.005)
    assert _tenants()[name].data_source.split.test == Count(value=_TEST_COUNTS[name])


@pytest.mark.parametrize("name", list(_TENANTS))
def test_cap_and_slo_match_the_task(name):
    cap, (ttft, tpot) = _TENANTS[name]
    w = _tenants()[name].workload
    assert w.output_length_cap.cap_tokens == cap
    assert isinstance(w.slo, StreamingLatencySLO) and (w.slo.ttft_ms, w.slo.tpot_ms) == (ttft, tpot)


def test_contract_is_streaming_completions_with_the_pinned_terminators():
    from tasks.tab_completion.task import CONTRACT, STOP_TOKEN_IDS
    assert CONTRACT.path == "/v1/completions" and CONTRACT.transport == "http-sse"
    assert isinstance(CONTRACT.protocol, OpenAICompletionProtocol)
    assert CONTRACT.protocol.stop_token_ids == STOP_TOKEN_IDS == (0,)


def test_the_model_is_the_pinned_mellum2_revision():
    from tasks.tab_completion.task import MODEL
    assert MODEL.base_checkpoint == MODEL_ID
    assert MODEL.revision == "271755e48ab6b2ed0ef224eaaabe2d25275fb8ee"


def test_the_terminators_are_the_checkpoints_own_declared_set(tokenizer):
    """The pinned set is read the same way the reference's load guard reads it, so a
    checkpoint that declares an id this contract omits fails here, not mid-run."""
    from hls.reference.engines import _declared_eos_ids
    from tasks.tab_completion.task import STOP_TOKEN_IDS
    assert _declared_eos_ids(tokenizer, MODEL_ID) == set(STOP_TOKEN_IDS)
    assert tokenizer.convert_tokens_to_ids("<|endoftext|>") == 0


def test_an_engine_built_from_the_task_uses_exactly_the_pinned_terminators():
    from tasks.tab_completion.task import STOP_TOKEN_IDS, TASK
    eng = TASK.effective_tenants()[0].verifiers[0].reference.build_engine(TASK)
    assert eng.conditioning == "raw"
    eng._tokenizer = type("T", (), {"eos_token_id": 0})()
    eng._model = type("M", (), {"generation_config": type("G", (), {"eos_token_id": [0, 11]})()})()
    assert eng._eos_ids() == set(STOP_TOKEN_IDS)


def test_each_sentinel_is_one_token_of_the_checkpoints_vocabulary(tokenizer):
    for marker, want in _FIM_IDS.items():
        assert tokenizer(marker, add_special_tokens=False)["input_ids"] == [want]


def test_raw_prompt_tokenization_parity(tokenizer):
    """The FIM prompt tokenizes to exactly [fim_prefix, *prefix, fim_suffix, *suffix,
    fim_middle] with no added specials; the sentinels never split."""
    from hls.prompt import JinjaPromptTemplate
    from tasks.tab_completion.task import FIM_PROMPT
    prefix, suffix = "def add(a, b):\n    ", "\n\nprint(add(1, 2))\n"
    text = JinjaPromptTemplate(template=FIM_PROMPT).render(prefix=prefix, suffix=suffix)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    expect = ([_FIM_IDS["<fim_prefix>"]] + tokenizer(prefix, add_special_tokens=False)["input_ids"]
              + [_FIM_IDS["<fim_suffix>"]] + tokenizer(suffix, add_special_tokens=False)["input_ids"]
              + [_FIM_IDS["<fim_middle>"]])
    assert ids == expect
    # The tokenizer adds no BOS, so the harness's add_special_tokens=False drops nothing.
    assert tokenizer(text, add_special_tokens=True)["input_ids"] == ids


def test_the_baseline_serves_the_declared_envelope():
    from tasks.tab_completion.task import SERVING_CONTEXT, TASK
    cmd = TASK.baseline.serve_command
    assert cmd[cmd.index("--max-model-len") + 1] == str(SERVING_CONTEXT)


def test_every_tenant_carries_the_one_output_cap():
    from tasks.tab_completion.task import OUTPUT_CAP
    for t in _tenants().values():
        assert t.workload.output_length_cap.cap_tokens == OUTPUT_CAP


def test_the_envelope_holds_the_longest_request_and_fits_the_checkpoint(checkpoint_config):
    """Three sentinel tokens ride every prompt, and the largest request any tenant can
    send is the longest full row plus the cap. That has to fit the declared envelope,
    which in turn has to fit the checkpoint's own window."""
    from tasks.tab_completion.task import LONGEST_FULL_ROW_TOKENS, OUTPUT_CAP, SERVING_CONTEXT
    longest_request = LONGEST_FULL_ROW_TOKENS + 3 + OUTPUT_CAP
    assert checkpoint_config["max_position_embeddings"] >= SERVING_CONTEXT >= longest_request


def test_the_sizing_sentence_is_interpolated_from_the_constants():
    """The brief's sizing sentence states SERVING_CONTEXT and the cap themselves, not
    copied literals, so re-pinning either can never leave a stale number in the text the
    agent reads. It must also say why 32,768 and not the checkpoint's larger window."""
    from tasks.tab_completion.task import ARTIFACT, OUTPUT_CAP, SERVING_CONTEXT, TASK
    text, _ = render_brief(TASK, ARTIFACT.brief)
    prose = " ".join(text.split())
    assert f"Size the server for a context window of {SERVING_CONTEXT:,} tokens" in prose
    assert f"longest prompt plus the {OUTPUT_CAP}-token output cap" in prose
    assert "checkpoint's own window is larger still" in prose


def test_the_brief_states_the_cap_for_every_tenant():
    from tasks.tab_completion.task import ARTIFACT, OUTPUT_CAP, TASK
    text, _ = render_brief(TASK, ARTIFACT.brief)
    workload = text.split("\n## Workload\n", 1)[1].split("\n## Network Contract", 1)[0]
    assert workload.count(f"- Per-request output cap: {OUTPUT_CAP} tokens") == len(_TENANTS)


def test_the_brief_shows_the_template_the_datasets_render():
    """The dev reference hands the contestant the served template verbatim; a drift
    between it and the one the tenants render would be silent."""
    from tasks.tab_completion.task import FIM_PROMPT
    for name in _TENANTS:
        ds = _tenants()[name].data_source
        assert ds.dataset.prompt.template == FIM_PROMPT
        assert FIM_PROMPT in ds.dev_reference["fim_prompt.txt"]


def test_baseline_decodes_greedily():
    from hls.reference.engines import _GREEDY_SAMPLING
    from tasks.tab_completion.task import TASK
    cmd = TASK.baseline.serve_command
    assert "--override-generation-config" in cmd
    assert json.loads(cmd[cmd.index("--override-generation-config") + 1]) == dict(_GREEDY_SAMPLING)


def test_every_tenant_shares_one_token_gate_and_one_hf_oracle():
    tenants = _tenants()
    gates = {id(t.verifiers[0]) for t in tenants.values()}
    assert len(gates) == 1
    v = next(iter(tenants.values())).verifiers[0]
    assert isinstance(v, TokenFaithfulnessVerifier) and v.rate_multiplier == 2.0
    assert v.calibration is None and v.needs_calibration is True and v.check_emit_consistency is False
    assert isinstance(v.reference, HFReference)
    assert {TeacherForce, StackReport} <= set(TokenFaithfulnessVerifier.requires)


def test_hardware_is_one_h200():
    from tasks.tab_completion.task import TASK
    assert (TASK.hardware.accelerator.kind, TASK.hardware.accelerator.count) == ("H200", 1)


def test_run_refuses_without_a_calibration_json(tmp_path):
    from hls.verify.calibration import CalibrationError, calibrated
    from tasks.tab_completion.task import TASK
    with pytest.raises(CalibrationError) as exc:
        calibrated(TASK, tmp_path / "calibration.json")[0]
    assert "hls calibrate" in str(exc.value)


def test_the_workload_fingerprint_is_pinned():
    from hls.verify.calibration import workload_fingerprint
    from tasks.tab_completion.task import TASK
    assert workload_fingerprint(TASK) == (
        "06cd15566f867d621a14d704f10092431659e08189dadf1be75d0aaa17d1a68b")


def test_brief_states_raw_prompts_the_terminator_rule_and_the_envelope():
    from tasks.tab_completion.task import ARTIFACT, TASK
    text, prov = render_brief(TASK, ARTIFACT.brief)
    sections = {s.block: s for s in prov.sections}
    prose = " ".join(text.split())
    assert sections["goal"].source == "task" and sections["contract"].source == "task"
    assert sections["build_instructions"].source == "task"
    assert "SLO-normalized latency" in prose
    assert "must not fail any request" in prose
    assert "exceeds the thresholds" in prose               # uncalibrated: qualitative gate
    assert "`stop_token_ids` is `<|endoftext|>`, id 0" in prose
    # The terminator rules are stated once, by the protocol's own section.
    assert prose.count("A reply whose first generated") == 1
    assert "stop at `<|endoftext|>`, id 0" in prose
    assert "never put it in the response text" in prose
    assert "`<|fim_pad|>`" not in prose and "<fim_pad>" not in prose
    assert "one token, not zero" in prose
    assert "32,768" in prose
    assert "Mellum2 Base" in prose and "JetBrains/Mellum2-12B-A2.5B-Base" in prose
    assert "`<fim_prefix>`, `<fim_suffix>`, `<fim_middle>`" in prose
    assert "never apply a chat template" in prose
    assert "add_special_tokens=False" in prose
    assert "fim_prompt.txt" in prose
    assert "chat-templated prompt already" not in prose
    assert "prompt_prefix.txt" not in prose and "`reasoning`" not in prose
    assert "linear-attention" not in prose
    assert "removed from the workload" not in prose
