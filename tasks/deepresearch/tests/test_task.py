"""Spec-construction + wiring tests for deepresearch (no GPU, no secrets).

Asserts the task module imports, the TaskSpec is well-formed, the agentic seam
points at DeepResearchFramework, the verifiers wire cleanly against the contract,
and the deep-research dataset registry / system prompt are intact.
"""

from __future__ import annotations

import asyncio
import json

from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from hls.reference.engines import SGLangEngine, VLLMEngine
from hls.spec.baseline import Baseline
from hls.spec.tools import LocalCallableToolRunner
from hls.verify import ReportFaithfulnessVerifier, TokenFaithfulnessVerifier
from hls.verify.base import check_wiring

from tasks.deepresearch import datasets as ds
from tasks.deepresearch import retriever
from tasks.deepresearch import task as t
from tasks.deepresearch.framework import DeepResearchFramework
from tasks.deepresearch.metric import (
    ReportOnlyMetricVerifier, ReportOnlyTrajectoryCompletionVerifier,
)
from tests.unit.brief_task import render_brief


def test_task_identity_and_theme():
    assert t.TASK.id == "deepresearch"
    assert t.TASK.metadata["theme"] == "deep-research"
    assert t.TASK.metadata["harness"] == "DeepResearch"
    assert t.TASK.metadata["prefix_cache_profile"] == "EXTREME"
    assert t.MODEL_ID == "google/gemma-4-31B-it"
    assert t.MODEL.base_checkpoint == t.MODEL_ID
    assert t.TASK.metadata["model_status"] == "gemma-4-31b-it-deepresearch"


def test_agent_seam_uses_deepresearch_framework():
    assert t.TASK.agent is not None
    assert isinstance(t.TASK.agent.framework, DeepResearchFramework)
    assert isinstance(t.TASK.agent.tool_runner, LocalCallableToolRunner)
    assert t.MAX_TURNS == 100
    assert t.MAX_SEARCH_TURNS == 80
    assert t.TASK.agent.max_turns == t.MAX_TURNS
    assert t.TASK.agent.framework.finalize_after_search_turns == t.MAX_SEARCH_TURNS
    # The arrival process alone decides how much work the server sees at once.
    assert t.TASK.agent.max_concurrent_llm is None
    # Corpus browser semantics: search results can be opened and read.
    assert [s.name for s in t.TASK.agent.tool_runner.describe_tools()] == [
        "search", "open", "find"]


def test_min_free_gpu_bytes_exceeds_what_either_calibration_engine_leaves_free():
    """The CPU-fallback threshold must beat what a vLLM or SGLang pool member leaves
    free on the declared accelerator, so the retriever never contends with the
    engine under measurement for GPU memory."""
    device_bytes = t.HARDWARE.accelerator.memory_gb_per_device * 2**30
    vllm_util = VLLMEngine.__init__.__kwdefaults__["gpu_memory_utilization"]
    sglang_util = SGLangEngine.__init__.__kwdefaults__["mem_fraction_static"]
    vllm_free = (1 - vllm_util) * device_bytes
    sglang_free = (1 - sglang_util) * device_bytes
    assert retriever.MIN_FREE_GPU_BYTES > max(vllm_free, sglang_free)


def test_the_agent_declares_the_browsecomp_bundle_and_a_gpu_driver():
    from tasks.deepresearch.assets import BrowseCompBundle
    assert isinstance(t.AGENT.assets, BrowseCompBundle)
    assert t.AGENT.assets.name == "browsecomp" and t.AGENT.assets.gpu is True
    assert t.AGENT.assets.memory_mb >= 40_000


def test_inter_turn_delay_approximates_live_tool_latency():
    from hls.spec.workload import GammaInterTurn
    d = t.WORKLOAD.inter_turn_delay
    assert isinstance(d, GammaInterTurn)
    assert (d.shape, d.scale, d.max_gap) == (2.0, 1.0, 15.0)


def test_the_tool_runner_warms_the_browser():
    assert t.TOOL_RUNNER._warmup == t._BROWSER.warmup


def test_the_framework_warms_its_prompt_tokenizer(monkeypatch):
    """The framework's own pre-clock load, which the driver awaits beside the tool
    runner's. Loading it lazily would bill about a second to whichever trajectory
    finished its first turn first."""
    from tasks.deepresearch import framework as fw
    loaded = []
    monkeypatch.setattr(fw, "prompt_tokenizer", lambda: loaded.append("tokenizer"))

    asyncio.run(t.TASK.agent.framework.warmup())

    assert loaded == ["tokenizer"]


def test_the_task_scores_speedup_over_the_baseline_serving_stack():
    """The objective is speedup in trajectory makespan, not goodput, so the task must
    declare the latency scorer and the CLI must be told to require a baseline credit."""
    from hls.scoring import SloLatencyScorer
    from hls.spec.workload import TrajectoryMakespanSLO

    assert isinstance(t.SCORER, SloLatencyScorer)
    assert t.TASK.scorer is t.SCORER
    assert t.TASK.scorer.scores_against_baseline() is True
    # None is authored: the CLI injects the measured credit from baseline.json.
    assert t.SCORER.baseline_log_credit is None
    # The scorer means over trajectories because the SLO's grain is the session.
    assert isinstance(t.TASK.workload.slo, TrajectoryMakespanSLO)
    assert t.TASK.workload.slo.grain == "session"
    assert t.TASK.workload.slo.target_sec == t.TRAJECTORY_BUDGET_SEC


def test_the_brief_states_the_speedup_objective_and_never_goodput():
    """The base template's goal block says the score is goodput, which is not this
    task's objective; the override has to reach the agent that reads it."""
    text, _ = render_brief(t.TASK, t.ARTIFACT.brief)

    assert ("**Your score is your speedup over a production serving framework in "
            "trajectory\nmakespan.**") in text
    assert "r = makespan / 1,800 s" in text
    assert "score = exp( mean over trajectories of log(1/r) )" in text
    assert "**Your implementation must not fail any request.**" in text
    assert "goodput" not in text.lower()
    # The task's own workload block still renders under the override.
    assert "**Shape of a trajectory.**" in text
    assert "poisson arrivals" in text


def _pinned_serve_command_kwargs(serve_command: list[str]) -> dict:
    """The chat-template pin a serve command carries, decoded from its flag value."""
    index = serve_command.index("--chat-template-kwargs")
    return json.loads(serve_command[index + 1])


def test_the_baseline_carries_only_the_chat_template_pin_as_a_launch_flag():
    """The measured baseline's server parses ``--chat-template-kwargs``. The decode is pinned in
    the servers rather than in the launch, so a baseline command that differed from an agent's
    launch could not exist."""
    assert isinstance(t.TASK.baseline, Baseline)
    assert t.TASK.baseline is t.BASELINE
    assert t.TASK.baseline.image == "hls-baseline-deepresearch"
    assert t.CHAT_TEMPLATE_KWARGS_FLAG == "--chat-template-kwargs"
    assert [a for a in t.BASELINE.serve_command if a.startswith("--")] == [t.CHAT_TEMPLATE_KWARGS_FLAG]
    assert _pinned_serve_command_kwargs(t.BASELINE.serve_command) == t.MODEL.chat_template_kwargs


def test_the_only_gate_is_the_report_faithfulness_verifier_sampling_at_temperature_one():
    """One pass/fail gate, the report gate over a reference sampling as the servers do.
    Everything else the task declares is report-only, so nothing but faithfulness can fail
    a run on correctness, and the token gate, which cannot grade a sampled decode, is absent."""
    from hls.core.decode import Sample

    gates = [v for v in t.TASK.verifiers
             if not isinstance(v, (ReportOnlyMetricVerifier,
                                   ReportOnlyTrajectoryCompletionVerifier))]
    assert [type(v).__name__ for v in gates] == ["ReportFaithfulnessVerifier"]
    assert isinstance(gates[0], ReportFaithfulnessVerifier)
    assert gates[0].reference.decode == t.DECODE == Sample(temperature=1.0)
    assert not [v for v in t.TASK.verifiers if isinstance(v, TokenFaithfulnessVerifier)]
    assert [type(v).__name__ for v in t.TASK.verifiers if v.needs_calibration] == ["ReportFaithfulnessVerifier"]


def test_the_contract_asks_for_the_report():
    """The gate reads the served distribution off the reply, with the served token ids, so the
    contract has to request both."""
    assert t.CONTRACT.protocol.emits_token_ids is True
    assert t.TASK.contract.protocol.return_logprobs is True
    assert t.TASK.contract.protocol.top_logprobs == 21


def test_report_only_completion_threshold():
    report_completion = [v for v in t.TASK.verifiers
                         if isinstance(v, ReportOnlyTrajectoryCompletionVerifier)]
    assert len(report_completion) == 1
    assert report_completion[0].min_completion_rate == 0.95


def test_verifiers_wire_against_contract():
    check_wiring(t.TASK.verifiers, t.CONTRACT)


def test_the_task_grades_fifty_sessions_of_two_hundred_dev_rows():
    """A 0.9-hour drive at the arrival rate."""
    assert t.DATA_SOURCE.split.dev.value == 200
    assert t.TASK.data_source.split.test.value == t.TEST_SESSIONS == 50


def test_dataset_and_system_prompt():
    assert ds.BROWSECOMP.path == "Tevatron/browsecomp-plus"
    assert isinstance(ds.BROWSECOMP, ds.BrowseCompQADataset)
    # Deep-research answer convention is the boxed short answer.
    assert "\\boxed{" in ds.DEEPRESEARCH_SYSTEM
    # The baseline must steer the model toward the full BrowserSession toolset,
    # not a snippet-only approximation of deep research.
    assert "web_search" in ds.DEEPRESEARCH_SYSTEM
    assert "web_open" in ds.DEEPRESEARCH_SYSTEM
    assert "web_find" in ds.DEEPRESEARCH_SYSTEM
    assert "opened primary or high-credibility sources" in ds.DEEPRESEARCH_SYSTEM
    # Tool schemas are rendered by the tokenizer; literal placeholder tags make
    # the model search for the word "QUERY".
    assert "<search>" not in ds.DEEPRESEARCH_SYSTEM
    assert "QUERY" not in ds.DEEPRESEARCH_SYSTEM


def test_default_task_uses_browsecomp():
    assert t.DATASET is ds.BROWSECOMP
    assert t.DATA_SOURCE.dataset.path == "Tevatron/browsecomp-plus"


def test_dataset_gold_answers_are_the_answer_field_as_one_string():
    browsecomp = ds.BROWSECOMP
    assert browsecomp._gold_answers({"answer": "Paris"}) == ["Paris"]
    assert browsecomp._gold_answers({"answer": 1999}) == ["1999"]
    assert browsecomp._gold_answers({"answer": ""}) == []
    assert browsecomp._gold_answers({}) == []


def test_the_model_pins_thinking_in_the_gemma_format_the_framework_reads():
    assert t.MODEL.chat_template_kwargs == {"enable_thinking": True}
    assert t.MODEL.chat_format == "gemma"
    assert t.TASK.agent.framework.chat_format == t.TASK.model.chat_format


def test_the_framework_budgets_against_the_window_the_task_declares():
    """One window: ``TaskSpec.max_context_len``, which the servers are told to serve in,
    the calibration pool's members are built at, and the workload payload names. The drive
    stamps it on every turn's ``HarnessContext``, beside the model's chat-template pin, and
    the trajectory loop stops a trajectory on that, so no window can change the
    trajectories without changing the workload fingerprint. The loop does not read the
    window a reply reports, because that would let a server shorten every trajectory, and
    the task scores trajectory makespan."""
    import dataclasses

    from hls.spec.drivers import driver_for

    assert "max_context_len" not in {f.name for f in dataclasses.fields(DeepResearchFramework)}
    assert t.MAX_CONTEXT_LEN == 131072
    task = t.TASK
    assert task.max_context_len == t.MAX_CONTEXT_LEN
    driver = driver_for(task.agent, request_timeout_sec=REQUEST_TIMEOUT_SEC,
                        max_context_len=task.max_context_len,
                        chat_template_kwargs=task.model.chat_template_kwargs)
    ctx = driver.harness_context("t0", max_output_tokens=None)
    assert ctx.max_context_len == t.MAX_CONTEXT_LEN
    assert dict(ctx.chat_template_kwargs) == dict(task.model.chat_template_kwargs)


def test_a_copy_with_another_window_leaves_the_original_task_and_its_framework_alone():
    """The framework object is shared by every copy of a TaskSpec, so it holds no window and
    no pin: a copy with another window drives at its own, and the original, and the module's
    TASK, still drive at theirs."""
    from hls.spec.drivers import driver_for
    from hls.taskref import TaskRef, load_task
    from hls.verify.calibration import workload_fingerprint

    module_fingerprint = workload_fingerprint(t.TASK)
    base = load_task(TaskRef(module="tasks.deepresearch.task"))
    framework_state = dict(vars(base.agent.framework))
    moved = load_task(TaskRef(module="tasks.deepresearch.task",
                              overrides={"task.max_context_len": 32_768}))
    assert moved.agent.framework is base.agent.framework
    assert dict(vars(base.agent.framework)) == framework_state
    assert not {"_max_context_len", "_chat_template_kwargs"} & set(framework_state)
    for task, window in ((base, t.MAX_CONTEXT_LEN), (moved, 32_768), (t.TASK, t.MAX_CONTEXT_LEN)):
        driver = driver_for(task.agent, request_timeout_sec=REQUEST_TIMEOUT_SEC,
                            max_context_len=task.max_context_len,
                            chat_template_kwargs=task.model.chat_template_kwargs)
        assert driver.harness_context("t0", max_output_tokens=None).max_context_len == window
    assert workload_fingerprint(t.TASK) == workload_fingerprint(base) == module_fingerprint
    assert workload_fingerprint(moved) != module_fingerprint


def test_a_model_the_framework_cannot_read_is_refused_at_construction():
    """The chat format the framework reads and the model's have to agree. A mismatch
    raises nothing at run time: every reply parses with the wrong delimiters, so the
    trajectory looks like a weak model rather than a wiring mistake."""
    import pytest
    from pydantic import ValidationError

    other = t.MODEL.model_copy(update={"chat_format": "chatml"})
    with pytest.raises(ValidationError) as excinfo:
        t.TASK.model_copy(update={"model": other})
    message = str(excinfo.value)
    assert "'chatml'" in message and "'gemma'" in message


def test_the_arrival_rate_budget_and_output_cap_are_the_tasks():
    """The trajectory budget is 30 minutes: the sampled decode thinks at length, and the budget
    only normalizes the score. Sampling leaves no greedy token cycles to bound, so the output cap
    binds the genuine long tail only."""
    assert t.TASK.workload.arrival.rate_per_sec == t.POISSON_RATE_PER_SEC == 0.015
    assert t.TASK.workload.slo.target_sec == t.TRAJECTORY_BUDGET_SEC == 1800.0
    assert t.MAX_NEW_TOKENS == 8192
    assert t.TASK.workload.output_length_cap.cap_tokens == t.MAX_NEW_TOKENS
    assert t.TASK.workload.output_length_cap.max_tokens() == t.MAX_NEW_TOKENS


def test_the_task_budgets_a_twelve_hour_build():
    soft = t.ARTIFACT.producer.soft_budget
    hard = t.ARTIFACT.producer.hard_budget

    assert soft.wallclock_sec == 12 * 3600
    assert hard.wallclock_sec > soft.wallclock_sec
    assert soft.turns is None and hard.turns is None


def test_the_framework_reports_the_tool_role_protocol_to_the_fingerprint():
    """The tool-result protocol decides what every prompt after a tool call carries, so the
    framework reports it with its settings."""
    from hls.verify.calibration import workload_payload

    config = t.TASK.agent.framework.traffic_config()
    assert config["tool_result_protocol"] == "tool_role"
    assert workload_payload(t.TASK)["agent"]["framework_config"] == config
