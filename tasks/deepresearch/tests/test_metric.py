"""Unit tests for deepresearch metrics: dataset EM, answer extraction and the completion diagnostic."""

from __future__ import annotations

import asyncio

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.base import EngineAccessor
from tasks.deepresearch.metric import (
    RetrievalAnswerEM,
    ReportOnlyTrajectoryCompletionVerifier, TrajectoryCompletionVerifier,
    TrajectoryServingMetrics, extract_final_answer,
)


def _event(final_answer, *, gt=None, request_id="r",
           session_id="s", stop_reason=None, output_text=None,
           answer_format=None) -> TraceEvent:
    env = Envelope(request_id=request_id, contents=[TextPayload(text="q")])
    if output_text is None:
        output_text = f"Reasoning \\boxed{{{final_answer}}}" if final_answer is not None else ""
    out = Envelope(request_id=request_id, contents=[TextPayload(text=output_text)])
    extras: dict = {}
    if final_answer is not None:
        extras["final_answer"] = final_answer
        extras["stop_reason"] = stop_reason or "answer"
        if answer_format is not None:
            extras["answer_format"] = answer_format
    elif stop_reason is not None:
        extras["stop_reason"] = stop_reason
    return TraceEvent(
        request_id=request_id, session_id=session_id, turn_index=0,
        input=env, response=ArtifactResponse(output=out), ground_truth=gt,
        t_send=0.0, t_recv=1.0, extras=extras,
    )


def test_dataset_em_reports_zero_when_browsecomp_trajectory_never_answers():
    result = RetrievalAnswerEM().score([
        _event(None, request_id="s0-t0", session_id="s0", stop_reason="search"),
        _event(None, request_id="s0-t1", session_id="s0", output_text="still thinking"),
    ])

    assert result.value == 0.0
    assert result.diagnostics["n_terminal_sessions"] == 0
    assert result.diagnostics["n_no_answer_sessions"] == 1


def test_extract_boxed_from_deep_research_output():
    assert extract_final_answer("Long reasoning \\boxed{Paris}") == "Paris"


def test_extract_qwen_answer_after_think():
    assert extract_final_answer("<think>checked evidence</think>\nParis") == "Paris"


def test_extract_plain_answer_only_when_forced_final_context_allows_it():
    assert extract_final_answer("RulinShao") is None
    assert extract_final_answer("RulinShao", allow_plain_final=True) == "RulinShao"


def test_extract_plain_forced_final_rejects_hedged_prose():
    assert extract_final_answer(
        "Based on the evidence, the founder is probably RulinShao.",
        allow_plain_final=True,
    ) is None


def test_extract_rejects_qwen_unfinished_search_planning():
    text = (
        "<think>Still uncertain.</think>\n"
        "The search results are not giving me the specific details. "
        "Let me try searching for more information with different terms."
    )
    assert extract_final_answer(text) is None


# ---- TrajectoryCompletion (report-only diagnostic) -------------------------

def _acc(events):
    return EngineAccessor(artifact_events=events, cap_to_provider={})


def test_report_only_completion_never_gates():
    s0 = _event("A", request_id="s0.1", session_id="s0", stop_reason="answer")
    s1_search = _event(None, request_id="s1.0", session_id="s1", stop_reason="search")
    v = ReportOnlyTrajectoryCompletionVerifier(min_completion_rate=0.75)
    verdict = asyncio.run(v.verify(_acc([s0, s1_search])))
    assert verdict.passed is True
    assert verdict.per_event == {"s0.1": True, "s1.0": True}
    assert verdict.diagnostics["report_only"] is True


def test_gate_passes_when_all_trajectories_complete():
    s0 = _event("A", request_id="s0.1", session_id="s0", stop_reason="answer",
                answer_format="boxed")
    s1 = _event("B", request_id="s1.1", session_id="s1", stop_reason="answer",
                answer_format="qwen_fallback")
    v = TrajectoryCompletionVerifier(min_completion_rate=0.5)
    verdict = asyncio.run(v.verify(_acc([s0, s1])))
    assert verdict.passed is True
    assert verdict.diagnostics["completion_rate"] == 1.0
    assert verdict.diagnostics["boxed_completion_rate"] == 0.5
    assert verdict.diagnostics["n_boxed_completed"] == 1
    assert verdict.diagnostics["n_fallback_completed"] == 1


def test_completion_diagnostic_does_not_count_empty_forced_fallback():
    s0 = _event("A", request_id="s0.1", session_id="s0", stop_reason="answer")
    s1 = _event("No concrete answer extracted", request_id="s1.1",
                session_id="s1", stop_reason="answer",
                answer_format="forced_empty_fallback")
    v = TrajectoryCompletionVerifier(min_completion_rate=0.75)
    verdict = asyncio.run(v.verify(_acc([s0, s1])))
    assert verdict.passed is False
    assert verdict.diagnostics["completion_rate"] == 0.5
    assert verdict.diagnostics["n_completed"] == 1
    assert verdict.per_event["s1.1"] is False


def test_completion_diagnostic_default_requires_high_answer_rate():
    assert TrajectoryCompletionVerifier().min_completion_rate == 0.95
    assert ReportOnlyTrajectoryCompletionVerifier().min_completion_rate == 0.95


def test_gate_fails_session_with_no_answer():
    s0 = _event("A", request_id="s0.1", session_id="s0", stop_reason="answer")
    s1_search = _event(None, request_id="s1.0", session_id="s1", stop_reason="search")
    v = TrajectoryCompletionVerifier(min_completion_rate=0.75)
    verdict = asyncio.run(v.verify(_acc([s0, s1_search])))
    assert verdict.passed is False
    assert verdict.per_event["s1.0"] is False


# ---- TrajectoryServingMetrics ----------------------------------------------

def test_serving_metrics_surfaces_streaming_axes():
    env = Envelope(request_id="s.0", contents=[TextPayload(text="q")],
                   metadata={"gpu": {"utilization_pct": 88.0, "memory_used_mb": 9000.0}})
    resp = ArtifactResponse(output=env, t_first_packet=0.2,
                            packet_timestamps=[0.2, 0.4, 0.7])
    ev = TraceEvent(request_id="s.0", session_id="s", turn_index=0, input=env,
                    response=resp, ground_truth=None, t_send=0.0, t_recv=0.7, extras={})
    r = TrajectoryServingMetrics().score([ev])
    assert r.diagnostics["ttft_itl_available"] is True
    assert abs(r.diagnostics["ttft_sec"]["mean"] - 0.2) < 1e-9
    assert r.diagnostics["gpu_utilization_available"] is True
