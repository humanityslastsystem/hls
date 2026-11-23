"""Metrics and report-only verifiers for the deepresearch task.

Every metric here is a diagnostic. The task's correctness gate is
``hls.verify.TokenFaithfulnessVerifier``, wired in ``task.py``; nothing in this
module can fail a run.

  * ``RetrievalAnswerEM`` -- dataset benchmark EM against ``event.ground_truth``
    (the rendered dataset label).
  * ``TrajectoryServingMetrics`` -- the serving telemetry axis: e2e / per-step
    latency, TTFT/ITL, GPU utilization.
  * ``TrajectoryCompletionVerifier`` -- how often a trajectory reached an answer
    at all.

``ReportOnlyMetricVerifier`` wraps any ``RolloutMetric`` so its value is surfaced
in the run diagnostics without gating; the completion verifier has its own
report-only subclass.

Answer extraction auto-detects the formats a deep-research trajectory can end in:
``\\boxed{...}`` (the answer convention this task prompts for), an
``<answer>...</answer>`` span, or a thinking model's text after ``</think>``.
Tool-free finalization turns may produce a concise plain answer even when the
model misses ``\\boxed{...}``; those parse only when the caller opts in, so
intermediate free text yields no answer.
"""

from __future__ import annotations
import re

from hls.core.trace import TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict, Verdict
from hls.verify.base import ArtifactRollout, EngineAccessor, RolloutMetric, Verifier


# Auto-detect final-answer spans from raw model text.
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation + articles,
    collapse whitespace. The de-facto standard for open-domain QA exact-match."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_placeholder_boxed_answer(answer: str) -> bool:
    return normalize_answer(answer) in {"", "answer"}


def _looks_incomplete_research_answer(answer: str) -> bool:
    norm = normalize_answer(answer)
    if len(answer.split()) > 80:
        return True
    markers = (
        "let me search",
        "try searching",
        "need to search",
        "i should search",
        "search results",
        "not giving me",
        "more information",
        "different approach",
    )
    return any(marker in norm for marker in markers)


def _looks_hedged_answer(answer: str) -> bool:
    norm = normalize_answer(answer)
    markers = (
        "based on",
        "probably",
        "possibly",
        "i think",
        "likely",
        "uncertain",
    )
    return any(marker in norm for marker in markers)


def _plain_final_answer(text: str) -> str | None:
    answer = text.strip()
    if (not answer
            or _is_placeholder_boxed_answer(answer)
            or _looks_incomplete_research_answer(answer)
            or _looks_hedged_answer(answer)
            or len(answer.split()) > 16):
        return None
    return answer


def extract_final_answer(text: str, *, allow_plain_final: bool = False) -> str | None:
    """Return the short final answer from model text, auto-detecting format.

    Tries ``\\boxed{...}`` first (deep-research convention), then
    ``<answer>...</answer>`` (Search-R1), then Qwen thinking-model final text after
    ``</think>``. Returns ``None`` when none match."""
    if not text:
        return None
    if "<tool_call" in text:
        return None
    for boxed in reversed(list(_BOXED_RE.finditer(text))):
        answer = boxed.group(1).strip()
        if not _is_placeholder_boxed_answer(answer):
            return answer
    for answer in reversed(list(_ANSWER_RE.finditer(text))):
        final_answer = answer.group(1).strip()
        if not _is_placeholder_boxed_answer(final_answer):
            return final_answer
    if "</think>" in text:
        final_answer = text.split("</think>", 1)[1].strip()
        if (final_answer
                and not _is_placeholder_boxed_answer(final_answer)
                and not _looks_incomplete_research_answer(final_answer)):
            return final_answer
    if allow_plain_final:
        return _plain_final_answer(text)
    return None


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile over an already-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _dist(vals: list[float]) -> dict:
    """mean/p50/p90/p99 over ``vals`` (empty -> all zeros)."""
    s = sorted(vals)
    return {
        "mean": (sum(vals) / len(vals)) if vals else 0.0,
        "p50": _percentile(s, 0.50),
        "p90": _percentile(s, 0.90),
        "p99": _percentile(s, 0.99),
    }


def _gpu_sample(event: TraceEvent) -> dict | None:
    """Pull the per-response GPU telemetry snapshot the artifact stamped under
    ``response.output.metadata["gpu"]`` (see ``OpenAIChatProtocol`` decode + the
    artifact servers' ``_GpuSampler``). ``None`` when the artifact emitted no
    sample (no GPU / sampler off / faulted event)."""
    resp = event.response
    if resp is None:
        return None
    gpu = resp.output.metadata.get("gpu")
    return gpu if isinstance(gpu, dict) else None


def _ttft(event: TraceEvent) -> float | None:
    """Per-call time-to-first-token: first content packet arrival minus the
    request send time. ``None`` on the unary path (no ``t_first_packet``) or a
    faulted event. Clamped at 0 to absorb sub-ms clock skew."""
    resp = event.response
    if resp is None or resp.t_first_packet is None:
        return None
    return max(0.0, resp.t_first_packet - event.t_send)


def _itl(event: TraceEvent) -> float | None:
    """Per-call inter-token latency: mean consecutive delta over the streamed
    ``packet_timestamps``. ``None`` unless at least two packets were timed (one
    delta), i.e. the unary path or a single-token response yields no ITL."""
    resp = event.response
    if resp is None or not resp.packet_timestamps or len(resp.packet_timestamps) < 2:
        return None
    ts = resp.packet_timestamps
    deltas = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    return sum(deltas) / len(deltas)


class RetrievalAnswerEM(RolloutMetric):
    """Report-only dataset benchmark EM against ``event.ground_truth``.

    A trajectory "succeeds" iff its terminal answer SQuAD-normalizes to any of
    the dataset's accepted gold answers (``event.ground_truth`` may be a single
    string OR a list of aliases). Wrap in ``ReportOnlyMetricVerifier``: it never
    gates."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        per_event: dict[str, PerEventVerdict] = {}
        for event in events:
            per_event[event.request_id] = self.score_one(event)

        sessions: dict[str, list[TraceEvent]] = {}
        for event in events:
            sessions.setdefault(event.session_id or event.request_id, []).append(event)

        correct_sessions = 0
        terminal_sessions = 0
        for session_events in sessions.values():
            terminal = [
                ev for ev in sorted(session_events, key=lambda e: e.turn_index)
                if not per_event[ev.request_id].diagnostics.get("non_terminal")
            ]
            if not terminal:
                for ev in session_events:
                    base = per_event[ev.request_id]
                    per_event[ev.request_id] = PerEventVerdict(
                        passed=False,
                        score=0.0,
                        rationale="trajectory produced no final answer",
                        diagnostics={
                            **base.diagnostics,
                            "no_terminal_answer": True,
                        },
                    )
                continue
            terminal_sessions += 1
            final_ev = terminal[-1]
            if per_event[final_ev.request_id].passed:
                correct_sessions += 1

        n_sessions = len(sessions)
        value = correct_sessions / n_sessions if n_sessions else 0.0
        return MetricResult(
            value=value,
            per_event=per_event,
            diagnostics={
                "n_events": len(per_event),
                "n_sessions": n_sessions,
                "n_terminal_sessions": terminal_sessions,
                "n_no_answer_sessions": n_sessions - terminal_sessions,
                "n_correct_sessions": correct_sessions,
            })

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        final_answer = event.extras.get("final_answer") if event.extras else None
        if final_answer is None:
            # Intermediate (search) event. The batch reducer groups by session and
            # counts a session with no terminal answer as incorrect.
            return PerEventVerdict(
                passed=True, score=None,
                rationale="non-terminal event; no final answer",
                diagnostics={"non_terminal": True})

        gt = event.ground_truth
        # HotpotQA / NQ sometimes ship multiple acceptable strings; accept a
        # list/tuple of aliases as well as a single rendered string.
        expected_list = gt if isinstance(gt, (list, tuple)) else [gt]
        predicted_norm = normalize_answer(str(final_answer))
        expected_norms = [normalize_answer(str(x)) for x in expected_list if x is not None]
        passed = predicted_norm != "" and predicted_norm in expected_norms
        return PerEventVerdict(
            passed=passed,
            score=1.0 if passed else 0.0,
            rationale=f"predicted={predicted_norm!r}, expected={expected_norms!r}",
            diagnostics={"predicted_norm": predicted_norm,
                         "expected_norms": expected_norms})


class TrajectoryServingMetrics(RolloutMetric):
    """Report-only serving telemetry derived from the trajectory trace.

    Pure reduction over the recorded ``TraceEvent`` timing + per-response
    telemetry; it surfaces the planned serving axes in the rollout record:

      * **end-to-end task latency** -- per-session makespan
        (``max(t_recv) - min(t_send)`` over a session's events): p50/p90/p99/mean.
      * **per-step latency** -- per-LLM-call wall time (``t_recv - t_send`` on each
        turn event): p50/p90/p99/mean.
      * **per-step TTFT** -- time-to-first-token per LLM call
        (``response.t_first_packet - t_send``): p50/p90/p99/mean. Populated only on
        the ``http-sse`` streaming path, where the artifact streams chunks and the
        client records the first content packet's arrival.
      * **per-step ITL** -- inter-token latency per LLM call (mean consecutive
        delta over ``response.packet_timestamps``, averaged within a call, then
        reduced across calls): p50/p90/p99/mean. Also only on the streaming path.
      * **GPU utilization / memory** -- mean+peak over the per-response GPU
        snapshots the artifact stamps under ``response.output.metadata["gpu"]``
        (a lightweight ``nvidia-smi`` sampler inside the artifact container; see
        the task servers' ``_GpuSampler``). Absent when no GPU / sampler off.
      * **counts** -- trajectories (sessions) and total LLM steps.

    TTFT/ITL cost nothing in token fidelity: the streaming codec reassembles the full
    ``choices[].token_ids`` from the per-chunk token ids (see
    ``OpenAIChatProtocol.decode_stream``), so the token-level verifier still reads
    the artifact's exact emitted ids -- streaming captures TIMING without dropping
    the ids.

    Episode throughput + SLO attainment are NOT recomputed here: they are the
    Scorer's job (``TrajectorySessionUnit`` / ``TrajectoryMakespanSLO``).

    ``MetricResult.value`` is the **mean end-to-end task latency (seconds)**; the
    full distributions ride in ``diagnostics``. Always pass at the per-event level
    (it is descriptive, not a gate); wrap with ``ReportOnlyMetricVerifier``."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        sessions: dict[str, list[TraceEvent]] = {}
        for ev in events:
            sessions.setdefault(ev.session_id or ev.request_id, []).append(ev)

        makespans: list[float] = []
        for evs in sessions.values():
            recvs = [e.t_recv for e in evs if e.t_recv is not None]
            if not recvs:
                continue
            makespans.append(max(0.0, max(recvs) - min(e.t_send for e in evs)))

        step_lat = [e.t_recv - e.t_send for e in events if e.t_recv is not None]

        # Per-step TTFT / ITL from the streaming timestamps (None on the unary
        # path). TTFT is the first content packet's arrival relative to the send;
        # ITL is the mean consecutive inter-packet delta within the call.
        ttfts: list[float] = []
        itls: list[float] = []
        gpu_utils: list[float] = []
        gpu_mems: list[float] = []
        for ev in events:
            ttft = _ttft(ev)
            if ttft is not None:
                ttfts.append(ttft)
            itl = _itl(ev)
            if itl is not None:
                itls.append(itl)
            gpu = _gpu_sample(ev)
            if gpu is not None:
                u = gpu.get("utilization_pct")
                m = gpu.get("memory_used_mb")
                if isinstance(u, (int, float)):
                    gpu_utils.append(float(u))
                if isinstance(m, (int, float)):
                    gpu_mems.append(float(m))

        per_event = {
            ev.request_id: PerEventVerdict(
                passed=True,
                score=(ev.t_recv - ev.t_send) if ev.t_recv is not None else None,
                rationale=None,
                diagnostics={
                    "session_id": ev.session_id,
                    "turn_index": ev.turn_index,
                    "step_latency_sec": ((ev.t_recv - ev.t_send)
                                         if ev.t_recv is not None else None),
                    "ttft_sec": _ttft(ev),
                    "itl_sec": _itl(ev),
                    "gpu": _gpu_sample(ev),
                })
            for ev in events
        }

        ms_sorted = sorted(makespans)
        sl_sorted = sorted(step_lat)
        mean_e2e = (sum(makespans) / len(makespans)) if makespans else 0.0
        diagnostics = {
            "n_trajectories": len(sessions),
            "n_steps": len(events),
            "e2e_task_latency_sec": {
                "mean": mean_e2e,
                "p50": _percentile(ms_sorted, 0.50),
                "p90": _percentile(ms_sorted, 0.90),
                "p99": _percentile(ms_sorted, 0.99),
            },
            "per_step_latency_sec": {
                "mean": (sum(step_lat) / len(step_lat)) if step_lat else 0.0,
                "p50": _percentile(sl_sorted, 0.50),
                "p90": _percentile(sl_sorted, 0.90),
                "p99": _percentile(sl_sorted, 0.99),
            },
            # Per-step TTFT/ITL: present once the contract streams (http-sse).
            "ttft_sec": _dist(ttfts),
            "itl_sec": _dist(itls),
            "ttft_itl_available": bool(ttfts or itls),
            # GPU utilization/memory: mean over samples + peak (max).
            "gpu_utilization_pct": {**_dist(gpu_utils),
                                    "peak": max(gpu_utils) if gpu_utils else 0.0},
            "gpu_memory_used_mb": {**_dist(gpu_mems),
                                   "peak": max(gpu_mems) if gpu_mems else 0.0},
            "gpu_utilization_available": bool(gpu_utils),
            "n_gpu_samples": len(gpu_utils),
        }
        return MetricResult(value=mean_e2e, per_event=per_event, diagnostics=diagnostics)


class ReportOnlyMetricVerifier(Verifier):
    """Compute a ``RolloutMetric`` and SURFACE its value for REPORTING, never gating.

    The sibling of the harness ``MetricVerifier``, but with the gate removed.
    ``MetricVerifier`` fails the run when the metric's reduced value drops below
    ``min_value`` AND marks each below-bar event ``passed=False`` -- and the
    verification core ANDs every verifier's per-event flags, so a failing event also loses
    its per-event throughput credit (``Scorer`` counts only events whose combined
    verdict is True). That double coupling -- run gate + throughput credit -- is
    exactly what makes a metric "gate".

    This verifier breaks both: it always returns ``passed=True`` at the run level
    AND ``per_event[rid]=True`` for every event, so it can never flip the
    verification core's AND-combined gate or zero an event's throughput credit. The metric's
    reduced ``value`` (e.g. the EM rate) rides into the combined run-level
    ``Verdict.diagnostics`` (-> ``evals/terminal/verdict.json`` ``diagnostics``
    and, on a passing gate, ``evals/terminal/score.json`` ``breakdown.verdict``),
    and the metric's per-event detail rides into ``event_diagnostics`` /
    ``event_rationale`` (-> ``per_verifier_results`` in ``verdict.json``). The
    metric is reported in full, never gated. No engine -- pure scoring like
    ``MetricVerifier``."""

    metric: RolloutMetric
    #: Optional headline name for the surfaced value (e.g. "task_success_rate"),
    #: rides into the run-level diagnostics so the viewer can label the number.
    label: str | None = None

    async def verify(self, engine: EngineAccessor) -> Verdict:
        r = self.metric.score(engine[ArtifactRollout])
        diagnostics = {"metric": type(self.metric).__name__, "value": r.value,
                       "report_only": True, "gated": False}
        if self.label is not None:
            diagnostics["label"] = self.label
        return Verdict(
            passed=True,
            # All-True so the verification core's AND-combine and throughput credit are untouched.
            per_event={rid: True for rid in r.per_event},
            diagnostics={**diagnostics, **r.diagnostics},
            event_diagnostics={rid: pv.diagnostics for rid, pv in r.per_event.items()},
            event_rationale={rid: pv.rationale for rid, pv in r.per_event.items()
                             if pv.rationale is not None},
        )


class TrajectoryCompletionVerifier(Verifier):
    """Trajectory termination diagnostic: did the contestant emit an answer?

    Report-only via ``ReportOnlyTrajectoryCompletionVerifier``. The correctness gate
    is the token-level faithfulness verifier wired in ``task.py``; this only
    describes how often a trajectory reached an answer at all.

    ``completion_rate`` is intentionally broad: any non-empty terminal
    ``final_answer`` counts, including Qwen fallback text. ``boxed_completion_rate``
    is the stricter deep-research formatting diagnostic.
    """

    min_completion_rate: float = 0.95

    async def verify(self, engine: EngineAccessor) -> Verdict:
        events: list[TraceEvent] = engine[ArtifactRollout]

        sessions: dict[str, list[TraceEvent]] = {}
        for ev in events:
            sessions.setdefault(ev.session_id or ev.request_id, []).append(ev)

        def _is_answer(ev: TraceEvent) -> bool:
            ex = ev.extras or {}
            answer = str(ex.get("final_answer") or "").strip()
            norm = normalize_answer(answer)
            return (ex.get("stop_reason") == "answer"
                    and bool(answer)
                    and norm not in {
                        "no concrete answer extracted",
                        "unable to determine with available evidence",
                    })

        def _is_boxed_answer(ev: TraceEvent) -> bool:
            return _is_answer(ev) and (ev.extras or {}).get("answer_format") == "boxed"

        completed = {sid for sid, evs in sessions.items() if any(_is_answer(e) for e in evs)}
        boxed_completed = {
            sid for sid, evs in sessions.items() if any(_is_boxed_answer(e) for e in evs)
        }
        fallback_completed = completed - boxed_completed

        per_event: dict[str, bool] = {}
        event_rationale: dict[str, str] = {}
        for ev in events:
            sid = ev.session_id or ev.request_id
            ok = sid in completed
            per_event[ev.request_id] = ok
            if not ok:
                event_rationale[ev.request_id] = "trajectory produced no well-formed answer"

        n_sessions = len(sessions)
        completion_rate = (len(completed) / n_sessions) if n_sessions else 0.0
        boxed_completion_rate = (
            len(boxed_completed) / n_sessions) if n_sessions else 0.0
        passed = completion_rate >= self.min_completion_rate

        return Verdict(
            passed=passed,
            per_event=per_event,
            diagnostics={
                "metric": "TrajectoryCompletionVerifier",
                "gated": False,
                "completion_rate": completion_rate,
                "boxed_completion_rate": boxed_completion_rate,
                "min_completion_rate": self.min_completion_rate,
                "n_trajectories": n_sessions,
                "n_completed": len(completed),
                "n_boxed_completed": len(boxed_completed),
                "n_fallback_completed": len(fallback_completed),
            },
            event_rationale=event_rationale,
        )


class ReportOnlyTrajectoryCompletionVerifier(TrajectoryCompletionVerifier):
    """Surface ``TrajectoryCompletionVerifier`` diagnostics without gating."""

    async def verify(self, engine: EngineAccessor) -> Verdict:
        r = await super().verify(engine)
        return Verdict(
            passed=True,
            per_event={rid: True for rid in r.per_event},
            diagnostics={**r.diagnostics, "report_only": True},
            event_rationale=r.event_rationale,
        )


__all__ = ["ReportOnlyMetricVerifier",
           "ReportOnlyTrajectoryCompletionVerifier", "RetrievalAnswerEM",
           "TrajectoryCompletionVerifier", "TrajectoryServingMetrics",
           "extract_final_answer", "normalize_answer"]
