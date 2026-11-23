"""Each task declares what one of its drives may make the harness hold (``DriveBounds`` on its
TaskSpec): the cap on its trace, derived from its measured drives by one rule (``trace_cap``),
and the most bytes of requests its trace holds beside the replies. How much trace a reply's byte
can become is its wire protocol's to say (``WireProtocol.trace_growth``), checked here against
what the trace's writer makes of the densest replies each decoder accepts; a request in flight
is held at the JSON its event takes, measured as the writer writes it. A task that declares
no bounds cannot be evaluated, and the bounds stay out of every dump and fingerprint of the task."""

from __future__ import annotations

import importlib
import json
import math

import pytest

from hls.artifact.artifact_client import sse_line_bytes, trace_growth
from hls.core.payloads import AudioPayload, Envelope, ImagePayload, TextPayload
from hls.core.poly import dump_spec
from hls.core.trace import (ArtifactResponse, FailureReason, PartialOutput, RequestAttempt, RequestFailure,
                            TraceEvent)
from hls.drive.bounds import (REQUEST_EVENT_BYTES, VERIFY_MEMORY_BYTES, VERIFY_SHARE, DriveBounds, UnknownTask,
                              VerifyPeak, bounds_for, drive_bounds, request_trace_bytes, trace_cap, trace_disk_bytes,
                              trace_event_bytes, verify_cap, verify_peak_bytes)
from hls.pipeline.pipelines import REFERENCE_SHAPE
from hls.settings import RunSettings
from hls.spec.contracts import ArtifactContract, RequestResponseContract
from hls.spec.drivers import stamp_request, user_message
from hls.spec.wire import OmniChatProtocol, OpenAIChatProtocol, TTSStreamProtocol
from hls.taskref import TaskRef, load_task
from hls.trace.store import (ESCAPE_GROWTH, TIMESTAMP_JSON_BYTES, SummaryEvent, _event_line, _json_line,
                             input_json_bytes, room_bytes)
from hls.verify.calibration import workload_fingerprint

GIB, MIB = 2**30, 2**20

#: Each graded task module: its trace cap, the most bytes of requests its trace holds, and the
#: trace its replies' bytes can grow into per byte.
TABLE = {
    "tasks.multitenant_slo.task": (6 * GIB, 160 * MIB, 13.5),
    "tasks.tab_completion.task": (6 * GIB, 384 * MIB, 13.5),
    "tasks.deepresearch.task": (3 * GIB, 160 * MIB, 13.5),
    "tasks.multimodal_asr.task": (3 * GIB, GIB, 6.0),
    "tasks.multimodal_omni.task": (3 * GIB, 2 * GIB, 6.0),
    "tasks.multimodal_tts.task": (61 * GIB, 4 * MIB, 127 / 12),
    "tasks.sample_task.task": (GIB, MIB, 6.0),
    "tasks.agentic_coding.task": (22 * GIB, 640 * MIB, 13.5),
}
#: Each graded task module's verify peak as it declares it, from a verify peak measurement: the GiB its verify role holds whatever the trace, the
#: bytes it holds besides per byte of the trace's events at the worst composition measured, and
#: whether that, not the cap rule, sets its cap.
PEAKS = {
    "tasks.multitenant_slo.task": (52.55, 12.38, False),
    "tasks.tab_completion.task": (24.0, 12.46, False),
    "tasks.deepresearch.task": (59.25, 26.3, True),
    "tasks.multimodal_asr.task": (19.25, 1.11, False),
    "tasks.multimodal_omni.task": (81.93, 13.6, True),
    "tasks.multimodal_tts.task": (29.4, 2.11, True),
    # Inferred, not measured: the checkpoint's 306 GiB and 1 GiB, and deepresearch's report-gate slope.
    "tasks.agentic_coding.task": (307.0, 26.3, False),
}
#: The fixture tasks the evaluate, credit and build tests plan, by the expression that makes
#: each, and their figures: no measured drive, so the least cap.
FIXTURES = {
    "tests.evaluate.fixture:evaluate_task": (GIB, MIB, 6.0),
    "tests.evaluate.fixture:agent_task": (GIB, MIB, 6.0),
    "tests.credit.fixture:credit_task": (GIB, MIB, 6.0),
    "tests.agent.build_fixture:build_task": (GIB, MIB, 6.0),
}


def _fixture(name: str):
    module, function = name.split(":")
    return getattr(importlib.import_module(module), function)()


def _figures(task) -> tuple[int, int, float]:
    bounds = drive_bounds(task)
    return bounds.trace_cap_bytes, bounds.trace_request_bytes, trace_growth(task.contract)


@pytest.mark.parametrize("module", sorted(TABLE))
def test_each_task_declares_its_bounds(module):
    assert _figures(load_task(TaskRef(module=module))) == pytest.approx(TABLE[module])


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_each_fixture_task_declares_its_bounds(name):
    assert _figures(_fixture(name)) == pytest.approx(FIXTURES[name])


def _stream(protocol) -> RequestResponseContract:
    return RequestResponseContract(path="/v1/chat/completions", input_modalities=frozenset({"text"}),
                                   output_modalities=frozenset({"text"}), transport="http-sse", protocol=protocol)


def _unary(protocol) -> RequestResponseContract:
    return _stream(protocol).model_copy(update={"transport": "http"})


def test_the_cap_rule():
    """Four times the task's largest measured trace, or that trace with every reply byte of its
    largest measured drive held unwritten at once at the protocol's growth, whichever is more,
    rounded up to whole GiB and never under one."""
    contract = _stream(OpenAIChatProtocol())
    assert trace_cap(contract, trace_bytes=0, held_reply_bytes=0) == GIB
    assert trace_cap(contract, trace_bytes=GIB, held_reply_bytes=0) == 4 * GIB
    assert trace_cap(contract, trace_bytes=GIB + 1, held_reply_bytes=0) == 5 * GIB
    assert trace_cap(contract, trace_bytes=GIB, held_reply_bytes=GIB) == 15 * GIB
    assert trace_cap(_unary(OpenAIChatProtocol()), trace_bytes=GIB, held_reply_bytes=GIB) == 7 * GIB


def test_each_tasks_bounds_are_the_rule_on_its_measured_figures():
    """The figures each task module cites: its largest measured trace, and the reply bytes of its
    largest measured drive on the wire (the stored reply bytes at the task's measured wire bytes
    per stored byte; for deepresearch, its fifty trajectories' calls at once at its largest
    measured reply), beside its requests and its measured verify peak (``bounds_for``)."""
    measured = {
        "tasks.multitenant_slo.task": (64_342_088, 462_268_100),
        "tasks.tab_completion.task": (359_684_131, 445_601_525),
        "tasks.deepresearch.task": (1_623_366_771, 50 * 15_518_126),
        "tasks.multimodal_asr.task": (760_508_726, 1_095_914),
        "tasks.multimodal_omni.task": (1_607_516_171, 1_171_424),
        "tasks.multimodal_tts.task": (22_886_907_405, 7_774_318_266),
        "tasks.agentic_coding.task": (4_935_885_306, 24 * 54_743_852),
    }
    for module, (trace, held) in measured.items():
        task = load_task(TaskRef(module=module))
        bounds = drive_bounds(task)
        assert bounds == bounds_for(task.contract, trace_bytes=trace, held_reply_bytes=held,
                                    trace_request_bytes=bounds.trace_request_bytes,
                                    verify_peak=bounds.verify_peak,
                                    verify_memory_bytes=bounds.verify_memory_bytes), module
        assert bounds.trace_cap_bytes == min(trace_cap(task.contract, trace_bytes=trace, held_reply_bytes=held),
                                             verify_cap(bounds.verify_peak,
                                                        trace_request_bytes=bounds.trace_request_bytes,
                                                        verify_memory_bytes=bounds.verify_memory_bytes)), module


@pytest.mark.parametrize("make", [lambda: _fixture("tests.evaluate.fixture:unbounded_task"),
                                  lambda: _fixture("tests.finalize.fixture:latency_task")],
                         ids=["evaluate-unbounded", "finalize-latency"])
def test_a_task_that_declares_no_bounds_is_refused(make):
    task = make()
    with pytest.raises(UnknownTask, match=f"no drive bounds for task {task.id!r}") as raised:
        drive_bounds(task)
    assert isinstance(raised.value, ValueError)  # a refusal: a command planning the task says so in one line


def test_the_deepresearch_bounds_are_at_its_output_cap():
    task = load_task(TaskRef(module="tasks.deepresearch.task"))
    assert drive_bounds(task).trace_cap_bytes == 3 * GIB
    assert [t.workload.output_length_cap.cap_tokens for t in task.effective_tenants()] == [8192]


@pytest.mark.parametrize("module", sorted(TABLE))
def test_the_bounds_stay_out_of_every_dump_and_fingerprint(module):
    """A record's spec, a role's task file and the workload fingerprint the committed calibrations
    and credits are keyed by read the task's dump or its payload; none carries the bounds, so
    declaring them moved none."""
    task = load_task(TaskRef(module=module))
    bare = task.model_copy(update={"drive_bounds": None})
    assert "drive_bounds" not in dump_spec(task) and "drive_bounds" not in task.model_dump_json()
    assert dump_spec(task) == dump_spec(bare)
    assert workload_fingerprint(task) == workload_fingerprint(bare)


def test_bounds_are_positive():
    peak = VerifyPeak(base_bytes=0, bytes_per_byte=1.0)
    for field in ("trace_cap_bytes", "trace_request_bytes"):
        with pytest.raises(ValueError, match="greater than 0"):
            DriveBounds(**{"trace_cap_bytes": 1, "trace_request_bytes": 1, "verify_peak": peak,
                           "verify_limited": False, field: 0})
    with pytest.raises(ValueError, match="greater than 0"):
        VerifyPeak(base_bytes=0, bytes_per_byte=0.0)


def test_the_growth_of_each_protocol():
    assert trace_growth(_unary(OpenAIChatProtocol())) == ESCAPE_GROWTH == 6
    assert trace_growth(_stream(OpenAIChatProtocol())) == (2 + TIMESTAMP_JSON_BYTES) / 2 == 13.5
    tts = TTSStreamProtocol(return_codec_codes=True, return_logprobs=True, top_logprobs=65)
    assert trace_growth(_stream(tts)) == pytest.approx(127 / 12)
    assert trace_growth(_stream(TTSStreamProtocol())) == ESCAPE_GROWTH
    with pytest.raises(ValueError, match="OmniChatProtocol"):
        trace_growth(_stream(OmniChatProtocol()))
    bare = ArtifactContract(path="/x", input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}))
    with pytest.raises(ValueError, match="ArtifactContract"):
        trace_growth(bare)


#: The longest ``repr`` a float has.
LONGEST_FLOAT = -2.2250738585072014e-308


def _trace_bytes(response) -> int:
    """The bytes of the line an event carrying *response* takes in the trace's chunks."""
    ev = TraceEvent(request_id="r.0", input=Envelope(request_id="r.0"), response=response, t_send=0.0, t_recv=1.0)
    return len(_event_line(ev))


def _growth(build, wire) -> float:
    """The trace bytes one more unit of a reply adds per wire byte it adds, from *build*, the
    decoded reply of n units, and *wire*, its bytes on the wire."""
    small, large = 100, 300
    return (_trace_bytes(build(large)) - _trace_bytes(build(small))) / (wire(large) - wire(small))


def test_the_densest_token_stream_grows_at_its_protocols_growth():
    """Ids one digit long, all in one frame, each timed at a stamp of the longest repr: the event
    keeps each id and its time."""
    protocol = OpenAIChatProtocol()

    def chunks(n):
        return [{"choices": [{"index": 0, "delta": {}, "token_ids": [0] * n}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]

    def build(n):
        return protocol.decode_stream([(LONGEST_FLOAT, c) for c in chunks(n)], "r.0")

    def wire(n):
        return sum(sse_line_bytes(c) for c in chunks(n))

    assert _growth(build, wire) == pytest.approx(trace_growth(_stream(protocol)))


def test_a_whole_reply_of_escaped_characters_grows_at_its_protocols_growth():
    protocol = OpenAIChatProtocol()

    def body(n):
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "\x7f" * n},
                             "finish_reason": "stop", "token_ids": [0]}]}

    def build(n):
        return ArtifactResponse(output=protocol.decode_response(body(n), "r.0"))

    def wire(n):
        # A DEL travels raw: one byte of the body.
        return len(json.dumps(body(n), ensure_ascii=False, separators=(",", ":")).encode())

    assert _growth(build, wire) == pytest.approx(trace_growth(_unary(protocol)))


def test_the_densest_codec_report_grows_at_its_protocols_growth():
    """Positions of one alternative, each id one digit at a log-probability of zero, the densest a
    position's probability mass allows: the decoder keeps the report, and the event grows by it at
    the growth the protocol states, less the brackets of each frame's row, and the reply as a whole
    by less."""
    from hls.core.audio import pcm16_wav_b64
    reporting = TTSStreamProtocol(return_codec_codes=True, return_logprobs=True, top_logprobs=65)
    plain = TTSStreamProtocol(return_codec_codes=True)
    eos = reporting.codec_eos_id

    def chunks(n, report):
        frames = [{"audio": pcm16_wav_b64(b"\0\0")} for _ in range(n)]
        codes = {"codec_codes": [[0] * 16 for _ in range(n)]}
        if report:
            codes.update(codec_report=[[[0, [[0, 0]]] for _ in range(16)] for _ in range(n)],
                         eos_report=[0, [[eos, 0]]], eos_code=eos)
        return [*frames, codes]

    def build(n, report=True):
        protocol = reporting if report else plain
        response = protocol.decode_stream([(1.0, c) for c in chunks(n, report)], "r.0")
        payload = response.output.contents[0]
        assert (payload.codec_logprobs is not None) is report, payload.codec_codes_defect
        return response

    def wire(n, report=True):
        return sum(sse_line_bytes(c) for c in chunks(n, report))

    def report_growth(small, large):
        trace = [_trace_bytes(build(n)) - _trace_bytes(build(n, report=False)) for n in (small, large)]
        wired = [wire(n) - wire(n, report=False) for n in (small, large)]
        return (trace[1] - trace[0]) / (wired[1] - wired[0])

    # The protocol counts a position alone; on the wire each frame's row adds its brackets.
    growth = trace_growth(_stream(reporting))
    assert 0.98 * growth < report_growth(100, 300) <= growth
    assert _growth(build, wire) < growth


def test_a_summary_line_is_never_longer_than_its_events():
    """So the summary holds no more than the chunks, and a cap on the chunks bounds both."""
    from hls.core.trace import FailureReason, RequestAttempt, RequestFailure
    for metadata in ({}, {"tenant_id": "t"}):
        bare = TraceEvent(request_id="", input=Envelope(request_id="", metadata=metadata), t_send=0.0)
        answered = bare.model_copy(update={"response": ArtifactResponse(output=Envelope(request_id=""),
                                                                        packet_timestamps=[LONGEST_FLOAT] * 3,
                                                                        t_first_packet=LONGEST_FLOAT)})
        failed = bare.model_copy(update={"error": "e", "failure": RequestFailure(
            reason=FailureReason.DEADLINE, t_abandoned=0.0, observed=RequestAttempt())})
        for ev in (bare, answered, failed):
            summary = _json_line(SummaryEvent.of_event(ev).model_dump(mode="python"))
            assert len(summary) <= len(_event_line(ev))


@pytest.mark.parametrize("module", sorted(TABLE))
def test_a_drives_disk_is_the_room_its_trace_is_written_into(module):
    """The driver's node-local disk is the room the stage makes for the trace before the server
    starts: the most its chunks and summary take in gzip, with its events' JSON at the cap and the
    requests beside it (``trace_event_bytes``)."""
    cap, requests, _ = TABLE[module]
    room = room_bytes(trace_event_bytes(cap, requests))
    assert trace_disk_bytes(cap, requests) == sum(room.values())


# What the verify role holds of a trace, and the cap that keeps it inside its memory.


def test_the_verify_peak_rule():
    peak = VerifyPeak(base_bytes=10 * GIB, bytes_per_byte=12.5)
    assert verify_peak_bytes(peak, GIB) == 10 * GIB + math.ceil(12.5 * GIB)
    assert verify_peak_bytes(peak, 0) == 10 * GIB


def test_the_verify_role_has_the_reference_jobs_memory_less_the_runners_reserve():
    """The reference job's 180 GiB, less the runner's default reserve: what
    ``hls.pipeline.pipelines.reference_shape`` gives the verify role."""
    assert VERIFY_MEMORY_BYTES == (REFERENCE_SHAPE.memory_gib - RunSettings().runner_memory_gib) * GIB
    assert VERIFY_SHARE == 0.9


def test_a_cap_is_the_largest_whole_gib_whose_verify_fits():
    """The verify role's peak over a trace at the cap and its requests stays within
    ``VERIFY_SHARE`` of its memory: the largest whole GiB that does, or the cap rule's
    cap when that is less."""
    budget = VERIFY_SHARE * VERIFY_MEMORY_BYTES
    peak = VerifyPeak(base_bytes=50 * GIB, bytes_per_byte=20.0)
    cap = verify_cap(peak, trace_request_bytes=100 * MIB)
    assert cap % GIB == 0 and cap >= GIB
    assert verify_peak_bytes(peak, cap + 100 * MIB) <= budget < verify_peak_bytes(peak, cap + GIB + 100 * MIB)
    contract = _stream(OpenAIChatProtocol())
    limited = bounds_for(contract, trace_bytes=GIB, held_reply_bytes=GIB, trace_request_bytes=100 * MIB,
                         verify_peak=peak)
    assert (limited.trace_cap_bytes, limited.verify_limited) == (cap, True) and cap < 15 * GIB
    loose = VerifyPeak(base_bytes=GIB, bytes_per_byte=1.0)
    ruled = bounds_for(contract, trace_bytes=GIB, held_reply_bytes=GIB, trace_request_bytes=100 * MIB,
                       verify_peak=loose)
    assert (ruled.trace_cap_bytes, ruled.verify_limited) == (15 * GIB, False)
    assert ruled.verify_peak == loose and ruled.trace_request_bytes == 100 * MIB


def test_a_task_whose_least_cap_verify_cannot_hold_is_refused():
    with pytest.raises(ValueError, match="verify"):
        verify_cap(VerifyPeak(base_bytes=VERIFY_MEMORY_BYTES, bytes_per_byte=1.0), trace_request_bytes=MIB)


def test_bounds_whose_cap_verify_cannot_hold_are_refused():
    """However they are made, so no task declares a cap its verification cannot hold."""
    peak = VerifyPeak(base_bytes=50 * GIB, bytes_per_byte=20.0)
    cap = verify_cap(peak, trace_request_bytes=MIB)
    DriveBounds(trace_cap_bytes=cap, trace_request_bytes=MIB, verify_peak=peak, verify_limited=True)
    with pytest.raises(ValueError, match="verify"):
        DriveBounds(trace_cap_bytes=cap + GIB, trace_request_bytes=MIB, verify_peak=peak, verify_limited=True)


def test_a_verify_role_with_more_memory_holds_a_larger_cap():
    """A task whose reference runs on a whole node declares the memory its verify role has; the cap
    and the refusal follow that memory, and a task that declares none keeps the reference job's,
    left out of the dump."""
    peak = VerifyPeak(base_bytes=300 * GIB, bytes_per_byte=26.3)
    with pytest.raises(ValueError, match="verify"):
        verify_cap(peak, trace_request_bytes=MIB)
    node = 1000 * GIB
    cap = verify_cap(peak, trace_request_bytes=MIB, verify_memory_bytes=node)
    assert verify_peak_bytes(peak, cap + MIB) <= VERIFY_SHARE * node < verify_peak_bytes(peak, cap + GIB + MIB)
    bounds = bounds_for(_stream(OpenAIChatProtocol()), trace_bytes=GIB, held_reply_bytes=0, trace_request_bytes=MIB,
                        verify_peak=peak, verify_memory_bytes=node)
    assert bounds.verify_memory_bytes == node and bounds.model_dump()["verify_memory_bytes"] == node
    with pytest.raises(ValueError, match="verify"):
        DriveBounds(trace_cap_bytes=cap + GIB, trace_request_bytes=MIB, verify_peak=peak, verify_limited=True,
                    verify_memory_bytes=node)
    default = DriveBounds(trace_cap_bytes=GIB, trace_request_bytes=MIB,
                          verify_peak=VerifyPeak(base_bytes=GIB, bytes_per_byte=1.0), verify_limited=False)
    assert default.verify_memory_bytes == VERIFY_MEMORY_BYTES and "verify_memory_bytes" not in default.model_dump()


@pytest.mark.parametrize("module", sorted(PEAKS))
def test_each_task_declares_its_measured_verify_peak(module):
    """The figures each task module cites from the verify peak measurement, and what they make of
    its cap: its verify role's peak at the cap stays within the share of its memory."""
    bounds = drive_bounds(load_task(TaskRef(module=module)))
    base_gib, per_byte, limited = PEAKS[module]
    assert (bounds.verify_peak.base_bytes, bounds.verify_peak.bytes_per_byte, bounds.verify_limited) == (
        round(base_gib * GIB), per_byte, limited)
    events = trace_event_bytes(bounds.trace_cap_bytes, bounds.trace_request_bytes)
    assert verify_peak_bytes(bounds.verify_peak, events) <= VERIFY_SHARE * bounds.verify_memory_bytes


def test_only_a_task_whose_reference_holds_a_node_declares_its_verify_memory():
    """Every task but agentic_coding verifies in the one-GPU reference job's memory; agentic_coding's
    reference shards over its node's eight GPUs, so its verify role has its evaluation's 1,911 GiB
    job less the runner's reserve."""
    for module in sorted(PEAKS):
        memory = drive_bounds(load_task(TaskRef(module=module))).verify_memory_bytes
        assert memory == ((1911 - 4) * GIB if module == "tasks.agentic_coding.task" else VERIFY_MEMORY_BYTES), module


def test_the_verify_role_is_sized_from_the_tasks_verify_peak():
    """The verify role gets the reference job's memory less the runner's reserve, or its measured
    peak over a trace at the task's cap when that is more (``reference_shape``)."""
    from hls.pipeline.pipelines import reference_shape

    task = load_task(TaskRef(module="tasks.multitenant_slo.task"))
    bounds = drive_bounds(task)
    peak = math.ceil(verify_peak_bytes(bounds.verify_peak, trace_event_bytes(bounds.trace_cap_bytes,
                                                                           bounds.trace_request_bytes)) / GIB)
    assert peak == 129
    assert reference_shape(task, runner_memory_gib=4).memory_gib == 176
    assert reference_shape(task, runner_memory_gib=60).memory_gib == peak


# --- what a request in flight takes in the trace --------------------------------------------------

#: The longest ``repr`` an int64 has, as a turn index and a token cap at their longest.
LONGEST_INT = -2**63


def _stamped(*contents, history: list[dict] | None = None, tools: list | None = None) -> Envelope:
    """A turn carrying *contents* as the drive loop sends it (``stamp_request``), after *history*,
    offering *tools* when given."""
    turn = Envelope(request_id="s.1", contents=list(contents))
    env = stamp_request(turn, [*(history or []), user_message(turn)], tenant_id="t0", session_id="s", turn_idx=1,
                        max_tokens=256)
    if tools is None:
        return env
    return env.model_copy(update={"metadata": {**env.metadata, "tools": tools}})


def _events(env: Envelope) -> list[TraceEvent]:
    """An answered and a failed event of request *env*, every value of its reply and of its task's
    dataset empty and every number the harness writes at its longest."""
    answered = TraceEvent(request_id="", session_id="", turn_index=LONGEST_INT, input=env,
                          response=ArtifactResponse(output=Envelope(request_id=""), t_first_packet=LONGEST_FLOAT,
                                                    t_complete=LONGEST_FLOAT),
                          t_send=LONGEST_FLOAT, t_recv=LONGEST_FLOAT, t_planned_send=LONGEST_FLOAT)
    attempt = RequestAttempt(n_frames=LONGEST_INT, t_first_frame=LONGEST_FLOAT, status_code=LONGEST_INT,
                             partial=PartialOutput(token_ids=[], text="", finish_reason=None))
    reason = max(FailureReason, key=lambda r: len(r.value))
    failed = TraceEvent(request_id="", session_id="", turn_index=LONGEST_INT, input=env, error="",
                        failure=RequestFailure(reason=reason, t_abandoned=LONGEST_FLOAT, observed=attempt),
                        t_send=LONGEST_FLOAT, t_planned_send=LONGEST_FLOAT)
    return [answered, failed]


#: Base64, as an omni request carries its media.
_B64 = "QUJD+/9=" * 64
#: A request of each shape the graded drives send, with every kind of value the trace writes
#: differently from the wire: a DEL, a quote, a backslash, a control character and characters
#: past ASCII, which it escapes, and a NaN and a set, which only its own encoder writes.
REQUESTS = {
    "a chat turn": _stamped(TextPayload(text='q \x7f " \\ \n \u00e9 \U0001d11e')),
    "a conversation that offers tools": _stamped(
        TextPayload(text="q"), history=[{"role": "user", "content": "q0"}, {"role": "assistant", "content": "\x7f" * 9}],
        tools=[{"type": "function", "function": {"name": "f\x7f", "parameters": {}}}]),
    "an omni image and its question": _stamped(ImagePayload(image_b64=_B64, width=640, height=480),
                                               TextPayload(text="What is on the table?")),
    "an omni clip to translate": _stamped(AudioPayload(audio_b64=_B64, sample_rate=16_000, duration_sec=3.25),
                                          TextPayload(text="Translate into English.")),
    "a clip to transcribe": _stamped(AudioPayload(audio_b64=_B64, sample_rate=16_000)),
    "metadata only its encoder writes": Envelope(request_id="r", metadata={"x": float("nan"), "s": {1, 2}}),
}


@pytest.mark.parametrize("name", sorted(REQUESTS))
def test_a_request_is_held_at_the_json_its_event_takes(name):
    """A request's event writes its envelope as its input exactly as the envelope is written alone
    (``input_json_bytes``), so the event of a request whose reply and task values are empty takes
    the request's charge (``request_trace_bytes``) at most, and exactly at the longer of an answered
    and a failed event."""
    env, empty = REQUESTS[name], Envelope(request_id="")
    events = _events(env)
    for ev in events:
        bare = ev.model_copy(update={"input": empty})
        assert len(_event_line(ev)) - len(_event_line(bare)) == input_json_bytes(env) - input_json_bytes(empty)
    assert max(len(_event_line(ev)) for ev in events) == request_trace_bytes(env)


def test_the_request_constant_is_the_longest_event_beside_its_input():
    """``REQUEST_EVENT_BYTES`` is the longest line an event takes beyond its input with every value
    of its reply and its task empty: the event's own fields, answered or failed, every number at
    its longest."""
    empty = Envelope(request_id="")
    assert REQUEST_EVENT_BYTES == max(len(_event_line(ev)) for ev in _events(empty)) - input_json_bytes(empty)
    assert input_json_bytes(empty) == len(json.dumps(empty.model_dump(mode="python"), separators=(",", ":")))
