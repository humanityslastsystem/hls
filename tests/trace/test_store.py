from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

from hls.core.payloads import AudioPayload, CodecPosition, Envelope, TextPayload
from hls.core.trace import ArtifactResponse, FailureReason, Trace, TraceEvent
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit, SloLatencyScorer, TrajectorySessionUnit
from hls.store.manifest import ReportedOutput
from hls.trace import store
from hls.trace.store import (END_FILE, SUMMARY_FILE, SummaryEvent, TraceEnd, TraceIncomplete, check_trace,
                             iter_events, read_end, read_summary, read_trace, write_trace)
from tests.trace.traces import HARDWARE, LATENCY_TENANTS, TENANTS, drive_trace, passing_verdict

DRIVER = {"class": "Driver", "drive_timeout_sec": "3600"}


def _write(tmp_path, trace, **kw):
    out = tmp_path / "trace"
    end = write_trace(trace, out, ending="drained", driver=DRIVER, **kw).end
    return out, end


def _nan_equal(a, b) -> bool:
    """Equality that counts NaN equal to NaN, for nested plain data."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_nan_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_nan_equal(x, y) for x, y in zip(a, b))
    return a == b


def _speech_event() -> TraceEvent:
    """A speech reply as the TTS wire protocol records it: codec codes, a packed report with
    non-finite values, and the completion time its trailing codes frame sets."""
    report = [CodecPosition(token_id=p, logprob=float("nan") if p == 0 else -0.5, ids=[p, 9],
                            lps=[-0.5, -math.inf]) for p in range(5)]
    audio = AudioPayload(audio_b64="UklGRg==", sample_rate=24_000, codec_codes=[[0, 1], [2, 3]],
                         codec_logprobs=report)
    return TraceEvent(request_id="v0.0", session_id="v0", input=Envelope(request_id="v0.0",
                                                                         metadata={"tenant_id": "voice"}),
                      response=ArtifactResponse(output=Envelope(request_id="v0.0", contents=[audio]),
                                                t_first_packet=10.2, packet_timestamps=[10.2, 10.25, 10.3],
                                                t_complete=10.3),
                      t_send=10.0, t_recv=10.3)


def test_every_event_reads_back_field_for_field_in_emission_order(tmp_path):
    trace = drive_trace()
    out, end = _write(tmp_path, trace)
    back = read_trace(out)
    assert [ev.request_id for ev in back.events] == [ev.request_id for ev in trace.events]
    for a, b in zip(trace.events, back.events):
        # Byte for byte the event's own JSON.
        assert b.model_dump_json() == a.model_dump_json()
        # And every value, NaN and infinity included, as the driver held it, but for the
        # two conversions JSON forces: tuples become lists, and a non-string key inside an
        # opaque value becomes a string.
        assert _nan_equal(b.model_dump(mode="python"), json.loads(json.dumps(a.model_dump(mode="python"),
                                                                                allow_nan=True)))
    assert end.events == len(trace.events)


def test_non_finite_values_survive_where_trace_json_wrote_null(tmp_path):
    trace = drive_trace()
    out, _ = _write(tmp_path, trace)
    first = read_trace(out).events[0]
    reply = first.response.output.contents[0]
    assert math.isnan(reply.logprobs[1].logprob)
    assert reply.logprobs[0].top_logprobs[1].logprob == -math.inf
    assert first.extras["gap"] == math.inf
    # The event's own JSON loses them: a NaN logprob comes back as null and fails validation.
    assert '"logprob":null' in trace.events[0].model_dump_json()


def test_tuples_come_back_as_lists_and_keys_as_strings(tmp_path):
    out, _ = _write(tmp_path, drive_trace())
    ev = read_trace(out).events[0]
    assert ev.ground_truth == ["gold", 0] and ev.extras["pair"] == [0, 4]
    assert ev.input.metadata["shape"] == [1, 2] and ev.input.metadata["by_id"] == {"7": "seven"}


def _with_extras(extras: dict) -> Trace:
    first = drive_trace().events[0]
    return Trace(events=[first.model_copy(update={"extras": extras})])


def test_a_set_and_bytes_come_back_as_trace_json_wrote_them(tmp_path):
    trace = _with_extras({"ids": {3, 1, 2}, "raw": b"abc"})
    out, _ = _write(tmp_path, trace)
    assert read_trace(out).events[0].extras == {"ids": [1, 2, 3], "raw": "abc"}
    assert '"extras":{"ids":[1,2,3],"raw":"abc"}' in trace.events[0].model_dump_json()


def test_a_none_key_comes_back_as_null_where_trace_json_wrote_none(tmp_path):
    trace = _with_extras({"by": {None: 1}})
    out, _ = _write(tmp_path, trace)
    assert read_trace(out).events[0].extras == {"by": {"null": 1}}
    assert '"by":{"None":1}' in trace.events[0].model_dump_json()


def test_a_tuple_key_trace_json_wrote_is_refused_and_leaves_no_trace(tmp_path):
    trace = _with_extras({"by": {(1, 2): 1}})
    assert '"by":{"1,2":1}' in trace.events[0].model_dump_json()
    with pytest.raises(TypeError, match="tuple"):
        _write(tmp_path, trace)
    assert not (tmp_path / "trace" / END_FILE).exists()


def test_a_file_is_written_under_a_dot_prefixed_temp_name_until_it_is_whole(tmp_path, monkeypatch):
    out = tmp_path / "trace"
    real = store._event_line
    seen: list[tuple[list[str], list[str]]] = []

    def watched(ev):
        names = sorted(p.name for p in out.iterdir())
        seen.append(([n for n in names if n.startswith(".")], [n for n in names if not n.startswith(".")]))
        return real(ev)

    monkeypatch.setattr(store, "_event_line", watched)
    end = write_trace(drive_trace(), out, ending="drained", driver=DRIVER, chunk_bytes=3_000).end
    assert len(end.chunks) >= 2
    # While the second chunk is written: the first is published, the second and the summary are not.
    assert seen[end.chunks[0].events] ==([f".{end.chunks[1].name}.tmp", f".{SUMMARY_FILE}.tmp"], [end.chunks[0].name])
    assert sorted(p.name for p in out.iterdir()) == sorted([*(c.name for c in end.chunks), SUMMARY_FILE, END_FILE])


def test_a_speech_reply_keeps_its_packed_codec_report_and_completion_time(tmp_path):
    ev = _speech_event()
    out, _ = _write(tmp_path, Trace(events=[ev]))
    back = read_trace(out).events[0]
    assert back.model_dump_json() == ev.model_dump_json()
    audio = back.response.output.contents[0]
    assert isinstance(audio, AudioPayload) and audio.codec_codes == [[0, 1], [2, 3]]
    assert all(isinstance(p, CodecPosition) for p in audio.codec_logprobs)
    assert math.isnan(audio.codec_logprobs[0].logprob)
    assert [tuple(a) for a in audio.codec_logprobs[1].top_logprobs] == [(1, -0.5), (9, -math.inf)]
    assert back.response.t_complete == 10.3


def test_failures_keep_their_reason_partial_output_and_abandonment(tmp_path):
    trace = drive_trace()
    out, _ = _write(tmp_path, trace)
    back = {ev.request_id: ev for ev in read_trace(out).events}
    failed = back["s2.0"]
    assert failed.response is None and failed.t_recv is None
    assert failed.failure.reason is FailureReason.DEADLINE
    assert failed.failure.t_abandoned == trace.events[6].failure.t_abandoned
    assert failed.failure.observed.partial.token_ids == [1, 2, 3]
    assert back["a1.2"].failure.reason is FailureReason.CIRCUIT_BREAKER


def test_trace_wide_aggregates_and_groupings_are_unchanged(tmp_path):
    trace = drive_trace()
    out, _ = _write(tmp_path, trace, chunk_bytes=2_000)
    back = read_trace(out)
    for name in ("duration_sec", "t_end", "latencies_sec", "ttft_sec", "tpot_sec", "scheduler_slack_sec"):
        assert getattr(back, name)() == getattr(trace, name)(), name
    assert {k: [e.request_id for e in v] for k, v in back.by_session().items()} == \
           {k: [e.request_id for e in v] for k, v in trace.by_session().items()}
    assert [ev.event_key for ev in back.events_for("chat")] == \
           [ev.event_key for ev in trace.events_for("chat")]


def test_a_long_trace_is_cut_into_chunks_read_back_in_order(tmp_path):
    trace = Trace(events=drive_trace().events * 5)
    out, end = _write(tmp_path, trace, chunk_bytes=4_000)
    assert len(end.chunks) > 3 and sum(c.events for c in end.chunks) == len(trace.events)
    assert [c.name for c in end.chunks] == [f"chunk-{i:06d}.jsonl.gz" for i in range(len(end.chunks))]
    assert [ev.request_id for ev in iter_events(out)] == [ev.request_id for ev in trace.events]


def test_writing_streams_one_event_at_a_time(tmp_path, monkeypatch):
    # The whole trace is never serialized as one object; each event is, once.
    def refuse(*a, **k):
        raise AssertionError("the whole trace was serialized")

    monkeypatch.setattr(Trace, "model_dump_json", refuse)
    monkeypatch.setattr(Trace, "model_dump", refuse)
    trace = drive_trace()
    before = [ev.model_dump_json() for ev in trace.events]
    _write(tmp_path, trace)
    assert [ev.model_dump_json() for ev in trace.events] == before   # the caller's events are untouched


def test_the_end_file_comes_last_and_names_every_chunk(tmp_path):
    out, end = _write(tmp_path, drive_trace(), chunk_bytes=3_000)
    assert TraceEnd.model_validate_json((out / END_FILE).read_bytes()) == end
    assert end.ending == "drained" and end.driver == DRIVER
    assert sorted(p.name for p in out.iterdir()) == \
           sorted([END_FILE, SUMMARY_FILE, *(c.name for c in end.chunks)])
    assert check_trace(out) == end


def test_the_same_trace_writes_the_same_bytes(tmp_path):
    trace = drive_trace()
    a = write_trace(trace, tmp_path / "a", ending="drained", driver=DRIVER).end
    b = write_trace(trace, tmp_path / "b", ending="drained", driver=DRIVER).end
    assert [c.sha256 for c in a.chunks] == [c.sha256 for c in b.chunks]
    assert a.summary_sha256 == b.summary_sha256


def test_write_trace_returns_the_digest_and_size_of_every_byte_it_wrote(tmp_path):
    """The writer's own record of each file, which a role reports without reading the trace a
    second time, is what is on disk: every chunk, the summary and END.json."""
    out = tmp_path / "trace"
    written = write_trace(Trace(events=drive_trace().events * 3), out, ending="drained", driver=DRIVER,
                          chunk_bytes=3_000)
    assert len(written.end.chunks) > 2
    assert sorted(written.files) == sorted(p.name for p in out.iterdir()) == \
           sorted([END_FILE, SUMMARY_FILE, *(c.name for c in written.end.chunks)])
    for name, record in written.files.items():
        data = (out / name).read_bytes()
        assert record == ReportedOutput(sha256=hashlib.sha256(data).hexdigest(), size=len(data)), name
    assert [written.files[c.name].sha256 for c in written.end.chunks] == [c.sha256 for c in written.end.chunks]
    assert written.files[SUMMARY_FILE].sha256 == written.end.summary_sha256
    assert TraceEnd.model_validate_json((out / END_FILE).read_bytes()) == written.end
    empty = write_trace(Trace(), tmp_path / "empty", ending="drained", driver=DRIVER)
    assert sorted(empty.files) == [END_FILE, SUMMARY_FILE]
    assert empty.files[SUMMARY_FILE].size == len((tmp_path / "empty" / SUMMARY_FILE).read_bytes())


def test_an_empty_drive_is_a_complete_trace_of_no_events(tmp_path):
    out, end = _write(tmp_path, Trace())
    assert end.events == 0 and end.chunks == []
    assert read_trace(out).events == [] and read_summary(out).events == []


def test_a_trace_without_its_end_file_is_incomplete(tmp_path):
    out, _ = _write(tmp_path, drive_trace())
    (out / END_FILE).unlink()
    for read in (read_end, check_trace, read_trace, read_summary, lambda d: list(iter_events(d))):
        with pytest.raises(TraceIncomplete, match="END.json"):
            read(out)


def test_a_writer_that_dies_mid_trace_leaves_no_end_file(tmp_path, monkeypatch):
    trace = drive_trace()
    calls = []
    real = store._event_line

    def dies(ev):
        calls.append(ev)
        if len(calls) == 5:
            raise RuntimeError("killed")
        return real(ev)

    monkeypatch.setattr(store, "_event_line", dies)
    with pytest.raises(RuntimeError, match="killed"):
        _write(tmp_path, trace, chunk_bytes=2_000)
    assert not (tmp_path / "trace" / END_FILE).exists()
    with pytest.raises(TraceIncomplete):
        read_trace(tmp_path / "trace")


def _open_files_under(root) -> list[str]:
    fds = Path("/proc/self/fd")
    targets = []
    for fd in fds.iterdir():
        try:
            targets.append(os.readlink(fd))
        except OSError:
            continue
    return [t for t in targets if t.startswith(str(root))]


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="reads the process's open files from /proc")
def test_a_writer_that_dies_closes_its_files_and_leaves_only_the_chunks_it_finished(tmp_path, monkeypatch):
    real = store._event_line
    calls = []

    def dies(ev):
        calls.append(ev)
        if len(calls) == 5:
            raise RuntimeError("killed")
        return real(ev)

    monkeypatch.setattr(store, "_event_line", dies)
    # The raised error's traceback keeps the writer's frame, and so its files, alive.
    with pytest.raises(RuntimeError, match="killed") as raised:
        _write(tmp_path, drive_trace(), chunk_bytes=2_000)
    out = tmp_path / "trace"
    assert raised.value is not None and _open_files_under(out) == []
    names = sorted(p.name for p in out.iterdir())
    assert names and all(n.startswith("chunk-") for n in names)


def test_a_tampered_or_truncated_chunk_is_refused_before_any_of_its_events(tmp_path):
    out, end = _write(tmp_path, drive_trace(), chunk_bytes=3_000)
    chunk = out / end.chunks[1].name
    data = chunk.read_bytes()
    chunk.write_bytes(data[: len(data) // 2])
    seen = []
    with pytest.raises(TraceIncomplete, match=end.chunks[1].name):
        for ev in iter_events(out):
            seen.append(ev)
    assert len(seen) == end.chunks[0].events     # nothing of the bad chunk was yielded
    with pytest.raises(TraceIncomplete, match="sha256"):
        check_trace(out)


def test_a_chunk_replaced_by_another_valid_one_yields_nothing_of_it(tmp_path):
    out, end = _write(tmp_path, drive_trace(), chunk_bytes=3_000)
    first, second = (out / c.name for c in end.chunks[:2])
    lines = gzip.decompress(first.read_bytes()).splitlines()
    swapped = (lines * end.chunks[1].events)[: end.chunks[1].events]   # a whole gzip file, right count
    second.write_bytes(gzip.compress(b"\n".join(swapped) + b"\n"))
    seen = []
    with pytest.raises(TraceIncomplete, match="sha256"):
        for ev in iter_events(out):
            seen.append(ev)
    assert len(seen) == end.chunks[0].events


def test_a_chunk_whose_count_differs_from_the_end_file_is_refused(tmp_path):
    out, end = _write(tmp_path, drive_trace(), chunk_bytes=3_000)
    # The same bytes and the same total, but one event moved between two chunks' counts.
    a, b = end.chunks[0], end.chunks[1]
    moved = [a.model_copy(update={"events": a.events + 1}), b.model_copy(update={"events": b.events - 1})]
    forged = end.model_copy(update={"chunks": [*moved, *end.chunks[2:]]})
    (out / END_FILE).write_text(forged.model_dump_json())
    with pytest.raises(TraceIncomplete, match=f"{a.name} holds {a.events} events"):
        read_trace(out)


def test_an_end_file_whose_total_differs_from_its_chunks_is_refused(tmp_path):
    out, end = _write(tmp_path, drive_trace(), chunk_bytes=3_000)
    (out / END_FILE).write_text(end.model_copy(update={"events": end.events + 1}).model_dump_json())
    with pytest.raises(TraceIncomplete, match="the chunks hold"):
        read_end(out)


def test_a_chunk_the_end_file_lists_but_the_store_lacks_is_refused(tmp_path):
    out, end = _write(tmp_path, drive_trace(), chunk_bytes=3_000)
    (out / end.chunks[-1].name).unlink()
    with pytest.raises(TraceIncomplete, match=end.chunks[-1].name):
        read_end(out)


def test_a_tampered_summary_is_refused(tmp_path):
    out, _ = _write(tmp_path, drive_trace())
    lines = gzip.decompress((out / SUMMARY_FILE).read_bytes()).splitlines()
    (out / SUMMARY_FILE).write_bytes(gzip.compress(b"\n".join(lines[:-1]) + b"\n"))
    with pytest.raises(TraceIncomplete, match="summary"):
        read_summary(out)


def test_a_summary_with_one_altered_timing_is_refused(tmp_path):
    out, _ = _write(tmp_path, drive_trace())
    lines = gzip.decompress((out / SUMMARY_FILE).read_bytes()).splitlines()
    record = json.loads(lines[0])
    record["t_send"] += 0.5
    lines[0] = json.dumps(record).encode()
    (out / SUMMARY_FILE).write_bytes(gzip.compress(b"\n".join(lines) + b"\n"))
    for read in (read_summary, check_trace):
        with pytest.raises(TraceIncomplete, match="sha256"):
            read(out)


def test_an_end_file_whose_summary_is_gone_is_refused(tmp_path):
    out, _ = _write(tmp_path, drive_trace())
    (out / SUMMARY_FILE).unlink()
    with pytest.raises(TraceIncomplete, match=SUMMARY_FILE):
        read_end(out)


def test_a_second_writer_into_the_same_directory_is_refused(tmp_path):
    out, _ = _write(tmp_path, drive_trace())
    with pytest.raises(FileExistsError):
        write_trace(drive_trace(), out, ending="drained", driver=DRIVER)


@pytest.mark.parametrize("ending", ["finished", "capped"])
def test_an_ending_the_store_does_not_know_is_refused_before_anything_is_written(tmp_path, ending):
    """A drive has no request cap, so ``capped`` is no ending either."""
    with pytest.raises(ValueError, match=ending):
        write_trace(drive_trace(), tmp_path / "t", ending=ending, driver=DRIVER)
    assert not (tmp_path / "t").exists()


@pytest.mark.parametrize("driver", [{"class": "Driver", "drive_timeout_sec": 3600.0}, {1: "Driver"}],
                         ids=["value", "key"])
def test_a_driver_record_the_end_file_cannot_hold_is_refused_before_anything_is_written(tmp_path, driver):
    with pytest.raises(ValueError, match="driver"):
        write_trace(drive_trace(), tmp_path / "t", ending="drained", driver=driver)
    assert not (tmp_path / "t").exists()


def test_the_summary_keeps_identity_timing_and_failure_and_no_content(tmp_path):
    trace = drive_trace()
    out, _ = _write(tmp_path, trace)
    summary = read_summary(out)
    for a, b in zip(trace.events, summary.events):
        assert (b.request_id, b.session_id, b.turn_index, b.tenant_id, b.event_key) == \
               (a.request_id, a.session_id, a.turn_index, a.tenant_id, a.event_key)
        assert (b.t_send, b.t_recv, b.t_planned_send, b.error) == \
               (a.t_send, a.t_recv, a.t_planned_send, a.error)
        assert (b.response is None) == (a.response is None)
        if a.response is not None:
            assert b.response.t_first_packet == a.response.t_first_packet
            ts, held = a.response.packet_timestamps, b.response.packet_timestamps
            assert (held is None) == (ts is None)
            if ts:
                assert (len(held), held[0], held[-1]) == (len(ts), ts[0], ts[-1])
            assert b.response.t_complete == a.response.t_complete
            assert b.response.output.contents == []
        if a.failure is not None:
            assert (b.failure.reason, b.failure.t_abandoned) == (a.failure.reason, a.failure.t_abandoned)
        assert b.input.contents == [] and b.extras == {} and b.ground_truth is None


def test_the_summary_keeps_the_completion_time_a_trailing_frame_sets(tmp_path):
    ev = _speech_event()
    out, _ = _write(tmp_path, Trace(events=[ev]))
    back = read_summary(out).events[0]
    assert back.response.t_complete == 10.3
    held = back.response.packet_timestamps
    assert (len(held), held[0], held[-1]) == (3, 10.2, 10.3) and back.response.output.contents == []


def test_the_summary_keeps_non_finite_timings(tmp_path):
    ev = drive_trace().events[0]
    response = ArtifactResponse(output=ev.response.output, t_first_packet=ev.t_send,
                                packet_timestamps=[ev.t_send, math.nan])
    odd = ev.model_copy(update={"t_planned_send": math.inf, "response": response})
    out, _ = _write(tmp_path, Trace(events=[odd]))
    back = read_summary(out).events[0]
    assert back.t_planned_send == math.inf and math.isnan(back.response.packet_timestamps[1])


def test_an_event_without_a_tenant_stays_without_one_in_the_summary(tmp_path):
    trace = drive_trace()
    ev = trace.events[0]
    stray = ev.model_copy(update={"input": ev.input.model_copy(update={"metadata": {}})})
    out, _ = _write(tmp_path, Trace(events=[stray, *trace.events[1:]]))
    assert read_summary(out).events[0].tenant_id is None
    assert SummaryEvent.of_event(stray).tenant_id is None


def _score(scorer, tenants, trace, verdict):
    return asyncio.run(scorer.score(tenants, trace, verdict, HARDWARE)).model_dump(mode="json")


@pytest.mark.parametrize("failures", [False, True])
@pytest.mark.parametrize("scorer, tenants", [
    (GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=2.0)), TENANTS),
    (GoodputScorer(unit=TrajectorySessionUnit(), sol=ConstantSoL(rps=1.0)), TENANTS),
    (SloLatencyScorer(baseline_log_credit=0.25), LATENCY_TENANTS),
    (SloLatencyScorer(), LATENCY_TENANTS),
])
def test_every_scorer_scores_the_summary_exactly_as_the_full_trace(tmp_path, scorer, tenants, failures):
    trace = drive_trace(failures=failures, tenants=tenants)
    out, _ = _write(tmp_path, trace)
    verdict = passing_verdict(trace)
    assert _score(scorer, tenants, read_summary(out), verdict) == _score(scorer, tenants, trace, verdict)
    # The comparison covers runs that pass every gate as well as gated zeros (a goodput
    # scorer zeroes only the failing tenant's credit, so its run stays ungated).
    full = _score(scorer, tenants, trace, verdict)
    if isinstance(scorer, SloLatencyScorer):
        assert full["gate_passed"] is (not failures)
    else:
        assert full["gate_passed"] is True and full["value"] > 0


def test_events_no_tenant_prices_gate_the_same_from_the_summary(tmp_path):
    trace = drive_trace(failures=False)          # the plain tenant has no latency ratio
    out, _ = _write(tmp_path, trace)
    scorer = SloLatencyScorer(baseline_log_credit=0.25)
    full = _score(scorer, LATENCY_TENANTS, trace, passing_verdict(trace))
    assert full["zero_reasons"] == ["slo"]
    assert _score(scorer, LATENCY_TENANTS, read_summary(out), passing_verdict(trace)) == full


def test_a_failed_verdict_scores_the_same_from_the_summary(tmp_path):
    trace = drive_trace(failures=False)
    out, _ = _write(tmp_path, trace)
    verdict = passing_verdict(trace).model_copy(update={"passed": False, "failure_mode": "correctness"})
    for scorer, tenants in ((GoodputScorer(unit=RequestUnit(), sol=ConstantSoL()), TENANTS),
                            (SloLatencyScorer(baseline_log_credit=0.1), LATENCY_TENANTS)):
        assert _score(scorer, tenants, read_summary(out), verdict) == _score(scorer, tenants, trace, verdict)


def test_the_empty_trace_scores_the_same_from_the_summary(tmp_path):
    out, _ = _write(tmp_path, Trace())
    for scorer, tenants in ((GoodputScorer(unit=RequestUnit(), sol=ConstantSoL()), TENANTS),
                            (SloLatencyScorer(baseline_log_credit=0.1), LATENCY_TENANTS)):
        verdict = passing_verdict(Trace())
        assert _score(scorer, tenants, read_summary(out), verdict) == \
               _score(scorer, tenants, Trace(), verdict)


# The streaming writer, its room on the node's disk, the publish to the store, and the readers
# that hold one line at a time.


def _files(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def _event_bytes(trace: Trace) -> int:
    """The bytes of JSON *trace*'s events take in its chunks."""
    return sum(len(store._event_line(ev)) for ev in trace.events)


def _room(local: Path, trace: Trace, *, chunk_bytes: int = store.CHUNK_TARGET_BYTES) -> Path:
    """*local*, made with the room a trace of *trace*'s events takes in chunks of *chunk_bytes*."""
    local.mkdir()
    store.make_room(local, event_bytes=_event_bytes(trace), chunk_bytes=chunk_bytes)
    return local


def _write_in_room(local: Path, trace: Trace, *, ending="drained", **kw) -> store.TraceWritten:
    writer = store.TraceWriter.in_room(local, driver=DRIVER, **kw)
    for ev in trace.events:
        writer.write(ev)
    return writer.finish(ending=ending)


#: While it holds a directory, this process may take no new space under it: no file, directory,
#: link or name made, and no file emptied or appended to. Every attempt is kept.
_NO_NEW_SPACE: list[Path] = []
_NEW_SPACE_TAKEN: list[str] = []


def _refuse_new_space(event: str, args: tuple) -> None:
    if not _NO_NEW_SPACE:
        return
    if event == "open" and len(args) == 3 and isinstance(args[2], int) and not isinstance(args[0], int):
        path, flags = args[0], args[2]
        new = flags & (os.O_TRUNC | os.O_APPEND) or (flags & os.O_CREAT and not os.path.lexists(path))
        targets = [path] if flags & (os.O_WRONLY | os.O_RDWR) and new else []
    elif event in ("os.mkdir", "os.symlink", "os.link", "os.rename", "os.replace"):
        targets = list(args[:2])
    else:
        return
    for target in targets:
        if isinstance(target, (str, bytes, os.PathLike)):
            where = Path(os.path.abspath(os.fsdecode(target))).resolve()
            if where.is_relative_to(_NO_NEW_SPACE[0]):
                _NEW_SPACE_TAKEN.append(f"{event} {where}")
                raise OSError(28, "No space left on device", str(where))


sys.addaudithook(_refuse_new_space)


def test_a_room_takes_every_block_the_trace_may_write_before_anything_is_written(tmp_path):
    """The drive role writes its trace into a room the stage makes on the node's disk before the
    server starts, at the most the trace's chunks and summary can take in gzip, every block given
    to it then: a server that fills the node's disk afterwards takes none of it."""
    event_bytes = 5 * 2**20 + 7
    store.make_room(tmp_path, event_bytes=event_bytes, chunk_bytes=2**20)
    sizes = store.room_bytes(event_bytes, chunk_bytes=2**20)
    assert sizes == {store.CHUNKS_ROOM: store.gzip_disk_bytes(event_bytes, files=6),
                     store.SUMMARY_ROOM: store.gzip_disk_bytes(event_bytes, files=1)}
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(sizes)
    for name, size in sizes.items():
        st = (tmp_path / name).stat()
        assert st.st_size == size and st.st_blocks * 512 >= size, name


def test_a_room_already_made_is_not_made_again(tmp_path):
    store.make_room(tmp_path, event_bytes=1_000)
    with pytest.raises(FileExistsError):
        store.make_room(tmp_path, event_bytes=1_000)


@pytest.mark.parametrize("chunk_bytes", [1, 3_000, store.CHUNK_TARGET_BYTES])
def test_a_trace_written_into_its_room_publishes_as_the_one_write_trace_writes(tmp_path, monkeypatch, chunk_bytes):
    """The drive role writes each event as it completes; the trace it publishes is byte for byte
    the one the whole trace written after the drive makes, END.json's write time aside."""
    monkeypatch.setattr(store.time, "time", lambda: 1234.5)
    trace = Trace(events=[*drive_trace().events, _speech_event()] * 3)
    whole = write_trace(trace, tmp_path / "whole", ending="aborted", driver=DRIVER, chunk_bytes=chunk_bytes)
    local = _room(tmp_path / "local", trace, chunk_bytes=chunk_bytes)
    streamed = _write_in_room(local, trace, ending="aborted", chunk_bytes=chunk_bytes)
    assert streamed == whole
    assert store.publish_trace(local, tmp_path / "store", streamed) == streamed
    assert _files(tmp_path / "store") == _files(tmp_path / "whole")


def test_writing_a_trace_into_its_room_takes_no_new_space_on_the_node(tmp_path):
    """Once the room is made, the writer makes no file and grows none: every byte of the trace
    goes into blocks the room already holds."""
    trace = Trace(events=[*drive_trace().events, _speech_event()] * 40)
    local = _room(tmp_path / "local", trace, chunk_bytes=3_000)
    before = {p.name: (p.stat().st_ino, p.stat().st_size) for p in local.iterdir()}
    _NEW_SPACE_TAKEN.clear()
    _NO_NEW_SPACE.append(local.resolve())
    try:
        written = _write_in_room(local, trace, chunk_bytes=3_000)
    finally:
        _NO_NEW_SPACE.clear()
    assert _NEW_SPACE_TAKEN == []
    assert {p.name: (p.stat().st_ino, p.stat().st_size) for p in local.iterdir()} == before
    assert len(written.end.chunks) > 1
    store.publish_trace(local, tmp_path / "store", written)
    assert check_trace(tmp_path / "store") == written.end


def test_a_trace_past_its_room_grows_it_and_is_still_the_trace(tmp_path, monkeypatch):
    """A room is made for the trace's bounds; a trace past them grows the room's files, which
    takes new space, and is still the trace written."""
    monkeypatch.setattr(store.time, "time", lambda: 1.0)
    trace = Trace(events=drive_trace().events * 3)
    write_trace(trace, tmp_path / "whole", ending="drained", driver=DRIVER, chunk_bytes=3_000)
    local = tmp_path / "local"
    local.mkdir()
    store.make_room(local, event_bytes=10, chunk_bytes=3_000)
    written = _write_in_room(local, trace, chunk_bytes=3_000)
    store.publish_trace(local, tmp_path / "store", written)
    assert _files(tmp_path / "store") == _files(tmp_path / "whole")


def test_a_writer_whose_room_was_never_made_fails_and_makes_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        store.TraceWriter.in_room(tmp_path, driver=DRIVER)
    assert list(tmp_path.iterdir()) == []


def test_the_writer_counts_every_plain_byte_and_event_it_wrote(tmp_path):
    trace = Trace(events=drive_trace().events * 4)
    writer = store.TraceWriter(tmp_path / "t", driver=DRIVER, chunk_bytes=3_000)
    for i, ev in enumerate(trace.events, 1):
        writer.write(ev)
        assert writer.events == i
    end = writer.finish(ending="drained").end
    chunks = sum(len(gzip.decompress((tmp_path / "t" / c.name).read_bytes())) for c in end.chunks)
    summary = len(gzip.decompress((tmp_path / "t" / SUMMARY_FILE).read_bytes()))
    assert (writer.event_bytes, writer.summary_bytes) == (chunks, summary)
    assert chunks == _event_bytes(trace)
    assert summary <= chunks and writer.events == end.events == len(trace.events)


def test_a_writer_refuses_a_driver_record_and_an_ending_end_json_cannot_hold(tmp_path):
    with pytest.raises(ValueError, match="driver"):
        store.TraceWriter(tmp_path / "a", driver={"class": 1})
    assert not (tmp_path / "a").exists()
    room = _room(tmp_path / "room", drive_trace())
    with pytest.raises(ValueError, match="driver"):
        store.TraceWriter.in_room(room, driver={"class": 1})
    writer = store.TraceWriter(tmp_path / "b", driver=DRIVER)
    writer.write(drive_trace().events[0])
    with pytest.raises(ValueError, match="capped"):
        writer.finish(ending="capped")
    assert not (tmp_path / "b" / END_FILE).exists()


class _FullDisk:
    """``os.write`` that fails with ENOSPC from its *at*-th call on, *times* times in a row."""

    def __init__(self, at: int, times: int) -> None:
        self.at, self.times, self.calls, self.failed = at, times, 0, 0
        self.real = os.write

    def __call__(self, fd, data):
        self.calls += 1
        if self.calls >= self.at and self.failed < self.times:
            self.failed += 1
            raise OSError(28, "No space left on device")
        return self.real(fd, data)


def test_a_full_disk_is_waited_out_when_the_writer_is_told_to(tmp_path, monkeypatch):
    """A trace past its room takes new space on the node's disk, which a contestant's server
    may fill for a moment, until its own disk cap ends it and its files are removed: the
    drive's writer waits that out, and the trace it leaves is the one it would have written."""
    monkeypatch.setattr(store.time, "time", lambda: 1.0)
    trace = Trace(events=drive_trace().events * 2)
    write_trace(trace, tmp_path / "ref", ending="drained", driver=DRIVER, chunk_bytes=3_000)
    local = tmp_path / "local"
    local.mkdir()
    store.make_room(local, event_bytes=10, chunk_bytes=3_000)
    full = _FullDisk(at=3, times=4)
    monkeypatch.setattr(store.os, "write", full)
    waits: list[OSError] = []
    written = _write_in_room(local, trace, chunk_bytes=3_000, wait_for_space=waits.append)
    monkeypatch.undo()
    monkeypatch.setattr(store.time, "time", lambda: 1.0)
    assert len(waits) == 4 and all(e.errno == 28 for e in waits)
    store.publish_trace(local, tmp_path / "store", written)
    assert _files(tmp_path / "store") == _files(tmp_path / "ref")


def test_a_full_disk_fails_a_writer_that_does_not_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(store.os, "write", _FullDisk(at=1, times=1))
    with pytest.raises(OSError, match="No space"):
        write_trace(drive_trace(), tmp_path / "t", ending="drained", driver=DRIVER)
    assert not (tmp_path / "t" / END_FILE).exists()


def test_a_partial_write_is_finished_and_counted_once(tmp_path, monkeypatch):
    real = os.write
    monkeypatch.setattr(store.os, "write", lambda fd, data: real(fd, bytes(data)[: max(1, len(data) // 3)]))
    written = write_trace(drive_trace(), tmp_path / "t", ending="drained", driver=DRIVER)
    monkeypatch.undo()
    for name, record in written.files.items():
        data = (tmp_path / "t" / name).read_bytes()
        assert record == ReportedOutput(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    assert check_trace(tmp_path / "t") == written.end


def test_publishing_copies_every_file_with_end_json_last(tmp_path, monkeypatch):
    trace = Trace(events=drive_trace().events * 3)
    local = _room(tmp_path / "local", trace, chunk_bytes=3_000)
    written = _write_in_room(local, trace, ending="wall", chunk_bytes=3_000)
    order: list[str] = []
    real = os.replace

    def replace(a, b):
        order.append(Path(b).name)
        real(a, b)

    monkeypatch.setattr(store.os, "replace", replace)
    assert store.publish_trace(local, tmp_path / "store", written) == written
    assert order[-1] == END_FILE and order == [*(c.name for c in written.end.chunks), SUMMARY_FILE, END_FILE]
    assert sorted(p.name for p in (tmp_path / "store").iterdir()) == sorted(written.files)
    assert check_trace(tmp_path / "store") == written.end


def _spoil(room: Path, at: int) -> None:
    """Flip the byte at offset *at* of the room file *room*, in place."""
    with open(room, "r+b") as f:
        f.seek(at)
        byte = f.read(1)
        f.seek(at)
        f.write(bytes([byte[0] ^ 0xFF]))


def test_a_stretch_of_the_room_unlike_its_record_is_not_published(tmp_path):
    trace = Trace(events=drive_trace().events * 3)
    local = _room(tmp_path / "local", trace, chunk_bytes=3_000)
    written = _write_in_room(local, trace, chunk_bytes=3_000)
    first, second = (written.files[c.name].size for c in written.end.chunks[:2])
    _spoil(local / store.CHUNKS_ROOM, first + second // 2)
    with pytest.raises(TraceIncomplete, match=written.end.chunks[1].name):
        store.publish_trace(local, tmp_path / "store", written)
    assert not (tmp_path / "store" / END_FILE).exists()
    assert not [p for p in (tmp_path / "store").iterdir() if p.name.startswith(".")]


def test_a_summary_unlike_its_record_is_not_published(tmp_path):
    trace = Trace(events=drive_trace().events * 3)
    local = _room(tmp_path / "local", trace, chunk_bytes=3_000)
    written = _write_in_room(local, trace, chunk_bytes=3_000)
    _spoil(local / store.SUMMARY_ROOM, written.files[SUMMARY_FILE].size // 2)
    with pytest.raises(TraceIncomplete, match=SUMMARY_FILE):
        store.publish_trace(local, tmp_path / "store", written)
    assert not (tmp_path / "store" / END_FILE).exists()


def test_a_room_cut_short_is_not_published(tmp_path):
    trace = Trace(events=drive_trace().events * 3)
    local = _room(tmp_path / "local", trace, chunk_bytes=3_000)
    written = _write_in_room(local, trace, chunk_bytes=3_000)
    os.truncate(local / store.CHUNKS_ROOM, written.files[written.end.chunks[0].name].size + 5)
    with pytest.raises(TraceIncomplete, match=written.end.chunks[1].name):
        store.publish_trace(local, tmp_path / "store", written)
    assert not (tmp_path / "store" / END_FILE).exists()


def test_an_end_record_unlike_the_one_written_is_not_published(tmp_path):
    trace = Trace(events=drive_trace().events * 3)
    local = _room(tmp_path / "local", trace, chunk_bytes=3_000)
    written = _write_in_room(local, trace, chunk_bytes=3_000)
    other = written.model_copy(update={"end": written.end.model_copy(update={"ending": "wall"})})
    with pytest.raises(TraceIncomplete, match=END_FILE):
        store.publish_trace(local, tmp_path / "store", other)
    assert not (tmp_path / "store" / END_FILE).exists()


def test_publishing_into_an_existing_directory_is_refused(tmp_path):
    local = _room(tmp_path / "local", drive_trace())
    written = _write_in_room(local, drive_trace())
    (tmp_path / "store").mkdir()
    with pytest.raises(FileExistsError):
        store.publish_trace(local, tmp_path / "store", written)


def test_no_reader_holds_a_whole_file(tmp_path, monkeypatch):
    """A trace may reach its cap, tens of GiB: every reader takes a file a piece at a time and
    holds at most one line of it."""
    out, end = _write(tmp_path, Trace(events=drive_trace().events * 3), chunk_bytes=3_000)
    real_read_bytes = Path.read_bytes

    def no_whole_file(path):
        if path.name.endswith(".gz"):
            raise AssertionError(f"{path.name} was read whole")
        return real_read_bytes(path)

    def no_whole_decompress(*a, **k):
        raise AssertionError("a file was decompressed whole")

    monkeypatch.setattr(Path, "read_bytes", no_whole_file)
    monkeypatch.setattr(store.gzip, "decompress", no_whole_decompress)
    assert check_trace(out) == end
    assert len(list(iter_events(out))) == end.events == len(read_summary(out).events)


def test_a_gzip_file_with_bytes_after_its_member_is_refused(tmp_path):
    out, end = _write(tmp_path, drive_trace())
    summary = out / SUMMARY_FILE
    data = summary.read_bytes() + gzip.compress(b"")
    summary.write_bytes(data)
    forged = end.model_copy(update={"summary_sha256": hashlib.sha256(data).hexdigest()})
    (out / END_FILE).write_text(forged.model_dump_json())
    with pytest.raises(TraceIncomplete, match="whole gzip"):
        read_summary(out)


def test_the_summary_keeps_each_replys_first_and_last_packet_and_their_count(tmp_path):
    """What every scorer reads of a reply's packets: the score role holds no more of them."""
    trace = drive_trace()
    out, _ = _write(tmp_path, trace)
    seen_long = False
    for full, held in zip(trace.events, read_summary(out).events):
        ts = None if full.response is None else full.response.packet_timestamps
        got = None if held.response is None else held.response.packet_timestamps
        if not ts:
            assert not got
            continue
        n = len(ts)
        assert (len(got), got[0], got[-1], got[n - 1], got[-n]) == (n, ts[0], ts[-1], ts[-1], ts[0])
        if n > 2:
            seen_long = True
            with pytest.raises(store.PacketTimeNotHeld):
                got[1]
        with pytest.raises(store.PacketTimeNotHeld):
            list(got)
        with pytest.raises(IndexError):
            got[n]
    assert seen_long


# The stored trace: a view that reads its events from the store on every pass and never holds
# them, for readers that take a trace's events more than once (the verify role).


def _stored(tmp_path: Path, trace: Trace, **kw) -> store.StoredTrace:
    write_trace(trace, tmp_path / "stored", ending="drained", driver=DRIVER, **kw)
    return store.StoredTrace(tmp_path / "stored")


def _lines(events) -> list[bytes]:
    return [store._event_line(ev) for ev in events]


def test_a_stored_trace_gives_every_event_on_every_pass(tmp_path):
    trace = Trace(events=[*drive_trace().events, _speech_event()] * 3)
    stored = _stored(tmp_path, trace, chunk_bytes=3_000)
    assert _lines(stored.events) == _lines(trace.events)
    assert _lines(stored.events) == _lines(trace.events)
    assert len(stored.events) == len(trace.events) and bool(stored.events)


def test_every_pass_reads_and_checks_the_store_again(tmp_path):
    """A pass yields nothing it kept from an earlier one: a chunk spoiled between two passes is
    refused on the second, before any of its events."""
    trace = Trace(events=drive_trace().events * 3)
    stored = _stored(tmp_path, trace, chunk_bytes=3_000)
    list(stored.events)
    chunk = tmp_path / "stored" / read_end(tmp_path / "stored").chunks[1].name
    data = bytearray(chunk.read_bytes())
    data[len(data) // 2] ^= 0xFF
    chunk.write_bytes(bytes(data))
    with pytest.raises(TraceIncomplete, match=chunk.name):
        list(stored.events)


@pytest.mark.parametrize("tenant", [*(t.id for t in TENANTS), "nobody"])
def test_a_stored_trace_slices_by_tenant_and_answer_as_the_whole_trace_does(tmp_path, tenant):
    trace = Trace(events=drive_trace(tenants=TENANTS).events * 2)
    stored = _stored(tmp_path, trace, chunk_bytes=3_000)
    answered = [ev for ev in trace.events if ev.response is not None and not ev.error]
    assert answered and len(answered) < len(trace.events)
    assert _lines(stored.events_for(tenant)) == _lines(trace.events_for(tenant))
    assert len(stored.events_for(tenant)) == len(trace.events_for(tenant))
    mine = [ev for ev in answered if ev.tenant_id == tenant]
    assert _lines(stored.answered(tenant)) == _lines(mine) and len(stored.answered(tenant)) == len(mine)
    assert bool(stored.answered(tenant)) == bool(mine)
    assert _lines(stored.answered()) == _lines(answered) and len(stored.answered()) == len(answered)


def test_a_stored_trace_counts_from_its_summary_without_reading_a_chunk(tmp_path):
    trace = Trace(events=drive_trace(tenants=TENANTS).events * 2)
    stored = _stored(tmp_path, trace, chunk_bytes=3_000)
    for chunk in read_end(tmp_path / "stored").chunks:
        (tmp_path / "stored" / chunk.name).write_bytes(b"not a chunk")
    assert len(stored.events) == len(trace.events)
    assert len(stored.answered(TENANTS[0].id)) == len([ev for ev in trace.events_for(TENANTS[0].id)
                                                       if ev.response is not None and not ev.error])
    assert [ev.event_key for ev in stored.summary()] == [ev.event_key for ev in trace.events]


def test_a_pass_over_a_stored_trace_holds_one_event_at_a_time(tmp_path):
    """160 replies of a MiB each: a pass whose reader keeps nothing holds what the reader takes of
    a file at once, two pieces of its decompression, and a few replies, never the trace."""
    import tracemalloc

    def big(i: int) -> TraceEvent:
        ev = drive_trace().events[0]
        out = Envelope(request_id=ev.request_id, contents=[TextPayload(text=f"{i}" + "x" * 2**20)])
        return ev.model_copy(update={"request_id": f"big{i}", "response": ArtifactResponse(output=out)})

    trace = Trace(events=[big(i) for i in range(160)])
    stored = _stored(tmp_path, trace)
    del trace
    tracemalloc.start()
    try:
        seen = sum(1 for _ in stored.events)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert seen == 160 and peak < 2 * store._PIECE_BYTES + 24 * 2**20 < 160 * 2**20
