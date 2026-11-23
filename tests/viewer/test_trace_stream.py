"""The viewer reads a trace an event at a time and holds of each event only its row: a store trace
through ``iter_events``, a published one (one JSON document, plain, gzipped or split) decompressed
and parsed incrementally (``hls.viewer.published``). Every view of an eval is served from the one
cached index, and a row holds each long text cut at its bound and marked where it was cut, so the
index takes at most its events count times a row's bound (``row_chars``)."""

from __future__ import annotations

import gzip
import json
import tracemalloc
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar
from unittest import mock

import pytest

import hls.viewer.scanner as scanner_mod
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.viewer import published
from hls.viewer.published import TraceUnreadable, published_events
from hls.viewer.scanner import (CUT_MARK, ROW_TEXT_CHARS, ROW_TEXTS, ROW_VALUE_CHARS, ROW_VALUES, RolloutScanner,
                                row_chars)
from tests.trace.traces import drive_trace
from tests.viewer.published_fixture import ARM_CELL, PARTS, published_runs
from tests.viewer.store_fixture import ROLLOUT, TASK, sealed_rollout

T = TypeVar("T")

#: A drive, as the store fixture publishes it and the published fixture writes it.
TRACE = drive_trace()
#: The forms a published trace is read from incrementally: plain, gzipped, and gzipped and split.
STREAMED_FORMS = ("trace.json", "trace.json.gz", PARTS)


@pytest.fixture(autouse=True)
def _no_cached_index():
    """Each test reads its traces afresh, and leaves no index of them behind."""
    scanner_mod._build_trace_index.cache_clear()
    yield
    scanner_mod._build_trace_index.cache_clear()


def _peak(read: Callable[[], T]) -> tuple[T, int]:
    """What *read* returns, and the most bytes Python allocated at once while it ran."""
    tracemalloc.start()
    try:
        result = read()
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _held_bytes(read: Callable[[], T]) -> tuple[T, int]:
    """What *read* returns, and the bytes Python still holds of what it allocated once it returned."""
    tracemalloc.start()
    try:
        result = read()
        return result, tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()


def _turn(k: int, chars: int) -> TraceEvent:
    """Turn *k* of one chat session, its prompt and its reply *chars* characters each, made anew."""
    rid = f"s.{k}"
    return TraceEvent(request_id=rid, session_id="s", turn_index=k,
                      input=Envelope(request_id=rid, contents=[TextPayload(text=f"{k:06d}" + "p" * chars)],
                                     metadata={"tenant_id": "chat"}),
                      response=ArtifactResponse(output=Envelope(request_id=rid,
                                                                contents=[TextPayload(text="a" * chars)])),
                      ground_truth="A", t_send=float(k), t_recv=k + 0.5)


def _cut(text: str, chars: int) -> str:
    """*text* as a row holds it under a bound of *chars* characters."""
    if len(text) <= chars:
        return text
    head, tail = chars - chars // 2, chars // 2
    return text[:head] + CUT_MARK.format(cut=len(text) - chars, total=len(text)) + text[len(text) - tail:]


def _write_published(eval_dir: Path, events: Iterator[dict[str, Any]], form: str, *, parts: int = 3) -> None:
    """Write *events* into *eval_dir* as a published ``trace.json``, in *form*, one event
    at a time, so the writer never holds the document."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    plain = eval_dir / ("trace.json" if form == "trace.json" else "trace.json.gz")
    with (open(plain, "wb") if form == "trace.json" else gzip.open(plain, "wb", compresslevel=1)) as f:
        f.write(b'{"events": [')
        for k, ev in enumerate(events):
            f.write((", " if k else "").encode() + json.dumps(ev).encode())
        f.write(b"]}")
    if form != PARTS:
        return
    packed = plain.read_bytes()
    plain.unlink()
    size = -(-len(packed) // parts)
    for n in range(parts):
        (eval_dir / f"{PARTS}{n:02d}").write_bytes(packed[n * size:(n + 1) * size])


def _arm_terminal(root: Path) -> Path:
    """The published arm cell's terminal eval, with its trace removed, for a test to write one."""
    published_runs(root, TRACE)
    terminal = root / TASK / ARM_CELL / "evals" / "terminal"
    (terminal / "trace.json.gz").unlink()
    return terminal


# ---------------------------------------------------------------------------
# Read an event at a time
# ---------------------------------------------------------------------------


def test_a_store_trace_is_read_an_event_at_a_time(tmp_path, monkeypatch):
    """The drive's trace as ``iter_events`` yields it, 64 turns of a 512 KiB prompt and a 512 KiB
    reply: building its index never holds a quarter of it at once, and the index holds every turn."""
    store = tmp_path / "store"
    sealed_rollout(store, tmp_path, trace=TRACE, grading_trace=TRACE)
    turns, chars = 64, 2**19

    def iter_events(_trace: Path) -> Iterator[TraceEvent]:
        for k in range(turns):
            yield _turn(k, chars)

    monkeypatch.setattr(scanner_mod, "iter_events", iter_events)
    page, peak = _peak(lambda: RolloutScanner(store / "runs").trace_page(TASK, ROLLOUT, "terminal", limit=turns))
    assert [r["request_id"] for r in page.rows] == [f"s.{k}" for k in range(turns)]
    assert peak < turns * 2 * chars / 4


@pytest.mark.parametrize("form", STREAMED_FORMS)
def test_a_published_trace_is_decompressed_and_parsed_an_event_at_a_time(tmp_path, form):
    """A published trace of 64 turns of a 512 KiB prompt and a 512 KiB reply, in each form: building
    its index never holds a quarter of the document at once, and the index holds every turn."""
    turns, chars = 64, 2**19
    terminal = _arm_terminal(tmp_path)
    _write_published(terminal, (json.loads(_turn(k, chars).model_dump_json()) for k in range(turns)), form)
    scanner = RolloutScanner(tmp_path)
    page, peak = _peak(lambda: scanner.trace_page(TASK, ARM_CELL, "terminal", limit=turns))
    assert [r["request_id"] for r in page.rows] == [f"s.{k}" for k in range(turns)]
    assert peak < turns * 2 * chars / 4


def test_the_index_holds_at_most_its_events_count_times_a_rows_bound(tmp_path):
    """Sixteen events whose every text and value runs far past its bound, judged by two verifiers
    whose records do too: the index holds at most 4 bytes a character of ``row_chars(2)`` a row."""
    events, big = 16, "x" * 2**20
    eval_dir = tmp_path / "big-task" / "20260101T000000Z-big" / "evals" / "terminal"
    _write_published(eval_dir, (_oversized_event(k, big) for k in range(events)), "trace.json")
    (eval_dir / "verdict.json").write_text(json.dumps(_oversized_verdict(events, big)))
    scanner = RolloutScanner(tmp_path)
    index, held = _held_bytes(lambda: scanner._trace_index(scanner._eval_files("big-task", "20260101T000000Z-big",
                                                                               "terminal")))
    assert len(index.rows) == events
    assert held <= events * 4 * row_chars(2)


def test_every_view_of_an_eval_is_served_from_one_read_of_its_trace(tmp_path):
    """The summary, a page, a row and the timeline of one eval read its trace once between them."""
    published_runs(tmp_path, TRACE, arm_form="trace.json.gz")
    scanner = RolloutScanner(tmp_path)
    with mock.patch.object(scanner_mod, "published_events", side_effect=published_events) as reader:
        assert scanner.trace_summary(TASK, ARM_CELL, "terminal").total == len(TRACE.events)
        assert scanner.trace_page(TASK, ARM_CELL, "terminal", limit=100).total == len(TRACE.events)
        assert scanner.trace_row(TASK, ARM_CELL, "terminal", TRACE.events[0].request_id) is not None
        assert len(scanner.timeline(TASK, ARM_CELL, "terminal").requests) == len(TRACE.events)
    assert reader.call_count == 1


# ---------------------------------------------------------------------------
# What a row holds
# ---------------------------------------------------------------------------


def _numbered(lines: int) -> str:
    """A text of *lines* numbered lines of ten characters, no two alike, so a cut shows where it fell."""
    return "".join(f"{i:09d}\n" for i in range(lines))


def test_a_long_text_is_held_cut_to_its_head_and_tail_and_marked_where_it_was_cut(tmp_path):
    """A row holds each text past ``ROW_TEXT_CHARS`` as its first and last halves of it with the
    mark between them, saying how much was cut; a list or object past it as its JSON so cut. The
    row is judged on the whole reply first: its answer line, which the cut leaves out, still
    counts. A short text, a short list and every other field are held as they are, and the page's
    dense rows show the same 160-character previews as before."""
    long = _numbered(4000)
    reply = long[:20_000] + "Answer: B\n" + long[20_000:]
    truth = [long[i:i + 10] for i in range(0, 40_000, 10)]
    eval_dir = tmp_path / "cut-task" / "20260101T000000Z-cut" / "evals" / "terminal"
    events = [
        {"request_id": "r.0", "session_id": "r", "turn_index": 0,
         "input": {"request_id": "r.0", "contents": [{"modality": "text", "text": long}]},
         "response": {"output": {"request_id": "r.0", "contents": [{"modality": "text", "text": reply}]}},
         "ground_truth": "B", "t_send": 1.0, "t_recv": 2.0, "error": None},
        {"request_id": "r.1", "session_id": "r", "turn_index": 1,
         "input": {"request_id": "r.1", "contents": [{"modality": "text", "text": "short"}]},
         "response": None, "ground_truth": truth, "t_send": 3.0, "t_recv": None, "error": long},
        {"request_id": "r.2", "session_id": "r", "turn_index": 2,
         "input": {"request_id": "r.2", "contents": [{"modality": "text", "text": "ask"}]},
         "response": {"output": {"request_id": "r.2", "contents": [{"modality": "text", "text": "C"}]}},
         "ground_truth": ["gold", 2], "t_send": 4.0, "t_recv": 4.5, "error": None},
    ]
    _write_published(eval_dir, iter(events), "trace.json")
    scanner = RolloutScanner(tmp_path)
    task, rid = "cut-task", "20260101T000000Z-cut"

    first = scanner.trace_row(task, rid, "terminal", "r.0")
    assert first["prompt"] == _cut(long, ROW_TEXT_CHARS)
    assert first["prompt"].count("[... cut") == 1
    assert "[... cut 23,616 of 40,000 characters ...]" in first["prompt"]
    assert first["artifact_text"] == _cut(reply, ROW_TEXT_CHARS)
    assert "Answer: B" not in first["artifact_text"]
    assert first["correct"] is True
    assert (first["request_id"], first["session_id"], first["t_send"], first["latency_ms"]) == ("r.0", "r", 1.0,
                                                                                                1000.0)
    second = scanner.trace_row(task, rid, "terminal", "r.1")
    assert second["error"] == _cut(long, ROW_TEXT_CHARS)
    assert second["ground_truth"] == _cut(json.dumps(truth), ROW_TEXT_CHARS)
    assert second["prompt"] == "short"
    third = scanner.trace_row(task, rid, "terminal", "r.2")
    assert (third["ground_truth"], third["artifact_text"], third["correct"]) == (["gold", 2], "C", False)

    dense = scanner.trace_page(task, rid, "terminal").rows
    assert dense[0]["artifact_text"] == reply[:160] + "..."
    assert "prompt" not in dense[0]


def _oversized_event(k: int, big: str) -> dict[str, Any]:
    """Event *k* of a published trace whose every field the row reads holds *big* or more."""
    rid = f"{k}{big}"
    return {"request_id": rid, "session_id": big, "turn_index": big,
            "input": {"request_id": rid, "contents": [{"modality": "text", "text": big}],
                      "metadata": {"tenant_id": big}},
            "response": {"output": {"request_id": rid, "contents": [{"modality": "text", "text": big,
                                                                     "tokens": big}]},
                         "t_first_packet": big, "packet_timestamps": [0.5, big]},
            "ground_truth": [big, big], "t_send": float(k), "t_recv": k + 0.5, "t_planned_send": big,
            "error": big}


def _oversized_verdict(events: int, big: str) -> dict[str, Any]:
    """A verdict of two verifiers whose every per-event record holds *big* in each field the row
    reads, on the trace of *events* events ``_oversized_event`` makes; it judges each event, so
    no row is judged by its reply's answer line."""
    diag = {"reference": {"text": big, "predicted": big}, "predicted": big, "expected": big + "!",
            "max_gap": big, "n_tokens": big, "first_above_cap_index": big}
    results = [{"passed": False, "diagnostics": diag} for _ in range(events)]
    return {"aggregate_passed": False, "per_event": {f"{big}:{k}{big}": False for k in range(events)},
            "diagnostics": {"0:ReferenceRelativeVerifier": {"artifact_value": 1.0},
                            "1:MetricVerifier": {"value": 1.0}},
            "per_verifier_results": [results, results]}


def _chars(value: Any) -> int:
    """The characters a held value takes: a text's own, anything else's JSON."""
    return len(value) if isinstance(value, str) else len(json.dumps(value))


def test_a_row_holds_at_most_its_stated_characters(tmp_path):
    """Every text and value of an event and of its verifiers' records past its bound: the row holds
    each text at ``ROW_TEXT_CHARS`` and each other value at ``ROW_VALUE_CHARS``, each with its mark,
    and all of it within ``row_chars``; its fields are the ones the bound counts."""
    big = "y" * 300_000
    eval_dir = tmp_path / "big-task" / "20260101T000000Z-big" / "evals" / "terminal"
    _write_published(eval_dir, iter([_oversized_event(0, big)]), "trace.json")
    (eval_dir / "verdict.json").write_text(json.dumps(_oversized_verdict(1, big)))
    scanner = RolloutScanner(tmp_path)
    [light] = scanner.trace_page("big-task", "20260101T000000Z-big", "terminal").rows
    row = scanner.trace_row("big-task", "20260101T000000Z-big", "terminal", light["request_id"])
    assert set(row) == {*ROW_TEXTS, *ROW_VALUES, "verifier_cells"}
    mark = len(CUT_MARK.format(cut=10**7, total=10**7))
    for name in ROW_TEXTS:
        assert _chars(row[name]) <= ROW_TEXT_CHARS + mark, name
    for name in ROW_VALUES:
        assert _chars(row[name]) <= ROW_VALUE_CHARS + mark, name
    assert row["prompt"] == _cut(big, ROW_TEXT_CHARS)
    assert row["tenant_id"] == _cut(big, ROW_VALUE_CHARS)
    assert len(row["verifier_cells"]) == 2
    for cell in row["verifier_cells"]:
        assert _chars(cell["detail"]) <= ROW_TEXT_CHARS + mark
        assert cell["detail"].startswith("y" * 100)
        assert _chars(cell["name"]) + _chars(cell["passed"]) <= 2 * (ROW_VALUE_CHARS + mark)
    held = sum(_chars(row[name]) for name in (*ROW_TEXTS, *ROW_VALUES))
    held += sum(_chars(v) for cell in row["verifier_cells"] for v in cell.values())
    assert held <= row_chars(2)


# ---------------------------------------------------------------------------
# The published reader
# ---------------------------------------------------------------------------

#: Documents ``json.loads`` reads whole, each read the same by the incremental reader: whitespace
#: everywhere JSON allows it, members before and after the events, escapes and brackets inside
#: strings, non-finite and large numbers, an empty, a null and an empty-object events value, and
#: text outside ASCII, raw and escaped.
DOCUMENTS = [
    '{"events": []}',
    ' \n\t{ "events" :\r\n[ {"a": 1} ,\n{"b": [1, 2, {"c": "x"}], "d": {}} ] , "after": {"k": [1, 2]} }\n ',
    '{"before": [true, false, null], "events": [{"t": NaN, "u": Infinity, "v": -Infinity, "w": -0.0}]}',
    '{"events": [{"s": "\\u00e9\\ud83d\\ude00 \\" \\\\ ]} [{ \\n"}, {"e": "\\\\"}]}',
    '{"events": [{"n": 12345678901234567890, "f": 1.5e300, "g": 123}, {"n": 7}]}',
    '{"events": [{"x": 1E+2, "y": 2e-3, "z": -12.5e-10, "w": [0, -1, 10.25]}], "after": 12.5}',
    '{"events": null}',
    '{"events": {}}',
    '{}',
    '{"events": [{"s": "\u00e9\u6f22\U0001f600"}, {"s": "\u00e9"}]}',
]


def _gzip_members(data: bytes, cuts: list[int], *, padding: bytes = b"") -> bytes:
    """*data* gzipped as one member per piece between *cuts*, with *padding* after each member."""
    edges = [0, *cuts, len(data)]
    return b"".join(gzip.compress(data[a:b]) + padding for a, b in zip(edges, edges[1:], strict=False))


def _read(tmp_path: Path, data: bytes, *, gzipped: bool, parts: int = 1) -> list[dict[str, Any]]:
    """The events the reader yields of *data*, written into *parts* files."""
    size = -(-len(data) // parts) if data else 1
    files = []
    for n in range(parts):
        path = tmp_path / f"{PARTS}{n:02d}"
        path.write_bytes(data[n * size:(n + 1) * size])
        files.append(path)
    return list(published_events(files, gzipped=gzipped))


def _whole(data: bytes) -> list[dict[str, Any]]:
    """The events ``json.loads`` of the whole document gives, as the viewer read them before."""
    return json.loads(data).get("events") or []


@pytest.mark.parametrize("piece", [1, 2, 3, 7, 2**20])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
@pytest.mark.parametrize("doc", DOCUMENTS)
def test_the_reader_yields_the_events_json_loads_reads_whole(tmp_path, monkeypatch, doc, encoding, piece):
    """Every document, in every encoding ``json.loads`` detects, read a few bytes at a time or a
    piece at once, plain, split, and gzipped in one member or several with zero padding between
    them: the events are those of ``json.loads`` of the whole, NaN and all."""
    monkeypatch.setattr(published, "PIECE_BYTES", piece)
    data = doc.encode(encoding)
    expected = json.dumps(_whole(data))
    assert json.dumps(_read(tmp_path, data, gzipped=False)) == expected
    assert json.dumps(_read(tmp_path, data, gzipped=False, parts=3)) == expected
    assert json.dumps(_read(tmp_path, gzip.compress(data), gzipped=True)) == expected
    members = _gzip_members(data, [len(data) // 3, len(data) // 2], padding=b"\0\0\0")
    assert gzip.decompress(members) == data
    assert json.dumps(_read(tmp_path, members, gzipped=True, parts=4)) == expected


#: Documents ``json.loads`` refuses, which the reader refuses too: a defect inside an event, after
#: the last one, or after the document; a document that ends early; an empty one.
INVALID = [
    '{"events": [{"a": }]}',
    '{"events": [{"a": 1}], "after": }',
    '{"events": [{"a": 1},]}',
    '{"events": [{"a": 1}]} x',
    '{"events": []}{}',
    '{"events": [{"a": 1}',
    '{"events": [{"a": "unterminated',
    '{"events" [] }',
    '{events: []}',
    '',
]


@pytest.mark.parametrize("doc", INVALID)
@pytest.mark.parametrize("piece", [1, 5, 2**20])
def test_the_reader_refuses_a_document_json_loads_refuses(tmp_path, monkeypatch, doc, piece):
    monkeypatch.setattr(published, "PIECE_BYTES", piece)
    data = doc.encode()
    with pytest.raises(ValueError):
        json.loads(data)
    with pytest.raises(TraceUnreadable):
        _read(tmp_path, data, gzipped=False)
    with pytest.raises(TraceUnreadable):
        _read(tmp_path, gzip.compress(data), gzipped=True)


#: Documents ``json.loads`` reads, that the reader refuses as no trace: it never held them as one
#: (a document that is not an object, events that are not a list, an event that is not an object,
#: events named twice, whose last ``json.loads`` keeps though the reader has yielded the first).
NOT_A_TRACE = [
    '[{"a": 1}]',
    '"events"',
    '{"events": {"a": 1}}',
    '{"events": "x"}',
    '{"events": 1}',
    '{"events": [1]}',
    '{"events": [{"a": 1}, []]}',
    '{"events": [], "events": []}',
    '{"events": [{"a": 1}], "events": null}',
]


@pytest.mark.parametrize("doc", NOT_A_TRACE)
def test_the_reader_refuses_a_document_that_is_no_trace(tmp_path, doc):
    json.loads(doc)
    with pytest.raises(TraceUnreadable):
        _read(tmp_path, doc.encode(), gzipped=False)


def test_the_reader_refuses_an_event_past_its_bound(tmp_path, monkeypatch):
    """An event whose JSON runs past ``EVENT_MAX_CHARS`` is refused, as is a member beside the
    events: the reader never holds more of one, so a document that never closes one ends its read
    there. One at the bound is read."""
    monkeypatch.setattr(published, "PIECE_BYTES", 64)
    monkeypatch.setattr(published, "EVENT_MAX_CHARS", 1000)
    at = {"s": "x" * (1000 - len('{"s": ""}'))}
    assert len(json.dumps(at)) == 1000
    assert _read(tmp_path, json.dumps({"events": [at, at]}).encode(), gzipped=False) == [at, at]
    past = {"s": "x" * 1000}
    with pytest.raises(TraceUnreadable):
        _read(tmp_path, json.dumps({"events": [at, past]}).encode(), gzipped=False)
    with pytest.raises(TraceUnreadable):
        _read(tmp_path, json.dumps({"events": [], "after": past}).encode(), gzipped=False)
    with pytest.raises(TraceUnreadable):
        _read(tmp_path, b'{"events": [{"s": "' + b"x" * 100_000, gzipped=False)


def _corrupt_crc(packed: bytes) -> bytes:
    """*packed* with its last member's CRC changed."""
    crc = packed[-8:-4]
    return packed[:-8] + bytes([crc[0] ^ 1]) + crc[1:] + packed[-4:]


@pytest.mark.parametrize("damage", [
    pytest.param(lambda packed: packed[:-5], id="cut-short"),
    pytest.param(lambda packed: packed[:len(packed) // 2], id="cut-in-half"),
    pytest.param(_corrupt_crc, id="bad-crc"),
    pytest.param(lambda packed: packed + b"xx", id="garbage-after"),
    pytest.param(lambda packed: packed + b"\0\0\x1f", id="partial-member-after-padding"),
    pytest.param(lambda packed: b"\0" + packed, id="zero-before-the-first"),
])
def test_the_reader_refuses_a_gzip_stream_gzip_decompress_refuses(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(published, "PIECE_BYTES", 3)
    damaged = damage(gzip.compress(json.dumps({"events": [{"a": 1}]}).encode()))
    with pytest.raises((OSError, EOFError, zlib.error)):
        gzip.decompress(damaged)
    with pytest.raises(TraceUnreadable):
        _read(tmp_path, damaged, gzipped=True)


def test_a_published_trace_the_reader_refuses_is_no_trace(tmp_path):
    """The scanner shows a trace the reader refuses part way as none, holding none of its events:
    no summary, an empty page, no row."""
    terminal = _arm_terminal(tmp_path)
    events = [json.loads(ev.model_dump_json()) for ev in TRACE.events]
    data = json.dumps({"events": events}).encode() + b" trailing"
    (terminal / "trace.json.gz").write_bytes(gzip.compress(data))
    scanner = RolloutScanner(tmp_path)
    assert scanner.trace_summary(TASK, ARM_CELL, "terminal") is None
    assert scanner.trace_page(TASK, ARM_CELL, "terminal").total == 0
    assert scanner.trace_row(TASK, ARM_CELL, "terminal", TRACE.events[0].request_id) is None
