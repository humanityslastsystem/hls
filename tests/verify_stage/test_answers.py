"""The reference answers on the store: one create-only file per call, merged earliest first,
keyed by fingerprint, GPU kind and reference image."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import contextmanager

import pytest

import hls.verify.answers as answers_module
from hls.core.decode import Greedy
from hls.core.payloads import Envelope, TextPayload
from hls.spec.drivers import stamp_request, user_message
from hls.spec.tenant import Tenant
from hls.store.posix import PosixStore
from hls.verify import LetterMatchMetric, MetricVerifier
from hls.verify.answers import (ANSWER_FILE_MAX_BYTES, CORPUS_CAPABILITIES, AnswerEntry, AnswerFile,
                                AnswersNotWritten, Question, QuestionSet, QuestionSetConflict, StoreAnswers,
                                answer_key, answers_area, multi_turn_caching, prompt_digest, questions_name)
from hls.verify.run import may_deploy
from tests.verify_stage import fixture

FP = "HFEngine_abc12345_greedy_def67890"
IMAGE = "a" * 64


class _Oracle:
    """Answers each prompt with its request id and this oracle's tag, recording what it was asked."""

    def __init__(self, tag: str = "first") -> None:
        self.tag = tag
        self.asked: list[list[str]] = []

    async def __call__(self, prompts):
        self.asked.append([p.request_id for p in prompts])
        return [Envelope(request_id=p.request_id, contents=[TextPayload(text=f"{self.tag} answer to {p.request_id}")])
                for p in prompts]


class _Dies:
    async def __call__(self, prompts):
        raise RuntimeError("the reference engine died")


def _prompts(*rids):
    return [Envelope(request_id=r, contents=[TextPayload(text=f"q {r}")]) for r in rids]


def _texts(envs):
    return [e.first_of("text").text for e in envs]


def _warnings(caplog) -> list[str]:
    """The warnings the answers module logged. Records are filtered by logger and level rather than
    taken whole: a test that configures logging, as smac's import does, lets other records reach
    the capture, the module's own records under WARNING among them."""
    return [r.getMessage() for r in caplog.records
            if r.name == answers_module.log.name and r.levelno >= logging.WARNING]


def _call(cache, oracle, prompts, fingerprint=FP):
    return asyncio.run(cache.wrap(fingerprint, oracle)(prompts))


@pytest.fixture
def area(tmp_path):
    return PosixStore(tmp_path / "store").path(answers_area("asr"))


def _answers(area, writer, *, gpu_kind="H200", image=IMAGE, clock=None):
    area.mkdir(parents=True, exist_ok=True)
    kw = {} if clock is None else {"clock": clock}
    return StoreAnswers(PosixStore(area), gpu_kind=gpu_kind, image=image, writer=writer, **kw)


def _files(area, gpu_kind="H200", image=IMAGE):
    key = area / answer_key(FP, gpu_kind, image)
    return sorted(p.name for p in key.iterdir()) if key.is_dir() else []


def test_the_area_of_a_task_is_one_directory_under_oracle(area):
    assert answers_area("asr") == "oracle/asr"
    with pytest.raises(ValueError, match="segment"):
        answers_area("asr/../runs")


def test_a_call_with_misses_adds_one_file_that_another_writer_is_served(area):
    first, oracle = _answers(area, "g1+verify+1"), _Oracle()
    out = _call(first, oracle, _prompts("r0", "r1"))
    assert _texts(out) == ["first answer to r0", "first answer to r1"]
    assert _files(area) == ["g1+verify+1.000001.json"]
    key = answer_key(FP, "H200", IMAGE)
    file = AnswerFile.model_validate(json.loads((area / key / "g1+verify+1.000001.json").read_bytes()))
    assert (file.fingerprint, file.gpu_kind, file.image, file.writer) == (FP, "H200", IMAGE, "g1+verify+1")
    assert [e.request_id for e in file.answers] == ["r0", "r1"]
    assert list(first.written) == [f"{key}/g1+verify+1.000001.json"]

    second, oracle2 = _answers(area, "g2+verify+1"), _Oracle("second")
    out = _call(second, oracle2, _prompts("r0", "r1"))
    assert oracle2.asked == [], "every answer was on the store"
    assert _texts(out) == ["first answer to r0", "first answer to r1"]
    assert second.written == {} and second.read == first.written
    # The reader keeps the writer each file it merged names.
    assert second.read_writers == {f"{key}/g1+verify+1.000001.json": "g1+verify+1"}


def test_every_call_that_answers_misses_banks_its_own_file_before_the_next_runs(area):
    """A role that dies keeps every call it finished."""
    cache = _answers(area, "g1+verify+1")
    _call(cache, _Oracle(), _prompts("r0"))
    with pytest.raises(RuntimeError, match="died"):
        _call(cache, _Dies(), _prompts("r1"))
    assert _files(area) == ["g1+verify+1.000001.json"]
    _call(cache, _Oracle(), _prompts("r2"))
    assert _files(area) == ["g1+verify+1.000001.json", "g1+verify+1.000002.json"]

    oracle = _Oracle("later")
    out = _call(_answers(area, "g2+verify+1"), oracle, _prompts("r0", "r1", "r2"))
    assert oracle.asked == [["r1"]]
    assert _texts(out) == ["first answer to r0", "later answer to r1", "first answer to r2"]


def test_a_call_with_no_misses_writes_nothing(area):
    _call(_answers(area, "g1+verify+1"), _Oracle(), _prompts("r0"))
    cache = _answers(area, "g1+verify+2")
    _call(cache, _Oracle(), _prompts("r0"))
    assert cache.written == {} and _files(area) == ["g1+verify+1.000001.json"]


def test_concurrent_writers_keep_each_others_answers(area):
    """Two gradings that both started cold each add their own file; neither replaces the other's."""
    a, b = _answers(area, "g1+verify+1"), _answers(area, "g2+verify+1")
    a.wrap(FP, _Oracle())
    b.wrap(FP, _Oracle())                                       # both loaded before either wrote
    _call(a, _Oracle("a"), _prompts("r0"))
    _call(b, _Oracle("b"), _prompts("r1"))
    assert _files(area) == ["g1+verify+1.000001.json", "g2+verify+1.000001.json"]

    oracle = _Oracle("third")
    out = _call(_answers(area, "g3+verify+1"), oracle, _prompts("r0", "r1"))
    assert oracle.asked == []
    assert _texts(out) == ["a answer to r0", "b answer to r1"]


def _plant(area, writer: str, written_at: float, text: str, *, fingerprint: str = FP, gpu_kind: str = "H200",
           image: str = IMAGE, asked: str = "q r0") -> None:
    """A file *writer* wrote at *written_at* into the directory of (FP, H200, IMAGE), answering
    r0, asked *asked*, with *text*. Its header names *fingerprint*, *gpu_kind* and *image*."""
    prompt = Envelope(request_id="r0", contents=[TextPayload(text=asked)])
    answer = Envelope(request_id="r0", contents=[TextPayload(text=text)]).model_dump(mode="json")
    file = AnswerFile(fingerprint=fingerprint, gpu_kind=gpu_kind, image=image, writer=writer, written_at=written_at,
                      answers=[AnswerEntry(request_id="r0", digest=prompt_digest(prompt), answer=answer)])
    key = area / answer_key(FP, "H200", IMAGE)
    key.mkdir(parents=True, exist_ok=True)
    (key / f"{writer}.000001.json").write_text(json.dumps(file.model_dump(mode="json")))


def test_the_earliest_written_answer_is_served_whatever_the_file_names(area):
    """Two cold gradings that answered one question differently: every later verify serves the
    one written first, not whichever file name sorts first."""
    _plant(area, "z-early", 100.0, "early answer to r0")
    _plant(area, "a-late", 200.0, "late answer to r0")
    oracle = _Oracle("third")
    out = _call(_answers(area, "g3+verify+1"), oracle, _prompts("r0"))
    assert oracle.asked == [] and _texts(out) == ["early answer to r0"]


def test_files_written_at_one_time_are_read_in_name_order(area):
    _plant(area, "b-second", 100.0, "b answer to r0")
    _plant(area, "a-first", 100.0, "a answer to r0")
    assert _texts(_call(_answers(area, "g3+verify+1"), _Oracle(), _prompts("r0"))) == ["a answer to r0"]


@pytest.mark.parametrize("other", [{"gpu_kind": "A100"}, {"image": "b" * 64}], ids=["gpu-kind", "image"])
def test_answers_from_other_silicon_or_another_image_miss(area, other):
    _call(_answers(area, "g1+verify+1"), _Oracle(), _prompts("r0"))
    oracle = _Oracle("other")
    out = _call(_answers(area, "g2+verify+1", **other), oracle, _prompts("r0"))
    assert oracle.asked == [["r0"]] and _texts(out) == ["other answer to r0"]


def test_another_fingerprint_misses(area):
    _call(_answers(area, "g1+verify+1"), _Oracle(), _prompts("r0"))
    oracle = _Oracle("other")
    _call(_answers(area, "g2+verify+1"), oracle, _prompts("r0"), fingerprint="HFEngine_other")
    assert oracle.asked == [["r0"]]


def test_an_answer_to_another_prompt_under_the_same_request_id_misses(area):
    _call(_answers(area, "g1+verify+1"), _Oracle(), _prompts("r0"))
    oracle = _Oracle("other")
    asked = [Envelope(request_id="r0", contents=[TextPayload(text="another question")])]
    assert _texts(_call(_answers(area, "g2+verify+1"), oracle, asked)) == ["other answer to r0"]
    assert oracle.asked == [["r0"]]


def test_an_earlier_answer_to_another_prompt_does_not_hide_a_later_answer_to_the_asked_one(area):
    """The first answer wins per question, a request id with its prompt's digest: a request id
    whose prompt changed is served the answer to its new prompt once some grading wrote one."""
    _plant(area, "g0+verify+1", 100.0, "answer to the old question", asked="old question")
    _plant(area, "g1+verify+1", 200.0, "answer to the new question", asked="q r0")
    oracle = _Oracle("fresh")
    assert _texts(_call(_answers(area, "g2+verify+1"), oracle, _prompts("r0"))) == ["answer to the new question"]
    assert oracle.asked == []


def test_two_prompts_under_one_request_id_are_both_served_from_the_store(area):
    _plant(area, "g0+verify+1", 100.0, "answer to the old question", asked="old question")
    _plant(area, "g1+verify+1", 200.0, "answer to the new question", asked="q r0")
    asked = [Envelope(request_id="r0", contents=[TextPayload(text="old question")]), *_prompts("r0")]
    oracle = _Oracle("fresh")
    out = _call(_answers(area, "g2+verify+1"), oracle, asked)
    assert oracle.asked == [] and _texts(out) == ["answer to the old question", "answer to the new question"]


def _capped(cap: int, *rids: str) -> list[Envelope]:
    return [Envelope(request_id=r, contents=[TextPayload(text=f"q {r}")], metadata={"max_tokens": cap})
            for r in rids]


def test_a_prompt_that_changed_under_its_request_id_is_answered_once_then_served(area):
    """A task edit that changes what the digest covers but not the answer key (here the output
    cap): one grading answers the new prompts, and every later one is served them."""
    rows = [f"r{i}" for i in range(100)]
    _call(_answers(area, "g1+verify+1"), _Oracle(), _capped(256, *rows))
    asked = []
    for n in (2, 3, 4):
        oracle = _Oracle(f"grading {n}")
        _call(_answers(area, f"g{n}+verify+1"), oracle, _capped(512, *rows))
        asked.append(sum(len(call) for call in oracle.asked))
    assert asked == [100, 0, 0]


def test_a_store_none_of_whose_answers_match_names_its_own_filler(area, caplog):
    _call(_answers(area, "g1+verify+1"), _Oracle(), _capped(256, "r0"))
    with caplog.at_level(logging.WARNING):
        _call(_answers(area, "g2+verify+1"), _Oracle(), _capped(512, "r0"))
    [warning] = [r.getMessage() for r in caplog.records if "none match" in r.getMessage()]
    assert warning.endswith("fill the area with hls oracle fill")


def test_a_chunked_fill_of_new_rows_does_not_warn(area, caplog):
    """A fill asks a split chunk by chunk through one wrapper: each chunk after the first misses
    every prompt while the area holds the chunks before it, which is no unreachable area."""
    call = _answers(area, "w1+oracle_fill+1").wrap(FP, _Oracle())
    with caplog.at_level(logging.WARNING):
        for rid in ("r0", "r1", "r2"):
            asyncio.run(call(_prompts(rid)))
    assert _warnings(caplog) == []


def test_a_resumed_fill_does_not_warn_once_a_stored_answer_was_served(area, caplog):
    """A fill that died resumes: its first chunks are served from what the last one stored,
    and the rest miss, which is no unreachable area either."""
    _call(_answers(area, "w1+oracle_fill+1"), _Oracle(), _prompts("r0"))
    call = _answers(area, "w2+oracle_fill+1").wrap(FP, _Oracle())
    with caplog.at_level(logging.WARNING):
        for rid in ("r0", "r1"):
            asyncio.run(call(_prompts(rid)))
    assert _warnings(caplog) == []


def test_a_call_with_no_prompts_does_not_warn(area, caplog):
    _call(_answers(area, "g1+verify+1"), _Oracle(), _prompts("r0"))
    with caplog.at_level(logging.WARNING):
        _call(_answers(area, "g2+verify+1"), _Oracle(), [])
    assert _warnings(caplog) == []


@pytest.mark.parametrize("plant", ["garbage", "other-reference", "other-gpu-kind", "other-image", "directory"])
def test_a_file_that_does_not_parse_or_names_another_key_is_skipped_with_a_warning(area, caplog, plant):
    _call(_answers(area, "g1+verify+1"), _Oracle(), _prompts("r0"))
    key = area / answer_key(FP, "H200", IMAGE)
    if plant == "garbage":
        (key / "g0+verify+1.000001.json").write_text("{not json")
    elif plant == "directory":
        (key / "g0+verify+1.000001.json").mkdir()
    else:
        # Copied into this key's directory from another reference's, GPU kind's or image's.
        other = {"other-reference": {"fingerprint": "HFEngine_other"}, "other-gpu-kind": {"gpu_kind": "A100"},
                 "other-image": {"image": "b" * 64}}[plant]
        _plant(area, "g0+verify+1", 0.0, "wrong answer to r0", **other)
    with caplog.at_level(logging.WARNING):
        cache = _answers(area, "g2+verify+1")
        out = _call(cache, _Oracle("x"), _prompts("r0"))
    assert _texts(out) == ["first answer to r0"]
    assert list(cache.read) == [f"{answer_key(FP, 'H200', IMAGE)}/g1+verify+1.000001.json"]
    assert any("g0+verify+1.000001.json" in r.getMessage() and "skipping" in r.getMessage()
               for r in caplog.records), caplog.text


def test_a_file_over_the_bound_is_skipped_unread(area, caplog):
    _call(_answers(area, "g1+verify+1"), _Oracle(), _prompts("r0"))
    big = area / answer_key(FP, "H200", IMAGE) / "g0+verify+1.000001.json"
    with open(big, "wb") as f:
        f.truncate(ANSWER_FILE_MAX_BYTES + 1)
    with caplog.at_level(logging.WARNING):
        out = _call(_answers(area, "g2+verify+1"), _Oracle("x"), _prompts("r0"))
    assert _texts(out) == ["first answer to r0"]
    assert any("g0+verify+1.000001.json" in r.getMessage() and f"{ANSWER_FILE_MAX_BYTES + 1} bytes" in r.getMessage()
               for r in caplog.records), caplog.text


class _Growing(PosixStore):
    """A store whose file grows by *grow* bytes of whitespace right after its size is read, as a
    concurrent writer's would, and that counts the bytes read from it."""

    def __init__(self, root, grow: int) -> None:
        super().__init__(root)
        self.grow = grow
        self.bytes_read = 0

    def size(self, key):
        n = super().size(key)
        with open(self.path(key), "ab") as f:
            f.write(b" " * self.grow)
        return n

    @contextmanager
    def open_read(self, key):
        store = self

        class Counting:
            def __init__(self, f):
                self.f = f

            def read(self, n=-1):
                data = self.f.read(n)
                store.bytes_read += len(data)
                return data

        with super().open_read(key) as f:
            yield Counting(f)


def test_a_file_that_grows_past_the_bound_after_its_size_was_read_is_skipped_reading_no_more_than_the_bound(
        area, caplog, monkeypatch):
    """The size checked first is no bound on the read: the file can grow before it is opened,
    so the read itself stops one byte past the bound, and a file that held more is skipped."""
    bound = 4096
    monkeypatch.setattr(answers_module, "ANSWER_FILE_MAX_BYTES", bound)
    # A whole answers file followed by whitespace still parses, so only the bound skips it.
    _plant(area, "g0+verify+1", 0.0, "planted")
    store = _Growing(area, grow=16 * bound)
    oracle = _Oracle("fresh")
    with caplog.at_level(logging.WARNING):
        out = _call(StoreAnswers(store, gpu_kind="H200", image=IMAGE, writer="g2+verify+1"), oracle, _prompts("r0"))
    assert oracle.asked == [["r0"]] and _texts(out) == ["fresh answer to r0"]
    assert store.bytes_read == bound + 1
    assert any("g0+verify+1.000001.json" in r.getMessage() and "skipping" in r.getMessage()
               for r in caplog.records), caplog.text


def test_a_file_whose_answer_is_not_an_envelope_is_skipped_rather_than_served(area, caplog):
    """Serving it would fail every verify of the key: the area is an optimisation, never a gate."""
    _plant(area, "g0+verify+1", 0.0, "planted")
    path = area / answer_key(FP, "H200", IMAGE) / "g0+verify+1.000001.json"
    body = json.loads(path.read_text())
    body["answers"][0]["answer"] = {"contents": "not a list of payloads"}
    path.write_text(json.dumps(body))
    oracle = _Oracle("fresh")
    with caplog.at_level(logging.WARNING):
        out = _call(_answers(area, "g2+verify+1"), oracle, _prompts("r0"))
    assert oracle.asked == [["r0"]] and _texts(out) == ["fresh answer to r0"]
    assert any("g0+verify+1.000001.json" in r.getMessage() and "skipping" in r.getMessage()
               for r in caplog.records), caplog.text


@pytest.mark.parametrize("position", [
    {"token_id": 1, "top_logprobs": []},
    {"token_id": 1, "logprob": 0.0, "top_logprobs": [1, 2]},
    {"token_id": 1, "logprob": [1], "top_logprobs": []},
    {"token_id": 1, "logprob": 0.0, "top_logprobs": [{"token_id": 2**70, "logprob": 0.0}]},
], ids=["logprob-missing", "alternatives-not-dicts", "logprob-a-list", "id-past-int64"])
def test_an_audio_answer_with_a_malformed_codec_position_is_skipped_rather_than_failing_the_verify(
        area, caplog, position):
    """The codec position's own validator raises KeyError, TypeError or OverflowError, which
    pydantic passes through unwrapped; any of them skips the file, as one that does not parse."""
    _plant(area, "g0+verify+1", 0.0, "planted")
    path = area / answer_key(FP, "H200", IMAGE) / "g0+verify+1.000001.json"
    body = json.loads(path.read_text())
    body["answers"][0]["answer"]["contents"] = [{"modality": "audio", "audio_b64": "AAAA", "sample_rate": 16000,
                                                 "codec_logprobs": [position]}]
    path.write_text(json.dumps(body))
    oracle = _Oracle("fresh")
    with caplog.at_level(logging.WARNING):
        out = _call(_answers(area, "g2+verify+1"), oracle, _prompts("r0"))
    assert oracle.asked == [["r0"]] and _texts(out) == ["fresh answer to r0"]
    assert any("g0+verify+1.000001.json" in r.getMessage() and "skipping" in r.getMessage()
               for r in caplog.records), caplog.text


def test_a_write_that_fails_is_logged_and_the_answers_still_returned(area, caplog):
    cache = _answers(area, "g1+verify+1")
    area.chmod(0o500)
    try:
        with caplog.at_level(logging.WARNING):
            out = _call(cache, _Oracle(), _prompts("r0"))
    finally:
        area.chmod(0o700)
    assert _texts(out) == ["first answer to r0"]
    assert cache.written == {}
    assert any("could not write reference answers" in r.getMessage() for r in caplog.records), caplog.text


def test_a_strict_writer_raises_a_write_that_fails_naming_its_error(area):
    """The oracle fill's answers are what it exists to bank, so it fails rather than succeed
    with answers it could not write; a verify's writer only logs (above)."""
    area.mkdir(parents=True)
    cache = StoreAnswers(PosixStore(area), gpu_kind="H200", image=IMAGE, writer="w1+oracle_fill+1", strict=True)
    area.chmod(0o500)
    try:
        with pytest.raises(AnswersNotWritten, match="could not write reference answers .*Permission denied"):
            _call(cache, _Oracle(), _prompts("r0"))
    finally:
        area.chmod(0o700)
    assert cache.written == {}


def test_a_writer_never_replaces_a_file(area, caplog):
    """A writer name that was used before (which the stage never hands out) loses its write
    rather than replacing the other's answers."""
    _call(_answers(area, "g1+verify+1"), _Oracle(), _prompts("r0"))
    key = area / answer_key(FP, "H200", IMAGE)
    before = (key / "g1+verify+1.000001.json").read_bytes()
    with caplog.at_level(logging.WARNING):
        again = _answers(area, "g1+verify+1")
        _call(again, _Oracle("again"), _prompts("r1"))
    assert (key / "g1+verify+1.000001.json").read_bytes() == before
    assert again.written == {}
    assert any("could not write" in r.getMessage() for r in caplog.records)


def test_the_answers_are_loaded_once_per_fingerprint(area):
    cache = _answers(area, "g1+verify+1")
    _call(cache, _Oracle(), _prompts("r0"))
    _call(_answers(area, "g2+verify+1"), _Oracle(), _prompts("r1"))
    oracle = _Oracle("again")
    _call(cache, oracle, _prompts("r1"))
    assert oracle.asked == [["r1"]], "a writer reads the area when it first wraps a fingerprint, not on every call"


def _at_turn(rid: str, turn: int) -> Envelope:
    """Row *rid*'s prompt as the drive stamps it at turn *turn* of its session."""
    [row] = _prompts(rid)
    return stamp_request(row, [user_message(row)], tenant_id="t0", session_id=rid, turn_idx=turn, max_tokens=8)


def test_a_prompt_past_its_sessions_first_turn_is_answered_afresh_and_never_banked(area):
    """From turn two a prompt's messages carry the server's replies, so its answer depends on a
    contestant's text, which never reaches the area, whatever the task declares about its
    turns. It is asked in the same call as the first turns' misses, as without a store."""
    first, oracle = _answers(area, "g1+verify+1"), _Oracle()
    out = _call(first, oracle, [_at_turn("r0", 0), _at_turn("r1", 1)])
    assert oracle.asked == [["r0", "r1"]]
    assert _texts(out) == ["first answer to r0", "first answer to r1"]
    [path] = first.written
    assert [e.request_id for e in AnswerFile.model_validate_json((area / path).read_bytes()).answers] == ["r0"]

    oracle = _Oracle("second")
    out = _call(_answers(area, "g2+verify+1"), oracle, [_at_turn("r0", 0), _at_turn("r1", 1)])
    assert oracle.asked == [["r1"]]
    assert _texts(out) == ["first answer to r0", "second answer to r1"]


def test_a_stored_answer_is_never_served_to_a_prompt_past_its_sessions_first_turn(area):
    """An answer to a later turn already in the area (written by hand, say) is
    not served: the engine answers afresh, and nothing is added."""
    asked = _at_turn("r0", 1)
    answer = Envelope(request_id="r0", contents=[TextPayload(text="stored answer to r0")]).model_dump(mode="json")
    file = AnswerFile(fingerprint=FP, gpu_kind="H200", image=IMAGE, writer="g0+verify+1", written_at=0.0,
                      answers=[AnswerEntry(request_id="r0", digest=prompt_digest(asked), answer=answer)])
    key = area / answer_key(FP, "H200", IMAGE)
    key.mkdir(parents=True)
    (key / "g0+verify+1.000001.json").write_text(file.model_dump_json())
    cache, oracle = _answers(area, "g1+verify+1"), _Oracle("fresh")
    assert _texts(_call(cache, oracle, [asked])) == ["fresh answer to r0"]
    assert oracle.asked == [["r0"]] and cache.written == {}
    assert _files(area) == ["g0+verify+1.000001.json"]


def test_a_call_of_later_turns_alone_adds_no_file_and_names_what_it_did_not_keep(area, caplog):
    _call(_answers(area, "g1+verify+1"), _Oracle(), [_at_turn("r0", 0)])
    cache = _answers(area, "g2+verify+1")
    with caplog.at_level(logging.WARNING):
        _call(cache, _Oracle(), [_at_turn("r1", 1), _at_turn("r2", 3)])
    assert cache.written == {} and _files(area) == ["g1+verify+1.000001.json"]
    # The area holds an answer and the call missed every prompt, but no prompt could have been
    # served: that is no unreachable area.
    [warning] = _warnings(caplog)
    assert "2 of 2 prompts come after their session's first turn" in warning


def test_a_caching_reference_is_refused_on_an_agentic_task_whatever_its_workload_declares():
    """An agentic task's driver runs one trajectory per row, as many calls as the framework
    takes, each carrying the server's earlier replies; its workload's declared turns per
    session say nothing of that."""
    assert multi_turn_caching(fixture.cached_task()) == []
    [reason] = multi_turn_caching(fixture.cached_task(agent=True))
    assert reason.startswith("the task declares an agent, so tenant t0's trajectories send calls")
    # One reason per reference, however many rules it breaks.
    [reason] = multi_turn_caching(fixture.cached_task(turns=2.5, agent=True))
    assert reason.startswith("the task declares an agent")
    [reason] = multi_turn_caching(fixture.cached_task(turns=2.5))
    assert reason.startswith("tenant t0's workload runs 2.5 turns per session")


def test_a_file_records_when_it_was_written(area):
    cache = _answers(area, "g1+verify+1", clock=lambda: 1234.5)
    _call(cache, _Oracle(), _prompts("r0"))
    [path] = cache.written
    assert AnswerFile.model_validate_json((area / path).read_bytes()).written_at == 1234.5


def test_the_writer_name_is_one_store_segment(area):
    with pytest.raises(ValueError, match="segment"):
        _answers(area, "runs/asr/x")


def test_each_reference_silicon_and_image_has_its_own_directory(area):
    """The header check alone keeps another key's answers out; the directories also keep each
    key's files apart, so a reader lists only its own."""
    keys = {answer_key(f, kind, image) for f in (FP, "HFEngine_other") for kind in ("H200", "A100")
            for image in (IMAGE, "b" * 64)}
    assert len(keys) == 8
    _call(_answers(area, "g1+verify+1"), _Oracle(), _prompts("r0"))
    _call(_answers(area, "g2+verify+1", gpu_kind="A100"), _Oracle(), _prompts("r0"))
    _call(_answers(area, "g3+verify+1", image="b" * 64), _Oracle(), _prompts("r0"))
    assert sorted(p.name for p in area.iterdir()) == sorted(
        [answer_key(FP, "H200", IMAGE), answer_key(FP, "A100", IMAGE), answer_key(FP, "H200", "b" * 64)])


# ---------------------------------------------------------------------------
# What the digest covers
# ---------------------------------------------------------------------------

_TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}


def _turn(**metadata) -> Envelope:
    """A sent corpus turn with *metadata* added to what the driver stamps."""
    return Envelope(request_id="r0", contents=[TextPayload(text="q r0")],
                    metadata={"messages": [{"role": "user", "content": "q r0"}], "max_tokens": 64,
                              "tenant_id": "t0", "session_id": "r0", "turn_index": 0, **metadata})


def _digested(tools) -> str:
    """The digest of ``_turn``'s question offering *tools*, as the digest's rule states it."""
    payload = {"contents": [TextPayload(text="q r0").model_dump(mode="json")],
               "metadata": {"messages": [{"role": "user", "content": "q r0"}], "max_tokens": 64, "tools": tools}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("turn", [_turn(), _turn(tools=None), _turn(tools=[]), _turn(tools={"type": "function"})],
                         ids=["absent", "null", "empty", "not-a-list"])
def test_every_digest_covers_the_turns_tools_and_no_tools_reads_one_way(turn):
    """The engines render a turn's tools into its prompt, so the digest covers them on every turn,
    read as the engines read them (``tools_of``): absent, null, an empty list and anything but a
    list are no tools alike. The harness bookkeeping (tenant, session, turn) is not covered."""
    assert prompt_digest(turn) == _digested(None)
    assert prompt_digest(_turn(tools=[_TOOL])) == _digested([_TOOL])


def test_a_turn_offered_other_tools_is_another_question(area):
    """A turn that differs from an answered one only in its tools misses."""
    _call(_answers(area, "g1+verify+1"), _Oracle(), [_turn(tools=[_TOOL])])
    for tools in ([{**_TOOL, "type": "other"}], None):
        oracle = _Oracle("other")
        _call(_answers(area, "g2+verify+1"), oracle, [_turn(tools=tools)])
        assert oracle.asked == [["r0"]]
    oracle = _Oracle("again")
    _call(_answers(area, "g3+verify+1"), oracle, [_turn(tools=[_TOOL])])
    assert oracle.asked == []


def test_only_a_corpus_question_is_stored():
    """A capability whose input is a contestant's reply has no answer another grading could be
    served, since no two arms ask it."""
    from hls.core.capabilities import DecodeCodes, Generate, ReferenceLogprobs, StackReport, TeacherForce

    assert CORPUS_CAPABILITIES == (Generate,)
    assert all(c not in CORPUS_CAPABILITIES for c in (TeacherForce, ReferenceLogprobs, DecodeCodes, StackReport))


# ---------------------------------------------------------------------------
# Deciding ahead of a verify whether a reference engine may load
# ---------------------------------------------------------------------------


def _stamped(rid: str, tenant: str = "t0", turn: int = 0) -> Envelope:
    return Envelope(request_id=rid, contents=[TextPayload(text=f"q {rid}")],
                    metadata={"tenant_id": tenant, "turn_index": turn})


def test_unanswered_is_what_a_wrapped_call_would_hand_its_engine(area):
    answered = _stamped("r0")
    _call(_answers(area, "fill"), _Oracle(), [answered])
    answers = _answers(area, "decide")
    assert answers.unanswered(FP, answered) is False
    # Another prompt under the same request id is another question.
    assert answers.unanswered(FP, answered.model_copy(update={"contents": [TextPayload(text="other")]})) is True
    # A prompt past its session's first turn is never served, though its question is held.
    assert answers.unanswered(FP, _stamped("r0", turn=1)) is True
    # Another fingerprint's key holds nothing.
    assert answers.unanswered("HFEngine_other", answered) is True
    assert answers.written == {}


def _two_tenant_task():
    """Tenant a asks a caching reference; tenant b only matches letters, with no engine."""
    ask = fixture.AskVerifier(reference=fixture.GeneratorReference(decode=Greedy(), cache=True))
    return fixture._task("verify-stage-decide", tenants=[
        Tenant(id="a", workload=fixture._workload(), verifiers=[ask]),
        Tenant(id="b", workload=fixture._workload(), verifiers=[MetricVerifier(metric=LetterMatchMetric())])])


def test_a_reference_served_whole_for_its_own_tenants_prompts_may_not_deploy(area):
    task = _two_tenant_task()
    [verifier] = task.effective_tenants()[0].verifiers
    _call(_answers(area, "fill"), _Oracle(), [_stamped("r0", "a"), _stamped("r1", "a")],
          fingerprint=verifier.reference.fingerprint(task))
    # Tenant b's prompts are never asked of tenant a's reference, so they do not count.
    prompts = [_stamped("r0", "a"), _stamped("r0", "b"), _stamped("r1", "a"), _stamped("r9", "b")]
    assert may_deploy(task, prompts, _answers(area, "decide")) is False
    assert may_deploy(task, [*prompts, _stamped("r2", "a")], _answers(area, "decide")) is True
    assert may_deploy(task, prompts, None) is True


def test_a_verifier_with_an_engine_that_does_not_cache_may_deploy(area):
    task = fixture._task("verify-stage-pair-cached", workload=fixture._workload(), verifiers=[fixture.PairVerifier(
        generator=fixture.GeneratorReference(decode=Greedy(), cache=True),
        forcer=fixture.ForcerReference(decode=Greedy()))])
    [verifier] = task.effective_tenants()[0].verifiers
    _call(_answers(area, "fill"), _Oracle(), [_stamped("r0")], fingerprint=verifier.generator.fingerprint(task))
    # Every generator answer is held, but the forcer teacher-forces the replies, which no area holds.
    assert may_deploy(task, [_stamped("r0")], _answers(area, "decide")) is True


def test_a_verifier_whose_two_engines_place_themselves_may_deploy(area):
    """Per-engine placement holds two engines of one verifier on GPUs of their own; the decision
    still weighs both: the generator's answers are all held, the forcer's are not."""
    task = fixture._task("verify-stage-pair-placed", workload=fixture._workload(), verifiers=[fixture.PairVerifier(
        generator=fixture.PlacedGeneratorReference(decode=Greedy(), cache=True),
        forcer=fixture.PlacedForcerReference(decode=Greedy()))])
    [verifier] = task.effective_tenants()[0].verifiers
    _call(_answers(area, "fill"), _Oracle(), [_stamped("r0")], fingerprint=verifier.generator.fingerprint(task))
    assert may_deploy(task, [_stamped("r0")], _answers(area, "decide")) is True


def test_no_engine_loads_only_once_every_caching_reference_of_every_verifier_is_served_whole(area):
    """Two verifiers, each asking a caching reference of its own, one of them placing itself:
    the weights stay behind only when both references' answers are all held."""
    placed = fixture.AskVerifier(reference=fixture.PlacedGeneratorReference(decode=Greedy(), cache=True))
    plain = fixture.AskVerifier(reference=fixture.GeneratorReference(decode=Greedy(), cache=True))
    task = fixture._task("verify-stage-two-asks", workload=fixture._workload(), verifiers=[placed, plain])
    first, second = (v.reference.fingerprint(task) for v in (placed, plain))
    assert first != second
    prompts = [_stamped("r0"), _stamped("r1")]
    _call(_answers(area, "fill"), _Oracle(), prompts, fingerprint=first)
    assert may_deploy(task, prompts, _answers(area, "decide")) is True
    _call(_answers(area, "fill"), _Oracle(), prompts, fingerprint=second)
    assert may_deploy(task, prompts, _answers(area, "decide")) is False


def test_a_task_without_engines_never_deploys_and_reads_no_prompt():
    def unread():
        raise AssertionError("a prompt was read")
        yield

    assert may_deploy(fixture.metric_task(), unread(), None) is False
    assert may_deploy(fixture.gate_task(), unread(), None) is True


# ---------------------------------------------------------------------------
# The question set a completed fill records, and the digest-level reader
# ---------------------------------------------------------------------------

ROWS_KEY, SHAPE = "rows-0123456789abcdef", "b" * 64


def _strict(area, writer="w1+oracle_fill+1"):
    area.mkdir(parents=True, exist_ok=True)
    return StoreAnswers(PosixStore(area), gpu_kind="H200", image=IMAGE, writer=writer, strict=True)


def _questions(*rids, tenant="t0"):
    return [Question(tenant=tenant, request_id=r, digest=hashlib.sha256(r.encode()).hexdigest()) for r in rids]


def test_answered_asks_by_request_id_and_digest_what_unanswered_asks_of_a_prompt(area):
    prompt = _stamped("r0")
    _call(_answers(area, "fill"), _Oracle(), [prompt])
    answers = _answers(area, "decide")
    assert answers.answered(FP, "r0", prompt_digest(prompt)) is True
    assert answers.answered(FP, "r0", "0" * 64) is False
    assert answers.answered(FP, "r1", prompt_digest(prompt)) is False
    assert answers.answered("HFEngine_other", "r0", prompt_digest(prompt)) is False
    assert answers.unanswered(FP, prompt) is False


def test_the_question_set_name_holds_the_rows_key_and_the_shape():
    assert questions_name(ROWS_KEY, SHAPE) == f"questions-{ROWS_KEY}-{SHAPE[:16]}.json"
    with pytest.raises(ValueError, match="not a rows key"):
        questions_name("rows-short", SHAPE)


def test_a_question_set_is_written_create_only_in_its_keys_directory_and_read_back(area):
    path, digest = _strict(area).record_questions(FP, rows=ROWS_KEY, shape=SHAPE,
                                                  questions=_questions("r1", "r0"))
    assert path == f"{answer_key(FP, 'H200', IMAGE)}/{questions_name(ROWS_KEY, SHAPE)}"
    assert digest == hashlib.sha256((area / path).read_bytes()).hexdigest()
    got = _answers(area, "check").question_set(FP, rows=ROWS_KEY, shape=SHAPE)
    assert got == QuestionSet(fingerprint=FP, gpu_kind="H200", image=IMAGE, rows=ROWS_KEY, shape=SHAPE,
                              questions=_questions("r0", "r1"))


def test_the_same_question_set_again_changes_nothing_and_another_one_is_refused(area):
    path, digest = _strict(area).record_questions(FP, rows=ROWS_KEY, shape=SHAPE, questions=_questions("r0", "r1"))
    again = _strict(area, writer="w2+oracle_fill+1")
    assert again.record_questions(FP, rows=ROWS_KEY, shape=SHAPE, questions=_questions("r1", "r0")) == (path, digest)
    with pytest.raises(QuestionSetConflict, match=rf"^{path} holds another question set") as refused:
        again.record_questions(FP, rows=ROWS_KEY, shape=SHAPE, questions=_questions("r0"))
    assert refused.value.path == path
    assert hashlib.sha256((area / path).read_bytes()).hexdigest() == digest


def test_a_question_set_that_cannot_be_written_is_raised(area):
    writer = _strict(area)
    area.chmod(0o500)
    try:
        with pytest.raises(AnswersNotWritten, match="could not write question set .*Permission denied"):
            writer.record_questions(FP, rows=ROWS_KEY, shape=SHAPE, questions=_questions("r0"))
    finally:
        area.chmod(0o700)


def test_no_question_set_reads_as_none(area):
    assert _answers(area, "check").question_set(FP, rows=ROWS_KEY, shape=SHAPE) is None


@pytest.mark.parametrize("field, value", [("fingerprint", "HFEngine_other"), ("gpu_kind", "B200"),
                                          ("image", "f" * 64), ("rows", "rows-fedcba9876543210"),
                                          ("shape", "c" * 64)])
def test_a_question_set_of_another_key_rows_or_shape_is_refused(area, field, value):
    fields = {"fingerprint": FP, "gpu_kind": "H200", "image": IMAGE, "rows": ROWS_KEY, "shape": SHAPE, field: value}
    planted = QuestionSet(**fields, questions=_questions("r0"))
    path = area / answer_key(FP, "H200", IMAGE) / questions_name(ROWS_KEY, SHAPE)
    path.parent.mkdir(parents=True)
    path.write_text(planted.model_dump_json())
    with pytest.raises(ValueError, match=rf"is the question set of another {field}"):
        _answers(area, "check").question_set(FP, rows=ROWS_KEY, shape=SHAPE)


def test_a_question_set_is_no_answers_file(area, caplog):
    _call(_strict(area), _Oracle(), [_stamped("r0")])
    _strict(area).record_questions(FP, rows=ROWS_KEY, shape=SHAPE, questions=_questions("r0"))
    reader = _answers(area, "check")
    with caplog.at_level(logging.WARNING):
        assert reader.unanswered(FP, _stamped("r0")) is False
    assert not caplog.records, caplog.text
    assert [k.rsplit("/", 1)[1] for k in reader.read] == ["w1+oracle_fill+1.000001.json"]
