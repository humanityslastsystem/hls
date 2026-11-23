"""The oracle fill role, run as a program against the fixture checkout: it answers a caching
reference's test split into the answers area, chunk by chunk, so every later grading serves
those answers and a fill that died resumes."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest

import hls.verify.fill as fill_module
from hls.core.capabilities import Generate
from hls.core.decode import Greedy
from hls.core.engine import Engine
from hls.core.trace import Trace
from hls.prompt import JinjaPromptTemplate
from hls.provision.rows import tenant_mount_path
from hls.verify.answers import prompt_digest
from hls.spec.datasets import Split
from hls.spec.drivers import stamp_request, user_message
from hls.spec.tenant import Tenant
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, ShortOutputCap, Workload
from hls.stages.oracle_fill import ROWS_ROOT
from hls.store.digest import sha256_file
from hls.store.manifest import read_outputs_report
from hls.store.posix import PosixStore
from hls.taskref import TaskRef, load_task
from hls.verify.answers import QuestionSet, StoreAnswers, answer_key, questions_name
from hls.verify.base import WiringError
from hls.verify.fill import (FILL_FILE, FillRecord, FillRefused, caching_references, corpus_prompts, fill,
                             question_scopes, question_shape)
from hls.verify.inprocess import EngineHost
from hls.verify.role import REFUSED_EXIT, REFUSED_PREFIX, VERIFY_FILE, VerifyRecord
from tests.verify_stage import fixture
from tests.verify_stage.fixture import AskVerifier, FakeGenerator, GeneratorReference, event
from tests.verify_stage.test_role import _role, _store_trace

IMAGE = "a" * 64
#: The key of the rows entry the fill role is told it asks the test rows of.
ROWS_KEY = "rows-0123456789abcdef"
ROWS = [{"q": "1"}, {"q": "2"}, {"q": "3"}]
#: Rows of two turns each, in ``fill_task(multi_turn=True)``'s dataset shape.
MULTI_TURN_ROWS = [{"q": "1", "ctx": "c", "qs": ["a", "b"], "as": ["x", "y"]}]


def _rows(root: Path, rows: list[dict] = ROWS, *, tenant: str = "t0") -> Path:
    """A data root holding *tenant*'s test rows as the oracle role's container sees them."""
    split = root / tenant_mount_path(tenant, "test").removeprefix(f"{ROWS_ROOT}/")
    split.mkdir(parents=True)
    (split / "rows.json").write_text(json.dumps(rows))
    return root


def _fill(checkout: Path, module: str, rows: Path, area: Path, out: Path, *extra: str,
          writer: str = "w1+oracle_fill+1") -> subprocess.CompletedProcess:
    """Run the fill role as a program, importing from *checkout* only. It reports its record in
    ``outputs.json`` beside *out*."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ref = out.parent / "task-ref.json"
    ref.write_text(TaskRef(module=module).model_dump_json())
    env = {**os.environ, "PYTHONPATH": f"{checkout / 'src'}:{checkout}"}
    return subprocess.run([sys.executable, "-m", "hls.verify.fill", "--task-ref", str(ref), "--rows", str(rows),
                           "--rows-key", ROWS_KEY, "--answers", str(area), "--writer", writer, "--gpu-kind", "B200",
                           "--reference-image", IMAGE, "--out", str(out),
                           "--outputs-out", str(out.parent / "outputs.json"), *extra],
                          env=env, cwd=out.parent, capture_output=True, text=True, timeout=120)


def _record(out: Path) -> FillRecord:
    return FillRecord.model_validate_json(out.read_text())


def _question_set_path(area: Path, task) -> Path:
    """Where a completed fill of *task*'s one caching reference over ``ROWS_KEY`` records its
    question set."""
    [(fingerprint, tenants)] = question_scopes(task).items()
    return area / answer_key(fingerprint, "B200", IMAGE) / questions_name(ROWS_KEY, question_shape(task, tenants))


def test_a_completed_fill_records_every_question_corpus_prompts_asks_and_a_second_finds_it_unchanged(checkout,
                                                                                                    tmp_path):
    """The question set is what a plan-time check reads in place of the rows: exactly the
    (tenant, request id, prompt digest) of each prompt ``corpus_prompts`` builds from those
    rows, which are the prompts the drive sends."""
    rows, area = _rows(tmp_path / "data"), tmp_path / "area"
    first = _fill(checkout, "vstage_fill.task", rows, area, tmp_path / "one" / FILL_FILE, "--chunk", "2")
    assert first.returncode == 0, first.stderr
    task = load_task(TaskRef(module="vstage_fill.task"))
    path = _question_set_path(area, task)
    recorded = QuestionSet.model_validate_json(path.read_bytes())
    assert {(q.tenant, q.request_id, q.digest) for q in recorded.questions} == {
        (p.metadata["tenant_id"], p.request_id, prompt_digest(p)) for p in corpus_prompts(task, rows)}
    assert len(recorded.questions) == 3 and recorded.rows == ROWS_KEY
    one = _record(tmp_path / "one" / FILL_FILE)
    assert one.questions == {path.relative_to(area).as_posix(): sha256_file(path)[0]}
    # A second fill of the same rows finds the same set there and records it again.
    second = _fill(checkout, "vstage_fill.task", rows, area, tmp_path / "two" / FILL_FILE, writer="w2+oracle_fill+1")
    assert second.returncode == 0, second.stderr
    assert _record(tmp_path / "two" / FILL_FILE).questions == one.questions


def test_a_fill_limited_to_its_first_rows_records_no_question_set(checkout, tmp_path):
    rows, area = _rows(tmp_path / "data"), tmp_path / "area"
    out = tmp_path / "out" / FILL_FILE
    done = _fill(checkout, "vstage_fill.task", rows, area, out, "--limit", "1")
    assert done.returncode == 0, done.stderr
    assert _record(out).questions == {}
    assert not _question_set_path(area, load_task(TaskRef(module="vstage_fill.task"))).exists()


def test_a_fill_that_finds_another_question_set_under_its_name_refuses_naming_the_file(checkout, tmp_path):
    # Same rows and the same declared shaping, other questions: the digests moved under code
    # the name does not cover, and the stale set must not stand for these rows. A retry would
    # find it again, so the role refuses.
    rows, area = _rows(tmp_path / "data"), tmp_path / "area"
    task = load_task(TaskRef(module="vstage_fill.task"))
    path = _question_set_path(area, task)
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    out = tmp_path / "out" / FILL_FILE
    done = _fill(checkout, "vstage_fill.task", rows, area, out)
    assert done.returncode == REFUSED_EXIT, done.stderr
    [line] = [x for x in done.stderr.splitlines() if x.startswith(REFUSED_PREFIX)]
    stored = f"oracle/{task.id}/{path.relative_to(area).as_posix()}"
    assert line == (f"{REFUSED_PREFIX}QuestionSetConflict: {stored} holds another question set than this fill asked: "
                    "the prompts of these rows digest otherwise now; remove that file and fill again")
    assert not out.exists() and path.read_text() == "{}"


def test_the_rows_root_is_where_the_mounts_put_every_tenants_rows():
    assert tenant_mount_path("t0", "test") == f"{ROWS_ROOT}/tenants/t0/test"


def test_a_fill_answers_the_split_and_a_second_one_loads_nothing(checkout, tmp_path):
    rows, area = _rows(tmp_path / "data"), tmp_path / "area"
    first = _fill(checkout, "vstage_fill.task", rows, area, tmp_path / "one" / FILL_FILE, "--chunk", "2")
    assert first.returncode == 0, first.stderr
    one = _record(tmp_path / "one" / FILL_FILE)
    [ref] = one.references
    assert (ref.engine, ref.prompts, ref.answered, ref.served) == ("FakeGenerator", 3, 3, 0)
    # Three rows in chunks of two: two calls that answered, two files.
    assert len(one.answers_written) == 2 and one.answers_read == {}
    assert "loading FakeGenerator" in first.stderr
    # Each chunk after the first misses whole while the area holds the chunks before it, which
    # is no unreachable cache.
    assert "none match" not in first.stderr

    second = _fill(checkout, "vstage_fill.task", rows, area, tmp_path / "two" / FILL_FILE,
                   writer="w2+oracle_fill+1")
    assert second.returncode == 0, second.stderr
    two = _record(tmp_path / "two" / FILL_FILE)
    [ref] = two.references
    assert (ref.answered, ref.served) == (0, 3)
    assert two.answers_read == one.answers_written and two.answers_written == {}
    assert "loading FakeGenerator" not in second.stderr, "a fill whose answers are all there loads no engine"


def test_the_fill_role_reports_its_record_for_the_stages_manifest(checkout, tmp_path):
    out = tmp_path / "out" / FILL_FILE
    done = _fill(checkout, "vstage_fill.task", _rows(tmp_path / "data"), tmp_path / "area", out)
    assert done.returncode == 0, done.stderr
    # Keyed by its path under the outputs directory as the role sees it, the file's parent.
    report = read_outputs_report(out.parent / "outputs.json")
    assert report is not None and sorted(report) == [FILL_FILE]
    assert (report[FILL_FILE].sha256, report[FILL_FILE].size) == sha256_file(out)


def test_a_fill_that_died_keeps_its_finished_chunks_and_resumes(checkout, tmp_path):
    # The flaky generator answers one call per process and dies on the next.
    rows, area = _rows(tmp_path / "data"), tmp_path / "area"
    questions = _question_set_path(area, load_task(TaskRef(module="vstage_flaky.task")))
    done, unfinished = [], []
    for i in range(3):
        done.append(_fill(checkout, "vstage_flaky.task", rows, area, tmp_path / f"run{i}" / FILL_FILE, "--chunk",
                          "1", writer=f"w{i}+oracle_fill+1"))
        unfinished.append(not questions.exists())
    assert [d.returncode for d in done] == [1, 1, 0], [d.stderr[-400:] for d in done]
    assert "the flaky generator died" in done[0].stderr
    assert not (tmp_path / "run0" / FILL_FILE).exists() and not (tmp_path / "run0" / "outputs.json").exists()
    # A fill that died records no question set; the one that banked the last answer does, last.
    assert unfinished == [True, True, False]
    files = sorted(p.name for key in area.iterdir() for p in key.iterdir())
    assert files == [questions.name, "w0+oracle_fill+1.000001.json", "w1+oracle_fill+1.000001.json",
                     "w2+oracle_fill+1.000001.json"]
    [ref] = _record(tmp_path / "run2" / FILL_FILE).references
    assert (ref.answered, ref.served) == (1, 2)


def test_a_grading_serves_the_filled_answers_without_deploying_its_reference(checkout, tmp_path):
    rows, area = _rows(tmp_path / "data"), tmp_path / "area"
    assert _fill(checkout, "vstage_fill.task", rows, area, tmp_path / "fill" / FILL_FILE).returncode == 0
    # The drive sends each row stamped as the fill shaped it.
    task = load_task(TaskRef(module="vstage_fill.task"))
    prompts = corpus_prompts(task, rows)
    trace = _store_trace(Trace(events=[event("t0", p.request_id, "B", "B").model_copy(update={"input": p})
                                       for p in prompts]), tmp_path)
    out = tmp_path / "verify" / "out"
    done = _role(checkout, "vstage_fill.task", trace, out, "--answers", str(area), "--writer", "g1+verify+1",
                 "--gpu-kind", "B200", "--reference-image", IMAGE)
    assert done.returncode == 0, done.stderr
    record = VerifyRecord.model_validate_json((out / VERIFY_FILE).read_text())
    assert record.engines == [] and record.answers_written == {}
    assert sorted(record.answers_read) == sorted(_record(tmp_path / "fill" / FILL_FILE).answers_written)
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["aggregate_passed"] is True


def test_the_prompts_digest_as_the_drive_stamps_them(checkout, tmp_path):
    task = load_task(TaskRef(module="vstage_fill.task"))
    [first, *_] = corpus_prompts(task, _rows(tmp_path / "data"))
    item = next(iter(task.data_source.dataset.items_at(tmp_path / "data" / "tenants" / "t0", Split.TEST)))
    sent = stamp_request(item.turns[0], [user_message(item.turns[0])], tenant_id="t0", session_id=item.request_id,
                         turn_idx=0, max_tokens=task.effective_tenants()[0].workload.output_length_cap.max_tokens())
    assert prompt_digest(first) == prompt_digest(sent)


def test_the_rows_the_drive_removes_are_not_asked(checkout, tmp_path):
    # The drive drops a task's removal list before it serves (``remove_listed_rows``), so no
    # grading asks about those rows.
    task = load_task(TaskRef(module="vstage_fill.task"))
    task = task.model_copy(update={"data_source": task.data_source.model_copy(update={"removed_rows": ("2",)})})
    assert [p.request_id for p in corpus_prompts(task, _rows(tmp_path / "data"))] == ["1", "3"]


def test_prompts_are_built_for_the_tenants_named_or_for_the_whole_task(tmp_path):
    task = fixture.mixed_fill_task()
    rows = _rows(_rows(tmp_path / "data", tenant="asr"), MULTI_TURN_ROWS, tenant="chat")
    asked = corpus_prompts(task, rows, tenants={"asr"})
    assert [(p.metadata["tenant_id"], p.request_id) for p in asked] == [("asr", "1"), ("asr", "2"), ("asr", "3")]
    with pytest.raises(FillRefused, match="chat/1 has 2 turns"):
        corpus_prompts(task, rows)
    assert corpus_prompts(task, rows, tenants=set()) == []


def test_a_tenant_with_no_data_source_is_asked_nothing(tmp_path):
    """As the drive serves it nothing (``load_mounted_sessions``), whether the whole task's
    prompts are built or its tenants are named."""
    source = fixture.fill_task().data_source
    ask = AskVerifier(reference=GeneratorReference(decode=Greedy(), cache=True))
    task = fixture._task("idle", tenants=[
        Tenant(id="asr", data_source=source, workload=fixture._workload(), verifiers=[ask]),
        Tenant(id="idle", data_source=None, workload=fixture._workload(), verifiers=[ask])])
    rows = _rows(tmp_path / "data", tenant="asr")
    for asked in (corpus_prompts(task, rows), corpus_prompts(task, rows, tenants={"asr", "idle"})):
        assert [(p.metadata["tenant_id"], p.request_id) for p in asked] == [("asr", "1"), ("asr", "2"), ("asr", "3")]


def test_only_caching_corpus_references_are_filled(checkout):
    [ref] = caching_references(load_task(TaskRef(module="vstage_fill.task")))
    assert (ref.capability, type(ref.reference), ref.tenants) == (Generate, GeneratorReference, ("t0",))
    assert caching_references(load_task(TaskRef(module="vstage_uncached.task"))) == []
    assert caching_references(load_task(TaskRef(module="vstage_gate.task"))) == []


@pytest.mark.parametrize("module, rows, reason", [
    ("vstage_multi", MULTI_TURN_ROWS, "FillRefused: t0/1 has 2 turns"),
    ("vstage_uncached", ROWS, "FillRefused: verify-stage-fill has no reference that caches"),
], ids=["multi-turn", "no-caching-reference"])
def test_a_task_the_fill_cannot_fill_exits_3_with_its_reason(checkout, tmp_path, module, rows, reason):
    out = tmp_path / "out" / FILL_FILE
    done = _fill(checkout, f"{module}.task", _rows(tmp_path / "data", rows), tmp_path / "area", out)
    assert done.returncode == REFUSED_EXIT, done.stderr
    [line] = [x for x in done.stderr.splitlines() if x.startswith(REFUSED_PREFIX)]
    assert line.startswith(REFUSED_PREFIX + reason), line
    assert not out.exists() and not (tmp_path / "area").exists()


def test_a_multi_turn_tenant_without_a_caching_reference_does_not_block_the_fill(checkout, tmp_path):
    """The fill builds prompts only for the tenants its caching references are asked about: a
    tenant of two turns per session that no reference caches for is neither read nor refused."""
    rows = _rows(_rows(tmp_path / "data", tenant="asr"), MULTI_TURN_ROWS, tenant="chat")
    out = tmp_path / "out" / FILL_FILE
    done = _fill(checkout, "vstage_fill_mixed.task", rows, tmp_path / "area", out)
    assert done.returncode == 0, done.stderr
    [ref] = _record(out).references
    assert (ref.prompts, ref.answered) == (3, 3)


def test_missing_rows_exit_1(checkout, tmp_path):
    done = _fill(checkout, "vstage_fill.task", tmp_path / "data", tmp_path / "area", tmp_path / "out" / FILL_FILE)
    assert done.returncode == 1 and "test split not mounted" in done.stderr


def test_a_limit_fills_only_the_first_rows(checkout, tmp_path):
    out = tmp_path / "out" / FILL_FILE
    done = _fill(checkout, "vstage_fill.task", _rows(tmp_path / "data"), tmp_path / "area", out, "--limit", "1")
    assert done.returncode == 0, done.stderr
    [ref] = _record(out).references
    assert (ref.prompts, ref.answered) == (1, 1)


def test_answers_that_cannot_be_written_fail_the_fill_and_a_verify_still_succeeds(checkout, tmp_path):
    """The fill exists to bank its answers: one it could not write ends the role with exit 1,
    which the stage retries as infra, naming the write's error, and no record is reported. A
    verify of the same read-only area still succeeds: there the area is never a gate."""
    rows, area = _rows(tmp_path / "data"), tmp_path / "area"
    area.mkdir()
    area.chmod(0o555)
    try:
        out = tmp_path / "fill" / FILL_FILE
        done = _fill(checkout, "vstage_fill.task", rows, area, out, "--chunk", "1")
        task = load_task(TaskRef(module="vstage_fill.task"))
        trace = _store_trace(Trace(events=[event("t0", p.request_id, "B", "B").model_copy(update={"input": p})
                                           for p in corpus_prompts(task, rows)]), tmp_path)
        verify_out = tmp_path / "verify" / "out"
        verified = _role(checkout, "vstage_fill.task", trace, verify_out, "--answers", str(area), "--writer",
                         "g1+verify+1", "--gpu-kind", "B200", "--reference-image", IMAGE)
    finally:
        area.chmod(0o755)
    assert done.returncode == 1, done.stderr
    assert "AnswersNotWritten: could not write reference answers" in done.stderr
    assert "Permission denied" in done.stderr
    assert not out.exists() and not (out.parent / "outputs.json").exists()
    assert list(area.iterdir()) == []
    assert verified.returncode == 0, verified.stderr
    record = VerifyRecord.model_validate_json((verify_out / VERIFY_FILE).read_text())
    assert [e.engine for e in record.engines] == ["FakeGenerator"] and record.answers_written == {}
    assert json.loads((verify_out / "verdict.json").read_text())["aggregate_passed"] is True


@pytest.mark.parametrize("option", [["--chunk", "0"], ["--limit", "0"]], ids=["chunk", "limit"])
def test_a_chunk_or_limit_below_one_is_refused_as_a_usage_error(checkout, tmp_path, option):
    out = tmp_path / "out" / FILL_FILE
    done = _fill(checkout, "vstage_fill.task", _rows(tmp_path / "data"), tmp_path / "area", out, *option)
    assert done.returncode == 2 and "--chunk and --limit must be positive" in done.stderr


# ---------------------------------------------------------------------------
# The library, in this process
# ---------------------------------------------------------------------------


def _answers(tmp_path: Path) -> StoreAnswers:
    return StoreAnswers(PosixStore(tmp_path / "area"), gpu_kind="B200", image=IMAGE, writer="w1+oracle_fill+1")


class _Another(GeneratorReference):
    """The fake generator's reference under another name, so under another fingerprint."""


def test_the_question_shape_covers_what_corpus_prompts_reads_of_the_task_besides_the_rows():
    task = fixture.fill_task()
    source = task.data_source
    shape = question_shape(task, ("t0",))
    assert question_shape(fixture.fill_task(), ("t0",)) == shape
    capped = task.model_copy(update={"workload": task.workload.model_copy(
        update={"output_length_cap": ShortOutputCap(cap_tokens=16)})})
    removed = task.model_copy(update={"data_source": source.model_copy(update={"removed_rows": ("2",)})})
    rendered = task.model_copy(update={"data_source": source.model_copy(update={"dataset": source.dataset.model_copy(
        update={"prompt": JinjaPromptTemplate(template="Q: {{q}}")})})})
    for changed in (capped, removed, rendered):
        assert question_shape(changed, ("t0",)) != shape
    # Disclosure beside the dev rows changes no prompt.
    disclosed = task.model_copy(update={"data_source": source.model_copy(update={"dev_reference": {"a.json": "{}"}})})
    assert question_shape(disclosed, ("t0",)) == shape


@pytest.mark.parametrize("seam", ["stamp_request", "prompt_digest"])
def test_a_change_to_how_every_prompt_is_stamped_or_digested_changes_the_shape(monkeypatch, seam):
    task = fixture.fill_task()
    shape = question_shape(task, ("t0",))
    shared = getattr(fill_module, seam)
    if seam == "stamp_request":
        def changed(turn, messages, **kw):
            stamped = shared(turn, messages, **kw)
            return stamped.model_copy(update={"metadata": {**stamped.metadata, "max_tokens": 7}})
    else:
        def changed(prompt):
            return shared(prompt)[::-1]
    monkeypatch.setattr(fill_module, seam, changed)
    assert question_shape(task, ("t0",)) != shape


def test_references_sharing_a_fingerprint_record_one_question_set_over_all_their_tenants(tmp_path):
    source = fixture.fill_task().data_source
    task = fixture._task("shared", tenants=[
        Tenant(id=t, data_source=source, workload=fixture._workload(),
               verifiers=[AskVerifier(reference=GeneratorReference(decode=Greedy(), cache=True))])
        for t in ("speech", "chat")])
    [(_, scope)] = question_scopes(task).items()
    assert scope == ("speech", "chat")
    rows = _rows(_rows(tmp_path / "data", tenant="speech"), tenant="chat")
    answers = StoreAnswers(PosixStore(tmp_path / "area"), gpu_kind="B200", image=IMAGE, writer="w1+oracle_fill+1",
                           strict=True)
    record = asyncio.run(fill(task, corpus_prompts(task, rows), answers, chunk=2, rows_key=ROWS_KEY))
    [path] = record.questions
    recorded = QuestionSet.model_validate_json((tmp_path / "area" / path).read_bytes())
    assert [(q.tenant, q.request_id) for q in recorded.questions] == [
        ("chat", "1"), ("chat", "2"), ("chat", "3"), ("speech", "1"), ("speech", "2"), ("speech", "3")]


def test_each_reference_is_released_before_the_next_is_loaded(tmp_path, monkeypatch):
    """Two caching references never share the role's GPU: the first is released after its
    last chunk, before the second loads."""
    held: list[int] = []

    class Counting(EngineHost):
        async def acquire(self, engine):
            held.append(self.outstanding)
            return await super().acquire(engine)

    monkeypatch.setattr(fill_module, "EngineHost", Counting)
    task = fixture._task("two", workload=fixture._workload(), data_source=fixture.fill_task().data_source,
                         verifiers=[fixture.AskVerifier(reference=GeneratorReference(decode=Greedy(), cache=True)),
                                    fixture.AskVerifier(reference=GeneratorReference(decode=Greedy(), cache=True))])
    prompts = corpus_prompts(task, _rows(tmp_path / "data"))
    record = asyncio.run(fill(task, prompts, _answers(tmp_path), chunk=2))
    assert [r.answered for r in record.references] == [3, 0], "the second reference shares the first's answers"
    assert held == [0]
    # Two fingerprints: another reference class asks the same engine class.
    other = fixture._task("two", workload=fixture._workload(), data_source=fixture.fill_task().data_source,
                          verifiers=[fixture.AskVerifier(reference=GeneratorReference(decode=Greedy(), cache=True)),
                                     fixture.AskVerifier(reference=_Another(decode=Greedy(), cache=True))])
    held.clear()
    record = asyncio.run(fill(other, corpus_prompts(other, tmp_path / "data"), _answers(tmp_path / "b"), chunk=2))
    assert [r.answered for r in record.references] == [3, 3] and held == [0, 0]


def test_a_fill_after_a_prompt_change_repairs_the_key(tmp_path):
    """A task edit that the reference's fingerprint does not cover, here the output cap stamped
    on every prompt, changes each prompt's digest under the same answer key. The next fill
    answers the new questions, and every fill after it is served them."""
    def task(cap: int):
        workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                            output_length_cap=ShortOutputCap(cap_tokens=cap), slo=NoSLO())
        return fixture._task("cap", workload=workload, data_source=fixture.fill_task().data_source,
                             verifiers=[fixture.AskVerifier(reference=GeneratorReference(decode=Greedy(), cache=True))])

    rows = _rows(tmp_path / "data")
    old, new = task(16), task(32)
    [ref] = caching_references(old)
    assert ref.reference.fingerprint(old) == ref.reference.fingerprint(new)

    def run(spec, writer: str) -> tuple[int, int]:
        answers = StoreAnswers(PosixStore(tmp_path / "area"), gpu_kind="B200", image=IMAGE, writer=writer)
        [done] = asyncio.run(fill(spec, corpus_prompts(spec, rows), answers, chunk=2)).references
        return done.answered, done.served

    assert run(old, "w1+oracle_fill+1") == (3, 0)
    assert run(new, "w2+oracle_fill+1") == (3, 0)
    assert run(new, "w3+oracle_fill+1") == (0, 3)


class _Recording(FakeGenerator):
    """The fake generator, noting each prompt it answers as (its class's name, the prompt's
    tenant)."""

    asked: ClassVar[list[tuple[str, str]]] = []

    async def generate(self, prompts):
        _Recording.asked.extend((type(self).__name__, p.metadata["tenant_id"]) for p in prompts)
        return await super().generate(prompts)


class _SpeechGenerator(_Recording):
    pass


class _ChatGenerator(_Recording):
    pass


class _SpeechReference(GeneratorReference):
    engine_cls: ClassVar[type[Engine]] = _SpeechGenerator


class _ChatReference(GeneratorReference):
    engine_cls: ClassVar[type[Engine]] = _ChatGenerator


class _PooledAsk(AskVerifier):
    """The ask verifier, pooled: verify hands it every tenant's events at once."""

    pooled: ClassVar[bool] = True


def _two_tenants(speech_verifier: type[AskVerifier] = AskVerifier):
    """Tenants speech and chat over the fill corpus, each verified against a caching reference of
    its own; speech's verifier is *speech_verifier*."""
    source = fixture.fill_task().data_source
    tenants = [Tenant(id="speech", data_source=source, workload=fixture._workload(),
                      verifiers=[speech_verifier(reference=_SpeechReference(decode=Greedy(), cache=True))]),
               Tenant(id="chat", data_source=source, workload=fixture._workload(),
                      verifiers=[AskVerifier(reference=_ChatReference(decode=Greedy(), cache=True))])]
    return fixture._task("two-tenants", tenants=tenants)


def test_each_reference_is_asked_only_the_rows_of_the_tenants_whose_verifiers_hold_it(tmp_path):
    """As verify asks it: a reference answers its verifier's tenant's events, so the fill never
    asks it another tenant's rows, which no grading reads and which may not even be its
    modality."""
    task = _two_tenants()
    rows = _rows(_rows(tmp_path / "data", tenant="speech"), tenant="chat")
    assert [(type(r.reference).__name__, r.tenants) for r in caching_references(task)] == [
        ("_SpeechReference", ("speech",)), ("_ChatReference", ("chat",))]
    _Recording.asked.clear()
    record = asyncio.run(fill(task, corpus_prompts(task, rows), _answers(tmp_path), chunk=2, rows_key=ROWS_KEY))
    assert [(r.engine, r.prompts, r.answered) for r in record.references] == [
        ("_SpeechGenerator", 3, 3), ("_ChatGenerator", 3, 3)]
    assert sorted(set(_Recording.asked)) == [("_ChatGenerator", "chat"), ("_SpeechGenerator", "speech")]
    # Each reference's question set holds its own tenant's questions only.
    sets = [QuestionSet.model_validate_json((tmp_path / "area" / path).read_bytes()) for path in record.questions]
    assert sorted(sorted({q.tenant for q in s.questions}) for s in sets) == [["chat"], ["speech"]]
    # A pooled verifier is handed every tenant's events, so its reference is asked every tenant's rows.
    pooled = _two_tenants(_PooledAsk)
    assert [r.tenants for r in caching_references(pooled)] == [("speech", "chat"), ("chat",)]


def test_an_engine_that_is_not_its_references_declared_class_is_refused(tmp_path):
    class Mismatched(GeneratorReference):
        engine_cls: ClassVar[type[Engine]] = fixture.FlakyGenerator

        def build_engine(self, task):
            return FakeGenerator(model_spec=task.model, decode=self.decode, gpu_kind=task.hardware.accelerator.kind)

    task = fixture._task("bad", workload=fixture._workload(), data_source=fixture.fill_task().data_source,
                         verifiers=[fixture.AskVerifier(reference=Mismatched(decode=Greedy(), cache=True))])
    with pytest.raises(WiringError, match="returned FakeGenerator, not the declared engine_cls FlakyGenerator"):
        asyncio.run(fill(task, corpus_prompts(task, _rows(tmp_path / "data")), _answers(tmp_path)))
