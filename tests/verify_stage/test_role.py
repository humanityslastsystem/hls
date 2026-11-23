from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from hls.core.decode import Greedy
from hls.core.trace import Trace
from hls.spec.tenant import Tenant
from hls.store.digest import sha256_file
from hls.store.manifest import read_outputs_report
from hls.taskref import MeasuredPaths, TaskRef, blob_sha, read_task
from hls.trace.store import END_FILE, read_end, write_trace
from hls.verify.calibration import CALIBRATION_FILE
from hls.verify.role import (LIBRARIES, REFUSED_EXIT, REFUSED_PREFIX, VERIFY_FILE, CalibrationBlob, VerifyRecord,
                             calibration_blob, verify)
from tests.verify_stage.conftest import default_calibration
from tests.verify_stage.fixture import (LAZY_DISTRIBUTION, STACK, AskVerifier, GeneratorReference, _task,
                                        _workload, event, gate_trace, single_trace)

#: What the role writes for the gate task over ``gate_trace``: one directory per
#: case, holding ``verdict.json``, each replies file uncompressed and ``verify.json`` with its
#: libraries emptied, since those name the environment the test runs in. Its calibration block
#: names the fixture task's default calibration, at the task's current payload.
GOLDEN = Path(__file__).parent / "golden"


def _store_trace(trace: Trace, root: Path) -> Path:
    out = root / "trace"
    write_trace(trace, out, ending="drained", driver={"class": "fixture"})
    return out


def _role(checkout: Path, module: str, trace_dir: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run the role as a program, importing from *checkout* only, handed the default calibration of
    *module* when it has one, as the verify stage hands it the file
    the plan pinned. It reports its outputs in ``outputs.json`` beside *out*."""
    out.mkdir(parents=True, exist_ok=True)
    ref = out.parent / "task-ref.json"
    ref.write_text(TaskRef(module=module).model_dump_json())
    calibration = default_calibration(checkout, module)
    given = ["--calibration", str(calibration)] if calibration is not None else []
    env = {**os.environ, "PYTHONPATH": f"{checkout / 'src'}:{checkout}"}
    return subprocess.run([sys.executable, "-m", "hls.verify.role", "--task-ref", str(ref), "--trace", str(trace_dir),
                           "--out", str(out), "--outputs-out", str(out.parent / "outputs.json"), *given, *extra],
                          env=env, cwd=out.parent, capture_output=True, text=True, timeout=120)


def _replies(eval_dir: Path) -> dict[str, bytes]:
    """Each replies file's name and uncompressed bytes. Gzip stamps the write time into its
    header, so the compressed bytes of two runs differ."""
    return {p.name: gzip.decompress(p.read_bytes()) for p in sorted(eval_dir.glob("replies.*.jsonl.gz"))}


@pytest.mark.parametrize("errored", [False, True], ids=["graded", "failed-requests"])
def test_the_role_writes_the_old_runners_verdict_and_replies_byte_for_byte(checkout, tmp_path, errored):
    golden = GOLDEN / ("failed-requests" if errored else "graded")
    new = tmp_path / "new" / "out"
    done = _role(checkout, "vstage_gate.task", _store_trace(gate_trace(errored=errored), tmp_path / "store"), new)
    assert done.returncode == 0, done.stderr
    assert (new / "verdict.json").read_bytes() == (golden / "verdict.json").read_bytes()
    assert _replies(new) == {f"{p.name}.gz": p.read_bytes() for p in sorted(golden.glob("replies.*.jsonl"))}
    written = (new / VERIFY_FILE).read_bytes()
    record = VerifyRecord.model_validate_json(written)
    assert written == record.model_dump_json(indent=2).encode()
    assert record.model_copy(update={"libraries": {}}).model_dump_json(indent=2).encode() == (
        golden / VERIFY_FILE).read_bytes()
    verdict = json.loads((new / "verdict.json").read_text())
    assert verdict["aggregate_passed"] is False
    if errored:
        assert verdict["failure_mode"] == "failed_requests" and _replies(new) == {}
        assert VerifyRecord.model_validate_json((new / VERIFY_FILE).read_text()).engines == []
    else:
        # The pooled gate's diagnostics once, then an alias; the letter match per tenant.
        assert verdict["failure_mode"] == "correctness"
        assert verdict["diagnostics"]["2:TokenFaithfulnessVerifier"]["pooled_alias_of"] == 0
        assert sorted(_replies(new)) == ["replies.0.TokenFaithfulnessVerifier.jsonl.gz",
                                         "replies.1.MetricVerifier.jsonl.gz", "replies.3.MetricVerifier.jsonl.gz"]
        record = VerifyRecord.model_validate_json((new / VERIFY_FILE).read_text())
        assert record.eval_id == "terminal"
        assert [(e.role, e.engine, e.stack, e.stack_error) for e in record.engines] == [
            ("reference", "FakeTeacherEngine", STACK, None)]


@pytest.mark.parametrize("errored", [False, True], ids=["graded", "failed-requests"])
def test_the_role_reports_every_file_it_wrote_by_its_path_under_the_outputs_directory(checkout, tmp_path, errored):
    out = tmp_path / "role" / "out"
    done = _role(checkout, "vstage_gate.task", _store_trace(gate_trace(errored=errored), tmp_path), out)
    assert done.returncode == 0, done.stderr
    report = read_outputs_report(out.parent / "outputs.json")
    assert report is not None and sorted(report) == sorted(p.name for p in out.iterdir())
    assert {"verdict.json", VERIFY_FILE} <= set(report)
    for name, reported in report.items():
        assert (reported.sha256, reported.size) == sha256_file(out / name)


def test_one_verifier_loads_all_its_engines_together(checkout, tmp_path):
    out = tmp_path / "role" / "out"
    done = _role(checkout, "vstage_pair.task", _store_trace(single_trace(), tmp_path), out, "--eval-id", "e7")
    assert done.returncode == 0, done.stderr
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["aggregate_passed"] is True
    assert verdict["diagnostics"]["0:PairVerifier"] == {"generated": 2, "forced": 2}
    record = VerifyRecord.model_validate_json((out / VERIFY_FILE).read_text())
    assert record.eval_id == "e7"
    assert [(e.role, e.engine, e.stack, e.stack_error) for e in record.engines] == [
        ("reference", "FakeGenerator", None, None), ("reference1", "FakeForcer", None, None)]


def test_verify_json_names_engines_in_load_order(tmp_path):
    # One counter over every load of the verify: an engine evicted by the next verifier's and
    # loaded again takes the next name.
    first, second = GeneratorReference(decode=Greedy()), GeneratorReference(decode=Greedy())
    task = _task("verify-stage-relay", tenants=[
        Tenant(id=t, workload=_workload(), verifiers=[AskVerifier(reference=ref)])
        for t, ref in (("a", first), ("b", second), ("c", first))])
    trace = Trace(events=[event(t, "r0", "B", "B") for t in ("a", "b", "c")])
    record = asyncio.run(verify(task, trace, out=tmp_path, eval_id="terminal", answers=None))
    assert [(e.role, e.engine) for e in record.engines] == [
        ("reference", "FakeGenerator"), ("reference1", "FakeGenerator"), ("reference2", "FakeGenerator")]
    assert VerifyRecord.model_validate_json((tmp_path / VERIFY_FILE).read_text()).engines == record.engines


def test_a_stack_report_that_fails_is_recorded_and_the_verifier_runs_as_on_main(checkout, tmp_path):
    out = tmp_path / "role" / "out"
    done = _role(checkout, "vstage_pairstack.task", _store_trace(single_trace(), tmp_path), out)
    assert done.returncode == 0, done.stderr
    assert json.loads((out / "verdict.json").read_text())["aggregate_passed"] is True
    record = VerifyRecord.model_validate_json((out / VERIFY_FILE).read_text())
    assert [(e.engine, e.stack, e.stack_error) for e in record.engines] == [
        ("FakeGenerator", None, None), ("BrokenStackForcer", None, "RuntimeError: no stack on this engine")]


@pytest.mark.parametrize("module, trace, reason", [
    ("vstage_drift.task", gate_trace(), "CalibrationError: reference stack drifted from calibration"),
    ("vstage_silicon.task", gate_trace(), "CalibrationError: reference is serving on 'NVIDIA H100 80GB HBM3'"),
    ("vstage_nocal.task", gate_trace(), "CalibrationError: task has a calibration-requiring verifier but no "
                                         "calibration.json was given"),
    ("vstage_split.task", gate_trace(), "WiringError: 2 differently-configured TokenFaithfulnessVerifier"),
    ("vstage_metric.task", Trace(events=[]), "HarnessFault: verify: the test split selected zero items"),
], ids=["stack-drift", "wrong-silicon", "no-calibration", "split-gate", "empty-trace"])
def test_a_deterministic_refusal_exits_3_with_its_reason_on_one_line(checkout, tmp_path, module, trace, reason):
    out = tmp_path / "role" / "out"
    done = _role(checkout, module, _store_trace(trace, tmp_path), out)
    assert done.returncode == REFUSED_EXIT, done.stderr
    [line] = [x for x in done.stderr.splitlines() if x.startswith(REFUSED_PREFIX)]
    assert line.startswith(REFUSED_PREFIX + reason), line
    assert not (out / "verdict.json").exists()


@pytest.mark.parametrize("module, raised", [("vstage_badload.task", "CalibrationError"),
                                            ("vstage_badstop.task", "HarnessFault")], ids=["load", "stop"])
def test_an_engine_that_fails_to_load_or_stop_exits_1_whatever_it_raised(checkout, tmp_path, module, raised):
    # Loading and stopping an engine is the harness's work on its own GPUs, which a retry can
    # do again: never a refusal, even when the engine raises a refusal's type.
    out = tmp_path / "role" / "out"
    done = _role(checkout, module, _store_trace(single_trace(), tmp_path), out)
    assert done.returncode == 1, done.stderr
    assert REFUSED_PREFIX not in done.stderr and raised in done.stderr
    assert not (out / "verdict.json").exists()


def test_an_engine_the_teardown_fails_to_stop_exits_1_whatever_it_raised(checkout, tmp_path):
    # Both engines of the pair raise a refusal's type when stopped. The generator's stop fails
    # first and ends the verification core's releases, so the host's teardown stops the forcer,
    # and the forcer's failure is the one the role sees: the harness's, retried.
    out = tmp_path / "role" / "out"
    done = _role(checkout, "vstage_badstops.task", _store_trace(single_trace(), tmp_path), out)
    assert done.returncode == 1, done.stderr
    assert REFUSED_PREFIX not in done.stderr
    assert "StopFailingGenerator failed to stop: CalibrationError" in done.stderr
    assert "a reference engine failed to stop: HarnessFault" in done.stderr
    assert not (out / "verdict.json").exists()


def _installed(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def test_verify_json_records_the_libraries_the_verification_ran_with(checkout, tmp_path):
    trace_dir = _store_trace(single_trace(), tmp_path)
    libraries = {}
    for module in ("vstage_importing.task", "vstage_metric.task"):
        out = tmp_path / module / "out"
        done = _role(checkout, module, trace_dir, out)
        assert done.returncode == 0, done.stderr
        libraries[module] = VerifyRecord.model_validate_json((out / VERIFY_FILE).read_text()).libraries
    importing = libraries["vstage_importing.task"]
    assert importing["python"] == platform.python_version()
    # The libraries the core's numbers can depend on, installed here or not.
    assert {name: importing[name] for name in LIBRARIES} == {name: _installed(name) for name in LIBRARIES}
    assert set(LIBRARIES) == {"numpy", "transformers", "soxr", "torch"}
    # A library a verifier imported as it ran, by the distribution that installs it; a role whose
    # verifiers did not import it does not record it.
    assert importing[LAZY_DISTRIBUTION] == _installed(LAZY_DISTRIBUTION) is not None
    assert LAZY_DISTRIBUTION not in libraries["vstage_metric.task"]


def test_verify_json_records_the_calibration_blob_the_verifiers_were_given(checkout, tmp_path):
    """The record names the calibration file the role injected by the id git commits it under,
    with its kind and the workload it was measured for; a task none of whose verifiers needs one
    records none. Where the file was found is the plan's pin of it."""
    gate = tmp_path / "gate" / "out"
    done = _role(checkout, "vstage_gate.task", _store_trace(gate_trace(), tmp_path / "gate"), gate)
    assert done.returncode == 0, done.stderr
    path = default_calibration(checkout, "vstage_gate.task")
    written = json.loads(path.read_text())
    assert VerifyRecord.model_validate_json((gate / VERIFY_FILE).read_text()).calibration == CalibrationBlob(
        blob_sha=blob_sha(path), kind="flips", workload_fingerprint=written["workload_fingerprint"])
    metric = tmp_path / "metric" / "out"
    done = _role(checkout, "vstage_metric.task", _store_trace(single_trace(), tmp_path / "metric"), metric)
    assert done.returncode == 0, done.stderr
    assert VerifyRecord.model_validate_json((metric / VERIFY_FILE).read_text()).calibration is None


def test_the_record_names_the_calibration_the_load_parsed_and_reads_no_file(checkout, tmp_path, monkeypatch):
    """The blob the record names is the one the load read and gave the verifiers, whatever the file
    holds by the time the record is made; the record reads no file of its own."""
    ref = TaskRef(module="vstage_gate.task")
    given = tmp_path / CALIBRATION_FILE
    defaults = default_calibration(checkout, "vstage_gate.task")
    given.write_bytes(defaults.read_bytes())
    loaded = read_task(ref, calibration=True, files=MeasuredPaths(calibration=given))
    # The file now holds the same content in another layout, so other bytes.
    given.write_text(json.dumps(json.loads(defaults.read_text()), indent=3))
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail(f"read {self}") if self.name == CALIBRATION_FILE
                        else real(self))
    assert calibration_blob(loaded) == CalibrationBlob(
        blob_sha=loaded.calibration.blob, kind="flips",
        workload_fingerprint=loaded.calibration.value.workload_fingerprint)
    assert loaded.calibration.blob == blob_sha(defaults)


def test_the_roles_own_log_lines_reach_its_stderr(checkout, tmp_path):
    # Run as a program the role's module is ``__main__``; its own lines still reach the handler.
    done = _role(checkout, "vstage_metric.task", _store_trace(single_trace(), tmp_path), tmp_path / "role" / "out")
    assert done.returncode == 0, done.stderr
    assert "verified 2 events with 0 reference engines" in done.stderr


def test_an_out_directory_that_already_holds_a_file_is_refused_and_nothing_is_reported(checkout, tmp_path):
    out = tmp_path / "role" / "out"
    out.mkdir(parents=True)
    (out / "replies.9.Stale.jsonl.gz").write_bytes(b"")
    done = _role(checkout, "vstage_metric.task", _store_trace(single_trace(), tmp_path), out)
    assert done.returncode == 1 and "already holds ['replies.9.Stale.jsonl.gz']" in done.stderr
    assert sorted(p.name for p in out.iterdir()) == ["replies.9.Stale.jsonl.gz"]
    assert read_outputs_report(out.parent / "outputs.json") is None


def test_a_trace_chunk_that_does_not_match_its_end_exits_1(checkout, tmp_path):
    trace_dir = _store_trace(gate_trace(), tmp_path)
    chunk = trace_dir / read_end(trace_dir).chunks[0].name
    lines = gzip.decompress(chunk.read_bytes()).splitlines(keepends=True)
    chunk.write_bytes(gzip.compress(b"".join(lines[:-1])))
    out = tmp_path / "role" / "out"
    done = _role(checkout, "vstage_gate.task", trace_dir, out)
    assert done.returncode == 1
    assert "TraceIncomplete" in done.stderr and REFUSED_PREFIX not in done.stderr
    assert not (out / "verdict.json").exists()


def test_a_trace_without_its_end_exits_1(checkout, tmp_path):
    trace_dir = _store_trace(gate_trace(), tmp_path)
    (trace_dir / END_FILE).unlink()
    done = _role(checkout, "vstage_gate.task", trace_dir, tmp_path / "role" / "out")
    assert done.returncode == 1 and "TraceIncomplete" in done.stderr


def test_a_module_without_a_task_exits_1(checkout, tmp_path):
    done = _role(checkout, "tests.verify_stage.fixture", _store_trace(single_trace(), tmp_path),
                 tmp_path / "role" / "out")
    assert done.returncode == 1 and "declares no TASK" in done.stderr


# ---------------------------------------------------------------------------
# Reference answers on the store
# ---------------------------------------------------------------------------

IMAGE_A, IMAGE_B = "a" * 64, "b" * 64


def _answers(area: Path, writer: str, *, gpu_kind: str = "B200", image: str = IMAGE_A) -> tuple[str, ...]:
    return ("--answers", str(area), "--writer", writer, "--gpu-kind", gpu_kind, "--reference-image", image)


def test_a_caching_reference_answers_once_and_is_then_served_without_being_deployed(checkout, tmp_path):
    area, trace_dir = tmp_path / "store" / "oracle" / "verify-stage-cached", _store_trace(single_trace(3), tmp_path)
    first = _role(checkout, "vstage_cached.task", trace_dir, tmp_path / "one" / "out", *_answers(area, "g1+verify+1"))
    assert first.returncode == 0, first.stderr
    one = VerifyRecord.model_validate_json((tmp_path / "one" / "out" / VERIFY_FILE).read_text())
    assert [e.engine for e in one.engines] == ["FakeGenerator"]
    assert one.answers_read == {} and len(one.answers_written) == 1
    [(key, digest)] = one.answers_written.items()
    assert key.endswith("/g1+verify+1.000001.json")
    assert hashlib.sha256((area / key).read_bytes()).hexdigest() == digest
    # The answers are kept in the area only: the role's outputs are the files of its out directory.
    report = read_outputs_report(tmp_path / "one" / "outputs.json")
    assert report is not None and sorted(report) == sorted(p.name for p in (tmp_path / "one" / "out").iterdir())

    second = _role(checkout, "vstage_cached.task", trace_dir, tmp_path / "two" / "out", *_answers(area, "g2+verify+1"))
    assert second.returncode == 0, second.stderr
    two = VerifyRecord.model_validate_json((tmp_path / "two" / "out" / VERIFY_FILE).read_text())
    assert two.engines == [], "every answer was on the store, so the reference was never deployed"
    assert two.answers_read == one.answers_written and two.answers_written == {}
    assert ((tmp_path / "two" / "out" / "verdict.json").read_bytes()
            == (tmp_path / "one" / "out" / "verdict.json").read_bytes())


def test_a_later_turns_prompt_never_reaches_the_area_whatever_the_task_declares(checkout, tmp_path):
    """vstage_cached declares one turn per session, so the plan admits its caching reference; a
    trace whose prompts are stamped at turn two, carrying the server's first reply, is answered
    by the deployed reference and none of it is written to the area."""
    def later(ev):
        meta = {**ev.input.metadata, "turn_index": 1,
                "messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "the server's reply"},
                             {"role": "user", "content": "Q?"}]}
        return ev.model_copy(update={"input": ev.input.model_copy(update={"metadata": meta})})

    area = tmp_path / "store" / "oracle" / "verify-stage-cached"
    trace_dir = _store_trace(Trace(events=[later(ev) for ev in single_trace(3).events]), tmp_path)
    for n in (1, 2):
        out = tmp_path / f"run{n}" / "out"
        done = _role(checkout, "vstage_cached.task", trace_dir, out, *_answers(area, f"g{n}+verify+1"))
        assert done.returncode == 0, done.stderr
        record = VerifyRecord.model_validate_json((out / VERIFY_FILE).read_text())
        assert [e.engine for e in record.engines] == ["FakeGenerator"]
        assert record.answers_read == {} and record.answers_written == {}
        assert json.loads((out / "verdict.json").read_text())["aggregate_passed"] is True
    assert not area.exists() or not any(p.is_file() for p in area.rglob("*"))


#: The name an entry is planted under in a key's directory, one the role never writes.
PLANTED = "g0+verify+1.000001.json"


def _plant_entry(case: str, key_dir: Path, good: Path, outside: Path) -> str:
    """Put an entry of kind *case* into *key_dir*, beside the good answers file *good*; return
    the name a warning about it names."""
    if case == "backup-name":
        name = f"{good.name}~"
        (key_dir / name).write_bytes(good.read_bytes())
        return name
    if case == "unreadable-key":
        key_dir.chmod(0)
        return key_dir.name
    entry = key_dir / PLANTED
    if case == "sparse-huge":
        with open(entry, "wb") as f:
            f.truncate(1 << 40)
    elif case == "fifo":
        os.mkfifo(entry)
    elif case == "link-inside":
        entry.symlink_to(good)
    elif case == "link-outside":
        outside.write_bytes(good.read_bytes())
        entry.symlink_to(outside)
    else:
        entry.mkdir()
    return PLANTED


@pytest.mark.parametrize("case, reason", [
    ("backup-name", "no store segment"), ("unreadable-key", "cannot be listed"), ("sparse-huge", "bytes, more than"),
    ("fifo", "not a regular file"), ("link-inside", "not a regular file"), ("link-outside", "not a regular file"),
    ("subdirectory", "not a regular file")])
def test_an_entry_of_a_key_that_is_no_answers_file_is_skipped_and_the_verify_succeeds(checkout, tmp_path, case,
                                                                                     reason):
    """The area is an optimisation, never a gate: whatever lies in a key's directory, the verify
    writes its verdict, and an entry it cannot read as an answers file is skipped with a warning
    naming it. A failure here would fail every verify of the key, and each retry with it."""
    area, trace_dir = tmp_path / "store" / "oracle" / "verify-stage-cached", _store_trace(single_trace(3), tmp_path)
    assert _role(checkout, "vstage_cached.task", trace_dir, tmp_path / "one" / "out",
                 *_answers(area, "g1+verify+1")).returncode == 0
    one = VerifyRecord.model_validate_json((tmp_path / "one" / "out" / VERIFY_FILE).read_text())
    [good] = one.answers_written
    key_dir = (area / good).parent
    named = _plant_entry(case, key_dir, area / good, tmp_path / "outside.json")
    try:
        done = _role(checkout, "vstage_cached.task", trace_dir, tmp_path / "two" / "out",
                     *_answers(area, "g2+verify+1"))
    finally:
        key_dir.chmod(0o700)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "two" / "out" / "verdict.json").is_file()
    assert any("skipping" in line and named in line and reason in line for line in done.stderr.splitlines()), \
        done.stderr
    two = VerifyRecord.model_validate_json((tmp_path / "two" / "out" / VERIFY_FILE).read_text())
    if case == "unreadable-key":
        # Nothing of the key could be read, so the reference answered; its write failed and was logged.
        assert ([e.engine for e in two.engines], two.answers_read, two.answers_written) == (["FakeGenerator"], {}, {})
        assert "could not write reference answers" in done.stderr
    else:
        assert (two.engines, two.answers_read) == ([], one.answers_written)


@pytest.mark.parametrize("other", [{"image": IMAGE_B}, {"gpu_kind": "H200"}], ids=["image", "gpu-kind"])
def test_answers_from_another_image_or_silicon_are_not_served(checkout, tmp_path, other):
    area, trace_dir = tmp_path / "store" / "oracle" / "verify-stage-cached", _store_trace(single_trace(), tmp_path)
    assert _role(checkout, "vstage_cached.task", trace_dir, tmp_path / "one" / "out",
                 *_answers(area, "g1+verify+1")).returncode == 0
    done = _role(checkout, "vstage_cached.task", trace_dir, tmp_path / "two" / "out",
                 *_answers(area, "g2+verify+1", **other))
    assert done.returncode == 0, done.stderr
    record = VerifyRecord.model_validate_json((tmp_path / "two" / "out" / VERIFY_FILE).read_text())
    assert [e.engine for e in record.engines] == ["FakeGenerator"] and record.answers_read == {}
    assert len(record.answers_written) == 1


def test_without_an_answers_area_a_caching_reference_answers_every_time(checkout, tmp_path):
    done = _role(checkout, "vstage_cached.task", _store_trace(single_trace(), tmp_path), tmp_path / "role" / "out")
    assert done.returncode == 0, done.stderr
    record = VerifyRecord.model_validate_json((tmp_path / "role" / "out" / VERIFY_FILE).read_text())
    assert [e.engine for e in record.engines] == ["FakeGenerator"]
    assert record.answers_read == {} and record.answers_written == {}


@pytest.mark.parametrize("given", [("--answers", "AREA"), ("--writer", "w", "--gpu-kind", "B200"),
                                   ("--answers", "AREA", "--writer", "w", "--gpu-kind", "B200")],
                         ids=["area-only", "no-area", "no-image"])
def test_the_answer_options_come_together_or_not_at_all(checkout, tmp_path, given):
    area = str(tmp_path / "area")
    done = _role(checkout, "vstage_cached.task", _store_trace(single_trace(), tmp_path), tmp_path / "role" / "out",
                 *(area if v == "AREA" else v for v in given))
    assert done.returncode == 2 and "come together or not at all" in done.stderr
    assert not (tmp_path / "area").exists()
