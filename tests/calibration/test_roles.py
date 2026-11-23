"""The calibration roles (``hls.verify.calibration_roles``) in this process. Each phase is a
function of ``hls.calibrate`` over the files its stage hands the role, and the role writes what
that function writes, reported for the stage's manifest. The programs run with the fake engines
of ``tests.calibration.engines``; the measurements are compared with those functions' over the
unit tests' doubles (``tests.unit.test_calibrate``, ``tests.unit.test_calibrate_report``)."""

from __future__ import annotations

import asyncio
import gzip
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hls.calibrate as calibrate
from hls.calibrate import (MIN_SERVINGS, CaptureStore, RowIndex, Serving, capture_items, load_sessions, score_draw,
                           serving_seed_of)
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from hls.spec.datasets import Split
from hls.store.digest import sha256_file
from hls.store.engine_cache import EngineCacheRecord
from hls.store.manifest import read_outputs_report
from hls.taskref import TaskRef
from hls.verify import calibration_roles as roles
from hls.verify import inprocess
from hls.verify.calibration import (CALIBRATION_FILE, PreflightResult, StackInfo, StopReason, Tolerances,
                                    read_calibration, workload_fingerprint)
from tests.calibration import engines, fake_roles, fixture
from tests.unit.test_calibrate import (_FakeMember, _FakePinned, _make_deploy, _penalized_task, _tf_item,
                                      _token_task)
from tests.unit.test_calibrate_report import _reply_payload, _ReportMember, _ReportReference

#: The per-token tail tolerance whose exposure target is 40,000 tokens.
MU_40K = math.log(40) * 1e6 / 40_000


def _main(tmp_path: Path, role: str, *args: str, module: str = "cal_flips.task",
          campaign: roles.CampaignSettings | None = None) -> tuple[int, Path, Path]:
    """Run *role* with *args* over *module*'s task and *campaign*; its exit status, its outputs
    directory and its report."""
    task_ref, settings = tmp_path / "task-ref.json", tmp_path / "campaign.json"
    task_ref.write_text(TaskRef(module=module).model_dump_json())
    settings.write_text((campaign or fixture.campaign()).model_dump_json())
    out, report = tmp_path / "out", tmp_path / "outputs.json"
    code = roles.main([role, "--task-ref", str(task_ref), "--campaign", str(settings), *args, "--out", str(out),
                       "--outputs-out", str(report)])
    return code, out, report


def _sessions(task, rows: Path) -> list[tuple]:
    return load_sessions(task, rows, Split.TEST)


def _refusal(capsys) -> str:
    [line] = [x for x in capsys.readouterr().err.splitlines() if x.startswith(roles.REFUSED_PREFIX)]
    return line.removeprefix(roles.REFUSED_PREFIX)


# ---- capture ----------------------------------------------------------------

def test_the_capture_role_drives_a_serving_with_the_old_function(tmp_path, login, fake_engines, rows, monkeypatch):
    calls = []
    drive = calibrate.drive_serving

    async def spy(serving, **kw):
        calls.append((serving, kw))
        return await drive(serving, **kw)

    monkeypatch.setattr(calibrate, "drive_serving", spy)
    task = fixture.flips_task()
    campaign = fixture.campaign(sessions_per_draw=3, drive_timeout_sec=900.0)
    seed = serving_seed_of(task.seed, 3)
    code, out, report = _main(tmp_path, "capture", "--rows", str(rows), "--engine", "vllm", "--draw", "3",
                              "--seed", str(seed), "--view", str(tmp_path / "view"), campaign=campaign)
    assert code == 0
    [(serving, kw)] = calls
    assert serving == Serving("vllm", 3, seed)
    assert (kw["sessions_per_draw"], kw["request_timeout_sec"], kw["drive_timeout_sec"]) == (
        3, campaign.request_timeout_sec, 900.0)
    assert kw["measure"].with_reports is False and kw["sessions"] == _sessions(task, rows)
    # The capture is the capture store's file, copied whole out of the role's view.
    name = roles.capture_file(campaign.workload_fingerprint, "vllm", 3)
    view = CaptureStore(tmp_path / "view", campaign.workload_fingerprint)
    assert (out / name).read_bytes() == view.capture_path("vllm", 3).read_bytes()
    reported = read_outputs_report(report)
    assert sorted(reported) == [name] and reported[name].sha256 == sha256_file(out / name)[0]
    with gzip.open(out / name, "rt") as fh:
        assert json.loads(fh.readline()) == {"engine": "vllm", "draw": 3, "seed": seed,
                                             "serving_version": engines.MEMBER_VERSION}
    items, _, _ = view.read_capture("vllm", 3, rows=RowIndex.from_sessions(_sessions(task, rows)))
    assert len(items) == len(fixture.TEST_ROWS)


def test_a_capture_handed_another_seed_refuses(tmp_path, login, fake_engines, rows, capsys):
    code, out, _ = _main(tmp_path, "capture", "--rows", str(rows), "--engine", "vllm", "--draw", "3", "--seed", "7",
                         "--view", str(tmp_path / "view"))
    seed = serving_seed_of(fixture.flips_task().seed, 3)
    assert code == roles.REFUSED_EXIT
    assert _refusal(capsys) == f"InputMismatch: serving 3 is clocked with seed {seed}, not 7"
    assert not (out / "capture").exists()


def test_a_serving_that_captured_nothing_fails_for_another_attempt(tmp_path, login, fake_engines, rows, monkeypatch,
                                                                   capsys):
    async def faults(self, loaded, **kw):
        raise RuntimeError("the member lost its GPU")

    monkeypatch.setattr(engines.FakeMember, "serve_traffic", faults)
    seed = serving_seed_of(fixture.flips_task().seed, 0)
    code, out, report = _main(tmp_path, "capture", "--rows", str(rows), "--engine", "vllm", "--draw", "0",
                              "--seed", str(seed), "--view", str(tmp_path / "view"))
    assert code == 1
    err = capsys.readouterr().err
    assert "ServingFailed: serving 0 of pool member 'vllm' captured nothing: the member lost its GPU" in err
    assert not report.exists()


def test_a_task_whose_fingerprint_is_not_the_plans_refuses(tmp_path, login, fake_engines, rows, capsys):
    campaign = fixture.campaign(workload_fingerprint="f" * 64)
    seed = serving_seed_of(fixture.flips_task().seed, 0)
    code, _, _ = _main(tmp_path, "capture", "--rows", str(rows), "--engine", "vllm", "--draw", "0", "--seed",
                       str(seed), "--view", str(tmp_path / "view"), campaign=campaign)
    assert code == roles.REFUSED_EXIT
    found = workload_fingerprint(fixture.flips_task())
    assert _refusal(capsys) == (f"InputMismatch: cal_flips.task has workload fingerprint {found}; the plan keys its "
                                f"captures by {'f' * 64}")


@pytest.mark.parametrize("response", [False, True], ids=["no-response", "an-error-beside-its-response"])
def test_a_serving_with_a_faulted_request_captures_nothing_and_says_which(tmp_path, login, fake_engines, rows,
                                                                       monkeypatch, capsys, response):
    # Dropping the faulted request would measure a smaller and easier workload than the one
    # offered, so the serving is refused whole and taken again.
    serve, faulted = engines.FakeMember.serve_traffic, []

    async def one_fault(self, loaded, **kw):
        trace = await serve(self, loaded, **kw)
        first = trace.events[0]
        first.error = "transport: boom"
        if not response:
            first.response, first.t_recv = None, None
        faulted.append(first.request_id)
        return trace

    monkeypatch.setattr(engines.FakeMember, "serve_traffic", one_fault)
    code, out, report = _main(tmp_path, "capture", "--rows", str(rows), "--engine", "vllm", "--draw", "0",
                              "--seed", str(serving_seed_of(fixture.flips_task().seed, 0)), "--view", str(tmp_path / "view"))
    assert code == 1
    [rid] = faulted
    err = capsys.readouterr().err
    assert (f"ServingFailed: serving 0 of pool member 'vllm' captured nothing: pool member 'vllm' faulted 1 of "
            f"{len(fixture.TEST_ROWS)} requests on serving 0 (first: {rid} -- transport: boom)") in err
    assert not report.exists() and not (out / "capture").exists()


def test_a_capture_whose_engine_fails_to_release_fails(tmp_path, login, fake_engines, rows, monkeypatch, capsys):
    # A role's outputs stand only when every engine it loaded was released: the capture was written
    # before the member failed to stop, but the role reports nothing, and another attempt drives the
    # serving again on an engine that ends cleanly.
    def stuck(self):
        raise RuntimeError("the member's engine process would not stop")

    monkeypatch.setattr(engines.FakeMember, "_unload", stuck)
    code, out, report = _main(tmp_path, "capture", "--rows", str(rows), "--engine", "vllm", "--draw", "0",
                              "--seed", str(serving_seed_of(fixture.flips_task().seed, 0)), "--view", str(tmp_path / "view"))
    assert code == 1
    assert not report.exists() and not (out / "capture").exists()
    assert ("ServingFailed: serving 0 of pool member 'vllm' captured, but its engine did not release: the member's "
            "engine process would not stop") in capsys.readouterr().err


def test_a_member_on_other_silicon_captures_nothing(tmp_path, login, fake_engines, rows, monkeypatch, capsys):
    # Each allocation is placed on its own, so each member checks its device before it serves; a
    # serving on the wrong one is taken again.
    monkeypatch.setitem(engines.STACK, "device_name", "NVIDIA H100")
    code, _, report = _main(tmp_path, "capture", "--rows", str(rows), "--engine", "vllm", "--draw", "0",
                            "--seed", str(serving_seed_of(fixture.flips_task().seed, 0)), "--view", str(tmp_path / "view"))
    assert code == 1 and not report.exists()
    assert ("ServingFailed: serving 0 of pool member 'vllm' captured nothing: calibration allocation 'vllm serving 0' "
            "is running on 'NVIDIA H100', but the task declares hardware.accelerator.kind='B200'") in (
        capsys.readouterr().err)


def test_a_capture_of_a_member_the_campaign_does_not_hold_refuses(tmp_path, login, fake_engines, rows, capsys):
    code, _, _ = _main(tmp_path, "capture", "--rows", str(rows), "--engine", "sglang", "--draw", "0", "--seed",
                       str(serving_seed_of(fixture.flips_task().seed, 0)), "--view", str(tmp_path / "view"),
                       campaign=fixture.campaign(engines=["vllm"]))
    assert code == roles.REFUSED_EXIT
    assert _refusal(capsys) == "InputMismatch: sglang is not one of the campaign's pool members ['vllm']"


# ---- warm -------------------------------------------------------------------

def test_the_warm_role_loads_its_member_once_and_writes_nothing(tmp_path, login, fake_engines, monkeypatch):
    loads, releases = [], []
    load, unload = engines.FakeMember._load, engines.FakeMember._unload

    def counted_load(self):
        loads.append(type(self).__name__)
        load(self)

    def counted_unload(self):
        releases.append(type(self).__name__)
        unload(self)

    monkeypatch.setattr(engines.FakeMember, "_load", counted_load)
    monkeypatch.setattr(engines.FakeMember, "_unload", counted_unload)
    code, out, report = _main(tmp_path, "warm", "--engine", "sglang", "--load-sec", "60")
    assert code == 0 and loads == releases == ["FakeMember"]
    assert read_outputs_report(report) == {} and list(out.iterdir()) == []


def test_a_warm_whose_member_fails_to_release_fails(tmp_path, login, fake_engines, monkeypatch, capsys):
    # The cache its load built is whole, but a role succeeds only when its engine was released, so
    # the stage publishes nothing a run that ended in an unknown state built.
    def stuck(self):
        raise RuntimeError("the member's engine process would not stop")

    monkeypatch.setattr(engines.FakeMember, "_unload", stuck)
    code, _, report = _main(tmp_path, "warm", "--engine", "sglang", "--load-sec", "60")
    assert code == 1 and not report.exists()
    assert "RuntimeError: the member's engine process would not stop" in capsys.readouterr().err


def test_a_warm_whose_member_never_finishes_releasing_ends_within_its_bound(tmp_path, checkout):
    # Its load leaves an object whose finalizer waits for ever, as a ZMQ context's can when nothing
    # closes its sockets: the engine host's bound on freeing what the load
    # held ends the role, as infra another attempt may pass, with every thread's stack in its log.
    cache = tmp_path / "engine-cache"
    cache.mkdir()
    (cache / engines.RELEASE_HANG).write_text("")
    task_ref, settings = tmp_path / "task-ref.json", tmp_path / "campaign.json"
    task_ref.write_text(TaskRef(module="cal_flips.task").model_dump_json())
    settings.write_text(fixture.campaign().model_dump_json())
    out, report = tmp_path / "out", tmp_path / "outputs.json"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(checkout / "src"), str(checkout)]),
           "FLASHINFER_WORKSPACE_BASE": str(cache), "TMPDIR": str(tmp_path), fake_roles.RELEASE_SEC_ENV: "1.5"}
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-m", "tests.calibration.fake_roles", "warm", "--task-ref", str(task_ref),
                           "--campaign", str(settings), "--engine", "vllm", "--load-sec", "60", "--out", str(out),
                           "--outputs-out", str(report)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == roles.ENGINE_FAULT_EXIT, done.stderr
    assert time.monotonic() - started < 60
    assert done.stderr.rstrip("\n").splitlines()[-1] == (
        f"{roles.ENGINE_FAULT_PREFIX}EngineReleaseTimeout: freeing what vllm-warm's load held did not return within "
        "1.5 s")
    assert "(most recent call first)" in done.stderr and "__del__" in done.stderr
    assert not report.exists()


def test_a_member_warmed_on_other_silicon_refuses(tmp_path, login, fake_engines, monkeypatch, capsys):
    # Its cache would be published under the task's GPU kind.
    monkeypatch.setitem(engines.STACK, "device_name", "NVIDIA H100")
    code, _, _ = _main(tmp_path, "warm", "--engine", "vllm", "--load-sec", "60")
    assert code == roles.REFUSED_EXIT
    refusal = _refusal(capsys)
    assert refusal.startswith("CalibrationError: calibration allocation 'vllm warm' is running on 'NVIDIA H100'")
    # The fix is the site's: the GPU kind its Slurm requests name.
    assert "site.gpu_gres" in refusal and "substrate" not in refusal and "uv run" not in refusal


def test_a_warm_of_a_member_the_campaign_does_not_hold_refuses(tmp_path, login, fake_engines, capsys):
    code, _, _ = _main(tmp_path, "warm", "--engine", "sglang", "--load-sec", "60",
                       campaign=fixture.campaign(engines=["vllm"]))
    assert code == roles.REFUSED_EXIT
    assert _refusal(capsys) == "InputMismatch: sglang is not one of the campaign's pool members ['vllm']"


# ---- the engine host ----------------------------------------------------------

def test_a_roles_deploy_seam_loads_one_engine_once(fake_engines):
    # Every engine of a role loads on the one engine host its deploy seam holds, which refuses an
    # engine it already holds and releases every engine it holds when the block ends.
    async def go():
        async with roles.in_process(0, load_sec=60) as deploy:
            engine = engines.FakeMember()
            await deploy("vllm-0", engine)
            with pytest.raises(inprocess.PlacementError, match="FakeMember is already acquired"):
                await deploy("vllm-1", engine)
            assert engines.MEMBER_UNLOADS == []

    asyncio.run(go())
    # Released once when the block ended, though nothing called its release.
    assert engines.MEMBER_LOADS == engines.MEMBER_UNLOADS == [None]


@pytest.mark.parametrize("role", ["warm", "capture"])
def test_a_member_loads_on_every_gpu_its_role_sees(tmp_path, login, fake_engines, rows, monkeypatch, role):
    # A serving stack spreads over every GPU its process sees: the host places it on none of its
    # own, and never reads the role's GPU count for it, which could start CUDA before its workers.
    reads = []

    def two() -> int:
        reads.append(2)
        return 2

    monkeypatch.setattr(inprocess, "_visible_gpus", two)
    if role == "warm":
        args = ("--engine", "vllm", "--load-sec", "60")
    else:
        args = ("--rows", str(rows), "--engine", "vllm", "--draw", "0", "--seed",
                str(serving_seed_of(fixture.flips_task().seed, 0)), "--view", str(tmp_path / "view"))
    code, _, _ = _main(tmp_path, role, *args)
    assert code == 0
    assert engines.MEMBER_LOADS == [None] and reads == []


# ---- instrument -------------------------------------------------------------

def test_the_instrument_places_the_anchor_on_the_second_gpu(tmp_path, login, fake_engines):
    code, out, report = _main(tmp_path, "instrument", "--gpus", "2")
    assert code == 0
    # The pinned bf16 reference on GPU 0, then the fp32 anchor beside it on GPU 1.
    assert fake_engines == [("bfloat16", 0), ("float32", 1)]
    record = roles.InstrumentRecord.model_validate_json((out / roles.INSTRUMENT_FILE).read_bytes())
    assert record.preflight is not None and record.preflight.passed
    assert record.stack.device_name == engines.STACK["device_name"]
    assert sorted(read_outputs_report(report)) == [roles.INSTRUMENT_FILE]


def test_the_sampling_gates_instrument_loads_the_reference_alone(tmp_path, login, fake_engines):
    code, out, _ = _main(tmp_path, "instrument", "--gpus", "1", module="cal_report.task",
                         campaign=fixture.campaign(fixture.report_task()))
    assert code == 0 and fake_engines == [("bfloat16", 0)]
    assert roles.InstrumentRecord.model_validate_json((out / roles.INSTRUMENT_FILE).read_bytes()).preflight is None


def test_an_instrument_on_other_silicon_refuses(tmp_path, login, fake_engines, monkeypatch, capsys):
    monkeypatch.setitem(engines.STACK, "device_name", "NVIDIA H100")
    code, out, _ = _main(tmp_path, "instrument", "--gpus", "2")
    assert code == roles.REFUSED_EXIT and "declared accelerator" in _refusal(capsys)
    assert not (out / roles.INSTRUMENT_FILE).exists()


# ---- score --------------------------------------------------------------------

def _captured(task, rows: Path, store: CaptureStore, member, draw: int, *, with_reports: bool) -> Serving:
    """Serving *draw* of vllm by the unit tests' *member* double, captured into *store*."""
    seed = serving_seed_of(task.seed, draw)
    trace = asyncio.run(member.serve_traffic(_sessions(task, rows), seed=seed,
                                             request_timeout_sec=REQUEST_TIMEOUT_SEC))
    store.write_capture("vllm", draw, seed, "0.0.0", capture_items(trace, engine="vllm", draw=draw,
                                                                  with_reports=with_reports))
    return Serving("vllm", draw, seed)


@pytest.mark.parametrize("kind", ["flips", "report"])
def test_the_score_role_scores_with_the_old_measure(tmp_path, login, rows, kind):
    task = fixture.flips_task() if kind == "flips" else fixture.report_task()
    member = _FakeMember() if kind == "flips" else _ReportMember(replies=3, tokens=4)
    store = CaptureStore(tmp_path / "role", workload_fingerprint(task))
    serving = _captured(task, rows, store, member, 0, with_reports=kind == "report")
    old = CaptureStore(tmp_path / "old", workload_fingerprint(task))
    old.capture_path("vllm", 0).parent.mkdir(parents=True)
    old.capture_path("vllm", 0).symlink_to(store.capture_path("vllm", 0))

    def reference():
        if kind == "report":
            return _ReportReference()
        pinned = _FakePinned(0.10)
        pinned._preflight_done = True  # this reference scores servings, not the preflight's probes
        return pinned

    index = RowIndex.from_sessions(_sessions(task, rows))
    stats = asyncio.run(roles.score(task, [serving], kind=kind, share=1, store=store, rows=index,
                                    deploy=_make_deploy(reference())))
    measure = (calibrate.flip_measure() if kind == "flips"
               else calibrate.report_measure(calibrate.report_verifier(task), task.max_context_len))
    result = asyncio.run(calibrate.score_share([serving], 1, task=task, deploy=_make_deploy(reference()), store=old,
                                               rows=index, measure=measure))
    assert result.error is None and list(result.stats) == stats
    # Gzip stamps the write time into its header, so the two files' contents are compared.
    assert gzip.decompress(store.replies_path("vllm", 0).read_bytes()) == gzip.decompress(
        old.replies_path("vllm", 0).read_bytes())
    assert type(stats[0]) is (calibrate.DrawStats if kind == "flips" else calibrate.ReportDrawStats)


def _scoring(pinned: _FakePinned) -> _FakePinned:
    """*pinned*, scoring servings from its first call: the preflight's probes are the
    instrument's, not a score role's."""
    pinned._preflight_done = True
    return pinned


def test_a_serving_that_measured_nothing_refuses(tmp_path, login, rows):
    task = fixture.flips_task()
    store = CaptureStore(tmp_path / "role", workload_fingerprint(task))
    serving = _captured(task, rows, store, _FakeMember(tokens=0), 0, with_reports=False)
    with pytest.raises(calibrate.CalibrationError, match="produced no tokens at all"):
        asyncio.run(roles.score(task, [serving], kind="flips", share=1, store=store,
                                rows=RowIndex.from_sessions(_sessions(task, rows)),
                                deploy=_make_deploy(_scoring(_FakePinned(0.10)))))


def test_a_share_the_old_measure_refuses_is_refused_not_retried(tmp_path, login, rows):
    # A campaign ends when its scoring raises CalibrationError and no share survives:
    # another attempt reads the same captures through the same reference. Here the reference
    # answers each reply with the preflight's three gaps, which the measure refuses.
    task = fixture.flips_task()
    store = CaptureStore(tmp_path / "role", workload_fingerprint(task))
    serving = _captured(task, rows, store, _FakeMember(), 0, with_reports=False)
    with pytest.raises(calibrate.CalibrationError, match="malformed gap list"):
        asyncio.run(roles.score(task, [serving], kind="flips", share=1, store=store,
                                rows=RowIndex.from_sessions(_sessions(task, rows)),
                                deploy=_make_deploy(_FakePinned(0.10))))


def test_a_share_that_failed_otherwise_fails_for_another_attempt(tmp_path, login, rows):
    class _Lost(_FakePinned):
        async def teacher_force(self, items):
            raise RuntimeError("the reference lost its GPU")

    task = fixture.flips_task()
    store = CaptureStore(tmp_path / "role", workload_fingerprint(task))
    serving = _captured(task, rows, store, _FakeMember(), 0, with_reports=False)
    with pytest.raises(roles.ServingFailed, match="share 1 scored 0 of its 1 servings: the reference lost its GPU"):
        asyncio.run(roles.score(task, [serving], kind="flips", share=1, store=store,
                                rows=RowIndex.from_sessions(_sessions(task, rows)),
                                deploy=_make_deploy(_scoring(_Lost(0.10)))))


class _LengthMember(_ReportMember):
    """The report member, whose every reply says it stopped for length short of its cap."""

    async def serve_traffic(self, sessions, **kw):
        trace = await super().serve_traffic(sessions, **kw)
        for ev in trace.events:
            ev.response = ArtifactResponse(output=Envelope(request_id=ev.request_id, contents=[
                _reply_payload(self.tokens, first_id=self.first_id, finish_reason="length")]))
        return trace


@pytest.mark.parametrize(("window", "violations"), [(16, 0), (17, 2)])
def test_the_score_role_reads_a_replys_ending_against_the_tasks_window(tmp_path, login, rows, window, violations):
    # A reply that stopped for length short of its cap keeps its contract only when the context
    # ran out: the reference's 12 prompt tokens and the reply's 4 fill a window of 16, not one of
    # 17. The task's window is the one its members served at; the checkpoint's own is 8,192.
    task = fixture.report_task().model_copy(update={"max_context_len": window})
    store = CaptureStore(tmp_path / "role", workload_fingerprint(task))
    serving = _captured(task, rows, store, _LengthMember(replies=2, tokens=4), 0, with_reports=True)
    [stats] = asyncio.run(roles.score(task, [serving], kind="report", share=1, store=store,
                                      rows=RowIndex.from_sessions(_sessions(task, rows)),
                                      deploy=_make_deploy(_ReportReference())))
    assert stats.n_violations == violations


def test_a_scorer_on_other_silicon_refuses(tmp_path, login, rows):
    task = fixture.flips_task()
    store = CaptureStore(tmp_path / "role", workload_fingerprint(task))
    serving = _captured(task, rows, store, _FakeMember(), 0, with_reports=False)
    pinned = _scoring(_FakePinned(0.10, stack={**engines.STACK, "device_name": "NVIDIA H100"}))
    with pytest.raises(calibrate.CalibrationError, match="calibration allocation 'reference scorer score-1' is running "
                                                         "on 'NVIDIA H100'"):
        asyncio.run(roles.score(task, [serving], kind="flips", share=1, store=store,
                                rows=RowIndex.from_sessions(_sessions(task, rows)), deploy=_make_deploy(pinned)))


def _published_captures(tmp_path: Path, task, rows: Path, fingerprint: str, draws: tuple[int, ...]) -> list[str]:
    """Captures of vllm's servings *draws* by the fake member, as capture roles published them,
    and the score role's ``--capture`` arguments naming them."""
    published = CaptureStore(tmp_path / "published", fingerprint)
    args = []
    for draw in draws:
        trace = asyncio.run(engines.FakeMember().serve_traffic(
            _sessions(task, rows), seed=serving_seed_of(task.seed, draw), request_timeout_sec=REQUEST_TIMEOUT_SEC))
        published.write_capture("vllm", draw, serving_seed_of(task.seed, draw), engines.MEMBER_VERSION,
                                capture_items(trace, engine="vllm", draw=draw))
        path = published.capture_path("vllm", draw)
        args += ["--capture", "vllm", str(draw), sha256_file(path)[0], str(path)]
    return args


def test_the_score_role_reports_each_servings_replies_and_its_record(tmp_path, login, fake_engines, rows):
    task, campaign = fixture.flips_task(), fixture.campaign()
    fingerprint = campaign.workload_fingerprint
    args = _published_captures(tmp_path, task, rows, fingerprint, (0, 2))
    code, out, report = _main(tmp_path, "score", "--rows", str(rows), "--gpus", "1", "--share", "3", *args,
                              "--view", str(tmp_path / "view"), campaign=campaign)
    assert code == 0
    tokens = len(fixture.TEST_ROWS) * engines.TOKENS_PER_REPLY
    scored = roles.ScoredServings.model_validate_json((out / roles.SCORED_FILE).read_bytes())
    assert [(s.engine, s.draw, s.n_tokens, s.n_replies, s.serving_version) for s in scored.servings] == [
        ("vllm", d, tokens, len(fixture.TEST_ROWS), engines.MEMBER_VERSION) for d in (0, 2)]
    names = [roles.replies_file(fingerprint, "vllm", d) for d in (0, 2)]
    assert sorted(read_outputs_report(report)) == sorted([*names, roles.SCORED_FILE])
    stats = CaptureStore(out / "replies", fingerprint).read_scored("vllm", 2)
    assert (stats.n_flips, stats.n_tokens) == (len(fixture.TEST_ROWS) * engines.FLIPS, tokens)
    # The captures were read where their stage put them: the view links them, and no copy of
    # them leaves the role.
    view = CaptureStore(tmp_path / "view", fingerprint).capture_path("vllm", 0)
    assert view.is_symlink() and view.resolve() == CaptureStore(tmp_path / "published", fingerprint).capture_path(
        "vllm", 0).resolve()
    assert not (out / "capture").exists()


def test_a_capture_that_is_not_its_manifests_refuses(tmp_path, login, fake_engines, rows, capsys):
    task, campaign = fixture.flips_task(), fixture.campaign()
    args = _published_captures(tmp_path, task, rows, campaign.workload_fingerprint, (0,))
    args[3] = "0" * 64
    code, _, _ = _main(tmp_path, "score", "--rows", str(rows), "--gpus", "1", "--share", "1", *args,
                       "--view", str(tmp_path / "view"), campaign=campaign)
    assert code == roles.REFUSED_EXIT
    assert _refusal(capsys).startswith(f"InputMismatch: {args[4]} is not the file its stage recorded")


@pytest.mark.parametrize(("capture", "refusal"), [
    pytest.param(lambda args: args + args, "InputMismatch: serving 0 of vllm is handed twice", id="twice"),
    pytest.param(lambda args: ["--capture", "sglang", *args[2:]],
                 "InputMismatch: sglang is not one of the campaign's pool members ['vllm']", id="stranger"),
])
def test_a_share_handed_captures_the_plan_did_not_deal_it_refuses(tmp_path, login, fake_engines, rows, capsys,
                                                                   capture, refusal):
    campaign = fixture.campaign(engines=["vllm"])
    args = capture(_published_captures(tmp_path, fixture.flips_task(), rows, campaign.workload_fingerprint, (0,)))
    code, out, _ = _main(tmp_path, "score", "--rows", str(rows), "--gpus", "1", "--share", "1", *args,
                         "--view", str(tmp_path / "view"), campaign=campaign)
    assert code == roles.REFUSED_EXIT and _refusal(capsys) == refusal
    assert not (out / roles.SCORED_FILE).exists()


def test_a_share_whose_capture_holds_a_reply_past_its_cap_is_refused_and_reports_nothing(tmp_path, login, fake_engines,
                                                                                        rows, capsys):
    # A token-level campaign ends on a pool member that ignored a request's cap: the score role
    # refuses, which its stage does not retry, and reports no scored serving, so no aggregate has
    # the records to assemble a candidate from.
    task = fixture.capped_task()
    campaign = fixture.campaign(task)
    seed = serving_seed_of(task.seed, 0)
    trace = asyncio.run(engines.FakeMember().serve_traffic(_sessions(task, rows), seed=seed,
                                                           request_timeout_sec=REQUEST_TIMEOUT_SEC))
    long = trace.events[1]
    long.response = ArtifactResponse(output=Envelope(request_id=long.request_id, contents=[
        TextPayload(text="x", tokens=list(range(fixture.CAP_TOKENS + 1)), finish_reason="length")]))
    published = CaptureStore(tmp_path / "published", campaign.workload_fingerprint)
    published.write_capture("vllm", 0, seed, engines.MEMBER_VERSION, capture_items(trace, engine="vllm", draw=0))
    path = published.capture_path("vllm", 0)
    code, out, report = _main(tmp_path, "score", "--rows", str(rows), "--gpus", "1", "--share", "1",
                              "--capture", "vllm", "0", sha256_file(path)[0], str(path),
                              "--view", str(tmp_path / "view"), module="cal_capped.task", campaign=campaign)
    assert code == roles.REFUSED_EXIT
    assert _refusal(capsys) == (
        f"CalibrationError: pool engine 'vllm' served {fixture.CAP_TOKENS + 1} tokens for request "
        f"{long.request_id!r} of tenant 't0' on serving 0, past the {fixture.CAP_TOKENS}-token cap the harness "
        "stamped on it; the member did not honor the cap, so the serving holds text no graded trace is measured on. "
        "Fix the stack and re-run.")
    assert not report.exists() and not (out / roles.SCORED_FILE).exists() and not (out / "replies").exists()
    assert not CaptureStore(tmp_path / "view", campaign.workload_fingerprint).replies_path("vllm", 0).exists()


# ---- aggregate ------------------------------------------------------------------

def _scored(tmp_path: Path, task, *, servings: int = MIN_SERVINGS, first_gap: float = 0.1) -> CaptureStore:
    """vllm's servings 0 to *servings* - 1, scored as score roles publish them: 10,000 tokens each,
    20 of them flips, the first of *first_gap* nats and the others of 0.1."""
    store = CaptureStore(tmp_path / "replies", workload_fingerprint(task))
    for d in range(servings):
        store.write_scored(score_draw("vllm", d, serving_seed_of(task.seed, d), "0.0.0", [_tf_item(10_000)],
                                      [[first_gap] + [0.1] * 19 + [0.0] * 9_980]))
    return store


def _member_cache(draw: int) -> EngineCacheRecord:
    """Serving *draw*'s engine cache as its capture stage recorded it: serving 0 started cold,
    every later one was seeded from the entry its owner left."""
    return EngineCacheRecord(key="B200-0123456789abcdef-fedcba9876543210",
                                   seeded_from=None if draw == 0 else "e" * 64)


def _aggregate_args(tmp_path: Path, store: CaptureStore, record: roles.InstrumentRecord, *,
                    draws: range = range(MIN_SERVINGS), tokens: int = 10_000,
                    cached: range | None = None) -> list[str]:
    """The aggregate role's arguments for *record* and vllm's servings *draws* in *store*, each
    recorded as measuring *tokens*, with the engine caches of the servings *cached* (*draws*)."""
    instrument = tmp_path / "instrument.json"
    instrument.write_text(record.model_dump_json())
    args = ["--instrument", sha256_file(instrument)[0], str(instrument)]
    for d in draws:
        path = store.replies_path("vllm", d)
        args += ["--serving", "vllm", str(d), str(tokens), sha256_file(path)[0], str(path)]
    caches = tmp_path / "member-caches.json"
    caches.write_text(json.dumps({"vllm": {str(d): _member_cache(d).model_dump(mode="json")
                                           for d in (draws if cached is None else cached)}}))
    return [*args, "--member-caches", str(caches), "--view", str(tmp_path / "view")]


RECORD = roles.InstrumentRecord(stack=StackInfo(**engines.STACK), preflight=PreflightResult(
    passed=True, flip_rate_vs_fp32=0.0, max_gap_vs_fp32=0.0, n_tokens=24))


def test_the_aggregate_writes_what_the_old_assembly_would(tmp_path, login):
    task = fixture.flips_task()
    campaign = fixture.campaign(engines=["vllm"], mu_per_mtok=MU_40K)
    store = _scored(tmp_path, task)
    code, out, report = _main(tmp_path, "aggregate", *_aggregate_args(tmp_path, store, RECORD), campaign=campaign)
    assert code == 0
    written = json.loads((out / CALIBRATION_FILE).read_text())
    state = calibrate.Campaign()
    for d in range(MIN_SERVINGS):
        state.add(store.read_scored("vllm", d))
    old = calibrate.assemble(
        state, task=task, module="cal_flips.task", engines=["vllm"],
        tol=Tolerances(eps=campaign.eps, mu_per_mtok=campaign.mu_per_mtok, delta=campaign.delta),
        stack=RECORD.stack, preflight=RECORD.preflight,
        engine_configs={"vllm": roles.engine_config("vllm", task)}, sessions_per_draw=None,
        stopped=StopReason.TARGET_REACHED).model_dump(mode="json")
    # The same artifact but for the moment it was assembled.
    assert {k: v for k, v in written.items() if k != "created_at"} == {k: v for k, v in old.items()
                                                                       if k != "created_at"}
    summary = roles.AggregateRecord.model_validate_json((out / roles.AGGREGATE_FILE).read_bytes())
    assert (summary.kind, summary.servings, summary.tokens) == ("flips", {"vllm": list(range(8))}, {"vllm": 80_000})
    # The compile products each serving's member replayed, as its capture stage recorded it.
    assert summary.member_caches == {"vllm": {d: _member_cache(d) for d in range(8)}}
    assert (summary.ceiling, summary.stopped) == (old["q_max"], StopReason.TARGET_REACHED)
    assert summary.warnings == []
    assert sorted(read_outputs_report(report)) == [roles.AGGREGATE_FILE, CALIBRATION_FILE]
    # The calibration loads as a file a role is handed.
    handed = tmp_path / "handed"
    handed.mkdir()
    (handed / CALIBRATION_FILE).write_bytes((out / CALIBRATION_FILE).read_bytes())
    assert read_calibration(handed / CALIBRATION_FILE, task)[0].q_max == old["q_max"]


def test_the_aggregate_reports_what_its_censuses_flag_and_the_candidate_holds_none_of_it(tmp_path, login, capsys):
    # The one reply of every serving flips by 2.5 nats at its first position: one site, in all
    # eight servings.
    task = fixture.flips_task()
    store = _scored(tmp_path, task, first_gap=2.5)
    code, out, _ = _main(tmp_path, "aggregate", *_aggregate_args(tmp_path, store, RECORD),
                         campaign=fixture.campaign(engines=["vllm"], mu_per_mtok=MU_40K))
    assert code == 0
    summary = roles.AggregateRecord.model_validate_json((out / roles.AGGREGATE_FILE).read_bytes())
    assert summary.warnings == [
        "pool engine 'vllm': severe flips recur across servings: event 't0:r.0' produced a flip of at least 0.5 "
        "nats in 8 of 8 servings (flagged at 4). Informational; the ceilings stand as measured."]
    assert summary.warnings[0] in capsys.readouterr().err
    written = json.loads((out / CALIBRATION_FILE).read_text())
    assert not {"trust_warnings", "recurrence_warnings"} & set(written)


@pytest.mark.parametrize("handed", ["instrument", "replies"])
def test_an_aggregate_input_that_is_not_its_manifests_refuses(tmp_path, login, capsys, handed):
    task = fixture.flips_task()
    args = _aggregate_args(tmp_path, _scored(tmp_path, task), RECORD)
    digest = args.index("--instrument") + 1 if handed == "instrument" else args.index("--serving") + 4
    path = args[digest + 1]
    args[digest] = "0" * 64
    code, out, _ = _main(tmp_path, "aggregate", *args, campaign=fixture.campaign(engines=["vllm"], mu_per_mtok=MU_40K))
    assert code == roles.REFUSED_EXIT
    assert _refusal(capsys).startswith(f"InputMismatch: {path} is not the file its stage recorded")
    assert not (out / CALIBRATION_FILE).exists()


def test_a_serving_whose_tokens_are_not_its_records_refuses(tmp_path, login, capsys):
    task = fixture.flips_task()
    campaign = fixture.campaign(engines=["vllm"], mu_per_mtok=MU_40K)
    store = _scored(tmp_path, task)
    code, _, _ = _main(tmp_path, "aggregate", *_aggregate_args(tmp_path, store, RECORD, tokens=9_999),
                       campaign=campaign)
    assert code == roles.REFUSED_EXIT
    assert _refusal(capsys) == "InputMismatch: serving 0 of vllm measured 10000 tokens; its score role recorded 9999"


def test_a_failed_precision_check_voids_with_the_servings_it_needs(tmp_path, login, rows, capsys):
    task = fixture.report_task()
    campaign = fixture.campaign(task, engines=["vllm"])
    store = CaptureStore(tmp_path / "replies", campaign.workload_fingerprint)
    serving = _captured(task, rows, store, _ReportMember(replies=2, tokens=3), 0, with_reports=True)
    # Two replies diverging by 0.1 and 0.5 nats per position: the mean is 0.3 and the
    # reply-clustered error 0.2, far outside the 5% target.
    reference = _ReportReference(tail_of=lambda item, index: (-0.1, -0.5)[index % 2])
    asyncio.run(roles.score(task, [serving], kind="report", share=1, store=store,
                            rows=RowIndex.from_sessions(_sessions(task, rows)), deploy=_make_deploy(reference)))
    stats = store.read_report_scored("vllm", 0)
    record = roles.InstrumentRecord(stack=RECORD.stack, preflight=None)
    code, out, _ = _main(tmp_path, "aggregate", *_aggregate_args(tmp_path, store, record, draws=range(1),
                                                                 tokens=stats.n_tokens),
                         module="cal_report.task", campaign=campaign)
    assert code == roles.REFUSED_EXIT
    refusal = _refusal(capsys)
    assert refusal.startswith("CalibrationError: calibration failed its precision target (eps=0.05): vllm:")
    assert "178 servings are needed" in refusal and "--servings-per-engine" in refusal
    assert not (out / CALIBRATION_FILE).exists()


def test_a_token_level_aggregate_needs_the_preflight(tmp_path, login, capsys):
    task = fixture.flips_task()
    store = _scored(tmp_path, task)
    record = roles.InstrumentRecord(stack=RECORD.stack, preflight=None)
    code, _, _ = _main(tmp_path, "aggregate", *_aggregate_args(tmp_path, store, record),
                       campaign=fixture.campaign(engines=["vllm"], mu_per_mtok=MU_40K))
    assert code == roles.REFUSED_EXIT
    assert _refusal(capsys).startswith("InputMismatch: a token-level calibration is assembled with the instrument's")


@pytest.mark.parametrize(("servings", "target", "refusal"), [
    pytest.param(MIN_SERVINGS, 120_000, "CalibrationError: calibration stopped (exposure target reached) short of its "
                 "exposure target: (T) vllm: 8e+04 of 1.2e+05 tokens", id="short-of-exposure"),
    pytest.param(1, 10_000, "CalibrationError: engines ['vllm'] have fewer than 2 scored servings; the between-serving "
                 "spread the rate ceiling rests on is not estimable from that.", id="one-serving"),
    pytest.param(5, 40_000, "CalibrationError: calibration reached its exposure target but on fewer than 8 servings "
                 "(vllm: 5); the between-serving spread (P) is computed from is not a usable estimate from that few.",
                 id="under-the-floor"),
])
def test_an_aggregate_of_fewer_servings_than_the_campaign_owes_refuses(tmp_path, login, capsys, servings, target,
                                                                      refusal):
    # The grower adds the aggregate only once no engine owes a serving; the assembly's own checks
    # hold the candidate to the exposure target and to the floor too.
    task = fixture.flips_task()
    store = _scored(tmp_path, task, servings=servings)
    code, out, _ = _main(tmp_path, "aggregate", *_aggregate_args(tmp_path, store, RECORD, draws=range(servings)),
                         campaign=fixture.campaign(engines=["vllm"], mu_per_mtok=math.log(40) * 1e6 / target))
    assert code == roles.REFUSED_EXIT and _refusal(capsys) == refusal
    assert not (out / CALIBRATION_FILE).exists()


def _report_scored(tmp_path: Path, task, rows: Path, reference: _ReportReference, draws: range) -> CaptureStore:
    """vllm's servings *draws* of four replies of six tokens each by the report member, scored by
    *reference* as score roles publish them."""
    store = CaptureStore(tmp_path / "replies", workload_fingerprint(task))
    servings = [_captured(task, rows, store, _ReportMember(replies=4, tokens=6), d, with_reports=True) for d in draws]
    asyncio.run(roles.score(task, servings, kind="report", share=1, store=store,
                            rows=RowIndex.from_sessions(_sessions(task, rows)), deploy=_make_deploy(reference)))
    return store


def test_the_report_aggregate_writes_what_the_old_assembly_would_at_the_verifiers_alpha(tmp_path, login, rows):
    # Every reply's ending is priced like any other position, at 0.1, so each serving's four
    # endings fall where the reference expects 2.4: the count's p is about 0.2, which the
    # deployed verifier's alpha of 0.01 accepts, as a trusted stack must be.
    task = fixture.report_task()
    reference = _ReportReference(tail_of=lambda item, index: -0.1 - 0.02 * (index % 5), eot_price=0.1, cut=True)
    store = _report_scored(tmp_path, task, rows, reference, range(1))
    campaign = fixture.campaign(task, engines=["vllm"], eps=0.5)
    record = roles.InstrumentRecord(stack=RECORD.stack, preflight=None)
    tokens = store.read_report_scored("vllm", 0).n_tokens
    code, out, _ = _main(tmp_path, "aggregate", *_aggregate_args(tmp_path, store, record, draws=range(1),
                                                                 tokens=tokens),
                         module="cal_report.task", campaign=campaign)
    assert code == 0
    alpha = calibrate.report_verifier(task).alpha
    state = calibrate.Campaign()
    state.add(store.read_report_scored("vllm", 0))
    old = calibrate.assemble_report(state, task=task, module="cal_report.task", engines=["vllm"], stack=RECORD.stack,
                                    eps=campaign.eps, eot_expected_max=campaign.eot_expected_max, alpha=alpha,
                                    n_servings=1, stopped=StopReason.TARGET_REACHED).model_dump(mode="json")
    written = json.loads((out / CALIBRATION_FILE).read_text())
    assert {k: v for k, v in written.items() if k != "created_at"} == {k: v for k, v in old.items()
                                                                       if k != "created_at"}
    count = written["end_of_turn"]["engines"]["vllm"]
    assert count["observed"] == 4 and alpha < count["p_value"] < 0.5


def test_a_report_aggregate_of_fewer_servings_than_asked_refuses(tmp_path, login, rows, capsys):
    task = fixture.report_task()
    store = _report_scored(tmp_path, task, rows, _ReportReference(), range(1))
    record = roles.InstrumentRecord(stack=RECORD.stack, preflight=None)
    code, _, _ = _main(tmp_path, "aggregate", *_aggregate_args(
        tmp_path, store, record, draws=range(1), tokens=store.read_report_scored("vllm", 0).n_tokens),
        module="cal_report.task", campaign=fixture.campaign(task, engines=["vllm"], servings_per_engine=2))
    assert code == roles.REFUSED_EXIT
    assert _refusal(capsys) == ("CalibrationError: engines ['vllm'] have fewer than the 2 servings this campaign "
                                "asked for (stopped: exposure target reached).")


def test_an_aggregate_handed_a_serving_twice_refuses(tmp_path, login, capsys):
    task = fixture.flips_task()
    args = _aggregate_args(tmp_path, _scored(tmp_path, task), RECORD)
    first = args.index("--serving")
    code, _, _ = _main(tmp_path, "aggregate", *args[:first + 6], *args[first:],
                       campaign=fixture.campaign(engines=["vllm"], mu_per_mtok=MU_40K))
    assert code == roles.REFUSED_EXIT and _refusal(capsys) == "InputMismatch: serving 0 of vllm is handed twice"


def test_an_aggregate_handed_the_engine_caches_of_other_servings_refuses(tmp_path, login, capsys):
    task = fixture.flips_task()
    args = _aggregate_args(tmp_path, _scored(tmp_path, task), RECORD, cached=range(1, 9))
    code, out, _ = _main(tmp_path, "aggregate", *args, campaign=fixture.campaign(engines=["vllm"], mu_per_mtok=MU_40K))
    assert code == roles.REFUSED_EXIT
    assert _refusal(capsys).startswith("InputMismatch: the member caches name servings [('vllm', 1), ")
    assert not (out / CALIBRATION_FILE).exists()



@pytest.mark.parametrize(("fields", "refusal"), [
    pytest.param({"seeded_from": "e" * 64, "entry": "e" * 64, "published": True},
                 "a stage that published an entry built its cache cold and names the entry", id="seeded-and-published"),
    pytest.param({"seeded_from": None, "published": True},
                 "a stage that published an entry built its cache cold and names the entry", id="published-nothing"),
    pytest.param({"seeded_from": "e" * 64, "unpublished": "the publish failed"},
                 "a stage that left no entry built its cache cold and names none", id="seeded-and-unpublished"),
    pytest.param({"seeded_from": None, "entry": "e" * 64, "unpublished": "the publish failed"},
                 "a stage that left no entry built its cache cold and names none", id="left-and-unpublished"),
    pytest.param({"seeded_from": "e" * 64, "entry": "f" * 64},
                 "an owner seeded from an entry leaves that entry", id="seeded-from-another"),
])
def test_an_engine_cache_record_tells_one_story(fields, refusal):
    with pytest.raises(ValueError, match=refusal):
        EngineCacheRecord(key="B200-0123456789abcdef-fedcba9876543210", **fields)


@pytest.mark.parametrize(("role", "args", "bound"), [
    ("warm", ("--engine", "vllm", "--load-sec", "999"), 999.0),
    ("instrument", ("--gpus", "2"), 111.0),
])
def test_each_role_bounds_its_engines_loads(tmp_path, login, fake_engines, monkeypatch, role, args, bound):
    # The warm role builds its member's cache cold, for as long as its stage gives it; every other
    # role loads a seeded or a reference engine, within the campaign's bound.
    hosts = []

    class Recorded(inprocess.EngineHost):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            hosts.append(self)

    monkeypatch.setattr(roles, "EngineHost", Recorded)
    code, _, _ = _main(tmp_path, role, *args, campaign=fixture.campaign(load_timeout_sec=111.0))
    assert code == 0 and [host.load_sec for host in hosts] == [bound]
    # An engine the host gives up on ends the role with the status the stage retries, quoting its line.
    assert [host.give_up for host in hosts] == [roles.GIVE_UP]
    assert (roles.GIVE_UP.exit_code, roles.GIVE_UP.prefix) == (roles.ENGINE_FAULT_EXIT, roles.ENGINE_FAULT_PREFIX)


def test_the_engine_config_line_records_the_penalty_only_when_it_is_set():
    # The auditable record of how a member served gains the penalty when one is in force, and is
    # byte-identical to a plain-greedy run's line when none is.
    plain = roles.engine_config("vllm", _token_task())
    assert "frequency_penalty" not in plain
    assert plain == "VLLMEngine(dtype='auto'), clocked serving"
    assert "frequency_penalty=0.1" in roles.engine_config("vllm", _penalized_task())
