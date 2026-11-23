"""``hls run`` is the one command a run of a task's default configuration needs: it finds the
baseline credit and the calibration the run loads, the store's own file before the task's
defaults, each only when it fits the task, and measures what it does not find, with the cache's
data and the reference answers the run needs, before it plans the rollout
(``hls.pipeline.prepare``). These tests plan multitenant_slo, which loads both files, from this
repository's checkout; the measuring and filling commands are planned for real and stood in for
only where they would submit, each stand-in leaving in the store what its instance leaves."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hls import cli, measured
from hls.baseline_credit import BaselineCredit, baseline_credit_bytes
from hls.measured import (BASELINE_CREDIT_FILE, CALIBRATION_FILE, DEV_REFERENCE_FILE, MeasuredFile, defaults_path,
                          pinned_key, write_measured)
from hls.pipeline import commands, credit, prepare
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.planner import PlanError
from hls.pipeline.preflight import AnswersMissing
from hls.stages.inputs import measured_copies
from hls.store.posix import PosixStore
from hls.taskref import TaskRef, blob_id
from tests.agent.build_fixture import build_config
from tests.pipeline.plans import CODE, pin_harbor
from tests.provision.cached import provision_task

MODULE = "tasks.multitenant_slo.task"
TASK_ID = "multitenant_slo"
REF = TaskRef(module=MODULE)
CREDIT, CALIBRATION = (defaults_path(REF, name) for name in (BASELINE_CREDIT_FILE, CALIBRATION_FILE))


@pytest.fixture
def login(monkeypatch, tmp_path, warmed_servers):
    """The login node planning from this repository's checkout, on a cluster whose images warmers
    built every server image (``warmed_servers``). The checks with tests of their own find nothing to refuse: the
    defaults' commit (no checkout of the tests commits them), the upload folder and the oracle
    answers."""
    monkeypatch.setattr(commands, "code_state", lambda path: CODE)
    pin_harbor(monkeypatch, CODE)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))
    monkeypatch.setattr(commands, "check_defaults_committed", lambda checkout, files: None)
    monkeypatch.setattr(commands, "check_upload_folder", lambda checkout, task_module, producer: None)
    monkeypatch.setattr(commands, "check_oracle_coverage", lambda store, **kw: None)
    return build_config(tmp_path)


def _no_step(*args, **kw):
    pytest.fail("a step ran, though the run lacked nothing")


def _prepare(config: Path, lines: list[str], **kw) -> int:
    return prepare.prepare(config=config, overrides={}, task_module=MODULE, arm="arm-a", arm_change=None,
                           echo=lines.append, **{"start": _no_step, "start_credit": _no_step, **kw})


def _plan(config: Path, lines: list[str]) -> Plan:
    return commands.plan_run(config=config, overrides={}, task_module=MODULE, arm="arm-a", arm_change=None,
                             job_name="hls-run", echo=lines.append)


def _one_drive_less(data: bytes) -> bytes:
    """The default credit with its first drive left out: another measurement that fits the task."""
    bc = BaselineCredit.model_validate_json(data)
    return baseline_credit_bytes(bc.model_copy(update={"drives": bc.drives[1:]}))


def test_a_task_with_its_defaults_measures_nothing_and_its_run_pins_them(tmp_path, login):
    provision_task(tmp_path / "store", MODULE)
    lines: list[str] = []
    assert _prepare(login, lines) == 0
    assert lines == []
    plan = _plan(login, lines)
    assert plan.measured.credit == MeasuredFile(source="defaults", path="tasks/multitenant_slo/defaults/baseline.json",
                                                blob=blob_id(CREDIT.read_bytes()))
    assert plan.measured.calibration == MeasuredFile(
        source="defaults", path="tasks/multitenant_slo/defaults/calibration.json", blob=blob_id(CALIBRATION.read_bytes()))
    assert (f"baseline credit: the task's defaults tasks/multitenant_slo/defaults/baseline.json "
            f"(blob {blob_id(CREDIT.read_bytes())[:12]})") in lines


def test_the_stores_own_file_wins_over_the_tasks_default(tmp_path, login):
    provision_task(tmp_path / "store", MODULE)
    store = PosixStore(tmp_path / "store")
    data = _one_drive_less(CREDIT.read_bytes())
    write_measured(store, TASK_ID, BASELINE_CREDIT_FILE, data)
    assert _prepare(login, []) == 0
    plan = _plan(login, [])
    assert plan.measured.credit == MeasuredFile(source="store", path=pinned_key(TASK_ID, "baseline.json", blob_id(data)),
                                                blob=blob_id(data))
    assert plan.measured.calibration is not None and plan.measured.calibration.source == "defaults"


def test_a_file_that_does_not_fit_the_task_is_not_used_and_the_run_says_why(tmp_path, login):
    provision_task(tmp_path / "store", MODULE)
    store = PosixStore(tmp_path / "store")
    stale = json.loads(CREDIT.read_text())
    stale["campaign"]["workload_fingerprint"] = "0" * 64
    write_measured(store, TASK_ID, BASELINE_CREDIT_FILE, json.dumps(stale, indent=2).encode())
    lines: list[str] = []
    assert _prepare(login, lines) == 0
    plan = _plan(login, lines)
    assert plan.measured.credit is not None and plan.measured.credit.source == "defaults"
    [said] = [line for line in lines if line.startswith("baseline credit:")]
    assert f"not store measured/{TASK_ID}/baseline.json: stale baseline credit" in said
    assert f"workload_fingerprint={'0' * 64!r}" in said


def test_a_second_measurement_leaves_a_run_that_already_planned_its_own_bytes(tmp_path, login, monkeypatch):
    """Two ``hls run`` of one task on a store with no fitting default. Run A's credit lands, A plans
    its rollout and pins it. Run B's credit, as fitting, lands later and becomes the store's current
    file. A's stages still read the bytes A pinned, and B pins its own."""
    monkeypatch.setattr(measured, "defaults_path", lambda ref, name: tmp_path / "none" / name)
    provision_task(tmp_path / "store", MODULE)
    store = PosixStore(tmp_path / "store")
    write_measured(store, TASK_ID, BASELINE_CREDIT_FILE, CREDIT.read_bytes())  # run A's fold
    write_measured(store, TASK_ID, CALIBRATION_FILE, CALIBRATION.read_bytes())
    plan_a = _plan(login, [])
    assert plan_a.measured.credit is not None and plan_a.measured.credit.source == "store"
    write_measured(store, TASK_ID, BASELINE_CREDIT_FILE, _one_drive_less(CREDIT.read_bytes()))  # run B's fold
    plan_b = _plan(login, [])
    assert plan_b.measured.credit is not None and plan_b.measured.credit.blob != plan_a.measured.credit.blob
    for plan, data in ((plan_a, CREDIT.read_bytes()), (plan_b, _one_drive_less(CREDIT.read_bytes()))):
        copies = measured_copies(SimpleNamespace(plan=plan), tmp_path / "role" / plan.instance / "measured")
        assert copies.credit is not None and copies.credit.read_bytes() == data


def test_a_probe_pins_the_dev_reference_that_fits_as_a_scored_run_does(tmp_path, login):
    """multimodal_asr's default dev reference fits; a probe's plan pins it as a scored plan does,
    so the probe's sandbox has ``oracle_dev.json`` beside the dev rows as its brief says."""
    module = "tasks.multimodal_asr.task"
    provision_task(tmp_path / "store", module)
    scored = commands.plan_run(config=login, overrides={}, task_module=module, arm="arm-a", arm_change=None,
                               job_name="hls-run", echo=[].append)
    probe = commands.plan_run(config=login, overrides={}, task_module=module, arm="arm-a", arm_change=None,
                              job_name="hls-run", echo=[].append, stop_after="probe")
    assert scored.measured.dev_reference is not None
    assert probe.measured.credit is not None
    assert probe.measured.dev_reference == scored.measured.dev_reference


@pytest.mark.parametrize("overrides", [{}, {"task.contract.port": "8123"}], ids=["as-declared", "overridden"])
def test_a_scored_run_with_no_file_that_fits_is_refused_naming_the_command_that_measures_it(tmp_path, login,
                                                                                             monkeypatch, overrides):
    """The command named measures the task the run plans, its task overrides included."""
    provision_task(tmp_path / "store", MODULE)
    monkeypatch.setattr(measured, "defaults_path", lambda ref, name: tmp_path / "none" / name)
    flags = "".join(f" --{path} {value}" for path, value in overrides.items())
    with pytest.raises(PlanError, match=re.escape(f"measure it with hls baseline-credit {MODULE}{flags}, or let hls "
                                                  "run measure it")):
        commands.plan_run(config=login, overrides=overrides, task_module=MODULE, arm="arm-a", arm_change=None,
                          job_name="hls-run", echo=[].append)


def test_a_task_with_nothing_measured_goes_from_one_hls_run_to_a_launched_arm(tmp_path, login, monkeypatch):
    """A cold store and a task with no defaults: ``hls run`` provisions, measures the credit with
    one drive, then the calibration, each saying why, and then plans and starts the rollout, which
    pins what the measurements wrote into the store."""
    monkeypatch.setattr(measured, "defaults_path", lambda ref, name: tmp_path / "none" / name)
    store = PosixStore(tmp_path / "store")
    started: list[Plan] = []

    def start(plan: Plan, overrides, **kw) -> int:
        """Each instance as it ends, leaving in the store what it would."""
        started.append(plan)
        if plan.pipeline == "provision":
            provision_task(tmp_path / "store", MODULE)
        elif plan.pipeline == "calibration":
            write_measured(store, TASK_ID, CALIBRATION_FILE, CALIBRATION.read_bytes())
        return 0

    def run_credit(plan: Plan, overrides, **kw) -> int:
        started.append(plan)
        write_measured(store, TASK_ID, BASELINE_CREDIT_FILE, _one_drive_less(CREDIT.read_bytes()))
        return 0

    monkeypatch.setattr(commands, "start", start)
    monkeypatch.setattr(credit, "run_credit", run_credit)
    result = CliRunner().invoke(cli.app, ["run", MODULE, "--arm", "arm-a", "--config", str(login)])
    assert result.exit_code == 0, result.output
    assert [p.pipeline for p in started] == ["provision", "credit", "calibration", "rollout"]
    # The credit is measured with one drive, the default.
    assert [s.name for s in started[1].stages if s.kind == "deploy_drive"] == ["drive_1"]
    rollout = started[-1]
    assert rollout.measured.credit is not None and rollout.measured.credit.source == "store"
    assert rollout.measured.calibration is not None and rollout.measured.calibration.source == "store"
    said = result.output
    assert f"{TASK_ID}: the cache lacks the task's data, so it is provisioned first (hls provision {MODULE})" in said
    assert (f"{TASK_ID}: baseline credit: none fits the task (neither the store nor the task's defaults hold one), so "
            f"it is measured first (hls baseline-credit {MODULE})") in said
    assert f"so it is measured first (hls calibrate {MODULE})" in said


def test_a_run_whose_reference_answers_are_missing_fills_them_first(tmp_path, login, monkeypatch):
    """multimodal_asr's verify reads its caching reference's answers from the store:
    with its data and its defaults there and no completed fill,
    ``hls run`` fills them, saying why, then plans the rollout."""
    module = "tasks.multimodal_asr.task"
    provision_task(tmp_path / "store", module)
    filled: list[str] = []

    def coverage(store, **kw):
        if not filled:
            raise AnswersMissing("multimodal-asr: no completed oracle fill answered its test prompts")

    monkeypatch.setattr(commands, "check_oracle_coverage", coverage)
    started: list[str] = []

    def start(plan: Plan, overrides, **kw) -> int:
        started.append(plan.pipeline)
        if plan.pipeline == "oracle_fill":
            filled.append(plan.instance)
        return 0

    monkeypatch.setattr(commands, "start", start)
    monkeypatch.setattr(credit, "run_credit", _no_step)
    result = CliRunner().invoke(cli.app, ["run", module, "--arm", "arm-a", "--config", str(login)])
    assert result.exit_code == 0, result.output
    assert started == ["oracle_fill", "rollout"]
    assert ("multimodal-asr: its reference answers are not all in the store: multimodal-asr: no completed oracle "
            f"fill answered its test prompts; so the oracle fill runs first (hls oracle fill {module})"
            ) in result.output


def test_a_store_whose_answers_cannot_be_read_is_refused_without_a_fill(tmp_path, login, monkeypatch):
    """Only answers a fill writes start one: a question set that cannot be read is refused as the
    grading's own plan refuses it, with nothing started."""
    module = "tasks.multimodal_asr.task"
    provision_task(tmp_path / "store", module)

    def coverage(store, **kw):
        raise PlanError("answers/x/questions.json cannot be read as a question set: ValueError: not JSON")

    monkeypatch.setattr(commands, "check_oracle_coverage", coverage)
    monkeypatch.setattr(commands, "start", _no_step)
    monkeypatch.setattr(credit, "run_credit", _no_step)
    result = CliRunner().invoke(cli.app, ["run", module, "--arm", "arm-a", "--config", str(login)])
    assert result.exit_code == 1, result.output
    assert result.output.splitlines() == ["hls run: answers/x/questions.json cannot be read as a question set: "
                                          "ValueError: not JSON"]


def test_a_run_whose_dev_reference_is_missing_has_the_oracle_fill_produce_it(tmp_path, login, monkeypatch):
    """multimodal_asr publishes a dev reference beside the dev rows. With neither the store nor the
    task's defaults holding one, ``hls run`` has the oracle fill produce it into the store, saying
    why, then plans the rollout, which pins the store's file; the rows entry is the same either
    way."""
    module = "tasks.multimodal_asr.task"
    real = measured.defaults_path
    default = real(TaskRef(module=module), DEV_REFERENCE_FILE)
    monkeypatch.setattr(measured, "defaults_path",
                        lambda ref, name: tmp_path / "none" / name if name == DEV_REFERENCE_FILE else real(ref, name))
    provisioned = provision_task(tmp_path / "store", module)
    store = PosixStore(tmp_path / "store")
    started: list[Plan] = []

    def start(plan: Plan, overrides, **kw) -> int:
        started.append(plan)
        if plan.pipeline == "oracle_fill":
            write_measured(store, "multimodal-asr", DEV_REFERENCE_FILE, default.read_bytes())
        return 0

    monkeypatch.setattr(commands, "start", start)
    monkeypatch.setattr(credit, "run_credit", _no_step)
    result = CliRunner().invoke(cli.app, ["run", module, "--arm", "arm-a", "--config", str(login)])
    assert result.exit_code == 0, result.output
    assert [p.pipeline for p in started] == ["oracle_fill", "rollout"]
    assert started[0].stage("oracle_fill").params["dev_reference"] is True
    assert ("multimodal-asr: dev reference: none fits the task (neither the store nor the task's defaults hold one); "
            f"so the oracle fill runs first (hls oracle fill {module})") in result.output
    pin = started[-1].measured.dev_reference
    assert pin is not None and (pin.source, pin.path) == ("store", pinned_key("multimodal-asr", DEV_REFERENCE_FILE,
                                                                               blob_id(default.read_bytes())))
    assert provision_task(tmp_path / "store", module) == provisioned


def test_a_step_that_does_not_succeed_ends_the_run_with_its_status_and_plans_no_rollout(tmp_path, login,
                                                                                         monkeypatch):
    provision_task(tmp_path / "store", MODULE)
    monkeypatch.setattr(measured, "defaults_path", lambda ref, name: tmp_path / "none" / name)
    started: list[str] = []
    monkeypatch.setattr(credit, "run_credit", lambda plan, overrides, **kw: started.append(plan.pipeline) or 1)
    monkeypatch.setattr(commands, "start", lambda plan, overrides, **kw: started.append(plan.pipeline) or 0)
    result = CliRunner().invoke(cli.app, ["run", MODULE, "--arm", "arm-a", "--config", str(login)])
    assert result.exit_code == 1 and started == ["credit"]
    said = [line for line in result.stderr.splitlines() if "did not succeed" in line]
    assert len(said) == 1 and re.fullmatch(
        r"hls run: hls baseline-credit did not succeed \((\S+)\), so no rollout is planned: hls resume \1 "
        r"drives it to its end, then hls run plans the rollout", said[0]), result.output


@pytest.mark.parametrize("unusable", ["read-only", "under-a-read-only-parent"])
@pytest.mark.parametrize("argv", [["run", MODULE, "--arm", "arm-a"], ["provision", MODULE]], ids=["run", "provision"])
def test_a_store_root_this_user_cannot_write_is_refused_in_one_line(tmp_path, login, monkeypatch, unusable, argv):
    """A ``site.store_root`` that cannot be made or written is refused when the command plans, in
    one line naming the setting and the path, with nothing submitted."""
    locked = tmp_path / "locked"
    locked.mkdir()
    root = locked if unusable == "read-only" else locked / "missing" / "store"
    locked.chmod(0o500)
    try:
        config = login.read_text().replace(f'store_root = "{tmp_path / "store"}"', f'store_root = "{root}"')
        assert config != login.read_text()
        login.write_text(config)
        monkeypatch.setattr(commands, "start", _no_step)
        result = CliRunner().invoke(cli.app, [*argv, "--config", str(login)])
    finally:
        locked.chmod(0o700)
    assert result.exit_code == 1, result.output
    why = ("is not writable by this user" if unusable == "read-only"
           else "does not exist and cannot be made: Permission denied")
    assert result.output.splitlines() == [f"hls {argv[0]}: site.store_root {root} {why}"]
    assert not root.exists() or not any(root.iterdir())


def test_an_arm_the_run_refuses_is_refused_before_anything_is_measured(tmp_path, login, monkeypatch):
    monkeypatch.setattr(measured, "defaults_path", lambda ref, name: tmp_path / "none" / name)
    monkeypatch.setattr(commands, "start", _no_step)
    monkeypatch.setattr(credit, "run_credit", _no_step)
    result = CliRunner().invoke(cli.app, ["run", MODULE, "--arm", "no-such-arm", "--config", str(login)])
    assert result.exit_code == 2 and "--arm" in result.output
