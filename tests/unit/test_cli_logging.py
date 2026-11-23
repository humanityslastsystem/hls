# tests/unit/test_cli_logging.py
"""The CLI commands must configure logging before doing work."""
from __future__ import annotations

import pytest
import typer
from typer.main import get_command
from typer.testing import CliRunner

import hls.cli as cli


def test_view_configures_logging(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: calls.append((a, k)))
    # view() echoes + raises typer.Exit(1) when runs_dir is missing -- AFTER it has
    # configured logging, which is all we assert here.
    with pytest.raises(typer.Exit):
        cli.view(runs_dir=tmp_path / "does-not-exist", host="127.0.0.1",
                 port=8080, no_build=True)
    assert calls, "view() did not call configure_logging"


def test_follow_configures_logging(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: calls.append((a, k)))
    # An instance the store does not hold exits 2, after logging is configured.
    with pytest.raises(typer.Exit) as ended:
        cli.follow_command(instance="runs/t/none", config=tmp_path / "hls.toml", store=tmp_path / "store")
    assert ended.value.exit_code == 2
    assert calls, "follow did not call configure_logging"


def test_the_old_command_bodies_are_gone():
    """The CLI holds no in-process run, grade, calibrate or baseline-credit body: calibrate is
    the stage pipeline's command, with its options; follow tails a stage-based instance, and
    baseline-credit plans a credit instance's drives."""
    for name in ("_assemble_run", "_run", "_grade", "_run_and_finalize", "_build_config", "_resolve_latest",
                 "_assemble_calibrate", "_assemble_baseline_credit", "follow", "RUNS_ROOT", "_grade_skip_roles",
                 "_gate_reasoning_effort", "_assemble_context"):
        assert not hasattr(cli, name), name
    calibrate = get_command(cli.app).get_command(None, "calibrate")
    options = {option for param in calibrate.params for option in param.opts}
    assert "--max-servings-per-engine" in options
    assert not options & {"--budget", "--captures", "--split", "--rollout"}
    result = CliRunner().invoke(cli.app, ["follow", "--help"])
    assert result.exit_code == 0 and "INSTANCE" in result.output
    credit = typer.main.get_command(cli.app).commands["baseline-credit"]
    assert {p.name for p in credit.params} == {"task_module", "drives", "fanout", "job_name", "config"}
