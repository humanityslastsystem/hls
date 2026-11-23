"""The command list of ``hls --help`` gives each command one line and leaves out the runner's entry
point, which a batch script calls; ``hls calibrate --help`` states each default it applies by its
value, read from the constant the command applies."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys

import click
import typer
from typer.testing import CliRunner

from hls import cli


def _help(*argv: str) -> str:
    result = CliRunner().invoke(cli.app, [*argv, "--help"], env={"COLUMNS": "400"}, terminal_width=400)
    assert result.exit_code == 0, result.output
    text = re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("\\[", "[")
    return " ".join(re.sub("[\u2502\u256d\u256e\u2570\u256f\u2500]", " ", text).split())


def _groups(group: click.Group, path: tuple[str, ...] = ()):
    yield path, group
    for name, sub in group.commands.items():
        if isinstance(sub, click.Group):
            yield from _groups(sub, (*path, name))


def test_the_runner_entry_point_is_left_out_of_the_command_list_and_still_runs():
    root = typer.main.get_command(cli.app)
    assert root.commands["runner"].hidden
    assert not re.search(r"\brunner\b", _help())
    assert "Run one allocation's stages" in _help("runner")


def test_every_listed_command_has_a_one_line_summary():
    for path, group in _groups(typer.main.get_command(cli.app)):
        for name, command in group.commands.items():
            if command.hidden:
                continue
            summary = command.short_help or command.help or ""
            assert summary and "\n" not in summary.strip() and len(summary) <= 100, (path, name, summary)
            assert summary.strip() in _help(*path), (path, name)


def test_calibrates_help_states_each_default_by_its_value():
    printed = _help("calibrate")
    assert "(default hls." not in printed and "DEFAULT_" not in printed
    for option, (module, name) in cli.CALIBRATE_DEFAULTS.items():
        value = getattr(importlib.import_module(module), name)
        flag = "--" + option.replace("_", "-")
        assert re.search(rf"{re.escape(flag)} \S+ [^[]*\[default: {re.escape(str(value))}\]", printed), (flag, value)


def test_the_stated_default_is_the_constants_value_when_the_help_is_printed(monkeypatch):
    from hls.pipeline import calibration

    monkeypatch.setattr(calibration, "DEFAULT_FANOUT", 7)
    assert re.search(r"--fanout INTEGER [^[]*\[default: 7\]", _help("calibrate"))


def test_the_cli_loads_no_numpy_until_a_command_needs_it():
    """The calibrate defaults' modules load numpy, whose thread pool the commands size first
    (``cli._cap_thread_pools``): importing the CLI must not load them."""
    code = "import sys, hls.cli; print('numpy' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout == "False\n"
