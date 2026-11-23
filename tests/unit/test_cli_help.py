"""Every help text the CLI declares is printed as written: the help renderer reads square brackets
as markup, so a bracket a help text means literally is escaped."""

from __future__ import annotations

import re

import click
import pytest
import typer
from typer.testing import CliRunner

from hls import cli


def _commands(command: click.Command, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], click.Command]]:
    """*command* and every command under it, each with the words that name it after ``hls``."""
    found = [(path, command)]
    if isinstance(command, click.Group):
        for name, sub in sorted(command.commands.items()):
            found += _commands(sub, (*path, name))
    return found


def _words(text: str) -> str:
    """*text* without colours, the help box's borders, markup escapes and line breaks."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text).replace("\\[", "[")
    return " ".join(re.sub(r"[\u2502\u256d\u256e\u2570\u256f\u2500]", " ", text).split())


COMMANDS = _commands(typer.main.get_command(cli.app))


@pytest.mark.parametrize("path, command", COMMANDS, ids=[" ".join(p) or "hls" for p, _ in COMMANDS])
def test_every_help_text_is_printed_as_written(path, command):
    result = CliRunner().invoke(cli.app, [*path, "--help"], env={"COLUMNS": "400"}, terminal_width=400)
    assert result.exit_code == 0, result.output
    printed = _words(result.output)
    if command.help:
        assert _words(command.help) in printed, command.help
    for param in command.params:
        if getattr(param, "help", None):
            assert _words(param.help) in printed, (param.name, param.help)


def test_the_arm_option_names_the_hls_toml_table():
    result = CliRunner().invoke(cli.app, ["run", "--help"], env={"COLUMNS": "400"}, terminal_width=400)
    assert "an [arms.<name>] table of hls.toml" in _words(result.output)
