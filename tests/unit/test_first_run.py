"""What a new site sees on its first commands: a missing hls.toml is named as missing, with the
path looked at and the example to copy."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.settings import SettingsError, load_settings


def test_a_missing_hls_toml_is_named_as_missing(tmp_path):
    missing = tmp_path / "hls.toml"
    with pytest.raises(SettingsError) as e:
        load_settings(missing)
    message = str(e.value)
    assert message.startswith(f"{missing}: no such file;")
    assert "docs/examples/hls.toml" in message and "--config" in message
    assert "Field required" not in message


def test_a_missing_relative_hls_toml_names_the_directory_it_was_looked_for_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SettingsError, match="^" + re.escape(f"{tmp_path / 'hls.toml'}: no such file;")):
        load_settings(Path("hls.toml"))


def test_a_command_with_no_hls_toml_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for argv in (["provision", "tasks.sample_task.task"], ["status", "runs/noop/x"]):
        result = CliRunner().invoke(cli.app, argv)
        assert result.exit_code == 1, result.output
        assert f"{tmp_path / 'hls.toml'}: no such file;" in result.output
        assert "Field required" not in result.output


def test_view_of_a_directory_needs_no_hls_toml(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.chdir(tmp_path)
    assert not os.path.exists("hls.toml")
    served: list = []
    monkeypatch.setattr(cli, "_serve", lambda app, host, port: served.append(app))
    monkeypatch.setattr(cli, "_viewer_static_dir", lambda: None)
    result = CliRunner().invoke(cli.app, ["view", str(runs), "--no-build"])
    assert result.exit_code == 0, result.output
    assert served
