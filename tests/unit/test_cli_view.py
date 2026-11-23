import re

import pytest
from typer.testing import CliRunner

from hls.cli import app


def test_view_builds_rollouts_app(tmp_path, monkeypatch):
    # make a minimal rollouts tree so detection/serving has something
    rdir = tmp_path / "example-task" / "rid1"
    rdir.mkdir(parents=True)
    (rdir / "rollout.json").write_text("{}")

    captured = {}

    def fake_serve(app_obj, host, port):
        captured["served"] = True

    # _serve is our thin wrapper around uvicorn.run
    monkeypatch.setattr("hls.cli._serve", fake_serve)
    # avoid bun build in tests: pretend static is absent (API-only is fine)
    monkeypatch.setattr("hls.cli._viewer_static_dir", lambda: None)

    result = CliRunner().invoke(app, ["view", str(tmp_path), "--no-build"])
    assert result.exit_code == 0, result.output
    assert captured.get("served") is True
    # The URL it prints carries the session's token, a fresh one each start.
    tokens = re.findall(r"\?token=([A-Za-z0-9_-]{43})", result.output)
    again = CliRunner().invoke(app, ["view", str(tmp_path), "--no-build"])
    assert len(tokens) == 1 and tokens != re.findall(r"\?token=([A-Za-z0-9_-]{43})", again.output)


def test_a_directory_given_is_served_without_reading_the_config(tmp_path, monkeypatch):
    runs = tmp_path / "published"
    runs.mkdir()
    config = tmp_path / "hls.toml"
    config.write_text("[site]\nnot_a_setting = 1\n")
    served: list = []
    monkeypatch.setattr("hls.cli._serve", lambda app_obj, host, port: served.append(app_obj))
    monkeypatch.setattr("hls.cli._viewer_static_dir", lambda: None)
    result = CliRunner().invoke(app, ["view", str(runs), "--no-build", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert f"(runs: {runs.resolve()})" in result.output and served


def test_without_a_directory_an_invalid_config_is_refused_in_one_line(tmp_path, monkeypatch):
    config = tmp_path / "hls.toml"
    config.write_text("[substrate]\nx = 1\n")
    monkeypatch.setattr("hls.cli._serve", lambda *a: (_ for _ in ()).throw(AssertionError("served")))
    monkeypatch.setattr("hls.cli._viewer_static_dir", lambda: None)
    result = CliRunner().invoke(app, ["view", "--no-build", "--config", str(config)])
    assert result.exit_code == 1
    assert result.stdout == ""
    lines = result.stderr.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("hls view: ") and "unknown setting substrate" in lines[0]


def _refused_in_one_line(args, monkeypatch) -> str:
    """The one stderr line ``hls view`` refuses *args* with, having served nothing and printed
    nothing on stdout."""
    monkeypatch.setattr("hls.cli._serve", lambda *a: (_ for _ in ()).throw(AssertionError("served")))
    monkeypatch.setattr("hls.cli._viewer_static_dir", lambda: None)
    result = CliRunner().invoke(app, ["view", *args, "--no-build"])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.exception
    assert result.stdout == ""
    lines = result.stderr.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("hls view: "), result.stderr
    return lines[0]


def test_without_a_directory_a_config_that_is_not_toml_is_refused_in_one_line(tmp_path, monkeypatch):
    config = tmp_path / "hls.toml"
    config.write_text("[site\n")
    line = _refused_in_one_line(["--config", str(config)], monkeypatch)
    assert line == f"hls view: {config}: not valid TOML (at line 1, column 6)"


def test_without_a_directory_a_config_that_is_not_utf8_is_refused_in_one_line(tmp_path, monkeypatch):
    config = tmp_path / "hls.toml"
    config.write_bytes(b"[site]\n# caf\xe9\n")
    line = _refused_in_one_line(["--config", str(config)], monkeypatch)
    assert line == f"hls view: {config}: not UTF-8 text"


def test_without_a_directory_a_config_that_is_a_directory_is_refused_in_one_line(tmp_path, monkeypatch):
    config = tmp_path / "hls.toml"
    config.mkdir()
    line = _refused_in_one_line(["--config", str(config)], monkeypatch)
    assert line == f"hls view: {config}: cannot be read (Is a directory)"


@pytest.mark.parametrize("make", [lambda p: None, lambda p: p.write_text("{}")], ids=["missing", "a-file"])
def test_a_directory_that_is_not_one_is_refused_in_one_line(tmp_path, monkeypatch, make):
    runs = tmp_path / "runs"
    make(runs)
    assert _refused_in_one_line([str(runs)], monkeypatch) == f"hls view: {runs.resolve()}: no such directory"


def test_a_root_that_holds_the_answers_area_is_refused_in_one_line(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    (tmp_path / "oracle" / "example-task").mkdir(parents=True)
    line = _refused_in_one_line([str(tmp_path)], monkeypatch)
    assert "answers area" in line
