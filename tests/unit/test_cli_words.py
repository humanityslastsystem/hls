"""The CLI's own words name only what exists: no "runner" for the tool, no qualifier for
commands that are all stage-based, and ``hls view`` serves the store's runs by default."""

from __future__ import annotations

import textwrap

from typer.testing import CliRunner

from hls import cli


def test_the_help_describes_the_tool():
    out = CliRunner().invoke(cli.app, ["--help"]).stdout
    assert "plan, run and look after benchmark instances on a Slurm site" in out
    assert "HLS runner" not in out


def test_no_group_is_qualified_as_stage_based():
    for group in ("images", "oracle"):
        assert "stage-based" not in CliRunner().invoke(cli.app, [group, "--help"]).stdout


def test_view_defaults_to_the_stores_runs(tmp_path, monkeypatch):
    store = tmp_path / "store"
    (store / "runs").mkdir(parents=True)
    config = tmp_path / "hls.toml"
    config.write_text(textwrap.dedent(f"""
        [site]
        partition = "gpu"
        store_root = "{store}"
        node_scratch = "/tmp"
        secrets_path = "{tmp_path / 'secrets.toml'}"
    """))
    served: list = []
    monkeypatch.setattr(cli, "_serve", lambda app, host, port: served.append(app))
    monkeypatch.setattr(cli, "_viewer_static_dir", lambda: None)
    result = CliRunner().invoke(cli.app, ["view", "--no-build", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert f"(runs: {(store / 'runs').resolve()})" in result.output and served
