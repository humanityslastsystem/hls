"""``hls provision`` at the command line: its defaults, its one-line refusals, and a traceback
that shows no local variable."""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.pipeline import commands
from tests.provision.warmer_config import SECRETS, TOKEN, write_config


@pytest.fixture
def quiet(monkeypatch):
    """configure_logging turns propagation off on the hls loggers, which would break caplog in
    every later test of the session."""
    monkeypatch.setattr(cli, "configure_logging", lambda verbose=False: None)


def test_hls_provision_asks_for_the_weights_unless_told_rows_only(tmp_path, monkeypatch, quiet):
    seen = {}
    monkeypatch.setattr(commands, "plan_provision", lambda **kw: seen.update(kw) or "plan")
    monkeypatch.setattr(commands, "start", lambda plan, overrides, **kw: 0)
    result = CliRunner().invoke(cli.app, ["provision", "tasks.multitenant_slo.task", "--config",
                                          str(tmp_path / "hls.toml")])
    assert result.exit_code == 0, result.output
    assert (seen["weights"], seen["job_name"], seen["overrides"]) == (True, "hls-provision", {})


def test_an_error_while_the_warmer_is_driven_keeps_its_traceback(tmp_path, login, quiet, monkeypatch):
    # A ValueError of the drive (a pydantic error, a JSON decode error) is a bug to debug, not a
    # refusal of what was asked, and does not exit 1 like a void instance.
    def broken(*args, **kwargs):
        raise ValueError("an orchestrator bug")

    monkeypatch.setattr(commands.Orchestrator, "create", broken)
    result = CliRunner().invoke(cli.app, ["provision", "tests.provision.tasks", "--config",
                                          str(write_config(tmp_path, secrets=SECRETS))])
    assert type(result.exception) is ValueError and str(result.exception) == "an orchestrator bug"
    assert "hls provision" not in result.stderr


def _refusal(args: list[str]) -> str:
    """The one line ``hls provision <args>`` prints on stderr when it refuses; the command must
    exit 1 with no traceback."""
    result = CliRunner().invoke(cli.app, ["provision", *args])
    assert result.exit_code == 1 and type(result.exception) is SystemExit, result.output
    [line] = result.stderr.splitlines()
    assert line.startswith("hls provision: ") and TOKEN not in result.output
    return line


@pytest.mark.parametrize("args, says", [
    (["--site.partition", "l40", "tasks.sample_task.task"],
     "expected the task module first, got override flag '--site.partition'. Put <task-module> before any "
     "--<dotted> overrides."),
    (["--site.partition=l40", "tasks.sample_task.task"], "got override flag '--site.partition=l40'"),
    (["..x"], "'..x' is not a dotted module name"),
    (["tasks"], "task module tasks has no file"),
    (["tests.provision.no_such_module"], "cannot import task module tests.provision.no_such_module"),
    (["tests.provision.tasks", "--artifact.image", "x"],
     "takes --site.*, --run.*, --resources.* and --task.* overrides, not artifact.image"),
])
def test_a_refusal_of_what_was_asked_is_one_line(tmp_path, login, quiet, args, says):
    assert says in _refusal([*args, "--config", str(write_config(tmp_path))])


@pytest.mark.parametrize("secrets, says", [
    (SECRETS + "broken = \n", "not valid TOML"),
    (SECRETS.replace("hf_token = ", "short = \"abc\"\nhf_token = "), "must be at least 8 characters"),
    # The token pasted as the grant's right-hand side, where its secret's name belongs.
    (SECRETS.replace('HF_TOKEN = "hf_token"', f'HF_TOKEN = "{TOKEN}"'), "grant stage.provision.HF_TOKEN"),
    # Nested deeper than the parser can recurse, and an integer longer than int() takes.
    (SECRETS + "x = " + "[" * 2000 + "]" * 2000 + "\n", "not valid TOML"),
    (SECRETS + "x = " + "1" * 5000 + "\n", "not valid TOML"),
], ids=["toml", "short", "pasted", "nested", "digits"])
def test_a_refused_secrets_file_is_one_line_that_names_no_value(tmp_path, login, quiet, secrets, says):
    assert says in _refusal(["tests.provision.tasks", "--config", str(write_config(tmp_path, secrets=secrets))])


def test_a_plan_the_site_cannot_run_is_one_line(tmp_path, login, quiet):
    line = _refusal(["tests.provision.tasks", "--config", str(write_config(tmp_path, cpu_partition=None, floor=1))])
    assert "set site.cpu_partition" in line


LOCALS = """
import functools, sys
from pathlib import Path
import typer

# Before 0.23, typer showed every frame's local variables in a traceback by default.
init = typer.Typer.__init__


@functools.wraps(init)
def old_default(self, *args, **kwargs):
    kwargs.setdefault("pretty_exceptions_show_locals", True)
    init(self, *args, **kwargs)


typer.Typer.__init__ = old_default
from hls import cli
from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from tests.pipeline.plans import CODE
commands.code_state = lambda path: CODE
commands.toolchain_here = lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=Path("/"))
from hls.substrate.secrets import load_secrets


def create(plan, scheduler, holder, on_failed):
    # A frame of the drive whose local variable holds a value of the secrets file.
    token = load_secrets(plan.settings.site.secrets_path).secrets["hf_token"]
    raise ValueError(f"the orchestrator failed holding a {len(token)}-character value")


commands.Orchestrator.create = create
cli.app(["provision", "tasks.sample_task.task", "--config", sys.argv[1]], prog_name="hls")
"""


#: Variables that decide whether rich colours a traceback; each test run sets its own.
_COLOUR_VARS = ("FORCE_COLOR", "NO_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE", "TERM", "COLUMNS")
# A terminal escape sequence, such as the colour rich puts around a number.
_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@pytest.mark.parametrize("terminal", [{"FORCE_COLOR": "1"}, {"NO_COLOR": "1", "TERM": "dumb"}],
                         ids=["colour-forced", "no-colour"])
def test_a_traceback_with_a_secret_in_reach_shows_no_value(tmp_path, terminal):
    # CI forces colour, and rich then puts escape sequences inside the text, a value's included,
    # so every check reads the output with them taken out.
    config = write_config(tmp_path, secrets=SECRETS)
    root = commands.checkout_root()
    env = {k: v for k, v in os.environ.items()
           if k not in ("TYPER_STANDARD_TRACEBACK", "_TYPER_STANDARD_TRACEBACK", *_COLOUR_VARS)}
    env.update(terminal, PYTHONPATH=os.pathsep.join([str(root / "src"), str(root)]))
    run = subprocess.run([sys.executable, "-c", LOCALS, str(config)], capture_output=True, text=True, cwd=root,
                         env=env, timeout=120)
    shown = _ESCAPE.sub("", run.stdout + run.stderr)
    assert run.returncode == 1 and "the orchestrator failed holding a 19-character value" in shown, shown[-2000:]
    assert "Traceback" in shown and TOKEN not in shown
