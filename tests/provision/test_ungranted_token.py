"""An ungranted hub token reaches nothing: with no ``HF_TOKEN`` the role runs with the hub
client's implicit token (a ``token`` file under ``HF_HOME``) switched off, in every library
that reads it, not only in the calls the role makes itself."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hls.provision import role

REPO = Path(__file__).resolve().parents[2]
FAKE_TOKEN = "hf_fake_token_that_sits_in_the_node_home"

# Runs in a fresh interpreter: ``huggingface_hub`` reads its settings once, at import. It prints
# facts about the token, never the token.
SCRIPT = """
import json, os, sys
from pathlib import Path

from hls.provision import role

assert "huggingface_hub" not in sys.modules
if os.environ["RUN_MAIN"] == "1":
    assert role.main(["resolve", "--task-module", "pinned_task", "--out", os.environ["OUT"], "--outputs-out",
                      os.environ["OUT"] + ".digests"]) == 0
from huggingface_hub.utils import build_hf_headers, get_token_to_send

fake = Path(os.environ["HF_HOME"], "token").read_text()
headers = build_hf_headers()
print(json.dumps({
    "implicit_token_sent": get_token_to_send(None) is not None,
    "authorization_header": any(k.lower() == "authorization" for k in headers),
    "value_in_a_header": any(fake in str(v) for v in headers.values()),
}))
"""


@pytest.fixture
def node(tmp_path):
    """A node whose home holds a stored hub token, and a task module that resolves with no hub
    call at all (its model and dataset are pinned to commits)."""
    home = tmp_path / "hf-home"
    home.mkdir()
    (home / "token").write_text(FAKE_TOKEN)
    (tmp_path / "pinned_task.py").write_text("from tests.provision.tasks import SHA, task\n"
                                             "TASK = task(revision=SHA)\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_HUB_DISABLE_IMPLICIT_TOKEN")}
    env.update(HF_HOME=str(home), PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO), str(tmp_path)]),
               OUT=str(tmp_path / "resolved.json"))
    return env


def _run(env, *, run_main: bool):
    done = subprocess.run([sys.executable, "-c", SCRIPT], env={**env, "RUN_MAIN": "1" if run_main else "0"},
                          capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr[-2000:]
    assert FAKE_TOKEN not in done.stdout + done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_without_the_role_a_stored_token_is_sent(node):
    """The control: this node does hand a stored token to every hub request."""
    sent = _run(node, run_main=False)
    assert sent == {"implicit_token_sent": True, "authorization_header": True, "value_in_a_header": True}


def test_with_no_token_granted_a_stored_token_reaches_nothing(node):
    assert _run(node, run_main=True) == {"implicit_token_sent": False, "authorization_header": False,
                                         "value_in_a_header": False}


def _stop_at_the_task(monkeypatch) -> list:
    """Make ``main`` stop when it loads its task, and record the switch's value at that moment."""
    seen = []

    def load_task(module):
        seen.append(os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"])
        raise ValueError("stop here")

    monkeypatch.setattr(role, "load_task", load_task)
    return seen


def _rows_args(tmp_path) -> list[str]:
    """The rows command of the role program, its entry and its digest report in *tmp_path*."""
    return ["rows", "--task-module", "m", "--out", str(tmp_path / "entry"), "--digests-out",
            str(tmp_path / "digests.json")]


def test_main_switches_the_implicit_token_off_before_it_loads_the_task(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")           # restored when the test ends
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    seen = _stop_at_the_task(monkeypatch)
    with pytest.raises(ValueError, match="stop here"):
        role.main(_rows_args(tmp_path))
    assert seen == ["1"]


def test_a_granted_token_leaves_the_switch_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_granted")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
    seen = _stop_at_the_task(monkeypatch)
    with pytest.raises(ValueError, match="stop here"):
        role.main(_rows_args(tmp_path))
    assert seen == ["0"]


def test_main_refuses_to_run_where_the_library_already_read_its_settings(tmp_path, monkeypatch):
    """With no token granted, a library imported before the switch would still send a stored one."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    loaded = _stop_at_the_task(monkeypatch)
    with pytest.raises(RuntimeError, match="already imported"):
        role.main(_rows_args(tmp_path))
    assert loaded == []
