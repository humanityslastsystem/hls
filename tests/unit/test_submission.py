"""The submission contract: what the brief's submission section tells the agent, which the
agent_build stage renders for the producer's server file, and what a collected commit must hold."""

from __future__ import annotations

import re
from pathlib import Path
import pytest
from hls.spec.submission import SubmissionContract, SubmissionError


def test_brief_section_mentions_pyproject_server_and_main():
    text = SubmissionContract().brief_section()
    assert "pyproject.toml" in text
    assert "server.py" in text
    assert "main" in text


def test_the_section_names_the_producers_server_file():
    """agent_build renders the section for the producer's ``server_rel_path``, so a task whose
    server is another file is told that file and its launch command."""
    text = SubmissionContract(server_filename="main.py").brief_section()
    assert "- `main.py`, launched as `python main.py`." in text
    assert "then `python main.py` in a fresh copy of this environment." in text
    assert "server.py" not in text


def test_validate_passes_on_complete_repo(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "server.py").write_text("x\n")
    SubmissionContract().validate(tmp_path)  # no raise


def test_validate_rejects_missing_files(tmp_path: Path):
    (tmp_path / "server.py").write_text("x\n")
    with pytest.raises(SubmissionError, match="pyproject.toml"):
        SubmissionContract().validate(tmp_path)


def test_validate_reports_all_missing_files(tmp_path: Path):
    with pytest.raises(SubmissionError) as ei:
        SubmissionContract().validate(tmp_path)   # empty dir
    msg = str(ei.value)
    assert "pyproject.toml" in msg and "server.py" in msg


def test_contract_owns_the_canonical_commands():
    c = SubmissionContract()
    assert c.install_command == "pip install ."
    assert c.launch_command == ["python", "server.py"]
    assert SubmissionContract(server_filename="main.py").launch_command == ["python", "main.py"]


def test_brief_states_the_install_rule_and_git_contract():
    brief = SubmissionContract().brief_section()
    assert "pip install ." in brief                 # the install command
    assert "python server.py" in brief              # the launch command
    # Dependency rule: declare in pyproject; undeclared deps are dropped at eval.
    assert "pyproject.toml" in brief and "declaring every dependency" in brief
    assert "absent at eval time" in brief
    # Git contract: must use git, only `main` is evaluated, commit each improvement.
    assert "git repository" in brief and "`main`" in brief
    assert "after every such improvement" in brief


def test_brief_never_mentions_uv_as_a_word():
    # Word-boundary, not substring: "uvicorn" must not trip this.
    brief = SubmissionContract().brief_section()
    assert not re.search(r"\buv\b", brief)
    assert not re.search(r"\buv\b", "uvicorn")  # guard the guard
