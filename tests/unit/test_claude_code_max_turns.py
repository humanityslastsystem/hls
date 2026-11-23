"""Tests: ClaudeCode max_turns=None omits the --max-turns flag."""
from pathlib import Path

from harbor.agents.installed.claude_code import ClaudeCode


def test_none_max_turns_omits_flag():
    """ClaudeCode(max_turns=None) should omit --max-turns from CLI flags."""
    a = ClaudeCode(
        logs_dir=Path("/tmp"),
        model_name="m",
        max_turns=None,
        inherit_process_env=False,
    )
    assert "--max-turns" not in a.build_cli_flags()


def test_int_max_turns_present():
    """ClaudeCode(max_turns=7) should include --max-turns 7 in CLI flags."""
    a = ClaudeCode(
        logs_dir=Path("/tmp"),
        model_name="m",
        max_turns=7,
        inherit_process_env=False,
    )
    assert "--max-turns 7" in a.build_cli_flags()
