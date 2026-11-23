"""Tests for ClaudeCode._build_run_env() env-dict construction."""
from pathlib import Path


from harbor.agents.installed.claude_code import ClaudeCode


def _make_agent(tmp_path: Path) -> ClaudeCode:
    return ClaudeCode(logs_dir=tmp_path)


def test_build_run_env_forwards_disable_telemetry_vars(tmp_path, monkeypatch):
    """_build_run_env() must forward DISABLE_TELEMETRY, DISABLE_ERROR_REPORTING,
    DISABLE_BUG_COMMAND, and ANTHROPIC_BASE_URL from os.environ."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")
    monkeypatch.setenv("DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DISABLE_ERROR_REPORTING", "1")
    monkeypatch.setenv("DISABLE_BUG_COMMAND", "1")

    agent = _make_agent(tmp_path)
    env = agent._build_run_env()

    assert env.get("DISABLE_TELEMETRY") == "1", (
        "DISABLE_TELEMETRY should be forwarded when set in os.environ"
    )
    assert env.get("DISABLE_ERROR_REPORTING") == "1", (
        "DISABLE_ERROR_REPORTING should be forwarded when set in os.environ"
    )
    assert env.get("DISABLE_BUG_COMMAND") == "1", (
        "DISABLE_BUG_COMMAND should be forwarded when set in os.environ"
    )
    assert env.get("ANTHROPIC_BASE_URL") == "https://proxy.example.com", (
        "ANTHROPIC_BASE_URL should be forwarded"
    )


def test_build_run_env_does_not_include_unset_disable_vars(tmp_path, monkeypatch):
    """_build_run_env() must NOT include DISABLE_* keys when they are absent."""
    monkeypatch.delenv("DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("DISABLE_ERROR_REPORTING", raising=False)
    monkeypatch.delenv("DISABLE_BUG_COMMAND", raising=False)
    # Provide minimal auth so the dict is non-empty
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")

    agent = _make_agent(tmp_path)
    env = agent._build_run_env()

    assert "DISABLE_TELEMETRY" not in env
    assert "DISABLE_ERROR_REPORTING" not in env
    assert "DISABLE_BUG_COMMAND" not in env


def test_build_run_env_always_sets_nonessential_traffic(tmp_path, monkeypatch):
    """CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC must always be '1'."""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-test")

    agent = _make_agent(tmp_path)
    env = agent._build_run_env()

    assert env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC") == "1"
