"""Unit tests for the shared console line layout + color gate."""
from __future__ import annotations

import io
import logging

from hls.obs.logsetup import (
    _HANDLER_TAG,
    configure_logging,
    format_console_line,
    supports_color,
)


def test_format_line_has_time_label_context_message():
    line = format_console_line(0.0, "INFO", "provisioning", "downloading weights X@main")
    parts = line.split()  # whitespace-collapsed: [time, label, "provisioning:", "downloading", ...]
    assert parts[1] == "INFO"                                   # label in its own column
    assert "provisioning: downloading weights X@main" in line   # context: message body
    assert line[2] == ":" and line[5] == ":"                    # leading HH:MM:SS timestamp


def test_format_line_omits_colon_when_message_empty():
    line = format_console_line(0.0, "INFO", "hls.pipeline.orchestrator", "")
    assert line.endswith("hls.pipeline.orchestrator")
    assert "hls.pipeline.orchestrator:" not in line  # no dangling colon


def test_format_line_color_wraps_only_the_label():
    plain = format_console_line(0.0, "WARNING", "x", "y", color=False)
    colored = format_console_line(0.0, "WARNING", "x", "y", color=True)
    assert "\033[" not in plain
    assert "\033[33m" in colored and "\033[0m" in colored
    # the message body is NOT colored
    assert colored.endswith("x: y")


def test_supports_color_gates(monkeypatch):
    class _TTY:
        def isatty(self): return True

    class _NotTTY:
        def isatty(self): return False

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert supports_color(_TTY()) is True
    assert supports_color(_NotTTY()) is False

    monkeypatch.setenv("NO_COLOR", "1")
    assert supports_color(_TTY()) is False

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert supports_color(_TTY()) is False


def test_configure_logging_emits_formatted_line_to_stream():
    buf = io.StringIO()
    configure_logging(verbose=False, stream=buf)
    logging.getLogger("hls.provision.rows").info("downloading weights X@main")
    out = buf.getvalue()
    # short name: the leading "hls." is stripped to the subsystem
    assert "INFO" in out
    assert "provision.rows: downloading weights X@main" in out


def test_configure_logging_is_idempotent():
    buf = io.StringIO()
    configure_logging(stream=buf)
    configure_logging(stream=buf)
    hls_logger = logging.getLogger("hls")
    ours = [h for h in hls_logger.handlers if getattr(h, _HANDLER_TAG, False)]
    assert len(ours) == 1  # not stacked


def _console_handler(name: str = "hls") -> logging.Handler:
    return next(h for h in logging.getLogger(name).handlers if getattr(h, _HANDLER_TAG, False))


def test_configure_logging_levels():
    """The threshold lives on the handler, not the logger.

    An owned tree may pin its own level for its own reasons, and a level set on an
    ancestor is not consulted once a descendant has one. The trees are therefore opened
    to DEBUG and the handler decides, which is the only point every tree passes through.
    """
    buf = io.StringIO()
    configure_logging(verbose=False, stream=buf)
    assert _console_handler().level == logging.INFO
    assert logging.getLogger("hls").level == logging.DEBUG  # open; handler filters
    logging.getLogger("hls.pipeline.orchestrator").debug("hidden detail")
    assert "hidden detail" not in buf.getvalue()

    configure_logging(verbose=True, stream=buf)
    assert _console_handler().level == logging.DEBUG
    logging.getLogger("hls.pipeline.orchestrator").debug("verbose detail")
    assert "verbose detail" in buf.getvalue()


def _harbor_env_logger():
    """A logger built the way harbor builds every environment and agent logger.

    `BaseEnvironment` does `global_logger.getChild(__name__)`, so the real name is
    `harbor.utils.logger.harbor.environments...` and its effective level comes from
    harbor's own root, which is pinned at DEBUG. A test that invents a plain
    `harbor.environments.*` name exercises a tree harbor never writes to, and would
    pass while production did the opposite."""
    from harbor.utils.logger import logger as harbor_root

    return harbor_root.getChild("harbor.environments.singularity.singularity")


def test_harbor_records_reach_the_console():
    """harbor is ours, not third-party noise: a sandbox that dies mid-run reports it
    through harbor's logger, and that must not fall through to the root's WARNING."""
    buf = io.StringIO()
    configure_logging(verbose=False, stream=buf)
    _harbor_env_logger().error("Exec channel lost: server killed by signal 9 (SIGKILL)")
    assert "signal 9 (SIGKILL)" in buf.getvalue()


def test_harbor_is_idempotent_and_quiet_until_verbose():
    buf = io.StringIO()
    configure_logging(stream=buf)
    configure_logging(stream=buf)
    harbor_logger = logging.getLogger("harbor")
    assert len([h for h in harbor_logger.handlers if getattr(h, _HANDLER_TAG, False)]) == 1
    # The container's own stdout is debug-level chatter, so it stays hidden by default
    # and appears under --verbose, where it carries the server's dying traceback.
    _harbor_env_logger().debug("[server] chatter")
    assert "chatter" not in buf.getvalue()
    configure_logging(verbose=True, stream=buf)
    _harbor_env_logger().debug("[server] traceback")
    assert "traceback" in buf.getvalue()
