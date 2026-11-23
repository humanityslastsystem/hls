"""Baseline -- a reference serving system the task ships (NOT a contestant).

Authored by the task, always prebuilt (a known system, e.g. stock vLLM behind the
task's contract). ``hls baseline-credit`` drives it, served from the baseline image the task
module declares (``IMAGES.baseline``), to measure the task's baseline credit B. It is a
distinct role from a contestant's artifact, hence a distinct type. How that image starts is
declared beside it (``hls.recipes.catalog.Served``); what the task tunes is its
``serve_command``.
"""
from __future__ import annotations

from pydantic import ConfigDict, Field

from hls.core.poly import Spec


class Baseline(Spec):
    """The reference serving system a task's baseline credit B is measured on: the identity of
    its image and the flags it is served with. Both halves are the credit's identity, which
    ``check_campaign`` compares with the ``baseline.json`` a run finds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The reference server image's name, an identity string the baseline credit records; the
    #: image served is the baseline image the task module declares.
    image: str
    #: Extra launch args the server is served with; host and port derive from the task's contract.
    serve_command: list[str] = Field(default_factory=list)
