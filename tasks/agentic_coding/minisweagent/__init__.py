"""A fork of mini-SWE-agent, cut down to the pieces this task drives.

Vendored from https://github.com/SWE-agent/mini-swe-agent at commit
``04d809ceab9df28f9adaed044884180159172930``, upstream version 2.4.6.
MIT licensed; see ``LICENSE`` beside this file for the notice the licence requires
be carried.

Why a fork and not a dependency
-------------------------------

``pip install minisweagent`` would bring rich, typer, litellm, platformdirs and a
config-file loader that reads the user's home directory, none of which belongs in a
sealed build sandbox. More to the point, a dependency cannot be diffed, and a
reimplementation has nothing to compare it against. Vendoring at a named commit means
``diff -r`` answers "what did we change" in one command, and every answer it gives is
either in the divergence list below or a bug.

What is here
------------

The layout mirrors upstream's ``src/minisweagent/`` so paths line up under ``diff -r``:

===================================  ====================================
this fork                            upstream
===================================  ====================================
``__init__.py``                      ``src/minisweagent/__init__.py``
``exceptions.py``                    ``src/minisweagent/exceptions.py``
``agents/default.py``                ``src/minisweagent/agents/default.py``
``models/utils/actions_toolcall.py`` ``src/minisweagent/models/utils/actions_toolcall.py``
``utils/serialize.py``               ``src/minisweagent/utils/serialize.py``
``config/mini.py``                   ``src/minisweagent/config/mini.yaml``
===================================  ====================================

Nothing here imports ``hls``, and nothing imports anything outside the standard
library except pydantic.

That one dependency matters in a build sandbox. pydantic is a core dependency of this
repository, but this package holds no ``hls`` imports so that it can also run inside a
sealed build sandbox, and a self-test that runs it there needs pydantic in that image,
or ``AgentConfig`` and the two adapter configs as plain dataclasses.

The ``Model`` and ``Environment`` protocols are implemented in
``tasks.agentic_coding.mini_swe_loop``, which is the adapter onto the harness. This
package holds no harness knowledge and imports nothing from ``hls``: the same code
runs on the driver during a graded run and inside a build sandbox that has no harness
to import from.

Divergences from upstream
-------------------------

Every one of these is also commented where it appears, with its own reason. Listed
together here so a reader can check the diff against a closed set.

1. **Async.** Upstream's agent is synchronous. The harness is async throughout
   (``await complete(...)``, ``await run_tool(...)``, ``await emit(...)``), so
   ``run``/``step``/``query``/``execute_actions`` and both protocols are ``async
   def``. Control flow and naming are otherwise unchanged, so the diff stays
   readable.
2. **No jinja2 and no YAML.** Upstream renders every template through
   ``jinja2.Template(..., undefined=StrictUndefined)`` and loads them from
   ``mini.yaml``. Here they are Python strings in ``config/mini.py`` rendered with
   ``str.format``, so the package is stdlib plus pydantic. Nothing has a placeholder
   ``str.format`` cannot fill, and both jinja2 and a YAML loader are real
   dependencies to carry into a build sandbox. Upstream's two templates that branch
   in jinja are two strings each, chosen in Python. What ``StrictUndefined`` buys
   is catching a typo'd placeholder; a test that renders every template buys it
   more cheaply.
3. **Tool calls are parsed from the reply text, not from a ``tool_calls`` field.**
   See ``models/utils/actions_toolcall.py``; the reason is the harness's trust model
   and it is written out there.
4. **``cost_limit`` and ``wall_time_limit_seconds`` are disabled**, not deleted, so
   the diff against upstream stays small. See ``agents/default.py``.
5. **No trajectory file: ``output_path``, ``save()`` and ``serialize()`` are gone.**
   The graded record here is the ``TraceEvent`` stream the adapter emits per LLM
   call, so a second serialization of the same conversation would be a redundant
   format to keep in step. Upstream's was also quadratic: it calls ``save`` after
   every step and ``serialize`` deep-merges the whole message list whether or not
   there is a path to write it to, over 250 turns of a transcript reaching hundreds
   of thousands of tokens.
6. **Relative imports** inside the package, where upstream's are absolute from its own
   root. Mechanical, and it keeps the package movable.
"""
from __future__ import annotations

from typing import Any, Protocol

#: Upstream's own version at the forked commit. Provenance only: upstream stamps it
#: into its trajectory file, which is not written here.
__version__ = "2.4.6"

#: The upstream commit this package was cut from. The point of the fork is that it can
#: be diffed, and a diff needs a left-hand side:
#:
#:     git clone https://github.com/SWE-agent/mini-swe-agent
#:     git -C mini-swe-agent checkout 04d809ceab9df28f9adaed044884180159172930
#:     diff -r mini-swe-agent/src/minisweagent tasks/agentic_coding/minisweagent
UPSTREAM_COMMIT = "04d809ceab9df28f9adaed044884180159172930"
UPSTREAM_URL = "https://github.com/SWE-agent/mini-swe-agent"


class Model(Protocol):
    """Protocol for language models.

    Upstream's, with ``query`` awaited. ``format_message`` and
    ``format_observation_messages`` stay synchronous because they only shape dicts.
    Upstream's ``serialize`` is dropped with the trajectory file it fed."""

    config: Any

    async def query(self, messages: list[dict], **kwargs) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...


class Environment(Protocol):
    """Protocol for execution environments.

    Upstream's, with ``execute`` awaited: a command runs in an Apptainer sandbox the
    harness owns, reached over an async tool runner."""

    config: Any

    async def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...


__all__ = ["Environment", "Model", "UPSTREAM_COMMIT", "UPSTREAM_URL", "__version__"]
