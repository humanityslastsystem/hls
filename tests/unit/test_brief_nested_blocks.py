"""The block nested inside Build Instructions: a task replaces only the dev-reference
sentence and inherits the rest. Decoding is a headed section of its own, so a task
overriding it replaces that section and leaves Build Instructions alone. Every task's brief
overrides only blocks the base brief defines and reads only names its render context binds:
Jinja drops any other block and renders any other name as nothing, both without an error."""

from __future__ import annotations

import importlib
from pathlib import Path

import jinja2.meta
import jinja2.nodes
import pytest

import hls.prompt
from hls.prompt import (_BRIEF, _ENV, _NESTED_BLOCKS, ImplementerBrief, _defined_blocks, _nested_block_parents,
                        _with_enclosing_blocks)
from tests.unit.brief_task import TASK, render_brief

#: Every task module of the checkout: each ``task*.py`` of a directory under ``tasks/``.
_TASK_MODULES = sorted(f"tasks.{path.parent.name}.{path.stem}"
                       for path in (Path(__file__).resolve().parents[2] / "tasks").glob("*/task*.py"))


def _task():
    from tasks.sample_task.task import TASK
    return TASK


def _dropped_blocks(template_str: str) -> set[str]:
    """The blocks *template_str*, a template extending the base brief, defines at its top level
    and the base does not. Jinja renders a child's top-level block only in place of the base's
    block of that name. A block nested in another of the child's blocks renders where it sits."""
    ast = _ENV.parse(template_str)
    if ast.find(jinja2.nodes.Extends) is None:
        return set()
    top = _defined_blocks(template_str) - _nested_block_parents(template_str).keys()
    return top - _defined_blocks(_BRIEF)


#: The names a brief's render context binds, which the agent_build stage gives every brief.
_BOUND = set(ImplementerBrief()._context(
    taskspec=TASK, server_path="/workspace/server.py", submission_section="", packaging_section="",
    budget_section="", uploads_section="", blocked_packages=None, reference_service=False)) | set(_ENV.globals)


def _unbound_names(template_str: str) -> set[str]:
    """The names *template_str* reads that neither it nor the render context binds."""
    return jinja2.meta.find_undeclared_variables(_ENV.parse(template_str)) - _BOUND


def _task_briefs() -> list[tuple[str, ImplementerBrief]]:
    """Each task module's brief, by the module's name; the modules that declare no brief are
    left out."""
    briefs = []
    for name in _TASK_MODULES:
        brief = getattr(getattr(importlib.import_module(name), "ARTIFACT", None), "brief", None)
        if brief is not None:
            briefs.append((name, brief))
    return briefs


def test_every_task_module_is_found():
    assert {"tasks.agentic_coding.task", "tasks.deepresearch.task", "tasks.sample_task.task"} \
        <= set(_TASK_MODULES)
    assert len(_task_briefs()) == len(_TASK_MODULES)


@pytest.mark.parametrize("module", _TASK_MODULES)
def test_a_tasks_brief_overrides_only_blocks_the_base_defines(module):
    """A section a task writes in a block the base does not define never reaches the
    contestant, and nothing says so."""
    brief = dict(_task_briefs())[module]
    if brief.template_str is not None:
        assert _dropped_blocks(brief.template_str) == set()


@pytest.mark.parametrize("module", _TASK_MODULES)
def test_a_tasks_brief_reads_only_names_the_context_binds(module):
    """A name the context does not bind renders as nothing: a number goes missing from its
    sentence, and a section behind an `if` on it is never stated."""
    brief = dict(_task_briefs())[module]
    assert _unbound_names(_BRIEF) == set()
    if brief.template_str is not None:
        assert _unbound_names(brief.template_str) == set()


def test_a_block_the_base_does_not_define_is_found():
    child = ('{% extends "brief" %}{% block packages %}{{ super() }}RULE{% endblock %}'
             '{% block goal %}GOAL{% block own %}OWN{% endblock %}{% endblock %}')
    assert _dropped_blocks(child) == {"packages"}
    # Jinja drops it without an error; the block nested in an override renders.
    text, _ = render_brief(TASK, ImplementerBrief(template_str=child))
    assert "RULE" not in text and "GOALOWN" in text
    assert _dropped_blocks('{% extends "brief" %}{% block package_policy %}{{ super() }}{% endblock %}') == set()


def test_a_name_the_context_does_not_bind_is_found():
    child = '{% extends "brief" %}{% block goal %}up to {{ turn_limit }}{% if service %}S{% endif %}{% endblock %}'
    assert _unbound_names(child) == {"turn_limit", "service"}
    assert _unbound_names('{% extends "brief" %}{% block goal %}{{ turn_bound }}{% endblock %}') == set()


def test_nested_block_map_is_derived_from_the_base_template():
    assert _NESTED_BLOCKS == {"accelerators": "build_instructions",
                              "dev_reference_note": "build_instructions",
                              "health": "contract",
                              "package_policy": "build_instructions",
                              "toolchain": "build_instructions"}


def test_enclosing_blocks_are_followed_all_the_way_out():
    parents = {"c": "b", "b": "a"}
    assert _with_enclosing_blocks({"c"}, parents) == {"a", "b", "c"}
    assert _with_enclosing_blocks({"a"}, parents) == {"a"}
    assert _with_enclosing_blocks(set(), parents) == set()


def test_a_task_overriding_a_block_two_levels_down_sources_the_headed_section(monkeypatch):
    """`_nested_block_parents` walks to any depth, so the base brief may grow a chain
    longer than the one level it has today; the map is patched to build one here."""
    monkeypatch.setattr(hls.prompt, "_NESTED_BLOCKS",
                        {"dev_reference_note": "decoding", "decoding": "build_instructions"})
    child = ImplementerBrief(template_str=(
        '{% extends "brief" %}{% block dev_reference_note %}Rows are rows.{% endblock %}'))
    _, prov = render_brief(_task(), child)
    assert {s.block: s for s in prov.sections}["build_instructions"].source == "task"


def test_default_brief_keeps_the_chat_decoding_paragraph():
    text, prov = render_brief(_task())
    sections = {s.block: s for s in prov.sections}
    build = sections["build_instructions"]
    assert "chat-templated prompt already" in sections["decoding"].rendered
    assert sections["decoding"].source == "base"
    # The dev-reference sentence ends the Data bullet: its last words, a newline, then the
    # next bullet. Asserting across that newline is what catches a lost line break -- the
    # bullet alone still matches when the two run together.
    assert "Answer:`).\n- `pip install`:" in build.rendered
    assert build.source == "base"


def test_a_task_can_override_the_decoding_paragraph_alone():
    child = ImplementerBrief(template_str=(
        '{% extends "brief" %}{% block decoding %}**Decoding.** Feed the prompt raw.{% endblock %}'))
    text, prov = render_brief(_task(), child)
    sections = {s.block: s for s in prov.sections}
    assert "Feed the prompt raw." in sections["decoding"].rendered
    assert "chat-templated prompt already" not in sections["decoding"].rendered
    assert sections["decoding"].source == "task"
    # Its own section, so Build Instructions is untouched and still the base text.
    assert "### Python" in sections["build_instructions"].rendered
    assert sections["build_instructions"].source == "base"


def test_a_task_can_override_the_dev_reference_sentence_alone():
    child = ImplementerBrief(template_str=(
        '{% extends "brief" %}{% block dev_reference_note %}The served prompt wraps a row; '
        '`fim_prompt.txt` beside the rows says how.{% endblock %}'))
    text, prov = render_brief(_task(), child)
    build = {s.block: s for s in prov.sections}["build_instructions"]
    assert "`fim_prompt.txt` beside the rows says how." in build.rendered
    assert "prompt_prefix.txt" not in build.rendered
    # The rest of the Data bullet, and of the block, is inherited.
    assert "the scored test split is withheld." in build.rendered
    assert "- `pip install`: any package is installable; nothing is blocked." in build.rendered
    # The overriding sentence ends the Data bullet, so the newline before the next bullet
    # must survive. The block's closing tag carries `+` for this; without it the two run
    # together and every assertion above still passes.
    assert "says how.\n- `pip install`:" in build.rendered
    assert build.source == "task"
