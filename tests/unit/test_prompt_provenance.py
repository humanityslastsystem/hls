"""Tests for per-section brief provenance."""

from __future__ import annotations

import pytest

from hls.prompt import _BLOCK_ORDER, BriefProvenance, ImplementerBrief, SectionProvenance
from hls.spec.contracts import RequestResponseContract
from hls.spec.workload import (
    AllAtOnce,
    NoDelay,
    NoSLO,
    ShortOutputCap,
    Workload,
)
from tests.unit.brief_task import render_brief


class _TS:
    """Minimal TaskSpec stand-in carrying only what the brief reads."""

    class _M:
        def prompt_fields(self) -> dict:
            return {
                "fingerprint": "ckpt@TX",
                "base_checkpoint": "ckpt",
                "arch": "TX",
                "chat_format": "chatml",
            }

    class _HW:
        def prompt_fields(self) -> dict:
            return {
                "accelerator_kind": "B200",
                "accelerator_count": 1,
                "memory_gb_per_device": 0,
                "interconnect": None,
                "cpu_count": 0,
                "memory_gb": 0,
                "storage_gb": 0,
                "allow_internet": False,
            }

    id = "example-task"
    model = _M()
    hardware = _HW()
    agent = None
    contract = RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
    )
    workload = Workload(
        arrival=AllAtOnce(),
        inter_turn_delay=NoDelay(),
        output_length_cap=ShortOutputCap(cap_tokens=64),
        slo=NoSLO(),
    )

    def effective_tenants(self):
        # The brief renderer reads tenants[0].workload. Synthesize the single "t0" tenant.
        from hls.spec.tenant import Tenant

        return [Tenant(id="t0", data_source=None, workload=self.workload, verifiers=[])]


# ---------------------------------------------------------------------------
# Test 1: Base brief (no template_str) -> all 7 blocks source=="base"
# ---------------------------------------------------------------------------


def test_base_brief_all_blocks_source_base() -> None:
    """All 10 blocks should have source=='base' when no template_str is set.

    Three blocks are optional and header-less-by-omission: ``uploads`` (Provided
    Files, task upload folders), ``budget``, and the trailing header-less
    ``directives`` block. With no task hook / section text, they render empty and
    are "base"-sourced like the rest."""
    brief = ImplementerBrief()
    rendered_text, provenance = render_brief(_TS(), brief)

    assert isinstance(provenance, BriefProvenance)
    assert len(provenance.sections) == len(_BLOCK_ORDER) == 12

    block_names = [s.block for s in provenance.sections]
    assert block_names == _BLOCK_ORDER

    for section in provenance.sections:
        assert isinstance(section, SectionProvenance)
        assert section.source == "base", (
            f"Expected source='base' for block '{section.block}', got '{section.source}'"
        )


def test_base_brief_rendered_text_contains_interpolated_values() -> None:
    """Rendered text from base brief should contain interpolated context values."""
    brief = ImplementerBrief()
    rendered_text, provenance = render_brief(_TS(), brief, server_path="/workspace/server.py")

    # Top-level rendering check
    assert "Chat format: `chatml`" in rendered_text  # model block rendered (arch family dropped)
    assert "B200" in rendered_text  # hardware accelerator kind
    assert "/workspace/server.py" in rendered_text  # server path

    # Per-section content checks
    sections_by_block = {s.block: s for s in provenance.sections}

    assert "Chat format: `chatml`" in sections_by_block["model"].rendered
    assert "B200" in sections_by_block["hardware"].rendered
    assert "all sessions dispatched at once" in sections_by_block["workload"].rendered
    assert "/workspace/server.py" in sections_by_block["build_instructions"].rendered


# ---------------------------------------------------------------------------
# Test 2: Child template overriding build_instructions
# ---------------------------------------------------------------------------


def test_child_override_build_instructions_source_task() -> None:
    """When a child overrides build_instructions, that block is source=='task', rest are 'base'."""
    child = '{% extends "brief" %}{% block build_instructions %}CUSTOM BUILD INSTRUCTIONS{% endblock %}'
    brief = ImplementerBrief(template_str=child)
    rendered_text, provenance = render_brief(_TS(), brief)

    sections_by_block = {s.block: s for s in provenance.sections}

    # The overridden block should be source=="task"
    assert sections_by_block["build_instructions"].source == "task"
    assert "CUSTOM BUILD INSTRUCTIONS" in sections_by_block["build_instructions"].rendered

    # All other blocks should be source=="base"
    for block in _BLOCK_ORDER:
        if block != "build_instructions":
            assert sections_by_block[block].source == "base", (
                f"Expected source='base' for block '{block}', got '{sections_by_block[block].source}'"
            )

    # The full rendered text should contain the override
    assert "CUSTOM BUILD INSTRUCTIONS" in rendered_text


def test_child_override_content_visible_in_section_rendered() -> None:
    """The rendered field of the overridden block contains the custom content."""
    child = '{% extends "brief" %}{% block build_instructions %}MY-UNIQUE-OVERRIDE-XYZ{% endblock %}'
    brief = ImplementerBrief(template_str=child)
    _, provenance = render_brief(_TS(), brief)

    sections_by_block = {s.block: s for s in provenance.sections}
    assert "MY-UNIQUE-OVERRIDE-XYZ" in sections_by_block["build_instructions"].rendered

    # The override fully replaces the base block, so base content must be absent.
    assert "python" not in sections_by_block["build_instructions"].rendered.lower()


def test_child_override_other_blocks_still_base_content() -> None:
    """Non-overridden blocks should still contain base content."""
    child = '{% extends "brief" %}{% block build_instructions %}CUSTOM{% endblock %}'
    brief = ImplementerBrief(template_str=child)
    _, provenance = render_brief(_TS(), brief)

    sections_by_block = {s.block: s for s in provenance.sections}
    # model block should still have model fingerprint
    assert "Chat format: `chatml`" in sections_by_block["model"].rendered
    # hardware block should still have hardware info
    assert "B200" in sections_by_block["hardware"].rendered


# ---------------------------------------------------------------------------
# Test 4: BriefProvenance structure
# ---------------------------------------------------------------------------


def test_brief_provenance_sections_are_section_provenance_instances() -> None:
    """BriefProvenance.sections should be a list of SectionProvenance."""
    brief = ImplementerBrief()
    _, provenance = render_brief(_TS(), brief)

    assert hasattr(provenance, "sections")
    assert len(provenance.sections) == 12
    for s in provenance.sections:
        assert hasattr(s, "block")
        assert hasattr(s, "source")
        assert hasattr(s, "rendered")
        assert s.source in ("base", "task")


def test_section_provenance_rendered_is_non_empty_for_most_blocks() -> None:
    """Most blocks should have non-empty rendered text (submission may be empty)."""
    brief = ImplementerBrief()
    _, provenance = render_brief(_TS(), brief)

    sections_by_block = {s.block: s for s in provenance.sections}

    # These blocks must have content in the base brief
    for block in ["goal", "model", "hardware", "workload", "contract", "build_instructions"]:
        assert sections_by_block[block].rendered.strip(), (
            f"Block '{block}' should have non-empty rendered text"
        )


def test_directives_hook_renders_as_trailing_task_block() -> None:
    """A subclass that overrides directives_section() contributes a trailing,
    header-LESS provenance block whose content appears verbatim in the brief and
    is recorded as a 'task'-sourced SectionProvenance (no render_with_provenance
    override needed)."""

    class _WithDirectives(ImplementerBrief):
        def directives_section(self) -> str:
            return "/goal keep improving until the budget runs out."

    text, provenance = render_brief(_TS(), _WithDirectives())

    sections_by_block = {s.block: s for s in provenance.sections}
    directives = sections_by_block["directives"]
    assert directives.source == "task"
    assert "/goal keep improving until the budget runs out." in directives.rendered
    # Emitted verbatim in the brief, with NO "## " heading of its own.
    assert "/goal keep improving until the budget runs out." in text
    assert "## directives" not in text.lower()


def test_directives_block_empty_and_base_by_default() -> None:
    """The default brief has no directives: the block is empty and 'base'-sourced."""
    _, provenance = render_brief(_TS())
    sections_by_block = {s.block: s for s in provenance.sections}
    assert sections_by_block["directives"].rendered == ""
    assert sections_by_block["directives"].source == "base"


def test_heading_slice_covers_every_always_present_block() -> None:
    """Every always-present block must be located by the heading slice with
    non-empty rendered text. This fails if a _BRIEF heading is renamed without
    updating heading_to_block (the slicer would otherwise silently emit
    rendered='' / source='base' for the orphaned block, or the in-code guard
    in render_with_provenance would raise)."""
    brief = ImplementerBrief()
    _, provenance = render_brief(_TS(), brief)

    sections_by_block = {s.block: s for s in provenance.sections}
    for block in ["goal", "model", "hardware", "workload", "contract", "build_instructions", "packaging"]:
        assert block in sections_by_block, f"Block '{block}' missing from provenance sections"
        assert sections_by_block[block].rendered.strip(), (
            f"Block '{block}' has empty rendered text -- heading slice likely missed it"
        )


def test_a_packaging_heading_the_slice_cannot_find_is_refused() -> None:
    """The packaging section always renders, so a heading the slice does not find is one
    renamed without updating heading_to_block: the render refuses rather than recording the
    section as empty."""
    renamed = ImplementerBrief(template_str=(
        '{% extends "brief" %}{% block packaging %}\n\n## Packaging\n'
        '{{ packaging_section }}\n{% endblock %}'))
    with pytest.raises(RuntimeError, match=r"heading slice missed blocks \['packaging'\]"):
        render_brief(_TS(), renamed)
