"""What the checkpoint's own template does with a continued assistant turn.

The two-phase turn rests on one claim: the prompt phase 2 sends is the prompt phase 1
sent, plus the text the server produced, plus the closing tag the loop appends. If that
is not exact, the reference teacher-forces a different prompt from the one the artifact
was served and the gap measures the harness rather than the server.

That claim is about the checkpoint's chat template, so nothing here stands in for it.
These tests load the real tokenizer and skip when it is absent, rather than asserting
against a stub that would make the claim look checked.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tasks.agentic_coding.framework import prompt_tokenizer
from tasks.agentic_coding.mini_swe_loop import BASH_TOOL, THINK_CLOSE
from tasks.agentic_coding.task import MODEL_ID


def _snapshot() -> str | None:
    """A local snapshot of the task's checkpoint, or None.

    ``HLS_WEIGHTS_DIR`` first, which is what a graded run mounts and what
    ``framework._mounted_weights`` reads. Otherwise the HuggingFace cache, because the
    tokenizer files are small enough to sit on a developer box without the weights.
    """
    mounted = os.environ.get("HLS_WEIGHTS_DIR")
    if mounted and Path(mounted).is_dir():
        return mounted
    cached = Path.home() / ".cache/huggingface/hub" / (
        "models--" + MODEL_ID.replace("/", "--")) / "snapshots"
    snaps = sorted(p for p in cached.glob("*") if (p / "tokenizer_config.json").is_file())
    return str(snaps[-1]) if snaps else None


@pytest.fixture
def tokenizer(monkeypatch):
    """The real tokenizer, through the loader the task ships."""
    snap = _snapshot()
    if snap is None:
        pytest.skip(f"no local snapshot of {MODEL_ID}; this claim is about its template")
    monkeypatch.setenv("HLS_WEIGHTS_DIR", snap)
    tok = prompt_tokenizer()
    if tok is None:
        pytest.skip(f"{snap} holds no loadable tokenizer")
    return tok


#: One conversation, and the served texts the loop can hand back at the cap. Every one
#: is a real shape: a thought cut mid-word, a thought that closed before the visible
#: half was cut, a command cut mid-call, and the degenerate empty reply.
MESSAGES = [{"role": "user", "content": "Do the thing."}]
CUTS = {
    "thought cut clean": ("I am thinking", THINK_CLOSE),
    "thought cut mid-word": ("I am thinki", THINK_CLOSE),
    "thought closed, visible cut": ("thinking" + THINK_CLOSE + "cat <<EO", ""),
    "closed, call cut mid-key": ("t" + THINK_CLOSE + "<tool_call>bash<arg_key>comm", ""),
    "nothing served": ("", THINK_CLOSE),
}


def _render(tok, messages, **kwargs):
    return tok.apply_chat_template(messages, tools=[BASH_TOOL], tokenize=False,
                                   **kwargs)


@pytest.mark.parametrize("served, appended", list(CUTS.values()), ids=list(CUTS))
def test_the_continuation_prompt_extends_the_first(tokenizer, served, appended):
    """Phase 2's prompt is phase 1's, plus what was served, plus what was appended.

    Byte for byte. This is the whole parity argument: the reference renders the same
    conversation through the same template, so if this holds there is no second render
    to disagree with the first -- the continuation is the first prompt with more text
    after it.
    """
    first = _render(tokenizer, MESSAGES, add_generation_prompt=True)
    carried = served + appended
    second = _render(tokenizer, MESSAGES + [{"role": "assistant", "content": carried}],
                     add_generation_prompt=False, continue_final_message=True)
    assert second == first + carried


@pytest.mark.parametrize("served, appended", list(CUTS.values()), ids=list(CUTS))
def test_the_continuation_ids_extend_the_first(tokenizer, served, appended):
    """The same claim in tokens, which is what the model is actually fed.

    Text equality does not imply it: a tokenizer is free to merge across a boundary,
    and the boundary here is wherever the cap happened to fall -- mid-word as often as
    not. If the ids were not a prefix, the continuation would re-tokenize the tail of
    phase 1 and the two runs would diverge before the model saw a new token.
    """
    first = _render(tokenizer, MESSAGES, add_generation_prompt=True)
    carried = served + appended
    second = _render(tokenizer, MESSAGES + [{"role": "assistant", "content": carried}],
                     add_generation_prompt=False, continue_final_message=True)

    first_ids = tokenizer(first, add_special_tokens=False).input_ids
    second_ids = tokenizer(second, add_special_tokens=False).input_ids
    assert second_ids[:len(first_ids)] == first_ids
    assert len(second_ids) >= len(first_ids)


def test_the_loop_need_not_reconstruct_the_opening_tag(tokenizer):
    """Whether the loop prepends `<think>` to what it carries makes no difference.

    Worth pinning rather than choosing on taste. The template opens the thought in the
    generation prompt, so the served text begins inside one and carries no opener of
    its own. The obvious worry is that an assistant message must therefore supply it,
    and that supplying it would double it. Neither happens: the template normalises the
    two to the same render, so the loop carries exactly what the server sent and
    invents nothing.
    """
    served = "I am thinking"
    without = _render(tokenizer,
                      MESSAGES + [{"role": "assistant",
                                   "content": served + THINK_CLOSE}],
                      add_generation_prompt=False, continue_final_message=True)
    with_tag = _render(tokenizer,
                       MESSAGES + [{"role": "assistant",
                                    "content": "<think>" + served + THINK_CLOSE}],
                       add_generation_prompt=False, continue_final_message=True)
    assert without == with_tag
    assert without.count("<think>") == 1


def test_the_continuation_ends_where_generation_should_resume(tokenizer):
    """The prompt stops at the appended tag: no end-of-turn, no fresh generation
    prompt. A template that closed the assistant turn would make phase 2 a new turn
    rather than a continuation, and the model would restart instead of finishing.
    """
    carried = "I am thinking" + THINK_CLOSE
    second = _render(tokenizer,
                     MESSAGES + [{"role": "assistant", "content": carried}],
                     add_generation_prompt=False, continue_final_message=True)
    assert second.endswith(carried)
