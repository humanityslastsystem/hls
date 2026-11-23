"""Does the hand-rolled tool-call parser meet what the checkpoint actually emits?

The grammar is the checkpoint's own, documented in its chat template::

    <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>

A parser written from a hand-made example can only ever meet the example. So the
corpus in ``fixtures/`` is 300 real completions from the live baseline server, and the
edge shapes below are pinned deliberately rather than discovered during a load search.

vLLM ships an independent parser for this same grammar (``vllm/parser/glm47_moe.py``).
It cannot run in this suite because it needs vLLM installed, so the shapes on which
the two must agree are restated here as assertions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tasks.agentic_coding.mini_swe_loop import visible_text
from tasks.agentic_coding.minisweagent.models.utils.actions_toolcall import (
    has_toolcall_delimiter, parse_toolcall_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "glm53flash_emissions.json"


def load_corpus() -> list[dict]:
    return json.loads(FIXTURE.read_text())["replies"]


def parsed(text: str) -> list[tuple[str, dict]]:
    """Name and arguments of every call in a reply, through the loop's own path."""
    return [(c.function.name, json.loads(c.function.arguments))
            for c in parse_toolcall_text(visible_text(text))]


# --- the corpus -------------------------------------------------------------

def test_the_corpus_is_the_size_it_claims():
    corpus = load_corpus()
    assert len(corpus) == json.loads(FIXTURE.read_text())["n"] == 300


def test_every_real_reply_that_carries_a_call_parses_to_one():
    # 298 of 300. The parser must not be the reason a trajectory stalls, so a miss
    # here is a defect rather than a statistic.
    corpus = load_corpus()
    missed = [r for r in corpus
              if has_toolcall_delimiter(visible_text(r["reply"])) and not parsed(r["reply"])]
    # Every miss must be a reply the server cut off mid-call, which is the one honest
    # reason a delimiter yields no action.
    assert all(r["finish_reason"] == "length" for r in missed), \
        [r["reply"][-200:] for r in missed if r["finish_reason"] != "length"]
    assert len(missed) == 2


def test_every_parsed_call_is_bash_with_a_command():
    # The whole corpus was collected with the bash tool offered, and a quarter of it
    # with a second two-argument tool alongside. Nothing invented a different shape.
    for r in load_corpus():
        for name, args in parsed(r["reply"]):
            assert name == "bash", (name, r["reply"][:200])
            assert list(args) == ["command"], (args, r["reply"][:200])
            assert isinstance(args["command"], str) and args["command"]


def test_two_calls_in_a_reply_is_the_common_case():
    # Worth pinning because it sets the arithmetic for trajectory length: a turn is
    # usually two commands, not one. 127 replies carried one call and 171 carried two.
    counts = [len(parsed(r["reply"])) for r in load_corpus()]
    assert counts.count(1) == 127
    assert counts.count(2) == 171
    assert max(counts) == 2


def test_no_real_reply_needed_whitespace_tolerance():
    # Recorded as a fact about this checkpoint, not as a licence to stop tolerating
    # whitespace: the parser accepts it (vLLM's does too) and the next test pins that.
    assert not any("<tool_call>\n" in r["reply"] or "</arg_key>\n" in r["reply"]
                   for r in load_corpus())


# --- edge shapes, pinned deliberately ---------------------------------------

def test_newlines_between_every_tag_still_parse():
    # Not in the corpus, but vLLM's parser for this grammar carries `\s*` between the
    # argument tags, so an independent implementation of the format accepts it.
    text = ("<tool_call>bash\n<arg_key>command</arg_key>\n"
            "<arg_value>ls -la</arg_value>\n</tool_call>")
    assert parsed(text) == [("bash", {"command": "ls -la"})]


def test_a_multi_argument_call_keeps_every_pair_in_order():
    text = ("<tool_call>edit<arg_key>path</arg_key><arg_value>/app/x.py</arg_value>"
            "<arg_key>old</arg_key><arg_value>a</arg_value>"
            "<arg_key>new</arg_key><arg_value>b</arg_value></tool_call>")
    assert parsed(text) == [("edit", {"path": "/app/x.py", "old": "a", "new": "b"})]


def test_a_doubled_opening_delimiter_is_a_format_error_not_a_call():
    # The model can emit `<tool_call><tool_call>bash...`. The decision is to
    # refuse it, and it is not arbitrary -- vLLM's state machine has no transition out
    # of TOOL_NAME on a second TOOL_START either, so the doubled delimiter lands inside
    # the name there too and fails its own tool-name validation. Both implementations
    # read this the same way, and the model gets a repair turn saying so.
    text = ("<tool_call><tool_call>bash\n<arg_key>command</arg_key>\n"
            "<arg_value>cd /app && ls -la</arg_value>\n</tool_call>")
    name, args = parsed(text)[0]
    assert name == "<tool_call>bash"
    assert args == {"command": "cd /app && ls -la"}


def test_an_arg_key_with_no_value_is_a_missing_argument_not_an_unknown_tool():
    # Splitting the body on the full argument pattern would put the stray `<arg_key>`
    # inside the name, and the turn would be refused as an unknown tool, which sends a
    # reader looking at the wrong thing. The name runs to the first `<arg_key>`, as
    # vLLM's does.
    text = "<tool_call>bash<arg_key>command</arg_key></tool_call>"
    assert parsed(text) == [("bash", {})]


def test_a_value_keeps_its_own_whitespace():
    # Keys are stripped and values are not, which is vLLM's rule: a command's leading
    # and trailing spaces are the model's to choose.
    text = ("<tool_call>bash<arg_key> command </arg_key>"
            "<arg_value>  grep -rn 'a  b' .  </arg_value></tool_call>")
    assert parsed(text) == [("bash", {"command": "  grep -rn 'a  b' .  "})]


def test_a_padded_name_is_stripped():
    text = ("<tool_call>  bash  <arg_key>command</arg_key>"
            "<arg_value>ls</arg_value></tool_call>")
    assert parsed(text) == [("bash", {"command": "ls"})]


def test_a_call_with_no_arguments_parses_to_an_empty_dict():
    # vLLM's docstring says calls may have no arguments, so this is the format's, not
    # a malformation. It fails later, on the missing `command`, which is the accurate
    # complaint.
    assert parsed("<tool_call>bash</tool_call>") == [("bash", {})]


def test_an_unterminated_call_yields_nothing():
    assert parsed("<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la") == []


def test_a_complete_call_followed_by_a_truncated_one_keeps_the_complete_one():
    # Six replies in the corpus end this way: the model got one call out and was cut
    # off writing the second. The complete one is a real action and must execute; the
    # truncated one is discarded; and the reply as a whole is NOT refused, which would
    # throw away work the model actually did and cost a repair turn for nothing.
    text = ("<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value>"
            "</tool_call><tool_call>bash<arg_key>command</arg_key><arg_value>pyt")
    assert parsed(text) == [("bash", {"command": "ls"})]


def test_every_such_reply_in_the_corpus_keeps_its_complete_call():
    # The six, found rather than constructed.
    corpus = load_corpus()
    partials = [r for r in corpus
                if visible_text(r["reply"]).count("<tool_call>")
                > visible_text(r["reply"]).count("</tool_call>")]
    assert len(partials) == 8
    # Two of the eight were cut off before any call closed and yield nothing; the
    # other six each keep exactly the calls that did close.
    kept = [len(parsed(r["reply"])) for r in partials]
    assert sorted(kept) == [0, 0, 1, 1, 1, 1, 1, 1]


def test_a_value_may_contain_tag_like_text():
    text = ("<tool_call>bash<arg_key>command</arg_key>"
            "<arg_value>echo '&lt;arg_key&gt;'</arg_value></tool_call>")
    assert parsed(text) == [("bash", {"command": "echo '&lt;arg_key&gt;'"})]


# --- telling our failure from the model's -----------------------------------

def test_prose_with_no_delimiter_is_not_a_parser_suspect():
    assert not has_toolcall_delimiter("I should look at the file first.")


def test_a_delimiter_that_yielded_nothing_is_a_parser_suspect():
    # The distinction that keeps a parser bug from being filed as a model failure.
    cut_off = "Let me look.<tool_call>bash<arg_key>command</arg_key><arg_value>ls"
    assert has_toolcall_delimiter(cut_off)
    assert parsed(cut_off) == []


def test_a_delimiter_inside_the_reasoning_does_not_count():
    # The flag reads the visible text, so a call the model drafted while thinking is
    # neither an action nor evidence against the parser.
    text = "<think>maybe <tool_call>bash</tool_call></think>Actually, let me think more."
    assert not has_toolcall_delimiter(visible_text(text))


# --- round trip against the checkpoint's own template -----------------------

#: The checkpoint, if this machine happens to have it. The round trip is the strongest
#: conformance evidence available -- it checks the parser against the authority in both
#: directions -- but it needs the real ``chat_template.jinja``, so it is skipped rather
#: than approximated with a copy that could drift.
def _checkpoint() -> str | None:
    env = os.environ.get("HLS_WEIGHTS_DIR")
    if env and (Path(env) / "chat_template.jinja").is_file():
        return env
    root = Path.home() / ".cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots"
    if root.is_dir():
        for snap in sorted(root.iterdir()):
            if (snap / "chat_template.jinja").is_file():
                return str(snap)
    return None


needs_checkpoint = pytest.mark.skipif(
    _checkpoint() is None,
    reason="needs the GLM-5.3-Flash snapshot for its real chat template")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_checkpoint(), local_files_only=True)


@needs_checkpoint
def test_parse_render_parse_is_a_fixpoint_on_the_whole_corpus(tokenizer):
    """Every real reply, put back through the template, parses to the same calls.

    The template both teaches the model the format and renders a reply back into the
    next prompt. If our reading and its writing disagreed, the conversation replayed to
    the oracle would drift from the one the model produced -- silently, and further
    every turn."""
    from tasks.agentic_coding.mini_swe_loop import reasoning_text

    checked = 0
    for row in load_corpus():
        reply = row["reply"]
        before = parsed(reply)
        if not before:
            continue
        # The history the loop builds from this reply.
        messages = [{"role": "user", "content": "go"},
                    {"role": "assistant", "content": visible_text(reply),
                     "reasoning_content": reasoning_text(reply)}]
        rendered = tokenizer.apply_chat_template(
            messages, tools=[], tokenize=False, add_generation_prompt=True)
        # Re-read the assistant turn out of the rendered prompt and parse it again.
        after = parsed(rendered)
        assert after == before, (before, after, reply[:200])
        checked += 1
    assert checked >= 290


@needs_checkpoint
def test_the_rendered_prompt_reproduces_the_served_reply_verbatim(tokenizer):
    """The teacher-forcing property, stated directly.

    The bytes the artifact emitted must appear in the next prompt exactly once. This
    is what makes the oracle's replay the same context the artifact was conditioned
    on."""
    from tasks.agentic_coding.mini_swe_loop import reasoning_text

    for row in load_corpus()[:50]:
        reply = row["reply"]
        if not parsed(reply):
            continue
        visible = visible_text(reply)
        messages = [{"role": "user", "content": "go"},
                    {"role": "assistant", "content": visible,
                     "reasoning_content": reasoning_text(reply)}]
        rendered = tokenizer.apply_chat_template(
            messages, tools=[], tokenize=False, add_generation_prompt=True)
        assert rendered.count(visible) == 1, reply[:200]


@needs_checkpoint
def test_the_round_trip_check_bites_when_the_call_is_dropped(tokenizer):
    """A check that only ever agrees proves nothing.

    This is the defect the fixpoint test exists to catch -- a history that dropped the
    call out of the content -- shown failing on purpose."""
    from tasks.agentic_coding.minisweagent.models.utils.actions_toolcall import (
        strip_toolcalls,
    )

    reply = next(r["reply"] for r in load_corpus() if parsed(r["reply"]))
    broken = [{"role": "user", "content": "go"},
              {"role": "assistant", "content": strip_toolcalls(visible_text(reply))}]
    rendered = tokenizer.apply_chat_template(
        broken, tools=[], tokenize=False, add_generation_prompt=True)
    assert parsed(reply) != parsed(rendered)
    assert parsed(rendered) == []
