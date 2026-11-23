"""The brief tells the agent how the research loop drives its server and how to run the self-test
the build sandbox carries."""
from __future__ import annotations

from tasks.deepresearch import task as t
from tests.unit.brief_task import render_brief


def _rendered_brief() -> str:
    """The task's brief with its line wrapping collapsed, the way the neighbouring brief tests
    read it."""
    return " ".join(render_brief(t.TASK, t.ARTIFACT.brief)[0].split())


def test_brief_describes_the_loop_and_the_self_test():
    from tasks.deepresearch.selftest_brief import DeepResearchBrief
    assert isinstance(t.ARTIFACT.brief, DeepResearchBrief)
    text = DeepResearchBrief().template_str
    for phrase in ("one chat completion per turn", "selftest.py", "/data/tools/browsecomp",
                   "separate harness container", "never executes a tool"):
        assert phrase in text


def test_brief_states_the_control_token_rule_and_the_chat_format_flag():
    brief, _ = render_brief(t.TASK, t.ARTIFACT.brief)
    normalized = " ".join(brief.split())   # the sentence wraps across lines like its neighbors
    assert ("The `content` you return must be the reply decoded with the model's "
            "control tokens kept (`skip_special_tokens=False`): the loop reads the "
            "tool-call and reasoning delimiters from that text, and a decoder that "
            "drops special tokens drops them too.") in normalized
    assert "--chat-format gemma" in brief


def test_brief_states_the_output_cap_with_its_unit_once():
    # The workload description carries the cap's unit, so a template that appended its
    # own would print it twice.
    brief, _ = render_brief(t.TASK, t.ARTIFACT.brief)
    cap = t.TASK.workload.output_length_cap.max_tokens()
    assert f"- Per-request output cap: {cap} tokens\n" in brief
    assert "tokens tokens" not in brief


def test_brief_states_how_a_tool_result_comes_back():
    """The server is told the harness speaks the model's own tool protocol."""
    text = _rendered_brief()
    assert ("sends the result back as a `tool` message whose `tool_call_id` matches "
            "the `tool_calls` it echoes on your assistant message, together with "
            "your `reasoning_content`, and the tokenizer's chat template renders "
            "all of it.") in text
    assert "<tool_response>" not in text


def test_brief_states_the_admission_the_driver_applies():
    """The task admits every LLM call as it arrives; a run that caps concurrent calls with the
    ``task.agent.max_concurrent_llm`` override is told the cap the driver applies."""
    from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
    from hls.spec.drivers import driver_for
    from hls.taskref import TaskRef, load_task

    default = _rendered_brief()
    assert ("Trajectories are admitted as they arrive, so how many are in flight at once is "
            "whatever the schedule produces rather than a fixed ceiling") in default
    assert "At most" not in default

    capped = load_task(TaskRef(module="tasks.deepresearch.task",
                               overrides={"task.agent.max_concurrent_llm": 8}))
    driver = driver_for(capped.agent, request_timeout_sec=REQUEST_TIMEOUT_SEC,
                        max_context_len=capped.max_context_len,
                        chat_template_kwargs=capped.model.chat_template_kwargs)
    assert driver.max_concurrent_llm == 8
    text = " ".join(render_brief(capped, t.ARTIFACT.brief)[0].split())
    assert ("At most 8 trajectories are in flight at once; turn-over-turn prompts share a "
            "long common prefix.") in text
    assert "rather than a fixed ceiling" not in text
