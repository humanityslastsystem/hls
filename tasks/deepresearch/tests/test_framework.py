"""Unit tests for DeepResearchFramework -- the deepresearch agentic loop."""

from __future__ import annotations
import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hls.core.dataitem import DatasetItem
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.spec.drivers import HarnessContext
from hls.spec.tools import LocalCallableToolRunner, ToolSpec
from tasks.deepresearch.framework import (
    CONTINUE_RESEARCH_PROMPT, MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN,
    MAX_OPEN_GUIDANCE_REPEATS, MAX_REPEATED_QUERY,
    MAX_REPEATED_TOOL_CALL, MAX_SEARCHES_WITHOUT_OPEN_BEFORE_FINAL,
    MAX_TOOL_RESPONSE_DOC_CHARS, MIN_FORCED_FINAL_TOKENS,
    INVALID_OPEN_REPAIR_PROMPT, LENGTH_NO_SEARCH_REPAIR_PROMPT,
    OPEN_AFTER_SEARCH_PROMPT,
    REPEATED_QUERY_REPAIR_PROMPT,
    REPEATED_TOOL_REPAIR_PROMPT, SEARCH_TOOL_NAME, DeepResearchFramework,
    _last_resort_forced_answer,
)
from tasks.deepresearch.research_loop import (
    AFTER_SEARCH_DECISION_PROMPT, BROWSER_TOOLS, DEEPRESEARCH_SYSTEM,
)


def _call(name: str, **args) -> str:
    """A Gemma 4 tool call as the model writes it: a string value between ``<|"|>`` tokens,
    any other value as JSON."""
    body = ",".join(f'{key}:<|"|>{value}<|"|>' if isinstance(value, str)
                    else f"{key}:{json.dumps(value)}" for key, value in args.items())
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


def _thought(text: str) -> str:
    """A Gemma 4 thought channel the reply opens and closes itself."""
    return f"<|channel>thought\n{text}<channel|>"


def _seed_item(question: str = "Who founded HLS?") -> DatasetItem:
    env = Envelope(request_id="s0", contents=[TextPayload(text=question)])
    return DatasetItem(request_id="s0", turns=(env,), ground_truth_turns=("RulinShao",))


def _research_seed_item(question: str = "Who founded HLS?") -> DatasetItem:
    """A seed item shaped the way the dataset shapes one: the system prompt and the
    question, pre-rendered on the turn's metadata."""
    env = Envelope(request_id="s0", contents=[TextPayload(text=question)],
                   metadata={"messages": [
                       {"role": "system", "content": DEEPRESEARCH_SYSTEM},
                       {"role": "user", "content": question}]})
    return DatasetItem(request_id="s0", turns=(env,), ground_truth_turns=("RulinShao",))


def _ctx(max_output_tokens: int | None = None, max_context_len: int | None = None) -> HarnessContext:
    """The context the trajectory driver hands a framework.

    ``max_output_tokens`` is the tenant's declared per-request output cap and
    ``max_context_len`` the task's context window. The loop's budget guards read them from
    here and never from a reply, so a test that wants those guards armed says what they
    are, exactly as the driver does."""
    return HarnessContext(tenant_id="t0", t_send=0.0,
                          max_output_tokens=max_output_tokens, max_context_len=max_context_len)


@dataclass
class _ScriptedLLM:
    responses: list[str | tuple[str, dict]]
    calls: int = 0
    inputs: list[Envelope] = field(default_factory=list)

    async def __call__(self, env: Envelope) -> ArtifactResponse:
        self.inputs.append(env)
        i = self.calls
        self.calls += 1
        item = self.responses[i]
        if isinstance(item, tuple):
            text, metadata = item
        else:
            text, metadata = item, {}
        return ArtifactResponse(
            output=Envelope(request_id=env.request_id,
                            contents=[TextPayload(text=text)],
                            metadata=metadata))


def _tool_runner(search_results: list[dict]):
    """LocalCallableToolRunner whose `search` returns canned results in order."""
    idx = {"i": 0}

    def search(query: str, **_) -> dict:
        i = idx["i"]
        idx["i"] += 1
        out = dict(search_results[i]) if i < len(search_results) else {"documents": []}
        out.setdefault("query", query)
        return out

    return LocalCallableToolRunner(
        specs=[ToolSpec(name=SEARCH_TOOL_NAME, description="d",
                        json_schema={"type": "object"})],
        callables={SEARCH_TOOL_NAME: search},
    )


def _counted_prompt(n_tokens: int):
    """A harness-side prompt counter that reports ``n_tokens`` for every prompt.

    NOT a mock standing in for something real, and not removable. The loop does not
    trust a reply's ``usage.prompt_tokens`` at all -- that number is the contestant's,
    and an inflated one would force every trajectory to answer on its first turn --
    so it counts the prompt it sent instead, and caps what a reply claims at the
    prompt's byte length wherever it cannot count. A test that wants the loop to
    believe it is 60,000 tokens into a conversation therefore has two options: build a
    conversation that long, or state the count here. This is the same seam the
    framework installs the checkpoint's tokenizer through
    (``DeepResearchFramework.prompt_counter``)."""
    async def count(messages: list[dict], tools: list[dict] | None) -> int:
        return n_tokens
    return count


def _tool_message_payload(messages: list[dict]) -> dict:
    """The evidence payload of the tool message that answers the last call."""
    assert messages[-2]["role"] == "tool"
    return json.loads(messages[-2]["content"])


async def _run(framework, llm, tool_runner, max_turns=8, seed_item=None,
               max_output_tokens=None, max_context_len=None):
    events: list[TraceEvent] = []

    async def on_event(ev):
        events.append(ev)

    await framework.run_trajectory(
        seed_item=seed_item or _seed_item(), llm_call=llm,
        tool_runner=tool_runner, on_event=on_event, ctx=_ctx(max_output_tokens, max_context_len),
        max_turns=max_turns,
    )
    return events


# ---- happy paths ----------------------------------------------------------

def test_immediate_boxed_answer_terminates_after_one_turn():
    llm = _ScriptedLLM(responses=["The founder is RulinShao. \\boxed{RulinShao}"])
    tr = _tool_runner([])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert llm.calls == 1
    assert len(events) == 1
    assert events[0].extras["final_answer"] == "RulinShao"
    assert events[0].extras["stop_reason"] == "answer"
    assert "final_text" in events[0].extras
    # ground_truth is plumbed onto the final-answer event so the metric grades.
    assert events[0].ground_truth == "RulinShao"


def test_placeholder_boxed_answer_does_not_terminate():
    llm = _ScriptedLLM(responses=[
        "Follow the format `\\boxed{ANSWER}`. " + _call("web_search", query="HLS founder"),
        "Now I know. \\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert len(events) == 2
    assert events[0].extras["stop_reason"] == "search"
    assert events[1].extras["final_answer"] == "RulinShao"


def test_unavailable_boxed_answer_does_not_terminate():
    llm = _ScriptedLLM(responses=[
        "I searched but \\boxed{Unable to locate the specific source}",
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[0].extras["stop_reason"] == "noop"
    assert "final_answer" not in events[0].extras
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_long_reasoning_boxed_answer_does_not_terminate():
    llm = _ScriptedLLM(responses=[
        "\\boxed{Based on my searches I need to identify the person from the "
        "question constraints before making a final answer}",
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[0].extras["stop_reason"] == "noop"
    assert "final_answer" not in events[0].extras
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_empty_boxed_answer_does_not_terminate():
    llm = _ScriptedLLM(responses=[
        "Output format: \\boxed{}. " + _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert events[0].extras["stop_reason"] == "search"
    assert events[1].extras["final_answer"] == "RulinShao"


def test_search_then_boxed_answer_two_turns():
    llm = _ScriptedLLM(responses=[
        "I should look this up. " + _call("web_search", query="HLS founder"),
        "Now I know. \\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert llm.calls == 2
    assert len(events) == 2

    search_ev = events[0]
    assert search_ev.extras["stop_reason"] == "search"
    assert search_ev.extras["n_queries"] == 1
    assert search_ev.extras["tool_calls"][0]["args"] == {"query": "HLS founder"}
    assert search_ev.extras["tool_results"][0]["result"]["documents"]
    assert "elapsed_sec" in search_ev.extras["tool_results"][0]
    assert search_ev.ground_truth == "RulinShao"

    answer_ev = events[1]
    assert answer_ev.extras["final_answer"] == "RulinShao"
    assert answer_ev.turn_index == 1


def test_the_tool_message_is_a_compact_evidence_payload():
    docs = [f"doc-{i}" for i in range(12)]
    long_doc = "x" * 1500
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    raw_result = {
        "documents": [long_doc] + docs,
        "raw_provider_payload": "NOISY" * 100,
    }
    tr = _tool_runner([raw_result])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    messages = llm.inputs[1].metadata["messages"]
    payload = _tool_message_payload(messages)
    assert payload == {
        "query": "HLS founder",
        "documents": ["x" * 1500] + [f"doc-{i}" for i in range(9)],
    }
    guidance = messages[-1]["content"]
    assert "<search_history>" in guidance
    assert "Do not repeat these exact web_search queries" in guidance
    assert "Prefer opened primary or credible sources over snippets" in guidance
    assert "NOISY" not in messages[-2]["content"] and "NOISY" not in guidance
    assert events[0].extras["tool_results"][0]["result"]["raw_provider_payload"]
    assert events[0].extras["tool_results"][0]["result"]["documents"][0] == long_doc
    assert events[1].extras["final_answer"] == "RulinShao"


def test_the_tool_message_preserves_browser_page_budget():
    long_page = "[2]\n" + ("source line with evidence\n" * 1000)
    llm = _ScriptedLLM(responses=[
        _call("web_open", cursor=1, id=2),
        "\\boxed{RulinShao}",
    ])
    tr = LocalCallableToolRunner(
        specs=[ToolSpec(name="open", description="open",
                        json_schema={"type": "object"})],
        callables={"open": lambda **_: {"query": "open", "documents": [long_page]}},
    )

    asyncio.run(_run(DeepResearchFramework(), llm, tr))

    payload = _tool_message_payload(llm.inputs[1].metadata["messages"])
    assert len(payload["documents"][0]) <= MAX_TOOL_RESPONSE_DOC_CHARS
    assert len(payload["documents"][0]) > len(long_page) - 100
    assert len(payload["documents"][0]) > 1200
    assert payload["documents"][0].startswith("[2] source line")


def test_the_tool_message_formats_raw_organic_results_when_needed():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{
        "organic": [{
            "title": "HLS profile",
            "snippet": "RulinShao founded HLS.",
            "link": "https://example.test/hls",
        }]
    }])

    asyncio.run(_run(DeepResearchFramework(), llm, tr))

    tool_response = llm.inputs[1].metadata["messages"][-2]["content"]
    assert "HLS profile: RulinShao founded HLS." in tool_response
    assert "Source: https://example.test/hls" in tool_response


def test_browser_open_and_find_tool_calls_are_supported():
    llm = _ScriptedLLM(responses=[
        _call("web_open", id=1),
        _call("web_find", pattern="RulinShao"),
        "\\boxed{RulinShao}",
    ])
    tr = LocalCallableToolRunner(
        specs=[
            ToolSpec(name=SEARCH_TOOL_NAME, description="search",
                     json_schema={"type": "object"}),
            ToolSpec(name="open", description="open",
                     json_schema={"type": "object"}),
            ToolSpec(name="find", description="find",
                     json_schema={"type": "object"}),
        ],
        callables={
            SEARCH_TOOL_NAME: lambda query, **_: {"documents": []},
            "open": lambda id, **_: {"query": f"open {id}", "documents": ["opened"]},
            "find": lambda pattern, **_: {"query": f"find {pattern}", "documents": ["found"]},
        },
    )

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert [e.extras["stop_reason"] for e in events] == ["search", "search", "answer"]
    assert events[0].extras["tool_calls"][0] == {
        "name": "open", "args": {"id": 1}, "model_name": "web_open", "model_args": {"id": 1}}
    assert events[1].extras["tool_calls"][0] == {
        "name": "find", "args": {"pattern": "RulinShao"},
        "model_name": "web_find", "model_args": {"pattern": "RulinShao"}}
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_repeated_searches_without_open_force_open_find_tools_next_turn():
    searches = []
    for i in range(MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN):
        searches.append(
            _call("web_search", query=f"HLS founder query {i}")
        )
    llm = _ScriptedLLM(responses=searches + ["\\boxed{RulinShao}"])
    tr = _tool_runner([
        {"documents": [f"[{i + 1}] L1: \u30101\u2020Result\u3011 L2: RulinShao founded HLS."]}
        for i in range(MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN)
    ])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=6))

    guarded_event = events[MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN - 1]
    assert guarded_event.extras["open_required_next_turn"] is True
    assert OPEN_AFTER_SEARCH_PROMPT in guarded_event.extras["tool_response_text"]
    next_tool_names = [
        tool["function"]["name"]
        for tool in llm.inputs[MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN]
        .metadata["tools"]
    ]
    assert next_tool_names == ["web_search", "web_open", "web_find"]
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_repeated_searches_without_open_force_finalization_budget():
    searches = [
        _call("web_search", query=f"HLS founder query {i}")
        for i in range(MAX_SEARCHES_WITHOUT_OPEN_BEFORE_FINAL)
    ]
    llm = _ScriptedLLM(responses=searches + ["\\boxed{RulinShao}"])
    tr = _tool_runner([
        {"documents": [f"[{i + 1}] L1: \u30101\u2020Result\u3011 L2: RulinShao founded HLS."]}
        for i in range(MAX_SEARCHES_WITHOUT_OPEN_BEFORE_FINAL)
    ])

    events = asyncio.run(_run(
        DeepResearchFramework(), llm, tr,
        max_turns=MAX_SEARCHES_WITHOUT_OPEN_BEFORE_FINAL + 2,
    ))

    assert events[-1].extras["final_answer"] == "RulinShao"
    assert llm.inputs[-1].metadata["tools"] == []
    assert llm.inputs[-1].metadata["finalize_reason"] == "search_without_open_budget"
    assert sum(
        1 for e in events
        if e.extras.get("open_required_next_turn")
    ) == MAX_OPEN_GUIDANCE_REPEATS


def test_a_trajectory_emits_exactly_one_event_per_llm_call():
    """A salvaged answer rides on the turn that produced it. A second event for the
    same call would let a per-call verifier score those tokens twice."""
    responses = [
        _call("web_search", query="HLS founder"),
        (
            'Based on my searches, I found references to:\n'
            '- "HLS founder formed in high school guitarist died 2003"\n'
            'But I still need the specific answer.'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    ids = [ev.request_id for ev in events]
    assert len(ids) == len(set(ids))
    assert not any(rid.endswith(".fallback") for rid in ids)
    assert len(events) == llm.calls == 2
    assert events[-1].extras["answer_format"] == "forced_empty_fallback"
    assert events[-1].extras["stop_reason"] == "answer"


def test_forced_final_fallback_rejects_question_fragment_answer():
    responses = [
        _call("web_search", query="HLS founder"),
        (
            'Based on my searches, I found references to:\n'
            '- "HLS founder formed in high school guitarist died 2003"\n'
            'But I still need the specific answer.'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[-1].extras["answer_format"] == "forced_empty_fallback"
    assert events[-1].extras["final_answer"] == "No concrete answer extracted"


def test_forced_final_guess_strips_explanatory_suffix():
    responses = [
        _call("web_search", query="HLS founder"),
        (
            'Based on my searches, I still need to reconcile several clues. '
            'The singer may be from a group that debuted before 2018, and '
            'the clues mention a zodiac sign and a showcase. After weighing '
            'the evidence, the best answer is "Dragon" as the nickname, '
            'given the zodiac connection.'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[-1].extras["answer_format"] == "forced_guess_fallback"
    assert events[-1].extras["final_answer"] == "Dragon"


def test_forced_final_guess_extracts_charge_candidate():
    responses = [
        _call("web_search", query="case charge"),
        (
            'Based on my searches, the snippets are incomplete and I still '
            'need to match the lawyer and sentencing details. One result says '
            'the defendant pleaded guilty to malice murder. I need more '
            'evidence before I can be certain.'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[-1].extras["answer_format"] == "forced_guess_fallback"
    assert events[-1].extras["final_answer"] == "malice murder"


def test_forced_final_guess_extracts_book_candidate():
    responses = [
        _call("web_search", query="history book"),
        (
            'From my searches, several books fit the date range. The book '
            'could be "The Rise and Fall of the Third Reich" by William Shirer, '
            'but I cannot verify the article yet.'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[-1].extras["answer_format"] == "forced_guess_fallback"
    assert events[-1].extras["final_answer"] == "The Rise and Fall of the Third Reich"


def test_forced_final_guess_skips_result_reference_for_later_candidate():
    responses = [
        _call("web_search", query="soap actor"),
        (
            'Based on my searches, I can see references to:\n'
            '- Result 17 mentions "first child of four siblings poetry"\n'
            'The most likely candidate is a well-known soap opera actor. '
            'One possibility is someone like:\n'
            '- John J. York\n'
            'The most promising lead appears to be John J. York, who has '
            'connections to soap operas.'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[-1].extras["answer_format"] == "forced_guess_fallback"
    assert events[-1].extras["final_answer"] == "John J. York"


def test_forced_final_guess_strips_or_the_suffix():
    responses = [
        _call("web_search", query="band guitarist"),
        (
            'Based on my searches, the evidence is still mixed across several '
            'snippets and I need to identify the guitarist. The best answer is '
            '"Shujaa" or the guitarist\'s name might be "Shujaa".'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[-1].extras["answer_format"] == "forced_guess_fallback"
    assert events[-1].extras["final_answer"] == "Shujaa"


def test_forced_final_guess_rejects_generic_description():
    responses = [
        _call("web_search", query="writer pseudonym"),
        (
            'Based on my searches, the best answer is a writer who wrote under '
            'a pseudonym, but I do not have a specific name yet.'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[-1].extras["answer_format"] == "forced_empty_fallback"
    assert events[-1].extras["final_answer"] == "No concrete answer extracted"


def test_forced_final_guess_strips_markdown_emphasis():
    responses = [
        _call("web_search", query="architect firm"),
        (
            'Based on my searches, I need to identify the firm. The best answer '
            'is **D2 Architects Vietnam**.'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[-1].extras["answer_format"] == "forced_guess_fallback"
    assert events[-1].extras["final_answer"] == "D2 Architects Vietnam"


def test_forced_final_guess_rejects_sentence_fragment():
    responses = [
        _call("web_search", query="chapter title"),
        (
            'Based on my searches, the best answer is that the chapter title is '
            '"Secrets" or something very similar.'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[-1].extras["answer_format"] == "forced_empty_fallback"
    assert events[-1].extras["final_answer"] == "No concrete answer extracted"


def test_forced_final_guess_rejects_generic_someone_phrase():
    responses = [
        _call("web_search", query="weights inspired"),
        (
            'Based on my searches, the best answer is someone who wore weights '
            'inspired, but no specific name was found.'
        ),
    ]
    llm = _ScriptedLLM(responses=responses)
    tr = _tool_runner([{"documents": ["[1] L1: \u30101\u2020Result\u3011 L2: weak snippet"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=2))

    assert events[-1].extras["answer_format"] == "forced_empty_fallback"
    assert events[-1].extras["final_answer"] == "No concrete answer extracted"


def test_invalid_open_result_reprompts_with_result_id_guidance():
    llm = _ScriptedLLM(responses=[
        _call("web_open", id=4),
        "\\boxed{RulinShao}",
    ])
    tr = LocalCallableToolRunner(
        specs=[ToolSpec(name="open", description="open",
                        json_schema={"type": "object"})],
        callables={
            "open": lambda **_: {
                    "query": "open",
                    "documents": [
                        "Error: link id=4 not found. Use browser.search first."
                    ],
            },
        },
    )

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    tool_response = events[0].extras["tool_response_text"]
    assert "dagger result numbers" in tool_response
    assert "they are not L-line numbers" in tool_response
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_invalid_open_id_is_rejected_before_browser_execution():
    llm = _ScriptedLLM(responses=[
        _call("web_open", id=21),
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    calls = {"open": 0, "search": 0}

    def open_page(**_):
        calls["open"] += 1
        return {"query": "open", "documents": ["should not happen"]}

    def search(**_):
        calls["search"] += 1
        return {"query": "search", "documents": ["RulinShao founded HLS."]}

    tr = LocalCallableToolRunner(
        specs=[
            ToolSpec(name="open", description="open",
                     json_schema={"type": "object"}),
            ToolSpec(name=SEARCH_TOOL_NAME, description="search",
                     json_schema={"type": "object"}),
        ],
        callables={"open": open_page, SEARCH_TOOL_NAME: search},
    )

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert calls == {"open": 0, "search": 1}
    assert events[0].extras["stop_reason"] == "loop_guard"
    assert events[0].extras["invalid_open_id"] == 21
    assert llm.inputs[1].metadata["messages"][-1]["content"] == INVALID_OPEN_REPAIR_PROMPT
    assert [t["function"]["name"] for t in llm.inputs[1].metadata["tools"]] == [
        "web_search"]
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_open_with_known_search_cursor_is_repaired_to_top_result():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        _call("web_open", id=11),
        "\\boxed{RulinShao}",
    ])
    seen_open_args = []

    def search(**_):
        return {
            "query": "search",
            "documents": ["[11] L1: Searched for HLS founder L2: L3: "
                          "\u30101\u2020HLS founder source\u3011 L4: URL: https://example.test"],
        }

    def open_page(**kwargs):
        seen_open_args.append(kwargs)
        return {"query": "open", "documents": ["opened result"]}

    tr = LocalCallableToolRunner(
        specs=[
            ToolSpec(name=SEARCH_TOOL_NAME, description="search",
                     json_schema={"type": "object"}),
            ToolSpec(name="open", description="open",
                     json_schema={"type": "object"}),
        ],
        callables={SEARCH_TOOL_NAME: search, "open": open_page},
    )

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert seen_open_args == [{"cursor": 11, "id": 1, "_session_id": "s0"}]
    # The loop repairs its own call; the call as the model wrote it stays as written.
    assert events[1].extras["tool_calls"][0] == {
        "name": "open", "args": {"cursor": 11, "id": 1},
        "model_name": "web_open", "model_args": {"id": 11}}
    assert events[1].extras["stop_reason"] == "search"
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_repeated_invalid_open_id_keeps_steering_back_to_search():
    llm = _ScriptedLLM(responses=[
        _call("web_open", id=21),
        _call("web_open", cursor=21, id=21),
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    calls = {"search": 0}

    def search(**_):
        calls["search"] += 1
        return {"query": "search", "documents": ["RulinShao founded HLS."]}

    tr = LocalCallableToolRunner(
        specs=[
            ToolSpec(name="open", description="open",
                     json_schema={"type": "object"}),
            ToolSpec(name=SEARCH_TOOL_NAME, description="search",
                     json_schema={"type": "object"}),
        ],
        callables={
            "open": lambda **_: {"query": "open", "documents": []},
            SEARCH_TOOL_NAME: search,
        },
    )

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert calls["search"] == 1
    assert [e.extras["stop_reason"] for e in events] == [
        "loop_guard", "loop_guard", "search", "answer"]
    assert [t["function"]["name"] for t in llm.inputs[1].metadata["tools"]] == [
        "web_search"]
    assert [t["function"]["name"] for t in llm.inputs[2].metadata["tools"]] == [
        "web_search"]
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_disallowed_tool_call_is_rejected_when_turn_is_search_only():
    llm = _ScriptedLLM(responses=[
        _call("web_open", id=21),
        _call("web_open", id=1),
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    calls = {"open": 0, "search": 0}

    def open_page(**_):
        calls["open"] += 1
        return {"query": "open", "documents": ["should not be called"]}

    def search(**_):
        calls["search"] += 1
        return {"query": "HLS founder", "documents": ["RulinShao founded HLS."]}

    tr = LocalCallableToolRunner(
        specs=[
            ToolSpec(name="open", description="open",
                     json_schema={"type": "object"}),
            ToolSpec(name=SEARCH_TOOL_NAME, description="search",
                     json_schema={"type": "object"}),
        ],
        callables={"open": open_page, SEARCH_TOOL_NAME: search},
    )

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert calls == {"open": 0, "search": 1}
    assert events[0].extras["invalid_open_id"] == 21
    assert events[1].extras["disallowed_tool_call"] == {
        "name": "open", "args": {"id": 1}, "model_name": "web_open", "model_args": {"id": 1}}
    assert events[1].extras["allowed_tools"] == ["search"]
    assert [t["function"]["name"] for t in llm.inputs[1].metadata["tools"]] == [
        "web_search"]
    assert [t["function"]["name"] for t in llm.inputs[2].metadata["tools"]] == [
        "web_search"]
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_repeated_non_search_tool_call_gets_repair_then_forced_final():
    responses = [
        _call("web_open", cursor=1, id=3)
    ] * (MAX_REPEATED_TOOL_CALL + 2)
    responses.append("\\boxed{RulinShao}")
    llm = _ScriptedLLM(responses=responses)
    calls = {"open": 0}

    def open_page(**_):
        calls["open"] += 1
        return {"query": "open", "documents": ["same opened page"]}

    tr = LocalCallableToolRunner(
        specs=[ToolSpec(name="open", description="open",
                        json_schema={"type": "object"})],
        callables={"open": open_page},
    )

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=8))

    assert calls["open"] == MAX_REPEATED_TOOL_CALL
    assert events[MAX_REPEATED_TOOL_CALL].extras["stop_reason"] == "loop_guard"
    assert events[MAX_REPEATED_TOOL_CALL].extras["repeated_tool_call"] == {
        "name": "open", "args": {"cursor": 1, "id": 3},
        "model_name": "web_open", "model_args": {"cursor": 1, "id": 3}}
    assert llm.inputs[MAX_REPEATED_TOOL_CALL + 1].metadata["messages"][-1]["content"] == (
        REPEATED_TOOL_REPAIR_PROMPT)
    assert llm.inputs[-1].metadata["finalize_reason"] == "repeated_tool_call"
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_tool_calls_receive_hidden_session_id_for_stateful_browser_tools():
    seen: list[tuple[str, str | None]] = []

    async def search(query, _session_id=None):
        seen.append((query, _session_id))
        return {"query": query, "documents": ["result"]}

    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    tr = LocalCallableToolRunner(
        specs=[ToolSpec(name=SEARCH_TOOL_NAME, description="search",
                        json_schema={"type": "object"})],
        callables={SEARCH_TOOL_NAME: search},
    )

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert seen == [("HLS founder", "s0")]
    assert events[0].extras["tool_calls"][0]["args"] == {"query": "HLS founder"}
    assert "_session_id" not in events[0].extras["tool_calls"][0]["args"]


def test_final_text_after_the_thought_terminates_as_answer():
    llm = _ScriptedLLM(responses=[_thought("Done.") + "\nRulinShao"])
    tr = _tool_runner([])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert len(events) == 1
    assert events[0].extras["stop_reason"] == "answer"
    assert events[0].extras["final_answer"] == "RulinShao"


def test_plain_non_tool_text_terminates_as_answer():
    llm = _ScriptedLLM(responses=["RulinShao"])
    tr = _tool_runner([])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert len(events) == 1
    assert events[0].extras["stop_reason"] == "answer"
    assert events[0].extras["final_answer"] == "RulinShao"


def test_search_planning_text_is_not_final_answer():
    llm = _ScriptedLLM(responses=[
        _thought("Still uncertain.") + "\n"
        "The search results are not giving me the specific details. "
        "Let me try searching for more information with different terms."
    ])
    tr = _tool_runner([])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=1))

    assert len(events) == 1
    assert events[0].extras["stop_reason"] == "noop"
    assert "final_answer" not in events[0].extras
    assert events[0].ground_truth == "RulinShao"


def test_a_placeholder_search_tag_is_repaired_without_searching_query_literal():
    llm = _ScriptedLLM(responses=[
        "Thinking Process... use <search>QUERY</search>",
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=4))

    assert [e.extras["stop_reason"] for e in events] == ["noop", "search", "answer"]
    assert events[0].extras["n_queries"] == 0
    assert all(
        call["args"]["query"] != "QUERY"
        for ev in events
        for call in ev.extras.get("tool_calls", [])
    )
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_a_placeholder_query_is_repaired_without_tool_execution():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="QUERY"),
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=4))

    assert [e.extras["stop_reason"] for e in events] == ["noop", "search", "answer"]
    assert events[0].extras["n_queries"] == 0
    assert events[1].extras["tool_calls"][0]["args"]["query"] == "HLS founder"
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_initial_noop_gets_one_tool_enabled_research_repair():
    llm = _ScriptedLLM(responses=[
        _thought("I need to research this."),
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=4))

    assert [e.extras["stop_reason"] for e in events] == ["noop", "search", "answer"]
    assert llm.inputs[1].metadata["messages"][-1]["content"] == CONTINUE_RESEARCH_PROMPT
    assert llm.inputs[1].metadata["tools"][0]["function"]["name"] == "web_search"
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_repeated_initial_noop_forces_boxed_final_answer():
    llm = _ScriptedLLM(responses=[
        _thought("I need to research this."),
        _thought("Still not using a tool."),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=4))

    assert [e.extras["stop_reason"] for e in events] == ["noop", "noop", "answer"]
    assert llm.inputs[2].metadata["tools"] == []
    assert llm.inputs[2].metadata["finalize_reason"] == "initial_noop"
    assert "Do not call web_search again" in llm.inputs[2].metadata["messages"][-1]["content"]


def test_noop_after_search_gets_one_tool_enabled_research_repair():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        _thought("I found it but forgot to answer."),
        _call("web_search", query="HLS founder exact answer"),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([
        {"documents": ["RulinShao founded HLS."]},
        {"documents": ["Exact evidence confirms RulinShao."]},
    ])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=5))

    assert [e.extras["stop_reason"] for e in events] == ["search", "noop", "search", "answer"]
    assert events[-1].extras["final_answer"] == "RulinShao"
    assert llm.inputs[2].metadata["messages"][-1]["content"] == CONTINUE_RESEARCH_PROMPT
    assert llm.inputs[2].metadata["tools"][0]["function"]["name"] == "web_search"


def test_length_stop_noop_after_search_forces_final_answer_without_repair_search():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        (
            "I am still thinking and never emitted a tool call or answer.",
            {
                "finish_reason": "length",
                "completion_tokens": 32768,
                "request_max_tokens": 32768,
            },
        ),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=5))

    assert [e.extras["stop_reason"] for e in events] == ["search", "noop", "answer"]
    assert llm.inputs[2].metadata["tools"] == []
    assert llm.inputs[2].metadata["finalize_reason"] == "length_stop_after_search"
    assert "Do not call web_search again" in llm.inputs[2].metadata["messages"][-1]["content"]
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_initial_length_stop_noop_gets_search_only_repair_before_final_answer():
    llm = _ScriptedLLM(responses=[
        (
            "I am still thinking and never emitted a tool call or answer.",
            {
                "completion_tokens": 1024,
                "request_max_tokens": 1024,
            },
        ),
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["HLS was founded by RulinShao"]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=4))

    assert [e.extras["stop_reason"] for e in events] == ["noop", "search", "answer"]
    assert [
        tool["function"]["name"] for tool in llm.inputs[1].metadata["tools"]
    ] == ["web_search"]
    assert llm.inputs[1].metadata["messages"][-1]["content"] == (
        LENGTH_NO_SEARCH_REPAIR_PROMPT)
    assert "finalize_reason" not in llm.inputs[1].metadata
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_a_calibration_pool_members_reply_drives_the_length_stop_branch():
    """The same branch on the metadata a trusted pool member reports.

    A calibration drive runs this framework against an in-process engine rather than an
    HTTP server, so the reply's serving facts come from ``_served_metadata`` instead of
    the wire decoder. They have to carry the loop past the same fork: a reply the output
    cap cut off gets the length repair turn, not the ordinary continue prompt."""
    from hls.reference.engines import _served_metadata

    turn = Envelope(request_id="s0.1", metadata={"max_tokens": 8192})
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        ("I am still thinking and never emitted a tool call or answer.",
         _served_metadata(turn, prompt_tokens=40_000, completion_tokens=8192,
                          max_context_len=131_072)),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=5))

    assert [e.extras["stop_reason"] for e in events] == ["search", "noop", "answer"]
    assert llm.inputs[2].metadata["finalize_reason"] == "length_stop_after_search"
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_length_stop_detection_accepts_max_tokens_metadata_fallback():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        (
            "I am still thinking and never emitted a tool call or answer.",
            {
                "completion_tokens": 32768,
                "max_tokens": 32768,
            },
        ),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=5))

    assert [e.extras["stop_reason"] for e in events] == ["search", "noop", "answer"]
    assert llm.inputs[2].metadata["tools"] == []
    assert llm.inputs[2].metadata["finalize_reason"] == "length_stop_after_search"
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_forced_final_length_stop_emits_best_effort_answer_event():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        (
            _thought("I am still thinking until the cap."),
            {
                "completion_tokens": 32768,
                "max_tokens": 32768,
            },
        ),
        (
            _thought("I am still thinking on the forced-final turn too."),
            {
                "completion_tokens": 32768,
                "max_tokens": 32768,
            },
        ),
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=5))

    assert llm.calls == len(events) == 3
    assert [e.extras["stop_reason"] for e in events] == ["search", "noop", "answer"]
    assert events[-1].extras["final_answer"] == "No concrete answer extracted"
    assert events[-1].extras["answer_format"] == "forced_empty_fallback"
    assert events[-1].extras["finalize_reason"] == "length_stop_after_search"
    assert events[-1].extras["length_stop"] is True


def test_forced_final_tool_call_emits_best_effort_answer_event():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        _call("web_search", query="ignored forced-final tool"),
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(
        DeepResearchFramework(finalize_after_search_turns=1),
        llm, tr, max_turns=5,
    ))

    assert llm.calls == len(events) == 2
    assert [e.extras["stop_reason"] for e in events] == ["search", "answer"]
    assert events[-1].extras["final_answer"] == "No concrete answer extracted"
    assert events[-1].extras["answer_format"] == "forced_empty_fallback"
    assert events[-1].extras["finalize_reason"] == "search_turn_budget"


def test_second_noop_after_search_forces_final_answer_without_tools():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        _thought("I found it but forgot to answer."),
        _thought("Still narrating instead of using the tool."),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=5))

    assert [e.extras["stop_reason"] for e in events] == ["search", "noop", "noop", "answer"]
    assert llm.inputs[2].metadata["messages"][-1]["content"] == CONTINUE_RESEARCH_PROMPT
    assert llm.inputs[2].metadata["tools"]
    assert "Do not call web_search again" in llm.inputs[3].metadata["messages"][-1]["content"]
    assert llm.inputs[3].metadata["tools"] == []
    assert llm.inputs[3].metadata["finalize_reason"] == "noop_after_search"


def test_search_threshold_prompts_final_answer_without_tools():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(
        DeepResearchFramework(finalize_after_search_turns=1), llm, tr, max_turns=3))

    assert [e.extras["stop_reason"] for e in events] == ["search", "answer"]
    assert llm.inputs[1].metadata["tools"] == []
    assert "Do not call web_search again" in llm.inputs[1].metadata["messages"][-1]["content"]


def test_default_policy_does_not_force_after_small_fixed_search_count():
    call = (
        _call("web_search", query="HLS founder")
    )
    llm = _ScriptedLLM(responses=[call, call, "\\boxed{RulinShao}"])
    tr = _tool_runner([
        {"documents": ["RulinShao founded HLS."]},
        {"documents": ["More evidence."]},
    ])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=5))

    assert [e.extras["stop_reason"] for e in events] == ["search", "search", "answer"]
    assert llm.inputs[1].metadata["tools"]
    assert "Do not call web_search again" not in llm.inputs[1].metadata["messages"][-1]["content"]


def test_context_budget_prompts_final_answer_without_tools():
    llm = _ScriptedLLM(responses=[
        (
            _call("web_search", query="HLS founder"),
            {"prompt_tokens": 900},
        ),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(
        DeepResearchFramework(finalize_context_utilization=0.85),
        llm, tr, max_turns=10, max_context_len=1000))

    assert [e.extras["stop_reason"] for e in events] == ["search", "answer"]
    assert llm.inputs[1].metadata["tools"] == []
    assert llm.inputs[1].metadata["finalize_reason"] == "context_budget"


def test_a_reported_context_window_cannot_shorten_the_trajectory():
    """A server's own ``hls_context.max_context_len`` decides nothing.

    The context guard stops a trajectory researching and makes it answer, and the task
    scores trajectory makespan, so a window read off the reply would be a server's
    lever on its own score: report one token of context and every trajectory answers
    on turn one. The window is declared, and a reply claiming a different one is
    ignored -- here a server claims a 1-token window against a declaration that leaves
    it 130,172 tokens of room, and the loop keeps searching."""
    dishonest = {"prompt_tokens": 900, "max_context_len": 1,
                 "request_max_tokens": 1}
    llm = _ScriptedLLM(responses=[
        (_call("web_search", query="HLS founder"), dishonest),
        (_call("web_search", query="HLS history"), dishonest),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]},
                       {"documents": ["HLS began in 2025."]}])

    events = asyncio.run(_run(
        DeepResearchFramework(), llm, tr, max_turns=10, max_context_len=131_072))

    assert [e.extras["stop_reason"] for e in events] == ["search", "search", "answer"]
    # Still holding the browser on every follow-up turn, and never told to answer.
    assert all(env.metadata["tools"] for env in llm.inputs[1:])
    assert not any("finalize_reason" in env.metadata for env in llm.inputs)


def test_a_reported_prompt_length_cannot_shorten_the_trajectory():
    """The harness counts its own prompt, so the reply's ``usage`` decides nothing.

    An inflated ``prompt_tokens`` is the other half of the same lever a small reported
    context window is: it puts the conversation over the finalize threshold on turn one, and
    the trajectory answers instead of researching. Here a server claims 130,000 tokens
    against a 131,072 window -- comfortably past 0.85 of it -- while the harness counts
    the 900 the prompt actually came to, and the loop keeps searching."""
    dishonest = {"prompt_tokens": 130_000, "request_max_tokens": 256}
    llm = _ScriptedLLM(responses=[
        (_call("web_search", query="HLS founder"), dishonest),
        (_call("web_search", query="HLS history"), dishonest),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]},
                       {"documents": ["HLS began in 2025."]}])
    framework = DeepResearchFramework()
    framework._prompt_token_count = _counted_prompt(900)

    events = asyncio.run(_run(framework, llm, tr, max_turns=10, max_context_len=131_072))

    assert [e.extras["stop_reason"] for e in events] == ["search", "search", "answer"]
    assert all(env.metadata["tools"] for env in llm.inputs[1:])
    assert not any("finalize_reason" in env.metadata for env in llm.inputs)


def test_without_a_tokenizer_a_reported_prompt_length_is_capped_at_its_bytes(monkeypatch):
    """The fallback where no checkpoint is mounted: the reply's figure, bounded.

    No tokenizer emits more tokens than the prompt has bytes, so the bound cannot bite
    an honest server. Here the same 130,000-token claim rides on a prompt of a few hundred
    bytes and buys nothing."""
    monkeypatch.delenv("HLS_WEIGHTS_DIR", raising=False)
    dishonest = {"prompt_tokens": 130_000, "request_max_tokens": 256}
    llm = _ScriptedLLM(responses=[
        (_call("web_search", query="HLS founder"), dishonest),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])
    framework = DeepResearchFramework()
    # The counter is installed and reports that it cannot count, which is what sends
    # the loop to the bound.
    assert asyncio.run(framework.prompt_counter({})([], None)) is None

    events = asyncio.run(_run(framework, llm, tr, max_turns=10, max_context_len=131_072))

    assert [e.extras["stop_reason"] for e in events] == ["search", "answer"]
    assert llm.inputs[1].metadata["tools"]
    assert "finalize_reason" not in llm.inputs[1].metadata


class _RecordingTokenizer:
    """Enough of a tokenizer to see how the counter renders, and nothing more."""

    def __init__(self) -> None:
        self.template_calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        return "".join(str(m.get("content") or "") for m in messages)

    def __call__(self, text, add_special_tokens=True):
        class _Out:
            input_ids = list(range(len(text)))
        assert add_special_tokens is False, "the prompt is already rendered"
        return _Out()


def test_the_prompt_count_is_the_render_a_pool_member_is_served(monkeypatch):
    """What makes the count mean anything: it is the same render.

    A trusted calibration member reports ``prompt_tokens`` as the length of
    ``hls.reference.engines.chat_prompt_ids`` over the turn's messages, its tools and
    the model's chat-template pin. The counter goes through that same function, so a
    graded trajectory and a calibration trajectory are budgeted on one number rather
    than on two that happen to be close. An empty tool list normalizes to no tools, as
    the engines' own ``tools_of`` does, or a forced-final turn would render with a
    tool block the member never sends."""
    from hls.core.model import ModelSpec

    tokenizer = _RecordingTokenizer()
    monkeypatch.setattr("tasks.deepresearch.framework.prompt_tokenizer",
                        lambda: tokenizer)
    model = ModelSpec(base_checkpoint="x", arch="TX", chat_format="gemma",
                      chat_template_kwargs={"enable_thinking": True})
    framework = DeepResearchFramework()
    framework.check_model(model)
    count = framework.prompt_counter(model.chat_template_kwargs)

    messages = [{"role": "user", "content": "hello"}]
    assert asyncio.run(count(messages, [])) == len("hello")
    assert asyncio.run(count(messages, list(BROWSER_TOOLS))) == len("hello")

    no_tools, with_tools = tokenizer.template_calls
    assert "tools" not in no_tools
    assert with_tools["tools"] == list(BROWSER_TOOLS)
    for call in (no_tools, with_tools):
        assert call["enable_thinking"] is True
        assert call["add_generation_prompt"] is True and call["tokenize"] is False


def test_one_llm_call_emits_one_event():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=5))

    ids = [e.request_id for e in events]
    assert len(ids) == len(set(ids)) == llm.calls


def test_a_second_event_for_one_request_id_fails_loudly(monkeypatch):
    """The one-event-per-call rule is enforced where events leave the loop.

    Seven salvage sites end a trajectory early, and each is followed by a return; a
    site that was not would rebuild an already-emitted turn's event from scratch, and a
    duplicate offers the same served tokens to a per-call verifier twice. That is a
    scoring error nothing downstream would notice, so it raises rather than passing
    quietly. Two turns forced onto one request id is the collision itself."""
    import tasks.deepresearch.research_loop as rl
    built = rl._turn
    monkeypatch.setattr(
        rl, "_turn",
        lambda session_id, turn_idx, *a, **kw: built(session_id, 0, *a, **kw))
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    with pytest.raises(RuntimeError, match="second event for request 's0.0'"):
        asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=5))


def test_a_reported_output_cap_cannot_shorten_the_trajectory():
    """The third number the stopping rule must not take from the reply.

    The request-budget guard asks whether the next turn's answer would still fit
    beside the prompt, so a server claiming a huge ``request_max_tokens`` makes every
    turn look like an overflow and forces the trajectory to answer immediately. The cap
    is the harness's: the driver stamps the tenant's ``output_length_cap`` on every
    turn and passes it on the ``HarnessContext``. Here a server claims 130,000 against
    a declared 8,192, and the trajectory researches to its turn budget."""
    dishonest = {"request_max_tokens": 130_000}
    llm = _ScriptedLLM(responses=[
        (_call("web_search", query=f"HLS founder {i}"), dishonest) for i in range(3)
    ] + ["\\boxed{RulinShao}"])
    tr = _tool_runner([{"documents": [f"fact {i}"]} for i in range(3)])
    framework = DeepResearchFramework()
    framework._prompt_token_count = _counted_prompt(1_800)

    events = asyncio.run(_run(framework, llm, tr, max_turns=10,
                              max_output_tokens=8_192, max_context_len=131_072))

    assert [e.extras["stop_reason"] for e in events] == [
        "search", "search", "search", "answer"]
    assert all(env.metadata["tools"] for env in llm.inputs[1:])
    assert not any("finalize_reason" in env.metadata for env in llm.inputs)


def test_the_window_and_the_pin_come_with_each_trajectory_not_with_the_framework():
    """The loop's window is ``TaskSpec.max_context_len``, which ``workload_payload`` records,
    and the drive hands it, with the model's chat-template pin, to each trajectory on its
    ``HarnessContext``. The framework holds neither, so it reports neither."""
    framework = DeepResearchFramework()
    assert "max_context_len" not in framework.traffic_config()
    assert not {"_max_context_len", "_chat_template_kwargs"} & set(vars(framework))
    seen = []

    class _Loop:
        async def run(self, **kwargs):
            pass

    def loop(max_output_tokens=None, *, max_context_len=None, chat_template_kwargs=None):
        seen.append((max_output_tokens, max_context_len, dict(chat_template_kwargs)))
        return _Loop()

    framework.loop = loop
    ctx = HarnessContext(tenant_id="t0", t_send=0.0, max_output_tokens=256,
                         max_context_len=131_072, chat_template_kwargs={"enable_thinking": True})

    async def on_event(event):
        pass

    asyncio.run(framework.run_trajectory(seed_item=_seed_item(), llm_call=None, tool_runner=None,
                                         on_event=on_event, ctx=ctx, max_turns=1))
    assert seen == [(256, 131_072, {"enable_thinking": True})]
    assert framework.loop is loop and DeepResearchFramework().loop().max_context_len is None


def test_request_budget_prompts_final_answer_with_reduced_cap():
    llm = _ScriptedLLM(responses=[
        (
            _call("web_search", query="HLS founder"),
            {},
        ),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])
    framework = DeepResearchFramework()
    framework._prompt_token_count = _counted_prompt(62_000)

    events = asyncio.run(_run(framework, llm, tr, max_turns=10,
                              max_output_tokens=65_536, max_context_len=131_072))

    assert [e.extras["stop_reason"] for e in events] == ["search", "answer"]
    assert llm.inputs[1].metadata["tools"] == []
    assert llm.inputs[1].metadata["finalize_reason"] == "request_budget"
    assert 0 < llm.inputs[1].metadata["max_tokens"] < 64_976


def test_request_budget_final_cap_never_exceeds_remaining_context():
    framework = DeepResearchFramework()

    assert framework._forced_final_max_tokens(
        {"prompt_tokens": 895}, max_output_tokens=256, max_context_len=5000) == 9


def test_pending_tool_response_budget_can_force_final_before_next_request():
    bulky_doc = "evidence " * 1200
    llm = _ScriptedLLM(responses=[
        (
            _call("web_search", query="HLS founder"),
            {"prompt_tokens": 1_000},
        ),
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": [bulky_doc]}])

    events = asyncio.run(_run(
        DeepResearchFramework(finalize_prompt_headroom=64),
        llm, tr, max_turns=10, max_output_tokens=512, max_context_len=1_600))

    assert [e.extras["stop_reason"] for e in events] == ["search", "answer"]
    assert llm.inputs[1].metadata["tools"] == []
    assert llm.inputs[1].metadata["finalize_reason"] == "request_budget"
    assert llm.inputs[1].metadata["max_tokens"] >= MIN_FORCED_FINAL_TOKENS
    assert llm.inputs[1].metadata["finalize_tool_response_truncated"] is True
    assert (
        llm.inputs[1].metadata["finalize_trimmed_pending_prompt_tokens"]
        < llm.inputs[1].metadata["finalize_pending_prompt_tokens"]
    )
    messages = llm.inputs[1].metadata["messages"]
    assert "Do not call web_search again" in messages[-1]["content"]
    assert "tool response truncated" in messages[-2]["content"]


def test_forced_final_rejects_hedged_non_boxed_answer():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        "Based on the evidence, the founder is probably RulinShao.",
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(
        DeepResearchFramework(finalize_after_search_turns=1), llm, tr, max_turns=4))

    assert [e.extras["stop_reason"] for e in events] == ["search", "noop", "answer"]
    assert "final_answer" not in events[1].extras
    assert events[-1].extras["final_answer"] == "RulinShao"
    assert events[-1].extras["answer_format"] == "boxed"
    assert all(inp.metadata["tools"] == [] for inp in llm.inputs[1:])


def test_forced_final_preserves_evidence_summary_before_boxed_answer():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        "The search evidence identifies RulinShao as the HLS founder. "
        "\\boxed{RulinShao}",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(
        DeepResearchFramework(finalize_after_search_turns=1), llm, tr, max_turns=3))

    assert [e.extras["stop_reason"] for e in events] == ["search", "answer"]
    assert events[-1].extras["final_answer"] == "RulinShao"
    assert events[-1].extras["answer_format"] == "boxed"
    assert "search evidence" in events[-1].extras["final_text"]
    final_prompt = llm.inputs[-1].metadata["messages"][-1]["content"]
    assert "one short evidence summary sentence" in final_prompt
    assert "end with exactly one final \\boxed{answer}" in final_prompt
    assert llm.inputs[-1].metadata["tools"] == []


def test_forced_final_accepts_concise_non_boxed_fallback_answer():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="HLS founder"),
        "RulinShao",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(
        DeepResearchFramework(finalize_after_search_turns=1), llm, tr, max_turns=3))

    assert [e.extras["stop_reason"] for e in events] == ["search", "answer"]
    assert events[-1].extras["final_answer"] == "RulinShao"
    assert events[-1].extras["answer_format"] == "qwen_fallback"
    assert llm.inputs[-1].metadata["tools"] == []


def test_a_non_boxed_final_answer_is_tagged_as_fallback():
    # "qwen_fallback" is the label recorded traces carry for this reading.
    llm = _ScriptedLLM(responses=[_thought("checked evidence") + "\nRulinShao"])
    tr = _tool_runner([])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert [e.extras["stop_reason"] for e in events] == ["answer"]
    assert events[0].extras["final_answer"] == "RulinShao"
    assert events[0].extras["answer_format"] == "qwen_fallback"


def test_a_repeated_query_gets_one_tool_enabled_repair():
    call = (
        _call("web_search", query="same query")
    )
    llm = _ScriptedLLM(responses=[call] * (MAX_REPEATED_QUERY + 1) + ["\\boxed{final}"])
    tr = _tool_runner([{"documents": ["D"]} for _ in range(MAX_REPEATED_QUERY)])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=10))

    assert len(events) == MAX_REPEATED_QUERY + 2
    assert events[-2].extras["stop_reason"] == "loop_guard"
    assert events[-2].extras["repeated_query"] == "same query"
    assert events[-2].extras["n_queries"] == 0
    assert events[-1].extras["stop_reason"] == "answer"
    assert events[-1].extras["final_answer"] == "final"
    assert llm.inputs[-1].metadata["tools"][0]["function"]["name"] == "web_search"
    assert "finalize_reason" not in llm.inputs[-1].metadata
    assert llm.inputs[-1].metadata["messages"][-1]["content"] == REPEATED_QUERY_REPAIR_PROMPT


def test_a_second_repeated_query_escalates_to_forced_final():
    call = (
        _call("web_search", query="same query")
    )
    llm = _ScriptedLLM(responses=[call] * (MAX_REPEATED_QUERY + 2) + ["\\boxed{final}"])
    tr = _tool_runner([{"documents": ["D"]} for _ in range(MAX_REPEATED_QUERY)])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=12))

    assert len(events) == MAX_REPEATED_QUERY + 3
    assert events[-3].extras["stop_reason"] == "loop_guard"
    assert events[-2].extras["stop_reason"] == "loop_guard"
    assert events[-1].extras["stop_reason"] == "answer"
    assert llm.inputs[-1].metadata["tools"] == []
    assert llm.inputs[-1].metadata["finalize_reason"] == "loop_guard"


def test_forced_final_noop_emits_best_effort_answer():
    call = (
        _call("web_search", query="same query")
    )
    llm = _ScriptedLLM(responses=[call] * (MAX_REPEATED_QUERY + 2) + [""])
    tr = _tool_runner([{"documents": ["D"]} for _ in range(MAX_REPEATED_QUERY)])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=12))

    assert events[-1].extras["stop_reason"] == "answer"
    assert events[-1].extras["final_answer"] == "No concrete answer extracted"
    assert events[-1].extras["answer_format"] == "forced_empty_fallback"
    assert events[-1].extras["finalize_reason"] == "loop_guard"


def test_last_resort_forced_answer_extracts_quoted_series_candidate():
    text = (
        'Based on the search results, I can see that Kody Brown is a reality TV '
        'star from "Sister Wives" which premiered in 2010. He married his first '
        'wife Meri Barber in 1990.'
    )

    answer, fmt = _last_resort_forced_answer(text)

    assert answer == "Sister Wives"
    assert fmt == "forced_guess_fallback"


def test_last_resort_forced_answer_extracts_percentage_candidate():
    text = (
        "The YouGov article says 53% of Americans knew about the Mandela Effect, "
        "while fewer respondents could define it precisely. The rest of the "
        "response continues to summarize Nelson Mandela, the postponed rugby "
        "tournament, and why this percentage is the requested concise answer."
    )

    answer, fmt = _last_resort_forced_answer(text)

    assert answer == "53%"
    assert fmt == "forced_guess_fallback"


def test_last_resort_forced_answer_extracts_named_candidate_from_search_summary():
    text = (
        "Based on my searches, I need to identify the architecture firm. "
        "From my searches, I found ACDF Architecture (Canadian firm based in "
        "Montreal) designed the Apple Tree House, but the plot size still needs "
        "verification. The response continues with enough prose to avoid being "
        "accepted as a plain final answer."
    )

    answer, fmt = _last_resort_forced_answer(text)

    assert answer == "ACDF Architecture"
    assert fmt == "forced_guess_fallback"


def test_last_resort_forced_answer_extracts_colon_named_candidate():
    text = (
        "From my searches, Calvin Coolidge: took naps after lunch, served fewer "
        "than fifteen years, and had three children. The remaining prose is a "
        "long uncertainty analysis rather than a concise final answer."
    )

    answer, fmt = _last_resort_forced_answer(text)

    assert answer == "Calvin Coolidge"
    assert fmt == "forced_guess_fallback"


# ---- bounds + termination -------------------------------------------------

def test_max_turns_limits_runaway_trajectory():
    # Text before the call is what the forced-final turn salvages as its answer.
    llm = _ScriptedLLM(responses=[f"Still looking for q{i}. " + _call("web_search", query=f"q{i}")
                                  for i in range(10)])
    tr = _tool_runner([{"documents": []} for _ in range(10)])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=3))

    assert llm.calls == 3
    assert len(events) == 3
    assert events[-1].extras["stop_reason"] == "answer"
    assert events[-1].extras["answer_format"] == "forced_tool_fallback"
    assert events[-1].extras["finalize_reason"] == "turn_budget"
    assert [e.extras["stop_reason"] for e in events] == [
        "search", "search", "answer"]
    assert llm.inputs[-1].metadata["tools"] == []


def test_turn_budget_prompts_final_answer_before_max_turns_exhausted():
    llm = _ScriptedLLM(responses=[
        _call("web_search", query="q0"),
        _call("web_search", query="q1"),
        "\\boxed{final}",
    ])
    tr = _tool_runner([{"documents": []}, {"documents": []}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=3))

    assert [e.extras["stop_reason"] for e in events] == ["search", "search", "answer"]
    assert llm.inputs[-1].metadata["tools"] == []
    assert llm.inputs[-1].metadata["finalize_reason"] == "turn_budget"


def test_unclosed_think_terminates_to_avoid_spin():
    llm = _ScriptedLLM(responses=["<|channel>thought\nStill working through the evidence."])
    tr = _tool_runner([])
    events = asyncio.run(_run(DeepResearchFramework(), llm, tr, max_turns=1))
    assert llm.calls == 1
    assert len(events) == 1
    assert events[0].extras["stop_reason"] == "noop"


def test_a_tool_call_beside_a_boxed_answer_runs_and_the_trajectory_goes_on():
    """A reply that both calls a tool and boxes an answer is acting: the call runs, and the
    trajectory ends only on a later reply that answers without one."""
    llm = _ScriptedLLM(responses=["\\boxed{guess} " + _call("web_search", query="q"),
                                  "\\boxed{final}"])
    searched = []
    tr = LocalCallableToolRunner(
        specs=[ToolSpec(name=SEARCH_TOOL_NAME, description="d", json_schema={"type": "object"})],
        callables={SEARCH_TOOL_NAME: lambda query, **_: searched.append(query) or {"documents": ["D"]}},
    )
    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))
    assert searched == ["q"]
    assert [e.extras["stop_reason"] for e in events] == ["search", "answer"]
    assert "final_answer" not in events[0].extras
    assert events[1].extras["final_answer"] == "final"


def test_search_tags_in_a_reply_run_no_tool():
    """``<search>...</search>`` is not a call in the model's tool syntax: no search runs, and
    the reply is not read as an answer either."""
    llm = _ScriptedLLM(responses=["I should look this up. <search>HLS founder</search>",
                                  "\\boxed{RulinShao}"])
    searched = []
    tr = LocalCallableToolRunner(
        specs=[ToolSpec(name=SEARCH_TOOL_NAME, description="d", json_schema={"type": "object"})],
        callables={SEARCH_TOOL_NAME: lambda query, **_: searched.append(query) or {"documents": []}},
    )
    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))
    assert searched == []
    assert not events[0].extras.get("tool_calls")
    assert "final_answer" not in events[0].extras
    assert events[-1].extras["final_answer"] == "RulinShao"


# ---- session + turn metadata ---------------------------------------------

def test_session_id_and_turn_index_are_stamped():
    llm = _ScriptedLLM(responses=[_call("web_search", query="q"), "\\boxed{x}"])
    tr = _tool_runner([{"documents": []}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert events[0].session_id == "s0"
    assert events[1].session_id == "s0"
    assert events[0].turn_index == 0
    assert events[1].turn_index == 1
    assert events[0].input.metadata["tenant_id"] == "t0"
    # A task serves its base model: no event names an adapter.
    assert all("adapter_id" not in e.input.metadata for e in events)


# ---- thought channel ------------------------------------------------------

def test_gemma_reply_with_a_thought_channel_searches_then_answers():
    llm = _ScriptedLLM(responses=[
        '<|channel>thought\nI need evidence.<channel|><|tool_call>call:web_search'
        '{query:<|"|>HLS founder<|"|>}<tool_call|><turn|>',
        "<|channel>thought\nThe page says RulinShao.<channel|>"
        "\\boxed{RulinShao}<turn|>",
    ])
    tr = _tool_runner([{"documents": ["RulinShao founded HLS."]}])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr))

    assert [e.extras["stop_reason"] for e in events] == ["search", "answer"]
    assert events[0].extras["tool_calls"][0]["args"] == {"query": "HLS founder"}
    assert events[-1].extras["final_answer"] == "RulinShao"
    history = llm.inputs[-1].metadata["messages"]
    # The reasoning is kept, in the field the chat template renders it from.
    assert history[-3]["reasoning_content"] == "I need evidence."
    assert history[-2]["role"] == "tool"


# ---- the Gemma tool protocol ----------------------------------------------

#: One scripted Gemma session: search, open, answer. Every reply opens its own
#: thought channel, because the harness's own turn follows each tool result and the
#: chat template opens no channel after a user turn. Each tool-calling reply ends
#: with the token the model writes to ask for the result.
_GEMMA_SEARCH_REPLY = (
    "<|channel>thought\nI need evidence.<channel|>"
    '<|tool_call>call:web_search{query:<|"|>HLS founder<|"|>}<tool_call|>'
    "<|tool_response>")
_GEMMA_OPEN_REPLY = (
    "<|channel>thought\nThe first result looks primary.<channel|>"
    "<|tool_call>call:web_open{id:1}<tool_call|><|tool_response>")
_GEMMA_ANSWER_REPLY = (
    "<|channel>thought\nThe page says RulinShao.<channel|>\\boxed{RulinShao}<turn|>")
_SEARCH_DOC = "RulinShao founded HLS."
_OPEN_DOC = "RulinShao founded HLS in 2024."


def _search_open_runner():
    """A runner whose search and open return one fixed document each."""
    return LocalCallableToolRunner(
        specs=[ToolSpec(name=SEARCH_TOOL_NAME, description="search",
                        json_schema={"type": "object"}),
               ToolSpec(name="open", description="open",
                        json_schema={"type": "object"})],
        callables={SEARCH_TOOL_NAME: lambda query, **_: {"documents": [_SEARCH_DOC]},
                   "open": lambda id, **_: {"query": f"open {id}",
                                            "documents": [_OPEN_DOC]}},
    )


def _tool_payload(query: str, documents: list[str]) -> str:
    """The evidence JSON a tool result goes back as, with nothing else in it."""
    return json.dumps({"query": query, "documents": documents}, ensure_ascii=False)


def _guidance(history: list[str]) -> str:
    """The queries already spent and what to decide next: the harness's own words,
    which go in the harness's own turn."""
    lines = "\n".join(f"- {q}" for q in history)
    return (f"<search_history>\nDo not repeat these exact web_search queries:\n"
            f"{lines}\n</search_history>\n\n{AFTER_SEARCH_DECISION_PROMPT}")


def _gemma_tool_messages(call_id: str, name: str, args: dict, reasoning: str,
                         query: str, documents: list[str],
                         history: list[str]) -> list[dict]:
    """The three messages a Gemma tool call leaves behind: the call, the result, and
    the harness's own turn."""
    return [
        {"role": "assistant", "content": "", "reasoning_content": reasoning,
         "reasoning": reasoning,
         "tool_calls": [{"id": call_id, "type": "function",
                         "function": {"name": name, "arguments": args}}]},
        {"role": "tool", "tool_call_id": call_id, "name": name,
         "content": _tool_payload(query, documents)},
        {"role": "user", "content": _guidance(history)},
    ]


def _gemma_session(responses: list[str] | None = None) -> _ScriptedLLM:
    """Run the scripted session and return the LLM that recorded what it was sent."""
    llm = _ScriptedLLM(responses=responses or [_GEMMA_SEARCH_REPLY, _GEMMA_OPEN_REPLY,
                                               _GEMMA_ANSWER_REPLY])
    events = asyncio.run(_run(DeepResearchFramework(), llm,
                              _search_open_runner(), seed_item=_research_seed_item()))
    assert events[-1].extras["final_answer"] == "RulinShao"
    return llm


def _seed_turn() -> list[dict]:
    """The two messages a research trajectory starts from."""
    return [{"role": "system", "content": DEEPRESEARCH_SYSTEM},
            {"role": "user", "content": "Who founded HLS?"}]


def test_gemma_tool_result_goes_back_in_the_models_own_tool_role():
    llm = _gemma_session()
    assert llm.inputs[1].metadata["messages"] == _seed_turn() + _gemma_tool_messages(
        "call_0", "web_search", {"query": "HLS founder"}, "I need evidence.",
        "HLS founder", [_SEARCH_DOC], ["HLS founder"])


def test_gemma_history_keeps_the_whole_tool_calling_chain():
    llm = _gemma_session()
    assert llm.inputs[2].metadata["messages"] == _seed_turn() + _gemma_tool_messages(
        "call_0", "web_search", {"query": "HLS founder"}, "I need evidence.",
        "HLS founder", [_SEARCH_DOC], ["HLS founder"]) + _gemma_tool_messages(
        "call_1", "web_open", {"id": 1}, "The first result looks primary.",
        "open 1", [_OPEN_DOC], ["HLS founder"])


def test_gemma_tool_message_carries_the_payload_and_nothing_else():
    """The harness's instructions reach the model in the harness's own turn. A tool
    message that carried them would put them in the model's mouth as tool output."""
    llm = _gemma_session()
    messages = llm.inputs[1].metadata["messages"]
    assert messages[-2]["role"] == "tool"
    assert messages[-2]["content"] == _tool_payload("HLS founder", [_SEARCH_DOC])
    assert "Decide the next step now" not in messages[-2]["content"]
    assert messages[-1] == {"role": "user", "content": _guidance(["HLS founder"])}


def test_gemma_text_written_outside_the_call_joins_the_reasoning():
    """A tool-calling assistant message keeps empty content. The template renders
    content after the tool result and closes the model turn on it, so text the model
    wrote before its call goes where it was written: with the thought."""
    llm = _gemma_session(responses=[
        "<|channel>thought\nI need evidence.<channel|>Let me search for the founder."
        '<|tool_call>call:web_search{query:<|"|>HLS founder<|"|>}<tool_call|>'
        "<|tool_response>",
        _GEMMA_ANSWER_REPLY,
    ])
    assistant = llm.inputs[1].metadata["messages"][2]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == (
        "I need evidence.\n\nLet me search for the founder.")
    assert assistant["reasoning"] == assistant["reasoning_content"]


def test_gemma_forced_final_sends_the_instruction_in_the_harness_turn():
    """The finalize instruction is the harness telling the model to stop searching.
    It replaces the harness's own turn, and the tool message keeps the evidence."""
    llm = _ScriptedLLM(responses=[_GEMMA_SEARCH_REPLY, _GEMMA_ANSWER_REPLY])
    events = asyncio.run(_run(
        DeepResearchFramework(finalize_after_search_turns=1),
        llm, _search_open_runner(), max_turns=3, seed_item=_research_seed_item()))

    messages = llm.inputs[1].metadata["messages"]
    assert messages[-2] == {"role": "tool", "tool_call_id": "call_0",
                            "name": "web_search",
                            "content": _tool_payload("HLS founder", [_SEARCH_DOC])}
    assert messages[-1]["role"] == "user"
    assert "You have already searched enough." in messages[-1]["content"]
    assert "Decide the next step now" not in messages[-1]["content"]
    assert llm.inputs[1].metadata["tools"] == []
    assert events[-1].extras["final_answer"] == "RulinShao"


def test_the_framework_refuses_a_model_that_does_not_think_or_writes_another_format():
    """The tool-role protocol is the thinking template's protocol: the loop reads a
    reasoning channel out of every reply and writes one back. With thinking off the
    template renders neither, so the pair is refused where a chat-format mismatch is."""
    from hls.core.model import ModelSpec

    framework = DeepResearchFramework()
    framework.check_model(ModelSpec(base_checkpoint="google/gemma-4-31B-it", arch="TX",
                                    chat_format="gemma",
                                    chat_template_kwargs={"enable_thinking": True}))
    for kwargs in ({}, {"enable_thinking": False}):
        with pytest.raises(ValueError, match="enable_thinking"):
            framework.check_model(ModelSpec(base_checkpoint="google/gemma-4-31B-it",
                                            arch="TX", chat_format="gemma",
                                            chat_template_kwargs=kwargs))
    with pytest.raises(ValueError, match="'chatml'"):
        framework.check_model(ModelSpec(base_checkpoint="x", arch="TX", chat_format="chatml",
                                        chat_template_kwargs={"enable_thinking": True}))


def test_the_framework_reports_the_format_and_protocol_it_speaks():
    """Neither is a constructor knob; both reach the workload fingerprint."""
    import dataclasses

    framework = DeepResearchFramework()
    assert {"chat_format", "tool_result_protocol"}.isdisjoint(
        f.name for f in dataclasses.fields(DeepResearchFramework))
    assert framework.chat_format == "gemma" and framework.tool_result_protocol == "tool_role"
    config = framework.traffic_config()
    assert (config["chat_format"], config["tool_result_protocol"]) == ("gemma", "tool_role")


def _render_gemma_template(messages: list[dict], tools: list[dict]) -> str:
    """One history rendered the way transformers renders a chat template."""
    import jinja2.sandbox

    def raise_exception(message):
        raise RuntimeError(message)

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True,
                                                       lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = json.dumps
    template = env.from_string(
        Path(os.environ["HLS_GEMMA_TEMPLATE"]).read_text(encoding="utf-8"))
    return template.render(messages=messages, tools=tools, bos_token="<bos>",
                           enable_thinking=True, add_generation_prompt=True)


def _model_turn(rendered: str) -> str:
    """The first model turn, from its header to the token that closes it."""
    body = rendered.split("<|turn>model\n", 1)[1]
    return body.split("<turn|>", 1)[0]


@pytest.mark.skipif(not os.environ.get("HLS_GEMMA_TEMPLATE"),
                    reason="set HLS_GEMMA_TEMPLATE to the model's chat template file")
def test_the_scripted_session_renders_through_the_real_gemma_template():
    """The history the loop builds renders as the model's own tool protocol: the call
    in the model's call syntax, the result inside the same model turn and carrying
    only the tool's payload, and the harness's words in the harness's own turn."""
    llm = _gemma_session()
    rendered = _render_gemma_template(llm.inputs[2].metadata["messages"],
                                      llm.inputs[2].metadata["tools"])

    chain = _model_turn(rendered)
    assert "<|tool_call>call:web_search{" in chain
    assert '<|tool_response>response:web_search{value:<|"|>' in chain
    assert chain.endswith('["RulinShao founded HLS."]}<|"|>}<tool_response|>')
    assert "<|tool_call>call:web_open{id:1}" in rendered
    assert "<|turn>user\n<search_history>" in rendered
    # A user turn follows every tool result, so the prompt opens a fresh model turn
    # rather than continuing the one that called the tool, and the template renders
    # no reasoning: it renders only what an assistant message after the last user
    # turn carries, and the harness's turn is always the last one.
    assert rendered.endswith("<|turn>model\n")
    assert "I need evidence." not in rendered


@pytest.mark.skipif(not os.environ.get("HLS_GEMMA_TEMPLATE"),
                    reason="set HLS_GEMMA_TEMPLATE to the model's chat template file")
def test_a_reply_that_wrote_text_outside_its_call_still_renders_one_model_turn():
    """Text the model wrote outside its call goes to the reasoning, so the call and
    its result stay in one model turn instead of the content closing it early."""
    llm = _gemma_session(responses=[
        "<|channel>thought\nI need evidence.<channel|>Let me search for the founder."
        '<|tool_call>call:web_search{query:<|"|>HLS founder<|"|>}<tool_call|>'
        "<|tool_response>",
        _GEMMA_ANSWER_REPLY,
    ])
    rendered = _render_gemma_template(llm.inputs[1].metadata["messages"],
                                      llm.inputs[1].metadata["tools"])

    chain = _model_turn(rendered)
    assert "<|tool_call>call:web_search{" in chain
    assert '<|tool_response>response:web_search{value:<|"|>' in chain
    assert "Let me search for the founder." not in rendered
    assert rendered.count("<|turn>model") == 2   # the chain, then the next reply


def test_gemma_open_steering_lands_on_the_harness_turn():
    """The line that tells the model to read a source is the harness speaking, so it
    joins the harness's own turn and leaves the tool result alone."""
    searches = [
        "<|channel>thought\nSearching.<channel|>"
        f'<|tool_call>call:web_search{{query:<|"|>HLS founder {i}<|"|>}}<tool_call|>'
        "<|tool_response>"
        for i in range(MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN)
    ]
    llm = _ScriptedLLM(responses=searches + [_GEMMA_ANSWER_REPLY])
    tr = _tool_runner([
        {"documents": [f"[{i + 1}] L1: \u30101\u2020Result\u3011 L2: {_SEARCH_DOC}"]}
        for i in range(MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN)
    ])

    events = asyncio.run(_run(DeepResearchFramework(), llm, tr,
                              max_turns=6, seed_item=_research_seed_item()))

    assert events[MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN - 1].extras[
        "open_required_next_turn"] is True
    messages = llm.inputs[MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN].metadata["messages"]
    assert messages[-2]["role"] == "tool"
    assert OPEN_AFTER_SEARCH_PROMPT not in messages[-2]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("\n\n" + OPEN_AFTER_SEARCH_PROMPT)
    assert "Decide the next step now" in messages[-1]["content"]
    assert events[-1].extras["final_answer"] == "RulinShao"
