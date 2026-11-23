"""The deepresearch loop, free of harness types.

One trajectory is a sequence of chat completions. Each turn sends the system prompt,
the tool definitions and the whole history; the reply either carries one browser
tool call, which is executed and folded back into the history in the protocol the
model's own chat template reads, or a final answer. Repair prompts steer a model
that repeats itself or misuses a tool, and finalization forces a boxed answer when a
budget runs out.

Everything the prompts depend on lives here so the graded run (``framework.py``)
and the build-sandbox self-test (``selftest.py``) execute the same code. This
module imports nothing from ``hls``.

A deep-research ReAct loop modeled on the open-source deep-research agent in
https://github.com/RulinShao/elastic-inference (the ``elastic_serving.dr_utils``
prompt/loop). That agent browses the web across many turns, then emits a final,
well-sourced answer that -- for short factual questions -- also carries the
answer as ``\\boxed{...}`` (see ``elastic_serving/dr_utils/prompts.py``).

How this differs from a Search-R1 retrieval loop (the other common agentic-QA
shape, and why deep research needs its own loop):

  * **Final-answer protocol.** Search-R1 terminates on ``<answer>...</answer>``.
    Deep research terminates on ``\\boxed{...}`` -- the elastic-inference DR
    answer convention -- so a long free-form answer body can precede the boxed
    short answer.
  * **Native tool calls.** Search-R1 uses bespoke ``<search>`` tags. This task
    mirrors elastic-inference's agent loop: the server renders a ``web_search``
    tool schema, the model emits a tool call in Gemma 4's syntax, and this loop
    folds the browser result back into the history as the model's own ``tool``
    message.
  * **Open horizon.** Deep research runs many turns and inflates the running
    context with thousands of tokens of search results; ``max_turns`` is large
    and the per-turn output cap / model context window are sized accordingly in
    ``task.py`` (NOT here -- the loop is budget-agnostic).

The trust boundary: the harness owns the loop and the browser
tool; the artifact (served Gemma 4) only emits ONE LLM turn per HTTP call and
never executes a tool. Tool calls are routed through the harness ``ToolRunner``
(see ``tasks.deepresearch.corpus_browser.CorpusBrowserTool``); the
``run_tool`` callback this module takes is that runner's harness-free seam. A
sandbox copy of this file runs the same code but is not the trusted one: only
the graded run through ``framework.py`` measures the artifact.

The invariant covers what stops the loop as well as what runs in it. The task
scores trajectory makespan, so a trajectory that answers sooner scores better, and
every number the stopping rules branch on has to be one the harness owns. The
context window is one of those numbers: it is declared (``max_context_len`` below)
rather than read off the server's ``hls_context``, because a server reporting a tiny
window would otherwise force every trajectory to answer on its first turn.
"""

from __future__ import annotations
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass


def _now() -> float:
    return time.monotonic()


# Gemma 4 renders a call as <|tool_call>call:NAME{key:value,...}<tool_call|>, strings
# delimited by the <|"|> token and never escaped. Every delimiter is a special token,
# so the serving engine must keep special tokens in the reply text (the harness's
# member engines do). The tool result goes back as a tool role message answering the
# structured call.
# A reply that writes <search>...</search> tags acts rather than answers, so it is never
# read as final text; the tags run no search.
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_GEMMA_TOOL_RE = re.compile(r"<\|tool_call>\s*call:(\w+)\{(.*?)\}\s*(?:<tool_call\|>|$)",
                            re.DOTALL)
_GEMMA_QUOTE = '<|"|>'
_GEMMA_KEY_RE = re.compile(r"\s*(\w+)\s*:\s*")
# Match \boxed{...}, allowing a single level of nested braces in the answer span
# (enough for short factual answers; deep nesting is not expected).
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_CURSOR_RE = re.compile(r"^\[(\d+)\]\s+L1:", re.DOTALL)

#: The ``ModelSpec.chat_format`` whose replies the loop reads.
CHAT_FORMAT = "gemma"
#: How a tool result goes back into the history: a ``tool`` role message answering a
#: structured call on the assistant message, which is what a chat template with native
#: function calling renders.
TOOL_RESULT_PROTOCOL = "tool_role"
# Gemma 4's delimiters in a reply. Every one is a special token, so a reply only carries
# them when the serving engine decodes with special tokens kept.
#: Opens the reasoning the model writes before its visible reply.
REASONING_OPEN = "<|channel>"
#: Follows the opener to name the channel, up to and including the newline that starts
#: the reasoning itself.
REASONING_CHANNEL = "thought\n"
#: Closes it. What follows is the visible reply.
REASONING_CLOSE = "<channel|>"
#: Opens a tool-call block.
TOOL_CALL_OPEN = "<|tool_call>"
#: Closes one.
TOOL_CALL_CLOSE = "<tool_call|>"
#: Token the model writes after a tool call to ask for the result.
TOOL_RESPONSE_OPEN = "<|tool_response>"
#: End-of-turn token an engine may leave at the end of the reply text.
TURN_END = "<turn|>"
# What Gemma 4's template does (Hub revision 842da37, transformers 5.5.3, rendered
# with ``enable_thinking=True``):
#   * after a user turn the generation prompt ends at ``<|turn>model`` with no channel
#     opened, so a thinking reply writes ``<|channel>thought`` itself; after a ``tool``
#     message it ends at ``<|channel>thought``, so the reply starts inside the channel.
#     ``ResearchLoop._thought_open`` tells the two apart. The loop sends its own words
#     in a user turn after every tool result, so every reply opens its own channel.
#   * the reasoning of every assistant message after the last user turn is rendered
#     back into the prompt, so a model turn that runs on pays prompt tokens for its
#     earlier thoughts. That same user turn ends each chain after one call, so no
#     earlier thought is rendered back.
#   * a reply that calls a tool ends with ``<|tool_response>``, the token the template
#     writes to ask for the result.


def split_reply(text: str, *, thought_open: bool) -> tuple[str | None, str | None]:
    """The reply's reasoning and its visible part.

    ``thought_open`` says the prompt already opened the reasoning channel, so the
    reply starts inside it and only closes it. Reasoning is None when the reply
    carries no channel; the visible part is None when the channel never closes,
    which is a runaway inside the thought.

    The loop reads answers and tool calls from the visible part only, so neither
    a boxed answer nor a tool call written while reasoning is acted on.
    """
    stripped = text.strip()
    if REASONING_CLOSE in stripped:
        reasoning, visible = stripped.split(REASONING_CLOSE, 1)
        return _reasoning_body(reasoning), visible.strip()
    if thought_open or stripped.startswith(REASONING_OPEN):
        return _reasoning_body(stripped), None
    return None, stripped


def _reasoning_body(text: str) -> str:
    """Reasoning without the opener the reply wrote before it, so it can go back
    into the history under an opener the chat template writes itself."""
    body = text.strip()
    if body.startswith(REASONING_OPEN):
        body = body[len(REASONING_OPEN):]
        if body.startswith(REASONING_CHANNEL):
            body = body[len(REASONING_CHANNEL):]
    return body.strip()


SEARCH_TOOL_NAME = "search"
OPEN_TOOL_NAME = "open"
FIND_TOOL_NAME = "find"
TOOL_NAME_MAP = {
    "web_search": SEARCH_TOOL_NAME,
    "web_open": OPEN_TOOL_NAME,
    "web_find": FIND_TOOL_NAME,
}
MAX_REPEATED_QUERY = 2
MAX_REPEATED_TOOL_CALL = 2
MAX_SEARCH_RESULT_ID = 10
MAX_INVALID_OPEN_REPAIR = 5
MAX_TOOL_RESPONSE_DOCS = 10
MAX_TOOL_RESPONSE_DOC_CHARS = 30000
MAX_SEARCH_HISTORY_QUERIES = 24
DEFAULT_FINALIZE_AFTER_SEARCH_TURNS = None
DEFAULT_FINALIZE_CONTEXT_UTILIZATION = 0.85
DEFAULT_FINALIZE_PROMPT_HEADROOM = 4096
DEFAULT_NOOP_REPAIR_ATTEMPTS = 1
FINALIZE_MAX_TOKENS = None
MIN_FORCED_FINAL_TOKENS = 256
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN = 3
MAX_OPEN_GUIDANCE_REPEATS = 3
MAX_SEARCHES_WITHOUT_OPEN_BEFORE_FINAL = 12
TRUNCATED_TOOL_RESPONSE_NOTICE = (
    "\n\n[tool response truncated to preserve final answer budget]"
)
# The prompts the loop sends as user turns.
CONTINUE_RESEARCH_PROMPT = (
    "Continue the research. If more evidence is needed, call web_search with a "
    "more targeted query. If the evidence is sufficient, answer now as "
    "\\boxed{answer}. Do not narrate that you will search; either call the tool "
    "or provide the boxed answer."
)
REPEATED_QUERY_REPAIR_PROMPT = (
    "Your last web_search query repeated an earlier exact query. Do not repeat "
    "that query again, and do not keep making small wording variants of the same "
    "search. Either call web_search with genuinely new entity/source terms from "
    "the evidence above, open or find within an existing result if one is "
    "promising, or provide the final answer now as exactly one \\boxed{answer}."
)
REPEATED_TOOL_REPAIR_PROMPT = (
    "Your last browser tool call repeated the same action without adding new "
    "evidence. Do not repeat that same tool call again. Either call web_search "
    "with a new targeted query, use web_find on an already opened page, or provide "
    "the final answer now as exactly one \\boxed{answer}."
)
DISALLOWED_TOOL_REPAIR_PROMPT = (
    "That browser tool is not available for this turn. Follow the currently "
    "available tool list exactly: call one available browser tool or provide "
    "the final answer now as exactly one \\boxed{answer}."
)
OPEN_AFTER_SEARCH_PROMPT = (
    "You have searched several times without reading a source. Do not call "
    "web_search on the next turn. Either call web_open on a promising dagger "
    "result id from the latest search results, use web_find on an opened page, "
    "or give the best supported concise answer as exactly one \\boxed{answer}."
)
LENGTH_NO_SEARCH_REPAIR_PROMPT = (
    "Your previous response spent the generation budget reasoning without using "
    "the browser. Do not continue analysis. On the next turn, call web_search "
    "with one targeted query using the most distinctive names, dates, titles, "
    "or constraints from the question. Output only that tool call."
)
INVALID_OPEN_REPAIR_PROMPT = (
    "The previous web_open failed because the id was not a valid search-result "
    "number. web_open id values are the dagger result numbers shown as "
    "\u30101\u2020title\u3011, \u30102\u2020title\u3011, etc.; they are not L-line numbers and are normally "
    "between 1 and 10. If you want a result from an earlier search after opening "
    "another page, pass that search [cursor] together with a valid result id. "
    "Now either call web_open with a valid id/cursor, call web_search with a "
    "better query, or answer as \\boxed{answer}."
)
AFTER_SEARCH_DECISION_PROMPT = (
    "Decide the next step now. Search results and opened pages start with a "
    "[cursor] marker. Search-result ids are ONLY the dagger numbers in "
    "\u30101\u2020title\u3011, \u30102\u2020title\u3011, etc.; do not use L-line numbers as ids. If you need "
    "to read a search result after opening another page, call web_open with both "
    "the result id and that search [cursor]. Prefer opened primary or credible "
    "sources over snippets, social pages, SEO pages, mirrors, and generic answer "
    "sites. If a high-quality source already identifies the answer, stop and "
    "answer with exactly one \\boxed{answer}. If the answer only appears in weak "
    "snippets, call web_open on a credible result or call web_search once with a "
    "new targeted query. Do not repeat any exact query in SEARCH HISTORY."
)
FINALIZE_PROMPT = (
    "You have already searched enough. Do not call web_search again and do not "
    "answer that the evidence is unavailable. Never write \"Unable to determine\" "
    "or an equivalent refusal. Use the accumulated snippets and question "
    "constraints to choose the most likely concise answer now. Write one short "
    "evidence summary sentence, then end with exactly one final \\boxed{answer}. "
    "If evidence is thin or conflicting, still give the best supported concrete "
    "guess inside the box."
)

# Deep-research system prompt: adapted from
# elastic_serving/dr_utils/prompts.py SYSTEM_PROMPT. Tool descriptions are rendered
# by the served model's own chat template from the server-side tool schema,
# mirroring elastic-inference's agent loop. Keep parseable placeholder tool calls
# OUT of this prompt: the base model copies examples literally.
DEEPRESEARCH_SYSTEM = (
    "You are a research assistant that answers questions by searching the web "
    "and reading sources. Use web_search to discover candidates, web_open to "
    "read promising numbered results, and web_find to locate terms inside an "
    "opened source. Refine searches as you learn more, and do not invent "
    "citations.\n"
    "If you need more evidence, output exactly one browser tool call instead of "
    "saying that you will search. Do not repeat the same search query. Prefer "
    "opened primary or high-credibility sources over snippets, social posts, "
    "SEO pages, mirrors, and generic answer sites. When a snippet suggests an "
    "answer but the source quality is weak, open a better result or run a more "
    "targeted search before answering.\n"
    "Support every non-trivial claim with evidence from your searches or opened "
    "pages. If sources disagree, note the conflict and explain which source is "
    "more reliable. When you have enough evidence, stop searching and give a "
    "concise final answer.\n"
    "For short factual answers, include exactly one final \\boxed{answer}. "
    "Acknowledge uncertainty when evidence is thin or conflicting.\n\n"
)

BROWSER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Returns numbered results that can be opened by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "topn": {"type": "integer", "description": "Maximum number of results."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_open",
            "description": (
                "Open a search-result id, URL, or cursor location from a previous "
                "page. Use this to read source pages after web_search. Search and "
                "open outputs begin with [cursor]; after you open a page, pass the "
                "original search [cursor] when opening another result from that "
                "search, e.g. web_open({\"cursor\": 3, \"id\": 1})."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"description": "Search result number or URL."},
                    "cursor": {"type": "integer", "description": "Open page cursor."},
                    "loc": {"type": "integer", "description": "Starting line number."},
                    "num_lines": {"type": "integer", "description": "Number of lines."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_find",
            "description": "Find text in the currently opened page or a cursor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text to find."},
                    "cursor": {"type": "integer", "description": "Open page cursor."},
                },
                "required": ["pattern"],
            },
        },
    },
]
SEARCH_ONLY_TOOLS = [BROWSER_TOOLS[0]]


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold()).strip()


def _is_placeholder_answer(text: str | None) -> bool:
    norm = _normalize_answer(text or "")
    if norm in {
        "",
        "answer",
        "final answer",
        "your answer",
        "unknown",
        "n/a",
        "none",
    }:
        return True
    unavailable = (
        "unable to determine",
        "unable to locate",
        "unable to find",
        "cannot determine",
        "cannot locate",
        "cannot find",
        "could not determine",
        "could not locate",
        "could not find",
        "not found",
        "information not found",
        "not enough information",
        "insufficient evidence",
        "available evidence",
        "no concrete answer",
        "no specific answer",
    )
    return any(marker in norm for marker in unavailable)


def _is_placeholder_query(query: str | None) -> bool:
    norm = _normalize_answer(query or "")
    if norm in {"", "query", "search query", "your query", "keyword", "keywords"}:
        return True
    if norm.startswith("query ") and len(norm.split()) <= 3:
        return True
    # The model sometimes copies prompt prose that describes the placeholder
    # instead of issuing an actual query. Treat that as a malformed action.
    return "<search>query" in norm or "placeholder query" in norm


def _boxed_answer(text: str, *, thought_open: bool = False) -> str | None:
    """The last usable ``\\boxed{...}`` in the visible reply, or None.

    A box written while reasoning is a candidate the model was still weighing, so
    only the visible reply is read. ``thought_open`` says the prompt opened the
    reasoning channel, so the reply starts inside it.
    """
    visible = split_reply(text, thought_open=thought_open)[1]
    if visible is None:
        return None
    for match in reversed(list(_BOXED_RE.finditer(visible))):
        answer = match.group(1).strip()
        if not _is_bad_answer_candidate(answer):
            return answer
    return None


def _is_bad_answer_candidate(text: str | None) -> bool:
    if _is_placeholder_answer(text):
        return True
    answer = (text or "").strip()
    if len(answer.split()) > 32:
        return True
    return _looks_incomplete_research(answer)


def _looks_incomplete_research(text: str) -> bool:
    norm = _normalize_answer(text)
    if len(text.split()) > 80:
        return True
    markers = (
        "let me search",
        "try searching",
        "need to search",
        "i should search",
        "search results",
        "not giving me",
        "more information",
        "different approach",
        "based on my searches",
        "i need to identify",
        "from my searches",
        "the question asks",
    )
    return any(marker in norm for marker in markers)


def _final_answer(text: str, *, thought_open: bool = False) -> str | None:
    """The reply read as final text, the way elastic-inference's agent reads a non-tool
    turn: the visible reply is the answer.

    A reply that acts (a tool call or ``<search>`` tags) is not an answer, and neither
    is one still inside its reasoning.
    """
    visible = split_reply(text, thought_open=thought_open)[1]
    if visible is None:
        return None
    if _SEARCH_RE.search(visible) or _has_tool_call(visible):
        return None
    if _looks_incomplete_research(visible):
        return None
    return None if _is_bad_answer_candidate(visible) else (visible or None)


def _forced_final_fallback_answer(text: str, *, thought_open: bool = False) -> str | None:
    """A forced-final turn's answer, accepted only when it is short and unhedged."""
    answer = _final_answer(text, thought_open=thought_open)
    if answer is None:
        return None
    norm = _normalize_answer(answer)
    hedged = (
        "based on",
        "probably",
        "possibly",
        "i think",
        "likely",
        "uncertain",
    )
    if len(answer.split()) > 16 or any(marker in norm for marker in hedged):
        return None
    return answer


_BEST_GUESS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:answer|best answer|best guess|candidate|name|series|film|movie|team|firm|person)\s+"
        r"(?:is|was|would be|appears to be|seems to be|likely is|is likely)\s+"
        r"[\"'\u201c\u201d\u2018\u2019]?([^.\n;:]{2,96})",
        r"(?:charge|book|title|person|name)\s+"
        r"(?:could be|might be|may be|would be|appears to be|seems to be|is likely)\s+"
        r"[\"'\u201c\u201d\u2018\u2019]?([^.\n;:]{2,96})",
        r"(?:pleaded guilty to|charge was|charge is)\s+"
        r"[\"'\u201c\u201d\u2018\u2019]?([^.\n;:]{2,96})",
        r"(?:someone like|(?:one\s+)?possibility is someone like)\s*:\s*(?:[-*\u2022]\s*)?"
        r"[\"'\u201c\u201d\u2018\u2019]?([^\n;:]{2,96})",
        r"(?:most promising lead|promising lead)\s+"
        r"(?:appears to be|is|could be|might be)\s+"
        r"[\"'\u201c\u201d\u2018\u2019]?([^.\n;:]{2,96})",
        r"(?:there'?s mention of|there is mention of|mentions? of|found mentions? of)\s+"
        r"[\"'\u201c\u201d\u2018\u2019]?([^.\n;:]{2,96})",
        r"(?:references to|found references to|found)\s*:\s*(?:\d+\.\s*)?"
        r"[\"'\u201c\u201d\u2018\u2019]?([^.\n;:]{2,96})",
        r"(?:from my searches,\s*)?i found\s+(?:references to:?\s+)?"
        r"[\"'\u201c\u201d\u2018\u2019]?([^.\n;:]{2,96})",
        r"(?:from my searches,\s*)?([A-Z][A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff0-9&'. -]{2,80})\s*:",
        r"(?:known for|best known for|reality TV star from|series called|show called)\s+"
        r"[\"'\u201c\u201d\u2018\u2019]([^\"\u201c\u201d\u2018\u2019\n]{2,96})[\"'\u201c\u201d\u2018\u2019]",
        r"(?:percentage|percent|share)\s+(?:is|was|would be|appears to be|seems to be|likely is)\s+"
        r"(\d{1,3}(?:\.\d+)?%)",
        r"(\d{1,3}(?:\.\d+)?%)\s+"
        r"(?:of\s+)?(?:americans|respondents|people|adults|users)\s+"
        r"(?:knew|know|heard|aware|recognized|said)",
    )
)


# One tool-call block, closed or cut off by the cap.
_TOOL_CALL_BLOCK_RE = re.compile(re.escape(TOOL_CALL_OPEN) + r".*?(?:" + re.escape(TOOL_CALL_CLOSE)
                                 + r"|$)", re.DOTALL)


def _best_guess_from_text(text: str) -> str | None:
    """A candidate answer scraped from anywhere in the reply, reasoning included.

    This runs only after every other reading failed, so it reads the whole reply
    rather than the visible part: a name the model settled on while thinking is
    better than no answer. Tool-call blocks go first, along with any
    ``<function=...>`` block, so a scraped guess is never call syntax.
    """
    cleaned = _TOOL_CALL_BLOCK_RE.sub(" ", text)
    cleaned = re.sub(r"<function=.*?</function>", " ", cleaned, flags=re.DOTALL)
    for pattern in _BEST_GUESS_PATTERNS:
        for match in pattern.finditer(cleaned):
            guess = match.group(1).strip(" \t\r\n\"'\u201c\u201d\u2018\u2019`")
            guess = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", guess)
            guess = re.sub(r"^\*+|\*+$", "", guess)
            guess = re.sub(r"\s+", " ", guess)
            guess = re.sub(r"\s*(?:,?\s+but|,?\s+however|,?\s+though)\b.*$", "", guess,
                           flags=re.IGNORECASE).strip()
            guess = re.sub(r"\s*\([^)]{1,120}\).*$", "", guess).strip()
            guess = re.sub(
                r"\s+(?:designed|served|was|is|published|mentions?)\b.*$",
                "",
                guess,
                flags=re.IGNORECASE,
            ).strip()
            guess = re.sub(
                r"\s*(?:,?\s+as\s+(?:the|a|an)\b|,?\s+given\b|,?\s+because\b).*$",
                "",
                guess,
                flags=re.IGNORECASE,
            ).strip()
            guess = re.sub(
                r"\s+(?:or|and)\s+(?:the|a|an)\b.*$",
                "",
                guess,
                flags=re.IGNORECASE,
            ).strip()
            guess = re.sub(r"\"?\s+by\s+[A-Z][^.\n;:]{1,80}$", "", guess).strip()
            guess = guess.strip(" \t\r\n\"'\u201c\u201d\u2018\u2019`*")
            norm_guess = _normalize_answer(guess)
            question_fragment_markers = (
                "result ",
                "that the",
                "formed in high school",
                "guitarist died",
                "died 2003",
                "sang in several languages",
                "specific source",
                "specific answer",
                "specific person",
                "well known",
                "well-known",
                "a writer",
                "a person",
                "a singer",
                "a politician",
                "someone who",
                "the user is asking",
                "let me analyze",
                "i need to find",
                "i need to identify",
                "i found references",
                "references to",
                "hls founder",
                "who wrote",
                "who has",
                " or something",
                "blog with",
                "partnership with",
                "matches all",
                "criteria",
            )
            if (
                1 <= len(guess.split()) <= 12
                and not _is_bad_answer_candidate(guess)
                and not any(marker in norm_guess
                            for marker in question_fragment_markers)
            ):
                return guess
    return None


def _last_resort_forced_answer(text: str, *, thought_open: bool = False) -> tuple[str, str]:
    """An answer salvaged from a reply that gave none, with the label to record.

    The labels name the reading that produced the answer, which the trace analysis
    counts; "forced_empty_fallback" is the reply that yielded nothing.
    """
    answer = _final_answer(text, thought_open=thought_open)
    if answer is not None:
        return answer, "forced_text_fallback"
    guess = _best_guess_from_text(text)
    if guess is not None:
        return guess, "forced_guess_fallback"
    # The label names the reading, so the test and the text it salvages from are
    # the same: a call the model only considered while reasoning is not an action.
    visible = split_reply(text, thought_open=thought_open)[1] or ""
    if _has_tool_call(visible) or _SEARCH_RE.search(visible):
        stripped = visible.split(TOOL_CALL_OPEN, 1)[0].strip()
        if stripped and not _is_bad_answer_candidate(stripped):
            return stripped[:512], "forced_tool_fallback"
        return "No concrete answer extracted", "forced_empty_fallback"
    if visible and not _is_bad_answer_candidate(visible):
        return visible[:512], "forced_raw_fallback"
    return "No concrete answer extracted", "forced_empty_fallback"


def _is_length_stop(response_metadata: dict) -> bool:
    finish = str(response_metadata.get("finish_reason") or "").casefold()
    if finish == "length":
        return True
    completion = response_metadata.get("completion_tokens")
    request_cap = _request_token_cap(response_metadata)
    return (
        isinstance(completion, (int, float))
        and isinstance(request_cap, (int, float))
        and request_cap > 0
        and completion >= request_cap
    )


def _request_token_cap(response_metadata: dict) -> int | float | None:
    """The output cap the reply says it decoded under.

    Only ``_is_length_stop`` reads this, and reading it there is right: whether the cap
    cut a reply off is a statement about that reply, which is the server's to make, and
    a server that lies about it only makes the harness repair a turn it need not have.
    Nothing that decides how long a trajectory runs may use it -- those guards take the
    harness's own ``ResearchLoop.max_output_tokens``."""
    cap = response_metadata.get("request_max_tokens")
    if isinstance(cap, (int, float)):
        return cap
    cap = response_metadata.get("max_tokens")
    return cap if isinstance(cap, (int, float)) else None


def _has_tool_call(text: str) -> bool:
    """Whether the text opens a tool-call block."""
    return TOOL_CALL_OPEN in text


# A run of end-of-turn tokens at the end of a reply.
_TURN_END_RE = re.compile(r"(?:\s*" + re.escape(TURN_END) + r")+\s*$")


def _strip_turn_end(text: str) -> str:
    """Drop end-of-turn tokens an engine left at the end of the reply text."""
    return _TURN_END_RE.sub("", text)


def _local_tool_call(func_name: str | None, args: dict) -> dict | None:
    """The loop's own call for a model's (declared name, arguments), or None when the
    call names no declared tool or lacks the argument the tool cannot run without."""
    local_name = TOOL_NAME_MAP.get(func_name)
    if local_name == SEARCH_TOOL_NAME and str(args.get("query", "")).strip():
        if _is_placeholder_query(str(args["query"])):
            return None
        return {"name": SEARCH_TOOL_NAME, "args": {
            "query": str(args["query"]).strip(),
            **({"topn": int(args["topn"])} if str(args.get("topn", "")).isdigit() else {}),
        }}
    if local_name == OPEN_TOOL_NAME:
        return {"name": OPEN_TOOL_NAME, "args": dict(args)}
    if local_name == FIND_TOOL_NAME and str(args.get("pattern", "")).strip():
        return {"name": FIND_TOOL_NAME, "args": dict(args)}
    return None


def _parse_gemma_arguments(body: str) -> dict:
    """The key:value pairs inside a Gemma 4 call's braces.

    A string value runs from one <|"|> token to the next, verbatim. Any other value is
    JSON (numbers, booleans, null, lists); a bare word that is not JSON is kept as text."""
    decoder = json.JSONDecoder()
    args: dict = {}
    i, n = 0, len(body)
    while i < n:
        key_match = _GEMMA_KEY_RE.match(body, i)
        if key_match is None:
            break
        key, i = key_match.group(1), key_match.end()
        if body.startswith(_GEMMA_QUOTE, i):
            start = i + len(_GEMMA_QUOTE)
            end = body.find(_GEMMA_QUOTE, start)
            end = n if end < 0 else end
            args[key] = body[start:end]
            i = end + len(_GEMMA_QUOTE)
        else:
            try:
                args[key], i = decoder.raw_decode(body, i)
            except json.JSONDecodeError:
                end = body.find(",", i)
                end = n if end < 0 else end
                args[key] = body[i:end].strip()
                i = end
        comma = re.compile(r"\s*,").match(body, i)
        i = comma.end() if comma else n
    return args


def _parse_gemma_tool_call(text: str) -> dict | None:
    """Parse a Gemma 4 call, keeping the call as the model wrote it.

    ``model_name`` and ``model_args`` are the declared tool and its arguments before
    any renaming, which the history echoes back on the assistant message; ``name``
    and ``args`` are the loop's own tool."""
    match = _GEMMA_TOOL_RE.search(text)
    if match is None:
        return None
    model_name = match.group(1)
    model_args = _parse_gemma_arguments(match.group(2))
    call = _local_tool_call(model_name, model_args)
    if call is None:
        return None
    return {**call, "model_name": model_name, "model_args": model_args}


def _parse_tool_call(text: str, *, thought_open: bool = False) -> dict | None:
    """The reply's tool call, read from the visible text.

    A call the model wrote while reasoning is not an action, and a reply still
    inside its reasoning has taken none.
    """
    visible = split_reply(text, thought_open=thought_open)[1]
    return None if visible is None else _parse_gemma_tool_call(visible)


def _compact_tool_document(doc) -> str:
    if isinstance(doc, dict):
        title = str(doc.get("title") or "").strip()
        snippet = str(doc.get("snippet") or doc.get("content") or "").strip()
        link = str(doc.get("link") or doc.get("url") or "").strip()
        parts = []
        if title and snippet:
            parts.append(f"{title}: {snippet}")
        else:
            parts.extend(p for p in (title, snippet) if p)
        if link:
            parts.append(f"Source: {link}")
        text = " ".join(parts) if parts else json.dumps(doc, ensure_ascii=False)
    else:
        text = str(doc)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TOOL_RESPONSE_DOC_CHARS]


def _tool_documents(result: dict) -> list[str]:
    docs = result.get("documents")
    if docs is None:
        docs = result.get("docs")
    if docs is None and isinstance(result.get("organic"), list):
        docs = result["organic"]
    if docs is None:
        docs = []
    if isinstance(docs, (str, dict)):
        docs = [docs]
    if not isinstance(docs, list):
        docs = []
    compacted = [_compact_tool_document(doc) for doc in docs[:MAX_TOOL_RESPONSE_DOCS]]
    return [doc for doc in compacted if doc]


def _format_search_history(search_history: list[str]) -> str:
    if not search_history:
        return ""
    recent = search_history[-MAX_SEARCH_HISTORY_QUERIES:]
    lines = "\n".join(f"- {query}" for query in recent)
    return (
        "<search_history>\n"
        "Do not repeat these exact web_search queries:\n"
        f"{lines}\n"
        "</search_history>"
    )


def _join_prompt(head: str, tail: str) -> str:
    """Two prompt parts as one, with a blank line between them; whichever part is
    there when the other is empty."""
    return "\n\n".join(part for part in (head, tail) if part)


def _research_guidance(search_history: list[str], repair_prompt: str = "") -> str:
    """What the harness tells the model after a tool result: the repair it owes, the
    queries already spent, and what to decide next.

    All of it is the harness speaking, never the tool, so it goes in a user turn of its
    own after the tool message."""
    parts = [p for p in (repair_prompt, _format_search_history(search_history)) if p]
    parts.append(AFTER_SEARCH_DECISION_PROMPT)
    return "\n\n".join(parts)


def _format_finalize_prompt(search_history: list[str]) -> str:
    """The turn that has to answer, after the queries already spent."""
    finalize = FINALIZE_PROMPT
    history = _format_search_history(search_history)
    return f"{history}\n\n{finalize}" if history else finalize


def _tool_call_key(tool_call: dict) -> str:
    return json.dumps(tool_call, ensure_ascii=False, sort_keys=True)


def _allowed_local_tool_names(turn_tools: list[dict]) -> set[str]:
    names: set[str] = set()
    for tool in turn_tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        local_name = TOOL_NAME_MAP.get(function.get("name"))
        if local_name:
            names.add(local_name)
    return names


def _is_invalid_open_result(tool_call: dict, result: dict) -> bool:
    if tool_call.get("name") != OPEN_TOOL_NAME:
        return False
    docs = result.get("documents") or []
    if isinstance(docs, str):
        docs = [docs]
    text = "\n".join(str(doc) for doc in docs[:1])
    return "Error: link id=" in text and "not found" in text


def _invalid_open_id(tool_call: dict) -> int | None:
    if tool_call.get("name") != OPEN_TOOL_NAME:
        return None
    raw_id = (tool_call.get("args") or {}).get("id")
    if isinstance(raw_id, str) and raw_id.startswith("http"):
        return None
    try:
        result_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    if result_id > MAX_SEARCH_RESULT_ID or result_id < 1:
        return result_id
    return None


def _result_cursor(result: dict) -> int | None:
    docs = result.get("documents") or []
    if isinstance(docs, str):
        docs = [docs]
    if not docs:
        return None
    match = _CURSOR_RE.match(str(docs[0]).strip())
    if not match:
        return None
    return int(match.group(1))


def _repair_cursor_open(tool_call: dict, search_cursors: list[int]) -> dict:
    """Repair the common mixup between elastic cursor ids and result ids."""
    invalid_id = _invalid_open_id(tool_call)
    if invalid_id is None:
        return tool_call
    args = dict(tool_call.get("args") or {})
    if args.get("cursor") is not None:
        return tool_call
    if invalid_id not in search_cursors:
        return tool_call
    args["cursor"] = invalid_id
    args["id"] = 1
    return {**tool_call, "args": args}


def _tool_response_json(result: dict) -> str:
    """The compact, evidence-only payload a tool result goes back as: it rides in a
    ``tool`` message, and the chat template writes its own tool-response delimiters
    around it.

    Trace extras keep the raw tool result; the prompt-facing payload is bounded so
    document metadata cannot crowd out useful evidence in long-context trajectories.
    """
    payload = {
        "query": str(result.get("query") or "").strip(),
        "documents": _tool_documents(result),
    }
    if not payload["documents"] and isinstance(result.get("error"), dict):
        error = result["error"]
        payload["error"] = {
            "type": str(error.get("type") or "unknown")[:80],
            "message": str(error.get("message") or "")[:500],
        }
    return json.dumps(payload, ensure_ascii=False)


def _estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE)


#: Bytes allowed per message for the chat template's own delimiters, when a prompt has
#: to be bounded without a tokenizer. Far above any real template's per-message
#: wrapper on purpose: the bound is only sound while it cannot fall below the true
#: token count, and clamping an honest server would be worse than the slack it costs.
TEMPLATE_BYTES_PER_MESSAGE = 1024


def _prompt_byte_bound(messages: list[dict], tools: list[dict] | None) -> int:
    """An upper bound on the tokens this conversation's prompt renders to.

    Every token covers at least one byte, so a prompt's UTF-8 length bounds its token
    count. The rendered prompt needs a tokenizer, which is the case this exists for, so
    the parts are bounded instead: each message's own JSON, the tool definitions, and
    a per-message allowance for the delimiters a template would add around them.
    """
    total = len(json.dumps(tools).encode()) if tools else 0
    for message in messages:
        total += len(json.dumps(message, default=str).encode())
        total += TEMPLATE_BYTES_PER_MESSAGE
    return total + TEMPLATE_BYTES_PER_MESSAGE


def _trim_text_to_token_budget(text: str, max_tokens: int) -> tuple[str, bool]:
    if max_tokens <= 0:
        return "", bool(text)
    if _estimate_text_tokens(text) <= max_tokens:
        return text, False
    max_chars = max_tokens * CHARS_PER_TOKEN_ESTIMATE
    notice = TRUNCATED_TOOL_RESPONSE_NOTICE
    if max_chars <= len(notice):
        return text[:max_chars].rstrip(), True
    return (text[:max_chars - len(notice)].rstrip() + notice), True


def _text_around_tool_call(visible: str | None) -> str:
    """A reply's visible text with its tool-call block, and the token that asks for
    the result, taken out.

    What is left is whatever the model said outside the call, which is usually
    nothing. The call is a structured field of the history message rather than part
    of its content, and this leftover text joins the thought instead."""
    if not visible:
        return ""
    text = _TOOL_CALL_BLOCK_RE.sub(" ", visible)
    text = text.replace(TOOL_RESPONSE_OPEN, " ")
    return text.strip()


def _turn(session_id: str, turn_idx: int, request: dict, completion: dict,
          t_send: float, t_recv: float, extras: dict) -> dict:
    """One LLM call: what was sent, what came back, and when."""
    return {"request_id": f"{session_id}.{turn_idx}", "session_id": session_id,
            "turn_index": turn_idx, "request": request, "completion": completion,
            "t_send": t_send, "t_recv": t_recv, "extras": dict(extras)}


@dataclass
class ResearchLoop:
    """Deep-research ReAct loop: many turns, a browser tool call per turn,
    terminate on ``\\boxed{...}``.

    Reads each reply's ``web_search`` / ``web_open`` / ``web_find`` call in Gemma 4's
    syntax, taken from the visible reply rather than from its reasoning. A call is run
    through ``run_tool`` and its result appended as a ``tool`` message answering the
    call. The trajectory ends when the
    model emits final text (preferably with ``\\boxed{answer}``) or ``max_turns``
    is hit.

    The terminal turn stows the parsed short answer in ``extras["final_answer"]``
    (the contract the metric reads) and the full answer body in
    ``extras["final_text"]`` for inspection.
    """

    finalize_after_search_turns: int | None = DEFAULT_FINALIZE_AFTER_SEARCH_TURNS
    finalize_context_utilization: float = DEFAULT_FINALIZE_CONTEXT_UTILIZATION
    finalize_prompt_headroom: int = DEFAULT_FINALIZE_PROMPT_HEADROOM
    noop_repair_attempts: int = DEFAULT_NOOP_REPAIR_ATTEMPTS
    #: Context window, in tokens, the trajectory is budgeted against. Declared by the
    #: benchmark, never read off a reply: the context guards below decide when a
    #: trajectory stops searching and answers, so a window taken from the server's own
    #: ``hls_context`` would let it shorten every trajectory by reporting a small one,
    #: and trajectory length is what the task scores. ``None`` budgets against no
    #: window, which is what a task that declares none means.
    max_context_len: int | None = None
    #: ``async (messages, tools) -> int | None``: how many tokens the prompt those
    #: messages render to, counted harness-side. Injected by ``framework.py``, which
    #: has the checkpoint's tokenizer; ``None`` in a process that has no checkpoint, and
    #: then ``_prompt_tokens`` falls back to bounding the reply's own figure.
    count_prompt_tokens: Callable | None = None
    #: The per-request output cap the harness stamps on every turn, taken from the
    #: tenant's ``Workload.output_length_cap`` by way of ``HarnessContext``. The budget
    #: guards below ask how much room an answer would have left, so this is the third
    #: number they must not read off a reply: a server reporting a huge
    #: ``request_max_tokens`` makes the next turn look like it would overflow the
    #: window, and the trajectory answers instead of researching. ``None`` leaves those
    #: guards unarmed, which is what a loop driven with no declared cap has to mean.
    max_output_tokens: int | None = None

    def _thought_open(self, messages: list[dict]) -> bool:
        """Whether the prompt these messages render leaves the reasoning channel
        open, so the reply starts inside it instead of opening one.

        The chat template opens the channel after a tool result: the model turn that
        made the call is still open, and the reply is its continuation."""
        return bool(messages) and messages[-1]["role"] == "tool"

    def _model_message(self, reasoning: str, content: str) -> dict:
        """One assistant message.

        The reasoning is spelled twice: a chat template reads whichever of the two it
        was written against, and vLLM's chat parsing keeps only ``reasoning``, so a
        message that carried only ``reasoning_content`` would reach a stock vLLM
        server without it."""
        return {"role": "assistant", "content": content,
                "reasoning_content": reasoning, "reasoning": reasoning}

    def _assistant_message(self, response_text: str, *, thought_open: bool) -> dict:
        """One reply as the history carries it: the parts the chat template renders,
        the visible reply as the content and the reasoning in its own field, which the
        template writes back under its own channel delimiters."""
        reasoning, visible = split_reply(response_text, thought_open=thought_open)
        return self._model_message(reasoning or "", visible or "")

    def _tool_turn_messages(self, response_text: str, tool_call: dict,
                            result_text: str, guidance_text: str, *, turn_idx: int,
                            thought_open: bool) -> list[dict]:
        """The messages a tool-calling turn leaves in the history: what the model did,
        what the tool answered, and what the harness says next.

        The call goes back as a structured field of the assistant message, a ``tool``
        message carrying only the tool's own payload answers it, and the harness's words
        follow in a user turn. The call id only has to be unique within the session and the
        same on both messages, so it is the turn's own index; the oracle renders the
        same history, so nothing has to recognise the id.

        The assistant message's content stays empty. The chat template renders content
        after the tool result and closes the model turn on it, so text the model wrote
        outside its call joins the thought it was written with instead."""
        reasoning, visible = split_reply(response_text, thought_open=thought_open)
        thought = _join_prompt(reasoning or "", _text_around_tool_call(visible))
        call_id = f"call_{turn_idx}"
        name = self._model_tool_name(tool_call)
        args = tool_call.get("model_args")
        return [
            {**self._model_message(thought, ""),
             "tool_calls": [{
                 "id": call_id, "type": "function",
                 "function": {"name": name,
                              "arguments": dict(args) if isinstance(args, dict) else {}},
             }]},
            {"role": "tool", "tool_call_id": call_id, "name": name,
             "content": result_text},
            {"role": "user", "content": guidance_text},
        ]

    def _model_tool_name(self, tool_call: dict) -> str:
        """The tool name the model itself wrote, which the history echoes back.

        The loop's local name is not a substitute: it is not a name the model was
        offered, so echoing it would show the model a call it could not have written."""
        name = tool_call.get("model_name")
        if not name:
            raise ValueError(
                f"the loop sends tool results in the model's "
                f"tool role, so its parsed call must carry model_name, the tool as the "
                f"model named it; got {tool_call!r}")
        return str(name)

    async def _prompt_tokens(self, messages: list[dict], tools: list[dict] | None,
                             reported: object) -> int | None:
        """How long the prompt this turn sent was, or ``None`` when nothing can say.

        Not the reply's ``usage.prompt_tokens``. The context guards below stop a
        trajectory researching and make it answer, and the task scores how long a
        trajectory takes, so a server that inflates its reported prompt length answers
        on its first turn and collects the speedup for it. ``count_prompt_tokens``
        renders the conversation the harness sent through the checkpoint's own
        tokenizer, which is the same count a trusted calibration member reports.

        Where nothing can count -- a unit test, a process with no checkpoint mounted --
        the reported figure is capped at the prompt's byte bound instead. A token
        covers at least one byte, so the cap cannot bite an honest server, which runs
        three to four bytes a token."""
        if self.count_prompt_tokens is not None:
            counted = await self.count_prompt_tokens(messages, tools)
            if counted is not None:
                return int(counted)
        if not isinstance(reported, (int, float)):
            return None
        return int(min(float(reported), float(_prompt_byte_bound(messages, tools))))

    def _should_force_final_after_search(
        self, *, search_turns: int, turn_idx: int, max_turns: int,
        response_metadata: dict, pending_prompt_text: str | None = None,
    ) -> tuple[bool, str | None]:
        """Decide whether the next turn must answer instead of searching.

        The benchmark should exercise real long-context trajectories: by default
        we do NOT stop at a fixed small number of searches. We force a final
        answer only when an explicit search-turn cap is configured, when the
        rendered prompt is close to the declared context window, when the next
        turn's declared output cap would not fit beside it, or when the trajectory
        is about to exhaust ``max_turns``.

        Every number here is the harness's own. The prompt length is counted rather
        than reported, and the window and the output cap are declared, because this is
        the decision that ends a trajectory early and the task scores how long a
        trajectory takes.
        """
        if (self.finalize_after_search_turns is not None
                and search_turns >= self.finalize_after_search_turns):
            return True, "search_turn_budget"
        prompt_tokens = response_metadata.get("prompt_tokens")
        max_context_len = self.max_context_len
        pending_prompt_tokens = _estimate_text_tokens(pending_prompt_text)
        effective_prompt_tokens = (
            float(prompt_tokens) + pending_prompt_tokens
            if isinstance(prompt_tokens, (int, float)) else None
        )
        if (isinstance(prompt_tokens, (int, float))
                and isinstance(max_context_len, (int, float))
                and max_context_len > 0
                and prompt_tokens / max_context_len >= self.finalize_context_utilization):
            return True, "context_budget"
        request_max_tokens = self.max_output_tokens
        if (effective_prompt_tokens is not None
                and isinstance(request_max_tokens, (int, float))
                and isinstance(max_context_len, (int, float))
                and max_context_len > 0
                and (effective_prompt_tokens + float(request_max_tokens)
                     + float(self.finalize_prompt_headroom)) > float(max_context_len)):
            return True, "request_budget"
        if turn_idx >= max_turns - 2:
            return True, "turn_budget"
        return False, None

    def _forced_final_max_tokens(
        self,
        response_metadata: dict,
        *,
        pending_prompt_text: str | None = None,
    ) -> int | None:
        prompt_tokens = response_metadata.get("prompt_tokens")
        request_max_tokens = self.max_output_tokens
        max_context_len = self.max_context_len
        if not (
            isinstance(prompt_tokens, (int, float))
            and isinstance(request_max_tokens, (int, float))
            and isinstance(max_context_len, (int, float))
            and max_context_len > 0
        ):
            return FINALIZE_MAX_TOKENS
        pending_prompt_tokens = _estimate_text_tokens(pending_prompt_text)
        remaining = int(
            max_context_len
            - prompt_tokens
            - pending_prompt_tokens
            - self.finalize_prompt_headroom
        )
        if remaining <= 0:
            return 1
        return min(int(request_max_tokens), remaining)

    def _forced_final_parts(
        self,
        response_metadata: dict,
        *,
        pending_prompt_text: str | None = None,
        search_history: list[str] | None = None,
    ) -> tuple[str, str, int | None, dict]:
        """The pieces of the turn that has to answer: what is left of the pending text
        once it is trimmed to leave room for the answer, the instruction that follows
        it, the turn's output cap, and what the trace records about the trim.

        Kept in pieces because the trimmed text and the instruction go in different
        messages."""
        pending = pending_prompt_text or ""
        final_instruction = _format_finalize_prompt(search_history or [])
        trim_meta = {
            "finalize_min_answer_tokens": MIN_FORCED_FINAL_TOKENS,
            "finalize_pending_prompt_tokens": _estimate_text_tokens(pending),
            "finalize_trimmed_pending_prompt_tokens": _estimate_text_tokens(pending),
            "finalize_tool_response_truncated": False,
        }
        prompt_tokens = response_metadata.get("prompt_tokens")
        request_max_tokens = self.max_output_tokens
        max_context_len = self.max_context_len
        if not pending:
            return (
                "",
                final_instruction,
                self._forced_final_max_tokens(
                    response_metadata,
                    pending_prompt_text=final_instruction,
                ),
                trim_meta,
            )
        if not (
            isinstance(prompt_tokens, (int, float))
            and isinstance(request_max_tokens, (int, float))
            and isinstance(max_context_len, (int, float))
            and max_context_len > 0
        ):
            return pending, final_instruction, FINALIZE_MAX_TOKENS, trim_meta
        target_final_tokens = min(
            int(request_max_tokens),
            int(MIN_FORCED_FINAL_TOKENS),
        )
        suffix = "\n\n" + final_instruction
        tool_budget = int(
            float(max_context_len)
            - float(prompt_tokens)
            - float(self.finalize_prompt_headroom)
            - float(target_final_tokens)
            - float(_estimate_text_tokens(suffix))
        )
        trimmed_pending, was_truncated = _trim_text_to_token_budget(
            pending, tool_budget)
        trim_meta.update({
            "finalize_trimmed_pending_prompt_tokens": _estimate_text_tokens(
                trimmed_pending),
            "finalize_tool_response_truncated": was_truncated,
            "finalize_tool_response_token_budget": max(0, tool_budget),
        })
        return (
            trimmed_pending,
            final_instruction,
            self._forced_final_max_tokens(
                response_metadata,
                pending_prompt_text=_join_prompt(trimmed_pending, final_instruction),
            ),
            trim_meta,
        )

    def _rewrite_pending_prompt(self, messages: list[dict], *, result_text: str,
                                prompt_text: str) -> None:
        """Replace what the next turn reads, in the messages it already lives in: the
        result stays in the tool message and the harness's words replace the user turn
        after it. A last exchange with no tool message carries both in its user turn."""
        if len(messages) >= 2 and messages[-2]["role"] == "tool":
            messages[-2]["content"] = result_text
            messages[-1]["content"] = prompt_text
            return
        messages[-1]["content"] = _join_prompt(result_text, prompt_text)

    def _force_final_turn(self, messages: list[dict], response_metadata: dict, *,
                          result_text: str | None, pending_prompt_text: str | None,
                          search_history: list[str]) -> tuple[str, int | None, dict]:
        """Rewrite the last exchange into the turn that has to answer, and report what
        that turn now reads, the cap it decodes under, and what the trace records.

        Only the tool result is ever shortened, because it is the part that grows; the
        harness's own words are replaced by the instruction to answer."""
        trimmable = (result_text if result_text is not None
                     else (pending_prompt_text or ""))
        trimmed, instruction, max_tokens, trim_meta = self._forced_final_parts(
            response_metadata, pending_prompt_text=trimmable,
            search_history=search_history)
        self._rewrite_pending_prompt(messages, result_text=trimmed,
                                     prompt_text=instruction)
        return _join_prompt(trimmed, instruction), max_tokens, trim_meta

    async def run(self, *, session_id: str, seed_text: str, seed_messages: list[dict],
                  base_metadata: dict, complete, run_tool, emit, max_turns: int) -> None:
        running_text = seed_text
        messages = list(seed_messages)
        meta = {**base_metadata, "tools": list(BROWSER_TOOLS)}
        query_counts: dict[str, int] = {}
        tool_call_counts: dict[str, int] = {}
        search_history: list[str] = []
        search_cursors: list[int] = []
        search_turns = 0
        consecutive_searches_without_open = 0
        open_guidance_repeats = 0
        last_open_guidance_search_count = 0
        repair_attempts = 0
        repeated_query_repairs = 0
        repeated_tool_repairs = 0
        invalid_open_repairs = 0
        force_final = False
        next_turn_tools: list[dict] | None = None
        # Whether the prompt just sent leaves the reasoning channel open, so every
        # reading of the reply knows where the reasoning ends. Set once per turn,
        # before the reply is parsed, and read by the salvage helper below.
        thought_open = False

        # A turn's event is held until the turn is over, so a salvaged answer can be
        # merged into it. One event per LLM call: a second event for the same call
        # would offer the same served tokens to a per-call verifier twice.
        pending: dict | None = None
        #: Request ids already emitted, so the one-event-per-call rule is enforced at
        #: the one place events leave rather than resting on every salvage site
        #: returning after it runs. A salvage that ran after its own turn had been
        #: flushed would rebuild that turn's id from scratch, and a duplicate event is a
        #: scoring error nothing downstream would notice.
        emitted: set[str] = set()

        async def flush() -> None:
            nonlocal pending
            if pending is None:
                return
            event, pending = pending, None
            if event["request_id"] in emitted:
                raise RuntimeError(
                    f"the deepresearch loop built a second event for request "
                    f"{event['request_id']!r}: one LLM call emits one event, or its "
                    f"served tokens are offered to a per-call verifier twice")
            emitted.add(event["request_id"])
            await emit(event)

        async def emit_best_effort_answer(
            *,
            turn_idx: int,
            request: dict,
            completion: dict,
            response_text: str,
            t_send: float,
            t_recv: float,
            finalize_reason: str | None,
            length_stop: bool = False,
        ) -> None:
            """Salvage an answer from this turn's reply and end the trajectory on it."""
            nonlocal pending
            best_effort_answer, best_effort_format = _last_resort_forced_answer(
                response_text, thought_open=thought_open)
            extras = {
                **(pending["extras"] if pending is not None else {}),
                "final_answer": best_effort_answer,
                "final_text": response_text,
                "answer_format": best_effort_format,
                "stop_reason": "answer",
                "finalize_reason": finalize_reason,
            }
            if length_stop:
                extras["length_stop"] = True
            # Every caller runs after this turn's own event was built, so the second
            # branch only covers a future path that does not. It cannot resurrect a turn
            # that was already emitted: ``flush`` refuses a request id it has seen.
            base = pending if pending is not None else _turn(
                session_id, turn_idx, request, completion, t_send, t_recv, {})
            pending = {**base, "extras": extras}
            await flush()

        try:
            for turn_idx in range(max_turns):
                await flush()
                turn_meta = {**meta, "turn_index": turn_idx}
                turn_tools = (
                    [] if force_final
                    else list(next_turn_tools or BROWSER_TOOLS)
                )
                next_turn_tools = None
                if force_final:
                    force_final_max_tokens = meta.get("finalize_max_tokens")
                    if force_final_max_tokens is None:
                        force_final_max_tokens = FINALIZE_MAX_TOKENS
                    if force_final_max_tokens is not None:
                        turn_meta = {**turn_meta, "max_tokens": force_final_max_tokens}
                request = {
                    "request_id": f"{session_id}.{turn_idx}",
                    "text": running_text,
                    "metadata": {**turn_meta, "messages": list(messages),
                                 "tools": turn_tools},
                }
                thought_open = self._thought_open(messages)

                t_send = _now()
                completion = await complete(request)
                t_recv = _now()
                # Held from here, before the tool call below, because the drive wall
                # cancels the trajectory wherever it stands. Only what ``pending``
                # holds reaches the trace through the ``finally``, and this reply was
                # served and paid for whatever the rest of the turn does with it. Every
                # branch below replaces it with the same turn and fuller extras.
                pending = _turn(session_id, turn_idx, request, completion,
                                t_send, t_recv, {})

                response_text = _strip_turn_end(str(completion.get("text") or ""))
                response_metadata = dict(completion.get("metadata") or {})
                # The prompt's own length, replaced with the harness's count of what it
                # sent (``_prompt_tokens``). Done on this local copy, so the trace still
                # records what the server said about itself. Counted after the reply
                # arrived, which keeps the work outside the turn's measured latency.
                response_metadata["prompt_tokens"] = await self._prompt_tokens(
                    request["metadata"]["messages"], turn_tools,
                    response_metadata.get("prompt_tokens"))
                length_stop = _is_length_stop(response_metadata)

                boxed_answer = _boxed_answer(response_text, thought_open=thought_open)
                fallback_answer = (
                    _forced_final_fallback_answer(response_text, thought_open=thought_open)
                    if force_final and not length_stop
                    else (_final_answer(response_text, thought_open=thought_open)
                          if not length_stop else None)
                )
                final_answer = boxed_answer if boxed_answer is not None else fallback_answer
                # "qwen_fallback" is a fixed label: recorded traces carry it and their
                # readers key on it.
                answer_format = (
                    "boxed" if boxed_answer is not None
                    else ("qwen_fallback" if fallback_answer is not None else None)
                )
                tool_call = (None if force_final
                             else _parse_tool_call(response_text, thought_open=thought_open))

                if final_answer is not None and tool_call is None:
                    pending = _turn(
                        session_id, turn_idx, request, completion, t_send, t_recv,
                        {"final_answer": final_answer,
                         "final_text": response_text,
                         "answer_format": answer_format,
                         "stop_reason": "answer"})
                    await flush()
                    return

                tool_calls: list[dict] = []
                tool_results: list[dict] = []
                tool_response_text = None
                # The two halves of what a tool-calling turn leaves for the next one:
                # the tool's own payload and the harness's own words, which go in two
                # messages, so the turn keeps them apart until it sends them.
                result_text: str | None = None
                guidance_text = ""
                if tool_call:
                    allowed_tool_names = _allowed_local_tool_names(turn_tools)
                    if allowed_tool_names and tool_call["name"] not in allowed_tool_names:
                        pending = _turn(
                            session_id, turn_idx, request, completion, t_send, t_recv,
                            {"tool_calls": [tool_call], "tool_results": [],
                             "n_queries": 0, "stop_reason": "loop_guard",
                             "disallowed_tool_call": tool_call,
                             "allowed_tools": sorted(allowed_tool_names)})
                        if repair_attempts < self.noop_repair_attempts:
                            repair_attempts += 1
                            next_turn_tools = list(turn_tools)
                            messages.extend([
                                self._assistant_message(response_text, thought_open=thought_open),
                                {"role": "user",
                                 "content": DISALLOWED_TOOL_REPAIR_PROMPT},
                            ])
                            running_text = DISALLOWED_TOOL_REPAIR_PROMPT
                            force_final = False
                            continue
                        force_final = True
                        meta["finalize_reason"] = "disallowed_tool_call"
                        running_text = _format_finalize_prompt(search_history)
                        messages.append({"role": "user", "content": running_text})
                        continue
                    tool_call = _repair_cursor_open(tool_call, search_cursors)
                    query = str(tool_call["args"].get("query") or "").strip()
                    is_search_call = tool_call["name"] == SEARCH_TOOL_NAME
                    invalid_open_id = _invalid_open_id(tool_call)
                    if invalid_open_id is not None:
                        pending = _turn(
                            session_id, turn_idx, request, completion, t_send, t_recv,
                            {"tool_calls": [tool_call], "tool_results": [],
                             "n_queries": 0, "stop_reason": "loop_guard",
                             "invalid_open_id": invalid_open_id})
                        if invalid_open_repairs < MAX_INVALID_OPEN_REPAIR:
                            invalid_open_repairs += 1
                            next_turn_tools = list(SEARCH_ONLY_TOOLS)
                            messages.extend([
                                self._assistant_message(response_text, thought_open=thought_open),
                                {"role": "user", "content": INVALID_OPEN_REPAIR_PROMPT},
                            ])
                            running_text = INVALID_OPEN_REPAIR_PROMPT
                            force_final = False
                            continue
                        if repair_attempts < self.noop_repair_attempts:
                            repair_attempts += 1
                            force_final = True
                            meta["finalize_reason"] = "invalid_open_id"
                            running_text = _format_finalize_prompt(search_history)
                            messages.append({"role": "user", "content": running_text})
                            continue
                        await emit_best_effort_answer(
                            turn_idx=turn_idx, request=request,
                            completion=completion, response_text=response_text,
                            t_send=t_send, t_recv=t_recv,
                            finalize_reason="invalid_open_id")
                        return
                    if is_search_call:
                        q_norm = _normalize_answer(query)
                        query_counts[q_norm] = query_counts.get(q_norm, 0) + 1
                        consecutive_searches_without_open += 1
                    else:
                        if tool_call["name"] == OPEN_TOOL_NAME:
                            consecutive_searches_without_open = 0
                            open_guidance_repeats = 0
                            last_open_guidance_search_count = 0
                        tool_key = _tool_call_key(tool_call)
                        tool_call_counts[tool_key] = tool_call_counts.get(tool_key, 0) + 1
                        if tool_call_counts[tool_key] > MAX_REPEATED_TOOL_CALL:
                            pending = _turn(
                                session_id, turn_idx, request, completion, t_send, t_recv,
                                {"tool_calls": [tool_call], "tool_results": [],
                                 "n_queries": 0, "stop_reason": "loop_guard",
                                 "repeated_tool_call": tool_call})
                            if repeated_tool_repairs < self.noop_repair_attempts:
                                repeated_tool_repairs += 1
                                messages.extend([
                                    self._assistant_message(response_text, thought_open=thought_open),
                                    {"role": "user",
                                     "content": REPEATED_TOOL_REPAIR_PROMPT},
                                ])
                                running_text = REPEATED_TOOL_REPAIR_PROMPT
                                force_final = False
                                continue
                            if repair_attempts < self.noop_repair_attempts:
                                repair_attempts += 1
                                force_final = True
                                meta["finalize_reason"] = "repeated_tool_call"
                                running_text = _format_finalize_prompt(search_history)
                                messages.append({"role": "user", "content": running_text})
                                continue
                            await emit_best_effort_answer(
                                turn_idx=turn_idx, request=request,
                                completion=completion, response_text=response_text,
                                t_send=t_send, t_recv=t_recv,
                                finalize_reason="repeated_tool_call")
                            return
                    if is_search_call and query_counts[q_norm] > MAX_REPEATED_QUERY:
                        pending = _turn(
                            session_id, turn_idx, request, completion, t_send, t_recv,
                            {"tool_calls": [tool_call], "tool_results": [],
                             "n_queries": 0, "stop_reason": "loop_guard",
                             "repeated_query": query})
                        if repeated_query_repairs < self.noop_repair_attempts:
                            repeated_query_repairs += 1
                            messages.extend([
                                self._assistant_message(response_text, thought_open=thought_open),
                                {"role": "user",
                                 "content": REPEATED_QUERY_REPAIR_PROMPT},
                            ])
                            running_text = REPEATED_QUERY_REPAIR_PROMPT
                            force_final = False
                            continue
                        if repair_attempts < self.noop_repair_attempts:
                            repair_attempts += 1
                            force_final = True
                            meta["finalize_reason"] = "loop_guard"
                            running_text = _format_finalize_prompt(search_history)
                            messages.append({"role": "user", "content": running_text})
                            continue
                        await emit_best_effort_answer(
                            turn_idx=turn_idx, request=request,
                            completion=completion, response_text=response_text,
                            t_send=t_send, t_recv=t_recv,
                            finalize_reason="loop_guard")
                        return
                    t_tool_start = _now()
                    tool_calls.append(tool_call)
                    run_args = {**tool_call["args"], "_session_id": session_id}
                    result = await run_tool(tool_call["name"], run_args)
                    result.setdefault("query", query or tool_call["name"])
                    if is_search_call:
                        search_history.append(query)
                        if (cursor := _result_cursor(result)) is not None:
                            search_cursors.append(cursor)
                    tool_results.append({"name": tool_call["name"], "result": result,
                                         "elapsed_sec": _now() - t_tool_start})
                    repair_prompt = (
                        INVALID_OPEN_REPAIR_PROMPT
                        if _is_invalid_open_result(tool_call, result) else ""
                    )
                    if repair_prompt:
                        next_turn_tools = list(SEARCH_ONLY_TOOLS)
                    result_text = _tool_response_json(result)
                    guidance_text = _research_guidance(search_history, repair_prompt)
                    tool_response_text = _join_prompt(result_text, guidance_text)
                    messages.extend(self._tool_turn_messages(
                        response_text, tool_call, result_text, guidance_text,
                        turn_idx=turn_idx, thought_open=thought_open))
                    search_turns += 1

                event_extras = {
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                    "n_queries": len(tool_calls),
                    "stop_reason": ("search" if tool_calls else "noop"),
                }
                if tool_response_text is not None:
                    event_extras["tool_response_text"] = tool_response_text

                require_open_next = (
                    bool(tool_calls)
                    and not force_final
                    and consecutive_searches_without_open
                    >= MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN
                    and bool(search_cursors)
                    and open_guidance_repeats < MAX_OPEN_GUIDANCE_REPEATS
                    and (
                        last_open_guidance_search_count == 0
                        or consecutive_searches_without_open
                        >= last_open_guidance_search_count
                        + MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN
                    )
                )
                if require_open_next:
                    open_prompt = OPEN_AFTER_SEARCH_PROMPT
                    tool_response_text = ((tool_response_text or "") + "\n\n"
                                          + open_prompt)
                    guidance_text = _join_prompt(guidance_text, open_prompt)
                    # The harness's words are the last message: the user turn that
                    # follows the tool message.
                    if messages and messages[-1]["role"] == "user":
                        messages[-1]["content"] = (
                            guidance_text if result_text is not None else tool_response_text)
                    event_extras["tool_response_text"] = tool_response_text
                    event_extras["open_required_next_turn"] = True
                    open_guidance_repeats += 1
                    last_open_guidance_search_count = consecutive_searches_without_open

                pending = _turn(
                    session_id, turn_idx, request, completion, t_send, t_recv,
                    event_extras)

                if not tool_calls:
                    if length_stop and force_final:
                        await emit_best_effort_answer(
                            turn_idx=turn_idx, request=request,
                            completion=completion, response_text=response_text,
                            t_send=t_send, t_recv=t_recv,
                            finalize_reason=meta.get("finalize_reason"),
                            length_stop=True)
                        return
                    if length_stop:
                        if not search_turns and repair_attempts < self.noop_repair_attempts:
                            repair_attempts += 1
                            next_turn_tools = list(SEARCH_ONLY_TOOLS)
                            messages.extend([
                                self._assistant_message(response_text, thought_open=thought_open),
                                {"role": "user",
                                 "content": LENGTH_NO_SEARCH_REPAIR_PROMPT},
                            ])
                            running_text = LENGTH_NO_SEARCH_REPAIR_PROMPT
                            force_final = False
                            continue
                        force_final = True
                        meta["finalize_reason"] = (
                            "length_stop_after_search" if search_turns
                            else "length_stop_noop"
                        )
                        running_text = _format_finalize_prompt(search_history)
                        messages.append({"role": "user", "content": running_text})
                        continue
                    if search_turns and force_final:
                        if not (_has_tool_call(response_text)
                                or _SEARCH_RE.search(response_text)):
                            if repair_attempts < self.noop_repair_attempts:
                                repair_attempts += 1
                                running_text = _format_finalize_prompt(search_history)
                                messages.append({"role": "user", "content": running_text})
                                continue
                        await emit_best_effort_answer(
                            turn_idx=turn_idx, request=request,
                            completion=completion, response_text=response_text,
                            t_send=t_send, t_recv=t_recv,
                            finalize_reason=meta.get("finalize_reason"))
                        return
                    if search_turns and repair_attempts < self.noop_repair_attempts:
                        repair_attempts += 1
                        messages.extend([
                            self._assistant_message(response_text, thought_open=thought_open),
                            {"role": "user", "content": CONTINUE_RESEARCH_PROMPT},
                        ])
                        running_text = CONTINUE_RESEARCH_PROMPT
                        force_final = False
                        continue
                    # Neither a boxed answer nor a search, and repair is exhausted --
                    # force one boxed answer turn instead of dropping the trajectory.
                    if search_turns:
                        force_final = True
                        meta["finalize_reason"] = "noop_after_search"
                        running_text = _format_finalize_prompt(search_history)
                        messages.append({"role": "user", "content": running_text})
                        continue
                    if repair_attempts < self.noop_repair_attempts and not force_final:
                        repair_attempts += 1
                        messages.extend([
                            self._assistant_message(response_text, thought_open=thought_open),
                            {"role": "user", "content": CONTINUE_RESEARCH_PROMPT},
                        ])
                        running_text = CONTINUE_RESEARCH_PROMPT
                        continue
                    if not force_final:
                        force_final = True
                        meta["finalize_reason"] = "initial_noop"
                        running_text = _format_finalize_prompt(search_history)
                        messages.extend([
                            self._assistant_message(response_text, thought_open=thought_open),
                            {"role": "user", "content": running_text},
                        ])
                        continue
                    await emit_best_effort_answer(
                        turn_idx=turn_idx, request=request,
                        completion=completion, response_text=response_text,
                        t_send=t_send, t_recv=t_recv,
                        finalize_reason=meta.get("finalize_reason") or "initial_noop")
                    return

                if (
                    not force_final
                    and consecutive_searches_without_open
                    >= MAX_SEARCHES_WITHOUT_OPEN_BEFORE_FINAL
                ):
                    force_final = True
                    force_reason = "search_without_open_budget"
                    tool_response_text, finalize_max_tokens, trim_meta = (
                        self._force_final_turn(
                            messages, response_metadata, result_text=result_text,
                            pending_prompt_text=tool_response_text,
                            search_history=search_history,
                        )
                    )
                    meta["finalize_max_tokens"] = finalize_max_tokens
                    meta.update(trim_meta)
                    meta["finalize_reason"] = force_reason
                    running_text = tool_response_text
                    continue

                should_force, force_reason = self._should_force_final_after_search(
                    search_turns=search_turns, turn_idx=turn_idx, max_turns=max_turns,
                    response_metadata=response_metadata,
                    pending_prompt_text=tool_response_text)
                if should_force:
                    force_final = True
                    tool_response_text, finalize_max_tokens, trim_meta = (
                        self._force_final_turn(
                            messages, response_metadata, result_text=result_text,
                            pending_prompt_text=tool_response_text,
                            search_history=search_history,
                        )
                    )
                    meta["finalize_max_tokens"] = finalize_max_tokens
                    meta.update(trim_meta)
                    meta["finalize_reason"] = force_reason
                else:
                    force_final = False
                running_text = tool_response_text or ""

            if 'request' in locals() and (search_turns or force_final):
                await emit_best_effort_answer(
                    turn_idx=max_turns - 1,
                    request=request,
                    completion=completion,
                    response_text=response_text,
                    t_send=t_send,
                    t_recv=t_recv,
                    finalize_reason=meta.get("finalize_reason") or "turn_budget",
                )
        finally:
            await flush()
