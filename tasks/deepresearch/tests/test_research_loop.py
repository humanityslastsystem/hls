"""The per-turn loop runs without the harness installed."""
from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parents[1]
SANDBOX_FILES = ("research_loop.py", "retriever.py", "corpus_browser.py", "limits.py", "selftest.py")
TASK_PACKAGE = "tasks.deepresearch"
SIBLING_MODULES = {name[:-3] for name in SANDBOX_FILES}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_sandbox_files_import_nothing_from_the_harness():
    for name in SANDBOX_FILES:
        path = TASK_DIR / name
        if not path.exists():
            continue  # a sandbox file this tree does not carry
        assert "hls" not in _imports(path), name


def _task_imports(tree: ast.AST) -> list[str]:
    """Every module in the tasks package that a file imports, as a dotted path.

    Covers ``import tasks.a.b``, ``from tasks.a.b import c``, ``from . import b``
    and ``from .b import c``. A relative import is resolved against this task's
    package; one that climbs out of it keeps its leading dots, so it cannot pass
    for a sibling. Imports outside the tasks package are the other test's job.
    """
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.startswith("tasks.")]
        elif isinstance(node, ast.ImportFrom) and node.level:
            if node.level > 1:
                found.append("." * node.level + (node.module or ""))
            elif node.module:
                found.append(f"{TASK_PACKAGE}.{node.module}")
            else:
                found += [f"{TASK_PACKAGE}.{a.name}" for a in node.names]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tasks."):
            found.append(node.module)
    return found


def _is_sandbox_sibling(module: str) -> bool:
    """True when a dotted path names another file that ships in the sandbox."""
    return (module.startswith(f"{TASK_PACKAGE}.")
            and module[len(TASK_PACKAGE) + 1:] in SIBLING_MODULES)


def test_sandbox_files_only_import_siblings_from_the_task_package():
    for name in SANDBOX_FILES:
        path = TASK_DIR / name
        if not path.exists():
            continue
        for module in _task_imports(ast.parse(path.read_text())):
            assert _is_sandbox_sibling(module), (name, module)


def test_the_sibling_scanner_reads_every_import_form():
    def only(source: str) -> str:
        found = _task_imports(ast.parse(source))
        assert len(found) == 1, (source, found)
        return found[0]

    for source in ("from tasks.deepresearch.retriever import Retriever",
                   "import tasks.deepresearch.corpus_browser",
                   "from . import corpus_browser",
                   "from .retriever import Retriever"):
        assert _is_sandbox_sibling(only(source)), source
    for source in ("import tasks.tab_completion.task",
                   "from tasks.deepresearch.metric import extract_final_answer",
                   "import tasks.deepresearch.datasets",
                   "from tasks.deepresearch.datasets import BROWSECOMP",
                   "from ..tab_completion import task",
                   "from .sub.mod import Thing"):
        assert not _is_sandbox_sibling(only(source)), source
    assert _task_imports(ast.parse("import json\nfrom pathlib import Path")) == []


def test_loop_runs_with_plain_callables_and_emits_one_turn_per_call():
    from tasks.deepresearch.research_loop import ResearchLoop, SEARCH_TOOL_NAME

    replies = iter([
        '<|tool_call>call:web_search{query:<|"|>who founded hls<|"|>}<tool_call|>',
        "The founder is RulinShao. \\boxed{RulinShao}",
    ])
    requests, tool_calls, turns = [], [], []

    async def complete(request):
        requests.append(request)
        return {"text": next(replies), "metadata": {"completion_tokens": 10}, "latency": 0.1}

    async def run_tool(name, args):
        tool_calls.append((name, args))
        return {"query": args["query"], "documents": ["[1]\nL1: Searched for \"x\""], "raw_text": "r"}

    async def emit(turn):
        turns.append(turn)

    loop = ResearchLoop(finalize_after_search_turns=None, finalize_context_utilization=0.85,
                        finalize_prompt_headroom=4096, noop_repair_attempts=1)
    asyncio.run(loop.run(
        session_id="s0", seed_text="Question: who founded hls\n",
        seed_messages=[{"role": "user", "content": "Question: who founded hls\n"}],
        base_metadata={"tenant_id": "t0", "session_id": "s0"},
        complete=complete, run_tool=run_tool, emit=emit, max_turns=8))

    assert len(turns) == 2 and len(requests) == 2
    assert tool_calls == [(SEARCH_TOOL_NAME, {"query": "who founded hls", "_session_id": "s0"})]
    assert turns[-1]["extras"]["final_answer"] == "RulinShao"
    assert turns[0]["completion"]["latency"] == 0.1          # passthrough keys survive
    assert [m["role"] for m in requests[1]["metadata"]["messages"][-3:]] == ["assistant", "tool", "user"]
    # The tool message carries the tool's own payload, and the next request's text opens with it.
    tool_message = requests[1]["metadata"]["messages"][-2]
    assert json.loads(tool_message["content"]) == {
        "query": "who founded hls", "documents": ['[1] L1: Searched for "x"']}
    assert requests[1]["text"].startswith(tool_message["content"])
    assert [t["turn_index"] for t in turns] == [0, 1]


def test_a_trajectory_cut_inside_a_tool_call_still_emits_the_turn_it_was_served():
    """The drive wall cuts a trajectory wherever it stands, usually inside a tool call.

    That turn's reply arrived before the tool ran, so the drive paid for its tokens and
    the trace has to carry them. Without the event the run records a turn that was never
    served, and the reply is scored for nobody.
    """
    from tasks.deepresearch.research_loop import ResearchLoop

    turns = []
    tool_started = asyncio.Event()

    async def complete(request):
        return {"text": '<|tool_call>call:web_search{query:<|"|>who founded hls<|"|>}<tool_call|>',
                "metadata": {"completion_tokens": 10}}

    async def run_tool(name, args):
        tool_started.set()
        await asyncio.Event().wait()            # the tool call the wall cuts through

    async def emit(turn):
        turns.append(turn)

    async def cut_during_the_tool_call():
        loop = ResearchLoop(finalize_after_search_turns=None, finalize_context_utilization=0.85,
                            finalize_prompt_headroom=4096, noop_repair_attempts=1)
        run = asyncio.ensure_future(loop.run(
            session_id="s0", seed_text="Question: who founded hls\n",
            seed_messages=[{"role": "user", "content": "Question: who founded hls\n"}],
            base_metadata={"tenant_id": "t0", "session_id": "s0"},
            complete=complete, run_tool=run_tool, emit=emit, max_turns=8))
        await tool_started.wait()
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    asyncio.run(cut_during_the_tool_call())

    assert [t["request_id"] for t in turns] == ["s0.0"]
    assert turns[0]["completion"]["metadata"]["completion_tokens"] == 10


# --- reading a Gemma 4 reply -----------------------------------------------------

from tasks.deepresearch import research_loop as _rl  # noqa: E402


def test_gemma_tool_call_parses_strings_numbers_and_keeps_commas_inside_strings():
    text = ('Let me search.<|tool_call>call:web_search{query:<|"|>actress, prosecuted: '
            '1970 play<|"|>,topn:5}<tool_call|>')
    assert _rl._parse_tool_call(text) == {
        "name": "search", "args": {"query": "actress, prosecuted: 1970 play", "topn": 5},
        "model_name": "web_search",
        "model_args": {"query": "actress, prosecuted: 1970 play", "topn": 5}}
    assert _rl._parse_tool_call('<|tool_call>call:web_open{id:3,cursor:0}<tool_call|>') == {
        "name": "open", "args": {"id": 3, "cursor": 0},
        "model_name": "web_open", "model_args": {"id": 3, "cursor": 0}}
    # A call cut off by the output cap still parses up to the closing brace.
    assert _rl._parse_tool_call('<|tool_call>call:web_find{pattern:<|"|>born 19<|"|>}') == {
        "name": "find", "args": {"pattern": "born 19"},
        "model_name": "web_find", "model_args": {"pattern": "born 19"}}


def test_gemma_tool_call_is_not_mistaken_for_a_final_answer():
    call = '<|tool_call>call:web_search{query:<|"|>x<|"|>}<tool_call|>'
    assert _rl._final_answer(call) is None
    assert _rl._parse_tool_call("no call here") is None


def test_only_the_end_of_turn_token_is_dropped_from_the_reply_text():
    assert _rl._strip_turn_end("The answer is \\boxed{X}<turn|>") == "The answer is \\boxed{X}"
    assert _rl._strip_turn_end("hi<turn|>\n") == "hi"
    assert _rl._strip_turn_end("untouched") == "untouched"
    # Another chat format's end-of-turn token is text.
    assert _rl._strip_turn_end("hi<|im_end|>") == "hi<|im_end|>"


def test_unfinished_reasoning_yields_no_answer_and_no_tool_call():
    text = '<|channel>thought\n<|tool_call>call:web_search{query:<|"|>x<|"|>}<tool_call|> still thinking'
    assert _rl.split_reply(text, thought_open=False)[1] is None
    assert _rl._parse_tool_call(text) is None
    assert _rl._final_answer(text) is None
    assert _rl._boxed_answer(text) is None


def test_boxed_answer_inside_reasoning_is_not_the_answer():
    text = "<|channel>thought\ncould be \\boxed{Rome}<channel|>\\boxed{Paris}"
    assert _rl._boxed_answer(text) == "Paris"
    assert _rl._boxed_answer("<|channel>thought\ncould be \\boxed{Rome}<channel|>no box here") is None


def test_tool_call_inside_reasoning_is_not_an_action():
    text = ('<|channel>thought\nI could call <|tool_call>call:web_search{query:<|"|>x<|"|>}'
            '<tool_call|><channel|>\\boxed{Paris}')
    assert _rl._parse_tool_call(text) is None
    assert _rl._boxed_answer(text) == "Paris"


def test_the_guess_scrape_removes_tool_calls():
    """The scrape reads the whole reply, so a block left in it is a source of guesses."""
    phrase = "the series is Sister Wives"
    gemma_block = '<|tool_call>call:web_search{query:<|"|>' + phrase + '\n<|"|>}<tool_call|>'
    assert _rl._best_guess_from_text(gemma_block) is None
    # Text in another format's call syntax is no call, so it is still scraped.
    assert _rl._best_guess_from_text("<tool_call>\n" + phrase + "\n</tool_call>") == "Sister Wives"


def test_a_tool_call_only_in_the_reasoning_is_not_the_recorded_reading():
    reasoned = ('<|channel>thought\nI could call <|tool_call>call:web_search{query:<|"|>x<|"|>}'
                '<tool_call|>, or answer now.<channel|>Kody Brown')
    assert _rl._last_resort_forced_answer(reasoned) == ("Kody Brown", "forced_text_fallback")
    acted = 'Kody Brown <|tool_call>call:web_search{query:<|"|>x<|"|>}'
    assert _rl._last_resort_forced_answer(acted) == ("Kody Brown", "forced_tool_fallback")


# --- user turns -------------------------------------------------------------------

def test_the_loops_prompts_are_byte_identical_to_their_constants():
    assert _rl.CONTINUE_RESEARCH_PROMPT == (
        "Continue the research. If more evidence is needed, call web_search with a "
        "more targeted query. If the evidence is sufficient, answer now as "
        "\\boxed{answer}. Do not narrate that you will search; either call the tool "
        "or provide the boxed answer.")

    assert _rl.REPEATED_QUERY_REPAIR_PROMPT == (
        "Your last web_search query repeated an earlier exact query. Do not repeat "
        "that query again, and do not keep making small wording variants of the same "
        "search. Either call web_search with genuinely new entity/source terms from "
        "the evidence above, open or find within an existing result if one is "
        "promising, or provide the final answer now as exactly one \\boxed{answer}.")

    assert _rl.REPEATED_TOOL_REPAIR_PROMPT == (
        "Your last browser tool call repeated the same action without adding new "
        "evidence. Do not repeat that same tool call again. Either call web_search "
        "with a new targeted query, use web_find on an already opened page, or provide "
        "the final answer now as exactly one \\boxed{answer}.")

    assert _rl.DISALLOWED_TOOL_REPAIR_PROMPT == (
        "That browser tool is not available for this turn. Follow the currently "
        "available tool list exactly: call one available browser tool or provide "
        "the final answer now as exactly one \\boxed{answer}.")

    assert _rl.LENGTH_NO_SEARCH_REPAIR_PROMPT == (
        "Your previous response spent the generation budget reasoning without using "
        "the browser. Do not continue analysis. On the next turn, call web_search "
        "with one targeted query using the most distinctive names, dates, titles, "
        "or constraints from the question. Output only that tool call.")

    assert _rl.INVALID_OPEN_REPAIR_PROMPT == (
        "The previous web_open failed because the id was not a valid search-result "
        "number. web_open id values are the dagger result numbers shown as "
        "\u30101\u2020title\u3011, \u30102\u2020title\u3011, etc.; they are not L-line numbers and are normally "
        "between 1 and 10. If you want a result from an earlier search after opening "
        "another page, pass that search [cursor] together with a valid result id. "
        "Now either call web_open with a valid id/cursor, call web_search with a "
        "better query, or answer as \\boxed{answer}.")

    assert _rl.AFTER_SEARCH_DECISION_PROMPT == (
        "Decide the next step now. Search results and opened pages start with a "
        "[cursor] marker. Search-result ids are ONLY the dagger numbers in "
        "\u30101\u2020title\u3011, \u30102\u2020title\u3011, etc.; do not use L-line numbers as ids. If you need "
        "to read a search result after opening another page, call web_open with both "
        "the result id and that search [cursor]. Prefer opened primary or credible "
        "sources over snippets, social pages, SEO pages, mirrors, and generic answer "
        "sites. If a high-quality source already identifies the answer, stop and "
        "answer with exactly one \\boxed{answer}. If the answer only appears in weak "
        "snippets, call web_open on a credible result or call web_search once with a "
        "new targeted query. Do not repeat any exact query in SEARCH HISTORY.")

    assert _rl.FINALIZE_PROMPT == (
        "You have already searched enough. Do not call web_search again and do not "
        "answer that the evidence is unavailable. Never write \"Unable to determine\" "
        "or an equivalent refusal. Use the accumulated snippets and question "
        "constraints to choose the most likely concise answer now. Write one short "
        "evidence summary sentence, then end with exactly one final \\boxed{answer}. "
        "If evidence is thin or conflicting, still give the best supported concrete "
        "guess inside the box.")

    assert _rl.OPEN_AFTER_SEARCH_PROMPT == (
        "You have searched several times without reading a source. Do not call "
        "web_search on the next turn. Either call web_open on a promising dagger "
        "result id from the latest search results, use web_find on an opened page, "
        "or give the best supported concise answer as exactly one \\boxed{answer}.")


def test_composed_user_turns_are_byte_identical():
    """The prompts the loop embeds in a longer user turn follow the queries already spent."""
    history = ("<search_history>\nDo not repeat these exact web_search queries:\n- q\n"
               "</search_history>")
    assert _rl._format_finalize_prompt([]) == _rl.FINALIZE_PROMPT
    assert _rl._format_finalize_prompt(["q"]) == history + "\n\n" + _rl.FINALIZE_PROMPT
    assert _rl._research_guidance([]) == _rl.AFTER_SEARCH_DECISION_PROMPT
    assert _rl._research_guidance(["q"]) == history + "\n\n" + _rl.AFTER_SEARCH_DECISION_PROMPT
    assert _rl._research_guidance(["q"], "FIX") == (
        "FIX\n\n" + history + "\n\n" + _rl.AFTER_SEARCH_DECISION_PROMPT)


# --- reply splitting --------------------------------------------------------------

def test_gemma_split_reads_a_channel_the_reply_opens_or_the_prompt_opened():
    """After a user turn the reply opens its own channel; after a tool result the
    chat template has already opened one, so the reply starts inside it."""
    assert _rl.split_reply("<|channel>thought\nX<channel|>Y", thought_open=False) == ("X", "Y")
    assert _rl.split_reply("X<channel|>Y", thought_open=True) == ("X", "Y")
    assert _rl.split_reply("X", thought_open=True) == ("X", None)
    assert _rl.split_reply("Y", thought_open=False) == (None, "Y")


def test_gemma_tool_call_keeps_the_call_the_model_wrote():
    """The history echoes the model's own call back to it, so the parser keeps the
    declared name and arguments beside the loop's renamed local call."""
    call = _rl._parse_tool_call(
        '<|tool_call>call:web_search{query:<|"|>HLS founder<|"|>,topn:5}<tool_call|>')
    assert call["name"] == "search"
    assert call["args"] == {"query": "HLS founder", "topn": 5}
    assert call["model_name"] == "web_search"
    assert call["model_args"] == {"query": "HLS founder", "topn": 5}
    opened = _rl._parse_tool_call('<|tool_call>call:web_open{id:3}<tool_call|>')
    assert opened["model_name"] == "web_open" and opened["model_args"] == {"id": 3}
