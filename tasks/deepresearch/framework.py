"""Task-specific AgentFramework for deepresearch, served by Gemma 4.

The loop itself lives in ``research_loop`` so the build-sandbox self-test can run it
without the harness. This adapter feeds it the harness's envelopes and turns its
per-call records into ``TraceEvent``s: one event per LLM call, the tool call and
result in ``extras``, and the parsed short answer in ``extras["final_answer"]`` for
the metric.

``research_loop`` documents the trust boundary this adapter enforces (the harness runs
the loop and the tools) and
the elastic-inference lineage of the prompts.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hls.core.model import ModelSpec, TemplateArg
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import TraceEvent
from hls.spec.agent import AgentFramework
from hls.spec.tools import ToolRunner
from hls.spec.drivers import chat_message

from tasks.deepresearch.research_loop import (  # noqa: F401  used here, and re-exported for task.py and tests
    BROWSER_TOOLS, CHAT_FORMAT, CONTINUE_RESEARCH_PROMPT, DEFAULT_FINALIZE_AFTER_SEARCH_TURNS,
    DEFAULT_FINALIZE_CONTEXT_UTILIZATION, DEFAULT_FINALIZE_PROMPT_HEADROOM,
    DEFAULT_NOOP_REPAIR_ATTEMPTS, FIND_TOOL_NAME, INVALID_OPEN_REPAIR_PROMPT,
    LENGTH_NO_SEARCH_REPAIR_PROMPT, MAX_CONSECUTIVE_SEARCHES_WITHOUT_OPEN,
    MAX_OPEN_GUIDANCE_REPEATS, MAX_REPEATED_QUERY,
    MAX_REPEATED_TOOL_CALL, MAX_SEARCHES_WITHOUT_OPEN_BEFORE_FINAL,
    MAX_TOOL_RESPONSE_DOC_CHARS, MIN_FORCED_FINAL_TOKENS, OPEN_AFTER_SEARCH_PROMPT,
    OPEN_TOOL_NAME, REPEATED_QUERY_REPAIR_PROMPT,
    REPEATED_TOOL_REPAIR_PROMPT, SEARCH_TOOL_NAME, TOOL_RESULT_PROTOCOL,
    ResearchLoop, _last_resort_forced_answer,
)


log = logging.getLogger(__name__)

#: The checkpoint's tokenizer, kept per weights directory, and ``None`` for a directory
#: whose tokenizer would not load. The driver runs every trajectory of a drive in one
#: process and loading a tokenizer costs about a second, so it is loaded once -- and a
#: failure is remembered too, so a drive does not retry it on every turn of every
#: trajectory. Keyed rather than a single slot only so a process that somehow saw two
#: snapshots could not count with the wrong one.
_TOKENIZERS: dict[str, object | None] = {}


def _mounted_weights() -> str | None:
    """The mounted checkpoint snapshot, or ``None`` when this process has none.

    Mount-only, deliberately. The driver role mounts the task's weights at
    ``HLS_WEIGHTS_DIR`` (``hls.provision.mounts``), so a graded drive and every
    calibration pool member have one. Falling back to the checkpoint name the way the
    reference engines do would put a Hub reach inside a sealed container or a unit
    test, and the prompt counter has its own fallback for having no tokenizer."""
    wdir = os.environ.get("HLS_WEIGHTS_DIR")
    return wdir if wdir and Path(wdir).is_dir() else None


def _load_tokenizer(wdir: str):
    """Load the mounted checkpoint's tokenizer, or report that it would not load.

    A load that raises does not end the drive. What it costs is the prompt count, and
    the loop has a fallback for having none, so a missing tokenizer dependency would
    otherwise turn a degraded measurement into no measurement at all. It is logged at
    error. The fallback caps a reply's reported prompt length at the prompt's byte bound
    instead of counting it."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        return AutoTokenizer.from_pretrained(wdir, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 -- any load failure is the same fallback
        log.error("deepresearch: no tokenizer from the mounted checkpoint %s (%r); "
                  "trajectories will bound each reply's reported prompt length "
                  "instead of counting the prompt", wdir, exc)
        return None


def prompt_tokenizer():
    """The checkpoint's tokenizer, or ``None`` where none is mounted or none loads.

    Loaded exactly as the reference engines load theirs (``HFEngine._load``), because
    the count is only worth having if it is the count they would get."""
    wdir = _mounted_weights()
    if wdir is None:
        return None
    if wdir not in _TOKENIZERS:
        _TOKENIZERS[wdir] = _load_tokenizer(wdir)
    return _TOKENIZERS[wdir]


def _count_prompt_ids(tokenizer, messages: list[dict], tools: list[dict] | None,
                      template_kwargs: dict) -> int:
    """How many tokens this conversation's prompt renders to.

    Goes through ``hls.reference.engines``'s own renderer rather than repeating it:
    that is the single definition of how messages become the prompt a serving stack is
    fed, and a count taken any other way would drift from the one a trusted pool member
    reports for the same turn. Imported here rather than at module import so the
    framework still loads on a host without the reference's dependencies."""
    from hls.reference.engines import chat_prompt_ids
    return len(chat_prompt_ids(tokenizer, messages, tools or None, template_kwargs))


def _extract_text(env: Envelope | None) -> str:
    if env is None:
        return ""
    payload = env.first_of("text")
    return payload.text if payload is not None else ""


def _seed_messages(seed_env: Envelope, text: str) -> list[dict]:
    """The conversation a trajectory starts from: the turn's own pre-rendered
    messages, or the question as a single user turn.

    Shaped through the same helper the request encoder and the oracle use, so a seed
    message keeps whatever a chat template would read off it."""
    messages = seed_env.metadata.get("messages")
    if isinstance(messages, list) and messages:
        return [chat_message(m) for m in messages
                if isinstance(m, dict) and "role" in m and "content" in m]
    return [{"role": "user", "content": text}]


@dataclass
class DeepResearchFramework(AgentFramework):
    """The research loop on the harness's driver. The four fields are the loop's
    knobs; ``chat_format`` and ``tool_result_protocol`` name the replies it reads and
    how it sends a tool result back. ``traffic_config`` reports them all, so the
    calibration fingerprint sees a change to any of them. The context window the loop
    budgets a trajectory against, and the chat-template pin its prompt counter renders
    with, are the task's: the drive stamps both on each trajectory's
    ``HarnessContext``, and the fingerprint records them as
    ``TaskSpec.max_context_len`` and the model's pin. The framework object is shared by
    every copy of a TaskSpec, so it holds neither."""

    finalize_after_search_turns: int | None = DEFAULT_FINALIZE_AFTER_SEARCH_TURNS
    finalize_context_utilization: float = DEFAULT_FINALIZE_CONTEXT_UTILIZATION
    finalize_prompt_headroom: int = DEFAULT_FINALIZE_PROMPT_HEADROOM
    noop_repair_attempts: int = DEFAULT_NOOP_REPAIR_ATTEMPTS

    def __post_init__(self) -> None:
        """Record the ``ModelSpec.chat_format`` whose replies the loop reads, as
        ``chat_format``, and how it sends a tool result, as ``tool_result_protocol``.

        Neither is a knob: the loop reads Gemma 4's replies and answers a call in the
        model's tool role. Both are reported because they decide what every prompt after
        a tool call carries, so they belong to the population of prompts a calibration
        was measured on. A model of another chat format is refused (``check_model``).

        The private attribute is machinery, not a setting, and ``traffic_config`` drops it
        with every other underscored name: the override the framework's own tests install
        to state a prompt length without building a conversation that long."""
        self.chat_format = CHAT_FORMAT
        self.tool_result_protocol = TOOL_RESULT_PROTOCOL
        self._prompt_token_count = None

    async def warmup(self) -> None:
        """Load the prompt counter's tokenizer, before the drive anchors its clock.

        About a second of CPU. Loading it lazily would bill that second to whichever
        trajectory finished its first turn first. A process with no snapshot mounted
        loads nothing and says nothing: the counter reports that it cannot count, and
        the loop bounds the reply's own figure instead."""
        await asyncio.to_thread(prompt_tokenizer)

    def check_model(self, model: ModelSpec) -> None:
        """Refuse a model whose replies this framework cannot read.

        Nothing downstream would raise: the loop would parse every reply with the
        wrong delimiters, find no tool call and no answer, spend its repair
        budget and finish, which reads as a weak model rather than a wiring mistake.
        Nothing is kept: the pin the prompt counter renders with comes with each
        trajectory's ``HarnessContext``.
        """
        if model.chat_format != self.chat_format:
            raise ValueError(
                f"DeepResearchFramework reads replies as {self.chat_format!r} but the "
                f"model writes {model.chat_format!r}: the loop would look for the wrong "
                f"reasoning and tool-call delimiters. Serve a model whose "
                f"ModelSpec.chat_format is {self.chat_format!r}.")
        if not model.chat_template_kwargs.get("enable_thinking"):
            raise ValueError(
                f"the loop sends tool results in the {self.chat_format!r} model's "
                f"own tool role, which is the protocol its template renders with "
                f"thinking on: the loop reads a reasoning channel out of every reply "
                f"and writes one back. This model's chat_template_kwargs are "
                f"{dict(model.chat_template_kwargs)!r}; set enable_thinking to True.")

    def prompt_counter(self, chat_template_kwargs: Mapping[str, TemplateArg]):
        """The loop's prompt-token counter, or ``None`` where this process cannot count.

        Counts the prompt the harness rendered, through the checkpoint's own tokenizer
        and the model's chat-template pin *chat_template_kwargs* -- the same render a trusted calibration
        member is served and reports its ``prompt_tokens`` from, so a graded trajectory
        and a calibration trajectory are budgeted on the same number.

        ``None`` where no checkpoint is mounted, which is every process but the driver
        and the pool members; the loop bounds the reply's own figure there instead."""
        if self._prompt_token_count is not None:
            return self._prompt_token_count
        template_kwargs = dict(chat_template_kwargs)

        async def count(messages: list[dict], tools: list[dict] | None) -> int | None:
            tokenizer = prompt_tokenizer()
            if tokenizer is None:
                return None
            # In a thread: a long conversation is hundreds of kilobytes to render and
            # tokenize, and one event loop drives every trajectory of the drive.
            return await asyncio.to_thread(_count_prompt_ids, tokenizer, messages,
                                           tools, template_kwargs)

        return count

    def loop(self, max_output_tokens: int | None = None, *, max_context_len: int | None = None,
             chat_template_kwargs: Mapping[str, TemplateArg] | None = None) -> ResearchLoop:
        """The loop for one trajectory.

        ``max_output_tokens`` is the per-request output cap, ``max_context_len`` the
        task's context window and ``chat_template_kwargs`` the model's chat-template pin,
        all three handed over per trajectory because they belong to the task rather than to
        this framework: ``run_trajectory`` reads them off the ``HarnessContext``."""
        return ResearchLoop(
            finalize_after_search_turns=self.finalize_after_search_turns,
            finalize_context_utilization=self.finalize_context_utilization,
            finalize_prompt_headroom=self.finalize_prompt_headroom,
            noop_repair_attempts=self.noop_repair_attempts,
            max_context_len=max_context_len,
            count_prompt_tokens=self.prompt_counter(chat_template_kwargs or {}),
            max_output_tokens=max_output_tokens)

    def _forced_final_max_tokens(self, response_metadata: dict, *,
                                 max_output_tokens: int | None = None,
                                 max_context_len: int | None = None,
                                 pending_prompt_text: str | None = None) -> int | None:
        """Forwards to the loop. Not public surface: the only caller is a framework
        test that computes this cap on a bare framework."""
        return self.loop(max_output_tokens, max_context_len=max_context_len)._forced_final_max_tokens(
            response_metadata, pending_prompt_text=pending_prompt_text)

    async def run_trajectory(self, seed_item, llm_call, tool_runner: ToolRunner,
                             on_event, ctx, max_turns: int) -> None:
        seed_env = seed_item.turns[0]
        seed_text = _extract_text(seed_env)
        session_id = seed_item.request_id
        ground_truth = (seed_item.ground_truth_turns[0]
                        if seed_item.ground_truth_turns else None)
        base_metadata = {**seed_env.metadata, "tenant_id": ctx.tenant_id, "session_id": session_id}

        async def complete(request: dict) -> dict:
            env = Envelope(request_id=request["request_id"],
                           contents=[TextPayload(text=request["text"])],
                           metadata=dict(request["metadata"]))
            response = await llm_call(env)
            output = response.output if response is not None else None
            payload = output.first_of("text") if output is not None else None
            return {"text": payload.text if payload is not None else "",
                    "metadata": dict(output.metadata) if output is not None else {},
                    "envelope": env, "response": response}

        async def run_tool(name: str, args: dict) -> dict:
            return await tool_runner.run(name, args)

        async def emit(turn: dict) -> None:
            completion = turn["completion"]
            await on_event(TraceEvent(
                request_id=turn["request_id"], session_id=session_id,
                turn_index=turn["turn_index"], input=completion["envelope"],
                response=completion["response"], ground_truth=ground_truth,
                t_send=turn["t_send"], t_recv=turn["t_recv"], extras=turn["extras"]))

        await self.loop(ctx.max_output_tokens, max_context_len=ctx.max_context_len,
                        chat_template_kwargs=ctx.chat_template_kwargs).run(
            session_id=session_id, seed_text=seed_text,
            seed_messages=_seed_messages(seed_env, seed_text),
            base_metadata=base_metadata, complete=complete, run_tool=run_tool,
            emit=emit, max_turns=max_turns)
