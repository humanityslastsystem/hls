"""L3 -- reference oracle engine (the runtime face; config lives in references.py).

An HFEngine is an Engine: the role that verifies loads it on its engine host
(``hls.verify.inprocess``), which runs its ``_load``, and it serves Generate/TeacherForce as
plain method calls in that process. Built from an HFReference config via
``HFReference.build_engine()``.
"""
from __future__ import annotations

import atexit
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from hls.core.capabilities import (PAD_ID, Generate, ReferenceLogprobs, ServeTraffic,
                                   ServingVersion, StackReport, TeacherForce)
from hls.core.decode import DecodeConfig
from hls.core.engine import Engine, ResourceRequest
from hls.core.model import ModelSpec, TemplateArg
from hls.core.payloads import Envelope, TextPayload, TokenLogprob, TopLogprob
from hls.reference.kernel_pins import fused_kernel_bindings, pin_fused_kernels
from hls.spec.drivers import message_content, messages_of

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec

# Legal HFEngine instrument pins, validated at construction so a bad reference config
# fails on the torch-free config host (no torch import here). The name -> SDPBackend
# map is built lazily in HFEngine._sdpa_ctx, where torch is available.
# "efficient" (mem-efficient) is absent because the kernel rejects GQA
# (enable_gqa) on every current torch, so it cannot run any modern checkpoint.
# "default" = no pin at all (stock torch dispatch, cuDNN eligible) -- the calibrate
# pool's model of an unpinned contestant server; never a reference instrument.
_SDPA_BACKENDS: tuple[str, ...] = ("math", "flash", "default")
_HF_DTYPES: tuple[str, ...] = ("auto", "float32")
_CONDITIONINGS: tuple[str, ...] = ("chat", "raw")


def contract_conditioning(task: "TaskSpec") -> dict[str, object]:
    """Engine constructor kwargs read off ``task.contract.protocol``: how the prompt
    reaches the model and which ids end a turn. Every engine built from a task (the
    grading oracle and the calibration pool alike) is constructed with these, so what
    the server saw and what the reference reproduces cannot drift apart.

    A contract that declares no wire protocol is refused, not defaulted: nothing then
    says how the prompt reached the server, and every engine of a calibration pool
    would be built on the guess."""
    proto = getattr(task.contract, "protocol", None)
    if proto is None:
        raise TypeError(
            f"task {task.id!r} carries a {type(task.contract).__name__} that declares no "
            f"wire protocol, so nothing says how its prompt reached the server; a "
            f"reference built from it would guess chat conditioning and the checkpoint's "
            f"own eos set. Give the contract a WireProtocol.")
    return {"conditioning": proto.conditioning, "stop_token_ids": proto.stop_token_ids}


def _messages_of(p: Envelope) -> list[dict]:
    """Chat messages for prompt *p*: delegates to ``messages_of`` (``hls.spec.drivers``),
    the single definition of a request's message list that the wire protocols also
    build their request bodies from, through the same ``chat_message`` the artifact's
    OpenAIChatProtocol encodes with. Shared by HFEngine and VLLMEngine so the oracle
    renders the conversation the same way regardless of backend, and so a message the
    wire protocol drops (missing ``role`` or ``content``) is never scored here either."""
    return messages_of(p)


def continues_of(p: Envelope) -> bool:
    """Whether this request resumes the last assistant message rather than opening a
    new turn.

    Read off the same envelope ``OpenAIChatProtocol._stamp_continuation`` encodes
    from, so the oracle and the artifact cannot disagree about which turn was
    resumed. A turn the harness cut at a cap is handed back to the model to finish,
    and the reference has to render the prompt that was actually sent -- a generation
    prompt here would teacher-force a different one and score the harness.

    A continuation whose conversation does not end on an assistant message is refused.
    The template would render it without complaint, as a user or tool message left
    open with no generation prompt after it, which is a prompt no loop means to send;
    the flag reached a request it does not belong to, and scoring it would hide that."""
    if not p.metadata.get("continue_final_message"):
        return False
    messages = _messages_of(p)
    last = messages[-1]["role"] if messages else None
    if last != "assistant":
        raise ValueError(
            f"request {p.request_id!r} asks to continue its final message, but that "
            f"message's role is {last!r}, not 'assistant'. Only an assistant message can "
            f"be resumed; the continuation flag was stamped on the wrong request.")
    return True


def tools_of(p: Envelope) -> list[dict] | None:
    """Tool definitions prompt *p*'s turn offers, or None when it offers none.

    A chat template folds tool definitions into the rendered prompt, so a turn's
    tools are part of its context: the oracle must render the same ones the artifact
    was sent, or every teacher-forced position would score a gap that measures the
    harness rather than the artifact. An empty list is None -- the same rendering the
    template produces with no tools argument at all."""
    tools = p.metadata.get("tools")
    return tools if isinstance(tools, list) and tools else None


def chat_prompt_text(tokenizer, messages: list[dict],
                     tools: list[dict] | None = None,
                     template_kwargs: dict[str, TemplateArg] | None = None,
                     continue_final_message: bool = False) -> str:
    """One conversation rendered through the model's own chat template, with the
    generation prompt appended: the chat half of ``ReferenceEngine._render_context``.

    The single definition of how every engine in this module turns messages into a chat
    prompt, so an oracle, a calibration pool member and the teacher-forcing pass all
    see byte-identical context. ``template_kwargs`` is the model's chat-template pin
    (``ModelSpec.chat_template_kwargs``), passed through as template variables the
    way ``enable_thinking`` is. A stack that renders chat its own way (vLLM's
    ``LLM.chat``, sglang's server-side chat) is passed over for that reason:
    determinism matters more here than convenience.

    ``continue_final_message`` renders a turn the harness cut and handed back: the last
    assistant message resumed, with no generation prompt after it.

    Public for the same reason ``chat_prompt_ids`` is: a harness-side trajectory loop
    has to read facts about the prompt a serving stack will be fed, and a template that
    turns thinking on ends the generation prompt inside an open thought -- which is
    what tells a reply that never closed one from a reply that had none."""
    # The pin goes down first and this call's own arguments over it, the same order
    # VLLMEngine._generate_uniq merges them in. ModelSpec rejects a pin naming any of
    # them, so the two can never actually disagree.
    kwargs: dict = dict(template_kwargs or {})
    kwargs.update({"tokenize": False, "add_generation_prompt": not continue_final_message})
    if continue_final_message:
        # Both, or the template closes the assistant turn and appends a fresh
        # generation prompt, which makes the render a new turn rather than the
        # resumption of the one the artifact was actually served.
        kwargs["continue_final_message"] = True
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def chat_prompt_ids(tokenizer, messages: list[dict],
                    tools: list[dict] | None = None,
                    template_kwargs: dict[str, TemplateArg] | None = None,
                    continue_final_message: bool = False) -> list[int]:
    """The prompt token ids for one conversation: ``chat_prompt_text`` tokenized
    with ``add_special_tokens=False``.

    Public because a harness-side trajectory loop needs the same number a serving
    stack will report. An agentic framework stops a trajectory on how full the context
    window is, and the pool member it is calibrated against reports ``prompt_tokens``
    as the length of exactly this list (``ClockedServing._context_ids`` ->
    ``_served_metadata``). A count taken any other way would budget a graded
    trajectory and a calibration trajectory on two different numbers, so the guarantee
    this name carries is that there is one render and everybody uses it.

    Ids, not text, are what the async serving APIs are fed. Handing a serving stack
    the rendered string makes it re-tokenize and prepend a second BOS -- the rendered
    chat template already carries one -- so the whole conversation shifts by a token.
    Fed text, sglang reports prompt_tokens one higher than these ids, and a pool member
    then disagrees with the reference at defect scale; fed ids, it does not.
    ``add_special_tokens=False`` is the same convention
    HFEngine tokenizes its rendered prompts with, for the same reason. Takes the
    model's chat-template pin and forwards it, like ``chat_prompt_text``."""
    return tokenizer(chat_prompt_text(tokenizer, messages, tools, template_kwargs,
                                      continue_final_message),
                     add_special_tokens=False).input_ids


def _max_new_tokens(prompts: list[Envelope]) -> int | None:
    """The decoding budget for a batch: the max positive ``metadata['max_tokens']``
    across prompts (the OutputLengthCap the driver stamps), or None if none set.
    Shared so both engines cap generation identically."""
    caps = [p.metadata.get("max_tokens") for p in prompts]
    return max((c for c in caps if isinstance(c, int) and c > 0), default=None)


def _served_metadata(prompt: Envelope, prompt_tokens: int, completion_tokens: int,
                     max_context_len: int | None) -> dict:
    """The serving facts one clocked serving's reply reports, under the names the wire
    protocol gives a graded server's (``OpenAIChatProtocol.decode_response``).

    An agentic framework reads them off every reply: how close the conversation has come
    to the context window decides when a trajectory must stop researching and answer,
    and the served count against the turn's own cap is how it recognizes a reply the cap
    cut off. A pool member that reported none of it would run the task's framework down
    branches a graded run never takes, so the two would not serve the same conversations.

    ``request_max_tokens`` is the turn's own output cap, which the driver stamps on
    every request (``hls.spec.drivers``) and ``_max_new_tokens`` reads to bound the
    generation. ``max_context_len`` is left out when the engine serves at its
    checkpoint's own context, exactly as it is absent from a reply whose server
    declared none."""
    meta: dict = {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens}
    cap = prompt.metadata.get("max_tokens")
    if isinstance(cap, int) and cap > 0:
        meta["request_max_tokens"] = cap
    if max_context_len is not None:
        meta["max_context_len"] = max_context_len
    return meta


def _reference_reply(request_id: str, text: str,
                     finish_reason: str | None,
                     tokens: list[int] | None = None,
                     logprobs: list[TokenLogprob] | None = None,
                     metadata: dict | None = None) -> Envelope:
    """The reply envelope both engines emit: one TextPayload, source-stamped.
    Shared so the real finish_reason ("stop" on natural EOS vs "length" on a
    cap cut) propagates from one place instead of a per-engine hardcode.
    ``tokens`` carries the exact ids the reference emitted (HF strips trailing batch
    padding first) so a downstream teacher_force scores the true continuation; None
    when the caller has no ids to attach.

    ``logprobs`` carries the reply's own report -- per served token, the
    log-probability the stack assigned to it and to that position's most probable
    alternatives (``_served_report``). Present only on a sampled clocked serving,
    where it is what the sampling gate's calibration measures; None everywhere else,
    because no other path reads one.

    ``metadata`` carries the serving facts a clocked serving reports with its reply
    (``_served_metadata``), merged beside the source tag. The batch paths attach none:
    nothing downstream of an oracle generation reads them.

    ``text`` follows the decode convention of the path that produced it. A clocked
    serving keeps the model's control tokens (``skip_special_tokens=False``): the
    agentic framework driving it parses tool-call and reasoning delimiters out of the
    text, and for a model whose delimiters are special tokens (Gemma 4) a stripping
    decoder leaves nothing to parse. An oracle ``generate`` strips them instead, the
    way a stock server does, because the metric verifiers score that text against a
    contestant reply decoded the same way. Ids are unaffected either way."""
    return Envelope(request_id=request_id,
                    contents=[TextPayload(text=text, finish_reason=finish_reason,
                                          tokens=tokens, logprobs=logprobs)],
                    metadata={"source": "reference", **(metadata or {})})


def _report_width(decode: DecodeConfig) -> int | None:
    """How many alternatives a clocked serving reports per served position, or ``None``
    when its decode calls for no report at all.

    A sampled decode is graded on the distribution the server reports rather than on
    which token it served (``hls.verify.report``), so a pool member serving one must
    report the width the contract asks a contestant for. A greedy decode is graded on
    the tokens themselves, and nothing on either side of that gate reads a reported
    distribution, so asking for one would cost the serving throughput for values that
    are then thrown away."""
    from hls.core.decode import Sample
    from hls.verify.report import TOP_LOGPROBS_REQUESTED
    return TOP_LOGPROBS_REQUESTED if isinstance(decode, Sample) else None


def _served_report(positions: list[tuple[int, float, str | None, list[tuple[int, float]]]],
                   width: int) -> list[TokenLogprob]:
    """One reply's report, in the shape the wire protocol delivers a contestant's.

    ``positions`` is one ``(token_id, logprob, text, alternatives)`` per served token,
    in order, where ``alternatives`` is whatever the stack reported at that position as
    ``(token_id, logprob)`` pairs.

    The alternatives are ranked by log-probability, ties by id, and cut to ``width``.
    That cut is what makes the two stacks report the same thing: SGLang sends its true
    top-k, while vLLM appends the token it sampled to its top-k, so an uncut vLLM row
    would be one entry wider exactly when the draw fell outside the top-k. Ranking and
    cutting drops the appended entry again precisely in that case, so the report never
    claims a token was among the most probable because the server happened to draw it.
    The served token's own log-probability is recorded separately, on the entry, so it
    survives the cut either way."""
    out: list[TokenLogprob] = []
    for token_id, logprob, text, alternatives in positions:
        ranked = sorted(alternatives, key=lambda kv: (-kv[1], kv[0]))[:width]
        out.append(TokenLogprob(
            token=text or "", token_id=token_id, logprob=logprob,
            top_logprobs=[TopLogprob(token_id=i, logprob=lp) for i, lp in ranked]))
    return out


def _vllm_positions(completion) -> list[tuple[int, float, str | None, list[tuple[int, float]]]]:
    """vLLM's reported log-probabilities as ``_served_report`` positions.

    ``completion.logprobs`` is one ``{token_id: Logprob}`` per served position, aligned
    with ``completion.token_ids``, holding that position's top-k plus the sampled token
    when the draw fell outside it. Direct indexing on the documented shape: a report
    that does not cover every served token, or that omits the log-probability of the
    token the stack itself sampled, must raise here rather than be scored as if the
    missing positions had never been served."""
    steps = completion.logprobs
    served_ids = list(completion.token_ids)
    if steps is None or len(steps) != len(served_ids):
        raise RuntimeError(
            f"vLLM reported {0 if steps is None else len(steps)} logprob positions for "
            f"{len(served_ids)} served tokens; the report must cover every token.")
    out = []
    for token_id, step in zip(served_ids, steps):
        served = step.get(token_id)
        if served is None:
            raise RuntimeError(
                f"vLLM reported no log-probability for the token it sampled ({token_id})")
        out.append((token_id, float(served.logprob),
                    getattr(served, "decoded_token", None),
                    [(i, float(alt.logprob)) for i, alt in step.items()]))
    return out


def _sglang_positions(meta: dict) -> list[tuple[int, float, str | None, list[tuple[int, float]]]]:
    """SGLang's reported log-probabilities as ``_served_report`` positions.

    ``meta_info['output_token_logprobs']`` is one ``(logprob, token_id, text)`` per
    served token and ``meta_info['output_top_logprobs']`` one list of the same triples
    per position: this stack's true top-k, which need not contain the token it sampled.
    Direct indexing on the documented reply shape, as the ids path beside it does, so a
    schema change raises here instead of silently degrading the report."""
    served = meta["output_token_logprobs"]
    tops = meta["output_top_logprobs"]
    if len(tops) != len(served):
        raise RuntimeError(
            f"sglang reported {len(tops)} alternative rows for {len(served)} served "
            f"tokens; the report must cover every token.")
    return [(token_id, float(logprob), text,
             [(alt_id, float(alt_lp)) for alt_lp, alt_id, _ in alts])
            for (logprob, token_id, text), alts in zip(served, tops)]



def _hf_finish_reason(generated_ids: list[int], eos_ids: set[int]) -> str:
    """Derive finish_reason for one generated row (HF generate has no native
    field). An EOS id anywhere in the row means the model ended naturally --
    batch padding after an early stop also lands here, since rows that finish
    early are padded (with EOS itself when pad is unset). No EOS means the
    max_new_tokens cap cut the row off."""
    return "stop" if any(t in eos_ids for t in generated_ids) else "length"


def _emitted_tokens(generated_ids: list[int], eos_ids: set[int]) -> list[int]:
    """The true emitted sequence for one generated row, stripped of trailing batch
    padding. generate right-pads rows that stop early (with EOS itself, since pad_token
    is set to eos when unset), so on a natural stop the emitted tokens are everything
    up to and including the first EOS; a length stop (no EOS in the row) is the whole
    row. Mirrors _hf_finish_reason's EOS test so tokens and finish_reason agree."""
    for i, t in enumerate(generated_ids):
        if t in eos_ids:
            return generated_ids[:i + 1]
    return list(generated_ids)


def _eos_id_set(*declared: object) -> set[int]:
    """The ids in a checkpoint's declared end-of-turn fields. Each field is one id, a
    list of ids (multi-EOS models like Llama 3, where ``<|eot_id|>`` differs from the
    tokenizer's eos), or absent."""
    ids: set = set()
    for value in declared:
        ids.update([value] if isinstance(value, int) else (value or []))
    ids.discard(None)
    return ids


def _checkpoint_eos_ids(tokenizer, model) -> set[int]:
    """The ids the checkpoint itself ends a turn on: the tokenizer's ``eos_token_id``
    plus the generation config's, which is what ``generate`` stops on."""
    return _eos_id_set(tokenizer.eos_token_id,
                       getattr(model.generation_config, "eos_token_id", None))


def _declared_eos_ids(tokenizer, src: str) -> set[int]:
    """The same ids as ``_checkpoint_eos_ids``, read from a tokenizer the caller
    already has plus the checkpoint's own generation config, so any engine can check
    the contract before it commits to a weight load without loading a second
    tokenizer just for this.

    ``generation_config.json`` is optional; where it is absent transformers falls back
    to ``config.json`` when it builds a model, and so does this -- reading only the first
    would report fewer terminators than the model itself declares. ``_from_model_config``
    is that fallback's own flag (transformers generation/utils.py): it also reads the
    ``decoder``/``generator``/``text_config`` sub-configs, where a vision-language
    checkpoint keeps its eos id. Without it this reports a narrower set than the loaded
    model ends on and the guard below lets a narrowing pin through."""
    from transformers import GenerationConfig  # type: ignore[import-untyped]
    try:
        generation = GenerationConfig.from_pretrained(src, local_files_only=True)
    except OSError:
        generation = GenerationConfig.from_pretrained(
            src, config_file_name="config.json", local_files_only=True,
            _from_model_config=True)
    return _eos_id_set(tokenizer.eos_token_id,
                       getattr(generation, "eos_token_id", None))


def _check_pinned_stop_ids(checkpoint_ids: set[int], pinned: tuple[int, ...] | None) -> None:
    """Refuse a contract whose pinned end-of-turn ids omit one the checkpoint itself
    ends on. The reference replaces the checkpoint's set with the pinned one, while vLLM
    and SGLang can only add to it, so an omitted id still stops a pool member and a
    contestant server while the reference reads it as an ordinary token. Nothing on the
    config plane can check this: only the checkpoint knows its own ids."""
    if pinned is None:
        return
    missing = sorted(checkpoint_ids - set(pinned))
    if missing:
        raise RuntimeError(
            f"the contract pins stop_token_ids={list(pinned)}, which omits the "
            f"checkpoint's own end-of-turn id(s) {missing}. vLLM and SGLang can only add "
            f"stop ids to the checkpoint's own, so a server and the reference would stop "
            f"in different places; add {missing} to the contract's stop_token_ids.")


def _auto_class_name(config) -> str:
    """The transformers auto-class whose module tree matches a checkpoint's stored keys.

    Read off the checkpoint's own config rather than assumed: a multimodal checkpoint
    stores its transformer body under ``model.language_model.*`` while the text-only
    classes of the same family expect ``model.layers.*``, with no declared conversion
    between the two. Loading the multimodal layout through the text-only class does not
    fail -- it builds a differently-shaped tree, loads nothing into it, and leaves the
    whole body at its random initialization, which the weight-coverage guard in
    ``HFEngine._load`` then refuses.

    The signal is a declared ``vision_config`` plus a generative
    ``...ForConditionalGeneration`` architecture, not the architecture name alone: that
    name suffix is shared by ~30 text-only encoder-decoder families (T5, BART, Pegasus,
    ...), whose keys the image-text class would not fit either. Those belong to neither
    branch here -- an encoder-decoder oracle wants ``AutoModelForSeq2SeqLM`` and is not
    supported; task-facing references are decoder-only or vision-language decoders.

    Returns the class name (not the class) so the resolution is testable without torch.
    """
    arch = ((getattr(config, "architectures", None) or [""])[0]) or ""
    is_multimodal = getattr(config, "vision_config", None) is not None
    return ("AutoModelForImageTextToText"
            if is_multimodal and arch.endswith("ForConditionalGeneration")
            else "AutoModelForCausalLM")


def _frequency_penalty_processor(penalty: float, prompt_len: int):
    """OpenAI-style frequency penalty for ``transformers.generate``: subtracts
    ``penalty * count(v)`` from each logit, counting only the tokens generated so far
    (``input_ids[:, prompt_len:]``). ``prompt_len`` is the padded prompt width of the
    batch, so the count never includes the prompt or its padding.

    Transformers has no built-in frequency penalty (only the multiplicative
    ``repetition_penalty``, which also reads the prompt), so the pinned decode supplies
    its own. Returns a ``transformers.LogitsProcessor``; the class is defined here
    rather than at module level because its base class pulls in torch, and the config
    plane imports this module on a torch-free host.

    The generation loop casts the logits to fp32 before it runs the processors, so the
    subtraction happens in the same precision the teacher-forcing path uses."""
    import torch  # type: ignore[import-untyped]
    from transformers import LogitsProcessor  # type: ignore[import-untyped]

    class _FrequencyPenalty(LogitsProcessor):
        """The penalty processor built by ``_frequency_penalty_processor``."""

        def __call__(self, input_ids, scores):
            out_ids = input_ids[:, prompt_len:]
            counts = torch.zeros(scores.shape[0], scores.shape[-1],
                                 dtype=torch.int64, device=scores.device)
            counts.scatter_add_(1, out_ids, torch.ones_like(out_ids))
            return scores - penalty * counts.to(scores.dtype)

    return _FrequencyPenalty()


#: Query rows per teacher-forcing forward pass when a reference declares no chunk of
#: its own (``HFReference.query_chunk``). The value that fits the longest context any
#: current task declares, so a reference that says nothing gets one that runs: at
#: 131,072 tokens 512 rows peak at 113.4 GiB of an H200's 139.8, and 1,024 and
#: above run out of memory. ``HFEngine._continuation_hidden`` explains the terms.
DEFAULT_QUERY_CHUNK: int = 512

#: How many rows one output-head block covers. ``rows x vocab`` fp32 logits plus the
#: fp32 log-softmax is the one whole-sequence allocation the query chunking does not
#: bound, so the head is applied a block at a time. Every scored number is a per-row
#: reduction, so the block size changes none of them.
_HEAD_BLOCK: int = 2048

#: Decimals the reference rounds its reported log-probabilities to. The reply carries
#: twenty of them per position, and six decimals is a part in a million of an atom's
#: probability -- well under the rounding the served report itself arrives with on either
#: serving stack's wire. Changing it changes what a calibration measures.
_REPORT_DECIMALS: int = 6


def _round_rows(rows: list[list[float]]) -> list[list[float]]:
    """Every value in a block of rows, rounded to ``_REPORT_DECIMALS``."""
    return [[round(v, _REPORT_DECIMALS) for v in row] for row in rows]


def _outside_vocabulary(continuation: list[int], report_ids: list[list[int]],
                        vocab: int) -> str | None:
    """The reason one item's ids cannot be scored, or ``None`` when they all can.

    Both the served ids and the reported ones come from the contestant, so both must be
    range-checked on the host before any tensor indexes with them. The pad id -1 is a
    slot that carries no mass and is allowed."""
    bad = next((t for t in continuation if not 0 <= t < vocab), None)
    if bad is not None:
        return f"served token id {bad} is outside the model's vocabulary of {vocab}"
    for j, row in enumerate(report_ids):
        bad = next((t for t in row if not (t == PAD_ID or 0 <= t < vocab)), None)
        if bad is not None:
            return (f"position {j} reports token id {bad}, outside the model's "
                    f"vocabulary of {vocab}")
    return None


def _kv_cache(decoder_config):
    """The key-value cache a checkpoint's declared layer types ask for.

    transformers builds this itself when a forward is handed no cache
    (``DynamicCache(config=...)``), but it first trims the layer-type list with
    ``layer_types[: -decoder_config.num_kv_shared_layers]`` whenever that field is
    present. A checkpoint that shares no keys and values declares 0, and
    ``layer_types[:-0]`` is the empty list, so the constructor builds no layers at all
    and falls back to replicating a full-length layer for every one of them. On a
    60-layer checkpoint whose config declares 50 of those layers sliding at a
    1,024-token window, that keeps the whole sequence 50 extra times, and also widens
    each sliding layer's score matrix to the whole sequence, because the mask builder
    asks the cache layer how many keys a query block faces. At 131,072 tokens the cache
    alone is then 110 GiB rather than 10.8 GiB, which no single card holds beside the
    weights.

    A sliding layer masks out every key beyond its window either way, so the windowed
    cache changes no attended value: only how much is stored and how wide the score
    matrix is.

    Takes the model's decoder (text) config, the same one ``DynamicCache`` reads, and
    maps every declared layer type the way transformers does, so this differs from
    transformers' own construction only in the shared-layer trim. A config declaring no
    layer types gets a plain ``DynamicCache``, whose lazily replicated full-length
    layers are what a checkpoint with no declared attention structure wants; where such
    a config still declares a ``sliding_window``, transformers would window every layer
    and this helper windows none, which costs memory and no accuracy. A
    config that really does share the keys and values of its trailing layers has no
    cache to build for them, so those are dropped, as transformers intends.
    """
    from transformers.cache_utils import (  # type: ignore[import-untyped]
        Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer,
        LinearAttentionAndFullAttentionLayer, LinearAttentionLayer)

    layer_types = getattr(decoder_config, "layer_types", None)
    if not layer_types:
        return DynamicCache()
    shared = getattr(decoder_config, "num_kv_shared_layers", 0) or 0
    if shared:
        layer_types = layer_types[:-shared]
    # Two spellings of a bounded reach. The cache holds the same states for either and
    # only the mask tells them apart, which is how transformers reads the two names too.
    sliding = ("sliding_attention", "chunked_attention")
    window = (getattr(decoder_config, "sliding_window", None)
              or getattr(decoder_config, "attention_chunk_size", None))
    if window is None and any(t in sliding for t in layer_types):
        raise ValueError(
            "the checkpoint declares layers whose attention reach is bounded but no "
            "sliding_window or attention_chunk_size to bound it by, so the cache its "
            "config asks for cannot be built; refusing rather than silently caching "
            "every layer at full length.")
    layers: list = []
    for layer_type in layer_types:
        if layer_type in sliding:
            layers.append(DynamicSlidingWindowLayer(sliding_window=window))
        elif layer_type in ("mamba", "conv", "linear_attention", "moe"):
            layers.append(LinearAttentionLayer())
        elif layer_type == "hybrid":
            layers.append(LinearAttentionAndFullAttentionLayer())
        else:
            layers.append(DynamicLayer())
    return Cache(layers=layers)


def _final_logit_softcap(config) -> float | None:
    """The logit softcapping a checkpoint declares, or None when it declares none.

    A Gemma checkpoint declares one, and every stack that serves it compresses each
    logit by ``tanh(logit / cap) * cap`` before anything reads the distribution: the
    model's own forward returns capped logits, and vLLM caps them in ``compute_logits``
    ahead of the sampler. An oracle that applies the output head directly bypasses that
    forward, so it has to apply the cap itself or it scores a distribution the model
    never emits. Takes the config the model was built from and reads the cap off its
    text sub-config, which is both where the model's own forward reads it and where a
    vision-language config carries it.
    """
    cap = getattr(config.get_text_config(), "final_logit_softcapping", None)
    return None if cap is None else float(cap)


def _softcapped(logits, cap: float | None):
    """*logits* compressed by a checkpoint's final logit softcapping (_final_logit_softcap),
    or handed back untouched when the checkpoint declares none.

    Taken in fp32, on the fp32 logits the caller already holds. A model applies the same
    tanh in its own weight dtype, so the two differ by one bf16 rounding, which is
    smaller than the noise floor a calibration measures; fp32 is the exact one and is
    also what vLLM caps in.
    """
    import torch  # type: ignore[import-untyped]
    if cap is None:
        return logits
    return torch.tanh(logits / cap) * cap


class ReferenceEngine(Engine):
    """Runtime base for reference engines (one GPU slot, or as many as the engine
    shards the weights over)."""

    #: Whether the engine computes on the GPU it is placed on (``device_index``), so that the
    #: engine host can hold it beside another engine on other GPUs of the same role
    #: (``hls.verify.inprocess``). False for an engine that takes whatever GPUs its process
    #: sees: a serving stack, or one that maps its model over every device or names its own.
    placeable: ClassVar[bool] = False

    def __init__(self, *, model_spec: ModelSpec, decode: DecodeConfig,
                 gpu_kind: str | None = None, conditioning: str = "chat",
                 stop_token_ids: tuple[int, ...] | None = None,
                 memory_mb: int | None = None, gpu_count: int = 1) -> None:
        self.model_spec = model_spec
        self.decode = decode
        if conditioning not in _CONDITIONINGS:
            raise ValueError(
                f"conditioning must be one of {list(_CONDITIONINGS)}, got {conditioning!r}")
        #: How the prompt reaches the model, off the contract's wire protocol
        #: (``contract_conditioning``): "chat" renders the checkpoint's chat template,
        #: "raw" feeds the final message's content verbatim.
        self.conditioning = conditioning
        #: The contract's end-of-turn ids, or None for the checkpoint's own eos set.
        #: Declared ids replace that set here; a serving stack can only extend it, so
        #: every engine's ``_load`` refuses a set that omits one of the checkpoint's own
        #: (``_check_stop_token_ids``).
        self.stop_token_ids = tuple(stop_token_ids) if stop_token_ids is not None else None
        # Accelerator kind for the oracle slot. The oracle's accelerator is
        # harness configuration, not task semantics -- oracle output must be
        # accelerator-invariant for task-level metrics, so this must never become
        # part of the task contract. Until a harness-level knob exists,
        # ``build_engine`` threads the task's kind through as a pragmatic default, so
        # the oracle asks for the accelerator the task declares.
        self.gpu_kind = gpu_kind
        # Host memory the engine needs. A pool member that loads a task's tool assets needs
        # room for the bundle; every other reference declares none.
        self.memory_mb = memory_mb
        if gpu_count < 1:
            raise ValueError(f"gpu_count must be at least 1 (it is a count of devices), got {gpu_count}")
        #: Devices this engine's slot holds: the tensor-parallel width of an engine that
        #: shards the weights across cards, which stores the two as one number so a slot
        #: sized differently from the sharding cannot be written; 1 for every other engine.
        self.gpu_count = gpu_count
        #: The first GPU, by its ordinal among those the process sees, that the engine host
        #: placed the engine on before its load; None when it was placed on none, and
        #: the engine then loads on the process's current device. Set only for an engine that
        #: places itself (``placeable``). Never part of the reference's config, its fingerprint
        #: or what calibration records.
        self.device_index: int | None = None
        self._device: str | None = None
        self._tokenizer = None
        self._model = None

    @property
    def resources(self) -> ResourceRequest:
        """``gpu_count`` GPUs of the engine's kind, and the host memory it declares."""
        return ResourceRequest(gpu_count=self.gpu_count, gpu_kind=self.gpu_kind, memory_mb=self.memory_mb)

    def _render_context(self, messages: list[dict],
                        tools: list[dict] | None = None,
                        continue_final_message: bool = False) -> str:
        """The prompt text one conversation becomes, conditioned as the contract says:
        the checkpoint's chat template with the generation prompt appended, or, under
        raw conditioning, the final message's content untouched (a fill-in-the-middle
        prompt already carries its sentinels).

        ``tools`` is the turn's tool definitions (``tools_of``), which the chat
        template folds into the prompt, so they are part of the context an oracle must
        reproduce. Raw conditioning has no template to fold them into and ignores them,
        which is what the server saw too.

        A conversation with no message left renders the empty prompt, which is what
        ``OpenAICompletionProtocol._prompt_of`` puts on the wire for the same list."""
        if self.conditioning == "raw":
            return message_content(messages[-1]) if messages else ""
        return chat_prompt_text(self._tokenizer, messages, tools,
                                self.model_spec.chat_template_kwargs, continue_final_message)

    def _context_ids(self, messages: list[dict],
                     tools: list[dict] | None = None,
                     continue_final_message: bool = False) -> list[int]:
        """``_render_context`` tokenized with ``add_special_tokens=False`` (the
        rationale is on ``chat_prompt_ids``, which this wraps under chat).

        ``continue_final_message`` renders the prompt of a turn the harness cut and
        handed back: no generation prompt, the last assistant message resumed. Raw
        conditioning ignores it, having no template to instruct -- a raw continuation
        is the prompt with more text on the end."""
        if self.conditioning == "raw":
            return self._tokenizer(self._render_context(messages),
                                   add_special_tokens=False).input_ids
        return chat_prompt_ids(self._tokenizer, messages, tools,
                               self.model_spec.chat_template_kwargs, continue_final_message)

    def _check_stop_token_ids(self, tokenizer, src: str) -> None:
        """Refuse the contract's pinned end-of-turn ids when they omit one the
        checkpoint itself ends on.

        Every engine calls this from its ``_load``, because every engine can be a task's
        oracle: the reference replaces the checkpoint's set with the pinned one while a
        serving stack can only add to it, so an omitted id stops a contestant server
        where the reference reads an ordinary token. Costs nothing until a contract
        actually pins ids, and then reads only the given tokenizer and the checkpoint's
        generation config, so it can run before the weights load. Takes the tokenizer
        rather than loading its own so a caller that needs one anyway (every engine but
        VLLMEngine's chat branch) does not build it twice."""
        if self.stop_token_ids is None:
            return
        _check_pinned_stop_ids(_declared_eos_ids(tokenizer, src), self.stop_token_ids)

    def _weights_source(self) -> str:
        """Where every reference engine loads its weights from: the snapshot of the model's
        pinned commit that provisioning mounts at ``$HLS_WEIGHTS_DIR`` (sealed container, no
        network), read with ``local_files_only``. Raise RuntimeError when none is mounted:
        the checkpoint name alone would read whatever the hub's default branch names, not
        the commit the task pins. Offline enforcement is the role's environment's job -- the
        provision mounts set ``HF_HUB_OFFLINE`` for every role before it starts
        (``hls.provision.mounts``) -- so engines must not re-pin it in-process
        (huggingface_hub snapshots the var at import time, making a late pin a no-op)."""
        import os
        from pathlib import Path
        wdir = os.environ.get("HLS_WEIGHTS_DIR")
        if not wdir or not Path(wdir).is_dir():
            raise RuntimeError(
                f"no weights mounted at $HLS_WEIGHTS_DIR ({wdir!r}); a reference loads only the "
                f"provisioned snapshot of {self.model_spec.base_checkpoint}@{self.model_spec.revision}")
        return wdir


#: The sampling every decode in this module is pinned to -- the clocked servings and
#: the synchronous oracle alike. Greedy decode is what the token-level gate is defined
#: against, and no stack gives it by default: a checkpoint's ``generation_config.json``
#: ships sampling defaults (Qwen's is temperature 0.7 / top_p 0.8 / top_k 20 /
#: repetition_penalty 1.05), and a repetition penalty alone moves several percent of
#: positions off the reference's argmax -- recorded as the model's noise floor instead
#: of as the sampler it is. Every field is set explicitly so it always wins over a
#: config-supplied default, whatever the stack's precedence rules.
#:
#: One dict for both stacks: these six field names are spelled identically by vLLM's
#: ``SamplingParams`` and by SGLang's sampling-params dict, so the pin has a single
#: definition and the two stacks cannot drift into decoding differently. ``top_k=-1``
#: is the only spelling of "disabled" both stacks take -- sglang expands -1 to the
#: whole vocabulary and rejects 0, vLLM prefers 0 but accepts -1 -- though under this
#: dict the value is moot, because each stack overrides top_k with its own greedy
#: setting once temperature is 0. ``max_tokens``/``max_new_tokens`` is left out: it is
#: the workload's per-request output cap, stamped per request.
_GREEDY_SAMPLING: dict[str, float | int] = {
    "temperature": 0.0, "top_p": 1.0, "top_k": -1,
    "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
}


def _sampling_for(decode: DecodeConfig) -> dict[str, float | int]:
    """The pinned sampling dict for ``decode``, for either stack.

    Under ``Greedy`` it is ``_GREEDY_SAMPLING`` with the decode's frequency penalty over
    it, which at a 0.0 penalty is ``_GREEDY_SAMPLING`` exactly. Both stacks spell the
    field ``frequency_penalty`` and both subtract an additive per-occurrence penalty over
    the request's own output tokens, in fp32, before sampling, so one definition covers
    them.

    Under ``Sample`` it is those fields with the temperature in place of 0, plus
    ``min_p``, and every truncation and penalty explicitly off, so the distribution
    served is the model's own softmax over the whole vocabulary. Setting them explicitly
    matters more here than under greedy: a checkpoint's ``generation_config.json`` ships
    top-p, top-k and repetition-penalty defaults that a zero temperature makes moot but
    a nonzero one does not, so an unset field would quietly truncate the distribution
    being graded. ``min_p`` is pinned only here, for the same reason and no other: it is
    a second truncation knob, moot at temperature 0, and the starter servers pin it too.
    Adding it to the greedy dict would change what a token-level calibration sends.

    ``top_k`` is re-pinned here rather than inherited, because a nonzero temperature is
    exactly the case where neither stack overrides it: the -1 is the live "no
    truncation" value the graded distribution rests on, not a moot one.

    No draw seed: the decode pins none, and the report gate reads the distribution the
    server reported rather than replaying its draw."""
    from hls.core.decode import Greedy, Sample
    if isinstance(decode, Greedy):
        return {**_GREEDY_SAMPLING, "frequency_penalty": decode.frequency_penalty}
    if isinstance(decode, Sample):
        return {**_GREEDY_SAMPLING, "temperature": decode.temperature,
                "top_k": -1, "min_p": 0.0}
    raise TypeError(
        f"the pinned sampling dict is defined for Greedy and Sample decode, got "
        f"{decode!r}")


class _StopSamplingParams:
    """The end-of-turn half of the sampling parameters a serving stack takes.

    Mixed into the two serving engines and only those: the value is a vLLM
    ``SamplingParams`` / SGLang sampling-dict fragment, which HFEngine has no way to
    pass -- ``generate`` takes ``eos_token_id`` instead (see ``HFEngine._eos_ids``)."""

    #: The contract's pinned ids, set by ``ReferenceEngine.__init__``.
    stop_token_ids: tuple[int, ...] | None

    def _stop_sampling(self) -> dict:
        """The sampling fields a serving stack needs to end a turn on the contract's ids,
        or ``{}`` when the checkpoint's own set applies.

        A serving stack can only extend the checkpoint's end-of-turn set, never replace
        it: sglang nests its ``stop_token_ids`` check inside the ``ignore_eos`` guard, so
        turning eos off there turns the pinned ids off too and the reply runs to the
        output cap. Extension and replacement agree whenever the pinned set already
        covers the checkpoint's own ids, which every engine's ``_load`` enforces
        (``ReferenceEngine._check_stop_token_ids``). No output cap is needed beside these
        fields: the driver stamps ``max_tokens`` on every request, and the context window
        bounds the rest.

        ``stop_token_ids`` is spelled identically by vLLM's ``SamplingParams`` and by
        SGLang's sampling-params dict, so the four call sites share one definition, as
        with ``_GREEDY_SAMPLING``."""
        if self.stop_token_ids is None:
            return {}
        return {"stop_token_ids": list(self.stop_token_ids)}


class _EngineClient:
    """An ``ArtifactClient`` over an engine's own async serving API.

    The drive loop only requires ``async submit(envelope) -> ArtifactResponse``
    (``hls.spec.drivers.ArtifactClient``), so this is all it takes to point the real
    driver at an in-process engine instead of an HTTP server: no wire protocol, no
    transport, no second process -- just the request, rendered and awaited. The reply
    carries the serving facts the wire protocol would have decoded off a graded server's
    (``_served_metadata``), because the task's own framework branches on them.

    A generation that raises is converted to an ``ArtifactTransportError``, which the
    drive records as a faulted event rather than aborting mid-serving. The caller
    (``hls.calibrate``) then refuses the whole serving on any fault: a trusted pool
    member that could not serve its own task's declared load has not produced a
    measurement, and dropping the faulted requests would silently measure a smaller,
    easier workload."""

    def __init__(self, engine: "ClockedServing") -> None:
        self._engine = engine

    async def submit(self, env: Envelope, *, attempt=None):
        """Serve one request in-process.

        ``attempt`` (``hls.core.trace.RequestAttempt``) is accepted and left untouched:
        a generation either returns the whole reply or raises, so there are no partial
        frames to record and nothing here is a witness to anything the caller cannot
        already see."""
        from hls.core.faults import ArtifactTransportError
        from hls.core.trace import ArtifactResponse
        try:
            (text, finish_reason, tokens, report,
             prompt_tokens) = await self._engine._serve_one(env)
        except Exception as exc:  # noqa: BLE001 -- any engine failure is a faulted request
            raise ArtifactTransportError(
                f"{type(self._engine).__name__} failed to serve request "
                f"{env.request_id!r}: {exc!r}") from exc
        return ArtifactResponse(
            output=_reference_reply(
                env.request_id, text, finish_reason, tokens, report,
                _served_metadata(env, prompt_tokens, len(tokens),
                                 self._engine.max_model_len)))


class TorchStackReport(StackReport):
    """``StackReport`` for any engine running on a torch/CUDA stack.

    Two readers, one implementation. The deployed reference's report is what the
    ``TokenFaithfulnessVerifier`` compares against the calibrated record to refuse a
    stack that drifted; every calibration allocation's report is what proves it landed
    on the task's declared accelerator, which matters most when a campaign holds
    several allocations at once and only the allocation's request says which silicon
    each one got."""

    async def stack_report(self) -> dict:
        """This running container's software stack (shape:
        ``hls.verify.calibration.StackInfo``). Needs no model load -- versions are
        process/device facts.

        NVIDIA-only today, and loud about it: a torch build without a CUDA runtime
        (CPU-only, or ROCm -- where ``torch.version.cuda`` is None even though
        ``torch.cuda`` works over HIP) raises instead of reporting missing or
        misleading component versions. Supporting another vendor means emitting that
        vendor's component set under ``runtime`` (e.g. hip/miopen), not changing the
        StackInfo schema."""
        import torch  # type: ignore[import-untyped]
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise RuntimeError(
                "stack_report supports NVIDIA CUDA stacks only: "
                f"torch.cuda.is_available()={torch.cuda.is_available()}, "
                f"torch.version.cuda={torch.version.cuda!r}. A calibration stack "
                "record from this build would be meaningless -- add this vendor's "
                "component versions to the report instead of running blind.")
        # Every leaf is str()-wrapped to an exact builtin (``torch.__version__`` is a
        # ``TorchVersion``, a str subclass), so the report is plain data wherever it is
        # recorded.
        # The GPU the engine was placed on (``ReferenceEngine.device_index``), else the
        # process's first, so each engine of a role holding two names its own.
        index = getattr(self, "device_index", None)
        return {"torch": str(torch.__version__),
                "runtime": {"cuda": str(torch.version.cuda),
                            "cudnn": str(torch.backends.cudnn.version())},
                "device_name": str(torch.cuda.get_device_name(0 if index is None else index))}


class ClockedServing(ServeTraffic, ServingVersion):
    """``ServeTraffic`` for an engine whose serving API takes one request at a time,
    asynchronously.

    The whole implementation is: build the driver ``hls.spec.drivers.driver_for``
    selects for the task, hand it an ``_EngineClient``, and let it run. Arrival
    schedules, think-time draws, ramp-down membership, session history stitching, the
    drive deadline and the consecutive-fault breaker are therefore the same code that
    drives a graded contestant (``Driver._drive_sessions``) -- not a reimplementation
    that could drift. An agentic task is served as trajectories for the same reason,
    with its framework and tools running in this process. What differs is only what
    sits behind ``submit``.

    Because requests are submitted as they come due rather than as one saturated
    batch, the engine batches exactly what the arrival process put in flight, which
    is the regime a graded run measures. Subclasses supply the two engine-specific
    halves: ``_serve_load`` (bring the async engine up) and ``_serve_one`` (render one
    conversation and await its generation)."""

    #: The engine's model, whose chat-template pin a trajectory renders its prompts with.
    model_spec: ModelSpec

    async def serve_traffic(self, loaded: list, *, seed: int, request_timeout_sec: float,
                            max_sessions: int | None = None,
                            drive_timeout_sec: float | None = None,
                            agent=None, max_context_len: int | None = None):
        """Drive one clocked serving and return its ``Trace`` (see ``ServeTraffic``). A
        trajectory is handed the task's window and this engine's model's chat-template pin."""
        from hls.core.trace import Trace
        from hls.spec.drivers import driver_for

        await self._serve_load()
        driver = driver_for(agent, max_context_len=max_context_len,
                            chat_template_kwargs=self.model_spec.chat_template_kwargs,
                            request_timeout_sec=request_timeout_sec,
                            drive_timeout_sec=drive_timeout_sec)
        events: list = []
        # The serving drives its own materialized sessions through _drive_sessions, so the
        # driver needs no tenants and no artifact client of its own.
        await driver._drive_sessions(_EngineClient(self), events.append, loaded,
                                     drive_seed=seed, max_sessions=max_sessions)
        return Trace(events=events)

    async def serving_version(self) -> str:
        """The installed version of ``_serving_package`` (see ``ServeTraffic``)."""
        from importlib.metadata import version
        return str(version(self._serving_package))

    #: Distribution name of the serving package, as its installed metadata records it.
    _serving_package: ClassVar[str]

    #: The context window this engine serves at, or None to serve at the checkpoint's
    #: own. Every reply reports it the way a graded server reports ``max_context_len``,
    #: which is what an agentic framework budgets its prompts against.
    max_model_len: int | None

    @abstractmethod
    async def _serve_load(self) -> None:
        """Bring this engine's async serving API up (idempotent)."""

    @abstractmethod
    async def _serve_one(self, prompt: Envelope) -> tuple[
            str, str | None, list[int], list[TokenLogprob] | None, int]:
        """Serve one request: render its conversation through the model's chat
        template and await the generation. Returns ``(text, finish_reason,
        token_ids, report, prompt_tokens)`` -- the batch paths' triple, ids included so
        the pinned reference can teacher-force the exact continuation; this reply's
        report (``_served_report``) when the decode calls for one (``_report_width``)
        and ``None`` when it does not; and the length of the rendered prompt, which the
        reply reports as a graded server reports its own (``_served_metadata``)."""


class HFEngine(ReferenceEngine, Generate, TeacherForce, ReferenceLogprobs,
               TorchStackReport):
    """HuggingFace transformers reference engine. Loads lazily on first generate so
    import + TaskSpec construction stay cheap. teacher_force runs a greedy per-token
    forward pass returning per-token logprob gaps. reference_logprobs runs the same
    forward pass and returns the reference distribution at the ids a served reply
    reported, which is what the sampling gate compares against. stack_report exposes
    the running container's torch/CUDA/cuDNN/GPU-name versions for the calibration
    drift check.

    The engine is a fixed numeric instrument: attention is pinned to one SDPA kernel
    (``sdpa_backend``) with the nondeterministic cuDNN-SDP backend disabled, fused
    linear-attention kernels are pinned to the torch reduction
    (``hls.reference.kernel_pins``), the teacher-forcing forward is tiled at one
    declared width (``query_chunk``),
    and there is no CPU fallback (a CPU forward is a different instrument). This is what
    lets the calibrated token gate trust that generate and teacher_force agree
    bit-for-bit.

    ``_load`` additionally refuses two ways of not being the checkpoint it claims: the
    model class is resolved from the checkpoint's own ``config.architectures`` (a
    vision-language layout loaded through the text-only auto-class builds a
    differently-shaped tree and silently keeps its random initialization), and every
    parameter of the built model must appear in the checkpoint's own weight inventory."""

    #: The engine names its own device, the one it was placed on (``ReferenceEngine.placeable``).
    placeable: ClassVar[bool] = True

    def __init__(self, *, model_spec: ModelSpec, decode: DecodeConfig,
                 gpu_kind: str | None = None, conditioning: str = "chat",
                 stop_token_ids: tuple[int, ...] | None = None,
                 sdpa_backend: str = "math", dtype: str = "auto",
                 query_chunk: int = DEFAULT_QUERY_CHUNK) -> None:
        super().__init__(model_spec=model_spec, decode=decode, gpu_kind=gpu_kind,
                         conditioning=conditioning, stop_token_ids=stop_token_ids)
        #: Fused linear-attention entry points the load pinned to the torch fallback, as
        #: ``{entry_point: "torch_fallback"}``; ``{}`` for a checkpoint family with no fused
        #: path, ``None`` before the first load. Diagnostic state: it names what the pin
        #: covered when the post-construction check refuses a model, and is readable on the
        #: engine when a reference has to be inspected. The pin itself is unconditional, so
        #: nothing branches on this.
        self._kernels_in_force: dict[str, str] | None = None
        # Validate the instrument pins at construction (pure string checks, no torch)
        # so a bad reference config fails on the torch-free config host, before any
        # deploy or GPU acquire.
        if sdpa_backend not in _SDPA_BACKENDS:
            raise ValueError(
                f"sdpa_backend must be one of {list(_SDPA_BACKENDS)}, got {sdpa_backend!r}")
        if dtype not in _HF_DTYPES:
            raise ValueError(f"dtype must be one of {list(_HF_DTYPES)}, got {dtype!r}")
        if query_chunk < 1:
            raise ValueError(
                f"query_chunk must be at least 1 (it is a count of query rows), "
                f"got {query_chunk}")
        # sdpa_backend pins the attention kernel for all forwards; "math" is the
        # reference contract (deterministic, shape-invariant), and every engine the
        # harness builds uses it: a task's reference and calibrate's fp32 anchor. "flash"
        # and "default" are not task semantics and no caller builds them today (the
        # calibration pool is vLLM and SGLang, ``hls.calibrate.POOL_ENGINE_NAMES``).
        # "flash" supports only causal masking: padded batched generate (transformers
        # passes a padding attn_mask) raises "No available kernel" under the pin, so
        # flash serves single-sequence teacher_force and unpadded batch-1 generate only.
        # dtype pins compute precision: "auto" = bf16
        # (the deployed reference), "float32" = the fp32 anchor calibrate teacher-forces
        # against at preflight. HFReference.build_engine passes neither, so a task's
        # reference is always the ("math","auto") instrument.
        self.sdpa_backend = sdpa_backend
        self.dtype = dtype
        # query_chunk trades peak attention memory against forward count on the
        # teacher-forcing path (_continuation_hidden). Rows are independent under the
        # math kernel, but the kernel tiles differently at different widths, so two
        # chunk sizes are two instruments to a gap near a threshold -- which is why the
        # value is declared per reference and travels in its fingerprint rather than
        # being tuned here.
        self.query_chunk = query_chunk

    def _sdpa_ctx(self):
        """The pinned-SDPA context every forward (generate + teacher_force) runs inside.

        Pinning which SDPA kernel runs makes the reference a fixed instrument: PyTorch
        would otherwise dispatch attention to different fused kernels by shape/dtype, so
        the "same" oracle could yield different logits for a batched vs single forward
        (the bf16 fragility the calibrated gate must not inherit). Built
        here rather than at module import because torch.nn.attention is torch-only and
        the config plane importing this module must stay torch-free."""
        import contextlib
        if self.sdpa_backend == "default":
            # Stock dispatch: no kernel pin at all (calibrate pool member only).
            return contextlib.nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore[import-untyped]
        backends = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}
        return sdpa_kernel([backends[self.sdpa_backend]])

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore[import-untyped]
        # No CPU fallback: a CPU forward is a different numeric instrument (different
        # kernels, no SDPA pin) whose gaps would not match the CUDA-calibrated
        # thresholds. Fail loud rather than silently swap the instrument; checked before
        # the heavy transformers import so the refusal is instant.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "HFEngine requires CUDA; a CPU reference is a different instrument "
                "(fail loud, no fallback)")
        import transformers  # type: ignore[import-untyped]
        from transformers import AutoConfig, AutoTokenizer  # type: ignore[import-untyped]
        # The GPU the engine host placed the engine on; the process's current one when it
        # placed it on none. Every tensor made from here on uses this device.
        self._device = "cuda" if self.device_index is None else f"cuda:{self.device_index}"
        dtype = torch.float32 if self.dtype == "float32" else torch.bfloat16
        # Turn off the cuDNN SDP backend for the pinned instruments: its fused attention
        # is nondeterministic across shapes/versions, so leaving it eligible would let
        # the "same" reference drift run-to-run. attn_implementation="sdpa" routes
        # attention through the SDPA path, and the per-forward sdpa_kernel([...])
        # context (self._sdpa_ctx) then pins which SDPA kernel actually runs --
        # together the instrument is fixed. The "default" backend leaves stock
        # dispatch alone (cuDNN included), as an unpinned server's stack would; no caller
        # builds it today (the calibration pool is vLLM and SGLang).
        if self.sdpa_backend != "default":
            torch.backends.cuda.enable_cudnn_sdp(False)
        # Weights come from the shared source policy, the mounted snapshot of the pinned
        # commit, read with local_files_only so a sealed container never reaches the network.
        src = self._weights_source()
        self._tokenizer = AutoTokenizer.from_pretrained(src, local_files_only=True)
        # Checked before the weights load: the checkpoint's own terminators are readable
        # from its config files and the tokenizer just loaded, with no weight load yet.
        self._check_stop_token_ids(self._tokenizer, src)
        # Read the checkpoint's config first: it decides both the auto-class the stored
        # keys fit and the kernel family that has to be pinned before construction.
        config = AutoConfig.from_pretrained(src, local_files_only=True)
        self._kernels_in_force = pin_fused_kernels(getattr(config, "model_type", "") or "")
        auto = getattr(transformers, _auto_class_name(config))
        self._model, loading_info = auto.from_pretrained(
            src, dtype=dtype, attn_implementation="sdpa", local_files_only=True,
            output_loading_info=True)
        # Both instrument guards run before the model is usable, so a mis-loaded or
        # unpinned reference can never score a token. The evidence is the loader's own
        # report: a missing key is a parameter the checkpoint did not supply, still at
        # its random initialization, and a mismatched key was re-initialized because the
        # stored tensor had another shape. Unexpected keys are tolerated, because a
        # multi-token-prediction head transformers does not instantiate is inert at serve
        # time too. The report is read after the loader's renames and fusions: a
        # mixture-of-experts checkpoint stored one tensor per expert has supplied its
        # weights once the loader has fused them. A mismatched entry is the
        # name followed by the two shapes.
        unfilled = sorted(set(loading_info["missing_keys"]) | {
            k[0] if isinstance(k, tuple | list) else k
            for k in loading_info["mismatched_keys"]})
        if unfilled:
            raise RuntimeError(
                f"{type(self._model).__name__} built from {src!r} has "
                f"{len(unfilled)} parameter(s) the checkpoint did not fill -- reported "
                f"missing or shape-mismatched by the loader, so they are randomly "
                f"initialized (e.g. {unfilled[:5]}). The reference is not this "
                f"checkpoint; check config.architectures against the auto-class "
                f"resolved for it ({_auto_class_name(config)}).")
        bound = fused_kernel_bindings(self._model)
        if bound:
            raise RuntimeError(
                f"the built reference holds fused kernels the pin should have removed: "
                f"{bound}. The fused linear-attention path is nondeterministic on "
                f"identical input, so it must never back a verdict; the entry points "
                f"pinned for model_type={getattr(config, 'model_type', None)!r} were "
                f"{sorted(self._kernels_in_force)}.")
        self._model.to(self._device)
        self._model.eval()

    def _decode_kwargs(self) -> dict:
        from hls.core.decode import Greedy, Sample
        match self.decode:
            case Greedy():
                # repetition_penalty=1.0 turns off the multiplicative penalty, so the
                # argmax is over the raw logits: plain greedy. The model's
                # generation_config defaults to 1.1, which penalizes tokens already in
                # the prompt -- including the MMLU answer letters (A/B/C/D appear in
                # the choices) -- and its effect differs between a padded batch and a
                # single sequence, so the oracle and a single-request artifact server
                # would diverge and break reference-relative parity. A decode that declares a
                # frequency penalty gets it from the logits processor _generate_uniq
                # attaches, not from here.
                return {"do_sample": False, "repetition_penalty": 1.0}
            case Sample():
                # The full-vocabulary softmax at the decode's temperature: top_k=0 and
                # top_p=1.0 are transformers' spellings of "no truncation", and
                # repetition_penalty=1.0 turns off the multiplicative penalty the
                # checkpoint's generation_config would otherwise supply. Sampling makes
                # these fields live, where a zero temperature made them moot.
                return {"do_sample": True, "temperature": self.decode.temperature,
                        "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
        raise TypeError(f"unsupported decode config {self.decode!r}")

    async def generate(self, prompts: list[Envelope]) -> list[Envelope]:
        """Batched generation under the engine's decode.

        Under ``Sample`` the replies are draws, not a fixed answer, and nothing pins
        them: a prompt's own reply depends on which prompts shared its batch. A
        verifier that compares an artifact against this output as a fixed bar
        therefore needs a ``Greedy`` reference; the report verifier does not use this
        path at all, and grades the distribution a server reported rather than a
        reference draw."""
        import asyncio
        # Load off the event loop -- mirrors VLLMEngine.generate: model load (weights
        # download/materialize) is the heavy, blocking part, and this runs on the
        # role's event loop, which a blocking call would stall.
        await asyncio.to_thread(self._load)
        if not prompts:
            return []
        # Render each prompt's chat messages: the full stitched history from
        # metadata['messages'] when the driver provides it (multi-turn), else the
        # single latest user turn -- single-turn is the degenerate one-message case
        # of the same path, mirroring the artifact (OpenAIChatProtocol also reads
        # metadata['messages']). Tools ride on the render because the chat template
        # folds them into the prompt (tools_of). Dedup is over the rendered prompt,
        # not raw text, so two turns differing only in their tools stay distinct.
        rendered_per_prompt = [self._render_context(_messages_of(p), tools_of(p))
                               for p in prompts]
        uniq = list(dict.fromkeys(rendered_per_prompt))
        max_new = _max_new_tokens(prompts)
        # The batched tokenize + model.generate + decode is the blocking torch/CUDA
        # work (mirrors VLLMEngine's _generate_uniq) -- run it in one to_thread call so
        # it never touches the event loop.
        decoded, finish, tokens = await asyncio.to_thread(self._generate_uniq, uniq, max_new)
        return [_reference_reply(p.request_id, decoded[rp], finish[rp], tokens=tokens[rp])
                for p, rp in zip(prompts, rendered_per_prompt)]

    def _generate_uniq(
        self, uniq: list[str], max_new: int | None
    ) -> tuple[dict[str, str], dict[str, str], dict[str, list[int]]]:
        """Blocking HF generate over the unique rendered prompts (run via to_thread);
        returns ``(decoded, finish, tokens)`` keyed by rendered prompt -- ``tokens`` is
        the exact emitted ids with trailing batch padding stripped, so a downstream
        teacher_force scores the true continuation. Mirrors VLLMEngine's
        ``_generate_uniq``."""
        import torch  # type: ignore[import-untyped]
        self._tokenizer.padding_side = "left"  # type: ignore[union-attr]
        if self._tokenizer.pad_token_id is None:  # type: ignore[union-attr]
            self._tokenizer.pad_token = self._tokenizer.eos_token  # type: ignore[union-attr]
        batch = self._tokenizer(uniq, return_tensors="pt", padding=True,  # type: ignore[union-attr]
                                add_special_tokens=False).to(self._device)
        eos_ids = self._eos_ids()
        gen_kwargs: dict = {"pad_token_id": self._tokenizer.eos_token_id,  # type: ignore[union-attr]
                            **self._decode_kwargs()}
        if self.stop_token_ids is not None:
            # generate stops only on the ids it is told, so the contract's set has to
            # reach it; left alone it would stop on the checkpoint's eos instead.
            gen_kwargs["eos_token_id"] = sorted(eos_ids)
        if max_new is not None:
            gen_kwargs["max_new_tokens"] = max_new
        # Only a Greedy decode carries a penalty; a sampled one declares none by
        # construction, so there is no processor to attach.
        if getattr(self.decode, "penalized", False):  # type: ignore[union-attr]
            # The prompt width of the padded batch: everything past it is what this
            # request has generated, which is the only history the penalty counts.
            from transformers import LogitsProcessorList  # type: ignore[import-untyped]
            gen_kwargs["logits_processor"] = LogitsProcessorList([
                _frequency_penalty_processor(self.decode.frequency_penalty,  # type: ignore[union-attr]
                                             batch["input_ids"].shape[-1])])
        with torch.no_grad(), self._sdpa_ctx():
            # Pinned SDPA kernel (self.sdpa_backend): generate must be the same numeric
            # instrument as teacher_force, else a faithful reply would score nonzero gaps.
            out = self._model.generate(**batch, **gen_kwargs)  # type: ignore[union-attr]
        gen = out[:, batch["input_ids"].shape[-1]:]
        decoded, finish, tokens = {}, {}, {}
        for r, g in zip(uniq, gen):
            row = g.tolist()
            decoded[r] = self._tokenizer.decode(g, skip_special_tokens=True)  # type: ignore[union-attr]
            finish[r] = _hf_finish_reason(row, eos_ids)
            # The reference's exact emitted ids, trailing batch padding stripped, keyed
            # by rendered prompt like decoded/finish so the dedup mapping stays aligned.
            tokens[r] = _emitted_tokens(row, eos_ids)
        return decoded, finish, tokens

    def _eos_ids(self) -> set:
        """The ids that end a turn: the contract's pinned set when it declares one, else
        the checkpoint's own (``_checkpoint_eos_ids``). ``_load`` has already refused a
        pinned set that drops one of the checkpoint's, so the two never disagree about
        whether a reply has ended."""
        if self.stop_token_ids is not None:
            return set(self.stop_token_ids)
        return _checkpoint_eos_ids(self._tokenizer, self._model)

    async def teacher_force(self, items: list[Envelope]) -> list[list[float]]:
        """Greedy teacher forcing. For each item, render its metadata["messages"]
        context the same way generate does, append the continuation (the contestant's
        exact emitted ids from the item's TextPayload.tokens), run one forward pass,
        and return the per-token greedy gap gap_j = logprob(argmax) - logprob(emitted
        token) >= 0 (0 iff the emitted token is the reference's greedy choice).

        Gaps are measured on the distribution the model itself emits: where the
        checkpoint declares logit softcapping, the cap is applied to the output head's
        logits first, the way the model's own forward and every serving stack apply it.

        Under a decode that declares a frequency penalty, every logit is then reduced
        by the penalty times the number of times that token id appears earlier in this
        same reply, and the log-softmax, the argmax and the gap are all taken over the
        penalized logits. The count restarts with each reply and never includes the
        prompt, so a faithful penalized server's token is the reference's own choice
        and scores 0, exactly as under plain greedy.

        A reply is scored at one further position, its stop, as logprob(argmax) -
        logprob(best end-of-turn id), unless it ended on an end-of-turn id (that token
        was teacher-forced like any other) or its length reached the request's
        ``max_tokens`` (it ran out of budget rather than choosing to end). Without the
        stop row a faithful prefix scores zero gaps however early it was cut off; with
        it an early stop is one large gap that spends the ceilings, and an entitled stop
        scores 0. Both halves of the exemption are the harness's own facts: the cap is
        ``metadata["max_tokens"]`` as ``hls.spec.drivers.stamp_request`` stamped it, and
        the end-of-turn ids come from the checkpoint. The reply's reported
        ``finish_reason`` is not read. An item without ``max_tokens`` raises rather than
        guessing which way the stop should go.

        The loop runs unpadded, one sequence at a time (provably equal to any batched
        variant). Inputs are validated before the
        model loads, so the raise/empty paths run on a torch-free host."""
        import asyncio
        from hls.core.decode import Sample
        if isinstance(self.decode, Sample):
            raise NotImplementedError(
                "HFReference.teacher_force supports only greedy decode; Sample is not implemented.")
        if not items:
            return []
        # Validate + extract (context messages, continuation ids, scores-its-stop)
        # before loading torch. The cap half of the stop test is decided here; the
        # end-of-turn half needs the checkpoint and is decided in
        # _teacher_force_prepared.
        prepared: list[tuple[list[dict], list[dict] | None, bool, list[int], bool]] = []
        for item in items:
            msgs = item.metadata.get("messages")
            if not (isinstance(msgs, list) and msgs):
                raise ValueError(
                    "teacher_force item is missing metadata['messages'] (the conversation "
                    "context); contents holds the continuation, so there is no fallback.")
            tp = item.first_of("text")
            if not isinstance(tp, TextPayload) or tp.tokens is None:
                raise TypeError(
                    "teacher_force item must carry the continuation token ids in its text "
                    "payload (TextPayload.tokens); none present.")
            # Through _messages_of, so the context is the list generate rendered and
            # the wire protocol sent -- an entry missing role or content is dropped
            # here too, instead of raising on it or scoring a message never sent. The
            # turn's tools ride along for the same reason: the chat template folds them
            # into the prompt, so an item that dropped them would be scored against a
            # context the artifact never saw.
            rendered_msgs = _messages_of(item)
            cap = item.metadata.get("max_tokens")
            if not isinstance(cap, int) or cap <= 0:
                raise ValueError(
                    f"teacher_force item {item.request_id!r} carries no positive "
                    f"metadata['max_tokens'] (got {cap!r}); the request's stamped output "
                    f"cap decides whether the reply's stop is scored, so it must be "
                    f"present on every item.")
            prepared.append((rendered_msgs, tools_of(item), continues_of(item), list(tp.tokens),
                             len(tp.tokens) < cap))

        # Load + run off the event loop -- mirrors VLLMEngine.generate / HFEngine.generate:
        # the model load and the per-item forward passes below are the heavy, blocking
        # torch work, and this runs on the role's event loop, which a blocking call would
        # stall. The whole batch runs in one to_thread call, matching how VLLMEngine
        # offloads its batch via _generate_uniq.
        await asyncio.to_thread(self._load)
        return await asyncio.to_thread(self._teacher_force_prepared, prepared)

    def _continuation_hidden(self, messages: list[dict], tools: list[dict] | None,
                             continuation: list[int], decoder_config,
                             continue_final_message: bool = False):
        """One item's forward pass: the hidden states that predict its continuation,
        and the rendered prompt's length in tokens.

        Returns ``[n_cont + 1, h]`` -- one row per served token, plus one past the end
        whose logits predict the token AFTER the last one served. That extra row is what
        says whether the reference would have stopped there; scoring only the emitted
        rows leaves the decision to stop unmeasured, so a faithful prefix of a reply
        would score clean however early it was cut off. A caller that has no use for it
        (the report path, where the server reported nothing at that position) drops it.

        Bounded-memory by construction. The query axis is walked in ``query_chunk``-token
        blocks over the key-value cache because the pinned math SDPA kernel materializes
        the whole score matrix instead of streaming it, and upcasts scores and softmax to
        fp32: 32 heads x chunk x sequence at about 9 bytes an element. At a
        131,072-token context that is 18 GiB for the worst layer at a 512-token chunk and
        144 GiB at 4,096. The same layer also holds fp32 copies of its keys and values,
        about 12 bytes an element and 24 GiB at that length, which no chunk size can
        shrink.

        Rows are independent under math SDPA, so the chunk size changes no scored number
        beyond kernel tiling, and calibration runs through this same engine at the same
        declared chunk, so the deployed gates and the instrument that calibrated them
        stay one implementation.

        The caller applies the output head to these rows itself, in _HEAD_BLOCK-row
        blocks, for a similar reason: rows x vocab logits plus the fp32 softmax is the
        other whole-sequence allocation that cannot be materialized at once. Every
        caller softcaps each block first, the way the model's own forward caps
        everything it returns."""
        import torch  # type: ignore[import-untyped]
        prompt_text = self._render_context(messages, tools, continue_final_message)
        ctx = self._tokenizer(  # type: ignore[union-attr]
            prompt_text, return_tensors="pt", add_special_tokens=False).to(self._device)
        ctx_ids = ctx["input_ids"]
        n_ctx = ctx_ids.shape[-1]
        cont = torch.tensor([continuation], dtype=ctx_ids.dtype, device=self._device)
        full = torch.cat([ctx_ids, cont], dim=1)
        with torch.no_grad(), self._sdpa_ctx():
            # Built here rather than left to the model, which builds one whose
            # sliding layers cache and attend over the whole sequence (_kv_cache).
            past = _kv_cache(decoder_config)
            hidden_parts = []
            for i in range(0, full.shape[1], self.query_chunk):
                o = self._model.model(  # type: ignore[union-attr]
                    full[:, i:i + self.query_chunk], past_key_values=past,
                    use_cache=True)
                past = o.past_key_values
                hidden_parts.append(o.last_hidden_state[0])
            return torch.cat(hidden_parts, 0)[n_ctx - 1:], int(n_ctx)

    def _teacher_force_prepared(
        self, prepared: list[tuple[list[dict], list[dict] | None, bool, list[int], bool]]
    ) -> list[list[float]]:
        """Blocking per-item greedy teacher-forcing forward passes (run via to_thread).
        See ``teacher_force`` for the returned gap semantics.

        The frequency penalty comes off the capped logits: a serving stack subtracts
        from capped logits too, and tanh is not linear, so penalizing raw logits would
        order the argmax differently and charge a faithful server a flip it never made.
        The multiplicative repetition_penalty is off on both sides, so the scored
        distribution is the capped logits under a plain-greedy decode and the capped
        logits less the declared frequency penalty under a penalized one -- matching
        generate either way."""
        import torch  # type: ignore[import-untyped]
        out: list[list[float]] = []
        vocab = self._model.get_input_embeddings().num_embeddings  # type: ignore[union-attr]
        eos_ids = self._eos_ids()
        # Sample decode is refused by teacher_force before it gets here, so the decode
        # is a Greedy and carries the penalty fields. False leaves every tensor
        # operation below exactly as it was.
        penalized = self.decode.penalized  # type: ignore[union-attr]
        # The decoder half of the model's config: on a vision-language checkpoint the
        # attention structure sits on the text sub-config, and this is the same
        # resolution transformers' own cache construction performs.
        decoder_config = self._model.config.get_text_config(decoder=True)  # type: ignore[union-attr]
        softcap = _final_logit_softcap(self._model.config)  # type: ignore[union-attr]
        if not eos_ids and any(score_stop for *_, score_stop in prepared):
            raise RuntimeError(
                "neither the checkpoint (tokenizer eos_token_id, generation_config "
                "eos_token_id) nor the contract (protocol stop_token_ids) declares an "
                "end-of-turn id, so a reply's stop cannot be scored; refusing rather "
                "than leaving every early stop unmeasured")
        for messages, tools, continues, continuation, score_stop in prepared:
            if not continuation:
                out.append([])
                continue
            if any(not (0 <= t < vocab) for t in continuation):
                # The served ids come from the contestant (token_ids is mandatory on
                # the wire), so they must be range-checked before touching the model:
                # an id outside the embedding table fires a device-side assert in the
                # embedding gather, which sticky-poisons the CUDA context and turns an
                # unfaithful artifact into a harness error for the rest of the run
                # (and a negative id would silently wrap via Python indexing).
                # Semantically the verdict is exact: a token this model can never
                # emit has probability 0, so its gap is +inf; the surrounding tokens
                # were conditioned on that impossible id, so the whole reply scores
                # +inf, which the verifier fails closed on as malformed evidence.
                out.append([float("inf")] * len(continuation))
                continue
            hidden, _ = self._continuation_hidden(messages, tools, continuation,
                                                  decoder_config, continues)
            if penalized:
                penalty = torch.tensor(
                    self.decode.frequency_penalty,  # type: ignore[union-attr]
                    dtype=torch.float32, device=self._device)
                # before[j, u] is how many times uniq[u] appears in continuation[:j]:
                # the count in force when the server chose position j. Row n_cont, the
                # stop row, counts the whole reply. Restricting the columns to the ids
                # the reply actually served keeps this at (n_cont + 1) x n_cont int64
                # rather than rows x vocab, which would not fit beside the logits.
                ids = torch.tensor(continuation, device=self._device)
                uniq, inv = torch.unique(ids, return_inverse=True)
                onehot = torch.nn.functional.one_hot(inv, uniq.numel()).to(torch.int64)
                before = torch.cat(
                    [torch.zeros(1, uniq.numel(), dtype=torch.int64, device=ids.device),
                     torch.cumsum(onehot, 0)], 0)
            with torch.no_grad():
                gaps: list[float] = []
                for i in range(0, hidden.shape[0], _HEAD_BLOCK):
                    lg = _softcapped(
                        self._model.lm_head(hidden[i:i + _HEAD_BLOCK]).float(),  # type: ignore[union-attr]
                        softcap)
                    if penalized:
                        # One fp32 subtraction of penalty times an integer count,
                        # which is vLLM's arithmetic exactly. SGLang instead keeps an
                        # fp32 running total it adds the penalty into once per
                        # occurrence; the two agree to well under a nat until a single
                        # id repeats thousands of times, and a reply that repetitive
                        # is the pathology the penalty exists to prevent. Columns
                        # outside uniq have count 0 and are left untouched, which is
                        # bitwise what subtracting penalty * 0 gives.
                        rows = slice(i, i + lg.shape[0])
                        lg[:, uniq] = lg[:, uniq] - penalty * before[rows].to(torch.float32)
                    lp = torch.log_softmax(lg, dim=-1)
                    for row in range(lp.shape[0]):
                        j = i + row
                        if j < len(continuation):
                            gaps.append((lp[row].max() - lp[row, continuation[j]]).item())
                        elif score_stop:
                            # The stop is one more position, scored the same way: how much
                            # the reference preferred continuing over ending the turn. It
                            # is 0 when its own argmax IS an end-of-turn id -- the server
                            # was entitled to stop -- and large when the reply was cut
                            # short, so an early stop simply spends the gap ceilings like
                            # any other deviation. Take the best end-of-turn id: on a
                            # multi-EOS checkpoint the alternatives differ by many nats.
                            #
                            # Skipped when the reply already ends in an end-of-turn id: the
                            # turn was ended in band and that token was teacher-forced like
                            # any other, so scoring the following position would charge the
                            # same decision twice. It charges it savagely, too -- after an
                            # end-of-turn token the reference expects anything but another
                            # one, so the gap is enormous. That is one catastrophic flip per
                            # reply for a server that reports its stop token explicitly,
                            # which is what vLLM's chat API does. Stopping early is still
                            # caught: an end-of-turn id where the reply should have
                            # continued is a flip at its own position.
                            if continuation[-1] not in eos_ids:
                                gaps.append(lp[row].max().item()
                                            - max(lp[row, e].item() for e in eos_ids))
            out.append(gaps)
        return out

    async def reference_logprobs(self, items: list[Envelope]) -> list[dict]:
        """The reference's log-probabilities at the ids each item's report names.

        For every item: render its ``metadata["messages"]`` context the way generate
        does, append the continuation (the served ids from the item's
        ``TextPayload.tokens``), run one forward pass, and at continuation position j
        return the reference log-probability of each id in
        ``metadata["report_ids"][j]`` together with the log of the mass outside them.
        The distribution is the model's own: the output head's logits, softcapped where
        the checkpoint declares softcapping, divided by the decode's temperature, and
        log-softmaxed in fp32 -- the same distribution a faithful server sampled from.

        No penalty, and no stop row unless the item asks for one. A sampled decode
        declares no penalty, and the server reported nothing at the position after its
        last token, so there is nothing there to compare against. An item carrying
        ``metadata["stop_row"]`` gets that position's end-of-turn log-probabilities
        back on their own (``p_stop_lp``, ``p_stop_eot_lp``), read off the row
        ``_continuation_hidden`` already computes.

        The reply per item is ``{"p_on_q", "p_tail_lp", "p_eot_lp", "n_ctx", "eot_ids",
        "max_context"}`` (see the ``ReferenceLogprobs`` capability) with floats rounded
        to ``_REPORT_DECIMALS``, or ``{"invalid": reason}`` for an item whose served or
        reported ids are not all inside the vocabulary. The rounding is part of the
        instrument, since a calibration measures the rounded values: six decimals
        of a log-probability is a part in a million of the atom's mass, far below the
        wire rounding the served report itself arrives with.

        Inputs are validated before the model loads, as ``teacher_force`` validates
        its own, so the raise paths run on a torch-free host. A ``Greedy`` decode that
        declares a penalty is refused: the reported distribution would then have to be
        the penalized one on one serving stack and the raw one on another, and there
        is no engine-neutral reading."""
        import asyncio
        from hls.core.decode import Greedy
        if isinstance(self.decode, Greedy) and self.decode.penalized:
            raise NotImplementedError(
                f"reference_logprobs cannot score a report under a decode declaring "
                f"frequency_penalty={self.decode.frequency_penalty}: vLLM reports the "
                f"raw distribution and SGLang the penalized one, so there is no single "
                f"distribution to compare a report against. Use a Sample decode.")
        if not items:
            return []
        temperature = float(getattr(self.decode, "temperature", 1.0))
        prepared: list[tuple[list[dict], list[dict] | None, bool, list[int], list[list[int]],
                             bool]] = []
        for item in items:
            msgs = item.metadata.get("messages")
            if not (isinstance(msgs, list) and msgs):
                raise ValueError(
                    "reference_logprobs item is missing metadata['messages'] (the "
                    "conversation context); contents holds the continuation, so there "
                    "is no fallback.")
            tp = item.first_of("text")
            if not isinstance(tp, TextPayload) or tp.tokens is None:
                raise TypeError(
                    "reference_logprobs item must carry the continuation token ids in "
                    "its text payload (TextPayload.tokens); none present.")
            report_ids = item.metadata.get("report_ids")
            if not isinstance(report_ids, list) or len(report_ids) != len(tp.tokens):
                raise ValueError(
                    f"reference_logprobs item {item.request_id!r} carries "
                    f"{len(report_ids) if isinstance(report_ids, list) else None} rows of "
                    f"metadata['report_ids'] for {len(tp.tokens)} served tokens; the "
                    f"reference scores one row of reported ids per served token.")
            prepared.append((_messages_of(item), tools_of(item), continues_of(item),
                             list(tp.tokens), [list(r) for r in report_ids],
                             bool(item.metadata.get("stop_row"))))
        await asyncio.to_thread(self._load)
        return await asyncio.to_thread(self._reference_logprobs_prepared, prepared,
                                       temperature)

    def _reference_logprobs_prepared(
        self,
        prepared: list[tuple[list[dict], list[dict] | None, bool, list[int], list[list[int]],
                             bool]],
        temperature: float,
    ) -> list[dict]:
        """Blocking per-item report-scoring forward passes (run via to_thread).
        See ``reference_logprobs`` for the returned shape."""
        import torch  # type: ignore[import-untyped]
        out: list[dict] = []
        vocab = self._model.get_input_embeddings().num_embeddings  # type: ignore[union-attr]
        decoder_config = self._model.config.get_text_config(decoder=True)  # type: ignore[union-attr]
        softcap = _final_logit_softcap(self._model.config)  # type: ignore[union-attr]
        # Two checkpoint facts the caller's ending rule needs and cannot learn on a
        # torch-free host. A config that declares no context window reports None, and a
        # caller that cannot tell whether a reply ran out of context fails it closed
        # rather than exempting it.
        eos_ids = self._eos_ids()
        if not eos_ids:
            raise RuntimeError(
                "neither the checkpoint (tokenizer eos_token_id, generation_config "
                "eos_token_id) nor the contract (protocol stop_token_ids) declares an "
                "end-of-turn id, so whether a reply ended its turn in band cannot be "
                "decided; refusing rather than leaving every early stop unmeasured")
        facts = {"eot_ids": sorted(eos_ids),
                 "max_context": getattr(decoder_config, "max_position_embeddings", None)}
        eot_column = sorted(eos_ids)
        for messages, tools, continues, continuation, report_ids, stop_row in prepared:
            if not continuation:
                out.append({"p_on_q": [], "p_tail_lp": [], "p_eot_lp": [],
                            "n_ctx": 0, **facts})
                continue
            bad = _outside_vocabulary(continuation, report_ids, vocab)
            if bad is not None:
                # Range-checked before touching the model for the reason
                # _teacher_force_prepared range-checks its own ids: an id outside the
                # embedding table fires a device-side assert in the embedding gather
                # or in the logit gather, which sticky-poisons the CUDA context for
                # the rest of the run. The reply is closed instead, and the caller
                # fails it as a contract violation.
                out.append({"invalid": bad})
                continue
            hidden, n_ctx = self._continuation_hidden(messages, tools, continuation,
                                                      decoder_config, continues)
            # The row past the end predicts the token after the reply, where the
            # server reported nothing. It is scored only as the stop row, and only
            # when the caller asks for one.
            stop = {}
            if stop_row:
                with torch.no_grad():
                    lg = _softcapped(
                        self._model.lm_head(hidden[len(continuation):]).float(),  # type: ignore[union-attr]
                        softcap)
                    lp = torch.log_softmax(lg / temperature, dim=-1)[0]
                    at_eot = lp[torch.tensor(eot_column, dtype=torch.int64,
                                             device=lp.device)]
                    stop = {"p_stop_lp": [round(v, _REPORT_DECIMALS)
                                          for v in at_eot.tolist()],
                            "p_stop_eot_lp": round(
                                torch.logsumexp(at_eot, dim=0).item(), _REPORT_DECIMALS)}
            hidden = hidden[:len(continuation)]
            ids = torch.tensor(report_ids, dtype=torch.int64, device=hidden.device)
            eot_ids_t = torch.tensor(eot_column, dtype=torch.int64, device=hidden.device)
            # A pad slot carries id -1, which would wrap under a gather; it is clamped
            # to 0 and masked to minus infinity afterwards, so it holds no mass.
            valid = ids >= 0
            safe_ids = ids.clamp_min(0)
            p_on_q: list[list[float]] = []
            p_tail_lp: list[float] = []
            p_eot_lp: list[float] = []
            with torch.no_grad():
                for i in range(0, hidden.shape[0], _HEAD_BLOCK):
                    lg = _softcapped(
                        self._model.lm_head(hidden[i:i + _HEAD_BLOCK]).float(),  # type: ignore[union-attr]
                        softcap)
                    lp = torch.log_softmax(lg / temperature, dim=-1)
                    rows = slice(i, i + lp.shape[0])
                    block_ids, block_valid = safe_ids[rows], valid[rows]
                    gathered = lp.gather(1, block_ids).masked_fill(
                        ~block_valid, float("-inf"))
                    # The mass outside the reported ids, in log space. Deriving it as
                    # log(1 - exp(logsumexp(gathered))) instead cancels to noise on a
                    # peaked row, where the reported ids hold nearly all of it.
                    # scatter_add_ over the valid flags rather than scatter_ of the
                    # flags themselves: pad slots all clamp to column 0, and a plain
                    # scatter would let one of them overwrite a real id 0's flag.
                    reported = torch.zeros_like(lp, dtype=torch.int8).scatter_add_(
                        1, block_ids, block_valid.to(torch.int8)) > 0
                    tail = torch.logsumexp(lp.masked_fill(reported, float("-inf")), dim=1)
                    # What the model gives ending the turn here, over the whole
                    # vocabulary: the end-of-turn ids' mass, summed in log space
                    # because the values this is read at run far below float range.
                    eot = torch.logsumexp(lp[:, eot_ids_t], dim=1)
                    p_on_q.extend(_round_rows(gathered.tolist()))
                    p_tail_lp.extend(round(v, _REPORT_DECIMALS) for v in tail.tolist())
                    p_eot_lp.extend(round(v, _REPORT_DECIMALS) for v in eot.tolist())
            out.append({"p_on_q": p_on_q, "p_tail_lp": p_tail_lp,
                        "p_eot_lp": p_eot_lp, "n_ctx": n_ctx, **stop, **facts})
        return out


class VLLMEngine(ReferenceEngine, _StopSamplingParams, Generate, ClockedServing,
                 TorchStackReport, ReferenceLogprobs):
    """vLLM reference engine. Builds the vLLM engine lazily, in ``_load``, which the
    engine host runs when it loads the engine. Generates under a greedy decode only;
    under a sampled one it scores a served report (``reference_logprobs``) and
    ``generate`` refuses. Renders the same messages as HFEngine via the shared
    _messages_of, so oracle/artifact agree on the conversation.

    Three faces, for three different jobs:

      - ``generate`` (Generate) -- the oracle path: one saturated batch through the
        synchronous ``LLM`` API, which is what scoring a whole rollout wants. Under chat
        conditioning vLLM applies the model's chat template internally in ``chat()``;
        under raw conditioning the prompt goes in as ids, untemplated.
      - ``serve_traffic`` (ClockedServing) -- the calibration path: clocked arrivals
        submitted one at a time to the async engine, so the batching is whatever the
        arrival process put in flight rather than everything at once. The two regimes
        measure different things, and a calibration must measure the one a graded run
        produces.
      - ``reference_logprobs`` (ReferenceLogprobs) -- the report gate's reference under a
        sampled decode at temperature 1: the served continuation teacher-forced through
        vLLM's prompt-logprob path, reduced on the device by ``hls.reference.vllm_report``.
    """

    _serving_package: ClassVar[str] = "vllm"
    #: vLLM runs its engine core in a process of its own, which the engine host watches.
    runs_processes: ClassVar[bool] = True

    def __init__(self, *, model_spec: ModelSpec, decode: DecodeConfig,
                 gpu_kind: str | None = None, conditioning: str = "chat",
                 stop_token_ids: tuple[int, ...] | None = None,
                 memory_mb: int | None = None,
                 gpu_memory_utilization: float = 0.9, max_model_len: int | None = None,
                 dtype: str = "auto", enforce_eager: bool = False,
                 batch_invariant: bool = False, tensor_parallel_size: int = 1,
                 clocked_serving: bool = False) -> None:
        super().__init__(model_spec=model_spec, decode=decode, gpu_kind=gpu_kind,
                         conditioning=conditioning, stop_token_ids=stop_token_ids,
                         memory_mb=memory_mb, gpu_count=tensor_parallel_size)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.batch_invariant = batch_invariant
        #: Which face this engine serves. The engine host runs ``_load`` when it loads
        #: the engine, and the two faces need different vLLM engines (sync ``LLM`` for the
        #: oracle, ``AsyncLLM`` for clocked serving) that cannot coexist on one
        #: card at 0.9 memory utilization -- a load that builds the wrong one
        #: leaves it holding the GPU and the right one fails to construct. Set by
        #: whoever builds the engine (calibration sets True for pool members).
        self.clocked_serving = clocked_serving
        self._llm = None
        #: vLLM's async engine, built at its load when ``clocked_serving`` (or
        #: lazily by _serve_load). Never coexists with _llm.
        self._async_llm = None
        #: Monotonic counter behind the request ids the async engine requires.
        self._serve_requests = 0

    def _apply_env(self) -> None:
        """Set process env vLLM reads at init -- must run before importing vllm /
        building the LLM. VLLM_BATCH_INVARIANT enables vLLM's batch-invariant
        (deterministic across batch composition) kernels, so the oracle's output does
        not depend on how requests were batched (relevant to the bf16 cache-vs-full
        divergence fragility)."""
        import os
        if self.batch_invariant:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"

    def _load(self) -> None:
        # The load builds whichever engine this one will use, so a serve-mode member
        # pays its weight load when the host loads it and serve_traffic finds the
        # engine already up.
        if self.clocked_serving:
            self._serve_build()
            return
        if self._llm is not None:
            return
        self._apply_env()
        from vllm import LLM  # type: ignore[import-untyped]
        # Same weights-source policy as HFEngine (shared helper). No in-engine offline
        # pin: huggingface_hub snapshots HF_HUB_OFFLINE at import time and the vllm
        # import above already pulled it in, so a pin here could never take effect.
        # Offline is enforced by the role's environment: the provision mounts set
        # HF_HUB_OFFLINE before the role starts (hls.provision.mounts).
        src = self._weights_source()
        if self.conditioning == "raw":
            # A raw prompt is fed to LLM.generate as ids, so this engine tokenizes it,
            # and the guard below reuses that same tokenizer rather than loading a
            # second one.
            from transformers import AutoTokenizer  # type: ignore[import-untyped]
            self._tokenizer = AutoTokenizer.from_pretrained(src, local_files_only=True)
            self._check_stop_token_ids(self._tokenizer, src)
        elif self.stop_token_ids is not None:
            # Under chat conditioning LLM.chat renders the template internally, so
            # nothing here needs a stored tokenizer; load one only for the guard, and
            # only when the contract actually pins ids -- otherwise the load must
            # not block on loading one at all.
            from transformers import AutoTokenizer  # type: ignore[import-untyped]
            self._check_stop_token_ids(
                AutoTokenizer.from_pretrained(src, local_files_only=True), src)
        self._llm = LLM(model=src, dtype=self.dtype,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        max_model_len=self.max_model_len,
                        enforce_eager=self.enforce_eager, **self._oracle_args())

    def _oracle_args(self) -> dict:
        """What the synchronous ``LLM`` is built with beyond the shared settings: the
        tensor-parallel width when the slot holds more than one card, and, under a sampled
        decode, what scoring a served report needs. Empty for a greedy oracle on one card,
        which is built as it always was."""
        from hls.core.decode import Sample
        args: dict = {}
        if self.gpu_count != 1:
            # Weights split across the slot's cards. vLLM defaults this to 1, which caps
            # the oracle at whatever one card holds.
            args["tensor_parallel_size"] = self.gpu_count
        if isinstance(self.decode, Sample):
            from hls.reference.vllm_report import WORKER_EXTENSION_CLS
            # Mixed into the worker class so the report capture's calls are reachable by
            # name: vLLM ships no function object to a worker unless insecure
            # serialization is enabled, and naming a method avoids needing that.
            args["worker_extension_cls"] = WORKER_EXTENSION_CLS
            # Off, because a reference must evaluate every position it is asked about.
            # Prefix caching serves a position whose KV is already resident without
            # recomputing it, and a position that is not recomputed produces no logits, so
            # the report capture would return fewer rows than the reply has tokens.
            args["enable_prefix_caching"] = False
        return args

    async def reference_logprobs(self, items: list[Envelope]) -> list[dict]:
        """The reference's log-probabilities at the ids each item's report names.

        The same three reductions ``HFEngine.reference_logprobs`` computes, over the
        same full-vocabulary rows, obtained a different way. HF runs one dense forward
        per item; this engine hands the conversation plus the served continuation to
        vLLM as a prompt and reads the rows vLLM already computes on its prompt-logprob
        path, so the prefill is paged and chunked. That is what makes a context of
        several hundred thousand tokens tractable on a sparse-attention checkpoint,
        where transformers has no flash-attention kernel for a per-query key selection
        and its SDPA fallback materializes that selection densely over queries by keys.

        Which checks read this: the divergence test needs ``p_on_q`` and
        ``p_tail_lp``, and the end-of-turn count needs ``p_eot_lp``. The two exact
        sampling tests read the server's own report and never reach the reference.

        An item may ask for the stop row (``metadata["stop_row"]``): the position after
        its last served token, reduced to its end-of-turn log-probabilities, which an
        implied end of turn is priced on.

        The capture is a wrap on a private vLLM method (``hls.reference.vllm_report``);
        it refuses to install against a build whose runner does not expose that path,
        so a vLLM upgrade fails loudly here rather than quietly returning numbers from
        somewhere else. Inputs are validated before the model loads, so the raise paths
        run on a vLLM-free host.
        """
        import asyncio
        from hls.core.decode import Sample
        if not isinstance(self.decode, Sample):
            raise NotImplementedError(
                "VLLMEngine scores a served report only under a sampled decode; a greedy "
                "reference is graded token by token, not on a report")
        # The worker reduces the raw logits (``hls.reference.vllm_report``), which is the
        # distribution at temperature 1 only; at any other a server that ignored the decode's
        # temperature would be measured against the distribution it served.
        if self.decode.temperature != 1.0:
            raise NotImplementedError(
                f"VLLMEngine scores a served report at temperature 1 only; this decode samples at "
                f"{self.decode.temperature}")
        if not items:
            return []
        prepared = []
        for item in items:
            msgs = item.metadata.get("messages")
            if not (isinstance(msgs, list) and msgs):
                raise ValueError(
                    "reference_logprobs item is missing metadata['messages'] (the "
                    "conversation context); contents holds the continuation, so there "
                    "is no fallback.")
            tp = item.first_of("text")
            if not isinstance(tp, TextPayload) or tp.tokens is None:
                raise TypeError(
                    "reference_logprobs item must carry the continuation token ids in "
                    "its text payload (TextPayload.tokens); none present.")
            report_ids = item.metadata.get("report_ids")
            if not isinstance(report_ids, list) or len(report_ids) != len(tp.tokens):
                raise ValueError(
                    f"reference_logprobs item {item.request_id!r} carries "
                    f"{len(report_ids) if isinstance(report_ids, list) else None} rows of "
                    f"metadata['report_ids'] for {len(tp.tokens)} served tokens; the "
                    f"reference scores one row of reported ids per served token.")
            prepared.append((_messages_of(item), tools_of(item), continues_of(item),
                             list(tp.tokens), [list(r) for r in report_ids],
                             bool(item.metadata.get("stop_row"))))
        await asyncio.to_thread(self._load)
        return await asyncio.to_thread(self._vllm_reference_logprobs, prepared)

    def _ensure_tokenizer(self):
        """The checkpoint's tokenizer, loaded on demand.

        ``_load`` builds one only under raw conditioning or when the contract pins
        stop ids, because ``LLM.chat`` renders the template internally for generation.
        Scoring a report needs the rendered prompt as *ids* -- the served continuation
        is appended to them -- so this path needs one either way."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # type: ignore[import-untyped]
            self._tokenizer = AutoTokenizer.from_pretrained(self._weights_source(), local_files_only=True,
                                                            trust_remote_code=True)
        return self._tokenizer

    def _eos_ids(self) -> set:
        """The ids that end a turn: the contract's pinned set when it declares one,
        else the checkpoint's own.

        Same rule as ``HFEngine._eos_ids``, read a different way. There is no
        transformers model here to take a generation config off, so the config is read
        from the checkpoint directory instead."""
        if self.stop_token_ids is not None:
            return set(self.stop_token_ids)
        return _declared_eos_ids(self._ensure_tokenizer(), self._weights_source())

    def _checkpoint_config(self):
        """The checkpoint's text config, for the two facts a caller's ending rule
        needs and cannot learn on a torch-free host."""
        from transformers import AutoConfig  # type: ignore[import-untyped]
        cfg = AutoConfig.from_pretrained(self._weights_source(), local_files_only=True,
                                         trust_remote_code=True)
        getter = getattr(cfg, "get_text_config", None)
        return getter(decoder=True) if getter is not None else cfg

    def _vocab_size(self) -> int:
        """Vocabulary width, used to refuse an out-of-range id before it reaches the
        embedding gather, where it would fire a device-side assert that poisons the
        CUDA context for the rest of the run."""
        return int(self._checkpoint_config().vocab_size)

    def _declared_max_context(self) -> "int | None":
        """The context window the checkpoint's config declares, or None when it
        declares none. A caller that cannot tell whether a reply ran out of context
        fails it closed rather than exempting it."""
        return getattr(self._checkpoint_config(), "max_position_embeddings", None)

    def _vllm_reference_logprobs(self, prepared: list) -> list[dict]:
        """Blocking report-scoring through vLLM's prompt-logprob path.

        One vLLM request per item: the rendered conversation followed by the served
        continuation, generating a single throwaway token. Asking for prompt logprobs
        is what makes vLLM evaluate the output head at every prompt position; the wrap
        installed by ``vllm_report`` reduces those rows to the gate's three statistics
        on the device, so the vocabulary axis never crosses to the host."""
        eot_column = sorted(self._eos_ids())
        if not eot_column:
            raise RuntimeError(
                "neither the checkpoint nor the contract declares an end-of-turn id, "
                "so whether a reply ended its turn in band cannot be decided; refusing "
                "rather than leaving every early stop unmeasured")
        facts = {"eot_ids": eot_column, "max_context": self._declared_max_context()}
        vocab = self._vocab_size()
        self._ensure_tokenizer()
        # The wrap lives in the worker processes, which is where the logits are. It
        # checks the live runner and raises there if this vLLM reduces prompt
        # log-probabilities somewhere the capture does not wrap.
        self._llm.collective_rpc("hls_install_report")  # type: ignore[union-attr]

        out: list[dict] = []
        for messages, tools, continues, continuation, report_ids, stop_row in prepared:
            if not continuation:
                out.append({"p_on_q": [], "p_tail_lp": [], "p_eot_lp": [],
                            "n_ctx": 0, **facts})
                continue
            bad = _outside_vocabulary(continuation, report_ids, vocab)
            if bad is not None:
                # Range-checked before the model sees the ids: an id outside the
                # embedding table fires a device-side assert that sticky-poisons the
                # CUDA context for the rest of the run.
                out.append({"invalid": bad})
                continue
            prompt_ids = self._context_ids(messages, tools, continues)
            out.append({**self._score_one(prompt_ids, continuation, report_ids,
                                          eot_column, stop_row),
                        "n_ctx": len(prompt_ids), **facts})
        return out

    def _score_one(self, prompt_ids: list[int], continuation: list[int],
                   report_ids: list[list[int]], eot_column: list[int],
                   stop_row: bool = False) -> dict:
        """One item's rows. The request's prompt is context + continuation, so vLLM's
        prompt-logprob path evaluates precisely the positions the server served.

        The capture holds one reply at a time rather than a request id, because
        ``generate`` mints its own id and does not take one. Arming it, submitting a
        single request and taking the rows back is what pairs them; the worker refuses
        a second prompt-logprobs request while armed, so the pairing is checked rather
        than assumed.

        ``stop_row`` also reduces the prompt's last row, which predicts the token
        after the reply. vLLM computes that row in the same prefill and only drops it
        from its own prompt log-probabilities, so no position the reply served
        changes."""
        from vllm import SamplingParams  # type: ignore[import-untyped]

        self._llm.collective_rpc(  # type: ignore[union-attr]
            "hls_arm_report",
            args=(report_ids, eot_column, True) if stop_row else (report_ids, eot_column))
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0,
                            detokenize=False)
        self._llm.generate(  # type: ignore[union-attr]
            {"prompt_token_ids": list(prompt_ids) + list(continuation)},
            sampling_params=sp, use_tqdm=False)
        taken = [r for r in self._llm.collective_rpc("hls_take_report") if r]  # type: ignore[union-attr]
        failed = next((r for r in taken if "error" in r), None)
        if failed is not None:
            raise RuntimeError(
                f"the reference report capture gave up on a reply of "
                f"{len(continuation)} tokens: {failed['error']}")
        got = next(iter(taken), None)
        if got is None:
            raise RuntimeError(
                "no vLLM worker had a report capture armed for a reply of "
                f"{len(continuation)} tokens. The capture is armed and taken on the "
                "same engine that runs the request, so this is a worker that never "
                "received the arming call.")
        rows = len(got["p_tail_lp"])
        if rows == 0:
            raise RuntimeError(
                f"the report capture reduced none of a reply's {len(continuation)} "
                f"positions: it was armed, but no request reached the prompt-logprob "
                f"reduction it wraps. Either the request did not ask for prompt "
                f"logprobs, or this vLLM reduces them somewhere the wrap does not "
                f"cover -- which install() checks, so check its log line names the "
                f"class the live runner uses.")
        if rows != len(continuation):
            raise RuntimeError(
                f"the report capture reduced {rows} of a reply's "
                f"{len(continuation)} positions. vLLM evaluates the output head only "
                f"at prompt positions it prefills, so a position served out of the "
                f"prefix cache produces no row. This engine is built with "
                f"enable_prefix_caching=False so that every position is recomputed; "
                f"check that it still is.")
        if stop_row and "p_stop_lp" not in got:
            raise RuntimeError(
                f"the report capture was asked for the stop row of a reply of "
                f"{len(continuation)} tokens and reduced none: the prompt's last row "
                f"never reached the prompt-logprob reduction it wraps")
        return got

    def _unload(self) -> None:
        """Shut down the engine core process that holds the GPU. Neither ``LLM`` nor its
        ``LLMEngine`` has a shutdown in vLLM 0.22 or 0.24, so the oracle's ``LLM`` is shut
        down through its engine core client, which terminates the process, joins it and
        kills what is left; ``AsyncLLM`` has a shutdown of its own.

        Either shutdown also closes every ZMQ socket vLLM's client opened in this process with
        linger 0 (``BackgroundResources`` in vLLM 0.22), the async client's on the loop that
        uses them, so terminating their context later has no queued message to wait for, as
        SGLang's had (``SGLangEngine._unload``). One exception, the synchronous client's
        shutdown signal, stays queued when its output thread already ended, which that thread
        does when the engine core died: the engine host ends the role on such a death before any
        release, and its release bound ends one that waits there all the same."""
        if self._llm is not None:
            self._llm.llm_engine.engine_core.shutdown()
        if self._async_llm is not None:
            self._async_llm.shutdown()

    def _health(self) -> str | None:
        """What ``AsyncLLM.errored`` says once the async engine is built: its engine core died, or
        the task that hands its outputs on ended. The synchronous ``LLM`` has no such report."""
        if self._async_llm is not None and self._async_llm.errored:
            return "vLLM's async engine reports it errored: its engine core died or its output handler ended"
        return None

    async def generate(self, prompts: list[Envelope]) -> list[Envelope]:
        import asyncio
        import json
        from hls.core.decode import Sample
        # Validate before loading so the raise/empty paths run on a vllm-free host.
        if isinstance(self.decode, Sample):
            raise NotImplementedError(
                "VLLMReference supports only greedy decode; Sample is not implemented.")
        if self.clocked_serving:
            raise RuntimeError(
                "this VLLMEngine was deployed for clocked serving; its GPU holds "
                "the async engine, and building the oracle's synchronous LLM "
                "beside it cannot fit. Deploy a separate engine for generate().")
        if not prompts:
            return []
        # Build the engine off the event loop -- the vLLM LLM construction (weights +
        # CUDA-graph capture) is the heavy, blocking part; the cheap dedup below stays
        # inline and _generate_uniq is offloaded too.
        await asyncio.to_thread(self._load)
        # Dedup identical conversations (mirrors HFEngine's dedup over rendered prompts);
        # key on the stable JSON of the extracted messages. identical keys -> identical
        # msgs, so the one-pass dict (last-wins) is O(n) and order-preserving.
        msgs_per = [_messages_of(p) for p in prompts]
        # Tools are part of the key, not just the messages: the chat template folds
        # them into the prompt, so two turns with the same conversation and different
        # tools are different prompts and must not collapse onto one another.
        tools_per = [tools_of(p) for p in prompts]
        keys = [json.dumps({"messages": m, "tools": t}, sort_keys=True)
                for m, t in zip(msgs_per, tools_per)]
        by_key = dict(zip(keys, zip(msgs_per, tools_per)))
        uniq_keys: list[str] = list(by_key)
        uniq_msgs = [by_key[k][0] for k in uniq_keys]
        uniq_tools = [by_key[k][1] for k in uniq_keys]
        max_new = _max_new_tokens(prompts)
        out_by_key = await asyncio.to_thread(self._generate_uniq, uniq_keys, uniq_msgs,
                                             uniq_tools, max_new)
        return [_reference_reply(p.request_id, *out_by_key[k])
                for p, k in zip(prompts, keys)]

    def _generate_uniq(self, uniq_keys: list[str], uniq_msgs: list[list[dict]],
                       uniq_tools: list[list[dict] | None],
                       max_new: int | None) -> dict[str, tuple[str, str | None, list[int]]]:
        """Blocking vLLM call (run via to_thread); returns key -> (text,
        finish_reason, token_ids) with vLLM's real finish_reason ("stop" on natural
        EOS, "length" on a cap cut) and the exact ids vLLM emitted (so a downstream
        teacher_force scores the true continuation, same contract as HFEngine's
        tokens). Decoding is the module's shared greedy pin, so the oracle and the
        served paths it is compared against sample identically. Under chat conditioning
        LLM.chat applies the model's chat template (add_generation_prompt=True)."""
        from vllm import SamplingParams  # type: ignore[import-untyped]
        # Always pass max_tokens through: SamplingParams defaults max_tokens=16, so
        # omitting it silently truncates. max_tokens=None means unlimited, matching
        # _max_new_tokens's None-means-no-cap contract (HFEngine likewise omits its
        # cap kwarg when None).
        sp = SamplingParams(max_tokens=max_new, **self._stop_sampling(),
                            **_sampling_for(self.decode))
        if self.conditioning == "raw":
            # Raw prompts go in as ids: LLM.chat would wrap them in the chat template,
            # and a raw contract's prompt is already the exact model input. Tools have
            # no template to be folded into and are not rendered.
            results = self._llm.generate(  # type: ignore[union-attr]
                [{"prompt_token_ids": self._context_ids(m)} for m in uniq_msgs],
                sampling_params=sp, use_tqdm=False)
            return {k: (r.outputs[0].text, r.outputs[0].finish_reason,
                        list(r.outputs[0].token_ids))
                    for k, r in zip(uniq_keys, results)}
        # LLM.chat takes one chat_template_kwargs for the whole batch, so conversations
        # are grouped by their tools and each group is one call. The model's pin is the
        # same for every group. Results are reassembled by key, so the caller's
        # per-prompt mapping is unaffected by the grouping.
        import json
        groups: dict[str, tuple[list[dict] | None, list[str], list[list[dict]]]] = {}
        for key, messages, tools in zip(uniq_keys, uniq_msgs, uniq_tools):
            _, keys, msgs = groups.setdefault(
                json.dumps(tools, sort_keys=True), (tools, [], []))
            keys.append(key)
            msgs.append(messages)
        out: dict[str, tuple[str, str | None, list[int]]] = {}
        for tools, keys, msgs in groups.values():
            template_kwargs = dict(self.model_spec.chat_template_kwargs)
            if tools is not None:
                template_kwargs["tools"] = tools
            kwargs: dict = {}
            if template_kwargs:
                kwargs["chat_template_kwargs"] = template_kwargs
            results = self._llm.chat(messages=msgs, sampling_params=sp,  # type: ignore[union-attr]
                                     use_tqdm=False, **kwargs)
            for k, r in zip(keys, results):
                out[k] = (r.outputs[0].text, r.outputs[0].finish_reason,
                          list(r.outputs[0].token_ids))
        return out

    def _serve_build(self) -> None:
        """Build ``vllm.v1.engine.async_llm.AsyncLLM`` from ``AsyncEngineArgs`` with
        the same weights source, dtype and memory settings the synchronous oracle
        uses. Idempotent.

        Safe to call off the event loop (the engine host runs ``_load`` in a worker
        thread): AsyncLLM starts its output-handler task eagerly only when
        constructed inside a running loop, otherwise lazily on the first
        ``add_request`` -- so the handler binds to whichever loop later awaits
        ``generate``, in vllm 0.22 and 0.24. The tokenizer loads here
        too: the async API takes a prompt, not messages, so this engine renders the
        context itself (``_context_ids``) instead of going through ``LLM.chat``."""
        if self._async_llm is not None:
            return
        self._apply_env()
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore[import-untyped]
        from vllm.v1.engine.async_llm import AsyncLLM  # type: ignore[import-untyped]
        src = self._weights_source()
        self._tokenizer = AutoTokenizer.from_pretrained(src, local_files_only=True)
        self._check_stop_token_ids(self._tokenizer, src)
        # max_logprobs caps what a request may ask for, and vLLM's default (20) is
        # below the report width, so a sampled serving would be refused per request
        # rather than at bring-up. Left unset under a decode that reports nothing.
        width = _report_width(self.decode)
        extra = {} if width is None else {"max_logprobs": width}
        if self.gpu_count != 1:
            extra["tensor_parallel_size"] = self.gpu_count
        self._async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=src, dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len, enforce_eager=self.enforce_eager,
            **extra))

    async def _serve_load(self) -> None:
        """Ensure the async engine is up. A serve-mode engine already built it at its
        load; this covers a call on an engine nothing loaded, off the loop so weight
        loading cannot stall it."""
        import asyncio
        if self._async_llm is None:
            await asyncio.to_thread(self._serve_build)

    async def _serve_one(self, prompt: Envelope) -> tuple[
            str, str | None, list[int], list[TokenLogprob] | None, int]:
        """Serve one request on the async engine.

        ``AsyncLLM.generate`` is an async generator keyed by a caller-supplied
        ``request_id`` (unlike the synchronous ``LLM``, it does no id bookkeeping), so
        ids come from a per-engine counter. The default output kind is cumulative, so
        the last output carries the whole reply; ``finished`` marks it.

        Under a sampled decode the request also asks for ``logprobs``, and the reply
        carries the report built from them (``_vllm_positions``)."""
        from vllm import SamplingParams  # type: ignore[import-untyped]
        self._serve_requests += 1
        ids = self._context_ids(_messages_of(prompt), tools_of(prompt))
        width = _report_width(self.decode)
        # Control tokens stay in the text; see _reference_reply. logprobs=None is
        # vLLM's own default for "report nothing".
        params = SamplingParams(max_tokens=_max_new_tokens([prompt]),
                                skip_special_tokens=False, logprobs=width,
                                **self._stop_sampling(), **_sampling_for(self.decode))
        final = None
        async for out in self._async_llm.generate(  # type: ignore[union-attr]
                request_id=f"hls-cal-{self._serve_requests}",
                prompt={"prompt_token_ids": ids}, sampling_params=params):
            final = out
        if final is None or not final.outputs:
            raise RuntimeError(
                f"vLLM returned no output for request {prompt.request_id!r}")
        completion = final.outputs[0]
        report = (None if width is None else
                  _served_report(_vllm_positions(completion), width))
        return (completion.text, completion.finish_reason,
                list(completion.token_ids), report, len(ids))


def _on_loop(loop, fn: Callable[[], None]) -> None:
    """Run *fn* on the thread that runs *loop* and wait for it to return; here when *loop* is
    None, runs nowhere, or runs on this thread. The loop must be free to run it: the engine
    host's stop runs in a worker thread while the loop waits for it."""
    import asyncio
    import concurrent.futures
    try:
        here = asyncio.get_running_loop()
    except RuntimeError:
        here = None
    if loop is None or not loop.is_running() or loop is here:
        fn()
        return
    done: concurrent.futures.Future = concurrent.futures.Future()

    def call() -> None:
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 -- raised in the thread that waits for it
            done.set_exception(e)
        else:
            done.set_result(None)

    loop.call_soon_threadsafe(call)
    done.result()


def _close_sglang_sockets(rpc, manager) -> None:
    """Close the ZMQ sockets an ``sglang.Engine`` holds in this process, each with linger 0,
    and terminate their contexts: *rpc*, the engine's own socket, and *manager*'s, its
    tokenizer manager's (None on a node that runs none).

    ZMQ's default linger waits for ever to deliver what a socket has queued, and terminating a
    context waits for its sockets. SGLang's shutdown kills the scheduler and detokenizer and
    closes the engine's own socket with linger 0, but leaves the tokenizer manager's two open,
    so whatever collected them later terminated their context, which waited for ever on
    messages for the processes the shutdown had killed, and the role hung in ``gc.collect``.
    Closed with linger 0 while the engine still holds
    them, they drop what is queued and leave nothing for a collector to wait on; a socket the
    engine dropped first would be closed by its own finalizer at the default linger.

    A serving's first request registered the tokenizer manager's sockets with the event loop
    it ran on (``TokenizerManager.event_loop``), whose tasks wait on them, and a socket is
    closed on the thread that uses it: so there the tasks are cancelled first, since SGLang
    ends the whole process when one of them finds its socket closed (its
    ``print_exception_wrapper``), and then the sockets are closed. With no such loop they are
    closed here."""
    if rpc is not None:
        rpc.context.destroy(linger=0)
    if manager is None:
        return
    contexts = {manager.recv_from_detokenizer.context, manager.send_to_scheduler.context}

    def close() -> None:
        for task in list(manager.asyncio_tasks):
            task.cancel()
        for context in contexts:
            context.destroy(linger=0)

    _on_loop(manager.event_loop, close)


class SGLangEngine(ReferenceEngine, _StopSamplingParams, Generate, ClockedServing,
                   TorchStackReport):
    """SGLang reference engine -- a Generate-only oracle (mirrors VLLMEngine); a
    trusted-engine pool member for ``hls calibrate``, never a
    task-facing reference -- it provides no TeacherForce/StackReport, so
    check_wiring rejects it for TokenFaithfulnessVerifier. Builds sglang's offline
    ``Engine`` lazily in _load, which the engine host runs when it loads the engine.
    Greedy only: Sample decode raises, mirroring HFEngine/VLLMEngine.

    Context rendering goes through our own tokenizer (``_render_context``, the same
    helper HFEngine uses), not sglang's server-side chat default -- determinism over
    convenience, so all pool engines render one conversation identically."""

    _serving_package: ClassVar[str] = "sglang"
    #: SGLang runs its scheduler and detokenizer in processes of their own, which the engine host
    #: watches.
    runs_processes: ClassVar[bool] = True

    def __init__(self, *, model_spec: ModelSpec, decode: DecodeConfig,
                 gpu_kind: str | None = None, conditioning: str = "chat",
                 stop_token_ids: tuple[int, ...] | None = None,
                 memory_mb: int | None = None,
                 mem_fraction_static: float = 0.85,
                 max_model_len: int | None = None,
                 swa_full_tokens_ratio: float | None = None) -> None:
        super().__init__(model_spec=model_spec, decode=decode, gpu_kind=gpu_kind,
                         conditioning=conditioning, stop_token_ids=stop_token_ids,
                         memory_mb=memory_mb)
        self.mem_fraction_static = mem_fraction_static
        self.max_model_len = max_model_len
        #: For a model that mixes sliding-window and full-attention layers, the share
        #: of the KV budget sglang reserves for the sliding-window layers; None keeps
        #: sglang's default. The rest is the full-attention pool, which sets the longest
        #: prompt the engine admits. Capacity only, never numerics.
        self.swa_full_tokens_ratio = swa_full_tokens_ratio
        self._engine = None

    def _load(self) -> None:
        if self._engine is not None:
            return
        # sglang is an optional dependency (only needed by this pool member, not the
        # task-facing HFEngine); a clean, actionable error beats a bare
        # ModuleNotFoundError deep in a calibrate run.
        try:
            import sglang  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "SGLangEngine requires the sglang package, which is not installed; "
                "install it with `uv pip install sglang`") from exc
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        # Same weights-source policy as HFEngine/VLLMEngine (shared helper).
        src = self._weights_source()
        self._tokenizer = AutoTokenizer.from_pretrained(src, local_files_only=True)
        self._check_stop_token_ids(self._tokenizer, src)
        kwargs: dict = {"model_path": src, "mem_fraction_static": self.mem_fraction_static}
        if self.max_model_len is not None:
            kwargs["context_length"] = self.max_model_len
        if self.swa_full_tokens_ratio is not None:
            kwargs["swa_full_tokens_ratio"] = self.swa_full_tokens_ratio
        self._engine = sglang.Engine(**kwargs)

    def _unload(self) -> None:
        """Shut down sglang's scheduler process, which holds the GPU, close the ZMQ sockets
        the engine holds in this process, and unhook it from interpreter exit.

        ``sglang.Engine`` registers its shutdown to run at exit, which keeps it alive
        after this engine drops it, so the scheduler would hold the GPU until the process
        ends. The hook stays when the shutdown raises, so exit tries again; the sockets are
        closed either way (``_close_sglang_sockets``), before anything collects them.

        sglang's shutdown kills every child process of this process, not only its own
        (``kill_process_tree(os.getpid(), include_parent=False)`` in 0.5.16 and 0.5.20),
        so the process that runs this engine must keep no other child it still needs,
        another engine's process among them. The verify role starts no process of its
        own and has one engine live at a time."""
        if self._engine is None:
            return
        engine = self._engine
        # Read first: the shutdown closes the engine's own socket and drops it, which leaves
        # its context to whatever collects it.
        rpc = engine.send_to_rpc
        try:
            engine.shutdown()
        finally:
            _close_sglang_sockets(rpc, engine.tokenizer_manager)
        atexit.unregister(engine.shutdown)

    async def generate(self, prompts: list[Envelope]) -> list[Envelope]:
        import asyncio
        from hls.core.decode import Sample
        # Validate before loading so the raise/empty paths run on an sglang-free host.
        if isinstance(self.decode, Sample):
            raise NotImplementedError(
                "SGLangReference supports only greedy decode; Sample is not implemented.")
        if not prompts:
            return []
        await asyncio.to_thread(self._load)
        # Render each prompt through our own tokenizer (see class docstring) --
        # mirrors HFEngine exactly; the turn's tools ride on the render because the
        # chat template folds them into the prompt. Dedup over the rendered text.
        rendered_per_prompt = [self._render_context(_messages_of(p), tools_of(p))
                               for p in prompts]
        uniq = list(dict.fromkeys(rendered_per_prompt))
        max_new = _max_new_tokens(prompts)
        out_by_key = await asyncio.to_thread(self._generate_uniq, uniq, max_new)
        return [_reference_reply(p.request_id, *out_by_key[rp])
                for p, rp in zip(prompts, rendered_per_prompt)]

    def _generate_uniq(self, uniq: list[str],
                       max_new: int | None) -> dict[str, tuple[str, str | None, list[int]]]:
        """Blocking sglang call (run via to_thread); returns rendered-prompt -> (text,
        finish_reason, token_ids). Greedy = temperature 0, max_new_tokens stamped
        explicitly (None means unlimited, the same _max_new_tokens no-cap contract
        as HFEngine/VLLMEngine).

        Prompts are fed as our tokenizer's ids (``add_special_tokens=False``, the
        same convention HFEngine tokenizes rendered text with), never as text:
        handing sglang the rendered string makes its server re-tokenize it and
        prepend a second BOS -- the rendered chat template already carries one --
        so the whole conversation shifts by a token (see ``chat_prompt_ids``).

        ``Engine.generate`` has no plain token-ids output field (sglang issue
        #2634); ``return_logprob=True`` is the documented way to recover them --
        ``meta_info['output_token_logprobs']`` is a list of (logprob, token_id,
        token_text) per emitted token, in order."""
        sampling_params = {"max_new_tokens": max_new, **self._stop_sampling(),
                           **_sampling_for(self.decode)}
        input_ids = [self._tokenizer(p, add_special_tokens=False).input_ids  # type: ignore[union-attr]
                     for p in uniq]
        outs = self._engine.generate(  # type: ignore[union-attr]
            input_ids=input_ids, sampling_params=sampling_params, return_logprob=True)
        result: dict[str, tuple[str, str | None, list[int]]] = {}
        for prompt, out in zip(uniq, outs):
            meta = out["meta_info"]
            tokens = [tid for _, tid, _ in meta["output_token_logprobs"]]
            # Direct indexing on the documented reply shape (sglang 0.5.16's): if a
            # future sglang changes meta_info's schema, this must raise
            # loudly here, not silently degrade the token stream to None.
            finish = meta["finish_reason"]["type"]
            result[prompt] = (out["text"], finish, tokens)
        return result

    async def _serve_load(self) -> None:
        """Bring sglang's engine up for the clocked-serving path.

        The same ``sglang.Engine`` the batch path builds: it exposes both the blocking
        ``generate`` and the coroutine ``async_generate``, so there is no second engine
        to construct. Built in a worker thread because the constructor blocks on weight
        loading and this runs on the container's event loop."""
        import asyncio
        await asyncio.to_thread(self._load)

    async def _serve_one(self, prompt: Envelope) -> tuple[
            str, str | None, list[int], list[TokenLogprob] | None, int]:
        """Serve one request on ``sglang.Engine.async_generate``.

        Fed as token ids for the reason ``chat_prompt_ids`` documents, and with
        ``return_logprob=True`` because that is the only way to recover the emitted
        ids: ``Engine`` has no plain token-ids output field (sglang issue #2634), and
        ``meta_info['output_token_logprobs']`` carries (logprob, token_id, token_text)
        per emitted token, in order. Direct indexing on the documented reply shape
        (sglang 0.5.16's): a future schema change must raise here, not
        silently degrade the token stream.

        Under a sampled decode the request also asks for each position's alternatives,
        and the reply carries the report built from them (``_sglang_positions``).
        ``top_logprobs_num`` is a parameter of ``async_generate`` itself, beside
        ``return_logprob``, and not one of the sampling params: sglang's
        ``SamplingParams`` refuses it as an unexpected keyword, which would fault every
        request of the serving. Asking for the alternatives is real extra work for the
        member -- a top-k per position and 21 detokenizations on top of the served ids
        ``return_logprob`` already yields -- so it is requested only when a report is
        what the decode is graded on."""
        ids = self._context_ids(_messages_of(prompt), tools_of(prompt))
        width = _report_width(self.decode)
        # Control tokens stay in the text; see _reference_reply.
        params = {"max_new_tokens": _max_new_tokens([prompt]),
                  "skip_special_tokens": False, **self._stop_sampling(),
                  **_sampling_for(self.decode)}
        report_kwargs = {} if width is None else {"top_logprobs_num": width}
        out = await self._engine.async_generate(  # type: ignore[union-attr]
            input_ids=ids, sampling_params=params, return_logprob=True,
            **report_kwargs)
        meta = out["meta_info"]
        tokens = [tid for _, tid, _ in meta["output_token_logprobs"]]
        report = (None if width is None else
                  _served_report(_sglang_positions(meta), width))
        return out["text"], meta["finish_reason"]["type"], tokens, report, len(ids)
