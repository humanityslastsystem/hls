"""L1 -- wire protocol: the Envelope <-> wire-JSON codec a Contract owns.

A WireProtocol encodes requests, decodes responses (unary + streaming), and
contributes example bodies and prose to the build brief. OpenAIChatProtocol is
the default: standard OpenAI chat/completions plus a dedicated
`choices[].token_ids` array of integer vocabulary ids. OpenAICompletionProtocol
is the raw-prompt sibling: standard OpenAI `/v1/completions` plus the same
`token_ids` extension, for tasks whose prompt is the exact model input rather
than a chat turn.
"""

from __future__ import annotations
import json
from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import jinja2
from pydantic import Field, TypeAdapter, ValidationError, field_validator, model_validator

from hls.core.faults import ArtifactProtocolError, HarnessFault
from hls.core.payloads import (MAX_TOKEN_ID, MIN_TOKEN_ID, AudioPayload, CodecPosition,
                               Envelope, ImagePayload, TextPayload, TokenLogprob,
                               TopLogprob)
from hls.core.poly import Poly
from hls.core.trace import ArtifactResponse, PartialOutput
from hls.spec.drivers import message_content, messages_of


class WireProtocol(Poly):
    """Owns the wire codec + brief contributions (L1). Frozen config object."""

    # Whether this protocol emits exact integer token ids on every response.
    # check_wiring requires it for TeacherForce verifiers and per-token streaming
    # SLOs. Class-level, not a field: a protocol opts in by actually emitting them.
    emits_token_ids: ClassVar[bool] = False
    #: How the prompt reaches the model. "chat": the server renders messages through
    #: the checkpoint's chat template. "raw": the prompt string is the model input,
    #: verbatim. Every reference engine built from the task reads this off the
    #: contract, so the oracle reproduces what the server saw.
    conditioning: ClassVar[Literal["chat", "raw"]] = "chat"
    #: Ids that end a turn. Sent on the wire as ``stop_token_ids`` and read by every
    #: engine built from the task as its end-of-turn set (stop row, finish_reason,
    #: emitted-token trimming). None means the checkpoint's own eos set, which is right
    #: for chat. Declared ids replace that set on the reference, and must include the
    #: checkpoint's own eos ids: vLLM and SGLang can only extend that set, so a pinned
    #: set that omits one would have a server and the reference stopping in different
    #: places. The engines refuse such a set when they load the checkpoint.
    stop_token_ids: tuple[int, ...] | None = None

    @field_validator("stop_token_ids", mode="before")
    @classmethod
    def _reject_bool_stop_token_ids(cls, value: object) -> object:
        """Refuse a bool among the ids, before pydantic coerces it.

        ``bool`` is a subclass of ``int``, so ``(True,)`` would validate and be stored
        as token id 1 -- a mistyped flag silently becomes a terminator.

        Materialises any iterable first rather than checking lists and tuples only: a
        set is an idiomatic spelling for a set of ids and would otherwise walk past the
        guard, and a generator inspected in place would reach pydantic already drained.
        A string is left alone so pydantic reports the type, not its characters."""
        if value is None or isinstance(value, (str, bytes)):
            return value
        try:
            ids = list(value)  # type: ignore[call-overload]
        except TypeError:
            return value        # not iterable; pydantic says what it is
        if any(isinstance(v, bool) for v in ids):
            raise ValueError(
                f"stop_token_ids must be integer token ids; got a bool in {ids}")
        return ids

    @model_validator(mode="after")
    def _validate_stop_token_ids(self) -> "WireProtocol":
        """Refuse a pinned set that cannot mean what it says. An empty tuple declares a
        set while leaving the engines no end-of-turn id; a negative id can never match an
        emitted token, so the reference would simply never terminate on it; a repeated id
        is dumped verbatim onto the wire. Ids past the vocabulary need the checkpoint and
        are refused where it is loaded."""
        ids = self.stop_token_ids
        if ids is None:
            return self
        if not ids:
            raise ValueError("stop_token_ids must be None or a non-empty tuple")
        if any(i < 0 for i in ids):
            raise ValueError(
                f"stop_token_ids must be non-negative vocabulary ids, got {list(ids)}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"stop_token_ids must not repeat an id, got {list(ids)}")
        return self

    @abstractmethod
    def encode_request(self, env: Envelope, *, stream: bool = False) -> dict: ...

    @abstractmethod
    def decode_response(self, data: dict, request_id: str) -> Envelope:
        """Decode a unary response, raising ArtifactProtocolError (not a raw
        KeyError/IndexError/ValidationError) on a non-conformant body."""
        ...

    def decode_stream(self, timed_chunks: list[tuple[float, dict]],
                      request_id: str) -> ArtifactResponse:
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    def decode_partial(self, timed_chunks: list[tuple[float, dict]]) -> PartialOutput | None:
        """What an incomplete stream managed to produce, for a failed event's
        diagnostic record. ``None`` when nothing decodable arrived.

        Unlike ``decode_stream`` this never raises and never validates: it runs while a
        failure is already being recorded, so raising would replace the fault under
        diagnosis with one about the diagnosis. Defaults to ``None`` -- a protocol that
        does not stream produces nothing partial."""
        return None

    def stream_packet_bytes(self, frame_bytes: Callable[[Mapping[str, Any]], int]) -> int | None:
        """The fewest reply bytes one packet timestamp of this protocol's stream costs, as
        ``decode_stream`` times packets, a stream frame carrying a chunk taking *frame_bytes* of
        that chunk as the client reads a stream. None when the protocol does not say, as one
        that does not stream cannot."""
        return None

    def trace_growth(self, *, stream: bool) -> float:
        """The most bytes of its event's JSON in the trace's chunks one byte of a reply under this
        protocol can become (``hls.trace.store``; its summary line is never longer). The drive
        role counts a reply it has received and not yet written at this
        (``hls.drive.replies``). Of a whole reply (*stream* False) it is
        ``hls.trace.store.ESCAPE_GROWTH``: what the trace keeps of it is values the reply
        carried, written again, and no packet of it is timed. Raise ValueError for a stream the
        protocol does not say it of."""
        from hls.trace.store import ESCAPE_GROWTH
        if stream:
            raise ValueError(f"{type(self).__name__} does not say how many trace bytes a byte of its stream "
                             "becomes (WireProtocol.trace_growth)")
        return float(ESCAPE_GROWTH)

    @property
    def emits_codec_codes(self) -> bool:
        """Whether replies carry the audio codec codes a frame-level gate reads out of
        ``AudioPayload.codec_codes``.

        ``check_wiring`` requires it of a verifier that declares
        ``requires_codec_codes``, the audio counterpart of the ``emits_token_ids``
        check on a TeacherForce verifier. A property rather than a ClassVar because
        asking for codes is a per-instance choice: the same protocol class serves a
        task whose gate reads the codes and one whose gate does not, and a server
        should not be told to compute what nothing reads."""
        return False

    def stream_token_count(self, chunk: dict) -> int:
        """How many output tokens one stream frame carries.

        Read by the transport while the stream is still arriving, to tell progress
        from a warm connection: the stall rule measures the gap between tokens, not
        between frames, so a server emitting empty frames through a stuck decode does
        not re-arm it. Never raises -- it runs inside the read loop, where a malformed
        frame is the fold's to reject, with the whole frame in hand. Defaults to zero,
        which is right for a protocol that does not stream and means a streaming
        protocol has to answer for itself."""
        return 0

    @abstractmethod
    def body_schema(self) -> dict: ...

    @abstractmethod
    def brief_section(self, *, path: str, streaming: bool = False) -> str: ...

    def probe_request(self) -> dict:
        """Minimal request body used as a second-stage readiness probe after
        ``/health`` returns 200. Defaults to the OpenAI-chat shape with
        ``max_tokens=1``; override so the probe matches non-OpenAI wire shapes."""
        return {
            "model": "probe",
            "messages": [{"role": "user", "content": "_"}],
            "max_tokens": 1,
        }


#: The serving-configuration keys a response's optional ``hls_context`` object may
#: carry into envelope metadata. A fixed list, not a passthrough of the whole object:
#: metadata feeds report-only serving metrics, and an artifact must not be able to
#: introduce arbitrary keys there.
_HLS_CONTEXT_METADATA_KEYS = (
    "max_context_len",
    "request_max_tokens",
    "enable_prefix_caching",
    "enforce_eager",
    "max_num_seqs",
    "tensor_parallel_size",
)


#: The JSON values that carry no nesting. Of the optional objects a reply sends beside
#: its tokens, only these values reach envelope metadata: the trace write refuses a value
#: nested 254 deep, so passing anything else through would let one reply end the run when
#: trace.json is written.
_SCALARS = (str, int, float, bool, type(None))

#: The token counts a reply's ``usage`` object may carry into envelope metadata.
_USAGE_KEYS = ("completion_tokens", "prompt_tokens", "total_tokens")


def _hls_context_metadata(ctx: dict) -> dict:
    return {key: ctx[key] for key in _HLS_CONTEXT_METADATA_KEYS
            if key in ctx and isinstance(ctx[key], _SCALARS)}


def _scalar_entries(obj: dict) -> dict:
    """``obj``'s entries whose values are JSON scalars (``_SCALARS``)."""
    return {k: v for k, v in obj.items() if isinstance(v, _SCALARS)}


def _usage_counts(usage: object) -> dict[str, int]:
    """The integer token counts in a reply's ``usage`` object, in ``_USAGE_KEYS`` order.

    A count that is not an integer is dropped, and so is a ``usage`` that is not an
    object: a list or a string naming a count passes a membership test and then raises
    TypeError when indexed by the key."""
    if not isinstance(usage, dict):
        return {}
    return {k: usage[k] for k in _USAGE_KEYS
            if isinstance(usage.get(k), int) and not isinstance(usage[k], bool)}


#: A frame's ``token_ids``, validated the way ``TextPayload.tokens`` validates the whole
#: reply's (so ``"5"`` and ``5.0`` read as 5), and early enough
#: that the stream fold only ever holds integers.
_TOKEN_ID_LIST = TypeAdapter(list[int])


def _check_token_ids(ids: object, where: str) -> None:
    """Refuse a reply whose ``ids`` include one outside ``MIN_TOKEN_ID`` to
    ``MAX_TOKEN_ID``.

    Only that bound. Whether the ids are the tokens the server emitted is the gate's
    question, and a value that is not a list of integers is the payload model's."""
    if not isinstance(ids, list):
        return
    for tid in ids:
        if isinstance(tid, int):
            _check_token_id(tid, where)


def _check_token_id(tid: int, where: str) -> None:
    """Refuse one id outside the int64 range (``MIN_TOKEN_ID`` to ``MAX_TOKEN_ID``)."""
    if not MIN_TOKEN_ID <= tid <= MAX_TOKEN_ID:
        raise ArtifactProtocolError(f"{where} carries a token id outside 64 bits: "
                                    f"{tid!r:.40}")


_OPENAI_BRIEF = """\
### POST {{ path }}
Accept OpenAI chat/completions requests and return chat completion responses.
The harness sends `max_tokens` on every request body; treat it as a binding
per-request output cap. Do not substitute a server-side default when the
request supplies `max_tokens`.

Return a `token_ids` array on each choice (`choices[].token_ids`): the integer
vocabulary IDs of the exact tokens you emitted (NOT a re-tokenization of the
text). This is mandatory on every response.
{% if sends_tools %}

The same faithfulness applies to the text: `message.content` must be the exact text
of the tokens in `token_ids`, with nothing removed. Do NOT run a reasoning parser or
a tool-call parser -- both move text out of `content` into a separate field, and the
harness reads `content`. Every request sets `tool_choice: "none"`:
render the tool definitions through the chat template and leave the model's tool call
in `content`, in the checkpoint's own syntax. A reply whose `content`
is missing text that its own `token_ids` carry fails the run.
{% endif %}{% if stop_token_ids %}
Every request carries `stop_token_ids` {{ stop_token_ids }}: generation ends when the
model emits any of them rather than on the checkpoint's own end-of-turn token.
`token_ids` must include that end-of-turn token when the reply ended on one, even
though it is not part of the returned message content. A reply whose first generated
token is one of them is one token, not zero. A response with an
empty `token_ids` array is unverifiable and fails the run.
{% endif %}{% if streaming %}
Support streaming: when the request sets `"stream": true`, respond with an OpenAI
SSE stream (`data: {chunk}` lines, `chat.completion.chunk` objects with `delta`,
terminated by `data: [DONE]`). Include the per-chunk `token_ids` array for that
chunk's tokens.
{% endif %}{% if return_logprobs %}
Return logprobs per the OpenAI spec (`choices[].logprobs.content[]`): one entry per
token you generated, in the order you generated them, carrying that token's own
`logprob`{% if top_logprobs %} and a `top_logprobs` array holding the top {{ top_logprobs }} alternatives at that
position -- the {{ top_logprobs }} most probable tokens, whether or not you emitted one of them{% endif %}.
In addition to OpenAI's decoded `token` string, every logprob entry{% if top_logprobs %} and every alternative{% endif %} MUST carry an integer `token_id`
field (the vocabulary ID of that token); the verifier keys on these IDs.
A reply that omits the logprobs, reports fewer entries than it emitted tokens{% if top_logprobs %},
reports fewer than {{ top_logprobs }} alternatives at a position{% endif %} or omits a `token_id` fails verification.
{% endif %}
Request:
```json
{{ request }}
```
Response{% if top_logprobs_shown %} (each `top_logprobs` array is abbreviated to {{ top_logprobs_shown }} entries below; a conforming reply carries {{ top_logprobs }} at every position){% endif %}:
```json
{{ response }}
```
"""


@dataclass
class _StreamFold:
    """One SSE stream's frames, folded so far.

    The state of ``decode_stream``, held in an object so ``decode_partial`` can read
    it after the fold raised. Nothing here is validated;
    the two callers decide what an incomplete fold means."""
    #: Content deltas in arrival order; joined into the reply's text.
    text_parts: list[str] = field(default_factory=list)
    #: Emitted token ids, concatenated across frames.
    tokens: list[int] = field(default_factory=list)
    #: Per-token logprob entries, when the contract requested them.
    logprobs: list[TokenLogprob] = field(default_factory=list)
    #: One arrival timestamp per emitted token, for the streaming SLOs.
    packet_timestamps: list[float] = field(default_factory=list)
    #: The last finish_reason any frame reported. ``None`` means the stream never
    #: declared itself finished.
    finish_reason: str | None = None
    #: The message from an ``{"error": ...}`` frame -- a server's own account of why it
    #: stopped, on an already-committed 200.
    server_error: object | None = None
    #: The server's reported token counts (``_usage_counts``): of each kind, the last
    #: integer value any usage frame carried.
    usage: dict[str, int] = field(default_factory=dict)
    #: The last ``hls_context`` object any frame carried (serving configuration).
    hls_context: dict | None = None
    #: The last ``gpu`` snapshot any frame carried (utilization, memory), its scalar
    #: entries only.
    gpu: dict | None = None
    #: Arrival time of the first token-bearing frame (the streaming SLOs' TTFT).
    t_first_packet: float | None = None


class _OpenAIProtocol(WireProtocol):
    """The codec both OpenAI-shaped protocols share: one request envelope, one
    ``choices[0]`` reply with a mandatory ``token_ids`` array, one SSE fold. Subclasses
    say where the text sits in a choice (``message.content`` for chat, ``text`` for
    completions) and whether logprobs are parsed."""

    # token_ids is mandatory on every response and checked in decode_response/
    # decode_stream -- the one deviation from stock OpenAI.
    emits_token_ids: ClassVar[bool] = True

    #: Labels whose plain meaning is that the reply did not finish. A server reporting
    #: one has failed that request whatever status it committed to first, so the
    #: no-failed-request rule must see it.
    #:
    #: Stated as the reasons we reject rather than the reasons we accept. The brief
    #: publishes no enumeration of permitted values, so zeroing a run on an unlisted
    #: but complete label -- ``content_filter``, ``tool_calls``, a stack's own spelling
    #: of a normal stop -- would enforce a rule no contestant was given. These four
    #: admit no such reading.
    #:
    #: Checked on both decode paths, because a streaming server commits its 200 and
    #: opens the SSE stream before it knows how the reply ends: an artifact that
    #: crashes mid-generation can still emit a well-formed terminal frame and a proper
    #: ``[DONE]``, and every downstream count reads that as served.
    #: Every spelling is listed rather than derived: trimming a trailing ``ed`` turns
    #: ``canceled`` into ``cance``, and ``timed_out`` shares no stem with ``timeout``.
    INCOMPLETE_FINISH_REASONS: ClassVar[frozenset[str]] = frozenset({
        "error", "errored", "failed", "failure",
        "abort", "aborted",
        "cancel", "canceled", "cancelled",
        "timeout", "timed_out",
    })

    model: str = "served-model"

    @classmethod
    def _check_finish_reason(cls, finish_reason: object) -> str:
        """Fault when the reply's own label says it never finished.

        Anything else passes, including labels this harness does not know: an
        unrecognised reason is a reporting question, while ``error`` and its siblings
        are a served-request question. Compared case-insensitively with ``-`` folded to
        ``_``, so ``timed-out`` and ``timed_out`` are the same label."""
        if not isinstance(finish_reason, str):
            raise ArtifactProtocolError(
                f"finish_reason is not a string: {type(finish_reason).__name__}")
        normalized = finish_reason.strip().lower().replace("-", "_")
        if normalized in cls.INCOMPLETE_FINISH_REASONS:
            raise ArtifactProtocolError(
                f"reply did not complete: finish_reason={finish_reason!r}")
        return finish_reason

    def _refuse_zero_token_reply(self, where: str) -> None:
        """Fault on a reply that emitted no tokens -- but only where the brief said so.

        Both brief templates carry "a response with an empty token_ids array is
        unverifiable and fails the run" inside their pinned-stop-ids block, next to the
        sentence that a reply ending on its first token is one token and not zero. A
        contract that pins no ids publishes neither, so its contestant is told only that
        token_ids is mandatory: it is never told that the end-of-turn token belongs in
        the array, and a server that strips it can return an empty one in good faith.
        Faulting there would zero a run on an unpublished rule, which is what
        INCOMPLETE_FINISH_REASONS avoids for finish_reason labels.

        Where no ids are pinned the reply decodes instead. The verifier then grades an
        empty continuation, or refuses the event outright when the stream carried no ids
        at all."""
        if self.stop_token_ids is None:
            return
        raise ArtifactProtocolError(
            f"{where}; a reply with no emitted tokens is unverifiable, and a reply that "
            f"ended on one of the pinned stop ids {list(self.stop_token_ids)} carries it")

    @abstractmethod
    def _unary_text(self, choice: dict) -> str | None:
        """The reply text in one unary/streaming-final choice object."""
        ...

    @abstractmethod
    def _stream_text(self, choice: dict) -> str | None:
        """The text delta in one streamed choice object."""
        ...

    def _parse_logprobs(self, lp: dict | None) -> list[TokenLogprob] | None:
        """Per-token logprobs from a choice's logprobs field. None when the
        protocol doesn't request them."""
        return None

    # --- request ---
    @staticmethod
    def _conversation_of(env: Envelope) -> list[dict]:
        """The messages one request carries, as ``messages_of`` selects them, with its
        TypeError retyped.

        An envelope holding neither a message list nor a text payload is a request this
        harness built wrong. That invalidates the run (HarnessFault), and is deliberately
        not an ArtifactFault: encoding happens before anything is sent, so charging it to
        the artifact would record a failed request against a server that never saw one --
        and under the no-failed-request rule, zero its score."""
        try:
            return messages_of(env)
        except TypeError as e:
            raise HarnessFault(
                f"request {env.request_id!r} carries no conversation to encode: {e}") from e

    def _flags(self) -> dict:
        # Asks for token_ids whenever this protocol requires them: stock vLLM only
        # populates choices[].token_ids when the request sets this field
        # (--return-tokens-as-token-ids is a different feature, reformatting logprob
        # strings), so omitting it would fault every stock-vLLM response. Custom
        # servers tolerate the extra field. A subclass that does not require token
        # ids does not ask for them either, so its brief and its decode agree.
        body: dict = {"return_token_ids": True} if self.emits_token_ids else {}
        if self.stop_token_ids is not None:
            body["stop_token_ids"] = list(self.stop_token_ids)
        return body

    @staticmethod
    def _stamp_continuation(body: dict, env: Envelope) -> None:
        """Ask the server to finish the last assistant message rather than start one.

        Per request, off the envelope, for the same reason ``max_tokens`` rides there:
        it is a property of this turn and not of the contract. A field on the protocol
        would make every request a continuation, and the model would never begin a
        reply at all.

        Both fields or neither. ``add_generation_prompt: false`` is not decoration
        beside the first: left out, the server appends a fresh generation prompt after
        the assistant message it was handed, and the model starts a new reply instead
        of finishing the one it was given. vLLM and SGLang both read this spelling at
        the top level of the body.

        What a caller uses it for: a harness that cut a reply at a cap can hand the
        text back and have the model carry on from exactly where it stopped, which is
        how a thinking block that never closed is brought to an end without throwing
        away the turn.
        """
        if env.metadata.get("continue_final_message"):
            body["continue_final_message"] = True
            body["add_generation_prompt"] = False

    @staticmethod
    def _stamp_max_tokens(body: dict, env: Envelope) -> None:
        """Copy the envelope's output cap onto the request body, in place.

        ``metadata["max_tokens"]`` is the workload's cap, stamped by the driver so
        artifact and oracle decode under the same budget; reference comparisons need
        matching budgets to be meaningful. Both OpenAI shapes spell the field the same
        way, so they stamp it the same way."""
        max_tokens = env.metadata.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            body["max_tokens"] = max_tokens

    # --- response (unary) ---
    def decode_response(self, data: dict, request_id: str) -> Envelope:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise ArtifactProtocolError(f"missing choices[0]: {e!r}") from e
        if not isinstance(choice, dict):
            raise ArtifactProtocolError(
                f"choices[0] is not an object: {type(choice).__name__}")
        finish_reason = choice.get("finish_reason")
        if finish_reason is None:
            raise ArtifactProtocolError("missing finish_reason (incomplete completion)")
        finish_reason = self._check_finish_reason(finish_reason)
        if self.emits_token_ids:
            if "token_ids" not in choice:
                raise ArtifactProtocolError("missing choices[0].token_ids (mandatory)")
            if not choice["token_ids"]:
                self._refuse_zero_token_reply("empty choices[0].token_ids")
        try:
            payload = TextPayload(
                text=self._unary_text(choice) or "",
                tokens=choice.get("token_ids"),
                logprobs=self._parse_logprobs(choice.get("logprobs")),
                finish_reason=finish_reason)
        except (ValidationError, KeyError, TypeError, AttributeError) as e:
            # AttributeError: a `message`, `logprobs` or logprob entry that is not an
            # object is read with `.get`.
            raise ArtifactProtocolError(f"malformed payload: {e!r}") from e
        # On the validated ids, so a numeric string reads as the number it names.
        _check_token_ids(payload.tokens, "choices[0].token_ids")
        meta: dict = _usage_counts(data.get("usage"))
        ctx = data.get("hls_context")
        if isinstance(ctx, dict):
            meta.update(_hls_context_metadata(ctx))
        # Optional per-response GPU snapshot (utilization, memory) a server may
        # stamp. Its scalar entries pass through for report-only serving metrics to
        # reduce; absent when the server samples none. Nothing scored reads it.
        gpu = data.get("gpu")
        if isinstance(gpu, dict):
            meta["gpu"] = _scalar_entries(gpu)
        return Envelope(request_id=request_id, contents=[payload], metadata=meta)

    def stream_token_count(self, chunk: dict) -> int:
        """The length of ``choices[0].token_ids``, which is where this protocol's
        stream carries its output tokens (``_fold_stream``). Zero for any frame that
        does not have one, including a malformed frame: the fold rejects those with
        the whole stream in hand."""
        if not isinstance(chunk, dict):
            return 0
        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return 0
        tids = choices[0].get("token_ids")
        return len(tids) if isinstance(tids, (list, tuple)) else 0

    def _fold_stream(self, timed_chunks: list[tuple[float, dict]],
                     acc: "_StreamFold") -> None:
        """Fold every frame of an SSE stream into ``acc``, raising
        ArtifactProtocolError on the first non-conformant one.

        Accumulates into a caller-owned object rather than into locals so a caller that
        catches the error still holds everything the stream sent before it -- which is
        what ``decode_partial`` reports on a request that never completed."""
        for t, chunk in timed_chunks:
            if not isinstance(chunk, dict):
                # A frame is a JSON object per the SSE contract. `data: 5` parses fine
                # and would otherwise raise a bare AttributeError below, escaping the
                # fault taxonomy and invalidating the whole run over one bad frame.
                raise ArtifactProtocolError(
                    f"stream frame is not an object: {type(chunk).__name__}")
            acc.usage.update(_usage_counts(chunk.get("usage")))
            # Serving configuration and GPU telemetry ride on whatever frame the
            # server chooses, usually the final one. Last frame wins.
            ctx = chunk.get("hls_context")
            if isinstance(ctx, dict):
                acc.hls_context = ctx
            gpu = chunk.get("gpu")
            if isinstance(gpu, dict):
                acc.gpu = _scalar_entries(gpu)
            choices = chunk.get("choices")
            if choices is not None and not isinstance(choices, list):
                raise ArtifactProtocolError(
                    f"stream frame choices is not a list: {type(choices).__name__}")
            if not choices:
                # A chunk with no choices carries no tokens, but it may carry the server's
                # own account of what went wrong. Keep the newest one: a server that dies
                # mid-stream typically emits {"error": ...} and then [DONE] on the already
                # committed 200, and without this the fault we report is
                # "stream ended without finish_reason" while the cause sat in a frame we
                # dropped.
                err = chunk.get("error")
                if err is not None:
                    acc.server_error = err.get("message") if isinstance(err, dict) else err
                continue
            try:
                ch = choices[0]
                if not isinstance(ch, dict):
                    raise ArtifactProtocolError(
                        f"choices[0] is not an object: {type(ch).__name__}")
                content = self._stream_text(ch)
                tids = ch.get("token_ids")
                finish = ch.get("finish_reason")
                # The whole frame is read before any of it is kept. A frame that fails
                # still leaves the fold behind for decode_partial, and that is written
                # into the trace, so what the fold holds has to be what the trace can
                # write: strings and integers, never a nested value.
                if content is not None and not isinstance(content, str):
                    raise ArtifactProtocolError(
                        f"stream chunk text is not a string: {type(content).__name__}")
                if finish is not None and not isinstance(finish, str):
                    raise ArtifactProtocolError(
                        f"finish_reason is not a string: {type(finish).__name__}")
                if tids:
                    try:
                        tids = _TOKEN_ID_LIST.validate_python(tids)
                    except ValidationError as e:
                        raise ArtifactProtocolError(
                            f"stream chunk token_ids is not a list of integers: "
                            f"{e.errors()[0]['msg']}") from e
                    _check_token_ids(tids, "stream chunk token_ids")
                lp = self._parse_logprobs(ch.get("logprobs"))
                # Each content-bearing chunk must carry its token_ids. Checking only
                # that some chunk had ids is too weak: a chunk that drops ids would
                # leave tokens shorter than / misaligned with text, silently
                # mis-scoring the token-level verifier. Only where the protocol
                # requires ids, though: one that does not stops asking for them in
                # `_flags` and stops documenting them in its brief, so demanding them
                # here would fault every streamed reply over a field the request never
                # asked the server for.
                if content:
                    if self.emits_token_ids and not tids:
                        raise ArtifactProtocolError(
                            "stream chunk emitted content without token_ids "
                            "(mandatory on every content-bearing delta)")
                    acc.text_parts.append(content)
                if tids:
                    # TTFT keys off the first token packet, not the first
                    # content-bearing delta: a leading token_ids-only chunk is a real
                    # packet arrival, and keying on content would desync TTFT from
                    # packet_timestamps[0].
                    if acc.t_first_packet is None:
                        acc.t_first_packet = t
                    acc.tokens.extend(tids)
                    # One timestamp per emitted token, aligned with the per-token
                    # stream for first-packet/TPOT SLOs.
                    acc.packet_timestamps.extend([t] * len(tids))
                elif content and not self.emits_token_ids:
                    # Without ids there is nothing else to count a packet by, so the
                    # delta itself is one. Otherwise such a stream reports no packets at
                    # all and a TTFT/TPOT SLO has nothing to read.
                    if acc.t_first_packet is None:
                        acc.t_first_packet = t
                    acc.packet_timestamps.append(t)
                if lp:
                    acc.logprobs.extend(lp)
                if finish:
                    acc.finish_reason = finish
            except ArtifactProtocolError:
                raise
            except (KeyError, IndexError, TypeError, AttributeError, ValidationError) as e:
                raise ArtifactProtocolError(f"malformed stream chunk: {e!r}") from e

    def decode_stream(self, timed_chunks: list[tuple[float, dict]],
                      request_id: str) -> ArtifactResponse:
        acc = _StreamFold()
        self._fold_stream(timed_chunks, acc)
        if acc.finish_reason is None:
            # Report the server's own words when it gave them; the request failed either
            # way, so this changes diagnosis, never the verdict.
            if acc.server_error is not None:
                raise ArtifactProtocolError(
                    "stream ended without finish_reason; server reported: "
                    f"{str(acc.server_error)[:2000]}")
            raise ArtifactProtocolError(
                "stream ended without finish_reason (truncated or empty)")
        self._check_finish_reason(acc.finish_reason)
        if self.emits_token_ids and not acc.tokens:
            # Same rule as the unary path, published on the same condition -- including
            # the condition that this protocol requires ids at all.
            self._refuse_zero_token_reply("stream carried no token_ids")
        meta: dict = {k: acc.usage[k] for k in _USAGE_KEYS if k in acc.usage}
        if acc.hls_context is not None:
            meta.update(_hls_context_metadata(acc.hls_context))
        if acc.gpu is not None:
            meta["gpu"] = acc.gpu
        try:
            payload = TextPayload(
                text="".join(acc.text_parts),
                tokens=acc.tokens or None,
                logprobs=acc.logprobs or None,
                finish_reason=acc.finish_reason)
        except (ValidationError, KeyError, TypeError) as e:
            raise ArtifactProtocolError(f"malformed payload: {e!r}") from e
        env = Envelope(request_id=request_id, contents=[payload], metadata=meta)
        return ArtifactResponse(output=env, t_first_packet=acc.t_first_packet,
                                packet_timestamps=acc.packet_timestamps or None)

    def stream_packet_bytes(self, frame_bytes: Callable[[Mapping[str, Any]], int]) -> int | None:
        """A stream that carries token ids times each id (``_fold_stream``), at its densest a
        one-digit id and its comma in a frame's id list, whatever the frame around it. One
        without ids times each content delta, whose smallest frame this protocol does not
        declare."""
        return len("0,") if self.emits_token_ids else None

    def trace_growth(self, *, stream: bool) -> float:
        """A whole reply's is the base protocol's. On a stream that carries token ids every id is
        a timed packet (``_fold_stream``): at its densest a one-digit id and its comma, two bytes
        of a frame's id list, which the event keeps as the id in the reply's tokens beside the
        packet's time, at most ``hls.trace.store.TIMESTAMP_JSON_BYTES``: 13.5 bytes of trace per
        byte. Its text grows no more than a whole reply's. One without ids times each content
        delta, whose smallest frame this protocol does not declare, so it does not say
        (ValueError)."""
        from hls.trace.store import ESCAPE_GROWTH, TIMESTAMP_JSON_BYTES
        if not stream:
            return super().trace_growth(stream=False)
        if not self.emits_token_ids:
            return super().trace_growth(stream=True)
        per_id = len("0,")
        return float(max(ESCAPE_GROWTH, (per_id + TIMESTAMP_JSON_BYTES) / per_id))

    def decode_partial(self, timed_chunks: list[tuple[float, dict]]) -> PartialOutput | None:
        """The tokens, text, report and token arrival times an incomplete stream sent,
        up to the point it stopped making sense.

        Folds the same way ``decode_stream`` does and keeps what accumulated before the
        first non-conformant frame, so a stream that broke mid-reply reports the reply
        so far rather than nothing."""
        acc = _StreamFold()
        try:
            self._fold_stream(timed_chunks, acc)
        except ArtifactProtocolError:
            pass                        # keep what arrived before the bad frame
        if not (acc.tokens or acc.text_parts or acc.finish_reason):
            return None
        return PartialOutput(token_ids=list(acc.tokens), text="".join(acc.text_parts),
                             finish_reason=acc.finish_reason,
                             logprobs=list(acc.logprobs) or None,
                             packet_timestamps=list(acc.packet_timestamps))


class OpenAIChatProtocol(_OpenAIProtocol):
    """OpenAI ``/v1/chat/completions``: chat-templated messages in, ``choices[].message``
    out. Conditioning stays ``"chat"`` (the base default): the server renders the
    checkpoint's chat template over the message list."""

    return_logprobs: bool = False
    top_logprobs: int | None = None
    #: Whether this task's turns carry tool definitions. Declared rather than
    #: inferred, because the brief and the sample request bodies are built before any
    #: turn exists: an agentic task sends ``metadata['tools']`` on every request and a
    #: contestant who built against a sample that did not show ``tools`` or
    #: ``tool_choice`` would meet both for the first time in a graded run. Only the
    #: brief and the samples read it; ``encode_request`` decides per turn on what the
    #: turn actually carries.
    #:
    #: ``exclude=True`` because this is documentation, not identity: it changes what
    #: the brief says and nothing about the traffic, so it must not enter the
    #: calibration fingerprint (``workload_payload`` hashes ``dump_spec(contract)``).
    #: Without it, declaring this on one task would stale every committed calibration
    #: in the repository, including tasks that send no tools at all.
    sends_tools: bool = Field(default=False, exclude=True)
    #: One reply as THIS task's checkpoint emits it: its reasoning, its prose and a
    #: tool call, all in ``content``. Shown as the response sample's ``content``.
    #:
    #: Required when ``sends_tools`` is set, and per-task because call syntax is the
    #: checkpoint's, not this protocol's -- GLM writes
    #: ``<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>``,
    #: Qwen writes ``<tool_call><function=NAME><parameter=K>V</parameter></function>``
    #: and Gemma 4 writes ``<|tool_call>call:NAME{K:<|"|>V<|"|>}<tool_call|>``. A
    #: single hard-coded example would ship a task a sample of a dialect its
    #: checkpoint does not speak, which is the class of defect that costs a
    #: contestant a run.
    #:
    #: Excluded from the dump for the same reason ``sends_tools`` is: it is prose in
    #: a brief, not traffic.
    tool_call_example: str | None = Field(default=None, exclude=True)
    #: Whether the brief says, above the response sample, that the sample's
    #: ``top_logprobs`` arrays are shorter than a conforming reply's. The sample shows
    #: two alternatives where the prose demands ``top_logprobs``, which reads as a
    #: contradiction unless the brief says which is the rule.
    #:
    #: Opt-in, so a brief carries the sentence only where its task asks for it, and
    #: excluded from the dump for the same reason ``sends_tools`` is: it is prose in a
    #: brief, not traffic.
    states_sample_abbreviation: bool = Field(default=False, exclude=True)

    @model_validator(mode="after")
    def _validate_tool_call_example(self) -> "OpenAIChatProtocol":
        """A task that sends tools must say what its checkpoint's calls look like.

        A construction error rather than a fallback to some default dialect, because
        silence is how the wrong markup travels: a task that inherited another
        checkpoint's example would ship a brief teaching a syntax its model never
        emits, and nothing downstream would notice."""
        if self.sends_tools and not self.tool_call_example:
            raise ValueError(
                "sends_tools=True needs tool_call_example: one reply as THIS task's "
                "checkpoint emits it, reasoning and tool call together in `content`. "
                "Call syntax is the checkpoint's, so there is no default that is not "
                "wrong for some task.")
        return self

    @model_validator(mode="after")
    def _validate_top_logprobs(self) -> "OpenAIChatProtocol":
        if self.top_logprobs is not None:
            if not self.return_logprobs:
                raise ValueError("top_logprobs requires return_logprobs=True")
            # OpenAI's own API caps this at 20. vLLM and SGLang both accept more,
            # and the report contract this harness grades asks for 21 alternatives
            # per position, so the bound is the serving stacks' and not OpenAI's.
            if not (0 <= self.top_logprobs <= 32):
                raise ValueError("top_logprobs must be in [0, 32]")
        return self

    @model_validator(mode="after")
    def _validate_sample_abbreviation(self) -> "OpenAIChatProtocol":
        """The sentence says the sample shows fewer alternatives than a reply carries, so
        it needs a sample that does."""
        if self.states_sample_abbreviation:
            entries = self.body_schema()["response"]["choices"][0].get("logprobs", {}).get("content", [])
            shown = len(entries[0].get("top_logprobs") or []) if entries else 0
            if not shown or shown >= (self.top_logprobs or 0):
                raise ValueError(f"states_sample_abbreviation needs a sample shorter than the top_logprobs it "
                                 f"abbreviates: the sample shows {shown}, the reply carries {self.top_logprobs}")
        return self

    def probe_request(self) -> dict:
        # Built by encode_request so the probe is a real instance of the schema the brief
        # publishes. A hand-written body would omit `return_token_ids`, which encode_request
        # sends on every request and the brief documents, so a server that validated the
        # documented body would reject the probe -- and the readiness gate reads a
        # rejection as "route not registered yet" and polls until the deploy times out.
        #
        # The model is self.model, never a placeholder. Stock vLLM 404s a request whose
        # `model` does not match --served-model-name, and the gate treats 404 as not-yet-ready, so a
        # placeholder makes a healthy server look permanently unready.
        return self.encode_request(
            Envelope(request_id="probe", contents=[TextPayload(text="_")],
                     metadata={"max_tokens": 1}))

    # --- request ---
    def _flags(self) -> dict:
        body = super()._flags()
        if self.return_logprobs:
            body["logprobs"] = True
            if self.top_logprobs is not None:
                body["top_logprobs"] = self.top_logprobs
        return body

    def encode_request(self, env: Envelope, *, stream: bool = False) -> dict:
        body: dict = {"model": self.model, "messages": self._conversation_of(env)}
        # metadata["tools"] carries the tool definitions this turn offers, when the
        # task drives an agentic loop. It rides on the request so the artifact and
        # the reference render one context: the chat template folds tool definitions
        # into the prompt, so a turn whose tools differ is a different prompt, and a
        # teacher-forced gap would then measure the harness, not the artifact. Absent
        # or empty means the turn offers none, and the key is left off entirely.
        tools = env.metadata.get("tools")
        if isinstance(tools, list) and tools:
            body["tools"] = tools
        if self.sends_tools and isinstance(tools, list) and tools:
            # "none", explicitly: vLLM defaults `tool_choice` to "auto" whenever `tools` is
            # present and then refuses the request unless it was launched with
            # --enable-auto-tool-choice and a --tool-call-parser. "none" still renders the
            # tool definitions into the prompt and leaves the model's own call in `content`,
            # which is the contract the brief states. It selects a server behaviour and is
            # never rendered by a chat template, so the prompt, and the fingerprint, do not
            # change with it. Sent only by a protocol that declares ``sends_tools``.
            body["tool_choice"] = "none"
        self._stamp_continuation(body, env)
        self._stamp_max_tokens(body, env)
        body.update(self._flags())
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    # --- response (unary) ---
    def _parse_logprobs(self, lp: dict | None) -> list[TokenLogprob] | None:
        if not lp or not lp.get("content"):
            return None
        out = []
        for e in lp["content"]:
            tid = e.get("token_id")
            if not isinstance(tid, int) or isinstance(tid, bool):
                raise ArtifactProtocolError(
                    f"logprob entry missing integer token_id: {e!r}")
            _check_token_id(tid, "logprob entry")
            tops = e.get("top_logprobs")
            parsed_tops = None
            if tops:
                parsed_tops = []
                for t in tops:
                    ttid = t.get("token_id")
                    if not isinstance(ttid, int) or isinstance(ttid, bool):
                        raise ArtifactProtocolError(
                            f"top_logprobs alternative missing integer token_id: {t!r}")
                    _check_token_id(ttid, "top_logprobs alternative")
                    # Only the id and the log-probability are kept. A reply carries
                    # up to 21 alternatives per generated token and the whole trace
                    # is dumped to disk, so holding each alternative's decoded text
                    # and utf-8 bytes costs gigabytes for fields nothing reads: the
                    # verifier keys on ids.
                    parsed_tops.append(TopLogprob(token_id=ttid, logprob=t["logprob"]))
            out.append(TokenLogprob(
                token=e["token"], token_id=tid, logprob=e["logprob"],
                bytes=e.get("bytes"), top_logprobs=parsed_tops))
        return out

    def _unary_text(self, choice: dict) -> str | None:
        return (choice.get("message") or {}).get("content")

    def _stream_text(self, choice: dict) -> str | None:
        return (choice.get("delta") or {}).get("content")

    # --- brief ---
    def body_schema(self) -> dict:
        request = {"model": self.model,
                   "messages": [{"role": "user", "content": "<prompt>"}]}
        # An agentic task's turns carry both of these, so the sample shows both. A
        # contestant who builds against a sample and then meets an unfamiliar key in a
        # graded run is the failure this exists to avoid.
        if self.sends_tools:
            request["tools"] = [{"type": "function", "function": {
                "name": "<tool-name>", "description": "<what it does>",
                "parameters": {"type": "object",
                               "properties": {"<arg>": {"type": "string"}},
                               "required": ["<arg>"]}}}]
            request["tool_choice"] = "none"
        # `max_tokens` and `stream` ride every graded request too, stamped per turn by the
        # driver; they are stated in the prose rather than shown here because this sample
        # is held equal to an encoded request that carries no envelope.
        request.update(self._flags())
        # `content` carries the whole generation -- reasoning, prose and any tool call
        # the model wrote -- because nothing on the server side is allowed to move
        # parts of it elsewhere. Shown as an actual reply rather than as "<answer>",
        # with the call sitting inside `content`.
        #
        # The reply itself comes from the task, because call syntax belongs to the
        # checkpoint and this schema is shared.
        answer = (f"{self.tool_call_example}   <-- the whole generation, verbatim"
                  if self.sends_tools else "<answer>")
        choice: dict = {"message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                        "token_ids": [1234, 5678]}
        if self.return_logprobs:
            entry: dict = {"token": "<tok>", "token_id": 1234, "logprob": -0.1,
                           "bytes": [60, 116]}
            if self.top_logprobs is not None:
                entry["top_logprobs"] = [
                    {"token": "<tok>", "token_id": 1234, "logprob": -0.1},
                    {"token": "<alt>", "token_id": 5678, "logprob": -1.5},
                ]
            choice["logprobs"] = {"content": [entry]}
        response = {"choices": [choice], "usage": {"completion_tokens": 2}}
        return {"request": request, "response": response}

    def brief_section(self, *, path: str, streaming: bool = False) -> str:
        schema = self.body_schema()
        # The count the sentence states is the sample's own, so the two cannot disagree.
        shown = None
        if self.states_sample_abbreviation:
            [entry] = schema["response"]["choices"][0]["logprobs"]["content"]
            shown = len(entry["top_logprobs"])
        return jinja2.Template(_OPENAI_BRIEF, trim_blocks=True, lstrip_blocks=True).render(
            path=path,
            request=json.dumps(schema["request"], indent=2),
            response=json.dumps(schema["response"], indent=2),
            streaming=streaming,
            stop_token_ids=list(self.stop_token_ids) if self.stop_token_ids else None,
            return_logprobs=self.return_logprobs,
            top_logprobs=self.top_logprobs,
            top_logprobs_shown=shown,
            sends_tools=self.sends_tools)


_COMPLETION_BRIEF = """\
### POST {{ path }}
Accept OpenAI text-completion requests and return text-completion responses. The
request carries a raw `prompt` string that is already the exact model input: feed it
to the model verbatim, with no chat template, no role markup and no added special
tokens. The harness sends `max_tokens` on every request body; treat it as a binding
per-request output cap. Do not substitute a server-side default when the request
supplies `max_tokens`.

Return a `token_ids` array on each choice (`choices[].token_ids`): the integer
vocabulary IDs of the exact tokens you emitted (NOT a re-tokenization of the text).
This is mandatory on every response.
{% if stop_token_ids %}
Every request carries `stop_token_ids` {{ stop_token_ids }}: generation ends when the
model emits any of them. `token_ids` must include that end-of-turn token when the
reply ended on one, even though it is not part of the returned `text`. A reply whose
first generated token is one of them is one token, not zero. A response with an
empty `token_ids` array is unverifiable and fails the run.
{% endif %}{% if streaming %}
Support streaming: when the request sets `"stream": true`, respond with an OpenAI
SSE stream (`data: {chunk}` lines of `text_completion` objects whose `choices[].text`
holds that chunk's text, terminated by `data: [DONE]`). Include the per-chunk
`token_ids` array for that chunk's tokens.
{% endif %}
Request:
```json
{{ request }}
```
Response:
```json
{{ response }}
```
"""


class OpenAICompletionProtocol(_OpenAIProtocol):
    """OpenAI ``/v1/completions``: a raw ``prompt`` string in, ``choices[].text`` out.

    Canonical for the fields it uses: the request envelope (``model``, ``prompt``,
    ``max_tokens``, ``stream``) and the response (``choices[{text, index,
    finish_reason}]``, ``usage``) are the published shapes. Three fields are not, and
    are the same extensions ``OpenAIChatProtocol`` already relies on: ``token_ids`` on
    every choice (mandatory here), ``return_token_ids`` and ``stop_token_ids`` on the
    request (vLLM/SGLang extensions; a server that does not know them ignores them).
    Streaming chunks have the same shape as the unary reply, with no ``delta``
    envelope. Stock vLLM 0.22.1 attaches ``token_ids`` to every content-bearing
    completions chunk when the request carries ``return_token_ids`` (its
    ``completion/serving.py`` streaming generator), which is what ``_fold_stream``
    requires; the unary reply carries the whole list.

    ``conditioning`` is ``"raw"``: the prompt is the model input. A reference engine
    built from a task using this protocol therefore never applies a chat template.
    Canonical completions logprobs are an integer count with a
    ``{tokens, token_logprobs, top_logprobs, text_offset}`` reply, not chat's
    ``logprobs.content[]``, so this protocol requests none and
    ``check_emit_consistency`` cannot be enabled on it."""

    conditioning: ClassVar[Literal["chat", "raw"]] = "raw"

    def probe_request(self) -> dict:
        return {"model": self.model, "prompt": "_", "max_tokens": 1}

    @classmethod
    def _prompt_of(cls, env: Envelope) -> str:
        """The raw prompt: the one message of the request's conversation -- a session
        step's own turn, the driver's single user turn, or the text payload when there
        is no message list.

        Reads that conversation through ``messages_of``, the same list the reference
        engine renders a raw context from, so the two pick the same message and not
        merely stringify it the same way. ``messages_of`` drops an entry missing
        ``role`` or ``content``, and a list of nothing else leaves no prompt to send.

        A raw prompt is the model input verbatim, so nothing in it can carry a
        conversation: only the last message is transmissible. A request holding more than
        one is refused rather than reduced to its last. The driver stitches every reply
        into the next request and measures the result against the tenant's context budget,
        so dropping the rest would quietly turn a multi-turn workload into a single-turn
        one. Nothing downstream would notice, since the reference reads the same last
        message (``ReferenceEngine._render_context``). A tenant that wants raw
        conditioning pre-renders each turn's own conversation in the turn envelope's
        ``metadata['messages']`` (``hls.spec.drivers.request_messages``)."""
        msgs = cls._conversation_of(env)
        if len(msgs) > 1:
            raise HarnessFault(
                f"request {env.request_id!r} carries {len(msgs)} messages under raw "
                f"conditioning, which can send only the last one. Pre-render each turn's "
                f"own conversation in the turn envelope's metadata['messages'], or give "
                f"the tenant a chat-conditioned protocol.")
        return message_content(msgs[-1]) if msgs else ""

    def encode_request(self, env: Envelope, *, stream: bool = False) -> dict:
        body: dict = {"model": self.model, "prompt": self._prompt_of(env)}
        self._stamp_max_tokens(body, env)
        body.update(self._flags())
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def _unary_text(self, choice: dict) -> str | None:
        return choice.get("text")

    def _stream_text(self, choice: dict) -> str | None:
        return choice.get("text")

    def body_schema(self) -> dict:
        request = {"model": self.model, "prompt": "<prompt>"}
        request.update(self._flags())
        response = {"choices": [{"text": "<completion>", "finish_reason": "stop",
                                 "token_ids": [1234, 5678]}],
                    "usage": {"completion_tokens": 2}}
        return {"request": request, "response": response}

    def brief_section(self, *, path: str, streaming: bool = False) -> str:
        schema = self.body_schema()
        return jinja2.Template(_COMPLETION_BRIEF, trim_blocks=True, lstrip_blocks=True).render(
            path=path, request=json.dumps(schema["request"], indent=2),
            response=json.dumps(schema["response"], indent=2), streaming=streaming,
            stop_token_ids=list(self.stop_token_ids) if self.stop_token_ids else None)


_OMNI_BRIEF = """\
### POST {{ path }}
Accept OpenAI multimodal chat/completions requests: each message's ``content`` is a
list of typed parts -- ``{"type": "text", ...}``, ``{"type": "image_url", "image_url":
{"url": "data:image/...;base64,..."}}``, and ``{"type": "input_audio", "input_audio":
{"data": "<base64 WAV>", "format": "wav"}}``. A request may carry an image + a question
or a speech clip; its text part says what to do with the input. Decode greedily and
return a normal chat completion whose message content is the text answer.

Request:
```json
{{ request }}
```
Response:
```json
{{ response }}
```
{% if return_token_ids %}Return a ``token_ids`` array on each choice (the integer ids of
the exact tokens you emitted).{% endif %}
"""


#: Sampling pinned on the wire for the omni protocol, spelled as
#: `hls.reference.engines._GREEDY_SAMPLING` spells it so the two cannot drift.
#:
#: The brief for this task already says "Decode greedily (argmax, no sampling)" -- it
#: inherits the base brief's decoding block -- so this enforces a stated rule rather than
#: adding one. Without it a contestant serving through vLLM inherits the checkpoint's
#: `generation_config` (a repetition_penalty of 1.05, say) while the oracle decodes
#: greedily, so a contestant could comply with the brief in good faith and still be
#: scored against a different decode than the oracle's.
_OMNI_GREEDY_ON_THE_WIRE: dict[str, float | int] = {
    "temperature": 0.0, "top_p": 1.0, "top_k": -1,
    "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
}


class OmniChatProtocol(OpenAIChatProtocol):
    """Multimodal OpenAI chat codec for omni models (multimodal-omni): the request carries image /
    audio / text input parts in one ``messages[].content`` list; the response is text
    (decoded by the inherited ``decode_response``/``decode_stream``).

    Only ``encode_request`` differs from the parent -- it renders each input payload to
    an OpenAI content part: TextPayload -> ``text``, ImagePayload -> ``image_url`` (a
    ``data:<mime>;base64,...`` URI), AudioPayload -> ``input_audio`` (base64 WAV). The
    decode path, logprobs, and streaming are all inherited.

    Token ids are not required of an omni server. The multimodal stacks that serve
    these models return a text answer and no per-token ids, and nothing gates this
    task at the token level (its verifiers compare task metrics against an oracle), so
    demanding them would fault every response for evidence no gate reads. The parent's
    ``decode_response`` and ``_flags`` both key off this marker, so the request stops
    asking for the field at the same time the decode stops requiring it."""

    emits_token_ids: ClassVar[bool] = False

    def _flags(self) -> dict:
        """The inherited flags, plus greedy decode pinned on the wire.

        Scoped to this protocol. The other OpenAI-shaped protocols share the gap, but
        pinning there would change what their contestants receive.
        """
        body = super()._flags()
        body.update(_OMNI_GREEDY_ON_THE_WIRE)
        return body

    def _content_parts(self, env: Envelope) -> list[dict]:
        parts: list[dict] = []
        for p in env.contents:
            if isinstance(p, TextPayload):
                parts.append({"type": "text", "text": p.text})
            elif isinstance(p, ImagePayload):
                parts.append({"type": "image_url", "image_url": {"url": p.data_uri()}})
            elif isinstance(p, AudioPayload):
                parts.append({"type": "input_audio",
                              "input_audio": {"data": p.audio_b64, "format": "wav"}})
        return parts

    def encode_request(self, env: Envelope, *, stream: bool = False) -> dict:
        body: dict = {"model": self.model,
                      "messages": [{"role": "user", "content": self._content_parts(env)}]}
        self._stamp_max_tokens(body, env)
        body.update(self._flags())
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def body_schema(self) -> dict:
        request: dict = {"model": self.model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "<prompt>"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}}]}]}
        request.update(self._flags())
        choice: dict = {"message": {"role": "assistant", "content": "<answer>"},
                        "finish_reason": "stop"}
        if self.emits_token_ids:
            choice["token_ids"] = [1234, 5678]
        return {"request": request, "response": {"choices": [choice],
                                                 "usage": {"completion_tokens": 2}}}

    def brief_section(self, *, path: str, streaming: bool = False) -> str:
        schema = self.body_schema()
        return jinja2.Template(_OMNI_BRIEF, trim_blocks=True, lstrip_blocks=True).render(
            path=path,
            request=json.dumps(schema["request"], indent=2),
            response=json.dumps(schema["response"], indent=2),
            return_token_ids=self.emits_token_ids)


_ASR_BRIEF = """\
### POST {{ path }}
Accept a JSON speech-transcription request and return the transcript.

Request body (JSON):
```json
{{ request }}
```
`audio_b64` is a base64-encoded PCM16 mono WAV at `sample_rate` Hz; transcribe it
and return the text. The harness sends `max_tokens` on every request as a binding
per-request decode budget -- pass it through as your `max_new_tokens` and do not
substitute a server-side default. An empty `audio_b64` is a readiness probe -- return
`{"text": ""}` quickly without invoking the model.

Response body (JSON):
```json
{{ response }}
```
Return the integer decoder token ids you emitted under `token_ids` (NOT a
re-tokenization of the text); the harness uses them for token-level checks.
"""


class AudioTranscriptionProtocol(WireProtocol):
    """JSON speech-to-text codec (audio in -> text out).

    Audio rides as a base64 PCM16 mono WAV inside the JSON request body (the same
    representation an ``AudioPayload`` carries on the wire), so it flows over the
    plain ``http`` transport with no multipart machinery: ``encode_request`` reads
    the envelope's ``AudioPayload`` into ``{audio_b64, sample_rate, ...}`` and
    ``decode_response`` reads ``{"text": ...}`` back into a ``TextPayload``. The
    server's ``token_ids`` (the emitted decoder ids) are carried through when
    present."""

    model: str = "served-model"
    sample_rate: int = 16_000
    language: str | None = None

    def encode_request(self, env: Envelope, *, stream: bool = False) -> dict:
        audio = env.first_of("audio")
        audio_b64 = audio.audio_b64 if isinstance(audio, AudioPayload) else ""
        sr = audio.sample_rate if isinstance(audio, AudioPayload) else self.sample_rate
        body: dict = {"model": self.model, "audio_b64": audio_b64, "sample_rate": sr}
        if self.language is not None:
            body["language"] = self.language
        # The driver stamps the workload's per-request output cap; forward it so the
        # artifact and the oracle decode under identical budgets (parity).
        max_tokens = env.metadata.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            body["max_tokens"] = max_tokens
        return body

    def decode_response(self, data: dict, request_id: str) -> Envelope:
        if not isinstance(data, dict) or "text" not in data:
            raise ArtifactProtocolError(f"transcription response missing 'text': {data!r}")
        text = data["text"]
        if not isinstance(text, str):
            raise ArtifactProtocolError(
                f"transcription 'text' is not a string: {type(text).__name__}")
        try:
            payload = TextPayload(text=text, tokens=data.get("token_ids"),
                                  finish_reason=data.get("finish_reason", "stop"))
        except (ValidationError, TypeError) as e:
            raise ArtifactProtocolError(f"malformed transcription payload: {e!r}") from e
        return Envelope(request_id=request_id, contents=[payload])

    def probe_request(self) -> dict:
        # Built by encode_request, so the probe is a real instance of the schema the
        # brief publishes and cannot drift from it. A server that validates the documented
        # body answers 422 to any other, which the readiness gate reads as "route not
        # registered yet" and polls until the deploy times out.
        #
        # An envelope with no audio payload encodes an empty audio_b64, which the brief
        # documents as the readiness probe a server may fast-return {"text": ""} for, so
        # the probe still never blocks on a real transcription.
        return self.encode_request(
            Envelope(request_id="probe", contents=[], metadata={"max_tokens": 1}))

    def body_schema(self) -> dict:
        request = {"model": self.model, "audio_b64": "<base64 PCM16 mono WAV>",
                   "sample_rate": self.sample_rate}
        if self.language is not None:
            request["language"] = self.language
        # The driver stamps a binding per-request decode cap; document it so an agent that
        # builds a strict request model from the brief accepts it (encode_request sends it).
        request["max_tokens"] = 440
        response = {"text": "<transcript>", "token_ids": [1234, 5678]}
        return {"request": request, "response": response}

    def brief_section(self, *, path: str, streaming: bool = False) -> str:
        schema = self.body_schema()
        return jinja2.Template(_ASR_BRIEF, trim_blocks=True, lstrip_blocks=True).render(
            path=path,
            request=json.dumps(schema["request"], indent=2),
            response=json.dumps(schema["response"], indent=2))


def _codec_codes_defect(value: object, *, groups: int, codebook_size: int) -> str | None:
    """What is wrong with the codec codes a reply sent, or None when nothing is.

    The shape: a non-empty list of frames, each frame exactly ``groups`` integer ids in
    ``[0, codebook_size)``. Reported rather than raised, and the first violation wins.
    The caller records the answer on the payload, so a reply that got this wrong fails
    its own correctness gate instead of taking down the drive.

    A bool is refused on its own account: ``bool`` subclasses ``int``, so a JSON
    ``true`` would otherwise be stored as code 1 and graded as an id the server chose.
    """
    if not isinstance(value, list) or not value:
        return f"codec_codes is not a non-empty list of frames (got {value!r:.80})"
    for i, row in enumerate(value):
        if not isinstance(row, list):
            return f"codec_codes frame {i} is not a list of ids: {type(row).__name__}"
        if len(row) != groups:
            return f"codec_codes frame {i} carries {len(row)} ids, expected {groups}"
        for j, code in enumerate(row):
            if isinstance(code, bool) or not isinstance(code, int):
                return f"codec_codes frame {i} group {j} is not an integer id: {code!r}"
            if not 0 <= code < codebook_size:
                return (f"codec_codes frame {i} group {j} is {code}, outside "
                        f"[0, {codebook_size})")
    return None


def _codec_report_entry(value: object, served: int, *, top_logprobs: int
                        ) -> CodecPosition | str:
    """One reported position as a ``CodecPosition``, or why it cannot be read as one.

    The shape: ``[lp_served, [[alt_id, alt_lp], ...]]``, at most ``top_logprobs``
    alternatives. The shape, and numbers the stored report can hold: a log-probability
    within float range and an alternative id within 64 bits. Whether the numbers are
    log-probabilities, whether enough distinct alternatives arrived to build the
    partition, and whether the mass is a distribution are all read by the gate
    (``hls.verify.report``), which says so far more precisely than this could and has
    to read them anyway.

    A row wider than the contract asked for is refused, though, where a narrower one is
    passed through. Width is the one property the gate cannot recover from: its
    partition keeps the atoms above a fixed threshold rank, so extra entries describe a
    coarsening other than the calibrated one. Too few is already a named gate failure
    with its own count."""
    if not isinstance(value, list) or len(value) != 2:
        return f"is not a [logprob, alternatives] pair: {value!r:.60}"
    lp, alts = value
    if isinstance(lp, bool) or not isinstance(lp, (int, float)):
        return f"reports a served log-probability that is not a number: {lp!r:.40}"
    served_lp = _as_float(lp)
    if served_lp is None:
        return f"reports a served log-probability past float range: {lp!r:.40}"
    if not isinstance(alts, list) or not alts:
        return f"carries no alternatives: {alts!r:.40}"
    if len(alts) > top_logprobs:
        return f"carries {len(alts)} alternatives, more than the {top_logprobs} requested"
    ids: list[int] = []
    lps: list[float] = []
    for alt in alts:
        if not isinstance(alt, list) or len(alt) != 2:
            return f"has an alternative that is not an [id, logprob] pair: {alt!r:.40}"
        alt_id, alt_lp = alt
        # No codebook bound on an alternative's id, unlike a served code's. A codebook-0
        # position's alternatives legitimately reach into the special range above the
        # codebook, which is where the end-of-speech id lives. The only bound is the
        # one the stored report can hold.
        if isinstance(alt_id, bool) or not isinstance(alt_id, int) or alt_id < 0:
            return f"has an alternative whose id is not a token id: {alt_id!r:.40}"
        if alt_id > MAX_TOKEN_ID:
            return f"has an alternative whose id is past 64 bits: {alt_id!r:.40}"
        if isinstance(alt_lp, bool) or not isinstance(alt_lp, (int, float)):
            return f"has an alternative whose log-probability is not a number: {alt_lp!r:.40}"
        alt_float = _as_float(alt_lp)
        if alt_float is None:
            return (f"has an alternative whose log-probability is past float range: "
                    f"{alt_lp!r:.40}")
        ids.append(alt_id)
        lps.append(alt_float)
    return CodecPosition(served, served_lp, ids, lps)


def _as_float(value: int | float) -> float | None:
    """``value`` as a float, or None for an integer too large for one.

    JSON integers have no size limit and Python reads them exactly, so a reply can send
    a log-probability that ``float`` refuses with OverflowError. A non-finite float is
    returned as it is: whether it can be a log-probability is the gate's question."""
    try:
        return float(value)
    except OverflowError:
        return None


def _codec_report(report: object, eos_report: object, eos_code: object,
                  codes: list[list[int]], *, groups: int, top_logprobs: int
                  ) -> tuple[list[CodecPosition] | None, str | None]:
    """A reply's report flattened over its positions, or why it does not conform.

    Exactly one of the two comes back. The flat order is the one
    ``AudioPayload.codec_logprobs`` documents and a consumer may assume: frames
    outermost, codebook groups within them, and the end-of-speech decision last. That
    last position has no row in ``codes``, so ``eos_code`` is the id served there."""
    if not isinstance(report, list):
        return None, f"codec_report is not a list of frames: {type(report).__name__}"
    if len(report) != len(codes):
        return None, (f"codec_report covers {len(report)} frames but codec_codes "
                      f"carries {len(codes)}")
    if isinstance(eos_code, bool) or not isinstance(eos_code, int) or eos_code < 0:
        return None, f"eos_code is not a token id: {eos_code!r:.40}"
    # The gate packs the served ids, this one included, into a 64-bit array.
    if eos_code > MAX_TOKEN_ID:
        return None, f"eos_code is past 64 bits: {eos_code!r:.40}"
    out: list[CodecPosition] = []
    for f, row in enumerate(report):
        if not isinstance(row, list):
            return None, (f"codec_report frame {f} is not a list of positions: "
                          f"{type(row).__name__}")
        if len(row) != groups:
            return None, (f"codec_report frame {f} covers {len(row)} groups, "
                          f"expected {groups}")
        for g, entry in enumerate(row):
            read = _codec_report_entry(entry, codes[f][g], top_logprobs=top_logprobs)
            if isinstance(read, str):
                return None, f"codec_report frame {f} group {g} {read}"
            out.append(read)
    read = _codec_report_entry(eos_report, eos_code, top_logprobs=top_logprobs)
    if isinstance(read, str):
        return None, f"eos_report {read}"
    out.append(read)
    return out, None


@dataclass
class _TTSCodesFold:
    """The codes side of a TTS stream, as the frames are read.

    Separate from the audio accumulation because the two have different failure modes:
    bad audio is a protocol fault that fails the request, while bad codes are a graded
    defect recorded on the reply. Holds what was collected and the first thing found
    wrong with it, so ``decode_stream`` can hand the payload one or the other.

    ``top_logprobs`` is the report width the contract asks for, or None when it asks
    for no report, in which case the report keys are not read at all."""

    groups: int
    codebook_size: int
    top_logprobs: int | None
    #: The codes of the one conforming codes frame seen so far.
    codes: list[list[int]] | None = None
    #: That frame's report, flattened over the positions the codes name.
    logprobs: list[TokenLogprob] | None = None
    #: The first thing found wrong, in the reply's own terms.
    defect: str | None = None

    def note(self, frame: dict) -> None:
        """Read one codes frame, keeping the first defect found."""
        if self.codes is not None or self.defect is not None:
            self._fail("stream carried more than one codec_codes frame")
            return
        defect = _codec_codes_defect(frame.get("codec_codes"), groups=self.groups,
                                     codebook_size=self.codebook_size)
        if defect is not None:
            self._fail(defect)
            return
        self.codes = [list(row) for row in frame["codec_codes"]]
        if self.top_logprobs is None:
            return
        self.logprobs, defect = _codec_report(
            frame.get("codec_report"), frame.get("eos_report"), frame.get("eos_code"),
            self.codes, groups=self.groups, top_logprobs=self.top_logprobs)
        if defect is not None:
            # The codes outlive a bad report. A gate that re-decodes them and compares
            # waveforms reads no report at all, so withholding them would fail that
            # gate over another gate's evidence.
            self.defect = defect

    def note_audio_after_codes(self) -> None:
        """Record that an audio frame followed the codes.

        The codes are the whole utterance the contract asks for, so audio after them
        means they describe only part of what was served. Nothing here can tell which
        part, so the reply loses them rather than being graded on a prefix."""
        if self.codes is not None or self.defect is not None:
            self._fail("an audio frame followed the codec_codes frame")

    def _fail(self, reason: str) -> None:
        """Record a defect that costs the reply both its codes and its report.

        The earliest reason is the one kept: it describes what the server did first,
        and every later complaint is downstream of it."""
        self.defect = self.defect or reason
        self.codes = None
        self.logprobs = None


#: One report entry for the brief's example codes frame: the served code's reported
#: log-probability, then alternatives as [id, log-probability]. Two of them where a
#: reply sends ``top_logprobs``, and the served id is 0 to match the example's codes.
#: Two rather than three because the example repeats this entry once per codebook
#: group, so every value in it costs a multiple of itself in a brief a contestant
#: reads under a token budget, and two is already a list.
_EXAMPLE_REPORT_ENTRY = [-0.412535, [[0, -0.412535], [17, -1.884102]]]

#: The same entry for the end-of-speech position, whose served id is the end-of-speech
#: id, so that is the one its alternatives carry. Reusing the frame entry there would
#: show an example the grader rejects: a served id outside the alternatives adds its own
#: mass to theirs, and 0.662 + 0.152 + 0.662 is more than a position has.
def _example_eos_entry(eos_id: int) -> list:
    return [-0.412535, [[eos_id, -0.412535], [17, -1.884102]]]


_TTS_BRIEF = """\
### POST {{ path }}
Accept a JSON text-to-speech request and stream the synthesized audio back as an SSE
stream: when ``"stream": true``, respond with ``data: {frame}`` lines (each frame a
base64 PCM16 mono WAV chunk under ``audio``{% if return_token_ids %}, optionally ``token_ids`` for the codec ids
you decoded{% endif %}), terminated by ``data: [DONE]``. You may stream frames as they are decoded or send them
together; only the arrival of the last audio frame is timed.

Each frame is a complete WAV file, its own RIFF header followed by mono PCM16 samples at
the rate pinned under ## Decoding, base64-encoded; the frames concatenated in order are
the utterance that is graded.

Request body (JSON):
```json
{{ request }}
```
`voice` is always `{{ voice }}`; the speaker to synthesize in is the one pinned under
## Decoding, whatever this field carries.

An empty `input` is a readiness probe, sent after your `/health` returns 200 and before
any real traffic. Answer it immediately without synthesizing, with a 2xx (an empty stream
is fine) or a 400. Those two count as ready; any other status, including 404, 405 and
422, means not ready yet, and the probe is repeated every 2 s until startup times out.

Per-frame SSE payload:
```json
{{ frame }}
```
{% if return_codec_codes %}
After the last audio frame and before `data: [DONE]`, send exactly one further frame
carrying the codec codes the audio was decoded from, and no audio:
```json
{{ codes_frame }}
```
`codec_codes` has one row per codec frame in generation order, {{ codec_groups }} integer ids per row
(one per codebook group), each in `[0, {{ codec_codebook_size }})`, with no end-of-speech row.
{% if return_codec_report %}
The example above is abbreviated twice: it shows one codec frame where a reply sends one
row per frame of audio, and two alternatives per position where a reply sends {{ top_logprobs }}.

`codec_report` covers exactly the same positions: `codec_report[f][g]` describes the one
whose served code is `codec_codes[f][g]`. Each is `[lp, [[id, lp], ...]]` -- the
log-probability your sampler gave the code you served there, then the {{ top_logprobs }} most probable
alternatives as `[id, log-probability]` pairs, whether or not you served one of them, all
under exactly the distribution you sampled from.

`eos_report` is the same for the end-of-speech decision, the codebook-0 position after
your last frame. `eos_code` is the id you served there, which for this checkpoint is {{ codec_eos_id }}.
It is a position on codebook 0's vocabulary, not a seventeenth group. Alternatives at any
codebook-0 position may name ids above the codebook.

Send the alternatives in whatever order your stack produces; do not deduplicate or
reorder them.
{% endif %}
This frame is the record of what your model chose, and correctness is graded on it; a
reply that omits it, sends it more than once, sends audio after it, or sends rows of
another shape fails. It is not charged as latency: the per-request clock stops at your
last audio frame.
{% endif %}
"""


#: The most bytes of trace one byte of a TTS frame's samples becomes: the frames' samples are kept
#: as one PCM16 WAV (``hls.core.audio.concat_wav_b64``), and the narrowest sample a frame's WAV may
#: carry, 8-bit (``hls.core.audio.decode_wav_b64``), is kept as two bytes. Both are base64.
_AUDIO_WIDENING = 2


class TTSStreamProtocol(WireProtocol):
    """Streaming text-to-speech codec: text in (one request), audio frames out (SSE).
    Each streamed frame is a base64 PCM16 WAV, optionally with the codec token ids it
    decoded, so the harness can assemble the full waveform and hand it to an audio
    verifier. Reuses the ``http-sse`` transport.

    Under ``return_codec_codes`` the reply also ends with the codes the whole utterance
    was decoded from, in one frame after the last audio frame. Under
    ``return_logprobs`` that frame also carries the server's own report of the
    distribution it sampled each code from, which is what a sampling gate reads. Both
    are the record of what the model chose, so a reply that gets them wrong is graded
    as wrong rather than faulted: ``decode_stream`` leaves the payload's
    ``codec_codes`` and ``codec_logprobs`` None and says why in ``codec_codes_defect``.
    Their frame is also not latency -- the response reports the last audio frame as its
    completion time, so sending them costs the server nothing."""

    model: str = "served-model"
    voice: str = "default"
    return_token_ids: bool = True
    #: Ask the server for the codec codes behind the whole utterance, in one frame at
    #: the end of the stream. Off unless a verifier reads them: a server told to
    #: compute and serialize what nothing grades is being charged for nothing.
    return_codec_codes: bool = False
    #: Ask that frame to carry the sampling distribution behind every code as well.
    #: Spelled as on ``OpenAIChatProtocol`` because ``check_wiring`` reads these two
    #: names off whatever protocol a report verifier is wired to.
    return_logprobs: bool = False
    #: How many alternatives per position the report must carry. The partition the gate
    #: builds keeps the atoms above a threshold rank counted in the gate's report width
    #: (``ReportFaithfulnessVerifier.report_width``), which this has to reach.
    top_logprobs: int | None = None
    #: Codebook groups per codec frame, the width of every ``codec_codes`` row and of
    #: every ``codec_report`` row. A property of the codec the task pins, checked on
    #: the wire so a reply whose rows are another width is caught here rather than
    #: misread as frames by the gate.
    codec_groups: int = 16
    #: Ids per codebook group: every code must lie in ``[0, codec_codebook_size)``. Not
    #: applied to the report's alternatives, which legitimately name ids above the
    #: codebook (the end-of-speech id among them).
    codec_codebook_size: int = 2048
    #: The id the talker emits to end an utterance, a codebook-0 vocabulary id above
    #: the codebook itself. Published in the brief and read off the contract by a gate
    #: that has to know it before it calls any model, which is why it is pinned here
    #: rather than learned from the checkpoint at verify time.
    #:
    #: Not enforced against the ``eos_code`` a reply sends. The trace records what the
    #: server said it served, so that a reply which ended on something else is visible
    #: as that rather than overwritten with what it should have been; the gate is what
    #: rules on the difference, with the reference's own end-of-turn ids in hand.
    codec_eos_id: int = 2150

    @model_validator(mode="after")
    def _validate_report_flags(self) -> "TTSStreamProtocol":
        """Refuse a report the reply could not carry or the server could not build.

        A report needs a width to be built at, and it needs the codes frame to ride in
        and to name the position each entry describes. Either flag alone asks the
        server for something this protocol has nowhere to put."""
        if self.return_logprobs and not self.return_codec_codes:
            raise ValueError(
                "return_logprobs requires return_codec_codes: the report rides in the "
                "codes frame and is indexed by the codes it reports on")
        if self.return_logprobs and not self.top_logprobs:
            raise ValueError("return_logprobs requires a positive top_logprobs")
        if self.top_logprobs is not None:
            if not self.return_logprobs:
                raise ValueError("top_logprobs requires return_logprobs=True")
            if self.top_logprobs < 1:
                raise ValueError("top_logprobs must be at least 1")
        return self

    @property
    def emits_codec_codes(self) -> bool:
        return self.return_codec_codes

    @staticmethod
    def smallest_frame() -> dict:
        """The smallest stream chunk this protocol times as a packet (``decode_stream``): one
        audio frame of a WAV with no samples, its header alone."""
        from hls.core.audio import pcm16_wav_b64
        return {"audio": pcm16_wav_b64(b"")}

    def stream_packet_bytes(self, frame_bytes: Callable[[Mapping[str, Any]], int]) -> int | None:
        """Each audio frame is one packet, so its smallest frame (``smallest_frame``) is the
        fewest bytes one costs."""
        return frame_bytes(self.smallest_frame())

    def trace_growth(self, *, stream: bool) -> float:
        """A whole reply's is the base protocol's. On the stream, the most of:
        - ``hls.trace.store.ESCAPE_GROWTH``, any value the reply carries written again;
        - ``_AUDIO_WIDENING``: the frames' samples are kept in one PCM16 WAV, and the narrowest
          sample a frame may carry is 8-bit;
        - a packet's timestamp (``hls.trace.store.TIMESTAMP_JSON_BYTES``) per smallest frame, its
          JSON alone;
        - under ``return_logprobs``, the report: a position that is a one-digit served
          log-probability and k alternatives each ``[0,0]``, k from none to ``top_logprobs``
          (the decoder keeps fewer than asked, which the gate grades), is kept as a
          ``TokenLogprob``'s dict whose alternatives are ``TopLogprob`` dicts
          (``CodecPosition.as_dict``), each with its separator. One alternative at a
          log-probability of zero is the densest a position's probability mass allows."""
        from hls.trace.store import ESCAPE_GROWTH, TIMESTAMP_JSON_BYTES
        if not stream:
            return super().trace_growth(stream=False)
        compact = (",", ":")
        packet = TIMESTAMP_JSON_BYTES / len(json.dumps(self.smallest_frame(), separators=compact))
        growth = [float(ESCAPE_GROWTH), float(_AUDIO_WIDENING), packet]
        if self.return_logprobs:
            for k in range(0, (self.top_logprobs or 0) + 1):
                kept = CodecPosition(0, 0.0, [0] * k, [0.0] * k).as_dict()
                growth.append((len(json.dumps(kept, separators=compact)) + 1)
                              / (len(json.dumps([0, [[0, 0]] * k], separators=compact)) + 1))
        return max(growth)

    def encode_request(self, env: Envelope, *, stream: bool = False) -> dict:
        text = env.first_of("text")
        body: dict = {"model": self.model, "input": text.text if text else "",
                      "voice": self.voice, "return_token_ids": self.return_token_ids}
        if self.return_codec_codes:
            body["return_codec_codes"] = True
        if self.return_logprobs:
            body["return_logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        if stream:
            body["stream"] = True
        return body

    def _fold(self) -> _TTSCodesFold:
        """A fold over this protocol's codes frame, at the shape it pins."""
        return _TTSCodesFold(
            groups=self.codec_groups, codebook_size=self.codec_codebook_size,
            top_logprobs=self.top_logprobs if self.return_logprobs else None)

    def decode_response(self, data: dict, request_id: str) -> Envelope:
        import wave

        from hls.core.audio import duration_sec_of_b64
        if not isinstance(data, dict) or "audio" not in data:
            raise ArtifactProtocolError(f"TTS response missing 'audio': {data!r}")
        b64 = data.get("audio") or ""
        codes = self._fold()
        if self.return_codec_codes:
            if "codec_codes" in data:
                codes.note(data)
            else:
                codes.defect = "response carried no codec_codes"
        try:
            payload = AudioPayload(
                audio_b64=b64, sample_rate=int(data.get("sample_rate", 16_000)),
                tokens=data.get("token_ids"),
                codec_codes=codes.codes, codec_logprobs=codes.logprobs,
                codec_codes_defect=codes.defect,
                duration_sec=duration_sec_of_b64(b64) if b64 else 0.0)
        except (ValidationError, TypeError, ValueError, OverflowError, EOFError,
                wave.Error) as e:
            # Reading the header is contestant-driven: audio that is not valid base64
            # raises binascii.Error (a ValueError), audio that is not a RIFF/WAV raises
            # wave.Error, and a truncated one raises EOFError. None of those is a
            # ValidationError, so without them here a single bad reply escapes the fault
            # taxonomy instead of failing its own request. The same goes for a sample
            # rate of 1e400, which JSON reads as infinity and `int` refuses with
            # OverflowError.
            raise ArtifactProtocolError(f"malformed TTS payload: {e!r}") from e
        return Envelope(request_id=request_id, contents=[payload])

    def decode_stream(self, timed_chunks: list[tuple[float, dict]],
                      request_id: str) -> ArtifactResponse:
        import wave

        from hls.core.audio import concat_wav_b64, sample_rate_of_b64

        frames: list[str] = []
        tokens: list[int] = []
        t_first: float | None = None
        packet_ts: list[float] = []
        codes = self._fold()
        # Only a fallback for a stream that carries no audio at all. The rate that gets
        # recorded comes from the assembled WAV below, not from here: a server that
        # streams 24 kHz audio without repeating `sample_rate` in every chunk would
        # otherwise have 16 kHz written into its trace, which is a lie about its own
        # output that anyone reading the trace later would believe.
        sr = 16_000
        for t, chunk in timed_chunks:
            try:
                audio = chunk.get("audio")
                if audio:
                    if t_first is None:
                        t_first = t
                    frames.append(audio)
                    packet_ts.append(t)
                    sr = int(chunk.get("sample_rate", sr))
                    if self.return_codec_codes:
                        codes.note_audio_after_codes()
                tids = chunk.get("token_ids")
                if tids:
                    tokens.extend(tids)
                if self.return_codec_codes and "codec_codes" in chunk:
                    codes.note(chunk)
            except (AttributeError, TypeError, ValueError, OverflowError) as e:
                # OverflowError: a sample rate of 1e400 reads as infinity, which `int`
                # cannot convert.
                raise ArtifactProtocolError(f"malformed TTS stream chunk: {e!r}") from e
        if self.return_codec_codes and codes.codes is None and codes.defect is None:
            codes.defect = "stream carried no codec_codes frame"
        # Assembling the frames reads contestant bytes, so it fails the same way a bad
        # frame does: one request, not the drive. A frame that is not valid base64 raises
        # binascii.Error (a ValueError), one that is not a RIFF/WAV raises wave.Error, a
        # stream that changes codec rate midway raises ValueError from concat_wav_b64,
        # and a 24-bit frame whose length is not a multiple of three raises ValueError
        # from the reshape that widens it. Outside this guard, a server that legitimately
        # switches rate mid-stream would take down the whole drive.
        try:
            full_b64, dur = concat_wav_b64(frames)
            # The audio is the authority on its own rate. `concat_wav_b64` has already
            # read every frame's header and raised on a mismatch, so this agrees with
            # the samples.
            payload = AudioPayload(audio_b64=full_b64,
                                   sample_rate=sample_rate_of_b64(full_b64) if frames else sr,
                                   tokens=tokens or None, duration_sec=dur,
                                   codec_codes=codes.codes,
                                   codec_logprobs=codes.logprobs,
                                   codec_codes_defect=codes.defect)
        except (ValidationError, TypeError, ValueError, EOFError, wave.Error) as e:
            raise ArtifactProtocolError(f"malformed TTS stream audio: {e!r}") from e
        env = Envelope(request_id=request_id, contents=[payload])
        # The utterance is complete at its last audio frame; what follows is the codes
        # the gate reads, which the contract excludes from latency. Only under the flag
        # that puts them there -- without it the stream ends with the audio and the
        # driver's own stamp is the honest one.
        t_complete = packet_ts[-1] if (self.return_codec_codes and packet_ts) else None
        return ArtifactResponse(output=env, t_first_packet=t_first,
                                packet_timestamps=packet_ts or None,
                                t_complete=t_complete)

    def probe_request(self) -> dict:
        # Built by encode_request for the same reason as the transcription probe: a
        # hand-written body drifts from the published schema, and a server that validates
        # the documented body is then never seen as ready.
        #
        # An envelope with no text payload encodes an empty input, which the server
        # fast-rejects (400) rather than synthesizing; the inference gate
        # (``hls.drive.readiness.wait_inference_ready``) opens on any status below 400 or a
        # bare 400 and keeps polling on any other.
        return self.encode_request(Envelope(request_id="probe", contents=[]), stream=True)

    def body_schema(self) -> dict:
        frame: dict = {"audio": "<b64 wav frame>"}
        # Only ask for codec ids when something will read them. The decoder treats them
        # as optional either way, so advertising them unconditionally sends a contestant
        # after a field no gate scores -- and one a codec need not even expose.
        if self.return_token_ids:
            frame["token_ids"] = [1, 2]
        request = {"model": self.model, "input": "<text>", "voice": self.voice,
                   "stream": True, "return_token_ids": self.return_token_ids}
        schema = {"request": request, "frame": frame}
        if self.return_codec_codes:
            request["return_codec_codes"] = True
            # Example rows of the pinned width, so the brief shows the shape the
            # decoder accepts instead of describing it.
            codes_frame: dict = {
                "codec_codes": [[0] * self.codec_groups,
                                [self.codec_codebook_size - 1] * self.codec_groups]}
            if self.return_logprobs:
                request["return_logprobs"] = True
                request["top_logprobs"] = self.top_logprobs
                # One frame, and two alternatives where a reply sends ``top_logprobs``
                # of them. At full width this example runs to tens of kilobytes inside
                # a brief a contestant reads under a token budget, and what it is for
                # is the nesting. Both abbreviations are stated in the prose beside it,
                # and the example still parses, so the test that decodes it keeps the
                # shape honest.
                codes_frame["codec_codes"] = [[0] * self.codec_groups]
                codes_frame["codec_report"] = [
                    [_EXAMPLE_REPORT_ENTRY] * self.codec_groups]
                codes_frame["eos_code"] = self.codec_eos_id
                codes_frame["eos_report"] = _example_eos_entry(self.codec_eos_id)
            schema["codes_frame"] = codes_frame
        return schema

    def brief_section(self, *, path: str, streaming: bool = False) -> str:
        schema = self.body_schema()
        codes_frame = schema.get("codes_frame")
        return jinja2.Template(_TTS_BRIEF, trim_blocks=True, lstrip_blocks=True).render(
            path=path,
            request=json.dumps(schema["request"], indent=2),
            frame=json.dumps(schema["frame"], indent=2),
            return_token_ids=self.return_token_ids,
            return_codec_codes=self.return_codec_codes,
            return_codec_report=self.return_logprobs,
            top_logprobs=self.top_logprobs,
            codec_groups=self.codec_groups,
            codec_codebook_size=self.codec_codebook_size,
            codec_eos_id=self.codec_eos_id,
            voice=self.voice,
            codes_frame=json.dumps(codes_frame) if codes_frame else "")
