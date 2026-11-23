"""L0 -- Payloads + Envelope.

A Payload is one block of data of a given modality. An Envelope is what flows on
the wire: a request_id, a list of Payloads, opaque metadata.

Modalities are an open set: new modalities register via `@register_payload("name")`.
The registry is wired into pydantic by a custom core schema on the `Payload` base:
a field declared as `Payload` (e.g. `list[Payload]`) validates JSON by
dispatching on `modality` to the concrete class, and serializes the concrete
instance's own fields. No per-field `SerializeAsAny`, no per-container validator.
"""

from __future__ import annotations
from array import array
from typing import Any, ClassVar, NamedTuple
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import core_schema


PAYLOAD_REGISTRY: dict[str, type["Payload"]] = {}

#: The range of token ids a reply may carry, that of a signed 64-bit integer. The
#: harness packs served and reported ids into int64 arrays (``CodecPosition``'s
#: alternatives, the report gate's rows), which raise OverflowError outside it, so the
#: wire refuses such an id as the reply's own fault.
MIN_TOKEN_ID = -2**63
MAX_TOKEN_ID = 2**63 - 1


def register_payload(modality: str):
    """Class decorator. Registers a Payload subclass under a modality string."""
    def wrap(cls: type[Payload]) -> type[Payload]:
        if modality in PAYLOAD_REGISTRY:
            raise ValueError(f"modality {modality!r} already registered")
        PAYLOAD_REGISTRY[modality] = cls
        cls.MODALITY = modality
        return cls
    return wrap


class Payload(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    MODALITY: ClassVar[str] = ""
    modality: str

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Concrete subclasses use pydantic's normal model schema. Only the OPEN
        # base dispatches through the registry.
        if cls.__name__ != "Payload" or cls.__module__ != __name__:
            return handler(source_type)

        def _validate(value: Any) -> "Payload":
            if isinstance(value, Payload):
                return value
            if isinstance(value, dict):
                modality = value.get("modality")
                klass = PAYLOAD_REGISTRY.get(modality)
                if klass is None:
                    raise ValueError(f"unknown payload modality {modality!r}")
                return klass.model_validate(value)
            raise ValueError(f"cannot validate Payload from {type(value)!r}")

        def _serialize(value: "Payload", info: core_schema.SerializationInfo) -> Any:
            return value.model_dump(mode="json" if info.mode_is_json() else "python")

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize, info_arg=True, when_used="always"
            ),
        )


class TopLogprob(BaseModel):
    """One alternative token at a position. `token_id` is HLS's mandatory
    integer-ID extension (OpenAI ships only the decoded `token` string).

    `token` and `bytes` are the alternative's decoded text and its utf-8 encoding.
    Both are optional and the wire parser drops them: a reply reports up to 21
    alternatives for every token it generated and the whole trace is written to
    disk, so a served text per alternative costs gigabytes for a field nothing
    reads. Verification keys on `token_id` and `logprob`.
    """
    model_config = ConfigDict(frozen=True)
    token: str | None = None
    token_id: int | None = None
    logprob: float
    bytes: list[int] | None = None


class TokenLogprob(BaseModel):
    """Per-token logprob == OpenAI ChatCompletionTokenLogprob + integer `token_id`.

    `token` is optional for the same reason it is on `TopLogprob`, and for one more:
    not every modality's unit of generation has decoded text at all. An audio codec's
    frame is a row of codebook ids with no string behind it, so a report over those
    positions leaves the field None rather than inventing an empty string for it.
    Nothing verification does reads it.
    """
    model_config = ConfigDict(frozen=True)
    token: str | None = None
    token_id: int | None = None
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] | None = None


class CodecAlternative(NamedTuple):
    """One alternative at a codec position, as a report verifier reads it: its id and
    log-probability, the two fields of a ``TopLogprob`` anything reads."""
    token_id: int
    logprob: float


class CodecPosition:
    """One reported codec position, held as two packed arrays.

    A ``TokenLogprob`` with its alternatives as ``TopLogprob`` models costs about 450
    bytes per alternative in memory. A speech reply reports every code of every frame,
    tens of thousands of alternatives, so models of a drive's replies would fill the
    driver's memory before the trace was even serialized. This keeps the same numbers in
    two arrays, a few bytes per alternative, and reads the way the report verifier reads a
    ``TokenLogprob``:
    ``token_id``, ``logprob`` and ``top_logprobs`` (built on demand as
    ``CodecAlternative`` pairs).

    It validates from and serializes to exactly a ``TokenLogprob``'s dict, so a trace
    reads and writes the same as one held as ``TokenLogprob`` models."""

    __slots__ = ("token_id", "logprob", "_ids", "_lps")

    #: Decoded text and bytes: a codec position has none (see ``TokenLogprob``).
    token = None
    bytes = None

    def __init__(self, token_id: int | None, logprob: float,
                 ids: "array | list[int]", lps: "array | list[float]") -> None:
        self.token_id = token_id
        self.logprob = float(logprob)
        self._ids = ids if isinstance(ids, array) else array("q", ids)
        self._lps = lps if isinstance(lps, array) else array("d", lps)
        if len(self._ids) != len(self._lps):
            raise ValueError(f"{len(self._ids)} alternative ids for "
                             f"{len(self._lps)} log-probabilities")

    @property
    def top_logprobs(self) -> list[CodecAlternative]:
        """The alternatives in reported order, as (id, log-probability) pairs."""
        return [CodecAlternative(i, lp) for i, lp in zip(self._ids, self._lps)]

    @classmethod
    def from_entry(cls, entry: Any) -> "CodecPosition":
        """From a ``TokenLogprob``, a ``CodecPosition``, or a ``TokenLogprob``'s dict."""
        if isinstance(entry, CodecPosition):
            return entry
        if isinstance(entry, dict):
            alts = entry.get("top_logprobs") or []
            return cls(entry.get("token_id"), entry["logprob"],
                       [a["token_id"] for a in alts], [a["logprob"] for a in alts])
        alts = entry.top_logprobs or []
        return cls(entry.token_id, entry.logprob,
                   [a.token_id for a in alts], [a.logprob for a in alts])

    def as_dict(self) -> dict:
        """A ``TokenLogprob``'s dict for the same position, key for key."""
        return {"token": None, "token_id": self.token_id, "logprob": self.logprob,
                "bytes": None,
                "top_logprobs": [{"token": None, "token_id": i, "logprob": lp,
                                  "bytes": None}
                                 for i, lp in zip(self._ids, self._lps)]}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodecPosition):
            return NotImplemented
        return (self.token_id == other.token_id and self.logprob == other.logprob
                and self._ids == other._ids and self._lps == other._lps)

    def __hash__(self) -> int:
        return hash((self.token_id, self.logprob, tuple(self._ids)))

    def __repr__(self) -> str:
        return (f"CodecPosition(token_id={self.token_id!r}, logprob={self.logprob!r}, "
                f"{len(self._ids)} alternatives)")

    def __getstate__(self):
        return (self.token_id, self.logprob, self._ids, self._lps)

    def __setstate__(self, state) -> None:
        self.token_id, self.logprob, self._ids, self._lps = state

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def _validate(value: Any) -> "CodecPosition":
            if isinstance(value, (CodecPosition, dict, TokenLogprob)):
                return cls.from_entry(value)
            raise ValueError(f"cannot read a codec position from {type(value)!r}")

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.as_dict(), when_used="always"))


@register_payload("text")
class TextPayload(Payload):
    modality: str = "text"
    text: str
    tokens: list[int] | None = None       # populated on responses when contract demands it
    logprobs: list[TokenLogprob] | None = None  # per emitted token; populated when requested
    finish_reason: str | None = None


@register_payload("audio")
class AudioPayload(Payload):
    """One block of audio (modality "audio").

    Audio rides inside the JSON Envelope as a base64 PCM16 WAV (mono) plus its
    sample rate, so it flows through the existing Envelope codec with no new
    transport -- the codec lives in ``hls.core.audio``. ``tokens`` carries the
    codec/decoder token ids on outputs (e.g. an ASR server's emitted ids, or TTS
    audio tokens); ``duration_sec`` is a convenience read off the WAV header.
    Inputs (the dataset's clips) leave ``tokens`` None."""

    modality: str = "audio"
    audio_b64: str                          # base64 of a PCM16 mono WAV
    sample_rate: int = 16_000
    duration_sec: float | None = None
    tokens: list[int] | None = None         # decoder/codec token ids on outputs
    finish_reason: str | None = None
    #: The residual-codec codes a neural TTS server decoded this audio from: one row
    #: per codec frame, one id per codebook group, in generation order, with no
    #: end-of-speech row. What ``TextPayload.tokens`` is for text, for a modality whose
    #: unit of generation is a frame of parallel codebooks rather than a single id, so
    #: a position-level gate can grade the audio the same way it grades a reply. The
    #: wire protocol fills it from the reply's codes frame and pins the row width and
    #: the id range it accepts. None on inputs, on a server that was never asked for
    #: codes, and on a reply whose codes did not conform (``codec_codes_defect`` then
    #: says so).
    codec_codes: list[list[int]] | None = None
    #: The server's own report of the distribution it sampled each of those codes
    #: from: per position, the log-probability it gave the code it served and the most
    #: probable alternatives with their ids, which is what a sampling gate reads
    #: (``hls.verify.report``). What ``TextPayload.logprobs`` is for text.
    #:
    #: Flat, in the one order a consumer may assume, of length ``16 * T + 1`` for
    #: ``T = len(codec_codes)`` at a row width of 16:
    #:
    #:   * index ``f * 16 + g`` is the position whose served code is
    #:     ``codec_codes[f][g]``, so frames run outermost and codebook groups within
    #:     them;
    #:   * the final index is the end-of-speech decision, the codebook-0 position
    #:     after the last frame. It has no row in ``codec_codes``, so the id served
    #:     there is its entry's own ``token_id`` and nothing else carries it.
    #:
    #: Each entry's ``token_id`` is the id served at that position and ``logprob`` is
    #: the server's report for it. None whenever ``codec_codes`` is, plus on a reply
    #: whose report alone did not conform; ``codec_codes_defect`` says which.
    #:
    #: Held as ``CodecPosition``: the same numbers as a ``TokenLogprob`` per position,
    #: packed, because a speech reply reports tens of thousands of alternatives and
    #: models of them would fill the driver's memory. On the wire and on disk each entry
    #: is a ``TokenLogprob``'s dict.
    codec_logprobs: list[CodecPosition] | None = None
    #: Why the two fields above are None on a reply that was supposed to carry them:
    #: the codes frame was absent, duplicated, late, or malformed, or the report in it
    #: was. Carried rather than raised so one non-conformant reply fails its own
    #: correctness gate instead of aborting the drive, and so the verdict can say what
    #: the server actually did. None when the reply conformed, and on every payload
    #: none of this was asked of.
    #:
    #: The two fields fail independently, because a gate that re-decodes the codes and
    #: compares waveforms needs no report at all: a bad report leaves ``codec_codes``
    #: populated and only ``codec_logprobs`` None. Bad codes null both, since without
    #: the served ids there is no reading the report's positions.
    codec_codes_defect: str | None = None


@register_payload("image")
class ImagePayload(Payload):
    """One still image (modality "image").

    Like audio, an image rides inside the JSON Envelope as a base64 blob of its
    *encoded* file bytes (PNG/JPEG) plus the MIME type, so it flows through the
    existing Envelope codec with no new transport -- an omni chat protocol renders
    it to an OpenAI ``image_url`` data URI. ``width``/``height`` are convenience
    metadata. Images are inputs only (no token/finish fields)."""

    modality: str = "image"
    image_b64: str                          # base64 of the encoded image file bytes
    mime_type: str = "image/png"
    width: int | None = None
    height: int | None = None

    def data_uri(self) -> str:
        return f"data:{self.mime_type};base64,{self.image_b64}"


class Envelope(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    request_id: str
    contents: list[Payload] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def modalities(self) -> frozenset[str]:
        return frozenset(p.modality for p in self.contents)

    def first_of(self, modality: str) -> Payload | None:
        for p in self.contents:
            if p.modality == modality:
                return p
        return None
