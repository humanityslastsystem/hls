"""Carry Qwen3-TTS codec codes out of the vLLM-Omni stage-0 worker, for the baseline.

vLLM-Omni builds the [T, 16] code matrix in its talker stage and ships it over a
shared-memory connector to the codec stage, but publishes no way to read it from
outside the server process: the speech request has no flag that asks for codes, the
response and its SSE events carry audio only, and the serving layer drops any non-audio
stage output. The arena's wire contract requires the codes, so the baseline would be
ungradable without a way out.

This is that way out, and it patches nothing. vLLM-Omni lets a stage name its own
producer hook (``custom_process_next_stage_input_func``), which it imports by dotted
path and calls for every chunk with the transfer manager, the stage's multimodal output
and the request. ``spooling_hook`` below is that hook: it calls the packaged
``talker2code2wav_async_chunk`` and returns exactly what it returns, so the pipeline
behaves as it would unhooked, and then copies the finished request's accumulated codes
to a file the adapter in the same container reads back.

Three things this file is careful about.

The signature is the packaged function's, argument for argument. The connector decides
whether to pass ``new_token_ids`` by inspecting the hook, treating a ``**kwargs`` as
"yes", and the packaged function does not take it. A wrapper that accepted ``**kwargs``
would be handed an argument it could not forward.

Nothing it does can break a request. The connector catches a hook exception, logs it,
and proceeds with no payload, which for the packaged hook means the codec stage never
receives its frames. So the packaged call is made first and its result returned
whatever happens afterwards, and every line of the spooling is inside its own guard: a
side channel that fails must cost the codes, never the audio.

The spool is on the container's own filesystem, which under a sealed container is a
small tmpfs shared with everything else in there. Files are therefore written one per
request and deleted by the reader as soon as it has them, so what is live at any moment
is bounded by concurrency rather than by the length of the drive. ``purge`` clears
anything a previous run left behind.

It also records what the engine actually decoded under. vLLM-Omni validates a stage's
``default_sampling_params`` when it builds the SamplingParams, so a typo there stops the
server; ``subtalker_sampling_params`` is a bare dict that nothing validates, so a
misspelled key there silently restores sampling on fifteen of the sixteen code groups.
On its first call the hook writes both, read off the objects the engine itself uses, so
the pin can be checked against an observation instead of against the file the adapter wrote.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

#: Environment variable naming the spool directory, set by the adapter on the vLLM
#: child process. Both sides read it, so neither holds the path as a literal.
SPOOL_DIR_ENV = "HLS_TTS_CODES_SPOOL"

#: Where the spool lives when the variable is unset.
DEFAULT_SPOOL_DIR = "/tmp/hls-tts-codes"

#: Key the adapter puts in the request's ``extra_params``, which vLLM-Omni copies into
#: stage 0's ``SamplingParams.extra_args``. It is how a file in the spool is matched to
#: the HTTP request that produced it: the engine's own request id is not reported to an
#: HTTP caller anywhere, so the caller has to supply a correlator of its own.
REQUEST_KEY_FIELD = "hls_codes_key"

#: File the observed decode is written to, once per server.
DECODE_FILE = "_effective_decode.json"

#: Codebook groups per frame and ids per group for this checkpoint, used only to reject
#: an obviously wrong matrix before it is written. The wire contract checks the same
#: two numbers on the harness side, which is what actually grades them.
CODEC_GROUPS = 16
CODEC_CODEBOOK_SIZE = 2048

#: Set once the observed decode has been written, so the read costs one call and not one
#: per chunk of every request.
_decode_recorded = False


def spool_dir() -> Path:
    """The spool directory, from the environment or the default."""
    return Path(os.environ.get(SPOOL_DIR_ENV) or DEFAULT_SPOOL_DIR)


def purge() -> None:
    """Empty the spool, creating it if it does not exist.

    Called by the adapter at startup. A file left by a previous server would otherwise
    be read as this server's answer for whichever request happened to mint the same
    key, and a spool that only grows is how a container's tmpfs fills."""
    d = spool_dir()
    d.mkdir(parents=True, exist_ok=True)
    for path in d.iterdir():
        if path.is_file():
            path.unlink(missing_ok=True)


def _write_atomic(path: Path, payload: Any) -> None:
    """Write JSON so a reader sees the whole file or no file.

    The reader is another process polling for this path to appear. Writing in place
    would let it read a half-serialized matrix and call the codes malformed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _request_key(request: Any) -> str | None:
    """The correlator the adapter put on this request, or None if it carries none."""
    params = getattr(request, "sampling_params", None)
    extra = getattr(params, "extra_args", None) or {}
    if not isinstance(extra, dict):
        return None
    key = extra.get(REQUEST_KEY_FIELD)
    return key if isinstance(key, str) and key else None


def _model_config(transfer_manager: Any) -> tuple[Any, str]:
    """The stage's model config and the attribute path it was found on.

    Tried in several places because the object the hook is handed is an internal one
    whose shape is not part of any published interface. The path is reported alongside
    the values so a reader can tell a real observation from a miss."""
    for path in ("vllm_config.model_config", "model_config", "config"):
        obj: Any = transfer_manager
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "subtalker_sampling_params"):
            return obj, path
    return None, ""


def _sampling_params_of(request: Any) -> dict:
    """The fields of stage 0's per-request sampling params that decide the talker's
    choice of code 0. Read off the request the sampler will actually run."""
    params = getattr(request, "sampling_params", None)
    fields = ("temperature", "top_p", "top_k", "repetition_penalty", "min_tokens",
              "max_tokens", "seed")
    return {f: getattr(params, f, None) for f in fields}


def _record_decode(transfer_manager: Any, request: Any) -> None:
    """Write what this engine is decoding under, once per server."""
    global _decode_recorded
    if _decode_recorded:
        return
    config, path = _model_config(transfer_manager)
    _write_atomic(spool_dir() / DECODE_FILE, {
        # Where the sub-talker's settings were read from. Empty means they were not
        # found at all, which is itself the answer: the pin cannot be confirmed.
        "config_attr_path": path,
        "subtalker_sampling_params":
            getattr(config, "subtalker_sampling_params", None) if config else None,
        "talker_sampling_params": _sampling_params_of(request),
        "transfer_manager_type": type(transfer_manager).__name__,
    })
    _decode_recorded = True


def _rows_of(transfer_manager: Any, request_id: str) -> list | None:
    """This request's accumulated codec frames, or None when there are none.

    Read with ``.get`` rather than by subscript: the engine holds these in a
    ``defaultdict``, so a subscript would create an empty entry for a request that
    never had one and leave it behind."""
    table = getattr(transfer_manager, "code_prompt_token_ids", None)
    if table is None:
        return None
    rows = table.get(request_id)
    return list(rows) if rows else None


def _well_formed(rows: Any) -> bool:
    """Whether the matrix is the shape this checkpoint's codec produces.

    A cheap reject so nothing obviously wrong is written. The harness checks the same
    shape on the wire and is what actually grades it; this only keeps a malformed
    matrix from being mistaken for a served one."""
    if not isinstance(rows, list) or not rows:
        return False
    return all(isinstance(r, list) and len(r) == CODEC_GROUPS
               and all(isinstance(c, int) and 0 <= c < CODEC_CODEBOOK_SIZE for c in r)
               for r in rows)


def _spool(transfer_manager: Any, request: Any, finished: bool) -> None:
    """Copy a finished request's codes into the spool under the adapter's key."""
    _record_decode(transfer_manager, request)
    if not finished:
        return
    key = _request_key(request)
    if key is None:
        return          # not one of the adapter's; nothing to correlate a file with
    rows = _rows_of(transfer_manager, getattr(request, "external_req_id", None))
    if not _well_formed(rows):
        return          # the adapter sends no codes frame, and the harness says so
    _write_atomic(spool_dir() / f"{key}.json", rows)


def spooling_hook(
    transfer_manager: Any,
    multimodal_output: Any,
    request: Any,
    is_finished: bool = False,
) -> Any:
    """vLLM-Omni's packaged Qwen3-TTS producer hook, plus a copy of the codes.

    Named in the deploy config as this module's dotted path. The packaged function runs
    first and its return value is returned unchanged, so the talker-to-codec pipeline is
    the stock one; the spooling that follows cannot raise into the connector."""
    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_async_chunk)

    payload = talker2code2wav_async_chunk(
        transfer_manager, multimodal_output, request, is_finished)
    try:
        _spool(transfer_manager, request,
               bool(is_finished or _is_finished(request)))
    except Exception as exc:   # noqa: BLE001 - the audio matters more than the codes
        print(f"[codes-spool] failed to spool codes: {exc!r}", flush=True)
    return payload


def _is_finished(request: Any) -> bool:
    """Whether the request is done, asked the way the packaged hook asks."""
    fn = getattr(request, "is_finished", None)
    try:
        return bool(fn()) if callable(fn) else False
    except Exception:   # noqa: BLE001 - an engine-internal call, never the hook's failure
        return False
