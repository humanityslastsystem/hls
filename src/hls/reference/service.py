"""Reference-scoring service: the grader's teacher-forced reference, on demand, in a build.

Runs as a harness role on the task's baseline image beside a contestant's build sandbox, on the same
GPUs. It builds the task's pinned vLLM reference exactly as the grader's
``hls.reference.engines.VLLMEngine._load`` does (same weights, dtype, memory fraction,
context length, tensor parallelism, prefix caching off), plus vLLM sleep mode, and puts it
to sleep. While asleep it holds almost no GPU memory, so the contestant's own server can
use the cards. A score request wakes it, teacher-forces ``prompt + continuation`` through
vLLM's prompt-logprob path with the grader's sampling parameters, and returns the
reference's top alternatives at every continuation position. It goes back to sleep after
an idle period, or on request.

It binds its socket the moment it starts and answers ``/status`` with state "starting"
until the engine is built and asleep, so the build does not wait for it: the agent starts
at once, and a score request before then is refused with 503 and the state. Its memory is
the node's, shared with the agent's session: when a load, a wake or a sleep is short of GPU
or host memory, it says so with HTTP 507, which memory and what it knows of by how much, and
stays up; a load short of memory leaves it "unloaded" until ``POST /wake`` loads it again.
SIGTERM or SIGINT closes the socket, shuts the engine down and exits.

The only interface is HTTP on a unix domain socket, whose directory the harness binds into
the build sandbox. There is no TCP listener.

    GET  /status   state, timings, per-device GPU memory
    POST /score    {"prompt_token_ids": [...], "continuation_token_ids": [...],
                    "prompt_positions": [...]}  (the last is optional)
    POST /sleep    offload the weights to host memory and free the KV cache
    POST /wake     restore them

Standalone: the harness mounts this one file into the service's container, so it imports
nothing outside the standard library, vLLM and the NVML bindings vLLM depends on. The
engine is imported lazily, so the service logic is testable with a stub engine and no GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import logging
import math
import os
import signal
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from typing import Callable, NoReturn, Protocol

log = logging.getLogger("hls-reference-service")

#: Largest request body accepted, in bytes. A million token ids as JSON is under 10 MB.
MAX_BODY_BYTES = 64 * 1024 * 1024

#: How often the idle watcher checks whether the engine should go back to sleep.
IDLE_POLL_SEC = 5.0

#: Headroom added to the measured wake footprint before deciding a wake fits. Covers the
#: allocator's rounding and the few buffers vLLM allocates when it resumes.
WAKE_MARGIN_BYTES = 1 << 30

#: Seconds the engine gets to shut its worker processes down after SIGTERM before the
#: process exits regardless. The build stage kills the role 10 s after its SIGTERM, so the
#: process exits within that.
SHUTDOWN_GRACE_SEC = 5.0

#: The phase a starting service reports in ``/status`` as "phase": building the engine and
#: taking its first sleep. A service that has left "starting" has no phase.
LOADING_PHASE = "loading"


class Engine(Protocol):
    """What the service needs from a reference engine."""

    def rows(self, token_ids: list[int]) -> list:
        """vLLM prompt-logprob rows for ``token_ids``: one per position, ``None`` at 0,
        each a mapping from token id to an object with a ``logprob`` attribute."""

    def sleep(self) -> None:
        """Free the engine's GPU memory, keeping what a wake needs to restore it."""

    def wake(self) -> None:
        """Restore the engine after ``sleep``."""

    def shutdown(self) -> None:
        """End the engine's worker processes."""


class VLLMEngine:
    """The grader's reference engine with sleep mode enabled.

    The constructor arguments are those ``VLLMEngine._load`` passes, with three additions,
    none of which enters a kernel: ``max_logprobs`` (a request validation bound, which
    must admit ``top_logprobs``), ``enable_sleep_mode`` (allocates weights and KV cache
    through vLLM's sleep-capable allocator), and no ``worker_extension_cls`` (the grader's
    only adds callable method names to the workers).
    """

    def __init__(self, *, weights: str, dtype: str, gpu_memory_utilization: float,
                 max_model_len: int | None, enforce_eager: bool, tensor_parallel_size: int,
                 batch_invariant: bool, top_logprobs: int, sleep_level: int) -> None:
        #: The checkpoint's vocabulary width, which bounds the token ids a request may carry.
        self.vocab_size = vocab_size_of(weights)
        if batch_invariant:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
        from vllm import LLM, SamplingParams  # type: ignore[import-untyped]

        self._llm = LLM(model=weights, dtype=dtype,
                        gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=max_model_len,
                        enforce_eager=enforce_eager,
                        tensor_parallel_size=tensor_parallel_size,
                        enable_prefix_caching=False,
                        max_logprobs=top_logprobs,
                        enable_sleep_mode=True)
        # The grader's call (VLLMEngine._score_one) with its report width replaced by
        # vLLM's own top-k reduction of the same logits.
        self._params = SamplingParams(temperature=0.0, max_tokens=1,
                                      prompt_logprobs=top_logprobs, detokenize=False)
        self._sleep_level = sleep_level
        #: The context length the engine settled on, which is the one given when given.
        self.max_model_len = int(self._llm.llm_engine.vllm_config.model_config.max_model_len)

    def rows(self, token_ids: list[int]) -> list:
        outs = self._llm.generate({"prompt_token_ids": token_ids},
                                  sampling_params=self._params, use_tqdm=False)
        rows = outs[0].prompt_logprobs
        if rows is None or len(rows) != len(token_ids):
            raise RuntimeError(
                f"vLLM returned {0 if rows is None else len(rows)} prompt-logprob rows "
                f"for {len(token_ids)} tokens; with prefix caching off it returns one "
                f"per position")
        return rows

    def sleep(self) -> None:
        self._llm.sleep(level=self._sleep_level)

    def wake(self) -> None:
        self._llm.wake_up()

    def shutdown(self) -> None:
        # The engine-core client owns the engine process, which owns the TP workers.
        self._llm.llm_engine.engine_core.shutdown()


def position_record(row, token_id: int, top_logprobs: int) -> dict:
    """One position: the token occupying it, its log-probability under the reference,
    and the reference's ``top_logprobs`` most probable ids, most probable first, ties
    broken on the id.

    vLLM adds the occupying token to a prompt-logprob row when it falls outside the top
    k, so taking the first ``top_logprobs`` of the ranked row drops that extra entry
    exactly when it does not belong among them. A row that does not name the occupying
    token belongs to some other position, which is an error rather than a gap.
    """
    entry = row.get(token_id)
    if entry is None:
        raise RuntimeError(f"the reference's row for a position holding token {token_id} "
                           f"does not name that token")
    ranked = sorted(row.items(), key=lambda kv: (-kv[1].logprob, kv[0]))
    return {"token_id": int(token_id), "logprob": float(entry.logprob),
            f"top{top_logprobs}": [[int(t), float(e.logprob)]
                                   for t, e in ranked[:top_logprobs]]}


class RequestError(Exception):
    """A request the service refuses, with the HTTP status and body to answer with."""

    def __init__(self, status: int, body: dict) -> None:
        super().__init__(body.get("error", ""))
        self.status = status
        self.body = body


def _id_list(value, field: str, vocab_size: int) -> list[int]:
    if not isinstance(value, list):
        raise RequestError(400, {"error": f"{field} must be a list of token ids"})
    for tid in value:
        if isinstance(tid, bool) or not isinstance(tid, int):
            raise RequestError(400, {"error": f"{field} holds a non-integer: {tid!r}"})
        if not 0 <= tid < vocab_size:
            raise RequestError(400, {"error": f"{field} holds token id {tid}, outside "
                                              f"the vocabulary [0, {vocab_size})"})
    return value


def parse_score_request(body: dict, *, vocab_size: int, max_model_len: int) -> tuple[
        list[int], list[int], list[int]]:
    """``(prompt ids, continuation ids, prompt positions)`` of a /score body.

    Ids are range-checked before the engine sees them: an id outside the embedding table
    fires a device-side assert that poisons the CUDA context for the rest of the run.
    """
    if not isinstance(body, dict):
        raise RequestError(400, {"error": "the body must be a JSON object"})
    unknown = set(body) - {"prompt_token_ids", "continuation_token_ids", "prompt_positions"}
    if unknown:
        raise RequestError(400, {"error": f"unknown fields: {sorted(unknown)}"})
    prompt = _id_list(body.get("prompt_token_ids"), "prompt_token_ids", vocab_size)
    cont = _id_list(body.get("continuation_token_ids", []), "continuation_token_ids",
                    vocab_size)
    positions = body.get("prompt_positions", [])
    if not isinstance(positions, list) or any(
            isinstance(p, bool) or not isinstance(p, int) for p in positions):
        raise RequestError(400, {"error": "prompt_positions must be a list of integers"})
    if not prompt:
        raise RequestError(400, {"error": "prompt_token_ids is empty"})
    if not cont and not positions:
        raise RequestError(400, {"error": "nothing to score: continuation_token_ids and "
                                          "prompt_positions are both empty"})
    bad = [p for p in positions if not 1 <= p < len(prompt)]
    if bad:
        raise RequestError(400, {"error": f"prompt_positions must lie in [1, "
                                          f"{len(prompt)}); got {bad[:10]}"})
    # One record is returned per entry, so the list is held to the number of positions
    # the prompt has: a longer one only repeats, and its reply grows without the prompt.
    if len(positions) > len(prompt) - 1:
        raise RequestError(400, {"error": f"prompt_positions has {len(positions)} entries; "
                                          f"at most {len(prompt) - 1} fit a prompt of "
                                          f"{len(prompt)} tokens"})
    total = len(prompt) + len(cont)
    if total > max_model_len - 1:
        raise RequestError(400, {"error": f"{total} tokens exceed the limit of "
                                          f"{max_model_len - 1}"})
    return prompt, cont, positions


def nvml_memory() -> list[dict]:
    """Per-device ``{"index", "free_bytes", "total_bytes"}`` for the devices this process
    was given, read through NVML so the service process never opens a CUDA context."""
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    parts = [p.strip() for p in visible.split(",") if p.strip()]
    if parts and all(p.isdigit() for p in parts):
        handles = [(int(p), pynvml.nvmlDeviceGetHandleByIndex(int(p))) for p in parts]
    elif parts:
        handles = [(i, pynvml.nvmlDeviceGetHandleByUUID(p)) for i, p in enumerate(parts)]
    else:
        handles = [(i, pynvml.nvmlDeviceGetHandleByIndex(i))
                   for i in range(pynvml.nvmlDeviceGetCount())]
    out = []
    for index, handle in handles:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        out.append({"index": index, "free_bytes": int(info.free),
                    "total_bytes": int(info.total)})
    return out


#: The state of a service whose engine is not built because memory was short when it loaded.
#: ``POST /wake`` loads it again.
UNLOADED_STATE = "unloaded"

#: What an engine's error says when the GPU's memory was short: torch's out-of-memory error, and
#: vLLM's refusals to start or to size its cache on the free memory it found.
_GPU_SHORT = ("outofmemoryerror", "cuda out of memory", "free memory on device",
              "no available memory for the cache blocks")
#: What an engine's error says when host memory was short: an allocation the operating system
#: refused, or a pinned host buffer CUDA could not allocate.
_HOST_SHORT = ("cannot allocate memory", "std::bad_alloc", "cudahostalloc", "pinned memory",
               "pin_memory")


def memory_shortage(exc: BaseException, *, operation: str) -> str | None:
    """Which memory an engine's error says was short, "gpu" or "host", read from the error and
    the errors it was raised from; None when it names no shortage. *operation* is what the
    engine was doing: a CUDA out-of-memory error while it sleeps, when it copies its weights
    into pinned host memory, is the host's; during a load or a wake it is the GPU's."""
    texts: list[str] = []
    seen: set[int] = set()
    err: BaseException | None = exc
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if isinstance(err, MemoryError) or (isinstance(err, OSError) and err.errno == errno.ENOMEM):
            return "host"
        texts.append(f"{type(err).__name__} {err}".lower())
        err = err.__cause__ or err.__context__
    text = " ".join(texts)
    if any(p in text for p in _HOST_SHORT):
        return "host"
    if any(p in text for p in _GPU_SHORT) or "out of memory" in text:
        return "host" if operation == "sleep" else "gpu"
    return None


class Service:
    """The service's state machine: starting, then asleep and awake in turn; unloaded when
    memory was short for its load, until a wake loads it again; failed for good otherwise.

    Every engine call runs under one lock, so a score, a sleep and a wake never overlap.
    ``status`` never takes that lock, so it answers while a long score runs. A shortage of
    memory is an answer, never a crash: a load, a sleep or a wake that the engine fails for lack
    of memory, GPU or host, leaves the service usable and returns HTTP 507 saying which memory was
    short and what the service knows of by how much (``memory_error``).
    """

    def __init__(self, *, engine_factory: Callable[[], Engine], vocab_size: int | None,
                 max_model_len: int | None, top_logprobs: int, idle_sleep_sec: float,
                 gpu_memory: Callable[[], list[dict]], config: dict,
                 load_fraction: float | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._engine_factory = engine_factory
        self._engine: Engine | None = None
        #: Token ids must lie below this. None until the engine is built when the caller
        #: left it to be read from the checkpoint.
        self._vocab_size = vocab_size
        #: Longest prompt + continuation accepted is one less than this. None until the
        #: engine is built when the caller left the context length to vLLM.
        self._max_model_len = max_model_len
        self._top_logprobs = top_logprobs
        self._idle_sleep_sec = idle_sleep_sec
        self._gpu_memory = gpu_memory
        #: The fraction of each card's memory the engine claims (vLLM's
        #: ``gpu_memory_utilization``), which vLLM requires free on every card when it starts; a
        #: load checks it first. None skips the check.
        self._load_fraction = load_fraction
        #: Engine settings echoed by /status, so a caller can see what it is scored by.
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        #: One of starting, asleep, awake, waking, sleeping, unloaded, failed.
        self.state = "starting"
        #: What a starting service is doing, LOADING_PHASE; None once it has left
        #: "starting".
        self.phase: str | None = LOADING_PHASE
        #: Why the service failed, when it has.
        self.error: str | None = None
        #: The last shortage of memory a load, a sleep or a wake met, while it stands: which
        #: memory, during what, the engine's own error and, for the GPU, each card's free and
        #: needed bytes; None when there is none.
        self.memory_error: dict | None = None
        self.started_at = clock()
        #: When the current load started; the starting state's clock.
        self.load_started_at = self.started_at
        self.startup_sec: float | None = None
        self.last_used: float | None = None
        self.last_wake_sec: float | None = None
        self.last_sleep_sec: float | None = None
        #: Per-device bytes the engine held awake and asleep, measured by the first
        #: sleep, when nothing else was on the cards. Their difference is what a wake
        #: has to find free.
        self.awake_used_bytes: list[int] | None = None
        self.asleep_used_bytes: list[int] | None = None
        self.wake_need_bytes: list[int] | None = None

    # -- lifecycle -------------------------------------------------------------------

    def load(self) -> None:
        """Build the engine, measure its footprint, and put it to sleep. When memory is short
        for it, stay unloaded with the shortage, or awake when only the first sleep was short of
        host memory; fail for good on any other error."""
        try:
            short = self._gpu_short_for_load()
            if short is not None:
                self._unload(short)
                return
            try:
                engine = self._engine_factory()
            except Exception as exc:  # noqa: BLE001 -- a shortage is answered, anything else fails
                kind = memory_shortage(exc, operation="load")
                if kind is None:
                    raise
                self._unload({"memory": kind, "operation": "load", "error": repr(exc)})
                return
            self._engine = engine
            if self._vocab_size is None:
                self._vocab_size = int(getattr(engine, "vocab_size"))
            if self._max_model_len is None:
                self._max_model_len = int(getattr(engine, "max_model_len"))
                self._config["max_model_len"] = self._max_model_len
            with self._lock:
                try:
                    self._sleep_locked()
                except RequestError as exc:
                    if exc.status != 507:
                        raise
                    # Only the copy of the weights to host memory was short: the engine is built
                    # and awake, and a later sleep may succeed.
                    log.warning("the first sleep was short of memory; the engine stays awake")
            self.startup_sec = self._clock() - self.started_at
            log.info("ready (%s) after %.1f s; wake needs %s bytes per device",
                     self.state, self.startup_sec, self.wake_need_bytes)
        except RequestError as exc:
            if self.state != "failed":
                self._fail(f"engine load failed: {exc.body.get('error')}")
        except Exception as exc:  # noqa: BLE001 -- reported through /status
            self._fail(f"engine load failed: {exc!r}")
            log.exception("engine load failed")

    def _fail(self, error: str) -> None:
        self.state = "failed"
        self.phase = None
        self.error = error

    def _gpu_short_for_load(self) -> dict | None:
        """The shortage a load would meet on the cards: vLLM starts only when every card has
        ``load_fraction`` of its memory free. None when they have, or no fraction was given."""
        if self._load_fraction is None:
            return None
        free = self._gpu_memory()
        need = [math.ceil(self._load_fraction * d["total_bytes"]) for d in free]
        short = [d["index"] for d, n in zip(free, need) if d["free_bytes"] < n]
        if not short:
            return None
        return {"memory": "gpu", "operation": "load",
                "error": f"devices {short} have less free memory than the engine claims when it starts",
                "gpu_free_bytes": [d["free_bytes"] for d in free], "gpu_need_bytes": need}

    def _unload(self, shortage: dict) -> None:
        self.state = UNLOADED_STATE
        self.phase = None
        self.memory_error = shortage
        log.warning("memory was short for the load: %s", shortage)

    def begin_load(self) -> bool:
        """Start a load in the background when the service is unloaded; whether it started."""
        with self._lock:
            if self.state != UNLOADED_STATE:
                return False
            self.state = "starting"
            self.phase = LOADING_PHASE
            self.load_started_at = self._clock()
        threading.Thread(target=self.load, name="engine-load", daemon=True).start()
        return True

    def _sleep_locked(self) -> None:
        """Put the engine to sleep, measuring its footprint at the first sleep that succeeds.
        A sleep the engine fails for lack of memory leaves it awake and answers 507; any other
        failure leaves its state unknown, so the service fails."""
        assert self._engine is not None
        # The first sleep is the last step of the start, and it takes tens of seconds:
        # the service is still starting until it ends.
        if self.state != "starting":
            self.state = "sleeping"
        awake = self._gpu_memory() if self.wake_need_bytes is None else None
        t0 = self._clock()
        try:
            self._engine.sleep()
        except Exception as exc:  # noqa: BLE001 -- a shortage is answered, anything else fails
            kind = memory_shortage(exc, operation="sleep")
            if kind is None:
                self._fail(f"sleep failed: {exc!r}")
                raise RequestError(503, {"error": self.error, "state": self.state}) from exc
            # The weights are still on the cards: the engine is awake, and the idle sleep tries
            # again after another idle period.
            self.state = "awake"
            self.phase = None
            self.last_used = self._clock()
            self.memory_error = {"memory": kind, "operation": "sleep", "error": repr(exc)}
            raise RequestError(507, self._short_body(f"the reference cannot sleep: {exc!r}")) from exc
        self.last_sleep_sec = self._clock() - t0
        if awake is not None:
            asleep = self._gpu_memory()
            self.awake_used_bytes = [d["total_bytes"] - d["free_bytes"] for d in awake]
            self.asleep_used_bytes = [d["total_bytes"] - d["free_bytes"] for d in asleep]
            self.wake_need_bytes = [max(0, a - s) + WAKE_MARGIN_BYTES for a, s in
                                    zip(self.awake_used_bytes, self.asleep_used_bytes)]
        self.state = "asleep"
        self.phase = None
        self.memory_error = None
        log.info("asleep (%.2f s)", self.last_sleep_sec)

    def _wake_locked(self) -> None:
        """Wake the engine, or refuse with 507 when the cards lack the room."""
        assert self._engine is not None
        free = self._gpu_memory()
        need = self.wake_need_bytes or [0] * len(free)
        short = [d["index"] for d, n in zip(free, need) if d["free_bytes"] < n]
        if short:
            self.memory_error = {"memory": "gpu", "operation": "wake",
                                 "error": f"devices {short} have less free memory than a wake needs",
                                 "gpu_free_bytes": [d["free_bytes"] for d in free],
                                 "gpu_need_bytes": self.wake_need_bytes}
            raise RequestError(507, self._no_room_body(
                free, f"devices {short} have less free memory than a wake needs"))
        self.state = "waking"
        t0 = self._clock()
        try:
            self._engine.wake()
        except Exception as exc:  # noqa: BLE001 -- answered as 507 below
            log.exception("wake failed; putting the engine back to sleep")
            try:
                self._engine.sleep()
                self.state = "asleep"
            except Exception as exc2:  # noqa: BLE001
                self._fail(f"wake failed ({exc!r}) and re-sleep failed ({exc2!r})")
                raise RequestError(503, {"error": self.error}) from exc2
            free = self._gpu_memory()
            self.memory_error = {"memory": memory_shortage(exc, operation="wake"), "operation": "wake",
                                 "error": repr(exc), "gpu_free_bytes": [d["free_bytes"] for d in free],
                                 "gpu_need_bytes": self.wake_need_bytes}
            raise RequestError(507, self._no_room_body(free, f"wake failed: {exc!r}")) from exc
        self.last_wake_sec = self._clock() - t0
        self.state = "awake"
        self.last_used = self._clock()
        self.memory_error = None
        log.info("awake (%.2f s)", self.last_wake_sec)

    def _no_room_body(self, free: list[dict], why: str) -> dict:
        return {"error": f"the reference cannot wake: {why}",
                "memory": self.memory_error["memory"] if self.memory_error else "gpu",
                "gpu_free_bytes": [d["free_bytes"] for d in free],
                "wake_need_bytes": self.wake_need_bytes,
                "instruction": "Stop or shrink your own server so that every device has "
                               "at least wake_need_bytes free, then retry."}

    def _short_body(self, error: str) -> dict:
        """The 507 body of a shortage ``memory_error`` names, with what to do about it."""
        short = dict(self.memory_error or {})
        memory = short.get("memory")
        release = ("Stop or shrink your own server so that every device has at least gpu_need_bytes free"
                   if memory == "gpu" else "Release host memory your processes hold")
        then = ("then POST /wake to load the reference again" if self.state == UNLOADED_STATE
                else "then retry")
        instruction = f"{release}, {then}."
        return {"error": error, "state": self.state, **short, "instruction": instruction}

    def _check_usable(self) -> None:
        if self.state == "starting":
            raise RequestError(503, {
                "error": "the reference is still starting; retry once /status reports "
                         "asleep",
                "state": self.state, "phase": self.phase,
                "starting_sec": round(self._clock() - self.load_started_at, 1)})
        if self.state == "failed":
            raise RequestError(503, {"error": self.error, "state": self.state})
        if self.state == UNLOADED_STATE:
            raise RequestError(507, self._short_body(
                f"the reference is not loaded: {(self.memory_error or {}).get('error')}"))

    # -- requests --------------------------------------------------------------------

    def status(self) -> dict:
        try:
            gpus = self._gpu_memory()
        except Exception as exc:  # noqa: BLE001 -- status must answer regardless
            gpus = [{"error": repr(exc)}]
        return {"state": self.state, "phase": self.phase, "error": self.error,
                "memory_error": self.memory_error,
                "startup_sec": self.startup_sec, "last_used": self.last_used,
                "last_wake_sec": self.last_wake_sec, "last_sleep_sec": self.last_sleep_sec,
                "idle_sleep_sec": self._idle_sleep_sec,
                "gpu_free_bytes": [d.get("free_bytes") for d in gpus],
                "gpu_total_bytes": [d.get("total_bytes") for d in gpus],
                "awake_used_bytes": self.awake_used_bytes,
                "asleep_used_bytes": self.asleep_used_bytes,
                "wake_need_bytes": self.wake_need_bytes,
                "engine": self._config}

    def sleep(self) -> dict:
        if self.state == UNLOADED_STATE:
            return self.status()
        self._check_usable()
        with self._lock:
            if self.state == "awake":
                self._sleep_locked()
        return self.status()

    def wake(self) -> dict:
        if self.begin_load():
            return self.status()
        self._check_usable()
        with self._lock:
            if self.state == "asleep":
                self._wake_locked()
        return self.status()

    def score(self, body: dict) -> dict:
        self._check_usable()
        assert self._max_model_len is not None and self._vocab_size is not None
        prompt, cont, positions = parse_score_request(
            body, vocab_size=self._vocab_size, max_model_len=self._max_model_len)
        with self._lock:
            woke = self.state == "asleep"
            if woke:
                self._wake_locked()
            assert self._engine is not None
            t0 = self._clock()
            ids = prompt + cont
            try:
                rows = self._engine.rows(ids)
            finally:
                self.last_used = self._clock()
            score_sec = self._clock() - t0
        n = len(prompt)
        k = self._top_logprobs
        out = {"n_prompt": n, "n_continuation": len(cont),
               "continuation": [position_record(rows[n + j], cont[j], k)
                                for j in range(len(cont))],
               "woke": woke, "wake_sec": self.last_wake_sec if woke else None,
               "score_sec": score_sec}
        if positions:
            out["prompt"] = [{"position": p, **position_record(rows[p], prompt[p], k)}
                             for p in positions]
        return out

    def shutdown(self) -> None:
        """End the engine's processes, when it has been built. Takes no lock: it runs
        on the way out, and a score in flight must not hold the exit up."""
        stop = getattr(self._engine, "shutdown", None)
        if stop is not None:
            stop()

    def sleep_if_idle(self) -> None:
        """Put an awake engine to sleep once it has been idle for ``idle_sleep_sec``. A sleep
        short of memory leaves it awake, and the next one comes after another idle period."""
        if self.state != "awake" or self.last_used is None:
            return
        if self._clock() - self.last_used < self._idle_sleep_sec:
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            if (self.state == "awake" and self.last_used is not None
                    and self._clock() - self.last_used >= self._idle_sleep_sec):
                log.info("idle for %.0f s; sleeping", self._clock() - self.last_used)
                try:
                    self._sleep_locked()
                except RequestError as exc:
                    log.warning("the idle sleep did not complete: %s", exc.body.get("error"))
        finally:
            self._lock.release()


class _Handler(BaseHTTPRequestHandler):
    """HTTP/1.0 over the unix socket: one request per connection."""

    server: "UnixHTTPServer"

    def address_string(self) -> str:
        # A unix-socket peer has no address; the base class indexes into one.
        return "unix"

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s", fmt % args)

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self, method: str) -> None:
        service = self.server.service
        path = self.path.split("?", 1)[0]
        routes = {("GET", "/status"): lambda: service.status(),
                  ("POST", "/sleep"): lambda: service.sleep(),
                  ("POST", "/wake"): lambda: service.wake(),
                  ("POST", "/score"): lambda: service.score(self._json_body())}
        handler = routes.get((method, path))
        try:
            if handler is None:
                known = sorted({p for _, p in routes})
                raise RequestError(404 if path not in known else 405,
                                   {"error": f"no route {method} {path}"})
            self._send(200, handler())
        except RequestError as exc:
            self._send(exc.status, exc.body)
        except Exception as exc:  # noqa: BLE001 -- answered, and logged in full
            log.exception("request %s %s failed", method, path)
            self._send(500, {"error": repr(exc)})

    def _json_body(self) -> dict:
        # A negative length would make ``read`` take the stream to its end, past the cap.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            raise RequestError(400, {"error": "Content-Length must be a non-negative integer"})
        if length > MAX_BODY_BYTES:
            raise RequestError(413, {"error": f"body over {MAX_BODY_BYTES} bytes"})
        try:
            return json.loads(self.rfile.read(length) or b"null")
        except json.JSONDecodeError as exc:
            raise RequestError(400, {"error": f"body is not JSON: {exc}"}) from exc

    def do_GET(self) -> None:  # noqa: N802 -- http.server's naming
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """A threaded HTTP server on a unix socket, carrying the service it answers for."""

    daemon_threads = True

    def __init__(self, path: str, service: Service) -> None:
        if os.path.exists(path):
            os.unlink(path)
        super().__init__(path, _Handler)
        # Anyone who can reach the socket's directory may use it.
        os.chmod(path, 0o666)
        self.service = service


def vocab_size_of(weights: str) -> int:
    """The checkpoint's vocabulary width, from its config.json (text config first)."""
    with open(os.path.join(weights, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    text = cfg.get("text_config") or {}
    size = text.get("vocab_size") or cfg.get("vocab_size")
    if not isinstance(size, int):
        raise ValueError(f"{weights}/config.json declares no vocab_size")
    return size


def _idle_loop(service: Service) -> None:
    while True:
        time.sleep(IDLE_POLL_SEC)
        try:
            service.sleep_if_idle()
        except Exception:  # noqa: BLE001 -- keep watching; the error is logged
            log.exception("idle sleep failed")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--socket", required=True, help="unix socket path to serve on")
    ap.add_argument("--weights", default="/data/weights")
    ap.add_argument("--tensor-parallel-size", type=int, required=True)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="context length; omitted, vLLM takes the checkpoint's")
    ap.add_argument("--gpu-memory-utilization", type=float, required=True)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--batch-invariant", action="store_true")
    ap.add_argument("--top-logprobs", type=int, default=21)
    ap.add_argument("--idle-sleep-sec", type=float, default=120.0)
    ap.add_argument("--sleep-level", type=int, choices=(1, 2), default=1)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config = {"tensor_parallel_size": args.tensor_parallel_size,
              "max_model_len": args.max_model_len,
              "gpu_memory_utilization": args.gpu_memory_utilization,
              "dtype": args.dtype, "enforce_eager": args.enforce_eager,
              "batch_invariant": args.batch_invariant,
              "top_logprobs": args.top_logprobs, "enable_prefix_caching": False,
              "sleep_level": args.sleep_level}

    def factory() -> Engine:
        return VLLMEngine(weights=args.weights, dtype=args.dtype,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          max_model_len=args.max_model_len,
                          enforce_eager=args.enforce_eager,
                          tensor_parallel_size=args.tensor_parallel_size,
                          batch_invariant=args.batch_invariant,
                          top_logprobs=args.top_logprobs, sleep_level=args.sleep_level)

    service = Service(engine_factory=factory, vocab_size=None,
                      max_model_len=args.max_model_len, top_logprobs=args.top_logprobs,
                      idle_sleep_sec=args.idle_sleep_sec, gpu_memory=nvml_memory,
                      config=config, load_fraction=args.gpu_memory_utilization)
    stopping = threading.Event()

    def on_signal(signum, _frame) -> None:
        log.info("received signal %d; shutting down", signum)
        stopping.set()

    # Installed before anything slow, so a stop during the engine build is prompt. Also
    # needed because the process may be its container's PID 1, which the kernel does not
    # hand a signal it has no handler for.
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    # The socket is bound before the engine builds, so /status reports "starting" while it
    # does, and a caller can tell a slow start from a dead one.
    server = UnixHTTPServer(args.socket, service)
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    log.info("serving on %s", args.socket)

    def start() -> None:
        service.load()
        _idle_loop(service)

    threading.Thread(target=start, name="engine", daemon=True).start()
    while not stopping.wait(timeout=1.0):
        pass
    return _shut_down(server, service, args.socket)


def _shut_down(server: "UnixHTTPServer", service: Service, socket_path: str) -> NoReturn:
    """Stop serving, give the engine ``SHUTDOWN_GRACE_SEC`` to end its workers, and exit
    without running interpreter exit hooks: vLLM's can block on workers already gone."""
    server.shutdown()
    server.server_close()
    with contextlib.suppress(OSError):
        os.unlink(socket_path)
    ender = threading.Thread(target=service.shutdown, name="shutdown", daemon=True)
    ender.start()
    ender.join(timeout=SHUTDOWN_GRACE_SEC)
    log.info("engine shutdown %s; exiting", "timed out" if ender.is_alive() else "done")
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
