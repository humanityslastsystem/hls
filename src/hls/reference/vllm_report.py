"""Reference log-probabilities out of vLLM, for the sampling gate.

The report-KL gate needs, at every position a server served, the reference's own
log-probability at the ~21 ids that server reported, the log of the mass outside them,
and the log of the mass on the end-of-turn ids. All three are reductions over the whole
vocabulary, so a top-k report cannot supply them: the tail has to be summed in log
space, and on a peaked row ``log(1 - sum(reported))`` cancels to noise.

Why vLLM and not the HF reference
---------------------------------

The HF engine already blocks its output head over row-chunks, so the logits were never
the problem. The problem is getting the hidden states at all. The checkpoints this path
scores use sparse attention, where each query attends to its own selection of keys, and
transformers has no flash-attention kernel for that shape. Attention falls back to SDPA,
which materializes the selection densely over queries by keys -- hundreds of GiB at a
128k prefix, against contexts here several times longer. vLLM implements the sparse
attention directly and prefills it paged and chunked.

Where the numbers come from
---------------------------

vLLM computes exactly what is needed and then throws it away. A request that asked for
prompt logprobs gets its prompt hidden states run through the output head, and then
reduced to a top-k list. On the GPU model runner that reduction lives in
``PromptLogprobsWorker.compute_prompt_logprobs``, which is handed the batch's hidden
states and the model's own logits function. This module wraps that method: for the
reply it has been armed for, it runs the same output head over that reply's rows and
applies the gate's three reductions on the device, so only about twenty floats per
position ever cross to the host. Everything else -- the prefill schedule, the paging,
the kernels -- is vLLM's.

How the pieces reach the worker
-------------------------------

The wrap has to live in the worker process, which is where the model and its logits
live. ``ReportWorker`` is mixed into vLLM's worker class at engine construction
(``worker_extension_cls``), so its methods are reachable by name through
``collective_rpc``. Naming them matters twice over: the engine refuses to ship function
objects to a worker unless insecure serialization is turned on, and a bound method
receives the worker, which ``collective_rpc`` supplies as the first argument to
anything it is handed.

Which reply the capture belongs to
----------------------------------

The host does not learn a request's vLLM id before submitting it -- ``LLM.generate``
mints its own -- so the capture is not keyed on one. It holds a single slot instead:
the host arms it, submits one request, and takes the rows back. The hook binds that
slot to the first request in the batch that asked for prompt logprobs, and refuses a
second, so scoring two replies at once fails instead of interleaving their rows.

Fragility, stated plainly
-------------------------

This wraps a private vLLM method. ``install`` reads the live model runner rather than
an import: it refuses unless that runner holds the prompt-logprob worker whose method
it is about to wrap, that method takes the arguments the capture reads, and the wrap
is what the live instance resolves to afterwards. A vLLM that moves this reduction
somewhere else therefore fails at install rather than leaving a wrap that nothing
calls.
"""
from __future__ import annotations

import inspect
import threading
from typing import Any

#: Decimals the reported floats are rounded to before crossing the RPC. A
#: log-probability to six decimals is a part in a million of the atom's mass, far below
#: the rounding a served report itself arrives with. Kept equal to the HF engine's
#: ``_REPORT_DECIMALS`` so the two references round identically.
REPORT_DECIMALS: int = 6

#: Rows of the output head evaluated at once. The head is the one tensor whose width is
#: the vocabulary, so it is blocked rather than materialized whole.
HEAD_BLOCK: int = 256

#: Qualified name of the worker mixin, handed to ``LLM`` as ``worker_extension_cls``.
WORKER_EXTENSION_CLS: str = "hls.reference.vllm_report.ReportWorker"

#: Module the wrapped reduction is expected to come from. The capture's index
#: arithmetic was written against this file, so a reduction defined anywhere else is a
#: different one and the wrap must not be applied to it.
_REDUCTION_MODULE: str = "vllm.v1.worker.gpu.sample.prompt_logprob"

#: Parameters the wrapped reduction must take, in order. The capture reads
#: ``logits_fn``, ``hidden_states``, ``input_batch`` and ``prompt_lens`` positionally,
#: so a signature that reordered or renamed them would silently feed it the wrong
#: tensors.
_REDUCTION_PARAMS: tuple[str, ...] = (
    "self", "logits_fn", "hidden_states", "input_batch", "all_token_ids",
    "num_computed_tokens", "prompt_lens",
)

#: Fields of vLLM's batch the capture indexes to find a request's rows.
_BATCH_FIELDS: tuple[str, ...] = (
    "req_ids", "idx_mapping_np", "query_start_loc_np", "prefill_len_np",
    "num_computed_prefill_tokens_np",
)


class ReportCapture:
    """The reply being scored, and the rows the hook reduces for it.

    One instance per worker process, holding one reply at a time. Guarded by a lock
    because vLLM's runner reduces on the engine loop while the RPC that arms the slot
    arrives on another thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        #: Reported ids, end-of-turn ids and whether to reduce the stop row, for the
        #: armed reply, or None when idle.
        self._spec: dict | None = None
        #: The vLLM request id the slot bound to, once the hook has seen one.
        self._req_id: str | None = None
        #: Reduced rows so far, plus the count that fixes where the next chunk starts.
        self._rows: dict | None = None
        #: Why the capture gave up on this reply, reported to the host by ``take``.
        self._error: str | None = None

    def arm(self, report_ids: list[list[int]], eot_ids: list[int],
            stop_row: bool = False) -> None:
        """Take the next prompt-logprobs request as the reply to score."""
        with self._lock:
            self._spec = {"report_ids": report_ids, "eot_ids": eot_ids}
            if stop_row:
                self._spec["stop_row"] = True
            self._req_id = None
            self._rows = {"p_on_q": [], "p_tail_lp": [], "p_eot_lp": [], "next": 0}
            self._error = None

    def claim(self, req_id: str) -> dict | None:
        """The armed spec, if ``req_id`` is the request the slot belongs to.

        Binds the slot on first sight. A different request asking for prompt logprobs
        while the slot is bound is refused: the host scores one reply per call, and
        rows from two requests would concatenate into one reply's."""
        with self._lock:
            if self._spec is None or self._error is not None:
                return None
            if self._req_id is None:
                self._req_id = req_id
            elif self._req_id != req_id:
                raise RuntimeError(
                    f"request {req_id!r} asked for prompt log-probabilities while the "
                    f"report capture was already bound to {self._req_id!r}; the "
                    f"capture scores one reply at a time and cannot tell two "
                    f"requests' rows apart")
            return self._spec

    def extend(self, req_id: str, start: int, rows: dict) -> None:
        """Append one prefill chunk's rows. Chunked prefill delivers a request's
        positions in pieces, in order, so the chunks are concatenated as they arrive
        and the start index is carried only to assert that order."""
        with self._lock:
            if self._rows is None:
                return
            if start != self._rows["next"]:
                raise RuntimeError(
                    f"prompt-logprob chunk for {req_id!r} starts at {start} but "
                    f"{self._rows['next']} rows have been collected; vLLM delivered "
                    f"prefill chunks out of order, which this accumulator assumes it "
                    f"cannot.")
            for key in ("p_on_q", "p_tail_lp", "p_eot_lp"):
                self._rows[key].extend(rows[key])
            self._rows["next"] = start + len(rows["p_tail_lp"])

    def stop(self, req_id: str, row: dict) -> None:
        """Keep the stop row: the end-of-turn log-probabilities at the position after
        the reply's last token. It arrives with the prefill's last chunk, after every
        row of the reply."""
        with self._lock:
            if self._rows is None:
                return
            if self._rows["next"] != len(self._spec["report_ids"]):
                raise RuntimeError(
                    f"the stop row for {req_id!r} arrived after "
                    f"{self._rows['next']} of the reply's "
                    f"{len(self._spec['report_ids'])} rows; it predicts the token after "
                    f"the last one, so it has to come last")
            self._rows.update(row)

    def fail(self, reason: str) -> None:
        """Abandon the armed reply, keeping why for the host.

        The hook runs inside the engine loop, where raising kills the engine for every
        later reply too. Recording the reason instead lets ``take`` hand it back and
        the host raise on its own thread."""
        with self._lock:
            self._error = reason
            self._rows = None

    def take(self) -> dict | None:
        """The armed reply's rows, its failure, or None when nothing was armed.

        Clears the slot either way, so a later reply cannot inherit these rows."""
        with self._lock:
            spec, rows, error = self._spec, self._rows, self._error
            self._spec = self._rows = self._error = self._req_id = None
        if spec is None:
            return None
        if error is not None:
            return {"error": error}
        got = dict(rows or {})
        got.pop("next", None)
        return got


#: The per-process capture. Module-level because the hook is a monkeypatch and has
#: nowhere else to reach for it.
CAPTURE = ReportCapture()


class ReportWorker:
    """Report-capture calls, reachable on a vLLM worker by name.

    vLLM mixes this into its worker class when the engine is built with
    ``worker_extension_cls``, which is what lets ``collective_rpc`` address these by
    name instead of shipping a function object."""

    def hls_install_report(self) -> str:
        """Wrap this worker's prompt-logprob reduction. Returns what was wrapped."""
        return install(self.model_runner)                  # type: ignore[attr-defined]

    def hls_arm_report(self, report_ids: list[list[int]],
                       eot_ids: list[int], stop_row: bool = False) -> None:
        """Arm the capture for the reply the host is about to submit."""
        CAPTURE.arm(report_ids, eot_ids, stop_row)

    def hls_take_report(self) -> dict | None:
        """Hand back the armed reply's rows and clear the slot."""
        return CAPTURE.take()


def reduce_rows(scores: Any, report_ids: Any, eot_ids: Any) -> dict:
    """The gate's three reductions over one block of full-vocabulary log-probability
    rows.

    ``scores`` is ``[rows, vocab]`` log-softmaxed; ``report_ids`` is ``[rows, width]``
    of ids with ``-1`` padding; ``eot_ids`` is the checkpoint's end-of-turn ids.
    Mirrors ``HFEngine._reference_logprobs_prepared`` exactly -- the two references
    must not differ in how they reduce, only in how they obtained the rows.
    """
    import torch

    valid = report_ids >= 0
    safe = report_ids.clamp_min(0)
    gathered = scores.gather(1, safe).masked_fill(~valid, float("-inf"))
    # scatter_add_ over the valid flags rather than scatter_ of the flags: pad slots
    # all clamp to column 0, and a plain scatter would let one overwrite a real id 0.
    reported = torch.zeros_like(scores, dtype=torch.int8).scatter_add_(
        1, safe, valid.to(torch.int8)) > 0
    tail = torch.logsumexp(scores.masked_fill(reported, float("-inf")), dim=1)
    eot = torch.logsumexp(scores[:, eot_ids], dim=1)
    return {
        "p_on_q": [[round(v, REPORT_DECIMALS) for v in row] for row in gathered.tolist()],
        "p_tail_lp": [round(v, REPORT_DECIMALS) for v in tail.tolist()],
        "p_eot_lp": [round(v, REPORT_DECIMALS) for v in eot.tolist()],
    }


def reduction_class(runner) -> type:
    """The class whose prompt-logprob reduction this module wraps, taken off the live
    model runner and checked for the shape the capture assumes.

    Reading the runner rather than importing a class is the whole point: it is the
    difference between wrapping something vLLM will call and wrapping something that
    merely still exists. Raises unless the runner holds a prompt-logprob worker, that
    worker comes from the module this capture was written against, and its reduction
    takes the arguments the capture reads positionally."""
    worker = getattr(runner, "prompt_logprobs_worker", None)
    if worker is None:
        raise RuntimeError(
            f"the live model runner ({type(runner).__module__}."
            f"{type(runner).__name__}) holds no prompt_logprobs_worker, so it does "
            f"not reduce prompt log-probabilities where this capture wraps them. The "
            f"report reference needs vLLM's GPU model runner V2 "
            f"(vllm.v1.worker.gpu.model_runner), which is what a sparse-attention "
            f"checkpoint selects; the older runner keeps that reduction on itself and "
            f"is not supported here.")
    cls = type(worker)
    if cls.__module__ != _REDUCTION_MODULE:
        raise RuntimeError(
            f"the runner's prompt-logprob worker is {cls.__module__}.{cls.__name__}, "
            f"not a class from {_REDUCTION_MODULE}; the capture's index arithmetic was "
            f"written against that module and must not be applied to another.")
    reduction = getattr(cls, "compute_prompt_logprobs", None)
    if reduction is None:
        raise RuntimeError(
            f"{cls.__module__}.{cls.__name__} has no compute_prompt_logprobs; this "
            f"vLLM does not expose the prompt-logprob path the reference reads. Pin "
            f"the reference image.")
    params = tuple(inspect.signature(reduction).parameters)
    if params != _REDUCTION_PARAMS:
        raise RuntimeError(
            f"{cls.__module__}.{cls.__name__}.compute_prompt_logprobs takes {params}, "
            f"not {_REDUCTION_PARAMS}; the capture reads its arguments by position, "
            f"so a changed signature would feed it the wrong tensors. Pin the "
            f"reference image.")
    return cls


def install(runner) -> str:
    """Wrap the prompt-logprob reduction the given model runner will call. Idempotent.

    Returns a short string naming what was patched, so the caller's log records which
    vLLM internal this run depended on."""
    import torch

    cls = reduction_class(runner)
    name = f"{cls.__module__}.{cls.__name__}.compute_prompt_logprobs"
    original = cls.compute_prompt_logprobs
    if getattr(original, "_hls_wrapped", False):
        return f"{name} (already wrapped)"

    def wrapper(self, logits_fn, hidden_states, input_batch, all_token_ids,
                num_computed_tokens, prompt_lens):
        try:
            _capture(self, logits_fn, hidden_states, input_batch, prompt_lens, torch)
        except Exception as exc:                           # noqa: BLE001
            CAPTURE.fail(f"{type(exc).__name__}: {exc}")
        return original(self, logits_fn, hidden_states, input_batch, all_token_ids,
                        num_computed_tokens, prompt_lens)

    wrapper._hls_wrapped = True                            # type: ignore[attr-defined]
    cls.compute_prompt_logprobs = wrapper                  # type: ignore[method-assign]
    # What the live worker resolves after patching, not what was patched: an instance
    # holding its own copy of the method, or a class that is not the one the runner
    # actually consults, would leave the wrap uncalled and every reply unscored.
    bound = getattr(runner.prompt_logprobs_worker, "compute_prompt_logprobs", None)
    if getattr(bound, "__func__", None) is not wrapper:
        cls.compute_prompt_logprobs = original             # type: ignore[method-assign]
        raise RuntimeError(
            f"wrapping {name} did not change what the live prompt-logprob worker "
            f"resolves, so the capture would never run. The wrap has been undone.")
    return name


def _capture(worker, logits_fn, hidden_states, input_batch, prompt_lens, torch) -> None:
    """Reduce the armed reply's newly prefilled positions into ``CAPTURE``.

    The per-request slicing mirrors vLLM's own reduction: a request owns the batch rows
    ``query_start_loc_np[i]`` to ``[i + 1]``, which carry prompt positions from
    ``num_computed_prefill_tokens_np[i]`` on."""
    batch = _batch_arrays(input_batch, prompt_lens)
    uses_prompt_logprobs = worker.uses_prompt_logprobs[batch["idx"]]
    for i, req_id in enumerate(input_batch.req_ids):
        if not uses_prompt_logprobs[i]:
            continue
        prompt_len = int(batch["prompt_lens"][i])
        computed = int(batch["computed"][i])
        if computed >= prompt_len:
            continue                                       # past the prompt, decoding
        if prompt_len < int(batch["prefill_len"][i]):
            continue                                       # resumed after preemption
        spec = CAPTURE.claim(req_id)
        if spec is None:
            continue
        start = int(batch["query_start"][i])
        scheduled = int(batch["query_start"][i + 1]) - start
        reply_len = len(spec["report_ids"])
        # Row r of the prompt predicts token r + 1, so a reply of n tokens at the end
        # of the prompt is predicted by the n rows from here. The prompt's own last
        # row predicts the first generated token and falls outside that window, which
        # is why nothing here has to drop it the way vLLM's reduction does.
        first_reply_row = prompt_len - 1 - reply_len
        lo = max(computed, first_reply_row)
        hi = min(computed + scheduled, first_reply_row + reply_len)
        if hi > lo:
            rows = _reduce_reply_rows(logits_fn, hidden_states, spec, torch,
                                      row_offset=start - computed, lo=lo, hi=hi,
                                      first_reply_row=first_reply_row)
            CAPTURE.extend(req_id, lo - first_reply_row, rows)
        # That last row is the stop row, reduced only when the host asked for it. It is
        # the last row of the prefill's last chunk.
        last = prompt_len - 1
        if spec.get("stop_row") and computed <= last < computed + scheduled:
            at = start + last - computed
            CAPTURE.stop(req_id, _reduce_stop_row(
                logits_fn, hidden_states[at:at + 1], spec["eot_ids"], torch))


def _batch_arrays(input_batch, prompt_lens) -> dict:
    """The per-request arrays the capture indexes, all keyed by batch position.

    ``prompt_lens`` arrives keyed by request-state index instead, so it is reindexed
    here the way vLLM reindexes it."""
    missing = [f for f in _BATCH_FIELDS if not hasattr(input_batch, f)]
    if missing:
        raise RuntimeError(
            f"vLLM's prompt-logprob batch is missing {', '.join(missing)}, which the "
            f"capture indexes to find a reply's rows. Pin the reference image.")
    idx = input_batch.idx_mapping_np
    return {"idx": idx,
            "prompt_lens": prompt_lens[idx],
            "computed": input_batch.num_computed_prefill_tokens_np,
            "prefill_len": input_batch.prefill_len_np,
            "query_start": input_batch.query_start_loc_np}


def _reduce_reply_rows(logits_fn, hidden_states, spec, torch, *, row_offset: int,
                       lo: int, hi: int, first_reply_row: int) -> dict:
    """The gate's reductions over prompt rows ``lo`` to ``hi``, blocked over the head.

    ``row_offset`` carries a prompt row into its place in this step's hidden states."""
    report_ids = spec["report_ids"]
    width = max((len(r) for r in report_ids), default=0)
    eot = torch.tensor(spec["eot_ids"], dtype=torch.int64, device=hidden_states.device)
    acc: dict[str, list] = {"p_on_q": [], "p_tail_lp": [], "p_eot_lp": []}
    with torch.no_grad():
        for block_lo in range(lo, hi, HEAD_BLOCK):
            block_hi = min(block_lo + HEAD_BLOCK, hi)
            rows = hidden_states[row_offset + block_lo:row_offset + block_hi]
            scores = torch.log_softmax(logits_fn(rows).float(), dim=-1)
            ids = torch.full((block_hi - block_lo, width), -1, dtype=torch.int64,
                             device=scores.device)
            for k in range(block_lo, block_hi):
                row = report_ids[k - first_reply_row]
                ids[k - block_lo, :len(row)] = torch.tensor(
                    row, dtype=torch.int64, device=scores.device)
            got = reduce_rows(scores, ids, eot)
            for key in acc:
                acc[key].extend(got[key])
    return acc


def _reduce_stop_row(logits_fn, row, eot_ids: list[int], torch) -> dict:
    """The end-of-turn log-probabilities in one hidden row, each id's and their
    mass, through the same head and the same reduction as the reply's rows."""
    eot = torch.tensor(eot_ids, dtype=torch.int64, device=row.device)
    with torch.no_grad():
        scores = torch.log_softmax(logits_fn(row).float(), dim=-1)
        got = reduce_rows(scores, eot[None, :], eot)
    return {"p_stop_lp": got["p_on_q"][0], "p_stop_eot_lp": got["p_eot_lp"][0]}


__all__ = ["CAPTURE", "HEAD_BLOCK", "REPORT_DECIMALS", "WORKER_EXTENSION_CLS",
           "ReportCapture", "ReportWorker", "install", "reduce_rows",
           "reduction_class"]
