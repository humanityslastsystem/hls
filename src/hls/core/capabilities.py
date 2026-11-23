"""Capabilities -- the verifier<->engine axis.

A Capability is an ABC with exactly one abstract async method. An engine gains a
capability by subclassing it and implementing that method, so presence is
language-enforced (an unimplemented method makes the engine un-instantiable) and
``supports(engine)`` is derivable from the MRO, never hand-maintained. A Verifier
declares the capabilities it ``requires`` abstractly; the verification core routes each
``engine[Cap]`` to the loaded engine that provides it, and the call is a method call on it.
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any


class Capability(ABC):
    """Base for a single engine capability (one abstract async method)."""


class Generate(Capability):
    @abstractmethod
    async def generate(self, prompts: list[Any]) -> list[Any]:
        """Produce one output envelope per prompt envelope."""


class DecodeCodes(Capability):
    """Codec codes -> audio: the neural-codec decoder half of a speech stack.

    A capability of its own because its input is a code matrix somebody else already
    produced, not a prompt. A fidelity gate that asks whether a served waveform is the
    decoder's output for the codes sent with it needs the decoder alone, without
    re-running the model that chose the codes."""

    @abstractmethod
    async def decode_codes(self, codes_list: list[Any]) -> list[Any]:
        """Per code matrix (one row per codec frame, one id per codebook group),
        the decoder's waveform as a float32 array at the codec's own sample rate."""


class TeacherForce(Capability):
    @abstractmethod
    async def teacher_force(self, items: list[Any]) -> list[list[float]]:
        """Per item, return per-token greedy gaps for the item's continuation."""


#: Id of an empty slot in a ``ReferenceLogprobs`` row of reported ids. Rows are a fixed
#: width so a record keeps a rectangular shape, and a position whose report is narrower
#: than that width pads with this. It carries no mass: the engine gathers minus infinity
#: there rather than indexing the vocabulary with it. Defined here, beside the capability
#: whose wire format it belongs to, so the engine and the verify layer that builds the
#: rows (``hls.verify.report.PAD_ID``) share one definition.
PAD_ID: int = -1


class ReferenceLogprobs(Capability):
    @abstractmethod
    async def reference_logprobs(self, items: list[Any]) -> list[dict]:
        """Per item, the reference's own log-probabilities at the ids the caller names.

        The caller carries ``metadata["report_ids"][j]`` for every continuation position
        j: the token ids a served reply reported alternatives for at that position. The
        engine returns, per item:

        - ``p_on_q``: the reference log-probability at each of those ids (the pad id
          ``PAD_ID`` gathers minus infinity).
        - ``p_tail_lp``: the log of the reference mass outside them. Summed in log space
          rather than as ``log(1 - sum)``, which cancels to noise on a peaked row where
          the named ids hold nearly all the mass.
        - ``p_eot_lp``: per position, the log of the reference's mass on the
          checkpoint's end-of-turn ids over the whole vocabulary. It prices ending the
          turn at every position the submission served, which is what the end-of-turn
          count is read against (``hls.verify.report``). In log space for the range:
          the interesting values run down to ``e**-40`` and below.
        - ``p_stop_lp`` and ``p_stop_eot_lp``, only for an item carrying
          ``metadata["stop_row"]``: at the position after the last served token, the
          log-probability of each end-of-turn id (parallel to ``eot_ids``) and the log
          of their mass, the same reduction as ``p_eot_lp``. That position is scored
          only on request because the server reported nothing there.
        - ``n_ctx``: the rendered prompt's length in tokens.
        - ``eot_ids``: the checkpoint's end-of-turn ids, and ``max_context``: the
          context window its config declares, or ``None`` when it declares none. Two
          facts about the checkpoint rather than about the item, returned here because
          the engine is the only side that holds the checkpoint: a caller enforcing an
          ending rule would otherwise have to carry them as hand-authored numbers that
          nothing keeps true.

        An item the engine cannot score -- a served or reported id outside its
        vocabulary -- comes back as ``{"invalid": reason}`` instead, closing that reply
        rather than failing the batch.

        Separate from ``TeacherForce``, which returns one number per position (the
        greedy gap) and so cannot express a distribution over a set of ids."""


class ServeTraffic(Capability):
    @abstractmethod
    async def serve_traffic(self, loaded: list[Any], *, seed: int, request_timeout_sec: float,
                            max_sessions: int | None = None,
                            drive_timeout_sec: float | None = None,
                            agent: Any = None, max_context_len: int | None = None) -> Any:
        """Drive one clocked serving of a task's traffic against this engine and
        return the resulting ``Trace``.

        The drive loop runs inside the engine's own process: requests come due on
        the task's arrival schedule and are submitted one at a time to the engine's
        asynchronous serving API, so the engine batches them exactly as a live
        server would. Crossing a process boundary per request would put transport
        latency inside the serving dynamics being measured; going through an HTTP
        endpoint would add a wire protocol on top of that.

        ``loaded`` is ``[(tenant, items)]``: the sessions to serve, materialized by
        the caller (a reference container has no dataset mount). ``seed`` clocks the
        drive -- arrivals, think-times and ramp-down membership. ``max_sessions``
        bounds how many sessions are served, trimmed per tenant in proportion.

        ``agent`` is the task's ``AgentSpec`` when it declares one. The serving then
        runs the task's own trajectory loop in this process, with its framework
        choosing the turns and its tools running here, so the prompts and
        continuations measured are the ones a graded run serves. Tool credentials and
        tool packages must therefore be present in this engine's container.
        ``max_context_len`` is the task's context window, which each trajectory is
        budgeted against, as in a graded run.

        ``request_timeout_sec`` bounds one request (the drive's per-request grace);
        ``drive_timeout_sec`` bounds the whole drive. The two are not the same bound: a
        session's turns are serial and the per-request clock re-arms on each one, so on a
        session or trajectory workload the per-request bound caps no total. Past the
        drive's wall nothing new starts and nothing in flight continues -- the cut units
        are recorded as faults, which is a measured outcome rather than an error. ``None``
        for the wall leaves it to the agent's own, or off for a task with no agent seam.

        Only an engine with an async per-request API can provide this; a
        batch-oriented one cannot take traffic on a clock."""


class ServingVersion(Capability):
    @abstractmethod
    async def serving_version(self) -> str:
        """The installed version of the serving package this engine drives traffic
        through, read inside this process.

        Asked of the engine rather than of the caller because the serving stack lives
        in the engine's own container: whatever the host has installed says nothing
        about which build produced a serving. Separate from ``StackReport``, which
        describes the accelerator stack every engine shares."""


class StackReport(Capability):
    @abstractmethod
    async def stack_report(self) -> dict:
        """Torch/CUDA/cuDNN/GPU-name versions of the loaded engine's stack.

        Every leaf value is an exact builtin (``str``), never a framework subclass of
        one, so the report is plain data wherever it is recorded."""


def cap_method(cap: type[Capability]) -> str:
    """The name of a capability's single abstract async method (its wire method)."""
    names = cap.__abstractmethods__
    if len(names) != 1:
        raise ValueError(f"capability {cap.__name__} must declare exactly one method, got {set(names)}")
    name = next(iter(names))
    if not inspect.iscoroutinefunction(getattr(cap, name)):
        raise ValueError(f"capability {cap.__name__}.{name} must be an async method")
    return name
