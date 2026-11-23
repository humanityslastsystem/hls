"""L3 -- Reference oracle CONFIG (the data face; engine plane lives in engines.py).

A Reference is a pure, frozen pydantic config the verifier owns and serializes
into provenance. It is 1:1 with its runtime Engine (built via ``build_engine(task)``),
which carries the capabilities (Generate/TeacherForce) and the loaded model.
``task`` (TaskSpec) is the single config object -- ``build_engine`` reads the model
from ``task.model`` so the oracle always matches the task's declared model identity.
"""
from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from hls.core.decode import DecodeConfig
from hls.core.engine import Engine
from hls.core.poly import Poly, dump_spec
from hls.reference.engines import (DEFAULT_QUERY_CHUNK, HFEngine, SGLangEngine, VLLMEngine,
                                   contract_conditioning)

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec


class Reference(Poly):
    """Config base for a reference oracle. 1:1 with a runtime Engine.

    ``engine_cls`` is the runtime ``Engine`` class this config builds. It is a
    class-level handle (NOT a field) so wiring/capabilities are checkable FROM
    CONFIG ALONE -- ``supports(ref.engine_cls)`` reads the capability set without
    constructing an engine (importing the class is cheap; the model load is lazy in
    ``Engine._load``). ``build_engine(task)`` builds the engine a role loads, and is
    not for config-time validation.

    ``task`` (TaskSpec) is the single source of truth: ``task.model`` carries the
    model identity so the oracle is always consistent with the task's declared model
    and no duplicated model field is needed on the Reference itself."""

    decode: DecodeConfig
    #: Keep this reference's own answers to corpus rows in the task's answers area of the
    #: store and serve them to every later verify (``hls.verify.answers``). Only those
    #: (``hls.verify.answers.CORPUS_CAPABILITIES``):
    #: nothing it answers about a contestant's reply or its stack is cached, and
    #: ``check_wiring`` refuses a caching reference that a verifier requires for any
    #: such answer, which rules it out for the oracle of a teacher-forcing or report
    #: gate. Off by default because an oracle returning audio or images would write an
    #: enormous file; on for an oracle whose rollout of every row is expensive and whose
    #: verifier requires nothing else of it.
    #:
    #: Kept out of ``dump_spec``, and so out of the reference's fingerprint and the
    #: workload payload: it says whether an answer is recomputed or served from the
    #: store, never what the answer is, so it is no part of what the reference measures.
    cache: bool = Field(default=False, exclude=True)
    engine_cls: ClassVar[type[Engine]]

    def fingerprint(self, task: "TaskSpec") -> str:
        # Fold the engine KIND (HFEngine vs VLLMEngine can produce different text)
        # plus a hash of the FULL config payload: ``dump_spec`` covers decode and
        # every output-affecting knob a subclass declares (e.g. VLLMReference's
        # dtype / max_model_len), so a future field can never silently collide two
        # oracles in the fingerprint / cache scope. The contract's wire protocol is
        # hashed alongside it because the same reference produces different text under
        # different contracts (conditioning, stop ids), so the protocol scopes the cache
        # too. ``decode.fingerprint()`` stays in the string as the human-readable scope
        # tag.
        artifact_hash = hashlib.sha256(task.model.fingerprint.encode()).hexdigest()[:8]
        proto = getattr(task.contract, "protocol", None)
        spec_hash = hashlib.sha256(json.dumps(
            {"reference": dump_spec(self),
             "protocol": dump_spec(proto) if proto is not None else None},
            sort_keys=True).encode()).hexdigest()[:8]
        return (f"{self.engine_cls.__name__}_{artifact_hash}_"
                f"{self.decode.fingerprint()}_{spec_hash}")

    @abstractmethod
    def build_engine(self, task: "TaskSpec"):
        """Construct (do not load) this reference's runtime Engine from ``task``."""


class HFReference(Reference):
    """HuggingFace-backed reference oracle -- the teacher-forcing instrument every
    calibrated verifier scores against.

    ``query_chunk`` is how many query rows one teacher-forcing forward pass covers
    (``HFEngine._continuation_hidden``). It is declared here, per task, rather than
    fixed in the engine because it is both a memory knob and a numeric one: a long
    context needs a small chunk to fit the card at all, and the math kernel tiles
    differently at different widths, so gaps in a long reply move with the chunk width,
    enough to carry a gap across a threshold. A
    calibration measures the thresholds on one instrument, so the chunk it ran at is
    part of that instrument's identity: it rides into the workload fingerprint through
    ``dump_spec`` (``hls.verify.calibration.workload_payload``), and a calibration taken
    at one chunk is refused as stale by a task declaring another. A task that declares
    nothing gets ``DEFAULT_QUERY_CHUNK``, recorded in the fingerprint like any other
    value; changing a calibrated task's chunk is a recalibration, not a retune."""

    engine_cls: ClassVar[type[Engine]] = HFEngine
    query_chunk: int = DEFAULT_QUERY_CHUNK

    def build_engine(self, task: "TaskSpec"):
        # gpu_kind: pragmatic default, NOT task semantics -- see ReferenceEngine.__init__.
        return self.engine_cls(model_spec=task.model, decode=self.decode,
                               gpu_kind=task.hardware.accelerator.kind,
                               query_chunk=self.query_chunk,
                               **contract_conditioning(task))


class VLLMReference(Reference):
    """vLLM-backed reference. Under a greedy decode it is a Generate provider for metric
    verifiers, faster than HFReference for throughput-bound oracle rollouts. Under a
    sampled decode it is the report gate's pinned reference: it scores a served report
    (``ReferenceLogprobs``) and cannot generate. A dense forward over a context of several
    hundred thousand tokens does not fit beside a large checkpoint; vLLM's paged, chunked
    prefill does.
    ``decode`` (inherited) is the cross-backend oracle identity threaded into the
    engine by ``build_engine``; the remaining fields are vLLM tuning knobs -- all
    defaulted so a bare ``VLLMReference(decode=Greedy())`` works; a task sets them
    when a model needs help fitting. Recorded in provenance via ``dump_spec`` AND
    folded into the reference fingerprint through the same payload's hash."""

    engine_cls: ClassVar[type[Engine]] = VLLMEngine
    gpu_memory_utilization: float = 0.9   # fraction of GPU mem vLLM reserves (weights + KV cache)
    max_model_len: int | None = None      # max context length; None -> vLLM infers from the model
    dtype: str = "auto"                   # "auto"|"bfloat16"|"float16"|"float32"; auto = model default
    enforce_eager: bool = False           # disable CUDA-graph capture (slower decode, faster startup)
    batch_invariant: bool = False         # if true, engine sets VLLM_BATCH_INVARIANT=1 before LLM init
    #: Cards the weights are split across. It also sizes the slot: the engine asks for that many
    #: GPUs. It is the reference's own size, not read off the task's hardware, because the
    #: reference is not the deployment being measured. Left out of the dump at one card, the
    #: width of every reference declared before the field, so their identities stay as they were.
    tensor_parallel_size: int = Field(default=1, ge=1, exclude_if=lambda n: n == 1)

    def build_engine(self, task: "TaskSpec"):
        # gpu_kind: pragmatic default, NOT task semantics -- see ReferenceEngine.__init__.
        return self.engine_cls(
            model_spec=task.model, decode=self.decode,
            gpu_kind=task.hardware.accelerator.kind, **contract_conditioning(task),
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len, dtype=self.dtype,
            enforce_eager=self.enforce_eager, batch_invariant=self.batch_invariant,
            tensor_parallel_size=self.tensor_parallel_size)


class SGLangReference(Reference):
    """SGLang-backed reference oracle. Generate-only, greedy, trusted-engine pool
    member for ``hls calibrate`` -- NOT a
    task-facing verifier reference (TokenFaithfulnessVerifier requires
    TeacherForce + StackReport, which SGLangEngine does not provide). ``decode``
    (inherited) is the cross-backend oracle identity threaded into the engine by
    ``build_engine``. Only the two knobs SGLangEngine genuinely needs are exposed
    here -- no speculative knobs (YAGNI)."""

    engine_cls: ClassVar[type[Engine]] = SGLangEngine
    mem_fraction_static: float = 0.85   # fraction of GPU mem sglang reserves (weights + KV cache)
    max_model_len: int | None = None    # max context length; None -> sglang infers from the model

    def build_engine(self, task: "TaskSpec"):
        # gpu_kind: pragmatic default, NOT task semantics -- see ReferenceEngine.__init__.
        return self.engine_cls(
            model_spec=task.model, decode=self.decode,
            gpu_kind=task.hardware.accelerator.kind, **contract_conditioning(task),
            mem_fraction_static=self.mem_fraction_static,
            max_model_len=self.max_model_len)
