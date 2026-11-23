"""hls.calibrate -- the offline calibration library behind ``hls calibrate``.

The token-level gate has two rules, both measured rather than authored: a rate
ceiling on how often a served token may disagree with the reference, and
per-gap-level flip-count ceilings on how many large disagreements a trace may
carry. Both are properties of the bf16 numerical noise floor of the reference
weights under this task's traffic. This module measures them and writes a
``calibration.json``.

The measurement compares independent trusted serving stacks against a single
pinned reference -- HF with the MATH attention kernel, the exact config the
live verifier deploys.

**A serving is a clocked drive.** The unit of measurement is one serving: the
task's own traffic generator -- clocked arrivals at the declared rate,
think-times, ramp-down trimming, live self-accumulated chat histories --
driven against a pool engine. Flip statistics are a property of the serving
dynamics the traffic induces, so the calibration side and the grading side
must induce the same ones; a saturated offline batch measures a different
regime (clean engines flip more often under a saturated batch than under the
drive, which would leave the deployed multiplier looser than the declared
one). Each serving is clocked with its own seed, so arrivals,
think-times and turn membership differ between servings while the workload
identity does not.

The drive runs inside the member's own process, submitting each request
directly to the engine's asynchronous Python API as it comes due
(``ClockedServing``, implemented by ``VLLMEngine`` and ``SGLangEngine``): the
loop itself is the harness's own driver, the one ``hls.spec.drivers.driver_for``
selects for the task, so the timing is grading's timing, with no wire protocol
or network hop between the schedule and the engine. A task that declares an
agent seam (``TaskSpec.agent``) is therefore served as trajectories, its
framework choosing each turn and its tools running inside the member -- which is
why an agentic member deploys unsealed, on its measured image, which carries the
tools' packages. Both APIs return the emitted token ids natively, which is what
the pinned reference teacher-forces.

That definition decides the pool: only stacks with an async per-request API
can take traffic on a clock, so the pool is vLLM and SGLang
(``POOL_ENGINE_NAMES``). The batch-oriented HF engines are still central to
the procedure -- one is the pinned reference that scores every serving,
another the fp32 preflight anchor -- but they cannot be pool members.

Each serving runs three phases: capture (the member drives the traffic and
records each reply's emitted token ids, against the exact conversation the
request carried and under the output cap the harness stamped on it), score
(teacher-force the capture through the pinned reference into per-position
log-probability gaps, counted through the shared measurement the deployed
verifier uses, ``hls.verify.flips``, into one persisted record per reply, from
which the serving's tally, ``DrawStats``, is read), and aggregate (pure CPU: per-engine rate
mean/spread, per-level event counts and their Garwood upper limits, the
recurrence census, and the sizing verdicts).

**Every reply is counted, on both sides.** Nothing a served reply reports about
itself keeps it out of the measurement; the reply's reported finish reason is
not read. A reply whose served length is past the cap stamped on its request is
measured by neither side: a token-level campaign refuses the pool member that
served it (``refuse_past_cap``), a report campaign records the reply as a
contract violation and leaves it out, and the verifier fails the trace on it.
A checkpoint with a greedy non-termination mode still produces replies
that run to the output cap, and how many a serving happens to contain is a
property of the draw that moves its flip rate without saying anything about the
server. A task removes those requests from the workload itself
(``DataSource.removed_rows``): ``capped_rows`` reads a trusted member's
captures and names every session whose reply reached the harness-stamped cap,
the task lists them, and neither the campaign nor a graded run serves them
again. The list is derived from trusted captures alone -- served length against
the stamped cap, never a reported label -- so nothing a contestant returns can
move it.

**How the campaign sizes itself.** One constant does it: the tail tolerance
``mu_per_mtok`` (expected flips per million graded tokens at gap levels
calibration never observed) fixes a per-engine exposure target
``exposure_target(mu_per_mtok)``, about 12.3M tokens at the default. Each
engine serves until it has that much teacher-forced exposure and at least
``MIN_SERVINGS`` servings -- a token counter against a constant, no looks, no
pilot, so the realized serving count is independent of the flip-rate noise
and the precision check (P) that follows is a single look at a pre-committed
sample size. (P) failing refuses the campaign, naming each failing engine and
both sides of its inequality: more servings shrink the confidence factor on
the spread, never the spread itself, so a second look would only spend the
bound's error budget for a number that has already been measured.

**The campaign is a stage pipeline.** ``hls calibrate`` plans it
(``hls.pipeline.calibration``), and its roles run the phases with this module's
functions (``hls.verify.calibration_roles``): the instrument check, each serving's
capture, a reference load per share of captures, and the assembly. Every capture and
every scored measurement is a published output laid out as the capture store lays it
out, keyed by workload fingerprint, engine and serving index; a failed stage is retried
by the orchestrator, and the servings the exposure counter still owes are appended to
the plan once every planned serving is scored (``servings_owed``).

Before trusting any of it, a preflight checks the instrument itself: an fp32
anchor generates a handful of committed probes and the pinned bf16 reference
teacher-forces them. fp32-vs-bf16 disagreement should be tiny (benign
rounding); a large disagreement (``PREFLIGHT_MAX_*``) means the reference is
numerically broken (eager-attention 33-nat gaps, zeroed attention heads) and
the whole calibration is refused -- fail loud, never emit a calibration
computed on a defective instrument.

Deployment goes through one injected seam so this module is unit-testable
without a GPU: a ``Deploy`` loads the pinned reference, the fp32 anchor and
every pool member alike, and the roles build it over an engine host
(``hls.verify.inprocess.EngineHost``), on which each engine loads in the role's own
process, on the role's own image.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import re
import statistics
from contextlib import contextmanager
from itertools import zip_longest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (Any, Awaitable, Callable, Iterable, Literal, Mapping, Protocol, Sequence,
                    TYPE_CHECKING, get_args)

from pydantic import BaseModel

from hls.core.decode import DecodeConfig
from hls.core.payloads import Envelope, TextPayload, TokenLogprob, TopLogprob
from hls.core.trace import ArtifactResponse, event_key
from hls.provision.rows import tenant_subpath
from hls.reference.engines import (HFEngine, SGLangEngine, VLLMEngine,
                                   contract_conditioning)
from hls.reference.references import Reference
from hls.spec.datasets import Split
from hls.spec.drivers import stamp_request, user_message
from hls.spec.taskspec import TaskSpec
from hls.spec.workload import CUT_EXTRA
from hls.verify.calibration import (
    CALIBRATION_FILE,
    UPPER_Z,
    Calibration,
    CalibrationError,
    CappedText,
    CurveSide,
    DrawSpec,
    EndOfTurnCheck,
    EndOfTurnEngine,
    EngineCurve,
    EngineRate,
    EngineRecord,
    EngineReport,
    LevelStats,
    PreflightResult,
    ReportCalibration,
    Sizing,
    StackInfo,
    StopReason,
    Tolerances,
    calibration_kind,
    device_matches_kind,
    precision_check,
    tail_check,
    workload_fingerprint,
    workload_payload,
)

if TYPE_CHECKING:
    from hls.verify.verifiers import ReportFaithfulnessVerifier
from hls.verify.flips import ReplyFlips, candidate_rungs, past_cap, score_reply
from hls.verify.poisson import chi2_quantile, garwood_upper
from hls.util import gzip_text_writer
from hls.verify.report import EOT_EPS_GRID, PooledReport, ReplyReport, ReportRows

# ---------------------------------------------------------------------------
# Tunables (module constants, documented so a reader sees why the number is that)
# ---------------------------------------------------------------------------

#: One-sided false-coverage probability of the upper bound (P) puts on an engine's
#: between-serving spread: ``c_M`` is a 95% one-sided bound on the true standard
#: deviation. Deliberately its own number rather than ``GARWOOD_ALPHA``: that one is
#: the coverage of the deployed per-level ceilings, this one is the confidence with
#: which the campaign claims its own sample size was adequate. Dimensionless.
SPREAD_ALPHA: float = 0.05

#: Default tolerances (all overridable per run and recorded in the artifact).
#:
#: ``EPS`` (P): one graded run's flip rate must be reproducible to 5% relative. That
#: keeps the measurement noise an order below the fidelity margin the deployed gate
#: leaves, so a faithful server never fails for having drawn an unlucky serving. It is
#: calibration's own budget: the deployed headroom is the benchmark's resolution dial,
#: never an error margin to spend calibration noise against.
#: ``MU_PER_MTOK`` (T): a flip at a gap level the calibration never observed is
#: tolerated at an expected 0.3 occurrences per million graded tokens. Per token, not
#: per run: a longer graded trace then receives a proportionally larger allowance, and
#: the exposure the campaign must reach is one constant for every task.
#: ``DELTA``: family-wise false-alarm budget per graded trace, split by Bonferroni
#: over the gap rungs plus the rate rule. 1e-3 keeps the whole gate's false-rejection
#: probability below one in a thousand clean runs.
DEFAULT_EPS: float = 0.05
DEFAULT_MU_PER_MTOK: float = 0.3
DEFAULT_DELTA: float = 1e-3

#: The fewest servings an engine may certify on, whatever its exposure. The exposure
#: target alone decides campaign size; this is an engineering floor under the spread
#: estimate that (P) is computed from, since a standard deviation over two or three
#: samples is barely an estimate.
MIN_SERVINGS: int = 8

#: Servings per engine a report calibration takes when the caller names no count.
#: The sampling ceiling is a mean over positions with a reply-clustered standard
#: error, so one serving of the task's own traffic already carries hundreds of
#: independent clusters and its own precision check says whether that was enough --
#: unlike the flip ceilings, whose between-serving spread cannot be estimated from one
#: serving at all. A campaign that fails the check is told how many servings to ask
#: for (see ``assemble_report``).
DEFAULT_REPORT_SERVINGS: int = 1

#: Preflight sanity bounds on fp32-anchor-vs-pinned-bf16 disagreement. They catch the
#: catastrophic instrument-defect class (eager attention's ~33-nat gaps, zeroed
#: attention heads -- a reference that is wrong) and sit above bf16 rounding; the
#: per-task envelope is the calibrated rate ceiling ``q_max`` and the per-level
#: flip-count ceilings, not this preflight. Units: flip rates are fractions of
#: teacher-forced tokens; gap bounds are nats (per-token log-probability differences --
#: ``log_softmax`` is natural-log).
PREFLIGHT_MAX_FLIP_RATE: float = 0.02
PREFLIGHT_MAX_GAP: float = 1.0

#: Trust bound on a pool member's own noise: the flip rate at which its disagreement
#: with the pinned reference stops looking like bf16 rounding and starts looking like a
#: defective stack (the cuDNN GQA-decode zeroing class). Same magnitude and unit
#: (fraction of teacher-forced tokens) as the preflight rate bound, and for the same
#: reason: it catches the instrument-is-wrong class, not benign rounding, which sits
#: far below.
#:
#: There is no companion bound on gap size. The gate bounds how many large-gap flips a
#: trace carries, not how large any one of them may be, precisely because a clean pool
#: produces rare multi-nat events of its own -- so a gap bound tight enough to mean
#: anything flags healthy members instead of defective ones.
#:
#: Advisory, not a gate: tripping the bound warns when the calibration is measured
#: (``pool_warnings``); the member stays in the pool and its servings are in the
#: ceilings. Refusing would discard every already-measured member of an
#: otherwise-clean pool. The warning names the member for review, and
#: ``Calibration.rate_side`` keeps every flagged member's stats, so recomputing the
#: ceilings without one is arithmetic, not a re-run.
MEMBER_MAX_TRUSTED_FLIP_RATE: float = 0.02

#: Cross-member outlier warning (softer than the absolute trust bound above): warn when
#: the ceiling-setting member's mean flip rate exceeds this multiple of the
#: next-highest member's. The absolute bound catches defect-scale noise; this one names
#: a member under it that sets q_max alone. 2x matches the gate's own tolerance
#: grammar. Warning only, never a refusal.
MEMBER_OUTLIER_RATIO: float = 2.0

#: A severe-event site (one prompt at one gap level) is flagged as recurring when it
#: fires in at least this fraction of an engine's servings. Half is well above the
#: handful of coincidental repeats a Poisson process produces over ~10 servings, and far
#: below the "every serving" a genuinely deterministic site would show.
RECURRENCE_FRACTION: float = 0.5

log = logging.getLogger(__name__)

#: Decoding budget stamped on every preflight probe. Short: the probes exist to catch
#: gross numeric breakage, not to profile long generations.
PREFLIGHT_MAX_TOKENS: int = 64

#: Replies named in the refusal for a member whose divergence is not finite. Enough to
#: go and look at, few enough that a serving where every reply is affected still leaves
#: the sentence readable; the count beside them says how many there were in total.
_UNBOUNDED_NAMED: int = 5


# A single ~2k-token synthetic-but-coherent long-context passage + question. Committed
# in-module (no external file) so the preflight is reproducible anywhere the code runs.
# The document is fictional; the point is a realistic long attention context, not facts.
_LONG_CONTEXT_PROBE = (
    "Read the following passage carefully and then answer the question at the end.\n\n"
    "The Meridian Archive occupies a terraced hillside above the town of Callindra, on "
    "the eastern shore of Lake Ossir. It was founded in the year 1418 of the local "
    "reckoning by the cartographer Deval Anse, who had spent three decades charting the "
    "braided river deltas of the southern coast and wished to preserve his survey rolls "
    "against the damp that had already ruined two earlier collections. Anse chose the "
    "hillside deliberately: the prevailing wind off the lake is dry in every season but "
    "the brief autumn rains, and the limestone terraces drain quickly, so that even the "
    "lowest vaults stay well below the moisture that spoils parchment. The first "
    "structure was a single long hall of pale stone, its windows set high and narrow to "
    "keep out direct sun, and its floor laid with a double course of slate over a bed "
    "of crushed charcoal that the builders believed would draw off any lingering damp.\n\n"
    "In the century after Anse's death the Archive grew by accretion rather than plan. "
    "Each successive keeper added a wing, a stair, or a set of vaults according to the "
    "needs and the fashions of the day, so that the building came to resemble a small "
    "town folded in upon itself. The most ambitious of these keepers was Sera Iolanthe, "
    "who held the post for forty-one years and who is remembered chiefly for the "
    "Cylinder Room, a circular chamber whose walls are lined from floor to ceiling with "
    "shallow drawers, each drawer sized to hold a single rolled map. Iolanthe also "
    "established the practice, still followed, of copying every incoming document twice: "
    "once onto durable vellum for the permanent vaults, and once onto cheap rag paper "
    "for the reading rooms, so that the originals need never be handled by visitors. It "
    "is said that she could recite the shelf-location of any of the Archive's holdings "
    "from memory, though the catalogue she left behind suggests she relied on a system "
    "of coloured tabs at least as much as on recollection.\n\n"
    "The Archive's holdings are not limited to maps. Over the generations it has taken "
    "in ships' logs, tide tables, land-grant deeds, the correspondence of three minor "
    "royal houses, and a large but poorly sorted collection of botanical drawings "
    "donated by the physician Corven Ashe. The maps remain the heart of the collection, "
    "however, and scholars travel from across the region to consult them. Among the most "
    "prized is the so-called Drowned Chart, a survey of the Ossir shoreline made before "
    "a series of floods raised the lake by nearly two meters and submerged the old "
    "harbor of Callindra entirely. Because no other record of the drowned harbor "
    "survives, the Drowned Chart is the only means by which the town's original "
    "foundations can be located, and it has been consulted by every engineer who has "
    "attempted to build a new quay along the modern waterline.\n\n"
    "Access to the Archive has always been governed by a careful and sometimes contested "
    "set of rules. In Anse's day any literate person could enter the long hall and read, "
    "provided they left a small token of payment in a bowl by the door. Iolanthe, worried "
    "by the wear that unrestricted handling caused, introduced the reading-room copies "
    "and confined visitors to them. Later keepers, facing a steady rise in the number of "
    "scholars, established a system of letters of introduction, so that a would-be reader "
    "must now present a recommendation from a recognized institution before being "
    "admitted to the more valuable holdings. Critics of this system argue that it favors "
    "the already-connected and shuts out independent researchers; its defenders reply "
    "that the Archive has survived six hundred years precisely because it has been "
    "cautious about who touches its most fragile documents. The current keeper, Malen "
    "Dorr, has proposed a compromise: a program of careful photographic reproduction that "
    "would let the images of the maps circulate freely while the originals stay sealed in "
    "the vaults. Whether the town council will fund the necessary equipment remains, at "
    "the time of writing, an open question.\n\n"
    "Question: According to the passage, why is the Drowned Chart considered "
    "irreplaceable, and what practical use has it been put to?"
)

#: Eight committed preflight probes spanning short factual, arithmetic, code, a
#: multi-step word problem, and one long-context passage-plus-question. They exercise a
#: mix of decoding regimes so a defect that only shows on, say, long contexts still
#: trips the preflight. Held in-module (no external fixtures) for reproducibility.
PREFLIGHT_PROBES: list[str] = [
    "What is the capital of Japan?",
    "Name the three primary additive colors of light.",
    "Compute 37 times 24, and show the result.",
    "If a train travels 60 kilometers in 45 minutes, what is its average speed in "
    "kilometers per hour? Explain your reasoning step by step.",
    "A shop sells apples at 3 for 2 dollars. Priya buys 12 apples and pays with a "
    "20-dollar bill. How much change does she receive? Work through it step by step.",
    "Write a Python function `is_palindrome(s)` that returns True if the string `s` "
    "reads the same forwards and backwards, ignoring case and spaces.",
    "Explain in two or three sentences why the sky appears blue during the day.",
    _LONG_CONTEXT_PROBE,
]


# ---------------------------------------------------------------------------
# The pool: which engines can be members
# ---------------------------------------------------------------------------

#: A pool member: a serving stack this module builds (``member_engines``), by the name
#: ``--engines`` takes. Serving stacks only -- a pool member must take clocked traffic one
#: request at a time on an asynchronous API (``ClockedServing``), which is what makes a
#: serving a serving. Independent implementations of the same weights, on independently
#: resolved wheel sets, is what the pool is for. Every other list of the members' names
#: (``POOL_ENGINE_NAMES``, the image catalog's member roles, the calibration grower's stage
#: names) is derived from this one.
PoolEngine = Literal["vllm", "sglang"]

#: The pool members' names, in recommended pool order.
POOL_ENGINE_NAMES: tuple[str, ...] = get_args(PoolEngine)

#: The batch-oriented HF engine configurations. They are not pool members -- their
#: serving face is one padded batch per call, which cannot be driven on a clock --
#: but they remain the instruments of the procedure: ``hf_pinned`` is the pinned
#: reference that teacher-forces every serving, and an fp32 variant of it is the
#: preflight anchor. Named here so a request for one gets an explanation instead of
#: "unknown engine".
OFFLINE_ENGINE_NAMES: tuple[str, ...] = ("hf_pinned", "hf_default", "hf_flash_attn")


def audio_replies(task: TaskSpec) -> bool:
    """Whether this task's servers answer with audio rather than text, which
    ``hls calibrate`` refuses (``check_campaign``): no role can drive a TTS serving
    stack in process. Read off the contract's declared output modalities, which is the
    same declaration ``check_wiring`` holds every verifier to."""
    return "audio" in frozenset(getattr(task.contract, "output_modalities", frozenset()))


def validate_pool(names: list[str]) -> None:
    """Refuse a pool that cannot serve a task's traffic on a clock, naming the reason for
    each rejected name. Called by ``check_campaign`` when a calibration is planned."""
    offline = [n for n in names if n in OFFLINE_ENGINE_NAMES]
    if offline:
        raise CalibrationError(
            f"{sorted(offline)} cannot be calibration pool members: a serving is a "
            f"clocked drive of the task's traffic -- requests submitted one at a time, "
            f"as they come due, to an engine's asynchronous serving API -- and these "
            f"are batch-oriented engines whose serving face is one padded batch per "
            f"call. The pool is the serving stacks {sorted(POOL_ENGINE_NAMES)}. (The "
            f"pinned HF reference that scores every serving, and the fp32 preflight "
            f"anchor, are unaffected: they are the procedure's instruments, not pool "
            f"members.)")
    unknown = [n for n in names if n not in POOL_ENGINE_NAMES]
    if unknown:
        raise CalibrationError(
            f"unknown calibration pool engine(s) {sorted(unknown)}; valid names are "
            f"{sorted(POOL_ENGINE_NAMES)}. Fix --engines and re-run `uv run hls calibrate`.")


def _calibrated_references(task: TaskSpec) -> list[Reference]:
    """The references a calibration run has to agree on: the ones the task's calibrated
    verifiers declare.

    Found the way ``hls.verify.calibration.workload_payload`` finds the references it
    records, so what the ceilings are measured under is what the fingerprint they are
    keyed by was computed from."""
    refs = [v.reference for t in task.effective_tenants() for v in t.verifiers
            if getattr(v, "needs_calibration", False)]
    if not refs:
        raise CalibrationError(
            "this task declares no calibrated verifier, so there is no reference to pin "
            "the calibration pool to. A calibration measures the noise floor of one "
            "verifier's reference; add a TokenFaithfulnessVerifier before calibrating.")
    return refs


def _task_decode(task: TaskSpec) -> DecodeConfig:
    """The decode every engine of a calibration run is pinned to: the one the task's
    calibrated verifier declares on its reference. A task whose calibrated verifiers
    disagree on the decode has no single noise floor to measure; refuse it."""
    decodes = {r.decode for r in _calibrated_references(task)}
    if len(decodes) > 1:
        raise CalibrationError(
            f"this task's calibrated verifiers declare {len(decodes)} different decode "
            f"configs ({sorted(d.fingerprint() for d in decodes)}); each would produce "
            f"its own flip rates, so there is no single noise floor to measure. Pin one "
            f"decode across the task's calibrated verifiers and re-run.")
    return decodes.pop()


def _task_query_chunk(task: TaskSpec) -> int:
    """The teacher-forcing chunk the run's HF engines are pinned to: the one the task's
    calibrated verifier declares on its reference (``HFReference.query_chunk``).

    The chunk is a numeric property of the instrument, not just a memory knob, so the
    ceilings have to be measured at the chunk the graded gate will run at. Two
    calibrated verifiers declaring different chunks are two instruments with no single
    noise floor between them; refuse rather than pick one."""
    chunks = {r.query_chunk for r in _calibrated_references(task)}
    if len(chunks) > 1:
        raise CalibrationError(
            f"this task's calibrated verifiers declare {len(chunks)} different "
            f"teacher-forcing query chunks ({sorted(chunks)}); the math kernel tiles "
            f"differently at each, so there is no single noise floor to measure. Pin "
            f"one query_chunk across the task's calibrated verifiers and re-run.")
    return chunks.pop()


def member_internet(task: TaskSpec) -> bool:
    """Whether a pool member serving ``task`` runs with its network open: only for a task
    whose agent runs tools that call the network themselves, which is one that provisions no
    tool assets (``AgentSpec.assets``). Every other member is sealed with no egress and reads
    the hub offline, as a graded artifact is served, so it measures the stack it loads from
    the provisioned weights and nothing it could fetch. The plan records the answer
    (``CampaignSettings.member_internet``) and each capture stage seals its role by it."""
    agent = task.agent
    return agent is not None and agent.assets is None


def member_engines(task: TaskSpec) -> dict[str, Callable[[], Any]]:
    """Name -> zero-arg builder for each pool member.

    Text tasks: both members run on the task's model (``task.model``), the task's own
    decode (``_task_decode``), the task's accelerator kind, the context window it serves
    in (``task.max_context_len``) and the contract's conditioning and stop ids
    (``contract_conditioning``). Both are independent serving stacks with an
    asynchronous per-request API, which is what lets them take the task's traffic on a
    clock (``ClockedServing``). A member's weights load when the role that serves it acquires
    it on its in-process engine host (``hls.verify.calibration_roles.in_process``), within
    that role's load bound; ``clocked_serving=True`` makes the vLLM load build the async
    serving engine rather than the oracle's synchronous one (the two cannot coexist on one
    card)."""
    model = task.model
    gpu_kind = task.hardware.accelerator.kind
    decode = _task_decode(task)
    # A member runs the task's tools itself. Tools that load a provisioned bundle get the
    # bundle's memory; whether the member's container is sealed is ``member_internet``'s.
    #
    # Memory only, never a card, even for a bundle whose tools would use one: the
    # serving engine holds this member's card, so the tools encode on CPU here, with the
    # same weights and recipe the graded driver runs on its own GPU.
    agent = task.agent
    memory_mb = agent.assets.memory_request() if agent is not None and agent.assets is not None else None
    # Both members serve at the task's context window. An agentic framework decides
    # when a trajectory must stop researching and answer by comparing its prompt against
    # the window the reply reports, so a member left at its checkpoint's own context
    # would serve conversations no graded run produces.
    max_context_len = task.max_context_len
    # How the prompt reaches the model and which ids end a turn, read off the task's
    # own contract, so a member stops where the graded server stopped.
    from_contract = contract_conditioning(task)
    builders = {
        "vllm": lambda: VLLMEngine(model_spec=model, decode=decode, gpu_kind=gpu_kind,
                                   memory_mb=memory_mb, max_model_len=max_context_len,
                                   clocked_serving=True, **from_contract),
        # 0.92 of the card for weights plus KV pool. SGLang admits a prompt only if
        # the full-attention pool holds it, and by default it charges 0.8 of every
        # pooled token to the sliding-window layers, which never hold more than their
        # window per request. For Gemma 4 31B on an H200 (10 full and 50 sliding
        # layers, window 1,024) that leaves a 99,535-token pool at 0.92, short of the
        # deep-research workload's longest prompts (118,784 tokens with the output cap);
        # at a 0.25 share the pool is 255,947 tokens. Both knobs change capacity, never
        # numerics.
        "sglang": lambda: SGLangEngine(model_spec=model, decode=decode, gpu_kind=gpu_kind,
                                       memory_mb=memory_mb, mem_fraction_static=0.92,
                                       max_model_len=max_context_len,
                                       swa_full_tokens_ratio=0.25, **from_contract),
    }
    # The builder keys are the public name set; a drift would let a plan accept a name
    # the roles can't build (or vice versa).
    assert tuple(builders) == POOL_ENGINE_NAMES, (builders.keys(), POOL_ENGINE_NAMES)
    return builders


def engine_cache_config(built: Any) -> str:
    """What the compile and kernel caches of the engine *built* (built, never loaded) depend on
    beside its image and its GPU kind, as one canonical JSON line: its class and every public
    setting it holds (``vars``), a model or a decode as it dumps. Engines whose lines differ get
    engine caches of their own (``hls.stages.engine_cache.engine_cache_key``). Raise TypeError for a
    setting JSON cannot hold, so an engine that grows one is refused at plan time."""
    settings = {name: value.model_dump(mode="json") if isinstance(value, BaseModel) else value
                for name, value in vars(built).items() if not name.startswith("_")}
    return json.dumps({"engine": type(built).__name__, "settings": settings}, sort_keys=True)


# ---------------------------------------------------------------------------
# The sessions a serving drives
# ---------------------------------------------------------------------------

def load_sessions(task: TaskSpec, rows: Path, split: Split) -> list[tuple]:
    """The sessions one serving drives, as ``[(tenant, items)]`` -- the same shape
    ``hls.spec.drivers.load_mounted_sessions`` builds for a graded run's drive (the
    deploy_drive stage's driver role), read from the provisioned *rows*: the rows entry a
    calibration role mounts under ``/data`` (the mount wall's role ``calibration``), each
    tenant's split at ``tenants/<id>/<split>/rows.json`` (``tenant_subpath``), where the
    driver's mount puts it, so calibration reads the rows a graded run would.

    The whole split, removal list not yet applied: the drive loop drops each tenant's
    ``DataSource.removed_rows`` before it draws the schedule
    (``hls.spec.drivers.remove_listed_rows``), for a serving exactly as for a graded
    run. The whole split is also what a tool deriving that list from an earlier
    campaign's captures keys them against, since those captures may name rows the
    list has since removed.

    One split, not the whole dataset: a serving is a clocked drive, and the arrival
    schedule is drawn for the session count of the split being served, so mixing
    splits would serve a workload no run ever offers. The default is the split grading
    serves.

    A tenant with no data source contributes no sessions, exactly as in the drive
    loop. A task where that leaves nothing is refused: a ceiling measured on an empty
    pool is meaningless.

    Each tenant is stripped of its verifiers: a pool member only generates traffic, and
    correctness is measured by teacher-forcing its capture, so nothing in the serving
    reads them."""
    loaded: list[tuple] = []
    for tenant in task.effective_tenants():
        if tenant.data_source is None:
            continue
        try:
            items = list(tenant.data_source.dataset.items_at(rows / tenant_subpath(tenant.id), split))
        except FileNotFoundError as exc:
            raise CalibrationError(
                f"calibration cannot read the {split.value!r} split for tenant "
                f"{tenant.id!r}: {exc}. The calibration's provision stage provisions the "
                f"rows the role mounts; this one has none for that tenant and split.") from exc
        if items:
            loaded.append((tenant.model_copy(update={"verifiers": []}), items))
    if not loaded:
        raise CalibrationError(
            f"the task rendered no {split.value!r} traffic to calibrate on: no effective "
            f"tenant has a data source with provisioned rows for that split")
    return loaded


# ---------------------------------------------------------------------------
# Deployment seam
# ---------------------------------------------------------------------------

class Deploy(Protocol):
    """``(name, engine) -> (proxy, release)``: the one deploy seam, for the pinned
    reference, the fp32 preflight anchor and every pool member alike. The calibration
    roles build it over an engine host (``hls.verify.calibration_roles.in_process``), on
    which each engine loads in the role's own process, on the role's own image: a member
    on its measured image, the reference and the anchor on the reference's. Tests inject a
    fake that loads nothing."""

    async def __call__(self, name: str, engine: Any) -> tuple[Any, Callable[[], Awaitable[None]]]:
        ...


# ---------------------------------------------------------------------------
# Sizing arithmetic
# ---------------------------------------------------------------------------

def _c_m(m: int) -> float:
    """The upper-confidence factor on a spread estimated from ``m`` servings:
    ``sqrt((m-1) / chi2_{m-1, SPREAD_ALPHA})``, the one-sided upper bound on the true sd.

    This is the only thing more servings buy the precision check. The population spread
    itself does not shrink with servings -- it is a property of the engine and the
    traffic -- so a campaign that fails (P) with a small ``c_M`` is telling you the
    engine is genuinely that variable, and no amount of serving will fix it."""
    if m < 2:
        raise CalibrationError(f"c_M is undefined below 2 servings (got {m})")
    return math.sqrt((m - 1) / chi2_quantile(SPREAD_ALPHA, m - 1))


# ---------------------------------------------------------------------------
# One serving's scored measurement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DrawStats:
    """One serving's tally: every reply the serving produced, reduced to what the
    aggregation needs, with the per-reply records it was reduced from.

    Never persisted on its own: the store keeps the records and derives this from them
    on read (``CaptureStore.read_scored``), so no total exists in two places.
    ``n_items`` counts the teacher-forced replies the serving produced (one per served
    turn, so a multi-turn session contributes several); ``n_tokens`` and ``n_flips``
    are over every one of them.

    Counts on the severity grid are not stored: the grid is chosen from the whole pool's
    counts after every serving is scored (``pool_grid``), so ``level_events`` and
    ``severe_sites`` take the grid and read it off the records.

    ``capped_events``/``capped_tokens`` record how many replies reached the request's
    output cap and how many positions they contributed; they describe the serving and
    are not subtracted from anything here.
    """

    engine: str
    draw: int
    seed: int
    #: The serving package version the member reported while it produced this serving.
    serving_version: str
    n_items: int
    n_tokens: int
    n_flips: int
    max_gap: float
    capped_events: int
    capped_tokens: int
    events: tuple[ReplyFlips, ...]

    def header(self) -> dict:
        """The serving's identity, as the scored file's first line
        (``CaptureStore.write_scored``): what the per-reply records cannot carry."""
        return {"engine": self.engine, "draw": self.draw, "seed": self.seed,
                "serving_version": self.serving_version}

    @property
    def flip_rate(self) -> float:
        """Flips over tokens (0.0 for an empty serving)."""
        return self.n_flips / self.n_tokens if self.n_tokens else 0.0

    def level_events(self, levels: Sequence[float]) -> list[int]:
        """Flips with gap at or above each rung of ``levels``, nested (non-increasing)."""
        totals = [0] * len(levels)
        for ev in self.events:
            for i, n in enumerate(ev.level_counts(levels)):
                totals[i] += n
        return totals

    def severe_sites(self, levels: Sequence[float]) -> list[tuple[str, ...]]:
        """Per rung of ``levels``, the event keys with at least one flip at or above
        it -- the census the recurrence check pools across servings to confirm that
        severe events are not a property of particular prompts."""
        sites: list[set[str]] = [set() for _ in levels]
        for ev in self.events:
            for i, n in enumerate(ev.level_counts(levels)):
                if n:
                    sites[i].add(ev.event_key)
        return [tuple(sorted(x)) for x in sites]

    @classmethod
    def from_events(cls, engine: str, draw: int, seed: int, serving_version: str,
                    events: Sequence[ReplyFlips]) -> "DrawStats":
        """Reduce a serving's per-reply records to its tally."""
        return cls(engine=engine, draw=draw, seed=seed,
                   serving_version=serving_version, n_items=len(events),
                   n_tokens=sum(ev.n_tokens for ev in events),
                   n_flips=sum(ev.n_flips for ev in events),
                   max_gap=max((ev.max_gap for ev in events), default=0.0),
                   capped_events=sum(1 for ev in events if ev.at_cap),
                   capped_tokens=sum(ev.n_tokens for ev in events if ev.at_cap),
                   events=tuple(events))


def pool_grid(stats: Mapping[str, Sequence[DrawStats]]) -> list[float]:
    """The severity grid for a pool, chosen from its own counts.

    At each rung the deployed ceiling is a function of the binding clean count -- the
    largest, over engines, of that engine's pooled flips with gap at or above the rung
    (grading is against the most permissive engine). Both clean and graded counts are
    nested along the lattice, so two rungs with the same binding count have the same
    ceiling and a trace passing one passes the other: only rungs where the count
    strictly drops carry information. The grid is therefore the floor, every lattice
    rung whose binding count is below the previous kept rung's, and the first rung with
    no clean event at all (the tightest ceiling, for gaps no clean engine produced).
    Recorded in the artifact, so grading reads the same rungs."""
    worst = max((d.max_gap for draws in stats.values() for d in draws), default=0.0)
    rungs = candidate_rungs(worst)
    binding = [0] * len(rungs)
    for draws in stats.values():
        totals = [0] * len(rungs)
        for d in draws:
            for i, n in enumerate(d.level_events(rungs)):
                totals[i] += n
        binding = [max(b, t) for b, t in zip(binding, totals)]
    grid = [rungs[0]]
    last = binding[0]
    for c, b in zip(rungs[1:], binding[1:]):
        if b < last:
            grid.append(c)
            last = b
        if b == 0:
            break
    return grid


def refuse_past_cap(engine: str, draw: int, items: Sequence[Envelope]) -> None:
    """Refuse (``CalibrationError``) serving *draw* of pool engine *engine* when one of its
    replies is longer than the output cap the harness stamped on its request, by the test
    the deployed verifier fails a trace on (``hls.verify.flips.past_cap``).

    The verifier never measures such a reply, so ceilings that counted one would be
    measured on text no graded trace is read against. The refusal names the first such
    reply: the engine, the serving, the request and its tenant, its length and the cap."""
    for item in items:
        n, cap = served_length(item), item_cap(item)
        if past_cap(n, cap):
            raise CalibrationError(
                f"pool engine {engine!r} served {n} tokens for request "
                f"{item.request_id!r} of tenant {item.metadata['tenant_id']!r} on serving "
                f"{draw}, past the {cap}-token cap the harness stamped on it; the member "
                f"did not honor the cap, so the serving holds text no graded trace is "
                f"measured on. Fix the stack and re-run.")


def score_draw(engine: str, draw: int, seed: int, serving_version: str,
               items: list[Envelope], gaps_per_item: list[list[float]]) -> DrawStats:
    """Count one serving's per-item per-token greedy gaps into a ``DrawStats``.

    The measurement itself is ``hls.verify.flips`` -- the same code the deployed verifier
    grades with, so "a flip", "a severe event" and "at the cap" mean exactly one thing
    on both sides of the gate. Every item is counted, and a serving that holds a reply
    past its request's cap is refused whole (``refuse_past_cap``).

    Any non-finite gap refuses the whole calibration (``CalibrationError``): every gap
    here comes from teacher-forcing a trusted engine's own reply, so NaN means the
    instrument's arithmetic is broken and +inf means the engine emitted an id outside
    the reference's vocabulary -- either way the ceilings would be measured on a
    defective setup. NaN especially must not pass silently: it loses every ``>``
    comparison, so without this check a NaN-producing engine would score a perfect
    0.0/0.0 and sail under the trust bounds written for exactly that defect class. A
    gap list that does not line up with its reply's positions is refused the same way."""
    refuse_past_cap(engine, draw, items)
    events: list[ReplyFlips] = []
    for item, gaps in zip(items, gaps_per_item, strict=True):
        tokens = item.first_of("text").tokens
        try:
            events.append(score_reply(
                gaps, event_key=item_event_key(item), n_served=len(tokens),
                at_cap=len(tokens) >= item_cap(item)))
        except ValueError as exc:
            raise CalibrationError(
                f"pinned reference returned a malformed gap list for pool engine "
                f"{engine!r}, serving {draw}: {exc}") from exc
    n_nan = sum(ev.n_nan for ev in events)
    n_inf = sum(ev.n_inf for ev in events)
    if n_nan or n_inf:
        raise CalibrationError(
            f"trusted pool engine {engine!r} produced non-finite teacher-forced gaps "
            f"({n_nan} NaN, {n_inf} inf) on serving {draw}; the calibration instrument "
            f"is broken (NaN: numeric defect in the reference or engine stack; inf: "
            f"engine emitted an out-of-vocabulary id). Fix the stack and re-run.")
    return DrawStats.from_events(engine, draw, seed, serving_version, events)


# ---------------------------------------------------------------------------
# One serving's scored report (the sampling gate's measurement)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReportDrawStats:
    """One serving's report measurement: every reply it produced, reduced the way the
    deployed sampling gate reduces a graded trace.

    The analogue of ``DrawStats`` for the report kind, and deliberately thinner: the
    sampling ceiling is one mean with one standard error, so there is no severity grid
    to count on and no per-serving tally worth storing beyond the records themselves.
    Everything the assembly needs comes out of a ``PooledReport`` over ``events``, and
    the store keeps only the records, so no total exists in two places.

    A reply whose report violated the contract is in ``events`` carrying its reason and
    is outside every measured sum (``PooledReport.add``). A pool member is trusted, so
    a violation here is a finding about the member rather than a verdict about a
    contestant: it is logged and counted, and the ceiling is measured on the replies
    that did describe their own emission."""

    engine: str
    draw: int
    seed: int
    #: The serving package version the member reported while it produced this serving.
    serving_version: str
    #: The checkpoint's end-of-turn ids, as the pinned reference reported them while
    #: scoring this serving. A property of the checkpoint rather than of the serving,
    #: kept here because the scoring pass is the only place that sees it and the
    #: artifact records it (``hls.verify.calibration.EndOfTurnCheck``).
    eot_ids: tuple[int, ...]
    events: tuple[ReplyReport, ...]

    def header(self) -> dict:
        """The serving's identity, as the scored file's first line
        (``CaptureStore.write_scored``): what the per-reply records cannot carry."""
        return {"engine": self.engine, "draw": self.draw, "seed": self.seed,
                "serving_version": self.serving_version,
                "eot_ids": list(self.eot_ids)}

    @property
    def n_items(self) -> int:
        """Replies the serving produced, violations included."""
        return len(self.events)

    @property
    def n_tokens(self) -> int:
        """Tokens SERVED over those replies, a violated reply's included -- the
        campaign's exposure counter reads this. Not the same count as
        ``hls.verify.calibration.EngineReport.n_tokens``, which is the measured
        positions the ceiling's standard error is over; ``pooled().n_tokens`` is
        that one."""
        return sum(ev.n_tokens for ev in self.events)

    @property
    def n_violations(self) -> int:
        """Replies whose report violated the contract."""
        return sum(1 for ev in self.events if ev.violation is not None)

    def pooled(self) -> PooledReport:
        """The serving's records as one pooled measurement. Built on demand rather
        than stored: it holds every position's transform, which is the serving's whole
        token count over again."""
        pooled = PooledReport()
        pooled.extend(self.events)
        return pooled


def _violation_reasons(events: Iterable[ReplyReport], limit: int = 3) -> list[str]:
    """The first few distinct contract-violation reasons among ``events``.

    Distinct rather than the first few records, because the failure this is read for --
    a member whose report shape is wrong -- makes every reply violate the same way, and
    a list of one reason repeated three times says less than the one reason does.
    Position numbers inside a reason are left alone: they say where to look."""
    out: list[str] = []
    for ev in events:
        if ev.violation is not None and ev.violation not in out:
            out.append(ev.violation)
            if len(out) == limit:
                break
    return out


@dataclass(frozen=True)
class _CapturedEvent:
    """One captured reply in the shape the report verifier's own helpers read.

    The verifier reads a graded trace's ``hls.core.trace.TraceEvent``; a capture
    records the stamped request and the served reply but not the session it belonged
    to, so this carries exactly the three attributes those helpers touch instead of a
    TraceEvent with invented fields. Going through them rather than around them is the
    point: the partition a calibration serving is measured on, and the contract
    violations it is credited with, are then named by the same code that names them on
    a contestant's trace."""

    request_id: str
    input: Envelope
    response: ArtifactResponse
    #: A graded event's per-event metadata. Always empty here: the flags those helpers
    #: read describe how the harness ended a request, and a capture holds only replies
    #: that ended on their own (``capture_items``).
    extras: dict = field(default_factory=dict)

    @property
    def event_key(self) -> str:
        """The same key the graded trace's event for this request carries."""
        return event_key(self.input.metadata.get("tenant_id"), self.request_id)


def _captured_event(item: Envelope) -> _CapturedEvent:
    """Split one capture record back into the request/reply pair it was stored from.

    The stored item folds both into one envelope -- the stamped context in its
    metadata, the served reply in its contents -- which is what the teacher-forcing
    path wants. The report path wants them apart, because the reply's report is read
    against the request's cap and conversation."""
    return _CapturedEvent(
        request_id=item.request_id,
        input=Envelope(request_id=item.request_id, contents=[],
                       metadata=dict(item.metadata)),
        response=ArtifactResponse(output=Envelope(
            request_id=item.request_id, contents=list(item.contents))))


def report_verifier(task: TaskSpec) -> "ReportFaithfulnessVerifier":
    """The task's report verifier -- the one whose ceiling this campaign measures.

    A report calibration is scored by the deployed gate's own code, so the campaign
    needs the configured verifier itself, not a copy of its settings. A task reaching
    here has already been dispatched on its calibration kind, so anything but exactly
    one report verifier is a caller bug rather than a user error."""
    from hls.verify.verifiers import ReportFaithfulnessVerifier
    found = [v for t in task.effective_tenants() for v in t.verifiers
             if isinstance(v, ReportFaithfulnessVerifier)]
    if len(found) != 1:
        raise CalibrationError(
            f"a report calibration measures one ReportFaithfulnessVerifier's ceiling, "
            f"but this task declares {len(found)}; declare exactly one.")
    return found[0]


def _without_report(ev: _CapturedEvent) -> _CapturedEvent:
    """The same captured reply with its report dropped.

    The partition is built from the report once and nothing reads it again: the
    reference gathers at the ids the partition names, and the ending rule reads the
    reply's served ids, its cap and its finish reason. The report is the expensive part
    of a captured reply by an order of magnitude -- twenty-one reported alternatives per
    served token, each an object -- so it is released as soon as the rows exist, which
    is what keeps a serving of millions of positions inside memory."""
    out = ev.response.output
    contents = [c.model_copy(update={"logprobs": None}) if isinstance(c, TextPayload) else c
                for c in out.contents]
    return _CapturedEvent(
        request_id=ev.request_id, input=ev.input,
        response=ArtifactResponse(output=Envelope(request_id=out.request_id,
                                                 contents=contents)))


async def score_report_capture(verifier: "ReportFaithfulnessVerifier",
                               store: "CaptureStore", engine: str, draw: int,
                               rows: "RowIndex", pinned_proxy, *,
                               max_context_len: int | None) -> ReportDrawStats:
    """Phase 2 for one serving of a sampled decode: score every captured reply's report
    against the pinned reference, exactly as the deployed gate scores a graded trace.

    ``max_context_len`` is ``TaskSpec.max_context_len``: the window the pool's members
    were built at, and the one the deployed gate reads a reply's ending against.
    Required rather than defaulted, because the fallback is the checkpoint's own
    declared window and that is a different, usually larger number -- defaulting it
    would hold the members to a rule the graded artifact is not held to. ``None`` where
    the task declares no window, which is also where its members serve at the
    checkpoint's.

    Three steps, the gate's own: build each reply's partition from the report it came
    with and name the contract violations (``ReportFaithfulnessVerifier._rows_for``),
    ask the reference for its own log-probabilities at the ids those partitions name,
    then reduce the two against each other position by position
    (``hls.verify.report``).

    Done in groups, reading the capture rather than taking it as a list, because a
    serving's reports do not fit in memory: an agentic serving reaches millions of
    served positions and a rebuilt report costs kilobytes per position. So each group is
    read, has its rows built (releasing each reply's report immediately,
    ``_without_report``), is scored in one reference call, and is reduced to one small
    record per reply before the next group is read. The group bound is the gate's own
    (``hls.verify.verifiers.report_call_is_full``), and grouping changes no scored
    number.

    A violated reply is not sent to the reference and enters no sum; it is recorded
    with its reason. Refusing the serving over it would be wrong here: the member is a
    trusted stack, so a violation says its report shape needs fixing, and the campaign
    reports that rather than discarding the replies that were fine.

    Like the teacher-forcing path, this needs no serving engine alive -- which is what
    lets captures be taken on one machine and scored on another."""
    from hls.verify.verifiers import report_call_is_full
    records: list[ReplyReport] = []
    refused: list[tuple[_CapturedEvent, str]] = []
    group: list[tuple[_CapturedEvent, Envelope, ReportRows]] = []
    positions = 0
    # The ending facts every reply is read against. The end-of-turn ids are the
    # checkpoint's, learned from the first reference reply and the same in every one of
    # them (see the ReferenceLogprobs capability). The context window is the task's
    # declaration, seeded here; a task that declares none falls back to the
    # checkpoint's, which is where its members serve.
    facts: dict = {"max_context": max_context_len}
    with store.streamed_capture(engine, draw, rows=rows) as (seed, version, replies):
        for item in replies:
            ev = _captured_event(item)
            built = verifier._rows_for(ev)
            light = _without_report(ev)
            if isinstance(built, str):
                refused.append((light, built))
                continue
            reply_rows, reference_item = built
            group.append((light, reference_item, reply_rows))
            positions += len(reply_rows.q_ids)
            if report_call_is_full(len(group), positions):
                records.extend(await _score_report_group(
                    verifier, pinned_proxy, group, facts, engine, draw))
                group, positions = [], 0
        if group:
            records.extend(await _score_report_group(
                verifier, pinned_proxy, group, facts, engine, draw))
    # The refused replies last, because their records name whether they ended on an
    # end-of-turn id and only the reference knows the checkpoint's.
    records.extend(verifier._violated(ev, reason, facts.get("eot_ids", set()))
                   for ev, reason in refused)
    return ReportDrawStats(engine=engine, draw=draw, seed=seed,
                           serving_version=version,
                           eot_ids=tuple(sorted(facts.get("eot_ids", set()))),
                           events=tuple(records))


async def _score_report_group(verifier: "ReportFaithfulnessVerifier", pinned_proxy,
                              group: list[tuple[_CapturedEvent, Envelope, ReportRows]],
                              facts: dict, engine: str, draw: int) -> list[ReplyReport]:
    """Score one reference call's worth of replies and reduce them to their records.

    The reduction is the deployed gate's own, down to the ending rule and the facts it
    accumulates (``ReportFaithfulnessVerifier.reduce_replies``): a ceiling measured
    under any other one would not be the ceiling a graded trace is read against. Its
    event key and violation reason are already fields of the record, and its short-report
    count describes the serving stack rather than the measurement, so only the records
    are kept here.

    ``facts`` accumulates the end-of-turn ids and the context window across calls:
    neither is a property of a reply, so the first call that reports them fixes them
    for the whole serving. A context window seeded by the caller from the task's
    declaration is kept; only an unseeded one takes the checkpoint's."""
    replies = await pinned_proxy.reference_logprobs([item for _ev, item, _rows in group])
    if len(replies) != len(group):
        raise CalibrationError(
            f"the pinned reference returned {len(replies)} replies for {len(group)} "
            f"items on serving {draw} of pool engine {engine!r}; the report is scored "
            f"one reply per item.")
    return [record for _key, _reason, record, _n_short
            in verifier.reduce_replies(group, replies, facts)]


# ---------------------------------------------------------------------------
# The pinned reference and its seeds
# ---------------------------------------------------------------------------

def pinned_reference(task: TaskSpec):
    """The pinned reference: the instrument that scores every pool member's reply and
    the preflight anchor's probes. It is the MATH-kernel bf16 HF instrument, built here
    from the task's model and contract -- which is the same engine the task's own
    ``HFReference`` produces, since that is ``HFEngine``'s default attention pin.

    One engine spec, deployed once for the instrument check and once per score share
    (``score_share``), never held alive between them. It is the fixed axis every member
    is compared against: what has to be identical across those deploys is this spec,
    not the allocation.

    It scores under the task's own decode (``_task_decode``), so a task that samples, or
    that penalizes, is measured against the distribution its members served under, and
    tiles its forward at the task's own chunk (``_task_query_chunk``), so the ceilings
    are measured on the instrument the graded gate will run."""
    return HFEngine(model_spec=task.model, decode=_task_decode(task),
                    gpu_kind=task.hardware.accelerator.kind, sdpa_backend="math",
                    query_chunk=_task_query_chunk(task),
                    **contract_conditioning(task))


def serving_seed_of(task_seed: int, index: int) -> int:
    """The scheduling seed serving ``index`` of a task whose seed is *task_seed* is clocked with.

    Derived from the task seed and the serving index, so servings differ from each
    other while every one of them is reproducible from the task alone -- which is what
    lets the grower plan a top-up serving on the login node and a retried capture keep
    its serving's clock. Hashed with sha256 rather than Python's ``hash`` because the
    latter is randomized per process.

    This seed moves arrivals, think-times and ramp-down turn membership. It does not
    move which rows are drawn (provisioning already materialized those from the task
    seed), and it sits outside the workload fingerprint because it identifies a
    measurement, not the workload the calibration is valid for."""
    digest = hashlib.sha256(f"hls.calibrate.serving:{task_seed}:{index}".encode()).digest()
    # 31 bits: a positive int comfortably inside every seeding path it reaches.
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Capture: one serving's trace, as teacher-force items
# ---------------------------------------------------------------------------

def _teacher_force_item(prompt: Envelope, reply: Envelope, *, source: str,
                        request_id: str | None = None,
                        with_report: bool = False) -> Envelope:
    """Build the teacher-force item for one (prompt, reply) pair: the reply's text and
    exact emitted token ids as the continuation, with the request's ``messages`` as the
    context the reference renders, and its ``tools`` (the request's tool definitions,
    when it offered any, because the reference must render the prompt the member
    rendered). The pinned reference's ``teacher_force`` reads it, or the sampling
    gate's rows.

    It carries the request's stamped ``tenant_id`` (the tenant half of the event key)
    and ``max_tokens`` (the cap the measurement tests against). A reply whose
    continuation cannot be read, or a request missing either stamp, is a hard error
    naming ``source`` -- calibration cannot score a continuation it cannot read or key.

    ``with_report`` also keeps the reply's own report and reported finish reason, which
    is what the sampling gate measures and what its ending rule is read against. Off by
    default because the flip measurement reads neither, and a reply carries a report
    entry per served position: keeping one where nothing reads it would grow every
    stored capture by the width of the report for nothing."""
    rid = request_id or reply.request_id
    tenant = prompt.metadata.get("tenant_id")
    cap = prompt.metadata.get("max_tokens")
    if not isinstance(tenant, str) or not isinstance(cap, int) or cap <= 0:
        raise CalibrationError(
            f"{source}: request {rid!r} is not stamped with a tenant_id and a positive "
            f"max_tokens (got {tenant!r}, {cap!r}); every request the drive loop sends "
            f"carries both (hls.spec.drivers.stamp_request), and the measurement needs them "
            f"to key the event and test its cap.")
    tp = reply.first_of("text")
    if tp is None or tp.tokens is None:
        raise CalibrationError(
            f"{source} produced a reply for {rid!r} with no token ids "
            f"(TextPayload.tokens is None); a calibration member must emit the exact ids "
            f"it generated so the pinned reference can teacher-force that continuation. "
            f"Check the served stack's token-id support and re-run.")
    meta: dict = {"messages": prompt.metadata["messages"], "tenant_id": tenant,
                  "max_tokens": cap}
    tools = prompt.metadata.get("tools")
    if isinstance(tools, list) and tools:
        meta["tools"] = tools
    payload = TextPayload(text=tp.text, tokens=list(tp.tokens))
    if with_report:
        payload = TextPayload(text=tp.text, tokens=list(tp.tokens),
                              logprobs=tp.logprobs, finish_reason=tp.finish_reason)
    return Envelope(request_id=rid, contents=[payload], metadata=meta)


def served_length(item: Envelope) -> int:
    """The served length a teacher-force item's stamped cap is compared against: its
    emitted token ids."""
    return len(item.first_of("text").tokens)


def item_event_key(item: Envelope) -> str:
    """The event key of a teacher-force item: the same ``hls.core.trace.event_key`` a
    graded trace's event carries for that request, built from the stamped tenant."""
    return event_key(item.metadata["tenant_id"], item.request_id)


def item_cap(item: Envelope) -> int:
    """The output cap stamped on a teacher-force item's request."""
    return item.metadata["max_tokens"]


def capture_items(trace, *, engine: str, draw: int,
                  with_reports: bool = False) -> list[Envelope]:
    """Phase 1's output: one serving's ``Trace`` as teacher-force items.

    Each event contributes the conversation its request actually carried
    (``metadata['messages']``, stitched by the drive loop turn by turn), the tenant and
    cap the drive loop stamped on it, and the reply's emitted ids, keyed by the event's
    request id so a severe event can be attributed to a specific prompt in the
    recurrence census. ``with_reports`` also keeps each reply's own report and finish
    reason, which is what a sampled serving is measured on (``_teacher_force_item``).

    A serving with any faulted request is refused outright. A pool member is a trusted
    stack serving its own task's declared load; a fault means it could not, and the
    alternative -- dropping the faulted requests -- would silently measure a different,
    smaller workload than the one that was offered (and, because a saturating server
    faults its longest requests first, a systematically easier one).

    A serving the drive's scheduled stop cut is refused for the same reason. A cut
    reply stops where the harness stopped listening, so capturing it would measure the
    member on a truncated reply, and dropping it would leave out exactly the requests
    that were still running when the window closed, the longest ones."""
    faults = [(ev.request_id, ev.error) for ev in trace.events
              if ev.response is None or ev.error]
    if faults:
        rid, err = faults[0]
        raise CalibrationError(
            f"pool member {engine!r} faulted {len(faults)} of {len(trace.events)} requests "
            f"on serving {draw} (first: {rid} -- {err}). A calibration serving must be a "
            f"clean drive of the task's declared load: dropping the faulted requests would "
            f"measure a smaller and easier workload than the one offered. Fix the member's "
            f"serving configuration (or the load it is driven at) and re-run; the capture "
            f"store keeps every serving already completed.")
    cut = [ev.request_id for ev in trace.events if ev.extras.get(CUT_EXTRA)]
    if cut:
        raise CalibrationError(
            f"the scheduled stop cut {len(cut)} of {len(trace.events)} requests of pool "
            f"member {engine!r} on serving {draw} (first: {cut[0]}). A cut reply ends "
            f"where the harness stopped listening, so the ceiling would be measured on "
            f"replies a graded run does not have. Drive fewer sessions per serving, or "
            f"widen the task's wall clock, and re-run; the capture store keeps every "
            f"serving already completed.")
    return [_teacher_force_item(ev.input, ev.response.output,
                                source=f"pool member {engine!r}",
                                request_id=ev.request_id, with_report=with_reports)
            for ev in trace.events]


_TURN_SUFFIX = re.compile(r"^(.*)\.(\d+)$")


@dataclass(frozen=True)
class RowIndex:
    """Which rows each tenant serves, and each tenant's output cap.

    A stored capture record names its request by tenant and by the dataset-style id the
    drive loop gave it, ``<row>.<turn>``. This index holds the same facts from the task
    alone, built from the sessions a serving drives (``load_sessions``), so every record
    is checked against the workload being calibrated: its tenant must serve its row in
    this split, and its cap must be that tenant's. Rows are keyed within their tenant,
    since native row ids are per corpus and two tenants' corpora may share them."""

    rows_of_tenant: dict[str, set[str]]
    cap_of_tenant: dict[str, int]

    @classmethod
    def from_sessions(cls, sessions: list[tuple]) -> "RowIndex":
        """Index the ``[(tenant, items)]`` a serving drives."""
        rows_of_tenant: dict[str, set[str]] = {}
        cap_of_tenant: dict[str, int] = {}
        for tenant, items in sessions:
            cap_of_tenant[tenant.id] = tenant.workload.output_length_cap.max_tokens()
            rows_of_tenant[tenant.id] = {item.request_id for item in items}
        return cls(rows_of_tenant=rows_of_tenant, cap_of_tenant=cap_of_tenant)

    @staticmethod
    def row_of(request_id: str) -> str:
        """The dataset item id behind a request id: ``<row>.<turn>`` less the turn."""
        m = _TURN_SUFFIX.match(request_id)
        return m.group(1) if m else request_id

    def check(self, tenant_id: str, request_id: str) -> None:
        """Refuse a record whose tenant does not serve its row in this split."""
        row = self.row_of(request_id)
        if row not in self.rows_of_tenant.get(tenant_id, ()):
            raise CalibrationError(
                f"stored capture names request {request_id!r} as tenant {tenant_id!r}, "
                f"but that tenant does not serve row {row!r} in the split this "
                f"calibration drives; the capture came from a different workload or "
                f"split and cannot be keyed against this one.")

    def cap_of(self, tenant_id: str) -> int:
        """The output cap every request of ``tenant_id`` is stamped with."""
        return self.cap_of_tenant[tenant_id]


def _packed_report(logprobs: list[TokenLogprob]) -> list:
    """One reply's report as nested plain JSON: per served position, ``[token_id,
    logprob, [[alternative_id, alternative_logprob], ...]]``.

    Positional rather than keyed, because a report carries an entry per served token
    and twenty-odd alternatives inside each one: at a long reply the field names would
    be most of the file. Only the three things the measurement reads survive the pack
    (``hls.verify.report`` keys on ids and log-probabilities), so the decoded text of
    each token, which is what makes the wire form large, is dropped rather than
    stored."""
    return [[e.token_id, e.logprob,
             [[a.token_id, a.logprob] for a in (e.top_logprobs or [])]]
            for e in logprobs]


def _unpacked_report(packed: list) -> list[TokenLogprob]:
    """Inverse of ``_packed_report``. The decoded token text is not in the packed form
    and comes back empty: nothing that reads a report reads it."""
    return [TokenLogprob(
                token="", token_id=token_id, logprob=logprob,
                top_logprobs=[TopLogprob(token_id=a_id, logprob=a_lp)
                              for a_id, a_lp in alternatives])
            for token_id, logprob, alternatives in packed]


class CaptureStore:
    """The record of what a campaign served and scored, as its roles lay it out.

    Layout, under ``root``::

        <fingerprint>/<engine>/<serving>.capture.jsonl.gz   phase 1: the served replies
        <fingerprint>/<engine>/<serving>.replies.jsonl.gz   phase 2: the scored replies

    The replies file opens with one header line carrying the serving's identity,
    ``{"engine", "draw", "seed"}``, and then holds one ``ReplyFlips`` record per
    scored reply. Everything the aggregation needs (``DrawStats``) is derived from
    those records on read, so no total is stored twice; the header carries only what
    the records cannot, and its engine and draw are checked against the path so a
    file moved between servings is refused. A serving counts as scored when its
    replies file exists.

    Keyed by workload fingerprint, so captures taken over traffic that has since changed
    describe a different workload and do not appear under the new fingerprint at all,
    rather than being silently pooled with fresh ones.

    Separate files per phase, because each is a stage's output: a capture role writes
    the served replies, a score role the scored ones, and the aggregate reads the scored
    ones back. The per-reply records make a re-tally under a changed rule a file read
    that needs no reference. A calibration role keeps one store in a view of its own
    (``hls.verify.calibration_roles``), and ``hls capped-requests`` reads a calibration's
    published captures through one to derive a task's removal list (``capped_rows``).
    """

    def __init__(self, root: Path, fingerprint: str) -> None:
        self.root = Path(root) / fingerprint

    def _dir(self, engine: str) -> Path:
        return self.root / engine

    def capture_path(self, engine: str, draw: int) -> Path:
        """Where serving ``draw``'s served replies live."""
        return self._dir(engine) / f"{draw}.capture.jsonl.gz"

    def replies_path(self, engine: str, draw: int) -> Path:
        """Where serving ``draw``'s scored replies live: the identity header, then one
        record per reply."""
        return self._dir(engine) / f"{draw}.replies.jsonl.gz"

    def write_capture(self, engine: str, draw: int, seed: int, version: str,
                      items: list[Envelope]) -> None:
        """Persist one serving: an identity header, then one record per reply.

        The header carries the seed the serving was clocked with (so a stored capture
        can be re-driven exactly) and the serving package's version as the member's own
        process reported it (so the artifact names the build that produced it, even when
        the captures are scored on another machine). Each record carries the stamped
        tenant and cap, plus what the reply was: the rendered context and the emitted
        text and ids.

        Under the flip measurement that is the whole record: the member's own reported
        finish reason is not among them, because nothing on either side of that gate
        reads it and a reply that ran to the cap is identified by its served length
        instead. A capture taken for the sampling gate carries two more fields, because
        that gate does read them -- the reply's report (``logprobs``, compactly: per
        served position its id, its log-probability, and the reported alternatives as
        id/log-probability pairs) and its reported finish reason, which the ending rule
        is read against. Both are written only when the reply carries them, so a flip
        capture holds neither.

        Written to a temporary sibling and renamed, so an interrupted write leaves no
        half-file that a later run would read as a complete serving."""
        path = self.capture_path(engine, draw)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        with gzip_text_writer(tmp) as fh:
            fh.write(json.dumps({"engine": engine, "draw": draw, "seed": seed,
                                 "serving_version": version}) + "\n")
            for item in items:
                rec = {"request_id": item.request_id,
                       "tenant_id": item.metadata["tenant_id"],
                       "max_tokens": item.metadata["max_tokens"]}
                tp = item.first_of("text")
                rec["messages"] = item.metadata["messages"]
                rec["text"] = tp.text
                rec["tokens"] = list(tp.tokens)
                tools = item.metadata.get("tools")
                if isinstance(tools, list) and tools:
                    rec["tools"] = tools
                if tp.finish_reason is not None:
                    rec["finish_reason"] = tp.finish_reason
                if tp.logprobs is not None:
                    rec["logprobs"] = _packed_report(tp.logprobs)
                fh.write(json.dumps(rec) + "\n")
        tmp.replace(path)

    def read_capture(self, engine: str, draw: int, *,
                     rows: RowIndex) -> tuple[list[Envelope], int, str]:
        """Rebuild one serving's teacher-force items from the store, with the seed it
        was clocked with and the member version that produced it.

        The whole serving at once, which is what the teacher-forcing path wants: it
        sends every item to the reference in one call. The report path reads the same
        file a group at a time instead (``streamed_capture``), because a serving's
        reports do not fit in memory."""
        with self.streamed_capture(engine, draw, rows=rows) as (seed, version, items):
            return list(items), seed, version

    @contextmanager
    def streamed_capture(self, engine: str, draw: int, *, rows: RowIndex):
        """One serving's identity and an iterator over its replies, read lazily.

        Yields ``(seed, version, items)``: the seed the serving was clocked with, the
        member version that produced it, and a generator of one teacher-force item per
        stored reply. The file stays open for the length of the ``with`` block and the
        generator must be consumed inside it.

        Lazy rather than a list because a sampled serving's stored reports are the
        largest thing the campaign handles -- twenty-one alternatives per served token,
        millions of positions -- so neither the decompressed file nor the rebuilt
        replies can be held whole.

        Every record's tenant must serve its row under ``rows`` and its cap must be that
        tenant's: a disagreement means the capture was taken over a different workload
        than the one being calibrated. A file with no header, a header naming an engine
        or serving other than the path it sits at, or a capture that holds no replies at
        all, is refused naming the file."""
        path = self.capture_path(engine, draw)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            first = fh.readline()
            if not first.strip():
                raise CalibrationError(
                    f"stored capture {path} is empty; one serving is one clocked drive, so "
                    f"it must carry at least its identity header. Re-drive it.")
            header = json.loads(first)
            if (header.get("engine"), header.get("draw")) != (engine, draw):
                raise CalibrationError(
                    f"stored capture {path} says it is serving {header.get('draw')!r} of "
                    f"engine {header.get('engine')!r}, not the {engine}/{draw} its path "
                    f"names; the file was moved. Put it back or re-drive it.")

            def items():
                n = 0
                for line in fh:
                    if not line.strip():
                        continue
                    n += 1
                    yield self._capture_item(json.loads(line), engine, draw, rows)
                if not n:
                    raise CalibrationError(
                        f"stored capture {path} holds no replies; the member answered "
                        f"nothing at all, so there is no serving here to score. "
                        f"Re-drive it.")

            yield header["seed"], header["serving_version"], items()

    def _capture_item(self, rec: dict, engine: str, draw: int,
                      rows: RowIndex) -> Envelope:
        """One stored record as the teacher-force item it was written from, refusing a
        record whose tenant or cap does not match the workload being calibrated."""
        tenant, cap = rec["tenant_id"], rec["max_tokens"]
        rows.check(tenant, rec["request_id"])
        if cap != rows.cap_of(tenant):
            raise CalibrationError(
                f"stored capture {engine}/{draw} records request "
                f"{rec['request_id']!r} of tenant {tenant!r} under a {cap}-token "
                f"cap, but this task serves that tenant under {rows.cap_of(tenant)}; "
                f"the capture describes a different workload and cannot be scored "
                f"against this one.")
        meta: dict = {"messages": rec["messages"], "tenant_id": tenant, "max_tokens": cap}
        tools = rec.get("tools")
        if isinstance(tools, list) and tools:
            meta["tools"] = tools
        report = rec.get("logprobs")
        return Envelope(
            request_id=rec["request_id"],
            contents=[TextPayload(
                text=rec["text"], tokens=rec["tokens"],
                finish_reason=rec.get("finish_reason"),
                logprobs=None if report is None else _unpacked_report(report))],
            metadata=meta)

    def write_scored(self, scored: "DrawStats | ReportDrawStats") -> None:
        """Persist one serving's scored replies: the identity header, then one record
        per reply, through a temporary sibling so an interrupted write is never read as
        a scored serving. The tally itself is not written; the matching reader derives
        it (``read_scored`` for flips, ``read_report_scored`` for reports). Both
        measurements persist the same way -- an identity header and one record per
        reply -- so the writer is shared, and each measurement says what its own header
        carries (``header()``)."""
        path = self.replies_path(scored.engine, scored.draw)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        with gzip_text_writer(tmp) as fh:
            fh.write(json.dumps(scored.header()) + "\n")
            for ev in scored.events:
                fh.write(json.dumps(ev.to_json()) + "\n")
        tmp.replace(path)

    def read_scored(self, engine: str, draw: int) -> DrawStats:
        """Read back one serving's scored replies and derive its tally from them.

        Refused, naming the file: a file with no header, and a header whose engine or
        draw disagrees with the path it sits at. The severity grid is not checked here
        and cannot be -- a record stores the gap of each severe flip, and the grid those
        gaps are counted on is chosen from the whole pool once every serving is
        scored."""
        header, lines = self._scored_lines(engine, draw)
        events = tuple(ReplyFlips.from_json(json.loads(line)) for line in lines)
        return DrawStats.from_events(engine, draw, header["seed"],
                                     header["serving_version"], events)

    def read_report_scored(self, engine: str, draw: int) -> ReportDrawStats:
        """Read back one serving's scored reports. The report reader beside
        ``read_scored``, refusing the same way: a file with no header, or a header
        whose engine or serving disagrees with the path it sits at.

        Refused too, and here rather than deep inside the pooling that would hit it: a
        record measured at fewer positions than the reply served. Every position of a
        measured reply is measured (``hls.verify.report.ReplyReport``), so such a record
        came from a reduction that excluded some, and its divergence total and its
        position count would be over different sets of positions."""
        header, lines = self._scored_lines(engine, draw)
        if "eot_ids" not in header:
            raise CalibrationError(
                f"stored serving {self.replies_path(engine, draw)} has no eot_ids in its "
                f"header, so it carries neither the checkpoint's end-of-turn ids nor a "
                f"price per position and cannot be pooled with servings that do. Delete "
                f"the replies file to re-score its capture, which needs the reference but "
                f"no serving stack.")
        records = [json.loads(line) for line in lines]
        # A violated reply was never measured and carries no transforms at all, which
        # its reason already says, so the count below is not a claim about it.
        partial = next((r for r in records if r.get("violation") is None
                        and (r.get("n_refused", 0) or len(r["u"]) != r["n_tokens"])),
                       None)
        if partial is not None:
            raise CalibrationError(
                f"stored serving {self.replies_path(engine, draw)} holds a reply measured "
                f"at only some of its positions: reply "
                f"{partial['event_key']!r} measured {len(partial['u'])} of the "
                f"{partial['n_tokens']} positions it served, so its divergence and its "
                f"position count are over different sets and it cannot be pooled with "
                f"servings that measure every position. Delete the replies file to "
                f"re-score its capture, which needs the reference but no serving "
                f"stack.")
        return ReportDrawStats(
            engine=engine, draw=draw, seed=header["seed"],
            serving_version=header["serving_version"],
            eot_ids=tuple(header["eot_ids"]),
            events=tuple(ReplyReport.from_json(r) for r in records))

    def _scored_lines(self, engine: str, draw: int) -> tuple[dict, list[str]]:
        """One scored serving's identity header and its record lines, refusing a file
        that carries no header or sits at a path its header contradicts."""
        path = self.replies_path(engine, draw)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        if not lines:
            raise CalibrationError(f"stored serving {path} is empty; re-score the capture.")
        header = json.loads(lines[0])
        if (header.get("engine"), header.get("draw")) != (engine, draw):
            raise CalibrationError(
                f"stored serving {path} says it is serving {header.get('draw')!r} of "
                f"engine {header.get('engine')!r}, not the {engine}/{draw} its path names; "
                f"the file was moved. Put it back or re-score the capture.")
        return header, lines[1:]

    def captured_draws(self, engine: str) -> list[int]:
        """Serving indices this engine has a capture for, ascending."""
        return sorted(int(p.name.split(".", 1)[0])
                      for p in self._dir(engine).glob("*.capture.jsonl.gz"))


@dataclass(frozen=True)
class CappedRows:
    """What one tenant's traffic did at the output cap under one engine, across every
    stored serving.

    ``rows`` are the row ids (``Dataset.row_id``, a whole session) with at least one
    turn whose reply reached the cap, sorted -- the list a task removes from its
    workload (``DataSource.removed_rows``). ``capped_turns`` counts
    the distinct turns that reached it; ``served_rows``/``served_turns`` count the
    distinct rows and turns the engine served at all, so the removal's cost to the
    tenant can be read off the same record."""

    rows: tuple[str, ...]
    capped_turns: int
    served_rows: int
    served_turns: int

    def summary(self) -> str:
        """What removing these rows costs the tenant, as one line: how many of its
        served sessions go, and how many of its served turns were at the cap. Lives
        here so every reader quotes the same counts the same way."""
        return (f"{len(self.rows)} of {self.served_rows} sessions removed "
                f"({self.capped_turns} turns at the cap, {self.served_turns} turns "
                f"served)")


def capped_rows(store: CaptureStore, engine: str, rows: RowIndex) -> dict[str, CappedRows]:
    """Per tenant, the sessions ``engine`` ran to the output cap in any of its stored
    servings -- the removal list a task derives from a calibration campaign.

    A reply is at the cap when its served length reached the request's stamped
    ``max_tokens``; the reply's reported finish reason is not read. A reply longer
    than its cap is refused outright: the member did not honor the cap the harness
    sent, so its captures do not describe the workload. Sessions count whole, because a
    multi-turn session's later prompts are its earlier replies: a request removed for
    one turn takes the rest of its conversation with it. ``rows`` keys every record
    (see ``RowIndex``)."""
    served: dict[str, set[str]] = {}
    served_turns: dict[str, set[str]] = {}
    capped: dict[str, set[str]] = {}
    capped_turns: dict[str, set[str]] = {}
    for draw in store.captured_draws(engine):
        items, _seed, _version = store.read_capture(engine, draw, rows=rows)
        for item in items:
            tenant = item.metadata["tenant_id"]
            row = rows.row_of(item.request_id)
            n, cap = served_length(item), item_cap(item)
            if past_cap(n, cap):
                raise CalibrationError(
                    f"pool member {engine!r} served {n} tokens for request "
                    f"{item.request_id!r} in serving {draw}, over the {cap}-token cap "
                    f"the harness stamped on it; the member did not honor the cap, so "
                    f"these captures do not describe the workload.")
            served.setdefault(tenant, set()).add(row)
            served_turns.setdefault(tenant, set()).add(item.request_id)
            if n == cap:
                capped.setdefault(tenant, set()).add(row)
                capped_turns.setdefault(tenant, set()).add(item.request_id)
    return {tenant: CappedRows(rows=tuple(sorted(capped.get(tenant, ()))),
                               capped_turns=len(capped_turns.get(tenant, ())),
                               served_rows=len(served[tenant]),
                               served_turns=len(served_turns[tenant]))
            for tenant in sorted(served)}


# ---------------------------------------------------------------------------
# Phase 2 and the preflight
# ---------------------------------------------------------------------------

async def score_capture(engine: str, draw: int, seed: int, serving_version: str,
                        pinned_proxy, items: list[Envelope]) -> DrawStats:
    """Phase 2 for one serving: teacher-force every captured reply through the pinned
    reference and count the gaps.

    Scoring does not need the serving engine alive -- which is what lets captures be
    taken in one stage and scored in another.

    A serving that holds a reply past its request's cap is refused before the reference
    is asked (``refuse_past_cap``, which ``score_draw`` applies to whatever it is handed):
    a member that ignores the cap does not choose how much the reference must hold."""
    refuse_past_cap(engine, draw, items)
    gaps = await pinned_proxy.teacher_force(items) if items else []
    return score_draw(engine, draw, seed, serving_version, items, gaps)


@dataclass(frozen=True)
class Measure:
    """What one calibration kind does with a serving, as the four things that differ.

    The campaign's shape is the same whichever gate is being calibrated: captures of
    the members' servings, then shares of them scored by pinned references. What
    differs is only the measurement -- whether a capture has to keep the reply's report,
    how a capture becomes a scored serving, how a scored serving is read back, and what
    one line of log about it says. Holding exactly those four here keeps the roles one
    implementation instead of two that could drift.

    Attributes:
        with_reports: whether a capture keeps each reply's report (``capture_items``).
        score:        ``(store, engine, draw, rows, proxy)`` -> the scored serving,
                      awaited on a live pinned reference. Each kind reads the capture
                      itself, because they read it differently: the teacher-forcing
                      path wants the whole serving in one call, the report path a group
                      at a time.
        read:         ``(store, engine, draw)`` -> a scored serving, as the aggregate
                      reads it back.
        log_line:     ``stats`` -> what one scored serving reports.
    """

    with_reports: bool
    score: Callable[..., Awaitable[Any]]
    read: Callable[[CaptureStore, str, int], Any]
    log_line: Callable[[Any], str]


def flip_measure() -> Measure:
    """The token-level gate's measurement: teacher-forced gaps, counted into flips."""
    async def score(store: CaptureStore, engine: str, draw: int, rows: RowIndex,
                    proxy) -> DrawStats:
        items, seed, version = store.read_capture(engine, draw, rows=rows)
        return await score_capture(engine, draw, seed, version, proxy, items)

    def log_line(stats: DrawStats) -> str:
        return (f"{stats.n_tokens} tokens, {stats.n_flips} flips "
                f"({stats.flip_rate:.4g}), {stats.capped_events} replies at the output "
                f"cap")

    return Measure(with_reports=False, score=score,
                   read=CaptureStore.read_scored, log_line=log_line)


def report_measure(verifier: "ReportFaithfulnessVerifier",
                   max_context_len: int | None) -> Measure:
    """The sampling gate's measurement: each reply's own report, scored against the
    reference's distribution at the ids it named.

    The scorer closes over the task's configured verifier, because the measurement is
    that verifier's own code (``score_report_capture``), and over the task's declared
    context window, which that code reads a reply's ending against."""
    async def score(store: CaptureStore, engine: str, draw: int, rows: RowIndex,
                    proxy) -> ReportDrawStats:
        return await score_report_capture(verifier, store, engine, draw, rows, proxy,
                                          max_context_len=max_context_len)

    def log_line(stats: ReportDrawStats) -> str:
        pooled = stats.pooled()
        _omega2, cvm_p = pooled.uniformity()
        line = (f"{pooled.n_replies} replies, {pooled.n_tokens} served tokens, "
                f"mean_kl={pooled.mean_kl:.6g} +/- {pooled.se_kl:.3g} over "
                f"{pooled.n_tokens} measured positions, "
                f"{stats.n_violations} contract violations, "
                f"{pooled.n_outside} served outside the reported atoms, "
                f"z={pooled.z:.3g}, cvm_p={cvm_p:.3g}")
        reasons = _violation_reasons(stats.events)
        # The reasons, not just the count: a wrong report shape makes every reply
        # violate identically, and this line is where a reader would see which way.
        return line if not reasons else f"{line} -- violations: {'; '.join(reasons)}"

    return Measure(with_reports=True, score=score,
                   read=CaptureStore.read_report_scored, log_line=log_line)


def _probe_envelope(idx: int, text: str) -> Envelope:
    """One preflight probe as a stamped single-turn request (max_tokens=64), shaped
    through the same helpers as real traffic so the anchor and the pinned reference
    render it the way any request would."""
    env = Envelope(request_id=f"preflight-{idx}", contents=[TextPayload(text=text)])
    return stamp_request(env, [user_message(env)], tenant_id="preflight",
                         session_id=f"preflight-{idx}", turn_idx=0,
                         max_tokens=PREFLIGHT_MAX_TOKENS)


async def _run_preflight(task: TaskSpec, deploy: Deploy, pinned_proxy) -> PreflightResult:
    """The instrument sanity check: an fp32 anchor (MATH kernel, float32, the task's
    own decode) generates the
    committed probes and the pinned bf16 reference teacher-forces the anchor's tokens.
    fp32-vs-bf16 disagreement should be tiny; a large one means the reference is
    numerically broken. The anchor takes the free GPU slot (the pinned reference holds
    the other) and is released in ``finally``. An fp32 load that OOMs propagates -- no
    CPU fallback (a CPU forward is a different instrument)."""
    anchor = HFEngine(model_spec=task.model, decode=_task_decode(task),
                      gpu_kind=task.hardware.accelerator.kind,
                      sdpa_backend="math", dtype="float32",
                      query_chunk=_task_query_chunk(task),
                      **contract_conditioning(task))
    anchor_proxy, anchor_release = await deploy("fp32_anchor", anchor)
    try:
        probes = [_probe_envelope(i, text) for i, text in enumerate(PREFLIGHT_PROBES)]
        replies = await anchor_proxy.generate(probes)
        items = [_teacher_force_item(p, reply, source="fp32 preflight anchor")
                 for p, reply in zip(probes, replies, strict=True)]
        gaps = await pinned_proxy.teacher_force(items) if items else []
    finally:
        await anchor_release()
    # Every probe token counts, including the ones the 64-token probe cap truncated --
    # as everywhere else the measurement is applied.
    flat = [g for gaps_i in gaps for g in gaps_i]
    n_tokens = len(flat)
    n_flips = sum(1 for g in flat if g > 0)
    flip_rate = n_flips / n_tokens if n_tokens else 0.0
    # A non-finite gap fails the preflight outright: NaN loses every comparison, so
    # max() would drop it order-dependently and a broken instrument would report clean.
    finite = all(math.isfinite(g) for g in flat)
    max_gap = max(flat, default=0.0) if finite else math.inf
    passed = (finite and flip_rate <= PREFLIGHT_MAX_FLIP_RATE
              and max_gap <= PREFLIGHT_MAX_GAP)
    return PreflightResult(passed=passed, flip_rate_vs_fp32=flip_rate,
                           max_gap_vs_fp32=max_gap, n_tokens=n_tokens)


# ---------------------------------------------------------------------------
# Phase 3: aggregation, the census, and the sizing verdicts
# ---------------------------------------------------------------------------

def _rate_side(stats: dict[str, list[DrawStats]]) -> dict[str, EngineRate]:
    """Per-engine flip rate across its servings: the mean is the engine's rate, the
    sample sd is the between-serving spread the precision check is evaluated on."""
    out: dict[str, EngineRate] = {}
    for engine, draws in sorted(stats.items()):
        rates = [d.flip_rate for d in draws]
        out[engine] = EngineRate(
            mean=sum(rates) / len(rates),
            sd=statistics.stdev(rates) if len(rates) > 1 else 0.0,
            draws=len(rates),
            rates=rates,
            tokens=[d.n_tokens for d in draws],
            max_gap=max(d.max_gap for d in draws))
    return out


def _curve_side(stats: dict[str, list[DrawStats]], levels: list[float]) -> CurveSide:
    """Per-engine clean flip-count envelope: events at each rung of ``levels`` pooled
    over the engine's servings, against its pooled exposure, as a Garwood upper limit
    on the per-token event rate.

    ``levels`` is the pool's severity grid (``pool_grid``): one rung per distinct step
    of the clean exceedance curve, plus the first rung no clean engine reached. The
    caller derives it once and hands the same list to every consumer, so the census and
    the ceilings can never be counted on different rungs.

    Pooling servings into one Poisson process is licensed by the recurrence census
    (``_recurrence_warnings``): severe flips are induced by serving numerics rather
    than prompt identity, so they do not repeat on the same prompt across servings and
    arrive independently even though the servings share a finite prompt pool."""
    engines: dict[str, EngineCurve] = {}
    for engine, draws in sorted(stats.items()):
        exposure = sum(d.n_tokens for d in draws)
        per_draw = [d.level_events(levels) for d in draws]
        rows = []
        for i, c in enumerate(levels):
            e = sum(counts[i] for counts in per_draw)
            rows.append(LevelStats(level_nats=c, events=e, lambda_hat=e / exposure,
                                   lambda_up=garwood_upper(e, exposure)))
        engines[engine] = EngineCurve(exposure=exposure, levels=rows)
    return CurveSide(levels_nats=levels, engines=engines)


def _recurrence_warnings(stats: dict[str, list[DrawStats]],
                         levels: list[float]) -> dict[str, str]:
    """The recurrence census: for each engine, does any (prompt, gap level) site
    produce a severe flip in at least ``RECURRENCE_FRACTION`` of its servings?

    This is what licenses pooling an engine's servings as one Poisson process. A task's
    prompt pool is finite, so servings repeat prompts; what must not repeat is the
    events. A site that fires in serving after serving is deterministic rather than
    Poisson, which is what the census flags.

    Returns engine -> human-readable reason for every flagged engine (empty dict when
    the pool is clean). Informational only: the ceilings stand as measured."""
    warnings: dict[str, str] = {}
    for engine, draws in sorted(stats.items()):
        threshold = math.ceil(RECURRENCE_FRACTION * len(draws))
        sites_per_draw = [d.severe_sites(levels) for d in draws]
        # The reported site is the one that recurred most; ties go to the most severe
        # level it reached, since the levels are nested and the top one is the
        # informative statement about the site.
        worst: tuple[int, str, float] | None = None   # (servings, event key, level)
        for i, c in enumerate(levels):
            counts: dict[str, int] = {}
            for sites in sites_per_draw:
                for key in sites[i]:
                    counts[key] = counts.get(key, 0) + 1
            for key, n in counts.items():
                if n >= threshold and (worst is None or n >= worst[0]):
                    worst = (n, key, c)
        if worst is None:
            continue
        n, key, c = worst
        reason = (f"severe flips recur across servings: event {key!r} produced a flip "
                  f"of at least {c} nats in {n} of {len(draws)} servings (flagged at "
                  f"{threshold}). Informational; the ceilings stand as measured.")
        log.warning("pool engine %r: %s", engine, reason)
        warnings[engine] = reason
    return warnings


def _trust_warnings(rates: dict[str, float]) -> dict[str, str]:
    """The stack-review census: engine -> why a human should look at its stack, for
    every engine that trips either check (empty dict when the pool is clean).

    Two checks, both on the mean flip rate, and both warnings rather than refusals. The
    absolute one fires when an engine leaves the benign band
    (``MEMBER_MAX_TRUSTED_FLIP_RATE``). The relative one fires when the ceiling-setting
    engine exceeds ``MEMBER_OUTLIER_RATIO`` times the next-highest, which an engine
    inside the absolute band can do.

    A flagged engine stays in the pool and its servings are in the ceilings; the warning
    names it for review. Refusing instead discards every serving the campaign already
    took. The reason is returned as well as logged, so the calibration run's record says
    what its log said (``pool_warnings``)."""
    warnings: dict[str, str] = {}
    ordered = sorted(rates.values(), reverse=True)
    top_name = max(rates, key=rates.__getitem__) if rates else None
    for name, rate in sorted(rates.items()):
        reasons = []
        if rate > MEMBER_MAX_TRUSTED_FLIP_RATE:
            reasons.append(
                f"flip_rate={rate:.4g} is outside the trusted band "
                f"(> {MEMBER_MAX_TRUSTED_FLIP_RATE})")
        if (name == top_name and len(ordered) >= 2 and ordered[1] > 0
                and rate > MEMBER_OUTLIER_RATIO * ordered[1]):
            reasons.append(
                f"it sets q_max alone: its mean flip rate {rate:.4g} is "
                f"{rate / ordered[1]:.1f}x the next-highest engine's {ordered[1]:.4g}")
        if not reasons:
            continue
        reason = ("; ".join(reasons)
                  + ". Review its stack; to exclude it, drop it (--engines) and recompute "
                    "over the remaining engines.")
        log.warning("pool engine %r: %s", name, reason)
        warnings[name] = reason
    return warnings


def pool_warnings(campaign: Campaign, cal: Calibration) -> list[str]:
    """What both censuses flag in *cal*'s pool, one line for each flagged engine of each, as
    it was logged: the stack-review census (``_trust_warnings``) on the engines' mean rates,
    and the recurrence census (``_recurrence_warnings``) on *campaign*'s servings at the
    rungs the curve ceilings were counted on. Advisory: nothing *cal* holds depends on them,
    and *cal* does not carry them. The instance that measured *cal* says them when it ends."""
    stats = {e: campaign.stats.get(e, []) for e in cal.rate_side}
    found = (_trust_warnings({e: r.mean for e, r in cal.rate_side.items()}),
             _recurrence_warnings(stats, cal.curve_side.levels_nats))
    return [f"pool engine {e!r}: {reason}" for census in found for e, reason in census.items()]


def _sizing(rate_side: dict[str, EngineRate], stopped: StopReason) -> Sizing:
    """The two facts the artifact stores about its own sizing: the confidence factor
    both (P) inequalities are evaluated at, and why the campaign stopped.

    ``c_M`` is taken at the smallest realized serving count in the pool and applied to
    every engine, so one recorded factor reproduces every engine's check; an engine that
    served more is judged slightly conservatively, which is the safe direction. The
    checks themselves are recomputed from it on demand (``Calibration.precision`` and
    ``Calibration.tail``)."""
    return Sizing(c_m=_c_m(min(r.draws for r in rate_side.values())), stopped=stopped)


# ---------------------------------------------------------------------------
# The campaign: what it measured, what it owes, and one serving
# ---------------------------------------------------------------------------

@dataclass
class Campaign:
    """What a campaign has measured.

    ``stats`` holds every scored serving's tally per engine, in serving order. Its
    entries are whichever measurement the run's ``Measure`` produces -- one kind
    throughout a campaign, since the artifact it assembles is one gate's."""

    stats: dict[str, list[DrawStats | ReportDrawStats]] = field(default_factory=dict)

    def add(self, stats: DrawStats | ReportDrawStats) -> None:
        """Record one scored serving, keeping each engine's list in serving order so
        the artifact's per-serving rates read the same however the shares interleaved."""
        servings = self.stats.setdefault(stats.engine, [])
        servings.append(stats)
        servings.sort(key=lambda d: d.draw)


@dataclass(frozen=True)
class Serving:
    """One planned serving: which engine takes it, its index, and the seed its clock
    is drawn from. The unit a capture stage takes."""

    engine: str
    draw: int
    seed: int


def servings_owed(have: int, exposure: float, target: float) -> int:
    """How many more servings an engine with *have* scored servings and *exposure*
    teacher-forced tokens needs to satisfy the counter stop.

    The stop itself is deterministic: pooled exposure at least *target* and at least
    ``MIN_SERVINGS`` servings, both required. Planning servings side by side means saying
    how many more up front rather than one at a time, which needs an expected
    per-serving yield, so this uses the engine's own realized mean. With nothing
    measured yet there is no mean, and the answer is the serving floor, which the
    campaign owes regardless; the yield is then known and the remainder planned against
    it.

    The estimate can undershoot (the next servings run short) or overshoot (they run
    long). Undershoot is handled by asking again once those servings are scored, which
    the calibration grower does, so the campaign converges on the counter; overshoot
    costs at most the rounding of one top-up, and lands above the target rather than
    below it, which is the safe direction."""
    if have >= MIN_SERVINGS and exposure >= target:
        return 0
    if have == 0:
        return MIN_SERVINGS
    short = max(0.0, target - exposure)
    per_serving = exposure / have
    by_exposure = math.ceil(short / per_serving) if per_serving > 0 else 0
    return max(MIN_SERVINGS - have, by_exposure)


def plan_servings(task_seed: int, next_draw: Mapping[str, int],
                  demand: Mapping[str, int]) -> list[Serving]:
    """The servings *demand* asks for, as one list: each engine's next
    ``demand[engine]`` serving indices from ``next_draw[engine]``, each clocked from its
    own seed (``serving_seed_of``), the engines interleaved round robin in name order so
    that servings dealt into allocations mix them. Indices continue past every serving
    an engine has taken, so none is reused and no capture is taken twice."""
    per_engine = [[Serving(e, d, serving_seed_of(task_seed, d))
                   for d in range(next_draw[e], next_draw[e] + n)]
                  for e, n in sorted(demand.items()) if n]
    return [s for row in zip_longest(*per_engine) for s in row if s is not None]


async def assert_declared_device(proxy, *, kind: str, role: str) -> None:
    """Refuse an allocation that landed on silicon other than the task's declared
    accelerator.

    The preflight proves the accelerator once, but a campaign holds many allocations
    and the scheduler places each one independently -- so the noise floor is only a
    property of the declared GPU if every allocation that measured it ran on one. The
    check is one ``stack_report`` call per deploy, before any work.

    An engine that reports no stack is skipped rather than refused; every engine this
    module deploys provides ``StackReport``, so in production this always fires, and a
    test double is not required to impersonate a GPU to exercise the orchestration."""
    report = getattr(proxy, "stack_report", None)
    if report is None:
        return
    device = str((await report()).get("device_name", ""))
    if not device_matches_kind(device, kind):
        raise CalibrationError(
            f"calibration allocation {role!r} is running on {device!r}, but the task "
            f"declares hardware.accelerator.kind={kind!r}. bf16 noise floors are "
            f"GPU-kind specific, so a campaign whose allocations land on mixed silicon "
            f"measures nothing. Check that hls.toml's site.gpu_gres names the gres of the "
            f"declared accelerator, then run `hls calibrate` again.")


async def _deploy_member(serving: Serving, *, task: TaskSpec, deploy: Deploy,
                         builders: dict[str, Callable[[], Any]]):
    """Deploy the pool member for one serving and return ``(proxy, release)``. The
    deploy name carries the serving index, so each serving's engine is named apart in
    the role's log."""
    name = f"{serving.engine}-{serving.draw}"
    return await deploy(name, builders[serving.engine]())


@dataclass(frozen=True)
class _DriveResult:
    """What one drive attempt did: whether it captured a serving, and what went wrong if
    anything. A failed drive returns its failure rather than raising it, so a capture
    written before a failed release still counts (``failed``)."""

    #: True when this attempt persisted its capture; False for a drive that never got
    #: that far. It can be True alongside an ``error``: the capture is written before the
    #: engine is released.
    captured: bool
    #: Why the attempt failed, or None.
    error: BaseException | None = None

    @property
    def failed(self) -> bool:
        """Whether this attempt owes the serving another drive.

        A drive that persisted its capture and then failed on the way out does not: the
        capture is in the store and a scorer can read it. Counting it as a failed
        serving would drive again a serving that captured everything it was asked
        for."""
        return self.error is not None and not self.captured


@dataclass(frozen=True)
class _ScoreResult:
    """What one share's scoring did: the tallies it produced, and what went wrong if
    anything. Reported, not raised, so the tallies a share did persist come back with
    its failure."""

    #: Each serving's tally, in the order scored.
    stats: tuple[DrawStats | ReportDrawStats, ...]
    #: Why the share failed, or None.
    error: BaseException | None = None


async def drive_serving(serving: Serving, *, task: TaskSpec, sessions: list[tuple],
                        sessions_per_draw: int | None,
                        request_timeout_sec: float, store: CaptureStore,
                        measure: Measure, drive_timeout_sec: float | None = None,
                        **deploy_kw) -> _DriveResult:
    """Capture one serving: deploy the member, have it drive the traffic on its own
    clock, persist the capture, release.

    A failure leaves no capture behind (the capture file is written last, through a
    temporary sibling), so the serving is driven again under its own index. The one
    exception is a release that fails after the capture is written: that reports both the
    capture and the error, and the capture is what decides whether the serving is owed
    another drive (``_DriveResult.failed``).

    One deploy per serving (rather than one per engine, reused) is what makes servings
    independent measurements: each starts on a freshly loaded engine, with no prefix
    cache or allocator state carried over from the last one."""
    error: BaseException | None = None
    captured = False
    try:
        proxy, release = await _deploy_member(serving, task=task, **deploy_kw)
        try:
            await assert_declared_device(
                proxy, kind=task.hardware.accelerator.kind,
                role=f"{serving.engine} serving {serving.draw}")
            # Asked of the member while it is still up: the serving package lives in
            # its environment, so this is the only place the artifact can learn which
            # build produced the capture.
            version = await proxy.serving_version()
            trace = await proxy.serve_traffic(sessions, seed=serving.seed,
                                              max_sessions=sessions_per_draw,
                                              request_timeout_sec=request_timeout_sec,
                                              drive_timeout_sec=drive_timeout_sec,
                                              agent=task.agent, max_context_len=task.max_context_len)
            items = capture_items(trace, engine=serving.engine, draw=serving.draw,
                                  with_reports=measure.with_reports)
            store.write_capture(serving.engine, serving.draw, serving.seed, version,
                                items)
            captured = True
        finally:
            await release()
    except Exception as exc:
        error = exc
    return _DriveResult(captured=captured, error=error)


async def score_share(share: list[Serving], index: int, *, task: TaskSpec,
                      deploy: Deploy, store: CaptureStore, rows: RowIndex,
                      measure: Measure) -> _ScoreResult:
    """Score one share of servings: deploy the pinned reference once, score each capture
    in the share in turn through ``measure``, persist each tally, release.

    One deploy for the whole share, not one per capture: the reference's weight load is
    the expensive part, and a plan deals each serving to one share, so a scorer works
    through its own without coordination. A share that fails part-way reports the
    tallies it did persist with its error. The deploy name carries the share's index."""
    name = f"score-{index}"
    out: list[DrawStats | ReportDrawStats] = []
    error: BaseException | None = None
    try:
        proxy, release = await deploy(name, pinned_reference(task))
        try:
            await assert_declared_device(proxy, kind=task.hardware.accelerator.kind,
                                         role=f"reference scorer {name}")
            for serving in share:
                scored = await measure.score(store, serving.engine, serving.draw, rows,
                                             proxy)
                store.write_scored(scored)
                out.append(scored)
        finally:
            await release()
    except Exception as exc:
        error = exc
    return _ScoreResult(stats=tuple(out), error=error)


def check_measured(scored: Iterable[DrawStats | ReportDrawStats]) -> None:
    """Refuse a serving that measured nothing.

    The counter can only advance on measured tokens, so a serving that yields none would
    have the campaign ask for servings without end. It also means the member answered
    nothing at all: every reply it did produce is counted, whatever it reported about how
    it ended."""
    empty = [d for d in scored if d.n_tokens == 0]
    if empty:
        e = empty[0]
        raise CalibrationError(
            f"serving {e.draw} of pool member {e.engine!r} produced no tokens at all "
            f"({e.n_items} replies, every one of them empty). No number of servings "
            f"can reach the exposure target from there: check that the member is "
            f"answering, and that its replies carry the emitted token ids.")


def _stack_of(task: TaskSpec, report: dict) -> StackInfo:
    """The pinned reference's stack record, refusing silicon that is not the task's
    declared accelerator.

    GPU-identity gate at the source: the noise floor is measured on whatever silicon
    the allocation actually got, and it is only valid for the task's declared
    accelerator if they are the same thing. Refusing here -- rather than recording the
    declared kind unverified -- is what makes ``read_calibration``'s gpu_kind refusal
    real: otherwise both sides of that comparison come from the same declaration and it
    can never fire. (Every later allocation re-checks its own device; see
    ``assert_declared_device``.)"""
    stack = StackInfo(**report)
    declared_kind = task.hardware.accelerator.kind
    if not device_matches_kind(stack.device_name, declared_kind):
        raise CalibrationError(
            f"calibration must be measured on the task's declared accelerator: the "
            f"task declares hardware.accelerator.kind={declared_kind!r} but the "
            f"deployed reference is running on {stack.device_name!r}. Run `uv run "
            f"hls calibrate` on the declared GPU, or fix the task's hardware "
            f"declaration.")
    return stack


async def check_stack(task: TaskSpec, deploy: Deploy) -> StackInfo:
    """Bring the pinned reference up once, record its stack, release.

    What a report calibration needs from the instrument phase: the fp32-agreement
    preflight teacher-forces committed probes, and teacher-forcing is refused under a
    sampled decode -- there is no argmax for a gap to be measured against. The sampling
    gate's own safeguard is in the measurement instead: every
    engine's divergence is published with its standard error, and a reference that was
    numerically broken would not land within a hair of a second, independent serving
    stack."""
    proxy, release = await deploy("pinned", pinned_reference(task))
    try:
        return _stack_of(task, await proxy.stack_report())
    finally:
        await release()


async def check_instrument(task: TaskSpec,
                           deploy: Deploy) -> tuple[StackInfo, PreflightResult]:
    """The instrument phase: prove the reference is the declared accelerator's, and
    that it is not numerically broken. Returns its stack record and preflight result,
    with every engine released.

    Runs before any serving, in the calibration's first allocation, because both
    answers are prerequisites for the campaign meaning anything: a reference on the
    wrong silicon measures a different noise floor, and a broken one measures nothing.
    Finding either out after the servings have run is the expensive way.

    The fp32 anchor and the pinned bf16 reference have to be up together to be
    compared, so the instrument role holds a GPU for each (``hls.verify.inprocess``
    places them)."""
    proxy, release = await deploy("pinned", pinned_reference(task))
    try:
        stack = _stack_of(task, await proxy.stack_report())
        preflight = await _run_preflight(task, deploy, proxy)
    finally:
        await release()
    if not preflight.passed:
        raise CalibrationError(
            f"calibration preflight FAILED: the fp32 anchor and the pinned bf16 "
            f"reference disagree far more than benign rounding "
            f"(flip_rate={preflight.flip_rate_vs_fp32:.4g} > {PREFLIGHT_MAX_FLIP_RATE}, "
            f"or max_gap={preflight.max_gap_vs_fp32:.4g} > {PREFLIGHT_MAX_GAP}). The "
            f"reference instrument is numerically broken (eager-attention gaps / zeroed "
            f"heads), so its noise floor is meaningless. Fix the reference (SDPA MATH "
            f"pin, cuDNN-SDP off) and re-run `uv run hls calibrate`.")
    return stack, preflight


def check_campaign(task: TaskSpec, *, engines: Sequence[str],
                   servings_per_engine: int | None) -> tuple[str, int | None]:
    """Refuse a calibration of ``task`` by the pool ``engines`` that cannot be taken,
    before anything is planned, and return which gate it measures (``"flips"`` or
    ``"report"``) and, for the sampling gate, how many servings each engine takes
    (``servings_per_engine``, or ``DEFAULT_REPORT_SERVINGS``; None for the token-level
    gate).

    Which gate follows from the task's own calibrated verifier
    (``hls.verify.calibration.calibration_kind``). The token-level gate under a greedy
    decode sizes itself from one constant: each engine serves until its pooled exposure
    reaches ``exposure_target(mu_per_mtok)`` and it has at least ``MIN_SERVINGS``
    servings (``servings_owed``), so a serving count is refused for it, and (P) is a
    single look at a pre-committed size. The sampling gate under a sampled decode takes
    a fixed count instead, because its ceiling is a mean with a reply-clustered standard
    error rather than a spread over servings, so one serving already carries its own
    precision check.

    The pinned reference is built here, never loaded, so calibrated verifiers that
    disagree on the decode or the teacher-forcing chunk are refused before a GPU is asked
    for. The members are built by the plan too, for their shape
    (``hls.pipeline.calibration.member_shape``), from the same decode.

    Refuses (all ``CalibrationError``, each naming its fix): no engine; a task that
    answers with audio, whose members no role can drive; a pool that cannot serve
    clocked traffic; a task with no calibrated verifier; a sampling gate whose serving
    count is under one, or whose task declares other than one report verifier; a serving
    count for the token-level gate; and calibrated verifiers that disagree on the decode
    or the chunk. A sampling gate's decode always samples: the task's own wiring check
    refuses a report verifier on any other (``hls.verify.base.check_wiring``)."""
    if not engines:
        raise CalibrationError(
            "a calibration needs at least one pool engine; got none. Pass e.g. "
            "--engines vllm to `hls calibrate`.")
    if audio_replies(task):
        raise CalibrationError(
            f"task {task.id!r} answers with audio, and hls calibrate cannot take it: a "
            f"serving is a clocked drive of an engine's asynchronous per-request API, "
            f"which no TTS serving stack publishes, so no role can drive its members. "
            f"A run of the task uses its default, defaults/calibration.json beside its "
            f"module, which fits the task as its module declares it, without task "
            f"overrides that change its workload.")
    served = sorted({type(r).__name__ for r in _calibrated_references(task) if not hasattr(r, "query_chunk")})
    sandboxed = task.agent is not None and bool(task.agent.tool_runner.host_paths)
    if served and sandboxed:
        raise CalibrationError(
            f"task {task.id!r}: hls calibrate does not calibrate a task whose gate's reference is a "
            f"served engine ({', '.join(served)}) and whose trajectories run in sandboxes. A run of "
            f"the task uses its default, defaults/calibration.json beside its module, which fits "
            f"the task as its module declares it, without task overrides that change its workload.")
    validate_pool(list(engines))
    kind = calibration_kind(task) or "flips"
    n_servings: int | None = None
    if kind == "report":
        n_servings = (DEFAULT_REPORT_SERVINGS if servings_per_engine is None
                      else servings_per_engine)
        if n_servings < 1:
            raise CalibrationError(
                f"servings_per_engine must be at least 1, got {n_servings}")
        report_verifier(task)
    elif servings_per_engine is not None:
        raise CalibrationError(
            "servings_per_engine does not apply to a token-level calibration: its "
            "serving count is an output of the exposure counter "
            "(exposure_target(mu_per_mtok) and MIN_SERVINGS), so that the precision "
            "check is a single look at a pre-committed size. Turn --mu-per-mtok "
            "instead.")
    pinned_reference(task)
    return kind, n_servings


def assemble(campaign: Campaign, *, task: TaskSpec, module: str, engines: list[str],
             tol: Tolerances, stack: StackInfo, preflight: PreflightResult,
             engine_configs: dict[str, str], sessions_per_draw: int | None,
             stopped: StopReason) -> Calibration:
    """Assemble the campaign into the artifact, refusing anything that fails its own checks."""
    stats = {e: campaign.stats.get(e, []) for e in engines}
    thin = [e for e, draws in stats.items() if len(draws) < 2]
    if thin:
        raise CalibrationError(
            f"engines {sorted(thin)} have fewer than 2 scored servings; the "
            f"between-serving spread the rate ceiling rests on is not estimable from "
            f"that.")
    rate_side = _rate_side(stats)
    # One grid for the whole assembly: the recurrence census (``pool_warnings``) reads it
    # back off ``curve_side``, so the ceilings and the census count on the same rungs.
    levels = pool_grid(stats)
    curve_side = _curve_side(stats, levels)
    q_max = max(r.mean for r in rate_side.values())
    if q_max <= 0:
        # Every engine agreed with the reference at every position. Serving more cannot
        # separate a genuinely noiseless setup from a scoring path that is not
        # measuring anything, and a 0 ceiling would fail any trace with one mismatch.
        raise CalibrationError(
            "every pool engine agreed with the pinned reference at every position, so "
            "the measured ceiling is 0 -- a degenerate gate that fails any trace with a "
            "single mismatch. Either the scoring path is not measuring what it should, "
            "or the pool carries no independent implementations; fix that rather than "
            "shipping the ceiling.")
    sizing = _sizing(rate_side, stopped)
    tail = {e: tail_check(eng.exposure, tol.mu_per_mtok)
            for e, eng in curve_side.engines.items()}
    precision = {e: precision_check(r, sizing.c_m, tol.eps) for e, r in rate_side.items()}

    short = [f"(T) {e}: {c.lhs:.4g} of {c.rhs:.4g} tokens"
             for e, c in tail.items() if not c.passed]
    if short:
        raise CalibrationError(
            f"calibration stopped ({stopped}) short of its exposure target:\n  "
            + "\n  ".join(short))
    thin_servings = [f"{e}: {r.draws}" for e, r in rate_side.items()
                     if r.draws < MIN_SERVINGS]
    if thin_servings:
        raise CalibrationError(
            f"calibration reached its exposure target but on fewer than {MIN_SERVINGS} "
            f"servings ({', '.join(thin_servings)}); the between-serving spread (P) is "
            f"computed from is not a usable estimate from that few.")
    failed = [f"{e}: spread {rate_side[e].sd / rate_side[e].mean:.1%} of the mean rate "
              f"over {rate_side[e].draws} servings, so {c.lhs:.4g} > {c.rhs:.4g}"
              for e, c in precision.items() if not c.passed]
    if failed:
        raise CalibrationError(
            f"calibration failed its precision target (eps={tol.eps}):\n  "
            + "\n  ".join(failed)
            + "\nNo further serving fixes this -- more servings shrink the confidence "
              "factor on the spread, not the spread itself. Either raise --eps (a "
              "deliberate widening of calibration's own error budget, never of the "
              "deployed headroom), or find and remove what makes the engine that "
              "variable.")

    # The config line is a build-time fact about how the member was asked to serve; the
    # version is what its own process reported while it served. A pool member whose
    # servings disagree on the version was two different builds, which the ceilings
    # cannot be a claim about.
    engine_records: dict[str, EngineRecord] = {}
    for engine, draws in stats.items():
        versions = sorted({d.serving_version for d in draws})
        if len(versions) > 1:
            raise CalibrationError(
                f"pool member {engine!r} served under {len(versions)} different package "
                f"versions ({', '.join(versions)}); the ceilings are a claim about one "
                f"build. Re-drive the servings taken under the odd one out.")
        engine_records[engine] = EngineRecord(version=versions[0],
                                              config=engine_configs[engine])

    capped_events = sum(d.capped_events for draws in stats.values() for d in draws)
    capped_tokens = sum(d.capped_tokens for draws in stats.values() for d in draws)
    # Per engine, in serving (draw-index) order -- the order ``Campaign.add`` keeps
    # each engine's list in. serving_seed_of() is engine-independent, so vllm's serving 3
    # and sglang's serving 3 are clocked from the same seed; collapsing the pool into
    # one set would halve the recorded count on a two-engine campaign and destroy the
    # index alignment that lets a named serving be re-driven.
    seeds = {engine: tuple(d.seed for d in draws) for engine, draws in stats.items()}
    return Calibration(
        task=module,
        workload_fingerprint=workload_fingerprint(task),
        workload=workload_payload(task),
        gpu_kind=task.hardware.accelerator.kind,
        stack=stack,
        preflight=preflight,
        engines=engine_records,
        draw=DrawSpec(max_sessions=sessions_per_draw, serving_seeds=seeds),
        tolerances=tol,
        rate_side=rate_side,
        curve_side=curve_side,
        sizing=sizing,
        capped_text=CappedText(events=capped_events, tokens=capped_tokens),
        q_max=q_max,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _precision_shortfall(engine: str, rec: EngineReport, eps: float) -> str:
    """One engine's line in the precision refusal: how far its measured error is from
    the target, and how many servings would meet it.

    The count scales as the square of the shortfall because the error falls as the
    square root of the reply count. An engine that measured no divergence at all gets
    no count: the ratio the target is about is undefined there, and more servings of an
    engine whose reports match the reference exactly would not create one."""
    if rec.mean_kl <= 0:
        return (f"{engine}: se {rec.se_kl:.4g} against a mean of {rec.mean_kl:.4g}, "
                f"which no number of servings makes a ratio of")
    want = math.ceil(rec.n_servings * (rec.se_kl / (eps * rec.mean_kl)) ** 2)
    return (f"{engine}: se {rec.se_kl:.4g} is {rec.se_kl / rec.mean_kl:.1%} of the mean "
            f"{rec.mean_kl:.4g} over {rec.n_servings} servings, so {want} servings are "
            f"needed")


def assemble_report(campaign: Campaign, *, task: TaskSpec, module: str,
                    engines: list[str], stack: StackInfo, eps: float,
                    eot_expected_max: float, alpha: float,
                    n_servings: int, stopped: str) -> ReportCalibration:
    """Assemble a report campaign into its artifact, refusing anything that fails its
    own checks.

    One number is measured: ``kappa``, the highest one-sided upper limit over the
    pool's engines, so the ceiling covers every trusted stack rather than their
    average. Each engine's limit is its mean divergence per measured position plus
    ``UPPER_Z`` standard errors, and the standard error is clustered by reply
    (``PooledReport``).

    One more is chosen rather than measured: the end-of-turn count's threshold
    (``_end_of_turn_check``), which decides which positions that check reads and
    nothing about what it reads them against.

    Refuses, each naming its fix: an engine that served nothing, an engine no reply of
    which could be measured, a pool whose measured divergence is zero, an engine
    whose precision target ``se <= eps * mean`` was missed -- that one saying how
    many servings would meet it, since unlike the flip campaign's spread this error
    does shrink with more servings -- and a pool member whose own end-of-turn count
    already rejects, which would mean the check's null is wrong on stacks known to be
    faithful.

    Contract violations do not refuse it. A pool member is a trusted stack, so a reply
    whose report does not describe its own emission is a finding about the member's
    report shape; it is logged, counted, and left out of every sum, and the ceiling is
    measured on the replies that were evidence."""
    stats = {e: campaign.stats.get(e, []) for e in engines}
    thin = [e for e, draws in stats.items() if len(draws) < n_servings]
    if thin:
        raise CalibrationError(
            f"engines {sorted(thin)} have fewer than the {n_servings} servings this "
            f"campaign asked for (stopped: {stopped}).")
    pooled: dict[str, PooledReport] = {}
    for engine, draws in stats.items():
        agg = PooledReport()
        for d in draws:
            agg.extend(d.events)
        pooled[engine] = agg
    blank = [e for e, agg in pooled.items() if not agg.n_tokens]
    if blank:
        # The reasons, not just the count: this refusal is the whole diagnosis a member
        # serving the wrong report shape gets, and the reasons are otherwise only inside
        # the gzipped per-reply records.
        named = "; ".join(
            f"{e}: {reason}" for e in sorted(blank)
            for reason in _violation_reasons(
                [ev for d in stats[e] for ev in d.events], limit=2))
        raise CalibrationError(
            f"engines {sorted(blank)} produced no gradable position at all: every reply "
            f"violated the report contract, so the ceiling would be measured on "
            f"nothing. Reasons: {named or 'none recorded'}. "
            f"Check that the member reports logprobs at the contracted width and re-drive.")
    unbounded = {e: sorted(ev.event_key for d in stats[e] for ev in d.events
                           if not math.isfinite(ev.kl_sum))
                 for e in pooled}
    unbounded = {e: ids for e, ids in unbounded.items() if ids}
    if unbounded:
        # Named here or nowhere. Nothing downstream catches it: the precision test is
        # se <= eps * mean, which an infinite mean satisfies vacuously, and a non-finite
        # float serializes as null, so the campaign would write an artifact whose
        # ceiling is absent and every arm read against it would be graded on nothing.
        named = "; ".join(
            f"{e}: {len(ids)} of {pooled[e].n_replies} replies "
            f"({', '.join(ids[:_UNBOUNDED_NAMED])}"
            f"{', and others' if len(ids) > _UNBOUNDED_NAMED else ''})"
            for e, ids in sorted(unbounded.items()))
        raise CalibrationError(
            f"the measured divergence is not a finite number for {sorted(unbounded)}, "
            f"so no ceiling can be read off it. {named}. A reply sums to infinity when "
            f"the reference prices one of the atoms its report weights at zero, which "
            f"on this partition means the served code fell outside the reference's own "
            f"top-k. Inspect those replies before re-driving: more servings will not "
            f"make an infinite mean finite.")
    for engine, agg in pooled.items():
        if agg.n_violations:
            log.warning(
                "pool member %r: %d of %d replies violated the report contract and are "
                "outside the measurement; the ceiling is measured on the other %d. "
                "Reasons: %s",
                engine, agg.n_violations, agg.n_replies,
                agg.n_replies - agg.n_violations,
                "; ".join(_violation_reasons(
                    [ev for d in stats[engine] for ev in d.events])))
        log.info("pool member %r: mean_kl=%.6g +/- %.3g over %d measured positions in "
                 "%d replies (%d servings), z=%.3g", engine, agg.mean_kl, agg.se_kl,
                 agg.n_tokens, agg.n_replies, len(stats[engine]), agg.z)
    # A pool member whose servings disagree on the version was two different builds,
    # which one ceiling cannot be a claim about.
    for engine, draws in stats.items():
        versions = sorted({d.serving_version for d in draws})
        if len(versions) > 1:
            raise CalibrationError(
                f"pool member {engine!r} served under {len(versions)} different package "
                f"versions ({', '.join(versions)}); the ceiling is a claim about one "
                f"build. Re-drive the servings taken under the odd one out.")
    if max(agg.mean_kl for agg in pooled.values()) <= 0:
        # Every engine's reports matched the reference exactly at every position.
        # Serving more cannot separate that from a scoring path that is not measuring
        # anything, and a 0 ceiling fails any trace whose reports differ at all.
        raise CalibrationError(
            "every pool engine's reports agreed with the pinned reference exactly, so "
            "the measured ceiling is 0 -- a degenerate gate that fails any trace whose "
            "reports differ from the reference at all, which two correct "
            "implementations of one checkpoint always do. Either the scoring path is "
            "not measuring what it should, or the pool carries no independent "
            "implementations; fix that rather than shipping the ceiling.")
    # The served-outside rate is the other measured constant, and it is measured on the
    # same records. A member that never served outside its own atoms did not sample at
    # the contract's temperature, or reported a width the partition never cuts; either
    # way the rate is not a measurement, and a 0 ceiling would fail a faithful trace on
    # its first tail token.
    never_outside = [e for e, agg in pooled.items() if not agg.n_outside]
    if never_outside:
        raise CalibrationError(
            f"engines {sorted(never_outside)} served no token outside their own reported "
            f"atoms over the whole campaign, so the served-outside ceiling would be 0 and "
            f"fail any faithful trace at its first tail token. A sampler at the "
            f"contract's temperature lands outside any 19 atoms at their tail mass; "
            f"check the member's sampling parameters and report width and re-drive.")
    measured = {
        engine: EngineReport(
            n_servings=len(stats[engine]), n_replies=agg.n_replies,
            n_tokens=agg.n_tokens, mean_kl=agg.mean_kl, se_kl=agg.se_kl,
            upper=agg.mean_kl + UPPER_Z * agg.se_kl,
            precision_passed=agg.se_kl <= eps * agg.mean_kl,
            outside_rate=agg.outside_rate, se_outside=agg.se_outside,
            outside_upper=agg.outside_rate + UPPER_Z * agg.se_outside)
        for engine, agg in pooled.items()}
    failed = [_precision_shortfall(engine, rec, eps)
              for engine, rec in measured.items() if not rec.precision_passed]
    if failed:
        raise CalibrationError(
            f"calibration failed its precision target (eps={eps}):\n  "
            + "\n  ".join(failed)
            + "\nThe standard error falls as the square root of the reply count, so "
              "re-run hls calibrate with --servings-per-engine at the count named above, "
              "or raise --eps -- a deliberate widening of calibration's own error budget, "
              "never of the deployed headroom.")
    for engine, agg in pooled.items():
        _omega2, cvm_p = agg.uniformity()
        log.info("pool member %r diagnostics: cvm_p=%.3g; served outside the reported "
                 "atoms at %d of %d positions (rate %.4g +/- %.2g)", engine, cvm_p,
                 agg.n_outside, agg.n_tokens, agg.outside_rate, agg.se_outside)
    return ReportCalibration(
        task=module,
        workload_fingerprint=workload_fingerprint(task),
        workload=workload_payload(task),
        gpu_kind=task.hardware.accelerator.kind,
        stack=stack,
        engines=measured,
        kappa=max(rec.upper for rec in measured.values()),
        outside_ceiling=max(rec.outside_upper for rec in measured.values()),
        eps=eps,
        end_of_turn=_end_of_turn_check(stats, pooled, expected_max=eot_expected_max,
                                       alpha=alpha),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _end_of_turn_check(stats: dict[str, list[ReportDrawStats]],
                       pooled: dict[str, PooledReport], *, expected_max: float,
                       alpha: float) -> EndOfTurnCheck:
    """Choose the end-of-turn count's threshold and record what the pool measured at it.

    The threshold is the largest published candidate (``EOT_EPS_GRID``) at which no
    trusted serving expects more than ``expected_max`` endings among the positions it
    prices at or below it. Larger is better -- it prices more positions, so a deviation
    has fewer places to hide -- and the expectation is what bounds the check's own
    noise, so the two are traded off by declaring the expectation and reading the
    threshold off the traffic.

    Refuses when no candidate holds the declared expectation, and when a pool member's
    own count would reject at ``alpha``. The second one is the check's assumption being
    tested on stacks known to be faithful: if a trusted engine ends its turns more often
    than the reference prices, the null is wrong and the check would fail honest
    servers."""
    # One pool per serving, built once: a pool holds the serving's positions over
    # again, so building them inside the loop would re-read the whole campaign at every
    # candidate.
    servings = [d.pooled() for draws in stats.values() for d in draws]
    expectations = {eps: max(p.expected_end_of_turns(eps) for p in servings)
                    for eps in EOT_EPS_GRID}
    within = [eps for eps, e in expectations.items() if e <= expected_max]
    if not within:
        smallest = min(EOT_EPS_GRID)
        raise CalibrationError(
            f"no published end-of-turn threshold holds the expected count under the "
            f"declared {expected_max}: the smallest, {smallest}, already expects "
            f"{expectations[smallest]:.3g} endings on a serving of this task's traffic. "
            f"Either this traffic ends turns at prices the grid cannot separate, or the "
            f"declaration is too tight -- raise --eot-expected-max, which widens the "
            f"count's own noise (its spread is about the square root of the "
            f"expectation).")
    eps = max(within)
    counts = {engine: agg.end_of_turn(eps) for engine, agg in pooled.items()}
    for engine, count in counts.items():
        log.info("pool member %r end-of-turn count at eps=%g: expected %.3g, observed "
                 "%d over %d priced positions (p=%.3g)", engine, eps, count.expected,
                 count.observed, count.n_priced, count.p_value)
    rejected = {e: c for e, c in counts.items() if c.p_value <= alpha}
    if rejected:
        raise CalibrationError(
            "the end-of-turn count rejects a trusted pool member at eps="
            f"{eps} (alpha={alpha}), so its null does not hold on stacks known to be "
            "faithful and the deployed check would fail honest servers: "
            + "; ".join(f"{e}: expected {c.expected:.3g}, observed {c.observed} over "
                        f"{c.n_priced} priced positions, p={c.p_value:.3g}"
                        for e, c in sorted(rejected.items()))
            + ". Investigate the member's endings against the reference before "
              "shipping this calibration.")
    ids = sorted({i for draws in stats.values() for d in draws for i in d.eot_ids})
    return EndOfTurnCheck(
        ids=ids, eps=eps, expected_max=expected_max,
        engines={engine: EndOfTurnEngine(priced_positions=c.n_priced,
                                         expected=c.expected, observed=c.observed,
                                         p_value=c.p_value)
                 for engine, c in counts.items()})


def write_calibration(cal: Calibration | ReportCalibration, directory: Path) -> Path:
    """Write ``cal`` to ``directory/calibration.json`` (indent=2, JSON mode) and return the
    path: the calibration a calibration's aggregate role writes into its attempt's outputs, which
    its stage writes into the store's measured area too, where ``read_calibration`` reads it
    back; ``model_dump(mode="json")`` keeps it plain-JSON (no Python-only floats)."""
    path = Path(directory) / CALIBRATION_FILE
    path.write_text(json.dumps(cal.model_dump(mode="json"), indent=2))
    return path
