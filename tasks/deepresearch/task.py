"""deepresearch: serving ``google/gemma-4-31B-it`` as a deep-research agent.

The harness owns the multi-turn tool loop (``DeepResearchFramework``) and the corpus browser
tool. The artifact serves one LLM turn per HTTP call and never executes a tool.

BASELINE INFERENCE: modeled on https://github.com/RulinShao/elastic-inference, which serves the
deep-research agent with vLLM, prefix caching and a 128K context, and fans agentic requests
across workers. HLS reproduces the fan-out with ``Poisson`` arrival and the concurrent
``TrajectorySessionDriver``; the measured baseline is a single prefix-caching vLLM engine
mirroring that approach.

MODEL: Gemma 4 31B is dense (31.3B parameters, 62.6 GB in bf16, one H200) with native
function calling. It renders tool calls in its own syntax, which the research loop parses;
the delimiters are special tokens, and the harness's member engines keep them in the reply
text so the loop can read them.

Thinking stays on: ``MODEL.chat_template_kwargs`` pins ``{"enable_thinking": True}``,
which the chat template renders as a ``<|think|>`` delimiter in the system turn.
The reference engines, every calibration pool member and the measured baseline all
render with the pin, so a faithful artifact emits the same delimiter under
teacher-forcing. Thinking stays on also because a serving must produce enough tokens to
reach the calibration's exposure target, and with thinking off it falls far short.

The decode is sampling at temperature one from the full distribution, not greedy.
Under greedy decode this checkpoint falls into exact token cycles that run replies to
the output cap; sampling has none and matches how the model is served in practice.

CORRECTNESS GATE: ``ReportFaithfulnessVerifier``. Every reply carries the server's own
report (the sampled token's log-probability and its top alternatives at every position),
the reference computes its distribution on the same prefix, and the verdict tests the mean
divergence between the two against a calibrated ceiling and the served tokens for being draws
from the report. ``DECODE`` below is the one value the oracle, the calibration pool and the
brief all read, and the measured baseline's server samples the same way. The ceilings come
from a ``calibration.json``, the store's or the task's default in defaults/, fingerprinted to this
task's model, contract, workload, split and seed; a stale one is not used, and ``hls run``
measures one when none fits.

Calibrate (one serving of the traffic per trusted stack, driven then scored; the
report gate's calibration sets two measured constants, the ceiling on the mean
divergence and the ceiling on how often a served token falls outside the reported
alternatives, and picks the end-of-turn count's threshold from the same servings):

    hls calibrate tasks.deepresearch.task --engines vllm,sglang --fanout 2

The per-serving lines in the log carry the mean divergence and its standard error,
the contract violations and the two sampling statistics; the summary adds the
end-of-turn threshold and each member's own count at it.

The browser tool searches the BrowseComp-Plus corpus with the Qwen3-Embedding-8B
index and opens documents from it. The corpus, the vectors and the encoder arrive as a
provisioned bundle mounted at /data/tools/browsecomp: the dev variant in the build
sandbox, the full variant in the driver of a graded run and in every calibration pool
member, which therefore run sealed.

Run: hls run tasks.deepresearch.task --arm <arm>
"""

from __future__ import annotations

import json

from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec, Budget, BuildAllocation,
                                   EvaluationAllocation, HighestAvailable)
from hls.core.decode import Sample
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.drive.bounds import VerifyPeak, bounds_for
from hls.reference.references import HFReference
from hls.scoring import SloLatencyScorer
from hls.spec.agent import AgentSpec
from hls.spec.baseline import Baseline
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.tools import LocalCallableToolRunner, ToolSpec
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.workload import (
    GammaInterTurn, Poisson, ShortOutputCap, TrajectoryMakespanSLO, Workload,
)
from hls.verify import ReportFaithfulnessVerifier
from hls.verify.report import TOP_LOGPROBS_REQUESTED
from hls.recipes.catalog import FLASHINFER_CUBINS, SERVER_ENV, ImageSet, Served
from hls.recipes.model import LayerImage

from tasks.deepresearch.assets import BrowseCompBundle
from tasks.deepresearch.corpus_browser import CorpusBrowserTool
from tasks.deepresearch.datasets import BROWSECOMP
from tasks.deepresearch.framework import (
    FIND_TOOL_NAME, OPEN_TOOL_NAME, SEARCH_TOOL_NAME, DeepResearchFramework,
)
from tasks.deepresearch.limits import MAX_NEW_TOKENS
from tasks.deepresearch.metric import (
    ReportOnlyMetricVerifier, ReportOnlyTrajectoryCompletionVerifier,
    RetrievalAnswerEM, TrajectoryServingMetrics,
)
from tasks.deepresearch.selftest_brief import DeepResearchBrief

# -----------------------------------------------------------------------------
# MODEL SEAM.
MODEL_ID = "google/gemma-4-31B-it"
# Hub commit of 2026-07-20.
MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
MAX_TURNS = 100                               # open-horizon: many turns/trajectory
# Long enough for real BrowseComp research, but finite so long-running
# trajectories still answer before MAX_TURNS.
MAX_SEARCH_TURNS = 80
#: Session arrival rate, sessions per second: the highest rate at which the declared
#: vLLM baseline's p99 session makespan stays under the 1,800 s target, with one H200
#: for the server and one for the retriever, the 50 test sessions, and no admission cap
#: in the load generator or the engine; at 0.0175 and above p99 exceeds the target. The
#: credit the speedup score divides by comes from `hls baseline-credit` on this module
#: at this rate, in baseline.json beside this file.
POISSON_RATE_PER_SEC = 0.015
#: Wall-time budget per trajectory the score normalizes makespans by. It cancels in the
#: speedup and survives only in the attainment diagnostic, so it sits where the products
#: this task models put it: 30 minutes, the upper end of the 5 to 30 minutes OpenAI
#: quotes for a deep-research answer and the per-task wall clock DeepWeb-Bench allows.
TRAJECTORY_BUDGET_SEC = 1800.0
# Total wall-clock bound on one drive of this workload (6 h). Sized off the baseline
# rather than a round number: the vLLM baseline drives the graded sessions in about
# 4,600 s on one H200, so 6 h admits an artifact roughly five times slower than the
# stack it is scored against. Nothing
# else bounds the total -- turns are serial and tool waits are neither a turn nor a
# request -- so without it a slow enough artifact drives until the orchestrator's
# allocation is killed, leaving no trace and no score.
DRIVE_TIMEOUT_SEC = 21600.0
SEED = 42
#: The context window, in tokens, the task declares (``TaskSpec.max_context_len``): the
#: calibration pool's trusted members are built at it, and the framework budgets every
#: trajectory against it. 131,072 matches the elastic-inference serve scripts.
MAX_CONTEXT_LEN = 131072

MODEL = ModelSpec(base_checkpoint=MODEL_ID, revision=MODEL_REVISION, arch="TX",
                  chat_format="gemma", chat_template_kwargs={"enable_thinking": True})
HARDWARE = Hardware(
    accelerator=AcceleratorSpec(kind="H200", count=1, memory_gb_per_device=141),
    cpu_count=64, memory_gb=512, storage_gb=2048,
)

#: The decode every server in this task implements and the reference grades against:
#: one draw per position from the softmax of the model's logits over the whole
#: vocabulary at temperature one, with no truncation and no penalty. The oracle, the
#: calibration pool and the brief all read this one value, and the measured baseline's
#: server samples the same way.
DECODE = Sample(temperature=1.0)

# BrowseComp (BrowseComp-Plus) is the deep-research eval corpus: 830 challenging
# queries with short verifiable answers (decrypted at provisioning from
# ``Tevatron/browsecomp-plus``). The dev/test partition is carved from the ``test``
# split by a hash of each row's id (``hls.spec.datasource.partition_by_hash``); the
# task seed plays no part in it.
DATASET = BROWSECOMP
#: Sessions in the graded test split. At the arrival rate above a drive of 50 sessions
#: is 0.9 hours of arrivals plus the last sessions' makespans, and it serves about
#: 0.95M tokens. The verifier's divergence test and its calibration precision need
#: under 0.2M tokens on this model.
TEST_SESSIONS = 50
DATA_SOURCE = DataSource(
    dataset=DATASET,
    split=SplitSpec(dev=Count(value=200), test=Count(value=TEST_SESSIONS)))

# Browser tools in elastic-inference's BrowserSession semantics: search the corpus,
# open a result, find text in the open page. These descriptions ride in every turn's
# tool definitions, which the chat template folds into the prompt the gate replays.
BROWSER_SEARCH_TOOL_SPEC = ToolSpec(
    name=SEARCH_TOOL_NAME,
    description=(
        "Corpus search over BrowseComp-Plus (Qwen3-Embedding-8B index). Returns "
        "numbered results that can be opened by id."
    ),
    json_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
BROWSER_OPEN_TOOL_SPEC = ToolSpec(
    name=OPEN_TOOL_NAME,
    description=(
        "Open a search result by id or URL from the corpus, or scroll an opened "
        "page by cursor."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "id": {},
            "cursor": {"type": "integer"},
            "loc": {"type": "integer"},
            "num_lines": {"type": "integer"},
        },
    },
)
BROWSER_FIND_TOOL_SPEC = ToolSpec(
    name=FIND_TOOL_NAME,
    description="Find text in the currently opened page.",
    json_schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "cursor": {"type": "integer"},
        },
        "required": ["pattern"],
    },
)
_BROWSER = CorpusBrowserTool()
TOOL_RUNNER = LocalCallableToolRunner(
    specs=[BROWSER_SEARCH_TOOL_SPEC, BROWSER_OPEN_TOOL_SPEC, BROWSER_FIND_TOOL_SPEC],
    callables={
        SEARCH_TOOL_NAME: _BROWSER.search,
        OPEN_TOOL_NAME: _BROWSER.open,
        FIND_TOOL_NAME: _BROWSER.find,
    },
    warmup=_BROWSER.warmup,
)
ASSETS = BrowseCompBundle()
# Think time between turns, standing in for the network latency of a live browser tool
# (a search plus a page fetch): mean 2.0 s, median about 1.7 s, p95 about 4.7 s.
INTER_TURN_DELAY = GammaInterTurn(shape=2.0, scale=1.0, max_gap=15.0)

# OpenAI chat/completions over SSE (transport="http-sse"). Streaming is what
# makes per-step TTFT/ITL measurable (the client records t_first_packet +
# per-token packet_timestamps, reduced by TrajectoryServingMetrics). The server
# keeps a unary path for the readiness probe (which posts a non-stream body).
# choices[].token_ids is mandatory on every response, and decode_stream reassembles
# the reply's full id sequence from the per-chunk ids. The contract asks for the report
# the gate grades: the served token's log-probability and its top alternatives at every
# position, with token ids.
CONTRACT = RequestResponseContract(
    path="/v1/chat/completions",
    input_modalities=frozenset({"text"}),
    output_modalities=frozenset({"text"}),
    transport="http-sse",
    protocol=OpenAIChatProtocol(return_logprobs=True, top_logprobs=TOP_LOGPROBS_REQUESTED),
)

# VERIFIERS (the first AND-gates; the rest are report-only):
#
#  1. ReportFaithfulnessVerifier -- the sole pass/fail correctness gate. Every LLM call of
#     every trajectory is one item: the conversation the request carried (including that
#     turn's tool definitions, which the chat template folds into the prompt), the
#     harness-stamped output cap, the artifact's emitted ids and its report. Its tolerance
#     and test size are its defaults. The reference is the served checkpoint over this
#     task's decode; query_chunk is the width the task's calibration was
#     measured at and the only one a full-context trajectory fits in on the reference card.
#  2. ReportOnlyMetricVerifier(RetrievalAnswerEM) -- dataset benchmark EM (diagnostic).
#  3. ReportOnlyMetricVerifier(TrajectoryServingMetrics) -- serving telemetry.
#  4. ReportOnlyTrajectoryCompletionVerifier -- effective answer rate plus the
#     stricter boxed-answer rate (diagnostic).
VERIFIERS = [
    ReportFaithfulnessVerifier(reference=HFReference(decode=DECODE, query_chunk=512)),
    ReportOnlyMetricVerifier(metric=RetrievalAnswerEM(), label="dataset_answer_em"),
    ReportOnlyMetricVerifier(metric=TrajectoryServingMetrics()),
    ReportOnlyTrajectoryCompletionVerifier(min_completion_rate=0.95),
]

# Per-trajectory makespan SLO, which normalizes each trajectory's wall time against
# the budget rather than gating on it. Poisson fans trajectories out so a contestant
# with continuous batching + prefix caching can overlap them.
WORKLOAD = Workload(
    arrival=Poisson(rate_per_sec=POISSON_RATE_PER_SEC),
    inter_turn_delay=INTER_TURN_DELAY,
    output_length_cap=ShortOutputCap(cap_tokens=MAX_NEW_TOKENS),
    slo=TrajectoryMakespanSLO(target_sec=TRAJECTORY_BUDGET_SEC),
)
# Objective: speedup over the task's baseline serving stack in trajectory makespan --
# the geometric mean over trajectories of 1/r, with r the makespan against
# TRAJECTORY_BUDGET_SEC, divided by the same quantity for the baseline. The baseline's
# value comes from baseline.json, the store's or the task's default in defaults/ (measured
# by `hls baseline-credit`), and is injected by the CLI; the budget cancels between the
# two, so it survives only
# in the attainment diagnostic. Not a goodput scorer: this task's arrival is open-loop
# and fixed, so trajectory throughput is the driver's choice rather than the server's,
# and a metric that plateaus once every makespan is met has a ceiling any competent
# server ties at. Any failed request fails the run. See hls.scoring.SloLatencyScorer.
SCORER = SloLatencyScorer()

# The agentic seam: harness-side framework + tool runner + per-trajectory budget.
# The framework budgets a trajectory against the window the TaskSpec below declares,
# which it takes from the task, and trajectory length is what this task scores.
#
# The load generator admits every LLM call as it comes. The arrival process is open-loop
# and is the only thing that should decide how much work the server sees at once: a
# semaphore in the load generator would compete with it, and past the rate where the mean
# concurrency approaches its ceiling a load ladder would measure the semaphore rather than
# the server. Concurrency is excluded from the workload fingerprint
# (`hls.verify.calibration.agent_payload`), so it changes no calibration.
AGENT = AgentSpec(
    framework=DeepResearchFramework(finalize_after_search_turns=MAX_SEARCH_TURNS),
    tool_runner=TOOL_RUNNER,
    max_turns=MAX_TURNS,
    max_concurrent_llm=None,
    assets=ASSETS,
    drive_timeout_sec=DRIVE_TIMEOUT_SEC,
)

#: The launch flag a ``Baseline.serve_command`` carries the model's
#: ``ModelSpec.chat_template_kwargs`` pin in, as JSON: the measured baseline's server takes
#: it, so its one server file renders prompts for whichever weights the harness mounts.
CHAT_TEMPLATE_KWARGS_FLAG = "--chat-template-kwargs"
# The image string is the identity the baseline credit records; the image served is the
# measured baseline the image catalog names. The chat-template pin travels to it as a launch
# flag because the baseline is an artifact like any other: it reads no harness spec. The
# decode needs no flag: the measured baseline's server samples at temperature one and
# reports as the contract asks.
BASELINE = Baseline(image="hls-baseline-deepresearch",
                    serve_command=[CHAT_TEMPLATE_KWARGS_FLAG, json.dumps(MODEL.chat_template_kwargs)])

# What one graded drive may make the harness hold (hls.drive.bounds), at 8,192 tokens a request.
# Each of a drive's TEST_SESSIONS (50) trajectories has one LLM call in flight at a time, and a
# reply at that cap is about 15.5 MB on the wire; all 50 held unwritten at once at the stream's
# 13.5 trace bytes per byte would make the cap 12 GiB. The verify role cannot hold that: it holds
# 59.25 GiB with its reference loaded, gemma-4-31B-it's 58.25 GiB
# checkpoint and 1 GiB of engine, and 26.3 bytes besides per byte of trace. Three report-only
# verifiers each keep the whole trace, and a reply's `gpu` metadata, of any size the protocol
# keeps, is copied into one's diagnostics while the trace is still held. That makes the cap
# 3 GiB, about twice a 50-session drive's trace (about 1.5 GiB).
DRIVE_BOUNDS = bounds_for(CONTRACT, trace_bytes=1_623_366_771, held_reply_bytes=TEST_SESSIONS * 15_518_126,
                          trace_request_bytes=160 * 2**20,
                          verify_peak=VerifyPeak(base_bytes=round(59.25 * 2**30), bytes_per_byte=26.3))

TASK = TaskSpec(
    id="deepresearch", model=MODEL, hardware=HARDWARE, contract=CONTRACT,
    max_context_len=MAX_CONTEXT_LEN,
    workload=WORKLOAD, verifiers=VERIFIERS, scorer=SCORER,
    data_source=DATA_SOURCE, seed=SEED, agent=AGENT, baseline=BASELINE,
    drive_bounds=DRIVE_BOUNDS,
    # 20 GiB: the measured resident peak of the role that builds the full browsecomp tool bundle,
    # 15.66 GiB, plus a quarter, rounded up; the dev bundle's role holds 11.40 GiB and the rows'
    # 5.77, each on 4 CPUs in its own cgroup as the stage runs it.
    provision_memory_gib=20,
    metadata={"theme": "deep-research", "host": HARDWARE.accelerator.kind,
              "harness": "DeepResearch",
              # Deep research shares >=95% of the prompt turn-over-turn (system +
              # ReAct history + accumulated tool responses): an EXTREME prefix-cache
              # profile is the headline serving lever (no spec knob yet; a hint).
              "prefix_cache_profile": "EXTREME",
              "model_status": "gemma-4-31b-it-deepresearch",
              "build_to_beat": "elastic-inference vLLM deep-research baseline"},
)

# Agent wall-clock budget conveyed in the brief: a graded build is 12 hours.
#   0.5 = 30-min illustrative end-to-end run   1.0 = 1-hour shakedown
#   6.0 = short run                            12  = graded run
_AGENT_BUDGET_HOURS = 12.0

# Grace the orchestrator waits above the soft budget before it force-collects
# committed main. 0.5 h (30 min) gives eval + a clean exit headroom on a graded
# session while staying tight enough that a hung agent doesn't idle for hours.
_HARD_CAP_HEADROOM_HOURS = 0.5

# DEFAULT contestant: the implementer agent (pure config). The arm names the model
# (`hls run --arm`).
ARTIFACT = AgentProducedArtifact(
    producer=AgentProducerSpec(
        # The soft budget is what the agent sees -- it arms the /goal kickoff and the
        # get_budget tool -- and the hard cap above it is where the orchestrator collects the
        # committed tree. Both set turns=None: only wall-clock bounds the session, so a
        # productive agent is never cut off by an arbitrary turn cap.
        soft_budget=Budget(wallclock_sec=int(_AGENT_BUDGET_HOURS * 3600), turns=None),
        hard_budget=Budget(wallclock_sec=int((_AGENT_BUDGET_HOURS + _HARD_CAP_HEADROOM_HOURS) * 3600),
                           turns=None),
        # The harness's top level. Unset is refused at launch: the harnesses disagree
        # on their own defaults (claude-code emits no flag at all, codex defaults to
        # "high"), which would leave two arms of a model comparison differing on an
        # axis the record never named. A rollout override still wins.
        reasoning_effort=HighestAvailable(),
        # What an evaluation holds: the server a 1-GPU step of 8 CPUs and 180 GiB, and the GPU
        # retriever driver a 1-GPU step of 8 CPUs of its own and 64 GiB; with the runner's reserve
        # (4 GiB by default) the job asks for 248 GiB. Its verify role takes 6 h 30 min after its
        # stage's staging: about twice what the verify role takes with the reference loaded on an
        # H200 for a graded drive (about 3 h), rounded up to the half hour.
        evaluation_allocation=EvaluationAllocation(cpus=8, step_memory_gib=180, writable_gib=180, driver_memory_gib=64,
                                                   driver_cpus=8, verify_time_sec=23400),
        # What a build holds beside its H200: 8 CPUs and 180 GiB.
        build_allocation=BuildAllocation(cpus=8, memory_gib=180)),
    brief=DeepResearchBrief(),
    serve_command=[])


def _copied(image: str, *paths: str) -> dict[str, str]:
    """The files under ``images/<image>/`` that *image* copies in, by the path it holds each at:
    the directory mirrors the image's own tree."""
    return {path: f"images/{image}{path}" for path in paths}


#: The measured baseline: vLLM 0.28.0 on the cuda base, from locks/baseline.txt, behind
#: /app/server.py, which speaks the task's wire, samples at temperature one and reports each
#: token's distribution. serving_config.py holds its serving knobs, and /app/server_hf.py is the
#: same wire on the transformers engine. /app is on PYTHONPATH for the module the servers import.
_BASELINE = LayerImage(name="deepresearch-baseline", parent="cuda", pip_lock="baseline.txt", owner="deepresearch",
                       server=True,
                       files=_copied("baseline", "/app/server.py", "/app/server_hf.py",
                                     "/app/tasks/deepresearch/__init__.py",
                                     "/app/tasks/deepresearch/serving_config.py"),
                       env={**SERVER_ENV, "UV_HTTP_TIMEOUT": "300", "PYTHONPATH": "/app"})
#: setuptools alone, from locks/reference-setuptools.txt: the build backend of the reference
#: image's source build.
_REFERENCE_BACKEND = LayerImage(name="deepresearch-reference-setuptools", parent="cuda",
                                pip_lock="reference-setuptools.txt", owner="deepresearch", server=True)
#: The image the reference engine and the task's harness roles run on: torch 2.11 and transformers
#: 5.9.0, with pyarrow for the retriever's corpus, from locks/reference.txt. langdetect has no
#: wheel, so it builds from its source distribution with the setuptools of the layer below.
_REFERENCE = LayerImage(name="deepresearch-reference", parent=_REFERENCE_BACKEND.name, pip_lock="reference.txt",
                        owner="deepresearch", server=True, source_builds={"langdetect==1.0.9": ["setuptools"]},
                        env={**SERVER_ENV, "UV_HTTP_TIMEOUT": "300"})
#: The vLLM calibration member: vLLM 0.22.1 with the reference's transformers, from
#: locks/member-vllm.txt.
_MEMBER_VLLM = LayerImage(name="deepresearch-member-vllm", parent="cuda", pip_lock="member-vllm.txt",
                          owner="deepresearch", server=True, env=SERVER_ENV)
#: The SGLang calibration member: SGLang 0.5.16, from locks/member-sglang.txt, with FlashInfer
#: 0.6.14's kernels fetched into the image, where FlashInfer looks for them, so a run downloads
#: none.
_MEMBER_SGLANG = LayerImage(name="deepresearch-member-sglang", parent="cuda", pip_lock="member-sglang.txt",
                            owner="deepresearch", server=True, fetched=FLASHINFER_CUBINS["0.6.14"],
                            env={**SERVER_ENV, "FLASHINFER_CUBIN_DIR": "/opt/flashinfer-cubins"})
#: The files of this directory the build sandbox holds under ``SANDBOX_DIR``, as the brief
#: states: the loop the grader runs, its browser and retriever, the task's output cap, and the
#: self-test that drives them against the agent's server. None of them imports the harness.
SANDBOX_FILES = ("__init__.py", "research_loop.py", "corpus_browser.py", "retriever.py", "limits.py",
                 "selftest.py")
#: Where the sandbox holds them.
SANDBOX_DIR = "/app/tasks/deepresearch"
#: The layer under the sandbox: torch 2.13 and the reference's transformers on the cuda base, with
#: the kernel toolchain and the libraries the brief lists, from locks/sandbox.txt; ``SANDBOX_FILES``,
#: with /app on PYTHONPATH; and a stdio budget server with its meter under /opt/hls-budget. It adds
#: to the base's environment and sets none of its variables.
_SANDBOX = LayerImage(name="deepresearch-sandbox", parent="cuda", pip_lock="sandbox.txt", owner="deepresearch",
                      files={**{f"{SANDBOX_DIR}/{name}": name for name in SANDBOX_FILES},
                             **_copied("sandbox", "/opt/hls-budget/meter.py", "/opt/hls-budget/server.py")},
                      env={"UV_TORCH_BACKEND": "cu130", "UV_HTTP_TIMEOUT": "300", "PYTHONPATH": "/app"})
# The images the task runs on: its own layer, whose agent cap is the sandbox and the submission
# base too, since its submissions are packaged from the sandbox; the images its servers run on,
# each from this directory's locks; and its reference image for its harness roles.
IMAGES = ImageSet(
    layer=_SANDBOX.name,
    images=[_SANDBOX, _REFERENCE_BACKEND],
    package_on_sandbox=True,
    harness_on_reference=True,
    baseline=Served(image=_BASELINE, entrypoint=["python", "/app/server.py"]),
    reference=_REFERENCE,
    members={"vllm": _MEMBER_VLLM, "sglang": _MEMBER_SGLANG},
)

__all__ = ["TASK", "ARTIFACT", "CONTRACT", "AGENT", "BASELINE", "MODEL", "MODEL_ID", "MAX_NEW_TOKENS",
           "MAX_TURNS", "POISSON_RATE_PER_SEC", "TRAJECTORY_BUDGET_SEC", "HARDWARE", "DATASET",
           "DATA_SOURCE", "TEST_SESSIONS", "VERIFIERS", "WORKLOAD", "SCORER", "DECODE",
           "TOOL_RUNNER", "ASSETS", "INTER_TURN_DELAY"]
