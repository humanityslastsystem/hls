"""tab_completion -- IDE tab-completion (fill-in-the-middle) serving arena.

Four tenants, one per completion archetype: two commit-derived typing sessions
(Real-FIM-Eval ``add`` / ``edit``) and two single-request algorithmic infills
(SAFIM ``block`` / ``control``), each with its own data source, Poisson arrival and
StreamingLatencySLO budgets.

Contract: ``/v1/completions`` over http-sse with ``OpenAICompletionProtocol``. The
prompt is a raw fill-in-the-middle string already carrying the checkpoint's sentinels
(``<fim_prefix>``, ``<fim_suffix>``, ``<fim_middle>``); the server feeds it to the
model verbatim, and the protocol's ``conditioning = "raw"`` makes every reference
engine built from this task do the same. The protocol also pins the id that ends a FIM
span, so the server, the oracle and the calibration pool stop in the same place.

Correctness is the calibrated token-level gate (one shared
``TokenFaithfulnessVerifier``); every measured parameter comes from ``calibration.json``,
and the score's baseline credit from ``baseline.json``, the store's or the task's defaults
in defaults/. Each is measured on the declared accelerator, by ``hls calibrate
tasks.tab_completion.task`` and ``hls baseline-credit tasks.tab_completion.task``, which
write it into the store; ``hls run`` measures each that no file it finds fits. Recalibrate
whenever the workload
fingerprint changes (model, contract, seed, any tenant's dataset, split sizes, output
cap, session cut count or FIM template). The arrival rate, the typing pace, the SLO
budgets and the scorer sit outside the fingerprint by the harness's design
(``hls.verify.calibration.workload_payload``); a change to any of them re-measures the
baseline credit, whose artifact is keyed by the load it was driven under, and nothing
else.
"""
from __future__ import annotations

from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec, Budget, BuildAllocation,
                                   EvaluationAllocation, HighestAvailable)
from hls.core.decode import Greedy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.drive.bounds import VerifyPeak, bounds_for
from hls.egress.policy import AccessList, NetworkPolicy
from hls.prompt import ImplementerBrief, JinjaPromptTemplate
from hls.reference.references import HFReference
from hls.scoring import SloLatencyScorer
from hls.spec.baseline import Baseline
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import FimSessionDataset, SafimDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.wire import OpenAICompletionProtocol
from hls.spec.workload import (GammaInterTurn, NoDelay, Poisson, ShortOutputCap,
                               StreamingLatencySLO, Workload)
from hls.verify import TokenFaithfulnessVerifier
from hls.recipes.catalog import FLASHINFER_CUBINS, SERVER_ENV, ImageSet, Served
from hls.recipes.model import LayerImage
from tasks.sample_task.task import (_DENIED_SERVING_FRAMEWORKS, _TOOLCHAIN_REFERENCES_SHA256,
                                    _TOOLCHAIN_UPLOAD_BRIEF)

SEED = 42

# SLO-normalized latency, as multitenant_slo: r = max(TTFT/ttft_ms, TPOT/tpot_ms) per
# request against its tenant's budgets, S = exp(mean log(1/r) - B) with B the measured
# baseline credit. Any failed request fails the run.
SCORER = SloLatencyScorer()

# A fill-in-the-middle base checkpoint: a mixture of experts, 12B weights with 2.5B
# active per token. Pinned by revision so the served weights cannot move under a
# calibration. It carries no chat template, and nothing on this path would apply one.
MODEL = ModelSpec(base_checkpoint="JetBrains/Mellum2-12B-A2.5B-Base", arch="MoE",
                  chat_format="none",
                  revision="271755e48ab6b2ed0ef224eaaabe2d25275fb8ee")

# One H200 with multitenant_slo's host
# figures (tasks/multitenant_slo/task.py, HARDWARE). No brief states the host's CPUs, memory or
# storage: the build's brief states the sandbox's own and the graded
# server's are the evaluation allocation's.
HARDWARE = Hardware(accelerator=AcceleratorSpec(kind="H200", count=1, memory_gb_per_device=141),
                    cpu_count=64, memory_gb=1024, storage_gb=2048)

# The declared serving context, which the brief tells the contestant to size for and
# the baseline passes as --max-model-len. It is not the checkpoint's own window: that
# is 131,072, far more than this workload needs. No request can exceed the longest
# prompt plus the output cap, so this is the smallest envelope that never refuses one.
SERVING_CONTEXT = 32_768

# The id that ends a FIM span: <|endoftext|>, which is also the checkpoint's declared
# end-of-text in both its config and its generation config. The reference replaces the
# checkpoint's own set with this one, and the load guard refuses a pin that drops any of
# it, so this id has to be here. The vocabulary also holds a fill-in-the-middle pad
# marker, <fim_pad> id 11, but the model does not emit it under greedy decode, so
# pinning it would only add a terminator the model does not use. Adding an id is a
# contract change and so a recalibration.
STOP_TOKEN_IDS = (0,)

# The fill-in-the-middle prompt every tenant renders, in prefix-suffix-middle order.
# These are the checkpoint's own sentinel spellings, one token each; the model
# generates the middle after the last one.
FIM_PROMPT = "<fim_prefix>{{ prefix }}<fim_suffix>{{ suffix }}<fim_middle>"

CONTRACT = RequestResponseContract(
    path="/v1/completions",
    input_modalities=frozenset({"text"}),
    output_modalities=frozenset({"text"}),
    transport="http-sse",
    protocol=OpenAICompletionProtocol(stop_token_ids=STOP_TOKEN_IDS),
)

# query_chunk: the width this task's calibration was measured at. Completions here are
# short and the prompts fit easily, so nothing forces a smaller chunk, and the ceilings
# in calibration.json only describe the instrument they were taken on, so the number is
# pinned until someone re-measures them.
_ORACLE = HFReference(decode=Greedy(), query_chunk=4096)

# ---------------------------------------------------------------------------
# Arrival and mix
# ---------------------------------------------------------------------------
# Equal request shares. They put half the score weight and a seventh of the graded
# exposure on the SAFIM tenants, and a serving draws about 90% of what their corpora
# have left after the dev split (6,700 of 7,501 rows on block, of 7,349 on control).
_PORTIONS = {"commit_add": 0.25, "commit_edit": 0.25, "algo_block": 0.25, "algo_control": 0.25}

# Most requests a commit row is cut into. Inside the workload fingerprint, and the
# contestant is told the same number, so both read it from here.
SESSION_STEPS = 8

# Requests per source row: the commit tenants realize these mean step counts (over the
# non-dev pool at SESSION_STEPS); SAFIM rows are one request each.
_STEPS = {"commit_add": 5.4, "commit_edit": 5.3, "algo_block": 1.0, "algo_control": 1.0}

# Aggregate offered request load, pinned at the stock baseline's peak goodput (offered
# rate times the share of requests meeting their SLO): 24.6125 req/s is the first rate at
# which that goodput stops rising on one H200, about 19.7 req/s. The baseline meets its
# SLO on 80% of requests at this load, and the rest is the headroom a contestant
# competes for. Outside the workload fingerprint; a
# re-pin re-measures the baseline credit only.
TOTAL_RATE = 24.6125

# Per-tenant test-split sizes, in source rows. Each serving drives the whole test
# split once, so these are the per-serving draw: at equal request shares each tenant
# serves ~6,700 requests, which is 1,241 / 1,264 commit rows at 5.4 / 5.3 steps and
# 6,700 SAFIM rows. Dev is 1,280 rows per tenant, disjoint from test by hash.
#
# Graded exposure is what the model generates, not what the corpora hold. On this model
# a terminating commit reply is near 35 tokens at the median and about a quarter of
# commit requests reach the cap, so a serving is near 1.7M tokens.
_TEST_COUNTS = {"commit_add": 1241, "commit_edit": 1264, "algo_block": 6700, "algo_control": 6700}
_DEV_COUNTS = {name: 1280 for name in _PORTIONS}

_REAL_FIM_EVAL_REV = "a36062544c7ed6c4e5ffb8dad8536fc7777e1f36"
_SAFIM_REV = "be132cc15372e90b6f03a608e77f2d940e384edb"

# The max_tokens every request carries, sized as a product would size it: 256 keeps
# about 95% of terminating commit completions whole under greedy decode. The other
# reason for a small number is that a fifth of commit requests fall into a repetition
# loop, which without a cap runs to the edge of the context window; 256 bounds one at
# 2.6 s of decode at the SLO's 10 ms per token.
OUTPUT_CAP = 256

# Prompt + cap must fit SERVING_CONTEXT or vLLM answers HTTP 400, which zeroes a run.
# An upper bound on the largest prompt any tenant can send, sentinels excluded, over
# every row of all four corpora. A commit row's last step has all but one line of
# its middle typed, so the bound is its whole prefix, middle and suffix; a SAFIM row is
# one request, so it is prefix and suffix. The maximum is a SAFIM block row at 18,891,
# above the commit corpora's 17,132: at a cap this small the long single-shot rows bind,
# not the sessions. The arithmetic is 18,891 + 3 + 256 = 19,150, inside 32,768.
LONGEST_FULL_ROW_TOKENS = 18_891

# Typing pace between a session's steps, outside the fingerprint: a
# 10 s median (scale = 10 / ln 2 at shape 1) with an exponential tail cut at 60 s, in
# the range the KVCache-in-the-Wild coder trace reports for reuse gaps. Like TOTAL_RATE,
# a re-pin re-measures the baseline credit only.
_TYPING_PACE = GammaInterTurn(shape=1.0, scale=14.4270, max_gap=60.0)

# One shared gate across the tenants: pooled, so the runner grades it once. Only the
# rate headroom is authored; q_max, the level grid and the per-level ceilings come
# from calibration.json.
_TOKEN_GATE = TokenFaithfulnessVerifier(reference=_ORACLE, rate_multiplier=2.0)

# Handed to the contestant beside every tenant's dev rows: the served prompt is the
# FIM template around the row's code, which the rows alone do not show.
_FIM_DEV_REFERENCE = {"fim_prompt.txt": (
    "Every request's prompt is this Jinja template rendered around a row's code:\n\n"
    f"    {FIM_PROMPT}\n\n"
    "commit_add / commit_edit: `prefix` is the row's `prompt` column plus the lines of\n"
    "`canonical_solution` typed so far, `suffix` is the `suffix` column. A row is served\n"
    f"as a session of up to {SESSION_STEPS} requests, each cutting the middle at a later line boundary;\n"
    "the first request is the row as given.\n"
    "algo_block / algo_control: `prefix` and `suffix` are the row's own columns of those\n"
    "names (the `eval_prompt` split at its placeholder).\n")}


def _arrival(name: str) -> Poisson:
    """Session arrivals: the tenant's request share of TOTAL_RATE, divided by its
    requests per source row."""
    return Poisson(rate_per_sec=_PORTIONS[name] * TOTAL_RATE / _STEPS[name])


def _commit_tenant(name: str, subset: str, ttft_ms: float) -> Tenant:
    """A Real-FIM-Eval typing-session tenant: SESSION_STEPS-step sessions with a typing
    pace between steps, trimmed at the ramp-down."""
    return Tenant(
        id=name,
        data_source=DataSource(
            dataset=FimSessionDataset(subset=subset, revision=_REAL_FIM_EVAL_REV,
                                      max_steps=SESSION_STEPS,
                                      prompt=JinjaPromptTemplate(template=FIM_PROMPT)),
            split=SplitSpec(dev=Count(value=_DEV_COUNTS[name]),
                            test=Count(value=_TEST_COUNTS[name])),
            dev_reference=_FIM_DEV_REFERENCE),
        workload=Workload(
            arrival=_arrival(name), mean_turns_per_session=_STEPS[name],
            inter_turn_delay=_TYPING_PACE,
            output_length_cap=ShortOutputCap(cap_tokens=OUTPUT_CAP),
            slo=StreamingLatencySLO(ttft_ms=ttft_ms, tpot_ms=10.0),
            trim_rampdown=True),
        verifiers=[_TOKEN_GATE])


def _algo_tenant(name: str, subset: str, ttft_ms: float) -> Tenant:
    """A SAFIM single-request tenant: one completion per row, no inter-turn delay."""
    return Tenant(
        id=name,
        data_source=DataSource(
            dataset=SafimDataset(subset=subset, revision=_SAFIM_REV,
                                 prompt=JinjaPromptTemplate(template=FIM_PROMPT)),
            split=SplitSpec(dev=Count(value=_DEV_COUNTS[name]),
                            test=Count(value=_TEST_COUNTS[name])),
            dev_reference=_FIM_DEV_REFERENCE),
        workload=Workload(
            arrival=_arrival(name), inter_turn_delay=NoDelay(),
            output_length_cap=ShortOutputCap(cap_tokens=OUTPUT_CAP),
            slo=StreamingLatencySLO(ttft_ms=ttft_ms, tpot_ms=10.0)),
        verifiers=[_TOKEN_GATE])


# TTFT budgets follow prefill: ~2,500-token commit prompts get 600 ms, ~450-token SAFIM
# prompts 250 / 200 ms. tpot_ms=10 across the tenants. Those prompt lengths are the
# checkpoint's own tokenization.
TENANT_COMMIT_ADD = _commit_tenant("commit_add", "add", 600.0)
TENANT_COMMIT_EDIT = _commit_tenant("commit_edit", "edit", 600.0)
TENANT_ALGO_BLOCK = _algo_tenant("algo_block", "block", 250.0)
TENANT_ALGO_CONTROL = _algo_tenant("algo_control", "control", 200.0)
_TENANTS = [TENANT_COMMIT_ADD, TENANT_COMMIT_EDIT, TENANT_ALGO_BLOCK, TENANT_ALGO_CONTROL]

TASK = TaskSpec(
    id="tab_completion",
    model=MODEL, hardware=HARDWARE, contract=CONTRACT, tenants=_TENANTS, scorer=SCORER,
    seed=SEED, metadata={"theme": "tab-completion"},
    # What one drive may make the harness hold (hls.drive.bounds). A drive's replies come to
    # about 446 MB on the wire, which held unwritten at once at the stream's 13.5 trace bytes per
    # byte make the cap 6 GiB. The verify role holds 24 GiB with its
    # reference loaded, Mellum2's 23 GiB checkpoint and 1 GiB of engine, and 12.46 bytes besides
    # per byte of trace: the token gate keeps every teacher-forced reply whole, densest with a
    # logprob per token. At the 6 GiB cap that is 103 GiB.
    drive_bounds=bounds_for(CONTRACT, trace_bytes=359_684_131, held_reply_bytes=445_601_525,
                            trace_request_bytes=384 * 2**20,
                            verify_peak=VerifyPeak(base_bytes=24 * 2**30, bytes_per_byte=12.46)),
    # Stock vLLM serving the mounted weights, the baseline the score is normalized to.
    baseline=Baseline(
        image="hls-baseline-tab-completion",
        serve_command=[
            "--served-model-name", "served-model",
            "--gpu-memory-utilization", "0.92",
            # No --max-num-seqs, so vLLM admits its own default of 128.
            # The declared envelope. Passed explicitly because the checkpoint's own
            # window is 131,072, four times what this workload can reach, and sizing the
            # KV cache for it would cost the baseline throughput for nothing.
            "--max-model-len", str(SERVING_CONTEXT),
            "--dtype", "bfloat16",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            # Greedy, explicitly. This checkpoint's generation_config carries no sampling
            # settings, so vllm serve would decode at its own default temperature 1.0.
            # The values are hls.reference.engines._GREEDY_SAMPLING, the calibration
            # pool's pin.
            "--override-generation-config", '{"temperature":0.0,"top_p":1.0,"top_k":-1,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0}',
        ]),
)

_AGENT_BUDGET_HOURS = 12.0
_HARD_CAP_HEADROOM_HOURS = 0.5

# The brief's sizing sentence, built from the constants themselves rather than copied
# literals, and spliced into _BRIEF at a line boundary. Re-pinning either can never
# leave a stale number in the text the contestant reads.
_SIZING = (f"context window of {SERVING_CONTEXT:,} tokens. No request can exceed the "
           f"longest prompt plus the {OUTPUT_CAP}-token output cap, and the "
           f"checkpoint's own window is larger still")

_BRIEF = ("""{% extends "brief" %}
{% block goal %}Build a hyperoptimized fill-in-the-middle code-completion server for
`{{ model.base_checkpoint }}` under the workload below. Optimize until you reach the
physical limits of the hardware or exhaust your budget. The grader drives all tenants
concurrently against one instance of your server.

**Your score is SLO-normalized latency.** Each tenant declares its own TTFT and TPOT budgets
(listed per tenant under ## Workload). For every request:

    r      = max( TTFT / that tenant's TTFT budget , mean TPOT / that tenant's TPOT budget )
    credit = log(1 / r)          # natural log

    score  = exp( mean credit over every request - baseline credit )   higher is better.

Sessions arrive open-loop on a schedule you do not control. The commit tenants send
typing sessions: a developer accepts a completion, types a little, and the editor sends
nearly the same prompt again with a longer prefix. Each step is a separate request, sent
and scored on its own, and a later step's prompt repeats the earlier one's prefix.

**Your implementation must not fail any request.** Every request must receive a complete
streaming response. Correctness is checked at the token level: the exact token ids you
serve are teacher-forced through a bf16 reference implementation of this model --
HuggingFace transformers, batch size 1, MATH SDPA backend, cuDNN attention disabled,
the model class resolved from the checkpoint's declared architectures, scored in
4,096-token chunks against a KV cache{% if verification %}. The run fails if more than
{{ verification.flip_rate_pct }}% of your served tokens have a lower reference log-probability than the
reference's most likely token{% if verification.flip_curve_levels %}, or if the number of such tokens exceeds a
calibrated ceiling at any of several severity levels, where severity is that
log-probability gap (the ceilings scale with the run's total token count){% endif %}. Exact
ties do not count: a token equally probable under the reference is not a deviation.
{% else %}. The run fails if your token-level disagreement with that reference exceeds
the thresholds.
{% endif %}

The thresholds are calibrated on fast production-grade serving stacks. What fails is
genuine deviation: quantized weights, altered algorithms, modified sampling. The gate is
pass/fail with no credit for margin, so latency spent buying faithfulness headroom below
the ceiling is pure loss.

Serve greedy: the argmax of the raw logits, with none of the checkpoint's
generation_config defaults applied. The reference also scores where each reply stops:
a reply that ends where the reference would have kept going is a deviation like any
other.

That check is one pooled measurement over every served token of the whole run, not one
per tenant. To reproduce it on the dev split, draw your sample in the per-tenant request
proportions the workload states and let each tenant generate under its output cap:
the tenants' individual rates differ severalfold, so a sample drawn in different
proportions, or a small one, can easily read 50% higher or lower than the number you
are graded on.{% endblock %}
{% block model %}- Checkpoint: `{{ model.base_checkpoint }}`
- Mellum2 Base, a fill-in-the-middle code-completion model from JetBrains: a mixture of
  experts, 12B weights with 2.5B active per token, no chat template. Every prompt
  arrives pre-assembled with its sentinels (`<fim_prefix>`, `<fim_suffix>`,
  `<fim_middle>`); never apply a chat template to it.{% endblock %}
{% block contract %}{{ super() }}

Sizing: prompts run long, and a request must be served at whatever length it arrives --
refusing or truncating one is a failed request. Size the server for a """
+ _SIZING +
""".

The `prompt` string is the complete model input. Tokenize it with
`add_special_tokens=False`: the sentinels are already in the string, and the reference
tokenizes the same way. The request's `stop_token_ids` is `<|endoftext|>`, id 0, the
checkpoint's only end-of-text token.{% endblock %}
{% block dev_reference_note %}Beside every tenant's dev rows, `fim_prompt.txt` gives the
  template a served prompt is assembled from, and how a commit row becomes a typing
  session; the rows alone do not show either.{% endblock %}
{% block decoding %}**Decoding.** Decode greedily (argmax, no sampling). Feed the prompt
verbatim with no added BOS; stop at `<|endoftext|>`, id 0, keep that id in `token_ids`,
and never put it in the response text.{% endblock %}""")


class TabCompletionBrief(ImplementerBrief):
    """ImplementerBrief plus this task's framing: the SLO-normalized-latency objective,
    the no-failed-request rule and the token gate (``goal``), the FIM model (``model``),
    the serving envelope and the terminator rule (``contract``), raw-prompt greedy
    decoding in place of the chat decoding paragraph (``decoding``), and this task's
    own dev reference file in place of the default's few-shot example
    (``dev_reference_note``)."""

    template_str: str | None = _BRIEF


ARTIFACT = AgentProducedArtifact(
    producer=AgentProducerSpec(
        reasoning_effort=HighestAvailable(),
        upload_dir=("../assets/references",), upload_sha256=_TOOLCHAIN_REFERENCES_SHA256,
        upload_dest="/workspace/references",
        upload_brief=_TOOLCHAIN_UPLOAD_BRIEF,
        soft_budget=Budget(wallclock_sec=int(_AGENT_BUDGET_HOURS * 3600), turns=None),
        hard_budget=Budget(wallclock_sec=int((_AGENT_BUDGET_HOURS + _HARD_CAP_HEADROOM_HOURS) * 3600),
                           turns=None),
        # What an evaluation holds: 1 H200, 8 CPUs and 180 GiB for the server, less the
        # runner's reserve (run.runner_memory_gib, 4 GiB by default), and 16 GiB beside it for
        # the driver, for a 196 GiB job. Its verify role takes 3 h after its stage's staging:
        # about twice what a verify stage takes with the reference loaded on an H200, rounded up
        # to the half hour.
        evaluation_allocation=EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180, driver_memory_gib=16,
                                                   verify_time_sec=3 * 3600),
        # What a build holds beside its H200: 8 CPUs and 180 GiB.
        build_allocation=BuildAllocation(cpus=8, memory_gib=180),
        network=NetworkPolicy(
            domains=AccessList(mode="allow"),
            packages=AccessList(mode="deny", entries=_DENIED_SERVING_FRAMEWORKS)),
    ),
    brief=TabCompletionBrief(),
    serve_command=[],
)

#: The measured baseline, which is also the vLLM calibration member: stock vLLM 0.22.1 on the cuda
#: base, from locks/baseline.txt.
_VLLM = LayerImage(name="tab-completion-baseline", parent="cuda", pip_lock="baseline.txt", owner="tab_completion", server=True,
                   env=SERVER_ENV)
#: The image the HF reference engine runs on: torch 2.11 and transformers 5.13.1 on the cuda base,
#: from locks/reference.txt.
_REFERENCE = LayerImage(name="tab-completion-reference", parent="cuda", pip_lock="reference.txt", owner="tab_completion",
                        server=True, env=SERVER_ENV)
#: The SGLang calibration member: stock SGLang 0.5.16 on the cuda base, from locks/member-sglang.txt,
#: with the kernels its FlashInfer 0.6.14 would otherwise download when it first loads them, in the
#: directory its environment names.
_SGLANG = LayerImage(name="tab-completion-member-sglang", parent="cuda", pip_lock="member-sglang.txt",
                     owner="tab_completion", server=True, fetched=FLASHINFER_CUBINS["0.6.14"],
                     env={**SERVER_ENV, "FLASHINFER_CUBIN_DIR": "/opt/flashinfer-cubins"})
# The images the task runs on: the shared layer its sandbox, submission base and harness build on,
# and the images its own servers run on.
IMAGES = ImageSet(
    layer="s-a-tf-5-13-1",
    baseline=Served(image=_VLLM,
                    entrypoint=["sh", "-c", 'exec vllm serve "$HLS_WEIGHTS_DIR" "$@"', "vllm-baseline"],
                    # The directory on PYTHONPATH is the checkout the drive stage mounts for its
                    # driver only, never for the served artifact.
                    runtime_env={"PYTHONPATH": "/opt/hls-src/src:/opt/hls-src", "PYTHONUNBUFFERED": "1"}),
    reference=_REFERENCE,
    members={"vllm": _VLLM, "sglang": _SGLANG},
)

__all__ = ["TASK", "ARTIFACT", "CONTRACT", "STOP_TOKEN_IDS", "TENANT_COMMIT_ADD",
           "TENANT_COMMIT_EDIT", "TENANT_ALGO_BLOCK", "TENANT_ALGO_CONTROL"]
