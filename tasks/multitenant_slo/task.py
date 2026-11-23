"""multitenant_slo -- multi-tenant streaming SLO arena (four tenants, uncapped replies).

Four tenants, one per production archetype (chat / api_short / reasoning /
long_session), each with its own dataset-backed DataSource, a Gamma arrival
process (one shared TOTAL_RATE knob, fixed per-tenant mix + burstiness), and a
distinct StreamingLatencySLO. Replies run to the model's own end-of-sequence
under a single 16,384-token output guard (the production max_tokens analog;
see the output-length block below), so output lengths are this checkpoint's
measured behavior, not authored constants. Every tenant gates correctness at the token level
(TokenFaithfulnessVerifier): served token ids are teacher-forced through the
pinned bf16 reference, and the run fails if the disagreement leaves the
calibrated trusted-engine envelope. Because that gate is content-free, the
corpora need not be programmatically gradable, and the tenant mix
is fitted so the workload's request-side statistics land between the Bailian
to-B and to-C production profiles (see the derivation block below). There is
no code-completion tenant: a one-shot, distinct-file interaction shape has
about no prefix reuse, against 66% in the public Bailian coder trace, so it
would misrepresent that archetype.

Contract: http-sse + OpenAIChatProtocol() for StreamingLatencySLO timing;
token ids are always emitted on the wire, which is what the token-level gate
teacher-forces. The wire protocol requests them per-request since stock vLLM
emits choices[].token_ids only on request.

Every measured parameter of the verifier comes from ``calibration.json``, the
store's or the task's default in defaults/ -- both rules' ceilings, the
trace-length floor, and the tolerances they were sized against; `hls run`
measures one when neither matches its workload fingerprint. The one authored
tolerance, the rate headroom, is anchored to the calibration (see
the gate comment block below).

Run: hls run tasks.multitenant_slo.task --arm <arm>
"""
from __future__ import annotations

from hls.artifact.artifact import (
    AgentProducedArtifact, AgentProducerSpec, Budget, BuildAllocation, EvaluationAllocation, HighestAvailable,
)
from hls.core.decode import Greedy
from hls.egress.policy import AccessList, NetworkPolicy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.drive.bounds import VerifyPeak, bounds_for
from hls.prompt import ImplementerBrief, JinjaPromptTemplate
from hls.reference.references import HFReference
from hls.scoring import SloLatencyScorer
from hls.spec.baseline import Baseline
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import (
    BFCLDataset, HFDataset, MRCRDataset, WildChatDataset,
)
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.workload import (
    GammaInterTurn,
    Gamma,
    NoDelay,
    ShortOutputCap,
    StreamingLatencySLO,
    Workload,
)
from hls.verify import TokenFaithfulnessVerifier
from hls.recipes.catalog import FLASHINFER_CUBINS, SERVER_ENV, ImageSet, Served
from hls.recipes.model import LayerImage
from tasks.multitenant_slo.prompts import GSM8K_8SHOT
from tasks.multitenant_slo.removed_requests import add_removed_rows, removed_rows
from tasks.sample_task.task import (
    _DENIED_SERVING_FRAMEWORKS, _TOOLCHAIN_REFERENCES_SHA256, _TOOLCHAIN_UPLOAD_BRIEF,
)

SEED = 42

# Objective: speedup over the task's baseline serving stack in SLO-normalized latency --
# the geometric mean over requests of 1/r, with r = max(TTFT/ttft_ms, TPOT/tpot_ms)
# against each tenant's own budgets, divided by the same quantity for the baseline. The
# baseline's value comes from baseline.json, the store's or the task's default in defaults/
# (measured by `hls baseline-credit`), and is injected by the CLI. Not sample_task's goodput Scorer:
# this task's arrival is open-loop and fixed, so raw throughput is the driver's choice,
# not the server's, and a metric that plateaus once every SLO is met has a ceiling any
# competent server ties at. Any failed request fails the run. See
# hls.scoring.SloLatencyScorer for the rationale.
SCORER = SloLatencyScorer()

# Production served model: a real open frontier-class model, not the 0.5B debug
# model. Qwen3.6-27B is ungated and its bf16 weights are 55.6 GB on 1x H200 (141 GB),
# leaving ample room for batching, so the agent need not shard across GPUs. HARDWARE
# is 1x H200, which fits.
#
# The checkpoint is hybrid: of its 64 decoder layers only every 4th (16 total) is full
# attention with a growing KV cache; the other 48 are gated-delta linear-attention
# layers holding a fixed-size recurrent state per sequence. Both facts reach the agent
# through ## Model / ## Hardware Envelope and the serving envelope note below, and both
# change the memory arithmetic a server has to do.
#
# It also ships in a vision-language layout (config.architectures =
# ["Qwen3_5ForConditionalGeneration"], text body under model.language_model.*), which is
# why the reference loader resolves its model class from the checkpoint's own
# ``architectures`` instead of assuming a text-only auto-class -- see
# ``hls.reference.engines.HFEngine``, whose guards this task's oracle depends on.
MODEL = ModelSpec(base_checkpoint="Qwen/Qwen3.6-27B", revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
                  arch="TX", chat_format="chatml")

# The task's declared host. No brief states its CPUs, memory or storage: the build's brief
# states the sandbox's own figures and the graded server's are the
# evaluation allocation's, both declared on ARTIFACT below. The 1024 GB
# is an H200 host's memory; it is the figure only where a task declares no allocation.
HARDWARE = Hardware(accelerator=AcceleratorSpec(kind="H200", count=1, memory_gb_per_device=141),
                    cpu_count=64, memory_gb=1024, storage_gb=2048)

# The oracle Reference: HF, because TokenFaithfulnessVerifier requires the TeacherForce +
# StackReport capabilities, and HFEngine is the only engine that provides them (VLLMEngine
# is Generate-only).
#
# HFEngine's generate batches a whole tenant's events at once and runs out of memory on
# a 27B model's long long_session contexts, but the gate does not generate.
# teacher_force is an unpadded per-sequence loop -- one forward pass per event, never a
# padded batch -- so peak memory is one context, not a tenant.
#
# query_chunk: the width this task's calibration was measured at. The longest
# context here is well short of what forces a smaller chunk, and the ceilings in
# calibration.json only describe the instrument they were taken on, so the number
# is pinned until someone re-measures them.
_ORACLE = HFReference(decode=Greedy(), query_chunk=4096)

# http-sse: streaming timing for StreamingLatencySLO. Token ids feed any
# teacher-forcing verifier (e.g. TokenFaithfulnessVerifier): the wire protocol requests
# them on every request (return_token_ids), because stock vLLM emits choices[].token_ids
# only on request.
CONTRACT = RequestResponseContract(
    path="/v1/chat/completions",
    input_modalities=frozenset({"text"}),
    output_modalities=frozenset({"text"}),
    transport="http-sse",
    protocol=OpenAIChatProtocol(),
)

# The shared oracle (_ORACLE) teacher-forces every tenant's served tokens.

# ---------------------------------------------------------------------------
# Arrival model + tenant mix -- one tunable knob (TOTAL_RATE)
# ---------------------------------------------------------------------------
# Portions are fitted to production request-side statistics: the portion-weighted
# mixture of the tenants' uncapped request profiles on this checkpoint lands its
# input-length distribution and multi-turn share between the released Bailian to-C and
# to-B traces (alibaba-edu/qwen-bailian-usagetraces-anon; aggregate: input p50/mean/p90
# ~= 807/1,606/3,636 tokens against to-B's 574/915/1,720 and to-C's 1,046/2,331/6,436; multi-turn 16% against to-B's 0% and
# to-C's 46%).
#   chat 0.18          the to-C interactive class (multi-turn, live replies)
#   api_short 0.60     the to-B API class; the fit's largest weight -- it is the
#                       only tenant with naturally short median outputs, so it
#                       anchors both the input p50 and the output p50 near to-B
#   reasoning 0.12     emerging reasoning class (ServeGen S5)
#   long_session 0.10  deep multi-turn stress class (Fig. 5 File analog)
# Deployment archetype (stated, not derived): "a reasoning-model deployment serving
# mixed consumer-chat and business-API traffic" -- request-side statistics between
# to-B and to-C, output statistics of the thinking class, which is what serving a
# reasoning checkpoint on that traffic is.
#
# long_session is capped at 8 turns (MRCRDataset.max_turns=8), the Bailian p99; the cap
# over-samples a rare deep tail for stress, not a natural frequency.
#
# No item repeats: each tenant's per-run request budget is realized by a
# test-split Count no larger than its pool, so no item is ever served twice and a
# response cache stays useless by construction.
#
# The driver schedules sessions, so the session rate divides the request rate by
# turns/session; each tenant's Gamma inter-arrival gaps have a fixed shape
# (burstiness, CV = 1/sqrt(shape)) and a scale derived so mean gap = 1/rate:
#     request_rate_t = w_t * TOTAL_RATE
#     session_rate_t = request_rate_t / turns_t
#     scale_t        = 1 / (shape_t * session_rate_t)
# The portions (w_t), turns and CVs are fixed; only the aggregate rate is tunable.
#
# The rate below is anchored on this workload's own saturation, not on a load-search
# knee: this model opens a reasoning block and runs most replies long, so its decode
# load per request is its own.
#
# The arrival span W stays at ~1,500 s whatever the rate: the Bailian-grounded
# think-times need it to cover a long_session conversation's internal span (up to
# D_max ~= 970 s of summed think-time delays) within the 0.7*W trim dial, so
# W >= ~1,400 s; the request count then scales as rate x W, and the graded trace with
# it (the counts block below). Any trace at or above the calibration's certified serving
# length is admissible: the rate threshold is per-token, the run-level spread only
# shrinks with length, and the curve ceilings recompute at the actual token count.
#
# The rate is anchored at 100% of the faster of the two reasonable baseline stacks on
# one H200: stock vLLM 0.24.0 saturates at about 1,478 generated tokens/s on a full draw
# of this workload (uncapped under the 16,384 output guard). At TOTAL_RATE = 1.11 req/s
# the workload offers its ~2,221,000 expected generated tokens over the 1,500 s nominal
# arrival span, i.e. ~1,459 tokens/s, 98.7% of that saturation.
# Stock SGLang saturates near 1,120 tokens/s and queues through the drive, which harms
# only its own SLO scores; the token gate is calibrated under this same load
# (calibration drives the workload through the task's own clocked driver at
# TOTAL_RATE). The intent of the 100% anchor: goodput headroom exists only by matching
# or beating the better stock baseline, never by slack in the offered load. The anchor
# is hardware-relative: re-anchor if the graded hardware class changes. The count tables
# below are re-derived whenever the rate moves.
TOTAL_RATE = 1.11  # aggregate offered request load (req/s) across all tenants -- the knob

# Derived at TOTAL_RATE=1.11 by the formulas above (recompute this table if the rate
# moves; it is also asserted by readback off the constructed tenants in
# tests/unit/test_multitenant_task_wiring.py, so it can never silently drift):
#     tenant         portion  req/s    turns    session/s  CV   shape     scale
#     chat            0.18    0.1998    2.3914   0.083549  1.0  1.000000  11.968969
#     api_short       0.60    0.6660    1        0.666000  1.5  0.444444   3.378378
#     reasoning       0.12    0.1332    1        0.133200  1.0  1.000000   7.507508
#     long_session    0.10    0.1110    7.6266   0.014554  1.0  1.000000  68.708108
#
# Counts: the per-tenant test-split sizes that realize the portions at the request level.
# Multi-turn tenants are counted in sessions, so their request budget is
# sessions x mean_turns; the two mean-turn constants are measured on the pinned
# revisions (see _CHAT_TURNS / _MRCR_TURNS below).
#
# The target is set by the token gate, not by coverage: the calibration campaign
# drives this very workload, so a graded trace realizes the length the ceilings were
# measured at, and the
# counts are sized to land the run's generated-token total comfortably past the
# verifier's fail-closed floor. The yields table under the correctness gate below is
# where that total is derived; the wiring test reads it back off the constructed
# tenants. Three derivation layers:
#   (1) the dev/test partition (a hash of each row's own id) fixes which sessions exist
#       and their full turn counts;
#   (2) the removal list (DataSource.removed_rows; the rule at OUTPUT_GUARD) drops the
#       listed sessions before the schedule is drawn;
#   (3) ramp-down trimming (Workload.trim_rampdown, deterministic at TaskSpec.seed and
#       decided at draw time against the nominal zero-service timeline --
#       drivers.rampdown_turn_budget) then excludes the follow-up turns of the two
#       think-time tenants whose nominal dispatch falls past the arrival span, so the
#       multi-turn tenants realize fewer requests than sessions x mean turns.
#     tenant         test split   removed   kept requests (readback at seed 42)
#     chat             126 convs     18         210   (  263 untrimmed)
#     api_short        999 items    160         839
#     reasoning        200 items      1         199
#     long_session      22 convs      4          78   (  133 untrimmed)
#                                             -----
#                                             1,326  -> nominal span 1,500 s
# "Kept requests" counts the workload exactly: membership, turn counts, think-time
# draws and the trim cut are all pure functions of the rows and the seed, so the table
# is read back off the materialized workload (the wiring test re-derives it; see
# tests/unit/test_multitenant_task_wiring.py). For a tenant with a history_char_budget
# (chat), the drawn count is an upper bound on what gets served: a session ends at the
# first turn whose conversation would exceed the budget, and that cutoff depends on the
# lengths of the replies actually generated, not on the seed alone. Every tenant's
# offered schedule spans ~W at this rate, so all four deplete together at any rate,
# by construction of the counts (spans use untrimmed session budgets; trimming only
# removes turns that would dispatch past W).
#
# The counts are small against the pools because this model emits ~1,322 generated
# tokens per kept request on this uncapped mix; a tenant's split is a hash-selected fraction of its pool,
# and no item is served twice.
#
# The calibration is in calibration.json, the default in defaults/, which carries this task's
# workload fingerprint and validator-locks every number it records to the counts stored
# beside it.
#
# The pools include their construction tokenizer, which the adapters default to
# Qwen/Qwen2.5-32B and which stays there rather than following the served model: it is a
# corpus knob (the length filters and prefix windows that build the rows), the
# calibration draws were built through it, and re-pointing it would rebuild the pools and
# break exactly the transfer above. What the server sees is measured directly instead --
# the graded test split's worst scripted prompt under this model's own tokenizer and chat
# template is long_session at ~8.2k tokens, and chat's live prompts are bounded by its
# history window; every prompt plus the 16,384 output guard fits the 32,768 serving
# envelope by construction.
#
# Dev: 25% of each tenant's request budget (not of its pool), so the agent's dev
# rehearsal is the same mix at the same rates and its pools deplete together too
# (~355 kept requests, last scheduled arrival ~590 s at seed 42). reasoning keeps its
# independent dev corpus
# (openai/gsm8k train, disjoint from the Platinum test set by corpus identity) and is
# sampled from it to the same 25%, rather than draining all 7,473 items -- an
# unsampled dev corpus would leave that one tenant driving for hours after the other
# four ran dry.

# Per-tenant portion of the offered request rate (requests, not sessions), mean
# turns/session, and arrival CV.
_PORTIONS = {"chat": 0.18, "api_short": 0.60, "reasoning": 0.12, "long_session": 0.10}
# Mean turns/session over each multi-turn tenant's pinned pool (single-turn tenants are
# 1 by construction):
#   chat          2.3914 -- mean over the 1,188-conversation pool (WildChat En, non-toxic,
#                            <=6 turns, user text <=3,072 tok, de-duplicated on the
#                            opening user message); about half single-turn, with a real
#                            2-6 spread.
#   long_session  7.6266 -- mean(min(turns, 8)) over the pinned pool's 316
#                            conversations. The 8-turn cap is the Bailian p99; see
#                            MRCRDataset.max_turns=8 below.
_CHAT_TURNS = 2.3914
_MRCR_TURNS = 7.6266   # mean(min(turns, 8)) over the pinned pool's 316 conversations
_TURNS = {"chat": _CHAT_TURNS, "api_short": 1.0,
          "reasoning": 1.0, "long_session": _MRCR_TURNS}
# Per-tenant test-split sizes (sessions for multi-turn tenants) -- see the COUNTS table
# in the derivation block above and the request-count readback test in
# tests/unit/test_multitenant_task_wiring.py.
_TEST_COUNTS = {"chat": 126, "api_short": 999, "reasoning": 200, "long_session": 22}
# DEV is 25% of each tenant's request budget to within session granularity (the dev
# comment in the counts block above).
_DEV_COUNTS = {"chat": 32, "api_short": 250, "reasoning": 50, "long_session": 6}
# Burstiness, ServeGen-anchored per class (clients differ in burstiness): human-initiated
# chat and reasoning arrive ~Poisson (CV 1.0); the tool/API client is the bursty
# top-client profile (CV 1.5).
_CVS = {"chat": 1.0, "api_short": 1.5, "reasoning": 1.0, "long_session": 1.0}
# Gamma shape from the CV, never authored twice: CV = 1/sqrt(shape).
_SHAPES = {name: 1.0 / cv ** 2 for name, cv in _CVS.items()}

# Correctness gate: token-level and pooled. Each tenant below carries a
# TokenFaithfulnessVerifier over the shared oracle, but the verifier is declared pooled,
# so the runner grades all four tenants' served tokens once as a single population and
# every tenant shares that one verdict (it does not run four gates). Pooling is not a
# convenience: the ceilings it compares against are themselves pooled worst-member
# statistics over the calibration's whole traffic mix, and intrinsic flip rates differ by
# stratum rather than by server quality -- on this checkpoint they differ about 14x
# between tenants (api_short 0.034% .. long_session 0.477%).
# Applied per stratum, one pooled ceiling would fail the hottest tenant whatever the
# server does while leaving the coldest an order of magnitude of headroom to be broken in.
#
# The gate teacher-forces the served token ids through the pinned bf16 reference and
# applies two deterministic checks plus a trace-length floor:
#     reject iff  flip_rate > 2.0 * q_max                              (the rate rule)
#              or  #{flips with gap >= c} > 2.0 * k(c, N) at any level (the curve rule)
#              or  the pooled trace carries fewer teacher-forced tokens than
#                  the calibrated floor -- an epistemic guard (below it the rate rule
#                  cannot fire at its nominal false-alarm rate), not an anti-truncation
#                  one: a server that ends replies early produces a large-gap flip on
#                  every truncated reply, which the curve rule catches at a handful of
#                  events.
# None of the measured parameters are authored here. q_max, the gap-level grid, each
# engine's per-level clean event rates, the floor and the tolerances the campaign was
# sized against all come from calibration.json, injected by the CLI loader and gated on
# the workload fingerprint -- `hls run` measures one when no file it finds matches.
# Exactly one constant is authored here:
#
#   rate_multiplier=2.0  The declared headroom over the measured rate ceiling: the
#                        deployed rate gate is this multiple of the campaign's q_max,
#                        so both numbers come off calibration.json. It is a policy
#                        choice, not a measurement, which is why it lives on the
#                        verifier rather than in the artifact -- and why the same
#                        calibration can grade several verifiers at different headrooms.
#                        The curve rule's headroom is the verifier's own default of the
#                        same 2.0.
#
# The curve rule bounds how many large-gap flips a trace carries, which is what separates
# clean from broken: no single-gap cap does, because the clean engines themselves produce
# rare multi-nat flips, single events past 2.5 nats. At each level c the clean count is
# modelled as a Poisson number of events at the exact Garwood 97.5% upper-bound rate lambda_up_c, and
#     k(c, N) = min{ k : P(Poisson(lambda_up_c * N) > k) <= delta_family / n_rungs }
# recomputed for the actual pooled trace length N, graded against the more permissive
# engine at each level. See hls.verify.calibration.CurveSide for the model.
#
# Every served reply is teacher-forced and counted into both rules, whatever the
# artifact reports about how it ended; finish_reason is a value the contestant supplies
# and nothing reads it. Greedy decode on this checkpoint has a reasoning-spiral mode, and
# how many spirals a serving happens to contain moves its flip rate without saying
# anything about the server, so the rows the calibration's own serving stack runs to
# OUTPUT_GUARD are removed from the workload (the rule at OUTPUT_GUARD below) and are
# served and graded on neither side. A residual spiral under a contestant's engine is
# graded as text with no stop row: the reference scores a reply's stop only when it
# neither ended on an end-of-turn id nor reached its cap.
#
# Expected served tokens per tenant per run -- the one place this total is stated.
# It is this task's kept request counts (the readback in the counts block above) times
# this checkpoint's per-request generated-token yields on each tenant's corpus (natural
# end-of-sequence under the output guard; a reply's length is the model's own stopping
# behavior; guard-hitting replies are included, so these bound the yields of the
# workload with those rows removed from above):
#     chat            210 x 3,352.2  ~   704,000   live replies
#     api_short       839 x   586.8  ~   492,000
#     reasoning       199 x 1,830.2  ~   364,000
#     long_session     78 x 2,361.4  ~   184,000
# Pooled that is ~1,745,000 generated tokens. Every served token is graded, and N is
# larger by one position per reply whose stop the reference scores (whether it would
# have ended there too). Raising a tenant's count raises N (the curve ceilings recompute
# at the larger N, and the per-run event budget grows with it); lowering one can push the drawn
# trace under the calibrated floor, which fails every run as unverifiable rather than
# only weakening the claim. Either direction changes the workload fingerprint and so
# requires re-running `hls calibrate`. Note what sets that length: it is the SLO
# measurement's, not the gate's. The 1,500 s arrival span is forced by long_session's
# think-time geometry and the rate by stock vLLM's saturation. The fail-closed floor is
# not a property of the task and is not quoted here: it is
# rate_rule_min_tokens(q_max, delta_per_rung, rate_multiplier), derived on the verifier
# from the campaign's ceiling and the headroom deployed below, and a tighter headroom
# needs a longer trace. It lands orders of magnitude under a trace this size, so the
# gate rides essentially free on a trace sized by the performance measurement.

# One shared instance across all four tenants, like _ORACLE above: the gate is pooled, so
# the runner grades it once for the whole run whatever the tenants declare, and stating
# the constant once keeps it from drifting per tenant. check_emit_consistency stays at
# its default (off): this contract's OpenAIChatProtocol() does not request per-token
# logprobs (and flipping that on would change the workload fingerprint, staleing the
# calibration), so check_wiring would refuse the flag here.
_TOKEN_GATE = TokenFaithfulnessVerifier(reference=_ORACLE, rate_multiplier=2.0)


def _arrival(name: str) -> Gamma:
    """Gamma session inter-arrivals for tenant ``name``: mean session rate =
    w_t*TOTAL_RATE/turns, scale derived so the mean gap = 1/rate."""
    session_rate = _PORTIONS[name] * TOTAL_RATE / _TURNS[name]
    shape = _SHAPES[name]
    return Gamma(shape=shape, scale=1.0 / (shape * session_rate))


# Pinned dataset revisions + their licenses (resolved from the HF API at wiring time).
# allenai/WildChat-1M: ODC-BY, ungated.
_WILDCHAT_REV = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
_BFCL_REV = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
_MRCR_REV = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
_GSM8K_PLATINUM_REV = "e762492455a1cf7967de89f05b6bef72fc713b66"
# openai/gsm8k train = the gsm8k dev corpus.
_GSM8K_TRAIN_REV = "740312add88f781978c0658806c59bc2815b9866"
# The corpus-time tokenizer of chat's and long_session's length filters (the adapters'
# Qwen/Qwen2.5-32B), at the commit the rows were built with: its main since 2024-09-20,
# and the commit that last touched the repo's tokenizer files.
_CORPUS_TOKENIZER_REV = "1818d35814b8319459f4bd55ed1ac8709630f003"

#: The single output guard every request carries as ``max_tokens`` -- the
#: production output-limit analog, not a class cap. Replies run to the model's own
#: end-of-sequence; the guard exists because greedy decode on this checkpoint can fall
#: into a reasoning spiral that never stops (the reply re-examines the same decision
#: indefinitely). 16,384 is well above any genuine completion of every tenant, so the
#: guard binds that pathological tail and nothing else, and how much of each side it
#: touches is read off a campaign's captures (``hls capped-requests``) rather than
#: restated here.
#:
#: Requests on which the calibration's own serving stack runs to this guard are
#: removed from the workload. Each tenant's ``DataSource.removed_rows`` lists them by
#: row id, derived from the campaign's captures by ``hls capped-requests`` (served
#: length equal to the stamped cap, the union over servings) and kept with its
#: derivation in ``removed_requests.json``; the drive drops the listed sessions before
#: the arrival schedule is drawn. The list is part of the workload fingerprint, so a
#: new list is a new calibration.
OUTPUT_GUARD = 16_384

#: What ``hls capped-requests --write`` calls to add newly observed rows to the lists.
#: The command derives which rows reached the guard; the file they are written to, and
#: the shape they are written in, belong here.
record_removed_rows = add_removed_rows

# Shared-prefix layer: mostly inert at this layer -- this checkpoint's chat template
# inserts no default system prompt (unlike Qwen2.5's 16-token one), so chat and
# long_session carry no authored prefix at all, and api_short carries its own official
# system message inside its adapter (BFCL tool-schema template). The only prefix authored here is the reasoning
# tenant's 8-shot CoT. It rides in the prompt template (identical for the
# platinum test corpus and the openai/gsm8k dev corpus, so oracle + artifact + dev/test
# all see the same prefix). Its prefix budget is the 715 tokens the exemplars measure
# under this model's tokenizer.
_GSM8K_PROMPT = JinjaPromptTemplate(template=GSM8K_8SHOT + "Question: {{ question }}\nAnswer:")
# Handed to the contestant beside the dev rows. Those rows are the bare question and
# answer, so nothing in them shows that a served prompt carries this prefix first --
# a contestant measuring prefill on a dev row would measure a tenth of the real
# prompt. The other tenants need no counterpart: their prompts are the corpus text.
_GSM8K_DEV_REFERENCE = {"prompt_prefix.txt": GSM8K_8SHOT}
_GSM8K_TARGET = JinjaPromptTemplate(template="{{ answer }}")

# --- chat -- interactive multi-turn chat (real user conversations, live replies) -------
# WildChat conversations, English, non-toxic, capped at 6 turns. The artifact generates
# every assistant turn (MultiIF-style session: the driver stitches the artifact's own
# reply into the history before the next user turn), so this tenant is the workload's live
# multi-turn decode load. The 1,188-conversation subsample is the task's draw pool
# (kept at the calibration's size; a pool is part of the distribution
# definition, not sized to a split); dev 32 + test 126 conversations (_DEV_COUNTS /
# _TEST_COUNTS) are the hash partition of it. The adapter's 3,072-token cap on a conversation's user text bounds
# the scripted side of a prompt; the live side (the artifact's own replies riding in
# later turns' history, about 3.6k tokens each on average on this checkpoint) is bounded
# by the history window below, so a deep session's prompt cannot outgrow the serving
# envelope.
TENANT_CHAT = Tenant(
    id="chat",
    data_source=DataSource(
        dataset=WildChatDataset(revision=_WILDCHAT_REV, language="English", max_turns=6,
                                subsample_size=1188, max_input_tokens=3072,
                                tokenizer_revision=_CORPUS_TOKENIZER_REV),
        split=SplitSpec(dev=Count(value=_DEV_COUNTS["chat"]),
                        test=Count(value=_TEST_COUNTS["chat"])),
        removed_rows=removed_rows("chat"),
    ),
    workload=Workload(
        arrival=_arrival("chat"),
        mean_turns_per_session=_CHAT_TURNS,
        # Bailian to-C reuse-time: median ~111 s, CV 1.57 (shape 1/1.57^2 = 0.4057).
        # Light truncation at 1800 s -- chat's ~2.4-turn walk-time is a non-issue for
        # the ramp-down. Shape/scale were pinned offline by sampling this Gamma under
        # the truncation until the realized moments matched the target:
        #   realized median=110.9 s  mean=286.7 s  cv=1.422
        inter_turn_delay=GammaInterTurn(shape=0.4057, scale=735.8009, max_gap=1800.0),
        output_length_cap=ShortOutputCap(cap_tokens=OUTPUT_GUARD),
        # Context capacity of a chat session: prompts must stay under the serving
        # envelope minus the output guard (32,768 - 16,384 = 16,384 tokens), and chat
        # is the one tenant whose prompts grow with live replies. 52,000 characters is
        # that token budget at a conservative 3.2 chars/token floor for this
        # checkpoint's reasoning-style prose. A session that would exceed it ends at
        # that turn -- what a real user does when a conversation fills the context
        # window -- so a chat session serves its drawn turns only up to the point its
        # own accumulated replies fill the envelope.
        history_char_budget=52_000,
        slo=StreamingLatencySLO(ttft_ms=500.0, tpot_ms=50.0),
        trim_rampdown=True,
    ),
    verifiers=[_TOKEN_GATE],
)

# --- api_short -- tool-calling API client (single-turn BFCL AST categories) ------------
# Burstiness CV 1.5 (ServeGen's bursty top-client profile). Pool 2,350 items;
# dev 250 + test 999 are the hash partition of it.
TENANT_API_SHORT = Tenant(
    id="api_short",
    data_source=DataSource(
        dataset=BFCLDataset(revision=_BFCL_REV),
        split=SplitSpec(dev=Count(value=_DEV_COUNTS["api_short"]),
                        test=Count(value=_TEST_COUNTS["api_short"])),
        removed_rows=removed_rows("api_short"),
    ),
    workload=Workload(
        arrival=_arrival("api_short"),
        inter_turn_delay=NoDelay(),
        output_length_cap=ShortOutputCap(cap_tokens=OUTPUT_GUARD),
        slo=StreamingLatencySLO(ttft_ms=300.0, tpot_ms=50.0),
    ),
    verifiers=[_TOKEN_GATE],
)

# --- reasoning -- short-request reasoning API (GSM8K-Platinum test / openai train dev) -
# TEST draws from the 1,209 label-noise-free Platinum items; DEV from the openai/gsm8k
# train split (same question/#### answer schema, un-denoised labels fine off the gate).
# Disjoint by corpus identity, so dev_dataset samples the two independently: Platinum is
# exactly a 1,209-item subset of gsm8k's 1,319-item test split, which shares no question
# with train. The two corpora are exchangeable for load purposes --
# question length 1.02x, reference-rationale length 1.01x, rationale steps and calculator
# operations 1.01x, test over dev -- so a contestant characterising reply length on dev is
# measuring the graded population. Uncapped, this checkpoint's reasoning block runs these
# to a natural stop at about 1,780 tokens median and 3,470 p99; the guard rarely binds
# here, and the rows it binds are removed like any other.
TENANT_REASONING = Tenant(
    id="reasoning",
    data_source=DataSource(
        dataset=HFDataset(path="madrylab/gsm8k-platinum", subset="main",
                          source_split="test", prompt=_GSM8K_PROMPT, target=_GSM8K_TARGET,
                          revision=_GSM8K_PLATINUM_REV),
        dev_dataset=HFDataset(path="openai/gsm8k", subset="main", source_split="train",
                              prompt=_GSM8K_PROMPT, target=_GSM8K_TARGET,
                              revision=_GSM8K_TRAIN_REV),
        dev_reference=_GSM8K_DEV_REFERENCE,
        split=SplitSpec(dev=Count(value=_DEV_COUNTS["reasoning"]),
                        test=Count(value=_TEST_COUNTS["reasoning"])),
        removed_rows=removed_rows("reasoning"),
    ),
    workload=Workload(
        arrival=_arrival("reasoning"),
        inter_turn_delay=NoDelay(),
        output_length_cap=ShortOutputCap(cap_tokens=OUTPUT_GUARD),
        slo=StreamingLatencySLO(ttft_ms=300.0, tpot_ms=50.0),
    ),
    verifiers=[_TOKEN_GATE],
)

# --- long_session -- long-session chat, cumulative scripted replay ---------------------
# Each MRCR conversation is one session capped at 8 turns (mean 7.6266 over the 316-item
# pool under the 8-turn cap): turn k's request carries the dataset's own message list
# through user turn k as pre-rendered metadata['messages'], and the artifact generates
# assistant turn k. The history is scripted, never the artifact's live replies, so every
# turn sees exactly the corpus's context and MRCR's needle structure survives -- while
# each turn is a distinct request whose prefill grows from 100s to ~8K tokens (real
# history resend). The token gate scores every generated turn. Pool 316 conversations;
# dev 6 + test 22 are the hash partition of it. The 8-turn cap (MRCRDataset.max_turns=8)
# is the Bailian p99; it over-samples a rare deep tail for stress. This tenant's counts
# are the workload's coarsest knob -- one conversation is ~5-8 kept requests -- and 22 is the value that holds its share and the
# depletion tolerance together at this workload's span (see the counts block above).
TENANT_LONG_SESSION = Tenant(
    id="long_session",
    data_source=DataSource(
        dataset=MRCRDataset(path="openai/mrcr", revision=_MRCR_REV, replay=True, max_turns=8,
                            tokenizer_revision=_CORPUS_TOKENIZER_REV),
        split=SplitSpec(dev=Count(value=_DEV_COUNTS["long_session"]),
                        test=Count(value=_TEST_COUNTS["long_session"])),
        removed_rows=removed_rows("long_session"),
    ),
    workload=Workload(
        arrival=_arrival("long_session"),
        mean_turns_per_session=_MRCR_TURNS,
        # Bailian coder reuse-time (CV 1.86, shape 1/1.86^2 = 0.2891); truncated at
        # max_gap=300 s (pinned offline the same way as chat's, by sampling under the
        # truncation until the realized moments matched). The cap is
        # the CV<->depth lever: the untruncated mean (487 s) >> median (112 s) at
        # shape=0.2891, so a tighter cap pulls realized CV down toward 0. With
        # trim_rampdown=True the drawn workload excludes follow-up turns whose
        # nominal dispatch falls past the arrival span (draw-time membership,
        # drivers.rampdown_turn_budget), so the tail budget is
        # D_max <= 0.7*W: the 7.6-turn walk D_max = (M-1)*mean_truncated
        # = 6.6266 * 146.4 ~= 970.4 s stays within one arrival span (W ~= 1,518 s, 64%),
        # so an early-arriving session still carries its full walk while later ones are
        # trimmed in the draw itself. Realized median=112.3 s  mean=146.4 s  cv=0.889
        # D_max=970.4 s.
        inter_turn_delay=GammaInterTurn(shape=0.2891, scale=1684.0497, max_gap=300.0),
        output_length_cap=ShortOutputCap(cap_tokens=OUTPUT_GUARD),
        # 500 ms would be uncreditable for the late turns at speed of light (their
        # prompts reach 6-8.2K tok); 1000 ms ~= 2x the FP8 prefill floor.
        slo=StreamingLatencySLO(ttft_ms=1000.0, tpot_ms=50.0),
        trim_rampdown=True,
    ),
    verifiers=[_TOKEN_GATE],
)

# Tenant order = the mix table above (chat, api_short, reasoning, long_session).
_TENANTS = [TENANT_CHAT, TENANT_API_SHORT, TENANT_REASONING,
            TENANT_LONG_SESSION]

# ---------------------------------------------------------------------------
# TaskSpec (multi-tenant form)
# ---------------------------------------------------------------------------
TASK = TaskSpec(
    id="multitenant_slo",
    model=MODEL,
    hardware=HARDWARE,
    contract=CONTRACT,
    tenants=_TENANTS,
    scorer=SCORER,
    seed=SEED,
    metadata={"theme": "multitenant-slo"},
    # What one drive may make the harness hold (hls.drive.bounds). A drive's replies come to
    # about 462 MB on the wire, which held unwritten at once at the stream's 13.5 trace bytes per
    # byte make the cap 6 GiB.
    #
    # The requests: a drive's trace holds about 11.6 MB of them. Once the cap is reached,
    # each chat session still sends its remaining turns, each carrying its conversation, which the
    # chat tenant's budget holds to 52,000 characters: at most 210 such turns in a drive, at up to
    # 12 bytes of trace JSON a character (an astral one escaped as two \u units), is 131 MB more.
    # 142.6 MB in all, so 160 MiB.
    #
    # The verify role holds 52.55 GiB with its vLLM reference
    # loaded, and 12.38 bytes besides per byte of trace: the token gate keeps every
    # teacher-forced reply whole, and a reply carrying a logprob per token it was not asked for is its densest. At the 6 GiB cap and its
    # requests that is 129 GiB, so the 6 GiB cap stands.
    drive_bounds=bounds_for(CONTRACT, trace_bytes=64_342_088, held_reply_bytes=462_268_100,
                            trace_request_bytes=160 * 2**20,
                            verify_peak=VerifyPeak(base_bytes=round(52.55 * 2**30), bytes_per_byte=12.38)),
    # The task's reference serving system, vLLM's own OpenAI-compatible server, unmodified,
    # on which its baseline credit is measured: it handles high concurrency and streams
    # token-by-token SSE (so StreamingLatencySLO gets real TTFT/TPOT), and serves
    # /v1/chat/completions and /health directly. The `.image` string is the identity the
    # baseline credit records; the image served is the measured baseline the image catalog
    # names, which starts `vllm serve` on the mounted weights with these flags appended.
    baseline=Baseline(
        image="hls-baseline-multitenant-slo",
        # Tuned single-GPU vLLM config (the task's per-contestant envelope is 1 GPU), passed
        # to `vllm serve` as flags. The client sends model="served-model" (OpenAIChatProtocol
        # default), so --served-model-name must match. Throughput levers: KV-cache utilization,
        # the concurrent-batch ceiling, prefix caching (long_session re-sends its scripted
        # history every turn), and chunked prefill.
        # No --max-model-len, so vLLM sizes the envelope from the checkpoint itself. The
        # reservation arithmetic is what matters: vLLM rejects a request
        # when prompt + max_tokens > max_model_len, and every request carries the 16,384
        # output guard as max_tokens, so the envelope must exceed the longest prompt by that
        # guard. A hand-set number cannot hold that invariant as the corpus changes, and a
        # request it rejects is a 400 and a zero under the no-failed-request gate. Reserve
        # for prompt plus the requested max_tokens, not the realized reply.
        # The vLLM oracle above drops the flag for the same reason; the HF oracle needs no
        # reservation, its teacher-forced forward pass sizes itself to the sequence.
        serve_command=[
            "--served-model-name", "served-model",
            "--gpu-memory-utilization", "0.92",
            "--max-num-seqs", "512",
            # No --max-model-len: vLLM defaults to the checkpoint's max_position_embeddings.
            # See the oracle above -- a hand-set envelope silently rejects requests the
            # workload does send, which zeroes the baseline under the no-failed-request gate.
            "--dtype", "bfloat16",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            # Greedy, explicitly. This checkpoint's generation_config.json ships
            # do_sample=true / temperature=1.0 / top_k=20 / top_p=0.95, and `vllm serve`
            # applies those defaults for any field a request omits -- which is every
            # sampling field, since the harness sends only messages and max_tokens. Left
            # alone the baseline samples, and a sampled trace fails the gate. The
            # brief requires of contestants exactly what this fixes, so a baseline
            # without it fails the task's own decoding rule.
            #
            # The values are hls.reference.engines._GREEDY_SAMPLING, the pin the calibrated
            # vLLM and SGLang pool members decode under, so the baseline is greedy in the
            # same way as the ceiling it is measured against.
            "--override-generation-config", '{"temperature":0.0,"top_p":1.0,"top_k":-1,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0}',
        ]),
)

# Agent wall-clock budget conveyed in the brief (the native /goal ralph loop keeps the agent
# working until it is exhausted -- armed by the agent build's kickoff, keyed off this soft budget). The
# orchestrator's hard wall-clock cap sits _HARD_CAP_HEADROOM_HOURS above it (see ARTIFACT below);
# on expiry the committed `main` is gracefully collected, so intermediate commits are never lost.
#   0.5 = 30-min illustrative end-to-end run   1.0 = 1-hour shakedown
#   6.0 = long overnight run                   12  = graded
_AGENT_BUDGET_HOURS = 12.0

# Grace the orchestrator waits above the soft budget before it force-collects committed main.
# 0.5 h (30 min) gives eval + a clean exit headroom on a long graded session
# while staying tight enough that a hung agent doesn't idle for hours on a short run.
_HARD_CAP_HEADROOM_HOURS = 0.5


# The wall-clock total reaches the agent through
# `## Your budget`, rendered by hls.prompt.budget_brief_section from the soft_budget set on
# ARTIFACT below (which also gates the `get_budget` MCP tool). One source, stated once.

# The base brief's `goal` block describes goodput ("the rate of requests served correctly and
# within their latency SLO"), which is not this task's objective: it is graded on speedup
# over the baseline stack in SLO-normalized latency (hls.scoring.SloLatencyScorer -- see
# SCORER above), which has no plateau and no
# ceiling, and whose failed-request rule is a hard run gate rather than a per-request penalty.
# Both facts must be unmissable in the brief the agent actually reads, so override the block
# (the authored task-override seam -- recorded in provenance like every other section).
#
# The correctness paragraph states the token gate in numbers, rendered from the calibration
# the CLI injects before the build phase (`verification`, hls.prompt.verification_fields) --
# never hardcoded here, where they would drift from calibration.json the next time the noise
# floor is re-measured. `--preview`/`--export` inject best-effort (a valid calibration on disk
# previews the same numbers the run renders; without one `verification` is None and the
# paragraph falls back to naming the gate qualitatively -- authoring inspection still works
# pre-calibration).
#
# Three corrective facts sit in this block rather than in a trailing directive, each beside
# the text it qualifies, because each is a fact a contestant can reasonably get wrong:
#   - Per-turn scoring, beside the scoring formula. A replay harness that collapses a
#     multi-turn session into one request carrying the whole history pays a single TTFT on
#     one long prefill where the graded drive pays one per turn, which inflates a
#     self-measured score.
#   - The 32,768-token ceiling, in the `contract` block beside the max_tokens rule. It is
#     not derivable: the workload table lists output caps but no input lengths, and /data/dev
#     is a different sample than the scored split, so an implementation can reasonably size
#     itself to the model's nominal 8192 and reject the long prompts (prompt 8185 +
#     max_tokens 1024 > 8192 -> HTTP 400 on a long long_session request, which under the
#     no-failed-request gate zeroes the run).
#     State the ceiling; do not prescribe how to reach it (KV cache, position handling and
#     per-request limits are the agent's problem).
#   - The message shape, in the `contract` block beside the request schema. Nothing the
#     contestant is given shows it: the base schema example is a lone user turn, and the
#     dev rows are raw dataset fields, so the system message this task's BFCL adapter
#     builds from `functions` appears nowhere on disk. Templating `messages[0]` alone, for
#     example to key a prefix cache, raises on a system-led request. The brief states no
#     per-tenant turn averages: `chat` and `long_session` dev rows carry their own turn
#     structure, and "1.0 turns" reads as "one message" for the one tenant that always
#     gets two.
_GOAL_BLOCK = """{% extends "brief" %}
{% block goal %}Build a hyperoptimized serving stack for `{{ model.base_checkpoint }}` under the
workload below. Optimize until you reach the physical limits of the hardware or exhaust
your budget. The grader drives all tenants concurrently against one instance of your
server.

**Your score is your speedup over a production serving framework in SLO-normalized
latency.** Each tenant declares its own TTFT and TPOT budgets (listed per tenant under
## Workload). For every request:

    r = max( TTFT / that tenant's TTFT budget , mean TPOT / that tenant's TPOT budget )

    score = exp( mean over requests of log(1/r) )
            / the same quantity for the production framework on this workload

Higher is better, with no ceiling. Sessions arrive open-loop on a schedule you do not
control.

Each turn of a multi-turn session is a separate request,
sent and scored on its own -- one TTFT per turn, not one per session. Each such request
carries the conversation so far: a later turn's `messages` array holds the earlier user
and assistant turns, in order, before the current user turn.

**Your implementation must not fail any request.** Every request must receive a complete
streaming response. Correctness is checked at the token level: the exact token ids you
serve are teacher-forced through a bf16 reference implementation of this model --
HuggingFace transformers, batch size 1, MATH SDPA backend, cuDNN attention disabled,
the model class resolved from the checkpoint's declared architectures, and the
linear-attention layers on the torch fallback path, scored in 4,096-token chunks against
a KV cache{% if verification %}. The run fails if more than
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
pass/fail with no credit for margin
-- a run that passes at 90% of a ceiling scores identically to one at 10% -- so latency
spent buying faithfulness headroom below the ceiling is pure loss.

Serve greedy: the argmax of the raw logits, with none of the checkpoint's
generation_config sampling defaults applied (a stack that silently inherits those
defaults fails outright). Report end-of-sequence as finish_reason, never as an emitted
token.

That check is one pooled measurement over every served token of the whole run -- not one
per tenant. To reproduce it on the dev split,
draw your sample in the per-tenant request proportions the workload states above, and let each
tenant generate under its own output cap: the pooled rate is token-weighted and the
tenants' individual rates differ severalfold, so a sample drawn in different proportions
-- or a small one -- can easily read 50% higher or lower than the number you are graded
on.{% endblock %}
{% block contract %}{{ super() }}

Sizing: prompts run long, and a request must be served at whatever length it arrives --
refusing or truncating one is a failed request. Size for the checkpoint's own context
window, which its config declares; do not assume a smaller limit.

`messages` carries `system`, `user` and `assistant` roles. A request may open with a
system message, and a later turn of a session repeats the turns before it, so a request
body can look like:
```json
{
  "model": "served-model",
  "messages": [
    {"role": "system", "content": "<instructions for this request>"},
    {"role": "user", "content": "<first turn>"},
    {"role": "assistant", "content": "<what you returned for it>"},
    {"role": "user", "content": "<the turn to answer now>"}
  ],
  "return_token_ids": true
}
```
Template the array you are given. Applying a chat template to part of it -- a single
message, or any slice -- is not equivalent and this checkpoint's template rejects some
fragments outright.{% endblock %}"""


class MultitenantBrief(ImplementerBrief):
    """ImplementerBrief + this task's framing: the SLO-normalized-latency objective, the
    no-failed-request gate and the token-level correctness gate (the ``goal`` block
    override), plus the serving-envelope ceiling (the ``contract`` block). The
    optimize-until-budget goal is stated once (the goal block's opening) and enforced by
    the harness ralph loop, not by directive text. Task-level only.

    Every fact a contestant is graded against sits in the section that defines it,
    above ``## Your budget``. The trailing header-less ``directives`` block is left
    empty: it renders after the budget section, which is past the point an agent
    reading top-down starts working, and these facts are corrective -- each is a fact a
    contestant can reasonably get wrong."""

    template_str: str | None = _GOAL_BLOCK


# Production contestant: the live implementer agent with a high turn
# budget for a long (ralph-loop) optimization session. The brief carries the objective and
# the wall-clock budget (see MultitenantBrief); the ralph loop is
# Claude Code's native /goal, armed by the agent build's kickoff (from soft_budget).
ARTIFACT = AgentProducedArtifact(
    producer=AgentProducerSpec(
        # The provider's top level for whatever harness and model an arm launches with,
        # resolved at launch (the top_effort entry of the arm's harness adapter) and
        # recorded in the run's config.json. Never implicit: the harnesses' own defaults
        # differ and go unrecorded, and the run path refuses to launch without a value.
        reasoning_effort=HighestAvailable(),
        # Offline toolchain documentation, landed in the working tree at this path so
        # the agent greps it like its own source; see _TOOLCHAIN_UPLOAD_BRIEF
        # (tasks/sample_task/task.py) for what it contains and why it's uploaded
        # rather than baked.
        upload_dir=("../assets/references",), upload_sha256=_TOOLCHAIN_REFERENCES_SHA256,
        upload_dest="/workspace/references",
        upload_brief=_TOOLCHAIN_UPLOAD_BRIEF,
        # Wall-time budget mode: only wall-clock bounds the ralph-loop session
        # (turns=None -> Claude Code's --max-turns is omitted), so a productive agent is never cut
        # off early by an arbitrary turn cap. Hard wall-clock cap (orchestrator wait_for) = soft
        # budget + _HARD_CAP_HEADROOM_HOURS; on expiry the committed main is gracefully collected +
        # evaluated. GPU cost is then hard-bounded by wallclock_sec x the build's GPUs.
        #
        # The soft budget arms the /goal kickoff (wallclock_sec set -> the agent build
        # composes the native /goal with the brief) and is what the agent can read: the
        # build registers the `budget` MCP server (the `get_budget` tool) only when
        # soft_budget is set, and the brief's `## Your budget` section renders only then.
        # Without it the goal condition (proven via get_budget output) could never be
        # judged. Keep soft strictly below hard (artifact.py validates per-dimension):
        # the grace above it lets the orchestrator collect committed main instead of
        # killing mid-commit.
        soft_budget=Budget(wallclock_sec=int(_AGENT_BUDGET_HOURS * 3600), turns=None),
        hard_budget=Budget(wallclock_sec=int((_AGENT_BUDGET_HOURS + _HARD_CAP_HEADROOM_HOURS) * 3600), turns=None),
        # What an evaluation holds: 1 H200, 8 CPUs and 180 GiB for the server, less the
        # runner's reserve (run.runner_memory_gib, 4 GiB by default), and 16 GiB beside it
        # for the driver, for a 196 GiB job.
        evaluation_allocation=EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180, driver_memory_gib=16),
        # What a build holds beside its H200: 8 CPUs and 180 GiB.
        build_allocation=BuildAllocation(cpus=8, memory_gib=180),
        # Sealed (air-gap raw internet) + build-from-scratch: the package policy denies
        # _DENIED_SERVING_FRAMEWORKS (tasks/sample_task/task.py) so the agent writes its
        # own engine.
        network=NetworkPolicy(
            domains=AccessList(mode="allow"),  # air-gap: no raw internet hosts
            packages=AccessList(mode="deny", entries=_DENIED_SERVING_FRAMEWORKS),
        )),
    brief=MultitenantBrief(),
    serve_command=[],
)

#: The measured baseline, which is also the vLLM calibration member: stock vLLM 0.22.1 on the cuda
#: base, from locks/baseline.txt.
_VLLM = LayerImage(name="multitenant-slo-baseline", parent="cuda", pip_lock="baseline.txt", owner="multitenant_slo", server=True,
                   env=SERVER_ENV)
#: The image the HF reference engine runs on: torch 2.11 and transformers 5.13.1 on the cuda base,
#: from locks/reference.txt.
_REFERENCE = LayerImage(name="multitenant-slo-reference", parent="cuda", pip_lock="reference.txt", owner="multitenant_slo",
                        server=True, env=SERVER_ENV)
#: The SGLang calibration member: stock SGLang 0.5.16 on the cuda base, from locks/member-sglang.txt,
#: with the kernels its FlashInfer 0.6.14 would otherwise download when it first loads them, in the
#: directory its environment names.
_SGLANG = LayerImage(name="multitenant-slo-member-sglang", parent="cuda", pip_lock="member-sglang.txt",
                     owner="multitenant_slo", server=True, fetched=FLASHINFER_CUBINS["0.6.14"],
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

__all__ = [
    "TASK",
    "ARTIFACT",
    "CONTRACT",
    "TENANT_CHAT",
    "TENANT_API_SHORT",
    "TENANT_REASONING",
    "TENANT_LONG_SESSION",
]
