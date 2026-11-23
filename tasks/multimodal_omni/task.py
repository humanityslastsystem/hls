"""multimodal-omni -- Omni (mixed-modality) serving arena + the DEFAULT (agentic) contestant.

Serve a native-omni model that takes a *mixture of modality requests* on ONE server --
some carry an image + a question (VQA), some carry speech to translate (S2TT) -- under a
fixed open-loop Poisson load, and answer faster than the vLLM-Omni baseline does: the
score is the geometric mean of the per-request speedups in SLO-normalized latency. The
omni-specific challenge is co-scheduling image-encoder and audio-encoder work, which a
contestant that gets it wrong pays for as queueing delay on every request.

``MixedModalityDataset`` interleaves a vision source (VQAv2 image+question -> answer)
and an audio source (FLEURS fr->en SPEECH TRANSLATION -- the S2TT setup the Qwen3-Omni
report evaluates with BLEU) into ONE merged corpus by ``MIX_RATIO``.

Correctness is reference-relative, one gate per arm. A greedy Qwen3-Omni oracle
(``Qwen3OmniReference``, loaded by the verify stage) answers the same requests;
both rollouts are scored by ``MixedModalityMatchMetric`` against the dataset ground
truth -- word F1 on the image arm, smoothed sentence BLEU on the audio arm -- and each
arm's mean must clear its own multiplicative bar against the oracle's. Both gates must
pass, so a strong image arm cannot pay for an abandoned audio one.
"""

from __future__ import annotations

from hls.prompt import ImplementerBrief
from hls.spec.baseline import Baseline
from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec,
                                   Budget, BuildAllocation, EvaluationAllocation, HighestAvailable)
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import FleursS2TTDataset, HFVQADataset, MixedModalityDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.core.decode import Greedy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.drive.bounds import VerifyPeak, bounds_for
from hls.reference.omni import Qwen3OmniReference
from hls.scoring import SloLatencyScorer
from hls.spec.taskspec import TaskSpec
from hls.spec.wire import OmniChatProtocol
from hls.verify import MixedModalityMatchMetric, ReferenceRelativeVerifier
from hls.spec.workload import (NoDelay, Poisson, RequestLatencySLO, ShortOutputCap,
                               Workload)
from hls.recipes.catalog import SERVER_ENV, ImageSet, Served
from hls.recipes.model import LayerImage

# Instruct, not Thinking: the Thinking variant
# spends the whole ShortOutputCap budget on <think> CoT before the answer, zeroing
# correctness by construction under a short-answer workload.
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SAMPLE_RATE = 16_000
#: Per-request output cap, stamped on every request so the contestant and the oracle
#: decode under the same budget. Sized to bind only a reply that fails to stop: a FLEURS
#: en_us reference translation runs to about 76 tokens at the longest, so 256 is over 3x
#: the longest genuine answer either arm produces.
MAX_NEW_TOKENS = 256
SEED = 42

# --- Difficulty + scoring knobs ----------------------------------------------
#: The offered rate the graded drive runs at, held fixed for the whole drive.
#:
#: Where it sits relative to the baseline's knee decides what the score measures. Well
#: below the knee, latency is service time and a server that barely batches can win
#: while holding less capacity. At the knee, latency is queueing time, which goes as
#: 1/(mu - lambda) and so is dominated by capacity, and the barely-batching server is
#: the first to collapse because its mu is the smaller one.
#:
#: 30.0 rps is one rung under the highest rate at which the vLLM-Omni baseline's queue
#: stays level on two H200s, 31.5; at 32 the queue grows without bound.
OFFERED_RPS = 30.0
#: Requests in the graded drive. Sets the resolution of the score: the geometric mean of
#: the per-request speedups has a relative standard error of roughly 1/sqrt(N), so 4600
#: buys about 1.5% -- far finer than a capacity search could afford, whose every value
#: has to be a rung of the grid that searched for it. At this rate that is a 2 minute
#: drive; the corpus below is sized to hold it, since a drive can only send what the
#: test split holds and at this rate a small corpus drains before the queue settles.
TEST_REQUESTS = 4600
#: The drive opens cold, so what the served stack does once and never again -- kernel
#: autotune, CUDA graph capture, a cold cache, and here two encoders and a 30B MoE
#: decoder each paying it separately -- lands inside the measured window. The baseline
#: pays it through this same drive, so it is in B as well as in L. Keeping it on the
#: clock measures the contestant's own bring-up.
#: Per-request JCT budget, the denominator of r = JCT / target. Not a pass/fail bar: r is
#: a continuous normalizer, so this sets the SCALE latencies are read on rather than a
#: cliff, and a request at exactly the budget earns credit 0. Under SloLatencyScorer the
#: budget also cancels out of the score -- log(target) enters both the contestant's mean
#: log(1/r) and the baseline's, so it sets where OFFERED_RPS lands, not how anyone ranks.
#:
#: 15 s is 1.25x MLPerf Inference's Server constraint for its Qwen3-VL-235B-A22B
#: vision-language benchmark, 12000 ms at p99 (mlcommons/inference, loadgen/mlperf.conf:
#: `qwen3-vl-235b-a22b.Server.target_latency = 12000`, with `use_token_latencies = 0`,
#: which is what makes it a whole-query bound rather than a TTFT). That benchmark
#: measures the same quantity this task does -- send to complete reply -- on the same
#: request shape, "one prompt that contains some text and a base64-encoded image", and
#: for a model roughly 7x larger than this one. Holding a smaller model to a slightly
#: looser budget is the conservative direction.
#:
#: Two limits on that citation: it comes from the in-development v6.1 round rather than
#: a released one, and MLPerf's benchmark is image-plus-text only, so the S2TT arm here
#: has no counterpart in it. Do NOT reach for Qwen3-Omni's own published 234 ms figure:
#: that is time to the first packet of streamed OUTPUT AUDIO, and this contract returns
#: unary text.
#:
#: On two H200s the baseline's p99 JCT at the pinned rate is about 10.5 s.
TARGET_JCT_MS = 15_000
MIX_RATIO = 0.5                              # fraction of image-input (VQA) requests
#: Multiplicative accuracy bar. With A = the oracle's mean on an arm and B = the
#: contestant's, that arm passes when B >= A * REL_FLOOR. One value for both arms: the
#: bar asks the same question of each -- did this implementation keep the checkpoint's
#: accuracy -- and an arm-specific number would be answering how noisy that arm's metric
#: is instead, which belongs in the metric rather than in the gate.
#:
#: A faithful vLLM-Omni serving keeps about 0.995 of the oracle's image score and 0.999
#: of its audio score. 0.98 rather than 0.99 because both gates must hold and image is
#: the weaker.
#:
#: Sentence BLEU, a product of four n-gram precisions, does not bind: it is the slacker
#: arm of the two by a wide margin.
REL_FLOOR = 0.98                             # word F1 (VQA arm), sentence BLEU (S2TT arm)
#: Per-event pass bars, on each arm's own scale. Nothing here scores off them -- the
#: scorer reads latency, and each gate compares its arm's MEAN against the oracle's --
#: so they only label each event in the verdict record. Each arm needs its own bar
#: because BLEU 0.15 is a usable translation while word F1 0.15 is a wrong VQA answer.
IMAGE_PASS = 0.5
AUDIO_PASS = 0.15
#: Per-arm corpus encode pool. At a 50/50 mix the merge draws evenly from the two arms,
#: so each pool has to cover half the merged corpus below, with a little spare in case a
#: source arm yields short.
#:
#: The audio arm is the binding one. FLEURS fr_fr holds 676 test utterances, which capped
#: the whole corpus near 1300 rows -- too few to drive OFFERED_RPS for long enough to
#: measure anything, since a drive can only send what the test split holds and 1300
#: requests drain in half a minute at this rate. Drawing train as well as test raises the
#: audio side to 3869 utterances. Train is a sound source here because nothing in this
#: task trains on it: the arm is scored against an oracle decode of the same rows, so
#: this is a pool of speech to serve, not a held-out set to claim a result on.
_POOL = 2700

MODEL = ModelSpec(base_checkpoint=MODEL_ID, revision="26291f793822fb6be9555850f06dfe95f2d7e695", arch="Omni",
                  chat_format="qwen3-omni")
#: Two cards, which is what vLLM-Omni's own deploy config for this model uses: the
#: thinker takes cuda:0 and the talker and code2wav share cuda:1. Nothing in the stock
#: CUDA profile is tensor-parallel, so a third and fourth card would sit idle. Handing out
#: four would let a contestant spread over hardware the reference
#: never touched and score the difference as a serving win.
HARDWARE = Hardware(accelerator=AcceleratorSpec(kind="H200", count=2, memory_gb_per_device=141),
                    cpu_count=64, memory_gb=512, storage_gb=2048)
# Multimodal chat: messages[].content is a list of typed parts (text / image_url /
# input_audio). Output is text. transport "http" (unary JSON).
CONTRACT = RequestResponseContract(
    path="/v1/chat/completions",
    input_modalities=frozenset({"image", "audio", "text"}),
    output_modalities=frozenset({"text"}), transport="http",
    protocol=OmniChatProtocol(model=MODEL_ID))

# Mixture of modality requests: VQAv2 (image+question) interleaved with FLEURS fr->en
# speech translation, merged into one provisionable corpus. Both carry a text GT
# (answer / English reference translation). The merged corpus must hold dev + test.
_VISION = HFVQADataset(
    path="lmms-lab/VQAv2", revision="32665d35052eb4a6d4414851c3c829a72754915a", source_split="validation",
    question_col="question", answer_col="multiple_choice_answer", image_col="image",
    question_template="Answer with one word or a short phrase (no explanation): "
                      "{question}", corpus_size=_POOL)
_AUDIO = FleursS2TTDataset(src_lang="fr_fr", tgt_lang="en_us",
                           revision="70bb2e84b976b7e960aa89f1c648e09c59f894dd", source_split="train+test",
                           target_sample_rate=SAMPLE_RATE, corpus_size=_POOL)
#: corpus_size caps the merged row count, and partition_by_hash refuses dev + test above
#: it, so the merge has to hold both splits (4664 rows) with room left over -- the spare
#: rows are the unassigned middle that keeps split membership stable when the corpus
#: shifts by a row.
DATASET = MixedModalityDataset(
    vision=_VISION, audio=_AUDIO, mix_ratio=MIX_RATIO, corpus_size=5200,
    label="multimodal_omni-omni",
    audio_instruction="Translate the French speech into English; "
                      "output only the English translation.")
# The oracle's answers to the dev rows and their scores are a measurement
# (hls.verify.dev_reference): the store's or the default in defaults/dev_reference.json,
# published beside the dev rows under the name declared below. Both gates here are relative, so without this a contestant can measure its
# own word F1 and sentence BLEU on dev and still not know what it has to beat.

DATA_SOURCE = DataSource(
    dataset=DATASET,
    # Dev is never driven. It exists to carry the disclosed oracle answers below, so 64
    # rows is a sample large enough to cover both modalities and a range of lengths for a
    # contestant checking its own metric implementation, and small enough that dev + test
    # fit inside the merged corpus with spare rows left.
    split=SplitSpec(dev=Count(value=64), test=Count(value=TEST_REQUESTS)),
    measured_dev_reference="oracle_dev.json")

# Reference-relative, per arm: the oracle answers every request once, and each arm's verifier
# reduces over its own modality's events only. Both verifiers must pass, so neither
# arm's score can subsidize the other -- at a 50/50 mix a single averaged floor is
# clearable with one arm abandoned entirely.
#
# The oracle is a greedy HF-transformers decode of the task's own checkpoint, and the
# bars below are relative to that decode.
#
# The oracle answers every request, which is affordable only because those answers are
# cached beside the task: they depend on the rows and the oracle's own config, never on
# the contestant, so the corpus is answered once and every later run and repeat reads it
# back. Uncached this decode runs about 15 to 33 min per graded run on the declared
# hardware, at the end of every contestant's budget.
_ORACLE = Qwen3OmniReference(decode=Greedy(), max_new_tokens=MAX_NEW_TOKENS,
                             sample_rate=SAMPLE_RATE, cache=True)
VERIFIERS = [
    ReferenceRelativeVerifier(
        metric=MixedModalityMatchMetric(arm="image", pass_threshold=IMAGE_PASS),
        reference=_ORACLE, min_ratio=REL_FLOOR),
    ReferenceRelativeVerifier(
        metric=MixedModalityMatchMetric(arm="audio", audio_measure="bleu",
                                        pass_threshold=AUDIO_PASS),
        reference=_ORACLE, min_ratio=REL_FLOOR),
]
# One seeded drive at a fixed offered rate. Open-loop: arrivals fire on the schedule
# whether or not the server has kept up, so concurrency is the server's problem and the
# queue it builds behind the two encoders shows up in every request's latency.
WORKLOAD = Workload(
    arrival=Poisson(rate_per_sec=OFFERED_RPS),
    inter_turn_delay=NoDelay(),
    output_length_cap=ShortOutputCap(cap_tokens=MAX_NEW_TOKENS),
    slo=RequestLatencySLO(target_ms=TARGET_JCT_MS))
#: score.value = exp(L - B): the geometric mean of the per-request speedups over the
#: baseline stack, where L is this drive's mean log(1/r) and B the baseline's. 1.0 ties
#: the baseline, 2 means a typical request is served at half its SLO-normalized latency,
#: and there is no ceiling -- unlike a sustained rate, whose every value has to be a rung
#: of the grid that searched for it.
#:
#: B is MEASURED, never authored:
#:   hls baseline-credit tasks.multimodal_omni.task
#: drives the vLLM-Omni build-to-beat over this exact workload, each drive on a fresh deploy,
#: and writes baseline.json into the store; the task's default is defaults/baseline.json.
#: The run loader injects the mean and refuses a record whose workload, load or
#: GPU kind has moved, so the denominator cannot silently rot.
SCORER = SloLatencyScorer()

# The default brief's goal block describes the goodput scorer: serve correctly and inside
# the SLO, and more requests per second scores higher. Under SloLatencyScorer none of that
# is the objective. Arrivals are open-loop at a fixed rate, so throughput is not the
# server's to choose, and there is no plateau at "every SLO met" -- that is where scoring
# starts. An agent reading the default would chase headroom it is not paid for and stop
# optimizing exactly when it should begin.
_GOAL_BLOCK = """{% extends "brief" %}
{% block goal %}Serve `{{ model.base_checkpoint }}` under the workload below. Optimize until
you reach the physical limits of the hardware or exhaust your budget.

**Your score is your speedup over a production serving framework in SLO-normalized
latency.** For every request, against the per-request SLO listed under ## Workload:

    r = (time from request sent to response complete) / that SLO

    score = exp( mean over requests of log(1/r) )
            / the same quantity for the production framework on this workload

Higher is better, with no ceiling. A score of 1.0 means you matched the reference stack,
2.0 means a typical request took half as long. Meeting every SLO is not the finish line:
the score keeps rising as latency falls, so there is no point at which further speed
stops paying.

Requests arrive open-loop on a Poisson schedule you do not control, at a fixed rate.
Serving faster does not earn you more requests -- it earns you a better score on the ones
you get. Every request is weighted equally, so a slow tail costs as much as it would in
production.

The traffic is an even mix: half the requests carry an image with a question about it,
half carry French speech to translate into English. Both go to the same endpoint and are
scored on one scale, so a design that serves one modality well and queues the other
behind it pays for the queue.

**Your implementation must not fail any request.** One failed request zeroes the run,
which means shedding or refusing load is never a way to protect your latency.

Correctness is scored against a bf16 HuggingFace reference decode of this same
checkpoint, answering the same requests. Each modality is gated separately and both must
hold: word F1 against the reference answer on the image arm, sentence BLEU against the
reference translation on the speech arm. Each must reach {{ (ts.verifiers[0].min_ratio * 100) | round | int }}% of what that reference
scores on the same run. Those bars are on your mean over the whole run, not on any single
request, and they are pass/fail with no credit for margin.

`oracle_dev.json` beside the dev rows carries the reference's answer and score for every
dev row, tagged with the arm it belongs to, plus a mean per arm. The bars are fractions of
those numbers, so this is what to compare your own dev scores against. Read them as the
levels to aim at rather than proof you clear the bars: the dev split is 64 rows and it
divides between the two arms, leaving about thirty on each, which is far too few to
separate a ratio of 0.98 from one of 1.00. A dev score near a bar does not settle the
graded run either way.{% endblock %}
{% block dev_reference_note %}`oracle_dev.json` sits beside those rows with the
reference's answer and score for each of them, tagged with the arm it belongs to.{% endblock %}
"""

TASK = TaskSpec(id="multimodal-omni", model=MODEL, hardware=HARDWARE, contract=CONTRACT,
                workload=WORKLOAD, verifiers=VERIFIERS, scorer=SCORER,
                data_source=DATA_SOURCE, seed=SEED,
                # 21 GiB: the rows role's measured resident peak, 16.12 GiB, plus a quarter,
                # rounded up. The role holds the 1.6 GB of test rows as several copies while it
                # writes them.
                provision_memory_gib=21,
                # What one drive may make the harness hold (hls.drive.bounds). A drive's trace is
                # almost all requests, about 1.5 GiB, and four times it would be a 6 GiB cap; its
                # replies come to about 1.2 MB on the wire. The verify role cannot hold a 6 GiB
                # trace: it holds 81.93 GiB whatever the trace,
                # Qwen3-Omni's 65.68 GiB checkpoint and 1 GiB for its reference engine, and
                # 15.25 GiB for scoring one 256 MiB reply; and 13.6 bytes besides per byte of
                # trace: an arm's gate keeps that arm's replies whole, densest with a logprob per
                # token the protocol keeps unasked. That makes the cap 3 GiB, about twice a
                # drive's trace.
                drive_bounds=bounds_for(CONTRACT, trace_bytes=1_607_516_171, held_reply_bytes=1_171_424,
                                        trace_request_bytes=2 * 2**30,
                                        verify_peak=VerifyPeak(base_bytes=round(81.93 * 2**30),
                                                               bytes_per_byte=13.6)),
                # The baseline is vLLM-Omni serving Qwen3-Omni (vllm serve --omni with its
                # packaged qwen3_omni_moe.yaml, behind a passthrough adapter pinning
                # modalities=["text"]), which the measured baseline image (IMAGES.baseline) starts.
                # No serve_command tuning: the packaged deploy config owns the per-stage
                # memory/device layout, and a global flag does not reach the stage engines.
                baseline=Baseline(image="hls-baseline-multimodal-omni"),
                metadata={"theme": "multimodal", "app": "omni",
                          "host": HARDWARE.accelerator.kind, "mix_ratio": str(MIX_RATIO)})

# Wall-clock budget the contestant is told it has, and works against through the
# `get_budget` tool. This contestant serves two encoders and a 30B MoE decoder, so its
# build and its tuning loop are both longer than the speech arenas'. 12 h, the graded
# build budget.
_AGENT_BUDGET_HOURS = 12.0

# Grace the orchestrator waits above the soft budget before it force-collects committed
# main. 30 min is enough for an agent that paced to the soft number to land its last
# commit, without letting a hung one idle for hours.
_HARD_CAP_HEADROOM_HOURS = 0.5

ARTIFACT = AgentProducedArtifact(
    producer=AgentProducerSpec(
        # Two budgets, as every other agentic task in this tree declares them.
        #
        # SOFT is the contestant's: it is the number the brief's "## Your budget"
        # section states and the only thing that arms the `get_budget` tool, because
        # the agent build gates both on soft_budget being set. Without it an agent cannot see
        # the clock at all -- not in the brief, not at runtime -- and simply stops
        # existing when the hard cap fires, shipping whatever is committed on main at the
        # instant of the kill.
        #
        # HARD is the orchestrator's, half an hour above: the grace lets it collect the
        # committed main rather than killing mid-commit.
        #
        # No turn cap on either. It cannot bind evenly: the harnesses take different
        # numbers of steps for the same work, so a turn cap could cut one and not the
        # other.
        #
        # No cost ceiling: a budget has no cost bound. The wall clock bounds the spend.
        soft_budget=Budget(wallclock_sec=int(_AGENT_BUDGET_HOURS * 3600), turns=None),
        hard_budget=Budget(
            wallclock_sec=int((_AGENT_BUDGET_HOURS + _HARD_CAP_HEADROOM_HOURS) * 3600),
            turns=None),
        # What an evaluation holds: 2 H200, 8 CPUs and 180 GiB for the server, less the
        # runner's reserve (run.runner_memory_gib, 4 GiB by default), and 16 GiB beside it for
        # the driver, for a 196 GiB job.
        evaluation_allocation=EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180, driver_memory_gib=16),
        # What a build holds beside its 2 H200s: 8 CPUs and 180 GiB.
        build_allocation=BuildAllocation(cpus=8, memory_gib=180),
        # The run path refuses a task that leaves this implicit: the harnesses
        # disagree on their own defaults (claude-code emits no --effort flag at all,
        # codex defaults to "high"), so an unstated effort makes two arms differ on
        # an axis the record never names.
        reasoning_effort=HighestAvailable()),
    brief=ImplementerBrief(template_str=_GOAL_BLOCK),
    serve_command=[])

# The images the task runs on: the shared layer its sandbox, submission base and harness build on,
# and the images its own servers run on, each from this directory's locks. The reference engine's
# image builds on the cuda base.
#
# The baseline is vLLM 0.28.0 and vLLM-Omni 0.28.0 behind server_vllm_omni.py, which speaks the
# arena's wire. Two packages of its lock are published as source distributions only, openai-whisper
# (which vLLM-Omni requires) and antlr4-python3-runtime (which its omegaconf requires): the image
# builds both against the setuptools of the layer below it.
#
# The empty file /.dockerenv decides how vLLM-Omni's stage workers size their KV cache. A worker
# that finds itself in a container with its own PID namespace takes vLLM's memory profile of the
# stage (the requested share less weights, peak activation and non-torch memory). Any other worker
# subtracts what NVML reports for its own PID, and NVML lists a process of a PID namespace under
# another PID, so that figure is 0 and the thinker stage asks for a KV cache that does not fit
# beside its weights. vLLM-Omni tells a container by this file, which Docker creates in every
# container and Apptainer in none, so the image holds it.
IMAGES = ImageSet(
    layer="s-a-tf-5-9-0",
    baseline=Served(image=LayerImage(name="multimodal-omni-baseline", parent="cuda-setuptools",
                                     pip_lock="baseline.txt", owner="multimodal_omni", server=True,
                                     source_builds={"antlr4-python3-runtime==4.9.3": ["setuptools"],
                                                    "openai-whisper==20250625": ["setuptools"]},
                                     files={"/app/server_vllm_omni.py": "server_vllm_omni.py",
                                            "/.dockerenv": "dockerenv"},
                                     env=SERVER_ENV),
                    entrypoint=["python", "/app/server_vllm_omni.py"]),
    reference=LayerImage(name="multimodal-omni-reference", parent="cuda", pip_lock="reference.txt",
                         owner="multimodal_omni", server=True, env={**SERVER_ENV, "PIP_NO_CACHE_DIR": "1"}),
)

__all__ = ["TASK", "ARTIFACT", "CONTRACT", "SEED",
           "MODEL", "HARDWARE", "DATASET", "DATA_SOURCE", "VERIFIERS", "WORKLOAD", "SCORER"]
