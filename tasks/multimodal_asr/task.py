"""multimodal-asr -- Standard ASR arena + the DEFAULT (agentic) contestant.

Serve Whisper-large-v3 under a fixed open-loop Poisson load pinned near the baseline
stack's knee, and answer faster than it does: the score is the geometric mean of the
per-request speedups in SLO-normalized latency. Near the knee latency is queueing
time, so serving more capacity is how a contestant lowers it. One arena (TASK);
the DEFAULT contestant is the live implementer agent (ARTIFACT =
AgentProducedArtifact). This module exports ONLY declarative constants.

Correctness is reference-relative: ``ReferenceRelativeVerifier`` runs a greedy Whisper oracle
(``WhisperReference``, loaded by the verify stage) over the same audio; both
rollouts are scored 1 - WER against the dataset transcript, and the contestant's mean
accuracy must be >= the oracle's * ``WER_REL_FLOOR`` (a multiplicative bar) -- the
streaming-engine contestant need not byte-match the reference, only retain (nearly all
of) its accuracy.

Audio rides as base64 PCM16 WAV inside the JSON Envelope and the JSON request body
(``AudioTranscriptionProtocol`` over plain ``http`` transport) -- no multipart.
"""

from __future__ import annotations

from hls.prompt import ImplementerBrief
from hls.spec.baseline import Baseline
from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec,
                                   Budget, BuildAllocation, EvaluationAllocation, HighestAvailable)
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import HFAudioDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.core.decode import Greedy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.drive.bounds import VerifyPeak, bounds_for
from hls.reference.asr import WhisperReference
from hls.scoring import SloLatencyScorer
from hls.spec.taskspec import TaskSpec
from hls.spec.wire import AudioTranscriptionProtocol
from hls.verify import ReferenceRelativeVerifier, WERMetric
from hls.spec.workload import (NoDelay, Poisson, RequestLatencySLO, ShortOutputCap,
                               Workload)
from hls.recipes.catalog import SERVER_ENV, ImageSet, Served
from hls.recipes.model import LayerImage

MODEL_ID = "openai/whisper-large-v3"
SAMPLE_RATE = 16_000
#: Per-request output cap, stamped on every request so the contestant and the oracle
#: decode under the same budget. 440 is whisper-large-v3's own 30-second decode budget,
#: and it is meant to bind only a reply that fails to stop: the longest LibriSpeech
#: test.clean transcript is about 120 tokens, so nothing genuine comes near it.
MAX_NEW_TOKENS = 440
#: Single source of truth for the decode language: fed to BOTH the request contract
#: (what the harness stamps on every request) and the greedy oracle, so the two can
#: never silently drift under a future non-English task. Whisper's HF generate accepts
#: the ISO code ("en") as well as the name ("english") -- both resolve to the same
#: forced-language token -- so the oracle consumes this code form directly.
LANGUAGE = "en"
SEED = 42

# --- Difficulty + scoring knobs ----------------------------------------------
#: The offered rate the graded drive runs at, held fixed for the whole drive.
#:
#: It has to sit near the baseline's knee, and that is the whole design of this task.
#: The score is SLO-normalized latency relative to the baseline's, and how much that
#: measures CAPACITY rather than per-request speed depends entirely on where the load
#: is pinned. Well below the knee, latency is just service time, and a server that
#: barely batches wins while having less capacity. At the knee, latency is queueing
#: time, which goes as 1/(mu - lambda) and so is dominated by capacity -- and the
#: barely-batching server now collapses first, because its mu is the smaller one.
#: Pinning here is what makes a latency measurement a capacity measurement.
#:
#: Pinned at the highest rate the baseline sustains, which is not the same as the
#: highest goodput it reports. Goodput keeps climbing after a server stops keeping up,
#: because attainment falls more slowly than the rate rises, so a rate chosen on goodput
#: alone can sit in overload and the latency it reports is then a function of how long
#: the drive ran rather than of the server. The baseline sustains about 52 requests/s
#: on one H200, with a level queue; above that its queue grows without bound.
#:
#: A rate that drops connections is excluded outright rather than ranked, because this
#: task zeroes a run on any failed request: a rate the reference stack cannot serve
#: cleanly defines no baseline credit and would zero contestants for the reference's own
#: behaviour.
#:
#: The baseline's adapter keeps one HTTP client: a client per request collapses it well
#: below the knee.
#:
#: whisper-large-v3 reads a fixed 30-second window through a large encoder whatever the
#: clip's length, so this arena is encoder-bound and decode length barely reaches it.
OFFERED_RPS = 52.0
#: Requests in the graded drive. Sets the resolution of the score: the geometric mean
#: of per-request speedups has a relative standard error of roughly 1/sqrt(N), so 2400
#: buys about 2%. At this rate that is a 46 second drive, opening cold: whatever the
#: served stack does once and never again -- kernel autotune, CUDA graph capture, a cold
#: cache -- lands inside the measured window. That is deliberate. The baseline pays the
#: same opening through the same drive, so it is in B as well as in L and divides out of
#: exp(L - B) to the extent the two stacks pay it alike; a server that warms in faster
#: than the baseline should be able to show it. The rate is one the baseline sustains, so
#: past the opening latency is stationary rather than still climbing.
#:
#: The ceiling is the corpus, and it is a ceiling worth keeping: LibriSpeech test.clean
#: holds 2620 utterances and the dev split takes 64, so 2400 is the largest round number
#: that sends each clip at most once. Going past it would repeat clips within a drive,
#: which a server could answer from a cache keyed on the audio rather than by
#: transcribing.
TEST_REQUESTS = 2400
#: Per-request JCT budget, the denominator of r = JCT / target. Not a pass/fail bar:
#: r is a continuous normalizer, so this sets the SCALE latencies are read on rather
#: than a cliff, and a request at exactly the budget earns credit 0. Under
#: SloLatencyScorer the budget also cancels out of the score -- log(target) enters both
#: the contestant's mean log(1/r) and the baseline's, so it sets where OFFERED_RPS
#: lands, not how anyone ranks.
#:
#: 6 s is 6x MLPerf Inference's Server constraint for its (retired) RNN-T speech
#: recognition benchmark, 1000 ms at p99 on LibriSpeech utterances under 15 seconds
#: (mlcommons/inference, loadgen/mlperf.conf: `rnnt.Server.target_latency = 1000`).
#: That benchmark measures the same quantity this task does -- whole-query latency from
#: send to complete reply -- on the same corpus and the same one-utterance-per-request
#: shape. The 6x covers the model gap: RNN-T is ~120M parameters against
#: whisper-large-v3's ~1.55B, and Whisper reads a fixed 30-second window whatever the
#: clip's length.
#:
#: No current benchmark constrains this directly. MLPerf replaced RNN-T with Whisper in
#: v5.1, and that benchmark is Offline-only with no latency constraint at all, so the
#: retired RNN-T bound is the most recent normative ASR latency figure published.
#: At the pinned rate the baseline's JCT on one H200 is about 1.5 s at the median and
#: 3.6 s at p95, so the budget sits above the loaded distribution without being so far
#: above it that everything passes regardless.
TARGET_JCT_MS = 6_000
#: Multiplicative accuracy bar: with A = the HF whisper oracle's mean accuracy
#: (1 - WER vs the LibriSpeech transcript) and B = the contestant's, require
#: B >= A * WER_REL_FLOOR.
#:
#: A faithful vLLM serving of these weights keeps 0.999 of the oracle's accuracy over
#: the test rows, and the bar has to sit below that with room for engines that are not
#: this one. 0.98 leaves about 2%, which covers
#: ordinary numeric differences and some latitude on decode, while still rejecting a
#: server that has traded real accuracy for speed. 0.99 would leave 0.9%, not enough
#: for an implementation that differs from the reference in any deliberate way.
WER_REL_FLOOR = 0.98

MODEL = ModelSpec(base_checkpoint=MODEL_ID, revision="06f233fe06e710322aca913c1bc4249a0d71fce1", arch="EncDec",
                  chat_format="whisper")
# Runtime egress is NOT a Hardware property (it lives on TaskSpec.network, which defaults
# to None = AIR_GAP at runtime -- the deployed artifact is already sealed).
HARDWARE = Hardware(accelerator=AcceleratorSpec(kind="H200", count=1, memory_gb_per_device=141),
                    cpu_count=64, memory_gb=512, storage_gb=2048)
CONTRACT = RequestResponseContract(
    path="/v1/audio/transcriptions",
    input_modalities=frozenset({"audio"}), output_modalities=frozenset({"text"}),
    transport="http",
    protocol=AudioTranscriptionProtocol(model=MODEL_ID, sample_rate=SAMPLE_RATE, language=LANGUAGE))

# ESB (English ASR leaderboard), librispeech config. ONE provider split is the corpus
# (source_split); SplitSpec partitions it into the dev/test units.
DATASET = HFAudioDataset(
    path="hf-audio/open-asr-leaderboard", subset="librispeech",
    revision="b6bdcd0beb34f8975dc659796176d88f43aff502", source_split="test.clean",
    audio_col="audio", text_col="text", target_sample_rate=SAMPLE_RATE)
# Published beside every tenant's dev rows: what the oracle transcribed for each of
# them and what that scored. The correctness bar is relative -- a contestant passes by
# keeping a fraction of the ORACLE's accuracy, not by clearing a fixed number -- so
# without this a contestant cannot tell whether it is passing. "A greedy Whisper decode"
# does not pin it either: the decode config, the normalizer inside the metric and the
# empty-reference convention each move the score. Per row rather than as an aggregate,
# so a contestant can check its own metric implementation example by example.
#
# A measurement (hls.verify.dev_reference): the store's or the default in
# defaults/dev_reference.json, published beside the dev rows under the name declared below. It
# covers the dev split only -- the test split it is measured on stays withheld.

DATA_SOURCE = DataSource(
    dataset=DATASET,
    # Dev is never driven. It exists to carry the disclosed oracle transcripts below,
    # so 64 clips is a sample large enough for a contestant to check its own metric
    # implementation against, and small enough to leave the corpus for the test split.
    split=SplitSpec(dev=Count(value=64), test=Count(value=TEST_REQUESTS)),
    measured_dev_reference="oracle_dev.json")

# Reference-relative correctness: the verifier owns the greedy Whisper oracle; the oracle's model identity
# comes from TASK.model (whisper-large-v3) so it matches the contestant's Whisper.
# Both sides are scored 1 - WER against the dataset transcript; the bar is
# MULTIPLICATIVE (B >= A * WER_REL_FLOOR), so the slack scales with the oracle's
# accuracy instead of granting a flat WER budget.
# cache=True: the oracle answers the same 2,400 clips for every arm, and what
# it answers depends only on those rows and its own config, never on the contestant. Left
# uncached it is recomputed inside each graded run, about 25 minutes on one H200,
# spent identically every time. The answers are precomputed by
# `hls oracle fill` into the task's answers area of the store (hls.verify.answers), which
# holds the reference's answers to the withheld test split, so only the harness's verify
# and oracle fill roles mount it and never a container a contestant can read.
_ORACLE = WhisperReference(decode=Greedy(), max_new_tokens=MAX_NEW_TOKENS,
                           language=LANGUAGE, cache=True)
VERIFIERS = [ReferenceRelativeVerifier(metric=WERMetric(), reference=_ORACLE,
                                       min_ratio=WER_REL_FLOOR)]
# One seeded drive at a fixed offered rate. Open-loop: arrivals fire on the schedule
# regardless of whether the server has kept up, so concurrency is the server's problem
# and the queue it builds shows up in every request's latency.
WORKLOAD = Workload(
    arrival=Poisson(rate_per_sec=OFFERED_RPS),
    inter_turn_delay=NoDelay(),
    output_length_cap=ShortOutputCap(cap_tokens=MAX_NEW_TOKENS),
    slo=RequestLatencySLO(target_ms=TARGET_JCT_MS))
#: score.value = exp(L - B): the geometric mean of the per-request speedups over the
#: baseline stack, where L is this drive's mean log(1/r) and B the baseline's. 1.0 is a
#: tie with the baseline, 2 means a typical request is served at half its
#: SLO-normalized latency, and there is no ceiling -- which is why the objective is
#: this rather than a sustained rate, whose every value must be a rung of whatever
#: grid searched for it.
#:
#: B is MEASURED, never authored:
#:   hls baseline-credit tasks.multimodal_asr.task
#: drives the vLLM build-to-beat over this exact workload, each drive on a fresh deploy,
#: and writes baseline.json into the store; the task's default is defaults/baseline.json.
#: The run loader injects the mean and refuses a record whose
#: workload, load or GPU kind has moved, so the denominator cannot silently rot.
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

**Your implementation must not fail any request.** One failed request zeroes the run,
which means shedding or refusing load is never a way to protect your latency.

Correctness is scored against a bf16 HuggingFace reference decode of this same
checkpoint on the same clips, decoding greedily with no sampling and no beam search.
Your mean transcription accuracy must be at least {{ (ts.verifiers[0].min_ratio * 100) | round | int }}% of what that reference scores
on the same run. That bar is on your mean over the whole run, not on any single request,
and it is pass/fail with no credit for margin.

Accuracy is `1 - WER`: word edit distance over the number of reference words, after
normalising both sides with Whisper's `EnglishTextNormalizer`
(`transformers.models.whisper.english_normalizer`), constructed with an empty spelling
map. That is the normalizer `open-asr-leaderboard` scores this corpus with. It
lowercases, strips punctuation, expands contractions, and folds numbers and currency to
one form, so `$250,000` and `two hundred and fifty thousand dollars` compare equal. An
empty reference scores 1.0 against an empty prediction and 0.0 otherwise.

`oracle_dev.json` beside the dev rows carries the reference's transcript and score for
every dev clip, its mean, and a worked normalisation example, so you can check your own
implementation of this metric row by row.{% endblock %}
{% block decoding %}**Decoding.** Transcribe greedily: argmax, no sampling, no beam
search.{% endblock %}
{% block dev_reference_note %}`oracle_dev.json` sits beside those rows with the
reference's answer and score for each of them.{% endblock %}
"""

TASK = TaskSpec(id="multimodal-asr", model=MODEL, hardware=HARDWARE, contract=CONTRACT,
                workload=WORKLOAD, verifiers=VERIFIERS, scorer=SCORER,
                data_source=DATA_SOURCE, seed=SEED,
                # What one drive may make the harness hold (hls.drive.bounds). A drive's trace is
                # almost all requests, about 0.76 GB, and four times it is the cap, 3 GiB; its
                # replies come to about 1.1 MB on the wire. The verify role
                # holds 19.25 GiB whatever the trace: whisper-large-v3's
                # 2.88 GiB checkpoint and 1 GiB for its reference engine, 15.25 GiB for the word
                # error rate of one 256 MiB reply, and 0.12 GiB; and 1.11 bytes besides per byte of
                # trace, the requests its reference-relative gate keeps. At the 3 GiB cap that is
                # 24 GiB.
                drive_bounds=bounds_for(CONTRACT, trace_bytes=760_508_726, held_reply_bytes=1_095_914,
                                        trace_request_bytes=2**30,
                                        verify_peak=VerifyPeak(base_bytes=round(19.25 * 2**30),
                                                               bytes_per_byte=1.11)),
                # The baseline is the vLLM build-to-beat: vllm serve behind a thin adapter
                # speaking the arena's JSON wire, which the measured baseline image
                # (IMAGES.baseline) starts. `.image` is the identity the baseline credit records.
                # serve_command flags ride through the adapter (parse_known_args) into
                # the inner `vllm serve`.
                # No serve_command: the baseline is stock vLLM. A --gpu-memory-utilization
                # under this vLLM's own 0.92 default would handicap it, and KV capacity
                # never binds on this workload.
                baseline=Baseline(image="hls-baseline-multimodal-asr", serve_command=[]),
                metadata={"theme": "multimodal", "app": "asr",
                          "host": HARDWARE.accelerator.kind})

# Wall-clock budget the contestant is told it has, and works against through the
# `get_budget` tool. 12 h, the graded build budget.
_AGENT_BUDGET_HOURS = 12.0

# Grace the orchestrator waits above the soft budget before it force-collects committed
# main. 30 min is enough for an agent that paced to the soft number to land its last
# commit, without letting a hung one idle for hours.
_HARD_CAP_HEADROOM_HOURS = 0.5

# DEFAULT contestant: the live implementer agent (pure config). The arm names the model
# (`hls run --arm`). The agent ships a runnable repo (pyproject + server.py), which the
# package stage builds the submission image from.
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
        # What an evaluation holds: 1 H200, 8 CPUs and 180 GiB for the server, less the
        # runner's reserve (run.runner_memory_gib, 4 GiB by default), and 16 GiB beside it for
        # the driver, for a 196 GiB job.
        evaluation_allocation=EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180, driver_memory_gib=16),
        # What a build holds beside its H200: 8 CPUs and 180 GiB.
        build_allocation=BuildAllocation(cpus=8, memory_gib=180),
        # The run path refuses a task that leaves this implicit: the harnesses
        # disagree on their own defaults (claude-code emits no --effort flag at all,
        # codex defaults to "high"), so an unstated effort makes two arms differ on
        # an axis the record never names.
        reasoning_effort=HighestAvailable()),
    brief=ImplementerBrief(template_str=_GOAL_BLOCK),
    serve_command=[])

# The images the task runs on: the shared layer its sandbox, submission base and harness build on,
# and the images its own servers run on, each on the cuda base from this directory's locks. The
# baseline is vLLM behind server_vllm.py, which speaks the arena's wire; the reference is the
# Whisper engine's torch and transformers.
IMAGES = ImageSet(
    layer="s-a-tf-5-9-0",
    baseline=Served(image=LayerImage(name="multimodal-asr-baseline", parent="cuda", pip_lock="baseline.txt",
                                     owner="multimodal_asr", server=True,
                                     files={"/app/server_vllm.py": "server_vllm.py"},
                                     env=SERVER_ENV),
                    entrypoint=["python", "/app/server_vllm.py"]),
    reference=LayerImage(name="multimodal-asr-reference", parent="cuda", pip_lock="reference.txt",
                         owner="multimodal_asr", server=True, env=SERVER_ENV),
)

__all__ = ["TASK", "ARTIFACT", "CONTRACT", "SEED",
           "MODEL", "HARDWARE", "DATASET", "DATA_SOURCE", "VERIFIERS", "WORKLOAD", "SCORER"]
