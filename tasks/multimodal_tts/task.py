"""multimodal-tts -- TTS serving arena + the DEFAULT (agentic) contestant.

Serve a streaming text-to-speech model: text in, audio frames streamed out (http-sse
via ``TTSStreamProtocol``) under a fixed open-loop Poisson load, and answer faster than
the vLLM-Omni baseline stack does: the score is the
geometric mean of the per-request speedups in SLO-normalized latency. The
baseline's own credit is measured by ``hls baseline-credit``, kept as the task's default
defaults/baseline.json, and injected by the run loader (see SCORER).

Correctness is faithfulness to the checkpoint's own decode. Every one of the sixteen
code groups is sampled, so there is no single right code a served one can be compared
against, and the reply carries what the server's own sampler was doing: the codes it
served, and for each position the probability it gave the code it picked together with
its top alternatives. Two gates read that. ``TtsReportFaithfulnessVerifier`` prices the
same positions under the pinned oracle and compares the two distributions, against a
calibrated envelope. ``CodecFidelityVerifier`` re-decodes the served codes with the
oracle's codec decoder and compares the result to the waveform that was actually
streamed. Between them they say that the contestant drew from the distribution this
checkpoint defines, and that the audio it shipped is those codes decoded.

That replaces an intelligibility gate. Round-trip word error rate asks only whether a
listener can make out the sentence, which any competent synthesizer manages: even a
classical formant synthesizer reaches near-oracle round-trip accuracy, so such a gate
cannot see that a different model answered. Faithfulness to the pinned
checkpoint is the property this arena pays for.

``TtsReportFaithfulnessVerifier`` needs a ``calibration.json`` fingerprinted to this task's
model, contract, seed and dataset, from the store or the task's defaults, and ``hls run``
measures one when neither holds one. The default was measured on
H200s (vLLM-Omni reporting its sampling distribution). The artifact is a measurement of
specific silicon running specific weights, so one taken on anything but the accelerator
this task declares would be a fiction. Recalibrate whenever the workload fingerprint
moves -- the model, the contract (including the codec-codes flag and the codec's shape), the seed, the dataset
or the split sizes. The arrival rate, the SLO budget and the scorer sit outside that
fingerprint by the harness's design (``hls.verify.calibration.workload_payload``); a
change to one of those re-measures the baseline credit and nothing else.
"""

from __future__ import annotations

from hls.spec.baseline import Baseline
from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec,
                                   Budget, BuildAllocation, EvaluationAllocation, HighestAvailable)
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import ColumnProjectedHFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.prompt import ImplementerBrief, JinjaPromptTemplate
from hls.core.decode import Sample
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.drive.bounds import VerifyPeak, bounds_for
from hls.reference.tts import (CODEC_SAMPLE_RATE, REPETITION_PENALTY, SUBTALKER_SAMPLE,
                               TEMPERATURE, Qwen3TTSReference)
from hls.scoring import SloLatencyScorer
from hls.spec.taskspec import TaskSpec
from hls.spec.wire import TTSStreamProtocol
from hls.verify.tts_faithfulness import (CodecFidelityVerifier, LOGMEL_FLOOR,
                                         LOGMEL_HOP, LOGMEL_N_FFT, LOGMEL_N_MELS,
                                         LOGMEL_PSNR_FLOOR_DB, MAX_LENGTH_DELTA_FRAMES,
                                         TtsReportFaithfulnessVerifier)
from tasks.sample_task.task import _TOOLCHAIN_REFERENCES_SHA256, _TOOLCHAIN_UPLOAD_BRIEF
from hls.spec.workload import (NoDelay, Poisson, RequestLatencySLO, ShortOutputCap,
                               Workload)
from hls.recipes.catalog import SERVER_ENV, ImageSet, Served
from hls.recipes.model import LayerImage

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SEED = 42

#: Codebook groups per codec frame, and ids per group, for this checkpoint's 12 Hz
#: tokenizer v2. Stated on the contract so a reply whose rows are another shape is
#: caught on the wire instead of being misread as frames by the gate, and rendered into
#: the brief so the contestant is told the shape it must send.
CODEC_GROUPS = 16
CODEC_CODEBOOK_SIZE = 2048

# --- Difficulty + scoring knobs ----------------------------------------------
#: The offered rate of the graded drive, in requests per second, held fixed for the whole drive.
OFFERED_RPS = 2.0
#: Requests in the graded drive, which is the size of the test split.
TEST_REQUESTS = 2400
#: Per-request latency target in milliseconds, the denominator of r = JCT / target.
TARGET_JCT_MS = 10_000
#: Voice every request is synthesized in, and the language it is synthesized under.
#:
#: Both are contract pins rather than preferences. The speaker embedding and the language
#: id enter the talker's prefix, so a request answered in another voice or another
#: language is a different conditioning and its codes are not the ones this checkpoint
#: would have chosen for the request that was sent. The oracle is conditioned on exactly
#: these, and the gate prices the served codes under them.
ORACLE_SPEAKER = "Ryan"
ORACLE_LANGUAGE = "English"
#: Style instruction prepended to the conditioning. Empty is the pin: an instruction is
#: another prefix, and changes the codes for the same sentence.
ORACLE_INSTRUCT = ""
#: Per-request budget in codec frames, which is what bounds the talker: one frame is
#: 80 ms of audio, so this is 102.4 seconds of speech.
#:
#: Under the pinned decode the 64 dev sentences run 91.4 frames on average and 315 at
#: the longest. Four times that longest is 1,260; 1,280 is the round number above it.
#:
#: Four times, and not tighter, because of what a low cap costs on each side. The graded
#: split is 2,400 sentences from the same corpus as those 64, so its longest is longer.
#: On the grading side a cap under the longest honest
#: utterance rejects a faithful reply. On the calibration side it is worse: ingest
#: refuses a capture whose replies ran past the stamped cap, on the grounds that the
#: trusted member did not honour the contract, so one long sentence discards a whole
#: serving rather than one reply.
#:
#: And not looser, because the ceiling is real: vLLM-Omni's talker stage runs
#: `max_tokens: 4096` and the checkpoint's own generation config says 8,192, so a cap
#: above the smaller of those would let the baseline produce replies the cap then calls
#: over-budget. 1,280 sits at about a third of that ceiling.
FRAME_CAP = 1_280

MODEL = ModelSpec(base_checkpoint=MODEL_ID, revision="0c0e3051f131929182e2c023b9537f8b1c68adfe", arch="TX",
                  chat_format="qwen3-tts")
#: The task's declared host: 8 CPUs and 240 GB. No brief states
#: these: the build's brief states the sandbox's own figures and the graded
#: server's are the evaluation allocation's, both declared on ARTIFACT;
#: these stand only where a task declares no allocation. Neither enters the workload fingerprint.
HARDWARE = Hardware(accelerator=AcceleratorSpec(kind="H200", count=1, memory_gb_per_device=141),
                    cpu_count=8, memory_gb=240, storage_gb=2048)
#: Alternatives the report carries at every position, well past the text tasks' 21.
#: The served-outside check (check 5 of the report gate) can only tell a faithful report
#: from one that hides its draws in the tail while a faithful server's served code falls
#: outside its reported alternatives at a rate well under one. Under the plain softmax
#: at temperature 1 a codec position spreads its mass over many codes: behind a report
#: of 21 about half of served codes fall outside it, which the gate's 2x tolerance lifts
#: past 100%.
REPORT_WIDTH = 65

CONTRACT = RequestResponseContract(
    path="/v1/audio/speech", input_modalities=frozenset({"text"}),
    output_modalities=frozenset({"audio"}), transport="http-sse",
    # The reply ends with the codec codes the audio was decoded from, which is the
    # evidence both gates read. Per-frame token ids stay off: the codes frame already
    # carries the whole utterance's codes once, and a second copy spread over the audio
    # frames would be a second thing to keep consistent for no gain.
    protocol=TTSStreamProtocol(model=MODEL_ID, return_token_ids=False,
                               return_codec_codes=True, return_logprobs=True,
                               top_logprobs=REPORT_WIDTH,
                               codec_groups=CODEC_GROUPS,
                               codec_codebook_size=CODEC_CODEBOOK_SIZE))

#: The talker's end-of-speech id, which the gate prices at every served position and the
#: servers report as the served code of the position after the last frame. It is the
#: checkpoint's ``talker_config.codec_eos_token_id``.
#:
#: Asserted rather than restated. The weights are not present when this module is
#: imported, so the strongest check available here is that the contract and the number
#: this task believes agree; the reference engine checks it against the loaded config,
#: which is where a checkpoint that moved it would be caught. Without this an id changed
#: on one side would leave the servers reporting a position the gate prices differently,
#: which is a wrong verdict rather than a missing one.
CODEC_EOS_ID = 2150
if CONTRACT.protocol.codec_eos_id != CODEC_EOS_ID:
    raise ValueError(
        f"this task believes end-of-speech is {CODEC_EOS_ID} and the wire contract says "
        f"{CONTRACT.protocol.codec_eos_id}; the servers report that position and the "
        f"gate prices it, so the two cannot differ")

# LibriTTS prompts (text to synthesize). keep_columns drops the raw audio clip we ignore
# so the corpus survives the rows.json materialization. Prompt and target are the same
# sentence, because that is what the request is: the TTS is told to say it.
DATASET = ColumnProjectedHFDataset(
    path="mythicinfinity/libritts", subset="clean",
    revision="35f26bdafa9883b5524f17931216af78632db8eb", source_split="dev.clean",
    keep_columns=["text_normalized"],
    prompt=JinjaPromptTemplate(template="{{ text_normalized }}"),
    target=JinjaPromptTemplate(template="{{ text_normalized }}"))

DATA_SOURCE = DataSource(
    dataset=DATASET,
    # Dev is never driven. It is there so a contestant can run its own server against
    # real sentences from this corpus before the graded split arrives; 64 leaves the
    # corpus for the test split. Nothing is published beside those rows: correctness is
    # agreement with the contestant's own checkpoint, which it holds, so there is no
    # oracle answer to disclose that it cannot compute.
    split=SplitSpec(dev=Count(value=64), test=Count(value=TEST_REQUESTS)))

# Correctness, in two gates, and between them they hold ONE accelerator at a time.
#
# The runner deploys per verifier, releasing any live engine the next one does not need
# before it acquires, so what must fit on the card is one verifier's engine set rather
# than the list's. Both of these need the same one: they share a single oracle config
# OBJECT, so the second reuses the first's deploy instead of acquiring. The cache is
# keyed on the config's identity, and two equal-but-distinct objects would be two
# deploys of one checkpoint.
#
# The oracle is the in-process checkpoint, not the vLLM-Omni serving stack of the
# measured baseline image: a Reference builds an engine inside the harness. vLLM-Omni is
# the performance baseline (the credit SCORER is read against), which is a separate
# question from whether the reported distributions are this checkpoint's.
#
# No cache=True on the oracle. It never synthesizes a corpus every arm shares: it prices
# the positions the contestant served, which are new each arm.
#
# Every decode field comes from the reference module's own constants rather than being
# restated here, so the oracle, both servers and the brief cannot disagree about what
# was decoded.
_ORACLE = Qwen3TTSReference(decode=Sample(temperature=TEMPERATURE),
                            speaker=ORACLE_SPEAKER,
                            language=ORACLE_LANGUAGE, instruct=ORACLE_INSTRUCT,
                            repetition_penalty=REPETITION_PENALTY,
                            subtalker_sample=SUBTALKER_SAMPLE)

#: Decode fields the brief renders straight off the oracle. Jinja renders a missing
#: attribute as an empty string, so a reference that stopped carrying one of these would
#: publish a contract with the number silently absent, and a contestant would read a rule
#: with a hole in it. Checked at import instead, where it is loud.
_RENDERED_DECODE_FIELDS = ("speaker", "language", "repetition_penalty", "temperature",
                           "top_k", "subtalker_temperature", "subtalker_top_k")
_missing = [f for f in _RENDERED_DECODE_FIELDS if not hasattr(_ORACLE, f)]
if _missing:
    raise AttributeError(
        f"{type(_ORACLE).__name__} does not carry {_missing}, which the brief states as "
        f"the pinned decode; the task cannot publish a contract it cannot render")

# The sampling gate. Under the checkpoint's own decode every one of the 16T+1 positions
# is sampled, so there is no argmax for a served code to be faithful to and the gate
# reads what the server REPORTED instead. Every threshold it enforces is measured by
# calibration; the task authors none of them.
_REPORT_GATE = TtsReportFaithfulnessVerifier(reference=_ORACLE, report_width=REPORT_WIDTH)
# The waveform half, and the only gate that reads the codes as codes. Both bounds are
# the verifier module's own constants rather than literals repeated here, so this gate
# and the brief state the same number without anyone copying it across.
#
# The 29.0 dB floor sits 3.0 dB under the worst honest decoder variant (a chunked decode
# at a short left context) and 2.7 dB over the best negative (the reference decode at
# +6 dB gain). LOGMEL_PSNR_FLOOR_DB carries the derivation.
_CODEC_GATE = CodecFidelityVerifier(reference=_ORACLE,
                                    min_logmel_psnr_db=LOGMEL_PSNR_FLOOR_DB,
                                    max_length_delta_frames=MAX_LENGTH_DELTA_FRAMES)
VERIFIERS = [_REPORT_GATE, _CODEC_GATE]
# One seeded drive at a fixed offered rate. Open-loop: arrivals fire on the schedule
# regardless of whether the server has kept up, so concurrency is the server's problem
# and the queue it builds shows up in every request's latency.
WORKLOAD = Workload(
    arrival=Poisson(rate_per_sec=OFFERED_RPS),
    inter_turn_delay=NoDelay(), output_length_cap=ShortOutputCap(cap_tokens=FRAME_CAP),
    slo=RequestLatencySLO(target_ms=TARGET_JCT_MS))
#: score.value = exp(L - B): the geometric mean of the per-request speedups over the
#: baseline stack, where L is this drive's mean log(1/r) and B the baseline's. 1.0 is a
#: tie with the baseline, 2 means a typical request is served at half its SLO-normalized
#: latency, and there is no ceiling -- which is why the objective is this rather than a
#: sustained rate, whose every value must be a rung of whatever grid searched for it.
#:
#: B is MEASURED, never authored:
#:   hls baseline-credit tasks.multimodal_tts.task
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
#
# The decoding block states the pins the gate enforces. Every one of them is a lever on
# latency under any gate that reads the audio alone, which is why each is written as a
# rule and not as a recommendation.
#
# The output rate is spliced from the codec's own constant rather than typed into the
# text, so a brief can never state a rate the decoder does not produce.
_OUTPUT_KHZ = f"{CODEC_SAMPLE_RATE / 1000:g} kHz"

# The analysis the waveform check runs on, spliced from the verifier's own constants for
# the same reason. A contestant reproducing this comparison has to run the identical
# analysis: each of these four choices moves the number, so a self-check under another
# convention says nothing about the one that grades.
_LOGMEL_PARAMS = (
    f"    {CODEC_SAMPLE_RATE // 1000} kHz, n_fft {LOGMEL_N_FFT}, hop {LOGMEL_HOP}, "
    f"{LOGMEL_N_MELS} mels, centred Hann window\n"
    f"    POWER spectrogram (power=2.0), HTK mel scale, UNNORMALIZED filters "
    f"(norm=None)\n"
    f"    floored at {LOGMEL_FLOOR:g}, then base-10 log")

_GOAL_BLOCK = """{% extends "brief" %}
{% block goal %}Serve `{{ model.base_checkpoint }}` under the workload below. Optimize until
you reach the physical limits of the hardware or exhaust your budget.

**Your score is your speedup over a production serving framework in SLO-normalized
latency.** For every request, against the per-request SLO listed under ## Workload:

    r = (time from request sent to the last audio frame) / that SLO

    score = exp( mean over requests of log(1/r) )
            / the same quantity for the production framework on this workload

Higher is better, with no ceiling. A score of 1.0 means you matched the production
framework, 2.0 means a typical utterance took half as long. The SLO is on the complete utterance,
not on first audio. Requests arrive open-loop on a Poisson schedule you do not control,
at a fixed rate.

**Your implementation must not fail any request.** One failed request zeroes the run.
The scored test split is """ + f"{TEST_REQUESTS:,}" + """ sentences, driven once and cold: no warm-up requests are
sent, so your first requests are timed like the rest. Every reply must end on end of
speech ({{ ts.contract.protocol.codec_eos_id }}) within the output cap under ## Workload; a reply that reaches the cap
without ending fails.

**Correctness is faithfulness to this checkpoint's own decode.** All {{ ts.contract.protocol.codec_groups }} code groups are
sampled, so you are not graded by comparing your codes against a reference's. You are
graded on what you report: at every position, the log-probability you gave the code you
served and that position's top {{ ts.contract.protocol.top_logprobs }} alternatives, under the distribution you actually
sampled from. Two checks read the whole run. The first reads the report. For every served
code, the reference model is run on the same prefix and the KL divergence from your reported
distribution to the reference's is averaged over all graded positions; the run fails if this
mean exceeds {% if verification and verification.kind == "report" %}{{ verification.kl_ceiling_str }} nats per position{% else %}a ceiling calibrated on trusted serving stacks{% endif %}.
The codes you served must be faithful draws from the distributions you report; this is
tested statistically, and a server that reports one distribution and samples from another
fails. The reported alternatives must be the most probable codes: the run fails if more
than {% if verification and verification.kind == "report" %}{{ verification.outside_limit_pct_str }} percent{% else %}a fraction calibrated on trusted serving stacks{% endif %} of your served codes are not among the
alternatives you reported at their position, counting only those above your
{{ ts.verifiers[0].report_width - 1 }}th-largest reported value (the last two set the cut-off). Ending utterances
early is tested too: how often you end where the reference gives end of speech little
probability is held to what the reference expects.

The second check reads the audio: the {{ ts.contract.protocol.codec_groups }} groups of codes you sent are decoded by the reference codec
decoder, and any utterance whose log-mel PSNR against that decode falls below
{{ ts.verifiers[1].min_logmel_psnr_db | round | int }} dB, or whose length differs by more than {{ ts.verifiers[1].max_length_delta_frames }} codec frames, fails.
The log-mel analysis is:

""" + _LOGMEL_PARAMS + """

and the PSNR's peak is the reference log-mel's own range (maximum minus minimum) over the
frames both utterances share.

The report's two ceilings are calibrated on a trusted serving stack of this checkpoint;
the audio floor sits 3 dB under the worst honest decoder variant measured. What fails is genuine
deviation: other weights, a quantization, another temperature, top-k or penalty, a report
of a distribution you did not sample from. Both checks are pass/fail with no credit for
margin -- a run that passes at 90% of a ceiling scores identically to one at 10% -- so
latency spent buying faithfulness headroom below a ceiling is pure loss. A reply whose
codes frame or report is missing, duplicated or malformed fails.

{% endblock %}
{% block decoding %}**Decoding.** Every request is answered by `{{ model.base_checkpoint }}`'s
own shipped weights, under this decode, with no substitute model in the path:

- Speaker `{{ ts.verifiers[0].reference.speaker }}`, language `{{ ts.verifiers[0].reference.language }}`, and an empty style instruction.
- Non-streaming text mode: the whole sentence is in the talker's prefix before the first
  frame is generated.
- All {{ ts.contract.protocol.codec_groups }} code groups sampled from the plain softmax at
  temperature {{ ts.verifiers[0].reference.temperature }}: no top-k, no top-p, no repetition penalty.
- On codebook 0, the talker's special ids (2048 through 3071, except end of speech,
  {{ ts.contract.protocol.codec_eos_id }}) are masked at every step, and end of speech is masked at frames 0 and 1.
- Pass every one of these explicitly. The checkpoint's `generation_config` supplies
  temperature 0.9, top-k 50 and a repetition penalty of 1.05 to anything you leave unset.
- Report, at every position, the log-probability you gave the code you served and that
  position's top {{ ts.contract.protocol.top_logprobs }} alternatives, under the plain softmax at temperature
  {{ ts.verifiers[0].reference.temperature }} after those masks, so a masked id carries no probability.
- Emit the codec decoder's own """ + _OUTPUT_KHZ + """ waveform unmodified: no resampling, no
  trimming, no gain, no time-stretching, no denoising. How you run the decoder is yours.
  Decoding in chunks with left context is allowed, but it does move the waveform: the
  worst chunked variant measured sits 3 dB above the audio floor.

The per-request output cap under ## Workload counts codec frames, not text tokens: one
frame is {{ ts.contract.protocol.codec_groups }} codes and 80 ms of audio.{% endblock %}
{% block build_instructions %}Write your server as `{{ server_path }}`, run as
`python {{ server_path }} --host <H> --port <P>`. It is built and evaluated OFFLINE.
`{{ server_path }}` need only be the entrypoint, and may exec or bind whatever you build.

- Weights: a local checkpoint (config + safetensors + tokenizer) mounted read-only at
  `$HLS_WEIGHTS_DIR`. The accelerator is `cuda:0`, with {{ hardware.cpu_count }} CPU cores and {{ hardware.memory_gb }} GB of host
  RAM beside it.
- Data: a development split of 64 sentences is on disk at `{{ dev_rows_path }}`; the
  scored test split is withheld. Each row's `text_normalized` is the sentence to say,
  verbatim, and `source_index` identifies it; nothing else sits beside the rows.
{{ self.package_policy() }}
- Also installed for this task: `qwen-tts` 0.1.1, the checkpoint's reference
  implementation, with the packages it depends on; `torchaudio`, `numpy`, `numba`,
  `soundfile`; and `ffmpeg`, `sox`, `libsndfile` on the system.

{{ self.toolchain() }}{% endblock %}
"""

TASK = TaskSpec(id="multimodal-tts", model=MODEL, hardware=HARDWARE, contract=CONTRACT,
                workload=WORKLOAD, verifiers=VERIFIERS, scorer=SCORER,
                data_source=DATA_SOURCE, seed=SEED,
                # What one drive may make the harness hold (hls.drive.bounds). A drive's replies
                # come to about 7.8 GB on the wire, which held unwritten at once at the stream's
                # 10.6 trace bytes per byte (a report position of one alternative) would make the
                # cap 98 GiB. The verify role cannot hold a 98 GiB trace: it holds 29.4 GiB
                # whatever the trace, 5.3 GiB for its reference
                # (Qwen3-TTS's 4.3 GiB checkpoint and 1 GiB of engine) and 24.1 GiB for the codec
                # gate's decode of every reply at the 1,280-frame cap, which the workload bounds;
                # and 2.11 bytes besides per byte of trace, the codec gate's served audio held as
                # float32 until its length check, densest at 96 kHz just inside it. That makes the
                # cap 61 GiB, about three times a drive's trace.
                drive_bounds=bounds_for(CONTRACT, trace_bytes=22_886_907_405, held_reply_bytes=7_774_318_266,
                                        trace_request_bytes=4 * 2**20,
                                        verify_peak=VerifyPeak(base_bytes=round(29.4 * 2**30),
                                                               bytes_per_byte=2.11)),
                # The baseline is vLLM-Omni serving Qwen3-TTS (vllm serve --omni with its
                # packaged qwen3_tts.yaml, rewritten to the contract's sampled decode, behind
                # a thin SSE adapter), which the measured baseline image (IMAGES.baseline) starts.
                # No serve_command tuning: the packaged deploy config owns the per-stage
                # memory/device layout, and a global flag like --gpu-memory-utilization
                # OOMs the stage pipeline.
                baseline=Baseline(image="hls-baseline-multimodal-tts"),
                metadata={"theme": "multimodal", "app": "tts",
                          "host": HARDWARE.accelerator.kind})

# Wall-clock budget the contestant is told it has, and works against through the
# `get_budget` tool. 12 h, the graded build budget.
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
        # What an evaluation holds: 1 H200, 8 CPUs and 240 GiB for the server, less the
        # runner's reserve (run.runner_memory_gib, 4 GiB by default), and 32 GiB beside it for
        # the driver; 272 GiB in all. A drive's replies come to about 12 GB packed, but the
        # streaming driver holds only the replies still arriving or not yet written, and past
        # 85% of its memory every further reply fails (hls.drive.memory). Its verify stage
        # takes the planner's 2 h, about twice what a verify stage takes on this hardware: the
        # report gate reads each utterance back in a few transfers, and a drive's verify comes
        # to about 2,450 s with image extraction and weights copy.
        evaluation_allocation=EvaluationAllocation(cpus=8, job_memory_gib=240, writable_gib=240, driver_memory_gib=32),
        # What a build holds beside its H200: 8 CPUs and 240 GiB.
        build_allocation=BuildAllocation(cpus=8, memory_gib=240),
        # The run path refuses a task that leaves this implicit: the harnesses
        # disagree on their own defaults (claude-code emits no --effort flag at all,
        # codex defaults to "high"), so an unstated effort makes two arms differ on
        # an axis the record never names.
        reasoning_effort=HighestAvailable(),
        # The offline documentation set the brief's toolchain section points into
        # (tasks/assets/references, gitignored; README.md's walkthrough says how it is put in place).
        upload_dir=("../assets/references",), upload_sha256=_TOOLCHAIN_REFERENCES_SHA256,
        upload_dest="/workspace/references",
        upload_brief=_TOOLCHAIN_UPLOAD_BRIEF,
        # No network here: the default producer network denies the harness's package list
        # (hls.egress.denied_packages), which includes vllm-omni, the production framework
        # the score is measured against. Installing it would reproduce the baseline rather
        # than build a server.
    ),
    brief=ImplementerBrief(template_str=_GOAL_BLOCK),
    serve_command=[])

# The images the task runs on. The shared layer its sandbox, submission base and harness build on
# holds qwen-tts and the torch it pins, so the reference engine runs on the harness image and the
# task declares no reference image. The baseline is vLLM and vLLM-Omni from this directory's lock,
# behind server_vllm_omni.py, which speaks the arena's wire; omni_codes_spool.py is the stage-0
# producer hook the adapter names to vLLM-Omni, which carries the codec codes out of the engine.
# openai-whisper and antlr4-python3-runtime are published as source distributions alone, so the
# image builds both, against the setuptools of the layer it builds on.
IMAGES = ImageSet(
    layer="s-a-tf-4-57-3-qwen-tts",
    baseline=Served(image=LayerImage(name="multimodal-tts-baseline", parent="cuda-setuptools",
                                     pip_lock="baseline.txt", owner="multimodal_tts", server=True,
                                     source_builds={"antlr4-python3-runtime==4.9.3": ["setuptools"],
                                                    "openai-whisper==20250625": ["setuptools"]},
                                     files={"/app/server_vllm_omni.py": "server_vllm_omni.py",
                                            "/app/omni_codes_spool.py": "omni_codes_spool.py"},
                                     env=SERVER_ENV),
                    entrypoint=["python", "/app/server_vllm_omni.py"]),
)

__all__ = ["TASK", "ARTIFACT", "CONTRACT", "SEED", "MODEL",
           "HARDWARE", "DATASET", "DATA_SOURCE", "VERIFIERS", "WORKLOAD", "SCORER"]
