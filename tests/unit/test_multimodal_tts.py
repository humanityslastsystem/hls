"""Unit tests for the multimodal-tts pieces.

Covers the TTSStreamProtocol JSON/SSE codec, the Qwen3TTSReference oracle config, and
that the multimodal-tts TaskSpec constructs and wires. The two verifiers this task declares
are tested in ``test_tts_faithfulness.py``.
"""

from __future__ import annotations

import numpy as np

from hls.core.audio import encode_wav_b64
from hls.core.payloads import AudioPayload, Envelope, TextPayload
from hls.verify.report import TOP_LOGPROBS_REQUESTED


def _tone(freq, n=4000, sr=16_000):
    return encode_wav_b64(np.sin(np.linspace(0, 6.28 * freq, n)).astype(np.float32), sr)


def _oracle_tts():
    from hls.core.decode import Greedy
    from hls.reference.tts import Qwen3TTSReference
    return Qwen3TTSReference(decode=Greedy())


# ---- TTSStreamProtocol codec ---------------------------------------------

def test_tts_stream_protocol_encode_and_unary_decode():
    from hls.spec.wire import TTSStreamProtocol
    proto = TTSStreamProtocol(model="tts", return_token_ids=True)
    env = Envelope(request_id="r", contents=[TextPayload(text="hello there")])
    body = proto.encode_request(env, stream=True)
    assert body["input"] == "hello there" and body["stream"] is True
    b64 = _tone(220, 1600)
    out = proto.decode_response({"audio": b64, "sample_rate": 16_000, "token_ids": [7, 8]}, "r")
    ap = out.first_of("audio")
    assert isinstance(ap, AudioPayload) and ap.tokens == [7, 8] and ap.sample_rate == 16_000


def test_tts_stream_protocol_assembles_frames():
    from hls.spec.wire import TTSStreamProtocol
    proto = TTSStreamProtocol(model="tts")
    f1, f2 = _tone(220, 800), _tone(220, 800)
    chunks = [(0.10, {"audio": f1, "sample_rate": 16_000, "token_ids": [1]}),
              (0.20, {"audio": f2, "sample_rate": 16_000, "token_ids": [2]})]
    resp = proto.decode_stream(chunks, "r")
    ap = resp.output.first_of("audio")
    assert resp.t_first_packet == 0.10 and ap.tokens == [1, 2]
    assert ap.duration_sec and ap.duration_sec > 0.0


def test_tts_probe_request_empty():
    from hls.spec.wire import TTSStreamProtocol
    assert TTSStreamProtocol().probe_request()["input"] == ""


# ---- TTS oracle config ------------------------------------------------------

def test_oracle_tts_serves_the_task_model():
    from tasks.multimodal_tts import task as t
    eng = _oracle_tts().build_engine(t.TASK)
    assert eng.model_spec.base_checkpoint == t.MODEL_ID
    assert eng.gpu_kind == t.HARDWARE.accelerator.kind
    assert (eng.speaker, eng.language) == (t.ORACLE_SPEAKER, t.ORACLE_LANGUAGE)


def test_the_oracle_cannot_be_teacher_forced_at_all():
    """Under the checkpoint's own decode every codebook is sampled, so there is no
    argmax for a served code to be compared against.

    The engine does not offer the capability rather than offering it and refusing when
    called. A gate that wanted teacher forcing then fails to wire, before any GPU is
    allocated, instead of raising part way through a verify phase."""
    from hls.core.capabilities import TeacherForce
    from hls.core.engine import supports
    from hls.reference.tts import Qwen3TTSEngine
    assert TeacherForce not in supports(Qwen3TTSEngine)
    assert not hasattr(Qwen3TTSEngine, "teacher_force")


# ---- multimodal-tts wiring ---------------------------------------------------------

def test_multimodal_tts_taskspec_constructs():
    from hls.core.capabilities import DecodeCodes, ReferenceLogprobs, StackReport
    from hls.core.engine import supports
    from tasks.multimodal_tts import task as t
    assert t.TASK.id == "multimodal-tts" and t.CONTRACT.transport == "http-sse"
    assert t.CONTRACT.output_modalities == frozenset({"audio"})
    assert [type(v).__name__ for v in t.VERIFIERS] == [
        "TtsReportFaithfulnessVerifier", "CodecFidelityVerifier"]
    # Under the checkpoint's own decode every one of the sixteen groups is sampled, so
    # there is no argmax to be faithful to and the gate reads the server's report.
    assert t.CONTRACT.protocol.emits_codec_codes
    assert t.CONTRACT.protocol.return_logprobs
    # The report is wider than the text tasks' default, and the contract and the gate
    # agree on it: the contract asks servers for the width the gate partitions in.
    assert t.REPORT_WIDTH > TOP_LOGPROBS_REQUESTED
    assert t.CONTRACT.protocol.top_logprobs == t.REPORT_WIDTH
    assert t.VERIFIERS[0].report_width == t.REPORT_WIDTH
    assert t.CONTRACT.protocol.codec_groups == t.CODEC_GROUPS
    assert t.CONTRACT.protocol.codec_codebook_size == t.CODEC_CODEBOOK_SIZE
    # ONE oracle object across both gates, so the verify phase deploys one engine.
    assert t.VERIFIERS[0].reference is t.VERIFIERS[1].reference
    assert (t.VERIFIERS[0].reference.speaker, t.VERIFIERS[0].reference.language) == (
        t.ORACLE_SPEAKER, t.ORACLE_LANGUAGE)
    # What each gate asks the engine for, by name rather than read off the gate's own
    # `requires`: a check derived from the gates would pass whatever they happened to
    # ask for. The report gate prices positions and needs the report stacked; the
    # waveform gate re-decodes the codes.
    assert t.VERIFIERS[0].requires == frozenset({ReferenceLogprobs, StackReport})
    assert t.VERIFIERS[1].requires == frozenset({DecodeCodes})
    caps = supports(t.VERIFIERS[0].reference.engine_cls)
    assert {ReferenceLogprobs, StackReport, DecodeCodes} <= caps


def test_the_oracle_declares_the_sampled_decode_the_gate_grades():
    """A report gate against a greedy oracle would price every position under a
    distribution nothing sampled from. check_wiring refuses that pairing; this pins the
    values rather than only the shape."""
    from tasks.multimodal_tts import task as t
    ref = t.VERIFIERS[0].reference
    # The plain softmax: temperature 1 everywhere, no truncation, no penalty. Top-k 0
    # is what "do not truncate" is spelled as on both the talker and the sub-talker.
    assert ref.decode.temperature == 1.0
    assert (ref.top_k, ref.repetition_penalty) == (0, 1.0)
    assert (ref.subtalker_temperature, ref.subtalker_top_k) == (1.0, 0)
    assert ref.subtalker_sample is True


def test_the_end_of_speech_id_agrees_across_the_contract_and_the_task():
    """The servers report that position and the gate prices it. An id changed on one
    side alone is a wrong verdict rather than a missing one, so the task refuses to
    import rather than publish a contract it disagrees with."""
    from tasks.multimodal_tts import task as t
    assert t.CODEC_EOS_ID == t.CONTRACT.protocol.codec_eos_id == 2150


def test_multimodal_tts_verifier_wiring_checks_out():
    from hls.verify import check_wiring
    from tasks.multimodal_tts import task as t
    check_wiring(t.VERIFIERS, t.CONTRACT, t.WORKLOAD.slo)


def test_multimodal_tts_latency_scorer_reads_a_measured_baseline():
    from hls.scoring import SloLatencyScorer
    from tasks.multimodal_tts import task as t
    assert isinstance(t.SCORER, SloLatencyScorer)
    assert t.SCORER.scores_against_baseline()
    # None until the run loader injects the measured credit; no hand-typed constant.
    assert t.SCORER.baseline_log_credit is None


def test_multimodal_tts_drives_a_fixed_rate_against_a_normalizing_slo():
    """The scorer means log(1/r) over requests, so the workload has to offer one fixed
    rate and carry an SLO that defines r."""
    from hls.spec.workload import Poisson, RequestLatencySLO
    from tasks.multimodal_tts import task as t
    assert t.WORKLOAD.arrival == Poisson(rate_per_sec=t.OFFERED_RPS)
    assert isinstance(t.WORKLOAD.slo, RequestLatencySLO)
    assert t.WORKLOAD.slo.target_ms == t.TARGET_JCT_MS
    assert t.DATA_SOURCE.split.test.value == t.TEST_REQUESTS


def test_multimodal_tts_dataset_projects_columns_without_touching_hfdataset():
    from hls.spec.datasets import ColumnProjectedHFDataset
    from tasks.multimodal_tts import task as t
    assert isinstance(t.DATASET, ColumnProjectedHFDataset)
    assert t.DATASET.keep_columns == ["text_normalized"]


def test_the_committed_calibration_is_fresh_for_the_shipped_task():
    """The committed report calibration loads end to end for the task as it
    stands: its kind is the report gate's, its workload fingerprint is the task's, and its audit
    copy hashes to that fingerprint. A task edit that moves the fingerprint flips this test."""
    from hls.measured import CALIBRATION_FILE, defaults_path
    from hls.taskref import TaskRef
    from hls.verify.calibration import ReportCalibration, read_calibration
    from tasks.multimodal_tts import task as t

    path = defaults_path(TaskRef(module="tasks.multimodal_tts.task"), CALIBRATION_FILE)
    cal = read_calibration(path, t.TASK, kind="report")[0]
    assert isinstance(cal, ReportCalibration)
    assert cal.workload_fingerprint == "c521957c6543c87e6e5813c5131eca5c54e39c1899d23934528da99d3bd8d98e"
    assert cal.kappa == 0.0013954442780677843
    assert cal.outside_ceiling == 0.3442667436870069
