"""Unit tests for the multimodal speech foundations.

Covers the multimodal-asr pieces: the audio codec +
AudioPayload/ImagePayload Envelope round-trip, the Poisson arrival process, the WER
metric (RolloutMetric) + ReferenceRelativeVerifier wiring (via a fake
Generate engine + EngineAccessor), the JSON AudioTranscriptionProtocol codec, the
HFAudioDataset mount-mode renderer, and that the multimodal-asr TaskSpec constructs + wires.
"""

from __future__ import annotations
import json

import numpy as np
import pytest
from pydantic import ValidationError

from hls.core.audio import decode_wav_b64, duration_sec_of_b64, encode_wav_b64
from hls.core.capabilities import Generate
from hls.core.decode import Greedy
from hls.core.payloads import AudioPayload, Envelope, ImagePayload, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.core.verdict import Verdict
from hls.verify.base import EngineAccessor
from hls.verify.metrics.asr import WERMetric, word_error_rate


# ---- audio codec + payloads ----------------------------------------------

def test_audio_wav_roundtrip():
    a = np.sin(np.linspace(0, 6.28 * 220, 16_000)).astype(np.float32)
    b64 = encode_wav_b64(a, 16_000)
    arr, sr = decode_wav_b64(b64)
    assert sr == 16_000 and arr.shape[0] == 16_000
    assert abs(duration_sec_of_b64(b64) - 1.0) < 1e-6
    assert float(np.abs(arr - a).max()) < 1e-3


def test_audio_payload_envelope_json_roundtrip():
    b64 = encode_wav_b64(np.zeros(800, dtype=np.float32), 16_000)
    env = Envelope(request_id="r", contents=[AudioPayload(audio_b64=b64, sample_rate=16_000)])
    env2 = Envelope.model_validate(env.model_dump(mode="json"))
    ap = env2.first_of("audio")
    assert isinstance(ap, AudioPayload) and ap.sample_rate == 16_000 and ap.audio_b64 == b64


def test_image_payload_envelope_json_roundtrip():
    env = Envelope(request_id="r", contents=[ImagePayload(image_b64="QUJD", mime_type="image/png")])
    env2 = Envelope.model_validate(env.model_dump(mode="json"))
    ip = env2.first_of("image")
    assert isinstance(ip, ImagePayload) and ip.data_uri() == "data:image/png;base64,QUJD"


# ---- arrival process -----------------------------------------------------

def test_poisson_schedule_absolute_monotonic_and_rate():
    from hls.spec.workload import Poisson
    offs = Poisson(rate_per_sec=5.0).schedule(2000, seed=7)
    assert len(offs) == 2000 and all(offs[i] <= offs[i + 1] for i in range(len(offs) - 1))
    gaps = [offs[0]] + [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]
    assert abs((sum(gaps) / len(gaps)) - 1 / 5.0) < 0.05


def test_poisson_rejects_nonpositive_rate():
    from hls.spec.workload import Poisson
    with pytest.raises(ValidationError):
        Poisson(rate_per_sec=0.0)


# ---- ASR WER metric ------------------------------------------------------

def test_word_error_rate():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    assert abs(word_error_rate("the cat", "the cat sat") - 1 / 3) < 1e-9
    assert word_error_rate("", "the cat") == 1.0


def _text_env(t):
    return Envelope(request_id="r", contents=[TextPayload(text=t)])


def _audio_env(rid):
    return Envelope(request_id=rid, contents=[AudioPayload(
        audio_b64=encode_wav_b64(np.zeros(800, dtype=np.float32), 16_000), sample_rate=16_000)])


def _asr_event(rid, transcript, gt):
    return TraceEvent(request_id=rid, input=_audio_env(rid),
                      response=ArtifactResponse(output=_text_env(transcript)),
                      ground_truth=gt, t_send=0.0, t_recv=1.0)


def test_wer_metric_reduces_mean_accuracy():
    metric = WERMetric()
    evs = [_asr_event("a.0", "the cat sat on the mat", "the cat sat on the mat"),  # 1.0
           _asr_event("b.0", "the dog", "the cat sat")]                            # 1 - 2/3
    r = metric.score(evs)
    assert abs(r.value - (1.0 + (1 - 2 / 3)) / 2) < 1e-6
    assert r.per_event["a.0"].score == 1.0


def test_wer_metric_charges_a_runaway_its_full_error_rate():
    """Accuracy is ``1 - WER`` with no floor, as the brief states it. A reply that loops
    past its reference has a WER above 1 and its row scores below zero, so one runaway
    costs the run what it actually got wrong rather than at most one row."""
    metric = WERMetric()
    evs = [_asr_event("a.0", "cat cat cat cat", "cat"),       # WER 3 -> accuracy -2
           _asr_event("b.0", "the cat sat", "the cat sat")]   # accuracy 1
    r = metric.score(evs)
    assert r.per_event["a.0"].score == -2.0
    assert not r.per_event["a.0"].passed
    assert abs(r.value - (-2.0 + 1.0) / 2) < 1e-9


# ---- Pattern-C verifier via a fake Generate engine -----------------------

class _FakeOracle:
    """Stands in for a deployed Whisper engine: returns a canned transcript per
    request, so engine[Generate] is its bound async ``generate``."""

    def __init__(self, transcript="the cat sat on the mat"):
        self._t = transcript
        self.calls = 0

    async def generate(self, prompts):
        self.calls += 1
        return [Envelope(request_id=p.request_id, contents=[TextPayload(text=self._t)])
                for p in prompts]


async def test_reference_relative_verifier_with_wer_metric():
    from hls.reference.asr import WhisperReference
    from hls.verify import ReferenceRelativeVerifier
    # The reference field is only config here; the deployed engine is injected via the
    # EngineAccessor so no model loads.
    v = ReferenceRelativeVerifier(metric=WERMetric(),
                                  reference=WhisperReference(decode=Greedy()), tol=0.15)
    events = [_asr_event("a.0", "the cat sat on the mat", "the cat sat on the mat"),
              _asr_event("b.0", "the cat sat on the bat", "the cat sat on the mat")]
    oracle = _FakeOracle(transcript="the cat sat on the mat")
    engine = EngineAccessor(artifact_events=events,
                            cap_to_provider={Generate: oracle.generate})
    verdict = await v.verify(engine)
    assert isinstance(verdict, Verdict)
    assert oracle.calls == 1                       # ONE batched oracle pass
    # artifact mean accuracy ~ (1.0 + 0.8) / 2 = 0.9; oracle is exact (1.0); within tol=0.15
    assert verdict.passed
    assert verdict.diagnostics["reference_value"] == 1.0


# ---- JSON transcription protocol -----------------------------------------

def test_audio_transcription_protocol_json_roundtrip():
    from hls.spec.wire import AudioTranscriptionProtocol
    b64 = encode_wav_b64(np.zeros(1600, dtype=np.float32), 16_000)
    env = Envelope(request_id="r", contents=[AudioPayload(audio_b64=b64, sample_rate=16_000)],
                   metadata={"max_tokens": 64})
    proto = AudioTranscriptionProtocol(model="whisper", language="en")
    body = proto.encode_request(env)
    assert body["audio_b64"] == b64 and body["model"] == "whisper"
    assert body["language"] == "en" and body["max_tokens"] == 64
    out = proto.decode_response({"text": "hello", "token_ids": [1, 2]}, "r")
    tp = out.first_of("text")
    assert tp.text == "hello" and tp.tokens == [1, 2]


def test_audio_transcription_protocol_probe_and_errors():
    from hls.core.faults import ArtifactProtocolError
    from hls.spec.wire import AudioTranscriptionProtocol
    proto = AudioTranscriptionProtocol(model="whisper")
    probe = proto.probe_request()
    assert probe["audio_b64"] == "" and probe["max_tokens"] == 1
    with pytest.raises(ArtifactProtocolError):
        proto.decode_response({"no_text": 1}, "r")
    with pytest.raises(ArtifactProtocolError):
        proto.decode_response({"text": 123}, "r")


# ---- HFAudioDataset mount-mode renderer ----------------------------------

def test_hf_audio_dataset_items_from_mount(tmp_path):
    from hls.spec.datasets import SOURCE_INDEX, HFAudioDataset, Split
    b64 = encode_wav_b64(np.zeros(16_000, dtype=np.float32), 16_000)
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "rows.json").write_text(json.dumps(
        [{SOURCE_INDEX: 0, "audio_b64": b64, "sample_rate": 16_000,
          "text": "hello world"}]))
    ds = HFAudioDataset(path="x/y", source_split="test.clean")
    items = list(ds.items_at(tmp_path, Split.DEV))
    assert len(items) == 1
    it = items[0]
    ap = it.turns[0].first_of("audio")
    assert isinstance(ap, AudioPayload) and ap.audio_b64 == b64
    assert it.ground_truth_turns == ("hello world",)


# ---- multimodal-asr wiring ---------------------------------------------------------

def test_multimodal_asr_taskspec_constructs():
    from tasks.multimodal_asr import task as t
    assert t.TASK.id == "multimodal-asr" and t.CONTRACT.transport == "http"
    assert t.CONTRACT.input_modalities == frozenset({"audio"})
    assert t.TASK.data_source is not None
    v = t.VERIFIERS[0]
    assert type(v).__name__ == "ReferenceRelativeVerifier"
    assert Generate in __import__("hls.core.engine", fromlist=["supports"]).supports(
        v.reference.engine_cls)
