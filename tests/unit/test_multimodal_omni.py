"""Unit tests for the multimodal-omni pieces.

Covers core/image header sniffing + ImagePayload, the OmniChatProtocol multimodal
content-part encoding, the per-arm MixedModalityMatchMetric and the two
reference-relative gates it feeds, the HFVQADataset + MixedModalityDataset mount-mode
renderers, and that the multimodal-omni TaskSpec constructs + wires.
"""

from __future__ import annotations
import json
import struct

import numpy as np
import pytest

from hls.core.audio import encode_wav_b64
from hls.core.capabilities import Generate
from hls.core.faults import HarnessFault
from hls.core.image import image_payload_from_bytes, sniff_dimensions, sniff_mime
from hls.core.payloads import AudioPayload, Envelope, ImagePayload, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent


def _png_bytes(w=640, h=480):
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", w, h) + b"\x00" * 8


# ---- core/image ----------------------------------------------------------

def test_image_sniff_and_payload():
    raw = _png_bytes(640, 480)
    assert sniff_mime(raw) == "image/png"
    assert sniff_dimensions(raw) == (640, 480)
    p = image_payload_from_bytes(raw)
    assert isinstance(p, ImagePayload) and p.mime_type == "image/png"
    assert p.width == 640 and p.height == 480
    assert p.data_uri().startswith("data:image/png;base64,")


# ---- OmniChatProtocol ----------------------------------------------------

def test_omni_protocol_content_parts():
    from hls.spec.wire import OmniChatProtocol
    proto = OmniChatProtocol(model="omni")
    b64 = encode_wav_b64(np.zeros(800, dtype=np.float32), 16_000)
    env = Envelope(request_id="r", contents=[
        ImagePayload(image_b64="QUJD", mime_type="image/png"),
        AudioPayload(audio_b64=b64, sample_rate=16_000),
        TextPayload(text="what is this?")], metadata={"max_tokens": 64})
    body = proto.encode_request(env)
    parts = body["messages"][0]["content"]
    kinds = [p["type"] for p in parts]
    assert kinds == ["image_url", "input_audio", "text"]
    assert parts[0]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert parts[1]["input_audio"]["data"] == b64 and parts[1]["input_audio"]["format"] == "wav"
    assert body["max_tokens"] == 64


def test_omni_protocol_decode_inherited_text():
    """An omni reply needs no token ids: the multimodal stacks return text only, and
    no multimodal-omni gate reads token ids."""
    from hls.spec.wire import OmniChatProtocol
    proto = OmniChatProtocol(model="omni")
    out = proto.decode_response(
        {"choices": [{"message": {"content": "a cat"}, "finish_reason": "stop"}]}, "r")
    assert out.first_of("text").text == "a cat"


def test_openai_chat_protocol_still_requires_token_ids():
    """The token-faithfulness path is untouched: the plain OpenAI codec, which every
    token-level task uses, still faults a reply that omits token_ids."""
    from hls.core.faults import ArtifactProtocolError
    from hls.spec.wire import OmniChatProtocol, OpenAIChatProtocol
    assert OpenAIChatProtocol.emits_token_ids is True
    assert OmniChatProtocol.emits_token_ids is False
    proto = OpenAIChatProtocol(model="m")
    with pytest.raises(ArtifactProtocolError, match="token_ids"):
        proto.decode_response(
            {"choices": [{"message": {"content": "a cat"}, "finish_reason": "stop"}]}, "r")
    # ... and it still asks for them on the wire, while the omni codec does not.
    assert proto.encode_request(_text_env("hi"))["return_token_ids"] is True
    assert "return_token_ids" not in OmniChatProtocol(model="omni").encode_request(
        _text_env("hi"))


def test_omni_protocol_brief_renders():
    """body_schema/brief_section must not read a field the parent does not have."""
    from hls.spec.wire import OmniChatProtocol
    proto = OmniChatProtocol(model="omni")
    schema = proto.body_schema()
    assert "token_ids" not in schema["response"]["choices"][0]
    assert "return_token_ids" not in schema["request"]
    brief = proto.brief_section(path="/v1/chat/completions")
    assert "/v1/chat/completions" in brief and "token_ids" not in brief


# ---- MixedModalityMatchMetric --------------------------------------------

def _text_env(t):
    return Envelope(request_id="r", contents=[TextPayload(text=t)])


def _omni_event(rid, modality, pred, gt):
    return TraceEvent(
        request_id=rid,
        input=Envelope(request_id=rid, contents=[TextPayload(text="q")],
                       metadata={"input_modality": modality}),
        response=ArtifactResponse(output=_text_env(pred)),
        ground_truth=gt, t_send=0.0, t_recv=1.0)


def test_word_f1():
    from hls.verify.metrics.omni import word_f1
    assert word_f1("a cat", "a cat") == 1.0
    assert word_f1("", "a cat") == 0.0
    assert 0.0 < word_f1("a dog", "a cat") < 1.0


def test_mixed_modality_metric_reduces_over_its_own_arm_only():
    from hls.verify.metrics.omni import MixedModalityMatchMetric
    evs = [_omni_event("a.0", "audio", "the cat sat", "the cat sat"),     # 1-WER = 1.0
           _omni_event("i.0", "image", "blue car", "red car")]           # F1 = 0.5
    img = MixedModalityMatchMetric(arm="image").score(evs)
    aud = MixedModalityMatchMetric(arm="audio").score(evs)
    assert abs(img.value - 0.5) < 1e-9 and aud.value == 1.0
    # Both maps cover every event: the runner ANDs the verifiers' per-event maps and
    # reads a missing key as a failure.
    assert set(img.per_event) == set(aud.per_event) == {"a.0", "i.0"}
    assert img.per_event["a.0"].score is None and img.per_event["a.0"].passed
    assert aud.per_event["i.0"].score is None and aud.per_event["i.0"].passed


def test_audio_wer_measure_charges_a_runaway_its_full_error_rate():
    """The WER measure is ``1 - WER`` with no floor, the same as the ASR metric: a reply
    that loops past its reference scores below zero."""
    from hls.verify.metrics.omni import MixedModalityMatchMetric
    evs = [_omni_event("a.0", "audio", "cat cat cat cat", "cat")]   # WER 3
    r = MixedModalityMatchMetric(arm="audio", audio_measure="wer").score(evs)
    assert r.value == -2.0 and not r.per_event["a.0"].passed


def test_arm_with_no_events_raises_instead_of_passing_vacuously():
    from hls.verify.metrics.omni import MixedModalityMatchMetric
    image_only = [_omni_event("i.0", "image", "red car", "red car")]
    assert MixedModalityMatchMetric(arm="image").score(image_only).value == 1.0
    with pytest.raises(HarnessFault, match="none of them are audio-input"):
        MixedModalityMatchMetric(arm="audio").score(image_only)


# ---- per-arm reference-relative gates ------------------------------------

#: The vLLM-Omni baseline's mean over the 50/50 mix, and its audio arm's mean sentence-BLEU.
_BASELINE_MEAN = 0.566
_BASELINE_AUDIO_BLEU = 0.42


class _CannedGenerate:
    """A Generate provider that replays a fixed reference answer per request id."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    async def generate(self, prompts):
        return [Envelope(request_id=p.request_id,
                         contents=[TextPayload(text=self._answers[p.request_id])])
                for p in prompts]


def _gate(events, answers, *, arm, min_ratio, **metric_kwargs):
    """Run one per-arm ReferenceRelativeVerifier over ``events`` against ``answers``."""
    import asyncio
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference
    from hls.verify.base import EngineAccessor
    from hls.verify.metrics.omni import MixedModalityMatchMetric
    from hls.verify.verifiers import ReferenceRelativeVerifier
    v = ReferenceRelativeVerifier(
        metric=MixedModalityMatchMetric(arm=arm, **metric_kwargs),
        reference=HFReference(decode=Greedy()), min_ratio=min_ratio)
    accessor = EngineAccessor(
        artifact_events=events,
        cap_to_provider={Generate: _CannedGenerate(answers).generate})
    return asyncio.run(v.verify(accessor))


#: What a faithful reference answers, and the ground truth the events carry, so the
#: reference scores 1.0 on both arms and a min_ratio bar is easy to read off.
_ANSWERS = {"i.0": "red car", "i.1": "blue bike",
            "a.0": "the cat sat on the mat", "a.1": "a dog ran fast"}


def _mixed_events(image_preds, audio_preds):
    """Two image-input and two audio-input events with the given predictions."""
    return [_omni_event(rid, "image", pred, _ANSWERS[rid])
            for rid, pred in zip(("i.0", "i.1"), image_preds)] + \
           [_omni_event(rid, "audio", pred, _ANSWERS[rid])
            for rid, pred in zip(("a.0", "a.1"), audio_preds)]


def _both_gates(events):
    image = _gate(events, _ANSWERS, arm="image", min_ratio=0.90)
    audio = _gate(events, _ANSWERS, arm="audio", min_ratio=0.85,
                  audio_measure="bleu", pass_threshold=0.15)
    return image, audio


def test_per_arm_gates_both_pass_when_both_arms_match_the_reference():
    image, audio = _both_gates(_mixed_events(
        ("red car", "blue bike"), ("the cat sat on the mat", "a dog ran fast")))
    assert image.passed and audio.passed
    assert image.diagnostics["artifact_value"] == image.diagnostics["reference_value"] == 1.0
    assert audio.diagnostics["artifact_value"] == audio.diagnostics["reference_value"] == 1.0


def test_abandoned_audio_arm_fails_its_own_gate():
    """A perfect image arm does not pay for an empty audio arm. A single aggregate
    floor would average the same rollout at 0.5 >= 0.35."""
    events = _mixed_events(("red car", "blue bike"), ("", ""))
    image, audio = _both_gates(events)
    assert image.passed and image.diagnostics["artifact_value"] == 1.0
    assert not audio.passed and audio.diagnostics["artifact_value"] == 0.0
    # The image gate scored only its own two events; the audio ones carry no score.
    assert [image.event_diagnostics[rid].get("graded", True) for rid in ("a.0", "a.1")] \
        == [False, False]
    # The aggregate a single floor would gate this rollout on.
    assert (1.0 + 1.0 + 0.0 + 0.0) / 4 >= 0.35


def test_abandoned_image_arm_fails_its_own_gate():
    events = _mixed_events(("", ""), ("the cat sat on the mat", "a dog ran fast"))
    image, audio = _both_gates(events)
    assert audio.passed and audio.diagnostics["artifact_value"] == 1.0
    assert not image.passed and image.diagnostics["artifact_value"] == 0.0


def test_per_arm_gate_grades_only_its_own_arms_events():
    """Each gate marks the other arm's events passed, so the runner's AND across the
    two verifiers leaves every event graded by exactly the gate that owns it."""
    events = _mixed_events(("red car", "blue bike"), ("", ""))
    image, audio = _both_gates(events)
    assert image.per_event == {"i.0": True, "i.1": True, "a.0": True, "a.1": True}
    assert audio.per_event == {"i.0": True, "i.1": True, "a.0": False, "a.1": False}


def test_documented_subsidy_arithmetic_holds():
    """At the baseline's figures a single 0.35 aggregate floor would pass a rollout whose audio
    arm answers nothing, which is why each arm has its own gate."""
    from hls.verify.metrics.omni import sentence_bleu
    assert sentence_bleu("", "the cat sat on the mat") == 0.0
    image_f1 = (_BASELINE_MEAN - 0.5 * _BASELINE_AUDIO_BLEU) / 0.5
    assert abs(image_f1 - 0.712) < 5e-4
    assert 0.5 * image_f1 + 0.5 * 0.0 >= 0.35


# ---- HFVQADataset + MixedModalityDataset mount renderers -----------------

def test_hf_vqa_dataset_items_from_mount(tmp_path):
    from hls.spec.datasets import SOURCE_INDEX, HFVQADataset, Split
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "rows.json").write_text(json.dumps(
        [{SOURCE_INDEX: 0, "image_b64": "QUJD", "mime_type": "image/png",
          "question": "what?", "answer": "cat"}]))
    ds = HFVQADataset(path="x/y", source_split="validation",
                      question_template="Answer: {question}")
    items = list(ds.items_at(tmp_path, Split.DEV))
    assert len(items) == 1
    it = items[0]
    assert isinstance(it.turns[0].first_of("image"), ImagePayload)
    assert it.turns[0].first_of("text").text == "Answer: what?"
    assert it.ground_truth_turns == ("cat",) and it.metadata["input_modality"] == "image"


def test_mixed_modality_dataset_items_from_mount(tmp_path):
    from hls.spec.datasets import (
        SOURCE_INDEX, HFAudioDataset, HFVQADataset, MixedModalityDataset, Split)
    b64 = encode_wav_b64(np.zeros(1600, dtype=np.float32), 16_000)
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "rows.json").write_text(json.dumps([
        {SOURCE_INDEX: 0, "modality": "image", "image_b64": "QUJD",
         "mime_type": "image/png", "audio_b64": "", "sample_rate": 0,
         "prompt": "what?", "gt": "cat"},
        {SOURCE_INDEX: 1, "modality": "audio", "image_b64": "", "mime_type": "",
         "audio_b64": b64, "sample_rate": 16_000, "prompt": "", "gt": "hello"}]))
    ds = MixedModalityDataset(
        vision=HFVQADataset(path="v", source_split="validation"),
        audio=HFAudioDataset(path="a", source_split="test"), mix_ratio=0.5)
    items = list(ds.items_at(tmp_path, Split.DEV))
    assert len(items) == 2
    img_it, aud_it = items
    assert isinstance(img_it.turns[0].first_of("image"), ImagePayload)
    assert img_it.metadata["input_modality"] == "image" and img_it.ground_truth_turns == ("cat",)
    assert isinstance(aud_it.turns[0].first_of("audio"), AudioPayload)
    assert aud_it.metadata["input_modality"] == "audio" and aud_it.ground_truth_turns == ("hello",)


def test_mixed_modality_interleave_unit():
    from hls.spec.datasets import HFAudioDataset, HFVQADataset, MixedModalityDataset
    ds = MixedModalityDataset(
        vision=HFVQADataset(path="v", source_split="validation"),
        audio=HFAudioDataset(path="a", source_split="test"), mix_ratio=0.5, corpus_size=4)
    v = [{"image_b64": f"v{i}", "mime_type": "image/png", "question": "q", "answer": "a"}
         for i in range(3)]
    a = [{"audio_b64": f"a{i}", "sample_rate": 16_000, "text": "t"} for i in range(3)]
    merged = ds._interleave(v, a)
    assert len(merged) == 4
    mods = [r["modality"] for r in merged]
    assert mods.count("image") == 2 and mods.count("audio") == 2   # 0.5 mix


# ---- multimodal-omni wiring ---------------------------------------------------------

def test_multimodal_omni_taskspec_constructs():
    from tasks.multimodal_omni import task as t
    assert t.TASK.id == "multimodal-omni"
    assert t.CONTRACT.input_modalities == frozenset({"image", "audio", "text"})
    assert type(t.DATASET).__name__ == "MixedModalityDataset"
    # One reference-relative gate per arm, both required.
    assert [type(v).__name__ for v in t.VERIFIERS] == ["ReferenceRelativeVerifier"] * 2
    assert [v.metric.arm for v in t.VERIFIERS] == ["image", "audio"]
    assert all(type(v.reference).__name__ == "Qwen3OmniReference" for v in t.VERIFIERS)
    assert all(v.min_ratio is not None for v in t.VERIFIERS)
    # Latency scoring: one fixed-rate open-loop drive, and the per-request SLO the
    # scorer reads its r = JCT/target from. A ramp or an aggregate-gating SLO would
    # leave the scorer with no per-request ratio to take the geometric mean of.
    from hls.scoring import SloLatencyScorer
    from hls.spec.workload import Poisson, RequestLatencySLO
    assert isinstance(t.SCORER, SloLatencyScorer)
    assert t.SCORER.baseline_log_credit is None      # the loader injects it
    assert t.WORKLOAD.arrival == Poisson(rate_per_sec=t.OFFERED_RPS)
    assert t.WORKLOAD.slo == RequestLatencySLO(target_ms=t.TARGET_JCT_MS)


def test_multimodal_omni_corpus_holds_both_splits():
    """partition_by_hash refuses dev + test > corpus, so the merged corpus must cover
    the whole drive."""
    from tasks.multimodal_omni import task as t
    n_dev = t.DATA_SOURCE.split.dev.value
    n_test = t.DATA_SOURCE.split.test.value
    assert n_test == t.TEST_REQUESTS
    assert n_dev + n_test <= t.DATASET.corpus_size
    # The merged corpus can only be as large as the two arm pools it interleaves, and at
    # a 50/50 mix it draws evenly, so neither pool alone may be the binding limit.
    assert t.DATASET.corpus_size <= t._VISION.corpus_size + t._AUDIO.corpus_size
    assert min(t._VISION.corpus_size, t._AUDIO.corpus_size) >= t.DATASET.corpus_size // 2


def test_the_oracle_is_cached():
    """Answering the whole split costs about 15 to 33 min cold, which only pays once: the
    answers are persisted beside the task and reused by every later run and repeat."""
    from tasks.multimodal_omni import task as t

    assert t.VERIFIERS, "multimodal_omni declares no verifiers"
    for v in t.VERIFIERS:
        assert v.reference.cache, (
            f"{type(v).__name__} with metric {getattr(v.metric, 'arm', '?')} would "
            "re-answer the whole corpus on every run")


# ---- S2TT (FLEURS fr->en) arm ----------------------------------------------

def test_sentence_bleu_scales():
    from hls.verify.metrics.omni import sentence_bleu
    assert abs(sentence_bleu("the cat sat on the mat", "the cat sat on the mat") - 1.0) < 1e-9
    assert sentence_bleu("", "the cat") == 0.0
    assert sentence_bleu("completely different words here", "the cat sat on the mat") == 0.0
    partial = sentence_bleu("the cat sat on a mat", "the cat sat on the mat")
    assert 0.2 < partial < 1.0
    # brevity penalty: a bare prefix scores below the full sentence
    assert sentence_bleu("the cat", "the cat sat on the mat") < partial


def test_mixed_metric_bleu_measure_and_per_arm_threshold():
    from hls.verify.metrics.omni import MixedModalityMatchMetric
    m = MixedModalityMatchMetric(arm="audio", audio_measure="bleu", pass_threshold=0.15)
    ev = _omni_event("a", "audio", pred="the cat sat on a mat", gt="the cat sat on the mat")
    pv = m.score_one(ev)
    assert pv.diagnostics["modality"] == "audio" and "bleu" in pv.diagnostics
    assert pv.passed is True                     # BLEU ~0.5 >= 0.15
    m_wer = MixedModalityMatchMetric(arm="audio")  # default stays the ASR shape
    assert "wer" in m_wer.score_one(ev).diagnostics


def test_fleurs_s2tt_items_render_instruction(tmp_path):
    import json as _json
    from hls.spec import datasets as D
    from hls.spec.datasets import FleursS2TTDataset
    from hls.core.audio import encode_wav_b64
    import numpy as np
    (tmp_path / "test").mkdir()
    b64 = encode_wav_b64(np.zeros(1600, dtype=np.float32), 16_000)
    (tmp_path / "test" / "rows.json").write_text(_json.dumps(
        [{D.SOURCE_INDEX: 0, "audio_b64": b64, "sample_rate": 16000,
          "text": "Hello world."}]))
    ds = FleursS2TTDataset()
    items = list(ds.items_at(tmp_path, D.Split.TEST))
    assert len(items) == 1
    it = items[0]
    assert it.ground_truth_turns == ("Hello world.",)
    mods = it.turns[0].modalities()
    assert "audio" in mods and "text" in mods    # audio + translate instruction
    txt = it.turns[0].first_of("text")
    assert "Translate" in txt.text
