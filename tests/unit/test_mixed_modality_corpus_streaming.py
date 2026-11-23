"""``MixedModalityDataset.corpus()`` must read its two arms lazily.

An omni row carries a whole image or a whole clip as base64, so a Python list of one
arm is among the largest objects provisioning handles. Building the merged corpus
with ``.to_list()`` on both arms would pay for every row in both, including
the tail the mix never reaches.

Both arms are Arrow datasets, memory-mapped and iterable a row at a time, and the mix
consumes them strictly in order, so nothing needs materializing. The merge's cost is
bounded by the rows the mix uses rather than by how big the arms happen to be.
"""
import pytest

from hls.spec.datasets import HFAudioDataset, HFVQADataset, MixedModalityDataset


class _ArmProbe:
    """Stands in for an arm's Arrow corpus, recording how it was consumed.

    ``rows_pulled`` counts the rows that reached Python by either route, so a
    ``to_list()`` charges the whole arm; ``to_list_calls`` counts the whole-arm
    materializations, which is what these tests forbid."""

    def __init__(self, rows):
        self._rows = rows
        self.rows_pulled = 0
        self.to_list_calls = 0

    def __iter__(self):
        for r in self._rows:
            self.rows_pulled += 1
            yield r

    def to_list(self):
        self.to_list_calls += 1
        self.rows_pulled += len(self._rows)
        return list(self._rows)

    def __len__(self):
        return len(self._rows)


@pytest.fixture
def arms(monkeypatch):
    """A 500-row vision arm and a 500-row audio arm, feeding a 10-row merge."""
    vision = HFVQADataset(path="v", source_split="validation")
    audio = HFAudioDataset(path="a", source_split="test")
    v_probe = _ArmProbe([{"image_b64": f"v{i}", "mime_type": "image/png",
                          "question": "q", "answer": f"a{i}"} for i in range(500)])
    a_probe = _ArmProbe([{"audio_b64": f"a{i}", "sample_rate": 16_000, "text": f"t{i}"}
                         for i in range(500)])
    monkeypatch.setattr(type(vision), "corpus", lambda self: v_probe)
    monkeypatch.setattr(type(audio), "corpus", lambda self: a_probe)
    ds = MixedModalityDataset(vision=vision, audio=audio, mix_ratio=0.5, corpus_size=10)
    return ds, v_probe, a_probe


def test_neither_arm_is_materialized(arms):
    ds, v_probe, a_probe = arms
    ds.corpus()
    assert (v_probe.to_list_calls, a_probe.to_list_calls) == (0, 0)


def test_an_arm_is_read_only_as_far_as_the_mix_consumes_it(arms):
    # 10 merged rows at a 0.5 mix is 5 from each arm, plus the one-row look-ahead the
    # exhaustion check needs. An arm of 500 must not be walked to the end.
    ds, v_probe, a_probe = arms
    ds.corpus()
    assert v_probe.rows_pulled <= 6 and a_probe.rows_pulled <= 6


def test_the_merged_corpus_is_unchanged_by_streaming(arms):
    # The mix is the same deterministic sequence whether the arms arrive as lists or
    # as a stream: same length, same modality order, same payloads.
    ds, _, _ = arms
    merged = ds.corpus().to_list()
    assert len(merged) == 10
    # The accumulator starts empty, so a 0.5 mix opens on the audio arm and alternates.
    assert [r["modality"] for r in merged] == ["audio", "image"] * 5
    assert [r["gt"] for r in merged[:4]] == ["t0", "a0", "t1", "a1"]


def test_an_arm_shorter_than_its_share_falls_back_to_the_other(monkeypatch):
    # The exhaustion path, which the look-ahead rewrite is the easiest thing to break:
    # two vision rows against a long audio arm still fills corpus_size.
    vision = HFVQADataset(path="v", source_split="validation")
    audio = HFAudioDataset(path="a", source_split="test")
    monkeypatch.setattr(type(vision), "corpus", lambda self: _ArmProbe(
        [{"image_b64": "v", "mime_type": "image/png", "question": "q", "answer": "a"}] * 2))
    monkeypatch.setattr(type(audio), "corpus", lambda self: _ArmProbe(
        [{"audio_b64": "a", "sample_rate": 16_000, "text": "t"}] * 20))
    ds = MixedModalityDataset(vision=vision, audio=audio, mix_ratio=0.5, corpus_size=10)
    mods = [r["modality"] for r in ds.corpus().to_list()]
    assert len(mods) == 10 and mods.count("image") == 2
