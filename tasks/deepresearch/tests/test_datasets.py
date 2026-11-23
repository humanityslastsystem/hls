"""Unit tests for the deepresearch task's dataset (BrowseComp)."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from hls.spec.datasets import SOURCE_INDEX, Split
from tasks.deepresearch.datasets import (
    BROWSECOMP, BROWSECOMP_CANARY, BrowseCompQADataset,
    _decrypt_browsecomp_string, _decrypt_browsecomp_value,
)


def _mount(tmp_path, rows, split="dev"):
    """Write rows where a tenant's mount would hold them, and return that directory
    (``items_at`` is passed a mount dir; nothing global is patched)."""
    (tmp_path / split).mkdir(parents=True, exist_ok=True)
    (tmp_path / split / "rows.json").write_text(json.dumps(rows))
    return tmp_path


def test_browsecomp_decrypt_roundtrip():
    plain = "Queen Arwa University"
    key = hashlib.sha256(BROWSECOMP_CANARY.encode()).digest()
    padded = key * (len(plain) // len(key)) + key[: len(plain) % len(key)]
    encrypted = base64.b64encode(bytes(a ^ b for a, b in zip(plain.encode(), padded))).decode()
    assert _decrypt_browsecomp_string(encrypted, BROWSECOMP_CANARY) == plain


def test_browsecomp_decrypt_skips_query_id():
    plain_q, plain_a = "Who?", "Paris"
    key = hashlib.sha256(BROWSECOMP_CANARY.encode()).digest()

    def _enc(text: str) -> str:
        padded = key * (len(text) // len(key)) + key[: len(text) % len(key)]
        return base64.b64encode(
            bytes(a ^ b for a, b in zip(text.encode(), padded))).decode()

    row = {"query_id": "42", "query": _enc(plain_q), "answer": _enc(plain_a)}
    out = _decrypt_browsecomp_value(row, BROWSECOMP_CANARY, skip_keys=frozenset({"query_id"}))
    assert out["query_id"] == "42"
    assert out["query"] == plain_q
    assert out["answer"] == plain_a


def test_browsecomp_seed_renders_decrypted_row(tmp_path):
    mount = _mount(tmp_path, [{
        "query": "Who founded HLS?",
        "answer": "RulinShao",
        "query_id": "1",
        "source_index": 0,
    }])
    item = next(BROWSECOMP.items_at(mount, Split.DEV))
    text = item.request.first_of("text").text
    assert "Question: Who founded HLS?" in text
    assert item.request.metadata["messages"][0]["role"] == "system"
    assert item.request.metadata["messages"][1] == {
        "role": "user",
        "content": "Question: Who founded HLS?\n",
    }
    assert item.ground_truth == ["RulinShao"]


def test_browsecomp_corpus_keeps_only_the_seed_fields(monkeypatch):
    """A source row also carries the relevance pools (gold, evidence and negative
    documents). Those build the tool bundle, so keeping them on the provisioned row
    would hand every answer's evidence list to whoever reads the mount."""
    key = hashlib.sha256(BROWSECOMP_CANARY.encode()).digest()

    def _enc(text: str) -> str:
        padded = key * (len(text) // len(key)) + key[: len(text) % len(key)]
        return base64.b64encode(
            bytes(a ^ b for a, b in zip(text.encode(), padded))).decode()

    source = [{"query_id": "1", "query": _enc("Who?"), "answer": _enc("Paris"),
               "gold_docs": [{"docid": _enc("d0")}], "negative_docs": []}]
    monkeypatch.setattr("datasets.load_dataset",
                        lambda *a, **k: [dict(r) for r in source])
    ds = BrowseCompQADataset(
        path="Tevatron/browsecomp-plus", revision="144cff8e35b5eaef7e526346aa60774a9deb941f",
        source_split="test", question_field="query", answer_field="answer")
    row = ds.corpus()[0]
    assert set(row) == {"query_id", "query", "answer", SOURCE_INDEX}
    assert row["query"] == "Who?" and row["answer"] == "Paris"
    assert row[SOURCE_INDEX] == 0


@pytest.mark.integration
def test_browsecomp_corpus_decrypts_from_hub():
    """Live HF fetch: first BrowseComp-Plus row decrypts to readable text."""
    ds = BrowseCompQADataset(
        path="Tevatron/browsecomp-plus", revision="144cff8e35b5eaef7e526346aa60774a9deb941f",
        source_split="test", question_field="query", answer_field="answer")
    row = ds.corpus()[0]
    assert len(row["query"]) > 20
    assert " " in row["query"]
    assert len(row["answer"]) > 0
    assert row["query_id"]  # opaque id preserved
    assert set(row) == {"query_id", "query", "answer", SOURCE_INDEX}


def test_the_dataset_is_browsecomp():
    assert isinstance(BROWSECOMP, BrowseCompQADataset)
