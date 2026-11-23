"""CorpusRetriever over a tiny synthetic bundle; no model, no network."""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tasks.deepresearch import retriever as r

D = 8


class _StubEncoder:
    """Maps a query to a one-hot vector by the digit it contains."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def load(self):
        pass

    def encode(self, queries):
        self.calls.append(list(queries))
        out = np.zeros((len(queries), D), dtype=np.float32)
        for i, q in enumerate(queries):
            out[i, int(q.strip()[-1])] = 1.0
        return out


def _bundle(tmp_path, n=6):
    docids = [str(100 + i) for i in range(n)]
    vectors = np.eye(D, dtype=np.float32)[:n]
    np.save(tmp_path / "vectors.npy", vectors)
    (tmp_path / "docids.json").write_text(json.dumps(docids))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"identity": "id-test", "variant": "dev", "n_docs": n}))
    table = pa.table({
        "docid": docids,
        "url": [f"https://example.org/{i}" for i in range(n)],
        "title": [f"Doc {i}" for i in range(n)],
        "text": [f"Doc {i}\nbody line one of doc {i}\nline two" for i in range(n)],
    })
    pq.write_table(table, tmp_path / "docs.parquet")
    (tmp_path / "encoder").mkdir()
    return tmp_path


def _loaded(tmp_path):
    enc = _StubEncoder()
    ret = r.CorpusRetriever(_bundle(tmp_path), device="cpu", encoder=enc)
    ret.load()
    return ret, enc


def test_title_and_body_and_snippet():
    title, body = r.title_and_body("\n  A Long Title  \n\nfirst para\nsecond para\n")
    assert title == "A Long Title" and body.startswith("first para")
    assert r.snippet("a  b\n\nc  d", limit=100) == "a b c d"
    assert len(r.snippet("x" * 400, limit=10)) == 10
    assert r.title_and_body("")[0] == ""


def test_title_and_body_reads_the_title_key_from_front_matter():
    text = ("---\ntitle: 7 Sports of Ancient Greece\ndate: 2022-04-06\n---\n"
            "Lunt cites the example of ...\n")
    title, body = r.title_and_body(text)
    assert title == "7 Sports of Ancient Greece"
    assert body == "Lunt cites the example of ...\n"


def test_title_and_body_front_matter_title_key_not_first():
    text = ("---\ntitle: MovieWeb\nauthor: Eric Janise\ndate: 2023-07-31\n---\n"
            "The 1980s was ...\n")
    title, body = r.title_and_body(text)
    assert title == "MovieWeb"
    assert body == "The 1980s was ...\n"


def test_title_and_body_front_matter_quoted_title_is_unquoted():
    text = '---\ntitle: "Quoted Title"\n---\nbody text\n'
    assert r.title_and_body(text)[0] == "Quoted Title"


def test_title_and_body_front_matter_title_is_capped():
    text = "---\ntitle: " + ("x" * 200) + "\n---\nbody\n"
    title, _ = r.title_and_body(text)
    assert len(title) == r.TITLE_CHARS


def test_title_and_body_front_matter_without_title_key_falls_back():
    text = "---\ndate: 2022-04-06\n---\nbody text\n"
    title, body = r.title_and_body(text)
    assert title == "---"
    assert body.startswith("date: 2022-04-06")


def test_title_and_body_unterminated_front_matter_falls_back():
    text = "---\ntitle: Foo\nbody without a closing marker\n"
    title, body = r.title_and_body(text)
    assert title == "---"
    assert body.startswith("title: Foo")


def test_title_and_body_without_front_matter_is_unaffected():
    assert r.title_and_body("Plain Title\nplain body\n") == ("Plain Title", "plain body\n")


def test_encoder_threads_caps_at_max_but_respects_affinity(monkeypatch):
    monkeypatch.setattr(r.os, "sched_getaffinity", lambda pid: set(range(4)))
    assert r.encoder_threads() == 4
    monkeypatch.setattr(r.os, "sched_getaffinity", lambda pid: set(range(112)))
    assert r.encoder_threads() == r.ENCODER_MAX_THREADS == 16


def test_encoder_threads_falls_back_to_cpu_count_without_affinity(monkeypatch):
    monkeypatch.delattr(r.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(r.os, "cpu_count", lambda: 3)
    assert r.encoder_threads() == 3


def test_search_ranks_by_inner_product_and_returns_hits(tmp_path):
    ret, enc = _loaded(tmp_path)
    hits = asyncio.run(ret.search("query 3", k=2))
    ret.close()
    assert len(hits) == 2 and hits[0].docid == "103"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].title == "Doc 3" and hits[0].url.endswith("/3")
    assert hits[0].snippet.startswith("body line one of doc 3")
    assert enc.calls == [["query 3"]]


def test_search_on_empty_bundle_returns_no_hits(tmp_path):
    enc = _StubEncoder()
    ret = r.CorpusRetriever(_bundle(tmp_path, n=0), device="cpu", encoder=enc)
    ret.load()
    assert ret.n_docs == 0
    assert asyncio.run(ret.search("query 0", k=5)) == []
    ret.close()


def test_document_and_by_url(tmp_path):
    ret, _ = _loaded(tmp_path)
    doc = ret.document("104")
    assert doc.title == "Doc 4" and "line two" in doc.text
    assert ret.by_url("https://example.org/4").docid == "104"
    assert ret.document("nope") is None and ret.by_url("https://x/") is None


def test_concurrent_searches_are_encoded_in_one_batch(tmp_path):
    ret, enc = _loaded(tmp_path)

    async def go():
        return await asyncio.gather(*(ret.search(f"q {i}", k=1) for i in range(4)))

    results = asyncio.run(go())
    assert [hs[0].docid for hs in results] == ["100", "101", "102", "103"]
    assert sum(len(c) for c in enc.calls) == 4
    ret.close()


def test_encode_queue_batches_queued_queries():
    enc = _StubEncoder()
    queue = r._EncodeQueue(enc)

    async def go():
        tasks = [asyncio.ensure_future(queue.encode(f"q {i}")) for i in range(3)]
        for _ in range(3):
            await asyncio.sleep(0)  # let each coroutine register before the worker runs
        queue.start()
        return await asyncio.gather(*tasks)

    results = asyncio.run(go())
    assert len(results) == 3
    assert enc.calls == [["q 0", "q 1", "q 2"]]
    queue.stop()


def test_identity_and_bundle_dir_env(tmp_path, monkeypatch):
    ret, _ = _loaded(tmp_path)
    assert ret.identity == "id-test" and ret.n_docs == 6
    monkeypatch.setenv(r.BUNDLE_DIR_ENV, str(tmp_path))
    assert r.bundle_dir() == tmp_path


def test_resolve_device_honours_override(monkeypatch):
    monkeypatch.setenv(r.DEVICE_ENV, "cpu")
    assert r.resolve_device() == "cpu"
    assert r.resolve_device("cuda") == "cuda"
