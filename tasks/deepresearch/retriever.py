"""Corpus retrieval for deepresearch: BrowseComp-Plus documents ranked by Qwen3-Embedding-8B.

Loads a tool bundle (layout in ``assets.py``) and answers two questions: which
documents best match a query, and what a document says. The query encoder runs on
the GPU when one has room, else on CPU. Ranking is an exhaustive inner product over
the stored unit-norm vectors, the computation behind the benchmark's flat index.

Concurrent searches from one process are encoded in a single batch by a worker
thread, so eight in-flight trajectories cost about one forward pass per round.

No ``hls`` imports: the build sandbox runs this file without the harness.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_BUNDLE_DIR = "/data/tools/browsecomp"
BUNDLE_DIR_ENV = "HLS_TOOLS_DIR"
DEVICE_ENV = "HLS_RETRIEVER_DEVICE"
MIN_FREE_GPU_BYTES = 40 * 2**30
QUERY_PREFIX = ("Instruct: Given a web search query, retrieve relevant passages that "
                "answer the query\nQuery: ")
QUERY_MAX_LENGTH = 8192
SNIPPET_CHARS = 300
TITLE_CHARS = 120
ENCODER_MAX_THREADS = 16
_WS = re.compile(r"\s+")
_TITLE_KEY = re.compile(r"^\s*title\s*:\s*(.*)$")


@dataclass(frozen=True)
class Hit:
    docid: str
    url: str
    title: str
    snippet: str
    score: float


@dataclass(frozen=True)
class Doc:
    docid: str
    url: str
    title: str
    text: str


def bundle_dir() -> Path:
    return Path(os.environ.get(BUNDLE_DIR_ENV, DEFAULT_BUNDLE_DIR))


def resolve_device(requested: str | None = None) -> str:
    """``cuda`` when a GPU has room for the encoder beside whatever else holds it,
    else ``cpu``. An explicit ``cuda``/``cpu`` (argument or env) wins."""
    choice = requested or os.environ.get(DEVICE_ENV) or "auto"
    if choice != "auto":
        return choice
    import torch
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info(0)
        if free >= MIN_FREE_GPU_BYTES:
            return "cuda"
    return "cpu"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def title_and_body(text: str) -> tuple[str, str]:
    """The first non-empty line is the title, unless it opens a YAML front-matter
    block (a line that is exactly ``---``): then the title is that block's
    ``title:`` key and the body starts after the closing ``---``. Falls back to
    the plain first-line rule when there is no front matter, no title key, or no
    closing marker, so a malformed block behaves exactly as an ordinary line."""
    lines = text.split("\n")
    start = next((i for i, line in enumerate(lines) if line.strip()), None)
    if start is None:
        return "", ""
    if lines[start].strip() == "---":
        title = None
        close = None
        for j in range(start + 1, len(lines)):
            if lines[j].strip() == "---":
                close = j
                break
            match = _TITLE_KEY.match(lines[j])
            if match and title is None:
                title = _unquote(match.group(1))
        if close is not None and title:
            body = "\n".join(lines[close + 1:]).lstrip("\n")
            return title[:TITLE_CHARS], body
    return lines[start].strip()[:TITLE_CHARS], "\n".join(lines[start + 1:]).lstrip("\n")


def snippet(body: str, limit: int = SNIPPET_CHARS) -> str:
    return _WS.sub(" ", body).strip()[:limit]


def encoder_threads() -> int:
    """CPU threads for the query encoder, capped so a many-core box does not
    thrash on a single query. Uses the process's scheduling affinity when the
    platform exposes it, since that respects a cgroup or Slurm CPU limit that
    ``os.cpu_count`` does not see; falls back to ``os.cpu_count`` otherwise."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return min(ENCODER_MAX_THREADS, n)


class QueryEncoder:
    """Qwen3-Embedding query side: instruction prefix, last-token pooling, L2 norm."""

    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._tok = None
        self._model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        if self.device == "cpu":
            torch.set_num_threads(encoder_threads())
        self._tok = AutoTokenizer.from_pretrained(self.model_dir, padding_side="left")
        self._model = AutoModel.from_pretrained(self.model_dir, dtype=torch.bfloat16)
        self._model.to(self.device).eval()

    def encode(self, queries: list[str]) -> np.ndarray:
        import torch
        batch = self._tok([QUERY_PREFIX + q for q in queries], padding=True,
                          truncation=True, max_length=QUERY_MAX_LENGTH,
                          return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self._model(**batch).last_hidden_state[:, -1]
        return torch.nn.functional.normalize(hidden.float(), dim=-1).cpu().numpy()


class _EncodeQueue:
    """One worker thread that encodes every query queued since its last pass."""

    def __init__(self, encoder) -> None:
        self._encoder = encoder
        self._pending: list[tuple[str, asyncio.AbstractEventLoop, asyncio.Future]] = []
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._worker, name="retriever-encode",
                                        daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()

    async def encode(self, query: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        with self._cv:
            self._pending.append((query, loop, fut))
            self._cv.notify()
        return await fut

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait()
                if self._stop and not self._pending:
                    return
                batch, self._pending = self._pending, []
            try:
                vectors = self._encoder.encode([q for q, _, _ in batch])
                for (_, loop, fut), vec in zip(batch, vectors):
                    loop.call_soon_threadsafe(_resolve, fut, vec)
            except Exception as exc:  # surface to every waiter, keep the thread alive
                for _, loop, fut in batch:
                    loop.call_soon_threadsafe(_reject, fut, exc)


def _resolve(fut: asyncio.Future, value) -> None:
    if not fut.done():
        fut.set_result(value)


def _reject(fut: asyncio.Future, exc: BaseException) -> None:
    if not fut.done():
        fut.set_exception(exc)


class CorpusRetriever:
    """Search and read the documents of one bundle."""

    def __init__(self, bundle: Path | None = None, *, device: str | None = None,
                 encoder=None) -> None:
        self.bundle = Path(bundle) if bundle is not None else bundle_dir()
        self._device_request = device
        self.device: str | None = None
        self._encoder = encoder
        self._queue: _EncodeQueue | None = None
        self._manifest: dict = {}
        self._docids: list[str] = []
        self._vectors = None          # np.ndarray on cpu, torch.Tensor on cuda
        self._docs: dict[str, tuple[str, str, str]] = {}
        self._by_url: dict[str, str] = {}

    @property
    def loaded(self) -> bool:
        return self._vectors is not None

    @property
    def identity(self) -> str:
        return str(self._manifest.get("identity", ""))

    @property
    def n_docs(self) -> int:
        return len(self._docids)

    def load(self) -> None:
        """Read the bundle's manifest, vectors, and documents, then start the encoder."""
        if self.loaded:
            return
        if not (self.bundle / "manifest.json").exists():
            raise FileNotFoundError(
                f"no tool bundle at {self.bundle}: expected manifest.json, docids.json, "
                f"vectors.npy, docs.parquet, encoder/ (set {BUNDLE_DIR_ENV} to override)")
        self._manifest = json.loads((self.bundle / "manifest.json").read_text())
        self._docids = json.loads((self.bundle / "docids.json").read_text())
        vectors = np.load(self.bundle / "vectors.npy", mmap_mode="r")
        self.device = resolve_device(self._device_request)
        if self.device == "cuda":
            import torch
            self._vectors = torch.from_numpy(np.ascontiguousarray(vectors)).to("cuda")
        else:
            self._vectors = np.ascontiguousarray(vectors)
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.bundle / "docs.parquet")
        for batch in pf.iter_batches(columns=["docid", "url", "title", "text"]):
            for docid, url, title, text in zip(
                    batch.column("docid").to_pylist(), batch.column("url").to_pylist(),
                    batch.column("title").to_pylist(), batch.column("text").to_pylist()):
                self._docs[docid] = (url, title, text)
                # Duplicate urls resolve first-write-wins.
                self._by_url.setdefault(url, docid)
        if self._encoder is None:
            self._encoder = QueryEncoder(self.bundle / "encoder", self.device)
        self._encoder.load()
        self._queue = _EncodeQueue(self._encoder)
        self._queue.start()

    def close(self) -> None:
        """Stop the encoder worker thread."""
        if self._queue is not None:
            self._queue.stop()

    async def search(self, query: str, k: int = 10) -> list[Hit]:
        """Encode the query through the shared batch queue and return its top-k hits."""
        if not self.loaded:
            self.load()
        vec = await self._queue.encode(query)
        return self._hits(vec, k)

    def _hits(self, vec: np.ndarray, k: int) -> list[Hit]:
        if self.n_docs == 0:
            return []
        k = max(1, min(int(k), self.n_docs))
        if self.device == "cuda":
            import torch
            scores = self._vectors @ torch.from_numpy(vec).to("cuda")
            top = torch.topk(scores, k)
            order = list(zip(top.indices.tolist(), top.values.tolist()))
        else:
            scores = self._vectors @ vec
            idx = np.argpartition(-scores, k - 1)[:k]
            order = sorted(((int(i), float(scores[i])) for i in idx), key=lambda t: -t[1])
        hits = []
        for i, score in order:
            docid = self._docids[i]
            url, title, text = self._docs[docid]
            hits.append(Hit(docid=docid, url=url, title=title,
                            snippet=snippet(title_and_body(text)[1]), score=score))
        return hits

    def document(self, docid: str) -> Doc | None:
        """Look up one document by id, or None if it is not in the bundle."""
        entry = self._docs.get(str(docid))
        if entry is None:
            return None
        url, title, text = entry
        return Doc(docid=str(docid), url=url, title=title, text=text)

    def by_url(self, url: str) -> Doc | None:
        """Look up one document by url, or None if it is not in the bundle."""
        docid = self._by_url.get(url)
        return self.document(docid) if docid is not None else None
