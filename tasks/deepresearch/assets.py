"""The BrowseComp-Plus tool bundle: what the corpus browser loads.

Built once per variant on the provisioning host, from three pinned sources: the
document corpus, the published Qwen3-Embedding-8B document vectors, and the encoder
checkpoint. ``full`` holds every document. ``dev`` holds the documents that belong to
the dev questions' relevance pools (gold, evidence and hard negatives) plus every
document that belongs to no question's pool, so it retrieves like the full corpus for
dev questions while carrying no test question's evidence.

Layout of a bundle directory:

    manifest.json   identity, variant, n_docs, sources
    encoder/        the encoder checkpoint snapshot
    docs.parquet    docid, url, title, text
    vectors.npy     float32 [n_docs, 4096], unit-norm, row i is docids[i]
    docids.json     list of docid strings
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from hls.spec.agent import AssetsRefused, ToolAssets
from tasks.deepresearch.datasets import BROWSECOMP_CANARY, _decrypt_browsecomp_string
from tasks.deepresearch.retriever import title_and_body

CORPUS_REPO = "Tevatron/browsecomp-plus-corpus"
CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
QUERIES_REPO = "Tevatron/browsecomp-plus"
QUERIES_REVISION = "144cff8e35b5eaef7e526346aa60774a9deb941f"
INDEX_REPO = "Tevatron/browsecomp-plus-indexes"
INDEX_REVISION = "b3f37f70c33829eb09d04784a54277a31871fd63"
INDEX_SUBDIR = "qwen3-embedding-8b"
INDEX_SHARDS = 4
ENCODER_ID = "Qwen/Qwen3-Embedding-8B"
ENCODER_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
# Bump on any change to how a bundle's contents are computed at build time (not
# just to a pinned source), so an already-built bundle from the old rule is not
# mistaken for a cache hit under the new one.
BUNDLE_FORMAT = 2


class BundleError(AssetsRefused):
    """A bundle could not be built as specified; nothing was written."""


def load_corpus(repo: str, revision: str, token: str | None) -> Iterator[tuple[str, str, str]]:
    from datasets import load_dataset
    ds = load_dataset(repo, revision=revision, split="train", token=token)
    for row in ds:
        yield str(row["docid"]), str(row["url"]), str(row["text"])


def load_index(repo: str, revision: str, subdir: str, token: str | None) -> tuple[np.ndarray, list[str]]:
    from huggingface_hub import hf_hub_download
    reps, ids = [], []
    for i in range(1, INDEX_SHARDS + 1):
        path = hf_hub_download(repo, f"{subdir}/corpus.shard{i}_of_{INDEX_SHARDS}.pkl",
                               repo_type="dataset", revision=revision, token=token)
        with open(path, "rb") as f:
            shard_reps, shard_ids = pickle.load(f)
        reps.append(np.asarray(shard_reps, dtype=np.float32))
        ids.extend(str(d) for d in shard_ids)
    return np.concatenate(reps, axis=0), ids


def load_query_pools(repo: str, revision: str, token: str | None) -> list[tuple[set[str], set[str]]]:
    """Per source row, in row order: (relevant docids, hard-negative docids)."""
    from datasets import load_dataset
    ds = load_dataset(repo, revision=revision, split="test", token=token)

    def _ids(row: dict, key: str) -> set[str]:
        return {_decrypt_browsecomp_string(d["docid"], BROWSECOMP_CANARY) for d in row[key]}

    pools = []
    for row in ds:
        pools.append((_ids(row, "gold_docs") | _ids(row, "evidence_docs"), _ids(row, "negative_docs")))
    return pools


def snapshot_encoder(model_id: str, revision: str, dest: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(model_id, revision=revision, local_dir=str(dest), token=token,
                      max_workers=16)


@dataclass(frozen=True)
class BrowseCompBundle(ToolAssets):
    # name/memory_mb/gpu are fields, not class attributes, so a smoke variant can
    # ``dataclasses.replace(..., gpu=False)`` for a single-GPU host.
    name: str = "browsecomp"
    memory_mb: int = 49152
    gpu: bool = True

    corpus_repo: str = CORPUS_REPO
    corpus_revision: str = CORPUS_REVISION
    queries_repo: str = QUERIES_REPO
    queries_revision: str = QUERIES_REVISION
    index_repo: str = INDEX_REPO
    index_revision: str = INDEX_REVISION
    index_subdir: str = INDEX_SUBDIR
    encoder_id: str = ENCODER_ID
    encoder_revision: str = ENCODER_REVISION

    def sources(self) -> dict[str, str]:
        return {"corpus": f"{self.corpus_repo}@{self.corpus_revision}",
                "queries": f"{self.queries_repo}@{self.queries_revision}",
                "index": f"{self.index_repo}@{self.index_revision}/{self.index_subdir}",
                "encoder": f"{self.encoder_id}@{self.encoder_revision}",
                "format": str(BUNDLE_FORMAT)}

    def hub_revisions(self) -> list[tuple[str, str]]:
        return [(self.corpus_repo, self.corpus_revision), (self.queries_repo, self.queries_revision),
                (self.index_repo, self.index_revision), (self.encoder_id, self.encoder_revision)]

    def identity(self) -> str:
        key = json.dumps(self.sources(), sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def build(self, variant: str, *, dev_row_ids: list[str], test_row_ids: list[str],
              out_dir: Path, hf_token: str | None = None) -> None:
        if variant not in ("dev", "full"):
            raise BundleError(f"unknown bundle variant {variant!r}")
        vectors, docids = load_index(self.index_repo, self.index_revision,
                                     self.index_subdir, hf_token)
        keep = self._dev_docids(dev_row_ids, test_row_ids, set(docids), hf_token) \
            if variant == "dev" else None
        tmp = out_dir.with_name(out_dir.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            self._write(tmp, variant, vectors, docids, keep, hf_token)
            os.replace(tmp, out_dir)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def _row_index(self, r, n_pools: int) -> int:
        """One row id as an index into the query pools list, or a named failure.

        A row id arrives as whatever the provisioning partition wrote; a non-integer
        id or one outside the pools list would otherwise fail as an opaque KeyError
        or IndexError deep in a set comprehension.
        """
        try:
            idx = int(r)
        except (TypeError, ValueError):
            raise BundleError(f"row id {r!r} is not an integer index into the query pools")
        if not (0 <= idx < n_pools):
            raise BundleError(f"row id {r!r} is out of range for {n_pools} query pool(s)")
        return idx

    def _dev_docids(self, dev_row_ids, test_row_ids, all_docids, hf_token) -> set[str]:
        pools = load_query_pools(self.queries_repo, self.queries_revision, hf_token)
        dev = {self._row_index(r, len(pools)) for r in dev_row_ids}
        test = {self._row_index(r, len(pools)) for r in test_row_ids}
        pooled = set().union(*(rel | neg for rel, neg in pools)) if pools else set()
        dev_pool = set().union(*(pools[i][0] | pools[i][1] for i in dev)) if dev else set()
        test_relevant = set().union(*(pools[i][0] for i in test)) if test else set()
        leaked = dev_pool & test_relevant
        if leaked:
            raise BundleError(
                f"dev bundle would carry test evidence: {len(leaked)} document(s) in a dev "
                f"question's pool are gold or evidence for a test question")
        return (dev_pool | (all_docids - pooled)) & all_docids

    def _write(self, dest: Path, variant: str, vectors, docids, keep, hf_token) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = [i for i, d in enumerate(docids) if keep is None or d in keep]
        kept_ids = [docids[i] for i in rows]
        kept_set = set(kept_ids)
        np.save(dest / "vectors.npy", np.ascontiguousarray(vectors[rows]))
        (dest / "docids.json").write_text(json.dumps(kept_ids))
        cols = {"docid": [], "url": [], "title": [], "text": []}
        for docid, url, text in load_corpus(self.corpus_repo, self.corpus_revision, hf_token):
            if docid in kept_set:
                cols["docid"].append(docid)
                cols["url"].append(url)
                cols["title"].append(title_and_body(text)[0])
                cols["text"].append(text)
        missing = kept_set - set(cols["docid"])
        if missing:
            raise BundleError(f"{len(missing)} indexed docid(s) have no corpus text")
        pq.write_table(pa.table(cols), dest / "docs.parquet")
        snapshot_encoder(self.encoder_id, self.encoder_revision, dest / "encoder", hf_token)
        (dest / "manifest.json").write_text(json.dumps({
            "identity": self.identity(), "variant": variant, "n_docs": len(kept_ids),
            "sources": self.sources()}, indent=2))
