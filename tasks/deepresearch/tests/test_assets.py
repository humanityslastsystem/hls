"""BrowseCompBundle builds dev/full bundles from monkeypatched loaders."""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from tasks.deepresearch import assets as a
from tasks.deepresearch.datasets import BROWSECOMP

# 8 docs: d0..d7. Row pools (relevant, negative), rows are source positions 0..3:
#   row0: rel {d0}, neg {d1}      row1: rel {d2}, neg {d3}
#   row2: rel {d4}, neg {d5}      row3: rel {d6}, neg {}
# d7 belongs to no pool.
_POOLS = [({"d0"}, {"d1"}), ({"d2"}, {"d3"}), ({"d4"}, {"d5"}), ({"d6"}, set())]


def _patch(monkeypatch, pools=_POOLS):
    docs = [(f"d{i}", f"https://x/{i}", f"Title {i}\nbody {i}") for i in range(8)]
    vectors = np.eye(8, dtype=np.float32)
    monkeypatch.setattr(a, "load_corpus", lambda repo, rev, token: iter(docs))
    monkeypatch.setattr(a, "load_index", lambda repo, rev, sub, token: (vectors, [d[0] for d in docs]))
    monkeypatch.setattr(a, "load_query_pools", lambda repo, rev, token: list(pools))

    def fake_snapshot(model_id, revision, dest, token):
        dest.mkdir()
        (dest / "config.json").write_text("{}")

    monkeypatch.setattr(a, "snapshot_encoder", fake_snapshot)


def test_the_bundle_declares_every_hub_source_it_builds_from_pinned():
    from hls.provision.model import is_revision
    from hls.spec.agent import AssetsRefused
    from hls.spec.hub import declared_hub_revisions

    bundle = a.BrowseCompBundle()
    declared = declared_hub_revisions(bundle, reads="build")
    assert declared == [(a.CORPUS_REPO, a.CORPUS_REVISION), (a.QUERIES_REPO, a.QUERIES_REVISION),
                        (a.INDEX_REPO, a.INDEX_REVISION), (a.ENCODER_ID, a.ENCODER_REVISION)]
    assert all(is_revision(revision) for _, revision in declared)
    # Each is one of the sources identity() digests.
    assert all(any(s.startswith(f"{repo}@{revision}") for s in bundle.sources().values()) for repo, revision in declared)
    assert issubclass(a.BundleError, AssetsRefused)


def test_full_bundle_has_every_document(tmp_path, monkeypatch):
    _patch(monkeypatch)
    out = tmp_path / "full"
    a.BrowseCompBundle().build("full", dev_row_ids=["0", "1"], test_row_ids=["2"], out_dir=out)
    assert json.loads((out / "docids.json").read_text()) == [f"d{i}" for i in range(8)]
    assert np.load(out / "vectors.npy").shape == (8, 8)
    table = pq.read_table(out / "docs.parquet")
    assert table.column("title").to_pylist()[3] == "Title 3"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["variant"] == "full" and manifest["n_docs"] == 8
    assert manifest["identity"] == a.BrowseCompBundle().identity()
    assert (out / "encoder" / "config.json").exists()


def test_dev_bundle_is_dev_pools_plus_unpooled_docs_and_no_test_evidence(tmp_path, monkeypatch):
    _patch(monkeypatch)
    out = tmp_path / "dev"
    a.BrowseCompBundle().build("dev", dev_row_ids=["0", "1"], test_row_ids=["2"], out_dir=out)
    docids = json.loads((out / "docids.json").read_text())
    assert set(docids) == {"d0", "d1", "d2", "d3", "d7"}      # pools of rows 0,1 + unpooled d7
    vectors = np.load(out / "vectors.npy")
    assert vectors.shape == (5, 8)
    for i, d in enumerate(docids):
        assert vectors[i, int(d[1:])] == 1.0                    # rows selected by docid


def test_dev_bundle_refuses_test_evidence_leaking_through_a_dev_pool(tmp_path, monkeypatch):
    leaky = list(_POOLS)
    leaky[0] = ({"d0"}, {"d4"})        # row 0's negatives include row 2's relevant doc
    _patch(monkeypatch, pools=leaky)
    with pytest.raises(a.BundleError, match="test evidence"):
        a.BrowseCompBundle().build("dev", dev_row_ids=["0"], test_row_ids=["2"], out_dir=tmp_path / "dev")
    assert not (tmp_path / "dev").exists()


def test_dev_bundle_refuses_a_non_integer_row_id(tmp_path, monkeypatch):
    _patch(monkeypatch)
    with pytest.raises(a.BundleError, match="not an integer"):
        a.BrowseCompBundle().build("dev", dev_row_ids=["abc"], test_row_ids=["2"], out_dir=tmp_path / "dev")
    assert not (tmp_path / "dev").exists()


def test_dev_bundle_refuses_a_row_id_outside_the_pools(tmp_path, monkeypatch):
    _patch(monkeypatch)
    with pytest.raises(a.BundleError, match="out of range"):
        a.BrowseCompBundle().build("dev", dev_row_ids=["99"], test_row_ids=["2"], out_dir=tmp_path / "dev")
    assert not (tmp_path / "dev").exists()


def test_identity_changes_with_any_pinned_source():
    base = a.BrowseCompBundle()
    assert base.identity() == a.BrowseCompBundle().identity()
    assert base.identity() != a.BrowseCompBundle(encoder_revision="other").identity()
    assert base.name == "browsecomp" and base.memory_mb == 49152 and base.gpu is True


def test_registered_browsecomp_dataset_pins_the_same_queries_revision():
    assert BROWSECOMP.revision == a.QUERIES_REVISION


def test_identity_changes_with_the_bundle_format(monkeypatch):
    before = a.BrowseCompBundle().identity()
    monkeypatch.setattr(a, "BUNDLE_FORMAT", a.BUNDLE_FORMAT + 1)
    assert a.BrowseCompBundle().identity() != before
