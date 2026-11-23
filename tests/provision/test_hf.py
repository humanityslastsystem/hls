from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from hls.provision.hf import SourceFile, check_against_source, git_blob_sha1, source_files
from hls.store.cache import hash_tree, scan_dir

SHA = "0123456789abcdef0123456789abcdef01234567"


def _tree(tmp_path):
    root = tmp_path / "entry"
    (root / "sub").mkdir(parents=True)
    (root / "config.json").write_text('{"a": 1}')
    (root / "sub" / "model.safetensors").write_bytes(b"\x00" * 1000)
    return root


def _source(root):
    """The listing the hub would give for _tree: config.json in git, the shard in LFS."""
    cfg = (root / "config.json").read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(cfg) + cfg).hexdigest()
    shard = (root / "sub" / "model.safetensors").read_bytes()
    return {"config.json": SourceFile(size=len(cfg), sha256=None, git_sha1=blob),
            "sub/model.safetensors": SourceFile(size=len(shard), sha256=hashlib.sha256(shard).hexdigest(),
                                                git_sha1=None)}


def test_git_blob_sha1_is_what_git_hash_object_prints(tmp_path):
    f = tmp_path / "f"
    f.write_text("hello\n")
    assert git_blob_sha1(f) == "ce013625030ba8dba906f756967f9e9ca394464a"


def _reported(root, source):
    """What the weights role reports for the snapshot at *root*: each file's record with the
    sha256 it hashed, and the git blob id of each file the hub keeps in git."""
    files = scan_dir(root, digests=hash_tree(root))
    blobs = {f.path: git_blob_sha1(root / f.path) for f in files
             if f.path in source and source[f.path].git_sha1 is not None}
    return files, blobs


def test_a_snapshot_that_matches_the_hub_has_no_problems(tmp_path):
    root = _tree(tmp_path)
    source = _source(root)
    assert check_against_source(*_reported(root, source), source) == []


def test_a_changed_shard_is_caught_by_its_lfs_sha256(tmp_path):
    root = _tree(tmp_path)
    source = _source(root)
    (root / "sub" / "model.safetensors").write_bytes(b"\x01" * 1000)
    problems = check_against_source(*_reported(root, source), source)
    assert len(problems) == 1 and "sub/model.safetensors: sha256" in problems[0]


def test_a_changed_git_file_is_caught_by_its_blob_id(tmp_path):
    root = _tree(tmp_path)
    source = _source(root)
    (root / "config.json").write_text('{"a": 2}')
    problems = check_against_source(*_reported(root, source), source)
    assert problems == [f"config.json: git blob id differs from the hub's {source['config.json'].git_sha1}"]


def test_a_git_file_whose_blob_id_was_not_reported_is_a_problem(tmp_path):
    """The check reads no file: a git file's blob id comes from the role that wrote it."""
    root = _tree(tmp_path)
    source = _source(root)
    files, _ = _reported(root, source)
    assert check_against_source(files, {}, source) == ["config.json: the role reported no git blob id"]


def test_missing_extra_and_truncated_files_are_each_named(tmp_path):
    root = _tree(tmp_path)
    source = _source(root)
    (root / "config.json").unlink()
    (root / "stray.bin").write_bytes(b"x")
    (root / "sub" / "model.safetensors").write_bytes(b"\x00" * 10)
    problems = check_against_source(*_reported(root, source), source)
    assert problems == ["missing config.json", "unexpected stray.bin",
                        "sub/model.safetensors: 10 bytes, the hub says 1000"]


def test_a_source_file_has_exactly_one_hash():
    with pytest.raises(ValueError, match="either"):
        SourceFile(size=1, sha256=None, git_sha1=None)
    with pytest.raises(ValueError, match="either"):
        SourceFile(size=1, sha256="a" * 64, git_sha1="b" * 40)


class _Api:
    def __init__(self, info):
        self.info = info
        self.calls = []

    def model_info(self, repo, **kwargs):
        self.calls.append((repo, kwargs))
        return self.info


def _sibling(name, *, size, blob_id=None, lfs=None):
    return SimpleNamespace(rfilename=name, size=size, blob_id=blob_id, lfs=lfs)


def test_source_files_reads_lfs_and_git_hashes_from_the_hub(monkeypatch):
    info = SimpleNamespace(sha=SHA, siblings=[
        _sibling("config.json", size=8, blob_id="c" * 40),
        _sibling("model.safetensors", size=134, blob_id="d" * 40,
                 lfs=SimpleNamespace(size=1000, sha256="e" * 64, pointer_size=134)),
    ])
    api = _Api(info)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    listing = source_files("org/m", SHA, token="tok")
    assert listing == {"config.json": SourceFile(size=8, sha256=None, git_sha1="c" * 40),
                       "model.safetensors": SourceFile(size=1000, sha256="e" * 64, git_sha1=None)}
    assert api.calls == [("org/m", {"revision": SHA, "files_metadata": True, "token": "tok"})]


def test_source_files_refuses_an_answer_for_another_commit(monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _Api(SimpleNamespace(sha="f" * 40, siblings=[])))
    with pytest.raises(ValueError, match="not " + SHA):
        source_files("org/m", SHA, token=False)


def test_source_files_refuses_a_file_without_a_size(monkeypatch):
    info = SimpleNamespace(sha=SHA, siblings=[_sibling("x", size=None, blob_id="c" * 40)])
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _Api(info))
    with pytest.raises(ValueError, match="without a size"):
        source_files("org/m", SHA, token=False)
