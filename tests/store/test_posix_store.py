from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

from hls.store.base import AlreadyExists, NotFound, StoreError, check_key, check_segment
from hls.store.posix import PosixStore

KEY = "runs/t/r1/stages/agent_build/MANIFEST.json"


def _race_create(root: str, idx: int, q) -> None:
    store = PosixStore(Path(root))
    try:
        store.create(KEY, f"writer-{idx}".encode())
        q.put(idx)
    except AlreadyExists:
        q.put(None)


def test_create_is_exclusive_across_processes(tmp_path):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_race_create, args=(str(tmp_path), i, q)) for i in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    results = [q.get(timeout=5) for _ in procs]
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert PosixStore(tmp_path).read_bytes(KEY) == f"writer-{winners[0]}".encode()


def test_create_refuses_existing_key(tmp_path):
    store = PosixStore(tmp_path)
    store.create("a/b.json", b"1")
    with pytest.raises(AlreadyExists):
        store.create("a/b.json", b"2")
    assert store.read_bytes("a/b.json") == b"1"


def test_write_bytes_replaces(tmp_path):
    store = PosixStore(tmp_path)
    store.write_bytes("a/status.json", b"old")
    store.write_bytes("a/status.json", b"new")
    assert store.read_bytes("a/status.json") == b"new"


def test_read_missing_raises_not_found(tmp_path):
    with pytest.raises(NotFound):
        PosixStore(tmp_path).read_bytes("nope/x.json")


def test_list_keys_is_recursive_sorted_and_hides_temp_files(tmp_path):
    store = PosixStore(tmp_path)
    store.write_bytes("i/b.json", b"")
    store.write_bytes("i/a/c.json", b"")
    (tmp_path / "i" / ".tmp-stray").write_bytes(b"partial")
    assert store.list_keys("i") == ["i/a/c.json", "i/b.json"]
    assert store.list_keys("missing") == []


def test_open_write_is_invisible_until_exit_and_aborts_on_error(tmp_path):
    store = PosixStore(tmp_path)
    with store.open_write("big/trace.jsonl") as f:
        f.write(b"line1\n")
        assert not store.exists("big/trace.jsonl")
    assert store.read_bytes("big/trace.jsonl") == b"line1\n"
    with pytest.raises(RuntimeError):
        with store.open_write("big/other.jsonl") as f:
            f.write(b"half")
            raise RuntimeError("writer died")
    assert not store.exists("big/other.jsonl")
    assert [p.name for p in (tmp_path / "big").iterdir()] == ["trace.jsonl"]


def test_open_write_exclusive(tmp_path):
    store = PosixStore(tmp_path)
    with store.open_write("x/m.json", exclusive=True) as f:
        f.write(b"1")
    with pytest.raises(AlreadyExists):
        with store.open_write("x/m.json", exclusive=True) as f:
            f.write(b"2")
    assert store.read_bytes("x/m.json") == b"1"


def test_open_read(tmp_path):
    store = PosixStore(tmp_path)
    store.write_bytes("r/x", b"abc")
    with store.open_read("r/x") as f:
        assert f.read() == b"abc"
    with pytest.raises(NotFound):
        with store.open_read("r/missing"):
            pass


def test_delete_is_quiet_on_missing(tmp_path):
    store = PosixStore(tmp_path)
    store.write_bytes("d/x", b"1")
    store.delete("d/x")
    store.delete("d/x")
    assert not store.exists("d/x")


@pytest.mark.parametrize(
    "bad", ["", "/abs", "a/../b", "a//b", "a/.hidden", ".x", "a/b/", "a b", "a/b\n", "a\n/b"]
)
def test_check_key_rejects(bad):
    with pytest.raises(ValueError):
        check_key(bad)


@pytest.mark.parametrize("bad", ["", "a/b", ".x", "a b", "a\n"])
def test_check_segment_rejects(bad):
    with pytest.raises(ValueError):
        check_segment(bad)


def test_check_segment_returns_valid_name():
    assert check_segment("r1-20261002T000000Z.x") == "r1-20261002T000000Z.x"


def test_bad_key_rejected_before_filesystem(tmp_path):
    with pytest.raises(ValueError):
        PosixStore(tmp_path).write_bytes("../escape", b"x")
    assert not (tmp_path.parent / "escape").exists()


def test_publish_dir(tmp_path):
    store = PosixStore(tmp_path)
    staging = store.staging_dir("cache")
    (staging / "w.bin").write_bytes(b"weights")
    store.publish_dir(staging, "cache/weights/k1")
    assert store.read_bytes("cache/weights/k1/w.bin") == b"weights"
    assert not staging.exists()
    again = store.staging_dir("cache")
    (again / "w.bin").write_bytes(b"other")
    with pytest.raises(AlreadyExists):
        store.publish_dir(again, "cache/weights/k1")


@pytest.mark.parametrize("area", ["../../escape", "a/b", "", ".hidden", "cache\n"])
def test_staging_area_must_be_one_segment(tmp_path, area):
    root = tmp_path / "store"
    with pytest.raises(ValueError):
        PosixStore(root).staging_dir(area)
    assert list(tmp_path.rglob("*")) == []


def test_a_directory_is_not_an_object(tmp_path):
    store = PosixStore(tmp_path)
    store.write_bytes("d/sub/x", b"1")
    assert store.exists("d/sub/x")
    assert not store.exists("d/sub")
    with pytest.raises(NotFound):
        store.read_bytes("d/sub")
    with pytest.raises(NotFound):
        with store.open_read("d/sub"):
            pass
    store.delete("d/sub")
    assert store.read_bytes("d/sub/x") == b"1"


def test_a_key_below_an_object_is_not_found(tmp_path):
    store = PosixStore(tmp_path)
    store.write_bytes("f", b"1")
    assert not store.exists("f/x")
    with pytest.raises(NotFound):
        store.read_bytes("f/x")
    with pytest.raises(NotFound):
        with store.open_read("f/x"):
            pass
    store.delete("f/x")
    assert store.read_bytes("f") == b"1"


def test_list_children_is_one_level_sorted_and_hides_dotfiles(tmp_path):
    store = PosixStore(tmp_path)
    store.write_bytes("i/b.json", b"")
    store.write_bytes("i/a/c.json", b"")
    store.write_bytes("i/a/d/e.json", b"")
    (tmp_path / "i" / ".tmp-stray").write_bytes(b"partial")
    assert store.list_children("i") == ["a", "b.json"]
    assert store.list_children("i/a") == ["c.json", "d"]
    assert store.list_children("i/b.json") == []
    assert store.list_children("missing") == []


def test_size_reads_an_objects_size_from_its_metadata_alone(tmp_path):
    store = PosixStore(tmp_path)
    store.write_bytes(KEY, b"12345")
    assert store.size(KEY) == 5
    with pytest.raises(NotFound):
        store.size("runs/t/r1/absent")
    (tmp_path / "elsewhere").write_bytes(b"x")
    store.path("runs/t/r1/link").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(StoreError, match="not a regular file"):
        store.size("runs/t/r1/link")
