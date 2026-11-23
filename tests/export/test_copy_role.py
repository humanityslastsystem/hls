"""The export's copy role, run in this process: each file read once without following a link
anywhere on its path, each copy created exclusively and read back, and the report it writes."""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from hls.store import export_copy
from hls.store.digest import COPY_FAILED_EXIT
from hls.store.export_copy import main, read_copy_report


def _store(tmp_path):
    root = tmp_path / "store"
    (root / "runs" / "t" / "r").mkdir(parents=True)
    (root / "runs" / "t" / "r" / "a.json").write_bytes(b'{"a": 1}\n')
    (root / "runs" / "t" / "r" / "name\nwith newline").write_bytes(b"agent log\n")
    (root / "runs" / "t" / "r" / ".env").write_bytes(b"EDITOR=vi\n")
    (root / "runs" / "t" / "r" / "not-utf8-\udcff.log").write_bytes(b"bytes\n")
    return root


def _list(tmp_path, keys):
    path = tmp_path / "list"
    path.write_bytes(b"".join(os.fsencode(k) + b"\0" for k in keys))
    return path


def _made(path) -> str:
    """``--to-made`` for the directory at *path*: its inode and change time."""
    found = os.lstat(path)
    return f"{found.st_ino}:{found.st_ctime_ns}"


def _run(tmp_path, root, keys, *, to=None, made=None):
    (tmp_path / "out").mkdir(exist_ok=True)
    to = to or tmp_path / "out"
    return main(["--root", str(root), "--list", str(_list(tmp_path, keys)), "--to", str(to),
                 "--to-made", made or _made(to), "--report", str(tmp_path / "report")])


def test_every_listed_file_is_copied_and_reported_by_what_landed(tmp_path):
    root = _store(tmp_path)
    keys = ["runs/t/r/a.json", "runs/t/r/name\nwith newline", "runs/t/r/.env", "runs/t/r/not-utf8-\udcff.log"]
    assert _run(tmp_path, root, keys) == 0
    report = read_copy_report(tmp_path / "report")
    for key in keys:
        data = (root / key).read_bytes()
        assert (tmp_path / "out" / key).read_bytes() == data
        assert report[key] == (hashlib.sha256(data).hexdigest(), len(data))
    lines = (tmp_path / "report").read_text().splitlines()
    assert [json.loads(line)["key"] for line in lines] == keys


def test_an_empty_list_copies_nothing_and_writes_an_empty_report(tmp_path):
    assert _run(tmp_path, _store(tmp_path), []) == 0
    assert list((tmp_path / "out").iterdir()) == [] and read_copy_report(tmp_path / "report") == {}


@pytest.mark.parametrize("bad", ["../escape", "runs//r", "/abs/key", "runs/./r", "runs/t/r/"])
def test_a_key_that_is_not_a_relative_path_fails(tmp_path, bad, capsys):
    assert _run(tmp_path, _store(tmp_path), [bad]) == 1
    assert "is not a relative path inside the store" in capsys.readouterr().err


def test_a_link_is_never_followed(tmp_path):
    root = _store(tmp_path)
    (root / "runs" / "t" / "r" / "link").symlink_to(root / "runs" / "t" / "r" / "a.json")
    assert _run(tmp_path, root, ["runs/t/r/link"]) == 1
    assert not (tmp_path / "out" / "runs" / "t" / "r" / "link").exists()


def test_a_link_above_the_file_is_never_followed(tmp_path):
    root = _store(tmp_path)
    (root / "runs" / "t" / "linked").symlink_to(root / "runs" / "t" / "r")
    assert _run(tmp_path, root, ["runs/t/linked/a.json"]) == 1
    assert not (tmp_path / "out" / "runs" / "t" / "linked").exists()


def test_a_source_that_is_not_a_regular_file_fails(tmp_path, capsys):
    root = _store(tmp_path)
    os.mkfifo(root / "runs" / "t" / "r" / "fifo")
    assert _run(tmp_path, root, ["runs/t/r/fifo"]) == 1
    assert "runs/t/r/fifo is not a regular file" in capsys.readouterr().err


def test_a_copy_already_there_fails(tmp_path):
    root = _store(tmp_path)
    assert _run(tmp_path, root, ["runs/t/r/a.json", "runs/t/r/a.json"]) == 1


def test_the_directory_it_copies_into_is_the_stages_and_must_be_there(tmp_path):
    root = _store(tmp_path)
    assert _run(tmp_path, root, ["runs/t/r/a.json"], to=tmp_path / "missing", made="1:1") == 1
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("where", ["the-directory", "above-it"])
def test_a_directory_reached_through_a_link_is_never_copied_into(tmp_path, where):
    root = _store(tmp_path)
    real = tmp_path / "real"
    (real / "out").mkdir(parents=True)
    if where == "the-directory":
        (tmp_path / "link").symlink_to(real / "out")
        to = tmp_path / "link"
    else:
        (tmp_path / "link").symlink_to(real)
        to = tmp_path / "link" / "out"
    assert _run(tmp_path, root, ["runs/t/r/a.json"], to=to, made=_made(real / "out")) == 1
    assert list((real / "out").iterdir()) == []


def test_a_directory_other_than_the_one_the_stage_made_is_never_copied_into(tmp_path, capsys):
    root = _store(tmp_path)
    (tmp_path / "out").mkdir()
    made = _made(tmp_path / "out")
    # Another writer put its own directory where the stage made the copy's.
    (tmp_path / "out").rmdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "out").mkdir()
    assert _run(tmp_path, root, ["runs/t/r/a.json"], made=made) == 1
    assert list((tmp_path / "out").iterdir()) == []
    assert "is not the directory the export made for its copy" in capsys.readouterr().err


def test_a_copy_whose_bytes_differ_from_what_was_read_fails_for_a_retry(tmp_path, monkeypatch, capsys):
    root = _store(tmp_path)
    monkeypatch.setattr(export_copy, "_read_back", lambda fd: ("0" * 64, os.fstat(fd).st_size))
    assert _run(tmp_path, root, ["runs/t/r/a.json"]) == COPY_FAILED_EXIT
    assert capsys.readouterr().err.startswith("copy failed: the copy of runs/t/r/a.json has sha256 ")


def test_an_unreadable_report_reads_as_none(tmp_path):
    (tmp_path / "report").write_text("not json\n")
    assert read_copy_report(tmp_path / "report") is None
    assert read_copy_report(tmp_path / "missing") is None
