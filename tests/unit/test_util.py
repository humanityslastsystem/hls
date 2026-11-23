from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from hls import util
from hls.util import (SHOWN_VALIDATION_ERRORS, JsonDepthError, entry_kind, gzip_text_writer, json_safe, load_json,
                      validation_errors, walk_tree)


def test_json_safe_coerces_non_native_scalars():
    class _NpFloat:  # mimics a numpy/torch scalar: float()-able, not a Python float
        def __init__(self, v): self._v = v
        def __float__(self): return self._v

    class _Opaque:  # neither native nor float()-able
        def __repr__(self): return "opaque"

    out = json_safe({"f1": _NpFloat(0.5), "nested": [_NpFloat(1.0), "x", 3],
                     "weird": _Opaque(), "ok": None})
    assert out == {"f1": 0.5, "nested": [1.0, "x", 3], "weird": "opaque", "ok": None}
    json.dumps(out)  # must be serializable


def test_json_safe_coerces_inf_and_nan_to_none():
    # inf/nan are not valid RFC-8259 JSON (json.dumps emits bare Infinity/NaN that
    # strict consumers reject). json_safe coerces them to null -- both as native
    # floats and as float()-able non-native scalars.
    class _NpInf:
        def __float__(self): return float("inf")

    out = json_safe({"pos": float("inf"), "neg": float("-inf"),
                     "nan": float("nan"), "finite": 1.5, "scalar": _NpInf()})
    assert out == {"pos": None, "neg": None, "nan": None, "finite": 1.5, "scalar": None}
    # strict=True rejects Infinity/NaN, so this proves the output is RFC-8259 clean.
    json.dumps(out, allow_nan=False)


def test_json_nested_past_its_depth_is_refused_before_it_is_parsed(monkeypatch):
    """Objects and arrays count alike, and brackets inside a string, after an escaped quote too,
    are no nesting. A text past the depth never reaches the parser."""
    assert load_json(b'{"a": [{"b": []}]}', max_depth=4) == {"a": [{"b": []}]}
    assert load_json(b'["[[[", "\\"{{{"]', max_depth=1) == ["[[[", '"{{{']
    # Side by side, each closes before the next opens: pip's report holds thousands at depth 6.
    assert load_json(b"[" + b", ".join([b'{"a": []}'] * 100) + b"]", max_depth=3) == [{"a": []}] * 100
    parsed = []
    monkeypatch.setattr(util.json, "loads", parsed.append)
    for text in (b'{"a": [{"b": [[]]}]}', b'["\\"", [[[[]]]]]', b"{" * 200_000):
        with pytest.raises(JsonDepthError, match="^nested deeper than 4 levels$"):
            load_json(text, max_depth=4)
    assert parsed == []


def test_json_is_read_as_utf8_the_encoding_its_depth_was_counted_in():
    """``json.loads`` of bytes would read UTF-16 and UTF-32 too. In UTF-16 U+4122 is the bytes of
    '"A', so the brackets after it are inside a string to the depth count and nested to the
    parser."""
    hiding = ('["\u4122", ' + "[" * 100 + "]" * 100 + "]").encode("utf-16-le")
    assert json.loads(hiding)[1] != []
    for text in (hiding, '{"a": 1}'.encode("utf-16"), '{"a": 1}'.encode("utf-32"), b'"\xff"'):
        with pytest.raises(ValueError):
            load_json(text, max_depth=1)
    assert load_json('["\u4122"]'.encode(), max_depth=1) == ["\u4122"]


class _Inner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[int]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    count: int
    inner: _Inner


#: A value a refusal must never print: the record field it was given in may hold a secret.
_VALUE = "value-the-refusal-never-prints"


def test_validation_errors_name_each_errors_type_and_place_never_its_input():
    with pytest.raises(ValidationError) as caught:
        _Record.model_validate({"count": _VALUE, "inner": {"items": [1, _VALUE]}, "extra": _VALUE})
    text = validation_errors(caught.value)
    assert text == ("missing name (required by this version of hls); int_parsing count; "
                    "int_parsing inner.items.1; extra_forbidden extra")
    assert _VALUE not in text and _VALUE in str(caught.value)


def test_validation_errors_place_an_error_of_the_whole_input_at_the_top_level():
    with pytest.raises(ValidationError) as caught:
        _Record.model_validate_json(b'{"name": "' + _VALUE.encode())
    assert validation_errors(caught.value) == "json_invalid at the top level"
    with pytest.raises(ValidationError) as caught:
        _Record.model_validate([_VALUE])
    assert validation_errors(caught.value) == "model_type at the top level"


def test_validation_errors_name_the_first_few_and_count_the_rest():
    with pytest.raises(ValidationError) as caught:
        _Inner.model_validate({"items": [_VALUE] * (SHOWN_VALIDATION_ERRORS + 4)})
    shown = "; ".join(f"int_parsing items.{i}" for i in range(SHOWN_VALIDATION_ERRORS))
    assert validation_errors(caught.value) == f"{shown}; and 4 more"


def _walked_tree(root: Path) -> None:
    """A tree whose names sort one way by name and another by path string ("a" below "a-b" and
    "a0"), with dot names, a link to a file and one to a directory, and a pipe."""
    for d in ("a/x", "a/.hidden", "a-b", "z"):
        (root / d).mkdir(parents=True)
    for f in ("a0", "a/x/f", "a/.hidden/g", "a-b/h", ".top"):
        (root / f).write_bytes(f.encode())
    (root / "z" / "to-file").symlink_to(root / "a0")
    (root / "z" / "to-dir").symlink_to(root / "a")
    os.mkfifo(root / "z" / "pipe")


def test_a_walk_yields_every_entry_once_with_its_kind_and_lstat_in_the_order_of_its_paths(tmp_path):
    _walked_tree(tmp_path)
    walked = list(walk_tree(tmp_path))
    assert [e.path for e in walked] == sorted(e.path for e in walked)
    assert {e.rel: e.kind for e in walked} == {
        ".top": "file", "a": "directory", "a/.hidden": "directory", "a/.hidden/g": "file", "a/x": "directory",
        "a/x/f": "file", "a-b": "directory", "a-b/h": "file", "a0": "file", "z": "directory", "z/pipe": "special",
        "z/to-dir": "link", "z/to-file": "link"}
    assert len(walked) == 13 and all(e.path == tmp_path / e.rel for e in walked)
    assert all(e.stat.st_ino == os.lstat(e.path).st_ino for e in walked)


def test_a_walk_reads_no_file_and_never_enters_a_link(tmp_path, monkeypatch):
    # A pipe opened would block; a link to a directory entered would walk that directory twice.
    _walked_tree(tmp_path)
    monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("the walk opened a file"))
    assert [e.rel for e in walk_tree(tmp_path) if e.rel.startswith("z/")] == ["z/pipe", "z/to-dir", "z/to-file"]


def test_a_directory_the_walk_cannot_list_raises_once_the_walk_reaches_it(tmp_path):
    _walked_tree(tmp_path)
    os.chmod(tmp_path / "a-b", 0)
    walked = []
    try:
        with pytest.raises(PermissionError):
            for entry in walk_tree(tmp_path):
                walked.append(entry.rel)
    finally:
        os.chmod(tmp_path / "a-b", 0o755)
    assert walked[-1] == "a-b" and "a/x/f" in walked


def test_a_skipped_directory_is_left_out_with_everything_below_it_and_never_listed(tmp_path):
    _walked_tree(tmp_path)
    os.chmod(tmp_path / "a", 0)
    try:
        rels = [e.rel for e in walk_tree(tmp_path, skip=[tmp_path / "a"])]
    finally:
        os.chmod(tmp_path / "a", 0o755)
    assert rels == [".top", "a-b", "a-b/h", "a0", "z", "z/pipe", "z/to-dir", "z/to-file"]


def test_an_entry_gone_between_its_directorys_listing_and_its_lstat_raises(tmp_path, monkeypatch):
    """Every tree a caller walks holds still while it is walked (a published store tree, an
    extracted image, a fill its role finished), so an entry that goes while it is walked means the
    tree changed under the walk: the walk raises rather than leave the entry unread."""
    (tmp_path / "gone").write_bytes(b"x")
    (tmp_path / "kept").write_bytes(b"x")
    real = os.scandir

    class _Listing:
        """The real listing of a directory, taken whole, with "gone" removed once it was listed."""

        def __init__(self, path):
            with real(path) as it:
                self.entries = list(it)
            (tmp_path / "gone").unlink(missing_ok=True)

        def __enter__(self):
            return iter(self.entries)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(util.os, "scandir", _Listing)
    with pytest.raises(FileNotFoundError):
        list(walk_tree(tmp_path))


def test_a_walk_keeps_its_own_list_whatever_the_interpreters_recursion_limit(tmp_path):
    deep = tmp_path
    for _ in range(150):
        deep = deep / "d"
    deep.mkdir(parents=True)
    (deep / "leaf").write_bytes(b"x")
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(100)
    try:
        rels = [e.rel for e in walk_tree(tmp_path)]
    finally:
        sys.setrecursionlimit(limit)
    assert rels[-1] == "/".join(["d"] * 150 + ["leaf"])


def test_a_walk_of_what_is_no_directory_raises(tmp_path):
    (tmp_path / "file").write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        list(walk_tree(tmp_path / "file"))
    with pytest.raises(FileNotFoundError):
        list(walk_tree(tmp_path / "absent"))


@pytest.mark.parametrize("mode, kind", [(stat.S_IFREG, "file"), (stat.S_IFDIR, "directory"), (stat.S_IFLNK, "link"),
                                        (stat.S_IFIFO, "special"), (stat.S_IFSOCK, "special"),
                                        (stat.S_IFCHR, "special"), (stat.S_IFBLK, "special")])
def test_every_mode_is_one_kind(mode, kind):
    assert entry_kind(mode | 0o644) == kind


def test_the_same_text_gzipped_is_the_same_bytes_whenever_and_wherever_it_is_written(tmp_path, monkeypatch):
    """A harness record written gzipped carries no write time and no file name in its header, so
    a record of the same content is the same file and its digest says so."""
    import gzip
    import time

    with gzip_text_writer(tmp_path / "a.jsonl.gz") as out:
        out.write('{"x": "\u00e9"}\n')
    monkeypatch.setattr(time, "time", lambda: 2_000_000_000.0)
    with gzip_text_writer(tmp_path / "elsewhere.jsonl.gz") as out:
        out.write('{"x": "\u00e9"}\n')
    first = (tmp_path / "a.jsonl.gz").read_bytes()
    assert first == (tmp_path / "elsewhere.jsonl.gz").read_bytes()
    assert first[4:8] == bytes(4) and gzip.decompress(first) == '{"x": "\u00e9"}\n'.encode()
