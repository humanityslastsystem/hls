"""``hls.util.remove_tree``: removes a tree of any depth without recursing, and never follows a link."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hls.util import remove_tree

#: Deeper than the interpreter recurses: shutil.rmtree before Python 3.12.4 recurses once a level.
DEEP = sys.getrecursionlimit() + 100


def nest(root: Path, levels: int) -> Path:
    """A chain of *levels* directories under *root* with a file at the bottom, made one level at a
    time so that building it does not recurse; the deepest directory."""
    p = root
    for _ in range(levels):
        p = p / "a"
        p.mkdir(parents=True)
    (p / "f").write_bytes(b"x")
    return p


def test_a_tree_deeper_than_the_interpreter_recurses_is_removed(tmp_path):
    nest(tmp_path / "tree", DEEP)
    remove_tree(tmp_path / "tree")
    assert list(tmp_path.iterdir()) == []


def test_a_link_inside_the_tree_is_removed_and_never_followed(tmp_path):
    outside = tmp_path / "outside"
    (outside / "sub").mkdir(parents=True)
    (outside / "keep.txt").write_text("keep")
    tree = tmp_path / "tree"
    (tree / "d").mkdir(parents=True)
    (tree / "d" / "to-dir").symlink_to(outside)
    (tree / "to-file").symlink_to(outside / "keep.txt")
    remove_tree(tree)
    assert not tree.exists() and (outside / "keep.txt").read_text() == "keep" and (outside / "sub").is_dir()


def test_a_link_given_itself_is_removed_alone(tmp_path):
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "f").write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "target")
    remove_tree(tmp_path / "link")
    assert not (tmp_path / "link").exists() and (tmp_path / "target" / "f").read_text() == "x"


def test_a_file_is_removed_and_nothing_is_nothing(tmp_path):
    (tmp_path / "f").write_text("x")
    remove_tree(tmp_path / "f")
    remove_tree(tmp_path / "absent")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root removes what the mode forbids")
def test_an_error_is_raised_unless_errors_are_ignored(tmp_path):
    locked = tmp_path / "tree" / "locked"
    locked.mkdir(parents=True)
    (locked / "f").write_text("x")
    (tmp_path / "tree" / "other").write_text("y")
    locked.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            remove_tree(tmp_path / "tree")
        remove_tree(tmp_path / "tree", ignore_errors=True)
        assert sorted(p.name for p in (tmp_path / "tree").iterdir()) == ["locked"]
    finally:
        locked.chmod(0o700)
