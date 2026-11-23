"""The tree digest of a task's uploaded folder (``hls.agent.uploads.tree_digest``) and the spec
field that commits it beside ``upload_dir``: the material a contestant is handed is the material
the task names, byte for byte, whatever checkout the build runs from."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls.agent.uploads import TreeDigestError, UploadRefused, checked_upload, merged_digest, merged_view, tree_digest
from hls.artifact.artifact import AgentProducerSpec, Budget, HighestAvailable


def _tree(root):
    (root / "docs" / "cuda").mkdir(parents=True)
    (root / "README.md").write_text("index\n")
    (root / "docs" / "cuda" / "guide.html").write_text("<p>guide</p>\n")
    (root / "tool.sh").write_text("#!/bin/sh\n")
    for path in (root / "README.md", root / "docs" / "cuda" / "guide.html"):
        path.chmod(0o644)
    (root / "tool.sh").chmod(0o755)
    return root


def test_the_digest_is_the_sha256_of_each_files_path_mode_and_bytes_in_path_order(tmp_path):
    root = _tree(tmp_path / "refs")
    lines = "".join(f"{rel}\0{mode}\0{hashlib.sha256(data).hexdigest()}\n" for rel, mode, data in [
        ("README.md", "644", b"index\n"), ("docs/cuda/guide.html", "644", b"<p>guide</p>\n"),
        ("tool.sh", "755", b"#!/bin/sh\n")])
    assert tree_digest(root) == hashlib.sha256(lines.encode()).hexdigest()


def test_the_same_tree_elsewhere_has_the_same_digest_and_an_empty_directory_adds_nothing(tmp_path):
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    (b / "empty").mkdir()
    assert tree_digest(a) == tree_digest(b)


@pytest.mark.parametrize("change", ["byte", "extra", "removed", "renamed", "mode"])
def test_any_change_to_a_file_moves_the_digest(tmp_path, change):
    root = _tree(tmp_path / "refs")
    before = tree_digest(root)
    guide = root / "docs" / "cuda" / "guide.html"
    if change == "byte":
        guide.write_text("<p>guidE</p>\n")
    elif change == "extra":
        (root / "docs" / "extra.txt").write_text("")
    elif change == "removed":
        guide.unlink()
    elif change == "renamed":
        guide.rename(guide.with_name("guide.htm"))
    else:
        guide.chmod(0o755)
    assert tree_digest(root) != before


def test_a_checkouts_umask_does_not_move_the_digest(tmp_path):
    """Git holds a file as 644 or 755, and a checkout writes the other bits from its umask: a
    folder a commit holds has one digest under umask 002 and 022 alike."""
    root = _tree(tmp_path / "refs")
    before = tree_digest(root)
    (root / "README.md").chmod(0o664)
    (root / "docs" / "cuda" / "guide.html").chmod(0o600)
    (root / "tool.sh").chmod(0o775)
    assert tree_digest(root) == before


@pytest.mark.parametrize("where", ["file", "directory", "root"])
def test_a_link_is_refused(tmp_path, where):
    root = _tree(tmp_path / "refs")
    if where == "file":
        os.symlink(root / "README.md", root / "docs" / "link.md")
        target = root
    elif where == "directory":
        os.symlink(root / "docs", root / "more")
        target = root
    else:
        os.symlink(root, tmp_path / "link")
        target = tmp_path / "link"
    with pytest.raises(TreeDigestError, match="is a link"):
        tree_digest(target)


def test_an_entry_that_is_no_file_or_directory_is_refused(tmp_path, monkeypatch):
    # A socket file: opening one fails at once, so even a digest that tried would not block.
    root = _tree(tmp_path / "refs")
    monkeypatch.chdir(root)  # a socket's path is bounded at 108 bytes, so it is bound relative
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind("sock")
        with pytest.raises(TreeDigestError, match="sock is neither a file nor a directory"):
            tree_digest(root)


#: A program that takes a digest of the folder named by its argument and exits 3, printing the
#: refusal, when the folder is refused.
_DIGEST = ("import sys\nfrom pathlib import Path\nfrom hls.agent.uploads import TreeDigestError, tree_digest\n"
           "try:\n    tree_digest(Path(sys.argv[1]))\nexcept TreeDigestError as e:\n    print(e)\n    sys.exit(3)\n")


def test_a_fifo_is_refused_without_being_opened(tmp_path):
    # A pipe with no writer blocks whoever opens it for reading, so the digest runs in a process
    # of its own under a time limit: a digest that opened the pipe fails here instead of hanging.
    root = _tree(tmp_path / "refs")
    os.mkfifo(root / "pipe")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    done = subprocess.run([sys.executable, "-c", _DIGEST, str(root)], capture_output=True, text=True, env=env,
                          timeout=30)
    assert (done.returncode, done.stderr) == (3, "")
    assert "pipe is neither a file nor a directory" in done.stdout


def test_a_directory_that_cannot_be_listed_is_refused(tmp_path):
    """The files below a directory the digest cannot list would be left out of it unseen."""
    root = _tree(tmp_path / "refs")
    os.chmod(root / "docs", 0)
    try:
        with pytest.raises(TreeDigestError, match="docs cannot be listed"):
            tree_digest(root)
    finally:
        os.chmod(root / "docs", 0o755)


def test_a_root_that_is_no_directory_is_refused(tmp_path):
    with pytest.raises(TreeDigestError, match="is not a directory"):
        tree_digest(tmp_path / "absent")


def test_one_folders_merged_digest_is_its_tree_digest(tmp_path):
    root = _tree(tmp_path / "refs")
    assert merged_digest([root]) == tree_digest(root)


def test_the_merged_digest_is_the_tree_digest_of_the_folders_copied_into_one(tmp_path):
    shared = _tree(tmp_path / "refs")
    own = tmp_path / "own"
    (own / "docs" / "model").mkdir(parents=True)
    (own / "docs" / "model" / "out.jsonl").write_text("{}\n")
    (own / "selfcheck.py").write_text("print()\n")
    both = tmp_path / "both"
    both.mkdir()
    for folder in (shared, own):
        # As the build's upload role copies each folder.
        subprocess.run(["cp", "-rL", "--", f"{folder}/.", str(both)], check=True)
    assert merged_digest([shared, own]) == merged_digest([own, shared]) == tree_digest(both)
    assert merged_view([shared, own])["docs/model/out.jsonl"] == own / "docs" / "model" / "out.jsonl"
    assert merged_view([shared, own])["docs/cuda/guide.html"] == shared / "docs" / "cuda" / "guide.html"


@pytest.mark.parametrize("held", ["README.md", "README.md/inside.txt", "docs/cuda"])
def test_a_path_two_folders_hold_is_refused(tmp_path, held):
    """Directories merge; a file two folders hold, or a file where another folder has a
    directory, would be replaced or buried by the second copy."""
    shared = _tree(tmp_path / "refs")
    own = tmp_path / "own"
    (own / held).parent.mkdir(parents=True, exist_ok=True)
    (own / held).write_text("own\n")
    first = held.split("/")[0] if held != "docs/cuda" else held
    with pytest.raises(TreeDigestError, match=f"^{re.escape(first)} is held by both {re.escape(str(shared))} and "):
        merged_digest([shared, own])


def test_the_upload_folders_are_a_tuple_and_never_a_bare_path():
    with pytest.raises(ValidationError, match="tuple"):
        _spec(upload_dir="docs", upload_brief="Docs.", upload_sha256="a" * 64)
    with pytest.raises(ValidationError, match="at least 1 item"):
        _spec(upload_dir=(), upload_brief="Docs.", upload_sha256="a" * 64)


UPLOADING = ("tasks.multitenant_slo.task", "tasks.tab_completion.task", "tasks.multimodal_tts.task")


def test_the_tasks_that_upload_the_references_commit_one_digest_for_them():
    producers = [importlib.import_module(m).ARTIFACT.producer for m in UPLOADING]
    assert {(p.upload_dir, p.upload_sha256) for p in producers} == {((
        "../assets/references",), "2458984914b96a16fbe591a48e412626599ea7a55782ae7b728d740c66636ca5")}
    folder = Path(__file__).resolve().parents[2] / "tasks" / "assets" / "references"
    if not folder.is_dir():
        pytest.skip(f"{folder} is not in this checkout (it is gitignored)")
    assert tree_digest(folder) == producers[0].upload_sha256


def test_the_running_page_says_where_the_references_go_and_quotes_the_refusal(tmp_path):
    """The folder is not committed, so docs/running.md names where it goes in the step before a
    task's first run, and the refusal it quotes there is in the plan's refusal without the
    folder, as the planner words it."""
    from hls.pipeline.planner import PlanError
    from hls.pipeline.preflight import check_upload_folder

    page = (Path(__file__).resolve().parents[2] / "docs" / "running.md").read_text()
    step = " ".join(page.split("## Before a task's first run")[1].split("\n## ")[0].split())
    assert "`tasks/assets/references`" in step
    quoted = re.search(r"a line that says `([^`]+)`", step)
    assert quoted is not None
    producer = importlib.import_module("tasks.multitenant_slo.task").ARTIFACT.producer
    with pytest.raises(PlanError) as refused:
        check_upload_folder(tmp_path, "tasks.multitenant_slo.task", producer)
    assert quoted.group(1) in str(refused.value)


# ---------------------------------------------------------------------------
# The one rule a build's upload folder is judged by, at plan time and when the build starts
# ---------------------------------------------------------------------------


def _checkout(tmp_path):
    """A checkout holding a task directory and the references it uploads, and their digest."""
    checkout = tmp_path / "checkout"
    (checkout / "tasks" / "mslo").mkdir(parents=True)
    folder = _tree(checkout / "tasks" / "assets" / "references")
    return checkout, folder, tree_digest(folder)


def test_the_upload_folder_the_task_pins_is_resolved_and_digested(tmp_path):
    checkout, folder, digest = _checkout(tmp_path)
    base = checkout / "tasks" / "mslo"
    assert checked_upload(("../assets/references",), base, checkout, digest) == ([folder], digest)
    assert checked_upload((str(folder),), base, checkout, digest) == ([folder], digest)


def test_a_link_on_the_way_that_resolves_inside_the_checkout_is_followed(tmp_path):
    checkout, folder, digest = _checkout(tmp_path)
    (checkout / "tasks" / "docs").symlink_to(folder)
    assert checked_upload(("../docs",), checkout / "tasks" / "mslo", checkout, digest) == ([folder], digest)


@pytest.mark.parametrize("case, words", [
    ("absent", "is not a directory"),
    ("link-out-of-the-checkout", "lies outside the checkout"),
    ("dot-dot-out-of-the-checkout", "lies outside the checkout"),
    ("link-inside", "is a link"),
    ("changed", "has tree digest"),
    ("extra", "has tree digest"),
])
def test_an_upload_folder_the_task_does_not_pin_is_refused(tmp_path, case, words):
    checkout, folder, digest = _checkout(tmp_path)
    value = "../assets/references"
    if case == "absent":
        value = "../assets/gone"
    elif case == "link-out-of-the-checkout":
        _tree(tmp_path / "elsewhere")
        (checkout / "tasks" / "docs").symlink_to(tmp_path / "elsewhere")
        value = "../docs"
    elif case == "dot-dot-out-of-the-checkout":
        _tree(tmp_path / "elsewhere")
        value = "../../../elsewhere"
    elif case == "link-inside":
        os.symlink(folder / "README.md", folder / "docs" / "README.md")
    elif case == "changed":
        (folder / "README.md").write_text("another index\n")
    else:
        (folder / "notes.txt").write_text("")
    with pytest.raises(UploadRefused, match=words):
        checked_upload((value,), checkout / "tasks" / "mslo", checkout, digest)


def _spec(**kw):
    return AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600), **kw)


def test_an_upload_folder_commits_its_digest():
    with pytest.raises(ValidationError, match="upload_sha256 is required when upload_dir is set"):
        _spec(upload_dir=("docs",), upload_brief="Docs.")
    with pytest.raises(ValidationError, match="upload_sha256 is set but upload_dir is not"):
        _spec(upload_sha256="a" * 64)
    with pytest.raises(ValidationError, match="not a bare hex SHA-256"):
        _spec(upload_dir=("docs",), upload_brief="Docs.", upload_sha256="A" * 64)
    assert _spec(upload_dir=("docs",), upload_brief="Docs.", upload_sha256="a" * 64).upload_sha256 == "a" * 64
