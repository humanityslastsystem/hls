"""The commit point: the workspace's main as a bundle, read without running anything the
contestant's repository configures."""

from __future__ import annotations

import ast
import contextlib
import errno
import hashlib
import importlib
import io
import json
import os
import re
import resource
import signal
import stat
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

import pytest

from hls.agent import commit
from hls.agent.commit import (BRANCHES_MAX, COMMIT_TARGET, IO_ERROR_EXIT, LINE_MAX_BYTES, LISTING_MAX_BYTES,
                              NO_MAIN_EXIT, NONBLOB_MAX_BYTES, REASON_MAX_CHARS, SNAPSHOT_KEEP, UNREADABLE_EXIT,
                              NoSubmission, check_repository, commit_command, errno_failure, git_failure,
                              prune_snapshots, snapshot_name)
from hls.artifact.artifact import AgentProducedArtifact
from hls.package import checkout
from hls.pipeline.pipelines import sandbox_disk_gib
from hls.store.digest import sha256_file
from hls.store.manifest import read_outputs_report
from tests.package.bundles import LIES, kind_tree, lying_repo, make_tree_bundle, repeated_tree

PROGRAM = Path(commit.__file__)
# Every hook git documents; each trap below writes a marker if git ever runs it.
HOOKS = ("applypatch-msg", "pre-applypatch", "post-applypatch", "pre-commit", "pre-merge-commit",
         "prepare-commit-msg", "commit-msg", "post-commit", "pre-rebase", "post-checkout", "post-merge", "pre-push",
         "pre-receive", "update", "proc-receive", "post-receive", "post-update", "reference-transaction",
         "push-to-checkout", "pre-auto-gc", "post-rewrite", "sendemail-validate", "fsmonitor-watchman",
         "p4-changelist", "p4-prepare-changelist", "p4-post-changelist", "p4-pre-submit", "post-index-change",
         "post-upload-pack")


def _env(home: Path) -> dict[str, str]:
    """git's environment for a test's own commands. Automatic maintenance is off: a commit starts
    it detached, and from git 2.54 its geometric repack can still be writing packs, a cruft pack
    and a multi-pack index while the test lays out and reads the object store it means to."""
    home.mkdir(parents=True, exist_ok=True)
    return {"PATH": os.environ["PATH"], "HOME": str(home), "LANG": "C.UTF-8", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_AUTHOR_DATE": "2026-10-02T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-10-02T00:00:00Z", "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "maintenance.auto", "GIT_CONFIG_VALUE_0": "false", "GIT_CONFIG_KEY_1": "gc.auto",
            "GIT_CONFIG_VALUE_1": "0"}


def _git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-c", "user.name=agent", "-c", "user.email=agent@example.com", "-c",
                        "init.defaultBranch=main", "-C", str(repo), *args], env=_env(repo.parent / ".home"),
                       capture_output=True, text=True, check=True)
    return p.stdout.strip()


def _repo(path: Path, files: dict[str, str], *, branch: str = "main") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", branch)
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "work")
    return path


def _readonly(root: Path) -> None:
    """Make *root* read-only, as the commit container mounts the workspace."""
    for dirpath, dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            if not p.is_symlink():
                p.chmod(p.stat().st_mode & ~0o222)
        Path(dirpath).chmod(Path(dirpath).stat().st_mode & ~0o222)


@pytest.fixture(autouse=True)
def _writable_again(tmp_path):
    yield
    for dirpath, dirs, files in os.walk(tmp_path):
        Path(dirpath).chmod(Path(dirpath).stat().st_mode | stat.S_IWUSR)


def _limit_memory(gib: float):
    """A ``preexec_fn`` that caps the child's address space at *gib* GiB."""
    def limit() -> None:
        cap = int(gib * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    return limit


def _run(ws: Path, out_dir: Path, *, readonly: bool = True, snapshot: bool = False,
         memory_gib: float | None = None) -> tuple[int, dict]:
    """Run the program the way the container does: an isolated interpreter, the stdlib only.
    Returns its status and what it wrote."""
    if readonly and ws.exists() and not ws.is_symlink():
        _readonly(ws)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir.parent / "container-tmp"
    tmp.mkdir(exist_ok=True)
    argv = [sys.executable, "-I", str(PROGRAM), "--workspace", str(ws), "--out", str(out_dir / "source.bundle"),
            "--record", str(out_dir / "record.json"), "--tmp", str(tmp), "--max-bytes", str(MAX),
            "--seconds", str(SECONDS)]
    if not snapshot:
        argv += ["--no-submission", str(out_dir / "no_submission.json")]
    p = subprocess.run(argv, env={"PATH": os.environ["PATH"]}, cwd=tmp, capture_output=True, text=True,
                       preexec_fn=None if memory_gib is None else _limit_memory(memory_gib))
    files = {f.name: f for f in out_dir.iterdir()}
    found: dict = {"stdout": p.stdout, "stderr": p.stderr, "files": sorted(files)}
    if "record.json" in files:
        found["record"] = json.loads(files["record.json"].read_text())
    if "no_submission.json" in files:
        found["no_submission"] = NoSubmission.from_json(files["no_submission.json"].read_text())
    assert list(tmp.iterdir()) == [], "the program leaves nothing in its temporary directory"
    return p.returncode, found


def _heads(bundle: Path) -> list[str]:
    p = subprocess.run(["git", "bundle", "list-heads", str(bundle)], env=_env(bundle.parent / ".home"),
                       capture_output=True, text=True, check=True)
    return sorted(p.stdout.split("\n")[:-1])


def _checkout(bundle: Path, tree: Path) -> checkout.SourceRecord:
    """The package's checkout of *bundle* into *tree*, as its role runs it."""
    home = tree.parent / "pkg-home"
    home.mkdir()
    return checkout.checkout(bundle, tree, home, bundle_sha256=sha256_file(bundle)[0], max_bytes=MAX)


# --- bundles -------------------------------------------------------------------------------


def test_a_workspace_with_commits_on_main_is_bundled_with_main_alone(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "print('v1')\n"})
    _git(ws, "checkout", "-q", "-b", "dev")
    (ws / "server.py").write_text("print('dev')\n")
    _git(ws, "commit", "-qam", "dev work")
    _git(ws, "checkout", "-q", "main")
    (ws / "server.py").write_text("print('v2')\n")
    _git(ws, "commit", "-qam", "more")
    _git(ws, "tag", "v2")
    main = _git(ws, "rev-parse", "main")
    code, found = _run(ws, tmp_path / "out")
    assert code == 0, found["stderr"]
    assert found["files"] == ["record.json", "source.bundle"]
    assert _heads(tmp_path / "out" / "source.bundle") == [f"{main} refs/heads/main"]
    assert found["record"] == {"status": "ok", "commit": main, "reason": None, "branches": ["dev", "main"],
                               "at": found["record"]["at"]}
    assert found["stdout"] == f"hls-commit: ok {main}\n"


def test_the_bundle_is_one_the_package_checkout_builds_from(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n", "pyproject.toml": "[project]\nname = 'x'\n"})
    main = _git(ws, "rev-parse", "main")
    assert _run(ws, tmp_path / "out")[0] == 0
    source = _checkout(tmp_path / "out" / "source.bundle", tmp_path / "tree")
    assert (source.status, source.commit, source.files) == ("ok", main, ["pyproject.toml", "server.py"])


def test_uncommitted_work_is_not_in_the_bundle(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "print('committed')\n"})
    (ws / "server.py").write_text("print('edited')\n")
    (ws / "new.py").write_text("x = 1\n")
    assert _run(ws, tmp_path / "out")[0] == 0
    _checkout(tmp_path / "out" / "source.bundle", tmp_path / "tree")
    assert (tmp_path / "tree" / "server.py").read_text() == "print('committed')\n"
    assert not (tmp_path / "tree" / "new.py").exists()


@pytest.mark.parametrize("paths", [
    {".git/config": False, "server.py": False},
    {"/".join(["d"] * checkout.TREE_MAX_DEPTH) + "/f": False},
], ids=["a-path-git-refuses", "past-the-depth-limit"])
def test_a_main_the_package_refuses_is_bundled_and_left_to_the_packages_checkout(tmp_path, paths):
    """The commit point reads main's objects and never checks its tree out, so a tree git would
    not write, or one past the package's limits, is bundled as it is. The package's checkout then
    refuses it from git's own listing as the submission's (``over_limit``) and writes no file."""
    made = make_tree_bundle(tmp_path, paths, name="made.bundle")
    ws = tmp_path / "ws"
    ws.mkdir()
    _git(ws, "init", "-q", "-b", "main")
    _git(ws, "fetch", "-q", "--update-head-ok", str(made), "main:main")
    main = _git(ws, "rev-parse", "main")
    code, found = _run(ws, tmp_path / "out")
    assert code == 0, found["stderr"]
    assert _heads(tmp_path / "out" / "source.bundle") == [f"{main} refs/heads/main"]
    source = _checkout(tmp_path / "out" / "source.bundle", tmp_path / "tree")
    assert (source.status, source.commit, source.files) == ("over_limit", main, [])
    assert not (tmp_path / "tree").exists()


# --- no submission -------------------------------------------------------------------------


def test_no_commit_on_main_is_the_contestants_with_the_branches_named(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "x\n"}, branch="dev")
    code, found = _run(ws, tmp_path / "out")
    assert code == NO_MAIN_EXIT
    assert found["files"] == ["no_submission.json", "record.json"]
    assert found["no_submission"].kind == "no_main" and "its branches are dev" in found["no_submission"].reason
    assert found["record"]["status"] == "no_main" and found["record"]["branches"] == ["dev"]


def test_a_workspace_without_a_repository_is_the_contestants(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "server.py").write_text("x\n")
    code, found = _run(ws, tmp_path / "out")
    assert code == NO_MAIN_EXIT and found["no_submission"].kind == "no_main"
    assert "no git repository" in found["no_submission"].reason


def test_a_repository_with_no_commits_is_the_contestants(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _git(ws, "init", "-q")
    code, found = _run(ws, tmp_path / "out")
    assert code == NO_MAIN_EXIT and "it has no commits" in found["no_submission"].reason


def test_a_snapshot_without_main_writes_no_submission_record(tmp_path):
    ws = _repo(tmp_path / "ws", {"a": "x\n"}, branch="dev")
    code, found = _run(ws, tmp_path / "out", snapshot=True)
    assert code == NO_MAIN_EXIT and found["files"] == ["record.json"]


def test_a_repository_with_a_great_many_branches_is_named_in_a_bounded_record(tmp_path):
    """The branches are the contestant's to make, so the record and the reason hold a bounded
    number of them."""
    ws = _repo(tmp_path / "ws", {"a": "x\n"}, branch="dev")
    commit_id = _git(ws, "rev-parse", "dev")
    heads = ws / ".git" / "refs" / "heads"
    for n in range(BRANCHES_MAX + 50):
        (heads / f"b{n:05d}").write_text(commit_id + "\n")
    code, found = _run(ws, tmp_path / "out")
    assert code == NO_MAIN_EXIT
    assert len(found["record"]["branches"]) == BRANCHES_MAX
    assert found["record"]["branches"] == sorted(found["record"]["branches"])
    reason = found["no_submission"].reason
    assert "its branches are b" in reason and "more" in reason and len(reason) <= REASON_MAX_CHARS


# --- the repository's own configuration never runs -------------------------------------------


def _trap(markers: Path, name: str) -> str:
    script = markers.parent / "traps" / name
    script.parent.mkdir(exist_ok=True)
    script.write_text(f"#!/bin/sh\ntouch {markers / name}\ncat\n")
    script.chmod(0o755)
    return str(script)


def test_nothing_the_workspaces_repository_configures_runs(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n", ".gitattributes": "* filter=trap diff=trap\n"})
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for name in HOOKS:
        for where in (hooks, ws / ".git" / "hooks"):
            (where / name).write_text(f"#!/bin/sh\ntouch {markers / ('hook-' + name)}\n")
            (where / name).chmod(0o755)
    (ws / "extra.cfg").write_text(f"[uploadpack]\n\tpackObjectsHook = {_trap(markers, 'included')}\n")
    settings = {"core.hooksPath": str(hooks), "core.fsmonitor": _trap(markers, "fsmonitor"),
                "filter.trap.clean": _trap(markers, "clean"), "filter.trap.smudge": _trap(markers, "smudge"),
                "filter.trap.process": _trap(markers, "process"), "filter.trap.required": "true",
                "diff.trap.textconv": _trap(markers, "textconv"), "diff.external": _trap(markers, "diff"),
                "uploadpack.packObjectsHook": _trap(markers, "packobjects"),
                "core.alternateRefsCommand": _trap(markers, "alternaterefs"),
                "core.sshCommand": _trap(markers, "ssh"), "core.gitProxy": _trap(markers, "proxy"),
                "core.pager": _trap(markers, "pager"), "core.editor": _trap(markers, "editor"),
                "credential.helper": "!" + _trap(markers, "credential"), "gpg.program": _trap(markers, "gpg"),
                "alias.clone": "!" + _trap(markers, "alias-clone"),
                "alias.bundle": "!" + _trap(markers, "alias-bundle"),
                "alias.ls-remote": "!" + _trap(markers, "alias-ls-remote"),
                "include.path": str(ws / "extra.cfg"), "core.fsyncMethod": "batch"}
    for key, value in settings.items():
        _git(ws, "config", key, value)
    main = _git(ws, "rev-parse", "main")
    code, found = _run(ws, tmp_path / "out")
    assert code == 0, found["stderr"]
    assert found["record"]["commit"] == main
    assert sorted(p.name for p in markers.iterdir()) == []


def test_a_partial_clones_promisor_remote_is_never_dialled(tmp_path):
    """A repository missing an object could make git fetch it from the repository's own remote,
    whose URL a contestant writes; every transport but local files is refused."""
    markers = tmp_path / "markers"
    markers.mkdir()
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    blob = _git(ws, "rev-parse", "main:server.py")
    (ws / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    for key, value in {"core.repositoryformatversion": "1", "extensions.partialClone": "origin",
                       "remote.origin.url": f"ext::sh -c touch% {markers / 'ext'}",
                       "remote.origin.promisor": "true"}.items():
        _git(ws, "config", key, value)
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT and found["no_submission"].kind == "unreadable"
    assert list(markers.iterdir()) == []


def test_a_git_that_still_fetches_lazily_never_dials_the_repositorys_remote(tmp_path, monkeypatch):
    """git before 2.45.1 fetched an object missing from a partial clone from the repository's
    promisor remote, through the transport that remote names; ssh runs the repository's own
    ``core.sshCommand``. ``GIT_NO_LAZY_FETCH=0`` makes this git fetch so too. git never opens
    the workspace's repository, so it never learns of the remote."""
    markers = tmp_path / "markers"
    markers.mkdir()
    ssh = tmp_path / "traps" / "ssh"
    ssh.parent.mkdir()
    ssh.write_text(f"#!/bin/sh\ntouch {markers / 'ssh'}\nexit 1\n")
    ssh.chmod(0o755)
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    blob = _git(ws, "rev-parse", "main:server.py")
    (ws / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    for key, value in {"core.repositoryformatversion": "1", "extensions.partialClone": "origin",
                       "remote.origin.url": "ssh://example.invalid/repo.git", "remote.origin.promisor": "true",
                       "core.sshCommand": str(ssh)}.items():
        _git(ws, "config", key, value)
    real = commit.git_env
    monkeypatch.setattr(commit, "git_env", lambda home: {**real(home), "GIT_NO_LAZY_FETCH": "0"})
    (tmp_path / "tmp").mkdir()
    with pytest.raises(commit._NoSubmission) as caught:
        commit.commit(ws, tmp_path / "out" / "source.bundle", tmp=tmp_path / "tmp", max_bytes=MAX,
                      seconds=SECONDS)
    assert caught.value.kind == "unreadable"
    assert list(markers.iterdir()) == []


def test_git_never_reads_the_workspaces_configuration(tmp_path):
    """Every git command that opens a repository refuses one of format version 99; the bundle
    is still made, so no git command ever opened the workspace's repository."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    main = _git(ws, "rev-parse", "main")
    _git(ws, "config", "core.repositoryformatversion", "99")
    code, found = _run(ws, tmp_path / "out")
    assert code == 0, found["stderr"]
    assert _heads(tmp_path / "out" / "source.bundle") == [f"{main} refs/heads/main"]


def test_main_kept_in_packed_refs_or_behind_a_symbolic_ref_is_read(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    _git(ws, "checkout", "-q", "-b", "real")
    (ws / "server.py").write_text("print('real')\n")
    _git(ws, "commit", "-qam", "real")
    real = _git(ws, "rev-parse", "real")
    _git(ws, "pack-refs", "--all")
    assert not (ws / ".git" / "refs" / "heads" / "main").exists()
    (ws / ".git" / "refs" / "heads" / "main").write_text("ref: refs/heads/real\n")
    code, found = _run(ws, tmp_path / "out")
    assert code == 0 and found["record"]["commit"] == real and found["record"]["branches"] == ["main", "real"]


# --- repositories whose history is not in the workspace --------------------------------------


def _borrowing(tmp_path: Path, base: Path, entry: str | bytes) -> Path:
    """A workspace whose main has no objects of its own: they are all in *base*, which its
    alternates name as *entry*."""
    _repo(base, {"server.py": "print('base')\n"})
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    _git(ws, "init", "-q", "-b", "main")
    line = entry if isinstance(entry, bytes) else entry.encode()
    (ws / ".git" / "objects" / "info" / "alternates").write_bytes(b"# borrowed\n" + line + b"\n")
    _git(ws, "fetch", "-q", "--update-head-ok", str(base), "main:main")
    assert list((ws / ".git" / "objects" / "pack").iterdir()) == []
    return ws


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_alternates_inside_the_workspace_are_followed(tmp_path, relative):
    base = tmp_path / "ws" / "vendor" / "base"
    entry = "../../vendor/base/.git/objects" if relative else str(base / ".git" / "objects")
    ws = _borrowing(tmp_path, base, entry)
    code, found = _run(ws, tmp_path / "out")
    assert code == 0, found["stderr"]
    # The bundle is whole: it needs no other repository to be cloned.
    source = _checkout(tmp_path / "out" / "source.bundle", tmp_path / "tree")
    assert (tmp_path / "tree" / "server.py").read_text() == "print('base')\n" and source.status == "ok"


@pytest.mark.parametrize("chain", [False, True], ids=["direct", "through-an-inside-alternate"])
def test_alternates_that_lead_outside_the_workspace_are_unreadable(tmp_path, chain):
    outside = tmp_path / "elsewhere"
    if chain:
        inner = tmp_path / "ws" / "vendor" / "inner" / "objects"
        (inner / "info").mkdir(parents=True)
        (inner / "info" / "alternates").write_text(str(outside / ".git" / "objects") + "\n")
        ws = _borrowing(tmp_path, outside, str(inner))
    else:
        ws = _borrowing(tmp_path, outside, str(outside / ".git" / "objects"))
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT and "outside the workspace" in found["no_submission"].reason
    assert found["files"] == ["no_submission.json", "record.json"]


def test_an_alternate_through_a_link_whose_name_is_not_utf8_is_judged_by_the_path_git_reads(tmp_path):
    """git reads an alternates entry byte for byte; the check resolves the same bytes, so a link
    whose name is not UTF-8 is followed to where it leads, outside the workspace."""
    outside = tmp_path / "elsewhere"
    vendor = tmp_path / "ws" / "vendor"
    vendor.mkdir(parents=True)
    os.symlink(outside / ".git", os.fsencode(vendor) + b"/\xff")
    ws = _borrowing(tmp_path, outside, os.fsencode(vendor) + b"/\xff/objects")
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT and "outside the workspace" in found["no_submission"].reason
    json.dumps(found["record"]).encode()


def test_every_unreadable_refusal_names_a_cause_the_brief_states():
    """The brief states every cause the commit point refuses a repository as unreadable for
    (``UNREADABLE``, which ``hls.prompt`` renders): each refusal is made through ``_unreadable``
    with one of them, every one is used, and no refusal of that kind is made another way."""
    tree = ast.parse(PROGRAM.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_unreadable"]
    causes = [call.args[0].value for call in calls if isinstance(call.args[0], ast.Constant)]
    assert len(causes) == len(calls) and set(causes) == set(commit.UNREADABLE)
    kinds = [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == "unreadable"]
    # The kind is written in _unreadable alone, and read where a record is checked.
    assert len(kinds) == 2
    with pytest.raises(ValueError, match="not a cause the brief states"):
        commit._unreadable("unstated", "a reason")


@pytest.mark.parametrize("layout, match", [
    ("symlink-out", "symbolic link"), ("symlink-in", "symbolic link"), ("gitfile", "is a file"),
    ("commondir", "linked worktree"),
])
def test_a_repository_kept_elsewhere_is_unreadable(tmp_path, layout, match):
    """Even a .git that links to a repository inside the workspace is refused: git never makes
    one, and the refusal does not depend on where it leads."""
    other = _repo(tmp_path / "other", {"server.py": "print('other')\n"})
    ws = tmp_path / "ws"
    ws.mkdir()
    if layout == "symlink-out":
        (ws / ".git").symlink_to(other / ".git")
    elif layout == "symlink-in":
        inner = _repo(ws / "inner", {"server.py": "print('inner')\n"})
        (ws / ".git").symlink_to(inner / ".git")
    elif layout == "gitfile":
        (ws / ".git").write_text(f"gitdir: {other / '.git'}\n")
    else:
        _repo(ws, {"server.py": "print('ws')\n"})
        (ws / ".git" / "commondir").write_text(str(other / ".git") + "\n")
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT and match in found["no_submission"].reason


def test_a_corrupt_object_on_main_is_unreadable_with_gits_message(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    blob = _git(ws, "rev-parse", "main:server.py")
    path = ws / ".git" / "objects" / blob[:2] / blob[2:]
    path.chmod(0o644)
    path.write_bytes(b"not a zlib stream")
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT
    reason = found["no_submission"].reason
    assert reason.startswith("main's history cannot be read from the workspace's repository: ")
    assert len(reason) > 70


def test_a_main_that_names_a_missing_object_is_unreadable(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    (ws / ".git" / "refs" / "heads" / "main").write_text("1" * 40 + "\n")
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT and found["no_submission"].kind == "unreadable"
    assert found["no_submission"].reason.startswith("main's history cannot be read from the workspace's repository")


@pytest.mark.parametrize("what", ["object", "ref"])
def test_a_file_of_the_repository_the_contestant_made_unreadable_is_unreadable(tmp_path, what):
    """The commit container reads as the workspace's owner, so a file of the repository it is
    refused is refused by permissions the contestant set: git's message about an object, or
    Python's about a ref, is the contestant's content (exit 4), the errno named in the record."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    if what == "object":
        blob = _git(ws, "rev-parse", "main:server.py")
        (ws / ".git" / "objects" / blob[:2] / blob[2:]).chmod(0)
    else:
        (ws / ".git" / "refs" / "heads" / "main").chmod(0)
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT and found["no_submission"].kind == "unreadable"
    assert found["no_submission"].reason.startswith("EACCES: ")
    assert (found["record"]["status"], found["record"]["reason"]) == ("unreadable", found["no_submission"].reason)
    assert "source.bundle" not in found["files"]


@pytest.mark.parametrize("errno_name, inside, code", [
    ("EIO", True, IO_ERROR_EXIT), ("ESTALE", True, IO_ERROR_EXIT), ("ENOSPC", True, IO_ERROR_EXIT),
    ("EACCES", False, IO_ERROR_EXIT), ("EACCES", True, UNREADABLE_EXIT), ("EPERM", True, UNREADABLE_EXIT),
], ids=["eio", "estale", "enospc", "eacces-outside", "eacces-inside", "eperm-inside"])
def test_an_os_error_reading_the_repository_is_the_contestants_only_when_it_refuses_a_workspace_path(
        tmp_path, monkeypatch, capsys, errno_name, inside, code):
    """EIO, ESTALE and ENOSPC are the machine's wherever they happen, and so is a refused path
    outside the workspace: exit 74, the errno named on stderr and in the record, and no
    no-submission record, so the stage tries again. A refused path in the workspace is the
    contestant's content."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    path = ws / ".git" / "packed-refs" if inside else tmp_path / "elsewhere"
    number = getattr(errno, errno_name)

    def failing(git):
        raise OSError(number, os.strerror(number), str(path))

    monkeypatch.setattr(commit, "main_commit", failing)
    out, record = tmp_path / "out", tmp_path / "record" / "commit.json"
    assert commit.main(["--workspace", str(ws), "--out", str(out / "source.bundle"), "--no-submission",
                        str(out / "no_submission.json"), "--record", str(record), "--tmp", str(tmp_path), "--max-bytes", str(MAX),
            "--seconds", str(SECONDS)]) == code
    written = json.loads(record.read_text())
    assert written["reason"].startswith(f"{errno_name}: ")
    if code == IO_ERROR_EXIT:
        assert f"hls-commit: io-error: {errno_name}: reading the workspace's repository failed" in capsys.readouterr().err
        assert written["status"] == "io_error" and not out.exists()
    else:
        assert written["status"] == "unreadable"
        assert NoSubmission.from_json((out / "no_submission.json").read_text()).reason == written["reason"]


@pytest.mark.parametrize("where", [".git", ".git/commondir", ".git/objects"])
@pytest.mark.parametrize("errno_name, code", [("EIO", IO_ERROR_EXIT), ("ESTALE", IO_ERROR_EXIT),
                                              ("EACCES", UNREADABLE_EXIT)])
def test_an_os_error_where_the_repository_is_looked_for_is_never_read_as_its_absence(
        tmp_path, monkeypatch, where, errno_name, code):
    """Looking for the repository's parts asks the filesystem; an error of the machine's there is
    the machine's (exit 74), never an absent repository or part, which would be the contestant's
    zero."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    failing_path, number, real = str(ws / where), getattr(errno, errno_name), os.lstat

    def lstat(path, *args, **kwargs):
        if os.fspath(path) == failing_path:
            raise OSError(number, os.strerror(number), failing_path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(commit.os, "lstat", lstat)
    out, record = tmp_path / "out", tmp_path / "record" / "commit.json"
    assert commit.main(["--workspace", str(ws), "--out", str(out / "source.bundle"), "--no-submission",
                        str(out / "no_submission.json"), "--record", str(record), "--tmp", str(tmp_path), "--max-bytes", str(MAX),
            "--seconds", str(SECONDS)]) == code
    written = json.loads(record.read_text())
    assert written["reason"].startswith(f"{errno_name}: ")
    assert written["status"] == ("io_error" if code == IO_ERROR_EXIT else "unreadable")


WS = "/workspace"


@pytest.mark.parametrize("stderr, found", [
    ("error: unable to open loose object 45b9: Permission denied\nfatal: update_ref failed\n", ("EACCES", True)),
    ("error: unable to open loose object 45b9: Operation not permitted\n", ("EPERM", True)),
    ("error: unable to open object pack directory: /workspace/.git/objects/pack: Permission denied\n",
     ("EACCES", True)),
    ("fatal: cannot lock ref 'refs/heads/main': Unable to create '/tmp/hls-commit-1/main.git/./refs/heads/main.lock': "
     "Permission denied\n", ("EACCES", False)),
    ("fatal: unable to create '/out/.tmp-partial/source.bundle-9.lock': Permission denied\n", ("EACCES", False)),
    ("fatal: read error: Input/output error\n", ("EIO", False)),
    ("error: unable to read 45b9: Stale file handle\n", ("ESTALE", False)),
    ("fatal: write error: No space left on device\n", ("ENOSPC", False)),
    ("error: unable to open loose object 45b9: Permission denied\nfatal: read error: Input/output error\n",
     ("EIO", False)),
    ("error: corrupt loose object '45b9'\nfatal: loose object 45b9 is corrupt\n", None),
], ids=["refused-object", "eperm-object", "refused-workspace-path", "refused-fresh-repository", "refused-output",
        "eio", "estale", "enospc", "machine-wins", "corrupt"])
def test_gits_errno_is_the_contestants_only_for_a_refused_workspace_path(stderr, found):
    assert errno_failure(stderr, Path(WS)) == found


@pytest.mark.parametrize("layout, match", [
    ("objects-link", "objects lead outside"),
    ("refs-link", "refs/heads/main links outside"),
    ("packed-refs-link", "packed-refs lead outside"),
    ("alternates-link", "objects/info/alternates links outside"),
    ("alternate-store-listing-link", "alternates file is outside"),
    ("reftable", "reftable"),
    ("sha256", "SHA-256"),
    ("not-an-object-name", "not an object name"),
    ("symref-out-of-refs", "not a ref"),
    ("symref-chain", "chain of more than 5"),
    ("main-names-a-tree", "cannot be read"),
    ("refs-loop", "loop"),
    ("objects-loop", "loop"),
])
def test_refs_and_objects_git_would_not_write_are_unreadable(tmp_path, layout, match):
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    git = ws / ".git"
    main = _git(ws, "rev-parse", "main")
    outside = tmp_path / "outside"
    outside.mkdir()
    if layout == "objects-link":
        (git / "objects").rename(outside / "objects")
        (git / "objects").symlink_to(outside / "objects")
    elif layout == "refs-link":
        (outside / "main").write_text(main + "\n")
        (git / "refs" / "heads" / "main").unlink()
        (git / "refs" / "heads" / "main").symlink_to(outside / "main")
    elif layout == "packed-refs-link":
        (outside / "packed-refs").write_text(f"{main} refs/heads/main\n")
        (git / "packed-refs").symlink_to(outside / "packed-refs")
    elif layout == "alternates-link":
        (outside / "alternates").write_text(str(git / "objects") + "\n")
        (git / "objects" / "info").mkdir(exist_ok=True)
        (git / "objects" / "info" / "alternates").symlink_to(outside / "alternates")
    elif layout == "alternate-store-listing-link":
        store = ws / "vendor" / "objects"
        (store / "info").mkdir(parents=True)
        (outside / "alternates").write_text(str(git / "objects") + "\n")
        (store / "info" / "alternates").symlink_to(outside / "alternates")
        (git / "objects" / "info").mkdir(exist_ok=True)
        (git / "objects" / "info" / "alternates").write_text(str(store) + "\n")
    elif layout == "reftable":
        (git / "reftable").mkdir()
    elif layout == "sha256":
        (git / "refs" / "heads" / "main").write_text("a" * 64 + "\n")
    elif layout == "not-an-object-name":
        (git / "refs" / "heads" / "main").write_text("hello\n")
    elif layout == "symref-out-of-refs":
        (git / "refs" / "heads" / "main").write_text("ref: refs/../../../outside/main\n")
    elif layout == "symref-chain":
        for n in range(6):
            (git / "refs" / "heads" / ("main" if n == 0 else f"hop{n}")).write_text(f"ref: refs/heads/hop{n + 1}\n")
    elif layout == "refs-loop":
        (git / "refs" / "heads" / "main").unlink()
        (git / "refs" / "heads" / "main").symlink_to("loop")
        (git / "refs" / "heads" / "loop").symlink_to("main")
    elif layout == "objects-loop":
        (git / "objects").rename(ws / "store")
        (git / "objects").symlink_to(ws / "hop")
        (ws / "hop").symlink_to(git / "objects")
    else:
        (git / "refs" / "heads" / "main").write_text(_git(ws, "rev-parse", "main^{tree}") + "\n")
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT and match in found["no_submission"].reason
    assert "source.bundle" not in found["files"]


def test_a_corrupt_object_off_main_does_not_matter(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    _git(ws, "checkout", "-q", "-b", "dev")
    (ws / "scratch.txt").write_text("only on dev\n")
    _git(ws, "add", "scratch.txt")
    _git(ws, "commit", "-qm", "dev")
    blob = _git(ws, "rev-parse", "dev:scratch.txt")
    _git(ws, "checkout", "-q", "main")
    path = ws / ".git" / "objects" / blob[:2] / blob[2:]
    path.chmod(0o644)
    path.write_bytes(b"garbage")
    assert _run(ws, tmp_path / "out")[0] == 0


# --- what the contestant wrote is read in bounded memory -------------------------------------


@pytest.mark.parametrize("where", ["ref", "packed-refs", "alternates"])
def test_a_file_far_larger_than_git_writes_is_unreadable_without_being_read_whole(tmp_path, where):
    """Refs, packed-refs and alternates are the contestant's bytes, read a bounded line at a
    time: a file of 8 GiB with no line break is refused in a process held to 1 GiB, never a
    harness role that ran out of memory."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    git = ws / ".git"
    if where == "ref":
        target = git / "refs" / "heads" / "main"
    elif where == "packed-refs":
        (git / "refs" / "heads" / "main").unlink()
        target = git / "packed-refs"
    else:
        target = git / "objects" / "info" / "alternates"
    target.write_bytes(b"")
    os.truncate(target, 8 * 2**30)
    code, found = _run(ws, tmp_path / "out", memory_gib=1)
    assert code == UNREADABLE_EXIT, found["stderr"]
    bound = LINE_MAX_BYTES if where == "ref" else LISTING_MAX_BYTES
    assert f"more than {bound} bytes" in found["no_submission"].reason


@pytest.mark.parametrize("where", ["packed-refs", "alternates"])
def test_a_line_longer_than_git_writes_is_unreadable(tmp_path, where):
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    git = ws / ".git"
    if where == "packed-refs":
        (git / "refs" / "heads" / "main").unlink()
        target = git / "packed-refs"
    else:
        target = git / "objects" / "info" / "alternates"
    target.write_bytes(b"x" * (LINE_MAX_BYTES + 1))
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT, found["stderr"]
    assert f"a line of more than {LINE_MAX_BYTES} bytes" in found["no_submission"].reason


@pytest.mark.parametrize("where", ["loose-ref", "packed-refs", "alternates", "among-objects", "linked-from-objects"])
def test_a_pipe_where_git_or_the_program_reads_is_unreadable_and_never_waited_on(tmp_path, where):
    """git writes only directories, regular files and links in a repository; opening a pipe
    would wait for a writer that never comes, an end that would be ours. A special file where
    either reads is refused as the contestant's, at once."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    git = ws / ".git"
    if where == "loose-ref":
        (git / "refs" / "heads" / "main").unlink()
        os.mkfifo(git / "refs" / "heads" / "main")
    elif where == "packed-refs":
        (git / "refs" / "heads" / "main").unlink()
        os.mkfifo(git / "packed-refs")
    elif where == "alternates":
        os.mkfifo(git / "objects" / "info" / "alternates")
    else:
        blob = _git(ws, "rev-parse", "main:server.py")
        loose = git / "objects" / blob[:2] / blob[2:]
        loose.unlink()
        if where == "among-objects":
            os.mkfifo(loose)
        else:
            os.mkfifo(ws / "pipe")
            loose.symlink_to(ws / "pipe")
    (tmp_path / "out").mkdir()
    argv = [sys.executable, "-I", str(PROGRAM), "--workspace", str(ws), "--out", str(tmp_path / "out" / "b"),
            "--no-submission", str(tmp_path / "out" / "ns.json"), "--tmp", str(tmp_path / "out"),
            "--max-bytes", str(MAX),
            "--seconds", str(SECONDS)]
    p = subprocess.run(argv, env={"PATH": os.environ["PATH"]}, capture_output=True, text=True, timeout=60)
    assert p.returncode == UNREADABLE_EXIT, p.stderr
    assert "not a regular file" in NoSubmission.from_json((tmp_path / "out" / "ns.json").read_text()).reason


def _walked(git: Path) -> int:
    """The entries the walk visits in a plain repository: every name under its objects and refs."""
    return sum(len(dirs) + len(files) for part in ("objects", "refs") for _, dirs, files in os.walk(git / part))


#: The cap the tests give the program unless a test is about it: far past any repository here.
MAX = 2**40
#: The time the tests give the program unless a test is about it: an hour, far past what any
#: repository here takes.
SECONDS = 3600


def _main(ws: Path, tmp_path: Path, max_bytes: int = MAX, seconds: int = SECONDS) -> tuple[int, dict]:
    """Run the program in this process under the cap *max_bytes* with *seconds*; its status and the
    no-submission record, if any."""
    out = tmp_path / "out"
    code = commit.main(["--workspace", str(ws), "--out", str(out / "source.bundle"), "--no-submission",
                        str(out / "no_submission.json"), "--tmp", str(tmp_path), "--max-bytes", str(max_bytes),
                        "--seconds", str(seconds)])
    record = out / "no_submission.json"
    return code, json.loads(record.read_text()) if record.exists() else {}


def _walk_bound(monkeypatch, bound: int) -> None:
    """Make the walk's bound, ``OBJECTS_MAX`` and ``WALK_ROOM_ENTRIES`` beside it, *bound*, by
    its room."""
    monkeypatch.setattr(commit, "WALK_ROOM_ENTRIES", bound - commit.OBJECTS_MAX)


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
def test_the_walk_of_the_object_stores_and_refs_is_bounded_by_entry_count(tmp_path, monkeypatch, past):
    """A contestant can fill objects or refs, or point an alternate store at a tree of millions
    of entries, and a walk that outlasts the role's time would be ours. Past the walk's bound
    (``walk_max_entries``) the repository is over the limit, the contestant's, named by its
    bound."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n", "a.py": "x\n"})
    walked = _walked(ws / ".git")
    bound = walked - 1 if past else walked
    _walk_bound(monkeypatch, bound)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"more than {bound} entries" in record["reason"]
    else:
        assert code == 0 and record == {}


def test_the_walks_bound_is_every_object_git_is_given_and_room_beside_them():
    assert commit.walk_max_entries() == commit.OBJECTS_MAX + commit.WALK_ROOM_ENTRIES
    # The room holds at least the fan-out directories, info and pack of every store followed.


# --- main's tree, as git lists it ----------------------------------------------------------------


def _repeated(ws: Path, levels: int, width: int, *, files: bool = True) -> Path:
    """A workspace whose main is one commit of ``tests.package.bundles.repeated_tree``."""
    ws.mkdir()
    _git(ws, "init", "-q", "-b", "main")
    repeated_tree(ws, levels, width, files=files)
    return ws


@pytest.mark.parametrize("files, words", [
    (True, "the tree tracks more than the 200000 paths allowed"),
    (False, "the tree holds more than the 200000 directories allowed"),
], ids=["paths", "empty-directories"])
def test_a_tree_whose_subtrees_repeat_is_over_limit_at_the_commit_point(tmp_path, files, words):
    """A git bomb: nine levels of ten directories that each name the one below list ten billion
    paths (over the empty tree, a billion directories and no path) from eleven objects, within
    every bound on the objects (``TREE_BYTES_MAX`` bounds the tree objects, not the paths they
    list). The commit point reads git's listing of main's tree as it streams, each path and
    directory counted wherever it appears, ends git at the first bound passed and keeps no
    bundle: the repository is the contestant's, over the limit, before any later step lists it."""
    ws = _repeated(tmp_path / "ws", 9, 10, files=files)
    started = time.monotonic()
    code, found = _run(ws, tmp_path / "out")
    assert time.monotonic() - started < 60
    assert code == UNREADABLE_EXIT, found["stderr"]
    assert (found["no_submission"].kind, found["no_submission"].reason) == ("over_limit", words)
    assert "source.bundle" not in found["files"]


@pytest.mark.parametrize("limit, value, files, words", [
    ("TREE_MAX_ENTRIES", 3, True, "the tree tracks more than the 3 paths allowed"),
    ("TREE_MAX_DIRECTORIES", 1, False, "the tree holds more than the 1 directories allowed"),
    ("TREE_MAX_PATH_BYTES", 4, True, "the tree tracks a path longer than the 4 bytes allowed: d0/f0"),
    ("TREE_MAX_PATH_BYTES", 1, False, "the tree tracks a path longer than the 1 bytes allowed: d0"),
], ids=["paths", "directories", "path", "an-empty-directorys-path"])
def test_each_bound_on_mains_listing_holds_at_it_and_refuses_one_past(tmp_path, monkeypatch, limit, value, files,
                                                                      words):
    """One level of two directories over two files lists four paths ("d0/f0" the longest) in two
    directories; over the empty tree, two directories and no path. Each bound holds main at it and
    refuses it one past, as the contestant's, with no bundle kept."""
    ws = _repeated(tmp_path / "ws", 1, 2, files=files)
    monkeypatch.setattr(commit, limit, value)
    code, record = _main(ws, tmp_path)
    assert (code, record.get("kind"), record.get("reason")) == (UNREADABLE_EXIT, "over_limit", words)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["no_submission.json"]
    monkeypatch.setattr(commit, limit, value + 1)
    (tmp_path / "out" / "no_submission.json").unlink()
    assert _main(ws, tmp_path) == (0, {})


def test_the_bounds_on_mains_listing_are_the_packages():
    """The commit point and the package's checkout hold main's tree to one set of bounds, the
    commit point's own, through the one reader of git's listing (``listing_problem``)."""
    assert (commit.TREE_MAX_ENTRIES, commit.TREE_MAX_DIRECTORIES, commit.TREE_MAX_PATH_BYTES) == (
        200_000, 200_000, 1024)
    assert checkout.listing_problem is commit.listing_problem
    assert not {"TREE_MAX_ENTRIES", "TREE_MAX_DIRECTORIES", "TREE_MAX_PATH_BYTES"} & set(vars(checkout))


def test_the_listing_holds_every_trees_base_in_gits_cache(tmp_path, monkeypatch):
    """git lists main's tree with a cache of delta bases as large as the trees may be
    (``TREE_CACHE``), so each tree it reads is rebuilt from its base once; after the bundle, which
    read every object of main's history, and before the check of the objects it names."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    started: list[list[str]] = []
    real = subprocess.Popen

    def spy(argv, *args, **kwargs):
        started.append([str(a) for a in argv])
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(commit.subprocess, "Popen", spy)
    assert _main(ws, tmp_path) == (0, {})
    verbs = [commit._verb(argv[1 + len(commit.GIT_FLAGS):]) for argv in started if argv[0] == "git"]
    assert verbs[-3:] == ["bundle", "ls-tree", "cat-file"]
    listing = started[-2]
    at = listing.index("ls-tree")
    assert listing[at - 4:at - 2] == list(commit.TREE_CACHE)
    assert commit.TREE_CACHE == ("-c", f"core.deltaBaseCacheLimit={commit.TREE_BYTES_MAX}")


def test_a_listing_past_its_time_ends_git_and_the_try_is_infra(tmp_path, monkeypatch):
    """git's listing of main's tree is timed (``TREE_LIST_SEC``): past it git's session is ended
    and the try is infra, with no bundle kept."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    pid = tmp_path / "git.pid"
    real = subprocess.run(["sh", "-c", "command -v git"], capture_output=True, text=True, check=True).stdout.strip()
    fake = tmp_path / "bin" / "git"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\ncase " $* " in *" ls-tree "*) echo $$ > {pid}; exec sleep 60;; esac\n'
                    f'exec {real} "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    monkeypatch.setattr(commit, "TREE_LIST_SEC", 2)
    out = tmp_path / "out" / "source.bundle"
    (tmp_path / "tmp").mkdir()
    started = time.monotonic()
    with pytest.raises(commit._Infra, match="git ls-tree took longer than the 2 s it was given"):
        commit.commit(ws, out, tmp=tmp_path / "tmp", max_bytes=MAX, seconds=SECONDS)
    assert time.monotonic() - started < FAST_GIT_SEC
    hung = int(pid.read_text())
    try:
        assert not Path(f"/proc/{hung}").exists(), "git was left running"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(hung, signal.SIGKILL)
    assert list(out.parent.iterdir()) == []


@pytest.mark.parametrize("mode, target", [("100644", "tree"), ("100644", "commit"), ("100644", "tag"),
                                          ("100755", "tree"), ("120000", "tree")],
                         ids=["file-tree", "file-commit", "file-tag", "executable-tree", "link-tree"])
def test_a_file_whose_object_is_not_a_blob_is_unreadable_and_never_kept(tmp_path, mode, target):
    """git's plumbing writes a tree entry whose mode says file or symbolic link and whose object
    is a tree, a commit or a tag. git's listing prints the entry's kind from its mode and git's
    bundle packs the object by its own, so both pass it, and the package's checkout then cannot
    write it: git's checkout fails on it. The commit point checks the
    object of every file and link of main's tree and refuses such a repository as unreadable, the
    contestant's, keeping no bundle."""
    ws = tmp_path / "ws"
    ws.mkdir()
    kind_tree(ws, mode, target)
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT, found["stderr"]
    assert (found["no_submission"].kind, found["no_submission"].reason) == (
        "unreadable", f"main's tree names y.py as a file or symbolic link, but its object is a {target}")
    assert found["files"] == ["no_submission.json", "record.json"]


@pytest.mark.parametrize("target", ["blob", "tree", "commit", "missing", "main-blob", "main-tree", "main-commit"])
def test_a_submodule_is_bundled_and_checked_out_whatever_its_id_names(tmp_path, target):
    """A submodule's entry names a commit of another repository, which this one's store may or
    may not hold, and git's checkout only makes an empty directory for it, reading nothing: one
    naming a blob, a tree or a commit of the store, or nothing in it, is bundled and checked out.
    The commit point never looks its id up (``FileIds``)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    kind_tree(ws, "160000", target)
    code, found = _run(ws, tmp_path / "out")
    assert code == 0, found["stderr"]
    record = _checkout(tmp_path / "out" / "source.bundle", tmp_path / "tree")
    assert record.status == "ok" and record.files == ["pyproject.toml", "server.py", "y.py"]
    assert (tmp_path / "tree" / "y.py").is_dir() and list((tmp_path / "tree" / "y.py").iterdir()) == []


def test_the_check_reads_the_ids_of_files_and_links_and_never_a_submodules():
    """The check of main's tree's objects is fed the id of each file and symbolic link as the
    listing streams, and keeps the start of its path for the reason; a directory's and a
    submodule's id are never looked up: each id git cannot find costs it a rescan of every pack
    directory, and a submodule's names a commit of another repository."""
    ids = io.BytesIO()
    files = commit.FileIds(ids)
    long = b"d/" * 150 + b"f.py"
    records = [TREE_RECORD + b"pkg", FILE_RECORD + b"pkg/mod.py", b"120000 blob " + b"3" * 40 + b"\tlink",
               b"160000 commit " + b"4" * 40 + b"\tvendor/sub", b"100755 blob " + b"5" * 40 + b"\t" + long]
    assert commit.listing_problem([b"".join(r + b"\0" for r in records)], files) is None
    assert ids.getvalue() == b"".join(c * 40 + b"\n" for c in (b"2", b"3", b"5"))
    assert files.paths == [b"pkg/mod.py", b"link", long[:commit.NAME_SHOWN_CHARS]]


@pytest.mark.parametrize("chunk", [1, 7, 4096])
def test_the_check_reads_each_files_size_whatever_its_chunks(chunk):
    out = b"blob 12\nblob 0\nblob 4294967296\n"
    pieces = (out[at:at + chunk] for at in range(0, len(out), chunk))
    assert commit.blob_sizes(pieces, [b"a", b"b", b"c"]) == [12, 0, 2**32]


def test_the_check_names_the_first_file_whose_object_is_not_a_blob():
    missing = b"1" * 40 + b" missing\n"
    assert commit.blob_sizes([b"blob 3\ntree 40\n" + missing], [b"a", b"pkg/b", b"c"]) == (
        "main's tree names pkg/b as a file or symbolic link, but its object is a tree")
    assert commit.blob_sizes([b"blob 3\n" + missing], [b"a", b"pkg/c\xff"]) == (
        "main's tree names pkg/c\ufffd as a file or symbolic link, but its object is missing")
    # Output that ends early is judged by git's exit status, which the caller reads.
    assert commit.blob_sizes([b"blob 1\n"], [b"a", b"b"]) == [1]


@pytest.mark.parametrize("out, words", [
    (b"blob 1\nblob 2\n", "a line past the 1 ids it was fed"),
    (b"blob " + b"9" * 100, "a line longer than 64 bytes"),
    (b"blob x\n", "invalid literal"),
], ids=["more-lines", "long-line", "no-size"])
def test_the_check_refuses_output_git_never_writes(out, words):
    with pytest.raises(ValueError, match=words):
        commit.blob_sizes([out], [b"a"])


def test_the_check_past_its_time_ends_git_and_the_try_is_infra(tmp_path, monkeypatch):
    """git's check of the objects main's tree names is timed (``KIND_CHECK_SEC``): past it git's
    session is ended and the try is infra, with no bundle kept."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    pid = tmp_path / "git.pid"
    real = subprocess.run(["sh", "-c", "command -v git"], capture_output=True, text=True, check=True).stdout.strip()
    fake = tmp_path / "bin" / "git"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\ncase " $* " in *" cat-file "*) echo $$ > {pid}; exec sleep 60;; esac\n'
                    f'exec {real} "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    monkeypatch.setattr(commit, "KIND_CHECK_SEC", 2)
    out = tmp_path / "out" / "source.bundle"
    (tmp_path / "tmp").mkdir()
    started = time.monotonic()
    with pytest.raises(commit._Infra, match="git cat-file took longer than the 2 s it was given"):
        commit.commit(ws, out, tmp=tmp_path / "tmp", max_bytes=MAX, seconds=SECONDS)
    assert time.monotonic() - started < FAST_GIT_SEC
    hung = int(pid.read_text())
    try:
        assert not Path(f"/proc/{hung}").exists(), "git was left running"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(hung, signal.SIGKILL)
    assert list(out.parent.iterdir()) == []


def _forged_commit_graph(tmp_path: Path) -> Path:
    """A workspace whose main is one commit over a tree of ``server.py`` (T1), beside another tree
    the store holds (T2, ``server.py`` and ``other.py``), whose commit-graph, written by git, names
    T2 as main's tree. git's traversal takes a commit's tree from the graph without reading the
    commit, while the commit, and git's listing of main, name T1."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('v1')\n"})
    first = _git(ws, "rev-parse", "HEAD^{tree}")
    (ws / "other.py").write_text("print('other')\n")
    _git(ws, "add", "other.py")
    other = _git(ws, "write-tree")
    _git(ws, "commit-graph", "write", "--reachable")
    graph = ws / ".git" / "objects" / "info" / "commit-graph"
    data = graph.read_bytes()
    assert data.count(bytes.fromhex(first)) == 1
    graph.chmod(0o644)
    graph.write_bytes(data.replace(bytes.fromhex(first), bytes.fromhex(other)))
    return ws


def test_a_forged_commit_graph_never_changes_what_is_bundled(tmp_path):
    """git reads every commit-graph of the object stores it is given, the workspace's among
    them, and its traversal takes a commit's tree from the graph without reading the commit: a
    graph naming another tree makes git bundle that tree's objects in place of main's, a bundle
    the package's checkout then fails on. The commit point's git reads no
    commit-graph (``GIT_FLAGS``): it bundles main's own tree, which the package checks out."""
    ws = _forged_commit_graph(tmp_path)
    code, found = _run(ws, tmp_path / "out")
    assert code == 0, found["stderr"]
    record = _checkout(tmp_path / "out" / "source.bundle", tmp_path / "tree")
    assert record.status == "ok" and record.files == ["server.py"]
    assert {"core.commitGraph=false", "core.multiPackIndex=false", "pack.useBitmaps=false",
            "pack.readReverseIndex=false"} <= set(commit.GIT_FLAGS)


@pytest.mark.parametrize("lie", sorted([*LIES, "commit-graph"]))
def test_a_bundle_git_made_from_a_lying_store_is_kept_and_the_packages_checkout_refuses_it(tmp_path, monkeypatch,
                                                                                            lie):
    """git trusts the files of the object stores it bundles from, which the contestant wrote, so
    its bundle may not hold what main's commit names, or may hold what git cannot read: a pack's
    index or a loose object's file can name one object at another's bytes (a blob, a tree, a
    parent commit), a pack's entry can hold a deflate stream or a delta git cannot read, and a
    commit-graph, read here as git would read it unbidden, can name another tree
    (``tests.package.bundles.lying_repo``). The commit point writes the bundle once and reads
    nothing back: it keeps the bundle git made. The package's checkout is the first to read it as
    git reads it; it proves every byte it gives git has the bundle's digest, so git's verdict on
    it is the submission's: over_limit, the verdict in the reason, nothing checked out."""
    if lie == "commit-graph":
        ws = _forged_commit_graph(tmp_path)
        settings = [commit.GIT_FLAGS[at:at + 2] for at in range(0, len(commit.GIT_FLAGS), 2)]
        assert all(flag == "-c" for flag, _ in settings)
        monkeypatch.setattr(commit, "GIT_FLAGS", tuple(part for setting in settings
                                                       if setting[1] != "core.commitGraph=false" for part in setting))
        verdict = r"bad tree object [0-9a-f]{40}"
    else:
        ws = tmp_path / "ws"
        lying_repo(ws, lie)
        verdict = LIES[lie]
    assert _main(ws, tmp_path) == (0, {})
    bundle = tmp_path / "out" / "source.bundle"
    assert _heads(bundle) == [f"{_git(ws, 'rev-parse', 'main')} refs/heads/main"]
    source = _checkout(bundle, tmp_path / "tree")
    assert (source.status, source.files) == ("over_limit", [])
    assert re.fullmatch(f"the bundle does not read back as git reads it: {verdict}", source.reason or ""), \
        source.reason
    assert not (tmp_path / "tree").exists()


def test_the_bundle_is_written_once_and_never_read_back(tmp_path, monkeypatch):
    """The commit point's git reads the workspace's objects and writes the bundle to the store
    once, hashed as it is written; nothing reads it back there, neither git (no index-pack, no
    rev-list) nor the program: a bundle made unreadable once written is kept all the same, reported
    by the digest of git's stream and its size."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n", "pkg/mod.py": "x = 1\n"})
    started: list[list[str]] = []
    real_popen, real_bundle = subprocess.Popen, commit._bundle

    def spy(argv, *args, **kwargs):
        started.append([str(a) for a in argv])
        return real_popen(argv, *args, **kwargs)

    def unreadable(fresh, out, *args, **kwargs):
        written = real_bundle(fresh, out, *args, **kwargs)
        out.chmod(0o200)
        return written

    monkeypatch.setattr(commit.subprocess, "Popen", spy)
    monkeypatch.setattr(commit, "_bundle", unreadable)
    out, report = tmp_path / "out", tmp_path / "role" / "outputs-report.json"
    report.parent.mkdir()
    code = commit.main(["--workspace", str(ws), "--out", str(out / "source.bundle"), "--no-submission",
                        str(out / "no_submission.json"), "--outputs-out", str(report), "--tmp", str(tmp_path),
                        "--max-bytes", str(MAX), "--seconds", str(SECONDS)])
    assert code == 0
    verbs = [commit._verb(argv[1 + len(commit.GIT_FLAGS):]) for argv in started]
    assert verbs == ["init", "update-ref", "bundle", "ls-tree", "cat-file"]
    (out / "source.bundle").chmod(0o600)
    reported = read_outputs_report(report)
    assert reported is not None
    assert (reported["source.bundle"].sha256, reported["source.bundle"].size) == sha256_file(out / "source.bundle")


#: How much a git below leaves in its messages' file before its last line: a sparse file, so it
#: takes no disk, past any memory a whole read of it could take.
SPARSE_MESSAGES = "4T"


@pytest.mark.parametrize("command", ["bundle create -", "ls-tree", "cat-file"])
def test_a_failed_gits_messages_are_read_from_their_end_alone(tmp_path, monkeypatch, command):
    """Each git of the fresh repository writes its messages to a file, of which a failure's reason
    reads only the last ``MESSAGE_BYTES`` (``git_messages``): here git leaves terabytes of them,
    then its last line, and fails, and the reason is that line."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    real = subprocess.run(["sh", "-c", "command -v git"], capture_output=True, text=True, check=True).stdout.strip()
    fake = tmp_path / "bin" / "git"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\ncase " $* " in *" {command} "*) truncate -s {SPARSE_MESSAGES} /proc/self/fd/2; '
                    f'printf "\\nfatal: the last line\\n" >> /proc/self/fd/2; exit 128;; esac\nexec {real} "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    code, record = _main(ws, tmp_path)
    assert (code, record["kind"], record["reason"]) == (
        UNREADABLE_EXIT, "unreadable", "main's history cannot be read from the workspace's repository: fatal: the "
                                       "last line")


def test_gits_messages_are_their_last_bytes_as_text(tmp_path):
    path = tmp_path / "git.err"
    with open(path, "wb") as f:
        f.truncate(2**42)
        f.seek(0, os.SEEK_END)
        f.write(b"x" * 5000 + b"\xff fatal: the end\n")
    assert commit.git_messages(path) == ("x" * 5000 + "\ufffd fatal: the end\n")[-commit.MESSAGE_BYTES:]
    assert len(commit.git_messages(path).encode()) == commit.MESSAGE_BYTES + 2
    (tmp_path / "short.err").write_bytes(b"fatal: short\n")
    assert commit.git_messages(tmp_path / "short.err") == "fatal: short\n"


def _listing(records: list[bytes], chunk: int) -> tuple[str | None, list[tuple[bytes, bytes]]]:
    """``listing_problem`` over *records* joined as git writes them, *chunk* bytes at a time, and
    what it kept."""
    data = b"".join(r + b"\0" for r in records)
    kept: list[tuple[bytes, bytes]] = []
    pieces = (data[at:at + chunk] for at in range(0, len(data), chunk))
    return commit.listing_problem(pieces, lambda meta, path: kept.append((meta, path))), kept


TREE_RECORD = b"040000 tree " + b"1" * 40 + b"\t"
FILE_RECORD = b"100644 blob " + b"2" * 40 + b"\t"


@pytest.mark.parametrize("chunk", [1, 5, 4096])
def test_the_listing_is_read_a_record_at_a_time_whatever_its_chunks(chunk):
    """Records cross the ends of the chunks git's output arrives in; each is judged whole, every
    record but a directory's is kept, and a directory's path counts toward the bound on paths'
    length as a file's does."""
    records = [TREE_RECORD + b"pkg", FILE_RECORD + b"pkg/mod.py", FILE_RECORD + b"name with\ttab"]
    assert _listing(records, chunk) == (None, [(FILE_RECORD[:-1], b"pkg/mod.py"), (FILE_RECORD[:-1],
                                                                                  b"name with\ttab")])


def test_a_record_longer_than_any_within_the_bound_is_judged_before_it_ends(monkeypatch):
    """A record is held only up to the most a record within the bound on paths' length holds
    (``LISTING_META_MAX_BYTES`` and ``TREE_MAX_PATH_BYTES``): a path git is still writing past it
    is too long, judged from its start, so no record is ever held whole."""
    monkeypatch.setattr(commit, "TREE_MAX_PATH_BYTES", 8)
    record = FILE_RECORD + b"x" * 10_000
    read = []

    def chunks():
        for at in range(0, len(record), 64):
            read.append(64)
            yield record[at:at + 64]

    problem = commit.listing_problem(chunks(), None)
    assert re.fullmatch("the tree tracks a path longer than the 8 bytes allowed: x{9,200}", problem or "")
    assert sum(read) <= commit.LISTING_META_MAX_BYTES + 8 + 64
    assert commit.WALK_ROOM_ENTRIES >= commit.ALTERNATES_MAX * (256 + 2)


@pytest.mark.parametrize("past", [False, True], ids=["room-for-the-rest", "room-short"])
def test_a_repository_of_loose_objects_just_under_objects_max_passes_the_walk(tmp_path, monkeypatch, past):
    """The count of objects is the bound a repository of loose objects meets first: the walk's
    bound is ``OBJECTS_MAX`` with room for the rest of the object store and the refs beside it.
    Scaled down to a repository of a few hundred loose objects, one under ``OBJECTS_MAX``, room
    for exactly its other entries passes the walk, and less does not."""
    ws = _repo(tmp_path / "ws", {f"f{i}.txt": f"{i}\n" for i in range(300)})
    fields = dict(line.split(": ", 1) for line in _git(ws, "count-objects", "-v").splitlines())
    held = int(fields["count"])
    assert int(fields["in-pack"]) == 0 and held > 300
    others = _walked(ws / ".git") - held
    monkeypatch.setattr(commit, "OBJECTS_MAX", held + 1)
    monkeypatch.setattr(commit, "WALK_ROOM_ENTRIES", others - 2 if past else others)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"more than {held + others - 1} entries" in record["reason"]
    else:
        assert code == 0 and record == {}


def _objects(repo: Path) -> int:
    """The objects *repo* holds itself, loose and packed, as git counts them."""
    fields = dict(line.split(": ", 1) for line in _git(repo, "count-objects", "-v").splitlines())
    return int(fields["count"]) + int(fields["in-pack"])


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
@pytest.mark.parametrize("where", ["loose-and-packed", "an-alternate-store"])
def test_a_repository_of_more_objects_than_the_harness_bundles_is_over_limit(tmp_path, monkeypatch, where, past):
    """git's bundle of a history of tens of millions of objects would outlast its time, an end
    that would be ours. The objects are counted before git runs, loose ones from the object
    stores' fan-out directories and packed ones from each pack index's header, across the
    workspace and every alternate store; past ``OBJECTS_MAX`` the repository is over the limit,
    the contestant's, named by its bound."""
    if where == "loose-and-packed":
        ws = _repo(tmp_path / "ws", {"server.py": "print('v1')\n", "a.py": "x\n"})
        _git(ws, "repack", "-a", "-d", "-q")
        (ws / "server.py").write_text("print('v2')\n")
        _git(ws, "commit", "-qam", "v2")
        held = _objects(ws)
        fields = dict(line.split(": ", 1) for line in _git(ws, "count-objects", "-v").splitlines())
        assert int(fields["count"]) > 0 and int(fields["in-pack"]) > 0
    else:
        base = tmp_path / "ws" / "vendor" / "base"
        ws = _borrowing(tmp_path, base, str(base / ".git" / "objects"))
        held = _objects(ws) + _objects(base)
        assert _objects(base) > 0
    bound = held - 1 if past else held
    monkeypatch.setattr(commit, "OBJECTS_MAX", bound)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"holds {held} objects, more than {bound} objects" in record["reason"]
    else:
        assert code == 0 and record == {}


def _empty_packs(store: Path, n: int) -> None:
    """Add *n* packs of no objects to the object store *store*, each an index and its pack under
    a name of its own, as git opens them; and an index with no pack beside it and a pack with no
    index, which git never opens."""
    made = store.parent / "empty-pack"
    made.mkdir(exist_ok=True)
    # git writes the pack in its repository's object store and renames it to *made*, so it runs in
    # the repository that holds *store*, on the same filesystem.
    name = subprocess.run(["git", "--git-dir", str(store.parent), "pack-objects", "-q", str(made / "empty")],
                          input=b"", capture_output=True, check=True,
                          env=_env(made / ".home")).stdout.decode().strip()
    index, pack = (made / f"empty-{name}.idx").read_bytes(), (made / f"empty-{name}.pack").read_bytes()
    (store / "pack").mkdir(exist_ok=True)
    for i in range(n):
        (store / "pack" / f"pack-{i:040x}.idx").write_bytes(index)
        (store / "pack" / f"pack-{i:040x}.pack").write_bytes(pack)
    (store / "pack" / f"pack-{n:040x}.idx").write_bytes(index)
    (store / "pack" / f"pack-{n + 1:040x}.pack").write_bytes(pack)


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
@pytest.mark.parametrize("where", ["its-own-store", "an-alternate-store"])
def test_a_repository_in_more_packs_than_the_harness_reads_is_over_limit_before_git(tmp_path, monkeypatch, where,
                                                                                   past):
    """git looks each object it reads up in its packs, most recently used first, so a repository
    whose objects cycle through many packs has git search every pack for each, and git's
    per-process work grows with them too, so enough empty packs take git's listing past its time.
    The packs git opens, each an index with its pack beside it,
    are counted over every object store before git runs; past ``PACKS_MAX`` the repository is over
    the limit, the contestant's, named by its bound, and no git runs on it."""
    if where == "its-own-store":
        ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
        _git(ws, "repack", "-a", "-d", "-q")
        store = ws / ".git" / "objects"
    else:
        base = tmp_path / "ws" / "vendor" / "base"
        ws = _borrowing(tmp_path, base, str(base / ".git" / "objects"))
        _git(base, "repack", "-a", "-d", "-q")
        store = base / ".git" / "objects"
    _empty_packs(store, 3)
    monkeypatch.setattr(commit, "PACKS_MAX", 3 if past else 4)
    started: list[list[str]] = []
    real = subprocess.Popen

    def spy(argv, *args, **kwargs):
        started.append([str(a) for a in argv])
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(commit.subprocess, "Popen", spy)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert record["reason"] == "the repository's objects lie in 4 packs, more than the 3 the harness reads"
        assert started == []
    else:
        assert code == 0 and record == {}


def _measured(repo: Path) -> tuple[int, int, int]:
    """What the program's size scan finds in *repo*'s own object store (``object_sizes``): what
    its objects inflate to, what its tree objects hold and what rebuilding its deltas reads."""
    sized = commit.object_sizes([repo / ".git" / "objects"], repo, MAX)
    assert not isinstance(sized, str), sized
    return sized


def _inflated(repo: Path) -> int:
    """The bytes every object *repo* holds itself inflates to, as git reports each one's size: a
    delta's is the object it rebuilds."""
    return sum(size for _, size in _sizes(repo).values())


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
def test_a_repository_whose_objects_inflate_past_the_bound_is_over_limit_and_never_bundled(tmp_path, monkeypatch,
                                                                                           past):
    """git inflates every object it bundles, and the package's clone and scan inflate them again,
    so a few GiB of objects that inflate to terabytes would outlast their time, an end that would
    be ours. The sizes are read from each object's header before git runs, over the whole object
    store, loose and packed, history and objects nothing reaches among them; past
    the cap its caller names (``--max-bytes``) together the repository is over the limit, the contestant's,
    with the figure, and no bundle is written. At the bound it is bundled."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('v1')\n", "a.py": "x\n"})
    _git(ws, "repack", "-a", "-d", "-q")
    (ws / "server.py").write_text("print('v2')\n" + "#" * 5000 + "\n")
    _git(ws, "commit", "-qam", "v2")
    (tmp_path / "loose.txt").write_text("nothing reaches this\n" * 400)
    _git(ws, "hash-object", "-w", str(tmp_path / "loose.txt"))
    fields = dict(line.split(": ", 1) for line in _git(ws, "count-objects", "-v").splitlines())
    assert int(fields["count"]) > 0 and int(fields["in-pack"]) > 0
    total = _inflated(ws)
    bound = total - 1 if past else total
    code, record = _main(ws, tmp_path, bound)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"inflate to {total} bytes together, more than {bound} bytes" in record["reason"]
        assert not (tmp_path / "out" / "source.bundle").exists()
    else:
        assert code == 0 and record == {} and (tmp_path / "out" / "source.bundle").is_file()


@pytest.mark.parametrize("offsets", [True, False], ids=["bases-by-offset", "bases-by-id"])
def test_a_delta_counts_the_size_of_the_object_it_rebuilds_not_its_own(tmp_path, monkeypatch, offsets):
    """A delta stored in a few bytes rebuilds an object of many, which git inflates whole: it
    counts toward the bound as the object it rebuilds, not as what the pack stores."""
    ws = _revisions(tmp_path, offsets=offsets)
    sizes, deltas = _sizes(ws), _deltas(ws)
    total = _inflated(ws)
    stored = sum(deltas.get(name, size) for name, (_, size) in sizes.items())
    assert stored < total, "the deltas are smaller than what they rebuild"
    code, record = _main(ws, tmp_path, total - 1)
    assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
    assert f"inflate to {total} bytes together" in record["reason"]


def test_every_builds_cap_bounds_its_repository():
    """A repository is held to its build's writable disk cap, from the owner the sandbox and the
    brief take it from, whatever the cap."""
    caps = {mod.__name__.split(".")[1]: sandbox_disk_gib(mod.TASK, mod.ARTIFACT.producer) for mod in _task_modules()
            if isinstance(getattr(mod, "ARTIFACT", None), AgentProducedArtifact)}
    assert len(caps) == 8 and set(caps.values()) == {32, 180, 240, 1911}
    assert not hasattr(commit, "DISK_CAP_MAX_GIB") and not hasattr(commit, "check_disk_cap")
    # No other figure bounds a repository: its cap is the build's writable disk cap.
    assert not hasattr(commit, "OBJECTS_INFLATED_MAX_BYTES") and not hasattr(commit, "BUNDLE_MAX_BYTES")


def _task_modules():
    """Every task module of the repository's tasks/ that loads."""
    found = []
    for task in sorted((Path(__file__).resolve().parents[2] / "tasks").glob("*/task.py")):
        found.append(importlib.import_module(f"tasks.{task.parent.name}.task"))
    return found


# --- a bundle past the package's limit is no submission at the commit point -----


def test_the_package_checkout_holds_a_bundle_to_the_cap_its_caller_names():
    """The package's checkout holds a bundle, and the objects it inflates to, to the cap its stage
    names (``--max-bytes``), the build's writable disk cap the commit point held the repository
    to; no figure of its own."""
    tree = ast.parse(Path(checkout.__file__).read_text())
    assigned = {target.id for node in tree.body if isinstance(node, ast.Assign | ast.AnnAssign)
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                if isinstance(target, ast.Name)}
    assert not {"BUNDLE_MAX_BYTES", "OBJECTS_INFLATED_MAX_BYTES", "INFLATED_MAX_BYTES"} & assigned


@pytest.mark.parametrize("sent, limit, kept", [(10_000, 4096, 4097), (4096, 4096, 4096), (100, 4096, 100)],
                         ids=["past", "at-the-limit", "under"])
def test_a_bundle_is_copied_through_a_budget_of_its_limit_and_one_byte(tmp_path, sent, limit, kept):
    r, w = os.pipe()
    os.write(w, b"x" * sent)
    os.close(w)
    out = tmp_path / "out"
    try:
        with open(out, "wb") as dest:
            assert commit._copy_bounded(r, dest, limit=limit, deadline=time.monotonic() + 30, what="git bundle", seconds=30) == kept
    finally:
        os.close(r)
    assert out.stat().st_size == kept


def test_a_bundle_that_reaches_the_limit_in_one_read_and_goes_on_is_past_it(tmp_path):
    """A read that ends exactly at the limit does not end the copy: one more byte is waited for,
    so a longer bundle is never taken for a whole one of the limit's size."""
    r, w = os.pipe()

    def write() -> None:
        os.write(w, b"x" * 4096)  # one atomic write, read whole
        time.sleep(0.3)
        os.write(w, b"y" * 100)
        os.close(w)

    writer = threading.Thread(target=write)
    writer.start()
    try:
        with open(tmp_path / "out", "wb") as dest:
            assert commit._copy_bounded(r, dest, limit=4096, deadline=time.monotonic() + 30, what="git bundle", seconds=30) == 4097
    finally:
        writer.join()
        os.close(r)


#: The time the program is given in the tests of what it does with its time, so a program that
#: never ends git fails them in seconds rather than after the default hour.
FAST_GIT_SEC = 10


def _bundling_git(tmp_path: Path, monkeypatch, body: str) -> None:
    """Put a git first on PATH that runs the shell *body* for ``bundle create -`` and the real git
    for anything else."""
    real = subprocess.run(["sh", "-c", "command -v git"], capture_output=True, text=True, check=True).stdout.strip()
    fake = tmp_path / "bin" / "git"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\ncase " $* " in *" bundle create - "*) {body};; esac\nexec {real} "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")


def test_a_bundle_far_past_the_limit_stops_git_there(tmp_path, monkeypatch):
    """One byte past the limit the program ends git's session: git never writes the rest of a
    bundle that could be as large as the sandbox's disk, and the program keeps none of it."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    finished = tmp_path / "finished"
    _bundling_git(tmp_path, monkeypatch, f"head -c {64 * 2**20} /dev/zero && touch {finished}; exit 0")
    code, record = _main(ws, tmp_path, 4096, seconds=FAST_GIT_SEC)
    assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
    assert not finished.exists(), "git wrote the whole bundle"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["no_submission.json"]


def test_a_git_that_closes_its_output_and_hangs_is_ended_and_the_try_is_infra(tmp_path, monkeypatch):
    """git that ends its output and never exits is waited for at most the program's time
    left (``--seconds``); then its session is ended, never left running beside the program, and the
    try is infra with no bundle kept."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    pid = tmp_path / "git.pid"
    _bundling_git(tmp_path, monkeypatch, f"head -c 100 /dev/zero; exec >&-; echo $$ > {pid}; exec sleep 60")
    out = tmp_path / "out" / "source.bundle"
    (tmp_path / "tmp").mkdir()
    started = time.monotonic()
    with pytest.raises(commit._Infra, match=r"git bundle took longer than the [12] s it was given"):
        commit.commit(ws, out, tmp=tmp_path / "tmp", max_bytes=MAX, seconds=2)
    assert time.monotonic() - started < FAST_GIT_SEC
    hung = int(pid.read_text())
    try:
        assert not Path(f"/proc/{hung}").exists(), "git was left running"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(hung, signal.SIGKILL)
    assert list(out.parent.iterdir()) == []


def test_the_programs_reads_spend_its_seconds_and_a_git_with_none_left_is_not_started(tmp_path, monkeypatch):
    """The program's time is counted from its start: its own reads of the repository spend it, and
    a git command with no time left is never started; the try is infra, exit 1, its stderr naming
    the command, with nothing written."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    started: list[list[str]] = []
    real_popen = commit.subprocess.Popen

    def popen(args, *a, **kw):
        started.append(list(args))
        return real_popen(args, *a, **kw)

    real_check = commit.check_repository

    def slow_check(workspace, max_bytes):
        time.sleep(1.5)
        return real_check(workspace, max_bytes)

    monkeypatch.setattr(commit.subprocess, "Popen", popen)
    monkeypatch.setattr(commit, "check_repository", slow_check)
    code, record = _main(ws, tmp_path, seconds=1)
    assert (code, record) == (1, {})
    assert started == [], "a git was started with no time left"
    assert not (tmp_path / "out" / "source.bundle").exists()


def test_no_time_left_is_infra_naming_what_it_was_for():
    with pytest.raises(commit._Infra, match="no time was left for git init: the program's --seconds were spent"):
        commit._left(time.monotonic() - 0.001, "git init")
    assert 0 < commit._left(time.monotonic() + 5, "git init") <= 5


@pytest.mark.parametrize("seconds, listing, check", [(SECONDS, commit.TREE_LIST_SEC, commit.KIND_CHECK_SEC),
                                                     (30, 30, 30)], ids=["plenty-left", "little-left"])
def test_the_listing_and_the_check_get_the_lesser_of_their_bound_and_the_time_left(tmp_path, monkeypatch, seconds,
                                                                                    listing, check):
    """git's listing of main's tree and its check of the objects it names keep their own bounds
    (``TREE_LIST_SEC``, ``KIND_CHECK_SEC``), each cut to the program's time left when that is less."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    given: dict[str, float] = {}
    real = commit._streamed

    def streamed(args, env, workspace, *, errors, seconds, read, stdin=None):
        given[commit._verb(args)] = seconds
        return real(args, env, workspace, errors=errors, seconds=seconds, read=read, stdin=stdin)

    monkeypatch.setattr(commit, "_streamed", streamed)
    (tmp_path / "tmp").mkdir()
    assert commit.commit(ws, tmp_path / "out" / "source.bundle", tmp=tmp_path / "tmp", max_bytes=MAX,
                         seconds=seconds).status == "ok"
    assert set(given) == {"ls-tree", "cat-file"}
    assert listing - 5 < given["ls-tree"] <= listing and check - 5 < given["cat-file"] <= check


def test_a_bundle_copy_that_outlasts_its_time_is_infra(tmp_path):
    r, w = os.pipe()
    closed = threading.Event()

    def close() -> None:
        os.close(w)
        closed.set()

    # The pipe ends after a few seconds, so a copy that never looks at its deadline returns and
    # fails here instead of waiting on the pipe for good.
    closer = threading.Timer(5, close)
    closer.start()
    try:
        with open(tmp_path / "out", "wb") as dest, pytest.raises(commit._Infra, match="git bundle took longer"):
            commit._copy_bounded(r, dest, limit=4096, deadline=time.monotonic() + 0.2, what="git bundle", seconds=1)
    finally:
        closer.cancel()
        closer.join()
        if not closed.is_set():
            os.close(w)
        os.close(r)


def test_the_bundle_packs_with_no_delta_search_and_no_compression():
    """git's delta search and its compression made the bundle's time depend on what the objects
    hold; with neither, it is the objects' bytes, read and written once. No setting of the delta
    search is left, since none has any effect without one."""
    flags = commit.GIT_FLAGS
    for setting in ("pack.window=0", "pack.compression=0"):
        assert flags[flags.index(setting) - 1] == "-c"
    assert not [f for f in flags if f.startswith(("pack.threads", "pack.windowMemory", "pack.deltaCacheSize"))]


def _pack_deltas(bundle: Path, tmp_path: Path) -> int:
    """How many objects of *bundle*'s pack are stored as deltas, as git's verify-pack reads them."""
    data = bundle.read_bytes()
    pack = tmp_path / "check" / "objects.pack"
    pack.parent.mkdir()
    pack.write_bytes(data[data.index(b"\n\n") + 2:])
    _git(pack.parent, "index-pack", str(pack))
    listed = _git(pack.parent, "verify-pack", "-v", str(pack.with_suffix(".idx")))
    return sum(1 for line in listed.splitlines() if len(line.split()) == 7)


def test_the_bundle_holds_no_delta_the_harness_computed_and_its_objects_uncompressed(tmp_path):
    """Revisions of a file that differ by a line would delta to a few bytes; the commit point
    stores each whole and uncompressed, so the bundle holds about the objects' own bytes."""
    ws = _repo(tmp_path / "ws", {"server.py": "x\n", "data.txt": "".join(f"line {i}\n" for i in range(3000))})
    for n in range(4):
        (ws / "data.txt").write_text("".join(f"line {i}\n" for i in range(3000)) + f"rev {n}\n")
        _git(ws, "commit", "-qam", f"rev {n}")
    out = tmp_path / "out" / "source.bundle"
    (tmp_path / "tmp").mkdir()
    assert commit.commit(ws, out, tmp=tmp_path / "tmp", max_bytes=MAX, seconds=SECONDS).status == "ok"
    assert _pack_deltas(out, tmp_path) == 0
    assert out.stat().st_size >= _inflated(ws)


def test_deltas_the_contestants_own_packs_hold_are_kept_as_they_are(tmp_path):
    """A delta already in the repository's own pack is the contestant's: git reuses it, unread."""
    ws = _revisions(tmp_path, offsets=True)
    assert _deltas(ws)
    out = tmp_path / "out" / "source.bundle"
    (tmp_path / "tmp").mkdir()
    assert commit.commit(ws, out, tmp=tmp_path / "tmp", max_bytes=MAX, seconds=SECONDS).status == "ok"
    assert _pack_deltas(out, tmp_path) > 0


def _bundle_size(ws: Path, tmp_path: Path) -> int:
    """The size of the bundle the program makes of *ws*'s main under a cap far past it."""
    out = tmp_path / "sized"
    assert commit.main(["--workspace", str(ws), "--out", str(out / "source.bundle"), "--tmp", str(tmp_path), "--max-bytes", str(MAX),
            "--seconds", str(SECONDS)]) == 0
    return (out / "source.bundle").stat().st_size


@pytest.mark.parametrize("past", [False, True], ids=["at-the-limit", "one-past"])
def test_a_bundle_past_the_packages_limit_is_no_submission_at_the_commit_point(tmp_path, past):
    """The program writes git's bundle through a byte budget, so it never stores a bundle the
    package's checkout would refuse; past the cap it is given it keeps none of it and records
    over_limit, the contestant's nothing-gradable end, the one the package reaches for such a
    bundle. A bundle of exactly the limit is kept, as the package keeps it."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n", "weights.txt": os.urandom(2**18).hex()})
    size = _bundle_size(ws, tmp_path)
    bound = size - 1 if past else size
    # The cap bounds what the objects inflate to as well, which here is less than the bundle.
    assert _inflated(ws) <= bound
    code, record = _main(ws, tmp_path, bound, seconds=FAST_GIT_SEC)
    left = sorted(p.name for p in (tmp_path / "out").iterdir())
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"more than {bound} bytes" in record["reason"]
        # Neither the bundle nor its partial file is kept.
        assert left == ["no_submission.json"]
    else:
        assert code == 0 and record == {} and left == ["source.bundle"]
        assert (tmp_path / "out" / "source.bundle").stat().st_size == size


def test_a_snapshot_past_the_packages_limit_keeps_no_bundle(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n", "weights.txt": os.urandom(2**18).hex()})
    cap = _bundle_size(ws, tmp_path) - 1
    snaps = tmp_path / "snapshots"
    code = commit.main(["--workspace", str(ws), "--out", str(snaps / snapshot_name(1)), "--record",
                        str(tmp_path / "record.json"), "--tmp", str(tmp_path), "--max-bytes", str(cap),
                        "--seconds", str(FAST_GIT_SEC)])
    assert code == UNREADABLE_EXIT
    assert json.loads((tmp_path / "record.json").read_text())["status"] == "over_limit"
    assert list(snaps.iterdir()) == []


def test_a_bundle_on_standard_output_is_the_bundle_git_writes_to_a_file(tmp_path):
    """The program bundles through git's standard output; that is byte for byte what git writes
    to a file of the same repository."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    env = _env(tmp_path / ".home")
    subprocess.run(["git", "-C", str(ws), "bundle", "create", str(tmp_path / "file.bundle"), "refs/heads/main"],
                   env=env, capture_output=True, check=True)
    piped = subprocess.run(["git", "-C", str(ws), "bundle", "create", "-", "refs/heads/main"], env=env,
                           capture_output=True, check=True).stdout
    assert piped == (tmp_path / "file.bundle").read_bytes()


def _big_tree(ws: Path, *, entries: int, present: bool, pack: bool) -> Path:
    """A workspace whose main is one commit holding a tree of *entries* entries. Each entry names
    one empty blob that is in the store (*present*), so the tree bundles, or a blob that is absent,
    so git could not bundle it. With *pack*, the objects are packed, not loose."""
    ws.mkdir(parents=True)
    _git(ws, "init", "-q", "-b", "main")
    empty = _git(ws, "hash-object", "-w", "--stdin")
    if present:
        body = "".join(f"100644 blob {empty}\t{i:x}\n" for i in range(entries)).encode()
        tree = subprocess.run(["git", "-C", str(ws), "mktree"], input=body, env=_env(ws.parent / ".home"),
                              capture_output=True, text=False, check=True).stdout.decode().strip()
    else:
        body = "".join(f"100644 blob {hashlib.sha1(f'{i}'.encode()).hexdigest()}\t{i:x}\n"
                       for i in range(entries)).encode()
        tree = subprocess.run(["git", "-C", str(ws), "mktree", "--missing"], input=body,
                              env=_env(ws.parent / ".home"), capture_output=True, text=False, check=True
                              ).stdout.decode().strip()
    head = _git(ws, "commit-tree", tree, "-m", "big")
    _git(ws, "update-ref", "refs/heads/main", head)
    if pack:
        _pack_every_object(ws)
    return ws


def _pack_every_object(repo: Path) -> None:
    """Pack every object *repo* holds by its id and drop the loose copies. git's repack walks
    history and refuses a tree whose blobs are absent; packing the objects as listed reads each
    one alone, as a contestant's ``git pack-objects`` would."""
    listed = _git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)")
    subprocess.run(["git", "-C", str(repo), "pack-objects", "-q", str(repo / ".git" / "objects" / "pack" / "pack")],
                   input=(listed + "\n").encode(), env=_env(repo.parent / ".home"), capture_output=True, check=True)
    _git(repo, "prune-packed")
    assert not any(d.name != "pack" and d.name != "info" for d in (repo / ".git" / "objects").iterdir()), \
        "every object is packed"


def _tree_bytes(repo: Path) -> int:
    """The bytes of every tree object *repo* holds, together, as git reports each one's size."""
    return sum(size for kind, size in _sizes(repo).values() if kind == "tree")


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
@pytest.mark.parametrize("pack", [False, True], ids=["loose", "packed"])
def test_a_repository_whose_tree_objects_hold_more_bytes_than_the_harness_walks_is_over_limit(tmp_path, monkeypatch,
                                                                                             pack, past):
    """git's bundle walks every entry of every tree reachable from main, the package's scan reads
    them again, and a tree entry naming an absent blob is no object, so it escapes ``OBJECTS_MAX``
    and the inflated bound while git still walks it. The bytes of every tree object are summed from
    each one's header before git runs, loose and packed; past ``TREE_BYTES_MAX`` together the
    repository is over the limit, the contestant's, named by its bound, and no bundle is written.
    At the bound a tree of blobs that are there is bundled."""
    ws = _big_tree(tmp_path / "ws", entries=40000, present=True, pack=pack)
    held = _tree_bytes(ws)
    assert held > 2 * len(_sizes(ws)), "the tree objects dominate the store's bytes"
    monkeypatch.setattr(commit, "TREE_BYTES_MAX", held - 1 if past else held)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"tree objects hold {held} bytes together, more than {held - 1} bytes" in record["reason"]
        assert not (tmp_path / "out" / "source.bundle").exists()
    else:
        assert code == 0 and record == {} and (tmp_path / "out" / "source.bundle").is_file()


@pytest.mark.parametrize("pack", [False, True], ids=["loose", "packed"])
def test_tree_entries_that_name_absent_blobs_count_toward_the_tree_bound_before_git_runs(tmp_path, monkeypatch, pack):
    """A tree of millions of entries each naming a blob that is not in the
    store passes ``OBJECTS_MAX`` (the tree is one object), the inflated bound (the named blobs are
    no objects and inflate to nothing) and the per-object bounds (the tree itself is under
    ``NONBLOB_MAX_BYTES``), yet ``git bundle create`` walks every entry; with the stock timeout a
    big enough history is SIGKILLed, an infra void the contestant chose. The tree's bytes are
    counted whether or not its entries name objects that are there, loose or packed (packed by
    id, as ``git pack-objects`` packs it), so the walk is bounded before git touches the
    repository: the repository is over the limit, the contestant's."""
    ws = _big_tree(tmp_path / "ws", entries=40000, present=False, pack=pack)
    held = _tree_bytes(ws)
    # The named blobs are no objects: the whole store is a handful of objects and inflates to a
    # trifle, so neither OBJECTS_MAX nor the inflated bound would catch it.
    assert len(_sizes(ws)) < 10 and _inflated(ws) < NONBLOB_MAX_BYTES
    monkeypatch.setattr(commit, "TREE_BYTES_MAX", held - 1)
    code, record = _main(ws, tmp_path)
    assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
    assert f"tree objects hold {held} bytes together, more than {held - 1} bytes" in record["reason"]


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
def test_a_tree_stored_as_a_delta_counts_the_tree_it_rebuilds(tmp_path, monkeypatch, past):
    """Revisions of a large directory make similar trees, which a repack stores as deltas of a few
    bytes each; git rebuilds each whole and walks every entry of it, so a tree stored as a delta
    counts toward ``TREE_BYTES_MAX`` as the tree it rebuilds, never as the delta's own bytes."""
    ws = _repo(tmp_path / "ws", {f"big/f{n:04d}.txt": f"{n}\n" for n in range(2000)})
    for n in range(5):
        (ws / "big" / f"f{n:04d}.txt").write_text(f"revision {n}\n")
        _git(ws, "commit", "-qam", f"rev {n}")
    _git(ws, "repack", "-a", "-d", "-f", "-q", "--depth=50", "--window=10")
    sizes, deltas = _sizes(ws), _deltas(ws)
    stored = {name: deltas[name] for name, (kind, _) in sizes.items() if kind == "tree" and name in deltas}
    assert stored and all(deltas[name] * 10 < sizes[name][1] for name in stored), \
        "the large trees are stored as deltas far smaller than what they rebuild"
    held = _tree_bytes(ws)
    bound = held - 1 if past else held
    monkeypatch.setattr(commit, "TREE_BYTES_MAX", bound)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"tree objects hold {held} bytes together, more than {bound} bytes" in record["reason"]
    else:
        assert code == 0 and record == {}


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
def test_a_delta_whose_base_is_outside_its_pack_counts_toward_the_tree_bound(tmp_path, monkeypatch, past):
    """A packed object's kind is found through its delta's base; git writes no pack whose delta
    names a base outside it (a thin pack is completed when it is fetched), so one that does shows
    no kind, and is read as the stricter: a tree, its rebuilt size counted toward
    ``TREE_BYTES_MAX``. git never reads it unless main reaches it; at the bound it is bundled."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    trees = _tree_bytes(ws)
    _crafted_pack(ws / ".git", [("outside",)], declared=5000)
    held = trees + 5000
    bound = held - 1 if past else held
    monkeypatch.setattr(commit, "TREE_BYTES_MAX", bound)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"tree objects hold {held} bytes together, more than {bound} bytes" in record["reason"]
    else:
        assert code == 0 and record == {}


def test_the_tree_bytes_bound_keeps_the_walk_to_minutes():
    # git 2.43.0 bundle walks a history of missing-blob trees at about 3.97 s per MiB of tree.
    # The bound keeps the walk to about 508 s.
    assert commit.TREE_BYTES_MAX == 128 * 2**20
    # Generous over a submission of 1.1 MiB of tree bytes and over the
    # history OBJECTS_MAX is sized for (ten 200,000-path revisions, about 61 MiB of trees).
    assert commit.TREE_BYTES_MAX > 100 * 1_153_320
    assert commit.TREE_BYTES_MAX > 2 * 61 * 2**20


def _index(count: int, *, version: int = 2, fanout: list[int] | None = None, size: int | None = None) -> bytes:
    """A pack index's bytes: its header and fan-out table for *count* objects, padded to the size
    git requires of it (or *size*)."""
    table = fanout if fanout is not None else [count] * 256
    head = (b"\xfftOc" + version.to_bytes(4, "big") if version != 1 else b"") + b"".join(
        n.to_bytes(4, "big") for n in table)
    whole = len(head) + count * (24 if version == 1 else 28) + 40
    return (head + b"\0" * whole)[:whole if size is None else size]


@pytest.mark.parametrize("index, pack, readable", [
    (b"short", True, False),
    (_index(3, fanout=[2] * 10 + [1] + [3] * 245), True, False),
    (_index(5, size=8 + 1024 + 40), True, False),
    (_index(3, version=1, size=1024 + 3 * 24), True, False),
    # Sized as a version 1 index of its count would be, so only its version refuses it.
    (_index(2, version=7, size=256 * 4 + 2 * 24 + 40), True, False),
    (b"short", False, True),
    (_index(3), True, True),
    (_index(3, version=1), True, True),
], ids=["truncated", "non-monotonic", "too-small-for-its-count", "v1-wrong-size", "unknown-version",
        "no-pack-beside-it", "well-formed-v2", "well-formed-v1"])
def test_a_pack_index_git_would_not_read_is_unreadable(tmp_path, monkeypatch, index, pack, readable):
    """A pack index is read for its header alone, through the same rules as every other file of
    the repository. One git would refuse (too short, a fan-out that decreases, a size that is not
    its count's, an unknown version) is the unreadable repository; one with no pack beside it git
    never opens, and neither does the count."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    (ws / ".git" / "objects" / "pack" / "pack-odd.idx").write_bytes(index)
    if pack:
        (ws / ".git" / "objects" / "pack" / "pack-odd.pack").write_bytes(b"PACK")
    monkeypatch.setattr(commit, "_git", lambda args, env: pytest.fail("git ran"))
    if readable:
        monkeypatch.setattr(commit, "OBJECTS_MAX", 0)
        code, record = _main(ws, tmp_path)
        assert record["kind"] == "over_limit"
    else:
        code, record = _main(ws, tmp_path)
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "unreadable")
        assert "objects/pack/pack-odd.idx is not a pack index git reads" in record["reason"]


def _write_loose(git_dir: Path, kind: str, start: bytes, fill: int = 0, *, padding: int = 0) -> str:
    """Write a loose object of *kind* holding *start* and then *fill* bytes of one letter, as
    git would write it (its name is its content's hash), streamed so that no copy of it is ever
    whole in memory. *padding* empty deflate blocks come before everything: git inflates through
    them. Returns its name."""
    head = f"{kind} {len(start) + fill}\0".encode()
    digest = hashlib.sha1(head + start)
    if padding:
        # A zlib header, empty stored blocks, the content as raw deflate, and its checksum.
        raw = zlib.compressobj(1, zlib.DEFLATED, -15)
        body = raw.compress(head + start) + raw.flush()
        data = b"\x78\x01" + b"\x00\x00\x00\xff\xff" * padding + body + zlib.adler32(head + start).to_bytes(4, "big")
        assert not fill and zlib.decompress(data) == head + start
        name = digest.hexdigest()
        (git_dir / "objects" / name[:2]).mkdir(exist_ok=True)
        (git_dir / "objects" / name[:2] / name[2:]).write_bytes(data)
        return name
    deflate = zlib.compressobj(1)
    parts = [deflate.compress(head + start)]
    chunk, left = b"A" * (64 << 20), fill
    while left:
        part = chunk[:min(left, len(chunk))]
        digest.update(part)
        parts.append(deflate.compress(part))
        left -= len(part)
    parts.append(deflate.flush())
    name = digest.hexdigest()
    path = git_dir / "objects" / name[:2] / name[2:]
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"".join(parts))
    return name


GIANT = 2_950_000_000


@pytest.mark.parametrize("kind", ["commit", "tree", "tag"])
def test_an_object_git_loads_whole_past_its_bound_is_over_limit_without_git_running(tmp_path, kind):
    """git loads every commit, tree and tag whole. One of ~2.7 GiB of a repeated letter is a few
    MB on disk, inside every other bound, and would exhaust the commit role's memory, an end that
    would be ours. Its size is read from its header alone, in a process held to 2.6 GiB: the
    repository is over the limit, the contestant's, naming the object's kind and the bound."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    git = ws / ".git"
    main = _git(ws, "rev-parse", "main")
    tree = _git(ws, "rev-parse", "main^{tree}")
    who = b"a <a@x> 1700000000 +0000"
    if kind == "commit":
        name = _write_loose(git, "commit", b"tree %s\nauthor %s\ncommitter %s\n\n" % (tree.encode(), who, who), GIANT)
        (git / "refs" / "heads" / "main").write_text(name + "\n")
    elif kind == "tree":
        giant = _write_loose(git, "tree", b"", GIANT)
        name = _write_loose(git, "commit", b"tree %s\nparent %s\nauthor %s\ncommitter %s\n\nbig\n"
                            % (giant.encode(), main.encode(), who, who))
        (git / "refs" / "heads" / "main").write_text(name + "\n")
    else:
        _write_loose(git, "tag", b"object %s\ntype commit\ntag big\ntagger %s\n\n" % (main.encode(), who), GIANT)
    assert sum(f.stat().st_size for f in (git / "objects").rglob("*") if f.is_file()) < 64 << 20
    code, found = _run(ws, tmp_path / "out", memory_gib=2.6)
    assert code == UNREADABLE_EXIT, found["stderr"]
    record = found["no_submission"]
    assert record.kind == "over_limit"
    assert f"a {kind} of " in record.reason and f"more than {NONBLOB_MAX_BYTES} bytes" in record.reason


def _sizes(repo: Path) -> dict[str, tuple[str, int]]:
    """Each object's type and size, as git reads them."""
    out = _git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype) %(objectsize)")
    return {name: (kind, int(size)) for name, kind, size in (line.split() for line in out.splitlines())}


def _deltas(repo: Path) -> dict[str, int]:
    """Each object stored as a delta in *repo*'s packs, with its delta's own size."""
    found = {}
    for index in (repo / ".git" / "objects" / "pack").glob("*.idx"):
        for line in _git(repo, "verify-pack", "-v", str(index)).splitlines():
            fields = line.split()
            if len(fields) == 7 and len(fields[0]) == 40:
                found[fields[0]] = int(fields[2])
    return found


def _chain_bytes(repo: Path) -> int:
    """What rebuilding every delta of *repo*'s packs reads, as git's own records give it: each
    delta's object and its delta's own bytes, on its base's, a whole object's its own size."""
    sizes, own = _sizes(repo), _deltas(repo)
    base = dict(line.split() for line in _git(repo, "cat-file", "--batch-all-objects",
                                              "--batch-check=%(objectname) %(deltabase)").splitlines())

    def chain(oid: str) -> int:
        if oid not in own:
            return sizes[oid][1]
        return sizes[oid][1] + own[oid] + chain(base[oid])

    return sum(chain(oid) for oid in own)


@pytest.mark.parametrize("offsets", [True, False], ids=["bases-by-offset", "bases-by-id"])
def test_what_rebuilding_every_delta_reads_is_measured_through_its_chain(tmp_path, offsets):
    """git rebuilds a delta from its chain: the object it rebuilds, its delta's own bytes, and the
    same for its base, down to a whole object. The program reads each from the pack entries'
    headers and sums it over every delta of the store, as git's own records give it."""
    ws = _revisions(tmp_path, offsets=offsets)
    expected = _chain_bytes(ws)
    assert expected > 0, "the pack holds deltas"
    assert _measured(ws)[2] == expected


def test_a_delta_met_before_the_delta_it_is_stored_on_counts_that_base_whole(tmp_path):
    """The scan meets a pack's entries in id order, so it can meet a delta before the delta it is
    stored on. Its walk down the chain then passes that base unread so far, whose chain bytes count
    the object it rebuilds as well as its delta's own bytes. A file revised fifty times in place,
    a seventh of its lines each time, and packed by git's own gc has chains up to four deltas deep,
    and some deeper delta sorts first."""
    ws = _repo(tmp_path / "ws", {"server.py": "".join(f"line {n}\n" for n in range(400))})
    for n in range(50):
        (ws / "server.py").write_text("".join(f"line {k} {n if k % 7 == n % 7 else 0}\n" for k in range(400)))
        _git(ws, "commit", "-qam", f"rev {n}")
    _git(ws, "gc", "-q")
    own = _deltas(ws)
    base = dict(line.split() for line in _git(ws, "cat-file", "--batch-all-objects",
                                              "--batch-check=%(objectname) %(deltabase)").splitlines())
    assert [oid for oid in own if base[oid] in own and oid < base[oid]], "a deeper delta sorts first"
    assert _measured(ws)[2] == _chain_bytes(ws)


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
def test_a_repository_whose_deltas_rebuild_past_the_bound_is_over_limit_before_git_runs(tmp_path, monkeypatch,
                                                                                       past):
    """Past ``CHAIN_BYTES_MAX`` together, rebuilding the repository's deltas could run git and the
    package's scan past their time, so the repository is the contestant's over_limit, decided from
    the pack entries before git runs, as the other bounds are. At the bound it is bundled."""
    ws = _revisions(tmp_path, offsets=True)
    total = _chain_bytes(ws)
    monkeypatch.setattr(commit, "CHAIN_BYTES_MAX", total - past)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert (f"rebuilding the repository's deltas reads {total} bytes together, more than the {total - 1}"
                in record["reason"])
        assert not (tmp_path / "out" / "source.bundle").exists()
    else:
        assert (code, record) == (0, {})


def test_the_chain_bound_is_eight_times_the_largest_rebuilding_measured():
    # A submission's history whose deltas rebuild 6,912,972,470 bytes: the bound is more than
    # eight times it.
    assert commit.CHAIN_BYTES_MAX == 64 * 2**30 > 8 * 6_912_972_470
    # git rebuilds deltas of one-byte copies, its slowest, at about 4.22 s per GiB of chain on the
    # commit role's core: about 270 s at the bound.


def _revisions(tmp_path: Path, *, offsets: bool) -> Path:
    """A repository of a file and its history in one pack, its revisions stored as deltas
    against each other, their bases named by offset or by id. With ids, 1,500 small files more,
    30 to a directory so every tree stays smaller than the file's revisions, put several ids
    under each first byte of the index, where a base is searched for."""
    files = {"server.py": "".join(f"line {n}\n" for n in range(400))}
    files |= {} if offsets else {f"data/d{n // 30}/f{n}.txt": f"{n}\n" for n in range(1500)}
    ws = _repo(tmp_path / "ws", files)
    for n in range(6):
        (ws / "server.py").write_text("".join(f"line {k}\n" for k in range(400 + 40 * n)) + f"rev {n}\n")
        _git(ws, "commit", "-qam", f"rev {n}\n\n" + "why " * (20 + n * 5))
    _git(ws, "-c", f"repack.useDeltaBaseOffset={str(offsets).lower()}", "repack", "-a", "-d", "-f", "-q",
         "--depth=50", "--window=10")
    assert _deltas(ws), "the pack holds deltas"
    return ws


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
@pytest.mark.parametrize("offsets", [True, False], ids=["bases-by-offset", "bases-by-id"])
@pytest.mark.parametrize("bound", ["NONBLOB_MAX_BYTES", "OBJECT_MAX_BYTES", "DELTA_MAX_BYTES"])
def test_each_objects_size_is_read_through_its_pack_entry_and_bounded(tmp_path, monkeypatch, bound, offsets,
                                                                         past):
    """A packed object's size is its entry header's, and a delta's is the larger of its own and
    the result its header declares, its kind found through its bases. A commit, tree or tag past
    ``NONBLOB_MAX_BYTES``, any object past ``OBJECT_MAX_BYTES`` and an object stored as a delta
    past ``DELTA_MAX_BYTES`` are over the limit; at each bound the repository is bundled."""
    ws = _revisions(tmp_path, offsets=offsets)
    sizes, deltas = _sizes(ws), _deltas(ws)
    largest = {name: max(size, deltas.get(name, 0)) for name, (_, size) in sizes.items()}
    chosen = {"NONBLOB_MAX_BYTES": [n for n, (k, _) in sizes.items() if k != "blob"],
              "OBJECT_MAX_BYTES": list(sizes), "DELTA_MAX_BYTES": list(deltas)}[bound]
    if bound == "DELTA_MAX_BYTES":
        limit = max(sizes[n][1] for n in chosen)
    else:
        limit = max(largest[n] for n in chosen)
    monkeypatch.setattr(commit, bound, limit - past)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"more than {limit - past} bytes" in record["reason"]
    else:
        assert code == 0 and record == {}


def _type_and_size(kind: int, size: int) -> bytes:
    """A pack entry's header: its type and size, as git encodes them."""
    out, c, size = bytearray(), (kind << 4) | (size & 0x0F), size >> 4
    while size:
        out.append(c | 0x80)
        c, size = size & 0x7F, size >> 7
    return bytes(out + bytes([c]))


def _back(distance: int) -> bytes:
    """An offset delta's distance back to its base, as git encodes it."""
    out = bytearray([distance & 0x7F])
    distance >>= 7
    while distance:
        distance -= 1
        out.insert(0, 0x80 | (distance & 0x7F))
        distance >>= 7
    return bytes(out)


def _number(n: int) -> bytes:
    """A delta header's little-endian base-128 number."""
    out = bytearray()
    while True:
        out.append((n & 0x7F) | (0x80 if n >> 7 else 0))
        n >>= 7
        if not n:
            return bytes(out)


#: A size between the bound on commits, trees and tags and the bound on deltas.
DECLARED = 100 * 2**20


def _crafted_pack(git_dir: Path, entries: list[tuple], *, declared: int = DECLARED) -> int:
    """Write a pack and its index that git never made: each entry ``("blob",)``, a small blob
    whose header declares *declared* bytes; ``("ofs", n)`` or ``("ref", n)``, a delta on entry
    *n* that declares a result of *declared* bytes; ``("outside",)``, such a delta whose base, by
    id, is in no pack of the store; or ``("back", distance)``, an offset delta whose distance back
    is written as given. Ids are made up, so git never reads these objects. Returns the number of
    entries."""
    ids = [hashlib.sha1(f"crafted {n}".encode()).digest() for n in range(len(entries))]
    delta = _number(10) + _number(declared) + b"\x01Z"
    pack, offsets = bytearray(b"PACK" + (2).to_bytes(4, "big") + len(entries).to_bytes(4, "big")), []
    for spec in entries:
        offsets.append(len(pack))
        if spec[0] == "blob":
            pack += _type_and_size(3, declared) + zlib.compress(b"x\n")
        elif spec[0] == "ofs":
            pack += _type_and_size(6, len(delta)) + _back(offsets[-1] - offsets[spec[1]]) + zlib.compress(delta)
        elif spec[0] == "ref":
            pack += _type_and_size(7, len(delta)) + ids[spec[1]] + zlib.compress(delta)
        elif spec[0] == "outside":
            pack += _type_and_size(7, len(delta)) + hashlib.sha1(b"a base in no pack").digest() + zlib.compress(delta)
        else:
            pack += _type_and_size(6, len(delta)) + _back(spec[1]) + zlib.compress(delta)
    pack += hashlib.sha1(bytes(pack)).digest()
    listed = sorted(zip(ids, offsets))
    index = bytearray(b"\xfftOc" + (2).to_bytes(4, "big"))
    index += b"".join(sum(1 for oid, _ in listed if oid[0] <= b).to_bytes(4, "big") for b in range(256))
    index += b"".join(oid for oid, _ in listed) + b"\0\0\0\0" * len(listed)
    index += b"".join(offset.to_bytes(4, "big") for _, offset in listed) + pack[-20:]
    index += hashlib.sha1(bytes(index)).digest()
    (git_dir / "objects" / "pack" / "pack-crafted.pack").write_bytes(bytes(pack))
    (git_dir / "objects" / "pack" / "pack-crafted.idx").write_bytes(bytes(index))
    return len(entries)


def _chains(how: str, chains: int, depth: int) -> list[tuple]:
    """*chains* blob roots, each with *depth* deltas on it, every one on the one before."""
    entries: list[tuple] = []
    for _ in range(chains):
        entries.append(("blob",))
        entries += [(how, len(entries) - 1 + n) for n in range(depth)]
    return entries


@pytest.mark.parametrize("how", ["ofs", "ref"])
def test_each_deltas_kind_is_found_once_however_long_its_chain(tmp_path, monkeypatch, how):
    """Every delta here declares a result past the bound on commits, trees and tags, so its kind
    is found through its chain to a blob. A kind found is kept for every entry of the chain, so
    the scan reads each entry at most twice, once in its pass over the pack and once on a chain,
    and a pack of long chains costs the harness no more than its count of entries: within twice
    the entries, the repository is bundled."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    held = _crafted_pack(ws / ".git", _chains(how, 3, 300))
    monkeypatch.setattr(commit, "SCAN_MAX_READS", 2 * held)
    # The chains declare 903 results of 100 MiB, past the cap the tests give and the chain bound,
    # whose own tests are test_a_repository_whose_objects_inflate_past_the_bound_* and
    # test_a_repository_whose_deltas_rebuild_past_the_bound_*: here the reads alone are bounded.
    monkeypatch.setattr(commit, "CHAIN_BYTES_MAX", 2**62)
    assert _main(ws, tmp_path, 2 * held * DECLARED) == (0, {})


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
def test_the_scans_reads_of_pack_entries_are_bounded(tmp_path, monkeypatch, past):
    """The scan counts every pack entry it reads, across every pack; past ``SCAN_MAX_READS`` the
    repository is over the limit, the contestant's, named by its bound. Packs of whole objects of
    ordinary sizes are read once per entry, and one count covers both packs here."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n", "a.py": "x\n"})
    _git(ws, "repack", "-a", "-d", "-q", "--window=0")
    (ws / "server.py").write_text("print('v2')\n")
    _git(ws, "commit", "-qam", "v2")
    _git(ws, "repack", "-d", "-q", "--window=0")
    assert len(list((ws / ".git" / "objects" / "pack").glob("*.pack"))) == 2
    held = _objects(ws)
    monkeypatch.setattr(commit, "SCAN_MAX_READS", held - past)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"more than {held - past} pack entries" in record["reason"]
    else:
        assert code == 0 and record == {}


@pytest.mark.parametrize("entries, readable", [
    (_chains("ofs", 1, 50), True),
    (_chains("ofs", 1, 51), False),
    (_chains("ref", 1, 50), True),
    (_chains("ref", 1, 51), False),
    ([("ref", 1), ("ref", 0)], False),
], ids=["ofs-at-the-depth", "ofs-one-deeper", "ref-at-the-depth", "ref-one-deeper", "ref-cycle"])
def test_a_delta_chain_deeper_than_git_writes_or_looping_is_unreadable(tmp_path, monkeypatch, entries, readable):
    """git writes delta chains at most ``_DELTA_DEPTH_MAX`` deep (4,095; 50 here). A deeper chain,
    or one that loops through its bases, is the contestant's unreadable repository, whatever
    order the scan meets its entries in."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    _crafted_pack(ws / ".git", entries)
    monkeypatch.setattr(commit, "_DELTA_DEPTH_MAX", 50)
    # Chains of 100 MiB results rebuild past the chain bound, whose own test it is; here the depth
    # alone is bounded.
    monkeypatch.setattr(commit, "CHAIN_BYTES_MAX", 2**62)
    code, record = _main(ws, tmp_path)
    if readable:
        assert (code, record) == (0, {})
    else:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "unreadable")
        assert "delta chain longer than the 50 git writes, or one that loops" in record["reason"]


def test_an_offset_delta_on_itself_is_left_to_git(tmp_path):
    """An offset delta whose distance back is 0 names itself as its base; git refuses the entry
    when it reads it, so the scan skips it as git would, never following it round."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    _crafted_pack(ws / ".git", [("blob",), ("back", 0)])
    assert _main(ws, tmp_path) == (0, {})


def test_the_objects_of_an_alternate_store_are_bounded_too(tmp_path, monkeypatch):
    base = tmp_path / "ws" / "vendor" / "base"
    ws = _borrowing(tmp_path, base, str(base / ".git" / "objects"))
    largest = max(size for _, size in _sizes(base).values())
    monkeypatch.setattr(commit, "OBJECT_MAX_BYTES", largest - 1)
    code, record = _main(ws, tmp_path)
    assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
    assert f"more than {largest - 1} bytes" in record["reason"]


def test_a_loose_objects_header_past_the_bytes_read_for_it_is_unreadable(tmp_path):
    """git inflates through any number of empty deflate blocks before a loose object's header;
    the program reads a bounded start of the file, so a header it does not find there, in a
    stream that is still whole, is the contestant's unreadable repository, never a size it
    skipped."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    _write_loose(ws / ".git", "blob", b"x\n", padding=4000)
    code, record = _main(ws, tmp_path)
    assert (code, record["kind"]) == (UNREADABLE_EXIT, "unreadable")
    assert "its header" in record["reason"]


def test_a_loose_object_git_cannot_inflate_is_left_to_git(tmp_path):
    """A loose object that is not a zlib stream fails git as soon as git reads it, before it
    allocates anything; the size scan leaves it to git, which bundles a main that does not
    reach it."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    (ws / ".git" / "objects" / "ab").mkdir(exist_ok=True)
    (ws / ".git" / "objects" / "ab" / ("c" * 38)).write_bytes(b"not zlib at all")
    assert _main(ws, tmp_path) == (0, {})


def test_a_pack_index_is_read_no_further_than_its_table(tmp_path):
    """An index of 8 GiB is judged from its header in a process held to 1 GiB."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    index = ws / ".git" / "objects" / "pack" / "pack-big.idx"
    index.write_bytes(b"")
    os.truncate(index, 8 * 2**30)
    (ws / ".git" / "objects" / "pack" / "pack-big.pack").write_bytes(b"PACK")
    code, found = _run(ws, tmp_path / "out", memory_gib=1)
    assert code == UNREADABLE_EXIT, found["stderr"]
    assert "pack-big.idx is not a pack index git reads" in found["no_submission"].reason


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
def test_the_object_stores_followed_through_alternates_are_bounded(tmp_path, monkeypatch, past):
    """Each store's alternates file is read whole, so the stores are bounded
    (``ALTERNATES_MAX``, the workspace's own among them): past it the repository is over the
    limit, the contestant's."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    for name in ("a", "b"):
        (ws / "vendor" / name / "objects").mkdir(parents=True)
    (ws / ".git" / "objects" / "info" / "alternates").write_text(
        "".join(f"{ws / 'vendor' / name / 'objects'}\n" for name in ("a", "b")))
    monkeypatch.setattr(commit, "ALTERNATES_MAX", 2 if past else 3)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert "more than 2 object directories" in record["reason"]
    else:
        assert code == 0 and record == {}


def test_a_link_back_to_a_directory_already_walked_is_walked_once(tmp_path, monkeypatch):
    """A link inside the object store to the store itself makes a cycle without a loop of links;
    the walk follows each link's target once, so it ends far inside the bound."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    (ws / ".git" / "objects" / "back").symlink_to(ws / ".git" / "objects")
    (ws / ".git" / "refs" / "heads" / "nested").mkdir()
    (ws / ".git" / "refs" / "heads" / "nested" / "up").symlink_to(ws / ".git" / "refs")
    _walk_bound(monkeypatch, 1000)
    assert _main(ws, tmp_path) == (0, {})


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
def test_the_walks_bound_counts_every_store_and_the_refs_together(tmp_path, monkeypatch, past):
    """One bound covers the workspace's objects, its refs and every alternate store together."""
    base = tmp_path / "ws" / "vendor" / "base"
    ws = _borrowing(tmp_path, base, str(base / ".git" / "objects"))
    store = sum(len(dirs) + len(files) for _, dirs, files in os.walk(base / ".git" / "objects"))
    bound = _walked(ws / ".git") + store - past
    _walk_bound(monkeypatch, bound)
    code, record = _main(ws, tmp_path)
    if past:
        assert (code, record["kind"]) == (UNREADABLE_EXIT, "over_limit")
        assert f"more than {bound} entries" in record["reason"]
    else:
        assert code == 0 and record == {}


@pytest.mark.parametrize("past", [False, True], ids=["at-the-bound", "one-past"])
@pytest.mark.parametrize("listing", ["packed-refs", "alternates"])
def test_a_listing_git_reads_line_by_line_is_bounded_in_bytes(tmp_path, monkeypatch, listing, past):
    """``packed-refs`` and an alternates file are read line by line, as often as a lookup needs;
    a file of millions of short lines would outlast the role's time. Past ``LISTING_MAX_BYTES``
    it is the contestant's unreadable repository, named by its bound."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    git = ws / ".git"
    if listing == "packed-refs":
        _git(ws, "pack-refs", "--all")
        path = git / "packed-refs"
    else:
        path = git / "objects" / "info" / "alternates"
        path.write_text(f"# the workspace's own store\n{git / 'objects'}\n")
    size = path.stat().st_size
    bound = size - 1 if past else size
    monkeypatch.setattr(commit, "LISTING_MAX_BYTES", bound)
    code, record = _main(ws, tmp_path)
    if past:
        assert code == UNREADABLE_EXIT and f"more than {bound} bytes" in record["reason"]
    else:
        assert code == 0 and record == {}


def test_a_reason_is_bounded_whatever_the_contestant_wrote(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    (ws / ".git" / "refs" / "heads" / "main").write_text("ref: " + "x" * 50_000 + "\n")
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT and "not a ref" in found["no_submission"].reason
    assert len(found["no_submission"].reason) <= REASON_MAX_CHARS
    assert len(found["record"]["reason"]) <= REASON_MAX_CHARS


def test_a_symbolic_ref_to_a_name_no_filesystem_holds_is_not_a_ref(tmp_path):
    """A name git would accept but no file could have (a part over 255 bytes) is refused as not
    a ref, never an error of the machine's reading it."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    (ws / ".git" / "refs" / "heads" / "main").write_text("ref: refs/heads/" + "y" * 300 + "\n")
    code, found = _run(ws, tmp_path / "out")
    assert code == UNREADABLE_EXIT and "not a ref" in found["no_submission"].reason


def test_gits_message_that_is_not_utf8_is_read_and_kept_printable(tmp_path, monkeypatch):
    """git's messages can carry the contestant's bytes; one that is not UTF-8 is read as text all
    the same, and the reason holds it printably."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    fake = tmp_path / "bin" / "git"
    fake.parent.mkdir()
    fake.write_bytes(b"#!/bin/sh\nprintf 'fatal: bad object \\377\\376 here\\n' >&2\nexit 128\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    with pytest.raises(commit._NoSubmission) as caught:
        commit._on_objects(["-C", str(tmp_path), "update-ref", "refs/heads/main", "1" * 40],
                           commit.git_env(tmp_path), ws, 60)
    assert caught.value.kind == "unreadable" and "bad object" in caught.value.reason
    caught.value.reason.encode("utf-8")


# --- infra -----------------------------------------------------------------------------------


def test_a_bundling_git_killed_mid_write_leaves_no_partial_bundle(tmp_path, monkeypatch):
    """git writes the bundle to the program, which writes it into a partial file beside the
    output; git killed mid-write, as the kernel's OOM killer kills it, leaves part of the
    contestant's compressed history there, which no scan reads. The end is infra, and the
    program's own cleanup removes the partial file."""
    ws = _repo(tmp_path / "ws", {"server.py": "print('hi')\n"})
    out = tmp_path / "out" / "source.bundle"
    _bundling_git(tmp_path, monkeypatch, 'printf "# v2 git bundle\\nPACK half"; kill -9 $$')
    (tmp_path / "tmp").mkdir()
    with pytest.raises(commit._Infra, match="status -9"):
        commit.commit(ws, out, tmp=tmp_path / "tmp", max_bytes=MAX, seconds=SECONDS)
    assert list(out.parent.iterdir()) == []


def test_a_workspace_path_that_is_a_link_is_infra(tmp_path):
    real = _repo(tmp_path / "real", {"server.py": "x\n"})
    (tmp_path / "ws").symlink_to(real)
    code, found = _run(tmp_path / "ws", tmp_path / "out")
    assert code == 1 and "not a directory" in found["stderr"]
    assert found["files"] == []


def test_an_output_that_cannot_be_written_is_an_io_error(tmp_path, capsys):
    """The program is refused the bundle's directory, outside the workspace: the refusal is the
    machine's, never the contestant's, though the no-submission and commit records could be
    written."""
    ws = _repo(tmp_path / "ws", {"server.py": "x\n"})
    out, records = tmp_path / "out", tmp_path / "records"
    out.mkdir()
    out.chmod(0o500)
    code = commit.main(["--workspace", str(ws), "--out", str(out / "source.bundle"), "--no-submission",
                        str(records / "no_submission.json"), "--record", str(records / "commit.json"), "--tmp",
                        str(tmp_path), "--max-bytes", str(MAX),
            "--seconds", str(SECONDS)])
    err = capsys.readouterr().err
    assert code == IO_ERROR_EXIT and "hls-commit: io-error: EACCES: " in err and f"{out}/{commit.PARTIAL_DIR}" in err
    assert not (records / "no_submission.json").exists()
    assert json.loads((records / "commit.json").read_text())["status"] == "io_error"


@pytest.mark.parametrize("returncode, stderr, verdict", [
    (128, "fatal: unable to read 1234\nerror: corrupt loose object\n", "error: corrupt loose object"),
    (128, "", "git exited with status 128"),
    (-9, "fatal: something\n", None),
    (128, "fatal: write error: No space left on device\n", None),
    (128, "fatal: Out of memory, malloc failed\n", None),
    (128, "error: Disk quota exceeded\n", None),
])
def test_only_failures_the_repository_causes_are_the_contestants(returncode, stderr, verdict):
    assert git_failure(returncode, stderr) == verdict


# --- the outputs report ----------------------------------------------------------------------


@pytest.mark.parametrize("bundled", [True, False], ids=["bundle", "no-submission"])
def test_the_output_written_is_reported_by_its_digest_for_the_manifest(tmp_path, bundled):
    """The stage publishes what the commit point wrote by the digest the program reports, of the
    bytes it wrote, keyed by its path under the outputs directory, as
    ``hls.store.manifest.report_written_outputs`` writes it."""
    ws = _repo(tmp_path / "ws", {"server.py": "x\n"}, branch="main" if bundled else "dev")
    _readonly(ws)
    out, report = tmp_path / "outputs", tmp_path / "role" / "outputs-report.json"
    report.parent.mkdir()
    code = commit.main(["--workspace", str(ws), "--out", str(out / "source.bundle"), "--no-submission",
                        str(out / "no_submission.json"), "--outputs-out", str(report), "--tmp", str(tmp_path), "--max-bytes", str(MAX),
            "--seconds", str(SECONDS)])
    name = "source.bundle" if bundled else "no_submission.json"
    assert code == (0 if bundled else NO_MAIN_EXIT)
    reported = read_outputs_report(report)
    assert reported is not None and list(reported) == [name]
    assert (reported[name].sha256, reported[name].size) == sha256_file(out / name)


def test_the_outputs_report_is_written_in_place_into_the_file_the_stage_made(tmp_path):
    """The stage makes the report file larger than any report before the sandbox starts, on the
    node-local disk the agent can fill; the program writes it in place and cuts it to the
    record, so it needs no new space there."""
    ws = _repo(tmp_path / "ws", {"server.py": "x\n"})
    _readonly(ws)
    out, report = tmp_path / "outputs", tmp_path / "role" / "outputs-report.json"
    report.parent.mkdir()
    report.write_bytes(b" " * 4096)
    inode = report.stat().st_ino
    code = commit.main(["--workspace", str(ws), "--out", str(out / "source.bundle"), "--no-submission",
                        str(out / "no_submission.json"), "--outputs-out", str(report), "--tmp", str(tmp_path), "--max-bytes", str(MAX),
            "--seconds", str(SECONDS)])
    assert code == 0 and report.stat().st_ino == inode
    reported = read_outputs_report(report)
    assert reported is not None and (reported["source.bundle"].sha256, reported["source.bundle"].size) == \
        sha256_file(out / "source.bundle")
    assert sorted(p.name for p in report.parent.iterdir()) == ["outputs-report.json"]


@pytest.mark.parametrize("bundled", [True, False], ids=["bundle", "no-submission"])
@pytest.mark.parametrize("change", ["lost", "gained"])
def test_an_output_the_store_holds_at_another_size_once_synced_is_infra_and_never_kept(tmp_path, monkeypatch,
                                                                                      capsys, bundled, change):
    """The program reports the digest of the bytes it wrote, never a read of the file: once the
    output is synced, the store must hold as many bytes as were written. One that lost or gained
    bytes on the way is not what the program wrote, so the try is infra, the store's fault, which
    the stage retries on the same workspace, and neither the output nor its report is kept."""
    ws = _repo(tmp_path / "ws", {"server.py": "x\n"}, branch="main" if bundled else "dev")
    _readonly(ws)
    out, report = tmp_path / "outputs", tmp_path / "role" / "outputs-report.json"
    report.parent.mkdir()
    real_fsync = os.fsync

    def store(fd: int) -> None:
        size = os.fstat(fd).st_size
        if change == "lost":
            os.ftruncate(fd, size - 1)
        else:
            os.pwrite(fd, b"x", size)
        real_fsync(fd)

    monkeypatch.setattr(commit.os, "fsync", store)
    code = commit.main(["--workspace", str(ws), "--out", str(out / "source.bundle"), "--no-submission",
                        str(out / "no_submission.json"), "--outputs-out", str(report), "--tmp", str(tmp_path),
                        "--max-bytes", str(MAX), "--seconds", str(SECONDS)])
    monkeypatch.setattr(commit.os, "fsync", real_fsync)
    what = "main's bundle" if bundled else "no_submission.json"
    assert code == 1
    assert re.search(f"infra: the store holds [0-9]+ bytes of {what}, not the [0-9]+ the program wrote",
                     capsys.readouterr().err)
    assert not (out / "source.bundle").exists() and not (out / "no_submission.json").exists()
    assert not report.exists()


def test_an_outputs_report_needs_the_record_beside_the_bundle(tmp_path, capsys):
    ws = _repo(tmp_path / "ws", {"server.py": "x\n"})
    with pytest.raises(SystemExit):
        commit.main(["--workspace", str(ws), "--out", str(tmp_path / "out" / "source.bundle"), "--no-submission",
                     str(tmp_path / "elsewhere" / "ns.json"), "--outputs-out", str(tmp_path / "r.json"),
                     "--max-bytes", str(MAX), "--seconds", str(SECONDS)])
    assert "beside the bundle" in capsys.readouterr().err


# --- the program itself ----------------------------------------------------------------------


def test_the_program_runs_on_a_bare_interpreter_with_the_stdlib_alone(tmp_path):
    tree = ast.parse(PROGRAM.read_text())
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported - {"__future__"} <= sys.stdlib_module_names
    p = subprocess.run([sys.executable, "-I", str(PROGRAM), "--help"], env={"PATH": os.environ["PATH"]},
                       cwd=tmp_path, capture_output=True, text=True)
    assert p.returncode == 0 and "--workspace" in p.stdout


def test_every_git_command_packs_without_a_delta_search_or_compression(tmp_path, monkeypatch):
    """Every git command the program runs, the bundle's own pack-objects among them, has the
    bundle's settings (``GIT_FLAGS``): no delta search and no compression. git's big-file
    threshold is left as it is."""
    ws = _repo(tmp_path / "ws", {"server.py": "x\n"})
    seen = tmp_path / "seen"
    wrapper = tmp_path / "bin" / "git"
    wrapper.parent.mkdir()
    real = subprocess.run(["sh", "-c", "command -v git"], capture_output=True, text=True, check=True).stdout.strip()
    wrapper.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> {seen}\nexec {real} "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{wrapper.parent}:{os.environ['PATH']}")
    (tmp_path / "tmp").mkdir()
    assert commit.commit(ws, tmp_path / "out" / "source.bundle", tmp=tmp_path / "tmp", max_bytes=MAX,
                         seconds=SECONDS).status == "ok"
    runs = seen.read_text().splitlines()
    assert any(" bundle create " in run for run in runs)
    for run in runs:
        for setting in ("pack.window=0", "pack.compression=0"):
            assert f"-c {setting}" in run, run
        assert "bigFileThreshold" not in run, run


def test_git_runs_with_the_package_checkouts_environment():
    assert commit.CLONE_ENV == checkout.CLONE_ENV


def test_git_never_sees_the_callers_git_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/elsewhere")
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", "/elsewhere/objects")
    env = commit.git_env(tmp_path)
    assert set(env) == {"PATH", "HOME", "LANG", *commit.CLONE_ENV} and env["HOME"] == str(tmp_path)


def test_the_stage_runs_the_bound_copy_on_the_images_interpreter():
    assert commit_command(workspace="/workspace", out="/out/source.bundle", max_bytes=MAX, seconds=SECONDS,
                          no_submission="/out/ns.json") == (
        "/usr/bin/python3", "-I", COMMIT_TARGET, "--workspace", "/workspace", "--out", "/out/source.bundle",
        "--max-bytes", str(MAX), "--seconds", str(SECONDS), "--no-submission", "/out/ns.json")
    one = {"workspace": "/w", "out": "/o", "max_bytes": 1, "seconds": 1}
    assert commit_command(**one, record="/r")[-2:] == ("--record", "/r")
    assert commit_command(**one, outputs_out="/rep")[-2:] == ("--outputs-out", "/rep")
    assert commit_command(**one, tmp="/t")[-2:] == ("--tmp", "/t")


@pytest.mark.parametrize("flag", ["--max-bytes", "--seconds"])
@pytest.mark.parametrize("value", [None, "0", "x"])
def test_the_program_holds_a_repository_only_to_a_cap_and_a_time_its_caller_names(tmp_path, flag, value):
    # No default: a caller that names none would hold the repository, or give its git a time, by
    # nothing it stated.
    given = {"--max-bytes": str(MAX), "--seconds": str(SECONDS)}
    argv = ["--workspace", str(tmp_path), "--out", str(tmp_path / "o" / "source.bundle")]
    for name, text in given.items():
        if name != flag:
            argv += [name, text]
    with pytest.raises(SystemExit):
        commit.main(argv + ([] if value is None else [flag, value]))


def test_a_plain_repository_is_measured_with_no_problem(tmp_path):
    ws = _repo(tmp_path / "ws", {"a": "x\n"})
    assert check_repository(ws, MAX) is None
    assert commit.object_count([ws / ".git" / "objects"], ws) == _objects(ws)
    assert _measured(ws) == (_inflated(ws), sum(size for kind, size in _sizes(ws).values() if kind == "tree"), 0)


@pytest.mark.parametrize("text", [
    "{", "[]", '{"kind": "no_main", "reason": "r"}', '{"kind": "other", "reason": "r", "at": 1}',
    '{"kind": "no_main", "reason": 3, "at": 1}', '{"kind": "no_main", "reason": "r", "at": true}',
    '{"kind": "no_main", "reason": "r", "at": 1, "extra": 1}', '{"kind": "no_main", "reason": "\\udcff", "at": 1}',
    "[" * 100_000,
])
def test_a_no_submission_record_must_be_exactly_its_shape(text):
    with pytest.raises(ValueError):
        NoSubmission.from_json(text)


@pytest.mark.parametrize("kind", ["no_main", "unreadable", "over_limit"])
def test_a_no_submission_record_round_trips(kind):
    record = NoSubmission(kind=kind, reason="why", at=12.5)
    assert NoSubmission.from_json(record.to_json()) == record


# --- snapshots -------------------------------------------------------------------------------


def test_snapshots_keep_the_highest_numbered_whatever_their_times(tmp_path):
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for n in range(1, 6):
        path = snaps / snapshot_name(n)
        path.write_bytes(b"x")
        os.utime(path, (1_000_000 - n, 1_000_000 - n))
    (snaps / "notes.txt").write_text("kept\n")
    assert prune_snapshots(snaps) == [snapshot_name(n) for n in (1, 2, 3)]
    assert sorted(p.name for p in snaps.iterdir()) == ["main-000004.bundle", "main-000005.bundle", "notes.txt"]
    assert SNAPSHOT_KEEP == 2 and snapshot_name(12) == "main-000012.bundle"
    assert prune_snapshots(tmp_path / "missing") == []
    with pytest.raises(ValueError):
        snapshot_name(0)


def test_a_snapshot_is_a_bundle_of_main_like_the_commit(tmp_path):
    ws = _repo(tmp_path / "ws", {"server.py": "x\n"})
    main = _git(ws, "rev-parse", "main")
    out = tmp_path / "out"
    out.mkdir()
    argv = [sys.executable, "-I", str(PROGRAM), "--workspace", str(ws), "--out",
            str(out / "snapshots" / snapshot_name(1)), "--tmp", str(out), "--max-bytes", str(MAX),
            "--seconds", str(SECONDS)]
    p = subprocess.run(argv, env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert _heads(out / "snapshots" / snapshot_name(1)) == [f"{main} refs/heads/main"]
