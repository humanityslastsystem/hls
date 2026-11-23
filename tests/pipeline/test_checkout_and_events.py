from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

import pytest

from hls.pipeline.checkout import code_state, secrets_copies
from hls.pipeline.events import EventLog, read_events
from hls.substrate.secrets import Scrubber


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    for name, text in files.items():
        (path / name).write_text(text)
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    return path


def _checkout(tmp_path: Path) -> Path:
    root = _repo(tmp_path / "hls", {"a.py": "x = 1\n", ".gitignore": "harbor/\nruns/\n"})
    _repo(root / "harbor", {"h.py": "y = 1\n"})
    return root


def test_a_clean_checkout_has_no_dirty_digest(tmp_path):
    root = _checkout(tmp_path)
    state = code_state(root)
    assert state.checkout == root.resolve() and state.dirty_diff_sha256 is None
    assert state.hls_commit == _git(root, "rev-parse", "HEAD").strip()
    assert state.harbor_commit == _git(root / "harbor", "rev-parse", "HEAD").strip()


@pytest.mark.parametrize("mark", ["--assume-unchanged", "--skip-worktree"])
@pytest.mark.parametrize("where, name", [(".", "a.py"), ("harbor", "h.py")], ids=["checkout", "harbor"])
def test_a_tracked_file_hidden_from_git_status_is_refused(tmp_path, mark, where, name):
    # git status passes over a file so marked whatever it holds, and so would the dirty digest.
    root = _checkout(tmp_path)
    repo = (root / where).resolve()
    _git(repo, "update-index", mark, name)
    (repo / name).write_text("edited = True\n")
    with pytest.raises(ValueError, match=rf"^{re.escape(str(repo))}: 1 tracked files are marked assume-unchanged or "
                                         rf"skip-worktree, {name} first, so git status and the dirty digest pass over "
                                         r"their changes; clear the marks with git update-index "
                                         r"--no-assume-unchanged --no-skip-worktree$"):
        code_state(root)


def test_ignored_files_do_not_count(tmp_path):
    root = _checkout(tmp_path)
    (root / "runs").mkdir()
    (root / "runs" / "out.json").write_text("{}")
    assert code_state(root).dirty_diff_sha256 is None


def test_every_kind_of_change_moves_the_digest(tmp_path):
    root = _checkout(tmp_path)
    seen = set()
    (root / "a.py").write_text("x = 2\n")
    seen.add(code_state(root).dirty_diff_sha256)
    (root / "new.py").write_text("z = 1\n")
    seen.add(code_state(root).dirty_diff_sha256)
    (root / "new.py").write_text("z = 2\n")
    seen.add(code_state(root).dirty_diff_sha256)
    (root / "harbor" / "h.py").write_text("y = 2\n")
    seen.add(code_state(root).dirty_diff_sha256)
    assert None not in seen and len(seen) == 4


def test_a_second_edit_to_a_modified_file_moves_the_digest(tmp_path):
    root = _checkout(tmp_path)
    (root / "a.py").write_text("x = 2\n")
    first = code_state(root).dirty_diff_sha256
    (root / "a.py").write_text("x = 3\n")
    assert code_state(root).dirty_diff_sha256 not in (None, first)


def test_an_edit_to_an_untracked_file_in_a_new_directory_moves_the_digest(tmp_path):
    root = _checkout(tmp_path)
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "sub" / "m.py").write_text("a = 1\n")
    first = code_state(root).dirty_diff_sha256
    (root / "pkg" / "sub" / "m.py").write_text("a = 2\n")
    assert first is not None and code_state(root).dirty_diff_sha256 not in (None, first)


def test_the_executable_bit_moves_the_digest(tmp_path):
    root = _checkout(tmp_path)
    (root / "a.py").chmod(0o755)
    tracked = code_state(root).dirty_diff_sha256
    (root / "new.sh").write_text("echo\n")
    plain = code_state(root).dirty_diff_sha256
    (root / "new.sh").chmod(0o755)
    assert tracked is not None and plain != tracked and code_state(root).dirty_diff_sha256 != plain


def test_deleted_and_replaced_paths_move_the_digest(tmp_path):
    root = _checkout(tmp_path)
    (root / "a.py").unlink()
    deleted = code_state(root).dirty_diff_sha256
    (root / "a.py").symlink_to("elsewhere")
    linked = code_state(root).dirty_diff_sha256
    (root / "a.py").unlink()
    (root / "a.py").symlink_to("other")
    retargeted = code_state(root).dirty_diff_sha256
    (root / "a.py").unlink()
    (root / "a.py").mkdir()
    seen = {deleted, linked, retargeted, code_state(root).dirty_diff_sha256}
    assert None not in seen and len(seen) == 4


def test_the_same_path_in_hls_and_in_harbor_is_not_the_same_change(tmp_path):
    in_hls, in_harbor = _checkout(tmp_path / "one"), _checkout(tmp_path / "two")
    (in_hls / "same.py").write_text("z = 1\n")
    (in_harbor / "harbor" / "same.py").write_text("z = 1\n")
    assert code_state(in_hls).dirty_diff_sha256 != code_state(in_harbor).dirty_diff_sha256


def test_the_digest_does_not_depend_on_git_config(tmp_path):
    root = _checkout(tmp_path)
    (root / "a.py").write_text("x = 2\n")
    (root / "new.py").write_text("z = 1\n")
    before = code_state(root).dirty_diff_sha256
    for key, value in [("diff.noprefix", "true"), ("color.diff", "always"), ("diff.algorithm", "patience"),
                       ("core.quotePath", "false")]:
        _git(root, "config", key, value)
        assert code_state(root).dirty_diff_sha256 == before


def test_the_same_changes_give_the_same_digest_in_another_checkout(tmp_path):
    first = _checkout(tmp_path / "one")
    second = _checkout(tmp_path / "two")
    for root in (first, second):
        (root / "a.py").write_text("x = 2\n")
        (root / "new.py").write_text("z = 1\n")
    assert code_state(first).dirty_diff_sha256 == code_state(second).dirty_diff_sha256 is not None


def test_computing_the_state_does_not_write_the_checkout(tmp_path):
    root = _checkout(tmp_path)
    (root / "a.py").write_text("x = 2\n")
    # A tracked file with another modification time has a stale index entry, which a
    # command that refreshes the index would write back.
    for path in (root / ".gitignore", root / "harbor" / "h.py"):
        os.utime(path, (1_000_000_000, 1_000_000_000))
    indexes = [root / ".git" / "index", root / "harbor" / ".git" / "index"]
    before = [os.stat(p).st_mtime_ns for p in indexes]
    code_state(root)
    assert [os.stat(p).st_mtime_ns for p in indexes] == before


def test_an_edit_within_one_clock_tick_of_the_index_is_still_seen(tmp_path):
    root = _checkout(tmp_path)
    # With only size and modification time compared, an edit of the same size that keeps the
    # file's time looks unchanged. Git then compares content for any file whose time is not
    # older than the index's, so the index must keep its own time when it is copied.
    _git(root, "config", "core.checkStat", "minimal")
    _git(root, "config", "core.trustctime", "false")
    stamp = 1_000_000_000 * 10**9
    os.utime(root / "a.py", ns=(stamp, stamp))
    _git(root, "update-index", "--refresh")
    (root / "a.py").write_text("x = 2\n")
    os.utime(root / "a.py", ns=(stamp, stamp))
    os.utime(root / ".git" / "index", ns=(stamp, stamp))
    assert code_state(root).dirty_diff_sha256 is not None


def test_a_staged_rename_is_a_deletion_and_an_addition(tmp_path):
    renamed, replaced = _checkout(tmp_path / "one"), _checkout(tmp_path / "two")
    _git(renamed, "mv", "a.py", "c.py")
    (replaced / "a.py").unlink()
    (replaced / "c.py").write_text("x = 1\n")
    assert code_state(renamed).dirty_diff_sha256 == code_state(replaced).dirty_diff_sha256 is not None


def test_a_staged_change_reverted_in_the_working_tree_is_still_listed(tmp_path):
    root = _checkout(tmp_path)
    (root / "a.py").write_text("x = 2\n")
    _git(root, "add", "a.py")
    (root / "a.py").write_text("x = 1\n")
    assert code_state(root).dirty_diff_sha256 is not None


def test_a_submodule_at_a_new_commit_is_recorded_by_its_commit_not_as_a_change(tmp_path):
    origin = _repo(tmp_path / "harbor-origin", {"h.py": "y = 1\n"})
    root = _repo(tmp_path / "hls", {"a.py": "x = 1\n"})
    _git(root, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(origin), "harbor")
    _git(root, "commit", "-qm", "add harbor")
    harbor = root / "harbor"
    _git(harbor, "config", "user.email", "t@example.com")
    _git(harbor, "config", "user.name", "t")
    assert code_state(root).dirty_diff_sha256 is None
    (harbor / "h.py").write_text("y = 2\n")
    _git(harbor, "commit", "-qam", "advance")
    state = code_state(root)
    assert state.dirty_diff_sha256 is None and state.harbor_commit == _git(harbor, "rev-parse", "HEAD").strip()
    (harbor / "h.py").write_text("y = 3\n")
    assert code_state(root).dirty_diff_sha256 is not None


def test_harbor_must_be_its_own_repository(tmp_path):
    # A fresh `git worktree add` leaves harbor/ as an empty directory of the hls repository.
    root = _repo(tmp_path / "hls", {"a.py": "x = 1\n"})
    (root / "harbor").mkdir()
    with pytest.raises(ValueError, match="harbor submodule is not initialized"):
        code_state(root)


def test_a_missing_harbor_directory_is_not_initialized_either(tmp_path):
    root = _repo(tmp_path / "hls", {"a.py": "x = 1\n"})
    with pytest.raises(ValueError, match="harbor submodule is not initialized"):
        code_state(root)


def test_every_secrets_file_copy_in_the_checkout_is_found(tmp_path):
    root = _checkout(tmp_path)
    assert secrets_copies(root) == []
    for rel in (".secrets.toml", "tasks/x/.secrets.toml.bak-1", "harbor/.secrets.toml~", ".venv/lib/.secrets.tomlx"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("[secrets]\n")
    (root / "notes.secrets.toml").write_text("")
    (root / ".secrets.toml.link").symlink_to(root / "a.py")
    (root / ".git" / ".secrets.toml").write_text("")
    (root / ".git" / "objects").mkdir(exist_ok=True)
    (root / ".git" / "objects" / ".secrets.toml").write_text("")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".secrets.toml").write_text("")
    (root / "linked").symlink_to(outside)
    assert secrets_copies(root) == sorted(root / rel for rel in (
        ".git/.secrets.toml", ".secrets.toml", ".secrets.toml.link", ".venv/lib/.secrets.tomlx",
        "harbor/.secrets.toml~", "tasks/x/.secrets.toml.bak-1"))


def test_a_directory_named_like_a_copy_counts_and_is_searched(tmp_path):
    root = _checkout(tmp_path)
    for rel in (".secrets.toml.d/values", ".secrets.toml.d/sub/.secrets.toml", ".secrets.toml.bak/secrets.toml",
                "tasks/y/ordinary/file"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".secrets.toml").write_text("")
    (root / ".secrets.toml.dirlink").symlink_to(outside, target_is_directory=True)
    assert secrets_copies(root) == sorted(root / rel for rel in (
        ".secrets.toml.bak", ".secrets.toml.d", ".secrets.toml.d/sub/.secrets.toml", ".secrets.toml.dirlink"))


def test_events_append_one_line_each(tmp_path):
    log = EventLog(tmp_path / "a" / "events.jsonl", clock=lambda: 5.0)
    log.emit("stage_started", stage="noop", path=tmp_path)
    log.emit("stage_published", stage="noop")
    events = read_events(tmp_path / "a" / "events.jsonl")
    assert [e["kind"] for e in events] == ["stage_started", "stage_published"]
    assert events[0] == {"at": 5.0, "kind": "stage_started", "stage": "noop", "path": str(tmp_path)}
    assert read_events(tmp_path / "missing.jsonl") == []


def test_an_event_line_has_sorted_keys(tmp_path):
    log = EventLog(tmp_path / "events.jsonl", clock=lambda: 5.0)
    log.emit("k", b=1, a=2)
    assert (tmp_path / "events.jsonl").read_text() == '{"a": 2, "at": 5.0, "b": 1, "kind": "k"}\n'


def test_a_scrubbed_log_writes_each_whole_line_through_its_scrub(tmp_path):
    secret = "hf-token-0123456789"
    log = EventLog(tmp_path / "events.jsonl", clock=lambda: 5.0,
                   scrub=Scrubber({"hf_token": secret}).scrub)
    log.emit("role_ended", detail=f"saw {secret}", job_ooms={"steps": {f"step_{secret}": 0}})
    text = (tmp_path / "events.jsonl").read_text()
    assert secret not in text
    [event] = read_events(tmp_path / "events.jsonl")
    assert event["detail"] == "saw <redacted:hf_token>" and event["job_ooms"] == {"steps": {"step_<redacted:hf_token>": 0}}
    plain = EventLog(tmp_path / "plain.jsonl", clock=lambda: 5.0)
    plain.emit("role_ended", detail=f"saw {secret}")
    assert secret in (tmp_path / "plain.jsonl").read_text()


def test_events_cannot_override_the_time_or_the_kind(tmp_path):
    log = EventLog(tmp_path / "events.jsonl", clock=lambda: 5.0)
    with pytest.raises(ValueError, match="'at'"):
        log.emit("x", at=1.0)
    with pytest.raises(ValueError, match="'kind'"):
        log.emit("x", kind="y")
    assert read_events(tmp_path / "events.jsonl") == []


def test_an_undecodable_line_is_skipped_with_a_warning(tmp_path, caplog):
    path = tmp_path / "events.jsonl"
    log = EventLog(path, clock=lambda: 5.0)
    log.emit("first")
    with open(path, "ab") as f:
        f.write(b'{"at": 6.0, "kind": "tor')
        f.write(b"\n\xff\xfe\n[1, 2]\n")
    log.emit("last")
    with caplog.at_level(logging.WARNING, logger="hls.pipeline.events"):
        events = read_events(path)
    assert [e["kind"] for e in events] == ["first", "last"]
    assert caplog.text.count(str(path)) == 3
    assert all(f"line {n}" in caplog.text for n in (2, 3, 4))


def test_every_git_command_on_the_checkout_waits_the_one_bound(tmp_path, monkeypatch):
    """The checkout's state is read with git under the bound the planner's own git commands take
    (``READ_GIT_TIMEOUT_SEC``), one figure for every read-only git command in the checkout."""
    from hls.pipeline import checkout

    repo = _checkout(tmp_path)
    seen = []
    real = subprocess.run

    def recording(argv, **kwargs):
        seen.append(kwargs.get("timeout"))
        return real(argv, **kwargs)

    monkeypatch.setattr(checkout, "READ_GIT_TIMEOUT_SEC", 123)
    monkeypatch.setattr(checkout.subprocess, "run", recording)
    code_state(repo)
    assert seen and set(seen) == {123}
