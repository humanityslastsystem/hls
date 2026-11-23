from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hls.agent import commit as commit_program
from hls.package import checkout
from hls.package.checkout import (CLONE_ENV, TREE_MAX_BYTES, TREE_MAX_DEPTH, SourceRecord,
                                  TreeEntry, git_env, git_refuses, limit_problem, receive)
from hls.store.digest import COPY_FAILED_EXIT, DIGEST_MISMATCH_EXIT, sha256_file
from hls.store.manifest import read_outputs_report
from hls.substrate.secret_scan import PackageScan, PackageScanArea
from tests.package.bundles import (BUNDLE_LIES, GOOD, SYMLINKS, make_bundle, make_history_bundle, make_kind_bundle,
                                   make_lying_bundle, make_repeated_bundle, make_stored_bundle, make_tree_bundle)


#: The cap the tests hold a bundle to unless a test is about it: a sparse file's worth, far past any
#: bundle here.
CAP = 2**36


def _argv(tmp_path: Path, bundle: Path, *extra: str, max_bytes: int = CAP) -> list[str]:
    """The role's command line for *bundle*, the size its record names the file's own unless
    *extra* names one."""
    size = [] if "--size" in extra else ["--size", str(bundle.stat().st_size)]
    return ["--bundle", str(bundle), "--dest", str(tmp_path / "tree"), "--out", str(tmp_path / "source.json"),
            "--outputs-out", str(tmp_path / "outputs.json"), "--max-bytes", str(max_bytes), *size, *extra]


def _reads(monkeypatch, then) -> None:
    """Have the role's reads of the bundle pass through *then*, given each chunk read and the
    bundle's path, which returns what the role gets of it."""
    real = checkout._chunks

    def chunks(fd: int, path: Path):
        for chunk in real(fd, path):
            yield then(chunk, path)

    monkeypatch.setattr(checkout, "_chunks", chunks)


def _run(tmp_path: Path, bundle: Path, expect: str | None = None, max_bytes: int = CAP) -> SourceRecord:
    """Run the role on *bundle* under the cap *max_bytes*, expecting *expect*, else the bundle's own
    digest."""
    assert checkout.main(_argv(tmp_path, bundle, "--expect", expect or sha256_file(bundle)[0],
                               max_bytes=max_bytes)) == 0
    return SourceRecord.model_validate_json((tmp_path / "source.json").read_text())


def test_the_checkout_refuses_to_run_without_an_expected_digest(tmp_path, capsys):
    """Every bundle the role reads comes with the digest it must have, the agent_build manifest's
    or the one the operator stated: without one the role reads nothing and writes nothing."""
    bundle = make_bundle(tmp_path, GOOD)
    before = set(os.listdir(tmp_path))
    with pytest.raises(SystemExit) as caught:
        checkout.main(_argv(tmp_path, bundle))
    assert caught.value.code == 2 and "the following arguments are required: --expect" in capsys.readouterr().err
    assert set(os.listdir(tmp_path)) == before


def test_the_checkout_refuses_to_run_without_the_recorded_size(tmp_path, capsys):
    """Every bundle the role reads comes with the size its record names, the agent_build
    manifest's or, for the copy of an operator's named bundle, the plan's: without one the role
    reads nothing and writes nothing."""
    bundle = make_bundle(tmp_path, GOOD)
    before = set(os.listdir(tmp_path))
    argv = _argv(tmp_path, bundle, "--expect", sha256_file(bundle)[0])
    at = argv.index("--size")
    with pytest.raises(SystemExit) as caught:
        checkout.main(argv[:at] + argv[at + 2:])
    assert caught.value.code == 2 and "the following arguments are required: --size" in capsys.readouterr().err
    assert set(os.listdir(tmp_path)) == before


def test_a_bundle_checks_out_main_as_a_plain_tree(tmp_path):
    bundle = make_bundle(tmp_path, {**GOOD, "pkg/mod.py": "x = 1\n"})
    record = _run(tmp_path, bundle)
    assert record.status == "ok" and len(record.commit or "") == 40
    assert record.files == ["pkg/mod.py", "pyproject.toml", "server.py"]
    tree = tmp_path / "tree"
    assert (tree / "server.py").read_text() == GOOD["server.py"] and not (tree / ".git").exists()


def test_the_record_names_the_bundle_by_its_digest_and_is_reported_for_the_manifest(tmp_path):
    bundle = make_bundle(tmp_path, GOOD)
    record = _run(tmp_path, bundle)
    assert record.bundle_sha256 == sha256_file(bundle)[0]
    reported = read_outputs_report(tmp_path / "outputs.json")
    assert reported is not None and list(reported) == ["source.json"]
    assert (reported["source.json"].sha256, reported["source.json"].size) == sha256_file(tmp_path / "source.json")


def test_a_bundle_with_another_digest_than_expected_is_refused_with_no_verdict_and_no_object_left(tmp_path,
                                                                                                    monkeypatch,
                                                                                                    capfd):
    """The bundle is streamed into git as it is read and hashed, so git has seen its objects by the
    time the digest is known: a bundle that is not the expected one is refused then, with no
    record, no scan, and nothing it held left on the node."""
    bundle = make_bundle(tmp_path, {**GOOD, ".env": "KEY=1\n"})
    monkeypatch.setenv("relay", "KEY=1")
    before = set(os.listdir(tmp_path))
    assert checkout.main(_argv(tmp_path, bundle, "--expect", "0" * 64, "--names", "relay", "--scan-out",
                               str(tmp_path / "scan.json"))) == DIGEST_MISMATCH_EXIT
    assert capfd.readouterr().err == f"refused: file {bundle} has sha256 {sha256_file(bundle)[0]}, not {'0' * 64}\n"
    assert set(os.listdir(tmp_path)) == before
    assert checkout.main(_argv(tmp_path, bundle, "--expect", sha256_file(bundle)[0])) == 0


def test_the_bundle_is_read_once_and_git_gets_only_the_bytes_the_role_hashed(tmp_path, monkeypatch):
    """The role reads the bundle where it lies once, hashing each byte as it hands it to git: no
    copy is made on the node, and a bundle that changes after that read reaches no git command."""
    bundle = make_bundle(tmp_path, GOOD)
    digest = sha256_file(bundle)[0]

    def then(chunk: bytes, path: Path) -> bytes:
        path.write_bytes(b"garbage")
        return chunk

    _reads(monkeypatch, then)
    before = set(os.listdir(tmp_path))
    assert checkout.main(_argv(tmp_path, bundle, "--expect", digest)) == 0
    record = SourceRecord.model_validate_json((tmp_path / "source.json").read_text())
    assert record.status == "ok" and record.bundle_sha256 == digest and bundle.read_bytes() == b"garbage"
    assert set(os.listdir(tmp_path)) - before == {"tree", "source.json", "outputs.json"}


def test_a_read_that_went_wrong_on_the_way_is_the_reads_failure(tmp_path, monkeypatch, capfd):
    """What was read differs while the bundle has the expected digest: the role says so with its
    own status, which the stage retries, and leaves no record and nothing it read."""
    bundle = make_bundle(tmp_path, GOOD)
    _reads(monkeypatch, lambda chunk, path: chunk[:-1] + bytes([chunk[-1] ^ 1]))
    before = set(os.listdir(tmp_path))
    assert checkout.main(_argv(tmp_path, bundle, "--expect", sha256_file(bundle)[0])) == COPY_FAILED_EXIT
    assert capfd.readouterr().err.endswith(f"read failed: what was read of file {bundle} differs from it\n")
    assert set(os.listdir(tmp_path)) == before


def _index_pack(tmp_path: Path, monkeypatch, body: str) -> None:
    """Put a git first on PATH whose ``index-pack`` runs the shell *body* and whose other verbs are
    the real git's."""
    real = subprocess.run(["sh", "-c", "command -v git"], capture_output=True, text=True, check=True).stdout.strip()
    bin_dir = tmp_path / "shim"
    bin_dir.mkdir()
    (bin_dir / "git").write_text(f'#!/bin/sh\nfor a in "$@"; do if [ "$a" = index-pack ]; then {body}; fi; done\n'
                                 f'exec {real} "$@"\n')
    (bin_dir / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")


def _big_bundle(tmp_path: Path) -> Path:
    """A bundle far past what a pipe holds, so a git that reads none of it blocks the role's write."""
    return make_bundle(tmp_path, {**GOOD, "blob.bin": os.urandom(4 * 2**20).hex()})


def test_a_clone_that_reads_nothing_for_a_while_is_waited_for_while_the_role_is_blocked_writing_to_it(
        tmp_path, monkeypatch):
    """A git that reads nothing blocks the role's write once the pipe is full: the clone has no
    time of its own, so the role waits for it, and the checkout completes once git reads. Only
    the stage's deadline ends a clone that never does."""
    bundle = _big_bundle(tmp_path)
    _index_pack(tmp_path, monkeypatch, "sleep 3")
    started = time.monotonic()
    record = _run(tmp_path, bundle)
    assert time.monotonic() - started >= 3
    assert record.status == "ok" and "blob.bin" in record.files


def test_a_git_that_fails_at_once_is_judged_only_once_the_whole_bundle_is_hashed(tmp_path, monkeypatch):
    """The role hashes the bundle to its end after git stops reading: with another digest than
    expected it is refused, and with the expected one git's failure is git's, infra, never a read
    gone wrong."""
    bundle = _big_bundle(tmp_path)
    _index_pack(tmp_path, monkeypatch, "exit 3")
    assert checkout.main(_argv(tmp_path, bundle, "--expect", "0" * 64)) == DIGEST_MISMATCH_EXIT
    assert not (tmp_path / "tree").exists()
    with pytest.raises(SystemExit) as caught:
        checkout.main(_argv(tmp_path, bundle, "--expect", sha256_file(bundle)[0]))
    assert "git index-pack failed with status 3" in str(caught.value.code)


def test_a_named_file_that_is_no_bundle_is_refused_by_its_digest_first(tmp_path):
    """The operator's wrong file, no bundle at all, is refused for its digest, which no retry
    changes, before its header is judged; a file with the expected digest that is no bundle ends
    the role, infra."""
    bad = tmp_path / "x.bundle"
    bad.write_bytes(b"not a bundle at all\n\n" + os.urandom(3 * 2**20))
    assert checkout.main(_argv(tmp_path, bad, "--expect", "0" * 64)) == DIGEST_MISMATCH_EXIT
    with pytest.raises(SystemExit) as caught:
        checkout.main(_argv(tmp_path, bad, "--expect", sha256_file(bad)[0]))
    assert "not a git bundle" in str(caught.value.code)


def test_a_bundle_without_main_and_another_digest_is_refused_with_no_record(tmp_path):
    bundle = make_bundle(tmp_path, GOOD, branch="other")
    assert checkout.main(_argv(tmp_path, bundle, "--expect", "0" * 64)) == DIGEST_MISMATCH_EXIT
    assert not (tmp_path / "source.json").exists()


def test_a_recorded_bundle_past_the_byte_limit_is_over_limit_by_its_size_alone(tmp_path):
    """A bundle whose size is the one its record names is that bundle. The clone would hold it
    whole on the node, so one over the cap its stage names, the build's writable disk cap, is the
    contestant's, refused by its size alone: the role may not even read it."""
    big = tmp_path / "big.bundle"
    with open(big, "wb") as f:
        f.truncate(CAP + 1)
    big.chmod(0o200)
    assert checkout.main(_argv(tmp_path, big, "--size", str(CAP + 1), "--expect", "0" * 64)) == 0
    record = SourceRecord.model_validate_json((tmp_path / "source.json").read_text())
    assert record == SourceRecord(status="over_limit", bundle_sha256=None, commit=None, files=[], heads=[],
                                  reason=f"the source bundle holds {CAP + 1} bytes, more than the {CAP} allowed")
    assert not (tmp_path / "tree").exists()


@pytest.mark.parametrize("past", [False, True], ids=["within-the-limit", "past-the-limit"])
def test_a_recorded_bundle_whose_size_is_not_its_records_is_refused_before_anything_reads_it(tmp_path, monkeypatch,
                                                                                            capfd, past):
    """The store's file is not the bytes its manifest names: refused (``DIGEST_MISMATCH_EXIT``),
    infra that no retry mends, whatever its size says of the limit, and before any read."""
    bundle = make_bundle(tmp_path, GOOD)
    size = bundle.stat().st_size
    cap = size - 1 if past else CAP
    bundle.chmod(0o200)
    before = set(os.listdir(tmp_path))
    assert checkout.main(_argv(tmp_path, bundle, "--size", str(size + 1), "--expect", "0" * 64,
                               max_bytes=cap)) == DIGEST_MISMATCH_EXIT
    assert capfd.readouterr().err == f"refused: file {bundle} holds {size} bytes, not the {size + 1} its record names\n"
    assert set(os.listdir(tmp_path)) == before


@pytest.mark.parametrize("expect", ["AB" * 32, "ab" * 31, f"sha256:{'ab' * 32}", ""],
                         ids=["uppercase", "short", "prefixed", "empty"])
def test_a_malformed_expected_digest_is_refused_before_anything_is_read(tmp_path, capsys, expect):
    bundle = make_bundle(tmp_path, GOOD)
    bundle.chmod(0o200)
    before = set(os.listdir(tmp_path))
    with pytest.raises(SystemExit) as caught:
        checkout.main(_argv(tmp_path, bundle, "--expect", expect))
    assert caught.value.code == 2 and "argument --expect: invalid check_sha256_hex value" in capsys.readouterr().err
    assert set(os.listdir(tmp_path)) == before


@pytest.mark.parametrize("below, status", [(0, "ok"), (1, "over_limit")], ids=["at-the-limit", "one-byte-past"])
def test_the_bundles_byte_limit_is_inclusive(tmp_path, below, status):
    bundle = make_bundle(tmp_path, GOOD)
    assert _run(tmp_path, bundle, max_bytes=bundle.stat().st_size - below).status == status


def test_the_role_needs_the_cap_its_stage_names(tmp_path, capsys):
    """The role has no cap of its own: its stage names the build's writable disk cap."""
    bundle = make_bundle(tmp_path, GOOD)
    argv = _argv(tmp_path, bundle, "--expect", sha256_file(bundle)[0])
    at = argv.index("--max-bytes")
    with pytest.raises(SystemExit):
        checkout.main(argv[:at] + argv[at + 2:])
    assert "--max-bytes" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        checkout.main([*argv[:at + 1], "0", *argv[at + 2:]])
    assert "--max-bytes must be positive" in capsys.readouterr().err
    assert not (tmp_path / "source.json").exists() and not (tmp_path / "tree").exists()


@pytest.mark.parametrize("cap", [180, 240])
def test_the_checkouts_disk_is_its_clones_pack_the_tree_and_gits_own_files(cap):
    """The clone's pack is as large as the bundle, since the bundle is not compressed; beside it
    the tree at its limit and git's own index and temporary files."""
    assert checkout.checkout_disk_gib(cap * 2**30) == cap + TREE_MAX_BYTES // 2**30 + 4


def test_a_bundle_without_main_is_reported_not_raised(tmp_path):
    bundle = make_bundle(tmp_path, GOOD, branch="dev")
    record = _run(tmp_path, bundle)
    assert record.status == "no_main" and record.commit is None and record.heads == ["refs/heads/dev"]
    assert not (tmp_path / "tree").exists()


def test_an_unreadable_bundle_fails_the_role(tmp_path):
    bad = tmp_path / "bad.bundle"
    bad.write_bytes(b"not a bundle\n")
    with pytest.raises(SystemExit) as caught:
        checkout.main(_argv(tmp_path, bad, "--expect", sha256_file(bad)[0]))
    assert caught.value.code not in (0, 2, None)
    assert not (tmp_path / "source.json").exists()


@pytest.mark.parametrize("lie", sorted(BUNDLE_LIES))
def test_a_bundle_that_does_not_read_back_is_over_limit_once_its_digest_is_proven(tmp_path, lie):
    """A bundle git made from a store whose objects lied, as the commit point keeps it or as one
    from outside the store may be: one carrying a blob whose deflate stream is broken or a delta
    it cannot apply, another object's bytes in a blob's, a tree's or a parent commit's place, or
    a header naming a main its pack does not hold. git trusts the object files it bundles from,
    so it carries each; the checkout is the first to read it as git reads it. The role read every
    byte it gave git with the digest the bundle must have, so git's verdict on its content
    (``bundle_verdict``) is the bundle's own: over_limit, nothing checked out, never infra."""
    record = _run(tmp_path, make_lying_bundle(tmp_path, lie))
    assert (record.status, record.files) == ("over_limit", [])
    assert re.fullmatch(f"the bundle does not read back as git reads it: {BUNDLE_LIES[lie]}",
                        record.reason or ""), record.reason
    assert len(record.commit or "") == 40 and record.heads == ["refs/heads/main"]
    assert not (tmp_path / "tree").exists()


#: git's verdicts on the content of a bundle it reads (``OBJECT_VERDICTS``, ``DATA_VERDICTS``,
#: ``HISTORY_VERDICTS``), by the git command that gives each: an object a link names that the
#: pack does not hold, one of another kind than its link names, an object whose id another content
#: has, main's commit missing or not a commit; an object's deflate stream it cannot inflate, a
#: delta it cannot apply or whose base lies outside the pack, an object of no kind, its bytes
#: ending early; and, walking main's history, an object it names that the bundle does not hold or
#: one of another kind.
CONTENT_VERDICTS = [
    ("index-pack", f"did not receive expected object {'a' * 40}"),
    ("index-pack", f"object {'a' * 40}: expected type blob, found tree"),
    ("index-pack", f"object type mismatch at {'a' * 40}"),
    ("index-pack", f"Not all child objects of {'a' * 40} are reachable"),
    ("index-pack", f"SHA1 COLLISION FOUND WITH {'a' * 40} !"),
    ("update-ref", f"trying to write ref 'refs/heads/main' with nonexistent object {'a' * 40}"),
    ("update-ref", f"trying to write non-commit object {'a' * 40} to branch 'refs/heads/main'"),
    ("index-pack", "pack has bad object at offset 1234: inflate returned -3"),
    ("index-pack", "pack has bad object at offset 1234: offset value overflow for delta base object"),
    ("index-pack", "pack has bad object at offset 1234: delta base offset is out of bound"),
    ("index-pack", "pack has bad object at offset 1234: unknown object type 5"),
    ("index-pack", "pack has bad object at offset 1234: failed to apply delta"),
    ("index-pack", "serious inflate inconsistency"),
    ("index-pack", "premature end of pack file, 17 bytes missing"),
    ("index-pack", "premature end of pack file, 1 byte missing"),
    ("index-pack", "early EOF"),
    ("rev-list", f"missing blob object '{'a' * 40}'"),
    ("rev-list", f"missing tree object '{'a' * 40}'"),
    ("rev-list", f"bad tree object {'a' * 40}"),
    ("rev-list", f"entry 'pkg' in tree {'a' * 40} has tree mode, but is not a tree"),
    ("rev-list", f"entry 'x.py' in tree {'a' * 40} has blob mode, but is not a blob"),
    ("rev-list", f"unable to load root tree for commit {'a' * 40}"),
    ("rev-list", f"unable to parse commit {'a' * 40}"),
    ("rev-list", f"bad object {'a' * 40}"),
    ("rev-list", f"Could not read {'a' * 40}"),
]


@pytest.mark.parametrize("verb, verdict", CONTENT_VERDICTS)
def test_gits_verdict_on_a_bundles_content_is_named_and_a_starved_or_killed_git_has_none(verb, verdict):
    """git's verdicts on a bundle's content are found among its messages, whatever it wrote
    before them, and are the bundle's own once the caller has proven every byte git read has the
    bundle's digest. A git killed or starved, or one whose messages name an errno, gives no
    verdict, whatever else it said: the machine's."""
    assert checkout.bundle_verdict(128, f"error: something before\nfatal: {verdict}\nfatal: after\n") == verdict
    assert checkout.bundle_verdict(-9, f"fatal: {verdict}\n") is None
    assert checkout.bundle_verdict(128, f"fatal: {verdict}\nfatal: Out of memory, malloc failed\n") is None
    assert checkout.bundle_verdict(128, f"fatal: {verdict}\nerror: unable to read: Input/output error\n") is None


@pytest.mark.parametrize("message", [
    "fatal: cannot pread pack file: Connection timed out",
    "fatal: unable to open /r/objects/pack/pack-main.pack: No such file or directory",
    "fatal: unable to create temporary file: Read-only file system",
    "fatal: pack is corrupted (SHA1 mismatch)",
    "fatal: cannot lock ref 'refs/heads/main': Unable to create '/r/refs/heads/main.lock': File exists.",
    "fatal: revision walk setup failed",
], ids=["timed-out", "no-such-file", "read-only", "pack-checksum", "lock", "walk-setup"])
def test_any_other_failure_of_git_on_a_bundle_names_no_verdict(message):
    """A failure that is no verdict on the bundle's content is the machine's, whatever it names."""
    assert checkout.bundle_verdict(128, message + "\n") is None


@pytest.mark.parametrize("pattern, body", [
    ('*" index-pack "*', 'cat > /dev/null; echo "fatal: unable to create temporary file: Read-only file system" >&2; '
                         "exit 128"),
    ('*" index-pack "*', 'cat > /dev/null; echo "fatal: pack has bad object at offset 12: inflate returned -3" >&2; '
                         "kill -9 $$"),
    ('*" rev-list "*', 'echo "fatal: revision walk setup failed" >&2; exit 128'),
    ('*" rev-list "*', 'echo "fatal: missing blob object \'$(printf %040d 0)\'" >&2; echo "Out of memory" >&2; '
                       "exit 128"),
], ids=["index-pack-read-only", "index-pack-killed", "rev-list-walk", "rev-list-starved"])
def test_any_other_failure_of_the_clone_stays_infra(tmp_path, monkeypatch, pattern, body):
    """git's other failures in the clone, of the machine's files, and a git killed or starved
    whatever it said, end the role as infra."""
    _git_shim(tmp_path, monkeypatch, pattern, body)
    with pytest.raises(SystemExit):
        _run(tmp_path, make_bundle(tmp_path, GOOD))
    assert not (tmp_path / "source.json").exists() and not (tmp_path / "tree").exists()


def test_the_hosts_git_configuration_never_applies(tmp_path, monkeypatch):
    """A hooks path in the global or system config, or a GIT_DIR in the environment, would let
    host configuration act on contestant data; the clone sees none of them."""
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    marker = tmp_path / "hook-ran"
    hook = hooks / "post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    config = tmp_path / "evil.gitconfig"
    config.write_text(f"[core]\n\thooksPath = {hooks}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(config))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))
    bundle = make_bundle(tmp_path, GOOD)
    assert _run(tmp_path, bundle).status == "ok"
    assert not marker.exists()


def test_the_clone_environment_is_minimal_and_switches_host_config_off(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/x")
    env = git_env(tmp_path / "home")
    assert {k: env[k] for k in CLONE_ENV} == CLONE_ENV
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null" and env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["HOME"] == str(tmp_path / "home") and "GIT_DIR" not in env
    assert set(env) == {"PATH", "HOME", "LANG", *CLONE_ENV}


@pytest.mark.parametrize("files, links, first", SYMLINKS.values(), ids=SYMLINKS)
def test_a_tree_with_any_symlink_is_over_limit_before_any_file_is_written(tmp_path, files, links, first):
    """apptainer's ``%files`` copy follows every link (``cp -fLr``), so a chain of links that
    each stay inside the tree can still carry a host file into the image, and links to
    directories can multiply the tree past its byte limit: a tree may hold no symlink at all,
    which git's listing shows before any file is written. The chain's ``t`` leads to a file
    beside the tree, where the role keeps its own files."""
    (tmp_path / "secret").write_text("HOST-SECRET\n")
    record = _run(tmp_path, make_bundle(tmp_path, {**GOOD, **files}, links=links))
    assert record.status == "over_limit" and record.files == [] and len(record.commit or "") == 40
    assert record.reason == f"the tree tracks a symlink, which a submission may not hold: {first!r}"
    assert not (tmp_path / "tree").exists() and not (tmp_path / ".tree-home").exists()


def test_the_role_writes_nothing_outside_its_dest_and_out(tmp_path):
    bundle = make_bundle(tmp_path, GOOD)
    before = set(os.listdir(tmp_path))
    _run(tmp_path, bundle)
    assert set(os.listdir(tmp_path)) - before == {"tree", "source.json", "outputs.json"}
    # The tree is checked out from the bundle's commit, nothing else.
    out = subprocess.run(["git", "bundle", "list-heads", str(bundle)], capture_output=True, text=True, check=True)
    assert _run_commit(tmp_path) in out.stdout


def _run_commit(tmp_path: Path) -> str:
    return SourceRecord.model_validate_json((tmp_path / "source.json").read_text()).commit or ""


# ---------------------------------------------------------------------------
# The limits, read from git's own listing before any file is written
# ---------------------------------------------------------------------------


def test_a_tree_deeper_than_the_interpreters_recursion_limit_is_refused_from_its_listing(tmp_path):
    """A tree nested deeper than Python can recurse is refused before git writes a file of it,
    so no step after the checkout ever walks it: its listing passes the bound on a path's length
    as it streams, at the directory 513 levels deep."""
    depth = sys.getrecursionlimit() + 200
    deep = "/".join(["d"] * (depth - 1)) + "/f"
    record = _run(tmp_path, make_bundle(tmp_path, {**GOOD, deep: "x"}))
    assert record.status == "over_limit" and len(record.commit or "") == 40 and record.files == []
    assert record.reason == ("the tree tracks a path longer than the 1024 bytes allowed: "
                             + ("d/" * 100)[:commit_program.NAME_SHOWN_CHARS])
    assert not (tmp_path / "tree").exists()


def test_a_tree_past_the_depth_limit_within_the_paths_length_is_refused_from_its_listing(tmp_path):
    deep = "/".join(["d"] * TREE_MAX_DEPTH) + "/f"
    record = _run(tmp_path, make_bundle(tmp_path, {**GOOD, deep: "x"}))
    assert record.status == "over_limit" and record.files == []
    assert record.reason == (f"the tree nests {TREE_MAX_DEPTH + 1} levels deep, more than the {TREE_MAX_DEPTH} "
                             f"allowed: {deep}")
    assert not (tmp_path / "tree").exists()


@pytest.mark.parametrize("module, limit, value, files, words", [
    (commit_program, "TREE_MAX_ENTRIES", 2, GOOD | {"a.py": ""}, "the tree tracks more than the 2 paths allowed"),
    (checkout, "TREE_MAX_DEPTH", 2, GOOD | {"a/b/c.py": ""},
     "nests 3 levels deep, more than the 2 allowed: a/b/c.py"),
    (commit_program, "TREE_MAX_PATH_BYTES", 17, GOOD | {"abcdefghijklmno.py": ""},
     "a path longer than the 17 bytes allowed: abcdefghijklmno.py"),
    (checkout, "TREE_MAX_BYTES", 10, GOOD, "more than the 10 allowed"),
], ids=["entries", "depth", "path", "bytes"])
def test_each_limit_refuses_a_tree_past_it_and_checks_out_nothing(tmp_path, monkeypatch, module, limit, value, files,
                                                                 words):
    monkeypatch.setattr(module, limit, value)
    record = _run(tmp_path, make_bundle(tmp_path, files))
    assert record.status == "over_limit" and words in (record.reason or "")
    assert not (tmp_path / "tree").exists() and not (tmp_path / ".tree-home").exists()


def test_a_tree_may_hold_four_gib_and_not_a_byte_more():
    """A submission with a committed compile cache can hold 2.08 GiB."""
    assert TREE_MAX_BYTES == 4 * 2**30
    half = TreeEntry("cache/a.bin", 2 * 2**30, False)
    assert limit_problem([half, TreeEntry("cache/b.bin", 2 * 2**30, False)]) is None
    assert limit_problem([half, TreeEntry("cache/b.bin", 2 * 2**30 + 1, False)]) == (
        f"the tree holds {4 * 2**30 + 1} bytes, more than the {4 * 2**30} allowed")


def test_a_tree_at_every_limit_checks_out(tmp_path, monkeypatch):
    files = GOOD | {"a/b.py": "1"}
    monkeypatch.setattr(commit_program, "TREE_MAX_ENTRIES", 3)
    monkeypatch.setattr(commit_program, "TREE_MAX_DIRECTORIES", 1)
    monkeypatch.setattr(checkout, "TREE_MAX_DEPTH", 2)
    monkeypatch.setattr(commit_program, "TREE_MAX_PATH_BYTES", len("pyproject.toml"))
    monkeypatch.setattr(checkout, "TREE_MAX_BYTES", sum(len(t.encode()) for t in files.values()))
    record = _run(tmp_path, make_bundle(tmp_path, files))
    assert record.status == "ok" and record.reason is None
    assert (tmp_path / "tree" / "a" / "b.py").read_text() == "1"


def _listed(listing: bytes, sizes: list[int], chunk: int = 7) -> list[TreeEntry]:
    """The entries the role makes of ``git ls-tree -r -t -z``'s output *listing*, fed to the
    listing's reader *chunk* bytes at a time, so records cross the chunks' ends, with *sizes*, what
    git's check of the objects gives each file and link."""
    listed: list[tuple[bytes, bytes]] = []
    pieces = (listing[at:at + chunk] for at in range(0, len(listing), chunk))
    assert commit_program.listing_problem(pieces, lambda meta, path: listed.append((meta, path))) is None
    return checkout.tree_entries(listed, sizes)


def test_the_listing_reads_files_links_and_submodules():
    listing = (b"040000 tree 5555555555555555555555555555555555555555\tpkg\0"
               b"100644 blob 1111111111111111111111111111111111111111\tpkg/mod.py\0"
               b"120000 blob 2222222222222222222222222222222222222222\tlink\0"
               b"160000 commit 3333333333333333333333333333333333333333\tvendor/sub\0"
               b"100644 blob 4444444444444444444444444444444444444444\tname with\ttab\0")
    entries = _listed(listing, [12, 9, 0])
    assert entries == [("pkg/mod.py", 12, False), ("link", 9, True), ("vendor/sub", 0, False),
                       ("name with\ttab", 0, False)]
    assert limit_problem(entries) == "the tree tracks a symlink, which a submission may not hold: 'link'"
    assert limit_problem([e for e in entries if not e.link]) is None
    assert limit_problem([]) is None


def test_a_path_that_is_not_utf8_is_listed_byte_for_byte_and_refused():
    entries = _listed(b"100644 blob 5555555555555555555555555555555555555555\tbad\xff.py\0", [1])
    assert entries == [("bad\udcff.py", 1, False)]
    assert limit_problem(entries) == "the tree tracks a path that is not UTF-8: b'bad\\xff.py'"


@pytest.mark.parametrize("mode, target", [("100644", "tree"), ("100644", "commit"), ("100644", "tag"),
                                          ("100755", "tree"), ("120000", "tree")],
                         ids=["file-tree", "file-commit", "file-tag", "executable-tree", "link-tree"])
def test_a_file_whose_object_is_not_a_blob_is_over_limit_before_any_file_is_written(tmp_path, mode, target):
    """``git bundle create`` packs a tree whose entry says file or symbolic link and names a tree,
    a commit or a tag, so a bundle from outside the store may hold one, and git's checkout cannot
    write it. The role checks the object of every file and link before git writes anything and
    refuses the tree, the contestant's."""
    record = _run(tmp_path, make_kind_bundle(tmp_path, mode, target))
    assert (record.status, record.reason, record.files) == (
        "over_limit", f"main's tree names y.py as a file or symbolic link, but its object is a {target}", [])
    assert not (tmp_path / "tree").exists()


def test_the_check_of_the_files_objects_is_held_to_its_time(tmp_path, monkeypatch):
    """git's check of the objects main's tree names is timed
    (``hls.agent.commit.KIND_CHECK_SEC``) like the listing: past it the role ends, infra."""
    _git_shim(tmp_path, monkeypatch, '*" cat-file --batch-check="*', "exec sleep 60")
    monkeypatch.setattr(checkout, "KIND_CHECK_SEC", 2)
    started = time.monotonic()
    with pytest.raises(SystemExit) as caught:
        _run(tmp_path, make_bundle(tmp_path, GOOD))
    assert time.monotonic() - started < 30
    assert str(caught.value.code) == "git cat-file took longer than 2 s"
    assert not (tmp_path / "tree").exists()


def test_the_bytes_the_limit_counts_are_the_blobs_sizes_the_check_reads(tmp_path, monkeypatch):
    """The tree's bytes are its files' blobs' sizes as git's check of the objects reads them; the
    listing reads no object."""
    files = {**GOOD, "data.bin": "x" * 1000}
    total = sum(len(text.encode()) for text in files.values())
    monkeypatch.setattr(checkout, "TREE_MAX_BYTES", total - 1)
    assert _run(tmp_path, make_bundle(tmp_path, files)).reason == (
        f"the tree holds {total} bytes, more than the {total - 1} allowed")


# ---------------------------------------------------------------------------
# The names git's checkout refuses, read from the same listing
# ---------------------------------------------------------------------------

#: Paths of files a tree built with git's plumbing can hold, and whether git's checkout writes
#: each: git 2.53 on Linux with the default configuration, as the role runs it. A symlink is
#: refused whatever its name, so the names git refuses on a symlink's path only never matter.
NAMES = [
    # git's own directory, as NTFS would read the name, as any name on the path.
    (".git", False), (".GIT", False), (".git.", False), (".git ", False), (".git. .", False), (".git:x", False),
    (".git::$INDEX_ALLOCATION", False), ("git~1", False), ("GIT~1", False), ("git~1 ", False), ("git~1.", False),
    ("git~1:", False), ("sub/.git", False), ("a/.git/b", False), (".git:x/y", False),
    # The same after a backslash, which NTFS reads as a separator, but never a name's first character.
    (".git\\x", False), ("git~1\\x", False), ("x\\.git", False), ("x\\git~1", False), ("a\\.git\\b", False),
    ("x\\.git:y", False), ("x\\.git.", False), ("x:\\.git", False), ("\\.git", True),
    # Names close to them that git writes.
    ("git~2", True), ("git~10", True), ("git~0", True), ("git~1.txt", True), (".git~1", True), (" .git", True),
    (".git\t", True), (".g\u0131t", True), ("G\u0130T~1", True), ("a\\b", True), ("a:b", True), ("C:x", True),
    ("foo.", True), ("con", True), ("\\", True), ("a\\", True), ("name\nnewline", True),
    # . and .., and names that only start like them.
    (".", False), ("sub/..", False), ("...", True), (".\\x", True), ("..\\x", True), ("x\\..", True),
    # .gitmodules and the names NTFS reads as it, which git refuses only on a symlink's path.
    (".gitmodules", True), ("sub/.gitmodules/x", True), ("gitmod~1", True), ("gi7eba~1", True),
    ("x\\.gitmodules", True),
    # The filesystem's limit on one name.
    ("x" * 255, True), ("x" * 256, False), ("\u00e9" * 127 + "x", True), ("\u00e9" * 128, False),
]


def _written_by_git(work: Path, path: str) -> tuple[bool, list[TreeEntry]]:
    """Whether git's checkout writes a tree holding the file *path*, by the role's own steps,
    and git's listing of the tree."""
    bundle = make_tree_bundle(work, {"server.py": False, path: False})
    env, dest = git_env(work / "home"), work / "tree"
    receive(bundle, dest, env, sha256_file(bundle)[0])
    entries = checkout.listed_tree(dest, env)
    assert not isinstance(entries, str), entries
    reset = subprocess.run(["git", "-C", str(dest), "reset", "--quiet", "--hard", "HEAD"], env=env,
                           capture_output=True)
    return reset.returncode == 0, entries


def test_the_limits_refuse_exactly_the_names_gits_checkout_refuses(tmp_path):
    """Each path is committed in a tree built with plumbing and checked out by real git; the
    limits, read from git's listing, refuse the tree exactly when git's checkout fails."""
    with ThreadPoolExecutor(8) as pool:
        found = list(pool.map(lambda i: _written_by_git(tmp_path / str(i), NAMES[i][0]), range(len(NAMES))))
    git_differs = [path for (path, writes), (written, _) in zip(NAMES, found) if written != writes]
    limits_differ = [path for (path, writes), (_, entries) in zip(NAMES, found)
                     if (limit_problem(entries) is None) != writes]
    assert git_differs == [] and limits_differ == []


@pytest.mark.parametrize("path, link, words", [
    (".git/hooks/post-checkout", False, "a path git refuses to check out: '.git/hooks/post-checkout'"),
    ("pkg/GIT~1/config", False, "a path git refuses to check out: 'pkg/GIT~1/config'"),
    (".gitmodules", True, "a symlink, which a submission may not hold: '.gitmodules'"),
    ("x" * 256, False, "a name of 256 bytes, longer than the 255 allowed: " + "x" * 200),
    ("bad\udcff.py", False, "a path that is not UTF-8: b'bad\\xff.py'"),
], ids=["dotgit", "short-name", "gitmodules-link", "long-name", "not-utf8"])
def test_a_tree_git_would_not_check_out_is_over_limit_before_any_file_is_written(tmp_path, path, link, words):
    record = _run(tmp_path, make_tree_bundle(tmp_path, {"pyproject.toml": False, "server.py": False, path: link}))
    assert record.status == "over_limit" and record.reason == f"the tree tracks {words}"
    assert not (tmp_path / "tree").exists() and not (tmp_path / ".tree-home").exists()


def test_an_empty_name_is_refused_though_no_bundle_carries_one():
    """git's pack writer refuses an empty name, but its checkout refuses one too."""
    assert git_refuses("a//b") and git_refuses("a/") and git_refuses("")
    assert not git_refuses("a/b")


# --- every object of the bundle, scanned before the limits decide ---------------------------------

#: Three secrets, one for each kind of object a bundle can hide a value in.
KEY, TOKEN, TAG_KEY = "sk-relay-0123456789abcdef", "hf-token-0123456789", "tag-key-0123456789"
VALUES = {"relay": KEY, "hf_token": TOKEN, "tag_key": TAG_KEY}


def _scan_run(tmp_path: Path, bundle: Path, monkeypatch, *extra: str,
              registry: dict[str, str] = VALUES, max_bytes: int = CAP) -> tuple[SourceRecord, PackageScan | None]:
    """Run the role on *bundle* under the cap *max_bytes* with every secret of *registry* in its
    environment; return its record and its scan of the bundle's objects, None when it wrote none."""
    for name, value in registry.items():
        monkeypatch.setenv(name, value)
    scan_out = tmp_path / "package-scan.json"
    argv = _argv(tmp_path, bundle, "--expect", sha256_file(bundle)[0], "--names", ",".join(sorted(registry)),
                 "--scan-out", str(scan_out), *extra, max_bytes=max_bytes)
    assert checkout.main(argv) == 0
    record = SourceRecord.model_validate_json((tmp_path / "source.json").read_text())
    return record, PackageScan.model_validate_json(scan_out.read_text()) if scan_out.exists() else None


def test_every_object_of_the_bundle_is_scanned_its_history_its_messages_and_its_tags(tmp_path, monkeypatch, capfd):
    """A key committed and removed later lives only in the bundle's history; one in a commit
    message or a tag's only in those objects. The tree that is checked out holds none of them,
    and the bundle's stored bytes are compressed: the role reads every object of its clone."""
    bundle = make_history_bundle(tmp_path, [({**GOOD, ".env": f"RELAY={KEY}\n"}, "first"),
                                            ({".env": None}, f"rotate {TOKEN}")], tag=("v1", f"signed {TAG_KEY}"))
    assert all(v.encode() not in bundle.read_bytes() for v in VALUES.values())
    record, scan = _scan_run(tmp_path, bundle, monkeypatch)
    assert record.status == "ok" and ".env" not in record.files
    assert scan == PackageScan(secrets=["hf_token", "relay", "tag_key"], forms=["value", "host"], findings=[],
                               read=["bundle"], areas=[
        PackageScanArea(secret="hf_token", area="bundle", count=1),
        PackageScanArea(secret="relay", area="bundle", count=1),
        PackageScanArea(secret="tag_key", area="bundle", count=1)])
    # The record and what the role printed name secrets, never their values; the record is
    # reported for the manifest beside source.json.
    printed = capfd.readouterr()
    assert all(v not in printed.out + printed.err for v in VALUES.values())
    assert all(v.encode() not in (tmp_path / "package-scan.json").read_bytes() for v in VALUES.values())
    assert set(read_outputs_report(tmp_path / "outputs.json")) == {"source.json", "package-scan.json"}


def test_an_over_limit_tree_is_scanned_before_the_limits_decide(tmp_path, monkeypatch):
    bundle = make_history_bundle(tmp_path, [({**GOOD, ".env": f"RELAY={KEY}\n"}, "submission")],
                                 links={"weights": "/data/weights"})
    record, scan = _scan_run(tmp_path, bundle, monkeypatch)
    assert record.status == "over_limit" and "symlink" in (record.reason or "") and not (tmp_path / "tree").exists()
    assert scan is not None and scan.areas == [PackageScanArea(secret="relay", area="bundle", count=1)]


def test_a_value_in_a_name_is_found_in_its_tree_object(tmp_path, monkeypatch):
    bundle = make_history_bundle(tmp_path, [({**GOOD, f"keys/{KEY}.txt": "nothing\n"}, "submission")])
    _, scan = _scan_run(tmp_path, bundle, monkeypatch)
    assert scan is not None and scan.areas == [PackageScanArea(secret="relay", area="bundle", count=1)]


#: A route's base URL: its host is one of its forms, found in any case.
URL = "https://Relay.Example:8443/v1/anthropic"


@pytest.mark.parametrize("printed", ["relay.example", "RELAY.EXAMPLE", "Relay.Example"])
def test_a_host_in_a_commit_message_is_found_in_the_objects_under_its_label(tmp_path, monkeypatch, printed):
    """The host alone, with no value beside it, in one commit's message and in another's file that a
    later commit removed: each object that holds it counts under the label ``<name>.host``."""
    bundle = make_history_bundle(tmp_path, [({**GOOD, "client.py": f"BASE = 'https://{printed}'\n"}, "first"),
                                            ({"client.py": None}, f"stop dialling {printed}")])
    record, scan = _scan_run(tmp_path, bundle, monkeypatch, registry={**VALUES, "relay_url": URL})
    assert record.status == "ok" and "client.py" not in record.files
    assert scan == PackageScan(secrets=["hf_token", "relay", "relay_url", "tag_key"], forms=["value", "host"],
                               findings=[], read=["bundle"],
                               areas=[PackageScanArea(secret="relay_url.host", area="bundle", count=2)])
    assert "relay.example" not in (tmp_path / "package-scan.json").read_text().lower()


def test_a_bundle_the_role_never_clones_has_no_scan_and_reads_as_unscanned(tmp_path, monkeypatch):
    """Without main nothing is cloned, and a recorded bundle past the byte limit is never read:
    no scan is written, so nothing claims the bundle was read."""
    record, scan = _scan_run(tmp_path, make_bundle(tmp_path, GOOD, branch="dev"), monkeypatch)
    assert record.status == "no_main" and scan is None
    big = tmp_path / "big"
    big.mkdir()
    huge = big / "huge.bundle"
    with open(huge, "wb") as f:
        f.truncate(CAP + 1)
    for name, value in VALUES.items():
        monkeypatch.setenv(name, value)
    argv = _argv(big, huge, "--expect", "ab" * 32, "--size", str(CAP + 1), "--names", "relay",
                 "--scan-out", str(big / "package-scan.json"))
    assert checkout.main(argv) == 0 and not (big / "package-scan.json").exists()


def test_a_bundle_whose_objects_inflate_past_the_bound_is_over_limit_and_never_scanned(tmp_path, monkeypatch):
    """A bundle's objects inflate far past its bytes: what the scan reads in its hour bounds what
    they may inflate to, read from git's own sizes before the scan starts. Past it the bundle is the
    contestant's over-limit artifact, the size in the reason, and no scan claims it was read; at
    the bound it is read as any other."""
    bundle = make_history_bundle(tmp_path, [({**GOOD, "pad.txt": "x" * 20_000, ".env": f"RELAY={KEY}\n"}, "pad")])
    assert bundle.stat().st_size < 4096 + 2000
    record, scan = _scan_run(tmp_path, bundle, monkeypatch, max_bytes=4096)
    assert record.status == "over_limit" and scan is None and not (tmp_path / "tree").exists()
    found = re.fullmatch(r"the bundle's objects inflate to (\d+) bytes, more than the 4096 bytes the scan reads",
                         record.reason or "")
    assert found is not None and int(found.group(1)) > 20_000, record.reason
    assert record.commit is not None and record.files == []
    again = tmp_path / "again"
    again.mkdir()
    record, scan = _scan_run(again, bundle, monkeypatch, max_bytes=int(found.group(1)))
    assert record.status == "ok" and scan is not None
    assert scan.areas == [PackageScanArea(secret="relay", area="bundle", count=1)]
    # One byte under what they inflate to, the bundle is past the bound.
    past = tmp_path / "past"
    past.mkdir()
    record, scan = _scan_run(past, bundle, monkeypatch, max_bytes=int(found.group(1)) - 1)
    assert record.status == "over_limit" and scan is None
    assert record.reason == (f"the bundle's objects inflate to {found.group(1)} bytes, more than the "
                             f"{int(found.group(1)) - 1} bytes the scan reads")


def test_without_secret_names_the_role_scans_nothing(tmp_path):
    record = _run(tmp_path, make_bundle(tmp_path, GOOD))
    assert record.status == "ok" and not (tmp_path / "package-scan.json").exists()
    assert set(read_outputs_report(tmp_path / "outputs.json")) == {"source.json"}


# --- the role's git: one thread, bounded output, its messages drained -----------------------------


def _git_shim(tmp_path: Path, monkeypatch, pattern: str, body: str) -> None:
    """Put a git first on PATH that runs the shell *body* when its arguments, as `` $* ``, match
    the case *pattern*, and is the real git otherwise (and after a *body* that does not exit)."""
    real = subprocess.run(["sh", "-c", "command -v git"], capture_output=True, text=True, check=True).stdout.strip()
    bin_dir = tmp_path / "shim"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "git").write_text(f'#!/bin/sh\ncase " $* " in {pattern}) {body};; esac\nexec {real} "$@"\n')
    (bin_dir / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")


def _spied(monkeypatch) -> list[list[str]]:
    """Every command the role starts from here on, as it started it."""
    started: list[list[str]] = []
    real = subprocess.Popen

    def spy(argv, *args, **kwargs):
        started.append([str(a) for a in argv])
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(checkout.subprocess, "Popen", spy)
    return started


def test_index_packs_messages_are_read_from_their_end_alone(tmp_path, monkeypatch):
    """index-pack writes its messages to a file, of which the role reads only the last
    ``hls.agent.commit.MESSAGE_BYTES`` for a failure's reason (``git_messages``): here it leaves
    terabytes of them, a sparse file, then its last line, and fails."""
    _git_shim(tmp_path, monkeypatch, '*" index-pack "*',
              'truncate -s 4T /proc/self/fd/2; printf "\\nfatal: the last line\\n" >> /proc/self/fd/2; exit 128')
    with pytest.raises(SystemExit) as caught:
        _run(tmp_path, make_bundle(tmp_path, GOOD))
    reason = str(caught.value.code)
    assert reason.startswith("git index-pack failed with status 128: ") and reason.endswith("\nfatal: the last line")
    assert len(reason) <= len("git index-pack failed with status 128: ") + commit_program.MESSAGE_BYTES


def test_the_clones_index_pack_rebuilds_deltas_on_one_thread(tmp_path, monkeypatch):
    """git's index-pack rebuilds deltas on one thread for every two CPUs the node has online, up
    to 20, whatever CPUs the role was given, and each thread holds a delta's base whole: a bundle
    within every bound of the commit point's, a dozen 4 GiB bases each with a small delta, would
    have it hold them all at once. The role's memory is measured for one thread
    (``hls.pipeline.pipelines.CHECKOUT_PEAK_GIB``)."""
    started = _spied(monkeypatch)
    assert _run(tmp_path, make_bundle(tmp_path, GOOD)).status == "ok"
    packs = [argv for argv in started if "index-pack" in argv]
    assert len(packs) == 1 and packs[0][packs[0].index("index-pack") + 1:] == ["--stdin", "--threads=1"]


def _timers(monkeypatch) -> list[float]:
    """The time of every timer started from here on, in order."""
    intervals: list[float] = []
    real = threading.Timer

    def timer(interval, function, *args, **kwargs):
        intervals.append(interval)
        return real(interval, function, *args, **kwargs)

    monkeypatch.setattr(checkout.threading, "Timer", timer)
    return intervals


def test_only_the_listing_and_the_check_of_its_objects_have_times_of_their_own(tmp_path, monkeypatch):
    """Every git command of the checkout runs to the stage's deadline, which ends the role as
    infra: the clone, the read of its objects' sizes, their scan, the reads of main and git's
    checkout start no timer. The tree's listing and the check of the objects it names keep their
    fixed times (``hls.agent.commit.TREE_LIST_SEC``, ``hls.agent.commit.KIND_CHECK_SEC``)."""
    bundle = make_bundle(tmp_path, {**GOOD, ".env": f"RELAY={KEY}\n"})
    timers = _timers(monkeypatch)
    started = _spied(monkeypatch)
    record, scan = _scan_run(tmp_path, bundle, monkeypatch)
    assert record.status == "ok" and scan is not None
    assert [checkout._verb(argv) for argv in started] == [
        "init", "index-pack", "update-ref", "rev-list", "symbolic-ref", "rev-parse", "cat-file", "cat-file", "ls-tree",
        "cat-file", "reset"]
    assert "--batch-all-objects" in started[6] and "--batch" in started[7]
    assert timers == [checkout.TREE_LIST_SEC, checkout.KIND_CHECK_SEC]


def test_a_git_with_no_time_of_its_own_runs_to_its_end(monkeypatch):
    """A git given no time starts no timer and is never ended by the role, however long it runs;
    the same git given a time is ended past it and fails naming it."""
    timers = _timers(monkeypatch)
    slow = ["git", "-c", "alias.slow=!sleep 2; printf done", "slow"]
    env = {**os.environ, **CLONE_ENV}
    started = time.monotonic()
    run = checkout._Git(slow, env)
    with run as out:
        output = out.read()
    run.check()
    assert output == b"done" and time.monotonic() - started >= 2 and timers == []
    timed = checkout._Git(slow, env, 1)
    with timed as out:
        out.read()
    with pytest.raises(checkout.GitFailed, match="^git slow took longer than 1 s$"):
        timed.check()
    assert timers == [1]


def test_a_listing_past_its_time_ends_the_role_named_with_its_session_gone(tmp_path, monkeypatch):
    """A git with a time of its own still running past it has its session ended and the role
    exits naming it, infra, never with an exception nothing attributes, and nothing of the clone
    left."""
    pid = tmp_path / "git.pid"
    _git_shim(tmp_path, monkeypatch, '*" ls-tree "*', f"echo $$ > {pid}; exec sleep 60")
    monkeypatch.setattr(checkout, "TREE_LIST_SEC", 2)
    started = time.monotonic()
    with pytest.raises(SystemExit) as caught:
        _run(tmp_path, make_bundle(tmp_path, GOOD))
    assert time.monotonic() - started < 30
    assert str(caught.value.code) == "git ls-tree took longer than 2 s"
    assert not (tmp_path / "tree").exists()
    hung = int(pid.read_text())
    try:
        assert not Path(f"/proc/{hung}").exists(), "git was left running"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(hung, signal.SIGKILL)


@pytest.mark.parametrize("pattern", ['*" --batch-check="*', '*" --batch-all-objects --batch "*', '*" ls-tree "*'],
                         ids=["size-read", "object-scan", "listing"])
def test_git_writing_its_messages_while_the_role_reads_its_output_never_stalls_it(tmp_path, pattern):
    """git that writes more messages than a pipe holds before its output would block on them if
    the role read its output first and its messages after: they go to a file, so the role, run
    as its stage runs it, finishes."""
    bundle = make_bundle(tmp_path, {**GOOD, ".env": f"RELAY={KEY}\n"})
    real = subprocess.run(["sh", "-c", "command -v git"], capture_output=True, text=True, check=True).stdout.strip()
    bin_dir = tmp_path / "shim"
    bin_dir.mkdir()
    (bin_dir / "git").write_text(f'#!/bin/sh\ncase " $* " in {pattern}) head -c {2**20} /dev/zero >&2;; esac\n'
                                 f'exec {real} "$@"\n')
    (bin_dir / "git").chmod(0o755)
    argv = _argv(tmp_path, bundle, "--expect", sha256_file(bundle)[0], "--names", "relay", "--scan-out",
                 str(tmp_path / "package-scan.json"))
    p = subprocess.run([sys.executable, "-m", "hls.package.checkout", *argv], capture_output=True, text=True,
                       timeout=120, env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "relay": KEY})
    assert p.returncode == 0, p.stderr[-2000:]
    assert SourceRecord.model_validate_json((tmp_path / "source.json").read_text()).status == "ok"


# --- main's tree, as git lists it, read as it streams ----------------------------------------------


def _role(tmp_path: Path, bundle: Path, timeout: float = 120) -> SourceRecord:
    """Run the role as its stage does, a process of its own, held to *timeout*."""
    argv = _argv(tmp_path, bundle, "--expect", sha256_file(bundle)[0])
    p = subprocess.run([sys.executable, "-m", "hls.package.checkout", *argv], capture_output=True, text=True,
                       timeout=timeout)
    assert p.returncode == 0, p.stderr[-2000:]
    return SourceRecord.model_validate_json((tmp_path / "source.json").read_text())


@pytest.mark.parametrize("files, words", [
    (True, "the tree tracks more than the 200000 paths allowed"),
    (False, "the tree holds more than the 200000 directories allowed"),
], ids=["paths", "empty-directories"])
def test_a_tree_whose_subtrees_repeat_is_over_limit_from_the_start_of_its_listing(tmp_path, files, words):
    """A git bomb: nine levels of ten directories that each name the one below list ten billion
    paths (over the empty tree, a billion directories and no path) from eleven objects, within
    every bound on the objects. git's listing of it is read as it streams and ended at the first
    bound it passes, each path and directory counted wherever it appears, so the role neither
    holds the listing nor waits for it: the tree is the contestant's, refused, and nothing is
    checked out."""
    bundle = make_repeated_bundle(tmp_path, 9, 10, files=files)
    started = time.monotonic()
    record = _role(tmp_path, bundle)
    assert time.monotonic() - started < 60
    assert (record.status, record.reason, record.files) == ("over_limit", words, [])
    assert not (tmp_path / "tree").exists()


@pytest.mark.parametrize("limit, value, files, words", [
    ("TREE_MAX_ENTRIES", 3, True, "the tree tracks more than the 3 paths allowed"),
    ("TREE_MAX_DIRECTORIES", 1, True, "the tree holds more than the 1 directories allowed"),
    ("TREE_MAX_DIRECTORIES", 1, False, "the tree holds more than the 1 directories allowed"),
    ("TREE_MAX_PATH_BYTES", 4, True, "the tree tracks a path longer than the 4 bytes allowed: d0/f0"),
    ("TREE_MAX_PATH_BYTES", 1, False, "the tree tracks a path longer than the 1 bytes allowed: d0"),
], ids=["paths", "directories", "empty-directories", "path", "an-empty-directorys-path"])
def test_each_bound_of_the_listing_holds_at_it_and_refuses_one_past(tmp_path, monkeypatch, limit, value, files,
                                                                    words):
    """One level of two directories over two files lists four paths ("d0/f0" the longest, five
    bytes) in two directories; over the empty tree, two directories and no path. Each bound of the
    listing holds a tree at it and refuses one past, a directory's path among the paths."""
    bundle = make_repeated_bundle(tmp_path, 1, 2, files=files)
    monkeypatch.setattr(commit_program, limit, value)
    record = _run(tmp_path, bundle)
    assert (record.status, record.reason) == ("over_limit", words)
    monkeypatch.setattr(commit_program, limit, value + 1)
    again = tmp_path / "again"
    again.mkdir()
    record = _run(again, bundle)
    assert record.status == "ok" and len(record.files) == (4 if files else 0)


def test_a_bundles_tree_objects_past_the_bound_are_over_limit_before_the_listing(tmp_path, monkeypatch):
    """The checkout holds a bundle's tree objects to the commit point's bound
    (``hls.agent.commit.TREE_BYTES_MAX``), a bundle the operator names among them: git's base cache
    for the checkout is that size, so every tree's base stays in it."""
    bundle = make_bundle(tmp_path, GOOD)
    monkeypatch.setattr(checkout, "TREE_BYTES_MAX", 10)
    record = _run(tmp_path, bundle)
    found = re.fullmatch(r"the bundle's tree objects hold (\d+) bytes together, more than the 10 allowed",
                         record.reason or "")
    assert record.status == "over_limit" and found is not None and record.files == []
    monkeypatch.setattr(checkout, "TREE_BYTES_MAX", int(found.group(1)))
    again = tmp_path / "again"
    again.mkdir()
    assert _run(again, bundle).status == "ok"


def test_the_checkout_holds_every_trees_base_in_gits_cache(tmp_path, monkeypatch):
    """git's checkout reads a directory's tree object again wherever the directory appears, and a
    tree stored as a delta is rebuilt from its base each time: with bases larger together than
    git's 96 MiB cache of them, each read rebuilds one whole. The checkout runs with a cache that
    holds every tree the bundle may have (``TREE_BYTES_MAX``), as its listing does."""
    started = _spied(monkeypatch)
    assert _run(tmp_path, make_bundle(tmp_path, GOOD)).status == "ok"
    walks = [argv for argv in started if "reset" in argv or "ls-tree" in argv]
    assert len(walks) == 2
    assert all(argv[1:3] == ["-c", f"core.deltaBaseCacheLimit={checkout.TREE_BYTES_MAX}"] for argv in walks)
    assert commit_program.TREE_CACHE == ("-c", f"core.deltaBaseCacheLimit={128 * 2**20}")


@pytest.mark.parametrize("attributes, name, stored", [
    ("* ident\n", "id.txt", "$Id$\n" * 3),
    ("* working-tree-encoding=UTF-16\n", "wide.txt", "plain text\n"),
    ("* text eol=crlf\n", "lines.txt", "a\nb\n"),
    ("[attr]all text eol=crlf ident working-tree-encoding=UTF-16\n* all\n", "all.txt", "$Id$\n"),
], ids=["ident", "working-tree-encoding", "eol", "a-macro"])
def test_the_trees_own_attributes_never_convert_what_is_checked_out(tmp_path, attributes, name, stored):
    """A tree's ``.gitattributes`` can have git's checkout convert what it writes: ``ident``
    writes 47 bytes for every 4 of ``$Id$``, ``working-tree-encoding`` re-encodes a file in
    memory (git held five times a file it wrote as UTF-32), ``eol`` doubles a file of newlines.
    The limits count each file at its blob's size, so the checkout writes every file as stored:
    its own attributes switch each conversion off, over any of the tree's."""
    files = {**GOOD, ".gitattributes": attributes, name: stored}
    record = _run(tmp_path, make_stored_bundle(tmp_path, {path: text.encode() for path, text in files.items()}))
    assert record.status == "ok"
    assert (tmp_path / "tree" / name).read_bytes() == stored.encode()
    assert (tmp_path / "tree" / ".gitattributes").read_text() == attributes
