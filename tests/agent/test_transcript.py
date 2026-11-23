"""The transcript copy: regular files only, every granted value scrubbed, even across reads, and an
index of what was written. Nothing the agent put in its logs fails the copy: what it cannot use is
skipped and recorded."""

from __future__ import annotations

import errno
import hashlib
import inspect
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from hls.agent import transcript
from hls.substrate.secret_forms import SCRUB_CHUNK_BYTES, ByteScrubber

PROGRAM = Path(transcript.__file__)
KEY = "sk-test-0123456789abcdef"


URL = "https://Relay.Example:8443/v1/anthropic"
HOST = "relay.example"


def _run(src: Path, out: Path, *scrub: str, env: dict | None = None,
         report: Path | None = None) -> subprocess.CompletedProcess:
    extra = ["--outputs-out", str(report)] if report is not None else []
    return subprocess.run([sys.executable, "-I", str(PROGRAM), "--in", str(src), "--out", str(out),
                           *(x for s in scrub for x in ("--scrub", s)), *extra], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", **(env or {})})


@pytest.fixture
def logs(tmp_path):
    src = tmp_path / "logs"
    (src / "sessions" / "projects" / "-workspace").mkdir(parents=True)
    (src / "claude-code.txt").write_text(f'{{"type": "system"}}\nthe key is {KEY}\n')
    (src / "sessions" / "projects" / "-workspace" / "s.jsonl").write_text("{}\n")
    return src


def test_regular_files_are_copied_scrubbed_and_indexed(logs, tmp_path):
    out = tmp_path / "out"
    done = _run(logs, out, "ANTHROPIC_API_KEY=anthropic_key", env={"ANTHROPIC_API_KEY": KEY})
    assert done.returncode == 0, done.stderr
    text = (out / transcript.COPY_DIR / "claude-code.txt").read_text()
    assert KEY not in text and "the key is <redacted:anthropic_key>" in text
    index = json.loads((out / transcript.INDEX_FILE).read_text())
    by_path = {f["path"]: f for f in index["files"]}
    assert sorted(by_path) == ["claude-code.txt", "sessions/projects/-workspace/s.jsonl"]
    for rel, entry in by_path.items():
        data = (out / transcript.COPY_DIR / rel).read_bytes()
        assert entry["size"] == len(data) and entry["sha256"] == hashlib.sha256(data).hexdigest()


def test_links_and_special_files_are_neither_followed_nor_kept(short_dir, tmp_path):
    src = short_dir / "logs"  # the socket's path must stay short
    src.mkdir()
    secret = tmp_path / "outside.txt"
    secret.write_text("host file")
    os.symlink(secret, src / "file-link")
    os.symlink(tmp_path, src / "dir-link")
    os.mkfifo(src / "pipe")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(src / "sock"))  # opening a socket fails outright, so its type must be checked first
    out = tmp_path / "out"
    try:
        assert _run(src, out).returncode == 0
    finally:
        sock.close()
    index = json.loads((out / transcript.INDEX_FILE).read_text())
    assert {s["path"]: s["why"] for s in index["skipped"]} == {
        "file-link": "symlink", "dir-link": "symlink", "pipe": "not a regular file", "sock": "not a regular file"}
    assert not any(p.is_symlink() for p in out.rglob("*"))
    assert "host file" not in "".join(p.read_text() for p in out.rglob("*") if p.is_file())


def test_a_value_split_across_two_reads_is_still_replaced_and_what_was_written_is_returned(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    scrubber = transcript.scrubber_for(["K=k"], {"K": KEY})
    for k in range(len(KEY) + 2):
        data = "x" * (SCRUB_CHUNK_BYTES - k) + (KEY + "y" * 10) * 3
        (src / "f.txt").write_text(data)
        fd = os.open(src / "f.txt", os.O_RDONLY)
        try:
            size, sha256 = transcript.copy_scrubbed(fd, out / "f.txt", scrubber)
        finally:
            os.close(fd)
        written = (out / "f.txt").read_bytes()
        assert written == data.replace(KEY, "<redacted:k>").encode(), k
        assert (size, sha256) == (len(written), hashlib.sha256(written).hexdigest())


#: A route's base URL whose host is an IPv4 address written in hex: its forms are the URL, the
#: host as written and the address as a client prints it.
MIRROR = "https://0x8.8.4.4:8443/v1"
#: Each form a log may print, with the label its marker names: a credential, a URL, its host in
#: another case, and an address host as written and as printed.
PRINTED_FORMS = [(KEY, "anthropic_key"), (URL, "relay_url"), ("RELAY.example", "relay_url.host"),
                 (MIRROR, "mirror_url"), ("0x8.8.4.4", "mirror_url.host"), ("8.8.4.4", "mirror_url.host")]


def test_every_form_split_at_the_read_boundary_is_replaced_whole(tmp_path, monkeypatch):
    """Each form straddles the copy's read boundary at every position: a URL split after its host
    is replaced whole too, never its host alone with the rest of the URL left in the copy."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    expected = {}
    for i, (text, label) in enumerate(PRINTED_FORMS):
        for k in range(1, len(text)):
            head = b"x" * (SCRUB_CHUNK_BYTES - k)
            (src / f"{i}-{k}.txt").write_bytes(head + text.encode() + b" tail\n")
            expected[f"{i}-{k}.txt"] = head + f"<redacted:{label}>".encode() + b" tail\n"
    monkeypatch.setenv("KEY", KEY)
    monkeypatch.setenv("URL", URL)
    monkeypatch.setenv("MIRROR", MIRROR)
    code = transcript.main(["--in", str(src), "--out", str(out), "--scrub", "KEY=anthropic_key",
                            "--scrub", "URL=relay_url", "--scrub", "MIRROR=mirror_url"])
    assert code == 0
    wrong = [name for name, data in expected.items() if (out / transcript.COPY_DIR / name).read_bytes() != data]
    assert wrong == []


def test_values_are_replaced_longest_first(tmp_path):
    long, short = "sk-test-0123456789abcdef-long", "sk-test-0123456789abcdef"
    scrubber = transcript.scrubber_for(["A=a_key", "B=b_key"], {"A": short, "B": long})
    assert scrubber.scrub(f"{long} {short}".encode()) == b"<redacted:b_key> <redacted:a_key>"


@pytest.mark.parametrize("scrub, env", [("ANTHROPIC_API_KEY=anthropic_key", {}), ("NOEQUALS", {"NOEQUALS": "x"})],
                         ids=["unset", "malformed"])
def test_a_scrub_it_cannot_do_fails_the_copy(logs, tmp_path, scrub, env):
    out = tmp_path / "out"
    done = _run(logs, out, scrub, env=env)
    assert done.returncode == 1 and "hls-transcript:" in done.stderr
    assert not (out / transcript.INDEX_FILE).exists()


def _index(out: Path) -> dict:
    return json.loads((out / transcript.INDEX_FILE).read_text())


def test_an_in_directory_that_is_a_link_is_skipped_and_the_copy_still_succeeds(logs, tmp_path):
    """The program runs on the host, so even ``--in`` is not followed: the agent could have
    replaced its logs directory with a link to anywhere. It is skipped, never a failure."""
    os.symlink(logs, tmp_path / "logs-link")
    out = tmp_path / "out"
    done = _run(tmp_path / "logs-link", out)
    assert done.returncode == 0, done.stderr
    assert _index(out) == {"files": [], "skipped": [{"path": ".", "why": "symlink"}]}
    assert not (out / "claude-code.txt").exists()


@pytest.mark.parametrize("what", ["file", "directory", "in"])
def test_what_the_agent_made_unreadable_is_skipped_with_its_errno_and_the_rest_copied(logs, tmp_path, what):
    """A file or directory the agent took its own permissions from (chmod 000), ``--in`` among
    them, is skipped with the errno that refused it, and the copy succeeds: a failure here would
    turn a graded build into infra that is never retried."""
    if os.geteuid() == 0:
        pytest.skip("root reads whatever the mode")
    locked = {"file": logs / "claude-code.txt", "directory": logs / "sessions", "in": logs}[what]
    locked.chmod(0)
    out = tmp_path / "out"
    try:
        done = _run(logs, out)
    finally:
        locked.chmod(0o700)
    assert done.returncode == 0, done.stderr
    index = _index(out)
    path = {"file": "claude-code.txt", "directory": "sessions", "in": "."}[what]
    assert {"path": path, "why": "unreadable: EACCES"} in index["skipped"]
    copied = {f["path"] for f in index["files"]}
    assert copied == {"file": {"sessions/projects/-workspace/s.jsonl"}, "directory": {"claude-code.txt"},
                      "in": set()}[what]


def test_a_directory_nested_past_the_depth_bound_is_skipped_at_the_bound(logs, tmp_path):
    deep = logs / "nest"
    for _ in range(transcript.MAX_DEPTH + 2):
        deep = deep / "d"
    deep.mkdir(parents=True)
    (deep / "bottom.txt").write_text("x\n")
    out = tmp_path / "out"
    done = _run(logs, out)
    assert done.returncode == 0, done.stderr
    [skip] = [s for s in _index(out)["skipped"] if s["path"].startswith("nest")]
    assert skip == {"path": "nest" + "/d" * transcript.MAX_DEPTH, "why": f"deeper than {transcript.MAX_DEPTH} levels"}


def test_the_walk_keeps_its_own_stack_whatever_the_interpreters_recursion_limit(tmp_path, monkeypatch):
    """The walk keeps its own stack, so a nesting deeper than the frames the interpreter has
    left is no failure."""
    monkeypatch.setattr(transcript, "MAX_DEPTH", 300)
    src, out = tmp_path / "in", tmp_path / "out"
    deep = src
    for _ in range(200):
        deep = deep / "d"
    deep.mkdir(parents=True)
    (deep / "bottom.txt").write_text("x\n")
    out.mkdir()
    limit = sys.getrecursionlimit()
    frames = len(inspect.stack(0))
    sys.setrecursionlimit(frames + 60)
    try:
        index = transcript.copy_tree(src, out, ByteScrubber([]))
    finally:
        sys.setrecursionlimit(limit)
    assert [f["path"].count("/") for f in index["files"]] == [200] and index["skipped"] == []


def test_a_path_past_the_path_bound_is_skipped(tmp_path, monkeypatch):
    """Every copied path must be one the machine can make under the output directory."""
    monkeypatch.setattr(transcript, "MAX_PATH_BYTES", 12)
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "abc" / "defgh").mkdir(parents=True)
    out.mkdir()
    (src / "abc" / "x.txt").write_text("x")
    (src / "abc" / "defgh" / "y.txt").write_text("y")
    index = transcript.copy_tree(src, out, ByteScrubber([]))
    assert [f["path"] for f in index["files"]] == ["abc/x.txt"]
    assert index["skipped"] == [{"path": "abc/defgh/y.txt", "why": "its path is longer than 12 bytes"}]


def _held(out: Path, index: dict) -> int:
    """The bytes the copy in *out* and *index*, written as the program writes it, hold together."""
    return sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) + len(json.dumps(index, indent=2))


def test_files_past_the_size_bound_are_skipped_and_the_rest_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "MAX_BYTES", 1000)
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "a.txt").write_text("123456")
    (src / "b.txt").write_text("1" * 1000)
    (src / "c.txt").write_text("1234")
    index = transcript.copy_tree(src, out, ByteScrubber([]))
    assert [f["path"] for f in index["files"]] == ["a.txt", "c.txt"]
    assert index["skipped"] == [{"path": "b.txt", "why": "its 1000 bytes would take the copy past 1000"}]
    assert _held(out, index) <= 1000


def test_a_file_whose_scrubbed_bytes_pass_the_bound_is_skipped_and_leaves_nothing(tmp_path, monkeypatch):
    """Scrubbing can make a file larger, a short host replaced by its longer name: the bound holds
    the bytes the copy writes, not the bytes the agent wrote."""
    monkeypatch.setattr(transcript, "MAX_BYTES", 1500)
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "dial.txt").write_text("h.io " * 100)
    (src / "z.txt").write_text("kept")
    scrubber = transcript.scrubber_for(["ANTHROPIC_BASE_URL=relay_url"], {"ANTHROPIC_BASE_URL": "https://h.io/v1"})
    index = transcript.copy_tree(src, out, scrubber)
    assert len("h.io " * 100) < 1000 and [f["path"] for f in index["files"]] == ["z.txt"]
    assert index["skipped"] == [{"path": "dial.txt", "why": "its bytes, scrubbed, would take the copy past 1500"}]
    assert not (out / "dial.txt").exists() and _held(out, index) <= 1500


def test_a_file_needs_room_for_its_index_entry_too(tmp_path, monkeypatch):
    """A file that would fill every byte the index leaves has no room for the entry naming it."""
    monkeypatch.setattr(transcript, "MAX_BYTES", 10_000)
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    size = transcript.MAX_BYTES - transcript._Index().used
    (src / "big.txt").write_bytes(b"x" * size)
    index = transcript.copy_tree(src, out, ByteScrubber([]))
    assert index["files"] == []
    assert index["skipped"] == [{"path": "big.txt", "why": f"its {size} bytes would take the copy past 10000"}]


def test_a_copy_filled_to_the_edge_keeps_room_for_the_entry_that_stops_it(tmp_path, monkeypatch):
    """A file that takes all the room left leaves none for the next entry: the walk stops with the
    entry that says so, which fits, since its room was kept from the start."""
    monkeypatch.setattr(transcript, "MAX_BYTES", 4000)
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "a.txt").write_bytes(b"x" * transcript._Index().room("a.txt"))
    (src / "b.txt").symlink_to(src / "a.txt")
    index = transcript.copy_tree(src, out, ByteScrubber([]))
    assert [f["path"] for f in index["files"]] == ["a.txt"]
    assert index["skipped"] == [{"path": ".", "why": "the copy and its index reached 4000 bytes: the rest is not read"}]
    assert _held(out, index) <= 4000


def test_the_index_counts_at_least_the_bytes_it_is_written_in():
    """Each entry counts what it adds to ``INDEX.json``, escapes and indentation included, so the
    bound the walk keeps holds for the index as written."""
    index = transcript._Index()
    sizes = [0, 7, 10**6, 10**9]
    for size, path in zip(sizes, ["a", '\u00fc/\x01"q', "x" * 300, "\udcff/\\n"], strict=True):
        index.add_file(path, size, "f" * 64)
        assert index.add_skipped(f"{path}/s", "symlink")
    assert len(json.dumps(index.record(), indent=2)) + sum(sizes) <= index.used


def test_the_copy_and_its_index_together_stay_within_the_bound(tmp_path, monkeypatch):
    """The index grows with every entry it names, so it counts against the bound too: once the next
    entry would not fit, the walk stops, and the index says so."""
    monkeypatch.setattr(transcript, "MAX_BYTES", 8000)
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    for n in range(300):
        (src / f"{n:03d}.txt").write_text("x")
    index = transcript.copy_tree(src, out, ByteScrubber([]))
    assert 0 < len(index["files"]) < 300
    assert index["skipped"][-1] == {"path": ".", "why": "the copy and its index reached 8000 bytes: the rest is not read"}
    assert _held(out, index) <= 8000


def test_entries_past_the_count_bound_are_not_read(tmp_path, monkeypatch):
    """The copy reads at most ``MAX_ENTRIES`` names in all, whatever directory they are in: here
    the top's three, so ``a``'s own are past the bound."""
    monkeypatch.setattr(transcript, "MAX_ENTRIES", 3)
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "a").mkdir(parents=True)
    out.mkdir()
    for name in ("a/1", "a/2", "b", "c"):
        (src / name).write_text("x")
    index = transcript.copy_tree(src, out, ByteScrubber([]))
    assert [f["path"] for f in index["files"]] == ["b", "c"]
    assert index["skipped"] == [{"path": "a", "why": "past the first 3 entries: the rest is not read"}]


def test_a_directory_of_more_names_than_the_bound_is_listed_only_to_the_bound(tmp_path, monkeypatch):
    """A copy that sorted a directory's every name before its bound applied would let an agent
    that made millions of names end the transcript role after the commit point. No listing hands
    out more names than the copy's bound, which counts every name already read or held, and the
    rest is skipped and recorded."""
    from hls.agent import listing

    monkeypatch.setattr(transcript, "MAX_ENTRIES", 5)
    counted = _CountedScandir(listing.os.scandir)
    monkeypatch.setattr(listing.os, "scandir", counted)
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "many").mkdir(parents=True)
    out.mkdir()
    for i in range(40):
        (src / "many" / f"{i:02d}").write_text("x")
    index = transcript.copy_tree(src, out, ByteScrubber([]))
    assert counted.read <= 5 + 2
    assert len(index["files"]) <= 4
    assert {"path": "many", "why": "past the first 5 entries: the rest is not read"} in index["skipped"]


def test_a_file_whose_read_fails_is_skipped_and_leaves_nothing_behind(tmp_path, monkeypatch):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "a.txt").write_text("a" * 100)
    (src / "b.txt").write_text("b")
    real = os.read
    calls = []

    def failing(fd, n):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError(errno.EIO, "Input/output error")
        return real(fd, 1)

    monkeypatch.setattr(transcript.os, "read", failing)
    index = transcript.copy_tree(src, out, ByteScrubber([]))
    assert index["skipped"] == [{"path": "a.txt", "why": "unreadable: EIO"}]
    assert [f["path"] for f in index["files"]] == ["b.txt"] and not (out / "a.txt").exists()


def test_a_copy_the_machine_cannot_write_fails(tmp_path):
    """The machine's failure is the harness's: a file of the copy that cannot be written ends
    the program with status 1, though the index could be."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    (src / "claude-code.txt").write_text("x\n")
    (out / transcript.COPY_DIR / "claude-code.txt").mkdir(parents=True)
    done = _run(src, out)
    assert done.returncode == 1 and "hls-transcript:" in done.stderr
    assert not (out / transcript.INDEX_FILE).exists()


def test_a_write_the_machine_fails_is_raised_never_skipped(tmp_path, monkeypatch):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "a.txt").write_text("a")

    def full(src_fd, dst, scrubber, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(transcript, "copy_scrubbed", full)
    with pytest.raises(OSError, match="No space left"):
        transcript.copy_tree(src, out, ByteScrubber([]))


def test_a_directory_that_cannot_be_listed_is_skipped_with_its_errno(tmp_path, monkeypatch):
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "a").mkdir(parents=True)
    out.mkdir()
    (src / "a" / "x.txt").write_text("x")
    (src / "b.txt").write_text("b")
    real = os.scandir

    def scandir(fd):
        if os.path.basename(os.readlink(f"/proc/self/fd/{fd}")) == "a":
            raise OSError(errno.EIO, "Input/output error")
        return real(fd)

    monkeypatch.setattr(transcript.os, "scandir", scandir)
    index = transcript.copy_tree(src, out, ByteScrubber([]))
    assert index["skipped"] == [{"path": "a", "why": "unreadable: EIO"}]
    assert [f["path"] for f in index["files"]] == ["b.txt"]


def test_a_link_below_the_top_is_kept_out_at_every_depth(logs, tmp_path):
    secret = tmp_path / "outside"
    (secret / "deep").mkdir(parents=True)
    (secret / "deep" / "host.txt").write_text("host file")
    os.symlink(secret, logs / "sessions" / "projects" / "linked")
    out = tmp_path / "out"
    assert _run(logs, out).returncode == 0
    index = json.loads((out / transcript.INDEX_FILE).read_text())
    assert {"path": "sessions/projects/linked", "why": "symlink"} in index["skipped"]
    assert "host file" not in "".join(p.read_text() for p in out.rglob("*") if p.is_file())


def test_the_module_runs_as_the_stage_starts_it(logs, tmp_path):
    out = tmp_path / "out"
    src = Path(transcript.__file__).parents[2]
    done = subprocess.run([sys.executable, "-m", "hls.agent.transcript", "--in", str(logs), "--out", str(out),
                           "--scrub", "ANTHROPIC_API_KEY=anthropic_key"], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(src), "ANTHROPIC_API_KEY": KEY})
    assert done.returncode == 0, done.stderr
    assert "<redacted:anthropic_key>" in (out / transcript.COPY_DIR / "claude-code.txt").read_text()


def test_a_url_values_host_is_scrubbed(tmp_path):
    """A base URL is a secret, and so is its host: a line that names the host alone, as a CLI
    printing the address it dialled does, comes out with the host replaced."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    (src / "stream.txt").write_text(f"base {URL}\nconnecting to {HOST}:8443\n")
    done = _run(src, out, "ANTHROPIC_BASE_URL=relay_url", env={"ANTHROPIC_BASE_URL": URL})
    assert done.returncode == 0, done.stderr
    text = (out / transcript.COPY_DIR / "stream.txt").read_text()
    assert text == "base <redacted:relay_url>\nconnecting to <redacted:relay_url.host>:8443\n"


def test_a_url_values_host_that_identifies_no_machine_is_left_and_the_value_replaced(tmp_path):
    """The copy takes its forms from ``hls.substrate.secret_forms``, the scrubber's: a host such
    as ``kimi`` names no particular machine, so a route's name in the agent's logs stays."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    (src / "stream.txt").write_text("base https://kimi:8443/v1 on route kimi\n")
    done = _run(src, out, "ANTHROPIC_BASE_URL=kimi_url", env={"ANTHROPIC_BASE_URL": "https://kimi:8443/v1"})
    assert done.returncode == 0, done.stderr
    assert (out / transcript.COPY_DIR / "stream.txt").read_text() == "base <redacted:kimi_url> on route kimi\n"


@pytest.mark.parametrize("printed", ["RELAY.EXAMPLE", "Relay.Example", "rElAy.eXaMpLe"])
def test_a_host_printed_in_mixed_case_is_scrubbed(tmp_path, printed):
    """A URL's host is case-insensitive, and a CLI may print it in any case: the copy replaces it
    whatever its case, also where it straddles two reads."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    data = f"dial {printed} then {printed}/v1\n"
    scrubber = transcript.scrubber_for(["ANTHROPIC_BASE_URL=relay_url"], {"ANTHROPIC_BASE_URL": URL})
    for k in range(len(HOST) + 1):
        # The first read ends k bytes into the first host printed.
        (src / "f.txt").write_text("x" * (SCRUB_CHUNK_BYTES - len("dial ") - k) + data)
        fd = os.open(src / "f.txt", os.O_RDONLY)
        try:
            transcript.copy_scrubbed(fd, out / "f.txt", scrubber)
        finally:
            os.close(fd)
        text = (out / "f.txt").read_text()
        assert printed not in text and text.count("<redacted:relay_url.host>") == 2, k


def test_a_credential_is_replaced_in_the_case_it_was_given(tmp_path):
    """A credential is a value of its own case: only a URL's host is matched in any case."""
    scrubber = transcript.scrubber_for(["A=a_key"], {"A": KEY})
    assert scrubber.scrub(f"{KEY} {KEY.upper()}".encode()) == f"<redacted:a_key> {KEY.upper()}".encode()


def test_the_index_is_reported_in_place_by_its_writer(logs, tmp_path):
    """The index is the transcript's one output: the program reports its sha256 and size, keyed
    by its path under the outputs directory, into the report file the stage made beforehand,
    written in place, so a full node-local disk cannot refuse it."""
    outputs = tmp_path / "outputs"
    out = outputs / "transcript"
    report = tmp_path / "role" / "outputs.json"
    report.parent.mkdir()
    report.write_bytes(b" " * 4096)
    inode = report.stat().st_ino
    done = _run(logs, out, report=report)
    assert done.returncode == 0, done.stderr
    data = (out / transcript.INDEX_FILE).read_bytes()
    assert report.stat().st_ino == inode
    assert json.loads(report.read_text()) == {
        "transcript/INDEX.json": {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}}


@pytest.mark.parametrize("name", [transcript.INDEX_FILE, f".tmp-{transcript.INDEX_FILE}"])
def test_an_entry_named_as_the_index_never_meets_it(logs, tmp_path, name):
    """The index and its temporary file live beside the copied tree, so an entry of the agent's
    logs under either name is copied as any other, and the index is written."""
    (logs / name / "x").mkdir(parents=True)
    (logs / name / "x" / "f.txt").write_text("f")
    out = tmp_path / "out"
    done = _run(logs, out)
    assert done.returncode == 0, done.stderr
    index = _index(out)
    assert f"{name}/x/f.txt" in {f["path"] for f in index["files"]}
    assert (out / transcript.COPY_DIR / name / "x" / "f.txt").read_text() == "f"


class _CountedScandir:
    """``os.scandir`` that counts the entries every listing hands out, as an agent that filled a
    directory with names would have them read."""

    def __init__(self, real) -> None:
        self.real = real
        #: Entries handed out by every listing so far.
        self.read = 0

    def __call__(self, target):
        counter = self

        class _Listing:
            def __init__(self) -> None:
                self.it = counter.real(target)

            def __enter__(self):
                self.it.__enter__()
                return self

            def __exit__(self, *exc):
                return self.it.__exit__(*exc)

            def __iter__(self):
                for entry in self.it:
                    counter.read += 1
                    yield entry

        return _Listing()


def test_the_program_needs_only_the_standard_library():
    """The role imports nothing but the standard library and hls's own stdlib-only modules: its
    forms (``hls.substrate.secret_forms``) and its bounded listings (``hls.agent.listing``)."""
    code = "import sys\nsys.modules['pydantic'] = None\nsys.modules['harbor'] = None\nimport hls.agent.transcript\n"
    src = Path(transcript.__file__).parents[2]
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(src)})
    assert done.returncode == 0, done.stderr
