from __future__ import annotations

import gzip
import io
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from hls.substrate import secret_scan
from hls.substrate.secret_forms import SecretForm, registry_forms
from hls.substrate.secret_scan import (CUT_PREFIX, FOUND_PREFIX, READ_PREFIX, UNREAD_PREFIX, files, find, scan_file,
                                       scan_stream)

VALUE = b"sk-relay-0123456789"
OTHER = b"hf-token-abcdefgh"


def _forms(**values: bytes) -> list[SecretForm]:
    """The forms of each secret of *values*, name to value."""
    return registry_forms({name: value.decode() for name, value in values.items()})


@pytest.mark.parametrize("chunk", range(1, len(VALUE) + 3))
@pytest.mark.parametrize("offset", [0, 3, 7])
def test_a_value_is_found_wherever_the_reads_split_it(chunk, offset):
    data = b"x" * offset + VALUE + b"y" * 5
    assert find(io.BytesIO(data), _forms(relay=VALUE, hub=OTHER), chunk=chunk) == {"relay"}


def test_the_reads_overlap_by_the_longest_value_so_a_short_one_is_not_missed():
    data = b"." * 30 + OTHER + b"." * 3 + VALUE
    assert find(io.BytesIO(data), _forms(relay=VALUE, hub=OTHER), chunk=8) == {"relay", "hub"}


def test_gzip_is_read_by_its_magic_whatever_the_name_and_member_after_member(tmp_path):
    path = tmp_path / "trace.bin"
    path.write_bytes(gzip.compress(b"first member") + gzip.compress(b"second " + VALUE))
    assert scan_file(path, _forms(relay=VALUE)) == {"relay"}
    plain = tmp_path / "plain.jsonl.gz"
    plain.write_bytes(b"not compressed " + VALUE)
    assert scan_file(plain, _forms(relay=VALUE)) == {"relay"}


def test_a_value_only_in_the_compressed_bytes_is_not_a_hit(tmp_path):
    path = tmp_path / "x.gz"
    path.write_bytes(gzip.compress(b"nothing to see"))
    assert scan_file(path, _forms(relay=VALUE)) == set()


def test_a_file_read_as_stored_is_never_decompressed(tmp_path):
    path = tmp_path / "artifact.out"
    path.write_bytes(gzip.compress(VALUE))
    assert scan_file(path, _forms(relay=VALUE), decompress=False) == set()
    path.write_bytes(b"\x1f\x8b" + VALUE)
    assert scan_file(path, _forms(relay=VALUE), decompress=False) == {"relay"}


def test_files_walks_directories_once_each_whatever_their_names_and_skips_links(tmp_path):
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / ".hidden").mkdir()
    (d / "a").write_bytes(b"a")
    (d / "sub" / "b").write_bytes(b"b")
    (d / ".tmp-c").write_bytes(b"c")
    (d / ".hidden" / "e").write_bytes(b"e")
    (tmp_path / "outside").write_bytes(VALUE)
    (d / "link").symlink_to(tmp_path / "outside")
    (d / ".hidden" / "out").symlink_to(tmp_path)
    listed = [d, d / "a", d / "link", d / ".hidden" / "e"]
    # A dot-named file is read and a dot-named directory walked, in the order of their paths; a
    # link, to a file or a directory, is never followed.
    assert list(files(listed)) == [d / ".hidden" / "e", d / ".tmp-c", d / "a", d / "sub" / "b"]
    with pytest.raises(FileNotFoundError):
        list(files([tmp_path / "gone"]))


def _run(tmp_path, monkeypatch, capsys, paths, raw=()):
    monkeypatch.setenv("relay", VALUE.decode())
    monkeypatch.setenv("hub", OTHER.decode())
    listing = tmp_path / "list.txt"
    listing.write_text("".join(f"{p}\n" for p in paths))
    report = tmp_path / "found.jsonl"
    argv = ["--names", "hub,relay", "--list", str(listing), "--report", str(report)]
    if raw:
        stored = tmp_path / "raw.txt"
        stored.write_text("".join(f"{p}\n" for p in raw))
        argv += ["--raw", str(stored)]
    code = secret_scan.main(argv)
    out, err = capsys.readouterr()
    return code, _found_and_cut(report), out + err


def _found_and_cut(report) -> str:
    """The lines of *report* that name a form found or a stream cut short; what the role read is
    the subject of its own tests."""
    lines = report.read_text().splitlines(keepends=True) if report.exists() else []
    return "".join(line for line in lines if not line.startswith(READ_PREFIX))


def test_the_role_reports_files_and_names_never_values(tmp_path, monkeypatch, capsys):
    (tmp_path / "clean").write_bytes(b"clean")
    (tmp_path / "hit.gz").write_bytes(gzip.compress(VALUE + b" and " + OTHER))
    code, out, err = _run(tmp_path, monkeypatch, capsys, [tmp_path / "clean", tmp_path / "hit.gz"])
    assert code == 0
    hits = [json.loads(line[len(FOUND_PREFIX):]) for line in out.splitlines()]
    assert hits == [{"file": str(tmp_path / "hit.gz"), "secret": "hub"},
                    {"file": str(tmp_path / "hit.gz"), "secret": "relay"}]
    assert VALUE.decode() not in out + err and OTHER.decode() not in out + err


def test_the_role_exits_0_once_it_read_every_file_and_1_when_it_cannot(tmp_path, monkeypatch, capsys):
    (tmp_path / "clean").write_bytes(b"clean")
    assert _run(tmp_path, monkeypatch, capsys, [tmp_path / "clean"])[0] == 0
    code, out, err = _run(tmp_path, monkeypatch, capsys, [tmp_path / "clean", tmp_path / "gone"])
    assert code == 1 and "secret scan failed" in err


#: A gzip stream of the value, bytes that do not compress, and the other value; and what cutting
#: it leaves.
WHOLE = gzip.compress(VALUE + random.Random(7).randbytes(300_000) + OTHER, mtime=0)
CUTS = {"ends early": WHOLE[:len(WHOLE) // 2], "ends in its trailer": WHOLE[:-3],
        "a corrupt block": WHOLE[:len(WHOLE) // 2] + bytes(64) + WHOLE[len(WHOLE) // 2 + 64:]}


@pytest.mark.parametrize("cut", sorted(CUTS))
def test_a_stream_cut_short_is_searched_as_far_as_it_goes_and_as_stored(tmp_path, cut):
    path = tmp_path / "replies.jsonl.gz"
    path.write_bytes(CUTS[cut])
    scanned = scan_stream(path, _forms(relay=VALUE, hub=OTHER))
    assert scanned.cut and "relay" in scanned.labels


def test_a_value_stored_after_a_cut_is_found(tmp_path):
    # A stream of compressed text, cut, with the value stored after it: the value's bytes are no
    # plain bytes of the stream, so only the stored bytes hold it.
    path = tmp_path / "x.gz"
    path.write_bytes(gzip.compress(b"a" * 100_000, mtime=0)[:60] + OTHER)
    scanned = scan_stream(path, _forms(hub=OTHER))
    assert (scanned.labels, scanned.cut) == ({"hub"}, True)


def test_whole_streams_are_not_cut_and_members_follow_one_another(tmp_path):
    path = tmp_path / "x.gz"
    path.write_bytes(gzip.compress(b"a" * 70_000) + gzip.compress(b"b" + VALUE))
    scanned = scan_stream(path, _forms(relay=VALUE))
    assert (scanned.labels, scanned.cut) == ({"relay"}, False)


def test_the_role_reports_a_cut_stream_and_carries_on(tmp_path, monkeypatch, capsys):
    (tmp_path / "cut.gz").write_bytes(CUTS["ends early"])
    (tmp_path / "hit").write_bytes(OTHER)
    code, report, logged = _run(tmp_path, monkeypatch, capsys, [tmp_path / "cut.gz", tmp_path / "hit"])
    lines = report.splitlines()
    assert code == 0
    assert CUT_PREFIX + json.dumps({"file": str(tmp_path / "cut.gz")}) in lines
    assert FOUND_PREFIX + json.dumps({"file": str(tmp_path / "hit"), "secret": "hub"}) in lines
    assert "1 stream cut short" in logged


def test_the_role_reads_its_raw_list_as_stored(tmp_path, monkeypatch, capsys):
    (tmp_path / "broken.out").write_bytes(gzip.compress(b"x" * 1000)[:20])
    (tmp_path / "hit.err").write_bytes(b"\x1f\x8b" + OTHER)
    code, out, _ = _run(tmp_path, monkeypatch, capsys, [], raw=[tmp_path / "broken.out", tmp_path / "hit.err"])
    assert code == 0
    assert [json.loads(line[len(FOUND_PREFIX):]) for line in out.splitlines()] == [
        {"file": str(tmp_path / "hit.err"), "secret": "hub"}]


def test_the_role_reads_values_from_its_environment_only(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("relay", raising=False)
    listing = tmp_path / "list.txt"
    listing.write_text("")
    assert secret_scan.main(["--names", "relay", "--list", str(listing), "--report", str(tmp_path / "r")]) == 1
    assert "relay" in capsys.readouterr().err and "relay" not in os.environ


def _main(tmp_path, monkeypatch, listed, raw):
    monkeypatch.setenv("relay", VALUE.decode())
    lst, stored = tmp_path / "list.txt", tmp_path / "raw.txt"
    lst.write_text("".join(f"{p}\n" for p in listed))
    stored.write_text("".join(f"{p}\n" for p in raw))
    return secret_scan.main(["--names", "relay", "--list", str(lst), "--raw", str(stored),
                             "--report", str(tmp_path / "found.jsonl")])


def test_a_raw_file_under_a_listed_directory_is_read_as_stored(tmp_path, monkeypatch):
    # A file named by --raw, or under a directory --raw names, is read as stored even when
    # --list also holds it.
    outputs = tmp_path / "outputs"
    (outputs / "transcript").mkdir(parents=True)
    (outputs / "transcript" / "agent.log").write_bytes(gzip.compress(b"x" * 1000)[:20])
    (outputs / "named.log").write_bytes(b"\x1f\x8b" + b"cut")
    assert _main(tmp_path, monkeypatch, [outputs, outputs / "named.log"],
                 [outputs / "transcript", outputs / "named.log"]) == 0


def test_each_file_is_read_once_whichever_lists_name_it(tmp_path, monkeypatch, capsys):
    outputs = tmp_path / "outputs"
    (outputs / "logs").mkdir(parents=True)
    (outputs / "logs" / "a.err").write_bytes(VALUE)
    (outputs / "b.json").write_bytes(VALUE)
    listed = [outputs, outputs / "b.json", outputs / "logs" / "a.err"]
    assert _main(tmp_path, monkeypatch, listed, [outputs / "logs", outputs / "logs" / "a.err"]) == 0
    lines = (tmp_path / "found.jsonl").read_text().splitlines()
    hits = [json.loads(line[len(FOUND_PREFIX):])["file"] for line in lines if line.startswith(FOUND_PREFIX)]
    read = [json.loads(line[len(READ_PREFIX):])["file"] for line in lines if line.startswith(READ_PREFIX)]
    assert sorted(hits) == sorted(read) == sorted([str(outputs / "b.json"), str(outputs / "logs" / "a.err")])


def test_an_unread_file_is_never_read_and_its_path_still_searched(tmp_path, monkeypatch):
    """A file of ``--unread``, or below a directory it names, is not read even when ``--list`` or
    ``--raw`` holds it: a value in its bytes is not found and no read line names it. Its path is
    searched like any other's, and an unread path that is not there fails the scan."""
    outputs = tmp_path / "outputs"
    (outputs / "snapshots").mkdir(parents=True)
    (outputs / "source.bundle").write_bytes(VALUE)
    (outputs / "snapshots" / "main-1.bundle").write_bytes(VALUE)
    (outputs / "snapshots" / f"{VALUE.decode()}.bundle").write_bytes(b"plain")
    (outputs / "b.json").write_bytes(VALUE)
    unread = tmp_path / "unread.txt"
    unread.write_text(f"{outputs / 'source.bundle'}\n{outputs / 'snapshots'}\n")
    monkeypatch.setenv("relay", VALUE.decode())
    lst, stored = tmp_path / "list.txt", tmp_path / "raw.txt"
    lst.write_text(f"{outputs}\n{outputs / 'source.bundle'}\n")
    stored.write_text(f"{outputs / 'source.bundle'}\n")
    argv = ["--names", "relay", "--list", str(lst), "--raw", str(stored), "--unread", str(unread),
            "--report", str(tmp_path / "found.jsonl")]
    assert secret_scan.main(argv) == 0
    lines = (tmp_path / "found.jsonl").read_text().splitlines()
    hits = sorted(json.loads(line[len(FOUND_PREFIX):])["file"] for line in lines if line.startswith(FOUND_PREFIX))
    read = sorted(json.loads(line[len(READ_PREFIX):])["file"] for line in lines if line.startswith(READ_PREFIX))
    assert read == [str(outputs / "b.json")]
    assert hits == sorted([str(outputs / "b.json"), str(outputs / "snapshots" / f"{VALUE.decode()}.bundle")])
    unread.write_text(f"{outputs / 'gone.bundle'}\n")
    assert secret_scan.main(argv) == 1


def test_an_unreadable_directory_under_a_listed_one_fails_the_scan(tmp_path, monkeypatch, capsys):
    outputs = tmp_path / "outputs"
    (outputs / "hidden").mkdir(parents=True)
    (outputs / "hidden" / "leak.txt").write_bytes(b"nothing")
    os.chmod(outputs / "hidden", 0)
    try:
        code = _main(tmp_path, monkeypatch, [outputs], [])
    finally:
        os.chmod(outputs / "hidden", 0o700)
    assert code == 1 and "secret scan failed" in capsys.readouterr().err


def test_an_unreadable_dot_directory_under_a_listed_one_fails_the_scan(tmp_path, monkeypatch, capsys):
    """A dot directory is walked like any other, for its files and for its paths, and one that
    cannot be listed holds files and paths the scan cannot check."""
    outputs = tmp_path / "outputs"
    (outputs / ".cache").mkdir(parents=True)
    (outputs / ".cache" / "leak.txt").write_bytes(b"nothing")
    (outputs / "clean.txt").write_bytes(b"clean")
    assert _main(tmp_path, monkeypatch, [outputs], []) == 0
    os.chmod(outputs / ".cache", 0)
    try:
        code = _main(tmp_path, monkeypatch, [outputs], [])
    finally:
        os.chmod(outputs / ".cache", 0o700)
    assert code == 1 and "secret scan failed" in capsys.readouterr().err


def test_a_file_that_goes_while_the_walk_runs_is_not_a_failure_but_a_named_one_is(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a").write_bytes(b"a")
    (outputs / "b").write_bytes(b"b")
    real = secret_scan.scan_stream

    def vanishing(path, values, **kwargs):
        if path.name == "a":
            path.unlink()
        return real(path, values, **kwargs)

    monkeypatch.setattr(secret_scan, "scan_stream", vanishing)
    assert _main(tmp_path, monkeypatch, [outputs], []) == 0
    (outputs / "a").write_bytes(b"a")
    assert _main(tmp_path, monkeypatch, [outputs / "a"], []) == 1


def test_the_findings_go_to_the_report_and_stdout_only_counts_them(tmp_path, monkeypatch, capsys):
    (tmp_path / "hit").write_bytes(VALUE)
    code, report, logged = _run(tmp_path, monkeypatch, capsys, [tmp_path / "hit"])
    assert code == 0 and report == FOUND_PREFIX + json.dumps({"file": str(tmp_path / "hit"),
                                                                      "secret": "relay"}) + "\n"
    assert FOUND_PREFIX not in logged and "1 value found in 1 file" in logged


def test_a_value_in_a_dot_named_file_under_a_transcript_copy_is_found(tmp_path, monkeypatch, capsys):
    """An agent's dot-files are bytes it chose: the scan reads them, and walks its dot-directories,
    as it reads every other file of the transcript's copy, which the transcript role's own scrubbing
    does not cover alone."""
    transcript = tmp_path / "outputs" / "transcript"
    (transcript / ".codex" / "sessions").mkdir(parents=True)
    (transcript / ".claude.json").write_bytes(b'{"apiKey": "' + VALUE + b'"}')
    (transcript / ".codex" / "sessions" / "rollout.jsonl").write_bytes(OTHER)
    code, report, _ = _run(tmp_path, monkeypatch, capsys, [], raw=[transcript])
    assert code == 0
    hits = [json.loads(line[len(FOUND_PREFIX):]) for line in report.splitlines()]
    assert hits == [{"file": str(transcript / ".claude.json"), "secret": "relay"},
                    {"file": str(transcript / ".codex" / "sessions" / "rollout.jsonl"), "secret": "hub"}]


def test_a_value_in_a_dot_named_file_or_directory_is_found(tmp_path, monkeypatch, capsys):
    """A killed driver leaves its trace's chunk under its writer's dot-named temp name, and a
    server's workdir holds its own dot-files and dot-directories: each is bytes like any other."""
    trace = tmp_path / "outputs" / "trace"
    (trace / ".cache" / "sub").mkdir(parents=True)
    (trace / ".chunk-000000.jsonl.gz.tmp").write_bytes(gzip.compress(b'{"reply": "' + VALUE + b'"}\n'))
    (trace / ".cache" / "sub" / "kept.json").write_bytes(OTHER)
    code, report, _ = _run(tmp_path, monkeypatch, capsys, [tmp_path / "outputs"])
    assert code == 0
    hits = [json.loads(line[len(FOUND_PREFIX):]) for line in report.splitlines()]
    assert hits == [{"file": str(trace / ".cache" / "sub" / "kept.json"), "secret": "hub"},
                    {"file": str(trace / ".chunk-000000.jsonl.gz.tmp"), "secret": "relay"}]


def test_a_link_under_a_listed_directory_is_not_followed_whatever_its_name(tmp_path, monkeypatch, capsys):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (tmp_path / "host").mkdir()
    (tmp_path / "host" / "secret.txt").write_bytes(VALUE)
    (outputs / ".dir").symlink_to(tmp_path / "host")
    (outputs / ".file").symlink_to(tmp_path / "host" / "secret.txt")
    assert _main(tmp_path, monkeypatch, [outputs], []) == 0


def test_every_link_and_special_file_is_reported_as_not_read_and_never_followed_or_opened(tmp_path):
    """A link is never followed and a pipe never opened, under a listed directory or listed itself,
    and none is passed over in silence: the report names each as not read, with its kind. The role
    runs in a process of its own under a time limit, so one that opened the pipe fails here instead
    of hanging."""
    outputs = tmp_path / "outputs"
    (outputs / "sub").mkdir(parents=True)
    (tmp_path / "host.txt").write_bytes(VALUE)
    (outputs / "kept.txt").write_bytes(b"clean")
    (outputs / "sub" / "link").symlink_to(tmp_path / "host.txt")
    os.mkfifo(outputs / "pipe")
    (tmp_path / "listed-link").symlink_to(tmp_path / "host.txt")
    listing, report = tmp_path / "list.txt", tmp_path / "found.jsonl"
    listing.write_text(f"{outputs}\n{tmp_path / 'listed-link'}\n")
    src = Path(secret_scan.__file__).resolve().parents[2]
    done = subprocess.run([sys.executable, "-m", "hls.substrate.secret_scan", "--names", "relay", "--list",
                           str(listing), "--report", str(report)], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "PYTHONPATH": str(src), "relay": VALUE.decode()})
    assert done.returncode == 0, done.stderr
    lines = report.read_text().splitlines()
    unread = [json.loads(line[len(UNREAD_PREFIX):]) for line in lines if line.startswith(UNREAD_PREFIX)]
    assert unread == [{"file": str(outputs / "pipe"), "kind": "special"},
                      {"file": str(outputs / "sub" / "link"), "kind": "link"},
                      {"file": str(tmp_path / "listed-link"), "kind": "link"}]
    read = [json.loads(line[len(READ_PREFIX):])["file"] for line in lines if line.startswith(READ_PREFIX)]
    assert read == [str(outputs / "kept.txt")] and not any(line.startswith(FOUND_PREFIX) for line in lines)
    assert "3 links or special files not read" in done.stdout


def test_a_temporary_file_removed_between_the_walk_and_its_read_is_skipped(tmp_path, monkeypatch):
    """A store's temporary file needs no rule of its own: one its writer renamed while the scan ran
    is gone, and skipped as any file the walk found that is gone by the time it is read; one a dead
    writer left holds what that writer was writing, and is read like any file."""
    stage = tmp_path / "stages" / "finalize"
    stage.mkdir(parents=True)
    renamed = stage / f".tmp-MANIFEST.json-{'0' * 32}"
    renamed.write_bytes(b"{}")
    left = stage / f".tmp-events.jsonl-{'1' * 32}"
    left.write_bytes(VALUE)
    real = secret_scan.scan_stream

    def renaming(path, values, **kwargs):
        if path == renamed:
            path.rename(stage / "MANIFEST.json")
        return real(path, values, **kwargs)

    monkeypatch.setattr(secret_scan, "scan_stream", renaming)
    assert _main(tmp_path, monkeypatch, [tmp_path / "stages"], []) == 0
    lines = _found_and_cut(tmp_path / "found.jsonl").splitlines()
    assert [json.loads(line[len(FOUND_PREFIX):])["file"] for line in lines] == [str(left)]


def test_a_listed_dot_named_path_that_is_gone_fails_the_scan(tmp_path, monkeypatch, capsys):
    """A path a list names must be there, whatever its name: a dot-named one is not passed
    over unread, so one that is gone fails the scan as any listed path that is gone does."""
    gone = tmp_path / f".tmp-plan.json-{'2' * 32}"
    code, _, logged = _run(tmp_path, monkeypatch, capsys, [gone])
    assert code == 1 and "secret scan failed" in logged
