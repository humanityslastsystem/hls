"""Finalize's scan role searches every form of each secret (``hls.substrate.secret_forms``): a URL
secret's host alone, with no value beside it, is found in any case, in a file's bytes and in its
path, and reported under the label ``<name>.host``; a value is reported under its secret's name."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import zlib
from pathlib import Path

import pytest

from hls.substrate import secret_scan
from hls.substrate.secret_forms import registry_forms
from hls.substrate.secret_scan import FOUND_PREFIX, READ_PREFIX, find, named_in, scan_stream

URL = "https://Relay.Example:8443/v1/anthropic"
HOST = "relay.example"
TOKEN = "hf-token-abcdefgh"
#: A URL that is no http(s) URL: its host is no form of the secret.
FTP = "ftp://files.example/pub"
REGISTRY = {"relay_url": URL, "hub": TOKEN, "mirror": FTP}
#: The host as a client may print it.
CASES = [HOST, HOST.upper(), "Relay.Example", "rElAy.ExAmPlE"]


def _scan(tmp_path: Path, monkeypatch, listed: list[Path], raw: list[Path] = ()) -> tuple[int, list[dict]]:
    """Run the role over *listed* and *raw* with ``REGISTRY`` in its environment; its exit status and
    the findings it reported."""
    for name, value in REGISTRY.items():
        monkeypatch.setenv(name, value)
    lst, stored, report = tmp_path / "list.txt", tmp_path / "raw.txt", tmp_path / "found.jsonl"
    lst.write_text("".join(f"{p}\n" for p in listed))
    stored.write_text("".join(f"{p}\n" for p in raw))
    code = secret_scan.main(["--names", ",".join(REGISTRY), "--list", str(lst), "--raw", str(stored),
                             "--report", str(report)])
    lines = report.read_text().splitlines() if report.exists() else []
    return code, [json.loads(line[len(FOUND_PREFIX):]) for line in lines if line.startswith(FOUND_PREFIX)]


@pytest.mark.parametrize("printed", CASES)
def test_a_host_alone_in_a_plain_file_is_found_under_its_label(tmp_path, monkeypatch, printed):
    out = tmp_path / "out"
    out.mkdir()
    (out / "server.log").write_text(f"connected to {printed}:8443\n")
    code, found = _scan(tmp_path, monkeypatch, [out])
    assert code == 0
    assert found == [{"file": str(out / "server.log"), "secret": "relay_url.host"}]


@pytest.mark.parametrize("printed", CASES)
def test_a_host_alone_in_a_gzip_file_of_the_list_is_found_in_its_plain_bytes(tmp_path, monkeypatch, printed):
    out = tmp_path / "out"
    out.mkdir()
    (out / "replies.jsonl.gz").write_bytes(gzip.compress(f'{{"dialled": "{printed}"}}\n'.encode(), mtime=0))
    assert printed.encode().lower() not in (out / "replies.jsonl.gz").read_bytes().lower()
    assert _scan(tmp_path, monkeypatch, [out]) == (
        0, [{"file": str(out / "replies.jsonl.gz"), "secret": "relay_url.host"}])


@pytest.mark.parametrize("printed", CASES)
def test_a_host_alone_in_a_raw_file_is_found_as_stored(tmp_path, monkeypatch, printed):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "role.err").write_bytes(b"\x1f\x8b" + f"retrying {printed}".encode())
    (logs / "packed.out").write_bytes(gzip.compress(printed.encode(), mtime=0))
    assert _scan(tmp_path, monkeypatch, [], raw=[logs]) == (
        0, [{"file": str(logs / "role.err"), "secret": "relay_url.host"}])


@pytest.mark.parametrize("printed", CASES)
def test_a_host_in_a_files_name_and_in_a_directorys_name_is_found(tmp_path, monkeypatch, printed):
    out = tmp_path / "out"
    (out / f"cache-{printed}" / "inner").mkdir(parents=True)
    (out / f"cache-{printed}" / "inner" / "clean.txt").write_text("clean\n")
    (out / f"{printed}.json").write_text("{}\n")
    code, found = _scan(tmp_path, monkeypatch, [out])
    assert code == 0
    named = [out / f"cache-{printed}", out / f"cache-{printed}" / "inner",
             out / f"cache-{printed}" / "inner" / "clean.txt", out / f"{printed}.json"]
    assert sorted(found, key=lambda f: f["file"]) == sorted(
        ({"file": str(path), "secret": "relay_url.host"} for path in named), key=lambda f: f["file"])


def test_a_value_is_found_under_its_secrets_name_and_a_url_holds_its_host(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "token.txt").write_text(f"token {TOKEN}\n")
    (out / "url.txt").write_text(f"base {URL}\n")
    code, found = _scan(tmp_path, monkeypatch, [out])
    assert code == 0
    assert found == [{"file": str(out / "token.txt"), "secret": "hub"},
                     {"file": str(out / "url.txt"), "secret": "relay_url"},
                     {"file": str(out / "url.txt"), "secret": "relay_url.host"}]


def test_a_secret_that_is_no_http_url_has_no_host_to_find(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "files.example").mkdir(parents=True)
    (out / "files.example" / "notes.txt").write_text("mirrored from FILES.EXAMPLE and hf-token\n")
    assert _scan(tmp_path, monkeypatch, [out]) == (0, [])


@pytest.mark.parametrize("printed", CASES)
def test_a_host_is_found_in_any_case_wherever_the_reads_split_it(printed):
    forms = registry_forms(REGISTRY)
    for chunk in range(1, len(HOST) + 3):
        for offset in (0, 3, 7):
            data = b"x" * offset + printed.encode() + b"y" * 5
            assert find(io.BytesIO(data), forms, chunk=chunk) == {"relay_url.host"}, (chunk, offset)


def test_the_reads_overlap_by_the_longest_form(tmp_path):
    """A host shorter than every value is still found where a read splits it: the overlap is the
    longest form's length."""
    data = b"." * 30 + HOST.upper().encode() + b"." * 3 + URL.encode()
    assert find(io.BytesIO(data), registry_forms(REGISTRY), chunk=len(URL)) == {"relay_url", "relay_url.host"}
    path = tmp_path / "x.gz"
    path.write_bytes(gzip.compress(b"a" * 70_000 + HOST.upper().encode()))
    scanned = scan_stream(path, registry_forms(REGISTRY))
    assert (scanned.labels, scanned.cut) == ({"relay_url.host"}, False)


def test_a_host_in_a_path_is_named_in_any_case():
    forms = registry_forms(REGISTRY)
    assert named_in(Path("out/RELAY.EXAMPLE/x"), forms) == {"relay_url.host"}
    assert named_in(Path(f"keys/{TOKEN}.txt"), forms) == {"hub"}
    assert named_in(Path("out/files.example/x"), forms) == set()


def _scan_reading(tmp_path: Path, monkeypatch, listed: list[Path], raw: list[Path] = ()) -> dict[str, str]:
    """Run the role over *listed* and *raw* and return each file it reported reading, by path, with
    the sha256 it reported."""
    assert _scan(tmp_path, monkeypatch, listed, raw)[0] == 0
    lines = (tmp_path / "found.jsonl").read_text().splitlines()
    read = [json.loads(line[len(READ_PREFIX):]) for line in lines if line.startswith(READ_PREFIX)]
    return {entry["file"]: entry["sha256"] for entry in read}


def test_the_role_reports_each_file_it_read_by_the_sha256_of_its_bytes_as_stored(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    files = {out / "plain.json": b'{"a": 1}\n', out / "trace.jsonl.gz": gzip.compress(b"x" * 1000, mtime=0),
             out / "logs" / "role.err": b"stderr\n", out / ".env": b"EDITOR=vi\n"}
    for path, data in files.items():
        path.write_bytes(data)
    read = _scan_reading(tmp_path, monkeypatch, [out, out / "plain.json"], [out / "logs"])
    assert read == {str(path): hashlib.sha256(data).hexdigest() for path, data in files.items()}
    # A file the walk found that is gone by the time it is read was not read.
    real = secret_scan.scan_stream

    def vanishing(path, forms, **kwargs):
        if path.name == "plain.json":
            path.unlink()
        return real(path, forms, **kwargs)

    monkeypatch.setattr(secret_scan, "scan_stream", vanishing)
    read = _scan_reading(tmp_path, monkeypatch, [out], [out / "logs"])
    assert str(out / "plain.json") not in read and str(out / ".env") in read


def test_the_bytes_of_a_whole_gzip_stream_are_searched_as_stored_too(tmp_path):
    path = tmp_path / "x.gz"
    path.write_bytes(gzip.compress(b"a" * 1000, mtime=0) + TOKEN.encode())
    scanned = scan_stream(path, registry_forms(REGISTRY))
    assert (scanned.labels, scanned.cut) == ({"hub"}, True)
    path.write_bytes(_member(b"plain", comment=TOKEN.encode()))
    assert scan_stream(path, registry_forms(REGISTRY)).labels == {"hub"}


def _member(data: bytes, *, name: bytes | None = None, comment: bytes | None = None,
            extra: bytes | None = None) -> bytes:
    """One gzip member of *data* with the optional header fields given (RFC 1952)."""
    flags = (4 if extra is not None else 0) | (8 if name is not None else 0) | (16 if comment is not None else 0)
    header = bytes([0x1F, 0x8B, 8, flags, 0, 0, 0, 0, 0, 255])
    if extra is not None:
        header += len(extra).to_bytes(2, "little") + extra
    header += (name + b"\0" if name is not None else b"") + (comment + b"\0" if comment is not None else b"")
    deflate = zlib.compressobj(9, zlib.DEFLATED, -15)
    body = deflate.compress(data) + deflate.flush()
    return header + body + zlib.crc32(data).to_bytes(4, "little") + len(data).to_bytes(4, "little")


@pytest.mark.parametrize("field", ["name", "comment", "extra"])
def test_a_value_in_a_gzip_header_is_found(tmp_path, monkeypatch, field):
    out = tmp_path / "out"
    out.mkdir()
    gz = out / "notes.json.gz"
    gz.write_bytes(_member(b"nothing to see\n", **{field: TOKEN.encode()}))
    assert gzip.decompress(gz.read_bytes()) == b"nothing to see\n"
    assert _scan(tmp_path, monkeypatch, [out]) == (0, [{"file": str(gz), "secret": "hub"}])


def test_a_listed_paths_own_name_is_searched_and_what_lies_above_it_is_not(tmp_path, monkeypatch):
    above = tmp_path / f"site-{TOKEN}"
    plain = above / "outputs"
    named = above / f"out-{TOKEN}"
    for folder in (plain, named):
        folder.mkdir(parents=True)
        (folder / "a.txt").write_text("plain\n")
    assert _scan(tmp_path, monkeypatch, [plain]) == (0, [])
    _, found = _scan(tmp_path, monkeypatch, [named])
    assert found == [{"file": str(named), "secret": "hub"}, {"file": str(named / "a.txt"), "secret": "hub"}]


def test_a_numeric_hosts_address_is_found_as_a_client_prints_it(tmp_path):
    path = tmp_path / "driver.err"
    path.write_bytes(b"connect to 8.8.4.4:443 refused\n")
    assert scan_stream(path, registry_forms({"relay_url": "https://0x8.8.4.4/v1"})).labels == {"relay_url.host"}
