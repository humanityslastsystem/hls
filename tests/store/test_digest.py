from __future__ import annotations

import hashlib
import io

import pytest
from pydantic import TypeAdapter, ValidationError

from hls.store import digest


def test_sha256_bytes():
    assert digest.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_sha256_file_streams_across_chunk_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "CHUNK", 4)
    data = b"0123456789"  # 10 bytes = 3 chunks of 4
    p = tmp_path / "f"
    p.write_bytes(data)
    assert digest.sha256_file(p) == (hashlib.sha256(data).hexdigest(), 10)


def test_sha256_stream_streams_across_chunk_boundary(monkeypatch):
    monkeypatch.setattr(digest, "CHUNK", 4)
    data = b"0123456789"
    assert digest.sha256_stream(io.BytesIO(data)) == (hashlib.sha256(data).hexdigest(), 10)


HEX = "0123456789abcdef" * 4


def test_sha256_hex_accepts_bare_lowercase_hex():
    assert TypeAdapter(digest.Sha256Hex).validate_python(HEX) == HEX
    assert digest.check_sha256_hex(HEX) == HEX


@pytest.mark.parametrize(
    "bad",
    ["sha256:" + HEX, HEX + "\n", "\n" + HEX, HEX.upper(), HEX[:-1], HEX + "0", "", " " + HEX],
    ids=["prefixed", "trailing_newline", "leading_newline", "upper", "short", "long", "empty", "space"],
)
def test_sha256_hex_rejects(bad):
    with pytest.raises(ValidationError):
        TypeAdapter(digest.Sha256Hex).validate_python(bad)
    with pytest.raises(ValueError):
        digest.check_sha256_hex(bad)


def test_helpers_return_sha256_hex():
    assert digest.check_sha256_hex(digest.sha256_bytes(b"abc"))
    assert digest.check_sha256_hex(digest.sha256_stream(io.BytesIO(b"abc"))[0])
