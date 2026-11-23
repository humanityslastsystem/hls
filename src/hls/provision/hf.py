"""Checking a downloaded hub snapshot against the hub's own file list.

The hub names every file of a revision with its size and a hash: the SHA-256 of the
content for a file stored in LFS (weights shards), or the git blob id (SHA-1 of
``"blob <size>\\0" + content``) for a small file kept in git (configs, tokenizers). A fill
is published only when the files the cache scanned are exactly the hub's, with matching
sizes and hashes, so a truncated or partly replaced download never enters the cache.
The weights role reports both kinds of hash from the bytes it wrote: the SHA-256 of
every file, which the cache records, and the blob id of every file the hub keeps in git
(``git_blob_sha1``). The check reads no file, so the stage that runs it reads none of the
download."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from hls.store.cache import FileRecord
from hls.store.digest import CHUNK, Sha256Hex


class SourceFile(BaseModel):
    """One file of a hub revision as the hub lists it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Size in bytes.
    size: int
    #: SHA-256 of the content, for a file stored in LFS; None for a git file.
    sha256: Sha256Hex | None
    #: Git blob id, for a file kept in git; None for an LFS file.
    git_sha1: str | None

    @model_validator(mode="after")
    def _one_hash(self) -> SourceFile:
        if (self.sha256 is None) == (self.git_sha1 is None):
            raise ValueError("a hub file has either an LFS sha256 or a git blob id")
        return self


def source_files(repo: str, revision: str, *, token: str | bool) -> dict[str, SourceFile]:
    """The hub's file list for *repo* at commit *revision*, by path. Raise ValueError when
    the hub reports another commit or a file without a size."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=token)
    if info.sha != revision:
        raise ValueError(f"the hub answered for {repo}@{info.sha}, not {revision}")
    out = {}
    for s in info.siblings or []:
        if s.lfs is not None:
            out[s.rfilename] = SourceFile(size=s.lfs.size, sha256=s.lfs.sha256, git_sha1=None)
        elif s.size is None or s.blob_id is None:
            raise ValueError(f"the hub lists {repo}/{s.rfilename} without a size or blob id")
        else:
            out[s.rfilename] = SourceFile(size=s.size, sha256=None, git_sha1=s.blob_id)
    return out


def git_blob_sha1(path: Path) -> str:
    """The git blob id of the file at *path*."""
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check_against_source(files: list[FileRecord], blobs: Mapping[str, str],
                         source: Mapping[str, SourceFile]) -> list[str]:
    """Every way a filled entry, whose files the cache scanned as *files* with the SHA-256 its
    role reported, differs from the hub's list *source*: a file missing or unexpected, a size
    that differs, or a hash that differs, a git file's hash being its blob id the role reported
    in *blobs*, by path. [] when they agree."""
    have = {f.path: f for f in files}
    problems = [f"missing {p}" for p in sorted(source.keys() - have.keys())]
    problems += [f"unexpected {p}" for p in sorted(have.keys() - source.keys())]
    for path in sorted(source.keys() & have.keys()):
        want, got = source[path], have[path]
        if want.size != got.size:
            problems.append(f"{path}: {got.size} bytes, the hub says {want.size}")
        elif want.sha256 is not None:
            if want.sha256 != got.sha256:
                problems.append(f"{path}: sha256 {got.sha256}, the hub says {want.sha256}")
        elif path not in blobs:
            problems.append(f"{path}: the role reported no git blob id")
        elif blobs[path] != want.git_sha1:
            problems.append(f"{path}: git blob id differs from the hub's {want.git_sha1}")
    return problems
