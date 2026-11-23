"""Image archives written by hand, with fixed bytes, for the image role's tests: an OCI layout and
a Docker ``manifest.json`` in one tar, as BuildKit's docker exporter writes them."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Archive:
    path: Path
    manifest_digest: str
    config_digest: str
    layer_digest: str


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), 0o444, 0
    tar.addfile(info, io.BytesIO(data))


def write_archive(path: Path, *, edit: Callable[[dict[str, bytes], dict], None] | None = None,
                  extra: tuple[tuple[str, bytes], ...] = ()) -> Archive:
    """An archive of one image of one empty layer. *edit* may change the members (name to bytes)
    and the index before they are written; *extra* adds members after them."""
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w"):
        pass
    layer_bytes = layer.getvalue()
    config = json.dumps({"architecture": "amd64", "os": "linux", "config": {"Labels": {"hls.test": "unit"}},
                         "rootfs": {"type": "layers", "diff_ids": [_digest(layer_bytes)]}}).encode()
    manifest = json.dumps({
        "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": _digest(config),
                   "size": len(config)},
        "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar", "digest": _digest(layer_bytes),
                    "size": len(layer_bytes)}]}).encode()
    index = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json",
             "manifests": [{"mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": _digest(manifest),
                            "size": len(manifest), "annotations": {"org.opencontainers.image.created": "0"}}]}
    members = {"oci-layout": json.dumps({"imageLayoutVersion": "1.0.0"}).encode(),
               "manifest.json": json.dumps([{"Config": f"blobs/sha256/{_digest(config)[7:]}", "RepoTags": None,
                                             "Layers": [f"blobs/sha256/{_digest(layer_bytes)[7:]}"]}]).encode()}
    for blob in (layer_bytes, config, manifest):
        members[f"blobs/sha256/{_digest(blob)[7:]}"] = blob
    if edit is not None:
        edit(members, index)
    with tarfile.open(path, "w") as tar:
        _add(tar, "index.json", json.dumps(index).encode())
        for name, data in members.items():
            _add(tar, name, data)
        for name, data in extra:
            _add(tar, name, data)
    return Archive(path=path, manifest_digest=_digest(manifest), config_digest=_digest(config),
                   layer_digest=_digest(layer_bytes))
