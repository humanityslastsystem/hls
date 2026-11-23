"""Image archives written from trees, as Docker's image stores write them, for the package's
checks of an image read from its archive (``hls.package.layers``). A tree is in
``fake_tools.lay_out``'s form, path to ``["dir", mode]``, ``["file", text, mode]`` or
``["link", target]``; a layer's tree may also name a path ``None``, a whiteout of what the
layers below hold there, or ``"opaque"``, a marker that empties the directory there of what
the layers below hold in it."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

#: The media types of a plain and of a gzip-compressed layer.
PLAIN, GZIP = "application/vnd.oci.image.layer.v1.tar", "application/vnd.oci.image.layer.v1.tar+gzip"


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def layer(tree: dict[str, list | str | None], *, extra: tuple[tarfile.TarInfo, ...] = ()) -> bytes:
    """The uncompressed tar of a layer holding *tree*, its members in path order, then *extra*."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(tree):
            entry = tree[path]
            parent, _, name = path.rpartition("/")
            prefix = f"{parent}/" if parent else ""
            if entry is None:
                tar.addfile(tarfile.TarInfo(f"{prefix}.wh.{name}"), io.BytesIO(b""))
                continue
            if entry == "opaque":
                tar.addfile(tarfile.TarInfo(f"{path}/.wh..wh..opq"), io.BytesIO(b""))
                continue
            info = tarfile.TarInfo(path)
            data = b""
            if entry[0] == "dir":
                info.type, info.mode = tarfile.DIRTYPE, entry[1]
            elif entry[0] == "file":
                data = entry[1].encode()
                info.size, info.mode = len(data), entry[2]
            else:
                info.type, info.linkname, info.mode = tarfile.SYMTYPE, entry[1], 0o777
            tar.addfile(info, io.BytesIO(data))
        for info in extra:
            tar.addfile(info, io.BytesIO(b"\0" * info.size) if info.isreg() else None)
    return out.getvalue()


def diff(base: dict[str, list], image: dict[str, list]) -> dict[str, list | str | None]:
    """The layer that makes *base* into *image*: every entry *image* holds otherwise than *base*,
    and a whiteout for each entry *base* holds that *image* does not, its highest only."""
    changed: dict[str, list | str | None] = {p: e for p, e in image.items() if base.get(p) != e}
    gone = [p for p in base if p not in image]
    for path in gone:
        if path.rpartition("/")[0] not in gone:
            changed[path] = None
    return changed


def write_image(path: Path, layers: list[bytes], *, media_type: str = PLAIN,
                diff_ids: list[str] | None = None) -> str:
    """Write an archive of one image of *layers* (uncompressed tars), stored as *media_type*, to
    *path*; return its manifest's digest. *diff_ids* replaces what the config names."""
    blobs = [gzip.compress(data, mtime=0) if media_type == GZIP else data for data in layers]
    config = json.dumps({"architecture": "amd64", "os": "linux", "config": {"Env": ["A=1"]},
                         "rootfs": {"type": "layers",
                                    "diff_ids": diff_ids or [_digest(data) for data in layers]}}).encode()
    manifest = json.dumps({
        "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": _digest(config),
                   "size": len(config)},
        "layers": [{"mediaType": media_type, "digest": _digest(blob), "size": len(blob)} for blob in blobs]}).encode()
    index = {"schemaVersion": 2, "manifests": [{"mediaType": "application/vnd.oci.image.manifest.v1+json",
                                                "digest": _digest(manifest), "size": len(manifest)}]}
    members = {"index.json": json.dumps(index).encode(), "oci-layout": b'{"imageLayoutVersion":"1.0.0"}'}
    for blob in (*blobs, config, manifest):
        members[f"blobs/sha256/{_digest(blob)[7:]}"] = blob
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return _digest(manifest)
