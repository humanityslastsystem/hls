"""L0 -- Image (de)serialization for the Envelope wire.

An image rides inside the JSON Envelope as a base64 blob of its *encoded* file bytes
(PNG/JPEG) plus a MIME type -- exactly like audio rides as a base64 WAV. This module is
the one place that converts between encoded image bytes and an ``ImagePayload``, so the
dataset (encode), the artifact server, and the omni chat protocol (render to a data
URI) all agree on the byte layout. Dimensions are sniffed from the PNG/JPEG header with
no third-party dependency, so importing the spec layer stays torch/PIL-free on the host;
PIL is imported lazily only by ``pil_to_payload`` (used inside GPU containers).
"""

from __future__ import annotations
import base64
import struct
from typing import TYPE_CHECKING

from hls.core.payloads import ImagePayload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _PILImage


_MIME_BY_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",        # RIFF....WEBP
}


def sniff_mime(raw: bytes) -> str:
    """Best-effort MIME from the magic bytes; defaults to image/png."""
    for magic, mime in _MIME_BY_MAGIC.items():
        if raw.startswith(magic):
            if mime == "image/webp" and raw[8:12] != b"WEBP":
                continue
            return mime
    return "image/png"


def sniff_dimensions(raw: bytes) -> tuple[int, int] | None:
    """(width, height) parsed from a PNG or JPEG header without PIL, or None."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        w, h = struct.unpack(">II", raw[16:24])           # IHDR width/height
        return int(w), int(h)
    if raw.startswith(b"\xff\xd8"):                         # JPEG: scan for SOF marker
        i = 2
        n = len(raw)
        while i + 9 < n:
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", raw[i + 5:i + 9])
                return int(w), int(h)
            seg = struct.unpack(">H", raw[i + 2:i + 4])[0]
            i += 2 + seg
    return None


def image_payload_from_bytes(raw: bytes, mime: str | None = None) -> ImagePayload:
    """Wrap encoded image bytes (PNG/JPEG/...) into an ImagePayload, sniffing the MIME
    type and dimensions from the header when not given."""
    dims = sniff_dimensions(raw)
    return ImagePayload(
        image_b64=base64.b64encode(raw).decode("ascii"),
        mime_type=mime or sniff_mime(raw),
        width=dims[0] if dims else None,
        height=dims[1] if dims else None)


def decode_image_bytes(b64: str) -> bytes:
    """Inverse of the b64 carried in ImagePayload -- the raw encoded image bytes."""
    return base64.b64decode(b64)


def pil_to_payload(img: "_PILImage", fmt: str = "PNG") -> ImagePayload:
    """Encode a PIL image to PNG/JPEG bytes and wrap as an ImagePayload. PIL is
    imported lazily so the host stays PIL-free; only GPU containers call this."""
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return image_payload_from_bytes(buf.getvalue(), f"image/{fmt.lower()}")
