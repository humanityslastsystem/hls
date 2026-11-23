"""The program each no-op role runs, in the minimal pipeline the test suite and a site's live
tests run. It writes what its step looks like from inside, optionally holds memory, and sleeps.

``python -m hls.substrate.noop_role --out <file> [--outputs-out <report>] --sleep-sec N [--hog-gib G]``

The report names the GPUs the CUDA driver exposes to the process, by UUID, which is what
the no-op stage compares across its roles to check that GPUs are isolated and that the
shared role sees the GPUs it shares. It is written, and for a role, whose file is a stage output,
its sha256 reported in *report* for the stage's manifest (``report_outputs``), before any memory
is held, so it survives the role being killed for memory. The process that holds memory in the
batch step writes its file to the logs and reports nothing."""

from __future__ import annotations

import argparse
import ctypes
import os
import socket
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from hls.store.manifest import report_outputs

_CHUNK = 256 * 2**20
_PAGE = 4096


class NoopRoleReport(BaseModel):
    """What a no-op role saw."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Node.
    host: str
    #: GPUs the CUDA driver exposes, by UUID; None when there is no view of them (see ``cuda_uuids``).
    cuda_uuids: list[str] | None
    #: CUDA_VISIBLE_DEVICES as the process saw it.
    cuda_visible_devices: str | None


def cuda_uuids() -> list[str] | None:
    """UUIDs of the devices the CUDA driver exposes; [] when it exposes none; None when there
    is no view of them: ``libcuda.so.1`` cannot be loaded, or a driver call fails."""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    if cu.cuInit(0) != 0:
        return None
    count = ctypes.c_int()
    if cu.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return None
    out = []
    for i in range(count.value):
        dev = ctypes.c_int()
        raw = (ctypes.c_ubyte * 16)()
        if cu.cuDeviceGet(ctypes.byref(dev), i) != 0 or cu.cuDeviceGetUuid(raw, dev) != 0:
            return None
        h = bytes(raw).hex()
        out.append(f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}")
    return out


def hold_memory(gib: int, *, step_sec: float = 0.1) -> list[bytearray]:
    """Allocate and touch *gib* GiB, a chunk at a time, and return the chunks."""
    chunks = []
    for _ in range(gib * (2**30 // _CHUNK)):
        chunk = bytearray(_CHUNK)
        chunk[::_PAGE] = b"\x01" * (_CHUNK // _PAGE)
        chunks.append(chunk)
        time.sleep(step_sec)
    return chunks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hls.substrate.noop_role")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outputs-out", type=Path)
    p.add_argument("--sleep-sec", type=float, default=0.0)
    p.add_argument("--hog-gib", type=int, default=0)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    report = NoopRoleReport(host=socket.gethostname(), cuda_uuids=cuda_uuids(),
                            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    tmp = args.out.with_name(f".{args.out.name}.tmp")
    tmp.write_text(report.model_dump_json(indent=2))
    os.replace(tmp, args.out)
    if args.outputs_out is not None:
        report_outputs(args.outputs_out, args.out.parent, args.out)
    held = hold_memory(args.hog_gib)
    time.sleep(args.sleep_sec)
    del held
    return 0


if __name__ == "__main__":
    sys.exit(main())
