"""Engines that stand in, in a calibration role's own process, for the HF reference, its fp32
anchor and the vLLM and SGLang pool members, so a role runs its real program with no model, no
GPU and no serving stack. ``install()`` makes ``hls.calibrate`` build them.

The member serves every session it is handed with one reply of ``TOKENS_PER_REPLY`` ids, never
more than the tenant's output cap; the reference scores ``GAP`` on each reply's first ``FLIPS``
positions and nothing at the preflight's probes, which its anchor answers with three tokens.
Every engine reports a B200, the fixture task's accelerator.

Each engine's load does what a JIT does with the engine cache its role mounts
(``FLASHINFER_WORKSPACE_BASE``, ``hls.stages.engine_cache.engine_cache_env``): it builds its
kernel there unless the cache holds it already, beside the object file it was linked from (which
a real build's tool reads to decide nothing is to build again), leaving the build's temporary
file in ``TMPDIR`` and noting each build in ``compiles.log``, and notes every load in ``loads.log``. A load whose engine
cache holds ``HANG`` never ends, as a build that deadlocks. A member's load whose engine cache holds
``RELEASE_HANG`` leaves an object behind whose finalizer never returns, so freeing what the load
held never ends. Outside a container with an engine cache it does nothing."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, ClassVar

import hls.calibrate as calibrate
from hls.core.decode import Greedy
from hls.core.engine import Engine, ResourceRequest
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent

#: Each reference engine's load, in order: its dtype, and the GPU the engine host placed it on.
LOADS: list[tuple[str, int | None]] = []
#: Each pool member's load, in order: the GPU the engine host placed it on, None for none.
MEMBER_LOADS: list[int | None] = []
#: Each pool member's release, in order: the GPU it held, None for none.
MEMBER_UNLOADS: list[int | None] = []
#: Ids each member reply carries.
TOKENS_PER_REPLY = 1250
#: Positions of each reply the reference scores as flips, and the gap each carries, in nats.
FLIPS, GAP = 20, 0.1
#: The serving package version every member reports.
MEMBER_VERSION = "0.0.0-fake"
#: What every engine reports of its stack.
STACK = {"torch": "2.4.0", "runtime": {"cuda": "12.4", "cudnn": "90100"}, "device_name": "NVIDIA B200"}


#: The engine cache's files a load writes, under ``FLASHINFER_WORKSPACE_BASE``: each kernel's
#: build, by the kernel's name, and the logs of builds and of loads, one name a line.
KERNEL, COMPILES, LOADS_LOG = ".cache/flashinfer/{}.so", "compiles.log", "loads.log"
#: The object file each kernel's build links it from, kept beside it.
KERNEL_OBJECT = ".cache/flashinfer/{}.o"
#: A file of the engine cache that makes a load hang.
HANG = "hang"
#: A file of the engine cache that makes a pool member's release hang.
RELEASE_HANG = "release-hang"


class _Lingering:
    """An object in a reference cycle, so only a collection frees it, whose finalizer waits for
    ever: what a ZMQ context's does when a socket of it holds a message for a peer that is gone."""

    def __init__(self) -> None:
        self.cycle = self

    def __del__(self) -> None:
        threading.Event().wait()


def _compile(kernel: str) -> None:
    """Build *kernel* into the engine cache unless the cache holds it, and note the load."""
    base = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if base is None:
        return
    root = Path(base)
    if (root / HANG).exists():
        threading.Event().wait()
    built = root / KERNEL.format(kernel)
    if not built.exists():
        (Path(os.environ["TMPDIR"]) / f"{kernel}.o").write_text(kernel)
        built.parent.mkdir(parents=True, exist_ok=True)
        (root / KERNEL_OBJECT.format(kernel)).write_text(kernel)
        built.write_text(f"{kernel} for {STACK['device_name']}\n")
        with open(root / COMPILES, "a") as log:
            log.write(f"{kernel}\n")
    with open(root / LOADS_LOG, "a") as log:
        log.write(f"{kernel}\n")


def _slot() -> ResourceRequest:
    return ResourceRequest(gpu_count=1, gpu_kind="B200")


class FakeReference(Engine):
    """The HF engine a calibration builds (``hls.calibrate.HFEngine``): the pinned bf16
    reference, or with ``dtype="float32"`` the preflight anchor. It places itself on one GPU, as
    the HF engine does."""

    placeable: ClassVar[bool] = True

    def __init__(self, **kwargs: Any) -> None:
        self.dtype = kwargs.get("dtype", "bfloat16")
        self.device_index: int | None = None

    @property
    def resources(self) -> ResourceRequest:
        return _slot()

    def _load(self) -> None:
        LOADS.append((self.dtype, self.device_index))
        _compile(f"reference-{self.dtype}")

    async def stack_report(self) -> dict:
        return {**STACK, "runtime": dict(STACK["runtime"])}

    async def generate(self, prompts: list[Envelope]) -> list[Envelope]:
        return [Envelope(request_id=p.request_id, contents=[TextPayload(text="x", tokens=[11, 22, 33])])
                for p in prompts]

    async def teacher_force(self, items: list[Envelope]) -> list[list[float]]:
        out = []
        for item in items:
            n = len(item.first_of("text").tokens)
            flips = 0 if item.request_id.startswith("preflight-") else min(FLIPS, n)
            out.append([GAP] * flips + [0.0] * (n - flips))
        return out


class FakeMember(Engine):
    """A pool member, which no serving stack's placement moves: it serves every session it is
    handed, stamped as the drive loop stamps a request."""

    def __init__(self) -> None:
        self.decode = Greedy()

    def _load(self) -> None:
        MEMBER_LOADS.append(getattr(self, "device_index", None))
        _compile("member")
        base = os.environ.get("FLASHINFER_WORKSPACE_BASE")
        if base is not None and (Path(base) / RELEASE_HANG).exists():
            self.context = _Lingering()

    def _unload(self) -> None:
        MEMBER_UNLOADS.append(getattr(self, "device_index", None))

    @property
    def resources(self) -> ResourceRequest:
        return _slot()

    async def stack_report(self) -> dict:
        return {**STACK, "runtime": dict(STACK["runtime"])}

    async def serving_version(self) -> str:
        return MEMBER_VERSION

    async def serve_traffic(self, loaded: list, *, seed: int, request_timeout_sec: float,
                            max_sessions: int | None = None, drive_timeout_sec: float | None = None,
                            agent: Any = None, max_context_len: int | None = None) -> Trace:
        events = []
        for tenant, items in loaded:
            cap = tenant.workload.output_length_cap.max_tokens()
            for item in items:
                rid, n = f"{item.request_id}.0", min(TOKENS_PER_REPLY, cap)
                request = Envelope(request_id=rid, metadata={
                    "messages": [{"role": "user", "content": item.request_id}], "tenant_id": tenant.id,
                    "max_tokens": cap})
                reply = Envelope(request_id=rid, contents=[TextPayload(
                    text="x", tokens=list(range(n)), finish_reason="length" if n >= cap else "stop")])
                events.append(TraceEvent(request_id=rid, session_id=item.request_id, turn_index=0, input=request,
                                         response=ArtifactResponse(output=reply), t_send=0.0, t_recv=1.0))
        return Trace(events=events)


def members(task: Any) -> dict[str, type[FakeMember]]:
    """``hls.calibrate.member_engines`` for these engines: a fake member for every pool name."""
    return {name: FakeMember for name in calibrate.POOL_ENGINE_NAMES}


def install() -> None:
    """Make ``hls.calibrate`` build these engines: its HF engine and every pool member."""
    calibrate.HFEngine = FakeReference
    calibrate.member_engines = members
