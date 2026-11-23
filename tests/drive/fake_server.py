"""A contestant's server for the deploy_drive tests, put into a fake image as
``/workspace/server.py``. It speaks sample_task's chat contract.

    server.py --mode MODE --host HOST --port PORT [--health-after SEC] [--chat-after SEC]

Modes:
- ``serve``: ``/health`` answers 200 and every chat request "B";
- ``echo``: as ``serve``, but every chat request is answered "B ", then the text of ``reply.txt``
  beside the server, as a server that baked a value into its own files sends it;
- ``crash``: exits 3 at once, before it is ready;
- ``refuse``: prints a ``refused:`` line, as a harness role marks a refusal, and exits 3 at once;
- ``hang``: runs, but ``/health`` answers 503 ``loading`` forever;
- ``fade``: runs as ``hang`` does for a second, then exits 3; ``fade0`` the same with status 0;
- ``noroute``: ``/health`` answers 200, but the chat route answers 404 forever;
- ``stall``: ready, and answers the first chat request (the readiness probe), then never
  answers another;
- ``die``: ready, answers the readiness probe, then exits 3 on the next chat request; ``die0``
  the same with status 0;
- ``fill``: writes ``FILL_BYTES`` into ``$TMPDIR``, then runs as ``hang`` does;
- ``spill``: ready, answers the readiness probe, then writes ``FILL_BYTES`` into ``$TMPDIR``
  on the next chat request and answers that request and every later one after a second;
- ``chmodrace``: makes a directory of 2,000 empty files in ``$TMPDIR``, starts a child that takes
  the owner's permissions off it again and again, then writes ``FILL_BYTES`` there in plain view
  and serves as ``serve`` does;
- ``hide``: holds ``FILL_BYTES`` in a file that never has a name (``O_TMPFILE``), which no walk
  of a directory can see, and serves as ``serve`` does;
- ``brim``: writes ``FILL_BYTES`` into ``$TMPDIR`` and, as though that filled the node's disk,
  leaves its seal's directory, on the same disk, unable to take a new file, a rename or a
  removal, as a full disk would refuse the new blocks those need; then exits 3 at once, before
  it is ready. A real container never sees its seal's directory: the fake container runs the
  server on the host, in ``<role>/wd/root``, and the seal keeps its files in ``<role>/seal``.

In every mode ``/health`` answers 503 ``loading`` until ``--health-after`` seconds after the
server started, and the chat route 503 until ``--chat-after`` seconds after (both default 0), as a
server still loading its model would.

With ``--jit-sec SEC`` it first does what a JIT build does with the engine cache
(``FLASHINFER_WORKSPACE_BASE``), before it listens: unless the cache holds its kernel
(``KERNEL``), it keeps a temporary file in ``TMPDIR`` for *SEC* seconds, as a build's
intermediates, then writes the kernel and the object file it was linked from
(``KERNEL_OBJECT``) and notes the build in ``COMPILES``; every start notes itself in ``LOADS``.
So a server seeded with a built cache listens at once, and one with an empty cache only after
*SEC* seconds.

It exits as soon as its parent does: every process of a real step dies with the step, and the
local launcher stops only a role's first process."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

#: Bytes the ``fill`` and ``spill`` modes write.
FILL_BYTES = 4 * 2**20
#: The files a ``--jit-sec`` start writes under ``FLASHINFER_WORKSPACE_BASE``: the kernel it builds,
#: the object file it links it from, and the logs of builds and of starts, a line each.
KERNEL, KERNEL_OBJECT = ".cache/flashinfer/kernel.so", ".cache/flashinfer/kernel.o"
COMPILES, LOADS = "compiles.log", "loads.log"
#: Descriptors the server keeps open while it runs.
_HELD: list[int] = []
ANSWER = {"id": "c", "choices": [{"message": {"role": "assistant", "content": "B"}, "finish_reason": "stop",
                                  "token_ids": [33]}], "usage": {"completion_tokens": 1}}


def _fill() -> None:
    with open(Path(os.environ["TMPDIR"]) / "fill", "wb") as f:
        for _ in range(FILL_BYTES // 2**20):
            f.write(b"x" * 2**20)
        f.flush()
        os.fsync(f.fileno())


def _chmod_race() -> None:
    d = Path(os.environ["TMPDIR"]) / "d"
    d.mkdir()
    for i in range(2000):
        (d / f"f{i}").write_bytes(b"")
    if os.fork() == 0:
        parent = os.getppid()
        while os.getppid() == parent:
            try:
                os.chmod(d, 0)
            except OSError:
                pass
        os._exit(0)
    _fill()


def _jit(sec: float) -> None:
    """Build the kernel into the engine cache unless it holds it, taking *sec* seconds, and note
    the start (the module docstring)."""
    root = Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
    if not (root / KERNEL).exists():
        intermediate = Path(os.environ["TMPDIR"]) / "kernel.o.part"
        intermediate.write_text("building")
        time.sleep(sec)
        (root / KERNEL).parent.mkdir(parents=True, exist_ok=True)
        (root / KERNEL_OBJECT).write_text("kernel object")
        (root / KERNEL).write_text("kernel")
        with open(root / COMPILES, "a") as log:
            log.write("kernel\n")
    with open(root / LOADS, "a") as log:
        log.write("load\n")


def _hide() -> int:
    fd = os.open(os.environ["TMPDIR"], os.O_TMPFILE | os.O_RDWR, 0o600)
    os.write(fd, b"x" * FILL_BYTES)
    os.fsync(fd)
    return fd


def app(mode: str, *, health_at: float = 0.0, chat_at: float = 0.0) -> FastAPI:
    """The server of *mode*, whose health route opens at ``time.monotonic()`` *health_at* and
    chat route at *chat_at*."""
    api = FastAPI()
    answered: list[int] = []

    @api.get("/health")
    async def _health():
        if mode in ("hang", "fade", "fade0", "fill") or time.monotonic() < health_at:
            return PlainTextResponse("loading", status_code=503)
        return {"status": "ok"}

    @api.post("/v1/chat/completions")
    async def _chat(body: dict):
        if time.monotonic() < chat_at:
            return PlainTextResponse("loading", status_code=503)
        if mode == "echo":
            text = "B " + Path(__file__).with_name("reply.txt").read_text()
            return {**ANSWER, "choices": [{**ANSWER["choices"][0], "message": {"role": "assistant",
                                                                                "content": text}}]}
        if mode == "noroute":
            return PlainTextResponse("no such model", status_code=404)
        if answered and mode == "stall":
            await asyncio.sleep(3600)
        if answered and mode in ("die", "die0"):
            os._exit(0 if mode == "die0" else 3)
        if answered and mode == "spill":
            if len(answered) == 1:
                _fill()
            await asyncio.sleep(1)
        answered.append(1)
        return ANSWER

    return api


def _exit_with_parent() -> None:
    parent = os.getppid()
    while os.getppid() == parent:
        time.sleep(0.2)
    os._exit(0)


def _fade(status: int) -> None:
    time.sleep(1.0)
    os._exit(status)


def main() -> int:
    since = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("serve", "echo", "crash", "refuse", "hang", "fade", "noroute",
                                                     "stall", "die", "fill", "spill", "chmodrace", "hide", "brim",
                                                     "fade0", "die0"))
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--health-after", type=float, default=0.0)
    p.add_argument("--chat-after", type=float, default=0.0)
    p.add_argument("--jit-sec", type=float, default=None)
    args = p.parse_args()
    threading.Thread(target=_exit_with_parent, daemon=True).start()
    if args.jit_sec is not None:
        _jit(args.jit_sec)
    if args.mode == "crash":
        print("server: the model does not fit", file=sys.stderr)
        return 3
    if args.mode == "brim":
        _fill()
        os.chmod(Path(os.environ["FAKE_CONTAINER_ROOT"]).parents[1] / "seal", 0o500)
        print("server: no space left on the node's disk", file=sys.stderr)
        return 3
    if args.mode == "refuse":
        print("refused: this server refuses the task as it is declared", file=sys.stderr)
        return 3
    if args.mode == "fill":
        _fill()
    if args.mode in ("fade", "fade0"):
        threading.Thread(target=_fade, args=(0 if args.mode == "fade0" else 3,), daemon=True).start()
    if args.mode == "chmodrace":
        _chmod_race()
    if args.mode == "hide":
        _HELD.append(_hide())  # open until the server exits
    served = app(args.mode, health_at=since + args.health_after, chat_at=since + args.chat_after)
    uvicorn.run(served, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
