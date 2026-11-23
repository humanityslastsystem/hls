"""multimodal-omni vLLM-Omni BASELINE server.

Two processes in one container: ``vllm serve <qwen3-omni> --omni`` (vLLM-Omni's
OpenAI-compatible chat completions) plus this thin adapter, which proxies the
arena's OmniChatProtocol wire -- already OpenAI-shaped -- to the inner server.

The adapter exists for two harness affordances the inner server lacks:
``/health`` (proxied from vLLM-Omni's ``/health``) and pinning
``modalities=["text"]`` on every request (the arena's contract is text-output only;
letting the talker/code2wav stages run would burn latency on audio nobody scores).
Weights load from $HLS_WEIGHTS_DIR.
"""

from __future__ import annotations
import argparse
import os
import shutil
import socket
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

MODEL_ID = os.environ.get("HLS_BASELINE_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")


def _inner_port() -> int:
    """A free port for the inner vLLM; the adapter owns the contract port.

    Asked for rather than fixed because these containers share the host's network
    namespace. On a fixed port, a second baseline on the same node cannot bind, and its
    adapter then forwards to the first one's engine while its own health check still
    passes -- both servers are then measured serving one engine, which is silent and
    wrong.
    """
    env = os.environ.get("HLS_VLLM_INNER_PORT")
    if env:
        return int(env)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


_VLLM_PORT = _inner_port()

# One client for the whole process, shared by every request. A client per request never
# reuses a connection, and the socket churn once a few hundred are in flight takes down
# the adapter's own listener rather than queueing in the engine.
#
# keepalive_expiry is shortened from the httpx default of 5.0 s. The inner server retires
# idle connections on a deadline of its own, and when the two deadlines are close the
# client can take a connection the server has just closed. A shorter deadline on this
# side closes it.
_LIMITS = httpx.Limits(max_connections=1024, max_keepalive_connections=512,
                       keepalive_expiry=2.0)


def _weights_source() -> str:
    wdir = os.environ.get("HLS_WEIGHTS_DIR")
    return wdir if (wdir and Path(wdir).is_dir()) else MODEL_ID


def _deploy_config() -> str | None:
    """Absolute path of the PACKAGED qwen3_omni_moe deploy config. Auto-resolution
    keys off the model identity in the registry, which a local /data/weights path
    may not match -- pass the packaged yaml explicitly (it owns the per-stage
    memory/device layout)."""
    try:
        import vllm_omni
        path = Path(vllm_omni.__file__).parent / "deploy" / "qwen3_omni_moe.yaml"
        return str(path) if path.exists() else None
    except ImportError:
        return None


def _spawn_vllm_omni(src: str, extra_args: list[str] | None = None) -> subprocess.Popen:
    exe = shutil.which("vllm-omni") or shutil.which("vllm")
    if exe is None:
        raise RuntimeError("neither 'vllm-omni' nor 'vllm' found on PATH")
    cmd = [exe, "serve", src,
           "--served-model-name", "baseline-omni",
           "--host", "127.0.0.1", "--port", str(_VLLM_PORT),
           "--trust-remote-code", "--omni",
           # 30B-class stage engines exceed the 300 s per-stage READY default: each of
           # the three pages a 66 GiB checkpoint.
           "--stage-init-timeout", "1800",
           # And the orchestrator's own budget has to cover all three of those. It
           # defaults to 600 s, which expires while a later stage is still loading and
           # makes the per-stage grant above unreachable. vllm-omni reads these from the
           # command line only -- there is no env var, and no deploy-config key.
           "--init-timeout", "5400"]
    cfg = _deploy_config()
    if cfg is not None:
        cmd += ["--deploy-config", cfg]
    # NOTE: no --gpu-memory-utilization -- the deploy yaml owns per-stage memory.
    cmd += list(extra_args or [])          # Baseline.serve_command tuning flags
    print(f"[baseline] launching: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def _build_app(vllm_proc: subprocess.Popen) -> FastAPI:
    base = f"http://127.0.0.1:{_VLLM_PORT}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=600.0, limits=_LIMITS) as cli:
            app.state.client = cli
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        if vllm_proc.poll() is not None:
            return Response(status_code=503, content="vllm-omni exited")
        try:
            r = await app.state.client.get(f"{base}/health", timeout=5.0)
            return {"status": "ok"} if r.status_code == 200 else Response(
                status_code=503, content="vllm-omni loading")
        except httpx.TransportError:
            return Response(status_code=503, content="vllm-omni starting")

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        # The arena wire IS OpenAI chat -- pass through, pinning the model name and
        # text-only output (the contract scores text; skip talker/code2wav).
        body = {**body, "model": "baseline-omni", "modalities": ["text"]}
        r = await app.state.client.post(f"{base}/v1/chat/completions", json=body)
        return JSONResponse(status_code=r.status_code, content=r.json())

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    # parse_known_args: Baseline.serve_command tuning flags ride AFTER --host/--port
    # and belong to the INNER vLLM server -- forward them.
    args, extra = ap.parse_known_args()
    proc = _spawn_vllm_omni(_weights_source(), extra)
    try:
        uvicorn.run(_build_app(proc), host=args.host, port=args.port, log_level="warning")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
