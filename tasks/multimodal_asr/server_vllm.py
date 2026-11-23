"""multimodal-asr's measured baseline server: stock vLLM behind the arena's wire.

Two processes in one container: ``vllm serve <whisper>`` (paged KV + continuous batching; its
OpenAI ``/v1/audio/transcriptions`` endpoint is the stable public surface across vLLM versions)
plus this thin adapter, which speaks the arena's JSON wire (``AudioTranscriptionProtocol``:
``{audio_b64, sample_rate, max_tokens?}`` -> ``{"text": ...}``) and forwards each request as a
multipart WAV to the local vLLM.

``hls baseline-credit`` drives it to measure the task's baseline credit. It is not part of the
agentic arena, and the build sandbox never holds it.

Weights load from the provisioned mount ($HLS_WEIGHTS_DIR) when present -- a local path, so
HF_HUB_OFFLINE=1 is satisfied; MODEL_ID is the non-provisioned fallback. An empty ``audio_b64``
is the readiness probe: fast-return without touching vLLM.
"""

from __future__ import annotations
import argparse
import base64
import os
import socket
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Response

MODEL_ID = os.environ.get("HLS_BASELINE_MODEL", "openai/whisper-large-v3")


def _inner_port() -> int:
    """A free port for the inner vLLM; the adapter owns the contract port.

    Asked for rather than fixed because these containers share the host's network namespace. On
    a fixed port, a second baseline on the same node cannot bind, and its adapter then forwards to
    the first one's engine while its own health check still passes.
    """
    env = os.environ.get("HLS_VLLM_INNER_PORT")
    if env:
        return int(env)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


_VLLM_PORT = _inner_port()

# One client for the whole process, shared by every request: a client per request never reuses a
# connection, and the socket churn once a few hundred are in flight resets this adapter's own
# listener. The pool is sized past the connections vLLM holds at the arena's rates, so requests
# queue in the engine, where it can batch them, rather than here.
#
# keepalive_expiry is shorter than the httpx default of 5.0 s. The inner server retires idle
# connections on a deadline of its own, and when the two deadlines are close the client can take
# a connection the server has just closed.
_LIMITS = httpx.Limits(max_connections=1024, max_keepalive_connections=512,
                       keepalive_expiry=2.0)


def _weights_source() -> str:
    wdir = os.environ.get("HLS_WEIGHTS_DIR")
    return wdir if (wdir and Path(wdir).is_dir()) else MODEL_ID


def _spawn_vllm(src: str, extra_args: list[str] | None = None) -> subprocess.Popen:
    # served-model-name pins the OpenAI `model` field the adapter sends, regardless
    # of whether src is a mount path or a hub id.
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", src, "--served-model-name", "baseline-whisper",
           "--host", "127.0.0.1", "--port", str(_VLLM_PORT)]
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
        # Ready only when the inner vLLM is: proxy its /health so the harness's
        # readiness gate reflects the model actually being loaded.
        if vllm_proc.poll() is not None:
            return Response(status_code=503, content="vllm exited")
        try:
            r = await app.state.client.get(f"{base}/health", timeout=5.0)
            return {"status": "ok"} if r.status_code == 200 else Response(
                status_code=503, content="vllm loading")
        except httpx.TransportError:
            return Response(status_code=503, content="vllm starting")

    @app.post("/v1/audio/transcriptions")
    async def transcribe(body: dict):
        audio_b64 = body.get("audio_b64") or ""
        if not audio_b64:
            return {"text": "", "token_ids": []}   # readiness probe fast-path
        wav = base64.b64decode(audio_b64)
        # Forward the language the arena sends. Without it vLLM detects the language with a
        # second engine request for the same audio, a full extra prefill, encoder included.
        form = {"model": "baseline-whisper", "response_format": "json", "temperature": "0"}
        language = body.get("language")
        if language:
            form["language"] = language
        r = await app.state.client.post(
            f"{base}/v1/audio/transcriptions",
            files={"file": ("clip.wav", wav, "audio/wav")},
            data=form)
        r.raise_for_status()
        return {"text": r.json().get("text", "")}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    # parse_known_args: Baseline.serve_command tuning flags ride after --host/--port and belong
    # to the inner vLLM server; forward them.
    args, extra = ap.parse_known_args()
    proc = _spawn_vllm(_weights_source(), extra)
    try:
        uvicorn.run(_build_app(proc), host=args.host, port=args.port, log_level="warning")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
