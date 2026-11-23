"""multimodal-tts vLLM-Omni BASELINE server (owner-side capacity measurement only).

Two processes in one container: ``vllm serve <qwen3-tts> --omni`` (vLLM-Omni's
OpenAI-compatible ``/v1/audio/speech``; at the 0.22 line the vllm entrypoint carries
the --omni flag, with ``vllm-omni serve`` as the newer spelling -- the launcher looks for both)
plus this thin adapter, which speaks the arena's TTSStreamProtocol wire:

    POST /v1/audio/speech {"input": text, "voice": ..., "stream": true}
      -> SSE: data:{"audio": <b64 wav>} data:{"codec_codes": [[...]]} data:[DONE]

The adapter requests the FULL wav from vLLM-Omni and emits it as a single SSE frame.
Empty ``input`` is the readiness probe:
fast-reject 400 (never synthesize on probe). Weights load from $HLS_WEIGHTS_DIR.

**The decode is the task contract's**: all sixteen code groups sampled from the plain
softmax at temperature 1, with no top-k, no top-p and no repetition penalty. Codebook 0
goes through ``default_sampling_params`` and the other fifteen through
``subtalker_sampling_params``; both are written into the deploy config below, along with
speaker, language and task type on every request. Argmax is not an option here: taken
on all sixteen groups it leaves utterances that never terminate.

**The pin is checked against what the engine actually did**, not against the file this
process wrote. vLLM-Omni builds a real SamplingParams out of a stage's
``default_sampling_params``, so a typo there stops the server at startup; but
``subtalker_sampling_params`` is a bare dict that nothing validates, and a misspelled key
there silently restores sampling on fifteen of the sixteen groups. So the spooling hook
reports both back from inside the engine, and this adapter refuses to serve another
request when they are not what was pinned. A baseline that measured the wrong decode
without saying so is worse than one that stops.

**The codec codes reach the wire through a side channel**, because vLLM-Omni publishes
no way to read them from an HTTP response: the speech request carries no flag that asks
for them, the response and its SSE events carry audio only, and the serving layer drops
any non-audio stage output it is handed. See ``omni_codes_spool``: the stage-0 producer
hook is pointed at a wrapper that writes each finished request's codes to a spool file,
and each request here carries a key naming its file. The codes frame is assembled after
the audio frame has already been yielded, so it cannot delay the per-request clock,
which stops at the last audio frame. A request whose codes do not arrive is streamed
without a codes frame, and the harness records that absence against the reply rather
than failing the drive.

``HLS_TTS_CODES=0`` serves the pinned decode without any of that: no hook, no codes, and
no read-back, which is the configuration to use for a latency measurement that should
not be staked on the side channel. It is printed at startup, because a baseline measured
that way carries no correctness evidence and its sub-talker pin is unconfirmed.
"""

from __future__ import annotations
import argparse
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

import omni_codes_spool

MODEL_ID = os.environ.get("HLS_BASELINE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")

#: Dotted path the deploy config names for the stage-0 producer hook. It is imported by
#: the vLLM worker process, which is why the launcher puts this file's directory on that
#: process's PYTHONPATH.
_SPOOL_HOOK = "omni_codes_spool.spooling_hook"


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

# --- The pinned request -------------------------------------------------------
# Contract values. They must match the reference oracle's
# (hls.reference.tts.Qwen3TTSReference), which cannot be imported here: this server runs
# in the baseline image, which carries no hls source.
#
# The speaker, the language and the (absent) style instruction all enter the talker's
# prefix, so each one changes the codes for the same sentence. The arena wire's `voice`
# field is not read: this baseline serves one pinned contract, and answering in whatever
# voice a caller names would make it a different measurement per request.
#
# task_type is stated rather than left to inference. vllm-omni infers "Base" (voice
# cloning) from a voice it finds among its precomputed speaker profiles, and a silent
# switch to a cloning path would be a different conditioning under the same name.
_SPEAKER = "ryan"           # lowercased: the adapter lowercases and matches on that
_LANGUAGE = "English"
_TASK_TYPE = "CustomVoice"

#: Codebook 0's decode, written over stage 0's ``default_sampling_params``. The contract
#: is the plain softmax: temperature 1, no truncation of either kind, no penalty. Top-k 0
#: and top-p 1.0 are how vLLM spells "keep the whole vocabulary".
#:
#: Written rather than left to the packaged yaml, which carries the checkpoint's own
#: 0.9, 50 and 1.05. Every value is stated so the config this server runs is the
#: contract's rather than the package's, and so the read-back below has something to
#: compare against.
#:
#: Sampled, not argmax. Taking the argmax on all sixteen code groups leaves utterances
#: that never terminate, because a frame's talker input is the sum of
#: all sixteen code embeddings and a degenerate sub-talker feeds the talker its own loop.
#: The gate reads reported distributions rather than a single right answer, so a sampled
#: decode is what it is built for.
_TALKER_SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0,
                    "repetition_penalty": 1.0}

#: The sub-talker's decode, the same plain softmax. It samples, and that is not a free
#: choice: a frame's talker input embedding is the sum of all sixteen code embeddings, so
#: a degenerate sub-talker feeds the talker its own loop. Its own
#: dict because the sub-talker is not a vLLM sampler and shares none of the keys above.
_SUBTALKER_SAMPLING = {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0}

#: End-of-speech is blocked for this many opening frames, which is the packaged value and
#: the one the reference reproduces when it prices the served codes. Checked rather than set: a
#: server blocking it for a different number of frames would have its opening scored
#: against a distribution the reference never saw.
_EXPECTED_MIN_TOKENS = 2

#: Values that both mean "no top-k truncation". vLLM accepts -1 and hands back 0 on the
#: SamplingParams it builds, so a pin of -1 is echoed as 0 and the two are one setting.
_TOP_K_OFF = (-1, 0)


def _same_setting(name: str, got: object, pinned: object) -> bool:
    """Whether an echoed sampling setting selects the decode that was pinned.

    Semantic rather than byte-for-byte. The engine rebuilds these into its own objects
    and may spell a value differently coming out, and this check exists to catch a
    different DECODE, not a different spelling of the same one. So the only values
    treated as equal are ones that select the same behaviour, and a value that is merely
    close is still a complaint."""
    if name.endswith("top_k") and got in _TOP_K_OFF and pinned in _TOP_K_OFF:
        return True
    try:
        # One numeric rule covers the switches too: bool subclasses int, so True, 1 and
        # 1.0 compare equal and False does not, which is what those three mean. Resist
        # adding a truthiness branch here -- it would accept a do_sample of 2, or of
        # "true", as though it were the pin, and a value nobody can account for is
        # exactly what this check exists to surface.
        return float(got) == float(pinned)
    except (TypeError, ValueError):
        # A missing key arrives here as None and compares unequal, which is the answer:
        # a setting the engine does not report is a setting nothing confirmed.
        return got == pinned

# No seed is written into the config. A seed makes vLLM-Omni build per-row generators
# and, on a graph-wrapped talker, run the sub-talker one row at a time, which costs
# throughput on the very measurement this server exists to take.

#: Whether the codes side channel is installed at all. Set ``HLS_TTS_CODES=0`` to serve
#: the pinned decode WITHOUT it: no producer hook in the deploy config, no key on any
#: request, no codes frame on any reply, and no read-back of what the engine decoded
#: under.
#:
#: The fallback exists because the side channel and the read-back stand or fall together.
#: The hook is what reports the engine's own decode, so with it installed a hook that
#: does not run makes this server refuse to serve; that is the right answer for a graded
#: measurement and the wrong one for a latency measurement that only needs the pin the
#: yaml already carries. Turning it off buys a baseline
#: whose sub-talker setting is unconfirmed and whose replies carry no gradable evidence.
#: Both facts are printed at startup, because a run that took this path has to say so.
_CODES_ENABLED = os.environ.get("HLS_TTS_CODES", "1") != "0"

#: How long a request waits for its codes to appear in the spool, after its audio has
#: already been streamed. The hook writes them before the codec stage runs, so they exist
#: by the time any audio does; this only covers the cross-process write becoming visible.
#: Past it the reply goes out without codes and the harness records the gap.
_CODES_WAIT_SEC = 2.0

#: Poll interval while waiting for a spool file.
_CODES_POLL_SEC = 0.01

#: What the engine reported about its own decode: None until the first synthesis
#: completes, then the list of disagreements with the pin, empty when it held.
_pin_complaints: list[str] | None = None


def _weights_source() -> str:
    wdir = os.environ.get("HLS_WEIGHTS_DIR")
    return wdir if (wdir and Path(wdir).is_dir()) else MODEL_ID


def _deploy_config() -> str | None:
    """Path to the qwen3_tts deploy config, with the pinned decode and the codes hook
    written into it.

    Starts from the PACKAGED yaml. The example's repo-relative
    'vllm_omni/deploy/qwen3_tts.yaml' does not resolve in a pip install ("Deploy
    config not found ... using pipeline defaults"), and the defaults stack every stage
    engine onto cuda:0 at high memory utilization -- the second stage then dies on OOM.
    The packaged yaml owns the per-stage memory/device layout (validated on one GPU,
    matching the arena's 1-GPU artifact slot; the _high_concurrency variant needs two
    GPUs).

    It is rewritten rather than used as shipped because the packaged decode is the
    checkpoint's sampling default on codebook 0 too, and this baseline is what every
    contestant's score is measured against: it has to synthesize under the decode the
    contract pins. Only the sampling keys and the producer hook are touched, so the rest
    of the layout keeps tracking upstream, which a forked copy of the file would not.

    A yaml whose talker stage declares no ``subtalker_sampling_params`` is refused. That
    key is the only handle on the fifteen sub-talker groups, and serving them under
    settings nothing states is worse than not serving.

    Returns None if the package or its yaml is missing, which leaves the caller on
    vllm-omni's own resolution."""
    try:
        import vllm_omni
    except ImportError:
        return None
    path = Path(vllm_omni.__file__).parent / "deploy" / "qwen3_tts.yaml"
    if not path.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(path.read_text())
        stages = cfg.get("stages") or []
        if not stages:
            raise ValueError("deploy config declares no stages")
        talker_stages = 0
        for stage in stages:
            params = stage.setdefault("default_sampling_params", {})
            params.update(_TALKER_SAMPLING)
            if "subtalker_sampling_params" in stage:
                stage["subtalker_sampling_params"].update(_SUBTALKER_SAMPLING)
                # The talker stage is also the stage that produces the codes, so its
                # producer hook is the one to redirect. vLLM-Omni has no deploy-config
                # field of this name, so the key rides through as a stage extra, which
                # is applied over the pipeline's own registration.
                if _CODES_ENABLED:
                    stage["custom_process_next_stage_input_func"] = _SPOOL_HOOK
                talker_stages += 1
        if not talker_stages:
            raise ValueError(
                "no stage declares subtalker_sampling_params, so the fifteen "
                "sub-talker code groups cannot be pinned at all")
        out = Path("/tmp/qwen3_tts_pinned.yaml")
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    except Exception as exc:
        # Serving unpinned is worse than not serving: the credit every contestant is
        # scored against would be measured on a decode the contract forbids.
        raise RuntimeError(f"could not pin the decode into {path}: {exc}") from exc
    hook = _SPOOL_HOOK if _CODES_ENABLED else "NONE (HLS_TTS_CODES=0)"
    print(f"[baseline] deploy config: {out} (all groups sampled at "
          f"{_TALKER_SAMPLING['temperature']}/top-k {_TALKER_SAMPLING['top_k']}, "
          f"codebook 0 at penalty {_TALKER_SAMPLING['repetition_penalty']}, no seed; "
          f"{len(stages)} stage(s), {talker_stages} with a sub-talker; "
          f"codes hook {hook})", flush=True)
    if not _CODES_ENABLED:
        print("[baseline] HLS_TTS_CODES=0: replies carry no codec codes, and the "
              "decode pin is UNCONFIRMED (nothing reads it back from the engine). "
              "Latency measured here is valid; correctness evidence is not.",
              flush=True)
    return str(out)


def _observed_decode() -> str:
    """What the hook reported, as one line for the log, or why it could not be read."""
    path = omni_codes_spool.spool_dir() / omni_codes_spool.DECODE_FILE
    try:
        return path.read_text().strip()
    except Exception as exc:   # noqa: BLE001 - this runs while a failure is reported
        return f"<unreadable at {path}: {exc!r}>"


def _check_pin() -> list[str]:
    """What the engine reported about its own decode, against what was pinned.

    One complaint per disagreement, empty when the pin held. Reads the file the spooling
    hook wrote from inside the stage-0 worker, so this compares the pin against an
    observation rather than against the config this process authored. A missing file is
    itself a complaint: it means the hook never ran, so neither the decode nor the codes
    side channel can be confirmed."""
    path = omni_codes_spool.spool_dir() / omni_codes_spool.DECODE_FILE
    try:
        observed = json.loads(path.read_text())
    except Exception as exc:   # noqa: BLE001 - absent, partial and unreadable are one case
        return [f"the engine reported no decode at {path} ({exc!r}), so the stage-0 "
                f"hook named {_SPOOL_HOOK} did not run"]

    complaints: list[str] = []
    sub = observed.get("subtalker_sampling_params")
    if not isinstance(sub, dict):
        complaints.append(
            f"sub-talker sampling params came back as {sub!r} from config attribute "
            f"{observed.get('config_attr_path')!r}, so the pin is unconfirmed")
    else:
        # Every key, not only do_sample. These are the settings the sub-talker actually
        # draws under, and vLLM-Omni validates none of them, so a typo in any one is the
        # same silent class of defect as a typo in do_sample. They reach the hook off a
        # bare config dict rather than through a SamplingParams, so nothing normalizes
        # them on the way, but they go through the same comparison anyway: one rule for
        # what counts as the same setting is easier to trust than two.
        for name, pinned in _SUBTALKER_SAMPLING.items():
            got = sub.get(name)
            if not _same_setting(name, got, pinned):
                complaints.append(f"sub-talker {name} is {got!r}, pinned {pinned!r}")

    talker = observed.get("talker_sampling_params") or {}
    for name, pinned in _TALKER_SAMPLING.items():
        got = talker.get(name)
        if not _same_setting(name, got, pinned):
            complaints.append(f"talker {name} is {got!r}, pinned {pinned!r}")
    if not _same_setting("min_tokens", talker.get("min_tokens"), _EXPECTED_MIN_TOKENS):
        complaints.append(
            f"talker min_tokens is {talker.get('min_tokens')!r}, expected "
            f"{_EXPECTED_MIN_TOKENS} (the reference blocks end-of-speech for exactly "
            f"that many opening frames)")
    if talker.get("seed") is not None:
        complaints.append(f"talker seed is {talker.get('seed')!r}, expected none")
    return complaints


async def _read_codes(key: str) -> list | None:
    """This request's codec codes from the spool, or None if they never arrived.

    The file is deleted as soon as it is read. Under a sealed container the spool is a
    small tmpfs shared with everything else in there, so what a drive leaves behind has
    to be bounded by how many requests are in flight rather than by how many were served.
    """
    path = omni_codes_spool.spool_dir() / f"{key}.json"
    loop = asyncio.get_event_loop()
    deadline = loop.time() + _CODES_WAIT_SEC
    while True:
        try:
            rows = json.loads(path.read_text())
            path.unlink(missing_ok=True)
            return rows if isinstance(rows, list) and rows else None
        except FileNotFoundError:
            if loop.time() >= deadline:
                return None
            await asyncio.sleep(_CODES_POLL_SEC)
        except Exception:   # noqa: BLE001 - unreadable is the same as absent here
            path.unlink(missing_ok=True)
            return None


def _spawn_vllm_omni(src: str, extra_args: list[str] | None = None) -> subprocess.Popen:
    exe = shutil.which("vllm-omni") or shutil.which("vllm")
    if exe is None:
        raise RuntimeError("neither 'vllm-omni' nor 'vllm' found on PATH")
    cmd = [exe, "serve", src,
           "--served-model-name", "baseline-tts",
           "--host", "127.0.0.1", "--port", str(_VLLM_PORT),
           "--trust-remote-code", "--omni",
           # Stage engines exceed the 300 s per-stage READY default when the checkpoint
           # pages off a shared network filesystem.
           "--stage-init-timeout", "1800",
           # And the orchestrator's own budget has to cover every stage. It defaults to
           # 600 s, which expires while a later stage is still loading and makes the
           # per-stage grant above unreachable. vllm-omni reads these from the command
           # line only -- there is no env var, and no deploy-config key.
           "--init-timeout", "5400"]
    cfg = _deploy_config()
    if cfg is not None:
        cmd += ["--deploy-config", cfg]
    # NOTE: no --gpu-memory-utilization -- the deploy yaml owns per-stage memory;
    # a global 0.9 makes stage 0 eat the card and stage 1 OOM on startup.
    cmd += list(extra_args or [])          # Baseline.serve_command tuning flags
    # The worker process imports the codes hook by dotted path, and it is a sibling of
    # this file rather than an installed package. A child inherits the environment but
    # not this process's sys.path, so the directory is named for it explicitly.
    env = dict(os.environ)
    here = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = f"{here}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else here
    env[omni_codes_spool.SPOOL_DIR_ENV] = str(omni_codes_spool.spool_dir())
    print(f"[baseline] launching: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, env=env)


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
        if _pin_complaints:
            return Response(status_code=503, content="decode pin not honoured")
        try:
            r = await app.state.client.get(f"{base}/health", timeout=5.0)
            return {"status": "ok"} if r.status_code == 200 else Response(
                status_code=503, content="vllm-omni loading")
        except httpx.TransportError:
            return Response(status_code=503, content="vllm-omni starting")

    @app.post("/v1/audio/speech")
    async def speech(body: dict):
        global _pin_complaints
        if _pin_complaints:
            return Response(status_code=503, content="decode pin not honoured")
        text = body.get("input") or ""
        if not text.strip():
            # Readiness probe / degenerate input: fast-reject so the probe never
            # blocks on a full synthesis.
            return Response(status_code=400, content="empty input")
        # Names this request's spool file. vLLM-Omni copies extra_params into stage 0's
        # sampling params, which is where the hook reads it back; the engine's own
        # request id never reaches an HTTP caller, so the caller supplies the correlator.
        key = uuid.uuid4().hex
        body = {"model": "baseline-tts", "input": text,
                "voice": _SPEAKER, "language": _LANGUAGE, "task_type": _TASK_TYPE,
                "response_format": "wav"}
        if _CODES_ENABLED:
            body["extra_params"] = {omni_codes_spool.REQUEST_KEY_FIELD: key}
        r = await app.state.client.post(f"{base}/v1/audio/speech", json=body)
        r.raise_for_status()

        # The first completed synthesis is the earliest point at which the engine has
        # reported what it decoded under. Checked here rather than at startup because a
        # self-issued warm-up request would change the bring-up this arena measures
        # inside its own drive.
        if _CODES_ENABLED and _pin_complaints is None:
            _pin_complaints = _check_pin()
            for complaint in _pin_complaints:
                print(f"[baseline] DECODE PIN NOT HONOURED: {complaint}", flush=True)
            if _pin_complaints:
                # The whole record, once, alongside the complaints. A trip stops every
                # later request, so this is the only chance to see what the engine
                # reported, and a run that ends here has already spent its bring-up.
                print(f"[baseline] engine reported: {_observed_decode()}", flush=True)
                return Response(status_code=503, content="decode pin not honoured")
            print("[baseline] decode pin confirmed by the engine", flush=True)

        frame = json.dumps({"audio": base64.b64encode(r.content).decode("ascii")})

        async def _sse():
            # Audio first and the codes after, never the other way round: the contract
            # stops the per-request clock at the last audio frame and reads audio after
            # codes as a defect, so waiting on the spool costs no measured latency.
            yield f"data: {frame}\n\n"
            if _CODES_ENABLED:
                codes = await _read_codes(key)
                if codes is None:
                    print(f"[baseline] no codes spooled for {key}", flush=True)
                else:
                    yield f"data: {json.dumps({'codec_codes': codes})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    # parse_known_args: Baseline.serve_command tuning flags ride AFTER --host/--port
    # and belong to the INNER vLLM server -- forward them.
    args, extra = ap.parse_known_args()
    if _CODES_ENABLED:
        omni_codes_spool.purge()
    proc = _spawn_vllm_omni(_weights_source(), extra)
    try:
        uvicorn.run(_build_app(proc), host=args.host, port=args.port, log_level="warning")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
