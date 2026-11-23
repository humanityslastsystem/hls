"""multimodal_omni's images build from recipes its module declares: the baseline is vLLM and
vLLM-Omni behind the adapter the image copies in, with two packages built from their pinned source
distributions and the marker file vLLM-Omni's workers read."""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from hls.pipeline.commands import measured_artifact
from hls.recipes import catalog, pins
from hls.recipes.model import LayerImage, pip_name
from hls.recipes.render import (_pip_lock, baked_environment, check_source_builds, files_for, lock_dir, recipe_key,
                                render_def)
from hls.stages.images import image_keys
from hls.taskref import TaskRef, load_task

TASK = "multimodal_omni"
TASK_DIR = Path(__file__).resolve().parents[2] / "tasks" / TASK
#: The engine packages of the baseline at the versions the measured image holds: the serving pair,
#: the torch line vLLM 0.28.0 brings, the kernel and decoding libraries under them, and the adapter's
#: own three.
ENGINE = {"vllm": "0.28.0", "vllm-omni": "0.28.0", "openai-whisper": "20250625", "torch": "2.13.0+cu130",
          "torchaudio": "2.11.0+cu130", "torchvision": "0.28.0+cu130", "torchcodec": "0.16.0+cu130",
          "transformers": "5.14.1", "tokenizers": "0.22.2", "accelerate": "1.12.0", "diffusers": "0.40.0",
          "flashinfer-python": "0.6.16.post3", "triton": "3.7.1", "numba": "0.65.0", "numpy": "2.3.5",
          "apache-tvm-ffi": "0.1.11", "nvidia-cutlass-dsl": "4.6.2", "quack-kernels": "0.6.4", "tilelang": "0.1.12",
          "onnxruntime": "1.30.0", "xgrammar": "0.2.7", "compressed-tensors": "0.17.0", "safetensors": "0.8.0",
          "huggingface-hub": "1.32.0", "fastapi": "0.136.3", "uvicorn": "0.53.0", "httpx": "0.28.1"}
#: The packages published as source distributions only, each to the sha256 of the one file its
#: release holds: ``openai_whisper-20250625.tar.gz`` and ``antlr4-python3-runtime-4.9.3.tar.gz``.
SOURCE_ONLY = {
    "openai-whisper==20250625": "37a91a3921809d9f44748ffc73c0a55c9f366c85a3ef5c2ae0cc09540432eb96",
    "antlr4-python3-runtime==4.9.3": "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b",
}
#: How many distributions the measured image's venv holds: the lock's and the setuptools below it.
MEASURED_DISTRIBUTIONS = 241
# httpx's client class, which the adapter's tests stand a recording transport in for.
_ASYNC_CLIENT = httpx.AsyncClient


def _images():
    return catalog.task_images(TASK)


def _baseline() -> LayerImage:
    images = _images()
    decl = {d.name: d for d in images.decls}[images.measured["baseline"]]
    assert isinstance(decl, LayerImage)
    return decl


def _lock_text() -> str:
    decl = _baseline()
    return (lock_dir(decl, pins.LOCKS_DIR) / decl.pip_lock).read_text()


def test_the_task_declares_its_baseline_and_reference():
    images = _images()
    assert set(images.measured) == {"baseline", "reference"}


def test_every_image_of_the_task_renders_and_has_a_key():
    images = _images()
    keys = image_keys(images.decls, recipe_key)
    assert len(set(keys.values())) == len(keys)
    for decl in images.decls:
        parent = getattr(decl, "parent", None)
        text = render_def(decl, parent_sif=Path("/node/parent.sif") if parent else None, locks=pins.LOCKS_DIR,
                          uv=pins.UV)
        assert text.startswith("Bootstrap: localimage" if parent else "Bootstrap: docker"), decl.name
        assert keys[decl.name] == recipe_key(decl, parent_key=keys.get(parent), locks_dir=pins.LOCKS_DIR,
                                             uv=pins.UV)


def test_the_baseline_is_a_server_layer_of_the_task_on_the_setuptools_layer():
    decl = _baseline()
    assert (decl.name, decl.parent, decl.owner, decl.server) == ("multimodal-omni-baseline", "cuda-setuptools",
                                                                 TASK, True)
    below = {d.name: d for d in _images().decls}[decl.parent]
    assert isinstance(below, LayerImage) and below.parent == "cuda"
    # The layer below holds the build backend and nothing else.
    assert set(_pip_lock(lock_dir(below, pins.LOCKS_DIR) / below.pip_lock)) == {"setuptools"}


def test_the_baseline_lock_holds_the_engine_versions():
    locked = _pip_lock(TASK_DIR / "locks" / "baseline.txt")
    assert {name: locked.get(name) for name in ENGINE} == ENGINE
    # The pair is pinned together: vLLM-Omni declares no vLLM requirement of its own.
    assert locked["vllm-omni"] == locked["vllm"]


def test_the_lock_and_the_layer_below_hold_every_distribution_of_the_measured_image():
    decl = _baseline()
    below = {d.name: d for d in _images().decls}[decl.parent]
    locked = _pip_lock(TASK_DIR / "locks" / "baseline.txt")
    held = locked | _pip_lock(lock_dir(below, pins.LOCKS_DIR) / below.pip_lock)
    assert (len(locked), len(held)) == (MEASURED_DISTRIBUTIONS - 1, MEASURED_DISTRIBUTIONS)
    # The venv's own installer and the build backend, at the versions the measured venv holds.
    assert (held["pip"], held["setuptools"]) == ("26.2.1", "80.10.2")
    # The lock's input names every one of them at its version, so nothing is left to resolve.
    named = re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$", (TASK_DIR / "locks" / "baseline.in").read_text(),
                       re.MULTILINE)
    assert {pip_name(f"{name}=={version}"): version for name, version in named} == locked


def test_the_two_source_only_packages_build_from_their_pinned_source_distributions():
    decl = _baseline()
    assert decl.source_builds == {pin: ["setuptools"] for pin in SOURCE_ONLY}
    text = _lock_text()
    for pin, sha256 in SOURCE_ONLY.items():
        entry = re.search(rf"^{re.escape(pin)} \\\n((?:    --hash=sha256:[0-9a-f]{{64}}(?: \\)?\n)+)", text,
                          re.MULTILINE)
        assert entry is not None, pin
        # One file, the source distribution: no wheel of the release exists to install instead.
        assert re.findall(r"sha256:([0-9a-f]{64})", entry.group(1)) == [sha256], pin
    # The lock was resolved the way the image installs it.
    named = re.findall(r"--source-build (\S+)", text.splitlines()[0])
    assert named == sorted(pin.split("==", 1)[0] for pin in decl.source_builds)
    post = render_def(decl, parent_sif=Path("/node/parent.sif"), locks=pins.LOCKS_DIR, uv=pins.UV)
    install = next(line for line in post.splitlines() if "uv pip install" in line)
    assert "--require-hashes --no-deps --only-binary :all:" in install
    for name in ("antlr4-python3-runtime", "openai-whisper"):
        assert f"--no-binary {name} --no-build-isolation-package {name}" in install


def test_the_source_builds_find_their_backend_in_the_layer_below():
    """They build without isolation, so setuptools is installed, from its hashed lock, before the
    baseline's own lock is; that lock does not install it again."""
    check_source_builds(_images().decls, pins.LOCKS_DIR)
    assert "setuptools" not in _pip_lock(TASK_DIR / "locks" / "baseline.txt")


def test_the_baseline_holds_its_server_and_the_marker_vllm_omni_reads():
    decl = _baseline()
    assert decl.files == {"/app/server_vllm_omni.py": "server_vllm_omni.py", "/.dockerenv": "dockerenv"}
    copied = files_for(decl, pins.LOCKS_DIR)
    assert copied["/app/server_vllm_omni.py"] == TASK_DIR / "server_vllm_omni.py"
    # vLLM-Omni tests that the file exists, nothing more.
    assert copied["/.dockerenv"].read_bytes() == b""
    text = render_def(decl, parent_sif=Path("/node/parent.sif"), locks=pins.LOCKS_DIR, uv=pins.UV)
    files = text.split("%files\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert f"    {TASK_DIR / 'dockerenv'} /.dockerenv" in files
    assert f"    {TASK_DIR / 'server_vllm_omni.py'} /app/server_vllm_omni.py" in files


def test_the_baseline_bakes_the_environment_the_measured_image_bakes():
    """The measured image's variables: the venv, uv's torch backend, the GPU architecture, the
    hub, output and cache settings, and vLLM's usage reporting turned off; with the shared base's
    constraint variables, which name its empty constraints file. The layer itself sets only what
    the base does not give."""
    decl = _baseline()
    baked = {"PATH": "/opt/venv/bin:$PATH", "VIRTUAL_ENV": "/opt/venv", "UV_TORCH_BACKEND": "cu130",
             "TORCH_CUDA_ARCH_LIST": "9.0", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1",
             "UV_NO_CACHE": "1", "VLLM_DO_NOT_TRACK": "1"}
    shared = {"UV_CONSTRAINT": "/opt/hls/constraints.txt", "PIP_CONSTRAINT": "/opt/hls/constraints.txt"}
    assert baked_environment(decl.name, _images().decls) == {**baked, **shared}
    text = render_def(decl, parent_sif=Path("/node/parent.sif"), locks=pins.LOCKS_DIR, uv=pins.UV)
    section = text.split("%environment\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == ["    export UV_TORCH_BACKEND=cu130", "    export TORCH_CUDA_ARCH_LIST=9.0"]


def test_the_baseline_starts_the_server_its_image_holds():
    served = _images().baseline
    assert served is not None
    assert (served.entrypoint, served.runtime_env) == (["python", "/app/server_vllm_omni.py"], {})
    assert served.entrypoint[1] in served.image.files
    loaded = load_task(TaskRef(module=catalog.task_module(TASK)))
    artifact, serve = measured_artifact(loaded, "baseline", TASK)
    assert artifact.image == "multimodal-omni-baseline"
    assert (artifact.entrypoint, artifact.runtime_env) == (served.entrypoint, served.runtime_env)
    assert serve == list(loaded.baseline.serve_command) == []


def _server(monkeypatch, override: str | None = None):
    """The baseline's server module, imported with its inner port given and the model override
    variable set to *override*, or unset."""
    monkeypatch.setenv("HLS_VLLM_INNER_PORT", "18431")
    if override is None:
        monkeypatch.delenv("HLS_BASELINE_MODEL", raising=False)
    else:
        monkeypatch.setenv("HLS_BASELINE_MODEL", override)
    name = f"tasks.{TASK}.server_vllm_omni"
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


@pytest.fixture
def server(monkeypatch):
    yield _server(monkeypatch)
    sys.modules.pop(f"tasks.{TASK}.server_vllm_omni", None)


def test_the_server_starts_vllm_omni_with_the_flags_the_baseline_was_measured_with(server, monkeypatch):
    started: list[list[str]] = []
    monkeypatch.setattr(server.shutil, "which", lambda exe: f"/opt/venv/bin/{exe}")
    monkeypatch.setattr(server, "_deploy_config", lambda: "/opt/venv/deploy/qwen3_omni_moe.yaml")
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd: started.append(cmd))
    server._spawn_vllm_omni("/weights", ["--extra", "1"])
    assert started == [[
        "/opt/venv/bin/vllm-omni", "serve", "/weights", "--served-model-name", "baseline-omni",
        "--host", "127.0.0.1", "--port", "18431", "--trust-remote-code", "--omni",
        "--stage-init-timeout", "1800", "--init-timeout", "5400",
        "--deploy-config", "/opt/venv/deploy/qwen3_omni_moe.yaml", "--extra", "1"]]


def test_the_server_passes_the_deploy_config_vllm_omni_packages(server, monkeypatch, tmp_path):
    package = tmp_path / "vllm_omni"
    (package / "deploy").mkdir(parents=True)
    (package / "deploy" / "qwen3_omni_moe.yaml").write_text("stages: []\n")
    monkeypatch.setitem(sys.modules, "vllm_omni", SimpleNamespace(__file__=str(package / "__init__.py")))
    assert server._deploy_config() == str(package / "deploy" / "qwen3_omni_moe.yaml")
    (package / "deploy" / "qwen3_omni_moe.yaml").unlink()
    assert server._deploy_config() is None


def test_the_server_serves_the_tasks_model_from_the_provisioned_weights(server, monkeypatch, tmp_path):
    task = importlib.import_module(catalog.task_module(TASK))
    assert server.MODEL_ID == task.MODEL_ID
    monkeypatch.setenv("HLS_WEIGHTS_DIR", str(tmp_path))
    assert server._weights_source() == str(tmp_path)
    monkeypatch.delenv("HLS_WEIGHTS_DIR")
    assert server._weights_source() == task.MODEL_ID


def test_the_model_override_names_what_is_served_when_no_weights_are_mounted(monkeypatch):
    server = _server(monkeypatch, override="some/other-model")
    monkeypatch.delenv("HLS_WEIGHTS_DIR", raising=False)
    assert server._weights_source() == server.MODEL_ID == "some/other-model"
    sys.modules.pop(server.__name__, None)


def test_the_adapter_keeps_one_client_with_the_limits_the_baseline_ran_with(server):
    assert (server._LIMITS.max_connections, server._LIMITS.max_keepalive_connections,
            server._LIMITS.keepalive_expiry) == (1024, 512, 2.0)


def _inner(server, monkeypatch, status: int = 200):
    """Stand in for the inner vLLM-Omni server: return the requests the adapter sends it and the
    arguments it makes its client with."""
    seen: list[tuple[str, str, object]] = []
    made: list[dict] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url), json.loads(request.content) if request.content else None))
        return httpx.Response(status, json={"choices": [{"message": {"content": "ok"}}]})

    def client(**kwargs):
        made.append(kwargs)
        return _ASYNC_CLIENT(transport=httpx.MockTransport(answer), **kwargs)

    monkeypatch.setattr(server.httpx, "AsyncClient", client)
    return seen, made


def test_the_adapter_pins_the_model_name_and_text_only_output_on_every_request(server, monkeypatch):
    seen, made = _inner(server, monkeypatch)
    messages = [{"role": "user", "content": "hi"}]
    with TestClient(server._build_app(SimpleNamespace(poll=lambda: None))) as client:
        reply = client.post("/v1/chat/completions", json={"model": "anything", "modalities": ["text", "audio"],
                                                          "max_tokens": 8, "messages": messages})
    assert (reply.status_code, reply.json()) == (200, {"choices": [{"message": {"content": "ok"}}]})
    assert seen == [("POST", "http://127.0.0.1:18431/v1/chat/completions",
                     {"model": "baseline-omni", "modalities": ["text"], "max_tokens": 8, "messages": messages})]
    assert made == [{"timeout": 600.0, "limits": server._LIMITS}]


def test_the_adapter_returns_the_inner_servers_status(server, monkeypatch):
    _inner(server, monkeypatch, status=422)
    with TestClient(server._build_app(SimpleNamespace(poll=lambda: None))) as client:
        assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 422


def test_health_is_the_inner_servers_and_fails_once_it_has_exited(server, monkeypatch):
    seen, _ = _inner(server, monkeypatch)
    with TestClient(server._build_app(SimpleNamespace(poll=lambda: None))) as client:
        assert (client.get("/health").status_code, client.get("/health").json()) == (200, {"status": "ok"})
    assert seen[0] == ("GET", "http://127.0.0.1:18431/health", None)
    with TestClient(server._build_app(SimpleNamespace(poll=lambda: 1))) as client:
        exited = client.get("/health")
    assert (exited.status_code, exited.text) == (503, "vllm-omni exited")
    _inner(server, monkeypatch, status=503)
    with TestClient(server._build_app(SimpleNamespace(poll=lambda: None))) as client:
        loading = client.get("/health")
    assert (loading.status_code, loading.text) == (503, "vllm-omni loading")


def test_main_serves_on_the_given_port_and_forwards_every_other_flag_to_vllm_omni(server, monkeypatch):
    spawned: list[tuple[str, list[str]]] = []
    ran: list[dict] = []
    proc = SimpleNamespace(terminated=0)
    proc.terminate = lambda: setattr(proc, "terminated", proc.terminated + 1)
    monkeypatch.setattr(server, "_weights_source", lambda: "/weights")
    monkeypatch.setattr(server, "_spawn_vllm_omni", lambda src, extra: spawned.append((src, extra)) or proc)
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: ran.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["server_vllm_omni.py", "--host", "0.0.0.0", "--port", "8000", "--seed", "7"])
    server.main()
    assert spawned == [("/weights", ["--seed", "7"])]
    assert ran == [{"host": "0.0.0.0", "port": 8000, "log_level": "warning"}]
    assert proc.terminated == 1
