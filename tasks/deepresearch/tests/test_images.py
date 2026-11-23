"""deepresearch's images: each builds from a recipe the task module declares, with its engine
packages at the versions its lock pins, the files it copies in at the paths the brief and the
baseline's start name, and the environment it bakes."""
from __future__ import annotations

import ast
import hashlib
import re
import shlex
from pathlib import Path

import pytest

from hls.egress.denied_packages import DENIED_PACKAGES
from hls.pipeline.commands import measured_artifact
from hls.recipes import catalog, pins
from hls.recipes.inventory import BRIEF_PACKAGES
from hls.recipes.locking.pip import denied_in
from hls.recipes.model import LayerImage, parent_of
from hls.recipes.render import (baked_environment, check_environments, check_source_builds, files_for, lock_dir,
                                recipe_key, render_def)
from hls.stages.images import image_keys
from hls.taskref import TaskRef, load_task
from tasks.deepresearch import task as t
from tests.unit.brief_task import render_brief

TASK = "deepresearch"
TASK_DIR = Path(__file__).resolve().parents[1]
#: The image of each role a run names.
SANDBOX, BASELINE, REFERENCE = "deepresearch-sandbox-agent", "deepresearch-baseline", "deepresearch-reference"
MEMBER_VLLM, MEMBER_SGLANG = "deepresearch-member-vllm", "deepresearch-member-sglang"
#: The environment the cuda base bakes, in its order: every image of the task keeps it.
_BASE = {"PATH": "/opt/venv/bin:$PATH", "VIRTUAL_ENV": "/opt/venv", "UV_CONSTRAINT": "/opt/hls/constraints.txt",
         "PIP_CONSTRAINT": "/opt/hls/constraints.txt", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1",
         "UV_NO_CACHE": "1", "VLLM_DO_NOT_TRACK": "1"}
#: What each layer adds to it, in its order. A server layer adds the GPU architecture its JIT
#: kernels compile for; the sandbox's layer does not, since the roles that compile in it set
#: their GPU's.
ADDED = {
    "deepresearch-sandbox": {"UV_TORCH_BACKEND": "cu130", "UV_HTTP_TIMEOUT": "300", "PYTHONPATH": "/app"},
    BASELINE: {"UV_TORCH_BACKEND": "cu130", "TORCH_CUDA_ARCH_LIST": "9.0", "UV_HTTP_TIMEOUT": "300",
               "PYTHONPATH": "/app"},
    REFERENCE: {"UV_TORCH_BACKEND": "cu130", "TORCH_CUDA_ARCH_LIST": "9.0", "UV_HTTP_TIMEOUT": "300"},
    MEMBER_VLLM: {"UV_TORCH_BACKEND": "cu130", "TORCH_CUDA_ARCH_LIST": "9.0"},
    MEMBER_SGLANG: {"UV_TORCH_BACKEND": "cu130", "TORCH_CUDA_ARCH_LIST": "9.0",
                    "FLASHINFER_CUBIN_DIR": "/opt/flashinfer-cubins"},
}
#: Each image's whole baked environment, in its order. The sandbox is its layer's agent cap,
#: which adds nothing.
BAKED = {SANDBOX: {**_BASE, **ADDED["deepresearch-sandbox"]},
         **{image: {**_BASE, **ADDED[image]} for image in (BASELINE, REFERENCE, MEMBER_VLLM, MEMBER_SGLANG)}}
#: The packages each lock must hold at exactly these versions: the engine of each server image
#: and what renders and scores with it, and the sandbox's torch, toolchain and libraries.
PINNED = {
    "baseline.txt": {"vllm": "0.28.0", "flashinfer-python": "0.6.16.post3", "torch": "2.13.0+cu130",
                     "torchaudio": "2.11.0+cu130", "torchvision": "0.28.0+cu130", "transformers": "5.9.0",
                     "tokenizers": "0.22.2", "accelerate": "1.15.0", "triton": "3.7.1", "numba": "0.65.0",
                     "numpy": "2.3.5", "apache-tvm-ffi": "0.1.11", "nvidia-cutlass-dsl": "4.6.2",
                     "quack-kernels": "0.6.4", "tilelang": "0.1.12", "fastapi": "0.136.3", "uvicorn": "0.53.0",
                     "pydantic": "2.13.5"},
    "reference.txt": {"torch": "2.11.0+cu130", "transformers": "5.9.0", "tokenizers": "0.22.2",
                      "accelerate": "1.14.0", "numpy": "2.5.2", "langdetect": "1.0.9", "pyarrow": "25.0.1",
                      "triton": "3.6.0", "datasets": "5.0.1"},
    "reference-setuptools.txt": {"setuptools": "81.0.0"},
    "member-vllm.txt": {"vllm": "0.22.1", "flashinfer-python": "0.6.11.post2", "flashinfer-cubin": "0.6.11.post2",
                        "apache-tvm-ffi": "0.1.9", "triton": "3.6.0", "nvidia-cutlass-dsl": "4.5.2",
                        "quack-kernels": "0.5.0", "tilelang": "0.1.9", "torch": "2.11.0+cu130",
                        "transformers": "5.9.0", "tokenizers": "0.22.2", "numba": "0.65.0", "pyarrow": "25.0.1"},
    "member-sglang.txt": {"sglang": "0.5.16", "sglang-kernel": "0.4.5", "sgl-deep-gemm": "0.1.4.post1",
                          "flashinfer-python": "0.6.14", "apache-tvm-ffi": "0.1.11", "triton": "3.6.0",
                          "nvidia-cutlass-dsl": "4.6.0", "quack-kernels": "0.6.3", "tilelang": "0.1.11",
                          "torch": "2.11.0+cu130", "transformers": "5.12.1", "tokenizers": "0.23.0rc0",
                          "numba": "0.65.1", "numpy": "2.3.5", "pyarrow": "25.0.1"},
    "sandbox.txt": {"torch": "2.13.0+cu130", "transformers": "5.9.0", "tokenizers": "0.22.2",
                    "accelerate": "1.15.0", "triton": "3.7.1", "numpy": "2.5.3", "nvidia-cutlass-dsl": "4.8.0",
                    "cuda-tile": "1.6.0", "cuda-python": "13.4.1", "cuda-bindings": "13.4.3", "cuda-core": "1.2.1",
                    "datasets": "5.0.1", "fastapi": "0.142.2", "uvicorn": "0.54.0", "pydantic": "2.13.5",
                    "pyarrow": "25.0.1", "httpx": "0.28.1", "aiofiles": "25.1.0", "pip": "26.2.1"},
}
#: How many distributions each lock pins.
PACKAGES = {"baseline.txt": 197, "reference.txt": 83, "reference-setuptools.txt": 1, "member-vllm.txt": 199,
            "member-sglang.txt": 206, "sandbox.txt": 93}
#: The sha256 of each file an image copies in from ``images/``, by the path the image holds it at.
COPIED = {
    BASELINE: {
        "/app/server.py": "fc9cc2ba99f7866a570449157857376b48eb39ce534f15ef962f9a69703beaa3",
        "/app/server_hf.py": "3785da3de65560268cd95a5e1bf78a3928cfa8ef9fc53a62849e8e0b8d24888c",
        "/app/tasks/deepresearch/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "/app/tasks/deepresearch/serving_config.py":
            "509c7f9e89bdc66bf64bb3e89388236dee4d4bf8b04abf56fa7258eb8709df24",
    },
    "deepresearch-sandbox": {
        "/opt/hls-budget/meter.py": "5e4cbfeeaaccf2db0a197fe589531684f189a6ffca813bcded80d6c91927fdf4",
        "/opt/hls-budget/server.py": "7a8c655b2013a54f8b584e5a48d43be60589df24b46400ce351ea55be03bcc12",
    },
}
#: The files the sandbox holds beside those, each a module of the task's directory.
SANDBOX_MODULES = ("__init__.py", "research_loop.py", "corpus_browser.py", "retriever.py", "limits.py", "selftest.py")


def _images() -> catalog.TaskImages:
    return catalog.task_images(TASK)


def _decl(name: str):
    [decl] = [d for d in _images().decls if d.name == name]
    return decl


def _locked(lock: str) -> dict[str, str]:
    text = (TASK_DIR / "locks" / lock).read_text()
    return dict(re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", text, re.MULTILINE))


def _copied(name: str) -> dict[str, Path]:
    """The files image *name* copies in beside its lock, by the path it holds each at."""
    decl = _decl(name)
    assert isinstance(decl, LayerImage)
    return {inside: TASK_DIR / source for inside, source in decl.files.items()}


def test_the_roles_run_on_the_images_the_module_declares():
    images = _images()
    assert (images.sandbox, images.submission_base, images.harness, images.reference) == (
        SANDBOX, SANDBOX, REFERENCE, REFERENCE)
    assert images.submission_constraints is None
    assert images.measured == {"baseline": BASELINE, "reference": REFERENCE, "member-vllm": MEMBER_VLLM,
                               "member-sglang": MEMBER_SGLANG}


def test_every_image_of_the_task_is_built_from_a_recipe():
    """The cuda base, the task's own layers and its sandbox's agent cap: each has a recipe."""
    decls = _images().decls
    assert {d.name: d.kind for d in decls} == {
        "cuda": "base", "deepresearch-sandbox": "layer", SANDBOX: "cap", BASELINE: "layer",
        "deepresearch-reference-setuptools": "layer", REFERENCE: "layer", MEMBER_VLLM: "layer",
        MEMBER_SGLANG: "layer"}


def test_every_recipe_renders_and_keys_from_the_tasks_own_locks():
    images = _images()
    check_source_builds(images.decls, pins.LOCKS_DIR)
    keys = image_keys(images.decls, recipe_key)
    for decl in images.decls:
        parent = Path("/node/parent.sif") if parent_of(decl) is not None else None
        text = render_def(decl, parent_sif=parent, locks=pins.LOCKS_DIR, uv=pins.UV)
        assert keys[decl.name].startswith(f"{decl.name}-"), decl.name
        if isinstance(decl, LayerImage) and decl.owner == TASK:
            assert lock_dir(decl, pins.LOCKS_DIR) == TASK_DIR / "locks"
            assert f"-r /opt/hls/locks/{decl.pip_lock}" in text, decl.name
    owned = {d.pip_lock for d in images.decls if isinstance(d, LayerImage) and d.owner == TASK}
    assert owned == set(PINNED)


@pytest.mark.parametrize("lock", sorted(PINNED))
def test_a_lock_pins_its_engine_and_toolchain_versions(lock):
    locked = _locked(lock)
    assert {name: locked.get(name) for name in PINNED[lock]} == PINNED[lock]
    assert len(locked) == PACKAGES[lock]


def test_a_lock_names_how_it_is_compiled():
    """Each lock's first line is the command that makes it again, a server layer's with --server."""
    for decl in _images().decls:
        if not isinstance(decl, LayerImage) or decl.owner != TASK:
            continue
        first = (TASK_DIR / "locks" / decl.pip_lock).read_text().splitlines()[0]
        name = decl.pip_lock.removesuffix(".txt")
        assert first.startswith(f"# Generated by: hls images lock-pip layer {name} --task {TASK}"), first
        assert first.endswith(" --server") == decl.server, first


def test_the_reference_builds_langdetect_with_the_setuptools_below_it():
    reference = _decl(REFERENCE)
    assert reference.source_builds == {"langdetect==1.0.9": ["setuptools"]}
    backend = _decl(reference.parent)
    assert (backend.owner, backend.server, backend.parent) == (TASK, True, "cuda")
    assert _locked(backend.pip_lock) == {"setuptools": "81.0.0"}
    text = render_def(reference, parent_sif=Path("/node/parent.sif"), locks=pins.LOCKS_DIR, uv=pins.UV)
    assert "--no-binary langdetect --no-build-isolation-package langdetect" in text


def test_the_sglang_member_holds_flashinfers_kernels_and_fetches_none_at_run_time():
    member = _decl(MEMBER_SGLANG)
    assert member.fetched == catalog.FLASHINFER_CUBINS["0.6.14"]
    assert BAKED[MEMBER_SGLANG][member.fetched.variable] == member.fetched.into
    locked = _locked(member.pip_lock)
    assert locked["flashinfer-python"] == "0.6.14" and "flashinfer-cubin" not in locked


@pytest.mark.parametrize("image", sorted(BAKED))
def test_an_image_bakes_its_environment(image):
    decls = _images().decls
    assert list(baked_environment(image, decls).items()) == list(BAKED[image].items())


@pytest.mark.parametrize("image", sorted(ADDED))
def test_a_layer_writes_what_it_adds_to_the_environment_below(image):
    text = render_def(_decl(image), parent_sif=Path("/node/parent.sif"), locks=pins.LOCKS_DIR, uv=pins.UV)
    section = text.split("%environment\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == [f"    export {k}={v}" for k, v in ADDED[image].items()]


def test_the_layers_only_add_to_the_environment_below():
    """No layer repeats a value the image below gives, and the sandbox's layer sets none of the
    base's variables nor the GPU architecture, which the roles that compile in it set."""
    check_environments(_images().decls)
    assert not set(ADDED["deepresearch-sandbox"]) & (set(_BASE) | {"TORCH_CUDA_ARCH_LIST"})
    assert all(not set(added) & set(_BASE) for added in ADDED.values())


# the baseline

def test_the_baseline_is_served_with_the_start_the_module_declares():
    loaded = load_task(TaskRef(module=catalog.task_module(TASK)))
    artifact, serve = measured_artifact(loaded, "baseline", TASK)
    declared = t.IMAGES.baseline
    assert artifact.image == BASELINE == declared.image.name
    assert (artifact.entrypoint, artifact.runtime_env) == (declared.entrypoint, declared.runtime_env)
    assert (artifact.entrypoint, artifact.runtime_env) == (["python", "/app/server.py"], {})
    assert serve == [t.CHAT_TEMPLATE_KWARGS_FLAG, '{"enable_thinking": true}']
    # The entrypoint's file is one the image holds, and the flag one that file takes.
    server = _copied(BASELINE)[artifact.entrypoint[1]]
    assert "ap.add_argument(CHAT_TEMPLATE_KWARGS_FLAG" in server.read_text()
    config = _copied(BASELINE)["/app/tasks/deepresearch/serving_config.py"].read_text()
    assert f'CHAT_TEMPLATE_KWARGS_FLAG = "{t.CHAT_TEMPLATE_KWARGS_FLAG}"' in config


def _imports_from(path: Path, module: str) -> set[str]:
    return {alias.name for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ImportFrom) and node.module == module for alias in node.names}


def _defined(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {target.id for target in node.targets if isinstance(target, ast.Name)}
    return names


def test_the_baselines_servers_import_what_the_image_holds_beside_them():
    """The servers run from /app with /app on PYTHONPATH, so ``tasks.deepresearch.serving_config``
    is the file the image holds at that path, and every name they import from it is one it has."""
    copied = _copied(BASELINE)
    assert set(copied) == set(COPIED[BASELINE])
    assert BAKED[BASELINE]["PYTHONPATH"] == "/app"
    defined = _defined(copied["/app/tasks/deepresearch/serving_config.py"])
    for server in ("/app/server.py", "/app/server_hf.py"):
        wanted = _imports_from(copied[server], "tasks.deepresearch.serving_config")
        assert wanted and wanted <= defined, (server, sorted(wanted - defined))
        every = {node.module for node in ast.walk(ast.parse(copied[server].read_text()))
                 if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tasks.")}
        assert every == {"tasks.deepresearch.serving_config"}, server


@pytest.mark.parametrize("image", sorted(COPIED))
def test_the_files_copied_in_from_images_are_the_pinned_ones(image):
    """``images/<image>/`` mirrors the image's tree, and each file there has the bytes the recipe
    was written for: a change to one is a change to what the image runs."""
    copied = {inside: host for inside, host in _copied(image).items() if "images" in host.parts}
    assert {inside: host.relative_to(TASK_DIR).as_posix() for inside, host in copied.items()} == {
        inside: f"images/{image.removeprefix('deepresearch-')}{inside}" for inside in COPIED[image]}
    assert {inside: hashlib.sha256(host.read_bytes()).hexdigest() for inside, host in copied.items()} == COPIED[image]
    held = sorted(p.relative_to(TASK_DIR / "images").as_posix()
                  for p in (TASK_DIR / "images" / image.removeprefix("deepresearch-")).rglob("*") if p.is_file())
    assert held == sorted(f"{image.removeprefix('deepresearch-')}{inside}" for inside in COPIED[image])


# the sandbox

def test_the_sandbox_holds_the_loop_and_the_self_test_where_the_brief_says():
    assert t.SANDBOX_FILES == SANDBOX_MODULES and t.SANDBOX_DIR == "/app/tasks/deepresearch"
    copied = _copied("deepresearch-sandbox")
    assert {inside: host for inside, host in copied.items() if inside.startswith("/app/")} == {
        f"/app/tasks/deepresearch/{name}": TASK_DIR / name for name in SANDBOX_MODULES}
    assert set(copied) == {f"/app/tasks/deepresearch/{name}" for name in SANDBOX_MODULES} | set(
        COPIED["deepresearch-sandbox"])
    # The agent cap copies in nothing of the task's, so the sandbox holds exactly those.
    cap = _decl(SANDBOX)
    assert parent_of(cap) == "deepresearch-sandbox"
    assert all(host.parent == pins.LOCKS_DIR for host in files_for(cap, pins.LOCKS_DIR).values())
    brief, _ = render_brief(t.TASK, t.ARTIFACT.brief)
    assert "installed at\n`/app/tasks/deepresearch/`" in brief
    [command] = [line.strip() for line in brief.splitlines() if "selftest.py" in line and "python " in line]
    assert shlex.split(command)[:2] == ["python", "/app/tasks/deepresearch/selftest.py"]
    assert shlex.split(command)[1] in copied


def test_the_sandbox_lock_holds_what_the_brief_lists_and_nothing_the_deny_list_names():
    text = (TASK_DIR / "locks" / "sandbox.txt").read_text()
    locked = {re.sub(r"[-_.]+", "-", name).lower() for name in _locked("sandbox.txt")}
    assert not denied_in(text) and DENIED_PACKAGES
    assert [name for name in BRIEF_PACKAGES if name not in locked] == []
    # What the self-test and the retriever it loads import.
    assert {"httpx", "numpy", "torch", "transformers", "pyarrow"} <= locked
    sandbox = _decl("deepresearch-sandbox")
    assert (sandbox.server, sandbox.parent, sandbox.apt_lock) == (False, "cuda", None)
