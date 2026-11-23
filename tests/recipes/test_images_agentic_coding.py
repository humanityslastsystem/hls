"""agentic_coding's images: the sandbox on its own layer, the baseline and the reference on the
vendor's vLLM image, the SGLang calibration member, and what each lock pins."""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from hls.recipes import pins
from hls.recipes.catalog import BASE_ENV, FLASHINFER_CUBINS, SERVER_ENV, task_images
from hls.recipes.inventory import BRIEF_PACKAGES
from hls.recipes.locking.pip import denied_in, pins_of
from hls.recipes.model import (BASE_ENVIRONMENT, BaseImage, CapImage, LayerImage, RegistryImage, parent_of,
                               registry_of)
from hls.recipes.render import TASKS_DIR, baked_environment, render_def
from hls.stages.deploy_drive import CHECKOUT, HARBOR_SRC

TASK = "agentic_coding"
LOCKS = TASKS_DIR / TASK / "locks"
REPO = TASKS_DIR.parent
#: The vendor's image of the vLLM build the brief names as the reference.
VLLM = "vllm/vllm-openai@sha256:4cbfd34aac145fd1870381c030131c7f868fcad45448f401ecdb5fd4ed020b42"
SANDBOX_LAYER = "agentic-coding-tf-5-17-0"
SANDBOX = f"{SANDBOX_LAYER}-agent"
MEMBER = "agentic-coding-member-sglang"
#: What each image's own %environment adds to the environment of the image below it.
ADDED = {
    SANDBOX_LAYER: {"UV_TORCH_BACKEND": "cu130"},
    "agentic-coding-baseline": {"PYTHONPATH": "/app"},
    "agentic-coding-reference": {"UV_HTTP_TIMEOUT": "300"},
    MEMBER: {"UV_TORCH_BACKEND": "cu130", "TORCH_CUDA_ARCH_LIST": "9.0",
             "FLASHINFER_CUBIN_DIR": "/opt/flashinfer-cubins"},
}
#: The cuda base's own baked variables, which every image on it keeps.
CUDA_ENV = {**BASE_ENVIRONMENT, **BASE_ENV}
#: The engine packages of the SGLang member, at the versions its lock holds.
SGLANG_PINS = {
    "sglang": "0.5.20", "sglang-kernel": "0.4.7", "sgl-deep-gemm": "0.2.0", "sgl-deep-ep": "0.1.2",
    "flashinfer-python": "0.6.18", "apache-tvm-ffi": "0.1.11", "triton": "3.7.1", "nvidia-cutlass-dsl": "4.6.2",
    "quack-kernels": "0.6.4", "tilelang": "0.1.12", "flash-attn-4": "4.0.0b19", "cuda-tile": "1.6.0rc5",
    "torch": "2.13.0+cu130", "torchaudio": "2.11.0+cu130", "transformers": "5.12.1", "tokenizers": "0.22.2",
    "numba": "0.65.1", "numpy": "2.3.5", "xgrammar": "0.2.1", "openai": "2.54.0", "harbor": "0.8.0",
    "litellm": "1.102.0", "datasets": "5.0.1",
}
MEMBER_APT = ["fakeroot=1.33-1", "libfakeroot=1.33-1", "libfuse3-3=3.14.0-5build1", "liblzo2-2=2.10-2build4",
              "libprotobuf-c1=1.4.1-1ubuntu4", "libsubid4=1:4.13+dfsg1-4ubuntu3.2", "libtalloc2=2.4.2-1build2",
              "uidmap=1:4.13+dfsg1-4ubuntu3.2"]
#: The sandbox's packages the brief names or its statements rest on, at their versions.
SANDBOX_PINS = {
    "torch": "2.11.0+cu130", "transformers": "5.17.0", "tokenizers": "0.23.2", "accelerate": "1.15.0",
    "datasets": "5.0.1", "fastapi": "0.141.1", "uvicorn": "0.53.0", "pydantic": "2.13.5", "triton": "3.6.0",
    "cuda-tile": "1.6.0", "nvidia-cutlass-dsl": "4.8.0", "cuda-python": "13.4.1", "cuda-bindings": "13.4.3",
    "cuda-core": "1.2.0", "nvidia-cudnn-cu13": "9.19.0.56", "nvidia-nccl-cu13": "2.28.9", "numpy": "2.5.3",
    "pip": "26.2.1", "setuptools": "81.0.0",
}
#: What the reference adds to the vendor image, or holds at another version than it does.
REFERENCE_PINS = {
    "harbor": "0.8.0", "claude-agent-sdk": "0.2.159", "datasets": "5.0.1", "deprecation": "2.1.0",
    "dirhash": "0.5.0", "fastuuid": "0.14.0", "fsspec": "2026.6.0", "h2": "4.4.1", "hpack": "4.2.0",
    "hyperframe": "6.1.0", "litellm": "1.102.1", "multiprocess": "0.70.19", "openai": "2.54.0", "pandas": "3.0.6",
    "pathspec": "1.1.1", "postgrest": "2.31.0", "pyarrow": "25.0.1", "realtime": "2.31.0", "ruff": "0.16.8",
    "scantree": "0.0.4", "shortuuid": "1.0.13", "storage3": "2.31.0", "strenum": "0.4.15", "supabase": "2.31.0",
    "supabase-auth": "2.31.0", "supabase-functions": "2.31.0", "tenacity": "9.1.4", "toml": "0.10.2",
    "websockets": "15.0.1", "xxhash": "4.0.1",
}
REFERENCE_APT = ["python-is-python3=3.11.4-1", "ca-certificates=20260601~24.04.1", "fakeroot=1.33-1",
                 "libfakeroot=1.33-1", "libfuse3-3=3.14.0-5build1", "liblzo2-2=2.10-2build4",
                 "libprotobuf-c1=1.4.1-1ubuntu4", "libsubid4=1:4.13+dfsg1-4ubuntu3.2", "libtalloc2=2.4.2-1build2",
                 "uidmap=1:4.13+dfsg1-4ubuntu3.2"]
APPTAINER = ("https://github.com/apptainer/apptainer/releases/download/v1.5.3/apptainer_1.5.3_amd64.deb",
             "82b0bdddf459087d202383360b8318d526ad6826c748a2f669913cc6aef9ee40")


def _decls() -> dict[str, object]:
    return {d.name: d for d in task_images(TASK).decls}


def _lines(lock: str) -> list[str]:
    return [s for line in (LOCKS / lock).read_text().splitlines() if (s := line.strip()) and not s.startswith("#")]


def test_the_sandbox_is_the_agent_cap_on_the_tasks_own_layer_on_the_cuda_base():
    images, decls = task_images(TASK), _decls()
    assert images.sandbox == SANDBOX and isinstance(decls[SANDBOX], CapImage)
    layer = decls[SANDBOX_LAYER]
    assert isinstance(layer, LayerImage) and not layer.server and layer.owner == TASK
    assert parent_of(decls[SANDBOX]) == SANDBOX_LAYER and parent_of(layer) == "cuda"
    assert isinstance(decls["cuda"], BaseImage) and decls["cuda"].pip_lock is None


def test_a_submission_is_packaged_on_the_sandbox_with_no_constraints_file():
    images = task_images(TASK)
    assert images.submission_base == images.sandbox == SANDBOX
    # The cuda base holds no torch, so there is nothing to hold a package's install to.
    assert images.submission_constraints is None


def test_harness_roles_run_on_the_reference_image():
    images = task_images(TASK)
    assert images.harness == images.reference == images.measured["reference"] == "agentic-coding-reference"


def test_the_baseline_and_the_reference_build_on_the_vendors_image_by_digest():
    decls = _decls()
    vendor = decls["agentic-coding-vllm"]
    assert isinstance(vendor, RegistryImage) and vendor.ref == VLLM
    assert (vendor.python, vendor.snapshot) == ("/usr/bin/python3", pins.SNAPSHOT)
    for name in ("agentic-coding-baseline", "agentic-coding-reference"):
        layer = decls[name]
        assert isinstance(layer, LayerImage) and layer.server and layer.owner == TASK
        assert registry_of(layer, decls) == vendor and parent_of(layer) == vendor.name


def test_the_baseline_adds_the_server_and_installs_nothing():
    baseline = _decls()["agentic-coding-baseline"]
    assert (baseline.pip_lock, baseline.apt_lock, baseline.debs) == (None, None, {})
    assert baseline.files == {"/app/server.py": "vllm_server.py",
                              "/app/tasks/agentic_coding/__init__.py": "__init__.py",
                              "/app/tasks/agentic_coding/serving_config.py": "serving_config.py"}


def test_the_server_imports_its_configuration_from_where_the_image_holds_it():
    """The server starts as ``/app/server.py`` with ``/app`` on its path, and its one import of
    the task's code names the module the image holds beside it."""
    baseline = _decls()["agentic-coding-baseline"]
    tree = ast.parse((TASKS_DIR / TASK / baseline.files["/app/server.py"]).read_text())
    task_imports = {node.module for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tasks.")}
    task_imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                     for alias in node.names if alias.name.startswith("tasks.")}
    assert task_imports == {"tasks.agentic_coding.serving_config"}
    for module in task_imports:
        path = "/app/" + module.replace(".", "/")
        assert f"{path}.py" in baseline.files and f"{path.rsplit('/', 1)[0]}/__init__.py" in baseline.files
    served = task_images(TASK).baseline
    assert served.entrypoint == ["python3", "/app/server.py"] and served.runtime_env == {"PYTHONPATH": "/app"}
    # The configuration module itself imports nothing of the checkout.
    config = ast.parse((TASKS_DIR / TASK / "serving_config.py").read_text())
    assert not [n for n in ast.walk(config) if isinstance(n, ast.ImportFrom) and (n.module or "").startswith(
        ("tasks", "hls"))]


def test_the_sandbox_lock_pins_the_toolchain_the_brief_states():
    locked = pins_of((LOCKS / "sandbox.txt").read_text())
    assert {name: locked.get(name) for name in SANDBOX_PINS} == SANDBOX_PINS
    assert len(locked) == 92
    # Every package the brief says is installed, and none the deny list names.
    assert not [name for name in BRIEF_PACKAGES if name not in locked]
    assert denied_in((LOCKS / "sandbox.txt").read_text()) == []
    # Its input names each of them at one version.
    assert sorted(_lines("sandbox.in")) == sorted(f"{name}=={version}" for name, version in locked.items())


def test_the_reference_lock_is_what_the_reference_adds_to_the_vendor_image():
    text = (LOCKS / "reference.txt").read_text()
    assert pins_of(text) == REFERENCE_PINS
    assert text.splitlines()[0] == ("# Generated by: hls images lock-pip layer reference --task agentic_coding "
                                    "--server --no-deps")
    # It moves nothing of the engine the vendor image serves with.
    assert not {"vllm", "torch", "transformers", "tokenizers", "flashinfer-python", "triton", "numpy", "numba",
                "pydantic", "fastapi"} & set(pins_of(text))


def test_the_reference_holds_apptainer_and_what_its_deb_depends_on():
    reference = _decls()["agentic-coding-reference"]
    assert (reference.apt_lock, reference.pip_lock) == ("apt-reference.txt", "reference.txt")
    assert reference.debs == dict([APPTAINER]) and not reference.files
    assert _lines("apt-reference.txt") == REFERENCE_APT
    assert (LOCKS / "apt-reference.txt").read_text().splitlines()[:2] == [
        "# Generated by: hls images lock-apt ubuntu apt-reference --task agentic_coding",
        f"# From snapshot {pins.SNAPSHOT}"]


@pytest.mark.parametrize("image", sorted(ADDED))
def test_an_images_own_environment_adds_what_the_image_below_it_lacks(image):
    decls = task_images(TASK).decls
    by_name = {d.name: d for d in decls}
    registry = registry_of(by_name[image], by_name)
    # A chain on the vendor image bakes its layer's variables alone; one on the cuda base keeps
    # the base's and adds.
    below = {} if registry is not None else CUDA_ENV
    assert baked_environment(image, decls) == {**below, **ADDED[image]}
    text = render_def(by_name[image], parent_sif=Path("/node/parent.sif"), locks=pins.LOCKS_DIR, uv=pins.UV,
                      registry=registry)
    section = text.split("%environment\n", 1)[1].split("\n\n", 1)[0]
    assert section.splitlines() == [f"    export {k}={v}" for k, v in ADDED[image].items()]


def test_the_sandbox_keeps_the_bases_variables_and_adds_uvs_torch_backend():
    baked = baked_environment(SANDBOX, task_images(TASK).decls)
    assert baked == {**CUDA_ENV, "UV_TORCH_BACKEND": "cu130"}
    # The roles that compile in it set the GPU architecture; the image bakes none.
    assert "TORCH_CUDA_ARCH_LIST" not in baked


def test_no_image_of_the_task_is_imported():
    kinds = {d.name: d.kind for d in task_images(TASK).decls}
    assert kinds == {"cuda": "base", SANDBOX_LAYER: "layer", SANDBOX: "cap", "agentic-coding-vllm": "registry",
                     "agentic-coding-baseline": "layer", "agentic-coding-reference": "layer", MEMBER: "layer"}


def test_the_sglang_member_serves_with_sglang_and_drives_its_own_trajectories():
    member = _decls()[MEMBER]
    assert task_images(TASK).measured["member-sglang"] == MEMBER
    assert isinstance(member, LayerImage) and member.server and member.owner == TASK and parent_of(member) == "cuda"
    assert (member.pip_lock, member.apt_lock) == ("member-sglang.txt", "apt-member-sglang.txt")
    # The kernels of the FlashInfer its lock holds, and no kernel wheel beside them.
    assert member.fetched == FLASHINFER_CUBINS["0.6.18"] and member.fetched.dropped == ".lock"
    assert dict(member.env) == {**SERVER_ENV, "FLASHINFER_CUBIN_DIR": member.fetched.into}
    # What the reference holds for the driver: the same Apptainer, and the packages its .deb needs.
    assert member.debs == _decls()["agentic-coding-reference"].debs == dict([APPTAINER])
    assert _lines("apt-member-sglang.txt") == MEMBER_APT
    assert set(MEMBER_APT) <= set(REFERENCE_APT)


def test_the_sglang_members_lock_pins_its_engine_and_harbors_packages():
    text = (LOCKS / "member-sglang.txt").read_text()
    locked = pins_of(text)
    assert {name: locked.get(name) for name in SGLANG_PINS} == SGLANG_PINS
    assert len(locked) == 241 and "flashinfer-cubin" not in locked
    assert text.splitlines()[0] == ("# Generated by: hls images lock-pip layer member-sglang --task agentic_coding "
                                    "--server --no-deps")
    # The one wheel no index of the lock carries is named by its URL, and hashed like the rest.
    [tile] = [line for line in text.splitlines() if " @ " in line]
    assert tile == ("cuda-tile @ https://pypi.nvidia.com/cuda-tile/"
                    "cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_x86_64.whl \\")
    # harbor's packages are the reference's, at the versions an install on top of SGLang's gives.
    harbors = set(REFERENCE_PINS) - {"fsspec", "pandas", "pyarrow", "multiprocess", "xxhash", "datasets"}
    assert not [name for name in harbors if name not in locked]
    assert locked["harbor"] == REFERENCE_PINS["harbor"] and locked["openai"] == REFERENCE_PINS["openai"]


def test_the_lock_pins_the_version_of_the_checkouts_harbor():
    """The driver runs the checkout's harbor and reads the version of the distribution the image
    holds, so the two are one version."""
    project = REPO / "harbor" / "pyproject.toml"
    if not project.is_file():
        pytest.skip("the harbor submodule is not checked out")
    assert tomllib.loads(project.read_text())["project"]["version"] == REFERENCE_PINS["harbor"]


def test_the_driver_runs_the_checkouts_harbor_and_reads_the_images_distribution(tmp_path):
    """The reference image holds harbor's distribution, which ``harbor/__init__.py`` reads its
    version from; the code the driver imports is the checkout's, first on its path."""
    if not (REPO / "harbor" / "src" / "harbor" / "__init__.py").is_file():
        pytest.skip("the harbor submodule is not checked out")
    # The image's own packages: a harbor distribution whose code must never run.
    site = tmp_path / "dist-packages"
    (site / "harbor").mkdir(parents=True)
    (site / "harbor" / "__init__.py").write_text("raise RuntimeError('the image harbor was imported')\n")
    (site / "harbor-0.8.0.dist-info").mkdir()
    (site / "harbor-0.8.0.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: harbor\nVersion: 0.8.0\n")
    # The driver's path as the drive stage sets it, the checkout at its place here.
    path = f"{CHECKOUT}/src:{CHECKOUT}:{HARBOR_SRC}".replace(CHECKOUT, str(REPO))
    code = ("import site, sys; site.addsitedir(sys.argv[1]); import harbor; "
            "print(harbor.__file__); print(harbor.__version__)")
    # -S: only the path above and the stand-in for the image's packages, not this environment's.
    done = subprocess.run([sys.executable, "-S", "-c", code, str(site)], env={"PYTHONPATH": path},
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    file, version = done.stdout.splitlines()
    assert Path(file) == REPO / "harbor" / "src" / "harbor" / "__init__.py"
    assert version == REFERENCE_PINS["harbor"]
