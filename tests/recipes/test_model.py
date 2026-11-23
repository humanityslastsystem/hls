from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from hls.recipes.model import (
    BASE_ENVIRONMENT,
    BaseImage,
    CapImage,
    Fetched,
    ImageDecl,
    LayerImage,
    NodePin,
    UvPin,
    order,
)

CUDA = "nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:" + "7" * 64
NVIDIA_REPO = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
NODE = NodePin(version="22.23.3", sha256="1084aa36196bba4c3a5e69a1ee388a6e4ff729dad09445fbcd434b28fe3c24af")


def base(**over) -> BaseImage:
    fields = dict(name="s-a", cuda_ref=CUDA, snapshot="20261001T000000Z", apt_lock="apt-base.txt",
                  nvidia_apt={"cuda-nsight-systems-13-0=2025.3.2.474-1": "d" * 64}, nvidia_repo=NVIDIA_REPO,
                  pip_lock="pip-base.txt", constraints=["torch==2.11.0+cu130"], env={"PYTHONUNBUFFERED": "1"})
    return BaseImage(**{**fields, **over})


def test_declarations_are_told_apart_by_kind():
    adapter = TypeAdapter(list[ImageDecl])
    decls = adapter.validate_python([
        base().model_dump(),
        {"kind": "layer", "name": "s-a-tf", "parent": "s-a", "pip_lock": "pip-layer.txt"},
        {"kind": "cap", "name": "s-a-tf-agent", "parent": "s-a-tf", "cap": "agent",
         "apt_lock": "apt-agent.txt", "npm": {"@openai/codex": "0.146.0"}, "npm_lock": "npm-agent.json",
         "node": NODE.model_dump()},
    ])
    assert [type(d) for d in decls] == [BaseImage, LayerImage, CapImage]
    with pytest.raises(ValidationError, match="imported"):
        adapter.validate_python([{"kind": "imported", "name": "mslo-baseline"}])


@pytest.mark.parametrize("over, match", [
    ({"name": "S-A"}, "image name"),
    ({"name": "s_a"}, "image name"),
    ({"cuda_ref": "nvidia/cuda:13.0.1-devel-ubuntu24.04"}, "pinned by digest"),
    ({"cuda_ref": "nvidia/cuda@sha256:" + "7" * 64}, "pinned by digest"),
    ({"snapshot": "2026-10-01"}, "snapshot"),
    ({"apt_lock": "locks/apt.txt"}, "lock file name"),
    ({"pip_lock": ".hidden"}, "lock file name"),
    ({"nvidia_apt": {"cuda-nsight-systems-13-0": "d" * 64}}, "name=version"),
    ({"nvidia_apt": {"cuda-nsight-systems-12-9=2025.1-1": "d" * 64}}, "CUDA 13.0"),
    ({"nvidia_apt": {"cuda-nsight-systems-13-0=13.0.3-1": "sha256:" + "d" * 64}}, "SHA-256"),
    ({"nvidia_apt": ["cuda-nsight-systems-13-0=13.0.3-1"]}, "dictionary"),
    ({"constraints": []}, "hold"),
    ({"nvidia_repo": None}, "NVIDIA repository"),
    ({"nvidia_repo": NVIDIA_REPO.replace("https:", "http:")}, "repository URL"),
    ({"nvidia_repo": NVIDIA_REPO.rstrip("/")}, "repository URL"),
    ({"nvidia_repo": "https://example.com/a b/"}, "repository URL"),
    ({"constraints": ["torch>=2.11"]}, "name==version"),
    ({"env": {"BAD-NAME": "1"}}, "variable name"),
    ({"env": {"A": "line\nbreak"}}, "not printable"),
    ({"env": {"A": "{{ secret }}"}}, "build argument"),
    ({"env": {"TORCH_CUDA_ARCH_LIST": "9.0"}}, "never baked"),
    ({"env": {"PIP_CONSTRAINT": ""}}, "sets itself"),
    ({"env": {"PATH": "/usr/bin"}}, "sets itself"),
])
def test_a_base_with_a_bad_field_is_refused(over, match):
    with pytest.raises(ValidationError, match=match):
        base(**over)


# Every character str.splitlines() breaks on, and the tab: each would end the value's line in the
# .def, and a line starting with % opens a section.
@pytest.mark.parametrize("separator", ["\n", "\r", "\0", "\t", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85",
                                       "\u2028", "\u2029"])
def test_an_env_value_that_is_not_printable_is_refused(separator):
    with pytest.raises(ValidationError, match="not printable"):
        base(env={"X": f"{separator}%post{separator}touch /pwned #"})


@pytest.mark.parametrize("name", sorted(BASE_ENVIRONMENT))
def test_a_base_may_not_declare_a_variable_it_sets_itself(name):
    with pytest.raises(ValidationError, match="sets itself"):
        base(env={name: "x"})


def test_a_layer_needs_a_lock_or_a_file_and_another_parent():
    with pytest.raises(ValidationError, match="a layer adds a lock or a file"):
        LayerImage(name="s-a-tf", parent="s-a")
    files_only = LayerImage(name="t-baseline", parent="t-stack", owner="t", files={"/app/server.py": "server.py"})
    assert files_only.pip_lock is None and files_only.apt_lock is None
    with pytest.raises(ValidationError, match="its own parent"):
        LayerImage(name="s-a", parent="s-a", pip_lock="pip-layer.txt")


def test_a_base_reads_its_apt_and_pip_locks_from_different_files():
    with pytest.raises(ValidationError, match="different files"):
        base(pip_lock="apt-base.txt")


@pytest.mark.parametrize("fields, match", [
    ({"apt_lock": "x.txt", "pip_lock": "x.txt"}, "different files"),
    ({"apt_lock": "a.txt", "source_builds": {"sox==1.5.0": ["setuptools"]}}, "pip lock"),
    ({"pip_lock": "x.txt", "source_builds": {"sox==1.5.0": ["setuptools"], "SoX==1.5.0": ["setuptools"]}},
     "twice"),
    ({"pip_lock": "x.txt", "source_builds": {"sox>=1.5": ["setuptools"]}}, "name==version"),
    ({"pip_lock": "x.txt", "source_builds": {"sox==1.5.0": []}}, "build backend"),
    ({"pip_lock": "x.txt", "source_builds": {"sox==1.5.0": ["setup tools"]}}, "project name"),
    ({"pip_lock": "x.txt", "source_builds": {"sox==1.5.0": ["setuptools"]}, "build_lock": "b.txt"},
     "Extra inputs"),
])
def test_a_layer_builds_from_source_only_what_it_names_with_its_backends(fields, match):
    with pytest.raises(ValidationError, match=match):
        LayerImage(name="tts", parent="s-a", **fields)


def test_a_layer_may_name_its_source_builds_and_their_backends():
    layer = LayerImage(name="tts", parent="s-a", pip_lock="pip-sdist.txt",
                       source_builds={"sox==1.5.0": ["setuptools"]})
    assert layer.source_builds == {"sox==1.5.0": ["setuptools"]}


@pytest.mark.parametrize("npm, match", [
    ({}, "at least one"),
    ({"Claude": "1.0.0"}, "npm package name"),
    ({"@openai/codex": "latest"}, "exact version"),
    ({"@openai/codex": "^0.146.0"}, "exact version"),
])
def test_a_cap_pins_every_cli_by_exact_version(npm, match):
    with pytest.raises(ValidationError, match=match):
        CapImage(name="agent", parent="s-a", cap="agent", apt_lock="apt-agent.txt", npm=npm,
                 npm_lock="npm-agent.json", node=NODE)


def test_a_cap_reads_its_apt_and_npm_locks_from_different_files():
    with pytest.raises(ValidationError, match="different files"):
        CapImage(name="agent", parent="s-a", cap="agent", apt_lock="agent.txt",
                 npm={"@openai/codex": "0.146.0"}, npm_lock="agent.txt", node=NODE)


def test_a_cap_runs_its_clis_on_a_node_pinned_by_version_and_digest():
    fields = dict(name="agent", parent="s-a", cap="agent", apt_lock="apt-agent.txt",
                  npm={"@openai/codex": "0.146.0"}, npm_lock="npm-agent.json")
    with pytest.raises(ValidationError, match="node"):
        CapImage(**fields)
    assert CapImage(**fields, node=NODE).node == NODE


@pytest.mark.parametrize("fields, match", [
    ({"version": "22", "sha256": "8" * 64}, "a Node version"),
    ({"version": "v22.23.3", "sha256": "8" * 64}, "a Node version"),
    ({"version": "22.23.3", "sha256": "sha256:" + "8" * 64}, "SHA-256"),
])
def test_a_node_pin_is_an_exact_version_and_a_digest(fields, match):
    with pytest.raises(ValidationError, match=match):
        NodePin(**fields)


def test_a_base_pulls_its_cuda_image_by_digest_alone():
    assert base().pull_ref == "nvidia/cuda@sha256:" + "7" * 64
    assert base(cuda_ref="nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:" + "8" * 64).pull_ref == (
        "nvcr.io/nvidia/cuda@sha256:" + "8" * 64)


def test_a_uv_pin_is_an_exact_version_and_a_digest():
    assert UvPin(version="0.11.21", sha256="8" * 64).version == "0.11.21"
    with pytest.raises(ValidationError):
        UvPin(version="0.11", sha256="8" * 64)


def test_order_puts_parents_first_and_keeps_the_rest_in_place():
    layer = LayerImage(name="s-a-tf", parent="s-a", pip_lock="pip-layer.txt")
    cap = CapImage(name="agent", parent="s-a-tf", cap="agent", apt_lock="apt-agent.txt",
                   npm={"@openai/codex": "0.146.0"}, npm_lock="npm-agent.json", node=NODE)
    other = LayerImage(name="baseline", parent="s-a", pip_lock="pip-layer.txt")
    assert [d.name for d in order([cap, other, layer, base()])] == ["s-a", "s-a-tf", "agent", "baseline"]


@pytest.mark.parametrize("decls, match", [
    ([LayerImage(name="s-a-tf", parent="s-a", pip_lock="pip-layer.txt")], "unknown parent s-a"),
    ([base(), base()], "declared twice"),
    ([LayerImage(name="a", parent="b", pip_lock="x.txt"), LayerImage(name="b", parent="a", pip_lock="x.txt")],
     "cycle"),
])
def test_order_refuses_what_cannot_be_built(decls, match):
    with pytest.raises(ValueError, match=match):
        order(decls)


def test_a_base_without_a_pip_lock_holds_nothing():
    bare = base(name="cuda", pip_lock=None, constraints=[])
    assert bare.pip_lock is None and bare.constraints == []
    with pytest.raises(ValidationError, match="installs nothing to hold"):
        base(pip_lock=None)


@pytest.mark.parametrize(("fields", "match"), [
    ({"files": {"/app/server.py": "server.py"}}, "names its task"),
    ({"owner": "Bad-Dir"}, "task directory"),
    ({"owner": "t", "files": {"/opt/hls/server.py": "server.py"}}, "images' own directory"),
    ({"owner": "t", "files": {"/opt/hls": "server.py"}}, "images' own directory"),
    ({"owner": "t", "files": {"app/server.py": "server.py"}}, "absolute path"),
    ({"owner": "t", "files": {"/app/../server.py": "server.py"}}, "none . or .."),
    ({"owner": "t", "files": {"/./server.py": "server.py"}}, "none . or .."),
    ({"owner": "t", "files": {"/app/server.py": "../server.py"}}, "relative to the task"),
    ({"owner": "t", "files": {"/app/server.py": "/server.py"}}, "relative to the task"),
    ({"owner": "t", "files": {"/app/server.py": "a/./server.py"}}, "relative to the task"),
])
def test_a_tasks_own_layer_names_its_task_and_keeps_its_files_out_of_the_images_directory(fields, match):
    with pytest.raises(ValidationError, match=match):
        LayerImage(name="t-baseline", parent="cuda", pip_lock="stack.txt", **fields)


def test_a_server_layer_is_a_tasks_own_or_shared():
    layer = LayerImage(name="t-baseline", parent="cuda", pip_lock="stack.txt", owner="t", server=True,
                       files={"/app/server.py": "server/vllm.py", "/.dockerenv": "dockerenv"})
    assert layer.server and layer.owner == "t" and layer.files["/.dockerenv"] == "dockerenv"
    shared = LayerImage(name="stack", parent="cuda", pip_lock="stack.txt", server=True)
    assert shared.server and shared.owner is None


def test_a_layer_declares_its_whole_environment_with_values_the_shell_reads_as_one_word():
    layer = LayerImage(name="t-baseline", parent="cuda", pip_lock="stack.txt", owner="t", server=True,
                       env={"PATH": "/opt/venv/bin:$PATH", "TORCH_CUDA_ARCH_LIST": "9.0"})
    assert layer.env == [("PATH", "/opt/venv/bin:$PATH"), ("TORCH_CUDA_ARCH_LIST", "9.0")]
    with pytest.raises(ValidationError, match="env names a variable twice"):
        LayerImage(name="t-baseline", parent="cuda", pip_lock="stack.txt", owner="t", server=True,
                   env=[("X", "1"), ("X", "2")])
    for value in ("a b", "'x'", "$(id)", "a;b"):
        with pytest.raises(ValidationError, match="no space or quote"):
            LayerImage(name="t-baseline", parent="cuda", pip_lock="stack.txt", owner="t", server=True,
                       env={"X": value})


def fetched(**over) -> Fetched:
    fields = dict(url="https://example.com/releases/v1.2.3/kernels-1.2.3-py3-none-any.whl", sha256="a" * 64,
                  member="kernels/files", lock="fetch-kernels.txt", into="/opt/kernels", variable="KERNELS_DIR",
                  loader="kernels.loader:load", refuse_env="KERNELS_NO_DOWNLOAD")
    return Fetched(**{**fields, **over})


def fetching(**over) -> LayerImage:
    fields = dict(name="stack", parent="cuda", pip_lock="stack.txt", server=True, fetched=fetched(),
                  env={"PATH": "/opt/venv/bin:$PATH", "KERNELS_DIR": "/opt/kernels"})
    return LayerImage(**{**fields, **over})


def test_a_server_layer_may_fetch_the_files_a_package_would_download():
    layer = fetching()
    assert layer.fetched.into == "/opt/kernels" and ("KERNELS_DIR", "/opt/kernels") in layer.env
    # The fetched files alone are something a layer adds.
    assert fetching(name="kernels", parent="stack", pip_lock=None).pip_lock is None


@pytest.mark.parametrize(("fields", "match"), [
    ({"server": False}, "server layer"),
    ({"pip_lock": "fetch-kernels.txt"}, "different files"),
    ({"apt_lock": "fetch-kernels.txt"}, "different files"),
    # The package looks where the image's environment says, so the layer bakes the variable there.
    ({"env": None}, "bakes KERNELS_DIR=/opt/kernels"),
    ({"env": {"PATH": "/opt/venv/bin:$PATH"}}, "bakes KERNELS_DIR=/opt/kernels"),
    ({"env": {"KERNELS_DIR": "/opt/elsewhere"}}, "bakes KERNELS_DIR=/opt/kernels"),
])
def test_only_a_server_layer_fetches_from_a_lock_of_its_own_into_the_directory_it_bakes(fields, match):
    with pytest.raises(ValidationError, match=match):
        fetching(**fields)


@pytest.mark.parametrize(("over", "match"), [
    ({"url": "http://example.com/kernels.whl"}, "https URL of a file"),
    ({"url": "https://example.com/releases/"}, "https URL of a file"),
    ({"url": "https://example.com/kernels.whl?token=x"}, "https URL of a file"),
    ({"url": "https://example.com/a b.whl"}, "https URL of a file"),
    ({"sha256": "sha256:" + "a" * 64}, "SHA-256"),
    ({"member": "/kernels/files"}, "relative path"),
    ({"member": "kernels/../files"}, "relative path"),
    ({"lock": "locks/fetch.txt"}, "lock file name"),
    ({"into": "opt/kernels"}, "absolute path"),
    ({"into": "/opt/hls/kernels"}, "share a directory with /opt/hls"),
    ({"into": "/opt/venv"}, "share a directory with /opt/venv"),
    ({"into": "/opt"}, "share a directory with"),
    ({"variable": "KERNELS-DIR"}, "variable name"),
    ({"variable": "KERNELS_NO_DOWNLOAD"}, "another variable refuses downloads"),
    ({"loader": "kernels.loader.load"}, "<module>:<function>"),
    ({"loader": "kernels.loader:load()"}, "<module>:<function>"),
    ({"loader": 'kernels"; import os; os.system("id"):load'}, "<module>:<function>"),
    ({"refuse_env": "NO DOWNLOAD"}, "variable name"),
    ({"env": {"KERNELS_DIR": "/opt/kernels"}}, "Extra inputs"),
])
def test_fetched_files_with_a_bad_field_are_refused(over, match):
    with pytest.raises(ValidationError, match=match):
        fetched(**over)


def test_a_layer_that_fetches_nothing_is_written_as_one_without_the_field():
    plain = LayerImage(name="stack", parent="cuda", pip_lock="stack.txt", server=True)
    assert "fetched" not in plain.model_dump() and "fetched" not in plain.model_dump_json()
    assert LayerImage.model_validate(plain.model_dump()) == plain
    assert fetching().model_dump()["fetched"]["variable"] == "KERNELS_DIR"
    assert LayerImage.model_validate_json(fetching().model_dump_json()) == fetching()
