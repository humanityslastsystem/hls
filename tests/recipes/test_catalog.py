from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hls.recipes import catalog, pins
from hls.recipes.catalog import (AGENT_CAPS, AGENT_CLIS, BASES, LAYERS, SERVER_LAYERS, ImageSet, Served, agent_cap,
                                 catalog_task, image_set, shared_decls, task_images, task_keys, task_module)
from pydantic import TypeAdapter

from hls.recipes.model import BASE_ENVIRONMENT, CONSTRAINTS, BaseImage, CapImage, ImageDecl, LayerImage, order
from hls.recipes.render import (baked_environment, check_source_builds, files_for, npm_manifest, recipe_key,
                                render_def)
from hls.recipes.render_docker import dockerfile_key
from hls.stages.images import image_keys
from tests.recipes.declared import declare

COMPARABILITY = ("multitenant_slo", "tab_completion", "multimodal_tts", "multimodal_asr", "multimodal_omni")
#: The NVIDIA pins and the sha256 of each .deb, as NVIDIA's index lists them.
NSIGHT = {
    "cuda-nsight-systems-13-0=13.0.3-1": "d5105643b0222da97f933f84f9d6b0f6d5d37e315253fa991e12a9f741c595ee",
    "nsight-systems-2025.3.2=2025.3.2.474-253236389321v0":
        "c7cfe27e2250eb91e1a67e7feb5f2c490c7f598e3b3a3d047aff000bc49f9d6b",
}


def test_both_bases_share_the_pins_and_hold_their_torch_line():
    for name, base in BASES.items():
        assert isinstance(base, BaseImage)
        assert base.name == name
        assert (base.cuda_ref, base.snapshot) == (pins.CUDA_BASE, pins.SNAPSHOT)
        assert base.apt_lock == "apt-base.txt"
        assert base.nvidia_apt == NSIGHT
        assert base.nvidia_repo == pins.NVIDIA_REPO
        assert base.env == {"HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1", "UV_NO_CACHE": "1",
                            "VLLM_DO_NOT_TRACK": "1"}
    assert BASES["s-a"].constraints[0] == "torch==2.11.0+cu130"
    assert BASES["s0"].constraints[0] == "torch==2.13.0+cu130"


def test_the_nvidia_repository_is_the_one_cuda_keyring_names():
    assert pins.NVIDIA_REPO == "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"


@pytest.mark.parametrize("name", sorted(BASES))
def test_a_base_pulls_the_cuda_image_by_digest_alone_and_keeps_its_tag_in_a_label(name):
    text = render_def(BASES[name], parent_sif=None, locks=pins.LOCKS_DIR, uv=pins.UV)
    digest = pins.CUDA_BASE.split("@", 1)[1]
    assert text.startswith(f"Bootstrap: docker\nFrom: nvidia/cuda@{digest}\n")
    assert f"    hls.cuda_ref {pins.CUDA_BASE}\n" in text
    # Both Nsight pins are in the version-checked set.
    locked = "$(sed -e '/^#/d' -e '/^$/d' /opt/hls/locks/apt-base.txt)"
    assert f"for hls_pin in {locked} {' '.join(NSIGHT)}; do" in text


@pytest.mark.parametrize("task", COMPARABILITY)
def test_the_comparability_tasks_build_on_s_a(task):
    images = task_images(task)
    built = {d.name: d for d in images.decls}
    assert built[images.submission_base].parent == "s-a"
    assert images.harness == images.submission_base
    sandbox = built[images.sandbox]
    assert isinstance(sandbox, CapImage) and sandbox.parent == images.submission_base


def test_sandbox_and_submission_share_base_and_layer():
    for layer, cap in AGENT_CAPS.items():
        assert cap.parent == layer and cap.npm == {"@anthropic-ai/claude-code": pins.CLAUDE_CODE_VERSION,
                                                   "@openai/codex": pins.CODEX_VERSION}
        assert cap.node == pins.NODE and cap.npm_lock == "npm-agent.json"
        assert cap.apt_lock == "apt-agent.txt"


def test_the_claude_code_cli_is_at_or_above_the_version_that_evaluates_goal():
    # 2.1.247 accepts /goal but never evaluates it; pins.CLAUDE_CODE_VERSION's comment says why.
    assert tuple(int(p) for p in pins.CLAUDE_CODE_VERSION.split(".")) >= (2, 1, 248)


def _engine_floor(requirement: str) -> tuple[int, ...]:
    """The version a ``>=X[.Y[.Z]]`` engines requirement starts at."""
    m = re.fullmatch(r">=\s*(\d+(?:\.\d+){0,2})", requirement)
    assert m, f"engines requirement {requirement!r} is not >=X.Y.Z"
    return tuple(int(p) for p in m.group(1).split("."))


def test_the_agent_npm_lock_holds_the_declared_clis_on_the_pinned_node():
    lock = json.loads((pins.LOCKS_DIR / "npm-agent.json").read_text())
    manifest = json.loads(npm_manifest(next(iter(AGENT_CAPS.values()))))
    assert lock["lockfileVersion"] in (2, 3)
    assert lock["name"] == lock["packages"][""]["name"] == manifest["name"]
    assert lock["packages"][""]["dependencies"] == manifest["dependencies"] == AGENT_CLIS
    node = tuple(int(p) for p in pins.NODE.version.split("."))
    for name in AGENT_CLIS:
        engines = lock["packages"][f"node_modules/{name}"].get("engines", {})
        assert node >= _engine_floor(engines.get("node", ">=0")), name
    claude = lock["packages"]["node_modules/@anthropic-ai/claude-code"]
    assert _engine_floor(claude["engines"]["node"]) >= (22,)


#: Each CLI's glibc build for linux/x64 in the committed npm lock: its path, version and the name it
#: is published under. The musl build also installs on glibc, since its entry names no libc.
GLIBC_BUILDS = {
    "@anthropic-ai/claude-code": ("@anthropic-ai/claude-code-linux-x64", "2.1.251",
                                  "@anthropic-ai/claude-code-linux-x64"),
    "@openai/codex": ("@openai/codex-linux-x64", "0.146.0-linux-x64", "@openai/codex"),
}


def test_the_agent_npm_lock_holds_each_clis_glibc_build():
    packages = json.loads((pins.LOCKS_DIR / "npm-agent.json").read_text())["packages"]
    for cli, (path, version, published) in GLIBC_BUILDS.items():
        entry = packages[f"node_modules/{path}"]
        assert (entry.get("name", path), entry["version"]) == (published, version), cli
        platform = (entry["os"], entry["cpu"], entry.get("libc", ["glibc"]))
        assert platform == (["linux"], ["x64"], ["glibc"]), cli
        assert path in packages[f"node_modules/{cli}"]["optionalDependencies"], cli


#: The images a reroll of each task runs on: sandbox, submission base, harness.
REROLLS = {
    "multitenant_slo": ("s-a-tf-5-13-1-agent", "s-a-tf-5-13-1", "s-a-tf-5-13-1"),
    "tab_completion": ("s-a-tf-5-13-1-agent", "s-a-tf-5-13-1", "s-a-tf-5-13-1"),
    "sample_task": ("s-a-tf-5-13-1-agent", "s-a-tf-5-13-1", "s-a-tf-5-13-1"),
    "multimodal_tts": ("s-a-tf-4-57-3-qwen-tts-agent", "s-a-tf-4-57-3-qwen-tts", "s-a-tf-4-57-3-qwen-tts"),
    "multimodal_asr": ("s-a-tf-5-9-0-agent", "s-a-tf-5-9-0", "s-a-tf-5-9-0"),
    "multimodal_omni": ("s-a-tf-5-9-0-agent", "s-a-tf-5-9-0", "s-a-tf-5-9-0"),
    "deepresearch": ("deepresearch-sandbox-agent", "deepresearch-sandbox-agent", "deepresearch-reference"),
    "agentic_coding": ("agentic-coding-tf-5-17-0-agent", "agentic-coding-tf-5-17-0-agent",
                       "agentic-coding-reference"),
}


def test_each_task_rerolls_on_the_images_its_module_declares():
    assert set(REROLLS) == set(task_keys())
    for task, expected in REROLLS.items():
        images = task_images(task)
        assert (images.sandbox, images.submission_base, images.harness) == expected, task


#: The constraints file each task's package installs under: the bases' file on every submission
#: base whose base holds torch, none on deepresearch's and agentic_coding's, whose base, cuda,
#: holds nothing.
SUBMISSION_CONSTRAINTS = ({task: CONSTRAINTS for task in (*COMPARABILITY, "sample_task")}
                          | {"deepresearch": None, "agentic_coding": None})


def test_a_package_installs_under_the_constraints_file_its_submission_base_holds():
    assert set(SUBMISSION_CONSTRAINTS) == set(task_keys())
    for task, expected in SUBMISSION_CONSTRAINTS.items():
        images = task_images(task)
        assert images.submission_constraints == expected, task
        if expected is not None:
            # The base under the submission base writes the file.
            by_name = {d.name: d for d in images.decls}
            base = by_name[by_name[images.submission_base].parent]
            text = render_def(base, parent_sif=None, locks=pins.LOCKS_DIR, uv=pins.UV)
            assert f"> {CONSTRAINTS}" in text, task


def test_deepresearch_builds_on_its_own_layer_and_packages_on_its_sandbox():
    """Its sandbox is the agent cap of a layer of its own on the cuda base, its submissions are
    packaged on that sandbox, and its harness roles run on its reference image."""
    declared = image_set("deepresearch")
    images = task_images("deepresearch")
    by_name = {d.name: d for d in images.decls}
    layer = by_name[declared.layer]
    assert isinstance(layer, LayerImage)
    assert (layer.owner, layer.server, layer.parent) == ("deepresearch", False, "cuda")
    assert declared.layer not in LAYERS and declared.package_on_sandbox and declared.harness_on_reference
    assert by_name[images.sandbox] == agent_cap(declared.layer)
    assert images.submission_base == images.sandbox
    assert set(images.measured) == {"baseline", "reference", "member-vllm", "member-sglang"}
    assert images.reference == images.harness == images.measured["reference"]


def test_no_task_builds_on_the_forward_line():
    """S0 and its layer are declared, and warmable, for new tasks; no existing task builds on them."""
    assert "s0" in BASES and LAYERS["s0-tf-5-17-0"].parent == "s0"
    assert {d.name for d in shared_decls()} >= {"s0", "s0-tf-5-17-0", "s0-tf-5-17-0-agent"}
    for task in task_keys():
        assert not any(d.name.startswith("s0") for d in task_images(task).decls), task


def test_a_task_declares_its_own_images_by_role():
    images = task_images("multitenant_slo")
    assert set(images.measured) == {"baseline", "reference", "member-vllm", "member-sglang"}
    assert images.reference == images.measured["reference"]
    assert task_images("sample_task").measured == {}
    assert task_images("sample_task").reference == task_images("sample_task").harness


def test_no_digest_of_an_image_is_declared_anywhere():
    """A task module says how its images are made; what file a store holds for an image is the
    store's, and every run records the digest of each image it ran."""
    tasks_dir = Path(__file__).resolve().parents[2] / "tasks"
    src = Path(catalog.__file__).resolve().parents[1]
    assert not (Path(catalog.__file__).with_name("measured.toml")).exists()
    for task in task_keys():
        images = task_images(task)
        for decl in images.decls:
            if decl.name in images.measured.values():
                # The digest of a .deb or of an archive a layer fetches pins an input of its build, as a
                # lock's hashes do, not the image.
                declared = decl.model_dump_json(exclude={"debs", "fetched"})
                assert not re.search(r"[0-9a-f]{64}", declared), decl.name
    sources = [*tasks_dir.glob("*/task*.py"), *src.rglob("*.py")]
    named = [p for p in sources if re.search(r"sha256-[0-9a-f]{64}", p.read_text())]
    assert named == []


def test_an_unknown_task_is_refused():
    for read in (lambda: task_images("nope"), lambda: task_module("nope"), lambda: catalog_task("tasks.nope.task")):
        with pytest.raises(ValueError, match="no task 'nope': a task is a directory under tasks/"):
            read()


def _server(name: str) -> LayerImage:
    return LayerImage(name=name, parent="cuda", pip_lock="x.txt", owner="t", server=True)


def _served(name: str = "t-baseline") -> Served:
    return Served(image=_server(name), entrypoint=["python", "/app/server.py"])


@pytest.mark.parametrize("fields, match", [
    ({}, "layer\n  Field required"),
    ({"layer": "s-a-tf-9-9-9"}, "layer 's-a-tf-9-9-9' is not one of"),
    ({"layer": "t-own", "images": [LayerImage(name="t-own", parent="s-a", pip_lock="x.txt", owner="t",
                                              server=True)]}, "nor one of the task's own images that is no server"),
    ({"layer": "s-a-tf-5-13-1", "harness_on_reference": True}, "declares none"),
    ({"layer": "s-a-tf-5-13-1", "images": [LayerImage(name="t-x", parent="s-a", pip_lock="x.txt")]},
     "image t-x of images names no owner"),
    ({"layer": "s-a-tf-5-13-1", "images": [LayerImage(name="s-a", parent="cuda", pip_lock="x.txt", owner="t")]},
     "image s-a of images has the name of a shared image"),
    ({"layer": "s-a-tf-5-13-1", "images": [LayerImage(name="t-x", parent="s-a", pip_lock="x.txt", owner="t")] * 2},
     "an image is named twice in images"),
    ({"layer": "s-a-tf-5-13-1", "reference": LayerImage(name="cuda-setuptools", parent="cuda", pip_lock="y.txt",
                                                        server=True)},
     "has the name of a shared image and is no shared server layer"),
    ({"layer": "s-a-tf-5-13-1", "reference": LayerImage(name="t-ref", parent="cuda", pip_lock="x.txt", owner="t")},
     "the reference image t-ref is no server layer"),
    ({"layer": "s-a-tf-5-13-1", "reference": _server("s-a")}, "the name of a shared image"),
    ({"layer": "s-a-tf-5-13-1", "baseline": _served("t-x"),
      "reference": LayerImage(name="t-x", parent="cuda", pip_lock="y.txt", owner="t", server=True)},
     "two different images are named t-x"),
    ({"layer": "s-a-tf-5-13-1", "members": {"Bad_Engine": _server("t-m")}}, "not an engine name"),
    ({"layer": "s-a-tf-5-13-1", "sandbox": _server("t-sandbox")}, "Extra inputs are not permitted"),
    ({"layer": "s-a-tf-5-13-1", "reference": {"kind": "imported", "name": "t-ref"}}, "Input should be 'layer'"),
])
def test_an_image_set_that_contradicts_itself_is_refused(fields, match):
    with pytest.raises(ValueError, match=match):
        ImageSet(**fields)


def test_one_image_may_serve_two_roles():
    image = LayerImage(name="t-vllm", parent="cuda", pip_lock="x.txt", owner="t", server=True)
    declared = ImageSet(layer="s-a-tf-5-13-1", baseline=Served(image=image, entrypoint=["vllm"]),
                        members={"vllm": image})
    assert declared.roles() == {"baseline": image, "member-vllm": image}


def test_a_shared_server_layer_may_serve_a_role_as_it_is(monkeypatch):
    shared = SERVER_LAYERS["cuda-setuptools"]
    declare(monkeypatch, {"sample_task": ImageSet(layer="s-a-tf-5-13-1", members={"vllm": shared})})
    images = task_images("sample_task")
    assert images.measured == {"member-vllm": "cuda-setuptools"}
    assert [d.name for d in images.decls][-2:] == ["cuda", "cuda-setuptools"]


def test_a_task_builds_its_sandbox_on_a_layer_of_its_own_with_the_shared_agent_cap(monkeypatch):
    own = LayerImage(name="sample-task-sandbox", parent="cuda", pip_lock="x.txt", owner="sample_task",
                     files={"/app/x.py": "task.py"})
    reference = LayerImage(name="sample-task-reference", parent="cuda", pip_lock="y.txt", owner="sample_task",
                           server=True)
    declare(monkeypatch, {"sample_task": ImageSet(layer=own.name, images=[own], reference=reference,
                                                  harness_on_reference=True)})
    images = task_images("sample_task")
    assert (images.sandbox, images.submission_base, images.harness, images.reference) == (
        "sample-task-sandbox-agent", "sample-task-sandbox", "sample-task-reference", "sample-task-reference")
    assert agent_cap(own.name) in images.decls and images.measured == {"reference": "sample-task-reference"}
    # On the cuda base, which holds no torch, the package installs under no constraints file.
    assert images.submission_constraints is None
    declare(monkeypatch, {"sample_task": ImageSet(layer=own.name, images=[own.model_copy(update={"parent": "s-a"})])})
    assert task_images("sample_task").submission_constraints == CONSTRAINTS


@pytest.mark.parametrize("parent", ["t-server", "cuda-setuptools"])
def test_a_sandbox_layer_on_a_server_image_is_refused(parent):
    """No agent cap or submission builds on a server image, which may hold what the deny list
    names: a package plan holds no cap, so the layer itself is refused."""
    server = LayerImage(name="t-server", parent="cuda", pip_lock="x.txt", owner="t", server=True)
    own = LayerImage(name="t-own", parent=parent, pip_lock="y.txt", owner="t")
    with pytest.raises(ValueError, match=f"layer 't-own' builds on server image {parent}"):
        ImageSet(layer="t-own", images=[own], reference=server)


def test_a_sandbox_layer_only_adds_to_the_environment_its_base_bakes(monkeypatch):
    """The variables of the images below a sandbox's layer stay as they are, the torch hold every
    install of an agent or a submission keeps among them."""
    own = LayerImage(name="sample-task-sandbox", parent="s-a", pip_lock="x.txt", owner="sample_task",
                     env={"PYTHONPATH": "/app"})
    declare(monkeypatch, {"sample_task": ImageSet(layer=own.name, images=[own])})
    baked = baked_environment(own.name, task_images("sample_task").decls)
    assert baked["PYTHONPATH"] == "/app" and baked["UV_CONSTRAINT"] == baked["PIP_CONSTRAINT"] == CONSTRAINTS
    for env, match in [({"PATH": "/opt/venv/bin:$PATH"}, r"sets \['PATH'\] to what s-a gives already"),
                       ({"UV_CONSTRAINT": "/elsewhere"}, r"sets \['UV_CONSTRAINT'\], which s-a sets"),
                       ({"TORCH_CUDA_ARCH_LIST": "9.0"}, "sets TORCH_CUDA_ARCH_LIST")]:
        changed = own.model_copy(update={"env": list(env.items())})
        declare(monkeypatch, {"sample_task": ImageSet(layer=own.name, images=[changed])})
        with pytest.raises(ValueError, match=match):
            task_images("sample_task")


def test_a_server_layer_sets_over_its_base_but_never_to_the_value_it_has(monkeypatch):
    server = LayerImage(name="sample-task-reference", parent="cuda", pip_lock="x.txt", owner="sample_task",
                        server=True, env={"UV_CONSTRAINT": "/elsewhere", "UV_TORCH_BACKEND": "cu130"})
    declare(monkeypatch, {"sample_task": ImageSet(layer="s-a-tf-5-13-1", reference=server)})
    assert baked_environment(server.name, task_images("sample_task").decls)["UV_CONSTRAINT"] == "/elsewhere"
    again = server.model_copy(update={"env": [("VLLM_DO_NOT_TRACK", "1")]})
    declare(monkeypatch, {"sample_task": ImageSet(layer="s-a-tf-5-13-1", reference=again)})
    with pytest.raises(ValueError, match=r"sets \['VLLM_DO_NOT_TRACK'\] to what cuda gives already"):
        task_images("sample_task")


def _path_entries(name: str, decls) -> list[str]:
    """The entries of PATH in image *name*, as its chain's %environment exports set it in turn
    over the host's (``HOST``)."""
    by_name = {d.name: d for d in decls}
    chain, current = [], name
    while current is not None:
        chain.append(by_name[current])
        current = getattr(by_name[current], "parent", None)
    path = "HOST"
    for decl in reversed(chain):
        pairs = (list(BASE_ENVIRONMENT.items()) if isinstance(decl, BaseImage)
                 else list(getattr(decl, "env", None) or []))
        for key, value in pairs:
            if key == "PATH":
                path = value.replace("$PATH", path)
    return path.split(":")


def test_no_image_of_a_task_repeats_an_entry_of_path():
    for task in task_keys():
        decls = task_images(task).decls
        for decl in decls:
            entries = _path_entries(decl.name, decls)
            assert len(entries) == len(set(entries)), (task, decl.name, entries)


def test_a_task_may_package_on_its_sandbox(monkeypatch):
    own = LayerImage(name="sample-task-sandbox", parent="cuda", pip_lock="x.txt", owner="sample_task")
    declare(monkeypatch, {"sample_task": ImageSet(layer=own.name, images=[own], package_on_sandbox=True)})
    images = task_images("sample_task")
    assert images.submission_base == images.sandbox == "sample-task-sandbox-agent"
    assert images.submission_constraints is None
    declare(monkeypatch, {"sample_task": ImageSet(layer="s-a-tf-5-13-1", package_on_sandbox=True)})
    images = task_images("sample_task")
    assert (images.submission_base, images.submission_constraints) == ("s-a-tf-5-13-1-agent", CONSTRAINTS)


def test_a_layer_a_task_declares_for_its_server_images_to_build_on(monkeypatch):
    backend = LayerImage(name="sample-task-backend", parent="cuda", pip_lock="x.txt", owner="sample_task", server=True)
    server = LayerImage(name="sample-task-reference", parent=backend.name, pip_lock="y.txt", owner="sample_task",
                        server=True)
    declare(monkeypatch, {"sample_task": ImageSet(layer="s-a-tf-5-13-1", images=[backend], reference=server)})
    names = [d.name for d in task_images("sample_task").decls]
    assert names.index("sample-task-backend") < names.index("sample-task-reference")
    unused = LayerImage(name="sample-task-unused", parent="cuda", pip_lock="z.txt", owner="sample_task")
    declare(monkeypatch, {"sample_task": ImageSet(layer="s-a-tf-5-13-1", images=[backend, unused], reference=server)})
    with pytest.raises(ValueError, match=r"declares images \['sample-task-unused'\], which none of its images builds on"):
        task_images("sample_task")


def test_a_served_image_starts_with_a_command():
    with pytest.raises(ValueError, match="a command, not an empty list"):
        Served(image=_server("t-baseline"), entrypoint=[])


def test_a_tasks_own_layer_keeps_its_locks_under_its_own_task(monkeypatch):
    other = LayerImage(name="sample-task-baseline", parent="cuda", pip_lock="x.txt", owner="multitenant_slo",
                       server=True)
    declare(monkeypatch, {"sample_task": ImageSet(layer="s-a-tf-5-13-1", reference=other)})
    with pytest.raises(ValueError, match="keeps its locks under tasks/multitenant_slo, not tasks/sample_task"):
        task_images("sample_task")


def test_a_task_image_on_an_image_no_one_declares_is_refused(monkeypatch):
    orphan = LayerImage(name="sample-task-reference", parent="nowhere", pip_lock="x.txt", owner="sample_task",
                        server=True)
    declare(monkeypatch, {"sample_task": ImageSet(layer="s-a-tf-5-13-1", reference=orphan)})
    with pytest.raises(ValueError, match="builds on nowhere, which is no shared image and none the task declares"):
        task_images("sample_task")


def test_a_module_that_declares_no_images_is_refused(monkeypatch):
    import tasks.sample_task.task as module

    monkeypatch.delattr(module, "IMAGES")
    with pytest.raises(ValueError, match=r"tasks.sample_task.task declares no IMAGES = ImageSet\(...\)"):
        task_images("sample_task")


def test_every_shared_image_has_its_lock_files_and_a_key():
    decls = shared_decls()
    assert order(decls) == decls
    assert {d.name for d in decls} == {*BASES, *LAYERS, *(c.name for c in AGENT_CAPS.values()), *SERVER_LAYERS}
    keys = {}
    for d in decls:
        for path in files_for(d, pins.LOCKS_DIR).values():
            assert path.is_file(), f"{d.name} copies in {path}, which is not committed"
        parent = getattr(d, "parent", None)
        keys[d.name] = recipe_key(d, parent_key=keys.get(parent), locks_dir=pins.LOCKS_DIR, uv=pins.UV)
        text = render_def(d, parent_sif=Path("/node/parent.sif") if parent else None, locks=pins.LOCKS_DIR,
                          uv=pins.UV)
        assert "TORCH_CUDA_ARCH_LIST" not in text
    assert len(set(keys.values())) == len(keys)


def test_every_image_a_task_declares_keys_alike_after_a_plans_json_round_trip():
    # A plan stores each declaration as JSON with sorted keys, and the images stage keys what it
    # reads back: the key a planner looks up must be the one the stage publishes under.
    adapter = TypeAdapter(list[ImageDecl])
    for task in task_keys():
        decls = task_images(task).decls
        copied = adapter.validate_python(json.loads(json.dumps(adapter.dump_python(decls, mode="json"),
                                                                sort_keys=True)))
        for key in (recipe_key, dockerfile_key):
            assert image_keys(copied, key) == image_keys(decls, key), task


def test_every_image_a_task_declares_has_its_files_and_a_key():
    for task in task_keys():
        images = task_images(task)
        keys = image_keys(images.decls, recipe_key)
        assert len(set(keys.values())) == len(keys), task
        for d in images.decls:
            for path in files_for(d, pins.LOCKS_DIR).values():
                assert path.is_file(), f"{task}: {d.name} copies in {path}, which is not committed"


def test_the_tts_layer_builds_sox_as_published_from_source_against_the_bases_setuptools():
    tts = LAYERS["s-a-tf-4-57-3-qwen-tts"]
    assert tts.source_builds == {"sox==1.5.0": ["setuptools"]}
    assert [name for name, layer in LAYERS.items() if layer.source_builds] == ["s-a-tf-4-57-3-qwen-tts"]
    check_source_builds(shared_decls(), pins.LOCKS_DIR)
    for task in task_keys():
        check_source_builds(task_images(task).decls, pins.LOCKS_DIR)


@pytest.mark.parametrize("name", sorted(LAYERS))
def test_each_layer_lock_resolved_wheels_only_but_for_its_source_builds(name):
    """The lock agrees with the install, which takes wheels alone but for the named source builds."""
    layer = LAYERS[name]
    header = (pins.LOCKS_DIR / layer.pip_lock).read_text().splitlines()[0]
    named = re.findall(r"--source-build (\S+)", header)
    assert named == sorted(pin.split("==", 1)[0] for pin in layer.source_builds)


def test_every_layer_builds_on_a_base():
    assert all(isinstance(layer, LayerImage) and layer.parent in BASES for layer in LAYERS.values())
    # A task's layer is a shared one or one of its own.
    own = {d.name for task in task_keys() for d in image_set(task).images}
    assert {image_set(task).layer for task in task_keys()} <= set(LAYERS) | own


def test_a_task_module_names_its_directory_in_the_catalog():
    """The catalog is keyed by a task's directory, whose task.py defines its TASK; a TaskSpec's id
    may differ from the directory."""
    tasks_dir = Path(__file__).resolve().parents[2] / "tasks"
    present = task_keys()
    assert {"deepresearch", "multimodal_asr", "multimodal_omni", "multimodal_tts", "multitenant_slo"} <= set(present)
    assert [catalog_task(task_module(d)) for d in present] == present
    assert all((tasks_dir / d / "task.py").is_file() for d in present)


@pytest.mark.parametrize("module", ["tasks.multimodal_asr", "multimodal_asr.task",
                                    "tasks.multimodal_asr.task.x", "src.multimodal_asr.task", "tasks..task",
                                    "tasks.multimodal_asr."])
def test_a_module_of_another_shape_is_refused(module):
    with pytest.raises(ValueError, match="is not a task module: a task module is tasks.<directory>.task"):
        catalog_task(module)


@pytest.mark.parametrize("module", ["tasks.multitenant_slo.prompts", "tasks.deepresearch.research_loop",
                                    "tasks.multitenant_slo.removed_requests"])
def test_another_module_of_a_task_directory_is_refused(module):
    with pytest.raises(ValueError, match=re.escape(f"{module!r} is not a task module")):
        catalog_task(module)


def test_each_task_is_defined_by_its_directorys_task_module():
    assert task_module("multitenant_slo") == "tasks.multitenant_slo.task"
    assert task_module("deepresearch") == "tasks.deepresearch.task"


def test_multimodal_tts_declares_its_baseline_alone():
    """TTS declares no calibration member, since nothing grades or rerolls on one, and no reference
    image: its reference engine runs on its harness image."""
    images = task_images("multimodal_tts")
    assert set(images.measured) == {"baseline"}
    assert images.reference == images.harness == "s-a-tf-4-57-3-qwen-tts"


def test_a_tasks_images_are_read_from_the_checkout_whatever_the_import_path_holds(tmp_path):
    # As a console script runs: from another directory, with no checkout on the import path.
    code = "from hls.recipes import catalog; print(catalog.task_images('sample_task').sandbox)"
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True,
                         check=True, timeout=120).stdout
    assert out.split() == ["s-a-tf-5-13-1-agent"]
