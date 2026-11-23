"""The ``.def`` every catalog image renders to, and its cache key, stay as recorded.

``recipe_key`` hashes the rendered ``.def``, so a change to the renderer that changes one byte of
an image's text moves its key, and the store does not find the images built and measured under
the old key. ``golden/def_keys.json`` records, for every image the catalog declares and
for the test images that reach the renderer's other branches, the sha256 of its ``.def``
rendered with fixed stand-ins for the parent's file and the lock directory, and its key. A
change that is meant to move keys updates the file in the same commit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hls.recipes import catalog, pins
from hls.recipes.model import BaseImage, CapImage, ImageDecl, LayerImage, NodePin, UvPin, parent_of
from hls.recipes.render import recipe_key, render_def

GOLDEN = Path(__file__).with_name("golden") / "def_keys.json"
FIXTURES = Path(__file__).with_name("fixtures")
# Stand-ins for the parent's file and the lock directory, so the text does not depend on the checkout.
PARENT_STAND_IN = Path("/hls/parent.sif")
LOCKS_STAND_IN = "/hls/locks"

_CUDA = "nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:" + "7" * 64
_UV = UvPin(version="0.11.21", sha256="8c88519b0ef0af9801fcdee419bbb12116bd9e6b18e162ae093c932d8b264050")
_NSIGHT = "cuda-nsight-systems-13-0=2025.3.2.474-1"
#: Test images that reach what the catalog's do not: a base without NVIDIA packages, with other
#: variables, and a layer with a source build.
TEST_IMAGES: list[ImageDecl] = [
    BaseImage(name="nvidia-base", cuda_ref=_CUDA, snapshot="20261001T000000Z", apt_lock="apt-base.txt",
              nvidia_apt={_NSIGHT: hashlib.sha256(f"deb {_NSIGHT}".encode()).hexdigest()},
              nvidia_repo="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/",
              pip_lock="pip-base.txt", constraints=["torch==2.11.0+cu130", "torchaudio==2.11.0+cu130"],
              env={"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}),
    BaseImage(name="plain-base", cuda_ref=_CUDA, snapshot="20261001T000000Z", apt_lock="apt-base.txt",
              pip_lock="pip-base.txt", constraints=["torch==2.11.0+cu130", "torchaudio==2.11.0+cu130"],
              env={"PYTHONUNBUFFERED": "1"}),
    LayerImage(name="wheels-layer", parent="plain-base", pip_lock="pip-layer.txt"),
    LayerImage(name="sdist-layer", parent="nvidia-base", pip_lock="pip-sdist.txt",
               source_builds={"sox==1.5.0": ["setuptools"]}),
    CapImage(name="test-agent", parent="wheels-layer", cap="agent", apt_lock="apt-agent.txt",
             npm={"@anthropic-ai/claude-code": "2.1.251", "@openai/codex": "0.146.0"}, npm_lock="npm-agent.json",
             node=NodePin(version="22.23.3", sha256="1" * 64)),
]


def _recorded(decls: list[ImageDecl], locks: Path, uv: UvPin) -> dict[str, dict[str, str]]:
    """Each image of *decls*, parents first, to the sha256 of its stand-in ``.def`` and its key."""
    out: dict[str, dict[str, str]] = {}
    for decl in decls:
        parent = parent_of(decl)
        key = recipe_key(decl, parent_key=out[parent]["key"] if parent else None, locks_dir=locks, uv=uv)
        text = render_def(decl, parent_sif=PARENT_STAND_IN if parent else None, locks=locks, uv=uv)
        text = text.replace(str(locks), LOCKS_STAND_IN)
        out[decl.name] = {"def_sha256": hashlib.sha256(text.encode()).hexdigest(), "key": key}
    return out


def current() -> dict[str, dict[str, dict[str, str]]]:
    """What ``GOLDEN`` holds when nothing has moved."""
    return {"catalog": _recorded(catalog.shared_decls(), pins.LOCKS_DIR, pins.UV),
            "test": _recorded(TEST_IMAGES, FIXTURES, _UV)}


def test_every_catalog_image_renders_the_recorded_def_and_key():
    golden = json.loads(GOLDEN.read_text())
    now = current()
    assert set(now["catalog"]) == {d.name for d in catalog.shared_decls()}
    moved = {group: {name: entry for name, entry in now[group].items() if golden[group].get(name) != entry}
             for group in now}
    assert golden == now, f"rendered .def or key moved: {json.dumps(moved, indent=2, sort_keys=True)}"
