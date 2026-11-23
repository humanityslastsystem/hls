"""The shared images, and each task's images as its module declares them.

The bases S-A (torch 2.11, the five comparability tasks) and S0 (torch 2.13, the forward line for
new tasks) carry everything tasks share. A layer adds a task line's pins that cannot share a base,
today a transformers version: 5.13.1 for multitenant_slo, tab_completion and sample_task, 4.57.3
with qwen-tts for TTS, 5.9.0 for ASR and omni, and 5.17.0 on S0, which no task builds on yet. The
agent cap on a layer is the unsealed image an agent works in. The ``cuda`` base holds no pip lock:
the serving stacks build on it, each a task's own server layer or a shared one
(``SERVER_LAYERS``). A task on a layer has three built images:
- ``sandbox``: base, layer and agent cap, where the agent builds;
- ``submission_base``: base and layer, which the package stage adds the contestant's wheels
  to, so sandbox and submission share base and layer;
- ``harness``: base and layer, for harness roles (reference engine, driver) whose hls code is
  bound read-only from the checkout at deploy time.

A task declares its images in its module, ``IMAGES = ImageSet(...)``: the layer it builds
on, a shared one or its own, and the images its servers run on, each a server layer (the task's
own, whose locks and files are in the task's directory, or a shared one). Those are its measured
baseline and how it starts (``Served``: the entrypoint and runtime variables are given at deploy
time, never baked), the image its reference engine runs on, which is the harness image when it
declares none, and each calibration member's image; and the task's own images that are none of
those but that they build on: a layer, or a vendor's image pulled by digest (a registry image),
which only its server layers build on. A task's own layer gets the agent cap every shared layer
gets. A submission base on a base that holds no torch holds no constraints to apply, so its
package installs with none (``TaskImages.submission_constraints``).

The catalog knows a task by its directory under ``tasks/``, not by its ``TaskSpec.id``;
``tasks.<directory>.task`` defines the task's TASK and its IMAGES (``task_module``), and
``catalog_task`` turns that module into its directory and refuses any other module."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from hls.recipes import pins
from hls.recipes.model import (CONSTRAINTS, BaseImage, CapImage, Fetched, ImageDecl, LayerImage, RegistryImage,
                               order, parent_of)
from hls.recipes.render import TASKS_DIR, check_environments
from hls.taskref import first_on_import_path

#: Variables of the bases' environment.
BASE_ENV = {"HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1", "UV_NO_CACHE": "1",
            "VLLM_DO_NOT_TRACK": "1"}
# A calibration member's engine name, which names its role ``member-<engine>``.
_ENGINE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


def _lines(lock: str) -> list[str]:
    """The non-comment lines of a lock file."""
    text = (pins.LOCKS_DIR / lock).read_text()
    return [s for line in text.splitlines() if (s := line.strip()) and not s.startswith("#")]


def _base(name: str, short: str | None) -> BaseImage:
    pip = {} if short is None else {"pip_lock": f"base-{short}.txt", "constraints": _lines(f"constraints-{short}.txt")}
    return BaseImage(name=name, cuda_ref=pins.CUDA_BASE, snapshot=pins.SNAPSHOT, apt_lock="apt-base.txt",
                     nvidia_apt=dict(line.split(" ", 1) for line in _lines("apt-nvidia.txt")),
                     nvidia_repo=pins.NVIDIA_REPO, env=BASE_ENV, **pip)


#: The bases, by name. ``cuda`` installs no pip lock and holds no torch: the serving stacks a task
#: declares for itself (vLLM, SGLang) pin their own torch and their own versions of packages S-A
#: holds, so they build on it.
BASES: dict[str, BaseImage] = {"s-a": _base("s-a", "s-a"), "s0": _base("s0", "s0"), "cuda": _base("cuda", None)}
#: The layers, by name: each adds one task line's pins to a base. The TTS layer builds sox 1.5.0
#: from its source distribution (it has no wheel), against S-A's setuptools.
LAYERS: dict[str, LayerImage] = {
    layer.name: layer for layer in (
        LayerImage(name="s-a-tf-5-13-1", parent="s-a", pip_lock="layer-tf-5-13-1.txt"),
        LayerImage(name="s-a-tf-4-57-3-qwen-tts", parent="s-a", pip_lock="layer-tf-4-57-3-qwen-tts.txt",
                   source_builds={"sox==1.5.0": ["setuptools"]}),
        LayerImage(name="s-a-tf-5-9-0", parent="s-a", pip_lock="layer-tf-5-9-0.txt"),
        LayerImage(name="s0-tf-5-17-0", parent="s0", pip_lock="layer-tf-5-17-0.txt"),
    )
}
#: The agent CLIs every agent cap installs, at the versions ``pins`` holds.
AGENT_CLIS = {"@anthropic-ai/claude-code": pins.CLAUDE_CODE_VERSION, "@openai/codex": pins.CODEX_VERSION}


def agent_cap(layer: str) -> CapImage:
    """The agent cap on layer *layer*: ``<layer>-agent``, with the CLIs, Node and apt packages
    every agent cap holds."""
    return CapImage(name=f"{layer}-agent", parent=layer, cap="agent", apt_lock="apt-agent.txt", npm=AGENT_CLIS,
                    npm_lock="npm-agent.json", node=pins.NODE)


#: The agent cap on each layer, by the layer's name.
AGENT_CAPS: dict[str, CapImage] = {name: agent_cap(name) for name in LAYERS}
#: The variables every server image on the ``cuda`` base adds to the base's (``BASE_ENV`` and the
#: venv's): uv's torch backend, and the GPU architecture its JIT kernels compile for, over which
#: the roles that start it set their GPU's.
SERVER_ENV = {"UV_TORCH_BACKEND": "cu130", "TORCH_CUDA_ARCH_LIST": "9.0"}
#: FlashInfer's prebuilt kernels, by FlashInfer version: what a server layer whose pip lock holds
#: that flashinfer-python names as its ``fetched`` (an SGLang calibration member does), setting
#: ``FLASHINFER_CUBIN_DIR`` at their directory in its environment.
#: FlashInfer downloads a kernel when it first loads one that directory lacks; a run downloads
#: nothing, so the image holds every file ``python -m flashinfer download-cubin`` fetches. They
#: come from the ``flashinfer_cubin`` wheel of FlashInfer's release, which holds exactly those
#: files and which PyPI does not serve at these versions. The wheel is unpacked, not installed,
#: so FlashInfer finds the files through the variable. To add a version, list the wheel's
#: ``flashinfer_cubin/cubins/`` with ``sha256sum`` into
#: ``locks/fetch-flashinfer-cubins-<version>.txt``, and check its ``checksums.txt`` lines against
#: the digests that version's ``flashinfer/artifacts.py`` names.
FLASHINFER_CUBINS: dict[str, Fetched] = {
    "0.6.14": Fetched(
        url=("https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.14/"
             "flashinfer_cubin-0.6.14-py3-none-any.whl"),
        sha256="7bbed9f3851b59f3f6f6cb344810775bbce8a1c012ecf9a502bebd91cfb6433e",
        member="flashinfer_cubin/cubins", lock="fetch-flashinfer-cubins-0.6.14.txt",
        into="/opt/flashinfer-cubins", variable="FLASHINFER_CUBIN_DIR",
        loader="flashinfer.jit.cubin_loader:get_artifact", refuse_env="FLASHINFER_NO_DOWNLOAD"),
    # This release's wheel also holds an empty .lock file beside each kernel.
    "0.6.18": Fetched(
        url=("https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18/"
             "flashinfer_cubin-0.6.18-py3-none-any.whl"),
        sha256="2dd65c0fcfc6bc44c67f148530de5372979c2e3d260e47935730f94156d4d873",
        member="flashinfer_cubin/cubins", lock="fetch-flashinfer-cubins-0.6.18.txt",
        into="/opt/flashinfer-cubins", variable="FLASHINFER_CUBIN_DIR",
        loader="flashinfer.jit.cubin_loader:get_artifact", refuse_env="FLASHINFER_NO_DOWNLOAD",
        dropped=".lock"),
}
#: The shared server layers, by name: what several tasks' server images build on, or run as
#: they are. No agent cap builds on one, and no task builds its sandbox on one.
#: ``cuda-setuptools`` holds setuptools alone, the build backend of the source builds of
#: server layers on ``cuda``.
SERVER_LAYERS: dict[str, LayerImage] = {
    layer.name: layer for layer in (
        LayerImage(name="cuda-setuptools", parent="cuda", pip_lock="layer-cuda-setuptools.txt", server=True),
    )
}
#: Every shared image by name.
SHARED: dict[str, ImageDecl] = {**BASES, **LAYERS, **{cap.name: cap for cap in AGENT_CAPS.values()},
                                **SERVER_LAYERS}


def _engine(value: str) -> str:
    if not _ENGINE.fullmatch(value):
        raise ValueError(f"{value!r} is not an engine name (lowercase words joined by hyphens)")
    return value


class Served(BaseModel):
    """A task's measured baseline: its image and how its server starts there, which the deploy
    stage gives it rather than the image holding it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The server layer the server runs on.
    image: LayerImage
    #: The command the server starts with, before the deploy stage's ``--host``, ``--port`` and
    #: the task's ``Baseline.serve_command``.
    entrypoint: list[str]
    #: Variables the server starts with, below the deploy stage's own.
    runtime_env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.entrypoint:
            raise ValueError("a served image's entrypoint is a command, not an empty list")
        return self


class ImageSet(BaseModel):
    """What a task module declares of its images (``IMAGES``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The layer the task's sandbox, submission base and harness build on: one of ``LAYERS``, or
    #: one of the task's own ``images`` that is no server layer. Its agent cap is the sandbox.
    layer: str
    #: The task's own images that are no role: the layer it builds on, when it is its own, and
    #: the layers and registry images its role images build on.
    images: list[Annotated[LayerImage | RegistryImage, Field(discriminator="kind")]] = Field(default_factory=list)
    #: Whether harness roles run on the task's reference image rather than on its layer.
    harness_on_reference: bool = False
    #: Whether the package stage adds the contestant's wheels to the sandbox, the agent cap the
    #: agent built in, rather than to the layer under it.
    package_on_sandbox: bool = False
    #: The task's measured baseline; None for a task that measures none.
    baseline: Served | None = None
    #: The server layer the task's reference engine runs on; None for the harness image.
    reference: LayerImage | None = None
    #: Each calibration member's server layer, by its engine's name.
    members: dict[Annotated[str, AfterValidator(_engine)], LayerImage] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        own = {d.name: d for d in self.images}
        if len(own) != len(self.images):
            raise ValueError(f"an image is named twice in images: {[d.name for d in self.images]}")
        if self.layer not in LAYERS:
            layer = own.get(self.layer)
            if not isinstance(layer, LayerImage) or layer.server:
                raise ValueError(f"layer {self.layer!r} is not one of {sorted(LAYERS)}, nor one of the task's own "
                                 "images that is no server layer")
            declared = {**own, **{d.name: d for d in self.roles().values()}}
            below: str | None = layer.parent
            while below is not None:
                image = declared.get(below) or SHARED.get(below)
                if image is None:
                    break
                if isinstance(image, LayerImage) and image.server:
                    raise ValueError(f"layer {self.layer!r} builds on server image {below}: no agent cap or "
                                     "submission builds on a server image")
                below = parent_of(image)
        if self.harness_on_reference and self.reference is None:
            raise ValueError("harness roles run on the task's reference image, and it declares none")
        for decl in self.images:
            if isinstance(decl, LayerImage) and decl.owner is None:
                raise ValueError(f"image {decl.name} of images names no owner: a task's own image keeps its locks "
                                 "in the task's directory")
            if decl.name in SHARED:
                raise ValueError(f"image {decl.name} of images has the name of a shared image")
        named: dict[str, ImageDecl] = dict(own)
        for role, decl in self.roles().items():
            if not decl.server:
                raise ValueError(f"the {role} image {decl.name} is no server layer (server=True): servers run on it")
            if decl.name in SHARED and SERVER_LAYERS.get(decl.name) != decl:
                raise ValueError(f"the {role} image {decl.name} has the name of a shared image and is no shared "
                                 "server layer")
            if named.setdefault(decl.name, decl) != decl:
                raise ValueError(f"two different images are named {decl.name}")
        return self

    def roles(self) -> dict[str, LayerImage]:
        """Each image the task declares for itself by its role: ``baseline``, ``reference`` and
        ``member-<engine>``."""
        out: dict[str, LayerImage] = {}
        if self.baseline is not None:
            out["baseline"] = self.baseline.image
        if self.reference is not None:
            out["reference"] = self.reference
        out.update({f"member-{engine}": decl for engine, decl in sorted(self.members.items())})
        return out


class TaskImages(BaseModel):
    """A task's images as its runs use them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task's directory under ``tasks/``, the catalog's name for it.
    task: str
    #: Name of the image the agent builds in.
    sandbox: str
    #: Name of the image the package stage adds the contestant's wheels to.
    submission_base: str
    #: The constraints file every install into the submission base applies, which the package's
    #: offline install applies too: ``model.CONSTRAINTS``, which every base with a pip lock writes,
    #: for a submission base built on one; None for one on a base that holds nothing.
    submission_constraints: str | None
    #: Name of the image harness roles run on.
    harness: str
    #: Name of the image the verify role runs on: the task's reference image, else the harness.
    reference: str
    #: Each image the task declares for itself, by role (``ImageSet.roles``), to its name.
    measured: dict[str, str]
    #: How the measured baseline starts; None for a task that measures none.
    baseline: Served | None
    #: Every image above, parents before children.
    decls: list[ImageDecl] = Field(default_factory=list)


def _checked_dir(task: str) -> None:
    """Raise ValueError unless *task* is a directory under ``tasks/`` holding a task module."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", task) or not (TASKS_DIR / task / "task.py").is_file():
        raise ValueError(f"no task {task!r}: a task is a directory under tasks/ holding task.py; there are "
                         f"{task_keys()}")


def task_module(task: str) -> str:
    """The module that defines task *task*'s TASK and IMAGES, *task* being its directory under
    ``tasks/``, as a dotted name: ``tasks.<task>.task``. Raise ValueError for a directory that is
    no task."""
    _checked_dir(task)
    return f"tasks.{task}.task"


def task_keys(tasks_dir: Path = TASKS_DIR) -> list[str]:
    """Every task directory of *tasks_dir*, the checkout's ``tasks/``, sorted: each directory that
    holds a package and its ``task.py``; none for a checkout that has no ``tasks/``."""
    if not tasks_dir.is_dir():
        return []
    return sorted(p.name for p in tasks_dir.iterdir() if (p / "__init__.py").is_file() and (p / "task.py").is_file())


def catalog_task(task_module: str) -> str:
    """The catalog's name of the task whose module is *task_module*: the directory ``<dir>`` of
    ``tasks.<dir>.task``. The catalog is keyed by that directory, the name ``hls images build``
    takes, while a task's ``TaskSpec.id`` names its plans and store layouts and may differ:
    ``multimodal-asr`` lives in ``tasks/multimodal_asr``. Another module of the directory is no
    task module: it would key the task's instances, budget and credit under a TASK that is not the
    task's. Raise ValueError for a module of another shape and for a directory that is no task."""
    parts = task_module.split(".")
    if len(parts) != 3 or parts[0] != "tasks" or not parts[1] or parts[2] != "task":
        raise ValueError(f"{task_module!r} is not a task module: a task module is tasks.<directory>.task, "
                         "the module of tasks/<directory>/task.py")
    _checked_dir(parts[1])
    return parts[1]


def image_set(task: str) -> ImageSet:
    """The images task *task*'s module declares (``IMAGES``), imported from the checkout that holds
    ``TASKS_DIR``, whatever the import path held. Raise ValueError for a directory that is no task
    and a module that declares no ``ImageSet``."""
    module = task_module(task)
    with first_on_import_path(TASKS_DIR.parent):
        declared = getattr(importlib.import_module(module), "IMAGES", None)
    if not isinstance(declared, ImageSet):
        raise ValueError(f"{module} declares no IMAGES = ImageSet(...): a task module declares the images it runs on")
    return declared


def shared_decls() -> list[ImageDecl]:
    """Every base, layer and agent cap, parents first. ``hls images build`` builds only those a
    task builds on (``task_images``)."""
    return order(SHARED.values())


def held_versions(images: TaskImages) -> dict[str, str] | None:
    """The packages the base under *images*' sandbox holds, name to version in the base's order,
    as its constraints pin them; None for a sandbox whose package installs under no constraints
    file (``submission_constraints``): one on a base that holds nothing."""
    if images.submission_constraints is None:
        return None
    by_name = {d.name: d for d in images.decls}
    decl: ImageDecl = by_name[images.sandbox]
    while not isinstance(decl, BaseImage):
        parent = parent_of(decl)
        assert parent is not None
        decl = by_name[parent]
    return {name: version for name, _, version in (pin.partition("==") for pin in decl.constraints)}


def _chain(name: str, own: dict[str, ImageDecl]) -> list[ImageDecl]:
    """Image *name*, one of the shared images or of *own*, and every image it builds on."""
    by_name = {**SHARED, **own}
    out: list[ImageDecl] = []
    current: str | None = name
    while current is not None:
        decl = by_name.get(current)
        if decl is None:
            raise ValueError(f"image {out[-1].name} builds on {current}, which is no shared image and none the "
                             "task declares")
        out.append(decl)
        current = parent_of(decl)
    return out


def task_images(task: str) -> TaskImages:
    """Task *task*'s images, *task* being its directory under ``tasks/``, as its module
    declares them (``image_set``). Raise ValueError for a directory that is no task, a module that
    declares no images, a task's own layer that names another task as its owner, an image that
    builds on one neither shared nor declared, a layer for the sandbox that builds on no base, and
    one of the task's own images that no image of the task builds on."""
    declared = image_set(task)
    roles = declared.roles()
    by_name: dict[str, ImageDecl] = {d.name: d for d in [*declared.images, *roles.values()]
                                     if d.name not in SERVER_LAYERS}
    for decl in by_name.values():
        if isinstance(decl, LayerImage) and decl.owner != task:
            raise ValueError(f"image {decl.name} of {task} keeps its locks under tasks/{decl.owner}, not tasks/{task}")
    layer = declared.layer
    cap = AGENT_CAPS.get(layer) or agent_cap(layer)
    if layer not in LAYERS and (cap.name in SHARED or cap.name in by_name):
        raise ValueError(f"the agent cap of {task}'s layer would be named {cap.name}, which another image is")
    by_name[cap.name] = cap
    sandbox = cap.name
    submission_base = sandbox if declared.package_on_sandbox else layer
    harness = roles["reference"].name if declared.harness_on_reference else layer
    chains = [_chain(sandbox, by_name)]
    root = chains[0][-1]
    if not isinstance(root, BaseImage):
        raise ValueError(f"{task}'s layer {layer} builds on {root.kind} image {root.name}: a sandbox builds on "
                         "a base")
    constraints = CONSTRAINTS if root.pip_lock is not None else None
    chains += [_chain(decl.name, by_name) for decl in roles.values()]
    decls = order({d.name: d for chain in chains for d in chain}.values())
    check_environments(decls)
    unused = sorted(d.name for d in declared.images if d.name not in {x.name for x in decls})
    if unused:
        raise ValueError(f"{task} declares images {unused}, which none of its images builds on")
    reference = declared.reference.name if declared.reference is not None else harness
    return TaskImages(task=task, sandbox=sandbox, submission_base=submission_base,
                      submission_constraints=constraints, harness=harness, reference=reference,
                      measured={role: decl.name for role, decl in roles.items()}, baseline=declared.baseline,
                      decls=decls)
