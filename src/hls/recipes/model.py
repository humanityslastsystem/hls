"""Image declarations: what the images stage builds, and from what.

Four kinds:
- a base starts from the CUDA devel image pinned by digest, installs apt packages pinned by
  exact version from a dated snapshot.ubuntu.com archive, and makes the venv; S-A and S0 then
  install a hashed pip lock and hold torch with a constraints file every later install sees,
  while the accelerated base installs no pip lock and holds nothing, for serving stacks that
  bring their own torch;
- a layer adds pins (a pip lock, an apt lock or both) and files on top of another image; its
  pip packages install from wheels, but for the source builds it names, which build
  without isolation against the backends the images below it installed from their hashed locks.
  A task's own layer keeps its locks, and the files it copies in, in the task's directory
  (``owner``); a server layer is one only servers run on (``server``): a task's baseline,
  reference engine or calibration member, or a shared stack they build on. A server layer may
  also hold the files one of its packages would download when it first needs them
  (``Fetched``): the build fetches their archive from its URL, checks its sha256 and each
  file's, and puts them where the package looks;
- a cap adds an HLS-owned set on top of another image; today only the agent cap, with the
  agent CLIs pinned by exact version and installed from a hashed npm lock, on a Node release
  pinned by the digest of its tarball;
- a registry image is a vendor's image pulled by digest and used as it is, for a serving stack
  that is published as an image alone. Only server layers build on it: their apt locks install
  from the snapshot it declares and their pip locks into the interpreter it declares, and such a
  lock names every package it adds, since the vendor's own packages are in no lock.

Lock files are named, not inlined: their bytes are part of an image's key (``render``). The
shared images keep theirs in one directory; a task's layer keeps its own under its task."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal, Self

from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, field_validator,
                      model_serializer, model_validator)

from hls.store.digest import Sha256Hex


_NAME = re.compile(r"[a-z][a-z0-9-]*")
_LOCK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_APT_PIN = re.compile(r"[a-z0-9][a-z0-9+.-]*=[A-Za-z0-9.+~:-]+")
_PIP_PIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9.+!_-]+")
_PROJECT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SNAPSHOT = re.compile(r"\d{8}T\d{6}Z")
# A registry reference that keeps its tag and pins the digest: "<repo>:<major>.<minor>...@sha256:<hex>".
_CUDA_REF = re.compile(r"(?P<repo>[a-z0-9][a-z0-9./_-]*):(?P<major>\d+)\.(?P<minor>\d+)[A-Za-z0-9._-]*"
                       r"@(?P<digest>sha256:[0-9a-f]{64})")
_REPO_URL = re.compile(r"https://[a-z0-9][a-z0-9.-]*(/[A-Za-z0-9._~+-]+)*/")
# A file-name suffix: a dot and letters or digits.
_SUFFIX = re.compile(r"\.[A-Za-z0-9]+")
# A registry reference by digest alone: "<repo>@sha256:<hex>".
_REGISTRY_REF = re.compile(r"[a-z0-9][a-z0-9./_-]*@sha256:[0-9a-f]{64}")
# Where a .deb is downloaded from: an https URL of plain segments whose last ends in .deb.
_DEB_URL = re.compile(r"https://[a-z0-9][a-z0-9.-]*(/[A-Za-z0-9._~+-]+)*/[A-Za-z0-9][A-Za-z0-9._~+-]*\.deb")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NPM_NAME = re.compile(r"(@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*")
_NPM_VERSION = re.compile(r"\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?")
_RELEASE = re.compile(r"\d+\.\d+\.\d+")
# Nsight Systems' package names the CUDA major and minor it belongs to.
_NSIGHT = re.compile(r"cuda-nsight-systems-(\d+)-(\d+)")
# A task's directory under tasks/, a Python package name.
_TASK_DIR = re.compile(r"[a-z][a-z0-9_]*")
# A file of a task's directory, relative to it: plain segments, none of them . or .. or hidden.
_TASK_FILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*(/[A-Za-z0-9_][A-Za-z0-9._-]*)*")
# A value a layer's environment exports as it stands: it may name another variable ($PATH), never
# hold a space, a quote or anything else the shell would read as more than one word.
_ENV_VALUE = re.compile(r"[A-Za-z0-9_./:$+,=@%-]*")
# Where an image keeps a copied file: an absolute path of plain segments, none of them . or ..
_IMAGE_PATH = re.compile(r"(/(?!\.\.?(?:/|$))[A-Za-z0-9_.][A-Za-z0-9._-]*)+")
# The https URL of one file: plain path segments, no query.
_FILE_URL = re.compile(r"https://[a-z0-9][a-z0-9.-]*(/[A-Za-z0-9._~+-]+)+")
# A function of an installed package, ``<dotted module>:<function>``.
_FUNCTION = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*")

#: The image's system interpreter, Ubuntu's: the base makes its venv from it, and the harness's
#: programs in the sandbox run on it, where the contestant cannot change it (the in-sandbox
#: ``hls.agent.exec_protocol.SANDBOX_PYTHON`` keeps a copy, since it imports nothing of hls).
SYSTEM_PYTHON = "/usr/bin/python3"
#: The base's Python environment.
VENV = "/opt/venv"
#: The constraints file of every base: the packages held at one version.
CONSTRAINTS = "/opt/hls/constraints.txt"
#: The variables every base's environment sets itself, in this order and before its declared ones:
#: the venv first on PATH, and the constraints file for uv and pip. A declared variable of the same
#: name would replace the venv's PATH or the torch hold, so a base refuses one.
BASE_ENVIRONMENT = {"PATH": f"{VENV}/bin:$PATH", "VIRTUAL_ENV": VENV, "UV_CONSTRAINT": CONSTRAINTS,
                    "PIP_CONSTRAINT": CONSTRAINTS}


def _matching(pattern: re.Pattern[str], what: str) -> AfterValidator:
    def check(value: str) -> str:
        if not pattern.fullmatch(value):
            raise ValueError(f"{value!r} is not {what}")
        return value

    return AfterValidator(check)


#: An image name: lowercase letters, digits and dashes, starting with a letter. It begins the
#: image's cache key, so it is one store key segment.
ImageName = Annotated[str, _matching(_NAME, "an image name ([a-z][a-z0-9-]*)")]
#: A lock file's name in the catalog's lock directory, one path segment.
LockName = Annotated[str, _matching(_LOCK, "a lock file name (one path segment)")]
#: An apt package pinned by exact version, ``name=version``.
AptPin = Annotated[str, _matching(_APT_PIN, "an apt pin name=version")]
#: A pip package held at one version, ``name==version``.
PipPin = Annotated[str, _matching(_PIP_PIN, "a pip pin name==version")]
#: A Python project's name, as a pip lock names it.
ProjectName = Annotated[str, _matching(_PROJECT, "a project name")]
#: A task's directory under ``tasks/``.
TaskDir = Annotated[str, _matching(_TASK_DIR, "a task directory ([a-z][a-z0-9_]*)")]
#: A file of a task's directory, by its path relative to the directory.
TaskFile = Annotated[str, _matching(_TASK_FILE, "a path relative to the task's directory")]
#: Where a .deb is downloaded from.
DebUrl = Annotated[str, _matching(_DEB_URL, "an https URL of a .deb file")]
#: A snapshot.ubuntu.com timestamp, ``YYYYMMDDTHHMMSSZ``.
Snapshot = Annotated[str, _matching(_SNAPSHOT, "a snapshot timestamp YYYYMMDDTHHMMSSZ")]
#: An absolute path in an image.
ImagePath = Annotated[str, _matching(_IMAGE_PATH, "an absolute path of plain segments, none . or ..")]
#: The images' own directory: their locks, constraints file and agent CLIs. No copied file lands
#: in it.
HLS_ROOT = "/opt/hls"
#: An environment variable's name.
EnvName = Annotated[str, _matching(_ENV_NAME, "a variable name")]


def pip_name(pin: str) -> str:
    """The normalized project name of pip pin *pin* (``name==version``): lowercase, with each run
    of ``-``, ``_`` and ``.`` one dash, as indexes and installers compare names."""
    return re.sub(r"[-_.]+", "-", pin.split("==", 1)[0]).lower()


class UvPin(BaseModel):
    """The uv release the bases install, checked by the digest of its release tarball."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Release version, ``major.minor.patch``.
    version: Annotated[str, _matching(_RELEASE, "a uv version major.minor.patch")]
    #: Hex SHA-256 of ``uv-x86_64-unknown-linux-gnu.tar.gz`` of that release.
    sha256: Sha256Hex


class NodePin(BaseModel):
    """The Node.js release the agent cap runs its CLIs on, checked by the digest of its release
    tarball. Ubuntu 24.04's own Node (18) is older than the CLIs need."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Release version, ``major.minor.patch``, without the ``v`` of its tag.
    version: Annotated[str, _matching(_RELEASE, "a Node version major.minor.patch")]
    #: Hex SHA-256 of ``node-v<version>-linux-x64.tar.gz`` of that release.
    sha256: Sha256Hex


class BaseImage(BaseModel):
    """A base: the CUDA devel image, the snapshot's apt packages, the venv and, but for a base
    whose children bring their own torch, a hashed pip lock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Declaration kind.
    kind: Literal["base"] = "base"
    #: Image name.
    name: ImageName
    #: The CUDA devel image, ``<repo>:<tag>@sha256:<digest>``: the digest pins it, the tag names
    #: its CUDA version.
    cuda_ref: Annotated[str, _matching(_CUDA_REF, "a registry reference with its tag, pinned by digest")]
    #: snapshot.ubuntu.com timestamp the Ubuntu archive is read at, ``YYYYMMDDTHHMMSSZ``.
    snapshot: Snapshot
    #: Lock file of the Ubuntu packages, one ``name=version`` per line.
    apt_lock: LockName
    #: Packages from NVIDIA's repository, which the snapshot does not hold (Nsight Systems): each
    #: pin to the hex SHA-256 of its .deb. The repository has no snapshot, so the build installs
    #: only .deb files it has checked against these.
    nvidia_apt: dict[AptPin, Sha256Hex] = Field(default_factory=dict)
    #: That repository's URL, ending in ``/``, the one source the build accepts beside the snapshot
    #: (with or without the trailing slash, as apt reads it); None for none, and then no ``nvidia_apt``.
    nvidia_repo: Annotated[str, _matching(_REPO_URL, "an https repository URL ending in /")] | None = None
    #: Hashed pip lock of the base's Python packages; None for a base that installs none, whose
    #: children each install their own torch.
    pip_lock: LockName | None = None
    #: Packages held at one version for every later install in this image and its children
    #: (torch, torchaudio, torchvision); empty for a base with no pip lock, which holds nothing.
    constraints: list[PipPin] = Field(default_factory=list)
    #: Variables of the image's environment, besides the venv and constraint variables the base
    #: sets itself; never ``TORCH_CUDA_ARCH_LIST``.
    env: dict[Annotated[str, _matching(_ENV_NAME, "a variable name")], str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.pip_lock is not None and not self.constraints:
            raise ValueError("a base with a pip lock must hold at least one package (torch) at one version")
        if self.pip_lock is None and self.constraints:
            raise ValueError("a base with no pip lock installs nothing to hold at one version")
        if self.apt_lock == self.pip_lock:
            raise ValueError("the apt lock and the pip lock must be different files")
        if self.nvidia_apt and self.nvidia_repo is None:
            raise ValueError("NVIDIA packages need the NVIDIA repository declared (nvidia_repo)")
        cuda = _CUDA_REF.fullmatch(self.cuda_ref)
        assert cuda is not None
        for pin in self.nvidia_apt:
            nsight = _NSIGHT.fullmatch(pin.split("=", 1)[0])
            if nsight and nsight.groups() != (cuda["major"], cuda["minor"]):
                raise ValueError(f"{pin} is not the Nsight Systems of CUDA {cuda['major']}.{cuda['minor']}, "
                                 f"which {self.cuda_ref} is")
        for name, value in self.env.items():
            # A line break of any kind would end the value's line in the .def, where a line starting
            # with % opens a section.
            if not value.isprintable():
                raise ValueError(f"env {name} holds a character that is not printable")
            if "{{" in value:
                raise ValueError(f"env {name} holds '{{{{', which Apptainer reads as a build argument")
            if name == "TORCH_CUDA_ARCH_LIST":
                raise ValueError("TORCH_CUDA_ARCH_LIST is never baked: roles that compile set it for "
                                 "their GPU")
            if name in BASE_ENVIRONMENT:
                raise ValueError(f"env {name} is one the base sets itself")
        return self

    @property
    def pull_ref(self) -> str:
        """The CUDA image as Apptainer pulls it, ``<repo>@sha256:<digest>``: Apptainer refuses a
        reference that carries both a tag and a digest, and the digest alone pins the bytes."""
        cuda = _CUDA_REF.fullmatch(self.cuda_ref)
        assert cuda is not None
        return f"{cuda['repo']}@{cuda['digest']}"


class Fetched(BaseModel):
    """Files a package of a server layer would download when it first needs them, which the build
    fetches instead and puts where the package looks, so a run downloads nothing.

    The build downloads the archive at ``url``, checks its ``sha256``, and unpacks its directory
    ``member`` as ``into``, less the by-products it names (``dropped``). It checks every file
    there against ``lock`` and that ``into`` holds no other file, and then calls ``loader`` on every file with ``variable`` naming ``into``, as the
    layer's environment bakes it, and ``refuse_env`` set: the build ends when the package would
    still download one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The archive that holds the files: the https URL of a zip file, which a wheel is.
    url: Annotated[str, _matching(_FILE_URL, "the https URL of a file")]
    #: Hex SHA-256 of that archive.
    sha256: Sha256Hex
    #: The directory of the archive whose files they are, by its path in the archive.
    member: Annotated[str, _matching(_TASK_FILE, "a relative path of plain segments")]
    #: Lock of the files, in the shared lock directory whichever task owns the layer: one
    #: ``<sha256>  <path>`` per line, as ``sha256sum`` writes and checks them, each path relative
    #: to ``member`` and to ``into``.
    lock: LockName
    #: The directory of the image the files are put in, under their paths; the build makes it.
    into: ImagePath
    #: The variable by which the package finds that directory. The layer's environment bakes it
    #: at ``into`` (``LayerImage.env``), and the build's check sets it so.
    variable: EnvName
    #: The package's own function that loads one of the files, called as ``function(path,
    #: sha256)``: ``<module>:<function>``. It returns the file's bytes, or nothing when it has
    #: none.
    loader: Annotated[str, _matching(_FUNCTION, "a function <module>:<function>")]
    #: The variable under which ``loader`` raises for a file it would download.
    refuse_env: EnvName
    #: The file-name suffix of files the archive's directory holds that are none of the
    #: package's: the empty lock file its build left beside each file (``.lock``). The build
    #: removes them before it checks the directory against ``lock``, which names none of them;
    #: None for an archive that holds none. A declaration without one is written without the
    #: field.
    dropped: Annotated[str, _matching(_SUFFIX, "a file-name suffix such as .lock")] | None = Field(
        default=None, exclude_if=lambda suffix: suffix is None)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        for root in (HLS_ROOT, VENV):
            if self.into == root or self.into.startswith(root + "/") or root.startswith(self.into + "/"):
                raise ValueError(f"fetched files in {self.into} would share a directory with {root}, which the "
                                 "image's own installs fill")
        if self.refuse_env == self.variable:
            raise ValueError(f"{self.variable} names the directory; another variable refuses downloads")
        return self


class LayerImage(BaseModel):
    """Pins on top of another image: a task line's shared layer, or an image a task declares for
    itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Declaration kind.
    kind: Literal["layer"] = "layer"
    #: Image name.
    name: ImageName
    #: The image it builds on.
    parent: ImageName
    #: Hashed pip lock of what it adds; None for none.
    pip_lock: LockName | None = None
    #: Lock of the apt packages it adds; None for none.
    apt_lock: LockName | None = None
    #: Packages no apt source of the image carries, each the URL of its .deb to the hex SHA-256
    #: of that file. The build installs each checked file beside the apt lock, which pins its
    #: dependencies. A layer with none is written without the field.
    debs: dict[DebUrl, Sha256Hex] = Field(default_factory=dict, exclude_if=lambda debs: not debs)
    #: Packages of its pip lock built from their source distributions, each at its locked version,
    #: to the build backends it needs (project names). They build without isolation, against the
    #: venv as the images below this one left it, so those backends must come from their hashed
    #: locks. Every other package installs from a wheel or not at all.
    source_builds: dict[PipPin, list[ProjectName]] = Field(default_factory=dict)
    #: The task directory under ``tasks/`` whose ``locks/`` holds this layer's lock files and
    #: which holds the files it copies in; None for a shared layer, whose locks are the catalog's.
    owner: TaskDir | None = None
    #: Files of the owner's directory the image holds: path in the image to path in the directory.
    #: Their bytes are part of the image's key.
    files: dict[ImagePath, TaskFile] = Field(default_factory=dict)
    #: Whether only servers run on the image (a task's baseline, its reference engine, a
    #: calibration member, or a shared stack they build on), never an agent or a submission: it
    #: may hold a serving framework the harness deny list names, and no agent cap or submission
    #: builds on it.
    server: bool = False
    #: Variables this layer adds to, or sets over, the environment of the image below, each
    #: exported in this order as it stands, after the variables of the images below it
    #: (``render.baked_environment``); None for none. What a layer may set is ruled by
    #: ``render.check_environments``. The roles that start the image set their own variables over
    #: it. A list of (name, value) pairs, given as a mapping or as pairs, so that its order
    #: survives every copy of the declaration, a plan's JSON with its sorted keys among them.
    env: list[tuple[Annotated[str, _matching(_ENV_NAME, "a variable name")],
                    Annotated[str, _matching(_ENV_VALUE, "a value with no space or quote")]]] | None = None
    #: Files one of the image's packages would download when it first needs them, fetched by the
    #: build; None for none. Only a server layer fetches: an inventory lists packages, not files,
    #: so nothing would judge what an agent's image fetched against the deny list.
    fetched: Fetched | None = None

    @field_validator("env", mode="before")
    @classmethod
    def _env_pairs(cls, value: object) -> object:
        return list(value.items()) if isinstance(value, dict) else value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.pip_lock is None and self.apt_lock is None and not self.files and self.fetched is None:
            raise ValueError("a layer adds a lock or a file")
        names = [name for name, _ in self.env or []]
        if len(set(names)) != len(names):
            raise ValueError(f"env names a variable twice: {names}")
        if self.fetched is not None and not self.server:
            raise ValueError("a layer that fetches files is a server layer (server=True)")
        if self.fetched is not None and self.fetched.lock in (self.pip_lock, self.apt_lock):
            raise ValueError("the lock of the fetched files and the layer's own locks must be different files")
        if self.fetched is not None and dict(self.env or []).get(self.fetched.variable) != self.fetched.into:
            raise ValueError(f"a layer that fetches files into {self.fetched.into} bakes {self.fetched.variable}="
                             f"{self.fetched.into} (env), where its package looks for them")
        if self.files and self.owner is None:
            raise ValueError("a layer that copies files is a task's own, and names its task (owner)")
        for target in self.files:
            if target == HLS_ROOT or target.startswith(HLS_ROOT + "/"):
                raise ValueError(f"file {target} would land in {HLS_ROOT}, the images' own directory")
        if self.pip_lock is not None and self.pip_lock == self.apt_lock:
            raise ValueError("the apt lock and the pip lock must be different files")
        if self.debs and self.apt_lock is None:
            raise ValueError("a layer that installs .deb files pins their dependencies in an apt lock, and "
                             "this layer has none")
        files = [url.rsplit("/", 1)[1] for url in self.debs]
        if len(set(files)) != len(files):
            raise ValueError(f"two .deb files of layer {self.name} have one file name: {sorted(files)}")
        if self.source_builds and self.pip_lock is None:
            raise ValueError("source builds are packages of a pip lock, and this layer has none")
        names = [pip_name(pin) for pin in self.source_builds]
        if len(set(names)) != len(names):
            raise ValueError(f"a source build is named twice in {list(self.source_builds)}")
        for pin, backends in self.source_builds.items():
            if not backends:
                raise ValueError(f"source build {pin} names no build backend")
        if self.parent == self.name:
            raise ValueError(f"layer {self.name} cannot be its own parent")
        return self

    @model_serializer(mode="wrap")
    def _dump(self, handler: SerializerFunctionWrapHandler) -> dict:
        """The layer's fields, ``fetched`` among them only when it fetches: a plan holds each
        image's declaration in full, and one that fetches nothing reads the same there whether
        or not its reader knows the field."""
        out: dict = handler(self)
        if out.get("fetched") is None:
            out.pop("fetched", None)
        return out


class CapImage(BaseModel):
    """An HLS-owned set on top of another image. The agent cap holds Node and its npm, the agent
    CLIs and tmux, and the directories the agent works in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Declaration kind.
    kind: Literal["cap"] = "cap"
    #: Image name.
    name: ImageName
    #: The image it builds on.
    parent: ImageName
    #: Which cap.
    cap: Literal["agent"]
    #: Lock of its apt packages.
    apt_lock: LockName
    #: The CLIs it installs, npm package name to exact version.
    npm: dict[Annotated[str, _matching(_NPM_NAME, "an npm package name")],
              Annotated[str, _matching(_NPM_VERSION, "an exact version")]]
    #: npm's ``package-lock.json`` of exactly those CLIs, with every package of their trees and
    #: the sha512 integrity hash of its tarball; ``npm ci`` installs that tree and nothing else.
    npm_lock: LockName
    #: The Node release the CLIs run on, installed from its tarball with the npm it ships, which
    #: installs them.
    node: NodePin

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not self.npm:
            raise ValueError("the agent cap installs at least one CLI")
        if self.npm_lock == self.apt_lock:
            raise ValueError("the apt lock and the npm lock must be different files")
        if self.parent == self.name:
            raise ValueError(f"cap {self.name} cannot be its own parent")
        return self


class RegistryImage(BaseModel):
    """A vendor's image as its registry serves it, pinned by digest and used as it is: nothing
    is installed into it. Only a task's server layers build on it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Declaration kind.
    kind: Literal["registry"] = "registry"
    #: Image name.
    name: ImageName
    #: The image, ``<repo>@sha256:<digest>``: the digest pins its bytes, so it carries no tag.
    ref: Annotated[str, _matching(_REGISTRY_REF, "a registry reference by digest, <repo>@sha256:<64 hex>")]
    #: The interpreter the image's Python packages live in, which a layer's pip lock installs
    #: into: the image has no venv of the bases' kind.
    python: ImagePath
    #: snapshot.ubuntu.com timestamp a layer's apt lock installs from: the layer's apt reads that
    #: snapshot alone, never the sources the image came with.
    snapshot: Snapshot


#: Any image declaration, told apart by ``kind``.
ImageDecl = Annotated[BaseImage | LayerImage | CapImage | RegistryImage, Field(discriminator="kind")]


def parent_of(decl: ImageDecl) -> str | None:
    """The name of the image *decl* builds on; None for a base or a registry image."""
    return decl.parent if isinstance(decl, LayerImage | CapImage) else None


def root_of(decl: ImageDecl, by_name: Mapping[str, ImageDecl]) -> ImageDecl:
    """The image *decl*'s chain of parents ends at, each found in *by_name*: *decl* itself when
    it builds on nothing. Raise KeyError for a parent *by_name* lacks."""
    while (parent := parent_of(decl)) is not None:
        decl = by_name[parent]
    return decl


def registry_of(decl: ImageDecl, by_name: Mapping[str, ImageDecl]) -> RegistryImage | None:
    """The registry image *decl*'s chain of parents ends at (``root_of``), which the renderer
    takes to render an image that builds on one; None for a registry image itself and for a chain
    that ends at any other image."""
    root = root_of(decl, by_name)
    return root if isinstance(root, RegistryImage) and root is not decl else None


def order(decls: Iterable[ImageDecl]) -> list[ImageDecl]:
    """*decls* with every image after the image it builds on, otherwise in the given order.
    Raise ValueError for a name declared twice, a parent that is not declared, or a cycle."""
    listed = list(decls)
    by_name: dict[str, ImageDecl] = {}
    for d in listed:
        if d.name in by_name:
            raise ValueError(f"image {d.name} is declared twice")
        by_name[d.name] = d
    out: list[ImageDecl] = []
    placed: set[str] = set()

    def place(d: ImageDecl, path: tuple[str, ...]) -> None:
        if d.name in placed:
            return
        if d.name in path:
            raise ValueError(f"images build on each other in a cycle: {' -> '.join((*path, d.name))}")
        parent = parent_of(d)
        if parent is not None:
            if parent not in by_name:
                raise ValueError(f"image {d.name} names an unknown parent {parent}")
            place(by_name[parent], (*path, d.name))
        placed.add(d.name)
        out.append(d)

    for d in listed:
        place(d, ())
    return out
