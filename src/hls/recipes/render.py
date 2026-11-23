"""An image declaration as an Apptainer ``.def``, and the cache key of what it builds.

A base bootstraps from its CUDA image by digest alone: Apptainer refuses a reference with both a
tag and a digest, so the tag is kept in a label. A registry image bootstraps from its digest and
has no other section: it is the vendor's image as pulled. A layer or a cap bootstraps from its
parent's ``.sif`` where the store holds it (``Bootstrap: localimage``). Every lock file is copied
in and installed from there, and its content is checked before anything renders:
- apt: the base points the Ubuntu sources at the dated snapshot and ends the
  build if any source is left that is neither the snapshot nor the declared NVIDIA repository.
  Every install names each package at its exact version, then checks with dpkg that each one is
  installed at that version. A layer or cap installs from the same snapshot, which its parent
  left configured. Each lock line must be ``name=version``, since it reaches apt-get's arguments.
  A layer's ``.deb`` files are downloaded into a fresh directory of the image's tree, checked
  against their sha256 and installed in the same ``apt-get install`` as its lock, which pins
  their dependencies; each must then be installed at the version its file names;
- pip: ``uv pip install --require-hashes --no-deps --only-binary :all:`` of a lock that is
  already resolved and hashed, so nothing is resolved at build time and an artifact whose hash
  differs is refused. An entry is ``name==version``, which the indexes serve, or, for a wheel
  neither index carries, ``name @ <https URL of the wheel>``, hashed like any other. A layer may name packages that exist only as source distributions; those
  alone build, without isolation (``--no-build-isolation-package``), so their build backends come
  from the venv, which hashed locks filled, and a backend missing there ends the build rather
  than being fetched (``check_source_builds`` refuses such a declaration before it builds). After
  every pip install, a check reads each held package's installed version against the base's
  constraints file and ends the build if a lock moved one (torch is held);
- on a registry image: the vendor's image is not a base's, so a layer whose chain starts at one
  (``model.registry_of``, which every caller passes as *registry*) configures what it uses.
  Its apt reads one sources file of its own, the registry image's snapshot of the Ubuntu release
  the apt locks are pinned for, and none of the sources the image came with; the build ends on an
  image of another release. Its pip lock installs into the interpreter the registry image
  declares, with the pinned uv fetched for the step and removed after it, and with none of the
  image's own ``UV_*`` and ``PIP_*`` variables. Nothing is held there, so nothing is checked;
- npm: the agent cap installs its CLIs with ``npm ci`` from a committed ``package-lock.json``
  that pins every package of their trees with the sha512 integrity hash of its tarball, and
  refuses one whose declared engines exclude the installed Node. That Node, with the npm it
  ships, comes from Node's release tarball, checked against its digest, into ``/usr/local``;
- fetched files: a server layer may name files one of its packages would download when it first
  needs them (``model.Fetched``). The build downloads their archive, a zip file, and unpacks
  it only once it matches its digest, then moves the archive's declared directory to the
  declared directory of the image. Their lock, from the shared lock directory, holds one
  ``<sha256>  <path>`` per file, each path relative to that directory and of plain segments:
  ``sha256sum`` checks every file against it, and the build ends if the directory holds
  anything the lock does not name. The build then calls the package's own loader on every
  file, with the variable that names the directory, as the layer's environment bakes it, and
  the one under which the loader raises rather than download, and ends if the package would
  still download one.

Every ``%post`` step is followed by its own status check, so a failed step ends the build:
``set -e`` alone exempts a command that fails in the middle of an ``&&``
list or behind a ``!``. The script never traces its commands (``set +x``). Rootless fakeroot
cannot run apt's download sandbox, so apt runs as the mapped root (``APT::Sandbox::User``); the
build itself never passes ``--ignore-fakeroot-command``. A build's ``/tmp`` is the
host's, so downloads go to a fresh directory of the image's own tree, and uv's and npm's caches
are removed after each install rather than kept in the image.

A base writes ``%environment`` and ``%runscript``, carrying the venv's PATH and the constraint
variables. A layer writes ``%environment`` only for the variables it adds or sets
(``LayerImage.env``), which the image's environment exports after those of the images below it:
a layer's environment adds to the one below and removes nothing; a cap writes none. A base never sets ``TORCH_CUDA_ARCH_LIST``: roles that
compile set it for the GPU they were granted, over whatever an image baked.

The build script (``post_script``), the files it copies in (``files_for``) and a base's
environment (``environment``) come from the declaration alone. ``render_def`` wraps them in a
``.def`` and ``hls.recipes.render_docker`` in a Dockerfile, so both runtimes run one script.

The cache key of an image is its name and 16 hex digits of a digest of the ``.def`` rendered
with fixed stand-ins for the parent's path and the lock directory, the bytes of every
file it copies in, and its parent's key. So every key is known before anything builds, it does
not depend on where the files are, and any change to what an image is made of gives
a new key."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NamedTuple
from urllib.parse import urlsplit

from pydantic import TypeAdapter, ValidationError

from hls.recipes import model
from hls.recipes.model import (
    BASE_ENVIRONMENT,
    CONSTRAINTS,
    VENV,
    AptPin,
    BaseImage,
    CapImage,
    Fetched,
    ImageDecl,
    LayerImage,
    NodePin,
    PipPin,
    RegistryImage,
    UvPin,
    order,
    parent_of,
    pip_name,
)

#: Where an image's lock files are copied to.
LOCK_ROOT = "/opt/hls/locks"
#: The checkout's task directories, which hold a task's own layers' locks (``locks/``) and the
#: files they copy in.
TASKS_DIR = Path(__file__).resolve().parents[3] / "tasks"
#: Where the agent cap's pinned Node lives. npm and the CLIs run on it; the image's own node and npm
#: stay Ubuntu's.
NODE_PREFIX = "/opt/hls/node"
#: npm's own install checks in the pinned Node, which decide which lock entries npm installs here.
NPM_INSTALL_CHECKS = f"{NODE_PREFIX}/lib/node_modules/npm/node_modules/npm-install-checks"
#: Where the agent cap installs its CLIs, from its ``package.json`` and ``package-lock.json``.
NPM_PREFIX = "/opt/hls/agent-cli"
#: The first index the pip locks were compiled against: PyTorch's CUDA 13.0 wheels.
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
#: The index the pip locks were compiled against after ``TORCH_INDEX``.
PYPI_INDEX = "https://pypi.org/simple"
#: The apt preferences file that keeps NVIDIA's repository from supplying any version.
NVIDIA_PREFERENCES = "/etc/apt/preferences.d/hls-nvidia"
#: Ubuntu's deb822 sources file in the CUDA devel image.
UBUNTU_SOURCES = "/etc/apt/sources.list.d/ubuntu.sources"
#: The pockets of the Ubuntu release the apt locks are pinned for, the release itself first, and
#: its components.
UBUNTU_DISTS = ("noble", "noble-updates", "noble-security")
UBUNTU_COMPONENTS = ("main", "restricted", "universe", "multiverse")
#: The one sources file the apt of a layer on a registry image reads: the snapshot's pockets.
SNAPSHOT_SOURCES = "/opt/hls/apt/sources.list"
#: The directory that apt reads further sources files from: one of the images' own, kept empty.
SNAPSHOT_SOURCE_PARTS = "/opt/hls/apt/sources.list.d"
#: The keyring Ubuntu's images verify the archive with.
UBUNTU_KEYRING = "/usr/share/keyrings/ubuntu-archive-keyring.gpg"
#: The check each step is followed by. Its own variable name, not a bare ``rc``.
STEP_CHECK = 'hls_rc=$?; [ "$hls_rc" -eq 0 ] || exit "$hls_rc"'
# Stand-ins for the parent's path and the lock directory when a key is computed.
_PARENT_STAND_IN = "/hls/parent.sif"
_LOCKS_STAND_IN = Path("/hls/locks")
_UV_TARBALL = "uv-x86_64-unknown-linux-gnu"
_APT_PIN = TypeAdapter(AptPin)
_PIP_PIN = TypeAdapter(PipPin)
_HASH_OPTION = re.compile(r"--hash=sha256:[0-9a-f]{64}")
# Where a lock entry's wheel is downloaded from when no index carries it: an https URL of plain
# segments whose last is the wheel's file name, "<distribution>-<version>-<tags>.whl".
_WHEEL_URL = re.compile(r"https://[a-z0-9][a-z0-9.-]*(?:/[A-Za-z0-9._~+-]+)*/"
                        r"(?P<dist>[A-Za-z0-9][A-Za-z0-9._]*)-(?P<version>[A-Za-z0-9.+!_]+)(?:-[A-Za-z0-9._]+){3,4}\.whl")
# A line of a fetch lock: a file's sha256 and its path, relative and of plain segments.
_FETCH_LINE = re.compile(r"[0-9a-f]{64}  (?P<path>[A-Za-z0-9_][A-Za-z0-9._-]*(/[A-Za-z0-9_][A-Za-z0-9._-]*)*)")
_INTEGRITY = re.compile(r"sha512-[A-Za-z0-9+/]{86}==")
_COMMAND = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_NPM_EXACT = re.compile(r"\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?")
_NPM_URL = re.compile(r"https://[A-Za-z0-9.-]+(/[A-Za-z0-9._~%@+-]+)+")
_NPM_SEGMENT = r"(@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*"
_NPM_PATH = re.compile(rf"node_modules/{_NPM_SEGMENT}(/node_modules/{_NPM_SEGMENT})*")
#: The platform the images run on, as npm's package os, cpu and libc fields name it: the agent cap
#: fetches and checks the lock's packages npm installs here.
NPM_PLATFORM = {"os": "linux", "cpu": "x64", "libc": "glibc"}
# Run by the pinned Node with npm-install-checks, the lock and the fetched paths: names each lock
# entry npm installs here that the build does not fetch, and each it fetches that npm does not
# install here, and fails if there is either.
_NPM_PLAN_JS = " ".join([
    "const { checkPlatform } = require(process.argv[1]);",
    "const packages = require(process.argv[2]).packages;",
    "const fetched = process.argv.slice(3);",
    'const here = Object.keys(packages).filter((key) => { if (key === "") return false;',
    "try { checkPlatform(packages[key]); return true; }",
    'catch (e) { if (e.code !== "EBADPLATFORM") throw e; return false; } });',
    "const unfetched = here.filter((key) => !fetched.includes(key));",
    "const unused = fetched.filter((key) => !here.includes(key));",
    'if (unfetched.length) console.error("npm installs here, but the build does not fetch: " +',
    'unfetched.join(" "));',
    'if (unused.length) console.error("the build fetches, but npm does not install here: " +',
    'unused.join(" "));',
    "process.exit(unfetched.length || unused.length ? 1 : 0);",
])
# Run by the pinned Node with ``npm ls --all --json --long``'s output, the prefix and the fetched
# packages as path=version: names each one npm did not install, and each it installed that the
# build did not check, and fails if there is either.
_NPM_INSTALLED_JS = " ".join([
    'const path = require("path");',
    "const tree = require(process.argv[1]);",
    "const prefix = process.argv[2];",
    "const want = process.argv.slice(3);",
    "const got = new Set();",
    "const walk = (deps) => { for (const node of Object.values(deps || {})) {",
    'if (node.path && node.version) got.add(path.relative(prefix, node.path) + "=" + node.version);',
    "walk(node.dependencies); } };",
    "walk(tree.dependencies);",
    'const shown = (entry) => entry.replace(/=([^=]*)$/, "@$1");',
    'const missing = want.filter((entry) => !got.has(entry)).map((entry) => "missing " + shown(entry));',
    "const unchecked = [...got].filter((entry) => !want.includes(entry)).sort()",
    '.map((entry) => "not checked " + shown(entry));',
    "const wrong = [...missing, ...unchecked];",
    'if (wrong.length) { console.error("npm\'s installed packages differ from the checked ones: " + '
    'wrong.join("; ")); process.exit(1); }',
])


def fail_closed(step: str) -> list[str]:
    """*step*, then a line that ends the script with the step's status if it failed. The step
    is emitted bare rather than wrapped, so a group or subshell cannot suspend errexit inside
    it and an ``export`` stays visible to later steps."""
    return [step.rstrip(), STEP_CHECK]


def uv_url(uv: UvPin) -> str:
    """Where *uv*'s release tarball is downloaded from."""
    return f"https://github.com/astral-sh/uv/releases/download/{uv.version}/{_UV_TARBALL}.tar.gz"


def _node_top(node: NodePin) -> str:
    """The directory *node*'s release tarball holds everything under."""
    return f"node-v{node.version}-linux-x64"


def node_url(node: NodePin) -> str:
    """Where *node*'s release tarball is downloaded from."""
    return f"https://nodejs.org/dist/v{node.version}/{_node_top(node)}.tar.gz"


def lock_dir(decl: ImageDecl, locks_dir: Path, tasks_dir: Path = TASKS_DIR) -> Path:
    """The directory *decl*'s lock files are in: ``locks/`` of its task's directory under
    *tasks_dir* for a task's own layer, *locks_dir*, the shared one, for every other image."""
    if isinstance(decl, LayerImage) and decl.owner is not None:
        return tasks_dir / decl.owner / "locks"
    return locks_dir


def files_for(decl: ImageDecl, locks_dir: Path, tasks_dir: Path = TASKS_DIR) -> dict[str, Path]:
    """The files *decl* copies in, sorted by container path: container path to each lock file,
    under ``lock_dir``, and to each file a task's layer copies from its directory under
    *tasks_dir*."""
    if isinstance(decl, BaseImage | LayerImage):
        names = [n for n in (decl.apt_lock, decl.pip_lock) if n is not None]
    elif isinstance(decl, CapImage):
        names = [decl.apt_lock]
    else:
        names = []
    locks = lock_dir(decl, locks_dir, tasks_dir)
    files = {f"{LOCK_ROOT}/{n}": locks / n for n in names}
    if isinstance(decl, CapImage):
        files[f"{NPM_PREFIX}/package-lock.json"] = locks / decl.npm_lock
    if isinstance(decl, LayerImage) and decl.fetched is not None:
        # Shared whichever task owns the layer: the files belong to a package's version.
        files[f"{LOCK_ROOT}/{decl.fetched.lock}"] = locks_dir / decl.fetched.lock
    if isinstance(decl, LayerImage) and decl.owner is not None:
        files.update({target: tasks_dir / decl.owner / source for target, source in decl.files.items()})
    return dict(sorted(files.items()))


def npm_manifest(decl: CapImage) -> str:
    """The ``package.json`` the agent cap installs its CLIs from: exactly ``decl.npm``. Its lock
    (``decl.npm_lock``) is generated from this text."""
    manifest = {"name": "hls-agent-cli", "private": True, "dependencies": dict(sorted(decl.npm.items()))}
    return json.dumps(manifest, sort_keys=True)


def _fits(adapter: TypeAdapter[str], text: str) -> bool:
    try:
        adapter.validate_python(text)
    except ValidationError:
        return False
    return True


def _lines(path: Path) -> list[str]:
    """The lines of *path* split at "\\n" alone, as sed and the shell read them."""
    return path.read_bytes().decode().split("\n")


def _check_apt_lock(path: Path) -> None:
    """Raise ValueError unless every line of apt lock *path* but a blank one or a comment is a
    ``name=version`` pin, and at least one is: those lines reach apt-get's arguments word-split."""
    pins = 0
    for number, line in enumerate(_lines(path), 1):
        if line == "" or line.startswith("#"):
            continue
        if not _fits(_APT_PIN, line):
            raise ValueError(f"line {number} of apt lock {path} is not an apt pin name=version: {line!r}")
        pins += 1
    if not pins:
        raise ValueError(f"apt lock {path} pins nothing")


def _pip_lock(path: Path) -> dict[str, str]:
    """Normalized name to version of every entry of hashed pip lock *path*. Raise ValueError
    unless each entry is ``name==version``, or ``name @ <URL>`` of a wheel whose file name
    carries that name and its version, followed only by its ``--hash=sha256:`` options, and no
    name is locked twice."""
    entries: list[str] = []
    current = ""
    for line in _lines(path):
        stripped = line.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        entries.append(current + stripped)
        current = ""
    if current:
        raise ValueError(f"pip lock {path} ends in a continued line")
    locked: dict[str, str] = {}
    for entry in entries:
        pin, *options = entry.split()
        if options[:1] == ["@"]:
            # A wheel by URL: its file name says which distribution and version it is.
            wheel = _WHEEL_URL.fullmatch(options[1]) if len(options) > 1 else None
            if wheel is None or not _fits(_PIP_PIN, f"{pin}==0") or pip_name(wheel["dist"]) != pip_name(pin):
                raise ValueError(f"pip lock {path}: {entry!r} is not name @ the https URL of a wheel of that name")
            pin, options = f"{pin}=={wheel['version']}", options[2:]
        if not _fits(_PIP_PIN, pin) or not options or not all(_HASH_OPTION.fullmatch(o) for o in options):
            raise ValueError(f"pip lock {path}: {entry!r} is not name==version with its sha256 hashes")
        if pip_name(pin) in locked:
            raise ValueError(f"pip lock {path} locks {pip_name(pin)} twice")
        locked[pip_name(pin)] = pin.split("==", 1)[1]
    return locked


def _check_fetch_lock(path: Path) -> None:
    """Raise ValueError unless every line of fetch lock *path* but a blank one or a comment is
    ``<sha256>  <path>``, the path relative and of plain segments, no path is named twice or is a
    directory of another, and at least one is named: ``sha256sum`` checks the fetched files
    against those lines, and the package's loader is called with each."""
    paths: set[str] = set()
    for number, line in enumerate(_lines(path), 1):
        if line == "" or line.startswith("#"):
            continue
        match = _FETCH_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"line {number} of fetch lock {path} is not a sha256, two spaces and a relative "
                             f"path of plain segments: {line!r}")
        if match["path"] in paths:
            raise ValueError(f"fetch lock {path} names {match['path']} twice")
        paths.add(match["path"])
    if not paths:
        raise ValueError(f"fetch lock {path} names no file")
    for name in paths:
        parts = name.split("/")
        inside = next(("/".join(parts[:n]) for n in range(1, len(parts)) if "/".join(parts[:n]) in paths), None)
        if inside is not None:
            raise ValueError(f"fetch lock {path} names {inside} as a file and as a directory of {name}")


def _installs_here(entry: dict) -> bool:
    """Whether npm installs lock entry *entry* on ``NPM_PLATFORM``, by its os, cpu and libc lists
    as npm-install-checks reads them: a list matches when it names the value, or holds only
    negations (``!value``) none of which is the value. A list that is absent matches."""
    for field, value in NPM_PLATFORM.items():
        wanted = entry.get(field)
        if wanted is None:
            continue
        wanted = [wanted] if isinstance(wanted, str) else list(wanted)
        if wanted == ["any"]:
            continue
        negated = [w[1:] for w in wanted if str(w).startswith("!")]
        if value in negated or (value not in wanted and len(negated) != len(wanted)):
            return False
    return True


@dataclass(frozen=True)
class _NpmPackage:
    """A package npm installs here, as the agent cap fetches and checks it."""

    #: Its path in the lock and under the prefix, ``node_modules/...``.
    path: str
    #: Its locked version.
    version: str
    #: Where its tarball is fetched from, an https URL.
    url: str
    #: The hex SHA-512 its tarball must have.
    sha512: str


@dataclass(frozen=True)
class _NpmLock:
    """What the agent cap installs from its npm lock."""

    #: Every package of the lock npm installs on ``NPM_PLATFORM``, in lock order.
    packages: list[_NpmPackage]
    #: Each CLI command and the version of the package it belongs to, in package order.
    commands: list[tuple[str, str]]


def _platform_packages(path: Path, packages: dict, name: str) -> None:
    """Raise ValueError unless every optional dependency of the CLI *name* in npm lock *path*
    (its *packages*) is locked at the one exact version it names, and one of them installs on
    ``NPM_PLATFORM``. The CLIs keep their binaries in such packages, and npm installs a CLI
    without them when they fail."""
    optional = packages[f"node_modules/{name}"].get("optionalDependencies") or {}
    here = False
    for dep, spec in sorted(optional.items()):
        if str(spec).startswith("npm:"):
            # An alias, npm:<package>@<version>, as codex names its platform builds.
            alias, _, version = str(spec).removeprefix("npm:").rpartition("@")
        else:
            alias, version = dep, str(spec)
        if not _NPM_EXACT.fullmatch(version):
            raise ValueError(f"npm lock {path}: {name} names {dep} at {spec}, which is not one exact version")
        entry = packages.get(f"node_modules/{dep}")
        if entry is None:
            raise ValueError(f"npm lock {path} does not lock {dep}, an optional dependency of {name}")
        locked = f"{entry.get('name', dep)}@{entry.get('version')}"
        if (entry.get("name", dep), entry.get("version")) != (alias, version):
            raise ValueError(f"npm lock {path} locks {dep} as {locked}, not {spec}")
        here = here or _installs_here(entry)
    if optional and not here:
        raise ValueError(f"npm lock {path}: {name} has no package for linux/x64 among its optional "
                         "dependencies")


def _npm_lock(path: Path, npm: dict[str, str]) -> _NpmLock:
    """What npm lock *path* installs on ``NPM_PLATFORM``. Raise ValueError unless the lock's root
    depends on exactly *npm*, locks each CLI at its version with every platform package it names,
    and fetches every package of the tree from a plain https URL, checked by a sha512 integrity
    hash."""
    try:
        lock = json.loads(path.read_bytes())
    except json.JSONDecodeError as e:
        raise ValueError(f"npm lock {path} is not JSON: {e}") from None
    packages = lock.get("packages") if isinstance(lock, dict) else None
    if not isinstance(packages, dict) or lock.get("lockfileVersion") not in (2, 3):
        raise ValueError(f"npm lock {path} is not a package-lock.json of lockfile version 2 or 3")
    root = packages.get("", {})
    if root.get("dependencies") != npm or set(root) - {"name", "dependencies"}:
        raise ValueError(f"npm lock {path} does not depend on exactly {npm}")
    here: list[_NpmPackage] = []
    for key, entry in packages.items():
        if key == "":
            continue
        if not key.startswith("node_modules/") or not isinstance(entry, dict):
            raise ValueError(f"npm lock {path}: {key!r} is not a package under node_modules")
        if entry.get("link"):
            raise ValueError(f"npm lock {path}: {key} is a link to a local directory")
        integrity = str(entry.get("integrity", ""))
        if not _INTEGRITY.fullmatch(integrity):
            raise ValueError(f"npm lock {path}: {key} has no sha512 integrity hash")
        if not _NPM_URL.fullmatch(str(entry.get("resolved", ""))):
            raise ValueError(f"npm lock {path}: {key} is not fetched from a plain https URL")
        if not _NPM_PATH.fullmatch(key) or not _NPM_EXACT.fullmatch(str(entry.get("version", ""))):
            raise ValueError(f"npm lock {path}: {key} is not a plain package path at one exact version")
        if _installs_here(entry):
            sha512 = base64.b64decode(integrity.removeprefix("sha512-")).hex()
            here.append(_NpmPackage(path=key, version=entry["version"], url=entry["resolved"], sha512=sha512))
    commands: list[tuple[str, str]] = []
    for name, version in sorted(npm.items()):
        entry = packages.get(f"node_modules/{name}", {})
        if entry.get("version") != version:
            raise ValueError(f"npm lock {path} does not lock {name} at {version}")
        _platform_packages(path, packages, name)
        bins = entry.get("bin")
        if not isinstance(bins, dict) or not bins:
            raise ValueError(f"npm package {name} has no command")
        for command in sorted(bins):
            if not _COMMAND.fullmatch(command):
                raise ValueError(f"npm package {name} names {command!r}, which is not a command name")
            if command in [c for c, _ in commands]:
                raise ValueError(f"npm lock {path} names command {command} twice")
            commands.append((command, version))
    return _NpmLock(packages=here, commands=commands)


def _checked_files(decl: ImageDecl, locks_dir: Path,
                   tasks_dir: Path = TASKS_DIR) -> tuple[dict[str, Path], _NpmLock | None]:
    """``files_for(decl, locks_dir, tasks_dir)`` once every file exists and every lock holds only
    what it may, and what the agent cap installs from its npm lock (None for another kind). Raise
    FileNotFoundError for a missing file and ValueError for a malformed lock."""
    files = files_for(decl, locks_dir, tasks_dir)
    for host in files.values():
        if not host.is_file():
            raise FileNotFoundError(f"file {host} of image {decl.name} does not exist")
    locks = lock_dir(decl, locks_dir, tasks_dir)
    if isinstance(decl, BaseImage | LayerImage | CapImage) and decl.apt_lock is not None:
        _check_apt_lock(locks / decl.apt_lock)
    if isinstance(decl, BaseImage | LayerImage) and decl.pip_lock is not None:
        locked = _pip_lock(locks / decl.pip_lock)
        for pin in decl.source_builds if isinstance(decl, LayerImage) else []:
            if locked.get(pip_name(pin)) != pin.split("==", 1)[1]:
                raise ValueError(f"source build {pin} of image {decl.name} is not in its pip lock "
                                 f"{decl.pip_lock}")
    if isinstance(decl, LayerImage) and decl.fetched is not None:
        _check_fetch_lock(locks_dir / decl.fetched.lock)
    if isinstance(decl, CapImage):
        return files, _npm_lock(locks / decl.npm_lock, decl.npm)
    return files, None


def _nvidia_debs(debs: dict[str, str]) -> list[str]:
    """Download the .deb of each pin of *debs* (pin to sha256) into a fresh directory, and check
    each against its sha256; ``$hls_nvidia_debs`` then names the checked files. NVIDIA's
    repository has no snapshot, so a file republished under the same version must not install."""
    steps = ['hls_debs="$(mktemp -d /opt/hls/nvidia-debs.XXXXXX)"', 'cd "$hls_debs"',
             "apt-get download " + " ".join(shlex.quote(pin) for pin in debs), "cd /", 'hls_nvidia_debs=""']
    for pin, sha256 in debs.items():
        name = pin.split("=", 1)[0]
        # apt-get download names the file <name>_<version, quoted>_<arch>.deb. The digest check
        # ends its step, so the step's own status check sees it fail.
        steps.append("\n".join([
            f'set -- "$hls_debs"/{name}_*.deb',
            '[ "$#" -eq 1 ] && [ -f "$1" ] || '
            f'{{ echo "apt-get download left no single .deb of {name}" >&2; exit 1; }}',
            f'echo "{sha256}  $1" | sha256sum -c -',
        ]))
        steps.append('hls_nvidia_debs="$hls_nvidia_debs $1"')
    return steps


def _fetched_debs(debs: dict[str, str]) -> list[str]:
    """Download each .deb of *debs* (URL to sha256) into a fresh directory of the image's tree and
    check it against its sha256; ``$hls_fetched_debs`` then names the checked files."""
    steps = ['hls_fetched="$(mktemp -d /opt/hls/debs.XXXXXX)"', 'hls_fetched_debs=""']
    for url, sha256 in sorted(debs.items()):
        deb = f'$hls_fetched/{url.rsplit("/", 1)[1]}'
        steps += [f'curl -fsSL --retry 5 -o "{deb}" {shlex.quote(url)}',
                  f'echo "{sha256}  {deb}" | sha256sum -c -',
                  f'hls_fetched_debs="$hls_fetched_debs {deb}"']
    return steps


# Ends the build unless each fetched .deb is installed, at the version the file itself names.
_FETCHED_CHECK = "\n".join([
    "for hls_deb in $hls_fetched_debs; do",
    '  hls_name="$(dpkg-deb -f "$hls_deb" Package)"; hls_want="$(dpkg-deb -f "$hls_deb" Version)"',
    "  hls_got=\"$(dpkg-query -W -f='${Status} ${Version}' \"$hls_name\" 2>/dev/null)\" || hls_got=absent",
    '  [ "$hls_got" = "install ok installed $hls_want" ] || '
    '{ echo "$hls_deb is not installed: $hls_name is $hls_got, not $hls_want" >&2; exit 1; }',
    "done",
])


def _apt_steps(lock: str, nvidia: dict[str, str] | None = None, nvidia_repo: str | None = None,
               debs: dict[str, str] | None = None) -> list[str]:
    """Install every package of *lock* at its pinned version, every pin of *nvidia* (pin to
    the sha256 of its .deb) from its checked .deb, and every file of *debs* (URL to sha256) once
    it is checked; check each version, and drop the package lists.
    With *nvidia_repo*, apt preferences then keep every version that repository carries from
    being installed, so it cannot stand in for a package or dependency the snapshot supplies."""
    nvidia = nvidia or {}
    debs = debs or {}
    listed = f"$(sed -e '/^#/d' -e '/^$/d' {LOCK_ROOT}/{lock})"
    pins = " ".join([listed, *(shlex.quote(p) for p in nvidia)])
    steps = ["apt-get update"]
    if debs:
        steps += _fetched_debs(debs)
    if nvidia:
        steps += _nvidia_debs(nvidia)
    if nvidia_repo is not None:
        # The pinned packages install from their checked files, never from the repository.
        origin = shlex.quote(f'Pin: origin "{urlsplit(nvidia_repo).hostname}"')
        steps.append(f"printf '%s\\n' 'Package: *' {origin} 'Pin-Priority: -1' > {NVIDIA_PREFERENCES}")
    files = (" $hls_nvidia_debs" if nvidia else "") + (" $hls_fetched_debs" if debs else "")
    check = "\n".join([
        'hls_missing=""',
        f"for hls_pin in {pins}; do",
        '  hls_name="${hls_pin%%=*}"; hls_want="${hls_pin#*=}"',
        "  hls_got=\"$(dpkg-query -W -f='${Status} ${Version}' \"$hls_name\" 2>/dev/null)\""
        " || hls_got=absent",
        '  [ "$hls_got" = "install ok installed $hls_want" ] || hls_missing="$hls_missing $hls_pin"',
        "done",
        '[ -z "$hls_missing" ] || '
        '{ echo "apt packages missing or at another version:$hls_missing" >&2; exit 1; }',
    ])
    steps += [f"apt-get install -y --no-install-recommends {listed}{files}", check]
    if debs:
        steps += [_FETCHED_CHECK, 'rm -rf "$hls_fetched"']
    if nvidia:
        steps.append('rm -rf "$hls_debs"')
    return [*steps, "rm -rf /var/lib/apt/lists/*"]


def _snapshot_sources(snapshot: str) -> list[str]:
    """Point the apt of a layer on a registry image at *snapshot* alone: end the build unless the
    image is the Ubuntu release the apt locks are pinned for, write the snapshot's pockets into
    ``SNAPSHOT_SOURCES``, and configure apt to read that file and no other source: the sources
    the image came with stay where they are, unread."""
    release = UBUNTU_DISTS[0]
    url = f"https://snapshot.ubuntu.com/ubuntu/{snapshot}/"
    lines = [f"deb [signed-by={UBUNTU_KEYRING}] {url} {dist} {' '.join(UBUNTU_COMPONENTS)}" for dist in UBUNTU_DISTS]
    conf = ['APT::Sandbox::User "root";', 'Acquire::Retries "5";', 'Acquire::Check-Valid-Until "false";',
            f'Dir::Etc::sourcelist "{SNAPSHOT_SOURCES}";', f'Dir::Etc::sourceparts "{SNAPSHOT_SOURCE_PARTS}";']
    return [
        "\n".join([
            'hls_release="$(. /etc/os-release && echo "$VERSION_CODENAME")"',
            f'[ "$hls_release" = {release} ] || '
            f'{{ echo "the image is Ubuntu $hls_release, and the apt locks are pinned for {release}" >&2; exit 1; }}',
        ]),
        f"mkdir -p /etc/apt/apt.conf.d {SNAPSHOT_SOURCE_PARTS}",
        # A file an earlier layer of the chain left there would be a source of its own.
        f'[ -z "$(ls -A {SNAPSHOT_SOURCE_PARTS})" ]',
        f"printf '%s\\n' {' '.join(shlex.quote(line) for line in lines)} > {SNAPSHOT_SOURCES}",
        f"printf '%s\\n' {' '.join(shlex.quote(line) for line in conf)} > /etc/apt/apt.conf.d/99-hls",
    ]


def _held_check() -> str:
    """A step that ends the build when an installed package differs from the base's
    constraints file."""
    # Double quotes only, so the shell quoting stays one plain pair of single quotes.
    code = "; ".join([
        "import sys",
        "from importlib.metadata import version",
        f'pins = [x.strip() for x in open("{CONSTRAINTS}") if x.strip()]',
        'bad = [p for p in pins if version(p.split("==")[0]) != p.split("==")[1]]',
        'sys.exit("held packages at another version: %s" % bad if bad else 0)',
    ])
    return f"{VENV}/bin/python -c '{code}'"


def _pip_steps(lock: str, source_builds: list[str]) -> list[str]:
    """Install hashed pip lock *lock* from wheels alone, but for *source_builds*, which build in
    the venv itself; check the held packages and drop uv's cache."""
    # An isolated build would resolve its build requirements from the live index, unhashed.
    binary = "--only-binary :all:" + "".join(f" --no-binary {name} --no-build-isolation-package {name}"
                                             for name in sorted(map(pip_name, source_builds)))
    install = (f"uv pip install --python {VENV}/bin/python --require-hashes --no-deps {binary} "
               f"--index-strategy unsafe-best-match --index-url {TORCH_INDEX} --extra-index-url {PYPI_INDEX} "
               f"-r {LOCK_ROOT}/{lock}")
    # The cache lives in the image (HOME=/root); --no-cache would stage the wheels in the host's /tmp.
    return [install, _held_check(), "uv cache clean"]


def _registry_pip_steps(lock: str, python: str, uv: UvPin) -> list[str]:
    """Install hashed pip lock *lock* from wheels alone into *python*, a registry image's own
    interpreter, with *uv* fetched into a fresh directory of the image's tree, checked against its
    digest and removed with its cache after the install. The image's own uv and pip settings do
    not reach the install: its ``UV_*`` and ``PIP_*`` variables are unset first."""
    tarball = '"$hls_dl/uv.tar.gz"'
    install = (f'"$hls_dl/{_UV_TARBALL}/uv" pip install --no-config --cache-dir "$hls_dl/cache" '
               f"--python {python} --break-system-packages --require-hashes --no-deps --only-binary :all: "
               f"--index-strategy unsafe-best-match --index-url {TORCH_INDEX} --extra-index-url {PYPI_INDEX} "
               f"-r {LOCK_ROOT}/{lock}")
    return [
        "mkdir -p /opt/hls",
        'hls_dl="$(mktemp -d /opt/hls/uv-download.XXXXXX)"',
        f"curl -fsSL --retry 5 -o {tarball} {uv_url(uv)}",
        f'echo "{uv.sha256}  $hls_dl/uv.tar.gz" | sha256sum -c -',
        f'tar -xzf {tarball} -C "$hls_dl"',
        "unset $(env | grep -oE '^(UV|PIP)_[A-Za-z0-9_]*')",
        install,
        'rm -rf "$hls_dl"',
    ]


# Prints every source URI of one apt sources file, deb822 or one-line, that is neither *snap* nor
# *repo*, as " <file>:<uri>". A deb822 field continues on the lines that start with whitespace.
# apt reads a URI with or without its one trailing slash as the same repository (the CUDA image's
# own cuda.list has none), so one is dropped from each side before they are compared.
_FOREIGN_SOURCES = (
    r'function foreign(u) { sub(/\/$/, "", u); return u != snap && u != repo } '
    r'BEGIN { sub(/\/$/, "", snap); sub(/\/$/, "", repo) } '
    r'/^[[:space:]]*#/ { next } '
    r'FILENAME ~ /[.]sources$/ { if (tolower($0) ~ /^uris:/) { uris = 1; sub(/^[^:]*:/, "") } '
    r'else if ($0 !~ /^[[:space:]]/) uris = 0; '
    r'if (uris) for (i = 1; i <= NF; i++) if (foreign($i)) '
    r'printf " %s:%s", FILENAME, $i; next } '
    r'$1 == "deb" || $1 == "deb-src" { i = 2; if ($i ~ /^\[/) { while (i < NF && $i !~ /\]$/) i++; i++ } '
    r'if (foreign($i)) printf " %s:%s", FILENAME, $i }'
)


def _sources_check(snapshot: str, repo: str | None) -> str:
    """A step that ends the build if any apt source is neither *snapshot* nor *repo*, so a source
    the rewrite missed, or another mirror, cannot resolve live."""
    awk = (f"awk -v snap={shlex.quote(snapshot)} -v repo={shlex.quote(repo or '')} "
           f"{shlex.quote(_FOREIGN_SOURCES)}")
    return "\n".join([
        'hls_foreign=""',
        "for hls_list in /etc/apt/sources.list /etc/apt/sources.list.d/*.list "
        "/etc/apt/sources.list.d/*.sources; do",
        f'  if [ -f "$hls_list" ]; then hls_foreign="$hls_foreign$({awk} "$hls_list")"; fi',
        "done",
        '[ -z "$hls_foreign" ] || { echo "apt sources other than the snapshot and the declared repository:'
        '$hls_foreign" >&2; exit 1; }',
    ])


def _base_post(decl: BaseImage, uv: UvPin) -> list[str]:
    snapshot = f"https://snapshot.ubuntu.com/ubuntu/{decl.snapshot}/"
    apt_conf = ("printf '%s\\n' 'APT::Sandbox::User \"root\";' 'Acquire::Retries \"5\";' "
                "'Acquire::Check-Valid-Until \"false\";' > /etc/apt/apt.conf.d/99-hls")
    # Downloaded into a fresh directory of the image's own tree: a build's /tmp is the host's, where
    # another build or user could replace the file between its check and its use.
    tarball = '"$hls_dl/uv.tar.gz"'
    return [
        "export DEBIAN_FRONTEND=noninteractive",
        "mkdir -p /etc/apt/apt.conf.d /etc/apt/preferences.d /opt/hls /usr/local/share/fonts",
        apt_conf,
        (f"sed -i -e 's|http://archive.ubuntu.com/ubuntu/*|{snapshot}|g' "
         f"-e 's|http://security.ubuntu.com/ubuntu/*|{snapshot}|g' {UBUNTU_SOURCES}"),
        f"grep -q '{snapshot}' {UBUNTU_SOURCES}",
        _sources_check(snapshot, decl.nvidia_repo),
        *_apt_steps(decl.apt_lock, decl.nvidia_apt, decl.nvidia_repo),
        'hls_dl="$(mktemp -d /opt/hls/uv-download.XXXXXX)"',
        f"curl -fsSL --retry 5 -o {tarball} {uv_url(uv)}",
        f'echo "{uv.sha256}  $hls_dl/uv.tar.gz" | sha256sum -c -',
        f'tar -xzf {tarball} -C "$hls_dl"',
        f'install -m 0755 "$hls_dl/{_UV_TARBALL}/uv" /usr/local/bin/uv',
        'rm -rf "$hls_dl"',
        f"uv venv --python {model.SYSTEM_PYTHON} {VENV}",
        *_constraints_steps(decl),
    ]


def _constraints_steps(decl: BaseImage) -> list[str]:
    """Write the base's constraints file and install its pip lock; a base with no pip lock holds
    nothing, so its file is empty."""
    if decl.pip_lock is None:
        return [f": > {CONSTRAINTS}"]
    return [f"printf '%s\\n' {' '.join(shlex.quote(c) for c in decl.constraints)} > {CONSTRAINTS}",
            *_pip_steps(decl.pip_lock, [])]


def _node_steps(node: NodePin) -> list[str]:
    """Install *node* into ``NODE_PREFIX`` from its release tarball once the tarball matches its
    digest, then check that the prefix's node is that release."""
    tarball = '"$hls_dl/node.tar.gz"'
    top = _node_top(node)
    # Only its tree, not the top-level README, LICENSE and CHANGELOG; tar fails on a missing member.
    members = " ".join(f"{top}/{part}" for part in ("bin", "include", "lib", "share"))
    return [
        'hls_dl="$(mktemp -d /opt/hls/node-download.XXXXXX)"',
        f"curl -fsSL --retry 5 -o {tarball} {node_url(node)}",
        f'echo "{node.sha256}  $hls_dl/node.tar.gz" | sha256sum -c -',
        f"mkdir -p {NODE_PREFIX}",
        f"tar -xzf {tarball} -C {NODE_PREFIX} --strip-components=1 --no-same-owner {members}",
        'rm -rf "$hls_dl"',
        "\n".join([
            f'hls_node="$({NODE_PREFIX}/bin/node --version)" || hls_node=absent',
            f'[ "$hls_node" = v{node.version} ] || '
            f'{{ echo "node is $hls_node, not v{node.version}" >&2; exit 1; }}',
        ]),
    ]


def _npm_steps(decl: CapImage, lock: _NpmLock) -> list[str]:
    """Install the agent cap's CLIs with the pinned Node's npm, offline, from tarballs the build
    fetched itself and checked against the lock's sha512s: npm treats an optional package whose
    download or integrity check fails as absent, and the CLIs' binaries live in optional platform
    packages. Then check npm installed exactly those packages, put each command on PATH, and
    check each CLI prints its pinned version.

    Which packages npm installs here is npm's own decision, so the build asks npm, not the
    renderer's filter: before fetching, npm's install checks must name exactly the packages it
    fetches (offline, npm would skip one the filter dropped as a failed optional package, and
    ``npm ls`` would not tell it from another platform's); after installing, ``npm ls`` must list
    exactly those packages at their versions."""
    # npm runs on the pinned Node: its own shebang and the install scripts it runs find node on PATH.
    npm = f'PATH="{NODE_PREFIX}/bin:$PATH" {NODE_PREFIX}/bin/npm'
    # A cache of the checked tarballs alone, removed with them; its logs go with it.
    cache = '--cache "$hls_npm/cache"'
    fetched = " ".join(shlex.quote(p.path) for p in lock.packages)
    steps = [
        f"{NODE_PREFIX}/bin/node -e {shlex.quote(_NPM_PLAN_JS)} {NPM_INSTALL_CHECKS} "
        f"{NPM_PREFIX}/package-lock.json {fetched}",
        'hls_npm="$(mktemp -d /opt/hls/npm-download.XXXXXX)"',
    ]
    # Every package npm installs here, claude-code's musl build among them: its lock entry names no
    # libc, so npm installs it beside the glibc one, though only the glibc build runs.
    for n, package in enumerate(lock.packages, 1):
        steps.append(f'curl -fsSL --retry 5 -o "$hls_npm/{n}.tgz" {shlex.quote(package.url)}')
        steps.append(f'echo "{package.sha512}  $hls_npm/{n}.tgz" | sha512sum -c -')
    files = " ".join(f'"$hls_npm/{n}.tgz"' for n in range(1, len(lock.packages) + 1))
    steps += [
        f"printf '%s\\n' {shlex.quote(npm_manifest(decl))} > {NPM_PREFIX}/package.json",
        f"{npm} cache add --offline --no-update-notifier {cache} {files}",
        # --engine-strict refuses a package whose engines exclude this Node rather than warning.
        f"{npm} ci --prefix {NPM_PREFIX} --offline {cache} --engine-strict --no-fund --no-audit "
        "--no-update-notifier",
        f'{npm} ls --prefix {NPM_PREFIX} --all --json --long --offline {cache} > "$hls_npm/ls.json"',
        f'{NODE_PREFIX}/bin/node -e {shlex.quote(_NPM_INSTALLED_JS)} "$hls_npm/ls.json" {NPM_PREFIX} '
        + " ".join(shlex.quote(f"{p.path}={p.version}") for p in lock.packages),
        'rm -rf "$hls_npm"',
    ]
    # A node script runs on the pinned Node, so the agent's own node on PATH stays Ubuntu's.
    wrapper = shlex.quote(f'#!/bin/sh\\nexec {NODE_PREFIX}/bin/node "%s" "$@"\\n')
    for command, _ in lock.commands:
        steps.append("\n".join([
            f'hls_target="$(readlink -f {NPM_PREFIX}/node_modules/.bin/{command})"',
            f'[ -f "$hls_target" ] || {{ echo "npm installed no {command}" >&2; exit 1; }}',
            "if head -n 1 \"$hls_target\" | grep -Eq '^#!.*[/ ]node([[:space:]]|$)'; then",
            f'  printf {wrapper} "$hls_target" > /usr/local/bin/{command}',
            f"  chmod 0755 /usr/local/bin/{command}",
            "else",
            f'  ln -s "$hls_target" /usr/local/bin/{command}',
            "fi",
        ]))
    clis = " ".join(shlex.quote(f"{command}={version}") for command, version in lock.commands)
    # codex writes into $HOME, so the check runs with one of its own, removed after.
    steps += [
        'hls_home="$(mktemp -d /opt/hls/cli-home.XXXXXX)"',
        "\n".join([
            f"for hls_cli in {clis}; do",
            '  hls_cmd="${hls_cli%%=*}"; hls_want="${hls_cli#*=}"',
            '  hls_out="$(HOME="$hls_home" "/usr/local/bin/$hls_cmd" --version 2>&1)" || '
            '{ echo "$hls_cmd --version failed: $hls_out" >&2; exit 1; }',
            '  case " $hls_out " in *" $hls_want "*) ;; '
            '*) echo "$hls_cmd --version printed $hls_out, not $hls_want" >&2; exit 1 ;; esac',
            "done",
        ]),
        'rm -rf "$hls_home"',
    ]
    return steps


def _loader_check(fetched: Fetched) -> str:
    """A step that calls the package's own loader on every file of ``$hls_fetch/sums``, with the
    variable that names the files' directory, as the layer's environment bakes it (``%post`` does
    not source ``%environment``), and the one under which the loader raises rather than download;
    it ends the build when the loader raises or returns nothing for a file."""
    module, function = fetched.loader.split(":")
    # Double quotes only, so the shell quoting stays one plain pair of single quotes.
    code = "; ".join([
        "import sys",
        "from importlib import import_module",
        f'load = getattr(import_module("{module}"), "{function}")',
        "pins = [line.split() for line in open(sys.argv[1])]",
        "bad = [path for sha256, path in pins if not load(path, sha256)]",
        'sys.exit("the package would still download %d files, the first: %s" % (len(bad), bad[:5]) if bad else 0)',
    ])
    # A home of its own, removed with the download directory, and no bytecode: the check leaves
    # nothing in the image.
    variables = ['HOME="$hls_fetch"', "PYTHONDONTWRITEBYTECODE=1", f"{fetched.variable}={fetched.into}",
                 f"{fetched.refuse_env}=1"]
    return f"env {' '.join(variables)} {VENV}/bin/python -c '{code}' \"$hls_fetch/sums\""


def _fetch_steps(fetched: Fetched) -> list[str]:
    """Download *fetched*'s archive into a fresh directory of the image's own tree, unpack it once
    it matches its digest, and move its directory of files to their directory of the image; check
    each file's sha256 against the lock and that the directory holds nothing else, and check that
    the package loads every file without downloading (``_loader_check``)."""
    archive = '"$hls_fetch/archive.zip"'
    steps = [
        'hls_fetch="$(mktemp -d /opt/hls/fetch.XXXXXX)"',
        f"curl -fsSL --retry 5 -o {archive} {fetched.url}",
        f'echo "{fetched.sha256}  $hls_fetch/archive.zip" | sha256sum -c -',
        # Unpacked by the system interpreter's zipfile, which every base has.
        f'{model.SYSTEM_PYTHON} -m zipfile -e {archive} "$hls_fetch/unpacked"',
        # mv would put the files inside a directory that is already there.
        f'[ ! -e {fetched.into} ] || '
        f'{{ echo "{fetched.into} is there before its files are fetched" >&2; exit 1; }}',
        f"mkdir -p {PurePosixPath(fetched.into).parent}",
        f'mv "$hls_fetch/unpacked/{fetched.member}" {fetched.into}',
        *([f"find {fetched.into} -type f -name '*{fetched.dropped}' -delete"] if fetched.dropped is not None else []),
        f"sed -e '/^#/d' -e '/^$/d' {LOCK_ROOT}/{fetched.lock} > \"$hls_fetch/sums\"",
        f"cd {fetched.into}",
        # --strict: a line sha256sum cannot read fails the check rather than being skipped.
        'sha256sum -c --strict --quiet "$hls_fetch/sums"',
        "cd /",
        "\n".join([
            f'hls_found="$(find {fetched.into} ! -type d | wc -l)"; hls_locked="$(wc -l < "$hls_fetch/sums")"',
            f'[ "$hls_found" -eq "$hls_locked" ] || {{ echo "{fetched.into} holds $hls_found files, and the lock '
            'names $hls_locked" >&2; exit 1; }',
        ]),
    ]
    return [*steps, _loader_check(fetched), 'rm -rf "$hls_fetch"']


def _child_post(decl: LayerImage | CapImage, npm: _NpmLock | None, registry: RegistryImage | None,
                uv: UvPin) -> list[str]:
    steps: list[str] = []
    if decl.apt_lock is not None:
        steps.append("export DEBIAN_FRONTEND=noninteractive")
        if registry is not None:
            steps += _snapshot_sources(registry.snapshot)
        steps += _apt_steps(decl.apt_lock, debs=decl.debs if isinstance(decl, LayerImage) else None)
    if isinstance(decl, LayerImage) and decl.pip_lock is not None:
        if registry is not None:
            steps += _registry_pip_steps(decl.pip_lock, registry.python, uv)
        else:
            steps += _pip_steps(decl.pip_lock, list(decl.source_builds))
    if isinstance(decl, LayerImage) and decl.fetched is not None:
        # After the pip install: the check calls the package the lock installs.
        steps += _fetch_steps(decl.fetched)
    if isinstance(decl, CapImage):
        assert npm is not None
        steps += [*_node_steps(decl.node), *_npm_steps(decl, npm), "mkdir -p /workspace /logs/agent"]
    return steps


def _script(decl: ImageDecl, npm: _NpmLock | None, uv: UvPin, registry: RegistryImage | None = None) -> str:
    """The build script of *decl* once its locks are checked; *npm* is what a cap installs, and
    *registry* the registry image its chain starts at, if it starts at one."""
    if isinstance(decl, RegistryImage):
        raise ValueError(f"registry image {decl.name} is the vendor's image as pulled: no script builds it")
    if registry is not None and isinstance(decl, LayerImage | CapImage):
        _on_registry(decl, registry)
    lines = ["set -eu", "set +x"]
    for step in _base_post(decl, uv) if isinstance(decl, BaseImage) else _child_post(decl, npm, registry, uv):
        lines += fail_closed(step)
    return "\n".join(lines)


def post_script(decl: ImageDecl, *, locks: Path, uv: UvPin, tasks_dir: Path = TASKS_DIR,
                registry: RegistryImage | None = None) -> str:
    """The shell script that builds *decl* on its parent, or a base on its CUDA image, from the
    files ``files_for(decl, locks, tasks_dir)`` copies in: every wrapper runs this text as it is.
    *registry* is the registry image *decl*'s chain starts at (``model.registry_of``), None for a
    chain that starts at a base. Raise ValueError for a registry image, which no script builds, an
    image that may not build on a registry image, or a lock holding what it may not, and
    FileNotFoundError for a missing lock file."""
    _, npm = _checked_files(decl, locks, tasks_dir)
    return _script(decl, npm, uv, registry)


class Variable(NamedTuple):
    """A variable of an image's environment."""

    #: Its name.
    name: str
    #: Its value: a literal one, or, when not *literal*, a shell word in which ``$PATH`` is the
    #: PATH the image had before, which both runtimes expand.
    value: str
    #: Whether *value* is taken as it is.
    literal: bool


def environment(decl: ImageDecl) -> list[Variable]:
    """The variables *decl*'s image sets for everything that runs in it and in its children, in
    order: ``BASE_ENVIRONMENT``, then the declared ones by name. Only a base sets any."""
    if not isinstance(decl, BaseImage):
        return []
    return [*(Variable(name, value, False) for name, value in BASE_ENVIRONMENT.items()),
            *(Variable(name, decl.env[name], True) for name in sorted(decl.env))]


def _indent(lines: list[str]) -> list[str]:
    # Apptainer reads a .def by "\n" alone; splitlines() would also break at characters it does not.
    return ["    " + line if line else "" for line in "\n".join(lines).split("\n")]


def _on_registry(decl: LayerImage | CapImage, registry: RegistryImage) -> None:
    """Raise ValueError unless *decl* may build on *registry*, a vendor's image: a server layer
    with no source build."""
    if not isinstance(decl, LayerImage) or not decl.server:
        raise ValueError(f"{decl.kind} {decl.name} builds on registry image {registry.name}: only a server layer "
                         "builds on a vendor's image, which may hold what the deny list names")
    if decl.source_builds:
        raise ValueError(f"image {decl.name} builds from source on registry image {registry.name}, whose "
                         "packages no lock records")


def _render(decl: ImageDecl, *, parent: str | None, files: dict[str, str], npm: _NpmLock | None,
            uv: UvPin, registry: RegistryImage | None = None) -> str:
    if isinstance(decl, RegistryImage):
        if parent is not None:
            raise ValueError(f"registry image {decl.name} has no parent")
        return "\n".join(["Bootstrap: docker", f"From: {decl.ref}", ""])
    if isinstance(decl, BaseImage):
        if parent is not None:
            raise ValueError(f"base {decl.name} has no parent")
        # The tag stays readable in a label: the digest alone is what Apptainer pulls.
        head = ["Bootstrap: docker", f"From: {decl.pull_ref}", "", "%labels",
                *_indent([f"hls.cuda_ref {decl.cuda_ref}"])]
    else:
        if parent is None:
            raise ValueError(f"{decl.kind} {decl.name} needs its parent's image file")
        head = ["Bootstrap: localimage", f"From: {parent}"]
    out = [*head, ""]
    if files:
        out += ["%files", *_indent([f"{host} {container}" for container, host in files.items()]), ""]
    out += ["%post", *_indent([_script(decl, npm, uv, registry)]), ""]
    if isinstance(decl, BaseImage):
        env = [f"export {v.name}={shlex.quote(v.value) if v.literal else v.value}" for v in environment(decl)]
        out += ["%environment", *_indent(env), "", "%runscript", *_indent(['exec "$@"']), ""]
    elif isinstance(decl, LayerImage) and decl.env is not None:
        out += ["%environment", *_indent([f"export {name}={value}" for name, value in decl.env]), ""]
    text = "\n".join(out)
    if "{{" in text:
        raise ValueError(f"the .def of image {decl.name} holds '{{{{', which Apptainer reads as a build "
                         "argument")
    return text


def baked_environment(name: str, decls: Iterable[ImageDecl]) -> dict[str, str]:
    """The variables image *name*, one of *decls*, bakes, as its ``%environment`` exports them: its
    base's, then each layer's of its chain from the bottom up. A chain on a registry image bakes
    its layers' alone: the vendor image's own environment is its image's, which no
    ``%environment`` here writes. Raise ValueError for an image that builds on neither."""
    by_name = {d.name: d for d in decls}
    chain: list[ImageDecl] = []
    current: str | None = name
    while current is not None:
        chain.append(by_name[current])
        current = parent_of(chain[-1])
    base = chain[-1]
    if isinstance(base, RegistryImage):
        env: dict[str, str] = {}
    elif isinstance(base, BaseImage):
        env = {**BASE_ENVIRONMENT, **{k: shlex.quote(base.env[k]) for k in sorted(base.env)}}
    else:
        raise ValueError(f"image {name} builds on no base")
    for decl in reversed(chain[:-1]):
        if isinstance(decl, LayerImage) and decl.env is not None:
            env.update(decl.env)
    return env


def check_environments(decls: Iterable[ImageDecl]) -> None:
    """Raise ValueError for a layer whose environment sets a variable to the value the image below
    already gives it, as a venv prefix on PATH set again would repeat; for a layer that is no server
    layer and sets a variable an image below it sets, so an agent's or a submission's image keeps
    the base's variables, the torch hold every install keeps among them, and only adds; and for such
    a layer that sets ``TORCH_CUDA_ARCH_LIST``, which the roles that compile in it set for their
    GPU. A server layer may set any other. Every image a layer builds on must be among *decls*."""
    listed = list(decls)
    for decl in listed:
        if not isinstance(decl, LayerImage) or decl.env is None:
            continue
        env = dict(decl.env)
        below = baked_environment(decl.parent, listed)
        same = sorted(name for name, value in env.items() if below.get(name) == value)
        if same:
            raise ValueError(f"layer {decl.name} sets {same} to what {decl.parent} gives already")
        if decl.server:
            continue
        if "TORCH_CUDA_ARCH_LIST" in env:
            raise ValueError(f"layer {decl.name} sets TORCH_CUDA_ARCH_LIST: the roles that compile in it set "
                             "their GPU's")
        set_below = sorted(set(env) & set(below))
        if set_below:
            raise ValueError(f"layer {decl.name} sets {set_below}, which {decl.parent} sets: a layer an agent or "
                             "a submission works in only adds to its environment")


def check_source_builds(decls: Iterable[ImageDecl], locks_dir: Path, tasks_dir: Path = TASKS_DIR) -> None:
    """Raise ValueError for a source build whose build backends no pip lock below its layer
    installs, naming the missing ones: it builds without isolation, against the venv as the
    images below left it, so a backend missing there would end the build. A layer's own lock
    installs with the build, not before it.
    Every image a layer builds on must be among *decls* (``order``'s refusals)."""
    by_name = {d.name: d for d in order(decls)}
    for decl in by_name.values():
        if not isinstance(decl, LayerImage) or not decl.source_builds:
            continue
        installed: set[str] = set()
        below = parent_of(decl)
        while below is not None:
            image = by_name[below]
            if isinstance(image, RegistryImage):
                raise ValueError(f"image {decl.name} builds from source on registry image {below}, whose "
                                 "packages no lock records")
            if isinstance(image, BaseImage | LayerImage) and image.pip_lock is not None:
                installed |= set(_pip_lock(lock_dir(image, locks_dir, tasks_dir) / image.pip_lock))
            below = parent_of(image)
        for pin, backends in sorted(decl.source_builds.items()):
            missing = [b for b in backends if pip_name(b) not in installed]
            if missing:
                raise ValueError(f"source build {pin} of image {decl.name} needs {', '.join(missing)}, which "
                                 "no pip lock below it installs")


def _checked_path(path: Path, what: str) -> str:
    text = str(path)
    if any(c.isspace() for c in text) or not text.isprintable():
        raise ValueError(f"{what} {text!r} holds whitespace or an unprintable character, which a .def "
                         "cannot carry")
    return text


def render_def(decl: ImageDecl, *, parent_sif: Path | None, locks: Path, uv: UvPin,
               tasks_dir: Path = TASKS_DIR, registry: RegistryImage | None = None) -> str:
    """The ``.def`` that builds *decl*: from the image file *parent_sif* for a layer or cap,
    copying its lock files from *locks*, or, for a task's own layer, its locks and files from its
    task's directory under *tasks_dir*; *registry* is the registry image its chain starts at
    (``model.registry_of``), None for a chain that starts at a base. Raise ValueError for a parent
    that does not fit the kind, an image that may not build on a registry image, a lock holding
    what it may not, or a path a ``.def`` cannot carry, and FileNotFoundError for a missing file."""
    files, npm = _checked_files(decl, locks, tasks_dir)
    shown = {container: _checked_path(host, "file") for container, host in files.items()}
    parent = None if parent_sif is None else _checked_path(parent_sif, "parent image")
    return _render(decl, parent=parent, files=shown, npm=npm, uv=uv, registry=registry)


def recipe_key(decl: ImageDecl, *, parent_key: str | None, locks_dir: Path, uv: UvPin,
               tasks_dir: Path = TASKS_DIR, registry: RegistryImage | None = None) -> str:
    """The cache key of *decl*: ``<name>-<16 hex>`` of its ``.def`` (with stand-ins for the
    paths, and *registry* as ``render_def`` takes it), the bytes of every file it copies in and
    *parent_key*. A registry image's key is its name's and its reference's. Raise ValueError when
    *parent_key* does not fit the kind or
    a lock holds what it may not, and FileNotFoundError for a missing lock file."""
    check_parent_key(decl, parent_key)
    parent = parent_of(decl)
    files, npm = _checked_files(decl, locks_dir, tasks_dir)
    stand_in = {container: str(_LOCKS_STAND_IN / host.name) for container, host in files.items()}
    text = _render(decl, parent=_PARENT_STAND_IN if parent is not None else None, files=stand_in,
                   npm=npm, uv=uv, registry=registry)
    return text_key(decl.name, text, files, parent_key)


def check_parent_key(decl: ImageDecl, parent_key: str | None) -> None:
    """Raise ValueError unless *parent_key* is given exactly when *decl* builds on a parent."""
    if (parent_of(decl) is None) != (parent_key is None):
        if parent_of(decl) is None:
            raise ValueError(f"{decl.kind} {decl.name} has no parent, so it takes no parent key")
        raise ValueError(f"{decl.kind} {decl.name} needs its parent's key")


def text_key(name: str, text: str, files: dict[str, Path], parent_key: str | None) -> str:
    """``<name>-<16 hex>`` of a SHA-256 over the rendered *text*, each file of *files* (path in
    the image to the file on the host) with its path, and *parent_key*."""
    h = hashlib.sha256(text.encode())
    for container, host in files.items():
        h.update(b"\0file\0" + container.encode() + b"\0")
        h.update(host.read_bytes())
    h.update(b"\0parent\0" + (parent_key or "").encode())
    return f"{name}-{h.hexdigest()[:16]}"
