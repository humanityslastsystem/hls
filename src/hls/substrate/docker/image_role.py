"""The harness roles that build, pin and read Docker images and build a submission image:
``python -m hls.substrate.docker.image_role build|pin|submit|read``. Their command lines come from
``hls.substrate.docker.images.DockerImages``; each runs in a harness role's container with the
node's Docker socket bound in, and reaches the daemon only through it.

An image archive is one tar file that both kinds of Docker image store load, as BuildKit's
``docker`` exporter writes it with OCI media types (``ARCHIVE_OUTPUT``): an OCI image layout
(``oci-layout``, an ``index.json`` that names exactly one image manifest, and the blobs that
manifest names), which containerd's image store reads, and beside it a ``manifest.json`` that
names the same config and layers, which the classic store reads (``archive_image``). Besides its
sha256, an archive is identified by that manifest's digest and its config's, both read from the
archive's own bytes, so both hold on every host. An archive is refused (``ArchiveRefused``) when a
member name appears twice, its index names anything but one manifest or names an image, its
``manifest.json`` names another image, or a manifest or config blob does not hash to its name:
one store could then load another image from it than the other, or the load would give the image
a name in the daemon that no pin owns.

``pin``: the role's shell has copied the archive into the role's own directory and checked the
copy's sha256 (``FETCH_EXEC``). The program reads the copy's identity, loads the copy, checks that
the daemon's image ID is the manifest's digest (containerd's image store) or the config's (the
classic store), gives the image the name it was handed and checks that the name reaches the same
ID. It then writes what it pinned (``Pinned``) and removes the copy. A refused archive, or a daemon
whose image is another, ends it with ``DIGEST_MISMATCH_EXIT`` and a ``refused:`` line on stderr.

``build``: for a layer or a cap, the shell has copied and checked the parent's archive the same
way, and the program unpacks it as an OCI layout, the parent build context
(``render_docker.PARENT_CONTEXT``). It creates a BuildKit builder of its own (buildx's
``docker-container`` driver) held to the role's own CPUs and memory, with no swap, runs
``docker buildx build`` with every directory under the contexts directory as a named build
context and the client writing the image archive, and removes the builder and its
state volume, whether the build succeeded or not. The builder's container and volume have names
fixed by the builder's name (``builder_objects``), so a builder that a killed role left is found
and removed by the next role of the same name before it creates its own. It writes the
archive's sha256 and its identity with the BuildKit version (``BuiltImage``).

buildx starts the ``docker-container`` driver's BuildKit container privileged and gives no way
to turn that off; ``DockerImages.build_role`` therefore refuses a build whose input holds a
contestant's files. The builder's memory is outside the role's container, so the role's grant
holds the client and the builder each up to the role's memory.

``submit``: the submission image, without a builder. The install of the submission's wheels runs
in a container of the pinned submission base made for it alone: as uid 0, which the install needs to
write the image's root-owned paths (a container of the host user cannot), with every capability
dropped but ``INSTALL_CAPABILITIES``, ``no-new-privileges``, the default seccomp profile, no network,
no device and no socket, the tree and the wheels bound read-only, and this role's own CPUs, memory
and tasks. It runs ``install_script`` and nothing a caller names: the ``.def``'s staging copy, its
``%post`` text and its runscript. The stopped container is committed with the base's configuration,
exported in the form both image stores load, and removed with the committed image; the job's label
lets the job's end remove what a killed role left.

``read``: one directory of an image, never its root, for a role that reads it, exported by the
daemon from a container of the image that never starts, and unpacked as the host user.

A registry login is the user name in ``REGISTRY_USER_VAR`` and the token in
``REGISTRY_TOKEN_GRANT``, which the role's env file holds. The program removes both from its
environment before it starts any command, passes the token to ``docker login --password-stdin``
on standard input, and logs out when the build ends; the credential lives only in the role's own
``DOCKER_CONFIG``. Docker's login names Docker Hub, the registry the catalog's bases come from."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import threading
import shlex
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hls.recipes.render_docker import PARENT_CONTEXT, parent_context
from hls.store.digest import DIGEST_MISMATCH_EXIT, sha256_file
from hls.substrate.apptainer.images import FETCH_SHELL, REGISTRY_TOKEN_GRANT, REGISTRY_USER_VAR
from hls.package.offline import STAGE, offline_install
from hls.recipes.render_docker import RUNSCRIPT as DOCKER_RUNSCRIPT
from hls.substrate.container import TREE_ABSENT_EXIT
from hls.substrate.docker.api import BindMount, ContainerConfig, DockerClient, DockerError, NotFound

#: The shell a role runs before its program when it reads an archive from the store: arguments
#: the archive, the path in the role's own directory it is copied to and its recorded sha256
#: (``FETCH_SHELL``), then the program's command line, which it becomes once the copy is checked.
FETCH_EXEC = FETCH_SHELL + 'hls_fetch "$1" "$2" "$3"; shift 3; exec "$@"'
#: Seconds one read or write of an archive's transfer to the daemon may wait.
LOAD_TIMEOUT_SEC = 600.0
#: The media types of an image manifest an index may name.
MANIFEST_TYPES = frozenset({"application/vnd.oci.image.manifest.v1+json",
                            "application/vnd.docker.distribution.manifest.v2+json"})
#: Index annotations that make ``docker load`` name the image it loads.
NAME_ANNOTATIONS = frozenset({"io.containerd.image.name", "org.opencontainers.image.ref.name"})
#: The largest index, manifest or config blob read into memory. A blob past it is no image's.
JSON_MAX_BYTES = 16 << 20
#: BuildKit's output of an image archive (``--output=<this>,dest=<file>``): its ``docker`` exporter
#: writes the OCI layout and a ``manifest.json`` in one tar; with OCI media types the layout's
#: blobs are the ones its ``oci`` exporter writes. A classic image store loads only an archive
#: that has ``manifest.json``.
ARCHIVE_OUTPUT = "type=docker,oci-mediatypes=true"
#: Exit status of a build or pin that failed for another reason than a refused file.
FAILED_EXIT = 1

_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")
_CONTEXT_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
_BUILDKIT_VERSION = re.compile(r"^BuildKit version:\s+(\S+)\s*$", re.MULTILINE)


class ArchiveRefused(ValueError):
    """An image archive that does not hold exactly one image, by its own bytes."""


class Refused(Exception):
    """What a role refuses, which it reports on a ``refused:`` line and with
    ``DIGEST_MISMATCH_EXIT``."""


@dataclass(frozen=True)
class OciImage:
    """The image an archive holds, by the digests its own bytes give."""

    #: ``sha256:<hex>`` of the one manifest the archive's index names.
    manifest_digest: str
    #: ``sha256:<hex>`` of that manifest's config.
    config_digest: str


class BuiltImage(BaseModel):
    """What a build role writes beside the archive it built: the archive's identity besides its
    sha256, and the BuildKit release that built it. An images cache entry's ``meta.json`` records
    it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    #: As ``docker buildx inspect`` names it, such as ``v0.35.1``.
    buildkit_version: str = Field(min_length=1)


class Pinned(BaseModel):
    """What a pin role writes once the daemon holds the image under the name it was handed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``<repository>:<tag>``.
    name: str
    #: The daemon's image ID, ``sha256:<hex>``: the manifest's digest or the config's.
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def _blob_name(digest: str) -> str:
    found = _DIGEST.fullmatch(digest) if isinstance(digest, str) else None
    if found is None:
        raise ArchiveRefused(f"{digest!r} is not a sha256 digest")
    return f"blobs/sha256/{found.group(1)}"


def _json(data: bytes | None, what: str) -> dict:
    if data is None:
        raise ArchiveRefused(f"it holds no {what}")
    try:
        value = json.loads(data)
    except ValueError as e:
        raise ArchiveRefused(f"its {what} is not JSON: {e}") from None
    if not isinstance(value, dict):
        raise ArchiveRefused(f"its {what} is not a JSON object")
    return value


def _checked_blob(read: Callable[[str], bytes | None], descriptor: object, what: str) -> bytes:
    """The bytes of the blob *descriptor* names, which must hash to its digest and have its size."""
    if not isinstance(descriptor, dict):
        raise ArchiveRefused(f"its {what} descriptor is not an object")
    digest, size = descriptor.get("digest"), descriptor.get("size")
    name = _blob_name(digest)
    data = read(name)
    if data is None:
        raise ArchiveRefused(f"it names {what} {digest}, which it does not hold")
    if "sha256:" + hashlib.sha256(data).hexdigest() != digest or len(data) != size:
        raise ArchiveRefused(f"its blob {name} does not hash to its name, or has not the size its descriptor gives")
    return data


def _docker_manifest(data: bytes | None, config: str, layers: list[str]) -> None:
    """Refuse a ``manifest.json`` that does not name exactly the config blob *config* and the layer
    blobs *layers*, in order, without a name: what the classic image store loads must be the image
    the index names."""
    if data is None:
        raise ArchiveRefused("it holds no manifest.json, which the classic image store loads")
    try:
        entries = json.loads(data)
    except ValueError as e:
        raise ArchiveRefused(f"its manifest.json is not JSON: {e}") from None
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ArchiveRefused("its manifest.json names other than one image")
    entry = entries[0]
    if entry.get("RepoTags"):
        raise ArchiveRefused("its manifest.json names the image, which a load would give it")
    if entry.get("Config") != config or entry.get("Layers") != layers:
        raise ArchiveRefused("its manifest.json names another config or other layers than its index's manifest")


def _image(read: Callable[[str], bytes | None], has: Callable[[str], bool]) -> OciImage:
    layout = _json(read("oci-layout"), "oci-layout file")
    if layout.get("imageLayoutVersion") != "1.0.0":
        raise ArchiveRefused(f"its oci-layout file names version {layout.get('imageLayoutVersion')!r}, not 1.0.0")
    index = _json(read("index.json"), "index.json")
    manifests = index.get("manifests")
    if not isinstance(manifests, list) or len(manifests) != 1 or not isinstance(manifests[0], dict):
        count = len(manifests) if isinstance(manifests, list) else "no list of"
        raise ArchiveRefused(f"its index.json names {count} manifests, not one")
    descriptor = manifests[0]
    if descriptor.get("mediaType") not in MANIFEST_TYPES:
        raise ArchiveRefused(f"its index.json names a {descriptor.get('mediaType')!r}, not an image manifest")
    named = sorted(NAME_ANNOTATIONS & set(descriptor.get("annotations") or {}))
    if named:
        raise ArchiveRefused(f"its index.json names the image ({', '.join(named)}), which a load would give it")
    manifest = _json(_checked_blob(read, descriptor, "manifest"), "manifest")
    config = manifest.get("config")
    _checked_blob(read, config, "config")
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        raise ArchiveRefused("its manifest has no list of layers")
    for layer in layers:
        if not isinstance(layer, dict) or not has(_blob_name(layer.get("digest"))):
            raise ArchiveRefused(f"its manifest names layer {layer!r}, which it does not hold")
    assert isinstance(config, dict)
    _docker_manifest(read("manifest.json"), _blob_name(config["digest"]),
                     [_blob_name(layer["digest"]) for layer in layers])
    return OciImage(manifest_digest=descriptor["digest"], config_digest=config["digest"])


def archive_image(archive: Path) -> OciImage:
    """The image the image archive *archive* holds. Raise ArchiveRefused when it does not hold
    exactly one, as the module docstring says, and OSError when it cannot be read."""
    try:
        with tarfile.open(archive, "r:") as tar:
            members: dict[str, tarfile.TarInfo] = {}
            for member in tar.getmembers():
                name = member.name.removeprefix("./")
                if name in members:
                    raise ArchiveRefused(f"member {name} appears twice")
                members[name] = member

            def read(name: str) -> bytes | None:
                member = members.get(name)
                if member is None or not member.isfile():
                    return None
                if member.size > JSON_MAX_BYTES and not name.startswith("blobs/"):
                    raise ArchiveRefused(f"member {name} is larger than any index")
                f = tar.extractfile(member)
                assert f is not None
                return f.read(JSON_MAX_BYTES + 1)

            def has(name: str) -> bool:
                member = members.get(name)
                return member is not None and member.isfile()

            return _image(read, has)
    except tarfile.TarError as e:
        raise ArchiveRefused(f"it is not a tar file: {e}") from None


def layout_image(layout: Path) -> OciImage:
    """The image the directory *layout*, an image archive unpacked, holds, by the rules of
    ``archive_image``."""

    def read(name: str) -> bytes | None:
        path = layout / name
        if not path.is_file() or path.is_symlink():
            return None
        with path.open("rb") as f:
            return f.read(JSON_MAX_BYTES + 1)

    def has(name: str) -> bool:
        path = layout / name
        return path.is_file() and not path.is_symlink()

    return _image(read, has)


def cpu_list(cpus: Sequence[int]) -> str:
    """*cpus* as a cpuset list, ranges joined: ``0-3,6``."""
    if not cpus:
        raise ValueError("no CPUs")
    ordered = sorted(set(cpus))
    ranges: list[list[int]] = []
    for cpu in ordered:
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def _own_limit(name: str, proc_cgroup: Path, root: Path) -> int | None:
    """The least value of the cgroup file *name* on the path from this process's cgroup to the root
    of its cgroup namespace; None when no cgroup on the path sets one."""
    line = next((x for x in proc_cgroup.read_text().splitlines() if x.startswith("0::")), None)
    if line is None:
        raise RuntimeError(f"{proc_cgroup} names no cgroup v2 path")
    here = root / line[3:].strip().lstrip("/")
    limits = []
    while True:
        try:
            text = (here / name).read_text().strip()
        except FileNotFoundError:
            text = "max"
        if text != "max":
            limits.append(int(text))
        if here == root or root not in here.parents:
            break
        here = here.parent
    return min(limits) if limits else None


def own_memory(*, proc_cgroup: Path = Path("/proc/self/cgroup"),
               root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """The memory limit, in bytes, of the cgroup this process runs in: the least ``memory.max``
    on the path from its cgroup to the root of its cgroup namespace, which in a container's own
    namespace is the container's limit. None when no cgroup on the path has a limit."""
    return _own_limit("memory.max", proc_cgroup, root)


def own_pids(*, proc_cgroup: Path = Path("/proc/self/cgroup"), root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """The tasks limit of the cgroup this process runs in, read as ``own_memory`` reads memory."""
    return _own_limit("pids.max", proc_cgroup, root)


#: What buildx's ``docker-container`` driver puts before a builder's name in the names of the
#: container and the state volume it makes for the builder's node.
BUILDER_OBJECT_PREFIX = "buildx_buildkit_"


def builder_objects(builder: str) -> tuple[str, str]:
    """The names of the container and the state volume buildx's ``docker-container`` driver makes
    for the one node of builder *builder*."""
    container = f"{BUILDER_OBJECT_PREFIX}{builder}0"
    return container, f"{container}_state"


def _run(argv: Sequence[str], env: Mapping[str, str], *, stdin: str | None = None,
         capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run *argv* in the environment *env*, with this process's stdout and stderr, or with stdout
    captured."""
    return subprocess.run(list(argv), input=stdin, text=True, check=False, env=dict(env),
                          stdout=subprocess.PIPE if capture else None,
                          stdin=None if stdin is not None else subprocess.DEVNULL)


def _volume_exists(docker: str, volume: str, env: Mapping[str, str]) -> bool:
    out = _run([docker, "volume", "ls", "--quiet", "--filter", f"name={volume}"], env, capture=True)
    if out.returncode != 0:
        raise RuntimeError(f"docker volume ls exited {out.returncode}")
    return volume in out.stdout.split()


def _container_exists(docker: str, container: str, env: Mapping[str, str]) -> bool:
    out = _run([docker, "container", "ls", "--all", "--format", "{{.Names}}", "--filter", f"name={container}"],
               env, capture=True)
    if out.returncode != 0:
        raise RuntimeError(f"docker container ls exited {out.returncode}")
    return container in out.stdout.split()


def remove_builder_objects(docker: str, builder: str, env: Mapping[str, str]) -> list[str]:
    """Remove the container and the state volume of builder *builder*, where they exist; return
    those that are still there after."""
    container, volume = builder_objects(builder)
    if _container_exists(docker, container, env):
        _run([docker, "container", "rm", "--force", "--volumes", container], env, capture=True)
    if _volume_exists(docker, volume, env):
        _run([docker, "volume", "rm", "--force", volume], env, capture=True)
    return [name for name, there in ((container, _container_exists(docker, container, env)),
                                     (volume, _volume_exists(docker, volume, env))) if there]


def _check_builder(docker: str, builder: str, env: Mapping[str, str], *, memory: int, cpus: str) -> None:
    """Fail unless the builder's container holds the memory, no swap and the CPUs it was given."""
    container, _ = builder_objects(builder)
    out = _run([docker, "container", "inspect", "--format", "{{json .HostConfig}}", container], env, capture=True)
    if out.returncode != 0:
        raise RuntimeError(f"the builder's container {container} cannot be inspected")
    host = json.loads(out.stdout)
    got = (host.get("Memory"), host.get("MemorySwap"), host.get("CpusetCpus"))
    if got != (memory, memory, cpus):
        raise RuntimeError(f"the builder's container {container} holds memory, memory and swap, and CPUs {got}, "
                           f"not {(memory, memory, cpus)}")


def driver_opt(key: str, value: str) -> str:
    """One ``--driver-opt`` of ``buildx create``, ``key=value``, as one CSV field: buildx reads the
    option as a CSV record, so a cpuset of several ranges (``0-7,128-135``) would otherwise split at
    its comma. Raise ValueError for a value with a double quote or a line break."""
    if '"' in value or "\n" in value or "\r" in value:
        raise ValueError(f"driver option {key} holds a character a CSV field cannot carry unescaped: {value!r}")
    return f'"{key}={value}"'


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _terminate(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def build(args: argparse.Namespace) -> int:
    """The build role's program (the module docstring): the builder is held to the role's own CPUs
    and memory."""
    user = os.environ.pop(REGISTRY_USER_VAR, "")
    token = os.environ.pop(REGISTRY_TOKEN_GRANT, "")
    if not os.environ.get("DOCKER_CONFIG"):
        _say("the build role's environment names no DOCKER_CONFIG, the directory its login and builder live in")
        return FAILED_EXIT
    if args.login and not (user and token):
        _say(f"the build logs in, and {REGISTRY_USER_VAR} or {REGISTRY_TOKEN_GRANT} is missing or empty")
        return FAILED_EXIT
    memory = own_memory()
    if memory is None:
        _say("the role has no memory limit to hold its builder to")
        return FAILED_EXIT
    contexts = sorted(p for p in args.contexts.iterdir() if p.is_dir()) if args.contexts.is_dir() else []
    return build_image(docker=args.docker, buildkit_image=args.buildkit_image, builder=args.builder,
                       recipe=args.recipe, contexts=contexts, out=args.out, digest=args.digest, record=args.record,
                       parent=args.parent, memory=memory, cpus=cpu_list(sorted(os.sched_getaffinity(0))),
                       env=dict(os.environ), login=(user, token) if args.login else None)


def build_image(*, docker: str, buildkit_image: str, builder: str, recipe: Path, contexts: Sequence[Path], out: Path,
                digest: Path, record: Path, parent: Path | None, memory: int, cpus: str, env: Mapping[str, str],
                login: tuple[str, str] | None = None) -> int:
    """Build *recipe* into the image archive *out* on a builder named *builder* held to *memory*
    bytes, no swap and the CPUs *cpus*, and write the archive's sha256 to *digest* and its
    ``BuiltImage`` to *record*; return the exit status a build role ends with. *contexts* are the
    named build contexts, each a directory named as its context; *parent*, for a layer or a cap, the
    parent's checked archive, unpacked beside *out* as the parent context and removed. The Docker
    CLI runs in *env*, whose ``DOCKER_CONFIG`` holds the builder's state and *login*'s credential,
    a user name and a token, while the build runs. Every exit removes the builder, its volume and
    the parent's layout. A refused parent prints ``refused:`` and gives ``DIGEST_MISMATCH_EXIT``."""
    if not env.get("DOCKER_CONFIG"):
        raise ValueError("a build's environment names the DOCKER_CONFIG its builder lives in")
    bad = [p.name for p in contexts if not _CONTEXT_NAME.fullmatch(p.name) or p.name == PARENT_CONTEXT]
    if bad:
        _say(f"build contexts {bad} have names a build context may not have")
        return FAILED_EXIT
    layout = out.parent / "parent-layout"
    try:
        return _build(docker=docker, buildkit_image=buildkit_image, builder=builder, recipe=recipe,
                      contexts=contexts, out=out, digest=digest, record=record, parent=parent, memory=memory,
                      cpus=cpus, env=env, login=login, layout=layout)
    except Refused as e:
        _say(f"refused: {e}")
        return DIGEST_MISMATCH_EXIT
    except (RuntimeError, OSError) as e:
        _say(str(e))
        return FAILED_EXIT
    finally:
        shutil.rmtree(layout, ignore_errors=True)


def _build(*, docker: str, buildkit_image: str, builder: str, recipe: Path, contexts: Sequence[Path], out: Path,
           digest: Path, record: Path, parent: Path | None, memory: int, cpus: str, env: Mapping[str, str],
           login: tuple[str, str] | None, layout: Path) -> int:
    """``build_image`` once its inputs are checked: the parent unpacked, the builder made, used and
    removed whatever happens, the archive's identity and digest written."""
    empty = out.parent / "empty-context"
    empty.mkdir(exist_ok=True)
    flags = [f"--build-context={p.name}={p}" for p in contexts]
    if parent is not None:
        try:
            with tarfile.open(parent, "r:") as tar:
                tar.extractall(layout, filter="data")
            parent.unlink()
            parent_image = layout_image(layout)
        except ArchiveRefused as e:
            raise Refused(f"the parent's image archive {parent}: {e}") from None
        except tarfile.TarError as e:
            raise Refused(f"the parent's image archive {parent} is not a tar file: {e}") from None
        flags.append(f"--build-context={PARENT_CONTEXT}={parent_context(layout, parent_image.manifest_digest)}")
    left = remove_builder_objects(docker, builder, env)
    if left:
        raise RuntimeError(f"an earlier builder's {left} cannot be removed")
    logged_in = False
    try:
        if login is not None:
            user, token = login
            if _run([docker, "login", "--username", user, "--password-stdin"], env, stdin=token).returncode != 0:
                raise RuntimeError("docker login failed")
            logged_in = True
        opts = [driver_opt("image", buildkit_image), driver_opt("memory", str(memory)),
                driver_opt("memory-swap", str(memory)), driver_opt("cpuset-cpus", cpus)]
        made = _run([docker, "buildx", "create", "--name", builder, "--driver", "docker-container",
                     *(word for opt in opts for word in ("--driver-opt", opt))], env)
        if made.returncode != 0:
            raise RuntimeError(f"docker buildx create exited {made.returncode}")
        booted = _run([docker, "buildx", "inspect", "--bootstrap", builder], env)
        if booted.returncode != 0:
            raise RuntimeError(f"docker buildx inspect --bootstrap exited {booted.returncode}")
        # The bootstrap's own status read shares its 20 s timeout with the boot, which a loaded host
        # can outlast, so the builder's status is read again once it runs, by an inspect with a
        # timeout of its own. buildx before 0.34 takes no --timeout on inspect.
        inspected = _run([docker, "buildx", "inspect", builder], env, capture=True)
        sys.stdout.write(inspected.stdout)
        version = _BUILDKIT_VERSION.search(inspected.stdout)
        if inspected.returncode != 0 or version is None:
            raise RuntimeError(f"docker buildx inspect exited {inspected.returncode} and named no BuildKit version")
        _check_builder(docker, builder, env, memory=memory, cpus=cpus)
        built = _run([docker, "buildx", "build", "--builder", builder, "--progress=plain", "--provenance=false",
                      "--sbom=false", "--file", str(recipe), *flags,
                      f"--output={ARCHIVE_OUTPUT},dest={out}", str(empty)], env)
        if built.returncode != 0:
            raise RuntimeError(f"docker buildx build exited {built.returncode}")
    finally:
        _run([docker, "buildx", "rm", "--force", builder], env)
        if logged_in:
            _run([docker, "logout"], env)
        (Path(env["DOCKER_CONFIG"]) / "config.json").unlink(missing_ok=True)
        left = remove_builder_objects(docker, builder, env)
        if left:
            _say(f"the builder's {left} could not be removed")
    if left:
        return FAILED_EXIT
    try:
        image = archive_image(out)
    except ArchiveRefused as e:
        raise RuntimeError(f"the build wrote {out}, which is not one image: {e}") from None
    sha256, _ = sha256_file(out)
    built_image = BuiltImage(manifest_digest=image.manifest_digest, config_digest=image.config_digest,
                             buildkit_version=version.group(1))
    record.write_text(built_image.model_dump_json() + "\n")
    digest.write_text(sha256 + "\n")
    return 0


def pin(args: argparse.Namespace) -> int:
    """The pin role's program (the module docstring)."""
    repo, tag = args.name.rsplit(":", 1)
    try:
        try:
            image = archive_image(args.copy)
        except ArchiveRefused as e:
            raise Refused(f"the image archive {args.source}: {e}") from None
        with DockerClient(args.socket) as client:
            loaded = client.load_image(args.copy, timeout=LOAD_TIMEOUT_SEC)
            if len(loaded) != 1:
                raise Refused(f"the Docker daemon loaded {len(loaded)} images from {args.source}, not one")
            info = client.inspect_image(loaded[0])
            if info.id not in (image.manifest_digest, image.config_digest):
                raise Refused(f"the Docker daemon loaded {args.source} as image {info.id}, which is neither its "
                              f"manifest {image.manifest_digest} nor its config {image.config_digest}")
            client.tag_image(info.id, repo=repo, tag=tag)
            named = client.inspect_image(args.name)
            if named.id != info.id:
                raise Refused(f"the name {args.name} reaches image {named.id}, not the {info.id} just loaded")
    except Refused as e:
        _say(f"refused: {e}")
        return DIGEST_MISMATCH_EXIT
    except DockerError as e:
        _say(f"the image archive {args.source} could not be pinned: {e}")
        return FAILED_EXIT
    record = Pinned(name=args.name, image_id=info.id, manifest_digest=image.manifest_digest,
                    config_digest=image.config_digest)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text(record.model_dump_json() + "\n")
    os.replace(partial, args.out)
    args.copy.unlink()
    return 0


#: Where the install container sees the checkout's tree, the wheels and its log FIFOs, each bound
#: read-only. Under /dev, the container's own tmpfs, the bind targets leave nothing in what is
#: committed; a target in the image's tree leaves its directory there.
TREE_MOUNT, WHEELS_MOUNT, LOG_MOUNT = "/dev/hls/tree", "/dev/hls/wheels", "/dev/hls/log"
#: The capabilities the install container holds, each because a step of the install fails without
#: it, as measured. DAC_READ_SEARCH: the staging copy (``cp -fLr`` of the tree) reads a file of the
#: tree only its owner, the host user, may read, as the ``.def``'s ``%files`` copy, run as that
#: user, does; it reads and searches, and writes nothing. No other: as uid 0 the container owns the
#: image's files, so pip's writes into /opt/venv, the copy to /workspace, the removals and the
#: runscript need no override.
INSTALL_CAPABILITIES: tuple[str, ...] = ("DAC_READ_SEARCH",)
#: The label the job's stopper finds the install container by, with the job's id.
JOB_LABEL = "hls.job"
#: Seconds one read or write of exporting the built image or reading a tree may wait.
EXPORT_TIMEOUT_SEC = 600.0


def install_script(*, project: str, server: str, nonce: str, constraints: str | None) -> str:
    """The shell text the install container runs: its output to the log FIFOs; the tree and the
    wheels staged as the ``.def``'s ``%files`` stages them, by apptainer's own ``cp -fLr``; the
    install's script, the text the ``.def``'s ``%post`` holds (``hls.package.offline``); and, once it
    succeeded, the runscript, which serves as the ``.def``'s ``%runscript`` does. Raise ValueError
    for a name, server, constraints path or nonce a build must not carry."""
    post, run = offline_install(project=project, server=server, nonce=nonce, constraints=constraints)
    return "\n".join([
        f"exec >{LOG_MOUNT}/out 2>{LOG_MOUNT}/err",
        f"cp -fLr {TREE_MOUNT} {STAGE} || exit 1",
        f"cp -fLr {WHEELS_MOUNT} /wheels || exit 1",
        *post,
        f"mkdir -p {posixpath.dirname(DOCKER_RUNSCRIPT)} || exit 1",
        f"printf '%s\\n' '#!/bin/sh' {shlex.quote(run)} > {DOCKER_RUNSCRIPT} || exit 1",
        f"chmod 0755 {DOCKER_RUNSCRIPT} || exit 1",
    ]) + "\n"


#: The fields of an image's configuration (the OCI image config's ``config``) a committed image takes
#: from its base, as the daemon reports them; the rest of what the daemon reports is the container's.
IMAGE_CONFIG_FIELDS = ("User", "ExposedPorts", "Env", "Entrypoint", "Cmd", "Volumes", "WorkingDir", "Labels",
                       "StopSignal", "Shell", "Healthcheck", "OnBuild", "ArgsEscaped")


def image_config(reported: Mapping[str, object]) -> dict[str, object]:
    """The image configuration of the fields in ``IMAGE_CONFIG_FIELDS`` that *reported*, a daemon's
    report of an image's ``Config``, sets: an unset or empty field is left out, as an image's own
    configuration leaves it."""
    return {k: reported[k] for k in IMAGE_CONFIG_FIELDS if reported.get(k) not in (None, "", [], {}, False)}


def with_config(archive: Path, config: Mapping[str, object]) -> OciImage:
    """Give the image of *archive* the configuration *config* in place of its own: a new config blob,
    and a manifest, ``index.json`` and ``manifest.json`` that name it, the layers unchanged. The
    daemon's commit merges the container's command, user, environment and labels into the image,
    which ``config`` replaces. Return the new image's identity."""
    image = archive_image(archive)
    rewritten = archive.with_name(archive.name + ".rewritten")
    with tarfile.open(archive) as src, tarfile.open(rewritten, "x") as dst:

        def read(name: str) -> bytes:
            f = src.extractfile(name)
            assert f is not None
            return f.read()

        def blob(data: bytes) -> tuple[str, str]:
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            return digest, _blob_name(digest)

        old_config, old_manifest = _blob_name(image.config_digest), _blob_name(image.manifest_digest)
        full = json.loads(read(old_config))
        full["config"] = dict(config)
        config_bytes = json.dumps(full, separators=(",", ":")).encode()
        config_digest, config_name = blob(config_bytes)
        manifest = json.loads(read(old_manifest))
        manifest["config"] = {**manifest["config"], "digest": config_digest, "size": len(config_bytes)}
        manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
        manifest_digest, manifest_name = blob(manifest_bytes)
        index = json.loads(read("index.json"))
        index["manifests"] = [{**index["manifests"][0], "digest": manifest_digest, "size": len(manifest_bytes)}]
        docker_manifest = json.loads(read("manifest.json"))
        docker_manifest[0]["Config"] = config_name
        replaced = {"index.json": json.dumps(index, separators=(",", ":")).encode(),
                    "manifest.json": json.dumps(docker_manifest, separators=(",", ":")).encode()}
        for member in src.getmembers():
            name = member.name.removeprefix("./")
            if name in (old_config, old_manifest):
                continue
            if name in replaced:
                info = tarfile.TarInfo(member.name)
                info.size, info.mode, info.mtime = len(replaced[name]), member.mode, member.mtime
                dst.addfile(info, io.BytesIO(replaced[name]))
            else:
                dst.addfile(member, src.extractfile(member) if member.isfile() else None)
        for name, data in ((config_name, config_bytes), (manifest_name, manifest_bytes)):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o444
            dst.addfile(info, io.BytesIO(data))
    os.replace(rewritten, archive)
    return archive_image(archive)


def _relay(path: Path, out) -> tuple[threading.Thread, int]:
    """Open the FIFO *path* for reading, and for writing so it reads no end before the container
    opens it, and copy what arrives to *out* in a thread. Return the thread and the write end, whose
    closing, once the container ended, ends the copy."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    keep = os.open(path, os.O_WRONLY)
    os.set_blocking(fd, True)

    def copy() -> None:
        with os.fdopen(fd, "rb", buffering=0) as f:
            while chunk := f.read(65536):
                out.buffer.write(chunk)
                out.buffer.flush()

    thread = threading.Thread(target=copy, daemon=True)
    thread.start()
    return thread, keep


def submit(args: argparse.Namespace) -> int:
    """The submission build's program on a Docker host (``DockerImages.submission_role``): install
    the submission into a container of the submission base, as uid 0 holding
    ``INSTALL_CAPABILITIES`` alone, with no network, no device and no socket, the tree and the
    wheels bound read-only, and this role's own CPUs, memory and tasks; then commit it with the
    base's configuration, export the image to *out* in the form both image stores load, write its
    sha256 to *digest*, and remove the container and the committed image. The container runs only
    ``install_script``; its output reaches this role's own stdout and stderr, so the stage reads
    the install's status lines there."""
    memory, pids = own_memory(), own_pids()
    if memory is None or pids is None:
        _say("the role has no memory or tasks limit to hold the install container to")
        return FAILED_EXIT
    try:
        script = install_script(project=args.project, server=args.server, nonce=args.nonce,
                                constraints=args.constraints)
    except ValueError as e:
        _say(f"the install cannot be rendered: {e}")
        return FAILED_EXIT
    log = args.out.parent / "install-log"
    log.mkdir(mode=0o711)
    log.chmod(0o711)
    for name in ("out", "err"):
        os.mkfifo(log / name)
        # The container's root holds no capability, so it writes these as any other user would.
        (log / name).chmod(0o622)
    relays = [_relay(log / "out", sys.stdout), _relay(log / "err", sys.stderr)]
    repo, tag = args.name.rsplit(":", 1)
    cid = image_id = None
    status = FAILED_EXIT
    with DockerClient(args.socket) as client:
        try:
            config = ContainerConfig(
                image=args.base, argv=("/bin/sh", "-c", script), user="0:0",
                labels={JOB_LABEL: args.job, "hls.role": "submission-install"}, network_mode="none",
                cpuset=tuple(sorted(os.sched_getaffinity(0))), memory_bytes=memory, pids_limit=pids,
                mounts=(BindMount(args.tree, TREE_MOUNT, read_only=True),
                        BindMount(args.wheels, WHEELS_MOUNT, read_only=True),
                        BindMount(log, LOG_MOUNT, read_only=True)),
                # No init process: the daemon binds its init binary into the image's tree, which the
                # commit would keep.
                cap_add=INSTALL_CAPABILITIES, read_only_root=False, init=False)
            base = client.inspect_image(args.base)
            cid = client.create_container(config)
            client.start_container(cid)
            while (ended := client.wait_container(cid, timeout=60)) is None:
                pass
            if ended.status_code != 0:
                _say(f"the install ended with status {ended.status_code}")
                return FAILED_EXIT
            image_id = client.commit_container(cid, repo=repo, tag=tag, config=image_config(base.config))
            client.save_image(image_id, args.out, timeout=EXPORT_TIMEOUT_SEC)
            image = archive_image(args.out)
            if image_id not in (image.manifest_digest, image.config_digest):
                raise Refused(f"the daemon exported image {image_id} as {image.manifest_digest}")
            # The image is the base's configuration over the install's files, as the .def's is.
            with_config(args.out, image_config(base.config))
            status = 0
        except Refused as e:
            _say(f"refused: {e}")
            status = DIGEST_MISMATCH_EXIT
        except (DockerError, ArchiveRefused, OSError) as e:
            _say(f"the submission image could not be built: {e}")
            status = FAILED_EXIT
        finally:
            for thread, keep in relays:
                os.close(keep)
            if cid is not None:
                try:
                    client.remove_container(cid, force=True)
                except DockerError as e:
                    _say(f"the install container {cid} could not be removed: {e}")
                    status = status or FAILED_EXIT
            if image_id is not None:
                try:
                    client.remove_image(args.name)
                except DockerError as e:
                    _say(f"the committed image {args.name} could not be removed: {e}")
                    status = status or FAILED_EXIT
            for thread, _ in relays:
                thread.join(timeout=60)
    if status == 0:
        sha256, _ = sha256_file(args.out)
        args.digest.write_text(sha256 + "\n")
    else:
        args.out.unlink(missing_ok=True)
    return status


def read(args: argparse.Namespace) -> int:
    """The tree read's program (``DockerImages.tree_role``): put what the image holds at *path* into
    *dest* as files of this user, by the daemon's archive of a container of the image that never
    starts. A path the image does not hold, or holds as no directory, is refused with
    ``TREE_ABSENT_EXIT``. The image's root is never copied: an image is read whole from its archive
    (``hls.package.layers``)."""
    if args.path.rstrip("/") == "":
        _say("a tree read copies one directory of an image, never its root")
        return FAILED_EXIT
    tar_file = args.dest.parent / f".{args.dest.name}.tar"
    cid = None
    try:
        with DockerClient(args.socket) as client:
            config = ContainerConfig(image=args.image, argv=("/bin/true",), user=f"{os.getuid()}:{os.getgid()}",
                                     labels={JOB_LABEL: args.job, "hls.role": "tree-read"}, network_mode="none",
                                     cpuset=(min(os.sched_getaffinity(0)),), memory_bytes=64 << 20, pids_limit=8)
            cid = client.create_container(config)
            try:
                client.container_archive(cid, args.path, tar_file, timeout=EXPORT_TIMEOUT_SEC)
            except NotFound:
                _say(f"refused: image {args.image} holds no directory at {args.path}")
                return TREE_ABSENT_EXIT
            finally:
                client.remove_container(cid, force=True)
        with tarfile.open(tar_file) as tar:
            members = tar.getmembers()
            for member in members:
                # One image store names the members of a copy from the root, with a leading slash.
                member.name = member.name.lstrip("/")
                if member.islnk():
                    member.linkname = member.linkname.lstrip("/")
            top = posixpath.basename(args.path.rstrip("/"))
            if not members or members[0].name.rstrip("/") not in (top, "./" + top) or not members[0].isdir():
                _say(f"refused: image {args.image} holds no directory at {args.path}")
                return TREE_ABSENT_EXIT
            prefix = members[0].name.rstrip("/") + "/"
            for member in members[1:]:
                if not member.name.startswith(prefix) or member.ischr() or member.isblk() or member.isfifo():
                    continue
                member.name = member.name[len(prefix):]
                if member.islnk():
                    if not member.linkname.startswith(prefix):
                        continue
                    member.linkname = member.linkname[len(prefix):]
                tar.extract(member, args.dest, filter="tar")
    except (DockerError, OSError, tarfile.TarError) as e:
        _say(f"the tree at {args.path} of image {args.image} could not be read: {e}")
        return FAILED_EXIT
    finally:
        tar_file.unlink(missing_ok=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--docker", required=True)
    b.add_argument("--buildkit-image", required=True)
    b.add_argument("--builder", required=True)
    b.add_argument("--recipe", type=Path, required=True)
    b.add_argument("--contexts", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--digest", type=Path, required=True)
    b.add_argument("--record", type=Path, required=True)
    b.add_argument("--parent", type=Path)
    b.add_argument("--login", action="store_true")
    p = sub.add_parser("pin")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--copy", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--socket", type=Path, required=True)
    m = sub.add_parser("submit")
    m.add_argument("--socket", type=Path, required=True)
    m.add_argument("--base", required=True)
    m.add_argument("--tree", type=Path, required=True)
    m.add_argument("--wheels", type=Path, required=True)
    m.add_argument("--project", required=True)
    m.add_argument("--server", required=True)
    m.add_argument("--nonce", required=True)
    m.add_argument("--constraints")
    m.add_argument("--job", required=True)
    m.add_argument("--name", required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--digest", type=Path, required=True)
    r = sub.add_parser("read")
    r.add_argument("--socket", type=Path, required=True)
    r.add_argument("--image", required=True)
    r.add_argument("--path", required=True)
    r.add_argument("--dest", type=Path, required=True)
    r.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _terminate)
    programs = {"build": build, "pin": pin, "submit": submit, "read": read}
    return programs[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
