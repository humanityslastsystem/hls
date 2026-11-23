"""Images on a Docker host: image archives in the store, built by BuildKit and pinned into the
node's Docker daemon (``DockerImages``, the Docker ``ImageRuntime``).

An images cache entry of a Docker image holds ``IMAGE_FILE``, the image archive (an OCI image
layout with a Docker ``manifest.json`` beside it, in one tar file, which both kinds of image store
load: ``image_role.archive_image``), ``RECIPE_FILE``, the Dockerfile it was built from,
``inventory.json``, from ``inventory.sh`` run in a container of the pinned image as for every
image, and ``meta.json``, which records the build role's ``BuiltImage``: the manifest digest, the
config digest and the BuildKit release. The image is identified by the archive's sha256 with its
manifest digest, both fixed by the archive's bytes on every host. The image ID a daemon gives it
depends on its image store: the manifest digest in containerd's store, the config digest in the
classic one. So a pin accepts either of the two digests the archive names, a container is created
by the ID the pin found (``PinnedImage.ref``), and the job's name for the image reaches it in
either store.

A build's role directory holds the Dockerfile, ``RECIPE_FILE``, and beside it ``CONTEXTS``, one
directory per named build context (``lay_out_build`` writes the lock files' one); the role
copies a layer's or a cap's parent archive into the directory and unpacks it as the parent
context (``hls.substrate.docker.image_role``).

A pin gives the image a name in the daemon that belongs to the job: the repository
``hls/job-<job id>`` and the tag the archive's sha256 (``image_name``). The job id is the one the
launcher labels the job's containers with, so the job's end finds the job's names as it finds
its containers. An image lives in the daemon while one name or container holds it, so removing
a job's name never removes an image another job pinned or a container of its own still runs.
The pin role also leaves a record of what it pinned in the job's scratch (``RECORDS``), against
which a later stage of the job checks an image an earlier one could not remove (``left``). A
build role's BuildKit builder is named after the job and the role (``builder_name``).

At the job's end, after its containers are gone, ``remove_job_leftovers`` removes what a role
killed before its own cleanup left: the job's builders, their state volumes and the job's image
names. It removes names only, never by force, so an image another job also names stays.

A submission image is never built by BuildKit, whose builder runs privileged: ``submission_role``
runs ``image_role.submit``, which installs the submission into an ordinary container of the pinned
submission base, commits and exports it. This runtime holds no image as a tree on the host. The
package's checks read an image whole from the archive it was pinned from (``reader_args``,
``hls.package.layers``), never through the daemon; ``tree_role`` copies one directory of an image,
never its root, out of the daemon (``image_role.read``).

Roles that run the Docker CLI or speak to the daemon (the build, the pin, the submission's build and
a tree read) need the daemon's socket in their container, at the path ``socket`` names; no other
role's container gets it."""

from __future__ import annotations

import os
import posixpath
import re
import shutil
from collections.abc import Mapping
from pathlib import Path

from hls.recipes.model import ImageDecl, RegistryImage, UvPin
from hls.recipes.render_docker import LOCKS_CONTEXT, build_inputs, dockerfile_key, render_dockerfile
from hls.recipes.render_docker import RUNSCRIPT as DOCKER_RUNSCRIPT
from hls.store.digest import check_sha256_hex
from hls.substrate.container import ImageRuntime, PinnedImage
from hls.substrate.docker.api import DEFAULT_SOCKET, DockerClient, DockerError, NotFound, RequestRefused
from hls.substrate.docker.image_role import BUILDER_OBJECT_PREFIX, FETCH_EXEC, BuiltImage, Pinned

#: The module the build and pin roles run as ``python -m``.
ROLE_MODULE = "hls.substrate.docker.image_role"
#: The file an images cache entry, and a build's directory, hold a Docker image in.
IMAGE_FILE = "image.oci.tar"
#: The file an images cache entry, and a build's directory, hold the image's Dockerfile in.
RECIPE_FILE = "Dockerfile"
#: The directory beside a build's Dockerfile that holds one directory per named build context.
CONTEXTS = "contexts"
#: The file a build role writes its ``BuiltImage`` to, in its own directory.
BUILT_FILE = "image.json"
#: The file a pin role writes its ``Pinned`` to, in its own directory.
PINNED_FILE = "pinned.json"
#: The file a build role copies a layer's or a cap's parent archive to, in its own directory.
PARENT_FILE = "parent.oci.tar"
#: The directory, in the job's node-local scratch, of the records of the images the job pinned.
RECORDS = "docker-images"
#: The repository every image name of hls's starts with.
REPO = "hls"
#: The program a submission image runs.
RUNSCRIPT = DOCKER_RUNSCRIPT
# The only characters a path on a role's command line may hold: buildx reads its build contexts
# and its output as comma-separated key=value lists, and a parent context as <dir>@<digest>.
_SAFE_PATH = re.compile(r"/[A-Za-z0-9._+/-]*")
_ROLE_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
_JOB_ID = re.compile(r"[a-z0-9]+")
_IMAGE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]*")


def _path(what: str, path: Path) -> str:
    """*path* as a role's command line passes it; raise ValueError unless it is absolute, in
    normal form and of the characters buildx reads back exactly."""
    text = str(path)
    if not text.startswith("/") or text.startswith("//") or posixpath.normpath(text) != text:
        raise ValueError(f"{what} must be an absolute path in normal form, got {text!r}")
    if not _SAFE_PATH.fullmatch(text):
        raise ValueError(f"{what} {text!r} cannot be passed to buildx exactly: the allowed characters are "
                         "[A-Za-z0-9._+/-]")
    return text


def _job(job_id: str) -> str:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError(f"job id {job_id!r} cannot name a Docker image: it is lowercase letters and digits")
    return job_id


def job_repository(job_id: str) -> str:
    """The repository of the image names of job *job_id*."""
    return f"{REPO}/job-{_job(job_id)}"


def image_name(job_id: str, sha256: str) -> tuple[str, str]:
    """The repository and tag job *job_id* names the image of the archive whose bare hex sha256 is
    *sha256* by."""
    return job_repository(job_id), check_sha256_hex(sha256)


def builder_prefix(job_id: str) -> str:
    """The start of the name of every BuildKit builder of job *job_id*."""
    return f"hls-job-{_job(job_id)}-"


def builder_name(job_id: str, own: Path) -> str:
    """The name of the BuildKit builder of the build role whose own directory is *own*, in job
    *job_id*: fixed by the two, so the next attempt of the role finds a builder a killed one left.
    Raise ValueError for a role directory whose name a builder's cannot hold."""
    if not _ROLE_NAME.fullmatch(own.name):
        raise ValueError(f"role directory {own} has a name a builder's name cannot hold")
    return builder_prefix(job_id) + own.name


def job_leftovers(client: DockerClient, job_id: str) -> list[str]:
    """What of job *job_id* the daemon holds that a role's own cleanup removes: its builders'
    containers and state volumes (``image_role.builder_objects``) and its image names."""
    prefix = BUILDER_OBJECT_PREFIX + builder_prefix(job_id)
    found = [n for row in client.list_containers_named(prefix) for n in row.names if n.startswith(prefix)]
    found += client.list_volumes(prefix)
    found += [tag for image in client.list_images(repository=job_repository(job_id)) for tag in image.repo_tags]
    return found


def remove_job_leftovers(client: DockerClient, job_id: str) -> None:
    """Remove, at the end of job *job_id* and once none of its containers is left, the builders,
    their state volumes and the image names its roles left: a builder of a build role that was
    killed, and the names of a stage that could not remove them. Each image name is removed
    without force, so an image another job also names stays, and one a container still uses is
    refused. Raise RuntimeError naming what is left when anything is."""
    prefix = BUILDER_OBJECT_PREFIX + builder_prefix(job_id)
    for row in client.list_containers_named(prefix):
        try:
            client.remove_container(row.id, force=True)
        except NotFound:
            pass
    for volume in client.list_volumes(prefix):
        try:
            client.remove_volume(volume)
        except NotFound:
            pass
    refused = []
    for image in client.list_images(repository=job_repository(job_id)):
        for tag in image.repo_tags:
            try:
                client.remove_image(tag)
            except NotFound:
                pass
            except RequestRefused as e:
                refused.append(f"{tag} ({e.status})")
    left = job_leftovers(client, job_id)
    if left or refused:
        raise RuntimeError(f"job {job_id} leaves {sorted(set(left))} in the Docker daemon"
                           + (f"; refused: {refused}" if refused else ""))


def _tree_path(path: str) -> str:
    if not path.startswith("/") or posixpath.normpath(path) != path:
        raise ValueError(f"a path in an image is absolute and in normal form, not {path!r}")
    return path


def lay_out_build(work: Path, dockerfile: str, *, locks: Mapping[str, Path]) -> Path:
    """Write the build of one image into *work*, its build role's own directory: the Dockerfile
    *dockerfile* and the ``LOCKS_CONTEXT`` build context, which holds a copy of each file *locks*
    names under its name there (``render_docker.build_inputs``) and nothing else. Each copy keeps
    its file's mode, which ``COPY`` gives the file in the image, as a ``.def``'s ``%files`` does.
    Return the Dockerfile's path."""
    bad = [name for name in locks if "/" in name or name in ("", ".", "..")]
    if bad:
        raise ValueError(f"{bad} are not file names of the locks context")
    recipe = work / RECIPE_FILE
    recipe.write_text(dockerfile)
    context = work / CONTEXTS / LOCKS_CONTEXT
    context.mkdir(parents=True)
    for name, host in locks.items():
        shutil.copy(host, context / name)
    return recipe


def reported_build(path: Path) -> BuiltImage | None:
    """The ``BuiltImage`` a build role wrote to *path*; None when the file is missing or does not
    hold one."""
    try:
        return BuiltImage.model_validate_json(path.read_text())
    except (OSError, ValueError):
        return None


def _oserror(what: str, e: DockerError) -> OSError:
    return OSError(f"{what}: {e}")


class DockerImages(ImageRuntime):
    """Images as image archives, each pinned by loading a checked copy into the node's Docker
    daemon under a name of the job's (the module docstring)."""

    image_file = IMAGE_FILE
    runscript = RUNSCRIPT
    recipe_file = RECIPE_FILE
    parent_file = PARENT_FILE

    def __init__(self, scratch: Path, *, job_id: str, python: str, buildkit_image: str,
                 socket: Path = DEFAULT_SOCKET, docker: str = "docker") -> None:
        if not _IMAGE_REF.fullmatch(buildkit_image) or "," in buildkit_image:
            raise ValueError(f"{buildkit_image!r} is not an image reference")
        #: The job's node-local scratch, which holds the records of its pins.
        self.scratch = scratch
        #: The job's id, which names its images and builders.
        self.job_id = _job(job_id)
        #: The interpreter a role runs the role program with.
        self.python = python
        #: The daemon's reference of the BuildKit image a builder runs: its image ID, or a name.
        self.buildkit_image = buildkit_image
        #: The daemon's socket, at the same path in every role container that is given it.
        self.socket = socket
        #: The Docker CLI with its buildx plugin: a name found on the role's PATH, or an absolute path.
        self.docker = docker
        # The archive and its sha256 of each pin role not yet finished, by the role's own directory.
        self._pending: dict[Path, tuple[str, Path]] = {}

    def _record(self, sha256: str) -> Path:
        return self.scratch / RECORDS / f"{check_sha256_hex(sha256)}.json"

    def _name(self, sha256: str) -> str:
        repo, tag = image_name(self.job_id, sha256)
        return f"{repo}:{tag}"

    def _untag(self, client: DockerClient, name: str) -> None:
        """Remove the name *name*, and with it the image once nothing else holds it; nothing when
        the daemon has no such name."""
        try:
            client.remove_image(name)
        except NotFound:
            pass

    def left(self, sha256: str, *, contestant: bool) -> PinnedImage | str | None:
        record_path, name = self._record(sha256), self._name(sha256)
        try:
            record: Pinned | None = Pinned.model_validate_json(record_path.read_text())
        except FileNotFoundError:
            record = None
        except (OSError, ValueError) as e:
            return f"the record {record_path} of a pin cannot be read: {e}"
        try:
            with DockerClient(self.socket) as client:
                try:
                    found: str | None = client.inspect_image(name).id
                except NotFound:
                    found = None
        except DockerError as e:
            raise _oserror(f"image {name} could not be looked up", e) from e
        if record is None and found is None:
            return None
        if record is None:
            return f"the Docker daemon holds image {name}, which no record of a pin names"
        if found is None:
            return f"the Docker daemon holds no image {name}, which a pin recorded"
        if record.name != name or found != record.image_id:
            return f"image {name} is {found}, not the {record.image_id} its pin recorded"
        return PinnedImage(ref=record.image_id, file_sha256=sha256,
                           check_sha256=record.manifest_digest.removeprefix("sha256:"), contestant=contestant)

    def remove_left(self, sha256: str) -> None:
        name = self._name(sha256)
        try:
            with DockerClient(self.socket) as client:
                self._untag(client, name)
        except DockerError as e:
            raise _oserror(f"image {name} could not be removed", e) from e
        self._record(sha256).unlink(missing_ok=True)

    def pin_role(self, image_file: Path, *, sha256: str, own: Path) -> tuple[list[str], dict[str, str]]:
        check_sha256_hex(sha256)
        source, copy = _path("image file", image_file), _path("copy", own / IMAGE_FILE)
        out, socket = _path("pin record", own / PINNED_FILE), _path("Docker socket", self.socket)
        (self.scratch / RECORDS).mkdir(mode=0o700, parents=True, exist_ok=True)
        argv = ["/bin/sh", "-c", FETCH_EXEC, "hls-pin", source, copy, sha256,
                self.python, "-m", ROLE_MODULE, "pin", "--source", source, "--copy", copy,
                "--name", self._name(sha256), "--out", out, "--socket", socket]
        self._pending[own] = (sha256, image_file)
        return argv, {}

    def pinned(self, own: Path, *, sha256: str, contestant: bool) -> PinnedImage:
        expected, image_file = self._pending.pop(own)
        if expected != sha256:
            raise ValueError(f"the pin role in {own} pinned the archive of {expected}, not of {sha256}")
        name = self._name(sha256)
        try:
            with DockerClient(self.socket) as client:
                try:
                    record = Pinned.model_validate_json((own / PINNED_FILE).read_text())
                except (OSError, ValueError):
                    self._untag(client, name)
                    raise ValueError(f"ended without loading {image_file} into the Docker daemon") from None
                try:
                    found: str | None = client.inspect_image(name).id
                except NotFound:
                    found = None
                if record.name != name or found != record.image_id:
                    self._untag(client, name)
                    raise ValueError(f"ended with image {name} at {found}, not at the {record.image_id} it recorded")
        except DockerError as e:
            raise _oserror(f"image {name} could not be checked", e) from e
        path = self._record(sha256)
        partial = path.with_name(path.name + ".partial")
        partial.write_text(record.model_dump_json() + "\n")
        os.replace(partial, path)
        return PinnedImage(ref=record.image_id, file_sha256=sha256,
                           check_sha256=record.manifest_digest.removeprefix("sha256:"), contestant=contestant)

    def discard(self, own: Path) -> None:
        pending = self._pending.pop(own, None)
        if pending is None:
            return
        name = self._name(pending[0])
        try:
            with DockerClient(self.socket) as client:
                self._untag(client, name)
        except DockerError as e:
            raise _oserror(f"image {name} could not be removed", e) from e

    def remove(self, image: PinnedImage) -> None:
        name = self._name(image.file_sha256)
        try:
            with DockerClient(self.socket) as client:
                self._untag(client, name)
        except DockerError as e:
            raise _oserror(f"image {name} could not be removed", e) from e
        self._record(image.file_sha256).unlink(missing_ok=True)

    @staticmethod
    def recipe_key(decl: ImageDecl, *, parent_key: str | None, locks_dir: Path, uv: UvPin, tasks_dir: Path,
                   registry: RegistryImage | None = None) -> str:
        return dockerfile_key(decl, parent_key=parent_key, locks_dir=locks_dir, uv=uv, tasks_dir=tasks_dir,
                              registry=registry)

    def write_recipe(self, decl: ImageDecl, work: Path, *, locks_dir: Path, uv: UvPin, tasks_dir: Path,
                     registry: RegistryImage | None = None) -> str:
        # A layer's or a cap's build unpacks its parent's copy, PARENT_FILE, itself; the Dockerfile
        # names the parent by its build context only.
        dockerfile = render_dockerfile(decl, locks=locks_dir, uv=uv, tasks_dir=tasks_dir, registry=registry)
        lay_out_build(work, dockerfile, locks=build_inputs(decl, locks_dir=locks_dir, tasks_dir=tasks_dir).locks)
        return dockerfile

    def build_record(self, work: Path, stderr: str) -> dict[str, object]:
        built = reported_build(work / BUILT_FILE)
        if built is None:
            raise ValueError(f"ended without recording the image it built in {work / BUILT_FILE}")
        return {"build_mode": None, "oci": built.model_dump()}

    def submission_role(self, out: Path, *, digest: Path, base_file: tuple[Path, str], base: PinnedImage, tree: Path,
                        wheels: Path, project: str, server: str, nonce: str,
                        constraints: str | None) -> tuple[list[str], dict[str, str]]:
        # Never the BuildKit builder, which runs privileged: the install runs in an ordinary
        # container of the pinned base (image_role.submit), whose program renders the one script it
        # runs from these values, so the role's caller names no command.
        tag = f"submission-{check_sha256_hex(base.file_sha256)[:12]}-{os.getpid()}"
        argv = [self.python, "-m", ROLE_MODULE, "submit", "--socket", _path("Docker socket", self.socket),
                "--base", base.ref, "--tree", _path("the tree", tree), "--wheels", _path("the wheels", wheels),
                "--project", project, "--server", server, "--nonce", nonce, "--job", self.job_id,
                "--name", f"{job_repository(self.job_id)}:{tag}", "--out", _path("the image", out),
                "--digest", _path("the digest file", digest)]
        if constraints is not None:
            argv += ["--constraints", constraints]
        return argv, {}

    def tree_path(self, image: PinnedImage, path: str) -> Path | None:
        _tree_path(path)
        return None

    def tree_role(self, image: PinnedImage, path: str, dest: Path, *, own: Path) -> tuple[list[str], dict[str, str]]:
        if _tree_path(path) == "/":
            raise ValueError("a tree read copies one directory of an image, never its root (reader_args)")
        argv = [self.python, "-m", ROLE_MODULE, "read", "--socket", _path("Docker socket", self.socket),
                "--image", image.ref, "--path", path, "--dest", _path("the destination", dest),
                "--job", self.job_id]
        return argv, {}

    def reader_args(self, flag: str, image: PinnedImage, file: Path, *, listing: Path | None = None,
                    listing_out: Path | None = None) -> list[str]:
        # No image is a tree on the host, and none is copied whole out of the daemon: the program
        # reads the archive the image was pinned from, by the manifest digest it was pinned by, and
        # the first to read it leaves its tree for the next.
        argv = [f"--{flag}-archive", _path(f"the {flag} image's archive", file), f"--{flag}-manifest",
                check_sha256_hex(image.check_sha256)]
        if listing_out is not None:
            argv += [f"--{flag}-listing-out", _path(f"the {flag} image's listing", listing_out)]
        if listing is not None:
            argv += [f"--{flag}-listing", _path(f"the {flag} image's listing", listing)]
        return argv

    def build_role(self, recipe: Path, out: Path, *, digest: Path, parent: tuple[Path, str, Path] | None,
                   login: bool, contestant: bool) -> tuple[list[str], dict[str, str]]:
        if contestant:
            raise ValueError("the Docker builder runs BuildKit in a privileged container, so it builds only images "
                             "from the catalog's own recipes; an image whose build reads a contestant's files, such "
                             "as a submission image, needs an unprivileged builder, which this runtime does not have")
        own = out.parent
        argv = [self.python, "-m", ROLE_MODULE, "build", "--docker", self.docker,
                "--buildkit-image", self.buildkit_image, "--builder", builder_name(self.job_id, own),
                "--recipe", _path("the Dockerfile", recipe), "--contexts", _path("the contexts", recipe.parent / CONTEXTS),
                "--out", _path("the image", out), "--digest", _path("the digest file", digest),
                "--record", _path("the build record", own / BUILT_FILE)]
        if login:
            argv.append("--login")
        if parent is not None:
            stored, sha256, copy = _path("the parent image", parent[0]), check_sha256_hex(parent[1]), parent[2]
            argv += ["--parent", _path("the parent's copy", copy)]
            argv = ["/bin/sh", "-c", FETCH_EXEC, "hls-build", stored, str(copy), sha256, *argv]
        config = own / "docker-config"
        config.mkdir(mode=0o700, exist_ok=True)
        env = {"DOCKER_CONFIG": _path("the Docker configuration", config),
               "DOCKER_HOST": f"unix://{_path('Docker socket', self.socket)}"}
        return argv, env
