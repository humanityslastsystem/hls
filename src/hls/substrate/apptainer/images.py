"""The ``apptainer build`` command line of the images stage, and the build mode it reports.

A build runs as a host-side harness role step: ``apptainer build --fakeroot <out> <def>``.
It never passes ``--ignore-fakeroot-command``, which makes a package's setgid helper fail to
unpack and breaks every later apt step. The build's environment names where
apptainer keeps the layers it pulls (``APPTAINER_CACHEDIR``) and where it unpacks the image
it assembles (``APPTAINER_TMPDIR``, tens of GB for a base), both in the build role's own
node-local directory, which goes when the stage closes; the step's clean environment carries
nothing else of the host's.

Image files move through roles, never through the runner, which reads none.
The build's step writes the new image file's sha256 beside it once the build has succeeded
(``hls.store.digest.reported_sha256`` reads it). A layer or a cap builds on its parent's image file from the
parent's entry in the store, read once: its step copies the file into the role's own directory,
checks the copy's sha256 against the one the entry records (``FETCH_SHELL``), builds from the
copy, which the ``.def`` names, and removes it. So the build reads exactly the bytes that were
checked. A copy with another digest sends the step back to the store's file, only then: the
step refuses that file, exiting ``DIGEST_MISMATCH_EXIT``, when it differs too, and otherwise
fails as a copy gone wrong on the way, exiting ``COPY_FAILED_EXIT``, which a retry copies
again. Every digest is of a file read on standard input, so no file name is ever parsed out of
``sha256sum``'s output.

An unprivileged ``--fakeroot`` build runs in one of three modes, which apptainer names at INFO
on stderr (``build_mode``): in a user namespace mapping the user's ``/etc/subuid`` range, of
which apptainer says nothing; in a root-mapped namespace where only the user exists, as root,
when there is no such range, with ``%post`` run under the host's ``fakeroot`` command when one is
installed; or under the ``fakeroot`` command alone, without a namespace.
The mode can change what a build writes, so each image records it; it is not part of the
image's key.

A base pulls the CUDA devel image from Docker Hub. Anonymous pulls from a shared site
address hit the hub's rate limit, so a build may log in, with a user name and a token, both or
neither. The user name is a plain setting (``site.registry_username``) and reaches the build in
its declared environment as ``REGISTRY_USER_VAR``. The token is the login's one secret, granted
as ``REGISTRY_TOKEN_GRANT``, and reaches the build only through the role's env file. Each goes
under its own name: the planner refuses a grant of any ``APPTAINER_`` variable, and the launcher
a declared one, since apptainer reads those as its options. ``LOGIN_SHIM``, run before the build
in the same step, exports the two to apptainer as ``APPTAINER_DOCKER_USERNAME`` and
``APPTAINER_DOCKER_PASSWORD``. The shim's text names the variables, never a value, so no
command line holds the login."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from hls.package.offline import render_offline_def
from hls.recipes import render
from hls.recipes.model import ImageDecl, RegistryImage, UvPin, parent_of
from hls.store.digest import COPY_FAILED_EXIT, DIGEST_MISMATCH_EXIT, check_sha256_hex
from hls.substrate.apptainer.argv import exact_path
from hls.substrate.container import ImageRuntime, PinnedImage
from hls.substrate.sandbox import (IMAGE_FILE, make_removable, remove_extracted, renew_directory_times, sandbox_paths,
                                   tree_problem, write_manifest)

#: The variable a base build's declared environment gives it the registry user name in, from
#: ``site.registry_username``: the user name is not a secret.
REGISTRY_USER_VAR = "REGISTRY_USERNAME"
#: The variable the registry token, the login's one secret, is granted to the images stage as.
REGISTRY_TOKEN_GRANT = "REGISTRY_PASSWORD"
#: The shell that hands the login to apptainer: it stops, naming the variable, if either of
#: the two is missing or empty, exports apptainer's names, unsets the two and runs its arguments.
LOGIN_SHIM = (': "${REGISTRY_USERNAME:?}" "${REGISTRY_PASSWORD:?}"; '
              'export APPTAINER_DOCKER_USERNAME="$REGISTRY_USERNAME" '
              'APPTAINER_DOCKER_PASSWORD="$REGISTRY_PASSWORD"; '
              'unset REGISTRY_USERNAME REGISTRY_PASSWORD; exec "$@"')
#: A shell function, ``hls_fetch <file> <copy> <sha256>``, that copies the file (an image file,
#: or the package checkout's source bundle) to *copy*, its one read of the file, hashes the copy
#: and stops the shell unless that digest is the one given. Only then does it hash the file
#: again: when the file has another digest too, it stops with ``DIGEST_MISMATCH_EXIT`` and a
#: ``refused:`` line on stderr naming the file and its digest; when the file has the given one,
#: the copy went wrong on the way, and it stops with ``COPY_FAILED_EXIT`` and a ``copy failed:``
#: line. It stops with 74 when the file cannot be
#: copied or read, or the copy read; whichever way, the copy is removed first. A plain ``cp``: a
#: ``tee`` into the hash would pass a short copy, as ``/bin/sh`` has no pipefail.
FETCH_SHELL = ('hls_fetch() { cp -- "$1" "$2" || { rm -f -- "$2"; exit 74; }; '
               'hls_got=$(sha256sum < "$2") || { rm -f -- "$2"; exit 74; }; hls_got=${hls_got%% *}; '
               '[ "$hls_got" = "$3" ] || { rm -f -- "$2"; '
               'hls_src=$(sha256sum < "$1") || exit 74; hls_src=${hls_src%% *}; '
               '[ "$hls_src" = "$3" ] && { '
               'printf "copy failed: the copy of file %s differs from it\\n" "$1" >&2; '
               f'exit {COPY_FAILED_EXIT}; }}; '
               'printf "refused: file %s has sha256 %s, not %s\\n" "$1" "$hls_src" "$3" >&2; '
               f'exit {DIGEST_MISMATCH_EXIT}; }}; }}; ')
#: The extraction's step: arguments the image file, its recorded sha256, apptainer, the
#: directory to extract into and the path in the role's own directory the file is copied to. It
#: copies the file and refuses the copy unless it has that digest (``FETCH_SHELL``), then
#: extracts the copy and removes it, whatever apptainer did.
EXTRACT_SHELL = (FETCH_SHELL + 'hls_fetch "$1" "$5" "$2"; "$3" build --sandbox "$4" "$5"; hls_rc=$?; '
                 'rm -f -- "$5"; exit "$hls_rc"')

#: The build's step: arguments apptainer, the image file, the ``.def``, the file the image's
#: sha256 goes to, and, for a layer or a cap, the parent's image file in the store, the path in
#: the role's own directory it is copied to (which the ``.def`` names) and its recorded sha256,
#: the last three empty for a base. The parent's copy is removed when the build ends.
BUILD_SHELL = (FETCH_SHELL + '[ -z "$5" ] || hls_fetch "$5" "$6" "$7"; '
               '"$1" build --fakeroot "$2" "$3"; hls_rc=$?; [ -z "$6" ] || rm -f -- "$6"; '
               '[ "$hls_rc" -eq 0 ] || exit "$hls_rc"; sha256sum < "$2" > "$4"')
# What apptainer 1.5 prints at INFO when an unprivileged --fakeroot build falls back from a
# subuid-mapped namespace (cmd/internal/cli/build_linux.go), each message's start.
_ROOT_MAPPED = "User not listed in "
_NO_NAMESPACE = "Could not start root-mapped namespace"
_POST_FAKEROOT = "The %post section will be run under the fakeroot command"
_NO_FAKEROOT = "fakeroot command not found"
_MODE_MESSAGES = (_ROOT_MAPPED, _NO_NAMESPACE, _POST_FAKEROOT, _NO_FAKEROOT,
                  "Installing some packages may fail",
                  "Using --fix-perms because building from a definition file",
                  "without either root user or unprivileged user namespaces")
_INFO = re.compile(r"INFO:\s*(.*)")


class BuildMode(BaseModel):
    """How apptainer ran an unprivileged ``build --fakeroot``, as its INFO lines say."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The user namespace the build ran in: ``root-mapped`` (only the user, mapped to root),
    #: ``none`` (apptainer could not start a namespace), or ``unknown`` when apptainer named
    #: neither: a build in the user's /etc/subuid range says nothing, and neither does one whose
    #: log was quieter or could not be read.
    user_namespace: Literal["root-mapped", "none", "unknown"]
    #: Whether ``%post`` ran under the host's ``fakeroot`` command, whose LD_PRELOAD fakes the
    #: ownership changes a root-mapped namespace cannot make; None when apptainer said neither.
    post_under_fakeroot_command: bool | None
    #: Apptainer's INFO messages about the mode, in order.
    info: list[str]


def build_mode(stderr: str) -> BuildMode:
    """The mode of a ``build --fakeroot`` whose stderr is *stderr*."""
    info = [m.group(1).strip() for line in stderr.split("\n") if (m := _INFO.match(line.strip()))]
    info = [message for message in info if message.startswith(_MODE_MESSAGES)]
    namespace: Literal["root-mapped", "none", "unknown"] = "unknown"
    if any(message.startswith(_NO_NAMESPACE) for message in info):
        namespace = "none"
    elif any(message.startswith(_ROOT_MAPPED) for message in info):
        namespace = "root-mapped"
    fakeroot = True if _POST_FAKEROOT in info else False if _NO_FAKEROOT in info else None
    return BuildMode(user_namespace=namespace, post_under_fakeroot_command=fakeroot, info=info)


def _absolute(path: Path, what: str) -> str:
    if not path.is_absolute():
        raise ValueError(f"{what} {path} must be an absolute path")
    return str(path)


def _absolute_tree_path(path: str) -> str:
    if not path.startswith("/"):
        raise ValueError(f"a path in an image is absolute, not {path!r}")
    return path


def build_argv(def_file: Path, out: Path, *, digest: Path, parent: tuple[Path, str, Path] | None = None,
               apptainer: str = "apptainer", login: bool = False) -> list[str]:
    """The command that builds the ``.def`` *def_file* into the image file *out* and then
    writes the file's sha256 to *digest* (``BUILD_SHELL``). *parent* is, for a layer or a cap,
    the image file it builds on, the sha256 its entry records and the path in the role's own
    directory the file is copied to, which the ``.def`` must name: the copy is refused unless
    it has that digest. With *login*, the command runs under ``LOGIN_SHIM``, for a role whose
    declared environment holds ``REGISTRY_USER_VAR`` and whose env file holds
    ``REGISTRY_TOKEN_GRANT``."""
    parent_args = ["", "", ""] if parent is None else [
        _absolute(parent[0], "the parent image"), _absolute(parent[2], "the parent's copy"),
        check_sha256_hex(parent[1])]
    argv = ["/bin/sh", "-c", BUILD_SHELL, "hls-build", apptainer, _absolute(out, "the image"),
            _absolute(def_file, "the .def"), _absolute(digest, "the digest file"), *parent_args]
    return ["/bin/sh", "-c", LOGIN_SHIM, "registry-login", *argv] if login else argv


def build_env(*, cache_dir: Path, tmp_dir: Path) -> dict[str, str]:
    """The declared environment of a build role: apptainer's layer cache and its unpacking
    directory, both of which must be on the job's node-local disk."""
    return {"APPTAINER_CACHEDIR": _absolute(cache_dir, "the cache directory"),
            "APPTAINER_TMPDIR": _absolute(tmp_dir, "the build directory")}


#: The program a submission image runs: the runscript apptainer writes into every image.
RUNSCRIPT = "/.singularity.d/runscript"


def extract_argv(image_file: Path, dest: Path, *, sha256: str, copy: Path, apptainer: str = "apptainer") -> list[str]:
    """The command that extracts the image file *image_file* into the directory *dest*, which
    must not exist yet, without privileges (``EXTRACT_SHELL``). It reads *image_file* once, into
    *copy*, a path in the extraction role's own directory, checks that the copy's sha256 is
    *sha256*, extracts the copy and removes it, so what is extracted is exactly what was checked.
    A copy with another digest is refused: the command prints a line starting ``refused:`` on
    stderr and exits ``DIGEST_MISMATCH_EXIT`` before it extracts anything; a file it cannot copy
    fails otherwise. Raise ValueError for a path the command line cannot express exactly, or a
    *sha256* that is not bare hex."""
    paths = [exact_path(what, str(path)) for what, path in (("image file", image_file), ("sandbox", dest),
                                                            ("copy", copy))]
    return ["/bin/sh", "-c", EXTRACT_SHELL, "hls-extract", paths[0], check_sha256_hex(sha256), apptainer, paths[1],
            paths[2]]


class ApptainerImages(ImageRuntime):
    """Images as ``.sif`` files, each pinned by extracting it into a directory in the job's
    node-local scratch (``hls.substrate.sandbox``), keyed by the file's sha256, with a manifest
    of the tree written beside it. The extraction role writes into a hidden directory beside the
    tree's place, which ``pinned`` renames into place once the role succeeded."""

    image_file = IMAGE_FILE
    runscript = RUNSCRIPT
    recipe_file = "recipe.def"
    parent_file = "parent.sif"

    def __init__(self, scratch: Path, apptainer: str = "apptainer") -> None:
        #: The job's node-local scratch, which holds the extracted trees.
        self.scratch = scratch
        #: The apptainer program: a name found on the role's PATH, or an absolute path.
        self.apptainer = apptainer
        # The directory each pin role not yet finished extracts into, and the file it reads, by
        # the role's own directory.
        self._pending: dict[Path, tuple[Path, Path]] = {}

    def left(self, sha256: str, *, contestant: bool) -> PinnedImage | str | None:
        root, manifest = sandbox_paths(self.scratch, sha256)
        if not (os.path.lexists(root) or os.path.lexists(manifest)):
            return None
        problem, digest = tree_problem(root, manifest, sha256)
        if problem is not None:
            return problem
        assert digest is not None
        return PinnedImage(ref=str(root), file_sha256=sha256, check_sha256=digest, contestant=contestant)

    def remove_left(self, sha256: str) -> None:
        root, manifest = sandbox_paths(self.scratch, sha256)
        remove_extracted(root)
        manifest.unlink(missing_ok=True)

    def pin_role(self, image_file: Path, *, sha256: str, own: Path) -> tuple[list[str], dict[str, str]]:
        root, _ = sandbox_paths(self.scratch, sha256)
        tmp = own / "tmp-build"
        tmp.mkdir()
        root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        dest = root.parent / f".{own.name}-{uuid.uuid4().hex[:8]}"
        argv = extract_argv(image_file, dest, sha256=sha256, copy=own / IMAGE_FILE, apptainer=self.apptainer)
        env = build_env(cache_dir=own / "cache", tmp_dir=tmp)
        self._pending[own] = (dest, image_file)
        return argv, env

    def pinned(self, own: Path, *, sha256: str, contestant: bool) -> PinnedImage:
        dest, image_file = self._pending.pop(own)
        if not dest.is_dir() or dest.is_symlink():
            remove_extracted(dest)
            raise ValueError(f"ended without extracting {image_file} into a directory")
        root, manifest = sandbox_paths(self.scratch, sha256)
        make_removable(dest)
        renew_directory_times(dest)
        os.rename(dest, root)
        digest = write_manifest(root, sha256, manifest)
        return PinnedImage(ref=str(root), file_sha256=sha256, check_sha256=digest, contestant=contestant)

    def discard(self, own: Path) -> None:
        pending = self._pending.pop(own, None)
        if pending is not None:
            remove_extracted(pending[0])

    def remove(self, image: PinnedImage) -> None:
        remove_extracted(Path(image.ref))
        sandbox_paths(self.scratch, image.file_sha256)[1].unlink(missing_ok=True)

    @staticmethod
    def recipe_key(decl: ImageDecl, *, parent_key: str | None, locks_dir: Path, uv: UvPin, tasks_dir: Path,
                   registry: RegistryImage | None = None) -> str:
        return render.recipe_key(decl, parent_key=parent_key, locks_dir=locks_dir, uv=uv, tasks_dir=tasks_dir,
                                 registry=registry)

    def write_recipe(self, decl: ImageDecl, work: Path, *, locks_dir: Path, uv: UvPin, tasks_dir: Path,
                     registry: RegistryImage | None = None) -> str:
        parent_sif = work / self.parent_file if parent_of(decl) is not None else None
        recipe = render.render_def(decl, parent_sif=parent_sif, locks=locks_dir, uv=uv, tasks_dir=tasks_dir,
                                   registry=registry)
        (work / self.recipe_file).write_text(recipe)
        return recipe

    def build_record(self, work: Path, stderr: str) -> dict[str, object]:
        return {"build_mode": build_mode(stderr)}

    def tree_path(self, image: PinnedImage, path: str) -> Path | None:
        return Path(image.ref) / _absolute_tree_path(path).lstrip("/")

    def tree_role(self, image: PinnedImage, path: str, dest: Path, *, own: Path) -> tuple[list[str], dict[str, str]]:
        raise ValueError("an apptainer image is an extracted tree on the host, read in place (tree_path)")

    def reader_args(self, flag: str, image: PinnedImage, file: Path, *, listing: Path | None = None,
                    listing_out: Path | None = None) -> list[str]:
        # The pinned image is its extracted tree, which the program walks in place: no listing.
        return [f"--{flag}", image.ref]

    def submission_role(self, out: Path, *, digest: Path, base_file: tuple[Path, str], base: PinnedImage, tree: Path,
                        wheels: Path, project: str, server: str, nonce: str,
                        constraints: str | None) -> tuple[list[str], dict[str, str]]:
        # An unprivileged --fakeroot build, as every image's: the install runs as root only in the
        # build's user namespace.
        work = out.parent
        def_file = work / "submission.def"
        # The build copies the base's file here, checks the copy and builds from it.
        base_copy = work / IMAGE_FILE
        def_file.write_text(render_offline_def(base_sif=base_copy, tree=tree, wheels=wheels, project=project,
                                               server=server, nonce=nonce, constraints=constraints))
        cache_dir, tmp_dir = work / "cache", work / "build-tmp"
        cache_dir.mkdir()
        tmp_dir.mkdir()
        argv = build_argv(def_file, out, digest=digest, parent=(base_file[0], base_file[1], base_copy),
                          apptainer=self.apptainer)
        return argv, build_env(cache_dir=cache_dir, tmp_dir=tmp_dir)

    def build_role(self, recipe: Path, out: Path, *, digest: Path, parent: tuple[Path, str, Path] | None,
                   login: bool, contestant: bool) -> tuple[list[str], dict[str, str]]:
        # An unprivileged --fakeroot build may build a contestant's files as well as the catalog's.
        tmp = out.parent / "build-tmp"
        tmp.mkdir()
        env = build_env(cache_dir=out.parent / "cache", tmp_dir=tmp)
        return build_argv(recipe, out, digest=digest, parent=parent, apptainer=self.apptainer, login=login), env
