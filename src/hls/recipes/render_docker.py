"""An image declaration as a Dockerfile, the build contexts its build needs, and its cache key.

The Dockerfile runs the build script ``render.post_script`` derives, the text the ``.def`` runs
in ``%post``, as one ``RUN`` heredoc whose delimiter is quoted, so the frontend expands nothing in
it. The frontend that parses the file is pinned by digest (``DOCKERFILE_SYNTAX``).

A base starts ``FROM`` its CUDA image by tag and digest, which Docker accepts together, and
keeps the reference in the label ``hls.cuda_ref`` as the ``.def`` does. A registry image starts
``FROM`` its reference by digest and runs nothing: it is the vendor's image as pulled, but for
its entrypoint and command, which it clears as a base does, so a role's command runs as given,
as ``apptainer exec`` runs it. A layer or a cap starts
``FROM`` the named build context ``PARENT_CONTEXT``: the parent's image archive unpacked as an
OCI layout once its sha256 was checked, named by its manifest digest (``parent_context``). A
build that does not pass that context reads the name as an image to pull, so the name is one
under the reserved ``.invalid`` domain, which no registry answers. The lock files, and the files
a task's own layer copies from its directory, come from the named build context
``LOCKS_CONTEXT``, a directory that holds the files ``files_for`` names and nothing else
(``build_inputs``), each copied to the path the ``.def`` copies it to.

The script runs in the environment of the image it builds on, as ``%post`` does: apptainer runs
``%post`` through its action script, which sources the image's ``/.singularity.d/env`` files, so a
base's script sees its CUDA image's variables and a child's also sees its base's environment. A
Dockerfile ``RUN`` sees the image configuration's variables, which are the same ones. So nothing
before the ``RUN`` sets or clears a variable, and a base writes its environment
(``render.environment``) after the ``RUN``, as apptainer writes ``%environment`` after ``%post``.
A base ends with ``ENTRYPOINT []`` and ``CMD []``, so no image keeps the CUDA image's entrypoint.

A layer that declares its whole environment (``LayerImage.env``) sets each of its variables
after the ``RUN``, in order, as its ``.def``'s ``%environment`` exports them. A Dockerfile cannot
remove a variable the image below set: one the layer leaves out keeps the value the image below
gave it, where the ``.def``'s replaced ``%environment`` drops it, and a value that names a
variable (``$PATH``) expands against the image below's value of it, where the ``.def``'s expands
against the CUDA image's.

A Docker image's cache key is its name and 16 hex digits of a digest of its Dockerfile, the
bytes of every file it copies in, and its parent's key (``dockerfile_key``), so it differs from
the key of the same declaration's ``.def``. The Dockerfile names no host path, so no stand-ins
are needed."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from hls.recipes.model import BaseImage, ImageDecl, LayerImage, RegistryImage, UvPin, parent_of
from hls.recipes.render import TASKS_DIR, check_parent_key, environment, files_for, post_script, text_key

#: The Dockerfile frontend every rendered Dockerfile names: release 1.29 by its index digest.
#: Heredocs and named build contexts need 1.4 or later.
DOCKERFILE_SYNTAX = ("docker/dockerfile:1.29"
                     "@sha256:572cebbc2b15e6f84c077742952674889cb63cd5e9b80abdffcbe587b6f19183")
#: The named build context a layer or a cap reads its parent image from. Without the context the
#: build looks the name up as a registry host, which a name under ``.invalid`` never resolves to.
PARENT_CONTEXT = "hls.invalid/parent"
#: The named build context the lock files are copied from.
LOCKS_CONTEXT = "locks"
#: The delimiter of the heredoc the build script runs in. No line of a script may be it.
SCRIPT_END = "HLS_POST"
#: Where a Docker image keeps the command it runs by, in place of ``/.singularity.d/runscript``.
RUNSCRIPT = "/.hls/runscript"
# A manifest digest, as an OCI layout's index names it.
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
# A value of BASE_ENVIRONMENT, or of a layer's environment, written unquoted so that Docker
# expands the variable it names ($PATH), as the .def's shell does.
_EXPANDED = re.compile(r"[A-Za-z0-9_/:.$+,=@%-]*")


def syntax_line() -> str:
    """The first line of every rendered Dockerfile, the parser directive naming its frontend."""
    return f"# syntax={DOCKERFILE_SYNTAX}"


def quoted(value: str) -> str:
    """*value* as one double-quoted Dockerfile word that the frontend reads back unchanged: a
    backslash escapes ``"``, ``$`` and itself there. Raise ValueError for a value that is not
    printable, since a line break would end the instruction."""
    if not value.isprintable():
        raise ValueError(f"{value!r} holds a character that is not printable, which a Dockerfile cannot carry")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'


def heredoc(script: str, end: str = SCRIPT_END) -> list[str]:
    """The lines of a heredoc that holds *script* exactly, ending at *end*, after the line
    that opens it. Raise ValueError when a line of *script* is *end*."""
    lines = script.split("\n")
    if end in lines:
        raise ValueError(f"a line of the script is {end}, the end of the heredoc that holds it")
    return [*lines, end]


def build_script_lines(script: str, *, network: bool = True) -> list[str]:
    """The ``RUN`` instruction that runs *script* with ``/bin/sh``, in the environment of the
    image it builds on; with *network* False, in no network."""
    flag = "" if network else "--network=none "
    return [f"RUN {flag}<<'{SCRIPT_END}'", *heredoc(script)]


def render_dockerfile(decl: ImageDecl, *, locks: Path, uv: UvPin, tasks_dir: Path = TASKS_DIR,
                      registry: RegistryImage | None = None) -> str:
    """The Dockerfile that builds *decl*, copying its lock files, from *locks* or, for a task's
    own layer, its task's directory under *tasks_dir*, and the files it copies from there, from
    the ``LOCKS_CONTEXT`` build context (``build_inputs``) and, for a layer or cap, building on
    the ``PARENT_CONTEXT`` one; *registry* is the registry image its chain starts at
    (``model.registry_of``), None for a chain that starts at a base. Raise ValueError for an image
    that may not build on a registry image or a lock holding what it may not, and
    FileNotFoundError for a missing file."""
    if isinstance(decl, RegistryImage):
        return "\n".join([syntax_line(), f"FROM {decl.ref}", "ENTRYPOINT []", "CMD []"]) + "\n"
    script = post_script(decl, locks=locks, uv=uv, tasks_dir=tasks_dir, registry=registry)
    # Each file is in the context by its own name, which build_inputs checks is one of its kind.
    build_inputs(decl, locks_dir=locks, tasks_dir=tasks_dir)
    out = [syntax_line()]
    if isinstance(decl, BaseImage):
        out += [f"FROM {decl.cuda_ref}", f"LABEL hls.cuda_ref={quoted(decl.cuda_ref)}"]
    else:
        out += [f"FROM {PARENT_CONTEXT}"]
    out += [f"COPY --from={LOCKS_CONTEXT} {host.name} {container}"
            for container, host in files_for(decl, locks, tasks_dir).items()]
    out += build_script_lines(script)
    for variable in environment(decl):
        out.append(f"ENV {variable.name}={quoted(variable.value)}" if variable.literal
                   else _expanded_env(decl, variable.name, variable.value))
    if isinstance(decl, LayerImage) and decl.env is not None:
        out += [_expanded_env(decl, name, value) for name, value in decl.env]
    if isinstance(decl, BaseImage):
        out += ["ENTRYPOINT []", "CMD []"]
    return "\n".join(out) + "\n"


def _expanded_env(decl: ImageDecl, name: str, value: str) -> str:
    """The ``ENV`` line that sets *name* to *value*, a shell word that may name a variable, which
    Docker expands as the ``.def``'s shell does. Raise ValueError for a value a Dockerfile word
    cannot carry unquoted."""
    if not _EXPANDED.fullmatch(value):
        raise ValueError(f"variable {name} of image {decl.name} expands {value!r}, which a Dockerfile word cannot "
                         "carry unquoted")
    return f"ENV {name}={value}"


@dataclass(frozen=True)
class BuildInputs:
    """What the Docker build of one image needs besides its Dockerfile."""

    #: The files the ``LOCKS_CONTEXT`` directory holds, and nothing else: name in the context to
    #: the file on the host: a lock file, or a file of a task's directory.
    locks: dict[str, Path]
    #: Whether the build reads its parent from the ``PARENT_CONTEXT`` context (``parent_context``).
    parent: bool


def build_inputs(decl: ImageDecl, *, locks_dir: Path, tasks_dir: Path = TASKS_DIR) -> BuildInputs:
    """The build contexts *decl*'s Dockerfile reads, with the lock files under *locks_dir* or,
    for a task's own layer, its task's directory under *tasks_dir*. Raise ValueError for two
    files of one name, which the build context cannot hold apart."""
    files = files_for(decl, locks_dir, tasks_dir)
    locks = {host.name: host for host in files.values()}
    if len(locks) != len(files):
        raise ValueError(f"image {decl.name} copies in two files of one name, which its build context cannot hold apart")
    return BuildInputs(locks=dict(sorted(locks.items())), parent=parent_of(decl) is not None)


def parent_context(layout: Path, manifest_digest: str) -> str:
    """What ``--build-context`` gives ``PARENT_CONTEXT``: the OCI layout at *layout*, which the
    parent's checked archive was unpacked to, at the image whose manifest digest is
    *manifest_digest*. Raise ValueError for a relative path or a digest that is not
    ``sha256:<64 hex>``."""
    if not layout.is_absolute():
        raise ValueError(f"the parent's OCI layout {layout} must be an absolute path")
    if not _DIGEST.fullmatch(manifest_digest):
        raise ValueError(f"{manifest_digest!r} is not a manifest digest sha256:<64 hex>")
    return f"oci-layout://{layout}@{manifest_digest}"


def dockerfile_key(decl: ImageDecl, *, parent_key: str | None, locks_dir: Path, uv: UvPin,
                   tasks_dir: Path = TASKS_DIR, registry: RegistryImage | None = None) -> str:
    """The cache key of *decl* built from its Dockerfile (with *registry* as ``render_dockerfile``
    takes it): ``<name>-<16 hex>`` of the Dockerfile, the bytes of every file it copies in and
    *parent_key*, its parent's Docker key, which is not the key of its ``.def``
    (``render.recipe_key``). Raise ValueError for a *parent_key* that does not fit the kind and
    for a lock holding what it may not, and FileNotFoundError for a missing lock file."""
    check_parent_key(decl, parent_key)
    text = render_dockerfile(decl, locks=locks_dir, uv=uv, tasks_dir=tasks_dir, registry=registry)
    return text_key(decl.name, text, files_for(decl, locks_dir, tasks_dir), parent_key)
