"""The offline build of a submission image: the submission base, the contestant's tree at
``/workspace``, and its wheels installed with no index into the base's venv.

Everything the submission supplies goes to two places: its tree to ``/workspace``, and its
wheels to the venv, ``/opt/venv``, where the venv's own pip installs them, each entry where the
venv's install scheme puts it: modules in site-packages, ``.data/scripts`` in ``/opt/venv/bin``,
``.data/headers`` under ``/opt/venv/include``, ``.data/data`` under ``/opt/venv`` itself. pip
refuses an entry that would land outside its scheme and keeps no cache (``--no-cache-dir``),
and the staged copies are removed. The bases' pip also refuses a console script whose name
would land outside ``/opt/venv/bin``, where an older pip writes it as named. Whatever pip an
image carries, ``hls.package.integrity`` compares the extracted images afterwards and refuses
any change to the base's system paths.
pip installs wheels only (``--only-binary :all:``): a source distribution among the wheels
would run its build in ``%post``, outside the seal.

The install applies the submission base's constraints file
(``catalog.TaskImages.submission_constraints``), the file the base's environment names for every
install, so it resolves as the dry run and the wheel step resolved in that environment. A
submission base built from the recipes holds torch with ``hls.recipes.model.CONSTRAINTS``; a
published image holds no constraints file and is installed into with none. ``%post`` checks
that the named file is in the base before it copies anything, so a base without it ends the
build as ours, not as an install the submission failed, with a line on the build's stderr that
names the file.

The install is by the project's name from the wheels directory, never ``.``: building from
source in ``%post`` would run the contestant's build hooks outside the sealed containers. A
pyproject without a valid ``[project].name`` is therefore refused before the build
(``project_name`` returns None). The name and every path are checked before they enter the
``.def``, so nothing the contestant chose can become shell text.

Once pip has exited, ``%post`` prints the install's status,
``hls-offline-install-rc-<nonce>=<status>``, so a failed build's log says whether the install
itself failed, which is the contestant's (a declared dependency that has no installable wheel),
or something before or after it, which is ours: a full disk, a failed copy, a squashfs step
killed for memory (``install_failed``), which is never booked as the contestant's. The status
goes to the build's log, the one
channel out of a build that fails, and only its own lines count there: installing a wheel runs
no project code, but pip echoes text the submission chose, such as an invalid ``Requires-Dist``
value of a wheel, verbatim. So every status line carries a nonce the stage draws each time it
renders the ``.def`` (``new_nonce``), after every step that ran contestant code has ended, and
the reader takes only whole lines that carry it, split at "\\n" alone: ``str.splitlines`` also
splits at U+2028 and other characters such a value can hold. The status line comes after
everything pip printed, and only the harness's own steps print after it, so it is in the tail
of the log the stage reads however much pip printed.

After a successful install, and after ``/wheels`` is removed, ``%post`` prints the bytes the
install added to ``/opt/venv`` (``hls-offline-install-bytes-<nonce>=<n>``, from ``du`` before
and after; ``installed_bytes``). A wheel unpacks to more than it ships, without bound, so the
stage holds the install to ``INSTALL_MAX_GIB`` before the image is extracted anywhere: past it the
submission is invalid, as past a tree limit, and the stage's disk holds a submission at it.
Past it ``%post`` also ends the build (``OVER_LIMIT_EXIT``) before apptainer squashes the tree
into an image file: an install a little past the limit would otherwise go on to fill the node's
disk in the squash or the extraction, which would read as the harness's failure."""

from __future__ import annotations

import re
import secrets
import tomllib
from pathlib import Path, PurePosixPath

from hls.recipes.render import VENV, fail_closed
from hls.spec.submission import PERMISSION_ERRNOS

#: The name of the line ``%post`` prints with the install's status once pip has exited:
#: ``<name>-<nonce>=<status>``.
INSTALL_RC = "hls-offline-install-rc"
#: The name of the line ``%post`` prints with the bytes the install added to the venv:
#: ``<name>-<nonce>=<bytes>``.
INSTALL_BYTES = "hls-offline-install-bytes"
# A nonce as ``new_nonce`` draws it.
_NONCE = re.compile(r"[0-9a-f]{32}")
#: The most the install may add to the venv: 24 GiB, room for a whole CUDA machine-learning
#: Python environment with margin.
INSTALL_MAX_GIB = 24
#: The status ``%post`` ends with once the install added more than ``INSTALL_MAX_GIB``.
OVER_LIMIT_EXIT = 3
#: Where the tree is staged inside the build before it moves to /workspace.
STAGE = "/srv-stage"
# A distribution name as PEP 508 allows it.
_PROJECT = re.compile(r"[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?")
# A server file relative to the workspace: plain path segments, no "..".
_SERVER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*(/[A-Za-z0-9_][A-Za-z0-9._-]*)*")


def project_name(tree: Path) -> str | None:
    """The ``[project].name`` of *tree*'s pyproject.toml; None when there is no file, its
    permissions keep this process out (``PERMISSION_ERRNOS``), it does not parse (not UTF-8,
    not TOML, or more than the parser can hold), or the name is missing or not a valid
    distribution name. Any other error reading the file is the node's, and raised."""
    try:
        data = tomllib.loads((tree / "pyproject.toml").read_text(encoding="utf-8"))
    except OSError as e:
        if isinstance(e, FileNotFoundError) or e.errno in PERMISSION_ERRNOS:
            return None
        raise
    except (ValueError, RecursionError, MemoryError):
        return None
    project = data.get("project")
    name = project.get("name") if isinstance(project, dict) else None
    return name if isinstance(name, str) and _PROJECT.fullmatch(name) else None


def _def_path(path: Path, what: str) -> str:
    text = str(path)
    if not path.is_absolute():
        raise ValueError(f"{what} {text!r} must be an absolute path")
    if any(c.isspace() for c in text) or not text.isprintable():
        raise ValueError(f"{what} {text!r} holds whitespace or an unprintable character, which a .def cannot carry")
    return text


def new_nonce() -> str:
    """A fresh nonce for one rendering of the offline ``.def``: 32 hex digits from the system's
    random source."""
    return secrets.token_hex(16)


def _status_prefix(name: str, nonce: str) -> str:
    """What a status line *name* of the build whose ``.def`` was rendered with *nonce* starts
    with, before its value."""
    return f"{name}-{nonce}="


def render_offline_def(*, base_sif: Path, tree: Path, wheels: Path, project: str, server: str, nonce: str,
                       constraints: str | None) -> str:
    """The ``.def`` that builds the submission image from *base_sif*: *tree* at /workspace and
    the project *project* installed from the wheels in *wheels* under the base's constraints file
    *constraints* (None for a base that holds none), served by ``python /workspace/<server>``,
    its status lines carrying *nonce* (``new_nonce``). Raise ValueError for a name, server, path
    or nonce a ``.def`` must not carry."""
    post, run = offline_install(project=project, server=server, nonce=nonce, constraints=constraints)
    base, src, whl = _def_path(base_sif, "base image"), _def_path(tree, "tree"), _def_path(wheels, "wheels")
    lines = ["Bootstrap: localimage", f"From: {base}", "", "%files", f"    {src} {STAGE}", f"    {whl} /wheels", "",
             "%post", *(f"    {line}" for line in post), "", "%runscript", f"    {run}", ""]
    return "\n".join(lines)


def offline_install(*, project: str, server: str, nonce: str, constraints: str | None) -> tuple[list[str], str]:
    """The install's script, one line each, and the runscript's command, the one text every
    substrate's build of a submission image runs: the script moves the tree staged at ``STAGE`` to
    /workspace and installs *project* from the wheels staged at /wheels, its status lines carrying
    *nonce*; the command serves ``/workspace/<server>``. Raise ValueError for a name, server,
    constraints path or nonce a build must not carry."""
    _check_install(project, server, nonce)
    return _offline_post(project, nonce, constraints), f'exec {VENV}/bin/python /workspace/{server} "$@"'


def _check_install(project: str, server: str, nonce: str) -> None:
    """Raise ValueError for a project name, server file or nonce no build may carry."""
    if not _PROJECT.fullmatch(project):
        raise ValueError(f"{project!r} is not a project name the install may name")
    if not _SERVER.fullmatch(server) or ".." in PurePosixPath(server).parts:
        raise ValueError(f"server file {server!r} must be a plain path relative to the workspace")
    if not _NONCE.fullmatch(nonce):
        raise ValueError("the nonce must be 32 lowercase hex digits, as new_nonce draws it")


def _offline_post(project: str, nonce: str, constraints: str | None) -> list[str]:
    """The lines of the script that moves the staged tree to /workspace and installs *project*
    from /wheels, both wrappers' one text."""
    held = None if constraints is None else _def_path(Path(constraints), "constraints file")
    prefix = "" if held is None else f"PIP_CONSTRAINT={held} "
    install = (f"{prefix}{VENV}/bin/python -m pip install --no-index --no-cache-dir "
               f"--only-binary :all: --find-links /wheels {project} || hls_rc=$?")
    status, added = _status_prefix(INSTALL_RC, nonce), _status_prefix(INSTALL_BYTES, nonce)
    post = ["set -eu", "set +x"]
    if held is not None:
        post.append(f'[ -f {held} ] || {{ echo "the submission base holds no constraints file {held}" >&2; exit 1; }}')
    for step in ("mkdir -p /workspace", f"cp -a {STAGE}/. /workspace/", f"rm -rf {STAGE}",
                 f'hls_venv="$(du -sb {VENV})"'):
        post += fail_closed(step)
    post += ["hls_rc=0", install, f'echo "{status}$hls_rc"', '[ "$hls_rc" -eq 0 ] || exit "$hls_rc"']
    for step in ("rm -rf /wheels", f'hls_added="$(du -sb {VENV})"',
                 f'echo "{added}$(( ${{hls_added%%[[:space:]]*}} - ${{hls_venv%%[[:space:]]*}} ))"'):
        post += fail_closed(step)
    # Past the limit the build ends here, before the tree is squashed into an image file.
    post.append(f'[ "$(( ${{hls_added%%[[:space:]]*}} - ${{hls_venv%%[[:space:]]*}} ))" -le '
                f'{INSTALL_MAX_GIB * 2**30} ] || exit {OVER_LIMIT_EXIT}')
    return post


def _status_value(log: str, name: str, nonce: str, number: str) -> int | None:
    """The value of the build's one status line *name* in its output *log*, from a ``.def``
    rendered with *nonce*, when it is a whole line, split at "\\n" alone, and its value fully
    matches the pattern *number*; None when there is no such line, or more than one."""
    prefix = _status_prefix(name, nonce)
    values = [line[len(prefix):] for line in log.split("\n") if line.startswith(prefix)]
    if len(values) != 1 or not re.fullmatch(number, values[0]):
        return None
    return int(values[0])


def install_failed(log: str, nonce: str) -> bool | None:
    """Whether the build whose output is *log*, from a ``.def`` rendered with *nonce*, ran the
    install and the install failed, by the install's status line: True when it failed, False
    when it succeeded, None when the log shows no status (the build failed before the install,
    or was killed during it)."""
    status = _status_value(log, INSTALL_RC, nonce, r"[0-9]+")
    return None if status is None else status != 0


def installed_bytes(log: str, nonce: str) -> int | None:
    """The bytes the install added to the venv, by the size line of the build whose output is
    *log*, from a ``.def`` rendered with *nonce*, when the install's status line says 0; None
    when there is no such size, which the stage reads also for a build that stopped past the
    limit. It can be negative: an install may replace a base package with a smaller one."""
    if install_failed(log, nonce) is not False:
        return None
    return _status_value(log, INSTALL_BYTES, nonce, r"-?[0-9]+")
