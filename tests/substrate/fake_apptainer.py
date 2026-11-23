"""A stand-in for ``apptainer exec`` and ``apptainer build --sandbox`` on machines without
apptainer.

``build --sandbox <dir> <image file>`` extracts as ``hls.substrate.apptainer.images.extract_argv`` asks: it
refuses a *dir* that exists, fails like apptainer (a ``FATAL:`` line, exit 255) for an image
file that is not a regular file, and otherwise makes *dir* with a few of a root's directories,
one of them without the owner's write bit as images hold them (``usr/share/ro``), the image
file's bytes in ``.fake-image`` and its path in ``.fake-source``, so the tree depends on what
was extracted.

``exec`` accepts exactly the command lines ``ApptainerRuntime.argv`` builds and refuses any that
lacks an isolation flag, asks for a writable image, or names an image that is not a directory
(exit 2): a container runs from an extracted image, read-only. It then behaves like the
container would: it runs what
follows the image (the ``/bin/sh -c`` wrapper and the command) with the env file's variables,
the ``APPTAINERENV_*`` and ``SINGULARITYENV_*`` variables of its own environment without the
prefix, the TERM and ``*_proxy`` variables (both cases) that apptainer's ``--cleanenv`` passes
through, PATH, and HOME under the workdir, and nothing else from its own environment. Like the
real CLI it adds the binds that ``APPTAINER_BIND``, ``APPTAINER_BINDPATH`` and their
``SINGULARITY_`` spellings name to those of the command line, and reads the env file to its end
first, so ``/dev/fd/3`` may be a pipe. It cannot isolate anything; it only lets tests drive the
code around containers.

Knobs, read from the fake's own environment:
- ``FAKE_APPTAINER_FAIL=1``: print a ``FATAL:`` line and exit 255 without running anything,
  the way apptainer fails before a container starts.
- ``FAKE_EXTRACT_FAIL=1``: a ``build --sandbox`` prints a ``FATAL:`` line and exits 255
  without writing anything.
- ``FAKE_EXTRACT_LOG=<file>``: each ``build --sandbox`` appends one line, the image file's path.
- ``FAKE_TREE_GIB=<n>``: each ``build --sandbox`` also writes ``usr/share/tree.bin``, a sparse
  file of *n* GiB, so the tree has an extracted image's apparent size and none of its disk.
- ``FAKE_TREE_LOCKED=1``: each ``build --sandbox`` also writes ``opt/sealed`` (mode 000) and
  ``opt/blind`` (mode 300, no read), each holding a file, directories whose owner can neither
  list nor enter them, as a contestant's image may hold them.
- ``FAKE_TREE_DATED=<seconds since the epoch>``: each ``build --sandbox`` gives every entry of
  the tree that access and modification time, as an image's tree carries the times it was built
  with.
- ``FAKE_APPTAINER_RECORD=<file>``: write what it was asked to do as JSON, variable names
  only, with whether the fake pasta of ``tests/substrate/fake_pasta.py`` started it
  (``in_pasta``) and the names of BASH_ENV and the apptainer variables in its own environment
  (``runtime_env_names``), which the step wrapper should have removed.

``install(dir)`` writes an executable ``apptainer`` into *dir* that runs this file with the
test's interpreter."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

_FLAGS = {"--containall", "--cleanenv", "--pid", "--nv"}
_OPTIONS = {"--no-mount", "--workdir", "--cwd", "--bind", "--env-file", "--drop-caps"}
_REQUIRED_FLAGS = ("--containall", "--cleanenv", "--pid")
_NO_MOUNT = "hostfs,bind-paths,cwd"
# Variables whose comma-separated binds apptainer adds to the command line's (cmd/internal/cli/action_flags.go).
_BIND_VARS = ("APPTAINER_BIND", "APPTAINER_BINDPATH", "SINGULARITY_BIND", "SINGULARITY_BINDPATH")
# Prefixes of the variables apptainer copies into a container without the prefix.
_COPIED_PREFIXES = ("APPTAINERENV_", "SINGULARITYENV_")
# Prefixes of every variable apptainer reads as its own.
_RUNTIME_PREFIXES = ("APPTAINER_", "SINGULARITY_", *_COPIED_PREFIXES)
# The host variables apptainer's --cleanenv still passes into the container (internal/pkg/util/env).
_PASSED_BY_CLEANENV = {"TERM", *(f"{p}_proxy" for p in ("http", "https", "no", "all", "ftp")),
                       *(f"{p}_PROXY" for p in ("HTTP", "HTTPS", "NO", "ALL", "FTP"))}


def install(directory: Path) -> Path:
    """Write an executable ``apptainer`` into *directory* and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "apptainer"
    path.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(__file__)} \"$@\"\n")
    path.chmod(0o755)
    return path


def read_record(path: Path) -> dict:
    """What a run with ``FAKE_APPTAINER_RECORD=<path>`` recorded."""
    return json.loads(path.read_text())


def _usage(message: str) -> int:
    print(f"fake apptainer: {message}", file=sys.stderr)
    return 2


def _read_env_file(path: str) -> dict[str, str]:
    env = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        words = shlex.split(line)
        if len(words) != 1 or "=" not in words[0]:
            raise ValueError(f"not NAME=value: {line!r}")
        name, _, value = words[0].partition("=")
        env[name] = value
    return env


def extract(argv: list[str]) -> int:
    """``build --sandbox <dir> <image file>``, as the module docstring says."""
    if len(argv) != 4 or argv[:2] != ["build", "--sandbox"]:
        return _usage(f"unexpected build arguments {argv}")
    dest, image = Path(argv[2]), Path(argv[3])
    if os.path.lexists(dest):
        return _usage(f"{dest} exists; apptainer would not overwrite it without --force")
    if os.environ.get("FAKE_EXTRACT_FAIL") == "1" or not image.is_file() or image.is_symlink():
        print(f"FATAL:   could not open image {image}", file=sys.stderr)
        return 255
    log = os.environ.get("FAKE_EXTRACT_LOG")
    if log:
        with open(log, "a") as f:
            f.write(f"{image}\n")
    for d in ("bin", "etc", "usr/bin", "usr/share/ro", "tmp"):
        (dest / d).mkdir(parents=True)
    (dest / ".fake-image").write_bytes(image.read_bytes())
    (dest / ".fake-source").write_text(str(image))
    (dest / "usr" / "share" / "ro" / "f").write_text("x")
    tree_gib = os.environ.get("FAKE_TREE_GIB")
    if tree_gib:
        with open(dest / "usr" / "share" / "tree.bin", "wb") as f:
            f.truncate(int(float(tree_gib) * 2**30))
    locked = {"opt/sealed": 0o000, "opt/blind": 0o300} if os.environ.get("FAKE_TREE_LOCKED") == "1" else {}
    for d in locked:
        (dest / d).mkdir(parents=True)
        (dest / d / "f").write_text("x")
    dated = os.environ.get("FAKE_TREE_DATED")
    if dated:
        # Children first: dating an entry never moves its directory's times, adding one does.
        for directory, dirs, files in os.walk(dest, topdown=False):
            for name in (*files, *dirs):
                os.utime(Path(directory) / name, (float(dated), float(dated)), follow_symlinks=False)
        os.utime(dest, (float(dated), float(dated)))
    (dest / "usr" / "share" / "ro").chmod(0o555)
    for d, mode in locked.items():
        (dest / d).chmod(mode)
    return 0


def main(argv: list[str]) -> int:
    if argv[:2] == ["build", "--sandbox"]:
        return extract(argv)
    if argv[:1] != ["exec"]:
        return _usage(f"only exec and build --sandbox are supported, got {argv[:1]}")
    flags: set[str] = set()
    options: dict[str, list[str]] = {}
    i = 1
    while i < len(argv) and argv[i].startswith("--"):
        arg = argv[i]
        if arg in _FLAGS:
            flags.add(arg)
            i += 1
        elif arg in _OPTIONS and i + 1 < len(argv):
            options.setdefault(arg, []).append(argv[i + 1])
            i += 2
        else:
            return _usage(f"unknown or incomplete option {arg}")
    for flag in _REQUIRED_FLAGS:
        if flag not in flags:
            return _usage(f"missing {flag}")
    if options.get("--no-mount") != [_NO_MOUNT]:
        return _usage(f"missing --no-mount {_NO_MOUNT}")
    if len(options.get("--workdir", [])) != 1:
        return _usage("missing --workdir")
    if i + 1 >= len(argv):
        return _usage("no image and command")
    image, inside = argv[i], argv[i + 1:]
    if not Path(image).is_dir():
        return _usage(f"image {image} is not an extracted directory")
    # The real CLI reads (and so drains) the env file before anything can fail.
    file_env: dict[str, str] = {}
    for path in options.get("--env-file", []):
        file_env.update(_read_env_file(path))
    if os.environ.get("FAKE_APPTAINER_FAIL") == "1":
        print("FATAL:   fake apptainer failure requested", file=sys.stderr)
        return 255
    workdir = Path(options["--workdir"][0])
    home = workdir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home)}
    env.update({k: v for k, v in os.environ.items() if k in _PASSED_BY_CLEANENV})
    for prefix in _COPIED_PREFIXES:
        env.update({k[len(prefix):]: v for k, v in os.environ.items() if k.startswith(prefix)})
    env.update(file_env)
    binds = options.get("--bind", []) + [b for var in _BIND_VARS for b in os.environ.get(var, "").split(",") if b]
    record = os.environ.get("FAKE_APPTAINER_RECORD")
    if record:
        Path(record).write_text(json.dumps({
            "image": image, "nv": "--nv" in flags, "cwd": (options.get("--cwd") or [None])[0],
            "workdir": str(workdir), "binds": binds, "no_mount": options["--no-mount"][0],
            "drop_caps": (options.get("--drop-caps") or [None])[0],
            "env_names": sorted(k for k in env if k not in ("PATH", "HOME")), "command": inside[4:],
            "in_pasta": os.environ.get("FAKE_PASTA_NAMESPACE") == "1",
            "runtime_env_names": sorted(k for k in os.environ if k == "BASH_ENV" or k.startswith(_RUNTIME_PREFIXES)),
        }))
    cwd = (options.get("--cwd") or [None])[0]
    if cwd is not None and Path(cwd).is_dir():
        os.chdir(cwd)
    os.execve(inside[0], inside, env)
    return 0  # not reached


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
