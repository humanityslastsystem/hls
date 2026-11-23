"""A stand-in for ``apptainer build`` and ``apptainer exec`` for the images stage's tests.

``build --fakeroot <out> <def>`` writes ``fake-sif``, the name of the directory *out* is in (the
build role's) and the ``.def``'s text to *out*, so the file's digest depends on the recipe, and
prints the INFO lines apptainer 1.5 prints for the
build mode ``FAKE_BUILD_MODE`` names. It refuses (exit 2) a command line without
``--fakeroot`` or with ``--ignore-fakeroot-command``, and fails like apptainer (a ``FATAL:``
line, exit 255) when a ``localimage`` parent does not exist.

``build --sandbox <dir> <image file>`` is ``tests.substrate.fake_apptainer``'s extraction.

``exec`` refuses an image that is not an extracted directory, then hands the command line to
``tests.substrate.fake_apptainer``, which checks its isolation flags and runs the command on
this machine. A bound file the command names by its container path runs from its host path
instead; the bound inventory script is replaced by one that prints ``$FAKE_INVENTORY``.

Knobs, read from the fake's environment:
- ``FAKE_BUILD_FAIL=1``: a build prints ``FATAL:`` and exits 255 without writing anything.
- ``FAKE_BUILD_MODE``: ``root-mapped-fakeroot`` (the default: no /etc/subuid range, and %post
  under the host's fakeroot command, as on a cluster login node), ``root-mapped`` (no fakeroot
  command), ``fakeroot-only`` (no user namespace at all) or ``subuid``, which prints nothing.
- ``FAKE_BUILD_LOG=<file>``: each build appends one JSON line: the ``.def``'s ``From:``, the
  output path, the names (never values) of its ``APPTAINER_*`` variables, the sha256 of each
  registry login variable apptainer reads (``APPTAINER_DOCKER_*``), and the names of any
  ``REGISTRY_*`` variable left in its environment.
- ``FAKE_INVENTORY=<file or directory>``: what the inventory script prints. For a directory,
  the file named after the build role that wrote the extracted image (``build-<image>.txt``, so
  each built image can report its own), else ``default.txt``.
- ``FAKE_EXTRACT_FAIL``, ``FAKE_EXTRACT_LOG`` and ``FAKE_TREE_GIB``, as
  ``tests.substrate.fake_apptainer`` reads them."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

from hls.recipes.inventory import INVENTORY_TARGET
from tests.substrate import fake_apptainer

# exec options that take no value; every other option takes one.
_FLAGS = {"--containall", "--cleanenv", "--pid", "--nv"}
#: What apptainer 1.5.3 prints at INFO for each build mode (cmd/internal/cli/build_linux.go).
MODE_LINES = {
    "subuid": [],
    "root-mapped-fakeroot": ["User not listed in /etc/subuid, trying root-mapped namespace",
                             "The %post section will be run under the fakeroot command"],
    "root-mapped": ["User not listed in /etc/subuid, trying root-mapped namespace",
                    "fakeroot command not found", "Installing some packages may fail"],
    "fakeroot-only": ["User not listed in /etc/subuid, trying root-mapped namespace",
                      "Could not start root-mapped namespace",
                      "The %post section will be run under the fakeroot command",
                      "Using --fix-perms because building from a definition file",
                      " without either root user or unprivileged user namespaces"],
}


def install(directory: Path) -> Path:
    """Write an executable ``apptainer`` into *directory* and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "apptainer"
    path.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(__file__)} \"$@\"\n")
    path.chmod(0o755)
    return path


def read_log(path: Path) -> list[dict]:
    """Every build a run with ``FAKE_BUILD_LOG=<path>`` recorded, in order."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _fatal(message: str) -> int:
    print(f"FATAL:   {message}", file=sys.stderr)
    return 255


def _build(argv: list[str]) -> int:
    if "--ignore-fakeroot-command" in argv or argv[:1] != ["--fakeroot"] or len(argv) != 3:
        print(f"fake apptainer build: unexpected arguments {argv}", file=sys.stderr)
        return 2
    out, def_file = Path(argv[1]), Path(argv[2])
    if os.environ.get("FAKE_BUILD_FAIL") == "1":
        return _fatal("fake build failure requested")
    for line in MODE_LINES[os.environ.get("FAKE_BUILD_MODE", "root-mapped-fakeroot")]:
        print(f"INFO:    {line}", file=sys.stderr)
    text = def_file.read_text()
    lines = text.splitlines()
    source = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("From:"))
    if "Bootstrap: localimage" in lines and not Path(source).is_file():
        return _fatal(f"could not open image {source}: no such file")
    log = os.environ.get("FAKE_BUILD_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps({
                "from": source, "out": str(out),
                "env": sorted(k for k in os.environ if k.startswith("APPTAINER_")),
                "login": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in os.environ.items()
                          if k.startswith("APPTAINER_DOCKER_")},
                "registry": sorted(k for k in os.environ if k.startswith("REGISTRY_"))}) + "\n")
    cache = os.environ.get("APPTAINER_CACHEDIR")
    if cache:
        # A pull leaves the layers it fetched in apptainer's cache.
        Path(cache).mkdir(parents=True, exist_ok=True)
        (Path(cache) / "layer.tar").write_bytes(b"pulled layer\n")
    out.write_bytes(f"fake-sif {out.parent.name}\n".encode() + text.encode())
    print("INFO:    Build complete: " + str(out), file=sys.stderr)
    return 0


def _exec(argv: list[str]) -> int:
    binds = []
    i = 1
    while i < len(argv) and argv[i].startswith("--"):
        if argv[i] == "--bind":
            binds.append(argv[i + 1].split(":"))
        i += 1 if argv[i] in _FLAGS else 2
    if i >= len(argv) or not Path(argv[i]).is_dir():
        return _fatal(f"could not open image {argv[i] if i < len(argv) else '?'}: not an extracted directory")
    hosts = {parts[1]: parts[0] for parts in binds}
    inventory = os.environ.get("FAKE_INVENTORY")
    if inventory and Path(inventory).is_dir():
        first = (Path(argv[i]) / ".fake-image").read_bytes().split(b"\n", 1)[0].decode(errors="replace")
        own = Path(inventory) / f"{first.removeprefix('fake-sif ')}.txt"
        inventory = str(own if own.is_file() else Path(inventory) / "default.txt")
    if inventory and INVENTORY_TARGET in hosts:
        workdir = Path(argv[argv.index("--workdir") + 1])
        workdir.mkdir(parents=True, exist_ok=True)
        stand_in = workdir / "fake-inventory.sh"
        stand_in.write_text(f"exec cat {shlex.quote(inventory)}\n")
        hosts[INVENTORY_TARGET] = str(stand_in)
    command = [hosts.get(word, word) for word in argv[i + 1:]]
    return fake_apptainer.main([*argv[:i + 1], *command])


def main(argv: list[str]) -> int:
    if argv[:2] == ["build", "--sandbox"]:
        return fake_apptainer.extract(argv)
    if argv[:1] == ["build"]:
        return _build(argv[1:])
    if argv[:1] == ["exec"]:
        return _exec(argv)
    print(f"fake apptainer: only build and exec are supported, got {argv[:1]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
