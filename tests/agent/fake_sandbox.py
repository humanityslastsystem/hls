"""A stand-in for ``apptainer exec`` and ``apptainer build --sandbox`` for the agent build's tests:
it gives a container its own paths on a machine without apptainer, and runs the real exec server
in it without namespaces.

``exec`` accepts the command lines ``ApptainerRuntime.argv`` builds and refuses one that lacks an
isolation flag or names an image that is not an extracted directory (exit 2), as
``tests.substrate.fake_apptainer`` does. It cannot mount anything, so every argument, the
``--cwd`` directory and every env-file value that names a bind's target, or ``/tmp``, is
rewritten to the bind's source, or to ``<workdir>/tmp`` (``/run/hls-control/exec.sock`` becomes
``<the control directory>/exec.sock``); ``/usr/bin/python3`` and ``/bin/sh`` stay the host's.
The exec server's command gains ``--no-namespaces``: the fake cannot make them. Harbor's own
commands, sent to the exec server later, are rewritten the same way by the tests' environment
(``tests.agent.test_stage.MappedEnvironment``). The container's variables are the env file's,
``HOME`` under the workdir and ``PATH`` from ``FAKE_SANDBOX_PATH``; nothing else of the fake's
own environment. A Slurm step's processes end together when the step is stopped; here the
container's command is killed when its parent ends (``PR_SET_PDEATHSIG``), and the exec
server's spawner takes every command with it.

``build --sandbox <dir> <image file>`` extracts as ``tests.substrate.fake_apptainer`` does, and
an image file whose bytes are a JSON object with a ``workspace`` object also gets each of its
entries, relative path to text, as a file under ``<dir>/workspace``: an image that baked a tree
into ``/workspace`` (``image_bytes``).

``FAKE_SANDBOX_RECORDS=<dir>`` writes ``<dir>/<role>.json`` with what ``exec`` was asked: the
binds, the GPU flag, the dropped capabilities, the env file's variables and the command. *role*
is the name of the workdir's parent, the role's own directory.

``FAKE_SANDBOX_START_DELAY=<sec>`` holds the build sandbox's container (the role named
``sandbox``) that long before it starts its command, as a slow runtime or seal does: its start
marker comes only then.

``install(dir)`` writes an executable ``apptainer`` into *dir* that runs this file with the test's
interpreter."""

from __future__ import annotations

import ctypes
import json
import os
import re
import shlex
import signal
import sys
import time
from pathlib import Path

# prctl's option that sends the process a signal when its parent ends.
_PR_SET_PDEATHSIG = 1

_FLAGS = {"--containall", "--cleanenv", "--pid", "--nv"}
_OPTIONS = {"--no-mount", "--workdir", "--cwd", "--bind", "--env-file", "--drop-caps"}
_REQUIRED = ("--containall", "--cleanenv", "--pid")


def install(directory: Path) -> Path:
    """Write an executable ``apptainer`` into *directory* and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "apptainer"
    path.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(__file__)} \"$@\"\n")
    path.chmod(0o755)
    return path


def image_bytes(workspace: dict[str, str] | None = None, *, has_workspace: bool = True) -> bytes:
    """The bytes of an image file whose extraction holds *workspace* at ``/workspace``, a
    directory every agent cap makes; with *has_workspace* False, one that makes none."""
    return json.dumps({"workspace": workspace or {}, "has_workspace": has_workspace}).encode()


def records(directory: Path) -> dict[str, dict]:
    """What each container recorded, by role."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))}


def mapped(paths: dict[str, str], text: str) -> str:
    """*text* with each container path of *paths* that stands as a whole path, or as a path's
    head, replaced by its host path."""
    if not paths:
        return text
    # One pass, longest first, so a host path that was just put in is never rewritten again.
    targets = "|".join(re.escape(t) for t in sorted(paths, key=len, reverse=True))
    pattern = r"(?<![\w./-])(" + targets + r")(?=$|[/\s'\":;)|&>])"
    return re.sub(pattern, lambda m: paths[m.group(1)], text)


def _read_env_file(path: str) -> dict[str, str]:
    env = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        [word] = shlex.split(line)
        name, _, value = word.partition("=")
        env[name] = value
    return env


def extract(dest: Path, image: Path) -> int:
    """``build --sandbox``: see the module docstring."""
    if os.path.lexists(dest):
        print(f"fake sandbox: {dest} exists", file=sys.stderr)
        return 2
    if not image.is_file() or image.is_symlink():
        print(f"FATAL:   could not open image {image}", file=sys.stderr)
        return 255
    for d in ("bin", "etc", "usr/bin", "tmp"):
        (dest / d).mkdir(parents=True)
    data = image.read_bytes()
    (dest / ".fake-image").write_bytes(data)
    try:
        declared = json.loads(data)
        baked, has_workspace = declared.get("workspace") or {}, declared.get("has_workspace", True)
    except (ValueError, AttributeError):
        baked, has_workspace = {}, True
    if has_workspace:
        (dest / "workspace").mkdir()
    for rel, text in baked.items():
        path = dest / "workspace" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return 0


def main(argv: list[str]) -> int:
    if argv[:2] == ["build", "--sandbox"] and len(argv) == 4:
        return extract(Path(argv[2]), Path(argv[3]))
    if argv[:1] != ["exec"]:
        print(f"fake sandbox: only exec and build --sandbox are supported, got {argv[:2]}", file=sys.stderr)
        return 2
    flags: set[str] = set()
    options: dict[str, list[str]] = {}
    i = 1
    while i < len(argv) and argv[i].startswith("--"):
        if argv[i] in _FLAGS:
            flags.add(argv[i])
            i += 1
        elif argv[i] in _OPTIONS and i + 1 < len(argv):
            options.setdefault(argv[i], []).append(argv[i + 1])
            i += 2
        else:
            print(f"fake sandbox: unknown or incomplete option {argv[i]}", file=sys.stderr)
            return 2
    if any(f not in flags for f in _REQUIRED) or len(options.get("--workdir", [])) != 1 \
            or not options.get("--no-mount"):
        print("fake sandbox: missing isolation", file=sys.stderr)
        return 2
    image, inside = argv[i], argv[i + 1:]
    if not Path(image).is_dir():
        print(f"fake sandbox: image {image} is not an extracted directory", file=sys.stderr)
        return 2
    workdir = Path(options["--workdir"][0])
    (workdir / "tmp").mkdir(parents=True, exist_ok=True)
    paths = {"/tmp": str(workdir / "tmp")}
    for bind in options.get("--bind", []):
        source, target = bind.split(":")[:2]
        if source != target:  # the runtime binds the host's resolver files at their own paths
            paths[target] = source
    given: dict[str, str] = {}
    for path in options.get("--env-file", []):
        given.update(_read_env_file(path))
    command = inside[4:]  # after /bin/sh -c <wrapper> <marker>
    record = os.environ.get("FAKE_SANDBOX_RECORDS")
    if record:
        Path(record).mkdir(parents=True, exist_ok=True)
        (Path(record) / f"{workdir.parent.name}.json").write_text(json.dumps({
            "image": image, "nv": "--nv" in flags, "binds": options.get("--bind", []),
            "drop_caps": (options.get("--drop-caps") or [None])[0], "cwd": (options.get("--cwd") or [None])[0],
            "env": given, "command": command}))
    home = workdir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = {k: mapped(paths, v) for k, v in given.items()}
    env.update(PATH=os.environ.get("FAKE_SANDBOX_PATH", os.environ.get("PATH", os.defpath)), HOME=str(home))
    command = [mapped(paths, a) for a in command]
    if any(a.endswith("/exec_server.py") for a in command):
        command.append("--no-namespaces")
    cwd = (options.get("--cwd") or [None])[0]
    if cwd is not None:
        os.chdir(mapped(paths, cwd))
    if workdir.parent.name == "sandbox":
        time.sleep(float(os.environ.get("FAKE_SANDBOX_START_DELAY", "0")))
    parent = os.getppid()
    ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    if os.getppid() != parent:
        return 1  # the role was stopped before the container started
    os.execve(inside[0], [*inside[:4], *command], env)
    return 0  # not reached


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
