"""The interpreter harbor's exec server runs on inside a sandbox container, staged once into the
site store a tool runner binds it from.

Harbor starts a small HTTP server inside every trajectory's container to run a turn's command,
and that server imports fastapi and uvicorn. An image that ships neither, and no usable pip,
cannot start it, so the interpreter lives outside every image: a standalone CPython build with
the two packages, bound read-only into each container, where the launch names it as the
server's interpreter (``SandboxToolRunner.server_python``). A standalone build rather than a
venv, because a venv records the absolute path it was built at, and this tree is built on a
host path and read at another inside the container.

Each tree lives in its own directory under the store, named for the Python and the package pins
(``RUNTIME_TAG``), so a change of either stages a new tree beside the old one. A tree is built
beside its final place and renamed into it only once it imports both packages at their pins; a
directory of that name that does not is moved aside, never removed."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: The CPython the runtime is: a standalone build, which uv fetches.
RUNTIME_PYTHON = "3.12.13"
#: What the exec server imports, at the versions the runtime holds.
RUNTIME_PACKAGES = {"fastapi": "0.136.3", "uvicorn": "0.48.0"}
#: The runtime's directory under its store.
RUNTIME_TAG = f"cpython-{RUNTIME_PYTHON}-" + "-".join(f"{n}-{v}" for n, v in RUNTIME_PACKAGES.items())
#: Seconds each staging step may take: a download and a pip install.
STEP_TIMEOUT_SEC = 1800

#: The check a staged tree passes: each package imports at its pinned version.
_CHECK = ("import importlib.metadata as m, sys; pins = dict(a.split('=', 1) for a in sys.argv[1:]); "
          "bad = [f'{n} {m.version(n)}' for n, v in pins.items() if m.version(n) != v]; "
          "__import__('fastapi'), __import__('uvicorn'); sys.exit(', '.join(bad) or 0)")


class RuntimeStageError(RuntimeError):
    """The runtime could not be staged; the message says which step failed and how."""


def held(root: Path) -> bool:
    """Whether *root* holds the runtime: its ``bin/python3`` imports each package at its pin."""
    python = root / "bin" / "python3"
    if not python.is_file():
        return False
    done = subprocess.run([str(python), "-c", _CHECK, *(f"{n}={v}" for n, v in RUNTIME_PACKAGES.items())],
                          capture_output=True, text=True, timeout=120)
    return done.returncode == 0


def _uv() -> str:
    """The uv binary: on PATH, beside this interpreter, or where uv's installer puts it."""
    for found in (shutil.which("uv"), Path(sys.executable).with_name("uv"), Path.home() / ".local" / "bin" / "uv"):
        if found is not None and Path(found).is_file():
            return str(found)
    raise RuntimeStageError("staging the exec server's runtime needs uv: on this node's PATH, beside "
                            f"{sys.executable} or in ~/.local/bin")


def _run(argv: list[str], step: str) -> None:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=STEP_TIMEOUT_SEC)
    if done.returncode != 0:
        tail = " ".join((done.stderr or done.stdout).split())[-300:]
        raise RuntimeStageError(f"staging the exec server's runtime: {step} failed: {tail}")


def stage(root: Path) -> str:
    """Put the runtime at *root*, and return ``"held"`` when it was already there or
    ``"staged"``. The tree is built at a sibling of *root* and renamed into place once it passes
    the check (``held``); a *root* that exists and does not pass is renamed aside first. Raise
    RuntimeStageError for a step that fails."""
    if held(root):
        return "held"
    root.parent.mkdir(parents=True, exist_ok=True)
    partial = root.with_name(f"{root.name}.partial-{int(time.time())}")
    with tempfile.TemporaryDirectory(prefix="exec-server-runtime-") as scratch:
        _run([_uv(), "python", "install", "--install-dir", f"{scratch}/python", RUNTIME_PYTHON],
             f"uv python install {RUNTIME_PYTHON}")
        built = sorted(Path(scratch, "python").glob("cpython-*"))
        built = [b for b in built if b.is_dir() and not b.is_symlink()]
        if not built:
            raise RuntimeStageError("staging the exec server's runtime: uv installed no cpython tree")
        shutil.move(str(built[0]), partial)
    # The build ships PEP 668's EXTERNALLY-MANAGED marker, which stops installs into a system
    # Python; this tree is the harness's own, and installing into it is the point.
    for marker in partial.glob("lib/python*/EXTERNALLY-MANAGED"):
        marker.unlink()
    _run([str(partial / "bin" / "python3"), "-m", "pip", "install", "--no-cache-dir", "--quiet",
          *(f"{n}=={v}" for n, v in RUNTIME_PACKAGES.items())], "pip install")
    if not held(partial):
        raise RuntimeStageError(f"staging the exec server's runtime: the tree built at {partial} does not import "
                                f"{', '.join(f'{n} {v}' for n, v in RUNTIME_PACKAGES.items())}")
    if root.exists():
        root.rename(root.with_name(f"{root.name}.unusable-{int(time.time())}"))
    partial.rename(root)
    return "staged"
