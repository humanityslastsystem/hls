"""Generate the agent cap's ``package-lock.json``.

    hls images lock-npm <lock name>      # npm-agent.json

It writes ``locks/<lock name>``, the lock every agent cap of ``hls.recipes.catalog`` that names
it installs from with ``npm ci``. The lock is resolved from the cap's ``package.json``
(``hls.recipes.render.npm_manifest``) by the npm of the Node the cap installs: that Node's
release tarball, checked against ``NodePin.sha256`` before it is unpacked into a private
directory. npm counts only versions published before ``pins.EXCLUDE_NEWER`` (``--before``), reads
no user or global configuration and no ``npm_*`` variable, asks the public registry, and runs
no install script, so resolving again gives the same lock. The lock is written only once the
renderer accepts it: lockfile version 2 or 3, exactly the declared CLIs at its root, every
package fetched over https and checked by a sha512 integrity hash. The command prints the Node
and npm versions it used."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from hls.recipes import catalog, pins, render
from hls.recipes.locking import LockRefused, lock_name, unreachable
from hls.recipes.model import NodePin

#: The registry the CLIs' trees resolve from.
REGISTRY = "https://registry.npmjs.org/"


def fetch_node(pin: NodePin) -> bytes:
    """The release tarball of Node *pin*, unchecked. Raise LockRefused, naming the host, when it
    cannot be read."""
    url = render.node_url(pin)
    try:
        with urllib.request.urlopen(url, timeout=300) as r:
            return r.read()
    except OSError as e:
        raise unreachable(url, e) from None


def unpack_node(data: bytes, pin: NodePin, dest: Path) -> Path:
    """Unpack Node tarball *data* into *dest*, without its top directory, once its sha256 is
    *pin*'s; return *dest*. Raise LockRefused, with nothing unpacked, when it is not."""
    got = hashlib.sha256(data).hexdigest()
    if got != pin.sha256:
        raise LockRefused(f"{render.node_url(pin)} has sha256 {got}, which does not match the pinned {pin.sha256}")
    top = f"node-v{pin.version}-linux-x64/"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = []
        for member in tar.getmembers():
            if not member.name.startswith(top) or member.name == top:
                continue
            member.name = member.name.removeprefix(top)
            if member.islnk():
                member.linkname = member.linkname.removeprefix(top)
            members.append(member)
        dest.mkdir(parents=True)
        tar.extractall(dest, members=members, filter="data")
    return dest


def _env(home: Path, node_home: Path) -> dict[str, str]:
    """The caller's environment without npm's settings, with *home* as HOME and Node first on PATH."""
    env = {k: v for k, v in os.environ.items() if not k.lower().startswith("npm_")}
    env.update(HOME=str(home), PATH=f"{node_home / 'bin'}:/usr/bin:/bin")
    return env


def _run(argv: list[str], cwd: Path, env: dict[str, str]) -> str:
    done = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise LockRefused(f"{' '.join(argv[1:3])} failed:\n{done.stderr}")
    return done.stdout


def versions(node_home: Path) -> tuple[str, str]:
    """``node --version`` and ``npm --version`` of the Node unpacked at *node_home*."""
    node, npm = str(node_home / "bin" / "node"), str(node_home / "bin" / "npm")
    env = _env(node_home, node_home)
    return (_run([node, "--version"], node_home, env).strip(),
            _run([node, npm, "--version"], node_home, env).strip())


def resolve_lock(manifest: str, node_home: Path, work: Path) -> bytes:
    """The ``package-lock.json`` npm writes for ``package.json`` *manifest*, resolved in a fresh
    directory *work* by the npm of the Node at *node_home*."""
    work.mkdir(parents=True)
    (work / "package.json").write_text(manifest)
    home = work / "home"
    home.mkdir()
    # npm refuses one file as both configurations; files that do not exist give it none.
    argv = [str(node_home / "bin" / "node"), str(node_home / "bin" / "npm"), "install", "--package-lock-only",
            "--ignore-scripts", "--no-audit", "--no-fund", "--no-update-notifier",
            "--before", pins.EXCLUDE_NEWER, "--registry", REGISTRY,
            "--userconfig", str(work / "no-user-npmrc"), "--globalconfig", str(work / "no-global-npmrc"),
            "--cache", str(work / "cache")]
    _run(argv, work, _env(home, node_home))
    return (work / "package-lock.json").read_bytes()


def lock_npm(name: str, *, fetch: Callable[[NodePin], bytes], resolve: Callable[[str, Path, Path], bytes],
             locks: Path = pins.LOCKS_DIR, echo: Callable[[str], None] = print) -> bytes:
    """Resolve lock *name* of the agent caps that install from it and write it under *locks*, once
    the renderer accepts it beside the other locks there; return it. *fetch* downloads a Node
    tarball, as ``fetch_node`` does, and *resolve* runs npm, as ``resolve_lock`` does. Raise
    LockRefused, with nothing written, for a name that is no lock name or a lock no cap installs
    from, caps that share it but declare different CLIs or Nodes, a Node that cannot be downloaded,
    a Node tarball or ``node --version`` that is not the pinned one, an npm that fails, and a lock
    the renderer refuses."""
    lock_name(name)
    caps = [cap for cap in catalog.AGENT_CAPS.values() if cap.npm_lock == name]
    if not caps:
        raise LockRefused(f"no agent cap installs from {name}")
    if len({render.npm_manifest(cap) for cap in caps}) != 1 or len({cap.node for cap in caps}) != 1:
        raise LockRefused(f"the agent caps that install from {name} declare different CLIs or Nodes")
    node = caps[0].node
    with tempfile.TemporaryDirectory(prefix="hls-npm-lock-") as tmp:
        node_home = unpack_node(fetch(node), node, Path(tmp) / "node")
        node_version, npm_version = versions(node_home)
        if node_version != f"v{node.version}":
            raise LockRefused(f"the unpacked node is {node_version}, not v{node.version}")
        lock = resolve(render.npm_manifest(caps[0]), node_home, Path(tmp) / "work")
        # Checked as the images stage would read it, beside the other committed locks.
        check = Path(tmp) / "locks"
        check.mkdir()
        for path in locks.iterdir():
            if path.is_file():
                (check / path.name).write_bytes(path.read_bytes())
        (check / name).write_bytes(lock)
        for cap in caps:
            try:
                render.render_def(cap, parent_sif=Path("/hls/parent.sif"), locks=check, uv=pins.UV)
            except (ValueError, FileNotFoundError) as e:
                raise LockRefused(f"{name}: the renderer refuses it: {e}") from None
        (locks / name).write_bytes(lock)
    echo(f"{name}: node {node_version}, npm {npm_version}")
    return lock
