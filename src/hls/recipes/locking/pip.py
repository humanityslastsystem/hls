"""Compile the hashed pip locks the image recipes install.

    hls images lock-pip base <name> [--layer-owned <package> ...]
    hls images lock-pip layer <name> [--task <task>] [--parent <lock>] [--source-build <package> ...] [--server]
    hls images lock-pip layer <name> --task <task> --server --no-deps

Each reads ``locks/<name>.in`` and writes ``locks/<name>.txt``: the shared lock directory's, or,
with ``--task``, ``tasks/<task>/locks/``, where a task keeps the locks of the layers it declares
for itself. A layer is compiled against the lock of the image it builds on, ``--parent``: one of
the task's own locks, else one of the shared directory's. A layer with no ``--parent`` builds on
the ``cuda`` base, which installs no pip lock.

Every lock is compiled by ``uv pip compile --generate-hashes`` for the images' interpreter and
platform (``hls.recipes.pins``), from the PyTorch cu130 index and PyPI, counting only uploads
before a cutoff, so that compiling again gives the same lock: ``pins.EXCLUDE_NEWER``, unless the
``.in`` names its own on a line ``# uploads before <YYYY-MM-DDTHH:MM:SSZ>``. The lock's header
records the cutoff it was compiled with. It must be the uv
``pins.LOCK_UV`` names, which the header records; another is refused. uv runs in a private
directory holding only its inputs, under their own names, with no uv configuration file and no
``UV_*`` or ``PIP_*`` variable, so neither the checkout's path nor a user's settings reach the
lock's bytes.

Resolution takes wheels alone, as the images install them, but for the packages a layer names
with ``--source-build``: those are the layer's ``source_builds`` in ``hls.recipes.catalog``,
which build from their source distributions. A lock that wheels alone could not install is
refused here rather than at build time.

A lock is refused when it names a package of the deny list
(``hls.egress.denied_packages``): no base may hold one, nor any layer an agent cap may build on.
A task's server layer (``--server``, with ``--task``), which no agent works in, may hold one: its
serving framework. Compiling a base also writes
``constraints-<base without "base-">.txt``: the torch, torchaudio and torchvision pins the
image holds with ``UV_CONSTRAINT`` and ``PIP_CONSTRAINT``.

``--layer-owned`` leaves a package the base resolves out of its lock, so each layer installs
its own version. huggingface-hub is one: transformers 4.57.3 (the TTS layer) needs it below
1.0 and transformers 5 above 1.5, while the base's datasets and accelerate take either. A base
installed alone then lacks that package; every image a role runs is a base with a layer.

A lock compiled with ``--no-deps`` is exactly the packages its ``.in`` names, each as
``name==version``, hashed: nothing is resolved. It is how a layer on a registry image is locked,
which has no parent lock to compile against (the vendor image's own packages are in no lock), so
nothing the vendor image holds can be moved by a dependency; and how a server layer is locked
whose set no single resolution gives, a stack installed in steps where a later step replaced a
package an earlier one had pinned.

A package neither index carries as a wheel is named in the ``.in`` by the URL of its wheel,
``name @ https://...whl``: the lock holds that URL with the wheel's hash, and the image installs
it from there.

A layer is compiled with every pin of its parent's lock as a constraint, then reduced to the
packages its parent lacks. A layer that cannot resolve against its parent cannot share that
base; uv's message says why, and the lock is refused."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from hls.egress.denied_packages import DENIED_PACKAGES
from hls.recipes import pins
from hls.recipes.locking import LockRefused, lock_name
from hls.recipes.render import TASKS_DIR

#: The command a lock's first line says made it.
COMMAND = "hls images lock-pip"
#: Packages a base holds at its lock's version for every later install.
HELD = ("torch", "torchaudio", "torchvision")
_REQ = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)")
# A requirement that names its wheel by URL, whose file name carries the version.
_URL_REQ = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*) @ https://\S+/[A-Za-z0-9][A-Za-z0-9._]*-([A-Za-z0-9.+!_]+)"
                      r"(?:-[A-Za-z0-9._]+){3,4}\.whl(?=\s|$)")
# A whole requirement line that pins one package at one version, by version or by its wheel's URL.
_PIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(==[A-Za-z0-9.+!_-]+| @ https://[A-Za-z0-9._~+/-]+\.whl)")
# The line of a lock's input that names its own upload cutoff.
_CUTOFF = re.compile(r"#\s*uploads before\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s*")
# The name a layer's parent pins are given to uv as, which its annotations show.
_PARENT_PINS = "parent-pins.txt"


def normalize(name: str) -> str:
    """PEP 503 normal form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def blocks(text: str) -> list[tuple[str, str, str]]:
    """Each requirement of a compiled lock as (normalized name, version, its lines)."""
    out: list[tuple[str, str, list[str]]] = []
    for line in text.splitlines():
        m = _REQ.match(line) or _URL_REQ.match(line)
        if m:
            out.append((normalize(m.group(1)), m.group(2), [line]))
        elif out and line.startswith((" ", "\t")):
            out[-1][2].append(line)
    return [(n, v, "\n".join(lines)) for n, v, lines in out]


def pins_of(text: str) -> dict[str, str]:
    """Normalized name to version, for every requirement of a compiled lock."""
    return {n: v for n, v, _ in blocks(text)}


def denied_in(text: str) -> list[str]:
    """The deny-listed packages a compiled lock names, normalized and sorted."""
    denied = {normalize(n) for n in DENIED_PACKAGES}
    return sorted(n for n in pins_of(text) if n in denied)


def _allowed(name: str, body: str, *, server: bool = False) -> str:
    """*body*, lock *name*'s requirements, unless it names a deny-listed package and is not a
    server layer's."""
    denied = [] if server else denied_in(body)
    if denied:
        raise LockRefused(f"{name} would hold deny-listed packages: {denied}")
    return body


def reduce_layer(layer_text: str, parent_text: str) -> str:
    """The requirements of *layer_text* that *parent_text* does not pin at the same version."""
    parent = pins_of(parent_text)
    return "\n".join(b for n, v, b in blocks(layer_text) if parent.get(n) != v) + "\n"


def cutoff(src_text: str) -> str:
    """The upload cutoff a lock's input *src_text* names, else ``pins.EXCLUDE_NEWER``. Raise
    LockRefused for an input that names two."""
    named = [m.group(1) for line in src_text.splitlines() if (m := _CUTOFF.fullmatch(line.strip()))]
    if len(named) > 1:
        raise LockRefused(f"the input names {len(named)} upload cutoffs: {named}")
    return named[0] if named else pins.EXCLUDE_NEWER


def compile_lock(src: Path, out: Path, *, constraints: Path | None = None, omit: tuple[str, ...] = (),
                 source_builds: tuple[str, ...] = (), exclude_newer: str = pins.EXCLUDE_NEWER,
                 no_deps: bool = False) -> None:
    """Compile *src* into *out*: wheels only but for *source_builds*, under *constraints*, leaving
    *omit* out, counting uploads before *exclude_newer*; with *no_deps*, the requirements *src*
    names and none of their dependencies. Raise LockRefused with uv's message when it cannot
    resolve."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    asked = subprocess.run(["uv", "--version"], env=env, capture_output=True, text=True)
    words = asked.stdout.split()
    if asked.returncode != 0 or words[:2] != ["uv", pins.LOCK_UV]:
        raise LockRefused(f"{COMMAND} needs uv {pins.LOCK_UV}, which compiled the committed locks; "
                         f"`uv --version` says {asked.stdout.strip() or asked.stderr.strip()!r}")
    with tempfile.TemporaryDirectory(prefix="hls-lock-") as tmp:
        work = Path(tmp)
        shutil.copyfile(src, work / src.name)
        argv = ["uv", "pip", "compile", src.name, "--no-config", "--generate-hashes", "--no-header",
                "--quiet", "--python-version", pins.PYTHON_VERSION, "--python-platform", pins.PYTHON_PLATFORM,
                "--index-url", pins.INDEX_URL, "--extra-index-url", pins.EXTRA_INDEX_URL,
                "--index-strategy", "unsafe-best-match", "--exclude-newer", exclude_newer,
                "--only-binary", ":all:", "-o", "lock.txt"]
        if constraints is not None:
            shutil.copyfile(constraints, work / _PARENT_PINS)
            argv += ["--constraint", _PARENT_PINS]
        for name in source_builds:
            argv += ["--no-binary", name]
        for name in omit:
            argv += ["--no-emit-package", name]
        if no_deps:
            argv.append("--no-deps")
        done = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise LockRefused(f"uv pip compile {src.name} failed:\n{done.stderr}")
        out.write_text((work / "lock.txt").read_text())


def arguments(kind: str, name: str, *, parent: str | None = None, layer_owned: Iterable[str] = (),
              source_builds: Iterable[str] = (), task: str | None = None, server: bool = False,
              no_deps: bool = False) -> list[str]:
    """The arguments of ``hls images lock-pip`` that make lock *name* again, options in one order."""
    args = [kind, name] + (["--task", task] if task else []) + (["--parent", parent] if parent else [])
    args += [word for package in layer_owned for word in ("--layer-owned", package)]
    args += [word for package in source_builds for word in ("--source-build", package)]
    return args + (["--server"] if server else []) + (["--no-deps"] if no_deps else [])


def _exact(name: str, src_text: str) -> None:
    """Raise LockRefused unless every requirement of *src_text*, lock *name*'s input, is one
    ``name==version``: a lock compiled without dependencies holds exactly what its input names."""
    loose = [s for line in src_text.splitlines() if (s := line.strip()) and not s.startswith("#")
             and not _PIN.fullmatch(s)]
    if loose:
        raise LockRefused(f"{name}: a lock compiled with --no-deps names every package as name==version, or by "
                          f"the URL of its wheel, and its input has {loose}")


def header(args: list[str], exclude_newer: str = pins.EXCLUDE_NEWER) -> str:
    """The two comment lines a lock starts with: how to make it again (*args*, ``arguments``), and
    from what, uploads before *exclude_newer* among it."""
    return (f"# Generated by: {COMMAND} {' '.join(args)}\n"
            f"# Python {pins.PYTHON_VERSION}, {pins.PYTHON_PLATFORM}; uv {pins.LOCK_UV}; "
            f"{pins.INDEX_URL} then {pins.EXTRA_INDEX_URL}; uploads before {exclude_newer}\n")


def lock_pip(kind: Literal["base", "layer"], name: str, *, compiler: Callable[..., None], parent: str | None = None,
             layer_owned: tuple[str, ...] = (), source_builds: tuple[str, ...] = (), locks: Path = pins.LOCKS_DIR,
             task: str | None = None, server: bool = False, no_deps: bool = False, tasks_dir: Path = TASKS_DIR,
             echo: Callable[[str], None] = print) -> dict[str, str]:
    """Compile ``<name>.in`` into ``<name>.txt`` beside it, under *locks*, or for a layer of task
    *task*, under ``<tasks_dir>/<task>/locks``: as a base (leaving *layer_owned* out, and writing
    its torch constraints) or as a layer of lock *parent*, the task's own or else one under
    *locks* (building *source_builds* from source), or with no parent, on the cuda base, which pins
    nothing; as a *server* layer it may hold a deny-listed serving framework. With *no_deps*, the
    lock of a task's server layer is exactly the pins its input names. Return its pins.
    *compiler* compiles, as ``compile_lock`` does. Raise LockRefused, with nothing written, for a
    base given a parent, source builds or a task, a server layer that is no task's, a layer given
    packages to leave out, a name or parent that is no lock name, a missing ``.in`` or parent
    lock, a lock uv cannot resolve, one that names a deny-listed package and is no server
    layer's, and for *no_deps* on anything but a task's server layer with no parent and no source
    build, or with an input that names a requirement other than one pin."""
    lock_name(name)
    if parent is not None:
        lock_name(parent)
    if server and not task:
        raise LockRefused("a server layer's lock may hold what the deny list names, so it is a task's own: "
                          "name its --task")
    if kind == "base" and (parent or task):
        raise LockRefused("a base has no parent and is shared")
    if kind == "base" and source_builds:
        raise LockRefused("a base builds nothing from source; its packages install from wheels")
    if kind == "layer" and layer_owned:
        raise LockRefused("a layer owns what it adds")
    if no_deps and (kind != "layer" or not task or not server or parent or source_builds):
        raise LockRefused("--no-deps compiles the lock of a task's server layer, with no parent lock and no "
                          "source build")
    own = locks if task is None else tasks_dir / task / "locks"
    src = own / f"{name}.in"
    if not src.is_file():
        raise LockRefused(f"{name}: {src}, the requirements to compile, does not exist")
    parent_lock = None
    if kind == "layer" and parent:
        parent_lock = own / f"{parent}.txt" if (own / f"{parent}.txt").is_file() else locks / f"{parent}.txt"
        if not parent_lock.is_file():
            raise LockRefused(f"{name}: its parent's lock {parent}.txt is in neither {own} nor {locks}")
    before = cutoff(src.read_text())
    if no_deps:
        _exact(name, src.read_text())
    with tempfile.TemporaryDirectory(prefix="hls-lock-") as tmp:
        raw = Path(tmp) / "raw.txt"
        if kind == "base":
            compiler(src, raw, omit=tuple(layer_owned), exclude_newer=before)
            body = _allowed(name, raw.read_text())
            held = pins_of(body)
            constraint_lines = [f"{n}=={held[n]}" for n in HELD if n in held]
            short = name.removeprefix("base-")
            (locks / f"constraints-{short}.txt").write_text("\n".join(constraint_lines) + "\n")
        elif no_deps:
            compiler(src, raw, exclude_newer=before, no_deps=True)
            body = _allowed(name, raw.read_text(), server=server)
        else:
            parent_text = parent_lock.read_text() if parent_lock is not None else ""
            pins_file = Path(tmp) / "parent-pins.txt"
            pins_file.write_text("".join(f"{n}=={v}\n" for n, v in pins_of(parent_text).items()))
            compiler(src, raw, constraints=pins_file, source_builds=tuple(source_builds), exclude_newer=before)
            body = _allowed(name, reduce_layer(raw.read_text(), parent_text), server=server)
        args = arguments(kind, name, parent=parent, layer_owned=layer_owned, source_builds=source_builds, task=task,
                         server=server, no_deps=no_deps)
        (own / f"{name}.txt").write_text(header(args, before) + body)
    locked = pins_of(body)
    echo(f"{name}: {len(locked)} packages")
    return locked
