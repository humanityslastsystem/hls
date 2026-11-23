"""Pin apt packages to exact versions for the image recipes.

    hls images lock-apt ubuntu <name>                 # from the Ubuntu snapshot
    hls images lock-apt nvidia <name>                 # from NVIDIA's CUDA repository
    hls images lock-apt ubuntu <name> --task <task>   # a task's own layer, from the snapshot

Each reads ``locks/<name>.in`` and writes ``locks/<name>.txt``: the shared lock directory's, or,
with ``--task``, ``tasks/<task>/locks/``, where a task keeps the locks of the layers it declares
for itself. Such a layer installs from the snapshot alone.

An ``.in`` file lists package names, one per line (``#`` starts a comment). The ``.txt`` file
holds one ``name=version`` per line, in the same order. Ubuntu packages resolve against the
dated snapshot archive ``hls.recipes.pins.SNAPSHOT`` names: the highest version any of the
noble, noble-updates and noble-security pockets carries, in any component. Only the named
packages are pinned; apt resolves their dependencies from the same snapshot, which is fixed in
time, so the installed set is reproducible.

NVIDIA's CUDA repository (``pins.NVIDIA_REPO``) has no snapshot, so its packages take the
highest version it carries today, and the committed file is the record. Each of its lines also
holds the sha256 the index lists for the pinned version's .deb, ``name=version <sha256>``: the
build installs only a .deb that matches it, so a file republished under the same version fails
the build. A dependency the repository itself carries would reach the image at whatever version
the repository has at build time, and once the build's apt preferences hold the repository back,
not at all; the same holds for a virtual name only one of its packages provides. So a lock that
leaves either unpinned is refused: name it, or the package that provides it, in the ``.in`` file.

A name no index carries is refused."""

from __future__ import annotations

import gzip
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from hls.recipes import pins
from hls.recipes.locking import LockRefused, lock_name, unreachable
from hls.recipes.render import TASKS_DIR, UBUNTU_COMPONENTS, UBUNTU_DISTS

#: The command a lock's first line says made it.
COMMAND = "hls images lock-apt"
SNAPSHOT_URL = ("https://snapshot.ubuntu.com/ubuntu/{snapshot}/dists/{dist}/{component}/binary-amd64/"
                "Packages.gz")
# A package name in a Depends field, before its version constraint or architecture qualifier.
_DEPENDENCY = re.compile(r"\s*([a-z0-9][a-z0-9+.-]*)")


def _order(c: str) -> int:
    """dpkg's weight of one character of a non-digit run: ``~`` sorts before everything, even
    the end of the string, and letters before other characters."""
    if c == "~":
        return -1
    if c.isdigit():
        return 0
    if c.isalpha():
        return ord(c)
    return ord(c) + 256


def _compare_part(a: str, b: str) -> int:
    """dpkg's comparison of two upstream-version or revision strings."""
    i = j = 0
    while i < len(a) or j < len(b):
        diff = 0
        while (i < len(a) and not a[i].isdigit()) or (j < len(b) and not b[j].isdigit()):
            ca = _order(a[i]) if i < len(a) and not a[i].isdigit() else 0
            cb = _order(b[j]) if j < len(b) and not b[j].isdigit() else 0
            if ca != cb:
                return -1 if ca < cb else 1
            i += i < len(a) and not a[i].isdigit()
            j += j < len(b) and not b[j].isdigit()
        while i < len(a) and a[i] == "0":
            i += 1
        while j < len(b) and b[j] == "0":
            j += 1
        while i < len(a) and a[i].isdigit() and j < len(b) and b[j].isdigit():
            if not diff:
                diff = ord(a[i]) - ord(b[j])
            i += 1
            j += 1
        if i < len(a) and a[i].isdigit():
            return 1
        if j < len(b) and b[j].isdigit():
            return -1
        if diff:
            return -1 if diff < 0 else 1
    return 0


def compare_versions(a: str, b: str) -> int:
    """-1, 0 or 1 as Debian version *a* sorts before, equal to or after *b*."""
    def split(v: str) -> tuple[int, str, str]:
        epoch, _, rest = v.partition(":") if ":" in v else ("0", "", v)
        upstream, _, revision = rest.rpartition("-") if "-" in rest else (rest, "", "0")
        return int(epoch or 0), upstream, revision

    ea, ua, ra = split(a)
    eb, ub, rb = split(b)
    if ea != eb:
        return -1 if ea < eb else 1
    return _compare_part(ua, ub) or _compare_part(ra, rb)


def _stanzas(text: str) -> list[dict[str, str]]:
    """The fields of each stanza of a Packages index, continuation lines folded in."""
    out: list[dict[str, str]] = []
    fields: dict[str, str] = {}
    last = None
    for line in text.splitlines():
        if not line.strip():
            if fields:
                out.append(fields)
            fields, last = {}, None
        elif line[0] in " \t" and last is not None:
            fields[last] += " " + line.strip()
        elif ":" in line:
            last, _, value = line.partition(":")
            fields[last] = value.strip()
    if fields:
        out.append(fields)
    return out


def parse_packages(text: str) -> dict[str, list[str]]:
    """Every version of every package a Packages index lists."""
    out: dict[str, list[str]] = {}
    for fields in _stanzas(text):
        if "Package" in fields and "Version" in fields:
            out.setdefault(fields["Package"], []).append(fields["Version"])
    return out


def highest(versions: list[str]) -> str:
    """The highest of Debian *versions*."""
    best = versions[0]
    for v in versions[1:]:
        if compare_versions(v, best) > 0:
            best = v
    return best


def read_names(path: Path) -> list[str]:
    """The package names an ``.in`` file lists, in order."""
    names = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def fetch_index(url: str) -> str:
    """The text of the gzipped Packages index at *url*. Raise LockRefused, naming the host, when it
    cannot be read."""
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return gzip.decompress(r.read()).decode("utf-8", errors="replace")
    except OSError as e:
        raise unreachable(url, e) from None


def resolve(names: list[str], indexes: list[str]) -> list[str]:
    """``name=version`` for each of *names*, from the Packages indexes *indexes*."""
    versions: dict[str, list[str]] = {}
    for text in indexes:
        for name, vs in parse_packages(text).items():
            versions.setdefault(name, []).extend(vs)
    missing = [n for n in names if n not in versions]
    if missing:
        raise LockRefused(f"no index carries {missing}")
    return [f"{n}={highest(versions[n])}" for n in names]


def deb_sha256(index: str, pin: str) -> str:
    """The SHA256 Packages index *index* lists for the .deb of *pin* (``name=version``). Raise
    LockRefused when its stanza lists none, or one that is not 64 hex digits."""
    name, version = pin.split("=", 1)
    for fields in _stanzas(index):
        if fields.get("Package") == name and fields.get("Version") == version:
            digest = fields.get("SHA256", "")
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                return digest
    raise LockRefused(f"{pin} has no sha256 in the index")


def _names(field: str) -> list[str]:
    """The package names of a Depends, Pre-Depends or Provides field, every alternative included,
    without versions or architecture qualifiers."""
    return [m.group(1) for part in re.split(r"[,|]", field) if (m := _DEPENDENCY.match(part))]


def unpinned_from(index: str, pinned: list[str]) -> list[str]:
    """The names a package of *pinned* (``name=version``, at that version) depends on or
    pre-depends on, in any alternative, that Packages index *index* alone would have to supply:
    a package it carries that *pinned* does not name, or a virtual name one of its packages
    provides and no pinned version does; sorted."""
    wanted = dict(pin.split("=", 1) for pin in pinned)
    carried = set(parse_packages(index))
    needed: set[str] = set()
    # Virtual name to whether a pinned version provides it.
    provided: dict[str, bool] = {}
    for fields in _stanzas(index):
        is_pinned = wanted.get(fields.get("Package", "")) == fields.get("Version")
        for name in _names(fields.get("Provides", "")):
            provided[name] = provided.get(name, False) or is_pinned
        if is_pinned:
            for field in ("Pre-Depends", "Depends"):
                needed.update(_names(fields.get(field, "")))
    loose = {name for name in needed - set(wanted) if name in carried}
    loose |= {name for name in needed - carried if provided.get(name) is False}
    return sorted(loose)


def lock_apt(repo: Literal["ubuntu", "nvidia"], name: str, *, fetch: Callable[[str], str],
             locks: Path = pins.LOCKS_DIR, task: str | None = None, tasks_dir: Path = TASKS_DIR,
             echo: Callable[[str], None] = print) -> list[str]:
    """Pin the packages ``<name>.in`` lists from *repo* and write ``<name>.txt`` beside it, under
    *locks*, or for a layer of task *task*, under ``<tasks_dir>/<task>/locks``; return its lines.
    *fetch* reads a Packages index, as ``fetch_index`` does. Raise LockRefused, with nothing
    written, for a name no index carries, for an NVIDIA lock that leaves a dependency unpinned or
    whose index lists no sha256 for a pin, for a task's lock from another source than the
    snapshot, and for a name that is no lock name, or a lock with no ``.in`` file."""
    lock_name(name)
    if task is not None:
        if repo != "ubuntu":
            raise LockRefused("a task's layer installs Ubuntu packages from the snapshot alone")
        locks = tasks_dir / task / "locks"
    if not (locks / f"{name}.in").is_file():
        raise LockRefused(f"{name}: {locks / name}.in, the packages to pin, does not exist")
    names = read_names(locks / f"{name}.in")
    if repo == "ubuntu":
        urls = [SNAPSHOT_URL.format(snapshot=pins.SNAPSHOT, dist=d, component=c)
                for d in UBUNTU_DISTS for c in UBUNTU_COMPONENTS]
        source = f"snapshot {pins.SNAPSHOT}"
    else:
        urls = [f"{pins.NVIDIA_REPO}Packages.gz"]
        source = urls[0]
    indexes = [fetch(u) for u in urls]
    pinned = resolve(names, indexes)
    lines = pinned
    if repo == "nvidia":
        loose = unpinned_from(indexes[0], pinned)
        if loose:
            raise LockRefused(f"{name}: the pinned packages depend on {loose}, which {source} carries "
                              "or provides and apt would take at any version, or not at all once the build "
                              f"holds the repository back; name them, or what provides them, in {name}.in")
        lines = [f"{pin} {deb_sha256(indexes[0], pin)}" for pin in pinned]
    made = f"{COMMAND} {repo} {name}" + (f" --task {task}" if task else "")
    header = f"# Generated by: {made}\n# From {source}\n"
    (locks / f"{name}.txt").write_text(header + "\n".join(lines) + "\n")
    for line in lines:
        echo(line)
    return lines
