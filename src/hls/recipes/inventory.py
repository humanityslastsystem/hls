"""What an image holds: the inventory every cached image carries as ``inventory.json``.

The images stage runs ``inventory.sh`` inside each image it builds, bound read-only
at ``INVENTORY_TARGET``, and parses its output here. The inventory is the record three checks
read, each a list of problems that is empty when the image is right:
- ``denied_in``: no base or unsealed image may hold a package of the harness deny list,
  compared by PEP 503-normalized name, because the index refuses those names to the
  agent and a base that already holds one hands it over anyway;
- ``version_mismatches``: a base holds torch, torchaudio and torchvision at exactly the
  versions its constraints name, so no layer replaced them;
- ``missing_tools``, ``missing_packages`` and ``missing_apt``: the images stage refuses an agent
  cap it builds whose inventory lacks something the brief promises.
  ``BRIEF_TOOLS``, ``BRIEF_PACKAGES`` and ``BRIEF_APT`` are what the brief's toolchain section
  promises today.

A check refuses one string where it expects names, and the deny and constraint checks refuse an
empty list, since either would pass any image.

A probe that fails is an error record, never a silent gap: a command the script needs, a
``dpkg-query`` or interpreter that fails, times out or is killed, a ``sys.path`` directory
that cannot be listed, a distribution whose metadata cannot be read or has no name. Such an
inventory is refused, and the error says whether a second run may succeed.

What the inventory trusts. The script runs on the image's own ``sh``, ``sed``, ``dpkg-query``
and python, and a ``.pth`` file runs code when the interpreter starts even under ``-I``, so the
inventory is evidence about images built from the harness's own recipes, not about an image a
contestant shaped. It reads distribution metadata, and
``denied_in`` compares the distribution names it records."""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import Path

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict

from hls.egress.policy import pep503_normalize

#: The script, beside this module.
SCRIPT = Path(__file__).with_name("inventory.sh")
#: Where the images stage binds the script inside the image it inventories.
INVENTORY_TARGET = "/opt/hls-inventory.sh"
#: Commands the brief's toolchain section says are installed.
BRIEF_TOOLS = ("gcc", "g++", "clang", "make", "cmake", "ninja", "pkg-config", "rustc", "cargo", "nvcc",
               "ptxas", "nvdisasm", "cuobjdump", "nsys", "ncu", "compute-sanitizer", "cuda-gdb")
#: Python distributions the brief says are installed, by PEP 503 name. The brief promises CUDA
#: Python as its modules ``cuda.bindings`` and ``cuda.core``, so the distributions that provide
#: them are listed beside the ``cuda-python`` metapackage, which does not ship either.
BRIEF_PACKAGES = ("torch", "transformers", "accelerate", "datasets", "fastapi", "uvicorn", "pydantic",
                  "triton", "cuda-tile", "nvidia-cutlass-dsl", "cuda-python", "cuda-bindings", "cuda-core")
#: Ubuntu packages the brief says are installed.
BRIEF_APT = ("python3-dev",)

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CONSTRAINT = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)")


class InventoryError(ValueError):
    """The script's output is not an inventory, or, parsed strictly, not a complete one."""

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        #: True only when every failed probe behind the refusal timed out or was killed, so a
        #: second run may succeed. A malformed output, and any other failure, repeats.
        self.transient = transient


class Inventory(BaseModel):
    """One image's contents, as ``inventory.sh`` saw them from inside it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Installed dpkg packages, name to version.
    apt: dict[str, str]
    #: Python distributions the image's interpreter imports, PEP 503-normalized name to version.
    pip: dict[str, str]
    #: Each tool the script looks for, to its version line; None when the tool is absent, exits
    #: nonzero or prints nothing for its version.
    tools: dict[str, str | None]
    #: Names of the variables the image's environment sets, sorted; never their values.
    env: list[str]


def _fail(number: int, line: str, why: str) -> InventoryError:
    return InventoryError(f"inventory line {number} ({line[:80]!r}): {why}")


def parse_inventory(text: str) -> Inventory:
    """The inventory ``inventory.sh`` printed. Every record must be well formed, and a name may
    appear once per kind; anything else raises InventoryError naming the line.

    The script prints an error record for each probe that failed: one that timed out or was
    killed (exit 124 or 137), which a second run may get past, or one that fails the same way
    every run. An inventory with one is refused too, so an inventory cut short never passes a
    check; the error says whether a second run may succeed."""
    apt: dict[str, str] = {}
    pip: dict[str, str] = {}
    tools: dict[str, str | None] = {}
    env: set[str] = set()
    # Each failed probe: its line, whether a second run may get past it, and what failed.
    failures: list[tuple[int, bool, str]] = []
    # Records end at a newline only: str.splitlines would also end one at a form feed or a
    # Unicode line separator inside a field.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line:
            continue
        fields = line.split("\t")
        kind = fields[0]
        if kind in ("apt", "pip", "tool"):
            if len(fields) != 3:
                raise _fail(number, line, f"a {kind} record has 3 fields, not {len(fields)}")
            _, name, version = fields
            if not name or (kind != "tool" and not version):
                raise _fail(number, line, f"a {kind} record needs a name and a version")
            seen = {"apt": apt, "pip": pip, "tool": tools}[kind]
            key = pep503_normalize(name) if kind == "pip" else name
            if key in seen:
                raise _fail(number, line, f"{kind} {key} appears twice")
            seen[key] = version if (version or kind != "tool") else None
        elif kind == "env":
            if len(fields) != 2 or not _ENV_NAME.fullmatch(fields[1]):
                raise _fail(number, line, "an env record is the name of one variable")
            if fields[1] in env:
                raise _fail(number, line, f"env {fields[1]} appears twice")
            env.add(fields[1])
        elif kind == "error":
            if len(fields) != 3:
                raise _fail(number, line, f"an error record has 3 fields, not {len(fields)}")
            if fields[1] not in ("deterministic", "transient"):
                raise _fail(number, line, "an error record says deterministic or transient")
            if not fields[2]:
                raise _fail(number, line, "an error record says what failed")
            failures.append((number, fields[1] == "transient", fields[2]))
        else:
            raise _fail(number, line, f"unknown record kind {kind!r}")
    if failures:
        raise InventoryError("the script reported failed probes: "
                             + "; ".join(f"line {number}: {what}" for number, _, what in failures),
                             transient=all(transient for _, transient, _ in failures))
    return Inventory(apt=dict(sorted(apt.items())), pip=dict(sorted(pip.items())),
                     tools=dict(sorted(tools.items())), env=sorted(env))


def _names(given: Collection[str], what: str, *, may_be_empty: bool) -> Collection[str]:
    """*given*, refused when it is one string (a check would read its characters as names) or,
    unless *may_be_empty*, empty (a check would pass whatever the image holds)."""
    if isinstance(given, str):
        raise TypeError(f"{what} is a collection of names, not the string {given!r}")
    if not given and not may_be_empty:
        raise ValueError(f"{what} is empty, so the check would pass any image")
    return given


def denied_in(inventory: Inventory, denied: Collection[str]) -> list[str]:
    """The image's Python distributions that *denied* names, by PEP 503 name, sorted. An empty
    *denied* raises ValueError."""
    names = {pep503_normalize(d) for d in _names(denied, "the deny list", may_be_empty=False)}
    return sorted(name for name in inventory.pip if name in names)


def missing_tools(inventory: Inventory, promised: Collection[str]) -> list[str]:
    """The *promised* tools the image lacks, in promise order. A tool that is absent, exits
    nonzero or prints nothing for its version counts as missing, and so does one the script did
    not look for: a promise can only be kept by a tool the inventory saw run."""
    promised = _names(promised, "the promise list", may_be_empty=True)
    return [t for t in promised if inventory.tools.get(t) is None]


def missing_packages(inventory: Inventory, promised: Collection[str]) -> list[str]:
    """The *promised* Python distributions the image lacks, in promise order."""
    promised = _names(promised, "the promise list", may_be_empty=True)
    return [p for p in promised if pep503_normalize(p) not in inventory.pip]


def missing_apt(inventory: Inventory, promised: Collection[str]) -> list[str]:
    """The *promised* Ubuntu packages the image lacks, in promise order."""
    promised = _names(promised, "the promise list", may_be_empty=True)
    return [p for p in promised if p not in inventory.apt]


def version_mismatches(inventory: Inventory, constraints: Collection[str]) -> list[str]:
    """Each ``name==version`` of *constraints* that the image does not hold exactly: absent, or
    at another PEP 440 version (a local label such as ``+cu130`` counts). A constraint of any
    other form, or no constraint at all, raises ValueError."""
    problems = []
    for spec in _names(constraints, "the constraint list", may_be_empty=False):
        m = _CONSTRAINT.fullmatch(spec.strip())
        if m is None:
            raise ValueError(f"constraint {spec!r} is not name==version")
        name, want = pep503_normalize(m.group(1)), m.group(2)
        have = inventory.pip.get(name)
        if have is None:
            problems.append(f"{name} is not installed; the base holds it at {want}")
            continue
        try:
            same = Version(have) == Version(want)
        except InvalidVersion:
            same = have == want
        if not same:
            problems.append(f"{name} is {have}; the base holds it at {want}")
    return problems
