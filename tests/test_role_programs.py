"""Every role program loads no module of ``hls.stages``. A role program is a module some code of
the harness starts with ``python -m``: what a stage runs as one of its roles (each wrapped by
``hls.substrate.role_exec``, itself started so) and the helpers a role or a seal starts beside
it. A role in its container needs the records and helpers it shares with its stage, which live
below the stages; the stage layer, which drives the substrate, is never its to load. The list is
read off the source, so a new role is held to this as soon as something starts it."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "hls"
#: Modules started with ``python -m`` that are no role: the runner of an allocation, which runs
#: the stages themselves.
NOT_ROLES = frozenset({"hls.substrate.runner"})
# A module started with ``python -m`` inside a text: a shell command line or a script.
_STARTED = re.compile(r"-m\s+(hls(?:\.\w+)+)")


def _docstrings(tree: ast.Module) -> set[int]:
    """The ids of the docstring nodes of *tree*'s module, classes and functions."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and \
                    isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def _constants(tree: ast.Module) -> dict[str, str]:
    """The module-level names *tree* binds to a text."""
    out = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, ast.AnnAssign) else []
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            out.update({t.id: value.value for t in targets if isinstance(t, ast.Name)})
    return out


def _argv_modules(path: Path, tree: ast.Module) -> set[str]:
    """The modules the command lines written as lists or tuples in *tree* start with ``-m``.
    Raise AssertionError for one whose module is neither a text nor a module-level name bound to
    one, which this reading could not follow."""
    constants, found = _constants(tree), set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        for flag, module in zip(node.elts, node.elts[1:]):
            if not (isinstance(flag, ast.Constant) and flag.value == "-m"):
                continue
            if isinstance(module, ast.Constant) and isinstance(module.value, str):
                name = module.value
            elif isinstance(module, ast.Name) and module.id in constants:
                name = constants[module.id]
            else:
                raise AssertionError(f"{path}:{node.lineno} starts a module this test cannot name; bind it to "
                                     "a module-level text")
            if name.startswith("hls."):
                found.add(name)
    return found


def _text_modules(tree: ast.Module) -> set[str]:
    """The modules the texts of *tree* other than its docstrings start with ``-m``, f-strings'
    literal parts among them."""
    skip = _docstrings(tree)
    return {m for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
            for m in _STARTED.findall(node.value)}


def started_modules() -> set[str]:
    """Every ``hls`` module the package's code and scripts start with ``python -m``."""
    found: set[str] = set()
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(), str(path))
            found |= _argv_modules(path, tree) | _text_modules(tree)
        else:
            found.update(_STARTED.findall(path.read_text(errors="replace")))
    return found


ROLE_PROGRAMS = sorted(started_modules() - NOT_ROLES)


def test_the_reading_finds_every_kind_of_start():
    # One of each kind, so a reading that missed a kind fails here: a stage's argv list, a module
    # named by a constant, the seal script's command line, and the batch script's f-string, which
    # starts the runner.
    assert {"hls.drive.role", "hls.verify.calibration_roles", "hls.egress.proxy"} <= set(ROLE_PROGRAMS)
    assert NOT_ROLES <= started_modules()


@pytest.mark.parametrize("module", ROLE_PROGRAMS)
def test_a_role_program_loads_no_stage(module):
    code = ("import importlib, json, sys\n"
            f"importlib.import_module({module!r})\n"
            "stages = (m for m in sys.modules if m == 'hls.stages' or m.startswith('hls.stages.'))\n"
            "print(json.dumps(sorted(stages)))")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env,
                          timeout=300)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == []
