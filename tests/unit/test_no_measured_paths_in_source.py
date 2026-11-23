"""No module of a task and no module of the hls package names where a measured value lies.

A baseline credit, a calibration and a dev split's reference answers are outputs of measurements.
The one module that knows where they are kept is ``hls.measured``: the store's measured area and
the task's shipped defaults. Every other module asks it, so no module builds a path to one of
those files on its own, from its own ``__file__`` or from a literal."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
#: The measured files' names.
MEASURED_NAMES = ("baseline.json", "calibration.json", "dev_reference.json")
#: The one module allowed to name them.
ALLOWED = REPO / "src" / "hls" / "measured.py"


def _docstrings(tree: ast.AST) -> set[int]:
    """The ids of the docstring nodes of *tree*'s module, classes and functions."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def _names_a_measured_file(text: str) -> bool:
    """Whether the string *text* is a measured file's name or a path ending in one."""
    return any(text == name or text.endswith(f"/{name}") for name in MEASURED_NAMES)


def measured_paths(source: str) -> list[str]:
    """Each place the Python *source* names a measured file, as ``<line>: <text>``: a string
    constant that is a measured file's name or a path ending in one, a docstring excepted."""
    tree = ast.parse(source)
    docstrings = _docstrings(tree)
    return [f"{node.lineno}: {node.value!r}" for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
            and _names_a_measured_file(node.value)]


def _modules() -> list[Path]:
    """Every Python module of the tasks and of the hls package but ``hls.measured``."""
    found = [*sorted((REPO / "tasks").rglob("*.py")), *sorted((REPO / "src" / "hls").rglob("*.py"))]
    return [p for p in found if p != ALLOWED and "__pycache__" not in p.parts]


@pytest.mark.parametrize("source", [
    'from pathlib import Path\nX = (Path(__file__).parent / "dev_reference.json").read_text()\n',
    'from pathlib import Path\nX = Path(__file__).with_name("baseline.json")\n',
    'X = "tasks/multitenant_slo/calibration.json"\n',
    'def f(d):\n    return d / "calibration.json"\n',
], ids=["beside-the-module", "with-name", "task-path", "joined"])
def test_the_detector_finds_a_measured_files_path(source):
    assert measured_paths(source) != []


@pytest.mark.parametrize("source", [
    '"""Reads ``baseline.json`` from the store."""\n',
    'def f():\n    """The calibration.json a role is handed."""\n',
    'X = f"no {NAME} was given"\n',
    'X = "removed_requests.json"\n',
], ids=["module-docstring", "function-docstring", "named-by-constant", "other-file"])
def test_the_detector_leaves_docstrings_and_other_files_alone(source):
    assert measured_paths(source) == []


def test_no_module_but_hls_measured_names_a_measured_file():
    found = {str(p.relative_to(REPO)): hits for p in _modules() if (hits := measured_paths(p.read_text()))}
    assert found == {}


def test_hls_measured_is_where_the_names_are_defined():
    # The detector would find them there: the one module that names them is the one excepted.
    assert measured_paths(ALLOWED.read_text())
