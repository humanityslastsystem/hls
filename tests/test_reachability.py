"""Every module under src/hls is reachable from what starts hls code, and every module under
tasks/ from a task's module or the server an image of its task runs.

The graph is static and generous. An import statement anywhere in a module (top level, in a
function, under TYPE_CHECKING) is an edge; so is a string constant equal to a module's name (a
role program started with ``python -m``, a constant such as ``ROLE_MODULE``, the package's lazy
re-exports), ``-m <module>`` inside a string, and a ``src/hls/<path>.py`` inside a string; a
docstring is not. Importing a module imports its parent packages.

What starts hls code: the ``hls`` command (``hls.cli``), the runner the batch script starts, the
stage registry, the package root task modules import, the programs the build sandbox runs by
path, the egress proxy the seal script starts, each catalog task's module, and the scripts under
``scripts/``."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
_DASH_M = re.compile(r"-m\s+((?:hls|tasks)(?:\.\w+)+)")
_PATH = re.compile(r"src/(hls(?:/\w+)+)\.py")


def module_files(root: Path, base: Path) -> dict[str, Path]:
    """Every module under *root* by its dotted name relative to *base*, task test packages (they
    are tests), the vendored, untracked ``tasks/assets`` and a task's ``references`` folder (files
    a build uploads into the agent's sandbox, not code the harness loads) left out."""
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(base).with_suffix("").parts
        if "tests" in parts[1:] or parts[:2] == ("tasks", "assets"):
            continue
        if parts[0] == "tasks" and parts[2:3] == ("references",):
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts)] = path
    return out


def edges(name: str, path: Path, known: set[str]) -> set[str]:
    """The modules of *known* that module *name* (at *path*) reaches by one edge."""
    tree = ast.parse(path.read_text(), str(path))
    docstrings = {id(n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    found: set[str] = set()

    def add(target: str) -> None:
        if target in known:
            found.add(target)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
                parent = parent[:len(parent) - node.level + 1]
                base = ".".join([*parent, *([node.module] if node.module else [])])
            for alias in node.names:
                full = f"{base}.{alias.name}"
                add(full if full in known else base)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            text = node.value
            if text in known:
                found.add(text)
            for module in _DASH_M.findall(text):
                add(module)
            for module in _PATH.findall(text):
                add(module.replace("/", "."))
    found.discard(name)
    return found


def entries(known: set[str]) -> set[str]:
    """What starts hls code (see the module docstring)."""
    from hls.agent.exec_protocol import SANDBOX_FILES
    from hls.recipes import catalog

    roots = {"hls", "hls.cli", "hls.substrate.runner", "hls.stages.registry"}
    roots |= {f"hls.agent.{Path(f).stem}" for f in SANDBOX_FILES}
    roots |= {catalog.task_module(t) for t in catalog.task_keys()}
    # A module a task's image copies in runs there, as its server, and no task module imports it.
    roots |= {f"tasks.{t}." + source.removesuffix(".py").replace("/", ".") for t in catalog.task_keys()
              for decl in catalog.task_images(t).decls for source in getattr(decl, "files", {}).values()
              if source.endswith(".py")}
    for script in (SRC / "hls").rglob("*.sh"):
        roots |= {m for m in _DASH_M.findall(script.read_text()) if m in known}
    for script in (REPO / "scripts").glob("*.py"):
        roots |= edges(f"scripts.{script.stem}", script, known)
    return roots


def reachable(roots: set[str], graph: dict[str, set[str]]) -> set[str]:
    """Every module reached from *roots* along *graph*, parent packages included."""
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        parts = module.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
        todo += graph.get(module, ())
    return seen


def _repo_graph() -> tuple[dict[str, Path], set[str]]:
    files = {**module_files(SRC / "hls", SRC), **module_files(REPO / "tasks", REPO)}
    known = set(files)
    graph = {name: edges(name, path, known) for name, path in files.items()}
    return files, reachable(entries(known), graph)


def test_every_module_under_src_hls_is_reachable_from_what_starts_hls_code():
    files, seen = _repo_graph()
    orphans = sorted(str(files[m].relative_to(REPO)) for m in files if m.startswith("hls") and m not in seen)
    assert orphans == [], f"unreachable from what starts hls code: {orphans}"


def test_every_task_module_is_reachable_from_a_catalog_tasks_module():
    files, seen = _repo_graph()
    orphans = sorted(str(files[m].relative_to(REPO)) for m in files if m.startswith("tasks.") and m not in seen)
    assert orphans == [], f"no command loads these task modules: {orphans}"


def test_the_check_finds_what_it_must(tmp_path):
    pkg = tmp_path / "hls"
    pkg.mkdir()
    for rel, text in {
        "__init__.py": "",
        "cli.py": ('"""Run as python -m hls.e; a docstring is no edge."""\nimport hls.a\n\n\ndef f():\n'
                   "    from hls import b\n\n\nCMD = 'python -m hls.c'\nP = 'x/src/hls/f.py'\n"),
        "a.py": "from typing import TYPE_CHECKING\n\nfrom . import g\n\nif TYPE_CHECKING:\n    from hls.d import X\n",
        "b.py": "", "c.py": "", "d.py": "X = 1\n", "f.py": "", "g.py": "",
        "orphan.py": '"""Nothing imports this module."""\n',
        "e.py": "",
    }.items():
        (pkg / rel).write_text(text)
    files = module_files(pkg, tmp_path)
    graph = {name: edges(name, path, set(files)) for name, path in files.items()}
    assert sorted(set(files) - reachable({"hls.cli"}, graph)) == ["hls.e", "hls.orphan"]
