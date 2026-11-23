"""No module of hls outside a substrate's own packages and the factory imports from a substrate:
everything else takes what it needs from ``hls.substrate.factory``, so a second substrate needs
only its own packages and its entry there."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
#: The substrates' own packages, which only themselves and the factory import.
SUBSTRATE_PACKAGES = ("hls.substrate.slurm", "hls.substrate.apptainer", "hls.substrate.local",
                      "hls.substrate.docker")
#: The module besides the packages' own that may import from them.
FACTORY = "hls.substrate.factory"
#: Modules of those packages that modules outside them may still import, each with its reason.
#: Deleting an entry makes the test refuse its importers.
STILL_IMPORTED = {
    "hls.substrate.apptainer.egress": "the stages read the network seal's evidence, which has no interface yet",
    "hls.substrate.apptainer.images": "the stages build images with apptainer, which has no interface yet",
}


def _within(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def _module(path: Path, src: Path) -> str:
    parts = path.relative_to(src).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imported(path: Path, src: Path) -> list[tuple[int, str]]:
    """Every name *path* imports, wherever in the file, by line, as fully as it is written: each
    module of an ``import``, each ``<module>.<name>`` of a ``from ... import``, and the literal
    argument of an ``importlib.import_module`` or ``__import__`` call. A relative import is
    resolved against the file's package."""
    module = _module(path, src)
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            anchor = package[: len(package) - node.level + 1] if node.level else []
            base = ".".join([*anchor, *([node.module] if node.module else [])])
            found += [(node.lineno, f"{base}.{alias.name}") for alias in node.names]
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and (getattr(node.func, "attr", None) == "import_module" or getattr(node.func, "id", None) == "__import__")):
            found.append((node.lineno, node.args[0].value))
    return found


def outside_imports(src: Path) -> dict[str, list[str]]:
    """Each name of the substrates' packages that a module under *src* imports from outside them
    and the factory, with where, as ``path:line``."""
    seen: dict[str, list[str]] = {}
    for path in sorted((src / "hls").rglob("*.py")):
        module = _module(path, src)
        if module == FACTORY or any(_within(module, p) for p in SUBSTRATE_PACKAGES):
            continue
        for line, name in _imported(path, src):
            if any(_within(name, p) for p in SUBSTRATE_PACKAGES):
                seen.setdefault(name, []).append(f"{path.relative_to(src)}:{line}")
    return seen


def test_no_module_outside_a_substrate_imports_from_it():
    refused = {name: where for name, where in outside_imports(SRC).items()
               if not any(_within(name, kept) for kept in STILL_IMPORTED)}
    assert not refused, refused


def test_each_module_still_imported_from_outside_is():
    # An entry no module needs any more is deleted, so the list only shrinks.
    outside = outside_imports(SRC)
    unused = [kept for kept in STILL_IMPORTED if not any(_within(name, kept) for name in outside)]
    assert not unused, unused


def test_the_check_finds_every_form_of_import(tmp_path):
    # The check itself, on a module outside the substrates that imports from them in each form
    # it reads: an import, a from-import of a name and of a module, a relative import inside a
    # function, and importlib.
    pkg = tmp_path / "hls" / "pipeline"
    pkg.mkdir(parents=True)
    (pkg / "x.py").write_text("import hls.substrate.slurm.scheduler\n"
                              "from hls.substrate.slurm.batch import batch_script\n"
                              "from hls.substrate import slurm\n"
                              "def f():\n    from ..substrate.apptainer.argv import ApptainerRuntime\n"
                              "import importlib\nimportlib.import_module('hls.substrate.slurm.launcher')\n"
                              "from hls.substrate.apptainer.egress import client_env\n")
    factory = tmp_path / "hls" / "substrate" / "factory.py"
    factory.parent.mkdir(parents=True)
    factory.write_text("from hls.substrate.slurm.substrate import slurm\n")
    found = outside_imports(tmp_path)
    assert found == {
        "hls.substrate.slurm.scheduler": ["hls/pipeline/x.py:1"],
        "hls.substrate.slurm.batch.batch_script": ["hls/pipeline/x.py:2"],
        "hls.substrate.slurm": ["hls/pipeline/x.py:3"],
        "hls.substrate.apptainer.argv.ApptainerRuntime": ["hls/pipeline/x.py:5"],
        "hls.substrate.slurm.launcher": ["hls/pipeline/x.py:7"],
        "hls.substrate.apptainer.egress.client_env": ["hls/pipeline/x.py:8"],
    }
