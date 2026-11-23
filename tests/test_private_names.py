"""No module of hls uses another hls module's private name, imported or read off the module: what
a module keeps to itself can change with nothing outside it relying on it, and a name another
module needs is made public in its owner."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import hls

PACKAGE = Path(hls.__file__).resolve().parent


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def private_uses(path: Path) -> list[str]:
    """Each private name of another hls module the source at *path* imports
    (``from hls.m import _x``) or reads off an hls module it imported (``m._x``)."""
    me, tree = _module_name(path), ast.parse(path.read_text())
    # What each local name of an imported hls module or name stands for, fully qualified.
    aliases: dict[str, str] = {}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "hls":
            for alias in node.names:
                if _private(alias.name) and node.module != me:
                    found.append(f"{me}:{node.lineno} imports {node.module}.{alias.name}")
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "hls":
                    aliases[alias.asname or "hls"] = alias.name if alias.asname else "hls"
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and _private(node.attr)):
            continue
        base, chain = node.value, []
        while isinstance(base, ast.Attribute):
            chain.append(base.attr)
            base = base.value
        if isinstance(base, ast.Name) and base.id in aliases:
            owner = ".".join([aliases[base.id], *reversed(chain)])
            if owner != me:
                found.append(f"{me}:{node.lineno} reads {owner}.{node.attr}")
    return found


def test_no_module_uses_another_modules_private_name():
    used = [use for path in sorted(PACKAGE.rglob("*.py")) for use in private_uses(path)]
    assert used == []


def test_the_reading_finds_each_form_of_a_private_use(tmp_path, monkeypatch):
    """Each form the guard reads is found: an import of a private name, and a private name read
    off a module imported whole, under its own name, an alias or its package."""
    sources = {
        "imported.py": "from hls.calibrate import _campaign\n",
        "aliased.py": "import hls.calibrate as calibrate\ncalibrate._campaign()\n",
        "whole.py": "import hls.calibrate\nhls.calibrate._campaign()\n",
        "package.py": "from hls import calibrate\ncalibrate._campaign()\n",
        "public.py": "import hls.calibrate as calibrate\ncalibrate.member_engines()\ncalibrate.__name__\n",
    }
    monkeypatch.setattr(sys.modules[__name__], "PACKAGE", tmp_path)
    for name, text in sources.items():
        (tmp_path / name).write_text(text)
    found = {name: private_uses(tmp_path / name) for name in sources}
    assert all(len(found[name]) == 1 for name in ("imported.py", "aliased.py", "whole.py", "package.py")), found
    assert found["public.py"] == []
