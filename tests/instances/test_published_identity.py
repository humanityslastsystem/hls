"""Scoring is unchanged: for every graded task, what decides how a submission is scored, read
through the functions the harness's commands use (``tests.instances.golden``), is byte for byte
what its golden file holds. The images and their identities, the baseline's start, B and the
calibration as the roles load them with their files' blob ids, the workload, the brief, the
plans of a run and a grading and the oracle answer keys are each a field of the file, and a failure names the task and every field that moved.

A change meant to move a value regenerates the files and shows the move in their diff:

    .venv/bin/python -m tests.instances.golden"""

from __future__ import annotations

import difflib
import json
from typing import Any

import pytest

from tests.instances import golden

#: How much of a value a failure prints.
SHOWN_CHARS = 160
#: How much of the harness's refusal to read a value a failure prints.
SHOWN_REFUSAL_CHARS = 600
#: How many moved fields a failure lists.
SHOWN_FIELDS = 40


def _shown(value: Any) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= SHOWN_CHARS else text[:SHOWN_CHARS] + "..."


def moved_fields(expected: Any, actual: Any, path: str = "") -> list[str]:
    """One line per field of *actual* that differs from *expected*, by its dotted path: a value
    that changed, a key or list item added or removed, or a value the harness now refuses to read,
    with the refusal."""
    if isinstance(actual, dict) and golden.UNREADABLE in actual:
        refusal = actual[golden.UNREADABLE]
        shown = refusal if len(refusal) <= SHOWN_REFUSAL_CHARS else refusal[:SHOWN_REFUSAL_CHARS] + "..."
        return [f"{path}: the harness now refuses to read it: {shown}"]
    if isinstance(expected, dict) and isinstance(actual, dict):
        lines = []
        for key in sorted(set(expected) | set(actual)):
            where = f"{path}.{key}" if path else key
            if key not in actual:
                lines.append(f"{where}: removed (golden {_shown(expected[key])})")
            elif key not in expected:
                lines.append(f"{where}: added ({_shown(actual[key])})")
            else:
                lines += moved_fields(expected[key], actual[key], where)
        return lines
    if isinstance(expected, list) and isinstance(actual, list):
        lines = []
        for i in range(max(len(expected), len(actual))):
            where = f"{path}[{i}]"
            if i >= len(actual):
                lines.append(f"{where}: removed (golden {_shown(expected[i])})")
            elif i >= len(expected):
                lines.append(f"{where}: added ({_shown(actual[i])})")
            else:
                lines += moved_fields(expected[i], actual[i], where)
        return lines
    if type(expected) is not type(actual) or expected != actual:
        return [f"{path}: golden {_shown(expected)}, now {_shown(actual)}"]
    return []


def _refusal(name: str, moved: list[str], brief_diff: list[str]) -> str:
    listed = "\n  ".join(moved[:SHOWN_FIELDS])
    more = f"\n  ... and {len(moved) - SHOWN_FIELDS} more" if len(moved) > SHOWN_FIELDS else ""
    brief = ("\nThe brief, against golden/" + name + ".brief.md:\n" + "\n".join(brief_diff[:SHOWN_FIELDS])
             if brief_diff else "")
    return (f"{name}: {len(moved)} field(s) moved from the golden file golden/{name}.json:\n  {listed}{more}{brief}\n"
            "If the change is meant to move them, regenerate with .venv/bin/python -m tests.instances.golden "
            "and review the diff")


def _check_file(name: str, text: str, brief: str | None = None) -> None:
    """The golden file *name* holds *text*, the values as they are now, byte for byte, and the
    golden brief beside it *brief* when one is given; a failure names each field that moved, and
    shows how a brief that moved differs."""
    path = golden.GOLDEN_DIR / name
    assert path.is_file(), f"no golden file {path.name}: generate it with .venv/bin/python -m tests.instances.golden"
    stem = name.removesuffix(".json")
    stored = path.read_text()
    brief_diff: list[str] = []
    if brief is not None:
        golden_brief = (golden.GOLDEN_DIR / f"{stem}.brief.md").read_text()
        brief_diff = list(difflib.unified_diff(golden_brief.splitlines(), brief.splitlines(),
                                               f"golden/{stem}.brief.md", "now", lineterm="", n=1))
    if stored == text and not brief_diff:
        return
    moved = moved_fields(json.loads(stored), json.loads(text))
    if not moved and brief_diff:
        moved = ["brief: the text differs from its golden file, whose sha256 the golden JSON holds"]
    assert not moved, _refusal(stem, moved, brief_diff)
    pytest.fail(f"{stem}: golden/{name} holds the same values in other bytes; regenerate it with "
                ".venv/bin/python -m tests.instances.golden")


def test_the_golden_files_are_every_graded_tasks_and_the_catalogs():
    names = {golden.CATALOG_FILE, *(f"{key}{suffix}" for key in golden.task_modules()
                                    for suffix in (".json", ".brief.md"))}
    found = {p.name for p in golden.GOLDEN_DIR.iterdir()}
    assert found == names, (f"golden/ holds {sorted(found - names)} that no graded task has and lacks "
                            f"{sorted(names - found)}; regenerate with .venv/bin/python -m tests.instances.golden")


def test_every_image_the_catalog_builds_keeps_its_recipe():
    _check_file(golden.CATALOG_FILE, golden.golden_text(golden.catalog_images()))


@pytest.mark.parametrize("key", sorted(golden.task_modules()))
def test_a_graded_task_is_scored_as_its_golden_file_says(key, tmp_path):
    values, brief = golden.collect(key, golden.task_modules()[key], tmp_path)
    _check_file(f"{key}.json", golden.golden_text(values), brief)
