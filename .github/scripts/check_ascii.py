#!/usr/bin/env python3
"""Fail if any tracked Python file contains a non-ASCII character.

Enforces a repo-wide ASCII-only policy for ``*.py`` source. Run locally with
``python .github/scripts/check_ascii.py``; CI runs the exact same script. The file
list comes from ``git ls-files``, so git-ignored files and submodule contents
(e.g. ``harbor``) are excluded automatically.

``tasks/assets`` is excluded EXPLICITLY: it is vendored upstream documentation
(the toolchain references, Triton and cuTile docs among them), which git ignores,
copied as-authored so it can be re-pinned verbatim. 8 of its 33 Python files are
Triton tutorials whose comments legitimately use non-ASCII, and they are not our
source. The policy is about code WE write; rewriting third-party examples to
satisfy it would defeat the point of vendoring them unmodified.

Exit status is 0 when everything is ASCII and 1 when any violation is found
(each is printed as ``path:line:col: ...`` so the report is clickable).
"""
from __future__ import annotations

import subprocess
import sys
import unicodedata

#: Vendored, as-authored upstream trees. Paths are matched by prefix.
EXCLUDED_PREFIXES = ("tasks/assets/",)


def tracked_python_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [p for p in out.split("\0")
            if p and not p.startswith(EXCLUDED_PREFIXES)]


def main() -> int:
    violations = 0
    for path in tracked_python_files():
        with open(path, "rb") as fh:
            raw = fh.read()
        if all(b < 0x80 for b in raw):
            continue  # pure ASCII, fast path
        text = raw.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for col, ch in enumerate(line, start=1):
                if ord(ch) > 0x7F:
                    violations += 1
                    try:
                        name = unicodedata.name(ch)
                    except ValueError:
                        name = "<unnamed>"
                    print(f"{path}:{lineno}:{col}: non-ASCII U+{ord(ch):04X} {name} ({ch!r})")

    if violations:
        print(f"\nFound {violations} non-ASCII character(s) in Python source.", file=sys.stderr)
        print(
            "All .py files must be ASCII-only. Replace with ASCII equivalents "
            "(e.g. em-dash -> --, arrow -> ->, ellipsis -> ...).",
            file=sys.stderr,
        )
        return 1
    print("OK: all tracked Python files are ASCII-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
