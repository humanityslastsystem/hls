"""Declared-dependency pre-scan: fast-fail before any network.

Parses a submitted ``pyproject.toml`` and checks each declared dependency's
distribution NAME against the package rule. This is a UX layer ON TOP of proxy
enforcement -- the proxy remains the actual wall (it catches transitive deps and
build-hook fetches a static scan cannot), but the pre-scan turns an obvious
policy violation into a clear "package X is not allowed" message before the build
sandbox is even launched.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from hls.egress.policy import NetworkPolicy
from hls.spec.submission import PERMISSION_ERRNOS, SubmissionError


def declared_dependencies(source_dir: Path) -> list[str]:
    """Distribution names declared in ``source_dir/pyproject.toml``, in order.

    Covers ``[project].dependencies`` and every ``[project.optional-dependencies]``
    group -- NOT ``[build-system].requires`` or PEP 735 ``[dependency-groups]`` (this is
    a UX pre-scan, not the wall: the egress proxy still gates transitive + build-hook
    fetches). Names are returned in declared order with duplicates removed (first
    occurrence wins). A missing pyproject or a missing ``[project]`` table yields an
    empty list.

    ``pyproject.toml`` is contestant-controlled, so EVERY malformed-but-TOML-valid shape
    is a contestant fault (``SubmissionError`` -> invalid_artifact), NEVER an uncaught
    harness crash. We validate the PEP 621 structure we read rather than trusting it:
    ``[project]`` must be a table; ``dependencies`` must be a list; ``optional-dependencies``
    must be a table of lists; and each dependency entry must be a string. An entry that is a
    string but not a valid PEP 508 requirement is silently skipped (the proxy is the real
    wall), but a non-string entry is a structural fault and raises. Anything outside this
    scanned scope is left untouched. Any way the file fails to parse is a ``SubmissionError``
    too: bytes that are not UTF-8, text that is not TOML, and TOML the parser cannot hold, nested
    deeper than it recurses or with a number Python will not convert. So is a file whose
    permissions keep the harness out (``PERMISSION_ERRNOS``); any other error reading it is the
    node's, raised as it is for the caller to report.
    """
    pyproject = source_dir / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError as exc:
        if exc.errno not in PERMISSION_ERRNOS:
            raise
        raise SubmissionError(f"pyproject.toml cannot be read: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise SubmissionError("pyproject.toml is not UTF-8") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SubmissionError(f"pyproject.toml is not valid TOML: {exc}") from exc
    except (ValueError, RecursionError, MemoryError) as exc:
        # Valid TOML can nest deeper than the parser recurses, or hold more than Python converts
        # or holds; only the submission chose the bytes.
        why = type(exc).__name__ + (f": {exc}" if str(exc) else "")
        raise SubmissionError(f"pyproject.toml cannot be parsed: {why}") from exc

    project = data.get("project", {})
    if not isinstance(project, dict):
        raise SubmissionError(
            f"pyproject.toml [project] must be a table, got {_typename(project)}.")

    specs: list[object] = []
    deps = project.get("dependencies", [])
    if not isinstance(deps, list):
        raise SubmissionError(
            "pyproject.toml [project].dependencies must be a list (array of PEP 508 "
            f"requirement strings), got {_typename(deps)}.")
    specs.extend(deps)

    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise SubmissionError(
            "pyproject.toml [project.optional-dependencies] must be a table of "
            f"requirement-string lists, got {_typename(optional)}.")
    for name, group in optional.items():
        if not isinstance(group, list):
            raise SubmissionError(
                f"pyproject.toml [project.optional-dependencies].{name} must be a list "
                f"(array of PEP 508 requirement strings), got {_typename(group)}.")
        specs.extend(group)

    names: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, str):
            raise SubmissionError(
                "pyproject.toml dependency entries must be PEP 508 requirement strings, "
                f"got a {_typename(spec)}.")
        try:
            name = Requirement(spec).name
        except InvalidRequirement:
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _typename(value: object) -> str:
    """The contestant-facing name for a TOML value's Python type, used in the
    structural-validation error messages above (e.g. ``str``, ``list``, ``int``)."""
    return type(value).__name__


def prescan(source_dir: Path, policy: NetworkPolicy) -> list[str]:
    """Declared dependency names the package rule does NOT permit (empty == clean).

    ``policy.packages is None`` means no package policy applies, so nothing is
    flagged. Names are matched PEP 503-normalized but reported verbatim as
    declared.
    """
    packages = policy.packages
    if packages is None:
        return []
    return [
        name
        for name in declared_dependencies(source_dir)
        if not packages.permits(name, normalize=True)
    ]
