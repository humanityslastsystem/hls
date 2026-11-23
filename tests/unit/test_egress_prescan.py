"""Unit tests for the pyproject declared-dependency pre-scan (fast-fail UX).

The pre-scan parses a submitted ``pyproject.toml`` and checks its declared
dependency names against the package rule before any network. It is a UX layer
on top of the proxy (which remains the actual wall); these tests cover name
extraction (incl. optional-dependencies) and the denied-name flagging.
"""
from __future__ import annotations

import errno
import os

import pytest

from hls.egress.policy import AccessList, NetworkPolicy
from hls.egress.prescan import declared_dependencies, prescan
from hls.spec.submission import SubmissionError


def test_declared_dependencies_parses_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\n'
        'dependencies=["numpy>=1.0", "torch==2.1", "fastapi"]\n'
    )
    assert declared_dependencies(tmp_path) == ["numpy", "torch", "fastapi"]


def test_declared_dependencies_includes_optionals_and_dedups(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\n'
        'dependencies=["numpy>=1.0", "fastapi"]\n'
        '[project.optional-dependencies]\n'
        'dev=["pytest", "numpy"]\n'        # numpy repeats -> deduped
        'extra=["uvicorn[standard]"]\n'
    )
    assert declared_dependencies(tmp_path) == ["numpy", "fastapi", "pytest", "uvicorn"]


def test_declared_dependencies_missing_pyproject_is_empty(tmp_path):
    assert declared_dependencies(tmp_path) == []


def test_declared_dependencies_no_project_table_is_empty(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[build-system]\nrequires=["setuptools"]\n')
    assert declared_dependencies(tmp_path) == []


def test_prescan_flags_denied(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\ndependencies=["torch"]\n'
    )
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="deny", entries=["torch"]))
    assert prescan(tmp_path, pol) == ["torch"]


def test_prescan_clean_when_all_allowed(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\ndependencies=["numpy", "fastapi"]\n'
    )
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="deny", entries=["torch"]))
    assert prescan(tmp_path, pol) == []


def test_prescan_allowlist_flags_unlisted(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\ndependencies=["numpy", "torch"]\n'
    )
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="allow", entries=["numpy"]))
    assert prescan(tmp_path, pol) == ["torch"]


def test_prescan_normalizes_names(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\ndependencies=["My_Pkg.X"]\n'
    )
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="deny", entries=["my-pkg-x"]))
    # declared name kept verbatim in the report, but matched normalized
    assert prescan(tmp_path, pol) == ["My_Pkg.X"]


def test_prescan_no_package_policy_flags_nothing(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\ndependencies=["torch", "tensorflow"]\n'
    )
    # packages is None -> no package policy -> nothing flagged
    assert prescan(tmp_path, NetworkPolicy()) == []


def test_prescan_malformed_pyproject_is_submission_error(tmp_path):
    """A contestant-controlled pyproject.toml that is invalid TOML is a contestant fault
    (SubmissionError -> invalid_artifact), NOT an uncaught harness crash."""
    (tmp_path / "pyproject.toml").write_text("this is = = not valid toml [[[")
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="deny", entries=["torch"]))
    with pytest.raises(SubmissionError, match="not valid TOML"):
        prescan(tmp_path, pol)


# --- Robustness: every malformed-but-TOML-valid shape is a SubmissionError, ---
# --- never an uncaught harness crash. pyproject.toml is contestant-controlled. ---


def test_declared_dependencies_project_not_a_table_is_submission_error(tmp_path):
    """[project] present but not a table (here a string) must not let project.get(...)
    raise AttributeError -- it is a contestant fault, so SubmissionError."""
    (tmp_path / "pyproject.toml").write_text('project = "x"\n')
    with pytest.raises(SubmissionError, match=r"\[project\] must be a table"):
        declared_dependencies(tmp_path)


def test_declared_dependencies_dependencies_not_a_list_is_submission_error(tmp_path):
    """dependencies as a string must NOT be char-iterated -- it is a contestant fault."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\ndependencies="numpy"\n'
    )
    with pytest.raises(SubmissionError, match="dependencies must be a list"):
        declared_dependencies(tmp_path)


def test_declared_dependencies_string_dependencies_not_char_scanned(tmp_path):
    """A string `dependencies` must NOT silently scan into single-char
    package names like ['n','u','m','p','y']; it raises instead of mis-scanning."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\ndependencies="numpy"\n'
    )
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="allow", entries=["numpy"]))
    with pytest.raises(SubmissionError):
        # would otherwise flag ['n','u','m','p','y'] as unlisted denials
        prescan(tmp_path, pol)


def test_declared_dependencies_optionals_not_a_table_is_submission_error(tmp_path):
    """optional-dependencies present but not a table is a contestant fault."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\n'
        'optional-dependencies=["numpy"]\n'  # list, not a table of groups
    )
    with pytest.raises(SubmissionError, match="optional-dependencies"):
        declared_dependencies(tmp_path)


def test_declared_dependencies_optional_group_not_a_list_is_submission_error(tmp_path):
    """An optional-dependencies group whose value is not a list is a contestant fault."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\n'
        '[project.optional-dependencies]\n'
        'dev="numpy"\n'  # group value must be a list, not a string
    )
    with pytest.raises(SubmissionError, match="optional-dependencies"):
        declared_dependencies(tmp_path)


def test_declared_dependencies_non_string_entry_is_submission_error(tmp_path):
    """A dependency entry that is not a string (here a number) is a contestant fault --
    Requirement() only accepts strings, so this must be a SubmissionError not a crash."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\ndependencies=[123]\n'
    )
    with pytest.raises(SubmissionError, match="requirement strings"):
        declared_dependencies(tmp_path)


def test_declared_dependencies_nested_table_entry_is_submission_error(tmp_path):
    """A dependency entry that is a nested table (inline table) is a contestant fault."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\n'
        'dependencies=[{name="numpy"}]\n'
    )
    with pytest.raises(SubmissionError, match="requirement strings"):
        declared_dependencies(tmp_path)


def test_declared_dependencies_valid_inputs_unchanged(tmp_path):
    """The robustness checks change nothing for valid inputs: declared order,
    first-occurrence dedup, optional-dependencies included, InvalidRequirement skipped."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\n'
        'dependencies=["numpy>=1.0", "fastapi", "!!not a requirement!!"]\n'
        '[project.optional-dependencies]\n'
        'dev=["pytest", "numpy"]\n'        # numpy repeats -> deduped
        'extra=["uvicorn[standard]"]\n'
    )
    assert declared_dependencies(tmp_path) == ["numpy", "fastapi", "pytest", "uvicorn"]


_DEEP = "RecursionError: maximum recursion depth exceeded"


@pytest.mark.parametrize("value, words", [
    ("[" * 1000 + "]" * 1000, _DEEP),
    ("{a = " * 1000 + "1" + "}" * 1000, _DEEP),
    ("1" * 5000, "ValueError: Exceeds the limit (4300 digits)"),
], ids=["arrays-1000-deep", "tables-1000-deep", "an-integer-of-5000-digits"])
def test_a_pyproject_the_parser_cannot_hold_is_a_submission_error(tmp_path, value, words):
    """Valid TOML can still nest deeper than the parser recurses, or hold an integer Python
    will not convert: the file is the submission's, so it is a SubmissionError, never an
    error of the harness, which would be retried."""
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "x"\nx = {value}\n')
    with pytest.raises(SubmissionError, match="pyproject.toml cannot be parsed") as caught:
        declared_dependencies(tmp_path)
    assert words in str(caught.value)


def test_a_pyproject_its_permissions_keep_out_is_a_submission_error(tmp_path):
    """The submission's own permissions keeping the harness out of its file are its doing."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\n')
    pyproject.chmod(0)
    try:
        with pytest.raises(SubmissionError, match=r"^pyproject.toml cannot be read: \[Errno 13\]"):
            declared_dependencies(tmp_path)
    finally:
        pyproject.chmod(0o644)


@pytest.mark.parametrize("code, submissions", [(errno.EACCES, True), (errno.EPERM, True), (errno.EIO, False),
                                               (errno.ESTALE, False), (errno.ENOSPC, False)],
                         ids=["EACCES", "EPERM", "EIO", "ESTALE", "ENOSPC"])
def test_a_pyproject_read_error_is_the_submissions_only_for_its_permissions(tmp_path, monkeypatch, code,
                                                                            submissions):
    """EACCES and EPERM come of the submission's own permissions; any other error reading the
    file is the node's, which the prescan leaves to its caller to report, errno and all."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

    def read_text(self, *args, **kwargs):
        raise OSError(code, os.strerror(code), str(self))

    monkeypatch.setattr("hls.egress.prescan.Path.read_text", read_text)
    expected = SubmissionError if submissions else OSError
    with pytest.raises(expected, match=rf"\[Errno {code}\]") as caught:
        declared_dependencies(tmp_path)
    assert isinstance(caught.value, SubmissionError) is submissions


def test_a_pyproject_that_is_not_utf8_is_a_submission_error(tmp_path):
    (tmp_path / "pyproject.toml").write_bytes(b'[project]\nname = "x"\n# \xff\n')
    with pytest.raises(SubmissionError, match="^pyproject.toml is not UTF-8$"):
        declared_dependencies(tmp_path)


def test_a_pyproject_too_big_to_parse_in_memory_is_a_submission_error(tmp_path, monkeypatch):
    """Only the submission chose the file's bytes, so running out of memory parsing them is
    its own failure too."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

    def loads(text):
        raise MemoryError

    monkeypatch.setattr("hls.egress.prescan.tomllib.loads", loads)
    with pytest.raises(SubmissionError, match=r"pyproject.toml cannot be parsed: MemoryError$"):
        declared_dependencies(tmp_path)
