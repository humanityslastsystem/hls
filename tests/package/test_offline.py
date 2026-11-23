from __future__ import annotations

import errno
import os
import subprocess
from pathlib import Path

import pytest

from hls.package.offline import (INSTALL_BYTES, INSTALL_MAX_GIB, INSTALL_RC, OVER_LIMIT_EXIT, install_failed,
                                 installed_bytes, new_nonce, offline_install, project_name, render_offline_def)
from hls.recipes.model import CONSTRAINTS

#: The nonce every .def of these tests is rendered with.
NONCE = "0123456789abcdef" * 2
#: The build's status line, before its status.
RC = f"{INSTALL_RC}-{NONCE}="
#: The build's size line, before its size.
ADDED = f"{INSTALL_BYTES}-{NONCE}="
#: What every .def of these tests is rendered from: a base built from the recipes, which holds
#: the bases' constraints file.
PATHS = {"base_sif": Path("/n/base.sif"), "tree": Path("/n/tree"), "wheels": Path("/n/wheels"), "project": "subm",
         "server": "server.py", "nonce": NONCE, "constraints": CONSTRAINTS}


def _tree(tmp_path: Path, pyproject: str) -> Path:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "pyproject.toml").write_text(pyproject)
    return tree


@pytest.mark.parametrize("text, name", [
    ('[project]\nname = "subm"\n', "subm"),
    ('[project]\nname = "My.Server-2"\n', "My.Server-2"),
    ('[project]\nversion = "1"\n', None),
    ("[tool.x]\ny = 1\n", None),
    ('[project]\nname = "x; rm -rf /"\n', None),
    ('[project]\nname = "-x"\n', None),
    ('[project]\nname = 3\n', None),
    ("not toml [", None),
])
def test_the_project_name_is_read_from_pyproject_or_refused(tmp_path, text, name):
    assert project_name(_tree(tmp_path, text)) == name


def test_a_missing_pyproject_has_no_name(tmp_path):
    (tmp_path / "t").mkdir()
    assert project_name(tmp_path / "t") is None


@pytest.mark.parametrize("tail", [
    b"x = " + b"[" * 1000 + b"]" * 1000 + b"\n",
    b"x = " + b"1" * 5000 + b"\n",
    b"# \xff\n",
], ids=["nested-1000-deep", "an-integer-of-5000-digits", "not-utf8"])
def test_a_pyproject_the_parser_cannot_hold_has_no_name(tmp_path, tail):
    """The submission chose the file's bytes, so any way they fail to parse leaves it without a
    name, which the stage refuses as the submission's, never an error of its own."""
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "pyproject.toml").write_bytes(b'[project]\nname = "subm"\n' + tail)
    assert project_name(tree) is None


@pytest.mark.parametrize("code, submissions", [(errno.EACCES, True), (errno.EPERM, True), (errno.EIO, False),
                                               (errno.ESTALE, False), (errno.ENOSPC, False)],
                         ids=["EACCES", "EPERM", "EIO", "ESTALE", "ENOSPC"])
def test_a_pyproject_read_error_leaves_no_name_only_for_its_permissions(tmp_path, monkeypatch, code,
                                                                        submissions):
    """The submission's own permissions keeping the file out leave it without a name, which is
    its own fault; any other error reading the file is the node's, raised for the stage."""
    tree = _tree(tmp_path, '[project]\nname = "subm"\n')

    def read_text(self, *args, **kwargs):
        raise OSError(code, os.strerror(code), str(self))

    monkeypatch.setattr("hls.package.offline.Path.read_text", read_text)
    if submissions:
        assert project_name(tree) is None
    else:
        with pytest.raises(OSError, match=rf"\[Errno {code}\]"):
            project_name(tree)


def test_a_pyproject_too_big_to_parse_in_memory_has_no_name(tmp_path, monkeypatch):
    def loads(text):
        raise MemoryError

    monkeypatch.setattr("hls.package.offline.tomllib.loads", loads)
    assert project_name(_tree(tmp_path, '[project]\nname = "subm"\n')) is None


def test_the_offline_def_installs_the_project_by_name_from_the_wheels(tmp_path):
    text = render_offline_def(**PATHS)
    assert text.splitlines()[:2] == ["Bootstrap: localimage", "From: /n/base.sif"]
    assert "    /n/tree /srv-stage" in text and "    /n/wheels /wheels" in text
    assert ("PIP_CONSTRAINT=/opt/hls/constraints.txt /opt/venv/bin/python -m pip install --no-index --no-cache-dir "
            "--only-binary :all: --find-links /wheels subm || hls_rc=$?") in text
    assert "pip install ." not in text and "set +x" in text
    assert text.rstrip().endswith('exec /opt/venv/bin/python /workspace/server.py "$@"')


def test_the_offline_def_checks_the_bases_constraints_file_before_it_copies_anything():
    post = _post(render_offline_def(**PATHS))
    assert post[2] == ('[ -f /opt/hls/constraints.txt ] || '
                       '{ echo "the submission base holds no constraints file /opt/hls/constraints.txt" >&2; exit 1; }')
    assert post.index("mkdir -p /workspace") == 3


def test_a_base_that_holds_no_constraints_file_is_installed_into_with_none():
    """A published base holds no constraints file, and its environment names none: the install
    names none either, as the dry run and the wheel step resolved with none."""
    text = render_offline_def(**{**PATHS, "constraints": None})
    assert "PIP_CONSTRAINT" not in text and "constraints.txt" not in text
    assert ("\n    /opt/venv/bin/python -m pip install --no-index --no-cache-dir --only-binary :all: --find-links "
            "/wheels subm || hls_rc=$?\n") in text


def test_the_offline_defs_status_lines_carry_its_nonce():
    """Its status lines are the only ones the stage reads, and pip, which echoes text the
    submission chose, cannot know the nonce."""
    post = _post(render_offline_def(**PATHS))
    assert f'echo "{RC}$hls_rc"' in post
    assert any(line.startswith(f'echo "{ADDED}$((') for line in post)
    assert not [line for line in post if "hls-offline-install" in line and NONCE not in line]


def test_each_nonce_is_fresh_and_32_hex_digits():
    nonces = {new_nonce() for _ in range(100)}
    assert len(nonces) == 100 and all(len(n) == 32 and set(n) <= set("0123456789abcdef") for n in nonces)


@pytest.mark.parametrize("kwargs, match", [
    ({"project": "x; rm"}, "project name"),
    ({"project": "."}, "project name"),
    ({"server": "../server.py"}, "server"),
    ({"server": "/abs.py"}, "server"),
    ({"tree": Path("rel/tree")}, "absolute"),
    ({"tree": Path("/n/my tree")}, "whitespace"),
    ({"wheels": Path("/n/w\nheels")}, "whitespace"),
    ({"nonce": "0123"}, "nonce"),
    ({"nonce": NONCE.upper()}, "nonce"),
    ({"nonce": NONCE[:-1] + "\n"}, "nonce"),
    ({"nonce": NONCE + "0"}, "nonce"),
    ({"constraints": "opt/hls/constraints.txt"}, "absolute"),
    ({"constraints": "/opt/hls/c onstraints.txt"}, "whitespace"),
])
def test_the_offline_def_refuses_what_it_cannot_carry(kwargs, match):
    with pytest.raises(ValueError, match=match):
        render_offline_def(**{**PATHS, **kwargs})


@pytest.mark.parametrize("log, failed", [
    (f"x\nERROR: No matching distribution\n{RC}1\n", True),
    (f"Successfully installed subm\n{RC}0\nFATAL: mksquashfs\n", False),
    (f"{RC}0", False),
    ("FATAL: while copying /n/tree: no space left on device\n", None),
    ("Killed\n", None),
    (f"{RC}abc\n", None),
    (f"{RC}1x\n", None),
    (f"{RC}\u0663\n", None),
    (f"{RC}\n", None),
    (f" {RC}1\n", None),
    (f"{RC}1\n{RC}1\n", None),
    ("hls-offline-install-begin\nhls-offline-install-rc=0\n", None),
    (f"{INSTALL_RC}-{'f' * 32}=0\n", None),
], ids=["failed", "succeeded", "no-final-newline", "before-the-install", "killed", "not-a-number", "trailing",
        "another-scripts-digit", "empty", "indented", "twice", "the-old-markers", "another-nonce"])
def test_the_build_log_says_whether_the_install_itself_failed(log, failed):
    assert install_failed(log, NONCE) is failed


@pytest.mark.parametrize("brk", ["\u2028", "\u2029", "\x85", "\x1c", "\x1d", "\x1e", "\v", "\f", "\r"],
                         ids=lambda c: f"U+{ord(c):04X}")
def test_only_a_newline_ends_a_line_of_the_build_log(brk):
    """``str.splitlines`` also breaks lines at these, which a wheel's metadata can hold and pip
    echoes: text after one is no line of its own, even holding the nonce."""
    assert install_failed(f"{RC}1\n    foo{brk}{RC}0{brk}\n", NONCE) is True
    assert install_failed(f"    foo{brk}{RC}0{brk}\n", NONCE) is None
    assert installed_bytes(f"{RC}0\n{ADDED}12\nx{brk}{ADDED}99\n", NONCE) == 12


def _post(text: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index("%post") + 1
    return [line.strip() for line in lines[start:lines.index("%runscript")] if line.strip()]


def test_the_install_reports_what_it_added_to_the_venv_after_the_wheels_are_gone():
    post = _post(render_offline_def(**PATHS))
    before = post.index('hls_venv="$(du -sb /opt/venv)"')
    install = next(i for i, line in enumerate(post) if line.startswith("PIP_CONSTRAINT="))
    status = post.index(f'echo "{RC}$hls_rc"')
    gone, after = post.index("rm -rf /wheels"), post.index('hls_added="$(du -sb /opt/venv)"')
    report = next(i for i, line in enumerate(post) if line.startswith(f'echo "{ADDED}'))
    assert before < install < status < gone < after < report
    assert INSTALL_MAX_GIB == 24


def test_the_size_steps_measure_the_venv_in_sh(tmp_path):
    """The size steps, as rendered, in the image's ``/bin/sh``, around a 100000-byte install into
    a venv directory standing in for ``/opt/venv``."""
    post = _post(render_offline_def(**PATHS))
    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    steps = [post[post.index('hls_venv="$(du -sb /opt/venv)"')], f"head -c 100000 /dev/zero > {venv}/lib/big",
             *post[post.index('hls_added="$(du -sb /opt/venv)"'):][:3]]
    script = "set -eu\n" + "\n".join(steps).replace("/opt/venv", str(venv))
    out = subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True, check=True).stdout
    # The size steps run once the install succeeded, after its status line.
    added = installed_bytes(f"{RC}0\n{out}", NONCE)
    assert added is not None and 100000 <= added < 100000 + 8192, out


@pytest.mark.parametrize("log, added", [
    (f"{RC}0\n{ADDED}12345\n", 12345),
    (f"{RC}0\n{ADDED}-20\n", -20),
    (f"{ADDED}7\n{RC}0\n", 7),
    (f"{RC}0\n{ADDED}12x\n", None),
    (f"{RC}0\n{ADDED}\n", None),
    (f"{RC}0\n{ADDED}1\n{ADDED}2\n", None),
    (f"{RC}0\n{INSTALL_BYTES}-{'f' * 32}=12\n", None),
    (f"{RC}0\n{INSTALL_BYTES}=12\n", None),
    (f"{RC}1\n{ADDED}12\n", None),
    (f"{ADDED}12\n", None),
    ("FATAL: no space\n", None),
], ids=["added", "smaller", "in-any-order", "not-a-number", "empty", "twice", "another-nonce", "the-old-marker",
        "after-a-failed-install", "no-status", "before-the-install"])
def test_the_build_log_says_what_the_install_added(log, added):
    """Only the build's own lines carry the nonce, so where they sit in the log does not matter;
    the size counts only once the install's status is 0."""
    assert installed_bytes(log, NONCE) == added



def _run_post(tmp_path: Path, sizes: tuple[int, int], *, install_rc: int = 0, constraints: str | None = CONSTRAINTS,
              holds_constraints: bool = True) -> subprocess.CompletedProcess:
    """Run the whole rendered ``%post``, its install under *constraints*, in ``/bin/sh`` with its
    paths moved under *tmp_path*, a base that holds the bases' constraints file when
    *holds_constraints*, an install that prints the constraints file it was given and exits
    *install_rc*, failing as pip does on one it cannot open, and a ``du`` that measures the venv
    as *sizes* before and after the install."""
    root = tmp_path / "root"
    venv_bin = root / "opt" / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    for d in ("srv-stage", "wheels"):
        (root / d).mkdir()
    (root / "srv-stage" / "server.py").write_text("print('serving')\n")
    if holds_constraints:
        (root / "opt" / "hls").mkdir()
        (root / "opt" / "hls" / "constraints.txt").write_text("torch==2.11.0+cu130\n")
    (venv_bin / "python").write_text(
        '#!/bin/sh\necho "pip-constraint=${PIP_CONSTRAINT-none}"\n'
        '[ -z "${PIP_CONSTRAINT-}" ] || [ -f "$PIP_CONSTRAINT" ] || '
        '{ echo "ERROR: Could not open constraint file" >&2; exit 1; }\n'
        f"exit {install_rc}\n")
    (venv_bin / "python").chmod(0o755)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    count, listed = tmp_path / "n", tmp_path / "sizes"
    listed.write_text("".join(f"{n}\n" for n in sizes))
    (stubs / "du").write_text(f'#!/bin/sh\nn=$(($(cat {count} 2>/dev/null || echo 0) + 1))\necho $n > {count}\n'
                              f'printf "%s\\t%s\\n" "$(sed -n "${{n}}p" {listed})" "$2"\n')
    (stubs / "du").chmod(0o755)
    post = "\n".join(_post(render_offline_def(**{**PATHS, "constraints": constraints})))
    for path in ("/opt/venv", "/opt/hls", "/workspace", "/srv-stage", "/wheels"):
        post = post.replace(path, f"{root}{path}")
    return subprocess.run(["/bin/sh", "-c", post], capture_output=True, text=True,
                          env={"PATH": f"{stubs}:/usr/bin:/bin"})


def test_the_install_applies_the_bases_constraints_file(tmp_path):
    done = _run_post(tmp_path, (1000, 2000))
    assert done.returncode == 0, done.stderr
    assert f"pip-constraint={tmp_path / 'root'}{CONSTRAINTS}" in done.stdout.split("\n")
    assert install_failed(done.stdout, NONCE) is False


def test_a_base_that_holds_no_constraints_file_installs_with_none(tmp_path):
    """A submission base that holds no constraints file: the install names none, and succeeds."""
    done = _run_post(tmp_path, (1000, 2000), constraints=None, holds_constraints=False)
    assert done.returncode == 0, done.stderr
    assert "pip-constraint=none" in done.stdout.split("\n") and install_failed(done.stdout, NONCE) is False


def test_a_named_constraints_file_the_base_lacks_ends_the_build_before_the_install(tmp_path):
    """The file is the harness's to supply: without it the build ends before it copies anything
    and before the install's status line, so the stage books the failure as its own, and its
    stderr names the file."""
    done = _run_post(tmp_path, (1000, 2000), holds_constraints=False)
    assert done.returncode == 1 and install_failed(done.stdout, NONCE) is None
    assert "pip-constraint" not in done.stdout and not (tmp_path / "root" / "workspace").exists()
    assert done.stderr == f"the submission base holds no constraints file {tmp_path / 'root'}{CONSTRAINTS}\n"


@pytest.mark.parametrize("added, rc", [(INSTALL_MAX_GIB * 2**30, 0), (INSTALL_MAX_GIB * 2**30 + 1, OVER_LIMIT_EXIT),
                                       (-4096, 0)], ids=["at-the-limit", "one-byte-past", "smaller"])
def test_the_build_stops_once_the_install_added_more_than_the_limit(tmp_path, added, rc):
    """Past the limit the build ends right after it reports what the install added, before
    apptainer squashes the tree into an image the stage would copy and extract: an install
    between the limit and what the node's disk holds would otherwise fill the disk there, which
    reads as the harness's failure."""
    done = _run_post(tmp_path, (10**6, 10**6 + added))
    assert done.returncode == rc, done.stderr
    assert installed_bytes(done.stdout, NONCE) == added
    assert not (tmp_path / "root" / "wheels").exists() and (tmp_path / "root" / "workspace" / "server.py").is_file()
    # The tree's staged copy goes once it is in /workspace: the image holds the tree once.
    assert not (tmp_path / "root" / "srv-stage").exists()


def test_a_failed_install_stops_before_the_venv_is_measured_again(tmp_path):
    done = _run_post(tmp_path, (1000, 2000), install_rc=1)
    assert done.returncode == 1 and install_failed(done.stdout, NONCE) is True
    assert installed_bytes(done.stdout, NONCE) is None


def test_the_limit_is_checked_right_after_the_report():
    post = _post(render_offline_def(**PATHS))
    report = next(i for i, line in enumerate(post) if line.startswith(f'echo "{ADDED}'))
    check = next(i for i, line in enumerate(post) if line.endswith(f" -le {INSTALL_MAX_GIB * 2**30} ] || exit "
                                                                    f"{OVER_LIMIT_EXIT}"))
    assert check == report + 2 and post[report + 1].startswith("hls_rc=$?")


@pytest.mark.parametrize("constraints", [CONSTRAINTS, None])
def test_every_substrates_install_is_the_defs_script_and_runscript(constraints):
    """``offline_install`` is the one text of the install: the .def's %post is its script and its
    %runscript its command, which a Docker build runs as they are."""
    definition = render_offline_def(**{**PATHS, "constraints": constraints})
    post, run = offline_install(**{**{k: v for k, v in PATHS.items() if k not in ("base_sif", "tree", "wheels")},
                                   "constraints": constraints})
    sections = definition.split("\n")
    assert sections[sections.index("%post") + 1:sections.index("%runscript") - 1] == [f"    {line}" for line in post]
    assert sections[sections.index("%runscript") + 1] == f"    {run}"


@pytest.mark.parametrize("kwargs, match", [
    ({"project": "x; rm"}, "project name"),
    ({"server": "../server.py"}, "server"),
    ({"server": "/abs.py"}, "server"),
    ({"nonce": NONCE.upper()}, "nonce"),
    ({"nonce": NONCE + "0"}, "nonce"),
    ({"constraints": "opt/hls/constraints.txt"}, "absolute"),
    ({"constraints": "/opt/hls/c onstraints.txt"}, "whitespace"),
])
def test_the_install_refuses_what_the_def_refuses(kwargs, match):
    base = {k: v for k, v in PATHS.items() if k not in ("base_sif", "tree", "wheels")}
    with pytest.raises(ValueError, match=match):
        offline_install(**{**base, **kwargs})
