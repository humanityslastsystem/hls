"""``hls build-pasta`` (``hls.substrate.pasta``): the pinned passt release built with a new
enough compiler, staged with its links, and refused before anything runs on a login node or
into the checkout. A fake stands in for every program the build runs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.pipeline import commands
from hls.substrate import pasta
from hls.substrate.pasta import PASST_COMMIT, PASST_REPO, PASST_TAG, PastaRefused, build_pasta


class FakeBuild:
    """Stands in for ``subprocess.run``: the compiler reports *version*, the clone makes the source
    directory, ``make`` writes the programs named in *makes* (or fails with *make_status*), and the
    staged pasta prints its version. Records every argv."""

    def __init__(self, version: str = "13.2.0", makes: tuple[str, ...] = ("passt", "passt.avx2"),
                 make_status: int = 0, head: str = PASST_COMMIT, version_status: int = 0) -> None:
        self.version, self.makes, self.make_status, self.head = version, makes, make_status, head
        self.version_status = version_status
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[1:] == ["-dumpversion"]:
            return subprocess.CompletedProcess(argv, 0, f"{self.version}\n", "")
        if argv[:2] == ["git", "clone"]:
            Path(argv[-1]).mkdir(parents=True)
            return subprocess.CompletedProcess(argv, 0)
        if argv[0] == "git":
            assert argv[3:] == ["rev-parse", "HEAD"] and Path(argv[2]).is_dir()
            return subprocess.CompletedProcess(argv, 0, f"{self.head}\n", "")
        if argv[0] == "make":
            if self.make_status == 0:
                for name in self.makes:
                    Path(argv[2], name).write_bytes(f"#!{name}\n".encode())
            return subprocess.CompletedProcess(argv, self.make_status)
        assert argv[1:] == ["--version"] and os.path.islink(argv[0])
        return subprocess.CompletedProcess(argv, self.version_status, f"pasta {PASST_TAG}\nCopyright\n", "")


def _node(tmp_path: Path, *, job: str | None = "123") -> dict[str, str]:
    """A compute node's environment: ``srun`` on PATH, and a Slurm job unless *job* is None."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    srun = bin_dir / "srun"
    srun.write_text("#!/bin/sh\n")
    srun.chmod(0o755)
    return {"PATH": str(bin_dir), **({"SLURM_JOB_ID": job} if job else {})}


def test_the_pinned_release_is_built_and_staged_with_its_links(tmp_path):
    fake, dest = FakeBuild(), tmp_path / "shared" / "pasta-build"
    printed: list[str] = []
    staged = build_pasta(dest, checkout=tmp_path / "checkout", cc="gcc-13", environ=_node(tmp_path), run=fake,
                         echo=printed.append)
    assert staged == dest / "pasta"
    source = fake.calls[1][-1]
    assert fake.calls == [["gcc-13", "-dumpversion"],
                          ["git", "clone", "--depth", "1", "--branch", PASST_TAG, PASST_REPO, source],
                          ["git", "-C", source, "rev-parse", "HEAD"],
                          ["make", "-C", source, "CC=gcc-13"],
                          [str(dest / "pasta"), "--version"]]
    assert sorted(p.name for p in dest.iterdir()) == ["passt", "passt.avx2", "pasta", "pasta.avx2"]
    for program, link in pasta.PROGRAMS:
        assert (dest / program).read_bytes() == f"#!{program}\n".encode()
        assert (dest / program).stat().st_mode & 0o777 == 0o755
        assert os.readlink(dest / link) == program
    # The clone and the build happen in a directory that is gone afterwards.
    assert not Path(source).exists()
    assert printed[-1] == f"pasta {PASST_TAG}"


def test_a_build_with_no_avx2_program_leaves_none_from_another_build(tmp_path):
    dest = tmp_path / "pasta-build"
    dest.mkdir()
    (dest / "passt.avx2").write_text("old")
    (dest / "pasta.avx2").symlink_to("passt.avx2")
    build_pasta(dest, checkout=tmp_path / "checkout", environ=_node(tmp_path), run=FakeBuild(makes=("passt",)),
                echo=lambda line: None)
    assert sorted(p.name for p in dest.iterdir()) == ["passt", "pasta"]


@pytest.mark.parametrize("version", ["12.3.0", "9", "", "clang"])
def test_a_compiler_older_than_13_is_refused_before_the_clone(tmp_path, version):
    fake = FakeBuild(version=version)
    with pytest.raises(PastaRefused, match="version 13 or later"):
        build_pasta(tmp_path / "out", checkout=tmp_path / "checkout", environ=_node(tmp_path), run=fake)
    assert fake.calls == [["cc", "-dumpversion"]] and not (tmp_path / "out").exists()


def test_a_login_node_is_refused_before_anything_runs(tmp_path):
    fake = FakeBuild()
    with pytest.raises(PastaRefused, match=r"in no Slurm allocation.*srun .* uv run hls build-pasta"):
        build_pasta(tmp_path / "out", checkout=tmp_path / "checkout", environ=_node(tmp_path, job=None), run=fake)
    assert fake.calls == []


def test_a_machine_without_slurm_builds_where_it_is(tmp_path):
    build_pasta(tmp_path / "out", checkout=tmp_path / "checkout", environ={"PATH": str(tmp_path / "empty")},
                run=FakeBuild(), echo=lambda line: None)
    assert (tmp_path / "out" / "passt").is_file()


@pytest.mark.parametrize("inside", ["pasta-build", "."])
def test_a_directory_inside_the_checkout_is_refused(tmp_path, inside):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    fake = FakeBuild()
    with pytest.raises(PastaRefused, match="inside the checkout"):
        build_pasta(checkout / inside, checkout=checkout, environ=_node(tmp_path), run=fake)
    assert fake.calls == [] and not (checkout / "pasta-build").exists()


@pytest.mark.parametrize("head", ["b" * 40, "a9c61ff" + "0" * 33, PASST_COMMIT[:7], ""])
def test_a_tag_that_names_another_commit_is_refused_before_the_build(tmp_path, head):
    """Only the whole commit id passes: one that shares the tag name's seven digits does not."""
    fake = FakeBuild(head=head)
    with pytest.raises(PastaRefused, match=f"names commit {head or 'none'}, not {PASST_COMMIT}"):
        build_pasta(tmp_path / "out", checkout=tmp_path / "checkout", environ=_node(tmp_path), run=fake,
                    echo=lambda line: None)
    assert [argv[0] for argv in fake.calls] == ["cc", "git", "git"] and not (tmp_path / "out").exists()


def test_a_link_at_a_staging_name_is_replaced_not_written_through(tmp_path):
    """The directory is shared: a link another user leaves at a staging name must not carry the
    program out of it."""
    dest, outside = tmp_path / "pasta-build", tmp_path / "outside.txt"
    dest.mkdir()
    outside.write_text("the user's own file\n")
    for program, _ in pasta.PROGRAMS:
        (dest / f".{program}.staging").symlink_to(outside)
    build_pasta(dest, checkout=tmp_path / "checkout", environ=_node(tmp_path), run=FakeBuild(),
                echo=lambda line: None)
    assert outside.read_text() == "the user's own file\n"
    for program, _ in pasta.PROGRAMS:
        assert not (dest / program).is_symlink() and (dest / program).read_bytes() == f"#!{program}\n".encode()
    assert sorted(p.name for p in dest.iterdir()) == ["passt", "passt.avx2", "pasta", "pasta.avx2"]


def test_a_staged_pasta_that_does_not_start_is_refused(tmp_path):
    with pytest.raises(PastaRefused, match=r"pasta --version failed with exit status 1"):
        build_pasta(tmp_path / "out", checkout=tmp_path / "checkout", environ=_node(tmp_path),
                    run=FakeBuild(version_status=1), echo=lambda line: None)


def test_a_failed_build_stages_nothing(tmp_path):
    with pytest.raises(PastaRefused, match=f"make of passt {PASST_TAG} failed with exit status 2"):
        build_pasta(tmp_path / "out", checkout=tmp_path / "checkout", environ=_node(tmp_path),
                    run=FakeBuild(make_status=2), echo=lambda line: None)
    assert not (tmp_path / "out").exists()


def test_the_cli_prints_the_setting_and_refuses_in_one_line(monkeypatch, tmp_path):
    seen = {}

    def build(dest, **kw):
        seen.update(kw, dest=dest)
        return dest / "pasta"

    monkeypatch.setattr(pasta, "build_pasta", build)
    result = CliRunner().invoke(cli.app, ["build-pasta", str(tmp_path / "out"), "--cc", "gcc-13"])
    assert result.exit_code == 0, result.output
    assert (seen["dest"], seen["cc"]) == (tmp_path / "out", "gcc-13")
    # The refusal of a directory inside the checkout is the checkout this hls runs from.
    assert seen["checkout"] == commands.checkout_root() and seen["run"] is subprocess.run
    assert f'pasta_path = "{tmp_path / "out" / "pasta"}"' in result.output

    def refuse(dest, **kw):
        raise PastaRefused("no compiler")

    monkeypatch.setattr(pasta, "build_pasta", refuse)
    result = CliRunner().invoke(cli.app, ["build-pasta", str(tmp_path / "out")])
    assert result.exit_code == 1 and result.stderr == "hls build-pasta: no compiler\n"
