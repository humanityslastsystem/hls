"""Build the ``pasta`` program the network seal runs (``site.pasta_path``) from a pinned passt
release.

Compute nodes often lack the passt package. pasta links only the system C library, so one copy,
built once in a directory every compute node mounts, runs on all of them. The build clones the
release tag ``PASST_TAG`` from ``PASST_REPO`` into a temporary directory and runs ``make`` there
with a C compiler of major version 13 or later, since passt uses C23's single-argument
``static_assert``. It stages ``passt`` and ``pasta``, a link to it (the program acts on the name
it is started by), and, when the build made one, the AVX2 build ``passt.avx2`` with its link
``pasta.avx2``: pasta starts that one from beside itself on an AVX2 host, and runs slower
without it.

A build is refused before anything runs on a node that has ``srun`` but is in no Slurm
allocation, which is a login node, and for a directory inside the checkout, since a file there
that git neither tracks nor ignores makes every scored command refuse the checkout."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

#: The passt repository and the release tag built from it. A new tag is a change to the seal:
#: its forwarding defaults are read from this release (``transparent_seal.sh``).
PASST_REPO = "https://passt.top/passt"
PASST_TAG = "2026_06_11.a9c61ff"
#: The commit the tag names (``git ls-remote`` of the tag); a clone at any other commit is refused
#: before it is built.
PASST_COMMIT = "a9c61ffaf15347b8dfcc2347c5440e4b0e82333b"
#: The least major version of the C compiler.
MIN_CC_MAJOR = 13
#: The programs staged, each a file the build makes and the link that starts it as pasta.
PROGRAMS = (("passt", "pasta"), ("passt.avx2", "pasta.avx2"))


class PastaRefused(ValueError):
    """A build that cannot start or did not finish, with the reason in one line."""


def _checked(run: Callable[..., subprocess.CompletedProcess], argv: list[str], what: str,
             **kwargs) -> subprocess.CompletedProcess:
    try:
        done = run(argv, **kwargs)
    except OSError as e:
        raise PastaRefused(f"{what} could not start: {e}") from None
    if done.returncode != 0:
        raise PastaRefused(f"{what} failed with exit status {done.returncode}")
    return done


def refusal(dest: Path, *, checkout: Path, environ: Mapping[str, str]) -> str | None:
    """Why pasta may not be built into *dest* from here, or None: *dest* lies inside *checkout*, or
    *environ*'s PATH holds ``srun`` while it names no Slurm job (``SLURM_JOB_ID``)."""
    if dest.resolve().is_relative_to(checkout.resolve()):
        return (f"{dest} is inside the checkout {checkout}, where a file git neither tracks nor ignores makes "
                "every scored command refuse the checkout; name a directory outside it that every compute node "
                "mounts")
    if not environ.get("SLURM_JOB_ID") and shutil.which("srun", path=environ.get("PATH", "")):
        return (f"this shell is in no Slurm allocation, which makes it a login node; build inside one: "
                f"srun -p <partition> --time 15 --mem 4G --cpus-per-task 4 uv run hls build-pasta {dest}")
    return None


def build_pasta(dest: Path, *, checkout: Path, run: Callable[..., subprocess.CompletedProcess], cc: str = "cc",
                environ: Mapping[str, str] = os.environ, echo: Callable[[str], None] = print) -> Path:
    """Build pasta from ``PASST_TAG`` with the C compiler *cc* and stage it in *dest*, made if
    missing; return the staged ``pasta``. Raise PastaRefused, with nothing staged, when
    ``refusal`` names a reason, when *cc* is older than ``MIN_CC_MAJOR``, when the clone fails or
    is not at ``PASST_COMMIT``, or when the build fails; raise it too when the staged pasta does
    not start. *run* runs every program, as ``subprocess.run`` does; the clone and the build print
    to this process's output."""
    why = refusal(dest, checkout=checkout, environ=environ)
    if why:
        raise PastaRefused(why)
    asked = _checked(run, [cc, "-dumpversion"], f"{cc} -dumpversion", capture_output=True, text=True)
    version = asked.stdout.strip()
    major = version.split(".", 1)[0]
    if not major.isdigit() or int(major) < MIN_CC_MAJOR:
        raise PastaRefused(f"passt needs a C compiler of version {MIN_CC_MAJOR} or later for C23, and {cc} is "
                           f"{version or 'of no version'}; load a newer gcc or name one with --cc")
    with tempfile.TemporaryDirectory(prefix="hls-pasta-") as tmp:
        source = Path(tmp) / "passt"
        echo(f"cloning passt {PASST_TAG} from {PASST_REPO}")
        _checked(run, ["git", "clone", "--depth", "1", "--branch", PASST_TAG, PASST_REPO, str(source)],
                 f"git clone of passt {PASST_TAG}")
        head = _checked(run, ["git", "-C", str(source), "rev-parse", "HEAD"], f"git rev-parse of passt {PASST_TAG}",
                        capture_output=True, text=True).stdout.strip()
        if head != PASST_COMMIT:
            raise PastaRefused(f"passt's tag {PASST_TAG} names commit {head or 'none'}, not {PASST_COMMIT}")
        echo(f"building with {cc} {version}")
        _checked(run, ["make", "-C", str(source), f"CC={cc}"], f"make of passt {PASST_TAG}")
        if not (source / "passt").is_file():
            raise PastaRefused(f"make of passt {PASST_TAG} made no passt program")
        dest.mkdir(parents=True, exist_ok=True)
        for program, link in PROGRAMS:
            if not (source / program).is_file():
                # pasta would start an AVX2 build left from another release.
                (dest / link).unlink(missing_ok=True)
                (dest / program).unlink(missing_ok=True)
                continue
            staged = dest / f".{program}.staging"
            # copyfile writes through a link, and the directory is shared.
            staged.unlink(missing_ok=True)
            shutil.copyfile(source / program, staged)
            staged.chmod(0o755)
            staged.replace(dest / program)
            pending = dest / f".{link}.staging"
            pending.unlink(missing_ok=True)
            pending.symlink_to(program)
            pending.replace(dest / link)
    pasta = dest / "pasta"
    shown = _checked(run, [str(pasta), "--version"], f"{pasta} --version", capture_output=True, text=True)
    echo(shown.stdout.splitlines()[0] if shown.stdout else f"{pasta} --version printed nothing")
    return pasta
