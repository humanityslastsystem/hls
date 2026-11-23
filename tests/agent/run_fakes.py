"""One ``apptainer`` for the tests that run a whole rollout on one machine: it hands each command
to the fake that owns its image, since the build, the package and the grading each have their
own (``tests.agent.fake_sandbox``, ``tests.package.fake_tools``, ``tests.drive.fake_container``):
- any command naming the agent cap's digest (its extraction directory, its tree under
  ``sandboxes/<sha256>``) goes to the sandbox fake;
- an offline build (``build --fakeroot``) goes to the package fake;
- an extraction (``build --sandbox <dir> <image>``) goes by the image's first bytes: a tree image
  (``fake_tools.TREE_MAGIC``) to the package fake, which lays its tree out, anything else to the
  drive fake, which unpacks a tar image;
- an ``exec`` goes by its role, the name of the directory that holds its ``--workdir``: the
  package's two pip steps to the package fake, every other role (the seal probe, the artifact,
  the driver, verify) to the drive fake. The drive fake runs any extracted directory, whichever
  fake laid it out. The source_scan stage's roles (the checkout and the scan) run on the host, as
  harness roles, through no ``apptainer``.

The submission image the package fake builds holds ``/workspace/server.py`` and a runscript that
start a placeholder; ``serving_image_changes`` replaces them with the drive tests' fake server and
a runscript that finds it relative to itself, as the drive fake runs a container's files on the
host."""

from __future__ import annotations

import sys
from pathlib import Path

from tests.drive.test_stage import SERVER
from tests.package import fake_tools

#: The package stage's pip steps, the two roles of a package that run in a container.
PACKAGE_STEPS = ("dry-run", "wheel")


def install(bin_dir: Path, *, sandbox: Path, package: Path, container: Path, cap: str) -> Path:
    """Write the dispatching ``apptainer`` into *bin_dir* and return it. *sandbox*, *package* and
    *container* are the three fakes' own ``apptainer`` programs, and *cap* the first 12 hex
    digits of the agent cap's image digest."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / "apptainer"
    magic = fake_tools.TREE_MAGIC
    path.write_text(f"""#!{sys.executable}
import os
import sys

argv = sys.argv[1:]


def run(fake):
    os.execv(fake, [fake, *argv])


if {cap!r} in " ".join(argv):
    run({str(sandbox)!r})
if argv[:2] == ["build", "--fakeroot"]:
    run({str(package)!r})
if argv[:1] == ["build"] and "--sandbox" in argv:
    with open(argv[-1], "rb") as f:
        tree = f.read({len(magic)}) == {magic!r}
    run({str(package)!r} if tree else {str(container)!r})
if argv[:1] == ["exec"] and "--workdir" in argv:
    role = os.path.basename(os.path.dirname(argv[argv.index("--workdir") + 1].rstrip("/")))
    if role in {PACKAGE_STEPS!r}:
        run({str(package)!r})
run({str(container)!r})
""")
    path.chmod(0o755)
    return path


def serving_image_changes(mode: str = "serve", reply: str | None = None) -> dict[str, list]:
    """The package fake's ``image_changes`` that make the submission image it builds serve the
    drive tests' fake server in *mode*, with ``reply.txt`` beside it holding *reply* when given."""
    runscript = f'#!/bin/sh\nexec python "$(dirname "$0")/../workspace/server.py" --mode {mode} "$@"\n'
    changes: dict[str, list] = {
        ".singularity.d/runscript": ["file", runscript, 0o755],
        "workspace/server.py": ["file", SERVER.read_text(), 0o644],
    }
    if reply is not None:
        changes["workspace/reply.txt"] = ["file", reply, 0o644]
    return changes
