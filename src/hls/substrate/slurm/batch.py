"""The batch script of one allocation submission. It starts the runner and does nothing else.

The job starts with Slurm's variables only (sbatch ``--export=NIL``). The script sets the
few the runner needs, all taken from the plan, and sets ``SLURM_EXPORT_ENV=ALL`` so that
srun passes each role step exactly the environment the launcher gives it. A site's
TaskProlog still adds its own variables to every step after that, such as
``BASH_ENV``, ``APPTAINER_CACHEDIR``, ``XDG_RUNTIME_DIR`` and ``SRUN_CPUS_PER_TASK``."""

from __future__ import annotations

import shlex
from pathlib import Path


def batch_script(*, python: Path, plan_dir: Path, allocation: int, submission: int, checkout: Path,
                 path: str, home: Path) -> str:
    """The script text. *python* is the checkout environment's interpreter, *plan_dir* the
    absolute path of the instance's plan directory, and *path* and *home* the PATH and
    HOME recorded in the plan."""
    q = shlex.quote
    return "\n".join([
        "#!/bin/bash",
        f"# hls allocation {allocation}, submission {submission}; written by the orchestrator.",
        f"export PATH={q(path)}",
        f"export HOME={q(str(home))}",
        "export LANG=C.UTF-8",
        "export SLURM_EXPORT_ENV=ALL",
        f"cd {q(str(checkout))} || exit 2",
        f"exec {q(str(python))} -m hls.substrate.runner {q(str(plan_dir))} "
        f"--allocation {allocation} --submission {submission}",
        "",
    ])
