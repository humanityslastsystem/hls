"""The git_repo submission contract.

Active in GitRepoSubmission mode. Tells the implementer agent to submit a runnable git
repo (`pyproject.toml` + `server.py`) and validates the collected tree before the harness
builds an image from it. Parallel to ArtifactContract (the served-HTTP boundary): this
governs the *code submission*, not the HTTP interface."""

from __future__ import annotations

import errno
from dataclasses import dataclass
from pathlib import Path

#: The errors a read of what a submission produced ends with when the permissions it left keep
#: the harness out: the submission's doing. Any other OSError there (EIO, ESTALE, ENOSPC and the
#: rest) is a fault of the node.
PERMISSION_ERRNOS = frozenset({errno.EACCES, errno.EPERM})


class SubmissionError(RuntimeError):
    """The collected repo does not satisfy the submission contract."""


@dataclass(frozen=True)
class SubmissionContract:
    server_filename: str = "server.py"

    @property
    def install_command(self) -> str:
        """The exact command the harness runs to install the submission -- and the
        exact command the brief tells the agent to test with. One string, two
        consumers (brief + packager), so they can never drift."""
        return "pip install ."

    @property
    def launch_command(self) -> list[str]:
        """How the artifact container launches the server (and how the agent is
        told to test it). The packager's ENTRYPOINT and the brief both derive
        from this."""
        return ["python", self.server_filename]

    def brief_section(self) -> str:
        launch = " ".join(self.launch_command)
        return (
            "You MUST work in a git repository at the repo root. Only the latest commit on `main` "
            "is evaluated; uncommitted or unmerged work is discarded. Merge to `main` the moment "
            "you have a correct server that performs better than your last commit, and again after "
            "every such improvement -- not only at the end.\n"
            "The repo must contain:\n"
            f"- `pyproject.toml` declaring every dependency (the harness installs with "
            f"`{self.install_command}`; undeclared packages are absent at eval time).\n"
            f"- `{self.server_filename}`, launched as `{launch}`.\n"
            f"The harness builds your submission with `{self.install_command}` then `{launch}` in a "
            "fresh copy of this environment."
        )

    def validate(self, tree_dir: Path) -> None:
        """Raise SubmissionError if the collected commit is not a runnable submission."""
        missing = [f for f in ("pyproject.toml", self.server_filename)
                   if not (tree_dir / f).is_file()]
        if missing:
            raise SubmissionError(
                f"submission missing required file(s): {', '.join(missing)}")
