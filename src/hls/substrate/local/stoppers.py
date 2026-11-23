"""What a job's end stops outside its supervisor's processes: anything labelled with the job id,
such as the role containers a launcher started for it.

A job record names its stopper by a key of ``STOPPERS``, with the options that stopper takes,
such as the Docker daemon's socket. The supervisor calls it once the batch script and every
process it left have been killed, and a pass calls it for a job whose supervisor is gone. The
job's reservation is given back only after the stopper returned, so a stopper that raises keeps
the job's GPUs and CPUs held until a later call succeeds. Each stopper takes exactly its options,
none defaulted: a stopper that asked the wrong daemon would find nothing to stop and give the
reservation back while the job's containers run."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path


class Stopper(ABC):
    """Stops what is labelled with a job id."""

    @abstractmethod
    def stop(self, job_id: str) -> None:
        """Stop everything labelled with job *job_id* and return once nothing of it runs; raise when
        that cannot be confirmed."""


class NoStop(Stopper):
    """For jobs whose roles are all processes of the batch script: there is nothing else to stop."""

    def stop(self, job_id: str) -> None:
        return None


def _docker(options: Mapping[str, str]) -> Stopper:
    # Imported here, so that a supervisor of a job with nothing to stop imports no Docker code.
    from hls.substrate.docker.launcher import DockerStopper
    return DockerStopper(socket_path=Path(options["socket"]))


@dataclass(frozen=True)
class StopperKind:
    """One kind of stopper: the options it takes, every one of them required, and how it is made."""

    options: frozenset[str]
    make: Callable[[Mapping[str, str]], Stopper]


#: The stoppers a job may name, by key. "docker" stops the containers labelled ``hls.job=<id>`` on
#: the daemon at its ``socket``.
STOPPERS: dict[str, StopperKind] = {
    "none": StopperKind(frozenset(), lambda options: NoStop()),
    "docker": StopperKind(frozenset({"socket"}), _docker),
}


def check_stopper(name: str, options: Mapping[str, str]) -> None:
    """Raise ValueError unless *name* is a key of ``STOPPERS`` and *options* are exactly its
    options."""
    kind = STOPPERS.get(name)
    if kind is None:
        raise ValueError(f"no stopper {name!r}; known: {sorted(STOPPERS)}")
    if set(options) != kind.options:
        raise ValueError(f"stopper {name!r} takes the options {sorted(kind.options)}, not {sorted(options)}")


def stopper(name: str, options: Mapping[str, str]) -> Stopper:
    """The stopper *name* names, made with *options* (``check_stopper``)."""
    check_stopper(name, options)
    return STOPPERS[name].make(options)
