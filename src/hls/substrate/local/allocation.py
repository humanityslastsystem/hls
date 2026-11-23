"""What the plan holds and checks of a local allocation: its placement, which names nothing, since
one host runs every allocation; the check that the host could ever hold it; and the name of its
job log."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from hls.spec.resources import ResourceShape
from hls.substrate.local.inventory import HostFacts
from hls.substrate.local.placement import never_fits

_JOB_LOG = re.compile(r"job-[0-9]+\.out")


class LocalPlacement(BaseModel):
    """Where a local allocation runs: on the host, so it names nothing. The pipeline reads a
    placement's partition, QOS, socket tasks and excluded nodes, which a local one has none of."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["local"] = "local"

    @property
    def partition(self) -> None:
        return None

    @property
    def qos(self) -> None:
        return None

    @property
    def socket_tasks(self) -> None:
        return None

    @property
    def exclude(self) -> list[str]:
        return []

    def request_problem(self, request: ResourceShape) -> str | None:
        """Nothing about a request's shape is the placement's to refuse; the host's size is
        ``check_placement``'s."""
        return None


def placement(*, partition: str | None, qos: str | None, socket_tasks: int | None,
              exclude: Sequence[str]) -> LocalPlacement:
    """The placement of a local allocation. Raise ValueError for a partition, a QOS, socket tasks or
    an excluded node, none of which the host has: an allocation that asks for one would not run as
    it asks."""
    asked = {name: value for name, value in (("partition", partition), ("qos", qos),
                                              ("socket_tasks", socket_tasks), ("exclude", list(exclude)))
             if value}
    if asked:
        raise ValueError(f"a local allocation has no {', '.join(asked)}: {asked}")
    return LocalPlacement()


def check_placement(request: ResourceShape, placement: LocalPlacement, host: HostFacts) -> None:
    """Raise ValueError when *host* could never hold *request*, even with nothing else running."""
    problems = never_fits(request, host)
    if problems:
        raise ValueError(f"the host {host.host} cannot hold the allocation: {'; '.join(problems)}")


def is_job_log(name: str) -> bool:
    """Whether *name*, a file in a submission's directory, is its job's log
    (``hls.substrate.local.supervisor.output_name``)."""
    return _JOB_LOG.fullmatch(name) is not None


def check_node_name(name: str) -> str:
    """Raise ValueError for every name: one host runs every local allocation, so there is no node a
    retry could leave out, and a retry after a failed grant check runs on the host again."""
    raise ValueError(f"{name!r}: a local allocation runs on its one host and excludes no node")
