"""The ceiling on the processes every contestant container may grow to. The stage context reads
it once per stage and gives it to every contestant launch (``StageContext.process_ceiling``): a
container's step, or, for the build sandbox, the commands its own harness forks.

``RLIMIT_NPROC`` is no share of what a step started. The kernel holds a fork against a count of
tasks (processes and threads) of the forking process's real uid, whoever started them: a process
that started nothing is refused its fork once that count reaches its limit. Which tasks count
depends on the kernel and on where the limit was set (``CeilingRule``):
- a limit set before any user namespace is made, as the step wrapper sets a step's, counts every
  task of the uid on the node, on any kernel: each namespace made under it records it as the
  limit of its parent level;
- the build sandbox's commands get theirs from their spawner, inside a user namespace it made
  with the runner's own limit. Kernels before 5.14 count that one against every task of the uid
  on the node too (``uid``, as on el8). From 5.14 the kernel counts a namespace's tasks against
  the limit set in it, and each parent level against the limit the namespace was made with
  (``user_namespace``, as on el9): the commands' ceiling counts their own tasks, and the
  uid's others reach them only through the runner's hard limit, and the node's through its
  ``pid_max``. The site says which (``site.nproc_accounting``); the probe checks it
  (``hls.agent.probe``, ``nproc-accounting``). With 600 of the uid's tasks outside and a limit
  of 512, a limit set inside a namespace made with the runner's limit allows 0 forks on el8
  and 511 on el9; one set before the namespace is made, or with no namespace, allows 0 on both.

The runner, srun, the harness's roles and the contestant's processes all run as one uid, and so
do the uid's other jobs on the node, so wherever every task of the uid counts, all of them draw
on one pool. A contestant's fork bomb that reached the uid's hard limit would stop the harness
from forking too: its next step, its next stop, its next thread, a failure that would read as
infra, a retry the contestant could force at will.

So the runner raises its soft limit to its hard one when it starts (``raise_soft_limit``), and
every harness process keeps that limit, while a contestant's step gets the ceiling
``min(hard - k * run.nproc_reserve, site.nproc_max)`` as its soft and hard limit
(``process_ceiling``), k the evaluations of its size the node can hold at once. Contestant
processes can then fill the pool up to the ceiling and no further, and the harness of each of
the k evaluations still has its reserve. The site's cap is there for a hard limit that leaves
more than the node has: one may leave over twice the kernel's pid_max with no pids controller
above the step, so its headroom alone would let a server take every pid of the node; an
unlimited hard limit leaves the cap alone. A site that sets no cap gets ``DEFAULT_NPROC_MAX`` on
a node, or the floor of the node's k evaluations when that is more (``node_nproc_max``), so the
default is refused on no node for its size. They bound the pool, not one step's share of it. So
when a contestant's step is reported,
once it and every task of it are gone, its unreaped zombies among them, the uid's tasks and the
node's are counted again (``PoolAtEnd``): the others' alone, which the contestant cannot raise.
When at a level that counts the step's tasks they leave it at most what ``POOL_FULL`` of its
ceiling leaves (``starvation``), its end was starved by them and is infra. A count that fails at
a level the step is judged by (``PoolAtEnd.uncounted``) leaves starvation not ruled out, and the
end is infra too (``hls.substrate.faults.starvation_fault``). Otherwise it is the contestant's, whatever the walks
saw while it ran (``PoolSample``, kept as context: a server's own zombies fill the pool without
showing in its cgroup's thread list, and a storm can come and go between two walks). A ceiling
under ``FLOOR_PER_EVALUATION`` tasks for each of the k is refused before any role starts, and so
is a site's cap under that floor (``nproc_max_problem``). A plan does not know its nodes, so it
is refused only a cap under one evaluation's floor, which no node can hold; the runner checks
the cap against its node's k for each contestant stage when the allocation starts, before any
stage runs (``node_cap_problem``). The pids limit of a step's own cgroup, where a site sets one,
is the bound that does isolate a step: a step that reached it ends ``contestant_pids``.

k is the node's GPUs over the step's GPUs, or for a step with no GPU the node's CPUs over its
CPUs (``node_evaluations``), as the controller gives the node's size, or else the node itself
(``Launcher.node_size``). A node whose count is unknown, or under the step's, holds no k that is
not a guess, so no contestant step starts on it (``ProcessCeiling.problem``)."""

from __future__ import annotations

import math
import os
import re
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hls.substrate.base import NodeSize
from hls.substrate.cgroup import read_int
from hls.substrate.procfs import status_fields

#: Tasks each evaluation's server may need at least. A ceiling under this many for every
#: evaluation the node holds is refused rather than scored as the crashes it would bring: room
#: for the thousands of threads a vLLM server or a parallel compile runs. A ceiling of 131,072,
#: ``DEFAULT_NPROC_MAX``, clears the floor of a GPU node's eight evaluations four times over.
FLOOR_PER_EVALUATION = 4096
#: The cap on a contestant step's ceiling where the site sets none, on a node that holds at most
#: 32 evaluations of the step's size: their floor. A node that holds more gets their floor
#: (``node_nproc_max``).
DEFAULT_NPROC_MAX = 32 * FLOOR_PER_EVALUATION
#: The fraction of a step's ceiling the others' tasks hold, at a level that counts the step's
#: tasks, when they starve it: they leave it at most the rest (``starvation``), and its end is
#: infra (a pool others filled). It is nearly all of the pool, short
#: of all of it by a twentieth for the others' tasks that end between the step's failed fork and
#: the count at its end.
POOL_FULL = 0.95
#: How the kernel counts the build sandbox's commands against their ceiling, which their spawner
#: sets inside a user namespace it made with the runner's own limit: against every task of the
#: uid on the node (``uid``), or against their namespace's own, with its parent levels counted
#: against the runner's limit (``user_namespace``). See the module docstring.
Accounting = Literal["uid", "user_namespace"]
#: The first kernel release, as (major, minor), that counts a user namespace's tasks against the
#: limit set in it (``user_namespace``); every earlier one counts every task of the uid (``uid``).
NAMESPACE_ACCOUNTING_SINCE = (5, 14)
# The major and minor version a kernel release starts with, such as "5.14" of "5.14.0-427.el9".
_RELEASE = re.compile(r"([0-9]+)\.([0-9]+)")


@dataclass(frozen=True)
class AccountingChoice:
    """How a stage counts the build sandbox's commands against their ceiling, and why."""

    #: The accounting the stage uses.
    accounting: Accounting
    #: True when the site set it (``site.nproc_accounting``); False when the node's kernel
    #: release decided it (``kernel_accounting``).
    from_site: bool
    #: The node's kernel release (``os.uname().release``).
    kernel: str


def kernel_release() -> str:
    """This node's kernel release, such as "5.14.0-427.13.1.el9_4.x86_64"."""
    return os.uname().release


def kernel_accounting(release: str) -> Accounting:
    """How a kernel of *release* counts the build sandbox's commands: ``uid`` before
    ``NAMESPACE_ACCOUNTING_SINCE``, ``user_namespace`` from it. Raise ValueError naming the
    setting for a release that does not start with its major and minor version."""
    m = _RELEASE.match(release)
    if m is None:
        raise ValueError(f"the kernel release {release!r} does not start with <major>.<minor>, so it does not tell "
                         "how the kernel counts the build sandbox's processes; set site.nproc_accounting to \"uid\" "
                         "for a kernel before 5.14 or \"user_namespace\" for 5.14 and later")
    return "uid" if (int(m[1]), int(m[2])) < NAMESPACE_ACCOUNTING_SINCE else "user_namespace"


def choose_accounting(setting: Accounting | None, release: str) -> AccountingChoice:
    """*setting* (``site.nproc_accounting``) when the site gives one, else what the kernel
    *release* says (``kernel_accounting``, whose ValueError this raises)."""
    if setting is not None:
        return AccountingChoice(accounting=setting, from_site=True, kernel=release)
    return AccountingChoice(accounting=kernel_accounting(release), from_site=False, kernel=release)


class ProcessCeiling(BaseModel):
    """A contestant step's process ceiling, and what it was computed from, for the record. One
    whose evaluations are unknown has no ceiling and refuses every contestant step (``problem``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The runner's hard ``RLIMIT_NPROC``; None when unlimited.
    hard: int | None
    #: The tasks kept for each evaluation's harness (``run.nproc_reserve``).
    reserve: int = Field(ge=0)
    #: The evaluations of the step's size the node can hold at once (``node_evaluations``); None
    #: when they are unknown (``evaluations_unknown``).
    evaluations: int | None = Field(ge=1)
    #: Why the evaluations of the step's size the node holds are unknown, None when they are
    #: known: the node's count of what the step's size is counted in (GPUs, or CPUs for a step
    #: with none) is unknown, from its controller and from the node itself, or under the step's.
    #: No evaluation count is guessed, so ``evaluations``, ``uncapped``, ``ceiling`` and ``floor``
    #: are then None, and this is the ``problem`` that refuses every contestant step.
    evaluations_unknown: str | None
    #: The uid's headroom for the step, ``hard - evaluations * reserve``; None when the hard limit
    #: is unlimited or the evaluations unknown.
    uncapped: int | None
    #: The cap on a contestant step's ceiling: the site's (``site.nproc_max``), or the node's
    #: default for a site that sets none (``node_nproc_max``).
    nproc_max: int = Field(ge=1)
    #: The ceiling the step runs under: the headroom capped at the site's cap,
    #: ``min(uncapped, nproc_max)``, or ``nproc_max`` when the hard limit is unlimited; None when
    #: the evaluations are unknown.
    ceiling: int | None
    #: The least ceiling a stage starts under, ``evaluations * FLOOR_PER_EVALUATION``; None when
    #: the evaluations are unknown.
    floor: int | None

    @model_validator(mode="after")
    def _counted_when_known(self) -> Self:
        # A record whose ceiling is missing while its evaluations are known would start a step
        # with no ceiling at all.
        counted = (self.evaluations, self.ceiling, self.floor)
        if self.evaluations_unknown is None and None in counted:
            raise ValueError("a ceiling whose evaluations are known has its evaluations, ceiling and floor")
        if self.evaluations_unknown is not None and (counted != (None, None, None) or self.uncapped is not None):
            raise ValueError("a ceiling whose evaluations are unknown has no evaluations, headroom, ceiling or floor")
        return self

    @property
    def problem(self) -> str | None:
        """Why no contestant step may start under this ceiling, or None."""
        if self.evaluations_unknown is not None:
            return self.evaluations_unknown
        capped = nproc_max_problem(self.nproc_max, self.evaluations)
        if capped is not None:
            return capped
        if self.uncapped is None or self.uncapped >= self.floor:
            return None
        return (f"the process ceiling {self.uncapped} (the hard limit of {self.hard} tasks less {self.reserve} for "
                f"the harness of each of {self.evaluations} evaluations) is under the floor of {self.floor}, "
                f"{FLOOR_PER_EVALUATION} for each evaluation's server")


@dataclass(frozen=True)
class PoolSample:
    """The uid's tasks on the node and a contestant step's own, read together by a walk while it
    ran: context for the record, which decides nothing."""

    #: When they were read.
    at: float
    #: Every task of the uid on the node (``uid_tasks``).
    uid_tasks: int
    #: The step's own tasks (``step_tasks``); None when they could not be read.
    step_tasks: int | None
    #: The step's process ceiling.
    ceiling: int


@dataclass(frozen=True)
class PoolPeak:
    """The most tasks the walks read over a contestant step's life (``PoolSample``), recorded
    when it is reported, beside the last sample. Context, which decides nothing."""

    #: The walks that read the pool.
    walks: int
    #: The most tasks of the uid on the node one walk read.
    uid_tasks: int
    #: The most of the step's own tasks one walk read; None when no walk read them.
    step_tasks: int | None
    #: The most tasks of the uid outside the step one walk read, the uid's less the step's as that
    #: walk read both: the runner's, srun's, the harness roles' and the uid's other jobs' (for the
    #: build sandbox, whose step also holds the harness's own processes, those count in the
    #: step's); None when no walk read the step's. A walk reads the uid's once for the pass and the
    #: step's after, so a step that grew between them reads as no others, never fewer.
    others_tasks: int | None


def peak_with(peak: PoolPeak | None, sample: PoolSample) -> PoolPeak:
    """*peak*, the walks' so far (None before the first), with the next walk's *sample* in it."""
    others = None if sample.step_tasks is None else max(0, sample.uid_tasks - sample.step_tasks)
    if peak is None:
        return PoolPeak(walks=1, uid_tasks=sample.uid_tasks, step_tasks=sample.step_tasks, others_tasks=others)
    return PoolPeak(walks=peak.walks + 1, uid_tasks=max(peak.uid_tasks, sample.uid_tasks),
                    step_tasks=_most(peak.step_tasks, sample.step_tasks), others_tasks=_most(peak.others_tasks, others))


def _most(a: int | None, b: int | None) -> int | None:
    """The larger of *a* and *b*, either of which may be unread (None)."""
    return b if a is None else a if b is None else max(a, b)


@dataclass(frozen=True)
class CeilingRule:
    """A contestant step's process ceiling and how the kernel counts tasks against it."""

    #: The ceiling, the ``RLIMIT_NPROC`` of the step or of its commands.
    ceiling: int
    #: ``uid`` for a ceiling set before any user namespace is made (a step's own), which counts
    #: every task of the uid on the node on any kernel; for one set inside a namespace made with
    #: the runner's own limit (the build sandbox's commands), the site's ``nproc_accounting``.
    accounting: Accounting
    #: Under ``user_namespace``, the limit the namespace's parent levels count every task of the
    #: uid on the node against: the runner's hard ``RLIMIT_NPROC``, which the namespace was made
    #: with; None when unlimited, and under ``uid``.
    parent_limit: int | None

    @property
    def uid_limit(self) -> int | None:
        """What every task of the uid on the node counts against for the step: its ceiling under
        ``uid``, its namespace's parent levels' limit under ``user_namespace``; None when that is
        unlimited."""
        return self.ceiling if self.accounting == "uid" else self.parent_limit

    def refusal(self, held: int) -> str | None:
        """Why no task under this ceiling could fork while the uid holds *held* tasks on the
        node, or None."""
        limit = self.uid_limit
        if limit is None or held < limit:
            return None
        if self.accounting == "uid":
            return (f"the uid already holds {held} tasks on the node, at or above its commands' process ceiling of "
                    f"{self.ceiling}, so they could fork nothing")
        return (f"the uid already holds {held} tasks on the node, at or above the runner's hard limit of {limit}, "
                "which the parent levels of its commands' user namespace count them against, so they could fork "
                "nothing")


@dataclass(frozen=True)
class PoolAtEnd:
    """The tasks counted when a contestant step was reported, its step and every task of it gone:
    the uid's and the node's, the others' alone, which decide whether it was starved
    (``starvation``). A count that failed is kept as one, with no tasks, never as no count at
    all (``uncounted``)."""

    #: When they were counted, or a count failed.
    at: float
    #: Every task of the uid on the node then (``uid_tasks``); None when they could not be
    #: counted.
    others_tasks: int | None
    #: The step's ceiling and how it was counted.
    rule: CeilingRule
    #: Every task on the node then, any user's (``node_tasks``); None when they could not be
    #: counted.
    node_tasks: int | None
    #: The node's ``pid_max`` (``node_pid_max``); None when it could not be read.
    pid_max: int | None

    @property
    def uncounted(self) -> str | None:
        """What could not be counted of the levels that judge the step, which leaves starvation
        not ruled out (``hls.substrate.faults.starvation_fault``), or None when each was. Under
        ``uid`` the ceiling counts every task of the uid on the node, so the uid's count is the
        level that judges it; the node's, when it was counted too, is judged beside it, and when
        it failed the uid's alone judges. Under ``user_namespace`` the ceiling counts only the
        commands' own tasks, and the others reach them only through the parent levels: the
        uid's count, needed where the runner's hard limit bounds them (``CeilingRule.uid_limit``),
        and the node's against its pid_max, needed always, since it bounds every step."""
        if self.rule.uid_limit is not None and self.others_tasks is None:
            return "the uid's tasks"
        if self.rule.accounting == "user_namespace" and (self.node_tasks is None or self.pid_max is None):
            return "the node's tasks"
        return None


def hard_limit() -> int | None:
    """This process's hard ``RLIMIT_NPROC``; None when unlimited."""
    hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
    return None if hard == resource.RLIM_INFINITY else hard


def raise_soft_limit() -> tuple[int | None, int | None]:
    """Raise this process's soft ``RLIMIT_NPROC`` to its hard one, as the runner does when it
    starts, and return both limits as it found them (None for unlimited)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    return (None if soft == resource.RLIM_INFINITY else soft, None if hard == resource.RLIM_INFINITY else hard)


def node_nproc_max(nproc_max: int | None, evaluations: int) -> int:
    """The cap on a contestant step's ceiling on a node that holds *evaluations* of its size at
    once: the site's *nproc_max*, or for a site that sets none (None) ``DEFAULT_NPROC_MAX``, or
    the evaluations' floor when that is more, so the default holds a node of any size."""
    if nproc_max is not None:
        return nproc_max
    return max(DEFAULT_NPROC_MAX, evaluations * FLOOR_PER_EVALUATION)


def nproc_max_problem(nproc_max: int | None, evaluations: int) -> str | None:
    """Why the site's cap *nproc_max* cannot hold a contestant step of which the node holds
    *evaluations* at once, or None: it is under their floor, ``FLOOR_PER_EVALUATION`` tasks for
    each. The reason names the setting and the least value that holds them. A site that sets no
    cap (None) has none under a floor (``node_nproc_max``)."""
    floor = evaluations * FLOOR_PER_EVALUATION
    if nproc_max is None or nproc_max >= floor:
        return None
    held = ("one evaluation's server" if evaluations == 1
            else f"each of the {evaluations} evaluations' servers the node holds")
    return (f"site.nproc_max of {nproc_max} tasks is under the floor of {floor}, {FLOOR_PER_EVALUATION} for "
            f"{held}; raise site.nproc_max to at least {floor}")


def node_evaluations(node: NodeSize, *, gpus: int, cpus: int) -> int:
    """How many evaluations of *gpus* GPUs and *cpus* CPUs a node of size *node* holds at once: the
    node's GPUs over the step's, or for a step with no GPU its CPUs over the step's. Raise
    ValueError naming why for a node whose count of them is unknown or under the step's: any
    count given for it would be a guess."""
    unit, have, want = ("GPU", node.gpus, gpus) if gpus > 0 else ("CPU", node.cpus, cpus)
    if have is None:
        raise ValueError(f"the node's {unit} count is unknown, so how many evaluations of this size it holds is "
                         "unknown too")
    if have < want:
        raise ValueError(f"the node's {unit} count, {have}, is under the {want} that one evaluation of this size takes")
    return have // want


def node_cap_problem(node: NodeSize, *, nproc_max: int | None, gpus: int, cpus: int) -> str | None:
    """Why no contestant step of *gpus* GPUs and *cpus* CPUs can start on a node of size *node*
    under the site's cap *nproc_max* (None for a site that sets none), whatever the uid's hard
    limit, or None: how many
    evaluations of its size the node holds is unknown (``node_evaluations``), or the cap is under
    their floor (``nproc_max_problem``). The runner checks it for each stage it has to run before
    any runs; ``ProcessCeiling.problem`` says the same, and checks the hard limit too."""
    try:
        evaluations = node_evaluations(node, gpus=gpus, cpus=cpus)
    except ValueError as e:
        return str(e)
    return nproc_max_problem(nproc_max, evaluations)


def process_ceiling(hard: int | None, *, reserve: int, nproc_max: int | None, node: NodeSize, gpus: int,
                    cpus: int) -> ProcessCeiling:
    """The ceiling of a contestant step of *gpus* GPUs and *cpus* CPUs on a node of size *node*:
    the headroom the hard limit *hard* (None for unlimited) leaves with *reserve* tasks kept for
    each evaluation's harness, capped at the site's *nproc_max*, or at the node's default for a
    site that sets none (``node_nproc_max``). One under its floor, or under a
    cap under its floor, is given as it is, with its ``problem``, and so is a node whose
    evaluations of the step's size are unknown, with no ceiling (``evaluations_unknown``)."""
    try:
        evaluations = node_evaluations(node, gpus=gpus, cpus=cpus)
    except ValueError as e:
        return ProcessCeiling(hard=hard, reserve=reserve, evaluations=None, evaluations_unknown=str(e),
                              uncapped=None, nproc_max=node_nproc_max(nproc_max, 1), ceiling=None, floor=None)
    uncapped = None if hard is None else hard - evaluations * reserve
    cap = node_nproc_max(nproc_max, evaluations)
    return ProcessCeiling(hard=hard, reserve=reserve, evaluations=evaluations, evaluations_unknown=None,
                          uncapped=uncapped, nproc_max=cap,
                          ceiling=cap if uncapped is None else min(uncapped, cap),
                          floor=evaluations * FLOOR_PER_EVALUATION)


def uid_tasks(uid: int, *, proc: Path = Path("/proc")) -> int:
    """Every task of real uid *uid* on the node, as the kernel counts them against
    ``RLIMIT_NPROC``: the threads of each process in *proc* whose real uid it is. A process that
    ends while it is read counts nothing, and so does one whose status cannot be read. Only the
    kernel's own ``Uid`` and ``Threads`` values are decoded (``hls.substrate.procfs``): a process
    names itself, any user's or a contestant's, and no name can stop the count. Raise OSError when
    *proc* cannot be listed."""
    total = 0
    for name in os.listdir(proc):
        if not name.isdecimal():
            continue
        try:
            fields = status_fields(proc, name, (b"Uid", b"Threads"))
        except OSError:
            continue
        if fields is None:
            continue
        try:
            real = int(fields[b"Uid"].split()[0])
            threads = int(fields[b"Threads"])
        except (KeyError, IndexError, ValueError):
            continue
        if real == uid:
            total += threads
    return total


def step_tasks(cgroup_dir: Path) -> int | None:
    """A step's own tasks: the pids controller's count in its cgroup *cgroup_dir* or its parent,
    or else the threads listed in every cgroup of *cgroup_dir*'s tree; None when none of them
    can be read."""
    for directory in (cgroup_dir, cgroup_dir.parent):
        try:
            return read_int(directory / "pids.current")
        except (OSError, ValueError):
            continue
    listed = None
    for threads in sorted(cgroup_dir.rglob("cgroup.threads")):
        try:
            count = sum(1 for line in threads.read_text().splitlines() if line.strip())
        except OSError:
            continue
        listed = (listed or 0) + count
    return listed


def starvation(at_end: PoolAtEnd) -> str | None:
    """Why the contestant step whose pool was counted at its end as *at_end* was starved, or
    None. It was when, at a level that counts its tasks, the others' leave it at most what
    ``POOL_FULL`` of its ceiling leaves: the uid's on the node, against its ceiling under ``uid``
    or the runner's hard limit under ``user_namespace`` (``CeilingRule.uid_limit``), and the
    node's against its ``pid_max``, which bounds every step on any kernel, each level as it was
    counted (``PoolAtEnd.uncounted`` says which a judgement needs). Then its end is infra.
    Raise ValueError for a pool not counted (``PoolAtEnd.uncounted``), which nothing rules out."""
    missing = at_end.uncounted
    if missing is not None:
        raise ValueError(f"{missing} were not counted, so starvation cannot be judged")
    rule = at_end.rule
    left = rule.ceiling - math.ceil(POOL_FULL * rule.ceiling)
    limit, others = rule.uid_limit, at_end.others_tasks
    if limit is not None and others is not None and limit - others <= left:
        if rule.accounting == "uid":
            return (f"the uid's other tasks on the node filled its process pool, {others} of its ceiling "
                    f"{rule.ceiling}")
        return (f"the uid's other tasks on the node, {others}, left it {limit - others} of the runner's hard limit "
                f"{limit}, which the parent levels of its user namespace count them against, under its ceiling "
                f"{rule.ceiling}")
    if at_end.node_tasks is not None and at_end.pid_max is not None and at_end.pid_max - at_end.node_tasks <= left:
        return (f"the node's tasks, {at_end.node_tasks} of its pid_max {at_end.pid_max}, left it "
                f"{at_end.pid_max - at_end.node_tasks} under its ceiling {rule.ceiling}")
    return None


def node_tasks(loadavg: Path = Path("/proc/loadavg")) -> int | None:
    """Every task on the node, any user's, as the kernel counts its threads: the total of
    ``/proc/loadavg``'s running/total field; None when it cannot be read."""
    try:
        return int(loadavg.read_text().split()[3].split("/")[1])
    except (OSError, ValueError, IndexError):
        return None


def node_pid_max(path: Path = Path("/proc/sys/kernel/pid_max")) -> int | None:
    """The node's ``pid_max``; None when it cannot be read."""
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
