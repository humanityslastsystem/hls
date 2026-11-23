"""The limits on the kernel's per-uid counts each sealed contestant role, and the build sandbox's
commands within one, may hold: namespaces of each kind, inotify instances and inotify watches.

The kernel counts the namespaces, inotify instances and inotify watches a process makes against
its uid at its own user namespace's level and at every level above it, each against that level's
limit (``/proc/sys/user/max_*``, ucounts). The node's own level, whose limits are the node's
(``fs.inotify.max_user_instances`` and ``fs.inotify.max_user_watches`` are the same counts there),
is shared by every process of the uid: the runner, the harness of every evaluation the node holds,
and every contestant's. A process that made namespaces, or opened inotify instances or watches,
until refused would leave every other role of the uid on the node unable to make its own: a
sandbox's spawner its namespaces, an agent's CLI the instance and the watches it follows files
with. Those failures read as infra, or as the other contestant's own end, and a contestant could
force them at will. The agent's CLI runs among the build sandbox's commands, in their namespaces
(``hls.agent.memory``), so its instance and its watches, one per directory of the tree it runs in,
count in the commands' limit with their tools'.

So each count gets two limits of its own, written into user namespaces before the contestant's
code runs (``ucount_limits``):
- (b) ``role``: each sealed contestant role's, written in the seal's (pasta's) user namespace,
  which holds everything of the role, an agent's CLI and a build sandbox's commands among it:
  ``(node - k * outside) // k``, the node's limit shared by the k evaluations it holds once each
  has kept ``outside`` for its harness outside the role, so no role can take another's share;
- (a) ``commands``: the build sandbox's commands', the CLI's among them, written in their user
  namespace by the spawner's reaper (``hls.agent.spawner``): ``role - inside``, what the role's
  share leaves once the sandbox's own harness beside the commands (apptainer, the exec server,
  the spawner) has kept ``inside``, so the commands cannot starve that harness either; less, a
  site's cap, only where the site sets one. An agent gets its whole share of the node: nothing
  smaller is imposed.
k is the stage's evaluations (``hls.substrate.nproc.ProcessCeiling.evaluations``). A count the
node's kernel has no limit for (no file) is not limited. On an el8 kernel a
limit written to a namespace's own file binds there (EMFILE for inotify
instances, ENOSPC for user namespaces) while the node's level still counts what was made under
it. A commands' limit under its floor refuses the
stage, infra and not retried, naming why; a site's cap under its floor refuses the plan
(``cap_problem``).

A graded server's sealed role gets its (b) from no node's reading: the build's brief states it on
the build's node, and the server is graded on another, whose own limits may differ (a node's
limit scales with its memory). So its limits come from what the site declares
of its grading nodes' own limits (``site.grading_node_*``), shared the same way among the k
evaluations of the served artifact's size the build's node holds (``served_limits``). They are
computed once, when the build renders its brief, and recorded with the artifact (``ServedUcounts``:
the build's ``served_ucounts`` output, which the package carries in ``package.json``): the brief
states them and the grading seal writes them. A grading node whose own limits cannot hold its own
k such roles beside their harnesses is refused, infra and retried (``node_shortfall``). A role's
limits reach its seal through its environment (``ucounts_env``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: The site setting that caps a count's commands' limit (``hls.settings.SiteSettings``).
CapSetting = Literal["sandbox_namespaces", "sandbox_inotify_instances", "sandbox_inotify_watches"]
#: The site setting that declares a grading node's own limit of a count (``hls.settings.SiteSettings``),
#: which a graded server's share is computed from (``served_limits``).
NodeSetting = Literal["grading_node_namespaces", "grading_node_inotify_instances", "grading_node_inotify_watches"]
#: The variable the seal reads its role's limits from (``ucounts_env``): ``NAME=LIMIT`` pairs, one
#: per count, separated by spaces, each NAME a file of ``/proc/sys/user``.
SEAL_UCOUNTS_VAR = "HLS_SEAL_UCOUNTS"
#: Namespaces each evaluation's harness holds outside its sealed role: the seal's own (pasta's
#: user, network and mount namespaces) and a harness container's: 16 is more than three times
#: the namespaces of each kind a sandbox's whole step makes.
NAMESPACE_OUTSIDE = 16
#: Namespaces the build sandbox's own harness holds inside its role beside the commands':
#: apptainer's and the spawner's, the commands' own among them; bounded as above.
NAMESPACE_INSIDE = 16
#: The fewest namespaces of each kind a sandbox's commands start with room for: the spawner's own
#: mount and PID namespaces count in theirs, and a command's tools make a few more.
NAMESPACE_FLOOR = 16
#: inotify instances each evaluation's harness holds outside its sealed role: none. The runner,
#: the interceptor, pasta (which watches only a filesystem-bound namespace, and falls back
#: to a timer) and harbor open no inotify instance.
INOTIFY_OUTSIDE = 0
#: inotify instances the build sandbox's own harness holds inside its role beside the commands:
#: the exec server and the spawner hold none; one is kept for apptainer's starter, which
#: links inotify calls.
INOTIFY_INSIDE = 1
#: The fewest inotify instances a sandbox's commands start with room for: the most an agent's
#: CLI (Claude Code, codex) holds working in a sandbox.
INOTIFY_FLOOR = 2
#: inotify watches each evaluation's harness holds outside its sealed role.
WATCHES_OUTSIDE = 0
#: inotify watches the build sandbox's own harness holds inside its role beside the commands:
#: none of their own; kept for apptainer's starter.
WATCHES_INSIDE = 1_024
#: The fewest inotify watches a sandbox's commands start with room for. In a sandbox neither CLI
#: watches ``/workspace``'s tree, so the floor is the agent's own tools' room.
WATCHES_FLOOR = 8_192


@dataclass(frozen=True)
class Ucount:
    """One per-uid count each sealed role, and the build sandbox's commands, get a limit for."""

    #: Its file in ``/proc/sys/user``, the node's and each namespace's.
    name: str
    #: The site setting that caps the commands' limit.
    setting: CapSetting
    #: The site setting that declares a grading node's own limit of it.
    declared: NodeSetting
    #: What each evaluation's harness keeps of the node's limit outside its sealed role.
    outside: int
    #: What the build sandbox's own harness keeps of its role's limit beside the commands.
    inside: int
    #: The least commands' limit a sandbox starts under.
    floor: int


NAMESPACE_KINDS = ("user", "pid", "mnt", "net", "ipc", "uts", "cgroup", "time")
#: Every count the roles and the commands are limited in.
UCOUNTS: tuple[Ucount, ...] = (
    *(Ucount(f"max_{kind}_namespaces", "sandbox_namespaces", "grading_node_namespaces", NAMESPACE_OUTSIDE,
             NAMESPACE_INSIDE, NAMESPACE_FLOOR) for kind in NAMESPACE_KINDS),
    Ucount("max_inotify_instances", "sandbox_inotify_instances", "grading_node_inotify_instances", INOTIFY_OUTSIDE,
           INOTIFY_INSIDE, INOTIFY_FLOOR),
    Ucount("max_inotify_watches", "sandbox_inotify_watches", "grading_node_inotify_watches", WATCHES_OUTSIDE,
           WATCHES_INSIDE, WATCHES_FLOOR),
)
#: Where a node's own limits are, read on the host.
NODE_LIMITS = Path("/proc/sys/user")


class UcountLimits(BaseModel):
    """The limits one count gets, and what they were computed from, for the record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The count's file in ``/proc/sys/user``.
    name: str
    #: The node's own limit.
    node: int
    #: The evaluations of the stage's size the node holds.
    evaluations: int = Field(ge=1)
    #: What each evaluation's harness keeps outside its sealed role (``Ucount.outside``).
    outside: int = Field(ge=0)
    #: (b) each sealed role's limit: ``(node - evaluations * outside) // evaluations``.
    role: int
    #: The setting that may cap the commands' limit.
    setting: CapSetting
    #: Its value; None where the site sets none.
    cap: int | None = Field(default=..., ge=1)
    #: What the build sandbox's own harness keeps inside its role (``Ucount.inside``).
    inside: int = Field(ge=0)
    #: (a) the build sandbox's commands' limit: ``role - inside``, or the cap where it is less.
    commands: int
    #: The least commands' limit a sandbox starts under.
    floor: int

    @property
    def problem(self) -> str | None:
        """Why no sandbox may start under these limits, or None."""
        if self.commands >= self.floor:
            return None
        return (f"the node's {self.name} of {self.node}, shared by {self.evaluations} evaluations that each keep "
                f"{self.outside} for their harness, leaves each sealed role {self.role} and a sandbox's commands "
                f"{self.commands} once its own harness keeps {self.inside}, under the floor of {self.floor}; raise the "
                "node's limit, or run fewer evaluations on it")


def node_limits(root: Path = NODE_LIMITS) -> dict[str, int]:
    """The node's own limit of each count in ``UCOUNTS`` its kernel has. Raise OSError or
    ValueError when one it has cannot be read."""
    found: dict[str, int] = {}
    for u in UCOUNTS:
        try:
            text = (root / u.name).read_text()
        except FileNotFoundError:
            continue
        found[u.name] = int(text.split()[0])
    return found


def ucount_limits(node: Mapping[str, int], caps: Mapping[CapSetting, int | None],
                  evaluations: int) -> tuple[UcountLimits, ...]:
    """The limits of each count *node* has (``node_limits``) for the sealed roles and the
    commands of each of the *evaluations* a node holds, under the site's *caps*, each None where
    the site sets none."""
    made = []
    for u in UCOUNTS:
        if u.name not in node:
            continue
        role = (node[u.name] - evaluations * u.outside) // evaluations
        cap = caps[u.setting]
        made.append(UcountLimits(name=u.name, node=node[u.name], evaluations=evaluations, outside=u.outside,
                                 role=role, setting=u.setting, cap=cap, inside=u.inside,
                                 commands=role - u.inside if cap is None else min(cap, role - u.inside),
                                 floor=u.floor))
    return tuple(made)


def commands_limits(limits: tuple[UcountLimits, ...]) -> dict[str, int]:
    """(a): each count's limit for the build sandbox's commands, by its file's name, for the
    spawner's ``--ucount``."""
    return {c.name: c.commands for c in limits}


def ucounts_env(limits: Mapping[str, int]) -> dict[str, str]:
    """The variable that gives a sealed role's seal the limit of each count in *limits*, by its
    file's name (``SEAL_UCOUNTS_VAR``), for the seal to write into its user namespace's
    ``/proc/sys/user`` before it starts the payload: the stage's own (b) (``UcountLimits.role``),
    or a graded server's (``ServedLimit.role``)."""
    return {SEAL_UCOUNTS_VAR: " ".join(f"{name}={limit}" for name, limit in limits.items())}


class ServedLimit(BaseModel):
    """The limit a graded server's sealed role is held to on one count, and what it was computed
    from, for the record (``served_limits``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The count's file in ``/proc/sys/user``.
    name: str
    #: The site setting that declares a grading node's own limit of it.
    setting: NodeSetting
    #: Its value.
    declared: int = Field(ge=1)
    #: The evaluations of the served artifact's size a grading node holds.
    evaluations: int = Field(ge=1)
    #: What each evaluation's harness keeps outside its sealed role (``Ucount.outside``).
    outside: int = Field(ge=0)
    #: (b) the served role's limit: ``(declared - evaluations * outside) // evaluations``.
    role: int
    #: The least limit any sealed contestant role starts under (``Ucount.floor``): room for the
    #: namespaces its own runtime makes and for the tools it runs.
    floor: int

    @property
    def problem(self) -> str | None:
        """Why no graded server of this size may start under the site's declaration, or None."""
        if self.role >= self.floor:
            return None
        return (f"site.{self.setting} of {self.declared}, shared by {self.evaluations} graded servers that each keep "
                f"{self.outside} for their harness, leaves each {self.role} {self.name}, under the floor of "
                f"{self.floor}; raise site.{self.setting} if the site's grading nodes hold more, or grade fewer on "
                "a node")


def served_limits(declared: Mapping[NodeSetting, int], evaluations: int) -> tuple[ServedLimit, ...]:
    """(b) for a graded server: each count's share of a grading node, as the site declares the
    node's own limits (*declared*, by setting), among the *evaluations* of the served artifact's
    size the node holds, for every count in ``UCOUNTS``. No node is read: the build's brief states
    these on one node and the grading seal writes them on another."""
    return tuple(ServedLimit(name=u.name, setting=u.declared, declared=declared[u.declared], evaluations=evaluations,
                             outside=u.outside, role=(declared[u.declared] - evaluations * u.outside) // evaluations,
                             floor=u.floor)
                 for u in UCOUNTS)


class ServedUcounts(BaseModel):
    """The limits a build's graded server is held to on each per-uid count, computed once when the
    build renders its brief (``hls.pipeline.stage.StageContext.served_ucounts``) and recorded with its
    artifact: the build's ``served_ucounts`` output, which the package carries in ``package.json``
    (``hls.package.result.PackageResult.served_ucounts``) and the grading seal writes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Each count's limit and what it was computed from.
    limits: tuple[ServedLimit, ...] = Field(min_length=1)

    def figures(self) -> dict[str, int]:
        """Each count's limit by its ``/proc/sys/user`` file's name."""
        return {c.name: c.role for c in self.limits}


def node_shortfall(limits: Sequence[ServedLimit], node: Mapping[str, int], evaluations: int) -> str | None:
    """Why a grading node whose own limits are *node* (``node_limits``) and which holds
    *evaluations* graded servers of the size at once cannot give each its share in *limits*: a
    count it has whose limit is under what those servers and their harnesses take together,
    ``evaluations * (role + outside)``; None when it holds every one. A count the node's kernel has
    no limit for cannot be made there, so it needs no room."""
    for c in limits:
        if c.name in node and node[c.name] < evaluations * (c.role + c.outside):
            return (f"this node's {c.name} of {node[c.name]} cannot hold {evaluations} graded servers of {c.role} "
                    f"each beside the {c.outside} each harness keeps")
    return None


def cap_problem(setting: CapSetting, cap: int | None) -> str | None:
    """Why the site's *cap* for *setting* can start no sandbox, or None: it is under the floor
    of a count it caps. No cap is none."""
    floor = max(u.floor for u in UCOUNTS if u.setting == setting)
    if cap is None or cap >= floor:
        return None
    return f"site.{setting} of {cap} is under the floor of {floor}; raise site.{setting} to at least {floor}"
