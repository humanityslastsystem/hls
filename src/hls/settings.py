"""Strict settings for the orchestrator, read from ``hls.toml``.

``[site]`` describes the cluster and where the store lives. ``[run]`` holds run
policy: GPU grouping, retries, heartbeats. ``[resources.<task>.<stage>]`` changes the fields
it names of a stage's declared shape (``ShapeOverride``). ``[routes.<name>]`` declares how an
agent reaches its model, by the
names of secrets, and ``[arms.<name>]`` an agent arm that names its route (``hls.arms``).
``[sandbox_stores]`` names the directory of each store a task's sandboxes start from
(``Settings.sandbox_stores``).
Every key is checked: a misspelled one fails, naming its full dotted path, instead of being
ignored.

A top-level key outside these six tables is an error."""

from __future__ import annotations

import json
import os
import posixpath
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, ValidationError,
                      ValidationInfo, field_validator, model_serializer, model_validator)
from pydantic_core import PydanticCustomError

from hls.arms import Arm, ArmName, Route, RouteName, check_arm_route
from hls.spec.resources import GpuKind, ResourceShape
from hls.spec.tools import DEVICE
from hls.store.base import check_segment
from hls.store.digest import sha256_bytes
from hls.store.layout import check_stage
from hls.util import TomlError, load_toml

#: A stage name as a settings key, checked like a stage name in the store layout.
StageName = Annotated[str, AfterValidator(check_stage)]
#: A task name as a settings key, checked as one store key segment.
TaskName = Annotated[str, AfterValidator(check_segment)]
# A key TOML can write unquoted.
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
#: One sbatch argument as ``site.sbatch_extra`` writes it: "--<name>=<value>", the value printable
#: ASCII without a space that does not start with "=".
SBATCH_ARG = re.compile(r"(--[a-z][a-z-]*)=([!-<>-~][!-~]*)")


class SettingsError(ValueError):
    """hls.toml or an override is invalid."""


#: A sandbox store's name (``Settings.sandbox_stores``, ``hls.spec.tools.HostPath.store``):
#: lowercase letters, digits and hyphens, starting with a letter.
SANDBOX_STORE_NAME = re.compile(r"[a-z][a-z0-9-]*")


def check_sandbox_store_name(value: str) -> str:
    """*value* when it can name a sandbox store (``SANDBOX_STORE_NAME``, and not
    ``hls.spec.tools.DEVICE``, which names a device node); else raise ValueError naming the rule."""
    if not SANDBOX_STORE_NAME.fullmatch(value):
        raise ValueError(f"a sandbox store's name is lowercase letters, digits and hyphens, starting with a letter, "
                         f"not {value!r}")
    if value == DEVICE:
        raise ValueError(f"{DEVICE!r} names a device node of a tool runner's host paths, never a store")
    return value


#: A sandbox store's name as a settings key.
SandboxStoreName = Annotated[str, AfterValidator(check_sandbox_store_name)]


def _store_directory(value: Path) -> Path:
    """*value*, ``~`` expanded, when it is an absolute directory path in normal form other than the
    root; else raise ValueError. A store is bound into a container as written, so its path must
    say the directory it names."""
    expanded = os.path.expanduser(str(value))
    if not expanded.startswith("/") or expanded.startswith("//") or posixpath.normpath(expanded) != expanded:
        raise ValueError(f"a sandbox store is an absolute path in normal form after ~ expansion, got {str(value)!r}")
    if expanded == "/":
        raise ValueError("a sandbox store cannot be the root directory")
    return Path(expanded)


class RetryPolicy(BaseModel):
    """How a stage is retried after an infrastructure fault."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Retries after the first attempt; 0 means none. None takes the stage kind's default
    #: (``hls.pipeline.stage.Stage.default_retries``).
    max: int | None = Field(default=None, ge=0)
    #: False turns retries off for the stage whatever *max* says.
    enabled: bool = True
    #: Seconds to wait before resubmitting.
    backoff_sec: int = Field(default=0, ge=0)


#: The BuildKit image a Docker host's image builds run: BuildKit v0.35.1, pinned by its digest.
DEFAULT_BUILDKIT_IMAGE = "moby/buildkit@sha256:baf6138eb8c37644800c7c913fba051e840ebb9ba13eaf92d92ef9a1712d0b4c"


class DockerSiteSettings(BaseModel):
    """How a Docker host runs roles (``hls.substrate.docker``), ``[site.docker]``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The Docker daemon's Unix socket.
    socket: Path = Path("/var/run/docker.sock")
    #: The image a build role's BuildKit builder runs, which buildx starts privileged: an image ID
    #: (``sha256:<hex>``) or a name pinned by its digest (``moby/buildkit@sha256:<hex>``), never a
    #: tag alone, which a later push or another local image of that name would change. The default
    #: is the BuildKit image hls is tested with.
    buildkit_image: str = Field(
        default=DEFAULT_BUILDKIT_IMAGE,
        pattern=r"^(sha256:[0-9a-f]{64}|[A-Za-z0-9][A-Za-z0-9._/:-]*@sha256:[0-9a-f]{64})$")
    #: A label, ``key=value``, every role container carries beside the job's, such as the one a
    #: host's GPU manager reads to tell hls's containers from others; None for none.
    owner_label: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*=[!-~]*$")
    #: GiB the gateway of each sealed role holds beside the role's own memory, in the allocation's:
    #: its interceptor and, for a build's seal, the package index it serves. A stage that seals holds
    #: it beside its roles, and the planner counts it with the runner's reserve.
    gateway_memory_gib: int = Field(default=2, ge=1)

    @field_validator("socket")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return _absolute_path(v)


class LocalSiteSettings(BaseModel):
    """How the local scheduler places allocations on one host (``hls.substrate.local``),
    ``[site.local]``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The scheduler's state directory, this user's own (mode 0700) on a local disk: its job
    #: records, its reservations and the supervisors' logs.
    state_dir: Path
    #: GiB of the host's memory no allocation is given: what the host's own processes, the daemon
    #: and each runner outside its containers need.
    kept_memory_gib: float = Field(ge=0)

    @field_validator("state_dir")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return _absolute_path(v)


def _absolute_path(v: Path) -> Path:
    # Login and compute nodes start in different directories, so a relative path would name
    # different places. An unknown ~user is left as is and fails here.
    expanded = Path(os.path.expanduser(v))
    if not expanded.is_absolute():
        raise ValueError(f"must be an absolute path after ~ expansion, got {str(v)!r}")
    return expanded


#: The ``[site]`` settings only a Slurm site has. A Docker site sets none of them.
SLURM_SITE_FIELDS = ("partition", "qos", "qos_by_hours", "account", "gpu_gres", "min_gpus_per_job",
                     "evaluation_socket_tasks", "sbatch_extra", "cpu_partition", "pasta_path")


class SiteSettings(BaseModel):
    """The cluster and the store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: What the site runs allocations on (``hls.substrate.factory.SUBSTRATES``): ``slurm``, a
    #: cluster's Slurm with Apptainer, or ``docker``, one host's Docker. The default is Slurm, so an
    #: hls.toml that names none describes a Slurm cluster. Declared before the fields that hold only on
    #: one substrate, so their checks can read it.
    substrate: Literal["slurm", "docker"] = "slurm"
    #: GPU partition every run allocation is submitted to; a Slurm site's, which it requires.
    partition: str | None = Field(default=None, validate_default=True)
    #: Slurm QOS; None uses the account default. Under ``qos_by_hours`` it must be one of its
    #: QOS, and an allocation it does not hold the time of runs under another.
    qos: str | None = None
    #: Each QOS a job on this site may run under, to the most hours a job may ask under it, such
    #: as ``{normal = 24, long = 168}``. The planner plans each allocation under
    #: ``qos`` while that holds its time, else under the QOS with the least limit that holds it
    #: (``hls.substrate.slurm.resources.allocation_qos``). An allocation the GPU policy groups that
    #: none holds is split into a chain of allocations the longest limit holds
    #: (``hls.pipeline.planner.policy_layout``); a stage none holds alone, and an allocation a
    #: pipeline lays out itself that none holds, are refused. Empty for a site whose QOS sets no
    #: limit: every allocation runs under ``qos``.
    qos_by_hours: dict[str, Annotated[float, Field(gt=0)]] = Field(default_factory=dict)
    #: Slurm account; None uses the user default.
    account: str | None = None
    #: Root directory of the store on the shared filesystem. The cache lives under it.
    store_root: Path
    #: Node-local scratch on compute nodes, where the runner stages images and weights.
    node_scratch: Path
    #: Cluster-local secrets file (mode 0600).
    secrets_path: Path
    #: The user name of the Docker Hub login a base image's build pulls its CUDA image with;
    #: None pulls anonymously. It is not a secret: the build role gets it in its declared
    #: environment, and only the token is granted, as REGISTRY_PASSWORD to the images stage in
    #: the secrets file. A user name that is the cluster login name appears in every path under
    #: the user's directories, so as a secret it would refuse every role naming one.
    registry_username: str | None = Field(default=None, pattern=r"^[!-~]+$")
    #: GPU kind to Slurm gres name.
    gpu_gres: dict[GpuKind, str] = Field(
        default_factory=lambda: {
            "H200": "h200",
            "H100": "h100",
            "B200": "b200",
            "A100": "a100",
            "L40": "l40",
            "L40S": "l40s",
        }
    )
    #: Smallest GPU count a job may request on this cluster (0 when CPU-only jobs are allowed).
    min_gpus_per_job: int = Field(default=0, ge=0)
    #: True holds a scored server to one socket. An allocation that runs one (a deploy_drive
    #: stage) asks for two tasks of the server's CPUs with at most one task per socket
    #: (``--ntasks=2 --ntasks-per-socket=1``), so it holds twice the server's CPUs and the
    #: server's step, which takes one task's CPUs, sits on one socket; a server whose CPUs still
    #: lie on two NUMA nodes is refused as infra and retried. False asks for the server's CPUs as
    #: one task, wherever Slurm finds them, and records where they lie without refusing. A server
    #: whose CPUs split across sockets runs slower than one on a single socket, and Slurm's own options
    #: (--ntasks-per-socket with one task, --sockets-per-node, --gres-flags=enforce-binding) do not
    #: keep one task's CPUs on one socket.
    #: A site that allows fewer CPUs per GPU than twice a server's cannot set it.
    #: Nor can a site whose sockets hold more than one NUMA node each (AMD NPS2 or NPS4, Intel
    #: sub-NUMA clustering): the refusal checks NUMA nodes and Slurm keeps a task to a socket only,
    #: so a server's CPUs could span two nodes of one socket and every drive be refused until the
    #: instance voids.
    evaluation_socket_tasks: bool = False
    #: Extra sbatch arguments added to every allocation on ``partition``, and to none on
    #: ``cpu_partition``, which they do not describe. Each is ``--<name>=<value>`` with <name> one of
    #: ``hls.substrate.slurm.resources.SITE_EXTRA_OPTIONS`` (exclude, constraint, reservation, nodelist, nice,
    #: mail-type, mail-user); any other option is refused when the run is planned. sbatch accepts
    #: option prefixes and options that change the cluster, so only listed options are allowed. An
    #: option given twice is refused when the settings load: sbatch would keep only the last.
    sbatch_extra: list[str] = Field(default_factory=list)
    #: CPU partition for the standalone cache warmers; None runs them on the GPU partition.
    cpu_partition: str | None = Field(default=None, min_length=1)
    #: The pasta program the network seal runs; None finds ``pasta`` on the allocation's PATH.
    #: Compute nodes often lack the passt package, so a site names a built copy here.
    pasta_path: Path | None = None
    #: Seconds the planner allows for each role's step to start, beyond the stages' own times: an
    #: allocation asks for each of its stages' most roles (``hls.pipeline.stage.most_roles``) times
    #: this, so the time a busy controller takes to start steps, by which a stage's deadline grows
    #: (``StageContext.deadline``), fits in the allocation; the runner keeps a stage's share free
    #: before starting it. The headroom is a sum over launches, so it is their mean wait that this
    #: must cover. The default, 150 s, is a conservative bound on the mean wait from launch to
    #: grant report on a busy controller; the ``launch_wait_sec`` of ``role_started`` events can
    #: check it for a site.
    step_launch_allowance_sec: int = Field(default=150, ge=0)
    #: An HTTP proxy, ``http://<host>:<port>``, that the network seal's interceptor reaches
    #: the internet through; None dials directly. No role inherits a proxy from the host (the
    #: step wrapper removes every ``*_proxy`` variable from a container's step), so a site
    #: whose compute nodes have no direct route out names its proxy here, without credentials.
    egress_upstream_proxy: str | None = None
    #: The most tasks (processes and threads) of the runner's uid on a node that a contestant's
    #: step may grow the uid's count to, whatever headroom the uid's hard ``RLIMIT_NPROC`` leaves:
    #: a contestant step's ceiling is the lesser of the two (``hls.substrate.nproc``). Three kinds
    #: of step run under it: the served artifact's (deploy_drive), each pip step of the package
    #: stage, and each build command of the agent build's sandbox. A node's hard limit can exceed
    #: twice the kernel's pid_max (4,194,304 at most), with no pids controller at
    #: any level of a job's cgroups, so without this a server that spawned threads without bound
    #: would take every pid of the node, the harness's forks and other users' jobs' included. The
    #: cap must hold the floor of the evaluations of a step's size its node holds
    #: (``hls.substrate.nproc.FLOOR_PER_EVALUATION`` tasks for each). The planner does not know a
    #: node's size, so a plan with a contestant's step is refused only a cap under one
    #: evaluation's floor, which no node can hold. The runner checks the cap against its node's
    #: evaluations of each contestant stage's size when its allocation starts, before any stage
    #: runs, and each stage again when its first contestant step starts; a cap under their floor
    #: is infra that is not retried, naming the value to raise it to. None, the default, sets no
    #: cap of the site's: each node then gets 131,072, the floor of 32 evaluations, or the floor
    #: of the evaluations of the step's size it holds when that is more
    #: (``hls.substrate.nproc.node_nproc_max``), so no node is refused for its size. A GPU node
    #: holds at most 8 evaluations, and a node holds one eight-CPU build per eight CPUs, 32 of
    #: them at 256 CPUs. 131,072 stays far under a node's pids: a quarter of a pid_max of
    #: 4,194,304 is about 1.05 million.
    nproc_max: int | None = Field(default=None, gt=0)
    #: The Slurm job name every cache warmer (``hls provision``, ``hls images build``,
    #: ``hls package``) runs under, which Slurm's ``singleton`` dependency holds them behind.
    warmer_job_name: str = Field(default="hls-warmer", pattern=r"^[A-Za-z0-9._-]+$")
    #: Run the cache warmers one at a time on this cluster, with Slurm's ``singleton``
    #: dependency under ``warmer_job_name``: concurrent cold warmers can leave a node's scratch
    #: under the package stage's floor (``hls.stages.package.min_disk_gib``).
    warmers_one_at_a_time: bool = True
    #: How the node's kernel counts the build sandbox's commands against their process ceiling,
    #: which their spawner sets inside a user namespace it made with the runner's own limit
    #: (``hls.substrate.nproc``): ``uid``, against every task of the uid on the node (kernels
    #: before 5.14, such as el8's), or ``user_namespace``, against the commands' own, their
    #: namespace's parent levels counted against the runner's hard limit and the node against its
    #: pid_max (5.14 and later, such as el9's). It decides what a build's end is judged by, when
    #: its start is refused, and what its brief says. Every build's seal probe checks it against the
    #: node's kernel (``nproc-accounting``) and refuses a mismatch before the agent starts: infra,
    #: not retried, since every build of the site would be judged by the wrong level. Unset, each
    #: stage decides from its node's kernel release (``hls.substrate.nproc.kernel_accounting``) and
    #: its ``nproc_accounting`` event records which it used.
    #: A step's own ceiling, a served artifact's or a package step's, is set before any namespace is
    #: made and counts every task of the uid on either kernel, whatever this says.
    nproc_accounting: Literal["uid", "user_namespace"] | None = None
    #: A cap on the namespaces of each kind (user, PID, mount, network, IPC, UTS, cgroup, time) the
    #: build sandbox's commands, the agent's CLI among them, may hold at once, below their share of
    #: the node (``hls.substrate.ucounts``): their user namespace's own limit of each, which the
    #: spawner's reaper writes before the first command, is that share, the node's own limit
    #: split between the evaluations it holds less what each harness keeps, unless
    #: this is less. None sets no cap. Under ``hls.substrate.ucounts.NAMESPACE_FLOOR`` a plan is refused.
    sandbox_namespaces: int | None = Field(default=None, gt=0)
    #: A cap on the inotify instances the build sandbox's commands may hold at once, as
    #: ``sandbox_namespaces`` is for namespaces. The node's own, ``fs.inotify.max_user_instances``, is
    #: 128 by the kernel's default: eight evaluations of a node get 16 each. None sets no cap. Under
    #: ``hls.substrate.ucounts.INOTIFY_FLOOR`` a plan is refused.
    sandbox_inotify_instances: int | None = Field(default=None, gt=0)
    #: A cap on the inotify watches the build sandbox's commands may hold at once, as
    #: ``sandbox_namespaces`` is for namespaces: the agent's CLI watches every directory of
    #: ``/workspace`` with them. The node's own, ``fs.inotify.max_user_watches``, is 1,048,576 on
    #: a 5.11 or later kernel with ample memory: eight evaluations of a node get 131,072 each.
    #: None sets no cap. Under ``hls.substrate.ucounts.WATCHES_FLOOR`` a plan is refused.
    sandbox_inotify_watches: int | None = Field(default=None, gt=0)
    #: The per-uid limit of each namespace kind (``/proc/sys/user/max_<kind>_namespaces``) the
    #: site's nodes that grade a served artifact hold at least. A graded server's sealed role is held
    #: to its share of it, ``(this - k * NAMESPACE_OUTSIDE) // k`` for the k evaluations of the
    #: served artifact's size the build's node holds (``hls.substrate.ucounts.served_limits``),
    #: computed once when the build renders its brief and recorded with the artifact, which the
    #: grading seal writes: never a node's own reading, since the brief is rendered on the build's
    #: node and the server is graded on another. A grading node whose own limit cannot hold its own k
    #: such shares beside their harnesses is refused, infra and retried
    #: (``StageContext.grading_ucounts``). The default, 2^20, must stay under each grading node's
    #: own limit, which the kernel sets to half its max_threads, about one per 256 KiB of memory,
    #: so it holds on any node of 256 GiB or more; at 8 a node, each share is 131,056.
    grading_node_namespaces: int = Field(default=1_048_576, gt=0)
    #: The per-uid limit of inotify instances (``fs.inotify.max_user_instances``) on the site's
    #: nodes that grade a served artifact, the least of them, which a graded server's share is
    #: computed from as ``grading_node_namespaces`` is for namespaces. 128 is the kernel's default.
    grading_node_inotify_instances: int = Field(default=128, gt=0)
    #: The per-uid limit of inotify watches (``fs.inotify.max_user_watches``) on the site's nodes
    #: that grade a served artifact, the least of them, which a graded server's share is computed
    #: from as ``grading_node_namespaces`` is for namespaces. 1,048,576 is a 5.11 or later kernel's
    #: default with ample memory.
    grading_node_inotify_watches: int = Field(default=1_048_576, gt=0)

    #: The Docker host's settings; set exactly when ``substrate`` is ``docker``.
    docker: DockerSiteSettings | None = None
    #: The local scheduler's settings; set exactly when ``substrate`` is ``docker``.
    local: LocalSiteSettings | None = None

    @field_validator("partition", mode="after")
    @classmethod
    def _slurm_partition(cls, v: str | None, info: ValidationInfo) -> str | None:
        # A Slurm site requires it, with the error a missing field gives; a Docker site's is
        # refused with the other Slurm settings (_tables_of_the_substrate).
        if info.data.get("substrate") == "slurm" and v is None:
            raise PydanticCustomError("missing", "Field required")
        return v

    @field_validator("store_root", "node_scratch", "secrets_path")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        # Login and compute nodes start in different directories, so a relative path
        # would name different places. An unknown ~user is left as is and fails here.
        expanded = Path(os.path.expanduser(v))
        if not expanded.is_absolute():
            raise ValueError(f"must be an absolute path after ~ expansion, got {str(v)!r}")
        return expanded

    @field_validator("pasta_path")
    @classmethod
    def _absolute_or_none(cls, v: Path | None) -> Path | None:
        return None if v is None else cls._absolute(v)

    @field_validator("sbatch_extra")
    @classmethod
    def _each_option_once(cls, v: list[str]) -> list[str]:
        # sbatch keeps the last of an option given twice, so the others would be dropped unseen.
        # Arguments that are not --<name>=<value> are refused when a plan is made.
        names = [m.group(1) for arg in v if (m := SBATCH_ARG.fullmatch(arg))]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"it gives {', '.join(repeated)} more than once; sbatch keeps only the last, so give "
                             "each option once")
        return v

    @field_validator("egress_upstream_proxy")
    @classmethod
    def _proxy_url(cls, v: str | None) -> str | None:
        # The interceptor's own rule; imported here so loading settings does not load httpx.
        from hls.egress.proxy import check_upstream_proxy

        return check_upstream_proxy(v)

    @model_validator(mode="before")
    @classmethod
    def _docker_table_defaults(cls, data: object) -> object:
        # Every key of [site.docker] has a default, so a Docker site may leave the table out.
        if isinstance(data, dict) and data.get("substrate") == "docker" and data.get("docker") is None:
            return {**data, "docker": {}}
        return data

    @model_validator(mode="after")
    def _tables_of_the_substrate(self) -> SiteSettings:
        if self.substrate == "docker":
            slurm = [f"site.{name}" for name in SLURM_SITE_FIELDS if name in self.model_fields_set]
            if slurm:
                raise ValueError(f"{', '.join(slurm)} are Slurm settings, which a site whose site.substrate is "
                                 "docker does not set")
            missing = [f"site.{name}" for name in ("docker", "local") if getattr(self, name) is None]
            if missing:
                raise ValueError(f"a site whose site.substrate is docker needs {' and '.join(missing)}")
        else:
            other = [f"site.{name}" for name in ("docker", "local") if getattr(self, name) is not None]
            if other:
                raise ValueError(f"{' and '.join(other)} are for a site whose site.substrate is docker, not "
                                 f"{self.substrate}")
        return self

    @model_validator(mode="after")
    def _qos_has_a_limit(self) -> SiteSettings:
        if self.qos_by_hours and self.qos is not None and self.qos not in self.qos_by_hours:
            raise ValueError(f"site.qos {self.qos!r} is not one of site.qos_by_hours {sorted(self.qos_by_hours)}, "
                             "which must name every QOS a job may run under with its limit")
        return self

    @model_serializer(mode="wrap")
    def _default_substrate_unwritten(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """The settings as a plan and a manifest's digest hold them, without ``substrate`` when it is
        the default and without a Docker or local table a site does not set, so a Slurm site's
        resolved settings and their digest are what they are without those fields; and a Docker
        site's without the Slurm settings, which it never sets."""
        data = handler(self)
        if self.substrate == "slurm":
            del data["substrate"]
        else:
            # A Docker site sets none of them, and a plan reloads its settings from this dump.
            for name in SLURM_SITE_FIELDS:
                data.pop(name, None)
        for name in ("docker", "local"):
            if data.get(name) is None:
                data.pop(name, None)
        return data


class ShapeOverride(BaseModel):
    """A site's change to a stage's declared shape, ``[resources.<task>.<stage>]``: the fields
    it sets replace the declaration's, and the rest are kept."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: GPUs; 0 also clears the GPU kind.
    gpus: int | None = Field(default=None, ge=0)
    #: Accelerator kind from ``GPU_KINDS``.
    gpu_kind: GpuKind | None = None
    #: CPU cores.
    cpus: int | None = Field(default=None, ge=1)
    #: Memory limit in GiB.
    memory_gib: int | None = Field(default=None, ge=1)
    #: Free local disk in GiB the stage needs on the node.
    disk_gib: int | None = Field(default=None, ge=0)
    #: Wall-clock limit in seconds.
    time_sec: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _sets_something(self) -> ShapeOverride:
        # A plan holds its settings as JSON, every unset field null, so set means not None.
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("a resources table sets at least one of gpus, gpu_kind, cpus, memory_gib, disk_gib "
                             "and time_sec")
        return self

    def apply(self, shape: ResourceShape) -> ResourceShape:
        """*shape* with the fields this sets. Raise ValueError when the result is no valid shape,
        such as GPUs without a kind."""
        fields = self.model_dump(exclude_none=True)
        if fields.get("gpus") == 0:
            fields["gpu_kind"] = None
        return ResourceShape.model_validate({**shape.model_dump(), **fields})


class RunSettings(BaseModel):
    """Run policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: "max" puts a run in one allocation sized to its largest stage; "split" starts a
    #: new allocation at each change in GPU count. Either way an allocation no QOS of the site
    #: holds (``SiteSettings.qos_by_hours``) is split into a chain the longest limit holds.
    gpu_policy: Literal["max", "split"] = "max"
    #: Stage name to retry policy; stages not listed get the defaults. Only the name's
    #: characters are checked here; checking names against the run's stages comes with plan
    #: building.
    retry: dict[StageName, RetryPolicy] = Field(default_factory=dict)
    #: Seconds between runner heartbeats.
    heartbeat_sec: int = Field(default=30, gt=0)
    #: Seconds after the last renewal at which an orchestrator lease may be taken over. The
    #: orchestrator renews it in the background every quarter of this and counts it lost three
    #: quarters after its last renewal, whatever ``heartbeat_sec`` is, so any positive value
    #: works; a takeover from another login node is safe while the nodes' clocks differ by less
    #: than a quarter of it.
    lease_ttl_sec: int = Field(default=120, gt=0)
    #: Seconds between orchestrator polls of the store and Slurm.
    poll_sec: int = Field(default=30, gt=0)
    #: Seconds added to every allocation's time limit beyond its stages' own, for the
    #: runner's startup checks and its shutdown: at least the time it keeps free before the job's
    #: end (``hls.pipeline.stage.END_MARGIN_SEC``), which the planner checks.
    allocation_overhead_sec: int = Field(default=900, ge=0)
    #: Seconds without a runner heartbeat after which a running allocation counts as hung:
    #: the orchestrator cancels it by id and retries its stage.
    allocation_dead_after_sec: int = Field(default=600, gt=0)
    #: GiB of each stage's declared memory that its roles leave to the runner. The runner and
    #: the srun client of every role run in the batch step, under the job's memory limit with
    #: the roles, so roles that declared all of a stage's memory could push the job past its
    #: limit, and the kernel would then kill a process of whichever step it picks. The reserve
    #: lies inside the declared shape, so the allocation request is still exactly the declared
    #: shapes; a stage whose roles do not leave it is refused when the plan is made.
    runner_memory_gib: int = Field(default=4, ge=0)
    #: Tasks, threads included, of the runner's uid that the harness, the runner and srun hold on
    #: a node for each evaluation it holds, with margin: a contestant's step gets the uid's
    #: process headroom less these for each (``hls.substrate.nproc``). The ``PoolPeak.others_tasks``
    #: of ``role_ended`` events on a node that ran no other evaluation can check it for a site.
    nproc_reserve: int = Field(default=1024, ge=0)
    #: Whether an agent build whose sandbox ends on its own memory (``contestant_oom``) is graded
    #: on its last commit; false scores such a build zero as the
    #: contestant's fault, with nothing graded. A write past the sandbox's disk cap and its step's
    #: process limit are graded on the last commit either way. The plan records it, so a resumed
    #: instance keeps the value it was planned with; ``hls.stages.agent_build`` reads it there.
    grade_after_sandbox_oom: bool = True
    #: Seconds an agent build's stage keeps after the agent's hard limit to collect the build
    #: (``hls.stages.agent_build``): stop the sandbox, bundle the repository at the commit point,
    #: copy the transcript and read it. The commit point gets what the fixed steps around it leave
    #: (``hls.stages.agent_build.COLLECT_STEPS_SEC``). A collection that does not end within this
    #: is infra that is not retried, and the build is void, never the contestant's;
    #: the brief states it. A plan whose value leaves the commit point less
    #: than ``hls.stages.agent_build.COMMIT_MIN_SEC`` is refused. A probe or preflight-only build,
    #: which runs no commit point, keeps its own short ending
    #: (``hls.stages.agent_build.STOP_BUILD_COLLECT_SEC``). The plan records the value, so a
    #: resumed instance keeps the one it was planned with. The default, an hour, leaves the commit
    #: point at least 1,850 s beside the fixed steps' bounds and is far above the time a collection
    #: takes.
    collect_allowance_sec: int = Field(default=3600, gt=0)

    @model_validator(mode="after")
    def _dead_after_covers_heartbeats(self) -> RunSettings:
        if self.allocation_dead_after_sec < 3 * self.heartbeat_sec:
            raise ValueError("allocation_dead_after_sec must be at least 3 x heartbeat_sec")
        return self

    def retry_for(self, stage: str, *, default_max: int) -> RetryPolicy:
        """The policy for *stage*: its own table if present, else the default. A table that
        leaves *max* unset takes *default_max*, its stage kind's count
        (``hls.pipeline.stage.retry_policy``)."""
        policy = self.retry[stage] if stage in self.retry else RetryPolicy()
        if policy.max is not None:
            return policy
        return policy.model_copy(update={"max": default_max})


class Settings(BaseModel):
    """Everything this loader reads from hls.toml."""

    # A refused value stays out of the error's text, which a traceback prints: an upstream proxy
    # pasted with its credentials is one.
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    #: The cluster and store.
    site: SiteSettings
    #: Run policy.
    run: RunSettings = Field(default_factory=RunSettings)
    #: Task name to stage name to the fields of the declared shape the site changes. Only the
    #: names' characters are checked here; the planner checks the names against the pipelines'.
    resources: dict[TaskName, dict[StageName, ShapeOverride]] = Field(default_factory=dict)
    #: Route name to the route: how an arm's agent reaches its model, by the names of secrets.
    routes: dict[RouteName, Route] = Field(default_factory=dict)
    #: Arm name to the arm: an agent, its model and the name of its route.
    arms: dict[ArmName, Arm] = Field(default_factory=dict)
    #: Sandbox store name to its directory on the site's nodes: what a task's sandboxes start from,
    #: such as its staged images, which its tool runner names by store (``HostPath.store``). A drive
    #: binds each store its task's runner names into the driver's container at the path the runner
    #: declares (``hls.stages.deploy_drive.sandbox_binds``), and a plan with a stage that binds a
    #: store this does not name is refused (``hls.pipeline.planner``). Every sandbox the driver starts
    #: can reach those binds, so a store stands apart from the store root, the secrets file's
    #: directory, the checkout and every other store (``sandbox_store_problems``): checked as written
    #: here, and with links followed when a plan is made and on the drive's node. Empty for a site
    #: that drives no sandboxed task, and then left out of the resolved settings, so their digest is
    #: that of a file without the table.
    sandbox_stores: dict[SandboxStoreName, Annotated[Path, AfterValidator(_store_directory)]] = Field(
        default_factory=dict, exclude_if=lambda stores: not stores)

    @model_validator(mode="after")
    def _arm_routes(self) -> Settings:
        problems = []
        for name, arm in self.arms.items():
            try:
                check_arm_route(arm, self.routes)
            except ValueError as e:
                problems.append(f"arms.{_toml_key(name)}: {e}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @model_validator(mode="after")
    def _stores_apart(self) -> Settings:
        # Compared as written: settings are validated again wherever a plan is read, so this reads no
        # filesystem. The planner and the drive compare them with their links followed too.
        problems = sandbox_store_problems(self.sandbox_stores, store_root=self.site.store_root,
                                          secrets_path=self.site.secrets_path)
        if problems:
            raise ValueError("; ".join(problems))
        return self


def sandbox_store_problems(stores: Mapping[str, Path], *, store_root: Path, secrets_path: Path,
                           checkout: Path | None = None, resolve: bool = False) -> list[str]:
    """Why the sandbox stores *stores*, by name, may not be bound where every sandbox a drive
    starts can reach them, each reason naming its store and no path: a store that is, holds or lies
    in the store root (*store_root*), the directory of the secrets file (*secrets_path*), the
    checkout (*checkout*, when given) or another store. A store that holds one of them would hand a
    sandbox's commands the store, the secrets or the harness's code; one that lies in one would
    hand them part of it. With *resolve* every path is also compared with its links followed, so
    an alias cannot hide where a store is. Empty when every store stands apart."""

    def forms(path: Path) -> set[PurePosixPath]:
        return {PurePosixPath(path)} | ({PurePosixPath(Path(path).resolve())} if resolve else set())

    def meets(a: Path, b: Path) -> bool:
        return any(x.is_relative_to(y) or y.is_relative_to(x) for x in forms(a) for y in forms(b))

    roots = {"the store root (site.store_root)": store_root,
             "the secrets file's directory (site.secrets_path)": Path(secrets_path).parent}
    if checkout is not None:
        roots["the checkout"] = checkout
    names = sorted(stores)
    problems = [f"sandbox_stores.{_toml_key(name)} is, holds or lies in {label}"
                for name in names for label, root in roots.items() if meets(stores[name], root)]
    problems += [f"sandbox_stores.{_toml_key(a)} and sandbox_stores.{_toml_key(b)} overlap"
                 for i, a in enumerate(names) for b in names[i + 1:] if meets(stores[a], stores[b])]
    return problems


def _set_dotted(table: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        nxt = table.setdefault(k, {})
        if not isinstance(nxt, dict):
            raise SettingsError(f"override {dotted}: {k} is not a table")
        table = nxt
    table[keys[-1]] = value


def _toml_key(part: str | int) -> str:
    """One part of an error's location written as TOML writes a key: bare when it can be,
    else quoted with escapes, so a newline in a key shows as ``\\n``."""
    text = str(part)
    return text if _BARE_KEY.fullmatch(text) else json.dumps(text)


def _describe(err: ValidationError, source: str) -> str:
    lines = []
    for e in err.errors():
        parts = list(e["loc"])
        # pydantic marks an error in a table key itself with a trailing "[key]".
        if parts and parts[-1] == "[key]":
            parts.pop()
        loc = ".".join(_toml_key(p) for p in parts)
        if e["type"] == "extra_forbidden":
            lines.append(f"unknown setting {loc}")
        else:
            # A rule over several tables has no location of its own; its message names them. A
            # rule's own message needs no word of pydantic's before it.
            msg = e["msg"].removeprefix("Value error, ")
            lines.append(f"{loc}: {msg}" if loc else msg)
    return f"{source}: " + "; ".join(lines)


def load_settings(path: Path, overrides: Mapping[str, str] | None = None) -> Settings:
    """Read the site, run, resources, routes, arms and sandbox_stores tables of *path*, apply *overrides*
    (dotted path to raw CLI string), and validate strictly. A top-level key that is not one
    of those tables is an error.

    An override stays a string until validation converts it to the type of the setting it
    names, so ``--site.account 12345`` stays a string and ``--run.poll_sec 10`` becomes an
    int. List and table settings, such as ``site.sbatch_extra``, are set in hls.toml.

    A *path* that cannot be read or is not TOML is a SettingsError of one line that quotes
    nothing of it: why it cannot be read, or why it does not load (``load_toml``)."""
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        raise SettingsError(f"{path.absolute()}: no such file; copy docs/examples/hls.toml of the hls checkout there "
                            "and fill in its [site] table, or name your hls.toml with --config") from None
    except OSError as e:
        raise SettingsError(f"{path}: cannot be read ({e.strerror or type(e).__name__})") from None
    try:
        # Settings forbids any other top-level key, which _describe names as an unknown setting.
        data = load_toml(content)
    except TomlError as e:
        raise SettingsError(f"{path}: {e}") from None
    for dotted, raw in (overrides or {}).items():
        _set_dotted(data, dotted, raw)
    try:
        return Settings.model_validate(data)
    except ValidationError as e:
        # The message says everything. The validator's own error is not chained, since its
        # text shows each refused input, which may be a pasted credential or base URL.
        raise SettingsError(_describe(e, str(path))) from None


def resolved(settings: Settings) -> dict[str, Any]:
    """Plain JSON form of *settings*, for the plan. Holds the secrets path, never a
    secret."""
    return settings.model_dump(mode="json")


def settings_sha256(settings: Settings) -> str:
    """Digest of the resolved settings, for manifests' provenance."""
    return sha256_bytes(json.dumps(resolved(settings), sort_keys=True, separators=(",", ":")).encode())
