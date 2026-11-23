"""Turning a pipeline's stages and the settings into an instance's first plan.

The planner:
- refuses a stage whose kind ``hls.stages.registry`` does not register: the runner imports the
  registry alone to find a stage by its kind, so the planner imports it too;
- applies ``[resources.<task>.<stage>]`` tables field by field (``ShapeOverride``): the fields a
  table sets replace the declaration's, and the plan records which stages they changed;
- refuses resources tables keyed by no known task: a table is keyed by a task's ``TaskSpec.id``,
  and only the plan's own task's tables apply to it, so a key that is no task's (a task's
  directory, say) would otherwise change nothing without a word;
- refuses retry and resources tables that name no known stage, every task's tables among them:
  a name is known when some pipeline declares it, it matches one of the numbered names
  pipelines make (``known_patterns``), or a declared grower may add it; one hls.toml serves
  every pipeline of a task, so a table of the plan's task for a known stage the plan does not
  run, nor its grower may add, is ignored with a warning;
- refuses a stage that binds a sandbox store (``Stage.sandbox_stores``) the site's
  ``[sandbox_stores]`` table does not name, or one that, with links followed, is, holds or lies in
  the store root, the secrets file's directory, the checkout or another bound store
  (``hls.settings.sandbox_store_problems``);
- groups stages into allocations under the GPU policy, splitting one that no QOS of the site
  holds into a chain of allocations that each one does (``policy_layout``), or as the pipeline
  lays them out (``AllocationDecl``: side by side, after one allocation's success, or once
  several have published);
- sizes each allocation's request under the site's rules, its time with headroom for its roles'
  steps to start (``launch_headroom_sec``) and its disk with each node-local copy of the task's
  data its stages have made by then once, as a copy lasts the job (``allocation_request``), and
  picks the QOS it runs under by that time against the site's limits (``site.qos_by_hours``),
  refusing a stage that no QOS holds alone and an allocation a pipeline laid out that none holds;
  ``layout_allocations`` sizes every allocation so, whether the policy grouped it, the pipeline
  laid it out, or a grower added it;
- records the checkout, the toolchain and each stage's retry policy;
- records the names of the secrets granted to each stage, and refuses a checkout that holds
  a copy of a secrets file, a ``site.secrets_path`` inside the checkout or the store, a
  ``site.registry_username`` that holds a value of the secrets file, and an images or
  sandbox_images stage that holds one half of a registry login (``registry_login_problem``);
- refuses a plan whose text would hold a value of the secrets file it was given, such as one
  passed as a ``task.*`` override (values never go into a plan, which the
  store keeps as the instance's provenance), naming each field and the secret, never the value.

Under the default max policy a run is one allocation sized to its largest stage, and the
planner warns how many GPUs idle during which stages. Under split a new allocation starts
at each change in GPU count.

A site may cap the time a job asks for under each QOS (``site.qos_by_hours``). An allocation the
policy groups that no QOS holds is split: its stages go in turn into consecutive allocations,
each taking as many of those that follow as the longest limit holds, timed as any allocation is
(``allocation_time_sec``), and each runs after the one before. The plan's warnings say which
stages were split and why. Each stage is sized for its own node-local copies and image trees, so
an allocation of some of a group's stages is sized right on its own. A pipeline's own layout is
never split: its widths, such as a calibration's ``--fanout`` or a credit's above 1, keep its
allocations under the limits, and an allocation of it that no QOS holds is refused.

A run always goes to the site's GPU partition. Only the cache warmers (``hls provision``,
``hls images build``) and an export (``hls export``) are planned for the CPU partition: their
stages ask for no GPU, and on a site with ``site.cpu_partition`` they go there, where the GPU
partition's floor does not apply.

A pipeline may lay its allocations out itself. Each ``AllocationDecl`` names its stages and the
earlier allocations it waits for: ``depends_on`` is Slurm's afterok on one, and ``waits_for``
holds it back until every stage of each named allocation has published, with no Slurm
dependency, so it can wait for several. A stage reads only stages earlier in its own allocation
or in one its allocation reaches through those. A plan may also name a grower
(``hls.pipeline.growth``), which adds stages and allocations once every stage has published;
``stage_plans`` and ``layout_allocations`` plan those as they plan a new plan's. The planner
reads what a grower may add from its declaration (``hls.pipeline.grown``) and imports no grower.

The cache warmers run one at a time on a cluster: their plans mark every allocation
``singleton``, which Slurm holds until no other job of the user with its name runs, and take
``site.warmer_job_name``. Such an allocation depends on no other by afterok: when the policy
makes several of a warmer's, each waits until the one before has published (``waits_for``).

A stage's grants come from one of two places. A stage kind may give ``Stage.declared_grants``,
a classmethod ``declared_grants(params, settings) -> dict[str, str]``, variable to secret name, as
the agent build does from its arm's route: the stage then gets exactly those, each secret checked to be
in the secrets file, and the file may not grant it as well. A declared grant of a harness's
base-URL variable must hold a URL that harness takes (``hls.arms.check_granted_base_urls``);
the value is read from the file in memory and never recorded. Any other stage gets what the
secrets file grants ``stage.<name>``. Either way the plan records the names in
``StagePlan.grants``, and the runner resolves their values into the stage's ``ctx.secrets``
when the stage starts."""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import hls.stages.registry  # noqa: F401  (registers every stage implementation, as the runner does)
from hls.arms import check_granted_base_urls
from hls.measured import MeasuredFiles
from hls.pipeline.checkout import CodeState, secrets_copies
from hls.pipeline.grown import GROWN_STAGES, grown_stages
from hls.pipeline.plan import Allocation, Plan, StagePlan, Toolchain
from hls.pipeline.stage import END_MARGIN_SEC, STAGES, Stage, most_roles, retry_policy, stage_class
from hls.provision.mounts import DataCopies, copies_gib
from hls.settings import Settings, sandbox_store_problems
from hls.spec.resources import ResourceShape, sequential
from hls.stages.images import ImagesStage, registry_login_problem
from hls.stages.sandbox_images import SandboxImagesStage
from hls.store.layout import InstanceLayout, check_stage
from hls.substrate.base import NoQOSHolds
from hls.substrate.factory import seal_memory_gib, substrate
from hls.substrate.nproc import nproc_max_problem
from hls.substrate.ucounts import cap_problem
from hls.substrate.secrets import Scrubber, SecretsFile, grant_names, secrets_path_problem

#: Directories every batch script and role step can find the system's programs in.
SYSTEM_PATH = ("/usr/local/bin", "/usr/bin", "/bin")
# Tools whose directories go first on the allocation's PATH.
_TOOLS = ("sbatch", "srun", "squeue", "sacct", "scontrol", "scancel", "git", "nvidia-smi", "apptainer")


class PlanError(ValueError):
    """The stages, their parameters or the settings cannot make a plan."""


class AllocationPastQOS(PlanError):
    """A stage alone, or an allocation a pipeline laid out, asks for more time than any QOS of the
    site holds (``allocation_qos``)."""


@dataclass(frozen=True)
class StageDecl:
    """One stage as a pipeline declares it."""

    #: Stage name, unique in the pipeline.
    name: str
    #: Registry name of its implementation.
    kind: str
    #: Its declared shape, before any ``[resources]`` override.
    shape: ResourceShape
    #: Parameters for the implementation.
    params: Mapping[str, Any] = field(default_factory=dict)
    #: Earlier stages of the same instance whose manifests it reads.
    after: tuple[str, ...] = ()
    #: Manifests of other instances it reads, by key, pinned to their digests.
    inputs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AllocationDecl:
    """One allocation as a pipeline lays it out, or as the GPU policy groups it
    (``policy_layout``)."""

    #: Its stages, in declaration order.
    stages: tuple[str, ...]
    #: The earlier allocation, by its position in the layout, whose success it waits for
    #: (Slurm's afterok); None for none.
    depends_on: int | None = None
    #: Earlier allocations, by position, every stage of which must be published before it is
    #: submitted, with no Slurm dependency on them.
    waits_for: tuple[int, ...] = ()


def toolchain_here() -> Toolchain:
    """The toolchain of this login node: this interpreter, and a PATH with the directories
    of the Slurm tools, git, nvidia-smi and apptainer first."""
    dirs: list[str] = []
    for tool in _TOOLS:
        found = shutil.which(tool)
        if found is not None and str(Path(found).parent) not in dirs:
            dirs.append(str(Path(found).parent))
    dirs += [d for d in SYSTEM_PATH if d not in dirs]
    return Toolchain(python=Path(sys.executable), path=":".join(dirs), home=Path.home())


def check_setting_names(settings: Settings, *, task: str, stage_names: Collection[str],
                        known_stage_names: Collection[str], known_tasks: Collection[str],
                        known_patterns: Sequence[str] = (), grower: str | None = None) -> list[str]:
    """Raise PlanError for a ``[resources.<task>]`` table keyed by neither *task*, the plan's
    own, nor one of *known_tasks*, naming each such key and the keys that are known. Raise
    PlanError for a ``[run.retry.<stage>]`` table, or a ``[resources.<task>.<stage>]`` table of
    any task, naming a stage that no pipeline declares (*known_stage_names*), no numbered name of
    *known_patterns* matches and no declared grower may add (``hls.pipeline.grown.grown_stages``);
    a table of *task* may also name a stage of this plan (*stage_names*). One hls.toml serves
    every pipeline of a task, so a table of *task* naming a stage that is known but that this
    plan does not run, nor its *grower* may add, is ignored: return one warning for each."""
    tasks = {task, *known_tasks}
    unknown = sorted(t for t in settings.resources if t not in tasks)
    if unknown:
        raise PlanError(f"settings name unknown tasks: {', '.join(f'resources.{t}' for t in unknown)}; a resources "
                        f"table is keyed by a task's TaskSpec id, one of {', '.join(sorted(tasks))}")
    grown = grown_stages()
    mine = grown_stages(grower) if grower is not None else None

    def known(name: str) -> bool:
        return name in known_stage_names or any(re.fullmatch(p, name) for p in known_patterns) or grown.holds(name)

    bad = [f"run.retry.{s}" for s in settings.run.retry if not known(s)]
    warnings = []
    for t, tables in settings.resources.items():
        for s in tables:
            if t == task and s in stage_names:
                continue
            if not known(s):
                bad.append(f"resources.{t}.{s}")
            elif t == task and not (mine is not None and mine.holds(s)):
                warnings.append(f"[resources.{t}.{s}] names stage {s}, which this plan does not run: the table is "
                                "ignored")
    if bad:
        raise PlanError(f"settings name stages that do not exist: {', '.join(bad)}")
    return warnings


def group(stages: Sequence[StagePlan], *, policy: Literal["max", "split"], min_gpus: int) -> list[list[StagePlan]]:
    """Stages grouped into allocations: one group under "max", a new group at each change in
    the GPU count the site will grant under "split". The site raises a request to *min_gpus*
    GPUs, so stages that differ only below that count share an allocation."""
    if policy == "max":
        return [list(stages)]
    groups: list[list[StagePlan]] = []
    for s in stages:
        if groups and max(groups[-1][-1].shape.gpus, min_gpus) == max(s.shape.gpus, min_gpus):
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


def launch_headroom_sec(stages: Sequence[StagePlan], allowance_sec: int) -> int:
    """Seconds an allocation running *stages* keeps for their roles' steps to start: each
    stage's most roles (``most_roles``) times *allowance_sec*, the site's
    ``step_launch_allowance_sec``. A stage's deadline grows by the time its roles' steps take to
    start (``StageContext.deadline``), and this is the room for it."""
    return allowance_sec * sum(most_roles(s.kind, s.params) for s in stages)


def _copies(stage: StagePlan) -> dict[str, int]:
    """The node-local copies of cache entries *stage* makes, as its parameters declare them
    (``DataCopies``); none for a kind whose parameters declare none."""
    model = stage_class(stage.kind).params_model
    if not issubclass(model, DataCopies):
        return {}
    return dict(model.model_validate(stage.params).data_copies)


def allocation_time_sec(stages: Sequence[StagePlan], settings: Settings) -> int:
    """Seconds an allocation running *stages* in order asks for: the sum of the stages' times plus
    the runner's overhead (``run.allocation_overhead_sec``) and the headroom for their roles'
    steps to start (``launch_headroom_sec``)."""
    return (sum(s.shape.time_sec for s in stages) + settings.run.allocation_overhead_sec
            + launch_headroom_sec(stages, settings.site.step_launch_allowance_sec))


def allocation_request(stages: Sequence[StagePlan], settings: Settings, *, gpu_kind: str | None,
                       partition: str | None = None, socket_tasks: int | None = None) -> ResourceShape:
    """What an allocation running *stages* in order asks for: the largest of each resource but
    disk, and their time (``allocation_time_sec``). Its disk is the most any stage needs: the
    stage's disk beside its copies of the task's data, and every entry the stages up to it have
    copied, once each (``DataCopies``), since a copy lasts the job and a later stage that mounts an
    entry an earlier one copied reuses its copy (``hls.store.cache.copies_root``). With
    *socket_tasks*, the CPUs of each of the allocation's two tasks (the caller's ``socket_tasks``
    of the same stages, which it also hands to the allocation and to sbatch), it asks for both
    tasks' CPUs instead. On the GPU
    partition (*partition* None) it is raised to the site's GPU floor, and *gpu_kind* is the kind
    the run uses, for a floor raised from no GPUs. The floor is the GPU partition's rule, so an
    allocation on the CPU partition is not raised."""
    seq = sequential(s.shape for s in stages)
    held: dict[str, int] = {}
    disk_gib = 0
    for s in stages:
        made = _copies(s)
        for entry, size in made.items():
            held[entry] = max(size, held.get(entry, 0))
        disk_gib = max(disk_gib, s.shape.disk_gib - copies_gib(made) + copies_gib(held))
    timed = ResourceShape(**{**seq.model_dump(), "time_sec": allocation_time_sec(stages, settings),
                             "disk_gib": disk_gib})
    if socket_tasks is not None:
        timed = timed.model_copy(update={"cpus": 2 * socket_tasks})
    return timed if partition is not None else substrate(settings.site).site_request(timed, settings.site, gpu_kind=gpu_kind)


def socket_tasks(stages: Sequence[StagePlan], settings: Settings, *, partition: str | None) -> int | None:
    """The CPUs of each of the two tasks, at most one per socket, that an allocation running
    *stages* asks for on a site with ``site.evaluation_socket_tasks``: the CPUs of the scored
    server one of its stages serves (the stage kind's ``scored_server_cpus``), so that server's
    step takes one task's CPUs, on one socket. None when the setting is off, the allocation goes
    to another partition than the GPU one (*partition* set), or no stage serves a scored server.
    Raise PlanError when the stages' servers differ in CPUs, or a stage asks for more CPUs than
    the two tasks hold together."""
    if not settings.site.evaluation_socket_tasks or partition is not None:
        return None
    counts: set[int] = set()
    for s in stages:
        cls = stage_class(s.kind)
        cpus = cls.scored_server_cpus(cls.params_model.model_validate(s.params))
        if cpus is not None:
            counts.add(cpus)
    if not counts:
        return None
    if len(counts) > 1:
        raise PlanError(f"the allocation serves scored servers of {sorted(counts)} CPUs, and its two tasks hold one "
                        "server's CPUs each")
    tasks = counts.pop()
    need = max(s.shape.cpus for s in stages)
    if need > 2 * tasks:
        raise PlanError(f"a stage asks for {need} CPUs, more than the two tasks of {tasks} CPUs (one per socket, "
                        "site.evaluation_socket_tasks) the allocation holds")
    return tasks


def _past_qos(stages: Sequence[StagePlan], settings: Settings, *, partition: str | None) -> NoQOSHolds | None:
    """Why no QOS of the site holds an allocation running *stages* on *partition* by its time
    (``allocation_time_sec``, ``allocation_qos``); None when one does."""
    try:
        substrate(settings.site).allocation_qos(allocation_time_sec(stages, settings), settings.site,
                                                partition=partition)
    except NoQOSHolds as e:
        return e
    return None


def _packed(stages: Sequence[StagePlan], settings: Settings, *, partition: str | None) -> list[list[StagePlan]]:
    """*stages* as consecutive allocations, packed greedily in order: each takes as many of the
    stages that follow as a QOS of the site holds by their time (``_past_qos``). Raise
    AllocationPastQOS for a stage that no QOS holds alone, naming it, its time and the longest
    limit."""
    pieces: list[list[StagePlan]] = []
    for s in stages:
        if pieces and _past_qos([*pieces[-1], s], settings, partition=partition) is None:
            pieces[-1].append(s)
            continue
        alone = _past_qos([s], settings, partition=partition)
        if alone is not None:
            raise AllocationPastQOS(f"stage {s.name} takes {s.shape.time_sec / 3600:.2f} h; alone, with the runner's "
                                    f"overhead and its launch headroom, {alone}")
        pieces.append([s])
    return pieces


def policy_layout(stages: Sequence[StagePlan], settings: Settings, *, partition: str | None,
                  singleton: bool) -> tuple[list[AllocationDecl], list[str]]:
    """The allocations *stages* run in under the GPU policy (``run.gpu_policy``, ``group``) on
    *partition* (None: the GPU partition), and why any was split. A group that no QOS of the site
    holds by its time is split in stage order into consecutive allocations that a QOS holds each
    (``_packed``), and a note for the plan's warnings says so. The allocations form one chain, each
    after the one before: by Slurm's afterok on it, or, for a *singleton* plan, whose allocations
    depend on none, once it has published (``waits_for``). Raise AllocationPastQOS for a stage
    that no QOS holds alone."""
    policy = settings.run.gpu_policy
    pieces: list[list[StagePlan]] = []
    notes: list[str] = []
    for members in group(stages, policy=policy, min_gpus=settings.site.min_gpus_per_job):
        past = _past_qos(members, settings, partition=partition)
        if past is None:
            pieces.append(list(members))
            continue
        packed = _packed(members, settings, partition=partition)
        notes.append(f"the {policy} GPU policy groups stages {members[0].name} to {members[-1].name} into one "
                     f"allocation, but {past}; they run as allocations {len(pieces)} to "
                     f"{len(pieces) + len(packed) - 1} instead, as many stages each as that limit holds, each queued "
                     "after the one before")
        pieces += packed
    decls = []
    for k, piece in enumerate(pieces):
        names = tuple(s.name for s in piece)
        if k == 0:
            decls.append(AllocationDecl(stages=names))
        elif singleton:
            decls.append(AllocationDecl(stages=names, waits_for=(k - 1,)))
        else:
            decls.append(AllocationDecl(stages=names, depends_on=k - 1))
    return decls, notes


def idle_gpu_warnings(alloc: Allocation, stages: Sequence[StagePlan], *, policy: Literal["max", "split", "layout"],
                      min_gpus: int) -> list[str]:
    """One warning per stage that holds fewer GPUs than its allocation, naming the cost and
    what would avoid it. When the site's floor of *min_gpus* alone sets the allocation's count for
    the stage, as it always does under "split", nothing in the run avoids it, and the warning names
    the floor; else, under "max", the "split" policy would; under "layout", the allocation is one
    the pipeline laid out itself, which no setting changes."""
    floor = f"site.min_gpus_per_job = {min_gpus} sets the allocation's GPU count"
    fixes = {"max": "--run.gpu_policy=split asks for each stage's own count instead",
             "layout": "the pipeline's layout puts it in this allocation"}
    out = []
    for s in stages:
        idle = alloc.request.gpus - s.shape.gpus
        if idle > 0:
            hours = s.shape.time_sec / 3600
            fix = floor if alloc.request.gpus <= max(s.shape.gpus, min_gpus) else fixes[policy]
            out.append(f"allocation {alloc.index} holds {alloc.request.gpus} GPUs; stage {s.name} uses "
                       f"{s.shape.gpus}, so {idle} idle for up to {hours:.2f} h ({idle * hours:.2f} GPU-hours). {fix}")
    return out


def _grants(name: str, cls: type[Stage], params: Any, settings: Settings,
            secrets: SecretsFile | None) -> dict[str, str]:
    """The names of the secrets granted to stage *name*, of class *cls* with *params*: what
    the class declares from them and *settings*, else what the secrets file grants it. A grant
    may not set a variable the launcher owns: the step wrapper loads a role's env file over the
    launcher's own variables, so such a grant could repoint PATH or the GPUs."""
    from_file = grant_names(secrets, f"stage.{name}") if secrets is not None else {}
    if cls.declared_grants is None:
        grants = from_file
    else:
        try:
            grants = dict(cls.declared_grants(params, settings))
        except ValueError as e:
            raise PlanError(f"stage {name}: {e}") from e
        if from_file:
            raise PlanError(f"stage {name} is granted what its parameters declare; the secrets file may not also "
                            f"grant stage.{name} ({sorted(from_file)})")
        missing = sorted({n for n in grants.values() if secrets is None or n not in secrets.secrets})
        if missing:
            where = "the secrets file has" if secrets is not None else "no secrets file was read, so it has"
            raise PlanError(f"stage {name} is granted secrets {missing}, which {where} none of")
        # A base URL the harness cannot take would otherwise stop the run only when the stage
        # starts, after the queue and the stages before it. The stage checks it again, since the
        # runner reads the file afresh at each allocation's start.
        try:
            check_granted_base_urls(grants, secrets.secrets if secrets is not None else {})
        except ValueError as e:
            raise PlanError(f"stage {name}: {e}") from None
    owned = sorted(v for v in grants if substrate(settings.site).reserved_variable(v))
    if owned:
        raise PlanError(f"stage {name}: grants set {owned}, which the launcher owns")
    return grants


def _registry_user_problem(settings: Settings, secrets: SecretsFile | None) -> str | None:
    """Why ``site.registry_username`` cannot go with *secrets*, or None: it holds a value of the
    file, which every role naming a path that holds the user name would then be refused for
    holding, the base build's own environment first. Names the secrets, never a value."""
    user = settings.site.registry_username
    if user is None or secrets is None:
        return None
    held = sorted(name for name, value in secrets.secrets.items() if value in user)
    if not held:
        return None
    return (f"site.registry_username holds the value of secret {', '.join(held)} of the secrets file; the registry "
            "user name is not a secret: remove it from the secrets file")


def stage_plans(decls: Sequence[StageDecl], settings: Settings, task: str, secrets: SecretsFile | None, *,
                earlier: Collection[str] = ()) -> list[StagePlan]:
    """Each of *decls* as planned for *task* under *settings*: its kind's parameters checked, its
    ``[resources]`` table merged, its retry policy and its grants (what *secrets* grants; None
    grants nothing). A stage that seals a container (``Stage.seals_containers``) keeps room for its
    seal's memory beside its roles, as for the runner's (``seal_memory_gib``), and its kind's check
    reads the two together. A stage may read stages declared before it and the stages of *earlier*,
    which a plan holding them already ran. Raise PlanError naming the stage, for a stage that binds
    a sandbox store (``Stage.sandbox_stores``) on a site that is no Slurm site, whose sandboxes
    need Apptainer on the node, or one the site does not name (``Settings.sandbox_stores``), and
    when a stage runs
    a contestant's code and ``site.nproc_max`` is under the floor of one evaluation, which no node
    can run a contestant's step under (``hls.substrate.nproc.nproc_max_problem``), or
    ``site.sandbox_namespaces``,
    ``site.sandbox_inotify_instances`` or ``site.sandbox_inotify_watches`` is under its floor
    (``hls.substrate.ucounts.cap_problem``), for a plan's first stages and for those a grower adds
    alike."""
    seen: set[str] = set()
    out = []
    overrides = settings.resources.get(task, {})
    runner, seal = settings.run.runner_memory_gib, seal_memory_gib(settings.site)
    for d in decls:
        try:
            check_stage(d.name)
        except ValueError as e:
            raise PlanError(str(e)) from e
        if d.name in seen or d.name in earlier:
            raise PlanError(f"stage {d.name} is declared twice")
        if d.kind not in STAGES:
            raise PlanError(f"stage {d.name}: stage kind {d.kind!r} is not registered by hls.stages.registry, which "
                            f"the runner imports to find it; it registers {sorted(STAGES)}")
        late = [a for a in d.after if a not in seen and a not in earlier]
        if late:
            raise PlanError(f"stage {d.name} reads {late}, which do not come before it")
        seen.add(d.name)
        override = overrides.get(d.name)
        try:
            shape = d.shape if override is None else override.apply(d.shape)
        except ValueError as e:
            raise PlanError(f"[resources.{task}.{d.name}] leaves no valid shape: {e}") from e
        sealed = 0
        try:
            cls = stage_class(d.kind)
            params = cls.params_model.model_validate(dict(d.params))
            sealed = seal if seal and cls.seals_containers(params) else 0
            cls.check(shape, params, runner_memory_gib=runner + sealed)
            cls.check_site(shape, params, site=settings.site)
        except ValueError as e:
            gateway = (f" (its seal holds {sealed} GiB beside its roles on this site, so the runner's {runner} GiB "
                       f"counts as {runner + sealed})") if sealed else ""
            raise PlanError(f"stage {d.name}: {e}{gateway}") from e
        stores = sorted(set(cls.sandbox_stores(params)))
        if stores and settings.site.substrate != "slurm":
            raise PlanError(f"stage {d.name} binds the sandbox stores {', '.join(stores)}, whose sandboxes run under "
                            f"Apptainer on the node, which a {settings.site.substrate} site does not run")
        unnamed = sorted(s for s in stores if s not in settings.sandbox_stores)
        if unnamed:
            raise PlanError(f"stage {d.name} binds the sandbox stores {', '.join(unnamed)}, which the site does not "
                            "name: give each its directory in hls.toml's [sandbox_stores]")
        out.append(StagePlan(name=d.name, kind=d.kind, shape=shape, shape_overridden=d.name in overrides,
                             params=params.model_dump(mode="json"), after=list(d.after), inputs=dict(d.inputs),
                             retry=retry_policy(settings.run, d.name, d.kind),
                             grants=_grants(d.name, cls, params, settings, secrets)))
    # A contestant's step runs under the site's process cap, and the node it will run on is not
    # known yet: the plan refuses only a cap under the floor of the one evaluation the step is,
    # and the runner checks the evaluations its node holds when the allocation starts.
    if any(stage_class(s.kind).runs_contestant_code for s in out):
        capped = nproc_max_problem(settings.site.nproc_max, 1)
        if capped is not None:
            raise PlanError(f"no node can run a contestant's step under site.nproc_max: {capped} (an allocation "
                            "checks it against the evaluations of each stage's size its node holds when it starts)")
        # The build sandbox's commands get their own limits on the per-uid kernel counts, at most
        # these caps (``hls.substrate.ucounts``).
        for problem in (cap_problem("sandbox_namespaces", settings.site.sandbox_namespaces),
                        cap_problem("sandbox_inotify_instances", settings.site.sandbox_inotify_instances),
                        cap_problem("sandbox_inotify_watches", settings.site.sandbox_inotify_watches)):
            if problem is not None:
                raise PlanError(problem)
    return out


def _reached(groups: Sequence[AllocationDecl], start: int) -> set[int]:
    """The allocations of *groups* that allocation *start* waits for, directly or through others."""
    seen: set[int] = set()
    pending = [start]
    while pending:
        g = groups[pending.pop()]
        for w in ([g.depends_on] if g.depends_on is not None else []) + list(g.waits_for):
            if w not in seen:
                seen.add(w)
                pending.append(w)
    return seen


def layout_allocations(stages: Sequence[StagePlan], groups: Sequence[AllocationDecl], settings: Settings, *,
                       gpu_kind: str | None, partition: str | None, job_name: str, first_index: int = 0,
                       singleton: bool = False, advice: str = "") -> list[Allocation]:
    """The allocations that run *stages* as *groups* lay them out, numbered from *first_index*.
    Each asks for its own stages (``allocation_request``: their time with the runner's overhead
    and the launch headroom of their roles, and a scored server's CPUs as two socket tasks,
    ``socket_tasks``) on *partition* (None: the GPU partition), runs under the QOS that holds
    that time (``allocation_qos``), and is checked against the site's sbatch rules under
    *job_name*; *advice* is added to a refusal of a request. *singleton* marks each to run one
    at a time with the site's other warmers.

    Raise PlanError when the overhead is shorter than the runner's end margin; when the groups
    do not hold every stage exactly once, each allocation's stages in declaration order; when an
    allocation is empty, names a stage the plan does not declare, waits for one allocation twice,
    or depends on or waits for one that does not come before it; when a stage reads a stage in
    another allocation that its own neither depends on nor waits for, directly or through
    others; when a singleton allocation depends on another; or when an allocation cannot be
    asked for, AllocationPastQOS among it for one that no QOS of the site holds. Such an
    allocation is one a pipeline or a grower laid out, which is never split: those the GPU policy
    groups come here already split to the site's limits (``policy_layout``)."""
    if settings.run.allocation_overhead_sec < END_MARGIN_SEC:
        raise PlanError(f"run.allocation_overhead_sec is {settings.run.allocation_overhead_sec}s, less than the "
                        f"{END_MARGIN_SEC:.0f}s the runner keeps free before an allocation's end (END_MARGIN_SEC), so "
                        "its last stage could never start")
    order = {s.name: i for i, s in enumerate(stages)}
    placed: dict[str, int] = {}
    for g, decl in enumerate(groups):
        k = first_index + g
        if not decl.stages:
            raise PlanError(f"allocation {k} holds no stage")
        for name in decl.stages:
            if name not in order:
                raise PlanError(f"allocation {k} names stage {name}, which the plan does not declare")
            if name in placed:
                raise PlanError(f"stage {name} is in allocations {first_index + placed[name]} and {k}")
            placed[name] = g
        positions = [order[name] for name in decl.stages]
        if positions != sorted(positions):
            raise PlanError(f"allocation {k} runs its stages out of declaration order: {list(decl.stages)}")
        if len(set(decl.waits_for)) != len(decl.waits_for):
            raise PlanError(f"allocation {k} waits for {[first_index + w for w in decl.waits_for]}, which names an "
                            "allocation twice")
        waited = ([decl.depends_on] if decl.depends_on is not None else []) + list(decl.waits_for)
        forward = [w for w in waited if not 0 <= w < g]
        if forward:
            raise PlanError(f"allocation {k} waits for {[first_index + w for w in forward]}, which do not come "
                            "before it")
        if singleton and decl.depends_on is not None:
            raise PlanError(f"allocation {k} runs one at a time with the site's other warmers and depends on "
                            f"allocation {first_index + decl.depends_on}; it may only wait for published stages")
    missing = [s.name for s in stages if s.name not in placed]
    if missing:
        raise PlanError(f"stages {missing} are in no allocation")
    for s in stages:
        g = placed[s.name]
        reached = _reached(groups, g)
        for name in s.after:
            # A stage outside *stages* is one a grown plan held before, already published.
            if name not in placed or placed[name] == g or placed[name] in reached:
                continue
            raise PlanError(f"stage {s.name} reads stage {name}, which runs in allocation "
                            f"{first_index + placed[name]}; allocation {first_index + g} neither depends on nor "
                            "waits for it")
    by_name = {s.name: s for s in stages}
    out: list[Allocation] = []
    for g, decl in enumerate(groups):
        k = first_index + g
        members = [by_name[n] for n in decl.stages]
        try:
            tasks = socket_tasks(members, settings, partition=partition)
            request = allocation_request(members, settings, gpu_kind=gpu_kind, partition=partition,
                                         socket_tasks=tasks)
        except PlanError as e:
            raise PlanError(f"allocation {k}: {e}") from e
        except ValueError as e:
            raise PlanError(f"allocation {k}: {e}{advice}") from e
        try:
            qos = substrate(settings.site).allocation_qos(request.time_sec, settings.site, partition=partition)
        except NoQOSHolds as e:
            raise AllocationPastQOS(f"allocation {k}: {e}; the pipeline lays this allocation out itself, so the "
                                    "planner does not split it") from e
        on = substrate(settings.site)
        placement = on.placement(partition=partition, qos=qos, socket_tasks=tasks, exclude=[])
        try:
            on.check_placement(request, settings.site, placement, job_name=job_name)
        except ValueError as e:
            raise PlanError(f"allocation {k}: {e}") from e
        out.append(Allocation(index=k, stages=list(decl.stages), request=request, placement=placement,
                              depends_on=None if decl.depends_on is None else first_index + decl.depends_on,
                              waits_for=[first_index + w for w in decl.waits_for], singleton=singleton))
    return out


def build_plan(*, decls: Sequence[StageDecl], task: str, pipeline: str, layout: InstanceLayout,
               settings: Settings, code: CodeState, toolchain: Toolchain, job_name: str,
               known_stage_names: Collection[str], known_tasks: Collection[str], now: float,
               secrets: SecretsFile | None = None, partition: Literal["gpu", "cpu"] = "gpu",
               groups: Sequence[AllocationDecl] | None = None, grower: str | None = None, singleton: bool = False,
               known_patterns: Sequence[str] = (), measured: MeasuredFiles = MeasuredFiles()) -> Plan:
    """The instance's first plan. Raise PlanError when the stages or settings are wrong, when
    the checkout holds a copy of a secrets file, when ``site.secrets_path`` lies inside the
    checkout or the store (``secrets_path_problem``), when an images or sandbox_images stage holds
    one half of a registry login (``registry_login_problem``), when ``site.registry_username`` holds a
    value of *secrets*, when a sandbox store a stage binds is, with links followed, not apart from
    the store root, the secrets file's directory, the checkout or another bound store
    (``hls.settings.sandbox_store_problems``), or when a stage runs a contestant's code and
    ``site.nproc_max`` is under the floor of one evaluation (``hls.substrate.nproc.nproc_max_problem``), or
    ``site.sandbox_namespaces``, ``site.sandbox_inotify_instances`` or
    ``site.sandbox_inotify_watches`` is under its floor (``hls.substrate.ucounts.cap_problem``).
    *secrets* is the site's secrets file, read for its grant names and that
    check; None grants nothing. *partition* is "cpu" only for a cache
    warmer or an export: every stage must then ask for no GPU, and the allocations go to
    ``site.cpu_partition``, or to the GPU partition with a warning when the site has none.

    *groups* lays the allocations out instead of the GPU policy (``layout_allocations``), and one
    of them that no QOS of the site holds is refused (AllocationPastQOS); the policy's allocations
    are split to the site's limits instead, and only a stage no QOS holds alone is refused
    (``policy_layout``).
    *grower* names the declared grower (``hls.pipeline.grown``) the orchestrator asks for more
    stages once every stage has published; a name no imported module declares is refused, and so
    is a grower on a singleton plan or on the CPU partition, where what it adds could not run as
    planned.
    *singleton* runs every allocation one at a time with the site's other warmers: the plan must
    take ``site.warmer_job_name`` as its job name. *known_patterns* are the numbered stage names
    pipelines make, which settings tables may name, and *known_tasks* the tasks other than *task*
    whose ids ``[resources]`` tables may be keyed by (``check_setting_names``). *measured* pins the
    measured files the roles load (``hls.measured``)."""
    # Harness roles bind the checkout, so a secrets file in it would reach them. A
    # checkout that is not a directory here is a test's stand-in: a real one came from
    # code_state, which resolved it.
    copies = secrets_copies(code.checkout) if code.checkout.is_dir() else []
    if copies:
        raise PlanError("the checkout holds secrets files, which no role may see: "
                        f"{', '.join(map(str, copies))}; move them out (the secrets file belongs at "
                        "site.secrets_path)")
    misplaced = secrets_path_problem(settings.site.secrets_path, checkout=code.checkout,
                                     store_root=settings.site.store_root)
    if misplaced is not None:
        raise PlanError(misplaced)
    registry_user = _registry_user_problem(settings, secrets)
    if registry_user is not None:
        raise PlanError(registry_user)
    stages = stage_plans(decls, settings, task, secrets)
    for s in stages:
        login = (registry_login_problem(s.name, settings.site.registry_username, s.grants)
                 if s.kind in (ImagesStage.kind, SandboxImagesStage.kind) else None)
        if login is not None:
            raise PlanError(login)
    # The stores a drive binds, with their links followed and beside the checkout, which the
    # settings alone cannot check.
    bound = {store for s in stages for store in stage_class(s.kind).sandbox_stores(
        stage_class(s.kind).params_model.model_validate(dict(s.params)))}
    apart = sandbox_store_problems({name: settings.sandbox_stores[name] for name in bound},
                                   store_root=settings.site.store_root, secrets_path=settings.site.secrets_path,
                                   checkout=code.checkout, resolve=True)
    if apart:
        raise PlanError("; ".join(apart) + ", with links followed: every sandbox a drive starts can reach the "
                        "stores its driver binds")
    if grower is not None:
        if grower not in GROWN_STAGES:
            raise PlanError(f"the plan grows by {grower!r}, which no imported module declares; declared: "
                            f"{sorted(GROWN_STAGES)}")
        if singleton:
            raise PlanError(f"the plan runs one at a time with the site's other warmers and may not grow by "
                            f"{grower!r}: what a grower adds is not held behind them")
        if partition == "cpu":
            raise PlanError(f"the plan runs on the CPU partition and may not grow by {grower!r}: a grown stage may "
                            "ask for GPUs, which only a run on the GPU partition holds")
    warnings = check_setting_names(settings, task=task, stage_names=[s.name for s in stages],
                                   known_stage_names=known_stage_names, known_tasks=known_tasks,
                                   known_patterns=known_patterns, grower=grower)
    if singleton and job_name != settings.site.warmer_job_name:
        raise PlanError(f"a warmer that runs one at a time with the site's others takes their job name, "
                        f"site.warmer_job_name ({settings.site.warmer_job_name}); --job-name {job_name} would "
                        "let it run beside them")
    kinds = {s.shape.gpu_kind for s in stages if s.shape.gpu_kind is not None}
    if len(kinds) > 1:
        raise PlanError(f"stages use different gpu kinds {sorted(kinds)}; one run uses one kind")
    gpu_kind = kinds.pop() if kinds else None
    on_partition = None
    if partition == "cpu":
        gpu_stages = [f"{s.name} ({s.shape.gpus})" for s in stages if s.shape.gpus]
        if gpu_stages:
            raise PlanError(f"a CPU pipeline runs CPU stages only; these ask for GPUs: {', '.join(gpu_stages)}")
        on_partition = settings.site.cpu_partition
        if on_partition is None:
            warnings.append("site.cpu_partition is not set, so this CPU-only pipeline runs on the GPU partition "
                            f"{settings.site.partition}")
    policy, floor = settings.run.gpu_policy, settings.site.min_gpus_per_job
    # The GPU partition's floor is the one rule a CPU plan's request can break there.
    advice = ""
    if partition == "cpu" and on_partition is None:
        advice = (f"; site.cpu_partition is not set, so this CPU-only pipeline would run on the GPU partition "
                  f"{settings.site.partition}: set site.cpu_partition to a partition that takes jobs with no GPU")
    if groups is None:
        laid_out, notes = policy_layout(stages, settings, partition=on_partition, singleton=singleton)
        warnings += notes
    else:
        laid_out = list(groups)
    allocations = layout_allocations(stages, laid_out, settings, gpu_kind=gpu_kind, partition=on_partition,
                                     job_name=job_name, singleton=singleton, advice=advice)
    by_name = {s.name: s for s in stages}
    for alloc in allocations:
        warnings += idle_gpu_warnings(alloc, [by_name[n] for n in alloc.stages],
                                      policy=policy if groups is None else "layout", min_gpus=floor)
    plan = Plan(instance=layout.root, store_root=settings.site.store_root, task=task, pipeline=pipeline,
                job_name=job_name, created_at=now, revision=1, lease_generation=1, code=code, toolchain=toolchain,
                settings=settings, stages=stages, allocations=allocations, warnings=warnings, grower=grower,
                measured=measured)
    if secrets is not None:
        held = _holding(plan.model_dump(mode="json"), secrets.secrets)
        if held:
            shown = Scrubber(secrets.secrets)
            raise PlanError("the plan would hold the value of a secret, which no plan may: "
                            + "; ".join(f"{shown.scrub(where)} holds {name}" for where, name in held))
    return plan


def _holding(data: Any, values: Mapping[str, str], where: str = "") -> list[tuple[str, str]]:
    """Each field of the JSON *data* whose text (a number's as JSON writes it), or key, holds one
    of *values*, with the secret's name; the path of the field as dotted keys and ``[i]`` for an
    item of a list."""
    found: list[tuple[str, str]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{where}.{key}" if where else str(key)
            found += [(path, name) for name, secret in values.items() if secret and secret in str(key)]
            found += _holding(value, values, path)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            found += _holding(value, values, f"{where}[{i}]")
    elif data is not None:
        # A number's JSON text can hold a secret made of digits.
        text = data if isinstance(data, str) else json.dumps(data)
        found += [(where, name) for name, secret in values.items() if secret and secret in text]
    return found
