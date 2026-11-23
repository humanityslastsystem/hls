"""The provision stage: a task's weights, rows and tool bundles into the content-addressed
cache, each skipped on a hit.

The stage runs in the runner and stays light. Every byte moves in a harness role step on the
host side, never in a container and never in the runner: the role program
``hls.provision.role`` runs from the plan's checkout with the plan's interpreter, one role at
a time, each with the stage's shape less the memory the stage leaves the runner
(``run.runner_memory_gib``) and no disk (``role_shape``). In order:
1. ``resolve`` works out what the task needs, with every hub revision pinned to a commit,
   and writes it to the attempt's ``outputs/resolved.json``;
2. the task model's weights entry is looked up and, on a miss, filled by a ``weights`` role
   under the entry's lock. The role also writes the hub's file list to
   ``outputs/source-0.json``, and the fill publishes only when the files the cache scanned are
   exactly those, with the same sizes and the hashes the role reported. A caller that
   needs only the rows skips this step;
3. the rows entry is filled by a ``rows`` role; it carries its own row ids in ``ids.json``;
4. the tool bundle's ``full`` variant, then its ``dev`` variant, keyed by the partition the
   published rows entry holds, are filled by ``tools`` roles.
The roles that load the task (resolve, rows, tools) get the plan's task reference, its module
and its task overrides (``task_args``), so what is provisioned is the task the plan's other
stages work on: an override of a split changes the rows entry's key and its rows.

Each fill role hashes and syncs every file it wrote and reports the digests into its own
node-local directory (``--digests-out``), which the fill hands the cache, so the runner reads
no file of an entry it fills. The one entry file it reads is a rows entry's
``ids.json``, kilobytes of row ids, checked against its manifest, to key the dev bundle.

The output ``provisioned.json`` names every entry with the digest of its ``ENTRY.json`` and the
disk its node-local copy takes (``copy_disk_bytes``), and each hit, fill and wait for another
provision is an event naming the entry. A planner reads the same record from the cache alone,
before any stage runs (``cached_provision``), to declare the copies the stages that copy the
entries make (``hls.provision.mounts.copied_entries``).

All or nothing: an entry is published whole or not at all, and whatever fails publishes
nothing. Whose the failure is decides whether it is retried:
- a role that fails or reports no digests, a download that does not match the hub, or a wait
  for another provision cut short by the deadline or a stop: a retryable infra fault;
- a task a role refuses as declared, a hub that answers what a role asks for with 401, 403
  or 404, a fill the cache refuses (it takes plain files only), and a published entry that
  is corrupt or whose ``ids.json`` cannot be used: an infra fault that is not retried, since
  a rerun would meet the same.
A rerun finds what an earlier attempt published. A second provision of the same entries waits
on the entry's lock while its holder renews it, then finds the entry.

The hub token is the stage's grant ``HF_TOKEN`` (``stage.provision`` in the
secrets file). A role that reads from the hub (``HUB_COMMANDS``: not resolve, which reads the
task's declaration alone) gets it through the 0600 env file the step wrapper takes before the
role starts, and only when the plan granted it; no argument or declared variable carries it,
and a role step's environment holds nothing of the runner's own. Each role's ``HF_HOME`` is
its own node-local directory, removed with it, so no token or hub cache of another process
is in reach."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hls.pipeline.stage import Stage, StageContext, StageFault, budget_problem, register, stage_fault
from hls.provision.hf import SourceFile, check_against_source
from hls.provision.model import (ModelRevision, Provisioned, Resolved, RowsRef, ToolRef, ToolSource, WeightsRef,
                                 tool_key, weights_key)
from hls.provision.role import HUB_REFUSED_EXIT, IDS_FILE, REFUSED_EXIT, resolve
from hls.provision.rows import dev_digest, tenant_subpath
from hls.spec.resources import ResourceShape
from hls.store.cache import (CacheCorrupt, CacheEntry, ContentCache, FileRecord, FillCheck, WaitEnded, copy_disk_bytes,
                             entry_sha256, subtree_disk_bytes,
                             read_digests)
from hls.store.digest import sha256_bytes
from hls.store.lock import Holder
from hls.taskref import TaskRef

#: Seconds between looks for a published entry while another provision fills it.
POLL_SEC = 5.0
#: The grant a role gets the hub token through.
TOKEN_GRANT = "HF_TOKEN"
#: The role program commands that read from the hub, and so get the token when it is granted.
#: Resolve reads the task's declaration alone.
HUB_COMMANDS = frozenset({"weights", "rows", "tools"})
#: The role program, run as ``<plan python> -m ROLE_MODULE <command>``.
ROLE_MODULE = "hls.provision.role"


class ProvisionParams(BaseModel):
    """What the provision stage provisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task to provision: its module, importable from the checkout root, and the task overrides
    #: the plan's other stages apply, which can change its rows and tool bundles.
    task: TaskRef
    #: False provisions the rows and tool bundles without the task model's weights, for a
    #: caller that needs only the rows.
    weights: bool = True
    #: The site's sandbox stores the task's tool runner names, which a ``stores`` role makes what
    #: the runner needs in (``ToolRunner.provision_stores``); empty for a task whose runner names
    #: none, and then left out of the dump.
    sandbox_stores: tuple[str, ...] = Field(default=(), exclude_if=lambda stores: not stores)


class _RowIds(BaseModel):
    """A rows entry's ``ids.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The dev row ids, deduplicated, in write order.
    dev: list[str]
    #: The test row ids, deduplicated, in write order.
    test: list[str]


def role_shape(shape: ResourceShape, runner_memory_gib: int) -> ResourceShape:
    """The shape of each role of a host-side stage that runs one role at a time, whose stage has
    *shape*: the stage's, less the *runner_memory_gib* GiB the stage leaves the runner
    (``run.runner_memory_gib``), and no disk. The stage's declared disk, which the runner checks
    when the job starts, is the one budget for what its roles write. The step wrapper checks a
    role's declared disk against the free disk of the node-local scratch when the role starts,
    so a role that declared the stage's would fail as soon as the scratch held anything. Raise
    ValueError when the stage's memory leaves a role none."""
    left = shape.memory_gib - runner_memory_gib
    if left < 1:
        raise ValueError(f"the stage declares {shape.memory_gib} GiB, which leaves its roles none beside the "
                         f"runner's {runner_memory_gib} GiB")
    return shape.model_copy(update={"memory_gib": left, "disk_gib": 0})


def read_row_ids(rows: CacheEntry) -> _RowIds:
    """The row ids the published *rows* entry holds in its ``ids.json``, read once and checked
    against the entry's manifest: kilobytes, which key the dev tool bundle. Raise ValueError,
    naming the file and never its bytes, when the manifest lists no such file or the file is
    missing, differs or does not parse."""
    recorded = next((f.sha256 for f in rows.manifest.files if f.path == IDS_FILE), None)
    try:
        if recorded is None:
            raise ValueError(f"its manifest lists no {IDS_FILE}")
        raw = (rows.path / IDS_FILE).read_bytes()
        if sha256_bytes(raw) != recorded:
            raise ValueError("it differs from its manifest")
        return _RowIds.model_validate_json(raw)
    except ValidationError as e:
        why = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'the file'}: {err['msg']}" for err in e.errors())
        raise ValueError(f"{IDS_FILE} cannot be used: {why}") from None
    except (OSError, ValueError) as e:
        raise ValueError(f"{IDS_FILE} cannot be used: {e}") from None


def weights_ref(model: ModelRevision, entry: CacheEntry) -> WeightsRef:
    """The provision record's entry for the task model's weights published as *entry*."""
    return WeightsRef(repo=model.repo, revision=model.revision, key=entry.key, entry_sha256=entry_sha256(entry),
                      disk_bytes=copy_disk_bytes(entry))


def rows_ref(resolved: Resolved, entry: CacheEntry) -> RowsRef:
    """The provision record's entry for the rows *resolved* names, published as *entry*."""
    return RowsRef(key=entry.key, entry_sha256=entry_sha256(entry), tenants=resolved.tenants,
                   disk_bytes=copy_disk_bytes(entry),
                   dev_disk_bytes={t: subtree_disk_bytes(entry, f"{tenant_subpath(t)}/dev") for t in resolved.tenants})


def tool_ref(bundle: ToolSource, variant: str, digest: str | None, entry: CacheEntry) -> ToolRef:
    """The provision record's entry for the *variant* of tool *bundle*, keyed by the partition
    *digest* for the dev one, published as *entry*."""
    return ToolRef(name=bundle.name, identity=bundle.identity, variant=variant, dev_digest=digest, key=entry.key,
                   entry_sha256=entry_sha256(entry), disk_bytes=copy_disk_bytes(entry))


class NotProvisioned(Exception):
    """The cache does not hold every entry a task needs."""

    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"the cache holds no {', '.join(missing)}")
        #: Each entry the cache lacks, as ``<kind>/<key>``.
        self.missing = missing


def cached_provision(task: Any, cache: ContentCache, *, weights: bool = True) -> Provisioned:
    """The provision record a provision of *task* (its ``TASK``) would publish now, read from
    *cache* alone and nothing fetched, as a planner reads it before any stage runs: every entry
    the task needs, with the disk its node-local copy takes, which the stages that copy them
    declare (``hls.provision.mounts.copied_entries``). Without *weights*, the record of a
    provision of everything but the task model's weights. The task is resolved as the resolve
    role resolves it (``hls.provision.role.resolve``), and the dev tool bundle keyed by the rows
    entry's ids (``read_row_ids``). Raise NotProvisioned naming every entry the cache lacks (the
    dev bundle only once the rows are there, since they key it), ``hls.provision.role.Refused``
    for a task the resolve role refuses, ValueError for a rows entry whose ids cannot be used,
    and CacheCorrupt for an entry the cache refuses."""
    resolved = resolve(task)
    missing: list[str] = []

    def found(kind: str, key: str) -> CacheEntry | None:
        entry = cache.lookup(kind, key)
        if entry is None:
            missing.append(f"{kind}/{key}")
        return entry

    model = resolved.weights
    entry = found("weights", weights_key(model.repo, model.revision)) if weights else None
    weights_record = weights_ref(model, entry) if entry is not None else None
    rows, tools = None, []
    if resolved.rows_key is not None:
        rows_entry = found("rows", resolved.rows_key)
        rows = rows_ref(resolved, rows_entry) if rows_entry is not None else None
        bundle = resolved.tools
        if bundle is not None:
            variants: list[tuple[str, str | None]] = [("full", None)]
            if rows_entry is not None:
                ids = read_row_ids(rows_entry)
                variants.append(("dev", dev_digest(ids.dev, ids.test)))
            for variant, digest in variants:
                tool = found("tools", tool_key(bundle.name, bundle.identity, variant, digest))
                if tool is not None:
                    tools.append(tool_ref(bundle, variant, digest, tool))
    if missing:
        raise NotProvisioned(missing)
    return Provisioned(weights=weights_record, rows=rows, tools=tools)


def _blob_ids(path: Path, role: str) -> dict[str, str]:
    """The git blob ids role *role* reported in *path*, by file. Raise StageFault (infra,
    retryable) when it reported none that can be read."""
    try:
        blobs = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise StageFault.infra(f"role {role} reported no git blob ids: {e}", retryable=True, role=role) from e
    if not isinstance(blobs, dict) or not all(isinstance(v, str) for v in blobs.values()):
        raise StageFault.infra(f"role {role} reported git blob ids that are not a map of strings", retryable=True,
                               role=role)
    return blobs


class _Provision:
    """One attempt's role steps and cache fills."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.cache is None:
            raise RuntimeError("the provision stage needs the store's cache")
        self.ctx = ctx
        self.params: ProvisionParams = ctx.params
        #: The cache's lock holder for this attempt's fills.
        self.holder = Holder.here(f"provision {ctx.plan.instance}", job_id=ctx.job_id)
        #: Output name to store key, for what this attempt wrote.
        self.outputs: dict[str, str] = {}

    def task_args(self) -> tuple[str, ...]:
        """The options that name a role's task: its module, and the plan's task overrides as JSON
        when there are any."""
        ref = self.params.task
        return ("--task-module", ref.module, *(("--overrides", json.dumps(ref.overrides)) if ref.overrides else ()))

    def run_role(self, name: str, *args: str) -> None:
        """Run the role program's command *args* as harness role *name* and wait for it.
        Raise StageFault when it does not end cleanly: not retryable when the role refused the
        task as declared or the hub refused what it asked for, each told by its exit status and
        a ``refused:`` line on stderr (``StageContext.refusal``); retryable otherwise."""
        ctx = self.ctx
        argv = [str(ctx.plan.toolchain.python), "-m", ROLE_MODULE, *args]
        env = {"PYTHONPATH": str(ctx.plan.code.checkout), "HF_HOME": str(ctx.role_dir(name) / "hf")}
        granted = (TOKEN_GRANT,) if TOKEN_GRANT in ctx.secrets and args[0] in HUB_COMMANDS else ()
        shape = role_shape(ctx.stage.shape, ctx.plan.settings.run.runner_memory_gib)
        ctx.start_role(name, argv, shape, env=env, secret_env=granted, owner="harness")
        outcomes = ctx.wait_roles()
        fault = stage_fault(outcomes)
        if fault is None:
            return
        refused = ctx.refusal(name, outcomes, (REFUSED_EXIT, HUB_REFUSED_EXIT))
        module = self.params.task.module
        if refused is not None and refused.exit_code == REFUSED_EXIT:
            fault = fault.model_copy(update={"reason": f"the provision role {name} refused task module {module}: "
                                                       f"{refused.reason}", "retryable": False})
        elif refused is not None:
            fault = fault.model_copy(update={"reason": f"the hub refused what role {name} asked for task module "
                                                       f"{module}: {refused.reason}", "retryable": False})
        raise StageFault(fault)

    def entry(self, kind: str, key: str, command: tuple[str, ...], *, role: str, check: FillCheck | None = None,
              source: dict[str, str] | None = None) -> CacheEntry:
        """The published entry *kind*/*key*. When nobody has published it, the role program's
        *command* runs as role *role* with ``--out`` the staging directory and
        ``--digests-out`` a file in the role's own directory, where it reports the digest of
        each file it wrote, which the cache records; *check* sees what it wrote. *source* goes
        into the entry's manifest beside the stage and instance. While another provision fills
        the entry this waits, until the stage's deadline at most and not once it is stopping.
        Raise StageFault for a wait cut short and for a role that reported no digests (both
        retryable), and for a corrupt published entry and a fill the cache refused (neither
        retryable)."""
        ctx = self.ctx
        what = f"{kind}/{key}"
        report = ctx.role_dir(role) / "digests.json"

        def fill(staging: Path) -> None:
            self.run_role(role, *command, "--digests-out", str(report), "--out", str(staging))

        def digests() -> dict[str, str]:
            reported = read_digests(report)
            if reported is None:
                raise StageFault.infra(f"role {role} reported no digests of the files it wrote", retryable=True,
                                       entry=what, role=role)
            return reported

        def waiting(holder: Holder) -> None:
            ctx.events.emit("cache_wait", entry_kind=kind, key=key, holder_purpose=holder.purpose,
                            holder_host=holder.host, holder_pid=holder.pid, holder_job_id=holder.job_id)

        try:
            got = ctx.cache.populate(kind, key, fill, holder=self.holder, digests=digests, check=check,
                                     poll_sec=POLL_SEC,
                                     source={"stage": ctx.stage.name, "instance": ctx.plan.instance, **(source or {})},
                                     deadline=ctx.deadline, stopping=lambda: ctx.stopping, on_wait=waiting)
        except WaitEnded as e:
            raise StageFault.infra(str(e), retryable=True, entry=what) from e
        except CacheCorrupt as e:
            try:
                ctx.cache.lookup(kind, key)
            except CacheCorrupt as published:
                raise StageFault.infra(f"cache entry {what} is corrupt: {published}", retryable=False,
                                       entry=what) from e
            raise StageFault.infra(f"the fill of {what} was refused: {e}", retryable=False, entry=what) from e
        ctx.events.emit("cache_filled" if got.published else "cache_hit", entry_kind=kind, key=key,
                        entry_sha256=entry_sha256(got.entry))
        return got.entry

    def resolve(self) -> Resolved:
        """Run the resolve role and read what it resolved."""
        name = "resolved.json"
        path = self.ctx.output_path(name)
        self.run_role("resolve", "resolve", *self.task_args(), "--out", str(path),
                      "--outputs-out", str(self.ctx.outputs_report("resolve")))
        self.outputs["resolved"] = self.ctx.reported_output(name, "resolve")
        return Resolved.model_validate_json(path.read_bytes())

    def weights(self, model: ModelRevision) -> WeightsRef:
        """The task model's weights entry, checked against the hub's file list when filled, by
        the hashes its role reported: the cache's record of each file and the git blob ids."""
        key = weights_key(model.repo, model.revision)
        name = "source-0.json"
        listing = self.ctx.output_path(name)
        role = "weights-0"
        blobs_file = self.ctx.role_dir(role) / "blobs.json"

        def check(files: list[FileRecord]) -> None:
            source = {p: SourceFile.model_validate(f) for p, f in json.loads(listing.read_text()).items()}
            problems = check_against_source(files, _blob_ids(blobs_file, role), source)
            if problems:
                raise StageFault.infra(f"weights {model.repo}@{model.revision} do not match the hub: "
                                       + "; ".join(problems), retryable=True, entry=f"weights/{key}")
            self.outputs["source-0"] = self.ctx.reported_output(name, role)

        found = self.entry("weights", key, ("weights", "--repo", model.repo, "--revision", model.revision,
                                            "--source-out", str(listing),
                                            "--outputs-out", str(self.ctx.outputs_report(role)),
                                            "--blobs-out", str(blobs_file)),
                           role=role, check=check, source={"repo": model.repo, "revision": model.revision})
        return weights_ref(model, found)

    def row_ids(self, rows: CacheEntry) -> _RowIds:
        """The row ids the published *rows* entry holds (``read_row_ids``). Raise StageFault, not
        retryable, when they cannot be used."""
        try:
            return read_row_ids(rows)
        except ValueError as e:
            what = f"rows/{rows.key}"
            raise StageFault.infra(f"cache entry {what}: {e}", retryable=False, entry=what) from e

    def tools(self, bundle: ToolSource, rows: CacheEntry) -> list[ToolRef]:
        """Both variants of the tool bundle, the dev one keyed by the partition the published
        *rows* entry holds. The ``tools`` roles read the ``ids.json`` that ``row_ids``
        checked."""
        ids = self.row_ids(rows)
        refs = []
        for variant, digest in (("full", None), ("dev", dev_digest(ids.dev, ids.test))):
            key = tool_key(bundle.name, bundle.identity, variant, digest)
            found = self.entry("tools", key, ("tools", *self.task_args(), "--variant", variant,
                                              "--ids", str(rows.path / IDS_FILE)), role=f"tools-{variant}")
            refs.append(tool_ref(bundle, variant, digest, found))
        return refs

    def stores(self, rows: CacheEntry | None) -> None:
        """Run the ``stores`` role, which makes in the site's sandbox stores what the task's
        tool runner needs for the corpus the published *rows* entry holds
        (``ToolRunner.provision_stores``), with each store's directory as the site names it."""
        named = self.ctx.plan.settings.sandbox_stores
        stores = {name: str(named[name]) for name in self.params.sandbox_stores}
        self.run_role("stores", "stores", *self.task_args(), "--stores", json.dumps(stores),
                      *(("--rows", str(rows.path)) if rows is not None else ()))
        self.ctx.events.emit("sandbox_stores_provisioned", stores=sorted(stores))

    def run(self) -> dict[str, str]:
        resolved = self.resolve()
        weights = self.weights(resolved.weights) if self.params.weights else None
        rows, tools, found = None, [], None
        if resolved.rows_key is not None:
            found = self.entry("rows", resolved.rows_key, ("rows", *self.task_args()),
                               role="rows")
            rows = rows_ref(resolved, found)
            if resolved.tools is not None:
                tools = self.tools(resolved.tools, found)
        if self.params.sandbox_stores:
            self.stores(found)
        provisioned = Provisioned(weights=weights, rows=rows, tools=tools)
        self.outputs["provisioned"] = self.ctx.write_output("provisioned.json",
                                                            provisioned.model_dump_json(indent=2).encode())
        return self.outputs


@register
class ProvisionStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "provision"
    params_model: ClassVar[type[BaseModel]] = ProvisionParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: resolve, the weights, the rows, the tool bundle's two
        variants, and the sandbox stores' role for a task whose runner names stores."""
        stores = isinstance(params, ProvisionParams) and bool(params.sandbox_stores)
        return 6 if stores else 5

    @classmethod
    def sandbox_stores(cls, params: Any) -> tuple[str, ...]:
        """The sandbox stores the ``stores`` role writes, which the site must name."""
        assert isinstance(params, ProvisionParams)
        return params.sandbox_stores

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        if shape.gpus:
            raise ValueError("provision runs on the host side and asks for no GPU")
        problem = budget_problem(shape, [(role_shape(shape, runner_memory_gib), False)],
                                 runner_memory_gib=runner_memory_gib)
        if problem is not None:
            raise ValueError(problem)

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Provision(ctx).run()
