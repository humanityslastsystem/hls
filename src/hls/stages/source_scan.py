"""The source_scan stage: every form of every secret (``hls.substrate.secret_forms``: its value,
and the host of a value that is an http(s) URL when it identifies a machine) in a submission's
source bundle, read while it is
plain, and published in this stage's own manifest before the package stage runs any of the
contestant's code.

A source bundle is a git pack, its objects compressed, so a secret's value the agent committed is
not in the bytes finalize's own scan reads. A manifest is the one way a record is published, and
a package that a contestant fault ends publishes none, so the scan is a stage of its own: what it
found stands whatever the package then does, a leak of one of our own secrets into the
contestant's tree among it, which finalize refuses the instance for. The stage runs in the runner
and stays light; both steps are harness roles on the host, and neither runs the contestant's code:
1. ``checkout`` (``hls.package.checkout``, as the package stage runs it): the bundle, the copy of
   the operator's named file that ``hls package`` made in the instance (``params.bundle``), the
   output of the agent_build manifest the plan pins (``params.build_manifest``, a grading that
   packages a rollout's source again) or this instance's agent_build output, checked by its size
   and digest, held to the build's writable disk cap (``params.disk_cap_gib``), and read once
   where it lies in the store into its clone, with no copy on the node. Before the tree's limits decide anything, it reads every object of the clone, the
   bundle's whole history, messages and tags among them, and writes the count of objects each form
   was found in (area ``bundle``) to ``package-scan.json``. Then it checks the tree out as the
   package will, or records why not, in ``source.json``;
2. ``package-scan`` (``hls.package.scan``), when a tree was checked out: every regular file of it,
   names starting with a dot among them, which the image will hold at ``/workspace``, added to
   that record (area ``tree``): each file by its path, with the form's label, never a value or a
   host.
A bundle the checkout never clones, one without ``main``, past the cap or that git does not read
back, gets no record, so nothing claims it was read: finalize keeps it out of any export as
unscanned. The commit point never reads its bundle back, so this stage's checkout is the first to
read it as git reads it. An agent_build that left nothing to package left nothing to scan, and the
stage publishes nothing. The values reach the roles only in their 0600 env files. On a site whose secrets file holds no secret nothing is read for
one: the stage writes a record that searched no secret and read nothing, which covers nothing on a
site that holds one.

The stage sets ``needs_secret_registry``, and a stage the runner did not hand the registry refuses
to run, before any role. ``source.json`` lists the tree's paths, which the contestant named, and
git's messages on the checkout's stderr can quote them, so both are contestant outputs
(``contestant_outputs``, ``contestant_log_roles``). ``package-scan.json`` holds no value or host: a
form in a path is recorded by its marker.

Whose failure it is: everything here is the harness's work on what the contestant left, so every
failure is infra. A bundle that is not the one its record or its operator's digest names, one the
store changed after the commit point wrote it among them, is refused and not retried, since a
retry would read the same file; any other failure of a role is retried. The checkout's verdict on
a bundle's content, every byte it read proven the bundle's, is no failure: ``source.json`` records
it (``over_limit``), and the package stage publishes the submission's zero."""

from __future__ import annotations

import math
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hls.agent.commit import KIND_CHECK_SEC, TREE_LIST_SEC
from hls.package.checkout import CHECKOUT_MEMORY_GIB, CHECKOUT_SEC, checkout_disk_gib
from hls.pipeline.stage import Stage, StageContext, StageFault, register, stage_fault
from hls.spec.resources import ResourceShape
from hls.stages.package import WORKSPACE, StoredBundle, check_named_bundle, left_nothing, run_checkout, source_bundle
from hls.stages.provision import role_shape
from hls.store.digest import Sha256Hex
from hls.store.manifest import ManifestKey, read_outputs_report
from hls.substrate.secret_forms import FORMS
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE, PackageScan

#: The checkout role, as the package stage names its own.
CHECKOUT = "checkout"
#: The role that scans the checked-out tree.
TREE_SCAN = "package-scan"
#: Seconds the stage allows beside the checkout's clone and object scan
#: (``hls.package.checkout.CHECKOUT_SEC``): in the checkout role, the tree's listing and the check
#: of the objects it names, each its own time (``hls.agent.commit.TREE_LIST_SEC``,
#: ``hls.agent.commit.KIND_CHECK_SEC``), and git's checkout; and the scan of the tree's files in the
#: second. The checkout and the scan are given twice the time they take at the tree's bounds on
#: two CPUs (200,000 paths of up to 1,008 bytes in 199,020 directories, 4 GiB of files over three
#: symbols): git's checkout takes 52.94 s and the scan 54.1 s. The roles' starts are the planner's
#: (``site.step_launch_allowance_sec``).
REST_SEC = TREE_LIST_SEC + KIND_CHECK_SEC + math.ceil(2 * (52.94 + 54.1))


def min_disk_gib(disk_cap_gib: int) -> int:
    """Node-local disk the stage needs for a bundle of a build whose writable disk cap is
    *disk_cap_gib*: the checkout's at the cap (``checkout_disk_gib``)."""
    return checkout_disk_gib(disk_cap_gib * 2**30)


#: The least time the stage may declare: the checkout's clone and object scan (``CHECKOUT_SEC``)
#: and ``REST_SEC``, fixed allowances. Each git command of the checkout runs to the stage's
#: deadline, which ends the role as infra.
MIN_TIME_SEC = CHECKOUT_SEC + REST_SEC


class SourceScanParams(BaseModel):
    """Which bundle the stage scans, and the cap it is held to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The source bundle its operator named, as ``hls package`` copied it into the store; None to
    #: read an agent_build output (``build_manifest``).
    bundle: StoredBundle | None = None
    #: The sha256 the operator states for ``bundle``, which the copy had when it was made and
    #: which what the checkout reads of it must have; None exactly when ``bundle`` is, since the
    #: agent_build manifest records its own.
    bundle_sha256: Sha256Hex | None = None
    #: Another instance's agent_build manifest, pinned in the stage's inputs, whose bundle (or its
    #: record that there is none) is scanned: a grading that packages a rollout's source again.
    #: None to read this instance's agent_build stage.
    build_manifest: ManifestKey | None = None
    #: The writable disk cap of the build the bundle is from, in GiB
    #: (``hls.pipeline.pipelines.sandbox_disk_gib``): the most bytes the bundle, and its objects
    #: inflated, may hold, as the commit point held the repository to it.
    disk_cap_gib: int = Field(gt=0)

    @model_validator(mode="after")
    def _checked(self) -> Self:
        check_named_bundle(self.bundle, self.bundle_sha256, self.build_manifest)
        return self


@register
class SourceScanStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "source_scan"
    params_model: ClassVar[type[BaseModel]] = SourceScanParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()
    #: The scans search for every value of the secrets file.
    needs_secret_registry: ClassVar[bool] = True
    #: ``source.json`` lists the tree's paths, which the contestant named (``Stage.contestant_outputs``).
    contestant_outputs: ClassVar[tuple[str, ...]] = ("source.json",)
    #: git's messages on the checkout's stderr can quote those paths (``Stage.contestant_log_roles``).
    contestant_log_roles: ClassVar[tuple[str, ...]] = (CHECKOUT,)

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the checkout and the tree's scan."""
        return 2

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel | None, *, runner_memory_gib: int) -> None:
        assert isinstance(params, SourceScanParams)
        if shape.gpus:
            raise ValueError("source_scan runs on the host side and asks for no GPU")
        disk = min_disk_gib(params.disk_cap_gib)
        if shape.disk_gib < disk:
            raise ValueError(f"source_scan needs at least {disk} GiB of node-local disk for the bundle's clone and "
                             f"the tree at the build's {params.disk_cap_gib} GiB cap; the stage declares "
                             f"{shape.disk_gib}")
        if shape.time_sec < MIN_TIME_SEC:
            raise ValueError(f"source_scan needs at least {MIN_TIME_SEC} s of time, the checkout's {CHECKOUT_SEC} s "
                             f"for the clone and the scan of its objects and {REST_SEC} s for the tree's listing, "
                             f"checkout and scan; the stage declares {shape.time_sec}")
        # One role at a time, of the stage's shape less the runner's memory, as the checkout needs.
        role = role_shape(shape, runner_memory_gib)
        if role.memory_gib < CHECKOUT_MEMORY_GIB:
            raise ValueError(f"source_scan's checkout needs {CHECKOUT_MEMORY_GIB} GiB of memory; the stage's "
                             f"{shape.memory_gib} GiB less the runner's {runner_memory_gib} gives its role "
                             f"{role.memory_gib}")

    def run(self, ctx: StageContext) -> dict[str, str]:
        if not ctx.secret_registry:
            raise StageFault.infra("source_scan runs only with the secrets registry, which the runner did not hand it: "
                                   "it could not scan what the bundle's compression hides from finalize",
                                   retryable=False)
        p: SourceScanParams = ctx.params
        if left_nothing(ctx, p.bundle, p.build_manifest) is not None:
            return {}
        bundle = source_bundle(ctx, p.bundle, p.bundle_sha256, p.build_manifest)
        shape = role_shape(ctx.stage.shape, ctx.plan.settings.run.runner_memory_gib)
        names = sorted(ctx.secrets)
        out = ctx.output_path(PACKAGE_SCAN_FILE)
        record, tree = run_checkout(ctx, bundle, shape, named=p.bundle is not None, max_bytes=p.disk_cap_gib * 2**30,
                                    scan=(names, out) if names else None)
        outputs = {"source": ctx.reported_output("source.json", CHECKOUT)}
        if not names:
            ctx.events.emit("package_scan_skipped", reason="the secrets file holds no secret")
            outputs["package_scan"] = ctx.write_output(PACKAGE_SCAN_FILE, PackageScan(
                secrets=[], forms=list(FORMS), findings=[], read=[]).model_dump_json(indent=2).encode())
            return outputs
        if PACKAGE_SCAN_FILE not in (read_outputs_report(ctx.outputs_report(CHECKOUT)) or {}):
            ctx.events.emit("source_scan_unscanned", reason=f"the checkout read no object of the bundle: "
                                                            f"{record.status}: {record.reason or record.status}")
            return outputs
        outputs["package_scan"] = ctx.reported_output(PACKAGE_SCAN_FILE, CHECKOUT)
        if record.status != "ok":
            return outputs
        argv = [str(ctx.plan.toolchain.python), "-m", "hls.package.scan", "--names", ",".join(names), "--tree",
                str(tree), "--tree-at", WORKSPACE, "--extend", str(out), "--out", str(out), "--outputs-out",
                str(ctx.outputs_report(TREE_SCAN))]
        ctx.start_role(TREE_SCAN, argv, shape, owner="harness", secret_env=names)
        fault = stage_fault(ctx.wait_roles())
        if fault is not None:
            raise StageFault(fault)
        outputs["package_scan"] = ctx.reported_output(PACKAGE_SCAN_FILE, TREE_SCAN)
        return outputs
