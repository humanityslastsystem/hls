"""L2 -- TaskSpec + compatibility checks.

Conjunction is a flat list[Verifier] (all must pass). The ``_check`` model
validator validates the hardware accelerator count and then calls
``check_wiring``, which (from config alone, before any GPU) confirms each
verifier's ``requires`` capabilities are provided by exactly one of its declared
reference engines and that protocol prerequisites hold. References live inside
Verifiers as their owned engine configs.

The Artifact is NOT on the TaskSpec -- a task module declares its ARTIFACT beside its TASK,
so one task hosts a swappable contestant.
"""

from __future__ import annotations
from typing import Any

from pydantic import ConfigDict, Field, model_serializer, model_validator

from hls.core.model import Hardware, ModelSpec
from hls.core.poly import Spec
from hls.drive.bounds import DriveBounds
from hls.egress.policy import NetworkPolicy
from hls.scoring import Scorer
from hls.verify import Verifier, check_wiring
from hls.spec.baseline import Baseline
from hls.spec.contracts import ArtifactContract
from hls.spec.datasource import DataSource
from hls.spec.tenant import DEFAULT_TENANT_ID, Tenant
from hls.spec.workload import Workload


class CompatibilityError(Exception):
    pass


class TaskSpec(Spec):
    """Complete specification of one evaluation task (L2).

    Fields:
        id:        Human-readable task identifier (e.g. ``"example-task"``).
        model:     Model specification (checkpoint, architecture, chat format).
        hardware:  GPU / CPU / memory profile for the artifact container.
        network:   RUNTIME egress policy for the deployed (rollout) artifact container
                   -- a peer of ``hardware``, NOT a hardware property. ``None`` (default)
                   means AIR_GAP at runtime (a served artifact gets no egress unless opted
                   in) -- the OPPOSITE of the producer phase, where
                   ``AgentProducerSpec.network=None`` means open. For a non-None policy the
                   posture is ``network.domain_posture()`` (open / air-gap / filter). The
                   deployed container has no egress proxy, so this is collapsed to harbor's
                   coarse launch gate at the deploy boundary; a non-trivial domain filter
                   is rejected there (no runtime proxy yet). Distinct from the
                   producer-build egress policy on ``AgentProducerSpec.network``.
        contract:  Wire protocol spec (input/output modalities, HTTP path).
        max_context_len: Context window, in tokens, the task's own server serves the
                   model in. The calibration pool's trusted members are built at it and
                   report it on every reply, so an agentic framework stops a calibration
                   trajectory to answer where it would stop a graded one. ``None`` (the
                   default) leaves every serving stack at its checkpoint's own
                   context.
        workload:  Arrival pattern, concurrency, SLO, adapter mix.  ``None``
                   only when the multi-tenant ``tenants`` list is provided.
        verifiers: Ordered list of Verifiers (each verified; all must pass).
                   ``None`` only when the multi-tenant ``tenants`` list is
                   provided.
        tenants:   Optional list of Tenants for multi-tenant tasks.  When set,
                   ``workload``, ``verifiers``, and ``data_source`` must all be
                   ``None`` -- those fields live per-tenant.  Mutually exclusive
                   with the single-tenant triple.
        scorer:    Aggregates the Trace into a scalar Score.
        data_source: Optional DataSource declaring the task's logical data needs
                   (weights + dataset + split). ``None`` means the task
                   provisions nothing -- no mount-mode data ingress at ``hls run``.
                   Must be ``None`` when ``tenants`` is set.
        baseline:  Optional reference serving system the task ships (its baseline
                   credit is measured on it). ``None`` = none declared.
        drive_bounds: What one drive of the task may make the harness hold (its trace's
                   cap and request bytes, ``hls.drive.bounds``). ``None``
                   = none declared, and the task cannot be evaluated. Only the task module
                   sets it: no task override may (``hls.taskref.FIXED_TASK_FIELDS``).
                   Excluded from every dump: the plan records the figures it enforces, and
                   no record or fingerprint of the task carries them.
        provision_memory_gib: Memory, in GiB, each role of the task's provision stage
                   gets: the roles run one at a time (the weights' download, the rows,
                   each tool bundle), and the planner adds the runner's reserve on top
                   (``hls.pipeline.pipelines.provision_shape``). A task declares it when
                   its largest provision role's peak, with a quarter more, passes the
                   planner's default (``PROVISION_ROLE_GIB``), as that rounded up.
                   ``None`` takes the default. Only the task module sets it: a site changes it by its
                   ``[resources.<task>.provision]`` table, never by a task override
                   (``hls.taskref.FIXED_TASK_FIELDS``). Excluded from every dump, as
                   ``drive_bounds`` is: it sizes a harness stage and no record or
                   fingerprint of the task carries it.
        metadata:  Free-form string annotations (theme, tier, host, ...).
        agent:     Optional ``AgentSpec`` (spec/agent.py) declaring the agentic
                   seam: the harness-side AgentFramework + ToolRunner + per-
                   trajectory budget. ``None`` (the default) means the task is
                   non-agentic and the drive uses the plain request/response
                   ``Driver``; when set, the drive selects a
                   ``TrajectorySessionDriver`` (``hls.spec.drivers.driver_for``).
                   Excluded from provenance: the framework/tool wiring is
                   rederivable from the task module, not serializable config.
                   Typed ``Any`` to keep the live framework/tool objects off the
                   pydantic core.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    model: ModelSpec
    hardware: Hardware
    network: NetworkPolicy | None = None   # runtime egress for the deployed artifact
    contract: ArtifactContract
    max_context_len: int | None = None
    workload: "Workload | None" = None
    verifiers: "list[Verifier] | None" = None
    tenants: "list[Tenant] | None" = Field(default=None, exclude=True)
    scorer: Scorer
    data_source: "DataSource | None" = None
    # The task's reference serving system, on which its baseline credit is measured; None = none declared.
    baseline: "Baseline | None" = None
    # What one drive of the task may make the harness hold; None = none declared.
    drive_bounds: DriveBounds | None = Field(default=None, exclude=True)
    # Memory each provision role of the task gets, in GiB; None = the planner's default.
    provision_memory_gib: int | None = Field(default=None, gt=0, exclude=True)
    metadata: dict[str, str] = Field(default_factory=dict)
    seed: int = 0
    agent: Any = Field(default=None, exclude=True)    # optional AgentSpec (agentic seam); rederivable task-module code

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any, info: Any) -> dict:
        """Serialize to the UNIFIED per-tenant shape regardless of authoring form.

        Config always lives under a ``tenants`` array -- never the top-level
        single-tenant triple -- so every dump (single- or multi-tenant) is read
        the same way (e.g. by the results viewer). The two single-tenant authoring
        forms (top-level ``data_source/workload/verifiers`` vs an explicit
        ``tenants=[Tenant(id="t0", ...)]``) therefore produce IDENTICAL dumps:
        ``effective_tenants()`` canonicalizes both to one ``Tenant(id="t0", ...)``.

        Single-tenant records carry a ``tenants`` key. That is safe -- provenance is
        write-only (``poly.py``: no round-trip, nothing reconstructs a spec from the
        dump). RUNTIME behavior is unaffected: provisioning mounts, the provisioning
        digest/cache key, and per-tenant seeds branch on the live
        ``effective_tenants()``, not on this dump.
        """
        out: dict = handler(self)
        # Config lives ONLY under tenants[] in the dump. Drop the top-level
        # single-tenant triple (the canonical copy is each tenant's own fields).
        for key in ("workload", "verifiers", "data_source"):
            out.pop(key, None)
        # ``tenants`` is exclude=True on the field, so the handler never emits it;
        # inject the canonical list here. mode=info.mode threads the caller's dump
        # mode (python vs json) so a model_dump() stays all-python and a
        # model_dump(mode="json") stays all-json. serialize_as_any=True is required
        # so Poly subfields (Verifier, arrival, ...) emit their subclass fields and
        # ``kind`` discriminators (without it pydantic serializes by the base schema
        # and drops subclass data).
        out["tenants"] = [
            t.model_dump(mode=info.mode, serialize_as_any=True)
            for t in self.effective_tenants()
        ]
        return out

    @model_validator(mode="after")
    def _check(self) -> "TaskSpec":
        if self.tenants is None:
            # Single-tenant form: workload + verifiers are required.
            if self.workload is None or self.verifiers is None:
                raise CompatibilityError(
                    "TaskSpec single-tenant form requires workload and verifiers")
        else:
            # Multi-tenant form: top-level triple must be absent.
            if any(v is not None for v in (self.data_source, self.workload, self.verifiers)):
                raise CompatibilityError(
                    "TaskSpec: when tenants is set, top-level data_source/workload/verifiers "
                    "must be None (they live per-tenant)")
            if not self.tenants:
                raise CompatibilityError("TaskSpec.tenants must be non-empty")
            ids = [t.id for t in self.tenants]
            if len(set(ids)) != len(ids):
                raise CompatibilityError(f"duplicate tenant ids: {ids}")
        if self.hardware.accelerator.count < 1:
            raise CompatibilityError("hardware.accelerator.count must be >= 1")
        if self.agent is not None:
            # The framework parses this model's replies, so it gets a veto on the pair.
            self.agent.framework.check_model(self.model)
        for t in self.effective_tenants():
            check_wiring(t.verifiers, self.contract,
                         getattr(t.workload, "slo", None))
        return self

    def effective_tenants(self) -> "list[Tenant]":
        """Return the canonical tenant list regardless of authoring form.

        Single-tenant tasks return a one-element list with a synthesized
        Tenant(id="t0") carrying the top-level data_source/workload/verifiers.
        Multi-tenant tasks return the authored tenants list verbatim.
        """
        if self.tenants is not None:
            return self.tenants
        return [Tenant(id=DEFAULT_TENANT_ID, data_source=self.data_source,
                       workload=self.workload, verifiers=self.verifiers)]
