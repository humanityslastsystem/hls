"""The no-op stage, which does no work and checks the substrate around it: the stage of the
minimal pipeline the test suite and a site's live tests run.

It starts the roles its parameters ask for, each in its own step, lets them sleep, and
publishes what each saw. It refuses to publish when the steps were not isolated the way
they must be: exclusive GPU roles each see only their own GPUs, a role
with no GPUs sees none, and the shared role sees exactly the GPUs of the role it shares
with. It can run its first role as the contestant's, to exercise how its OOM kills are charged,
hold memory in the batch step beside its roles, to push the job past its own memory limit, and
fail its first attempts, to exercise retries."""

from __future__ import annotations

import json
import subprocess
import time
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from hls.pipeline.stage import Stage, StageContext, StageFault, budget_problem, register, stage_fault
from hls.spec.resources import ResourceShape
from hls.substrate.noop_role import NoopRoleReport

#: CPUs and memory of each GPU role and of the CPU role. The CPU count is even because a
#: site that allocates whole cores would round an odd request up, and the role's grant check
#: would then refuse it.
ROLE_CPUS, ROLE_MEMORY_GIB = 2, 2
#: CPUs and memory of the shared role.
SHARED_CPUS, SHARED_MEMORY_GIB = 2, 1
#: Seconds the stage gives its roles to take their memory before it holds memory in the batch
#: step.
BATCH_HOG_DELAY_SEC = 2.0


class NoopParams(BaseModel):
    """What the no-op stage runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Seconds each role sleeps after it reports.
    sleep_sec: int = Field(default=0, ge=0)
    #: Exclusive GPU roles that run at once; the stage's GPUs are split evenly among them.
    gpu_roles: int = Field(default=1, ge=0)
    #: Also run a role on the last GPU role's GPUs, the way the reference service shares
    #: the build's.
    shared_role: bool = False
    #: Also run a role with no GPUs.
    cpu_role: bool = False
    #: Run the first role as the contestant's, so that its OOM kills are charged by the
    #: contestant rules. Its step wrapper starts it with oom_score_adj 1000, as it does every
    #: contestant role.
    contestant_role: bool = False
    #: GiB the stage holds in the batch step, in a process of its own, while its roles run, to
    #: push the job past its own memory limit.
    batch_hog_gib: int = Field(default=0, ge=0)
    #: Attempts up to this number fail with an infra fault before starting any role.
    fail_attempts: int = Field(default=0, ge=0)


def role_shapes(shape: ResourceShape, p: NoopParams) -> dict[str, ResourceShape]:
    """Each role's shape, by role name, in start order."""
    shapes: dict[str, ResourceShape] = {}
    per = shape.gpus // p.gpu_roles if p.gpu_roles else 0
    for i in range(p.gpu_roles):
        shapes[f"gpu{i}"] = ResourceShape(gpus=per, gpu_kind=shape.gpu_kind, cpus=ROLE_CPUS,
                                          memory_gib=ROLE_MEMORY_GIB, time_sec=shape.time_sec)
    if p.cpu_role:
        shapes["cpu"] = ResourceShape(cpus=ROLE_CPUS, memory_gib=ROLE_MEMORY_GIB, time_sec=shape.time_sec)
    if p.shared_role:
        shapes["shared"] = ResourceShape(gpus=per, gpu_kind=shape.gpu_kind, cpus=SHARED_CPUS,
                                         memory_gib=SHARED_MEMORY_GIB, time_sec=shape.time_sec)
    return shapes


def isolation_problems(seen: dict[str, NoopRoleReport], p: NoopParams, per: int) -> list[str]:
    """Every way the roles' views of the GPUs break the isolation rules."""
    problems = []
    gpu_names = [f"gpu{i}" for i in range(p.gpu_roles)]
    taken: dict[str, str] = {}
    for name in gpu_names:
        uuids = seen[name].cuda_uuids
        if uuids is None or len(uuids) != per:
            problems.append(f"{name} sees GPUs {uuids}, not {per}")
            continue
        for u in uuids:
            if u in taken:
                problems.append(f"{name} and {taken[u]} both see {u}")
            taken[u] = name
    if p.cpu_role and seen["cpu"].cuda_uuids:
        problems.append(f"cpu sees GPUs {seen['cpu'].cuda_uuids}")
    if p.shared_role and seen["shared"].cuda_uuids != seen[gpu_names[-1]].cuda_uuids:
        problems.append(f"shared sees {seen['shared'].cuda_uuids}, not {gpu_names[-1]}'s "
                        f"{seen[gpu_names[-1]].cuda_uuids}")
    return problems


@register
class NoopStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "noop"
    params_model: ClassVar[type[BaseModel]] = NoopParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()
    #: Its first role may run as the contestant's on the host (``contestant_role``), which no
    #: scored stage may do: the OOM rules are exercised on a plain step, with no image to pin.
    contestant_host_roles: ClassVar[bool] = True
    #: It may start a contestant role, whose logs then hold the contestant's bytes
    #: (``Stage.runs_contestant_code``).
    runs_contestant_code: ClassVar[bool] = True

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the GPU roles, the shared one and the CPU one *params* asks
        for."""
        assert isinstance(params, NoopParams)
        return params.gpu_roles + params.shared_role + params.cpu_role

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        p = NoopParams.model_validate(params.model_dump())
        if p.gpu_roles > shape.gpus:
            raise ValueError(f"{p.gpu_roles} GPU roles need at least {p.gpu_roles} GPUs; the stage has {shape.gpus}")
        if p.gpu_roles and shape.gpus % p.gpu_roles:
            raise ValueError(f"{shape.gpus} GPUs do not split evenly among {p.gpu_roles} roles")
        if not p.gpu_roles and shape.gpus:
            raise ValueError("a stage with GPUs needs at least one GPU role")
        if p.shared_role and not p.gpu_roles:
            raise ValueError("the shared role needs a GPU role to share with")
        shapes = role_shapes(shape, p)
        if not shapes:
            raise ValueError("the stage starts no role")
        problem = budget_problem(shape, [(s, name == "shared") for name, s in shapes.items()],
                                 runner_memory_gib=runner_memory_gib)
        if problem is not None:
            raise ValueError(problem)

    def run(self, ctx: StageContext) -> dict[str, str]:
        p: NoopParams = ctx.params
        if ctx.attempt <= p.fail_attempts:
            raise StageFault.infra(f"planted fault on attempt {ctx.attempt} of {p.fail_attempts}", retryable=True)
        shapes = role_shapes(ctx.stage.shape, p)
        handles = {}
        for i, (name, shape) in enumerate(shapes.items()):
            argv = [str(ctx.plan.toolchain.python), "-m", "hls.substrate.noop_role",
                    "--out", str(ctx.output_path(f"{name}.json")),
                    "--outputs-out", str(ctx.outputs_report(name)), "--sleep-sec", str(p.sleep_sec)]
            contestant = i == 0 and p.contestant_role
            share = handles[f"gpu{p.gpu_roles - 1}"] if name == "shared" else None
            handles[name] = ctx.start_role(name, argv, shape, owner="contestant" if contestant else "harness",
                                           share_gpus_of=share)
        hog = _hold_batch_memory(ctx, p.batch_hog_gib) if p.batch_hog_gib else None
        try:
            outcomes = ctx.wait_roles()
        finally:
            if hog is not None:
                hog.kill()
                hog.wait()
        fault = stage_fault(outcomes)
        if fault is not None:
            raise StageFault(fault)
        seen = {name: NoopRoleReport.model_validate_json(ctx.output_path(f"{name}.json").read_bytes())
                for name in shapes}
        blind = [name for name, shape in shapes.items() if shape.gpus and seen[name].cuda_uuids is None]
        if blind:
            raise StageFault.infra(f"roles {', '.join(blind)} could not read their GPUs from the driver",
                                   retryable=True)
        per = ctx.stage.shape.gpus // p.gpu_roles if p.gpu_roles else 0
        problems = isolation_problems(seen, p, per)
        if problems:
            raise StageFault.infra("role steps were not isolated: " + "; ".join(problems), retryable=False)
        outputs = {name: ctx.reported_output(f"{name}.json", name) for name in shapes}
        summary = {name: r.model_dump() for name, r in seen.items()}
        outputs["summary"] = ctx.write_output("summary.json", json.dumps(summary, sort_keys=True).encode())
        return outputs


def _hold_batch_memory(ctx: StageContext, gib: int) -> subprocess.Popen:
    """Start a process in the batch step, beside the runner, that holds *gib* GiB until it is
    killed, once the roles have had ``BATCH_HOG_DELAY_SEC`` to take their own memory."""
    time.sleep(BATCH_HOG_DELAY_SEC)
    logs = ctx.attempt_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    argv = [str(ctx.plan.toolchain.python), "-m", "hls.substrate.noop_role", "--out",
            str(logs / "batch-hog.json"), "--hog-gib", str(gib), "--sleep-sec", str(ctx.stage.shape.time_sec)]
    with open(logs / "batch-hog.out", "ab") as out:
        return subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
