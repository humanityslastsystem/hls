"""Tasks, arms, site files and earlier builds for the ``hls run`` tests.

The task is the ``hls grade`` tests' task (``tests.evaluate.fixture``), whose model and
corpus the fake hub serves, on hardware with room for the sandbox's ``/dev/shm`` and the
harness's reserve, with an agent-produced artifact whose wall-clock budgets the runner keeps.
``conftest.py`` writes each variant into a checkout of the tests' own and adds the task to the
image catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hls.agent.commit import CommitRecord
from hls.agent.preflight import PreflightResult
from hls.arms import ArmRecord, Route
from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec, Budget, BuildAllocation,
                                   EvaluationAllocation, HighestAvailable, ProviderLevel)
from hls.core.model import AcceleratorSpec, Hardware
from hls.pipeline.plan import Failure, Outcome, Plan
from hls.recipes.catalog import ImageSet
from tests.recipes.declared import SAMPLE_LAYER
from hls.spec.resources import GrantedResources
from hls.stages.agent_build import BUILD_FILE, BUILD_OUTPUT, BuildRecord, Ending, PidsCounts, SealProbe
from hls.stages.finalize import failed_records, finalize_params, index_key, index_row
from hls.store.digest import sha256_bytes
from hls.store.layout import InstanceLayout
from hls.store.manifest import Manifest, Provenance, Resources, Times, output_ref, publish_manifest
from hls.store.posix import PosixStore
from hls.substrate.base import FaultRecord
from hls.substrate.nproc import ProcessCeiling
from tests.evaluate import fixture
from tests.provision.warmer_config import TOKEN, write_config

#: The task's id, which the tests add to the image catalog.
TASK_ID = fixture.TASK_ID
#: The task's images: on sample_task's layer, declaring none of its own.
IMAGE_SET = ImageSet(layer=SAMPLE_LAYER)
#: The agent's wall-clock budget and the limit the stage enforces, in seconds.
SOFT_SEC, HARD_SEC = 1000, 1100
#: The build's hardware: one GPU, two CPUs, 16 GiB and 2 GiB of storage, the sandbox's disk cap.
HARDWARE = Hardware(accelerator=AcceleratorSpec(kind="B200", count=1), cpu_count=2, memory_gb=16, storage_gb=2)
#: The evaluation allocation ``allocation_artifact`` declares: what its graded server runs in, a
#: job of 180 GiB that the server shares with the harness.
EVALUATION_ALLOCATION = EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180, driver_memory_gib=16)
#: The direct route's credential, by its secret's name, and its value.
KEY_NAME, KEY = "anthropic_key", "sk-test-0123456789abcdef"
#: The relay route's token and base URL, by their secrets' names, and their values. The base URL
#: is a secret like the token: a plan may name its secret, never the URL or its host.
RELAY_TOKEN_NAME, RELAY_TOKEN = "kimi_token", "sk-kimi-0123456789abcdef"
RELAY_URL_NAME, RELAY_URL, RELAY_HOST = "kimi_url", "https://relay.example/anthropic", "relay.example"
#: The routes hls.toml declares: the provider's own, and a relay.
ROUTES = {"anthropic": Route(credential=KEY_NAME),
          "kimi": Route(credential=RELAY_TOKEN_NAME, base_url=RELAY_URL_NAME, auth="bearer")}


def secrets_text(*, relay_url: str = RELAY_URL, key: str = KEY) -> str:
    """A secrets file with the provision stage's hub token, the direct route's credential *key*
    and the relay route's token and base URL *relay_url*."""
    return (f'[secrets]\nhf_token = "{TOKEN}"\n{KEY_NAME} = "{key}"\n{RELAY_TOKEN_NAME} = "{RELAY_TOKEN}"\n'
            f'{RELAY_URL_NAME} = "{relay_url}"\n[grants]\n"stage.provision".HF_TOKEN = "hf_token"\n')


#: The secrets file the tests plan with.
SECRETS = secrets_text()
#: The arm the tests build with, on the direct route.
OPUS = {"harness": "claude-code", "model": "claude-opus-4-8", "context_window": 1_000_000, "output_limit": 64_000,
        "effort": "max", "route": "anthropic"}
#: An arm on the relay route.
K3 = {"harness": "claude-code", "model": "k3", "context_window": 200_000, "output_limit": 64_000, "effort": "max",
      "route": "kimi"}


def build_task():
    """The task, on ``HARDWARE``."""
    return fixture.evaluate_task(hardware=HARDWARE)


def artifact(**producer) -> AgentProducedArtifact:
    """An agent-produced artifact with the build's budgets, the highest effort, as every task
    authors it, and the producer's *producer* fields."""
    return AgentProducedArtifact(producer=AgentProducerSpec(**{
        "hard_budget": Budget(wallclock_sec=HARD_SEC), "soft_budget": Budget(wallclock_sec=SOFT_SEC),
        "reasoning_effort": HighestAvailable(), **producer}))


def effort_artifact() -> AgentProducedArtifact:
    """The artifact with a producer effort of "high", which an arm that sets none takes, and its
    server in another file."""
    return artifact(reasoning_effort=ProviderLevel(level="high"), server_rel_path="srv/main.py")


def allocation_artifact() -> AgentProducedArtifact:
    """The artifact with a build allocation of 3 CPUs and 72 GiB, the images stage's memory and
    more, which sizes the build in place of ``HARDWARE``, and an evaluation allocation, which
    sizes its graded server."""
    return artifact(build_allocation=BuildAllocation(cpus=3, memory_gib=72),
                    evaluation_allocation=EVALUATION_ALLOCATION)


def soft_turns_artifact() -> AgentProducedArtifact:
    """An artifact whose soft budget counts turns, which no tool reports to the agent."""
    return artifact(soft_budget=Budget(wallclock_sec=SOFT_SEC, turns=5))


def _toml(value) -> str:
    """*value* as a TOML value: a string, a number or an inline table."""
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml(v)}" for k, v in value.items()) + " }"
    return json.dumps(value)


def tables(section: str, entries: dict[str, dict]) -> str:
    """``[<section>.<name>]`` tables for *entries*."""
    return "".join(f"\n[{section}.{name}]\n" + "".join(f"{k} = {_toml(v)}\n" for k, v in entry.items())
                   for name, entry in entries.items())


def build_config(tmp_path: Path, *, arms: dict[str, dict] | None = None, site: str = "", scratch: Path | None = None,
                 secrets: str | None = SECRETS) -> Path:
    """The provision tests' warmer hls.toml, with *site* lines added
    to ``[site]``, the routes ``ROUTES``, the arm tables (``arm-a`` and ``k3`` by default), and the
    node-local scratch at *scratch* when given (a short path, so the sandbox's sockets stay under
    the kernel's limit). Writing it again replaces it."""
    path = write_config(tmp_path, secrets=secrets)
    text = path.read_text().replace("[run]", f"{site}\n[run]", 1)
    if scratch is not None:
        text = text.replace(f'node_scratch = "{tmp_path / "node"}"', f'node_scratch = "{scratch}"')
    routes = tables("routes", {name: route.model_dump(exclude_none=True) for name, route in ROUTES.items()})
    path.write_text(text + routes + tables("arms", arms if arms is not None else {"arm-a": OPUS, "k3": K3}))
    return path


def arm_record(name: str = "arm-a", table: dict | None = None, **changes) -> ArmRecord:
    """Arm *name*'s record as the planner resolves it from *table* (``OPUS`` by default), with
    *changes*; its route table is the one ``ROUTES`` names unless *changes* gives another."""
    fields = {**(table or OPUS), **changes}
    route_table = fields.pop("route_table", ROUTES[fields["route"]])
    return ArmRecord(name=name, route_table=route_table, **fields)


def publish_build(store: PosixStore, plan: Plan, arm: ArmRecord, *, created_at: float,
                  ending: Ending = "agent_ended", without: Any = None) -> str:
    """Publish an agent_build of rollout *plan*, its plan written as created at *created_at*, whose
    ``build.json`` records *arm*, as an earlier build that ended *ending* would have (a probe or
    preflight-only build with ``probe`` or ``preflight``), and lacks the fields *without* (a
    pydantic ``exclude``: a set of names, or a mapping for nested ones); return its build.json
    key. A plan with a finalize stage gets the index row its
    finalize writes, naming the arm; a stop build's has none, and gets none."""
    layout = InstanceLayout(plan.instance)
    store.create(layout.plan_revision(1), plan.model_copy(update={"created_at": created_at}).model_dump_json().encode())
    preflight = PreflightResult(ok=True, window=arm.context_window, expected_window=arm.context_window,
                                output_limit=arm.output_limit, model_key="m", model_seen="m", budget_tool="connected",
                                reason="", retryable=False)
    record = BuildRecord(arm=arm, arm_change=None, sandbox_image="e" * 64, upload_sha256=None, soft_sec=SOFT_SEC,
                         hard_sec=HARD_SEC, max_turns=None,
                         seal_probe=SealProbe(checks=[], passed=True, accounting="uid", sec=1.0),
                         preflight=None if ending == "probe" else preflight, started_at=created_at,
                         ended_at=created_at + 1, agent_sec=1.0, ending=ending, reentries=0, sessions_ended=1,
                         derailed=None,
                         commit=CommitRecord(status="ok", commit="c" * 40, reason=None, branches=["main"],
                                             at=created_at), commit_tries=[], memory=None, fault=None,
                         grade_after_sandbox_oom=True,
                         process_ceiling=ProcessCeiling(hard=None, reserve=1024, evaluations=1,
                                                        evaluations_unknown=None, uncapped=None, nproc_max=131_072,
                                                        ceiling=131_072, floor=4096),
                         pids_max_events=PidsCounts(at_start=0, at_end=0, rose_at=None), extraction=None)
    key = layout.attempt_file("agent_build", 1, f"outputs/{BUILD_FILE}")
    data = record.model_dump_json(exclude=without).encode()
    store.write_bytes(key, data)
    publish_manifest(store, layout, Manifest(
        instance=plan.instance, stage="agent_build", attempt=1,
        outputs={BUILD_OUTPUT: output_ref(store, key, sha256_bytes(data))},
        inputs={}, provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                                         image_digests={}, config_sha256="d" * 64),
        resources=Resources(declared=plan.stage("agent_build").shape,
                            granted=GrantedResources(gpus=1, gpu_kind="B200", cpus=3, memory_gib=22.0,
                                                     disk_free_gib=100.0)),
        times=Times(submitted_at=created_at, allocation_started_at=created_at, started_at=created_at,
                    ended_at=created_at + 1)))
    if finalize_params(plan) is not None:
        row, rollout_id, _ = index_key(plan.instance, "rollout")
        store.create(row, json.dumps(index_row(
            rollout_id=rollout_id, grading_id=None, mode="original", created_at=created_at,
            finished_at=created_at + 1, status="ok", score=None, gate_passed=None, cost_usd=None, image_digests={},
            failed_stage=None, fault=None, scanned=True, quarantined=[], spec={}, arm=arm)).encode())
    return key


def publish_running_build(store: PosixStore, plan: Plan, arm: ArmRecord, *, created_at: float) -> str:
    """Write rollout *plan* as a build that is running: its plan, created at *created_at*, naming
    *arm* in its agent_build stage, with no outcome, no published stage and no index row; return
    its instance."""
    layout = InstanceLayout(plan.instance)
    stages = [s.model_copy(update={"params": {**s.params, "arm": arm.model_dump(mode="json")}})
              if s.name == "agent_build" else s for s in plan.stages]
    running = plan.model_copy(update={"created_at": created_at, "stages": stages})
    store.create(layout.plan_revision(1), running.model_dump_json().encode())
    return plan.instance


def publish_ended_build(store: PosixStore, plan: Plan, arm: ArmRecord, *, created_at: float,
                        status: str = "failed") -> str:
    """Write rollout *plan* as an earlier build that ran *arm* and ended *status* with no published
    agent_build: failed, the contestant's zero, with the records the orchestrator writes for it
    (``failed_records``), its index row among them, or void, which is relaunched and gets none;
    return its instance."""
    layout = InstanceLayout(plan.instance)
    stages = [s.model_copy(update={"params": {**s.params, "arm": arm.model_dump(mode="json")}})
              if s.name == "agent_build" else s for s in plan.stages]
    fault = FaultRecord(fault_class="contestant" if status == "failed" else "infra", retryable=False,
                        reason="build sandbox sandbox wrote past its disk cap")
    # The allocation that ran the build failed with the fault that ended the instance.
    failure = Failure(at=created_at + 1, stage="agent_build", fault=fault)
    allocations = [a.model_copy(update={"failure": failure}) if "agent_build" in a.stages else a
                   for a in plan.allocations]
    ended = plan.model_copy(update={"created_at": created_at, "stages": stages, "allocations": allocations,
                                    "outcome": Outcome(status=status, reason=f"agent_build: {fault.reason}",
                                                       at=created_at + 1, fault=fault)})
    store.create(layout.plan_revision(1), ended.model_dump_json().encode())
    if status == "failed":
        failed_records(store, ended)
    return plan.instance
