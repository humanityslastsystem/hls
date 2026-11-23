"""``hls run --build-only``'s planning: the five build stages and a finalize that records the
rollout, in one GPU allocation, the arm resolved into the plan, and a rerun whose arm changed
refused unless the change is declared; the allocation's QOS picked by its time against
the site's limits; how the command reports a run's end; and the probe and preflight-only builds
and per-run wall budgets."""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.arms import ArmError
from hls.pipeline import commands, prepare
from hls.pipeline.orchestrator import Orchestrator
from hls.pipeline.pipelines import (COMMIT_SHAPE, EVALUATE_PROVISION_SEC, FINALIZE_SHAPE, KNOWN_STAGE_NAMES,
                                    PROBE_HARD_SEC, SNAPSHOT_SHAPE, STAGING_SEC, artifact_shape, drive_time,
                                    images_shape, package_shape, sandbox_disk_gib, source_scan_shape)
from hls.pipeline.plan import Failure, Outcome, Plan, Toolchain
from hls.pipeline.planner import PlanError, launch_headroom_sec
from hls.pipeline.stage import most_roles
from hls.recipes import catalog
from hls.recipes.model import CONSTRAINTS
from hls.settings import RunSettings
from hls.spec.resources import ResourceShape, concurrent
from hls.stages.agent_build import (COLLECT_STEPS_SEC, COMMIT_MIN_SEC, STOP_BUILD_COLLECT_SEC, AgentBuildParams,
                                   AgentBuildStage)
from hls.stages.images import KEPT_DISK_GIB
from hls.stages import package as package_stage
from hls.stages import source_scan
from hls.stages.finalize import failed_key, index_key, index_row
from hls.stages.package import pip_limits
from hls.store.layout import InstanceLayout
from hls.store.posix import PosixStore
from hls.substrate.base import FaultRecord
from tests.agent import build_fixture
from tests.agent.build_fixture import (HARD_SEC, KEY, KEY_NAME, OPUS, RELAY_HOST, RELAY_TOKEN_NAME, RELAY_URL,
                                       RELAY_URL_NAME, SOFT_SEC, TASK_ID, arm_record, build_config, publish_build,
                                       publish_ended_build, publish_running_build, secrets_text)
from tests.pipeline.plans import CODE, pin_harbor
from tests.provision.cached import provision_module
from hls.provision.mounts import copied_entries, copies_gib
from hls.substrate.slurm.scheduler import submit_argv

#: A build-only run's stages, in the order its plan runs them.
STAGES = ["provision", "images", "agent_build", "source_scan", "package", "finalize"]
LAYER = catalog.image_set("sample_task").layer
CAP = catalog.AGENT_CAPS[LAYER].name
#: The time a full build keeps after its hard limit to collect it when the run sets none
#: (``run.collect_allowance_sec``).
COLLECT_SEC = RunSettings().collect_allowance_sec
#: The setup allowance a build is given beside its staging.
SETUP_SEC = AgentBuildParams.model_fields["setup_allowance_sec"].default
#: A site's QOS limits: a day under normal, a week under long.
LONG_QOS = 'qos = "normal"\nqos_by_hours = { normal = 24, long = 168 }'


def _plan(tmp_path: Path, *, module: str = "bd_task.task", arm: str = "arm-a", arm_change: str | None = None,
          config: Path | None = None, overrides: dict[str, str] | None = None, **flags) -> Plan:
    """``hls run --build-only``'s plan, or with *flags* naming ``stop_after`` a probe or
    preflight-only run's, the task's provisioning in the cache first."""
    provision_module(tmp_path / "store", module)
    build_only = flags.get("stop_after") is None
    return commands.plan_run(config=config or build_config(tmp_path), overrides=overrides or {},
                             task_module=module, arm=arm, arm_change=arm_change, job_name="hls-run",
                             build_only=build_only, echo=[].append, **flags)


def _params(plan) -> AgentBuildParams:
    return AgentBuildParams.model_validate(plan.stage("agent_build").params)


def _build_sec(*, hard_sec: int = HARD_SEC, collect_sec: int = COLLECT_SEC) -> int:
    """The agent_build stage's time: its setup allowance, its staging, the hard limit and the
    *collect_sec* it keeps after it."""
    return SETUP_SEC + STAGING_SEC + hard_sec + collect_sec


def _sbatch(tmp_path: Path, plan: Plan, k: int = 0) -> tuple[str, ...]:
    """The sbatch line of allocation *k*'s first submission, from the request the orchestrator makes."""
    here = SimpleNamespace(store=PosixStore(tmp_path / "store"), layout=plan.layout)
    return submit_argv(Orchestrator._request(here, plan, k, 1, script=Path("/b"), tag="t", after=None))


def test_a_build_only_run_plans_its_build_and_a_recording_finalize_in_one_gpu_allocation(tmp_path, build_login,
                                                                                         build_catalog):
    plan = _plan(tmp_path)
    assert [s.name for s in plan.stages] == STAGES and set(STAGES) <= KNOWN_STAGE_NAMES
    assert plan.pipeline == "rollout" and plan.task == TASK_ID
    assert plan.instance.startswith(f"runs/{TASK_ID}/") and plan.instance.count("/") == 2
    [alloc] = plan.allocations
    assert alloc.stages == STAGES and alloc.placement.partition is None
    # With no build allocation, the sandbox is the brief's hardware, its storage the disk cap on
    # what it writes; the stage holds it and one harness role on its CPUs, the runner's memory,
    # the agent cap's extracted tree and the copies of the agent's data, by what the cache's
    # provision record states of them. Its decimal GB are whole GiB, rounded down.
    hw = build_fixture.HARDWARE
    # Its setup comes after its staging, an hour for the copies of the agent's data and the agent
    # cap's extraction; after its hard limit it keeps the run's hour to collect the build.
    build_sec = _build_sec()
    assert _params(plan).staging_sec == STAGING_SEC == 3600
    assert _params(plan).setup_allowance_sec == SETUP_SEC == 3600
    assert (_params(plan).collect_allowance_sec, _params(plan).collection_sec) == (3600, 3600)
    assert build_sec == 3 * 3600 + HARD_SEC
    sandbox = ResourceShape(gpus=1, gpu_kind="B200", cpus=2, memory_gib=hw.memory_gib, disk_gib=hw.storage_gib,
                            time_sec=build_sec)
    snapshot = SNAPSHOT_SHAPE.model_copy(update={"time_sec": build_sec})
    assert (snapshot.gpus, snapshot.disk_gib) == (0, 0)
    roles = concurrent([sandbox, snapshot])
    data = copied_entries(("agent",), provision_module(tmp_path / "store", "bd_task.task"))
    assert _params(plan).data_copies == data and copies_gib(data) == 1
    assert plan.stage("agent_build").shape == roles.model_copy(update={
        "cpus": 2, "memory_gib": hw.memory_gib + SNAPSHOT_SHAPE.memory_gib + 4,
        "disk_gib": hw.storage_gib + KEPT_DISK_GIB + 1})
    images = plan.stage("images")
    assert [d["name"] for d in images.params["images"]] == ["s-a", LAYER, CAP]
    assert images.params["catalog_task"] == TASK_ID
    # Fifteen minutes, and half an hour for each of the three images it fills.
    assert images.shape == images_shape(3) and images.shape.time_sec == 900 + 3 * 1800
    package = plan.stage("package")
    # The source scan and the package hold the bundle to the sandbox's writable disk cap, its
    # storage in whole GiB, and their shapes hold a checkout of a bundle at it, for fixed times.
    cap = hw.storage_gib
    assert package.shape == package_shape(cap) and package.params["disk_cap_gib"] == cap
    assert package.shape.time_sec == package_stage.MIN_TIME_SEC == 25_200
    # The package stage, which holds two extracted trees, its work and three times what the
    # install may add, needs the most disk here; the sandbox writes little.
    assert package.shape.disk_gib == package_stage.min_disk_gib(cap) == package_stage.BUILD_STEPS_DISK_GIB
    finalize = plan.stage("finalize")
    assert finalize.shape == FINALIZE_SHAPE and finalize.shape.time_sec == 3600
    assert package.shape.disk_gib > max(images.shape.disk_gib, plan.stage("agent_build").shape.disk_gib,
                                        finalize.shape.disk_gib)
    # The source scan reads the bundle the build published, in its own stage, before the
    # package stage runs any of the contestant's code.
    scan = plan.stage("source_scan")
    assert scan.shape == source_scan_shape(cap, runner_memory_gib=4)
    assert scan.shape.time_sec == source_scan.MIN_TIME_SEC == 4104
    assert scan.after == ["agent_build"]
    assert scan.params == {"bundle": None, "bundle_sha256": None, "build_manifest": None, "disk_cap_gib": cap}
    # The job asks for the package stage's disk and the copies of the agent's data, which last the
    # job: the package stage runs while they are still on the node.
    assert (alloc.request.gpus, alloc.request.gpu_kind, alloc.request.disk_gib) == (
        1, "B200", package_stage.BUILD_STEPS_DISK_GIB + 1)
    # The stages' times, the overhead, and the headroom for each role's step to start: the build's
    # roles count a snapshot for every interval of the hard limit (``AgentBuildStage.max_roles``).
    roles = [most_roles(s.kind, s.params) for s in plan.stages]
    assert roles == [5, 12, AgentBuildStage.max_roles(_params(plan)), 2, 14, 1]
    assert alloc.request.time_sec == (EVALUATE_PROVISION_SEC + images.shape.time_sec + build_sec
                                      + scan.shape.time_sec + package.shape.time_sec + finalize.shape.time_sec
                                      + plan.settings.run.allocation_overhead_sec
                                      + sum(roles) * plan.settings.site.step_launch_allowance_sec)
    # The GPU idles while the five CPU stages run, and the plan says so.
    idle = [w for w in plan.warnings if w.startswith("allocation 0 holds 1 GPUs")]
    assert [w.split("; stage ")[1].split(" ")[0] for w in idle] == ["provision", "images", "source_scan", "package",
                                                                    "finalize"]

    p = _params(plan)
    assert p.arm == arm_record() and p.arm_change is None
    assert (p.soft_sec, p.hard_sec, p.max_turns) == (SOFT_SEC, HARD_SEC, None)
    assert (p.sandbox_shape, p.snapshot_shape) == (sandbox, snapshot)
    # The brief states what a graded server gets: the shape an evaluation of the task runs it in,
    # here the task's hardware, as its producer declares no evaluation allocation.
    assert p.served_shape == artifact_shape(build_fixture.build_task(),
                                            runner_memory_gib=plan.settings.run.runner_memory_gib)
    assert (p.sandbox_image, p.images_stage, p.provision_stage) == (CAP, "images", "provision")
    assert p.network == build_fixture.artifact().producer.network.model_dump(mode="json")
    assert p.network["packages"]["mode"] == "deny"  # the producer's own policy, deny list and all
    assert p.task.module == "bd_task.task"
    assert package.after == ["images", "agent_build", "source_scan"]
    assert plan.stage("agent_build").after == ["provision", "images"]
    # The pip steps' limits come from the package stage's own shape, and the build's brief states
    # the same figures the stage holds its steps to.
    pip = pip_limits(package.shape, plan.settings.run.runner_memory_gib)
    assert package.params == {"task": TASK_ID, "submission_base": LAYER, "constraints": CONSTRAINTS,
                              "images_stage": "images",
                              "network": p.network, "gpu_kind": "B200", "server_filename": "server.py",
                              "bundle": None, "bundle_sha256": None, "build_manifest": None,
                              "disk_cap_gib": cap, "pip": pip.model_dump()}
    assert p.package_pip == pip
    # A build-only run grades nothing: its finalize records the rollout and no score.
    assert plan.stage("finalize").params["score_stage"] is None


def test_a_build_keeps_the_runs_collection_allowance_after_its_hard_limit(tmp_path, build_login, build_catalog):
    """``run.collect_allowance_sec`` is the time a full build's stage keeps after the agent's hard
    limit: the plan records it in the build's parameters, which the brief and the stage read, and
    the stage's time, and the times of the roles that run for the whole stage, grow by it."""
    plan = _plan(tmp_path, overrides={"run.collect_allowance_sec": "5000"})
    assert plan.settings.run.collect_allowance_sec == 5000
    p = _params(plan)
    assert (p.collect_allowance_sec, p.collection_sec) == (5000, 5000)
    build_sec = _build_sec(collect_sec=5000)
    assert plan.stage("agent_build").shape.time_sec == build_sec == 2 * 3600 + HARD_SEC + 5000
    assert (p.sandbox_shape.time_sec, p.snapshot_shape.time_sec, p.commit_shape.time_sec) == (build_sec,) * 3
    # Only the build's time changes: the stages after it keep theirs.
    default = _plan(tmp_path)
    assert [(s.name, s.shape.time_sec) for s in plan.stages if s.name != "agent_build"] == [
        (s.name, s.shape.time_sec) for s in default.stages if s.name != "agent_build"]
    assert plan.allocations[0].request.time_sec - default.allocations[0].request.time_sec == 5000 - 3600


@pytest.mark.parametrize("stop_after", [None, "probe"], ids=["full", "probe"])
def test_a_collection_allowance_under_what_a_collection_needs_is_refused_naming_its_setting(
        tmp_path, build_login, build_catalog, stop_after):
    """A collection needs its fixed steps and at least ten minutes for the commit point; an
    allowance under that is refused when the run is planned, naming the setting, a probe's too,
    whose brief states the allowance a full build gets. The least it needs is accepted."""
    least = COLLECT_STEPS_SEC + COMMIT_MIN_SEC
    assert (COLLECT_STEPS_SEC, COMMIT_MIN_SEC, least) == (1750, 600, 2350)
    with pytest.raises(PlanError) as e:
        _plan(tmp_path, overrides={"run.collect_allowance_sec": str(least - 1)}, stop_after=stop_after)
    assert str(e.value) == (
        "stage agent_build: the collection allowance (run.collect_allowance_sec) is 2349s, under the 2350s a build's "
        "collection needs: 1750s for its fixed steps (the sandbox's and the snapshot's stops, the sandbox's log "
        "scrub, the wait for its processes to be gone, the transcript copy and the extraction) and at least 600s for "
        "the commit point; set it to 2350 or more")
    plan = _plan(tmp_path, overrides={"run.collect_allowance_sec": str(least)}, stop_after=stop_after)
    assert _params(plan).collect_allowance_sec == least


@pytest.mark.parametrize("stop_after", ["probe", "preflight"])
@pytest.mark.parametrize("allowance", [None, "2350", "5000"], ids=["unset", "least", "longer"])
def test_a_stop_build_keeps_its_own_short_ending_whatever_the_collection_allowance(
        tmp_path, build_login, build_catalog, stop_after, allowance):
    """A probe or preflight-only build runs no commit point: after its hard limit it keeps six
    minutes to end and stop the sandbox, whatever ``run.collect_allowance_sec`` is, which its
    parameters still record for the brief."""
    overrides = {} if allowance is None else {"run.collect_allowance_sec": allowance}
    plan = _plan(tmp_path, overrides=overrides, stop_after=stop_after)
    p = _params(plan)
    assert p.collect_allowance_sec == (COLLECT_SEC if allowance is None else int(allowance))
    assert p.collection_sec == STOP_BUILD_COLLECT_SEC == 360
    assert plan.stage("agent_build").shape.time_sec == _build_sec(hard_sec=PROBE_HARD_SEC, collect_sec=360)


def test_a_build_gives_its_commit_point_and_snapshots_the_measured_worst_peak_with_a_quarter_more(
        tmp_path, build_login, build_catalog):
    """The commit point's worst peak measured: 8.335 GiB of anonymous memory, git rebuilding
    reachable objects stored as deltas on three unreachable 4 GiB bases, the largest object it is
    given, two bases held at once since git keeps the last in its cache while it reads the next;
    one such base, 5.003 GiB; more objects (5,000,000 commits, 1.907 GiB), the largest trees
    (0.673 GiB) and the program's own reads at their worst (0.940 GiB) take less. The commit point
    gets its peak and a quarter more, rounded up: one CPU and 11 GiB, for the stage's time, once the sandbox has stopped. A
    snapshot runs beside the sandbox, so its memory comes out of the sandbox's: it gets the peak of
    one base read at a time and a quarter more, 7 GiB, and a snapshot past it is only an event."""
    plan = _plan(tmp_path)
    assert COMMIT_SHAPE.memory_gib == math.ceil(8.335 * 1.25) == 11
    assert SNAPSHOT_SHAPE.memory_gib == math.ceil(5.003 * 1.25) == 7
    assert (COMMIT_SHAPE.cpus, COMMIT_SHAPE.memory_gib, COMMIT_SHAPE.gpus, COMMIT_SHAPE.disk_gib) == (1, 11, 0, 0)
    assert (SNAPSHOT_SHAPE.cpus, SNAPSHOT_SHAPE.gpus, SNAPSHOT_SHAPE.disk_gib) == (1, 0, 0)
    assert _params(plan).commit_shape == COMMIT_SHAPE.model_copy(update={"time_sec": _build_sec()})
    assert plan.stage("agent_build").shape.memory_gib == (build_fixture.HARDWARE.memory_gib
                                                         + SNAPSHOT_SHAPE.memory_gib + 4)


def test_a_build_allocation_sizes_the_sandbox_and_every_stage_fits_its_cpus(tmp_path, build_login, build_catalog):
    """A producer that declares a build allocation sizes the build to it: the
    sandbox has every CPU of it, the memory the runner and a harness role leave, and as much
    writable space as the allocation has memory; no stage takes more CPUs than it has, so the
    job asks for the allocation's CPUs and memory."""
    plan = _plan(tmp_path, module="bd_alloc.task")
    p = _params(plan)
    assert p.sandbox_shape == ResourceShape(gpus=1, gpu_kind="B200", cpus=3, memory_gib=72 - 4 - 7, disk_gib=72,
                                            time_sec=_build_sec())
    # Its graded server's shape is its evaluation allocation's, the one an evaluation runs it in.
    task = build_fixture.build_task()
    served = artifact_shape(task, build_fixture.EVALUATION_ALLOCATION,
                            runner_memory_gib=plan.settings.run.runner_memory_gib)
    assert p.served_shape == served == ResourceShape(gpus=1, gpu_kind="B200", cpus=8, memory_gib=180 - 4,
                                                     disk_gib=180, time_sec=drive_time(task))
    assert plan.stage("agent_build").shape == p.sandbox_shape.model_copy(
        update={"memory_gib": 72, "disk_gib": 72 + KEPT_DISK_GIB + p.data_gib})
    assert [s.shape.cpus for s in plan.stages] == [3, 3, 3, 2, 3, 2]
    assert plan.stage("images").shape == images_shape(3).model_copy(update={"cpus": 3})
    assert plan.stage("package").shape == package_shape(72).model_copy(update={"cpus": 3})
    assert plan.stage("source_scan").shape == source_scan_shape(72, runner_memory_gib=4)
    [alloc] = plan.allocations
    assert (alloc.request.gpus, alloc.request.cpus, alloc.request.memory_gib) == (1, 3, 72)
    with pytest.raises(ValueError, match="leaves the sandbox no memory beside the runner's 70 GiB"):
        _plan(tmp_path, module="bd_alloc.task", overrides={"run.runner_memory_gib": "70"})


#: Each task's declared build allocation, CPUs and GiB.
PUBLISHED_ALLOCATIONS = {"multitenant_slo": (8, 180), "multimodal_asr": (8, 180), "multimodal_omni": (8, 180),
                         "multimodal_tts": (8, 240), "deepresearch": (8, 180), "tab_completion": (8, 180)}


@pytest.mark.parametrize("task", sorted(PUBLISHED_ALLOCATIONS))
def test_each_task_with_a_build_allocation_declares_it(task):
    allocation = importlib.import_module(f"tasks.{task}.task").ARTIFACT.producer.build_allocation
    assert (allocation.cpus, allocation.memory_gib) == PUBLISHED_ALLOCATIONS[task]


@pytest.fixture
def repo_login(monkeypatch, tmp_path):
    """The login node planning from this repository, whose ``tasks/`` holds the real tasks. The
    folder a task uploads into the sandbox is not committed, so no checkout of the tests holds
    it, and the blob check finds nothing to load; each check has tests of its own."""
    monkeypatch.setattr(commands, "code_state", lambda path: CODE)
    pin_harbor(monkeypatch, CODE)
    monkeypatch.setattr(commands, "check_upload_folder", lambda checkout, task_module, producer: None)
    monkeypatch.setattr(commands, "check_defaults_committed", lambda checkout, files: None)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))


def test_multitenant_slo_plans_its_declared_build_allocation(tmp_path, repo_login):
    """One H200, 8 CPUs and 180 GiB, the build allocation multitenant_slo declares: the sandbox
    all 8 CPUs, 169 GiB (the allocation less the runner's 4 and the concurrent snapshot's 7) and
    180 GiB of writable space, the cap its repository is held to; the disk the package stage's,
    which checks out a bundle at that cap beside the submission base's tree, and the copies of the
    agent's data, which last the job; the time the five build stages' (provision 2 h, images'
    15 min and half an hour for each of the three built images, the build's hour of setup, hour of
    staging, 12.5 h and hour of collection, the source scan's and the package's fixed times),
    finalize's hour, the runner's overhead and the headroom for its roles' steps to start."""
    plan = _plan(tmp_path, module="tasks.multitenant_slo.task")
    sandbox = _params(plan).sandbox_shape
    assert (sandbox.gpus, sandbox.cpus, sandbox.memory_gib, sandbox.disk_gib) == (1, 8, 169, 180)
    # One owner gives the cap to the sandbox, its brief and commit point, the source scan and the
    # package, as it does to ``hls package`` of the task's bundles.
    module = importlib.import_module("tasks.multitenant_slo.task")
    cap = sandbox_disk_gib(module.TASK, module.ARTIFACT.producer)
    assert sandbox.disk_gib == cap == 180
    assert plan.stage("source_scan").params["disk_cap_gib"] == plan.stage("package").params["disk_cap_gib"] == cap
    build_sec = _build_sec(hard_sec=int(12.5 * 3600))
    assert plan.stage("agent_build").shape.time_sec == build_sec == 55_800
    [alloc] = plan.allocations
    r = alloc.request
    data = _params(plan).data_gib
    assert _params(plan).data_copies == copied_entries(
        ("agent",), provision_module(tmp_path / "store", "tasks.multitenant_slo.task"))
    assert plan.stage("agent_build").shape.disk_gib == 180 + KEPT_DISK_GIB + data
    assert plan.stage("package").shape.disk_gib == KEPT_DISK_GIB + 180 + 4 + 4 == 228
    # The disk is the larger of the build's (its writable space and the agent cap's tree) and the
    # package's, here its checkout at the cap, with the copies of the agent's data, which last the job.
    assert (r.gpus, r.gpu_kind, r.cpus, r.memory_gib, r.disk_gib) == (1, "H200", 8, 180, 228 + data)
    scan_sec, package_sec = 4104, 25_200
    assert (plan.stage("source_scan").shape.time_sec, plan.stage("package").shape.time_sec) == (scan_sec, package_sec)
    images = plan.stage("images").shape
    assert images == images_shape(3, cpus=8) and images.time_sec == 6300
    staged = 2 * 3600 + images.time_sec + scan_sec + package_sec + build_sec
    assert sum(s.shape.time_sec for s in plan.stages if s.name != "finalize") == staged == 98_604
    finalize = plan.stage("finalize").shape.time_sec
    assert finalize == 3600
    assert r.time_sec == (staged + finalize + plan.settings.run.allocation_overhead_sec
                          + launch_headroom_sec(plan.stages, plan.settings.site.step_launch_allowance_sec))
    # A site that sets no QOS limits plans it under its QOS, if any.
    assert alloc.placement.qos is None


def test_a_build_on_a_site_holding_scored_servers_to_one_socket_asks_for_one_task(tmp_path, build_login,
                                                                                  build_catalog):
    """Only a scored server is held to one socket (``site.evaluation_socket_tasks``): a build serves
    none, so its allocation asks for its CPUs as one task wherever Slurm finds them."""
    plan = _plan(tmp_path, overrides={"site.evaluation_socket_tasks": "true"})
    assert plan.settings.site.evaluation_socket_tasks
    [alloc] = plan.allocations
    assert alloc.placement.socket_tasks is None
    args = _sbatch(tmp_path, plan)
    assert "--ntasks=1" in args and f"--cpus-per-task={alloc.request.cpus}" in args
    assert "--ntasks-per-socket=1" not in args


def test_a_build_past_the_sites_one_day_qos_is_planned_under_its_long_qos(tmp_path, repo_login):
    """A site's normal QOS caps a job at a day; a multitenant_slo build asks for its stages' 28.4 h,
    the overhead and its launch headroom, so it is planned under long, and sbatch is asked for it."""
    plan = _plan(tmp_path, module="tasks.multitenant_slo.task", config=build_config(tmp_path, site=LONG_QOS))
    [alloc] = plan.allocations
    assert alloc.request.time_sec > 24 * 3600 and alloc.placement.qos == "long"
    args = _sbatch(tmp_path, plan)
    assert "--qos=long" in args and "--qos=normal" not in args


def test_the_build_stage_leaves_the_runner_the_memory_the_site_sets(tmp_path, build_login, build_catalog):
    plan = _plan(tmp_path, overrides={"run.runner_memory_gib": "6"})
    assert plan.stage("agent_build").shape.memory_gib == (build_fixture.HARDWARE.memory_gib
                                                         + SNAPSHOT_SHAPE.memory_gib + 6)


def test_the_agent_role_is_granted_its_arms_credential_by_name_only(tmp_path, build_login, build_catalog):
    plan = _plan(tmp_path)
    assert plan.stage("agent_build").grants == {"ANTHROPIC_API_KEY": KEY_NAME}
    assert KEY not in plan.model_dump_json()
    # A relay route's grants hold both its secrets' names: its token and its base URL.
    relay = _plan(tmp_path, arm="k3")
    assert relay.stage("agent_build").grants == {"ANTHROPIC_AUTH_TOKEN": RELAY_TOKEN_NAME,
                                                 "ANTHROPIC_BASE_URL": RELAY_URL_NAME}


def test_the_plan_names_the_route_and_holds_no_base_url(tmp_path, build_login, build_catalog):
    plan = _plan(tmp_path, arm="k3")
    text = plan.model_dump_json()
    assert _params(plan).arm.route == "kimi" and '"route":"kimi"' in text.replace(" ", "")
    assert RELAY_URL not in text and RELAY_HOST not in text


def test_a_base_url_secret_that_is_not_an_https_url_is_refused_when_planned(tmp_path, build_login, build_catalog):
    plain = "http://relay.example/anthropic"
    config = build_config(tmp_path, secrets=secrets_text(relay_url=plain))
    with pytest.raises(PlanError) as e:
        _plan(tmp_path, arm="k3", config=config)
    assert str(e.value) == ("stage agent_build: the base URL secret kimi_url, granted as ANTHROPIC_BASE_URL, must "
                            "hold an https URL with a host")
    assert plain not in str(e.value) and e.value.__cause__ is None


def test_an_arm_without_an_effort_takes_the_producers(tmp_path, build_login, build_catalog):
    arms = {"arm-a": {k: v for k, v in OPUS.items() if k != "effort"}}
    config = build_config(tmp_path, arms=arms)
    assert _params(_plan(tmp_path, module="bd_effort.task", config=config)).arm.effort == "high"
    # The highest available is Claude Code's top level.
    assert _params(_plan(tmp_path, config=config)).arm.effort == "max"


def test_the_package_serves_the_producers_server_file(tmp_path, build_login, build_catalog):
    plan = _plan(tmp_path, module="bd_effort.task")
    assert plan.stage("package").params["server_filename"] == "srv/main.py"


def test_the_build_stages_take_retry_settings(tmp_path, build_login, build_catalog):
    plan = _plan(tmp_path, overrides={"run.retry.agent_build.max": "0", "run.retry.package.max": "3"})
    assert (plan.stage("agent_build").retry.max, plan.stage("package").retry.max) == (0, 3)
    # Unset, a build is retried once.
    assert _plan(tmp_path).stage("agent_build").retry.max == 1


def test_an_arm_hls_toml_does_not_declare_is_an_arm_error_naming_it(tmp_path, build_login, build_catalog):
    with pytest.raises(ArmError, match="no arm named 'arm-b' in hls.toml; .arms. declares .'arm-a', 'k3'."):
        _plan(tmp_path, arm="arm-b")


def test_a_producer_whose_budgets_the_runner_cannot_keep_is_refused(tmp_path, build_login, build_catalog):
    with pytest.raises(ValueError, match="a soft turn budget has no tool that reports it"):
        _plan(tmp_path, module="bd_turns.task")


def test_only_an_agent_produced_submission_is_built(tmp_path, build_login, build_catalog):
    with pytest.raises(ValueError, match="ARTIFACT is None, not an agent-produced artifact"):
        _plan(tmp_path, module="bd_none.task")


def test_only_site_and_run_overrides_are_accepted(tmp_path, build_login, build_catalog):
    with pytest.raises(ValueError, match="arms.arm-a.model"):
        _plan(tmp_path, overrides={"arms.arm-a.model": "x"})


# ---------------------------------------------------------------------------
# How a build ends
# ---------------------------------------------------------------------------


def test_a_failed_build_only_run_reports_the_zero_its_records_hold_and_exits_0(tmp_path, build_login,
                                                                             build_catalog):
    """A build-only run plans a finalize, so the orchestrator writes a failed run's records, its
    zero among them, and the command prints that zero from where it lies. A probe or
    preflight-only build plans none and is never the contestant's result: it ends succeeded,
    printing its probe once it published one, or void."""
    plan = _plan(tmp_path)
    fault = FaultRecord(fault_class="contestant", reason="contestant role sandbox wrote past its disk cap",
                        retryable=False)
    failed = plan.model_copy(update={
        "allocations": [plan.allocations[0].model_copy(update={"failure": Failure(at=1.0, stage="agent_build",
                                                                                  fault=fault)})],
        "outcome": Outcome(status="failed", reason="stage agent_build: a contestant fault", at=1.0, fault=fault)})
    store = PosixStore(tmp_path / "store")
    store.create(failed.layout.plan_revision(1), failed.model_dump_json().encode())
    assert commands.failed_records(store, failed)
    lines: list[str] = []
    assert commands._finish(failed, lines.append) == 0
    key = failed_key(plan.layout, "score.json")
    assert lines == [f"{plan.instance}: failed: stage agent_build: a contestant fault",
                     f"{plan.instance}: score 0.0, zero for contestant_fault ({key})"]
    # The zero is the orchestrator's own record, which no manifest pins; one of another shape is
    # refused, saying where.
    store.write_bytes(key, b'{"zero": true}')
    with pytest.raises(ValueError, match=f"{key} is not a Score: missing value"):
        commands._finish(failed, [].append)
    stop = _plan(tmp_path, stop_after="probe")
    assert commands.failed_records(store, stop) == {}
    for status, code in (("succeeded", 0), ("void", 1), ("cancelled", 1)):
        lines = []
        ended = stop.model_copy(update={"outcome": Outcome(status=status, reason="why", at=1.0)})
        assert commands._finish(ended, lines.append) == code
        assert lines == [f"{stop.instance}: {status}: why"]


def _published_probe(tmp_path: Path, plan: Plan, probe: bytes, published: bytes) -> str:
    """Publish *plan*'s agent_build with *published* as its probe.json's recorded bytes, then put
    *probe* in its place; the probe's key."""
    from hls.spec.resources import GrantedResources
    from hls.store.digest import sha256_bytes
    from hls.store.manifest import Manifest, Provenance, Resources, Times, output_ref, publish_manifest

    store = PosixStore(tmp_path / "store")
    key = plan.layout.attempt_file("agent_build", 1, "outputs/probe.json")
    store.write_bytes(key, published)
    publish_manifest(store, plan.layout, Manifest(
        instance=plan.instance, stage="agent_build", attempt=1,
        outputs={"probe": output_ref(store, key, sha256_bytes(published))}, inputs={},
        provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                              image_digests={}, config_sha256="d" * 64),
        resources=Resources(declared=plan.stage("agent_build").shape,
                            granted=GrantedResources(gpus=1, gpu_kind="B200", cpus=8, memory_gib=180.0,
                                                     disk_free_gib=100.0)),
        times=Times(submitted_at=1.0, allocation_started_at=1.0, started_at=1.0, ended_at=2.0)))
    store.write_bytes(key, probe)
    return key


def _probe(all_ok: bool, failed: list[str]) -> bytes:
    """A probe.json whose checks named *failed* did not pass."""
    checks = [{"name": "python-version", "ok": True, "observed": "3.12.1", "expected": ">= 3.12"}]
    checks += [{"name": n, "ok": False, "observed": "x", "expected": "y"} for n in failed]
    return json.dumps({"checks": checks, "all_ok": all_ok, "oom": None, "sandbox_ending": None}).encode()


@pytest.mark.parametrize("all_ok, failed, note", [
    (True, [], ""), (False, ["unlisted-refused", "nspid-depth"], "; failed: unlisted-refused, nspid-depth")],
    ids=["all-ok", "some-failed"])
def test_a_stop_build_prints_the_probes_result(tmp_path, build_login, build_catalog, all_ok, failed, note):
    """A probe or preflight-only build records no score, so ``hls run`` prints the probe's
    result: its ``all_ok`` and the names of the checks that did not pass."""
    plan = _plan(tmp_path, stop_after="probe")
    probe = _probe(all_ok, failed)
    _published_probe(tmp_path, plan, probe, probe)
    ended = plan.model_copy(update={"outcome": Outcome(status="succeeded", reason="every stage published", at=1.0)})
    lines: list[str] = []
    assert commands._finish(ended, lines.append) == 0
    assert lines == [f"{plan.instance}: succeeded: every stage published",
                     f"{plan.instance}: probe all_ok={str(all_ok).lower()}{note}"]


@pytest.mark.parametrize("probe, published, match", [
    (_probe(True, []), _probe(False, ["nspid-depth"]), "does not match its manifest"),
    (b'{"checks": [{"name": "uid-gid"}], "all_ok": true}', None, "is not a ProbeRecord: missing checks.0.ok"),
], ids=["rewritten", "another-shape"])
def test_a_stop_build_says_only_the_probe_its_manifest_recorded(tmp_path, build_login, build_catalog, probe,
                                                                published, match):
    plan = _plan(tmp_path, stop_after="probe")
    key = _published_probe(tmp_path, plan, probe, published or probe)
    ended = plan.model_copy(update={"outcome": Outcome(status="succeeded", reason="every stage published", at=1.0)})
    lines: list[str] = []
    with pytest.raises(ValueError, match=f"{key} {match}"):
        commands._finish(ended, lines.append)
    assert lines == [f"{plan.instance}: succeeded: every stage published"]


# ---------------------------------------------------------------------------
# A rerun runs the arm it ran
# ---------------------------------------------------------------------------


def _earlier(tmp_path: Path, *builds: tuple, names: list[str] | None = None) -> list[str]:
    """Earlier published builds of the task, each with its arm and its plan's creation time, and
    each rollout named by *names* when given; returns their instances. Every one is planned
    before any is published, so the rerun check never compares them with each other."""
    plans = [_plan(tmp_path) for _ in builds]
    if names is not None:
        plans = [plan.model_copy(update={"instance": f"runs/{TASK_ID}/{name}"})
                 for plan, name in zip(plans, names, strict=True)]
    for plan, (arm, created_at) in zip(plans, builds, strict=True):
        publish_build(PosixStore(tmp_path / "store"), plan, arm, created_at=created_at)
    return [plan.instance for plan in plans]


def test_a_rerun_of_the_same_arm_is_planned(tmp_path, build_login, build_catalog):
    _earlier(tmp_path, (arm_record(), 100.0))
    assert _params(_plan(tmp_path)).arm_change is None


def test_a_rerun_whose_arm_differs_is_refused_naming_each_difference(tmp_path, build_login, build_catalog):
    [earlier] = _earlier(tmp_path, (arm_record(output_limit=32_000, route_table={"credential": "other_key"}), 100.0))
    with pytest.raises(ArmError) as e:
        _plan(tmp_path)
    message = str(e.value)
    assert f"arm arm-a differs from its latest build of {TASK_ID}, {earlier}" in message
    assert "output_limit: 32000 -> 64000" in message
    assert "route_table.credential: 'other_key' -> 'anthropic_key'" in message
    assert "--arm-change REASON" in message


def test_a_secret_changed_under_the_same_name_is_not_an_arm_change(tmp_path, build_login, build_catalog):
    """The record holds the route's secrets by name: a new credential under the same name is the
    same arm, and the rerun needs no declaration."""
    _earlier(tmp_path, (arm_record(), 100.0))
    config = build_config(tmp_path, secrets=secrets_text(key="sk-test-a-new-key-0123456789"))
    assert _params(_plan(tmp_path, config=config)).arm_change is None


def test_a_declared_change_is_planned_and_recorded(tmp_path, build_login, build_catalog):
    _earlier(tmp_path, (arm_record(output_limit=32_000), 100.0))
    assert _params(_plan(tmp_path, arm_change="the relay raised its output cap")).arm_change == (
        "the relay raised its output cap")


@pytest.mark.parametrize("earlier, change, match", [
    (True, "nothing changed", "--arm-change was given, but the arm is the one"),
    (False, "first build", "--arm-change was given, but evaluate_task has no earlier build of this arm"),
    (True, "  ", "--arm-change needs the reason the arm differs"),
], ids=["same-arm", "no-earlier-build", "blank-reason"])
def test_a_declared_change_must_name_a_real_difference(tmp_path, build_login, build_catalog, earlier, change, match):
    if earlier:
        _earlier(tmp_path, (arm_record(), 100.0))
    with pytest.raises(ArmError, match=match):
        _plan(tmp_path, arm_change=change)


def test_the_latest_earlier_build_of_the_arm_is_the_one_compared(tmp_path, build_login, build_catalog):
    # The store lists r-1 first; the latest build of the arm is r-2, and r-3 is another arm's.
    _earlier(tmp_path, (arm_record(output_limit=32_000), 100.0), (arm_record(), 200.0),
             (arm_record("other", output_limit=1_000), 300.0), names=["r-1", "r-2", "r-3"])
    assert _params(_plan(tmp_path)).arm_change is None
    [newest] = _earlier(tmp_path, (arm_record(effort="high"), 400.0), names=["r-4"])
    with pytest.raises(ArmError, match=f"{newest}: effort: 'high' -> 'max'"):
        _plan(tmp_path)


def test_a_rerun_after_a_failed_build_runs_the_arm_it_ran(tmp_path, build_login, build_catalog):
    """A build that ended failed, the contestant's zero, publishes no build record; its plan names
    the arm it ran, and a rerun of the same arm name is compared with it. A void build is
    not the contestant's result, and is relaunched as planned."""
    store = PosixStore(tmp_path / "store")
    void = publish_ended_build(store, _plan(tmp_path), arm_record(output_limit=1_000), created_at=300.0,
                               status="void")
    failed = publish_ended_build(store, _plan(tmp_path), arm_record(output_limit=32_000), created_at=200.0)
    assert void != failed
    with pytest.raises(ArmError, match=f"{failed}: output_limit: 32000 -> 64000"):
        _plan(tmp_path)
    assert _params(_plan(tmp_path, arm_change="the relay raised its cap")).arm_change == "the relay raised its cap"


def test_an_earlier_build_record_that_is_not_its_manifests_is_refused(tmp_path, build_login, build_catalog):
    plan = _plan(tmp_path)
    store = PosixStore(tmp_path / "store")
    key = publish_build(store, plan, arm_record(), created_at=100.0)
    store.write_bytes(key, store.read_bytes(key).replace(b'"max"', b'"low"'))
    with pytest.raises(ValueError, match="build.json does not match its manifest"):
        _plan(tmp_path)


# ---------------------------------------------------------------------------
# Real tasks, the CLI and the login node
# ---------------------------------------------------------------------------


def test_a_task_whose_id_is_not_its_directory_is_built_from_its_directorys_images(tmp_path, repo_login):
    # multimodal_asr's TaskSpec id is multimodal-asr; the catalog knows it by its directory.
    plan = _plan(tmp_path, module="tasks.multimodal_asr.task")
    assert plan.task == "multimodal-asr" and plan.instance.startswith("runs/multimodal-asr/")
    layer = catalog.image_set("multimodal_asr").layer
    assert _params(plan).sandbox_image == catalog.AGENT_CAPS[layer].name
    assert plan.stage("images").params["catalog_task"] == "multimodal_asr"
    assert plan.stage("package").params["submission_base"] == layer
    assert plan.stage("package").params["task"] == "multimodal-asr"


def _text(output: str) -> str:
    """What the CLI printed, without its colours and its error box, on one line."""
    return " ".join(re.sub(r"[\u2502\u256d\u256e\u2570\u256f\u2500]", " ", re.sub(r"\x1b\[[0-9;]*m", "", output)).split())


def test_the_cli_plans_a_build_only_run_with_its_arm_and_starts_it(monkeypatch):
    seen, started = {}, []

    def plan(**kw):
        seen.update(kw)
        return "the plan"

    monkeypatch.setattr(commands, "plan_run", plan)
    monkeypatch.setattr(commands, "start", lambda made, overrides, **kw: started.append((made, overrides)) or 0)
    monkeypatch.setattr(prepare, "prepare", lambda **kw: 0)
    result = CliRunner().invoke(cli.app, ["run", "tasks.multitenant_slo.task", "--arm", "arm-a", "--arm-change",
                                          "relay cap", "--build-only", "--site.qos", "normal"])
    assert result.exit_code == 0, result.output
    assert (seen["task_module"], seen["arm"], seen["arm_change"], seen["job_name"], seen["build_only"]) == (
        "tasks.multitenant_slo.task", "arm-a", "relay cap", "hls-run", True)
    assert seen["overrides"] == {"site.qos": "normal"}
    assert started == [("the plan", {"site.qos": "normal"})]


def test_the_cli_reports_an_arm_error_as_a_usage_error(monkeypatch):
    def plan(**kw):
        raise ArmError("no arm named 'arm-b' in hls.toml; [arms] declares ['arm-a']")

    monkeypatch.setattr(commands, "plan_run", plan)
    monkeypatch.setattr(prepare, "prepare", lambda **kw: 0)
    result = CliRunner().invoke(cli.app, ["run", "tasks.multitenant_slo.task", "--arm", "arm-b"])
    assert result.exit_code == 2
    assert "Invalid value for --arm: no arm named 'arm-b' in hls.toml" in _text(result.output)


def test_the_cli_refuses_what_cannot_be_planned_in_one_line(monkeypatch):
    def plan(**kw):
        raise ValueError("a soft turn budget has no tool that reports it")

    monkeypatch.setattr(commands, "plan_run", plan)
    monkeypatch.setattr(prepare, "prepare", lambda **kw: 0)
    result = CliRunner().invoke(cli.app, ["run", "tasks.multitenant_slo.task", "--arm", "arm-a"])
    assert result.exit_code == 1
    assert result.output.strip().splitlines() == ["hls run: a soft turn budget has no tool that reports it"]


def test_a_run_on_a_site_without_its_secrets_file_is_refused_in_one_line_before_submitting(
        tmp_path, build_login, build_catalog, monkeypatch):
    """The source scan, the package stage and finalize scan the submission for every value of the
    site's secrets file: a site with none is refused when the run is planned, and nothing is
    submitted."""
    monkeypatch.setattr(commands, "start", lambda *args, **kw: pytest.fail("the plan was submitted"))
    config = build_config(tmp_path, secrets=None)
    provision_module(tmp_path / "store", "bd_task.task")
    result = CliRunner().invoke(cli.app, ["run", "bd_task.task", "--arm", "arm-a", "--build-only", "--config",
                                          str(config)])
    assert result.exit_code == 1
    assert result.output.strip().splitlines() == [
        f"hls run: stages source_scan, package, finalize need every value of the site's secrets file, and there is "
        f"none at {tmp_path / 'secrets.toml'}"]


def test_the_cli_needs_an_arm():
    result = CliRunner().invoke(cli.app, ["run", "tasks.multitenant_slo.task"])
    assert result.exit_code == 2 and "Missing option '--arm'" in _text(result.output)


def test_the_cli_refuses_an_override_before_the_task_module(monkeypatch):
    monkeypatch.setattr(commands, "plan_run", lambda **kw: pytest.fail("planned"))
    result = CliRunner().invoke(cli.app, ["run", "--site.qos=normal", "--arm", "arm-a"])
    assert result.exit_code == 1 and "expected the task module first" in result.output


PROBE = """
import json, sys
from pathlib import Path
from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
import pytest
from tests.pipeline.plans import CODE, pin_harbor
commands.code_state = lambda path: CODE
pin_harbor(pytest.MonkeyPatch(), CODE)
commands.toolchain_here = lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=Path("/"))
# The folder the task uploads is not committed, so no checkout of the tests holds it.
commands.check_upload_folder = lambda checkout, task_module, producer: None
plan = commands.plan_run(config=Path(sys.argv[1]), overrides={}, task_module="tasks.multitenant_slo.task",
                         arm="arm-a", arm_change=None, job_name="x", build_only=True, echo=lambda line: None)
heavy = sorted(m for m in ("torch", "transformers", "vllm", "datasets", "huggingface_hub", "pyarrow")
               if m in sys.modules)
print(json.dumps({"heavy": heavy, "stages": [s.name for s in plan.stages]}))
"""


def _fresh(code: str, *args: str) -> str:
    """The last line *code* prints in a fresh interpreter run from this checkout, with the
    import path these tests run with after it (harbor's source, when it is given there)."""
    root = commands.checkout_root()
    inherited = [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root), *inherited])}
    out = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, cwd=root, env=env,
                         check=True, timeout=120).stdout
    return out.splitlines()[-1]


def test_planning_a_build_on_the_login_node_loads_nothing_heavy(tmp_path):
    provision_module(tmp_path / "store", "tasks.multitenant_slo.task")
    assert json.loads(_fresh(PROBE, str(build_config(tmp_path)))) == {"heavy": [], "stages": STAGES}


def test_the_stages_registry_registers_agent_build():
    code = ("import json, hls.stages.registry\nfrom hls.pipeline.stage import STAGES\n"
            "print(json.dumps(sorted(STAGES)))")
    assert "agent_build" in json.loads(_fresh(code))



# ---------------------------------------------------------------------------
# Probe and preflight-only builds, and per-build wall budgets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stop_after", ["probe", "preflight"])
def test_a_stop_build_has_no_scan_or_package_stage_and_a_short_budget(tmp_path, build_login, build_catalog,
                                                                      stop_after):
    plan = _plan(tmp_path, stop_after=stop_after)
    assert [s.name for s in plan.stages] == ["provision", "images", "agent_build"]
    p = _params(plan)
    assert (p.stop_after, p.probe_oom, p.soft_sec, p.hard_sec, p.max_turns) == (stop_after, False, None,
                                                                                 PROBE_HARD_SEC, None)
    assert plan.stage("agent_build").shape.time_sec == _build_sec(hard_sec=PROBE_HARD_SEC,
                                                                  collect_sec=STOP_BUILD_COLLECT_SEC)
    assert plan.allocations[0].stages == ["provision", "images", "agent_build"]


def test_a_stop_build_runs_on_a_full_builds_allocation(tmp_path, build_login, build_catalog):
    """The same sandbox, harness role and allocation as the task's builds, for its own time."""
    full = _plan(tmp_path, module="bd_alloc.task")
    probe = _plan(tmp_path, module="bd_alloc.task", stop_after="probe", probe_oom=True)
    assert _params(probe).probe_oom is True

    def timeless(shape):
        return shape.model_copy(update={"time_sec": 1})

    for of in (lambda plan: _params(plan).sandbox_shape, lambda plan: _params(plan).snapshot_shape,
               lambda plan: plan.stage("agent_build").shape):
        assert timeless(of(probe)) == timeless(of(full))
    full_request, probe_request = full.allocations[0].request, probe.allocations[0].request
    # Only the disk may differ: the full build's package stage may ask for the most.
    assert (probe_request.gpus, probe_request.gpu_kind, probe_request.cpus, probe_request.memory_gib) == (
        full_request.gpus, full_request.gpu_kind, full_request.cpus, full_request.memory_gib)
    assert probe_request.disk_gib == max(s.shape.disk_gib for s in probe.stages)


def test_a_multitenant_slo_probe_holds_the_declared_allocation_for_its_own_time(tmp_path, repo_login):
    plan = _plan(tmp_path, module="tasks.multitenant_slo.task", stop_after="probe")
    [alloc] = plan.allocations
    r = alloc.request
    assert (r.gpus, r.gpu_kind, r.cpus, r.memory_gib, r.disk_gib) == (1, "H200", 8, 180,
                                                                     180 + KEPT_DISK_GIB + _params(plan).data_gib)
    assert r.time_sec == (2 * 3600 + plan.stage("images").shape.time_sec
                          + _build_sec(hard_sec=PROBE_HARD_SEC, collect_sec=STOP_BUILD_COLLECT_SEC)
                          + plan.settings.run.allocation_overhead_sec
                          + launch_headroom_sec(plan.stages, plan.settings.site.step_launch_allowance_sec))


def test_per_build_wall_budgets_replace_the_tasks(tmp_path, build_login, build_catalog):
    plan = _plan(tmp_path, soft_sec=600, hard_sec=900)
    p = _params(plan)
    assert (p.soft_sec, p.hard_sec, p.max_turns, p.stop_after) == (600, 900, None, None)
    assert plan.stage("agent_build").shape.time_sec == _build_sec(hard_sec=900)
    assert [s.name for s in plan.stages] == STAGES


@pytest.mark.parametrize("flags, match", [
    ({"soft_sec": 600}, "together"), ({"hard_sec": 900}, "together"),
    ({"soft_sec": 900, "hard_sec": 600}, "over --hard-sec"),
    ({"stop_after": "probe", "soft_sec": 600, "hard_sec": 900}, "refused with --probe-only"),
    ({"stop_after": "preflight", "probe_oom": True}, "probe_oom runs only"),
], ids=["soft-alone", "hard-alone", "soft-over-hard", "budgets-with-a-stop", "oom-with-preflight"])
def test_flags_a_build_cannot_run_are_refused(tmp_path, build_login, build_catalog, flags, match):
    with pytest.raises(ValueError, match=match):
        _plan(tmp_path, **flags)


def test_a_probe_or_preflight_build_is_no_earlier_build_of_its_arm(tmp_path, build_login, build_catalog):
    """A stop build runs no agent and builds nothing, so a rerun is compared with the arm's latest
    full build, never with a probe of it: a probe of an arm with another output limit needs no
    --arm-change, and is not a build to declare a change from."""
    store = PosixStore(tmp_path / "store")
    publish_build(store, _plan(tmp_path, stop_after="probe"), arm_record(output_limit=32_000), created_at=300.0,
                  ending="probe")
    publish_build(store, _plan(tmp_path, stop_after="preflight"), arm_record(output_limit=16_000),
                  created_at=400.0, ending="preflight")
    assert _params(_plan(tmp_path)).arm_change is None
    with pytest.raises(ArmError, match="has no earlier build of this arm"):
        _plan(tmp_path, arm_change="the relay raised its cap")
    [earlier] = _earlier(tmp_path, (arm_record(effort="high"), 100.0))
    with pytest.raises(ArmError, match=f"{earlier}: effort: 'high' -> 'max'"):
        _plan(tmp_path)


@pytest.mark.parametrize("stop_after", ["probe", "preflight"])
def test_a_stop_builds_record_is_not_read_so_one_older_code_wrote_blocks_no_run(tmp_path, build_login,
                                                                                build_catalog, stop_after):
    """Whether a build stopped is its plan's to say: a probe or preflight-only build's record is
    never read, so one that does not parse does not block a run."""
    store = PosixStore(tmp_path / "store")
    publish_build(store, _plan(tmp_path, stop_after=stop_after), arm_record(output_limit=32_000),
                  created_at=300.0, ending=stop_after, without={"upload_sha256"})
    assert _params(_plan(tmp_path)).arm_change is None


def test_another_arms_records_are_never_read_so_none_that_does_not_parse_blocks_a_run(tmp_path, build_login,
                                                                                       build_catalog):
    """The task's index rows name the arm each build ran, so a rerun reads the records of its own
    arm's builds alone: another arm's build record and plan, which may not parse, block nothing."""
    store = PosixStore(tmp_path / "store")
    other = _plan(tmp_path)
    publish_build(store, other, arm_record("other", output_limit=1_000), created_at=300.0,
                  without={"upload_sha256", "arm"})
    store.create(InstanceLayout(other.instance).plan_revision(2), b'{"settings": {"no_such_setting": 1}}')
    assert _params(_plan(tmp_path)).arm_change is None


@pytest.mark.parametrize("row", [b"{not json", b"[]", b'{"rollout_id": "r-9"}', b'{"arm": "arm-a"}'],
                         ids=["not-json", "not-an-object", "no-arm", "no-creation-time"])
def test_an_index_row_that_may_be_the_arms_and_cannot_be_read_refuses_the_run(tmp_path, build_login,
                                                                               build_catalog, row):
    """A row that does not say which arm it is, or when the arm's build was made, may be the
    latest of this arm's builds, so the rerun is refused, naming the row."""
    store = PosixStore(tmp_path / "store")
    key = f"runs/{TASK_ID}/index/r-9.json"
    store.create(key, row)
    with pytest.raises(ValueError, match=rf"^{re.escape(key)} "):
        _plan(tmp_path)


def test_a_builds_record_is_read_for_its_arm_alone(tmp_path, build_login, build_catalog):
    """The rerun rule compares arms, so of an earlier build's record only the arm is read: one
    without a field a record holds today is compared by its arm, and blocks a rerun only when that
    arm differs."""
    store = PosixStore(tmp_path / "store")
    older = {"upload_sha256": True, "process_ceiling": {"nproc_max", "uncapped"}}
    publish_build(store, _plan(tmp_path), arm_record(), created_at=300.0, without=older)
    assert _params(_plan(tmp_path)).arm_change is None
    [changed] = _earlier(tmp_path, (arm_record(output_limit=32_000), 400.0), names=["r-changed"])
    with pytest.raises(ArmError, match=f"{changed}: output_limit: 32000 -> 64000"):
        _plan(tmp_path)


@pytest.mark.parametrize("without, errors", [
    ({"arm": True}, "missing arm (required by this version of hls)"),
    ({"arm": {"model"}}, "missing arm.model (required by this version of hls)"),
], ids=["no-arm", "an-arm-with-no-model"])
def test_a_full_builds_record_whose_arm_does_not_parse_refuses_the_run(tmp_path, build_login, build_catalog,
                                                                         without, errors):
    """The arm is what the rerun is compared with, so a record whose arm does not parse as the arm
    a build records today refuses the run, naming each error by its type and place and none of
    the record's values: an arm's schema changing is a change of the arm."""
    store = PosixStore(tmp_path / "store")
    key = publish_build(store, _plan(tmp_path), arm_record(), created_at=300.0, without=without)
    with pytest.raises(ValueError) as e:
        _plan(tmp_path)
    assert str(e.value) == f"{key} names no arm record: {errors}"


def test_a_builds_record_that_is_not_the_file_its_manifest_recorded_refuses_the_run(tmp_path, build_login,
                                                                                      build_catalog):
    store = PosixStore(tmp_path / "store")
    key = publish_build(store, _plan(tmp_path), arm_record(), created_at=300.0)
    store.path(key).write_bytes(store.read_bytes(key) + b" ")
    with pytest.raises(ValueError, match=rf"^{re.escape(key)} does not match its manifest"):
        _plan(tmp_path)


def test_an_earlier_builds_manifest_that_does_not_parse_refuses_naming_its_errors(tmp_path, build_login,
                                                                                    build_catalog):
    store = PosixStore(tmp_path / "store")
    [earlier] = _earlier(tmp_path, (arm_record(), 300.0))
    manifest = InstanceLayout(earlier).manifest("agent_build")
    store.path(manifest).write_bytes(b'{"instance": 7}')
    with pytest.raises(ValueError) as e:
        _plan(tmp_path)
    assert str(e.value).startswith(f"{manifest} is not a manifest: string_type instance; missing stage")
    assert "validation error" not in str(e.value)


def test_a_gradings_row_names_its_rollouts_build_and_is_no_build_of_its_own(tmp_path, build_login,
                                                                             build_catalog):
    """A grading's index row names the arm that built the package it graded, its rollout's: the
    rerun compares the rollout's build, and the grading's row, newer, names no rollout of its own."""
    store = PosixStore(tmp_path / "store")
    [rollout] = _earlier(tmp_path, (arm_record(), 100.0))
    _, rollout_id, _ = index_key(rollout, "rollout")
    row = index_row(rollout_id=rollout_id, grading_id="g-1", mode="adopt", created_at=200.0, finished_at=201.0,
                    status="ok", score=None, gate_passed=None, cost_usd=None, image_digests={}, failed_stage=None,
                    fault=None, scanned=True, quarantined=[], spec={}, arm=arm_record())
    store.create(index_key(f"{rollout}/gradings/g-1", "grading")[0], json.dumps(row).encode())
    assert _params(_plan(tmp_path)).arm_change is None


def test_a_running_build_of_the_arm_is_compared_with(tmp_path, build_login, build_catalog):
    """A build still running has no index row yet, and is one of the arm's builds all the same: a
    rerun with another configuration under its name is refused unless it declares the change, so
    two builds at once under one arm name cannot differ unrecorded."""
    store = PosixStore(tmp_path / "store")
    running = publish_running_build(store, _plan(tmp_path), arm_record(output_limit=32_000), created_at=300.0)
    with pytest.raises(ArmError, match=f"{running}: output_limit: 32000 -> 64000"):
        _plan(tmp_path)
    assert _params(_plan(tmp_path, arm_change="the relay raised its cap")).arm_change == "the relay raised its cap"


def test_a_running_build_of_another_arm_or_a_probe_blocks_nothing(tmp_path, build_login, build_catalog):
    """Of a running build only its plan's JSON is read, for its outcome, its stop and its arm's
    name: another arm's, whose plan this code would not parse, a probe, running or ended without
    a row, and a void build of the arm, ended without a row, are no build of this arm."""
    store = PosixStore(tmp_path / "store")
    other = publish_running_build(store, _plan(tmp_path), arm_record("other", output_limit=1_000), created_at=300.0)
    plan_key = InstanceLayout(other).plan_revision(1)
    raw = json.loads(store.read_bytes(plan_key))
    raw["settings"]["site"]["no_such_setting"] = 1
    store.path(plan_key).write_text(json.dumps(raw))
    publish_running_build(store, _plan(tmp_path, stop_after="probe"), arm_record(output_limit=32_000),
                          created_at=400.0)
    ended = _plan(tmp_path, stop_after="probe")
    publish_running_build(store, ended, arm_record(output_limit=16_000), created_at=500.0)
    raw = json.loads(store.read_bytes(InstanceLayout(ended.instance).plan_revision(1)))
    raw["outcome"] = {"status": "succeeded", "reason": "every stage published", "at": 501.0}
    store.path(InstanceLayout(ended.instance).plan_revision(1)).write_text(json.dumps(raw))
    publish_ended_build(store, _plan(tmp_path), arm_record(output_limit=8_000), created_at=600.0, status="void")
    assert _params(_plan(tmp_path)).arm_change is None


@pytest.mark.parametrize("plan_text", [b"{not json", b'{"created_at": 1.0, "stages": [{"kind": "agent_build", '
                                                     b'"params": {}}]}'],
                         ids=["not-json", "no-arm"])
def test_a_running_plan_that_does_not_say_its_arm_refuses_the_run(tmp_path, build_login, build_catalog,
                                                                  plan_text):
    """A rollout with no index row whose plan does not say it ended, or whose arm it builds, may be
    a running build of this arm, so the rerun is refused, naming the plan."""
    store = PosixStore(tmp_path / "store")
    key = InstanceLayout.rollout(TASK_ID, "r-running").plan_revision(1)
    store.create(key, plan_text)
    with pytest.raises(ValueError, match=rf"^{re.escape(key)} "):
        _plan(tmp_path)


def test_a_failed_stop_build_is_no_earlier_build_of_its_arm(tmp_path, build_login, build_catalog):
    store = PosixStore(tmp_path / "store")
    publish_ended_build(store, _plan(tmp_path, stop_after="probe"), arm_record(output_limit=32_000),
                        created_at=200.0)
    assert _params(_plan(tmp_path)).arm_change is None


def test_the_cli_passes_the_stop_and_budget_flags(monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(commands, "plan_run", lambda **kw: seen.append(kw) or "the plan")
    monkeypatch.setattr(commands, "start", lambda made, overrides, **kw: 0)
    monkeypatch.setattr(prepare, "prepare", lambda **kw: 0)
    for argv in (["--probe-only", "--probe-oom"], ["--preflight-only"], ["--soft-sec", "600", "--hard-sec", "900"]):
        result = CliRunner().invoke(cli.app, ["run", "tasks.multitenant_slo.task", "--arm", "arm-a", *argv])
        assert result.exit_code == 0, result.output
    assert [(kw["stop_after"], kw["probe_oom"], kw["soft_sec"], kw["hard_sec"]) for kw in seen] == [
        ("probe", True, None, None), ("preflight", False, None, None), (None, False, 600, 900)]


@pytest.mark.parametrize("argv, words", [
    (["--probe-only", "--preflight-only"], "give at most one of --probe-only and --preflight-only"),
    (["--probe-oom"], "--probe-oom runs only with --probe-only"),
    (["--preflight-only", "--probe-oom"], "--probe-oom runs only with --probe-only"),
])
def test_the_cli_refuses_stop_flags_that_cannot_go_together(monkeypatch, argv, words):
    monkeypatch.setattr(commands, "plan_run", lambda **kw: pytest.fail("planned"))
    result = CliRunner().invoke(cli.app, ["run", "tasks.multitenant_slo.task", "--arm", "arm-a", *argv])
    assert result.exit_code == 2 and words in _text(result.output)
