from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from hls.pipeline.pipelines import (CALIBRATION_STAGE_NAMES, CREDIT_STAGE_NAMES, EXPORT_STAGE_NAMES, GRADING_STAGES,
                                    KNOWN_STAGE_NAMES, NOOP_TASK, ORACLE_FILL_STAGES, ROLLOUT_STAGES, SANDBOX_IMAGES_STAGE,
                                    images_stages, noop_one, noop_two, package_stages, provision_stages)
from hls.pipeline.commands import known_task_ids
from hls.pipeline.plan import Toolchain
from hls.pipeline.planner import (SYSTEM_PATH, PlanError, StageDecl, build_plan, check_setting_names, group,
                                  idle_gpu_warnings, toolchain_here)
from hls.pipeline.stage import END_MARGIN_SEC, most_roles
from hls.settings import RetryPolicy, RunSettings, load_settings
from hls.spec.resources import ResourceShape
from hls.stages.package import StoredBundle
from hls.store.layout import InstanceLayout
from hls.taskref import TaskRef
from hls.recipes.render import recipe_key as def_key
from tests.pipeline.plans import CODE, noop_plan, settings

CPU_STAGE = StageDecl("prep", "noop", ResourceShape(cpus=2, memory_gib=6, time_sec=60),
                      {"gpu_roles": 0, "cpu_role": True})


def test_max_policy_is_one_allocation_sized_to_the_largest_stage(tmp_path):
    plan = noop_plan(tmp_path)
    assert [a.stages for a in plan.allocations] == [["noop_a", "noop_b"]]
    req = plan.allocation(0).request
    assert (req.gpus, req.gpu_kind, req.cpus, req.memory_gib, req.time_sec) == (2, "H200", 8, 12, 600 + 600 + 360)
    assert plan.allocation(0).depends_on is None
    assert len(plan.warnings) == 1
    assert "stage noop_a uses 1, so 1 idle for up to 0.17 h (0.17 GPU-hours)" in plan.warnings[0]


def test_an_allocation_asks_for_its_stages_time_and_launch_headroom_for_each_role_they_may_start(tmp_path):
    plan = noop_plan(tmp_path, site={"step_launch_allowance_sec": 150})
    roles = [most_roles(s.kind, s.params) for s in plan.stages]
    assert roles == [1, 4]
    assert plan.allocation(0).request.time_sec == 600 + 600 + 360 + 150 * 5


def test_an_overhead_shorter_than_the_runners_end_margin_is_refused(tmp_path):
    with pytest.raises(PlanError, match="allocation_overhead_sec is 299s, less than the 300s"):
        noop_plan(tmp_path, allocation_overhead_sec=int(END_MARGIN_SEC) - 1)
    noop_plan(tmp_path, allocation_overhead_sec=int(END_MARGIN_SEC))


def test_each_allocation_asks_for_the_headroom_of_its_own_stages(tmp_path):
    plan = noop_plan(tmp_path, gpu_policy="split", site={"step_launch_allowance_sec": 20})
    assert [a.request.time_sec for a in plan.allocations] == [600 + 360 + 20 * 1, 600 + 360 + 20 * 4]


def test_split_policy_starts_an_allocation_at_each_gpu_change(tmp_path):
    plan = noop_plan(tmp_path, gpu_policy="split")
    assert [a.stages for a in plan.allocations] == [["noop_a"], ["noop_b"]]
    assert [a.request.gpus for a in plan.allocations] == [1, 2]
    assert [a.request.cpus for a in plan.allocations] == [4, 8]
    assert [a.request.memory_gib for a in plan.allocations] == [6, 12]
    assert [a.request.time_sec for a in plan.allocations] == [600 + 360, 600 + 360]
    assert [a.depends_on for a in plan.allocations] == [None, 0]
    assert plan.warnings == []


def test_split_policy_does_not_split_on_a_change_in_cpus_or_memory(tmp_path):
    a = ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=6, time_sec=100)
    b = ResourceShape(gpus=1, gpu_kind="H200", cpus=6, memory_gib=8, time_sec=200)
    decls = [StageDecl("noop_a", "noop", a, {"gpu_roles": 1}), StageDecl("noop_b", "noop", b, {"gpu_roles": 1})]
    plan = noop_plan(tmp_path, decls=decls, gpu_policy="split")
    assert [x.stages for x in plan.allocations] == [["noop_a", "noop_b"]]
    req = plan.allocation(0).request
    assert (req.gpus, req.cpus, req.memory_gib, req.time_sec) == (1, 6, 8, 100 + 200 + 360)
    assert plan.warnings == []


def test_resources_overrides_replace_the_declaration_and_are_recorded(tmp_path):
    shape = ResourceShape(gpus=2, gpu_kind="H200", cpus=10, memory_gib=12, time_sec=900)
    plan = noop_plan(tmp_path, resources={"noop": {"noop_b": shape.model_dump()}})
    assert plan.stage("noop_b").shape == shape and plan.stage("noop_b").shape_overridden
    assert not plan.stage("noop_a").shape_overridden


def test_settings_must_name_stages_that_exist(tmp_path):
    with pytest.raises(PlanError, match=r"run\.retry\.noop_c"):
        noop_plan(tmp_path, retry={"noop_c": {"max": 1}})
    shape = ResourceShape(cpus=1, memory_gib=1, time_sec=1).model_dump()
    with pytest.raises(PlanError, match=r"resources\.noop\.noop_z"):
        noop_plan(tmp_path, resources={"noop": {"noop_z": shape}})
    # Another task's tables name stages some pipeline declares.
    assert noop_plan(tmp_path, resources={"multitenant_slo": {"verify": shape}})


def test_a_table_for_another_pipelines_stage_of_the_task_warns_and_is_ignored(tmp_path):
    # One hls.toml serves every pipeline of a task: a table for a stage this plan does not run
    # (the one-stage pipeline's, here) is no typo, and changes nothing here.
    shape = ResourceShape(cpus=1, memory_gib=1, time_sec=1).model_dump()
    assert "noop" in KNOWN_STAGE_NAMES
    plan = noop_plan(tmp_path, resources={"noop": {"noop": shape}})
    assert plan.warnings[0] == ("[resources.noop.noop] names stage noop, which this plan does not run: the table is "
                                "ignored")
    assert [s.shape for s in plan.stages] == [s.shape for s in noop_plan(tmp_path).stages]
    assert not any(s.shape_overridden for s in plan.stages)


#: A site's QOS limits, in hours: a day under normal, a week under long.
SITE_QOS = {"normal": 24, "long": 168}


#: What the planner adds to a stage's time in these tests' plans: their allocation overhead. Their
#: sites give no launch allowance unless a test sets one (``tests.pipeline.plans.settings``).
ADDED_SEC = 360


def _long(hours: float) -> list[StageDecl]:
    """One CPU no-op stage whose allocation asks for *hours*, with the overhead (``ADDED_SEC``)."""
    shape = ResourceShape(cpus=2, memory_gib=6, time_sec=int(hours * 3600) - ADDED_SEC)
    return [replace(CPU_STAGE, shape=shape)]


def test_a_stage_no_configured_qos_holds_is_refused_naming_the_site_and_the_limit(tmp_path):
    with pytest.raises(PlanError) as e:
        noop_plan(tmp_path, decls=_long(200), site={"qos": "normal", "qos_by_hours": SITE_QOS})
    assert str(e.value) == ("stage prep takes 199.90 h; alone, with the runner's overhead and its launch headroom, it "
                            "asks for 200.00 h, more than any QOS of the site lets a job on partition gpu-a ask "
                            "(site.qos_by_hours): the longest, long, allows 168 h")


def test_a_site_with_no_qos_limits_plans_any_time_under_its_qos(tmp_path):
    plan = noop_plan(tmp_path, decls=_long(200), site={"qos": "normal"})
    assert plan.allocation(0).placement.qos == "normal"
    assert noop_plan(tmp_path, decls=_long(200)).allocation(0).placement.qos is None


def test_the_sites_qos_is_kept_while_it_holds_the_time_and_else_the_least_that_holds_it(tmp_path):
    site = {"qos": "normal", "qos_by_hours": SITE_QOS}
    assert noop_plan(tmp_path, decls=_long(24), site=site).allocation(0).placement.qos == "normal"
    assert noop_plan(tmp_path, decls=_long(24.01), site=site).allocation(0).placement.qos == "long"
    # With no QOS of its own, the site plans each allocation under the least limit that holds it.
    limits = {"long": 168, "short": 4, "normal": 24}
    assert [noop_plan(tmp_path, decls=_long(h), site={"qos_by_hours": limits}).allocation(0).placement.qos
            for h in (1, 4, 5, 168)] == ["short", "short", "normal", "long"]
    # A site whose own QOS is the long one keeps it while it holds the time.
    assert noop_plan(tmp_path, decls=_long(1), site={"qos": "long", "qos_by_hours": limits}).allocation(0).placement.qos == (
        "long")


def test_a_qos_holds_the_time_sbatch_asks_for_in_whole_minutes(tmp_path):
    """sbatch asks for the time rounded up to whole minutes: 30 minutes and a second is 31 minutes,
    which a 30.5-minute limit does not hold."""
    limits = {"short": 30.5 / 60, "long": 168}
    shape = ResourceShape(cpus=2, memory_gib=6, time_sec=30 * 60 + 1 - ADDED_SEC)
    plan = noop_plan(tmp_path, decls=[replace(CPU_STAGE, shape=shape)], site={"qos_by_hours": limits})
    assert plan.allocation(0).request.time_sec == 30 * 60 + 1 and plan.allocation(0).placement.qos == "long"
    exact = ResourceShape(cpus=2, memory_gib=6, time_sec=30 * 60 - ADDED_SEC)
    assert noop_plan(tmp_path, decls=[replace(CPU_STAGE, shape=exact)],
                     site={"qos_by_hours": limits}).allocation(0).placement.qos == "short"


def test_the_qos_holds_the_allocations_time_with_its_launch_headroom(tmp_path):
    """The QOS is picked on what sbatch asks for, the launch headroom in it: a stage whose time
    and overhead fit a day, with its roles' launch allowance past it, runs under long."""
    roles = most_roles("noop", CPU_STAGE.params)
    shape = ResourceShape(cpus=2, memory_gib=6, time_sec=24 * 3600 - 360)
    decls = [replace(CPU_STAGE, shape=shape)]
    site = {"qos": "normal", "qos_by_hours": SITE_QOS, "step_launch_allowance_sec": 150}
    plan = noop_plan(tmp_path, decls=decls, site=site)
    assert plan.allocation(0).request.time_sec == 24 * 3600 + 150 * roles and plan.allocation(0).placement.qos == "long"
    # With no allowance the same stage fits the day.
    plan = noop_plan(tmp_path, decls=decls, site={**site, "step_launch_allowance_sec": 0})
    assert plan.allocation(0).request.time_sec == 24 * 3600 and plan.allocation(0).placement.qos == "normal"


def test_a_site_qos_outside_its_limits_table_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"site\.qos 'debug' is not one of site\.qos_by_hours \['long', 'normal'\]"):
        noop_plan(tmp_path, site={"qos": "debug", "qos_by_hours": SITE_QOS})


def test_every_tasks_resources_tables_name_known_stages(tmp_path):
    shape = {"time_sec": 60}
    with pytest.raises(PlanError, match=r"resources\.multitenant_slo\.verfy"):
        noop_plan(tmp_path, resources={"multitenant_slo": {"verfy": shape}})
    with pytest.raises(PlanError, match=r"resources\.multitenant_slo\.drive_3"):
        noop_plan(tmp_path, resources={"multitenant_slo": {"drive_3": shape}})
    # A numbered name a pipeline makes is known by its pattern, for any task; for this one, whose
    # plan does not run it, the table is ignored with a warning.
    patterns = (r"drive_[1-9][0-9]*",)
    assert noop_plan(tmp_path, resources={"multitenant_slo": {"drive_3": shape}}, known_patterns=patterns).warnings == (
        noop_plan(tmp_path).warnings)
    own = noop_plan(tmp_path, resources={"noop": {"drive_12": shape}}, known_patterns=patterns)
    assert own.warnings[0] == ("[resources.noop.drive_12] names stage drive_12, which this plan does not run: the "
                               "table is ignored")
    with pytest.raises(PlanError, match=r"run\.retry\.drive_0"):
        noop_plan(tmp_path, retry={"drive_0": {"max": 1}}, known_patterns=patterns)
    assert noop_plan(tmp_path, retry={"drive_10": {"max": 1}}, known_patterns=patterns)


def test_a_resources_table_keyed_by_no_known_task_is_refused_naming_it_and_the_known_tasks(tmp_path):
    # A task's directory is not its id: tasks/multimodal_omni holds the task multimodal-omni, and a
    # table keyed by the directory would change nothing in any plan.
    table = {"provision": {"memory_gib": 48}}
    with pytest.raises(PlanError) as refused:
        noop_plan(tmp_path, resources={"multimodal_omni": table, "multimodal-omni": table, "deep": table})
    known = ", ".join(sorted(known_task_ids()))
    assert str(refused.value) == ("settings name unknown tasks: resources.deep, resources.multimodal_omni; a resources "
                                  f"table is keyed by a task's TaskSpec id, one of {known}")
    assert noop_plan(tmp_path, resources={"multimodal-omni": table})


def test_a_plans_own_task_and_the_known_tasks_key_resources_tables(tmp_path):
    tables = settings(tmp_path, resources={"own": {"noop_a": {"time_sec": 60}}, "other": {"verify": {"time_sec": 60}}})
    assert check_setting_names(tables, task="own", stage_names=["noop_a"], known_stage_names=KNOWN_STAGE_NAMES,
                               known_tasks=["other"]) == []
    with pytest.raises(PlanError, match=r"unknown tasks: resources\.own; .* one of other, someone$"):
        check_setting_names(tables, task="someone", stage_names=[], known_stage_names=KNOWN_STAGE_NAMES,
                            known_tasks=["other"])


def test_the_known_tasks_are_the_noop_task_and_each_graded_task_of_the_checkout():
    assert known_task_ids() == {NOOP_TASK, "multitenant_slo", "tab_completion", "sample_task", "multimodal-tts",
                                "multimodal-asr", "multimodal-omni", "deepresearch", "agentic_coding"}


def test_a_resources_table_sets_one_field_of_the_declared_shape(tmp_path):
    plan = noop_plan(tmp_path, resources={"noop": {"noop_b": {"time_sec": 900}}})
    declared = noop_two(gpu_kind="H200", sleep_sec=0, stage_time_sec=600)[1].shape
    assert plan.stage("noop_b").shape == declared.model_copy(update={"time_sec": 900})
    assert plan.stage("noop_b").shape_overridden and not plan.stage("noop_a").shape_overridden


def test_a_resources_table_that_leaves_no_valid_shape_is_refused_naming_it(tmp_path):
    # pydantic's message spans lines, so the pattern lets "." match a newline.
    with pytest.raises(PlanError,
                       match=r"(?s)\[resources\.noop\.prep\] leaves no valid shape: .*gpu_kind must be set"):
        noop_plan(tmp_path, decls=[CPU_STAGE], resources={"noop": {"prep": {"gpus": 1}}})


def test_a_resources_override_on_the_command_line_merges_into_the_shape(tmp_path):
    path = tmp_path / "hls.toml"
    path.write_text(f'[site]\npartition = "gpu"\nstore_root = "{tmp_path / "store"}"\n'
                    f'node_scratch = "{tmp_path / "node"}"\nsecrets_path = "/x"\n'
                    "[resources.noop.noop_b]\ncpus = 10\n")
    settings = load_settings(path, {"resources.noop.noop_b.time_sec": "900"})
    plan = build_plan(decls=noop_two(gpu_kind="H200", sleep_sec=0, stage_time_sec=600), task=NOOP_TASK,
                      pipeline="noop2", layout=InstanceLayout.rollout(NOOP_TASK, "r1"), settings=settings, code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-noop", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=1000.0)
    shape = plan.stage("noop_b").shape
    assert (shape.cpus, shape.time_sec, shape.gpus, shape.memory_gib) == (10, 900, 2, 12)


def test_a_stage_is_checked_against_its_overridden_shape(tmp_path):
    # Three GPUs do not split between noop_b's two GPU roles; the declared two GPUs do.
    shape = ResourceShape(gpus=3, gpu_kind="H200", cpus=10, memory_gib=12, time_sec=900)
    with pytest.raises(PlanError, match="do not split evenly"):
        noop_plan(tmp_path, resources={"noop": {"noop_b": shape.model_dump()}})


def test_known_stage_names_are_those_of_the_pipelines():
    decls = [*noop_one(gpus=0, gpu_kind="H200", sleep_sec=0, stage_time_sec=1),
             *noop_one(gpus=1, gpu_kind="H200", sleep_sec=0, stage_time_sec=1),
             *noop_two(gpu_kind="H200", sleep_sec=0, stage_time_sec=1),
             *provision_stages(task_ref=TaskRef(module="m"), shape=ResourceShape(cpus=1, memory_gib=1, time_sec=1),
                               weights=False),
             *images_stages(task="multitenant_slo", held=lambda keys: set(keys), recipe_key=def_key),
             *package_stages(task="t", catalog_key="sample_task",
                             bundle=StoredBundle(key="warmers/t/x/source.bundle", size=1), bundle_sha256="ab" * 32,
                             network={}, gpu_kind=None, held=lambda keys: set(), recipe_key=def_key, disk_cap_gib=180,
                             runner_memory_gib=RunSettings().runner_memory_gib, seal_memory_gib=0)]
    # A rollout's, a grading's, an oracle fill's, a credit's, a calibration's and an export's
    # stages, and a provision's sandbox_images stage, need a task or a command's arguments to
    # declare them; their own tests check that their plans have them.
    assert KNOWN_STAGE_NAMES == ({d.name for d in decls} | set(ROLLOUT_STAGES) | set(GRADING_STAGES)
                                 | set(ORACLE_FILL_STAGES) | set(CREDIT_STAGE_NAMES) | set(CALIBRATION_STAGE_NAMES)
                                 | set(EXPORT_STAGE_NAMES) | {SANDBOX_IMAGES_STAGE})


BROKEN_TASKS = """
import sys
from hls.recipes import catalog

def broken(task):
    # What reading a task module left broken by an edit raises.
    raise SyntaxError("a task module left broken by an edit")

# The one reader of a task module's images, which task_images goes through.
catalog.image_set = broken
from typer.testing import CliRunner
from hls import cli
cli.configure_logging = lambda verbose=False: None
for command in ("status", "cancel", "resume"):
    result = CliRunner().invoke(cli.app, [command, "warmers/x/y", "--store", sys.argv[1]])
    print(command, result.exit_code, result.output.strip())
try:
    catalog.task_images("sample_task")
except SyntaxError:
    print("task module broken")
"""


def test_status_cancel_and_resume_read_no_task_module(tmp_path):
    """Importing the stage-based commands reads no task module's images: an instance can be looked
    at, cancelled or resumed while a task module is broken, as it may be mid-edit on a cluster."""
    root = Path(__file__).resolve().parents[2]
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONPATH": f"{root / 'src'}:{root}"}
    out = subprocess.run([sys.executable, "-c", BROKEN_TASKS, str(tmp_path / "store")], capture_output=True,
                         text=True, env=env, cwd=root, timeout=120)
    assert out.stdout.splitlines() == [f"{command} 2 no such instance under {tmp_path / 'store'}"
                                       for command in ("status", "cancel", "resume")] + ["task module broken"], (
        out.stdout + out.stderr)


def test_retry_policies_are_resolved_per_stage(tmp_path):
    plan = noop_plan(tmp_path, retry={"noop_a": {"max": 0}})
    assert plan.stage("noop_a").retry == RetryPolicy(max=0)
    assert plan.stage("noop_b").retry == RetryPolicy(max=2)


@pytest.mark.parametrize("decls, match", [
    ([StageDecl("noop", "noop", ResourceShape(gpus=3, gpu_kind="H200", cpus=2, memory_gib=4, time_sec=60),
                {"gpu_roles": 2})], "do not split evenly"),
    ([StageDecl("noop", "noop", ResourceShape(cpus=2, memory_gib=4, time_sec=60), {"bogus": 1})], "bogus"),
    ([StageDecl("noop", "missing", ResourceShape(cpus=2, memory_gib=4, time_sec=60))],
     "stage kind .missing. is not registered by hls.stages.registry"),
    ([StageDecl("b", "noop", ResourceShape(cpus=2, memory_gib=4, time_sec=60), {"gpu_roles": 0, "cpu_role": True},
                after=("a",))], "do not come before it"),
    ([StageDecl("Bad", "noop", ResourceShape(cpus=2, memory_gib=4, time_sec=60))], "invalid stage name"),
    ([CPU_STAGE, CPU_STAGE], "declared twice"),
    ([StageDecl("noop", "noop", ResourceShape(gpus=1, gpu_kind="H200", cpus=1, memory_gib=4, time_sec=60),
                {"gpu_roles": 1})], "the roles need 2 CPUs"),
    ([StageDecl("noop", "noop", ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=2, time_sec=60),
                {"gpu_roles": 1, "shared_role": True})], "the roles need 3 GiB"),
    ([StageDecl("noop", "noop", ResourceShape(cpus=2, memory_gib=4, time_sec=60), {"gpu_roles": 1})],
     "1 GPU roles need at least 1 GPUs"),
    ([StageDecl("noop", "noop", ResourceShape(gpus=1, gpu_kind="H200", cpus=4, memory_gib=4, time_sec=60),
                {"gpu_roles": 2})], "2 GPU roles need at least 2 GPUs"),
])
def test_bad_stages_are_refused(tmp_path, decls, match):
    with pytest.raises(PlanError, match=match):
        noop_plan(tmp_path, decls=decls)


def test_two_gpu_kinds_in_one_run_are_refused(tmp_path):
    other = StageDecl("other", "noop", ResourceShape(gpus=1, gpu_kind="L40", cpus=2, memory_gib=6, time_sec=60))
    with pytest.raises(PlanError, match="different gpu kinds"):
        noop_plan(tmp_path, decls=[*noop_one(gpus=1, gpu_kind="H200", sleep_sec=0, stage_time_sec=60), other])


def test_the_site_gpu_floor_gives_a_cpu_allocation_the_runs_kind(tmp_path):
    decls = [CPU_STAGE, *noop_one(gpus=2, gpu_kind="H200", sleep_sec=0, stage_time_sec=60)]
    plan = noop_plan(tmp_path, decls=decls, site={"min_gpus_per_job": 1}, gpu_policy="split")
    assert [(a.request.gpus, a.request.gpu_kind) for a in plan.allocations] == [(1, "H200"), (2, "H200")]
    with pytest.raises(PlanError, match="no stage of the run names a gpu kind"):
        noop_plan(tmp_path, decls=[CPU_STAGE], site={"min_gpus_per_job": 1})


def test_split_groups_on_the_gpus_the_site_will_grant(tmp_path):
    decls = [CPU_STAGE, *noop_one(gpus=1, gpu_kind="H200", sleep_sec=0, stage_time_sec=60)]
    plan = noop_plan(tmp_path, decls=decls, site={"min_gpus_per_job": 1}, gpu_policy="split")
    assert [a.stages for a in plan.allocations] == [["prep", "noop"]]
    assert [(a.request.gpus, a.request.gpu_kind) for a in plan.allocations] == [(1, "H200")]
    [warning] = plan.warnings
    assert "stage prep uses 0, so 1 idle" in warning and "min_gpus_per_job" in warning
    assert "gpu_policy=split" not in warning


def test_group_and_the_idle_warnings_need_the_policy_and_the_floor_spelled_out(tmp_path):
    plan = noop_plan(tmp_path)
    with pytest.raises(TypeError):
        group(plan.stages, "split")
    with pytest.raises(TypeError):
        group(plan.stages, policy="split")
    with pytest.raises(TypeError):
        idle_gpu_warnings(plan.allocation(0), plan.stages)
    with pytest.raises(TypeError):
        idle_gpu_warnings(plan.allocation(0), plan.stages, policy="max")
    assert [[s.name for s in g] for g in group(plan.stages, policy="split", min_gpus=0)] == [["noop_a"], ["noop_b"]]


def test_the_idle_warning_under_max_suggests_split(tmp_path):
    [warning] = noop_plan(tmp_path).warnings
    assert "--run.gpu_policy=split" in warning and "min_gpus_per_job" not in warning


def test_a_kind_the_site_cannot_request_is_refused(tmp_path):
    with pytest.raises(PlanError, match="no gres name for gpu kind B200"):
        noop_plan(tmp_path, decls=noop_one(gpus=1, gpu_kind="B200", sleep_sec=0, stage_time_sec=60),
                  site={"gpu_gres": {"H200": "h200"}})


def test_toolchain_here():
    tc = toolchain_here()
    assert tc.python == Path(sys.executable)
    parts = tc.path.split(":")
    assert set(SYSTEM_PATH) <= set(parts) and len(parts) == len(set(parts))


def test_each_stage_keeps_the_runners_memory_and_the_request_is_the_declared_shapes(tmp_path):
    # The runner and its roles' srun clients run in the batch step, under the job's memory
    # limit, so a stage's roles leave the runner its reserve; the request adds nothing.
    decl = StageDecl("noop", "noop", ResourceShape(cpus=2, memory_gib=5, time_sec=60), {"gpu_roles": 0, "cpu_role": True})
    with pytest.raises(PlanError, match="the roles need 2 GiB and the runner 4 GiB; the stage declares 5"):
        noop_plan(tmp_path, decls=[decl])
    fits = replace(decl, shape=ResourceShape(cpus=2, memory_gib=6, time_sec=60))
    assert noop_plan(tmp_path, decls=[fits]).allocation(0).request.memory_gib == 6
    assert noop_plan(tmp_path, decls=[decl], runner_memory_gib=3).allocation(0).request.memory_gib == 5


def test_a_cpu_role_beside_the_one_stage_pipelines_role_needs_a_gpu_role():
    with pytest.raises(ValueError, match="needs a GPU role"):
        noop_one(gpus=0, gpu_kind="H200", sleep_sec=0, stage_time_sec=60, with_cpu_role=True)
