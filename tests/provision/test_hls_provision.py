from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hls.pipeline import commands
from hls.pipeline.pipelines import PROVISION_ROLE_GIB
from hls.pipeline.plan import PlanStore
from hls.pipeline.planner import PlanError
from hls.provision.model import Provisioned, RowsRef, ToolRef, WeightsRef, rows_key, tool_key, weights_key
from hls.recipes import catalog
from hls.spec.resources import GrantedResources
from hls.stages.provision import role_shape
from hls.taskref import TaskRef, load_task
from hls.store.digest import sha256_bytes
from hls.store.layout import InstanceLayout
from hls.store.manifest import Manifest, Provenance, Resources, Times, output_ref, publish_manifest
from hls.store.lock import Holder, instance_lease
from hls.store.posix import PosixStore
from hls.substrate.secrets import SecretsError
from tests.pipeline.fake_scheduler import FakeScheduler
from tests.provision.warmer_config import SECRETS, TOKEN, task_sandbox_stores, write_config


def _plan(config: Path, *, module: str = "tests.provision.tasks", weights: bool = True,
          overrides: dict[str, str] | None = None, lines: list[str] | None = None):
    return commands.plan_provision(config=config, overrides=overrides or {}, task_module=module, weights=weights,
                                   job_name="hls-provision", echo=(lines if lines is not None else []).append)


def _checkout(tmp_path: Path, monkeypatch) -> Path:
    """A checkout of the test's own, which the command takes for the one it runs from."""
    root = (tmp_path / "checkout").resolve()
    root.mkdir()
    monkeypatch.setattr(commands, "checkout_root", lambda: root)
    return root


@pytest.fixture
def forget():
    """Names of modules the test imports from its own files, dropped from sys.modules after it."""
    names: list[str] = []
    yield names
    for name in names:
        sys.modules.pop(name, None)


def test_a_warmer_instance_lives_apart_from_every_rollout():
    assert InstanceLayout.warmer("sample_task", "w1").root == "warmers/sample_task/w1"
    with pytest.raises(ValueError, match="invalid store key segment"):
        InstanceLayout.warmer("a/b", "w1")
    with pytest.raises(ValueError, match="invalid store key segment"):
        InstanceLayout.warmer("sample_task", "w/1")


def test_hls_provision_plans_one_cpu_allocation_with_no_gpu(tmp_path, login):
    # The site's GPU floor belongs to its GPU partition; a cache warmer on the CPU partition
    # asks for no GPU at all.
    plan = _plan(write_config(tmp_path, floor=1))
    assert plan.instance.startswith("warmers/T-provision/")
    # A warmer runs one at a time with the site's other warmers, under their one job name.
    assert (plan.task, plan.pipeline, plan.job_name) == ("T-provision", "provision", "hls-warmer")
    [alloc] = plan.allocations
    assert alloc.stages == ["provision"] and alloc.placement.partition == "cpu-a"
    assert (alloc.request.gpus, alloc.request.gpu_kind) == (0, None)
    assert plan.stage("provision").params == {"task": {"module": "tests.provision.tasks", "overrides": {}},
                                              "weights": True}
    assert _plan(write_config(tmp_path), weights=False).stage("provision").params["weights"] is False


def test_without_a_cpu_partition_the_warmer_runs_on_the_gpu_partition_and_says_so(tmp_path, login):
    plan = _plan(write_config(tmp_path, cpu_partition=None))
    assert plan.allocations[0].placement.partition is None and plan.allocations[0].request.gpus == 0
    assert any("site.cpu_partition is not set" in w for w in plan.warnings)


def test_a_warmer_provisions_the_task_its_overrides_make(tmp_path, login):
    config = write_config(tmp_path)
    plan = _plan(config, overrides={"task.data_source.split.test.value": "3"})
    assert plan.stage("provision").params["task"] == {"module": "tests.provision.tasks",
                                                      "overrides": {"task.data_source.split.test.value": 3}}
    assert _plan(config, overrides={"run.poll_sec": "5"}).settings.run.poll_sec == 5
    with pytest.raises(ValueError, match="not artifact.image"):
        _plan(config, overrides={"artifact.image": "x"})
    # An override that names no field of the task is refused before anything is submitted.
    with pytest.raises(ValueError, match="name no field of the task"):
        _plan(config, overrides={"task.no_such_field": "1"})


#: The largest memory measured of any provision role of each task in this checkout, in GiB, the basis
#: of what each provision role of it gets (``TaskSpec.provision_memory_gib``, ``PROVISION_ROLE_GIB``).
PROVISION_PEAK_GIB = {
    "tasks.multitenant_slo.task": 5.35,
    "tasks.tab_completion.task": 4.15,
    "tasks.multimodal_asr.task": 5.55,
    "tasks.multimodal_tts.task": 2.67,
    "tasks.sample_task.task": 1.00,
    # The rows role's resident peak, which holds its footprint and its mapped dataset files.
    "tasks.multimodal_omni.task": 16.12,
    # The tools-full role's footprint as the stage runs the role.
    "tasks.deepresearch.task": 15.66,
}
#: Graded tasks of the checkout that declare no provision memory, so their provision roles get the
#: planner's default (``PROVISION_ROLE_GIB``).
DEFAULT_PROVISION = ("tasks.agentic_coding.task",)


@pytest.mark.parametrize("module, peak", sorted(PROVISION_PEAK_GIB.items()))
def test_each_provision_role_gets_its_tasks_measured_peak_and_a_quarter_more(tmp_path, login, module, peak):
    # A task whose peak and a quarter more passes the default declares that, rounded up; the others
    # take the default, which holds theirs.
    needed = math.ceil(peak * 1.25)
    declared = needed if needed > PROVISION_ROLE_GIB else None
    task = load_task(TaskRef(module=module))
    assert task.provision_memory_gib == declared
    plan = _plan(write_config(tmp_path, secrets=SECRETS), module=module)
    stage, reserve = plan.stage("provision"), plan.settings.run.runner_memory_gib
    assert stage.shape.memory_gib == (declared or PROVISION_ROLE_GIB) + reserve and not stage.shape_overridden
    assert role_shape(stage.shape, reserve).memory_gib >= peak * 1.25
    # The warmer's allocation asks for the stage's memory.
    assert plan.allocations[0].request.memory_gib == stage.shape.memory_gib


@pytest.mark.parametrize("module", DEFAULT_PROVISION)
def test_a_task_that_declares_no_provision_memory_plans_its_provision_at_the_default(tmp_path, login, module):
    assert load_task(TaskRef(module=module)).provision_memory_gib is None
    plan = _plan(write_config(tmp_path, secrets=SECRETS, sandbox_stores=task_sandbox_stores(module)), module=module)
    # Its tool runner names stores, which its provision stage makes what the runner needs in.
    assert plan.stage("provision").params["sandbox_stores"] == list(task_sandbox_stores(module))
    stage, reserve = plan.stage("provision"), plan.settings.run.runner_memory_gib
    assert stage.shape.memory_gib == PROVISION_ROLE_GIB + reserve and not stage.shape_overridden


@pytest.mark.xfail(strict=True, reason="agentic_coding states no provision peak: the resident peak of its "
                                       "provision roles is not measured, so they plan at the default")
def test_every_graded_task_of_the_checkout_states_its_provision_peak():
    graded = {catalog.task_module(t) for t in catalog.task_keys()}
    assert graded == set(PROVISION_PEAK_GIB)


def test_the_graded_tasks_that_state_no_provision_peak_are_the_ones_listed_as_taking_the_default():
    # The expected failure above stands for the listed tasks only: another task without a peak fails here.
    graded = {catalog.task_module(t) for t in catalog.task_keys()}
    assert graded - set(PROVISION_PEAK_GIB) == set(DEFAULT_PROVISION)
    assert set(PROVISION_PEAK_GIB) <= graded


def test_a_site_sets_a_tasks_provision_memory_by_the_tasks_id_and_never_by_a_task_override(tmp_path, login):
    config = write_config(tmp_path, secrets=SECRETS)
    module = "tasks.multimodal_omni.task"
    plan = _plan(config, module=module, overrides={"resources.multimodal-omni.provision.memory_gib": "60"})
    assert plan.stage("provision").shape.memory_gib == 60 and plan.stage("provision").shape_overridden
    # The task's directory names no task: a table keyed by it would change nothing.
    with pytest.raises(PlanError, match=r"^settings name unknown tasks: resources\.multimodal_omni; a resources table "
                                        r"is keyed by a task's TaskSpec id, one of .*multimodal-omni"):
        _plan(config, module=module, overrides={"resources.multimodal_omni.provision.memory_gib": "60"})
    with pytest.raises(ValueError, match=r"overrides may not set task\.provision_memory_gib"):
        _plan(config, module=module, overrides={"task.provision_memory_gib": "60"})


@pytest.mark.parametrize("port", [8888, 8899])
def test_a_warmer_of_a_task_on_an_interceptor_port_is_refused(tmp_path, login, port):
    with pytest.raises(ValueError, match=f"^T-provision's contract port {port} is refused: the egress interceptor's "
                                         f"[a-z ]+ listens on {port} "):
        _plan(write_config(tmp_path), overrides={"task.contract.port": str(port)})


def test_the_plan_holds_grant_names_and_never_a_value(tmp_path, login):
    plan = _plan(write_config(tmp_path, secrets=SECRETS))
    assert plan.stage("provision").grants == {"HF_TOKEN": "hf_token"}
    assert TOKEN not in plan.model_dump_json()


def test_without_a_secrets_file_nothing_is_granted_and_the_command_says_so(tmp_path, login):
    lines: list[str] = []
    plan = _plan(write_config(tmp_path), lines=lines)
    assert plan.stage("provision").grants == {}
    assert lines == [f"note: no secrets file at {tmp_path / 'secrets.toml'}; no stage is granted a secret"]


def test_a_secrets_file_that_is_not_private_is_refused(tmp_path, login):
    config = write_config(tmp_path, secrets=SECRETS)
    (tmp_path / "secrets.toml").chmod(0o644)
    with pytest.raises(SecretsError):
        _plan(config)
    (tmp_path / "secrets.toml").unlink()
    (tmp_path / "secrets.toml").symlink_to(tmp_path / "elsewhere.toml")
    # A dangling symlink at the secrets path is refused, not taken for a missing file.
    with pytest.raises(SecretsError):
        _plan(config)


def test_a_secrets_path_that_cannot_be_looked_at_is_refused_not_taken_for_a_missing_file(tmp_path, login):
    (tmp_path / "plain").write_text("x")
    with pytest.raises(SecretsError, match="NotADirectoryError"):
        _plan(write_config(tmp_path), overrides={"site.secrets_path": str(tmp_path / "plain" / "secrets.toml")})


@pytest.mark.parametrize("module, match", [
    ("tests.provision.fake_hub", "has no TASK"),
    ("tests.provision.no_such_module", "cannot import task module tests.provision.no_such_module"),
    ("tasks", "task module tasks has no file: it is a namespace package or a built-in module"),
    ("..x", "'..x' is not a dotted module name"),
    ("tasks..x", "'tasks..x' is not a dotted module name"),
])
def test_a_module_that_names_no_task_is_refused(tmp_path, login, module, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        _plan(write_config(tmp_path), module=module)


@pytest.mark.parametrize("task", [SimpleNamespace(), SimpleNamespace(id=5), SimpleNamespace(id=None)])
def test_a_task_without_a_string_id_is_refused(tmp_path, login, monkeypatch, task):
    here = str(commands.checkout_root() / "tests" / "provision" / "odd.py")
    monkeypatch.setitem(sys.modules, "odd_task", SimpleNamespace(TASK=task, __file__=here))
    with pytest.raises(ValueError, match="task module odd_task: its TASK has no string id"):
        _plan(write_config(tmp_path), module="odd_task")


def test_the_checkout_is_on_the_import_path_only_while_the_task_module_imports(tmp_path, monkeypatch):
    # A checkout that is not on the import path, as when hls runs from an installed script.
    (tmp_path / "warmer_probe_task.py").write_text("from types import SimpleNamespace\n"
                                                   "TASK = SimpleNamespace(id='probe')\n")
    monkeypatch.setattr(commands, "checkout_root", lambda: tmp_path.resolve())
    before = list(sys.path)
    try:
        assert commands.task_id_of("warmer_probe_task") == "probe"
    finally:
        sys.modules.pop("warmer_probe_task", None)
    assert sys.path == before


def test_a_task_module_outside_the_checkout_is_refused(tmp_path, login, monkeypatch):
    monkeypatch.setattr(commands, "checkout_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="which is not in the checkout"):
        _plan(write_config(tmp_path))


def test_an_imported_module_whose_file_climbs_out_of_the_checkout_is_refused(tmp_path, login, monkeypatch):
    # Read without resolving, the file name lies in the checkout.
    climbing = str(commands.checkout_root() / ".." / "elsewhere" / "climb.py")
    monkeypatch.setitem(sys.modules, "climb_task", SimpleNamespace(TASK=SimpleNamespace(id="climb"),
                                                                   __file__=climbing))
    with pytest.raises(ValueError, match="which is not in the checkout"):
        _plan(write_config(tmp_path), module="climb_task")


@pytest.mark.parametrize("module, files", [
    ("outside_task", {"outside_task.py": "TASK = None"}),
    # The package is refused before its __init__ runs, which finding the module inside would do.
    ("outside_pkg.task", {"outside_pkg/__init__.py": "", "outside_pkg/task.py": "TASK = None"}),
])
def test_a_module_outside_the_checkout_is_refused_before_it_runs(tmp_path, monkeypatch, forget, module, files):
    _checkout(tmp_path, monkeypatch)
    ran = tmp_path / "ran"
    for rel, text in files.items():
        (tmp_path / "lib" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "lib" / rel).write_text(f"open({str(ran)!r}, 'w').close()\n{text}\n")
    monkeypatch.syspath_prepend(str(tmp_path / "lib"))
    forget.extend(["outside_task", "outside_pkg", "outside_pkg.task"])
    with pytest.raises(ValueError, match="which is not in the checkout"):
        commands.task_id_of(module)
    assert not ran.exists()


def test_a_module_linked_from_outside_the_checkout_is_refused_before_it_runs(tmp_path, monkeypatch, forget):
    root = _checkout(tmp_path, monkeypatch)
    ran = tmp_path / "ran"
    (tmp_path / "elsewhere").mkdir()
    target = tmp_path / "elsewhere" / "linked_task.py"
    target.write_text(f"open({str(ran)!r}, 'w').close()\nTASK = None\n")
    (root / "linked_task.py").symlink_to(target)
    forget.append("linked_task")
    with pytest.raises(ValueError, match=re.escape(f"task module linked_task is {target.resolve()}, which is not in "
                                                   "the checkout")):
        commands.task_id_of("linked_task")
    assert not ran.exists()


@pytest.mark.parametrize("imported", [False, True])
def test_a_package_installed_in_an_environment_inside_the_checkout_is_refused(tmp_path, monkeypatch, forget,
                                                                              imported):
    # A virtual environment inside the checkout holds installed libraries, not the checkout's code.
    root = _checkout(tmp_path, monkeypatch)
    site = root / ".venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    ran = tmp_path / "ran"
    (site / "installed_task.py").write_text(f"open({str(ran)!r}, 'w').close()\nTASK = None\n")
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    monkeypatch.syspath_prepend(str(site))
    forget.append("installed_task")
    if imported:
        monkeypatch.setitem(sys.modules, "installed_task", SimpleNamespace(TASK=SimpleNamespace(id="x"),
                                                                           __file__=str(site / "installed_task.py")))
    with pytest.raises(ValueError, match="which is installed in this interpreter's environment"):
        commands.task_id_of("installed_task")
    assert not ran.exists()


def test_the_checkout_comes_first_on_the_import_path(tmp_path, monkeypatch, forget):
    # A package of the same name earlier on the path, as a working directory in another checkout would put there.
    root = _checkout(tmp_path, monkeypatch)
    other = tmp_path / "other-checkout"
    for where, task_id in ((root, "mine"), (other, "theirs")):
        (where / "shadowed").mkdir(parents=True)
        (where / "shadowed" / "__init__.py").write_text("")
        (where / "shadowed" / "task.py").write_text("from types import SimpleNamespace\n"
                                                    f"TASK = SimpleNamespace(id={task_id!r})\n")
    monkeypatch.syspath_prepend(str(other))
    forget.extend(["shadowed", "shadowed.task"])
    assert commands.task_id_of("shadowed.task") == "mine"


REFUSE = """
import sys
from hls.pipeline import commands
try:
    commands.task_id_of(sys.argv[1])
except ValueError as e:
    print("refused:", e)
print(sys.argv[1] in sys.modules)
"""


@pytest.mark.parametrize("module", ["transformers", "datasets"])
def test_an_installed_library_named_as_the_task_module_is_refused_without_importing_it(module):
    root = commands.checkout_root()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    out = subprocess.run([sys.executable, "-c", REFUSE, module], capture_output=True, text=True, cwd=root, env=env,
                         check=True, timeout=120).stdout.splitlines()
    assert out[0].startswith(f"refused: task module {module} is ") and out[1] == "False"


def test_the_task_id_must_be_one_store_key_segment(tmp_path, login, monkeypatch):
    here = str(commands.checkout_root() / "tests" / "provision" / "slashed.py")
    monkeypatch.setitem(sys.modules, "slashed_task", SimpleNamespace(TASK=SimpleNamespace(id="a/b"), __file__=here))
    with pytest.raises(ValueError, match="invalid store key segment 'a/b'"):
        _plan(write_config(tmp_path), module="slashed_task")


PROBE = """
import json, sys
from importlib.abc import MetaPathFinder
from pathlib import Path

HEAVY = {"torch", "transformers", "datasets", "huggingface_hub", "vllm", "pyarrow"}
attempted = set()


class Recorder(MetaPathFinder):
    # Sees every import attempt, of a library this interpreter lacks as well.
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in HEAVY:
            attempted.add(name.partition(".")[0])
        return None


sys.meta_path.insert(0, Recorder())
from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from tests.pipeline.plans import CODE
commands.code_state = lambda path: CODE
commands.toolchain_here = lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=Path("/"))
plan = commands.plan_provision(config=Path(sys.argv[1]), overrides={}, task_module=sys.argv[2], weights=True,
                               job_name="x", echo=lambda line: None)
print(json.dumps({"heavy": sorted(attempted), "plan": plan.model_dump_json()}))
"""


@pytest.mark.parametrize("module, task_id", [
    ("tasks.sample_task.task", "sample_task"),
    ("tasks.multitenant_slo.task", "multitenant_slo"),
    ("tasks.tab_completion.task", "tab_completion"),
    ("tasks.deepresearch.task", "deepresearch"),
    ("tests.support.token_level_task", "sample-token-level"),
])
def test_planning_on_the_login_node_loads_no_model_dataset_or_hub_library(tmp_path, module, task_id):
    # A real task module, in a fresh interpreter: nothing is downloaded on the login node,
    # and the secrets file is read for its grant names only. No import of a heavy library is
    # even tried, so one this interpreter lacks would be caught too.
    root = commands.checkout_root()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    out = subprocess.run([sys.executable, "-c", PROBE, str(write_config(tmp_path, secrets=SECRETS)), module],
                         capture_output=True, text=True, cwd=root, env=env, check=True, timeout=120).stdout
    result = json.loads(out.splitlines()[-1])
    assert result["heavy"] == []
    plan = json.loads(result["plan"])
    assert plan["instance"].startswith(f"warmers/{task_id}/")
    assert plan["stages"][0]["grants"] == {"HF_TOKEN": "hf_token"} and TOKEN not in result["plan"]


def test_a_site_whose_gpu_partition_needs_a_gpu_refuses_the_warmer_and_names_the_setting(tmp_path, login):
    with pytest.raises(PlanError, match=r"set site\.cpu_partition to a partition that takes jobs with no GPU"):
        _plan(write_config(tmp_path, cpu_partition=None, floor=1))


def _status_lines(config: Path) -> list[str]:
    """``hls status`` of a cache warmer planned and stored under *config*, nothing submitted."""
    plan = _plan(config)
    store = PosixStore(plan.store_root)
    handle = instance_lease(store, plan.layout, ttl_sec=30).try_acquire(Holder.here("test"))
    PlanStore(store, plan.layout).create(plan, handle)
    return commands.status_text(plan.instance, config=config, scheduler=FakeScheduler()).splitlines()


def test_status_names_the_partition_of_a_cache_warmer(tmp_path, login):
    text = "\n".join(_status_lines(write_config(tmp_path)))
    assert "allocation 0: stages provision; 0 x no GPU, 4 CPUs, 16 GiB, 258 min, partition cpu-a" in text


def test_status_leaves_out_a_cpu_partition_that_is_the_gpu_partition(tmp_path, login):
    lines = _status_lines(write_config(tmp_path, cpu_partition="gpu-a"))
    assert "allocation 0: stages provision; 0 x no GPU, 4 CPUs, 16 GiB, 258 min" in lines


def test_status_says_what_the_provision_stage_put_in_the_cache(tmp_path, login):
    """Once the provision stage published, ``hls status`` names each cache entry it provisioned:
    the weights at their commit, the rows and their tenants, each tool bundle's variant; a record
    that does not parse is named as unreadable. Before it publishes there is nothing to say."""
    config = write_config(tmp_path)
    plan = _plan(config)
    store = PosixStore(plan.store_root)
    handle = instance_lease(store, plan.layout, ttl_sec=30).try_acquire(Holder.here("test"))
    PlanStore(store, plan.layout).create(plan, handle)
    assert not any(line.startswith("provisioned") for line in commands.status_text(
        plan.instance, config=config, scheduler=FakeScheduler()).splitlines())
    tools = [ToolRef(name="corpus", identity="c1", variant=variant, dev_digest=digest,
                     key=tool_key("corpus", "c1", variant, digest), entry_sha256="2" * 64, disk_bytes=4096)
             for variant, digest in (("full", None), ("dev", "3" * 12))]
    weights, rows = weights_key("org/model", "a" * 40), rows_key("4" * 16)
    record = Provisioned(weights=WeightsRef(repo="org/model", revision="a" * 40, key=weights, entry_sha256="0" * 64,
                                            disk_bytes=4096),
                         rows=RowsRef(key=rows, entry_sha256="1" * 64, tenants=["t0", "t1"], disk_bytes=4096),
                         tools=tools)
    for data, expected in ((record.model_dump_json().encode(), [
            f"provisioned weights: org/model at {'a' * 40} (cache weights/{weights})",
            f"provisioned rows: tenants t0, t1 (cache rows/{rows})",
            "provisioned tool bundle: corpus c1, full (cache tools/corpus-c1-full)",
            f"provisioned tool bundle: corpus c1, dev (cache tools/corpus-c1-dev-{'3' * 12})"]),
            (b'{"weights": 1}', ["provisioned: unreadable (model_type weights; missing rows (required by this "
                                 "version of hls); missing tools (required by this version of hls))"])):
        key = plan.layout.attempt_file("provision", 1, "outputs/provisioned.json")
        store.write_bytes(key, data)
        publish_manifest(store, plan.layout, Manifest(
            instance=plan.instance, stage="provision", attempt=1,
            outputs={"provisioned": output_ref(store, key, sha256_bytes(data))}, inputs={},
            provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                                  image_digests={}, config_sha256="d" * 64),
            resources=Resources(declared=plan.stage("provision").shape,
                                granted=GrantedResources(gpus=0, cpus=4, memory_gib=16.0, disk_free_gib=30.0)),
            times=Times(submitted_at=1.0, allocation_started_at=2.0, started_at=3.0, ended_at=4.0)))
        lines = commands.status_text(plan.instance, config=config, scheduler=FakeScheduler()).splitlines()
        assert [line for line in lines if line.startswith("provisioned")] == expected
        if data != b'{"weights": 1}':
            # The same manifest over a record changed since, then over none.
            store.path(key).write_bytes(data.replace(b"org/model", b"org/other"))
            lines = commands.status_text(plan.instance, config=config, scheduler=FakeScheduler()).splitlines()
            assert [line for line in lines if line.startswith("provisioned")] == [
                f"provisioned: unreadable ({key} is not the file its manifest recorded)"]
            store.path(key).unlink()
            lines = commands.status_text(plan.instance, config=config, scheduler=FakeScheduler()).splitlines()
            assert [line for line in lines if line.startswith("provisioned")] == [
                f"provisioned: unreadable ({key} is missing)"]
        store.path(plan.layout.manifest("provision")).unlink()
