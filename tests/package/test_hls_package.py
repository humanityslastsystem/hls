from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec, Artifact, Budget,
                                   BuildAllocation, HighestAvailable)
from hls.pipeline import commands
from hls.pipeline.pipelines import images_shape, package_shape, source_scan_shape
from hls.pipeline.planner import PlanError
from hls.pipeline.stage import most_roles
from hls.recipes import catalog
from hls.recipes.model import CONSTRAINTS
from hls.stages.package import pip_limits
from hls.store.digest import sha256_file
from hls.store.layout import InstanceLayout
from hls.store.posix import PosixStore
from tests.package.bundles import GOOD, make_bundle
from tests.provision.warmer_config import SECRETS, write_config


def _config(tmp_path: Path) -> Path:
    """The warmer tests' hls.toml with a secrets file, which the package stage scans for."""
    return write_config(tmp_path, secrets=SECRETS)


def _producer(**kw) -> AgentProducerSpec:
    """A producer with the build allocation every task declares, whose writable disk cap the
    package holds a bundle to, and the effort every producer authors."""
    return AgentProducerSpec(hard_budget=Budget(wallclock_sec=60), build_allocation=BuildAllocation(cpus=8, memory_gib=180),
                             reasoning_effort=HighestAvailable(), **kw)


#: Stands for the digest an operator states for a bundle: its own sha256.
OWN = object()


def _store_files(tmp_path: Path) -> list[Path]:
    """Every file in the test's store, the store's temporary files among them."""
    return sorted(p for p in (tmp_path / "store").rglob("*") if p.is_file())


def _plan(config: Path, bundle: Path, *, module: str = "tasks.sample_task.task",
          overrides: dict[str, str] | None = None, digest: object = OWN):
    return commands.plan_package(config=config, overrides=overrides or {}, task_module=module, bundle=bundle,
                                 bundle_sha256=sha256_file(bundle)[0] if digest is OWN else digest,
                                 job_name="hls-package", echo=[].append)


def _stub_module(monkeypatch, directory: str, *, artifact, task_id: str | None = None) -> str:
    """Register the task module of ``tasks/<directory>``, ``tasks.<directory>.task``, whose ``TASK``
    (id *task_id*, else the directory) and ``ARTIFACT`` are given and whose ``IMAGES`` are the real
    module's, at a path in the checkout, and return its name."""
    name = f"tasks.{directory}.task"
    images = catalog.image_set(directory) if (commands.checkout_root() / "tasks" / directory).is_dir() else None
    here = str(commands.checkout_root() / "tests" / "package" / f"{directory}.py")
    task = SimpleNamespace(id=task_id or directory,
                           hardware=SimpleNamespace(accelerator=SimpleNamespace(kind="H200", count=1)))
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(TASK=task, ARTIFACT=artifact, IMAGES=images, __file__=here))
    return name


@pytest.fixture
def quiet(monkeypatch):
    """configure_logging turns propagation off on the hls loggers, which would break caplog in
    every later test of the session."""
    monkeypatch.setattr(cli, "configure_logging", lambda verbose=False: None)


def test_hls_package_plans_the_submission_base_the_source_scan_then_the_package_on_one_cpu_allocation(tmp_path,
                                                                                                       login):
    from tasks.sample_task.task import ARTIFACT
    bundle = make_bundle(tmp_path, GOOD)
    plan = _plan(_config(tmp_path), bundle)
    assert plan.instance.startswith("warmers/sample_task/") and plan.pipeline == "package"
    [alloc] = plan.allocations
    assert alloc.stages == ["images", "source_scan", "package"]
    assert alloc.placement.partition == "cpu-a" and alloc.request.gpus == 0
    images, scan, package = plan.stage("images"), plan.stage("source_scan"), plan.stage("package")
    assert [d["name"] for d in images.params["images"]] == ["s-a", "s-a-tf-5-13-1"]
    assert images.shape == images_shape(2)
    assert package.after == ["images", "source_scan"]
    # The source scan reads the bundle for every secret value and publishes what it found before
    # the package stage runs any of the contestant's code.
    assert scan.shape == source_scan_shape(32, runner_memory_gib=plan.settings.run.runner_memory_gib)
    assert scan.after == []
    # Both read the copy the command made in the instance, with the digest the operator stated,
    # and hold the bundle to the task's build cap, the writable disk its sandbox has.
    copy = {"key": f"{plan.instance}/source.bundle", "size": bundle.stat().st_size}
    assert scan.params == {"bundle": copy, "bundle_sha256": sha256_file(bundle)[0], "build_manifest": None,
                           "disk_cap_gib": 32}
    # The stages' times, the overhead, and the headroom for the steps of their 8, 2 and 14 roles to
    # start at the site's allowance.
    assert [most_roles(s.kind, s.params) for s in plan.stages] == [8, 2, 14]
    assert alloc.request.time_sec == (images.shape.time_sec + scan.shape.time_sec + package.shape.time_sec
                                      + plan.settings.run.allocation_overhead_sec
                                      + 24 * plan.settings.site.step_launch_allowance_sec)
    # Each stage's extracted trees go when it closes, so the package stage's own floor holds the
    # trees it pins and its build's room, more than its checkout at the task's 32 GiB cap, and the
    # allocation asks for the larger stage's disk.
    assert package.shape == package_shape(32) and package.shape.disk_gib == 224
    assert alloc.request.disk_gib == max(images.shape.disk_gib, scan.shape.disk_gib,
                                         package.shape.disk_gib) == 224
    assert package.params == {
        "task": "sample_task", "submission_base": "s-a-tf-5-13-1", "constraints": CONSTRAINTS,
        "images_stage": "images", "gpu_kind": "H200",
        "network": ARTIFACT.producer.network.model_dump(mode="json"), "server_filename": "server.py",
        "bundle": copy, "bundle_sha256": sha256_file(bundle)[0], "build_manifest": None,
        "disk_cap_gib": 32,
        "pip": pip_limits(package_shape(32), plan.settings.run.runner_memory_gib).model_dump()}


def test_the_bundle_is_copied_into_the_instance_and_the_plan_names_only_the_copy(tmp_path, login):
    """The command copies the operator's bundle into the new instance before it plans, the copy a
    file of its own, never a link to the operator's: the plan records the copy's key and size and
    the stated digest, and names the operator's file nowhere, so every substrate's roles read the
    copy where the store is."""
    bundle = make_bundle(tmp_path / "operator", GOOD)
    plan = _plan(_config(tmp_path), bundle)
    copy = PosixStore(plan.store_root).path(InstanceLayout(plan.instance).source_bundle)
    assert copy.read_bytes() == bundle.read_bytes()
    assert not os.path.samefile(copy, bundle) and copy.stat().st_nlink == bundle.stat().st_nlink == 1
    assert str(bundle) not in plan.model_dump_json()
    assert _store_files(tmp_path) == [copy]


def test_a_bundle_already_in_the_store_is_copied_all_the_same(tmp_path, login):
    """Under the store root the operator's file is still the operator's, which they may change
    later: the instance gets a copy of its own."""
    bundle = make_bundle(tmp_path / "store" / "inputs", GOOD)
    plan = _plan(_config(tmp_path), bundle)
    copy = PosixStore(plan.store_root).path(InstanceLayout(plan.instance).source_bundle)
    assert copy.read_bytes() == bundle.read_bytes()
    assert not os.path.samefile(copy, bundle) and copy.stat().st_nlink == bundle.stat().st_nlink == 1
    assert plan.stage("source_scan").params["bundle"]["key"] == f"{plan.instance}/source.bundle"


def test_a_bundle_that_is_not_the_stated_digest_is_refused_and_leaves_no_copy(tmp_path, login, monkeypatch):
    """The operator named the wrong file or the wrong digest: the command hashes the bundle as it
    copies it and refuses it before planning, with the digest it found, and the store keeps
    nothing of it."""
    planned = []
    monkeypatch.setattr(commands, "build_plan", lambda **kw: planned.append(kw))
    bundle = make_bundle(tmp_path, GOOD)
    with pytest.raises(ValueError, match=re.escape(f"--bundle {bundle} has sha256 {sha256_file(bundle)[0]}, not the "
                                                   f"{'ab' * 32} --sha256 states")):
        _plan(_config(tmp_path), bundle, digest="ab" * 32)
    assert planned == [] and _store_files(tmp_path) == []


def test_a_bundle_the_command_cannot_read_is_refused(tmp_path, login):
    bundle = make_bundle(tmp_path, GOOD)
    bundle.chmod(0)
    try:
        with pytest.raises(ValueError, match=re.escape(f"--bundle {bundle} cannot be read: Permission denied")):
            _plan(_config(tmp_path), bundle, digest="ab" * 32)
    finally:
        bundle.chmod(0o644)
    assert _store_files(tmp_path) == []


def test_a_bundle_past_the_tasks_build_disk_is_refused_before_a_byte_is_copied(tmp_path, login, monkeypatch):
    """The task's builds may write 1 GiB here: a file one byte past it is refused by its size,
    which the message names beside the cap, before the command copies any of it."""
    monkeypatch.setattr(commands, "sandbox_disk_gib", lambda task, producer: 1)
    monkeypatch.setattr(PosixStore, "open_write", lambda *a, **kw: pytest.fail("the bundle was copied"))
    big = tmp_path / "image.sif"
    with open(big, "wb") as f:
        f.truncate(2**30 + 1)
    with pytest.raises(ValueError, match=re.escape(f"--bundle {big} holds {2**30 + 1} bytes, more than the 1 GiB "
                                                   f"({2**30} bytes) the task's builds may write")):
        _plan(_config(tmp_path), big, digest="ab" * 32)
    assert _store_files(tmp_path) == []


class _Interrupting:
    """A copy's file that writes each chunk, then is interrupted as *how* says: Ctrl-C, or the
    signal that ends the command."""

    def __init__(self, f, how: str) -> None:
        self.f, self.how = f, how

    def write(self, data: bytes) -> None:
        self.f.write(data)
        if self.how == "ctrl-c":
            raise KeyboardInterrupt
        os.kill(os.getpid(), getattr(signal, self.how))
        time.sleep(10)


@pytest.mark.parametrize("how", ["ctrl-c", "SIGTERM", "SIGHUP"])
def test_an_interrupted_copy_leaves_nothing_in_the_store(tmp_path, login, monkeypatch, how):
    """A copy cut short by Ctrl-C, or by the signal a closed terminal or a kill sends, leaves no
    copy and no temporary file, and the command's own handling of those signals is put back."""
    bundle = make_bundle(tmp_path, GOOD)
    real = PosixStore.open_write

    @contextlib.contextmanager
    def interrupted(self, key, *, exclusive=False):
        with real(self, key, exclusive=exclusive) as f:
            yield _Interrupting(f, how)

    monkeypatch.setattr(PosixStore, "open_write", interrupted)
    monkeypatch.setattr(commands, "CHUNK", 64)
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGHUP)}
    with pytest.raises(KeyboardInterrupt):
        _plan(_config(tmp_path), bundle)
    assert _store_files(tmp_path) == []
    assert {signum: signal.getsignal(signum) for signum in handlers} == handlers


def test_the_package_stage_leaves_the_runner_its_memory(tmp_path, login):
    plan = _plan(_config(tmp_path), make_bundle(tmp_path, GOOD))
    assert plan.stage("package").shape.memory_gib - plan.settings.run.runner_memory_gib == 32


def test_a_package_warmer_waits_for_the_sites_other_warmers_unless_they_run_at_once(tmp_path, login):
    """Slurm's singleton keys on the job name, so a warmer run one at a time is submitted under the
    site's warmer job name and waits for the others; one run at once keeps its own name."""
    config, bundle = _config(tmp_path), make_bundle(tmp_path, GOOD)
    plan = _plan(config, bundle, overrides={"site.warmer_job_name": "hls-warm.kl"})
    assert plan.job_name == "hls-warm.kl" and plan.allocations and all(a.singleton for a in plan.allocations)
    plan = _plan(config, bundle, overrides={"site.warmer_job_name": "hls-warm.kl",
                                            "site.warmers_one_at_a_time": "false"})
    assert plan.job_name == "hls-package" and not any(a.singleton for a in plan.allocations)


def test_a_task_whose_id_is_not_its_directory_is_packaged_on_its_catalog_tasks_base(tmp_path, login):
    """ASR's TaskSpec is ``multimodal-asr`` and its directory, the catalog's name for it,
    ``multimodal_asr``: the submission base is the catalog's, the warmer and ``package.json`` the id's."""
    bundle = make_bundle(tmp_path, GOOD)
    plan = _plan(_config(tmp_path), bundle, module="tasks.multimodal_asr.task")
    assert plan.task == "multimodal-asr" and plan.instance.startswith("warmers/multimodal-asr/")
    assert [d["name"] for d in plan.stage("images").params["images"]] == ["s-a", "s-a-tf-5-9-0"]
    assert plan.stage("images").params["catalog_task"] == "multimodal_asr"
    params = plan.stage("package").params
    assert (params["task"], params["submission_base"]) == ("multimodal-asr", "s-a-tf-5-9-0")
    with pytest.raises(ValueError, match="is not a task module: a task module is tasks.<directory>.task"):
        _plan(_config(tmp_path), bundle, module="multimodal_asr.task")


def _fifo(path: Path) -> Path:
    os.mkfifo(path)
    return path


@pytest.mark.parametrize("make, match", [
    (lambda tmp: Path("source.bundle"), "must be an absolute path"),
    (lambda tmp: tmp / "missing.bundle", "is not a regular file"),
    (lambda tmp: tmp, "is not a regular file"),
    (lambda tmp: _fifo(tmp / "pipe"), "is not a regular file"),
], ids=["relative", "missing", "a-directory", "a-pipe"])
def test_the_bundle_is_an_absolute_regular_file(tmp_path, login, make, match):
    with pytest.raises(ValueError, match=match):
        _plan(_config(tmp_path), make(tmp_path), digest="ab" * 32)
    assert _store_files(tmp_path) == []


def test_a_symlink_is_not_taken_for_the_bundle(tmp_path, login):
    bundle = make_bundle(tmp_path, GOOD)
    link = tmp_path / "link.bundle"
    link.symlink_to(bundle)
    with pytest.raises(ValueError, match="is a symlink"):
        _plan(_config(tmp_path), link)


@pytest.mark.parametrize("digest, match", [
    (None, "--bundle needs --sha256"),
    ("", "--sha256: not a bare hex SHA-256: ''"),
    ("AB" * 32, "--sha256: not a bare hex SHA-256"),
    ("ab" * 31, "--sha256: not a bare hex SHA-256"),
    (f"sha256:{'ab' * 32}", "--sha256: not a bare hex SHA-256"),
], ids=["none", "empty", "uppercase", "short", "prefixed"])
def test_a_bundle_without_its_digest_is_refused_when_planned(tmp_path, login, monkeypatch, digest, match):
    """The operator states the bundle's sha256, which the allocation's copy is checked against:
    a plan without it, or with anything but 64 lowercase hex, is refused before anything is
    planned, and the CLI says so in one line."""
    planned = []
    monkeypatch.setattr(commands, "build_plan", lambda **kw: planned.append(kw))
    with pytest.raises(ValueError, match=match):
        _plan(_config(tmp_path), make_bundle(tmp_path, GOOD), digest=digest)
    assert planned == []


def test_the_cli_refuses_a_bundle_without_its_digest_in_one_line(tmp_path, login, quiet, monkeypatch):
    planned = []
    monkeypatch.setattr(commands, "build_plan", lambda **kw: planned.append(kw))
    bundle = make_bundle(tmp_path, GOOD)
    for extra, says in (([], "--bundle needs --sha256"), (["--sha256", "deadbeef"], "--sha256: not a bare hex")):
        result = CliRunner().invoke(cli.app, ["package", "tasks.sample_task.task", "--bundle", str(bundle), *extra,
                                              "--config", str(_config(tmp_path))])
        assert result.exit_code == 1 and type(result.exception) is SystemExit, result.output
        [line] = result.stderr.splitlines()
        assert line.startswith("hls package: ") and says in line
    assert planned == []


def test_a_task_the_image_catalog_does_not_know_is_refused(tmp_path, login, monkeypatch):
    module = _stub_module(monkeypatch, "not_in_catalog", artifact=AgentProducedArtifact(producer=_producer()))
    with pytest.raises(ValueError, match="no task 'not_in_catalog'"):
        _plan(_config(tmp_path), make_bundle(tmp_path, GOOD), module=module)


@pytest.mark.parametrize("directory", ["deepresearch"])
def test_a_submission_base_whose_base_holds_no_torch_is_packaged_with_no_constraints_file(tmp_path, login,
                                                                                         monkeypatch, directory):
    """It packages on its sandbox, built on the cuda base, which holds nothing: the package's
    install names no constraints file."""
    # The site's images warmer holds every image.
    monkeypatch.setattr(commands, "cache_holds", lambda settings: set)
    module = _stub_module(monkeypatch, directory, artifact=AgentProducedArtifact(producer=_producer()))
    params = _plan(_config(tmp_path), make_bundle(tmp_path, GOOD), module=module).stage("package").params
    assert params["submission_base"] == f"{directory.replace('_', '-')}-sandbox-agent"
    assert params["constraints"] is None


@pytest.mark.parametrize("artifact, match", [
    (None, "is None, not an agent-produced artifact"),
    (Artifact(), "is Artifact, not an agent-produced artifact"),
])
def test_only_an_agent_produced_submission_is_packaged(tmp_path, login, monkeypatch, artifact, match):
    module = _stub_module(monkeypatch, "sample_task", artifact=artifact)
    with pytest.raises(ValueError, match=match):
        _plan(_config(tmp_path), make_bundle(tmp_path, GOOD), module=module)


def test_an_open_producer_network_is_refused_when_planned(tmp_path, login, monkeypatch):
    """The pip steps run behind a seal with an index, which carries the producer's own deny
    list; an open producer network has no package rule, so its submission cannot be packaged."""
    artifact = AgentProducedArtifact(producer=_producer(network=None))
    module = _stub_module(monkeypatch, "sample_task", artifact=artifact)
    with pytest.raises(PlanError, match="stage package: the producer's network has no package-deny rule"):
        _plan(_config(tmp_path), make_bundle(tmp_path, GOOD), module=module)


def test_the_producers_network_and_server_file_are_the_ones_packaged(tmp_path, login, monkeypatch):
    artifact = AgentProducedArtifact(producer=_producer(server_rel_path="serve.py"))
    module = _stub_module(monkeypatch, "sample_task", artifact=artifact)
    params = _plan(_config(tmp_path), make_bundle(tmp_path, GOOD), module=module).stage("package").params
    assert params["network"] == artifact.producer.network.model_dump(mode="json")
    assert params["server_filename"] == "serve.py"


def test_a_task_without_a_gpu_compiles_for_none(tmp_path, login, monkeypatch):
    module = _stub_module(monkeypatch, "sample_task", artifact=AgentProducedArtifact(producer=_producer()))
    sys.modules[module].TASK.hardware.accelerator.kind = "CPU"
    assert _plan(_config(tmp_path), make_bundle(tmp_path, GOOD), module=module).stage(
        "package").params["gpu_kind"] is None


def test_only_site_and_run_overrides_are_accepted(tmp_path, login):
    with pytest.raises(ValueError, match="artifact.producer.network"):
        _plan(_config(tmp_path), make_bundle(tmp_path, GOOD), overrides={"artifact.producer.network": "x"})


def test_the_cli_routes_package_with_its_bundle_and_its_digest(tmp_path, monkeypatch, quiet):
    seen, started = {}, []
    monkeypatch.setattr(commands, "plan_package", lambda **kw: seen.update(kw) or "package plan")
    monkeypatch.setattr(commands, "start", lambda plan, overrides, **kw: started.append((plan, overrides)) or 0)
    bundle = make_bundle(tmp_path, GOOD)
    digest = sha256_file(bundle)[0]
    result = CliRunner().invoke(cli.app, ["package", "tasks.sample_task.task", "--bundle", str(bundle),
                                          "--sha256", digest, "--site.qos", "normal"])
    assert result.exit_code == 0, result.output
    assert (seen["task_module"], seen["bundle"], seen["bundle_sha256"], seen["job_name"]) == (
        "tasks.sample_task.task", bundle, digest, "hls-package")
    assert seen["overrides"] == {"site.qos": "normal"}
    assert started == [("package plan", {"site.qos": "normal"})]


@pytest.mark.parametrize("args, says", [
    (["--site.partition", "l40", "tasks.sample_task.task", "--bundle", "BUNDLE", "--sha256", "DIGEST"],
     "expected the task module first, got override flag"),
    (["tasks.sample_task.task", "--bundle", "relative.bundle", "--sha256", "DIGEST"], "must be an absolute path"),
    (["multimodal_asr.task", "--bundle", "BUNDLE", "--sha256", "DIGEST"], "is not a task module"),
])
def test_a_refusal_is_one_line(tmp_path, login, quiet, args, says):
    bundle = make_bundle(tmp_path, GOOD)
    swap = {"BUNDLE": str(bundle), "DIGEST": sha256_file(bundle)[0]}
    args = [swap.get(a, a) for a in args]
    result = CliRunner().invoke(cli.app, ["package", *args, "--config", str(_config(tmp_path))])
    assert result.exit_code == 1 and type(result.exception) is SystemExit, result.output
    [line] = result.stderr.splitlines()
    assert line.startswith("hls package: ") and says in line


PROBE = """
import json, sys
from pathlib import Path
from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from tests.pipeline.plans import CODE
commands.code_state = lambda path: CODE
commands.toolchain_here = lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=Path("/"))
plan = commands.plan_package(config=Path(sys.argv[1]), overrides={}, task_module="tasks.sample_task.task",
                             bundle=Path(sys.argv[2]), bundle_sha256=sys.argv[3], job_name="x",
                             echo=lambda line: None)
heavy = sorted(m for m in ("torch", "transformers", "datasets", "huggingface_hub", "vllm", "pyarrow")
               if m in sys.modules)
print(json.dumps({"heavy": heavy, "stages": [s.name for s in plan.stages]}))
"""


def test_planning_a_package_on_the_login_node_imports_nothing_heavy(tmp_path):
    bundle = make_bundle(tmp_path, GOOD)
    digest = sha256_file(bundle)[0]
    root = commands.checkout_root()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    out = subprocess.run([sys.executable, "-c", PROBE, str(_config(tmp_path)), str(bundle), digest],
                         capture_output=True, text=True, cwd=root, env=env, check=True, timeout=120).stdout
    assert json.loads(out.splitlines()[-1]) == {"heavy": [], "stages": ["images", "source_scan", "package"]}


def test_a_package_on_a_site_without_its_secrets_file_is_refused_when_planned(tmp_path, login):
    """The source scan and the package stage scan the submission for every value of the site's
    secrets file, so on a site with none its allocation would only stop as it started: the plan is
    refused instead."""
    config = write_config(tmp_path)
    with pytest.raises(PlanError) as e:
        _plan(config, make_bundle(tmp_path, GOOD))
    assert str(e.value) == (f"stages source_scan, package need every value of the site's secrets file, and there "
                            f"is none at {tmp_path / 'secrets.toml'}")


def test_the_cli_refuses_a_package_without_the_secrets_file_in_one_line_before_submitting(tmp_path, login, quiet,
                                                                                          monkeypatch):
    monkeypatch.setattr(commands, "start", lambda *args, **kw: pytest.fail("the plan was submitted"))
    bundle = make_bundle(tmp_path, GOOD)
    result = CliRunner().invoke(cli.app, ["package", "tasks.sample_task.task", "--bundle", str(bundle), "--sha256",
                                          sha256_file(bundle)[0], "--config", str(write_config(tmp_path))])
    assert result.exit_code == 1 and type(result.exception) is SystemExit, result.output
    [line] = result.stderr.splitlines()
    assert line.startswith("hls package: stages source_scan, package need every value of the site's secrets file")


def test_the_runners_import_of_the_stages_registers_images_source_scan_and_package():
    # The runner resolves a plan's stage kinds after `import hls.stages.registry` alone, never
    # through the pipelines module, so the registry registers each itself.
    probe = ("import hls.stages.registry\nfrom hls.pipeline.stage import stage_class\n"
             "print(stage_class('images').kind, stage_class('source_scan').kind, stage_class('package').kind)")
    root = commands.checkout_root()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=root, env=env,
                         check=True, timeout=120).stdout
    assert out.split() == ["images", "source_scan", "package"]
