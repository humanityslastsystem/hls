"""``hls run --build-only`` on one machine: the command plans and submits, the in-process Slurm of
``tests/pipeline/test_end_to_end_local.py`` runs the real runner, and all four stages run their
real roles in one allocation. Each job's node has exactly the disk its allocation asked for,
which every role's grant check counts its scratch against, and a fake job cgroup the runner
watches. Only srun, the cluster, the hub, pasta, the container runtime, pip and the agent's CLI
are stood in for: the provision role reads the fake hub; the agent cap is extracted, and the
sandbox, the commit point and the snapshots run from its tree, by the agent build tests' fake
container (``tests.agent.fake_sandbox``), which runs the real exec server without namespaces,
and harbor's real Claude Code adapter drives the fake ``claude`` (``tests.agent.fake_cli``),
which commits a submission; the package stage's extractions and steps run in the package
tests' fake container and builds (``tests.package.fake_tools``). One ``apptainer`` on PATH hands
each command to its fake by the image: the agent cap's to the sandbox fake, every other to the
package fake.

These tests need harbor with ``SessionReentry``: the checkout's ``harbor`` submodule, or its
source first on PYTHONPATH."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hls.agent import environment, harnesses
from hls.arms import ArmError
from hls.package.result import PackageResult
from hls.pipeline import commands
from hls.pipeline import stage as stage_module
from hls.pipeline.plan import PlanStore
from hls.recipes import catalog
from hls.recipes.model import order
from hls.stages import agent_build
from hls.stages.agent_build import AgentBuildParams, BuildRecord
from hls.stages.finalize import FAILED_DIR, failed_key
from hls.stages.images import IMAGE_FILE, image_keys
from hls.store.cache import ContentCache
from hls.store.digest import sha256_bytes
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.base import NodeSize
from hls.substrate.memory_watch import MemoryWatch
from hls.substrate.secret_forms import FORMS
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE, PackageScan
from hls.substrate.apptainer.images import ApptainerImages
from tests.agent import fake_cli, fake_sandbox
from tests.agent.build_fixture import KEY, OPUS, TASK_ID, build_config
from tests.agent.test_stage import MappedEnvironment
from tests.evaluate import fixture
from tests.package import fake_tools
from tests.package.bundles import PYPROJECT, SERVER
from tests.package.test_stage import REPORT
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.hls_cli import run_hls
from tests.pipeline.test_end_to_end_local import LocalSlurm
from tests.provision import fake_hub
from tests.provision.warmer_config import TOKEN
from tests.store.entries import populate
from tests.substrate import fake_pasta
from tests.support import node

FAKE_ROLE = Path(fake_hub.__file__).with_name("fake_role.py")
STAGES = ["provision", "images", "agent_build", "source_scan", "package", "finalize"]
LAYER = catalog.image_set("sample_task").layer
CAP = catalog.AGENT_CAPS[LAYER].name
#: The agent cap's image file: an image the sandbox fake extracts, with nothing at /workspace.
CAP_IMAGE = fake_sandbox.image_bytes()
#: The node the build's one job runs on: its allocation, one GPU and 16 CPUs, which holds one
#: agent build and two package stages of their sizes.
JOB_NODE = NodeSize(gpus=1, cpus=16)


@pytest.fixture
def bins(tmp_path, tmp_path_factory, monkeypatch, fake_seal):
    """The fakes on PATH, the fake hub, the probe's sealed checks answered as a working seal's
    (``fake_seal``), the runner's clock moved by the fake CLI, the commands' node
    (``tests.support.node``: the hard limit whose ceiling this host's kernel sets on the sandbox's
    real commands, on a node of the job's own allocation, ``JOB_NODE``), and the runner's
    environment clean of what harbor would read. Returns (bin dir, the CLI's control directory)."""
    node.require_commands_ceiling()
    for var in harnesses.harness_env_leaks(os.environ):
        monkeypatch.delenv(var)
    base = tmp_path_factory.mktemp("b")
    bin_dir, sandbox_dir, package_dir = base / "bin", base / "sandbox", base / "package"
    cli_dir, control = base / "cli", base / "control"
    fake_pasta.install(bin_dir)
    sandbox = fake_sandbox.install(sandbox_dir)
    tools = base / "tools.json"
    fake_tools.install(package_dir, tools)
    tools.write_text(json.dumps({"dry_run": "ok", "wheel": "ok", "build": "ok", "report": REPORT,
                                 "record": str(base / "tools.log")}))
    # The cap's tree is extracted into .extract-<12 hex>-* and runs from sandboxes/<sha256>,
    # both named by its image file's digest.
    cap = sha256_bytes(CAP_IMAGE)[:12]
    (bin_dir / "apptainer").write_text(
        "#!/bin/sh\n"
        f'case "$*" in *"{cap}"*) exec {shlex.quote(str(sandbox))} "$@";; esac\n'
        f'exec {shlex.quote(str(package_dir / "apptainer"))} "$@"\n')
    (bin_dir / "apptainer").chmod(0o755)
    fake_cli.install(cli_dir, control)
    venv = Path(sys.executable).parent
    monkeypatch.setenv("PATH", f"{bin_dir}:{package_dir}:{venv}:/usr/bin:/bin")
    # The seal resolves its firewall tools from here, not from the system's /usr/sbin.
    monkeypatch.setenv("HLS_SEAL_TOOL_PATH", str(bin_dir))
    monkeypatch.setenv("FAKE_SANDBOX_PATH", f"{cli_dir}:{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_SANDBOX_RECORDS", str(base / "records"))
    (base / "pasta.plan").write_text("pass\n" * 16)
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(base / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(base / "pasta.log"))
    monkeypatch.setenv("FAKE_PYTHON_COUNTER", str(base / "counter"))
    for var in ("FAKE_APPTAINER_FAIL", "FAKE_EXTRACT_FAIL", "FAKE_PROXY_ERRORS", "FAKE_PROXY_MID_STREAM",
                "FAKE_PROXY_OVERSIZED", "FAKE_PROXY_REFUSED", "FAKE_PROXY_HEARTBEAT_AGE", "FAKE_TREE_GIB"):
        monkeypatch.delenv(var, raising=False)
    hub = tmp_path / "hub"
    fake_hub.make_hub(hub, "org/model", fixture.SHA, fixture.FILES)
    monkeypatch.setenv("FAKE_HUB", str(hub))
    offset = control / "offset"
    monkeypatch.setattr(agent_build, "monotonic",
                        lambda: time.monotonic() + (float(offset.read_text()) if offset.exists() else 0.0))
    monkeypatch.setattr(stage_module, "hard_limit", lambda: node.COMMANDS_HARD)
    monkeypatch.setattr(LocalLauncher, "node_size", lambda self: JOB_NODE if self.node is None else self.node)
    monkeypatch.setattr(environment, "SandboxEnvironment", MappedEnvironment)
    monkeypatch.setattr(MappedEnvironment, "SENT", [])
    monkeypatch.setattr(MappedEnvironment, "TIMEOUTS", [])
    monkeypatch.setattr(agent_build, "_ipv4s", lambda host: [])  # no DNS from a test
    return bin_dir, control


def _slurm(tmp_path: Path) -> LocalSlurm:
    """An in-process Slurm whose jobs run as soon as they are submitted, on a node with exactly
    the disk each asked for and a fake job cgroup, with the provision role program served by
    the fake hub."""
    slurm = LocalSlurm(replace_modules={"hls.provision.role": FAKE_ROLE}, disk_free_gib=None, node_disk=True,
                       job_cgroups=tmp_path / "cgroups")

    def submit_and_run(request):
        job_id = LocalSlurm.submit(slurm, request)
        slurm.tick()
        return job_id

    slurm.submit = submit_and_run
    return slurm


def _cache_images(store: PosixStore) -> None:
    """Publish the agent cap and the submission base, with what they build on, as ``hls images
    build`` would have: the base and the layer are tree images the package fake lays out."""
    keys = image_keys(order(catalog.task_images(TASK_ID).decls), ApptainerImages.recipe_key)
    cache, holder = ContentCache(store), Holder.here("test")
    for name, data in (("s-a", fake_tools.tree_image(fake_tools.BASE_TREE)),
                       (LAYER, fake_tools.tree_image(fake_tools.BASE_TREE)), (CAP, CAP_IMAGE)):
        populate(cache, "images", keys[name], lambda d, data=data: (d / IMAGE_FILE).write_bytes(data), holder=holder)


def _output(store: PosixStore, instance: str, stage: str, name: str) -> bytes:
    manifest, _ = read_manifest(store, InstanceLayout(instance), stage)
    return store.read_bytes(manifest.outputs[name].key)


def _bundle_main(bundle: Path, cwd: Path) -> str:
    out = subprocess.run(["git", "bundle", "list-heads", str(bundle)], capture_output=True, text=True, check=True,
                         cwd=cwd, env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1",
                                       "GIT_CONFIG_GLOBAL": "/dev/null", "HOME": str(cwd)})
    [line] = out.stdout.splitlines()
    sha, ref = line.split()
    assert ref == "refs/heads/main"
    return sha


def _build(config: Path, tmp_path: Path) -> tuple[int, list[str]]:
    """``hls provision`` of the task, which the plan sizes the build's copies of its data by,
    then ``hls run --build-only``; the run's exit status and what it said."""
    slurm = _slurm(tmp_path)
    code, provisioned = run_hls(slurm, ["provision", "bd_task.task", "--job-name", "hls-provision-test",
                                        "--config", str(config)])
    assert code == 0, provisioned
    return run_hls(slurm, ["run", "bd_task.task", "--arm", "arm-a", "--build-only", "--job-name", "hls-run-test",
                           "--config", str(config)])


def _instance(lines: list[str]) -> str:
    """The instance the command said it made, after what it printed of the checks it ran."""
    return next(line for line in lines if line.startswith("instance ")).removeprefix("instance ")


def _recorded(store: PosixStore, instance: str) -> str:
    """The key of the rollout record *instance*'s finalize published."""
    return read_manifest(store, InstanceLayout(instance), "finalize")[0].outputs["rollout"].key


def test_a_build_only_run_records_in_one_allocation_refuses_a_changed_arm_and_zeroes_nothing_gradable(
        tmp_path, short_dir, build_login, build_catalog, bins, monkeypatch):
    bin_dir, control = bins
    store = PosixStore(tmp_path / "store")
    _cache_images(store)
    (control / "script.json").write_text(json.dumps({"window": 1_000_000, "preflight": {}, "claude": [
        {"commit": SERVER, "commit_files": {"pyproject.toml": PYPROJECT}, "advance": 900}]}))
    scratch = short_dir  # the node's scratch, where the sandbox and the seals bind their sockets
    site = f'pasta_path = "{bin_dir / "pasta"}"'
    config = build_config(tmp_path, site=site, scratch=scratch)
    code, lines = _build(config, tmp_path)
    assert code == 0, lines
    instance = _instance(lines)
    assert instance.startswith(f"runs/{TASK_ID}/")
    assert any(line.startswith("allocation 0: provision, images, agent_build, source_scan, package, finalize; "
                               "1 x B200") for line in lines)
    assert lines[-2:] == [f"{instance}: succeeded: every stage published",
                          f"{instance}: rollout built ({_recorded(store, instance)})"]
    assert all(read_manifest(store, InstanceLayout(instance), s) is not None for s in STAGES)
    # The node had exactly the allocation's disk, the package stage's and the copies of the agent's
    # data, which last the job, and every role's grant was checked against what the job's scratch
    # left of it.
    plan = PlanStore(store, InstanceLayout(instance)).latest()
    data_gib = AgentBuildParams.model_validate(plan.stage("agent_build").params).data_gib
    assert plan.allocations[0].request.disk_gib == plan.stage("package").shape.disk_gib + data_gib

    # The agent committed on main; the build bundled that commit, and the package built it.
    build = BuildRecord.model_validate_json(_output(store, instance, "agent_build", "build"))
    assert (build.ending, build.arm.name, build.arm_change) == ("agent_ended", "arm-a", None)
    assert build.sandbox_image == sha256_bytes(CAP_IMAGE)
    manifest, _ = read_manifest(store, InstanceLayout(instance), "agent_build")
    commit = _bundle_main(store.path(manifest.outputs["source_bundle"].key), tmp_path)
    assert build.commit.commit == commit
    package = PackageResult.model_validate_json(_output(store, instance, "package", "package"))
    assert (package.status, package.commit, package.task) == ("built", commit, TASK_ID)
    assert package.bundle_sha256 == manifest.outputs["source_bundle"].sha256
    # The source scan read every object of the bundle the build published and its tree, the
    # package what its install added, each for every secret of the file in every form, and found
    # none.
    names = ["anthropic_key", "hf_token", "kimi_token", "kimi_url"]
    source = PackageScan.model_validate_json(_output(store, instance, "source_scan", "package_scan"))
    assert source == PackageScan(secrets=names, read=["bundle", "tree"], findings=[], forms=list(FORMS))
    scan = PackageScan.model_validate_json(_output(store, instance, "package", "package_scan"))
    assert scan == PackageScan(secrets=names, read=["install"], findings=[], forms=list(FORMS))
    assert read_manifest(store, InstanceLayout(instance), "package")[0].outputs["package_scan"].key.endswith(
        f"/outputs/{PACKAGE_SCAN_FILE}")
    # The arm's credential reached the CLI over the exec channel, and no published file holds it;
    # the hub token reached the hub through the provision stage's grant.
    [call] = [c for c in map(json.loads, (control / "calls.jsonl").read_text().splitlines()) if not c["preflight"]]
    assert call["credential_set"] and call["env"]["ANTHROPIC_MODEL"] == "claude-opus-4-8[1m]"
    assert [p for p in store.path(instance).rglob("*") if p.is_file() and KEY.encode() in p.read_bytes()] == []
    assert {c["token"] for c in fake_hub.calls(tmp_path / "hub")} == {TOKEN}
    # Every seal the build ran, the sandbox's and the package's two steps', forwards no port either
    # way, nor maps its gateway's address to the host's loopback.
    seals = fake_pasta.calls(bin_dir.parent / "pasta.log")
    assert len(seals) == 3 and all(fake_pasta.forwards(flags) == fake_pasta.NO_FORWARDS for flags in seals)
    assert all("--no-map-gw" in flags for flags in seals)

    # A rerun of the arm with another output limit is refused at plan time, and planned with the
    # change declared.
    changed = build_config(tmp_path, arms={"arm-a": {**OPUS, "output_limit": 32_000}}, site=site, scratch=scratch)
    with pytest.raises(ArmError, match=f"{instance}: output_limit: 64000 -> 32000"):
        commands.plan_run(config=changed, overrides={}, task_module="bd_task.task", arm="arm-a", arm_change=None,
                          job_name="x", build_only=True, echo=[].append)
    plan = commands.plan_run(config=changed, overrides={}, task_module="bd_task.task", arm="arm-a",
                             arm_change="the relay caps output at 32k", job_name="x", build_only=True, echo=[].append)
    params = AgentBuildParams.model_validate(plan.stage("agent_build").params)
    assert (params.arm.output_limit, params.arm_change) == (32_000, "the relay caps output at 32k")

    # The same arm again needs no declaration. This time the agent commits, then writes past the
    # sandbox's disk cap: the contestant's fault, which ends the build there. The build is graded
    # on its last commit: the commit point bundles it, and the package builds it.
    build_config(tmp_path, site=site, scratch=scratch)
    (control / "claude.count").unlink()
    (control / "script.json").write_text(json.dumps({"window": 1_000_000, "preflight": {}, "claude": [
        {"commit": SERVER, "commit_files": {"pyproject.toml": PYPROJECT}, "write": 3 << 20, "hang": True}]}))
    watch_disk = MemoryWatch.watch_disk

    def capped(self, step, paths, limit_bytes, kill, **kw):
        # The test cannot write 2 GiB: the cap the context asked for, scaled to 1 MiB.
        watch_disk(self, step, paths, limit_bytes // 2048, kill, **kw)

    monkeypatch.setattr(MemoryWatch, "watch_disk", capped)
    code, ended = _build(config, tmp_path)
    assert code == 0, ended
    capped_out = _instance(ended)
    assert ended[-2:] == [f"{capped_out}: succeeded: every stage published",
                          f"{capped_out}: rollout built ({_recorded(store, capped_out)})"]
    build = BuildRecord.model_validate_json(_output(store, capped_out, "agent_build", "build"))
    assert (build.ending, build.fault.fault_class) == ("disk_limit", "contestant")
    manifest, _ = read_manifest(store, InstanceLayout(capped_out), "agent_build")
    commit = _bundle_main(store.path(manifest.outputs["source_bundle"].key), tmp_path)
    package = PackageResult.model_validate_json(_output(store, capped_out, "package", "package"))
    assert (package.status, package.commit) == ("built", commit)
    assert InstanceLayout(capped_out).attempts(store, "agent_build") == [1]

    # Once more, the agent writes past the cap before it commits anything: nothing is gradable, so
    # the contestant's fault ends the build failed, never retried. The orchestrator writes the
    # failed run's records, its zero and its rollout record but no grading, which it never had;
    # the command reports the zero, and nothing after the build runs.
    (control / "claude.count").unlink()
    (control / "script.json").write_text(json.dumps({"window": 1_000_000, "preflight": {}, "claude": [
        {"write": 3 << 20, "hang": True}]}))
    code, again = _build(config, tmp_path)
    assert code == 0, again
    failed = _instance(again)
    plan = PlanStore(store, InstanceLayout(failed)).latest()
    assert (plan.outcome.status, plan.outcome.fault.fault_class) == ("failed", "contestant")
    reason = plan.outcome.fault.reason
    assert reason.startswith("build sandbox sandbox wrote past its disk cap")
    layout = InstanceLayout(failed)
    assert again[-2:] == [f"{failed}: failed: {plan.outcome.reason}",
                          f"{failed}: score 0.0, zero for contestant_fault ({failed_key(layout, 'score.json')})"]
    assert [s for s in STAGES if read_manifest(store, layout, s) is not None] == ["provision", "images"]
    assert layout.attempts(store, "agent_build") == [1]
    assert layout.attempts(store, "source_scan") == layout.attempts(store, "package") == []
    records = sorted(p.name for p in store.path(f"{failed}/{FAILED_DIR}").iterdir())
    assert "score.json" in records and "rollout.json" in records and "grading.json" not in records
