"""The driver's sandbox stores: a task whose tool runner starts containers of its own names what
its driver must be given (``ToolRunner.host_paths``), and the drive stage binds the site's
sandbox stores (``Settings.sandbox_stores``) at the paths the runner declares, with the device a
rootless runtime mounts an image through. Only the driver gets them; a task whose runner names
nothing gets the driver it always got."""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hls.pipeline.stage import StageContext, StageFault
from hls.spec.tools import DEVICE, HostPath
from hls.stages import deploy_drive
from hls.stages.deploy_drive import (DRIVER_TARGETS, SANDBOX_DEVICES, DeployDriveParams, DeployDriveStage,
                                     is_device, sandbox_binds, sandbox_problems)
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.egress.policy import SealPolicy
from hls.substrate.container import ContainerSpec, Mount, SealPlan
from tasks.agentic_coding.task import TOOL_RUNNER
from tests.drive import fake_container
from tests.drive import test_stage as stage_tests
from tests.drive.test_stage import ARTIFACT_SHAPE, REPO, _ctx, _drive, _params, _started
from tests.substrate.sandboxes import stand_in

#: The drive stage tests' fake runtime and seal, which every drive here runs on.
bins, interceptor = stage_tests.bins, stage_tests.interceptor
#: The host paths the drive's params carry: agentic_coding's tool runner's own.
PATHS = TOOL_RUNNER.host_paths
#: The stores they name.
STORES = tuple(dict.fromkeys(p.store for p in PATHS if p.store != DEVICE))


def _declared() -> list[dict]:
    return [dataclasses.asdict(p) for p in PATHS]


def _stores(tmp_path: Path) -> dict[str, str]:
    """A directory for each store, beside the test's store root, never in it."""
    stores = {}
    for name in STORES:
        d = tmp_path / "sandbox-stores" / name
        d.mkdir(parents=True)
        stores[name] = str(d)
    return stores


def _site(tmp_path: Path) -> dict[str, str]:
    """A secrets file in a directory of its own, which no store lies in."""
    (tmp_path / "secrets").mkdir(exist_ok=True)
    return {"secrets_path": str(tmp_path / "secrets" / "secrets.toml")}


@pytest.fixture
def specs(monkeypatch) -> dict[str, ContainerSpec]:
    """Every container spec the stage starts, by role."""
    seen: dict[str, ContainerSpec] = {}
    start = StageContext.start_container

    def recording(self, name, spec, *args, **kwargs):
        seen[name] = spec
        return start(self, name, spec, *args, **kwargs)

    monkeypatch.setattr(StageContext, "start_container", recording)
    return seen


@pytest.fixture
def fuse(monkeypatch):
    """This node has the fuse device, whatever the machine running the test has."""
    monkeypatch.setattr(deploy_drive, "is_device", lambda path: path in SANDBOX_DEVICES)


def _targets(spec: ContainerSpec) -> list[tuple[str, bool]]:
    return [(m.target, m.read_only) for m in spec.mounts]


#: The driver's own mounts after its data mounts, by container path and read-only flag, as every
#: drive gives them.
OWN = [("/opt/hls-src", True), ("/hls/task-ref.json", True), ("/hls/local", False), ("/hls/out", False),
       ("/hls/outputs.json", False), ("/hls/stream.json", False), ("/hls/cache", False)]
#: The driver's data mounts in the stage tests' drives.
DATA = [("/data/weights", True), ("/data/tenants/t0/dev", True), ("/data/tenants/t0/test", True)]


def test_a_driver_whose_runner_names_no_host_path_gets_its_mounts_and_no_device(tmp_path, bins, specs):
    outputs = DeployDriveStage().run(_ctx(tmp_path, bins))
    driver = specs["driver"]
    assert _targets(driver) == DATA + OWN
    assert driver.devices == ()
    assert all(spec.devices == () for spec in specs.values())
    record = fake_container.records(tmp_path / "records")["driver"]
    assert not [b for b in record["binds"] if "/dev/" in b or "/hls/sandboxes" in b]
    assert record["env"]["PYTHONPATH"] == "/opt/hls-src/src:/opt/hls-src"
    assert outputs["drive"]


def test_the_driver_gets_each_store_at_its_path_and_the_device_and_nothing_else_changes(tmp_path, bins, specs, fuse):
    stores = _stores(tmp_path)
    ctx = _ctx(tmp_path, bins, stores=stores, site=_site(tmp_path), sandbox_paths=_declared())
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served"
    driver = specs["driver"]
    added = [Mount(Path(stores[p.store]), p.path, read_only=p.read_only) for p in PATHS if p.store != DEVICE]
    # The driver's own mounts first, as in every drive, then the stores, read-only but for the
    # overlay root the runner writes each session under.
    assert _targets(driver) == DATA + OWN + [(m.target, m.read_only) for m in added]
    assert list(driver.mounts[len(DATA + OWN):]) == added
    assert [m.target for m in added if not m.read_only] == ["/hls/sandboxes/overlays"]
    assert driver.devices == ("/dev/fuse",)
    # The runtime's command line is the one it renders without them, with a bind of each and of the
    # device before the env file, and nothing else.
    runtime = ApptainerRuntime()
    plain = dataclasses.replace(driver, mounts=driver.mounts[:len(DATA + OWN)], devices=())
    with_stores, without = runtime.argv(driver, env_file="/dev/fd/3"), runtime.argv(plain, env_file="/dev/fd/3")
    at = without.index("--env-file")
    binds = [arg for m in added for arg in ("--bind", f"{m.source}:{m.target}" + (":ro" if m.read_only else ""))]
    assert with_stores == [*without[:at], *binds, "--bind", "/dev/fuse:/dev/fuse", *without[at:]]
    seen = fake_container.records(tmp_path / "records")["driver"]
    assert seen["binds"][-5:] == [*binds[1::2], "/dev/fuse:/dev/fuse"]
    # The runner launches its sandboxes through the harbor the plan pins, in the checkout.
    assert seen["env"]["PYTHONPATH"] == f"/opt/hls-src/src:/opt/hls-src:{deploy_drive.HARBOR_SRC}"
    assert (REPO / "harbor" / "src" / "harbor" / "environments" / "launch.py").is_file()


def test_no_container_but_the_driver_gets_a_store_or_the_device(tmp_path, bins, specs, fuse):
    stores = _stores(tmp_path)
    ctx = _ctx(tmp_path, bins, stores=stores, site=_site(tmp_path), sandbox_paths=_declared())
    DeployDriveStage().run(ctx)
    assert {"artifact", "seal-probe", "driver"} <= set(specs)
    roots = [Path(d) for d in stores.values()]
    for name, spec in specs.items():
        if name == "driver":
            continue
        assert spec.devices == (), name
        assert not [m for m in spec.mounts if any(m.source.is_relative_to(r) for r in roots)], name
        assert not [m for m in spec.mounts if m.target.startswith("/hls/sandboxes")], name
    records = fake_container.records(tmp_path / "records")
    for role in ("artifact", "seal-probe"):
        assert not [b for b in records[role]["binds"] if "sandbox-stores" in b or b.startswith("/dev/")], role
    assert set(b.split(":")[1] for b in records["artifact"]["binds"]) == {
        "/etc/resolv.conf", "/etc/hosts", "/data/weights", "/data/tenants/t0/dev", "/engine-cache"}


def test_a_store_that_is_no_directory_on_the_node_is_infra_not_retried_before_the_server_starts(tmp_path, bins,
                                                                                               fuse):
    stores = _stores(tmp_path)
    Path(stores["agentic-coding-images"]).rmdir()
    ctx = _ctx(tmp_path, bins, stores=stores, site=_site(tmp_path), sandbox_paths=_declared())
    with pytest.raises(StageFault) as e:
        DeployDriveStage().run(ctx)
    assert e.value.fault.fault_class == "infra" and not e.value.fault.retryable
    assert "/hls/sandboxes/images" in e.value.fault.reason and "no directory" in e.value.fault.reason
    assert "artifact" not in _started(ctx) and "driver" not in _started(ctx)


def test_a_node_without_the_device_is_infra_retried_before_the_server_starts(tmp_path, bins, monkeypatch):
    monkeypatch.setattr(deploy_drive, "is_device", lambda path: False)
    ctx = _ctx(tmp_path, bins, stores=_stores(tmp_path), site=_site(tmp_path), sandbox_paths=_declared())
    with pytest.raises(StageFault) as e:
        DeployDriveStage().run(ctx)
    assert e.value.fault.fault_class == "infra" and e.value.fault.retryable
    assert "/dev/fuse" in e.value.fault.reason
    assert "artifact" not in _started(ctx)


def test_the_driver_imports_the_checkouts_harbor_over_an_older_one_its_image_installed(tmp_path):
    # The driver's path as the stage gives it, with the checkout where the container binds it, and
    # after it an image's own site directory holding an older harbor, which must never be the one
    # imported, and its installed metadata, which harbor's import reads.
    site = tmp_path / "image-site"
    (site / "harbor").mkdir(parents=True)
    (site / "harbor" / "__init__.py").write_text('raise ImportError("the image\'s own harbor was imported")\n')
    (site / "harbor-0.1.0.dist-info").mkdir()
    (site / "harbor-0.1.0.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: harbor\nVersion: 0.1.0\n")
    checkout = deploy_drive.CHECKOUT
    path = f"{checkout}/src:{checkout}:{deploy_drive.HARBOR_SRC}".replace(checkout, str(REPO)) + f":{site}"
    probe = ("import dataclasses, harbor, harbor.environments.launch as launch, "
             "harbor.environments.singularity.singularity\n"
             "fields = {f.name for f in dataclasses.fields(launch.LaunchSpec)} if dataclasses.is_dataclass("
             "launch.LaunchSpec) else set(launch.LaunchSpec.model_fields)\n"
             "print(harbor.__file__, sorted(fields & {'server_python', 'start_in_staging_dir', "
             "'ready_poll_sec', 'overlay_dir', 'lower_overlay_dirs'}))")
    run = subprocess.run([sys.executable, "-c", probe], env={"PATH": "/usr/bin:/bin", "PYTHONPATH": path},
                         capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    where, fields = run.stdout.split(" ", 1)
    assert Path(where).is_relative_to(REPO / "harbor" / "src")
    assert fields.strip() == str(sorted(["lower_overlay_dirs", "overlay_dir", "ready_poll_sec", "server_python",
                                         "start_in_staging_dir"]))


@pytest.mark.parametrize("target, why", [
    ("store", r"the store root \(site\.store_root\)"),
    ("secrets", r"the secrets file's directory \(site\.secrets_path\)"),
    ("checkout", "the checkout"),
    ("agentic-coding-images", r"sandbox_stores\.agentic-coding-images and sandbox_stores\.agentic-coding-runtime "
                              "overlap"),
])
def test_a_store_that_reaches_the_store_the_secrets_the_checkout_or_another_store_on_the_node_is_refused(
        tmp_path, bins, fuse, target, why):
    # The plan saw a path that led nowhere yet; on the node it is a link to what no sandbox may
    # reach. The drive follows it, refuses before the server starts, and names no path.
    stores = _stores(tmp_path)
    runtime = Path(stores["agentic-coding-runtime"])
    runtime.rmdir()
    ctx = _ctx(tmp_path, bins, stores=stores, site=_site(tmp_path), sandbox_paths=_declared())
    leads_to = {"store": ctx.store.root, "secrets": tmp_path / "secrets", "checkout": REPO}.get(target)
    os.symlink(leads_to or stores[target], runtime)
    with pytest.raises(StageFault) as e:
        DeployDriveStage().run(ctx)
    assert e.value.fault.fault_class == "infra" and not e.value.fault.retryable
    assert re.search(why, e.value.fault.reason) and "agentic-coding-runtime" in e.value.fault.reason
    assert str(tmp_path) not in e.value.fault.reason
    assert "artifact" not in _started(ctx) and "driver" not in _started(ctx)


def test_the_device_check_reads_the_nodes_own_nodes(tmp_path):
    (tmp_path / "file").write_text("")
    assert is_device("/dev/null")
    assert not is_device(str(tmp_path / "file")) and not is_device(str(tmp_path / "missing"))


def test_a_contestant_container_is_given_no_device(tmp_path, bins):
    ctx = _ctx(tmp_path, bins)
    spec = ContainerSpec(image=stand_in(tmp_path / "image"), command=("true",), workdir=ctx.role_dir("x") / "wd",
                         devices=("/dev/fuse",))
    with pytest.raises(ValueError, match="only an unsealed harness container is given a device node; x names "
                                         "/dev/fuse"):
        ctx.start_container("x", spec, ARTIFACT_SHAPE, owner="contestant")


@pytest.mark.parametrize("changes, what", [
    ({"seal": SealPlan(policy=SealPolicy(network=deploy_drive.AIR_GAP), index=False, ingress_port=8000)}, "sealed"),
    ({"contestant_payload": True}, "a contestant payload"),
])
def test_a_sealed_container_or_a_contestant_payload_of_any_owner_is_given_no_device(tmp_path, changes, what):
    # A measured baseline's server is harness-owned and sealed as a contestant's is: only the
    # unsealed driver opens a device.
    with pytest.raises(ValueError, match=f"only an unsealed harness container is given a device node; this one is "
                                         f"{what} and names /dev/fuse"):
        ContainerSpec(image=stand_in(tmp_path / "image"), command=("true",), workdir=tmp_path / "wd",
                      devices=("/dev/fuse",), **changes)
    assert ContainerSpec(image=stand_in(tmp_path / "image"), command=("true",), workdir=tmp_path / "wd",
                         devices=("/dev/fuse",)).devices == ("/dev/fuse",)


# what a runner's host paths become


def test_each_store_is_bound_from_the_sites_directory_at_the_runners_path():
    stores = {name: Path(f"/site/{name}") for name in STORES}
    mounts, devices = sandbox_binds(PATHS, stores)
    assert mounts == tuple(Mount(Path(f"/site/{p.store}"), p.path, read_only=p.read_only)
                           for p in PATHS if p.store != DEVICE)
    assert devices == ("/dev/fuse",)
    assert sandbox_binds((), {}) == ((), ())


def test_every_store_the_site_does_not_name_is_named():
    with pytest.raises(ValueError, match="names no sandbox store agentic-coding-overlays, agentic-coding-runtime"):
        sandbox_binds(PATHS, {"agentic-coding-images": Path("/i"), "agentic-coding-prefixes": Path("/p")})


@pytest.mark.parametrize("path, why", [
    (HostPath("/dev/sda", read_only=False, store=DEVICE), "is not one a driver is given"),
    (HostPath("/dev/fuse", read_only=True, store=DEVICE), "declared read-only"),
    (HostPath("/hls/sandboxes/../out", read_only=True, store="s"), "normal form"),
    (HostPath("/hls/sandboxes/images/", read_only=True, store="s"), "normal form"),
    (HostPath("/hls/sandboxes", read_only=True, store="s"), "images meets /hls/sandboxes;"),
    (HostPath("/hls/sandboxes/images/x", read_only=True, store="s"), "images meets /hls/sandboxes/images/x"),
])
def test_a_host_path_the_driver_cannot_be_given_is_refused(path, why):
    with pytest.raises(ValueError, match=why):
        sandbox_binds((*PATHS, path), {name: Path(f"/site/{name}") for name in (*STORES, "s")})


@pytest.mark.parametrize("target", DRIVER_TARGETS)
def test_no_store_lies_at_under_or_over_the_drivers_own_mounts(target):
    for path in (target, f"{target}/x", str(Path(target).parent)):
        if path == "/":
            continue
        assert sandbox_problems((HostPath(path, read_only=True, store="s"),), beside=DRIVER_TARGETS), path
        with pytest.raises(ValueError, match="sandbox paths"):
            DeployDriveParams.model_validate(_params(sandbox_paths=[{"path": path, "read_only": True,
                                                                     "store": "s"}]))


def test_the_tasks_own_host_paths_meet_none_of_the_drivers_mounts():
    assert sandbox_problems(PATHS, beside=[t for t, _ in DATA + OWN]) == []


def test_a_drive_names_the_stores_its_driver_binds_each_once():
    again = {"path": "/hls/sandboxes/again", "read_only": True, "store": STORES[0]}
    params = DeployDriveParams.model_validate(_params(sandbox_paths=[*_declared(), again]))
    assert DeployDriveStage.sandbox_stores(params) == STORES
    assert DeployDriveStage.sandbox_stores(DeployDriveParams.model_validate(_params())) == ()


def test_a_drive_without_host_paths_dumps_no_such_field():
    params = DeployDriveParams.model_validate(_params())
    assert params.sandbox_paths == () and "sandbox_paths" not in params.model_dump(mode="json")
    declared = DeployDriveParams.model_validate(_params(sandbox_paths=_declared()))
    assert declared.model_dump(mode="json")["sandbox_paths"] == _declared()
    assert declared.sandbox_paths == PATHS
