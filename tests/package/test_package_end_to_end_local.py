"""``hls package`` on one machine: the command plans and submits, the in-process Slurm of
``tests/pipeline/test_end_to_end_local.py`` runs the real runner, the images stage finds the
submission base in the cache, and the package stage runs its real roles against the fake
pip, pasta and apptainer (exec, builds and extractions) of the package tests. The source_scan
stage reads the bundle for every secret value first, and what it found is published in its own
manifest whatever the package then does."""

from __future__ import annotations

import base64
import json
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hls.package.result import PackageResult
from hls.pipeline import commands
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.stage import StageContext
from hls.recipes import catalog
from hls.recipes.model import order
from hls.stages import package as package_stage
from hls.stages.images import IMAGE_FILE, image_keys
from hls.stages.source_scan import SourceScanStage
from hls.store.cache import ContentCache, entry_sha256
from hls.store.digest import sha256_file
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from tests.package import fake_tools
from tests.package.bundles import GOOD, make_bundle
from tests.package.test_stage import REPORT
from tests.pipeline.hls_cli import run_hls
from tests.pipeline.test_end_to_end_local import LocalSlurm
from hls.substrate.base import FaultRecord
from hls.substrate.secret_scan import PackageScan, PackageScanFinding
from hls.substrate.docker import substrate as docker_substrate
from hls.substrate.docker.launcher import USER_FILES_LABEL
from hls.substrate.secrets import Scrubber
from hls.substrate.slurm.scheduler import submit_argv
from hls.substrate.apptainer.images import ApptainerImages
from tests.pipeline.fakes import LocalLauncher
from tests.provision.warmer_config import SECRETS, TOKEN, write_config
from tests.recipes.warmer_site import local_slurm, site_config
from tests.substrate.docker.launcher_fakes import fake_world, launcher, play_gate
from tests.substrate.local.helpers import FakeInventory
from tests.store.entries import populate
from tests.substrate import fake_pasta


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """The fake apptainer, pip and pasta on PATH; returns the bin dir."""
    bin_dir, plan = tmp_path / "bin", tmp_path / "tools.json"
    fake_pasta.install(bin_dir)
    fake_tools.install(bin_dir, plan)
    plan.write_text(json.dumps({"dry_run": "ok", "wheel": "ok", "build": "ok", "report": REPORT,
                                "record": str(tmp_path / "tools.log")}))
    (tmp_path / "pasta.plan").write_text("pass\n" * 8)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    # The seal resolves its firewall tools from here, not from the system's /usr/sbin.
    monkeypatch.setenv("HLS_SEAL_TOOL_PATH", str(bin_dir))
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(tmp_path / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(tmp_path / "pasta.log"))
    monkeypatch.setenv("FAKE_PYTHON_COUNTER", str(tmp_path / "counter"))
    for knob in ("FAKE_APPTAINER_FAIL", "FAKE_EXTRACT_FAIL", "FAKE_PROXY_ERRORS", "FAKE_PROXY_MID_STREAM"):
        monkeypatch.delenv(knob, raising=False)
    return bin_dir


def _cache_the_submission_base(store: PosixStore) -> dict[str, str]:
    """Publish sample_task's base and layer as hls images build would have, the layer's file a
    tree image of the fake base, and return the keys."""
    keys = image_keys(order(catalog.task_images("sample_task").decls), ApptainerImages.recipe_key)
    cache = ContentCache(store)
    for name in ("s-a", "s-a-tf-5-13-1"):
        populate(cache, "images", keys[name],
                 lambda d: (d / IMAGE_FILE).write_bytes(fake_tools.tree_image(fake_tools.BASE_TREE)),
                 holder=Holder.here("test"))
    return keys


def _package(tmp_path: Path, bin_dir: Path, slurm: LocalSlurm, bundle: Path,
             digest: str | None = None) -> tuple[int, list[str]]:
    """``hls package tasks.sample_task.task --bundle <bundle> --sha256 <digest>``, the digest
    the bundle's own unless *digest* is given."""
    # The package stage scans the submission for every value of the site's secrets file.
    config = site_config(tmp_path, site=f'pasta_path = "{bin_dir / "pasta"}"', secrets=SECRETS)
    return run_hls(slurm, ["package", "tasks.sample_task.task", "--bundle", str(bundle),
                           "--sha256", digest or sha256_file(bundle)[0], "--job-name", "hls-package-test",
                           "--config", str(config)])


def _after_planning(monkeypatch, then) -> None:
    """Have *then*, given the plan, run once the command has planned and before it starts the
    instance."""
    start = commands.start

    def planned_then_start(plan, overrides, **kw):
        then(plan)
        return start(plan, overrides, **kw)

    monkeypatch.setattr(commands, "start", planned_then_start)


def _warmer_files(tmp_path: Path) -> list[Path]:
    """Every file under the store's warmers, the store's temporary files among them."""
    return sorted(p for p in (tmp_path / "store" / "warmers").rglob("*") if p.is_file())


def _package_events(store: PosixStore, instance: str, stage: str = "package") -> list[dict]:
    return read_events(store.path(InstanceLayout(instance).attempt_file(stage, 1, "events.jsonl")))


def _scan(store: PosixStore, instance: str, stage: str) -> PackageScan:
    """The ``package-scan.json`` *stage* of *instance* published."""
    manifest, _ = read_manifest(store, InstanceLayout(instance), stage)
    return PackageScan.model_validate_json(store.read_bytes(manifest.outputs["package_scan"].key))


def _roles(events: list[dict]) -> list[str]:
    return ["extract" if e["role"].startswith("extract-") else e["role"] for e in events if e["kind"] == "role_started"]


def _result(store: PosixStore, instance: str) -> PackageResult:
    manifest, _ = read_manifest(store, InstanceLayout(instance), "package")
    return PackageResult.model_validate_json(store.read_bytes(manifest.outputs["package"].key))


def test_hls_package_builds_a_good_bundle_and_publishes_an_invalid_one_as_a_result(tmp_path, login, tools):
    store = PosixStore(tmp_path / "store")
    keys = _cache_the_submission_base(store)
    slurm = local_slurm()
    good = make_bundle(tmp_path / "good", GOOD)
    code, lines = _package(tmp_path, tools, slurm, good)
    assert code == 0, lines
    instance = lines[0].removeprefix("instance ")
    assert instance.startswith("warmers/sample_task/")
    # Fifteen minutes of images, which finds the submission base's entries and fills none; the source
    # scan's 4,104 s, an hour for the checkout and scan of the bundle and the tree's own 504 s; then
    # the package's 25,200 s (the four hours both pip steps may take together, an hour for the
    # checkout and two hours for the rest); and the allocation's six minutes: 509.4 min of stages and
    # overhead, and 60 min for their 8, 2 and 14 roles' steps to start.
    assert lines[1] == ("allocation 0: images, source_scan, package; 0 x no GPU, 16 CPUs, 68 GiB, 569 min, "
                        "partition cpu-a")
    args = submit_argv(slurm.jobs["100"].request)
    assert "--partition=cpu-a" in args and "--gpus=0" in args
    result = _result(store, instance)
    assert result.status == "built" and result.task == "sample_task"
    # The digest recorded is the one the operator stated, which the checkout's copy was checked against.
    assert result.bundle_sha256 == sha256_file(good)[0]
    images = read_events(store.path(InstanceLayout(instance).attempt_file("images", 1, "events.jsonl")))
    assert [e["kind"] for e in images].count("cache_hit") == 2
    assert [e for e in images if e["kind"] == "role_started"] == []
    # The package stage pinned the base the images stage recorded, by its entry digest.
    base = ContentCache(store).lookup("images", keys["s-a-tf-5-13-1"])
    manifest, _ = read_manifest(store, InstanceLayout(instance), "package")
    assert manifest.provenance.image_digests["submission"] == result.sif_sha256
    assert manifest.provenance.cache_entries[f"images/{keys['s-a-tf-5-13-1']}"] == entry_sha256(base)
    integrity = json.loads(store.read_bytes(manifest.outputs["integrity"].key))
    assert (integrity["differences"], integrity["unattributed"]) == (0, 0) and "usr" in integrity["compared"]
    # The base is pinned before any other role, each pip step's writes go once it has ended,
    # every wheel is read whole before the build, and both trees the stage pinned go when it
    # closes.
    events = _package_events(store, instance)
    assert _roles(events) == ["extract", "checkout", "dry-run-copy", "dry-run", "dry-run-clean", "wheel-copy",
                              "wheel", "wheel-clean", "wheel-check", "build", "extract", "integrity", "install-scan",
                              "publish"]
    assert [e["kind"] for e in events].count("image_removed") == 2
    # The source scan read the bundle's objects and its plain tree, and the package what the
    # install added, for the secrets file's one value.
    assert _roles(_package_events(store, instance, "source_scan")) == ["checkout", "package-scan"]
    assert _scan(store, instance, "source_scan") == PackageScan(secrets=["hf_token"], forms=["value", "host"],
                                                                read=["bundle", "tree"], findings=[])
    assert _scan(store, instance, "package") == PackageScan(secrets=["hf_token"], forms=["value", "host"],
                                                            read=["install"], findings=[])

    # A submission without its server is the contestant's: the instance succeeds and its
    # package.json says why the submission is invalid.
    code, again = _package(tmp_path, tools, slurm, make_bundle(tmp_path / "bad", {"pyproject.toml": GOOD["pyproject.toml"]}))
    assert code == 0, again
    bad = _result(store, again[0].removeprefix("instance "))
    assert bad.status == "invalid_artifact" and "server.py" in bad.reason
    assert _roles(_package_events(store, again[0].removeprefix("instance "))) == ["extract", "checkout"]


def test_hls_package_reads_the_copy_it_made_and_never_the_operators_file(tmp_path, login, tools, monkeypatch):
    """Once the command has planned, the operator's file may be gone: the allocation reads the
    copy in the instance, checked against the stated digest, which package.json records."""
    store = PosixStore(tmp_path / "store")
    _cache_the_submission_base(store)
    bundle = make_bundle(tmp_path / "operator", GOOD)
    digest = sha256_file(bundle)[0]
    _after_planning(monkeypatch, lambda plan: bundle.unlink())
    code, lines = _package(tmp_path, tools, local_slurm(), bundle, digest=digest)
    assert code == 0, lines
    instance = lines[0].removeprefix("instance ")
    result = _result(store, instance)
    assert result.status == "built" and result.bundle_sha256 == digest
    assert sha256_file(store.path(InstanceLayout(instance).source_bundle))[0] == digest


def test_hls_package_refuses_a_bundle_that_is_not_the_stated_digest_before_anything_is_submitted(tmp_path, login,
                                                                                                 tools):
    """The operator named the wrong file or digest: the command refuses it as it copies it, in one
    line, before anything is planned or submitted, and the store keeps nothing of it."""
    _cache_the_submission_base(PosixStore(tmp_path / "store"))
    slurm = local_slurm()
    bundle = make_bundle(tmp_path, GOOD)
    code, lines = _package(tmp_path, tools, slurm, bundle, digest="ab" * 32)
    assert code == 1, lines
    assert lines[-1] == (f"hls package: --bundle {bundle} has sha256 {sha256_file(bundle)[0]}, not the {'ab' * 32} "
                         "--sha256 states")
    assert slurm.jobs == {} and _warmer_files(tmp_path) == []


def test_hls_package_refuses_a_copy_changed_after_planning_and_never_retries_it(tmp_path, login, tools, monkeypatch):
    """The instance's copy is not the bytes the plan records when the allocation reads it: the
    source scan's checkout checks it against the stated digest again and refuses it before git
    reads it, and a retry would read the same file, so the instance ends void after one attempt,
    nothing published and no package started."""
    store = PosixStore(tmp_path / "store")
    _cache_the_submission_base(store)

    def change(plan) -> None:
        copy = store.path(InstanceLayout(plan.instance).source_bundle)
        data = copy.read_bytes()
        copy.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))

    _after_planning(monkeypatch, change)
    code, lines = _package(tmp_path, tools, local_slurm(), make_bundle(tmp_path, GOOD))
    assert code == 1, lines
    instance = lines[0].removeprefix("instance ")
    assert lines[-1].startswith(f"{instance}: void: ") and "the source bundle is not the sha256 stated" in lines[-1]
    layout = InstanceLayout(instance)
    assert layout.attempts(store, "source_scan") == [1] and not store.exists(layout.manifest("source_scan"))
    assert layout.attempts(store, "package") == []
    assert _roles(_package_events(store, instance, "source_scan")) == ["checkout"]


def test_hls_package_refuses_a_file_past_the_bundle_limit_before_anything_is_submitted(tmp_path, login, tools,
                                                                                       monkeypatch):
    """The operator names a file past the build's writable disk cap, say an image file, with the
    digest of the bundle they meant: the command refuses it by its size, which it names beside
    the cap, before it copies any of it, so no allocation reads it and no ``over_limit`` result is
    ever recorded against a submission. The task's cap is 1 GiB here."""
    monkeypatch.setattr(commands, "sandbox_disk_gib", lambda task, producer: 1)
    _cache_the_submission_base(PosixStore(tmp_path / "store"))
    slurm = local_slurm()
    meant = make_bundle(tmp_path / "meant", GOOD)
    wrong = tmp_path / "image.sif"
    with open(wrong, "wb") as f:
        f.truncate(2**30 + 1)
    code, lines = _package(tmp_path, tools, slurm, wrong, digest=sha256_file(meant)[0])
    assert code == 1, lines
    assert lines[-1] == (f"hls package: --bundle {wrong} holds {2**30 + 1} bytes, more than the 1 GiB ({2**30} "
                         "bytes) the task's builds may write")
    assert slurm.jobs == {} and _warmer_files(tmp_path) == []


#: A bundle of ``GOOD``, its bytes fixed, so that what the package stage makes of it can be held
#: to what it made of the same bytes before ``hls package`` copied bundles into the store.
FIXED_BUNDLE = base64.b64decode("""IyB2MiBnaXQgYnVuZGxlCjljM2IxOGEwYzFkZjViNzJhNmFkNDcyNmUyNzIxMWJiY2E0YWQ0ZDkgcmVmcy9oZWFkcy9tYWlu
CgpQQUNLAAAAAgAAAASdB3icfcoxDsMgDADAnVd474IR2EGqon7FIbbCQJACkfL89gW9+ealCmbBC6H6KJiRjHYmUY4h7RYK
lcxs0XRxcs+jXyDwls+zAnJOETOnBC//40pvrc6pf4ob99bqGLWf7guytyLQrwR4nDM0MDAzMVEoqCwoys9KTS7RK8nPzWEo
2BAVGPHFMfnRg9fv3fSlJ/o2z5xmCFFZnFpUllqkV1DJ8IeB66ZJhe36vUfMtXI2XtXKevz7PgDCpSEqtwR4nIsuKMrPSk0u
ieXKS8xNVbBVUCouTcpV4ipLLSrOzM8DCRjoGSpxpaQWpOalpOYlZ6YWAwWjlTJKSgoq7GwN9IzMlWK5AEV4FiGxAXicKyjK
zCvRUC9OLSrLzEtX1+QCADobBdVahjN9h2o6BSFGhyfO/I/nCil4EA==""")
#: What the package stage published from ``FIXED_BUNDLE`` when it read the operator's file where
#: it lay: the sha256 of each output but two that name the instance itself, the submission image,
#: whose definition names the instance's role directories, and ``package.json``, which names that
#: image's key and digest; and ``package.json`` without those two.
BEFORE_OUTPUTS = {"source": "1546a79d7c318b791142826bdecab52e417c646f3f5567828f7ee3913c05a599",
                  "package_scan": "f75e8081626c63017f85093b34665ab9b017abba4c33f1abc27047ef9aa7d989",
                  "install_report": "d5f08a41bd5ac1e87cb50bfcef257a5a5b5c2aac605b2eee2162692aa6ad80ab",
                  "integrity": "dade22e1060d745b67e3eb3c8ea034458b565747709ea66e46504b53294b6c2a"}
BEFORE_PACKAGE = {"status": "built", "reason": None, "returncode": None, "log_tail": None, "served_ucounts": None,
                  "process_ceiling": {"hard": None, "reserve": 1024, "evaluations": 8, "evaluations_unknown": None,
                                      "uncapped": None, "nproc_max": 131072, "ceiling": 131072, "floor": 32768},
                  "task": "sample_task",
                  "bundle_sha256": "033286b4bca9e330902ddd36f4b37bb8c418110c4d6f599e726e48dff62b0adf",
                  "commit": "9c3b18a0c1df5b72a6ad4726e27211bbca4ad4d9", "wheels": ["httpx==0.27.0"]}


def test_the_package_of_a_bundle_is_what_it_was_before_the_bundle_was_copied(tmp_path, login, tools):
    store = PosixStore(tmp_path / "store")
    _cache_the_submission_base(store)
    bundle = tmp_path / "operator" / "source.bundle"
    bundle.parent.mkdir()
    bundle.write_bytes(FIXED_BUNDLE)
    assert sha256_file(bundle)[0] == BEFORE_PACKAGE["bundle_sha256"]
    code, lines = _package(tmp_path, tools, local_slurm(), bundle)
    assert code == 0, lines
    manifest, _ = read_manifest(store, InstanceLayout(lines[0].removeprefix("instance ")), "package")
    assert sorted(manifest.outputs) == sorted([*BEFORE_OUTPUTS, "package", "submission"])
    assert {name: ref.sha256 for name, ref in manifest.outputs.items()
            if name not in ("package", "submission")} == BEFORE_OUTPUTS
    package = json.loads(store.read_bytes(manifest.outputs["package"].key))
    assert {k: v for k, v in package.items() if k not in ("image", "sif_sha256")} == BEFORE_PACKAGE


def _docker_config(tmp_path: Path) -> Path:
    """An hls.toml of a Docker site with the warmers' store, scratch and secrets file
    (``write_config``)."""
    write_config(tmp_path, secrets=SECRETS)
    path = tmp_path / "docker.toml"
    path.write_text(textwrap.dedent(f"""
        [site]
        substrate = "docker"
        store_root = "{tmp_path / 'store'}"
        node_scratch = "{tmp_path / 'node'}"
        secrets_path = "{tmp_path / 'secrets.toml'}"
        [site.docker]
        buildkit_image = "moby/buildkit@sha256:{'b' * 64}"
        [site.local]
        state_dir = "{tmp_path / 'state'}"
        kept_memory_gib = 8
        [run]
        heartbeat_sec = 1
        lease_ttl_sec = 30
        poll_sec = 1
        allocation_overhead_sec = 360
        allocation_dead_after_sec = 10
        """))
    return path


def test_on_a_docker_site_the_checkout_reads_the_copy_through_the_store_its_container_binds(tmp_path, login,
                                                                                             monkeypatch):
    """``hls package --bundle`` plans a Docker site's instance as it plans a Slurm site's: the
    copy in the instance and the plan's record of it. The source scan's checkout role, as the
    stage starts it, reads the copy; the Docker launcher starts that same role in a container that
    sees the copy at its own path through the bind of the store root, and nothing the container is
    given names the operator's file. The host is a fake one large enough for the allocation, with
    the Docker CLI."""
    monkeypatch.setattr(docker_substrate, "SysInventory", lambda: FakeInventory(gpus=0, cores_per_node=16,
                                                                                memory_gib=256))
    monkeypatch.setattr(docker_substrate, "docker_cli_problem", lambda: None)
    bundle = make_bundle(tmp_path / "operator", GOOD)
    plan = commands.plan_package(config=_docker_config(tmp_path), overrides={}, task_module="tasks.sample_task.task",
                                 bundle=bundle, bundle_sha256=sha256_file(bundle)[0], job_name="hls-package",
                                 echo=[].append)
    assert plan.settings.site.substrate == "docker"
    key = InstanceLayout(plan.instance).source_bundle
    assert plan.stage("source_scan").params["bundle"] == {"key": key, "size": bundle.stat().st_size}
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("source_scan", 1, "events.jsonl")))
    ctx = StageContext(store=store, plan=plan, stage=plan.stage("source_scan"), attempt=1, launcher=LocalLauncher(),
                       watch=None, events=events, inputs={}, deadline=time.time() + 120,
                       scrubber=Scrubber({"hf_token": TOKEN}), cache=ContentCache(store),
                       secrets={"hf_token": TOKEN}, secret_registry=True, scratch=tmp_path / "node" / "hls-100",
                       job_id="100")
    assert sorted(SourceScanStage().run(ctx)) == ["package_scan", "source"]
    [checkout] = [s for s in ctx.launcher.launched if s.name == "checkout"]
    copy = str(store.path(key))
    assert checkout.argv[checkout.argv.index("--bundle") + 1] == copy
    (tmp_path / "docker").mkdir()
    with fake_world(tmp_path / "docker") as (daemon, client, host):
        handle = launcher(client, host, ctx.scratch, data_dirs=(plan.settings.site.store_root,)).launch(checkout)
        play_gate(daemon, host, handle, stay=True).join()
        [created] = [b for b in daemon.create_bodies
                     if b["Labels"]["hls.step"] == handle.step and USER_FILES_LABEL not in b["Labels"]]
        assert created["Entrypoint"][-len(checkout.argv):] == list(checkout.argv)
        assert any(m["Source"] == m["Target"] and copy.startswith(m["Source"] + "/")
                   for m in created["HostConfig"]["Mounts"])
        assert str(bundle) not in json.dumps(created)
        host.ended(handle.container_id)
        daemon.exit(handle.container_id, 0)
        assert handle.wait(10) == 0


def test_a_key_the_source_scan_found_stays_published_when_a_pip_step_ends_on_the_contestants_fault(
        tmp_path, login, tools, monkeypatch):
    """The wheel step ends on the contestant's own fault, so the package publishes no manifest and
    the instance fails; what the source scan found in the tree, a value in ``.env``, was published
    in its own manifest before the step ran, and stands."""
    store = PosixStore(tmp_path / "store")
    _cache_the_submission_base(store)
    decided = FaultRecord(fault_class="contestant", reason="contestant role wheel ran out of memory",
                          retryable=False, evidence={})
    real = package_stage.stage_fault
    monkeypatch.setattr(package_stage, "stage_fault",
                        lambda outcomes: decided if [o.name for o in outcomes] == ["wheel"] else real(outcomes))
    bundle = make_bundle(tmp_path, {**GOOD, ".env": f"HF_TOKEN={TOKEN}\n"})
    code, lines = _package(tmp_path, tools, local_slurm(), bundle)
    instance = lines[0].removeprefix("instance ")
    assert lines[-1].startswith(f"{instance}: failed: ") and "ran out of memory" in lines[-1], lines
    layout = InstanceLayout(instance)
    assert layout.attempts(store, "package") == [1] and not store.exists(layout.manifest("package"))
    assert "wheel" in _roles(_package_events(store, instance))
    assert _scan(store, instance, "source_scan") == PackageScan(
        secrets=["hf_token"], forms=["value", "host"], read=["bundle", "tree"],
        findings=[PackageScanFinding(path="/workspace/.env", secret="hf_token")],
        areas=[{"secret": "hf_token", "area": "bundle", "count": 1}])
