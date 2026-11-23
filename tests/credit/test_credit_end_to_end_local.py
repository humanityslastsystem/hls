"""``hls baseline-credit`` on one machine: the command plans and submits, the in-process Slurm of
``tests/pipeline/test_end_to_end_local.py`` runs the real runner, and every stage runs its real
roles: two drives of the measured baseline, each a fresh deploy of the deploy_drive tests' fake
server in a fake container image, each scored by the credit role, then the fold. Only srun, the
cluster, the hub, pasta and the container runtime are stood in for, with the stand-ins of the
``hls evaluate`` end to end. As in a real seal, each seal probe has an interceptor of its own:
an interceptor records a host its policy refused once, so one shared by two probes would leave
the second probe's seal with no record of its refusal. The drives run in one allocation, or with
``--fanout 2`` in two side by side, the fold's allocation submitted once both have published."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import hls.drive.readiness as readiness
from hls.baseline_credit import BaselineCredit, read_baseline_credit
from hls.measured import BASELINE_CREDIT_FILE, MeasuredFile, find, pinned_key, store_key
from hls.pipeline import commands
from hls.drive.record import DriveRecord
from hls.pipeline.events import read_events
from hls.pipeline.plan import PlanStore
from hls.recipes import catalog
from hls.recipes.model import order
from hls.stages import deploy_drive
from hls.stages.credit import CreditSummary
from hls.stages.images import IMAGE_FILE, image_keys
from hls.store.cache import ContentCache
from hls.store.digest import sha256_bytes
from hls.store.engine_cache import EngineCacheRecord
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.taskref import TaskRef, blob_id, load_task
from hls.substrate.apptainer.images import ApptainerImages
from tests.drive import fake_container
from tests.drive.interceptor import Interceptor
from tests.drive.test_stage import SERVER, _free_port
from tests.evaluate import fixture as evaluate
from tests.pipeline.hls_cli import run_hls
from tests.pipeline.test_end_to_end_local import LocalSlurm
from tests.provision import fake_hub
from tests.provision.warmer_config import SECRETS
from tests.recipes.warmer_site import site_config
from tests.store.entries import populate
from tests.substrate import fake_pasta

FAKE_ROLE = Path(fake_hub.__file__).with_name("fake_role.py")


@pytest.fixture
def bins(tmp_path, monkeypatch):
    """The fake container runtime and seal on PATH, the fake hub, fast readiness polls and no
    process ceiling, as the ``hls evaluate`` end to end sets them; returns the bin dir."""
    bin_dir = tmp_path / "bin"
    fake_container.install(bin_dir)
    fake_pasta.install(bin_dir)
    venv = Path(sys.executable).parent
    monkeypatch.setenv("PATH", f"{bin_dir}:{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CONTAINER_PATH", f"{venv}:/usr/bin:/bin")
    monkeypatch.setenv("HLS_SEAL_TOOL_PATH", str(bin_dir))
    (tmp_path / "pasta.plan").write_text("pass\n" * 16)
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(tmp_path / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(tmp_path / "pasta.log"))
    (tmp_path / "counter").write_text(str(_free_port() - 40001))
    monkeypatch.setenv("FAKE_PYTHON_COUNTER", str(tmp_path / "counter"))
    for var in ("FAKE_PROXY_ERRORS", "FAKE_APPTAINER_FAIL", "FAKE_TREE_GIB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FAKE_CONTAINER_RECORDS", str(tmp_path / "records"))
    hub = tmp_path / "hub"
    fake_hub.make_hub(hub, "org/model", evaluate.SHA, evaluate.FILES)
    monkeypatch.setenv("FAKE_HUB", str(hub))
    monkeypatch.setattr(readiness, "HEALTH_POLL_SEC", 0.1)
    monkeypatch.setattr(readiness, "INFERENCE_POLL_SEC", 0.1)
    monkeypatch.setattr(deploy_drive, "PORT_POLL_SEC", 0.05)
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: None)
    return bin_dir


@pytest.fixture
def interceptors(monkeypatch) -> list[Interceptor]:
    """Each seal probe's own interceptor, as a real seal starts one for each sealed role: started
    when the drive stage runs its probe, writing the status file of that probe's seal, dialled
    by the probe's refusal check, and closed when the probe ends. Returns the interceptors, in
    the order the probes ran."""
    started: list[Interceptor] = []
    probe_seal = deploy_drive._Drive.probe_seal

    def own_interceptor(drive, harness, seal, policy):
        status = drive.ctx.role_dir(deploy_drive.SEAL_PROBE) / "seal" / "status.json"
        interceptor = Interceptor(status, policy)
        started.append(interceptor)
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(deploy_drive, "PROBE_DIAL", interceptor.address)
                return probe_seal(drive, harness, seal, policy)
        finally:
            interceptor.close()

    monkeypatch.setattr(deploy_drive._Drive, "probe_seal", own_interceptor)
    return started


def _slurm(tmp_path: Path) -> LocalSlurm:
    """An in-process Slurm whose jobs run as soon as they are submitted, on a node with exactly
    the disk each asked for and a fake job cgroup, the provision role served by the fake hub."""
    slurm = LocalSlurm(replace_modules={"hls.provision.role": FAKE_ROLE}, disk_free_gib=None, node_disk=True,
                       job_cgroups=tmp_path / "cgroups")

    def submit_and_run(request):
        job_id = LocalSlurm.submit(slurm, request)
        slurm.tick()
        return job_id

    slurm.submit = submit_and_run
    return slurm


def _cache_images(store: PosixStore, baseline: bytes) -> None:
    """The harness chain and the measured baseline in the cache, as ``hls images build`` would
    have left them: the harness layer is an image the fake container runtime can run."""
    keys = image_keys(order(catalog.task_images(evaluate.TASK_ID).decls), ApptainerImages.recipe_key)
    cache, holder = ContentCache(store), Holder.here("test")
    layer = catalog.image_set("sample_task").layer
    populate(cache, "images", keys["s-a"], lambda d: (d / IMAGE_FILE).write_bytes(b"s-a image"), holder=holder)
    populate(cache, "images", keys[layer], lambda d: fake_container.make_image(
        d / IMAGE_FILE, {".singularity.d/labels.json": (b"{}", 0o644)}), holder=holder)
    populate(cache, "images", keys["evaluate-task-baseline"], lambda d: (d / IMAGE_FILE).write_bytes(baseline),
             holder=holder)


def _output(store: PosixStore, layout: InstanceLayout, stage: str, name: str) -> bytes:
    manifest, _ = read_manifest(store, layout, stage)
    return store.read_bytes(manifest.outputs[name].key)


def _credit(tmp_path: Path, bins: Path, *, fanout: int, task_module: str = "cr_task.task",
            drives: int | None = 2) -> tuple[int, list[str], PosixStore, LocalSlurm, str]:
    """``hls provision`` of the task, which the plan sizes the drives' copies of its data by, then
    ``hls baseline-credit <task_module> --drives <drives> --fanout <fanout>``, with no ``--drives``
    when *drives* is None, over the fake server's image, imported as the measured baseline: the
    exit status, what it said, the store, the in-process Slurm and the image's digest."""
    store = PosixStore(tmp_path / "store")
    baseline = fake_container.make_image(tmp_path / "baseline.sif", {"workspace/server.py": (SERVER.read_bytes(),
                                                                                             0o644)}).read_bytes()
    digest = sha256_bytes(baseline)
    _cache_images(store, baseline)
    slurm = _slurm(tmp_path)
    config = site_config(tmp_path, site=f'pasta_path = "{bins / "pasta"}"', secrets=SECRETS)
    code, provisioned = run_hls(_slurm(tmp_path), ["provision", task_module, "--job-name", "hls-provision-test",
                                                   "--config", str(config)])
    assert code == 0, provisioned
    told = ["--drives", str(drives)] if drives is not None else []
    code, lines = run_hls(slurm, ["baseline-credit", task_module, *told, "--fanout", str(fanout),
                                  "--job-name", "hls-credit-test", "--config", str(config)])
    return code, lines, store, slurm, digest


def _check_credit(store: PosixStore, lines: list[str], digest: str) -> InstanceLayout:
    """Check what a succeeded credit of two drives published, wrote into the store's measured area
    and said; return its layout."""
    instance = lines[0].removeprefix("instance ")
    assert instance.startswith(f"credits/{evaluate.TASK_ID}/")
    assert f"{instance}: succeeded: every stage published" in lines
    layout = InstanceLayout(instance)
    # The warm stage built the baseline's engine cache once and published it, and each drive's
    # server started from a copy of that entry, which its record names.
    warm = EngineCacheRecord.model_validate_json(_output(store, layout, "warm_baseline", "engine_cache"))
    assert warm.published and warm.entry is not None
    for k in (1, 2):
        drive = DriveRecord.model_validate_json(_output(store, layout, f"drive_{k}", "drive"))
        # Each drive served the measured file, from a deploy of its own.
        assert (drive.outcome, drive.trace.events, drive.artifact.image_sha256) == ("served", 4, digest)
        assert drive.artifact.engine_cache == EngineCacheRecord(key=warm.key, seeded_from=warm.entry)
    data = _output(store, layout, "fold", "baseline")
    measured = BaselineCredit.model_validate_json(data)
    assert [d.n_units for d in measured.drives] == [4, 4]
    assert all(d.mean_log_credit > 0 for d in measured.drives)
    # The credit records the hardware its drives ran on.
    assert measured.hardware is not None
    summary = CreditSummary.model_validate_json(_output(store, layout, "fold", "credit"))
    key = store_key(evaluate.TASK_ID, BASELINE_CREDIT_FILE)
    assert (summary.drives, summary.image_sha256, summary.measured_key) == (2, digest, key)
    # The fold wrote it into the store's measured area too, byte for byte, where a run of the task
    # finds it and the task loads it.
    assert store.read_bytes(key) == data
    assert read_baseline_credit(store.path(key), load_task(TaskRef(module="cr_task.task")))[0] == measured
    manifest, _ = read_manifest(store, layout, "fold")
    said = [line for line in lines if line.startswith("baseline credit: B = ")]
    assert len(said) == 1 and said[0].endswith(
        f", in {manifest.outputs['baseline'].key}; the store's {key} holds it, and runs of the task use it")
    return layout


def test_hls_baseline_credit_end_to_end_local(tmp_path, login, in_catalog, bins, interceptors):
    code, lines, store, slurm, digest = _credit(tmp_path, bins, fanout=1)
    assert code == 0, "\n".join(lines)
    _check_credit(store, lines, digest)
    # One allocation; each drive's probe had its seal's own interceptor.
    assert len(slurm.jobs) == 1 and len(interceptors) == 2


def test_a_credit_takes_one_drive_unless_told_more_and_says_it_has_no_spread(tmp_path, login, in_catalog, bins,
                                                                              interceptors):
    """``hls baseline-credit`` with no ``--drives`` drives the baseline once; its credit records
    that one drive, says plainly that one drive has no spread between drives, and is the file the
    store's measured area holds for the task, which a run of the task then finds before anything
    else."""
    code, lines, store, slurm, digest = _credit(tmp_path, bins, fanout=1, drives=None)
    assert code == 0, "\n".join(lines)
    key = store_key(evaluate.TASK_ID, BASELINE_CREDIT_FILE)
    one = BaselineCredit.model_validate_json(store.read_bytes(key))
    assert len(one.drives) == 1 and (one.log_credit_sd, one.log_credit_se) == (None, None)
    summary = CreditSummary.model_validate_json(_output(store, InstanceLayout(lines[0].removeprefix("instance ")),
                                                        "fold", "credit"))
    assert (summary.drives, summary.log_credit_sd, summary.log_credit_se) == (1, None, None)
    [said] = [line for line in lines if line.startswith("baseline credit: B = ")]
    assert f"B = {one.log_credit_mean:.6f} (1 drive, so no spread between drives) measured on " in said
    ref = TaskRef(module="cr_task.task")
    found = find(BASELINE_CREDIT_FILE, task_ref=ref, task=load_task(ref), task_id=evaluate.TASK_ID,
                 checkout=commands.checkout_root(), store=store)
    blob = blob_id(store.read_bytes(key))
    assert found.pin == MeasuredFile(source="store", path=pinned_key(evaluate.TASK_ID, BASELINE_CREDIT_FILE, blob),
                                     blob=blob)


def test_a_second_credit_finds_the_engine_cache_the_first_built(tmp_path, login, in_catalog, bins, interceptors):
    """The baseline's engine cache is built once per image and build environment: a second credit
    of the task finds the first one's entry, starts no server to build it, and seeds its drives
    with it."""
    code, lines, store, slurm, digest = _credit(tmp_path, bins, fanout=1)
    assert code == 0, "\n".join(lines)
    first = EngineCacheRecord.model_validate_json(
        _output(store, InstanceLayout(lines[0].removeprefix("instance ")), "warm_baseline", "engine_cache"))
    config = site_config(tmp_path, site=f'pasta_path = "{bins / "pasta"}"', secrets=SECRETS)
    code, again = run_hls(slurm, ["baseline-credit", "cr_task.task", "--drives", "2", "--job-name", "hls-credit-test",
                                  "--config", str(config)])
    assert code == 0, "\n".join(again)
    layout = InstanceLayout(again[0].removeprefix("instance "))
    warm = EngineCacheRecord.model_validate_json(_output(store, layout, "warm_baseline", "engine_cache"))
    assert warm == EngineCacheRecord(key=first.key, seeded_from=None, entry=first.entry)
    events = read_events(store.path(layout.attempt_file("warm_baseline", 1, "events.jsonl")))
    assert not [e for e in events if e["kind"] == "role_started"]
    for k in (1, 2):
        drive = DriveRecord.model_validate_json(_output(store, layout, f"drive_{k}", "drive"))
        assert drive.artifact.engine_cache == EngineCacheRecord(key=first.key, seeded_from=first.entry)


def test_hls_baseline_credit_with_fanout_end_to_end_local(tmp_path, login, in_catalog, bins, interceptors):
    code, lines, store, slurm, digest = _credit(tmp_path, bins, fanout=2)
    assert code == 0, "\n".join(lines)
    layout = _check_credit(store, lines, digest)
    # Provision and images, then a drive in each of two allocations, then the fold's, submitted
    # only once both drive allocations had published.
    plan = PlanStore(store, layout).latest()
    assert [a.stages for a in plan.allocations] == [["provision", "images", "warm_baseline"], ["drive_1", "credit_1"],
                                                    ["drive_2", "credit_2"], ["fold"]]
    submitted = [a.submissions[0].intended_at for a in plan.allocations]
    published = [read_manifest(store, layout, s)[0].times.ended_at for s in ("credit_1", "credit_2")]
    assert submitted[3] >= max(published)
    assert len(slurm.jobs) == 4 and len(interceptors) == 2


def test_a_credit_whose_baseline_does_not_serve_builds_no_engine_cache_drives_nothing_and_exits_1(
        tmp_path, login, in_catalog, bins, interceptors):
    """A baseline that exits before it is ready builds no engine cache: its warm stage is infra,
    retried until its retries are spent, so the instance ends void, no drive runs, no fold runs and
    no credit is said."""
    code, lines, store, slurm, digest = _credit(tmp_path, bins, fanout=1, task_module="cr_crash.task")
    assert code == 1, "\n".join(lines)
    instance = lines[0].removeprefix("instance ")
    layout = InstanceLayout(instance)
    plan = PlanStore(store, layout).latest()
    assert (plan.outcome.status, plan.outcome.fault.fault_class) == ("void", "infra")
    [ended] = [line for line in lines if line.startswith(f"{instance}: ")]
    assert ended == (f"{instance}: void: warm_baseline: harness role server ended exited with status 3; retries "
                     "exhausted (2 of 2)")
    assert all(read_manifest(store, layout, s) is None for s in ("warm_baseline", "drive_1", "drive_2", "fold"))
    assert not any(line.startswith("baseline credit") for line in lines)
    assert not store.exists(store_key(evaluate.TASK_ID, BASELINE_CREDIT_FILE))
