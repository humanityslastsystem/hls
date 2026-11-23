"""A task's sandbox images: the store's record of each, the role program that pulls them, the
stage that runs it and the plan of ``hls provision`` that carries the stage. A stand-in answers
for apptainer, so nothing here reaches a registry."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hls.pipeline import commands
from hls.pipeline.events import EventLog
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, SANDBOX_IMAGES_STAGE, provision_stages, sandbox_images_stage
from hls.pipeline.plan import Plan
from hls.pipeline.planner import PlanError
from hls.pipeline.stage import STAGES, StageContext, StageFault, run_stage
from hls.provision import sandbox_images as si
from hls.provision.sandbox_images import (FAILED_EXIT, ImageRecord, Outcome, Report, SandboxImage, SandboxImageSet,
                                          failure_line, held, image_file, missing_images, pull_argv, pull_env,
                                          pull_missing, read_record, read_report, record_file, write_report)
from hls.spec.resources import ResourceShape
from hls.stages.sandbox_images import (RECORD_FILE, ROLE, ROLE_MODULE, SandboxImagesParams, SandboxImagesRecord,
                                       SandboxImagesStage, role_argv)
from hls.store.digest import sha256_bytes
from hls.store.posix import PosixStore
from hls.substrate.secrets import SecretsFile
from hls.taskref import TaskRef
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan
from tests.provision.sandbox_images_task import DIGESTS, SANDBOX_IMAGES, STORE
from tests.provision.warmer_config import sandbox_store, write_config

MODULE = "tests.provision.sandbox_images_task"
#: A stand-in for apptainer. ``--version`` prints one. ``pull OUT SOURCE`` appends its arguments
#: and what it sees of its environment to ``$FAKE_APPTAINER_LOG`` and writes an image file that
#: names its source. An image named in ``$FAKE_APPTAINER_FAIL`` fails as a registry refusal does;
#: one named in ``$FAKE_APPTAINER_HANG`` never ends; and with ``$FAKE_APPTAINER_MEET`` a pull
#: waits until a second pull is in flight beside it.
STAND_IN = r"""#!/bin/sh
if [ "$1" = "--version" ]; then echo "apptainer version 1.5.3"; exit 0; fi
[ "$1" = "pull" ] && [ "$#" -eq 3 ] || { echo "unexpected arguments: $*" >&2; exit 2; }
name=$(basename "$2" .sif)
echo "pull $2 $3 cache=$APPTAINER_CACHEDIR tmp=$APPTAINER_TMPDIR user=${APPTAINER_DOCKER_USERNAME-} \
password=${APPTAINER_DOCKER_PASSWORD-} role_user=${REGISTRY_USERNAME-} role_token=${REGISTRY_PASSWORD-}" \
    >> "$FAKE_APPTAINER_LOG"
[ -d "$APPTAINER_CACHEDIR" ] && [ -d "$APPTAINER_TMPDIR" ] || { echo "no cache or tmp directory" >&2; exit 4; }
case " ${FAKE_APPTAINER_FAIL-} " in *" $name "*)
    echo "INFO:    Converting OCI blobs to SIF format" >&2
    echo "FATAL:   While making image from oci registry: error fetching image: manifest unknown" >&2
    exit 255;;
esac
case " ${FAKE_APPTAINER_HANG-} " in *" $name "*) printf 'partial' > "$2"; sleep 600;; esac
if [ -n "${FAKE_APPTAINER_MEET-}" ]; then
    : > "$FAKE_APPTAINER_MEET/$name"
    tries=0
    while [ "$(ls "$FAKE_APPTAINER_MEET" | wc -l)" -lt 2 ]; do
        tries=$((tries + 1)); [ "$tries" -lt 200 ] || { echo "FATAL:   no second pull in flight" >&2; exit 3; }
        sleep 0.1
    done
fi
echo "INFO:    Creating SIF file..." >&2
printf 'image of %s\n' "$3" > "$2"
"""
VERSION = "apptainer version 1.5.3"


def _source(name: str) -> str:
    return f"docker://registry.example/bench@{DIGESTS[name]}"


def _content(name: str) -> bytes:
    """What the stand-in writes as image *name*'s file."""
    return f"image of {_source(name)}\n".encode()


@pytest.fixture
def apptainer(tmp_path, monkeypatch) -> Path:
    """The stand-in, first on PATH as ``apptainer``, with its log at ``tmp_path/apptainer.log``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    path = bin_dir / "apptainer"
    path.write_text(STAND_IN)
    path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_APPTAINER_LOG", str(tmp_path / "apptainer.log"))
    for name in ("FAKE_APPTAINER_FAIL", "FAKE_APPTAINER_HANG", "FAKE_APPTAINER_MEET", "REGISTRY_USERNAME",
                 "REGISTRY_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    return path


def _pulls(tmp_path: Path) -> list[str]:
    """Each pull the stand-in was asked for, by image name, in order."""
    log = tmp_path / "apptainer.log"
    return [Path(line.split()[1]).stem for line in log.read_text().splitlines()] if log.exists() else []


def _store_names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# ---- the record and what a store holds


def test_a_pulled_image_is_in_the_store_with_a_record_that_round_trips(tmp_path, apptainer):
    store = tmp_path / "images"
    report = pull_missing(SANDBOX_IMAGES, store, tmp_path / "work")
    assert [(o.name, o.status) for o in report.outcomes] == [(n, "pulled") for n in DIGESTS]
    assert _store_names(store) == sorted([f"{n}.sif" for n in DIGESTS] + [f"{n}.json" for n in DIGESTS])
    for outcome in report.outcomes:
        name, record = outcome.name, outcome.record
        # The record names the published manifest by digest, and the file by the bytes that landed.
        assert record == ImageRecord(name=name, source=_source(name), sha256=sha256_bytes(_content(name)),
                                     size=len(_content(name)), apptainer=VERSION)
        assert image_file(store, name).read_bytes() == _content(name)
        on_disk = record_file(store, name).read_bytes()
        assert ImageRecord.model_validate_json(on_disk) == record == read_record(store, name)
        assert ImageRecord.model_validate_json(record.model_dump_json()) == record
        assert json.loads(on_disk)["source"].endswith("@" + DIGESTS[name])
        assert held(store, name, _source(name)) == record
    assert missing_images(store, SANDBOX_IMAGES) == []
    # Nothing is left on the node, and nothing but the images and their records in the store.
    assert list((tmp_path / "work").iterdir()) == []


def test_a_store_holds_an_image_only_with_the_record_of_its_declared_source_and_a_file_of_that_size(tmp_path,
                                                                                                    apptainer):
    store = tmp_path / "images"
    pull_missing(SANDBOX_IMAGES, store, tmp_path / "work")
    assert missing_images(store, SANDBOX_IMAGES) == []
    # Another digest is another image.
    moved = SANDBOX_IMAGES.model_copy(update={"images": (
        SandboxImage(name="alpha", digest="sha256:" + "0f" * 32), *SANDBOX_IMAGES.images[1:])})
    assert missing_images(store, moved) == ["alpha"]
    # A file of another size is not the recorded one.
    with open(image_file(store, "beta"), "ab") as f:
        f.write(b"x")
    assert missing_images(store, SANDBOX_IMAGES) == ["beta"]
    # A file with no record says nothing of what it is, and a record with no file holds nothing.
    record_file(store, "alpha").unlink()
    image_file(store, "gamma").unlink()
    assert missing_images(store, SANDBOX_IMAGES) == ["alpha", "beta", "gamma"]
    assert missing_images(store, SANDBOX_IMAGES, names={"gamma", "alpha"}) == ["alpha", "gamma"]
    assert missing_images(tmp_path / "nowhere", SANDBOX_IMAGES) == list(DIGESTS)


def test_a_record_that_names_another_image_or_cannot_be_read_holds_nothing(tmp_path, apptainer):
    store = tmp_path / "images"
    pull_missing(SANDBOX_IMAGES, store, tmp_path / "work")
    record_file(store, "alpha").write_bytes(record_file(store, "beta").read_bytes())
    assert held(store, "alpha", _source("alpha")) is None and read_record(store, "alpha").name == "beta"
    record_file(store, "beta").write_text("{not json")
    assert read_record(store, "beta") is None and held(store, "beta", _source("beta")) is None
    # A link in the file's place is not a file of the store.
    target = tmp_path / "elsewhere.sif"
    target.write_bytes(_content("gamma"))
    image_file(store, "gamma").unlink()
    image_file(store, "gamma").symlink_to(target)
    assert held(store, "gamma", _source("gamma")) is None


def test_a_set_refuses_a_name_twice_a_digest_that_is_none_and_a_name_that_is_no_file_name():
    base = SANDBOX_IMAGES.model_dump()
    with pytest.raises(ValueError, match="named twice: alpha"):
        SandboxImageSet.model_validate({**base, "images": [base["images"][0], base["images"][0]]})
    with pytest.raises(ValueError, match="not a manifest digest"):
        SandboxImage(name="alpha", digest="latest")
    with pytest.raises(ValueError, match="invalid store key segment"):
        SandboxImage(name="../alpha", digest=DIGESTS["alpha"])
    with pytest.raises(ValueError, match="at least one image"):
        SandboxImageSet.model_validate({**base, "images": []})
    assert SandboxImageSet.model_validate_json(SANDBOX_IMAGES.model_dump_json()) == SANDBOX_IMAGES


# ---- the role program


def test_the_command_for_one_image_pulls_its_published_manifest_by_digest_into_its_own_directory(tmp_path):
    out = Path("/node/pulls/alpha/alpha.sif")
    assert pull_argv(out, SANDBOX_IMAGES.source(SANDBOX_IMAGES.images[0])) == [
        "apptainer", "pull", "/node/pulls/alpha/alpha.sif",
        "docker://registry.example/bench@sha256:" + "a1" * 32]
    with pytest.raises(ValueError, match="absolute"):
        pull_argv(Path("alpha.sif"), _source("alpha"))
    # The layer cache and the unpacking directory are the pull's own.
    env = pull_env(Path("/node/pulls/alpha"), {"PATH": "/usr/bin", "HOME": "/home/u"})
    assert env == {"PATH": "/usr/bin", "HOME": "/home/u", "APPTAINER_CACHEDIR": "/node/pulls/alpha/cache",
                   "APPTAINER_TMPDIR": "/node/pulls/alpha/tmp"}


def test_each_pull_works_in_its_own_directory_and_two_are_in_flight_at_once(tmp_path, apptainer, monkeypatch):
    meet = tmp_path / "meet"
    meet.mkdir()
    monkeypatch.setenv("FAKE_APPTAINER_MEET", str(meet))
    work = tmp_path / "work"
    report = pull_missing(SANDBOX_IMAGES, tmp_path / "images", work)
    # A pull ends only once a second one runs beside it, so none would end one at a time.
    assert [o.status for o in report.outcomes] == ["pulled"] * 3
    lines = (tmp_path / "apptainer.log").read_text().splitlines()
    assert sorted(line.split()[1:5] for line in lines) == sorted(
        [str(work / n / f"{n}.sif"), _source(n), f"cache={work / n / 'cache'}", f"tmp={work / n / 'tmp'}"]
        for n in DIGESTS)


def test_a_second_run_pulls_nothing_and_a_run_after_one_went_missing_pulls_that_one(tmp_path, apptainer):
    store = tmp_path / "images"
    pull_missing(SANDBOX_IMAGES, store, tmp_path / "work")
    assert sorted(_pulls(tmp_path)) == sorted(DIGESTS)
    before = {n: os.stat(image_file(store, n)).st_mtime_ns for n in DIGESTS}
    again = pull_missing(SANDBOX_IMAGES, store, tmp_path / "work")
    assert [(o.status, o.seconds) for o in again.outcomes] == [("found", 0.0)] * 3
    assert len(_pulls(tmp_path)) == 3
    assert {n: os.stat(image_file(store, n)).st_mtime_ns for n in DIGESTS} == before
    image_file(store, "beta").unlink()
    third = pull_missing(SANDBOX_IMAGES, store, tmp_path / "work")
    assert [o.status for o in third.outcomes] == ["found", "pulled", "found"]
    assert _pulls(tmp_path)[3:] == ["beta"]


def test_a_failed_pull_is_named_with_its_reason_and_leaves_nothing_of_that_image_in_the_store(tmp_path, apptainer,
                                                                                            monkeypatch, capsys):
    monkeypatch.setenv("FAKE_APPTAINER_FAIL", "beta")
    store, report_path, declared = tmp_path / "images", tmp_path / "report.json", tmp_path / "images.json"
    declared.write_text(SANDBOX_IMAGES.model_dump_json())
    code = si.main(["--images", str(declared), "--store", str(store), "--work", str(tmp_path / "work"),
                    "--report", str(report_path)])
    why = ("apptainer pull exited 255: FATAL: While making image from oci registry: error fetching image: "
           "manifest unknown")
    assert code == FAILED_EXIT
    # The role's last line names the image and why; the pull's own output is above it.
    err = capsys.readouterr().err
    assert err.splitlines()[-1] == f"failed: beta: {why}"
    assert "== beta: apptainer pull exited 255" in err and "== alpha: apptainer pull exited 0" in err
    report = read_report(report_path)
    assert [(o.name, o.status, o.why) for o in report.outcomes] == [
        ("alpha", "pulled", None), ("beta", "failed", why), ("gamma", "pulled", None)]
    assert failure_line(report.failed()) == f"beta: {why}"
    # The others are whole and recorded; the failed one has no file, no record and no part.
    assert _store_names(store) == ["alpha.json", "alpha.sif", "gamma.json", "gamma.sif"]
    assert missing_images(store, SANDBOX_IMAGES) == ["beta"]
    # The run after pulls only the one that failed.
    monkeypatch.delenv("FAKE_APPTAINER_FAIL")
    assert si.main(["--images", str(declared), "--store", str(store), "--work", str(tmp_path / "work"),
                    "--report", str(report_path)]) == 0
    assert _pulls(tmp_path)[3:] == ["beta"]
    assert [o.status for o in read_report(report_path).outcomes] == ["found", "pulled", "found"]


def test_a_pull_that_does_not_end_is_ended_and_named_and_the_others_go_on(tmp_path, apptainer, monkeypatch):
    monkeypatch.setenv("FAKE_APPTAINER_HANG", "alpha")
    images = SANDBOX_IMAGES.model_copy(update={"pull_timeout_sec": 2})
    store = tmp_path / "images"
    started = time.monotonic()
    report = pull_missing(images, store, tmp_path / "work")
    assert time.monotonic() - started < 60
    assert [(o.name, o.status, o.why) for o in report.outcomes] == [
        ("alpha", "failed", "apptainer pull did not end in 2 s"), ("beta", "pulled", None), ("gamma", "pulled", None)]
    # The part the pull had written never reached the store.
    assert _store_names(store) == ["beta.json", "beta.sif", "gamma.json", "gamma.sif"]
    assert list((tmp_path / "work").iterdir()) == []


def test_a_role_told_to_stop_ends_its_pulls_and_reports_them_as_failed(tmp_path, apptainer, monkeypatch):
    monkeypatch.setenv("FAKE_APPTAINER_HANG", "alpha beta gamma")
    declared, report_path = tmp_path / "images.json", tmp_path / "report.json"
    declared.write_text(SANDBOX_IMAGES.model_dump_json())
    role = subprocess.Popen([sys.executable, "-m", "hls.provision.sandbox_images", "--images", str(declared),
                             "--store", str(tmp_path / "images"), "--work", str(tmp_path / "work"),
                             "--report", str(report_path)], stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 60
        while len(_pulls(tmp_path)) < 2 and time.monotonic() < deadline:
            time.sleep(0.1)
        assert len(_pulls(tmp_path)) == 2
        role.send_signal(signal.SIGTERM)
        _, err = role.communicate(timeout=60)
    finally:
        role.kill()
    assert role.returncode == FAILED_EXIT and err.splitlines()[-1].startswith("failed: alpha: the role was told to stop")
    report = read_report(report_path)
    assert [o.status for o in report.outcomes] == ["failed"] * 3
    assert {o.why for o in report.outcomes} == {"the role was told to stop while the pull ran",
                                                "the role was told to stop before the pull started"}
    # Two pulls were started and no third, and the store holds no part of any.
    assert len(_pulls(tmp_path)) == 2 and _store_names(tmp_path / "images") == []


def test_an_image_file_replaces_an_unrecorded_one_and_its_record_comes_last(tmp_path, apptainer):
    store = tmp_path / "images"
    store.mkdir()
    # A file someone left with no record is not held, so it is pulled over.
    image_file(store, "alpha").write_bytes(b"something else")
    report = pull_missing(SANDBOX_IMAGES, store, tmp_path / "work")
    assert report.outcomes[0].status == "pulled" and image_file(store, "alpha").read_bytes() == _content("alpha")
    assert not [p for p in store.iterdir() if p.name.startswith(".")]
    assert stat.S_ISREG(os.lstat(image_file(store, "alpha")).st_mode)


def test_a_registry_login_reaches_apptainer_under_its_own_names_and_no_command_line(tmp_path, apptainer,
                                                                                    monkeypatch):
    monkeypatch.setenv("REGISTRY_USERNAME", "site-user")
    monkeypatch.setenv("REGISTRY_PASSWORD", "registry-token-0123456789")
    pull_missing(SANDBOX_IMAGES, tmp_path / "images", tmp_path / "work")
    for line in (tmp_path / "apptainer.log").read_text().splitlines():
        words = line.split()
        assert words[5:] == ["user=site-user", "password=registry-token-0123456789", "role_user=", "role_token="]
        assert "registry-token-0123456789" not in " ".join(words[:5])
    # One half of a login is no login.
    assert "APPTAINER_DOCKER_USERNAME" not in pull_env(Path("/w"), {"REGISTRY_USERNAME": "site-user"})
    assert pull_env(Path("/w"), {"REGISTRY_USERNAME": "u", "REGISTRY_PASSWORD": "t"}) == {
        "APPTAINER_CACHEDIR": "/w/cache", "APPTAINER_TMPDIR": "/w/tmp", "APPTAINER_DOCKER_USERNAME": "u",
        "APPTAINER_DOCKER_PASSWORD": "t"}


def test_a_report_round_trips_and_an_unreadable_one_is_none(tmp_path):
    record = ImageRecord(name="alpha", source=_source("alpha"), sha256="ab" * 32, size=7, apptainer=VERSION)
    report = Report(outcomes=[Outcome(name="alpha", status="pulled", record=record, seconds=1.5),
                              Outcome(name="beta", status="failed", why="no route")])
    path = tmp_path / "report.json"
    write_report(path, report)
    assert read_report(path) == report and [p.name for p in tmp_path.iterdir()] == ["report.json"]
    path.write_text("{")
    assert read_report(path) is None and read_report(tmp_path / "none.json") is None


# ---- the stage


def _stage_plan(tmp_path: Path, *, secrets: SecretsFile | None = None, site: dict | None = None,
                images: SandboxImageSet = SANDBOX_IMAGES) -> Plan:
    decls = [*provision_stages(task_ref=TaskRef(module=MODULE), shape=ResourceShape(cpus=4, memory_gib=16,
                                                                                    time_sec=600)),
             sandbox_images_stage(images, runner_memory_gib=4)]
    # A secrets file in a directory of its own, which no store lies in.
    site = {"secrets_path": str(tmp_path / "secrets" / "secrets.toml"), **(site or {})}
    return noop_plan(tmp_path, decls=decls, sandbox_stores={STORE: str(sandbox_store(tmp_path, STORE))},
                     secrets=secrets, site=site)


def _ctx(plan: Plan, launcher: LocalLauncher, tmp_path: Path, *, secrets: dict[str, str] | None = None,
         attempt: int = 1) -> StageContext:
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file(SANDBOX_IMAGES_STAGE, attempt, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage(SANDBOX_IMAGES_STAGE), attempt=attempt,
                        launcher=launcher, watch=None, events=events, inputs={}, deadline=time.time() + 120,
                        secrets=secrets or {}, scratch=tmp_path / "node" / "hls-100", job_id="100")


def _events(ctx: StageContext) -> list[dict]:
    return [json.loads(line) for line in (ctx.attempt_dir / "events.jsonl").read_text().splitlines()]


def test_the_stage_runs_one_host_role_that_leaves_every_image_in_the_sites_store_and_names_each(tmp_path,
                                                                                               apptainer):
    plan = _stage_plan(tmp_path)
    launcher = LocalLauncher(gpus=0)
    ctx = _ctx(plan, launcher, tmp_path)
    outputs = run_stage(SandboxImagesStage(), ctx)
    store_dir = sandbox_store(tmp_path, STORE)
    own = ctx.scratch / "roles" / ROLE
    (spec,) = launcher.launched
    # The role's command line: the role program under the plan's interpreter, on node-local disk.
    assert spec.argv == (str(plan.toolchain.python), "-m", "hls.provision.sandbox_images",
                         "--images", str(own / "images.json"), "--store", str(store_dir),
                         "--work", str(own / "pulls"), "--report", str(own / "report.json"))
    assert list(spec.argv) == role_argv(plan.toolchain.python, images=own / "images.json", store=store_dir,
                                        work=own / "pulls", report=own / "report.json")
    assert ROLE_MODULE == "hls.provision.sandbox_images" and spec.name == ROLE == "sandbox-images"
    assert spec.env == {"PYTHONPATH": str(plan.code.checkout)} and spec.owner == "harness" and spec.env_file is None
    # The set's fixed allowances: the role gets them, the stage the runner's memory on top and the disk.
    assert plan.stage(SANDBOX_IMAGES_STAGE).shape == ResourceShape(cpus=2, memory_gib=7, disk_gib=1, time_sec=600)
    assert spec.shape == ResourceShape(cpus=2, memory_gib=3, disk_gib=0, time_sec=600)
    assert set(outputs) == {"sandbox_images"} and outputs["sandbox_images"].endswith(RECORD_FILE)
    record = SandboxImagesRecord.model_validate_json(ctx.store.read_bytes(outputs["sandbox_images"]))
    assert record.store == STORE and [i.name for i in record.images] == list(DIGESTS)
    for image in record.images:
        assert image == read_record(store_dir, image.name) and image.source == _source(image.name)
        assert image.sha256 == sha256_bytes(image_file(store_dir, image.name).read_bytes())
    assert missing_images(store_dir, SANDBOX_IMAGES) == []
    pulled = [e for e in _events(ctx) if e["kind"] == "sandbox_image_pulled"]
    assert [(e["name"], e["source"], e["sha256"]) for e in pulled] == [
        (i.name, i.source, i.sha256) for i in record.images]


def test_a_failed_image_fails_the_stage_with_one_line_naming_it_and_the_rest_stay_held(tmp_path, apptainer,
                                                                                      monkeypatch):
    monkeypatch.setenv("FAKE_APPTAINER_FAIL", "beta gamma")
    plan = _stage_plan(tmp_path)
    ctx = _ctx(plan, LocalLauncher(gpus=0), tmp_path)
    with pytest.raises(StageFault) as raised:
        run_stage(SandboxImagesStage(), ctx)
    fault = raised.value.fault
    why = ("apptainer pull exited 255: FATAL: While making image from oci registry: error fetching image: "
           "manifest unknown")
    assert fault.reason == f"sandbox images failed: beta: {why}; gamma: {why}"
    assert "\n" not in fault.reason and fault.fault_class == "infra" and fault.retryable
    assert fault.evidence["images"] == "beta,gamma" and fault.evidence["store"] == STORE
    store_dir = sandbox_store(tmp_path, STORE)
    assert _store_names(store_dir) == ["alpha.json", "alpha.sif"]
    kinds = [(e["kind"], e["name"]) for e in _events(ctx) if e["kind"].startswith("sandbox_image_")]
    assert kinds == [("sandbox_image_pulled", "alpha"), ("sandbox_image_failed", "beta"),
                     ("sandbox_image_failed", "gamma")]
    # No record of the stage is published: its output names every image or there is none.
    assert not (ctx.attempt_dir / "outputs" / RECORD_FILE).exists()
    # The retry finds the one that was pulled and pulls the rest.
    monkeypatch.delenv("FAKE_APPTAINER_FAIL")
    retry = _ctx(plan, LocalLauncher(gpus=0), tmp_path, attempt=2)
    run_stage(SandboxImagesStage(), retry)
    kinds = [(e["kind"], e["name"]) for e in _events(retry) if e["kind"].startswith("sandbox_image_")]
    assert kinds == [("sandbox_image_found", "alpha"), ("sandbox_image_pulled", "beta"),
                     ("sandbox_image_pulled", "gamma")]
    assert sorted(_pulls(tmp_path)[:3]) == ["alpha", "beta", "gamma"]
    assert sorted(_pulls(tmp_path)[3:]) == ["beta", "gamma"]


def test_a_role_that_ends_without_its_report_fails_the_stage(tmp_path, monkeypatch):
    # No apptainer at all: the role cannot ask its version and ends before it reports.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "apptainer").write_text("#!/bin/sh\nexit 127\n")
    (bin_dir / "apptainer").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    ctx = _ctx(_stage_plan(tmp_path), LocalLauncher(gpus=0), tmp_path)
    with pytest.raises(StageFault) as raised:
        run_stage(SandboxImagesStage(), ctx)
    assert raised.value.fault.fault_class == "infra" and ROLE in raised.value.fault.evidence["roles"]
    assert not sandbox_store(tmp_path, STORE).exists() or _store_names(sandbox_store(tmp_path, STORE)) == []


def test_the_stage_is_registered_asks_for_no_gpu_and_names_the_store_it_writes():
    assert STAGES["sandbox_images"] is SandboxImagesStage and SANDBOX_IMAGES_STAGE in KNOWN_STAGE_NAMES
    params = SandboxImagesParams(images=SANDBOX_IMAGES)
    assert SandboxImagesStage.sandbox_stores(params) == (STORE,) and SandboxImagesStage.max_roles(params) == 1
    with pytest.raises(ValueError, match="asks for no GPU"):
        SandboxImagesStage.check(ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=8, time_sec=60), params,
                                 runner_memory_gib=4)
    with pytest.raises(ValueError, match="leaves its roles none beside the runner's 4 GiB"):
        SandboxImagesStage.check(ResourceShape(cpus=2, memory_gib=4, time_sec=60), params, runner_memory_gib=4)


def test_a_login_is_the_user_name_declared_and_the_token_through_the_env_file(tmp_path, apptainer):
    token = "registry-token-0123456789"
    secrets = SecretsFile.model_validate({"secrets": {"registry_token": token},
                                          "grants": {"stage.sandbox_images": {"REGISTRY_PASSWORD": "registry_token"}}})
    plan = _stage_plan(tmp_path, secrets=secrets, site={"registry_username": "site-user"})
    launcher = LocalLauncher(gpus=0)
    ctx = _ctx(plan, launcher, tmp_path, secrets={"REGISTRY_PASSWORD": token})
    run_stage(SandboxImagesStage(), ctx)
    (spec,) = launcher.launched
    assert spec.env == {"PYTHONPATH": str(plan.code.checkout), "REGISTRY_USERNAME": "site-user"}
    assert spec.env_file is not None and token not in " ".join(spec.argv)
    for line in (tmp_path / "apptainer.log").read_text().splitlines():
        assert line.split()[5:7] == ["user=site-user", f"password={token}"]
    # The token is in no log or event of the attempt.
    for path in ctx.attempt_dir.rglob("*"):
        if path.is_file():
            assert token.encode() not in path.read_bytes(), path


def test_one_half_of_a_login_is_refused_when_the_plan_is_made(tmp_path):
    with pytest.raises(PlanError, match="site.registry_username is set but stage sandbox_images is not granted "
                                        "REGISTRY_PASSWORD"):
        _stage_plan(tmp_path, site={"registry_username": "site-user"})


# ---- the plan of hls provision


def _provision_plan(config: Path, lines: list[str] | None = None, module: str = MODULE) -> Plan:
    return commands.plan_provision(config=config, overrides={}, task_module=module, weights=True,
                                   job_name="hls-provision", echo=(lines if lines is not None else []).append)


def test_the_plan_for_a_store_that_lacks_images_carries_the_stage_and_for_a_full_store_leaves_it_out(
        tmp_path, login, apptainer):
    config = write_config(tmp_path, sandbox_stores=(STORE,))
    store_dir = sandbox_store(tmp_path, STORE)
    lines: list[str] = []
    plan = _provision_plan(config, lines)
    assert [s.name for s in plan.stages] == ["provision", "sandbox_images"]
    stage = plan.stage("sandbox_images")
    assert stage.kind == "sandbox_images" and stage.after == ["provision"]
    assert SandboxImagesParams.model_validate(stage.params).images == SANDBOX_IMAGES
    # The task's fixed allowances, with the runner's memory on top.
    assert stage.shape == ResourceShape(cpus=2, memory_gib=3 + plan.settings.run.runner_memory_gib, disk_gib=1,
                                        time_sec=600)
    assert lines[0] == f"sandbox store {STORE} lacks 3 of {MODULE}'s 3 sandbox images; the plan pulls them"
    # One image short is still short: the stage is planned, and its role leaves the held ones alone.
    pull_missing(SANDBOX_IMAGES, store_dir, tmp_path / "work")
    record_file(store_dir, "gamma").unlink()
    lines = []
    assert [s.name for s in _provision_plan(config, lines).stages] == ["provision", "sandbox_images"]
    assert lines[0] == f"sandbox store {STORE} lacks 1 of {MODULE}'s 3 sandbox images; the plan pulls them"
    # A full store: provision alone.
    pull_missing(SANDBOX_IMAGES, store_dir, tmp_path / "work")
    lines = []
    assert [s.name for s in _provision_plan(config, lines).stages] == ["provision"]
    assert not [line for line in lines if "sandbox" in line]


def test_a_task_that_declares_no_sandbox_images_plans_provision_alone(tmp_path, login):
    config = write_config(tmp_path, sandbox_stores=(STORE,))
    assert [s.name for s in _provision_plan(config, module="tests.provision.tasks").stages] == ["provision"]
    assert commands.declared_sandbox_images("tests.provision.tasks") is None
    assert commands.declared_sandbox_images(MODULE) == SANDBOX_IMAGES


def test_a_site_that_names_no_directory_for_the_store_is_refused_before_anything_is_submitted(tmp_path, login):
    config = write_config(tmp_path)
    with pytest.raises(PlanError, match=r"stage sandbox_images binds the sandbox stores test-images, which the site "
                                        r"does not name: give each its directory in hls.toml's \[sandbox_stores\]"):
        _provision_plan(config)
