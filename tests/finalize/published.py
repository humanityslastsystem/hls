"""One instance's published fixture stages, provision, images, package, deploy_drive and
verify, as their manifests and outputs look in the store, and the context the score and
finalize stages run in over them. Each stage's times are fixed (``TIMES``) so a record's
times can be checked against them."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from hls.core.payloads import TextPayload
from hls.core.trace import Trace
from hls.core.verdict import Verdict
from hls.drive.bounds import REPLY_MAX_BYTES
from hls.measured import BASELINE_CREDIT_FILE, CALIBRATION_FILE, DEFAULTS_DIR, MeasuredFile, MeasuredFiles
from hls.drive.record import ArtifactRecord, DriveRecord, DriverRecord, SealProbeRecord
from hls.drive.stream import TraceStreamRecord
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from hls.package.result import PackageResult
from hls.pipeline.checkout import CodeState
from hls.pipeline.events import EventLog
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES
from hls.pipeline.plan import Plan, StagePlan, Toolchain
from hls.pipeline.planner import StageDecl, build_plan
from hls.pipeline.stage import StageContext
from hls.provision.model import weights_key
from hls.provision.mounts import copies_gib
from hls.spec.resources import ArtifactPlacement, GrantedResources, ResourceShape, concurrent
from hls.stages.deploy_drive import ImageArtifact, PackagedArtifact, PublishedArtifact, extracted_disk_gib
from hls.store.digest import sha256_bytes
from hls.store.layout import InstanceLayout
from hls.store.manifest import (
    Manifest,
    Provenance,
    Resources,
    Times,
    output_ref,
    publish_manifest,
    read_manifest,
)
from hls.store.posix import PosixStore
from hls.taskref import blob_id
from hls.substrate.nproc import ProcessCeiling
from hls.substrate.secret_forms import FORMS
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE, PackageScan
from hls.substrate.secrets import Scrubber
from hls.trace.store import write_trace
from hls.verify.role import VerifyRecord
from hls.verify.run import write_verdict
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import settings
from tests.trace.traces import drive_trace

ARTIFACT_SHAPE = ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=4, disk_gib=1, time_sec=600)
#: What a served drive's record says of its trace writer: it kept up and did not fail.
KEPT_UP = TraceStreamRecord(backlog_peak=1, lag_max_sec=0.01, space_wait_max_sec=0.0, failure=None)
DRIVER_SHAPE = ResourceShape(cpus=1, memory_gib=2, time_sec=600)
#: The memory the test plans leave to the runner (``run.runner_memory_gib``'s default).
RUNNER_GIB = 4
_ROLES = concurrent([ARTIFACT_SHAPE, DRIVER_SHAPE])
#: The copies of the drive's data the fixture plans declare (``data_copies``): the weights, a GiB.
DATA_COPIES = {f"weights/{weights_key('org/model', 'c' * 40)}": 2**30}
_ARTIFACTS = {"package": PackagedArtifact, "image": ImageArtifact, "published": PublishedArtifact}


def drive_shape(artifact: dict[str, Any]) -> ResourceShape:
    """The fixture drive's shape for *artifact*: its roles, the runner's memory beside them, and the
    disk of the images it extracts (``extracted_disk_gib``) and of its data's copies."""
    extracted = extracted_disk_gib(_ARTIFACTS[artifact["kind"]].model_validate(artifact))
    return _ROLES.model_copy(update={"memory_gib": _ROLES.memory_gib + RUNNER_GIB,
                                     "disk_gib": _ROLES.disk_gib + extracted + copies_gib(DATA_COPIES)})


#: The fixture drive's shape for this instance's package.
DRIVE_SHAPE = drive_shape({"kind": "package"})
#: A host stage's shape, and its role's: the stage's less the runner's memory.
CPU = ResourceShape(cpus=2, memory_gib=4 + RUNNER_GIB, time_sec=600)
CPU_ROLE = CPU.model_copy(update={"memory_gib": 4})
#: The fixture drives' trace cap; their drives are kilobytes, so the least a task gets.
DRIVE_CAP = 2**30
#: The fixture drives' bytes of requests in their traces beside the replies
#: (``DeployDriveParams.trace_request_bytes``): a few kilobytes, a MiB with room.
DRIVE_REQUESTS = 2**20
#: What every fixture plan's drive stage is given beside its task and artifact.
DRIVE_PARAMS = {"harness_image": "harness-0123", "port": 8000, "artifact_shape": ARTIFACT_SHAPE.model_dump(),
                "driver_shape": DRIVER_SHAPE.model_dump(), "readiness_timeout_sec": 5, "wall_margin_sec": 2, "drive_timeout_sec": 10, "staging_sec": 0,
                "runtime_start_sec": 0, "request_timeout_sec": REQUEST_TIMEOUT_SEC,
                "reply_max_bytes": REPLY_MAX_BYTES, "trace_cap_bytes": DRIVE_CAP,
                "trace_request_bytes": DRIVE_REQUESTS, "measured_baseline": False, "engine_cache_owner": None,
                "data_copies": DATA_COPIES}
TASK = "fin"
ROLLOUT, GRADING = "20261002T000000Z-abcdef", "g-20261002T010000Z-123456"
SIF = b"submission image"
SIF_SHA = hashlib.sha256(SIF).hexdigest()
#: Each fixture stage's start and end.
TIMES = {"provision": (10.0, 20.0), "images": (20.0, 30.0), "package": (30.0, 40.0),
         "deploy_drive": (100.0, 200.0), "verify": (200.0, 250.0), "score": (250.0, 260.0)}
#: The image digests each fixture stage records.
IMAGES = {"images": {"sa-base": "1" * 64}, "package": {"submission": SIF_SHA},
          "deploy_drive": {"artifact": SIF_SHA, "driver": "2" * 64}, "verify": {"verify": "3" * 64}}
#: The role logs each fixture stage's attempt holds.
LOGS = {"deploy_drive": ("artifact.out", "artifact.err", "driver.out", "driver.err"), "verify": ("verify.err",)}
#: Whose code each fixture role ran, as the attempt's events record it.
OWNERS = {"artifact": "contestant", "driver": "harness", "verify": "harness"}
#: The source_scan's record of a bundle and its tree in which a scan of no secret found nothing, and
#: the package's of what the install added.
TREE_SCAN = PackageScan(secrets=["hf_token", "relay"], findings=[], read=["bundle", "tree"],
                        forms=list(FORMS)).model_dump_json().encode()
INSTALL_SCAN = PackageScan(secrets=["hf_token", "relay"], findings=[], read=["install"],
                           forms=list(FORMS)).model_dump_json().encode()
#: The reason a fixture drive that did not serve gives.
NOT_SERVED = {"invalid_artifact": "pyproject.toml is missing",
              "crashed": "the server exited with status 1 before it was ready",
              "not_ready": "the server ran but was not ready within the 5s readiness budget"}


#: The stages ``publish_instance`` publishes before the drive, by name and kind, in the order an
#: evaluation of a build runs them, and verify's after it.
PUBLISHED_BEFORE = (("provision", "provision"), ("images", "images"), ("agent_build", "agent_build"),
                    ("source_scan", "source_scan"), ("package", "package"))
PUBLISHED_AFTER = (("verify", "verify"),)


def naming_published(plan: Plan) -> Plan:
    """*plan* naming every stage ``publish_instance`` publishes that it does not name yet
    (``PUBLISHED_BEFORE``, ``PUBLISHED_AFTER``) under its kind, as a real plan names every stage
    whose files its instance holds, so finalize reads each stage's files by what its kind
    declares. Their parameters are none of finalize's business: they are left empty, unchecked
    by the planner, and the stages are in no allocation, so no runner runs them."""
    model = plan.stages[-1]
    named = {s.name for s in plan.stages}

    def unchecked(stages: tuple[tuple[str, str], ...]) -> list:
        return [model.model_copy(update={"name": name, "kind": kind, "params": {}, "after": []})
                for name, kind in stages if name not in named]

    return plan.model_copy(update={"stages": [*unchecked(PUBLISHED_BEFORE), *plan.stages,
                                              *unchecked(PUBLISHED_AFTER)]})


def with_stages(plan: Plan, *stages: StagePlan) -> Plan:
    """*plan* with *stages* first, each in place of the stage of its name the plan names already
    (``naming_published`` names every stage the fixture publishes), so the plan names each once."""
    names = {s.name for s in stages}
    return plan.model_copy(update={"stages": [*stages, *(s for s in plan.stages if s.name not in names)]})


def default_pins(checkout: Path, module: str) -> MeasuredFiles:
    """The pins a plan of the task module *module* makes of its default measured files in
    *checkout* (``<package>/defaults/``), as ``hls run`` and ``hls grade`` pin a default file; none
    for a file it has no default of."""
    found = {}
    for name in (BASELINE_CREDIT_FILE, CALIBRATION_FILE):
        path = checkout / module.split(".")[0] / DEFAULTS_DIR / name
        if path.is_file():
            found[name] = MeasuredFile(source="defaults", path=path.relative_to(checkout).as_posix(),
                                       blob=blob_id(path.read_bytes()))
    return MeasuredFiles(credit=found.get(BASELINE_CREDIT_FILE), calibration=found.get(CALIBRATION_FILE))


def make_plan(tmp_path: Path, checkout: Path, module: str, *, kind: str = "rollout",
              artifact: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None,
              before: tuple[StageDecl, ...] = (), names_published: bool = True) -> Plan:
    """A plan of *before*, a drive, its score and its finalize for task module *module*; for
    *kind* "grading", a grading of the fixture rollout, finalized under its grading id. It pins the
    default measured files of *module* in *checkout* (``default_pins``). It names every other
    stage ``publish_instance`` publishes (``naming_published``) unless *names_published* is False,
    for a test of the plan's own stages that publishes nothing."""
    layout = InstanceLayout.rollout(TASK, ROLLOUT)
    if kind == "grading":
        layout = layout.grading(GRADING)
    task = {"module": module, "overrides": overrides or {}}
    drive = {"task": task, "artifact": artifact or {"kind": "package"}, **DRIVE_PARAMS}
    finalize = {"task": task, "kind": kind, "eval_id": GRADING if kind == "grading" else "terminal"}
    decls = [*before, StageDecl("deploy_drive", "deploy_drive", drive_shape(drive["artifact"]), drive),
             StageDecl("score", "score", CPU, {"task": task}),
             StageDecl("finalize", "finalize", CPU, finalize)]
    plan = build_plan(decls=decls, task=TASK, pipeline="evaluate", layout=layout, settings=settings(tmp_path),
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-fin", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=1000.0,
                      measured=default_pins(checkout, module))
    return naming_published(plan) if names_published else plan


def with_params(plan: Plan, stage: str, **changes: Any) -> Plan:
    """*plan* with *changes* to *stage*'s parameters, as a planner could have written them."""
    stages = [s.model_copy(update={"params": {**s.params, **changes}}) if s.name == stage else s
              for s in plan.stages]
    return plan.model_copy(update={"stages": stages})


def with_inputs(plan: Plan, stage: str, pins: dict[str, str]) -> Plan:
    """*plan* with *stage* pinning the manifests of other instances *pins* (key to digest), as a
    planner pins the package a grading's drive serves."""
    stages = [s.model_copy(update={"inputs": {**s.inputs, **pins}}) if s.name == stage else s
              for s in plan.stages]
    return plan.model_copy(update={"stages": stages})


def write(store: PosixStore, plan: Plan, stage: str, file: str, data: bytes, *, attempt: int = 1) -> str:
    """Write *data* to the attempt's ``outputs/<file>`` and return its key."""
    key = plan.layout.attempt_file(stage, attempt, f"outputs/{file}")
    store.write_bytes(key, data)
    return key


def publish(store: PosixStore, plan: Plan, stage: str, outputs: dict[str, str], *, attempt: int = 1,
            inputs: dict[str, str] | None = None) -> Manifest:
    """Publish *stage*'s manifest over *outputs* (name to key), each recorded by the digest of
    the bytes its fixture writer left, with its fixture times, images and role logs, and the
    owner of each role in the attempt's events."""
    events = EventLog(store.path(plan.layout.attempt_file(stage, attempt, "events.jsonl")))
    for role in dict.fromkeys(log.rsplit(".", 1)[0] for log in LOGS.get(stage, ())):
        events.emit("role_started", role=role, owner=OWNERS[role])
    for log in LOGS.get(stage, ()):
        store.write_bytes(plan.layout.attempt_file(stage, attempt, f"logs/{log}"), f"{log} of {stage}\n".encode())
    start, end = TIMES.get(stage, (5.0, 6.0))
    manifest = Manifest(
        instance=plan.instance, stage=stage, attempt=attempt,
        outputs={name: output_ref(store, key, sha256_bytes(store.read_bytes(key))) for name, key in outputs.items()},
        inputs=inputs or {},
        provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                              image_digests=IMAGES.get(stage, {}), config_sha256="d" * 64),
        resources=Resources(declared=CPU, granted=GrantedResources(gpus=0, cpus=2, memory_gib=4.0,
                                                                   disk_free_gib=100.0)),
        times=Times(submitted_at=1.0, allocation_started_at=2.0, started_at=start, ended_at=end))
    publish_manifest(store, plan.layout, manifest)
    return manifest


def _pin(store: PosixStore, plan: Plan, stage: str) -> dict[str, str]:
    key = plan.layout.manifest(stage)
    return {key: hashlib.sha256(store.read_bytes(key)).hexdigest()}


def package_result(plan: Plan, *, invalid: bool, returncode: int | None = 1,
                   log_tail: str | None = "ERROR: No matching distribution found for torch==9") -> bytes:
    """A package record: invalid with the given diagnostics, or built."""
    if invalid:
        result = PackageResult(task=TASK, status="invalid_artifact", bundle_sha256="e" * 64, commit="f" * 40,
                               reason=NOT_SERVED["invalid_artifact"], returncode=returncode, log_tail=log_tail,
                               served_ucounts=None)
    else:
        result = PackageResult(task=TASK, status="built", bundle_sha256="e" * 64, commit="f" * 40,
                               image=plan.layout.attempt_file("package", 1, "outputs/submission.sif"),
                               sif_sha256=SIF_SHA, served_ucounts=None)
    return result.model_dump_json(indent=2).encode()


def drive_record(outcome: str, end, placement: ArtifactPlacement | None = None) -> bytes:
    """A drive record that ended *outcome*, its artifact's step at *placement* when it ran."""
    served = outcome == "served"
    ran = outcome != "invalid_artifact"
    artifact = ArtifactRecord(
        image_sha256=SIF_SHA, ending=None if served else "exited", exit_code=None if served else 1,
        command_started_after_sec=0.5, ready_after_command_sec=1.0 if served else None,
        placement=placement, engine_cache=None) if ran else None
    probe = SealProbeRecord(refused_host="hls-seal-probe.invalid", refusal_sec=0.1, ingress_sec=0.2,
                            icmp_sec=0.1, ucounts_sec=0.0, caps_sec=0.0, inbound_tcp=20001, inbound_udp=20002,
                            inbound_addresses=("127.0.0.1", "10.0.0.5"),
                            inbound_sec=0.1, sec=1.0) if ran else None
    ceiling = ProcessCeiling(hard=None, reserve=1024, evaluations=8, evaluations_unknown=None, uncapped=None,
                             nproc_max=65_536, ceiling=65_536, floor=8 * 4096) if ran else None
    return DriveRecord(
        outcome=outcome, reason=None if served else NOT_SERVED[outcome], seal_probe=probe,
        process_ceiling=ceiling, artifact=artifact,
        driver=DriverRecord(ending="exited", exit_code=0, drive_timeout_sec=600, trace_stream=KEPT_UP)
        if served else None,
        url="http://127.0.0.1:8000" if ran else None,
        drive_started_at=100.5 if served else None, drive_ended_at=199.5 if served else None,
        trace=end).model_dump_json(indent=2).encode()


def publish_instance(store: PosixStore, plan: Plan, *, outcome: str = "served", trace: Trace | None = None,
                     verdict: Verdict | None = None, package: bytes | None = None,
                     extra: dict[str, bytes] | None = None,
                     package_extra: dict[str, bytes] | None = None,
                     drive_inputs: dict[str, str] | None = None, bundle: bytes | None = None,
                     build_extra: dict[str, tuple[str, bytes]] | None = None,
                     tree_scan: bytes | None = TREE_SCAN, install_scan: bytes | None = INSTALL_SCAN,
                     placement: ArtifactPlacement | None = None,
                     verify_record: VerifyRecord | None = None,
                     image: str = "submission.sif") -> dict[str, Manifest]:
    """Publish the fixture instance's stages up to verify: a drive that ended *outcome*,
    with *trace* when it served, verified as *verdict*. *package* replaces the package
    record; *extra* adds files (name to bytes) to the images stage's outputs, and
    *package_extra* to the package stage's; *drive_inputs* adds manifests the drive pinned (key
    to digest), such as another instance's package (``publish_rollout_package``). With *bundle*,
    an agent_build publishes it as its source bundle first, and *build_extra* beside it (output name
    to file name and bytes). The source_scan stage publishes
    *tree_scan* as its ``package-scan.json`` (no stage for None), pinning the agent_build, and the
    package publishes *install_scan* as its own (none for None, or for an invalid artifact),
    pinning both. The drive's record places the artifact's step at *placement*, and the verify
    stage publishes *verify_record* as its ``verify.json``, by default one of no engine. The
    package publishes its submission image as *image*, Apptainer's name by default."""
    layout = plan.layout
    published = {}
    provisioned = write(store, plan, "provision", "provisioned.json", b'{"weights": null}')
    published["provision"] = publish(store, plan, "provision", {"provisioned": provisioned})
    images = {"images": write(store, plan, "images", "images.json", b'{"sa-base": "built"}')}
    images |= {name: write(store, plan, "images", name, data) for name, data in (extra or {}).items()}
    published["images"] = publish(store, plan, "images", images)
    invalid = outcome == "invalid_artifact"
    pins: dict[str, str] = {}
    if bundle is not None:
        built = {name: write(store, plan, "agent_build", file, data) for name, (file, data) in (build_extra or {}).items()}
        built["source_bundle"] = write(store, plan, "agent_build", "source.bundle", bundle)
        published["agent_build"] = publish(store, plan, "agent_build", built)
        pins |= _pin(store, plan, "agent_build")
    if tree_scan is not None:
        published["source_scan"] = publish(store, plan, "source_scan", {
            "source": write(store, plan, "source_scan", "source.json", b"{}"),
            "package_scan": write(store, plan, "source_scan", PACKAGE_SCAN_FILE, tree_scan)}, inputs=dict(pins))
        pins |= _pin(store, plan, "source_scan")
    record = package if package is not None else package_result(plan, invalid=invalid)
    package_outputs = {"package": write(store, plan, "package", "package.json", record)}
    if not invalid:
        package_outputs["submission"] = write(store, plan, "package", image, SIF)
        if install_scan is not None:
            package_outputs["package_scan"] = write(store, plan, "package", PACKAGE_SCAN_FILE, install_scan)
    package_outputs |= {name: write(store, plan, "package", name, data) for name, data in (package_extra or {}).items()}
    published["package"] = publish(store, plan, "package", package_outputs, inputs=pins)
    end = None
    drive_outputs = {}
    if outcome == "served":
        trace_dir = store.path(layout.attempt_file("deploy_drive", 1, "outputs/trace"))
        end = write_trace(trace, trace_dir, ending="drained", driver={"class": "Driver"}).end
        drive_outputs["trace"] = layout.attempt_file("deploy_drive", 1, "outputs/trace/END.json")
    drive_outputs["drive"] = write(store, plan, "deploy_drive", "drive.json", drive_record(outcome, end, placement))
    pins = _pin(store, plan, "provision") | _pin(store, plan, "package") | dict(drive_inputs or {})
    published["deploy_drive"] = publish(store, plan, "deploy_drive", drive_outputs, inputs=pins)
    out = store.path(layout.attempt_file("verify", 1, "outputs/verdict.json")).parent
    out.mkdir(parents=True, exist_ok=True)
    if verdict is None:
        verdict = Verdict(passed=False, per_event={}, failure_mode=outcome,
                          diagnostics={"reason": NOT_SERVED[outcome]})
    write_verdict(out, verdict, [])
    record = verify_record or VerifyRecord(eval_id="terminal", engines=[], libraries={})
    verify_outputs = {"verdict": layout.attempt_file("verify", 1, "outputs/verdict.json"),
                      "verify": write(store, plan, "verify", "verify.json", record.model_dump_json().encode())}
    if outcome == "served":
        lines = "".join(json.dumps({"event": i, "passed": True}) + "\n" for i in range(3)).encode()
        name = "replies.0.MetricVerifier"
        verify_outputs[name] = write(store, plan, "verify", f"{name}.jsonl.gz", gzip.compress(lines))
    published["verify"] = publish(store, plan, "verify", verify_outputs, inputs=_pin(store, plan, "deploy_drive"))
    return published


def echoed_trace(value: str) -> Trace:
    """A drive whose first reply echoes *value*, as a server that baked in a key would."""
    trace = drive_trace()
    first = trace.events[0]
    reply = first.response.output.model_copy(update={"contents": [TextPayload(text=f"echo {value}")]})
    leaked = first.model_copy(update={"response": dataclasses.replace(first.response, output=reply)})
    return Trace(events=[leaked, *trace.events[1:]])


def publish_rollout_package(store: PosixStore, plan: Plan, extra: dict[str, bytes] | None = None, *,
                            tree_scan: bytes = TREE_SCAN, build: bytes | None = None) -> dict[str, str]:
    """Publish the package stage of the rollout that the grading *plan* grades, after the source_scan
    stage it pinned, whose record is *tree_scan*, with *extra* files in its outputs, and return its
    manifest's key and digest, as a grading's drive pins it. With *build*, the rollout's agent
    build publishes it as its ``build.json`` first, and both later stages pin it, as a rollout's
    package pins the agent build whose bundle it packaged; without it the package is one no agent
    built, as a package warmer's."""
    rollout = plan.model_copy(update={"instance": InstanceLayout.rollout(TASK, ROLLOUT).root})
    pins: dict[str, str] = {}
    if build is not None:
        publish(store, rollout, "agent_build", {"build": write(store, rollout, "agent_build", "build.json", build)})
        pins = _pin(store, rollout, "agent_build")
    publish(store, rollout, "source_scan", {"package_scan": write(store, rollout, "source_scan", PACKAGE_SCAN_FILE,
                                                                  tree_scan)}, inputs=dict(pins))
    outputs = {"package": write(store, rollout, "package", "package.json", package_result(rollout, invalid=False)),
               "submission": write(store, rollout, "package", "submission.sif", SIF),
               "package_scan": write(store, rollout, "package", PACKAGE_SCAN_FILE, INSTALL_SCAN)}
    outputs |= {name: write(store, rollout, "package", name, data) for name, data in (extra or {}).items()}
    publish(store, rollout, "package", outputs, inputs=pins | _pin(store, rollout, "source_scan"))
    return _pin(store, rollout, "package")


def context(store: PosixStore, plan: Plan, stage: str, *, attempt: int = 1, secrets: dict[str, str] | None = None,
            launcher: LocalLauncher | None = None, registry: bool = True) -> StageContext:
    """An attempt of *stage* reading every published manifest of the instance, with
    *secrets* as its values and its scrubber's, which are the secrets registry unless
    *registry* is False."""
    layout = plan.layout
    inputs = {}
    for name in store.list_children(f"{plan.instance}/stages"):
        found = read_manifest(store, layout, name)
        if found is not None:
            inputs[layout.manifest(name)] = found[0]
    events = EventLog(store.path(layout.attempt_file(stage, attempt, "events.jsonl")))
    values = dict(secrets or {})
    return StageContext(store=store, plan=plan, stage=plan.stage(stage), attempt=attempt,
                        launcher=launcher or LocalLauncher(gpus=0), watch=None, events=events, inputs=inputs,
                        deadline=time.time() + 120, secrets=values, scrubber=Scrubber(values),
                        scratch=plan.settings.site.node_scratch / f"hls-100-{stage}-{attempt}", job_id="100",
                        secret_registry=registry)


def run_and_publish(store: PosixStore, plan: Plan, stage: str, stage_class, **kwargs) -> StageContext:
    """Run *stage* on its context, as the runner does, and publish its outputs."""
    ctx = context(store, plan, stage, **kwargs)
    publish(store, plan, stage, stage_class().run(ctx), attempt=ctx.attempt)
    return ctx
