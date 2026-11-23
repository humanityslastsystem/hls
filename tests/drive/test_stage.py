from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
import random
import re
import signal
import socket
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import hls.drive.readiness as readiness
import hls.pipeline.stage as stage_mod
import hls.substrate.apptainer.argv as apptainer_argv
from hls.drive.bounds import REPLY_MAX_BYTES, trace_event_bytes
from hls.drive.readiness import NotReady
from hls.drive.record import DriveRecord
from hls.drive.role import LATE_START_EXIT, NOFILE_SOFT
from hls.egress import seal_probe
from hls.substrate.ucounts import SEAL_UCOUNTS_VAR, UCOUNTS, ServedUcounts, served_limits, ucounts_env
from hls.egress.policy import PROBE_HOST, AccessList, EgressPosture, NetworkPolicy, SealPolicy, refused_host
from hls.egress.proxy import STATUS_FILE_BYTES
from hls.package.result import PackageResult
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from hls.pipeline.plan import Plan
from hls.pipeline.planner import PlanError, StageDecl
from hls.pipeline.stage import StageContext, StageFault, run_stage
from hls.provision.model import Provisioned, RowsRef, ToolRef, WeightsRef, rows_key, tool_key, weights_key
from hls.provision.mounts import copies_gib
from hls.spec.resources import GrantedResources, ResourceShape, concurrent
from hls.stages import deploy_drive
from hls.stages.deploy_drive import (DeployDriveParams, DeployDriveStage, ImageArtifact, PackagedArtifact,
                                     PublishedArtifact, served_cache_key)
from hls.stages.engine_cache import engine_build_env
from hls.stages.package import PACKAGE_ADDS_BYTES
from hls.store.cache import CacheEntry, ContentCache, copy_disk_bytes, entry_sha256
from hls.store.digest import sha256_bytes, sha256_file
from hls.store.engine_cache import ENGINE_CACHE_FILE, ENGINE_CACHE_KIND, EngineCacheRecord
from hls.store.lock import Holder
from hls.store.manifest import Manifest, Provenance, Resources, Times, output_ref
from hls.store.posix import PosixStore
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.seal import PayloadEnding, payload_ending, policy_refusals
from hls.substrate.base import JobState, NodeSize
from hls.substrate.memory_watch import MemorySample, MemoryWatch
from hls.substrate import memory_watch, nproc
from hls.substrate.nproc import FLOOR_PER_EVALUATION, ProcessCeiling, nproc_max_problem
from hls.substrate.sandbox import sandbox_paths, tree_problem
from hls.trace import store as trace_store
from hls.trace.store import CHUNKS_ROOM, END_FILE, SUMMARY_ROOM, read_end, read_trace, room_bytes
from tests.drive import fake_container, fake_server
from tests.drive.copy_order import CallOrder
from tests.drive.interceptor import Interceptor
from tests.pipeline.fakes import NODE, LocalLauncher
from tests.pipeline.plans import noop_plan
from tests.store.entries import entry_reads, populate, recording_reads
from tests.substrate import fake_pasta

REPO = Path(__file__).resolve().parents[2]
SERVER = Path(fake_container.__file__).with_name("fake_server.py")
ARTIFACT_SHAPE = ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=4, disk_gib=1, time_sec=600)
DRIVER_SHAPE = ResourceShape(cpus=1, memory_gib=2, time_sec=600)
#: The memory the test plans leave to the runner (``run.runner_memory_gib``'s default).
RUNNER_GIB = 4
ROLES = concurrent([ARTIFACT_SHAPE, DRIVER_SHAPE])
#: The servers of the artifact's size the fake launcher's node holds, by its GPUs.
EVALUATIONS = NODE.gpus // ARTIFACT_SHAPE.gpus
HARNESS, MEASURED = "harness-0123456789abcdef", "measured-0123456789abcdef"
#: A grading node's own per-uid limits, which every drive here runs on unless a test gives its
#: own, so no test depends on its host's.
GRADING_NODE = {u.name: {"max_inotify_instances": 128, "max_inotify_watches": 1_048_576}.get(u.name, 4_124_579)
                for u in UCOUNTS}
#: What the site declares of its grading nodes when it declares nothing (``site.grading_node_*``).
DECLARED = {"grading_node_namespaces": 1_048_576, "grading_node_inotify_instances": 128,
            "grading_node_inotify_watches": 1_048_576}
#: The figures a build of the package recorded with it, as its brief stated them: its share of a
#: grading node as the site declares it, among the 8 a node holds.
BUILD_SERVED = ServedUcounts(limits=served_limits(DECLARED, 8))
MODEL, REVISION, ROWS = "org/model", "c" * 40, rows_key("0123456789abcdef")
#: A tool bundle's two variants: the driver mounts the full one, which no other role does.
TOOL_KEYS = {"full": tool_key("tiny", "v1", "full"), "dev": tool_key("tiny", "v1", "dev", "0123456789ab")}
#: The copies of the artifact's and the driver's data, as the planner declares them
#: (``data_copies``): the weights, the rows and the full tool bundle, a MiB each, more than each
#: copy takes.
DATA_COPIES = {f"weights/{weights_key(MODEL, REVISION)}": 2**20, f"rows/{ROWS}": 2**20,
               f"tools/{TOOL_KEYS['full']}": 2**20}
#: GiB they take on the node.
DATA_GIB = copies_gib(DATA_COPIES)
#: The disk the drive of the fixtures' packaged artifact extracts its images into.
EXTRACTED_GIB = deploy_drive.extracted_disk_gib(PackagedArtifact())
SHAPE = ROLES.model_copy(update={"memory_gib": ROLES.memory_gib + RUNNER_GIB,
                                 "disk_gib": ROLES.disk_gib + EXTRACTED_GIB + DATA_GIB})


def _free_port() -> int:
    """A port the fake seal's port scan reports first (it prints 40000 + n), which is free with
    the next, which the second sealed role of a stage gets. Both lie below the kernel's range of
    ports for outgoing connections, so no client socket on the machine takes either meanwhile."""
    while True:
        port = random.randrange(20000, 32000)
        with socket.socket() as first, socket.socket() as second:
            try:
                first.bind(("0.0.0.0", port))
                second.bind(("0.0.0.0", port + 1))
            except OSError:
                continue
        return port


class _Answering:
    """Stands in for a seal that lets every host through to a server that answers: it reads each
    request behind the address the seal probe dials and answers it."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.sock.getsockname()[1]}"

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                head = b""
                while b"\r\n\r\n" not in head:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    head += chunk
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    def close(self) -> None:
        # Shutting the listener down wakes the accept its thread is blocked in; a close alone would not.
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()


def _probe_status(tmp_path: Path) -> Path:
    """Where the seal probe's seal keeps its interceptor's status, in the stage of ``_ctx``."""
    return tmp_path / "node" / "hls-100" / "roles" / "seal-probe" / "seal" / "status.json"


@pytest.fixture
def interceptor(tmp_path, monkeypatch):
    """The seal's interceptor under the air gap, which the seal probe dials in place of the seal's
    redirect, writing its status where the probe's seal keeps it."""
    running = Interceptor(_probe_status(tmp_path), deploy_drive.AIR_GAP)
    monkeypatch.setattr(deploy_drive, "PROBE_DIAL", running.address)
    yield running
    running.close()


@pytest.fixture
def bins(tmp_path, monkeypatch, interceptor):
    """The fake container runtime and seal on PATH, fast readiness polls, and no process ceiling
    (an unlimited hard limit) unless a test sets one, so no test depends on its host's limit;
    returns the bin dir. The node's own per-uid limits are ``GRADING_NODE``'s, for the same reason."""
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: None)
    monkeypatch.setattr(stage_mod, "node_limits", lambda: GRADING_NODE)
    bin_dir = tmp_path / "bin"
    fake_container.install(bin_dir)
    fake_pasta.install(bin_dir)
    venv = Path(sys.executable).parent
    monkeypatch.setenv("PATH", f"{bin_dir}:{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CONTAINER_PATH", f"{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CONTAINER_RECORDS", str(tmp_path / "records"))
    (tmp_path / "pasta.plan").write_text("pass\n" * 4)
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(tmp_path / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(tmp_path / "pasta.log"))
    (tmp_path / "counter").write_text(str(_free_port() - 40001))
    monkeypatch.setenv("FAKE_PYTHON_COUNTER", str(tmp_path / "counter"))
    monkeypatch.delenv("FAKE_PROXY_ERRORS", raising=False)
    monkeypatch.setattr(readiness, "HEALTH_POLL_SEC", 0.1)
    monkeypatch.setattr(readiness, "INFERENCE_POLL_SEC", 0.1)
    monkeypatch.setattr(deploy_drive, "PORT_POLL_SEC", 0.05)
    return bin_dir


def _server_image(path: Path, mode: str, *, runscript_at: str = ".singularity.d/runscript") -> Path:
    """A server image whose runscript, at *runscript_at*, runs ``fake_server.py`` with *mode*,
    which may carry the server's other options, after it writes the time it started where
    ``FAKE_COMMAND_STARTED`` names, its first act."""
    runscript = ('#!/bin/sh\n[ -z "$FAKE_COMMAND_STARTED" ] || date +%s.%N > "$FAKE_COMMAND_STARTED"\n'
                 f'exec python "$(dirname "$0")/../workspace/server.py" --mode {mode} "$@"\n')
    return fake_container.make_image(path, {runscript_at: (runscript.encode(), 0o755),
                                            "workspace/server.py": (SERVER.read_bytes(), 0o644)})


def _rows(n: int) -> list[dict]:
    return [{"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": i % 4, "subject": "x", "source_index": i}
            for i in range(n)]


def _fill_rows(d: Path) -> None:
    for split, n in (("dev", 5), ("test", 10)):
        (d / "tenants" / "t0" / split).mkdir(parents=True)
        (d / "tenants" / "t0" / split / "rows.json").write_text(json.dumps(_rows(n)))


def _manifest(plan: Plan, store: PosixStore, stage: str, outputs: dict[str, str]) -> Manifest:
    refs = {name: output_ref(store, key, sha256_file(store.path(key))[0]) for name, key in outputs.items()}
    return Manifest(instance=plan.instance, stage=stage, attempt=1, outputs=refs, inputs={},
                    provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                                          image_digests={}, config_sha256="d" * 64),
                    resources=Resources(declared=SHAPE, granted=GrantedResources(gpus=0, cpus=3, memory_gib=6.0,
                                                                                 disk_free_gib=100.0)),
                    times=Times(submitted_at=1.0, allocation_started_at=2.0, started_at=2.5, ended_at=3.0))


#: The drive's trace cap in the fixtures; the stage takes it as the planner sets it from the
#: task's drive bounds.
DRIVE_CAP = 2**30
#: The request bytes the fixture drives record, which finalize reads; the stage reads none.
DRIVE_REQUESTS = 2**20
#: How long after a command's own first instant the stage may first see it started: one pass of
#: its wait, and half a second for a loaded machine.
SEEN_WITHIN_SEC = stage_mod._POLL_SEC + 0.5


def _params(**changes) -> dict:
    return {"task": {"module": "tasks.sample_task.task"}, "artifact": {"kind": "package"}, "harness_image": "harness",
            "port": 8000, "artifact_shape": ARTIFACT_SHAPE.model_dump(), "driver_shape": DRIVER_SHAPE.model_dump(),
            "readiness_timeout_sec": 5, "wall_margin_sec": 2, "drive_timeout_sec": 10, "staging_sec": 0,
            "runtime_start_sec": 0, "request_timeout_sec": REQUEST_TIMEOUT_SEC,
            "reply_max_bytes": REPLY_MAX_BYTES,
            "trace_cap_bytes": DRIVE_CAP, "trace_request_bytes": DRIVE_REQUESTS, "measured_baseline": False,
            "engine_cache_owner": None,
            "data_copies": DATA_COPIES, **changes}


def _package(plan: Plan, store: PosixStore, *, mode: str, invalid: bool, image: bytes | None,
             served: ServedUcounts | None) -> Manifest:
    layout = plan.layout
    outputs = {}
    if invalid:
        result = PackageResult(task="sample", status="invalid_artifact", bundle_sha256="e" * 64, commit="f" * 40,
                               reason="pyproject.toml is missing", served_ucounts=served)
    else:
        outputs["submission"] = layout.attempt_file("package", 1, "outputs/submission.sif")
        sif = store.path(outputs["submission"])
        sif.parent.mkdir(parents=True, exist_ok=True)
        if image is None:
            _server_image(sif, mode)
        else:
            sif.write_bytes(image)
        result = PackageResult(task="sample", status="built", bundle_sha256="e" * 64, commit="f" * 40,
                               image=outputs["submission"], sif_sha256=sha256_file(sif)[0], served_ucounts=served)
    outputs["package"] = layout.attempt_file("package", 1, "outputs/package.json")
    store.write_bytes(outputs["package"], result.model_dump_json().encode())
    return _manifest(plan, store, "package", outputs)


def _ctx(tmp_path: Path, bin_dir: Path, *, mode: str = "serve", invalid: bool = False, image: bytes | None = None,
         launcher: LocalLauncher | None = None, deadline_sec: float = 120, shape: ResourceShape | None = None,
         watch=None, images: dict[str, dict] | None = None, tools: bool = False, site: dict | None = None,
         served: ServedUcounts | None = BUILD_SERVED, stores: dict[str, str] | None = None,
         **params) -> StageContext:
    decl = _params(**params)
    if shape is None:
        # SHAPE, with the disk of the images a drive of the artifact extracts.
        extracted = deploy_drive.extracted_disk_gib(DeployDriveParams.model_validate(decl).artifact)
        shape = SHAPE.model_copy(update={"disk_gib": ROLES.disk_gib + extracted + DATA_GIB})
    plan = noop_plan(tmp_path, decls=[StageDecl("deploy_drive", "deploy_drive", shape, decl)], site=site,
                     sandbox_stores=stores)
    plan = plan.model_copy(update={"code": plan.code.model_copy(update={"checkout": REPO}),
                                   "toolchain": plan.toolchain.model_copy(update={"python": bin_dir / "python"})})
    store = PosixStore(plan.store_root)
    cache = ContentCache(store)
    holder = Holder.here("test")
    harness = populate(cache, "images", HARNESS, lambda d: fake_container.make_image(
        d / "image.sif", {".singularity.d/labels.json": (b"{}", 0o644)}), holder=holder).entry
    measured = populate(cache, "images", MEASURED, lambda d: _server_image(d / "image.sif", mode), holder=holder).entry
    weights = populate(cache, "weights", weights_key(MODEL, REVISION), lambda d: (d / "config.json").write_text("{}"),
                       holder=holder).entry
    rows = populate(cache, "rows", ROWS, _fill_rows, holder=holder).entry
    bundles = []
    for variant, key in TOOL_KEYS.items() if tools else ():
        entry = populate(cache, "tools", key, lambda d, v=variant: (d / "bundle.txt").write_text(v),
                         holder=holder).entry
        bundles.append(ToolRef(name="tiny", identity="v1", variant=variant,
                               dev_digest=None if variant == "full" else "0123456789ab", key=key,
                               entry_sha256=entry_sha256(entry), disk_bytes=copy_disk_bytes(entry)))
    provisioned = Provisioned(
        weights=WeightsRef(repo=MODEL, revision=REVISION, key=weights.key, entry_sha256=entry_sha256(weights),
                           disk_bytes=copy_disk_bytes(weights)),
        rows=RowsRef(key=ROWS, entry_sha256=entry_sha256(rows), disk_bytes=copy_disk_bytes(rows), tenants=["t0"]),
        tools=bundles)
    prov_key = plan.layout.attempt_file("provision", 1, "outputs/provisioned.json")
    store.write_bytes(prov_key, provisioned.model_dump_json().encode())
    if images is None:
        images = {name: {"key": e.key, "entry_sha256": entry_sha256(e), "sif_sha256": _sif_of(e)}
                  for name, e in (("harness", harness), ("measured", measured))}
    images_key = plan.layout.attempt_file("images", 1, "outputs/images.json")
    store.write_bytes(images_key, json.dumps(images).encode())
    inputs = {plan.layout.manifest("provision"): _manifest(plan, store, "provision", {"provisioned": prov_key}),
              plan.layout.manifest("images"): _manifest(plan, store, "images", {"images": images_key}),
              plan.layout.manifest("package"): _package(plan, store, mode=mode, invalid=invalid, image=image,
                                                        served=served)}
    events = EventLog(store.path(plan.layout.attempt_file("deploy_drive", 1, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage("deploy_drive"), attempt=1,
                        launcher=(launcher or LocalLauncher()).use_plan(plan), watch=watch, events=events,
                        inputs=inputs,
                        deadline=time.time() + deadline_sec, cache=cache, scratch=tmp_path / "node" / "hls-100",
                        job_id="100", runtime=ApptainerRuntime(), images=ApptainerImages(tmp_path / "node" / "hls-100"))


def _sif_of(entry) -> str:
    return next(f.sha256 for f in entry.manifest.files if f.path == "image.sif")


def _drive(ctx: StageContext, outputs: dict[str, str]) -> DriveRecord:
    return DriveRecord.model_validate_json(ctx.store.read_bytes(outputs["drive"]))


def _events(ctx: StageContext) -> list[dict]:
    return read_events(ctx.attempt_dir / "events.jsonl")


def _started(ctx: StageContext) -> list[str]:
    """The roles started, in order, with each copy of a cache entry as ``copy`` and each image
    extraction as ``extract``."""
    def kind(role: str) -> str:
        return next((k for k in ("copy", "extract") if role.startswith(f"{k}-")), role)

    return [kind(e["role"]) for e in _events(ctx) if e["kind"] == "role_started"]


def _ended(ctx: StageContext, role: str) -> dict:
    [event] = [e for e in _events(ctx) if e["kind"] == "role_ended" and e["role"] == role]
    return event


def _submission(ctx: StageContext) -> Path:
    return ctx.store.path(ctx.inputs[ctx.plan.layout.manifest("package")].outputs["submission"].key)


def _submission_sha(ctx: StageContext) -> str:
    return sha256_file(_submission(ctx))[0]


def _cached_sha(ctx: StageContext, key: str) -> str:
    assert ctx.cache is not None
    entry = ctx.cache.lookup("images", key)
    assert entry is not None
    return _sif_of(entry)


def _fault(ctx: StageContext) -> StageFault:
    with pytest.raises(StageFault) as e:
        DeployDriveStage().run(ctx)
    assert all(h.poll() is not None for h in ctx.launcher.handles), "a role outlived the stage"
    assert not ctx.store.path(ctx.output_key("drive.json")).exists()
    return e.value


def _artifact_argv(ctx: StageContext) -> list[str]:
    return list(next(s.argv for s in ctx.launcher.launched if s.name == "artifact"))


def test_a_served_artifact_is_driven_and_its_trace_published(tmp_path, bins):
    ctx = _ctx(tmp_path, bins)
    outputs = DeployDriveStage().run(ctx)
    record = _drive(ctx, outputs)
    assert (record.outcome, record.reason) == ("served", None)
    assert outputs["trace"] == ctx.output_key("trace/END.json")
    trace_dir = ctx.store.path(outputs["trace"]).parent
    assert read_end(trace_dir) == record.trace and (record.trace.events, record.trace.ending) == (10, "drained")
    assert all(ev.response is not None and ev.error is None for ev in read_trace(trace_dir).events)
    # The trace's END.json reaches the manifest by the driver's own report of it.
    assert ctx.output_ref(outputs["trace"]).sha256 == sha256_file(trace_dir / END_FILE)[0]
    # Every step is a role: the copies of the cache entries, the two extractions, the seal's probe,
    # the server and the driver; no role copies the submission off the store.
    assert _started(ctx) == ["copy", "copy", "extract", "extract", "seal-probe", "artifact", "driver"]
    owners = {s.name: s.owner for s in ctx.launcher.launched if not s.name.startswith(("extract-", "copy-"))}
    assert owners == {"seal-probe": "harness", "artifact": "contestant", "driver": "harness"}
    assert all(s.owner == "harness" for s in ctx.launcher.launched if s.name.startswith(("extract-", "copy-")))
    assert ctx.image_digests == {"artifact": _submission_sha(ctx), "driver": _cached_sha(ctx, HARNESS),
                                 "seal-probe": _cached_sha(ctx, HARNESS)}
    assert record.artifact.image_sha256 == ctx.image_digests["artifact"]
    # The stage stopped the server after the drive, as it planned: no fault, and no ending of its own.
    assert (record.artifact.ending, record.artifact.exit_code) == (None, None) and record.artifact.ready_after_command_sec > 0
    assert _ended(ctx, "artifact")["ending"] == "planned_stop"
    assert (record.driver.ending, record.driver.exit_code) == ("exited", 0)
    assert record.drive_started_at < record.drive_ended_at
    assert set(ctx.cache_entries) == {f"images/{HARNESS}", f"weights/{weights_key(MODEL, REVISION)}", f"rows/{ROWS}"}
    # The seal's probe and the artifact are sealed, each forwarding a host port of its own to the
    # served port and nothing else: no UDP port in, nothing out to the host's loopback.
    calls = fake_pasta.calls(tmp_path / "pasta.log")
    assert len(calls) == 2
    for flags in calls:
        [tcp_in] = fake_pasta.forwards(flags)["-t"]
        assert tcp_in.startswith("0.0.0.0/") and tcp_in.endswith(":8000"), flags
        assert fake_pasta.forwards(flags) == {**fake_pasta.NO_FORWARDS, "-t": [tcp_in]} and "--no-map-gw" in flags
    assert all(h.poll() is not None for h in ctx.launcher.handles)


def test_a_seal_its_substrate_holds_to_no_count_is_probed_for_none(tmp_path, bins, interceptor, monkeypatch):
    """A Docker seal applies no per-namespace count limit (``Substrate.served_ucounts``), so its
    probe is given none to read back, as the build's brief states none."""
    held = deploy_drive.substrate

    def without_counts(site):
        return dataclasses.replace(held(site), served_ucounts=frozenset())

    monkeypatch.setattr(deploy_drive, "substrate", without_counts)
    ctx = _ctx(tmp_path, bins)
    _drive(ctx, DeployDriveStage().run(ctx))
    command = fake_container.records(tmp_path / "records")["seal-probe"]["command"]
    assert command[command.index("--ucounts") + 1] == ""
    [event] = [e for e in _events(ctx) if e["kind"] == "drive.seal_probe"]
    assert event["passed"] is True and event["ucounts"] == "the seal was given no limits"


def test_the_seal_is_probed_before_the_artifact_starts(tmp_path, bins, interceptor):
    # The probe runs once every copy and extraction is done, sealed as the artifact will be: a
    # host the policy does not name is refused, and the port the artifact will listen on is
    # reached through the seal's host port.
    ctx = _ctx(tmp_path, bins)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert _started(ctx).index("seal-probe") == _started(ctx).index("artifact") - 1
    assert _started(ctx)[:4] == ["copy", "copy", "extract", "extract"]
    probe = next(s for s in ctx.launcher.launched if s.name == "seal-probe")
    artifact = next(s for s in ctx.launcher.launched if s.name == "artifact")
    assert probe.argv[:10] == artifact.argv[:10]  # the same seal: its policy and its ingress port
    # The seal's interceptor itself recorded the refusal, by its policy, where no container sees it.
    assert policy_refusals(ctx.seal_files("seal-probe").status) == (PROBE_HOST,)
    seal = record.seal_probe
    assert seal is not None and seal.refused_host == PROBE_HOST
    assert 0 <= seal.refusal_sec <= seal.sec and 0 <= seal.ingress_sec <= seal.sec < deploy_drive.SEAL_PROBE_SEC
    [event] = [e for e in _events(ctx) if e["kind"] == "drive.seal_probe"]
    assert (event["passed"], event["refused_host"], event["refusal_sec"], event["ingress_sec"]) == (
        True, seal.refused_host, seal.refusal_sec, seal.ingress_sec)
    # Two more ports the probe bound in its seal, TCP and UDP, below the node's ephemeral range,
    # were refused from every address of the node, and nothing reached them.
    low = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0])
    assert seal.inbound_tcp != seal.inbound_udp and max(seal.inbound_tcp, seal.inbound_udp) < low
    assert "127.0.0.1" in seal.inbound_addresses and len(seal.inbound_addresses) >= 2
    assert 0 <= seal.inbound_sec <= seal.sec and "nothing reached" in event["inbound"]
    assert {(k["proto"], k["port"]) for k in event["inbound_knocks"]} == {
        ("tcp", seal.inbound_tcp), ("udp", seal.inbound_udp)}
    assert {k["address"] for k in event["inbound_knocks"]} == set(seal.inbound_addresses)
    assert all(k["outcome"] in ("refused", "unanswered") for k in event["inbound_knocks"])
    # No echo could leave the seal: its firewall refused the probe's sends.
    assert "no echo left" in event["icmp"] and 0 <= seal.icmp_sec == event["icmp_sec"]
    # The probe was given the limits the artifact's seal was, a graded server's share of a grading
    # node as the site declares it (StageContext.grading_ucounts), read each back inside the seal,
    # and could not raise the first with the artifact's capabilities, before the artifact started.
    limits = ucounts_env(BUILD_SERVED.figures())[SEAL_UCOUNTS_VAR]
    command = fake_container.records(tmp_path / "records")["seal-probe"]["command"]
    assert command[command.index("--ucounts") + 1] == limits
    # Each of its checks has the seal probe's own check time, the one a build's seal probe gives
    # each of its commands.
    assert command[command.index("--check-sec") + 1] == str(seal_probe.CHECK_SEC)
    assert probe.env[SEAL_UCOUNTS_VAR] == artifact.env[SEAL_UCOUNTS_VAR] == limits
    assert limits and event["ucounts"].startswith("read as given: ")
    first, value = limits.split()[0].split("=")
    assert f"raising {first} to {int(value) + 1} was refused" in event["ucounts"]
    # It held no capability, as the artifact will hold none.
    assert event["caps"].startswith("no capability in any set") and 0 <= seal.caps_sec == event["caps_sec"]
    assert 0 <= seal.ucounts_sec == event["ucounts_sec"]
    assert all(f"{name} {value}" in event["ucounts"] for name, value in (p.split("=") for p in limits.split()))
    probe_record = fake_container.records(tmp_path / "records")["seal-probe"]
    assert probe_record["in_pasta"] and probe_record["drop_caps"] == "ALL"
    # The probe is the checkout's own stdlib program, run in isolated mode.
    assert probe_record["command"][:3] == ["python", "-I", deploy_drive.PROBE_SCRIPT]
    assert f"{REPO}:/opt/hls-src:ro" in probe_record["binds"]
    assert REPO / deploy_drive.PROBE_SCRIPT.removeprefix("/opt/hls-src/") == Path(seal_probe.__file__).resolve()


def test_a_packaged_servers_seal_writes_the_figures_its_build_recorded_whatever_the_grading_would_compute(
        tmp_path, bins, monkeypatch):
    """A packaged server's seal and its probe write exactly the figures its build computed and
    its brief stated, which the package carries (``PackageResult.served_ucounts``), never what this
    grading's site declaration or node would give. Only the counts this node's kernel has are
    written, since one it lacks cannot be made here."""
    node = {name: limit * 2 for name, limit in GRADING_NODE.items() if name != "max_time_namespaces"}
    monkeypatch.setattr(stage_mod, "node_limits", lambda: node)
    recorded = ServedUcounts(limits=served_limits({**DECLARED, "grading_node_inotify_instances": 64}, 8))
    ctx = _ctx(tmp_path, bins, site={"grading_node_inotify_instances": 256}, served=recorded)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served"
    want = {name: role for name, role in recorded.figures().items() if name in node}
    assert want["max_inotify_instances"] == 8 and "max_time_namespaces" not in want
    for role in ("seal-probe", "artifact"):
        assert next(s for s in ctx.launcher.launched if s.name == role).env[SEAL_UCOUNTS_VAR] == (
            ucounts_env(want)[SEAL_UCOUNTS_VAR])
    [event] = [e for e in _events(ctx) if e["kind"] == "grading_ucounts"]
    assert (event["source"], event["evaluations"], event["held"]) == ("recorded", EVALUATIONS, want)
    assert not [e for e in _events(ctx) if e["kind"] == "served_ucounts"]  # nothing computed at grading


def test_a_measured_image_no_build_stated_figures_for_is_held_to_the_sites_declared_share(tmp_path, bins):
    """A measured image has no build, so no brief stated its figures: its grading computes them
    from the site's declaration, among the servers of its size a node holds."""
    artifact = {"kind": "image", "image": "measured", "entrypoint": ["python", "/workspace/server.py", "--mode", "serve"],
                "runtime_env": {}}
    ctx = _ctx(tmp_path, bins, artifact=artifact, site={"grading_node_inotify_instances": 64})
    assert _drive(ctx, DeployDriveStage().run(ctx)).outcome == "served"
    want = ServedUcounts(limits=served_limits({**DECLARED, "grading_node_inotify_instances": 64},
                                              EVALUATIONS)).figures()
    assert next(s for s in ctx.launcher.launched if s.name == "artifact").env[SEAL_UCOUNTS_VAR] == (
        ucounts_env(want)[SEAL_UCOUNTS_VAR])
    [event] = [e for e in _events(ctx) if e["kind"] == "grading_ucounts"]
    assert event["source"] == "declared" and event["held"]["max_inotify_instances"] == 8


@pytest.mark.parametrize("count, short", [("max_inotify_instances", 8 * 16 - 1),
                                          ("max_user_namespaces", 8 * ((1_048_576 - 8 * 16) // 8 + 16) - 1)])
def test_a_node_whose_own_limits_cannot_hold_its_graded_servers_is_infra_retried_before_any_role(
        tmp_path, bins, monkeypatch, count, short):
    """A node that holds less of a count than the servers of this size it holds take with the
    share the build recorded, beside what each harness keeps, cannot give each its share: another
    node may, so the stage is infra and retried, before anything starts."""
    monkeypatch.setattr(stage_mod, "node_limits", lambda: {**GRADING_NODE, count: short})
    ctx = _ctx(tmp_path, bins)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason.startswith(f"this node cannot give its graded servers the limits they are held to: "
                                   f"this node's {count} of {short} cannot hold 8 graded servers")
    assert _started(ctx) == []


class _CpuNode(LocalLauncher):
    """A local launcher whose node has no GPU and 192 CPUs."""

    def node_size(self) -> NodeSize:
        return NodeSize(gpus=0, cpus=192)


def test_a_cpu_only_graded_servers_limits_count_the_servers_of_its_own_cpus_a_node_holds(tmp_path, bins):
    """With no GPU, k is the node's CPUs over the served artifact's own, as the build computes it
    (``StageContext.served_ucounts``), never over the stage's, which hold the driver's CPUs too:
    an 8-CPU server on a 192-CPU node is one of 24, 5 inotify instances, where the stage's 9 CPUs
    would have made it one of 21 and given it 6. The grading node is held to that many servers'
    shares of what the build recorded."""
    artifact = ARTIFACT_SHAPE.model_copy(update={"gpus": 0, "gpu_kind": None, "cpus": 8})
    roles = concurrent([artifact, DRIVER_SHAPE])
    shape = roles.model_copy(update={"memory_gib": roles.memory_gib + RUNNER_GIB, "disk_gib": SHAPE.disk_gib})
    ctx = _ctx(tmp_path, bins, launcher=_CpuNode(), shape=shape, artifact_shape=artifact.model_dump(),
               site={"nproc_max": 24 * FLOOR_PER_EVALUATION})
    assert ctx.process_ceiling().evaluations == 192 // 9
    computed = ctx.served_ucounts(artifact)
    assert computed.figures()["max_inotify_instances"] == 128 // 24 == 5
    assert computed == ServedUcounts(limits=served_limits(DECLARED, 24))
    assert ctx.grading_ucounts(artifact, None) == ctx.grading_ucounts(artifact, computed) == computed.figures()
    # A share recorded for fewer servers to a node does not fit 24 of them on this node.
    with pytest.raises(StageFault, match="cannot hold 24 graded servers"):
        ctx.grading_ucounts(artifact, BUILD_SERVED)


def test_the_brief_states_the_figures_the_build_records_and_the_grading_seal_writes(tmp_path, bins, monkeypatch):
    """The figures a graded server is held to are computed once, when the build renders its brief
    (``StageContext.served_ucounts``, from ``hls.substrate.ucounts.served_limits``): the brief
    states them, the build records them (its ``served_ucounts`` output, which the package carries
    into ``package.json``), and the grading seal and its probe write exactly those, computing
    nothing of their own."""
    from hls.stages import agent_build
    from hls.taskref import TaskRef, load_task
    from hls.substrate.ucounts import ServedLimit
    from tests.agent import test_stage as agent_tests

    def marked(declared, evaluations):
        return tuple(ServedLimit(name=u.name, setting=u.declared, declared=declared[u.declared],
                                 evaluations=evaluations, outside=u.outside, role=u.floor + 7_001 + i, floor=u.floor)
                     for i, u in enumerate(UCOUNTS))

    monkeypatch.setattr(stage_mod, "served_limits", marked)
    # A grading node that holds whatever it returns.
    monkeypatch.setattr(stage_mod, "node_limits", lambda: {u.name: 10**9 for u in UCOUNTS})
    marks = {c.name: c.role for c in marked(DECLARED, EVALUATIONS)}
    # The brief, rendered as a build renders it, and the record it writes beside it.
    (tmp_path / "build").mkdir()
    build_ctx = agent_tests._ctx(tmp_path / "build", agent_tests._params(served_shape=ARTIFACT_SHAPE.model_dump()))
    try:
        build = agent_build._Build(build_ctx)
        build.ceiling = build_ctx.process_ceiling()
        brief = " ".join(build.brief(load_task(TaskRef(module="tests.agent.build_task")), build.artifact()).split())
        recorded = ServedUcounts.model_validate_json(
            build_ctx.store.read_bytes(build_ctx.output_key(agent_build.SERVED_UCOUNTS_FILE)))
    finally:
        while agent_tests._CONTEXTS:
            agent_tests._CONTEXTS.pop().close()
        while agent_tests._SCRATCHES:
            agent_tests.remove_tree(agent_tests._SCRATCHES.pop(), ignore_errors=True)
    assert (f"It and every process it starts may hold at most {marks['max_inotify_instances']:,} inotify instances, "
            f"{marks['max_inotify_watches']:,} inotify watches, ") in brief
    assert all(f"{marks[name]:,} {name.removeprefix('max_').removesuffix('_namespaces')} namespaces" in brief
               for name in marks if name.endswith("_namespaces"))
    assert recorded.figures() == marks
    # The grading of the package that carries the record, which computes nothing of its own.

    def computed_again(declared, evaluations):
        raise AssertionError("a grading of a build's package computes its figures again")

    monkeypatch.setattr(stage_mod, "served_limits", computed_again)
    ctx = _ctx(tmp_path, bins, served=recorded)
    _drive(ctx, DeployDriveStage().run(ctx))
    written = ucounts_env(marks)[SEAL_UCOUNTS_VAR]
    for role in ("seal-probe", "artifact"):
        assert next(s for s in ctx.launcher.launched if s.name == role).env[SEAL_UCOUNTS_VAR] == written


def test_the_seal_probe_asks_for_a_host_its_policy_refuses(tmp_path, bins, interceptor):
    network = NetworkPolicy(domains=AccessList(mode="allow", entries=["example.com", "*.example.org"]))
    ctx = _ctx(tmp_path, bins, network=network.model_dump(mode="json"))
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served"
    assert policy_refusals(ctx.seal_files("seal-probe").status) == (PROBE_HOST,)
    # The probe is sealed with the artifact's own policy, not the air gap.
    probe = next(s for s in ctx.launcher.launched if s.name == "seal-probe")
    assert SealPolicy.model_validate_json(probe.argv[8]) == SealPolicy(network=network) == (
        SealPolicy.model_validate_json(_artifact_argv(ctx)[8]))
    assert refused_host(network) == PROBE_HOST
    deny = NetworkPolicy(domains=AccessList(mode="deny", entries=["*.invalid", "bad.example"]))
    assert refused_host(deny) == f"{PROBE_HOST.split('.')[0]}.invalid"
    assert not deny.domains.permits(refused_host(deny), wildcard=True)
    assert refused_host(NetworkPolicy(domains=AccessList(mode="deny"))) is None


def test_a_failed_seal_probe_is_infra_retried_and_starts_no_artifact(tmp_path, bins, monkeypatch):
    answering = _Answering()  # the seal let a host its policy does not name through
    monkeypatch.setattr(deploy_drive, "PROBE_DIAL", answering.address)
    try:
        ctx = _ctx(tmp_path, bins)
        fault = _fault(ctx).fault
    finally:
        answering.close()
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "seal probe" in fault.reason
    assert "artifact" not in _started(ctx)
    [event] = [e for e in _events(ctx) if e["kind"] == "drive.seal_probe"]
    assert event["passed"] is False and "answered" in event["refusal"]


def test_a_seal_that_forwards_the_containers_other_ports_fails_the_probe(tmp_path, bins, monkeypatch):
    # The fake seal then leaves the probe's inbound ports where it bound them, on the node itself,
    # as a seal that forwards every port its container binds would: the stage reaches them.
    monkeypatch.setenv("FAKE_SEAL_FORWARDS", "1")
    ctx = _ctx(tmp_path, bins)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "the seal forwarded tcp port" in fault.reason and "inbound: " in fault.reason
    assert "artifact" not in _started(ctx)
    [event] = [e for e in _events(ctx) if e["kind"] == "drive.seal_probe"]
    assert event["passed"] is False and "reached" in event["inbound"]
    assert {k["outcome"] for k in event["inbound_knocks"] if k["proto"] == "tcp"} == {"connected"}
    # Every knock came before the stage's second request, which the probe answers only once it
    # has closed its ports, so each found them held: the order alone decides it.
    assert {k["outcome"] for k in event["inbound_knocks"] if k["proto"] == "udp"} == {"unanswered"}


def test_a_payload_that_keeps_a_capability_fails_the_probe(tmp_path, bins, monkeypatch):
    # The probe holds what the artifact would: here the harness's sealed set, which leaves it
    # unable to raise its limits but not without capabilities.
    monkeypatch.setattr(apptainer_argv, "PAYLOAD_DROP_CAPS", apptainer_argv.SEALED_DROP_CAPS)
    ctx = _ctx(tmp_path, bins)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "caps: the probe holds capabilities: CapInh " in fault.reason and "ucounts: " not in fault.reason
    assert "artifact" not in _started(ctx)
    [event] = [e for e in _events(ctx) if e["kind"] == "drive.seal_probe"]
    assert event["passed"] is False and "CapEff" in event["caps"] and "was refused" in event["ucounts"]


def test_a_payload_that_can_raise_its_limits_fails_the_probe(tmp_path, bins, monkeypatch):
    # Root of pasta's user namespace holding CAP_SYS_RESOURCE there could lift the limits the seal
    # wrote once the probe had read them; the probe, with the artifact's own capabilities, raised one.
    monkeypatch.setattr(apptainer_argv, "PAYLOAD_DROP_CAPS", "CAP_NET_ADMIN,CAP_NET_RAW")
    ctx = _ctx(tmp_path, bins)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    first, value = ucounts_env(BUILD_SERVED.figures())[SEAL_UCOUNTS_VAR].split()[0].split("=")
    assert f"ucounts: {first} was raised from {value} to {int(value) + 1} inside the seal" in fault.reason
    assert "artifact" not in _started(ctx)
    [event] = [e for e in _events(ctx) if e["kind"] == "drive.seal_probe"]
    assert event["passed"] is False and "was raised" in event["ucounts"]


def test_a_seal_probe_refused_without_the_policys_record_of_it_fails(tmp_path, bins, monkeypatch):
    # A connection closed unanswered is no proof: an interceptor that lets every host through
    # closes it too, when the host does not resolve. Only its record of a refusal by its policy is.
    open_policy = NetworkPolicy(domains=AccessList(mode="deny", entries=[]))
    permissive = Interceptor(_probe_status(tmp_path), open_policy)
    monkeypatch.setattr(deploy_drive, "PROBE_DIAL", permissive.address)
    try:
        ctx = _ctx(tmp_path, bins)
        fault = _fault(ctx).fault
    finally:
        permissive.close()
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "recorded no refusal of hls-seal-probe.invalid by its policy" in fault.reason
    assert "artifact" not in _started(ctx)
    [event] = [e for e in _events(ctx) if e["kind"] == "drive.seal_probe"]
    assert event["passed"] is False and event["refused_by_policy"] is False


class _FakeProbe(ApptainerRuntime):
    """Runs the seal probe's container with a stand-in for its program: it asks the interceptor
    for the host as the real one does, then writes a report that every check passed, with no
    refusal check when *report_refusal* is False, a failed inbound check when *inbound* is False,
    a failed ICMP check when *icmp* is False, a failed ucounts check when *ucounts* is False and a
    failed caps check when *caps* is False, and answers every request with its own path, or with
    ``ok`` when *echo* is False."""

    def __init__(self, *, report_refusal: bool = True, echo: bool = True, inbound: bool = True,
                 icmp: bool = True, ucounts: bool = True, caps: bool = True) -> None:
        super().__init__()
        #: Whether the report holds the refusal check.
        self.report_refusal = report_refusal
        #: Whether each request is answered with its own path.
        self.echo = echo
        #: Whether the report's inbound check passed.
        self.inbound = inbound
        #: Whether the report's ICMP check passed.
        self.icmp = icmp
        #: Whether the report's ucounts check passed.
        self.ucounts = ucounts
        #: Whether the report's caps check passed.
        self.caps = caps

    def argv(self, spec, *, env_file, seal_python=None):
        if spec.workdir.parent.name == "seal-probe":
            args = list(spec.command)
            given = {k: args[args.index(k) + 1] for k in ("--port", "--out", "--refused-host", "--dial")}
            code = (
                "import json, socket, sys\n"
                "port, out = int(sys.argv[2]), sys.argv[3]\n"
                f"addr, _, dial = {given['--dial']!r}.rpartition(':')\n"
                "with socket.create_connection((addr, int(dial)), timeout=5) as c:\n"
                f"    c.sendall(b'GET / HTTP/1.1\\r\\nHost: {given['--refused-host']}\\r\\n\\r\\n')\n"
                "    c.recv(1)\n"
                "check = {'passed': True, 'sec': 0.1, 'detail': 'stand-in'}\n"
                "reached = {'passed': False, 'sec': 0.1, 'detail': 'stand-in: tcp port 1 was reached'}\n"
                "left = {'passed': False, 'sec': 0.1, 'detail': 'stand-in: a ping echo left the seal'}\n"
                "wrong = {'passed': False, 'sec': 0.0, 'detail': 'stand-in: max_user_namespaces reads 7, not 8'}\n"
                "held = {'passed': False, 'sec': 0.0, 'detail': 'stand-in: CapEff 0000000001000000'}\n"
                f"report = {{'refusal': check if {self.report_refusal!r} else None, 'ingress': check,\n"
                f"          'inbound': check if {self.inbound!r} else reached,\n"
                f"          'icmp': check if {self.icmp!r} else left,\n"
                f"          'ucounts': check if {self.ucounts!r} else wrong,\n"
                f"          'caps': check if {self.caps!r} else held, 'reached': []}}\n"
                "open(out, 'w').write(json.dumps(report))\n"
                "s = socket.socket()\n"
                "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
                "s.bind(('0.0.0.0', port))\n"
                "s.listen()\n"
                "s.settimeout(20)\n"
                "while True:\n"
                "    conn, _ = s.accept()\n"
                "    path = conn.recv(4096).split(b' ')[1].lstrip(b'/')\n"
                f"    body = path if {self.echo!r} else b'ok'\n"
                "    conn.sendall(b'HTTP/1.0 200 OK\\r\\nContent-Length: %d\\r\\n\\r\\n%s' % (len(body), body))\n"
                "    conn.close()\n")
            # The fake container rewrites the port to the seal's host port and the report's path to
            # where its mount lies.
            spec = dataclasses.replace(spec, command=("python", "-c", code, "--port", given["--port"],
                                                      given["--out"]))
        return super().argv(spec, env_file=env_file, seal_python=seal_python)


@pytest.mark.parametrize("probe, why", [
    (_FakeProbe(echo=False), "not answered with its path"),
    (_FakeProbe(report_refusal=False), "made no refusal check"),
    (_FakeProbe(inbound=False), "inbound: stand-in: tcp port 1 was reached"),
    (_FakeProbe(icmp=False), "icmp: stand-in: a ping echo left the seal"),
    (_FakeProbe(ucounts=False), "ucounts: stand-in: max_user_namespaces reads 7, not 8"),
    (_FakeProbe(caps=False), "caps: stand-in: CapEff 0000000001000000"),
], ids=["another-answer", "no-refusal-check", "inbound-reached", "icmp-left", "ucounts-wrong", "caps-held"])
def test_a_seal_probe_that_does_not_show_both_checks_fails(tmp_path, bins, interceptor, probe, why):
    ctx = _ctx(tmp_path, bins)
    ctx.launcher.runtime = probe
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and why in fault.reason
    assert "artifact" not in _started(ctx)


def test_a_seal_probe_past_its_time_is_infra_and_stopped(tmp_path, bins, monkeypatch):
    # A probe whose ingress check never hears from the stage outlives the probe's time.
    monkeypatch.setattr(deploy_drive, "SEAL_PROBE_SEC", 2.0)
    assert seal_probe.CHECK_SEC > 2.0  # a check may wait longer than the probe's whole time
    monkeypatch.setattr(deploy_drive._Drive, "_knock", lambda self, handle, port, give_up: False)
    ctx = _ctx(tmp_path, bins)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "more than 2s" in fault.reason
    assert _ended(ctx, "seal-probe")["ending"] == "planned_stop" and "artifact" not in _started(ctx)


class _RunscriptElsewhere(ApptainerImages):
    """Apptainer's image runtime, but keeping a submission image's runscript where another
    runtime's images keep it."""

    runscript = "/.hls/runscript"


def test_a_packaged_server_runs_the_runscript_where_its_image_runtime_keeps_it(tmp_path, bins):
    """The package stage writes the image's runscript where the site's image runtime keeps it, so
    the drive serves the package by that path, never by one runtime's path on another's image."""
    built = _server_image(tmp_path / "elsewhere.sif", "serve", runscript_at=".hls/runscript")
    ctx = _ctx(tmp_path, bins, image=built.read_bytes())
    ctx.images = _RunscriptElsewhere(ctx.scratch)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served"
    assert fake_container.records(tmp_path / "records")["artifact"]["command"][:1] == ["/.hls/runscript"]


def test_the_artifact_sees_only_node_local_copies_and_its_own_variables(tmp_path, bins):
    ctx = _ctx(tmp_path, bins)
    stale = ctx.scratch / "roles" / "artifact" / "engine-cache" / "inductor" / "stale"
    stale.parent.mkdir(parents=True)
    stale.write_text("an earlier attempt's kernels")
    DeployDriveStage().run(ctx)
    assert not stale.exists()  # each attempt starts from an empty engine cache
    records = fake_container.records(tmp_path / "records")
    art, drv = records["artifact"], records["driver"]
    scratch = ctx.scratch.resolve()
    binds = {b.split(":")[1]: Path(b.split(":")[0]).resolve() for b in art["binds"]}
    assert set(binds) == {"/etc/resolv.conf", "/etc/hosts", "/data/weights", "/data/tenants/t0/dev", "/engine-cache"}
    assert all(source.is_relative_to(scratch) for target, source in binds.items() if not target.startswith("/etc/"))
    assert art["command"] == ["/.singularity.d/runscript", "--host", "0.0.0.0", "--port", "8000"]
    assert art["nv"] and art["drop_caps"] == "ALL"
    argv = _artifact_argv(ctx)
    assert SealPolicy.model_validate_json(argv[8]).network.domain_posture() is EgressPosture.AIR_GAP
    env = art["env"]
    assert (env["HF_HUB_OFFLINE"], env["HF_DATASETS_OFFLINE"], env["TORCH_CUDA_ARCH_LIST"]) == ("1", "1", "9.0")
    assert (env["HLS_WEIGHTS_DIR"], env["VLLM_CACHE_ROOT"]) == ("/data/weights", "/engine-cache/vllm")
    assert env["NO_PROXY"] == "*" and "PYTHONPATH" not in env
    # The driver is ours: the checkout read-only, its node-local trace directory, the outputs
    # directory and its own report writable, every split.
    assert f"{REPO}:/opt/hls-src:ro" in drv["binds"] and f"{ctx.attempt_dir / 'outputs'}:/hls/out" in drv["binds"]
    assert f"{ctx.role_dir('driver') / deploy_drive.LOCAL_DIR}:/hls/local" in drv["binds"]
    assert f"{ctx.outputs_report('driver')}:/hls/outputs.json" in drv["binds"]
    assert f"{ctx.role_dir('driver') / deploy_drive.STREAM_RECORD_FILE}:/hls/stream.json" in drv["binds"]
    assert {"/data/weights", "/data/tenants/t0/dev", "/data/tenants/t0/test"} <= {b.split(":")[1] for b in drv["binds"]}
    assert (drv["env"]["PYTHONPATH"], drv["env"]["HF_HUB_OFFLINE"]) == ("/opt/hls-src/src:/opt/hls-src", "0")
    assert not drv["nv"] and drv["drop_caps"] is None and not drv["in_pasta"]
    command = drv["command"]
    assert command[:3] == ["python", "-m", "hls.drive.role"]
    flags = dict(zip(command[3::2], command[4::2]))
    named = ("--local", "--out", "--outputs-out", "--stream-out", "--nofile-soft")
    assert [flags[name] for name in named] == [
        "/hls/local", "/hls/out/trace", "/hls/outputs.json", "/hls/stream.json", str(NOFILE_SOFT)]
    assert flags["--request-timeout-sec"] == str(REQUEST_TIMEOUT_SEC)


def test_the_artifacts_writable_paths_are_on_the_nodes_disk(tmp_path, bins):
    # What the server writes, its /tmp, /var/tmp and $HOME (the workdir) and its engine
    # cache, is node-local disk in its own role directory, never the runtime's small session tmpfs.
    ctx = _ctx(tmp_path, bins)
    DeployDriveStage().run(ctx)
    art = fake_container.records(tmp_path / "records")["artifact"]
    role = ctx.scratch / "roles" / "artifact"
    assert art["workdir"] == str(role / "wd")
    argv = _artifact_argv(ctx)
    assert argv[argv.index("--workdir") + 1] == str(role / "wd")
    writable = [b for b in art["binds"] if not b.endswith(":ro")]
    assert writable == [f"{role / 'engine-cache'}:/engine-cache"]
    assert art["env"]["TMPDIR"] == "/engine-cache/tmp"


def test_the_served_image_is_its_pinned_tree_read_only(tmp_path, bins):
    # The server's /workspace and site-packages are the extracted image, run read-only;
    # its writable paths are the workdir and the engine cache alone.
    ctx = _ctx(tmp_path, bins)
    DeployDriveStage().run(ctx)
    sha = _submission_sha(ctx)
    root, manifest = sandbox_paths(ctx.scratch, sha)
    argv = _artifact_argv(ctx)
    assert argv[argv.index("--env-file") + 2] == str(root)
    assert not {"--writable", "--writable-tmpfs", "--overlay"} & set(argv)
    assert fake_container.records(tmp_path / "records")["artifact"]["image"] == str(root)
    # The server ran from the tree and left it as it was extracted; the context, not yet closed,
    # still holds it.
    problem, _ = tree_problem(root, manifest, sha)
    assert problem is None


class _Decoy(BaseHTTPRequestHandler):
    """Another job's server on the node, which counts what reaches it."""

    hits: list[str] = []

    def do_GET(self):  # noqa: N802
        self.hits.append(self.path)
        self.send_response(200)
        self.end_headers()

    do_POST = do_GET  # noqa: N815

    def log_message(self, *args):
        pass


def test_the_artifact_listens_in_its_own_seal_and_is_driven_at_the_seals_host_port(tmp_path, bins):
    # Another server on the node at the port the artifact listens on inside its seal is
    # never reached; the stage drives the host port the seal wrote.
    decoy = ThreadingHTTPServer(("127.0.0.1", 0), _Decoy)
    _Decoy.hits = []
    threading.Thread(target=decoy.serve_forever, daemon=True).start()
    try:
        port = decoy.server_address[1]
        ctx = _ctx(tmp_path, bins, port=port)
        record = _drive(ctx, DeployDriveStage().run(ctx))
    finally:
        decoy.shutdown()
    assert record.outcome == "served" and record.trace.events == 10
    assert record.url.startswith("http://127.0.0.1:") and not record.url.endswith(f":{port}")
    assert _artifact_argv(ctx)[9] == str(port)
    assert fake_container.records(tmp_path / "records")["artifact"]["in_pasta"]
    assert _Decoy.hits == []


def test_the_runner_reads_no_trace_chunk_and_no_image_file(tmp_path, bins):
    # The runner reads the trace's END.json alone; the driver reported every file's digest, and
    # the verify role checks the chunks as it reads them. Every image file is read by its
    # extraction role.
    ctx = _ctx(tmp_path, bins)
    with recording_reads() as reads:
        outputs = DeployDriveStage().run(ctx)
    trace_dir = ctx.store.path(outputs["trace"]).parent
    assert [r for r in entry_reads(reads, [trace_dir, _submission(ctx)]) if Path(r).name != END_FILE] == []
    assert len(list(trace_dir.glob("chunk-*.jsonl.gz"))) == 1


def test_an_invalid_package_publishes_its_result_and_starts_no_role(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, invalid=True)
    outputs = DeployDriveStage().run(ctx)
    record = _drive(ctx, outputs)
    assert set(outputs) == {"drive"}
    assert (record.outcome, record.reason) == ("invalid_artifact", "pyproject.toml is missing")
    assert (record.artifact, record.driver, record.trace, record.seal_probe) == (None, None, None, None)
    assert ctx.launcher.launched == [] and ctx.cache_entries == {} and ctx.image_digests == {}


def test_a_server_that_exits_before_it_is_ready_crashed(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, mode="crash")
    outputs = DeployDriveStage().run(ctx)
    record = _drive(ctx, outputs)
    assert set(outputs) == {"drive"} and record.outcome == "crashed" and "status 3" in record.reason
    assert (record.artifact.ending, record.artifact.exit_code, record.artifact.ready_after_command_sec) == ("exited", 3, None)
    assert record.driver is None and _started(ctx)[-1] == "artifact" and record.seal_probe is not None


def test_a_server_that_prints_a_refusal_and_exits_3_crashed_and_is_scored(tmp_path, bins):
    # A refused: line is a harness role's mark; from a contestant's server it is just output.
    ctx = _ctx(tmp_path, bins, mode="refuse")
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert (record.outcome, record.artifact.exit_code) == ("crashed", 3) and "status 3" in record.reason
    assert "refused: this server" in (ctx.attempt_dir / "logs" / "artifact.err").read_text()


def test_a_server_that_dies_as_its_readiness_budget_runs_out_crashed(tmp_path, bins, monkeypatch):
    # The gate checks the server before each probe; a server that dies after its last check,
    # just as the budget runs out, is seen dead by the stage's last look, and has crashed.
    async def gate(url, *, timeout_sec, alive):
        while alive():
            await __import__("asyncio").sleep(0.05)
        raise NotReady(f"service not healthy at {url} in {timeout_sec}s")

    monkeypatch.setattr(deploy_drive, "wait_healthy", gate)
    ctx = _ctx(tmp_path, bins, mode="fade")
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert (record.outcome, record.artifact.ending, record.artifact.exit_code) == ("crashed", "exited", 3)


class _NeverStarts(ApptainerRuntime):
    """Runs the artifact's container with a runtime that never starts its command and prints no
    start marker: *program* in apptainer's place, behind the seal as apptainer would be."""

    def __init__(self, program: Path) -> None:
        super().__init__()
        #: The program run in apptainer's place.
        self.program = program

    def argv(self, spec, *, env_file, seal_python=None):
        argv = super().argv(spec, env_file=env_file, seal_python=seal_python)
        if spec.workdir.parent.name != "artifact":
            return argv
        at = argv.index(self.apptainer)
        return [*argv[:at], str(self.program), *argv[at + 1:]]


@pytest.mark.parametrize("script, held, clean_stop", [
    ("exit 0\n", False, False),
    ("exec sleep 30\n", True, False),
    ("exec sleep 30\n", True, True),
], ids=["ended-0-before-ready", "held-to-the-deadline", "held-to-the-deadline-stop-read-as-clean"])
def test_a_server_whose_container_never_started_its_command_is_infra(tmp_path, bins, script, held, clean_stop):
    # The seal's started file says only that the runtime was launched; a server's readiness
    # budget counts only from its start marker on its stderr. A runtime that never starts the
    # command is held to the stage's deadline, past its one-second readiness budget, and is
    # infra, however its stop then reads.
    program = tmp_path / "bin" / "stuck-runtime"
    program.write_text("#!/bin/sh\n" + script)
    program.chmod(0o755)
    ctx = _ctx(tmp_path, bins, readiness_timeout_sec=1, deadline_sec=10)
    ctx.launcher.runtime = _NeverStarts(program)
    if clean_stop:
        stop = ctx.stop_role
        ctx.stop_role = lambda name: dataclasses.replace(stop(name), exit_code=0, ending="planned_stop", detail="")
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "driver" not in _started(ctx)
    assert "no start marker" in fault.reason + fault.evidence.get("artifact.detail", "")
    assert (_ended(ctx, "artifact")["at"] >= ctx.deadline) is held


def test_a_server_whose_runtime_starts_late_still_gets_its_whole_readiness_budget(tmp_path, bins, monkeypatch):
    """The readiness budget is the server's own time, from its command's start (its start
    marker): a runtime that takes longer than the whole budget to start the container after its
    seal started the payload costs the server none of it. The server, ready at once, is served,
    and its readiness figure counts from its command's start."""
    gate, first = tmp_path / "artifact-runtime-starts", tmp_path / "server-started"
    monkeypatch.setenv("FAKE_CONTAINER_HOLD", f"artifact:{gate}")
    monkeypatch.setenv("FAKE_COMMAND_STARTED", str(first))
    budget = 2
    ctx = _ctx(tmp_path, bins, readiness_timeout_sec=budget)
    released: list[float] = []
    done = threading.Event()

    def release() -> None:
        # Held for twice the budget once the seal has started the payload, which is when the
        # host port is there and a budget counted from the seal's start would begin.
        while not done.is_set() and ((files := ctx.seal_files("artifact")) is None or not files.started.exists()):
            time.sleep(0.05)
        at = ctx.clock() + 2 * budget
        while not done.is_set() and ctx.clock() < at:
            time.sleep(0.05)
        released.append(ctx.clock())
        gate.touch()

    releaser = threading.Thread(target=release)
    releaser.start()
    try:
        outputs = DeployDriveStage().run(ctx)
    finally:
        done.set()
        releaser.join()
    record = _drive(ctx, outputs)
    assert (record.outcome, record.reason) == ("served", None)
    began = ctx.command_started_at("artifact")
    assert began is not None and released[0] <= began <= float(first.read_text()) + SEEN_WITHIN_SEC
    [ready] = [e for e in _events(ctx) if e["kind"] == "artifact_ready"]
    after = record.artifact.ready_after_command_sec
    assert after is not None and 0 < after <= ready["at"] - began and ready["after_command_sec"] == round(after, 3)
    # The runtime's start, held past twice the budget, is recorded as the server's own.
    step = ctx.role_started_at("artifact")
    assert step is not None and record.artifact.command_started_after_sec == began - step >= 2 * budget
    assert ready["command_started_after_sec"] == round(began - step, 3)


def test_a_server_whose_command_starts_as_the_deadline_passes_is_infra(tmp_path, bins, monkeypatch):
    """The stage's deadline passed before the stage saw the server's command start, so the server
    had none of its readiness budget: infra, even when its start marker came in time for its
    stop to read it, never a crash of the server's."""
    gate = tmp_path / "artifact-runtime-starts"
    monkeypatch.setenv("FAKE_CONTAINER_HOLD", f"artifact:{gate}")
    ctx = _ctx(tmp_path, bins, readiness_timeout_sec=1, deadline_sec=10)
    real = ctx.clock

    def clock() -> float:
        now = real()
        if now >= ctx.deadline and not gate.exists() and "artifact" in ctx.running_roles():
            # The first look past the deadline: the runtime starts the command before it returns.
            gate.touch()
            give_up = real() + 60
            while not ctx.command_started("artifact"):
                assert real() < give_up, "the artifact's command never started"
                time.sleep(0.05)
        return now

    ctx.clock = clock
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason.startswith("the artifact's container did not start its command before the stage's deadline")
    assert gate.exists() and ctx.command_started("artifact") and "driver" not in _started(ctx)


def test_a_server_healthy_late_gets_a_whole_budget_for_its_inference_route(tmp_path, bins):
    """Each gate has a whole budget: the inference gate's counts from when the health gate
    opened. A server healthy three seconds into a six-second budget, whose inference route opens
    four and a half seconds after that, is served, readiness having taken more than one budget."""
    budget = 6
    ctx = _ctx(tmp_path, bins, mode="serve --health-after 3 --chat-after 7.5", readiness_timeout_sec=budget)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert (record.outcome, record.reason) == ("served", None)
    assert record.artifact.ready_after_command_sec is not None and record.artifact.ready_after_command_sec > budget


def test_an_inference_gate_the_allocation_cuts_after_a_whole_health_gate_is_infra(tmp_path, bins, monkeypatch):
    """The allocation's end leaves the health gate all of its budget but cuts the inference
    gate's, which counts from when the health gate opened: the stage's time cut a gate, infra,
    never a server that was not ready."""
    monkeypatch.setattr(stage_mod.StageContext, "hold_until", lambda self, until: False)
    budget, margin = 6, 2
    ctx = _ctx(tmp_path, bins, mode="noroute --health-after 2", readiness_timeout_sec=budget, wall_margin_sec=margin)
    real = ctx.wait_command_started

    def wait_command_started(name):
        began = real(name)
        if began is not None:
            # Room for the health gate's whole budget and a second of the inference gate's.
            ctx.deadline = began + budget + 1 + margin
        return began

    ctx.wait_command_started = wait_command_started
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert re.fullmatch(r"the stage's time cut a readiness gate to [0-5]s of its 6s budget", fault.reason), fault.reason


def test_a_server_still_loading_after_its_budget_is_not_ready(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, mode="hang", readiness_timeout_sec=1)
    outputs = DeployDriveStage().run(ctx)
    record = _drive(ctx, outputs)
    assert record.outcome == "not_ready" and "1s readiness budget" in record.reason
    assert (record.artifact.ending, record.driver) == (None, None) and "driver" not in _started(ctx)
    assert record.artifact.command_started_after_sec is not None and record.artifact.command_started_after_sec > 0
    assert _ended(ctx, "artifact")["ending"] == "planned_stop"
    assert all(h.poll() is not None for h in ctx.launcher.handles)


def test_a_server_whose_inference_route_never_opens_is_not_ready(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, mode="noroute", readiness_timeout_sec=1)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "not_ready" and "driver" not in _started(ctx)


def test_a_readiness_budget_the_stage_cut_short_is_infra(tmp_path, bins, monkeypatch):
    # The allocation's end leaves the gates no room to be held, and the stage keeps its margin, so
    # a gate gets at most the stage's time less the margin.
    monkeypatch.setattr(stage_mod.StageContext, "hold_until", lambda self, until: False)
    ctx = _ctx(tmp_path, bins, mode="hang", readiness_timeout_sec=30, deadline_sec=60, wall_margin_sec=52)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "stage's time" in fault.reason


def test_a_staging_overrun_does_not_cut_a_readiness_gate_the_allocation_has_room_for(tmp_path, bins):
    # The staging left the stage less than a gate's budget and the margin, but the allocation
    # ends far later: the gates hold the stage's deadline for their two budgets and the margin, as
    # the drive holds it for its wall, first from the wait for the server's command and then from
    # the command's start, and the server that never got ready is scored not ready.
    ctx = _ctx(tmp_path, bins, mode="hang", readiness_timeout_sec=4, deadline_sec=30, wall_margin_sec=27,
               drive_timeout_sec=5)
    ctx.deadline_cap = ctx.clock() + 1000.0
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "not_ready" and "4s readiness budget" in record.reason
    held = [e for e in _events(ctx) if e["kind"] == "deadline_held"]
    assert held and all(e["was"] < e["until"] <= ctx.deadline_cap for e in held)
    began = ctx.command_started_at("artifact")
    assert began is not None and held[-1]["until"] == ctx.deadline >= began + 2 * 4 + 27


def test_a_staging_overrun_does_not_cut_the_wait_for_the_servers_command(tmp_path, bins, monkeypatch):
    # The staging left the stage less than the server's runtime takes to start, but the allocation
    # ends far later: before it waits for the command the stage holds its deadline for both budgets
    # and the margin, so a command that starts after the stage's first deadline is still served.
    gate = tmp_path / "artifact-runtime-starts"
    monkeypatch.setenv("FAKE_CONTAINER_HOLD", f"artifact:{gate}")
    ctx = _ctx(tmp_path, bins, readiness_timeout_sec=4, wall_margin_sec=2, deadline_sec=8)
    ctx.deadline_cap = ctx.clock() + 1000.0
    first_deadline = ctx.deadline
    done = threading.Event()

    def release() -> None:
        while not done.is_set() and ctx.clock() < first_deadline + 0.5:
            time.sleep(0.05)
        gate.touch()

    releaser = threading.Thread(target=release)
    releaser.start()
    try:
        record = _drive(ctx, DeployDriveStage().run(ctx))
    finally:
        done.set()
        releaser.join()
    assert (record.outcome, record.reason) == ("served", None)
    began = ctx.command_started_at("artifact")
    assert began is not None and began > first_deadline


def test_a_submission_image_that_cannot_be_extracted_is_infra(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, image=b"not an image")
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "extract-" in fault.reason
    assert "artifact" not in _started(ctx)


def test_an_allocation_stopped_during_readiness_is_infra(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, mode="hang", readiness_timeout_sec=60)
    start = ctx.start_container

    def start_and_stop_soon(name, *args, **kwargs):
        handle = start(name, *args, **kwargs)
        if name == "artifact":
            threading.Timer(2.0, ctx.stop_roles).start()
        return handle

    ctx.start_container = start_and_stop_soon
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "stopped" in fault.reason
    assert fault.evidence["roles"] == "artifact"


def test_a_seal_that_never_starts_its_payload_is_infra(tmp_path, bins):
    (tmp_path / "pasta.plan").write_text("pass\n" + "fail:1\n" * 5)
    fault = _fault(_ctx(tmp_path, bins)).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "runtime failed" in fault.reason
    assert fault.evidence["roles"] == "artifact"


def test_a_seal_that_writes_no_host_port_is_infra(tmp_path, bins, monkeypatch):
    real = deploy_drive.ingress_port
    monkeypatch.setattr(deploy_drive, "PORT_WAIT_SEC", 1.0)
    ctx = _ctx(tmp_path, bins)
    monkeypatch.setattr(deploy_drive, "ingress_port",
                        lambda files: None if files == ctx.seal_files("artifact") else real(files))
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "no host port" in fault.reason


@pytest.mark.parametrize("knob, reason", [("mismatch", "grant mismatch"), ("no_report", "launch failed"),
                                          # Ended before its container started, so it left no start marker.
                                          ("scheduler_ended", "runtime failed")])
def test_an_artifact_role_the_substrate_failed_is_infra(tmp_path, bins, knob, reason):
    ctx = _ctx(tmp_path, bins, launcher=LocalLauncher(**{knob: ("artifact",)}))
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and reason in fault.reason
    assert "driver" not in [s.name for s in ctx.launcher.launched]


# The server's 8 CPUs split four on each socket.
SPLIT = ([16, 17, 18, 19, 76, 77, 80, 81], [0, 1], True)
ONE_SOCKET = ([3, 4, 5, 6, 7, 8, 9, 10], [0], False)
SOCKET_TASKS = {"evaluation_socket_tasks": True}


def test_on_a_site_with_socket_tasks_a_scored_server_on_split_cpus_is_infra_with_its_cpus(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, site=SOCKET_TASKS, launcher=LocalLauncher(placement={"artifact": SPLIT}))
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "grant mismatch" in fault.reason
    detail = fault.evidence["artifact.detail"]
    assert detail.startswith("cpus [16, 17, 18, 19, 76, 77, 80, 81] lie on NUMA nodes [0, 1]"), fault
    assert "driver" not in [s.name for s in ctx.launcher.launched]


def test_on_a_site_with_socket_tasks_a_scored_server_on_one_socket_is_driven_and_its_cpus_recorded(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, site=SOCKET_TASKS, launcher=LocalLauncher(placement={"artifact": ONE_SOCKET}))
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served"
    placement = record.artifact.placement
    assert (placement.cpus, placement.numa_nodes, placement.numa_split) == ONE_SOCKET
    roles = {s.name: s for s in ctx.launcher.launched}
    # Only the scored server is held to one socket; its driver overlaps whatever it got.
    assert roles["artifact"].one_numa_node and not roles["driver"].one_numa_node
    assert not roles["seal-probe"].one_numa_node


def test_on_a_site_without_socket_tasks_a_split_server_is_driven_and_its_split_recorded(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, launcher=LocalLauncher(placement={"artifact": SPLIT}))
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served"
    placement = record.artifact.placement
    assert (placement.cpus, placement.numa_nodes, placement.numa_split) == SPLIT
    assert not next(s for s in ctx.launcher.launched if s.name == "artifact").one_numa_node


def test_a_driver_that_fails_is_infra(tmp_path, bins):
    ctx = _ctx(tmp_path, bins)
    ctx.output_path("trace/stale")  # the driver refuses to write over an existing trace directory
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "harness role driver ended exited with status 1" in fault.reason


class _RefusingDriver(ApptainerRuntime):
    """Runs the driver's command as one that prints what a server's reply could hold, a
    ``refused:`` line among it, and exits 3."""

    def argv(self, spec, *, env_file, seal_python=None):
        if spec.workdir.parent.name == "driver":
            line = "refused: the server answered this"
            spec = dataclasses.replace(spec, command=("python", "-c", f"import sys; print({line!r}, "
                                                                        "file=sys.stderr); sys.exit(3)"))
        return super().argv(spec, env_file=env_file, seal_python=seal_python)


def test_a_driver_that_prints_a_refusal_and_exits_3_is_retried(tmp_path, bins):
    # Its stderr carries text the contestant's server controls, so no line in it decides.
    ctx = _ctx(tmp_path, bins)
    ctx.launcher.runtime = _RefusingDriver()
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "harness role driver ended exited with status 3"


def test_a_driver_role_the_substrate_failed_is_infra(tmp_path, bins):
    fault = _fault(_ctx(tmp_path, bins, launcher=LocalLauncher(mismatch=("driver",)))).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "grant mismatch" in fault.reason


def test_an_incomplete_trace_is_infra(tmp_path, bins, monkeypatch):
    real = deploy_drive.read_end

    def chunk_gone(out_dir: Path):
        (out_dir / "chunk-000000.jsonl.gz").unlink()
        return real(out_dir)

    monkeypatch.setattr(deploy_drive, "read_end", chunk_gone)
    fault = _fault(_ctx(tmp_path, bins)).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "complete trace" in fault.reason


def test_a_trace_the_driver_did_not_report_is_infra(tmp_path, bins):
    class _Unreported(ApptainerRuntime):
        def argv(self, spec, *, env_file, seal_python=None):
            if spec.workdir.parent.name == "driver":
                at = spec.command.index("--outputs-out")
                spec = dataclasses.replace(spec, command=(*spec.command[:at + 1], "/hls/out/elsewhere.json",
                                                          *spec.command[at + 2:]))
            return super().argv(spec, env_file=env_file, seal_python=seal_python)

    ctx = _ctx(tmp_path, bins)
    ctx.launcher.runtime = _Unreported()
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "reported no digest of output trace/END.json" in fault.reason


def test_the_drive_record_says_how_the_drivers_trace_writer_kept_up(tmp_path, bins):
    ctx = _ctx(tmp_path, bins)
    stream = _drive(ctx, DeployDriveStage().run(ctx)).driver.trace_stream
    assert stream.failure is None and 1 <= stream.backlog_peak <= 10
    assert 0 < stream.lag_max_sec < 60 and stream.space_wait_max_sec == 0


def test_a_driver_whose_trace_writer_failed_is_infra_and_its_evidence_says_why(tmp_path, bins, monkeypatch):
    """The room's chunks file is one the driver's writer cannot open: the role exits 1, and the
    stage's fault carries the writer's own account of why, from the record the role wrote."""
    real = deploy_drive.make_room

    def unwritable(local, **kwargs):
        real(local, **kwargs)
        (local / CHUNKS_ROOM).chmod(0o400)

    monkeypatch.setattr(deploy_drive, "make_room", unwritable)
    fault = _fault(_ctx(tmp_path, bins)).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason.startswith("harness role driver ended exited with status 1; its trace writer failed: ")
    assert "PermissionError" in fault.reason.splitlines()[0]
    assert "PermissionError" in fault.evidence["driver.trace_stream.failure"]
    assert int(fault.evidence["driver.trace_stream.backlog_peak"]) >= 1


def test_a_driver_that_exits_0_without_the_record_of_its_trace_writer_is_infra(tmp_path, bins):
    class _Unrecorded(ApptainerRuntime):
        def argv(self, spec, *, env_file, seal_python=None):
            if spec.workdir.parent.name == "driver":
                at = spec.command.index("--stream-out")
                spec = dataclasses.replace(spec, command=(*spec.command[:at + 1], "/hls/out/elsewhere.json",
                                                          *spec.command[at + 2:]))
            return super().argv(spec, env_file=env_file, seal_python=seal_python)

    ctx = _ctx(tmp_path, bins)
    ctx.launcher.runtime = _Unrecorded()
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "no record of its trace writer" in fault.reason


def test_the_drive_wall_not_the_stage_deadline_ends_a_stalled_drive(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, mode="stall", deadline_sec=25, wall_margin_sec=7, drive_timeout_sec=8)
    outputs = DeployDriveStage().run(ctx)
    record = _drive(ctx, outputs)
    # The driver's own wall cut the drive and recorded every turn as a fault, so the drive is a
    # scored result; the stage's deadline killed nothing.
    assert record.outcome == "served" and record.trace.ending == "wall"
    assert (record.driver.ending, record.driver.exit_code) == ("exited", 0)
    assert record.driver.drive_timeout_sec == 8 and float(record.trace.driver["drive_timeout_sec"]) == 8
    assert record.drive_ended_at < ctx.deadline
    assert all(ev.error is not None for ev in read_trace(ctx.store.path(outputs["trace"]).parent).events)


def test_a_wall_the_stages_time_left_cannot_hold_gets_the_deadline_grown_and_is_driven_whole(tmp_path, bins):
    # A 15 s wall with 20 s of stage time and a 7 s margin. The drive gets all
    # 15 s, never the 11 s or so left, and the stage's deadline grows to hold them.
    ctx = _ctx(tmp_path, bins, mode="stall", deadline_sec=20, wall_margin_sec=7, drive_timeout_sec=15)
    planned = ctx.deadline
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served" and record.trace.ending == "wall"
    assert record.driver.drive_timeout_sec == 15 and float(record.trace.driver["drive_timeout_sec"]) == 15
    assert record.drive_ended_at - record.drive_started_at >= 15
    # The deadline holds the whole wall and the margin after it from the drive's start, less the
    # moment between the hold and the driver's launch.
    assert ctx.deadline >= record.drive_started_at + 15 + 7 - 1 > planned and record.drive_ended_at < ctx.deadline


@pytest.mark.parametrize("runtime_start, cap", [(0, 45.0), (10, 65.0)], ids=["readiness", "runtime-start"])
def test_a_drive_that_cannot_fit_with_the_servers_readiness_is_refused_before_the_server_starts(tmp_path, bins,
                                                                                            runtime_start, cap):
    # Two 20 s readiness budgets, the 15 s wall and the 7 s margin need 62 s; an allocation that
    # ends in 45 would hold the wall and the margin alone. With 10 s for the runtime's start they
    # need 72 s, more than an allocation that ends in 65 holds. No server is started for a drive
    # that its start and readiness could leave no room to drive whole.
    ctx = _ctx(tmp_path, bins, mode="stall", deadline_sec=120, wall_margin_sec=7, drive_timeout_sec=15,
               readiness_timeout_sec=20, runtime_start_sec=runtime_start)
    ctx.deadline_cap = ctx.clock() + cap
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert (f"the {runtime_start}s allowed for the server's runtime to start its command, its two 20s readiness "
            "budgets, the drive's full wall of 15s and the 7s margin after it do not fit before the allocation's end, "
            "so the server was not started") in fault.reason
    assert "launch_wait.seal-probe" in fault.evidence and "deadline_cap" in fault.evidence
    assert "artifact" not in _started(ctx) and "driver" not in _started(ctx)


def _no_room_once_ready(monkeypatch, before=None) -> None:
    """The allocation's end leaves no room for the drive's wall once the server is ready: the stage
    cannot hold its deadline for it. *before*, when given, runs first, as what befell the server."""
    def hold_until(self, until):
        if before is not None:
            before()
        return False

    monkeypatch.setattr(stage_mod.StageContext, "hold_until", hold_until)


def _ends_soon_after_the_driver_starts(monkeypatch, *, spare: float, after=None) -> None:
    """The allocation ends *spare* seconds after the drive's wall and margin would, counted from the
    driver's launch, so a driver whose step starts more than *spare* seconds late does not drive.
    *after*, when given, runs once the driver has started, as what befell the server meanwhile."""
    real = deploy_drive._Drive.start_driver

    def start_driver(self, image, files, url, wall):
        self.ctx.deadline_cap = self.ctx.clock() + wall + self.params.wall_margin_sec + spare
        handle = real(self, image, files, url, wall)
        if after is not None:
            after()
        return handle

    monkeypatch.setattr(deploy_drive._Drive, "start_driver", start_driver)


def test_a_drive_with_no_room_once_the_server_is_ready_stops_a_clean_server_and_is_infra(tmp_path, bins,
                                                                                          monkeypatch):
    _no_room_once_ready(monkeypatch)
    ctx = _ctx(tmp_path, bins, drive_timeout_sec=15, wall_margin_sec=7)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "the drive's full wall of 15s and the 7s margin after it do not fit" in fault.reason
    assert "launch_wait.artifact" in fault.evidence and "driver" not in _started(ctx)
    assert _ended(ctx, "artifact")["ending"] == "planned_stop"


def test_a_server_at_its_own_memory_limit_is_the_contestants_fault_when_the_drive_has_no_room(tmp_path, bins,
                                                                                               monkeypatch):
    _no_room_once_ready(monkeypatch)
    ctx = _ctx(tmp_path, bins, drive_timeout_sec=15, wall_margin_sec=7, watch=_ArtifactBreached())
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False)
    assert fault.evidence["artifact.ending"] == "memory_limit" and "driver" not in _started(ctx)


@pytest.mark.parametrize("path", ["no room once ready", "driver too late"])
def test_a_no_room_fault_keeps_the_servers_own_infra_evidence(tmp_path, bins, monkeypatch, path):
    # The server was OOM-killed at a limit not certainly its own, which is infra: the no-room
    # fault says so and carries that evidence.
    cgroups = tmp_path / "cg"

    def oom() -> None:
        (cgroups / "cg-artifact" / "memory.events").write_text("oom 1\noom_kill 1\n")

    if path == "no room once ready":
        _no_room_once_ready(monkeypatch, before=oom)
        launcher = LocalLauncher(cgroup_root=cgroups)
    else:
        _ends_soon_after_the_driver_starts(monkeypatch, spare=3.0, after=oom)
        launcher = LocalLauncher(cgroup_root=cgroups, late_sec={"driver": 9.0})
    ctx = _ctx(tmp_path, bins, mode="stall", wall_margin_sec=7, drive_timeout_sec=15, launcher=launcher)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "OOM-killed" in fault.reason and fault.evidence["artifact.oom_kills"] == "1"
    assert "launch_wait.artifact" in fault.evidence and "deadline_cap" in fault.evidence


def test_a_driver_whose_step_starts_9_s_late_gets_its_full_wall_and_is_scored_the_same(tmp_path, bins):
    # Past the 7 s margin: the stage's deadline grows by the driver's wait for its step, so the
    # wall the driver counts from its own start fits, and the drive is the one a prompt start gives.
    ctx = _ctx(tmp_path, bins, mode="stall", deadline_sec=30, wall_margin_sec=7, drive_timeout_sec=15,
               launcher=LocalLauncher(late_sec={"driver": 9.0}))
    planned = ctx.deadline
    outputs = DeployDriveStage().run(ctx)
    record = _drive(ctx, outputs)
    assert ctx.launch_waits()["driver"] >= 9.0 and ctx.deadline >= planned + 9.0
    assert record.outcome == "served" and record.trace.ending == "wall"
    assert (record.driver.ending, record.driver.exit_code) == ("exited", 0)
    # The driver drove with the whole wall it was given, from its own start.
    assert float(record.trace.driver["drive_timeout_sec"]) == record.driver.drive_timeout_sec
    started = ctx.role_started_at("driver")
    assert started is not None and record.drive_ended_at - started >= record.driver.drive_timeout_sec
    # The drive's start is the driver step's, past its wait for the step.
    assert record.drive_started_at == started
    assert record.drive_ended_at < ctx.deadline
    assert all(ev.error is not None for ev in read_trace(ctx.store.path(outputs["trace"]).parent).events)


def test_a_driver_that_starts_too_late_for_its_full_wall_is_infra_and_drives_nothing(tmp_path, bins, monkeypatch):
    # Its step starts 9 s late, and the allocation ends 3 s after its wall and margin would from its
    # launch: they do not fit, so it drives nothing, and the stage is infra.
    _ends_soon_after_the_driver_starts(monkeypatch, spare=3.0)
    ctx = _ctx(tmp_path, bins, mode="stall", deadline_sec=30, wall_margin_sec=7, drive_timeout_sec=15,
               launcher=LocalLauncher(late_sec={"driver": 9.0}))
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "started too late to drive its full wall" in fault.reason
    assert float(fault.evidence["launch_wait.driver"]) >= 9.0
    assert not (ctx.outputs_dir / "trace").exists()


def test_a_contestant_fault_stands_over_a_driver_that_starts_too_late(tmp_path, bins, monkeypatch):
    # The artifact reached its own memory limit while the driver waited for its step and then
    # refused: the contestant's fault decides the stage, not the late start.
    _ends_soon_after_the_driver_starts(monkeypatch, spare=3.0)
    ctx = _ctx(tmp_path, bins, deadline_sec=30, wall_margin_sec=7, drive_timeout_sec=15, watch=_ArtifactBreached(),
               launcher=LocalLauncher(late_sec={"driver": 9.0}))
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False)
    assert fault.evidence["artifact.ending"] == "memory_limit"
    assert _ended(ctx, "driver")["exit_code"] == LATE_START_EXIT


def test_the_tasks_own_wall_holds_when_it_is_shorter(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, mode="stall", drive_timeout_sec=2)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.driver.drive_timeout_sec == 2 and record.trace.ending == "wall"
    assert record.drive_ended_at - record.drive_started_at < 30


def test_an_artifact_that_dies_during_the_drive_is_scored(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, mode="die")
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served" and (record.artifact.ending, record.artifact.exit_code) == ("exited", 3)
    assert record.trace.events == 10


class _SchedulerEnds(LocalLauncher):
    """The artifact's step reads as one the scheduler ended, whenever it ends, as a step whose
    SIGTERM came through scancel can."""

    def launch(self, spec, *, share_gpus_of=None):
        handle = super().launch(spec, share_gpus_of=share_gpus_of)
        handle.scheduler_ended = spec.name == "artifact"
        return handle


def test_an_artifact_the_scheduler_ends_during_the_drive_is_infra(tmp_path, bins):
    fault = _fault(_ctx(tmp_path, bins, mode="die", launcher=_SchedulerEnds())).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "scheduler ended" in fault.reason


def test_the_planned_stop_of_the_artifact_after_the_drive_is_no_fault(tmp_path, bins):
    # The stage's own SIGTERM is no fault even where the scheduler records the step as cancelled.
    ctx = _ctx(tmp_path, bins, launcher=_SchedulerEnds())
    record = _drive(ctx, run_stage(DeployDriveStage(), ctx))
    assert record.outcome == "served" and record.artifact.ending is None
    event = _ended(ctx, "artifact")
    assert event["ending"] == "planned_stop" and event["exit_code"] != 0


class _ArtifactBreached:
    """A memory watch that says the artifact's step was killed at its own memory limit."""

    #: No job cgroup is watched.
    job_dir = None

    def __init__(self) -> None:
        self.roles: dict[str, str] = {}

    def watch(self, step, cgroup_dir, limit_bytes, kill, *, role) -> None:
        self.roles[step] = role

    def last(self, step) -> None:
        return None

    def breach(self, step) -> MemorySample | None:
        if self.roles.get(step) != "artifact":
            return None
        return MemorySample(at=0.0, current=5 * 2**30, swap_current=0, reclaimable=0, oom_kills=0)

    def job_ooms(self) -> None:
        return None

    def watch_disk(self, step, paths, limit_bytes, kill, *, role, cgroup_dir=None, ceiling=None,
                   harness_pids=None) -> None:
        pass

    def unwatch_disk(self, step) -> None:
        pass

    def disk_breach(self, step) -> None:
        return None

    def pids_hits(self, step) -> None:
        return None

    def pool(self, step) -> None:
        return None

    def pool_peak(self, step) -> None:
        return None

    def disk_at_end(self, step) -> None:
        return None


class _KernelOom(LocalLauncher):
    """The artifact's step reads as OUT_OF_MEMORY in accounting when it ends, while its command
    runs as given: the kernel killed a process of it, at whichever limit."""

    def launch(self, spec, *, share_gpus_of=None):
        handle = super().launch(spec, share_gpus_of=share_gpus_of)
        if spec.name == "artifact":
            handle.step_state = JobState.OUT_OF_MEMORY
        return handle


def test_a_kernel_oom_kill_in_the_artifacts_planned_stop_leaves_the_drive_served(tmp_path, bins):
    # A completed drive is never zeroed during shutdown: a kill in the stop's grace period, which
    # accounting then records for the step, is part of the stop.
    cgroups = tmp_path / "cg"
    ctx = _ctx(tmp_path, bins, launcher=_KernelOom(cgroup_root=cgroups))
    real = ctx.launcher.terminate

    def terminate(handles):
        if any(h.spec.name == "artifact" for h in handles):
            (cgroups / "cg-artifact" / "memory.events").write_text("oom 1\noom_kill 1\n")
        real(handles)

    ctx.launcher.terminate = terminate
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served" and record.artifact.ending is None
    assert _ended(ctx, "artifact")["ending"] == "planned_stop" and _ended(ctx, "artifact")["oom_kills"] == 0


@pytest.mark.parametrize("how, fault_class", [("memory_limit", "contestant"), ("kernel", "infra")])
def test_an_artifact_out_of_memory_before_it_was_ready_is_the_contestants_only_when_certain(
        tmp_path, bins, how, fault_class):
    if how == "memory_limit":
        ctx = _ctx(tmp_path, bins, mode="crash", watch=_ArtifactBreached())
    else:
        ctx = _ctx(tmp_path, bins, mode="crash", launcher=_KernelOom())
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == (fault_class, fault_class == "infra")
    assert fault.evidence["roles"] == "artifact" and "driver" not in _started(ctx)
    assert "out of memory" in fault.reason or "OOM-killed" in fault.reason


def test_a_server_not_ready_at_its_memory_limit_is_the_contestants_fault(tmp_path, bins):
    # The stage stops a server that is not ready; what ended it first decides.
    ctx = _ctx(tmp_path, bins, mode="hang", readiness_timeout_sec=1, watch=_ArtifactBreached())
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False)
    assert fault.evidence["artifact.ending"] == "memory_limit"


@pytest.fixture
def disk_watch(monkeypatch):
    """A running memory watch that walks the disk every 50 ms, with a declared GiB of disk
    counted as a MiB, so a server that writes a few MiB passes its cap."""
    monkeypatch.setattr(stage_mod, "_GIB", 2**20)
    watch = MemoryWatch(enforce=False, interval_sec=0.05, disk_interval_sec=0.05)
    watch.start()
    yield watch
    watch.stop()


def test_an_artifact_that_writes_past_its_disk_cap_is_the_contestants_fault(tmp_path, bins, disk_watch):
    assert fake_server.FILL_BYTES > ARTIFACT_SHAPE.disk_gib * 2**20
    ctx = _ctx(tmp_path, bins, mode="fill", watch=disk_watch, readiness_timeout_sec=30)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False) and "disk cap" in fault.reason
    assert _ended(ctx, "artifact")["ending"] == "disk_limit" and "driver" not in _started(ctx)


def test_an_artifact_that_writes_past_its_cap_during_the_drive_is_the_contestants_fault(tmp_path, bins,
                                                                                         disk_watch):
    ctx = _ctx(tmp_path, bins, mode="spill", watch=disk_watch)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False) and "disk cap" in fault.reason
    # The artifact's fault decides the stage: the driver either ran to its end or was stopped once
    # the artifact's end was read, and its stop adds no fault.
    driver = _ended(ctx, "driver")
    assert (driver["ending"], driver["exit_code"]) == ("exited", 0) or driver["ending"] == "stopped"
    assert _ended(ctx, "artifact")["ending"] == "disk_limit"


def test_a_stage_error_after_a_decided_contestant_fault_still_raises_the_contestant_fault(
        tmp_path, bins, disk_watch, monkeypatch):
    real = deploy_drive.stage_fault

    def failing(outcomes):
        if any(o.name == "artifact" for o in outcomes):
            raise RuntimeError("planted")
        return real(outcomes)

    monkeypatch.setattr(deploy_drive, "stage_fault", failing)
    ctx = _ctx(tmp_path, bins, mode="fill", watch=disk_watch, readiness_timeout_sec=30)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False) and "disk cap" in fault.reason


def test_a_stop_that_fails_after_a_decided_contestant_fault_still_raises_the_contestant_fault(
        tmp_path, bins, disk_watch, monkeypatch):
    real = deploy_drive.stage_fault

    def failing(outcomes):
        if any(o.name == "artifact" for o in outcomes):
            raise RuntimeError("planted")
        return real(outcomes)

    real_stop_all = deploy_drive._Drive.stop_all

    def stop_all_fails(self):
        real_stop_all(self)
        raise OSError(errno.EIO, "the store did not take the role's event")

    monkeypatch.setattr(deploy_drive, "stage_fault", failing)
    monkeypatch.setattr(deploy_drive._Drive, "stop_all", stop_all_fails)
    ctx = _ctx(tmp_path, bins, mode="fill", watch=disk_watch, readiness_timeout_sec=30)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False) and "disk cap" in fault.reason


def test_a_server_that_keeps_taking_its_trees_permissions_away_passes_its_cap(tmp_path, bins, disk_watch):
    # The walk is refused in the server's own tree again and again, which only the server's own
    # processes can do: that is past the cap, not a walk to try again forever.
    ctx = _ctx(tmp_path, bins, mode="chmodrace", watch=disk_watch, readiness_timeout_sec=30)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False) and "disk cap" in fault.reason
    assert _ended(ctx, "artifact")["ending"] == "disk_limit"


#: While it holds a directory, this process may make nothing under it: the node's disk there is
#: full. Every attempt is kept.
_FULL: list[Path] = []
_FULL_ATTEMPTS: list[str] = []
_WRITES = os.O_WRONLY | os.O_RDWR | os.O_CREAT


def _refuse_on_a_full_node(event: str, args: tuple) -> None:
    if not _FULL:
        return
    if event == "open" and len(args) == 3 and isinstance(args[2], int) and args[2] & _WRITES:
        targets = [args[0]]
    elif event in ("os.mkdir", "os.symlink", "os.link", "os.rename", "os.replace"):
        targets = list(args[:2])
    else:
        return
    for target in targets:
        if isinstance(target, (str, bytes, os.PathLike)) and os.path.isabs(os.fsdecode(target)):
            if Path(os.fsdecode(target)).resolve().is_relative_to(_FULL[0]):
                _FULL_ATTEMPTS.append(f"{event} {os.fsdecode(target)}")
                raise OSError(errno.ENOSPC, "No space left on device", os.fsdecode(target))


sys.addaudithook(_refuse_on_a_full_node)


def test_a_node_disk_the_server_filled_unseen_still_drives_and_scores(tmp_path, bins, disk_watch):
    # The server holds bytes no walk can see, the size of the node's free disk, and passes both
    # gates. Nothing the runner does after the server started needs a new block of the node's
    # disk: the driver's files and its trace directory were made before the server started. Only
    # this process's writes stand for the full disk here: the driver's trace writer, which writes
    # into that directory as the drive goes and waits out a disk with no room
    # (``hls.drive.stream``), is another process, which the full disk does not refuse.
    ctx = _ctx(tmp_path, bins, mode="hide", watch=disk_watch, readiness_timeout_sec=30)
    start = ctx.start_container

    def start_then_fill(name, *args, **kwargs):
        handle = start(name, *args, **kwargs)
        if name == "artifact":
            _FULL.append(ctx.scratch.resolve())
        return handle

    ctx.start_container = start_then_fill
    _FULL_ATTEMPTS.clear()
    try:
        record = _drive(ctx, DeployDriveStage().run(ctx))
    finally:
        _FULL.clear()
    assert record.outcome == "served" and record.trace.events == 10
    assert _FULL_ATTEMPTS == []


def test_a_server_that_fills_its_writable_disk_to_the_brim_still_has_its_end_recorded_and_judged(tmp_path, bins):
    # The server fills the node's disk, where its seal's files are, once it started, then exits
    # 3. The seal made its ending records and its interceptor's status at full size before the
    # server started and only writes over them afterwards, so the server's exit is still
    # recorded and judged its own: it crashed, a scored result, never infra for a seal that
    # recorded nothing. Its seal's directory taking no new file, rename or removal stands for
    # the full disk.
    ctx = _ctx(tmp_path, bins, mode="brim")
    seal = tmp_path / "node" / "hls-100" / "roles" / "artifact" / "seal"
    try:
        record = _drive(ctx, DeployDriveStage().run(ctx))
        full = stat.S_IMODE(seal.stat().st_mode)
    finally:
        if seal.exists():
            seal.chmod(0o700)
    assert full == 0o500
    assert record.outcome == "crashed" and "status 3" in record.reason
    assert (record.artifact.ending, record.artifact.exit_code, record.artifact.ready_after_command_sec) == ("exited", 3, None)
    files = ctx.seal_files("artifact")
    assert files is not None and files.status.parent == seal
    assert payload_ending(files) == PayloadEnding(exit_code=3, interceptor_alive=True)
    sizes = {p.name: p.stat().st_size for p in seal.iterdir() if p.name != "port"}
    assert sizes == {"status.json": STATUS_FILE_BYTES, "started": 0, "ended": 512, "ended.pending": 512}


class _Hub(BaseHTTPRequestHandler):
    """The hub's resolve route for one file, as ``hf_hub_download`` asks it: its commit, its
    ETag and its bytes."""

    #: The file's bytes.
    BODY = b'{"stand-in": "tokenizer"}\n'
    #: The commit the hub names for it.
    COMMIT = "c" * 40

    def _head(self) -> None:
        self.send_response(200)
        self.send_header("X-Repo-Commit", self.COMMIT)
        self.send_header("ETag", f'"{hashlib.sha256(self.BODY).hexdigest()}"')
        self.send_header("Content-Length", str(len(self.BODY)))
        self.end_headers()

    def do_HEAD(self) -> None:
        self._head()

    def do_GET(self) -> None:
        self._head()
        self.wfile.write(self.BODY)

    def log_message(self, *args) -> None:
        pass


class _DriverDownloadingOnAFullNode(ApptainerRuntime):
    """Runs the driver's command beside a download of a hub file from *hub*, both as programs
    on a node whose disk under *full* is full (``FAKE_FULL_NODE`` of ``fake_site``), which note
    each attempt to take space there in *log*. The download starts once the driver runs, and the
    role fails if either fails."""

    def __init__(self, full: Path, log: Path, hub: str, fetched: Path) -> None:
        super().__init__()
        #: The node-local directory whose disk is full.
        self.full = full
        #: Where each refused attempt is noted.
        self.log = log
        #: The hub's endpoint.
        self.hub = hub
        #: Where the download notes the path it fetched the file to.
        self.fetched = fetched

    def argv(self, spec, *, env_file, seal_python=None):
        if spec.workdir.parent.name == "driver":
            fetch = ("from huggingface_hub import hf_hub_download; "
                     f"path = hf_hub_download('org/tool', 'tokenizer.json', endpoint={self.hub!r}); "
                     f"open({str(self.fetched)!r}, 'w').write(path)")
            script = (
                "import os, subprocess, sys\n"
                f"env = {{**os.environ, 'FAKE_FULL_NODE': {str(self.full)!r}, "
                f"'FAKE_FULL_NODE_LOG': {str(self.log)!r}}}\n"
                "driver = subprocess.Popen(sys.argv[1:], env=env)\n"
                f"fetched = subprocess.run([sys.executable, '-c', {fetch!r}], env=env).returncode\n"
                "sys.exit(driver.wait() or fetched)\n")
            spec = dataclasses.replace(spec, command=("python", "-c", script, *spec.command))
        return super().argv(spec, env_file=env_file, seal_python=seal_python)


def test_a_driver_that_downloads_from_the_hub_on_a_full_node_keeps_its_caches_on_the_store(tmp_path, bins):
    # Every node-local write that takes new space is refused once the server starts, the
    # runner's and the driver's programs' alike. The stage made the driver's trace room before
    # the server started, at the most the trace can take, and the role writes the drive's trace
    # into it in place as the drive goes. The driver's hub and model caches lie on the store's
    # filesystem, so a hub file it fetches mid-drive lands there and the drive completes.
    hub = ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
    threading.Thread(target=hub.serve_forever, daemon=True).start()
    ctx = _ctx(tmp_path, bins)
    log = tmp_path / "full-node.log"
    fetched = tmp_path / "fetched"
    ctx.launcher.runtime = _DriverDownloadingOnAFullNode(ctx.scratch.resolve(), log,
                                                f"http://127.0.0.1:{hub.server_address[1]}", fetched)
    start = ctx.start_container

    def start_then_fill(name, *args, **kwargs):
        handle = start(name, *args, **kwargs)
        if name == "artifact":
            _FULL.append(ctx.scratch.resolve())
        return handle

    ctx.start_container = start_then_fill
    _FULL_ATTEMPTS.clear()
    try:
        record = _drive(ctx, DeployDriveStage().run(ctx))
    finally:
        _FULL.clear()
        hub.shutdown()
    assert record.outcome == "served" and record.trace.events == 10
    assert _FULL_ATTEMPTS == [] and not log.exists()
    # The file went to the caches' mount, from the attempt's directory on the store, which the
    # stage then measured and removed.
    assert fetched.read_text().endswith(f"/hls/cache/hf/hub/models--org--tool/snapshots/{_Hub.COMMIT}/tokenizer.json")
    record = fake_container.records(tmp_path / "records")["driver"]
    assert f"{ctx.attempt_dir / 'driver-cache'}:/hls/cache" in record["binds"]
    [dropped] = [e for e in _events(ctx) if e["kind"] == "drive.driver_cache"]
    assert dropped["bytes"] >= len(_Hub.BODY) and dropped["removed"] is True
    assert not (ctx.attempt_dir / "driver-cache").exists()
    # Where the bytes land is all that changed: the hub stays as reachable as it was.
    assert record["env"]["HF_HUB_OFFLINE"] == "0" and record["env"]["HF_HOME"] == "/hls/cache/hf"


@pytest.mark.parametrize("ends", ["served", "crashed", "driver-failed", "error"])
def test_the_driver_caches_go_when_the_stage_ends_however_it_ends(tmp_path, bins, monkeypatch, ends):
    # They are made before the server starts; a stage that ends before that has none.
    ctx = _ctx(tmp_path, bins, mode="crash" if ends == "crashed" else "serve")
    if ends == "driver-failed":
        ctx.launcher.runtime = _RefusingDriver()
    if ends == "error":
        def planted(self, artifact, driver):
            raise RuntimeError("planted")

        monkeypatch.setattr(deploy_drive._Drive, "drive", planted)
    try:
        DeployDriveStage().run(ctx)
    except (StageFault, RuntimeError):
        assert ends in ("driver-failed", "error")
    else:
        assert ends in ("served", "crashed")
    assert not (ctx.attempt_dir / "driver-cache").exists()
    [dropped] = [e for e in _events(ctx) if e["kind"] == "drive.driver_cache"]
    assert dropped["removed"] is True and isinstance(dropped["bytes"], int)


class _DriverWaitingForSpace(ApptainerRuntime):
    """Runs the driver's command as one that sends two requests, the second of which makes a
    ``spill`` server pass its cap, and then needs the node's disk: it fails as the real driver
    would on a full disk while the server's bytes stay there, and exits 0 once they are gone."""

    def __init__(self, held: Path) -> None:
        super().__init__()
        #: The server's file that fills the node's disk.
        self.held = held

    def argv(self, spec, *, env_file, seal_python=None):
        if spec.workdir.parent.name == "driver":
            url = spec.command[spec.command.index("--artifact-url") + 1]
            script = (
                "import json, os, sys, time, urllib.request\n"
                f"url = {url!r} + '/v1/chat/completions'\n"
                "body = json.dumps({'model': 'm', 'messages': [{'role': 'user', 'content': 'q'}]}).encode()\n"
                "for _ in range(2):\n"
                "    try:\n"
                "        request = urllib.request.Request(url, body, {'Content-Type': 'application/json'})\n"
                "        urllib.request.urlopen(request, timeout=10).read()\n"
                "    except Exception:\n"
                "        pass\n"
                "deadline = time.time() + 10\n"
                f"while os.path.exists({str(self.held)!r}) and time.time() < deadline:\n"
                "    time.sleep(0.05)\n"
                f"if os.path.exists({str(self.held)!r}):\n"
                "    print('OSError: [Errno 28] No space left on device', file=sys.stderr)\n"
                "    sys.exit(1)\n")
            spec = dataclasses.replace(spec, command=("python", "-c", script))
        return super().argv(spec, env_file=env_file, seal_python=seal_python)


def test_a_server_killed_at_its_cap_during_the_drive_frees_the_node_before_the_driver_needs_it(
        tmp_path, bins, disk_watch):
    # The walk kills the server past its cap during the drive. Its
    # writable directories go at once, so the driver's need for the node's disk is met, and the
    # stage ends with the contestant's fault, never infra.
    ctx = _ctx(tmp_path, bins, mode="spill", watch=disk_watch)
    ctx.launcher.runtime = _DriverWaitingForSpace(ctx.scratch / "roles" / "artifact" / "engine-cache" / "tmp" / "fill")
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False) and "disk cap" in fault.reason
    assert _ended(ctx, "artifact")["ending"] == "disk_limit" and _ended(ctx, "driver")["exit_code"] != 1
    [cleared] = [e for e in _events(ctx) if e["kind"] == "drive.artifact_cleared"]
    assert cleared["at"] < _ended(ctx, "driver")["at"]
    assert not (ctx.scratch / "roles" / "artifact" / "engine-cache").exists()


def test_every_node_local_file_the_driver_needs_is_made_before_the_artifact_starts(tmp_path, bins):
    ctx = _ctx(tmp_path, bins)
    role = ctx.scratch / "roles" / "driver"
    made: dict[str, bool] = {}
    report: list[tuple[int, int]] = []
    start = ctx.start_container

    def look_then_start(name, *args, **kwargs):
        if name == "artifact":
            for part in ("tmp", "task-ref.json", "wd/tmp", "wd/var_tmp", "wd/home", "container.env",
                         f"{deploy_drive.LOCAL_DIR}/{CHUNKS_ROOM}", f"{deploy_drive.LOCAL_DIR}/{SUMMARY_ROOM}"):
                made[part] = (role / part).exists()
            for part in ("outputs.json", deploy_drive.STREAM_RECORD_FILE):
                st = (role / part).stat()
                report.append((st.st_size, st.st_ino))
        return start(name, *args, **kwargs)

    ctx.start_container = look_then_start
    outputs = DeployDriveStage().run(ctx)
    assert made and all(made.values()), made
    # The two reports were set aside whole and written in place, cut to what the driver wrote.
    for part, size, (made_size, inode) in zip(("outputs.json", deploy_drive.STREAM_RECORD_FILE),
                                              (deploy_drive.OUTPUTS_REPORT_BYTES, deploy_drive.STREAM_RECORD_BYTES),
                                              report, strict=True):
        after = (role / part).stat()
        assert made_size == size and after.st_ino == inode and 0 < after.st_size < size, part
    assert outputs["trace"] == ctx.output_key("trace/END.json")


def test_the_driver_writes_its_trace_into_a_room_made_before_the_artifact_starts(tmp_path, bins):
    """The role writes the drive's trace, as the drive goes, into a room on the node's disk,
    which the stage makes in the driver's role directory before the artifact starts, at the most
    the trace's chunks and summary can take with its events' JSON at the cap and the requests
    beside it, and mounts read-write at ``LOCAL``: the role's ``--local``. Its ``--out`` is the
    attempt's outputs on the store, where it publishes the trace once the drive is over. The
    stage removes the room once its roles are stopped, and says so."""
    ctx = _ctx(tmp_path, bins)
    local = ctx.role_dir("driver") / deploy_drive.LOCAL_DIR
    seen: list[dict[str, tuple[int, int]] | None] = []
    start = ctx.start_container

    def look_then_start(name, *args, **kwargs):
        if name == "artifact":
            seen.append({p.name: (p.stat().st_size, p.stat().st_blocks * 512) for p in local.iterdir()}
                        if local.is_dir() else None)
        return start(name, *args, **kwargs)

    ctx.start_container = look_then_start
    outputs = DeployDriveStage().run(ctx)
    room = room_bytes(trace_event_bytes(DRIVE_CAP, DRIVE_REQUESTS))
    [made] = seen
    assert made is not None and {name: size for name, (size, _) in made.items()} == room
    assert all(blocks >= size for size, blocks in made.values())
    # On the node's disk, never on the store's filesystem.
    assert local.resolve().is_relative_to(ctx.scratch.resolve())
    assert not local.resolve().is_relative_to(ctx.attempt_dir.resolve())
    driver = fake_container.records(tmp_path / "records")["driver"]
    assert f"{local}:{deploy_drive.LOCAL}" in driver["binds"]
    assert not [b for b in driver["binds"] if b.startswith(f"{local}:") and b.endswith(":ro")]
    command = driver["command"]
    assert (_flag(command, "--local"), _flag(command, "--out")) == (deploy_drive.LOCAL, "/hls/out/trace")
    assert read_end(ctx.store.path(outputs["trace"]).parent).events == 10
    [dropped] = [e for e in _events(ctx) if e["kind"] == "drive.trace_room"]
    assert (dropped["path"], dropped["removed"], dropped["error"]) == (str(local), True, None)
    assert not local.exists()


@pytest.mark.parametrize("ending", ["crashed", "raised"])
def test_the_trace_room_is_removed_however_the_stage_ends(tmp_path, bins, monkeypatch, ending):
    ctx = _ctx(tmp_path, bins, mode="crash" if ending == "crashed" else "serve")
    local = ctx.role_dir("driver") / deploy_drive.LOCAL_DIR
    if ending == "raised":
        def fails(self, *args, **kwargs):
            raise RuntimeError("planted")

        monkeypatch.setattr(deploy_drive._Drive, "start_driver", fails)
        with pytest.raises(RuntimeError, match="planted"):
            DeployDriveStage().run(ctx)
    else:
        assert _drive(ctx, DeployDriveStage().run(ctx)).outcome == "crashed"
    assert not local.exists()
    [dropped] = [e for e in _events(ctx) if e["kind"] == "drive.trace_room"]
    assert (dropped["removed"], dropped["error"]) == (True, None)


def test_a_trace_room_the_disk_cannot_hold_is_infra_retried_before_the_artifact_starts(tmp_path, bins,
                                                                                       monkeypatch):
    """The disk gives the room's first file its blocks and has none for the second: the stage
    starts no server, removes the half-made room, and is infra, retried on another try."""
    ctx = _ctx(tmp_path, bins)
    local = ctx.role_dir("driver") / deploy_drive.LOCAL_DIR
    real, calls = os.posix_fallocate, []

    def second_refused(fd, offset, size):
        calls.append(size)
        if len(calls) == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(fd, offset, size)

    monkeypatch.setattr(trace_store.os, "posix_fallocate", second_refused)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "trace's room" in fault.reason and "No space left on device" in fault.reason
    assert fault.evidence["room"] == str(local)
    assert int(fault.evidence["room_bytes"]) == sum(room_bytes(trace_event_bytes(DRIVE_CAP, DRIVE_REQUESTS)).values())
    assert len(calls) == 2 and "artifact" not in _started(ctx)
    assert not local.exists()
    [dropped] = [e for e in _events(ctx) if e["kind"] == "drive.trace_room"]
    assert (dropped["removed"], dropped["error"]) == (True, None)


def test_the_submission_is_pinned_as_the_contestants_image(tmp_path, bins):
    ctx = _ctx(tmp_path, bins)
    asked: list[dict] = []
    real = ctx.pin_file

    def recording(path, **kwargs):
        asked.append(kwargs)
        return real(path, **kwargs)

    ctx.pin_file = recording
    DeployDriveStage().run(ctx)
    assert asked == [{"sha256": _submission_sha(ctx), "contestant": True}]


def test_a_node_whose_size_cannot_be_told_refuses_the_artifacts_step_before_any_role(tmp_path, bins, monkeypatch):
    # No evaluation count is guessed for it: the step's reserve and floor would be guesses too.
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: 1_000_000)
    ctx = _ctx(tmp_path, bins, launcher=LocalLauncher(node=NodeSize(gpus=None, cpus=None)))
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    why = "the node's GPU count is unknown, so how many evaluations of this size it holds is unknown too"
    assert fault.reason == f"no contestant step of stage {ctx.stage.name} can run on this node: {why}"
    assert ctx.launcher.launched == []
    [event] = [e for e in _events(ctx) if e["kind"] == "process_ceiling"]
    assert (event["evaluations"], event["evaluations_unknown"], event["ceiling"]) == (None, why, None)


class _EightGpuNode(LocalLauncher):
    """A local launcher whose node has eight GPUs and 64 CPUs."""

    def node_size(self) -> NodeSize:
        return NodeSize(gpus=8, cpus=64)


def test_the_reserve_is_kept_for_every_evaluation_the_node_holds(tmp_path, bins, monkeypatch):
    # The node holds eight evaluations of the artifact's one GPU, each with a harness of its own.
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: 1_000_000)
    ctx = _ctx(tmp_path, bins, launcher=_EightGpuNode(), site={"nproc_max": 2_000_000})
    record = _drive(ctx, DeployDriveStage().run(ctx))
    reserve = ctx.plan.settings.run.nproc_reserve
    want = ProcessCeiling(hard=1_000_000, reserve=reserve, evaluations=8, evaluations_unknown=None,
                          uncapped=1_000_000 - 8 * reserve, nproc_max=2_000_000, ceiling=1_000_000 - 8 * reserve,
                          floor=8 * FLOOR_PER_EVALUATION)
    assert record.process_ceiling == want
    ceilings = {s.name: s.nproc for s in ctx.launcher.launched if s.name in ("seal-probe", "artifact", "driver")}
    assert ceilings == {"seal-probe": None, "artifact": want.ceiling, "driver": None}
    [event] = [e for e in _events(ctx) if e["kind"] == "process_ceiling"]
    assert (event["ceiling"], event["floor"], event["evaluations_unknown"]) == (want.ceiling, want.floor, None)


def test_the_artifacts_step_runs_under_the_sites_cap_where_the_uids_pool_is_larger(tmp_path, bins, monkeypatch):
    # A hard limit leaves a pool larger than the node's pid_max; the cap of a site that sets none
    # (site.nproc_max; 131,072 on this node) holds the server's step under it, and drive.json says
    # which bound it ran under.
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: 9_287_489)
    ctx = _ctx(tmp_path, bins, launcher=_EightGpuNode())
    record = _drive(ctx, DeployDriveStage().run(ctx))
    reserve = ctx.plan.settings.run.nproc_reserve
    assert ctx.plan.settings.site.nproc_max is None
    assert record.process_ceiling == ProcessCeiling(hard=9_287_489, reserve=reserve, evaluations=8,
                                                    evaluations_unknown=None, uncapped=9_287_489 - 8 * reserve,
                                                    nproc_max=131_072, ceiling=131_072,
                                                    floor=8 * FLOOR_PER_EVALUATION)
    assert next(s.nproc for s in ctx.launcher.launched if s.name == "artifact") == 131_072
    [event] = [e for e in _events(ctx) if e["kind"] == "process_ceiling"]
    assert (event["uncapped"], event["nproc_max"], event["ceiling"]) == (9_287_489 - 8 * reserve, 131_072, 131_072)


def test_the_fixtures_process_limit_decides_not_the_hosts(tmp_path, bins, monkeypatch):
    # A host whose hard limit is 4,096 would refuse every drive of eight evaluations at the floor.
    real = nproc.resource.getrlimit
    monkeypatch.setattr(nproc.resource, "getrlimit",
                        lambda which: (4096, 4096) if which == nproc.resource.RLIMIT_NPROC else real(which))
    assert nproc.hard_limit() == 4096
    ctx = _ctx(tmp_path, bins)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served" and record.process_ceiling.hard is None


def test_an_unlimited_process_limit_leaves_the_sites_cap(tmp_path, bins, monkeypatch):
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: None)
    ctx = _ctx(tmp_path, bins)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert (record.process_ceiling.uncapped, record.process_ceiling.ceiling) == (None, 131_072)
    assert next(s.nproc for s in ctx.launcher.launched if s.name == "artifact") == 131_072


class _CpuNode(LocalLauncher):
    """A local launcher whose node is a CPU partition's: no GPU and 192 CPUs."""

    def node_size(self) -> NodeSize:
        return NodeSize(gpus=0, cpus=192)


def test_a_cpu_only_server_of_eight_cpus_on_a_192_cpu_node_is_served_under_the_default_cap(tmp_path, bins,
                                                                                          monkeypatch):
    # The node holds 24 such evaluations, a floor of 98,304 tasks, under the default 131,072.
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: 9_287_489)
    artifact = ARTIFACT_SHAPE.model_copy(update={"gpus": 0, "gpu_kind": None, "cpus": 7})
    roles = concurrent([artifact, DRIVER_SHAPE])
    shape = roles.model_copy(update={"memory_gib": roles.memory_gib + RUNNER_GIB,
                                     "disk_gib": roles.disk_gib + EXTRACTED_GIB + DATA_GIB})
    # Its build, on a node of the same size, recorded its share among the 27 servers of its 7 CPUs.
    served = ServedUcounts(limits=served_limits(DECLARED, 192 // 7))
    ctx = _ctx(tmp_path, bins, launcher=_CpuNode(), shape=shape, artifact_shape=artifact.model_dump(), served=served)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served"
    assert (record.process_ceiling.evaluations, record.process_ceiling.floor, record.process_ceiling.ceiling) == (
        24, 24 * FLOOR_PER_EVALUATION, 131_072)


class _SixteenGpuNode(LocalLauncher):
    """A local launcher whose node has sixteen GPUs and 128 CPUs."""

    def node_size(self) -> NodeSize:
        return NodeSize(gpus=16, cpus=128)


def test_a_sites_cap_under_the_floor_of_the_nodes_servers_is_infra_before_any_role(tmp_path, bins, monkeypatch):
    # A node that holds sixteen evaluations has a floor of 65,536 tasks, one more than the cap.
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: 9_287_489)
    ctx = _ctx(tmp_path, bins, launcher=_SixteenGpuNode(), site={"nproc_max": 16 * FLOOR_PER_EVALUATION - 1})
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == (f"no contestant step of stage {ctx.stage.name} can run on this node: site.nproc_max of "
                            "65535 tasks is under the "
                            "floor of 65536, 4096 for each of the 16 evaluations' servers the node holds; raise "
                            "site.nproc_max to at least 65536")
    assert ctx.launcher.launched == []


@pytest.mark.parametrize("setting, floor", [("sandbox_namespaces", 16), ("sandbox_inotify_instances", 2),
                                            ("sandbox_inotify_watches", 8192)])
def test_a_sites_per_uid_count_cap_under_its_floor_refuses_the_plan(tmp_path, setting, floor):
    decls = [StageDecl("deploy_drive", "deploy_drive", SHAPE, _params())]
    noop_plan(tmp_path / "ok", decls=decls, site={setting: floor})
    with pytest.raises(PlanError, match=rf"^site.{setting} of {floor - 1} is under the floor of {floor}; raise "
                                        rf"site.{setting} to at least {floor}$"):
        noop_plan(tmp_path / "low", decls=decls, site={setting: floor - 1})


def test_a_sites_cap_under_one_evaluations_floor_refuses_the_plan(tmp_path):
    # A plan does not know its nodes: it refuses only a cap no node can run a contestant's step
    # under, and an allocation checks the cap against its own node's evaluations when it starts.
    decls = [StageDecl("deploy_drive", "deploy_drive", SHAPE, _params())]
    noop_plan(tmp_path / "ok", decls=decls, site={"nproc_max": FLOOR_PER_EVALUATION})
    with pytest.raises(PlanError) as e:
        noop_plan(tmp_path / "low", decls=decls, site={"nproc_max": FLOOR_PER_EVALUATION - 1})
    assert str(e.value) == (f"no node can run a contestant's step under site.nproc_max: "
                            f"{nproc_max_problem(FLOOR_PER_EVALUATION - 1, 1)} (an allocation checks it against the "
                            "evaluations of each stage's size its node holds when it starts)")


@pytest.mark.parametrize("hard", [900, 8 * 1024 + 8 * FLOOR_PER_EVALUATION - 1])
def test_a_ceiling_under_the_floor_of_the_nodes_servers_is_infra_before_any_role(tmp_path, bins, monkeypatch, hard):
    # Eight servers of the uid share one pool; under 4,096 tasks each they would crash, and the
    # crashes would be scored.
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: hard)
    ctx = _ctx(tmp_path, bins)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False) and "under the floor of 32768" in fault.reason
    assert ctx.launcher.launched == []


#: The process ceiling of the stage of ``_ctx`` under a hard limit of a million: eight evaluations
#: on a node the local launcher cannot size, each keeping the default reserve, under the site's
#: default cap of 131,072 tasks.
_CEILING = min(1_000_000 - 8 * 1024, 131_072)


def _starving(tmp_path, bins, monkeypatch, disk_watch, *, others: int, own: int, mode: str,
              **params) -> StageContext:
    """A stage whose uid holds *others* tasks on the node beside its artifact's step once that
    starts (other jobs'), and *own* in the step while it runs, which go with it. The walks and the
    count when the artifact is reported read the same pool."""
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: 1_000_000)
    pool = {"others": 500, "server": None}

    def count(uid):
        server = pool["server"]
        return pool["others"] + (own if server is not None and server.poll() is None else 0)

    monkeypatch.setattr(memory_watch, "uid_tasks", count)
    monkeypatch.setattr(stage_mod, "uid_tasks", count)
    ctx = _ctx(tmp_path, bins, mode=mode, watch=disk_watch, launcher=LocalLauncher(cgroup_root=tmp_path / "cg"),
               **params)
    start = ctx.start_container

    def start_then_fill(name, *args, **kwargs):
        handle = start(name, *args, **kwargs)
        if name == "artifact":
            pool.update(others=others, server=handle)
        return handle

    ctx.start_container = start_then_fill
    return ctx


@pytest.mark.parametrize("mode, params", [
    ("fade", {}), ("fade0", {}), ("hang", {"readiness_timeout_sec": 2}), ("die", {}), ("die0", {})],
    ids=["crashed", "exited-0-before-ready", "not-ready", "died-in-the-drive", "exited-0-in-the-drive"])
def test_a_server_that_ends_while_others_fill_the_uids_pool_was_starved_and_is_infra(
        tmp_path, bins, monkeypatch, disk_watch, mode, params):
    others = int(0.96 * _CEILING)
    ctx = _starving(tmp_path, bins, monkeypatch, disk_watch, others=others, own=40, mode=mode, **params)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "starved" in fault.reason
    assert [v for k, v in fault.evidence.items() if k.endswith("others_tasks")] == [str(others)]


def _uncountable(uid: int) -> int:
    """A count of the uid's tasks that fails, as listing ``/proc`` out of descriptors would."""
    raise OSError(errno.EMFILE, "Too many open files", "/proc")


@pytest.mark.parametrize("mode, params", [
    ("fade", {}), ("fade0", {}), ("hang", {"readiness_timeout_sec": 2}), ("die", {}), ("die0", {})],
    ids=["crashed", "exited-0-before-ready", "not-ready", "died-in-the-drive", "exited-0-in-the-drive"])
def test_a_server_that_ends_while_the_uids_tasks_cannot_be_counted_is_infra(
        tmp_path, bins, monkeypatch, disk_watch, mode, params):
    # The walks see room; the count once the server's step is gone, which alone rules starvation
    # out, fails. Its end may be the others', so it is retried rather than scored.
    ctx = _starving(tmp_path, bins, monkeypatch, disk_watch, others=500, own=40, mode=mode, **params)
    monkeypatch.setattr(stage_mod, "uid_tasks", _uncountable)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "could not be counted" in fault.reason
    assert [v for k, v in fault.evidence.items() if k.endswith("others_tasks")] == ["uncounted"]


def test_a_served_artifacts_ceiling_counts_every_task_of_the_uid_whatever_the_sites_accounting(
        tmp_path, bins, monkeypatch, disk_watch):
    # Its ceiling is set before pasta makes the seal's namespace, so on a kernel that counts the
    # build sandbox's commands per namespace the uid's other tasks still fill this one's pool.
    others = int(0.96 * _CEILING)
    ctx = _starving(tmp_path, bins, monkeypatch, disk_watch, others=others, own=40, mode="fade",
                    site={"nproc_accounting": "user_namespace"})
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "starved" in fault.reason
    assert [v for k, v in fault.evidence.items() if k.endswith(".accounting") or k == "accounting"] == ["uid"]


def test_a_server_that_serves_the_whole_drive_is_published_whatever_the_pool_held(tmp_path, bins, monkeypatch,
                                                                                    disk_watch):
    # It ended no failure: stopped as planned once the drive was over, it is not asked whether
    # others starved it, however full they left the pool.
    ctx = _starving(tmp_path, bins, monkeypatch, disk_watch, others=int(0.96 * _CEILING), own=40, mode="serve")
    assert _drive(ctx, DeployDriveStage().run(ctx)).outcome == "served"


@pytest.mark.parametrize("mode, params, outcome", [
    ("fade", {}, "crashed"), ("fade0", {}, "crashed"), ("hang", {"readiness_timeout_sec": 2}, "not_ready"),
    ("die", {}, "served"), ("die0", {}, "served")],
    ids=["crashed", "exited-0-before-ready", "not-ready", "died-in-the-drive", "exited-0-in-the-drive"])
def test_a_server_whose_own_tasks_filled_the_pool_keeps_its_result(tmp_path, bins, monkeypatch, disk_watch, mode,
                                                                     params, outcome):
    # The walks see the pool full, half of it the server's; once its step is gone the others
    # left room.
    ctx = _starving(tmp_path, bins, monkeypatch, disk_watch, others=_CEILING // 2, own=int(0.46 * _CEILING),
                    mode=mode, **params)
    assert _drive(ctx, DeployDriveStage().run(ctx)).outcome == outcome


def test_a_server_that_reached_its_pids_limit_beside_a_driver_that_failed_leaves_the_stage_infra(tmp_path, bins):
    # The stage is the contestant's only when every failed role's fault is: a driver that failed
    # as infra beside a server at its pids limit leaves it infra, retried, naming both.
    cgroups = tmp_path / "cg"
    ctx = _ctx(tmp_path, bins, launcher=LocalLauncher(cgroup_root=cgroups))
    ctx.launcher.runtime = _RefusingDriver()
    start = ctx.start_container

    def start_then_hit(name, *args, **kwargs):
        handle = start(name, *args, **kwargs)
        if name == "artifact":
            (cgroups / "cg-artifact" / "pids.events").write_text("max 1\n")
        return handle

    ctx.start_container = start_then_hit
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "process limit" in fault.reason
    assert fault.evidence["roles"] == "artifact,driver"
    assert (fault.evidence["artifact.ending"], fault.evidence["driver.ending"]) == ("contestant_pids", "exited")


def test_a_server_at_its_pids_limit_whose_log_capper_died_is_still_the_contestants(tmp_path, bins):
    # Its processes may have left its capper unable to fork, so the pids limit is read before a
    # lost log, which would make its end infra.
    cgroups = tmp_path / "cg"
    ctx = _ctx(tmp_path, bins, launcher=LocalLauncher(cgroup_root=cgroups, log_keep_bytes=4096))
    ctx.launcher.runtime = _RefusingDriver()
    start = ctx.start_container

    def start_then_hit(name, *args, **kwargs):
        handle = start(name, *args, **kwargs)
        if name == "artifact":
            (cgroups / "cg-artifact" / "pids.events").write_text("max 1\n")
            handle.log_cappers[1].send_signal(signal.SIGKILL)
        return handle

    ctx.start_container = start_then_hit
    fault = _fault(ctx).fault
    # The server's own end is its pids limit, never its lost log; the driver's refusal beside it
    # leaves the stage infra.
    assert "process limit" in fault.reason and fault.evidence["artifact.ending"] == "contestant_pids"
    [ended] = [e for e in read_events(ctx.attempt_dir / "events.jsonl")
               if e["kind"] == "role_ended" and e["role"] == "artifact"]
    assert ended["ending"] == "contestant_pids"


@pytest.mark.parametrize("artifact", ["package", "measured"])
def test_every_copy_and_pin_comes_before_the_stages_own_roles(tmp_path, bins, artifact):
    # A copy or an extraction runs alone, so the driver's data, its full tool bundle among it,
    # is copied before the server starts, and the submission is pinned straight from the store.
    changes = {} if artifact == "package" else {"artifact": {
        "kind": "image", "image": "measured", "entrypoint": ["python", "/workspace/server.py", "--mode", "serve"],
        "runtime_env": {}}}
    ctx = _ctx(tmp_path, bins, tools=True, **changes)
    order = CallOrder(ctx)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served" and f"tools/{TOOL_KEYS['full']}" in ctx.cache_entries
    assert f"tools/{TOOL_KEYS['dev']}" not in ctx.cache_entries
    # The copies of the weights, the rows and the full tool bundle, the two extractions, the seal
    # probe, the artifact and the driver: the most the stage declares one attempt of a drive other
    # than the measured baseline's starts, which sizes finalize's time.
    assert len(ctx.launcher.launched) == DeployDriveStage.max_roles(ctx.params) == 8
    assert order.while_running() == [] and order.roles_before_staged() == []
    assert [name for kind, name, _ in order.calls if kind == "role"] == ["seal-probe", "artifact", "driver"]
    driver = fake_container.records(tmp_path / "records")["driver"]
    assert any(b.endswith(":/data/tools/tiny:ro") for b in driver["binds"])


def test_a_contestant_host_role_is_refused(tmp_path, bins):
    ctx = _ctx(tmp_path, bins)
    with pytest.raises(ValueError, match="runs only in a container"):
        ctx.start_role("server", [sys.executable, "-c", "pass"], DRIVER_SHAPE, owner="contestant")
    assert ctx.launcher.launched == []


def test_a_measured_image_is_served_from_the_cache_under_the_stages_variables(tmp_path, bins):
    artifact = {"kind": "image", "image": "measured", "entrypoint": ["python", "/workspace/server.py", "--mode", "serve"],
                "runtime_env": {"TORCH_CUDA_ARCH_LIST": "7.0", "VLLM_CACHE_ROOT": "/root/.cache/vllm",
                                "HF_HUB_OFFLINE": "0", "ENGINE_FLAG": "1"}}
    network = NetworkPolicy(domains=AccessList(mode="allow", entries=["example.com"]))
    ctx = _ctx(tmp_path, bins, artifact=artifact, network=network.model_dump(mode="json"))
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served"
    assert _started(ctx) == ["copy", "copy", "extract", "extract", "seal-probe", "artifact", "driver"]
    # The artifact's own runtime policy, vouching for no host.
    assert SealPolicy.model_validate_json(_artifact_argv(ctx)[8]) == SealPolicy(network=network)
    assert ctx.image_digests["artifact"] == _cached_sha(ctx, MEASURED) == record.artifact.image_sha256
    assert f"images/{MEASURED}" in ctx.cache_entries
    art = fake_container.records(tmp_path / "records")["artifact"]
    # The image's own variables reach the server, below the stage's: the GPU's arch, the per-run
    # engine cache and the offline hub win.
    env = art["env"]
    assert (env["ENGINE_FLAG"], env["TORCH_CUDA_ARCH_LIST"], env["VLLM_CACHE_ROOT"], env["HF_HUB_OFFLINE"]) == (
        "1", "9.0", "/engine-cache/vllm", "1")
    assert art["command"] == ["python", "/workspace/server.py", "--mode", "serve", "--host", "0.0.0.0", "--port", "8000"]


# ---- the server's engine cache ---------------------------------------

#: The measured baseline as its drives serve it: the fake server, which with an empty engine cache
#: builds its kernel for a minute before it listens, past the fixtures' 5 s readiness budget.
JIT_BASELINE = {"kind": "image", "image": "measured", "entrypoint": ["python", "/workspace/server.py", "--mode", "serve"],
                "runtime_env": {}}
JIT_SERVE = ["--jit-sec", "60"]
#: The measured baseline's drive parameters: its engine cache owned by its warm stage.
BASELINE_DRIVE = {"artifact": JIT_BASELINE, "serve_command": JIT_SERVE, "measured_baseline": True,
                  "engine_cache_owner": "warm_baseline"}
#: The provisioned weights, by the repo and revision an engine cache key digests.
WEIGHTS = WeightsRef(repo=MODEL, revision=REVISION, key=weights_key(MODEL, REVISION), entry_sha256="0" * 64,
                     disk_bytes=0)


def _left(ctx: StageContext, record: EngineCacheRecord) -> None:
    """The warm stage of *ctx*'s plan published, *record* its ``engine-cache.json``, among *ctx*'s
    inputs."""
    out = ctx.plan.layout.attempt_file("warm_baseline", 1, f"outputs/{ENGINE_CACHE_FILE}")
    ctx.store.write_bytes(out, record.model_dump_json().encode())
    ctx.inputs[ctx.plan.layout.manifest("warm_baseline")] = _manifest(ctx.plan, ctx.store, "warm_baseline",
                                                                       {"engine_cache": out})


def _warm_entry(ctx: StageContext, *, published: bool = True) -> tuple[str, CacheEntry | None]:
    """The measured baseline's engine cache key, as the drive of *ctx* computes it when it serves
    the baseline and as ``BASELINE_DRIVE`` does otherwise, and the entry the warm stage of *ctx*'s
    plan left under it (``_left``): built once, the kernel beside its object file and the logs of
    its one build and its one load; with *published* False, a warm stage whose publish failed and
    that left none."""
    params = ctx.params if ctx.params.measured_baseline else DeployDriveParams.model_validate(_params(**BASELINE_DRIVE))
    key = served_cache_key(params, _cached_sha(ctx, MEASURED), WEIGHTS)
    entry = None
    if published:
        built = {fake_server.KERNEL: "kernel", fake_server.KERNEL_OBJECT: "kernel object",
                 fake_server.COMPILES: "kernel\n", fake_server.LOADS: "load\n"}

        def fill(d: Path) -> None:
            for rel, text in built.items():
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                (d / rel).write_text(text)

        entry = populate(ctx.cache, ENGINE_CACHE_KIND, key, fill, holder=Holder.here("warm_baseline")).entry
        _left(ctx, EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry), published=True))
    else:
        _left(ctx, EngineCacheRecord(key=key, seeded_from=None, unpublished="the publish failed: refused"))
    return key, entry


def test_a_drive_of_the_measured_baseline_starts_from_the_entry_its_warm_stage_left(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, tools=True, **BASELINE_DRIVE)
    key, entry = _warm_entry(ctx)
    assert entry is not None
    order = CallOrder(ctx)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    # A server that would build for a minute against its 5 s budget found its kernel built and was
    # ready at once, and the record names the entry it started from.
    assert record.outcome == "served" and record.artifact.ready_after_command_sec < 5
    assert record.artifact.engine_cache == EngineCacheRecord(key=key, seeded_from=entry_sha256(entry))
    assert ctx.cache_entries[f"{ENGINE_CACHE_KIND}/{key}"] == entry_sha256(entry)
    # It ran on a writable copy of the entry made for this attempt: it built nothing and noted its
    # load there, and the entry itself is as the warm stage left it.
    cache = ctx.role_dir("artifact") / "engine-cache"
    assert (cache / fake_server.COMPILES).read_text() == "kernel\n"
    assert (cache / fake_server.LOADS).read_text() == "load\nload\n"
    assert (entry.path / fake_server.LOADS).read_text() == "load\n"
    binds = fake_container.records(tmp_path / "records")["artifact"]["binds"]
    assert f"{cache}:/engine-cache" in binds and not any(str(entry.path) in b for b in binds)
    # The copy is made with the other copies and extractions, before any role starts.
    assert order.staged()[-1] == "seed" and order.roles_before_staged() == [] and order.while_running() == []
    assert _started(ctx) == ["copy", "copy", "copy", "extract", "extract", "copy", "seal-probe", "artifact",
                             "driver"]
    # The copies of the data, the two extractions, the seed's copy, the seal probe, the artifact
    # and the driver: the most the stage declares one attempt starts.
    assert len(ctx.launcher.launched) == DeployDriveStage.max_roles(ctx.params) == 9


def test_a_contestants_server_starts_with_an_empty_engine_cache(tmp_path, bins):
    # Its own build and JIT time count against its readiness budget: it is given no entry, though
    # the store holds the measured baseline's and the plan names its warm stage.
    ctx = _ctx(tmp_path, bins, mode="serve --jit-sec 0.5")
    _, entry = _warm_entry(ctx)
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served" and record.artifact.engine_cache is None
    cache = ctx.role_dir("artifact") / "engine-cache"
    assert (cache / fake_server.COMPILES).read_text() == "kernel\n"
    assert (cache / fake_server.LOADS).read_text() == "load\n"
    assert not any(e.startswith(f"{ENGINE_CACHE_KIND}/") for e in ctx.cache_entries)
    assert _started(ctx) == ["copy", "copy", "extract", "extract", "seal-probe", "artifact", "driver"]
    assert (entry.path / fake_server.LOADS).read_text() == "load\n"


def test_a_baseline_drive_whose_warm_stage_left_no_entry_is_refused_before_any_role(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, **BASELINE_DRIVE)
    _warm_entry(ctx, published=False)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == ("the measured baseline's warm stage warm_baseline left no engine cache entry to start its "
                            "server from")
    assert _started(ctx) == ["copy", "copy", "extract", "extract"]


def test_a_baseline_drive_is_seeded_from_its_warm_stages_build_and_no_other(tmp_path, bins):
    # The entry under the key is not the build the warm stage left, as if published again since:
    # the copy is refused, and a retry would read the same entry.
    ctx = _ctx(tmp_path, bins, **BASELINE_DRIVE)
    key, entry = _warm_entry(ctx)
    assert entry is not None
    _left(ctx, EngineCacheRecord(key=key, seeded_from=None, entry="f" * 64, published=True))
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == (f"cache entry {ENGINE_CACHE_KIND}/{key} has ENTRY.json digest {entry_sha256(entry)}, not "
                            f"the {'f' * 64} its producer recorded")
    assert "artifact" not in _started(ctx)


def test_a_baseline_drive_whose_warm_stage_keyed_its_entry_otherwise_is_refused(tmp_path, bins):
    # The two stages would key one engine differently, and every retry would too: refused before
    # anything is copied or started, naming both keys.
    ctx = _ctx(tmp_path, bins, **BASELINE_DRIVE)
    key, entry = _warm_entry(ctx)
    assert entry is not None
    other = f"{key[:-16]}{'0' * 16}"
    _left(ctx, EngineCacheRecord(key=other, seeded_from=None, entry=entry_sha256(entry), published=True))
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == (f"the engine's owner warm_baseline left the engine cache entry {other}, and this stage "
                            f"keys the engine {key}")
    assert _started(ctx) == ["copy", "copy", "extract", "extract"]


def test_a_baseline_drive_that_is_not_ready_still_names_the_entry_it_started_from(tmp_path, bins):
    hanging = {**JIT_BASELINE, "entrypoint": ["python", "/workspace/server.py", "--mode", "hang"]}
    ctx = _ctx(tmp_path, bins, readiness_timeout_sec=1, **{**BASELINE_DRIVE, "artifact": hanging})
    key, entry = _warm_entry(ctx)
    assert entry is not None
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "not_ready"
    assert record.artifact.engine_cache == EngineCacheRecord(key=key, seeded_from=entry_sha256(entry))


def test_a_servers_build_environment_is_the_one_its_engine_cache_key_digests():
    # The build environment comes after the image's and the mounts' variables, so a server gets
    # exactly the environment its engine cache key digests, whatever its image sets.
    build = engine_build_env("H200")
    clash = {name: "/elsewhere" for name in build}
    env = deploy_drive.artifact_env(runtime_env={**clash, "ENGINE_FLAG": "1"}, mounts_env={"HF_HUB_OFFLINE": "1"},
                                    gpu_kind="H200")
    assert {name: env[name] for name in build} == build
    assert env == {**build, "ENGINE_FLAG": "1", "HF_HUB_OFFLINE": "1"}
    assert deploy_drive.artifact_env(runtime_env={}, mounts_env={}, gpu_kind=None) == engine_build_env(None)
    assert "TORCH_CUDA_ARCH_LIST" not in engine_build_env(None)


def test_only_a_measured_baseline_drive_is_seeded():
    # A drive of the measured baseline, and only one, names the warm stage that owns its entry.
    with pytest.raises(ValueError, match="engine_cache_owner"):
        DeployDriveParams.model_validate(_params(**{**BASELINE_DRIVE, "engine_cache_owner": None}))
    with pytest.raises(ValueError, match="engine_cache_owner"):
        DeployDriveParams.model_validate(_params(engine_cache_owner="warm_baseline"))
    with pytest.raises(ValueError, match="measured image served on GPUs"):
        DeployDriveParams.model_validate(_params(**{**BASELINE_DRIVE, "artifact": {"kind": "package"}}))
    params = DeployDriveParams.model_validate(_params(**BASELINE_DRIVE))
    # The key is the measured server's: its image, start, GPUs, weights, build environment and seal.
    key = served_cache_key(params, "a" * 64, WEIGHTS)
    changed = [served_cache_key(params, "b" * 64, WEIGHTS),
               served_cache_key(params.model_copy(update={"serve_command": ["--jit-sec", "1"]}), "a" * 64, WEIGHTS),
               served_cache_key(params, "a" * 64, WEIGHTS.model_copy(update={"revision": "d" * 40})),
               served_cache_key(params.model_copy(update={"network": NetworkPolicy(domains=AccessList(
                   mode="allow", entries=["example.com"])).model_dump(mode="json")}), "a" * 64, WEIGHTS)]
    assert len({key, *changed}) == 5
    # Where the server listens changes nothing it builds.
    assert served_cache_key(params.model_copy(update={"port": 9000}), "a" * 64, WEIGHTS) == key


def test_a_submission_that_differs_from_its_recorded_digest_is_refused(tmp_path, bins):
    # The extraction role refuses its copy of a file whose bytes are not the recorded ones. Under
    # the pin rule a refused file of the store is retried.
    ctx = _ctx(tmp_path, bins)
    sif = _submission(ctx)
    recorded = _submission_sha(ctx)
    sif.write_bytes(sif.read_bytes() + b"\0")
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and f"not {recorded}" in fault.reason
    assert _started(ctx) == ["copy", "copy", "extract"]


def test_a_package_manifest_that_disagrees_with_package_json_is_refused(tmp_path, bins):
    ctx = _ctx(tmp_path, bins)
    key = ctx.plan.layout.manifest("package")
    manifest = ctx.inputs[key]
    other = manifest.outputs["submission"].model_copy(update={"sha256": "0" * 64})
    ctx.inputs[key] = manifest.model_copy(update={"outputs": {**manifest.outputs, "submission": other}})
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False) and "not the one package.json names" in fault.reason
    assert ctx.launcher.launched == []


def test_a_package_json_that_differs_from_its_manifest_is_refused(tmp_path, bins):
    ctx = _ctx(tmp_path, bins)
    record = ctx.store.path(ctx.inputs[ctx.plan.layout.manifest("package")].outputs["package"].key)
    record.write_bytes(record.read_bytes() + b"\n")
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False) and "does not match its manifest" in fault.reason


def _rewrite_package(ctx: StageContext, change) -> None:
    """Rewrite the ``package.json`` the drive reads through *change* (its fields to the fields
    written), and the package manifest's record of it to match, as another package stage would
    have written and published it."""
    key = ctx.plan.layout.manifest("package")
    manifest = ctx.inputs[key]
    ref = manifest.outputs["package"]
    data = json.dumps(change(json.loads(ctx.store.read_bytes(ref.key)))).encode()
    ctx.store.path(ref.key).write_bytes(data)
    ctx.inputs[key] = manifest.model_copy(update={"outputs": {
        **manifest.outputs, "package": ref.model_copy(update={"sha256": sha256_bytes(data), "size": len(data)})}})


def test_a_package_another_package_stage_wrote_is_served_by_the_fields_a_grading_reads(tmp_path, bins):
    """A ``package.json`` that lacks a field and holds one this code does not know is served: of
    ``package.json`` a grading reads only what it grades."""
    ctx = _ctx(tmp_path, bins)
    _rewrite_package(ctx, lambda record: {k: v for k, v in record.items() if k not in ("wheels", "commit")}
                     | {"added_since": 1})
    outputs = DeployDriveStage().run(ctx)
    assert _drive(ctx, outputs).outcome == "served"


@pytest.mark.parametrize("change, errors", [
    (lambda record: {k: v for k, v in record.items() if k != "status"}, "missing status (required by this version of hls)"),
    (lambda record: record | {"image": None}, "image_set_exactly_when_built at the top level"),
], ids=["no-status", "built-without-its-image"])
def test_a_package_record_a_grading_cannot_read_is_refused_naming_its_errors(tmp_path, bins, change, errors):
    ctx = _ctx(tmp_path, bins)
    _rewrite_package(ctx, change)
    fault = _fault(ctx).fault
    key = ctx.inputs[ctx.plan.layout.manifest("package")].outputs["package"].key
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == f"{key} is not a package record: {errors}"
    assert ctx.launcher.launched == []


@pytest.mark.parametrize("stage", ["provision", "images"])
def test_a_missing_input_manifest_is_refused(tmp_path, bins, stage):
    ctx = _ctx(tmp_path, bins)
    del ctx.inputs[ctx.plan.layout.manifest(stage)]
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False) and "without input manifest" in fault.reason
    assert ctx.launcher.launched == []


def test_an_image_images_json_does_not_name_is_refused(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, harness_image="other")
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False) and "no image other" in fault.reason
    assert "artifact" not in _started(ctx)


@pytest.mark.parametrize("images, why", [
    ({"harness": {"key": HARNESS}}, "does not name each image's key and entry digest"),
    ({"harness": {"key": HARNESS, "entry_sha256": "not hex"}}, "records no digest"),
    ({"harness": {"key": HARNESS, "entry_sha256": "0" * 64}}, "not the " + "0" * 64),
])
def test_an_image_pinned_to_another_digest_than_its_producers_is_refused(tmp_path, bins, images, why):
    ctx = _ctx(tmp_path, bins, images=images)
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False) and why in fault.reason
    assert "artifact" not in _started(ctx)


def test_a_task_that_does_not_load_is_infra_and_not_retried(tmp_path, bins):
    ctx = _ctx(tmp_path, bins, task={"module": "tasks.no_such_task.task"})
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False) and "does not load" in fault.reason
    assert ctx.launcher.launched == []


@pytest.mark.parametrize("field", ["request_timeout_sec", "readiness_timeout_sec", "wall_margin_sec"])
def test_the_stage_has_no_grace_or_readiness_budget_of_its_own(field):
    """The planner names each (``hls.pipeline.pipelines``, ``DRIVE_MARGIN_SEC`` for the margin):
    a stage missing any of them is refused, so no default can stand in for the planned figure."""
    params = _params()
    del params[field]
    with pytest.raises(ValueError, match=field):
        DeployDriveParams.model_validate(params)


def test_a_submission_images_drive_holds_what_a_package_adds_to_its_image():
    """A submission image is its base's with what the install and the tree add
    (``PACKAGE_ADDS_BYTES``, which the package stage and the checkout hold it to); the stage's
    extraction holds its copy of the file and its tree at once, each carrying those bytes, so the
    stage of a package, this instance's or a published one, or of a published image file holds
    them twice beside the two images' ``KEPT_DISK_GIB``. A measured image adds nothing."""
    adds = -(-PACKAGE_ADDS_BYTES // 2**30)
    base = deploy_drive.PINNED_IMAGES * deploy_drive.KEPT_DISK_GIB
    assert deploy_drive.extracted_disk_gib(PackagedArtifact()) == base + 2 * adds
    published = PackagedArtifact(manifest="warmers/t/20261002T000000Z-abcdef/stages/package/MANIFEST.json")
    assert deploy_drive.extracted_disk_gib(published) == base + 2 * adds
    image_file = PublishedArtifact(sha256="a" * 64, source="/shared/published/submission.sif")
    assert deploy_drive.extracted_disk_gib(image_file) == base + 2 * adds
    measured = ImageArtifact(image="baseline", entrypoint=["serve"])
    assert deploy_drive.extracted_disk_gib(measured) == base
    params = DeployDriveParams.model_validate(_params())
    short = SHAPE.model_copy(update={"disk_gib": ROLES.disk_gib + base + DATA_GIB})
    with pytest.raises(ValueError, match=f"{base + 2 * adds} GiB more disk for the images it extracts"):
        DeployDriveStage.check(short, params, runner_memory_gib=RUNNER_GIB)


def test_the_stage_is_its_roles_the_runners_memory_and_its_images_disk():
    params = DeployDriveParams.model_validate(_params())
    DeployDriveStage.check(SHAPE, params, runner_memory_gib=RUNNER_GIB)
    # A site may give the stage another time, never other resources.
    DeployDriveStage.check(SHAPE.model_copy(update={"time_sec": 5000}), params, runner_memory_gib=RUNNER_GIB)
    for shape in (ROLES, SHAPE.model_copy(update={"cpus": 8}), SHAPE.model_copy(update={"disk_gib": 10}),
                  SHAPE.model_copy(update={"memory_gib": ROLES.memory_gib})):
        with pytest.raises(ValueError, match="resources must be the artifact's and the driver's together"):
            DeployDriveStage.check(shape, params, runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError, match=f"with 2 GiB more memory for the runner, {EXTRACTED_GIB} GiB more disk for "
                                         "the images it extracts and 1 GiB for the copies of their data"):
        DeployDriveStage.check(SHAPE, params, runner_memory_gib=2)
    with pytest.raises(ValueError, match="does not cover its floor of 602s"):
        DeployDriveStage.check(SHAPE, DeployDriveParams.model_validate(_params(readiness_timeout_sec=295)),
                               runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError, match="does not cover its floor of 602s"):
        DeployDriveStage.check(SHAPE, DeployDriveParams.model_validate(_params(drive_timeout_sec=590)),
                               runner_memory_gib=RUNNER_GIB)
    # The staging the stage allows before its server counts too, so no site's resources table can
    # give the stage a time its ordinary staging would cut the drive's wall in.
    with pytest.raises(ValueError, match="does not cover its floor of 601s: the staging"):
        DeployDriveStage.check(SHAPE, DeployDriveParams.model_validate(_params(staging_sec=579)),
                               runner_memory_gib=RUNNER_GIB)
    DeployDriveStage.check(SHAPE, DeployDriveParams.model_validate(_params(staging_sec=578)),
                           runner_memory_gib=RUNNER_GIB)
    # So does the time allowed for the server's runtime to start, before its budgets count.
    with pytest.raises(ValueError, match="does not cover its floor of 601s: the staging, the runtime's start"):
        DeployDriveStage.check(SHAPE, DeployDriveParams.model_validate(_params(runtime_start_sec=579)),
                               runner_memory_gib=RUNNER_GIB)
    DeployDriveStage.check(SHAPE, DeployDriveParams.model_validate(_params(runtime_start_sec=578)),
                           runner_memory_gib=RUNNER_GIB)
    other = ResourceShape(gpus=1, gpu_kind="A100", cpus=1, memory_gib=2, time_sec=600)
    with pytest.raises(ValueError, match="different gpu kinds"):
        DeployDriveStage.check(SHAPE, DeployDriveParams.model_validate(_params(driver_shape=other.model_dump())),
                               runner_memory_gib=RUNNER_GIB)


def test_the_driver_shares_the_artifacts_cpus(tmp_path, bins):
    # An overlapping driver takes no CPU of its own: a stage of the artifact's 8 CPUs holds both, and the same stage without the
    # overlap is refused.
    artifact = ARTIFACT_SHAPE.model_copy(update={"cpus": 8})
    driver = DRIVER_SHAPE.model_copy(update={"cpus": 8})
    roles = concurrent([artifact, driver])
    stage = roles.model_copy(update={"cpus": 8, "memory_gib": roles.memory_gib + RUNNER_GIB,
                                     "disk_gib": roles.disk_gib + EXTRACTED_GIB + DATA_GIB})
    shapes = {"artifact_shape": artifact.model_dump(), "driver_shape": driver.model_dump()}
    DeployDriveStage.check(stage, DeployDriveParams.model_validate(_params(driver_overlaps=True, **shapes)),
                           runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError, match="resources must be the artifact's and the driver's together"):
        DeployDriveStage.check(stage, DeployDriveParams.model_validate(_params(**shapes)),
                               runner_memory_gib=RUNNER_GIB)
    # An overlapping driver with a GPU would land on GPUs the artifact holds.
    with pytest.raises(ValueError, match="overlapping driver"):
        DeployDriveParams.model_validate(_params(driver_overlaps=True, driver_shape=ARTIFACT_SHAPE.model_dump()))
    # And at run time: the driver's step overlaps the artifact's two CPUs, the stage's only two.
    small = SHAPE.model_copy(update={"cpus": ARTIFACT_SHAPE.cpus})
    ctx = _ctx(tmp_path, bins, shape=small, driver_overlaps=True,
               driver_shape=DRIVER_SHAPE.model_copy(update={"cpus": ARTIFACT_SHAPE.cpus}).model_dump())
    record = _drive(ctx, DeployDriveStage().run(ctx))
    overlapping = {s.name for s in ctx.launcher.launched if s.overlap}
    assert record.outcome == "served" and overlapping == {"driver"}


def test_a_driver_that_shares_the_artifacts_memory_adds_none_to_the_stage(tmp_path, bins):
    """An overlapping driver whose memory is oversubscribed on the artifact's: the stage holds the
    artifact's memory and the runner's, the driver's own must fit it beside the runner's, and at
    run time the driver's step is started sharing the artifact's memory."""
    artifact = ARTIFACT_SHAPE.model_copy(update={"cpus": 8})
    driver = DRIVER_SHAPE.model_copy(update={"cpus": 8})
    roles = concurrent([artifact, driver])
    stage = roles.model_copy(update={"cpus": 8, "memory_gib": artifact.memory_gib + RUNNER_GIB,
                                     "disk_gib": roles.disk_gib + EXTRACTED_GIB + DATA_GIB})
    shapes = {"artifact_shape": artifact.model_dump(), "driver_shape": driver.model_dump()}
    params = DeployDriveParams.model_validate(_params(driver_overlaps=True, driver_memory_shared=True, **shapes))
    DeployDriveStage.check(stage, params, runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError, match="resources must be the artifact's and the driver's together"):
        DeployDriveStage.check(stage, DeployDriveParams.model_validate(_params(driver_overlaps=True, **shapes)),
                               runner_memory_gib=RUNNER_GIB)
    big = driver.model_copy(update={"memory_gib": stage.memory_gib})
    with pytest.raises(ValueError, match="which shares the artifact's memory, and the runner's"):
        DeployDriveStage.check(stage, DeployDriveParams.model_validate(_params(
            driver_overlaps=True, driver_memory_shared=True, artifact_shape=artifact.model_dump(),
            driver_shape=big.model_dump())), runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError, match="only an overlapping driver shares the artifact's memory"):
        DeployDriveParams.model_validate(_params(driver_memory_shared=True, **shapes))
    assert "driver_memory_shared" not in DeployDriveParams.model_validate(_params(**shapes)).model_dump()
    # At run time, on a stage that holds the artifact's memory and not the driver's beside it.
    small = SHAPE.model_copy(update={"cpus": ARTIFACT_SHAPE.cpus,
                                     "memory_gib": SHAPE.memory_gib - DRIVER_SHAPE.memory_gib})
    ctx = _ctx(tmp_path, bins, shape=small, driver_overlaps=True, driver_memory_shared=True,
               driver_shape=DRIVER_SHAPE.model_copy(update={"cpus": ARTIFACT_SHAPE.cpus}).model_dump())
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served"
    assert "driver" in ctx._memory_shared and "artifact" not in ctx._memory_shared


def test_the_artifact_must_declare_the_disk_that_caps_it():
    uncapped = ARTIFACT_SHAPE.model_copy(update={"disk_gib": 0})
    params = DeployDriveParams.model_validate(_params(artifact_shape=uncapped.model_dump()))
    shape = SHAPE.model_copy(update={"disk_gib": SHAPE.disk_gib - 1})
    with pytest.raises(ValueError, match="must declare the disk its workdir and engine cache may fill"):
        DeployDriveStage.check(shape, params, runner_memory_gib=RUNNER_GIB)


def test_the_stage_refuses_to_copy_data_its_plan_made_no_room_for(tmp_path, bins):
    """The stage's disk holds the copies of the artifact's and the driver's data its plan
    declared (``data_copies``), so a node that passed the job's start check never refuses the
    artifact for its scratch. A provision record with a copy the plan did not declare is infra not
    retried, before anything is copied or served."""
    ctx = _ctx(tmp_path, bins, shape=SHAPE.model_copy(update={"disk_gib": SHAPE.disk_gib - DATA_GIB}),
               data_copies={})
    with pytest.raises(StageFault) as e:
        DeployDriveStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    weights, rows = list(DATA_COPIES)[:2]
    assert e.value.fault.reason == (f"the artifact role's mounts: the plan made no room on the node for a copy of "
                                    f"{weights}, {rows}: plan the instance again")
    assert not [s for s in ctx.launcher.launched if s.argv[1:3] == ("-m", "hls.store.cache_role")]
    assert "artifact" not in {s.name for s in ctx.launcher.launched}


def test_every_role_runs_for_the_stages_own_time(tmp_path, bins):
    # The role shapes declare the stage's default time; a site's [resources] time is theirs too.
    ctx = _ctx(tmp_path, bins, shape=SHAPE.model_copy(update={"time_sec": 900}))
    DeployDriveStage().run(ctx)
    assert {s.shape.time_sec for s in ctx.launcher.launched} == {900}
    assert [s.name for s in ctx.launcher.launched if not s.name.startswith(("extract-", "copy-"))] == [
        "seal-probe", "artifact", "driver"]


def _driver_command(tmp_path) -> list[str]:
    return fake_container.records(tmp_path / "records")["driver"]["command"]


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_the_drivers_command_passes_the_reply_limits_and_its_memory_limit(tmp_path, bins):
    # The role holds each reply to its limit and the trace to the task's cap, and its memory
    # guard is on at the driver's declared memory, whatever the drive. The cap is not the
    # fixtures' and is smaller: the stage allocates the trace's room, about twice the cap, on the
    # test's disk.
    ctx = _ctx(tmp_path, bins, reply_max_bytes=64 * 2**20, trace_cap_bytes=3 * 2**28)
    assert _drive(ctx, DeployDriveStage().run(ctx)).outcome == "served"
    command = _driver_command(tmp_path)
    assert (_flag(command, "--reply-max-bytes"), _flag(command, "--trace-cap-bytes"),
            _flag(command, "--memory-limit-bytes")) == (str(64 * 2**20), str(3 * 2**28),
                                                        str(DRIVER_SHAPE.memory_gib * 2**30))
    assert "--measured-baseline" not in command and "--drive-budget-bytes" not in command


@pytest.mark.parametrize("measured", [True, False])
def test_only_a_measured_baseline_drive_passes_measured_baseline(tmp_path, bins, measured):
    ctx = _ctx(tmp_path, bins, **(BASELINE_DRIVE if measured else {}))
    if measured:
        _warm_entry(ctx)
    assert _drive(ctx, DeployDriveStage().run(ctx)).outcome == "served"
    command = _driver_command(tmp_path)
    assert ("--measured-baseline" in command) == measured
    assert _flag(command, "--memory-limit-bytes") == str(DRIVER_SHAPE.memory_gib * 2**30)


@pytest.mark.parametrize("field, value", [
    ("reply_max_bytes", 0), ("reply_max_bytes", -1), ("trace_cap_bytes", 0), ("trace_cap_bytes", -1),
    ("trace_request_bytes", 0),
    ("measured_baseline", "yes"), ("measured_baseline", 1)])
def test_the_reply_limits_are_positive_and_the_baseline_flag_a_bool(field, value):
    with pytest.raises(ValueError, match=field):
        DeployDriveParams.model_validate(_params(**{field: value}))
    for name in ("reply_max_bytes", "trace_cap_bytes", "trace_request_bytes", "measured_baseline"):
        params = _params()
        del params[name]
        with pytest.raises(ValueError, match=name):
            DeployDriveParams.model_validate(params)


def test_the_parameters_refuse_what_cannot_be_served():
    with pytest.raises(ValueError):
        DeployDriveParams.model_validate(_params(network={"domains": {"mode": "sideways"}}))
    with pytest.raises(ValueError):
        DeployDriveParams.model_validate(_params(artifact={"kind": "image", "image": "measured", "entrypoint": []}))
    with pytest.raises(ValueError):
        DeployDriveParams.model_validate(_params(port=0))
    with pytest.raises(ValueError, match="invalid stage name"):
        DeployDriveParams.model_validate(_params(artifact={"kind": "package", "stage": "../package"}))
    with pytest.raises(ValueError, match="invalid stage name"):
        DeployDriveParams.model_validate(_params(provision_stage="Provision"))
    with pytest.raises(ValueError, match="invalid stage name"):
        DeployDriveParams.model_validate(_params(images_stage="Images"))
    with pytest.raises(ValueError):
        DeployDriveParams.model_validate(_params(harness_image="../harness"))
    with pytest.raises(ValueError, match="named by its absolute path"):
        DeployDriveParams.model_validate(_params(artifact={"kind": "published", "sha256": "a" * 64,
                                                           "source": "published/submission.sif"}))
    with pytest.raises(ValueError):
        DeployDriveParams.model_validate(_params(artifact={"kind": "published", "sha256": "md5",
                                                           "source": "/published/submission.sif"}))


def _published(tmp_path: Path, mode: str = "serve") -> tuple[Path, str]:
    """A published submission image on the shared filesystem, and its file's digest."""
    path = _server_image(tmp_path / "published" / "submission.sif", mode)
    return path, sha256_file(path)[0]


def test_a_published_image_is_read_by_its_digest_pinned_as_the_contestants_and_served(tmp_path, bins):
    """A published submission image graded again is read off the shared filesystem by the
    extraction role alone, checked against the digest the plan names, pinned as an image holding
    the contestant's code, served by its runscript, and its digest recorded as the one served. No
    package record is read."""
    path, digest = _published(tmp_path)
    ctx = _ctx(tmp_path, bins, artifact={"kind": "published", "sha256": digest, "source": str(path)})
    del ctx.inputs[ctx.plan.layout.manifest("package")]
    asked: list[tuple[Path, dict]] = []
    real = ctx.pin_file

    def recording(file, **kwargs):
        asked.append((file, kwargs))
        return real(file, **kwargs)

    ctx.pin_file = recording
    record = _drive(ctx, DeployDriveStage().run(ctx))
    assert record.outcome == "served" and record.artifact.image_sha256 == digest == ctx.image_digests["artifact"]
    assert asked == [(path, {"sha256": digest, "contestant": True, "stored": True})]
    assert fake_container.records(tmp_path / "records")["artifact"]["command"][:1] == ["/.singularity.d/runscript"]


def test_a_published_file_with_other_bytes_than_its_digest_is_refused_and_not_retried(tmp_path, bins):
    """A retry would read the same file, so a file whose bytes are not the named digest's is a
    final refusal, before any role of the stage's own starts."""
    path, digest = _published(tmp_path)
    ctx = _ctx(tmp_path, bins, artifact={"kind": "published", "sha256": "0" * 64, "source": str(path)})
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False) and f"has sha256 {digest}, not {'0' * 64}" in (
        fault.reason)
    assert _started(ctx) == ["copy", "copy", "extract"]
