"""The agent_build stage end to end on one machine: harbor's real Claude Code and Codex adapters
drive fake CLIs (``tests.agent.fake_cli``) through the real exec server and spawner, run without
namespaces in a fake container (``tests.agent.fake_sandbox``) behind the real seal under the fake
pasta, on the runner's clock, which the fake CLIs move forward. The agent cap is pinned from this
instance's ``images.json`` and extracted by the fake's ``build --sandbox``. Then the real commit
point and transcript copy run on the workspace.

Harbor's adapters name container paths (``/logs/agent``, ``/workspace``, ``/tmp/codex-home``) in
the commands they send. ``MappedEnvironment`` rewrites them to the host directories the sandbox
binds, as the container's mount namespace would, and keeps what harbor sent."""

from __future__ import annotations

import asyncio
import base64
import errno
import hashlib
import json
import logging
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.agents.installed import claude_code
from pydantic import ValidationError

import hls.pipeline.stage as stage_module
from hls.substrate.ucounts import ServedUcounts
from hls.agent import environment, exec_protocol, harnesses, workspace
from hls.agent.environment import SandboxEnvironment
from hls.agent import commit as commit_program
from hls.agent.commit import snapshot_name
from hls.agent import probe as probe_module
from hls.agent.memory import SplitSample
from hls.agent.probe import ProbeRecord
from hls.agent.uploads import tree_digest
from hls.agent.workspace import SandboxLayout
from hls.arms import ArmRecord, Route, adapter_for
from hls.egress import seal_probe
from hls.pipeline.checkout import CodeState
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, NOOP_TASK, artifact_shape
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.planner import PlanError, StageDecl, build_plan
from hls.pipeline.stage import LOG_CLOSE_SEC, STOP_GRACE_SEC, StageContext, StageFault, run_stage
from hls.prompt import budget_brief_section, packaging_brief_section
from hls.provision.model import Provisioned, WeightsRef, weights_key
from hls.provision.mounts import copied_entries, copies_gib
from hls.settings import RunSettings, Settings, load_settings
from hls.spec.resources import GrantedResources, ResourceShape, concurrent
from hls.stages import agent_build
from hls.stages.agent_build import AgentBuildParams, AgentBuildStage, BuildRecord, seal_policy, seed_source
from hls.stages.images import KEPT_DISK_GIB
from hls.store.cache import ContentCache, copy_disk_bytes, entry_sha256
from hls.store.digest import sha256_file
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import Manifest, Provenance, Resources, Times, output_ref
from hls.store.posix import PosixStore
from hls.substrate import log_cap, memory_watch
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.container import TREE_ABSENT_EXIT, PinnedImage
from hls.substrate.seal import EGRESS_INDEX_PORT, EGRESS_TRANSPARENT_PORT
from hls.substrate.base import FaultRecord, NodeSize
from hls.substrate.faults import OOM_EVIDENCE_MAX_AGE_SEC, RoleOutcome, SplitEvidence, infra_fault
from hls.substrate.memory_watch import MemoryWatch, PidsHits
from hls.substrate.nproc import DEFAULT_NPROC_MAX, FLOOR_PER_EVALUATION, ProcessCeiling
from hls.substrate.secrets import Scrubber, SecretsFile
from hls.taskref import TaskRef, load_task
from hls.util import remove_tree
from tests.agent import fake_cli, fake_sandbox
from tests.agent.fake_sandbox import mapped
from tests.agent.probe_answers import ICMP_LEFT, NAMESPACE_CHECKS, SEAL_ANSWERS, STEP_CHECKS, answering
from tests.pipeline.fakes import NODE, LocalLauncher
from tests.store.entries import populate
from tests.substrate import fake_pasta
from tests.support import node
from tests.unit.brief_task import PIP, PIP_LIMITS

REPO = Path(__file__).resolve().parents[2]
IMAGE_KEY, IMAGE_NAME = "agentcap-0123456789abcdef", "agent-cap"
KEY = "sk-test-0123456789abcdef"
#: A relay's base URL, a secret like its credential, and its host.
RELAY_URL = "https://relay.example/v1"
HOST = "relay.example"
SANDBOX = ResourceShape(cpus=2, memory_gib=16, disk_gib=8, time_sec=600)
SNAPSHOT = ResourceShape(cpus=1, memory_gib=2, time_sec=600)
COMMIT = ResourceShape(cpus=1, memory_gib=4, time_sec=600)
#: The extraction role's shape (``extract_shape``).
EXTRACT = ResourceShape(cpus=1, memory_gib=4, time_sec=600)
#: The memory the test plans leave to the runner (``run.runner_memory_gib``'s default).
RUNNER_GIB = 4
#: The shape a graded server of the build task runs in: its hardware, as it declares no evaluation
#: allocation.
SERVED = artifact_shape(load_task(TaskRef(module="tests.agent.build_task")), runner_memory_gib=RUNNER_GIB)
#: The setup allowance, the staging and the hard limit the test params declare; the stage's time
#: must cover these and the collection allowance.
SETUP_SEC, STAGING_SEC, HARD_SEC = 60, 30, 1100
#: The collection allowance the test params declare: the least a plan accepts, the collection's
#: fixed steps and the commit point's least time.
COLLECT_SEC = agent_build.COLLECT_STEPS_SEC + agent_build.COMMIT_MIN_SEC
STAGE_TIME = SETUP_SEC + STAGING_SEC + HARD_SEC + COLLECT_SEC
#: What the packaging section of a test build's brief is rendered from: the pip limits, the
#: build's collection allowance and the commit point's least part of it.
PACKAGING = {**PIP, "collect_sec": COLLECT_SEC, "bundle_sec": agent_build.COMMIT_MIN_SEC}
ROLES = concurrent([SANDBOX, SNAPSHOT])
# The harness role runs on CPUs the sandbox holds, so the stage's CPUs are the sandbox's.
#: The copies of the agent's data, as the planner declares them (``data_copies``): one weights
#: entry, a MiB, more than its copy takes.
DATA_COPIES = {f"weights/{weights_key('org/model', 'c' * 40)}": 2**20}
#: GiB they take on the node.
DATA_GIB = copies_gib(DATA_COPIES)
STAGE = ROLES.model_copy(update={"cpus": SANDBOX.cpus, "memory_gib": ROLES.memory_gib + RUNNER_GIB,
                                 "disk_gib": ROLES.disk_gib + KEPT_DISK_GIB + DATA_GIB, "time_sec": STAGE_TIME})
ROUTES = {"anthropic": Route(credential="anthropic_key"),
          "openai_relay": Route(credential="openai_key", base_url="openai_url"),
          "kimi": Route(credential="kimi_token", base_url="kimi_url", auth="bearer")}
OPUS = ArmRecord(name="arm-a", harness="claude-code", model="claude-opus-4-8", context_window=1_000_000,
                 output_limit=64_000, effort="max", route="anthropic", route_table=ROUTES["anthropic"])
GPT = ArmRecord(name="arm-b", harness="codex", model="gpt-5.5", context_window=272_000, output_limit=128_000,
                effort="xhigh", route="openai_relay", route_table=ROUTES["openai_relay"])
K3 = ArmRecord(name="k3", harness="claude-code", model="k3", context_window=200_000, output_limit=64_000,
               effort="max", route="kimi", route_table=ROUTES["kimi"])
#: The relay's Anthropic-style base URL for the kimi route, on the same host. A site file holds
#: no secret's value inside another's (``SecretsFile``), so it is not a path under ``RELAY_URL``.
KIMI_URL = f"https://{HOST}/anthropic"
SECRETS = {"anthropic_key": KEY, "openai_key": KEY.replace("sk-test", "sk-open"), "openai_url": RELAY_URL,
           "kimi_token": KEY.replace("sk-test", "sk-kimi"), "kimi_url": KIMI_URL}
#: The cap on a contestant step's process ceiling where the site sets none (``site.nproc_max``), on
#: a node that holds at most 32 evaluations of the step's size.
NPROC_MAX = DEFAULT_NPROC_MAX
#: The tasks kept for each evaluation's harness when a site sets none (``run.nproc_reserve``).
RESERVE = RunSettings.model_fields["nproc_reserve"].default
#: The node a sandbox test's launcher runs on unless the test gives one: the local launcher's own
#: node's GPUs, so a graded server's share of a node is as on it, and the CPUs of one build of the
#: stage's size, so the commands' ceiling is one this host's kernel can set (``tests.support.node``).
COMMANDS_NODE = NodeSize(gpus=NODE.gpus, cpus=STAGE.cpus)
#: The producer's network: no host but the route's, and the packages its index refuses.
NETWORK = {"domains": {"mode": "allow", "entries": []}, "packages": {"mode": "deny", "entries": ["vllm"]}}
GIB = 2**30
#: Every context a test made, which the test's teardown closes.
_CONTEXTS: list[StageContext] = []
#: The scratch directory of each, which the teardown removes after it. It is a short path of its
#: own, never under ``tmp_path``: the sandbox's sockets are made below it, and the kernel takes at
#: most 107 bytes of a socket's path.
_SCRATCHES: list[Path] = []


class MappedEnvironment(SandboxEnvironment):
    """Harbor's environment for these tests: every container path in a command, its working
    directory and its variables rewritten to the host directory the sandbox binds there. What
    harbor sent is kept, unrewritten, in ``SENT``."""

    #: (command, env) of every exec harbor sent, as it sent them.
    SENT: list[tuple[str, dict]] = []
    #: (command, timeout_sec) of every exec, so a test can pin a command's timeout.
    TIMEOUTS: list[tuple[str, int | None]] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        MappedEnvironment.SENT.append((command, dict(env or {})))
        MappedEnvironment.TIMEOUTS.append((command, timeout_sec))
        layout = SandboxLayout(self.socket.parent.parent)
        paths = {m.target: str(m.source) for m in layout.mounts()} | {"/tmp": str(layout.workdir / "tmp")}
        return await super().exec(mapped(paths, command), cwd=mapped(paths, cwd) if cwd else None,
                                  env={k: mapped(paths, v) for k, v in env.items()} if env else None,
                                  timeout_sec=timeout_sec, user=user)


@pytest.fixture(autouse=True)
def closed_contexts():
    """Close every context the test made, as the runner closes a stage's however it ended
    (``run_stage``), so no role of a test that failed, or that ran the stage alone, outlives it."""
    yield
    while _CONTEXTS:
        _CONTEXTS.pop().close()
    while _SCRATCHES:
        remove_tree(_SCRATCHES.pop(), ignore_errors=True)


@pytest.fixture
def sandbox_bins(tmp_path_factory, monkeypatch, fake_seal):
    """The fake container, seal and CLIs, the probe's sealed checks answered as a working seal's
    (``fake_seal``), the runner's clock moved by the fake CLIs, the commands' node, whose ceiling
    this host's kernel sets on the sandbox's real commands (``tests.support.node``), as
    ``COMMANDS_NODE`` unless a test's launcher gives its own, and the runner's environment clean of
    what harbor would read. Returns the CLIs' control directory."""
    node.require_commands_ceiling()
    for var in harnesses.harness_env_leaks(os.environ):
        monkeypatch.delenv(var)
    base = tmp_path_factory.mktemp("b")
    bin_dir, cli_dir, control = base / "bin", base / "cli", base / "control"
    fake_sandbox.install(bin_dir)
    fake_pasta.install(bin_dir)
    fake_cli.install(cli_dir, control)
    venv = Path(sys.executable).parent
    monkeypatch.setenv("PATH", f"{bin_dir}:{venv}:/usr/bin:/bin")
    monkeypatch.setenv("HLS_SEAL_TOOL_PATH", str(bin_dir))
    monkeypatch.setenv("FAKE_SANDBOX_PATH", f"{cli_dir}:{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_SANDBOX_RECORDS", str(base / "records"))
    (base / "pasta.plan").write_text("pass\n" * 8)
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(base / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(base / "pasta.log"))
    for knob in ("FAKE_PROXY_ERRORS", "FAKE_PROXY_MID_STREAM", "FAKE_PROXY_OVERSIZED", "FAKE_PROXY_REFUSED",
                 "FAKE_PROXY_HEARTBEAT_AGE"):
        monkeypatch.delenv(knob, raising=False)
    offset = control / "offset"
    monkeypatch.setattr(agent_build, "monotonic",
                        lambda: time.monotonic() + (float(offset.read_text()) if offset.exists() else 0.0))
    monkeypatch.setattr(stage_module, "hard_limit", lambda: node.COMMANDS_HARD)
    monkeypatch.setattr(LocalLauncher, "node_size", lambda self: COMMANDS_NODE if self.node is None else self.node)
    monkeypatch.setattr(environment, "SandboxEnvironment", MappedEnvironment)
    monkeypatch.setattr(MappedEnvironment, "SENT", [])
    monkeypatch.setattr(MappedEnvironment, "TIMEOUTS", [])
    monkeypatch.setattr(agent_build, "_ipv4s", lambda host: [])  # no DNS from a test
    return control


def _script(control: Path, **script) -> None:
    (control / "script.json").write_text(json.dumps(script))


def _calls(control: Path, *, preflight: bool = False) -> list[dict]:
    path = control / "calls.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [r for r in rows if r["preflight"] == preflight]


def _params(arm: ArmRecord = OPUS, **changes) -> dict:
    p = {"task": {"module": "tests.agent.build_task"}, "arm": arm.model_dump(), "sandbox_image": IMAGE_NAME,
         "sandbox_held": None, "sandbox_shape": SANDBOX.model_dump(), "served_shape": SERVED.model_dump(),
         "snapshot_shape": SNAPSHOT.model_dump(),
         "commit_shape": COMMIT.model_dump(), "extract_shape": EXTRACT.model_dump(), "soft_sec": 1000,
         "hard_sec": HARD_SEC, "setup_allowance_sec": SETUP_SEC, "staging_sec": STAGING_SEC,
         "collect_allowance_sec": COLLECT_SEC, "harness_reserve_gib": 4.0,
         "data_copies": DATA_COPIES, "network": NETWORK, "package_pip": PIP_LIMITS.model_dump()}
    return {**p, **changes}


def _settings_table(tmp_path: Path, site: dict | None = None, run: dict | None = None) -> dict:
    """The tables of the test site's hls.toml, with *site* and *run* over its own settings."""
    return {
        "site": {"partition": "gpu", "account": "acct",
                 "store_root": str(tmp_path / "store"), "node_scratch": str(tmp_path / "node"), "secrets_path": "/x",
                 "pasta_path": str(Path(os.environ["PATH"].split(":")[0]) / "pasta"), **(site or {})},
        "run": {"heartbeat_sec": 1, "lease_ttl_sec": 30, "poll_sec": 1, "allocation_overhead_sec": 360,
                "allocation_dead_after_sec": 10, **(run or {})},
        "routes": {name: route.model_dump(exclude_none=True) for name, route in ROUTES.items()}}


def _settings(tmp_path: Path, site: dict | None = None, run: dict | None = None) -> Settings:
    return Settings.model_validate(_settings_table(tmp_path, site, run))


def _site_file(path: Path, tables: dict) -> Path:
    """Write *tables*, each a table of strings, numbers and booleans or a table of such tables, as
    the hls.toml at *path*, and return it."""
    def value(v) -> str:
        return ("true" if v else "false") if isinstance(v, bool) else json.dumps(v)

    lines = []
    for name, table in tables.items():
        flat = {k: v for k, v in table.items() if not isinstance(v, dict)}
        lines += [f"[{name}]", *(f"{k} = {value(v)}" for k, v in flat.items()), ""]
        for sub, inner in ((k, v) for k, v in table.items() if isinstance(v, dict)):
            lines += [f"[{name}.{sub}]", *(f"{k} = {value(v)}" for k, v in inner.items()), ""]
    path.write_text("\n".join(lines))
    return path


def _plan(tmp_path: Path, params: dict, *, shape: ResourceShape = STAGE, secrets: dict[str, str] = SECRETS,
          site: dict | None = None, run: dict | None = None, settings: Settings | None = None):
    decl = StageDecl("agent_build", "agent_build", shape, params)
    plan = build_plan(decls=[decl], task=NOOP_TASK, pipeline="build", layout=InstanceLayout.rollout(NOOP_TASK, "r1"),
                      settings=settings if settings is not None else _settings(tmp_path, site, run),
                      code=CodeState(checkout=Path("/c/hls"), hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-build", known_stage_names=KNOWN_STAGE_NAMES | {"agent_build"},
                      known_tasks=(), now=1000.0, secrets=SecretsFile(secrets=secrets))
    # Planned on a stand-in checkout: a worktree may hold a secrets file the planner refuses. The
    # transcript role runs from the real one.
    return plan.model_copy(update={"code": plan.code.model_copy(update={"checkout": REPO})})


def _manifest(plan, store: PosixStore, stage: str, outputs: dict[str, str]) -> Manifest:
    refs = {name: output_ref(store, key, sha256_file(store.path(key))[0]) for name, key in outputs.items()}
    return Manifest(instance=plan.instance, stage=stage, attempt=1, outputs=refs, inputs={},
                    provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                                          image_digests={}, config_sha256="d" * 64),
                    resources=Resources(declared=STAGE, granted=GrantedResources(gpus=0, cpus=3, memory_gib=22.0,
                                                                                 disk_free_gib=100.0)),
                    times=Times(submitted_at=1.0, allocation_started_at=2.0, started_at=2.5, ended_at=3.0))


def _ctx(tmp_path: Path, params: dict, *, watch: MemoryWatch | None = None,
         launcher: LocalLauncher | None = None, image: bytes | None = None,
         extra_secrets: dict[str, str] | None = None, site: dict | None = None, run: dict | None = None,
         settings: Settings | None = None) -> StageContext:
    """A context for an agent_build attempt of *params*, planned with the test site's settings and
    *site* and *run* over them, or with *settings* as given."""
    scratch = Path(tempfile.mkdtemp(prefix="hls-"))
    _SCRATCHES.append(scratch)
    plan = _plan(tmp_path, params, site=site, run=run, settings=settings)
    store = PosixStore(plan.store_root)
    cache = ContentCache(store)
    holder = Holder.here("test")
    cap = populate(cache, "images", IMAGE_KEY,
                   lambda d: (d / "image.sif").write_bytes(image or fake_sandbox.image_bytes()), holder=holder).entry
    sif = next(f.sha256 for f in cap.manifest.files if f.path == "image.sif")
    images_key = plan.layout.attempt_file("images", 1, "outputs/images.json")
    store.write_bytes(images_key, json.dumps({IMAGE_NAME: {"key": IMAGE_KEY, "entry_sha256": entry_sha256(cap),
                                                           "sif_sha256": sif}}).encode())
    key = weights_key("org/model", "c" * 40)
    weights = populate(cache, "weights", key, lambda d: (d / "config.json").write_text("{}"), holder=holder).entry
    provisioned = Provisioned(weights=WeightsRef(repo="org/model", revision="c" * 40, key=key,
                                                 entry_sha256=entry_sha256(weights), disk_bytes=copy_disk_bytes(weights)), rows=None, tools=[])
    prov_key = plan.layout.attempt_file("provision", 1, "outputs/provisioned.json")
    store.write_bytes(prov_key, provisioned.model_dump_json().encode())
    inputs = {plan.layout.manifest("provision"): _manifest(plan, store, "provision", {"provisioned": prov_key}),
              plan.layout.manifest("images"): _manifest(plan, store, "images", {"images": images_key})}
    stage = plan.stage("agent_build")
    granted = {var: SECRETS[name] for var, name in stage.grants.items()} | (extra_secrets or {})
    events = EventLog(store.path(plan.layout.attempt_file("agent_build", 1, "events.jsonl")))
    ctx = StageContext(store=store, plan=plan, stage=stage, attempt=1,
                       launcher=(launcher or LocalLauncher(gpus=1)).use_plan(plan),
                       watch=watch, events=events, inputs=inputs, deadline=time.time() + 300, secrets=granted,
                       scrubber=Scrubber(SECRETS), cache=cache, scratch=scratch, job_id="100",
                       runtime=ApptainerRuntime(), images=ApptainerImages(scratch))
    _CONTEXTS.append(ctx)
    return ctx


def _events(ctx: StageContext) -> list[dict]:
    return read_events(ctx.attempt_dir / "events.jsonl")


def _ended(ctx: StageContext, role: str) -> dict:
    [event] = [e for e in _events(ctx) if e["kind"] == "role_ended" and e["role"] == role]
    return event


def _record(ctx: StageContext, outputs: dict) -> BuildRecord:
    return BuildRecord.model_validate_json(ctx.store.read_bytes(outputs["build"]))


def _bundle_main(bundle: Path, tmp: Path) -> str:
    out = subprocess.run(["git", "bundle", "list-heads", str(bundle)], capture_output=True, text=True, check=True,
                         cwd=tmp, env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1",
                                       "GIT_CONFIG_GLOBAL": "/dev/null", "HOME": str(tmp)})
    [line] = out.stdout.splitlines()
    sha, ref = line.split()
    assert ref == "refs/heads/main"
    return sha


def _workspace_main(ctx: StageContext) -> str:
    ws = ctx.role_dir("sandbox") / "workspace"
    done = subprocess.run(["git", "rev-parse", "main"], capture_output=True, text=True, check=True, cwd=ws,
                          env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
                               "HOME": str(ws)})
    return done.stdout.strip()


def _gone(ctx: StageContext) -> bool:
    """Whether every process of the sandbox, the CLI among them, is gone."""
    time.sleep(0.5)
    found = subprocess.run(["pgrep", "-f", str(ctx.scratch / "roles" / "sandbox")], capture_output=True)
    return found.stdout == b""


# ---------------------------------------------------------------------------
# A whole build
# ---------------------------------------------------------------------------


def test_a_session_that_commits_main_publishes_a_bundle_of_that_commit(sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    assert sorted(outputs) == ["agent_env", "brief", "brief_provenance", "build", "served_ucounts", "source_bundle",
                               "trajectory", "transcript"]
    bundle = ctx.store.path(outputs["source_bundle"])
    assert _bundle_main(bundle, tmp_path) == _workspace_main(ctx)
    # The bundle and the transcript's index reach the manifest by their writers' own reports.
    assert ctx.output_ref(outputs["source_bundle"]).sha256 == sha256_file(bundle)[0]
    assert ctx.output_ref(outputs["transcript"]).sha256 == sha256_file(ctx.store.path(outputs["transcript"]))[0]
    record = _record(ctx, outputs)
    assert (record.ending, record.reentries, record.sessions_ended) == ("agent_ended", 0, 1)
    assert record.preflight.ok and record.preflight.window == 1_000_000
    assert record.commit.status == "ok" and record.commit.commit == _workspace_main(ctx)
    # The sandbox was stopped as planned, and its memory split is the record's.
    assert _ended(ctx, "sandbox")["ending"] == "planned_stop"
    assert record.memory is not None and record.memory.reserve_bytes == 4 * GIB
    # One preflight, then the build's one invocation, with the arm's model and output limit.
    [pre] = _calls(control, preflight=True)
    [build] = _calls(control)
    assert pre["env"]["ANTHROPIC_MODEL"] == build["env"]["ANTHROPIC_MODEL"] == "claude-opus-4-8[1m]"
    assert build["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"
    # The CLI compacts against the arm's window, not only at a refusal it may not recognise.
    assert build["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "1000000"
    assert build["credential_set"] and not build["token_seen"]
    register = [c for c, _ in MappedEnvironment.SENT if ".claude.json" in c and "mcpServers" in c]
    assert len(register) == 2  # the preflight's and the build's
    assert all('"command": "/opt/hls-budget/get-budget-mcp"' in c and '"args": []' in c for c in register)
    kinds = [e["kind"] for e in _events(ctx)]
    for kind in ("build.preflight", "build.started", "build.agent_session_ended", "build.source_collected",
                 "build.finished"):
        assert kind in kinds, kind
    # The uploaded references are in the workspace, the record names their tree's digest, and the
    # brief names the budget and where they are.
    assert (ctx.role_dir("sandbox") / "workspace" / "references" / "docs" / "ref.md").is_file()
    assert record.upload_sha256 == tree_digest(REPO / "tests" / "agent" / "build_task_files")
    brief = ctx.store.read_bytes(outputs["brief"]).decode()
    assert "/workspace/references" in brief and "get_budget" in brief
    assert ctx.running_roles() == []


def test_the_build_brief_states_the_package_limits_and_the_caps_held_versions(sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "advance": 900}])
    # The versions are those the plan read from the base under the sandbox (catalog.held_versions).
    ctx = _ctx(tmp_path, _params(sandbox_held={"torch": "2.11.0+cu130"}))
    outputs = AgentBuildStage().run(ctx)
    section = packaging_brief_section(held={"torch": "2.11.0+cu130"}, **PACKAGING)
    brief = ctx.store.read_bytes(outputs["brief"]).decode()
    assert f"\n## Packaging limits\n{section}\n" in brief
    # The time to collect the repository is the build's own allowance, as the plan gives it, and the
    # commit point's part what the collection's fixed steps leave of it.
    assert ("The harness then has at least 2,350 seconds to collect your repository, at least 10 minutes of it to "
            "bundle it; a repository it cannot bundle in that time is not collected, and your build is not "
            "graded.") in brief
    provenance = json.loads(ctx.store.read_bytes(outputs["brief_provenance"]))
    assert [s["rendered"] for s in provenance["sections"] if s["block"] == "packaging"] == [section]


def test_a_sandbox_that_holds_no_versions_has_a_brief_that_states_the_limits_and_no_held_versions(
        sandbox_bins, tmp_path):
    """A sandbox on a base that holds nothing: its brief states the limits and holds no version."""
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "print('served')\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    brief = ctx.store.read_bytes(outputs["brief"]).decode()
    # The section ends with the package's limits: the budget follows them directly.
    assert f"\n## Packaging limits\n{packaging_brief_section(held=None, **PACKAGING)}\n\n## Your budget\n" in brief
    assert " held when your server is built" not in brief


def test_the_agent_cap_is_pinned_once_from_images_json_and_every_container_runs_from_its_tree(
        sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 2.5}])
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    outputs = AgentBuildStage().run(ctx)
    image = fake_sandbox.image_bytes()
    assert _record(ctx, outputs).sandbox_image == ctx.image_digests["sandbox"] == hashlib.sha256(image).hexdigest()
    assert ctx.cache_entries[f"images/{IMAGE_KEY}"]
    extracted = [e for e in _events(ctx) if e["kind"] == "image_extracted"]
    assert len(extracted) == 1
    records = fake_sandbox.records(Path(os.environ["FAKE_SANDBOX_RECORDS"]))
    trees = {records[role]["image"] for role in ("sandbox", "commit", "snapshot-1")}
    assert len(trees) == 1 and (Path(trees.pop()) / ".fake-image").read_bytes() == image
    # Every contestant step is a container: the sandbox is the one contestant role.
    contestant = [s for s in ctx.launcher.launched if s.owner == "contestant"]
    assert [s.name for s in contestant] == ["sandbox"] and contestant[0].env_fd


def test_the_workspace_starts_with_what_the_image_baked_at_workspace(sandbox_bins, tmp_path):
    """The workspace's bind hides the image's own /workspace, so the stage copies it in
    before the agent starts, and the agent sees it from its working directory."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    baked = {"baked.txt": "from the image\n", "vllm-src/setup.py": "# a build tree\n"}
    ctx = _ctx(tmp_path, _params(), image=fake_sandbox.image_bytes(baked))
    AgentBuildStage().run(ctx)
    [build] = _calls(control)
    assert {"baked.txt", "vllm-src", "references"} <= set(build["listing"])
    workspace = ctx.role_dir("sandbox") / "workspace"
    assert (workspace / "vllm-src" / "setup.py").read_text() == "# a build tree\n"
    assert "build.workspace_seeded" in [e["kind"] for e in _events(ctx)]


def test_an_image_with_nothing_at_workspace_starts_no_seed_role(sandbox_bins, tmp_path):
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    AgentBuildStage().run(ctx)
    assert "seed-workspace" not in [s.name for s in ctx.launcher.launched]


class _CopiesOut(ApptainerImages):
    """An image runtime that holds no image as a host tree, as Docker's: a role of its own copies
    a path out of the image, here from the extracted tree, and refuses a path the image does not
    hold as a directory as the seam says; with *broken*, the role cannot read the image at all."""

    def __init__(self, scratch: Path, *, broken: bool = False) -> None:
        super().__init__(scratch)
        self.broken = broken

    def tree_path(self, image, path):
        return None

    def tree_role(self, image, path, dest, *, own):
        if self.broken:
            return [sys.executable, "-c", "import sys; print('the daemon is gone', file=sys.stderr); sys.exit(1)"], {}
        script = ("import shutil, sys; from pathlib import Path; src = Path(sys.argv[1]); "
                  "ok = src.is_dir() and not src.is_symlink(); "
                  "ok or print(f'refused: image {sys.argv[3]} holds no directory at {sys.argv[4]}', file=sys.stderr); "
                  "ok or sys.exit(%d); shutil.copytree(src, sys.argv[2], dirs_exist_ok=True)" % TREE_ABSENT_EXIT)
        return ([sys.executable, "-c", script, str(Path(image.ref) / path.lstrip("/")), str(dest), image.ref, path],
                {})


class _Unreadable(ApptainerImages):
    """Apptainer's image runtime, but holding the image's tree where no directory is."""

    def tree_path(self, image, path):
        return self.scratch / "no-tree" / path.lstrip("/")


@pytest.mark.parametrize("has_workspace", [True, False])
def test_a_runtime_that_holds_no_host_tree_seeds_the_workspace_by_its_own_role(sandbox_bins, tmp_path,
                                                                                 has_workspace):
    """Where the image runtime holds no image as a host tree, its role, which speaks to the
    container daemon, writes the image's /workspace into the workspace; an image that holds no
    directory there copies nothing, and the build goes on."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    baked = {"vllm-src/setup.py": "# a build tree\n"} if has_workspace else {}
    ctx = _ctx(tmp_path, _params(), image=fake_sandbox.image_bytes(baked, has_workspace=has_workspace))
    ctx.images = _CopiesOut(ctx.scratch)
    AgentBuildStage().run(ctx)
    [seed] = [s for s in ctx.launcher.launched if s.name == "seed-workspace"]
    assert seed.daemon and seed.owner == "harness"
    assert "sandbox" in {s.name for s in ctx.launcher.launched}
    workspace = ctx.role_dir("sandbox") / "workspace"
    seeded = "build.workspace_seeded" in [e["kind"] for e in _events(ctx)]
    if has_workspace:
        assert seeded and (workspace / "vllm-src" / "setup.py").read_text() == "# a build tree\n"
    else:
        assert not seeded and not (workspace / "vllm-src").exists()


@pytest.mark.parametrize("runtime", [_Unreadable, lambda scratch: _CopiesOut(scratch, broken=True)])
def test_an_image_whose_tree_cannot_be_read_is_refused_before_the_sandbox(sandbox_bins, tmp_path, runtime):
    """Only an image that cannot be read at all is refused, infra, before the sandbox starts: its
    tree is no directory, or the runtime's role fails otherwise than finding no /workspace."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    ctx.images = runtime(ctx.scratch)
    with pytest.raises(StageFault) as refused:
        AgentBuildStage().run(ctx)
    assert refused.value.fault.fault_class == "infra"
    assert "sandbox" not in {s.name for s in ctx.launcher.launched}


def test_an_image_without_a_workspace_directory_starts_no_seed_role(sandbox_bins, tmp_path):
    """A tree with no directory at /workspace copies nothing, and the build goes on, as with an
    empty one."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params(), image=fake_sandbox.image_bytes(has_workspace=False))
    AgentBuildStage().run(ctx)
    launched = {s.name for s in ctx.launcher.launched}
    assert "seed-workspace" not in launched and "sandbox" in launched



def test_an_image_not_held_as_a_directory_tree_is_refused_not_seeded_with_nothing(tmp_path):
    """An image reference that is no directory on the node, such as an image ID, where the
    runtime says it holds a host tree, has no /workspace to read: the seed is the harness's
    fault, refused, never an empty copy."""
    images = ApptainerImages(tmp_path / "scratch")
    held = tmp_path / "tree"
    (held / "workspace").mkdir(parents=True)
    (held / "workspace" / "a.txt").write_text("a")
    assert seed_source(images, PinnedImage(ref=str(held), file_sha256="0" * 64, check_sha256="0" * 64)) == (
        held / "workspace")
    with pytest.raises(StageFault, match="is no directory") as refused:
        seed_source(images, PinnedImage(ref="sha256:" + "0" * 64, file_sha256="0" * 64, check_sha256="0" * 64))
    assert (refused.value.fault.fault_class, refused.value.fault.retryable) == ("infra", False)

def test_a_session_that_ends_early_is_resumed_on_the_runners_clock(sandbox_bins, tmp_path):
    """Each session moves the runner's clock 400 s. After the first, 600 s of the 1,000 s
    budget are left and harbor resumes the session; after the second, 200 s, under harbor's
    300 s floor, and it stops. The transcript's timestamps say the session began in 2001, which
    an in-sandbox clock would have read as a budget long spent: nothing reads them."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 400, "stamp": "2001-01-01T00:00:00Z"},
                                               {"commit": "v2\n", "advance": 400}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    first, second = _calls(control)
    assert "--resume" not in first["args"] and second["args"][second["args"].index("--resume") + 1]
    assert second["stdin"] == "Your session ended, but its goal is not met."
    record = _record(ctx, outputs)
    assert (record.ending, record.reentries, record.sessions_ended) == ("agent_ended", 1, 1)
    assert 799 < record.agent_sec < 810
    [ended] = [e for e in _events(ctx) if e["kind"] == "build.agent_session_ended"]
    assert ended["reason"] == "budget_spent"
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)


def test_the_hard_limit_stops_an_agent_that_never_ends_and_the_build_is_still_collected(
        sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 1200, "hang": True}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert record.ending == "hard_limit" and record.agent_sec >= 1100
    assert "source_bundle" in outputs
    [cut] = [e for e in _events(ctx) if e["kind"] == "build.agent_interrupted"]
    assert cut["reason"] == "build_timeout"
    assert _ended(ctx, "sandbox")["ending"] == "planned_stop"
    assert _gone(ctx)


def test_a_build_without_a_commit_on_main_publishes_no_submission(sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"advance": 900}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    assert "source_bundle" not in outputs and "no_submission" in outputs
    nothing = json.loads(ctx.store.read_bytes(outputs["no_submission"]))
    assert nothing["kind"] == "no_main"
    assert _record(ctx, outputs).commit.status == "no_main"
    assert ctx.output_ref(outputs["no_submission"]).sha256 == sha256_file(ctx.store.path(outputs["no_submission"]))[0]


def test_an_agent_that_exits_nonzero_without_reentry_is_collected(sandbox_bins, tmp_path):
    """Without a soft budget there is no goal and no re-entry, so the CLI's nonzero exit ends the
    agent's part; it is not a fault."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "exit": 1}])
    ctx = _ctx(tmp_path, _params(soft_sec=None))
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert record.ending == "agent_nonzero_exit" and record.preflight.budget_tool == "unchecked"
    [cut] = [e for e in _events(ctx) if e["kind"] == "build.agent_interrupted"]
    assert cut["reason"] == "agent_nonzero_exit"
    assert not [c for c, _ in MappedEnvironment.SENT if "mcpServers" in c]
    assert "source_bundle" in outputs


@pytest.mark.parametrize("preflight, words", [
    ({}, "context window of 200,000"), ({"model": "claude-sonnet-4-5"}, "ran 'claude-sonnet-4-5'")],
    ids=["window", "model"])
def test_a_preflight_that_reads_back_another_window_or_model_is_infra_not_retried_and_starts_no_agent(
        sandbox_bins, tmp_path, preflight, words):
    control = sandbox_bins
    _script(control, window=1_000_000 if preflight else 200_000, preflight=preflight, claude=[{}])
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        run_stage(AgentBuildStage(), ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert words in e.value.fault.reason
    assert _calls(control) == []
    assert "build.started" not in [ev["kind"] for ev in _events(ctx)]
    assert _gone(ctx)


def test_a_runner_environment_harbor_would_read_refuses_before_any_role(sandbox_bins, tmp_path,
                                                                        monkeypatch):
    _script(sandbox_bins, window=1_000_000, claude=[{}])
    ctx = _ctx(tmp_path, _params())
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "low")
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert e.value.fault.retryable is False and "CLAUDE_CODE_EFFORT_LEVEL" in e.value.fault.reason
    assert ctx.launcher.launched == []


def test_an_allocation_end_that_leaves_no_room_for_the_agent_refuses_before_it_starts(sandbox_bins, tmp_path):
    """A build whose allocation lost time to slow launches would run the agent for hours and then
    be cut at the allocation's end, wasting its GPU time and API spend. So before the agent
    starts, once the sandbox answers and its seal is probed, the stage refuses as infra, retried,
    when the agent's hard limit and the collection after it do not fit before the allocation's end
    (``deadline_cap``), with the launch waits and the cap in the evidence (mirroring
    deploy_drive's ``room_for_readiness``). No agent command and no preflight run."""
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "v1\n"}])
    # The collection term (derived from the commit tries and the transcript) is far larger than the
    # hard limit, and the cap fits the hard limit but not the collection after it, so a build that
    # forgot to reserve the collection would not be refused (M-new-2).
    ctx = _ctx(tmp_path, _params(hard_sec=400, soft_sec=300))
    ctx.deadline_cap = ctx.deadline + 400  # room for the 400 s hard limit, not the collection
    with pytest.raises(StageFault) as e:
        run_stage(AgentBuildStage(), ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "do not fit before the allocation's end" in fault.reason
    assert fault.evidence["deadline_cap"] == str(ctx.deadline_cap)
    # The preflight and the agent's setup precede the clock, so they ran; no agent command did.
    assert _calls(control) == []
    assert "build.started" not in [ev["kind"] for ev in _events(ctx)]
    assert _gone(ctx)


def test_an_allocation_with_room_for_the_agent_runs_it_and_holds_its_deadline(sandbox_bins, tmp_path):
    """A build with room runs, and the room check holds the stage's deadline for the agent's wall
    and the collection after it (``hold_until``), grown up to the cap, so the collection is not
    cut short."""
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "v1\n", "advance": 900}])
    need = _params()["hard_sec"] + _params()["collect_allowance_sec"]
    start = time.time()
    ctx = _ctx(tmp_path, _params())
    ctx.deadline_cap = start + need + 600  # the stage's own deadline (start + 300) is shorter than need
    assert "source_bundle" in AgentBuildStage().run(ctx)
    assert "build.started" in [ev["kind"] for ev in _events(ctx)]
    # The deadline was grown to the agent's wall plus its collection, past the stage's own 300 s.
    assert ctx.deadline >= start + need - 1


def test_a_sandbox_that_dies_without_an_oom_is_infra_retried_and_collects_nothing(
        sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "kill_server": True, "hang": True}])
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "no OOM recorded" in fault.reason
    assert fault.evidence["roles"] == "sandbox"
    assert not ctx.output_path("source.bundle").exists() and "commit" not in [s.name for s in ctx.launcher.launched]
    assert ctx.running_roles() == []


class _PlantedSampler:
    """The split sampler, with one sample planted five seconds before it is read."""

    harness = 1 << 30

    def __init__(self, cgroup_dir, **kw) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def sample_once(self) -> SplitSample:
        return self.samples()[-1]

    def samples(self) -> list[SplitSample]:
        return [SplitSample(at=time.time() - 5, total=12 << 30, tmpfs=0, harness=self.harness,
                            contestant=11 << 30, oom_kills=0)]


class _NoContestantToEnd:
    """The contestant killer, finding no contestant process to end: the watch kills the step."""

    def __init__(self, cgroup_dir, **kw) -> None:
        pass

    def end_one(self) -> bool:
        return False

    def exiting(self) -> bool:
        return False

    def close(self) -> None:
        pass


def _over_limit(cgroup: Path) -> dict[str, str]:
    """The fake memory files that put a step at 20 GiB, past the sandbox's 16."""
    return {str(cgroup / "memory.current"): f"{20 * GIB}\n", str(cgroup / "memory.swap.current"): "0\n",
            str(cgroup / "memory.stat"): f"anon {20 * GIB}\nfile 0\nshmem 0\n",
            str(cgroup / "memory.events"): "oom 0\noom_kill 0\n"}


@pytest.fixture
def oom_watch(tmp_path, monkeypatch):
    """The runner's memory watch, enforcing, over fake cgroup files under ``tmp_path/cg``, the
    split sampler planted, and no contestant process the watch could end alone: a step past its
    memory is killed whole. Yields the watch and the cgroups directory."""
    monkeypatch.setattr(stage_module, "SplitSampler", _PlantedSampler)
    monkeypatch.setattr(stage_module, "ContestantKiller", _NoContestantToEnd)
    watch = MemoryWatch(enforce=True, interval_sec=0.05)
    watch.start()
    yield watch, tmp_path / "cg"
    watch.stop()


@pytest.mark.parametrize("harness_gib, contestant", [(1, True), (10, False)], ids=["within-reserve", "over-reserve"])
def test_a_sandbox_out_of_memory_after_commits_is_graded_on_its_last_commit_only_with_the_harness_in_its_reserve(
        sandbox_bins, tmp_path, monkeypatch, oom_watch, harness_gib, contestant):
    """The agent's work takes the sandbox past its memory, and the runner's watch, finding no
    contestant process to end, kills the step. The build rule reads the split: within its
    reserve the harness was not the cause, the end is the contestant's, and the build is graded
    on its last commit; over it, the end is infra and retried."""
    control = sandbox_bins
    watch, cgroups = oom_watch
    _script(control, window=1_000_000,
            claude=[{"commit": "v1\n", "touch": _over_limit(cgroups / "cg-sandbox"), "hang": True}])
    monkeypatch.setattr(_PlantedSampler, "harness", harness_gib << 30)
    ctx = _ctx(tmp_path, _params(), watch=watch,
               launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    if not contestant:
        with pytest.raises(StageFault) as e:
            AgentBuildStage().run(ctx)
        assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
        assert "reserve" in e.value.fault.reason and "commit" not in [s.name for s in ctx.launcher.launched]
        return
    outputs = AgentBuildStage().run(ctx)
    assert _ended(ctx, "sandbox")["ending"] == "memory_limit" and "transcript" in outputs
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)
    record = _record(ctx, outputs)
    assert record.ending == "contestant_oom" and record.fault is not None
    # The plan sets no rule, so the build ran under the setting's default, which the record names.
    assert record.grade_after_sandbox_oom is True
    assert (record.fault.fault_class, record.fault.evidence["sandbox.harness_bytes"]) == ("contestant", str(1 << 30))
    assert record.fault.evidence["sandbox.oom_by"] == "runner" and record.memory.oom_by == "runner"
    [cut] = [e for e in _events(ctx) if e["kind"] == "build.agent_interrupted"]
    assert cut["reason"] == "contestant_oom"


@pytest.mark.parametrize("case", ["no-commit", "unreadable"])
def test_a_sandbox_out_of_memory_with_nothing_gradable_fails_the_build_with_a_zero(
        sandbox_bins, tmp_path, monkeypatch, oom_watch, case):
    """A contestant end after which the commit point finds nothing gradable, no commit on main or
    a main the contestant left naming a missing object, is the contestant's zero: the stage
    declares the fault and raises it."""
    control = sandbox_bins
    watch, cgroups = oom_watch
    ctx = _ctx(tmp_path, _params(), watch=watch,
               launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    touch = _over_limit(cgroups / "cg-sandbox")
    if case == "no-commit":
        _script(control, window=1_000_000, claude=[{"touch": touch, "hang": True}])
    else:
        main = ctx.scratch / "roles" / "sandbox" / "workspace" / ".git" / "refs" / "heads" / "main"
        _script(control, window=1_000_000, claude=[{"commit": "v1\n", "touch": {str(main): "1" * 40 + "\n", **touch},
                                                    "hang": True}])
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable, fault.evidence["roles"]) == ("contestant", False, "sandbox")
    assert ctx.contestant_fault() == fault and not ctx.output_path("build.json").exists()
    assert [s.name for s in ctx.launcher.launched if s.name.startswith("commit")] == ["commit"]
    kind = json.loads(ctx.output_path("no_submission.json").read_text())["kind"]
    assert kind == ("no_main" if case == "no-commit" else "unreadable")


def _zeroed_by_the_oom_rule(ctx: StageContext) -> FaultRecord:
    """Run the build of *ctx*, whose sandbox the test runs out of its memory after a commit under
    a plan whose ``run.grade_after_sandbox_oom`` is false, and check that it is the contestant's
    zero with nothing graded: the stage declares the fault, which names the
    rule; no commit point, transcript copy or extraction runs, so no output is written but the
    brief's; and the build's record, kept in the attempt as evidence and never as an output, names
    the rule beside the ending and the fault. Return the fault."""
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable, fault.evidence["roles"]) == ("contestant", False, "sandbox")
    assert ctx.contestant_fault() == fault
    assert fault.evidence["grade_after_sandbox_oom"] == "false"
    assert "run.grade_after_sandbox_oom is false" in fault.reason and "nothing graded" in fault.reason
    assert _ended(ctx, "sandbox")["ending"] == "memory_limit"
    launched = [s.name for s in ctx.launcher.launched]
    assert not [name for name in launched if name.startswith("commit")]
    assert "transcript" not in launched and "extract" not in launched
    for name in (agent_build.SOURCE_BUNDLE_FILE, agent_build.NO_SUBMISSION_FILE, agent_build.BUILD_FILE):
        assert not ctx.output_path(name).exists(), name
    record = BuildRecord.model_validate_json((ctx.attempt_dir / agent_build.BUILD_FILE).read_bytes())
    assert (record.ending, record.grade_after_sandbox_oom, record.fault) == ("contestant_oom", False, fault)
    assert (record.commit, record.commit_tries, record.extraction) == (None, [], None)
    return fault


def test_a_sandbox_out_of_memory_after_commits_is_the_contestants_zero_when_the_plan_does_not_grade_it(
        sandbox_bins, tmp_path, oom_watch):
    """With the plan's ``run.grade_after_sandbox_oom`` false, the same end
    the build rule judges the contestant's, the sandbox past its own memory with the harness within
    its reserve, scores zero though the agent committed: nothing is graded."""
    control = sandbox_bins
    watch, cgroups = oom_watch
    _script(control, window=1_000_000,
            claude=[{"commit": "v1\n", "touch": _over_limit(cgroups / "cg-sandbox"), "hang": True}])
    ctx = _ctx(tmp_path, _params(), watch=watch, launcher=LocalLauncher(gpus=1, cgroup_root=cgroups),
               run={"grade_after_sandbox_oom": False})
    fault = _zeroed_by_the_oom_rule(ctx)
    assert (fault.evidence["sandbox.oom_by"], fault.evidence["sandbox.harness_bytes"]) == ("runner", str(1 << 30))


def test_the_sandbox_oom_rule_is_the_plans_whatever_the_site_file_says_since(sandbox_bins, tmp_path, monkeypatch,
                                                                             oom_watch):
    """The rule is decided from the settings the plan recorded when it was made: the site's hls.toml
    changed since, here in the working directory a command would read it from, does not change
    what a planned or resumed instance does."""
    control = sandbox_bins
    watch, cgroups = oom_watch
    site = tmp_path / "site"
    site.mkdir()
    planned = load_settings(_site_file(site / "hls.toml",
                                       _settings_table(tmp_path, run={"grade_after_sandbox_oom": False})))
    _site_file(site / "hls.toml", _settings_table(tmp_path, run={"grade_after_sandbox_oom": True}))
    monkeypatch.chdir(site)
    assert load_settings(site / "hls.toml").run.grade_after_sandbox_oom is True
    _script(control, window=1_000_000,
            claude=[{"commit": "v1\n", "touch": _over_limit(cgroups / "cg-sandbox"), "hang": True}])
    ctx = _ctx(tmp_path, _params(), watch=watch, launcher=LocalLauncher(gpus=1, cgroup_root=cgroups),
               settings=planned)
    # The plan as the store keeps it, which a runner or a resume reads, holds the value it was made with.
    assert Plan.model_validate_json(ctx.plan.model_dump_json()).settings.run.grade_after_sandbox_oom is False
    _zeroed_by_the_oom_rule(ctx)


def _failing_commit_point(monkeypatch, fails: int, control: Path, *, advance: float = 100.0) -> list[dict]:
    """Make the first *fails* runs of the commit point fail as a read the machine failed (exit 74),
    each moving the runner's clock (*control*'s offset) *advance* s, and every later one run the
    real program; no pause between them. Return the keywords each run's command was built with, in
    order, ``seconds`` among them. The runs are counted here, as the stage builds each one's
    command: a path in the container's command line is rewritten by the fake sandbox when it starts
    with ``/tmp``, so a counter file there would be a different file in every run."""
    real = agent_build.commit_command
    offset = control / "offset"
    runs: list[dict] = []

    def command(**kw):
        if "no_submission" not in kw:
            return real(**kw)
        runs.append(kw)
        if len(runs) > fails:
            return real(**kw)
        offset.write_text(str(float(offset.read_text() if offset.exists() else 0) + advance))
        return ("/bin/sh", "-c", 'echo "hls-commit: io-error: EIO: a test" >&2; exit 74', "commit")

    monkeypatch.setattr(agent_build, "commit_command", command)
    monkeypatch.setattr(agent_build, "COMMIT_PAUSE_SEC", 0.0)
    return runs


def _commit_time_left(monkeypatch) -> list[float]:
    """The time each commit try had left before the collection's last steps when the stage built
    its command (``commit_deadline`` less the stage's clock), in order."""
    left: list[float] = []
    tried = agent_build._Build.commit

    def commit(self, layout, image, name, seconds):
        left.append(self.commit_deadline() - self.ctx.clock())
        return tried(self, layout, image, name, seconds)

    monkeypatch.setattr(agent_build._Build, "commit", commit)
    return left


def _runner_clock(ctx: StageContext, control: Path) -> StageContext:
    """*ctx* with its clock moved as the fake CLIs and commit points move the runner's (*control*'s
    offset), so the stage's deadline is spent as the time they stand for passes."""
    offset = control / "offset"
    ctx.clock = lambda: time.time() + (float(offset.read_text()) if offset.exists() else 0.0)
    return ctx


@pytest.mark.parametrize("end", ["contestant_oom", "hard_limit"])
def test_a_commit_point_that_fails_as_infra_is_tried_again_on_the_same_workspace(
        sandbox_bins, tmp_path, monkeypatch, oom_watch, end):
    """Collection is the harness's work. A commit point that fails as infra after the agent's
    part is over runs again on the intact workspace, inside the attempt, with the agent's clock
    closed, so its tries spend none of it; build.json records every try. The three tries fit in
    the collection allowance."""
    control = sandbox_bins
    watch, cgroups = oom_watch
    _failing_commit_point(monkeypatch, 2, control)
    if end == "contestant_oom":
        step = {"commit": "v1\n", "touch": _over_limit(cgroups / "cg-sandbox"), "hang": True}
    else:
        step = {"commit": "v1\n", "advance": 1200, "hang": True}
    _script(control, window=1_000_000, claude=[step])
    params = _params()
    assert params["collect_allowance_sec"] == COLLECT_SEC
    ctx = _runner_clock(_ctx(tmp_path, params, watch=watch, launcher=LocalLauncher(gpus=1, cgroup_root=cgroups)),
                        control)
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert record.ending == end and [c.wrote for c in record.commit_tries] == [None, None, "bundle"]
    assert [c.role for c in record.commit_tries] == ["commit", "commit-2", "commit-3"]
    assert all("io-error: EIO: a test" in c.reason for c in record.commit_tries[:2])
    # The two failed tries moved the runner's clock 200 s after the agent's part ended.
    assert record.agent_sec < {"contestant_oom": 100, "hard_limit": 1300}[end]
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)
    assert len(_calls(control)) == 1


def test_what_a_killed_commit_try_left_in_the_outputs_is_cleared_before_each_try_and_before_publishing(
        sandbox_bins, tmp_path, monkeypatch):
    """A commit point killed outright leaves its partial bundle where it writes it, under the
    outputs' ``PARTIAL_DIR``: history no scan reads. The stage clears every ``.tmp-*`` there before
    each try, so the next finds none, and before it publishes, so nothing the last try left is
    kept."""
    control = sandbox_bins
    real = agent_build.commit_command
    runs: list[dict] = []
    partials = f"{agent_build.OUT}/{commit_program.PARTIAL_DIR}"
    left = f"{partials}/source.bundle-7"

    def command(**kw):
        if "no_submission" not in kw:
            return real(**kw)
        runs.append(kw)
        if len(runs) == 1:
            return ("/bin/sh", "-c", f'mkdir -p {partials}; printf PACK > {left}; '
                    'echo "hls-commit: io-error: EIO: a test" >&2; exit 74', "commit")
        found = f'if [ -e {left} ]; then echo "hls-commit: left behind" >&2; exit 74; fi; '
        return ("/bin/sh", "-c", found + f'mkdir -p {partials}; printf PACK > {partials}/source.bundle-8; exec "$@"',
                "commit", *real(**kw))

    monkeypatch.setattr(agent_build, "commit_command", command)
    monkeypatch.setattr(agent_build, "COMMIT_PAUSE_SEC", 0.0)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert [c.wrote for c in record.commit_tries] == [None, "bundle"], record.commit_tries
    assert sorted(p.name for p in ctx.output_path("build.json").parent.glob(".tmp-*")) == []


@pytest.mark.parametrize("why", ["shutdown", "spent"])
def test_an_attempt_the_commit_point_leaves_without_a_bundle_keeps_no_partial_bundle(sandbox_bins, tmp_path,
                                                                                   monkeypatch, why):
    """A commit try killed outright leaves its partial bundle in the outputs. When the attempt ends
    without a bundle, because the allocation is shutting down after a try (retried) or once the
    tries are spent (not retried), the stage clears it first: no attempt keeps a partial bundle for
    finalize to find."""
    control = sandbox_bins
    real = agent_build.commit_command
    partials = f"{agent_build.OUT}/{commit_program.PARTIAL_DIR}"

    def command(**kw):
        if "no_submission" not in kw:
            return real(**kw)
        return ("/bin/sh", "-c", f'mkdir -p {partials}; printf PACK > {partials}/source.bundle-7; '
                'echo "hls-commit: io-error: EIO: a test" >&2; exit 74', "commit")

    monkeypatch.setattr(agent_build, "commit_command", command)
    monkeypatch.setattr(agent_build, "COMMIT_PAUSE_SEC", 0.0)
    if why == "shutdown":
        tried = agent_build._Build.commit

        def shutting(self, layout, image, name, seconds):
            found = tried(self, layout, image, name, seconds)
            self.ctx.stopping = True
            return found

        monkeypatch.setattr(agent_build._Build, "commit", shutting)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert e.value.fault.retryable is (why == "shutdown")
    outputs = ctx.output_path("build.json").parent
    assert not (outputs / commit_program.PARTIAL_DIR).exists()
    assert sorted(p.name for p in outputs.glob(".tmp-*")) == []


@pytest.mark.parametrize("end", ["contestant_oom", "agent_ended"])
def test_spent_commit_point_tries_end_the_attempt_as_infra_never_retried_and_never_a_zero(
        sandbox_bins, tmp_path, monkeypatch, oom_watch, end):
    """Once its tries are spent, the commit point's failure ends the attempt as infra that is not
    retried, since a new attempt could only run the agent again, naming the agent's end and the
    snapshots kept; it is never the contestant's zero. build.json, beside the snapshots and never
    an output, records the agent's end and every try."""
    control = sandbox_bins
    watch, cgroups = oom_watch
    _failing_commit_point(monkeypatch, 99, control)
    if end == "contestant_oom":
        step, params = {"commit": "v1\n", "touch": _over_limit(cgroups / "cg-sandbox"), "hang": True}, _params()
    else:
        step, params = {"commit": "v1\n", "sleep": 2.5, "advance": 900}, _params(snapshot_interval_sec=1)
    _script(control, window=1_000_000, claude=[step])
    ctx = _ctx(tmp_path, params, watch=watch, launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert f"failed 3 times after the agent's part ended ({end})" in fault.reason
    assert fault.evidence["agent_ending"] == end and "commit-3: " in fault.evidence["commit_tries"]
    if end == "agent_ended":
        assert fault.evidence["snapshots"].startswith("snapshots/main-") and "Snapshots kept" in fault.reason
    else:
        assert fault.evidence["snapshots"] == "none"
    assert ctx.contestant_fault() is None
    assert [s.name for s in ctx.launcher.launched if s.name.startswith("commit")] == ["commit", "commit-2",
                                                                                       "commit-3"]
    evidence = BuildRecord.model_validate_json((ctx.attempt_dir / "build.json").read_text())
    assert (evidence.ending, evidence.commit) == (end, None) and not ctx.output_path("build.json").exists()
    assert [(c.role, c.exit_code, c.wrote) for c in evidence.commit_tries] == [
        ("commit", 74, None), ("commit-2", 74, None), ("commit-3", 74, None)]
    assert all("io-error: EIO: a test" in c.reason for c in evidence.commit_tries)


def test_each_commit_try_runs_with_the_time_left_before_the_collections_last_steps(sandbox_bins, tmp_path,
                                                                                   monkeypatch):
    """A commit try gets no time of its own: its program is handed the time left before the
    transcript copy and the extraction must start (``commit_deadline``, ``AFTER_COMMIT_SEC`` before
    the stage's deadline), whole seconds of it, so a later try, after one that ran long, gets that
    much less. While time is left the later try runs, whatever the failed one took."""
    control = sandbox_bins
    runs = _failing_commit_point(monkeypatch, 1, control, advance=COLLECT_SEC / 4)
    left = _commit_time_left(monkeypatch)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _runner_clock(_ctx(tmp_path, _params()), control)
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert [(c.role, c.wrote) for c in record.commit_tries] == [("commit", None), ("commit-2", "bundle")]
    assert [kw["seconds"] for kw in runs] == [math.floor(s) for s in left]
    # The agent ended early, so the first try has nearly all of the allowance beside the fixed steps.
    assert COLLECT_SEC - agent_build.AFTER_COMMIT_SEC < left[0] <= COLLECT_SEC - agent_build.AFTER_COMMIT_SEC + 200
    assert left[1] <= left[0] - COLLECT_SEC / 4
    # The real program ran the second try with the time it was handed.
    records = fake_sandbox.records(Path(os.environ["FAKE_SANDBOX_RECORDS"]))
    command = records["commit-2"]["command"]
    assert command[command.index("--seconds") + 1] == str(runs[1]["seconds"])
    assert not [e for e in _events(ctx) if e["kind"] == "build.commit_point_out_of_time"]


def test_a_commit_try_that_runs_out_of_the_time_left_ends_the_build_void_with_no_later_try(
        sandbox_bins, tmp_path, monkeypatch):
    """When a failed try leaves no time before the collection's last steps, no later try starts:
    the attempt ends infra that is not retried, a void, never the contestant's zero, and the
    ``build.commit_point_out_of_time`` event says how many tries ran and what was left."""
    control = sandbox_bins
    runs = _failing_commit_point(monkeypatch, 99, control, advance=COLLECT_SEC)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _runner_clock(_ctx(tmp_path, _params()), control)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert len(runs) == 1
    assert [s.name for s in ctx.launcher.launched if s.name.startswith("commit")] == ["commit"]
    assert "the commit point failed 1 time after the agent's part ended (agent_ended)" in fault.reason
    assert "no time was left for a later try before the collection's last steps" in fault.reason
    assert ctx.contestant_fault() is None
    [out] = [e for e in _events(ctx) if e["kind"] == "build.commit_point_out_of_time"]
    assert out["tries"] == 1 and out["left_sec"] < 1


def test_a_commit_try_still_running_past_its_time_is_ended_and_the_build_is_void(sandbox_bins, tmp_path,
                                                                                 monkeypatch):
    """A try that never exits, as one whose read of the workspace hangs would not, is ended by the
    stage ``COMMIT_WIND_DOWN_SEC`` past the seconds it was handed, counted from its command's start,
    not at the stage's deadline, which lies the collection's last steps beyond. It ran out of the
    time left, so no later try starts, and the build is void."""
    control = sandbox_bins
    offset = control / "offset"
    real = agent_build.commit_command
    runs: list[dict] = []

    def command(**kw):
        if "no_submission" not in kw:
            return real(**kw)
        runs.append(kw)
        if len(runs) == 1:
            return ("/bin/sh", "-c", 'echo "hls-commit: io-error: EIO: a test" >&2; exit 74', "commit")
        return ("/bin/sh", "-c", "exec sleep 600", "commit")

    tried = agent_build._Build.commit

    def commit(self, layout, image, name, seconds):
        found = tried(self, layout, image, name, seconds)
        if name == "commit":
            # The first try leaves the next one four seconds before the collection's last steps.
            moved = self.commit_deadline() - self.ctx.clock() - 4
            offset.write_text(str(float(offset.read_text() if offset.exists() else 0) + moved))
        return found

    monkeypatch.setattr(agent_build, "commit_command", command)
    monkeypatch.setattr(agent_build._Build, "commit", commit)
    monkeypatch.setattr(agent_build, "COMMIT_PAUSE_SEC", 0.0)
    monkeypatch.setattr(agent_build, "COMMIT_WIND_DOWN_SEC", 1)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _runner_clock(_ctx(tmp_path, _params()), control)
    began = time.monotonic()
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    # The stage's deadline was over 1,280 s away when the hung try started.
    assert time.monotonic() - began < agent_build.AFTER_COMMIT_SEC / 4
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert "no time was left for a later try before the collection's last steps" in fault.reason
    assert ctx.contestant_fault() is None
    assert len(runs) == 2 and runs[1]["seconds"] <= 4
    evidence = BuildRecord.model_validate_json((ctx.attempt_dir / "build.json").read_text())
    hung = evidence.commit_tries[1]
    assert (hung.role, hung.ending, hung.wrote) == ("commit-2", "time_limit", None)
    assert f"still running past the {runs[1]['seconds']} s it was given, and the stage ended it" in hung.reason


def test_a_sandbox_that_ends_before_the_agents_clock_is_infra_whatever_its_roles_say(
        sandbox_bins, tmp_path, oom_watch):
    """Before the clock the agent never ran: a sandbox out of memory during the preflight is
    the harness's, retried, however the build rule reads it."""
    control = sandbox_bins
    watch, cgroups = oom_watch
    _script(control, window=1_000_000, preflight={"touch": _over_limit(cgroups / "cg-sandbox"), "hang": True},
            claude=[{}])
    ctx = _ctx(tmp_path, _params(), watch=watch,
               launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "the agent never ran" in fault.reason
    assert fault.evidence["sandbox_end.roles"] == "sandbox" and ctx.contestant_fault() is None
    assert "commit" not in [s.name for s in ctx.launcher.launched] and _calls(control) == []


@pytest.mark.parametrize("timing", ["before-the-bundle", "after-the-bundle"])
def test_a_sigterm_during_collection_after_a_contestant_end_is_infra_and_never_the_zero(
        sandbox_bins, tmp_path, monkeypatch, oom_watch, timing):
    """The runner's records of a SIGTERM read ``contestant_fault``, which for this stage is only
    the zero it declared: during collection there is none. The stage's own fault is infra,
    retried before the commit point wrote the bundle and not after it."""
    control = sandbox_bins
    watch, cgroups = oom_watch
    _script(control, window=1_000_000,
            claude=[{"commit": "v1\n", "touch": _over_limit(cgroups / "cg-sandbox"), "hang": True}])
    ctx = _ctx(tmp_path, _params(), watch=watch,
               launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    read: list = []
    if timing == "before-the-bundle":
        real = agent_build.commit_command
        monkeypatch.setattr(agent_build, "commit_command",
                            lambda **kw: ("/bin/sh", "-c", 'sleep 3; exec "$@"', "commit", *real(**kw))
                            if "no_submission" in kw else real(**kw))
        start = ctx.start_container

        def sigterm_in_the_commit_point(name, *args, **kw):
            handle = start(name, *args, **kw)
            if name == "commit":
                read.append(ctx.contestant_fault())
                ctx.stop_roles()
            return handle

        monkeypatch.setattr(ctx, "start_container", sigterm_in_the_commit_point)
    else:
        start = ctx.start_role

        def sigterm_after_the_bundle(name, *args, **kw):
            if name == "transcript":
                # The roles reported so far hold the sandbox's contestant end; the stage declared none.
                read.append(ctx.contestant_fault())
                ctx.stop_roles()
            return start(name, *args, **kw)

        monkeypatch.setattr(ctx, "start_role", sigterm_after_the_bundle)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert read == [None] and ctx.contestant_fault() is None and fault.fault_class == "infra"
    if timing == "before-the-bundle":
        assert fault.retryable and "shutting down" in fault.reason
        evidence = BuildRecord.model_validate_json((ctx.attempt_dir / "build.json").read_text())
        assert evidence.ending == "contestant_oom" and [c.wrote for c in evidence.commit_tries] == [None]
    else:
        assert not fault.retryable and "after its commit point" in fault.reason


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_a_cli_killed_by_an_oom_kill_of_its_own_with_budget_left_is_re_entered(sandbox_bins, tmp_path, monkeypatch,
                                                                                 arm):
    """An OOM kill of the CLI, a contestant process in the spawner's namespace, leaves the
    container running: the session ended early, and harbor resumes it with the fixed sentence on
    the runner's clock, as for any early end, for both harnesses."""
    control = sandbox_bins
    monkeypatch.setattr(claude_code, "REENTRY_ERROR_PAUSE_SEC", 0.0)
    monkeypatch.setattr(harnesses, "REENTRY_ERROR_PAUSE_SEC", 0.0)
    if arm is OPUS:
        _script(control, window=1_000_000, claude=[{"commit": "v1\n", "oom_self": True},
                                                   {"commit": "v2\n", "advance": 900}])
    else:
        _script(control, window=258_400, codex=[{"commit": "v1\n"}, {"oom_self": True},
                                                {"commit": "v2\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params(arm))
    outputs = AgentBuildStage().run(ctx)
    calls = _calls(control)
    if arm is OPUS:
        assert len(calls) == 2 and "--resume" in calls[1]["args"]
        assert calls[1]["stdin"] == "Your session ended, but its goal is not met."
    else:
        assert len(calls) == 3 and calls[2]["args"][:3] == ["exec", "resume", "0199aaaa-0000-7000-8000-000000000001"]
    record = _record(ctx, outputs)
    assert (record.ending, record.reentries, record.fault) == ("agent_ended", 1, None)
    assert _ended(ctx, "sandbox")["ending"] == "planned_stop"
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)
    [entered] = [e for e in _events(ctx) if e["kind"] == "build.agent_reentered"]
    assert entered["exit_code"] != 0


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


def test_an_oom_that_ends_a_test_server_the_agent_started_leaves_the_build_running(
        sandbox_bins, tmp_path, monkeypatch, oom_watch):
    """A breach of the sandbox's memory while the CLI runs ends the contestant's largest
    process, here the test server the agent started; the sandbox lives, and the build runs to
    its normal end."""
    control = sandbox_bins
    watch, cgroups = oom_watch
    cgroup = cgroups / "cg-sandbox"

    class _EndsTheServer(_NoContestantToEnd):
        def end_one(self) -> bool:
            os.kill(int((control / "server.pid").read_text()), signal.SIGKILL)
            (cgroup / "memory.current").write_text(f"{GIB}\n")
            (cgroup / "memory.stat").write_text(f"anon {GIB}\nfile 0\nshmem 0\n")
            return True

    monkeypatch.setattr(stage_module, "ContestantKiller", _EndsTheServer)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "server": True, "touch": _over_limit(cgroup),
                                                "sleep": 3, "advance": 900}])
    ctx = _ctx(tmp_path, _params(), watch=watch,
               launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    outputs = AgentBuildStage().run(ctx)
    assert not _alive(int((control / "server.pid").read_text()))
    record = _record(ctx, outputs)
    assert (record.ending, record.fault) == ("agent_ended", None)
    assert record.memory.process_kill_count == 1 and record.memory.oom_by is None
    assert _ended(ctx, "sandbox")["ending"] == "planned_stop" and "source_bundle" in outputs


@pytest.mark.parametrize("graded", [True, False], ids=["oom-graded", "oom-zero"])
def test_a_sandbox_that_writes_past_its_disk_after_commits_is_graded_on_its_last_commit(
        sandbox_bins, tmp_path, monkeypatch, graded):
    """The sandbox's declared disk caps its writable directories, which hold nothing the
    harness writes; a write past it ends the build as the contestant's, and the build is graded
    on its last commit, whatever the plan's rule for a sandbox's end on its memory.
    What the sandbox wrote is removed before the commit point runs, but
    for the repository and the logs."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "write": 3 << 20, "hang": True}])
    watch = MemoryWatch(enforce=False, interval_sec=0.05, disk_interval_sec=0.1)
    walked: dict[str, tuple] = {}
    watch_disk = watch.watch_disk

    def capped(step, paths, limit_bytes, kill, **kw):
        walked[kw["role"]] = (tuple(paths), limit_bytes)
        # The test cannot write 8 GiB: the cap the context asked for, scaled to 1 MiB.
        watch_disk(step, paths, limit_bytes // (8 * 1024), kill, **kw)

    monkeypatch.setattr(watch, "watch_disk", capped)
    watch.start()
    try:
        ctx = _ctx(tmp_path, _params(), watch=watch, run={"grade_after_sandbox_oom": graded})
        outputs = AgentBuildStage().run(ctx)
    finally:
        watch.stop()
    record = _record(ctx, outputs)
    assert record.ending == "disk_limit" and _ended(ctx, "sandbox")["ending"] == "disk_limit"
    assert record.grade_after_sandbox_oom is graded
    assert (record.fault.fault_class, record.fault.retryable) == ("contestant", False)
    assert "bytes in its writable directories over 8 GiB" in record.fault.reason
    layout = SandboxLayout(ctx.scratch / "roles" / "sandbox")
    assert walked == {"sandbox": (layout.writable(), 8 * GIB)}
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)
    assert not (layout.workspace / "blob.bin").exists() and (layout.workspace / ".git").is_dir()
    assert "build.sandbox_cleared" in [e["kind"] for e in _events(ctx)]


def test_a_sandbox_that_ends_before_the_clock_is_judged_by_its_roles(sandbox_bins, tmp_path):
    """The preflight's CLI kills the exec server: the setup fails because the sandbox ended, and
    the sandbox's roles, not the setup, say whose fault that is."""
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={"kill_server": True, "hang": True}, claude=[{}])
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and fault.evidence["roles"] == "sandbox"
    assert "build.started" not in [ev["kind"] for ev in _events(ctx)] and _calls(control) == []


def test_an_arm_the_stage_holds_no_credential_for_is_refused_before_the_sandbox(sandbox_bins, tmp_path):
    _script(sandbox_bins, window=1_000_000, claude=[{}])
    ctx = _ctx(tmp_path, _params())
    ctx.secrets = {}
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert e.value.fault.retryable is False and "cannot be built" in e.value.fault.reason
    assert "sandbox" not in [s.name for s in ctx.launcher.launched]


def test_a_route_whose_base_url_grant_is_missing_is_refused_before_the_sandbox(sandbox_bins, tmp_path):
    """A record that names a base URL secret the stage holds no value for would open the
    harness's own host in its place: refused, not retried, before any role, never showing a
    value."""
    _script(sandbox_bins, window=258_400, codex=[{}])
    ctx = _ctx(tmp_path, _params(GPT))
    ctx.secrets = {k: v for k, v in ctx.secrets.items() if k != "OPENAI_BASE_URL"}
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert "OPENAI_BASE_URL" in fault.reason and "openai_url" in fault.reason and HOST not in fault.reason
    assert ctx.launcher.launched == []


def test_a_derailed_session_stops_reentry(sandbox_bins, tmp_path):
    """The first session writes one turn again and again and runs past a heartbeat; the health
    check stops re-entry, so the session is not resumed though most of the budget is left."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "turns": 25, "kind": "repeat", "sleep": 2.5},
                                               {"commit": "v2\n"}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    assert len(_calls(control)) == 1
    record = _record(ctx, outputs)
    assert record.derailed == "repeat" and record.reentries == 0
    kinds = [e["kind"] for e in _events(ctx)]
    assert "build.agent_derailed" in kinds and "build.heartbeat" in kinds
    [beat] = [e for e in _events(ctx) if e["kind"] == "build.heartbeat"][:1]
    assert beat["log_bytes"] > 0 and beat["turns"] == 25 and beat["remaining_sec"] > 0


@pytest.mark.parametrize("arm, script", [
    (OPUS, {"window": 1_000_000, "claude": [{"commit": "v1\n", "advance": 900}]}),
    (GPT, {"window": 258_400, "codex": [{"commit": "v1\n"}, {"advance": 900}]}),
], ids=["claude-code", "codex"])
def test_agent_env_names_every_variable_of_every_exec_request_and_no_value(sandbox_bins, tmp_path, arm, script):
    """``agent_env.json`` names each variable the agent's requests carry, harbor's own
    launch variables among them (its retry watchdog, the model, the config directory), with where
    each came from, and never a value."""
    control = sandbox_bins
    _script(control, **script)
    ctx = _ctx(tmp_path, _params(arm))
    outputs = AgentBuildStage().run(ctx)
    recorded = json.loads(ctx.store.read_bytes(outputs["agent_env"]))["exec"]
    sent = {var for _, env in MappedEnvironment.SENT for var in env}
    assert sent and sent <= set(recorded), sorted(sent - set(recorded))
    assert {source for var, source in recorded.items() if var not in {*adapter_for(arm).grants(arm.route_table),
                                                                      *adapter_for(arm).plain_env(arm)}} == {"harness"}
    text = json.dumps(recorded)
    assert not any(value in text for value in ctx.secrets.values())


def test_a_codex_arm_gets_its_window_settings_and_the_budget_server(sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=258_400, codex=[{"commit": "v1\n"}, {"advance": 900}])
    ctx = _ctx(tmp_path, _params(GPT))
    outputs = AgentBuildStage().run(ctx)
    [pre] = _calls(control, preflight=True)
    calls = _calls(control)
    assert [c["args"][:2] for c in calls] == [["exec", "--dangerously-bypass-approvals-and-sandbox"],
                                              ["exec", "resume"]]
    for c in [pre, *calls]:
        assert "-c" in c["args"] and "model_context_window=272000" in c["args"]
        assert "model_max_output_tokens=128000" in c["args"] and c["credential_set"]
        assert c["env"]["OPENAI_BASE_URL"] == RELAY_URL
    register = [c for c, _ in MappedEnvironment.SENT if "[mcp_servers.budget]" in c]
    assert register and all('command = "/opt/hls-budget/get-budget-mcp"' in c for c in register)
    record = _record(ctx, outputs)
    assert record.preflight.ok and record.preflight.window == 258_400 and record.ending == "agent_ended"
    assert "source_bundle" in outputs


def test_the_route_rides_harbors_exec_requests_and_no_secret_reaches_the_sandboxs_env_file(
        sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    # A value the stage holds that is not the arm's credential never reaches the sandbox.
    ctx = _ctx(tmp_path, _params(), extra_secrets={"HF_TOKEN": "hf-test-0123456789"})
    outputs = AgentBuildStage().run(ctx)
    sandbox = fake_sandbox.records(Path(os.environ["FAKE_SANDBOX_RECORDS"]))["sandbox"]
    assert set(sandbox["env"]) == {"HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HLS_WEIGHTS_DIR", "HLS_EXEC_TOKEN",
                                   "NO_PROXY", "no_proxy", "PIP_INDEX_URL", "UV_INDEX_URL", "PIP_TRUSTED_HOST",
                                   "UV_INSECURE_HOST"}
    assert sandbox["drop_caps"] == "CAP_NET_ADMIN,CAP_NET_RAW,CAP_SYS_RESOURCE" and sandbox["cwd"] == "/workspace"
    # The credential and the output limit go with every exec request harbor makes.
    launch = [e for c, e in MappedEnvironment.SENT if "exec claude " in c or "claude --verbose" in c]
    assert len(launch) == 2 and all(e["ANTHROPIC_API_KEY"] == KEY for e in launch)
    assert all(e["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000" for e in launch)
    [pre] = _calls(control, preflight=True)
    [build] = _calls(control)
    assert pre["credential_set"] and build["credential_set"] and not build["token_seen"]
    variables = json.loads(ctx.store.read_bytes(outputs["agent_env"]))
    route = {var: source for var, source in variables["exec"].items() if source != "harness"}
    assert route == {"ANTHROPIC_API_KEY": "grant:anthropic_key", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "plain",
                     "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "plain"}
    assert variables["container"]["HLS_EXEC_TOKEN"] == "harness" and variables["container"]["PIP_INDEX_URL"] == "seal"
    assert variables["container"]["HLS_WEIGHTS_DIR"] == "plain" and "HF_TOKEN" not in json.dumps(variables)
    token = sandbox["env"]["HLS_EXEC_TOKEN"]
    for name in ("agent_env", "build", "brief", "transcript"):
        text = ctx.store.read_bytes(outputs[name]).decode()
        assert KEY not in text and token not in text, name
    for log in (p for p in (ctx.attempt_dir / "logs").iterdir() if p.is_file()):
        assert KEY not in log.read_text(errors="replace"), log.name


def test_the_sandboxs_index_refuses_the_producers_deny_list(sandbox_bins, tmp_path):
    """The sandbox's seal serves the package index, whose policy refuses every name the
    producer's own deny list declares."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    AgentBuildStage().run(ctx)
    [spec] = [s for s in ctx.launcher.launched if s.name == "sandbox"]
    policy = json.loads(spec.argv[spec.argv.index("--policy-json") + 1]) if "--policy-json" in spec.argv else None
    sealed = policy or next(json.loads(a) for a in spec.argv if a.startswith("{") and '"packages"' in a)
    assert sealed["network"]["packages"] == {"mode": "deny", "entries": ["vllm"]}
    assert sealed["network"]["domains"]["entries"] == ["api.anthropic.com"]


def test_the_transcript_is_scrubbed_by_a_host_role_given_the_value_in_its_env_file_only(
        sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900, "leak": "ANTHROPIC_API_KEY",
                                                "link": "/etc/hostname"}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    index = json.loads(ctx.store.read_bytes(outputs["transcript"]))
    copied = {f["path"] for f in index["files"]}
    assert {"claude-code.txt", "printed-env.txt"} <= copied and "planted-link" not in copied
    assert {"path": "planted-link", "why": "symlink"} in index["skipped"]
    out = ctx.store.path(outputs["transcript"]).parent / "logs"
    assert (out / "printed-env.txt").read_text() == "KEY=<redacted:anthropic_key>\n"
    assert not (out / "planted-link").exists() and not (out / "planted-link").is_symlink()
    [spec] = [s for s in ctx.launcher.launched if s.name == "transcript"]
    assert spec.owner == "harness" and not spec.env_fd and spec.env_file is not None
    assert spec.argv[1:3] == ("-m", "hls.agent.transcript") and "ANTHROPIC_API_KEY=anthropic_key" in spec.argv
    assert not any(KEY in a for a in spec.argv) and not spec.env_file.exists()
    # Harbor read the trajectory from the scrubbed copy and wrote it beside the index, outside
    # the names the agent's logs can take.
    trajectory = out.parent / "trajectory.json"
    assert trajectory.is_file() and KEY not in trajectory.read_text()
    assert not (out / "trajectory.json").exists()
    assert ctx.store.path(outputs["trajectory"]) == trajectory


# ---------------------------------------------------------------------------
# Harbor's reading of the trajectory and the spend, in a role of its own
# ---------------------------------------------------------------------------


def _never_in_the_runner(self, context):
    raise AssertionError("harbor parsed the agent's logs in the runner")


def test_the_trajectory_and_spend_are_read_by_a_role_of_their_own_and_the_trajectory_pinned_by_digest(
        sandbox_bins, tmp_path, monkeypatch):
    """Harbor reads the agent's run from the transcript's copy in the extraction role, on its
    own memory (``extract_shape``), never in the runner. Its trajectory is an output by the role's
    own digest, and the spend it read is the record's."""
    control = sandbox_bins
    monkeypatch.setattr(claude_code.ClaudeCode, "populate_context_post_run", _never_in_the_runner)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "turns": 3, "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    assert outputs["trajectory"] == ctx.output_key("transcript/trajectory.json")
    trajectory = ctx.store.path(outputs["trajectory"])
    sha = sha256_file(trajectory)[0]
    assert ctx.output_ref(outputs["trajectory"]).sha256 == sha
    assert json.loads(trajectory.read_text())["steps"]
    extraction = _record(ctx, outputs).extraction
    assert (extraction.trajectory_sha256, extraction.unavailable) == (sha, None)
    assert extraction.usage is not None and extraction.usage.cost_usd == 0.5
    [finished] = [e for e in _events(ctx) if e["kind"] == "build.finished"]
    # The event says the spend the record holds, field by field.
    usage = extraction.usage
    assert (finished["cost_usd"], finished["n_input_tokens"], finished["n_output_tokens"]) == (
        0.5, usage.n_input_tokens, usage.n_output_tokens)
    [spec] = [s for s in ctx.launcher.launched if s.name == "extract"]
    assert spec.owner == "harness" and spec.env_file is None
    assert spec.shape.memory_gib == EXTRACT.memory_gib and spec.shape.disk_gib == 0
    assert spec.argv[1:3] == ("-m", "hls.agent.extract")
    assert spec.argv[spec.argv.index("--harness") + 1] == "claude-code"
    assert spec.argv[spec.argv.index("--model") + 1] == "claude-opus-4-8[1m]"
    # The role's files were made before the sandbox started, as the transcript role's were.
    started = [e["role"] for e in _events(ctx) if e["kind"] == "role_started"]
    assert started.index("sandbox") < started.index("extract")


class _RefusingExtract(LocalLauncher):
    """A local launcher whose scheduler refuses the extraction role's step."""

    def launch(self, spec, **kw):
        if spec.name == "extract":
            raise StageFault(infra_fault("the scheduler refused the step", retryable=True))
        return super().launch(spec, **kw)


def test_an_extraction_that_cannot_start_leaves_the_trajectory_and_spend_unavailable_and_the_build_graded(
        sandbox_bins, tmp_path):
    """A failed start of the extraction role is the extraction's failure like any other: past the
    commit point it never voids the build."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params(), launcher=_RefusingExtract(gpus=1))
    outputs = AgentBuildStage().run(ctx)
    assert "trajectory" not in outputs and {"source_bundle", "transcript", "build"} <= set(outputs)
    extraction = _record(ctx, outputs).extraction
    assert (extraction.trajectory_sha256, extraction.usage) == (None, None)
    assert extraction.unavailable == "the extraction role could not start: the scheduler refused the step"
    [event] = [e for e in _events(ctx) if e["kind"] == "build.extraction_unavailable"]
    assert event["reason"] == extraction.unavailable


def test_a_trajectory_past_its_bound_is_unavailable_and_the_spend_still_read(sandbox_bins, tmp_path, monkeypatch):
    """The stage holds the reported trajectory to ``TRAJECTORY_MAX_BYTES`` too, as the role does:
    one past it is no output, the trajectory alone is unavailable, with why, and the spend harbor
    read is the record's."""
    monkeypatch.setattr(agent_build, "TRAJECTORY_MAX_BYTES", 64)
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "turns": 3, "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    assert "trajectory" not in outputs
    assert not (ctx.store.path(outputs["transcript"]).parent / "trajectory.json").exists()
    extraction = _record(ctx, outputs).extraction
    assert (extraction.trajectory_sha256, extraction.unavailable) == (None, None)
    assert extraction.usage is not None and extraction.usage.cost_usd == 0.5
    assert extraction.trajectory_unavailable.startswith("harbor's trajectory took ")
    assert extraction.trajectory_unavailable.endswith(" bytes, past the 64 the build keeps")
    [event] = [e for e in _events(ctx) if e["kind"] == "build.trajectory_unavailable"]
    assert event["reason"] == extraction.trajectory_unavailable


class _Sleeper:
    """Stands in for the extraction role's program: sleeps until killed, as a role reading a huge
    transcript would."""

    path = Path(__file__).resolve().parent / "fake_extract_sleep.py"


@pytest.mark.parametrize("how", ["kernel-oom", "runner-oom", "failed", "time", "oversized"])
def test_an_extraction_that_fails_leaves_the_trajectory_and_spend_unavailable_and_the_build_graded(
        sandbox_bins, tmp_path, monkeypatch, oom_watch, how):
    """The agent writes a session file of nested JSON, which harbor's reader holds at many
    times its size, and the extraction role runs out of its memory: killed by the kernel at its
    step's limit, or by the runner's watch where swap is unconstrained. Its failure, or its running
    past ``EXTRACT_SEC``, or a record longer than the stage reads, records the trajectory and spend
    as unavailable, with why, and what it wrote of a trajectory is no output; the build is
    collected and graded on its commit: the submission was taken at the commit point, so an
    extraction never fails or voids the build."""
    control = sandbox_bins
    watch, cgroups = oom_watch
    knobs: dict = {"cgroup_root": cgroups}
    if how == "kernel-oom":
        knobs["out_of_memory"] = ("extract",)
    elif how == "runner-oom":
        knobs["replace_modules"] = {"hls.agent.extract": _Sleeper.path}
        for path, text in _over_limit(cgroups / "cg-extract").items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
    elif how == "failed":
        knobs["replace_modules"] = {"hls.agent.extract": Path(__file__).resolve().parent / "fake_extract_fail.py"}
    elif how == "oversized":
        knobs["replace_modules"] = {"hls.agent.extract": Path(__file__).resolve().parent / "fake_extract_oversized.py"}
    else:
        knobs["replace_modules"] = {"hls.agent.extract": _Sleeper.path}
        monkeypatch.setattr(agent_build, "EXTRACT_SEC", 2.0)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "nested": 8 << 20, "advance": 900}])
    ctx = _ctx(tmp_path, _params(), watch=watch, launcher=LocalLauncher(gpus=1, **knobs))
    outputs = AgentBuildStage().run(ctx)
    assert "trajectory" not in outputs and {"source_bundle", "transcript", "build"} <= set(outputs)
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)
    record = _record(ctx, outputs)
    assert record.ending == "agent_ended"
    extraction = record.extraction
    assert (extraction.trajectory_sha256, extraction.usage) == (None, None)
    why = {"kernel-oom": "ran out of memory", "runner-oom": "memory_limit", "failed": "RuntimeError: boom",
           "time": "ran past its 2 s", "oversized": f"longer than {agent_build.EXTRACT_RECORD_MAX_BYTES} bytes"}[how]
    assert why in extraction.unavailable, extraction.unavailable
    [event] = [e for e in _events(ctx) if e["kind"] == "build.extraction_unavailable"]
    assert event["reason"] == extraction.unavailable
    assert not (ctx.store.path(outputs["transcript"]).parent / "trajectory.json").exists()
    assert ctx.running_roles() == []


def test_snapshots_of_main_are_taken_while_the_agent_works_the_newest_two_kept_and_dropped_once_main_is_bundled(
        sandbox_bins, tmp_path):
    """Snapshots are evidence of the build in progress, of which the newest two are kept while the
    agent works. Once the commit point has bundled main, they are superseded: the stage removes
    them, so an attempt that collected a bundle stores the bundle alone."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 4.5}])
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    outputs = AgentBuildStage().run(ctx)
    snaps = [e for e in _events(ctx) if e["kind"] == "build.snapshot"]
    assert len(snaps) >= 2 and all(e["status"] == "ok" for e in snaps)
    taken = {snapshot_name(e["n"]) for e in snaps}
    kept = taken - {name for e in snaps for name in e["pruned"]}
    assert 1 <= len(kept) <= 2
    [dropped] = [e for e in _events(ctx) if e["kind"] == "build.snapshots_dropped"]
    assert sorted(dropped["names"]) == sorted(kept)
    assert list((ctx.attempt_dir / "snapshots").glob("main-*.bundle")) == []
    assert "source_bundle" in outputs


def test_a_partial_the_stage_cannot_remove_once_main_is_bundled_is_an_event_and_the_build_stands(
        sandbox_bins, tmp_path, monkeypatch):
    """Once the bundle is collected the stage clears what a killed commit point may have left in
    the outputs; a store that refuses (a stale handle) leaves it where finalize reads nothing, so
    the failure is the ``build.partials_left`` event and the collected build stands."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n"}])
    ctx = _ctx(tmp_path, _params())
    real = agent_build._Build.clear_partials

    def clear(self):
        if (self.out / agent_build.SOURCE_BUNDLE_FILE).is_file():
            raise OSError(errno.ESTALE, os.strerror(errno.ESTALE), str(self.out / ".tmp-partial"))
        real(self)

    monkeypatch.setattr(agent_build._Build, "clear_partials", clear)
    outputs = AgentBuildStage().run(ctx)
    assert {"source_bundle", "brief", "transcript", "build"} <= set(outputs)
    assert _record(ctx, outputs).fault is None
    [left] = [e for e in _events(ctx) if e["kind"] == "build.partials_left"]
    assert left["error"].startswith(f"OSError: [Errno {errno.ESTALE}] {os.strerror(errno.ESTALE)}: ")
    assert ctx.running_roles() == []


def test_a_snapshot_the_stage_cannot_remove_once_main_is_bundled_is_an_event_and_the_build_stands(
        sandbox_bins, tmp_path, monkeypatch):
    """The store refuses to remove the snapshots (a stale handle) after the commit point bundled
    main: they are superseded and quarantined whether or not they stay, so the failure is the
    ``build.snapshots_left`` event, naming each snapshot left with the error, and the build is
    collected as if they had gone."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 4.5}])
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    snapshots = ctx.attempt_dir / "snapshots"
    real_drop, real_unlink = agent_build._Build.drop_snapshots, Path.unlink

    def stale(path, missing_ok=False):
        if path.parent == snapshots:
            raise OSError(errno.ESTALE, os.strerror(errno.ESTALE), str(path))
        real_unlink(path, missing_ok=missing_ok)

    def drop(self):
        # The store fails only while the stage drops them, not while it prunes them.
        with monkeypatch.context() as m:
            m.setattr(Path, "unlink", stale)
            real_drop(self)

    monkeypatch.setattr(agent_build._Build, "drop_snapshots", drop)
    outputs = AgentBuildStage().run(ctx)
    assert {"source_bundle", "brief", "brief_provenance", "served_ucounts", "agent_env", "transcript",
            "build"} <= set(outputs)
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)
    assert _record(ctx, outputs).fault is None
    kept = sorted(path.name for path in snapshots.glob("main-*.bundle"))
    assert kept
    [dropped] = [e for e in _events(ctx) if e["kind"] == "build.snapshots_dropped"]
    assert dropped["names"] == []
    [left] = [e for e in _events(ctx) if e["kind"] == "build.snapshots_left"]
    assert left["paths"] == [f"snapshots/{name}" for name in kept]
    assert left["error"].startswith(f"OSError: [Errno {errno.ESTALE}] {os.strerror(errno.ESTALE)}: ")
    assert ctx.running_roles() == []


def test_a_commit_point_that_wrote_no_bundle_leaves_the_snapshots_as_evidence(sandbox_bins, tmp_path):
    """Nothing gradable on main at the end (the agent left main naming a missing object after the
    snapshots took it whole): the snapshots of the build in progress stay as its evidence."""
    control = sandbox_bins
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    main = ctx.scratch / "roles" / "sandbox" / "workspace" / ".git" / "refs" / "heads" / "main"
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 2.5},
                                               {"touch": {str(main): "1" * 40 + "\n"}, "advance": 1200}])
    outputs = AgentBuildStage().run(ctx)
    assert "no_submission" in outputs and "source_bundle" not in outputs
    assert list((ctx.attempt_dir / "snapshots").glob("main-*.bundle"))
    assert not [e for e in _events(ctx) if e["kind"] == "build.snapshots_dropped"]


def test_a_snapshot_past_a_limit_is_reported_and_the_build_goes_on(sandbox_bins, tmp_path, monkeypatch):
    """A snapshot the commit program found past one of its limits kept no bundle; the stage
    reports it beside ``build.snapshot`` with the program's reason scrubbed, and the build goes on
    to its commit point."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 2.5}])
    # The record's reason names a secret value, as a reason quoting a name the contestant chose
    # can; the container puts it together from two halves, since no secret may stand in its argv.
    over = json.dumps({"status": "over_limit", "commit": None, "branches": [], "at": 1.0,
                       "reason": "main's bundle holds more than 4 bytes %s%s"})
    real = agent_build.commit_command

    def command(**kw):
        if "no_submission" in kw:
            return real(**kw)
        return ("/bin/sh", "-c", 'printf "$2" "$3" "$4" > "$1/commit.json"; exit 4', "snapshot", agent_build.RECORD,
                over, KEY[:8], KEY[8:])

    monkeypatch.setattr(agent_build, "commit_command", command)
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    outputs = AgentBuildStage().run(ctx)
    events = _events(ctx)
    snaps = {e["n"]: e for e in events if e["kind"] == "build.snapshot"}
    reported = [e for e in events if e["kind"] == "build.snapshot_over_limit"]
    assert snaps and all(e["status"] == "over_limit" and e["exit_code"] == 4 for e in snaps.values())
    assert sorted(e["n"] for e in reported) == sorted(snaps)
    assert all(e["reason"].startswith("main's bundle holds more than 4 bytes") and KEY not in e["reason"]
               for e in reported)
    assert "source_bundle" in outputs and _record(ctx, outputs).commit.status == "ok"


def test_a_snapshot_stopped_mid_bundle_leaves_no_partial_bundle(sandbox_bins, tmp_path, monkeypatch):
    """A snapshot still bundling when the agent's part ends is stopped, and its program, killed,
    never removes the partial bundle it was writing: compressed history no scan reads. The stage
    removes every ``.tmp-*`` in the snapshots once it has stopped one."""
    control = sandbox_bins
    real = subprocess.run(["sh", "-c", "command -v git"], capture_output=True, text=True, check=True).stdout.strip()
    # A git whose bundle never ends for a snapshot, writing slowly; the commit point's, whose
    # HOME lies in its record directory, is the real one.
    slow = tmp_path / "slow-git"
    slow.mkdir()
    (slow / "git").write_text('#!/bin/sh\ncase " $* " in *" bundle create - "*) case "$HOME" in '
                              f'*/{agent_build.COMMIT_RECORDS}/*) ;; '
                              '*) while head -c 4096 /dev/zero; do sleep 0.05; done; exit 1;; esac;; esac\n'
                              f'exec {real} "$@"\n')
    (slow / "git").chmod(0o755)
    monkeypatch.setenv("FAKE_SANDBOX_PATH", f"{slow}:{os.environ['FAKE_SANDBOX_PATH']}")
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 4.5}])
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    outputs = AgentBuildStage().run(ctx)
    stopped = [e for e in _events(ctx) if e["kind"] == "build.snapshot" and e["ending"] == "planned_stop"]
    assert stopped, "no snapshot was stopped while it bundled"
    assert sorted(p.name for p in (ctx.attempt_dir / "snapshots").iterdir() if p.name.startswith(".tmp-")) == []
    assert _record(ctx, outputs).commit.status == "ok"


def test_every_role_of_a_build_asks_for_the_time_left_to_the_allocations_end_and_records_its_waits(
        sandbox_bins, tmp_path):
    """With an allocation's end to keep to (``StageContext.deadline_cap``), each role the build
    starts, the sandbox, its snapshots, the commit point and the transcript's copy among them,
    asks Slurm for the time left to that end and never the stage's own time, and its start
    records how long its step took to start and to report, and the deadline it then had."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 2.5}])
    launcher = LocalLauncher(gpus=1)
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1), launcher=launcher)
    ctx.deadline_cap = time.time() + 100_000
    assert "source_bundle" in AgentBuildStage().run(ctx)
    names = [spec.name for spec in launcher.launched]
    assert {"sandbox", "commit", "transcript"} <= set(names) and any(n.startswith("snapshot-") for n in names)
    assert all(99_000 < spec.shape.time_sec <= 100_000 for spec in launcher.launched)
    assert ctx.stage.shape.time_sec < 99_000
    started = [e for e in _events(ctx) if e["kind"] == "role_started"]
    assert sorted(e["role"] for e in started) == sorted(names)
    assert all(e["launch_wait_sec"] >= 0 and e["report_wait_sec"] >= 0 and e["deadline"] >= ctx.deadline - 1e6
               for e in started)


def test_a_failed_snapshot_is_an_event_and_the_build_still_commits(sandbox_bins, tmp_path,
                                                                   monkeypatch):
    """A snapshot that fails is the context's fault of a role, which would stop every role the
    context waits for; the stage waits for its later roles alone, so the commit point runs."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 2.5}])
    real = agent_build.commit_command
    monkeypatch.setattr(agent_build, "commit_command",
                        lambda **kw: real(**kw) if "no_submission" in kw else ("/bin/false",))
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    outputs = AgentBuildStage().run(ctx)
    snaps = [e for e in _events(ctx) if e["kind"] == "build.snapshot"]
    assert snaps and snaps[0]["ending"] == "exited" and snaps[0]["exit_code"] == 1
    assert "source_bundle" in outputs and _ended(ctx, "commit")["ending"] == "exited"


@pytest.mark.parametrize("error", [OSError(28, "No space left on device"),
                                   StageFault(infra_fault("the stage is stopping", retryable=True))],
                         ids=["disk-full", "refused-start"])
def test_a_snapshot_that_cannot_start_is_skipped_with_a_warning_and_the_drive_goes_on(
        sandbox_bins, tmp_path, monkeypatch, caplog, error):
    """Making a snapshot's directories or starting its container can fail, on a node the agent
    filled among others: the snapshot is skipped, an event and a warning say so, the next one is
    tried when due, and the build is graded."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 3.5, "advance": 900}])
    tried: list[str] = []

    def refuse_snapshots(self, name, spec):
        tried.append(name)
        raise error

    # The launch runs off the drive loop; the container start is its only off-loop step.
    monkeypatch.setattr(agent_build._Build, "_snapshot_container", refuse_snapshots)
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    with caplog.at_level("WARNING", logger="hls.stages.agent_build"):
        outputs = AgentBuildStage().run(ctx)
    assert _record(ctx, outputs).ending == "agent_ended" and "source_bundle" in outputs
    assert len(tried) >= 2
    skipped = [e for e in _events(ctx) if e["kind"] == "build.snapshot_skipped"]
    assert len(skipped) == len(tried) and type(error).__name__ in skipped[0]["error"]
    assert any("snapshot" in r.getMessage() and "skipped" in r.getMessage() for r in caplog.records)


def test_a_snapshot_launch_in_flight_at_the_hard_limit_is_waited_for_only_once_the_sandbox_is_stopped(
        sandbox_bins, tmp_path, monkeypatch):
    """A snapshot's step is started off the drive loop. When the hard limit passes while the
    scheduler still grants it, the stage stops the sandbox first and only then waits for the
    launch: the agent's processes end with its part, so a commit the CLI makes after the limit
    never reaches the bundle; the snapshot, once started, is stopped and reaped."""
    control = sandbox_bins
    _script(control, window=1_000_000,
            claude=[{"commit": "v1\n", "late": {"after": 6, "commit": "v2\n"}, "hang": True}])
    real = agent_build._Build._snapshot_container
    offset = control / "offset"
    seen: dict[str, bool] = {}

    def granted_once_the_sandbox_is_stopped(self, name, spec):
        # The hard limit passes while this step is granted; the grant comes once the sandbox is
        # stopped, or after 30 s, past the CLI's late commit.
        offset.write_text(str((float(offset.read_text()) if offset.exists() else 0.0) + 1200))
        give_up = time.monotonic() + 30
        while agent_build.SANDBOX in self.running and time.monotonic() < give_up:
            time.sleep(0.05)
        seen.setdefault("sandbox_stopped_first", agent_build.SANDBOX not in self.running)
        return real(self, name, spec)

    monkeypatch.setattr(agent_build._Build, "_snapshot_container", granted_once_the_sandbox_is_stopped)
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    outputs = AgentBuildStage().run(ctx)
    assert seen == {"sandbox_stopped_first": True}
    assert _record(ctx, outputs).ending == "hard_limit"
    assert _ended(ctx, "sandbox")["ending"] == "planned_stop"
    # Stopped as planned once started, maybe before its runtime marked the command's start: only
    # an event either way, and reported.
    assert [e["n"] for e in _events(ctx) if e["kind"] == "build.snapshot"] == [1]
    _ended(ctx, "snapshot-1")
    assert _gone(ctx)
    # The CLI was stopped before its late commit: main, and the bundle, hold v1 alone.
    assert (ctx.role_dir("sandbox") / "workspace" / "server.py").read_text() == "v1\n"
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)
    assert ctx.running_roles() == []


def test_a_snapshot_launch_in_flight_when_the_sandbox_ends_is_collected_before_every_role_stops(
        sandbox_bins, tmp_path, monkeypatch):
    """The sandbox can end while a snapshot's step is still granted: the stage reports the sandbox
    at once, from its own thread while the launch goes on in its worker, then takes the launch's
    handle before it stops every role, so no snapshot outlives the build."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "hang": True}])
    real = agent_build._Build._snapshot_container
    seen: dict[str, bool] = {}

    def granted_after_the_sandbox_is_reported(self, name, spec):
        self.running[agent_build.SANDBOX].kill()
        give_up = time.monotonic() + 30
        while self.sandbox_outcome is None and time.monotonic() < give_up:
            time.sleep(0.05)
        seen.setdefault("sandbox_reported_first", self.sandbox_outcome is not None)
        return real(self, name, spec)

    monkeypatch.setattr(agent_build._Build, "_snapshot_container", granted_after_the_sandbox_is_reported)
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert seen == {"sandbox_reported_first": True}
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    # The snapshot started after the sandbox ended, and the stage stopped and reported it.
    assert "role_ended" in [ev["kind"] for ev in _events(ctx) if ev.get("role") == "snapshot-1"]
    assert ctx.running_roles() == []
    assert _gone(ctx)


def test_a_try_that_wrote_a_no_submission_record_and_then_exited_74_is_tried_again(
        sandbox_bins, tmp_path, monkeypatch):
    """Only the exits for no commit on main (5) and an unreadable repository (4) make a
    no-submission record a result. The first try finds no commit on main and writes and reports
    its no-submission record, and then its commit record fails to write (74): it is tried again,
    never read as nothing gradable from what it left."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"advance": 900}])
    real = agent_build.commit_command
    tries: list[dict] = []

    def first_fails(**kw):
        if "no_submission" in kw:
            tries.append(kw)
            if len(tries) == 1:
                # A record the machine cannot write: under /proc no directory can be made.
                kw = {**kw, "record": "/proc/hls-no-record/commit.json"}
        return real(**kw)

    monkeypatch.setattr(agent_build, "commit_command", first_fails)
    monkeypatch.setattr(agent_build, "COMMIT_PAUSE_SEC", 0.0)
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    assert "no_submission" in outputs
    assert [(t.exit_code, t.wrote) for t in _record(ctx, outputs).commit_tries] == [
        (74, None), (agent_build.NO_MAIN_EXIT, "no_submission")]


@pytest.mark.parametrize("what", ["transcript-role", "record"])
def test_a_failure_after_the_commit_point_is_infra_never_retried_and_keeps_the_bundle(
        sandbox_bins, tmp_path, monkeypatch, what):
    """Once the commit point wrote the bundle, a retry would run the whole session
    again, so a failure after it, the transcript role's or the runner's own, is not retried."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    launcher = LocalLauncher(gpus=1)
    if what == "transcript-role":
        failing = tmp_path / "failing_transcript.py"
        failing.write_text("import sys\nsys.exit(3)\n")
        launcher = LocalLauncher(gpus=1, replace_modules={"hls.agent.transcript": failing})
    else:
        def unwritable(**fields):
            raise ValueError("the record cannot be written")

        monkeypatch.setattr(agent_build, "BuildRecord", unwritable)
    ctx = _ctx(tmp_path, _params(), launcher=launcher)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert "after its commit point" in fault.reason and fault.evidence["commit_point"] == "bundle"
    if what == "transcript-role":
        assert fault.evidence["roles"] == "transcript"
    else:
        assert "ValueError: the record cannot be written" in fault.reason
    assert _bundle_main(ctx.output_path("source.bundle"), tmp_path) == _workspace_main(ctx)
    # The runner's SIGTERM from the commit point on writes this, never a retryable termination.
    declared = ctx.no_retry_fault()
    assert declared is not None and (declared.fault_class, declared.retryable) == ("infra", False)
    assert "after its commit point" in declared.reason and declared.evidence["commit_point"] == "bundle"


def test_before_the_commit_point_an_end_may_still_be_retried(sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "hang": True}])
    ctx = _ctx(tmp_path, _params())
    threading.Timer(2.0, ctx.stop_roles).start()
    with pytest.raises(StageFault):
        AgentBuildStage().run(ctx)
    assert ctx.no_retry_fault() is None


def test_a_shutting_down_allocation_stops_the_agent_as_infra(sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "hang": True}])
    ctx = _ctx(tmp_path, _params())
    threading.Timer(2.0, ctx.stop_roles).start()
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert e.value.fault.retryable and "shutting down" in e.value.fault.reason
    ctx.close()
    assert _gone(ctx)


def test_an_earlier_stages_files_in_the_sandbox_directory_are_gone_before_the_build(sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    stale = ctx.scratch / "roles" / "sandbox" / "workspace"
    stale.mkdir(parents=True)
    (stale / "stale.txt").write_text("from an earlier stage")
    AgentBuildStage().run(ctx)
    assert not (stale / "stale.txt").exists()


def test_a_scratch_too_deep_for_the_sandboxs_sockets_is_refused_before_any_role(sandbox_bins, tmp_path):
    _script(sandbox_bins, window=1_000_000, claude=[{}])
    ctx = _ctx(tmp_path, _params())
    ctx.scratch = ctx.scratch / ("d" * 80)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    socket = workspace.SandboxLayout(ctx.role_dir("sandbox")).socket
    assert e.value.fault.retryable is False and "too deep" in e.value.fault.reason
    # Each path with its length, the limit, and the setting to change.
    assert f"{socket} ({len(os.fsencode(socket))} bytes)" in e.value.fault.reason
    assert e.value.fault.reason.endswith("over the 107 bytes a socket's path may hold; set site.node_scratch to a "
                                         "shorter path")
    assert ctx.launcher.launched == []


# ---------------------------------------------------------------------------
# The brief, the plan, the seal and the budgets
# ---------------------------------------------------------------------------


def test_the_stage_and_every_role_program_import_without_harbor():
    """The role programs and the runner import ``hls.stages.registry``, which loads this stage,
    inside images that have no harbor; only running a build may import it."""
    modules = ["hls.stages", "hls.stages.registry", "hls.stages.agent_build", "hls.verify.role", "hls.drive.role",
               "hls.score.role", "hls.agent.transcript", "hls.agent.workspace", "hls.agent.health",
               "hls.agent.extract"]
    code = ("import importlib, sys\nsys.modules['harbor'] = None\n"
            f"for name in {modules!r}:\n    importlib.import_module(name)\n"
            "assert not [m for m in sys.modules if m.startswith('harbor.')], 'harbor was imported'\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ))
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_briefs_figures_are_the_sandbox_roles_own(tmp_path):
    """The brief states the sandbox role's declared memory, never the task's, the
    stage's or the node's, with its CPUs, its disk cap, the paths it may write, and that a
    graded server's /workspace is read-only."""
    ctx = _ctx(tmp_path, _params())
    build = agent_build._Build(ctx)
    build.ceiling = ctx.process_ceiling()          # read before the brief, as setup does
    task = load_task(TaskRef(module="tests.agent.build_task"))
    assert task.hardware.memory_gb == 32 and STAGE.memory_gib == 22
    text = build.brief(task, build.artifact())
    assert "32 GB host RAM" not in text
    # The ceiling the brief states is the one the spawner gives every command.
    assert (f"Your commands can start a process or a thread only while the node holds fewer than "
            f"{build.ceiling.ceiling:,} of them for the user they run as") in text
    assert ctx.plan.settings.site.nproc_max is None and build.ceiling.ceiling <= NPROC_MAX
    # The hardware block states the same memory in decimal GB, never overstated: 16 GiB is 17.18 GB.
    assert "17 GB host RAM" in text and "16 GB host RAM" not in text and "22 GiB" not in text
    assert ("Your sandbox has 16 GiB of memory, of which the harness's own processes may hold up to 4 GiB, and 2 "
            "CPUs. Its `/dev/shm` holds up to 16 GiB, and what is there counts in that memory.") in text
    assert ("You can write to `/workspace`, `/tmp`, `/var/tmp` and `$HOME`: everything there and your "
            "session's logs share 8 GiB of writable space, separate from your memory, and hold at most "
            f"{memory_watch.DISK_WALK_MAX_ENTRIES:,} files, directories and links, with no directory nested more "
            f"than {memory_watch.DISK_WALK_MAX_DEPTH} levels deep.") in text
    # What a memory overrun does: a process ends first, the sandbox only
    # when that does not bring the memory back.
    assert ("When your sandbox's processes pass its memory, one of them is ended (the largest of your commands' "
            "processes, or the one the kernel's out-of-memory killer picks) and the build goes on; when ending "
            "processes does not bring the memory back under it, the sandbox ends.") in text
    assert ("A sandbox that ends so, or that writes past its writable space, ends the build, which is graded on "
            "what you last committed on main; past the files or the nesting, it has run out of its writable "
            "space, and so it has when one of its processes holds open files the harness cannot list, as a "
            "process running a program from a file it cannot read does. Your commands run with `no_new_privs` "
            "set, and `prctl(PR_SET_DUMPABLE, 0)` fails with EPERM.") in text
    assert "runs out of its memory or its writable space ends there" not in text
    assert ("`/workspace` is read-only there: it may write only to `/tmp`, `/var/tmp`, `$HOME` and "
            "`/engine-cache`.") in text
    provenance = json.loads(ctx.store.read_bytes(ctx.output_key("brief_provenance.json")))
    [budget] = [s for s in provenance["sections"] if s["block"] == "budget"]
    assert "16 GiB of memory" in budget["rendered"] and "get_budget" in budget["rendered"]


def test_the_briefs_dev_shm_cap_is_the_one_the_spawner_mounts_the_sandboxs_whole_memory(sandbox_bins, tmp_path):
    """/dev/shm is capped at the sandbox's memory budget (``AgentBuildParams.shm_gib``),
    since its pages count against that memory anyway; the brief states the figure the spawner
    mounts, from the same field, and no plan can set another."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    p = AgentBuildParams.model_validate(ctx.stage.params)
    assert p.shm_gib == p.sandbox_shape.memory_gib == 16
    command = fake_sandbox.records(Path(os.environ["FAKE_SANDBOX_RECORDS"]))["sandbox"]["command"]
    assert command[command.index("--shm-bytes") + 1] == str(p.shm_gib * 2**30)
    brief = ctx.store.read_bytes(outputs["brief"]).decode()
    assert f"Its `/dev/shm` holds up to {p.shm_gib} GiB" in brief
    with pytest.raises(ValidationError, match="shm_gib"):
        AgentBuildParams.model_validate(_params(shm_gib=8.0))


def test_the_brief_states_the_repository_cap_the_commit_point_and_snapshots_hold_it_to(sandbox_bins, tmp_path):
    """One limit bounds a submission's repository: the sandbox's writable disk cap
    (``sandbox_shape.disk_gib``). The brief states it with the writable space, and every commit
    point and snapshot holds the repository's objects and main's bundle to exactly it."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 2.5}])
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1))
    outputs = AgentBuildStage().run(ctx)
    p = AgentBuildParams.model_validate(ctx.stage.params)
    assert p.sandbox_shape.disk_gib == 8
    brief = " ".join(ctx.store.read_bytes(outputs["brief"]).decode().split())
    assert ("The objects your repository's `.git` holds, its history and unreachable objects among them, may "
            "come to at most those 8 GiB together uncompressed, and the bundle of main the harness makes of them "
            "to at most 8 GiB; past either, the build has no submission and scores zero.") in brief
    # The bound on what rebuilding its deltas reads, from the commit program's own figure.
    assert commit_program.CHAIN_BYTES_MAX == 64 * 2**30
    assert ("Every object git stores as a delta counts with what rebuilding it reads: itself, its delta, and "
            "the same for the object it is a delta on, down its chain. Those may come to at most 64 GiB "
            "together; past that, the build has no submission and scores zero.") in brief
    records = fake_sandbox.records(Path(os.environ["FAKE_SANDBOX_RECORDS"]))
    for role in ("commit", "snapshot-1"):
        command = records[role]["command"]
        assert command[command.index("--max-bytes") + 1] == str(8 * 2**30), role


def test_a_builds_collection_allowance_does_not_grow_with_its_cap(tmp_path):
    """The collection allowance is the run's, whatever the build's writable disk cap: a build of a
    241 GiB cap is planned with the least allowance the collection's fixed steps and the commit
    point's least time need, as one of 8 GiB is, and one under it is refused, naming the setting."""
    sandbox = SANDBOX.model_copy(update={"disk_gib": 241})
    roles = concurrent([sandbox, SNAPSHOT])
    shape = roles.model_copy(update={"cpus": sandbox.cpus, "memory_gib": roles.memory_gib + RUNNER_GIB,
                                     "disk_gib": 241 + KEPT_DISK_GIB + DATA_GIB, "time_sec": STAGE_TIME})
    _plan(tmp_path, _params(sandbox_shape=sandbox.model_dump()), shape=shape)
    with pytest.raises(PlanError, match=r"the collection allowance \(run.collect_allowance_sec\) is 2349s"):
        _plan(tmp_path, _params(sandbox_shape=sandbox.model_dump(), collect_allowance_sec=COLLECT_SEC - 1),
              shape=shape)


def test_the_brief_states_the_graded_servers_figures(tmp_path, monkeypatch):
    """The brief's serving section states what a graded server gets, the
    shape its grading runs it in (``served_shape``, the task's GPUs and the evaluation's allocation,
    its job's memory less the runner's reserve), for which the agent builds; the
    provenance records it with the budget section."""
    from hls.artifact.artifact import EvaluationAllocation

    task = load_task(TaskRef(module="tests.agent.build_task"))
    allocation = EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180, driver_memory_gib=16)
    served = artifact_shape(task, allocation, runner_memory_gib=RUNNER_GIB)
    ctx = _ctx(tmp_path, _params(served_shape=served.model_dump()))
    build = agent_build._Build(ctx)
    build.ceiling = ctx.process_ceiling()          # read before the brief, as setup does
    text = build.brief(task, build.artifact())
    gpus = task.hardware.accelerator
    serving = (f"When your server is graded, it runs with {gpus.count} {gpus.kind} GPU, 8 CPUs, 176 GiB of memory "
               "and 180 GiB of writable space, which holds at most")
    assert serving in text
    provenance = json.loads(ctx.store.read_bytes(ctx.output_key("brief_provenance.json")))
    [budget] = [s for s in provenance["sections"] if s["block"] == "budget"]
    assert serving in budget["rendered"]


def test_the_brief_states_the_drive_bounds_the_task_declares(tmp_path, monkeypatch):
    """The serving brief states the cap on a graded drive's trace, the task's own
    (``DriveBounds.trace_cap_bytes``), and the cap on one reply (``REPLY_MAX_BYTES``), each read
    off its owner when the brief renders; a task that declares no bounds, which cannot be graded,
    gets no such sentence."""
    from hls.drive import bounds as bounds_mod
    from hls.drive.bounds import DriveBounds, VerifyPeak
    from hls.prompt import drive_bounds_brief_section

    task = load_task(TaskRef(module="tests.agent.build_task"))
    task = task.model_copy(update={"drive_bounds": DriveBounds(
        trace_cap_bytes=3 * 2**30, trace_request_bytes=1, verify_peak=VerifyPeak(base_bytes=0, bytes_per_byte=1.0),
        verify_limited=False)})
    ctx = _ctx(tmp_path, _params())
    build = briefing(ctx)
    text = build.brief(task, build.artifact())
    stated = drive_bounds_brief_section(trace_cap_bytes=3 * 2**30, reply_max_bytes=bounds_mod.REPLY_MAX_BYTES,
                                        verify_limited=False)
    assert stated in text and "at most 3 GiB of JSON" in stated and "at most 256 MiB" in stated
    monkeypatch.setattr(bounds_mod, "REPLY_MAX_BYTES", 7 * 2**20)
    again = briefing(_ctx(tmp_path / "again", _params()))
    assert "One reply may carry at most 7 MiB" in again.brief(task, again.artifact())
    bare = briefing(_ctx(tmp_path / "bare", _params()))
    unbounded = task.model_copy(update={"drive_bounds": None})
    assert "drive's trace" not in bare.brief(unbounded, bare.artifact())
    # A cap the verification sets is said to be so, read off the task's bounds.
    limited = task.model_copy(update={"drive_bounds": task.drive_bounds.model_copy(update={"verify_limited": True})})
    told = briefing(_ctx(tmp_path / "limited", _params()))
    assert "what the harness's verification of the trace can hold" in told.brief(limited, told.artifact())
    assert "what the harness's verification of the trace can hold" not in text


def test_a_cpu_only_graded_servers_limits_count_the_servers_of_its_cpus_a_node_holds(tmp_path):
    """A graded server with no GPU shares a node with as many others as the node's CPUs hold of
    its own, its served shape's (``StageContext.served_ucounts``), never the build stage's: on a
    192-CPU node an 8-CPU server is one of 24, so it gets 128 // 24 = 5 inotify instances."""
    served = ResourceShape(cpus=8, memory_gib=176, disk_gib=180, time_sec=600)
    ctx = _ctx(tmp_path, _params(served_shape=served.model_dump()), launcher=_node(0, 192),
               site={"nproc_max": 96 * FLOOR_PER_EVALUATION})
    build = agent_build._Build(ctx)
    build.ceiling = ctx.process_ceiling()
    text = " ".join(build.brief(load_task(TaskRef(module="tests.agent.build_task")), build.artifact()).split())
    assert "GPUs, 8 CPUs, 176 GiB of memory and 180 GiB of writable space" in text
    assert (f"It and every process it starts may hold at most 5 inotify instances, {1_048_576 // 24:,} inotify "
            "watches") in text
    assert ctx.served_ucounts(served).figures()["max_inotify_instances"] == 5


def test_a_server_with_no_evaluation_allocation_is_stated_on_the_tasks_hardware_in_whole_gib(tmp_path):
    """Without an allocation, a graded server's figures are the task's hardware's
    (``artifact_shape``), its decimal GB stated in whole GiB, rounded down."""
    task = load_task(TaskRef(module="tests.agent.build_task"))
    served = artifact_shape(task, runner_memory_gib=RUNNER_GIB)
    ctx = _ctx(tmp_path, _params(served_shape=served.model_dump()))
    build = briefing(ctx)
    hardware = task.hardware
    assert (f"{hardware.cpu_count} CPUs, {hardware.memory_gib} GiB of memory and {hardware.storage_gib} GiB of "
            "writable space") in build.brief(task, build.artifact())
    assert hardware.memory_gib < hardware.memory_gb


def test_the_brief_states_the_sandboxs_hardware_in_the_hardware_blocks_decimal_gb():
    """The hardware block states GB where the sandbox's shape is in GiB: memory and storage both,
    rounded down so neither overstates the other."""
    from hls.core.model import gb_of_gib
    from tests.agent import build_task

    hardware = agent_build.brief_hardware(build_task.TASK.hardware, SANDBOX)
    assert (hardware.cpu_count, hardware.memory_gb, hardware.storage_gb) == (
        SANDBOX.cpus, gb_of_gib(SANDBOX.memory_gib), gb_of_gib(SANDBOX.disk_gib))
    assert hardware.accelerator == build_task.TASK.hardware.accelerator


def briefing(ctx: StageContext) -> agent_build._Build:
    """A build of *ctx* ready to render its brief: the commands' process ceiling read first, as the
    stage's setup reads it, since the brief states it."""
    build = agent_build._Build(ctx)
    build.ceiling = ctx.process_ceiling()
    return build


def test_the_brief_names_the_checkpoint_the_agent_serves(tmp_path):
    """The brief tells the agent which model it serves: its goal and its model section both name
    the task's base checkpoint."""
    ctx = _ctx(tmp_path, _params())
    build = briefing(ctx)
    task = load_task(TaskRef(module="tests.agent.build_task"))
    checkpoint = task.model.base_checkpoint
    text = build.brief(task, build.artifact())
    assert f"Serve `{checkpoint}` under the workload below." in text
    assert f"## Model\n- Checkpoint: `{checkpoint}`\n" in text


def test_the_brief_tells_the_agent_to_submit_a_repository_whose_main_holds_its_server(tmp_path, monkeypatch):
    """A git repository producer's brief carries its submission contract under its own heading:
    only the latest commit on main is graded, and the repository holds a pyproject.toml and the
    producer's server file, which the harness installs and launches. An agent not told so does
    not commit, and its build ends with no submission."""
    from hls.spec.submission import SubmissionContract
    from tests.agent import build_task

    producer = build_task.ARTIFACT.producer.model_copy(update={"server_rel_path": "app/serve.py"})
    monkeypatch.setattr(build_task, "ARTIFACT", build_task.ARTIFACT.model_copy(update={"producer": producer}))
    ctx = _ctx(tmp_path, _params())
    build = briefing(ctx)
    text = build.brief(load_task(TaskRef(module="tests.agent.build_task")), build.artifact())
    section = SubmissionContract(server_filename="app/serve.py").brief_section()
    assert f"## Submission\n{section}\n" in text
    assert "Only the latest commit on `main` is evaluated" in section
    assert "- `app/serve.py`, launched as `python app/serve.py`." in section


def test_a_build_with_no_soft_budget_is_promised_no_budget_and_no_budget_tool(tmp_path):
    """A build with a soft budget runs the budget server, and its brief states that budget and
    the tool that reads what is left of it. One without runs no budget server, so its brief
    states no budget for the build and never names the tool; the sandbox's and the graded
    server's figures are stated either way."""
    task = load_task(TaskRef(module="tests.agent.build_task"))
    (tmp_path / "soft").mkdir()
    (tmp_path / "none").mkdir()
    ctx = _ctx(tmp_path / "soft", _params(soft_sec=1000))
    build = briefing(ctx)
    assert budget_brief_section(1000) in build.brief(task, build.artifact())
    ctx = _ctx(tmp_path / "none", _params(soft_sec=None))
    build = briefing(ctx)
    text = build.brief(task, build.artifact())
    assert "for this build" not in text and "get_budget" not in text
    provenance = json.loads(ctx.store.read_bytes(ctx.output_key("brief_provenance.json")))
    [budget] = [s for s in provenance["sections"] if s["block"] == "budget"]
    assert budget["rendered"].startswith("Your sandbox has 16 GiB of memory")
    assert "When your server is graded, it runs with" in budget["rendered"]


def _uploading(folder: Path, digest: str):
    from tests.agent import build_task

    return build_task.ARTIFACT.producer.model_copy(update={"upload_dir": (str(folder),), "upload_sha256": digest})


def _references(checkout: Path) -> Path:
    folder = checkout / "tasks" / "assets" / "references"
    (folder / "cuda").mkdir(parents=True)
    (folder / "README.md").write_text("index\n")
    (folder / "cuda" / "guide.html").write_text("<p>guide</p>\n")
    return folder


def test_an_upload_folder_whose_tree_is_the_one_the_task_commits_is_uploaded(tmp_path):
    folder = _references(tmp_path / "checkout")
    digest = tree_digest(folder)
    base = tmp_path / "checkout" / "tasks" / "mslo"
    relative = _uploading(folder, digest).model_copy(update={"upload_dir": ("../assets/references",)})
    assert agent_build.upload_folder(relative, base, tmp_path / "checkout") == ([folder], digest)
    assert agent_build.upload_folder(_uploading(folder, digest), base, tmp_path / "checkout") == ([folder], digest)


@pytest.mark.parametrize("case, words", [
    ("absent", "is not a directory"),
    ("changed", "the task commits"),
    ("extra", "the task commits"),
    ("link", "is a link"),
    ("outside", "lies outside the checkout"),
])
def test_an_upload_folder_that_is_not_the_one_the_task_commits_is_refused_as_infra_not_retried(
        tmp_path, case, words):
    """A contestant is handed exactly the material the task names: a folder that is gone, differs
    from the digest the task commits, holds a link, or lies outside the checkout is refused before
    anything is copied, as the harness's fault."""
    checkout = tmp_path / "checkout"
    folder = _references(checkout)
    producer = _uploading(folder, tree_digest(folder))
    if case == "absent":
        remove_tree(folder)
    elif case == "changed":
        (folder / "cuda" / "guide.html").write_text("<p>guidE</p>\n")
    elif case == "extra":
        (folder / "cuda" / "notes.txt").write_text("")
    elif case == "link":
        os.symlink(folder / "README.md", folder / "cuda" / "README.md")
    else:
        checkout = tmp_path / "another-checkout"
        checkout.mkdir()
    with pytest.raises(StageFault) as e:
        agent_build.upload_folder(producer, checkout / "tasks" / "mslo", checkout)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert words in e.value.fault.reason


def test_a_task_folder_outside_the_checkout_is_refused_through_dot_dot_too(tmp_path):
    checkout = tmp_path / "checkout"
    _references(tmp_path / "elsewhere")
    producer = _uploading(Path("../../../elsewhere/tasks/assets/references"), "0" * 64)
    with pytest.raises(StageFault) as e:
        agent_build.upload_folder(producer, checkout / "tasks" / "mslo", checkout)
    assert "lies outside the checkout" in e.value.fault.reason


def test_the_stage_is_granted_exactly_the_arms_route(tmp_path):
    plan = _plan(tmp_path, _params())
    assert plan.stage("agent_build").grants == {"ANTHROPIC_API_KEY": "anthropic_key"}
    gpt = _plan(tmp_path, _params(GPT))
    assert gpt.stage("agent_build").grants == {"OPENAI_API_KEY": "openai_key", "OPENAI_BASE_URL": "openai_url"}
    k3 = _plan(tmp_path, _params(K3))
    assert k3.stage("agent_build").grants == {"ANTHROPIC_AUTH_TOKEN": "kimi_token", "ANTHROPIC_BASE_URL": "kimi_url"}
    missing = {k: v for k, v in SECRETS.items() if k != "anthropic_key"}
    with pytest.raises(PlanError, match="anthropic_key"):
        _plan(tmp_path, _params(), secrets=missing)
    repointed = OPUS.model_copy(update={"route_table": Route(credential="openai_key")})
    with pytest.raises(PlanError, match="resolve the arm again"):
        _plan(tmp_path, _params(repointed))


@pytest.mark.parametrize("change, words", [
    ({"shape": ROLES.model_copy(update={"time_sec": 1300})}, "more memory for the runner"),
    ({"params": {"collect_allowance_sec": COLLECT_SEC - 1}}, "collection allowance"),
    ({"shape": STAGE.model_copy(update={"disk_gib": ROLES.disk_gib})}, "disk for the agent cap's tree"),
    ({"shape": STAGE.model_copy(update={"disk_gib": ROLES.disk_gib + KEPT_DISK_GIB})}, "copies of its data"),
    ({"shape": STAGE.model_copy(update={"time_sec": STAGE_TIME - 1})}, f"floor of {STAGE_TIME}s"),
    # The floor holds the staging the plan gives, a longer one too.
    ({"params": {"staging_sec": STAGING_SEC + 1}}, f"floor of {STAGE_TIME + 1}s: the setup allowance, the "
                                                  "staging"),
    ({"params": {"harness_reserve_gib": 16.0}}, "harness reserve"),
    ({"params": {"sandbox_shape": SANDBOX.model_copy(update={"disk_gib": 0}).model_dump()},
      "shape": STAGE.model_copy(update={"disk_gib": KEPT_DISK_GIB + DATA_GIB})}, "the sandbox must declare the disk"),
    ({"params": {"snapshot_shape": SNAPSHOT.model_copy(update={"disk_gib": 1}).model_dump()},
      "shape": STAGE.model_copy(update={"disk_gib": STAGE.disk_gib + 1})}, "no GPUs and no disk"),
    ({"params": {"commit_shape": COMMIT.model_copy(update={"disk_gib": 1}).model_dump()}},
     r"no GPUs and no disk \(commit_shape\)"),
    ({"params": {"commit_shape": COMMIT.model_copy(update={"cpus": 3}).model_dump()}},
     r"commit_shape: 3 CPUs\) run on CPUs the sandbox holds"),
    ({"params": {"extract_shape": EXTRACT.model_copy(update={"disk_gib": 1}).model_dump()}},
     r"no GPUs and no disk \(extract_shape\)"),
    ({"params": {"extract_shape": EXTRACT.model_copy(update={"memory_gib": STAGE.memory_gib}).model_dump()}},
     "the extraction role's"),
    ({"params": {"arm": GPT.model_copy(update={"effort": None})}}, "must declare one"),
    ({"params": {"arm": OPUS.model_copy(update={"harness": "terminus-2"})}},
     r"arm\.harness\n  Input should be 'claude-code' or 'codex'"),
    ({"shape": STAGE.model_copy(update={"cpus": ROLES.cpus})}, "the stage's resources must be the sandbox's"),
    ({"params": {"snapshot_shape": SNAPSHOT.model_copy(update={"cpus": 3}).model_dump()}},
     "run on CPUs the sandbox holds"),
    ({"params": {"network": {"domains": {"mode": "allow", "entries": []}}}}, "no package-deny rule"),
], ids=["runner-memory", "collect-allowance", "tree-disk", "data-disk", "time", "staging",
        "reserve", "sandbox-disk", "harness-disk", "commit-disk", "commit-cpus", "extract-disk",
        "extract-memory", "codex-effort", "unknown-harness", "summed-cpus", "harness-cpus", "no-deny-list"])
def test_check_refuses_what_the_stage_cannot_run(tmp_path, change, words):
    params = _params(**change.get("params", {}))
    with pytest.raises(PlanError, match=words):
        _plan(tmp_path, params, shape=change.get("shape", STAGE))


def test_the_collections_fixed_steps_are_fixed_bounds_and_the_commit_point_gets_the_rest():
    """Every step of a full build's collection but the commit point has a fixed bound, none sized
    from a rate: a lost channel's wait; the sandbox's and the snapshot's stops, each its SIGTERM
    grace, the wait after SIGKILL and the wait for its logs; the sandbox's in-place log scrub; the
    wait for the sandbox gone; and, once the commit point has written, the transcript copy with its
    logs and the extraction role's ``EXTRACT_SEC`` with its stop. The commit point gets what the
    run's allowance leaves, at least ``COMMIT_MIN_SEC``, so the least allowance a plan accepts is
    2,350 s, and the default hour leaves the commit point 1,850 s."""
    stopped = 2 * STOP_GRACE_SEC + LOG_CLOSE_SEC
    assert agent_build.STOPPED_ROLE_SEC == stopped == 50
    assert (agent_build.LOST_WAIT_SEC, agent_build.SANDBOX_SCRUB_SEC, agent_build.GONE_WAIT_SEC) == (10, 300, 60)
    assert (agent_build.TRANSCRIPT_COPY_SEC, agent_build.EXTRACT_SEC, agent_build.COMMIT_MIN_SEC) == (600, 600, 600)
    assert agent_build.AFTER_COMMIT_SEC == math.ceil(agent_build.TRANSCRIPT_COPY_SEC + LOG_CLOSE_SEC
                                                     + agent_build.EXTRACT_SEC + stopped) == 1280
    assert agent_build.STOP_BUILD_COLLECT_SEC == math.ceil(agent_build.LOST_WAIT_SEC + stopped
                                                           + agent_build.SANDBOX_SCRUB_SEC) == 360
    assert agent_build.COLLECT_STEPS_SEC == math.ceil(agent_build.STOP_BUILD_COLLECT_SEC + stopped
                                                      + agent_build.GONE_WAIT_SEC + agent_build.AFTER_COMMIT_SEC) == 1750
    assert COLLECT_SEC == 2350
    assert RunSettings().collect_allowance_sec - agent_build.COLLECT_STEPS_SEC == 1850
    assert agent_build.COMMIT_TRIES == 3
    # The sandbox's log scrub is a generous margin over the ~6 s its two files take.
    assert agent_build.SANDBOX_SCRUB_SEC >= 10 * 6


@pytest.mark.parametrize("stop_after", ["probe", "preflight"])
def test_a_stop_build_keeps_its_own_short_ending_whatever_the_collection_allowance(tmp_path, stop_after):
    """A probe or preflight-only build stops the sandbox and writes its records: no snapshot, no
    commit point, no transcript. Its stage keeps ``STOP_BUILD_COLLECT_SEC`` after its hard limit,
    whatever the run's collection allowance, which its brief still states and its check still holds
    to the least a full build needs."""
    floor = agent_build.STOP_BUILD_COLLECT_SEC
    assert AgentBuildParams.model_validate(_params(stop_after=stop_after)).collection_sec == floor
    assert AgentBuildParams.model_validate(_params(stop_after=stop_after, collect_allowance_sec=9000)).collection_sec \
        == floor
    assert AgentBuildParams.model_validate(_params(collect_allowance_sec=9000)).collection_sec == 9000
    shape = STAGE.model_copy(update={"time_sec": SETUP_SEC + STAGING_SEC + HARD_SEC + floor})
    _plan(tmp_path, _params(stop_after=stop_after), shape=shape)
    with pytest.raises(PlanError, match=f"under its floor of {SETUP_SEC + STAGING_SEC + HARD_SEC + floor}s"):
        _plan(tmp_path, _params(stop_after=stop_after), shape=shape.model_copy(update={"time_sec": shape.time_sec - 1}))
    with pytest.raises(PlanError, match=r"run.collect_allowance_sec\) is 2349s"):
        _plan(tmp_path, _params(stop_after=stop_after, collect_allowance_sec=COLLECT_SEC - 1), shape=shape)


def test_check_refuses_a_harness_role_with_gpus():
    gpu = ResourceShape(gpus=1, gpu_kind="H200", cpus=1, memory_gib=2, time_sec=600)
    params = AgentBuildParams.model_validate(_params(snapshot_shape=gpu.model_dump()))
    roles = concurrent([SANDBOX, gpu])
    shape = roles.model_copy(update={"memory_gib": roles.memory_gib + RUNNER_GIB,
                                     "disk_gib": roles.disk_gib + KEPT_DISK_GIB + DATA_GIB, "time_sec": STAGE_TIME})
    with pytest.raises(ValueError, match="no GPUs"):
        AgentBuildStage.check(shape, params, runner_memory_gib=RUNNER_GIB)


def test_a_site_whose_host_refuses_the_spawners_namespaces_plans_no_build(tmp_path, monkeypatch):
    """The sandbox's spawner makes the commands' user, mount and PID namespaces: a site whose
    substrate can tell when the plan is made that its host refuses them (one Docker host) plans
    no build there, and says what to set; Slurm's login node cannot tell, and leaves it to the
    runner's seal check on the node."""
    from hls.stages import agent_build
    from hls.substrate.site_checks import SiteProblem
    from tests.pipeline.plans import settings

    slurm = settings(tmp_path).site
    assert AgentBuildStage.check_site(SANDBOX, None, site=slurm) is None
    said = "this host refuses ...: kernel.apparmor_restrict_unprivileged_userns is 1; set it to 0"
    monkeypatch.setattr(agent_build, "substrate",
                        lambda site: SimpleNamespace(spawner_problem=lambda: SiteProblem(said, retryable=False)))
    with pytest.raises(ValueError, match=re.escape(f"the build sandbox cannot run on this site: {said}")):
        AgentBuildStage.check_site(SANDBOX, None, site=slurm)




def test_the_stage_refuses_to_copy_more_data_than_its_plan_made_room_for(tmp_path):
    """The stage's disk holds the copies of the agent's data its plan declared
    (``data_copies``), so a node that passed the job's start check never refuses the sandbox for
    its scratch. A provision record whose copy takes more than declared, an entry published again
    larger since the plan, is infra not retried before anything is copied."""
    ctx = _ctx(tmp_path, _params())
    build = agent_build._Build(ctx)
    provisioned = build.provisioned()
    assert copied_entries(("agent",), provisioned).keys() == DATA_COPIES.keys()
    larger = provisioned.model_copy(update={"weights": provisioned.weights.model_copy(
        update={"disk_bytes": 2**20 + 1})})
    with pytest.raises(StageFault) as e:
        build.data(larger)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    [weights] = DATA_COPIES
    assert e.value.fault.reason == (f"the agent's data cannot be mounted: the copy of {weights} takes 1048577 bytes, "
                                    "more than the 1048576 the plan made room for (the entry was published again "
                                    "since): plan the instance again")
    assert ctx.launcher.launched == []
    mounts, _ = build.data(provisioned)
    assert [m.target for m in mounts] == ["/data/weights"]


def test_the_most_roles_an_attempt_starts_counts_a_snapshot_for_every_interval():
    params = AgentBuildParams.model_validate(_params(hard_sec=1100, snapshot_interval_sec=100))
    # The agent cap's extraction, three copies, the seed, the uploads, the sandbox, 12 snapshots,
    # three commit point tries, the transcript and harbor's reading of it.
    assert AgentBuildStage.max_roles(params) == 1 + 3 + 1 + 1 + 1 + 12 + 3 + 1 + 1


def test_the_seal_opens_exactly_the_route_host_and_vouches_its_addresses():
    from hls.egress.policy import NetworkPolicy

    air = NetworkPolicy.model_validate({"domains": {"mode": "allow", "entries": ["pypi.org"]}})
    sealed = seal_policy(air, "relay.example", resolve=lambda h: ["10.0.0.7"])
    assert sealed.network.domains.entries == ["pypi.org", "relay.example"]
    assert sealed.trusted_internal_hosts == ("relay.example", "10.0.0.7")
    # The task's policy is left as it was: the vouch is the seal's, outside it.
    assert air.domains.entries == ["pypi.org"] and sealed.network.packages == air.packages
    deny = NetworkPolicy.model_validate({"domains": {"mode": "deny", "entries": ["evil.example"]}})
    assert seal_policy(deny, "api.anthropic.com", resolve=lambda h: []).network.domains.entries == ["evil.example"]


def test_the_seal_vouches_the_route_host_whether_or_not_it_adds_it_and_names_each_host_once():
    from hls.egress.policy import NetworkPolicy

    # A deny list adds nothing, and an allow list already naming the host is left as it is; the
    # host and its addresses are vouched for either way.
    deny = NetworkPolicy.model_validate({"domains": {"mode": "deny", "entries": ["evil.example"]}})
    sealed = seal_policy(deny, "relay.example", resolve=lambda h: ["10.0.0.7"])
    assert sealed.network.domains.entries == ["evil.example"]
    assert sealed.trusted_internal_hosts == ("relay.example", "10.0.0.7")
    listed = NetworkPolicy.model_validate({"domains": {"mode": "allow",
                                                       "entries": ["relay.example", "pypi.org"]}})
    sealed = seal_policy(listed, "relay.example", resolve=lambda h: ["10.0.0.7"])
    assert sealed.network.domains.entries == ["relay.example", "pypi.org"]
    assert sealed.trusted_internal_hosts == ("relay.example", "10.0.0.7")
    # A host given as an address resolves to itself, and is vouched for once.
    air = NetworkPolicy.model_validate({"domains": {"mode": "allow", "entries": []}})
    sealed = seal_policy(air, "10.0.0.9")
    assert (sealed.network.domains.entries, sealed.trusted_internal_hosts) == (["10.0.0.9"], ("10.0.0.9",))


def test_budget_params_keep_the_wall_clock_and_refuse_what_the_runner_cannot_keep():
    from hls.artifact.artifact import AgentProducerSpec, Budget, HighestAvailable

    ok = AgentProducerSpec(hard_budget=Budget(wallclock_sec=45000, turns=None),
                           soft_budget=Budget(wallclock_sec=43200), reasoning_effort=HighestAvailable())
    assert agent_build.budget_params(ok) == {"soft_sec": 43200, "hard_sec": 45000, "max_turns": None}
    capped = AgentProducerSpec(hard_budget=Budget(wallclock_sec=3600, turns=40), reasoning_effort=HighestAvailable())
    assert agent_build.budget_params(capped) == {"soft_sec": None, "hard_sec": 3600, "max_turns": 40}
    soft_turns = AgentProducerSpec(hard_budget=Budget(wallclock_sec=3600),
                                   soft_budget=Budget(wallclock_sec=10, turns=5), reasoning_effort=HighestAvailable())
    with pytest.raises(ValueError, match="a soft turn budget has no tool that reports it"):
        agent_build.budget_params(soft_turns)


def test_sample_tasks_budgets_are_ones_the_runner_keeps():
    """sample_task builds under an hour's hard wall and 40 turns
    with a 30-minute soft wall, so a build of it plans."""
    from tasks.sample_task.task import ARTIFACT

    assert agent_build.budget_params(ARTIFACT.producer) == {"soft_sec": 1800, "hard_sec": 3600, "max_turns": 40}


# ---------------------------------------------------------------------------
# How the agent's part ended
# ---------------------------------------------------------------------------


def _fresh_status(ctx: StageContext, monkeypatch, age: float = 0.0) -> None:
    """The sandbox's interceptor, as the seal reads it, having reported *age* seconds ago."""
    from hls.substrate.seal import IndexStatus

    status = IndexStatus(total=0, mid_stream=0, oversized=(), policy_refused=(), fetches_refused=0,
                         heartbeat=time.time() - age)
    monkeypatch.setattr(ctx, "seal_status", lambda name: status)


@pytest.mark.parametrize("message, ending", [
    ("Command failed (exit 1): mkdir -p /logs/agent /workspace && printf ...", "harbor_command_failed"),
    ("Command failed (exit 1): export PATH=x; (echo $BASHPID > /logs/agent/claude.pid; exec claude --verbose",
     "agent_nonzero_exit"),
    ("Command failed (exit 137): tail -c +1 /logs/agent/claude-code.txt\nstdout: exec claude ", "harbor_command_failed"),
], ids=["harbor-command", "the-cli", "cli-named-only-in-its-output"])
def test_a_failed_harbor_command_after_the_clock_ends_the_agents_part(tmp_path, message, ending):
    """After the clock started, the agent's leftover processes can kill any of harbor's commands:
    a failed one ends the agent's part, graded, the CLI's own launch as its nonzero exit and any
    other as a failed harbor command, told by the command alone, never by its output."""
    from harbor.agents.installed.base import NonZeroAgentExitCodeError

    build = agent_build._Build(_ctx(tmp_path, _params()))
    assert build.ending("agent_ended", NonZeroAgentExitCodeError(message, return_code=1)) == ending
    [cut] = [e for e in _events(build.ctx) if e["kind"] == "build.agent_interrupted"]
    assert cut["reason"] == ending
    with pytest.raises(StageFault, match="harbor's run failed: KeyError") as e:
        build.ending("agent_ended", KeyError("x"))
    assert e.value.fault.retryable
    assert build.ending("hard_limit", None) == "hard_limit" and build.ending("agent_ended", None) == "agent_ended"
    with pytest.raises(StageFault, match="shutting down"):
        build.ending("stopping", None)


def test_a_lost_channel_is_the_sandboxs_end_whatever_harbor_raised(tmp_path, monkeypatch):
    """Harbor can turn a lost channel into the agent's own nonzero exit; the environment's record
    of the loss decides, and the sandbox's roles say whose fault it is."""
    from harbor.agents.installed.base import NonZeroAgentExitCodeError

    from hls.agent.environment import SandboxLost

    class _Decided(Exception):
        pass

    build = agent_build._Build(_ctx(tmp_path, _params()))

    def judged(env, failed_at):
        raise _Decided(str(env.lost))

    monkeypatch.setattr(build, "sandbox_ended", judged)
    error = NonZeroAgentExitCodeError("Command failed (exit 1): (echo $BASHPID > p; exec claude --verbose",
                                      return_code=1)
    env = SimpleNamespace(lost=SandboxLost("the channel was lost"))
    running = SimpleNamespace(poll=lambda: None)
    with pytest.raises(_Decided, match="the channel was lost"):
        build.end("agent_ended", error, env, running, ended_at=time.time())


def _split() -> SplitEvidence:
    return SplitEvidence(reserve_bytes=4 * GIB, ended_at=10.0, oom_at=9.0, oom_by="runner", sample_at=8.0,
                         total_bytes=17 * GIB, tmpfs_bytes=0, harness_bytes=GIB, contestant_bytes=16 * GIB)


def test_a_sandbox_that_ended_by_itself_as_the_stage_stopped_it_is_judged_by_its_roles(tmp_path,
                                                                                        monkeypatch):
    """The agent's part ended normally, but the sandbox ran out of memory just before the stage
    stopped it: its own ending stands, the build rule makes it the contestant's, and the build is
    collected as one that ended there."""
    build = agent_build._Build(_ctx(tmp_path, _params()))
    build.running["sandbox"] = SimpleNamespace(poll=lambda: None)
    oom = RoleOutcome("sandbox", "contestant", 137, "memory_limit", 1, split=_split())
    monkeypatch.setattr(build.ctx, "stop_role", lambda name: oom)
    monkeypatch.setattr(build.ctx, "pids_hits", lambda name: None)
    _fresh_status(build.ctx, monkeypatch)
    ending, outcome, fault = build.end("agent_ended", None, SimpleNamespace(lost=None),
                                       SimpleNamespace(poll=lambda: None), ended_at=time.time())
    assert (ending, outcome) == ("contestant_oom", oom)
    assert (fault.fault_class, fault.evidence["roles"]) == ("contestant", "sandbox")


#: When the agent's part ended in the evidence tests, on the runner's clock.
FAILED_AT = 50_000.0
#: An age past the evidence's (``OOM_EVIDENCE_MAX_AGE_SEC``).
STALE = OOM_EVIDENCE_MAX_AGE_SEC + 1


def _lost(tmp_path, monkeypatch, *, pids: int | None = None, pids_age: float = 1.0, kills: tuple[float, ...] = ()):
    """A build whose exec channel was lost while its container still runs, the sandbox's pids
    limit count at its start 0 and now *pids*, first read at its rise *pids_age* seconds before
    the agent's part ended (``FAILED_AT``), and the runner's memory kills in it *kills* seconds
    before then; the stage's stop of the sandbox ends it as planned, or ``contestant_pids`` once
    the count rose within the evidence's age, as the context reads it before the stop."""
    from hls.agent.environment import SandboxLost

    build = agent_build._Build(_ctx(tmp_path, _params()))
    build.running["sandbox"] = SimpleNamespace(poll=lambda: None, wait=lambda sec: None)
    build.pids = agent_build.PidsCounts(at_start=0, at_end=None, rose_at=None)
    hits = None if pids is None else PidsHits(count=pids, rose_at=FAILED_AT - pids_age if pids else None)
    monkeypatch.setattr(build.ctx, "pids_hits", lambda name: hits)
    monkeypatch.setattr(build, "_memory_kill_times", lambda: [FAILED_AT - age for age in kills])
    recent = bool(pids) and pids_age <= OOM_EVIDENCE_MAX_AGE_SEC
    stopped = "contestant_pids" if recent else "planned_stop"
    monkeypatch.setattr(build.ctx, "stop_role", lambda name: RoleOutcome(
        "sandbox", "contestant", 0, stopped, 0, "its step reached its pids limit 2 times" if recent else "",
        split=SplitEvidence(reserve_bytes=4 * GIB, ended_at=10.0)))
    env = SimpleNamespace(lost=SandboxLost("the channel was lost"))
    return build, env


def _end(build, env, error: BaseException | None = None):
    return build.end("agent_ended", error, env, SimpleNamespace(poll=lambda: None), ended_at=FAILED_AT)


@pytest.mark.parametrize("age", [1.0, OOM_EVIDENCE_MAX_AGE_SEC], ids=["recent", "at-the-bound"])
def test_a_channel_lost_after_the_pids_count_rose_is_the_contestants_end(tmp_path, monkeypatch, age):
    build, env = _lost(tmp_path, monkeypatch, pids=2, pids_age=age)
    ending, _, fault = _end(build, env)
    assert (ending, fault.fault_class) == ("contestant_pids", "contestant")
    # When the count rose is the record's.
    assert (build.pids.at_end, build.pids.rose_at) == (2, FAILED_AT - age)


def test_a_pids_count_that_rose_long_before_the_channel_was_lost_is_no_evidence_of_it(tmp_path, monkeypatch):
    """A fork bomb the sandbox survived hours before says nothing about a later loss of its
    channel, which is infra, retried."""
    build, env = _lost(tmp_path, monkeypatch, pids=2, pids_age=STALE)
    with pytest.raises(StageFault) as e:
        _end(build, env)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "exec channel was lost while its container ran" in e.value.fault.reason


@pytest.mark.parametrize("ages, counted", [((1.0,), "1"), ((OOM_EVIDENCE_MAX_AGE_SEC,), "1"),
                                           ((STALE + 3600, 2.0), "1")],
                         ids=["recent", "at-the-bound", "only-the-recent-counts"])
def test_a_channel_lost_after_a_contestant_memory_kill_is_the_contestants_end(tmp_path, monkeypatch, ages,
                                                                             counted):
    """A kill of the runner's ended processes of the sandbox at its memory limit, and then the
    channel was lost: whatever became of the channel, the contestant's evidence ends it. Only
    kills within the evidence's age of the loss count."""
    build, env = _lost(tmp_path, monkeypatch, kills=ages)
    ending, _, fault = _end(build, env)
    assert (ending, fault.fault_class, fault.evidence["memory_kills"]) == ("contestant_oom", "contestant", counted)


def test_a_memory_kill_long_before_the_channel_was_lost_is_no_evidence_of_it(tmp_path, monkeypatch):
    build, env = _lost(tmp_path, monkeypatch, kills=(STALE,))
    with pytest.raises(StageFault) as e:
        _end(build, env)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)


def test_contestant_evidence_read_before_the_stop_outweighs_a_stop_that_ended_as_infra(tmp_path,
                                                                                         monkeypatch):
    """The runner's memory kill in the sandbox, read before the stop, decides even when the stop
    itself ended as an infra end would (its log capper died, ``log_lost``)."""
    build, env = _lost(tmp_path, monkeypatch, kills=(1.0,))
    monkeypatch.setattr(build.ctx, "stop_role", lambda name: RoleOutcome(
        "sandbox", "contestant", 0, "log_lost", 0, split=SplitEvidence(reserve_bytes=4 * GIB, ended_at=10.0)))
    ending, outcome, fault = _end(build, env)
    assert (ending, outcome.ending, fault.fault_class) == ("contestant_oom", "log_lost", "contestant")


def test_no_interceptor_status_when_the_agents_part_ended_is_infra(tmp_path, monkeypatch):
    """An interceptor that recorded no status at all may never have carried the agent's route."""
    build, _ = _lost(tmp_path, monkeypatch, pids=0)
    monkeypatch.setattr(build.ctx, "seal_status", lambda name: None)
    with pytest.raises(StageFault) as e:
        _end(build, SimpleNamespace(lost=None))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "recorded no status" in e.value.fault.reason


def test_a_channel_lost_with_no_pids_or_memory_evidence_is_infra(tmp_path, monkeypatch):
    build, env = _lost(tmp_path, monkeypatch, pids=0)
    with pytest.raises(StageFault) as e:
        _end(build, env)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "exec channel was lost while its container ran" in e.value.fault.reason


def test_a_harbor_command_that_fails_after_the_pids_limit_is_reached_is_not_infra(tmp_path,
                                                                                 monkeypatch):
    """A fork bomb that reached the step's pids limit makes harbor's later commands fail: their
    failure is part of the contestant's end, whatever harbor raised. One the sandbox survived
    long before does not explain it: harbor's failure is then infra."""
    fork = RuntimeError("fork: retry: Resource temporarily unavailable")
    build, _ = _lost(tmp_path, monkeypatch, pids=3)
    ending, _, fault = _end(build, SimpleNamespace(lost=None), fork)
    assert (ending, fault.fault_class) == ("contestant_pids", "contestant")
    build, _ = _lost(tmp_path / "stale", monkeypatch, pids=3, pids_age=STALE)
    with pytest.raises(StageFault) as e:
        _end(build, SimpleNamespace(lost=None), fork)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "harbor's run failed" in e.value.fault.reason


def test_a_failure_of_harbors_run_after_a_memory_kill_is_the_contestants_end(tmp_path,
                                                                            monkeypatch):
    build, _ = _lost(tmp_path, monkeypatch, kills=(1.0, 2.0))
    ending, _, fault = _end(build, SimpleNamespace(lost=None), RuntimeError("the CLI's pipe broke"))
    assert (ending, fault.evidence["memory_kills"]) == ("contestant_oom", "2")
    # A memory kill that left the run to its normal end is no end of the sandbox's.
    build, _ = _lost(tmp_path / "again", monkeypatch, kills=(1.0, 2.0))
    _fresh_status(build.ctx, monkeypatch)
    ending, _, fault = _end(build, SimpleNamespace(lost=None))
    assert (ending, fault) == ("agent_ended", None)
    # Nor do kills long before a failure of harbor's run explain it.
    build, _ = _lost(tmp_path / "stale", monkeypatch, kills=(STALE, STALE + 60))
    with pytest.raises(StageFault) as e:
        _end(build, SimpleNamespace(lost=None), RuntimeError("the CLI's pipe broke"))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)


@pytest.mark.parametrize("age, infra", [(0.0, False), (120.0, True)], ids=["answering", "stale"])
def test_an_agents_end_while_its_interceptor_had_stopped_reporting_is_infra(tmp_path,
                                                                            monkeypatch, age, infra):
    """The sandbox's interceptor carries the agent's route: one that had not reported for more
    than a minute when the agent's part ended may have cut it off from its model, the harness's
    doing, so the end is infra and retried."""
    build, _ = _lost(tmp_path, monkeypatch, pids=0)
    _fresh_status(build.ctx, monkeypatch, age)
    if not infra:
        assert build.end("agent_ended", None, SimpleNamespace(lost=None), SimpleNamespace(poll=lambda: None),
                         ended_at=time.time())[0] == "agent_ended"
        return
    with pytest.raises(StageFault) as e:
        build.end("agent_ended", None, SimpleNamespace(lost=None), SimpleNamespace(poll=lambda: None),
                  ended_at=time.time())
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "interceptor had not reported for 120 s" in e.value.fault.reason


def test_every_pin_and_copy_happens_before_the_first_role(sandbox_bins, tmp_path, monkeypatch):
    """A pin or a cache copy cannot run beside another role, so the agent cap and every
    entry the sandbox mounts are on the node before the workspace seed, the first role."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params(), image=fake_sandbox.image_bytes({"baked.txt": "x\n"}))
    order: list[tuple[str, str]] = []
    depth = [0]

    def spy(name, method):
        def call(*args, **kw):
            if depth[0] == 0:
                order.append((name, args[0]))
            depth[0] += 1
            try:
                return method(*args, **kw)
            finally:
                depth[0] -= 1
        return call

    for name in ("materialize", "node_image", "start_role", "start_container"):
        monkeypatch.setattr(ctx, name, spy(name, getattr(ctx, name)))
    AgentBuildStage().run(ctx)
    pins = [(name, what) for name, what in order if name in ("materialize", "node_image")]
    assert pins == [("node_image", IMAGE_KEY), ("materialize", "weights")]
    assert order[:2] == pins and order[2] == ("start_role", "seed-workspace")


@pytest.mark.parametrize("line", ["", "echo 'refused: there is nothing to serve' >&2; "], ids=["plain", "refused"])
def test_a_sandbox_that_prints_a_refusal_is_judged_as_one_that_does_not(sandbox_bins, tmp_path,
                                                                        monkeypatch, line):
    """A refusal (exit 3 or 4 with a ``refused:`` line) is the provision roles' own: a
    contestant's sandbox that prints one and exits 3 is judged by its roles alone, here infra and
    retried, as without the line."""
    _script(sandbox_bins, window=1_000_000, claude=[{}])
    monkeypatch.setattr(exec_protocol, "server_command", lambda **kw: ("/bin/sh", "-c", f"{line}exit 3"))
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable, fault.evidence["roles"]) == ("infra", True, "sandbox")
    assert "ended exited with status 3 and no OOM recorded" in fault.reason


def test_an_agent_cli_that_prints_a_refusal_and_exits_3_is_a_nonzero_exit_resumed_then_collected(
        sandbox_bins, tmp_path, monkeypatch):
    control = sandbox_bins
    monkeypatch.setattr(claude_code, "REENTRY_ERROR_PAUSE_SEC", 0.0)
    _script(control, window=1_000_000,
            claude=[{"commit": "v1\n", "stderr": "refused: I will not build this", "exit": 3},
                    {"stderr": "refused: still not", "exit": 3}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert (record.ending, record.fault) == ("agent_nonzero_exit", None) and record.reentries >= 1
    assert len(_calls(control)) == record.reentries + 1 and "source_bundle" in outputs
    [cut] = [e for e in _events(ctx) if e["kind"] == "build.agent_interrupted"]
    assert cut["reason"] == "agent_nonzero_exit"


def test_the_harness_roles_run_on_the_sandboxs_cpus(sandbox_bins, tmp_path):
    """Every harness role of the stage, the workspace's seed, the uploads, a snapshot, the commit
    point or the transcript copy, overlaps the CPUs the sandbox's step holds; the sandbox's step
    takes its own. The commit point runs with its own shape."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 2.5, "advance": 900}])
    ctx = _ctx(tmp_path, _params(snapshot_interval_sec=1),
               image=fake_sandbox.image_bytes({"baked.txt": "x\n"}))
    AgentBuildStage().run(ctx)
    # An image's extraction runs alone, before every other role, on the stage's whole shape.
    launched = [s for s in ctx.launcher.launched if not s.name.startswith(("extract-", "copy-"))]
    assert {"sandbox", "seed-workspace", "uploads", "snapshot-1", "commit", "transcript"} <= {s.name for s in launched}
    assert all(s.overlap == (s.owner == "harness") for s in launched)


def test_the_commit_point_runs_with_its_own_shape(sandbox_bins, tmp_path):
    """Git's bundling peaks past a harness role's 2 GiB: the commit point has its own shape, one
    CPU and 4 GiB, which it can take once the sandbox has stopped."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    AgentBuildStage().run(ctx)
    shapes = {s.name: s.shape for s in ctx.launcher.launched}
    assert shapes["commit"] == COMMIT.model_copy(update={"time_sec": STAGE.time_sec})
    assert shapes["transcript"] == SNAPSHOT.model_copy(update={"time_sec": STAGE.time_sec})


def test_a_stage_that_decides_its_contestant_faults_reads_only_the_one_it_declared(tmp_path):
    """``contestant_fault``, which the runner reads at a SIGTERM, is for this stage the zero it
    declared, and None before."""
    ctx = _ctx(tmp_path, _params())
    assert AgentBuildStage.decides_contestant_faults and ctx.contestant_fault() is None
    zero = FaultRecord(fault_class="contestant", reason="a test", retryable=False)
    with pytest.raises(ValueError, match="is not a contestant fault"):
        ctx.declare_contestant_fault(infra_fault("a test", retryable=True))
    assert ctx.declare_contestant_fault(zero) == zero and ctx.contestant_fault() == zero


# ---------------------------------------------------------------------------
# The route's base URL, the logs and the records
# ---------------------------------------------------------------------------


def test_no_record_event_log_or_transcript_holds_the_base_url_or_its_host(sandbox_bins, tmp_path):
    """A relay's base URL is a secret, and so is its host: after a build on a relay route,
    neither is in any record, event, role log or the transcript copy, while the seal's policy,
    which the sandbox's command line carries, names the host."""
    control = sandbox_bins
    _script(control, window=200_000, claude=[{"commit": "v1\n", "advance": 900, "leak": "ANTHROPIC_BASE_URL"}])
    ctx = _ctx(tmp_path, _params(K3))
    outputs = AgentBuildStage().run(ctx)
    url = SECRETS["kimi_url"]
    [sandbox] = [s for s in ctx.launcher.launched if s.name == "sandbox"]
    assert any(HOST in a for a in sandbox.argv) and not any(url in a for a in sandbox.argv)
    texts = {name: ctx.store.read_bytes(outputs[name]).decode() for name in ("build", "agent_env", "brief")}
    texts["events"] = (ctx.attempt_dir / "events.jsonl").read_text()
    texts |= {p.name: p.read_text(errors="replace") for p in (ctx.attempt_dir / "logs").iterdir() if p.is_file()}
    copy = ctx.store.path(outputs["transcript"]).parent
    texts |= {f"transcript/{p.name}": p.read_text(errors="replace") for p in copy.rglob("*") if p.is_file()}
    for name, text in texts.items():
        assert url not in text and HOST not in text.lower(), name
    assert (copy / "logs" / "printed-env.txt").read_text() == "KEY=<redacted:kimi_url>\n"
    variables = json.loads(texts["agent_env"])
    assert {var: source for var, source in variables["exec"].items() if source != "harness"} == {
        "ANTHROPIC_AUTH_TOKEN": "grant:kimi_token", "ANTHROPIC_BASE_URL": "grant:kimi_url",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "plain", "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "plain"}


def test_the_sandboxs_seal_argument_naming_the_route_host_is_not_refused(sandbox_bins, tmp_path):
    """The seal's policy names the route's host on the sandbox's command line: the host alone
    gives no access, so the context, which refuses a whole secret value there, starts it."""
    _script(sandbox_bins, window=200_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params(K3))
    AgentBuildStage().run(ctx)
    [sandbox] = [s for s in ctx.launcher.launched if s.name == "sandbox"]
    policy = next(json.loads(a) for a in sandbox.argv if a.startswith("{") and '"domains"' in a)
    assert policy["network"]["domains"]["entries"] == [HOST] and HOST in policy["trusted_internal_hosts"]
    assert _ended(ctx, "sandbox")["ending"] == "planned_stop"


def test_the_sandboxs_log_reaches_the_attempt_scrubbed(sandbox_bins, tmp_path, monkeypatch):
    """The exec server logs the first 120 characters of each command; a command that names the
    route's host reaches the attempt's log of the sandbox only scrubbed."""
    control = sandbox_bins
    _script(control, window=200_000, claude=[{"commit": "v1\n", "advance": 900}])
    setup = claude_code.ClaudeCode.setup

    async def naming_the_host(self, environment):
        await self.exec_as_agent(environment, f"echo dial {HOST.upper()} and {HOST}")
        await setup(self, environment)

    monkeypatch.setattr(claude_code.ClaudeCode, "setup", naming_the_host)
    ctx = _ctx(tmp_path, _params(K3))
    AgentBuildStage().run(ctx)
    log = (ctx.attempt_dir / "logs" / "sandbox.err").read_text()
    # The relays of the test's routes share a host, which either secret's name stands for.
    assert re.search(r"cmd=set -o pipefail; echo dial <redacted:\w+\.host> and <redacted:\w+\.host>", log)
    assert HOST not in log.lower()


def test_codexs_config_reaches_the_transcript_with_its_base_url_redacted(sandbox_bins, tmp_path):
    """Codex reads its base URL from its config file, which harbor copies into the agent's logs;
    the transcript copy scrubs it like every file it copies."""
    control = sandbox_bins
    _script(control, window=258_400, codex=[{"commit": "v1\n"}, {"advance": 900}])
    ctx = _ctx(tmp_path, _params(GPT))
    outputs = AgentBuildStage().run(ctx)
    copy = ctx.store.path(outputs["transcript"]).parent / "logs"
    config = (copy / "codex-config.toml").read_text()
    assert "<redacted:openai_url>" in config and RELAY_URL not in config and HOST not in config
    live = SandboxLayout(ctx.scratch / "roles" / "sandbox").agent_logs
    assert RELAY_URL in (live / "codex-config.toml").read_text()


def test_a_credential_value_never_reaches_a_log_at_debug(sandbox_bins, tmp_path):
    """Harbor logs each command's output as an extra of its records; while the stage runs, every
    record of harbor's loggers is scrubbed before any handler sees it, its message, its extras
    and its traceback, also of loggers made during the build."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900, "say": "ANTHROPIC_API_KEY"}])

    class _Whole(logging.Handler):
        def __init__(self) -> None:
            super().__init__(logging.DEBUG)
            self.seen: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            text = json.dumps({k: str(v) for k, v in vars(record).items()})
            if record.exc_info:
                text += logging.Formatter().formatException(record.exc_info)
            self.seen.append(text)

    whole = _Whole()
    loggers = [logging.getLogger("harbor"), logging.getLogger("hls")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(whole)
        lg.setLevel(logging.DEBUG)
    try:
        ctx = _ctx(tmp_path, _params())
        AgentBuildStage().run(ctx)
    finally:
        for lg, level in zip(loggers, levels):
            lg.removeHandler(whole)
            lg.setLevel(level)
    said = [t for t in whole.seen if "env ANTHROPIC_API_KEY=" in t]
    assert said and all("<redacted:anthropic_key>" in t for t in said)
    assert not [t for t in whole.seen if KEY in t]
    # Once the stage is over, harbor's loggers are as they were.
    assert not [f for lg in logging.Logger.manager.loggerDict.values()
                if isinstance(lg, logging.Logger) for f in lg.filters if isinstance(f, agent_build._ScrubRecords)]


# ---------------------------------------------------------------------------
# The contestant's evidence, the harbor commands, the guards
# ---------------------------------------------------------------------------


def test_the_commit_points_output_is_published_by_its_writers_report(sandbox_bins, tmp_path):
    """The commit point reports the digest and size of what it wrote, keyed by its name, in the
    report the stage made in its node-local directory; the manifest records the output by it."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    report = json.loads(ctx.outputs_report("commit").read_text())
    bundle = ctx.store.path(outputs["source_bundle"])
    assert report == {"source.bundle": {"sha256": sha256_file(bundle)[0], "size": bundle.stat().st_size}}
    assert ctx.output_ref(outputs["source_bundle"]).sha256 == report["source.bundle"]["sha256"]


def test_the_transcripts_files_are_published_by_their_writers_report(sandbox_bins, tmp_path):
    """The transcript role reports its index, which pins every file it copied by its digest; the
    manifest records the transcript by that report, and the index's digests are the copies'."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    index_path = ctx.store.path(outputs["transcript"])
    report = json.loads(ctx.outputs_report("transcript").read_text())
    assert report == {"transcript/INDEX.json": {"sha256": sha256_file(index_path)[0],
                                                "size": index_path.stat().st_size}}
    for entry in json.loads(index_path.read_text())["files"]:
        assert sha256_file(index_path.parent / "logs" / entry["path"])[0] == entry["sha256"]


@pytest.mark.parametrize("graded", [True, False], ids=["oom-graded", "oom-zero"])
def test_a_fork_bomb_that_exhausts_the_pids_limit_ends_the_build_and_its_last_commit_is_graded(
        sandbox_bins, tmp_path, graded):
    """A fork bomb that reaches the sandbox step's pids limit is a certain contestant end, judged
    as an own-limit OOM: the build ends, and its last commit is graded, whatever the plan's rule
    for a sandbox's end on its memory."""
    control = sandbox_bins
    cgroups = tmp_path / "cg"
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900,
                                                "touch": {str(cgroups / "cg-sandbox" / "pids.events"): "max 2\n"}}])
    ctx = _ctx(tmp_path, _params(), launcher=LocalLauncher(gpus=1, cgroup_root=cgroups),
               run={"grade_after_sandbox_oom": graded})
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert (record.ending, record.fault.fault_class, record.pids_max_events.at_end) == ("contestant_pids",
                                                                                       "contestant", 2)
    assert record.grade_after_sandbox_oom is graded
    assert _ended(ctx, "sandbox")["ending"] == "contestant_pids"
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)


def test_a_pids_exhaustion_with_nothing_gradable_is_the_contestants_zero(sandbox_bins, tmp_path):
    control = sandbox_bins
    cgroups = tmp_path / "cg"
    _script(control, window=1_000_000, claude=[{"advance": 900,
                                                "touch": {str(cgroups / "cg-sandbox" / "pids.events"): "max 1\n"}}])
    ctx = _ctx(tmp_path, _params(), launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("contestant", False) and ctx.contestant_fault() == fault
    assert "process limit" in fault.reason and not ctx.output_path("build.json").exists()


def test_a_sandbox_that_dies_while_the_agent_runs_is_reported_within_a_tick(sandbox_bins, tmp_path, monkeypatch):
    """The stage looks at the sandbox every tick while the agent runs and reports it as soon as
    it ended, before anything else, here before harbor's environment takes its time to close:
    the time of its end, which the build rule ages its last OOM against, is its own."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "kill_server": True, "hang": True}])
    stop = MappedEnvironment.stop

    async def slow_stop(self, delete=False):
        await asyncio.sleep(4)
        await stop(self, delete)

    monkeypatch.setattr(MappedEnvironment, "stop", slow_stop)
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault):
        AgentBuildStage().run(ctx)
    killed = float((control / "killed_at").read_text())
    ended = _ended(ctx, "sandbox")["split"]["ended_at"]
    assert 0 <= ended - killed < agent_build.TICK_SEC + 2.0


def test_the_contestants_evidence_comes_before_the_interceptors(sandbox_bins, tmp_path,
                                                                  monkeypatch):
    """A fork bomb that reached the pids limit may also have starved the interceptor: the
    contestant's evidence decides first, so the build ends ``contestant_pids`` and is graded."""
    control = sandbox_bins
    cgroups = tmp_path / "cg"
    monkeypatch.setenv("FAKE_PROXY_HEARTBEAT_AGE", "300")
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900,
                                                "touch": {str(cgroups / "cg-sandbox" / "pids.events"): "max 4\n"}}])
    ctx = _ctx(tmp_path, _params(), launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    outputs = AgentBuildStage().run(ctx)
    assert _record(ctx, outputs).ending == "contestant_pids"


def test_a_harbor_command_killed_after_the_cli_exits_0_ends_the_agents_part_and_is_graded(
        sandbox_bins, tmp_path, monkeypatch):
    """The agent's leftover processes can kill harbor's look at the CLI's exit after it exited
    0; harbor raises that command's failure, and the build is graded on its last commit."""
    from harbor.agents.installed.base import NonZeroAgentExitCodeError

    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])

    async def killed(self, environment, env, stream_start, *, failed):
        raise NonZeroAgentExitCodeError(f"Command failed (exit 137): echo size {KEY}; tail -c +1 x", return_code=137)

    monkeypatch.setattr(claude_code.ClaudeCode, "_read_exit", killed)
    ctx = _ctx(tmp_path, _params())
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert (record.ending, record.fault) == ("harbor_command_failed", None) and "source_bundle" in outputs
    [cut] = [e for e in _events(ctx) if e["kind"] == "build.agent_interrupted"]
    assert cut["reason"] == "harbor_command_failed" and "tail -c +1" in cut["command"]
    assert KEY not in cut["command"] and "<redacted:anthropic_key>" in cut["command"]


def test_a_failed_harbor_command_during_setup_is_infra(sandbox_bins, tmp_path, monkeypatch):
    """Before the clock no contestant process exists: a harbor command that fails in setup is
    the harness's, retried."""
    from harbor.agents.installed.base import NonZeroAgentExitCodeError

    _script(sandbox_bins, window=1_000_000, claude=[{}])

    async def failing(self, environment):
        raise NonZeroAgentExitCodeError("Command failed (exit 1): mkdir -p /logs/agent", return_code=1)

    monkeypatch.setattr(claude_code.ClaudeCode, "setup", failing)
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "the sandbox's setup failed" in e.value.fault.reason and _calls(sandbox_bins) == []


class _QuickStart(MappedEnvironment):
    """The sandbox's environment with a one-second wait for its exec server's answer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs, start_timeout_sec=1.0)


@pytest.mark.parametrize("stop_after", [None, pytest.param("probe", marks=pytest.mark.skipif(
    os.getuid() == 0, reason="the probe's PID 1 checks would reach the machine's init as root"))],
    ids=["build", "probe-build"])
def test_the_sandboxs_start_wait_counts_from_its_runtimes_start_marker(sandbox_bins, tmp_path, monkeypatch,
                                                                       stop_after):
    """The runtime and the seal may take long to start the sandbox's command, its exec server: that
    time is the harness's, bounded by the stage's deadline alone. The wait for the server's answer
    counts from the runtime's start marker, so a start slower than the whole wait is no failure."""
    monkeypatch.setenv("FAKE_SANDBOX_START_DELAY", "3")
    monkeypatch.setattr(environment, "SandboxEnvironment", _QuickStart)
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params(stop_after=stop_after) if stop_after else _params())
    outputs = AgentBuildStage().run(ctx)
    if stop_after is None:
        assert _record(ctx, outputs).ending == "agent_ended"
    else:
        [ready] = [c for c in _probe(ctx, outputs).checks if c.name == "exec-server-ready"]
        assert ready.ok, ready


def test_a_sandbox_whose_runtime_never_starts_its_command_is_infra_by_the_stages_deadline(sandbox_bins, tmp_path,
                                                                                          monkeypatch):
    """Only the stage's deadline bounds the runtime's start: a sandbox whose command has not started
    by then is infra, retried, and no agent starts."""
    monkeypatch.setenv("FAKE_SANDBOX_START_DELAY", "60")
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    ctx.deadline = time.time() + 3
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "did not start its command" in fault.reason
    assert _calls(sandbox_bins, preflight=True) == [] and _calls(sandbox_bins) == []


#: Each harness's script for one session that commits once and ends (``tests.agent.fake_cli``).
_ONE_SESSION = {"claude-code": {"window": 1_000_000, "claude": [{"commit": "v1\n", "advance": 900}]},
                "codex": {"window": 258_400, "codex": [{"commit": "v1\n"}, {"advance": 900}]}}


def _harness_class(arm: ArmRecord) -> type:
    """The harbor class whose ``run`` drives *arm*'s CLI."""
    from harbor.agents.installed.codex import Codex

    return claude_code.ClaudeCode if arm.harness == "claude-code" else Codex


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_the_exec_server_is_given_the_arms_own_launch_marker_and_no_other(sandbox_bins, tmp_path, arm):
    """The exec server's start lines say whether a command launches the agent's CLI by the markers
    it is given: the arm's adapter's alone (``HarnessAdapter.cli_launch``), so a command that
    launches another harness's CLI never counts as this agent's launch."""
    _script(sandbox_bins, **_ONE_SESSION[arm.harness])
    ctx = _ctx(tmp_path, _params(arm))
    AgentBuildStage().run(ctx)
    command = fake_sandbox.records(Path(os.environ["FAKE_SANDBOX_RECORDS"]))["sandbox"]["command"]
    given = [command[i + 1] for i, arg in enumerate(command) if arg == "--cli-launch"]
    assert given == [adapter_for(arm).cli_launch]


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_a_harbor_command_that_fails_after_the_clock_before_the_clis_first_launch_is_infra(
        sandbox_bins, tmp_path, monkeypatch, arm):
    """Harbor's run sends commands of its own before it launches the CLI (its setup, the brief, a
    config): one that fails there, the clock started, failed with no contestant process ever run,
    since the preflight's launch came before the clock. The exec server's own start lines in the
    sandbox's log show no launch of the CLI since the clock started, so the end is infra, retried,
    never the contestant's zero."""
    _script(sandbox_bins, **_ONE_SESSION[arm.harness])

    async def fails_before_the_cli(self, instruction, environment, context):
        await self.exec_as_agent(environment, command="false")

    monkeypatch.setattr(_harness_class(arm), "run", fails_before_the_cli)
    ctx = _ctx(tmp_path, _params(arm))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "before the agent's CLI was launched" in fault.reason
    assert fault.evidence["agent_ending"] == "harbor_command_failed"
    assert _calls(sandbox_bins) == [] and "build.started" in [ev["kind"] for ev in _events(ctx)]


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_a_harbor_command_that_fails_after_the_clis_launch_ends_the_agents_part_and_is_graded(
        sandbox_bins, tmp_path, monkeypatch, arm):
    """Once the CLI has been launched in the agent's part, the agent's leftover processes can kill
    any of harbor's commands: a failed one ends the agent's part as ``harbor_command_failed``, and
    the build is graded on its last commit, for both harnesses."""
    _script(sandbox_bins, **_ONE_SESSION[arm.harness])
    cls = _harness_class(arm)
    real = cls.run

    async def fails_after_the_cli(self, instruction, environment, context):
        await real(self, instruction, environment, context)
        await self.exec_as_agent(environment, command="false")

    monkeypatch.setattr(cls, "run", fails_after_the_cli)
    ctx = _ctx(tmp_path, _params(arm))
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert (record.ending, record.fault) == ("harbor_command_failed", None) and "source_bundle" in outputs
    assert _calls(sandbox_bins)


def test_a_cli_launch_the_exec_server_never_started_is_infra(sandbox_bins, tmp_path, monkeypatch):
    """The CLI's launch can fail before any contestant process starts: the exec server answers 126
    when it cannot hand the command over, and logs no start line for it. Harbor raises the launch's
    nonzero exit all the same; with no launch in the server's own log the end is infra, retried."""
    from harbor.environments.base import ExecResult

    _script(sandbox_bins, **_ONE_SESSION["codex"])
    real = MappedEnvironment.exec

    async def refused_launch(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if "codex exec " in command and cwd is None:
            return ExecResult(stdout="hls-exec: cannot serve the command: OSError: [Errno 24] Too many open files",
                              stderr=None, return_code=126)
        return await real(self, command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    monkeypatch.setattr(MappedEnvironment, "exec", refused_launch)
    ctx = _ctx(tmp_path, _params(GPT))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.evidence["agent_ending"] == "agent_nonzero_exit" and _calls(sandbox_bins) == []


def test_a_sandbox_whose_interceptor_stopped_reporting_is_infra(sandbox_bins, tmp_path,
                                                                 monkeypatch):
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    monkeypatch.setenv("FAKE_PROXY_HEARTBEAT_AGE", "300")
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "egress interceptor had not reported" in e.value.fault.reason
    assert "commit" not in [s.name for s in ctx.launcher.launched]


# ---------------------------------------------------------------------------
# The process ceiling
# ---------------------------------------------------------------------------


def _node(gpus: int | None, cpus: int | None) -> LocalLauncher:
    """A local launcher with one GPU on a node of *gpus* GPUs and *cpus* CPUs, as its controller
    gives them. The stage takes no GPU, so its evaluations are counted in the node's CPUs; a
    graded server's, which the brief states, in its GPUs."""
    return LocalLauncher(gpus=1, node=NodeSize(gpus=gpus, cpus=cpus))


def test_the_ceiling_is_the_hard_limit_less_k_reserves(sandbox_bins, tmp_path):
    """Every contestant step of the uid draws on its one pool; the ceiling keeps a reserve for
    each sandbox of the stage's size the node holds, and the spawner gives it to every command
    while the sandbox's step keeps its own limit. The site's cap is above that headroom here: the
    commands' node holding its most sandboxes (``tests.support.node``), whose ceiling this host's
    kernel sets on the commands."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    k = node.COMMANDS_EVALUATIONS_MAX
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, k * STAGE.cpus))
    outputs = AgentBuildStage().run(ctx)
    ceiling = _record(ctx, outputs).process_ceiling
    reserve = ctx.plan.settings.run.nproc_reserve
    assert ceiling == ProcessCeiling(hard=node.COMMANDS_HARD, reserve=reserve, evaluations=k,
                                     evaluations_unknown=None, uncapped=node.COMMANDS_HARD - k * reserve,
                                     nproc_max=NPROC_MAX, ceiling=node.COMMANDS_HARD - k * reserve,
                                     floor=k * FLOOR_PER_EVALUATION)
    [build] = _calls(control)
    assert build["nproc"] == [ceiling.ceiling, ceiling.ceiling]
    [spec] = [s for s in ctx.launcher.launched if s.name == "sandbox"]
    assert spec.nproc is None and ["--command-nproc", str(ceiling.ceiling)] == [
        a for a in spec.argv if a in ("--command-nproc", str(ceiling.ceiling))]


@pytest.mark.parametrize("gpus, k", [(1, 8), (2, 4), (8, 1)], ids=["1-gpu", "2-gpu", "whole-node"])
def test_a_gpu_builds_reserve_counts_the_builds_of_its_size_an_8_gpu_node_holds(tmp_path,
                                                                                monkeypatch, gpus, k):
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: 1_000_000)
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 64))
    ctx.stage = ctx.stage.model_copy(update={"shape": STAGE.model_copy(update={"gpus": gpus, "gpu_kind": "H200"})})
    ceiling = ctx.process_ceiling()
    assert (ceiling.evaluations, ceiling.uncapped) == (k, 1_000_000 - k * ctx.plan.settings.run.nproc_reserve)
    assert ceiling.ceiling == NPROC_MAX


def test_an_unreadable_node_size_refuses_the_build_before_any_role(sandbox_bins, tmp_path, monkeypatch):
    # No evaluation count is guessed for it: the commands' reserve and floor would be guesses too.
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: 1_000_000)
    _script(sandbox_bins, window=1_000_000, claude=[{}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, None))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    why = "the node's CPU count is unknown, so how many evaluations of this size it holds is unknown too"
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"no contestant step of stage {ctx.stage.name} can run on this node: {why}"
    assert "sandbox" not in [s.name for s in ctx.launcher.launched]
    [event] = [e for e in _events(ctx) if e["kind"] == "process_ceiling"]
    assert (event["evaluations"], event["evaluations_unknown"], event["ceiling"]) == (None, why, None)


def test_a_node_that_holds_no_graded_server_cannot_state_its_limits(tmp_path):
    # The brief states the limits a graded server is held to on a node like the build's; a node
    # with fewer GPUs than the server takes holds none of it, and no count is guessed.
    ctx = _ctx(tmp_path, _params(), launcher=_node(0, 64))
    with pytest.raises(StageFault) as e:
        ctx.served_ucounts(SERVED.model_copy(update={"gpus": 1, "cpus": 8}))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == ("the limits a graded server is held to on a node like this one cannot be told: "
                                    "the node's GPU count, 0, is under the 1 that one evaluation of this size takes")


class _SandboxLaunched(Exception):
    """The sandbox's step was launched, and the test ends the stage there (``_stop_at_the_sandbox``)."""


def _stop_at_the_sandbox(ctx: StageContext, monkeypatch) -> None:
    """End the stage once its sandbox's step is launched, before any command runs in it: what the
    stage gives the sandbox is read from the launch, under a ceiling a host whose hard limit is
    lower could set on no command (``tests.support.node``)."""
    start = ctx.start_container

    def start_then_stop(name, *args, **kwargs):
        handle = start(name, *args, **kwargs)
        if name == "sandbox":
            raise _SandboxLaunched
        return handle

    monkeypatch.setattr(ctx, "start_container", start_then_stop)


def _sandbox_ceiling(ctx: StageContext) -> tuple[int | None, list[str]]:
    """The launched sandbox step's own ceiling, and the one its exec server gives every command."""
    [spec] = [s for s in ctx.launcher.launched if s.name == "sandbox"]
    return spec.nproc, [spec.argv[i + 1] for i, a in enumerate(spec.argv) if a == "--command-nproc"]


def test_the_sites_cap_holds_the_commands_under_a_hard_limit_past_the_nodes_pids(sandbox_bins, tmp_path,
                                                                                  monkeypatch):
    # A hard limit leaving a headroom over twice the kernel's pid_max; the cap of a site that sets
    # none (site.nproc_max; 131,072 on this node) is the ceiling the spawner gives every command,
    # the brief states it, and the event says where it came from.
    monkeypatch.setattr(stage_module, "hard_limit", lambda: 9_287_489)
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 64))
    _stop_at_the_sandbox(ctx, monkeypatch)
    with pytest.raises(_SandboxLaunched):
        AgentBuildStage().run(ctx)
    reserve = ctx.plan.settings.run.nproc_reserve
    assert ctx.plan.settings.site.nproc_max is None and NPROC_MAX == 131_072
    assert ctx.process_ceiling() == ProcessCeiling(
        hard=9_287_489, reserve=reserve, evaluations=32, evaluations_unknown=None,
        uncapped=9_287_489 - 32 * reserve, nproc_max=NPROC_MAX, ceiling=NPROC_MAX, floor=32 * FLOOR_PER_EVALUATION)
    [event] = [e for e in _events(ctx) if e["kind"] == "process_ceiling"]
    assert (event["uncapped"], event["nproc_max"], event["ceiling"]) == (9_287_489 - 32 * reserve, NPROC_MAX,
                                                                          NPROC_MAX)
    assert _sandbox_ceiling(ctx) == (None, [str(NPROC_MAX)])
    brief = " ".join(ctx.store.read_bytes(ctx.output_key(agent_build.BRIEF_FILE)).decode().split())
    assert f"only while the node holds fewer than {NPROC_MAX:,} of them" in brief


def test_an_unlimited_hard_limit_leaves_the_sites_cap_alone(sandbox_bins, tmp_path, monkeypatch):
    monkeypatch.setattr(stage_module, "hard_limit", lambda: None)
    ctx = _ctx(tmp_path, _params())
    _stop_at_the_sandbox(ctx, monkeypatch)
    with pytest.raises(_SandboxLaunched):
        AgentBuildStage().run(ctx)
    ceiling = ctx.process_ceiling()
    assert (ceiling.hard, ceiling.uncapped, ceiling.ceiling) == (None, None, NPROC_MAX)
    assert _sandbox_ceiling(ctx) == (None, [str(NPROC_MAX)])


def test_the_spawner_gives_the_commands_their_limit_in_each_per_uid_count_the_node_limits(sandbox_bins, tmp_path,
                                                                                         monkeypatch):
    # The node's own limits shared by the 8 builds of its size; the commands get the site's caps
    # within each build's share (hls.substrate.ucounts), on the exec server's command line, and the
    # event records how each was reached.
    node = {"max_user_namespaces": 4_124_579, "max_inotify_instances": 128, "max_inotify_watches": 1_048_576}
    monkeypatch.setattr(stage_module, "node_limits", lambda: node)
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 8),
               site={"sandbox_namespaces": 1024, "sandbox_inotify_instances": 8, "sandbox_inotify_watches": 65_536,
                     "grading_node_namespaces": 2_000_000, "grading_node_inotify_instances": 256,
                     "grading_node_inotify_watches": 524_288})
    outputs = AgentBuildStage().run(ctx)
    k = ctx.process_ceiling().evaluations
    # The brief states the commands' limits in the sandbox's section, this node's share; the
    # serving one states a graded server's share of a grading node as the site declares it, among
    # the 8 servers of its one GPU a node of 8 GPUs holds, whatever this node has: the server is
    # graded on another.
    brief = " ".join(ctx.store.read_bytes(outputs["brief"]).decode().split())
    assert ("Your commands, the agent's own among them, may hold at most 8 inotify instances, 65,536 inotify "
            "watches and 1,024 namespaces of each kind") in brief
    assert (f"It and every process it starts may hold at most {256 // 8:,} inotify instances, {524_288 // 8:,} "
            f"inotify watches and {(2_000_000 - 8 * 16) // 8:,} namespaces of each kind (user, pid, mnt, net, ipc, "
            "uts, cgroup, time) at once") in brief
    [served] = [e for e in _events(ctx) if e["kind"] == "served_ucounts"]
    assert {(c["name"], c["setting"], c["declared"], c["evaluations"], c["role"]) for c in served["limits"]
            if c["name"] == "max_inotify_instances"} == {
        ("max_inotify_instances", "grading_node_inotify_instances", 256, 8, 32)}
    # The build records the figures its brief states, once, for its package to carry to the grading.
    recorded = ServedUcounts.model_validate_json(ctx.store.read_bytes(outputs["served_ucounts"]))
    assert [c.model_dump() for c in recorded.limits] == served["limits"]
    assert recorded.figures()["max_inotify_instances"] == 32 and recorded.figures()["max_inotify_watches"] == 65_536
    [spec] = [s for s in ctx.launcher.launched if s.name == "sandbox"]
    given = [spec.argv[i + 1] for i, a in enumerate(spec.argv) if a == "--ucount"]
    assert given == ["max_user_namespaces=1024", "max_inotify_instances=8", "max_inotify_watches=65536"]
    [event] = [e for e in _events(ctx) if e["kind"] == "ucount_limits"]
    assert [(c["name"], c["evaluations"], c["role"], c["commands"]) for c in event["limits"]] == [
        ("max_user_namespaces", k, (4_124_579 - k * 16) // k, 1024), ("max_inotify_instances", k, 128 // k, 8),
        ("max_inotify_watches", k, 1_048_576 // k, 65_536)]


def test_the_brief_states_a_graded_servers_count_limits_its_substrate_holds_it_to(sandbox_bins, tmp_path,
                                                                                    monkeypatch):
    """The serving section states a graded server's limit of each per-uid count the site's
    substrate holds it to (``Substrate.served_ucounts``): every count on Slurm, whose pasta seal
    writes them, the line unchanged; none on Docker, so its brief states no such line, while the
    sandbox section's commands limits, which the spawner writes on both, stay."""
    from hls.stages import agent_build
    from hls.substrate.docker.substrate import docker
    from hls.substrate.factory import substrate
    from hls.substrate.ucounts import UCOUNTS
    from tests.pipeline.plans import settings
    from tests.substrate.docker.test_substrate import site as docker_site

    node = {"max_user_namespaces": 4_124_579, "max_inotify_instances": 128, "max_inotify_watches": 1_048_576}
    monkeypatch.setattr(stage_module, "node_limits", lambda: node)
    site = {"sandbox_namespaces": 1024, "sandbox_inotify_instances": 8, "sandbox_inotify_watches": 65_536,
            "grading_node_namespaces": 2_000_000, "grading_node_inotify_instances": 256,
            "grading_node_inotify_watches": 524_288}
    served = "It and every process it starts may hold at most"
    commands = "Your commands, the agent's own among them, may hold at most 8 inotify instances"
    briefs = {}
    for name in ("slurm", "docker"):
        _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
        ctx = _ctx(tmp_path / name, _params(), launcher=_node(8, 8), site=site)
        if name == "docker":
            monkeypatch.setattr(agent_build, "substrate", lambda _site: docker(docker_site(tmp_path)))
        outputs = AgentBuildStage().run(ctx)
        briefs[name] = " ".join(ctx.store.read_bytes(outputs["brief"]).decode().split())
    assert substrate(settings(tmp_path).site).served_ucounts == {u.name for u in UCOUNTS}
    assert (f"{served} {256 // 8:,} inotify instances, {524_288 // 8:,} inotify watches and "
            f"{(2_000_000 - 8 * 16) // 8:,} namespaces of each kind (user, pid, mnt, net, ipc, uts, cgroup, time) at "
            "once") in briefs["slurm"]
    assert served not in briefs["docker"] and "inotify instances, 65,536 inotify watches" in briefs["docker"]
    assert commands in briefs["slurm"] and commands in briefs["docker"]


def test_a_node_whose_share_leaves_the_commands_under_a_floor_refuses_the_build_before_any_role(sandbox_bins, tmp_path,
                                                                                               monkeypatch):
    # The builds of a node share its few inotify instances: none could start with room for the
    # agent's CLI, so the stage is infra and not retried, naming why, before the sandbox starts.
    monkeypatch.setattr(stage_module, "node_limits", lambda: {"max_inotify_instances": 2})
    _script(sandbox_bins, window=1_000_000, claude=[{}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 8))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "max_inotify_instances of 2, shared by" in e.value.fault.reason
    assert "sandbox" not in [s.name for s in ctx.launcher.launched]


def test_a_node_whose_limits_cannot_be_read_refuses_the_build(sandbox_bins, tmp_path, monkeypatch):
    def unreadable():
        raise OSError(13, "Permission denied", "/proc/sys/user/max_user_namespaces")

    monkeypatch.setattr(stage_module, "node_limits", unreadable)
    _script(sandbox_bins, window=1_000_000, claude=[{}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 2))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "per-uid limits could not be read" in e.value.fault.reason


def test_a_sites_cap_under_the_floor_of_the_nodes_builds_is_refused_before_any_role(sandbox_bins, tmp_path):
    # A plan checks the cap against one evaluation; a node that holds more builds of the stage's
    # size is refused when the stage starts too, naming the value to raise it to.
    _script(sandbox_bins, window=1_000_000, claude=[{}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 64), site={"nproc_max": 32 * FLOOR_PER_EVALUATION - 1})
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == (f"no contestant step of stage {ctx.stage.name} can run on this node: "
                                    "site.nproc_max of 131071 tasks is under the "
                                    "floor of 131072, 4096 for each of the 32 evaluations' servers the node holds; "
                                    "raise site.nproc_max to at least 131072")
    assert ctx.launcher.launched == []


#: The builds of the stage's size the local launcher's own node holds: the stage takes no GPU.
NODE_BUILDS = NODE.cpus // STAGE.cpus


@pytest.mark.parametrize("hard", [900, NODE_BUILDS * RESERVE + NODE_BUILDS * FLOOR_PER_EVALUATION - 1])
def test_a_ceiling_under_k_times_the_floor_is_refused_before_any_role(sandbox_bins, tmp_path, monkeypatch, hard):
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: hard)
    _script(sandbox_bins, window=1_000_000, claude=[{}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(NODE.gpus, NODE.cpus))
    assert ctx.plan.settings.run.nproc_reserve == RESERVE
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert f"under the floor of {NODE_BUILDS * FLOOR_PER_EVALUATION}," in e.value.fault.reason
    assert ctx.launcher.launched == []


def test_a_uid_already_at_the_ceiling_fails_the_sandbox_start_as_infra(sandbox_bins, tmp_path,
                                                                       monkeypatch):
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: 3_000_000)
    _script(sandbox_bins, window=1_000_000, claude=[{}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 2))
    monkeypatch.setattr(stage_module, "uid_tasks", lambda uid: 3_000_000)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "at or above its commands' process ceiling" in e.value.fault.reason
    assert "sandbox" not in [s.name for s in ctx.launcher.launched]


def test_a_sandbox_that_ends_while_others_fill_the_uids_pool_was_starved_and_is_infra(
        sandbox_bins, tmp_path, monkeypatch):
    """Counted once the sandbox's step is gone, the uid's other tasks alone fill its commands'
    pool: its end may be theirs, and is infra, retried, saying so."""
    control = sandbox_bins
    # The commands' node (tests.support.node): the hard limit's headroom, under the site's cap, is
    # the pool.
    ceiling = node.COMMANDS_CEILING
    pool = {"others": 500}
    monkeypatch.setattr(stage_module, "uid_tasks", lambda uid: pool["others"])
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "kill_server": True, "hang": True}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 2))
    start = ctx.start_container

    def start_then_fill(name, *args, **kwargs):
        handle = start(name, *args, **kwargs)
        if name == "sandbox":
            pool["others"] = int(0.96 * ceiling)
        return handle

    monkeypatch.setattr(ctx, "start_container", start_then_fill)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "starved" in fault.reason
    assert fault.evidence["sandbox.others_tasks"] == str(int(0.96 * ceiling))


def _starving(ctx: StageContext, monkeypatch, pool: dict[str, int | None], fill: int | None) -> None:
    """The uid's other tasks on the node count *fill* from the moment the sandbox has started;
    None makes the count fail from then on (``_count``)."""
    start = ctx.start_container

    def start_then_fill(name, *args, **kwargs):
        handle = start(name, *args, **kwargs)
        if name == "sandbox":
            pool["others"] = fill
        return handle

    monkeypatch.setattr(ctx, "start_container", start_then_fill)


@pytest.mark.parametrize("ending", ["agent_nonzero_exit", "harbor_command_failed"])
def test_an_agent_end_while_others_fill_the_uids_pool_was_starved_and_is_infra(
        sandbox_bins, tmp_path, monkeypatch, ending):
    """An end of the agent's part that is a failure may be its forks failing for what others
    started: counted once the sandbox's step is gone, the uid's other tasks alone fill its
    commands' pool, so the end is infra, retried, naming the agent's end."""
    control = sandbox_bins
    # The commands' node (tests.support.node): the hard limit's headroom, under the site's cap, is
    # the pool.
    ceiling = node.COMMANDS_CEILING
    pool = {"others": 500}
    monkeypatch.setattr(stage_module, "uid_tasks", lambda uid: pool["others"])
    monkeypatch.setattr(agent_build._Build, "ending", lambda self, how, error: ending)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 2000}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 2))
    _starving(ctx, monkeypatch, pool, int(0.96 * ceiling))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "starved" in fault.reason
    assert fault.evidence["agent_ending"] == ending
    assert fault.evidence["sandbox.others_tasks"] == str(int(0.96 * ceiling))


def _count(pool: dict[str, int | None]):
    """A count of the uid's tasks that reads *pool*'s ``others``, and fails, as listing ``/proc``
    out of descriptors would, while they are None."""
    def count(uid):
        if pool["others"] is None:
            raise OSError(errno.EMFILE, "Too many open files", "/proc")
        return pool["others"]

    return count


def test_a_sandbox_that_ends_while_the_uids_tasks_cannot_be_counted_is_infra_saying_so(
        sandbox_bins, tmp_path, monkeypatch):
    """Counted before the sandbox started, the uid's tasks could not be counted once its step was
    gone: its end may be the others', and is infra, retried, naming the failed count."""
    pool: dict[str, int | None] = {"others": 500}
    monkeypatch.setattr(stage_module, "uid_tasks", _count(pool))
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "kill_server": True, "hang": True}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 2))
    _starving(ctx, monkeypatch, pool, None)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "could not be counted" in fault.reason
    assert fault.evidence["sandbox.others_tasks"] == "uncounted"


@pytest.mark.parametrize("ending", ["agent_nonzero_exit", "harbor_command_failed"])
def test_an_agent_end_that_is_a_failure_while_the_uids_tasks_cannot_be_counted_is_infra(
        sandbox_bins, tmp_path, monkeypatch, ending):
    """A failing end of the agent's part may be its forks failing for what others started, and
    with the uid's tasks uncounted once the sandbox's step is gone nothing rules that out: infra,
    retried, naming the agent's end."""
    pool: dict[str, int | None] = {"others": 500}
    monkeypatch.setattr(stage_module, "uid_tasks", _count(pool))
    monkeypatch.setattr(agent_build._Build, "ending", lambda self, how, error: ending)
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 2000}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 2))
    _starving(ctx, monkeypatch, pool, None)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "could not be counted" in fault.reason
    assert (fault.evidence["agent_ending"], fault.evidence["sandbox.others_tasks"]) == (ending, "uncounted")


@pytest.mark.parametrize("ending", ["hard_limit", "agent_ended"])
def test_a_normal_end_is_graded_while_the_uids_tasks_cannot_be_counted(sandbox_bins, tmp_path, monkeypatch, ending):
    """The build's normal ends are no failure starvation could have caused, so an uncounted pool
    leaves them graded."""
    pool: dict[str, int | None] = {"others": 500}
    monkeypatch.setattr(stage_module, "uid_tasks", _count(pool))
    monkeypatch.setattr(agent_build._Build, "ending", lambda self, how, error: ending)
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 2000}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 2))
    _starving(ctx, monkeypatch, pool, None)
    outputs = AgentBuildStage().run(ctx)
    assert _record(ctx, outputs).ending == ending and "source_bundle" in outputs


@pytest.mark.parametrize("ending", ["agent_nonzero_exit", "harbor_command_failed"])
def test_counted_per_namespace_the_uid_at_the_commands_ceiling_neither_refuses_the_sandbox_nor_starves_the_agent(
        sandbox_bins, tmp_path, monkeypatch, ending):
    """On a kernel that counts the commands' own tasks against their ceiling (el9),
    another build's commands holding theirs leave this build's every fork: the sandbox starts
    though the uid already holds the ceiling, a failing end of the agent's is graded, never a free
    rerun, and the brief says what counts."""
    ceiling = node.COMMANDS_CEILING
    pool = {"others": ceiling}
    monkeypatch.setattr(stage_module, "uid_tasks", lambda uid: pool["others"])
    monkeypatch.setattr(agent_build._Build, "ending", lambda self, how, error: ending)
    answering({"nproc-accounting": "echo forks=511"}, monkeypatch)
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 2000}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 2), site={"nproc_accounting": "user_namespace"})
    _starving(ctx, monkeypatch, pool, int(0.96 * ceiling))
    outputs = AgentBuildStage().run(ctx)
    assert _record(ctx, outputs).ending == ending and "source_bundle" in outputs
    brief = ctx.store.read_bytes(outputs["brief"]).decode()
    assert f"while they hold fewer than {ceiling:,} of them together" in brief


@pytest.mark.parametrize("ending, share", [("hard_limit", 0.96), ("agent_ended", 0.96),
                                           ("agent_nonzero_exit", 0.94)])
def test_a_normal_end_or_a_pool_others_leave_room_in_is_graded(sandbox_bins, tmp_path,
                                                               monkeypatch, ending, share):
    """The hard limit and the agent's own end, finished or with its budget spent, are the build's
    normal ends: a starved pool may have slowed the agent, but it ended no failure, so they are
    graded whatever the others held. Others under ``POOL_FULL`` of the ceiling left the agent's
    commands room, so a failure then is graded too."""
    control = sandbox_bins
    # The commands' node (tests.support.node): the hard limit's headroom, under the site's cap, is
    # the pool.
    ceiling = node.COMMANDS_CEILING
    pool = {"others": 500}
    monkeypatch.setattr(stage_module, "uid_tasks", lambda uid: pool["others"])
    monkeypatch.setattr(agent_build._Build, "ending", lambda self, how, error: ending)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 2000}])
    ctx = _ctx(tmp_path, _params(), launcher=_node(8, 2))
    _starving(ctx, monkeypatch, pool, int(share * ceiling))
    outputs = AgentBuildStage().run(ctx)
    assert _record(ctx, outputs).ending == ending and "source_bundle" in outputs


# ---------------------------------------------------------------------------
# The disk cap
# ---------------------------------------------------------------------------


@pytest.fixture
def disk_watch(monkeypatch):
    """The runner's memory watch, walking each contestant's writable directories every tenth of
    a second against its cap, scaled to an eighth of a MiB per GiB so a test can pass it."""
    watch = MemoryWatch(enforce=False, interval_sec=0.05, disk_interval_sec=0.1)
    watch_disk = watch.watch_disk

    def capped(step, paths, limit_bytes, kill, **kw):
        watch_disk(step, paths, limit_bytes // (8 * 1024), kill, **kw)

    monkeypatch.setattr(watch, "watch_disk", capped)
    watch.start()
    yield watch
    watch.stop()


@pytest.mark.parametrize("step, bound, words", [
    ({"deep": 8}, ("DISK_WALK_MAX_DEPTH", 5), "a directory more than 5 levels deep in its writable directories"),
    ({"files": 600}, ("DISK_WALK_MAX_ENTRIES", 300), "more than 300 entries in its writable directories"),
], ids=["depth", "entries"])
def test_a_sandbox_tree_past_a_walk_bound_ends_the_build_and_its_record_names_the_bound(
        sandbox_bins, tmp_path, monkeypatch, disk_watch, step, bound, words):
    """A writable tree deeper, or with more entries, than the disk walk measures is past the cap,
    the contestant's: the build ends ``disk_limit`` and is graded on its last commit, its record
    naming the bound."""
    control = sandbox_bins
    monkeypatch.setattr(memory_watch, *bound)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", **step, "hang": True}])
    ctx = _ctx(tmp_path, _params(), watch=disk_watch)
    outputs = AgentBuildStage().run(ctx)
    record = _record(ctx, outputs)
    assert record.ending == "disk_limit" and words in record.fault.reason
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)


def test_a_sandbox_directory_kept_unsearchable_ends_the_build_as_disk_limit(sandbox_bins, tmp_path, disk_watch):
    """A contestant that keeps taking its directory's permissions away, which the walk gives back,
    cannot switch its cap off: a refused walk is past the cap."""
    if os.geteuid() == 0:
        pytest.skip("root reads whatever the mode")
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "chmodrace": "hidden", "hang": True}])
    ctx = _ctx(tmp_path, _params(), watch=disk_watch)
    try:
        outputs = AgentBuildStage().run(ctx)
    finally:
        racer = control / "racer.pid"
        if racer.exists():
            try:
                os.kill(int(racer.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    record = _record(ctx, outputs)
    assert record.ending == "disk_limit" and "refused permission" in record.fault.reason


def test_the_commit_and_transcript_roles_files_exist_before_the_sandbox_starts(sandbox_bins, tmp_path, monkeypatch):
    """Every node-local file the harness writes once the sandbox runs is made before it starts,
    so a contestant that fills the node's disk leaves the commit point and the transcript role
    nothing to make there, their log cappers' error files among them; what they write, their
    logs among it, goes to the store."""
    _script(sandbox_bins, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    seen: dict[str, object] = {}
    start = ctx.start_container

    def look_then_start(name, *args, **kw):
        if name == "sandbox":
            roles = ctx.scratch / "roles"
            capped = {role: all(log_cap.errors_file(roles / role, suffix).is_file() for suffix in (".out", ".err"))
                      for role in ("commit", "commit-2", "commit-3", "transcript")}
            for role in ("commit", "commit-2", "commit-3"):
                seen[role] = (sorted(p.name for p in (roles / role / "wd").iterdir()),
                              (roles / role / "outputs.json").stat().st_size,
                              (roles / role / "container.env").exists(),
                              (ctx.attempt_dir / "commit-points" / role / "tmp").is_dir(), capped[role])
            seen["transcript"] = ((roles / "transcript" / "role.env").exists(), capped["transcript"],
                                  (roles / "transcript" / "outputs.json").stat().st_size)
        return start(name, *args, **kw)

    monkeypatch.setattr(ctx, "start_container", look_then_start)
    AgentBuildStage().run(ctx)
    for role in ("commit", "commit-2", "commit-3"):
        assert seen[role] == (["home", "tmp", "var_tmp"], agent_build.OUTPUTS_REPORT_BYTES, True, True, True), role
    assert seen["transcript"] == (True, True, agent_build.OUTPUTS_REPORT_BYTES)
    [transcript] = [s for s in ctx.launcher.launched if s.name == "transcript"]
    assert transcript.log_dir == ctx.attempt_dir / "logs"


def test_a_disk_breach_frees_the_sandboxs_writes_before_the_commit_point_runs(sandbox_bins, tmp_path, monkeypatch,
                                                                              disk_watch):
    """Once a sandbox that wrote past its cap is gone, everything it wrote goes before the commit
    point or the transcript role writes anything, but the repository, which the commit point
    bundles, and the logs, which the transcript role copies."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "write": 3 << 20, "hang": True}])
    ctx = _ctx(tmp_path, _params(), watch=disk_watch)
    layout = SandboxLayout(ctx.scratch / "roles" / "sandbox")
    seen: dict[str, object] = {}
    start = ctx.start_container

    def look_then_start(name, *args, **kw):
        if name == "commit":
            seen["workspace"] = sorted(p.name for p in layout.workspace.iterdir())
            seen["gone"] = [p.exists() for p in (layout.workdir, layout.installed, layout.control)]
            seen["logs"] = layout.agent_logs.is_dir()
        return start(name, *args, **kw)

    monkeypatch.setattr(ctx, "start_container", look_then_start)
    outputs = AgentBuildStage().run(ctx)
    assert seen == {"workspace": [".git"], "gone": [False, False, False], "logs": True}
    assert _record(ctx, outputs).ending == "disk_limit" and "source_bundle" in outputs


@pytest.mark.parametrize("where", [".git", "logs"])
def test_a_scratch_filled_through_git_or_logs_is_graded_or_over_limit_never_infra(
        sandbox_bins, tmp_path, disk_watch, where):
    """What the breach leaves, the repository and the logs, the commit point and the transcript
    copy read within their own bounds: the build is graded on its last commit, never infra."""
    control = sandbox_bins
    ctx = _ctx(tmp_path, _params(), watch=disk_watch)
    layout = SandboxLayout(ctx.scratch / "roles" / "sandbox")
    target = layout.workspace / ".git" / "fill.bin" if where == ".git" else layout.agent_logs / "fill.bin"
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "fill": {str(target): 3 << 20}, "hang": True}])
    outputs = AgentBuildStage().run(ctx)
    assert _record(ctx, outputs).ending == "disk_limit"
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)


def test_a_sandbox_that_fills_its_writable_disk_to_the_brim_still_has_its_end_recorded_and_judged(
        sandbox_bins, tmp_path, disk_watch):
    """The sandbox fills the node's disk, where its seal's files are; its seal made its ending
    records and its interceptor's status at full size before the payload started, so nothing the
    seal writes needs new space. The build is judged on the sandbox's own end, past its disk cap,
    and graded on its last commit, never infra. The seal's directory taking no new file stands
    for the full disk."""
    from hls.egress.proxy import STATUS_FILE_BYTES

    control = sandbox_bins
    ctx = _ctx(tmp_path, _params(), watch=disk_watch)
    seal = ctx.scratch / "roles" / "sandbox" / "seal"
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "write": 3 << 20, "chmod": {str(seal): 0o500},
                                                "hang": True}])
    try:
        outputs = AgentBuildStage().run(ctx)
        sizes = {p.name: p.stat().st_size for p in seal.iterdir() if p.name != "port"}
    finally:
        if seal.exists():
            seal.chmod(0o700)
    record = _record(ctx, outputs)
    assert (record.ending, record.fault.fault_class) == ("disk_limit", "contestant")
    assert _bundle_main(ctx.store.path(outputs["source_bundle"]), tmp_path) == _workspace_main(ctx)
    assert sizes["status.json"] == STATUS_FILE_BYTES and sizes["ended.pending"] == 512


def test_the_commit_point_starts_only_after_the_sandboxs_processes_are_gone(sandbox_bins, tmp_path, monkeypatch):
    """Files the contestant held open unlinked, or made with ``O_TMPFILE``, free their space only
    once every process of the sandbox is gone: the commit point waits for the step's cgroup to
    list none."""
    control = sandbox_bins
    cgroups = tmp_path / "cg"
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params(), launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    lingering = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    procs = cgroups / "cg-sandbox" / "cgroup.procs"
    gone: list[float] = []
    stop = ctx.stop_role

    def stop_leaving_one(name):
        outcome = stop(name)
        if name == "sandbox":
            procs.write_text(f"{lingering.pid}\n")

            def end_it():
                lingering.kill()
                lingering.wait()
                procs.write_text("")
                gone.append(time.time())

            threading.Timer(1.5, end_it).start()
        return outcome

    monkeypatch.setattr(ctx, "stop_role", stop_leaving_one)
    try:
        AgentBuildStage().run(ctx)
    finally:
        lingering.kill()
    [started] = [e for e in _events(ctx) if e["kind"] == "role_started" and e["role"] == "commit"]
    assert gone and started["at"] >= gone[0]


# ---------------------------------------------------------------------------
# A grading of what an agent built
# ---------------------------------------------------------------------------


def test_the_route_secrets_of_the_arm_that_built_a_submission_are_named(tmp_path):
    """A plan that evaluates a package an agent built must count the building arm's route
    secrets contestant-facing: the agent may have baked them into its
    server. The stage names them from its parameters, for the grading's planner."""
    gpt, opus = (AgentBuildParams.model_validate(p).arm for p in (_params(GPT), _params()))
    assert agent_build.arm_route_secrets(gpt) == frozenset({"openai_key", "openai_url"})
    assert agent_build.arm_route_secrets(opus) == frozenset({"anthropic_key"})


# ---------------------------------------------------------------------------
# What the agent writes never fails the harness that reads it
# ---------------------------------------------------------------------------


def _agent_logs(ctx: StageContext) -> Path:
    return SandboxLayout(ctx.scratch / "roles" / "sandbox").agent_logs


def test_a_transcript_line_nested_past_the_decoders_depth_is_no_evidence_and_the_build_is_graded(
        sandbox_bins, tmp_path):
    """The agent writes a line nested 50,000 deep into a newer session transcript: the heartbeat's
    health check reads it as unreadable, and the build runs to its end and is graded."""
    control = sandbox_bins
    ctx = _ctx(tmp_path, _params())
    planted = _agent_logs(ctx) / "sessions" / "projects" / "-workspace" / "zz-planted.jsonl"
    _script(control, window=1_000_000,
            claude=[{"commit": "v1\n", "touch": {str(planted): "[" * 50_000 + "\n"}, "advance": 1090,
                     "hang": True}])
    outputs = AgentBuildStage().run(ctx)
    assert _record(ctx, outputs).ending == "hard_limit" and "source_bundle" in outputs
    assert [e for e in _events(ctx) if e["kind"] == "build.heartbeat"]


def test_a_result_line_nested_past_the_decoders_depth_on_claudes_stream_is_no_evidence(
        sandbox_bins, tmp_path):
    """A line holding "result" and a value nested 50,000 deep, written to the CLI's stream: harbor
    reads it as unreadable when it looks at the CLI's exit, and the build is graded."""
    control = sandbox_bins
    ctx = _ctx(tmp_path, _params())
    _script(control, window=1_000_000,
            claude=[{"commit": "v1\n", "stderr": '{"type": "result", "x": ' + "[" * 50_000},
                    {"commit": "v2\n", "advance": 2000}])
    outputs = AgentBuildStage().run(ctx)
    assert _record(ctx, outputs).ending == "agent_ended" and "source_bundle" in outputs


@pytest.mark.parametrize("name", ["INDEX.json", ".tmp-INDEX.json"])
def test_a_directory_named_as_the_transcripts_index_is_copied_and_the_build_graded(sandbox_bins, tmp_path, name):
    """The transcript's index and its temporary file live outside the copied tree, so no name the
    agent gives an entry of its logs can meet them."""
    control = sandbox_bins
    ctx = _ctx(tmp_path, _params())
    _script(control, window=1_000_000,
            claude=[{"commit": "v1\n", "mkdirs": [str(_agent_logs(ctx) / name / "x")], "advance": 2000}])
    outputs = AgentBuildStage().run(ctx)
    assert _record(ctx, outputs).ending == "agent_ended" and "source_bundle" in outputs
    index = json.loads(ctx.store.read_bytes(outputs["transcript"]))
    assert "claude-code.txt" in {f["path"] for f in index["files"]}


def test_a_planted_timestamp_gives_bounded_heartbeat_events(sandbox_bins, tmp_path):
    """Turns whose timestamps the agent made huge: an 8 MB line, past the health check's line
    bound, is unreadable, and a 0.5 MB one is a turn whose time is no timestamp. The heartbeat
    reports one turn and no last turn's time, never the agent's string."""
    control = sandbox_bins
    ctx = _ctx(tmp_path, _params())
    planted = _agent_logs(ctx) / "sessions" / "projects" / "-workspace" / "zz-planted.jsonl"

    def turn(stamp_chars: int, mid: str) -> str:
        return json.dumps({"type": "assistant", "timestamp": "\u00e9" * stamp_chars,
                           "message": {"id": mid, "content": [{"type": "text", "text": "hi"}]}})

    lines = turn(1_300_000, "big") + "\n" + turn(80_000, "m") + "\n"
    assert len(lines) > 8_000_000
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "touch": {str(planted): lines},
                                                "sleep": 2.5, "advance": 900}])
    AgentBuildStage().run(ctx)
    beats = [e for e in _events(ctx) if e["kind"] == "build.heartbeat"]
    assert beats and all(len(json.dumps(e)) < 1000 for e in beats)
    assert beats[-1]["turns"] == 1 and beats[-1]["last_turn_at"] is None


def test_whatever_the_health_check_raises_is_an_error_event_and_the_build_goes_on(
        sandbox_bins, tmp_path, monkeypatch):
    """The health check reads files the agent wrote: an exception of any kind from it is an
    ``error`` event, and the build runs to its end."""
    control = sandbox_bins

    def refuse(logs, harness):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(agent_build, "check_health", refuse)
    ctx = _ctx(tmp_path, _params())
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "sleep": 2.5, "advance": 900}])
    outputs = AgentBuildStage().run(ctx)
    assert _record(ctx, outputs).ending == "agent_ended" and "source_bundle" in outputs
    assert any(e["kind"] == "error" and e["where"] == "health" and "RecursionError" in e["msg"]
               for e in _events(ctx))



# ---------------------------------------------------------------------------
# A node's disk the agent left full
# ---------------------------------------------------------------------------

#: While it holds a directory, this process may make nothing under it: the node's disk there is
#: full (``tests.drive.test_stage`` has the same for the drive). Every attempt is kept.
_FULL: list[Path] = []
_FULL_ATTEMPTS: list[str] = []
_WRITES = os.O_WRONLY | os.O_RDWR | os.O_CREAT
#: The programs a role starts load ``sitecustomize`` from here, which refuses them the same.
FAKE_SITE = Path(__file__).resolve().parents[1] / "drive" / "fake_site"


def _refuse_on_a_full_node(event: str, args: tuple) -> None:
    if not _FULL:
        return
    if event == "open" and len(args) == 3 and isinstance(args[2], int) and args[2] & _WRITES:
        path, flags = args[0], args[2]
        if isinstance(path, int):
            return
        # A write in place into a file already there takes no new space.
        if not flags & (os.O_TRUNC | os.O_APPEND) and not (flags & os.O_CREAT and not os.path.lexists(path)):
            return
        targets = [path]
    elif event == "os.mkdir":
        # A name already there fails with EEXIST before any space is asked for.
        targets = [] if isinstance(args[0], int) or os.path.lexists(args[0]) else [args[0]]
    elif event in ("os.symlink", "os.link", "os.rename", "os.replace"):
        targets = list(args[:2])
    else:
        return
    for target in targets:
        if isinstance(target, (str, bytes, os.PathLike)) and os.path.isabs(os.fsdecode(target)):
            if Path(os.fsdecode(target)).resolve().is_relative_to(_FULL[0]):
                _FULL_ATTEMPTS.append(f"{event} {os.fsdecode(target)}")
                raise OSError(errno.ENOSPC, "No space left on device", os.fsdecode(target))


sys.addaudithook(_refuse_on_a_full_node)


def test_a_node_disk_the_agent_left_full_still_collects_its_transcript_and_is_graded(
        sandbox_bins, tmp_path, monkeypatch):
    """The agent's part ends with the node's disk full, as one ``fallocate`` of its free bytes
    leaves it. Nothing the stage starts after the sandbox needs a new block of that disk, in this
    process or in a role's: the commit point's and the transcript role's files were made before
    the sandbox started, and their logs and reports go to the store. The bundle and the transcript
    are collected, and the build is graded."""
    control = sandbox_bins
    launcher = LocalLauncher(gpus=1)
    ctx = _ctx(tmp_path, _params(), launcher=launcher)
    full, log = ctx.scratch.resolve(), tmp_path / "full-node.log"
    gone = agent_build._Build.wait_gone

    def gone_then_full(self, *args, **kwargs):
        gone(self, *args, **kwargs)
        _FULL.append(full)
        launcher.task_prolog.update(FAKE_FULL_NODE=str(full), FAKE_FULL_NODE_LOG=str(log),
                                    PYTHONPATH=os.pathsep.join([str(FAKE_SITE), *(p for p in sys.path if p)]))

    monkeypatch.setattr(agent_build._Build, "wait_gone", gone_then_full)
    _script(control, window=1_000_000, claude=[{"commit": "v1\n", "advance": 2000}])
    _FULL_ATTEMPTS.clear()
    try:
        outputs = AgentBuildStage().run(ctx)
    finally:
        _FULL.clear()
    assert _record(ctx, outputs).ending == "agent_ended" and "source_bundle" in outputs
    index = json.loads(ctx.store.read_bytes(outputs["transcript"]))
    assert "claude-code.txt" in {f["path"] for f in index["files"]}
    assert _FULL_ATTEMPTS == []
    assert not log.exists(), log.read_text()
    # The transcript role's logs and its report are in the store, scrubbed.
    assert (ctx.attempt_dir / "logs" / "transcript.report.json").is_file()


# ---------------------------------------------------------------------------
# The seal probed before every build's agent
# ---------------------------------------------------------------------------

#: The probe's real inbound check, which ``fake_seal`` replaces with a working seal's answer.
REAL_INBOUND_CHECK = probe_module.inbound_check
#: The seal probe's checks, in order: its sealed commands, with the interceptor's record of the
#: refusal read host side after the refused one, then the accounting check and the inbound knocks.
SEAL_CHECKS = ["route-reachable", "unlisted-refused", "refusal-recorded", "icmp-held", "index-reachable",
               "nproc-accounting", "inbound-closed"]


def test_every_build_probes_the_seal_before_the_preflight(sandbox_bins, tmp_path):
    """After the sandbox answers and before the preflight, the seal's checks run as contestant
    commands, without the route's variables; build.json and the ``build.seal_probe`` event
    record each check and the seconds the probe took."""
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    record = _record(ctx, AgentBuildStage().run(ctx))
    seal = record.seal_probe
    assert [c.name for c in seal.checks] == SEAL_CHECKS and seal.passed and all(c.ok for c in seal.checks)
    assert 0 <= seal.sec and seal.accounting == "uid"
    kinds = [e["kind"] for e in _events(ctx)]
    assert kinds.index("build.seal_probe") < kinds.index("build.preflight") < kinds.index("build.started")
    [event] = [e for e in _events(ctx) if e["kind"] == "build.seal_probe"]
    assert (event["passed"], event["sec"], event["accounting"]) == (True, seal.sec, "uid")
    assert [c["name"] for c in event["checks"]] == SEAL_CHECKS
    # Its commands came before the preflight's first, with no variable of the arm's route.
    sent = MappedEnvironment.SENT
    commands = [c for c, _ in sent]
    seal_at = [commands.index(SEAL_ANSWERS[name]) for name in ("route-reachable", "unlisted-refused",
                                                               "icmp-held", "index-reachable")]
    preflight_at = next(i for i, c in enumerate(commands) if ".claude.json" in c)
    assert seal_at == sorted(seal_at) and seal_at[-1] < preflight_at
    assert all(sent[i][1] == {} for i in seal_at)
    # The stage gives each seal command the seal probe's own check time.
    timeouts = dict(MappedEnvironment.TIMEOUTS)
    assert all(timeouts[SEAL_ANSWERS[name]] == seal_probe.CHECK_SEC
               for name in ("route-reachable", "unlisted-refused", "icmp-held", "index-reachable"))


def test_the_seal_probe_holds_two_ports_in_the_sandbox_and_a_forwarding_seal_fails_it(sandbox_bins, tmp_path,
                                                                                       monkeypatch):
    """The real inbound check: a sandbox command holds a TCP and a UDP port, none the seal's
    namespace holds already, and the stage knocks on them from every address of the node. This
    test's sandbox has no seal, so its ports are the node's own and the knocks reach them, as
    through a seal that forwards them: the probe fails, infra, before the preflight, naming the
    seal. Its files are gone after."""
    control = sandbox_bins
    left: list[list[str]] = []

    async def inbound(exec_, inputs, **kw):
        check = await REAL_INBOUND_CHECK(exec_, inputs, **kw)
        left.append(sorted(p.name for p in kw["scratch"].iterdir()))
        return check

    monkeypatch.setattr(probe_module, "inbound_check", inbound)
    picked: list[tuple[frozenset[int], int]] = []
    pick = seal_probe.inbound_ports
    monkeypatch.setattr(seal_probe, "inbound_ports", lambda *, avoid, floor: picked.append(
        (frozenset(avoid), floor)) or pick(avoid=avoid, floor=floor))
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "v1\n"}])
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        run_stage(AgentBuildStage(), ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "inbound-closed" in fault.reason
    assert "the seal forwarded tcp port" in fault.reason and "the probe saw it inside the seal" in fault.reason
    assert picked == [(frozenset({EGRESS_TRANSPARENT_PORT, EGRESS_INDEX_PORT}), seal_probe.NAMESPACE_PORT_FLOOR)]
    assert _calls(control, preflight=True) == [] and _calls(control) == []
    assert left == [[]]


def test_the_seal_probe_passes_when_every_knock_finds_its_port_closed(sandbox_bins, tmp_path, monkeypatch):
    """The same check with the knocks answered as through a working seal (every port refused):
    the hold command saw nothing, and the probe passes."""
    control = sandbox_bins
    monkeypatch.setattr(probe_module, "inbound_check", REAL_INBOUND_CHECK)
    monkeypatch.setattr(seal_probe, "knock_inbound", lambda addresses, tcp, udp: tuple(
        seal_probe.Knock(a, proto, port, "refused") for proto, port in (("tcp", tcp), ("udp", udp))
        for a in addresses))
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "v1\n", "advance": 900}])
    ctx = _ctx(tmp_path, _params())
    record = _record(ctx, AgentBuildStage().run(ctx))
    [inbound] = [c for c in record.seal_probe.checks if c.name == "inbound-closed"]
    assert inbound.ok and "nothing reached" in inbound.observed and "refused from 127.0.0.1" in inbound.observed


@pytest.mark.parametrize("site, seen, measured", [("uid", "forks=511", "user_namespace"),
                                                   ("uid", "forks=256", "user_namespace"),
                                                   ("user_namespace", "forks=0", "uid")])
def test_a_kernel_that_clearly_counts_otherwise_than_the_site_says_is_infra_not_retried_before_the_preflight(
        sandbox_bins, tmp_path, monkeypatch, site, seen, measured):
    """The site says one level counts, and the kernel clearly counts the other: no fork at all
    where the site says the commands' own count, or half the limit forked where it says every
    task of the uid does. Every build of the site would be judged by the wrong level, so the build
    stops before its preflight and its clock, infra and not retried, naming both."""
    answering({"nproc-accounting": f"echo {seen}"}, monkeypatch)
    _script(sandbox_bins, window=1_000_000, preflight={}, claude=[{"commit": "v1\n"}])
    ctx = _ctx(tmp_path, _params(), site={"nproc_accounting": site})
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == (f"site.nproc_accounting does not match this node's kernel: {seen} held=600 "
                                    f"limit=512 accounting={measured}, expected accounting={site}")
    kinds = [ev["kind"] for ev in _events(ctx)]
    assert "build.seal_probe" in kinds and "build.preflight" not in kinds and "build.started" not in kinds
    [event] = [ev for ev in _events(ctx) if ev["kind"] == "build.seal_probe"]
    assert event["accounting"] == measured


@pytest.mark.parametrize("site, seen", [("uid", "forks=1"), ("user_namespace", "forks=255"),
                                        ("uid", "error=OSError:[Errno 12] Cannot allocate memory"),
                                        ("user_namespace", "error=no user namespace was made")])
def test_an_unclear_or_failed_accounting_reading_is_infra_retried_before_the_preflight(
        sandbox_bins, tmp_path, monkeypatch, site, seen):
    """A count between no fork and half the limit says neither accounting, and a reading that
    failed, a fork refused with another errno such as a transient ENOMEM among them, says
    nothing of the kernel: neither shows the site's setting wrong, so the build stops before its
    preflight and its clock as infra that a retry, on this node or another, may get past."""
    answering({"nproc-accounting": f"echo '{seen}'"}, monkeypatch)
    _script(sandbox_bins, window=1_000_000, preflight={}, claude=[{"commit": "v1\n"}])
    ctx = _ctx(tmp_path, _params(), site={"nproc_accounting": site})
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert e.value.fault.reason == (f"the nproc-accounting check gave no clear reading of how this node's kernel "
                                    f"counts the commands' processes: {seen} held=600 limit=512 accounting=unclear, "
                                    f"expected accounting={site}")
    kinds = [ev["kind"] for ev in _events(ctx)]
    assert "build.seal_probe" in kinds and "build.preflight" not in kinds and "build.started" not in kinds
    [event] = [ev for ev in _events(ctx) if ev["kind"] == "build.seal_probe"]
    assert event["accounting"] is None


@pytest.mark.parametrize("plant, failed", [("answered", "unlisted-refused"), ("unrecorded", "refusal-recorded"),
                                           ("unreachable", "route-reachable"), ("no-index", "index-reachable"),
                                           ("icmp-left", "icmp-held")])
def test_a_failed_seal_probe_is_infra_retried_and_runs_no_preflight_or_agent(sandbox_bins, tmp_path, monkeypatch,
                                                                             plant, failed):
    """A seal that let a host its policy refuses through, whose interceptor recorded no refusal by
    its policy, that reaches neither the route nor the index, or that let an ICMP echo out fails
    the probe: infra, retried, since the agent never started. The fault and the event carry the
    seconds it took."""
    control = sandbox_bins
    if plant == "unrecorded":
        monkeypatch.setenv("FAKE_PROXY_POLICY_REFUSED", "[]")
    else:
        answer = {"answered": "echo answered", "unreachable": "echo fail ConnectionRefusedError; exit 1",
                  "no-index": "exit 1", "icmp-left": ICMP_LEFT}[plant]
        answering({failed: answer}, monkeypatch)
    _script(control, window=1_000_000, preflight={}, claude=[{"commit": "v1\n"}])
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        run_stage(AgentBuildStage(), ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "the seal probe failed" in fault.reason and failed in fault.reason
    assert _calls(control, preflight=True) == [] and _calls(control) == []
    kinds = [ev["kind"] for ev in _events(ctx)]
    assert "build.preflight" not in kinds and "build.started" not in kinds
    [event] = [ev for ev in _events(ctx) if ev["kind"] == "build.seal_probe"]
    assert event["passed"] is False and float(fault.evidence["seal_probe_sec"]) == event["sec"] >= 0
    assert [c["name"] for c in event["checks"] if not c["ok"]] == [failed]
    assert _gone(ctx)


# ---------------------------------------------------------------------------
# Probe and preflight-only builds
# ---------------------------------------------------------------------------

#: The checks the stage makes host side, beside the sandbox's.
HOST_CHECKS = {"exec-server-ready", "workspace-owner", "server-oom-score", "spawner-oom-score", "nspid-depth",
               "reaper-no-socket", "refusal-recorded", "env-file-clean", "memory-split", "outside-scores",
               "pid-max", "harness-ucounts", "spawner-limits"}
#: The probe's PID 1 checks lower PID 1's limits. Without a namespace PID 1 is the machine's own
#: init, which an unprivileged user cannot touch; root could.
NOT_ROOT = pytest.mark.skipif(os.getuid() == 0, reason="the PID 1 checks would reach the machine's init as root")


def _probe(ctx: StageContext, outputs: dict) -> ProbeRecord:
    return ProbeRecord.model_validate_json(ctx.store.read_bytes(outputs["probe"]))


@NOT_ROOT
def test_a_probe_only_build_runs_the_probe_and_stops(sandbox_bins, tmp_path):
    """The sandbox starts as a build's does and is stopped as a build's normal end does; no CLI,
    no clock, no commit point, and every check is in probe.json, build.json and an event."""
    control = sandbox_bins
    _script(control, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe"))
    outputs = AgentBuildStage().run(ctx)
    assert sorted(outputs) == ["agent_env", "brief", "brief_provenance", "build", "probe", "served_ucounts"]
    assert _calls(control) == [] and _calls(control, preflight=True) == []
    record = _record(ctx, outputs)
    assert (record.ending, record.preflight, record.commit, record.commit_tries) == ("probe", None, None, [])
    probe = _probe(ctx, outputs)
    assert record.probe == probe
    names = {c.name: c for c in probe.checks}
    # The inbound check is the sandbox's command and the stage's knocks together; the accounting
    # check runs with the seal probe.
    assert set(names) == ({c.name for c in probe_module.SANDBOX_CHECKS} | HOST_CHECKS
                          | {"nproc-accounting", probe_module.INBOUND_CHECK})
    # The seal probe ran first, as every build's does, and passed.
    assert record.seal_probe.passed and [c.name for c in record.seal_probe.checks] == SEAL_CHECKS
    assert [c.name for c in probe.checks[:len(SEAL_CHECKS) + 1]] == ["exec-server-ready", *SEAL_CHECKS]
    # The namespace checks fail in a fake container, the step checks in a fake step, and the
    # outside-scores check, which finds no serving spawner in a fake step; the host-independent
    # ones pass.
    assert not probe.all_ok and all(not names[n].ok for n in (*NAMESPACE_CHECKS, *STEP_CHECKS))
    assert not names["outside-scores"].ok, names["outside-scores"]
    for name in ("python-version", "uid-gid", "cap-eff", "oom-score-adj", "pid1-non-dumpable", "pid1-route-env",
                 "route-env-absent", "command-nproc", "workspace-owner", "exec-server-ready", "env-file-clean",
                 "memory-split", *SEAL_CHECKS):
        assert names[name].ok, names[name]
    # The accounting this host's kernel uses is its own; what the check compares it with is the
    # site's.
    assert names["nproc-accounting"].expected == "accounting=uid"


@NOT_ROOT
def test_the_probe_checks_the_kernels_accounting_against_the_sites(sandbox_bins, tmp_path, monkeypatch):
    answering({"nproc-accounting": "echo forks=511"}, monkeypatch)
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe"), site={"nproc_accounting": "user_namespace"})
    probe = _probe(ctx, AgentBuildStage().run(ctx))
    [check] = [c for c in probe.checks if c.name == "nproc-accounting"]
    assert (check.ok, check.expected) == (True, "accounting=user_namespace") and "held=600 limit=512" in check.observed
    kinds = [e["kind"] for e in _events(ctx)]
    assert kinds.count("build.probe_check") == len(probe.checks) and "build.started" not in kinds
    assert not [s.name for s in ctx.launcher.launched if s.name.startswith(("commit", "snapshot", "transcript"))]
    assert _ended(ctx, "sandbox")["ending"] == "planned_stop"
    assert ctx.running_roles() == []


@NOT_ROOT
def test_a_preflight_only_build_also_runs_the_preflight(sandbox_bins, tmp_path):
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={}, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="preflight"))
    outputs = AgentBuildStage().run(ctx)
    assert sorted(outputs) == ["agent_env", "brief", "brief_provenance", "build", "probe", "served_ucounts"]
    [pre] = _calls(control, preflight=True)
    assert pre["env"]["ANTHROPIC_MODEL"] == "claude-opus-4-8[1m]"
    assert _calls(control) == []  # the agent itself never ran
    record = _record(ctx, outputs)
    assert record.ending == "preflight" and record.preflight is not None and record.preflight.ok
    assert (record.preflight.window, record.preflight.output_limit) == (1_000_000, 64_000)
    assert "build.preflight" in [e["kind"] for e in _events(ctx)]


@NOT_ROOT
def test_a_stop_build_registers_the_budget_server_from_the_tasks_soft_budget(sandbox_bins, tmp_path):
    """A stop build runs with no soft budget of its own (the planner gives it none), so the MCP
    registration is decided by the task's producer budget: multitenant_slo-like tasks have one, so
    the preflight reads the budget tool connected."""
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={}, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="preflight", soft_sec=None))
    record = _record(ctx, AgentBuildStage().run(ctx))
    assert record.preflight.budget_tool == "connected"
    assert [c for c, _ in MappedEnvironment.SENT if ".claude.json" in c and "mcpServers" in c]


@NOT_ROOT
def test_a_stop_build_of_a_task_without_a_soft_budget_registers_no_server(sandbox_bins, tmp_path, monkeypatch):
    """A task whose full build has no soft budget registers no budget server in a stop build
    either, matching the full build: the preflight reads it unchecked."""
    monkeypatch.setattr(agent_build, "budget_params",
                        lambda producer: {"soft_sec": None, "hard_sec": 3600, "max_turns": None})
    control = sandbox_bins
    _script(control, window=1_000_000, preflight={}, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="preflight", soft_sec=None))
    record = _record(ctx, AgentBuildStage().run(ctx))
    assert record.preflight.budget_tool == "unchecked"
    assert not [c for c, _ in MappedEnvironment.SENT if "mcpServers" in c]


@NOT_ROOT
def test_a_preflight_only_build_publishes_a_readback_that_differs_as_not_ok(sandbox_bins, tmp_path):
    """The model-route probe for an operator: another window read back is a not-ok preflight in
    build.json, the readback a full build fails on, and the build still publishes."""
    control = sandbox_bins
    _script(control, window=200_000, preflight={}, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="preflight"))
    record = _record(ctx, AgentBuildStage().run(ctx))
    assert record.ending == "preflight" and not record.preflight.ok and "200,000" in record.preflight.reason


@NOT_ROOT
def test_probe_oom_records_what_ended_the_hog(sandbox_bins, tmp_path, monkeypatch):
    # A harmless hog, so the test allocates nothing.
    monkeypatch.setattr(agent_build, "OOM_HOG", "true")
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe", probe_oom=True))
    oom = _probe(ctx, AgentBuildStage().run(ctx)).oom
    assert oom is not None and (oom.survived, oom.exit_code, oom.error, oom.runner_kills) == (True, 0, None, 0)
    assert (oom.oom_kills, oom.harness_max_bytes) == (0, 0)


def test_the_oom_probe_reads_the_sandboxs_survival_from_its_handle(tmp_path):
    ctx = _ctx(tmp_path, _params(stop_after="probe", probe_oom=True))
    build = agent_build._Build(ctx)

    class _Env:
        async def exec(self, command, timeout_sec=None):
            assert (command, timeout_sec) == (agent_build.OOM_HOG, agent_build.OOM_PROBE_SEC)
            return SimpleNamespace(return_code=137, stdout=None)

    class _Dead:
        def poll(self):
            return 137  # the hog took the sandbox down

    oom = asyncio.run(build.oom_probe(_Env(), _Dead()))
    assert (oom.survived, oom.exit_code, oom.samples) == (False, 137, 0)


def _as_command(code: str) -> str:
    """A command that runs *code* on this interpreter, base64-encoded so no path in it is
    rewritten by ``MappedEnvironment.exec`` (which maps ``/tmp`` path heads). ``shlex.quote``
    wraps the whole argument, so the base64 blob's own quotes are safe."""
    packed = base64.b64encode(code.encode()).decode()
    return f"{sys.executable} -c {shlex.quote(f'import base64;exec(base64.b64decode({packed!r}))')}"


def _hog_over(cgroup: Path) -> str:
    """A memory hog that, instead of allocating, writes the fake memory files of *cgroup* past
    the sandbox's limit and waits to be killed."""
    writes = "".join(f"pathlib.Path({k!r}).write_text({v!r})\n" for k, v in _over_limit(cgroup).items())
    return _as_command("import pathlib, time\n" + writes + "time.sleep(60)\n")


@NOT_ROOT
@pytest.mark.parametrize("harness_gib, cls", [(1, "contestant"), (10, "infra")], ids=["within-reserve", "over-reserve"])
def test_a_sandbox_that_ends_on_its_memory_during_the_probe_is_a_void_probe_with_the_judgement(
        sandbox_bins, tmp_path, monkeypatch, oom_watch, harness_gib, cls):
    """The hog takes the whole sandbox past its memory. A stop build reports no score and declares
    no contestant fault: whatever its roles judge the end, the instance ends
    infra that is not retried, a void probe outcome, and probe.json kept in the attempt carries
    the judgement. Within the reserve the roles call it the contestant's; over it,
    infra."""
    watch, cgroups = oom_watch
    monkeypatch.setattr(_PlantedSampler, "harness", harness_gib << 30)
    monkeypatch.setattr(agent_build, "OOM_HOG", _hog_over(cgroups / "cg-sandbox"))
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe", probe_oom=True), watch=watch,
               launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    # A stop build never declares a contestant fault or publishes build.json/probe.json as outputs.
    assert (fault.fault_class, fault.retryable) == ("infra", False) and ctx.contestant_fault() is None
    assert not ctx.output_path("probe.json").exists() and not ctx.output_path("build.json").exists()
    assert not [s.name for s in ctx.launcher.launched if s.name.startswith("commit")]
    assert fault.evidence["sandbox.fault_class"] == cls and fault.evidence["roles"] == "sandbox"
    evidence = ProbeRecord.model_validate_json((ctx.attempt_dir / "probe.json").read_text())
    assert evidence.sandbox_ending is not None and evidence.sandbox_ending["fault_class"] == cls
    assert evidence.oom is not None and evidence.oom.survived is False
    assert evidence.oom.harness_max_bytes == harness_gib << 30


@NOT_ROOT
def test_a_sandbox_that_ends_without_an_oom_during_the_probe_is_a_void_probe_judged_infra(sandbox_bins, tmp_path,
                                                                                         monkeypatch):
    # The "hog" kills the exec server, its own ancestor, as a crash of the sandbox would. It is
    # told by an argument that ends in the server's file name, which no shell in between has.
    code = ("import os, pathlib\npid = os.getpid()\nwhile pid > 1:\n"
            "    pid = int(pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])\n"
            "    try:\n        cmd = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()\n"
            "    except OSError:\n        continue\n"
            "    if any(arg.endswith(b'/exec_server.py') for arg in cmd.split(bytes(1))):\n"
            "        os.kill(pid, 9)\n        break\n")
    monkeypatch.setattr(agent_build, "OOM_HOG", _as_command(code))
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe", probe_oom=True))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable, fault.evidence["roles"]) == ("infra", False, "sandbox")
    assert fault.evidence["sandbox.fault_class"] == "infra" and ctx.contestant_fault() is None
    assert ProbeRecord.model_validate_json((ctx.attempt_dir / "probe.json").read_text()).sandbox_ending is not None


@NOT_ROOT
@pytest.mark.parametrize("cls", ["contestant", "infra"])
def test_a_sandbox_that_ended_by_itself_as_a_stop_build_stopped_it_is_a_void_probe(
        sandbox_bins, tmp_path, monkeypatch, cls):
    """The probe passed, but the sandbox ended on its own just before the stage stopped it: its
    own end stands, and a stop build records the roles' judgement in probe.json and ends infra
    that is not retried, whatever the class, never a contestant fault or a score."""
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe"))
    now = time.time()
    if cls == "contestant":
        split = SplitEvidence(reserve_bytes=4 * GIB, ended_at=now, oom_at=now - 1, oom_by="runner", sample_at=now - 2,
                              total_bytes=17 * GIB, tmpfs_bytes=0, harness_bytes=GIB, contestant_bytes=16 * GIB)
        ended = RoleOutcome("sandbox", "contestant", 137, "memory_limit", 1, split=split)
    else:
        ended = RoleOutcome("sandbox", "contestant", 1, "exited", split=SplitEvidence(reserve_bytes=4 * GIB,
                                                                                      ended_at=now))
    stop = ctx.stop_role

    def ended_first(name):
        outcome = stop(name)
        return ended if name == "sandbox" else outcome

    monkeypatch.setattr(ctx, "stop_role", ended_first)
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False) and ctx.contestant_fault() is None
    assert not ctx.output_path("probe.json").exists() and not ctx.output_path("build.json").exists()
    assert fault.evidence["sandbox.fault_class"] == cls
    evidence = ProbeRecord.model_validate_json((ctx.attempt_dir / "probe.json").read_text())
    assert evidence.sandbox_ending["fault_class"] == cls and evidence.checks


class _OnDemandSampler(_PlantedSampler):
    """A split sampler that samples only when asked, each sample's harness 1 GiB more than the last."""

    def __init__(self, cgroup_dir, **kw) -> None:
        self.taken: list[SplitSample] = []

    def sample_once(self) -> SplitSample:
        sample = SplitSample(at=time.time(), total=12 << 30, tmpfs=0, harness=(len(self.taken) + 1) << 30,
                             contestant=11 << 30, oom_kills=0)
        self.taken.append(sample)
        return sample

    def samples(self) -> list[SplitSample]:
        return list(self.taken)


@NOT_ROOT
def test_the_probe_reads_a_split_sample_taken_when_it_asks(sandbox_bins, tmp_path, monkeypatch):
    """The memory-split check and the OOM probe's harness maximum read a sample of the sandbox
    taken when they ask, not only the sampler's periodic ones: what the harness held once the
    hog ended is in the record."""
    monkeypatch.setattr(stage_module, "SplitSampler", _OnDemandSampler)
    monkeypatch.setattr(agent_build, "OOM_HOG", "true")
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe", probe_oom=True))
    probe = _probe(ctx, AgentBuildStage().run(ctx))
    [split] = [c for c in probe.checks if c.name == "memory-split"]
    assert split.observed.startswith(f"harness={1 << 30} ")
    assert probe.oom is not None and (probe.oom.samples, probe.oom.harness_max_bytes) == (2, 2 << 30)


def test_sample_split_takes_a_sample_of_a_reserved_role_now_and_none_of_another(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, _params())
    assert ctx.sample_split("sandbox") == []  # never started
    monkeypatch.setitem(ctx._splits, "sandbox", stage_module._Split(reserve_bytes=GIB,
                                                                    sampler=_OnDemandSampler(None)))
    assert [s.harness for s in ctx.sample_split("sandbox")] == [1 << 30]
    assert [s.harness for s in ctx.sample_split("sandbox")] == [1 << 30, 2 << 30]
    monkeypatch.setitem(ctx._splits, "other", stage_module._Split(reserve_bytes=GIB))  # no grant report yet
    assert ctx.sample_split("other") == []


def test_probe_oom_runs_only_with_a_probe():
    with pytest.raises(ValueError, match="probe_oom runs only"):
        AgentBuildParams.model_validate(_params(stop_after="preflight", probe_oom=True))
    with pytest.raises(ValueError, match="probe_oom runs only"):
        AgentBuildParams.model_validate(_params(probe_oom=True))


class _EndsOne(_NoContestantToEnd):
    """The contestant killer, ending one contestant process at each breach."""

    def end_one(self) -> bool:
        return True


@NOT_ROOT
def test_probe_oom_records_the_runners_kills_and_the_harness_maximum(sandbox_bins, tmp_path, monkeypatch, oom_watch):
    """Where the runner enforces memory it ends the hog alone and the sandbox survives: the
    record carries its kills in the step and the harness's largest resident memory."""
    watch, cgroups = oom_watch
    monkeypatch.setattr(stage_module, "ContestantKiller", _EndsOne)
    cgroup = cgroups / "cg-sandbox"
    under = {str(cgroup / "memory.current"): f"{GIB}\n", str(cgroup / "memory.stat"): f"anon {GIB}\nfile 0\nshmem 0\n"}
    writes = "".join(f"pathlib.Path({k!r}).write_text({v!r})\n" for k, v in _over_limit(cgroup).items())
    restores = "".join(f"pathlib.Path({k!r}).write_text({v!r})\n" for k, v in under.items())
    code = "import pathlib, time\n" + writes + "time.sleep(1)\n" + restores
    monkeypatch.setattr(agent_build, "OOM_HOG", _as_command(code))
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe", probe_oom=True), watch=watch,
               launcher=LocalLauncher(gpus=1, cgroup_root=cgroups))
    oom = _probe(ctx, AgentBuildStage().run(ctx)).oom
    assert oom is not None and oom.survived is True and oom.exit_code == 0
    assert oom.runner_kills >= 1 and oom.harness_max_bytes == 1 << 30 and oom.samples >= 1


@NOT_ROOT
def test_a_probe_that_fails_is_infra_retried(sandbox_bins, tmp_path, monkeypatch):
    def failing(self):
        raise RuntimeError("a test")

    monkeypatch.setattr(agent_build._Build, "probe_inputs", failing)
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe"))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "the probe failed: RuntimeError: a test" in e.value.fault.reason
    assert not ctx.output_path("probe.json").exists()


@NOT_ROOT
def test_a_failed_seal_check_in_a_probe_only_build_is_recorded_not_raised(sandbox_bins, tmp_path, monkeypatch):
    """The operator's probe keeps the whole record: a seal check that fails there is a not-ok
    check in probe.json and build.json, never a retry."""
    answering({"unlisted-refused": "echo answered"}, monkeypatch)
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe"))
    outputs = AgentBuildStage().run(ctx)
    record, probe = _record(ctx, outputs), _probe(ctx, outputs)
    assert not record.seal_probe.passed and not probe.all_ok
    assert {c.name: c.ok for c in probe.checks}["unlisted-refused"] is False


@NOT_ROOT
def test_a_route_variable_in_the_sandboxs_env_file_fails_both_credential_checks(sandbox_bins, tmp_path, monkeypatch):
    """A route variable the sandbox's env file held would reach every command: the probe names
    it, host side and in a command's environment, never its value."""
    real = agent_build._Build.data

    def planted(self, provisioned):
        mounts, env = real(self, provisioned)
        return mounts, {**env, "ANTHROPIC_API_KEY": "planted-not-a-secret"}

    monkeypatch.setattr(agent_build._Build, "data", planted)
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe"))
    probe = _probe(ctx, AgentBuildStage().run(ctx))
    names = {c.name: c for c in probe.checks}
    for name in ("env-file-clean", "route-env-absent"):
        assert (names[name].ok, names[name].observed) == (False, "ANTHROPIC_API_KEY")
    assert "planted-not-a-secret" not in probe.model_dump_json()


@NOT_ROOT
def test_a_stop_builds_probe_records_hold_no_relay_host(sandbox_bins, tmp_path, monkeypatch):
    """The probe records and events of a stop build on a route whose base URL is a secret hold
    neither the relay's URL nor its host, even when a check's observed value quotes it: the stage
    scrubs every check, the sealed ones and the rest. The K3 arm's route names
    kimi_url, host relay.example. A sealed check (route-reachable, scrubbed in the seal probe) and
    an unsealed one (cap-eff, scrubbed in the probe session) both carry the planted host."""
    answering({"route-reachable": f"echo ok TLSv1.3 {HOST}", "cap-eff": f"echo {HOST}"}, monkeypatch)
    _script(sandbox_bins, window=200_000, claude=[])
    ctx = _ctx(tmp_path, _params(arm=K3, stop_after="probe"))
    outputs = AgentBuildStage().run(ctx)
    probe_json = ctx.store.read_bytes(outputs["probe"]).decode()
    build_json = ctx.store.read_bytes(outputs["build"]).decode()
    events_json = (ctx.attempt_dir / "events.jsonl").read_text()
    for text in (probe_json, build_json, events_json):
        assert HOST not in text and RELAY_URL not in text
    # The host was seen and scrubbed, not simply absent: the route-reachable check holds a
    # redaction mark where the planted host was (relay.example is the host of kimi_url and
    # openai_url alike, so either secret's tag may be the one that replaced it).
    assert ".host>" in probe_json


@NOT_ROOT
def test_a_failed_seal_probes_fault_reason_holds_no_relay_host(sandbox_bins, tmp_path, monkeypatch):
    """A seal check that fails with the relay's host in what it saw: the infra fault's reason, the
    event and the attempt's logs name neither the host nor the URL."""
    answering({"unlisted-refused": f"echo answered {HOST}"}, monkeypatch)
    _script(sandbox_bins, window=200_000, preflight={}, claude=[{"commit": "v1\n"}])
    ctx = _ctx(tmp_path, _params(arm=K3))
    with pytest.raises(StageFault) as e:
        run_stage(AgentBuildStage(), ctx)
    reason = e.value.fault.reason
    assert "the seal probe failed" in reason and HOST not in reason and RELAY_URL not in reason
    events_json = (ctx.attempt_dir / "events.jsonl").read_text()
    assert HOST not in events_json and RELAY_URL not in events_json


@NOT_ROOT
def test_a_stop_build_whose_allocation_shuts_down_during_the_probe_is_infra_retried(sandbox_bins, tmp_path,
                                                                                    monkeypatch):
    """The allocation is told to stop while the sandbox still runs the probe: the stop build ends
    infra, retried, and writes no probe.json."""
    real = agent_build._Build.probe_session

    async def shutting(self, env, agent, layout, sandbox):
        record, pre, error = await real(self, env, agent, layout, sandbox)
        self.ctx.stopping = True
        return record, pre, error

    monkeypatch.setattr(agent_build._Build, "probe_session", shutting)
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe"))
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "shutting down" in e.value.fault.reason
    assert not ctx.output_path("probe.json").exists() and not (ctx.attempt_dir / "probe.json").exists()
