from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from hls.pipeline.pipelines import KNOWN_STAGE_NAMES
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.planner import StageDecl, build_plan
from hls.spec.resources import ResourceShape
from hls.store.layout import InstanceLayout
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.runner import EXIT_FAILED, EXIT_PUBLISHED
from hls.substrate.secrets import load_secrets
from tests.finalize import registry_stage  # noqa: F401  (registers the registry probe)
from tests.pipeline import probe_stage  # noqa: F401  (registers the grant probe)
from tests.pipeline.plans import CODE, settings
from tests.pipeline.test_runner import _runner, _status, _store, _sub

RELAY, HUB = "relay-token-0123456789", "hf-token-0123456789"
SECRETS = f"""\
[secrets]
relay = "{RELAY}"
hf_token = "{HUB}"
"""
#: What ``tests.pipeline.test_runner._grant`` grants.
SHAPE = ResourceShape(gpus=2, gpu_kind="H200", cpus=8, memory_gib=12, time_sec=600)


def _secrets_file(tmp_path: Path, text: str = SECRETS) -> Path:
    path = tmp_path / "site" / "secrets.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    path.chmod(0o600)
    return path


def _plan(tmp_path: Path, secrets_path: Path, *decls: StageDecl) -> Plan:
    decls = decls or (StageDecl("registry", "registry_probe", SHAPE, {}),)
    secrets = load_secrets(secrets_path) if secrets_path.exists() else None
    plan = build_plan(decls=list(decls), task="noop", pipeline="registry",
                      layout=InstanceLayout.rollout("noop", "q1"),
                      settings=settings(tmp_path, site={"secrets_path": str(secrets_path)}), code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-registry", known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=(), now=1000.0, secrets=secrets)
    return _store(plan, _sub(plan))


def _seen(plan: Plan, stage: str) -> dict:
    store = PosixStore(plan.store_root)
    manifest, _ = read_manifest(store, plan.layout, stage)
    return json.loads(store.read_bytes(manifest.outputs["seen"].key))


def test_a_stage_that_needs_the_registry_gets_every_value_without_grants(tmp_path):
    plan = _plan(tmp_path, _secrets_file(tmp_path))
    assert all(not s.grants for s in plan.stages)
    assert _runner(plan).run() == EXIT_PUBLISHED
    digests = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in (("hf_token", HUB),
                                                                                     ("relay", RELAY))}
    assert _seen(plan, "registry") == {"names": ["hf_token", "relay"], "digests": digests, "scrubbed": True,
                                       "registry": True}


def test_only_the_stage_that_needs_the_registry_gets_it(tmp_path):
    path = _secrets_file(tmp_path)
    plan = _plan(tmp_path, path, StageDecl("probe", "probe", SHAPE, {}),
                 StageDecl("registry", "registry_probe", SHAPE, {}, after=("probe",)))
    assert _runner(plan).run() == EXIT_PUBLISHED
    assert _seen(plan, "probe")["names"] == []
    assert _seen(plan, "registry")["names"] == ["hf_token", "relay"]


def test_without_the_secrets_file_the_allocation_fails_infra_and_is_not_retried(tmp_path):
    plan = _plan(tmp_path, tmp_path / "site" / "missing.toml")
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert "secrets file" in fault.reason and "does not exist" in fault.reason
    assert PosixStore(plan.store_root).list_children(plan.layout.stage("registry")) == []


def test_a_grant_that_takes_a_secrets_name_is_refused_before_the_stage_runs(tmp_path):
    path = _secrets_file(tmp_path, SECRETS + '\n[grants]\n"stage.registry".relay = "hf_token"\n')
    plan = _plan(tmp_path, path)
    assert plan.stage("registry").grants == {"relay": "hf_token"}
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert not fault.retryable and "['relay']" in fault.reason
    assert RELAY not in fault.reason and HUB not in fault.reason
    assert PosixStore(plan.store_root).list_children(plan.layout.stage("registry")) == []


def test_a_registry_with_no_secret_is_still_handed_over_as_loaded(tmp_path):
    plan = _plan(tmp_path, _secrets_file(tmp_path, "[secrets]\n"))
    assert _runner(plan).run() == EXIT_PUBLISHED
    assert _seen(plan, "registry") == {"names": [], "digests": {}, "scrubbed": True, "registry": True}
