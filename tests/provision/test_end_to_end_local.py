"""``hls provision`` on one machine: the command plans and submits, the in-process Slurm of
``tests/pipeline/test_end_to_end_local.py`` runs the real runner, and the provision roles run
the real role program against the fake hub. Stood in for: srun and the step wrapper, the
cluster and its measured grant, the hub, and the checkout's git state."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from hls.pipeline import commands
from hls.pipeline.events import read_events
from hls.pipeline.plan import PlanStore
from hls.provision.model import Provisioned, weights_key
from hls.store.cache import ContentCache
from hls.store.layout import InstanceLayout
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.slurm.scheduler import submit_argv
from tests.pipeline.hls_cli import flags, run_hls
from tests.pipeline.test_end_to_end_local import LocalSlurm
from tests.provision import fake_hub
from tests.provision.tasks import SHA
from tests.provision.warmer_config import SECRETS, TOKEN, write_config

FAKE_ROLE = Path(fake_hub.__file__).with_name("fake_role.py")
FILES = {"config.json": b'{"model_type": "tx"}', "model.safetensors": b"\x01" * 4096}


@pytest.fixture
def hub(tmp_path, monkeypatch) -> Path:
    """The fake hub, with the task model at ``SHA``."""
    root = tmp_path / "hub"
    fake_hub.make_hub(root, "org/model", SHA, FILES)
    monkeypatch.setenv("FAKE_HUB", str(root))
    # The local launcher passes the test process's environment on; a Slurm step gets none of it.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return root


@pytest.fixture
def records(caplog, monkeypatch):
    """Every log record of the test. configure_logging turns propagation off on the hls
    loggers, which would keep their records from caplog."""
    monkeypatch.setattr(logging.getLogger("hls"), "propagate", True)
    caplog.set_level(logging.DEBUG)
    return caplog


def _slurm() -> LocalSlurm:
    """An in-process Slurm whose jobs run as soon as they are submitted, with the provision
    role program served by the fake hub."""
    slurm = LocalSlurm(replace_modules={"hls.provision.role": FAKE_ROLE})

    def submit_and_run(request):
        job_id = LocalSlurm.submit(slurm, request)
        slurm.tick()
        return job_id

    slurm.submit = submit_and_run
    return slurm


def _provision(tmp_path: Path, slurm: LocalSlurm, *, module: str = "tests.provision.tasks", weights: bool = True,
               overrides: dict[str, str] | None = None) -> tuple[int, list[str]]:
    """``hls provision <module>``, with ``--rows-only`` unless *weights*, and *overrides*: the exit
    status and what it printed."""
    return run_hls(slurm, ["provision", module, *flags(overrides or {}), *([] if weights else ["--rows-only"]),
                           "--job-name", "hls-provision-test",
                           "--config", str(write_config(tmp_path, secrets=SECRETS))])


def _events(store: PosixStore, instance: str) -> list[dict]:
    return read_events(store.path(InstanceLayout(instance).attempt_file("provision", 1, "events.jsonl")))


def _tokens(hub: Path) -> list[str | bool]:
    """The token of each call the roles made to the hub."""
    return [c["token"] for c in fake_hub.calls(hub)]


def _holding_the_token(tmp_path: Path, hub: Path, lines: list[str], records) -> list[str]:
    """Every echoed line, log record and file under *tmp_path* that holds the token, other
    than the secrets file and the hub's own record of its calls."""
    skip = {tmp_path / "secrets.toml", hub / "calls.jsonl"}
    files = [str(p) for p in tmp_path.rglob("*") if p.is_file() and p not in skip and TOKEN.encode() in p.read_bytes()]
    return [line for line in lines if TOKEN in line] + ([records.text] if TOKEN in records.text else []) + files


def test_warmers_are_planned_one_at_a_time(tmp_path, login):
    config = write_config(tmp_path, secrets=SECRETS)
    plan = commands.plan_provision(config=config, overrides={}, task_module="tests.provision.tasks", weights=True,
                                   job_name="hls-provision", echo=[].append)
    assert plan.job_name == "hls-warmer" and all(a.singleton for a in plan.allocations)
    plan = commands.plan_provision(config=config, overrides={"site.warmers_one_at_a_time": "false"},
                                   task_module="tests.provision.tasks", weights=True, job_name="hls-provision",
                                   echo=[].append)
    assert plan.job_name == "hls-provision" and not any(a.singleton for a in plan.allocations)


def test_hls_provision_fills_the_cache_once_and_then_finds_it(tmp_path, login, hub, records):
    slurm = _slurm()
    code, lines = _provision(tmp_path, slurm)
    assert code == 0, lines
    instance = lines[0].removeprefix("instance ")
    assert instance.startswith("warmers/T-provision/")
    # 246 min of the stage and the overhead, and 12.5 min for its 5 roles' steps to start.
    assert lines[1] == "allocation 0: provision; 0 x no GPU, 4 CPUs, 16 GiB, 258 min, partition cpu-a"
    assert lines[-1] == f"{instance}: succeeded: every stage published"
    request = slurm.jobs["100"].request
    # A warmer waits for the site's other warmers: Slurm's singleton, under their one job name.
    assert "--partition=cpu-a" in submit_argv(request) and "--gpus=0" in submit_argv(request)
    assert "--job-name=hls-warmer" in submit_argv(request) and request.singleton
    store = PosixStore(tmp_path / "store")
    manifest, _ = read_manifest(store, InstanceLayout(instance), "provision")
    prov = Provisioned.model_validate_json(store.read_bytes(manifest.outputs["provisioned"].key))
    weights = ContentCache(store).lookup("weights", weights_key("org/model", SHA))
    assert prov.weights.key == weights.key and sorted(f.path for f in weights.manifest.files) == sorted(FILES)
    # hls status says what the provision stage put in the cache, by key.
    status = commands.status_text(instance, config=write_config(tmp_path, secrets=SECRETS), scheduler=slurm)
    assert status.splitlines()[-2:] == [
        f"provisioned weights: org/model at {SHA} (cache weights/{prov.weights.key})",
        f"provisioned rows: tenants {', '.join(prov.rows.tenants)} (cache rows/{prov.rows.key})"]
    # The granted token reached the hub, and no file of the instance holds it.
    assert {c["token"] for c in fake_hub.calls(hub)} == {TOKEN}
    held = [p for p in store.path(instance).rglob("*") if p.is_file() and TOKEN.encode() in p.read_bytes()]
    assert held == []

    code, again = _provision(tmp_path, slurm)
    assert code == 0, again
    second = again[0].removeprefix("instance ")
    assert second != instance and submit_argv(slurm.jobs["101"].request).count("--partition=cpu-a") == 1
    events = _events(store, second)
    assert [e["role"] for e in events if e["kind"] == "role_started"] == ["resolve"]
    assert [e["kind"] for e in events].count("cache_hit") == 2
    # Neither run echoed, logged or left a file with the token.
    assert _holding_the_token(tmp_path, hub, lines + again, records) == []


def test_hls_provision_rows_only_fills_the_store_an_override_names_and_says_which(tmp_path, login, hub, records):
    other = tmp_path / "other"
    code, lines = _provision(tmp_path, _slurm(), weights=False, overrides={"site.store_root": str(other)})
    assert code == 0, lines
    instance = lines[0].removeprefix("instance ")
    assert lines[1] == f"store root {other}; resume, status and cancel need --store {other}"
    store = PosixStore(other)
    manifest, _ = read_manifest(store, InstanceLayout(instance), "provision")
    prov = Provisioned.model_validate_json(store.read_bytes(manifest.outputs["provisioned"].key))
    assert prov.weights is None and prov.rows is not None
    assert not (tmp_path / "store").exists()
    # Leaving out the weights keeps the grant: a gated dataset needs the token as well.
    assert PlanStore(store, InstanceLayout(instance)).latest().stage("provision").grants == {"HF_TOKEN": "hf_token"}
    assert _tokens(hub) and set(_tokens(hub)) == {TOKEN}
    assert _holding_the_token(tmp_path, hub, lines, records) == []


def test_a_warmer_that_does_not_succeed_exits_1(tmp_path, login, hub):
    code, lines = _provision(tmp_path, _slurm(), module="tests.provision.unpinned_task")
    assert code == 1
    assert re.fullmatch(r"warmers/T-provision/\S+: void: provision: .*refused.*", lines[-1]), lines[-1]


def test_hls_provision_without_a_secrets_file_says_so_and_sends_the_hub_no_token(tmp_path, login, hub):
    code, lines = run_hls(_slurm(), ["provision", "tests.provision.tasks", "--config",
                                     str(write_config(tmp_path))])
    assert code == 0, lines
    assert lines[0] == f"note: no secrets file at {tmp_path / 'secrets.toml'}; no stage is granted a secret"
    assert lines[1].startswith("instance warmers/T-provision/") and lines[-1].endswith(": succeeded: every stage published")
    # token=False tells the hub client to send none.
    assert _tokens(hub) and set(_tokens(hub)) == {False}
    assert {c["implicit_token_off"] for c in fake_hub.calls(hub)} == {True}
