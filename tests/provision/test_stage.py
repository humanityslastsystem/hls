from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest

from hls.pipeline.events import EventLog
from hls.pipeline.pipelines import (KNOWN_STAGE_NAMES, PROVISION_CPUS, PROVISION_ROLE_GIB, PROVISION_SEC,
                                    provision_stages)
from hls.pipeline.plan import Plan
from hls.pipeline.planner import PlanError, StageDecl
from hls.pipeline.stage import StageContext, StageFault, budget_problem, run_stage
from hls.provision.model import Provisioned, rows_key, weights_key
from hls.provision.rows import dev_digest, provisioning_digest, unit_descriptions
from hls.spec.resources import ResourceShape
from hls.stages import provision as provision_stage
from hls.stages.provision import ROLE_MODULE, ProvisionStage
from hls.store.cache import ContentCache, copy_disk_bytes, entry_sha256
from hls.store.lock import GenerationLock, Holder
from hls.store.posix import STAGING, PosixStore
from hls.substrate.base import RoleReport
from hls.substrate.slurm.launcher import step_env
from hls.taskref import TaskRef
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan
from tests.provision import fake_hub, tool_task
from tests.provision.tasks import SHA
from tests.store.entries import entry_reads, populate, recording_reads

FAKE_ROLE = Path(fake_hub.__file__).with_name("fake_role.py")
#: A stand-in for the role program whose digest report is gone when it ends.
UNREPORTING = Path(fake_hub.__file__).with_name("unreporting_role.py")
TOKEN = "hf-token-0123456789"
FILES = {"config.json": b'{"model_type": "tx"}', "model.safetensors": b"\x01" * 4096,
         "tokenizer/vocab.txt": b"a\nb\n"}
#: A warmer's provision stage for a task that declares no provision memory, with the runner's
#: default reserve beside its roles' memory (``provision_shape``).
SHAPE = ResourceShape(cpus=PROVISION_CPUS, memory_gib=PROVISION_ROLE_GIB + 4, time_sec=PROVISION_SEC)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    fake_hub.make_hub(root, "org/model", SHA, FILES)
    monkeypatch.setenv("FAKE_HUB", str(root))
    # The local launcher passes the test process's environment on; a Slurm step gets none of it.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(provision_stage, "POLL_SEC", 0.1)
    return root


def _plan(tmp_path: Path, *, module: str = "tests.provision.tasks", instance: str = "r1", weights: bool = True,
          overrides: dict | None = None, **run):
    ref = TaskRef(module=module, overrides=overrides or {})
    return noop_plan(tmp_path, decls=provision_stages(task_ref=ref, shape=SHAPE, weights=weights), instance=instance,
                     **run)


def _launcher() -> LocalLauncher:
    return LocalLauncher(gpus=0, replace_modules={"hls.provision.role": FAKE_ROLE})


def _ctx(plan: Plan, launcher: LocalLauncher, tmp_path: Path, *, attempt: int = 1,
         secrets: dict[str, str] | None = None, job: str = "100") -> StageContext:
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("provision", attempt, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage("provision"), attempt=attempt, launcher=launcher,
                        watch=None, events=events, inputs={}, deadline=time.time() + 120, secrets=secrets or {},
                        cache=ContentCache(store, lock_ttl_sec=30), scratch=tmp_path / "node" / f"hls-{job}",
                        job_id=job)


def _run(plan, tmp_path, **kw) -> tuple[dict[str, str], LocalLauncher, StageContext]:
    launcher = _launcher()
    ctx = _ctx(plan, launcher, tmp_path, **kw)
    return run_stage(ProvisionStage(), ctx), launcher, ctx


def _provisioned(ctx: StageContext, outputs: dict[str, str]) -> Provisioned:
    return Provisioned.model_validate_json(ctx.store.read_bytes(outputs["provisioned"]))


def _roles(launcher: LocalLauncher) -> list[str]:
    return [s.name for s in launcher.launched]


def _kinds(ctx: StageContext) -> list[str]:
    return [e["kind"] for e in map(json.loads, (ctx.attempt_dir / "events.jsonl").read_text().splitlines())]


def _hf_home(ctx: StageContext, role: str) -> str:
    return str(ctx.scratch / "roles" / role / "hf")


def _entry_events(ctx: StageContext, kind: str) -> dict[str, str]:
    """``<entry kind>/<key>`` to the entry digest of each event of *kind* in the attempt's log."""
    events = map(json.loads, (ctx.attempt_dir / "events.jsonl").read_text().splitlines())
    return {f"{e['entry_kind']}/{e['key']}": e["entry_sha256"] for e in events if e["kind"] == kind}


def _role_shape(plan: Plan) -> ResourceShape:
    """The stage's shape less the memory it leaves the runner, and no disk: what each role must
    get."""
    shape = plan.stage("provision").shape
    return shape.model_copy(update={"memory_gib": shape.memory_gib - plan.settings.run.runner_memory_gib,
                                    "disk_gib": 0})


def _refs(prov: Provisioned) -> dict[str, str]:
    """``<kind>/<key>`` to the entry digest of every entry *prov* names."""
    refs = {} if prov.weights is None else {f"weights/{prov.weights.key}": prov.weights.entry_sha256}
    if prov.rows is not None:
        refs[f"rows/{prov.rows.key}"] = prov.rows.entry_sha256
    return refs | {f"tools/{t.key}": t.entry_sha256 for t in prov.tools}


def test_a_cold_run_fills_the_weights_and_rows_and_names_every_entry(tmp_path, hub):
    plan = _plan(tmp_path)
    outputs, launcher, ctx = _run(plan, tmp_path)
    assert _roles(launcher) == ["resolve", "weights-0", "rows"]
    for spec in launcher.launched:
        # Host-side roles: the plan's interpreter, the checkout's task modules, node-local hub caches.
        assert spec.argv[:3] == (str(plan.toolchain.python), "-m", "hls.provision.role")
        assert spec.env == {"PYTHONPATH": str(plan.code.checkout), "HF_HOME": _hf_home(ctx, spec.name)}
        assert spec.shape == _role_shape(plan) and spec.shape.gpus == 0 and spec.owner == "harness"
    # Each fill role reports the digest of every file it wrote, and the weights role the git blob
    # ids, into its own node-local directory, outside the entry.
    weights_role, rows_role = launcher.launched[1:]
    own = ctx.scratch / "roles"
    assert weights_role.argv[-6:] == ("--blobs-out", str(own / "weights-0" / "blobs.json"), "--digests-out",
                                      str(own / "weights-0" / "digests.json"), "--out", weights_role.argv[-1])
    assert rows_role.argv[-4:] == ("--digests-out", str(own / "rows" / "digests.json"), "--out",
                                   rows_role.argv[-1])
    assert set(outputs) == {"resolved", "source-0", "provisioned"}
    prov = _provisioned(ctx, outputs)
    cache = ContentCache(ctx.store)
    weights = cache.lookup("weights", weights_key("org/model", SHA))
    assert prov.weights.key == weights.key and prov.weights.entry_sha256 == entry_sha256(weights)
    # The download's files, as plain files, and none of the hub client's bookkeeping.
    assert sorted(f.path for f in weights.manifest.files) == sorted(FILES)
    assert (weights.path / "model.safetensors").read_bytes() == FILES["model.safetensors"]
    assert weights.manifest.source == {"stage": "provision", "instance": plan.instance, "repo": "org/model",
                                       "revision": SHA}
    rows = cache.lookup("rows", prov.rows.key)
    assert prov.rows.entry_sha256 == entry_sha256(rows) and prov.rows.tenants == ["t0"]
    assert (rows.path / "tenants" / "t0" / "dev" / "rows.json").exists()
    assert prov.tools == []
    kinds = _kinds(ctx)
    assert "cache_hit" not in kinds and "cache_wait" not in kinds
    assert _entry_events(ctx, "cache_filled") == _refs(prov)
    # Each role is waited for once, though the stage runs them one after another.
    assert kinds.count("role_ended") == 3
    # Each role's private hub cache went with its role directory.
    assert not any((ctx.scratch / "roles").iterdir())
    # The fills held each entry's lock as this instance's provision, in this job.
    for kind, key in (("weights", prov.weights.key), ("rows", prov.rows.key)):
        holder = GenerationLock(ctx.store, f"cache-locks/{kind}/{key}", ttl_sec=30).current().holder
        assert (holder.purpose, holder.job_id) == (f"provision {plan.instance}", "100")


def test_tool_bundles_are_filled_full_then_dev_keyed_by_the_partition(tmp_path, hub):
    plan = _plan(tmp_path, module="tests.provision.tool_task")
    outputs, launcher, ctx = _run(plan, tmp_path)
    assert _roles(launcher) == ["resolve", "weights-0", "rows", "tools-full", "tools-dev"]
    prov = _provisioned(ctx, outputs)
    cache = ContentCache(ctx.store)
    ids = json.loads((cache.lookup("rows", prov.rows.key).path / "ids.json").read_text())
    by_variant = {t.variant: t for t in prov.tools}
    assert by_variant["full"].key == "tiny-tiny-v1-full" and by_variant["full"].dev_digest is None
    assert by_variant["dev"].dev_digest == dev_digest(ids["dev"], ids["test"])
    assert by_variant["dev"].key == f"tiny-tiny-v1-dev-{dev_digest(ids['dev'], ids['test'])}"
    dev = json.loads((cache.lookup("tools", by_variant["dev"].key).path / "manifest.json").read_text())
    assert dev == {"variant": "dev", "rows": ids["dev"]}
    full = json.loads((cache.lookup("tools", by_variant["full"].key).path / "manifest.json").read_text())
    assert full == {"variant": "full", "identity": "tiny-v1"}
    assert _entry_events(ctx, "cache_filled") == _refs(prov)


def test_a_new_partition_finds_the_full_bundle_and_fills_the_rows_and_the_dev_bundle(tmp_path, hub):
    first, _, ctx1 = _run(_plan(tmp_path, module="tests.provision.tool_task"), tmp_path)
    plan = _plan(tmp_path, module="tests.provision.tool_task_repartitioned", instance="r2")
    again, launcher, ctx2 = _run(plan, tmp_path, job="101")
    assert _roles(launcher) == ["resolve", "rows", "tools-dev"]
    old, new = _provisioned(ctx1, first), _provisioned(ctx2, again)
    assert new.weights == old.weights and new.rows.key != old.rows.key
    full, dev = ({t.variant: t for t in new.tools}[v] for v in ("full", "dev"))
    assert full == {t.variant: t for t in old.tools}["full"] and dev != {t.variant: t for t in old.tools}["dev"]
    assert _entry_events(ctx2, "cache_hit") == {f"weights/{new.weights.key}": new.weights.entry_sha256,
                                                f"tools/{full.key}": full.entry_sha256}
    assert _entry_events(ctx2, "cache_filled") == {f"rows/{new.rows.key}": new.rows.entry_sha256,
                                                   f"tools/{dev.key}": dev.entry_sha256}


def test_a_second_run_finds_every_entry_and_starts_only_the_resolve_role(tmp_path, hub):
    plan = _plan(tmp_path, module="tests.provision.tool_task")
    first, _, ctx1 = _run(plan, tmp_path)
    again, launcher, ctx2 = _run(plan, tmp_path, attempt=2)
    assert _roles(launcher) == ["resolve"]
    assert _provisioned(ctx2, again) == _provisioned(ctx1, first)
    assert set(again) == {"resolved", "provisioned"}
    assert "cache_filled" not in _kinds(ctx2)
    assert _entry_events(ctx2, "cache_hit") == _refs(_provisioned(ctx2, again))


def test_a_download_that_does_not_match_the_hub_publishes_nothing(tmp_path, hub):
    fake_hub.serve_instead(hub, "org/model", SHA, "model.safetensors", b"\x02" * 4096)
    plan = _plan(tmp_path)
    launcher = _launcher()
    ctx = _ctx(plan, launcher, tmp_path)
    with pytest.raises(StageFault, match="do not match the hub: model.safetensors: sha256") as caught:
        run_stage(ProvisionStage(), ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    store = PosixStore(plan.store_root)
    assert store.list_children("cache/weights") == []
    leftovers = store.root / STAGING / "cache" / "weights" / weights_key("org/model", SHA)
    assert not leftovers.exists() or list(leftovers.iterdir()) == []
    assert _roles(launcher) == ["resolve", "weights-0"]


def test_a_git_file_that_does_not_match_the_hub_publishes_nothing(tmp_path, hub):
    """The weights role reports the git blob id of each file the hub keeps in git, so the check
    reads no file of the fill."""
    fake_hub.serve_instead(hub, "org/model", SHA, "config.json", b'{"model_type": "ty"}')
    plan = _plan(tmp_path)
    with pytest.raises(StageFault, match="do not match the hub: config.json: git blob id differs") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert PosixStore(plan.store_root).list_children("cache/weights") == []


@pytest.mark.parametrize("text, says", [
    (None, "reported no git blob ids"), ("not json", "reported no git blob ids"),
    ('["a"]', "not a map of strings"), ('{"a": 1}', "not a map of strings"),
], ids=["missing", "garbled", "list", "number"])
def test_blob_ids_a_weights_role_did_not_report_are_a_retryable_fault(tmp_path, text, says):
    path = tmp_path / "blobs.json"
    if text is not None:
        path.write_text(text)
    with pytest.raises(StageFault, match=says) as caught:
        provision_stage._blob_ids(path, "weights-0")
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    path.write_text('{"config.json": "' + "c" * 40 + '"}')
    assert provision_stage._blob_ids(path, "weights-0") == {"config.json": "c" * 40}


def test_a_fill_role_that_reports_no_digests_publishes_nothing_and_is_retried(tmp_path, hub):
    plan = _plan(tmp_path)
    launcher = LocalLauncher(gpus=0, replace_modules={"hls.provision.role": UNREPORTING})
    with pytest.raises(StageFault, match="role weights-0 reported no digests of the files") as caught:
        run_stage(ProvisionStage(), _ctx(plan, launcher, tmp_path))
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert PosixStore(plan.store_root).list_children("cache/weights") == []


def test_the_runner_reads_no_file_of_the_entries_it_fills_or_copies_but_the_row_ids(tmp_path, hub):
    """Weights, rows and tool bundles are written, hashed and copied by roles. The one entry file
    the stage's own process reads is the rows entry's ``ids.json``, kilobytes of row ids, once,
    to key the dev bundle (``row_ids``)."""
    plan = _plan(tmp_path, module="tests.provision.tool_task")
    store = PosixStore(plan.store_root)
    with recording_reads() as reads:
        outputs, launcher, ctx = _run(plan, tmp_path)
        prov = _provisioned(ctx, outputs)
        later = _ctx(plan, _launcher(), tmp_path, attempt=2)
        refs = {"weights": [prov.weights], "rows": [prov.rows], "tools": prov.tools}
        copies = [later.materialize(kind, ref.key, ref.entry_sha256) for kind, items in refs.items()
                  for ref in items]
    assert _roles(launcher) == ["resolve", "weights-0", "rows", "tools-full", "tools-dev"]
    roots = [store.root / "cache", store.root / STAGING / "cache", later.scratch / "100"]
    assert entry_reads(reads, roots) == [str(store.path(f"cache/rows/{prov.rows.key}") / "ids.json")]
    assert len(copies) == 4 and all(c.is_dir() for c in copies)


@pytest.mark.parametrize("links", ["symlink", "hardlink"])
def test_a_download_of_links_into_a_blob_cache_publishes_nothing(tmp_path, hub, monkeypatch, links):
    """The cache takes plain files only, so a client that links its files to a blob cache
    fills nothing. The same download would be refused again, so it is not retried."""
    monkeypatch.setenv("FAKE_HUB_LINKS", links)
    plan = _plan(tmp_path)
    with pytest.raises(StageFault, match="the fill of weights/.* was refused") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert PosixStore(plan.store_root).list_children("cache/weights") == []


def test_a_failed_role_is_a_retryable_infra_fault(tmp_path, hub):
    plan = _plan(tmp_path, module="tests.provision.no_such_module")
    with pytest.raises(StageFault, match="harness role resolve ended exited") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable


@pytest.mark.parametrize("code", ["3", "4"])
def test_a_role_that_exits_with_a_refusal_status_but_says_no_refusal_is_retried(tmp_path, hub, monkeypatch, code):
    # A library that ends the process with the same status is no refusal of the task.
    monkeypatch.setenv("FAKE_HUB_EXIT", code)
    launcher = _launcher()
    with pytest.raises(StageFault, match=f"harness role weights-0 ended exited with status {code}") as caught:
        run_stage(ProvisionStage(), _ctx(_plan(tmp_path), launcher, tmp_path))
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", True)
    assert _roles(launcher) == ["resolve", "weights-0"]


@pytest.mark.parametrize("status", [401, 403, 404])
@pytest.mark.parametrize("weights, role", [(True, "weights-0"), (False, "rows")])
def test_a_hub_that_refuses_what_a_fill_role_asks_for_is_a_fault_a_retry_would_repeat(tmp_path, hub, monkeypatch,
                                                                                      status, weights, role):
    # resolve's model lookup is answered; the fill's download or corpus then gets the refusal.
    plan = _plan(tmp_path, weights=weights)
    monkeypatch.setenv("FAKE_HUB_STATUS", str(status))
    launcher = _launcher()
    with pytest.raises(StageFault, match=f"the hub refused what role {role} asked for task module "
                                         f"tests.provision.tasks: .*{status}") as caught:
        run_stage(ProvisionStage(), _ctx(plan, launcher, tmp_path))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable, fault.evidence[f"{role}.exit_code"]) == ("infra", False, "4")
    assert _roles(launcher)[-1] == role


@pytest.mark.parametrize("status", [429, 500, 503])
def test_a_hub_that_fails_for_now_is_retried(tmp_path, hub, monkeypatch, status):
    monkeypatch.setenv("FAKE_HUB_STATUS", str(status))
    with pytest.raises(StageFault, match="harness role weights-0 ended exited with status 1") as caught:
        run_stage(ProvisionStage(), _ctx(_plan(tmp_path), _launcher(), tmp_path))
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", True)


@pytest.mark.parametrize("module, reason", [
    ("tests.provision.oversized_task", r"dev\(15\) \+ test\(10\) exceeds corpus size 20"),
    ("tests.provision.empty_split_task", r"tenant t0: test split resolves to 0 rows of a corpus of 20 rows"),
], ids=["larger-than-the-corpus", "rounds-to-no-row"])
def test_rows_the_corpus_cannot_partition_as_declared_are_a_fault_a_retry_would_repeat(tmp_path, hub, module,
                                                                                      reason):
    launcher = _launcher()
    with pytest.raises(StageFault,
                       match=f"the provision role rows refused task module {module}: .*{reason}") as caught:
        run_stage(ProvisionStage(), _ctx(_plan(tmp_path, module=module, weights=False), launcher, tmp_path))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable, fault.evidence["rows.exit_code"]) == ("infra", False, "3")
    assert _roles(launcher) == ["resolve", "rows"]


def test_the_roles_are_given_the_task_overrides(tmp_path, hub):
    """The roles that load the task, resolve and rows here, are handed the plan's overrides with its
    module; a plan without overrides hands none."""
    overrides = {"task.data_source.split.test.value": 3}
    _, launcher, _ = _run(_plan(tmp_path, weights=False, overrides=overrides), tmp_path)
    named = {s.name: list(s.argv) for s in launcher.launched if s.name in ("resolve", "rows")}
    assert set(named) == {"resolve", "rows"}
    for argv in named.values():
        at = argv.index("--overrides")
        assert argv[at + 1] == json.dumps(overrides)
        assert argv[argv.index("--task-module") + 1] == "tests.provision.tasks"
    _, bare, _ = _run(_plan(tmp_path / "bare", weights=False), tmp_path / "bare", job="101")
    assert all("--overrides" not in s.argv for s in bare.launched)


def test_a_garbled_hub_reply_to_a_fill_is_retried(tmp_path, hub, monkeypatch):
    """A reply the hub client cannot parse says nothing about the task, so it is no refusal. The
    weights role is the first to ask the hub: resolve reads the task's declaration alone."""
    monkeypatch.setenv("FAKE_HUB_GARBLED", "1")
    with pytest.raises(StageFault, match="harness role weights-0 ended exited with status 1") as caught:
        _run(_plan(tmp_path), tmp_path)
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", True)


@pytest.mark.parametrize("module, reason", [("tests.provision", "defines no TASK"),
                                            ("tests.provision.unpinned_task", "revision 'main' is not a commit id"),
                                            ("tests.provision.invalid_task", "does not validate: .*missing ")])
def test_a_task_resolve_refuses_is_a_fault_a_retry_would_repeat(tmp_path, hub, module, reason):
    launcher = _launcher()
    with pytest.raises(StageFault, match=f"the provision role resolve refused task module {module}: .*{reason}") as caught:
        run_stage(ProvisionStage(), _ctx(_plan(tmp_path, module=module), launcher, tmp_path))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable, fault.evidence["resolve.exit_code"]) == ("infra", False, "3")
    assert _roles(launcher) == ["resolve"]


def _command(spec) -> str:
    """The role program command a launched step runs: ``resolve``, ``weights``, ``rows`` or ``tools``."""
    return spec.argv[spec.argv.index(ROLE_MODULE) + 1]


def test_resolve_never_gets_the_token(tmp_path, hub):
    """Resolve reads the task's declaration alone and asks the hub nothing, so the grant goes
    only to the roles that fill entries from the hub."""
    _, launcher, _ = _run(_plan(tmp_path), tmp_path, secrets={"HF_TOKEN": TOKEN})
    by_command = {_command(s): s.env_file for s in launcher.launched}
    assert sorted(by_command) == ["resolve", "rows", "weights"]
    assert by_command["resolve"] is None
    assert by_command["rows"] is not None and by_command["weights"] is not None


def test_the_token_reaches_the_roles_only_when_granted_and_never_on_argv(tmp_path, hub):
    plan = _plan(tmp_path)
    _, launcher, ctx = _run(plan, tmp_path, secrets={"HF_TOKEN": TOKEN})
    assert {c["token"] for c in fake_hub.calls(hub)} == {TOKEN}
    assert all(TOKEN not in " ".join(s.argv) for s in launcher.launched)
    fills = [s for s in launcher.launched if _command(s) != "resolve"]
    assert fills and all(s.env_file is not None and not s.env_file.exists() for s in fills)
    assert all(TOKEN not in json.dumps(dict(s.env)) for s in launcher.launched)
    hub_calls = len(fake_hub.calls(hub))
    plan2 = _plan(tmp_path / "other")
    _, bare, _ = _run(plan2, tmp_path / "other", job="101")
    ungranted = fake_hub.calls(hub)[hub_calls:]
    assert {c["token"] for c in ungranted} == {False}
    # The role itself switched off the implicit token before the hub library was imported.
    assert {c["implicit_token_off"] for c in ungranted} == {True}
    assert all(s.env_file is None for s in bare.launched)


def test_no_role_inherits_a_token_from_the_runners_environment(tmp_path, hub):
    """On a cluster a role step's environment is built afresh from the role's declared
    variables, so a token in the runner's own environment never reaches a provision role."""
    _, launcher, ctx = _run(_plan(tmp_path), tmp_path)
    runner_env = {"HF_TOKEN": "hf-runner-token", "HUGGING_FACE_HUB_TOKEN": "hf-runner-token",
                  "HF_HOME": "/home/u/.cache/huggingface", "PATH": "/bin", "HOME": "/home/u", "SLURM_JOB_ID": "100"}
    for spec in launcher.launched:
        env = step_env(runner_env, spec.env, tmpdir=tmp_path / "tmp")
        assert not any("hf-runner-token" in v for v in env.values())
        assert env["HF_HOME"] == _hf_home(ctx, spec.name)


def test_rows_only_skips_the_weights(tmp_path, hub):
    plan = _plan(tmp_path, weights=False)
    outputs, launcher, ctx = _run(plan, tmp_path)
    assert _roles(launcher) == ["resolve", "rows"]
    prov = _provisioned(ctx, outputs)
    assert prov.weights is None and prov.rows is not None
    assert set(outputs) == {"resolved", "provisioned"}


def test_a_concurrent_provision_waits_for_the_first_and_fills_nothing_twice(tmp_path, hub, monkeypatch):
    monkeypatch.setenv("FAKE_HUB_DELAY", "1.0")
    plans = [_plan(tmp_path, instance="r1"), _plan(tmp_path, instance="r2")]
    results: dict[int, tuple] = {}
    errors: list[BaseException] = []

    def go(i: int) -> None:
        try:
            results[i] = _run(plans[i], tmp_path, job=f"10{i}")
        except BaseException as e:  # noqa: BLE001 (reported below)
            errors.append(e)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
        time.sleep(0.3)
    for t in threads:
        t.join(60)
    assert errors == []
    launched = [name for _, launcher, _ in results.values() for name in _roles(launcher)]
    assert launched.count("weights-0") == 1 and launched.count("rows") == 1 and launched.count("resolve") == 2
    (o0, _, c0), (o1, _, c1) = results[0], results[1]
    assert _provisioned(c0, o0) == _provisioned(c1, o1)
    # A wait names the other provision, which held the entry's lock in its own job.
    waits = [(i, e) for i, c in enumerate((c0, c1))
             for e in map(json.loads, (c.attempt_dir / "events.jsonl").read_text().splitlines())
             if e["kind"] == "cache_wait"]
    assert waits
    for i, e in waits:
        assert (e["holder_purpose"], e["holder_job_id"]) == (f"provision {plans[1 - i].instance}", f"10{1 - i}")


ROWS_KEY = "rows-0123456789abcdef"


def _within(seconds: float, call) -> BaseException | None:
    """What *call* raised, run in a thread that must end within *seconds*."""
    raised: list[BaseException | None] = [None]

    def go() -> None:
        try:
            call()
        except BaseException as e:  # noqa: BLE001 (returned)
            raised[0] = e

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"still waiting after {seconds} s"
    return raised[0]


def test_waiting_for_another_provision_ends_at_the_deadline_or_a_stop(tmp_path, monkeypatch):
    """While another provision holds an entry's lock, the stage waits for it only until its
    deadline, or until the allocation stops, and then fails retryably having started nothing."""
    plan = _plan(tmp_path)
    launcher = _launcher()
    ctx = _ctx(plan, launcher, tmp_path)
    other = Holder.here("provision r0", job_id="99")
    GenerationLock(ctx.store, f"cache-locks/rows/{ROWS_KEY}", ttl_sec=600).try_acquire(other)
    attempt = provision_stage._Provision(ctx)
    ctx.deadline = time.time() + 1.5
    started = time.time()
    fault = _within(10, lambda: attempt.entry("rows", ROWS_KEY, ("rows",), role="rows"))
    # Ended at the deadline, not at the next five-second look.
    assert isinstance(fault, StageFault) and 1.4 <= time.time() - started < 4.0
    assert re.search("rows/rows-0123456789abcdef: still filled by provision r0 .* at the deadline", str(fault))
    assert fault.fault.fault_class == "infra" and fault.fault.retryable
    ctx.deadline = time.time() + 60
    monkeypatch.setattr(provision_stage, "POLL_SEC", 1.0)
    threading.Timer(0.5, ctx.stop_roles).start()
    started = time.time()
    fault = _within(10, lambda: attempt.entry("rows", ROWS_KEY, ("rows",), role="rows"))
    # Noticed at the next look, not at the deadline a minute away.
    assert isinstance(fault, StageFault) and "stopped while provision r0" in str(fault) and fault.fault.retryable
    assert time.time() - started < 3.0
    assert launcher.launched == []
    waits = [e for e in map(json.loads, (ctx.attempt_dir / "events.jsonl").read_text().splitlines())
             if e["kind"] == "cache_wait"]
    assert [(e["entry_kind"], e["key"], e["holder_purpose"], e["holder_job_id"], e["holder_pid"]) for e in waits] \
        == [("rows", ROWS_KEY, "provision r0", "99", other.pid)] * 2


def test_a_corrupt_published_entry_is_a_fault_a_retry_would_repeat(tmp_path, hub):
    plan = _plan(tmp_path)
    store = PosixStore(plan.store_root)
    (store.root / "cache" / "weights" / weights_key("org/model", SHA)).mkdir(parents=True)
    with pytest.raises(StageFault, match="cache entry weights/.* is corrupt: .* no ENTRY.json") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable


GOOD_IDS = b'{"dev": ["1", "2"], "test": ["3"]}'


@pytest.mark.parametrize("ids, damage, why", [
    (None, None, "its manifest lists no ids.json"),
    (GOOD_IDS, lambda f: f.unlink(), "No such file"),
    (GOOD_IDS, lambda f: f.write_bytes(b'{"dev": [], "test": []}'), "differs from its manifest"),
    (b"{", None, "the file: Invalid JSON"),
    (b'{"dev": ["1"]}', None, "test: Field required"),
    (b'{"dev": "1", "test": [], "x": 1}', None, "x: Extra inputs are not permitted; dev: Input should be a valid"),
])
def test_a_rows_entry_whose_ids_cannot_be_used_keys_no_bundle(tmp_path, hub, ids, damage, why):
    """The rows entry was published by hand, so only the resolve role runs before the stage
    reads its ``ids.json``; a retry would read the same entry."""
    key = rows_key(provisioning_digest(unit_descriptions(tool_task.TASK)))
    files = {"tenants/t0/dev/rows.json": b"[]", "tenants/t0/test/rows.json": b"[]"}
    if ids is not None:
        files["ids.json"] = ids
    plan = _plan(tmp_path, module="tests.provision.tool_task", weights=False)
    store = PosixStore(plan.store_root)

    def fill(d: Path) -> None:
        for rel, data in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(data)

    rows = populate(ContentCache(store), "rows", key, fill, holder=Holder.here("test")).entry
    if damage is not None:
        damage(rows.path / "ids.json")
    launcher = _launcher()
    with pytest.raises(StageFault, match=re.escape(f"cache entry rows/{key}: ids.json cannot be used: ")) as caught:
        run_stage(ProvisionStage(), _ctx(plan, launcher, tmp_path))
    assert why in str(caught.value)
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", False)
    assert _roles(launcher) == ["resolve"]


def test_provision_takes_no_gpu_and_a_python_module_name(tmp_path):
    gpu = StageDecl("provision", "provision", ResourceShape(gpus=1, gpu_kind="H200", cpus=4, memory_gib=16,
                                                            time_sec=600),
                    {"task": {"module": "tests.provision.tasks"}})
    with pytest.raises(PlanError, match="asks for no GPU"):
        noop_plan(tmp_path, decls=[gpu])
    path = StageDecl("provision", "provision", SHAPE, {"task": {"module": "tasks/multitenant_slo/task.py"}})
    with pytest.raises(PlanError, match=r"task\.module"):
        noop_plan(tmp_path, decls=[path])
    assert "provision" in KNOWN_STAGE_NAMES


@pytest.mark.parametrize("reserve", [4, 0])
def test_each_role_leaves_the_runner_its_memory_and_fits_the_stage(tmp_path, hub, reserve):
    plan = _plan(tmp_path, runner_memory_gib=reserve)
    stage = plan.stage("provision").shape
    _, launcher, _ = _run(plan, tmp_path)
    assert _roles(launcher) == ["resolve", "weights-0", "rows"]
    for spec in launcher.launched:
        assert spec.shape.memory_gib == stage.memory_gib - reserve and spec.shape.disk_gib == 0
        assert spec.shape.model_copy(update={"memory_gib": stage.memory_gib,
                                             "disk_gib": stage.disk_gib}) == stage
        assert budget_problem(stage, [(spec.shape, False)], runner_memory_gib=reserve) is None
    # The stage's whole shape, as each role had it before the reserve, does not fit.
    assert budget_problem(stage, [(stage, False)], runner_memory_gib=4) is not None


def test_a_cold_provision_runs_on_a_node_with_exactly_the_stages_disk(tmp_path, hub):
    """No role declares disk: the stage's, which the runner checks when the job starts, is the
    one budget. The node has exactly the stage's disk, and every role's grant is checked as the
    step wrapper checks it, while the scratch holds what the roles before it left there: each role
    logs on the node, and a role given the hub token has its env file written before it starts.
    Resolve, the first role, gets no token."""
    disk = {"noop": {"provision": SHAPE.model_copy(update={"disk_gib": 20}).model_dump()}}
    plan = _plan(tmp_path, resources=disk)
    assert plan.stage("provision").shape.disk_gib == 20
    launcher = LocalLauncher(gpus=0, replace_modules={"hls.provision.role": FAKE_ROLE}, node_disk_gib=20,
                             scratch=tmp_path / "node" / "hls-100")
    ctx = _ctx(plan, launcher, tmp_path, secrets={"HF_TOKEN": TOKEN})
    prov = _provisioned(ctx, run_stage(ProvisionStage(), ctx))
    assert prov.weights is not None and prov.rows is not None
    assert _roles(launcher) == ["resolve", "weights-0", "rows"]
    assert {s.shape.disk_gib for s in launcher.launched} == {0}
    # Each role logged on the node; the attempt's logs hold the scrubbed copies.
    logs = ctx.attempt_dir / "logs"
    reports = [RoleReport.model_validate_json((logs / f"{s.name}.report.json").read_bytes())
               for s in launcher.launched]
    assert [r.mismatches for r in reports] == [[], [], []]
    assert all(r.granted.disk_free_gib < 20 for r in reports[1:]), "the scratch held nothing when a role started"


@pytest.mark.parametrize("memory_gib, reserve", [(4, 4), (3, 4), (5, 8)])
def test_a_stage_whose_memory_leaves_its_roles_none_is_refused_when_the_plan_is_made(tmp_path, memory_gib, reserve):
    small = {"provision": {"cpus": 4, "memory_gib": memory_gib, "time_sec": 600}}
    with pytest.raises(PlanError, match=f"declares {memory_gib} GiB, which leaves its roles none"):
        noop_plan(tmp_path, decls=provision_stages(task_ref=TaskRef(module="tests.provision.tasks"), shape=SHAPE),
                  resources={"noop": small}, runner_memory_gib=reserve)
    assert noop_plan(tmp_path, decls=provision_stages(task_ref=TaskRef(module="tests.provision.tasks"), shape=SHAPE),
                     resources={"noop": {"provision": {**small["provision"], "memory_gib": reserve + 1}}},
                     runner_memory_gib=reserve).stage("provision").shape.memory_gib == reserve + 1


def test_the_record_states_the_disk_each_entrys_copy_takes_and_the_cache_alone_gives_the_same_record(tmp_path,
                                                                                                     hub):
    """Each entry of ``provisioned.json`` states the disk its node-local copy takes
    (``copy_disk_bytes``), and the planner, which reads the cache alone before any stage runs,
    finds the same record (``cached_provision``): the stages' declared disk counts their data's
    copies from it."""
    import importlib

    from hls.stages.provision import NotProvisioned, cached_provision

    task = importlib.import_module("tests.provision.tool_task").TASK
    cache = ContentCache(PosixStore(_plan(tmp_path, module="tests.provision.tool_task").store_root))
    with pytest.raises(NotProvisioned) as missing:
        cached_provision(task, cache)
    # The dev bundle is keyed by the rows' ids, so it is named only once the rows are there.
    assert missing.value.missing == [f"weights/{weights_key('org/model', SHA)}", f"rows/{_rows_key(task)}",
                                     "tools/tiny-tiny-v1-full"]
    outputs, _, ctx = _run(_plan(tmp_path, module="tests.provision.tool_task"), tmp_path)
    prov = _provisioned(ctx, outputs)
    refs = [("weights", prov.weights), ("rows", prov.rows), *(("tools", t) for t in prov.tools)]
    for kind, ref in refs:
        assert ref.disk_bytes == copy_disk_bytes(cache.lookup(kind, ref.key)) > 0
    assert cached_provision(task, cache) == prov
    rows_only = cached_provision(task, cache, weights=False)
    assert rows_only.weights is None and rows_only.rows == prov.rows


def test_an_entry_whose_records_carry_no_mtimes_is_not_provisioned_for_the_planner(tmp_path):
    """An entry whose file records carry no times is absent (``lookup``): the
    planner's record names it among what the cache lacks, so the plan names the provision to run,
    which fills it again."""
    from hls.stages.provision import NotProvisioned, cached_provision
    from tests.provision.cached import provision_into

    task = tool_task.TASK
    store = PosixStore(tmp_path / "store")
    provision_into(store.root, task)
    mf = store.path(f"cache/rows/{_rows_key(task)}") / "ENTRY.json"
    manifest = json.loads(mf.read_text())
    mf.write_text(json.dumps({**manifest, "files": [{k: v for k, v in f.items() if k != "mtime_ns"}
                                                    for f in manifest["files"]]}))
    with pytest.raises(NotProvisioned) as missing:
        cached_provision(task, ContentCache(store))
    assert missing.value.missing == [f"rows/{_rows_key(task)}"]


def _rows_key(task) -> str:
    return rows_key(provisioning_digest(unit_descriptions(task)))
