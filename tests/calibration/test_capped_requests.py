"""``hls capped-requests``: the sessions a calibration's pool member ran to the output cap, read
from the published captures of a calibration instance (``hls.pipeline.calibration.capped_requests``),
each checked against its manifest, keyed by the rows the instance provisioned; and the command,
which lists them and hands them to the task's own recorder with --write."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.core.payloads import Envelope
from hls.core.trace import ArtifactResponse
from hls.calibrate import (DEFAULT_DELTA, DEFAULT_EPS, DEFAULT_MU_PER_MTOK, CappedRows, CaptureStore, capture_items,
                           load_sessions, serving_seed_of)
from hls.pipeline import calibration, commands
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from hls.pipeline.plan import Plan
from hls.provision.model import Provisioned, RowsRef, rows_key
from hls.spec.datasets import Split
from hls.stages.calibration import InstrumentParams
from hls.store.cache import ContentCache, copy_disk_bytes, entry_sha256
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.verify import calibration_roles as roles
from hls.verify.calibration import DEFAULT_EOT_EXPECTED_MAX
from tests.calibration import engines, fixture
from tests.calibration.published import create, publish
from tests.provision.warmer_config import write_config
from tests.provision.cached import provision_task
from tests.store.entries import populate

ROWS = rows_key("fedcba9876543210")


def _instance(tmp_path: Path, rows: Path, *, module: str = "cal_capped.task", draws: tuple[int, ...] = (0, 2, 5),
              member: engines.FakeMember | None = None,
              rows_entry_sha256: str | None = None) -> tuple[Plan, PosixStore]:
    """A calibration of vllm whose provision stage and the capture stages of *draws* published,
    each capture *member*'s (the fake member's by default) of the task's test split. The provision
    record names the rows entry by *rows_entry_sha256*, the cache entry's own digest by default. The
    task's provisioning is in the cache before the plan, which reads it."""
    provision_task(tmp_path / "store", module)
    plan = calibration.plan_calibrate(
        config=write_config(tmp_path), overrides={}, task_module=module, engines=["vllm"], fanout=2,
        servings_per_engine=None, sessions_per_draw=None, eps=DEFAULT_EPS, mu_per_mtok=DEFAULT_MU_PER_MTOK,
        delta=DEFAULT_DELTA, eot_expected_max=DEFAULT_EOT_EXPECTED_MAX, max_servings_per_engine=24,
        job_name="hls-calibrate", echo=[].append)
    store = PosixStore(plan.store_root)
    plan = create(store, plan)

    def fill_rows(d: Path) -> None:
        (d / "tenants" / "t0" / "test").mkdir(parents=True)
        (d / "tenants" / "t0" / "test" / "rows.json").write_text(json.dumps(fixture.TEST_ROWS))

    entry = populate(ContentCache(store), "rows", ROWS, fill_rows, holder=Holder.here("test")).entry
    named = rows_entry_sha256 or entry_sha256(entry)
    provisioned = Provisioned(weights=None, rows=RowsRef(key=ROWS, entry_sha256=named, tenants=["t0"],
                                                         disk_bytes=copy_disk_bytes(entry)), tools=[])
    publish(store, plan, "provision", {"provisioned": ("provisioned.json", provisioned.model_dump_json().encode())})
    task = sys.modules[module].TASK
    sessions = load_sessions(task, rows, Split.TEST)
    fingerprint = InstrumentParams.model_validate(plan.stage("instrument").params).campaign.workload_fingerprint
    made = CaptureStore(tmp_path / "made", fingerprint)
    for draw in draws:
        trace = asyncio.run((member or engines.FakeMember()).serve_traffic(
            sessions, seed=serving_seed_of(task.seed, draw), request_timeout_sec=REQUEST_TIMEOUT_SEC))
        made.write_capture("vllm", draw, serving_seed_of(task.seed, draw), engines.MEMBER_VERSION,
                           capture_items(trace, engine="vllm", draw=draw))
        publish(store, plan, f"capture_vllm_{draw}", {"capture": (roles.capture_file(fingerprint, "vllm", draw),
                                                                  made.capture_path("vllm", draw).read_bytes())})
    return plan, store


def test_capped_requests_reads_an_instances_captures(tmp_path, login, in_catalog, rows):
    plan, store = _instance(tmp_path, rows)
    found = calibration.capped_requests("cal_capped.task", instance=plan.instance, engine="vllm", store=store)
    # Every reply of the fake member runs to the task's 8-token cap, so every session is listed.
    served = len(fixture.TEST_ROWS)
    assert found == {"t0": CappedRows(rows=tuple(sorted(r["q"] for r in fixture.TEST_ROWS)), capped_turns=served,
                                      served_rows=served, served_turns=served)}


class _ShortOnSomeRows(engines.FakeMember):
    """The fake member, answering rows 1 and 3 with one token short of the cap in every serving."""

    async def serve_traffic(self, loaded, **kw):
        trace = await super().serve_traffic(loaded, **kw)
        for ev in trace.events:
            if ev.session_id in ("1", "3"):
                [reply] = ev.response.output.contents
                ev.response = ArtifactResponse(output=Envelope(request_id=ev.request_id, contents=[
                    reply.model_copy(update={"tokens": reply.tokens[:-1], "finish_reason": "stop"})]))
        return trace


def test_only_the_sessions_that_reached_the_cap_are_listed(tmp_path, login, in_catalog, rows):
    plan, store = _instance(tmp_path, rows, member=_ShortOnSomeRows())
    found = calibration.capped_requests("cal_capped.task", instance=plan.instance, engine="vllm", store=store)
    served = len(fixture.TEST_ROWS)
    assert found == {"t0": CappedRows(rows=("2", "4"), capped_turns=2, served_rows=served, served_turns=served)}


def test_a_capture_that_is_not_its_manifests_is_refused(tmp_path, login, in_catalog, rows):
    plan, store = _instance(tmp_path, rows)
    key = read_manifest(store, plan.layout, "capture_vllm_2")[0].outputs["capture"].key
    store.path(key).write_bytes(store.path(key).read_bytes() + b"\0")
    with pytest.raises(ValueError, match=f"{key} is not the capture its manifest recorded"):
        calibration.capped_requests("cal_capped.task", instance=plan.instance, engine="vllm", store=store)


def test_a_provision_record_that_is_not_its_manifests_is_refused(tmp_path, login, in_catalog, rows):
    plan, store = _instance(tmp_path, rows)
    key = read_manifest(store, plan.layout, "provision")[0].outputs["provisioned"].key
    store.path(key).write_bytes(store.path(key).read_bytes() + b" ")
    with pytest.raises(ValueError, match=re.escape(f"{key} is not the provisioned.json its manifest recorded")):
        calibration.capped_requests("cal_capped.task", instance=plan.instance, engine="vllm", store=store)


def test_rows_that_are_not_the_entry_the_instance_provisioned_are_refused(tmp_path, login, in_catalog, rows):
    plan, store = _instance(tmp_path, rows, rows_entry_sha256="0" * 64)
    with pytest.raises(ValueError, match=re.escape(f"cache entry rows/{ROWS} is not the one {plan.instance} "
                                                   "provisioned")):
        calibration.capped_requests("cal_capped.task", instance=plan.instance, engine="vllm", store=store)


def test_an_instance_of_another_pipeline_is_refused(tmp_path, login, in_catalog, rows):
    _, store = _instance(tmp_path, rows)
    warmer = create(store, commands.plan_provision(config=write_config(tmp_path), overrides={},
                                                   task_module="cal_capped.task", weights=True,
                                                   job_name="hls-provision", echo=[].append))
    with pytest.raises(ValueError, match=re.escape(f"{warmer.instance} is no calibration of {fixture.TASK_ID}")):
        calibration.capped_requests("cal_capped.task", instance=warmer.instance, engine="vllm", store=store)


@pytest.mark.parametrize(("module", "engine", "refusal"), [
    pytest.param("cal_flips.task", "vllm", "calibrated cal_capped.task, not cal_flips.task", id="other-module"),
    pytest.param("cal_capped.task", "sglang", "served no sglang; its pool is ['vllm']", id="other-engine"),
])
def test_capped_requests_refuses_what_the_instance_did_not_serve(tmp_path, login, in_catalog, rows, module, engine,
                                                                 refusal):
    plan, store = _instance(tmp_path, rows)
    with pytest.raises(ValueError, match=refusal.replace("[", r"\[").replace("]", r"\]")):
        calibration.capped_requests(module, instance=plan.instance, engine=engine, store=store)


def test_the_command_lists_the_rows_and_hands_them_to_the_tasks_recorder(tmp_path, login, in_catalog, rows):
    plan, _ = _instance(tmp_path, rows)
    fixture.RECORDED.clear()
    result = CliRunner().invoke(cli.app, ["capped-requests", "cal_capped.task", "--from", plan.instance,
                                          "--engine", "vllm", "--write", "--config", str(tmp_path / "hls.toml")])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        f"=== capped requests under vllm ({plan.instance}) ===",
        "t0: 4 of 4 sessions removed (4 turns at the cap, 4 turns served)",
        "    '1',", "    '2',", "    '3',", "    '4',",
        "t0: 4 row(s) added to the list", "wrote removed_requests.json"]
    assert fixture.RECORDED == [{"t0": ("1", "2", "3", "4")}]


def test_the_command_lists_the_rows_and_leaves_the_tasks_list_alone_without_write(tmp_path, login, in_catalog, rows):
    plan, _ = _instance(tmp_path, rows)
    fixture.RECORDED.clear()
    result = CliRunner().invoke(cli.app, ["capped-requests", "cal_capped.task", "--from", plan.instance,
                                          "--engine", "vllm", "--config", str(tmp_path / "hls.toml")])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        f"=== capped requests under vllm ({plan.instance}) ===",
        "t0: 4 of 4 sessions removed (4 turns at the cap, 4 turns served)",
        "    '1',", "    '2',", "    '3',", "    '4',"]
    assert fixture.RECORDED == []


def test_the_command_refuses_write_for_a_task_that_owns_no_list(tmp_path, login, in_catalog, rows, monkeypatch):
    plan, _ = _instance(tmp_path, rows)
    monkeypatch.delattr(sys.modules["cal_capped.task"], "record_removed_rows")
    result = CliRunner().invoke(cli.app, ["capped-requests", "cal_capped.task", "--from", plan.instance, "--write",
                                          "--config", str(tmp_path / "hls.toml")])
    assert result.exit_code == 2 and "record_removed_rows" in result.output


def test_the_command_says_a_refusal_in_one_line_and_exits_1(tmp_path, login, in_catalog, rows):
    plan, _ = _instance(tmp_path, rows)
    result = CliRunner().invoke(cli.app, ["capped-requests", "cal_capped.task", "--from", plan.instance, "--engine",
                                          "sglang", "--config", str(tmp_path / "hls.toml")])
    assert result.exit_code == 1
    [line] = result.stderr.splitlines()
    assert line == f"hls capped-requests: {plan.instance} served no sglang; its pool is ['vllm']"
