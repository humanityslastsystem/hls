"""How the build sandbox's commands are counted against their process ceiling when the site does
not say: the node's kernel release decides, and the stage records which it used and why."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import hls.pipeline.stage as stage_module
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.plan import Plan
from hls.pipeline.stage import StageContext, StageFault
from hls.store.posix import PosixStore
from hls.substrate.nproc import AccountingChoice, choose_accounting, kernel_accounting
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan


@pytest.mark.parametrize("release, accounting", [
    ("4.18.0-553.el8_10.x86_64", "uid"),
    ("5.4.0-150-generic", "uid"),
    ("5.9.16", "uid"),
    ("5.13.19-2-pve", "uid"),
    ("5.14.0-427.13.1.el9_4.x86_64", "user_namespace"),
    ("5.15.0-1-generic", "user_namespace"),
    ("6.8.0-45-generic", "user_namespace"),
    ("10.0.0", "user_namespace"),
])
def test_the_kernel_release_says_how_the_commands_are_counted(release, accounting):
    assert kernel_accounting(release) == accounting


@pytest.mark.parametrize("release", ["", "linux", "v5.14", "5", "5-14"])
def test_a_release_without_a_version_names_the_setting_and_both_values(release):
    with pytest.raises(ValueError) as e:
        kernel_accounting(release)
    assert 'site.nproc_accounting to "uid"' in str(e.value) and '"user_namespace"' in str(e.value)


def test_the_sites_setting_wins_over_the_kernel_release():
    assert choose_accounting("uid", "6.8.0") == AccountingChoice(accounting="uid", from_site=True, kernel="6.8.0")
    assert choose_accounting(None, "6.8.0") == AccountingChoice(accounting="user_namespace", from_site=False,
                                                                kernel="6.8.0")
    assert choose_accounting(None, "4.18.0") == AccountingChoice(accounting="uid", from_site=False, kernel="4.18.0")


def _ctx(tmp_path: Path, site: dict | None = None) -> tuple[Plan, StageContext]:
    plan = noop_plan(tmp_path, site=site)
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("noop_a", 1, "events.jsonl")))
    ctx = StageContext(store=store, plan=plan, stage=plan.stage("noop_a"), attempt=1, launcher=LocalLauncher(),
                       watch=None, events=events, inputs={}, deadline=time.time() + 60)
    return plan, ctx


def _recorded(ctx: StageContext) -> list[dict]:
    return [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "nproc_accounting"]


def test_a_site_that_says_nothing_is_counted_as_its_nodes_kernel_counts_and_the_choice_is_recorded(
        tmp_path, monkeypatch):
    plan, ctx = _ctx(tmp_path)
    assert plan.settings.site.nproc_accounting is None
    monkeypatch.setattr(stage_module, "kernel_release", lambda: "5.14.0-427.el9.x86_64")
    assert ctx.nproc_accounting() == AccountingChoice(accounting="user_namespace", from_site=False,
                                                      kernel="5.14.0-427.el9.x86_64")
    # Chosen once: a later reading of another release changes nothing.
    monkeypatch.setattr(stage_module, "kernel_release", lambda: "4.18.0")
    assert ctx.nproc_accounting().accounting == "user_namespace"
    [event] = _recorded(ctx)
    assert (event["accounting"], event["source"], event["kernel"]) == (
        "user_namespace", "kernel release", "5.14.0-427.el9.x86_64")


def test_a_pre_5_14_node_keeps_the_uid_accounting_a_site_set_no_value_for(tmp_path, monkeypatch):
    _, ctx = _ctx(tmp_path)
    monkeypatch.setattr(stage_module, "kernel_release", lambda: "4.18.0-553.el8_10.x86_64")
    assert ctx.nproc_accounting().accounting == "uid"


def test_a_site_that_says_is_counted_as_it_says_and_that_is_recorded(tmp_path, monkeypatch):
    _, ctx = _ctx(tmp_path, site={"nproc_accounting": "uid"})
    monkeypatch.setattr(stage_module, "kernel_release", lambda: "6.8.0-45-generic")
    assert ctx.nproc_accounting() == AccountingChoice(accounting="uid", from_site=True, kernel="6.8.0-45-generic")
    [event] = _recorded(ctx)
    assert (event["accounting"], event["source"]) == ("uid", "site.nproc_accounting")


def test_a_release_that_says_nothing_refuses_the_stage_naming_the_setting(tmp_path, monkeypatch):
    _, ctx = _ctx(tmp_path)
    monkeypatch.setattr(stage_module, "kernel_release", lambda: "custom")
    with pytest.raises(StageFault) as e:
        ctx.nproc_accounting()
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert "site.nproc_accounting" in fault.reason and "'custom'" in fault.reason
