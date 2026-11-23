from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from hls.pipeline.events import EventLog
from hls.pipeline.stage import StageContext
from hls.spec.resources import ResourceShape
from hls.store.posix import PosixStore
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan

# Two of them fit in noop_a's stage (4 CPUs, 6 GiB) beside the runner's 4 GiB.
SHAPE = ResourceShape(cpus=2, memory_gib=1, time_sec=60)


def _ctx(tmp_path: Path) -> StageContext:
    plan = noop_plan(tmp_path)
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("noop_a", 1, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage("noop_a"), attempt=1, launcher=LocalLauncher(),
                        watch=None, events=events, inputs={}, deadline=time.time() + 60,
                        scratch=tmp_path / "node" / "hls-100", job_id="100")


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _still_running(handle) -> bool:
    """Whether the role still runs a second on, long enough for a stop the wait asked for to
    have ended it."""
    time.sleep(1.0)
    return handle.poll() is None


def test_waiting_for_named_roles_leaves_the_others_running(tmp_path):
    ctx = _ctx(tmp_path)
    server = ctx.start_role("server", _py("import time; time.sleep(60)"), SHAPE, owner="harness")
    ctx.start_role("driver", _py("import sys; sys.exit(3)"), SHAPE, owner="harness")
    started = time.time()
    [outcome] = ctx.wait_roles("driver")
    assert (outcome.name, outcome.ending, outcome.exit_code) == ("driver", "exited", 3)
    assert time.time() - started < 15 and _still_running(server)
    ctx.stop_roles()
    [rest] = ctx.wait_roles()
    assert (rest.name, rest.ending) == ("server", "stopped") and server.poll() is not None


def test_a_role_already_waited_for_or_never_started_cannot_be_waited_for(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.start_role("quick", _py("pass"), SHAPE, owner="harness")
    ctx.wait_roles("quick")
    with pytest.raises(ValueError, match="quick"):
        ctx.wait_roles("quick")
    with pytest.raises(ValueError, match="nobody"):
        ctx.wait_roles("nobody")
    assert ctx.wait_roles() == []


def test_named_roles_are_waited_for_in_the_order_given(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.start_role("a", _py("pass"), SHAPE, owner="harness")
    ctx.start_role("b", _py("pass"), SHAPE, owner="harness")
    assert [o.name for o in ctx.wait_roles("b", "a")] == ["b", "a"]


def test_a_named_role_that_faults_leaves_the_rest_to_the_next_wait_which_stops_them(tmp_path):
    """The driver's fault ends the stage, so the next wait stops the server it was driving; the
    wait for the driver alone does not, so its outcome is read first."""
    ctx = _ctx(tmp_path)
    server = ctx.start_role("server", _py("import time; time.sleep(60)"), SHAPE, owner="harness")
    ctx.start_role("driver", _py("import sys; sys.exit(1)"), SHAPE, owner="harness")
    [driver] = ctx.wait_roles("driver")
    assert driver.ending == "exited" and _still_running(server)
    [rest] = ctx.wait_roles()
    assert (rest.name, rest.ending) == ("server", "stopped") and server.poll() is not None


def test_a_role_named_twice_is_refused_before_anything_is_waited_for(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.start_role("a", _py("pass"), SHAPE, owner="harness")
    with pytest.raises(ValueError, match="more than once"):
        ctx.wait_roles("a", "a")
    assert [o.name for o in ctx.wait_roles("a")] == ["a"]
