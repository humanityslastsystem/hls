"""The engine host ends a role whose engine died, or whose engine's load or release does not
return within its bound, and reaps the processes its engine left (``hls.verify.inprocess``).

Each scenario runs as a program of its own (``tests.stages_lib.engine_watch_prog``), since the host
ends the process it runs in. Without the host's watch and bounds, every scenario but the healthy
ones waits for ever, which the test reads as a program still running after ``HUNG_SEC``."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

import pytest

from hls.core.engine import Engine, ResourceRequest
from hls.verify import inprocess
from hls.verify.inprocess import EngineHost, PlacementError
from tests.stages_lib import engine_watch_prog as prog

REPO = Path(__file__).resolve().parents[2]
#: How long a scenario may run before the test calls it hung. The host ends each within a few
#: seconds of its engine's death or of its bound, as measured below (``ENDS_WITHIN_SEC``).
HUNG_SEC = 60
#: How long a scenario the host ends may take in all: the program's start and its fake engine's
#: process's (a spawned interpreter that imports hls, about a second each), the death or the bound,
#: and the host's next check, which comes a second later at most.
ENDS_WITHIN_SEC = 20
#: How faulthandler heads each thread's stack.
STACK_HEAD = "(most recent call first)"


def _run(scenario: str) -> tuple[subprocess.CompletedProcess, float]:
    """Run *scenario*; what it did, and how long it took."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO)])}
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-m", "tests.stages_lib.engine_watch_prog", scenario], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=HUNG_SEC)
    return done, time.monotonic() - started


def _child(done: subprocess.CompletedProcess) -> int:
    """The pid of the fake engine's process, as the program printed it."""
    [pid] = re.findall(r"^child (\d+)$", done.stdout, re.MULTILINE)
    return int(pid)


def _last_line(done: subprocess.CompletedProcess) -> str:
    return done.stderr.rstrip("\n").splitlines()[-1]


@pytest.mark.parametrize(("scenario", "status", "prefix"), [
    pytest.param("dies-loading-default", 1, "hls-engine-host: ", id="the-hosts-own"),
    pytest.param("dies-loading", prog.GIVE_UP.exit_code, prog.GIVE_UP.prefix, id="the-roles"),
])
def test_an_engine_whose_process_exits_while_it_loads_ends_its_role(scenario, status, prefix):
    # The load waits on its process for ever, as SGLang's constructor did once its peers were gone,
    # so only the host's watch ends the role, with the status and line of the role's give-up.
    done, took = _run(scenario)
    pid = _child(done)
    assert (done.returncode, "finished" in done.stdout) == (status, False)
    assert took < ENDS_WITHIN_SEC
    assert _last_line(done) == (f"{prefix}EngineDied: dies (FakeServing) died while loading: process {pid} "
                                f"(fake-scheduler) exited with code {prog.CHILD_EXIT}")
    # What the process wrote last is in the role's own log, right above the report.
    assert prog.LAST_WORDS in done.stderr
    # A death is not a hang: no stacks.
    assert STACK_HEAD not in done.stderr


def test_an_engine_whose_process_is_killed_while_it_serves_ends_its_role():
    done, took = _run("killed-held")
    pid = _child(done)
    assert (done.returncode, "finished" in done.stdout) == (prog.GIVE_UP.exit_code, False)
    assert took < ENDS_WITHIN_SEC
    assert _last_line(done) == (f"{prog.GIVE_UP.prefix}EngineDied: killed (FakeServing) died after it loaded: "
                                f"process {pid} (fake-scheduler) was killed by signal 9 (SIGKILL)")


def test_an_engine_that_reports_itself_dead_while_it_serves_ends_its_role():
    # Its process still runs, so the host ends it before the role goes, and says so.
    done, took = _run("unhealthy-held")
    pid = _child(done)
    assert (done.returncode, "finished" in done.stdout) == (prog.GIVE_UP.exit_code, False)
    assert took < ENDS_WITHIN_SEC
    assert _last_line(done) == (f"{prog.GIVE_UP.prefix}EngineDied: unhealthy (FakeServing) died after it loaded: "
                                f"it reports: the fake engine reports its scheduler gone; the host killed its processes "
                                f"still running: process {pid} (fake-scheduler)")


def test_a_healthy_engine_runs_to_its_release_and_leaves_no_zombie():
    # Its stop kills its process and returns without waiting or reaping, as SGLang's reaps nothing
    # and vLLM's kills what outlived its grace: the host waits for it to end and reaps it, and does
    # not take a process still dying of its stop's kill for one the stop left.
    done, _ = _run("healthy")
    pid = _child(done)
    assert (done.returncode, done.stdout.splitlines()[-2:]) == (0, [f"reaped {pid}", "finished"]), done.stderr
    assert "EngineDied" not in done.stderr and STACK_HEAD not in done.stderr
    assert "the host killed it" not in done.stderr


def test_a_process_its_engines_stop_leaves_running_is_ended_and_reaped():
    done, _ = _run("stop-leaves-it")
    pid = _child(done)
    assert (done.returncode, done.stdout.splitlines()[-2:]) == (0, [f"reaped {pid}", "finished"]), done.stderr
    assert f"the stop of healthy left process {pid} (fake-scheduler) running; the host killed it" in done.stderr


@pytest.mark.parametrize(("scenario", "line", "frame"), [
    pytest.param("free-hangs", "EngineReleaseTimeout: freeing what lingering's load held did not return within 1.5 s",
                 "__del__", id="a-finalizer"),
    pytest.param("stop-hangs", "EngineReleaseTimeout: lingering's stop did not return within 1.5 s", "_unload",
                 id="the-stop"),
    pytest.param("load-hangs", "EngineLoadTimeout: hangs did not load within 1.5 s", "_load", id="the-load"),
])
def test_a_step_that_does_not_return_within_its_bound_ends_its_role_with_every_stack(scenario, line, frame):
    # A finalizer that waits for ever, as a ZMQ context did whose sockets held messages for peers
    # the engine's stop had killed, holds a release that frees what the load held.
    done, took = _run(scenario)
    assert (done.returncode, "finished" in done.stdout) == (prog.GIVE_UP.exit_code, False)
    assert took < ENDS_WITHIN_SEC
    assert _last_line(done) == f"{prog.GIVE_UP.prefix}{line}"
    # Every thread's stack is in the role's log, the step that hung among them.
    assert STACK_HEAD in done.stderr and "_term_forever" in done.stderr and frame in done.stderr


class _Serving(Engine):
    """An engine that says it runs processes of its own, and starts none."""

    runs_processes: ClassVar[bool] = True

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1)

    def _load(self) -> None:
        self.model = object()


def test_a_second_engine_that_runs_processes_is_refused_while_one_is_held():
    # The host takes every process that starts while such an engine loads as the engine's own.
    async def go():
        host = EngineHost(devices=0)
        async with host:
            await host.acquire(_Serving(), name="first")
            with pytest.raises(PlacementError, match="_Serving runs processes of its own, as first does"):
                await host.acquire(_Serving(), name="second")
            assert host.outstanding == 1
        return host.outstanding

    assert asyncio.run(go()) == 0


def test_the_hosts_bounds_and_cadence():
    # A load is unbounded unless the role bounds it; every release step is bounded, and how the
    # process ends is the role's to say.
    host = EngineHost()
    assert (host.load_sec, host.release_sec, host.give_up) == (None, inprocess.RELEASE_STEP_SEC,
                                                               inprocess.ANY_FAILURE)
    # A check costs a non-blocking wait per process: every second costs nothing measurable.
    assert 1 <= inprocess.ENGINE_POLL_SEC <= 5


@pytest.mark.parametrize("bounds", [{"load_sec": 0}, {"release_sec": -1.0}], ids=["load", "release"])
def test_a_bound_is_a_positive_number_of_seconds(bounds):
    with pytest.raises(ValueError, match="a bound is a positive number of seconds"):
        EngineHost(**bounds)
