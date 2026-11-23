"""The per-trajectory sandbox: its session lifetime, and what it does when a command
or the container itself goes wrong."""
from __future__ import annotations

import asyncio
import os
import shutil

import pytest

from hls.spec.sandbox import (
    SERVER_READY_POLL_SEC,
    BASH_TOOL, SANDBOX_ERROR_EXIT_CODE, TIMEOUT_EXIT_CODE, SandboxToolRunner,
)
from hls.spec.tools import DEVICE, HostPath


class FakeExecResult:
    def __init__(self, stdout="", stderr="", return_code=0):
        self.stdout, self.stderr, self.return_code = stdout, stderr, return_code


class FakeEnv:
    """Stands in for a harbor environment: records what was exec'd, replies from a
    scripted map, and records whether it was started and stopped.

    ``exec`` before ``start`` raises the way a real backend does, so a runner that
    handed back an unstarted container would fail these tests rather than pass them
    against a fake more forgiving than the thing it stands for."""

    def __init__(self, spec, replies=None, fail=None, start_error=None):
        self.spec = spec
        self.replies = dict(replies or {})
        self.fail = fail
        self.start_error = start_error
        self.calls: list[tuple[str, str | None, int | None]] = []
        self.started = False
        self.force_build = None
        self.stopped = False

    async def start(self, force_build):
        if self.start_error is not None:
            raise self.start_error
        self.force_build = force_build
        self.started = True

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if not self.started:
            raise RuntimeError("environment not started")
        self.calls.append((command, cwd, timeout_sec))
        if self.fail is not None:
            raise self.fail
        return self.replies.get(command, FakeExecResult(stdout="ok"))

    async def stop(self, delete=False):
        self.stopped = True


def runner(**kwargs) -> tuple[SandboxToolRunner, list[FakeEnv]]:
    # The fake's script is popped before construction, not inside the factory: the
    # factory runs on the first open_session, long after __init__ read its kwargs.
    replies = kwargs.pop("replies", None)
    fail = kwargs.pop("fail", None)
    start_error = kwargs.pop("start_error", None)
    made: list[FakeEnv] = []

    def factory(spec):
        env = FakeEnv(spec, replies=replies, fail=fail, start_error=start_error)
        made.append(env)
        return env

    kwargs.setdefault("image", "img:1")
    return SandboxToolRunner(env_factory=factory, **kwargs), made


# --- the tool it offers ----------------------------------------------------

def test_it_offers_exactly_the_bash_tool():
    r, _ = runner()
    assert [t.name for t in r.describe_tools()] == ["bash"]
    assert "command" in BASH_TOOL.json_schema["properties"]


async def test_an_unknown_tool_is_refused():
    r, _ = runner()
    await r.open_session("s0")
    with pytest.raises(KeyError):
        await r.run("python", {"command": "x", "session_id": "s0"})


# --- a container that stops executing commands -----------------------------

class DyingEnv(FakeEnv):
    """An environment whose exec raises after a set number of calls.

    What a lost exec server looks like from the runner's side: the container is up as
    far as anyone can tell, and every command raises: an ``httpx.ReadError`` with an
    empty message, after which every command of the trajectory fails."""

    def __init__(self, spec, healthy_calls=1, **kwargs):
        super().__init__(spec, **kwargs)
        self.healthy_calls = healthy_calls

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if not self.started:
            raise RuntimeError("environment not started")
        self.calls.append((command, cwd, timeout_sec))
        if len(self.calls) > self.healthy_calls:
            raise RuntimeError("ReadError: ")
        return FakeExecResult(stdout="ok")


def dying_runner(healthy_calls=1, **kwargs):
    """A runner whose every container dies after ``healthy_calls`` commands."""
    made: list[DyingEnv] = []

    def factory(spec):
        env = DyingEnv(spec, healthy_calls=healthy_calls)
        made.append(env)
        return env

    kwargs.setdefault("image", "img:1")
    return SandboxToolRunner(env_factory=factory, **kwargs), made


async def test_a_dead_container_is_restarted_rather_than_reported():
    # A lost sandbox reported to the model as a failed command would leave the
    # trajectory running on against a corpse. A session is the traffic this benchmark
    # exists to generate, so the first move is to get one back.
    r, made = dying_runner(healthy_calls=1, max_sandbox_restarts=2)
    await r.open_session("s0")
    assert await r.run("bash", {"command": "ok", "session_id": "s0"})
    out = await r.run("bash", {"command": "boom", "session_id": "s0"})
    assert len(made) == 2, "a second container should have been built"
    assert made[0].stopped, "the dead one should have been released"
    assert out["exit_code"] == 0 and out["restarted"] == 1


async def test_a_restart_reuses_the_same_launch_spec(tmp_path):
    # The overlay upper lives on the host at a path derived from the session id, so a
    # container rebuilt from the same spec comes back with everything the agent wrote.
    # A restart onto a fresh overlay would silently discard the trajectory's work.
    r, made = dying_runner(healthy_calls=1, max_sandbox_restarts=1,
                           overlay_root=str(tmp_path))
    await r.open_session("s0")
    await r.run("bash", {"command": "ok", "session_id": "s0"})
    await r.run("bash", {"command": "boom", "session_id": "s0"})
    assert len(made) == 2
    assert (made[1].spec.overlay_dir == made[0].spec.overlay_dir
            == r.overlay_dir("s0"))
    assert made[1].spec.image == made[0].spec.image


async def test_a_restart_runs_the_task_setup_again():
    # Setup is what stages a repository into the working directory. A restarted
    # container that skipped it would drop the trajectory into an empty checkout,
    # which is the failure open_session already refuses to create.
    # Only the first container dies. ``dying_runner`` counts healthy calls per
    # container, so one parameter cannot both kill the original on its second call and
    # spare the replacement on its second; the factory is written out instead.
    made: list[FakeEnv] = []

    def factory(spec):
        env = DyingEnv(spec, healthy_calls=1) if not made else FakeEnv(spec)
        made.append(env)
        return env

    r = SandboxToolRunner(image="img:1", env_factory=factory,
                          max_sandbox_restarts=1,
                          setup_command=lambda ctx: "stage.sh")
    await r.open_session("s0")
    await r.run("bash", {"command": "boom", "session_id": "s0"})
    assert [c[0] for c in made[1].calls][0] == "stage.sh"


async def test_the_restart_budget_is_per_session_and_bounded():
    # Unbounded restarts would spend a drive's wall clock rebuilding a container that
    # cannot survive, which is worse than ending the session: the trajectory produces
    # no turns and the drive still pays for them. Running out raises, because a
    # container that could not be replaced inside its budget is a broken run
    # environment; see the harness-fault test below.
    from hls.core.faults import SandboxGone

    r, made = dying_runner(healthy_calls=0, max_sandbox_restarts=2)
    await r.open_session("s0")
    with pytest.raises(SandboxGone):
        await r.run("bash", {"command": "boom", "session_id": "s0"})
    # The original plus two replacements, and no more.
    assert len(made) == 3


async def test_restarts_are_counted_on_the_session():
    # A session that restarted three times is not the same measurement as one that
    # never did, so the count has to reach the diagnostics rather than stay internal.
    r, _ = dying_runner(healthy_calls=1, max_sandbox_restarts=3)
    await r.open_session("s0")
    await r.run("bash", {"command": "ok", "session_id": "s0"})
    out = await r.run("bash", {"command": "boom", "session_id": "s0"})
    assert out["restarted"] == 1
    assert r.restarts("s0") == 1


class DiesThenHangs(FakeEnv):
    """A container that loses its exec server, and whose replacement then hangs.

    The two failures a session really does see in sequence, which is what makes the
    timeout land on a session whose restart tally is already non-zero. ``which`` is the
    container's position in the session: the first raises once it has answered one
    command, the replacement answers the retry and then times out."""

    def __init__(self, spec, which, **kwargs):
        super().__init__(spec, **kwargs)
        self.which = which

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if not self.started:
            raise RuntimeError("environment not started")
        self.calls.append((command, cwd, timeout_sec))
        if self.which == 0:
            if len(self.calls) > 1:
                raise RuntimeError("ReadError: ")
            return FakeExecResult(stdout="ok")
        if len(self.calls) > 1:
            raise asyncio.TimeoutError
        return FakeExecResult(stdout="ok")


def dies_then_hangs_runner(**kwargs):
    """A runner whose container dies once and whose replacement then times out."""
    made: list[DiesThenHangs] = []

    def factory(spec):
        env = DiesThenHangs(spec, which=len(made))
        made.append(env)
        return env

    kwargs.setdefault("image", "img:1")
    kwargs.setdefault("max_sandbox_restarts", 2)
    return SandboxToolRunner(env_factory=factory, **kwargs), made


async def test_a_timed_out_command_still_reports_the_sessions_restarts():
    # A command that times out after a restart reports the session's restart count, not
    # zero. A zero there would make the trace say a restart un-happened and happened
    # again, and a reader counting rises would see one restart as several.
    #
    # Worse than a missing key, which is why it is asserted on the value rather than on
    # the key's presence: a literal zero written by a path that forgot cannot be told
    # apart from an observed zero.
    r, made = dies_then_hangs_runner()
    await r.open_session("s0")

    await r.run("bash", {"command": "ok", "session_id": "s0"})
    restarted = await r.run("bash", {"command": "boom", "session_id": "s0"})
    assert restarted["restarted"] == 1, "the restart itself should be reported"

    timed_out = await r.run("bash", {"command": "slow", "session_id": "s0"})
    assert timed_out["timed_out"] is True
    assert timed_out["exit_code"] == TIMEOUT_EXIT_CODE
    # The session's tally, not the command's outcome. Read off the runner so the
    # assertion is against what actually happened rather than against a literal.
    assert timed_out["restarted"] == r.restarts("s0") == 1


async def test_every_result_carries_the_restart_count():
    # The promise, enforced rather than described. Each of these is a different return
    # path through `run`, and each must carry the field: an ordinary command, a restart,
    # a timeout after a restart, and a sandbox fault with restarts off. A reader told the field is always there has given up the one signal that
    # would have caught a writer that forgot.
    r, _ = runner()
    await r.open_session("s0")
    ordinary = await r.run("bash", {"command": "ok", "session_id": "s0"})
    assert ordinary["restarted"] == 0

    dead, _ = dying_runner(healthy_calls=1, max_sandbox_restarts=0)
    await dead.open_session("s1")
    await dead.run("bash", {"command": "ok", "session_id": "s1"})
    faulted = await dead.run("bash", {"command": "boom", "session_id": "s1"})
    assert faulted["sandbox_error"] is True
    assert faulted["restarted"] == 0

    hung, _ = dies_then_hangs_runner()
    await hung.open_session("s2")
    await hung.run("bash", {"command": "ok", "session_id": "s2"})
    after_restart = await hung.run("bash", {"command": "boom", "session_id": "s2"})
    timed_out = await hung.run("bash", {"command": "slow", "session_id": "s2"})
    for out in (after_restart, timed_out):
        assert out["restarted"] == 1


async def test_a_recovered_restart_says_what_it_recovered_from():
    # A restart that works reports the fault it replaced the container for. Without it a
    # drive whose restarts all succeed exhausts no budget and its record cannot say what
    # the fault was.
    r, _ = dying_runner(healthy_calls=1, max_sandbox_restarts=2)
    await r.open_session("s0")
    await r.run("bash", {"command": "ok", "session_id": "s0"})

    out = await r.run("bash", {"command": "boom", "session_id": "s0"})
    assert out["restarted"] == 1, "the restart itself is already reported"
    assert out["exit_code"] == 0, "and the command really did run on the replacement"
    # What was missing: the cause.
    assert "RuntimeError" in out["restart_cause"]
    assert "ReadError" in out["restart_cause"]


class DiesOnce(FakeEnv):
    """The first container dies after one command; its replacement is healthy.

    ``dying_runner`` kills every container it builds, which is right for testing a
    budget and wrong for testing what a turn after a recovery looks like: there, the
    replacement dies too and the later turn restarts on its own merits."""

    def __init__(self, spec, which, **kwargs):
        super().__init__(spec, **kwargs)
        self.which = which

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if not self.started:
            raise RuntimeError("environment not started")
        self.calls.append((command, cwd, timeout_sec))
        if self.which == 0 and len(self.calls) > 1:
            raise RuntimeError("ReadError: ")
        return FakeExecResult(stdout="ok")


def dies_once_runner(**kwargs):
    made: list[DiesOnce] = []

    def factory(spec):
        env = DiesOnce(spec, which=len(made))
        made.append(env)
        return env

    kwargs.setdefault("image", "img:1")
    kwargs.setdefault("max_sandbox_restarts", 2)
    return SandboxToolRunner(env_factory=factory, **kwargs), made


async def test_the_cause_belongs_to_the_turn_that_restarted():
    # Per turn, not per session. The tally is cumulative and must be; the cause is not,
    # or every later turn would claim a fault that happened once, fifty turns ago.
    r, _ = dies_once_runner()
    await r.open_session("s0")
    await r.run("bash", {"command": "ok", "session_id": "s0"})
    restarted = await r.run("bash", {"command": "boom", "session_id": "s0"})
    after = await r.run("bash", {"command": "ok again", "session_id": "s0"})

    assert restarted["restart_cause"]
    assert after["restart_cause"] == ""
    # The tally does carry forward, which is the contrast worth pinning.
    assert after["restarted"] == restarted["restarted"] == 1


async def test_every_result_carries_a_cause_field():
    # Always present, for the same reason the restart count is: a reader told the
    # field is there has given up the signal that would catch a writer who forgot.
    r, _ = runner()
    await r.open_session("s0")
    out = await r.run("bash", {"command": "ok", "session_id": "s0"})
    assert out["restart_cause"] == ""

    dead, _ = dying_runner(healthy_calls=1, max_sandbox_restarts=0)
    await dead.open_session("s1")
    await dead.run("bash", {"command": "ok", "session_id": "s1"})
    faulted = await dead.run("bash", {"command": "boom", "session_id": "s1"})
    assert faulted["sandbox_error"] is True
    # Restarts were off, so nothing was replaced and there is no cause to report;
    # what went wrong is already in the output the model is shown.
    assert faulted["restart_cause"] == ""


async def test_restarting_is_off_by_default():
    # Off unless a caller asks, so no shipped path changes behaviour: the runner's
    # documented contract is that a sandbox fault reaches the model as a failed
    # command, and that is right for a one-off blip.
    r, made = dying_runner(healthy_calls=1)
    await r.open_session("s0")
    await r.run("bash", {"command": "ok", "session_id": "s0"})
    out = await r.run("bash", {"command": "boom", "session_id": "s0"})
    assert len(made) == 1
    assert out["exit_code"] == SANDBOX_ERROR_EXIT_CODE


async def test_a_timeout_is_not_a_reason_to_restart():
    # A command that ran too long is the command's doing and the container is fine.
    # Restarting on it would throw away a working sandbox, and on a task whose test
    # suite legitimately exceeds the limit it would do so every turn.
    r, made = runner(fail=asyncio.TimeoutError(), max_sandbox_restarts=2)
    await r.open_session("s0")
    out = await r.run("bash", {"command": "sleep 999", "session_id": "s0"})
    assert out["exit_code"] == TIMEOUT_EXIT_CODE
    assert len(made) == 1


async def test_a_replacement_that_cannot_start_is_also_a_broken_environment():
    # The replacement may fail to launch -- the node may be out of disk. That is the
    # same situation as running out of restarts, and it gets the same treatment: the
    # session that had a container has none, so it ends on SandboxGone.
    from hls.core.faults import SandboxGone

    made: list[FakeEnv] = []

    def factory(spec):
        env = (DyingEnv(spec, healthy_calls=1) if not made
               else FakeEnv(spec, start_error=RuntimeError("no disk")))
        made.append(env)
        return env

    r = SandboxToolRunner(image="img:1", env_factory=factory, max_sandbox_restarts=2)
    await r.open_session("s0")
    await r.run("bash", {"command": "ok", "session_id": "s0"})
    with pytest.raises(SandboxGone):
        await r.run("bash", {"command": "boom", "session_id": "s0"})


class _HangsUntilCancelled(FakeEnv):
    """A replacement whose start, or whose setup command, waits until it is cancelled."""

    def __init__(self, spec, reached: asyncio.Event, hang_in: str):
        super().__init__(spec)
        self.reached = reached
        self.hang_in = hang_in

    async def start(self, force_build):
        if self.hang_in == "start":
            self.reached.set()
            await asyncio.Event().wait()
        await super().start(force_build)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if self.hang_in == "setup" and command == "stage.sh" and not self.calls:
            self.reached.set()
            await asyncio.Event().wait()
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)


@pytest.mark.parametrize("hang_in", ["start", "setup"])
async def test_a_session_cancelled_while_its_container_is_replaced_ends_cancelled(hang_in):
    # The drive's scheduled cut cancels a session wherever it is. One cut during a
    # replacement is a cut session, and the driver records it as one; SandboxGone here
    # would record a dropped session whose sandbox could not be replaced.
    from hls.core.faults import SandboxGone

    made: list[FakeEnv] = []
    reached = asyncio.Event()

    def factory(spec):
        env = (DyingEnv(spec, healthy_calls=2) if not made
               else _HangsUntilCancelled(spec, reached, hang_in))
        made.append(env)
        return env

    r = SandboxToolRunner(image="img:1", env_factory=factory, max_sandbox_restarts=2,
                          setup_command=lambda ctx: "stage.sh")
    await r.open_session("s0")
    await r.run("bash", {"command": "ok", "session_id": "s0"})
    command = asyncio.ensure_future(r.run("bash", {"command": "boom", "session_id": "s0"}))
    await asyncio.wait_for(reached.wait(), timeout=5)
    command.cancel()
    try:
        await command
    except asyncio.CancelledError:
        pass
    except SandboxGone:  # pragma: no cover - the point of the test
        pytest.fail("a cancellation during a replacement was reported as a sandbox gone")
    assert command.cancelled()
    assert len(made) == 2


# --- a restart budget that runs out ----------------------------------------

async def test_exhausting_the_restart_budget_raises_a_harness_fault():
    # A container that cannot be brought back inside its budget is a broken sandbox,
    # not a command that failed. The fault propagates out of the runner, and the
    # trajectory driver drops the session rather than score a zero against the
    # contestant.
    from hls.core.faults import HarnessFault, SandboxGone

    assert issubclass(SandboxGone, HarnessFault)
    r, made = dying_runner(healthy_calls=0, max_sandbox_restarts=2)
    await r.open_session("s0")
    with pytest.raises(SandboxGone) as caught:
        await r.run("bash", {"command": "boom", "session_id": "s0"})
    # The original plus two replacements, then it gives up.
    assert len(made) == 3
    # The message is the record: the driver writes it into the trace as the drop's
    # reason, so what is not in the message is not in the record.
    message = str(caught.value)
    assert "s0" in message
    assert "2" in message


async def test_the_fault_names_what_the_runner_knows_and_not_what_it_does_not():
    # The runner knows containers; the loop knows turns. Turn counts reach the record
    # through the session's final trace event, not through this message, and pushing
    # them in here would mean teaching the runner about turns.
    from hls.core.faults import SandboxGone

    r, _ = dying_runner(healthy_calls=0, max_sandbox_restarts=1)
    await r.open_session("s0")
    with pytest.raises(SandboxGone) as caught:
        await r.run("bash", {"command": "boom", "session_id": "s0"})
    assert "ReadError" in str(caught.value)


async def test_the_fault_is_one_structured_line():
    # The driver writes this fault's message as a dropped session's whole reason, in a
    # single field of the trace's END.json. Prose that wrapped, or a newline out of the
    # underlying error, would leave the record holding a fragment.
    from hls.core.faults import SANDBOX_GONE, SandboxGone

    r, _ = dying_runner(healthy_calls=0, max_sandbox_restarts=2)
    await r.open_session("s0")
    with pytest.raises(SandboxGone) as caught:
        await r.run("bash", {"command": "boom", "session_id": "s0"})
    message = str(caught.value)
    assert "\n" not in message
    assert message.startswith(SANDBOX_GONE)
    fields = dict(part.split("=", 1) for part in message.split(" ") if "=" in part)
    assert fields["session"] == "'s0'"
    # Spent and budget are separate fields, and `restarts` means what it means in the
    # message a task's agent loop writes for the same fault: how many were spent.
    assert fields["restarts"] == "2"
    assert fields["restart_budget"] == "2"


async def test_the_message_survives_into_the_drops_reason():
    # The trajectory driver records a dropped session with ``str()`` of this fault as its
    # reason (``hls.core.trace.DroppedSession``), which END.json carries: the fields must
    # still be readable there, on one line.
    from hls.core.faults import SandboxGone
    from hls.core.trace import DroppedSession

    r, _ = dying_runner(healthy_calls=0, max_sandbox_restarts=1)
    await r.open_session("s0")
    with pytest.raises(SandboxGone) as caught:
        await r.run("bash", {"command": "boom", "session_id": "s0"})
    reason = DroppedSession(session_id="s0", tenant_id="t0", reason=str(caught.value)).reason
    assert "\n" not in reason
    assert "session='s0'" in reason
    assert "restarts=1 restart_budget=1" in reason


async def test_with_restarts_off_a_fault_still_reaches_the_model():
    # The documented contract, unchanged for a caller who did not opt in: a sandbox
    # fault is reported as a failed command so one blip does not end a trajectory,
    # and the loop's own fault-run stop is what ends it if they keep coming.
    r, made = dying_runner(healthy_calls=0)
    await r.open_session("s0")
    out = await r.run("bash", {"command": "boom", "session_id": "s0"})
    assert out["exit_code"] == SANDBOX_ERROR_EXIT_CODE
    assert out["sandbox_error"] is True
    assert len(made) == 1


async def test_a_session_inside_its_budget_does_not_raise():
    from hls.core.faults import SandboxGone

    r, _ = dying_runner(healthy_calls=1, max_sandbox_restarts=3)
    await r.open_session("s0")
    await r.run("bash", {"command": "ok", "session_id": "s0"})
    try:
        out = await r.run("bash", {"command": "boom", "session_id": "s0"})
    except SandboxGone:  # pragma: no cover - the point of the test
        pytest.fail("raised inside the budget")
    assert out["exit_code"] == 0 and out["restarted"] == 1


# --- session lifetime ------------------------------------------------------

async def test_a_command_without_an_open_session_is_refused():
    # An implicit open would hand the model a container with no repository in it.
    r, made = runner()
    with pytest.raises(KeyError, match="no open sandbox"):
        await r.run("bash", {"command": "ls", "session_id": "nope"})
    assert made == []


async def test_open_starts_one_container_and_close_stops_it():
    r, made = runner()
    await r.open_session("s0")
    assert len(made) == 1 and not made[0].stopped
    await r.close_session("s0")
    assert made[0].stopped


async def test_the_container_is_running_when_open_session_returns():
    # A constructed-but-unstarted environment refuses every exec, so the trajectory
    # would spend its whole budget reporting sandbox errors. The factory only builds;
    # starting is the runner's half of the lifetime it already ends in close_session.
    r, made = runner()
    await r.open_session("s0")
    assert made[0].started is True
    assert made[0].force_build is False


async def test_a_container_that_will_not_start_is_released_not_leaked():
    # The session is not registered yet when start fails, so close_session cannot
    # reach this container; open_session has to release it on the way out.
    r, made = runner(start_error=RuntimeError("no such image"))
    with pytest.raises(RuntimeError, match="no such image"):
        await r.open_session("s0")
    assert made[0].stopped
    assert "s0" not in r._sessions


async def test_a_failed_start_leaves_no_session_to_short_circuit_a_retry():
    # A dead session left in the table would make every retry the no-op a live
    # session gets, so the trajectory would never get a container at all.
    r, made = runner(start_error=RuntimeError("transient"))
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await r.open_session("s0")
    assert len(made) == 2


async def test_reopening_a_live_session_does_not_leak_a_second_container():
    r, made = runner()
    await r.open_session("s0")
    await r.open_session("s0")
    assert len(made) == 1


async def test_an_overlay_is_reclaimed_when_the_session_ends(tmp_path):
    # The default, and it has to stay the default: a drive is 60-odd sessions each
    # allowed 30 GiB, so a runner that stopped reclaiming would fill the filesystem
    # they all share and nothing would say why.
    r, _ = runner(overlay_root=str(tmp_path))
    await r.open_session("s0")
    overlay = tmp_path / r.overlay_dir("s0")
    assert overlay.is_dir()
    await r.close_session("s0")
    assert not overlay.exists()


async def test_an_overlay_survives_when_the_runner_is_told_to_keep_it(tmp_path):
    # For the one caller whose output IS that directory: an offline pass that
    # captures what a container wrote so a later session can start from it.
    # Reclaiming there deletes the artifact the pass exists to produce, and the pass
    # reports success either way.
    r, _ = runner(overlay_root=str(tmp_path), reclaim_overlays=False)
    await r.open_session("s0")
    written = tmp_path / r.overlay_dir("s0") / "written-by-the-pass"
    written.write_text("x")
    await r.close_session("s0")
    assert written.read_text() == "x"


async def test_keeping_overlays_still_stops_the_container(tmp_path):
    # Keeping the directory is not keeping the container. A pass that leaked one per
    # band would hold a node's memory until the job ended.
    r, made = runner(overlay_root=str(tmp_path), reclaim_overlays=False)
    await r.open_session("s0")
    await r.close_session("s0")
    assert made[0].stopped
    assert "s0" not in r._sessions


async def test_closing_an_unopened_session_is_harmless():
    r, made = runner()
    await r.close_session("never-opened")
    assert made == [] and r._sessions == {}


async def test_a_teardown_failure_does_not_fail_the_run():
    r, made = runner()
    await r.open_session("s0")

    async def boom(delete=False):
        raise RuntimeError("daemon gone")

    made[0].stop = boom
    await r.close_session("s0")          # must not raise
    assert "s0" not in r._sessions


async def test_each_session_gets_its_own_container():
    r, made = runner()
    await r.open_session("a")
    await r.open_session("b")
    await r.run("bash", {"command": "ls", "session_id": "a"})
    assert len(made) == 2
    assert made[0].calls and not made[1].calls


# --- staging ---------------------------------------------------------------

async def test_the_setup_command_runs_in_the_workdir_before_any_turn():
    seen: list[dict] = []

    def setup(context):
        seen.append(context)
        return f"tar xf /data/{context['instance_id']}.tar"

    r, made = runner(setup_command=setup, workdir="/testbed")
    await r.open_session("s0", instance_id="django__django-1")
    assert seen == [{"instance_id": "django__django-1"}]
    assert made[0].calls[0][0] == "tar xf /data/django__django-1.tar"
    assert made[0].calls[0][1] == "/testbed"


async def test_a_failed_setup_fails_the_trajectory_and_releases_the_container():
    r, made = runner(setup_command=lambda ctx: "false",
                     replies={"false": FakeExecResult(stderr="no such file",
                                                      return_code=2)})
    with pytest.raises(RuntimeError, match="sandbox setup failed"):
        await r.open_session("s0", instance_id="x")
    assert made[0].stopped
    assert "s0" not in r._sessions


async def test_a_setup_command_that_raises_also_releases_the_container():
    # Staging can fail without an exit code at all -- a broken container drops the
    # connection. The session never opened either way, so the container has to go.
    r, made = runner(setup_command=lambda ctx: "tar xf snapshot.tar",
                     fail=ConnectionError("container died"))
    with pytest.raises(ConnectionError):
        await r.open_session("s0", instance_id="x")
    assert made[0].stopped
    assert "s0" not in r._sessions


# --- a container that never finishes starting ------------------------------

class NeverStarts(FakeEnv):
    """A container whose start hangs rather than fails.

    Such a session writes no turn, keeps its overlay and holds a concurrency slot.
    Nothing raises on its own, because a fault is recorded when a session ends and this
    one does not end."""

    async def start(self, force_build):
        await asyncio.Event().wait()


def hanging_runner(**kwargs):
    made: list[NeverStarts] = []

    def factory(spec):
        env = NeverStarts(spec)
        made.append(env)
        return env

    kwargs.setdefault("image", "img:1")
    return SandboxToolRunner(env_factory=factory, **kwargs), made


async def test_a_launch_that_never_answers_is_refused():
    from hls.core.faults import SandboxNotStarted

    r, _ = hanging_runner(open_timeout_sec=0.05)
    with pytest.raises(SandboxNotStarted, match="did not start"):
        await r.open_session("s0")


async def test_a_launch_that_never_answers_is_a_harness_fault_the_driver_does_not_drop():
    # A session that never had a sandbox is the harness's failure: a drive that went on
    # without it would score a population nobody chose. So it is not SandboxGone, which the
    # trajectory driver drops a session on, and it fails the drive.
    from hls.core.faults import HarnessFault, SandboxGone, SandboxNotStarted

    assert issubclass(SandboxNotStarted, HarnessFault)
    assert not issubclass(SandboxNotStarted, SandboxGone)
    r, _ = hanging_runner(open_timeout_sec=0.05)
    with pytest.raises(SandboxNotStarted):
        await r.open_session("s0")


async def test_a_launch_that_never_answers_gives_its_slot_back():
    # The damage the drive wall clock does not bound. A hung open that kept its
    # admission slot would hold it until the backstop ended the whole drive; releasing the container
    # and the overlay is what makes the failure cost one session rather than the tail
    # of the run.
    from hls.core.faults import SandboxNotStarted

    r, made = hanging_runner(open_timeout_sec=0.05)
    with pytest.raises(SandboxNotStarted):
        await r.open_session("s0")
    assert made[0].stopped, "the container it gave up on should have been released"
    assert "s0" not in r._sessions, "a session that never opened must not be tracked"


async def test_a_launch_inside_the_bound_is_untouched():
    # The control. A bound that refused every launch would pass the three above.
    r, made = runner(open_timeout_sec=30)
    await r.open_session("s0")
    assert made[0].started
    assert "s0" in r._sessions


# --- the resources one session's container is given ------------------------

async def test_a_flat_budget_is_given_to_every_session():
    r, _ = runner(cpus=4, memory_mb=16384)
    await r.open_session("s0")
    spec = r.launch_spec("s0", task_name="anything")
    assert (spec.cpus, spec.memory_mb) == (4, 16384)


async def test_a_budget_can_be_resolved_per_session():
    # The corpus declares what each task's environment needs, and those declarations
    # differ by a factor of eight. One figure for all of them either starves the
    # largest -- which is not a short trajectory but an OOM kill, after which the
    # runtime's error comes back as ordinary command output -- or hands every small
    # task the largest one's budget, which is what bounds how many run at once.
    sizes = {"big": (16, 32768), "small": (2, 4096)}
    r, made = runner(cpus=lambda ctx: sizes[ctx["task_name"]][0],
                     memory_mb=lambda ctx: sizes[ctx["task_name"]][1])
    # Opened with the context, because that is when the container is built and so
    # when the budget has to be known.
    await r.open_session("s0", task_name="big")
    await r.open_session("s1", task_name="small")

    big, small = (e.spec for e in made)
    assert (big.cpus, big.memory_mb) == (16, 32768)
    assert (small.cpus, small.memory_mb) == (2, 4096)


async def test_an_unset_budget_stays_unset():
    # None means "whatever the backend gives", which is not the same as zero and must
    # not be resolved into one.
    r, _ = runner()
    await r.open_session("s0")
    spec = r.launch_spec("s0", task_name="anything")
    assert spec.cpus is None and spec.memory_mb is None


# --- running commands ------------------------------------------------------

async def test_a_command_returns_its_output_and_exit_code():
    r, _ = runner(replies={"pytest": FakeExecResult(stdout="2 passed", return_code=0)})
    await r.open_session("s0")
    got = await r.run("bash", {"command": "pytest", "session_id": "s0"})
    # Whole-dict rather than key by key, so a field appearing here has to be accounted
    # for. `restarted` is on every result, including this one: it is the session's
    # tally and a session that has never lost a container has zero.
    assert got == {"output": "2 passed", "exit_code": 0, "timed_out": False,
                   "restarted": 0, "restart_cause": ""}


async def test_stdout_and_stderr_are_merged_as_a_terminal_shows_them():
    r, _ = runner(replies={"make": FakeExecResult(stdout="building", stderr="warning")})
    await r.open_session("s0")
    got = await r.run("bash", {"command": "make", "session_id": "s0"})
    assert got["output"] == "building\nwarning"


async def test_a_nonzero_exit_is_reported_not_raised():
    r, _ = runner(replies={"grep x": FakeExecResult(return_code=1)})
    await r.open_session("s0")
    got = await r.run("bash", {"command": "grep x", "session_id": "s0"})
    assert got["exit_code"] == 1 and got["output"] == ""


async def test_the_requested_timeout_reaches_the_backend():
    r, made = runner(default_timeout_sec=300)
    await r.open_session("s0")
    await r.run("bash", {"command": "ls", "timeout_sec": 42, "session_id": "s0"})
    assert made[0].calls[-1][2] == 42


async def test_the_default_timeout_applies_when_none_is_asked_for():
    r, made = runner(default_timeout_sec=77)
    await r.open_session("s0")
    await r.run("bash", {"command": "ls", "session_id": "s0"})
    assert made[0].calls[-1][2] == 77


async def test_a_timeout_is_reported_to_the_model_not_raised():
    r, _ = runner(fail=asyncio.TimeoutError())
    await r.open_session("s0")
    got = await r.run("bash", {"command": "sleep 999", "timeout_sec": 5,
                               "session_id": "s0"})
    assert got["timed_out"] is True
    assert got["exit_code"] == TIMEOUT_EXIT_CODE
    assert "timed out after 5 seconds" in got["output"]


async def test_a_broken_sandbox_is_flagged_so_triage_can_tell_it_apart():
    r, _ = runner(fail=ConnectionError("container died"))
    await r.open_session("s0")
    got = await r.run("bash", {"command": "ls", "session_id": "s0"})
    assert got["sandbox_error"] is True
    assert got["exit_code"] == SANDBOX_ERROR_EXIT_CODE
    assert "ConnectionError" in got["output"]


async def test_enormous_output_is_bounded_before_it_crosses_the_driver():
    r, _ = runner(replies={"find /": FakeExecResult(stdout="x" * 10_000)},
                  output_limit_chars=100)
    await r.open_session("s0")
    got = await r.run("bash", {"command": "find /", "session_id": "s0"})
    assert len(got["output"]) == 100


# --- the launch spec it builds --------------------------------------------

async def test_the_launch_spec_carries_the_declared_placement():
    r, made = runner(workdir="/testbed", cpus=2, memory_mb=4096,
                     allow_internet=False, scratch_workdir=True)
    await r.open_session("s0")
    spec = made[0].spec
    assert spec.image == "img:1"
    assert spec.workdir == "/testbed"
    assert (spec.cpus, spec.memory_mb) == (2, 4096)
    assert spec.allow_internet is False
    assert spec.scratch_workdir is True


async def test_a_session_starts_from_its_image_alone_by_default():
    # Every task but the coding agent, and every session of that one that is not
    # resuming pre-rolled work.
    r, made = runner(workdir="/testbed")
    await r.open_session("s0")
    assert made[0].spec.lower_overlay_dirs == ()


async def test_a_session_can_start_from_state_an_earlier_container_wrote():
    # The warm start: what the pre-roll's container left, mounted read-only under this
    # session's own writable layer, so the session sees the work and writes elsewhere.
    r, made = runner(workdir="/testbed",
                     lower_overlays=lambda ctx: (f"/prefixes/{ctx['task_name']}.state",))
    await r.open_session("s0", task_name="git-to-zig")
    spec = made[0].spec
    assert spec.lower_overlay_dirs == ("/prefixes/git-to-zig.state",)
    # Its own writable layer is still its own: the lower is read-only and shared.
    assert spec.overlay_dir != "/prefixes/git-to-zig.state"


async def test_two_sessions_resuming_one_prefix_share_its_lower(tmp_path):
    # The reason the lower is read-only. Both draws of a task resume the same
    # directory at the same time, and each writes into its own layer.
    r, made = runner(workdir="/testbed", overlay_root=str(tmp_path),
                     lower_overlays=lambda ctx: (f"/prefixes/{ctx['task_name']}.state",))
    await r.open_session("git-to-zig", task_name="git-to-zig")
    await r.open_session("git-to-zig#1", task_name="git-to-zig")
    lowers = [env.spec.lower_overlay_dirs for env in made]
    assert lowers[0] == lowers[1] == ("/prefixes/git-to-zig.state",)
    assert made[0].spec.overlay_dir != made[1].spec.overlay_dir


# --- settings that differ from one session to the next ---------------------

async def test_a_string_image_is_the_same_for_every_session():
    # The single-image path, which every task but one uses: no callable, no context,
    # and two sessions land in containers built from the same image.
    r, made = runner(image="img:1", workdir="/testbed")
    await r.open_session("a")
    await r.open_session("b", task_name="ignored")
    assert [env.spec.image for env in made] == ["img:1", "img:1"]
    assert [env.spec.workdir for env in made] == ["/testbed", "/testbed"]


async def test_a_callable_image_routes_each_session_to_its_own():
    r, made = runner(image=lambda ctx: f"/sif/{ctx['task_name']}.sif")
    await r.open_session("a", task_name="formal-crypto")
    await r.open_session("b", task_name="wdm-design")
    assert [env.spec.image for env in made] == ["/sif/formal-crypto.sif",
                                                "/sif/wdm-design.sif"]


async def test_a_callable_workdir_is_where_that_sessions_commands_run():
    # The directory has to follow the image: a container started in a directory its
    # image does not have fails to launch, and a command run in the wrong one lands
    # somewhere the task's inputs are not.
    r, made = runner(image=lambda ctx: ctx["task_name"],
                     workdir=lambda ctx: ctx["workdir"])
    await r.open_session("a", task_name="one", workdir="/app")
    await r.open_session("b", task_name="two", workdir="/workspace")
    await r.run("bash", {"command": "ls", "session_id": "a"})
    await r.run("bash", {"command": "ls", "session_id": "b"})
    assert made[0].spec.workdir == "/app"
    assert made[1].spec.workdir == "/workspace"
    assert made[0].calls[0][1] == "/app"
    assert made[1].calls[0][1] == "/workspace"


async def test_the_setup_command_runs_in_the_sessions_own_workdir():
    r, made = runner(setup_command=lambda ctx: "stage",
                     workdir=lambda ctx: ctx["workdir"])
    await r.open_session("s0", workdir="/task")
    assert made[0].calls[0] == ("stage", "/task", r.setup_timeout_sec)


# --- what the driver's own container has to be given ----------------------

def test_a_runner_declares_the_host_paths_its_launch_needs():
    # The driver is in a container of its own, so the staged images and the device the
    # runtime mounts them through are invisible to it until these are bound in.
    paths = (HostPath("/hls/sandboxes/images", read_only=True, store="images"),
             HostPath("/dev/fuse", read_only=False, store=DEVICE))
    r, _ = runner(host_paths=paths)
    assert r.host_paths == paths


def test_a_runner_that_needs_nothing_from_the_host_says_so():
    r, _ = runner()
    assert r.host_paths == ()


def test_a_host_path_is_absolute_and_names_where_it_comes_from():
    with pytest.raises(ValueError, match="absolute"):
        HostPath("relative", read_only=True, store="images")
    with pytest.raises(ValueError, match="names no store"):
        HostPath("/hls/sandboxes/images", read_only=True, store="")
    with pytest.raises(ValueError, match="device"):
        HostPath("/hls/sandboxes/images", read_only=False, store=DEVICE)


def test_the_launch_sets_what_every_sandbox_start_needs():
    # The launch runs from its staging directory, polls a starting server quickly, and
    # names the exec server's interpreter where the task stages one.
    r, _ = runner(server_python="/opt/harbor-server/bin/python3")
    spec = r.launch_spec("s0")
    assert spec.server_python == "/opt/harbor-server/bin/python3"
    assert spec.start_in_staging_dir is True
    assert spec.ready_poll_sec == SERVER_READY_POLL_SEC
    assert runner()[0].launch_spec("s0").server_python is None


# --- where a session's container keeps what it writes ---------------------

async def test_each_session_gets_its_own_overlay_directory(tmp_path):
    # An overlay's upper layer is the container's whole writable filesystem, so two
    # sessions sharing one would write over each other.
    r, made = runner(overlay_root=str(tmp_path / "ovl"))
    await r.open_session("a")
    await r.open_session("b")
    a, b = made[0].spec.overlay_dir, made[1].spec.overlay_dir
    assert a != b
    assert os.path.basename(a) == "a" and os.path.basename(b) == "b"
    assert os.path.dirname(a) == os.path.dirname(b)
    assert os.path.dirname(os.path.dirname(a)) == str(tmp_path / "ovl")


async def test_the_overlay_directory_exists_before_the_container_starts(tmp_path):
    # The backend names the directory but does not create it, and a launch against a
    # path that is not there fails.
    seen: list[bool] = []
    root = tmp_path / "ovl"

    def factory(spec):
        import os
        seen.append(os.path.isdir(spec.overlay_dir))
        return FakeEnv(spec)

    r = SandboxToolRunner(image="img:1", env_factory=factory,
                          overlay_root=str(root))
    await r.open_session("s0")
    assert seen == [True]


async def test_closing_a_session_reclaims_its_overlay(tmp_path):
    root = tmp_path / "ovl"
    r, _ = runner(overlay_root=str(root))
    await r.open_session("s0")
    assert os.path.isdir(r.overlay_dir("s0"))
    await r.close_session("s0")
    assert not os.path.exists(r.overlay_dir("s0"))


async def test_a_container_that_will_not_start_leaves_no_overlay_behind(tmp_path):
    root = tmp_path / "ovl"
    r, _ = runner(overlay_root=str(root), start_error=RuntimeError("no start"))
    with pytest.raises(RuntimeError):
        await r.open_session("s0")
    assert os.listdir(root) == []


async def test_no_overlay_root_leaves_the_backend_its_own_default():
    r, made = runner()
    await r.open_session("s0")
    assert made[0].spec.overlay_dir is None


# --- two drives on one overlay root ----------------------------------------
#
# A session id is derived from the task, so every drive of the corpus opens the same
# ones, and drives on different nodes can share one root on a cluster filesystem.
# Two concurrent drives that shared a directory per task would each delete it under the
# other when they closed, and the other's sessions would fail at their next command.

async def test_two_pools_on_one_root_do_not_share_a_session_directory(tmp_path):
    a, made_a = runner(overlay_root=str(tmp_path))
    b, made_b = runner(overlay_root=str(tmp_path))
    await a.open_session("s0")
    await b.open_session("s0")
    assert made_a[0].spec.overlay_dir != made_b[0].spec.overlay_dir
    for spec in (made_a[0].spec, made_b[0].spec):
        assert spec.overlay_dir.startswith(str(tmp_path) + "/")


async def test_closing_one_pools_session_leaves_the_other_pools_intact(tmp_path):
    a, _ = runner(overlay_root=str(tmp_path))
    b, _ = runner(overlay_root=str(tmp_path))
    await a.open_session("s0")
    await b.open_session("s0")
    written = tmp_path.joinpath(b.overlay_dir("s0"), "written-by-b")
    written.write_text("x")
    await a.close_session("s0")
    assert not os.path.exists(a.overlay_dir("s0"))
    assert written.read_text() == "x"


async def test_the_pool_directory_goes_with_its_last_session(tmp_path):
    r, _ = runner(overlay_root=str(tmp_path))
    await r.open_session("a")
    await r.open_session("b")
    pool = os.path.dirname(r.overlay_dir("a"))
    await r.close_session("a")
    assert os.path.isdir(pool), "b is still open in it"
    await r.close_session("b")
    assert not os.path.exists(pool)
    assert tmp_path.is_dir()


class OverlayNeedingEnv(DyingEnv):
    """A dying container whose start fails when its overlay directory is missing, the
    way apptainer refuses an overlay path that does not exist."""

    async def start(self, force_build):
        if not os.path.isdir(self.spec.overlay_dir):
            raise RuntimeError(f"overlay {self.spec.overlay_dir} does not exist")
        await super().start(force_build)


async def test_a_restart_recreates_an_overlay_that_was_deleted(tmp_path):
    # The replacement starts on an empty layer, since what the trajectory wrote is
    # gone, and the cause says so rather than reading as a clean restart.
    made: list[OverlayNeedingEnv] = []

    def factory(spec):
        env = OverlayNeedingEnv(spec, healthy_calls=1)
        made.append(env)
        return env

    r = SandboxToolRunner(image="img:1", env_factory=factory,
                          overlay_root=str(tmp_path), max_sandbox_restarts=1)
    await r.open_session("s0")
    await r.run("bash", {"command": "ok", "session_id": "s0"})
    shutil.rmtree(r.overlay_dir("s0"))
    out = await r.run("bash", {"command": "boom", "session_id": "s0"})
    assert len(made) == 2 and made[1].started
    assert os.path.isdir(r.overlay_dir("s0"))
    assert out["exit_code"] == 0 and out["restarted"] == 1
    assert "overlay recreated empty" in out["restart_cause"]
