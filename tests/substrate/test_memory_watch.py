from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from hls.spec.resources import ResourceShape
from hls.substrate.base import RoleSpec
from hls.substrate.memory_watch import (END_ONE_MAX_KILLS, END_ONE_STILL_OVER_SEC, CannotEndOne, JobOoms, MemoryWatch,
                                        read_sample)
from hls.substrate.slurm.launcher import SlurmRoleHandle

GIB = 2**30
STEP = "1234110.1"


def _spec(tmp_path: Path) -> RoleSpec:
    return RoleSpec(name="artifact", argv=("/bin/true",), shape=ResourceShape(cpus=1, memory_gib=2, time_sec=60),
                    owner="contestant", log_dir=tmp_path)


def _step(d: Path, *, current: int, swap: int, file: int = 0, shmem: int = 0, slab: int = 0,
          oom_kill: int = 0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "memory.current").write_text(f"{current}\n")
    (d / "memory.swap.current").write_text(f"{swap}\n")
    (d / "memory.stat").write_text(
        f"anon {current - file - slab}\nfile {file}\nshmem {shmem}\nslab_reclaimable {slab}\nslab_unreclaimable 0\n")
    (d / "memory.events").write_text(f"low 0\nhigh 0\nmax 0\noom 0\noom_kill {oom_kill}\n")
    return d


def test_a_step_swapping_past_its_limit_is_killed_where_swap_is_unlimited(tmp_path):
    # A 2 GiB role grows to 2 GiB resident plus 1 GiB of swap.
    killed, breaches = [], []
    watch = MemoryWatch(enforce=True, clock=lambda: 5.0, on_breach=lambda s, sample, lim: breaches.append((s, lim)))
    watch.watch(STEP, _step(tmp_path / "s", current=2 * GIB, swap=GIB), 2 * GIB, lambda: killed.append(1),
                role="gpu0")
    watch.sample_once()
    watch.sample_once()
    assert killed == [1]
    assert breaches == [(STEP, 2 * GIB)]
    assert watch.breach(STEP).footprint == 3 * GIB


def test_page_cache_at_the_limit_is_not_a_breach(tmp_path):
    # Reading big files fills the cgroup with cache, and the kernel swapped a little anon.
    killed = []
    watch = MemoryWatch(enforce=True)
    step = _step(tmp_path / "s", current=2 * GIB, swap=64 * 2**20, file=GIB, shmem=0)
    watch.watch(STEP, step, 2 * GIB, lambda: killed.append(1), role="driver")
    watch.sample_once()
    assert killed == [] and watch.breach(STEP) is None
    assert watch.last(STEP).footprint == GIB + 64 * 2**20


def test_shmem_counts_against_the_limit(tmp_path):
    killed = []
    watch = MemoryWatch(enforce=True)
    step = _step(tmp_path / "s", current=2 * GIB, swap=GIB // 2, file=GIB, shmem=GIB)
    watch.watch(STEP, step, 2 * GIB, lambda: killed.append(1), role="gpu0")
    watch.sample_once()
    assert killed == [1]


def test_reclaimable_slab_is_not_a_breach(tmp_path):
    # The kernel reclaims slab before it kills anything, so it cannot make a step OOM.
    killed = []
    watch = MemoryWatch(enforce=True)
    step = _step(tmp_path / "s", current=2 * GIB, swap=GIB // 2, slab=GIB)
    watch.watch(STEP, step, 2 * GIB, lambda: killed.append(1), role="driver")
    watch.sample_once()
    assert killed == [] and watch.last(STEP).footprint == GIB + GIB // 2


def test_the_footprint_never_goes_below_zero(tmp_path):
    watch = MemoryWatch(enforce=True)
    step = _step(tmp_path / "s", current=GIB, swap=0, file=GIB, slab=GIB)
    watch.watch(STEP, step, 2 * GIB, lambda: None, role="driver")
    watch.sample_once()
    assert watch.last(STEP).footprint == 0


def test_where_the_kernel_enforces_the_watch_only_records(tmp_path):
    killed = []
    watch = MemoryWatch(enforce=False)
    watch.watch(STEP, _step(tmp_path / "s", current=2 * GIB, swap=4 * GIB, oom_kill=1), 2 * GIB,
                lambda: killed.append(1), role="gpu0")
    watch.sample_once()
    assert killed == [] and watch.last(STEP).oom_kills == 1


def test_a_finished_step_keeps_its_last_sample(tmp_path):
    watch = MemoryWatch(enforce=True)
    step = _step(tmp_path / "s", current=GIB, swap=0, oom_kill=1)
    watch.watch(STEP, step, 2 * GIB, lambda: None, role="gpu0")
    watch.sample_once()
    shutil.rmtree(step)
    watch.sample_once()
    assert watch.last(STEP).oom_kills == 1
    assert watch.last("nobody") is None


def test_the_breach_is_visible_during_the_kill_and_a_failed_kill_is_retried(tmp_path):
    attempts, breaches, seen_during_kill = [], [], []
    watch = MemoryWatch(enforce=True, on_breach=lambda s, sample, lim: breaches.append(s))

    def kill():
        attempts.append(1)
        seen_during_kill.append(watch.breach(STEP) is not None)
        if len(attempts) == 1:
            raise OSError("cgroup.procs vanished")

    watch.watch(STEP, _step(tmp_path / "s", current=2 * GIB, swap=GIB), 2 * GIB, kill, role="gpu0")
    watch.sample_once()
    # The kill failed: nothing is recorded as killed, and on_breach has not run.
    assert attempts == [1] and watch.breach(STEP) is None and breaches == []
    watch.sample_once()
    assert attempts == [1, 1] and watch.breach(STEP) is not None and breaches == [STEP]
    assert seen_during_kill == [True, True]
    watch.sample_once()
    assert attempts == [1, 1]


def test_the_breach_is_visible_when_the_killed_role_is_seen_to_end(tmp_path):
    # A real child stands in for srun. The stage reads the breach as soon as wait() returns, so
    # the watch must have recorded it before the child can end.
    missed = 0
    for i in range(20):
        step = _step(tmp_path / f"s{i}", current=2 * GIB, swap=GIB)
        child = subprocess.Popen(["sleep", "60"])
        handle = SlurmRoleHandle(_spec(tmp_path), child, tmp_path / "none.json")
        watch = MemoryWatch(enforce=True)
        watch.watch(STEP, step, 2 * GIB, handle.kill, role="artifact")
        thread = threading.Thread(target=watch.sample_once)
        thread.start()
        try:
            code = None
            while code is None:
                code = handle.wait(1.0)
            if watch.breach(STEP) is None:
                missed += 1
        finally:
            thread.join(10)
            if child.poll() is None:
                child.kill()
    assert missed == 0


def test_a_failing_on_breach_does_not_stop_the_pass(tmp_path):
    killed, breaches = [], []

    def on_breach(step, sample, limit):
        breaches.append(step)
        raise RuntimeError("callback failed")

    watch = MemoryWatch(enforce=True, on_breach=on_breach)
    watch.watch("1234110.1", _step(tmp_path / "a", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append("a"),
                role="driver")
    watch.watch("1234110.2", _step(tmp_path / "b", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append("b"),
                role="gpu0")
    watch.sample_once()
    assert killed == ["a", "b"] and breaches == ["1234110.1", "1234110.2"]
    assert watch.breach("1234110.1") is not None and watch.breach("1234110.2") is not None


def test_one_unreadable_step_does_not_stop_the_pass_for_the_others(tmp_path):
    bad = _step(tmp_path / "bad", current=GIB, swap=0)
    (bad / "memory.current").unlink()
    (bad / "memory.current").mkdir()
    killed = []
    watch = MemoryWatch(enforce=True)
    watch.watch("1234110.1", bad, 2 * GIB, lambda: None, role="driver")
    watch.watch("1234110.2", _step(tmp_path / "good", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(1),
                role="gpu0")
    watch.sample_once()
    assert killed == [1] and watch.last("1234110.1") is None


def test_steps_are_keyed_by_step_id_not_by_role_name(tmp_path):
    # The same role name runs again in a later attempt; its old step must keep its record.
    watch = MemoryWatch(enforce=False)
    watch.watch("1234110.1", _step(tmp_path / "a", current=GIB, swap=0, oom_kill=1), 2 * GIB, lambda: None,
                role="driver")
    watch.watch("1234110.4", _step(tmp_path / "b", current=GIB, swap=0), 2 * GIB, lambda: None, role="driver")
    watch.sample_once()
    assert watch.last("1234110.1").oom_kills == 1 and watch.last("1234110.4").oom_kills == 0


def test_job_ooms_reads_whose_limits_the_job_records_reached(tmp_path):
    # memory.events.local counts the job's own limit; memory.events counts it and every limit
    # below it; each step's memory.events counts its own.
    job = tmp_path / "job_1"
    for step, oom in (("step_batch", 1), ("step_extern", 0), ("step_3", 2)):
        (job / step).mkdir(parents=True)
        (job / step / "memory.events").write_text(f"low 0\nhigh 0\nmax 0\noom {oom}\noom_kill {oom}\n")
    (job / "memory.events.local").write_text("low 0\nhigh 0\nmax 7\noom 2\noom_kill 0\noom_group_kill 0\n")
    (job / "memory.events").write_text("low 0\nhigh 0\nmax 9\noom 5\noom_kill 3\noom_group_kill 0\n")
    assert MemoryWatch(enforce=False, job_dir=job).job_ooms() == JobOoms(
        own=2, below=3, steps={"step_batch": 1, "step_extern": 0, "step_3": 2})
    assert MemoryWatch(enforce=False).job_ooms() is None
    assert MemoryWatch(enforce=False, job_dir=tmp_path / "gone").job_ooms() is None
    (job / "memory.events.local").write_text("low 0\nhigh 0\nmax 7\noom_kill 0\n")
    assert MemoryWatch(enforce=False, job_dir=job).job_ooms() is None


def test_a_sample_keeps_the_steps_own_oom_count(tmp_path):
    step = _step(tmp_path / "s", current=GIB, swap=0, oom_kill=1)
    assert read_sample(step, 0.0).ooms == 0
    (step / "memory.events").write_text("low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\n")
    assert (read_sample(step, 0.0).ooms, read_sample(step, 0.0).oom_kills) == (1, 1)
    (step / "memory.events").write_text("oom_kill 1\n")
    assert read_sample(step, 0.0).ooms is None


# A step watched with end_one, the build sandbox, survives a breach.


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_end_one_ends_a_process_again_only_after_its_grace_and_the_watch_keeps_sampling(tmp_path):
    clock, ended, killed, breaches, singles = _Clock(), [], [], [], []
    watch = MemoryWatch(enforce=True, clock=clock, on_breach=lambda s, sample, lim: breaches.append(sample.at),
                        on_single_kill=lambda s, sample, lim: singles.append((s, sample.at, lim)))
    step = _step(tmp_path / "s", current=3 * GIB, swap=0)
    watch.watch(STEP, step, 2 * GIB, lambda: killed.append(clock.now), role="sandbox",
                end_one=lambda: ended.append(clock.now) or True)
    watch.sample_once()
    clock.now = 4.9
    watch.sample_once()                     # within the 5 s grace: no second kill yet
    clock.now = 5.0
    watch.sample_once()
    assert ended == [0.0, 5.0] and killed == []
    # A single kill is reported apart from a kill of the step.
    assert singles == [(STEP, 0.0, 2 * GIB), (STEP, 5.0, 2 * GIB)] and breaches == []
    assert [s.at for s in watch.kills(STEP)] == [0.0, 5.0]
    assert watch.breach(STEP) is None and watch.last(STEP).at == 5.0
    _step(step, current=GIB, swap=0)
    clock.now = 20.0
    watch.sample_once()
    assert ended == [0.0, 5.0] and watch.last(STEP).at == 20.0


def test_a_breach_with_no_process_left_to_end_kills_the_step(tmp_path):
    clock, killed = _Clock(), []
    watch = MemoryWatch(enforce=True, clock=clock)
    step = _step(tmp_path / "s", current=3 * GIB, swap=0)
    answers = iter([True, False])
    watch.watch(STEP, step, 2 * GIB, lambda: killed.append(clock.now), role="sandbox", end_one=lambda: next(answers))
    watch.sample_once()
    clock.now = 5.0
    watch.sample_once()
    assert killed == [5.0] and watch.breach(STEP).at == 5.0 and [s.at for s in watch.kills(STEP)] == [0.0]
    assert watch.kill_reason(STEP) == "no process of it was left to end"
    clock.now = 10.0
    watch.sample_once()                     # a killed step is not sampled again
    assert killed == [5.0] and watch.last(STEP).at == 5.0


def test_an_end_one_that_raised_is_tried_again_at_the_next_pass(tmp_path):
    clock, attempts, killed = _Clock(), [], []

    def end_one():
        attempts.append(clock.now)
        if len(attempts) == 1:
            raise OSError("no such process")
        return True

    watch = MemoryWatch(enforce=True, clock=clock)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(1),
                role="sandbox", end_one=end_one)
    watch.sample_once()
    assert watch.kills(STEP) == [] and watch.breach(STEP) is None
    clock.now = 1.0
    watch.sample_once()
    assert attempts == [0.0, 1.0] and [s.at for s in watch.kills(STEP)] == [1.0] and killed == []


def test_end_one_is_never_called_where_the_kernel_enforces(tmp_path):
    ended = []
    watch = MemoryWatch(enforce=False)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: None, role="sandbox",
                end_one=lambda: ended.append(1) or True)
    watch.sample_once()
    assert ended == [] and watch.kills(STEP) == []


def test_a_step_without_end_one_is_killed_whole_and_has_no_single_kills(tmp_path):
    watch = MemoryWatch(enforce=True)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: None, role="gpu0")
    watch.sample_once()
    assert watch.breach(STEP) is not None and watch.kills(STEP) == [] and watch.kills("nobody") == []
    assert watch.kill_reason(STEP) is None and watch.kill_reason("nobody") is None


# A step watched with end_one that single kills do not bring under its limit is killed whole.


def test_a_step_still_over_its_limit_after_three_kills_is_killed_whole(tmp_path):
    # A contestant that starts a new process as fast as one is ended.
    clock, ended, killed, breaches = _Clock(), [], [], []
    watch = MemoryWatch(enforce=True, clock=clock, on_breach=lambda s, sample, lim: breaches.append(sample.at))
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(clock.now),
                role="sandbox", end_one=lambda: ended.append(clock.now) or True, exiting=lambda: False)
    for now in (0.0, 5.0, 10.0, 14.9):      # the last within the grace after the third kill
        clock.now = now
        watch.sample_once()
    assert ended == [0.0, 5.0, 10.0] and killed == [] and watch.breach(STEP) is None
    clock.now = 15.0
    watch.sample_once()
    assert ended == [0.0, 5.0, 10.0] and killed == [15.0] and watch.breach(STEP).at == 15.0 and breaches == [15.0]
    assert [s.at for s in watch.kills(STEP)] == [0.0, 5.0, 10.0]
    assert watch.kill_reason(STEP) == ("3 of its processes were ended since it was last within its limit, and it is "
                                       "still over it")
    assert (END_ONE_MAX_KILLS, END_ONE_STILL_OVER_SEC) == (3, 30.0)


def test_a_step_still_over_its_limit_30_s_after_a_kill_is_killed_whole(tmp_path):
    clock, ended, killed = _Clock(), [], []

    def end_one():
        ended.append(clock.now)
        if len(ended) > 1:
            raise OSError("the processes cannot be listed")
        return True

    watch = MemoryWatch(enforce=True, clock=clock)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(clock.now),
                role="sandbox", end_one=end_one)
    for second in range(30):
        clock.now = float(second)
        watch.sample_once()
    clock.now = 29.9
    watch.sample_once()
    assert killed == [] and [s.at for s in watch.kills(STEP)] == [0.0]
    clock.now = 30.0
    watch.sample_once()
    assert killed == [30.0]
    assert watch.kill_reason(STEP) == "still over its limit 30 s after the first of its processes was ended"


def test_the_30_s_count_from_the_first_unrelieved_kill_not_the_last(tmp_path):
    clock, ended, killed = _Clock(), [], []

    def end_one():
        ended.append(clock.now)
        if len(ended) > 2:
            raise OSError("the processes cannot be listed")
        return True

    watch = MemoryWatch(enforce=True, clock=clock)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(clock.now),
                role="sandbox", end_one=end_one)
    for now in (0.0, 5.0, 10.0, 20.0, 29.9):
        clock.now = now
        watch.sample_once()
    assert [s.at for s in watch.kills(STEP)] == [0.0, 5.0] and killed == []
    clock.now = 30.0                        # 30 s after the first kill, 25 s after the second
    watch.sample_once()
    assert killed == [30.0]


def test_no_process_is_ended_while_the_last_one_ended_is_still_exiting(tmp_path):
    # Where the kernel kills, it picks no new victim while one is exiting; the 30 s still bound it.
    clock, ended, killed, exiting = _Clock(), [], [], [True]
    watch = MemoryWatch(enforce=True, clock=clock)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(clock.now),
                role="sandbox", end_one=lambda: ended.append(clock.now) or True, exiting=lambda: exiting[0])
    clock.now = 0.0
    exiting[0] = False
    watch.sample_once()
    exiting[0] = True
    for now in (5.0, 10.0, 15.0, 20.0, 29.9):
        clock.now = now
        watch.sample_once()
    assert ended == [0.0] and killed == []
    clock.now = 30.0
    watch.sample_once()
    assert killed == [30.0] and watch.kill_reason(STEP).startswith("still over its limit 30 s")


def test_an_exiting_that_raises_reads_as_still_exiting_and_the_30_s_still_bound_it(tmp_path):
    # Not knowing whether the victim is gone is no reason to end the next process, which may be
    # one the victim's exit would have saved.
    clock, ended, killed, calls = _Clock(), [], [], []

    def exiting():
        if not ended:
            return False                        # no victim yet
        calls.append(clock.now)
        raise OSError("the victim's pidfd cannot be polled")

    watch = MemoryWatch(enforce=True, clock=clock)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(clock.now),
                role="sandbox", end_one=lambda: ended.append(clock.now) or True, exiting=exiting)
    clock.now = 0.0
    watch.sample_once()
    assert ended == [0.0] and calls == []
    for now in (5.0, 10.0, 15.0, 20.0, 29.9):
        clock.now = now
        watch.sample_once()
    assert calls == [5.0, 10.0, 15.0, 20.0, 29.9]
    assert ended == [0.0] and killed == []
    clock.now = 30.0
    watch.sample_once()
    assert killed == [30.0] and watch.kill_reason(STEP).startswith("still over its limit 30 s")
    assert ended == [0.0]


def test_an_exiting_that_raised_once_and_then_answers_lets_the_next_process_be_ended(tmp_path):
    clock, ended, killed, answers = _Clock(), [], [], iter([OSError("poll failed"), False])

    def exiting():
        if not ended:
            return False                        # no victim yet
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    watch = MemoryWatch(enforce=True, clock=clock)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(clock.now),
                role="sandbox", end_one=lambda: ended.append(clock.now) or True, exiting=exiting)
    for now in (0.0, 5.0, 6.0):
        clock.now = now
        watch.sample_once()
    assert ended == [0.0, 6.0] and killed == []


def _stuck(ended: list[float]):
    """An ``exiting`` for a victim that never exits: it has no victim before the first kill."""
    return lambda: bool(ended)


def _unreadable(ended: list[float]):
    """An ``exiting`` that can never be read once there is a victim."""
    def exiting():
        if not ended:
            return False
        raise OSError("the victim's pidfd cannot be polled")
    return exiting


@pytest.mark.parametrize("make_exiting", [_stuck, _unreadable], ids=["stuck-in-exit", "cannot-be-told"])
def test_a_victim_that_never_exits_holds_off_the_next_kill_for_30_s_after_its_own_kill_only(tmp_path, make_exiting):
    # The step goes over its limit, one process is ended, the step drops under the limit and goes
    # over again: no kill is unrelieved, so the 30 s rule from the first of them is not running.
    # Waiting on the victim is bounded from the victim's own kill, so the step is never left over
    # its limit with no action.
    clock, ended, killed = _Clock(), [], []
    step = _step(tmp_path / "s", current=3 * GIB, swap=0)
    watch = MemoryWatch(enforce=True, clock=clock)
    watch.watch(STEP, step, 2 * GIB, lambda: killed.append(clock.now), role="sandbox",
                end_one=lambda: ended.append(clock.now) or True, exiting=make_exiting(ended))
    clock.now = 0.0
    watch.sample_once()
    clock.now = 1.0
    _step(step, current=GIB, swap=0)
    watch.sample_once()                         # back within its limit
    _step(step, current=3 * GIB, swap=0)
    for now in (10.0, 15.0, 20.0, 25.0, 29.9):
        clock.now = now
        watch.sample_once()
    assert ended == [0.0] and killed == []      # within 30 s of the victim's kill: held off
    clock.now = 30.0
    watch.sample_once()
    assert ended == [0.0, 30.0] and killed == []
    for now in (35.0, 40.0, 50.0, 59.9):        # the new victim is waited on the same way
        clock.now = now
        watch.sample_once()
    assert ended == [0.0, 30.0] and killed == []
    clock.now = 60.0
    watch.sample_once()
    assert killed == [60.0] and watch.kill_reason(STEP).startswith("still over its limit 30 s")
    assert ended == [0.0, 30.0]


def test_the_step_is_not_killed_after_three_kills_while_the_third_victim_is_still_exiting(tmp_path):
    clock, ended, killed, exiting = _Clock(), [], [], [False]
    watch = MemoryWatch(enforce=True, clock=clock)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(clock.now),
                role="sandbox", end_one=lambda: ended.append(clock.now) or True, exiting=lambda: exiting[0])
    for now in (0.0, 5.0, 10.0):
        clock.now = now
        watch.sample_once()
    exiting[0] = True
    for now in (15.0, 20.0, 29.9):
        clock.now = now
        watch.sample_once()
    assert ended == [0.0, 5.0, 10.0] and killed == []
    clock.now = 30.0
    watch.sample_once()
    assert killed == [30.0] and watch.kill_reason(STEP).startswith("still over its limit 30 s")


def test_a_victim_that_has_exited_lets_the_next_process_be_ended(tmp_path):
    clock, ended, killed, exiting = _Clock(), [], [], [False]
    watch = MemoryWatch(enforce=True, clock=clock)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(clock.now),
                role="sandbox", end_one=lambda: ended.append(clock.now) or True, exiting=lambda: exiting[0])
    watch.sample_once()
    exiting[0] = True
    clock.now = 5.0
    watch.sample_once()
    exiting[0] = False
    clock.now = 6.0
    watch.sample_once()
    assert ended == [0.0, 6.0] and killed == []


def test_an_end_one_that_cannot_tell_which_process_to_end_kills_the_step_and_says_why(tmp_path):
    clock, killed = _Clock(), []

    def end_one():
        raise CannotEndOne("its processes could not be told apart by owner (no NSpid line)")

    watch = MemoryWatch(enforce=True, clock=clock)
    watch.watch(STEP, _step(tmp_path / "s", current=3 * GIB, swap=0), 2 * GIB, lambda: killed.append(clock.now),
                role="sandbox", end_one=end_one)
    watch.sample_once()
    assert killed == [0.0] and watch.kills(STEP) == []
    assert watch.kill_reason(STEP) == "its processes could not be told apart by owner (no NSpid line)"


def test_swap_cache_is_counted_once(tmp_path):
    # memory.stat's swapcached pages are charged to both memory.current and memory.swap.current.
    killed = []
    watch = MemoryWatch(enforce=True)
    step = _step(tmp_path / "s", current=2 * GIB, swap=GIB)
    with (step / "memory.stat").open("a") as f:
        f.write(f"swapcached {GIB}\n")
    watch.watch(STEP, step, 2 * GIB, lambda: killed.append(1), role="artifact")
    watch.sample_once()
    assert killed == [] and watch.last(STEP).footprint == 2 * GIB and watch.last(STEP).swap_cached == GIB
    # Where memory.stat has no such field, nothing is subtracted.
    assert read_sample(_step(tmp_path / "t", current=2 * GIB, swap=GIB), 0.0).footprint == 3 * GIB


def test_a_kill_that_brings_the_step_under_its_limit_does_not_escalate(tmp_path):
    clock, ended, killed = _Clock(), [], []
    watch = MemoryWatch(enforce=True, clock=clock)
    step = _step(tmp_path / "s", current=3 * GIB, swap=0)
    watch.watch(STEP, step, 2 * GIB, lambda: killed.append(clock.now), role="sandbox",
                end_one=lambda: ended.append(clock.now) or True)
    watch.sample_once()
    _step(step, current=GIB, swap=0)
    for now in (1.0, 31.0, 61.0):
        clock.now = now
        watch.sample_once()
    assert ended == [0.0] and killed == []
    # Each later breach that one kill ends starts afresh, however close together they come.
    for over in (62.0, 68.0, 74.0, 80.0):
        clock.now = over
        _step(step, current=3 * GIB, swap=0)
        watch.sample_once()
        clock.now = over + 1
        _step(step, current=GIB, swap=0)
        watch.sample_once()
    assert ended == [0.0, 62.0, 68.0, 74.0, 80.0] and killed == [] and watch.breach(STEP) is None
    assert watch.kill_reason(STEP) is None
