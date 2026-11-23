from __future__ import annotations

import math
import os
import resource
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls.substrate import nproc
from hls.settings import SiteSettings
from hls.substrate.base import NVIDIA_GPUS, ONLINE_CPUS, Launcher, NodeSize, own_node_size
from hls.substrate.nproc import (DEFAULT_NPROC_MAX, FLOOR_PER_EVALUATION, CeilingRule, PoolAtEnd, PoolPeak, PoolSample,
                                 ProcessCeiling, node_cap_problem, node_evaluations, node_nproc_max, node_pid_max,
                                 node_tasks, nproc_max_problem, peak_with, process_ceiling, starvation, step_tasks,
                                 uid_tasks)
from hls.substrate.slurm.launcher import node_size
from tests.substrate.nproc_pool import UidPool

HARD = 4_127_000
RESERVE = 1024
#: A site cap above every headroom these tests compute, so the headroom alone decides.
NO_CAP = 10**9
#: ``site.nproc_max`` of a site that sets none.
UNSET = SiteSettings.model_fields["nproc_max"].default
#: The cap such a site's node gets while it holds at most 32 evaluations of a step's size.
NPROC_MAX = DEFAULT_NPROC_MAX


@pytest.mark.parametrize("gpus, evaluations", [(1, 8), (2, 4), (8, 1)])
def test_the_ceiling_keeps_a_reserve_for_the_harness_of_every_evaluation_the_node_holds(gpus, evaluations):
    ceiling = process_ceiling(HARD, reserve=RESERVE, nproc_max=NO_CAP, node=NodeSize(gpus=8, cpus=64), gpus=gpus,
                              cpus=8)
    assert ceiling == ProcessCeiling(hard=HARD, reserve=RESERVE, evaluations=evaluations, evaluations_unknown=None,
                                     uncapped=HARD - evaluations * RESERVE, nproc_max=NO_CAP,
                                     ceiling=HARD - evaluations * RESERVE,
                                     floor=evaluations * FLOOR_PER_EVALUATION)
    assert ceiling.problem is None


def test_every_contestant_of_the_node_shares_one_pool_and_the_harness_keeps_its_reserves():
    # Eight evaluations of one uid on an 8-GPU node under a hard limit of 65,536: their servers
    # draw on the uid's one pool, whatever their limit, so a ceiling of a share each (8,064)
    # would fail them all at 16,000 tasks. The ceiling holds back only the harnesses' reserves.
    hard = 65_536
    ceiling = process_ceiling(hard, reserve=RESERVE, nproc_max=NO_CAP, node=NodeSize(gpus=8, cpus=64), gpus=1, cpus=8)
    assert ceiling.ceiling == 57_344 and ceiling.floor == 32_768 and ceiling.problem is None
    pool = UidPool(tasks=8 * 300)  # the runners, sruns and drivers already there
    assert [pool.grow(ceiling.ceiling, 2_000) for _ in range(8)] == [2_000] * 8
    assert UidPool(tasks=8 * 300).grow((hard - RESERVE) // 8, 16_000) < 16_000
    # A fork storm takes the pool up to the ceiling and no further: every harness, whose soft
    # limit the runner raised to the hard one, still starts its reserve.
    pool.grow(ceiling.ceiling, 10**6)
    assert pool.tasks == ceiling.ceiling and not pool.fork(ceiling.ceiling)
    assert [pool.grow(hard, RESERVE) for _ in range(8)] == [RESERVE] * 8


@pytest.mark.parametrize("hard, refused", [(40_959, True), (40_960, False)])
def test_a_ceiling_under_the_floor_of_the_nodes_servers_is_refused(hard, refused):
    # The floor is 4,096 tasks for each of the node's eight servers: the ceiling hard - 8 * 1,024
    # reaches it at a hard limit of 8 * 5,120.
    ceiling = process_ceiling(hard, reserve=RESERVE, nproc_max=NO_CAP, node=NodeSize(gpus=8, cpus=64), gpus=1, cpus=8)
    assert (ceiling.problem is not None) == refused
    if refused:
        assert "32768" in ceiling.problem and str(ceiling.ceiling) in ceiling.problem


def test_the_sites_cap_binds_where_the_uids_headroom_is_above_it():
    # A hard limit of 9,287,489 tasks, eight evaluations of one GPU on its eight-GPU nodes
    # and a reserve of 1,024 leave 9,279,297, over twice the kernel's pid_max of 4,194,304, with no
    # pids controller at any level of the job's cgroups. A server that spawned threads without
    # bound would take every pid of the node before its ceiling.
    assert UNSET is None and NPROC_MAX == 131_072
    ceiling = process_ceiling(9_287_489, reserve=RESERVE, nproc_max=UNSET, node=NodeSize(gpus=8, cpus=64),
                              gpus=1, cpus=8)
    assert ceiling == ProcessCeiling(hard=9_287_489, reserve=RESERVE, evaluations=8, evaluations_unknown=None,
                                     uncapped=9_279_297, nproc_max=NPROC_MAX, ceiling=NPROC_MAX, floor=32_768)
    assert ceiling.problem is None


def test_the_uids_headroom_binds_where_it_is_under_the_sites_cap():
    ceiling = process_ceiling(65_536, reserve=RESERVE, nproc_max=UNSET, node=NodeSize(gpus=8, cpus=64),
                              gpus=1, cpus=8)
    assert (ceiling.uncapped, ceiling.nproc_max, ceiling.ceiling) == (57_344, NPROC_MAX, 57_344)
    assert ceiling.problem is None


def test_a_site_that_sets_no_cap_gets_the_default_on_a_node_of_at_most_32_evaluations():
    # A 192-CPU node holds 24 evaluations of eight CPUs, a floor of 98,304 tasks, and a 256-CPU
    # node 32 of them; a 64-CPU node holds 32 of two CPUs. Each gets the default, 32 floors.
    for cpus, step_cpus, evaluations in ((192, 8, 24), (256, 8, 32), (64, 2, 32)):
        ceiling = process_ceiling(9_287_489, reserve=RESERVE, nproc_max=UNSET, node=NodeSize(gpus=0, cpus=cpus),
                                  gpus=0, cpus=step_cpus)
        assert (ceiling.evaluations, ceiling.floor) == (evaluations, evaluations * FLOOR_PER_EVALUATION)
        assert (ceiling.nproc_max, ceiling.ceiling, ceiling.problem) == (NPROC_MAX, NPROC_MAX, None)
        assert node_cap_problem(NodeSize(gpus=0, cpus=cpus), nproc_max=UNSET, gpus=0, cpus=step_cpus) is None


@pytest.mark.parametrize("cpus, step_cpus, evaluations", [(264, 8, 33), (66, 2, 33), (256, 2, 128), (512, 8, 64)])
def test_a_site_that_sets_no_cap_gets_the_floor_of_a_node_that_holds_more_than_32_evaluations(cpus, step_cpus,
                                                                                              evaluations):
    # The default's 32 floors would be under such a node's own floor, so its cap is that floor,
    # the least the floor rule accepts: no node is refused for its size.
    node = NodeSize(gpus=0, cpus=cpus)
    floor = evaluations * FLOOR_PER_EVALUATION
    assert node_nproc_max(UNSET, evaluations) == floor > NPROC_MAX
    ceiling = process_ceiling(9_287_489, reserve=RESERVE, nproc_max=UNSET, node=node, gpus=0, cpus=step_cpus)
    assert ceiling == ProcessCeiling(hard=9_287_489, reserve=RESERVE, evaluations=evaluations,
                                     evaluations_unknown=None, uncapped=9_287_489 - evaluations * RESERVE,
                                     nproc_max=floor, ceiling=floor, floor=floor)
    assert ceiling.problem is None
    assert node_cap_problem(node, nproc_max=UNSET, gpus=0, cpus=step_cpus) is None
    # The uid's headroom still binds where it is under that cap.
    low = process_ceiling(floor + evaluations * RESERVE - 1, reserve=RESERVE, nproc_max=UNSET, node=node, gpus=0,
                          cpus=step_cpus)
    assert low.ceiling == floor - 1 and low.problem is not None


def test_a_cap_the_site_sets_is_refused_on_a_node_whose_floor_it_is_under():
    # A site that sets the default's own value keeps the strict check: 33 evaluations are refused
    # by name, with the value that holds them.
    assert node_nproc_max(NPROC_MAX, 33) == NPROC_MAX
    assert process_ceiling(9_287_489, reserve=RESERVE, nproc_max=NPROC_MAX, node=NodeSize(gpus=0, cpus=256),
                           gpus=0, cpus=8).problem is None
    refused = ("site.nproc_max of 131072 tasks is under the floor of 135168, 4096 for each of the 33 evaluations' "
               "servers the node holds; raise site.nproc_max to at least 135168")
    assert process_ceiling(9_287_489, reserve=RESERVE, nproc_max=NPROC_MAX, node=NodeSize(gpus=0, cpus=264),
                           gpus=0, cpus=8).problem == refused
    assert node_cap_problem(NodeSize(gpus=0, cpus=264), nproc_max=NPROC_MAX, gpus=0, cpus=8) == refused


def test_a_node_of_unknown_size_is_refused_whether_or_not_the_site_sets_a_cap():
    for nproc_max in (UNSET, NPROC_MAX):
        ceiling = process_ceiling(HARD, reserve=RESERVE, nproc_max=nproc_max, node=NodeSize(gpus=0, cpus=None),
                                  gpus=0, cpus=8)
        assert ceiling.ceiling is None and ceiling.nproc_max == NPROC_MAX
        assert ceiling.problem == node_cap_problem(NodeSize(gpus=0, cpus=None), nproc_max=nproc_max, gpus=0, cpus=8)
        assert "CPU count is unknown" in ceiling.problem


def test_an_unlimited_hard_limit_leaves_the_sites_cap():
    for nproc_max in (UNSET, NPROC_MAX):
        ceiling = process_ceiling(None, reserve=RESERVE, nproc_max=nproc_max, node=NodeSize(gpus=8, cpus=64), gpus=1,
                                  cpus=8)
        assert (ceiling.uncapped, ceiling.ceiling) == (None, NPROC_MAX) and ceiling.problem is None


@pytest.mark.parametrize("nproc_max, refused", [(32_767, True), (32_768, False)])
def test_a_sites_cap_under_the_floor_of_the_nodes_servers_is_refused(nproc_max, refused):
    # The floor is 4,096 tasks for each of the node's eight servers, whatever the headroom above.
    ceiling = process_ceiling(9_287_489, reserve=RESERVE, nproc_max=nproc_max, node=NodeSize(gpus=8, cpus=64),
                              gpus=1, cpus=8)
    assert ceiling.ceiling == nproc_max
    assert (ceiling.problem is not None) == refused
    if refused:
        assert ceiling.problem == ("site.nproc_max of 32767 tasks is under the floor of 32768, 4096 for each of "
                                   "the 8 evaluations' servers the node holds; raise site.nproc_max to at least "
                                   "32768")
    assert (nproc_max_problem(nproc_max, 8) is not None) == refused


def test_one_evaluations_floor_is_what_no_node_can_hold_under():
    # A plan does not know its nodes: a cap under the floor of the one evaluation a contestant's
    # step is cannot hold it on any node.
    assert nproc_max_problem(FLOOR_PER_EVALUATION, 1) is None
    assert nproc_max_problem(UNSET, 1) is None and nproc_max_problem(UNSET, 10**6) is None
    assert nproc_max_problem(FLOOR_PER_EVALUATION - 1, 1) == (
        f"site.nproc_max of {FLOOR_PER_EVALUATION - 1} tasks is under the floor of {FLOOR_PER_EVALUATION}, "
        f"{FLOOR_PER_EVALUATION} for one evaluation's server; raise site.nproc_max to at least {FLOOR_PER_EVALUATION}")


@pytest.mark.parametrize("node, gpus, cpus, why", [
    (NodeSize(gpus=None, cpus=64), 1, 8,
     "the node's GPU count is unknown, so how many evaluations of this size it holds is unknown too"),
    (NodeSize(gpus=0, cpus=64), 1, 8, "the node's GPU count, 0, is under the 1 that one evaluation of this size takes"),
    (NodeSize(gpus=8, cpus=None), 0, 8,
     "the node's CPU count is unknown, so how many evaluations of this size it holds is unknown too"),
    (NodeSize(gpus=8, cpus=4), 0, 8, "the node's CPU count, 4, is under the 8 that one evaluation of this size takes"),
], ids=["gpus-unknown", "fewer-gpus", "cpus-unknown", "fewer-cpus"])
def test_a_node_whose_size_is_unknown_or_under_the_steps_refuses_the_step_and_says_why(node, gpus, cpus, why):
    # No evaluation count is guessed for it: the reserve and the floor would be guesses too.
    with pytest.raises(ValueError) as e:
        node_evaluations(node, gpus=gpus, cpus=cpus)
    assert str(e.value) == why
    ceiling = process_ceiling(HARD, reserve=RESERVE, nproc_max=NO_CAP, node=node, gpus=gpus, cpus=cpus)
    assert ceiling == ProcessCeiling(hard=HARD, reserve=RESERVE, evaluations=None, evaluations_unknown=why,
                                     uncapped=None, nproc_max=NO_CAP, ceiling=None, floor=None)
    assert ceiling.problem == why


def test_a_ceiling_is_counted_exactly_when_its_evaluations_are_known():
    known = dict(hard=HARD, reserve=RESERVE, evaluations=8, evaluations_unknown=None, uncapped=HARD - 8 * RESERVE,
                 nproc_max=NO_CAP, ceiling=HARD - 8 * RESERVE, floor=8 * FLOOR_PER_EVALUATION)
    ProcessCeiling(**known)
    # A ceiling with no count would let a step start with no limit: a record is never made so.
    for field in ("evaluations", "ceiling", "floor"):
        with pytest.raises(ValidationError, match="evaluations"):
            ProcessCeiling(**{**known, field: None})
    unknown = dict(hard=HARD, reserve=RESERVE, evaluations=None, evaluations_unknown="the node's GPU count is unknown",
                   uncapped=None, nproc_max=NO_CAP, ceiling=None, floor=None)
    ProcessCeiling(**unknown)
    for field, value in (("evaluations", 8), ("uncapped", HARD), ("ceiling", HARD), ("floor", FLOOR_PER_EVALUATION)):
        with pytest.raises(ValidationError, match="evaluations"):
            ProcessCeiling(**{**unknown, field: value})


def test_a_gpu_less_step_counts_the_evaluations_by_the_nodes_cpus():
    ceiling = process_ceiling(HARD, reserve=RESERVE, nproc_max=NO_CAP, node=NodeSize(gpus=8, cpus=64), gpus=0, cpus=16)
    assert (ceiling.evaluations, ceiling.evaluations_unknown) == (4, None)
    # A node's GPUs, known or not, do not count a step that takes none.
    assert process_ceiling(HARD, reserve=RESERVE, nproc_max=NO_CAP, node=NodeSize(gpus=None, cpus=64), gpus=0,
                           cpus=16).evaluations == 4


@pytest.mark.skipif(os.getuid() == 0, reason="RLIMIT_NPROC does not bind root")
def test_the_uids_task_count_is_what_the_kernel_holds_a_fork_against():
    # A child that started nothing is refused its fork under a limit of half the uid's count, and
    # allowed it under twice the count: the count is the kernel's, threads and all.
    code = ("import os, resource, sys\n"
            "from hls.substrate.nproc import uid_tasks\n"
            "limit = int(sys.argv[1] if sys.argv[1] != 'half' else uid_tasks(os.getuid()) // 2)\n"
            "resource.setrlimit(resource.RLIMIT_NPROC, (limit, limit))\n"
            "try:\n"
            "    pid = os.fork()\n"
            "except BlockingIOError:\n"
            "    sys.exit(11)\n"
            "if pid == 0:\n"
            "    os._exit(0)\n"
            "os.waitpid(pid, 0)\n")
    held = uid_tasks(os.getuid())
    assert held > 0
    # Twice the count, or the host's hard limit when that is lower, which the uid is under.
    hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
    room = 2 * held + 1000 if hard == resource.RLIM_INFINITY else min(2 * held + 1000, hard)
    for limit, code_wanted in (("half", 11), (str(room), 0)):
        done = subprocess.run([sys.executable, "-c", code, limit], capture_output=True, text=True, timeout=60)
        assert done.returncode == code_wanted, done.stderr


def _status(proc: Path, pid: int, uid: int, threads: int) -> None:
    (proc / str(pid)).mkdir()
    (proc / str(pid) / "status").write_text(f"Name:\tx\nUid:\t{uid}\t{uid + 1}\t{uid}\t{uid}\n"
                                            f"Threads:\t{threads}\n")


def test_a_process_name_that_is_not_utf8_or_holds_a_carriage_return_is_counted_like_any_other(tmp_path):
    # A process names itself (its comm), and the kernel passes bytes above 0x7f through raw and a
    # carriage return unescaped: neither may stop the count or forge a line of the status. The
    # forged lines are the only Uid and Threads in pid 12's status, so a reader that took a
    # carriage return for a line's end would count its 900.
    _status(tmp_path, 10, 1000, 5)
    (tmp_path / "11").mkdir()
    (tmp_path / "11" / "status").write_bytes(b"Name:\thog\xff\xfe\nUid:\t1000\t1000\t1000\t1000\nThreads:\t3\n")
    (tmp_path / "12").mkdir()
    (tmp_path / "12" / "status").write_bytes(b"Name:\tx\rUid:\t1000\t0\t0\t0\rThreads:\t900\nState:\tS (sleeping)\n")
    assert uid_tasks(1000, proc=tmp_path) == 8


def test_the_uids_tasks_count_a_live_process_that_named_itself_in_bytes_that_are_not_utf8():
    # prctl(PR_SET_NAME) sets the comm the kernel shows in the status's Name line.
    namer = ("import ctypes, sys, time\n"
             "libc = ctypes.CDLL(None, use_errno=True)\n"
             "assert libc.prctl(15, ctypes.c_char_p(b'hog\\xff\\xfe'), 0, 0, 0) == 0\n"
             "print('named', flush=True)\n"
             "time.sleep(30)\n")
    proc = subprocess.Popen([sys.executable, "-c", namer], stdout=subprocess.PIPE)
    try:
        assert proc.stdout.readline() == b"named\n"
        assert Path(f"/proc/{proc.pid}/comm").read_bytes() == b"hog\xff\xfe\n"
        assert uid_tasks(os.getuid()) >= 1
    finally:
        proc.kill()
        proc.wait()


def test_the_uids_tasks_are_its_processes_threads_by_their_real_uid(tmp_path):
    _status(tmp_path, 10, 1000, 5)
    _status(tmp_path, 11, 1000, 1)
    _status(tmp_path, 12, 0, 40)
    _status(tmp_path, 13, 999, 7)  # its effective uid is 1000, but the kernel counts the real one
    (tmp_path / "14").mkdir()  # gone between the listing and the read
    (tmp_path / "self").symlink_to("10")
    (tmp_path / "meminfo").write_text("")
    assert uid_tasks(1000, proc=tmp_path) == 6


def test_a_steps_tasks_are_its_pids_count_or_its_cgroups_threads(tmp_path):
    user = tmp_path / "step_3" / "user"
    (user / "task_0").mkdir(parents=True)
    assert step_tasks(user) is None
    (user / "task_0" / "cgroup.threads").write_text("101\n102\n103\n")
    (user / "cgroup.threads").write_text("")
    assert step_tasks(user) == 3
    (user.parent / "pids.current").write_text("9\n")
    assert step_tasks(user) == 9
    (user / "pids.current").write_text("7\n")
    assert step_tasks(user) == 7


#: A ceiling counted against every task of the uid on the node, and one counted against its
#: namespace's own tasks, its parent levels against a hard limit of eight million.
UID = CeilingRule(ceiling=10_000, accounting="uid", parent_limit=None)
SPACE = CeilingRule(ceiling=10_000, accounting="user_namespace", parent_limit=8_000_000)
PID_MAX = 4_194_304


def _at_end(others: int | None, rule: CeilingRule = UID, *, node: int | None = 3_000,
            pid_max: int | None = PID_MAX) -> PoolAtEnd:
    return PoolAtEnd(at=1.0, others_tasks=others, rule=rule, node_tasks=node, pid_max=pid_max)


def test_counted_by_uid_an_ending_is_starved_when_the_others_alone_fill_its_ceiling_once_its_step_is_gone():
    full = math.ceil(0.95 * 10_000)
    assert starvation(_at_end(full)) == "the uid's other tasks on the node filled its process pool, 9500 of its ceiling 10000"
    assert starvation(_at_end(full - 1)) is None


def test_counted_by_user_namespace_the_others_starve_the_commands_only_through_the_parent_levels():
    # The commands' ceiling counts their own tasks: another build's commands at their whole
    # ceiling, or the uid at twice it, leave them all of theirs.
    assert starvation(_at_end(20_000, SPACE)) is None
    # The parent levels count every task of the uid against the runner's hard limit: others that
    # leave at most what 95% of the ceiling leaves, 500, starve them.
    assert starvation(_at_end(8_000_000 - 500, SPACE)) == (
        "the uid's other tasks on the node, 7999500, left it 500 of the runner's hard limit 8000000, which the parent "
        "levels of its user namespace count them against, under its ceiling 10000")
    assert starvation(_at_end(8_000_000 - 501, SPACE)) is None
    # An unlimited hard limit leaves only the node's pid_max.
    unlimited = CeilingRule(ceiling=10_000, accounting="user_namespace", parent_limit=None)
    assert starvation(_at_end(7_999_999, unlimited)) is None


@pytest.mark.parametrize("rule", [UID, SPACE], ids=["uid", "user_namespace"])
def test_a_node_whose_tasks_fill_its_pid_max_starves_any_step(rule):
    assert starvation(_at_end(10, rule, node=PID_MAX - 500)) == (
        "the node's tasks, 4193804 of its pid_max 4194304, left it 500 under its ceiling 10000")
    assert starvation(_at_end(10, rule, node=PID_MAX - 501)) is None


def test_a_count_that_failed_is_named_and_judges_nothing():
    assert _at_end(None).uncounted == _at_end(None, SPACE).uncounted == "the uid's tasks"
    assert _at_end(10, SPACE, node=None).uncounted == _at_end(10, SPACE, pid_max=None).uncounted == "the node's tasks"
    assert _at_end(10).uncounted is None and _at_end(10, SPACE).uncounted is None
    with pytest.raises(ValueError, match="were not counted"):
        starvation(_at_end(10, SPACE, node=None))
    with pytest.raises(ValueError, match="were not counted"):
        starvation(_at_end(None))


@pytest.mark.parametrize("missing", [{"node": None}, {"pid_max": None}, {"node": None, "pid_max": None}],
                         ids=["node-tasks", "pid-max", "both"])
def test_counted_by_uid_the_uids_count_judges_alone_when_the_nodes_failed(missing):
    # Under uid the ceiling counts every task of the uid, so the uid's count is the level that
    # decides: with it counted, a node-level count that failed leaves the end judged by it alone.
    full = math.ceil(0.95 * 10_000)
    assert _at_end(10, **missing).uncounted is None
    assert starvation(_at_end(10, **missing)) is None
    assert starvation(_at_end(full, **missing)) == (
        "the uid's other tasks on the node filled its process pool, 9500 of its ceiling 10000")


def test_counted_by_user_namespace_the_uids_count_is_needed_only_where_the_parent_levels_are_limited():
    # Per namespace the commands' ceiling counts their own tasks; the others reach them through
    # the parent levels' limit, the runner's hard limit, and through the node's pid_max. With no
    # hard limit the uid's count decides nothing, while the node's always may.
    unlimited = CeilingRule(ceiling=10_000, accounting="user_namespace", parent_limit=None)
    assert _at_end(None, unlimited).uncounted is None and starvation(_at_end(None, unlimited)) is None
    assert starvation(_at_end(None, unlimited, node=PID_MAX - 500)) == (
        "the node's tasks, 4193804 of its pid_max 4194304, left it 500 under its ceiling 10000")
    assert _at_end(10, unlimited, node=None).uncounted == "the node's tasks"


def test_a_start_is_refused_only_once_what_the_uids_tasks_count_against_is_full():
    assert (UID.uid_limit, UID.refusal(9_999)) == (10_000, None)
    assert UID.refusal(10_000) == ("the uid already holds 10000 tasks on the node, at or above its commands' "
                                   "process ceiling of 10000, so they could fork nothing")
    # Per namespace, a uid at the ceiling, or far past it, leaves the commands every fork.
    assert (SPACE.uid_limit, SPACE.refusal(10_000), SPACE.refusal(7_999_999)) == (8_000_000, None, None)
    assert "at or above the runner's hard limit of 8000000, which the parent levels" in SPACE.refusal(8_000_000)
    assert CeilingRule(ceiling=10_000, accounting="user_namespace", parent_limit=None).refusal(10**9) is None


def test_the_nodes_tasks_and_pid_max_are_read_from_proc(tmp_path):
    (tmp_path / "loadavg").write_text("37.10 45.99 49.20 75/3719 368294\n")
    (tmp_path / "pid_max").write_text("4194304\n")
    assert (node_tasks(tmp_path / "loadavg"), node_pid_max(tmp_path / "pid_max")) == (3719, 4194304)
    (tmp_path / "bad").write_text("x\n")
    for path in (tmp_path / "bad", tmp_path / "missing"):
        assert node_tasks(path) is None and node_pid_max(path) is None
    assert node_tasks() > 0 and node_pid_max() > 0


def test_a_peak_keeps_the_most_of_each_count_and_the_others_as_one_walk_read_them():
    # The others are the uid's less the step's as one walk read both: the most uid tasks less the
    # most step tasks, 10, would be no count the node ever held.
    first = PoolSample(at=1.0, uid_tasks=100, step_tasks=90, ceiling=1_000)
    second = PoolSample(at=2.0, uid_tasks=95, step_tasks=10, ceiling=1_000)
    unread = PoolSample(at=3.0, uid_tasks=120, step_tasks=None, ceiling=1_000)
    peak = peak_with(peak_with(None, first), second)
    assert peak == PoolPeak(walks=2, uid_tasks=100, step_tasks=90, others_tasks=85)
    # A walk that could not read the step's tasks adds to the uid's peak alone.
    assert peak_with(peak, unread) == PoolPeak(walks=3, uid_tasks=120, step_tasks=90, others_tasks=85)
    assert peak_with(None, unread) == PoolPeak(walks=1, uid_tasks=120, step_tasks=None, others_tasks=None)
    assert peak_with(peak_with(None, unread), second) == PoolPeak(walks=2, uid_tasks=120, step_tasks=10,
                                                                  others_tasks=85)
    # The uid's tasks are read once for a walk's pass, the step's after: a step that grew between
    # them reads as no others, never fewer.
    grown = PoolSample(at=4.0, uid_tasks=100, step_tasks=130, ceiling=1_000)
    assert peak_with(None, grown).others_tasks == 0


def test_the_runner_raises_its_soft_process_limit_to_its_hard_one_and_says_what_it_found():
    # In a child: the limits of this test process stay as they are.
    code = ("import resource; from hls.substrate.nproc import raise_soft_limit\n"
            "soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)\n"
            "low = 100 if hard == resource.RLIM_INFINITY else min(100, hard)\n"
            "resource.setrlimit(resource.RLIMIT_NPROC, (low, hard))\n"
            "found = raise_soft_limit()\n"
            "now = resource.getrlimit(resource.RLIMIT_NPROC)\n"
            "inf = None if hard == resource.RLIM_INFINITY else hard\n"
            "assert found == (low, inf), (found, low, inf)\n"
            "assert now == (hard, hard), now\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_the_hard_limit_reads_unlimited_as_none(monkeypatch):
    monkeypatch.setattr(nproc.resource, "getrlimit", lambda which: (5, resource.RLIM_INFINITY))
    assert nproc.hard_limit() is None
    monkeypatch.setattr(nproc.resource, "getrlimit", lambda which: (5, 4096))
    assert nproc.hard_limit() == 4096


class _Commands:
    """Answers ``scontrol show node`` with *stdout*, or fails."""

    def __init__(self, stdout: str, ok: bool = True) -> None:
        self.stdout, self.ok, self.calls = stdout, ok, []

    def __call__(self, *argv: str):
        self.calls.append(argv)
        from hls.substrate.slurm.commands import CommandResult

        return CommandResult(argv=tuple(argv), returncode=0 if self.ok else 1, stdout=self.stdout,
                             stderr="" if self.ok else "slurm_load_node error")


def test_the_node_size_comes_from_the_controllers_record_of_the_node():
    line = ("NodeName=node04 Arch=x86_64 CoresPerSocket=32 CPUAlloc=8 CPUTot=64 CfgTRES=cpu=64,mem=1000000M,"
            "billing=64,gres/gpu=8 AllocTRES=cpu=8,gres/gpu=1\n")
    commands = _Commands(line)
    assert node_size(commands, "node04") == NodeSize(gpus=8, cpus=64)
    assert commands.calls == [("scontrol", "show", "node", "--oneliner", "node04")]
    assert node_size(_Commands("NodeName=c1 CfgTRES=cpu=40,mem=10G\n"), "c1") == NodeSize(gpus=0, cpus=40)
    assert node_size(_Commands("", ok=False), "node04") == NodeSize(gpus=None, cpus=None)
    assert node_size(_Commands("NodeName=g1 CfgTRES=cpu=x,gres/gpu=y\n"), "g1") == NodeSize(gpus=None, cpus=None)


def node_root(at: Path, *, online: str | None, gpus: int | None) -> Path:
    """A node's file system root at *at* holding the node's own records of its size: the kernel's
    list of its online CPUs, *online* (None leaves it out), and the NVIDIA driver's entry for each
    of *gpus* GPUs (None leaves the driver's directory out, as on a node without the driver)."""
    if online is not None:
        (at / ONLINE_CPUS).parent.mkdir(parents=True)
        (at / ONLINE_CPUS).write_text(online)
    if gpus is not None:
        (at / NVIDIA_GPUS).mkdir(parents=True)
        for i in range(gpus):
            (at / NVIDIA_GPUS / f"0000:{0x18 + i:02x}:00.0").mkdir()
    return at


def test_a_node_reads_its_own_size_from_its_kernel_and_its_gpu_driver(tmp_path):
    # Every CPU the kernel has online and every GPU the driver sees, whatever the job's affinity
    # and cgroup let it use.
    assert own_node_size(node_root(tmp_path, online="0-63,128-191\n", gpus=8)) == NodeSize(gpus=8, cpus=128)


def test_a_node_without_a_gpu_driver_has_no_gpu_and_a_count_it_cannot_read_is_unknown(tmp_path):
    assert own_node_size(node_root(tmp_path / "cpu", online="0-191\n", gpus=None)) == NodeSize(gpus=0, cpus=192)
    assert own_node_size(node_root(tmp_path / "unlisted", online=None, gpus=2)) == NodeSize(gpus=2, cpus=None)
    assert own_node_size(node_root(tmp_path / "garbled", online="zero\n", gpus=2)) == NodeSize(gpus=2, cpus=None)
    # A driver whose list of GPUs cannot be read is no answer either.
    unreadable = node_root(tmp_path / "unreadable", online="0-7\n", gpus=None)
    (unreadable / NVIDIA_GPUS).parent.mkdir(parents=True)
    (unreadable / NVIDIA_GPUS).write_text("")
    assert own_node_size(unreadable) == NodeSize(gpus=None, cpus=8)


def test_a_launcher_that_asks_no_controller_reads_its_nodes_own_size(tmp_path):
    root = node_root(tmp_path, online="0-15\n", gpus=4)

    class Bare(Launcher):
        node_root = root

        def launch(self, spec, *, share_gpus_of=None): ...

        def terminate(self, handles): ...

    assert Bare().node_size() == NodeSize(gpus=4, cpus=16)
