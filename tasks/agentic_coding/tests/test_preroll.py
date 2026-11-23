"""The offline pass that cuts prefixes: where it cuts, and what it records."""
from __future__ import annotations

import pytest

from tasks.agentic_coding import prefix as prefix_format
from tasks.agentic_coding.mini_swe_loop import (
    MiniSweLoop, PREFILL_TOKENS_EXTRA, STOP_SUBMITTED,
)
from tasks.agentic_coding.preroll import (
    Preroll, STOP_BANDS_CUT, STOP_COST, STOP_DEADLINE, declared_prefixes,
    next_target,
)
from tasks.agentic_coding.tests.test_mini_swe_loop import (
    DENSE, Recorder, SUBMIT, call, counter,
)

CHECKPOINT = "zai-org/GLM-5.3-Flash"


def captures() -> tuple[list, object]:
    """A state capture that records what it was asked for and names an image."""
    seen: list[tuple[str, int, str]] = []

    def capture(task_name: str, depth: int, session_id: str) -> str:
        seen.append((task_name, depth, session_id))
        return f"{task_name}@{depth}.sif"

    return seen, capture


async def cut(replies, *, targets, chars_per_token=4, tool_result=None,
              capture=None, stop_after_band=None) -> tuple:
    """Run one pre-roll pass against a scripted server."""
    rec = Recorder(list(replies), tool_result)
    count = counter(chars_per_token)
    seen, default_capture = captures()
    preroll = Preroll(loop=MiniSweLoop(count_prompt_tokens=count),
                      checkpoint=CHECKPOINT, depth_targets=targets,
                      capture_state=capture or default_capture,
                      count_prompt_tokens=count)
    result = await preroll.run(task_name="git-to-zig",
                               problem_statement="Port it to Zig.",
                               session_id="s0", base_metadata={"tenant_id": "t0"},
                               complete=rec.complete, run_tool=rec.run_tool,
                               stop_after_band=stop_after_band)
    return result, seen, rec


# --- where it cuts ---------------------------------------------------------

async def test_a_target_is_a_floor_the_prompt_crosses():
    # Observations run to tens of thousands of tokens, so a prompt steps past a
    # target rather than landing on it. The prefix records the target it was cut for
    # and the depth it actually reached, and they are not the same number.
    result, _, _ = await cut([call("xxd fixture.bin")] * 6 + [SUBMIT],
                             targets=[1_000],
                             tool_result={"output": DENSE * 200, "exit_code": 0})
    assert result.reached() == [1_000]
    cut_at = result.cuts[0]
    assert cut_at.depth_tokens == 1_000
    assert cut_at.prefill_tokens >= 1_000


async def test_one_pass_cuts_every_band_the_trajectory_reaches():
    # The reason a pass is per task rather than per band: one trajectory supplies all
    # of them, in the order it grows through them.
    result, seen, _ = await cut([call("xxd fixture.bin")] * 12 + [SUBMIT],
                                targets=[500, 2_000, 8_000],
                                tool_result={"output": DENSE * 200, "exit_code": 0})
    assert result.reached() == [500, 2_000, 8_000]
    assert [d for _, d, _ in seen] == [500, 2_000, 8_000]
    assert result.stop_reason == STOP_SUBMITTED


async def test_a_trajectory_that_ends_early_supplies_fewer_bands():
    # A Terminal-Bench trajectory ends when its task is done, which is why the deep
    # bands come from elsewhere. Not a failure: the pass still reports what it got.
    result, _, _ = await cut([SUBMIT], targets=[100_000])
    assert result.reached() == []
    assert result.stop_reason == STOP_SUBMITTED


def test_a_prompt_that_jumps_two_bands_cuts_the_deeper_one():
    # Taking both would cut two prefixes from one conversation, which are the same
    # conversation under two names. The shallower band goes unfilled and says so.
    assert next_target([500, 2_000], cut=[]) == 500
    assert next_target([500, 2_000], cut=[2_000]) == 500
    assert next_target([500, 2_000], cut=[500, 2_000]) is None


# --- what it records -------------------------------------------------------

async def test_a_cut_prefix_resumes_the_conversation_it_was_cut_from():
    # The whole output: what the pass writes has to be what a graded session can
    # resume, including the opening the templates render.
    result, _, rec = await cut([call("ls")] * 4 + [SUBMIT], targets=[100])
    frozen = result.cuts[0]
    sent = rec.requests[frozen.turns]["metadata"]["messages"]
    assert [dict(m) for m in frozen.messages] == sent


async def test_the_state_capture_runs_while_the_container_is_alive():
    # The conversation and the filesystem have to be the same instant, so the capture
    # is called during the pass rather than afterwards from a record of it.
    order: list[str] = []
    rec = Recorder([call("ls")] * 4 + [SUBMIT])
    count = counter(4)

    def capture(task_name, depth, session_id):
        order.append(f"capture@{len(rec.requests)}")
        return "img.sif"

    preroll = Preroll(loop=MiniSweLoop(count_prompt_tokens=count),
                      checkpoint=CHECKPOINT, depth_targets=[100],
                      capture_state=capture, count_prompt_tokens=count)
    await preroll.run(task_name="t", problem_statement="p", session_id="s0",
                      base_metadata={}, complete=rec.complete, run_tool=rec.run_tool)
    # Captured before the trajectory finished, not after the last reply.
    assert order and int(order[0].split("@")[1]) < len(rec.requests)


async def test_the_capture_is_asked_for_this_sessions_state():
    result, seen, _ = await cut([call("ls")] * 4 + [SUBMIT], targets=[100])
    assert seen == [("git-to-zig", 100, "s0")]


async def test_the_prefix_records_the_base_image_not_a_derivative(tmp_path):
    # A resuming session starts from the image the pass ran, with the captured
    # directory stacked under its own writable layer. Nothing is baked, so the image
    # store does not grow and the staged .sif stays the one the corpus pinned.
    rec = Recorder([call("ls"), SUBMIT])
    count = counter(4)
    preroll = Preroll(loop=MiniSweLoop(count_prompt_tokens=count),
                      checkpoint=CHECKPOINT, depth_targets=[100],
                      capture_state=lambda *a: str(tmp_path),
                      count_prompt_tokens=count, image="git-to-zig.sif")
    result = await preroll.run(task_name="git-to-zig", problem_statement="p",
                               session_id="s0", base_metadata={},
                               complete=rec.complete, run_tool=rec.run_tool)
    assert result.cuts[0].image == "git-to-zig.sif"


async def test_the_prefix_records_what_the_captured_directory_held(tmp_path):
    # The digest covers the filesystem a session resumes into, not only the
    # conversation: two prefixes over the same messages and different state are two
    # different prefixes, and the corpus pin has to tell them apart.
    (tmp_path / "built").mkdir()
    (tmp_path / "built" / "out.o").write_bytes(b"x" * 40)
    rec = Recorder([call("ls"), SUBMIT])
    count = counter(4)
    preroll = Preroll(loop=MiniSweLoop(count_prompt_tokens=count),
                      checkpoint=CHECKPOINT, depth_targets=[100],
                      capture_state=lambda *a: str(tmp_path),
                      count_prompt_tokens=count, image="i.sif")
    result = await preroll.run(task_name="t", problem_statement="p", session_id="s0",
                               base_metadata={}, complete=rec.complete,
                               run_tool=rec.run_tool)
    manifest = result.cuts[0].seal().state_manifest
    assert manifest["built/out.o"] == 40
    assert manifest["built"] == prefix_format.MARKER_SIZE


async def test_the_prefix_records_the_counters_a_resume_needs():
    result, _, _ = await cut([call("ls")] * 6 + [SUBMIT], targets=[200])
    frozen = result.cuts[0]
    # Commands run before the cut, not including the turn about to be sent.
    assert frozen.commands_run == frozen.turns
    assert frozen.checkpoint == CHECKPOINT


async def test_the_repair_counter_is_the_one_the_next_turn_would_have():
    # The reason the hook runs before a turn rather than after one: upstream resets
    # the counter after a clean step and increments it after a failed one, so a cut
    # taken at the end of a step freezes a budget that is about to change. Two
    # unparseable replies, then the cut: a resumed trajectory has one repair left.
    # The target lands the cut on turn 2, whose measured prompt is 888 tokens.
    result, _, _ = await cut(["no call here", "no call here"] + [call("ls")] * 4,
                             targets=[880])
    frozen = result.cuts[0]
    assert frozen.turns == 2
    assert frozen.consecutive_format_errors == 2


async def test_a_cut_prefix_is_loadable(tmp_path):
    result, _, _ = await cut([call("ls")] * 4 + [SUBMIT], targets=[100])
    frozen = result.cuts[0].seal()
    prefix_format.write(tmp_path, frozen)
    assert prefix_format.load(tmp_path, "git-to-zig",
                              expect_digest=frozen.digest,
                              expect_checkpoint=CHECKPOINT) == frozen


async def test_a_prefix_describes_the_state_as_it_is_left_not_as_it_was_cut(tmp_path):
    # The container is alive when a band is cut and it keeps writing until it is
    # stopped. A manifest taken at the cut therefore describes a directory that has
    # moved on by the time anything reads it, and ``load`` refuses the prefix -- at
    # the first resumed trajectory of a graded run, not here. So a cut carries no
    # manifest at all, and sealing it takes one from the directory as the caller left
    # it. Written after teardown here, which is what a real pass's teardown does.
    state = prefix_format.state_path(tmp_path, "git-to-zig")
    (state / "work").mkdir(parents=True)
    (state / "work" / "built.o").write_bytes(b"x" * 8)
    result, _, _ = await cut([call("ls")] * 4 + [SUBMIT], targets=[100],
                             capture=lambda *a: str(state))
    (state / "work" / "flushed-during-teardown").write_bytes(b"y")

    frozen = result.cuts[0].seal()
    prefix_format.write(tmp_path, frozen)
    loaded = prefix_format.load(tmp_path, "git-to-zig")
    assert loaded == frozen
    assert "work/flushed-during-teardown" in loaded.state_manifest


# --- the growth curve ------------------------------------------------------

async def test_every_turn_is_measured_even_when_no_band_is_reached():
    # The pass exists as much for this as for the prefixes: the per-turn growth rate
    # is what the depth targets are computed from.
    result, _, _ = await cut([call("ls")] * 5 + [SUBMIT], targets=[10**9])
    assert result.reached() == []
    assert [d.turn for d in result.depths] == [0, 1, 2, 3, 4, 5]
    assert result.growth_per_turn() > 0
    depths = [d.prefill_tokens for d in result.depths]
    assert depths == sorted(depths)


async def test_growth_needs_two_turns_to_be_a_rate():
    result, _, _ = await cut([SUBMIT], targets=[10**9])
    assert result.growth_per_turn() == 0.0


# --- what the pass lets a caller record and stop on ------------------------

async def test_a_pass_hands_every_turn_to_the_caller():
    # The pass scores nothing and needs none of this, but it is the only cheap
    # chance to measure this loop against depth on the baseline server, so the events
    # have to be reachable rather than dropped.
    seen: list[dict] = []

    async def emit(turn):
        seen.append(turn)

    rec = Recorder([call("ls"), call("pwd"), SUBMIT])
    count = counter(4)
    preroll = Preroll(loop=MiniSweLoop(count_prompt_tokens=count),
                      checkpoint=CHECKPOINT, depth_targets=[10**9],
                      capture_state=lambda *a: "s", count_prompt_tokens=count)
    await preroll.run(task_name="t", problem_statement="p", session_id="s0",
                      base_metadata={}, complete=rec.complete,
                      run_tool=rec.run_tool, emit=emit)
    assert len(seen) == 3
    # The two things the run is there to measure ride on the turn already.
    assert all(PREFILL_TOKENS_EXTRA in t["extras"] for t in seen)
    assert seen[0]["extras"]["n_tool_calls"] == 1


async def test_a_pass_can_stop_once_it_has_cut_what_was_asked_for():
    # For taking one prefix and checking it resumes before spending the rest of the
    # pass on a mechanism that might not work.
    result, _, rec = await cut([call("xxd f")] * 12 + [SUBMIT],
                               targets=[500, 2_000, 8_000],
                               tool_result={"output": DENSE * 200, "exit_code": 0},
                               stop_after_band=1)
    assert result.reached() == [500]
    assert result.stop_reason == STOP_BANDS_CUT
    assert len(rec.requests) < 12


async def test_stopping_early_is_not_read_as_a_task_running_out_of_work():
    # Its own reason, so a deliberately short pass and a trajectory that finished
    # are never confused in the record.
    result, _, _ = await cut([call("xxd f")] * 12 + [SUBMIT], targets=[500],
                             tool_result={"output": DENSE * 200, "exit_code": 0},
                             stop_after_band=1)
    assert result.stop_reason != STOP_SUBMITTED


# --- the pin a pass produces -----------------------------------------------

def sealed(task_name="git-to-zig", depths=(0, 150_000, 600_000)):
    """The prefixes one task's passes wrote, as the pin is given them."""
    from tasks.agentic_coding.tests.test_prefix import a_prefix

    return [a_prefix(task_name=task_name, depth=d) for d in depths]


def test_the_pin_names_the_digest_of_the_band_each_task_was_given():
    written = sealed()
    pin = declared_prefixes(written, {"git-to-zig": 150_000})
    assert pin == {"git-to-zig": next(p.digest for p in written
                                      if p.depth_tokens == 150_000)}


def test_a_task_left_out_of_the_choice_is_left_out_of_the_pin():
    # Absent from the pin is a cold start, which is what an empty map already means.
    assert declared_prefixes(sealed(), {}) == {}


def test_a_band_the_pass_never_reached_is_refused():
    # The one error that cannot pass quietly: the pin would name a prefix no pass
    # wrote, and it would surface as a missing file at the first resumed trajectory
    # of a graded run rather than here.
    with pytest.raises(ValueError, match="reached \\[0\\]"):
        declared_prefixes(sealed(depths=(0,)), {"git-to-zig": 600_000})


def test_a_task_with_no_pass_at_all_is_refused():
    with pytest.raises(ValueError, match="never cut"):
        declared_prefixes([], {"git-to-zig": 600_000})


async def test_a_pass_with_nothing_to_count_refuses_rather_than_guessing():
    # Depth is the whole basis for where a prefix is cut, so a pass that cannot count
    # the prompt must not cut one anywhere.
    rec = Recorder([call("ls")] * 3)

    async def no_count(messages):
        return None

    preroll = Preroll(loop=MiniSweLoop(), checkpoint=CHECKPOINT, depth_targets=[100],
                      capture_state=lambda *a: "img.sif",
                      count_prompt_tokens=no_count)
    with pytest.raises(RuntimeError, match="cannot count the prompt"):
        await preroll.run(task_name="t", problem_statement="p", session_id="s0",
                          base_metadata={}, complete=rec.complete,
                          run_tool=rec.run_tool)


# --- the wall-clock bound --------------------------------------------------

async def test_the_bound_cuts_where_it_is_rather_than_returning_nothing():
    # A bound whose firing destroys the work is one nobody can afford to set. The
    # pass ran out of time short of its target, so it yields a shallower band rather
    # than nothing, and the prefix records the depth actually reached.
    rec = Recorder([call("ls")] * 6 + [SUBMIT])
    count = counter(4)
    preroll = Preroll(loop=MiniSweLoop(count_prompt_tokens=count),
                      checkpoint=CHECKPOINT, depth_targets=[10**9],
                      capture_state=lambda *a: "s", count_prompt_tokens=count)
    result = await preroll.run(task_name="t", problem_statement="p", session_id="s0",
                               base_metadata={}, complete=rec.complete,
                               run_tool=rec.run_tool, deadline_sec=0.0)
    assert result.stop_reason == STOP_DEADLINE
    assert len(result.cuts) == 1
    frozen = result.cuts[0]
    # The band it was cut for IS the depth reached, since no target was crossed.
    assert frozen.depth_tokens == frozen.prefill_tokens > 0


async def test_the_cost_bound_cuts_where_it_is_rather_than_returning_nothing():
    # The wall bound stops a pass that is slow and the cost bound one that is
    # expensive; against a metered server those are different runaways. Same
    # semantics as the deadline: it yields a shallower band and records the depth.
    rec = Recorder([call("ls")] * 6 + [SUBMIT])
    count = counter(4)
    spent = [0.0]

    async def complete(request):
        spent[0] += 1.0
        return await rec.complete(request)

    preroll = Preroll(loop=MiniSweLoop(count_prompt_tokens=count),
                      checkpoint=CHECKPOINT, depth_targets=[10**9],
                      capture_state=lambda *a: "s", count_prompt_tokens=count)
    result = await preroll.run(task_name="t", problem_statement="p", session_id="s0",
                               base_metadata={}, complete=complete,
                               run_tool=rec.run_tool, cost_budget_usd=2.5,
                               spent_usd=lambda: spent[0])
    assert result.stop_reason == STOP_COST
    assert len(result.cuts) == 1
    frozen = result.cuts[0]
    assert frozen.depth_tokens == frozen.prefill_tokens > 0
    # Stopped once the third turn's spend crossed the budget, not at the end.
    assert frozen.turns == 3


async def test_a_pass_inside_its_cost_bound_is_untouched_by_it():
    rec = Recorder([call("ls")] * 3 + [SUBMIT])
    count = counter(4)
    result, _, _ = await cut([call("ls")] * 3 + [SUBMIT], targets=[100])
    assert result.stop_reason == STOP_SUBMITTED
    preroll = Preroll(loop=MiniSweLoop(count_prompt_tokens=count),
                      checkpoint=CHECKPOINT, depth_targets=[100],
                      capture_state=lambda *a: "s", count_prompt_tokens=count)
    result = await preroll.run(task_name="t", problem_statement="p", session_id="s0",
                               base_metadata={}, complete=rec.complete,
                               run_tool=rec.run_tool, cost_budget_usd=1000.0,
                               spent_usd=lambda: 0.01)
    assert result.stop_reason == STOP_SUBMITTED
    assert result.reached() == [100]


async def test_a_cost_bound_with_no_way_to_read_the_spend_is_refused():
    # A budget nothing reports against is a budget that never fires, and the pass
    # would run to the wall looking bounded.
    rec = Recorder([call("ls"), SUBMIT])
    count = counter(4)
    preroll = Preroll(loop=MiniSweLoop(count_prompt_tokens=count),
                      checkpoint=CHECKPOINT, depth_targets=[100],
                      capture_state=lambda *a: "s", count_prompt_tokens=count)
    with pytest.raises(ValueError, match="no way to read what it has spent"):
        await preroll.run(task_name="t", problem_statement="p", session_id="s0",
                          base_metadata={}, complete=rec.complete,
                          run_tool=rec.run_tool, cost_budget_usd=5.0)


async def test_a_pass_inside_its_bound_is_untouched_by_it():
    # The bound must not change a pass that finishes in time.
    rec = Recorder([call("ls")] * 3 + [SUBMIT])
    count = counter(4)
    preroll = Preroll(loop=MiniSweLoop(count_prompt_tokens=count),
                      checkpoint=CHECKPOINT, depth_targets=[100],
                      capture_state=lambda *a: "s", count_prompt_tokens=count)
    result = await preroll.run(task_name="t", problem_statement="p", session_id="s0",
                               base_metadata={}, complete=rec.complete,
                               run_tool=rec.run_tool, deadline_sec=3600.0)
    assert result.stop_reason == STOP_SUBMITTED
    assert result.reached() == [100]
