"""The frozen conversation a trajectory resumes from: its file, and the resume."""
from __future__ import annotations

import json
import os

import pytest

from tasks.agentic_coding import prefix as pfx
from tasks.agentic_coding.mini_swe_loop import (
    INSTANCE_TEMPLATE, PREFILL_TOKENS_EXTRA, SYSTEM_TEMPLATE,
    WORK_DIR, SUBMIT_SENTINEL, wire_messages,
)

from tasks.agentic_coding.tests.test_mini_swe_loop import (
    SUBMIT, call, counter, drive,
)

CHECKPOINT = "zai-org/GLM-5.3-Flash"
STATEMENT = "Fix the bug."


def opening_messages(statement: str = STATEMENT) -> list[dict]:
    """The two messages the loop renders at turn 0, as it renders them.

    Built the way ``DefaultAgent.run`` does, so a prefix carrying these is one the
    templates still produce and ``check_opening`` accepts."""
    variables = {"task": statement, "work_dir": WORK_DIR,
                 "submit_sentinel": SUBMIT_SENTINEL,
                 "output_limit_chars": 32_000, "command_timeout_sec": 300,
                 "step_limit": 10, "cost_limit": 0.0, "max_context_len": None,
                 "max_output_tokens": None, "max_consecutive_format_errors": 3,
                 "system_template": "", "instance_template": "",
                 "wall_time_limit_seconds": 0, "n_model_calls": 0, "model_cost": 0.0,
                 "elapsed_seconds": 0}
    return [{"role": "system", "content": SYSTEM_TEMPLATE.format(**variables)},
            {"role": "user", "content": INSTANCE_TEMPLATE.format(**variables)}]


def a_prefix(*, turns: int = 12, commands_run: int = 14,
             consecutive_format_errors: int = 0, depth: int = 150_000,
             body: list[dict] | None = None, messages: list[dict] | None = None,
             task_name: str = "git-to-zig", statement: str = STATEMENT,
             state_manifest: dict | None = None,
             checkpoint: str = CHECKPOINT) -> pfx.Prefix:
    """A prefix whose opening is what the templates render now.

    ``statement`` is the task statement the instance message was rendered around, so
    a prefix meant for a given seed has to be cut with that seed's own.

    ``body`` replaces the turns after the opening; ``messages`` replaces the whole
    conversation, opening included, which is how a test cuts a prefix under templates
    that no longer render."""
    default_body = [
        {"role": "assistant", "content": "I will look around.",
         "reasoning_content": "The repository is large, so start with a listing."},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>src\n</output>"},
    ]
    whole = (messages if messages is not None
             else opening_messages(statement) + (default_body if body is None
                                                 else body))
    return pfx.build(task_name=task_name, depth_tokens=depth, prefill_tokens=depth + 71,
                     turns=turns, commands_run=commands_run,
                     consecutive_format_errors=consecutive_format_errors,
                     checkpoint=checkpoint, image=f"{task_name}.sif",
                     state_manifest=state_manifest, messages=whole)


# --- the file --------------------------------------------------------------

def test_a_prefix_round_trips_through_its_file(tmp_path):
    written = a_prefix()
    pfx.write(tmp_path, written)
    assert pfx.load(tmp_path, "git-to-zig") == written


def test_the_wire_shape_survives_the_round_trip(tmp_path):
    # The oracle teacher-forces the messages the harness sent, so a resumed prompt is
    # only reproducible if the stored list is what goes on the wire. An assistant turn
    # keeps its reasoning beside its content, which is how this checkpoint's chat
    # template renders the thinking back into the prompt; dropping it would render a
    # shorter prompt than the artifact was served.
    pfx.write(tmp_path, a_prefix())
    loaded = pfx.load(tmp_path, "git-to-zig")
    assistant = [m for m in loaded.messages if m["role"] == "assistant"]
    assert assistant and assistant[0]["reasoning_content"]
    assert list(loaded.messages) == wire_messages(list(loaded.messages))


def test_an_edited_prefix_is_refused(tmp_path):
    # The digest is what the corpus pins, so a file edited after it was cut must not
    # load under a fingerprint that has not moved.
    path = pfx.write(tmp_path, a_prefix())
    payload = json.loads(path.read_text())
    payload["messages"][-1]["content"] = "<returncode>0</returncode>\n<output>x</output>"
    path.write_text(json.dumps(payload))
    with pytest.raises(pfx.PrefixError, match="edited after it was cut"):
        pfx.load(tmp_path, "git-to-zig")


def test_a_prefix_that_is_not_the_declared_one_is_refused(tmp_path):
    pfx.write(tmp_path, a_prefix())
    with pytest.raises(pfx.PrefixError, match="not the one this task was calibrated"):
        pfx.load(tmp_path, "git-to-zig", expect_digest="0" * 64)


def test_a_prefix_cut_against_other_weights_is_refused(tmp_path):
    # A conversation is the writing of the weights that produced it.
    pfx.write(tmp_path, a_prefix(checkpoint="some/other-model"))
    with pytest.raises(pfx.PrefixError, match="writing of the"):
        pfx.load(tmp_path, "git-to-zig", expect_checkpoint=CHECKPOINT)


def test_the_checkpoint_is_inside_the_digest():
    # Same conversation, different weights, different address -- so a model change
    # cannot leave a stale prefix running under a corpus pin that did not move.
    assert a_prefix().digest != a_prefix(checkpoint="some/other-model").digest


def test_each_stored_field_moves_the_digest():
    # The digest is the corpus's whole handle on a prefix. A field it did not cover
    # could change under a fingerprint that had not.
    base = a_prefix()
    for change in (dict(turns=13), dict(commands_run=15), dict(depth=300_000),
                   dict(consecutive_format_errors=2), dict(task_name="other-task")):
        assert a_prefix(**change).digest != base.digest, change


def test_a_missing_prefix_says_to_stage_it(tmp_path):
    with pytest.raises(pfx.PrefixError, match="stage it before the run"):
        pfx.load(tmp_path, "git-to-zig")


def test_one_tasks_prefix_cannot_load_as_anothers(tmp_path):
    # A mis-staged file would otherwise resume one task's conversation inside
    # another's container, which reads as a model that lost the plot.
    path = pfx.prefix_path(tmp_path, "git-to-zig")
    path.write_text(json.dumps(a_prefix(task_name="other-task").payload()))
    with pytest.raises(pfx.PrefixError, match="cut from 'other-task'"):
        pfx.load(tmp_path, "git-to-zig")


# --- the sandbox state beside it -------------------------------------------

def stage_state(tmp_path, task_name="git-to-zig", entries=(("src/main.zig", 12),)):
    """Write a state directory as a pre-roll's captured overlay upper would look."""
    root = pfx.state_path(tmp_path, task_name)
    for rel, size in entries:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    return root


def test_the_manifest_records_every_entry_not_only_the_files(tmp_path):
    # A deletion is recorded by a marker whose contents are empty, so a manifest over
    # file contents alone would miss exactly the difference between a prefix that
    # deleted a file and one that did not.
    root = stage_state(tmp_path, entries=(("src/main.zig", 12), (".wh.removed", 0)))
    manifest = pfx.manifest_of(root)
    assert manifest["src/main.zig"] == 12
    assert manifest[".wh.removed"] == 0
    assert manifest["src"] == pfx.MARKER_SIZE


def test_a_prefix_with_no_state_directory_has_an_empty_manifest(tmp_path):
    assert pfx.manifest_of(tmp_path / "nothing-here") == {}


def test_a_replaced_directory_is_not_an_ordinary_one(tmp_path):
    # The difference is the whole of what it means: merged, the resumed container also
    # sees what the image had there; replaced, it sees only what the pre-roll left. A
    # manifest that read them the same would address two different filesystems alike.
    merged = tmp_path / "state" / "merged"
    replaced = tmp_path / "state" / "replaced"
    merged.mkdir(parents=True)
    replaced.mkdir()
    try:
        os.setxattr(replaced, pfx.OPAQUE_XATTR, b"y")
    except OSError:
        pytest.skip("this filesystem carries no extended attributes")
    manifest = pfx.manifest_of(tmp_path / "state")
    assert manifest["merged"] == pfx.MARKER_SIZE
    assert manifest["replaced"] == pfx.OPAQUE_SIZE


def test_a_deletion_marker_is_recorded_by_being_there(tmp_path):
    # It is an ordinary empty file, so what carries the meaning is its presence and
    # its name. A manifest over contents alone would see nothing at all.
    root = stage_state(tmp_path, entries=((".wh.deleted", 0),))
    assert pfx.manifest_of(root)[".wh.deleted"] == 0


def test_state_and_conversation_are_addressed_together():
    # Two prefixes over the same conversation and different state are two different
    # prefixes; the corpus pin has to tell them apart or a re-captured directory
    # would run under a fingerprint that had not moved.
    plain = a_prefix()
    with_state = a_prefix(state_manifest={"src/main.zig": 12})
    assert plain.digest != with_state.digest


def test_a_state_directory_that_does_not_match_is_refused(tmp_path):
    # The failure this catches: a directory staged short, or from another task. The
    # session would otherwise resume a conversation about files that are not there,
    # and read as a model that lost its place.
    prefix = a_prefix(state_manifest={"src/main.zig": 12, "src": pfx.MARKER_SIZE})
    pfx.write(tmp_path, prefix)
    stage_state(tmp_path, entries=(("src/main.zig", 9),))
    with pytest.raises(pfx.PrefixError, match="not the one this prefix was cut with"):
        pfx.load(tmp_path, "git-to-zig")


def test_a_state_directory_that_matches_loads(tmp_path):
    # The other direction, so the check is pinned as not firing on a good prefix.
    staged = stage_state(tmp_path)
    prefix = a_prefix(state_manifest=pfx.manifest_of(staged))
    pfx.write(tmp_path, prefix)
    assert pfx.load(tmp_path, "git-to-zig") == prefix


def test_a_missing_state_directory_is_refused(tmp_path):
    # A prefix whose file was staged and whose state was not. The conversation alone
    # would drive a trajectory through a filesystem that never had the work done.
    pfx.write(tmp_path, a_prefix(state_manifest={"src/main.zig": 12}))
    with pytest.raises(pfx.PrefixError, match="not the one this prefix was cut with"):
        pfx.load(tmp_path, "git-to-zig")


# --- resuming from it ------------------------------------------------------

async def test_a_resumed_trajectory_sends_the_frozen_conversation():
    # The point of the whole mechanism: the first request of a resumed session
    # carries the prefix, so its prefill is the depth the prefix was cut at rather
    # than an opening pair.
    prefix = a_prefix()
    rec, _ = await drive([SUBMIT], prefix=prefix, count_prompt_tokens=counter(4))
    sent = rec.requests[0]["metadata"]["messages"]
    assert sent[:len(prefix.messages)] == [dict(m) for m in prefix.messages]


async def test_a_cold_trajectory_is_unchanged():
    # Passing no prefix must leave every existing trajectory exactly as it was.
    cold, _ = await drive([SUBMIT], count_prompt_tokens=counter(4))
    assert len(cold.requests[0]["metadata"]["messages"]) == 2


async def test_the_resumed_session_counts_its_own_turns_from_zero():
    # The turn budget is what a session serves, not where it began, so every
    # contestant serves the same number of turns from the same prefix. The driver
    # numbers a session's fault events from zero too, so the two agree.
    rec, summary = await drive([call("ls"), SUBMIT], prefix=a_prefix(turns=12),
                               count_prompt_tokens=counter(4))
    assert [ev["turn_index"] for ev in rec.events] == [0, 1]
    assert summary["turns"] == 2


async def test_every_event_carries_the_depth_it_resumed_at():
    # Turns are session-local, so without this the record cannot say how deep the
    # conversation actually was.
    rec, summary = await drive([SUBMIT], prefix=a_prefix(turns=12),
                               count_prompt_tokens=counter(4))
    assert rec.events[0]["extras"]["prefix_turns"] == 12
    assert summary["prefix_turns"] == 12


async def test_commands_carry_over_so_the_shape_reads_as_one_trajectory():
    rec, summary = await drive([call("ls"), SUBMIT], prefix=a_prefix(commands_run=14),
                               count_prompt_tokens=counter(4))
    assert summary["commands_run"] == 16
    assert rec.events[-1]["extras"]["commands_run"] == 16


async def test_the_repair_budget_carries_over():
    # A stopping rule, and the reason it is stored. This prefix was cut with two
    # unparseable replies behind it, so one more ends the trajectory -- where a
    # session that started the counter at zero would take three and serve two turns
    # of traffic that a continuous trajectory never would.
    rec, summary = await drive(["no call here"] * 3, prefix=a_prefix(
        consecutive_format_errors=2), count_prompt_tokens=counter(4))
    assert summary["stop_reason"] == "format_repair_exhausted"
    assert summary["turns"] == 1
    fresh, fresh_summary = await drive(["no call here"] * 3, prefix=a_prefix(
        consecutive_format_errors=0), count_prompt_tokens=counter(4))
    assert fresh_summary["turns"] == 3


async def test_the_resumed_prompt_length_is_the_one_that_is_recorded():
    # The prefill stamped on a resumed turn has to be the deep prompt, since that is
    # what any budget reading it prices the turn against.
    prefix = a_prefix(body=[{"role": "user", "content": "x" * 40_000}])
    rec, _ = await drive([SUBMIT], prefix=prefix, count_prompt_tokens=counter(4))
    cold, _ = await drive([SUBMIT], count_prompt_tokens=counter(4))
    deep = rec.events[0]["extras"][PREFILL_TOKENS_EXTRA]
    assert deep > 10_000
    assert deep > 10 * cold.events[0]["extras"][PREFILL_TOKENS_EXTRA]


async def test_a_prefix_cut_under_different_templates_is_refused():
    # Nothing downstream would catch this: both openings are well-formed and the gate
    # reads the served distribution. A resumed session would simply be served a
    # different prompt from a cold one in the same drive.
    stale = opening_messages()
    stale[0]["content"] += "\nAlso, always run tests first."
    prefix = a_prefix(messages=stale)
    with pytest.raises(pfx.PrefixError, match="templates no longer render"):
        await drive([SUBMIT], prefix=prefix, count_prompt_tokens=counter(4))


async def test_the_opening_check_passes_on_what_the_templates_render_now():
    # The other direction, so the check is pinned as not firing on a good prefix --
    # one that rejected every prefix would be indistinguishable from one that worked.
    rec, summary = await drive([SUBMIT], prefix=a_prefix(), count_prompt_tokens=counter(4))
    assert summary["turns"] == 1 and len(rec.events) == 1
