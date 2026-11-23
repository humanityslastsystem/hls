"""The orchestrator tests written against ``FakeScheduler`` (``tests.pipeline.test_orchestrator``), run
against the local scheduler through ``LocalFake``.

Left out: tests that remove or purge a job (``jobs.pop``, ``purge``), which a local scheduler never
does, and tests whose scheduler is a subclass of the fake (``Answering``, ``SlowPreflight``,
``NoCancel``, ``RewireFails``, ``SlowEnd``) or that put a ``FakeJob`` in it by hand, which test the
orchestrator against answers no real scheduler gives on demand. Left out too: a test that starts a
dependent while the job it waits for still runs, which ``afterok`` forbids. A test that needs a
job to end TIMEOUT or OUT_OF_MEMORY on demand is skipped."""

from __future__ import annotations

import pytest

import tests.pipeline.test_orchestrator as orchestrator_tests
from tests.substrate.local.local_fake import LocalFake

RUN = [
    "test_the_recorded_sbatch_line_has_every_secret_value_scrubbed",
    "test_a_taking_over_orchestrator_records_its_sbatch_lines_scrubbed",
    "test_an_orchestrator_that_could_not_read_the_secrets_file_only_cancels",
    "test_a_cancel_of_a_failed_instance_cancels_and_leaves_its_records_to_a_full_take_over",
    "test_a_take_over_whose_writer_fails_on_the_left_records_still_cancels",
    "test_a_lease_lost_while_the_left_records_are_written_stops_the_pass",
    "test_a_writer_that_refuses_when_the_instance_ends_holds_back_no_cancel_and_is_asked_once",
    "test_a_lease_lost_while_the_records_are_written_ends_the_pass_with_nothing_recorded",
    "test_a_writer_that_fails_on_the_store_when_the_instance_ends_is_asked_again_after_the_cancel",
    "test_a_cancelling_orchestrator_without_a_scrubber_withholds_what_failed_in_its_record_and_log",
    "test_a_taking_over_orchestrator_whose_secrets_file_cannot_be_used_takes_no_lease",
    "test_an_orchestrator_whose_secrets_file_cannot_be_used_takes_no_lease",
    "test_create_checks_each_allocation_then_submits_the_chain",
    "test_dying_between_sbatch_and_recording_the_id_never_submits_twice",
    "test_a_lost_sbatch_answer_is_looked_up_by_its_tag",
    "test_reattaching_never_relaunches_a_running_allocation",
    "test_attach_waits_for_a_live_orchestrator",
    "test_a_stale_orchestrator_stops_without_acting",
    "test_a_preflight_refusal_voids_before_anything_is_submitted",
    "test_hls_cancel_reaches_the_orchestrator_holding_the_lease",
    "test_a_failed_allocation_is_retried_and_its_pending_dependent_rewired",
    "test_run_passes_until_the_instance_ends",
    "test_a_store_error_in_a_pass_is_tried_again_at_the_next_poll",
    "test_a_pass_that_keeps_failing_ends_the_orchestrator_after_a_bounded_number_of_tries",
    "test_a_lost_lease_ends_the_orchestrator_at_once",
    "test_attach_keeps_the_lease_alive_before_the_run",
    "test_a_dependent_of_a_done_predecessor_whose_job_failed_runs_without_the_dependency",
    "test_a_retry_that_finished_before_the_rewire_leaves_no_dependent_stuck",
    "test_a_dependent_of_a_completed_predecessor_is_left_to_slurm",
    "test_cancels_the_controller_does_not_answer_are_not_a_revision_each",
    "test_run_sleeps_only_after_a_pass_that_changed_nothing",
    "test_a_holder_taken_over_between_its_write_and_its_effects_never_submits",
    "test_each_effect_checks_the_lease_before_it_runs",
    "test_a_holder_whose_lease_lapsed_stops_before_it_looks_at_anything",
    "test_the_takeover_revision_is_written_before_anything_else",
    "test_a_job_that_publishes_and_ends_between_the_two_reads_does_not_void_the_instance",
    "test_a_runner_that_keeps_reporting_is_never_declared_dead",
    "test_the_runners_own_fault_is_the_failure_reason",
    "test_a_status_that_is_unreadable_or_another_jobs_is_ignored_with_an_event",
    "test_the_runners_status_names_a_job_whose_sbatch_answer_was_lost",
    "test_one_tag_lookup_a_pass_and_no_state_query_for_a_job_that_ended",
    "test_a_cancel_reaches_a_job_whose_sbatch_answer_was_lost_and_unfound",
    "test_three_dependency_refusals_in_a_row_void_the_instance",
    "test_the_orchestrator_reports_its_memory_every_so_many_passes",
    "test_a_write_that_conflicts_does_nothing_and_logs_nothing",
    "test_an_effect_that_raises_is_recorded_and_the_others_still_are",
    "test_a_submission_that_raises_before_sbatch_is_found_absent_and_tried_again",
    "test_create_releases_the_lease_when_the_instance_exists",
    "test_create_releases_the_lease_when_the_preflight_breaks",
    "test_attach_releases_the_lease_when_the_takeover_cannot_be_written",
    "test_a_dependent_slurm_already_released_is_not_cancelled_after_a_lost_rewire_answer",
    "test_a_dependent_of_a_job_hls_cancelled_is_released_when_the_site_gives_no_reason",
    "test_a_job_a_holder_submitted_after_its_takeover_is_cancelled_by_the_new_holder",
    "test_a_submission_that_always_fails_is_named_in_the_void_reason",
    "test_a_status_that_names_something_other_than_a_job_id_is_ignored",
    "test_a_void_instance_stays_finished_on_the_passes_after_its_jobs_ended",
    "test_a_lost_lease_is_never_an_effect_failure",
    "test_a_runner_killed_after_its_stage_declared_no_retry_is_not_relaunched",
    "test_a_runner_killed_before_any_declaration_is_relaunched",
    "test_a_declaration_is_read_only_for_the_attempt_of_a_stage_that_has_not_published",
    "test_runner_statuses_are_read_after_job_states",
    "test_job_states_are_read_before_manifests",
    "test_failed_passes_are_counted_in_a_row_only",
]
for _name in RUN:
    globals()[_name] = getattr(orchestrator_tests, _name)
console = orchestrator_tests.console


@pytest.fixture(autouse=True)
def _local_scheduler(tmp_path, monkeypatch):
    LocalFake.for_test(tmp_path)
    monkeypatch.setattr(orchestrator_tests, "FakeScheduler", LocalFake)
    yield
    LocalFake.end_all()
