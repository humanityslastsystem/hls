"""``FakeScheduler``'s test controls over a real ``LocalScheduler``, so the orchestrator tests written
against the fake (``tests.pipeline.fake_scheduler``) run against the local scheduler.

Each job runs a batch script that waits for a control file and exits with the status written in
it, in place of the orchestrator's batch script, which would start a real runner. ``start`` waits
until the job runs, ``finish`` ends it as asked, and ``cancel`` waits until the job has ended, as
the fake's cancel ends a job at once. The faults the fake scripts (``down``, ``submit_fault``,
``tags_down``, ``test_only_refusal``, ``pending_reason``) are kept: ``down`` makes the state
directory unusable, which is the local scheduler's own way of not answering; the others stand in
for a scheduler's answer as the fake's do. ``jobs`` is a view of the real records; a test that
removes a job from it or purges one asks for what the local scheduler never does and does not
run here."""

from __future__ import annotations

import os
import signal
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from hls.substrate.base import (JobInfo, JobState, Refused, Scheduler, SchedulerUnavailable, SubmitRequest,
                                SubmitUnknown)
from hls.substrate.local.records import StateDir
from hls.substrate.local.scheduler import LocalScheduler
from tests.pipeline.fake_scheduler import FakeJob
from tests.substrate.local.helpers import FakeInventory, Request, wait_for


def local_request(request: SubmitRequest, script: Path) -> Request:
    """*request* as the local scheduler reads it, running *script*."""
    return Request(script=script, shape=request.shape, job_name=request.job_name, output_dir=request.output_dir,
                   tag=request.tag, after=request.after, singleton=request.singleton)


@dataclass
class _Shared:
    """What the schedulers of one test share: the host's state directory and what was submitted."""

    root: Path
    requests: dict[str, SubmitRequest]
    scheduler: LocalScheduler
    #: When each job started on the test's clock, as ``start`` was told.
    started_at: dict[str, float]


class LocalFake(Scheduler):
    """A real local scheduler under ``FakeScheduler``'s controls. ``shared`` is set per test."""

    shared: _Shared | None = None

    def __init__(self, *, submit_fault: str | None = None, test_only_refusal: str | None = None,
                 refusal: str = "Batch job submission failed: Invalid account", after_create=None,
                 down: bool = False, tags_down: bool = False, pending_reason: str | None = None) -> None:
        assert self.shared is not None
        self.real = self.shared.scheduler
        self.submit_fault = submit_fault
        self.test_only_refusal = test_only_refusal
        self.refusal = refusal
        self.after_create = after_create
        self.tags_down = tags_down
        self.pending_reason = pending_reason
        self.calls: list[tuple] = []
        self._down = False
        self.down = down

    @classmethod
    def for_test(cls, tmp_path: Path) -> _Shared:
        root = tmp_path / "host-state"
        state = StateDir(root)
        state.check()
        (root / "next_id").write_text("100\n")
        (root / "ctl").mkdir()
        real = LocalScheduler(state_dir=root, inventory=FakeInventory(gpus=8, numa=2, cores_per_node=32,
                                                                      memory_gib=2048),
                              kept_memory_gib=0, python=Path(sys.executable), stopper="none", stopper_options={},
                              kill_wait_sec=0.5,
                              poll_sec=0.02)
        script = root / "ctl" / "job.sh"
        script.write_text(f'#!/bin/bash\nctl={root / "ctl"}\n'
                          'while [ ! -e "$ctl/$HLS_JOB_ID" ]; do sleep 0.02; done\n'
                          'exit "$(cat "$ctl/$HLS_JOB_ID")"\n')
        cls.shared = _Shared(root=root, requests={}, scheduler=real, started_at={})
        return cls.shared

    @classmethod
    def end_all(cls) -> None:
        """End every job still there, so no supervisor outlives its test."""
        shared = cls.shared
        if shared is None:
            return
        os.chmod(shared.root, 0o700)
        for job_id in shared.requests:
            shared.scheduler.cancel(job_id)
        for job_id in shared.requests:
            wait_for(lambda: shared.scheduler.states([job_id])[job_id].state.terminal, 30, f"job {job_id} ending")
        cls.shared = None

    # FakeScheduler's controls

    @property
    def down(self) -> bool:
        return self._down

    @down.setter
    def down(self, value: bool) -> None:
        self._down = value
        os.chmod(self.shared.root, 0o750 if value else 0o700)

    @property
    def jobs(self) -> dict[str, FakeJob]:
        was = self.down
        self.down = False
        try:
            out = {}
            infos = self.real.states(list(self.shared.requests))
            for job_id, request in self.shared.requests.items():
                record = self.real.state.read(job_id)
                info = infos[job_id]
                out[job_id] = FakeJob(job_id, request, record.after, state=info.state, reason=info.reason,
                                      started_at=self.shared.started_at.get(job_id, info.started_at))
            return out
        finally:
            self.down = was

    def start(self, job_id: str, at: float = 0.0) -> None:
        """Wait until the job runs, and report its start as *at*, on the test's clock."""
        wait_for(lambda: self.real.state.read(job_id).runner is not None, 30, f"job {job_id} starting")
        self.shared.started_at[job_id] = at

    def finish(self, job_id: str, state: JobState = JobState.COMPLETED) -> None:
        if state is JobState.COMPLETED or state is JobState.FAILED:
            (self.shared.root / "ctl" / job_id).write_text("0" if state is JobState.COMPLETED else "1")
        elif state is JobState.CANCELLED:
            self.real.cancel(job_id)
        elif state is JobState.NODE_FAIL:
            os.kill(self.real.state.read(job_id).supervisor.pid, signal.SIGKILL)
        else:
            pytest.skip(f"a local job cannot be made to end {state.value}")
        wait_for(lambda: self.real.states([job_id])[job_id].state is state, 30, f"job {job_id} ending {state}")

    def submits(self) -> list[tuple]:
        return [c for c in self.calls if c[0] == "submit"]

    def cancels(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "cancel"]

    # Scheduler

    def submit_argv(self, request: SubmitRequest) -> tuple[str, ...]:
        return self.real.submit_argv(local_request(request, self.shared.root / "ctl" / "job.sh"))

    def submit(self, request: SubmitRequest) -> str:
        self.calls.append(("submit", request.tag, request.after))
        fault, self.submit_fault = self.submit_fault, None
        if fault == "refuse":
            raise Refused(self.refusal)
        job_id = self.real.submit(local_request(request, self.shared.root / "ctl" / "job.sh"))
        self.shared.requests[job_id] = request
        if self.after_create is not None:
            self.after_create(FakeJob(job_id, request, request.after))
        if fault == "lose_answer":
            raise SubmitUnknown("sbatch timed out")
        return job_id

    def test_only(self, request: SubmitRequest) -> str:
        self.calls.append(("test_only", request.tag))
        if self.test_only_refusal:
            raise Refused(self.test_only_refusal)
        if self.down:
            raise SchedulerUnavailable("the state directory cannot be used")
        return self.real.test_only(local_request(request, self.shared.root / "ctl" / "job.sh"))

    def find_by_tags(self, tags: Sequence[str]) -> dict[str, list[str] | None]:
        self.calls.append(("find_by_tags", tuple(tags)))
        if self.tags_down:
            return {t: None for t in dict.fromkeys(tags)}
        return self.real.find_by_tags(tags)

    def states(self, job_ids: Sequence[str]) -> dict[str, JobInfo]:
        self.calls.append(("states", tuple(job_ids)))
        out = self.real.states(job_ids)
        out = {j: JobInfo(j, i.state, reason=i.reason, exit_code=i.exit_code, started_at=self.shared.started_at[j])
               if j in self.shared.started_at and i.started_at is not None else i for j, i in out.items()}
        if self.pending_reason is not None:
            out = {j: JobInfo(j, i.state, reason=self.pending_reason, exit_code=i.exit_code,
                              started_at=i.started_at) if i.state is JobState.PENDING else i for j, i in out.items()}
        return out

    def rewire(self, job_id: str, after: str) -> None:
        self.calls.append(("rewire", job_id, after))
        self.real.rewire(job_id, after)

    def cancel(self, job_id: str) -> None:
        self.calls.append(("cancel", job_id))
        self.real.cancel(job_id)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.real.states([job_id])[job_id].state.terminal:
                return
            time.sleep(0.02)
