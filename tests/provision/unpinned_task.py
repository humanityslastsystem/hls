"""A task whose dataset names a branch, not a commit, which ``resolve`` refuses."""

from __future__ import annotations

from tests.provision.tasks import task

TASK = task(data_revision="main")
