"""A task module whose ``TaskSpec`` does not validate: importing it raises, and ``resolve``
refuses it as declared."""

from __future__ import annotations

from hls.spec.taskspec import TaskSpec

TASK = TaskSpec.model_validate({"id": "T-invalid"})
