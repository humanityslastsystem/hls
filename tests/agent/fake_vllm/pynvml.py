"""A stand-in for the NVML bindings, as far as the in-build reference service reads them: one
device of 140 GiB, whatever handle is asked for, with ``gpu_free_gib`` of ``FAKE_VLLM_PLAN``'s
file free (all of it by default)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

_GIB = 2**30
_TOTAL_GIB = 140


def _free_gib() -> float:
    path = os.environ.get("FAKE_VLLM_PLAN")
    plan = json.loads(Path(path).read_text()) if path and Path(path).exists() else {}
    return float(plan.get("gpu_free_gib", _TOTAL_GIB))


def nvmlInit() -> None:  # noqa: N802 -- NVML's naming
    pass


def nvmlDeviceGetCount() -> int:  # noqa: N802
    return 1


def nvmlDeviceGetHandleByIndex(index: int) -> int:  # noqa: N802
    return index


def nvmlDeviceGetHandleByUUID(uuid: str) -> str:  # noqa: N802
    return uuid


def nvmlDeviceGetMemoryInfo(handle) -> SimpleNamespace:  # noqa: N802
    return SimpleNamespace(free=int(_free_gib() * _GIB), total=_TOTAL_GIB * _GIB)
