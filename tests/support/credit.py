"""A task's baseline credit, written in a test the way the credit's fold stage writes it
(``hls.baseline_credit.baseline_credit_bytes``)."""

from __future__ import annotations

from pathlib import Path

from hls.baseline_credit import BASELINE_CREDIT_FILE, BaselineCredit, baseline_credit_bytes


def write_credit(directory: Path, credit: BaselineCredit) -> Path:
    """Write *credit* to ``directory/baseline.json`` and return the path."""
    path = Path(directory) / BASELINE_CREDIT_FILE
    path.write_bytes(baseline_credit_bytes(credit))
    return path
