"""A reference engine loads its weights only from the provisioned snapshot of the pinned commit."""

from __future__ import annotations

import pytest

from hls.core.decode import Greedy
from hls.core.model import ModelSpec
from hls.reference.engines import HFEngine

SHA = "0123456789abcdef0123456789abcdef01234567"


def _engine() -> HFEngine:
    return HFEngine(model_spec=ModelSpec(base_checkpoint="org/m", revision=SHA, arch="TX"), decode=Greedy())


def test_the_mounted_snapshot_is_the_source(monkeypatch, tmp_path):
    monkeypatch.setenv("HLS_WEIGHTS_DIR", str(tmp_path))
    assert _engine()._weights_source() == str(tmp_path)


@pytest.mark.parametrize("mounted", [None, "missing"], ids=["unset", "not-a-directory"])
def test_without_a_mounted_snapshot_the_engine_refuses_rather_than_reading_the_hub(monkeypatch, tmp_path, mounted):
    """The checkpoint name alone would read whatever the hub's default branch names, not the
    commit the task pins."""
    if mounted is None:
        monkeypatch.delenv("HLS_WEIGHTS_DIR", raising=False)
    else:
        monkeypatch.setenv("HLS_WEIGHTS_DIR", str(tmp_path / mounted))
    with pytest.raises(RuntimeError, match=r"no weights mounted at \$HLS_WEIGHTS_DIR"):
        _engine()._weights_source()
