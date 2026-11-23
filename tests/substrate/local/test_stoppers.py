"""The stoppers a job may name, and the options each takes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hls.substrate.local.scheduler import LocalScheduler
from hls.substrate.local.stoppers import NoStop, check_stopper, stopper
from tests.substrate.local.helpers import FakeInventory, request


def test_each_stopper_takes_exactly_its_options():
    assert isinstance(stopper("none", {}), NoStop)
    check_stopper("docker", {"socket": "/run/docker.sock"})
    for name, options in (("none", {"socket": "/x"}), ("docker", {}), ("docker", {"socket": "/x", "y": "1"}),
                          ("podman", {})):
        with pytest.raises(ValueError):
            check_stopper(name, options)


def test_the_scheduler_refuses_a_stopper_without_its_options_and_passes_them_to_submit(tmp_path):
    kw = dict(state_dir=tmp_path / "state", inventory=FakeInventory(), kept_memory_gib=0, python=Path(sys.executable))
    with pytest.raises(ValueError):
        LocalScheduler(**kw, stopper="docker", stopper_options={})
    sched = LocalScheduler(**kw, stopper="docker", stopper_options={"socket": "/run/docker.sock"})
    assert "--stopper-option=socket=/run/docker.sock" in sched.submit_argv(request(tmp_path))
