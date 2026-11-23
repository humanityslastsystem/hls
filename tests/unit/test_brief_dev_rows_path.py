"""The development rows the brief names are rows the agent actually has mounted.

The brief prints the path the provision stage's mounts put the rows at
(``hls.provision.mounts.data_mounts``), so the two cannot state different directories. These
tests render each shipped task's brief and check every rows file it names against the mounts
that task's agent role would be given.
"""

from __future__ import annotations

import importlib
import re

import pytest

from hls.provision.model import Provisioned, RowsRef
from hls.provision.mounts import data_mounts
from tests.unit.brief_task import render_brief

# The text tasks and the token-level fixture, by module. The Data bullet reads differently
# for a single-tenant task than for a multi-tenant one, so both shapes are covered (asserted
# below).
_TASKS = ["tasks.sample_task.task", "tests.support.token_level_task", "tasks.multitenant_slo.task",
          "tasks.tab_completion.task"]


def _task(name: str):
    """Import a task module and return its ``(taskspec, brief)``."""
    mod = importlib.import_module(name)
    return mod.TASK, mod.ARTIFACT.brief


def _agent_mounts(taskspec, tmp_path) -> set[str]:
    """The container paths the agent role mounts rows at for *taskspec*: a rows entry holding
    every tenant with a data source, as the provision stage publishes it. The weights mount is
    left out; the brief names no file in it."""
    tenants = [t.id for t in taskspec.effective_tenants() if t.data_source is not None]
    provisioned = Provisioned(weights=None, rows=RowsRef(key="rows-" + "0" * 16, entry_sha256="0" * 64, disk_bytes=0,
                                                         tenants=tenants), tools=[])
    declared = {f"rows/{provisioned.rows.key}": provisioned.rows.disk_bytes}
    mounts, _env = data_mounts("agent", provisioned, lambda kind, key, sha: tmp_path / kind / key,
                               declared=declared, weights=False)
    return {str(m.target) for m in mounts}


@pytest.mark.parametrize("name", _TASKS)
def test_the_brief_names_rows_the_agent_has_mounted(name, tmp_path):
    """Every rows file the brief names sits in a directory the agent role mounts."""
    taskspec, brief = _task(name)
    text, _prov = render_brief(taskspec, brief)
    named = set(re.findall(r"/data/\S*?rows\.json", text))
    assert named, f"{name}: the brief names no rows file"
    mounted = _agent_mounts(taskspec, tmp_path)
    ids = [t.id for t in taskspec.effective_tenants()]
    for path in named:
        # A multi-tenant brief states the family once with the id left open, so it
        # holds only if it holds for every tenant.
        concrete = [path.replace("<tenant>", tid) for tid in ids] if "<tenant>" in path else [path]
        for p in concrete:
            assert p.rsplit("/", 1)[0] in mounted, (
                f"{name}: brief names {p}, agent mounts {sorted(mounted)}")


def test_both_tenant_shapes_are_covered():
    """The bullet takes one form for a single-tenant task and another for a multi-tenant
    one; the cases above are only a check on both forms if both are in the list."""
    shapes = {len(_task(name)[0].effective_tenants()) > 1 for name in _TASKS}
    assert shapes == {False, True}
