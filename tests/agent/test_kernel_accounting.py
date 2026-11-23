"""An agent build on a site that sets no ``site.nproc_accounting`` counts its commands as its node's
kernel release says, states that in its brief and its probe, and refuses a kernel whose own count
the release does not tell, naming the value to set."""

from __future__ import annotations

import pytest

import hls.pipeline.stage as stage_module
from hls.pipeline.stage import StageFault
from hls.stages.agent_build import AgentBuildStage
from tests.agent.probe_answers import answering
from tests.agent.test_stage import (NOT_ROOT, _ctx, _events, _params, _script, closed_contexts,  # noqa: F401
                                    sandbox_bins)


@NOT_ROOT
@pytest.mark.parametrize("release, accounting, seen", [("4.18.0-553.el8_10.x86_64", "uid", "forks=0"),
                                                       ("5.14.0-427.el9.x86_64", "user_namespace", "forks=511")])
def test_a_build_on_a_site_that_sets_no_accounting_probes_what_its_kernel_release_says(
        sandbox_bins, tmp_path, monkeypatch, release, accounting, seen):  # noqa: F811
    monkeypatch.setattr(stage_module, "kernel_release", lambda: release)
    answering({"nproc-accounting": f"echo {seen}"}, monkeypatch)
    _script(sandbox_bins, window=1_000_000, claude=[])
    ctx = _ctx(tmp_path, _params(stop_after="probe"))
    assert ctx.plan.settings.site.nproc_accounting is None
    AgentBuildStage().run(ctx)
    [chosen] = [ev for ev in _events(ctx) if ev["kind"] == "nproc_accounting"]
    assert (chosen["accounting"], chosen["source"], chosen["kernel"]) == (accounting, "kernel release", release)
    [probe] = [ev for ev in _events(ctx) if ev["kind"] == "build.seal_probe"]
    assert probe["accounting"] == accounting


@NOT_ROOT
def test_a_kernel_that_counts_otherwise_than_its_release_says_names_the_value_to_set(
        sandbox_bins, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(stage_module, "kernel_release", lambda: "4.18.0-553.el8_10.x86_64")
    answering({"nproc-accounting": "echo forks=511"}, monkeypatch)
    _script(sandbox_bins, window=1_000_000, preflight={}, claude=[{"commit": "v1\n"}])
    ctx = _ctx(tmp_path, _params())
    with pytest.raises(StageFault) as e:
        AgentBuildStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason.startswith("this node's kernel release 4.18.0-553.el8_10.x86_64 says its kernel "
                                           "counts the commands' processes as 'uid', but the nproc-accounting check "
                                           "read 'user_namespace'")
    assert e.value.fault.reason.endswith('set site.nproc_accounting = "user_namespace"')
