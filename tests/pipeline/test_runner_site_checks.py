"""The runner refuses a node that does not lay out cgroups as every role step needs, or that could
not run the network seal of a stage that seals, before any stage runs, naming what it found and
what fixes it."""

from __future__ import annotations

from types import SimpleNamespace

from hls.stages.agent_build import AgentBuildStage
from hls.stages.calibration import CaptureStage, WarmStage
from hls.stages.deploy_drive import DeployDriveStage
from hls.stages.noop import NoopStage
from hls.stages.package import PackageStage
from hls.substrate.runner import EXIT_FAILED, EXIT_PUBLISHED
from hls.substrate.site_checks import SiteProblem
from tests.pipeline.plans import noop_plan
from tests.pipeline.test_runner import _no_attempts, _runner, _status, _store, _sub


def test_the_stages_that_start_sealed_containers_say_so():
    for cls in (AgentBuildStage, DeployDriveStage, PackageStage):
        assert cls.seals_containers(None), cls.kind
    assert not NoopStage.seals_containers(None)


def test_a_calibration_member_is_sealed_unless_its_campaign_gives_it_the_internet():
    for cls in (WarmStage, CaptureStage):
        for internet in (False, True):
            params = cls.params_model.model_construct(campaign=SimpleNamespace(member_internet=internet))
            assert cls.seals_containers(params) is not internet, (cls.kind, internet)


def test_a_node_whose_cgroups_are_not_slurms_v2_layout_is_refused_before_the_grant(tmp_path):
    plan = _store(noop_plan(tmp_path), _sub(noop_plan(tmp_path)))
    measured = []

    def measure():
        measured.append(True)
        raise AssertionError("the grant is not measured on a node the cgroup check refused")

    problem = SiteProblem("/proc/self/cgroup has no cgroup v2 entry ('0::'), only cgroup v1 entries such as "
                          "'12:memory:/slurm'. hls reads each job's ... CgroupPlugin=cgroup/v2 ...")
    runner = _runner(plan, measure=measure, check_cgroups=lambda: problem)
    assert runner.run() == EXIT_FAILED
    fault = _status(plan).fault
    assert (fault.fault_class, fault.retryable, fault.reason) == ("infra", True, problem.reason)
    assert fault.evidence == {"check": "cgroups", "node": "g1"}
    assert not measured and _no_attempts(plan, "noop_a")


def test_the_seal_is_checked_only_for_an_allocation_with_a_stage_that_seals(tmp_path, monkeypatch):
    plan = _store(noop_plan(tmp_path), _sub(noop_plan(tmp_path)))
    asked = []

    def problems():
        asked.append(True)
        return [SiteProblem("site.pasta_path /x does not exist on this node", retryable=False)]

    assert _runner(plan, check_seal=problems).run() == EXIT_PUBLISHED
    assert not asked


def test_a_node_that_could_not_run_the_seal_is_refused_naming_each_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(NoopStage, "seals_containers", classmethod(lambda cls, params: True))
    plan = _store(noop_plan(tmp_path), _sub(noop_plan(tmp_path)))
    found = [SiteProblem("site.pasta_path /x does not exist on this node", retryable=False),
             SiteProblem("the network seal runs iptables, ip6tables, sysctl from /usr/sbin:/sbin, and ip6tables is "
                         "not there on this node")]
    assert _runner(plan, check_seal=lambda: found).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert fault.reason == ("the network seal of stage noop_a, noop_b cannot run on this node: "
                            + "; ".join(p.reason for p in found))
    # A wrong site.pasta_path is the site's setting, which another node repeats.
    assert (fault.fault_class, fault.retryable, fault.evidence["check"]) == ("infra", False, "seal")
    assert _no_attempts(plan, "noop_a")


def test_a_seal_problem_of_the_node_alone_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(NoopStage, "seals_containers", classmethod(lambda cls, params: True))
    plan = _store(noop_plan(tmp_path), _sub(noop_plan(tmp_path)))
    found = [SiteProblem("this node refuses unprivileged user namespaces (...)")]
    assert _runner(plan, check_seal=lambda: found).run() == EXIT_FAILED
    assert _status(plan).fault.retryable
