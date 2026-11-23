"""A task stand-in a brief renders for, and the one way the tests render a brief: with every
argument the agent_build stage passes, each at the value the stage gives it unless a test says
otherwise."""

from __future__ import annotations

from typing import Any

from hls.pipeline.pipelines import package_shape
from hls.prompt import BriefProvenance, ImplementerBrief, packaging_brief_section
from hls.settings import RunSettings
from hls.spec.contracts import RequestResponseContract
from hls.spec.submission import SubmissionContract
from hls.spec.tenant import Tenant
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.stages.agent_build import commit_point_sec
from hls.stages.package import pip_limits

#: The writable disk cap of the build the brief is rendered for, in GiB, as most tasks have it.
BUILD_DISK_CAP_GIB = 180
#: What each pip step of a package of that build's bundle gets, in the shape the pipelines declare
#: for it, as the build's brief states it, under the default runner reserve.
PIP_LIMITS = pip_limits(package_shape(BUILD_DISK_CAP_GIB), RunSettings().runner_memory_gib)
#: Those limits, and the run's default collection allowance with the commit point's least part of
#: it, as the packaging section takes them.
PIP = {"pip_disk_gib": PIP_LIMITS.disk_gib, "pip_memory_gib": PIP_LIMITS.memory_gib,
       "collect_sec": RunSettings().collect_allowance_sec,
       "bundle_sec": commit_point_sec(RunSettings().collect_allowance_sec)}


class _FakeModel:
    fingerprint = "meta-llama/Llama-3.1-8B-Instruct@TX"
    base_checkpoint = "meta-llama/Llama-3.1-8B-Instruct"
    arch = "TX"
    chat_format = "llama3-instruct"

    def prompt_fields(self):
        return {"fingerprint": self.fingerprint, "base_checkpoint": self.base_checkpoint,
                "arch": self.arch, "chat_format": self.chat_format}


class _FakeHardware:
    def prompt_fields(self):
        return {"accelerator_kind": "B200", "accelerator_count": 1,
                "memory_gb_per_device": 0, "interconnect": None,
                "cpu_count": 0, "memory_gb": 0, "storage_gb": 0}


class _FakeTaskSpec:
    id = "example-task"
    model = _FakeModel()
    hardware = _FakeHardware()
    network = None
    agent = None
    contract = RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
    )
    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(),
                        slo=NoSLO())

    def effective_tenants(self):
        return [Tenant(id="t0", workload=self.workload, verifiers=[])]


#: A single-tenant task with a text contract, the least a brief renders for.
TASK = _FakeTaskSpec()


def render_brief(task: Any, brief: ImplementerBrief | None = None,
                 **overrides: Any) -> tuple[str, BriefProvenance]:
    """*brief* (the default brief when None) rendered for *task*, with its text and provenance, as
    the agent_build stage renders it: the server at ``/workspace/server.py``, the git repository's
    submission section, the packaging section of a sandbox whose base holds no version, no budget,
    no uploads, nothing blocked and no reference service. *overrides* replace any of these."""
    arguments: dict[str, Any] = {
        "server_path": "/workspace/server.py",
        "submission_section": SubmissionContract(server_filename="server.py").brief_section(),
        "packaging_section": packaging_brief_section(held=None, **PIP),
        "budget_section": "",
        "uploads_section": "",
        "blocked_packages": [],
        "reference_service": False,
    }
    brief = brief if brief is not None else ImplementerBrief()
    return brief.render_with_provenance(taskspec=task, **{**arguments, **overrides})
