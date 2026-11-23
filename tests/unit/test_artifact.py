"""The contestant Artifact is pure config (a Poly): no produce(), no config().
``dump_spec(artifact)`` serializes it whole into provenance (kind + knobs, no
secrets). The deployable image is built by the agent_build and package stages."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hls.core.poly import dump_spec
from hls.artifact.artifact import (Artifact, AgentProducedArtifact,
                                   AgentProducerSpec, Budget, HighestAvailable)
from hls.egress.denied_packages import DENIED_PACKAGES
from hls.prompt import ImplementerBrief


def test_the_producer_names_no_harness_of_its_own():
    """The arm names the harness and the model (``hls.arms``); the producer holds neither the
    harness nor a model's context length and prices, and a task that still names either is
    refused when the module loads."""
    assert not {"harness", "model_info"} & set(AgentProducerSpec.model_fields)
    for field, value in (("harness", "codex"), ("model_info", {"max_input_tokens": 1})):
        with pytest.raises(ValidationError, match=field):
            AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600),
                              **{field: value})


def test_the_producer_names_no_model_and_no_submission_mode():
    """The arm names the model, and the package stage builds a git repository and nothing else,
    so the producer holds neither, and a task that still sets either is refused when the module
    loads."""
    assert not {"model_name", "artifact_mode"} & set(AgentProducerSpec.model_fields)
    for field, value in (("model_name", "claude-x"), ("artifact_mode", "git_repo")):
        with pytest.raises(ValidationError, match=field):
            AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600),
                              **{field: value})


def test_a_budget_bounds_wall_clock_and_turns_and_no_cost():
    """No runner keeps a cost bound so a budget has none, and a task that sets one is refused when the module loads."""
    assert set(Budget.model_fields) == {"wallclock_sec", "turns"}
    with pytest.raises(ValidationError, match="cost_usd"):
        Budget(wallclock_sec=3600, cost_usd=60.0)


def test_an_agent_produced_artifacts_brief_is_an_implementer_brief():
    """The agent_build stage renders the brief with its provenance and every section, which
    only an implementer brief takes; a task that gives another template is refused when the
    module loads."""
    from hls.prompt import JinjaPromptTemplate

    producer = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600))
    assert isinstance(AgentProducedArtifact(producer=producer).brief, ImplementerBrief)
    with pytest.raises(ValidationError):
        AgentProducedArtifact(producer=producer, brief=JinjaPromptTemplate(template="{{ x }}"))


def test_artifact_is_pure_config_and_dumps_with_kind():
    a = AgentProducedArtifact(
        producer=AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600, turns=5),
                                   server_rel_path="srv.py"))
    d = dump_spec(a)
    assert d["kind"] == "AgentProducedArtifact"
    assert d["producer"]["server_rel_path"] == "srv.py"
    assert d["producer"]["hard_budget"]["turns"] == 5
    # Secrets are never modeled on the Poly, so they can never leak into provenance.
    assert "agent_secrets" not in str(d)


def test_agent_producer_spec_defaults():
    s = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600))
    assert s.hard_budget.turns is None
    assert s.hard_budget.wallclock_sec == 3600
    assert s.soft_budget is None
    assert s.server_rel_path == "server.py"
    # Build egress is SEALED by default: raw internet air-gapped (empty allow-list), and
    # the package index refuses the harness deny list.
    assert s.network is not None
    assert s.network.domains.mode == "allow" and s.network.domains.entries == []
    assert s.network.packages.mode == "deny"
    assert s.network.packages.entries == list(DENIED_PACKAGES)


def test_budget_dimensions_default_none():
    b = Budget()
    assert (b.wallclock_sec, b.turns) == (None, None)


def test_agent_producer_spec_hard_budget_serializes_nested():
    s = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600, turns=40))
    d = s.model_dump()
    assert d["hard_budget"]["wallclock_sec"] == 3600
    assert d["hard_budget"]["turns"] == 40
    assert d["soft_budget"] is None


def test_hard_budget_requires_wallclock():
    import pytest
    with pytest.raises(ValueError):
        AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(turns=40))  # no wallclock_sec


def test_hard_budget_requires_positive_wallclock():
    import pytest
    # A build must always be bounded by a strictly-positive wall clock: 0 and negative are
    # rejected.
    for bad in (0, -10):
        with pytest.raises(ValueError):
            AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=bad))


def test_soft_ge_hard_warns(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600, turns=40),
                          soft_budget=Budget(turns=40))  # soft turns == hard turns
    assert any("budget" in r.message.lower() for r in caplog.records)


def test_task_folder_defaults():
    s = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600))
    assert s.upload_dir is None
    assert s.upload_brief is None
    assert s.skills_dir is None


def test_upload_dir_requires_upload_brief():
    import pytest
    # The uploaded-files brief section has NO default: a task shipping an upload folder
    # must explain the files -- None and blank both fail.
    with pytest.raises(ValueError, match="upload_brief is required"):
        AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600),
                          upload_dir=("docs",))
    with pytest.raises(ValueError, match="upload_brief is required"):
        AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600),
                          upload_dir=("docs",), upload_brief="   ")


def test_upload_brief_without_upload_dir_rejected():
    import pytest
    with pytest.raises(ValueError, match="upload_dir is not"):
        AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600),
                          upload_brief="CUDA docs for the kernels")


def test_upload_dir_with_brief_accepted_and_serializes():
    s = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=3600),
                          upload_dir=("docs",), upload_sha256="a" * 64, upload_brief="CUDA PTX ISA reference.",
                          skills_dir="skills")
    d = s.model_dump()
    assert (d["upload_dir"], d["upload_sha256"]) == (("docs",), "a" * 64)
    assert d["upload_brief"] == "CUDA PTX ISA reference."
    assert d["skills_dir"] == "skills"


def test_artifact_has_no_produce_or_config():
    # An Artifact is config only: it has no runtime methods.
    assert not hasattr(Artifact, "produce")
    assert not hasattr(Artifact, "config")


def test_an_evaluation_allocation_names_the_published_memory_its_server_gets_from():
    """The server's memory is derived in one place from the figure the task declares:
    a job the server shared with the harness, less the runner's reserve, so a site's larger reserve
    comes out of the server and never adds to the job; or a step of the server's own, whole."""
    from hls.artifact.artifact import EvaluationAllocation

    shared = EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180, driver_memory_gib=16)
    assert (shared.server_memory_gib(4), shared.server_memory_gib(6)) == (176, 174)
    own = EvaluationAllocation(cpus=8, step_memory_gib=180, writable_gib=180, driver_memory_gib=64, driver_cpus=8)
    assert own.server_memory_gib(4) == own.server_memory_gib(6) == 180
    for both_or_neither in ({"job_memory_gib": 180, "step_memory_gib": 180}, {}):
        with pytest.raises(ValueError, match="exactly one of job_memory_gib and step_memory_gib"):
            EvaluationAllocation(cpus=8, writable_gib=180, driver_memory_gib=16, **both_or_neither)
    with pytest.raises(ValueError, match="leaves the server no memory"):
        shared.server_memory_gib(180)


def test_only_a_driver_in_the_servers_job_and_on_its_cpus_shares_the_servers_memory():
    from hls.artifact.artifact import EvaluationAllocation
    from hls.core.poly import dump_spec

    shared = EvaluationAllocation(cpus=64, job_memory_gib=1911, writable_gib=3814, driver_memory_gib=768,
                                  driver_memory_shared=True)
    assert shared.server_memory_gib(4) == 1907
    for apart in ({"driver_cpus": 8, "job_memory_gib": 180}, {"step_memory_gib": 180}):
        with pytest.raises(ValueError, match="only a driver on the server's CPUs in the server's job"):
            EvaluationAllocation(cpus=8, writable_gib=180, driver_memory_gib=16, driver_memory_shared=True, **apart)
    # Left out of the dump when off, so no other allocation's record moves.
    plain = EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180, driver_memory_gib=16)
    assert "driver_memory_shared" not in dump_spec(plain)
