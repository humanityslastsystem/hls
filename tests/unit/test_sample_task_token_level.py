"""The token-level fixture task (``tests.support.token_level_task``) wires a
TokenFaithfulnessVerifier verifier over a contract that returns token ids, and its contestant
is the live agent."""

import pytest

from hls.artifact.artifact import AgentProducedArtifact
from hls.core.capabilities import TeacherForce
from hls.core.decode import Greedy
from hls.reference.references import Reference, VLLMReference
from hls.spec.taskspec import TaskSpec
from hls.verify import TokenFaithfulnessVerifier, WiringError


def test_sample_token_level_task_wires_token_faithfulness():
    from tests.support import token_level_task as t

    assert len(t.TASK.verifiers) == 1
    v = t.TASK.verifiers[0]
    assert isinstance(v, TokenFaithfulnessVerifier)
    # TokenFaithfulnessVerifier owns its reference engine config and requires TeacherForce.
    assert isinstance(v.reference, Reference)
    assert TeacherForce in v.requires
    # Fails closed: hls run refuses this task until a matching calibration.json is
    # injected by the CLI loader.
    assert TokenFaithfulnessVerifier.needs_calibration is True


def test_sample_token_level_verifier_rejects_generate_only_reference():
    """A VLLMReference is Generate-only (no TeacherForce/StackReport); wiring it into
    this task's TokenFaithfulnessVerifier in place of the real oracle must fail
    check_wiring at TaskSpec construction, before any GPU is touched."""
    from tests.support import token_level_task as t

    kwargs = {name: getattr(t.TASK, name) for name in TaskSpec.model_fields}
    kwargs["verifiers"] = [TokenFaithfulnessVerifier(reference=VLLMReference(decode=Greedy()))]
    with pytest.raises(WiringError):
        TaskSpec(**kwargs)


def test_sample_token_level_task_contract_returns_token_ids():
    from tests.support import token_level_task as t

    # Token ids are always emitted now; this task also enables top-k logprobs
    # (with per-alternative token_id) for a future token-level verifier.
    assert t.CONTRACT.protocol.return_logprobs is True
    assert t.CONTRACT.protocol.top_logprobs == 20
    assert t.CONTRACT.transport == "http"
    # There is no Driver config: the Runner instantiates the Driver runtime
    # directly and late-binds its contract from TaskSpec.contract at deploy. The
    # arena's single source of truth is TASK.contract, the token-ids contract.
    assert t.TASK.contract is t.CONTRACT


def test_sample_token_level_default_contestant_is_agentic():
    from tests.support import token_level_task as t

    assert isinstance(t.ARTIFACT, AgentProducedArtifact)


def test_sample_token_level_fixture_declares_the_calibrated_gpu_kind():
    """The fixture declares the kind its calibration was measured on (B200): hls calibrate
    refuses to measure on other silicon, and read_calibration refuses a kind mismatch at
    run time."""
    from tests.support import token_level_task as t

    assert t.TASK.hardware.accelerator.kind == "B200"
