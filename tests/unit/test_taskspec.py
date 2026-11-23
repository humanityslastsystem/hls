"""Tests for hls.spec.taskspec.

TaskSpec has no `reference` field: References live inside Verifiers as their owned
engine configs. TaskSpec.__post_init__ validates wiring via ``check_wiring``: a
verifier's ``requires`` capabilities must be provided by exactly one of its
declared reference engines. Token ids are always emitted on the wire contract,
so a TeacherForce requirement has no extra contract prerequisite. These tests
check that TaskSpec rejects mis-wired verifiers.
"""

import pytest
from pydantic import ValidationError

from hls.core.decode import Greedy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.prompt import JinjaPromptTemplate
from hls.reference.references import HFReference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.verify import (LetterMatchMetric, MetricVerifier,
                        ReferenceRelativeVerifier, TokenFaithfulnessVerifier)
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.agent import AgentFramework, AgentSpec
from hls.spec.taskspec import CompatibilityError, TaskSpec
from hls.spec.tools import LocalCallableToolRunner, ToolSpec
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, Uncapped, Workload)


def _dataset():
    return HFDataset(path="cais/mmlu", subset="abstract_algebra", revision="main", source_split="test",
                     prompt=JinjaPromptTemplate(template="{{ question }}"),
                     target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))


def _data_source():
    return DataSource(dataset=_dataset(), split=SplitSpec(dev=Count(value=5), test=Count(value=10)))


def _contract():
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        protocol=OpenAIChatProtocol())


def _oracle():
    return HFReference(decode=Greedy())


def _base_kwargs():
    contract = _contract()
    model = ModelSpec(base_checkpoint="m", arch="TX")
    return dict(
        id="T", model=model,
        hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
        contract=contract,
        workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(), slo=NoSLO()),
        data_source=_data_source(),
        verifiers=[MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))


def test_taskspec_constructs_with_verifier_list():
    ts = TaskSpec(**_base_kwargs())
    assert len(ts.verifiers) == 1


def test_taskspec_has_no_reference_field():
    """TaskSpec.reference was deleted -- References live inside Verifiers."""
    assert "reference" not in TaskSpec.model_fields


def test_a_task_declares_no_images():
    """The images a task builds in and is served from are the image catalog's
    (``hls.recipes.catalog``), so a task module that passes recipes is refused."""
    assert "images" not in TaskSpec.model_fields
    with pytest.raises(ValidationError, match="images"):
        TaskSpec(**_base_kwargs(), images=object())


def test_taskspec_rejects_bad_accelerator_count():
    """The construction-time guard still rejects an accelerator count < 1."""
    kwargs = _base_kwargs()
    kwargs["hardware"] = Hardware(accelerator=AcceleratorSpec(kind="B200", count=0))
    with pytest.raises(CompatibilityError):
        TaskSpec(**kwargs)


def test_dataset_verifier_needs_no_engine_and_constructs():
    """A dataset-based verifier (MetricVerifier) declares no required
    capabilities, so check_wiring is satisfied with no reference engine at all."""
    ts = TaskSpec(**_base_kwargs())
    assert ts.verifiers[0].requires == frozenset()
    assert ts.verifiers[0].engines() == []


def test_reference_relative_accuracy_with_generate_engine_is_valid():
    """ReferenceRelativeVerifier requires Generate; its HFReference engine provides
    Generate -> check_wiring passes."""
    kwargs = _base_kwargs()
    kwargs["verifiers"] = [ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=_oracle(), tol=0.0)]
    ts = TaskSpec(**kwargs)   # must not raise
    assert len(ts.verifiers) == 1


def test_token_faithfulness_is_valid():
    """TokenFaithfulnessVerifier requires TeacherForce + StackReport, both provided by
    HFReference; the wire contract always emits token ids, so no extra contract
    prerequisite applies."""
    kwargs = _base_kwargs()
    kwargs["verifiers"] = [TokenFaithfulnessVerifier(reference=_oracle())]
    ts = TaskSpec(**kwargs)   # must not raise
    assert len(ts.verifiers) == 1


def test_oracle_uses_task_model_and_constructs():
    """The oracle's model is always task.model (HFReference carries no
    model_spec field), so a verifier owning an HFReference constructs without any
    model identity of its own."""
    kwargs = _base_kwargs()
    oracle = HFReference(decode=Greedy())
    kwargs["verifiers"] = [ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=oracle, tol=0.0)]
    ts = TaskSpec(**kwargs)   # must not raise
    assert len(ts.verifiers) == 1


def test_taskspec_has_seed_default_zero():
    ts = TaskSpec(**_base_kwargs())
    assert ts.seed == 0


# ---- the framework's veto on the task's model ------------------------------

class _AnyModelFramework(AgentFramework):
    """A framework with the default ``check_model``, which accepts any model."""

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx,
                             max_turns):   # pragma: no cover - never driven here
        raise NotImplementedError


class _PickyFramework(_AnyModelFramework):
    """A framework that refuses every model, so the refusal itself is under test."""

    def check_model(self, model):
        raise ValueError(f"cannot drive {model.base_checkpoint!r}")


def _agent(framework):
    """The agentic seam a task declares, around ``framework``."""
    runner = LocalCallableToolRunner(
        specs=[ToolSpec(name="lookup", description="d", json_schema={})],
        callables={"lookup": lambda q: {"text": q}})
    return AgentSpec(framework=framework, tool_runner=runner, max_turns=2)


def test_the_default_check_model_accepts_any_model():
    ts = TaskSpec(**_base_kwargs(), agent=_agent(_AnyModelFramework()))
    assert ts.agent.framework.check_model(ts.model) is None


def test_a_framework_can_refuse_the_task_model_at_construction():
    """A framework parses the served model's replies, so it gets a veto on the pair.
    Without this the mismatch surfaces only as a trajectory that parses nothing."""
    with pytest.raises(ValidationError, match="cannot drive 'm'"):
        TaskSpec(**_base_kwargs(), agent=_agent(_PickyFramework()))


def test_model_copy_re_runs_the_construction_checks():
    """Task variants are authored as copies with one field swapped, and pydantic's
    own ``model_copy`` skips validation, so a variant could otherwise carry a pair
    a directly constructed TaskSpec refuses."""
    ts = TaskSpec(**_base_kwargs(), agent=_agent(_AnyModelFramework()))
    assert ts.model_copy(update={"seed": 3}).seed == 3
    with pytest.raises(ValidationError, match="cannot drive 'm'"):
        ts.model_copy(update={"agent": _agent(_PickyFramework())})
