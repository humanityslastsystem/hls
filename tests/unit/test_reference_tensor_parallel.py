"""A vLLM reference's tensor-parallel width reaches both the serving stack and the slot.

A reference engine that cannot shard is capped at whatever one card holds, so a checkpoint
larger than a card cannot be an oracle at all. The width is authored on the reference config
(not derived from the task's declared hardware: the reference is not the deployment being
measured) and has to arrive in two places at once: the stack, or the engine builds at width 1
and runs out of memory; and ``ReferenceEngine.resources``, or the role gets one card to shard
eight ways.

These tests pin both halves, the default that keeps every reference declared before the field
on exactly one card and the vLLM oracle built as it was, and the fingerprint the width rides into.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

from hls.core.decode import Greedy, Sample
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.poly import dump_spec
from hls.reference.engines import VLLMEngine
from hls.reference.references import VLLMReference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.taskspec import TaskSpec
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify import LetterMatchMetric, MetricVerifier

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL = ModelSpec(base_checkpoint="m", arch="TX")


def _task() -> TaskSpec:
    return TaskSpec(
        id="T",
        model=ModelSpec(base_checkpoint="x", arch="TX", chat_format="chatml"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="H200", count=8)),
        contract=RequestResponseContract(
            path="/v1/chat/completions",
            input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"text"})),
        workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(), slo=NoSLO()),
        verifiers=[MetricVerifier(metric=LetterMatchMetric())],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)),
    )


def _fake_vllm(monkeypatch, captured: dict) -> None:
    """A stand-in ``vllm`` module whose ``LLM`` records the kwargs it was built with."""
    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM))


def _weights(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HLS_WEIGHTS_DIR", str(tmp_path))


# --- the default: one card, built as before -----------------------------------------


def test_an_unauthored_width_is_one_card_and_absent_from_the_dump():
    """The default has to stay 1, and stay out of the reference's dump, so every reference
    declared before the field keeps its slot and its identity: those calibrations are
    measurements on one card."""
    ref = VLLMReference(decode=Greedy())
    assert ref.tensor_parallel_size == 1
    assert "tensor_parallel_size" not in dump_spec(ref)
    assert ref.build_engine(_task()).resources.gpu_count == 1


def test_a_greedy_oracle_on_one_card_is_built_as_it_was(monkeypatch, tmp_path):
    """No width, no worker extension, prefix caching left at vLLM's default: the oracle of
    every greedy task is the engine it was."""
    captured: dict = {}
    _fake_vllm(monkeypatch, captured)
    _weights(monkeypatch, tmp_path)
    VLLMReference(decode=Greedy()).build_engine(_task())._load()
    assert set(captured) == {"model", "dtype", "gpu_memory_utilization", "max_model_len", "enforce_eager"}


def test_a_width_below_one_is_refused():
    """A count of devices; zero or negative is a wiring bug, caught at construction."""
    with pytest.raises(ValueError, match="gpu_count must be at least 1"):
        VLLMEngine(model_spec=_MODEL, decode=Greedy(), tensor_parallel_size=0)
    with pytest.raises(ValueError):
        VLLMReference(decode=Greedy(), tensor_parallel_size=0)


# --- an authored width reaches the stack and the slot -------------------------------


def test_an_authored_width_reaches_the_vllm_oracle_and_its_slot(monkeypatch, tmp_path):
    """vLLM defaults ``tensor_parallel_size`` to 1, so an omitted kwarg pins the oracle to
    one card. The slot grows with it."""
    captured: dict = {}
    _fake_vllm(monkeypatch, captured)
    _weights(monkeypatch, tmp_path)
    engine = VLLMReference(decode=Greedy(), tensor_parallel_size=8).build_engine(_task())
    engine._load()
    assert captured["tensor_parallel_size"] == 8
    assert engine.resources.gpu_count == 8


def test_a_sampled_reference_is_built_to_score_reports(monkeypatch, tmp_path):
    """Under a sampled decode the oracle scores served reports: the capture's worker methods
    are mixed in by name, and prefix caching is off so every served position is recomputed
    and produces its row."""
    captured: dict = {}
    _fake_vllm(monkeypatch, captured)
    _weights(monkeypatch, tmp_path)
    VLLMReference(decode=Sample(temperature=1.0)).build_engine(_task())._load()
    assert captured["worker_extension_cls"] == "hls.reference.vllm_report.ReportWorker"
    assert captured["enable_prefix_caching"] is False
    assert "tensor_parallel_size" not in captured


def test_an_authored_width_reaches_the_vllm_serving_engine(monkeypatch, tmp_path):
    """The clocked-serving face builds a second engine from its own args object, so it needs
    the width threaded separately; a calibration pool member serves through this path."""
    captured: dict = {}

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.engine", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils",
                        types.SimpleNamespace(AsyncEngineArgs=FakeAsyncEngineArgs))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm",
                        types.SimpleNamespace(AsyncLLM=types.SimpleNamespace(
                            from_engine_args=lambda args: object())))
    _weights(monkeypatch, tmp_path)
    engine = VLLMEngine(model_spec=_MODEL, decode=Sample(temperature=1.0),
                        tensor_parallel_size=8, clocked_serving=True)
    engine._serve_build()
    assert captured["tensor_parallel_size"] == 8
    assert engine.resources.gpu_count == 8


# --- the faces a decode allows ------------------------------------------------------


def test_a_greedy_engine_scores_no_report():
    """A greedy reference is graded token by token; scoring a report under it would compare
    the served distribution against a decode no server sampled from."""
    import asyncio
    engine = VLLMEngine(model_spec=_MODEL, decode=Greedy())
    with pytest.raises(NotImplementedError, match="sampled decode"):
        asyncio.run(engine.reference_logprobs([]))


@pytest.mark.parametrize("temperature", [0.7, 2.0])
def test_an_engine_sampling_away_from_temperature_one_scores_no_report(temperature):
    """The worker reduces the raw logits, the distribution at temperature one only
    (test_report_reduction_agrees), so a report served at any other temperature would be
    measured against the wrong distribution, and one that ignored its temperature would pass."""
    import asyncio
    engine = VLLMEngine(model_spec=_MODEL, decode=Sample(temperature=temperature))
    with pytest.raises(NotImplementedError, match=f"temperature 1 only; this decode samples at {temperature}"):
        asyncio.run(engine.reference_logprobs([]))


def test_a_sampled_engine_generates_nothing():
    import asyncio
    engine = VLLMEngine(model_spec=_MODEL, decode=Sample(temperature=1.0))
    with pytest.raises(NotImplementedError):
        asyncio.run(engine.generate([]))


# --- the width is part of the instrument's identity ---------------------------------


def test_the_width_changes_the_reference_fingerprint():
    """A ceiling measured with the oracle sharded eight ways is a claim about that oracle, so
    the width rides ``dump_spec`` into the fingerprint."""
    task = _task()
    one = VLLMReference(decode=Greedy())
    eight = VLLMReference(decode=Greedy(), tensor_parallel_size=8)
    assert dump_spec(eight)["tensor_parallel_size"] == 8
    assert one.fingerprint(task) != eight.fingerprint(task)


def test_the_width_changes_the_calibrated_workload_fingerprint():
    """``workload_payload`` carries each calibrated verifier's reference config, so a
    calibration taken at one width is refused as stale by a task declaring another."""
    from hls.spec.wire import OpenAIChatProtocol
    from hls.verify import ReportFaithfulnessVerifier
    from hls.verify.calibration import workload_fingerprint
    from hls.verify.report import TOP_LOGPROBS_REQUESTED

    def task_at(width: int) -> TaskSpec:
        ref = VLLMReference(decode=Sample(temperature=1.0), tensor_parallel_size=width)
        return _task().model_copy(update={
            "verifiers": [ReportFaithfulnessVerifier(reference=ref)],
            "contract": RequestResponseContract(
                path="/v1/chat/completions",
                input_modalities=frozenset({"text"}),
                output_modalities=frozenset({"text"}),
                protocol=OpenAIChatProtocol(
                    return_logprobs=True, top_logprobs=TOP_LOGPROBS_REQUESTED)),
        })

    assert workload_fingerprint(task_at(1)) != workload_fingerprint(task_at(8))


# --- every task in the tree, including ones added later -----------------------------


def _in_tree_task_modules() -> list[str]:
    return sorted(str(p.relative_to(_REPO_ROOT).with_suffix("")).replace("/", ".")
                  for p in (_REPO_ROOT / "tasks").glob("*/task*.py"))


@pytest.mark.parametrize("module_path", _in_tree_task_modules())
def test_a_task_reference_asks_for_exactly_the_width_it_authored(module_path):
    """A reference gets the cards it asked for and no others; a task that authors no width
    gets one card, which is what every committed calibration was measured on."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    task = getattr(importlib.import_module(module_path), "TASK", None)
    if task is None:
        pytest.skip(f"{module_path} declares no TASK")
    refs = [v.reference for t in task.effective_tenants() for v in t.verifiers
            if isinstance(getattr(v, "reference", None), VLLMReference)]
    if not refs:
        pytest.skip(f"{module_path} has no vLLM verifier reference")
    for ref in refs:
        authored = ref.tensor_parallel_size if "tensor_parallel_size" in ref.model_fields_set else 1
        assert ref.build_engine(task).resources.gpu_count == authored, (module_path, ref)
