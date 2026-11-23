"""Tests for how calibration treats a task whose servers answer with audio.

``hls calibrate`` refuses such a task: no role can drive a TTS serving stack in process.
What is tested here is that the shape is read off the contract, that the refusal holds
whichever gate the task declares, and that the pinned reference of a text task is the HF
instrument. ``tests.calibration.fixture.audio_task`` builds its speaking task from ``_task``.
Everything runs on the torch-free host; nothing deploys.
"""

from __future__ import annotations

import pytest

from hls.calibrate import audio_replies, check_campaign, pinned_reference
from hls.core.decode import Greedy, Sample
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.prompt import JinjaPromptTemplate
from hls.reference.references import HFReference
from hls.reference.tts import TEMPERATURE, Qwen3TTSReference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.wire import OpenAIChatProtocol, TTSStreamProtocol
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify import TokenFaithfulnessVerifier
from hls.verify.calibration import CalibrationError
from hls.verify.report import TOP_LOGPROBS_REQUESTED
from hls.verify.tts_faithfulness import CodecFidelityVerifier, TtsReportFaithfulnessVerifier


def _dataset() -> HFDataset:
    return HFDataset(path="mythicinfinity/libritts", subset="clean", revision="main",
                     source_split="dev.clean",
                     prompt=JinjaPromptTemplate(template="{{ text }}"),
                     target=JinjaPromptTemplate(template="{{ text }}"))


def _tts_ref(**kwargs) -> Qwen3TTSReference:
    """The oracle config as the task declares it. The decode has to be the sampled one:
    the report gate refuses a greedy reference when the task is wired."""
    kwargs.setdefault("decode", Sample(temperature=TEMPERATURE))
    return Qwen3TTSReference(**kwargs)


def _task(*, audio: bool = True, verifiers=None) -> TaskSpec:
    """A calibrated single-tenant TaskSpec, speaking or writing."""
    if audio:
        contract = RequestResponseContract(
            path="/v1/audio/speech", input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"audio"}), transport="http-sse",
            protocol=TTSStreamProtocol(
                return_codec_codes=True, return_logprobs=True,
                top_logprobs=TOP_LOGPROBS_REQUESTED))
        default = [TtsReportFaithfulnessVerifier(reference=_tts_ref())]
    else:
        contract = RequestResponseContract(
            path="/v1/chat/completions", input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"text"}), protocol=OpenAIChatProtocol())
        default = [TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()))]
    return TaskSpec(
        id="T", model=ModelSpec(base_checkpoint="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                                arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="H200", count=1)),
        contract=contract,
        workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(), slo=NoSLO()),
        data_source=DataSource(dataset=_dataset(),
                               split=SplitSpec(dev=Count(value=2), test=Count(value=4))),
        verifiers=default if verifiers is None else verifiers,
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))


def test_the_output_modality_decides_the_shape():
    assert audio_replies(_task(audio=True)) is True
    assert audio_replies(_task(audio=False)) is False


@pytest.mark.parametrize("gate", ["sampling", "token-level"])
def test_a_speaking_task_is_refused_before_anything_is_planned(gate):
    """No role can drive a TTS pool member: a serving is a clocked drive of an
    asynchronous per-request API, which the packaged TTS stacks do not publish. So
    ``hls calibrate`` refuses a speaking task whichever gate it declares."""
    task = _task() if gate == "sampling" else _task(verifiers=[CodecFidelityVerifier(reference=_tts_ref())])
    with pytest.raises(CalibrationError, match="answers with audio"):
        check_campaign(task, engines=["vllm"], servings_per_engine=None)


def test_the_text_instrument_is_the_math_pinned_hf_engine():
    pinned = pinned_reference(_task(audio=False))
    assert type(pinned).__name__ == "HFEngine"
    assert pinned.sdpa_backend == "math"
