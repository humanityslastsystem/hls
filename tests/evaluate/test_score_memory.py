"""The score role reads the drive's summary a line at a time and keeps of each reply's packet
timestamps the first, the last and their count, so what it holds whole is one reply's line, every
packet timestamp of it: its memory follows from how many timestamps one reply of
``REPLY_MAX_BYTES`` can buy on the task's stream. A role sized for less would run out of memory
on a trace the driver accepted, and the attempt would fail as infra, retried until the instance
voids. These tests hold the sizing (``hls.stages.score.score_memory_gib``) to the published
tasks' contracts, to ``REPLY_MAX_BYTES`` and to what the role measurably holds.

The figures are the role's peaks (``python -m hls.score.role``, Python 3.13) parsing the
summary whole, under a cgroup limit with no swap, over traces at the densest reply each task's
decoder accepts:
- one 240 MB reply of 120,000,000 token ids, each its own packet timestamp: 8.83 GiB, more than
  an 8 GiB role holds. That trace's summary is that one reply's line, so a reader that holds one
  line at a time holds no more for it;
- a summary of ten small replies, the role's own floor: 0.08 GiB."""

from __future__ import annotations

import asyncio
import base64
import importlib
import io
import json
import wave

import httpx
import pytest

from hls.artifact.artifact_client import HTTPArtifactClient, packet_wire_bytes
from hls.core.faults import ArtifactProtocolError
from hls.core.payloads import Envelope, TextPayload
from hls.drive.bounds import REPLY_MAX_BYTES
from hls.pipeline.pipelines import artifact_shape, driver_shape
from hls.spec.resources import concurrent
from hls.spec.wire import TTSStreamProtocol, _OpenAIProtocol
from hls.stages.score import score_memory_gib

GIB = 2**30
#: The published tasks, by the module each is graded with.
PUBLISHED = ("tasks.multitenant_slo.task", "tasks.tab_completion.task", "tasks.deepresearch.task",
             "tasks.multimodal_asr.task", "tasks.multimodal_omni.task", "tasks.multimodal_tts.task")
#: The most the role measurably holds per packet timestamp: one reply holding all of them, parsed
#: whole beside the rest (8.83 GiB for 120,000,000).
MEASURED_PER_PACKET = 8.83 * GIB / 120_000_000
#: The role with a summary of a few small replies.
MEASURED_FLOOR = 0.08 * GIB


def _wav(n_bytes: int = 44) -> str:
    """A mono PCM16 WAV of no samples, its first *n_bytes* bytes, in base64: 44 is the header
    alone, the least ``wave`` reads."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24_000)
        w.writeframes(b"")
    return base64.b64encode(buf.getvalue()[:n_bytes]).decode()


def _frame_line(audio: str) -> str:
    """One TTS stream frame as a server writes it at its shortest: the client reads a stream
    line by line, so a frame is ``data:``, its JSON and the newline that ends the line."""
    return "data:" + json.dumps({"audio": audio}, separators=(",", ":")) + "\n"


#: The fewest wire bytes of one TTS audio frame the client reads and the decoder accepts.
TTS_FRAME_BYTES = len(_frame_line(_wav()).encode())


def _task(module: str):
    return importlib.import_module(module).TASK


def _packet_bytes(contract) -> int | None:
    """The fewest reply bytes one packet timestamp costs under *contract*, from its own transport
    and protocol: None for a unary reply, which is timed whole."""
    if contract.transport == "http":
        return None
    if isinstance(contract.protocol, TTSStreamProtocol):
        return TTS_FRAME_BYTES
    assert isinstance(contract.protocol, _OpenAIProtocol) and contract.protocol.emits_token_ids
    return len("0,")


@pytest.mark.parametrize("module", PUBLISHED)
def test_each_published_tasks_packet_cost_is_the_one_its_contract_allows(module):
    task = _task(module)
    assert packet_wire_bytes(task.contract) == _packet_bytes(task.contract)


@pytest.mark.parametrize("module", PUBLISHED)
def test_the_score_role_holds_the_worst_reply_line_its_tasks_stream_allows(module):
    task = _task(module)
    per_packet = _packet_bytes(task.contract)
    packets = 0 if per_packet is None else REPLY_MAX_BYTES // per_packet
    assert score_memory_gib(task.contract) * GIB >= MEASURED_FLOOR + packets * MEASURED_PER_PACKET


#: The score role's memory each published task gets, in GiB, from the rule: 8, and for a stream a
#: quarter more than 80 bytes for each packet timestamp one reply of 268,435,456 bytes can buy,
#: rounded up. A token id's packet takes 2 bytes: 134,217,728 packets, 12.5 GiB, so 8 + 13. A TTS
#: frame takes 78: 3,441,480 packets, 0.32 GiB, so 8 + 1. A unary reply has no packets: 8.
PUBLISHED_ROLE_GIB = {"tasks.multitenant_slo.task": 21, "tasks.tab_completion.task": 21,
                      "tasks.deepresearch.task": 21, "tasks.multimodal_asr.task": 8,
                      "tasks.multimodal_omni.task": 8, "tasks.multimodal_tts.task": 9}


@pytest.mark.parametrize("module", PUBLISHED)
def test_each_published_tasks_score_role_is_sized_by_one_reply_at_its_cap(module):
    contract = _task(module).contract
    per_packet = _packet_bytes(contract)
    packets = 0 if per_packet is None else REPLY_MAX_BYTES // per_packet
    assert score_memory_gib(contract) == 8 + -(-125 * 80 * packets // (100 * GIB)) == PUBLISHED_ROLE_GIB[module]


def _served(body: str):
    """multimodal-tts's own client and contract, answered with the stream *body*."""
    client = HTTPArtifactClient(contract=_task("tasks.multimodal_tts.task").contract, base_url="http://localhost:9000")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})))
    return asyncio.run(client.submit(Envelope(request_id="r0", contents=[TextPayload(text="Hello.")],
                                              metadata={"max_tokens": 64})))


def test_the_tts_packet_cost_is_the_smallest_frame_its_client_times():
    # Five frames of a header-only WAV, one line each, are five packets of 78 bytes.
    line = _frame_line(_wav())
    response = _served(line * 5)
    assert len(response.packet_timestamps) == 5
    contract = _task("tasks.multimodal_tts.task").contract
    assert len(line.encode()) == TTS_FRAME_BYTES == packet_wire_bytes(contract) == 78
    # A header one byte shorter is no WAV the decoder reads, so no frame is shorter.
    with pytest.raises(ArtifactProtocolError):
        _served(_frame_line(_wav(43)) * 5)


def test_the_measured_reply_fits_the_role_one_reply_at_its_cap_gets():
    mslo, tts, asr = "tasks.multitenant_slo.task", "tasks.multimodal_tts.task", "tasks.multimodal_asr.task"
    # A reply of 120,000,000 token ids takes the role to 8.83 GiB; one at REPLY_MAX_BYTES holds
    # 134,217,728, so at that cost per timestamp 9.88 GiB, past an 8 GiB role.
    ids = REPLY_MAX_BYTES // 2
    assert score_memory_gib(_task(mslo).contract) * GIB > ids * MEASURED_PER_PACKET > 8.83 * GIB > 8 * GIB
    assert score_memory_gib(_task(tts).contract) * GIB > MEASURED_FLOOR + REPLY_MAX_BYTES // 78 * MEASURED_PER_PACKET
    # A unary task's summary holds nothing per byte of its replies: the role's base.
    assert score_memory_gib(_task(asr).contract) == 8


@pytest.mark.parametrize("module", PUBLISHED)
def test_the_score_stage_fits_in_the_evaluation_allocation_the_drive_stage_already_holds(module):
    """Stages run in turn in one allocation of the largest stage's memory, which is the drive's,
    so the score stage's growth does not grow the allocation."""
    mod = importlib.import_module(module)
    task = mod.TASK
    allocation = mod.ARTIFACT.producer.evaluation_allocation
    runner = 4
    drive = concurrent([artifact_shape(task, allocation, runner_memory_gib=runner),
                        driver_shape(task, allocation)]).memory_gib + runner
    assert score_memory_gib(task.contract) + runner <= drive


def test_a_contract_without_a_packet_cost_is_refused():
    stream = _task("tasks.multimodal_omni.task").contract.model_copy(update={"transport": "http-sse"})
    with pytest.raises(ValueError, match="no packet cost for a OmniChatProtocol stream"):
        score_memory_gib(stream)
