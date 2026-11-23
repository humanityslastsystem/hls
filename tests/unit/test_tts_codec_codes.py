"""The codec codes and report a TTS reply carries, and what happens when it does not.

A sampling gate grades the distribution the server says it drew each code from, so both
the codes and that report have to reach the verifier intact: one frame after the last
audio frame, rows of a pinned width, ids inside the codebook, a report entry per
position and one more for the end-of-speech decision. Two properties are what these
tests hold to. Bad codes or a bad report are the contestant's answer being wrong, so
they are recorded on the reply and graded, never raised as a harness fault that would
abort a drive over one reply. And the codes frame is not product, so it is not charged
as latency: the response reports the last audio frame as its completion time and the
driver stamps the event from that.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from hls.core.audio import encode_wav_b64
from hls.core.faults import ArtifactProtocolError
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse
from hls.spec.wire import TTSStreamProtocol
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC

GROUPS = 16


def _tone(n: int = 160, *, sample_rate: int = 24_000) -> str:
    """A short well-formed base64 PCM16 WAV frame."""
    return encode_wav_b64(np.zeros(n, dtype=np.float32), sample_rate)


def _rows(n: int, *, width: int = GROUPS) -> list[list[int]]:
    """``n`` plausible codec frames of ``width`` ids each."""
    return [[(i * width + j) % 2048 for j in range(width)] for i in range(n)]


def _proto(**kw) -> TTSStreamProtocol:
    """The protocol as the TTS task pins it: codes on, per-frame ids off."""
    return TTSStreamProtocol(return_token_ids=False, return_codec_codes=True, **kw)


def _stream(*frames: dict) -> list[tuple[float, dict]]:
    """Time-stamp frames 0.1 s apart, as the transport does."""
    return [(0.1 * (i + 1), f) for i, f in enumerate(frames)]


def _payload(resp: ArtifactResponse):
    return resp.output.contents[0]


# ---- the conforming case ---------------------------------------------------

def test_codes_reach_the_payload_and_the_audio_is_unchanged():
    resp = _proto().decode_stream(
        _stream({"audio": _tone()}, {"audio": _tone()}, {"codec_codes": _rows(3)}), "r1")

    p = _payload(resp)
    assert p.codec_codes == _rows(3)
    assert p.codec_codes_defect is None
    assert p.sample_rate == 24_000
    assert p.duration_sec == pytest.approx(320 / 24_000)


def test_the_codes_frame_is_not_a_packet_and_does_not_extend_the_clock():
    """The gate's evidence must not be charged to the server as latency. The codes
    frame carries no audio, so it is not a packet, and the response reports the last
    audio frame as the moment the utterance was complete."""
    resp = _proto().decode_stream(
        _stream({"audio": _tone()}, {"audio": _tone()}, {"codec_codes": _rows(2)}), "r1")

    assert resp.t_first_packet == pytest.approx(0.1)
    assert resp.packet_timestamps == [pytest.approx(0.1), pytest.approx(0.2)]
    assert resp.t_complete == pytest.approx(0.2)   # the second audio frame, not the third


def test_the_driver_stamps_t_recv_from_the_completion_time():
    """The other half of the same property: the protocol's completion time is what
    lands on the event, so a request's measured job completion time stops at its last
    audio frame rather than when the stream closed."""
    from hls.spec.drivers import Driver

    class _Client:
        async def submit(self, env, *, attempt=None):
            return ArtifactResponse(
                output=Envelope(request_id=env.request_id,
                                contents=[TextPayload(text="A")]),
                t_first_packet=1.0, packet_timestamps=[1.0, 2.0], t_complete=2.0)

    env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
    sent = asyncio.run(Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)._submit_or_fault(_Client(), env))
    assert sent.t_recv == 2.0


def test_without_the_completion_time_the_driver_stamps_its_own_clock():
    """Every other protocol is unaffected: its stream ends with its last product
    frame, so the driver's own stamp stays the honest one."""
    from hls.spec.drivers import Driver, harness_now

    class _Client:
        async def submit(self, env, *, attempt=None):
            return ArtifactResponse(output=Envelope(
                request_id=env.request_id, contents=[TextPayload(text="A")]))

    before = harness_now()
    env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
    sent = asyncio.run(Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)._submit_or_fault(_Client(), env))
    assert sent.t_recv >= before


# ---- absent, duplicated, late ----------------------------------------------

def test_a_stream_with_no_codes_frame_is_recorded_not_faulted():
    resp = _proto().decode_stream(_stream({"audio": _tone()}), "r1")

    p = _payload(resp)
    assert p.codec_codes is None
    assert "no codec_codes" in p.codec_codes_defect
    assert p.duration_sec == pytest.approx(160 / 24_000)   # the audio still arrived


def test_two_codes_frames_lose_both():
    """Nothing here can say which of the two the server means, and picking one would
    invent a rule the contract does not have."""
    resp = _proto().decode_stream(
        _stream({"audio": _tone()}, {"codec_codes": _rows(2)}, {"codec_codes": _rows(3)}),
        "r1")

    p = _payload(resp)
    assert p.codec_codes is None
    assert "more than one" in p.codec_codes_defect


def test_audio_after_the_codes_loses_them():
    """The codes are supposed to cover the whole utterance; audio after them means
    they cover only part of it."""
    resp = _proto().decode_stream(
        _stream({"audio": _tone()}, {"codec_codes": _rows(2)}, {"audio": _tone()}), "r1")

    p = _payload(resp)
    assert p.codec_codes is None
    assert "followed" in p.codec_codes_defect


def test_codes_riding_on_the_last_audio_frame_are_accepted():
    """The contract asks for the codes after all the audio, which a server that
    appends them to its final frame satisfies. Nothing about the grade or the clock
    changes, so there is nothing to reject."""
    resp = _proto().decode_stream(
        _stream({"audio": _tone()}, {"audio": _tone(), "codec_codes": _rows(2)}), "r1")

    p = _payload(resp)
    assert p.codec_codes == _rows(2)
    assert p.codec_codes_defect is None


# ---- malformed codes -------------------------------------------------------

@pytest.mark.parametrize("label,value", [
    ("not a list", {"codec_codes": "1,2,3"}),
    ("no frames", {"codec_codes": []}),
    ("a flat list of ids", {"codec_codes": [1, 2, 3]}),
    ("a short row", {"codec_codes": [_rows(1)[0][:-1]]}),
    ("a long row", {"codec_codes": [_rows(1)[0] + [7]]}),
    ("a code past the codebook", {"codec_codes": [[2048] * GROUPS]}),
    ("a negative code", {"codec_codes": [[-1] * GROUPS]}),
    ("a float code", {"codec_codes": [[1.5] * GROUPS]}),
    ("a string code", {"codec_codes": [["7"] * GROUPS]}),
    ("a null code", {"codec_codes": [[None] * GROUPS]}),
    ("a bool code", {"codec_codes": [[True] * GROUPS]}),
    ("a good row then a bad one", {"codec_codes": _rows(2) + [[0] * (GROUPS - 1)]}),
])
def test_malformed_codes_fail_the_reply_not_the_drive(label, value):
    """Each of these is the contestant's own output, so each is graded, and each says
    in the reply what was wrong with it. A bool is its own case: ``bool`` subclasses
    ``int``, so a JSON ``true`` would otherwise be stored as the id 1 and teacher-forced
    as a code the server chose."""
    resp = _proto().decode_stream(_stream({"audio": _tone()}, value), "r1")

    p = _payload(resp)
    assert p.codec_codes is None, label
    assert p.codec_codes_defect and "codec_codes" in p.codec_codes_defect, label


def test_the_first_defect_is_the_one_reported():
    """A malformed frame followed by a well-formed one stays failed: the reply already
    violated the contract, and the later frame cannot repair it."""
    resp = _proto().decode_stream(
        _stream({"audio": _tone()}, {"codec_codes": [[0]]}, {"codec_codes": _rows(2)}),
        "r1")

    p = _payload(resp)
    assert p.codec_codes is None
    assert "1 ids, expected 16" in p.codec_codes_defect


def test_bad_audio_is_still_a_protocol_fault_when_the_codes_are_fine():
    """Codes are graded; audio is decoded. Making the codes a recorded defect must not
    have turned an unreadable waveform into one too."""
    with pytest.raises(ArtifactProtocolError):
        _proto().decode_stream(
            _stream({"audio": "!!!not base64!!!"}, {"codec_codes": _rows(2)}), "r1")


def test_a_row_width_the_task_pins_is_the_one_enforced():
    """The width is the codec's, not the harness's, so a task that pins another one
    grades against that."""
    proto = _proto(codec_groups=4, codec_codebook_size=64)
    ok = proto.decode_stream(_stream({"audio": _tone()}, {"codec_codes": [[1, 2, 3, 4]]}), "r")
    bad = proto.decode_stream(_stream({"audio": _tone()}, {"codec_codes": [[1, 2, 3, 64]]}), "r")

    assert _payload(ok).codec_codes == [[1, 2, 3, 4]]
    assert "outside [0, 64)" in _payload(bad).codec_codes_defect


# ---- the flag is what turns any of this on ---------------------------------

def test_a_protocol_that_asks_for_no_codes_reads_none_and_records_none():
    """Nothing grades codes on such a task, so a stream that happens to carry them is
    not read, their absence is not a defect, and the latency clock is untouched."""
    proto = TTSStreamProtocol(return_token_ids=False)
    assert proto.emits_codec_codes is False

    resp = proto.decode_stream(_stream({"audio": _tone()}, {"codec_codes": _rows(2)}), "r1")

    p = _payload(resp)
    assert p.codec_codes is None
    assert p.codec_codes_defect is None
    assert resp.t_complete is None


def test_the_flag_is_on_the_wire_and_in_the_brief():
    """A server is asked for the codes in the request body and told the frame's shape
    in the brief; a task that does not want them says neither."""
    env = Envelope(request_id="r", contents=[TextPayload(text="hello")])

    on = _proto()
    assert on.encode_request(env, stream=True)["return_codec_codes"] is True
    brief = on.brief_section(path="/v1/audio/speech", streaming=True)
    assert "codec_codes" in brief and "16 integer ids" in brief and "[0, 2048)" in brief

    off = TTSStreamProtocol(return_token_ids=False)
    assert "return_codec_codes" not in off.encode_request(env, stream=True)
    assert "codec_codes" not in off.brief_section(path="/v1/audio/speech", streaming=True)


def test_the_example_codes_frame_is_one_the_decoder_accepts():
    """The brief's example is built from the pinned shape, so it cannot drift from what
    the parser will take."""
    proto = _proto()
    example = proto.body_schema()["codes_frame"]
    resp = proto.decode_stream(_stream({"audio": _tone()}, example), "r1")

    assert _payload(resp).codec_codes_defect is None


# ---- the unary path --------------------------------------------------------

def test_the_unary_reply_carries_codes_the_same_way():
    proto = _proto()
    env = proto.decode_response({"audio": _tone(), "sample_rate": 24_000,
                                 "codec_codes": _rows(2)}, "r1")
    assert env.contents[0].codec_codes == _rows(2)

    env = proto.decode_response({"audio": _tone(), "sample_rate": 24_000}, "r1")
    assert env.contents[0].codec_codes is None
    assert "no codec_codes" in env.contents[0].codec_codes_defect


# ---- wiring ----------------------------------------------------------------

def _audio_contract(protocol):
    from hls.spec.contracts import RequestResponseContract

    return RequestResponseContract(
        path="/v1/audio/speech", transport="http-sse",
        input_modalities=frozenset({"text"}), output_modalities=frozenset({"audio"}),
        protocol=protocol)


def _codes_verifier(*, teacher_force: bool):
    """A gate over the reply's codes, with and without teacher forcing: one grades the
    codes against what the checkpoint would have chosen, the other re-decodes them and
    compares waveforms."""
    from hls.core.capabilities import TeacherForce
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference
    from hls.verify import Verifier

    class _CodesGate(Verifier):
        reference: HFReference
        requires = frozenset({TeacherForce}) if teacher_force else frozenset()
        accepts = frozenset({"audio"})
        requires_codec_codes = True

        async def verify(self, engine):
            ...

    return _CodesGate(reference=HFReference(decode=Greedy()))


@pytest.mark.parametrize("teacher_force", [True, False])
def test_check_wiring_refuses_a_codes_gate_over_a_protocol_that_asks_for_none(teacher_force):
    """Caught at authoring. Left to run, every reply would arrive with no codes and
    fail closed, reporting a contestant defect that is really a mis-wired task. The
    teacher-forcing gate is refused for the same reason and must not be refused for
    the wrong one: the units it forces are codes, so its prerequisite is the codes
    marker and not the text protocol's token ids."""
    from hls.verify import WiringError, check_wiring

    contract = _audio_contract(TTSStreamProtocol(return_token_ids=False))
    with pytest.raises(WiringError, match="(?i)codec codes"):
        check_wiring([_codes_verifier(teacher_force=teacher_force)], contract)


@pytest.mark.parametrize("teacher_force", [True, False])
def test_check_wiring_accepts_a_codes_gate_over_a_protocol_that_asks_for_them(teacher_force):
    from hls.verify import check_wiring

    check_wiring([_codes_verifier(teacher_force=teacher_force)], _audio_contract(_proto()))


# ---- the report the sampling gate grades -----------------------------------

TOP = 21


def _reporting_proto(**kw) -> TTSStreamProtocol:
    """The protocol as the sampling gate needs it: codes, and a report on every one."""
    return _proto(return_logprobs=True, top_logprobs=TOP, **kw)


def _entry(served: int, *, n_alts: int = TOP) -> list:
    """One reported position: the served code's log-probability, then alternatives.

    The served code is among them, which is the ordinary case; the gate's own rules
    cover the position where it is not."""
    alts = [[served, -0.5]] + [[2000 + i, -2.0 - i] for i in range(n_alts - 1)]
    return [-0.5, alts]


EOS = 2150      # the checkpoint's talker_config.codec_eos_token_id


def _codes_frame(codes: list[list[int]] | None = None, *, eos: int = EOS, **over) -> dict:
    """A conforming codes frame with its report, before ``over`` breaks one key."""
    codes = _rows(2) if codes is None else codes
    frame = {"codec_codes": codes,
             "codec_report": [[_entry(c) for c in row] for row in codes],
             "eos_code": eos,
             "eos_report": _entry(eos)}
    frame.update(over)
    return frame


def test_the_report_lands_flat_in_the_documented_order():
    """The one order a consumer may assume: frames outermost, groups within them, the
    end-of-speech decision last. The gate walks this list against the codes flattened
    the same way, so a transposition here is a silently wrong verdict rather than an
    error."""
    codes = _rows(3)
    resp = _reporting_proto().decode_stream(
        _stream({"audio": _tone()}, _codes_frame(codes)), "r1")

    p = _payload(resp)
    assert p.codec_codes == codes
    assert p.codec_codes_defect is None
    assert len(p.codec_logprobs) == 3 * GROUPS + 1
    assert [e.token_id for e in p.codec_logprobs] == [c for row in codes for c in row] + [EOS]
    assert len(p.codec_logprobs[0].top_logprobs) == TOP
    assert p.codec_logprobs[0].top_logprobs[0].token_id == codes[0][0]


def test_the_end_of_speech_id_is_carried_by_its_own_entry():
    """It has no row in the codes, so its entry's token_id is the only record of what
    the server served there, and the gate cross-checks that against the checkpoint's
    own end-of-speech id."""
    resp = _reporting_proto().decode_stream(
        _stream({"audio": _tone()}, _codes_frame(eos=4096)), "r1")

    assert _payload(resp).codec_logprobs[-1].token_id == 4096


def test_a_bad_report_costs_the_reply_its_report_and_not_its_codes():
    """A gate that re-decodes the codes and compares waveforms reads no report at all.
    Withholding the codes over a bad report would fail that gate on another gate's
    evidence, so the two fail independently."""
    codes = _rows(2)
    resp = _reporting_proto().decode_stream(
        _stream({"audio": _tone()}, _codes_frame(codes, codec_report="not a list")), "r1")

    p = _payload(resp)
    assert p.codec_codes == codes
    assert p.codec_logprobs is None
    assert "codec_report" in p.codec_codes_defect


def test_bad_codes_cost_the_reply_its_report_too():
    """The other direction does not hold: the report's positions are named by the
    codes, so without them there is no reading it."""
    resp = _reporting_proto().decode_stream(
        _stream({"audio": _tone()}, _codes_frame(codec_codes=[[1, 2]])), "r1")

    p = _payload(resp)
    assert p.codec_codes is None
    assert p.codec_logprobs is None


@pytest.mark.parametrize("label,over", [
    ("absent", {"codec_report": None}),
    ("not a list", {"codec_report": {"0": []}}),
    ("one frame short", {"codec_report": [[_entry(0)] * GROUPS]}),
    ("a narrow frame", {"codec_report": [[_entry(0)] * 3, [_entry(0)] * GROUPS]}),
    ("a frame that is not a list", {"codec_report": ["nope", [_entry(0)] * GROUPS]}),
    ("an entry that is not a pair", {"codec_report": [[-0.5] * GROUPS] * 2}),
    ("a served logprob that is a string", {"codec_report": [[["x", [[1, -0.5]]]] * GROUPS] * 2}),
    ("a served logprob that is a bool", {"codec_report": [[[True, [[1, -0.5]]]] * GROUPS] * 2}),
    ("no alternatives", {"codec_report": [[[-0.5, []]] * GROUPS] * 2}),
    ("more alternatives than requested",
     {"codec_report": [[[-0.5, [[1, -0.5]] * (TOP + 1)]] * GROUPS] * 2}),
    ("an alternative that is not a pair", {"codec_report": [[[-0.5, [[1]]]] * GROUPS] * 2}),
    ("an alternative id that is a float", {"codec_report": [[[-0.5, [[1.5, -0.5]]]] * GROUPS] * 2}),
    ("an alternative id that is negative", {"codec_report": [[[-0.5, [[-1, -0.5]]]] * GROUPS] * 2}),
    ("an alternative logprob that is a string",
     {"codec_report": [[[-0.5, [[1, "x"]]]] * GROUPS] * 2}),
    ("no eos_code", {"eos_code": None}),
    ("an eos_code that is a bool", {"eos_code": True}),
    ("an eos_code that is negative", {"eos_code": -1}),
    ("no eos_report", {"eos_report": None}),
    ("an eos_report that is not a pair", {"eos_report": [-0.5]}),
    ("one frame too long", {"codec_report": [[_entry(0)] * GROUPS] * 3}),
    ("a wide frame", {"codec_report": [[_entry(0)] * (GROUPS + 1)] * 2}),
    ("a served logprob past float range",
     {"codec_report": [[[10**400, [[1, -0.5]]]] * GROUPS] * 2}),
    ("an alternative logprob past float range",
     {"codec_report": [[[-0.5, [[1, -10**400]]]] * GROUPS] * 2}),
    ("an alternative id past 64 bits",
     {"codec_report": [[[-0.5, [[2**63, -0.5]]]] * GROUPS] * 2}),
    ("an eos_report alternative id past 64 bits", {"eos_report": [-0.5, [[2**63, -0.5]]]}),
    ("an eos_code past 64 bits", {"eos_code": 2**63}),
])
def test_a_malformed_report_fails_the_reply_not_the_drive(label, over):
    """Every one of these is the contestant's own output, so every one is recorded and
    graded rather than raised. A bool is its own case twice over, since bool subclasses
    int and would otherwise pass as the id 1 or the code 1.

    The numbers past range are JSON integers that Python reads exactly and then cannot
    store: a log-probability as a float, an alternative id in the report's 64-bit array,
    and the end-of-speech id, which the gate packs into a 64-bit array. Each would raise
    OverflowError, which no fault class covers, so one such reply would end the drive or
    the verify phase."""
    resp = _reporting_proto().decode_stream(
        _stream({"audio": _tone()}, _codes_frame(**over)), "r1")

    p = _payload(resp)
    assert p.codec_logprobs is None, label
    assert p.codec_codes_defect, label


def test_the_largest_id_the_report_stores_is_accepted():
    """The bound on an alternative's id and on the end-of-speech id is the storage's,
    the largest signed 64-bit integer, and not a tighter one: a codebook-0 position's
    alternatives reach above the codebook into the special ids."""
    codes, top = _rows(1), 2**63 - 1
    resp = _reporting_proto().decode_stream(
        _stream({"audio": _tone()},
                _codes_frame(codes, eos=top,
                             codec_report=[[[-0.5, [[top, -0.5]]]] * GROUPS])),
        "r1")

    p = _payload(resp)
    assert p.codec_codes_defect is None
    assert p.codec_logprobs[0].top_logprobs[0].token_id == top
    assert p.codec_logprobs[-1].token_id == top


def test_an_overflowing_report_in_a_unary_reply_is_a_report_defect():
    """The unary path reads the codes frame outside its payload guard, so a report the
    wire cannot store has to come back as a defect from the reader itself."""
    reply = {"audio": _tone(), "sample_rate": 24_000,
             **_codes_frame(codec_report=[[[10**400, [[1, -0.5]]]] * GROUPS] * 2)}
    p = _reporting_proto().decode_response(reply, "r1").contents[0]

    assert p.codec_codes == _rows(2)
    assert p.codec_logprobs is None
    assert "float range" in p.codec_codes_defect


def test_a_narrow_report_is_passed_to_the_gate():
    """Too few alternatives is a named gate failure with its own count, reported there
    far more precisely than here. Too many is refused above, because the gate's
    partition keeps the atoms above a fixed threshold rank and cannot recover from a
    row that describes a wider coarsening than the calibrated one."""
    codes = _rows(2)
    resp = _reporting_proto().decode_stream(
        _stream({"audio": _tone()},
                _codes_frame(codes, codec_report=[[_entry(c, n_alts=4) for c in row]
                                                  for row in codes])),
        "r1")

    p = _payload(resp)
    assert p.codec_codes_defect is None
    assert len(p.codec_logprobs[0].top_logprobs) == 4


def test_values_that_are_not_log_probabilities_reach_the_gate():
    """NaN and a positive log-probability are read by the gate, which names the first
    one that cannot be a log-probability. Rejecting them here would replace that
    message with a worse one and would duplicate a rule that lives with the mass
    arithmetic it protects."""
    codes = _rows(1)
    resp = _reporting_proto().decode_stream(
        _stream({"audio": _tone()},
                _codes_frame(codes, codec_report=[[[float("nan"), [[7, 4.0]]]] * GROUPS])),
        "r1")

    p = _payload(resp)
    assert p.codec_codes_defect is None
    assert p.codec_logprobs[0].logprob != p.codec_logprobs[0].logprob    # NaN survived
    assert p.codec_logprobs[0].top_logprobs[0].logprob == 4.0


def test_a_protocol_that_asks_for_no_report_reads_none():
    """The codes half stands alone, for a task whose gate re-decodes the codes and
    never looks at a distribution."""
    resp = _proto().decode_stream(_stream({"audio": _tone()}, _codes_frame()), "r1")

    p = _payload(resp)
    assert p.codec_codes == _rows(2)
    assert p.codec_logprobs is None
    assert p.codec_codes_defect is None


def test_the_report_flags_are_on_the_wire_and_in_the_brief():
    env = Envelope(request_id="r", contents=[TextPayload(text="hello")])
    body = _reporting_proto().encode_request(env, stream=True)
    assert body["return_logprobs"] is True and body["top_logprobs"] == TOP

    brief = _reporting_proto().brief_section(path="/v1/audio/speech", streaming=True)
    assert "codec_report" in brief and "eos_code" in brief
    assert "21 most probable" in brief

    quiet = _proto().brief_section(path="/v1/audio/speech", streaming=True)
    assert "codec_report" not in quiet


def test_the_end_of_speech_id_is_pinned_on_the_contract_and_published():
    """A gate builds its partition before it calls any model, so it cannot learn this
    id from the checkpoint; it reads it off the contract. The brief publishes the same
    number, and the example frame serves it, so the server and the gate agree from one
    place."""
    proto = _reporting_proto()
    assert proto.codec_eos_id == EOS

    brief = proto.brief_section(path="/v1/audio/speech", streaming=True)
    assert f"is {EOS}" in brief
    assert proto.body_schema()["codes_frame"]["eos_code"] == EOS


def test_the_wire_records_the_end_of_speech_id_the_server_actually_sent():
    """Not the pinned one. A reply that ended on something else has to be visible as
    that, so the gate can rule on it against the reference's own end-of-turn ids;
    overwriting it with the expected value would hide exactly the thing being graded."""
    resp = _reporting_proto().decode_stream(
        _stream({"audio": _tone()}, _codes_frame(eos=999)), "r1")

    p = _payload(resp)
    assert p.codec_codes_defect is None
    assert p.codec_logprobs[-1].token_id == 999


def test_the_example_report_frame_is_one_the_decoder_accepts():
    """The brief's example is abbreviated in two ways and still has to parse, which is
    what keeps the abbreviation from drifting into a shape the parser rejects."""
    proto = _reporting_proto()
    resp = proto.decode_stream(
        _stream({"audio": _tone()}, proto.body_schema()["codes_frame"]), "r1")

    p = _payload(resp)
    assert p.codec_codes_defect is None
    assert len(p.codec_logprobs) == GROUPS + 1


@pytest.mark.parametrize("kw,match", [
    ({"return_logprobs": True}, "return_codec_codes"),
    ({"return_codec_codes": True, "return_logprobs": True}, "top_logprobs"),
    ({"return_codec_codes": True, "top_logprobs": 21}, "return_logprobs"),
])
def test_a_report_the_reply_could_not_carry_is_refused_at_authoring(kw, match):
    """The report rides in the codes frame and is indexed by the codes, and it needs a
    width to be built at. Either flag alone asks a server for something with nowhere to
    go, which is worth catching where the task is written."""
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError, match=match):
        TTSStreamProtocol(**kw)


def test_the_report_survives_a_round_trip_through_the_trace():
    """The trace is the only surviving evidence once the containers are gone, and it is
    written by dumping the model to JSON. A field that does not come back is a verdict
    that cannot be re-derived from a graded run."""
    from hls.core.trace import Trace, TraceEvent

    codes = _rows(2)
    resp = _reporting_proto().decode_stream(
        _stream({"audio": _tone()}, _codes_frame(codes)), "r1")
    trace = Trace(events=[TraceEvent(
        request_id="r1", input=Envelope(request_id="r1", contents=[TextPayload(text="q")]),
        response=resp, t_send=0.0, t_recv=1.0)])

    back = Trace.model_validate_json(trace.model_dump_json())

    p = back.events[0].response.output.contents[0]
    assert p.codec_codes == codes
    assert [e.token_id for e in p.codec_logprobs] == [c for row in codes for c in row] + [EOS]
    assert len(p.codec_logprobs[0].top_logprobs) == TOP
    assert p.codec_logprobs[0].top_logprobs[3].logprob == pytest.approx(-4.0)
