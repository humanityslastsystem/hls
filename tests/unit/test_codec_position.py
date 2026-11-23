"""The packed codec report: same numbers, same JSON, same verifier reading, less memory.

``AudioPayload.codec_logprobs`` holds ``CodecPosition`` rather than ``TokenLogprob``
models because a speech reply's report is tens of thousands of alternatives, and models
of them take an order of magnitude more of the load driver's memory. Each test pins one
property against the ``TokenLogprob`` form.
"""
from __future__ import annotations

import base64
import io
import json
import math
import tracemalloc
import wave

from hls.core.payloads import (AudioPayload, CodecAlternative, CodecPosition, Envelope,
                               TextPayload, TokenLogprob, TopLogprob)
from hls.verify.report import reply_report


def _wav_b64() -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24_000)
        w.writeframes(b"\x00\x00" * 240)
    return base64.b64encode(buf.getvalue()).decode()


def _models(n_pos: int = 5, width: int = 65) -> list[TokenLogprob]:
    """A report as TokenLogprob models."""
    return [TokenLogprob(token_id=p, logprob=math.log(0.5),
                         top_logprobs=[TopLogprob(token_id=(p + j) % 2048,
                                                  logprob=-5.0 - 0.01 * j - 1e-6 * p)
                                       for j in range(width)])
            for p in range(n_pos)]


def _audio(report) -> AudioPayload:
    return AudioPayload(audio_b64=_wav_b64(), sample_rate=24_000,
                        codec_codes=[[0] * 4], codec_logprobs=report)


def test_a_trace_written_as_token_logprobs_reads_and_writes_back_identically():
    """A trace on disk carries each position as a TokenLogprob's dict. Reading it gives
    packed positions, and writing those gives the same bytes back."""
    models = _models()
    old_json = json.dumps([m.model_dump(mode="json") for m in models])
    payload = AudioPayload.model_validate(
        {"modality": "audio", "audio_b64": _wav_b64(), "sample_rate": 24_000,
         "codec_codes": [[0] * 4], "codec_logprobs": json.loads(old_json)})
    assert all(isinstance(p, CodecPosition) for p in payload.codec_logprobs)
    new_json = json.dumps(json.loads(payload.model_dump_json())["codec_logprobs"])
    assert new_json == old_json


def test_an_envelope_round_trips_through_json_with_the_positions_packed():
    """The polymorphic payload path (Envelope -> AudioPayload) keeps the packed type
    both ways, which is the path a Trace takes to and from disk."""
    env = Envelope(request_id="r", contents=[_audio(_models())])
    back = Envelope.model_validate_json(env.model_dump_json())
    assert back == env
    assert isinstance(back.contents[0].codec_logprobs[0], CodecPosition)


def test_the_verifier_reads_packed_positions_exactly_as_it_read_models():
    """The report verifier's rows come out the same, id for id and value for value."""
    models = _models(n_pos=8)
    served = [m.token_id for m in models]
    packed = [CodecPosition.from_entry(m) for m in models]
    want = reply_report(TextPayload(text="", logprobs=models), served, width=65)
    got = reply_report(TextPayload.model_construct(text="", logprobs=packed), served,
                       width=65)
    assert not isinstance(want, str)
    assert got == want


def test_alternatives_read_as_id_and_logprob_pairs_in_reported_order():
    pos = CodecPosition(7, -0.5, [3, 1, 2], [-1.0, -2.0, -3.0])
    assert pos.top_logprobs == [CodecAlternative(3, -1.0), CodecAlternative(1, -2.0),
                                CodecAlternative(2, -3.0)]
    assert pos.top_logprobs[0].token_id == 3 and pos.top_logprobs[0].logprob == -1.0
    assert pos.token is None and pos.bytes is None


def test_mismatched_ids_and_logprobs_are_refused():
    import pytest
    with pytest.raises(ValueError, match="alternative ids"):
        CodecPosition(0, -1.0, [1, 2], [-1.0])


def test_a_median_width_65_reply_fits_in_a_few_megabytes():
    """1,169 positions (73 frames of 16 plus end of speech) at 65 alternatives: 39 MB
    as TokenLogprob models. The packed form must stay an order of magnitude under that,
    or a drive's worth of replies does not fit beside its server."""
    ids = list(range(65))
    lps = [-1.0 - 0.01 * j for j in range(65)]
    tracemalloc.start()
    reply = [CodecPosition(p, -0.5, ids, lps) for p in range(1169)]
    used, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(reply) == 1169
    assert used < 3_000_000
