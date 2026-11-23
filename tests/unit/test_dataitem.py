import pytest
from hls.core.dataitem import DatasetItem
from hls.core.payloads import Envelope, TextPayload


def _item(n_turns=1):
    env = Envelope(request_id="r", contents=[TextPayload(text="q")])
    return DatasetItem(
        request_id="r",
        turns=tuple(env for _ in range(n_turns)),
        ground_truth_turns=tuple("B" for _ in range(n_turns)),
    )


def test_single_turn_accessors():
    it = _item(1)
    assert it.request.first_of("text").text == "q"
    assert it.ground_truth == "B"


def test_request_accessor_rejects_multi_turn():
    it = _item(2)
    with pytest.raises(ValueError):
        _ = it.request
