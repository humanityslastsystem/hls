"""The two report reductions, run side by side on the same rows.

The gate reads three numbers per served position -- the reference's log-probability at
each reported id, the log of the mass outside those ids, and the log of the mass on the
end-of-turn ids -- and two engines compute them: ``HFEngine._reference_logprobs_prepared``
and ``hls.reference.vllm_report.reduce_rows``. A graded reply is scored against whichever
one its task's reference runs, so the two must not disagree.

Both real implementations run here. The HF one needs only a model that turns hidden
states into logits, so it is driven through a stub whose output head is the identity and
whose forward hands back a synthetic logit matrix; every line of its reduction (the
gather, the scatter, the two logsumexps, the rounding) is the shipped one. The vLLM side
is handed the same matrix log-softmaxed the way vLLM log-softmaxes it.

The cases are chosen to discriminate. A duplicate id, a padding sentinel that clamps onto
a real id 0, a row so peaked that the complement of the reported mass cancels, and an
end-of-turn id that is also reported each separate an implementation that gets the algebra
right from one that does not.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from hls.reference.engines import HFEngine  # noqa: E402
from hls.reference.vllm_report import reduce_rows  # noqa: E402

#: Both sides round to six decimals, so any real disagreement lands at 1e-6 or above.
#: Agreement is asserted far below that grid: the two paths apply the same fp32
#: operations in the same order to the same rows, so they match bit for bit in practice.
TOL = 1e-9


class _Config:
    """The two config facts the HF reduction reads: no softcapping, and a window."""

    final_logit_softcapping = None
    max_position_embeddings = 4096

    def get_text_config(self, decoder: bool = False):
        return self


class _Embeddings:
    """Stands in for the input embedding table, which the HF path reads for its width."""

    def __init__(self, vocab: int) -> None:
        self.num_embeddings = vocab


class _Model:
    """A model whose output head is the identity, so the caller supplies the logits."""

    def __init__(self, vocab: int) -> None:
        self.config = _Config()
        self._embeddings = _Embeddings(vocab)

    def get_input_embeddings(self):
        return self._embeddings

    def lm_head(self, hidden):
        return hidden


class _StubEngine:
    """Enough of an ``HFEngine`` to run its reduction on a given logit matrix.

    ``_continuation_hidden`` normally runs the forward pass; here it hands back the
    logits directly, which the identity output head then passes through unchanged.
    """

    def __init__(self, logits, eot_ids: list[int]) -> None:
        self._model = _Model(logits.shape[-1])
        self._logits = logits
        self._eot = set(eot_ids)

    def _eos_ids(self) -> set:
        return self._eot

    def _continuation_hidden(self, messages, tools, continuation, decoder_config,
                             continue_final_message=False):
        return self._logits, 7


def hf_reduce(logits, report_ids: list[list[int]], eot_ids: list[int],
              temperature: float = 1.0) -> dict:
    """The HF engine's own reduction over *logits*, one row per served position."""
    engine = _StubEngine(logits, eot_ids)
    continuation = [0] * logits.shape[0]
    prepared = [([{"role": "user", "content": "x"}], None, False, continuation, report_ids,
                 False)]
    return HFEngine._reference_logprobs_prepared(engine, prepared, temperature)[0]


def vllm_reduce(logits, report_ids: list[list[int]], eot_ids: list[int],
                temperature: float = 1.0) -> dict:
    """The vLLM capture's reduction over the same logits.

    vLLM log-softmaxes in fp32 inside its sampler and hands the rows straight to
    ``reduce_rows``; *temperature* is applied here only so a caller can ask what the
    reduction would return had the rows been scaled.
    """
    scores = torch.log_softmax(logits.float() / temperature, dim=-1)
    ids = torch.tensor(report_ids, dtype=torch.int64)
    return reduce_rows(scores, ids, torch.tensor(eot_ids, dtype=torch.int64))


def assert_same(a: dict, b: dict) -> None:
    """The three graded statistics agree position by position and slot by slot."""
    assert len(a["p_on_q"]) == len(b["p_on_q"])
    for row_a, row_b in zip(a["p_on_q"], b["p_on_q"]):
        assert len(row_a) == len(row_b)
        for x, y in zip(row_a, row_b):
            if x == float("-inf") or y == float("-inf"):
                assert x == y
            else:
                assert x == pytest.approx(y, abs=TOL)
    for key in ("p_tail_lp", "p_eot_lp"):
        assert len(a[key]) == len(b[key])
        for x, y in zip(a[key], b[key]):
            if x == float("-inf") or y == float("-inf"):
                assert x == y
            else:
                assert x == pytest.approx(y, abs=TOL)


def both(logits, report_ids: list[list[int]], eot_ids: list[int],
         temperature: float = 1.0) -> dict:
    """Run both reductions, assert they agree, and hand back the shared answer."""
    hf = hf_reduce(logits, report_ids, eot_ids, temperature)
    vllm = vllm_reduce(logits, report_ids, eot_ids, temperature)
    assert_same(hf, vllm)
    return hf


def test_a_plain_row_agrees():
    """Distinct reported ids, no padding: the ordinary case both engines mostly see."""
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, -1.0, 0.5]])
    got = both(logits, [[3, 2, 1]], [5])
    lp = torch.log_softmax(logits, dim=-1)
    assert got["p_on_q"][0] == [
        pytest.approx(lp[0, i].item(), abs=1e-6) for i in (3, 2, 1)]
    assert got["p_tail_lp"][0] == pytest.approx(
        torch.logsumexp(lp[0, [0, 4, 5]], dim=0).item(), abs=1e-6)


def test_a_duplicate_id_in_a_row_agrees():
    """A report naming the same id twice.

    The gather repeats the value, so ``p_on_q`` carries it twice on both sides; the tail
    must still drop that id exactly once. An implementation that subtracted the gathered
    mass from the total, or that counted the duplicate twice, would move the tail here.
    """
    logits = torch.tensor([[0.0, 2.0, 1.0, 0.5, -0.5]])
    got = both(logits, [[1, 1, 3]], [4])
    lp = torch.log_softmax(logits, dim=-1)
    assert got["p_on_q"][0][0] == got["p_on_q"][0][1]
    # The tail is ids 0, 2 and 4: id 1 leaves it once, not twice.
    assert got["p_tail_lp"][0] == pytest.approx(
        torch.logsumexp(lp[0, [0, 2, 4]], dim=0).item(), abs=1e-6)


def test_a_row_of_only_duplicates_agrees():
    """Every slot naming one id. The tail is everything else, on both sides."""
    logits = torch.tensor([[1.0, 3.0, 0.0, -2.0]])
    got = both(logits, [[1, 1, 1, 1]], [2])
    lp = torch.log_softmax(logits, dim=-1)
    assert got["p_tail_lp"][0] == pytest.approx(
        torch.logsumexp(lp[0, [0, 2, 3]], dim=0).item(), abs=1e-6)


def test_padding_sentinels_agree():
    """A pad slot carries id -1, clamps to column 0 under the gather, and must neither
    contribute mass nor mark a real id 0 as reported.

    Row 0 reports id 2 and pads the rest while id 0 holds most of the mass, so an
    implementation that let the clamped pad flag column 0 would drop that mass from the
    tail. Row 1 is all pad, where the tail is the whole distribution.
    """
    logits = torch.tensor([[4.0, 0.0, 1.0, 0.0],
                           [0.0, 1.0, 2.0, 3.0]])
    got = both(logits, [[2, -1, -1], [-1, -1, -1]], [3])
    lp = torch.log_softmax(logits, dim=-1)
    assert got["p_on_q"][0][1] == float("-inf")
    assert got["p_on_q"][1] == [float("-inf")] * 3
    assert got["p_tail_lp"][0] == pytest.approx(
        torch.logsumexp(lp[0, [0, 1, 3]], dim=0).item(), abs=1e-6)
    assert got["p_tail_lp"][1] == pytest.approx(0.0, abs=1e-6)


def test_a_pad_beside_a_reported_id_zero_agrees():
    """Id 0 reported in the same row as a pad slot.

    Both the pad and the real id land on column 0, so the reported flag has to survive
    the pad rather than be overwritten by it: id 0 must leave the tail.
    """
    logits = torch.tensor([[3.0, 1.0, 0.0, 0.5]])
    got = both(logits, [[0, -1]], [3])
    lp = torch.log_softmax(logits, dim=-1)
    assert got["p_on_q"][0][0] == pytest.approx(lp[0, 0].item(), abs=1e-6)
    assert got["p_tail_lp"][0] == pytest.approx(
        torch.logsumexp(lp[0, [1, 2, 3]], dim=0).item(), abs=1e-6)


def test_a_peaked_row_agrees():
    """A row where the reported ids hold all but about 1e-9 of the mass.

    This is the case that separates summing the complement in log space from deriving it
    as ``log(1 - exp(logsumexp(head)))``: in fp32 the second cancels. The naive value is
    computed here too, and asserted useless, so the case is known to discriminate.
    """
    logits = torch.tensor([[40.0, 39.0, 0.0, -5.0]])
    got = both(logits, [[0, 1]], [3])
    lp = torch.log_softmax(logits.double(), dim=-1)
    assert got["p_tail_lp"][0] == pytest.approx(
        torch.logsumexp(lp[0, [2, 3]], dim=0).item(), abs=1e-5)
    head = torch.logsumexp(torch.log_softmax(logits, dim=-1)[0, [0, 1]], dim=0)
    naive = torch.log1p(-torch.exp(head)).item()
    assert not math.isfinite(naive) or naive < got["p_tail_lp"][0] - 1.0


def test_a_row_where_the_report_holds_no_mass_agrees():
    """Reported ids the reference gives essentially nothing. The tail is then the whole
    distribution and must not round to zero mass on either side."""
    logits = torch.tensor([[0.0, 1.0, 2.0, -60.0, -60.0]])
    got = both(logits, [[3, 4]], [2])
    assert got["p_on_q"][0][0] < -50
    assert got["p_tail_lp"][0] == pytest.approx(0.0, abs=1e-6)


def test_an_end_of_turn_id_that_is_also_reported_agrees():
    """The end-of-turn mass is read off the whole row, so a reported end-of-turn id is
    counted there and dropped from the tail -- the same way on both sides."""
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    got = both(logits, [[3, 0]], [3])
    lp = torch.log_softmax(logits, dim=-1)
    assert got["p_eot_lp"][0] == pytest.approx(lp[0, 3].item(), abs=1e-6)
    assert got["p_on_q"][0][0] == pytest.approx(got["p_eot_lp"][0], abs=1e-6)
    assert got["p_tail_lp"][0] == pytest.approx(
        torch.logsumexp(lp[0, [1, 2]], dim=0).item(), abs=1e-6)


def test_several_end_of_turn_ids_agree():
    """Three end-of-turn ids, one of them reported: their mass sums in log space and is
    unaffected by which of them the report names."""
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 1.5]])
    got = both(logits, [[3, 0]], [1, 3, 4])
    lp = torch.log_softmax(logits, dim=-1)
    assert got["p_eot_lp"][0] == pytest.approx(
        torch.logsumexp(lp[0, [1, 3, 4]], dim=0).item(), abs=1e-6)


def test_a_single_end_of_turn_id_agrees():
    """One end-of-turn id: the log-sum degenerates to that id's own log-probability."""
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    got = both(logits, [[0]], [2])
    lp = torch.log_softmax(logits, dim=-1)
    assert got["p_eot_lp"][0] == pytest.approx(lp[0, 2].item(), abs=1e-6)


def test_an_empty_continuation_agrees():
    """No served positions: both sides return three empty lists rather than a row."""
    logits = torch.zeros((0, 4))
    hf = hf_reduce(logits, [], [3])
    vllm = reduce_rows(torch.zeros((0, 4)), torch.zeros((0, 2), dtype=torch.int64),
                       torch.tensor([3]))
    assert hf["p_on_q"] == hf["p_tail_lp"] == hf["p_eot_lp"] == []
    assert_same(hf, vllm)


def test_many_positions_cross_the_hf_head_block():
    """Enough rows to span the HF path's 2,048-row output-head block.

    The two sides block the head differently (2,048 rows against 256), so a block-relative
    slice that mis-indexed the reported ids would show up as a disagreement past the
    first block rather than in any single-row case above.
    """
    torch.manual_seed(0)
    rows, vocab = 2_100, 32
    logits = torch.randn(rows, vocab) * 3.0
    ids = (torch.arange(rows).unsqueeze(1) + torch.tensor([[0, 5, 11]])) % vocab
    report_ids = ids.tolist()
    for j in range(0, rows, 7):
        report_ids[j][2] = -1
    for j in range(0, rows, 11):
        report_ids[j][1] = report_ids[j][0]
    got = both(logits, report_ids, [0, vocab - 1])
    assert len(got["p_on_q"]) == len(got["p_tail_lp"]) == len(got["p_eot_lp"]) == rows


def test_both_round_to_the_same_grid():
    """Every reported float lands on the six-decimal grid on both sides."""
    torch.manual_seed(1)
    logits = torch.randn(16, 24) * 2.0
    hf = hf_reduce(logits, [[1, 5, -1]] * 16, [7])
    vllm = vllm_reduce(logits, [[1, 5, -1]] * 16, [7])
    for got in (hf, vllm):
        for row in got["p_on_q"]:
            for v in row:
                assert v == float("-inf") or round(v, 6) == v
        for key in ("p_tail_lp", "p_eot_lp"):
            for v in got[key]:
                assert round(v, 6) == v


def test_the_vllm_reduction_never_sees_the_decodes_temperature():
    """The HF path divides its logits by the decode's temperature before the
    log-softmax; the vLLM capture reduces vLLM's own log-softmax of the raw logits and
    has no temperature input at all.

    At temperature one the two coincide. Away from one they do not, and this pins that:
    the divergence is in how the rows are produced, not in how they are reduced.
    """
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    report_ids, eot = [[3, 1]], [0]
    assert_same(hf_reduce(logits, report_ids, eot, 1.0),
                vllm_reduce(logits, report_ids, eot, 1.0))
    hot = hf_reduce(logits, report_ids, eot, 2.0)
    raw = vllm_reduce(logits, report_ids, eot, 1.0)
    assert abs(hot["p_tail_lp"][0] - raw["p_tail_lp"][0]) > 0.1
    # Reducing rows that were scaled agrees again, so only the scaling is missing.
    assert_same(hot, vllm_reduce(logits, report_ids, eot, 2.0))


def test_the_hf_path_refuses_ragged_report_rows():
    """The HF reduction tensorizes the reported ids directly, so it needs every row the
    same width; the vLLM capture pads to the widest row itself. Rows reaching HF must
    already be padded."""
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        hf_reduce(logits, [[1, 2], [1]], [3])
    # The same report, padded to the widest row the way the vLLM capture pads it.
    both(logits, [[1, 2], [1, -1]], [3])


@pytest.mark.parametrize("eot_ids", [[3], [0, 3, 4]])
def test_the_stop_row_agrees(eot_ids):
    """The row past the reply, which an implied end of turn is priced on.

    HF reads it off the last of the ``n + 1`` rows its forward returns; the vLLM capture
    reduces the prompt's last row through ``reduce_rows`` with the end-of-turn ids as the
    reported ids. Both must name the same per-id log-probabilities and the same mass,
    and asking for the row must leave the reply's own rows as they were."""
    from hls.reference.vllm_report import _reduce_stop_row

    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, -1.0],
                           [1.0, 0.0, 2.0, 0.5, 3.0],
                           [2.5, -0.5, 0.0, 4.0, 1.0]])
    engine = _StubEngine(logits, eot_ids)
    item = ([{"role": "user", "content": "x"}], None, False, [0, 0], [[1, 2], [4, -1]])
    plain, stopped = HFEngine._reference_logprobs_prepared(
        engine, [(*item, False), (*item, True)], 1.0)
    vllm = _reduce_stop_row(lambda h: h, logits[2:3], eot_ids, torch)
    assert "p_stop_lp" not in plain
    assert_same(plain, stopped)
    assert len(stopped["p_stop_lp"]) == len(eot_ids)
    for x, y in zip(stopped["p_stop_lp"], vllm["p_stop_lp"]):
        assert x == pytest.approx(y, abs=TOL)
    assert stopped["p_stop_eot_lp"] == pytest.approx(vllm["p_stop_eot_lp"], abs=TOL)
    lp = torch.log_softmax(logits[2], dim=-1)
    assert stopped["p_stop_eot_lp"] == pytest.approx(
        torch.logsumexp(lp[eot_ids], dim=0).item(), abs=1e-6)
