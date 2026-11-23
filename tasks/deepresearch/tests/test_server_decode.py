"""The HF baseline's sampled decode and the report it serves (no GPU: the recorder is
pure tensor math).

The image's ``/app/server_hf.py`` imports on a CPU host, so the recorder and the
generate kwargs that carry it are tested directly rather than through source reading.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.deepresearch.tests import baseline_sources

torch = pytest.importorskip("torch")

server, _config = baseline_sources.server("server_hf.py")
TokenText = _config.TokenText

#: A five-token vocabulary, most probable id 1, least probable id 4.
_LOGITS = torch.tensor([[0.0, 3.0, 1.0, 2.0, -5.0]])
_EXPECTED = torch.log_softmax(_LOGITS[0], dim=-1)


def _recorder(monkeypatch, *, top: int) -> server._ReportStreamer:
    """A recorder reporting ``top`` alternatives, with ids rendered as ``<id>``."""
    monkeypatch.setattr(server, "TOP_LOGPROBS", top)
    monkeypatch.setitem(server._STATE, "token_text", TokenText(lambda tid: f"<{tid}>"))
    return server._ReportStreamer()


def _draw(report: server._ReportStreamer, token_id: int, logits=_LOGITS) -> dict:
    """One generate step as transformers runs it: processors, then the sampled token.

    The first ``put`` is the prompt, which generate makes before any step.
    """
    report.put(torch.tensor([[7, 7, 7]]))
    assert report(torch.tensor([[7, 7, 7]]), logits) is logits, \
        "the recorder must hand generate back the logits it was given"
    report.put(torch.tensor([token_id]))
    report.end()
    [(served, entry)] = list(report)
    assert served == token_id
    return entry


def test_the_report_pairs_each_token_with_the_row_it_was_drawn_from(monkeypatch):
    """The gate compares the reported distribution with the reference's, so an entry
    has to carry the served token's own log-probability and the most probable
    alternatives from the same position."""
    entry = _draw(_recorder(monkeypatch, top=3), token_id=3)

    assert (entry["token_id"], entry["token"]) == (3, "<3>")
    assert entry["logprob"] == pytest.approx(_EXPECTED[3].item())
    assert entry["bytes"] == list("<3>".encode())
    assert [a["token_id"] for a in entry["top_logprobs"]] == [1, 3, 2]
    assert [a["logprob"] for a in entry["top_logprobs"]] == pytest.approx(
        [_EXPECTED[1].item(), _EXPECTED[3].item(), _EXPECTED[2].item()])


def test_a_token_drawn_from_the_tail_is_not_promoted_into_the_top_alternatives(
        monkeypatch):
    """Sampling draws unlikely tokens sometimes. The top list stays the most probable
    ids at that position, and the served token's own log-probability is reported
    separately, so a reply never claims a token was likely because it was drawn."""
    entry = _draw(_recorder(monkeypatch, top=2), token_id=4)

    assert [a["token_id"] for a in entry["top_logprobs"]] == [1, 3]
    assert entry["logprob"] == pytest.approx(_EXPECTED[4].item())


def test_a_token_with_no_recorded_distribution_is_refused(monkeypatch):
    """Reporting the previous position's row for this position would be a lie the gate
    could not see, so the pairing fails loudly instead."""
    report = _recorder(monkeypatch, top=2)
    report.put(torch.tensor([[7, 7, 7]]))          # the prompt

    with pytest.raises(RuntimeError, match="no recorded distribution"):
        report.put(torch.tensor([1]))


def test_generate_samples_the_whole_distribution_and_wires_the_recorder(monkeypatch):
    """Both response paths build their kwargs here, so this is where the decode is
    pinned: a draw at temperature 1 with every truncation and penalty knob neutral,
    stated explicitly so a checkpoint's own generation_config cannot apply its own."""
    monkeypatch.setitem(server._STATE, "tokenizer", SimpleNamespace(eos_token_id=0))
    report = server._ReportStreamer()

    kwargs = server._generate_kwargs(
        {"input_ids": torch.zeros(1, 5, dtype=torch.long)}, 16, report)

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 1.0
    assert (kwargs["top_p"], kwargs["top_k"], kwargs["min_p"]) == (1.0, 0, None)
    assert kwargs["repetition_penalty"] == 1.0
    assert list(kwargs["logits_processor"]) == [report]
    assert kwargs["streamer"] is report


def test_both_response_paths_build_their_generate_kwargs_in_one_place():
    source = Path(server.__file__).read_text()
    assert source.count("_generate_kwargs(inputs, max_new, report)") == 2
