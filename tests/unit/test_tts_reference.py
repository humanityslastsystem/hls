"""Tests for the Qwen3-TTS reference: its pinned decode, the distributions the sampling
gate scores against, and its codec decode.

Everything here runs on a CPU with no weights. The model is a fake built from real
torch embeddings and injected logits, shaped exactly like the packaged one: a talker
with a codec head, a sub-talker with one embedding table per code group after the
first, a codec decoder, and a ``generate`` that hands its prefix tensors to
``talker.generate`` the way the real one does. That is enough to exercise the arithmetic
the gate rests on -- where a scored row sits relative to the prefix, what each frame's
input embedding sums, which ids generation suppresses, how the repetition penalty reads
its history -- none of which needs a GPU to be wrong.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

# Imported after the torch guard above, which is what E402 is reporting: without torch
# these modules are importable but nothing in the file can run.
from hls.core.decode import Greedy, Sample       # noqa: E402
from hls.core.model import ModelSpec             # noqa: E402
from hls.core.payloads import AudioPayload, Envelope, TextPayload    # noqa: E402
from hls.reference.tts import (CODEC_SAMPLE_RATE, NON_STREAMING_MODE,   # noqa: E402
                               REPETITION_PENALTY, SUBTALKER_SAMPLE, TEMPERATURE,
                               TOP_K, Qwen3TTSEngine, Qwen3TTSReference,
                               _frame_embeds, _MIN_NEW_TOKENS, _processed_logprobs,
                               _trailing_term)

#: Fake model shape. Deliberately small everywhere except the talker vocabulary, which
#: has to exceed the 1,024-id special block generation suppresses or the suppression
#: range would run off the front of the table.
HIDDEN = 4
TALKER_VOCAB = 1200
SUB_VOCAB = 32
N_GROUPS = 4
EOS_ID = 1100
#: The first suppressed id: ids from here up are the special block.
FIRST_SPECIAL = TALKER_VOCAB - 1024
#: Length of the fake's talker prefix, in positions.
PREFIX_LEN = 5


class _FakeSpeechTokenizer:
    """The codec decoder: one waveform per code matrix, 1,920 samples per frame."""

    def __init__(self) -> None:
        self.calls: list[list] = []
        self.rate = CODEC_SAMPLE_RATE

    def decode(self, encoded):
        self.calls.append([e["audio_codes"] for e in encoded])
        wavs = [torch.zeros(int(e["audio_codes"].shape[0]) * 1920) + 0.25
                for e in encoded]
        return wavs, self.rate


class _FakeCodePredictor:
    """The sub-talker: one embedding table per code group after the first."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(vocab_size=SUB_VOCAB)
        self.generation_config = SimpleNamespace(repetition_penalty=1.0)
        self._embeds = torch.nn.ModuleList(
            [torch.nn.Embedding(SUB_VOCAB, HIDDEN) for _ in range(N_GROUPS - 1)])

    def get_input_embeddings(self):
        return self._embeds


class _FakeTalker:
    """The talker: an input embedding table, the sub-talker, and forwards that return
    whatever logits the test injected."""

    def __init__(self, owner: "_FakeNetwork") -> None:
        self.owner = owner
        self.code_predictor = _FakeCodePredictor()
        self._embed = torch.nn.Embedding(TALKER_VOCAB, HIDDEN)
        self.generation_config = SimpleNamespace(repetition_penalty=1.0)

    def get_input_embeddings(self):
        return self._embed

    def parameters(self):
        return iter([self._embed.weight])

    def generate(self, **kwargs):
        """What the real talker does with the prefix: decode from it. The engine's
        prefix capture works by intercepting this call, so a test that reaches here has
        gone down the generation path rather than the teacher-forcing one."""
        self.owner.prefix_seen = kwargs
        return self.owner.generated_codes

    def __call__(self, *, inputs_embeds, attention_mask, use_cache, output_hidden_states):
        """Return logits whose scored window is exactly what the test injected.

        The engine reads rows ``prefix_len - 1`` through ``prefix_len + n_frames``, so
        the injected rows are placed there and everything else is left at zero. The
        embeddings are recorded so a test can assert what each frame summed to."""
        self.owner.forwarded = inputs_embeds
        self.owner.forwarded_mask = attention_mask
        total = int(inputs_embeds.shape[1])
        logits = torch.zeros(1, total, TALKER_VOCAB)
        scored = self.owner.talker_logits
        if scored is not None:
            logits[0, PREFIX_LEN - 1: PREFIX_LEN - 1 + scored.shape[0]] = scored
        hidden = torch.zeros(1, total, HIDDEN)
        return SimpleNamespace(logits=logits, hidden_states=([hidden],))

    def forward_sub_talker_finetune(self, codec_ids, talker_hidden_states):
        self.owner.sub_inputs = (codec_ids, talker_hidden_states)
        return self.owner.sub_logits, None


class _FakeNetwork:
    """``Qwen3TTSForConditionalGeneration``'s shape: a talker, a codec decoder, a
    talker config, and a ``generate`` that builds a prefix and hands it to the talker."""

    def __init__(self) -> None:
        self.talker = _FakeTalker(self)
        self.speech_tokenizer = _FakeSpeechTokenizer()
        self.config = SimpleNamespace(talker_config=SimpleNamespace(
            vocab_size=TALKER_VOCAB, codec_eos_token_id=EOS_ID,
            num_code_groups=N_GROUPS, hidden_size=HIDDEN))
        self.talker_logits = None
        self.sub_logits = None
        self.forwarded = None
        self.forwarded_mask = None
        self.sub_inputs = None
        self.prefix_seen = None
        self.generate_calls: list[dict] = []
        self.generated_codes: list = []

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        prefix = {
            "inputs_embeds": torch.arange(
                PREFIX_LEN * HIDDEN, dtype=torch.float32).reshape(1, PREFIX_LEN, HIDDEN),
            "attention_mask": torch.ones(1, PREFIX_LEN, dtype=torch.long),
            # One row of real trailing text, then the pad embedding for every frame
            # after it, which is the shape the packaged model produces when the whole
            # sentence sits in the prefix.
            "trailing_text_hidden": torch.full((1, 1, HIDDEN), 3.0),
            "tts_pad_embed": torch.full((1, 1, HIDDEN), -1.0),
        }
        codes = self.talker.generate(**prefix)
        return codes, [torch.zeros(int(c.shape[0]), HIDDEN) for c in codes]


class _FakeModel:
    """``Qwen3TTSModel``'s shape: the wrapper that owns the network, the tokenizer
    helpers and the merged generate defaults."""

    def __init__(self, network: _FakeNetwork) -> None:
        self.model = network
        self.generate_defaults = {"do_sample": True, "repetition_penalty": 1.05,
                                  "subtalker_dosample": True, "max_new_tokens": 2048}

    def _build_assistant_text(self, text):
        return f"<assistant>{text}"

    def _build_instruct_text(self, instruct):
        return f"<instruct>{instruct}"

    def _tokenize_texts(self, texts):
        return [torch.tensor([[len(t)]]) for t in texts]

    def _merge_generate_kwargs(self, **kwargs):
        merged = dict(self.generate_defaults)
        merged.update({k: v for k, v in kwargs.items() if v is not None})
        return merged


def _engine(**kwargs) -> tuple[Qwen3TTSEngine, _FakeNetwork]:
    """An engine with its model already loaded from the fake, so ``_load`` never runs."""
    network = _FakeNetwork()
    eng = Qwen3TTSEngine(
        model_spec=ModelSpec(base_checkpoint="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                             arch="TX", chat_format="qwen3-tts"),
        decode=Greedy(), **kwargs)
    eng._model = _FakeModel(network)
    eng._device = "cpu"
    return eng, network


# ---- the pinned decode ------------------------------------------------------

def test_the_pinned_decode_names_every_argument_it_relies_on():
    """The checkpoint's generation config would supply all of these silently. Naming
    them makes the decode a thing the task declares, and makes a change to any of them
    move the workload fingerprint instead of passing unnoticed.

    The values are the checkpoint's own: taking the argmax on all sixteen codebooks
    stops the model terminating."""
    eng, _ = _engine()
    from hls.reference.tts import SUBTALKER_TEMPERATURE, SUBTALKER_TOP_K
    assert eng._generate_kwargs() == {
        "do_sample": False, "subtalker_dosample": SUBTALKER_SAMPLE,
        "repetition_penalty": REPETITION_PENALTY,
        "subtalker_temperature": SUBTALKER_TEMPERATURE,
        "subtalker_top_k": SUBTALKER_TOP_K}
    assert (REPETITION_PENALTY, SUBTALKER_SAMPLE) == (1.0, True)
    # A greedy configuration carries no sampling shape it never reads.
    greedy, _ = _engine(subtalker_sample=False)
    assert greedy._generate_kwargs() == {"do_sample": False, "subtalker_dosample": False,
                                         "repetition_penalty": REPETITION_PENALTY}


def test_the_prefix_is_built_under_the_pinned_decode_and_the_pinned_text_mode():
    """The pins reach the model as per-call arguments, which win over its defaults, and
    the text mode is pinned alongside them because streaming text conditions each frame
    differently."""
    eng, network = _engine()
    eng._talker_prefix("hello")
    call = network.generate_calls[-1]
    assert call["do_sample"] is False
    assert call["subtalker_dosample"] is SUBTALKER_SAMPLE
    assert call["repetition_penalty"] == REPETITION_PENALTY
    assert call["non_streaming_mode"] is NON_STREAMING_MODE
    assert call["speakers"] == ["Ryan"] and call["languages"] == ["English"]


def test_an_empty_instruction_is_none_not_an_empty_block():
    """The packaged path passes None for an absent instruction. Conditioning on an
    empty instruction block instead would be a different prefix, and so different
    codes."""
    eng, _ = _engine()
    assert eng._instruct_ids() is None
    spoken, _ = _engine(instruct="Speak quickly.")
    assert spoken._instruct_ids() is not None


def test_two_places_naming_the_temperature_may_not_disagree():
    """The brief renders the reference's field and the gate scores against it. A decode
    config naming a different number would grade a contestant on a distribution nobody
    told it about, so the engine refuses instead of silently preferring one."""
    import asyncio
    from hls.core.faults import HarnessFault
    eng, _ = _engine()
    eng.decode = Sample(temperature=0.7)
    with pytest.raises(HarnessFault, match="cannot differ"):
        asyncio.run(eng.reference_logprobs([_rl_item([[1, 2, 3, 4]])]))


# ---- the pieces of the forward ---------------------------------------------

def test_a_frames_embedding_is_the_sum_of_its_per_group_code_embeddings():
    """Codebook 0 through the talker's own table, each later group through its own
    sub-talker table, summed. A missing term would shift every frame's conditioning."""
    eng, network = _engine()
    codes = torch.tensor([[7, 1, 2, 3]])
    got = _frame_embeds(network.talker, codes)
    expected = network.talker.get_input_embeddings()(codes[:, 0])
    tables = network.talker.code_predictor.get_input_embeddings()
    for g in range(1, N_GROUPS):
        expected = expected + tables[g - 1](codes[:, g])
    assert torch.allclose(got, expected)


def test_the_trailing_term_runs_out_and_falls_back_to_the_pad_embedding():
    """While the text lasts each frame gets the next text hidden state; after it, the
    pad embedding. The fake's prefix carries one row of text, so frame 0 takes it and
    every later frame takes the pad."""
    prefix = {"trailing_text_hidden": torch.full((1, 1, HIDDEN), 3.0),
              "tts_pad_embed": torch.full((1, 1, HIDDEN), -1.0)}
    got = _trailing_term(prefix, 3)
    assert got.shape == (3, HIDDEN)
    assert torch.allclose(got[0], torch.full((HIDDEN,), 3.0))
    assert torch.allclose(got[1], torch.full((HIDDEN,), -1.0))
    assert torch.allclose(got[2], torch.full((HIDDEN,), -1.0))


def test_the_special_block_is_suppressed_except_the_end_of_speech_id():
    """Generation may only choose the end-of-speech id out of the special block, so
    teacher forcing must suppress the rest. Otherwise a special the model can never emit
    could be the argmax and every honest frame would score a gap against it."""
    eng, _ = _engine()
    suppressed = set(eng._suppress_ids())
    assert EOS_ID not in suppressed
    assert FIRST_SPECIAL in suppressed and TALKER_VOCAB - 1 in suppressed
    assert FIRST_SPECIAL - 1 not in suppressed


def _processed(rows, codes0, **kwargs):
    """``_processed_logprobs`` at the contract's sampling transforms, which are the
    identity: temperature 1 and top-k 0 leave the row a plain softmax, so the tests
    below read the generation processors on their own."""
    kwargs.setdefault("temperature", TEMPERATURE)
    kwargs.setdefault("top_k", TOP_K)
    return _processed_logprobs(rows, codes0, **kwargs)


def test_every_priced_row_is_a_distribution_over_everything_it_may_emit():
    """The contract's whole point. A truncated row prices some ids at exactly zero, and
    a served code priced at zero makes the divergence between two implementations
    infinite however small the real disagreement was. Under the plain softmax the only
    zeros are the ones the model's own structure puts there: the suppressed special
    block, and the end-of-speech id at the positions generation blocks it.

    Pinned against the module's own constants rather than literals, so a decode that
    quietly reacquires a top-k fails here rather than at a verdict."""
    torch.manual_seed(0)
    n_rows = _MIN_NEW_TOKENS + 3
    rows = torch.randn(n_rows, TALKER_VOCAB) * 4.0
    eng, _ = _engine()
    suppress = eng._suppress_ids()
    logp = _processed(rows, torch.arange(n_rows - 1),
                      suppress=torch.as_tensor(suppress), eos_id=EOS_ID,
                      repetition_penalty=REPETITION_PENALTY)

    assert (TEMPERATURE, TOP_K, REPETITION_PENALTY) == (1.0, 0, 1.0)
    assert torch.allclose(logp.exp().sum(dim=-1), torch.ones(n_rows), atol=1e-5)

    expected_zero = [set(suppress) for _ in range(n_rows)]
    for row in range(_MIN_NEW_TOKENS):
        expected_zero[row].add(EOS_ID)
    for row in range(n_rows):
        got_zero = {int(i) for i in torch.nonzero(logp[row] == -math.inf).flatten()}
        assert got_zero == expected_zero[row], f"row {row}"
        assert torch.isfinite(logp[row][logp[row] != -math.inf]).all()


def test_suppressed_ids_can_never_win_the_argmax():
    rows = torch.zeros(2, TALKER_VOCAB)
    rows[:, FIRST_SPECIAL + 3] = 99.0     # a special with a huge logit
    rows[:, 4] = 1.0                      # the best id the model may actually choose
    eng, _ = _engine()
    logp = _processed(
        rows, torch.tensor([4]), suppress=torch.as_tensor(eng._suppress_ids()),
        repetition_penalty=1.0, eos_id=EOS_ID)
    assert int(logp[0].argmax()) == 4
    assert logp[0, FIRST_SPECIAL + 3] == -math.inf


def test_the_end_of_speech_id_is_blocked_for_the_first_two_positions():
    """Generation carries min_new_tokens=2, so the model cannot stop before frame two.
    Scoring those positions against a distribution that allowed it would charge a gap
    for a choice generation never had."""
    rows = torch.zeros(4, TALKER_VOCAB)
    logp = _processed(rows, torch.tensor([1, 2, 3]),
                               suppress=torch.as_tensor([], dtype=torch.long), repetition_penalty=1.0,
                               eos_id=EOS_ID)
    assert logp[0, EOS_ID] == -math.inf and logp[1, EOS_ID] == -math.inf
    assert logp[2, EOS_ID] > -math.inf and logp[3, EOS_ID] > -math.inf


def test_the_repetition_penalty_reads_only_the_codes_before_each_position():
    """Position j is penalized for codes 0..j-1 and no further. Reading its own code
    too would penalize every frame for itself, which generation never does."""
    rows = torch.zeros(3, TALKER_VOCAB)
    rows[:, 9] = 2.0
    penalized = _processed(rows, torch.tensor([9, 9]),
                                    suppress=torch.as_tensor([], dtype=torch.long),
                                    repetition_penalty=2.0, eos_id=EOS_ID)
    plain = _processed(rows, torch.tensor([9, 9]), suppress=torch.as_tensor([], dtype=torch.long),
                                repetition_penalty=1.0, eos_id=EOS_ID)
    # Row 0 has an empty history, so it is untouched; rows 1 and 2 have seen id 9.
    assert penalized[0, 9] == plain[0, 9]
    assert penalized[1, 9] < plain[1, 9] and penalized[2, 9] < plain[2, 9]


# ---- codec decode ----------------------------------------------------------

def test_a_decoder_at_another_rate_is_a_broken_instrument():
    """Every waveform comparison downstream is at the codec's own rate. A decoder
    reporting a different one would make every one of them a resampling artifact."""
    from hls.core.faults import HarnessFault
    eng, network = _engine()
    network.speech_tokenizer.rate = 16_000
    with pytest.raises(HarnessFault, match="codec decoder reported"):
        eng._decode([torch.tensor([[1, 2, 3, 4]])])


def test_decode_codes_returns_one_waveform_per_matrix_and_batches():
    import asyncio
    eng, network = _engine(batch_size=2)
    wavs = asyncio.run(eng.decode_codes([[[1, 2, 3, 4]], [[1, 2, 3, 4], [5, 6, 7, 8]],
                                         [[2, 2, 2, 2]]]))
    assert [len(w) for w in wavs] == [1920, 3840, 1920]
    assert [len(c) for c in network.speech_tokenizer.calls] == [2, 1]


def test_decode_codes_refuses_an_id_outside_the_codebooks():
    import asyncio
    eng, _ = _engine()
    with pytest.raises(ValueError, match="outside this codec"):
        asyncio.run(eng.decode_codes([[[1, 2, 3, SUB_VOCAB]]]))


def test_decode_codes_on_nothing_deploys_nothing():
    import asyncio
    eng, network = _engine()
    assert asyncio.run(eng.decode_codes([])) == []
    assert network.speech_tokenizer.calls == []


# ---- the serialized config --------------------------------------------------

def test_the_reference_config_carries_every_knob_that_moves_the_codes():
    """Voice, language, instruction and penalty all change what the checkpoint would
    have chosen, so each is a serialized field: they ride into provenance and into the
    workload fingerprint, and changing one stales the calibration rather than silently
    re-pointing it."""
    fields = set(Qwen3TTSReference.model_fields)
    assert {"speaker", "language", "instruct", "repetition_penalty", "dtype",
            "batch_size"} <= fields
    ref = Qwen3TTSReference(decode=Greedy(), speaker="Aiden", instruct="Speak slowly.",
                            repetition_penalty=1.05)
    assert (ref.speaker, ref.instruct, ref.repetition_penalty) == (
        "Aiden", "Speak slowly.", 1.05)


# ---- the sampling gate's reference distributions ----------------------------

def _report_ids(n_positions: int, width: int = 4) -> list[list[int]]:
    """A partition per position: a few real ids and one pad slot, which is the shape
    the report module pads its rows to."""
    return [[1, 2, 3, -1] for _ in range(n_positions)]


def _rl_item(codes, report_ids=None, text="say this"):
    n = sum(len(r) for r in codes) + 1
    return Envelope(
        request_id="r.0",
        contents=[AudioPayload(audio_b64="", sample_rate=CODEC_SAMPLE_RATE,
                               codec_codes=codes)],
        metadata={"text": text,
                  "report_ids": _report_ids(n) if report_ids is None else report_ids})


def _run_rl(eng, network, codes, report_ids=None):
    import asyncio
    n_frames = len(codes)
    network.talker_logits = torch.zeros(n_frames + 1, TALKER_VOCAB)
    network.sub_logits = torch.zeros(n_frames, N_GROUPS - 1, SUB_VOCAB)
    return asyncio.run(eng.reference_logprobs([_rl_item(codes, report_ids)]))[0]


def _sampled_engine(**kwargs):
    eng, network = _engine(**kwargs)
    eng.decode = Sample(temperature=eng.temperature)
    return eng, network


def test_the_reference_returns_one_row_per_position_in_the_wire_order():
    """Row-major over (frame, group), then the end-of-speech decision. The reference
    and the report are zipped position by position, so an order disagreement produces
    numbers rather than an error."""
    eng, network = _sampled_engine()
    codes = [[1, 2, 3, 4], [5, 6, 7, 8]]
    out = _run_rl(eng, network, codes)
    want = len(codes) * N_GROUPS + 1
    assert len(out["p_on_q"]) == len(out["p_tail_lp"]) == len(out["p_eot_lp"]) == want
    assert out["eot_ids"] == [EOS_ID]
    # This checkpoint declares no text context window; the frame budget is what bounds a
    # reply, and the caller already holds it.
    assert out["max_context"] is None
    assert out["n_ctx"] == PREFIX_LEN


def test_a_pad_slot_gathers_no_mass_and_the_tail_holds_the_rest():
    eng, network = _sampled_engine()
    out = _run_rl(eng, network, [[1, 2, 3, 4]])
    row = out["p_on_q"][0]
    assert row[-1] == -math.inf, "the pad slot must gather minus infinity"
    assert all(math.isfinite(v) for v in row[:-1])
    # Three atoms out of a flat vocabulary leave almost everything in the tail.
    assert out["p_tail_lp"][0] > max(row[:-1])


def test_the_end_of_speech_price_is_zero_where_the_id_is_not_in_the_vocabulary():
    """At a residual group the sub-talker's vocabulary has no end-of-speech id at all,
    so the model's price for ending there is zero and its log is minus infinity. That
    is the model's own answer, not a mask: the count multiplies by it and the position
    contributes nothing."""
    eng, network = _sampled_engine()
    codes = [[1, 2, 3, 4], [5, 6, 7, 8]]
    out = _run_rl(eng, network, codes)
    prices = out["p_eot_lp"]
    for pos in range(len(prices) - 1):
        frame, group = divmod(pos, N_GROUPS)
        if group != 0:
            assert prices[pos] == -math.inf, f"position {pos} is a residual group"
        elif frame < 2:
            # Generation blocks ending before the third frame (min_new_tokens), so the
            # model's price here is zero for a different reason: it could not have
            # ended, rather than could not have represented ending.
            assert prices[pos] == -math.inf, f"frame {frame} cannot end"
        else:
            assert math.isfinite(prices[pos]), f"position {pos} is a first codebook"
    assert math.isfinite(prices[-1]), "the end-of-speech decision is priced"


def test_only_first_codebook_positions_can_ever_be_priced_into_the_count():
    """The end-of-turn count selects positions whose price is at or below a threshold.
    A residual group's price is zero, so it contributes a Bernoulli trial that can never
    fire; this asserts the count's evidence comes from the first codebook and the end of
    speech and nowhere else."""
    import numpy as np
    from hls.verify.report import end_of_turn_count
    eng, network = _sampled_engine()
    codes = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    out = _run_rl(eng, network, codes)
    prices = np.asarray(out["p_eot_lp"], dtype=np.float64)
    served = np.zeros(prices.shape, dtype=bool)
    served[-1] = True
    count = end_of_turn_count(prices, served, eps=0.5)
    # Every residual position carries probability zero, so the expected count is the
    # sum over the first-codebook positions alone.
    first = [p for i, p in enumerate(prices) if i % N_GROUPS == 0 or i == len(prices) - 1]
    assert count.expected == pytest.approx(sum(math.exp(p) for p in first))
    assert count.observed == 1


def test_the_transforms_are_temperature_then_top_k_then_the_softmax():
    """Order matters and is not the natural reading. Scaling then truncating keeps the
    same ids as truncating then scaling, but renormalizes over the scaled values, which
    is a different distribution. Generation does it this way."""
    from hls.reference.tts import _sampling_logprobs
    rows = torch.tensor([[4.0, 2.0, 1.0, 0.0]])
    got = _sampling_logprobs(rows, temperature=2.0, top_k=2)
    assert got[0, 2] == -math.inf and got[0, 3] == -math.inf
    scaled = rows / 2.0
    want = torch.log_softmax(
        torch.tensor([[scaled[0, 0], scaled[0, 1], -math.inf, -math.inf]]), dim=-1)
    assert torch.allclose(got, want, atol=1e-6)
    # And the check has teeth in both directions: a different temperature is a
    # different distribution, and no truncation is a different distribution again.
    assert not torch.allclose(got, _sampling_logprobs(rows, temperature=1.0, top_k=2))
    assert not torch.allclose(got, _sampling_logprobs(rows, temperature=2.0, top_k=0))


def test_a_reported_id_outside_its_position_vocabulary_closes_the_reply():
    """A residual group draws from the acoustic codebook alone, so an id above it is
    not something that position could report. Checked before the model is touched: an
    out-of-range gather is a device-side assert that poisons the run."""
    import asyncio
    eng, network = _sampled_engine()
    codes = [[1, 2, 3, 4]]
    ids = _report_ids(len(codes) * N_GROUPS + 1)
    ids[1] = [SUB_VOCAB + 5, 2, 3, -1]      # position 1 is a residual group
    out = asyncio.run(eng.reference_logprobs([_rl_item(codes, ids)]))[0]
    assert "invalid" in out and "outside" in out["invalid"]
    assert network.forwarded is None, "nothing should have reached the model"
    # The same id at a first-codebook position is legitimate: that one draws from the
    # talker's whole vocabulary, including the range generation suppresses.
    ids2 = _report_ids(len(codes) * N_GROUPS + 1)
    ids2[0] = [SUB_VOCAB + 5, 2, 3, -1]
    assert "invalid" not in _run_rl(eng, network, codes, ids2)


def test_a_report_that_does_not_cover_every_position_is_refused():
    import asyncio
    eng, _ = _sampled_engine()
    codes = [[1, 2, 3, 4], [5, 6, 7, 8]]
    short = _report_ids(len(codes) * N_GROUPS)      # one row short: no end of speech
    with pytest.raises(ValueError, match="rows of metadata"):
        asyncio.run(eng.reference_logprobs([_rl_item(codes, short)]))


def test_a_reply_with_no_frames_scores_nothing():
    import asyncio
    eng, _ = _sampled_engine()
    out = asyncio.run(eng.reference_logprobs([_rl_item([], [[1, 2, 3, -1]])]))[0]
    assert out["p_on_q"] == [] and out["n_ctx"] == 0
    assert out["eot_ids"] == [EOS_ID]


def test_a_ragged_code_matrix_is_refused():
    """Every frame carries the same number of groups. A ragged matrix has no position
    layout at all, so it is refused before anything is scored rather than silently
    measured against a partition of another shape."""
    import asyncio
    eng, _ = _sampled_engine()
    with pytest.raises(ValueError, match="ragged code matrix"):
        asyncio.run(eng.reference_logprobs([_rl_item([[1, 2, 3, 4], [5, 6]])]))


def test_a_served_code_outside_the_codebooks_closes_the_reply():
    """An out-of-range id is checked before the embedding gather. On a GPU that index
    is a device-side assert which poisons the context for the rest of the run, so one
    unfaithful reply would become a harness fault instead of a failed reply."""
    import asyncio
    eng, network = _sampled_engine()
    codes = [[1, 2, 3, 4], [5, SUB_VOCAB + 4, 7, 8]]
    out = asyncio.run(eng.reference_logprobs([_rl_item(codes)]))[0]
    assert "outside this checkpoint" in out["invalid"]
    assert network.forwarded is None, "nothing should have reached the model"


def _scored_position_by_position(first, residual, report_ids, *, eos_id, decimals=6):
    """What the reference returned for one utterance when it priced each position on its
    own: the gather, the masked logsumexp and the end-of-speech price, one row at a time,
    each read back to the host as it was computed. The oracle for the batched scoring.
    ``decimals`` None leaves the gathered values and the tails unrounded."""
    def rounded(x):
        return x if decimals is None else round(x, decimals)

    n_frames, n_groups = residual.shape[0], residual.shape[1] + 1
    out = {"p_on_q": [], "p_tail_lp": [], "p_eot_lp": []}
    for pos, ids in enumerate(report_ids):
        frame, group = divmod(pos, n_groups)
        if pos == len(report_ids) - 1:
            lp = first[n_frames]
        elif group == 0:
            lp = first[frame]
        else:
            lp = residual[frame, group - 1]
        kept = [t for t in ids if t >= 0]
        out["p_on_q"].append([rounded(float(lp[t])) if t >= 0 else -math.inf for t in ids])
        if kept:
            mask = torch.zeros_like(lp, dtype=torch.bool)
            mask[torch.as_tensor(kept, dtype=torch.long)] = True
            tail = float(torch.logsumexp(lp.masked_fill(mask, -math.inf), dim=-1))
        else:
            tail = float(torch.logsumexp(lp, dim=-1))
        out["p_tail_lp"].append(rounded(tail))
        out["p_eot_lp"].append(float(lp[eos_id]) if lp.shape[-1] > eos_id else -math.inf)
    return out


def _random_report(n_frames: int, gen: "torch.Generator") -> list[list[int]]:
    """Report ids for every position of an utterance, at a width of 6: distinct ids from
    the position's own vocabulary with pad slots mixed in, plus the rows that are easy to
    get wrong in a batch: a row of pads only, id 0 beside a pad (pads must not land on
    id 0), the vocabulary's last id, the end-of-speech id and a suppressed special."""
    rows = []
    for pos in range(n_frames * N_GROUPS + 1):
        vocab = TALKER_VOCAB if pos % N_GROUPS == 0 or pos == n_frames * N_GROUPS else SUB_VOCAB
        ids = torch.randperm(vocab, generator=gen)[:6].tolist()
        n_pad = int(torch.randint(0, 3, (1,), generator=gen))
        rows.append(ids[:6 - n_pad] + [-1] * n_pad)
    rows[1] = [-1] * 6
    rows[2] = [0, 5, -1, 5, -1, 1]
    rows[3] = [SUB_VOCAB - 1, 0, 1, 2, 3, -1]
    rows[0] = [EOS_ID, FIRST_SPECIAL + 2, TALKER_VOCAB - 1, 0, -1, -1]
    rows[-1] = [EOS_ID, 0, 1, -1, -1, -1]
    return rows


def _captured_rows(monkeypatch):
    """Record the two row blocks ``_score_report`` prices from: the processed codebook-0
    rows and the residual rows, in the order it computes them."""
    import hls.reference.tts as tts
    seen: dict[str, object] = {}
    processed, sampling = tts._processed_logprobs, tts._sampling_logprobs

    def record_processed(*a, **k):
        seen["first"] = processed(*a, **k)
        return seen["first"]

    def record_sampling(*a, **k):
        seen["residual"] = sampling(*a, **k)
        return seen["residual"]

    monkeypatch.setattr(tts, "_processed_logprobs", record_processed)
    monkeypatch.setattr(tts, "_sampling_logprobs", record_sampling)
    return seen


def _bits(values) -> tuple:
    """A row, or rows, of floats as their IEEE 754 bit patterns, which tell 0.0 from -0.0."""
    import struct
    if isinstance(values, list):
        return tuple(_bits(v) for v in values)
    return struct.pack("<d", values)


def test_the_batched_scoring_returns_exactly_the_per_position_numbers(monkeypatch):
    """Every number the reference returns is the one pricing each position on its own
    gave: the same gathered log-probabilities, the same tails to the last bit before
    rounding, the same end-of-speech prices. Batching may change how the numbers reach
    the host and nothing else, because the calibration was computed from the
    per-position numbers. Compared by bit pattern, once with the
    rounding switched off and once as returned."""
    import asyncio

    import hls.reference.tts as tts
    gen = torch.Generator().manual_seed(7)
    eng, network = _sampled_engine()
    seen = _captured_rows(monkeypatch)
    for n_frames in (1, 3, 9):
        codes = torch.randint(0, SUB_VOCAB, (n_frames, N_GROUPS), generator=gen).tolist()
        report = _random_report(n_frames, gen)
        network.talker_logits = torch.randn(n_frames + 1, TALKER_VOCAB, generator=gen) * 3.0
        network.sub_logits = torch.randn(n_frames, N_GROUPS - 1, SUB_VOCAB, generator=gen) * 3.0
        for decimals in (None, 6):
            with monkeypatch.context() as m:
                if decimals is None:
                    m.setattr(tts, "round", lambda x, ndigits=None: x, raising=False)
                got = asyncio.run(eng.reference_logprobs([_rl_item(codes, report)]))[0]
            want = _scored_position_by_position(seen["first"], seen["residual"], report,
                                                eos_id=EOS_ID, decimals=decimals)
            for key in ("p_on_q", "p_tail_lp", "p_eot_lp"):
                assert _bits(got[key]) == _bits(want[key]), (
                    f"{key} differs at {n_frames} frames, rounding to {decimals}")


def test_each_tail_is_one_reduction_over_one_vocabulary_row(monkeypatch):
    """Each tail is its own ``logsumexp`` over one contiguous row the length of its
    position's vocabulary, the reduction pricing the position alone ran. One reduction over
    the whole block sums in another order, which can move rounded tails on a GPU where a
    CPU shows no difference, so the structure is what this pins."""
    calls = []
    original = torch.logsumexp

    def spy(x, *a, **k):
        calls.append((x.dim(), x.is_contiguous(), x.shape[-1]))
        return original(x, *a, **k)

    monkeypatch.setattr(torch, "logsumexp", spy)
    eng, network = _sampled_engine()
    n_frames = 5
    _run_rl(eng, network, [[1, 2, 3, 4]] * n_frames)
    assert all(dim == 1 and contiguous for dim, contiguous, _ in calls)
    assert sorted(length for *_, length in calls) == sorted(
        [TALKER_VOCAB] * (n_frames + 1) + [SUB_VOCAB] * (n_frames * (N_GROUPS - 1)))


def test_scoring_an_utterance_reads_back_to_the_host_a_fixed_number_of_times(monkeypatch):
    """Each read of a device tensor's value on the host (``float``, ``item``, ``tolist``)
    waits for the device. Pricing positions one read at a time makes the report gate a
    stream of tiny synchronizing copies, so its time is set by the host core's latency
    rather than by the GPU's work. The reads per utterance must not
    grow with its length."""
    counts = {"reads": 0}
    for name in ("__float__", "item", "tolist"):
        original = getattr(torch.Tensor, name)

        def counted(self, *a, _original=original, **k):
            counts["reads"] += 1
            return _original(self, *a, **k)

        monkeypatch.setattr(torch.Tensor, name, counted)
    eng, network = _sampled_engine()

    def reads(n_frames: int) -> int:
        counts["reads"] = 0
        _run_rl(eng, network, [[1, 2, 3, 4]] * n_frames)
        return counts["reads"]

    assert reads(2) == reads(12)


def test_an_item_without_its_sentence_or_its_codes_is_refused():
    """The sentence is the conditioning every distribution is computed under and the
    codes are what is being priced. Neither has a default."""
    import asyncio
    eng, _ = _sampled_engine()
    with pytest.raises(ValueError, match=r"metadata\['text'\]"):
        asyncio.run(eng.reference_logprobs([Envelope(
            request_id="r.0",
            contents=[AudioPayload(audio_b64="", codec_codes=[[1, 2, 3, 4]])],
            metadata={"report_ids": [[1, -1]] * 5})]))
    with pytest.raises(TypeError, match="codec_codes"):
        asyncio.run(eng.reference_logprobs([Envelope(
            request_id="r.0", contents=[TextPayload(text="x")],
            metadata={"text": "say this", "report_ids": [[1, -1]]})]))
