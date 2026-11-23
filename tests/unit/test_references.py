"""Reference is the pure config face; its runtime Engine (built via
``build_engine(task)``) is what the verify role loads on its engine host, on one GPU. Both
engine operations take and return a BATCH so the engine packs many prompts into one forward
pass."""

import asyncio

import pytest

from hls.core.decode import Greedy, Sample
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.reference.engines import ReferenceEngine
from hls.reference.references import HFReference, Reference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.taskspec import TaskSpec
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify import LetterMatchMetric, MetricVerifier


def test_reference_is_config_poly():
    """Reference is a pure pydantic config (Poly); the runtime face is its engine."""
    from hls.core.engine import Engine
    from hls.core.poly import Poly
    assert issubclass(Reference, Poly)
    assert issubclass(ReferenceEngine, Engine)
    assert "resources" not in ReferenceEngine.__abstractmethods__


def test_reference_has_no_dependencies_method():
    assert not hasattr(Reference, "dependencies")


def _model() -> ModelSpec:
    return ModelSpec(base_checkpoint="x", arch="TX", chat_format="chatml")


def _task(model: ModelSpec | None = None) -> TaskSpec:
    m = model or _model()
    return TaskSpec(
        id="T",
        model=m,
        hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
        contract=RequestResponseContract(
            path="/v1/chat/completions",
            input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"text"})),
        workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(), slo=NoSLO()),
        verifiers=[MetricVerifier(metric=LetterMatchMetric())],  # stub: a task needs >=1 verifier
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)),
    )


def test_reference_fingerprint_folds_in_decode():
    task = _task()
    greedy = HFReference(decode=Greedy())
    sample = HFReference(decode=Sample(temperature=0.7))
    assert greedy.fingerprint(task) != sample.fingerprint(task)
    assert "greedy" in greedy.fingerprint(task)


def test_reference_fingerprint_folds_every_config_knob():
    # The fingerprint hashes the dump_spec payload, so EVERY output-affecting
    # field a subclass declares scopes the cache -- not just (model, decode).
    # dtype is the canary: bfloat16 vs float16 oracles produce different text.
    from hls.reference.references import VLLMReference
    task = _task()
    bf16 = VLLMReference(decode=Greedy(), dtype="bfloat16")
    fp16 = VLLMReference(decode=Greedy(), dtype="float16")
    assert bf16.fingerprint(task) != fp16.fingerprint(task)
    # Stable: an identical config re-fingerprints identically.
    assert bf16.fingerprint(task) == VLLMReference(
        decode=Greedy(), dtype="bfloat16").fingerprint(task)


def test_the_declared_query_chunk_reaches_the_engine():
    """The chunk is declared on the reference and read off the engine by
    ``_continuation_hidden``, so what a task writes is what the forward is tiled at. A
    reference that declares nothing gets the width that fits the longest context."""
    from hls.reference.engines import DEFAULT_QUERY_CHUNK

    task = _task()
    assert HFReference(decode=Greedy(), query_chunk=4096).build_engine(task).query_chunk == 4096
    assert HFReference(decode=Greedy()).build_engine(task).query_chunk == DEFAULT_QUERY_CHUNK


def test_the_query_chunk_scopes_the_reference_fingerprint():
    """Two chunks are two instruments, so they cannot share a cache scope."""
    task = _task()
    assert (HFReference(decode=Greedy(), query_chunk=512).fingerprint(task)
            != HFReference(decode=Greedy(), query_chunk=4096).fingerprint(task))


def test_a_query_chunk_below_one_row_is_refused():
    """A chunk of zero rows walks the query axis forever; the pin is checked where the
    other instrument pins are, on the torch-free config host."""
    with pytest.raises(ValueError, match="query_chunk"):
        HFReference(decode=Greedy(), query_chunk=0).build_engine(_task())


def test_chat_prompt_text_folds_the_turns_tools_into_the_prompt():
    # A turn's tools are part of its rendered context, so the reference must pass them
    # to the chat template; without this the artifact and the oracle render different
    # prompts and every teacher-forced position scores a gap.
    from hls.reference.engines import chat_prompt_text

    class _Tok:
        def apply_chat_template(self, messages, **kwargs):
            return repr(sorted(kwargs.items()))

    tools = [{"type": "function", "function": {"name": "web_search"}}]
    assert "tools" not in chat_prompt_text(_Tok(), [{"role": "user", "content": "q"}])
    assert "web_search" in chat_prompt_text(_Tok(), [{"role": "user", "content": "q"}], tools)


def test_tools_of_treats_an_empty_tool_list_as_no_tools():
    # An empty list must render identically to no tools argument at all, or a
    # final-answer turn (which offers none) would differ from the same turn in a
    # task that never sets the key.
    from hls.reference.engines import tools_of

    assert tools_of(Envelope(request_id="r", contents=[TextPayload(text="q")],
                              metadata={"tools": []})) is None
    assert tools_of(Envelope(request_id="r", contents=[TextPayload(text="q")])) is None


def _tf_item(tokens=(1, 2), messages=(("user", "q"),), max_tokens=64):
    msgs = [{"role": r, "content": c} for r, c in messages]
    return Envelope(request_id="r", metadata={"messages": msgs, "max_tokens": max_tokens},
                    contents=[TextPayload(text="x", tokens=list(tokens))])


def test_hf_teacher_force_empty_returns_empty():
    task = _task()
    eng = HFReference(decode=Greedy()).build_engine(task)
    assert asyncio.run(eng.teacher_force([])) == []


def test_hf_teacher_force_requires_messages():
    """Context lives in metadata['messages']; contents is the continuation, so there
    is no fallback -- a missing 'messages' raises (before any model load)."""
    task = _task()
    eng = HFReference(decode=Greedy()).build_engine(task)
    item = Envelope(request_id="r", contents=[TextPayload(text="x", tokens=[1, 2])])
    with pytest.raises(ValueError, match="messages"):
        asyncio.run(eng.teacher_force([item]))


def test_hf_teacher_force_requires_token_ids():
    """The continuation must be exact ids (TextPayload.tokens); none present raises."""
    task = _task()
    eng = HFReference(decode=Greedy()).build_engine(task)
    item = Envelope(request_id="r", metadata={"messages": [{"role": "user", "content": "q"}],
                                               "max_tokens": 64},
                    contents=[TextPayload(text="x")])  # tokens defaults None
    with pytest.raises(TypeError, match="token ids"):
        asyncio.run(eng.teacher_force([item]))


def test_hf_teacher_force_requires_the_stamped_cap():
    """The request's max_tokens decides whether the reply's stop is scored, so an item
    without it raises (before any model load) rather than scoring the stop under a
    guess in either direction."""
    task = _task()
    eng = HFReference(decode=Greedy()).build_engine(task)
    item = Envelope(request_id="r", metadata={"messages": [{"role": "user", "content": "q"}]},
                    contents=[TextPayload(text="x", tokens=[1, 2])])
    with pytest.raises(ValueError, match="max_tokens"):
        asyncio.run(eng.teacher_force([item]))


def test_hf_teacher_force_sample_decode_raises():
    task = _task()
    eng = HFReference(decode=Sample(temperature=0.7)).build_engine(task)
    with pytest.raises(NotImplementedError):
        asyncio.run(eng.teacher_force([_tf_item()]))


# ---------------------------------------------------------------------------
# stop gap: the position after the last served token
# ---------------------------------------------------------------------------


def _stub_engine(logprob_rows, eos_ids=(9,), vocab=16, decode=None):
    """An HFEngine with the model/tokenizer replaced by stubs, so _teacher_force_prepared
    runs its real slicing and gap arithmetic against chosen logits.

    `logprob_rows[i]` are the raw logits for position i of the full sequence; the context
    is one token, so row 0 predicts continuation[0]. `decode` is the decode config the
    engine scores under, defaulting to plain greedy. The gap arithmetic runs on real
    tensors, so these tests skip where torch is not installed (the CI runner).
    """
    torch = pytest.importorskip("torch")

    from hls.core.decode import Greedy
    from hls.reference.references import HFReference

    eng = HFReference(decode=decode or Greedy()).build_engine(_task())
    rows = torch.tensor(logprob_rows, dtype=torch.float32)

    class _Config:
        """Declares no attention structure and no logit softcapping, so the scoring
        pass builds transformers' stock cache and scores the stub's rows as they
        stand. A text-only config answers ``get_text_config`` with itself."""

        final_logit_softcapping = None

        def get_text_config(self, decoder=None, encoder=None):
            return self

    class _Model:
        def model(self, ids, past_key_values=None, use_cache=True):
            # Every fixture here is one forward per item, so the rows are read from
            # the start each time and several items can be scored in one call.
            n = ids.shape[1]
            return type("O", (), {"last_hidden_state": rows[:n].unsqueeze(0),
                                  "past_key_values": None})()

        def lm_head(self, h):
            return h  # hidden rows ARE the logits in this stub

        def get_input_embeddings(self):
            return type("E", (), {"num_embeddings": vocab})()

        config = _Config()
        generation_config = type("G", (), {"eos_token_id": list(eos_ids)})()

    class _Tok:
        eos_token_id = eos_ids[0]

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "ctx"

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            # Real tokenizers return a BatchEncoding, which the engine .to(device)s.
            class _Enc(dict):
                def to(self, _device):
                    return self
            return _Enc(input_ids=torch.tensor([[0]]))

    eng._model, eng._tokenizer, eng._device = _Model(), _Tok(), "cpu"
    eng._load = lambda: None
    return eng


def test_entitled_stop_costs_nothing():
    """A server that stops where the reference itself would stop scores a 0 gap there,
    so the extra position is free for an honest reply."""
    # 3 rows: two predict the served tokens, the third is the stop, where the argmax IS
    # the eos id 9 -> gap 0.
    rows = [[0.0] * 16, [0.0] * 16, [0.0] * 16]
    rows[0][1] = 5.0   # served token 1 is argmax -> gap 0
    rows[1][2] = 5.0   # served token 2 is argmax -> gap 0
    rows[2][9] = 5.0   # stop position: eos is argmax
    eng = _stub_engine(rows)
    out = eng._teacher_force_prepared([([{"role": "user", "content": "q"}], None, False, [1, 2], True)])
    assert out[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)


def test_early_stop_spends_the_gap_ceilings():
    """The truncation case: a faithful prefix cut off where the reference would not have
    stopped. Every emitted token scores 0 -- which is why the flip rules alone cannot see
    it -- and the stop lands as one large gap, counted like any other deviation."""
    rows = [[0.0] * 16, [0.0] * 16, [0.0] * 16]
    rows[0][1] = 5.0
    rows[1][2] = 5.0
    rows[2][7] = 12.0  # stop position: an ordinary token dominates eos by a wide margin
    eng = _stub_engine(rows)
    out = eng._teacher_force_prepared([([{"role": "user", "content": "q"}], None, False, [1, 2], True)])
    assert out[0][:2] == pytest.approx([0.0, 0.0], abs=1e-5), "prefix is faithful"
    assert out[0][2] > 10.0, "the stop is where the deviation shows up"


def test_a_length_reply_has_no_stop_to_score():
    """A reply cut off at the output cap never claimed to have ended, so it gets no stop
    gap -- otherwise every reply that ran to the cap would score a huge deviation for
    obeying our own cap."""
    rows = [[0.0] * 16, [0.0] * 16, [0.0] * 16]
    rows[0][1] = 5.0
    rows[1][2] = 5.0
    rows[2][7] = 12.0
    eng = _stub_engine(rows)
    out = eng._teacher_force_prepared([([{"role": "user", "content": "q"}], None, False, [1, 2], False)])
    assert out[0] == pytest.approx([0.0, 0.0], abs=1e-5)


def test_a_reply_ending_in_eos_is_not_charged_for_stopping_twice():
    """A server that reports its end-of-turn token in band -- which vLLM's chat API does --
    already had that token teacher-forced. Scoring the position after it would charge the
    same decision again, and charge it savagely: after an end-of-turn token the reference
    expects anything but another one, so the gap is enormous."""
    rows = [[0.0] * 16, [0.0] * 16, [0.0] * 16]
    rows[0][1] = 5.0    # served token 1 is argmax -> gap 0
    rows[1][9] = 5.0    # served token 2 IS the eos id, and is the argmax -> gap 0
    rows[2][7] = 40.0   # would-be stop row: eos is 40 nats behind an ordinary token
    eng = _stub_engine(rows)
    out = eng._teacher_force_prepared([([{"role": "user", "content": "q"}], None, False, [1, 9], True)])
    assert out[0] == pytest.approx([0.0, 0.0], abs=1e-5), (
        "one gap per served token and no stop gap")


def test_stopping_early_is_still_caught_when_the_eos_is_emitted():
    """Skipping the stop row does not create a hole. An end-of-turn id emitted where the
    reply should have continued is a flip at its own position, so the deviation is charged
    once, where it happened."""
    rows = [[0.0] * 16, [0.0] * 16, [0.0] * 16]
    rows[0][1] = 5.0
    rows[1][7] = 12.0   # the reference wanted an ordinary token; the server sent eos
    rows[2][7] = 40.0
    eng = _stub_engine(rows)
    out = eng._teacher_force_prepared([([{"role": "user", "content": "q"}], None, False, [1, 9], True)])
    assert len(out[0]) == 2, "no stop row"
    assert out[0][0] == pytest.approx(0.0, abs=1e-5)
    assert out[0][1] > 10.0, "the premature end-of-turn token is the flip"


def test_teacher_force_gap_count_still_matches_the_continuation():
    """The extra scored row must not leak into the per-token gaps -- one gap per served
    token, however many rows the forward pass produced."""
    rows = [[float(i == j) for i in range(16)] for j in range(6)]
    eng = _stub_engine(rows)
    out = eng._teacher_force_prepared(
        [([{"role": "user", "content": "q"}], None, False, [1, 2, 3, 4, 5], False)])
    assert len(out[0]) == 5


# ---------------------------------------------------------------------------
# frequency penalty: the reference's argmax after penalty * count(v)
# ---------------------------------------------------------------------------


def _penalty_rows(a=5.0, b=4.95):
    """Two rows where id 1 leads id 2 by a margin smaller than one unit of a 0.1
    penalty, so the second occurrence of id 1 moves the reference's choice."""
    rows = [[0.0] * 16, [0.0] * 16]
    for row in rows:
        row[1] = a
        row[2] = b
    return rows


def test_penalty_moves_the_argmax():
    """The whole point of the feature: a token the server already used is penalized,
    so at its second occurrence the reference prefers something else and the served
    token scores a nonzero gap."""
    torch = pytest.importorskip("torch")
    from hls.core.decode import Greedy

    rows = _penalty_rows()
    eng = _stub_engine(rows, decode=Greedy(frequency_penalty=0.1))
    out = eng._teacher_force_prepared([([{"role": "user", "content": "q"}], None, False, [1, 1], False)])

    # Position 0: no history, so the raw argmax (id 1) is what the server sent.
    assert out[0][0] == pytest.approx(0.0, abs=1e-6)
    # Position 1: id 1 has been served once, so its logit drops to 4.9 and id 2 at
    # 4.95 becomes the reference's choice. The expected gap is read off the same
    # penalized row rather than hardcoded.
    penalized = torch.tensor(rows[1], dtype=torch.float32)
    penalized[1] -= 0.1
    lp = torch.log_softmax(penalized, dim=-1)
    assert out[0][1] == pytest.approx((lp[2] - lp[1]).item(), abs=1e-6)
    assert out[0][1] > 0.0


def test_penalty_counts_only_this_reply():
    """Counts are per request: the second item starts from an empty history, so its
    first position scores 0 exactly as the first item's did."""
    from hls.core.decode import Greedy

    eng = _stub_engine(_penalty_rows(), decode=Greedy(frequency_penalty=0.1))
    item = ([{"role": "user", "content": "q"}], None, False, [1, 1], False)
    out = eng._teacher_force_prepared([item, item])

    assert out[0][0] == pytest.approx(0.0, abs=1e-6)
    assert out[1][0] == pytest.approx(0.0, abs=1e-6), "the second reply's count restarts"
    assert out[0] == pytest.approx(out[1], abs=1e-6)


def test_stop_row_is_scored_on_full_reply_counts():
    """The stop row is one further position, so it sees the counts of the whole reply.
    Here id 1 was served twice and leads the end-of-turn id by less than two units of
    penalty, so the penalized argmax at the stop row is the end-of-turn id and the
    server was entitled to stop."""
    from hls.core.decode import Greedy

    rows = [[0.0] * 16, [0.0] * 16, [0.0] * 16]
    rows[0][1] = 5.0     # served token 1 is the argmax -> gap 0
    rows[1][1] = 5.0     # served again; one unit of penalty still leaves it ahead
    rows[1][2] = 4.5
    rows[2][1] = 5.0     # stop row: raw argmax is id 1 ...
    rows[2][9] = 4.9     # ... by 0.1, less than the 2 * 0.1 the count charges id 1
    eng = _stub_engine(rows, decode=Greedy(frequency_penalty=0.1))
    out = eng._teacher_force_prepared([([{"role": "user", "content": "q"}], None, False, [1, 1], True)])

    assert len(out[0]) == 3, "two served tokens and the stop"
    assert out[0][:2] == pytest.approx([0.0, 0.0], abs=1e-6), "the reply is faithful"
    assert out[0][2] == pytest.approx(0.0, abs=1e-6), (
        "at the stop row id 1 carries a count of 2, so the end-of-turn id wins")


def test_zero_penalty_takes_the_plain_greedy_path(monkeypatch):
    """A 0.0 penalty runs the unpenalized code, not the penalty code with a zero in it:
    identical gaps, and none of the penalty tensors built at all."""
    from hls.core.decode import Greedy

    torch = pytest.importorskip("torch")
    rows = [[float(i == j) for i in range(16)] for j in range(6)]
    prepared = [([{"role": "user", "content": "q"}], None, False, [1, 2, 3, 4, 5], True)]

    plain = _stub_engine(rows)._teacher_force_prepared(prepared)
    explicit_zero = _stub_engine(
        rows, decode=Greedy(frequency_penalty=0.0))._teacher_force_prepared(prepared)
    assert plain == explicit_zero, "bit-identical, not merely close"

    # torch.unique is reached only on the penalty branch, so a plain-greedy engine
    # that scores successfully with it removed proves nothing new runs.
    def _refuse(*args, **kwargs):
        raise AssertionError("plain greedy must not build the penalty count matrix")

    monkeypatch.setattr(torch, "unique", _refuse)
    assert _stub_engine(rows)._teacher_force_prepared(prepared) == plain


# ---------------------------------------------------------------------------
# frequency penalty: the generate-side logits processor
# ---------------------------------------------------------------------------


def _penalty_processor(penalty, prompt_len):
    from hls.reference.engines import _frequency_penalty_processor
    return _frequency_penalty_processor(penalty, prompt_len)


def test_processor_ignores_the_prompt():
    """Only the tokens generated so far are counted, so a token that appears in the
    prompt (or in its left padding) is not penalized."""
    torch = pytest.importorskip("torch")
    proc = _penalty_processor(0.1, prompt_len=2)
    # Ids 3 and 4 are the prompt; nothing has been generated yet.
    input_ids = torch.tensor([[3, 4]])
    scores = torch.zeros(1, 8)
    assert torch.equal(proc(input_ids, scores), torch.zeros(1, 8))


def test_processor_charges_once_per_occurrence():
    """The penalty is additive in the count, so a token generated twice is charged
    twice and a token generated once is charged once."""
    torch = pytest.importorskip("torch")
    proc = _penalty_processor(0.1, prompt_len=1)
    input_ids = torch.tensor([[0, 5, 5, 6]])   # generated: 5, 5, 6
    out = proc(input_ids, torch.zeros(1, 8))
    assert out[0, 5].item() == pytest.approx(-0.2, abs=1e-6)
    assert out[0, 6].item() == pytest.approx(-0.1, abs=1e-6)


def test_processor_leaves_unseen_tokens_bitwise_untouched():
    """A token with count 0 must come out of the subtraction unchanged, bit for bit:
    that is what makes the penalized reference agree with a serving stack, which
    likewise subtracts penalty times zero."""
    torch = pytest.importorskip("torch")
    proc = _penalty_processor(0.1, prompt_len=1)
    scores = torch.tensor([[0.1, -2.5, 3.75, 0.0, 1e-8, -0.3, 7.0, 42.0]])
    out = proc(torch.tensor([[0, 5]]), scores.clone())
    untouched = [i for i in range(8) if i != 5]
    assert torch.equal(out[0, untouched], scores[0, untouched])


# ---------------------------------------------------------------------------
# reference_logprobs: the reference distribution at the ids a report names
# ---------------------------------------------------------------------------


def _report_item(tokens=(1, 2), report_ids=((1, 2), (2, 3)), max_tokens=64):
    """One reference_logprobs item: the served continuation and, per position, the ids
    the reply's own report named."""
    return Envelope(
        request_id="r",
        metadata={"messages": [{"role": "user", "content": "q"}],
                  "max_tokens": max_tokens,
                  "report_ids": [list(r) for r in report_ids]},
        contents=[TextPayload(text="x", tokens=list(tokens))])


def test_reference_logprobs_empty_returns_empty():
    eng = HFReference(decode=Sample(temperature=1.0)).build_engine(_task())
    assert asyncio.run(eng.reference_logprobs([])) == []


def test_reference_logprobs_requires_messages():
    """Validated before the model loads, as teacher_force validates its own, so the
    refusal runs on a torch-free host."""
    eng = HFReference(decode=Sample(temperature=1.0)).build_engine(_task())
    item = Envelope(request_id="r", metadata={"report_ids": [[1]], "max_tokens": 8},
                    contents=[TextPayload(text="x", tokens=[1])])
    with pytest.raises(ValueError, match="messages"):
        asyncio.run(eng.reference_logprobs([item]))


def test_reference_logprobs_requires_token_ids():
    eng = HFReference(decode=Sample(temperature=1.0)).build_engine(_task())
    item = Envelope(request_id="r",
                    metadata={"messages": [{"role": "user", "content": "q"}],
                              "max_tokens": 8, "report_ids": [[1]]},
                    contents=[TextPayload(text="x")])
    with pytest.raises(TypeError, match="token ids"):
        asyncio.run(eng.reference_logprobs([item]))


def test_reference_logprobs_requires_one_report_row_per_served_token():
    """The reply is scored position by position, so a report that does not line up with
    the emission cannot be read at all."""
    eng = HFReference(decode=Sample(temperature=1.0)).build_engine(_task())
    item = _report_item(tokens=(1, 2, 3), report_ids=((1, 2), (2, 3)))
    with pytest.raises(ValueError, match="2 rows of metadata"):
        asyncio.run(eng.reference_logprobs([item]))


def test_reference_logprobs_refuses_a_penalized_decode():
    """vLLM reports the raw distribution and SGLang the penalized one, so under a
    penalty there is no single distribution a report can be compared against."""
    eng = HFReference(decode=Greedy(frequency_penalty=0.1)).build_engine(_task())
    with pytest.raises(NotImplementedError, match="frequency_penalty"):
        asyncio.run(eng.reference_logprobs([_report_item()]))


def test_reference_logprobs_gathers_the_named_ids_and_the_tail():
    """The numbers themselves: the log-softmax of the stub's rows at the named ids, and
    the log of everything else, which is what the divergence is computed against."""
    torch = pytest.importorskip("torch")
    rows = [[0.0] * 16, [0.0] * 16]
    rows[0][1] = 2.0
    rows[1][2] = 1.0
    eng = _stub_engine(rows, decode=Sample(temperature=1.0))
    out = eng._reference_logprobs_prepared(
        [([{"role": "user", "content": "q"}], None, False, [1, 2], [[1, 3], [2, 5]], False)], 1.0)[0]
    lp = torch.log_softmax(torch.tensor(rows, dtype=torch.float32), dim=-1).double()
    assert out["n_ctx"] == 1
    assert out["p_on_q"][0] == pytest.approx(
        [lp[0, 1].item(), lp[0, 3].item()], abs=1e-6)
    assert out["p_on_q"][1] == pytest.approx(
        [lp[1, 2].item(), lp[1, 5].item()], abs=1e-6)
    kept = torch.tensor([lp[0, 1], lp[0, 3]])
    tail = torch.logsumexp(
        lp[0].masked_fill(torch.tensor([i in (1, 3) for i in range(16)]),
                          float("-inf")), dim=0)
    assert out["p_tail_lp"][0] == pytest.approx(tail.item(), abs=1e-6)
    # The tail plus the kept mass is the whole row, which is what makes the partition a
    # distribution.
    assert torch.logsumexp(torch.cat([kept, tail[None]]), dim=0).item() == pytest.approx(
        0.0, abs=1e-6)


def test_reference_logprobs_pads_carry_no_mass_and_leave_the_tail_alone():
    """A padded slot gathers minus infinity, and it must not remove the real id 0 from
    the tail: every pad clamps to column 0 under a gather."""
    torch = pytest.importorskip("torch")
    rows = [[0.0] * 16]
    rows[0][0] = 3.0
    eng = _stub_engine(rows, decode=Sample(temperature=1.0))
    out = eng._reference_logprobs_prepared(
        [([{"role": "user", "content": "q"}], None, False, [5], [[5, -1, -1]], False)], 1.0)[0]
    lp = torch.log_softmax(torch.tensor(rows, dtype=torch.float32), dim=-1).double()
    assert out["p_on_q"][0][1:] == [float("-inf"), float("-inf")]
    kept = lp[0, 5]
    tail = torch.logsumexp(
        lp[0].masked_fill(torch.tensor([i == 5 for i in range(16)]), float("-inf")),
        dim=0)
    assert out["p_tail_lp"][0] == pytest.approx(tail.item(), abs=1e-6)
    assert torch.logsumexp(torch.stack([kept, tail]), dim=0).item() == pytest.approx(
        0.0, abs=1e-6)


def test_reference_logprobs_applies_the_decode_temperature():
    """The distribution a faithful server sampled from is the softmax at its own
    temperature, so that is the one the reference reports."""
    torch = pytest.importorskip("torch")
    rows = [[0.0] * 16]
    rows[0][1] = 2.0
    eng = _stub_engine(rows, decode=Sample(temperature=2.0))
    out = eng._reference_logprobs_prepared(
        [([{"role": "user", "content": "q"}], None, False, [1], [[1]], False)], 2.0)[0]
    want = torch.log_softmax(torch.tensor(rows, dtype=torch.float32) / 2.0, dim=-1)
    assert out["p_on_q"][0][0] == pytest.approx(want[0, 1].item(), abs=1e-6)


def test_reference_logprobs_closes_a_reply_with_an_out_of_range_id():
    """Range-checked on the host: an id outside the embedding table would fire a
    device-side assert that poisons the CUDA context for the rest of the run."""
    pytest.importorskip("torch")
    rows = [[0.0] * 16]
    eng = _stub_engine(rows, decode=Sample(temperature=1.0))
    served = eng._reference_logprobs_prepared(
        [([{"role": "user", "content": "q"}], None, False, [99], [[1]], False)], 1.0)[0]
    assert "served token id 99" in served["invalid"]
    reported = eng._reference_logprobs_prepared(
        [([{"role": "user", "content": "q"}], None, False, [1], [[99]], False)], 1.0)[0]
    assert "position 0 reports token id 99" in reported["invalid"]


def test_reference_logprobs_prices_ending_the_turn_at_every_position():
    """What the end-of-turn count is read against: the mass the model puts on ending
    its turn, over the whole vocabulary and summed across every end-of-turn id the
    checkpoint declares -- not the report's view of it, which is what makes the count
    the one check a report cannot influence."""
    torch = pytest.importorskip("torch")
    rows = [[0.0] * 16, [0.0] * 16]
    rows[0][9] = 4.0   # the model wants to end the turn here
    rows[1][2] = 4.0   # and to continue here
    eng = _stub_engine(rows, eos_ids=(9, 11), decode=Sample(temperature=1.0))
    out = eng._reference_logprobs_prepared(
        [([{"role": "user", "content": "q"}], None, False, [1, 2], [[1], [2]], False)], 1.0)[0]
    lp = torch.log_softmax(torch.tensor(rows, dtype=torch.float32), dim=-1).double()
    want = [torch.logsumexp(lp[i, [9, 11]], dim=0).item() for i in range(2)]
    assert out["p_eot_lp"] == pytest.approx(want, abs=1e-6)
    ending, continuing = (float(torch.tensor(v).exp()) for v in out["p_eot_lp"])
    assert ending > 0.5 > continuing


def test_reference_logprobs_reports_the_checkpoints_ending_facts():
    """The end-of-turn ids and the context window come from the checkpoint, which is the
    only side that holds it; a caller enforcing an ending rule would otherwise carry
    them as numbers nothing keeps true."""
    pytest.importorskip("torch")
    rows = [[0.0] * 16]
    eng = _stub_engine(rows, eos_ids=(9, 11), decode=Sample(temperature=1.0))
    out = eng._reference_logprobs_prepared(
        [([{"role": "user", "content": "q"}], None, False, [1], [[1]], False)], 1.0)[0]
    assert out["eot_ids"] == [9, 11]
    # The stub's config declares no context window, and an unverifiable exemption is
    # reported as absent rather than guessed at.
    assert out["max_context"] is None
