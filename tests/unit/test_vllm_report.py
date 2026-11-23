"""The vLLM report capture: its reduction, its accumulator, and its installation check.

The reduction is checked against the definition rather than against the HF engine,
which needs a model: what is pinned here is that this side computes what the gate's
algebra says it should (``test_report_reduction_agrees`` holds the two reductions equal).

The installation check gets more attention than the rest, because a monkeypatch against
a moving library is the one thing a unit test cannot really cover: the fake runner below
is by construction the shape the capture expects. What the tests can pin is that the
check *refuses* a runner of the wrong shape -- a runner with no prompt-logprob worker,
one whose worker comes from somewhere else, one whose reduction takes different
arguments, and one where the wrap does not reach what the live instance resolves. Those
are the ways a vLLM upgrade breaks this, and a check that passed them anyway would be
the silent no-op this path had before.
"""
from __future__ import annotations

import math
import types

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from hls.reference import vllm_report  # noqa: E402
from hls.reference.vllm_report import (  # noqa: E402
    REPORT_DECIMALS, ReportCapture, ReportWorker, install, reduce_rows,
    reduction_class,
)


def log_softmax(rows):
    return torch.log_softmax(torch.tensor(rows, dtype=torch.float32), dim=-1)


# --- the reduction ---------------------------------------------------------

def test_p_on_q_is_the_reference_logprob_at_each_named_id():
    scores = log_softmax([[0.0, 1.0, 2.0, 3.0]])
    ids = torch.tensor([[3, 1]])
    got = reduce_rows(scores, ids, torch.tensor([0]))
    assert got["p_on_q"][0][0] == pytest.approx(scores[0, 3].item(), abs=1e-6)
    assert got["p_on_q"][0][1] == pytest.approx(scores[0, 1].item(), abs=1e-6)


def test_the_tail_is_the_mass_outside_the_named_ids():
    scores = log_softmax([[0.0, 1.0, 2.0, 3.0]])
    ids = torch.tensor([[3, 2]])
    got = reduce_rows(scores, ids, torch.tensor([0]))
    # Everything not named: ids 0 and 1.
    expected = torch.logsumexp(scores[0, [0, 1]], dim=0).item()
    assert got["p_tail_lp"][0] == pytest.approx(expected, abs=1e-6)


def test_the_tail_and_the_named_mass_together_are_the_whole_distribution():
    scores = log_softmax([[0.5, -1.0, 2.0, 0.25, 3.0]])
    ids = torch.tensor([[4, 2]])
    got = reduce_rows(scores, ids, torch.tensor([0]))
    total = sum(math.exp(v) for v in got["p_on_q"][0]) + math.exp(got["p_tail_lp"][0])
    assert total == pytest.approx(1.0, abs=1e-5)


def test_a_pad_slot_carries_no_mass_and_does_not_steal_id_zero():
    # Pad is -1 and clamps to column 0 under a gather; a naive scatter of the flags
    # would then mark real id 0 as reported and wrongly drop it from the tail.
    scores = log_softmax([[3.0, 1.0, 0.0, 0.0]])
    ids = torch.tensor([[1, -1]])
    got = reduce_rows(scores, ids, torch.tensor([3]))
    assert got["p_on_q"][0][1] == float("-inf")
    # id 0 holds most of the mass and was never named, so it must be in the tail.
    expected = torch.logsumexp(scores[0, [0, 2, 3]], dim=0).item()
    assert got["p_tail_lp"][0] == pytest.approx(expected, abs=1e-6)


def test_the_end_of_turn_mass_sums_its_ids_in_log_space():
    scores = log_softmax([[0.0, 1.0, 2.0, 3.0]])
    got = reduce_rows(scores, torch.tensor([[0]]), torch.tensor([1, 3]))
    expected = torch.logsumexp(scores[0, [1, 3]], dim=0).item()
    assert got["p_eot_lp"][0] == pytest.approx(expected, abs=1e-6)


def test_the_end_of_turn_mass_is_reported_far_below_float_resolution():
    # Check 4 prices ending the turn at every served position, and at most of them the
    # model is nowhere near stopping. Those values are the point of the statistic.
    scores = log_softmax([[0.0, -60.0]])
    got = reduce_rows(scores, torch.tensor([[0]]), torch.tensor([1]))
    assert got["p_eot_lp"][0] < -50


def test_several_positions_reduce_independently():
    scores = log_softmax([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    ids = torch.tensor([[2], [0]])
    got = reduce_rows(scores, ids, torch.tensor([1]))
    assert len(got["p_on_q"]) == len(got["p_tail_lp"]) == len(got["p_eot_lp"]) == 2
    assert got["p_on_q"][0][0] == pytest.approx(scores[0, 2].item(), abs=1e-6)
    assert got["p_on_q"][1][0] == pytest.approx(scores[1, 0].item(), abs=1e-6)


def test_values_are_rounded_for_the_wire():
    scores = log_softmax([[0.0, 1.0, 2.0, 3.0]])
    got = reduce_rows(scores, torch.tensor([[1]]), torch.tensor([0]))
    for v in (got["p_on_q"][0][0], got["p_tail_lp"][0], got["p_eot_lp"][0]):
        assert round(v, REPORT_DECIMALS) == v


# --- the capture slot ------------------------------------------------------

def rows(n, base=0.0):
    return {"p_on_q": [[base + i] for i in range(n)],
            "p_tail_lp": [base + i for i in range(n)],
            "p_eot_lp": [base + i for i in range(n)]}


def test_chunks_concatenate_in_order():
    cap = ReportCapture()
    cap.arm([[1]] * 5, [0])
    cap.claim("r")
    cap.extend("r", 0, rows(2))
    cap.extend("r", 2, rows(3, base=10.0))
    got = cap.take()
    assert len(got["p_tail_lp"]) == 5
    assert got["p_tail_lp"] == [0.0, 1.0, 10.0, 11.0, 12.0]


def test_a_chunk_arriving_out_of_order_is_refused():
    # The accumulator concatenates; a gap would silently misalign every later position
    # against the reply's tokens.
    cap = ReportCapture()
    cap.arm([[1]] * 5, [0])
    cap.extend("r", 0, rows(2))
    with pytest.raises(RuntimeError, match="out of order"):
        cap.extend("r", 4, rows(1))


def test_taking_clears_the_slot():
    cap = ReportCapture()
    cap.arm([[1]], [0])
    cap.extend("r", 0, rows(1))
    assert cap.take() is not None
    assert cap.take() is None


def test_an_unarmed_capture_takes_nothing():
    assert ReportCapture().take() is None


def test_arming_again_drops_a_previous_partial():
    # A retried reply must not concatenate onto the abandoned attempt's rows.
    cap = ReportCapture()
    cap.arm([[1]] * 2, [0])
    cap.extend("r", 0, rows(1))
    cap.arm([[1]] * 2, [0])
    cap.extend("r", 0, rows(2))
    assert len(cap.take()["p_tail_lp"]) == 2


def test_the_slot_binds_to_the_first_request_that_claims_it():
    cap = ReportCapture()
    cap.arm([[7, 8]], [1, 2])
    assert cap.claim("vllm-minted-id") == {"report_ids": [[7, 8]], "eot_ids": [1, 2]}
    assert cap.claim("vllm-minted-id") is not None       # same request, still bound


def test_a_second_request_cannot_claim_a_bound_slot():
    # The host cannot name the id vLLM will mint, so the slot binds to whichever
    # request arrives. Two at once would concatenate into one reply's rows.
    cap = ReportCapture()
    cap.arm([[1]], [0])
    cap.claim("first")
    with pytest.raises(RuntimeError, match="one reply at a time"):
        cap.claim("second")


def test_an_unarmed_slot_is_claimed_by_nobody():
    assert ReportCapture().claim("anything") is None


def test_a_failure_is_handed_to_the_host_rather_than_raised_on_the_engine_loop():
    cap = ReportCapture()
    cap.arm([[1]], [0])
    cap.fail("ValueError: something")
    assert cap.take() == {"error": "ValueError: something"}
    assert cap.take() is None


def test_a_failed_capture_stops_claiming():
    cap = ReportCapture()
    cap.arm([[1]], [0])
    cap.fail("boom")
    assert cap.claim("r") is None


# --- the installation check ------------------------------------------------

REDUCTION_MODULE = "vllm.v1.worker.gpu.sample.prompt_logprob"


def make_reduction_class(module=REDUCTION_MODULE, params=None):
    """A stand-in for vLLM's prompt-logprob worker, shaped as the capture expects."""
    if params is None:
        def compute_prompt_logprobs(self, logits_fn, hidden_states, input_batch,
                                    all_token_ids, num_computed_tokens, prompt_lens):
            return {}
    else:
        src = f"def compute_prompt_logprobs({', '.join(params)}):\n    return {{}}\n"
        ns: dict = {}
        exec(src, ns)                                     # noqa: S102
        compute_prompt_logprobs = ns["compute_prompt_logprobs"]
    cls = type("PromptLogprobsWorker", (),
               {"compute_prompt_logprobs": compute_prompt_logprobs,
                "uses_prompt_logprobs": np.ones(8, dtype=bool)})
    cls.__module__ = module
    return cls


def make_runner(worker_cls=None):
    worker = worker_cls() if worker_cls is not None else make_reduction_class()()
    return types.SimpleNamespace(prompt_logprobs_worker=worker)


@pytest.fixture(autouse=True)
def _fresh_capture(monkeypatch):
    """Each test gets its own slot; the module-level one is per worker process."""
    monkeypatch.setattr(vllm_report, "CAPTURE", ReportCapture())


def test_a_runner_without_a_prompt_logprob_worker_is_refused():
    # vLLM's older GPU model runner keeps this reduction on itself, where a wrap of the
    # worker's class would be one that nothing calls.
    runner = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="holds no prompt_logprobs_worker"):
        reduction_class(runner)


def test_a_runner_holding_none_is_refused():
    runner = types.SimpleNamespace(prompt_logprobs_worker=None)
    with pytest.raises(RuntimeError, match="holds no prompt_logprobs_worker"):
        reduction_class(runner)


def test_a_reduction_from_another_module_is_refused():
    # The capture's index arithmetic was written against one file. A same-named class
    # from elsewhere is a different reduction.
    runner = make_runner(make_reduction_class(module="somewhere.else"))
    with pytest.raises(RuntimeError, match="not a class from"):
        reduction_class(runner)


def test_a_reduction_missing_the_method_is_refused():
    cls = type("PromptLogprobsWorker", (), {})
    cls.__module__ = REDUCTION_MODULE
    with pytest.raises(RuntimeError, match="no compute_prompt_logprobs"):
        reduction_class(make_runner(cls))


def test_a_reduction_whose_arguments_moved_is_refused():
    # The capture reads logits_fn, hidden_states, input_batch and prompt_lens by
    # position, so a reordered signature would feed it the wrong tensors silently.
    moved = ["self", "hidden_states", "logits_fn", "input_batch", "all_token_ids",
             "num_computed_tokens", "prompt_lens"]
    with pytest.raises(RuntimeError, match="reads its arguments by position"):
        reduction_class(make_runner(make_reduction_class(params=moved)))


def test_a_good_runner_yields_the_class_the_wrap_goes_on():
    cls = make_reduction_class()
    assert reduction_class(make_runner(cls)) is cls


def test_install_wraps_what_the_live_worker_resolves():
    cls = make_reduction_class()
    runner = make_runner(cls)
    name = install(runner)
    assert name.endswith("PromptLogprobsWorker.compute_prompt_logprobs")
    bound = runner.prompt_logprobs_worker.compute_prompt_logprobs
    assert getattr(bound.__func__, "_hls_wrapped", False)


def test_install_refuses_when_the_wrap_does_not_reach_the_live_instance():
    # An instance carrying its own copy of the method never sees a class-level wrap:
    # the patch succeeds and nothing calls it.
    cls = make_reduction_class()
    runner = make_runner(cls)
    runner.prompt_logprobs_worker.compute_prompt_logprobs = (
        lambda *a, **k: {})                                # shadows the class attribute
    with pytest.raises(RuntimeError, match="did not change what the live"):
        install(runner)
    assert not getattr(cls.compute_prompt_logprobs, "_hls_wrapped", False)


def test_install_is_idempotent():
    runner = make_runner()
    first = install(runner)
    assert install(runner) == f"{first} (already wrapped)"


# --- the capture's arithmetic, driven through the wrap ---------------------

VOCAB = 16


def fake_batch(*, req_ids, prompt_len, computed, scheduled):
    """One step of vLLM's batch, for a single request part-way through its prefill."""
    n = len(req_ids)
    query_start = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(scheduled, out=query_start[1:])
    return types.SimpleNamespace(
        req_ids=req_ids,
        idx_mapping_np=np.arange(n, dtype=np.intp),
        num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
        query_start_loc_np=query_start,
        prefill_len_np=np.asarray(prompt_len, dtype=np.int32),
        num_computed_prefill_tokens_np=np.asarray(computed, dtype=np.int32))


def run_step(runner, *, prompt_len, computed, scheduled, req_id="r"):
    """Call the wrapped reduction the way the model runner does.

    Each hidden row carries its own prompt position, and the fake output head records
    the positions it is asked to evaluate, so a misaligned slice is visible as the
    wrong rows rather than only as the wrong count."""
    total = int(scheduled)
    hidden = torch.zeros(total, 4)
    for i in range(total):
        hidden[i, 0] = float(computed + i)
    seen: list[float] = []

    def logits_fn(h):
        seen.extend(h[:, 0].tolist())
        # Same logits at every row, but distinct per id, so a reply scored against a
        # shifted row of reported ids shows up in the values.
        return -torch.arange(VOCAB, dtype=torch.float32).expand(h.shape[0], VOCAB)

    batch = fake_batch(req_ids=[req_id], prompt_len=[prompt_len],
                       computed=[computed], scheduled=[scheduled])
    runner.prompt_logprobs_worker.compute_prompt_logprobs(
        logits_fn, hidden, batch, None, None,
        np.asarray([prompt_len], dtype=np.int32))
    return [int(v) for v in seen]


def armed_runner(reply_len, report_ids=None):
    cls = make_reduction_class()
    runner = make_runner(cls)
    install(runner)
    vllm_report.CAPTURE.arm(report_ids or [[1]] * reply_len, [0])
    return runner


def test_the_capture_reduces_exactly_the_reply_rows():
    # A 100-token prompt ending in a 4-token reply, prefilled in one step. Row r
    # predicts token r+1, so the reply is predicted by rows 95..98.
    runner = armed_runner(4)
    assert run_step(runner, prompt_len=100, computed=0, scheduled=100) == [95, 96, 97, 98]
    assert len(vllm_report.CAPTURE.take()["p_tail_lp"]) == 4


def test_each_reply_row_is_scored_against_its_own_reported_ids():
    # The reply's first row carries the first row of reported ids. An offset here
    # would score every position against a neighbour's ids and still return the right
    # number of rows.
    runner = armed_runner(3, report_ids=[[1], [2], [3]])
    run_step(runner, prompt_len=10, computed=0, scheduled=10)
    lp = torch.log_softmax(-torch.arange(VOCAB, dtype=torch.float32), dim=-1)
    assert vllm_report.CAPTURE.take()["p_on_q"] == [
        [round(lp[j].item(), REPORT_DECIMALS)] for j in (1, 2, 3)]


def test_chunked_prefill_collects_the_reply_across_steps():
    # The reply straddles two chunks; the accumulator concatenates them in order.
    runner = armed_runner(4)
    assert run_step(runner, prompt_len=100, computed=0, scheduled=97) == [95, 96]
    assert run_step(runner, prompt_len=100, computed=97, scheduled=3) == [97, 98]
    assert len(vllm_report.CAPTURE.take()["p_tail_lp"]) == 4


def test_a_chunk_holding_no_reply_row_contributes_nothing():
    runner = armed_runner(4)
    assert run_step(runner, prompt_len=100, computed=0, scheduled=50) == []
    assert vllm_report.CAPTURE.take()["p_tail_lp"] == []


def test_the_row_that_predicts_the_first_generated_token_is_never_reduced():
    # The prompt's last row predicts the token after the reply, where the server
    # reported nothing. With the reply running to the end of the prompt, a window that
    # did not stop at the reply's length would take row 99 as well.
    runner = armed_runner(99)                              # reply is all but token 0
    assert run_step(runner, prompt_len=100, computed=0, scheduled=100)[-1] == 98
    assert len(vllm_report.CAPTURE.take()["p_tail_lp"]) == 99


def test_a_request_that_did_not_ask_for_prompt_logprobs_is_skipped():
    cls = make_reduction_class()
    cls.uses_prompt_logprobs = np.zeros(8, dtype=bool)
    runner = make_runner(cls)
    install(runner)
    vllm_report.CAPTURE.arm([[1]] * 4, [0])
    assert run_step(runner, prompt_len=100, computed=0, scheduled=100) == []
    assert vllm_report.CAPTURE.take()["p_tail_lp"] == []


def test_a_batch_missing_a_field_the_capture_indexes_says_which_and_why():
    # Letting the attribute access raise on its own would name the field but read as
    # a bug in the capture. The reply is failed with the field and the remedy, since
    # a batch that lost a field means the pinned image moved.
    runner = armed_runner(4)
    batch = fake_batch(req_ids=["r"], prompt_len=[100], computed=[0], scheduled=[100])
    del batch.prefill_len_np
    runner.prompt_logprobs_worker.compute_prompt_logprobs(
        lambda h: torch.zeros(h.shape[0], VOCAB), torch.zeros(100, 4), batch,
        None, None, np.asarray([100], dtype=np.int32))
    error = vllm_report.CAPTURE.take()["error"]
    assert "prefill_len_np" in error
    assert "Pin the reference image" in error


def test_the_wrap_still_returns_what_vllm_computed():
    # The capture is a side effect; vLLM's own prompt logprobs must still come back,
    # or the request the host submitted would not complete.
    cls = make_reduction_class()
    cls.compute_prompt_logprobs = (
        lambda self, logits_fn, hidden_states, input_batch, all_token_ids,
        num_computed_tokens, prompt_lens: {"vllm-own": 1})
    runner = make_runner(cls)
    install(runner)
    vllm_report.CAPTURE.arm([[1]] * 4, [0])
    batch = fake_batch(req_ids=["r"], prompt_len=[100], computed=[0], scheduled=[100])
    out = runner.prompt_logprobs_worker.compute_prompt_logprobs(
        lambda h: torch.zeros(h.shape[0], VOCAB), torch.zeros(100, 4), batch,
        None, None, np.asarray([100], dtype=np.int32))
    assert out == {"vllm-own": 1}


# --- the worker mixin ------------------------------------------------------

def test_the_worker_methods_take_the_worker_as_their_first_argument():
    # vLLM calls an rpc target as partial(target, worker). A plain module function
    # taking no worker raises there; a bound method on the worker does not.
    for name in ("hls_install_report", "hls_arm_report", "hls_take_report"):
        params = list(
            __import__("inspect").signature(getattr(ReportWorker, name)).parameters)
        assert params[0] == "self"


def test_the_worker_methods_do_not_collide_with_vllms_own():
    # vLLM asserts the extension class adds no attribute the worker already has.
    assert all(name.startswith("hls_") for name in vars(ReportWorker)
               if not name.startswith("__"))


def test_arming_and_taking_go_through_the_module_capture():
    worker = ReportWorker()
    worker.hls_arm_report([[3]], [9])
    assert vllm_report.CAPTURE.claim("r") == {"report_ids": [[3]], "eot_ids": [9]}
    vllm_report.CAPTURE.extend("r", 0, rows(1))
    assert len(worker.hls_take_report()["p_tail_lp"]) == 1


def test_installing_through_the_worker_reads_its_live_runner():
    worker = ReportWorker()
    worker.model_runner = make_runner()
    assert worker.hls_install_report().endswith("compute_prompt_logprobs")
