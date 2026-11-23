"""GPU integration test for HFEngine.teacher_force (greedy gap forward pass).

Requires torch + the model weights cached; skipped otherwise (mirrors how generate's
HF path is integration-only). Validates the gap semantics end to end: the model's OWN
greedy continuation has all-zero gaps; flipping a token to a non-argmax id makes that
position's gap positive.
"""
import asyncio

import pytest

from hls.core.decode import Greedy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.reference.references import HFReference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.taskspec import TaskSpec
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, Uncapped, Workload)
from hls.verify import LetterMatchMetric, MetricVerifier

pytestmark = pytest.mark.gpu

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _greedy_ids(ref, messages, n):
    """Greedily decode n continuation ids with the loaded reference model, so the
    continuation is by construction the model's argmax at each step."""
    import torch

    rendered = ref._tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    ids = ref._tokenizer(rendered, return_tensors="pt",
                         add_special_tokens=False).to(ref._device)["input_ids"]
    out = []
    for _ in range(n):
        with torch.no_grad():
            logits = ref._model(ids).logits[0, -1]
        nxt = int(logits.argmax().item())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ref._device)], dim=1)
    return out


def _generate_greedy_ids(ref, messages, n):
    """Greedy continuation from ``model.generate`` (the KV-cache decode path),
    matching what artifact servers emit under ``do_sample=False``."""
    import torch

    rendered = ref._tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = ref._tokenizer(rendered, return_tensors="pt",
                            add_special_tokens=False).to(ref._device)
    with torch.no_grad():
        out = ref._model.generate(
            **inputs, max_new_tokens=n,
            do_sample=False, repetition_penalty=1.0,
            pad_token_id=ref._tokenizer.eos_token_id)
    return out[0, inputs["input_ids"].shape[-1]:].tolist()


def _item(messages, tokens):
    return Envelope(request_id="r", metadata={"messages": messages},
                    contents=[TextPayload(text="", tokens=tokens)])


def _task():
    return TaskSpec(
        id="T",
        model=ModelSpec(base_checkpoint=MODEL_ID, arch="TX", chat_format="chatml"),
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


def _reference():
    # The runtime that loads weights + runs teacher_force is the engine; the config's
    # model identity is task.model (HFReference carries no model_spec).
    eng = HFReference(decode=Greedy()).build_engine(_task())
    try:
        eng._load()
    except Exception as exc:           # noqa: BLE001 -- model not cached / no GPU
        pytest.skip(f"model {MODEL_ID} unavailable: {exc}")
    return eng


def test_teacher_force_greedy_continuation_has_zero_gaps():
    pytest.importorskip("torch")
    ref = _reference()
    messages = [{"role": "user", "content": "Reply with one short word."}]
    gids = _greedy_ids(ref, messages, n=3)

    gaps = asyncio.run(ref.teacher_force([_item(messages, gids)]))[0]
    assert len(gaps) == 3
    assert all(g >= 0.0 for g in gaps)
    assert max(gaps) < 1e-6            # greedy ids ARE the argmax -> ~0 gap


def test_teacher_force_non_greedy_token_has_positive_gap():
    pytest.importorskip("torch")
    ref = _reference()
    messages = [{"role": "user", "content": "Reply with one short word."}]
    gids = _greedy_ids(ref, messages, n=3)

    wrong = (gids[0] + 1) % ref._model.config.vocab_size   # a non-argmax id at position 0
    gaps = asyncio.run(ref.teacher_force([_item(messages, [wrong] + gids[1:])]))[0]
    assert gaps[0] > 0.0


def test_teacher_force_out_of_vocab_ids_score_inf_not_crash():
    """Served ids come from the CONTESTANT: an id outside the embedding table must
    score +inf for the whole reply (a token the model can never emit has
    probability 0) WITHOUT reaching the embedding gather -- there it fires a
    device-side assert that sticky-poisons the CUDA context and turns an
    unfaithful artifact into a harness error. The follow-up call proves the
    context survived."""
    pytest.importorskip("torch")
    ref = _reference()
    messages = [{"role": "user", "content": "Reply with one short word."}]
    gids = _greedy_ids(ref, messages, n=3)
    vocab = ref._model.get_input_embeddings().num_embeddings

    for bad in (vocab, -1):
        gaps = asyncio.run(ref.teacher_force([_item(messages, gids[:2] + [bad])]))[0]
        assert gaps == [float("inf")] * 3
    # CUDA context still healthy: a normal teacher-force works right after.
    gaps = asyncio.run(ref.teacher_force([_item(messages, gids)]))[0]
    assert max(gaps) < 1e-6


def test_generate_reply_self_agrees_under_teacher_force():
    """The strongest instrument check: a reply's OWN emitted tokens, teacher-forced
    against the same reference, score all-zero gaps. generate and teacher_force run the
    SAME pinned SDPA-MATH kernel with cuDNN-SDP off, so the reference's greedy argmax at
    each step IS the token it emitted -- any nonzero gap would mean the two forward
    paths diverged (the bf16 fragility the pin exists to remove). Exercises the tokens
    carried on the generate reply end to end."""
    pytest.importorskip("torch")
    ref = _reference()
    messages = [{"role": "user", "content": "Reply with one short word."}]
    prompt = Envelope(request_id="g", contents=[TextPayload(text="")],
                      metadata={"messages": messages, "max_tokens": 8})

    reply = asyncio.run(ref.generate([prompt]))[0]
    tokens = reply.first_of("text").tokens
    assert tokens, "generate reply must carry the emitted token ids"

    gaps = asyncio.run(ref.teacher_force([_item(messages, tokens)]))[0]
    assert len(gaps) == len(tokens)
    # Exact zeros: a served token equal to the reference argmax subtracts a tensor
    # element from itself (bitwise 0.0) under the shared MATH pin.
    assert all(g == 0.0 for g in gaps)


def test_stack_report_populates_stack_info_shape():
    """stack_report is the drift instrument the token gate refuses on; on real hardware
    every StackInfo field must be present and non-empty, with the NVIDIA component
    versions under the vendor-neutral `runtime` map."""
    pytest.importorskip("torch")
    ref = _reference()
    report = asyncio.run(ref.stack_report())
    assert set(report) == {"torch", "runtime", "device_name"}
    assert set(report["runtime"]) == {"cuda", "cudnn"}
    assert report["torch"] and report["device_name"]
    assert all(report["runtime"].values())   # every component version non-empty
