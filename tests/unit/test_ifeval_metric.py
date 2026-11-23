from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.metrics.ifeval import IFEvalConstraint


def _ev(rid, response, gt):
    inp = Envelope(request_id=rid, contents=[TextPayload(text="prompt")],
                   metadata={"messages": [{"role": "user", "content": "Write something."}]})
    out = Envelope(request_id=rid, contents=[TextPayload(text=response)])
    return TraceEvent(request_id=rid, input=inp,
                      response=ArtifactResponse(output=out), t_send=0.0, t_recv=1.0,
                      ground_truth=gt)


def test_passes_when_all_constraints_hold():
    gt = {"instruction_id_list": ["keywords:existence"],
          "kwargs": [{"keywords": ["banana"]}]}
    ev = _ev("r0", "I really like banana bread a lot.", gt)
    res = IFEvalConstraint().score([ev])
    assert res.per_event["r0"].passed is True


def test_fails_when_a_constraint_is_violated():
    gt = {"instruction_id_list": ["keywords:existence"],
          "kwargs": [{"keywords": ["banana"]}]}
    ev = _ev("r0", "No fruit mentioned here.", gt)
    res = IFEvalConstraint().score([ev])
    assert res.per_event["r0"].passed is False


def test_multiple_constraints_all_must_hold():
    gt = {"instruction_id_list": ["keywords:existence", "length_constraints:number_words"],
          "kwargs": [{"keywords": ["cat"]}, {"num_words": 3, "relation": "at least"}]}
    ev = _ev("r0", "the cat sat quietly", gt)   # has 'cat' and >=3 words
    res = IFEvalConstraint().score([ev])
    assert res.per_event["r0"].passed is True


def test_unsupported_instruction_fails_closed_without_raising():
    gt = {"instruction_id_list": ["totally:unknown_instruction"], "kwargs": [{}]}
    ev = _ev("r0", "anything", gt)
    res = IFEvalConstraint().score([ev])           # must NOT raise
    assert res.per_event["r0"].passed is False
    assert "totally:unknown_instruction" in str(res.per_event["r0"].diagnostics)


def test_ground_truth_accepts_json_string():
    import json
    gt = json.dumps({"instruction_id_list": ["keywords:existence"],
                     "kwargs": [{"keywords": ["banana"]}]})
    ev = _ev("r0", "banana", gt)
    res = IFEvalConstraint().score([ev])
    assert res.per_event["r0"].passed is True
