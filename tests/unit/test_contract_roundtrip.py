from hls.core.poly import dump_spec
from hls.spec.contracts import RequestResponseContract
from hls.spec.wire import OpenAIChatProtocol


def test_contract_with_protocol_dumps_nested_kinds():
    c = RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
        protocol=OpenAIChatProtocol(return_logprobs=True, top_logprobs=20))
    rec = dump_spec(c)
    assert rec["kind"] == "RequestResponseContract"
    assert rec["protocol"]["kind"] == "OpenAIChatProtocol"
    assert rec["protocol"]["top_logprobs"] == 20
    assert set(rec["input_modalities"]) == {"text"}          # frozenset -> json list


def test_protocol_methods_still_work():
    p = OpenAIChatProtocol(return_logprobs=True)
    body = p.body_schema()
    assert "request" in body and "response" in body          # behavior preserved
