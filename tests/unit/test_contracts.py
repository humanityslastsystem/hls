import pytest
from pydantic import ValidationError

from hls.spec.contracts import RequestResponseContract
from hls.spec.wire import OpenAIChatProtocol


def _c(**kw):
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}), **kw)


def test_defaults_to_openai_protocol_and_http():
    c = _c()
    assert c.transport == "http"
    assert isinstance(c.protocol, OpenAIChatProtocol)


def test_a_request_response_contract_has_no_timestamp_flag():
    """A contract is its path, modalities, transport and protocol; a task that asks for a
    per-token timestamp flag is refused."""
    assert "timestamp_per_token" not in RequestResponseContract.model_fields
    with pytest.raises(ValidationError, match="timestamp_per_token"):
        _c(timestamp_per_token=False)


def test_body_schema_is_openai_shaped():
    schema = _c().body_schema()
    assert "messages" in schema["request"]
    assert "choices" in schema["response"]
    assert "request_id" not in schema["request"]          # NOT HLS Envelope


def test_brief_section_delegates_to_protocol_with_path_and_transport():
    c = _c(transport="http-sse",
           protocol=OpenAIChatProtocol())
    text = c.brief_section()
    assert "POST /v1/chat/completions" in text
    assert "token_ids" in text
    assert "[DONE]" in text                                # streaming surfaced


def test_prompt_fields_is_contract_only():
    f = _c().prompt_fields()
    assert f["path"] == "/v1/chat/completions"
    assert f["transport"] == "http"
    # Exact key set: protocol toggles (logprobs, top_logprobs, ...) live on the
    # protocol, never leak onto the contract's own prompt fields.
    assert set(f.keys()) == {"path", "input_modalities", "output_modalities", "port", "transport"}


def test_contract_has_port_default_8000():
    from hls.spec.contracts import RequestResponseContract
    c = RequestResponseContract(path="/v1/chat/completions",
        input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}))
    assert c.port == 8000
    assert RequestResponseContract(path="/x", input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}), port=9001).port == 9001
