"""A reply may only claim it ended for a reason that means it finished.

An artifact whose decode worker dies on CUDA out-of-memory can turn each killed request into
a well-formed SSE stream carrying `finish_reason: "error"`, empty text, an empty `token_ids`
array and a proper `[DONE]`, with status 200 committed before the crash. A harness that
folds any truthy finish_reason records those dead requests as served and counts no failed
request.
"""

import pytest

from hls.core.faults import ArtifactProtocolError
from hls.spec.wire import OpenAIChatProtocol, OpenAICompletionProtocol

PROTOCOLS = [OpenAIChatProtocol, OpenAICompletionProtocol]


def _unary(protocol_cls, finish_reason):
    """One unary reply body in the shape this protocol reads."""
    choice = {"token_ids": [1, 2], "finish_reason": finish_reason}
    if protocol_cls is OpenAIChatProtocol:
        choice["message"] = {"content": "hi"}
    else:
        choice["text"] = "hi"
    return {"choices": [choice]}


@pytest.mark.parametrize("protocol_cls", PROTOCOLS)
@pytest.mark.parametrize(
    "reason",
    # The two that mean a complete reply, then labels this harness does not define.
    # The brief publishes no enumeration, so an unlisted label must not zero a run:
    # only a label that plainly says "did not finish" does.
    ["stop", "length", "content_filter", "tool_calls", "function_call", "eos"],
)
def test_completed_and_unrecognised_labels_decode(protocol_cls, reason):
    env = protocol_cls().decode_response(_unary(protocol_cls, reason), "r1")
    assert env.contents[0].finish_reason == reason


@pytest.mark.parametrize("protocol_cls", PROTOCOLS)
@pytest.mark.parametrize(
    "reason",
    ["error", "errored", "failed", "abort", "aborted",
     "cancel", "canceled", "cancelled", "timeout", "timed_out"],
)
def test_unfinished_reply_faults_on_the_unary_path(protocol_cls, reason):
    with pytest.raises(ArtifactProtocolError, match="did not complete"):
        protocol_cls().decode_response(_unary(protocol_cls, reason), "r1")


@pytest.mark.parametrize("reason", ["ERROR", "Aborted", "timed-out", " error "])
def test_label_matching_is_case_and_separator_insensitive(reason):
    """A server's casing or hyphenation must not decide whether a crash counts."""
    with pytest.raises(ArtifactProtocolError, match="did not complete"):
        OpenAIChatProtocol().decode_response(_unary(OpenAIChatProtocol, reason), "r1")


def test_the_exact_shape_that_scored_as_served():
    """The artifact's own terminal frame: 200, empty payload, `error`, then [DONE].

    It must fault, so the no-failed-request rule
    counts it, rather than decoding into a served reply with zero tokens.
    """
    p = OpenAIChatProtocol()
    chunks = [
        (0.0, {"choices": [{"delta": {"content": "def f("}, "token_ids": [1, 2]}]}),
        (0.1, {"choices": [{"delta": {"content": ""}, "token_ids": [],
                            "finish_reason": "error"}]}),
    ]
    with pytest.raises(ArtifactProtocolError, match="did not complete"):
        p.decode_stream(chunks, "r1")


def test_streaming_completed_reply_still_decodes():
    """The guard must not fault a stream that ended properly."""
    p = OpenAIChatProtocol()
    chunks = [(0.0, {"choices": [{"delta": {"content": "hi"}, "token_ids": [1, 2],
                                  "finish_reason": "stop"}]})]
    assert p.decode_stream(chunks, "r1").output.first_of("text").tokens == [1, 2]


def test_non_string_finish_reason_faults():
    with pytest.raises(ArtifactProtocolError, match="not a string"):
        OpenAIChatProtocol().decode_response(_unary(OpenAIChatProtocol, 7), "r1")


def test_missing_finish_reason_still_reports_as_missing():
    """The pre-existing message for an absent reason is unchanged."""
    body = {"choices": [{"message": {"content": "hi"}, "token_ids": [1]}]}
    with pytest.raises(ArtifactProtocolError, match="missing finish_reason"):
        OpenAIChatProtocol().decode_response(body, "r1")
