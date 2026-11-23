"""Parse actions & format observations with toolcalls

Forked from mini-SWE-agent ``src/minisweagent/models/utils/actions_toolcall.py`` at
``04d809ceab9df28f9adaed044884180159172930``. MIT licensed, Copyright (c) 2025
Kilian A. Lieret and Carlos E. Jimenez; see ``LICENSE`` beside this package.

Divergences from upstream
-------------------------

**The calls are parsed out of the reply text, not read off a ``tool_calls`` field.**
Upstream talks to an OpenAI-compatible client that hands it structured tool calls.
Here the reply arrives through the harness's wire protocol, and two things follow
from that.

The first is the trust model. What this benchmark grades is the token stream the
artifact served, checked position by position against an oracle. A ``tool_calls``
field is a server-side *parse* of those tokens, produced by whichever tool-call parser
the contestant configured, and nothing checks it against the tokens. Reading the call
out of the reply text keeps the loop reading the model rather than the server's
account of the model.

The second is that it is the only reading that is the same on every server. A stack
run with a tool-call parser strips the call out of ``content`` and reports it
structurally; a stack run without one leaves it in ``content`` verbatim. Two honest
contestants would then drive different trajectories from the same tokens. Parsing the
text is what both of them can be held to.

**Templates are rendered with ``str.format``, not jinja2.** Upstream renders every
template through ``jinja2.Template(..., undefined=StrictUndefined)``. Nothing here has
a placeholder ``str.format`` cannot fill, and jinja2 is a real dependency to carry
into a build sandbox; what ``StrictUndefined`` buys is catching a typo'd placeholder,
which a test that renders each template buys more cheaply. Upstream's two templates
that branch in jinja are two templates each here, selected in Python.

**No multimodal expansion.** Upstream's ``format_toolcall_observation_messages`` takes
a ``multimodal_regex`` and calls ``expand_multimodal_content``. Command output is text
here, and the task declares ``output_modalities={"text"}``, so the parameter and its
import are dropped rather than carried as an argument nothing can set.

Conformance
-----------

vLLM ships an independent implementation of this same grammar
(``vllm/parser/glm47_moe.py``). Its argument regex is character-for-character the one
below, and :func:`parse_toolcall_text` reads a call's name by vLLM's rule.
"""

import json
import re
from dataclasses import dataclass

from ...exceptions import FormatError

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

#: One call as the model writes it. This checkpoint's chat template documents the
#: format in its own system preamble::
#:
#:     <tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>
#:
#: Non-greedy, so a reply carrying two calls yields two rather than one span running
#: from the first open to the last close. Two calls in a reply is the common case on
#: this checkpoint rather than the exception.
#:
#: An unterminated ``<tool_call>`` matches nothing at all, which is the right reading:
#: the reply hit its output cap partway through writing the call, and half a command
#: is not an action.
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_ARG_KEY_OPEN = "<arg_key>"

#: One argument of a call, character-for-character vLLM's own ``_ARG_RE`` for this
#: grammar. The ``\s*`` between the two tags is theirs and is kept, because an
#: independent implementation of the same format accepts it.
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                     re.DOTALL)


@dataclass
class ToolCallFunction:
    """The ``function`` member of an OpenAI tool call: a name and a JSON argument
    string."""

    name: str
    arguments: str


@dataclass
class ToolCall:
    """One tool call in the shape ``parse_toolcall_actions`` expects.

    Upstream is handed these by its OpenAI client. Here they are built from the
    model's own text by :func:`parse_toolcall_text`, so that upstream's parser can be
    carried across unchanged."""

    id: str
    function: ToolCallFunction


def parse_toolcall_text(text: str, *, call_id_prefix: str = "call") -> list[ToolCall]:
    """The calls a reply's visible text carries, in the order it wrote them.

    Not upstream: upstream never sees raw text. The module docstring says why this
    exists. Pass the *visible* text -- a call the model wrote while reasoning is not an
    action it asked for, and the caller strips the reasoning channel first.

    Arguments come back as a JSON string because that is how OpenAI spells them and
    how ``parse_toolcall_actions`` reads them."""
    calls = []
    for idx, body in enumerate(_TOOL_CALL_RE.findall(text)):
        # Keys stripped, values not: vLLM's parser for this grammar does exactly this
        # (``params[match.group("key").strip()] = match.group("value")``), and a
        # command's own leading or trailing whitespace is the model's to choose.
        args = {k.strip(): v for k, v in _ARG_RE.findall(body)}
        # The name runs from the opening delimiter to the first `<arg_key>`, or is the
        # whole body for a call that takes none -- vLLM's rule, read off its state
        # machine. Splitting on the full argument pattern instead looks equivalent and
        # is not: a stray `<arg_key>` with no `<arg_value>` after it would land inside
        # the name, and the turn would be refused as an unknown tool rather than as a
        # call missing its argument.
        cut = body.find(_ARG_KEY_OPEN)
        name = (body if cut < 0 else body[:cut]).strip()
        calls.append(ToolCall(id=f"{call_id_prefix}_{idx}",
                              function=ToolCallFunction(name=name,
                                                        arguments=json.dumps(args))))
    return calls


def has_toolcall_delimiter(text: str) -> bool:
    """Whether the reply tried to write a call, whatever came of it.

    Not upstream, and the reason it exists is diagnostic rather than control flow. A
    reply that parses to no action is two different failures wearing one face: the
    model wrote prose and never acted, or it wrote something call-shaped that this
    parser could not read. The first is the model's, the second is ours. Reporting
    both as the model's is how a parser bug hides as a model failure. Recorded per
    turn so a run with a non-zero count says, without anyone re-reading raw
    completions, that the parser needs looking at."""
    return "<tool_call>" in text


def strip_toolcalls(text: str) -> str:
    """``text`` with the call spans removed: the prose the model wrote around them."""
    return _TOOL_CALL_RE.sub("", text).strip()


def parse_toolcall_actions(
    tool_calls: list, *, format_error_template: str, template_kwargs: dict | None = None
) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args.

    ``template_kwargs`` are extra variables exposed to ``format_error_template`` (e.g.
    ``{"finish_reason": ...}`` so a template can distinguish a real format mistake from a
    ``max_tokens`` truncation).
    """
    template_kwargs = template_kwargs or {}
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": format_error_template.format(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    **template_kwargs,
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        if tool_call.function.name != "bash":
            error_msg += f"Unknown tool '{tool_call.function.name}'."
        if not isinstance(args, dict) or "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": format_error_template.format(
                        error=error_msg.strip(), **template_kwargs
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        actions.append({"command": args["command"], "tool_call_id": tool_call.id})
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    render_observation,
) -> list[dict]:
    """Format execution outputs into tool result messages.

    ``render_observation(output) -> str`` stands in for upstream's
    ``observation_template``: the template branches on output length, and the branch
    lives in Python here rather than in jinja."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        msg = {
            "content": render_observation(output),
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        results.append(msg)
    return results
