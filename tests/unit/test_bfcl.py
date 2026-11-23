"""Unit tests for BFCLAstMatch (BFCL v3's pure-Python AST-category grader)
and BFCLDataset (the gorilla-llm/Berkeley-Function-Calling-Leaderboard
adapter): the checker's real matching rules (exact/alternate allowed values,
missing/extra params, wrong function name, unparseable output, case/space
-insensitive strings, order-insensitive parallel calls, optional-param
defaults) and the corpus-row -> DatasetItem rendering (system template +
JSON schemas in metadata["messages"], {"ground_truth", "functions"} ground
truth). No network access -- item rendering is exercised via `items_at` on
fixture rows, mirroring tests/unit/test_mrcr.py."""
from __future__ import annotations

import json

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.metrics.bfcl_ast import BFCLAstMatch, parse_calls


def _event(response_text: str, gt: dict) -> TraceEvent:
    inp = Envelope(request_id="r0", contents=[TextPayload(text="prompt")])
    out = Envelope(request_id="r0", contents=[TextPayload(text=response_text)])
    return TraceEvent(request_id="r0", input=inp, response=ArtifactResponse(output=out),
                      ground_truth=gt, t_send=0.0, t_recv=1.0)


# --- BFCLAstMatch --------------------------------------------------------------

GT = [{"calc.add": {"a": [1, 2], "b": [3]}}]           # allowed-value lists per param
FNS = [{"name": "calc.add",
        "parameters": {"type": "dict", "required": ["a", "b"],
                       "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}]


def _ev(text, gt=GT, fns=FNS):
    return _event(response_text=text, gt={"ground_truth": gt, "functions": fns})


def test_exact_call_passes():
    assert BFCLAstMatch().score_one(_ev("[calc.add(a=1, b=3)]")).score == 1.0


def test_alternate_allowed_value_passes():
    assert BFCLAstMatch().score_one(_ev("[calc.add(a=2, b=3)]")).score == 1.0


def test_wrong_value_fails():
    assert BFCLAstMatch().score_one(_ev("[calc.add(a=9, b=3)]")).score == 0.0


def test_missing_required_param_fails():
    assert BFCLAstMatch().score_one(_ev("[calc.add(a=1)]")).score == 0.0


def test_wrong_function_name_fails():
    assert BFCLAstMatch().score_one(_ev("[calc.sub(a=1, b=3)]")).score == 0.0


def test_unparseable_output_fails_closed():
    assert BFCLAstMatch().score_one(_ev("I would call calc.add with a=1")).score == 0.0


def test_string_match_is_case_and_space_insensitive():
    gt = [{"f": {"city": ["New York", "NYC"]}}]
    fns = [{"name": "f", "parameters": {"type": "dict", "required": ["city"],
                                        "properties": {"city": {"type": "string"}}}}]
    assert BFCLAstMatch().score_one(_ev("[f(city='new york')]", gt, fns)).score == 1.0


def test_parallel_calls_order_insensitive():
    gt = [{"f": {"x": [1]}}, {"f": {"x": [2]}}]
    fns = [{"name": "f", "parameters": {"type": "dict", "required": ["x"],
                                        "properties": {"x": {"type": "integer"}}}}]
    assert BFCLAstMatch().score_one(_ev("[f(x=2), f(x=1)]", gt, fns)).score == 1.0


def test_optional_param_default_allowed():
    # optional params in possible_answer carry "" in their allowed list, meaning "may be omitted"
    gt = [{"f": {"x": [1], "verbose": ["", True]}}]
    fns = [{"name": "f", "parameters": {"type": "dict", "required": ["x"],
                                        "properties": {"x": {"type": "integer"},
                                                       "verbose": {"type": "boolean"}}}}]
    assert BFCLAstMatch().score_one(_ev("[f(x=1)]", gt, fns)).score == 1.0


def test_optional_param_may_also_be_emitted_empty():
    """BFCL's "" sentinel says the parameter may be OMITTED or supplied as the
    empty/default value, so a reply that emits it empty is right, not wrong."""
    gt = [{"f": {"x": [1], "unit": ["", "celsius"]}}]
    fns = [{"name": "f", "parameters": {"type": "dict", "required": ["x"],
                                        "properties": {"x": {"type": "integer"},
                                                       "unit": {"type": "string"}}}}]
    assert BFCLAstMatch().score_one(_ev("[f(x=1, unit='')]", gt, fns)).score == 1.0
    assert BFCLAstMatch().score_one(_ev("[f(x=1, unit='celsius')]", gt, fns)).score == 1.0
    # A value the gold list does not permit is still wrong.
    assert BFCLAstMatch().score_one(_ev("[f(x=1, unit='kelvin')]", gt, fns)).score == 0.0


def test_parse_calls_handles_dotted_and_kwargs():
    assert parse_calls('[f(a=1), m.g(b="x")]') == [("f", {"a": 1}), ("m.g", {"b": "x"})]


def test_parse_calls_unparseable_returns_none():
    assert parse_calls("not a call list") is None


# --- BFCLDataset -----------------------------------------------------------------

def _rows():
    return [{"id": "simple_0", "category": "simple",
             "question": "What is the area of a triangle with base 10 and height 5?",
             "functions": [{"name": "calc.area", "parameters": {"type": "dict"}}],
             "ground_truth": [{"calc.area": {"base": [10], "height": [5]}}]}]


def _write(tmp_path, rows):
    d = tmp_path / "test"
    d.mkdir()
    (d / "rows.json").write_text(json.dumps(rows))
    return tmp_path


# GOLDEN full system-message text for _rows()[0]["functions"]. This is NOT
# derived from `hls.spec.datasets.BFCL_SYSTEM_TEMPLATE` -- it's transcribed
# independently from replicating gorilla's OWN live composition functions
# (`formulate_system_prompt` + `format_function_doc`, pinned sha
# 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8) against this same fixture function
# list, so it fails loud if the vendored template ever drifts from gorilla's
# real prompting-mode rendering (persona+task with NO space, the single
# "params...)].  You SHOULD NOT" sentence with a double space, lowercase
# "json format", and json.dumps(..., indent=4) embedded mid-template).
_GORILLA_GOLDEN_SYSTEM_PROMPT = (
    "You are an expert in composing functions."
    "You are given a question and a set of possible functions. Based on the question, you will "
    "need to make one or more function/tool calls to achieve the purpose. If none of the "
    "functions can be used, point it out. If the given question lacks the parameters required "
    "by the function, also point it out.\n\n"
    "You should only return the function calls in your response.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of "
    "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]."
    "  You SHOULD NOT include any other text in the response.\n\n"
    "At each turn, you should try your best to complete the tasks requested by the user "
    "within the current turn. Continue to output functions to call until you have fulfilled "
    "the user's request to the best of your ability. Once you have no more functions to call, "
    "the system will consider the current turn complete and proceed to the next turn "
    "or task.\n\n"
    "Here is a list of functions in json format that you can invoke.\n"
    '[\n'
    '    {\n'
    '        "name": "calc.area",\n'
    '        "parameters": {\n'
    '            "type": "dict"\n'
    '        }\n'
    '    }\n'
    ']\n'
)


#: A revision to construct the dataset with. The field has no default -- the pin is
#: part of a corpus's identity, so a task states it -- and these tests read rows off
#: disk, so any value serves.
_REV = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"


def test_bfcl_item_renders_system_and_user_messages(tmp_path):
    from hls.spec.datasets import BFCLDataset, Split
    ds = BFCLDataset(revision=_REV)
    items = list(ds.items_at(_write(tmp_path, _rows()), Split.TEST))
    assert len(items) == 1 and len(items[0].turns) == 1
    env = items[0].turns[0]
    messages = env.metadata["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == _rows()[0]["question"]
    assert messages[0]["content"] == _GORILLA_GOLDEN_SYSTEM_PROMPT
    assert env.first_of("text").text == _rows()[0]["question"]


def test_bfcl_ground_truth_shape(tmp_path):
    from hls.spec.datasets import BFCLDataset, Split
    ds = BFCLDataset(revision=_REV)
    items = list(ds.items_at(_write(tmp_path, _rows()), Split.TEST))
    gt = items[0].ground_truth_turns[0]
    assert gt == {"ground_truth": _rows()[0]["ground_truth"], "functions": _rows()[0]["functions"]}


def test_bfcl_tolerates_json_encoded_functions_and_ground_truth(tmp_path):
    # corpus() JSON-encodes these fields (Arrow can't infer a shared struct
    # schema across BFCL's heterogeneous per-function-parameter dicts); rows.json
    # written by real provisioning therefore carries them as JSON strings.
    from hls.spec.datasets import BFCLDataset, Split
    raw = _rows()[0]
    encoded = [{**raw, "functions": json.dumps(raw["functions"]),
               "ground_truth": json.dumps(raw["ground_truth"])}]
    ds = BFCLDataset(revision=_REV)
    items = list(ds.items_at(_write(tmp_path, encoded), Split.TEST))
    assert items[0].ground_truth_turns[0] == {"ground_truth": raw["ground_truth"],
                                              "functions": raw["functions"]}


def test_unmounted_split_fails_loud(tmp_path):
    import pytest
    from hls.spec.datasets import BFCLDataset, Split
    ds = BFCLDataset(revision=_REV)
    with pytest.raises(FileNotFoundError, match="(?i)test split not mounted"):
        list(ds.items_at(tmp_path, Split.TEST))


def test_corpus_id_includes_repo_revision_and_categories():
    from hls.spec.datasets import BFCLDataset
    ds = BFCLDataset(revision=_REV)
    cid = ds.corpus_id()
    assert "gorilla-llm/Berkeley-Function-Calling-Leaderboard" in cid
    assert _REV in cid
    assert "live_parallel_multiple" in cid
