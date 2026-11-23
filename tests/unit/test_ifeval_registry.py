"""Unit tests for the IFEval instruction registry checkers that Multi-IF exercises.

The 13 dependency-free checkers plus `language:response_language` (langdetect) are
tested through `IFEvalConstraint` (the real metric path), which also verifies each id
is wired into INSTRUCTION_DICT. Cases follow Google IFEval semantics and the concrete
kwargs seen in facebook/Multi-IF (English subset)."""
from __future__ import annotations

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.metrics.ifeval import INSTRUCTION_DICT, IFEvalConstraint


def _ev(response: str, ids, kwargs):
    inp = Envelope(request_id="r0", contents=[TextPayload(text="prompt")],
                   metadata={"messages": [{"role": "user", "content": "Write something."}]})
    out = Envelope(request_id="r0", contents=[TextPayload(text=response)])
    gt = {"instruction_id_list": list(ids), "kwargs": list(kwargs)}
    return TraceEvent(request_id="r0", input=inp,
                      response=ArtifactResponse(output=out), t_send=0.0, t_recv=1.0,
                      ground_truth=gt)


def _passes(response, iid, kw) -> bool:
    res = IFEvalConstraint().score([_ev(response, [iid], [kw])])
    pv = res.per_event["r0"]
    # a supported+followed instruction must never be reported unsupported
    assert not pv.diagnostics.get("unsupported"), pv.diagnostics
    return pv.passed


# --- every English Multi-IF id must be registered (no fail-closed on coverage) ---

MULTIIF_ENGLISH_IDS = [
    "detectable_format:title", "length_constraints:number_words", "startend:end_checker",
    "keywords:forbidden_words", "startend:quotation", "keywords:frequency",
    "length_constraints:number_sentences", "keywords:existence",
    "length_constraints:number_paragraphs", "punctuation:no_comma",
    "detectable_format:number_highlighted_sections", "combination:two_responses",
    "change_case:english_lowercase", "detectable_content:postscript",
    "change_case:english_capital", "combination:repeat_prompt",
    "change_case:capital_word_frequency", "keywords:letter_frequency",
    "detectable_format:number_bullet_lists", "language:response_language",
    "detectable_format:multiple_sections", "detectable_content:number_placeholders",
    "length_constraints:nth_paragraph_first_word", "detectable_format:json_format",
]


def test_all_multiif_english_ids_registered():
    missing = [i for i in MULTIIF_ENGLISH_IDS if i not in INSTRUCTION_DICT]
    assert missing == [], f"unsupported Multi-IF ids: {missing}"


# --- detectable_format:title -------------------------------------------------

def test_title_present_passes():
    assert _passes("<<A Fine Title>>\nBody text here.", "detectable_format:title", {}) is True


def test_title_absent_fails():
    assert _passes("No title at all here.", "detectable_format:title", {}) is False


def test_title_empty_brackets_fails():
    assert _passes("<<   >> body", "detectable_format:title", {}) is False


def test_title_found_after_an_empty_one():
    """Official TitleChecker scans every <<...>> in the response, so a real title on a
    later line still satisfies the instruction."""
    assert _passes("<<  >>\n<<A Fine Title>>\nBody.", "detectable_format:title",
                   {}) is True


# --- startend:quotation ------------------------------------------------------

def test_quotation_wrapped_passes():
    assert _passes('"the whole thing is quoted"', "startend:quotation", {}) is True


def test_quotation_unwrapped_fails():
    assert _passes("not quoted at all", "startend:quotation", {}) is False


# --- length_constraints:number_paragraphs (*** divider) ----------------------

def test_number_paragraphs_exact_passes():
    body = "first paragraph *** second paragraph"
    assert _passes(body, "length_constraints:number_paragraphs", {"num_paragraphs": 2}) is True


def test_number_paragraphs_wrong_count_fails():
    body = "first paragraph *** second paragraph"
    assert _passes(body, "length_constraints:number_paragraphs", {"num_paragraphs": 3}) is False


def test_number_paragraphs_empty_middle_fails():
    body = "first *** *** third"
    assert _passes(body, "length_constraints:number_paragraphs", {"num_paragraphs": 3}) is False


# --- punctuation:no_comma ----------------------------------------------------

def test_no_comma_clean_passes():
    assert _passes("a sentence without any commas", "punctuation:no_comma", {}) is True


def test_no_comma_with_comma_fails():
    assert _passes("a sentence, with a comma", "punctuation:no_comma", {}) is False


# --- detectable_format:number_highlighted_sections ---------------------------

def test_highlights_enough_passes():
    assert _passes("*one* and *two* here", "detectable_format:number_highlighted_sections",
                   {"num_highlights": 2}) is True


def test_highlights_too_few_fails():
    assert _passes("*only one*", "detectable_format:number_highlighted_sections",
                   {"num_highlights": 3}) is False


def test_highlights_double_asterisk_counts():
    assert _passes("**bold one** and **bold two**", "detectable_format:number_highlighted_sections",
                   {"num_highlights": 2}) is True


# --- combination:two_responses (****** separator) ----------------------------

def test_two_responses_passes():
    assert _passes("first answer ****** second answer", "combination:two_responses", {}) is True


def test_two_responses_single_fails():
    assert _passes("only one answer", "combination:two_responses", {}) is False


def test_two_responses_must_differ():
    """Official TwoResponsesChecker requires the two responses to differ after strip(),
    so repeating one answer twice is not two responses."""
    assert _passes("answer A ****** answer A", "combination:two_responses", {}) is False
    assert _passes("answer A ****** answer B", "combination:two_responses", {}) is True


def test_two_responses_blank_middle_fails_but_blank_edge_passes():
    """A blank segment BETWEEN dividers is a missing response and fails; one at the very
    start or end is just a reply that opens or closes with the divider."""
    assert _passes("answer A ****** ****** answer B",
                   "combination:two_responses", {}) is False
    assert _passes("****** answer A ****** answer B",
                   "combination:two_responses", {}) is True
    assert _passes("answer A ****** answer B ******",
                   "combination:two_responses", {}) is True


# --- detectable_content:postscript -------------------------------------------

def test_postscript_ps_passes():
    assert _passes("Main body.\nP.S. an afterthought", "detectable_content:postscript",
                   {"postscript_marker": "P.S."}) is True


def test_postscript_pps_passes():
    assert _passes("Main body.\nP.P.S this too", "detectable_content:postscript",
                   {"postscript_marker": "P.P.S"}) is True


def test_postscript_absent_fails():
    assert _passes("Main body with no postscript.", "detectable_content:postscript",
                   {"postscript_marker": "P.S."}) is False


# --- combination:repeat_prompt -----------------------------------------------

def test_repeat_prompt_prefix_passes():
    prompt = "Write a haiku about moms."
    assert _passes(prompt + " here it is: ...", "combination:repeat_prompt",
                   {"prompt_to_repeat": prompt}) is True


def test_repeat_prompt_missing_prefix_fails():
    prompt = "Write a haiku about moms."
    assert _passes("I will not repeat.", "combination:repeat_prompt",
                   {"prompt_to_repeat": prompt}) is False


# --- change_case:capital_word_frequency --------------------------------------

def test_capital_word_frequency_at_least_passes():
    assert _passes("THIS IS BIG news", "change_case:capital_word_frequency",
                   {"capital_frequency": 2, "capital_relation": "at least"}) is True


def test_capital_word_frequency_less_than_fails():
    assert _passes("THIS IS BIG news", "change_case:capital_word_frequency",
                   {"capital_frequency": 2, "capital_relation": "less than"}) is False


# --- keywords:letter_frequency -----------------------------------------------

def test_letter_frequency_at_least_passes():
    assert _passes("banana", "keywords:letter_frequency",
                   {"letter": "a", "let_frequency": 3, "let_relation": "at least"}) is True


def test_letter_frequency_less_than_fails():
    assert _passes("banana", "keywords:letter_frequency",
                   {"letter": "a", "let_frequency": 3, "let_relation": "less than"}) is False


# --- language:response_language (langdetect) ---------------------------------

def test_language_english_passes():
    text = "This is clearly an English sentence with several ordinary words."
    assert _passes(text, "language:response_language", {"language": "en"}) is True


def test_language_wrong_target_fails():
    text = "This is clearly an English sentence with several ordinary words."
    assert _passes(text, "language:response_language", {"language": "fr"}) is False


def test_language_kannada_passes():
    # Kannada sample as \u escapes: the repo enforces ASCII-only Python source.
    text = "\u0c86\u0ca6\u0cb0\u0cc6 \u0c87\u0ca6\u0cc1 \u0c92\u0c82\u0ca6\u0cc1 \u0caa\u0cb0\u0cc0\u0c95\u0ccd\u0cb7\u0cc6 \u0c86\u0c97\u0cbf\u0ca6\u0cc6 \u0cae\u0ca4\u0ccd\u0ca4\u0cc1 \u0c87\u0ca6\u0cc1 \u0c95\u0ca8\u0ccd\u0ca8\u0ca1\u0ca6\u0cb2\u0ccd\u0cb2\u0cbf \u0cac\u0cb0\u0cc6\u0ca6\u0cbf\u0ca6\u0cc6"
    assert _passes(text, "language:response_language", {"language": "kn"}) is True


def test_language_empty_fails_closed():
    assert _passes("   ", "language:response_language", {"language": "en"}) is False


# --- detectable_format:multiple_sections -------------------------------------

def test_multiple_sections_enough_passes():
    body = "SECTION 1\nintro\nSECTION 2\nbody"
    assert _passes(body, "detectable_format:multiple_sections",
                   {"section_spliter": "SECTION", "num_sections": 2}) is True


def test_multiple_sections_too_few_fails():
    body = "SECTION 1\nonly one section"
    assert _passes(body, "detectable_format:multiple_sections",
                   {"section_spliter": "SECTION", "num_sections": 2}) is False


# --- length_constraints:nth_paragraph_first_word -----------------------------

def test_nth_paragraph_first_word_passes():
    body = "Intro paragraph.\n\nBonding is the theme.\n\nClosing paragraph."
    assert _passes(body, "length_constraints:nth_paragraph_first_word",
                   {"first_word": "bonding", "nth_paragraph": 2, "num_paragraphs": 3}) is True


def test_nth_paragraph_wrong_first_word_fails():
    body = "Intro paragraph.\n\nWrong start here.\n\nClosing paragraph."
    assert _passes(body, "length_constraints:nth_paragraph_first_word",
                   {"first_word": "bonding", "nth_paragraph": 2, "num_paragraphs": 3}) is False


def test_nth_paragraph_wrong_count_fails():
    body = "Intro.\n\nBonding is the theme."
    assert _passes(body, "length_constraints:nth_paragraph_first_word",
                   {"first_word": "bonding", "nth_paragraph": 2, "num_paragraphs": 3}) is False


def test_nth_paragraph_indexes_the_unfiltered_split():
    """Official ParagraphFirstWordCheck counts non-blank paragraphs but INDEXES the raw
    \\n\\n split, so a leading blank segment shifts which paragraph nth selects: here
    paragraph 2 of the split is the INTRO, not the bonding paragraph."""
    body = "\n\nIntro paragraph.\n\nBonding is the theme.\n\nClosing paragraph."
    kw = {"nth_paragraph": 2, "num_paragraphs": 3}
    assert _passes(body, "length_constraints:nth_paragraph_first_word",
                   {**kw, "first_word": "intro"}) is True
    assert _passes(body, "length_constraints:nth_paragraph_first_word",
                   {**kw, "first_word": "bonding"}) is False


def test_nth_paragraph_first_word_truncates_at_punctuation():
    """Official reads the first word only up to its first {. , ? ! ' "} character, so
    "Don't" is the word "don"."""
    body = "Intro paragraph.\n\nDon't stop believing.\n\nClosing paragraph."
    kw = {"nth_paragraph": 2, "num_paragraphs": 3}
    assert _passes(body, "length_constraints:nth_paragraph_first_word",
                   {**kw, "first_word": "don"}) is True
    assert _passes(body, "length_constraints:nth_paragraph_first_word",
                   {**kw, "first_word": "don't"}) is False


# --- detectable_format:json_format -------------------------------------------

def test_json_object_passes():
    assert _passes('{"a": 1, "b": [2, 3]}', "detectable_format:json_format", {}) is True


def test_json_fenced_passes():
    assert _passes('```json\n{"a": 1}\n```', "detectable_format:json_format", {}) is True


def test_json_invalid_fails():
    assert _passes("this is not json at all", "detectable_format:json_format", {}) is False


def test_json_unfencing_matches_official():
    """Official JsonFormat strips the three ```json spellings it knows, a bare ```
    prefix and a ``` suffix, each unconditionally -- so an uppercase fence and a reply
    closed with a fence it never opened both still parse."""
    assert _passes('```JSON\n{"a": 1}\n```', "detectable_format:json_format", {}) is True
    assert _passes('```Json\n{"a": 1}\n```', "detectable_format:json_format", {}) is True
    assert _passes('{"a": 1}\n```', "detectable_format:json_format", {}) is True


# --- cumulative multi-instruction turn (Multi-IF turns stack instructions) ----

def test_cumulative_all_hold_passes():
    ids = ["change_case:english_lowercase", "startend:end_checker"]
    kw = [{}, {"end_phrase": "the end"}]
    res = IFEvalConstraint().score([_ev("all lowercase text the end", ids, kw)])
    assert res.per_event["r0"].passed is True


def test_cumulative_one_violated_fails():
    ids = ["change_case:english_lowercase", "startend:end_checker"]
    kw = [{}, {"end_phrase": "the end"}]
    res = IFEvalConstraint().score([_ev("all lowercase text but wrong ending", ids, kw)])
    assert res.per_event["r0"].passed is False
