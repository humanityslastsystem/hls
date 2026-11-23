"""Self-contained IFEval instruction checkers.

Each checker has signature ``(response: str, **kwargs) -> bool`` and returns True
iff the response satisfies that instruction. INSTRUCTION_DICT maps the IFEval
instruction_id to its checker. Coverage spans every instruction type exercised by
facebook/Multi-IF (the English subset); unsupported ids are handled by the metric
(fail closed), not here.

Only ``language:response_language`` pulls a dependency (``langdetect``); every other
checker is pure-Python/regex. Instruction ids and kwargs follow Google's IFEval
(google-research/.../instruction_following_eval).

``langdetect`` is imported lazily, inside that one checker, not at module top:
importing this registry happens transitively whenever ``hls.spec.tenant`` is imported
(``Tenant.verifiers: list[Verifier]`` -> ``hls.verify`` package -> this metric), which
the drive role does when it loads the task. langdetect is needed only where verification
runs; a module-top import would make every importer of a tenant need it. The checker is the
only place it is actually needed.

Fidelity to the official reference (every checker below audited against
instructions.py):
  Exact -- same verdict as the official checker on every input:
    keywords:existence, keywords:forbidden_words, keywords:frequency,
    keywords:letter_frequency, length_constraints:number_words,
    length_constraints:number_paragraphs,
    length_constraints:nth_paragraph_first_word,
    detectable_content:number_placeholders, detectable_content:postscript,
    detectable_format:number_bullet_lists, detectable_format:title,
    detectable_format:number_highlighted_sections,
    detectable_format:multiple_sections, detectable_format:json_format,
    punctuation:no_comma, combination:two_responses, combination:repeat_prompt,
    startend:end_checker, startend:quotation.
    Three of these build their pattern from a kwarg with ``re.escape`` where official
    interpolates it raw (keywords:frequency, detectable_content:postscript,
    detectable_format:multiple_sections): identical for the plain-text kwargs IFEval
    ships, and it avoids an re.error on a metacharacter instead of matching wildly.
  Dependency-free, or stricter than official (noted at each checker):
    - length_constraints:number_sentences splits on [.!?]; official uses nltk punkt.
    - change_case:english_lowercase / english_capital check case alone; official also
      runs langdetect=="en".
    - change_case:capital_word_frequency counts \\w+ runs; official counts nltk
      word_tokenize tokens.
    - language:response_language fails closed on an undetectable response and seeds
      langdetect, so it is deterministic across processes; official counts such a
      response as compliant and leaves langdetect unseeded.
``startend:start_checker`` is an HLS-only extension (no official
counterpart -- the official ConstrainedStartChecker is disabled and keyed differently).
"""
from __future__ import annotations

import json
import re
from typing import Callable


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def _sentences(text: str) -> list[str]:
    # Splits on [.!?], with no dependency. Google's count_sentences uses the nltk punkt
    # tokenizer, which does not split on abbreviations ("Dr."), decimals ("3.5"), or
    # initials. See module docstring.
    return [s for s in (p.strip() for p in re.split(r"[.!?]+", text)) if s]


def _relation_ok(count: int, relation: str | None, target: int) -> bool:
    rel = (relation or "at least").strip()
    if rel == "at least":
        return count >= target
    if rel == "less than":
        return count < target
    if rel == "at most":
        return count <= target
    if rel == "more than":
        return count > target
    if rel == "exactly":
        return count == target
    raise ValueError(f"unknown relation {relation!r}")


def keywords_existence(response: str, keywords=None, **_) -> bool:
    low = response.lower()
    return all(str(k).lower() in low for k in (keywords or []))


def _word_boundary_re(word: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(str(word)) + r"\b", re.IGNORECASE)


def keywords_forbidden(response: str, forbidden_words=None, **_) -> bool:
    return all(_word_boundary_re(w).search(response) is None for w in (forbidden_words or []))


def keywords_frequency(response: str, keyword=None, frequency=None, relation=None, **_) -> bool:
    if keyword is None or frequency is None:
        return False
    # Official KeywordFrequencyChecker counts substring occurrences (re.findall on the
    # bare keyword, IGNORECASE), not word-boundary matches (only forbidden_words uses
    # \b). re.escape keeps plain-word keywords value-identical to official's raw-regex
    # count while avoiding re.error on metacharacters.
    count = len(re.findall(re.escape(str(keyword)), response, re.IGNORECASE))
    return _relation_ok(count, relation, int(frequency))


def length_number_words(response: str, num_words=None, relation=None, **_) -> bool:
    if num_words is None:
        return False
    return _relation_ok(len(_words(response)), relation, int(num_words))


def length_number_sentences(response: str, num_sentences=None, relation=None, **_) -> bool:
    if num_sentences is None:
        return False
    return _relation_ok(len(_sentences(response)), relation, int(num_sentences))


def format_number_bullets(response: str, num_bullets=None, **_) -> bool:
    if num_bullets is None:
        return False
    # Match official BulletListChecker: sum of two patterns -- "*" markdown bullets (a
    # star followed by a non-star char) and "-" bullets (a dash with nothing required
    # after it). Requiring "[ \t]+\S" after the marker would undercount ("*c", "-y",
    # bare "-").
    bullets = re.findall(r"^\s*\*[^\*].*$", response, re.MULTILINE)
    bullets += re.findall(r"^\s*-.*$", response, re.MULTILINE)
    return len(bullets) == int(num_bullets)


def change_case_lowercase(response: str, **_) -> bool:
    # Checks case alone, with no dependency. Google's LowercaseLettersEnglishChecker also
    # requires langdetect.detect(value) == "en". See module docstring.
    return any(c.isalpha() for c in response) and response == response.lower()


def change_case_uppercase(response: str, **_) -> bool:
    # Checks case alone, as change_case_lowercase; official also requires langdetect == "en".
    return any(c.isalpha() for c in response) and response == response.upper()


def startend_end_checker(response: str, end_phrase=None, **_) -> bool:
    if end_phrase is None:
        return False
    # Official EndChecker strips surrounding double-quotes before matching (a model that
    # wraps its answer in quotes otherwise fails spuriously).
    return response.strip().strip('"').lower().endswith(str(end_phrase).strip().lower())


def startend_start_checker(response: str, start_phrase=None, **_) -> bool:
    if start_phrase is None:
        return False
    return response.strip().lower().startswith(str(start_phrase).strip().lower())


def content_number_placeholders(response: str, num_placeholders=None, **_) -> bool:
    if num_placeholders is None:
        return False
    return len(re.findall(r"\[.*?\]", response)) >= int(num_placeholders)


def format_title(response: str, **_) -> bool:
    # Official TitleChecker scans every <<...>> match and passes on the first with
    # non-blank content; checking only the first match would fail a response whose
    # opening line holds an empty <<>> and whose real title comes later.
    return any(title.lstrip("<").rstrip(">").strip()
               for title in re.findall(r"<<[^\n]+>>", response))


def startend_quotation(response: str, **_) -> bool:
    v = response.strip()
    return len(v) >= 2 and v.startswith('"') and v.endswith('"')


def length_number_paragraphs(response: str, num_paragraphs=None, **_) -> bool:
    if num_paragraphs is None:
        return False
    parts = re.split(r"\s?\*\*\*\s?", response)
    n = len(parts)
    for i, para in enumerate(parts):
        if not para.strip():
            if i == 0 or i == len(parts) - 1:
                n -= 1  # a leading/trailing divider is allowed
            else:
                return False  # an empty paragraph in the middle is not
    return n == int(num_paragraphs)


def punctuation_no_comma(response: str, **_) -> bool:
    return "," not in response


def format_number_highlighted_sections(response: str, num_highlights=None, **_) -> bool:
    if num_highlights is None:
        return False
    count = 0
    for h in re.findall(r"\*[^\n\*]*\*", response):
        if h.strip("*").strip():
            count += 1
    for h in re.findall(r"\*\*[^\n\*]*\*\*", response):
        if h.removeprefix("**").removesuffix("**").strip():
            count += 1
    return count >= int(num_highlights)


def combination_two_responses(response: str, **_) -> bool:
    # Official TwoResponsesChecker: exactly two non-blank responses around the
    # ``******`` divider, and they must differ after strip(). A blank segment is
    # tolerated only at the very start or end (a response that opens or closes with the
    # divider); a blank one between them means a missing response, so it fails.
    segments = response.split("******")
    valid = []
    for i, segment in enumerate(segments):
        if not segment.strip():
            if i not in (0, len(segments) - 1):
                return False
        else:
            valid.append(segment)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def content_postscript(response: str, postscript_marker=None, **_) -> bool:
    if postscript_marker is None:
        return False
    marker = str(postscript_marker).strip().lower()
    if marker == "p.p.s":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "p.s.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(marker) + r".*$"
    return re.findall(pattern, response.lower(), flags=re.MULTILINE) != []


def combination_repeat_prompt(response: str, prompt_to_repeat=None, **_) -> bool:
    if prompt_to_repeat is None:
        return False
    return response.strip().lower().startswith(str(prompt_to_repeat).strip().lower())


def change_case_capital_word_frequency(
    response: str, capital_frequency=None, capital_relation=None, **_
) -> bool:
    if capital_frequency is None:
        return False
    caps = [w for w in _words(response) if w.isupper() and any(c.isalpha() for c in w)]
    return _relation_ok(len(caps), capital_relation, int(capital_frequency))


def keywords_letter_frequency(
    response: str, letter=None, let_frequency=None, let_relation=None, **_
) -> bool:
    if letter is None or let_frequency is None:
        return False
    count = response.lower().count(str(letter).strip().lower())
    return _relation_ok(count, let_relation, int(let_frequency))


def language_response_language(response: str, language=None, **_) -> bool:
    if language is None:
        return False
    text = response.strip()
    if not text:
        return False  # langdetect has no signal on empty input -> fail closed
    # Lazy import (see module docstring): importing this registry must not hard-require
    # langdetect -- only running this checker does. seed=0 pins deterministic language
    # id per process (idempotent to set on each call).
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0
    try:
        return detect(text) == str(language).strip().lower()
    except LangDetectException:
        return False


def format_multiple_sections(response: str, section_spliter=None, num_sections=None, **_) -> bool:
    if section_spliter is None or num_sections is None:
        return False
    pattern = r"\s?" + re.escape(str(section_spliter)) + r"\s?\d+\s?"
    sections = re.split(pattern, response)
    return (len(sections) - 1) >= int(num_sections)


#: Characters official's ParagraphFirstWordCheck truncates the first word at, so
#: "don't" is judged as "don" and "Hello," as "hello".
_FIRST_WORD_PUNCTUATION = frozenset({".", ",", "?", "!", "'", '"'})


def length_nth_paragraph_first_word(
    response: str, first_word=None, nth_paragraph=None, num_paragraphs=None, **_
) -> bool:
    if first_word is None or nth_paragraph is None or num_paragraphs is None:
        return False
    # Official ParagraphFirstWordCheck counts only non-blank paragraphs but indexes the
    # unfiltered split, so a blank segment (a response opening with "\n\n", a tripled
    # break) shifts which paragraph nth_paragraph selects. Filtering before indexing
    # would silently select a different paragraph than the reference does.
    paragraphs = re.split(r"\n\n", response)
    n_nonblank = sum(1 for p in paragraphs if p.strip())
    n = int(nth_paragraph)
    if n < 1 or n > n_nonblank:      # n < 1 guards a malformed kwarg from indexing -1
        return False
    paragraph = paragraphs[n - 1].strip()
    if not paragraph:
        return False
    word = paragraph.split()[0].lstrip("'").lstrip('"')
    first = ""
    for letter in word:
        if letter in _FIRST_WORD_PUNCTUATION:
            break
        first += letter.lower()
    return n_nonblank == int(num_paragraphs) and first == str(first_word).strip().lower()


def format_json(response: str, **_) -> bool:
    # Official JsonFormat's exact unfencing: the three ```json spellings it knows, then
    # a bare ``` prefix, then a ``` suffix -- each removed unconditionally, so a reply
    # closed with a fence it never opened is unfenced too.
    v = (response.strip()
         .removeprefix("```json").removeprefix("```Json").removeprefix("```JSON")
         .removeprefix("```").removesuffix("```").strip())
    try:
        json.loads(v)
        return True
    except ValueError:      # json.JSONDecodeError is a ValueError
        return False


INSTRUCTION_DICT: dict[str, Callable[..., bool]] = {
    "keywords:existence": keywords_existence,
    "keywords:forbidden_words": keywords_forbidden,
    "keywords:frequency": keywords_frequency,
    "length_constraints:number_words": length_number_words,
    "length_constraints:number_sentences": length_number_sentences,
    "detectable_format:number_bullet_lists": format_number_bullets,
    "change_case:english_lowercase": change_case_lowercase,
    "change_case:english_capital": change_case_uppercase,
    "startend:end_checker": startend_end_checker,
    "startend:start_checker": startend_start_checker,
    "startend:quotation": startend_quotation,
    "detectable_content:number_placeholders": content_number_placeholders,
    "detectable_content:postscript": content_postscript,
    "detectable_format:title": format_title,
    "detectable_format:number_highlighted_sections": format_number_highlighted_sections,
    "detectable_format:multiple_sections": format_multiple_sections,
    "detectable_format:json_format": format_json,
    "length_constraints:number_paragraphs": length_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": length_nth_paragraph_first_word,
    "punctuation:no_comma": punctuation_no_comma,
    "combination:two_responses": combination_two_responses,
    "combination:repeat_prompt": combination_repeat_prompt,
    "change_case:capital_word_frequency": change_case_capital_word_frequency,
    "keywords:letter_frequency": keywords_letter_frequency,
    "language:response_language": language_response_language,
}
