"""CorpusBrowser reproduces elastic-inference's BrowserSession text format over the
corpus retriever, and the tool face returns errors as text."""
from __future__ import annotations

import asyncio

from tasks.deepresearch.corpus_browser import CorpusBrowser, CorpusBrowserTool
from tasks.deepresearch.retriever import Doc, Hit

L, D, R = "\u3010", "\u2020", "\u3011"  # the bracket marker characters


class _StubRetriever:
    def __init__(self):
        self.loaded = False
        self.docs = {
            "11": Doc("11", "https://a.example/1", "Alpha One", "Alpha One\nfirst line\nsecond line"),
            "12": Doc("12", "https://a.example/2", "Beta Two",
                      "Beta Two\n" + "\n".join(f"line {i}" for i in range(1, 60))),
        }

    def load(self):
        self.loaded = True

    async def search(self, query, k=10):
        if query == "nothing":
            return []
        return [Hit("11", "https://a.example/1", "Alpha One", "alpha snippet", 0.9),
                Hit("12", "https://a.example/2", "Beta Two", "beta snippet", 0.8)][:k]

    def document(self, docid):
        return self.docs.get(docid)

    def by_url(self, url):
        return next((d for d in self.docs.values() if d.url == url), None)


def _browser():
    return CorpusBrowser(_StubRetriever())


def test_search_page_matches_the_reference_format():
    b = _browser()
    text = asyncio.run(b.search("alpha", topn=10))
    assert text == "\n".join([
        "[1]",
        'L1: Searched for "alpha"',
        "L2: ",
        f"L3: {L}1{D}Alpha One{R}",
        "L4: URL: https://a.example/1",
        "L5: alpha snippet",
        "L6: ",
        f"L7: {L}2{D}Beta Two{R}",
        "L8: URL: https://a.example/2",
        "L9: beta snippet",
        "L10: ",
    ])
    assert asyncio.run(b.search("nothing")) == "No results found for: nothing"


def test_open_by_id_url_and_scroll_and_find():
    b = _browser()
    asyncio.run(b.search("alpha"))
    # a fresh open of a short document returns every line, not just a window.
    assert b.open(id=1) == "[2]\nL1: Alpha One\nL2: first line\nL3: second line"
    assert b.open(cursor=2, loc=2, num_lines=1) == "[2]\nL2: first line"
    assert b.find("second") == 'Found 1 match(es) for "second" in [2]:\nL3: second line'
    assert b.find("zzz") == 'No matches for "zzz" in [2].'
    # a fresh open of a 60-line document also returns every line, not a 50-line window.
    opened = b.open(id="https://a.example/2")
    assert opened.startswith("[3]\nL1: Beta Two\nL2: line 1")
    assert opened.count("\nL") == 60
    # re-opening the same page by cursor with no num_lines scrolls a window instead.
    scrolled = b.open(cursor=3)
    assert scrolled.startswith("[3]\nL1: Beta Two\nL2: line 1")
    assert scrolled.count("\nL") == 50
    assert b.open(id=1, cursor=1) == "[4]\nL1: Alpha One\nL2: first line\nL3: second line"


def test_open_errors_match_the_reference_wording():
    b = _browser()
    assert b.open(id=3) == ("Error: link id=3 not found. Use browser.search first, then "
                            "browser.open with a valid id from the results.")
    assert b.open() == ('Error: provide an id from search results or a URL string. '
                        'Example: browser.open({"id": 1}) or '
                        'browser.open({"id": "https://example.com"})')
    assert b.open(id="https://nowhere.example/x") == "Error: URL not in corpus: https://nowhere.example/x"
    assert b.find("x") == "Error: no page open. Use browser.open first."


def test_page_text_is_capped_and_find_lists_at_most_twenty():
    r = _StubRetriever()
    text = "T\n" + "\n".join(["hit"] * 25) + "\n" + "x" * 40000
    r.docs["11"] = Doc("11", "https://a.example/1", "T", text)
    b = CorpusBrowser(r)
    asyncio.run(b.search("alpha"))
    page = b.open(id=1)
    header, _, body = page.partition("\n")
    assert header == "[2]"
    # the page carries at most 30,000 characters of document text; the last line is truncated.
    lines = [line.split(": ", 1)[1] for line in body.split("\n")]
    assert lines == text[:30000].split("\n")
    found = b.find("hit")
    assert found.startswith('Found 25 match(es) for "hit" in [2]:')
    assert found.endswith("... and 5 more matches") and found.count("\nL") == 20


def test_tool_face_returns_errors_as_text_and_keeps_sessions_apart():
    tool = CorpusBrowserTool(retriever=_StubRetriever())
    asyncio.run(tool.warmup())
    assert tool.retriever.loaded
    out = asyncio.run(tool.search(query="alpha", _session_id="a"))
    assert out["documents"] == [out["raw_text"]] and out["query"] == "alpha"
    bad = asyncio.run(tool.find(pattern="x", loc=3, _session_id="a"))
    assert bad["raw_text"].startswith("Error in browser.find: TypeError: unexpected argument(s): loc")
    other = asyncio.run(tool.open(id=1, _session_id="b"))
    assert other["raw_text"].startswith("Error: link id=1 not found")   # session b never searched
    opened = asyncio.run(tool.open(id="1", _session_id="a"))
    assert opened["raw_text"].startswith("[2]\nL1: Alpha One")           # digit strings coerce
    assert opened["query"] == "open(id=1, cursor=None, loc=None)"


def test_a_failing_retriever_call_becomes_an_error_result():
    class _Boom(_StubRetriever):
        async def search(self, query, k=10):
            raise RuntimeError("boom")

    tool = CorpusBrowserTool(retriever=_Boom())
    out = asyncio.run(tool.search(query="q", _session_id="s"))
    assert out["raw_text"] == "Error in browser.search: RuntimeError: boom"
    assert out["documents"] == [out["raw_text"]]
