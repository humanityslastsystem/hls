"""A stateful browser over the corpus retriever, in elastic-inference's text format.

The served model was tuned on elastic-inference's ``BrowserSession`` output: search
results as ``[cursor]`` pages of ``L<n>: `` numbered lines with bracketed
``<id><dagger><title>`` markers, pages opened by result id or URL and scrolled by
cursor, and ``find`` over the current page. This module keeps that format exactly and
swaps the two network calls (a web search, a page fetch) for the corpus retriever.

Transcribed from ``BrowserSession`` in ``elastic_serving/dr_utils/tools.py`` of
https://github.com/RulinShao/elastic-inference at commit
e68c379927ad44c30c27a269d2da774c3473f590, so the format can be re-derived or
re-checked against the original.

No ``hls`` imports: the build sandbox runs this file without the harness.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from tasks.deepresearch.retriever import CorpusRetriever, Doc

PAGE_CHAR_CAP = 30000
SCROLL_LINES = 50
FIND_MAX_MATCHES = 20
_OPEN_USAGE = ('Error: provide an id from search results or a URL string. '
               'Example: browser.open({"id": 1}) or '
               'browser.open({"id": "https://example.com"})')


def _numbered(cursor: int, lines: list[str], start: int) -> str:
    body = "\n".join(f"L{start + i + 1}: {line}" for i, line in enumerate(lines))
    return f"[{cursor}]\n{body}"


class CorpusBrowser:
    """One model's browsing state: cursors, search results, opened pages."""

    def __init__(self, retriever: CorpusRetriever) -> None:
        self._retriever = retriever
        self._cursor_counter = 0
        self._pages: dict[int, dict[str, Any]] = {}
        self._search_results: dict[int, list[dict[str, Any]]] = {}
        self._current_cursor: int | None = None

    def _next_cursor(self) -> int:
        self._cursor_counter += 1
        return self._cursor_counter

    async def search(self, query: str, topn: int = 10) -> str:
        hits = await self._retriever.search(query, k=topn)
        if not hits:
            return f"No results found for: {query}"
        cursor = self._next_cursor()
        self._current_cursor = cursor
        results: list[dict[str, Any]] = []
        lines = [f'Searched for "{query}"', ""]
        for i, hit in enumerate(hits[:topn], 1):
            results.append({"id": i, "docid": hit.docid, "url": hit.url, "title": hit.title})
            lines.append(f"\u3010{i}\u2020{hit.title}\u3011")
            lines.append(f"URL: {hit.url}")
            lines.append(hit.snippet)
            lines.append("")
        self._search_results[cursor] = results
        self._pages[cursor] = {"url": None, "title": f"Search: {query}", "lines": lines}
        return _numbered(cursor, lines, 0)

    def open(self, id: Any = None, cursor: Any = None, loc: Any = None,
             num_lines: Any = None) -> str:
        doc: Doc | None
        if isinstance(id, str) and id.startswith("http"):
            doc = self._retriever.by_url(id)
            if doc is None:
                return f"Error: URL not in corpus: {id[:80]}"
        elif id is not None and id != -1:
            search_cursor = cursor if cursor and cursor != -1 else self._current_cursor
            docid = None
            if search_cursor and search_cursor in self._search_results:
                for r in self._search_results[search_cursor]:
                    if r["id"] == id:
                        docid = r["docid"]
                        break
            doc = self._retriever.document(docid) if docid is not None else None
            if doc is None:
                return (f"Error: link id={id} not found. Use browser.search first, then "
                        f"browser.open with a valid id from the results.")
        elif cursor and cursor != -1 and cursor in self._pages:
            page = self._pages[cursor]
            start = (loc - 1) if loc and loc > 0 else 0
            n = num_lines if num_lines and num_lines > 0 else SCROLL_LINES
            return _numbered(cursor, page["lines"][start:start + n], start)
        else:
            return _OPEN_USAGE
        all_lines = doc.text[:PAGE_CHAR_CAP].split("\n")
        new_cursor = self._next_cursor()
        self._current_cursor = new_cursor
        self._pages[new_cursor] = {"url": doc.url, "title": doc.url, "lines": all_lines}
        start = (loc - 1) if loc and loc > 0 else 0
        n = num_lines if num_lines and num_lines > 0 else len(all_lines)
        return _numbered(new_cursor, all_lines[start:start + n], start)

    def find(self, pattern: str, cursor: Any = None) -> str:
        target = cursor if cursor and cursor != -1 else self._current_cursor
        if not target or target not in self._pages:
            return "Error: no page open. Use browser.open first."
        matches = [(i + 1, line) for i, line in enumerate(self._pages[target]["lines"])
                   if pattern.lower() in line.lower()]
        if not matches:
            return f'No matches for "{pattern}" in [{target}].'
        out = [f'Found {len(matches)} match(es) for "{pattern}" in [{target}]:']
        out.extend(f"L{n}: {line}" for n, line in matches[:FIND_MAX_MATCHES])
        if len(matches) > FIND_MAX_MATCHES:
            out.append(f"... and {len(matches) - FIND_MAX_MATCHES} more matches")
        return "\n".join(out)


def _coerce_int(value: Any) -> Any:
    """Result ids and cursors arrive as whatever the model wrote; digit strings mean ints."""
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return value


@dataclass
class CorpusBrowserTool:
    """The tool-runner face: one browser per trajectory, errors returned as text.

    ``warmup`` loads the bundle before the driver's clock starts. A bad call (an
    argument the model invented, or a retriever failure) comes back as tool output the
    model can read and correct; raising would end the trajectory."""

    retriever: CorpusRetriever = field(default_factory=CorpusRetriever)
    # Keyed by session id, never evicted: process lifetime is the eviction policy,
    # since a member and a driver are fresh per serving and per drive.
    _browsers: dict[str, CorpusBrowser] = field(default_factory=dict)

    async def warmup(self) -> None:
        await asyncio.to_thread(self.retriever.load)

    def _session(self, session_id: str) -> CorpusBrowser:
        if session_id not in self._browsers:
            self._browsers[session_id] = CorpusBrowser(self.retriever)
        return self._browsers[session_id]

    async def _guarded(self, name: str, unexpected: dict[str, Any], call) -> dict:
        if unexpected:
            names = ", ".join(sorted(unexpected))
            text = f"Error in browser.{name}: TypeError: unexpected argument(s): {names}"
        else:
            try:
                return await call()
            except Exception as exc:
                text = f"Error in browser.{name}: {type(exc).__name__}: {exc}"
        return {"query": name, "documents": [text], "raw_text": text}

    async def search(self, query: str, topn: int = 10, _session_id: str = "default",
                     **unexpected: Any) -> dict:
        async def call() -> dict:
            text = await self._session(_session_id).search(query=query, topn=_coerce_int(topn))
            return {"query": query, "documents": [text], "raw_text": text}
        return await self._guarded("search", unexpected, call)

    async def open(self, id: Any = None, cursor: Any = None, loc: Any = None,
                   num_lines: Any = None, view_source: bool = False,
                   source: str | None = None, _session_id: str = "default",
                   **unexpected: Any) -> dict:
        async def call() -> dict:
            rid, cur, at, n = (_coerce_int(id), _coerce_int(cursor), _coerce_int(loc),
                               _coerce_int(num_lines))
            text = self._session(_session_id).open(id=rid, cursor=cur, loc=at, num_lines=n)
            return {"query": f"open(id={rid}, cursor={cur}, loc={at})",
                    "documents": [text], "raw_text": text}
        return await self._guarded("open", unexpected, call)

    async def find(self, pattern: str, cursor: Any = None, _session_id: str = "default",
                   **unexpected: Any) -> dict:
        async def call() -> dict:
            cur = _coerce_int(cursor)
            text = self._session(_session_id).find(pattern=pattern, cursor=cur)
            return {"query": f"find(pattern={pattern}, cursor={cur})",
                    "documents": [text], "raw_text": text}
        return await self._guarded("find", unexpected, call)
