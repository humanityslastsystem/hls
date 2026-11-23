"""A published trace's events, one at a time.

A published trace is one JSON document, ``{"events": [...]}``, in the shape of
``trace.json``, kept plain, gzipped, or gzipped and split into parts (``hls.viewer.scanner``'s
``PUBLISHED_TRACES``). Its format lets it be read an event at a time: ``published_events``
decompresses it a bounded piece at a time and parses one event at a time, holding a piece of the
files, its decompressed bytes and its decoded text, at most twice one event's JSON (a long event is
read on until twice what was read is held) and that event's parse. It never holds the document.

It reads a document as ``json.loads`` of the whole would, on the bytes ``gzip.decompress`` of the
whole would give: the same encoding, the same parse of each event, and the same refusals, the
document being checked whole, so a defect after the last event refuses it too. Four documents
``json.loads`` reads are refused: one that is not an object, one whose events are neither a list
nor empty, one with an event that is not an object, and one that
names its events twice, whose last ``json.loads`` keeps though the reader has yielded the first.
An event, or a member beside the events, longer than ``EVENT_MAX_CHARS`` is refused: the reader
holds no more of one, so a document that never closes one is refused there and not read on to its
end. A refusal raises ``TraceUnreadable`` where it is met, after the events before it: a reader
that must not show part of a trace drops what it took of it."""

from __future__ import annotations

import codecs
import json
import re
import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

#: The most bytes the reader takes from a file at once, and the most bytes one step of
#: decompression gives, so a piece that expands far on decompression is taken a bounded piece at
#: a time.
PIECE_BYTES = 2**20
#: The most characters of one event's JSON, or of one member beside the events, the reader holds:
#: a power of two above four times the longest event a task produces (an agentic coding turn with
#: its reply's logprobs, tens of millions of characters), as a task's trace cap is at least four
#: times its largest trace (``hls.drive.bounds.TRACE_MARGIN``).
EVENT_MAX_CHARS = 256 * 2**20
#: The characters a JSON number is written in, any of which may go on a number read so far.
_NUMBER_CHARS = re.compile(r"[0-9.eE+-]*")
#: JSON's whitespace, which ``json.loads`` skips between tokens.
_WHITESPACE = re.compile(r"[ \t\n\r]*")
#: The decoder ``json.loads`` parses with, parsing one value at a position.
_DECODER = json.JSONDecoder()


class TraceUnreadable(ValueError):
    """A trace the viewer does not read: one that is not whole or not valid, or that it refuses
    (see the module docstring)."""


def published_events(files: Sequence[Path], *, gzipped: bool) -> Iterator[dict[str, Any]]:
    """The events of the published trace that *files* hold one after another (one file, or a split
    trace's parts in order), gzipped when *gzipped*, one at a time. Raise TraceUnreadable for one
    the reader refuses, where it is met."""
    pieces = _file_pieces(files)
    try:
        yield from _document_events(_Text(_decoded(_gunzipped(pieces) if gzipped else pieces)))
    except (OSError, UnicodeDecodeError, zlib.error) as e:
        raise TraceUnreadable(f"{files[0]}: {e}") from None


def _file_pieces(files: Sequence[Path]) -> Iterator[bytes]:
    """The bytes of *files*, one after another, ``PIECE_BYTES`` at a time."""
    for path in files:
        with open(path, "rb") as f:
            while data := f.read(PIECE_BYTES):
                yield data


def _gunzipped(pieces: Iterator[bytes]) -> Iterator[bytes]:
    """The bytes the gzip members in *pieces* hold, one member after another, as ``gzip.decompress``
    reads them: zero bytes after a member pad it, anything else begins the next member, and a member
    that is corrupt or cut short raises zlib.error or TraceUnreadable. No step gives more than
    ``PIECE_BYTES``."""
    inflate = None
    ended = False
    for data in pieces:
        while data:
            if inflate is None:
                if ended:
                    data = data.lstrip(b"\0")
                    if not data:
                        break
                inflate = zlib.decompressobj(wbits=31)
            while True:
                out = inflate.decompress(data, PIECE_BYTES)
                if out:
                    yield out
                data = inflate.unconsumed_tail
                if inflate.eof:
                    data, inflate, ended = inflate.unused_data, None, True
                    break
                # A step that gave all it may can leave output inside zlib with no input left.
                if not data and len(out) < PIECE_BYTES:
                    break
    if inflate is not None:
        raise TraceUnreadable("a gzip member is cut short")


def _decoded(pieces: Iterator[bytes]) -> Iterator[str]:
    """The text of the bytes in *pieces*, in the encoding ``json.loads`` finds for them from their
    first four bytes (``json.detect_encoding``), decoded as it decodes them."""
    head = b""
    for piece in pieces:
        head += piece
        if len(head) >= 4:
            break
    decoder = codecs.getincrementaldecoder(json.detect_encoding(head))("surrogatepass")
    yield decoder.decode(head)
    for piece in pieces:
        yield decoder.decode(piece)
    yield decoder.decode(b"", final=True)


class _Text:
    """The text of a document, read a piece at a time: ``buf`` holds it from the first character
    still needed on, and reading stands at ``pos`` in it."""

    def __init__(self, pieces: Iterator[str]) -> None:
        self._pieces = pieces
        self.buf = ""
        self.pos = 0
        #: Whether ``buf`` holds the text to its end.
        self.ended = False

    def _more(self, want: int) -> None:
        """Read on until ``buf`` holds *want* characters from ``pos`` on, or the text ends,
        dropping what lies before ``pos``."""
        kept = [self.buf[self.pos:]]
        self.buf = ""
        held = len(kept[0])
        while held < want:
            piece = next(self._pieces, None)
            if piece is None:
                self.ended = True
                break
            kept.append(piece)
            held += len(piece)
        self.buf, self.pos = "".join(kept), 0

    def next_char(self) -> str:
        """Move past whitespace to the next character, and return it; "" at the end of the text."""
        while True:
            self.pos = _WHITESPACE.match(self.buf, self.pos).end()
            if self.pos < len(self.buf):
                return self.buf[self.pos]
            if self.ended:
                return ""
            self._more(1)

    def value(self) -> Any:
        """The JSON value at ``pos``, moving past it. Raise TraceUnreadable for one that is not
        valid, or longer than ``EVENT_MAX_CHARS``.

        A value that does not parse in what has been read may go on past it, so it is parsed again
        once twice as much has been read, and a long one is parsed a bounded number of times over.
        So is a number followed by nothing but what may go on a number to the end of what has been
        read (``1.`` of ``1.5``)."""
        while True:
            try:
                value, end = _DECODER.raw_decode(self.buf, self.pos)
            except json.JSONDecodeError:
                value, end = None, -1
            except (ValueError, RecursionError) as e:
                raise TraceUnreadable(f"a value is not one the reader parses: {e}") from None
            held = len(self.buf) - self.pos
            number = isinstance(value, (int, float)) and not isinstance(value, bool)
            unfinished = number and not self.ended and _NUMBER_CHARS.fullmatch(self.buf, end) is not None
            if end >= 0 and not unfinished:
                if end - self.pos > EVENT_MAX_CHARS:
                    raise TraceUnreadable(f"a value runs past {EVENT_MAX_CHARS:,} characters")
                self.pos = end
                return value
            if self.ended:
                raise TraceUnreadable("the document is not valid JSON")
            if held > EVENT_MAX_CHARS:
                raise TraceUnreadable(f"a value runs past {EVENT_MAX_CHARS:,} characters, or is not valid JSON")
            self._more(held + 1 if end >= 0 else min(2 * held + 1, EVENT_MAX_CHARS + 1))

    def expect(self, char: str) -> None:
        """Move past *char*, the next character past whitespace; raise TraceUnreadable for any other."""
        if self.next_char() != char:
            raise TraceUnreadable(f"the document is not valid JSON: {char!r} expected")
        self.pos += 1


def _document_events(text: _Text) -> Iterator[dict[str, Any]]:
    """The events of the document *text* holds, as the module docstring says it reads them."""
    if text.next_char() != "{":
        raise TraceUnreadable("the document is not a JSON object")
    text.pos += 1
    named = False
    if text.next_char() == "}":
        text.pos += 1
    else:
        while True:
            if text.next_char() != '"':
                raise TraceUnreadable("the document is not valid JSON: a member's name expected")
            name = text.value()
            text.expect(":")
            first = text.next_char()
            if name != "events":
                text.value()
            elif named:
                raise TraceUnreadable("the document names its events twice")
            else:
                named = True
                if first == "[":
                    text.pos += 1
                    yield from _list_events(text)
                elif text.value():
                    raise TraceUnreadable("the document's events are not a list")
            end = text.next_char()
            text.pos += 1
            if end == "}":
                break
            if end != ",":
                raise TraceUnreadable("the document is not valid JSON: ',' or '}' expected")
    if text.next_char() != "":
        raise TraceUnreadable("the document is not valid JSON: more follows it")


def _list_events(text: _Text) -> Iterator[dict[str, Any]]:
    """The events of the list whose opening bracket *text* has just moved past, up to its end."""
    if text.next_char() == "]":
        text.pos += 1
        return
    while True:
        text.next_char()
        event = text.value()
        if not isinstance(event, dict):
            raise TraceUnreadable("an event is not a JSON object")
        yield event
        end = text.next_char()
        text.pos += 1
        if end == "]":
            return
        if end != ",":
            raise TraceUnreadable("the document is not valid JSON: ',' or ']' expected")
