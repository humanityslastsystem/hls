"""A secret's searchable forms: what the scrubber replaces and every secret scan searches for.

A secret is found by its value, as given. A value that is an http(s) URL, a route's base URL, is
found by its host too, in any ASCII case, when the host identifies a machine (``identifies``): a
host is case-insensitive, a client may print the address it dialled in either case, and a record
that names the host alone still says where the route goes. A DNS name's trailing dot, which a
resolver ignores, is left off the host (``url_host``), so its form finds the name with the dot and
without it. A host that names no particular machine, ``localhost`` or a name under it, a
single-label name such as ``relay``, or a loopback, private, shared (CGNAT), link-local or
unspecified address, appears in any server's records and is ordinary text there (a single label is
often the route's own name), so searching for it would find it in files that hold no secret; it is
no form, and only the whole value is searched. A host of numbers alone, such as ``127.1`` or
``0x7f.0.0.1``, is an IPv4 address as ``inet_aton`` reads it, as a client dials it, and one
``inet_aton`` refuses names no machine. An IP host is found as written and as the address a client
prints once it dialled it (``0x8.8.4.4`` as ``8.8.4.4`` too, an IPv6 address compressed), both
under the host's label. ``forms_of`` gives one secret's forms and ``registry_forms`` a whole
registry's: the scrubber (``hls.substrate.secrets.Scrubber``), the transcript's copy
(``hls.agent.transcript``), the agent's route checks (``hls.arms.route_values``) and every secret
scan take them from here, so one rule says which host is a secret's.

Each form has a label, the name records, reports and redactions give it: the secret's name for its
value, and the name with ``.host`` for its host, so ``relay_url.host``. A secret's name matches
``SECRET_NAME``, which holds no dot, so ``label_form`` reads a label back as its secret and form.

``ByteScrubber`` replaces forms in bytes with their markers, whole or as a stream read a piece at a
time: the runner's scrubber (``hls.substrate.secrets.Scrubber``) and the transcript's copy both
replace through it, so one rule says how a form split between two reads is still replaced whole.

A scrub finds a form whole, so a text cut short, the first characters of a command's output in an
error, the last of a long output, either end of a capped log, must not end inside one: every such
cut backs off past the piece of a form it may have gone through (``back_off``).

The module uses only the standard library: the transcript role imports it."""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import AnyStr, Literal
from urllib.parse import urlsplit

#: Which of a secret's forms a text is: its value, or the host of a value that is an http(s) URL.
Form = Literal["value", "host"]
#: Every form a scan searches, in the order a secret's forms are given.
FORMS: tuple[Form, ...] = ("value", "host")
#: A secret's name in ``[secrets]``.
SECRET_NAME = re.compile(r"[a-z][a-z0-9_]*")
#: Bytes a streamed scrub reads at a time (``ByteScrubber.copy``).
SCRUB_CHUNK_BYTES = 1 << 20
#: A host of numbers alone, each decimal, octal or hex as ``inet_aton`` reads them, dot-separated:
#: one that resolver reads as an IPv4 address whatever its labels' count, rather than as a name.
_NUMERIC_HOST = re.compile(r"(?:0x[0-9a-f]*|[0-9]+)(?:\.(?:0x[0-9a-f]*|[0-9]+))*", re.IGNORECASE)


def url_host(value: str) -> str | None:
    """The host of *value* when it is an http(s) URL with one, lowercase and without the trailing
    dot a resolver ignores; None otherwise."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    host = parts.hostname.rstrip(".") if parts.scheme in ("http", "https") and parts.hostname else None
    return host or None


def host_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The IP address *host*, a URL's host as ``url_host`` gives it, names: an IP literal, or a
    host of numbers alone read as ``inet_aton`` reads it (``127.1`` is 127.0.0.1); None for a name.
    Raise ValueError for a host of numbers alone ``inet_aton`` refuses, which is neither."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    if not _NUMERIC_HOST.fullmatch(host):
        return None
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except OSError:
        raise ValueError(f"{host!r} is numbers alone that inet_aton refuses") from None


def identifies(host: str) -> bool:
    """Whether *host*, a URL's host as ``url_host`` gives it, names a particular machine: a public
    IP address (``is_global``, ``host_address``), or a DNS name of more than one label that is
    neither ``localhost`` nor a name under it, in any case and a trailing dot apart. A loopback,
    private, shared, link-local, unspecified or otherwise non-global address, a host of numbers
    ``inet_aton`` refuses, and a single-label name, identify nothing."""
    host = host.rstrip(".")
    try:
        address = host_address(host)
    except ValueError:
        return False
    if address is not None:
        return address.is_global
    labels = host.split(".")
    return len(labels) > 1 and all(labels) and labels[-1].lower() != "localhost"


@dataclass(frozen=True)
class SecretForm:
    """One form of one secret: the text searched for, and the secret and form it stands for."""

    #: The secret's name.
    name: str
    #: Which of the secret's forms this is.
    form: Form
    #: What is searched for: the value, or its URL's host. Left out of the repr, which would print
    #: it.
    text: str = field(repr=False)

    def __post_init__(self) -> None:
        # An empty text would be found in everything.
        if not self.text:
            raise ValueError(f"form {self.form} of secret {self.name!r} is empty")

    @property
    def label(self) -> str:
        """The name records, reports and redactions give this form: the secret's name for its
        value, ``<name>.<form>`` for any other form."""
        return self.name if self.form == "value" else f"{self.name}.{self.form}"

    @property
    def any_case(self) -> bool:
        """Whether the text is found in any ASCII case, as a host is, or only as given."""
        return self.form == "host"

    @property
    def marker(self) -> str:
        """What the scrubber puts in this form's place."""
        return f"<redacted:{self.label}>"


def forms_of(name: str, value: str) -> list[SecretForm]:
    """The forms of the secret *name* holding *value*: the value, then its host (``url_host``)
    when it is an http(s) URL whose host identifies a machine (``identifies``), and, for an IP host
    written otherwise, the address as a client prints it (``host_address``). No form for an empty
    value."""
    if not value:
        return []
    forms = [SecretForm(name=name, form="value", text=value)]
    host = url_host(value)
    if host is None or not identifies(host):
        return forms
    forms.append(SecretForm(name=name, form="host", text=host))
    address = host_address(host)
    if address is not None and str(address) != host:
        forms.append(SecretForm(name=name, form="host", text=str(address)))
    return forms


def registry_forms(registry: Mapping[str, str]) -> list[SecretForm]:
    """The forms of every secret of *registry* (name to value): every value in the registry's
    order, then every host in the same order."""
    forms = [form for name, value in registry.items() for form in forms_of(name, value)]
    return [form for kind in FORMS for form in forms if form.form == kind]


class PastLimit(Exception):
    """A streamed scrub's output would pass the limit it was given (``ByteScrubber.copy``)."""


class ByteScrubber:
    """Replaces each of a list of forms in bytes with its marker (``SecretForm.marker``): a value as
    given, a host in any ASCII case. The forms are replaced longest first, ties in the order given,
    so a form inside another, a URL's host inside the URL, is replaced only where the longer one is
    not."""

    def __init__(self, forms: Iterable[SecretForm]) -> None:
        #: The forms, longest first.
        self.forms = sorted(forms, key=lambda form: len(form.text), reverse=True)
        # Each form as bytes, with its marker and whether it is found in any case.
        self._subs = [(form.text.encode(), form.marker.encode(), form.any_case) for form in self.forms]
        #: The bytes a streamed scrub holds back from each read: one fewer than the longest form has.
        self.hold = max((len(text) for text, _, _ in self._subs), default=1) - 1

    def scrub(self, data: bytes) -> bytes:
        """*data* with every form replaced by its marker."""
        for text, marker, any_case in self._subs:
            if any_case:
                data = re.sub(re.escape(text), lambda _m, m=marker: m, data, flags=re.IGNORECASE)
            else:
                data = data.replace(text, marker)
        return data

    def copy(self, read: Callable[[int], bytes], write: Callable[[bytes], object], *,
             chunk_bytes: int = SCRUB_CHUNK_BYTES, limit: int | None = None) -> tuple[int, str]:
        """Write the bytes *read* gives, asked for *chunk_bytes* at a time until it gives none, to
        *write*, scrubbed, and return the size and sha256 of what was written. No piece written ends
        inside a form: the last bytes read, one fewer than the longest form has, wait for the next
        read, and so does every form that starts before them and ends among them, so a form split
        between two reads is replaced whole, as it is in the bytes scrubbed at once. Raise PastLimit,
        with no more than *limit* bytes written, once the scrubbed bytes would pass *limit*."""
        digest, size, carry = hashlib.sha256(), 0, b""

        def put(data: bytes) -> None:
            nonlocal size
            piece = self.scrub(data)
            if limit is not None and size + len(piece) > limit:
                raise PastLimit()
            write(piece)
            digest.update(piece)
            size += len(piece)

        while chunk := read(chunk_bytes):
            data = carry + chunk
            cut = self._uncut(data, max(0, len(data) - self.hold))
            put(data[:cut])
            carry = data[cut:]
        put(carry)
        return size, digest.hexdigest()

    def _uncut(self, data: bytes, cut: int) -> int:
        """The largest position at or before *cut* that no occurrence of a form in *data* spans."""
        folded = data.lower()
        moved = True
        while moved:
            moved = False
            for text, _, any_case in self._subs:
                where, what = (folded, text.lower()) if any_case else (data, text)
                start = where.find(what, max(0, cut - len(text) + 1))
                if start != -1 and start < cut:
                    cut, moved = start, True
        return cut


def back_off(text: AnyStr, forms: Iterable[SecretForm], *, side: Literal["end", "start"]) -> AnyStr:
    """*text*, cut short at its *side*, without the piece there that may be part of one of *forms*
    the cut went through: at its end, a head, the longest piece that starts a form without being
    all of it; at its start, a tail, the longest that ends one. A form found in any case is matched
    in any case, and bytes against each form's UTF-8 encoding. Dropping a piece may leave another
    form's, so it goes on until none is left. A whole form stays, for the scrubber to replace; a
    piece that only looks like part of a form costs the cut a few characters."""
    texts = [(f.text if isinstance(text, str) else f.text.encode(), f.any_case) for f in forms]
    # A one-character form has no piece but itself.
    texts = [(needle, any_case) for needle, any_case in texts if len(needle) > 1]
    while True:
        longest = 0
        for needle, any_case in texts:
            window = text[-(len(needle) - 1):] if side == "end" else text[:len(needle) - 1]
            if any_case:
                window, needle = window.lower(), needle.lower()
            for k in range(min(len(needle) - 1, len(window)), longest, -1):
                if (window.endswith(needle[:k]) if side == "end" else window.startswith(needle[-k:])):
                    longest = k
                    break
        if not longest:
            return text
        text = text[:len(text) - longest] if side == "end" else text[longest:]


def label_form(label: str) -> tuple[str, Form]:
    """The secret's name and the form *label* names (``SecretForm.label``). Raise ValueError for
    anything that is no form's label."""
    name, dot, suffix = label.partition(".")
    if SECRET_NAME.fullmatch(name):
        if not dot:
            return name, "value"
        for form in FORMS:
            if form != "value" and suffix == form:
                return name, form
    others = ", ".join(f".{form}" for form in FORMS if form != "value")
    raise ValueError(f"{label!r} is no form's label: a secret's name, alone or with one of {others}")


def form_text(label: str) -> str:
    """How a record says which form *label* names (``label_form``): the secret's name for its
    value, "the host of <name>" for its host."""
    name, form = label_form(label)
    return name if form == "value" else f"the {form} of {name}"
