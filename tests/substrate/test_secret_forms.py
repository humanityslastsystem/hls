"""A secret's searchable forms (``hls.substrate.secret_forms``): its value, and the host of a value
that is an http(s) URL, each with the label records name it by. The Scrubber and the transcript's
scrub are built on them, and their output on a fixture is pinned."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys

import pytest

from hls.agent import transcript
from hls.substrate.secret_forms import (FORMS, ByteScrubber, PastLimit, SecretForm, back_off, forms_of,
                                        identifies, label_form, registry_forms, url_host)
from hls.substrate.secrets import Scrubber, SecretsFile

URL = "https://Relay.Example:8443/v1/anthropic"
HOST = "relay.example"
KEY = "sk-test-0123456789"


def test_a_value_is_one_form_searched_as_given():
    assert forms_of("relay_key", KEY) == [SecretForm(name="relay_key", form="value", text=KEY)]


@pytest.mark.parametrize("value, host", [(URL, HOST),
                                         ("http://mirror.example.org/path", "mirror.example.org"),
                                         ("https://relay-0123456789abcdef.example.net/v1",
                                          "relay-0123456789abcdef.example.net"),
                                         ("https://localhost.example:8443/v1", "localhost.example"),
                                         ("https://8.8.4.4:9000", "8.8.4.4"),
                                         ("https://[2001:4860:4860::8888]/v1", "2001:4860:4860::8888")],
                         ids=["https", "http", "relay-dns-name", "localhost-as-a-label", "public-ipv4",
                              "public-ipv6"])
def test_an_http_url_whose_host_identifies_something_gives_it_after_its_value(value, host):
    assert forms_of("relay_url", value) == [SecretForm(name="relay_url", form="value", text=value),
                                            SecretForm(name="relay_url", form="host", text=host)]


@pytest.mark.parametrize("value", ["http://localhost:8000/v1", "https://LOCALHOST/v1", "http://api.localhost/v1",
                                   "http://relay/v1", "http://127.0.0.1:8000", "http://127.9.9.9",
                                   "http://[::1]:8000/v1", "https://10.0.0.7:9000", "http://192.168.1.20/v1",
                                   "http://172.16.0.4/v1", "http://[fd00::4]/v1", "http://169.254.169.254/latest",
                                   "http://[fe80::1]/v1", "http://0.0.0.0:8000", "http://[::]:8000"],
                         ids=["localhost", "localhost-upper", "under-localhost", "single-label", "loopback-v4",
                              "loopback-v4-range", "loopback-v6", "private-10", "private-192", "private-172",
                              "private-v6", "link-local-v4", "link-local-v6", "unspecified-v4", "unspecified-v6"])
def test_a_host_that_identifies_nothing_is_no_form(value):
    # Any server's records mention such a host; searching for it would find it in harness files
    # that hold no secret.
    assert forms_of("relay_url", value) == [SecretForm(name="relay_url", form="value", text=value)]


def test_the_scrubber_replaces_a_url_whose_host_identifies_nothing_and_leaves_the_host_alone():
    scrubber = Scrubber({"local_url": "http://localhost:8000/v1/secret-path"})
    assert scrubber.scrub("at http://localhost:8000/v1/secret-path, then localhost:8000/health") == (
        "at <redacted:local_url>, then localhost:8000/health")


@pytest.mark.parametrize("value", ["ftp://relay.example/x", "relay.example-0123456789",
                                   "https:///no-host/path", "mailto:someone@relay.example", "https://[::1"],
                         ids=["ftp", "no-url", "no-host", "mailto", "unparsable"])
def test_a_value_that_is_no_http_url_with_a_host_has_no_host_form(value):
    assert [f.form for f in forms_of("token", value)] == ["value"]


def test_an_empty_value_has_no_form():
    assert forms_of("token", "") == []
    with pytest.raises(ValueError, match="empty"):
        SecretForm(name="token", form="value", text="")


def test_every_form_a_scan_searches_is_one_a_secret_can_have():
    assert FORMS == ("value", "host")
    assert tuple(f.form for f in forms_of("relay_url", URL)) == FORMS


def test_a_form_is_labelled_by_its_secrets_name_and_a_host_by_its_suffix():
    value, host = forms_of("relay_url", URL)
    assert (value.label, value.marker, value.any_case) == ("relay_url", "<redacted:relay_url>", False)
    assert (host.label, host.marker, host.any_case) == ("relay_url.host", "<redacted:relay_url.host>", True)


def test_a_forms_repr_names_its_secret_and_never_shows_its_text():
    for form in forms_of("relay_url", URL):
        assert repr(form) == f"SecretForm(name='relay_url', form='{form.form}')"


def test_a_label_reads_back_as_its_secret_and_form():
    for form in registry_forms({"relay_url": URL, "relay_key": KEY, "x2_y": "http://a.example/"}):
        assert label_form(form.label) == (form.name, form.form)
    assert label_form("relay_url.host") == ("relay_url", "host")
    assert label_form("relay_url") == ("relay_url", "value")


@pytest.mark.parametrize("label", ["", "Relay", "relay.", ".host", "relay.value", "relay.port", "a.b.host",
                                   "relay.HOST", "relay host", "1relay", "relay.host.host", "relay-url"])
def test_anything_but_a_label_is_refused(label):
    with pytest.raises(ValueError):
        label_form(label)


def test_a_registrys_forms_are_every_value_then_every_host_in_its_order():
    registry = {"b_url": "https://b.example/v1", "a_key": KEY, "c_url": "http://c.example", "empty": ""}
    forms = registry_forms(registry)
    assert [f.label for f in forms] == ["b_url", "a_key", "c_url", "b_url.host", "c_url.host"]
    assert [f.text for f in forms][3:] == ["b.example", "c.example"]


# --- the Scrubber and the transcript's scrub, pinned on one fixture ---------------------------------

#: Values that overlap (``first`` and ``second``), one inside another (``relay_key`` in ``outer``),
#: a URL whose host is printed in other cases, a value that holds that host (``hosted``) and one as
#: long as the host that overlaps it (``y_key``), which shows which of the two is replaced first.
REGISTRY = {
    "relay_url": URL,
    "y_key": "y.example-key",
    "relay_key": KEY,
    "outer": "xxsk-test-0123456789yy",
    "first": "abcdefgh",
    "second": "efghijkl",
    "hosted": "relay.example-0123456789",
    "mirror_url": "http://mirror.example.org/path",
}
TEXT = ("url https://Relay.Example:8443/v1/anthropic shouted HTTPS://RELAY.EXAMPLE:8443/v1/anthropic\n"
        "dial relay.example and RELAY.EXAMPLE and Relay.Example:8443\n"
        "hosted relay.example-0123456789 tie relay.example-key\n"
        "keys xxsk-test-0123456789yy and sk-test-0123456789 overlap abcdefghijkl\n"
        "mirror http://mirror.example.org/path and MIRROR.example.ORG/other\n")
DATA = TEXT.encode() + b"\xff tail Relay.Example\n"
#: What the Scrubber makes of ``TEXT``.
SCRUBBED = ("url <redacted:relay_url> shouted HTTPS://<redacted:relay_url.host>:8443/v1/anthropic\n"
            "dial <redacted:relay_url.host> and <redacted:relay_url.host> and "
            "<redacted:relay_url.host>:8443\n"
            "hosted <redacted:hosted> tie rela<redacted:y_key>\n"
            "keys <redacted:outer> and <redacted:relay_key> overlap <redacted:first>ijkl\n"
            "mirror <redacted:mirror_url> and <redacted:mirror_url.host>/other\n")
SCRUBBED_DATA = SCRUBBED.encode() + b"\xff tail <redacted:relay_url.host>\n"


def test_the_scrubbers_output_is_what_it_was(tmp_path):
    scrub = Scrubber(REGISTRY)
    assert scrub.scrub(TEXT) == SCRUBBED
    assert scrub.scrub_bytes(DATA) == SCRUBBED_DATA
    src = tmp_path / "role.out"
    src.write_bytes(DATA)
    for chunk in (1, 2, 7, len(HOST) - 1, len(HOST), 64, 1 << 20):
        scrub.scrub_file(src, to=tmp_path / "scrubbed.out", chunk_bytes=chunk)
        assert (tmp_path / "scrubbed.out").read_bytes() == SCRUBBED_DATA, chunk


def test_the_scrubber_holds_whole_values_only():
    scrub = Scrubber(REGISTRY)
    assert all(scrub.holds(f"x{value}y") for value in REGISTRY.values())
    assert not scrub.holds(f"{HOST} RELAY.EXAMPLE mirror.example.org")


#: The transcript's ``--scrub`` pairs over three of ``REGISTRY``'s secrets, and its replacements and
#: output on ``DATA``: per pair, its value then its host, longest first, so the host is replaced
#: before ``y_key``, which is as long.
PAIRS = ["URL=relay_url", "Y=y_key", "KEY=relay_key"]
ENVIRON = {"URL": REGISTRY["relay_url"], "Y": REGISTRY["y_key"], "KEY": REGISTRY["relay_key"]}
SUBS = [(b"https://Relay.Example:8443/v1/anthropic", b"<redacted:relay_url>", False),
        (b"sk-test-0123456789", b"<redacted:relay_key>", False),
        (b"relay.example", b"<redacted:relay_url.host>", True),
        (b"y.example-key", b"<redacted:y_key>", False)]
COPIED = (b"url <redacted:relay_url> shouted HTTPS://<redacted:relay_url.host>:8443/v1/anthropic\n"
          b"dial <redacted:relay_url.host> and <redacted:relay_url.host> and <redacted:relay_url.host>:8443\n"
          b"hosted <redacted:relay_url.host>-0123456789 tie <redacted:relay_url.host>-key\n"
          b"keys xx<redacted:relay_key>yy and <redacted:relay_key> overlap abcdefghijkl\n"
          b"mirror http://mirror.example.org/path and MIRROR.example.ORG/other\n"
          b"\xff tail <redacted:relay_url.host>\n")


def test_the_transcripts_replacements_are_what_they_were():
    scrubber = transcript.scrubber_for(PAIRS, ENVIRON)
    assert [(form.text.encode(), form.marker.encode(), form.any_case) for form in scrubber.forms] == SUBS
    assert scrubber.scrub(DATA) == COPIED


@pytest.mark.parametrize("forms", [registry_forms(REGISTRY), transcript.scrubber_for(PAIRS, ENVIRON).forms],
                         ids=["scrubber", "transcript"])
def test_a_stream_scrubbed_a_read_at_a_time_is_the_bytes_scrubbed_at_once(forms):
    """Every read size, down to one byte, splits some form of the fixture: the stream comes out as
    the whole bytes scrubbed at once, a URL split after its host replaced whole, overlapping values
    replaced as they are at once, and the size and sha256 returned are those of what was written."""
    scrubber = ByteScrubber(forms)
    whole = scrubber.scrub(DATA)
    for chunk in range(1, max(len(form.text) for form in forms) + 2):
        written: list[bytes] = []
        size, sha256 = scrubber.copy(io.BytesIO(DATA).read, written.append, chunk_bytes=chunk)
        out = b"".join(written)
        assert out == whole, chunk
        assert (size, sha256) == (len(out), hashlib.sha256(out).hexdigest())


def test_a_stream_scrubbed_with_no_form_is_copied_as_it_is():
    written: list[bytes] = []
    assert ByteScrubber([]).copy(io.BytesIO(DATA).read, written.append, chunk_bytes=3)[0] == len(DATA)
    assert b"".join(written) == DATA


def test_a_stream_stops_before_the_piece_that_would_take_it_past_its_limit():
    """The limit holds the scrubbed bytes, which a host replaced by its longer marker makes more than
    were read: the piece that would pass it is never written."""
    scrubber = ByteScrubber(forms_of("relay_url", URL))
    data = b"dial relay.example now " * 100
    whole = scrubber.scrub(data)
    assert len(whole) > len(data)
    written: list[bytes] = []
    with pytest.raises(PastLimit):
        scrubber.copy(io.BytesIO(data).read, written.append, chunk_bytes=64, limit=len(whole) - 1)
    assert 0 < len(b"".join(written)) <= len(whole) - 1 and whole.startswith(b"".join(written))
    written = []
    assert scrubber.copy(io.BytesIO(data).read, written.append, chunk_bytes=64, limit=len(whole))[0] == len(whole)
    assert b"".join(written) == whole


@pytest.mark.parametrize("module", ["hls.substrate.secret_forms", "hls.agent.transcript"])
def test_the_forms_and_the_transcript_role_use_the_standard_library_only(module):
    """The transcript role runs on the host with the toolchain's interpreter and imports the forms:
    neither loads a module outside the standard library but the forms' own."""
    probe = ("import sys\nbefore = set(sys.modules)\n"
             f"import {module}\n"
             "print('\\n'.join(sorted(m for m in set(sys.modules) - before "
             "if m.split('.')[0] not in sys.stdlib_module_names)))")
    done = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    loaded = set(done.stdout.split())
    assert loaded == {"hls", "hls.substrate", "hls.substrate.secret_forms"} | (
        {"hls.agent", "hls.agent.transcript", "hls.agent.listing"} if module == "hls.agent.transcript" else set())


def test_a_value_inside_another_secrets_host_is_refused_with_the_file():
    # A URL's host lies inside the URL, so the rule that no value lies inside another's covers a
    # value inside a host: no find of one secret's value is ever a find of another's host.
    inner = "token-0123456789abcdef"
    with pytest.raises(ValueError, match="contains secret hub's"):
        SecretsFile.model_validate({"secrets": {"relay_url": f"https://{inner}.example.net/v1", "hub": inner}})


def test_a_dns_hosts_trailing_dot_is_left_off_its_form_so_both_spellings_are_found():
    [_, host] = forms_of("relay_url", "https://Relay.Example.com./v1")
    assert host.text == "relay.example.com"
    assert Scrubber({"relay_url": "https://x"}).scrub("x") == "x"
    scrubber = Scrubber({"relay_url": "https://relay.example.com./v1"})
    assert scrubber.scrub("dialled relay.example.com. and RELAY.EXAMPLE.COM") == (
        "dialled <redacted:relay_url.host>. and <redacted:relay_url.host>")


@pytest.mark.parametrize("host", ["127.1", "10.1", "0x7f.1", "0177.0.0.1", "10.1.", "2130706433", "100.64.0.1",
                                  "1.2.3.4.5", "999.1"],
                         ids=["short-loopback", "short-private", "hex", "octal", "trailing-dot", "one-number",
                              "cgnat", "five-numbers", "too-large"])
def test_an_all_numeric_host_is_an_address_by_inet_aton_rules(host):
    # A host of numbers alone that inet_aton refuses is no address, and dials nothing.
    assert forms_of("relay_url", f"http://{host}:8000/v1") == [
        SecretForm(name="relay_url", form="value", text=f"http://{host}:8000/v1")]


@pytest.mark.parametrize("host, texts", [("8.8.4.4.", ["8.8.4.4"]), ("0x8.8.4.4", ["0x8.8.4.4", "8.8.4.4"]),
                                         ("8.8.1028", ["8.8.1028", "8.8.4.4"]),
                                         ("[2001:4860:4860:0:0:0:0:8888]",
                                          ["2001:4860:4860:0:0:0:0:8888", "2001:4860:4860::8888"])],
                         ids=["trailing-dot", "hex", "three-parts", "ipv6-uncompressed"])
def test_an_ip_host_is_a_form_as_written_and_as_the_address_a_client_prints(host, texts):
    assert [f.text for f in forms_of("relay_url", f"https://{host}/v1") if f.form == "host"] == texts
    scrubber = Scrubber({"relay_url": f"https://{host}/v1"})
    shown = [text.upper() for text in texts]
    assert scrubber.scrub(f"dialled {' and '.join(shown)}") == (
        f"dialled {' and '.join(['<redacted:relay_url.host>'] * len(texts))}")


# --- which hosts are forms: the rule the scrubber, the transcript and the route's guards share ----

@pytest.mark.parametrize("host, identifying", [
    ("relay.example", True), ("api.example.com", True), ("8.8.8.8", True), ("2001:4860:4860::8888", True),
    # A single label, localhost or a name under it, names no particular machine.
    ("gateway", False), ("relay", False), ("localhost", False), ("gw.localhost", False),
    # Nor does a loopback, private, link-local or unspecified address, in any spelling a resolver
    # takes: an all-numeric dotted host is an IP literal (127.1 is 127.0.0.1, 10.7 is 10.0.0.7).
    ("127.0.0.1", False), ("127.1", False), ("10.7", False), ("0x7f.1", False), ("2130706433", False),
    ("10.0.0.7", False), ("192.168.1.2", False), ("169.254.1.1", False), ("0.0.0.0", False), ("::1", False),
    ("fe80::1", False),
    # A dotted all-numeric name no resolver reads as an address is no host at all.
    ("999.1", False), ("1.2.3.4.5", False),
], ids=lambda value: str(value))
def test_a_host_is_a_form_only_when_it_identifies_a_machine(host, identifying):
    assert identifies(host) is identifying


def test_a_dns_host_is_the_form_without_its_trailing_dot():
    """A resolver reads ``relay.example.`` as ``relay.example``, and a record names it so: the form
    is the name without the dot, which is found in both."""
    assert url_host("https://relay.example.:8443/v1") == "relay.example"
    [_, host] = forms_of("relay_url", "https://Relay.Example.:8443/v1")
    assert (host.form, host.text, host.label, host.marker, host.any_case) == (
        "host", "relay.example", "relay_url.host", "<redacted:relay_url.host>", True)


@pytest.mark.parametrize("value", ["https://gateway:8443/v1", "https://localhost/v1", "http://127.1:8080",
                                   "https://10.0.0.7/v1", "https://[::1]:8443/v1"])
def test_a_url_whose_host_identifies_nothing_has_its_value_alone_as_a_form(value):
    assert forms_of("gateway_url", value) == [SecretForm(name="gateway_url", form="value", text=value)]


def test_a_short_route_host_never_rewrites_a_route_name_in_a_record():
    """With ``gateway_url = "https://gateway:8443/v1"`` every "gateway" in a build record, a route's name
    and its credential's among them, would become a redaction and the record would no longer
    validate. The URL is still replaced whole."""
    scrub = Scrubber({"gateway_url": "https://gateway:8443/v1", "gateway_token": "sk-gateway-0123456789"})
    record = json.dumps({"route": "gateway", "route_table": {"credential": "gateway_token", "base_url": "gateway_url"},
                         "dialled": "https://gateway:8443/v1", "Gateway": "GATEWAY"})
    assert json.loads(scrub.scrub(record)) == {
        "route": "gateway", "route_table": {"credential": "gateway_token", "base_url": "gateway_url"},
        "dialled": "<redacted:gateway_url>", "Gateway": "GATEWAY"}


# --- backing off a cut ----------------------------------------------------------------------------

#: A route's key and its relay's base URL, whose host is a form too.
CUT_FORMS = forms_of("anthropic_key", "sk-ant-api03-0123456789") + forms_of("relay_url", "https://relay.example/v1")


@pytest.mark.parametrize("text, side, kept", [
    ("out: sk-ant-api0", "end", "out: "),
    ("3-0123456789 more", "start", " more"),
    ("see RELAY.EXA", "end", "see "),
    ("ample/v1 and on", "start", " and on"),
    ("out: sk-ant-api03-0123456789", "end", "out: sk-ant-api03-0123456789"),
    ("sk-ant-api03-0123456789 rest", "start", "sk-ant-api03-0123456789 rest"),
    ("plain text", "end", "plain text"),
    ("", "start", ""),
], ids=["value-at-a-head", "value-at-a-tail", "host-any-case", "url-at-a-tail", "whole-value-at-a-head",
        "whole-value-at-a-tail", "nothing-cut", "empty"])
def test_a_cut_backs_off_so_no_form_is_left_cut_through(text, side, kept):
    """A text cut at its end, a head, or at its start, a tail, loses the piece at the cut that may
    be part of a secret's form the cut went through: a whole form stays, for the scrubber to
    replace, and so does text no form starts or ends there."""
    assert back_off(text, CUT_FORMS, side=side) == kept
    assert back_off(text.encode(), CUT_FORMS, side=side) == kept.encode()


def test_backing_off_goes_on_until_no_piece_of_a_form_is_left_at_the_cut():
    """Dropping one form's piece may leave another's: the text ends in ``abcd``, a whole form, whose
    ``cd`` starts another form; with ``cd`` gone, ``ab`` starts the first. Both go."""
    forms = forms_of("one", "abcd") + forms_of("two", "cdxyz")
    assert back_off("..abcd", forms, side="end") == ".."
    assert back_off("xyzab..", forms_of("one", "abxyz") + forms_of("two", "zzxyzab"), side="start") == ".."
