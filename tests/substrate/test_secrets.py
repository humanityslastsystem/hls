from __future__ import annotations

import os
import subprocess
import threading
import tomllib
import traceback
from pathlib import Path

import pytest

from hls.substrate import secrets as secrets_mod
from hls.substrate.secret_forms import SCRUB_CHUNK_BYTES
from hls.substrate.secrets import (MIN_SECRET_LENGTH, Scrubber, SecretsError, SecretsFile, grant_names,
                                   load_secrets, parse_env_text, read_secrets, resolve_grants, write_env_file)

GOOD = """\
[secrets]
anthropic_relay = "relay-token-0123456789"
gateway_token = "gateway-token-abcdefgh"
hf_token = "hf_abcdefghijklmnop"
ungranted_token = "ungranted-token-zzzzzz"

[grants]
"stage.provision".HF_TOKEN = "hf_token"
"stage.images".REGISTRY_TOKEN = "gateway_token"
"""


def _file(tmp_path: Path, text: str = GOOD, mode: int = 0o600) -> Path:
    path = tmp_path / "secrets.toml"
    path.write_text(text)
    path.chmod(mode)
    return path


def test_a_private_file_loads_with_its_grants(tmp_path):
    sf = load_secrets(_file(tmp_path))
    assert sf.secrets["hf_token"] == "hf_abcdefghijklmnop"
    assert grant_names(sf, "stage.provision") == {"HF_TOKEN": "hf_token"}
    assert grant_names(sf, "stage.images") == {"REGISTRY_TOKEN": "gateway_token"}
    assert grant_names(sf, "stage.verify") == {}
    assert load_secrets(_file(tmp_path, mode=0o400)) == sf


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o660, 0o644, 0o700 | 0o001])
def test_a_file_others_can_read_or_write_is_refused(tmp_path, mode):
    with pytest.raises(SecretsError, match="mode"):
        load_secrets(_file(tmp_path, mode=mode))


def test_a_symlink_is_refused_even_to_a_private_file(tmp_path):
    target = _file(tmp_path)
    link = tmp_path / "link.toml"
    link.symlink_to(target)
    with pytest.raises(SecretsError, match="regular file"):
        load_secrets(link)


def test_a_directory_or_a_missing_file_is_refused(tmp_path):
    with pytest.raises(SecretsError, match="regular file"):
        load_secrets(tmp_path)
    with pytest.raises(SecretsError, match="does not exist"):
        load_secrets(tmp_path / "absent.toml")


def test_a_file_owned_by_someone_else_is_refused(tmp_path, monkeypatch):
    path = _file(tmp_path)
    monkeypatch.setattr(secrets_mod.os, "getuid", lambda: os.stat(path).st_uid + 1)
    with pytest.raises(SecretsError, match="owned by uid"):
        load_secrets(path)


@pytest.mark.parametrize("text, match", [
    ('[secrets]\nhf_token = "short"\n', f"at least {MIN_SECRET_LENGTH}"),
    ('[secrets]\nhf_token = "has\\"quote-in-it"\n', "printable ASCII"),
    ('[secrets]\nhf_token = "has-a-\\\\backslash-in-it"\n', "printable ASCII"),
    ('[secrets]\nhf_token = "tab\\tinside-value"\n', "printable ASCII"),
    ('[secrets]\nhf_token = "non-ascii-\u00e9-value"\n', "printable ASCII"),
    ('[secrets]\nhf_token = "has-a-$dollar-in-it"\n', "printable ASCII"),
    ('[secrets]\nhf_token = "has-a-`tick`-in-it"\n', "printable ASCII"),
    ('[secrets]\nHF = "long-enough-value"\n', "secret name"),
    ('[secrets]\nhf_token = "long-enough-value"\n[grants]\n"stage.provision".HF_TOKEN = "nope"\n',
     r"HF_TOKEN names no secret of \[secrets\]"),
    ('[secrets]\nhf_token = "long-enough-value"\n[grants]\n"provision".HF_TOKEN = "hf_token"\n', "grantee"),
    ('[secrets]\nhf_token = "long-enough-value"\n[grants]\n"stage.agent-build".HF_TOKEN = "hf_token"\n',
     "grantee"),
    ('[secrets]\nhf_token = "long-enough-value"\n[grants]\n"agent.claude-code".X_TOKEN = "hf_token"\n',
     r"\[arms\.<name>\]\.route"),
    ('[secrets]\nhf_token = "long-enough-value"\n[grants]\n"agent.claude-code@k3".X_TOKEN = "hf_token"\n',
     "agents are granted their arm's route"),
    ('[secrets]\nhf_token = "long-enough-value"\n[grants]\n"stage.provision"."1BAD" = "hf_token"\n',
     "variable name"),
    ('[secrets]\nhf_token = "long-enough-value"\n[env]\nX = "y"\n', "env"),
    ('[secrets]\nhf_token = 12345678\n', "string"),
    ("not toml = = \n", "TOML"),
])
def test_a_malformed_file_is_refused(tmp_path, text, match):
    with pytest.raises(SecretsError, match=match):
        load_secrets(_file(tmp_path, text))


def test_a_swapped_in_symlink_is_refused_at_open(tmp_path, monkeypatch):
    # The path is a regular private file when it is checked and a link when it is opened.
    path = _file(tmp_path)
    other = tmp_path / "other.toml"
    other.write_text(GOOD)
    other.chmod(0o600)
    real_open = os.open

    def swapping_open(p, flags, *args, **kwargs):
        if Path(p) == path:
            path.unlink()
            path.symlink_to(other)
        return real_open(p, flags, *args, **kwargs)

    monkeypatch.setattr(secrets_mod.os, "open", swapping_open)
    with pytest.raises(SecretsError, match="regular file"):
        load_secrets(path)


def test_a_file_that_is_not_utf8_is_refused_without_its_bytes(tmp_path):
    path = _file(tmp_path)
    path.write_bytes(b'[secrets]\nhf_token = "\xff\xfe-latin-bytes-0123"\n')
    with pytest.raises(SecretsError, match="UTF-8") as info:
        load_secrets(path)
    assert info.value.__cause__ is None and info.value.__context__ is None
    assert "xff" not in str(info.value)


@pytest.mark.parametrize("text, fragment", [
    ('[secrets]\nhf_token = "short7x"\n', "short7x"),
    ('[secrets]\nhf_token = "long-enough-value"\n[grants]\n"stage.provision".HF_TOKEN = "nope"\n',
     "long-enough-value"),
    ('[secrets]\nhf_token = "long-enough-value" junk\n', "long-enough-value"),
    ('[secrets]\nhf_token = "long-enough-value"\n[env]\nX = "y"\n', "long-enough-value"),
])
def test_an_error_carries_neither_a_value_nor_the_error_it_came_from(tmp_path, text, fragment):
    with pytest.raises(SecretsError) as info:
        load_secrets(_file(tmp_path, text))
    err = info.value
    # Not even as context, which a logger that walks the chain would print.
    assert err.__cause__ is None and err.__context__ is None
    assert fragment not in "".join(traceback.format_exception(err))


#: A relay's base URL, and a route's URL under it.
RELAY_BASE = "https://relay.invalid/v1-0123456789"


def test_a_file_that_holds_one_secret_inside_another_is_refused_naming_both_and_neither_value(tmp_path):
    # A file of the contestant's that holds its own route's URL holds the bare base too, which the
    # registry names another secret: judged by value, its own URL would refuse the instance.
    text = f'[secrets]\nrelay_base = "{RELAY_BASE}"\nanthropic_url = "{RELAY_BASE}/anthropic"\n'
    with pytest.raises(SecretsError) as info:
        load_secrets(_file(tmp_path, text))
    assert str(info.value).endswith("secret anthropic_url's value contains secret relay_base's; a site file must "
                                    "not hold one secret inside another")
    assert RELAY_BASE not in "".join(traceback.format_exception(info.value))


def test_two_secrets_with_one_value_are_one_secret_and_load():
    # One relay URL may serve two routes; a scan judges them as one value.
    sf = SecretsFile(secrets={"route_a_url": RELAY_BASE, "route_b_url": RELAY_BASE})
    assert sf.secrets == {"route_a_url": RELAY_BASE, "route_b_url": RELAY_BASE}


#: A fake hub token, pasted where the name of its secret belongs.
PASTED = "hf_FAKEVALUE1234567890"


@pytest.mark.parametrize("right, says", [
    (PASTED, "the right-hand side must be the name of a secret in [secrets]"),
    ("hf-token", "the right-hand side must be the name of a secret in [secrets]"),
    ("no_such_secret", "names no secret of [secrets]"),
])
def test_a_grant_whose_right_hand_side_is_no_secret_name_names_only_the_grant(tmp_path, right, says):
    # An operator who reads a grant as "variable = value" pastes the token on the right.
    text = f'[secrets]\nhf_token = "{PASTED}"\n[grants]\n"stage.provision".HF_TOKEN = "{right}"\n'
    with pytest.raises(SecretsError, match="grant stage.provision.HF_TOKEN") as info:
        load_secrets(_file(tmp_path, text))
    assert says in str(info.value)
    shown = "".join(traceback.format_exception(info.value))
    assert right not in shown and PASTED not in shown


#: A registry user name that is also the cluster login name.
LOGIN_NAME = "loginuser"


@pytest.mark.parametrize("var", ["REGISTRY_USERNAME", "DOCKER_USERNAME"])
@pytest.mark.parametrize("grantee", ["stage.images", "stage.provision"])
def test_a_grant_of_the_registry_user_name_is_refused_pointing_to_its_setting(tmp_path, var, grantee):
    text = (f'[secrets]\nregistry_user = "{LOGIN_NAME}"\nregistry_token = "dckr_pat_0123456789"\n'
            f'[grants]\n"{grantee}".{var} = "registry_user"\n"stage.images".REGISTRY_PASSWORD = "registry_token"\n')
    with pytest.raises(SecretsError, match=f"grant {grantee}.{var}: ") as info:
        load_secrets(_file(tmp_path, text))
    assert "site.registry_username in hls.toml" in str(info.value)
    shown = "".join(traceback.format_exception(info.value))
    assert LOGIN_NAME not in shown and "dckr_pat_0123456789" not in shown


@pytest.mark.parametrize("var", ["REGISTRY_USERNAME", "DOCKER_USERNAME"])
def test_a_registry_user_name_granted_to_an_agent_is_refused_as_an_agent_grant(tmp_path, var):
    # An agent is granted only its arm's route, so the file refuses the grantee before any of
    # its variables, and shows neither value.
    text = (f'[secrets]\nregistry_user = "{LOGIN_NAME}"\nregistry_token = "dckr_pat_0123456789"\n'
            f'[grants]\n"agent.claude-code".{var} = "registry_user"\n'
            '"stage.images".REGISTRY_PASSWORD = "registry_token"\n')
    with pytest.raises(SecretsError, match="agents are granted their arm's route") as info:
        load_secrets(_file(tmp_path, text))
    shown = "".join(traceback.format_exception(info.value))
    assert LOGIN_NAME not in shown and "dckr_pat_0123456789" not in shown


def test_the_registry_token_alone_is_granted_as_a_secret(tmp_path):
    text = ('[secrets]\nregistry_token = "dckr_pat_0123456789"\n'
            '[grants]\n"stage.images".REGISTRY_PASSWORD = "registry_token"\n')
    sf = load_secrets(_file(tmp_path, text))
    assert grant_names(sf, "stage.images") == {"REGISTRY_PASSWORD": "registry_token"}


@pytest.mark.parametrize("text", [
    # Nested deeper than the parser can recurse.
    "x = " + "[" * 2000 + "]" * 2000 + "\n",
    # An integer with more digits than int() converts.
    "x = " + "1" * 5000 + "\n",
], ids=["nested", "digits"])
def test_a_file_the_parser_cannot_take_is_refused_like_any_malformed_file(tmp_path, text):
    with pytest.raises(SecretsError, match="not valid TOML") as info:
        load_secrets(_file(tmp_path, GOOD + text))
    err = info.value
    assert str(err).startswith(f"{tmp_path / 'secrets.toml'}: ")
    assert err.__cause__ is None and err.__context__ is None
    assert "hf_abcdefghijklmnop" not in "".join(traceback.format_exception(err))


def test_any_other_error_of_the_parser_is_refused_without_its_message(tmp_path, monkeypatch):
    def broken(text):
        raise RuntimeError(f"the parser quoted {text[-30:]}")

    monkeypatch.setattr(tomllib, "loads", broken)
    with pytest.raises(SecretsError, match="not valid TOML") as info:
        load_secrets(_file(tmp_path))
    assert "relay-token" not in str(info.value) and info.value.__context__ is None


def test_the_parsers_message_is_not_repeated_since_it_can_quote_a_value(tmp_path):
    # tomllib names an illegal character of a string, which is a character of the value.
    with pytest.raises(SecretsError, match=r"not valid TOML \(at line 2, column \d+\)$") as info:
        load_secrets(_file(tmp_path, '[secrets]\nhf_token = "fake-value-\x01-0123"\n'))
    assert "\\x01" not in str(info.value) and "Illegal" not in str(info.value)


def test_a_named_pipe_at_the_secrets_path_is_refused_without_waiting_for_a_writer(tmp_path):
    fifo = tmp_path / "secrets.toml"
    os.mkfifo(fifo, 0o600)
    result: list[BaseException] = []
    reader = threading.Thread(target=lambda: _catch(result, lambda: load_secrets(fifo)), daemon=True)
    reader.start()
    reader.join(10)
    if reader.is_alive():
        # Opening the write end lets the stuck open return, so the thread ends.
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        reader.join(10)
        pytest.fail("opening a named pipe at the secrets path waited for a writer")
    [error] = result
    assert isinstance(error, SecretsError) and "regular file" in str(error)


def _catch(into: list[BaseException], call) -> None:
    try:
        call()
    except BaseException as e:
        into.append(e)


def test_a_loaded_file_does_not_show_its_values(tmp_path):
    sf = load_secrets(_file(tmp_path))
    for text in (repr(sf), str(sf)):
        assert "hf_abcdefghijklmnop" not in text and "relay-token-0123456789" not in text
    fake = SecretsFile(secrets={"hf_token": "hf_abcdefghijklmnop"}, grants={})
    assert "hf_abcdefghijklmnop" not in repr(fake) and "hf_abcdefghijklmnop" not in str(fake)


def test_resolving_grants_gives_values_and_names_an_unknown_secret():
    sf = SecretsFile(secrets={"hf_token": "hf_abcdefghijklmnop"}, grants={})
    assert resolve_grants(sf, {"HF_TOKEN": "hf_token"}) == {"HF_TOKEN": "hf_abcdefghijklmnop"}
    with pytest.raises(SecretsError, match="gateway_token"):
        resolve_grants(sf, {"ANTHROPIC_AUTH_TOKEN": "gateway_token"})


def test_the_scrubber_replaces_longer_values_first():
    scrub = Scrubber({"inner": "abcdefgh", "outer": "xxabcdefghyy"})
    assert scrub.scrub("a xxabcdefghyy b abcdefgh") == "a <redacted:outer> b <redacted:inner>"
    assert scrub.scrub_bytes(b"k=abcdefgh\xff") == b"k=<redacted:inner>\xff"


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 8, 11, 13, 64])
@pytest.mark.parametrize("offset", range(-13, 3))
def test_a_file_is_scrubbed_in_chunks_also_where_a_value_straddles_a_chunk_boundary(tmp_path, chunk, offset):
    scrub = Scrubber({"inner": "abcdefgh", "outer": "xxabcdefghyy"})
    data = (b"." * max(0, 64 + offset) + b"xxabcdefghyy" + b"," * 7 + b"abcdefgh" + b"\nxxabcdefgh"
            + b"abcdefghabcdefgh" + b"\xff tail abcdefg")
    path = tmp_path / "role.out"
    path.write_bytes(data)
    scrub.scrub_file(path, chunk_bytes=chunk)
    assert path.read_bytes() == scrub.scrub_bytes(data)
    assert b"abcdefgh" not in path.read_bytes() and path.read_bytes().endswith(b"\xff tail abcdefg")
    assert [p.name for p in tmp_path.iterdir()] == ["role.out"]


def test_a_file_scrubbed_to_another_path_leaves_the_original_as_it_was(tmp_path):
    src = tmp_path / "role.out"
    src.write_bytes(b"a abcdefgh b")
    Scrubber({"x": "abcdefgh"}).scrub_file(src, to=tmp_path / "published.out")
    assert src.read_bytes() == b"a abcdefgh b" and (tmp_path / "published.out").read_bytes() == b"a <redacted:x> b"
    # A scrubber that knows no value still makes the copy.
    Scrubber({}).scrub_file(src, to=tmp_path / "plain.out")
    assert (tmp_path / "plain.out").read_bytes() == src.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plain.out", "published.out", "role.out"]


def test_a_value_split_at_the_default_chunk_is_scrubbed(tmp_path):
    scrub = Scrubber({"hf_token": "hf-token-0123456789"})
    path = tmp_path / "role.err"
    path.write_bytes(b"x" * (SCRUB_CHUNK_BYTES - 5) + b"hf-token-0123456789\n")
    scrub.scrub_file(path)
    assert path.read_bytes() == b"x" * (SCRUB_CHUNK_BYTES - 5) + b"<redacted:hf_token>\n"


#: A credential, a route's base URL and a mirror whose host is an IPv4 address written in hex.
SPLIT_REGISTRY = {"relay_key": "sk-test-0123456789abcdef", "relay_url": "https://Relay.Example:8443/v1/anthropic",
                  "mirror_url": "https://0x8.8.4.4:8443/v1"}
#: Each form a file may hold, with the label its marker names: a value, a URL, its host in another
#: case, and an address host as written and as a client prints it.
SPLIT_FORMS = [(SPLIT_REGISTRY["relay_key"], "relay_key"), (SPLIT_REGISTRY["relay_url"], "relay_url"),
               ("RELAY.example", "relay_url.host"), (SPLIT_REGISTRY["mirror_url"], "mirror_url"),
               ("0x8.8.4.4", "mirror_url.host"), ("8.8.4.4", "mirror_url.host")]


@pytest.mark.parametrize("text, label", SPLIT_FORMS)
def test_every_form_split_at_the_default_chunk_is_replaced_whole(tmp_path, text, label):
    """Each form straddles the default read boundary at every position, a URL split after its host
    among them, and comes out whole as its marker."""
    scrub = Scrubber(SPLIT_REGISTRY)
    path = tmp_path / "role.out"
    for k in range(1, len(text)):
        head = b"x" * (SCRUB_CHUNK_BYTES - k)
        path.write_bytes(head + text.encode() + b" tail\n")
        scrub.scrub_file(path)
        assert path.read_bytes() == head + f"<redacted:{label}>".encode() + b" tail\n", k


def test_the_scrubber_covers_every_secret_whatever_grants_it(tmp_path):
    scrub = Scrubber.from_secrets(load_secrets(_file(tmp_path)))
    text = "relay-token-0123456789 gateway-token-abcdefgh hf_abcdefghijklmnop ungranted-token-zzzzzz"
    assert scrub.scrub(text) == ("<redacted:anthropic_relay> <redacted:gateway_token> <redacted:hf_token> "
                                 "<redacted:ungranted_token>")


def read_env_file(path: Path) -> dict[str, str]:
    """The variables in the env file at *path*."""
    return parse_env_text(path.read_text(encoding="utf-8"))


def test_the_module_reads_no_env_file_itself():
    # Only the step wrapper reads an env file, from the pipe it moved the file into.
    assert not hasattr(secrets_mod, "read_env_file")


def test_an_env_file_is_private_sorted_and_quoted(tmp_path):
    path = tmp_path / "role.env"
    write_env_file(path, {"B": "two words", "A": "it's", "C": ""})
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert path.read_text() == "A='it'\"'\"'s'\nB='two words'\nC=''\n"
    assert read_env_file(path) == {"A": "it's", "B": "two words", "C": ""}


def test_a_posix_shell_reads_an_env_file_back_unchanged(tmp_path):
    # apptainer evaluates --env-file with a POSIX shell interpreter on the host.
    env = {"TOKEN": "abc;rm -rf /;`x`", "SPACES": " a  b ", "QUOTES": "it's \"q\""}
    path = tmp_path / "role.env"
    write_env_file(path, env)
    out = subprocess.run(["/bin/sh", "-c", f'set -a; . "{path}"; env -0'], capture_output=True, check=True,
                         env={"PATH": "/usr/bin:/bin"}).stdout
    seen = dict(item.split("=", 1) for item in out.decode().split("\0") if item)
    assert {k: seen[k] for k in env} == env


def test_an_env_file_is_never_written_over_or_through_a_link(tmp_path):
    path = tmp_path / "role.env"
    write_env_file(path, {"A": "1"})
    with pytest.raises(FileExistsError):
        write_env_file(path, {"A": "2"})
    link = tmp_path / "link.env"
    link.symlink_to(tmp_path / "elsewhere.env")
    with pytest.raises(FileExistsError):
        write_env_file(link, {"A": "1"})
    assert not (tmp_path / "elsewhere.env").exists()


def _refused_as_a_secret(char: str) -> bool:
    try:
        SecretsFile(secrets={"s": f"abcdefgh{char}"}, grants={})
    except ValueError:
        return True
    return False


def test_every_value_the_file_accepts_is_one_a_shell_reads_back(tmp_path):
    # One rule covers loading and writing: nothing a secret may hold is refused by the env file.
    accepted = [c for c in map(chr, range(0x20, 0x7F)) if not _refused_as_a_secret(c)]
    assert " " in accepted and "'" in accepted and "$" not in accepted and "`" not in accepted
    value = "abcdefgh" + "".join(accepted)
    path = tmp_path / "role.env"
    write_env_file(path, {"TOKEN": value})
    out = subprocess.run(["/bin/sh", "-c", f'set -a; . "{path}"; env -0'], capture_output=True, check=True,
                         env={"PATH": "/usr/bin:/bin"}).stdout
    seen = dict(item.split("=", 1) for item in out.decode().split("\0") if item)
    assert seen["TOKEN"] == value and read_env_file(path) == {"TOKEN": value}


@pytest.mark.parametrize("env, match", [
    ({"1A": "x"}, "variable name"),
    ({"A-B": "x"}, "variable name"),
    ({"A": "line\nbreak"}, "control character"),
    ({"A": "nul\0byte"}, "control character"),
    ({"A": "cr\rreturn"}, "control character"),
    ({"A": "tab\tbed"}, "control character"),
    ({"A": "vt\x0bform"}, "control character"),
    ({"A": "ff\x0cfeed"}, "control character"),
    ({"A": "fs\x1cseparator"}, "control character"),
    ({"A": "esc\x1b[0m"}, "control character"),
    ({"A": "del\x7fchar"}, "control character"),
    ({"PATH": "/opt/x/bin:$PATH"}, r"\$"),
])
def test_an_env_file_refuses_what_a_shell_would_change(tmp_path, env, match):
    path = tmp_path / "role.env"
    with pytest.raises(ValueError, match=match):
        write_env_file(path, env)
    assert not path.exists()


def test_parse_env_text_refuses_a_line_that_is_not_one_assignment():
    assert parse_env_text("A='x'\n\nB=y\n") == {"A": "x", "B": "y"}
    with pytest.raises(ValueError, match="line 1"):
        parse_env_text("A='x' B='y'\n")
    with pytest.raises(ValueError, match="line 1"):
        parse_env_text("not an assignment\n")


def test_parse_env_text_names_the_line_of_an_unclosed_quote_without_its_text():
    with pytest.raises(ValueError, match="line 2") as info:
        parse_env_text("A='x'\nB='fake-value-0123\n")
    assert "fake-value" not in str(info.value) and info.value.__context__ is None


def test_parse_env_text_splits_lines_on_newline_only():
    # str.splitlines would also break at these, inside a quoted value.
    text = "A='x\u2028y'\nB='p\x85q'\nC='r\x1cs'\n"
    assert parse_env_text(text) == {"A": "x\u2028y", "B": "p\x85q", "C": "r\x1cs"}


def test_the_documented_example_loads(tmp_path):
    """docs/examples/secrets.toml is the file an operator copies; it must load as written."""
    example = Path(__file__).resolve().parents[2] / "docs" / "examples" / "secrets.toml"
    sf = load_secrets(_file(tmp_path, example.read_text()))
    # The registry login is left out, as the example hls.toml leaves out its user name.
    assert set(sf.grants) == {"stage.provision"}
    # It shows a route's base URL, which is a secret like its credential.
    assert sf.secrets["relay_url"].startswith("https://")


#: A base URL, a secret like the credential, and its host.
BASE_URL = "https://relay.example:8443/v1/anthropic"
HOST = "relay.example"


def test_a_url_secrets_host_is_redacted_but_not_held():
    """A route's base URL is a secret, and so is its host: a record line holding the host alone
    comes out scrubbed. The host alone gives no access, so ``holds`` is true only for a whole
    value, which lets the seal's policy argument, which names the host, reach a container."""
    scrub = Scrubber({"relay_url": BASE_URL, "relay_key": "sk-test-0123456789"})
    line = f"dialled {HOST}:8443 and {BASE_URL}"
    assert scrub.scrub(line) == "dialled <redacted:relay_url.host>:8443 and <redacted:relay_url>"
    assert scrub.scrub_bytes(line.encode()) == scrub.scrub(line).encode()
    assert not scrub.holds(f'{{"domains": {{"entries": ["{HOST}"]}}}}')
    assert scrub.holds(f"--url={BASE_URL}") and scrub.holds("key sk-test-0123456789")
    assert not scrub.holds("nothing here")


@pytest.mark.parametrize("printed", ["RELAY.EXAMPLE", "Relay.Example"])
def test_a_host_printed_in_mixed_case_is_scrubbed(tmp_path, printed):
    scrub = Scrubber({"relay_url": BASE_URL})
    assert scrub.scrub(f"to {printed}/v1") == "to <redacted:relay_url.host>/v1"
    assert scrub.scrub_bytes(f"to {printed}".encode()) == b"to <redacted:relay_url.host>"
    src = tmp_path / "log.out"
    src.write_text(("x" * 7 + printed) * 20)
    for chunk in (1, 3, len(HOST) - 1, len(HOST), 64):
        scrub.scrub_file(src, to=tmp_path / "scrubbed.out", chunk_bytes=chunk)
        text = (tmp_path / "scrubbed.out").read_text()
        assert printed not in text and text.count("<redacted:relay_url.host>") == 20, chunk


def test_a_value_that_is_no_http_url_has_no_host_to_scrub():
    scrub = Scrubber({"token": "relay.example-0123456789", "path": "ftp://relay.example/x"})
    assert scrub.scrub("relay.example") == "relay.example"


def test_a_secrets_file_is_read_scrubbed_by_or_found_missing_only_where_it_can_be_looked_at(tmp_path):
    """A path under a directory this user may not search is no missing file: taking it for one
    would scrub nothing."""
    assert read_secrets(tmp_path / "absent.toml") is None
    assert Scrubber.from_path(tmp_path / "absent.toml").scrub("hf_abcdefghijklmnop") == "hf_abcdefghijklmnop"
    assert Scrubber.from_path(_file(tmp_path)).scrub("hf_abcdefghijklmnop") == "<redacted:hf_token>"
    shut = tmp_path / "shut"
    shut.mkdir()
    hidden = _file(shut)
    shut.chmod(0o000)
    try:
        for read in (read_secrets, Scrubber.from_path):
            with pytest.raises(SecretsError, match=f"{hidden}: the secrets path cannot be looked at: PermissionError"):
                read(hidden)
    finally:
        shut.chmod(0o700)



def test_a_scrubbed_logs_cut_leaves_no_value_cut_through_on_either_side(tmp_path):
    """The log capper keeps a log's first and last bytes and says, on a line between them, how many
    it dropped. A value the cut went through leaves a piece on either side of that line, which a
    scrub of whole values cannot find: the scrub of a capped log drops each, and the line counts
    them among the dropped."""
    from hls.substrate.log_cap import dropped_line

    value = "hf-token-0123456789"
    head = b"x" * 100 + value[:7].encode()
    tail = value[-6:].encode() + b"z" * 100
    log = tmp_path / "role.err"
    log.write_bytes(head + dropped_line(1000, len(head)) + tail)
    Scrubber({"HF_TOKEN": value}).scrub_file(log, cut=True, chunk_bytes=64)
    assert log.read_bytes() == b"x" * 100 + dropped_line(1000 + 7 + 6, len(head)) + b"z" * 100
    plain = tmp_path / "plain.err"
    plain.write_bytes(head + dropped_line(1000, len(head)) + tail)
    Scrubber({"HF_TOKEN": value}).scrub_file(plain, chunk_bytes=64)
    assert plain.read_bytes() == head + dropped_line(1000, len(head)) + tail
