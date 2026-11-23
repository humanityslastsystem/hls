"""The cluster-local secrets file, role-scoped grants, env files and the scrubber.

The secrets file lives at ``site.secrets_path``, outside the checkout and the store, which
harness roles mount (the planner and the runner refuse a path inside either,
``secrets_path_problem``), and only its owner may read it::

    [secrets]
    hf_token    = "..."
    relay_token = "..."
    relay_url   = "https://relay.example/v1"

    [grants]
    "stage.provision".HF_TOKEN = "hf_token"

A grant gives one stage one environment variable holding one named secret. Grants are
default deny: a secret nobody is granted reaches nothing. An agent is granted only its arm's
route: hls.toml's ``[routes.<name>]`` names the secrets that hold the route's credential and
its base URL, which is a secret as much as the credential is, and ``[arms.<name>].route``
names the route. So the file grants stages only, and a stage whose parameters declare its
grants, as an agent build does from its arm's route, takes those and none from the file. A
registry user name is no secret (``REGISTRY_USER_GRANTS``): a file that grants one is refused,
pointing to the plain setting ``site.registry_username``. The planner records grant names
only; the runner resolves values when a stage starts. A value reaches a role only through an
env file: node-local, mode 0600, created exclusively, and unlinked by the step wrapper before
the role starts. Values never go into argv, the store, the plan or a record.

Every value is printable ASCII without a quote, a backslash, a ``$`` or a backtick, so it
reads the same in every JSON encoder, the scrubber's plain substring match finds it in any
record, and an env file holds it without a shell changing it. Every value is at least
``MIN_SECRET_LENGTH`` characters, so scrubbing never rewrites ordinary words: a shorter
value is refused rather than left unscrubbed. A value that is an http(s) URL, a route's base
URL, is a secret with its host when the host identifies a machine: the scrubber replaces each
secret's forms (``hls.substrate.secret_forms``), the host in any case, since a record that names
the host alone still says where the route goes; a host such as ``relay``,
``localhost`` or a private address names no machine and stays, so a route's name is never
rewritten. No value lies inside another secret's, though two secrets may hold the same one:
finalize's scan judges what it finds by value, and a value inside another would be found a second
time, under its own name, in every file that holds the outer one. That rule covers a value inside a
host too, since a URL's host lies inside the URL. A host may lie inside another secret's value: a
host found in a contestant's file only quarantines it."""

from __future__ import annotations

import errno
import os
import re
import shlex
import stat
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hls.substrate.apptainer.images import REGISTRY_TOKEN_GRANT, REGISTRY_USER_VAR
from hls.substrate import log_cap
from hls.substrate.secret_forms import SCRUB_CHUNK_BYTES, SECRET_NAME, ByteScrubber, back_off, registry_forms
from hls.util import TomlError, load_toml

#: Variables no grant may set: a registry user name, which is not a secret. It is often the
#: cluster login name, which every path under the user's directories holds, and as a value of
#: this file it would refuse every role whose arguments name such a path. ``DOCKER_USERNAME`` is
#: a common name for it.
REGISTRY_USER_GRANTS = frozenset({REGISTRY_USER_VAR, "DOCKER_USERNAME"})
#: Shortest secret value the file may hold.
MIN_SECRET_LENGTH = 8
#: An environment variable name.
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: A grantee: a stage by name.
GRANTEE = re.compile(r"stage\.[a-z][a-z0-9_]*")
#: Why a grant to an agent is refused.
AGENT_GRANTEE = ("agents are granted their arm's route: declare it in [routes.<name>] and name it in "
                 "[arms.<name>].route in hls.toml, not in [grants]")
# Control characters: they end an env file's line early or change how a shell reads it.
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class SecretsError(ValueError):
    """The secrets file, or a grant, is unusable."""


def _transparent(value: str) -> bool:
    """Whether *value* is printable ASCII without a character that a JSON encoder or a shell
    would write differently: a quote, a backslash, a ``$`` or a backtick."""
    return all(0x20 <= ord(c) <= 0x7E and c not in '"\\$`' for c in value)


class SecretsFile(BaseModel):
    """The parsed secrets file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Secret name to value. Left out of the model's repr, which would print the values.
    secrets: dict[str, str] = Field(repr=False)
    #: Grantee, ``stage.<name>``, to environment variable to secret name.
    grants: dict[str, dict[str, str]] = Field(default_factory=dict)

    @field_validator("secrets")
    @classmethod
    def _values(cls, secrets: dict[str, str]) -> dict[str, str]:
        for name, value in secrets.items():
            if not SECRET_NAME.fullmatch(name):
                raise ValueError(f"secret name {name!r} must match {SECRET_NAME.pattern}")
            if not _transparent(value):
                raise ValueError(f"secret {name!r} must be printable ASCII without a quote, a backslash, a $ "
                                 "or a backtick")
            if len(value) < MIN_SECRET_LENGTH:
                raise ValueError(f"secret {name!r} must be at least {MIN_SECRET_LENGTH} characters, so scrubbing "
                                 "it cannot rewrite ordinary text")
        # A scan judges what it finds by value (hls.stages.finalize.contestant_facing): two names
        # with one value are one secret, but a value inside another is a second find in every file
        # holding the outer one, which a grant of the outer one does not make the contestant's.
        for outer, inner in ((a, b) for a in sorted(secrets) for b in sorted(secrets)):
            if secrets[inner] != secrets[outer] and secrets[inner] in secrets[outer]:
                raise ValueError(f"secret {outer}'s value contains secret {inner}'s; a site file must not hold one "
                                 "secret inside another")
        return secrets

    @model_validator(mode="after")
    def _grants(self) -> SecretsFile:
        for grantee, env in self.grants.items():
            if grantee.startswith("agent."):
                raise ValueError(f"grantee {grantee!r}: {AGENT_GRANTEE}")
            if not GRANTEE.fullmatch(grantee):
                raise ValueError(f"grantee {grantee!r} must match {GRANTEE.pattern}")
            for var, name in env.items():
                if not ENV_NAME.fullmatch(var):
                    raise ValueError(f"grant {grantee}.{var}: variable name must match {ENV_NAME.pattern}")
                if var in REGISTRY_USER_GRANTS:
                    raise ValueError(f"grant {grantee}.{var}: the registry user name is not a secret; set it as "
                                     "site.registry_username in hls.toml, remove it from this file and grant only "
                                     f"the token, as {REGISTRY_TOKEN_GRANT}")
                # The right-hand side is never quoted: a value pasted there by mistake would be shown.
                if not SECRET_NAME.fullmatch(name):
                    raise ValueError(f"grant {grantee}.{var}: the right-hand side must be the name of a secret in "
                                     f"[secrets], matching {SECRET_NAME.pattern}")
                if name not in self.secrets:
                    raise ValueError(f"grant {grantee}.{var} names no secret of [secrets]")
        return self


def _read_private(path: Path) -> bytes:
    """The bytes of the secrets file at *path*. The file is opened without following a link
    and checked through the open descriptor, so the file that was checked is the file that
    is read."""
    try:
        # O_NONBLOCK keeps a named pipe at the path from hanging the open; it does nothing to a regular file.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except FileNotFoundError as e:
        raise SecretsError(f"{path}: the secrets file does not exist") from e
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SecretsError(f"{path}: the secrets file must be a regular file, not a link or a directory") from e
        raise SecretsError(f"{path}: the secrets file cannot be opened: {type(e).__name__}") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SecretsError(f"{path}: the secrets file must be a regular file, not a link or a directory")
        if st.st_uid != os.getuid():
            raise SecretsError(f"{path}: the secrets file is owned by uid {st.st_uid}, not this user")
        if st.st_mode & 0o077:
            raise SecretsError(f"{path}: the secrets file has mode {stat.S_IMODE(st.st_mode):o}; group and others "
                               "must have no access (chmod 600)")
        with os.fdopen(fd, "rb", closefd=False) as f:
            return f.read()
    finally:
        os.close(fd)


def load_secrets(path: Path) -> SecretsFile:
    """Read the secrets file at *path*. Refuse it unless it is a regular file, not a
    symlink, owned by this user, with no permission for group or others. Every way the file
    cannot be used is a SecretsError, whatever the parser raised, and none shows a value.

    Every error is raised outside the ``except`` block that caught what went wrong, with no
    cause, so the parser's or the validator's exception, which holds the file's text and the
    offending value, is neither printed with the error nor reachable from it."""
    data = _read_private(path)
    try:
        raw = load_toml(data)
    except TomlError as e:
        problem = str(e)
    else:
        problem = None
    if problem is not None:
        raise SecretsError(f"{path}: {problem}") from None
    try:
        return SecretsFile.model_validate(raw)
    except ValidationError as e:
        # The message names the field and the rule, never a value.
        problem = "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'file'}: {err['msg']}"
                            for err in e.errors(include_input=False))
    raise SecretsError(f"{path}: {problem}") from None


def read_secrets(path: Path) -> SecretsFile | None:
    """The secrets file at *path*; None when nothing is at the path. Anything else at the path
    that is not a private regular file is refused (``load_secrets``), a symlink included, so a
    dangling symlink is not taken for a missing file; so is a path that cannot be looked at, under
    a file or a directory this user may not search."""
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as e:
        raise SecretsError(f"{path}: the secrets path cannot be looked at: {type(e).__name__}") from e
    return load_secrets(path)


def secrets_path_problem(path: Path, *, checkout: Path, store_root: Path) -> str | None:
    """Why the secrets file may not live at *path*, or None: a path inside the checkout or the
    store, where a harness role that mounts either would read the file. Links in every path are
    followed first, so a link cannot hide where the file is."""
    where = path.resolve()
    for label, root in (("the checkout", checkout), ("the store", store_root)):
        if where.is_relative_to(root.resolve()):
            return (f"site.secrets_path {path} is in {label} {root}, which harness roles mount; the secrets file "
                    "belongs outside the checkout and the store")
    return None


def grant_names(secrets: SecretsFile, grantee: str) -> dict[str, str]:
    """The variables granted to *grantee*, each to the name of its secret; {} for none."""
    return dict(secrets.grants.get(grantee, {}))


def resolve_grants(secrets: SecretsFile, grants: Mapping[str, str]) -> dict[str, str]:
    """Each variable of *grants* (variable to secret name) with its secret's value."""
    missing = sorted(name for name in grants.values() if name not in secrets.secrets)
    if missing:
        raise SecretsError(f"the secrets file has no secret named {', '.join(map(repr, missing))}")
    return {var: secrets.secrets[name] for var, name in grants.items()}


class Scrubber:
    """Replaces every form of each known secret in text with its marker (``SecretForm.marker``):
    each value with ``<redacted:NAME>``, and the host of each value that is an http(s) URL, when it
    identifies a machine, in any case, with ``<redacted:NAME.host>`` (``registry_forms``)."""

    def __init__(self, values: Mapping[str, str]) -> None:
        # What replaces the forms in bytes, whole or streamed.
        self._bytes = ByteScrubber(registry_forms(values))
        # The forms, longest first, so a value that contains another is replaced whole: a URL
        # before its host.
        self._forms = self._bytes.forms
        # The whole values, which alone are what ``holds`` looks for.
        self._values = [form.text for form in self._forms if form.form == "value"]

    @classmethod
    def from_secrets(cls, secrets: SecretsFile) -> Scrubber:
        """A scrubber for every value in the file, whatever grants it."""
        return cls(secrets.secrets)

    @classmethod
    def from_path(cls, path: Path) -> Scrubber:
        """A scrubber for every value of the secrets file at *path*, which replaces nothing when no
        file is there. Raise SecretsError as ``read_secrets`` does."""
        secrets = read_secrets(path)
        return cls.from_secrets(secrets) if secrets is not None else cls({})

    def scrub(self, text: str) -> str:
        for form in self._forms:
            if form.any_case:
                text = re.sub(re.escape(form.text), lambda _m, m=form.marker: m, text, flags=re.IGNORECASE)
            else:
                text = text.replace(form.text, form.marker)
        return text

    def scrub_bytes(self, data: bytes) -> bytes:
        return self._bytes.scrub(data)

    def _backed_off(self, src: BinaryIO, chunk_bytes: int) -> Iterator[bytes]:
        """The bytes of the capped log open as *src*, *chunk_bytes* at a time, without the piece
        of a form on either side of each of its cuts (``log_cap.cuts``) that the cut may have gone
        through, at most one fewer byte each than the longest form has, and with each cut's line
        counting them among the dropped."""
        hold = self._bytes.hold
        at = 0
        for cut in log_cap.cuts(src, chunk_bytes=chunk_bytes):
            src.seek(max(at, cut.start - hold))
            before = src.read(cut.start - max(at, cut.start - hold))
            src.seek(cut.end)
            after = src.read(hold)
            head = len(before) - len(back_off(before, self._forms, side="end"))
            tail = len(after) - len(back_off(after, self._forms, side="start"))
            yield from _read(src, at, cut.start - head, chunk_bytes)
            yield log_cap.dropped_line(cut.dropped + head + tail, cut.keep)
            at = cut.end + tail
        yield from _read(src, at, None, chunk_bytes)

    def holds(self, text: str) -> bool:
        """Whether *text* holds a whole secret value. A URL's host alone is not one: it is
        scrubbed from records, but it gives no access, so a container's argument may name it,
        as the seal's policy names the route's host."""
        return any(value in text for value in self._values)

    def scrub_file(self, path: Path, *, to: Path | None = None, chunk_bytes: int = SCRUB_CHUNK_BYTES,
                   cut: bool = False) -> None:
        """Replace every form of every known secret in the file at *path*, read *chunk_bytes* at a
        time (``ByteScrubber.copy``, so a form split between two reads is replaced whole), and put
        the scrubbed file at *to*, or in the place of *path* when *to* is None; a reader of *to*
        sees all of it or nothing. With *cut*, the file is a capped log (``hls.substrate.log_cap``):
        at each of its cuts the piece of a form on either side that the cut may have gone through
        is left out (``back_off``), and the cut's line counts those bytes among the dropped."""
        to = path if to is None else to
        if not self._forms and to == path:
            return
        tmp = to.with_name(f".{to.name}.scrubbing")
        try:
            with open(path, "rb") as src, open(tmp, "wb") as dst:
                if cut:
                    # The pieces a capped log's cuts leave, each read as the copy asks for more.
                    pieces = (piece for piece in self._backed_off(src, chunk_bytes) if piece)
                    self._bytes.copy(lambda _size: next(pieces, b""), dst.write, chunk_bytes=chunk_bytes)
                else:
                    self._bytes.copy(src.read, dst.write, chunk_bytes=chunk_bytes)
            os.replace(tmp, to)
        finally:
            tmp.unlink(missing_ok=True)


def _read(src: BinaryIO, start: int, stop: int | None, chunk_bytes: int) -> Iterator[bytes]:
    """The bytes of *src* from offset *start* to *stop*, or to its end when None, *chunk_bytes*
    at a time."""
    src.seek(start)
    left = None if stop is None else stop - start
    while left is None or left > 0:
        chunk = src.read(chunk_bytes if left is None else min(chunk_bytes, left))
        if not chunk:
            return
        if left is not None:
            left -= len(chunk)
        yield chunk


def write_env_file(path: Path, env: Mapping[str, str]) -> None:
    """Create *path* with mode 0600, exclusively and not through a link, holding one
    ``NAME='value'`` line per variable, sorted by name. Raise FileExistsError if anything
    is at *path*, and ValueError for a bad name or a value a shell would change."""
    for name, value in env.items():
        if not ENV_NAME.fullmatch(name):
            raise ValueError(f"{name!r} is not a variable name")
        if _CONTROL.search(value):
            raise ValueError(f"the value of {name} holds a control character")
        if "$" in value:
            raise ValueError(f"the value of {name} holds a $, which an env file would expand on the host")
    text = "".join(f"{name}={shlex.quote(env[name])}\n" for name in sorted(env))
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def parse_env_text(text: str) -> dict[str, str]:
    """The variables of an env file's text, as ``write_env_file`` writes it: one assignment
    to a line, lines ending at a newline and nowhere else. A line that is not one assignment
    raises ValueError naming its number, never its text."""
    env = {}
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            words = shlex.split(line)
        except ValueError:
            words = []
        name, sep, value = words[0].partition("=") if len(words) == 1 else ("", "", "")
        if not sep or not ENV_NAME.fullmatch(name):
            raise ValueError(f"line {number} is not one NAME=value assignment")
        env[name] = value
    return env
