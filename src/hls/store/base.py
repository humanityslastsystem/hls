"""The store interface every backend implements.

Keys are relative ``/``-separated paths such as ``runs/<task>/<rollout>/plan.json``.
A key holds an object only if one was written there. A key that is just a prefix of
other keys holds none, so reads raise NotFound and ``exists`` is False for it.
Every write is atomic: a reader sees a whole object or nothing. ``create`` is
create-only, and of any number of concurrent creators exactly one succeeds. That is
how a stage's manifest is published once."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import BinaryIO


class StoreError(Exception):
    """Base class for store failures."""


class AlreadyExists(StoreError):
    """A create-only write found the key already present."""


class NotFound(StoreError):
    """A read named a key that does not exist."""


_SEGMENT = re.compile(r"[A-Za-z0-9@][A-Za-z0-9._@=+-]*")


def check_segment(name: str) -> str:
    """Return *name* if it is a valid single key segment, else raise ValueError.

    A segment starts with a letter, digit or ``@`` and contains only letters, digits
    and ``._@=+-``. That rules out ``/``, ``..``, the empty string, whitespace
    (including a trailing newline) and dotfiles, which the store keeps for its own
    temporary files."""
    if not _SEGMENT.fullmatch(name):
        raise ValueError(f"invalid store key segment {name!r}")
    return name


def check_key(key: str) -> str:
    """Return *key* if it is a valid store key, else raise ValueError.

    A key is one or more ``/``-separated segments, each valid under
    ``check_segment``. That rules out absolute paths, empty segments and a trailing
    ``/``."""
    if not key or not all(_SEGMENT.fullmatch(part) for part in key.split("/")):
        raise ValueError(f"invalid store key {key!r}")
    return key


class Store(ABC):
    """Atomic object storage addressed by key."""

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """The whole object; NotFound if absent."""

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None:
        """Atomically replace *key*. Use only for objects with one writer, such as a
        status file or a heartbeat."""

    @abstractmethod
    def create(self, key: str, data: bytes) -> None:
        """Atomically create *key*; AlreadyExists if it is present."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether *key* holds an object."""

    @abstractmethod
    def size(self, key: str) -> int:
        """The size in bytes of the object at *key*, from its metadata alone; nothing of it is
        read. NotFound if absent; StoreError for something at *key* that is no plain object."""

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """Every key at or under *prefix*, sorted; an empty list if there are none."""

    @abstractmethod
    def list_children(self, prefix: str) -> list[str]:
        """Names of the objects and sub-prefixes directly under *prefix*, sorted; an empty
        list if there are none. It does not descend, so listing a lock's generations or a
        stage's attempts costs one directory read however much each attempt holds."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key*; a missing key is not an error."""

    @abstractmethod
    def open_write(self, key: str, *, exclusive: bool = False) -> AbstractContextManager[BinaryIO]:
        """Stream a large object. It becomes visible only when the block exits without
        an error. With *exclusive* the publish is create-only."""

    @abstractmethod
    def open_read(self, key: str) -> AbstractContextManager[BinaryIO]:
        """Stream an object; NotFound if absent."""
