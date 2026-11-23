"""The hub sources a dataset or a tool bundle reads, as it declares them.

A dataset's ``corpus()`` and a tool bundle's ``build()`` read repos from the Hugging Face hub,
and the cache key of what they write stands for fixed bytes only when every one of those
reads is pinned to a commit. Each declares its reads through ``hub_revisions()``, a list of
``(repo, revision)``, beside the method that reads, so that a subclass which reads something
more has to say so: ``declared_hub_revisions`` refuses a class whose reading method sits
below the declaration it inherits."""

from __future__ import annotations


class UndeclaredHubSources(ValueError):
    """An object reads the hub without declaring what it reads (``hub_revisions``)."""


def undeclared_by(cls: type, *, reads: str) -> str | None:
    """Why class *cls* does not declare the hub sources its method *reads* reads, or None:
    the class that defines *reads* must be the one that defines ``hub_revisions``, or a base
    of it, and both must be there."""
    reader = next((c for c in cls.__mro__ if reads in vars(c)), None)
    declarer = next((c for c in cls.__mro__ if "hub_revisions" in vars(c)), None)
    if reader is None or declarer is None or not issubclass(declarer, reader):
        return (f"{cls.__name__} defines {reads}() without declaring the hub sources it reads (hub_revisions)"
                if reader is not None else f"{cls.__name__} has no {reads}() and declares no hub sources (hub_revisions)")
    return None


def declared_hub_revisions(obj: object, *, reads: str) -> list[tuple[str, str]]:
    """The ``(repo, revision)`` pairs *obj* declares through ``hub_revisions()``, as declared,
    branch names included. *reads* names the method of *obj* that reads the hub. Raise
    UndeclaredHubSources when the class does not declare them beside that method
    (``undeclared_by``) or declares none at all, which the base classes' ``hub_revisions``
    says by raising NotImplementedError. A ValueError of the declaration itself is raised as it
    is."""
    problem = undeclared_by(type(obj), reads=reads)
    if problem is not None:
        raise UndeclaredHubSources(problem)
    try:
        sources = obj.hub_revisions()
    except NotImplementedError as e:
        raise UndeclaredHubSources(f"{type(obj).__name__} declares no hub sources (hub_revisions)") from e
    return [(str(repo), str(revision)) for repo, revision in sources]
