"""Provenance core -- self-describing pydantic config nodes (write-only).

The config tree is pure, frozen pydantic, serialized into rollout provenance.
Provenance is WRITE-ONLY: we serialize a spec to record what ran, never to
reconstruct a task from it, so there is no registry.

Polymorphic *families* (one base, many interchangeable implementations -- decode
strategies, verifiers, arrival processes, ...) still need to be identifiable in
the record: two field-compatible members would otherwise be indistinguishable
(e.g. ``LetterMatchMetric`` and ``ContainsMatchMetric`` both serialize to
``{}``). So a family node serializes a ``kind`` discriminator
equal to its concrete class name -- derived automatically by pydantic, no
per-subclass field and no registry.

Two bases:
  - ``Spec`` -- a plain frozen config node (concrete; no discriminator). A keyword that is not
    one of its fields is refused: a misspelled keyword in a task module would otherwise
    be dropped without a word.
  - ``Poly`` -- a family base; additionally serializes ``kind`` = class name.

Serialize a config tree with ``dump_spec``. It passes ``serialize_as_any=True``
so a field declared as a family base still serializes its runtime subclass's data
(a pydantic rule: without it, a base-typed field is serialized by the base schema
and subclass fields are dropped).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, computed_field, model_validator


class Spec(BaseModel):
    """A plain, frozen config node. Serializes with stock pydantic, and refuses a keyword that is
    not one of its fields, in a copy's update too."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """A copy with *update*'s fields replaced, validated as a constructed node is.

        Pydantic's ``model_copy`` validates nothing: an update key that is not a field, or a
        value its field does not take, would land in the copy without a word. Task variants
        are authored as copies with a field swapped, so the node the copy would be is first
        built from its fields through ``model_validate``, which refuses both and runs the
        model's checks. The copy then takes the validated values of the updated fields and
        keeps the other fields' objects, as pydantic's copy does."""
        if not update:
            return super().model_copy(deep=deep)
        fields = {name: getattr(self, name) for name in type(self).model_fields}
        checked = type(self).model_validate({**fields, **update})
        return super().model_copy(update={name: getattr(checked, name) for name in update}, deep=deep)


class Poly(Spec):
    """Base for an ABC-replacing family.

    Serializes a ``kind`` discriminator equal to the concrete subclass's name so
    the family member is identifiable in provenance. There is no registry. A dump
    read back may hold the tag, as an override that writes a family member's whole
    table does; it must name this class, and a tag naming another member is
    refused, so a union of members picks the one the tag names.
    """

    @model_validator(mode="before")
    @classmethod
    def _kind_as_written(cls, data: Any) -> Any:
        if isinstance(data, dict) and "kind" in data:
            if data["kind"] != cls.__name__:
                raise ValueError(f"kind is {data['kind']!r}, but this is a {cls.__name__}")
            data = {key: value for key, value in data.items() if key != "kind"}
        return data

    @computed_field
    @property
    def kind(self) -> str:
        return type(self).__name__


def dump_spec(node: Spec) -> dict:
    """Serialize a config node to a JSON-ready provenance dict.

    ``serialize_as_any=True`` makes nested polymorphic fields serialize by their
    runtime type, so subclass data (and each family member's ``kind``) is never
    silently dropped.
    """
    return node.model_dump(mode="json", serialize_as_any=True)
