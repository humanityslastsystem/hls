"""Dotted-path overrides of a task: rebuild an immutable spec tree with one leaf
replaced, or with a whole map of leaves replaced, and parse an override's value from the
command line.

The CLI parses each ``--task.*`` value with ``parse_scalar``, and every role rebuilds the task
its task reference names (``hls.taskref``) with ``apply_overrides``."""

from __future__ import annotations

import copy
import dataclasses
import tomllib
from collections.abc import Mapping
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError


def apply_overrides(obj: Any, overrides: Mapping[str, Any]) -> Any:
    """Return a copy of *obj* with every dotted override in *overrides* applied.

    Taking the whole map does one thing a loop over ``apply_override`` cannot: build
    an optional sub-spec that is currently ``None``. The ``--task.*`` flags give one
    dotted path per leaf, and none of those leaves can construct the spec on its own,
    so the leaves under an absent optional are collected and the spec is built once
    from the group.

    A group is validated against the type it builds, so a group missing one of that
    type's required fields is reported as the dotted paths the operator still owes,
    rather than as a pydantic error relative to a spec they never named. An unknown
    segment under an absent optional still raises ``KeyError``, as one under a spec
    that is present does.

    A sub-spec that is already there is not a group: its leaves go through
    ``apply_override`` one at a time and its other fields keep their values.

    Grouping is decided against *obj* as given, before anything is applied, so the
    order of the map does not change which paths group together. A group applies at
    the position of its first member; every other path applies in map order.
    """
    groups: dict[str, dict[str, Any]] = {}          # prefix -> {path below it: value}
    targets: dict[str, type[BaseModel]] = {}        # prefix -> the spec type to build there
    group_of: dict[str, str] = {}                   # override path -> the prefix it feeds
    for path, value in overrides.items():
        found = _absent_optional(obj, path)
        if found is None:
            continue
        prefix, target = found
        group_of[path] = prefix
        targets[prefix] = target
        groups.setdefault(prefix, {})[path[len(prefix) + 1 :]] = value

    out = obj
    built: set[str] = set()
    for path, value in overrides.items():
        prefix = group_of.get(path)
        if prefix is None:
            out = apply_override(out, path, value)
        elif prefix not in built:
            built.add(prefix)
            out = apply_override(out, prefix, _build_absent(prefix, targets[prefix], groups[prefix]))
    return out


def apply_override(obj: Any, path: str, value: Any) -> Any:
    """Return a copy of *obj* with the leaf at dotted *path* replaced by *value*.

    Rebuilds immutably from the leaf up, re-validating at each level: a frozen
    pydantic model is reconstructed (re-running validation), a dataclass goes
    through ``dataclasses.replace`` (re-running ``__post_init__``), and a plain
    object is shallow-copied with ``setattr``. Dict-valued fields are descended
    by key (only existing keys; unknown keys fail loud). Raises ``KeyError`` on
    an unknown path segment and propagates the constructor's error on a bad
    value -- fail loud.

    Overrides are tree-structured: rebuilding a node gives it a new identity, so a
    sibling that cached the same instance by reference keeps the old one. A holder
    that must agree should therefore re-derive its copy from the single owner at use
    time rather than cache it -- e.g. the drive role hands the driver
    ``TaskSpec.contract`` when it drives, so overriding ``task.contract.*`` here just
    works.
    """
    head, _, rest = path.partition(".")
    if isinstance(obj, dict):
        if head not in obj:
            raise KeyError(f"unknown override path segment {head!r} on dict (keys: {sorted(obj)})")
        new_child = apply_override(obj[head], rest, value) if rest else value
        return {**obj, head: new_child}
    if not hasattr(obj, head):
        raise KeyError(f"unknown override path segment {head!r} on {type(obj).__name__}")
    new_child = apply_override(getattr(obj, head), rest, value) if rest else value
    return _with_field(obj, head, new_child)


def _absent_optional(obj: Any, path: str) -> tuple[str, type[BaseModel]] | None:
    """The prefix of *path* that addresses an absent optional sub-spec with leaves
    below it, paired with the spec type to build there; None when *path* has no such
    prefix.

    None when the path reaches an existing value all the way down (the ordinary
    case), when a segment does not exist at all (``apply_override`` reports that),
    or when the absent field admits more than one spec type -- a discriminated
    family is written as one whole ``{ kind = ... }`` table, never assembled from
    loose leaves.
    """
    node = obj
    parts = path.split(".")
    for i, head in enumerate(parts[:-1]):  # a trailing leaf is a value, not a spec to build
        if isinstance(node, dict) or not hasattr(node, head):
            return None
        child = getattr(node, head)
        if child is None:
            target = _field_model(node, head)
            return (".".join(parts[: i + 1]), target) if target is not None else None
        node = child
    return None


def _build_absent(prefix: str, target: type[BaseModel], leaves: dict[str, Any]) -> BaseModel:
    """Build the sub-spec at dotted *prefix* from the override leaves below it,
    reporting what a partial group is missing by full dotted path."""
    data: dict[str, Any] = {}
    for rel, value in leaves.items():
        _check_known(prefix, target, rel)
        node = data
        *branch, last = rel.split(".")
        for seg in branch:
            if not isinstance(node.get(seg), dict):
                node[seg] = {}
            node = node[seg]
        node[last] = value
    try:
        return target.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join([prefix, *(str(p) for p in err['loc'])])}: {err['msg']}" for err in e.errors()
        )
        raise ValueError(
            f"cannot build {prefix} ({target.__name__}) from the overrides under it -- {problems}"
        ) from e


def _check_known(prefix: str, target: type[BaseModel], rel: str) -> None:
    """Raise ``KeyError`` on a segment of *rel* that is not a field of the spec being
    built, so an unknown path under an absent optional fails as loudly as one under a
    spec that is present. Checking stops where the annotation stops being a single
    spec type; validation has the say from there."""
    model: type[BaseModel] | None = target
    walked = prefix
    for seg in rel.split("."):
        if model is None:
            return
        if seg not in model.model_fields:
            raise KeyError(f"unknown override path segment {seg!r} on {model.__name__} (at {walked})")
        walked = f"{walked}.{seg}"
        model = _sole_model(model.model_fields[seg].annotation)


def _field_model(node: Any, name: str) -> type[BaseModel] | None:
    """The spec type of field *name* on *node*, or None when *node* is not a pydantic
    config node or the field is not typed as a single spec. The config tree is pure
    pydantic (hls.core.poly), so that is where an absent optional can be built."""
    if not isinstance(node, BaseModel):
        return None
    field = type(node).model_fields.get(name)
    return None if field is None else _sole_model(field.annotation)


def _sole_model(annotation: Any) -> type[BaseModel] | None:
    """The one pydantic model an annotation admits besides ``None``, or None if it
    admits no model or several."""
    members = [m for m in _union_members(annotation) if m is not type(None)]
    if len(members) != 1:
        return None
    sole = members[0]
    return sole if isinstance(sole, type) and issubclass(sole, BaseModel) else None


def _union_members(annotation: Any) -> list[Any]:
    """The alternatives an annotation admits, flattened, with ``Annotated`` metadata
    (a discriminator, a constraint) stripped off."""
    if hasattr(annotation, "__metadata__"):
        return _union_members(annotation.__origin__)
    if get_origin(annotation) in (Union, UnionType):
        return [m for arg in get_args(annotation) for m in _union_members(arg)]
    return [annotation]


def _with_field(obj: Any, name: str, value: Any) -> Any:
    """Return a copy of *obj* with attribute *name* set to *value*, rebuilt by the
    type-appropriate immutable mechanism (pydantic reconstruct / dataclass replace
    / plain shallow-copy)."""
    if isinstance(obj, BaseModel):
        # Rebuild from the live field VALUES, NOT model_dump(). model_dump()
        # recursively serializes nested submodels to plain dicts, which loses the
        # concrete subclass of a polymorphic / ABC-typed field (e.g.
        # ``RequestResponseContract.protocol: WireProtocol``) and makes the
        # constructor try to instantiate the abstract base. Passing the existing
        # concrete instances straight back preserves them; the constructor still
        # re-validates *value* (and is cheaper -- no deep serialize of the subtree).
        data = {f: getattr(obj, f) for f in type(obj).model_fields}
        data[name] = value
        return type(obj)(**data)  # re-validates value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **{name: value})  # re-runs __post_init__
    new = copy.copy(obj)
    setattr(new, name, value)
    return new


def parse_scalar(text: str) -> Any:
    """Coerce a ``--task.*`` value from the command line to a typed scalar
    (int/float/bool), a list (comma-delimited; each element parsed the same way), a TOML
    inline table (``{ kind = "provider", level = "xhigh" }``, for a field whose value is a
    discriminated family member), else leave it a str.

    Comma support lets a list-valued field be set from the flat ``--task.*`` flags,
    e.g. ``0,1,2,3`` -> ``[0, 1, 2, 3]``. A value with no comma stays a scalar (``"0"``
    -> ``0``, not ``[0]``); a string value that holds a comma is set in the task module,
    not on the command line."""
    if text.startswith("{") and text.endswith("}"):
        return tomllib.loads(f"v = {text}")["v"]
    if "," in text:
        return [parse_scalar(part.strip()) for part in text.split(",")]
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            pass
    return text
