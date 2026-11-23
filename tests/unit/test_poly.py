import json

import pytest
from pydantic import ValidationError

from hls.core.poly import Poly, Spec, dump_spec


# A throwaway family for testing the core.
class Color(Poly):
    pass


class Red(Color):
    pass


class Blue(Color):
    shade: int


class Box(Spec):                 # plain concrete node holding a polymorphic field
    label: str
    color: Color                 # base annotation; dump_spec serializes the subclass


def test_concrete_poly_dumps_class_name_as_kind():
    # The kind discriminator is the concrete class name, derived automatically.
    assert Red().model_dump() == {"kind": "Red"}
    assert Blue(shade=4).model_dump() == {"shade": 4, "kind": "Blue"}


def test_field_identical_members_are_distinguishable():
    # Two members with the SAME field shape are told apart only by kind --
    # this is why the discriminator exists (cf. LetterMatch vs ContainsMatch).
    class A(Color):
        n: int = 1

    class B(Color):
        n: int = 1

    assert dump_spec(A()) != dump_spec(B())
    assert dump_spec(A())["kind"] == "A"
    assert dump_spec(B())["kind"] == "B"


def test_nested_polymorphic_field_serializes_subclass_data():
    # serialize_as_any (inside dump_spec) makes a base-typed field serialize its
    # runtime subclass's data + kind, instead of being flattened to the base schema.
    rec = dump_spec(Box(label="x", color=Blue(shade=9)))
    assert rec == {"label": "x", "color": {"shade": 9, "kind": "Blue"}}


def test_specs_are_frozen():
    b = Blue(shade=1)
    with pytest.raises(ValidationError):
        b.shade = 2


def test_dump_is_json_serializable():
    rec = dump_spec(Box(label="y", color=Blue(shade=3)))
    assert json.loads(json.dumps(rec)) == rec
