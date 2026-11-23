import dataclasses
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field
from hls.override import apply_override, apply_overrides, parse_scalar


class _Accel(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: str
    count: int = 1


@dataclasses.dataclass
class _Task:
    accel: _Accel


class _Limits(BaseModel):
    model_config = ConfigDict(frozen=True)
    ceiling: int


class _Prices(BaseModel):
    model_config = ConfigDict(frozen=True)
    inp: float
    out: float
    limits: _Limits | None = None


class _Cheap(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["cheap"] = "cheap"


class _Metered(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["metered"] = "metered"
    rate: float


class _Producer(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str = "m"
    prices: _Prices | None = None
    billing: Annotated[_Cheap | _Metered, Field(discriminator="kind")] | None = None


class _Plain:
    def __init__(self, model_name=None):
        self.model_name = model_name


def test_override_frozen_pydantic_leaf():
    t = _Task(accel=_Accel(kind="H200"))
    out = apply_override(t, "accel.kind", "L4")
    assert out.accel.kind == "L4"
    assert t.accel.kind == "H200"  # original unchanged


def test_override_plain_object_leaf():
    p = _Plain(model_name=None)
    out = apply_override(p, "model_name", "opus")
    assert out.model_name == "opus"
    assert p.model_name is None


def test_unknown_path_segment_raises():
    t = _Task(accel=_Accel(kind="H200"))
    with pytest.raises(KeyError):
        apply_override(t, "accel.nope", "x")


def test_bad_value_revalidates_and_raises():
    t = _Task(accel=_Accel(kind="H200"))
    with pytest.raises(Exception):  # pydantic ValidationError: count must be int
        apply_override(t, "accel.count", "not-an-int")


def test_override_dict_value_by_key():
    obj = {"a": {"k": "v"}}
    out = apply_override(obj, "a.k", "v2")
    assert out["a"]["k"] == "v2"
    assert obj["a"]["k"] == "v"  # original unchanged


def test_unknown_dict_key_raises():
    import pytest

    with pytest.raises(KeyError):
        apply_override({"a": 1}, "b", "x")


def test_absent_optional_is_built_from_the_whole_group():
    p = _Producer()
    out = apply_overrides(p, {"prices.inp": 1.0, "prices.out": 2.0, "name": "glm"})
    assert out.prices == _Prices(inp=1.0, out=2.0)
    assert out.name == "glm"                      # a path outside the group still applies
    assert p.prices is None                       # original unchanged


def test_a_group_can_reach_a_spec_nested_inside_the_absent_one():
    out = apply_overrides(_Producer(), {"prices.inp": 1.0, "prices.out": 2.0,
                                        "prices.limits.ceiling": 9})
    assert out.prices.limits == _Limits(ceiling=9)


def test_a_partial_group_names_the_missing_paths_not_a_bare_pydantic_error():
    with pytest.raises(ValueError) as e:
        apply_overrides(_Producer(), {"prices.inp": 1.0})
    assert "prices.out" in str(e.value)            # named by its full dotted path
    assert "_Prices" in str(e.value)


def test_a_bad_value_in_a_group_names_its_path():
    with pytest.raises(ValueError) as e:
        apply_overrides(_Producer(), {"prices.inp": "not-a-number", "prices.out": 2.0})
    assert "prices.inp" in str(e.value)


def test_unknown_segment_under_an_absent_optional_raises():
    with pytest.raises(KeyError):
        apply_overrides(_Producer(), {"prices.inp": 1.0, "prices.out": 2.0, "prices.nope": 3})
    with pytest.raises(KeyError):                  # nested one segment deeper
        apply_overrides(_Producer(), {"prices.inp": 1.0, "prices.out": 2.0,
                                      "prices.limits.nope": 3})


def test_an_absent_discriminated_family_is_not_built_from_leaves():
    # Which member to build is undecidable from a loose leaf, so the family stays a
    # whole ``{ kind = ... }`` value and a leaf under an absent one fails as before.
    with pytest.raises(KeyError):
        apply_overrides(_Producer(), {"billing.rate": 0.5})
    out = apply_overrides(_Producer(), {"billing": {"kind": "metered", "rate": 0.5}})
    assert out.billing == _Metered(rate=0.5)


def test_a_leaf_of_a_present_optional_keeps_its_siblings():
    p = _Producer(prices=_Prices(inp=1.0, out=2.0))
    out = apply_overrides(p, {"prices.inp": 3.0})
    assert out.prices == _Prices(inp=3.0, out=2.0)


def test_a_non_spec_none_field_is_not_a_group():
    # ``str | None`` has nothing to build, so a path through it fails loud as before.
    class _Holder(BaseModel):
        model_config = ConfigDict(frozen=True)
        label: str | None = None

    with pytest.raises(KeyError):
        apply_overrides(_Holder(), {"label.x": 1})


def test_apply_overrides_of_nothing_is_the_object():
    p = _Producer()
    assert apply_overrides(p, {}) is p


def test_override_preserves_polymorphic_submodel():
    # _with_field must NOT round-trip through model_dump(), which would
    # serialize the concrete OpenAIChatProtocol to a plain dict and then crash trying
    # to instantiate the abstract WireProtocol field type on reconstruct.
    from hls.spec.contracts import RequestResponseContract
    from hls.spec.wire import OpenAIChatProtocol

    c = RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        transport="http",
        protocol=OpenAIChatProtocol(return_logprobs=True),
    )
    out = apply_override(c, "transport", "http-sse")
    assert out.transport == "http-sse"
    assert type(out.protocol) is OpenAIChatProtocol  # concrete subclass preserved
    assert out.protocol.return_logprobs is True  # nested value preserved
    assert isinstance(out.input_modalities, frozenset)  # frozenset not coerced
    assert c.transport == "http"  # original unchanged


def test_parse_scalar_comma_becomes_typed_list():
    assert parse_scalar("0,1,2,3") == [0, 1, 2, 3]      # each element int-parsed
    assert parse_scalar("a,b") == ["a", "b"]            # strs stay strs
    assert parse_scalar("1.5,2,x") == [1.5, 2, "x"]     # parsed per element
    assert parse_scalar("0, 1 , 2") == [0, 1, 2]        # whitespace trimmed
    assert parse_scalar("0") == 0                        # no comma -> scalar, not [0]


def test_parse_scalar_inline_table_becomes_one_value():
    """A ``--task.*`` value written as a TOML inline table is one value, for a field whose value
    is a discriminated family member; a value without a comma or braces is one scalar."""
    assert parse_scalar('{ kind = "provider", level = "low" }') == {"kind": "provider", "level": "low"}
    assert parse_scalar("true") is True and parse_scalar("2.5") == 2.5 and parse_scalar("x") == "x"
