"""Tests for hls.reference.kernel_pins: the one provider table, the two guards derived
from it, and the drift check that holds the table to the installed transformers."""

from __future__ import annotations

import ast
import re
import sys
import types
from pathlib import Path

import pytest

from hls.reference.kernel_pins import (
    FUSED_KERNEL_ENTRY_POINTS,
    fused_kernel_bindings,
    pin_fused_kernels,
)


def _every_entry_point() -> list[str]:
    return [name for names in FUSED_KERNEL_ENTRY_POINTS.values() for name in names]


def _fused_class(name, provider="fla"):
    """A class that reports itself as coming from a fused-kernel provider package."""
    cls = type(name, (), {})
    cls.__module__ = f"{provider}.modules"
    return cls


def test_pin_fused_kernels_nulls_every_table_name_the_family_defines(monkeypatch):
    fam = types.ModuleType("transformers.models.faux.modeling_faux")
    for name in _every_entry_point():
        setattr(fam, name, _fused_class(name))
    monkeypatch.setitem(sys.modules, fam.__name__, fam)

    pinned = pin_fused_kernels("faux")
    assert set(pinned) == set(_every_entry_point())
    assert set(pinned.values()) == {"torch_fallback"}
    # The pin is what the layers read at construction, so every name must now be None.
    assert all(getattr(fam, name) is None for name in _every_entry_point())


def test_pin_fused_kernels_is_a_noop_for_a_family_with_no_fused_path(monkeypatch):
    fam = types.ModuleType("transformers.models.dense.modeling_dense")
    monkeypatch.setitem(sys.modules, fam.__name__, fam)

    assert pin_fused_kernels("dense") == {}          # nothing to pin
    assert pin_fused_kernels("not_a_family") == {}   # no such family module


def test_pin_fused_kernels_reraises_a_broken_family_import(monkeypatch):
    """Only "there is no such family module" means "nothing to pin". A family module
    that exists but fails to import must not be swallowed: that would leave the pin
    silently covering nothing while the fused kernels stay live."""
    def _explode(name):
        raise ModuleNotFoundError("No module named 'triton'", name="triton")

    monkeypatch.setattr("importlib.import_module", _explode)
    with pytest.raises(ModuleNotFoundError, match="triton"):
        pin_fused_kernels("faux")


def test_fused_kernel_bindings_finds_submodules_and_captured_callables_of_every_provider():
    """The post-construction check, which is what actually holds: layers capture the
    fused entry points at construction, so the built model is inspected rather than the
    pin being trusted. Every provider in the table is recognized."""
    def torch_reduction():
        pass

    def fused_fn():                             # the entry point fla would have bound
        pass

    fused_fn.__module__ = "fla.ops.gated_delta_rule"

    def scan_fn():                              # ...and one mamba_ssm would have
        pass

    scan_fn.__module__ = "mamba_ssm.ops.selective_scan_interface"

    class Layer:
        def __init__(self, fn):
            self.chunk_gated_delta_rule = fn

    clean_layer, dirty_layer, scan_layer = Layer(torch_reduction), Layer(fused_fn), Layer(scan_fn)
    fused_norm = _fused_class("FusedRMSNormGated")()
    fused_conv = _fused_class("CausalConv1d", provider="causal_conv1d")()

    class Model:
        def named_modules(self):
            return [("", self), ("layers.0", clean_layer), ("layers.1", dirty_layer),
                    ("layers.1.norm", fused_norm), ("layers.2", scan_layer),
                    ("layers.2.conv", fused_conv)]

    found = fused_kernel_bindings(Model())
    assert any("layers.1.chunk_gated_delta_rule" in f for f in found)
    assert any("layers.1.norm is fla.modules" in f for f in found)
    assert any("layers.2.chunk_gated_delta_rule is mamba_ssm" in f for f in found)
    assert any("layers.2.conv is causal_conv1d.modules" in f for f in found)
    assert not any("layers.0" in f for f in found)

    class CleanModel:
        def named_modules(self):
            return [("", self), ("layers.0", clean_layer)]

    assert fused_kernel_bindings(CleanModel()) == []


# ---- drift: the table against the installed transformers ---------------------

#: The hub kernel names ``lazy_load_kernel`` resolves, and the provider package each
#: one is. The mamba family binds its kernels through this call at layer construction
#: rather than by import, so its names are read off the ``getattr`` and
#: ``resolve_internal_import`` lookups on the loaded kernel.
_HUB_KERNEL_PROVIDERS = {
    "causal-conv1d": "causal_conv1d",
    "mamba-ssm": "mamba_ssm",
    "falcon_mamba-ssm": "mamba_ssm",
}

_MENTIONS = re.compile(r"\b(fla|causal_conv1d|mamba_ssm|lazy_load_kernel)\b")


def _call_name(node) -> str | None:
    if isinstance(node, ast.Call):
        fn = node.func
        return fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
    return None


def _names_taken(tree: ast.AST) -> tuple[dict[str, set[str]], set[str]]:
    """Per provider, the names a modeling module takes from it, by import or by lookup
    on a lazily loaded kernel; plus every hub kernel name it loads."""
    taken: dict[str, set[str]] = {p: set() for p in FUSED_KERNEL_ENTRY_POINTS}
    hubs: set[str] = set()
    lazy_vars: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in taken:
                taken[root].update(alias.name for alias in node.names)
        if (isinstance(node, ast.Assign) and _call_name(node.value) == "lazy_load_kernel"
                and isinstance(node.value.args[0], ast.Constant)):
            hub = node.value.args[0].value
            hubs.add(hub)
            for target in node.targets:
                if isinstance(target, ast.Name) and hub in _HUB_KERNEL_PROVIDERS:
                    lazy_vars[target.id] = _HUB_KERNEL_PROVIDERS[hub]
    for node in ast.walk(tree):
        name = _call_name(node)
        if name not in ("getattr", "resolve_internal_import") or not node.args:
            continue
        source = node.args[0]
        if not (isinstance(source, ast.Name) and source.id in lazy_vars):
            continue
        provider = lazy_vars[source.id]
        if name == "getattr" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            taken[provider].add(node.args[1].value)
        for kw in node.keywords:
            if kw.arg == "chained_path" and isinstance(kw.value, ast.Constant):
                taken[provider].add(kw.value.value.rsplit(".", 1)[-1])
    return taken, hubs


def test_the_table_covers_every_fused_name_the_installed_transformers_takes():
    """Every modeling module of the installed transformers is parsed for the names it
    takes from the table's providers, whether by import or by lookup on a lazily
    loaded kernel. A name outside the table is a fused path the pin does not cover, so
    a transformers release that adds one fails here rather than in a reference."""
    transformers = pytest.importorskip("transformers")
    models_dir = Path(transformers.__file__).parent / "models"
    modeling = sorted(models_dir.glob("*/modeling_*.py"))
    assert modeling, models_dir

    seen_providers: set[str] = set()
    for path in modeling:
        text = path.read_text()
        if not _MENTIONS.search(text):
            continue
        taken, hubs = _names_taken(ast.parse(text))
        unknown_hubs = hubs - set(_HUB_KERNEL_PROVIDERS)
        assert not unknown_hubs, (
            f"{path.name} lazily loads kernel(s) {sorted(unknown_hubs)} that "
            f"_HUB_KERNEL_PROVIDERS does not map to a provider")
        for provider, names in taken.items():
            if names:
                seen_providers.add(provider)
            extra = names - set(FUSED_KERNEL_ENTRY_POINTS[provider])
            assert not extra, (
                f"{path.name} takes {sorted(extra)} from {provider!r}, which "
                f"FUSED_KERNEL_ENTRY_POINTS does not list; the pin would leave them live")
    # The table must also not outlive its providers: every one still has a taker.
    assert seen_providers == set(FUSED_KERNEL_ENTRY_POINTS)
