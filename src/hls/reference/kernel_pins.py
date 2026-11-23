"""Pinning a transformers reference to torch arithmetic on its fused-kernel paths.

A checkpoint family's modeling module binds fused kernels at import time
(``from fla.ops.gated_delta_rule import chunk_gated_delta_rule`` when the package is
installed, else ``None``) and its layers capture the binding at construction
(``self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule``).
Rebinding every such entry point to ``None`` before the model is built therefore pins
every layer to the torch reduction whether or not the fused packages are importable,
which is the point: an fla/triton install in the reference image must not be able to
change the instrument's arithmetic.

On Qwen3.6-27B, scoring one identical tensor twice in one process, the torch reduction
is bit-exact while the fused path disagrees with itself at about a third of positions
(triton autotune config drift). A reference that returns different
log-probabilities for the same tokens cannot certify anything.

``FUSED_KERNEL_ENTRY_POINTS`` is the one table both guards derive from: each provider
package and the names transformers' modeling modules take from it. ``pin_fused_kernels``
nulls those names on a family's module before construction; ``fused_kernel_bindings``
walks the built model afterwards for anything still bound to one of the providers. A
test parses every modeling module of the installed transformers for the names it takes
from these providers, so a release that adds one fails the suite instead of slipping
past the pin.

The mamba family (mamba, mamba2, jamba, bamba, zamba, zamba2, falcon_h1, nemotron_h,
falcon_mamba) binds its kernels differently on the installed transformers: not at
import but inside the layer constructor, through ``lazy_load_kernel``, into module
globals the forward reads. Both guards here read imports and attributes, so this module
does not pin that family; its names are in the table so the drift test tracks them.
"""

from __future__ import annotations

import importlib

#: Provider package -> the entry-point names transformers' modeling modules take from
#: it. ``falcon_mamba_inner_fn`` is a name falcon_mamba's module asks the mamba-ssm
#: kernel for; the package does not define it, so the lookup yields ``None`` there and
#: the family runs its torch path.
FUSED_KERNEL_ENTRY_POINTS: dict[str, tuple[str, ...]] = {
    "fla": (
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
        "ShortConvolution",
    ),
    "causal_conv1d": (
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ),
    "mamba_ssm": (
        "selective_scan_fn",
        "mamba_inner_fn",
        "selective_state_update",
        "mamba_chunk_scan_combined",
        "mamba_split_conv1d_scan_combined",
        "falcon_mamba_inner_fn",
    ),
}


def pin_fused_kernels(model_type: str) -> dict[str, str]:
    """Rebind ``model_type``'s fused entry points to None, before any model is built,
    and report which ones the pin covered.

    Returns ``{entry_point: "torch_fallback"}`` for every table name the family's
    modeling module defines, ``{}`` for a family that defines none (a plain dense
    checkpoint has no fused path to pin) or whose modeling module is not a
    transformers-native one. The report is for the log and for
    ``fused_kernel_bindings``, which verifies the built model afterwards rather than
    trusting the pin. That check recognizes kernels by provider package, so between
    them the two catch a renamed entry point from a known provider. Both cover the
    providers the table names.
    """
    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        # No such family module (a remote-code or unknown model_type) means there is
        # nothing to pin. A module that exists but fails to import is a different
        # story: swallowing that would leave the pin silently uncovered.
        if exc.name in (module_path, f"transformers.models.{model_type}"):
            return {}
        raise
    pinned = {}
    for names in FUSED_KERNEL_ENTRY_POINTS.values():
        for name in names:
            if hasattr(mod, name):
                setattr(mod, name, None)
                pinned[name] = "torch_fallback"
    return pinned


def fused_kernel_bindings(model) -> list[str]:
    """Every place the built model holds a fused-kernel callable or submodule.

    Walks the module tree and reports ``<module path>.<attribute>`` for each attribute
    bound to something defined in a provider package of ``FUSED_KERNEL_ENTRY_POINTS``
    (submodules are reported by their own type). Checked by ``__module__`` root because
    that is what survives the import: a layer holding
    ``fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule`` reports a module under
    ``fla`` whatever the version. Empty on a correctly pinned torch-fallback instrument.
    """
    providers = FUSED_KERNEL_ENTRY_POINTS.keys()
    found: list[str] = []
    for path, sub in model.named_modules():
        if type(sub).__module__.split(".")[0] in providers:
            found.append(f"{path or '<root>'} is {type(sub).__module__}.{type(sub).__name__}")
        for attr, value in vars(sub).items():
            # Any attribute whose value comes from a provider counts, callable or not:
            # what a layer captured is a fused entry point either way, and the
            # instrument's promise is that none of them is reachable.
            owner = getattr(value, "__module__", None)
            if owner and owner.split(".")[0] in providers:
                found.append(
                    f"{path or '<root>'}.{attr} is {owner}."
                    f"{getattr(value, '__name__', type(value).__name__)}")
    return found
