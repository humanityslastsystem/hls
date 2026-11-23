"""The top-level ``hls`` facade must stay coherent: every name in ``__all__``
resolves through the lazy ``_EXPORTS`` table (so ``from hls import *`` works),
which is exactly what breaks when a rename updates ``hls.<submodule>`` but not
the facade's three parallel lists (TYPE_CHECKING import / __all__ / _EXPORTS)."""

import hls


def test_every_facade_export_resolves():
    missing = [name for name in hls.__all__ if not hasattr(hls, name)]
    assert not missing, f"hls.__all__ names that do not resolve: {missing}"


def test_the_raw_prompt_protocol_and_its_datasets_are_on_the_facade():
    """A task authored against the raw-prompt wire protocol imports it from ``hls`` the
    way it imports the chat one; the same for the fill-in-the-middle datasets beside the
    already-exported ``Dataset``/``HFDataset``."""
    from hls.spec.datasets import FimSessionDataset, SafimDataset
    from hls.spec.wire import OpenAICompletionProtocol

    assert hls.OpenAICompletionProtocol is OpenAICompletionProtocol
    assert hls.FimSessionDataset is FimSessionDataset
    assert hls.SafimDataset is SafimDataset
    listed = set(hls.__all__)
    assert {"OpenAICompletionProtocol", "FimSessionDataset", "SafimDataset"} <= listed


def test_headline_verifier_is_on_the_facade():
    # The calibrated token-level gate is the documented top-level surface.
    from hls.verify.verifiers import TokenFaithfulnessVerifier

    assert hls.TokenFaithfulnessVerifier is TokenFaithfulnessVerifier


def test_the_facade_exports_no_image_recipe_search_knob_or_prebuilt_artifact():
    """A task module declares no image and no search space, and a contestant is built by an
    agent: the image recipes, the search knobs and the prebuilt artifact are no task-authoring
    vocabulary."""
    old = {"Image", "ImageSet", "Recipe", "SearchKnob", "PrebuiltArtifact"}
    assert old.isdisjoint(hls.__all__)
    assert not [name for name in old if hasattr(hls, name)]
