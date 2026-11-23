"""L0 -- Decode configuration.

How the reference decodes: the strategy (greedy or sampled) and, for greedy, the
frequency penalty applied to the logits before the argmax.

Neither strategy carries a draw seed. Greedy has no draw to pin, and a sampled
reference is not replayed: the report verifier reads the distribution the server
reported at each position and tests the served token against it, so an unseeded
server is graded exactly as a seeded one. Pinning a draw also costs more than it
sounds -- sglang only acts on a per-request seed when the whole engine is built for
deterministic inference, which swaps its sampling and attention backends, so a
seeded member would measure a different kernel set than its unseeded neighbours.
"""

from __future__ import annotations

from abc import abstractmethod

from pydantic import Field

from hls.core.poly import Poly


class DecodeConfig(Poly):
    """How the reference decodes, bound at construction (L0/L3).

    `fingerprint` returns a stable string folded into the reference fingerprint and
    cache scope, so changing the decode config invalidates cached oracle outputs.
    Subclass per strategy (Greedy, Sample).
    """

    @abstractmethod
    def fingerprint(self) -> str:
        """Stable string folded into the reference fingerprint / cache scope."""


class Greedy(DecodeConfig):
    """Argmax decode, optionally after an additive frequency penalty.

    ``frequency_penalty`` is subtracted from every logit, times the number of times
    that token id already appears in the reply being generated (the current request's
    output tokens only; the prompt is never counted). Applied in fp32 before the
    softmax, the same way vLLM and SGLang apply it, so the pinned reference and a
    faithful server choose the same argmax. 0.0 is plain greedy over the raw logits.
    """

    #: Additive per-occurrence logit penalty; 0.0 (the default) is plain greedy.
    #: Negative values would reward repetition and are refused.
    frequency_penalty: float = Field(default=0.0, ge=0.0)

    @property
    def penalized(self) -> bool:
        """Whether any penalty is in force (a 0.0 penalty is plain greedy)."""
        return self.frequency_penalty != 0.0

    def fingerprint(self) -> str:
        if not self.penalized:
            return "greedy"
        return f"greedy-fp{self.frequency_penalty}"


class Sample(DecodeConfig):
    """Draw each token from the model's own next-token distribution at ``temperature``.

    The distribution is the softmax of the logits divided by the temperature, over the
    whole vocabulary: no top-p, no top-k, no min-p, and no repetition, presence or
    frequency penalty. Anything that truncates or reweights it is a different
    distribution, which is the deviation the report verifier
    (``hls.verify.verifiers.ReportFaithfulnessVerifier``) exists to detect, so it is
    excluded from the decode rather than configured in it.

    The draw is left to each stack's own generator. Nothing downstream replays it:
    the report verifier reads the distribution the server reported at each position
    and tests the served token against it, so an unseeded server is graded exactly as
    a seeded one.
    """

    #: Softmax temperature. Strictly positive: 0 is the argmax, which is ``Greedy``.
    temperature: float = Field(default=1.0, gt=0.0)

    def fingerprint(self) -> str:
        return f"sample-t{self.temperature}"
