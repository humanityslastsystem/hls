"""Verification -- the correctness gates (Verifier + RolloutMetric).

``base`` holds the abstractions + wiring; ``metrics`` holds the provided metrics;
``verifiers`` holds the provided verifiers. The public surface is re-exported here,
so ``from hls.verify import Verifier, LetterMatchMetric, TokenFaithfulnessVerifier`` works.

The two calibrated gates' measurements live beside them, each defined once so grading
and calibration cannot drift apart: ``flips`` for the token-level gate under a greedy
decode, ``report`` for the sampling gate under a sampled one.

This is the correctness domain; performance scoring (Scorer/SLO/throughput) lives in
``hls.scoring``.
"""

from hls.verify.base import (
    ArtifactRollout,
    EngineAccessor,
    RolloutMetric,
    Verifier,
    WiringError,
    check_wiring,
)
from hls.verify.metrics import (
    ContainsMatchMetric,
    CoqaF1Metric,
    LetterMatchMetric,
    MixedModalityMatchMetric,
    WERMetric,
    coqa_f1,
    normalize_answer,
    sentence_bleu,
    word_error_rate,
    word_f1,
)
from hls.verify.tts_faithfulness import (
    CodecFidelityVerifier,
)
from hls.verify.verifiers import (
    MetricVerifier,
    ReferenceRelativeVerifier,
    ReportFaithfulnessVerifier,
    TokenFaithfulnessVerifier,
)

__all__ = [
    # abstractions + wiring
    "Verifier",
    "RolloutMetric",
    "EngineAccessor",
    "ArtifactRollout",
    "WiringError",
    "check_wiring",
    # provided metrics
    "LetterMatchMetric",
    "ContainsMatchMetric",
    "CoqaF1Metric",
    "coqa_f1",
    "normalize_answer",
    "WERMetric",
    "word_error_rate",
    "MixedModalityMatchMetric",
    "word_f1",
    "sentence_bleu",
    # provided verifiers
    "MetricVerifier",
    "ReferenceRelativeVerifier",
    "ReportFaithfulnessVerifier",
    "TokenFaithfulnessVerifier",
    "CodecFidelityVerifier",
]
