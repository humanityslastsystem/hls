"""Provided correctness metrics.

Generic primitives live in ``common``; each benchmark gets its own module
(``coqa``, ...). All are first-class ``RolloutMetric``s re-exported here -- there is
no separate 'dataset metric' category.
"""

from hls.verify.metrics.asr import WERMetric, word_error_rate
from hls.verify.metrics.bfcl_ast import BFCLAstMatch
from hls.verify.metrics.coqa import CoqaF1Metric, coqa_f1, normalize_answer
from hls.verify.metrics.common import ContainsMatchMetric, LetterMatchMetric
from hls.verify.metrics.gsm8k import Gsm8kExactMatch, extract_final_number
from hls.verify.metrics.hotpot import HotpotEmF1
from hls.verify.metrics.ifeval import IFEvalConstraint
from hls.verify.metrics.mmlu_pro import MMLUProMatch, extract_letter
from hls.verify.metrics.mrcr import MRCRRatio
from hls.verify.metrics.omni import MixedModalityMatchMetric, sentence_bleu, word_f1

__all__ = [
    "LetterMatchMetric",
    "ContainsMatchMetric",
    "CoqaF1Metric",
    "coqa_f1",
    "normalize_answer",
    "Gsm8kExactMatch",
    "extract_final_number",
    "HotpotEmF1",
    "IFEvalConstraint",
    "MRCRRatio",
    "BFCLAstMatch",
    "MMLUProMatch",
    "extract_letter",
    "WERMetric",
    "word_error_rate",
    "MixedModalityMatchMetric",
    "sentence_bleu",
    "word_f1",
]
