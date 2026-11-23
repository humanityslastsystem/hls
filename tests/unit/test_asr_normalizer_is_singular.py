"""These metrics must offer exactly one text normalizer.

``word_error_rate`` here, and ``word_f1``/``sentence_bleu`` in
``hls.verify.metrics.omni``, all fold both sides through ``normalize_transcript``
(Whisper's ``EnglishTextNormalizer``). A second, plainer normalizer sitting beside it
is an invitation for the next measure to pick the wrong one, and the two disagree by
a lot rather than a little: the whole reason for the Whisper normalizer is that a
speech corpus spells numbers out while a recognizer formats them, and a
lowercase-and-strip-punctuation pass leaves those as different tokens.
"""
from hls.verify.metrics import asr
from hls.verify.metrics.omni import sentence_bleu, word_f1

#: The same sentence in the two spellings a speech corpus and a recognizer produce.
_SPELLED = "we raised two hundred fifty thousand dollars"
_FORMATTED = "We raised $250,000."


def test_the_module_exposes_one_normalizer():
    public = {n for n in vars(asr) if n.startswith("normalize")}
    assert public == {"normalize_transcript"}, (
        f"{public - {'normalize_transcript'}} is a second normalizer on a different "
        f"scale from the one every measure here uses")


def test_every_measure_agrees_the_two_spellings_are_the_same_sentence():
    # What a plainer normalizer would cost: these three all score a match, because
    # they share normalize_transcript. Strip punctuation and lowercase instead and
    # "250 000" against "two hundred fifty thousand" shares no token, so word F1 and
    # BLEU collapse and WER charges six edits.
    assert asr.word_error_rate(_FORMATTED, _SPELLED) == 0.0
    assert word_f1(_FORMATTED, _SPELLED) == 1.0
    assert sentence_bleu(_FORMATTED, _SPELLED) == 1.0


def test_a_plainer_normalizer_would_score_the_same_pair_at_zero():
    # Makes the cost concrete rather than asserting it: the obvious alternative --
    # lowercase, drop punctuation, collapse whitespace -- leaves the two spellings
    # sharing no token at all, so a measure built on it reports 0 where every measure
    # here reports 1.
    def plain(s):
        return " ".join("".join(c if c.isalnum() or c.isspace() else " "
                                for c in s.lower()).split())

    assert asr.normalize_transcript(_FORMATTED) == asr.normalize_transcript(_SPELLED)
    assert set(plain(_FORMATTED).split()) & set(plain(_SPELLED).split()) == {"we", "raised"}
    assert plain(_FORMATTED) != plain(_SPELLED)
