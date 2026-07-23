"""Dictionary scorer: length normalization, coverage, and liberty handling."""

from __future__ import annotations

from scoring.dictionary import DictionaryScorer
from scoring.foundations import CLASSIC_FOUNDATIONS
from scoring.lexicon import Entry, Lexicon


def test_length_normalization_is_a_rate_not_a_count():
    """Repeating a document must not inflate its foundation scores."""
    scorer = DictionaryScorer()
    short = "The community must unite to protect the sacred family."
    long = short + " " + " ".join(["neutral filler word"] * 40)
    s_short = scorer.score(short)
    s_long = scorer.score(long)
    # Same moral words, far more filler -> lower per-token rate for the long doc.
    assert s_long.foundations["loyalty"] < s_short.foundations["loyalty"]
    # And all scores are bounded rates in [0, 1].
    for f in CLASSIC_FOUNDATIONS:
        assert 0.0 <= s_short.foundations[f] <= 1.0


def test_empty_document_is_safe():
    s = DictionaryScorer().score("")
    assert s.word_count == 0
    assert s.moral_word_ratio == 0.0
    assert all(v == 0.0 for v in s.foundations.values())


def test_liberty_is_unscored_not_zero():
    s = DictionaryScorer().score("liberty freedom oppression tyranny")
    assert s.liberty is None  # dictionary baseline has no liberty signal


def test_custom_lexicon_and_pole():
    lex = Lexicon()
    lex.add("harm", Entry({"care": 1.0}, pole=-1), wildcard=False)
    lex.add("protect", Entry({"care": 1.0}, pole=+1), wildcard=False)
    s = DictionaryScorer(lex).score("protect protect harm neutral words here now")
    assert s.matched_words == 3
    assert s.foundations["care"] == 3 / 7
    # sentiment = (+1 +1 -1) / 7
    assert abs(s.sentiment - (1 / 7)) < 1e-9


def test_wildcard_prefix_matching():
    lex = Lexicon()
    lex.add("nurtur", Entry({"care": 1.0}, pole=+1), wildcard=True)
    s = DictionaryScorer(lex).score("nurturing nurture nurtured")
    assert s.matched_words == 3
