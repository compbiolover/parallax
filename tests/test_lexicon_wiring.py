"""eMFD CSV loading, lexicon selection/provenance, and meta round-trip."""

from __future__ import annotations

from ingestion.datastore import Datastore
from scoring.dictionary import DictionaryScorer
from scoring.lexicon import (
    SEED_NAME,
    build_lexicon,
    is_demo_lexicon,
    load_emfd_csv,
)

EMFD_HEADER = (
    "word,care_p,fairness_p,loyalty_p,authority_p,sanctity_p,"
    "care_sent,fairness_sent,loyalty_sent,authority_sent,sanctity_sent"
)


def _write_emfd(path):
    rows = [
        EMFD_HEADER,
        "harm,0.9,0.0,0.0,0.0,0.0,-0.8,0,0,0,0",
        "justice,0.0,0.95,0.0,0.0,0.0,0,0.6,0,0,0",
        "sacred,0.0,0.0,0.0,0.0,0.88,0,0,0,0,0.7",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_load_emfd_csv_parses_probabilities_and_sentiment(tmp_path):
    lex = load_emfd_csv(_write_emfd(tmp_path / "emfd.csv"))
    assert len(lex) == 3
    entry = lex.lookup("harm")
    assert entry is not None
    assert abs(entry.foundations["care"] - 0.9) < 1e-9
    assert entry.pole == -1  # negative sentiment
    assert lex.lookup("justice").pole == 1


def test_emfd_scores_via_dictionary_scorer(tmp_path):
    scorer = DictionaryScorer(load_emfd_csv(_write_emfd(tmp_path / "emfd.csv")))
    s = scorer.score("harm and justice and sacred things happen here")
    assert s.matched_words == 3
    assert s.foundations["care"] > 0
    assert s.foundations["fairness"] > 0
    assert s.foundations["sanctity"] > 0


def test_build_lexicon_prefers_path_falls_back_to_seed(tmp_path):
    lex, name = build_lexicon(None)
    assert name == SEED_NAME and is_demo_lexicon(name)

    lex, name = build_lexicon(tmp_path / "missing.csv")  # nonexistent -> seed
    assert name == SEED_NAME

    lex, name = build_lexicon(_write_emfd(tmp_path / "emfd.csv"))
    assert name.startswith("eMFD (") and not is_demo_lexicon(name)
    assert len(lex) == 3


def test_argmax_assignment_discriminates_probabilistic_lexicon(tmp_path):
    # A word with mass on all foundations should count only toward its dominant
    # one under argmax, but spread under probability mode.
    p = tmp_path / "emfd.csv"
    p.write_text(
        EMFD_HEADER + "\n" + "spread,0.5,0.2,0.1,0.1,0.1,0,0,0,0,0\n",
        encoding="utf-8",
    )
    lex = load_emfd_csv(p)
    argmax = DictionaryScorer(lex, assignment="argmax").score("spread")
    prob = DictionaryScorer(lex, assignment="probability").score("spread")
    # argmax: only care gets weight
    assert argmax.foundations["care"] > 0
    assert argmax.foundations["fairness"] == 0.0
    # probability: fairness also gets its share
    assert prob.foundations["fairness"] > 0.0


def test_invalid_assignment_rejected():
    import pytest

    with pytest.raises(ValueError):
        DictionaryScorer(assignment="nonsense")


def test_seed_unaffected_by_argmax():
    # Single-foundation seed entries score identically under both modes.
    a = DictionaryScorer(assignment="argmax").score("protect the sacred family from harm")
    b = DictionaryScorer(assignment="probability").score("protect the sacred family from harm")
    assert a.foundations == b.foundations


def test_meta_roundtrip():
    store = Datastore(":memory:")
    assert store.get_meta("lexicon") is None
    assert store.get_meta("lexicon", "seed") == "seed"
    store.set_meta("lexicon", "eMFD (emfd_scoring.csv)")
    assert store.get_meta("lexicon") == "eMFD (emfd_scoring.csv)"
    store.set_meta("lexicon", "updated")
    assert store.get_meta("lexicon") == "updated"
    store.close()
