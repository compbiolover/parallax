"""The MFQ-2 equality/proportionality split: partition, persistence, aggregation."""

from __future__ import annotations

import sqlite3

from compare.fairness import (
    LOW_COVERAGE,
    all_persona_fairness,
    gap,
    persona_fairness_profile,
)
from ingestion.datastore import Datastore
from scoring.dictionary import DictionaryScorer
from scoring.fairness_split import (
    EQUALITY_TERMS,
    PROPORTIONALITY_TERMS,
    FairnessSplitter,
    apply_split,
)
from scoring.foundations import (
    CLASSIC_FOUNDATIONS,
    EXTENDED_FOUNDATIONS,
    FAIRNESS_SUBFOUNDATIONS,
    MFQ2_FOUNDATIONS,
)
from scoring.seed_lexicon import SEED_LEXICON

from .registries import pair, registry

EQUALITY_TEXT = (
    "The ruling entrenches unequal access to justice. Systemic discrimination has "
    "excluded marginalized families, and equal treatment requires equity for everyone."
)
PROPORTIONALITY_TEXT = (
    "The program rewards those who contributed nothing while punishing the diligent. "
    "People who earn their way deserve proportionate reward; handouts to freeloaders "
    "make the responsible bear the consequence."
)
NO_EVIDENCE_TEXT = "The council approved the zoning variance after a short hearing."


# -- vocabulary ------------------------------------------------------------

def test_mfq2_replaces_fairness_with_its_two_halves():
    assert "fairness" not in MFQ2_FOUNDATIONS
    assert set(FAIRNESS_SUBFOUNDATIONS) <= set(MFQ2_FOUNDATIONS)
    # MFQ-2 does not include liberty; Parallax's widest set adds it back.
    assert "liberty" not in MFQ2_FOUNDATIONS
    assert "liberty" in EXTENDED_FOUNDATIONS
    assert len(EXTENDED_FOUNDATIONS) == 7


def test_seed_lexicon_fairness_terms_are_not_all_equality_flavoured():
    """Regression guard. The seed's fairness list once had zero proportionality
    terms, so merit-framed arguments scored as containing no fairness at all —
    a symmetry failure hiding inside a word list."""
    splitter = FairnessSplitter()
    sides = [
        splitter._side(stem)
        for stem, (founds, _pole) in SEED_LEXICON.items()
        if "fairness" in founds
    ]
    assert sides.count("proportionality") >= 3
    assert sides.count("equality") >= 1


# -- the splitter ----------------------------------------------------------

def test_splitter_reads_the_framing():
    splitter = FairnessSplitter()
    eq = splitter.split(EQUALITY_TEXT.lower().split())
    pr = splitter.split(PROPORTIONALITY_TEXT.lower().split())
    assert eq.leans == "equality"
    assert pr.leans == "proportionality"
    assert eq.equality > 0.5 and pr.proportionality > 0.5


def test_shares_always_sum_to_one():
    splitter = FairnessSplitter()
    for text in (EQUALITY_TEXT, PROPORTIONALITY_TEXT):
        split = splitter.split(text.lower().split())
        assert abs(split.equality + split.proportionality - 1.0) < 1e-12


def test_thin_evidence_yields_no_split_rather_than_a_guess():
    splitter = FairnessSplitter()
    assert splitter.split(NO_EVIDENCE_TEXT.lower().split()) is None


def test_min_evidence_is_enforced():
    tokens = ["the", "equal", "treatment", "clause"]   # exactly one split-term
    assert FairnessSplitter(min_evidence=1).split(tokens) is not None
    assert FairnessSplitter(min_evidence=2).split(tokens) is None


def test_min_evidence_must_be_positive():
    try:
        FairnessSplitter(min_evidence=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_apply_split_keeps_unsplit_distinct_from_zero():
    assert apply_split(0.5, None) == {"equality": None, "proportionality": None}


def test_term_lists_do_not_overlap():
    """A stem on both sides would make its side an artifact of sort order."""
    assert not (set(EQUALITY_TERMS) & set(PROPORTIONALITY_TERMS))


# -- scorer integration ----------------------------------------------------

def test_splitter_never_changes_the_classic_five():
    plain = DictionaryScorer().score(EQUALITY_TEXT)
    split = DictionaryScorer(splitter=FairnessSplitter()).score(EQUALITY_TEXT)
    assert plain.foundations == split.foundations


def test_halves_reconstitute_the_fairness_score():
    score = DictionaryScorer(splitter=FairnessSplitter()).score(EQUALITY_TEXT)
    assert abs(score.equality + score.proportionality - score.foundations["fairness"]) < 1e-12


def test_no_splitter_means_no_split():
    score = DictionaryScorer().score(EQUALITY_TEXT)
    assert score.equality is None and score.proportionality is None


def test_both_framings_register_comparable_fairness():
    """The seed lexicon must not be so equality-flavoured that a proportionality
    argument reads as containing no fairness."""
    scorer = DictionaryScorer(splitter=FairnessSplitter())
    eq = scorer.score(EQUALITY_TEXT).foundations["fairness"]
    pr = scorer.score(PROPORTIONALITY_TEXT).foundations["fairness"]
    assert eq > 0 and pr > 0
    assert 0.25 < (pr / eq) < 4.0


# -- persistence -----------------------------------------------------------

def _reg():
    """Both personas, each reading its own source."""
    return registry(self={"src_self": 1.0}, modeled_ce={"src_modeled_ce": 1.0})


def _store_with_split():
    store = Datastore(":memory:")
    scorer = DictionaryScorer(splitter=FairnessSplitter())
    for diet, text in (("self", EQUALITY_TEXT), ("modeled_ce", PROPORTIONALITY_TEXT)):
        for i in range(3):
            did = f"{diet}-{i}"
            score = scorer.score(text)
            store.upsert_document(
                doc_id=did, source_id=f"src_{diet}", stratum_id=None, url=None,
                title="t", published_utc=None, fetched_utc="2026-07-25T00:00:00+00:00",
                word_count=score.word_count, minhash=None,
            )
            store.upsert_scores(
                document_id=did, scorer="dictionary", foundations=score.foundations,
                sentiment=score.sentiment, moral_word_ratio=score.moral_word_ratio,
                matched_words=score.matched_words,
                equality=score.equality, proportionality=score.proportionality,
            )
    return store


def test_split_round_trips_through_the_datastore():
    store = _store_with_split()
    rows, total = store.fairness_split_for_sources(["src_self"])
    assert total == 3 and len(rows) == 3
    assert all(eq >= 0 and pr >= 0 for _w, eq, pr in rows)
    store.close()


def test_unsplit_rows_are_excluded_not_zeroed():
    store = Datastore(":memory:")
    store.upsert_document(
        doc_id="d", source_id="src_self", stratum_id=None, url=None,
        title="t", published_utc=None, fetched_utc="2026-07-25T00:00:00+00:00",
        word_count=50, minhash=None,
    )
    store.upsert_scores(
        document_id="d", scorer="dictionary", foundations={"fairness": 0.4},
        sentiment=0.0, moral_word_ratio=0.1, matched_words=5,
    )
    rows, total = store.fairness_split_for_sources(["src_self"])
    assert total == 1 and rows == []       # counted, but not treated as 0/0
    store.close()


def test_migration_adds_columns_to_a_pre_split_database(tmp_path):
    """Opening a store created before the split must upgrade it in place, not
    fail on the first write that mentions the new columns."""
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE documents (id TEXT PRIMARY KEY, diet_id TEXT NOT NULL,
            source_id TEXT NOT NULL, stratum_id TEXT, url TEXT, title TEXT,
            published_utc TEXT, fetched_utc TEXT NOT NULL, word_count INTEGER NOT NULL,
            minhash TEXT, weight REAL NOT NULL DEFAULT 1.0,
            is_duplicate INTEGER NOT NULL DEFAULT 0, duplicate_of TEXT);
        CREATE TABLE foundation_scores (document_id TEXT NOT NULL, scorer TEXT NOT NULL,
            care REAL, fairness REAL, loyalty REAL, authority REAL, sanctity REAL,
            liberty REAL, sentiment REAL, moral_word_ratio REAL, matched_words INTEGER,
            PRIMARY KEY (document_id, scorer));
        """
    )
    conn.commit()
    conn.close()

    store = Datastore(path)
    cols = {r["name"] for r in store.conn.execute("PRAGMA table_info(foundation_scores)")}
    assert {"equality", "proportionality"} <= cols
    store.upsert_document(
        doc_id="d", source_id="src_self", stratum_id=None, url=None,
        title="t", published_utc=None, fetched_utc="2026-07-25T00:00:00+00:00",
        word_count=50, minhash=None,
    )
    store.upsert_scores(
        document_id="d", scorer="dictionary", foundations={"fairness": 0.4},
        sentiment=0.0, moral_word_ratio=0.1, matched_words=5,
        equality=0.3, proportionality=0.1,
    )
    rows, _ = store.fairness_split_for_sources(["src_self"])
    assert len(rows) == 1
    store.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "s.sqlite"
    Datastore(path).close()
    store = Datastore(path)          # second open must not re-ALTER
    cols = [r["name"] for r in store.conn.execute("PRAGMA table_info(foundation_scores)")]
    assert cols.count("equality") == 1
    store.close()


# -- aggregation -----------------------------------------------------------

def test_diets_lean_opposite_ways():
    store = _store_with_split()
    profiles = all_persona_fairness(store, _reg())
    assert profiles["self"].leans == "equality"
    assert profiles["modeled_ce"].leans == "proportionality"
    store.close()


def test_profile_reports_coverage():
    store = _store_with_split()
    profile = persona_fairness_profile(store, {"src_self": 1.0})
    assert profile.docs_split == 3 and profile.docs_total == 3
    assert profile.coverage == 1.0
    assert not profile.thin
    store.close()


def test_thin_coverage_is_flagged():
    store = _store_with_split()
    # Add unsplit documents until the split share drops below the threshold.
    for i in range(60):
        did = f"self-pad-{i}"
        store.upsert_document(
            doc_id=did, source_id="src_self", stratum_id=None, url=None,
            title="t", published_utc=None, fetched_utc="2026-07-25T00:00:00+00:00",
            word_count=50, minhash=None,
        )
        store.upsert_scores(
            document_id=did, scorer="dictionary", foundations={"fairness": 0.1},
            sentiment=0.0, moral_word_ratio=0.1, matched_words=2,
        )
    profile = persona_fairness_profile(store, {"src_self": 1.0})
    assert profile.coverage < LOW_COVERAGE
    assert profile.thin
    store.close()


def test_gap_is_none_when_the_pair_is_not_both_profiled():
    store = Datastore(":memory:")
    assert gap({}, pair()) is None
    store.close()


def test_gap_is_oriented_mine_first_not_alphabetically():
    """The gap used to take the first two ids in sorted order, so its sign was an
    accident of spelling: `modeled_ce` sorts before `self`, which inverted it
    relative to CLAUDE.md §3(5)."""
    store = _store_with_split()
    profiles = all_persona_fairness(store, _reg())
    g = gap(profiles, pair())
    assert g["pair"] == ["self", "modeled_ce"]
    # `self` leans equality, so mine-first makes the equality gap positive.
    assert g["equality_gap"] > 0
    store.close()


def test_classic_foundations_are_untouched_by_all_of_this():
    """The headline metrics must keep running on the five-way vocabulary."""
    assert CLASSIC_FOUNDATIONS == ("care", "fairness", "loyalty", "authority", "sanctity")
