"""Datastore: schema round-trip, dedup flags, and diet aggregation reads."""

from __future__ import annotations

from ingestion.datastore import Datastore


def _store():
    return Datastore(":memory:")


def test_document_and_score_roundtrip():
    store = _store()
    store.upsert_document(
        doc_id="abc", diet_id="self", source_id="src", stratum_id="st",
        url="http://x", title="T", published_utc=None, fetched_utc="2026-07-23T00:00:00+00:00",
        word_count=120, minhash=[1, 2, 3], weight=0.4,
    )
    store.upsert_scores(
        document_id="abc", scorer="dictionary",
        foundations={"care": 0.1, "fairness": 0.2, "loyalty": 0.0, "authority": 0.0, "sanctity": 0.0},
        sentiment=0.05, moral_word_ratio=0.3, matched_words=36, liberty=None,
    )
    assert store.has_document("abc")
    rows = store.scores_for_diet("self")
    assert len(rows) == 1
    assert rows[0]["care"] == 0.1
    assert rows[0]["liberty"] is None
    assert rows[0]["weight"] == 0.4


def test_duplicates_excluded_from_reads():
    store = _store()
    store.upsert_document(
        doc_id="dup", diet_id="self", source_id="s", stratum_id=None, url=None,
        title=None, published_utc=None, fetched_utc="2026-07-23T00:00:00+00:00",
        word_count=80, minhash=[9], is_duplicate=True, duplicate_of="canon",
    )
    store.upsert_scores(
        document_id="dup", scorer="dictionary",
        foundations={"care": 0.5}, sentiment=0.0, moral_word_ratio=0.1, matched_words=8,
    )
    assert store.scores_for_diet("self") == []  # duplicates excluded
    assert list(store.iter_minhash_signatures()) == []  # and not seeded into the index
    assert store.counts() == {"documents": 1, "duplicates": 1, "unique": 0}


def test_minhash_signatures_returned_for_unique_docs():
    store = _store()
    store.upsert_document(
        doc_id="u1", diet_id="modeled_ce", source_id="s", stratum_id=None, url=None,
        title=None, published_utc=None, fetched_utc="2026-07-23T00:00:00+00:00",
        word_count=100, minhash=[4, 5, 6],
    )
    sigs = dict(store.iter_minhash_signatures())
    assert sigs == {"u1": [4, 5, 6]}
