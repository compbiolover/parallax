"""Transformer-at-ingestion + dictionary-vs-transformer confidence bands."""

from __future__ import annotations

from cluster.embed import HashingEmbedder
from compare.confidence import diet_band
from dashboard.export import build_payload
from ingestion.datastore import Datastore
from ingestion.pipeline import RunStats, _build_transformer, _ingest_one
from scoring.dictionary import DictionaryScorer
from scoring.foundations import CLASSIC_FOUNDATIONS


class _StubTransformer:
    """Deterministic P(present) per foundation — no torch, no model download."""

    name = "transformer/stub"

    def __init__(self, probs: dict[str, float]):
        self._probs = probs

    def score(self, text: str) -> dict[str, float]:
        return {f: self._probs.get(f, 0.0) for f in CLASSIC_FOUNDATIONS}


def _source():
    from ingestion.config import load_registry
    return load_registry().backfillable()[0]


# -- ingestion writes a second (transformer) score row --------------------

def test_ingest_one_stores_transformer_row_when_supplied():
    from ingestion.dedup import NearDuplicateIndex

    store = Datastore(":memory:")
    tr = _StubTransformer({"care": 0.9, "loyalty": 0.2})
    _ingest_one(
        store, _source(), DictionaryScorer(), HashingEmbedder(dim=32),
        NearDuplicateIndex(), RunStats(),
        title="A caring headline about people", link="https://x.com/a",
        published_utc=None, text="people care help harm suffer compassion victim",
        cluster_text="A caring headline", min_words=3, transformer=tr,
    )
    assert "transformer/stub" in store.scorer_names()
    assert "dictionary" in store.scorer_names()
    rows = store.scores_for_diet(_source().diet_id, "transformer/stub")
    assert len(rows) == 1 and rows[0]["care"] == 0.9
    store.close()


def test_no_transformer_row_without_transformer():
    from ingestion.dedup import NearDuplicateIndex

    store = Datastore(":memory:")
    _ingest_one(
        store, _source(), DictionaryScorer(), HashingEmbedder(dim=32),
        NearDuplicateIndex(), RunStats(),
        title="Some headline about things", link="https://x.com/b",
        published_utc=None, text="one two three four five six seven",
        cluster_text="Some headline", min_words=3,
    )
    assert store.scorer_names() == ["dictionary"]
    store.close()


def test_build_transformer_disabled_returns_none():
    from ingestion.pipeline import PipelineConfig
    assert _build_transformer(PipelineConfig(transformer_enabled=False)) is None


# -- band math ------------------------------------------------------------

def _store_with_paired_scores():
    """One diet, two docs, dictionary + transformer rows on each."""
    store = Datastore(":memory:")
    for i, (dic, tr) in enumerate([
        # doc 0: both fire care; only dict fires loyalty (a split there)
        ({"care": 0.4, "loyalty": 0.3}, {"care": 0.9, "loyalty": 0.1}),
        # doc 1: both fire care
        ({"care": 0.6}, {"care": 0.8}),
    ]):
        did = f"d{i}"
        store.upsert_document(doc_id=did, diet_id="self", source_id="s", stratum_id=None,
                              url=None, title="t", published_utc=None,
                              fetched_utc="2026-07-24T00:00:00+00:00", word_count=90, minhash=None)
        store.upsert_scores(document_id=did, scorer="dictionary", foundations=dic,
                            sentiment=0.0, moral_word_ratio=0.1, matched_words=5)
        store.upsert_scores(document_id=did, scorer="transformer/stub", foundations=tr,
                            sentiment=0.0, moral_word_ratio=0.0, matched_words=0)
    return store


def test_diet_band_point_is_between_the_two_estimates():
    store = _store_with_paired_scores()
    bands = diet_band(store, "self", "transformer/stub")
    care = bands["care"]
    assert care.low == min(care.dictionary, care.transformer)
    assert care.high == max(care.dictionary, care.transformer)
    assert care.low <= care.point <= care.high
    store.close()


def test_diet_band_disagreement_uses_ensemble_vote_convention():
    # loyalty: dict rate>0 on doc0 (present), transformer 0.1<=0.5 (absent) -> split;
    # doc1 has no loyalty on either -> agree(absent). So 1/2 docs split.
    store = _store_with_paired_scores()
    bands = diet_band(store, "self", "transformer/stub")
    assert bands["loyalty"].disagreement == 0.5
    assert bands["care"].disagreement == 0.0   # both fire care on both docs
    store.close()


def test_diet_band_none_without_transformer_rows():
    store = _store_with_paired_scores()
    assert diet_band(store, "self", "does-not-exist") is None
    store.close()


# -- export ---------------------------------------------------------------

def test_payload_has_bands_when_transformer_scorer_recorded():
    store = _store_with_paired_scores()
    store.set_meta("transformer_scorer", "transformer/stub")
    p = build_payload(store)
    assert p["has_confidence_bands"] is True
    assert p["band_scorers"]["transformer"] == "transformer/stub"
    diet = p["diets"][0]
    assert diet["band"] is not None
    assert set(diet["band"]["care"]) == {
        "point", "low", "high", "dictionary", "transformer", "disagreement",
    }
    assert "confidence" in p["caveat"].lower() or "disagree" in p["caveat"].lower()
    store.close()


def test_payload_no_bands_without_transformer_meta():
    store = _store_with_paired_scores()   # rows exist but meta not set
    p = build_payload(store)
    assert p["has_confidence_bands"] is False
    assert p["band_scorers"] is None
    assert p["diets"][0]["band"] is None
    store.close()
