"""Transformer-at-ingestion + dictionary-vs-transformer confidence bands."""

from __future__ import annotations

from cluster.embed import HashingEmbedder
from compare.confidence import persona_band
from dashboard.export import build_payload
from ingestion.datastore import Datastore
from ingestion.pipeline import RunStats, _build_transformer, _ingest_one
from scoring.dictionary import DictionaryScorer
from scoring.foundations import CLASSIC_FOUNDATIONS

from .registries import pair, registry


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


def _one_source_registry(source_id: str):
    """A registry whose single persona reads just this source."""
    return registry(self={source_id: 1.0})


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
    rows = store.scores_for_sources([_source().id], "transformer/stub")
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


class _RaisingTransformer:
    name = "transformer/boom"

    def score(self, text: str) -> dict[str, float]:
        raise RuntimeError("model exploded")


def test_transformer_failure_does_not_abort_or_half_ingest():
    # A transformer that raises must not kill the run or leave the doc without an
    # embedding — the dictionary row and embedding still land, just no band row.
    from ingestion.dedup import NearDuplicateIndex

    store = Datastore(":memory:")
    stats = RunStats()
    _ingest_one(
        store, _source(), DictionaryScorer(), HashingEmbedder(dim=32),
        NearDuplicateIndex(), stats,
        title="A headline that should still ingest", link="https://x.com/c",
        published_utc=None, text="care help harm compassion victim suffer people",
        cluster_text="A headline", min_words=3, transformer=_RaisingTransformer(),
    )
    assert stats.stored == 1
    assert store.scorer_names() == ["dictionary"]        # no transformer row
    assert store.embedding_count() == 1                  # embedding still stored
    store.close()


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


def _weights():
    return {"s": 1.0}


def test_persona_band_point_is_between_the_two_estimates():
    store = _store_with_paired_scores()
    bands = persona_band(store, _weights(), "transformer/stub")
    care = bands["care"]
    assert care.low == min(care.dictionary, care.transformer)
    assert care.high == max(care.dictionary, care.transformer)
    assert care.low <= care.point <= care.high
    store.close()


def test_persona_band_disagreement_uses_ensemble_vote_convention():
    # loyalty: dict rate>0 on doc0 (present), transformer 0.1<=0.5 (absent) -> split;
    # doc1 has no loyalty on either -> agree(absent). So 1/2 docs split.
    store = _store_with_paired_scores()
    bands = persona_band(store, _weights(), "transformer/stub")
    assert bands["loyalty"].disagreement == 0.5
    assert bands["care"].disagreement == 0.0   # both fire care on both docs
    store.close()


def test_persona_band_none_without_transformer_rows():
    store = _store_with_paired_scores()
    assert persona_band(store, _weights(), "does-not-exist") is None
    store.close()


def test_persona_band_ignores_dictionary_only_backfill_docs():
    # Both compositions must be built over the *paired* docs only. A dict-only
    # doc (as GDELT backfill produces) would shift the dictionary composition if
    # wrongly included, making the band conflate population with method.
    store = Datastore(":memory:")
    # paired doc: dictionary says care, transformer says loyalty
    store.upsert_document(doc_id="p", diet_id="self", source_id="s", stratum_id=None,
                          url=None, title="t", published_utc=None,
                          fetched_utc="2026-07-24T00:00:00+00:00", word_count=90, minhash=None)
    store.upsert_scores(document_id="p", scorer="dictionary", foundations={"care": 1.0},
                        sentiment=0.0, moral_word_ratio=0.1, matched_words=5)
    store.upsert_scores(document_id="p", scorer="transformer/stub", foundations={"loyalty": 1.0},
                        sentiment=0.0, moral_word_ratio=0.0, matched_words=0)
    # dict-only backfill doc that would drag the dict composition toward loyalty
    store.upsert_document(doc_id="bf", diet_id="self", source_id="s", stratum_id=None,
                          url=None, title="t2", published_utc=None,
                          fetched_utc="2026-07-24T00:00:00+00:00", word_count=90, minhash=None)
    store.upsert_scores(document_id="bf", scorer="dictionary", foundations={"loyalty": 1.0},
                        sentiment=0.0, moral_word_ratio=0.1, matched_words=5)

    bands = persona_band(store, _weights(), "transformer/stub")
    # Paired-only: dict composition is pure care (1.0). If the backfill doc leaked
    # in it would be care=0.5, loyalty=0.5.
    assert bands["care"].dictionary == 1.0
    assert bands["loyalty"].dictionary == 0.0
    assert bands["loyalty"].transformer == 1.0
    store.close()


# -- export ---------------------------------------------------------------

def test_payload_has_bands_when_transformer_scorer_recorded():
    store = _store_with_paired_scores()
    store.set_meta("transformer_scorer", "transformer/stub")
    p = build_payload(store, registry(self={"s": 1.0}), pair())
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
    p = build_payload(store, registry(self={"s": 1.0}), pair())
    assert p["has_confidence_bands"] is False
    assert p["band_scorers"] is None
    assert p["diets"][0]["band"] is None
    store.close()
