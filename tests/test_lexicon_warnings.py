"""Two ways the dictionary silently scores with the wrong instrument.

Both were found the same way: by someone following the documented setup and
getting numbers that looked fine. Neither crashed, neither logged, and both
produced a dashboard whose caveat was wrong in the direction of confidence.
"""

from __future__ import annotations

import logging

import ingestion.pipeline as pipeline
from ingestion.datastore import Datastore
from ingestion.pipeline import _check_lexicon_change
from scoring.lexicon import SEED_NAME, build_lexicon

# -- a configured lexicon that isn't there ----------------------------------


def test_a_missing_lexicon_path_warns_rather_than_falling_back_quietly(tmp_path, caplog):
    """settings.example.yaml ships pointing at data/emfd_scoring.csv, which is
    gitignored and absent until you download it. The default configuration lands
    in this branch, and the only signal used to be a caveat at the bottom of a
    rendered page — so a whole corpus gets demo-scored before anyone notices."""
    missing = tmp_path / "emfd_scoring.csv"
    with caplog.at_level(logging.WARNING):
        _, name = build_lexicon(missing)

    assert name == SEED_NAME  # still degrades, as before
    assert "no such file exists" in caplog.text
    assert str(missing) in caplog.text  # says which path it looked at
    assert "not a validated instrument" in caplog.text
    assert "eMFDscore" in caplog.text  # and where to get the real one


def test_no_configured_path_is_not_a_warning(caplog):
    """Running on the seed deliberately is a choice, not a misconfiguration —
    the dashboard caveat already labels it DEMO."""
    with caplog.at_level(logging.WARNING):
        _, name = build_lexicon(None)
    assert name == SEED_NAME
    assert caplog.text == ""


def test_a_present_lexicon_loads_without_complaint(tmp_path, caplog):
    csv = tmp_path / "emfd_scoring.csv"
    csv.write_text("word,care_p,care_sent\nharm,0.8,-1\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        _, name = build_lexicon(csv)
    assert name == "eMFD (emfd_scoring.csv)"
    assert caplog.text == ""


# -- swapping the lexicon under an existing corpus --------------------------


def _store_with(docs: int, lexicon: str | None) -> Datastore:
    store = Datastore(":memory:")
    if lexicon:
        store.set_meta("lexicon", lexicon)
    for i in range(docs):
        store.upsert_document(
            doc_id=f"d{i}",
            diet_id="self",
            source_id="s",
            stratum_id=None,
            url=None,
            title="t",
            published_utc=None,
            fetched_utc="2026-07-29T00:00:00+00:00",
            word_count=200,
            minhash=None,
        )
    return store


def test_changing_lexicon_on_a_populated_store_warns(caplog):
    """Dictionary scores all land under the single scorer key 'dictionary', and
    ingestion skips documents it already has — so a swap appends a second
    instrument's numbers into the same column and every aggregate blends them,
    while the meta label reports only the new name."""
    store = _store_with(5, SEED_NAME)
    try:
        with caplog.at_level(logging.WARNING):
            _check_lexicon_change(store, "eMFD (emfd_scoring.csv)")
    finally:
        store.close()

    assert "lexicon changed" in caplog.text
    assert SEED_NAME in caplog.text  # names both instruments
    assert "emfd_scoring.csv" in caplog.text
    assert "NOT re-scored" in caplog.text
    assert "fresh datastore" in caplog.text  # the only clean answer


def test_an_empty_store_does_not_warn(caplog):
    """Nothing to blend yet — the first run after configuring a lexicon is the
    normal case, not a mistake."""
    store = _store_with(0, SEED_NAME)
    try:
        with caplog.at_level(logging.WARNING):
            _check_lexicon_change(store, "eMFD (emfd_scoring.csv)")
    finally:
        store.close()
    assert caplog.text == ""


def test_the_same_lexicon_does_not_warn(caplog):
    store = _store_with(5, "eMFD (emfd_scoring.csv)")
    try:
        with caplog.at_level(logging.WARNING):
            _check_lexicon_change(store, "eMFD (emfd_scoring.csv)")
    finally:
        store.close()
    assert caplog.text == ""


def test_a_store_with_no_recorded_lexicon_does_not_warn(caplog):
    """Pre-dates the meta key; there is nothing to compare against, and guessing
    would produce a warning on every run of an older datastore."""
    store = _store_with(5, None)
    try:
        with caplog.at_level(logging.WARNING):
            _check_lexicon_change(store, "eMFD (emfd_scoring.csv)")
    finally:
        store.close()
    assert caplog.text == ""


class _Registry:
    def ingestable(self, _kinds, _source_ids=None):
        return []


def test_a_real_run_warns_and_records_the_new_name(monkeypatch, tmp_path, caplog):
    """End to end through `run`, because the ordering is the whole trick:
    `set_meta` clobbers the previous name, so a check placed after it would
    compare a value against itself and could never fire."""
    store = _store_with(3, SEED_NAME)
    monkeypatch.setattr(pipeline, "parse_feed", lambda *a, **k: [])
    csv = tmp_path / "emfd_scoring.csv"
    csv.write_text("word,care_p,care_sent\nharm,0.8,-1\n", encoding="utf-8")

    cfg = pipeline.PipelineConfig(
        lexicon_path=str(csv), transformer_enabled=False, liberty_enabled=False
    )
    try:
        with caplog.at_level(logging.WARNING):
            pipeline.run(store, _Registry(), cfg)
        assert "lexicon changed" in caplog.text
        assert store.get_meta("lexicon") == "eMFD (emfd_scoring.csv)"
    finally:
        store.close()


def test_a_real_run_with_an_unchanged_lexicon_stays_quiet(monkeypatch, caplog):
    store = _store_with(3, SEED_NAME)
    monkeypatch.setattr(pipeline, "parse_feed", lambda *a, **k: [])
    cfg = pipeline.PipelineConfig(
        lexicon_path=None, transformer_enabled=False, liberty_enabled=False
    )
    try:
        with caplog.at_level(logging.WARNING):
            pipeline.run(store, _Registry(), cfg)
        assert "lexicon changed" not in caplog.text
        assert store.get_meta("lexicon") == SEED_NAME
    finally:
        store.close()
