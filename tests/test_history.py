"""Snapshot history: date windowing, per-date upsert, reconstruction, series shape."""

from __future__ import annotations

from compare.history import (
    LIVE,
    RECONSTRUCTED,
    backfill_series,
    build_snapshot,
    load_series,
    record_snapshot,
)
from ingestion.datastore import Datastore

# Emphasis profiles used to build a corpus that visibly changes partway through.
CARE_HEAVY = {"care": 0.60, "fairness": 0.20, "loyalty": 0.07, "authority": 0.07, "sanctity": 0.06}
BINDING_HEAVY = {"care": 0.06, "fairness": 0.07, "loyalty": 0.35,
                 "authority": 0.22, "sanctity": 0.30}


def _add(store, diet, day, scores, suffix="a", published=True):
    doc_id = f"{diet}-{day}-{suffix}"
    store.upsert_document(
        doc_id=doc_id, diet_id=diet, source_id="s", stratum_id=None, url=None,
        title="t",
        published_utc=f"{day}T12:00:00+00:00" if published else None,
        fetched_utc=f"{day}T12:00:00+00:00",
        word_count=300, minhash=None,
    )
    store.upsert_scores(
        document_id=doc_id, scorer="dictionary", foundations=scores,
        sentiment=0.0, moral_word_ratio=0.2, matched_words=30,
    )
    return doc_id


def _store(days=("2026-07-01", "2026-07-02", "2026-07-03")):
    store = Datastore(":memory:")
    for day in days:
        _add(store, "self", day, CARE_HEAVY)
        _add(store, "modeled_ce", day, CARE_HEAVY)
    return store


# -- date windowing --------------------------------------------------------

def test_scores_filtered_to_half_open_window():
    store = _store()
    # [07-02, 07-03) is exactly one day: the 2nd.
    rows = store.scores_for_diet("self", "dictionary", since="2026-07-02", until="2026-07-03")
    assert len(rows) == 1
    assert store.doc_count("self", since="2026-07-02", until="2026-07-03") == 1
    assert store.doc_count("self") == 3          # unbounded is still the whole corpus
    store.close()


def test_undated_documents_fall_back_to_fetch_time():
    store = Datastore(":memory:")
    _add(store, "self", "2026-07-05", CARE_HEAVY, published=False)
    assert store.doc_count("self", since="2026-07-05", until="2026-07-06") == 1
    assert store.doc_count("self", since="2026-07-06", until="2026-07-07") == 0
    store.close()


def test_document_date_range_reports_calendar_bounds():
    store = _store()
    assert store.document_date_range() == ("2026-07-01", "2026-07-03")
    store.close()


def test_date_range_of_empty_store_is_none():
    store = Datastore(":memory:")
    assert store.document_date_range() == (None, None)
    store.close()


# -- the two bases ---------------------------------------------------------

def test_window_reacts_to_a_swing_that_cumulative_damps():
    """The point of carrying both bases: the trailing window moves on a week of
    coverage, the all-time average barely does."""
    store = Datastore(":memory:")
    old = [f"2026-07-{d:02d}" for d in range(1, 21)]     # 20 quiet days
    for day in old:
        _add(store, "self", day, CARE_HEAVY)
        _add(store, "modeled_ce", day, CARE_HEAVY)
    for day in ("2026-07-21", "2026-07-22"):             # then the modeled diet swings
        _add(store, "self", day, CARE_HEAVY)
        _add(store, "modeled_ce", day, BINDING_HEAVY)

    snap = build_snapshot(store, "2026-07-22", window_days=7)
    assert snap.window.jsd > snap.cumulative.jsd * 2
    store.close()


def test_single_diet_yields_no_divergence_but_still_records():
    store = Datastore(":memory:")
    _add(store, "self", "2026-07-01", CARE_HEAVY)
    snap = build_snapshot(store, "2026-07-01")
    assert snap.cumulative.jsd is None
    assert snap.cumulative.pair is None
    assert snap.cumulative.diets["self"]["doc_count"] == 1
    store.close()


def test_diet_absent_from_the_window_is_reported_at_zero():
    """A diet going quiet is a fact about that week — it must not silently vanish
    from the series and reshape the comparison."""
    store = _store()
    _add(store, "self", "2026-07-10", CARE_HEAVY)        # only `self` publishes later
    snap = build_snapshot(store, "2026-07-10", window_days=3)
    assert snap.window.diets["modeled_ce"]["doc_count"] == 0
    assert snap.window.diets["modeled_ce"]["composition"] is None
    assert snap.window.jsd is None                       # one scored diet in-window
    assert snap.cumulative.jsd is not None               # but all-time still compares
    store.close()


def test_values_are_rounded_for_storage():
    store = _store()
    snap = build_snapshot(store, "2026-07-03")
    comp = snap.cumulative.diets["self"]["composition"]
    assert all(v == round(v, 6) for v in comp.values())
    store.close()


# -- persistence -----------------------------------------------------------

def test_same_day_reruns_leave_one_row():
    store = _store()
    record_snapshot(store, "2026-07-03")
    record_snapshot(store, "2026-07-03")
    record_snapshot(store, "2026-07-03")
    assert store.snapshot_count() == 1
    store.close()


def test_series_is_chronological_and_carries_both_bases():
    store = _store()
    for day in ("2026-07-03", "2026-07-01", "2026-07-02"):   # recorded out of order
        record_snapshot(store, day)
    series = load_series(store)
    assert [s["date"] for s in series] == ["2026-07-01", "2026-07-02", "2026-07-03"]
    for s in series:
        assert s["source"] == LIVE
        assert s["window_days"] == 7
        assert set(s["cumulative"]["diets"]) == {"modeled_ce", "self"}
        assert s["jsd_cumulative"] is not None
    store.close()


def test_limit_keeps_the_most_recent_still_oldest_first():
    store = _store()
    for day in ("2026-07-01", "2026-07-02", "2026-07-03"):
        record_snapshot(store, day)
    assert [s["date"] for s in load_series(store, limit=2)] == ["2026-07-02", "2026-07-03"]
    store.close()


# -- reconstruction --------------------------------------------------------

def test_backfill_reconstructs_past_days_and_marks_them():
    store = _store()
    written = backfill_series(store, days=10, end_date="2026-07-03")
    assert written == ["2026-07-01", "2026-07-02", "2026-07-03"]
    assert all(s["source"] == RECONSTRUCTED for s in load_series(store))
    store.close()


def test_backfill_stops_at_the_oldest_document():
    store = _store()
    backfill_series(store, days=365, end_date="2026-07-03")
    # Never walks back past 07-01 even though 365 days were requested.
    assert store.snapshot_count() == 3
    store.close()


def test_backfill_leaves_live_rows_alone():
    """A reconstruction must not overwrite a row that recorded the corpus as it
    actually stood — the two are not equivalent evidence."""
    store = _store()
    record_snapshot(store, "2026-07-02")
    written = backfill_series(store, days=10, end_date="2026-07-03")
    assert "2026-07-02" not in written
    by_date = {s["date"]: s for s in load_series(store)}
    assert by_date["2026-07-02"]["source"] == LIVE
    assert by_date["2026-07-01"]["source"] == RECONSTRUCTED
    store.close()


def test_backfill_overwrite_is_opt_in():
    store = _store()
    record_snapshot(store, "2026-07-02")
    written = backfill_series(store, days=10, end_date="2026-07-03", overwrite=True)
    assert "2026-07-02" in written
    by_date = {s["date"]: s for s in load_series(store)}
    assert by_date["2026-07-02"]["source"] == RECONSTRUCTED
    store.close()


def test_backfill_on_an_empty_store_is_a_no_op():
    store = Datastore(":memory:")
    assert backfill_series(store, days=30) == []
    assert store.snapshot_count() == 0
    store.close()
