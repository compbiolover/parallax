"""Snapshot history: date windowing, per-date upsert, reconstruction, series shape."""

from __future__ import annotations

import pytest

from compare.history import (
    LIVE,
    RECONSTRUCTED,
    backfill_series,
    build_snapshot,
    load_series,
    record_snapshot,
)
from ingestion.datastore import Datastore

from .registries import pair, registry

# Emphasis profiles used to build a corpus that visibly changes partway through.
CARE_HEAVY = {"care": 0.60, "fairness": 0.20, "loyalty": 0.07, "authority": 0.07, "sanctity": 0.06}
BINDING_HEAVY = {
    "care": 0.06,
    "fairness": 0.07,
    "loyalty": 0.35,
    "authority": 0.22,
    "sanctity": 0.30,
}


def _add(store, diet, day, scores, suffix="a", published=True):
    doc_id = f"{diet}-{day}-{suffix}"
    store.upsert_document(
        doc_id=doc_id,
        source_id=f"src_{diet}",
        stratum_id=None,
        url=None,
        title="t",
        published_utc=f"{day}T12:00:00+00:00" if published else None,
        fetched_utc=f"{day}T12:00:00+00:00",
        word_count=300,
        minhash=None,
    )
    store.upsert_scores(
        document_id=doc_id,
        scorer="dictionary",
        foundations=scores,
        sentiment=0.0,
        moral_word_ratio=0.2,
        matched_words=30,
    )
    return doc_id


def _reg(*personas):
    """A registry over `src_<persona>` sources, one per persona."""
    names = personas or ("self", "modeled_ce")
    return registry(**{name: {f"src_{name}": 1.0} for name in names})


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
    mine = ["src_self"]
    rows = store.scores_for_sources(mine, "dictionary", since="2026-07-02", until="2026-07-03")
    assert len(rows) == 1
    assert store.doc_count_for_sources(mine, since="2026-07-02", until="2026-07-03") == 1
    # Unbounded is still the whole corpus.
    assert store.doc_count_for_sources(mine) == 3
    store.close()


def test_undated_documents_fall_back_to_fetch_time():
    store = Datastore(":memory:")
    _add(store, "self", "2026-07-05", CARE_HEAVY, published=False)
    assert store.doc_count_for_sources(["src_self"], since="2026-07-05", until="2026-07-06") == 1
    assert store.doc_count_for_sources(["src_self"], since="2026-07-06", until="2026-07-07") == 0
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
    old = [f"2026-07-{d:02d}" for d in range(1, 21)]  # 20 quiet days
    for day in old:
        _add(store, "self", day, CARE_HEAVY)
        _add(store, "modeled_ce", day, CARE_HEAVY)
    for day in ("2026-07-21", "2026-07-22"):  # then the modeled diet swings
        _add(store, "self", day, CARE_HEAVY)
        _add(store, "modeled_ce", day, BINDING_HEAVY)

    snap = build_snapshot(store, _reg(), pair(), "2026-07-22", window_days=7)
    assert snap.window.jsd > snap.cumulative.jsd * 2
    store.close()


def test_single_diet_yields_no_divergence_but_still_records():
    store = Datastore(":memory:")
    _add(store, "self", "2026-07-01", CARE_HEAVY)
    snap = build_snapshot(store, _reg(), pair(), "2026-07-01")
    assert snap.cumulative.jsd is None
    assert snap.cumulative.pair is None
    assert snap.cumulative.diets["self"]["doc_count"] == 1
    store.close()


def test_diet_absent_from_the_window_is_reported_at_zero():
    """A diet going quiet is a fact about that week — it must not silently vanish
    from the series and reshape the comparison."""
    store = _store()
    _add(store, "self", "2026-07-10", CARE_HEAVY)  # only `self` publishes later
    snap = build_snapshot(store, _reg(), pair(), "2026-07-10", window_days=3)
    assert snap.window.diets["modeled_ce"]["doc_count"] == 0
    assert snap.window.diets["modeled_ce"]["composition"] is None
    assert snap.window.jsd is None  # one scored diet in-window
    assert snap.cumulative.jsd is not None  # but all-time still compares
    store.close()


def test_values_are_rounded_for_storage():
    store = _store()
    snap = build_snapshot(store, _reg(), pair(), "2026-07-03")
    comp = snap.cumulative.diets["self"]["composition"]
    assert all(v == round(v, 6) for v in comp.values())
    store.close()


# -- persistence -----------------------------------------------------------


def test_same_day_reruns_leave_one_row():
    store = _store()
    record_snapshot(store, _reg(), pair(), "2026-07-03")
    record_snapshot(store, _reg(), pair(), "2026-07-03")
    record_snapshot(store, _reg(), pair(), "2026-07-03")
    assert store.snapshot_count() == 1
    store.close()


def test_series_is_chronological_and_carries_both_bases():
    store = _store()
    for day in ("2026-07-03", "2026-07-01", "2026-07-02"):  # recorded out of order
        record_snapshot(store, _reg(), pair(), day)
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
        record_snapshot(store, _reg(), pair(), day)
    assert [s["date"] for s in load_series(store, limit=2)] == ["2026-07-02", "2026-07-03"]
    store.close()


# -- reconstruction --------------------------------------------------------


def test_backfill_reconstructs_past_days_and_marks_them():
    store = _store()
    written = backfill_series(store, _reg(), pair(), days=10, end_date="2026-07-03")
    assert written == ["2026-07-01", "2026-07-02", "2026-07-03"]
    assert all(s["source"] == RECONSTRUCTED for s in load_series(store))
    store.close()


def test_backfill_stops_at_the_oldest_document():
    store = _store()
    backfill_series(store, _reg(), pair(), days=365, end_date="2026-07-03")
    # Never walks back past 07-01 even though 365 days were requested.
    assert store.snapshot_count() == 3
    store.close()


def test_backfill_leaves_live_rows_alone():
    """A reconstruction must not overwrite a row that recorded the corpus as it
    actually stood — the two are not equivalent evidence."""
    store = _store()
    record_snapshot(store, _reg(), pair(), "2026-07-02")
    written = backfill_series(store, _reg(), pair(), days=10, end_date="2026-07-03")
    assert "2026-07-02" not in written
    by_date = {s["date"]: s for s in load_series(store)}
    assert by_date["2026-07-02"]["source"] == LIVE
    assert by_date["2026-07-01"]["source"] == RECONSTRUCTED
    store.close()


def test_backfill_overwrite_is_opt_in():
    store = _store()
    record_snapshot(store, _reg(), pair(), "2026-07-02")
    written = backfill_series(store, _reg(), pair(), days=10, end_date="2026-07-03", overwrite=True)
    assert "2026-07-02" in written
    by_date = {s["date"]: s for s in load_series(store)}
    assert by_date["2026-07-02"]["source"] == RECONSTRUCTED
    store.close()


def test_backfill_on_an_empty_store_is_a_no_op():
    store = Datastore(":memory:")
    assert backfill_series(store, _reg(), pair(), days=30) == []
    assert store.snapshot_count() == 0
    store.close()


# -- the CLI, which nothing exercised ----------------------------------------
# `--record` and `--backfill` were both broken and had been for as long as the
# registry has been a parameter. `record_snapshot(store, registry, pair, ...)`
# was called as `record_snapshot(store, window_days=...)` — a TypeError — and
# `backfill_series(store, registry, pair, days=...)` was called positionally
# with two ints, so an int arrived where a registry belonged.
#
# Nothing caught it because `make history` runs the CLI with no flags, and the
# plain listing path never touches either function. The flags are the manual
# recovery path after a failed scheduled run, which is exactly when nobody wants
# to discover them broken.


def _cli_store(tmp_path, monkeypatch):
    """A real store on disk, with the CLI pointed at it and at a real registry."""
    db = tmp_path / "cli.sqlite"
    store = _store()
    dest = Datastore(str(db))
    for row in store.conn.execute("SELECT * FROM documents"):
        dest.conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(id, source_id, stratum_id, url, title, published_utc, fetched_utc, word_count) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                row["id"],
                row["source_id"],
                row["stratum_id"],
                row["url"],
                row["title"],
                row["published_utc"],
                row["fetched_utc"],
                row["word_count"],
            ),
        )
    for row in store.conn.execute("SELECT * FROM foundation_scores"):
        dest.conn.execute(
            "INSERT OR REPLACE INTO foundation_scores "
            "(document_id, scorer, care, fairness, loyalty, authority, sanctity) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                row["document_id"],
                row["scorer"],
                row["care"],
                row["fairness"],
                row["loyalty"],
                row["authority"],
                row["sanctity"],
            ),
        )
    dest.conn.commit()
    dest.close()
    store.close()

    monkeypatch.setattr(
        "ingestion.config.load_registry", lambda **kw: _reg("mine", "theirs"), raising=True
    )
    return db


def test_cli_record_actually_records(tmp_path, monkeypatch, capsys):
    from compare import history as history_module

    db = _cli_store(tmp_path, monkeypatch)

    assert history_module.main(["--db", str(db), "--record"]) == 0
    out = capsys.readouterr().out
    assert "Recorded live snapshot for" in out


def test_cli_backfill_actually_backfills(tmp_path, monkeypatch, capsys):
    from compare import history as history_module

    db = _cli_store(tmp_path, monkeypatch)

    assert history_module.main(["--db", str(db), "--backfill", "5"]) == 0
    assert "Reconstructed" in capsys.readouterr().out


def test_cli_listing_still_works_without_flags(tmp_path, monkeypatch, capsys):
    """The one path that did work, kept working."""
    from compare import history as history_module

    db = _cli_store(tmp_path, monkeypatch)

    assert history_module.main(["--db", str(db)]) == 0
    # Nothing has been recorded into this fresh store, so the listing says so
    # rather than printing an empty table.
    assert "No snapshots recorded yet" in capsys.readouterr().out


def test_cli_listing_does_not_need_a_loadable_registry(tmp_path, monkeypatch, capsys):
    """Inspecting a database must not require the configuration to be valid.

    The listing path reads nothing but the SQLite file. Resolving the registry
    for it anyway would take away the one command that still works when the
    config is the thing that broke — which is exactly when you want to look at
    the history.
    """
    from compare import history as history_module

    db = _cli_store(tmp_path, monkeypatch)

    def _broken(**kwargs):
        raise ValueError("duplicate persona id in registry")

    monkeypatch.setattr("ingestion.config.load_registry", _broken, raising=True)

    assert history_module.main(["--db", str(db)]) == 0
    assert "No snapshots recorded yet" in capsys.readouterr().out

    # And the writers still fail loudly, rather than inventing a pair.
    with pytest.raises(ValueError, match="duplicate persona id"):
        history_module.main(["--db", str(db), "--record"])
