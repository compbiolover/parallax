"""Weekly movement, shared-story framings, and gap drivers."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from compare.weekly import MIN_WEEK_DOCS, gap_drivers, weekly_movement
from ingestion.datastore import Datastore
from tests.registries import two_personas

END = date(2026, 10, 22)
CALM = {"care": 0.3, "fairness": 0.3, "loyalty": 0.1, "authority": 0.1, "sanctity": 0.2}


def _doc(store, doc_id, source, day, scores, title="Headline"):
    store.upsert_document(
        doc_id=doc_id,
        diet_id="",
        source_id=source,
        stratum_id=None,
        url=f"https://example.test/{doc_id}",
        title=title,
        published_utc=f"{day.isoformat()}T12:00:00+00:00",
        fetched_utc=f"{day.isoformat()}T13:00:00+00:00",
        word_count=200,
        minhash=None,
    )
    store.upsert_scores(
        document_id=doc_id,
        scorer="dictionary",
        foundations=scores,
        sentiment=0.0,
        moral_word_ratio=0.1,
        matched_words=10,
        liberty=None,
    )


def _history(store, theirs_this_week: dict, weeks: int = 8, jitter: float = 0.02):
    """MIN_WEEK_DOCS a side per week; theirs gets a little week-to-week wobble."""
    for back in range(weeks + 1):
        day = END - timedelta(days=7 * back)
        theirs = theirs_this_week if back == 0 else dict(CALM, sanctity=0.2 + jitter * (back % 3))
        for i in range(MIN_WEEK_DOCS):
            _doc(store, f"m{back}-{i}", "s_mine", day, CALM)
            _doc(store, f"t{back}-{i}", "s_theirs", day, theirs)


def test_a_sanctity_spike_moves_and_nothing_else_does():
    store = Datastore(":memory:")
    reg, pair = two_personas()
    _history(store, dict(CALM, sanctity=0.9))
    out = weekly_movement(store, reg, pair, end=END)
    moved = {e["foundation"] for e in out["foundations"] if e["moved"]}
    assert "sanctity" in moved
    sanctity = next(e for e in out["foundations"] if e["foundation"] == "sanctity")
    assert sanctity["now"] < sanctity["low"]  # mine-first: theirs leans on it, so negative
    assert out["baseline_weeks"] == 8
    assert len(out["jsd"]["series"]) == 9
    assert out["jsd"]["moved"]


def test_a_week_like_the_others_is_quiet():
    store = Datastore(":memory:")
    reg, pair = two_personas()
    _history(store, dict(CALM, sanctity=0.22))
    out = weekly_movement(store, reg, pair, end=END)
    assert not any(e["moved"] for e in out["foundations"])
    assert out["drivers"] == {}


def test_a_thin_week_returns_none_and_short_history_has_no_baseline():
    store = Datastore(":memory:")
    reg, pair = two_personas()
    _doc(store, "m", "s_mine", END, CALM)
    assert weekly_movement(store, reg, pair, end=END) is None
    store = Datastore(":memory:")
    _history(store, CALM, weeks=2)
    out = weekly_movement(store, reg, pair, end=END)
    assert out["baseline_weeks"] == 2
    assert "moved" not in out["foundations"][0]


def test_framings_need_both_sides_and_rank_by_difference():
    store = Datastore(":memory:")
    reg, pair = two_personas()
    _history(store, CALM)
    for i in range(3):
        _doc(store, f"sm{i}", "s_mine", END, dict(CALM, care=0.9), "Chaplaincy bill passes")
        _doc(store, f"st{i}", "s_theirs", END, dict(CALM, sanctity=0.9), "Chaplaincy bill passes")
        _doc(store, f"one{i}", "s_theirs", END, CALM, "Only one side")
    store.replace_clustering(
        [(1, "chaplaincy", 6), (2, "one", 3)],
        [*((f"s{s}{i}", 1) for s in "mt" for i in range(3)), *((f"one{i}", 2) for i in range(3))],
    )
    framings = weekly_movement(store, reg, pair, end=END)["framings"]
    assert [f["cluster_id"] for f in framings] == [1]
    sides = framings[0]["sides"]
    assert sides["self"]["top"][0]["foundation"] == "care"
    assert sides["modeled_ce"]["top"][0]["foundation"] == "sanctity"
    assert sides["self"]["headline"]["title"] == "Chaplaincy bill passes"


def test_gap_drivers_shares_sum_to_one():
    store = Datastore(":memory:")
    reg, pair = two_personas()
    for i in range(4):
        _doc(store, f"a{i}", "s_theirs", END, dict(CALM, sanctity=0.9))
        _doc(store, f"b{i}", "s_mine", END, dict(CALM, sanctity=0.0))
    store.replace_clustering([(1, "a", 2), (2, "b", 2)], [("a0", 1), ("a1", 1), ("a2", 2)])
    rows = store.clustered_scores(since=END.isoformat())
    weights = {p: reg.weights_for(p) for p in pair}
    drivers = gap_drivers(rows, pair, weights, "sanctity")
    assert [d["cluster_id"] for d in drivers] == [1, 2, None]
    assert sum(d["share"] for d in drivers) == pytest.approx(1.0)
    assert drivers[0]["share"] == pytest.approx(2 * drivers[1]["share"])
