"""The weekly brief, end to end: store -> payload -> email."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from daily.runner import DailyConfig, _step_digest, _weekday
from dashboard.export import build_payload
from digest.render import build_digest
from ingestion.datastore import Datastore
from tests.registries import two_personas
from tests.test_weekly import CALM, MIN_WEEK_DOCS, _doc

TODAY = datetime.now(UTC).date()


def _store(theirs_this_week: dict) -> Datastore:
    store = Datastore(":memory:")
    for back in range(9):
        day = TODAY - timedelta(days=7 * back)
        theirs = theirs_this_week if back == 0 else dict(CALM, sanctity=0.2 + 0.02 * (back % 3))
        for i in range(MIN_WEEK_DOCS):
            _doc(store, f"m{back}-{i}", "s_mine", day, CALM)
            _doc(store, f"t{back}-{i}", "s_theirs", day, theirs)
        # A story both sides cover, every week, so it is part of the baseline too.
        for i in range(3):
            title = "Chaplaincy bill passes"
            _doc(store, f"sm{back}-{i}", "s_mine", day, dict(CALM, care=0.9), title)
            _doc(store, f"st{back}-{i}", "s_theirs", day, dict(CALM, sanctity=0.9), title)
    store.replace_clustering(
        [(1, "chaplaincy", 6)], [(f"s{s}0-{i}", 1) for s in "mt" for i in range(3)]
    )
    return store


def test_a_week_that_moved_leads_with_the_move():
    store = _store(dict(CALM, sanctity=0.9))
    reg, pair = two_personas()
    digest = build_digest(build_payload(store, reg, pair))
    # Shares are zero-sum, so the spike nudges the others out of range too; the
    # subject still names the one that moved most.
    assert digest.subject.startswith("Parallax · Sanctity shifted toward the modeled_ce diet")
    for section in ("What moved", "Where the sanctity gap comes from", "One story, two framings"):
        assert section in digest.html
    assert "Chaplaincy bill passes" in digest.html
    assert "OUTSIDE RANGE" in digest.text
    assert "Parallax weekly brief" in digest.html


def test_a_quiet_week_says_so():
    store = _store(dict(CALM, sanctity=0.22))
    reg, pair = two_personas()
    digest = build_digest(build_payload(store, reg, pair))
    assert digest.subject.startswith("Parallax · Quiet week")
    assert "A quiet week." in digest.html
    assert "Where the" not in digest.html


def test_a_blocked_source_shows_up_in_the_brief():
    store = _store(CALM)
    reg, pair = two_personas()
    now = datetime.now(UTC)
    for days in (2, 1, 0):
        when = (now - timedelta(days=days)).isoformat()
        store.record_source_run("s_theirs", when, "blocked", 10, 0, 10)
    digest = build_digest(build_payload(store, reg, pair))
    assert "Sources that need a look" in digest.html
    assert "100% of its weighted diet is affected" in digest.text


def test_the_digest_waits_for_its_weekday_and_catches_up(monkeypatch):
    monkeypatch.setattr("digest.send.send", lambda *a, **k: None)
    monkeypatch.setattr("dashboard.export.build_payload", lambda *a, **k: {})
    monkeypatch.setattr("digest.render.build_digest", lambda *a, **k: SimpleNamespace(subject="s"))
    store = Datastore(":memory:")
    yesterday = (TODAY - timedelta(days=1)).strftime("%a")  # "Thu"-style works too
    cfg = DailyConfig(digest_weekday=_weekday(yesterday))
    reg, pair = two_personas()
    # Yesterday's run never happened, so today's sends it; tomorrow's does not again.
    assert _step_digest(store, cfg, reg, pair).startswith("sent")
    assert _step_digest(store, cfg, reg, pair).startswith("not sent")


def test_a_misspelled_weekday_is_refused_at_load():
    import pytest

    assert _weekday("Fri") == _weekday("friday") == "friday"
    assert _weekday(None) is None
    for bad in ("fridays", "fr", 5):
        with pytest.raises(ValueError):
            _weekday(bad)


def test_divergence_outside_its_band_is_not_called_inside():
    from digest.render import _trend_context

    jsd = {"now": 0.08, "low": 0.0, "high": 0.07, "moved": False}
    assert "Inside" not in _trend_context(jsd, 9)
    assert _trend_context(dict(jsd, now=0.05), 9).startswith("Inside")


def _weekly_payload(foundations, jsd=None, framings=None, diets=2):
    """A hand-built weekly payload, for wording that a store fixture can't pin down."""
    ids = ["self", "modeled_ce", "other"][:diets]
    return {
        "generated_utc": "2026-10-23T00:00:00+00:00",
        "diets": [{"id": i, "short_label": f"The {i} diet"} for i in ids],
        "comparison": {"pair": ["self", "modeled_ce"]},
        "blindspots": [],
        "weekly": {
            "pair": ["self", "modeled_ce"],
            "week_start": "2026-10-17",
            "week_end": "2026-10-23",
            "counts": {"self": 30, "modeled_ce": 30},
            "baseline_weeks": 8,
            "foundations": foundations,
            "jsd": jsd or {"now": 0.02, "low": 0.01, "high": 0.03, "moved": False, "series": []},
            "framings": framings or [],
            "drivers": {},
        },
    }


def _f(name, now, moved=False, shift=0.0, low=-0.1, high=0.1):
    return {
        "foundation": name,
        "now": now,
        "low": low,
        "high": high,
        "median": 0.0,
        "moved": moved,
        "shift": shift,
    }


def test_a_divergence_break_is_not_called_a_quiet_week():
    calm = [_f(n, 0.0) for n in ("care", "fairness")]
    jsd = {"now": 0.05, "low": 0.01, "high": 0.03, "moved": True, "series": []}
    digest = build_digest(_weekly_payload(calm, jsd))
    assert "Quiet" not in digest.subject and "quiet" not in digest.html
    assert "overall gap is the widest" in digest.html


def test_the_lead_move_is_the_largest_share_shift_and_the_only_one_named():
    moved = [
        _f("care", 0.3, True, shift=0.17),  # the one that moved
        _f("fairness", -0.4, True, shift=0.05),  # pushed further in log space
    ]
    digest = build_digest(_weekly_payload(moved))
    assert digest.subject.startswith("Parallax · Care shifted toward the self diet")
    assert "Fairness shifted" not in digest.html
    assert "1 other foundation(s) left their range with it" in digest.html


def test_the_weekly_brief_says_the_pair_is_a_choice():
    digest = build_digest(_weekly_payload([_f("care", 0.0)], diets=3))
    assert "+1 more persona in the dashboard" in digest.html
    assert "+1 more persona in the dashboard" in digest.text


def test_only_http_links_become_links():
    side = {"docs": 3, "top": [], "headline": {"title": "T", "url": "javascript:alert(1)"}}
    framing = {"title": "Story", "jsd": 0.1, "sides": {"self": side, "modeled_ce": side}}
    digest = build_digest(_weekly_payload([_f("care", 0.0)], framings=[framing]))
    assert "javascript:" not in digest.html and "javascript:" not in digest.text
