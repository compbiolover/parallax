"""The weekly brief, end to end: store -> payload -> email."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from daily.runner import DailyConfig, _step_digest
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


def test_the_digest_waits_for_its_weekday():
    other = (TODAY + timedelta(days=1)).strftime("%A").lower()
    cfg = DailyConfig(digest_weekday=other)
    assert _step_digest(None, cfg, None, None).startswith("not sent")
