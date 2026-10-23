"""Source health: the per-run ledger, and what it flags."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ingestion.datastore import Datastore
from ingestion.health import format_report, health_payload, source_issues
from ingestion.pipeline import _run_status
from tests.registries import registry

NOW = datetime(2026, 10, 23, 12, tzinfo=UTC)


def _runs(store, source_id, statuses, stored=10, start_days_ago=None):
    """One run a day, oldest first, ending today."""
    n = len(statuses)
    for i, status in enumerate(statuses):
        day = NOW - timedelta(days=(start_days_ago or n - 1) - i)
        store.record_source_run(source_id, day.isoformat(), status, 10, stored, 0)


def test_run_status_classifies_each_failure():
    assert _run_status(True, 0, 1) == "unreadable"
    assert _run_status(False, 0, 0) == "empty"
    assert _run_status(False, 5, 5) == "blocked"
    assert _run_status(False, 5, 2) == "ok"


def test_three_failed_runs_flag_a_source_and_one_does_not():
    store = Datastore(":memory:")
    reg = registry(mine={"a": 1.0, "b": 3.0}, theirs={"b": 1.0})
    _runs(store, "a", ["ok", "ok", "unreadable", "unreadable", "unreadable"])
    _runs(store, "b", ["ok", "ok", "ok", "ok", "blocked"])
    issues = source_issues(store, reg, NOW)
    assert [(i.source_id, i.status) for i in issues] == [("a", "unreadable")]
    assert issues[0].last_ok == (NOW - timedelta(days=3)).date().isoformat()


def test_volume_drop_is_flagged_against_the_prior_weeks():
    store = Datastore(":memory:")
    reg = registry(mine={"a": 1.0})
    _runs(store, "a", ["ok"] * 28, stored=5, start_days_ago=34)  # ~35 a week before
    _runs(store, "a", ["ok"] * 6, stored=0)  # nothing new this week
    issues = source_issues(store, reg, NOW)
    assert [i.status for i in issues] == ["dropping"]


def test_persona_impact_is_the_weighted_share_affected():
    store = Datastore(":memory:")
    reg = registry(mine={"a": 1.0, "b": 3.0}, theirs={"b": 1.0})
    _runs(store, "a", ["blocked"] * 3)
    payload = health_payload(store, reg, NOW)
    assert payload["persona_impact"] == {"mine": 0.25}
    assert "blocked" in format_report(payload)


def test_no_runs_means_nothing_to_report():
    store = Datastore(":memory:")
    payload = health_payload(store, registry(mine={"a": 1.0}), NOW)
    assert payload == {"issues": [], "persona_impact": {}}
    assert "healthy" in format_report(payload)


def test_parse_feed_tells_an_unreadable_feed_from_an_empty_one():
    import pytest

    from ingestion.extract import parse_feed

    empty = '<?xml version="1.0"?><rss version="2.0"><channel><title>t</title></channel></rss>'
    assert parse_feed(empty) == []
    with pytest.raises(ValueError):
        parse_feed("<html><body>403 Forbidden</body")
