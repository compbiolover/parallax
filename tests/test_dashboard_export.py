"""Dashboard exporter: payload shape and file writing."""

from __future__ import annotations

import json

from dashboard.export import build_payload, write_payload
from ingestion.datastore import Datastore
from scoring.foundations import CLASSIC_FOUNDATIONS


def _store_with_two_diets():
    store = Datastore(":memory:")
    for diet, care in [("self", 0.3), ("modeled_ce", 0.1)]:
        did = f"{diet}-doc"
        store.upsert_document(
            doc_id=did, diet_id=diet, source_id="s", stratum_id=None, url=None,
            title="t", published_utc=None, fetched_utc="2026-07-23T00:00:00+00:00",
            word_count=90, minhash=None,
        )
        store.upsert_scores(
            document_id=did, scorer="dictionary",
            foundations={"care": care, "fairness": 0.1, "loyalty": 0.2,
                         "authority": 0.1, "sanctity": 0.1},
            sentiment=0.0, moral_word_ratio=0.2, matched_words=18,
        )
    store.upsert_summary(scope="self", generated_utc="t", model="m",
                         method="deterministic", text="self summary")
    store.upsert_summary(scope="executive", generated_utc="t", model="m",
                         method="deterministic", text="exec summary")
    return store


def test_payload_shape():
    store = _store_with_two_diets()
    p = build_payload(store)
    assert p["foundations"] == list(CLASSIC_FOUNDATIONS)
    assert len(p["diets"]) == 2
    for d in p["diets"]:
        assert set(d["profile"]) == set(CLASSIC_FOUNDATIONS)
        assert abs(sum(d["profile"].values()) - 1.0) < 1e-6
    assert p["comparison"]["pair"] == ["modeled_ce", "self"]
    assert 0.0 <= p["comparison"]["jsd"] <= 1.0
    assert p["executive_summary"] == "exec summary"
    # No lexicon recorded -> treated as demo, strong caveat.
    assert p["lexicon"] is None
    assert "DEMO lexicon" in p["caveat"]
    store.close()


def test_caveat_softens_for_real_lexicon():
    store = _store_with_two_diets()
    store.set_meta("lexicon", "eMFD (emfd_scoring.csv)")
    p = build_payload(store)
    assert p["lexicon"] == "eMFD (emfd_scoring.csv)"
    assert "DEMO" not in p["caveat"]
    assert "eMFD (emfd_scoring.csv)" in p["caveat"]
    store.close()


def test_write_js_payload(tmp_path):
    store = _store_with_two_diets()
    out = write_payload(store, tmp_path / "latest.js")
    text = out.read_text()
    assert text.startswith("window.PARALLAX_DATA = ")
    data = json.loads(text[len("window.PARALLAX_DATA = "):].rstrip().rstrip(";"))
    assert data["diets"]
    store.close()


def test_blindspots_in_payload():
    store = _store_with_two_diets()
    # add two more modeled_ce docs so a cluster can be one-sided
    for i in range(3):
        did = f"ce-extra-{i}"
        store.upsert_document(doc_id=did, diet_id="modeled_ce", source_id="s", stratum_id=None,
            url=None, title=f"faith story {i}", published_utc=None,
            fetched_utc="2026-07-23T00:00:00+00:00", word_count=80, minhash=None)
        store.upsert_scores(document_id=did, scorer="dictionary", foundations={"sanctity": 0.5},
                            sentiment=0.0, moral_word_ratio=0.1, matched_words=5)
    # seed a persisted clustering directly (cluster 5 = modeled_ce-only)
    store.replace_clustering(
        clusters=[(5, "faith · story", 3)],
        assignments=[(f"ce-extra-{i}", 5) for i in range(3)],
    )
    p = build_payload(store)
    assert p["blindspots"], "expected a blindspot"
    b = p["blindspots"][0]
    assert b["dominant_diet"] == "modeled_ce"
    assert b["other_diet"] == "self"
    assert b["label"] == "faith · story"
    assert b["representative_titles"]
    store.close()


def test_no_blindspots_when_unclustered():
    store = _store_with_two_diets()
    assert build_payload(store)["blindspots"] == []
    store.close()


def test_single_diet_has_no_comparison():
    store = Datastore(":memory:")
    store.upsert_document(
        doc_id="d", diet_id="self", source_id="s", stratum_id=None, url=None,
        title="t", published_utc=None, fetched_utc="2026-07-23T00:00:00+00:00",
        word_count=90, minhash=None,
    )
    store.upsert_scores(document_id="d", scorer="dictionary",
                        foundations={"care": 0.5}, sentiment=0.0,
                        moral_word_ratio=0.1, matched_words=5)
    assert build_payload(store)["comparison"] is None
    store.close()


# -- snapshot history in the payload ---------------------------------------

def test_history_is_empty_until_a_snapshot_is_recorded():
    store = _store_with_two_diets()
    p = build_payload(store)
    assert p["history"] == []
    assert p["history_window_days"] is None
    store.close()


def test_exporting_never_records_a_snapshot(tmp_path):
    """Export is a read. Recording is the daily runner's `snapshot` step, so
    rebuilding the payload can't invent history."""
    store = _store_with_two_diets()
    build_payload(store)
    write_payload(store, tmp_path / "latest.js")
    assert store.snapshot_count() == 0
    store.close()


def test_history_carries_dated_points_and_the_window():
    from compare.history import record_snapshot

    store = _store_with_two_diets()
    for day in ("2026-07-23", "2026-07-24"):
        record_snapshot(store, day, window_days=5)
    p = build_payload(store)
    assert [s["date"] for s in p["history"]] == ["2026-07-23", "2026-07-24"]
    assert p["history_window_days"] == 5
    # The series' all-time basis is the headline number, up to the 6-decimal
    # rounding snapshots are stored at.
    assert p["history"][-1]["jsd_cumulative"] == round(p["comparison"]["jsd"], 6)
    store.close()


def test_history_limit_caps_what_is_serialized():
    from compare.history import record_snapshot

    store = _store_with_two_diets()
    for day in ("2026-07-21", "2026-07-22", "2026-07-23"):
        record_snapshot(store, day)
    assert len(build_payload(store, history_limit=2)["history"]) == 2
    store.close()


# -- fairness split in the payload -----------------------------------------

def test_fairness_split_absent_when_nothing_was_partitioned():
    store = _store_with_two_diets()
    assert build_payload(store)["fairness_split"] is None
    store.close()


def test_fairness_split_carries_shares_and_coverage():
    store = _store_with_two_diets()
    store.upsert_scores(
        document_id="self-doc", scorer="dictionary",
        foundations={"care": 0.3, "fairness": 0.2, "loyalty": 0.2,
                     "authority": 0.1, "sanctity": 0.1},
        sentiment=0.0, moral_word_ratio=0.2, matched_words=18,
        equality=0.15, proportionality=0.05,
    )
    fs = build_payload(store)["fairness_split"]
    assert fs["diets"]["self"]["leans"] == "equality"
    assert fs["diets"]["self"]["docs_split"] == 1
    assert fs["diets"]["self"]["coverage"] == 1.0
    # The other diet was never split — reported, but with nothing behind it.
    assert fs["diets"]["modeled_ce"]["docs_split"] == 0
    assert fs["diets"]["modeled_ce"]["thin"]
    store.close()
