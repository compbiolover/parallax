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
    store.upsert_summary(scope="self", generated_utc="t", model="m", method="deterministic", text="self summary")
    store.upsert_summary(scope="executive", generated_utc="t", model="m", method="deterministic", text="exec summary")
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
    assert "demo lexicon" in p["caveat"]
    store.close()


def test_write_js_payload(tmp_path):
    store = _store_with_two_diets()
    out = write_payload(store, tmp_path / "latest.js")
    text = out.read_text()
    assert text.startswith("window.PARALLAX_DATA = ")
    data = json.loads(text[len("window.PARALLAX_DATA = "):].rstrip().rstrip(";"))
    assert data["diets"]
    store.close()


def test_single_diet_has_no_comparison():
    store = Datastore(":memory:")
    store.upsert_document(
        doc_id="d", diet_id="self", source_id="s", stratum_id=None, url=None,
        title="t", published_utc=None, fetched_utc="2026-07-23T00:00:00+00:00",
        word_count=90, minhash=None,
    )
    store.upsert_scores(document_id="d", scorer="dictionary",
                        foundations={"care": 0.5}, sentiment=0.0, moral_word_ratio=0.1, matched_words=5)
    assert build_payload(store)["comparison"] is None
    store.close()
