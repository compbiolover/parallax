"""Headlines and articles are not the same evidence.

Two mechanisms over one predicate. `cluster.cluster_stubs` keeps headlines out
of the *clustering*, because a title cannot say what a story is.
`depth_weight` keeps them in the *profile* at a quarter, because a headline is
still real evidence that an outlet ran something.

The predicate is `ingest_kind`, recorded at ingestion, not a word count. A
120-word floor was tried first and demoted 1,297 documents that had real text
— 421 from PBS NewsHour, 338 from The Hill, both outlets whose article
extraction fails and whose RSS summary is what gets stored. Extraction depth
is not random, so a length test measures which outlets the scraper can read,
which is the bias this weighting exists to remove.
"""

from __future__ import annotations

import yaml

from cluster import DEFAULT_CLUSTER_STUBS, cluster_stubs
from cluster.cluster import compute_clustering
from ingestion.datastore import Datastore
from ingestion.pipeline import persona_profiles
from scoring.depth import (
    BODY,
    LEGACY_STUB_WORDS,
    STUB_WEIGHT,
    SUMMARY,
    TITLE_ONLY,
    depth_weight,
    is_stub,
    row_weights,
)

from .registries import registry

FOUNDATIONS = ("care", "fairness", "loyalty", "authority", "sanctity")


def _doc(store, doc_id, source_id, words, foundations, *, kind=BODY, vector=None, title="t"):
    store.upsert_document(
        doc_id=doc_id,
        source_id=source_id,
        stratum_id=None,
        url=None,
        title=title,
        published_utc=None,
        fetched_utc="2026-10-25T00:00:00+00:00",
        word_count=words,
        minhash=None,
        ingest_kind=kind,
    )
    store.upsert_scores(
        document_id=doc_id,
        scorer="dictionary",
        foundations=foundations,
        sentiment=0.0,
        moral_word_ratio=0.1,
        matched_words=5,
    )
    if vector is not None:
        store.upsert_embedding(document_id=doc_id, vector=vector, embedder="test-embedder")


# -- the predicate ----------------------------------------------------------


def test_a_headline_is_a_stub():
    assert is_stub(TITLE_ONLY) is True


def test_an_article_is_not():
    assert is_stub(BODY) is False


def test_a_feed_summary_is_not_a_stub():
    """The regression that matters: 1,297 documents with real text were being
    demoted because they were short. A summary is the publisher's own prose."""
    assert is_stub(SUMMARY) is False
    assert depth_weight(SUMMARY, 90) == 1.0
    assert depth_weight(SUMMARY, 31) == 1.0


def test_length_does_not_override_provenance():
    """A long headline is still a headline; a short article is still an article."""
    assert is_stub(TITLE_ONLY, 5000) is True
    assert is_stub(BODY, 1) is False


def test_rows_written_before_the_column_fall_back_to_length():
    assert is_stub(None, LEGACY_STUB_WORDS - 1) is True
    assert is_stub(None, LEGACY_STUB_WORDS) is False
    assert is_stub(None, None) is True


# -- the weight -------------------------------------------------------------


def test_a_headline_counts_for_a_quarter():
    """Pinned to the number, not to the constant: comparing depth_weight to
    STUB_WEIGHT holds for any value and catches no drift."""
    assert depth_weight(TITLE_ONLY) == 0.25
    assert STUB_WEIGHT == 0.25


def test_an_article_counts_for_one():
    assert depth_weight(BODY) == 1.0


def test_a_headline_is_not_dropped():
    """Zero would bias the corpus toward whoever publishes in a parsable format."""
    assert depth_weight(TITLE_ONLY) > 0


# -- what it does to a profile ---------------------------------------------


def _profile_store(tmp_path):
    """One article on care, three headlines on sanctity."""
    store = Datastore(str(tmp_path / "p.sqlite"))
    care = {f: (0.9 if f == "care" else 0.025) for f in FOUNDATIONS}
    sanctity = {f: (0.9 if f == "sanctity" else 0.025) for f in FOUNDATIONS}
    _doc(store, "full-1", "src_self", 800, care, kind=BODY)
    for i in range(3):
        _doc(store, f"stub-{i}", "src_self", 12, sanctity, kind=TITLE_ONLY)
    return store


def test_three_headlines_no_longer_outvote_one_article(tmp_path):
    profile = persona_profiles(_profile_store(tmp_path), registry(self={"src_self": 1.0}))["self"]
    assert profile["care"] > profile["sanctity"]


def test_the_margin_holds_well_past_three(tmp_path):
    """The k=3 case passes at any weight below 1/3, so it pins almost nothing.
    Five headlines against one article is where 0.25 has to actually hold."""
    store = Datastore(str(tmp_path / "k5.sqlite"))
    care = {f: (0.9 if f == "care" else 0.025) for f in FOUNDATIONS}
    sanctity = {f: (0.9 if f == "sanctity" else 0.025) for f in FOUNDATIONS}
    _doc(store, "full-1", "src_self", 800, care, kind=BODY)
    for i in range(5):
        _doc(store, f"stub-{i}", "src_self", 12, sanctity, kind=TITLE_ONLY)
    profile = persona_profiles(store, registry(self={"src_self": 1.0}))["self"]
    assert profile["sanctity"] > profile["care"], "five headlines should still win"
    # ...but by far less than their count suggests: 5 * 0.25 = 1.25, not 5.
    assert profile["sanctity"] / profile["care"] < 2.0


def test_headlines_still_move_the_profile(tmp_path):
    """A quarter, not zero: they are still in there."""
    store = _profile_store(tmp_path)
    with_stubs = persona_profiles(store, registry(self={"src_self": 1.0}))["self"]

    only_full = Datastore(str(tmp_path / "only.sqlite"))
    care = {f: (0.9 if f == "care" else 0.025) for f in FOUNDATIONS}
    _doc(only_full, "full-1", "src_self", 800, care, kind=BODY)
    without = persona_profiles(only_full, registry(self={"src_self": 1.0}))["self"]

    assert with_stubs["sanctity"] > without["sanctity"]


# -- one rule, every aggregation path ---------------------------------------


def test_row_weights_is_what_every_aggregation_path_uses(tmp_path):
    """Four places build a weighted composition from these rows. When the depth
    factor lived in only one, `make sensitivity` stress-tested a baseline the
    dashboard never showed and flipped the sign on an authority claim."""
    store = _profile_store(tmp_path)
    weights = {"src_self": 1.0}
    rows = store.scores_for_sources(weights, "dictionary")
    applied = row_weights(rows, weights)
    by_id = {row["document_id"]: w for row, w in zip(rows, applied, strict=True)}
    assert by_id["full-1"] == 1.0
    assert all(by_id[f"stub-{i}"] == STUB_WEIGHT for i in range(3))


def test_scores_for_sources_carries_what_the_weighting_needs(tmp_path):
    """row_weights reads ingest_kind off the row; if the query stops selecting
    it every caller silently reverts to counting headlines fully."""
    store = _profile_store(tmp_path)
    rows = store.scores_for_sources({"src_self": 1.0}, "dictionary")
    assert rows and rows[0]["ingest_kind"] in (BODY, TITLE_ONLY)


# -- the clustering side ----------------------------------------------------


def _cluster_store(tmp_path):
    store = Datastore(str(tmp_path / "c.sqlite"))
    store.set_meta("embedder", "test-embedder")
    neutral = dict.fromkeys(FOUNDATIONS, 0.2)
    _doc(store, "full-1", "src_self", 800, neutral, kind=BODY, vector=[1.0, 0.0])
    _doc(store, "full-2", "src_self", 900, neutral, kind=BODY, vector=[0.9, 0.1])
    _doc(store, "head-1", "src_self", 12, neutral, kind=TITLE_ONLY, vector=[0.0, 1.0])
    _doc(store, "summ-1", "src_self", 90, neutral, kind=SUMMARY, vector=[0.1, 0.9])
    return store


def test_clustering_can_exclude_headlines(tmp_path):
    store = _cluster_store(tmp_path)
    assert set(compute_clustering(store, min_cluster_size=2).doc_ids) == {
        "full-1",
        "full-2",
        "head-1",
        "summ-1",
    }
    gated = compute_clustering(store, min_cluster_size=2, exclude_stubs=True)
    assert set(gated.doc_ids) == {"full-1", "full-2", "summ-1"}


def test_a_short_summary_still_forms_stories(tmp_path):
    """The 1,297 again, on the clustering side: they are real stories."""
    gated = compute_clustering(_cluster_store(tmp_path), min_cluster_size=2, exclude_stubs=True)
    assert "summ-1" in gated.doc_ids


def test_legacy_rows_are_excluded_by_length(tmp_path):
    """A store that predates ingest_kind still gets the intended behaviour."""
    store = Datastore(str(tmp_path / "legacy.sqlite"))
    store.set_meta("embedder", "test-embedder")
    neutral = dict.fromkeys(FOUNDATIONS, 0.2)
    _doc(store, "a", "src_self", 800, neutral, kind=BODY, vector=[1.0, 0.0])
    _doc(store, "b", "src_self", 12, neutral, kind=BODY, vector=[0.0, 1.0])
    store.conn.execute("UPDATE documents SET ingest_kind = NULL")
    store.conn.commit()
    gated = compute_clustering(store, min_cluster_size=2, exclude_stubs=True)
    assert set(gated.doc_ids) == {"a"}


def test_the_floor_is_off_by_default(tmp_path):
    assert len(compute_clustering(_cluster_store(tmp_path), min_cluster_size=2).doc_ids) == 4


# -- the setting is actually read -------------------------------------------


def test_the_shipped_example_config_parses_to_the_default():
    """The predecessor of this key shipped nested one level too deep, so it
    read as None and the documented escape hatch did nothing. Only the
    coincidence that the stray value equalled the default hid it."""
    with open("config/settings.example.yaml", encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    assert (settings.get("cluster") or {}).get("cluster_stubs") is not None, (
        "cluster_stubs must be a direct child of `cluster:`"
    )
    assert cluster_stubs(settings) is False


def test_a_non_default_value_survives_the_read():
    assert cluster_stubs({"cluster": {"cluster_stubs": True}}) is True
    assert cluster_stubs({"cluster": {}}) is DEFAULT_CLUSTER_STUBS
    assert cluster_stubs({}) is DEFAULT_CLUSTER_STUBS


def test_the_setting_reaches_run_clustering():
    """Hard-coding the argument inside run_clustering used to leave all 740
    tests green: the whole settings -> DailyConfig -> run_clustering chain was
    unpinned."""
    from daily.runner import DailyConfig

    assert DailyConfig.from_settings({"cluster": {"cluster_stubs": True}}).cluster_stubs is True
    assert DailyConfig.from_settings({"cluster": {"cluster_stubs": False}}).cluster_stubs is False
    assert DailyConfig.from_settings({}).cluster_stubs is DEFAULT_CLUSTER_STUBS


def test_a_headline_canonical_is_kept_when_its_copies_have_text(tmp_path):
    """Dedup picks the canonical by arrival order, never by length, so a GDELT
    headline that landed first can front a syndicated story with full copies
    behind it. Dropping it would take the whole story out of the clustering."""
    store = Datastore(str(tmp_path / "wire.sqlite"))
    store.set_meta("embedder", "test-embedder")
    neutral = dict.fromkeys(FOUNDATIONS, 0.2)
    _doc(store, "canon", "src_a", 12, neutral, kind=TITLE_ONLY, vector=[1.0, 0.0])
    _doc(store, "other", "src_b", 800, neutral, kind=BODY, vector=[0.9, 0.1])
    # A full-length copy collapsed into the headline canonical.
    store.upsert_document(
        doc_id="copy",
        source_id="src_c",
        stratum_id=None,
        url=None,
        title="t",
        published_utc=None,
        fetched_utc="2026-10-25T00:00:00+00:00",
        word_count=900,
        minhash=None,
        is_duplicate=True,
        duplicate_of="canon",
        ingest_kind=BODY,
    )
    gated = compute_clustering(store, min_cluster_size=2, exclude_stubs=True)
    assert "canon" in gated.doc_ids, "the story has text, it is just not on the canonical"


def test_a_headline_with_only_headline_copies_is_still_excluded(tmp_path):
    store = Datastore(str(tmp_path / "nowire.sqlite"))
    store.set_meta("embedder", "test-embedder")
    neutral = dict.fromkeys(FOUNDATIONS, 0.2)
    _doc(store, "canon", "src_a", 12, neutral, kind=TITLE_ONLY, vector=[1.0, 0.0])
    _doc(store, "other", "src_b", 800, neutral, kind=BODY, vector=[0.9, 0.1])
    store.upsert_document(
        doc_id="copy",
        source_id="src_c",
        stratum_id=None,
        url=None,
        title="t",
        published_utc=None,
        fetched_utc="2026-10-25T00:00:00+00:00",
        word_count=11,
        minhash=None,
        is_duplicate=True,
        duplicate_of="canon",
        ingest_kind=TITLE_ONLY,
    )
    gated = compute_clustering(store, min_cluster_size=2, exclude_stubs=True)
    assert "canon" not in gated.doc_ids
