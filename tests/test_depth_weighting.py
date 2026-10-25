"""Headline-only documents count, but not as a full article.

Two mechanisms, deliberately separate. `cluster.min_words` keeps stubs out of
the *clustering*, because a headline is not enough to say what a story is.
`depth_weight` keeps them in the *profile* at a quarter, because a headline is
still real evidence that an outlet ran something.

The hazard both address is the same one: extraction depth is not random.
Paywalled outlets and the ones trafilatura cannot parse yield stubs, so equal
treatment lets "which outlets the scraper can read" act on the comparison as if
it were a finding.
"""

from __future__ import annotations

from cluster.cluster import compute_clustering
from ingestion.datastore import Datastore
from ingestion.pipeline import STUB_WEIGHT, STUB_WORDS, depth_weight, persona_profiles

from .registries import registry

FOUNDATIONS = ("care", "fairness", "loyalty", "authority", "sanctity")


def _doc(store, doc_id, source_id, words, foundations, *, vector=None, title="t"):
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


# -- the weight itself ------------------------------------------------------


def test_a_stub_counts_at_a_quarter():
    assert depth_weight(10) == STUB_WEIGHT
    assert depth_weight(STUB_WORDS - 1) == STUB_WEIGHT


def test_a_full_article_counts_once():
    assert depth_weight(STUB_WORDS) == 1.0
    assert depth_weight(1100) == 1.0


def test_a_stub_is_not_dropped():
    """Zero would bias the corpus toward whoever publishes in a parsable format."""
    assert depth_weight(10) > 0


def test_a_missing_word_count_is_treated_as_a_stub():
    assert depth_weight(None) == STUB_WEIGHT


# -- what it does to a profile ---------------------------------------------


def _profile_store(tmp_path):
    """One source, one full article on care, three stubs on sanctity."""
    store = Datastore(str(tmp_path / "p.sqlite"))
    care = {f: (0.9 if f == "care" else 0.025) for f in FOUNDATIONS}
    sanctity = {f: (0.9 if f == "sanctity" else 0.025) for f in FOUNDATIONS}
    _doc(store, "full-1", "src_self", 800, care)
    for i in range(3):
        _doc(store, f"stub-{i}", "src_self", 12, sanctity)
    return store


def test_three_stubs_no_longer_outvote_one_article(tmp_path):
    """Counted equally, 3 sanctity headlines beat 1 care feature. They are
    three-quarters of a document between them instead."""
    profile = persona_profiles(_profile_store(tmp_path), registry(self={"src_self": 1.0}))["self"]
    assert profile["care"] > profile["sanctity"]


def test_stubs_still_move_the_profile(tmp_path):
    """A quarter, not zero: the headlines are still in there."""
    store = _profile_store(tmp_path)
    with_stubs = persona_profiles(store, registry(self={"src_self": 1.0}))["self"]

    only_full = Datastore(str(tmp_path / "only.sqlite"))
    _doc(
        only_full,
        "full-1",
        "src_self",
        800,
        {f: (0.9 if f == "care" else 0.025) for f in FOUNDATIONS},
    )
    without = persona_profiles(only_full, registry(self={"src_self": 1.0}))["self"]

    assert with_stubs["sanctity"] > without["sanctity"]


def test_a_corpus_of_one_depth_is_unaffected(tmp_path):
    """The weight is relative and the profile is normalized, so an all-stub
    corpus reads exactly as it did before this existed."""
    store = Datastore(str(tmp_path / "s.sqlite"))
    care = {f: (0.9 if f == "care" else 0.025) for f in FOUNDATIONS}
    for i in range(4):
        _doc(store, f"stub-{i}", "src_self", 12, care)
    profile = persona_profiles(store, registry(self={"src_self": 1.0}))["self"]
    assert profile["care"] > 0.8


# -- the clustering floor ---------------------------------------------------


def test_clustering_skips_stubs_when_asked(tmp_path):
    store = Datastore(str(tmp_path / "c.sqlite"))
    store.set_meta("embedder", "test-embedder")
    neutral = dict.fromkeys(FOUNDATIONS, 0.2)
    _doc(store, "full-1", "src_self", 800, neutral, vector=[1.0, 0.0])
    _doc(store, "full-2", "src_self", 900, neutral, vector=[0.9, 0.1])
    _doc(store, "stub-1", "src_self", 12, neutral, vector=[0.0, 1.0])

    everything = compute_clustering(store, min_cluster_size=2)
    assert set(everything.doc_ids) == {"full-1", "full-2", "stub-1"}

    gated = compute_clustering(store, min_cluster_size=2, min_words=120)
    assert set(gated.doc_ids) == {"full-1", "full-2"}, "the headline is not a story"


def test_the_floor_is_off_by_default(tmp_path):
    """A hand-built corpus clusters whatever it was given; the policy that a
    real corpus needs lives in `cluster.min_words`, not in this primitive."""
    store = Datastore(str(tmp_path / "d.sqlite"))
    store.set_meta("embedder", "test-embedder")
    neutral = dict.fromkeys(FOUNDATIONS, 0.2)
    _doc(store, "stub-1", "src_self", 5, neutral, vector=[1.0, 0.0])
    _doc(store, "stub-2", "src_self", 6, neutral, vector=[0.9, 0.1])
    assert len(compute_clustering(store, min_cluster_size=2).doc_ids) == 2
