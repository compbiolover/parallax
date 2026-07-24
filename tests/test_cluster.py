"""Phase 2: embedder, clustering, blindspot detection, and export shape."""

from __future__ import annotations

import numpy as np

from cluster.blindspot import (
    blindspots_from_store,
    detect_blindspots,
    label_cluster,
    label_clusters,
    run_clustering,
)
from cluster.cluster import ClusterResult
from cluster.embed import HashingEmbedder, build_embedder
from ingestion.datastore import Datastore

TOPIC = {
    "election": "election ballot vote campaign candidate polls senate race congress",
    "faith": "abortion faith church sanctity life prayer scripture pastor congregation",
    "climate": "climate emissions renewable solar carbon warming policy energy transition",
}


# -- embedder --------------------------------------------------------------

def test_hashing_embedder_deterministic_and_normalized():
    e = HashingEmbedder(dim=128)
    v1, v2 = e.embed("hello world news"), e.embed("hello world news")
    assert v1 == v2
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5
    assert len(v1) == 128


def test_hashing_embedder_topic_similarity():
    e = HashingEmbedder(dim=512)
    a = np.array(e.embed(TOPIC["climate"] + " one"))
    b = np.array(e.embed(TOPIC["climate"] + " two"))
    c = np.array(e.embed(TOPIC["election"]))
    assert float(a @ b) > float(a @ c)  # same topic closer than different


def test_build_embedder_default_is_hashing():
    emb, name = build_embedder({})
    assert name.startswith("hashing(")


def test_empty_text_embeds_to_zero_vector():
    assert set(HashingEmbedder(dim=16).embed("")) == {0.0}


# -- labels & blindspot logic ---------------------------------------------

def test_label_cluster_skips_stopwords():
    label = label_cluster(["The new climate policy debate", "A climate summit and policy"])
    assert "climate" in label
    assert "the" not in label.split(" · ")


def test_label_clusters_ctfidf_prefers_distinctive_terms():
    # "nuclear/saudi" and "climate/emissions" are distinctive; "trump" is shared
    # across both clusters so c-TF-IDF should down-weight it.
    labels = label_clusters({
        0: ["Trump signs nuclear deal with Saudi Arabia", "Saudi nuclear enrichment concerns"],
        1: ["Trump climate policy on emissions", "New climate emissions targets"],
    })
    assert "nuclear" in labels[0] or "saudi" in labels[0]
    assert "climate" in labels[1] or "emissions" in labels[1]
    assert "trump" not in labels[0]  # shared term down-weighted / dropped


def test_run_clustering_uses_ctfidf_labels():
    store = Datastore(":memory:")
    _seed_topics(store, HashingEmbedder(dim=256))
    run_clustering(store)
    labels = [r["label"] for r in store.cluster_rows()]
    # a cluster label should reflect one of the seeded topics, not generic filler
    joined = " ".join(labels).lower()
    assert any(k in joined for k in ("faith", "church", "climate", "vote", "election",
                                     "prayer", "carbon", "renewable", "campaign", "scripture"))
    store.close()


def test_detect_blindspots_direction_and_symmetry():
    # cluster 0: both diets (not a blindspot); 1: modeled_ce only; 2: self only
    labels = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    diets = ["self", "self", "modeled_ce", "modeled_ce",
             "modeled_ce", "modeled_ce", "modeled_ce",
             "self", "self", "self"]
    titles = ["t"] * 10
    result = ClusterResult([f"d{i}" for i in range(10)], diets, titles, labels)
    bs = detect_blindspots(result, dominance=0.8, min_size=3)
    by_diet = {b.dominant_diet: b for b in bs}
    assert set(by_diet) == {"modeled_ce", "self"}   # both directions surfaced
    assert all(b.cluster_id != 0 for b in bs)       # shared cluster excluded


def _seed_topics(store, emb):
    i = 0
    def add(diet, title, text):
        nonlocal i
        did = f"d{i}"; i += 1
        store.upsert_document(doc_id=did, diet_id=diet, source_id="s", stratum_id=None,
            url=None, title=title, published_utc=None,
            fetched_utc="2026-07-23T00:00:00+00:00", word_count=50, minhash=None)
        store.upsert_embedding(document_id=did, vector=emb.embed(text), embedder=emb.name)
    for n in range(4):
        add("self", f"Vote {n}", TOPIC["election"] + f" {n}")
        add("modeled_ce", f"Vote {n}", TOPIC["election"] + f" {n}")
    for n in range(4):
        add("modeled_ce", f"Faith {n}", TOPIC["faith"] + f" {n}")
    for n in range(4):
        add("self", f"Climate {n}", TOPIC["climate"] + f" {n}")


def test_run_clustering_end_to_end_separates_topics():
    store = Datastore(":memory:")
    _seed_topics(store, HashingEmbedder(dim=256))
    outcome = run_clustering(store, min_cluster_size=3, dominance=0.8, min_blindspot_size=3)
    assert outcome.n_clusters >= 2
    dirs = {b.dominant_diet for b in outcome.blindspots}
    assert "modeled_ce" in dirs and "self" in dirs
    # persisted assignment can be re-read without sklearn
    assert len(blindspots_from_store(store)) == len(outcome.blindspots)
    store.close()


def test_iter_embeddings_filters_by_embedder():
    store = Datastore(":memory:")
    for i, emb in [(0, "hashing(d=8)"), (1, "hashing(d=8)"), (2, "sentence-transformers/x")]:
        store.upsert_document(doc_id=f"d{i}", diet_id="self", source_id="s", stratum_id=None,
            url=None, title="t", published_utc=None,
            fetched_utc="2026-07-23T00:00:00+00:00", word_count=10, minhash=None)
        store.upsert_embedding(document_id=f"d{i}", vector=[0.0] * 8, embedder=emb)
    assert len(list(store.iter_embeddings())) == 3
    assert len(list(store.iter_embeddings(embedder="hashing(d=8)"))) == 2
    assert set(store.embedder_names()) == {"hashing(d=8)", "sentence-transformers/x"}
    store.close()


def test_clustering_clusters_only_active_embedder():
    # Two embedders with different dims in one DB; meta marks one active.
    store = Datastore(":memory:")
    emb = HashingEmbedder(dim=64)
    _seed_topics(store, emb)  # writes embedder "hashing(d=64)"
    # inject a stray doc from a different embedder/dim
    store.upsert_document(doc_id="stray", diet_id="self", source_id="s", stratum_id=None,
        url=None, title="stray", published_utc=None,
        fetched_utc="2026-07-23T00:00:00+00:00", word_count=10, minhash=None)
    store.upsert_embedding(document_id="stray", vector=[0.0] * 384, embedder="other(d=384)")
    store.set_meta("embedder", emb.name)
    outcome = run_clustering(store)  # must not crash; ignores the stray dim-384 doc
    assert outcome.n_docs == 16  # only the 16 seeded same-embedder docs
    store.close()


def test_clustering_raises_on_mixed_dims_without_active_embedder():
    from cluster.cluster import compute_clustering
    store = Datastore(":memory:")
    for i, (dim, emb) in enumerate([(8, "a"), (16, "b")]):
        store.upsert_document(doc_id=f"d{i}", diet_id="self", source_id="s", stratum_id=None,
            url=None, title="t", published_utc=None,
            fetched_utc="2026-07-23T00:00:00+00:00", word_count=10, minhash=None)
        store.upsert_embedding(document_id=f"d{i}", vector=[0.0] * dim, embedder=emb)
    # no meta['embedder'] -> no filter -> mixed dims -> clear error, not a numpy crash
    import pytest
    with pytest.raises(ValueError, match="mixed dimensions"):
        compute_clustering(store, min_cluster_size=2)
    store.close()


def test_too_few_docs_is_all_noise():
    store = Datastore(":memory:")
    emb = HashingEmbedder(dim=32)
    store.upsert_document(doc_id="d0", diet_id="self", source_id="s", stratum_id=None,
        url=None, title="t", published_utc=None,
        fetched_utc="2026-07-23T00:00:00+00:00", word_count=10, minhash=None)
    store.upsert_embedding(document_id="d0", vector=emb.embed("hello"), embedder=emb.name)
    outcome = run_clustering(store)
    assert outcome.n_clusters == 0 and outcome.blindspots == []
    store.close()
