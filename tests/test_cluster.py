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
    labels = label_clusters(
        {
            0: ["Trump signs nuclear deal with Saudi Arabia", "Saudi nuclear enrichment concerns"],
            1: ["Trump climate policy on emissions", "New climate emissions targets"],
        }
    )
    assert "nuclear" in labels[0] or "saudi" in labels[0]
    assert "climate" in labels[1] or "emissions" in labels[1]
    assert "trump" not in labels[0]  # shared term down-weighted / dropped


def test_run_clustering_uses_ctfidf_labels():
    store = Datastore(":memory:")
    _seed_topics(store, HashingEmbedder(dim=256))
    run_clustering(store, MEMBERS)
    labels = [r["label"] for r in store.cluster_rows()]
    # a cluster label should reflect one of the seeded topics, not generic filler
    joined = " ".join(labels).lower()
    assert any(
        k in joined
        for k in (
            "faith",
            "church",
            "climate",
            "vote",
            "election",
            "prayer",
            "carbon",
            "renewable",
            "campaign",
            "scripture",
        )
    )
    store.close()


MEMBERS = {"self": {"src_self"}, "modeled_ce": {"src_modeled_ce"}}


def _result(doc_ids, sources, titles, labels, coverage=None):
    """A ClusterResult whose coverage defaults to each document's own source.

    Coverage is a separate list because a collapsed near-duplicate's outlet also
    carried the story; pass it explicitly to model that.
    """
    cov = coverage if coverage is not None else [frozenset({s}) for s in sources]
    return ClusterResult(doc_ids, sources, cov, titles, labels)


def test_detect_blindspots_direction_and_symmetry():
    # cluster 0: both personas (not a blindspot); 1: modeled_ce only; 2: self only
    labels = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    sources = [
        "src_self",
        "src_self",
        "src_modeled_ce",
        "src_modeled_ce",
        "src_modeled_ce",
        "src_modeled_ce",
        "src_modeled_ce",
        "src_self",
        "src_self",
        "src_self",
    ]
    titles = ["t"] * 10
    result = _result([f"d{i}" for i in range(10)], sources, titles, labels)
    bs = detect_blindspots(result, MEMBERS, dominance=0.8, min_size=3)
    by_diet = {b.dominant_diet: b for b in bs}
    assert set(by_diet) == {"modeled_ce", "self"}  # both directions surfaced
    assert all(b.cluster_id != 0 for b in bs)  # shared cluster excluded


def test_a_third_persona_does_not_dilute_a_blindspot_out_of_existence():
    """The dominance share used to be measured against every member of the
    cluster, so a second persona on the same side split the coverage and pushed
    the leader under the threshold — dropping a real cross-family blindspot for an
    arithmetic reason."""
    # One cluster: two docs from the pair's `modeled_ce`, three from a third
    # persona nobody is comparing against, none from `self`.
    labels = [0, 0, 0, 0, 0]
    sources = ["src_modeled_ce", "src_modeled_ce", "src_third", "src_third", "src_third"]
    result = _result([f"d{i}" for i in range(5)], sources, ["t"] * 5, labels)
    bs = detect_blindspots(result, MEMBERS, dominance=0.8, min_size=2)
    assert [b.dominant_diet for b in bs] == ["modeled_ce"]
    assert bs[0].dominant_share == 1.0  # of the pair's members, not the cluster's
    assert bs[0].size == 2  # the pair accounts for two
    assert bs[0].cluster_size == 5  # the cluster itself is larger


def test_a_source_both_personas_read_makes_a_cluster_shared():
    """Two personas over one shared catalog can read the same outlet. A story it
    carried reached both of them, so it is not a blindspot for either."""
    shared = {"self": {"src_shared"}, "modeled_ce": {"src_shared"}}
    result = _result([f"d{i}" for i in range(4)], ["src_shared"] * 4, ["t"] * 4, [0, 0, 0, 0])
    assert detect_blindspots(result, shared, dominance=0.8, min_size=2) == []


def _seed_topics(store, emb):
    i = 0

    def add(diet, title, text):
        nonlocal i
        did = f"d{i}"
        i += 1
        store.upsert_document(
            doc_id=did,
            source_id=f"src_{diet}",
            stratum_id=None,
            url=None,
            title=title,
            published_utc=None,
            fetched_utc="2026-07-23T00:00:00+00:00",
            word_count=50,
            minhash=None,
        )
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
    outcome = run_clustering(
        store, MEMBERS, min_cluster_size=3, dominance=0.8, min_blindspot_size=3
    )
    assert outcome.n_clusters >= 2
    dirs = {b.dominant_diet for b in outcome.blindspots}
    assert "modeled_ce" in dirs and "self" in dirs
    # persisted assignment can be re-read without sklearn
    assert len(blindspots_from_store(store, MEMBERS)) == len(outcome.blindspots)
    store.close()


def test_iter_embeddings_filters_by_embedder():
    store = Datastore(":memory:")
    for i, emb in [(0, "hashing(d=8)"), (1, "hashing(d=8)"), (2, "sentence-transformers/x")]:
        store.upsert_document(
            doc_id=f"d{i}",
            diet_id="self",
            source_id="s",
            stratum_id=None,
            url=None,
            title="t",
            published_utc=None,
            fetched_utc="2026-07-23T00:00:00+00:00",
            word_count=10,
            minhash=None,
        )
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
    store.upsert_document(
        doc_id="stray",
        diet_id="self",
        source_id="s",
        stratum_id=None,
        url=None,
        title="stray",
        published_utc=None,
        fetched_utc="2026-07-23T00:00:00+00:00",
        word_count=10,
        minhash=None,
    )
    store.upsert_embedding(document_id="stray", vector=[0.0] * 384, embedder="other(d=384)")
    store.set_meta("embedder", emb.name)
    outcome = run_clustering(store, MEMBERS)  # must not crash; ignores the stray dim-384 doc
    assert outcome.n_docs == 16  # only the 16 seeded same-embedder docs
    store.close()


def test_clustering_raises_on_mixed_dims_without_active_embedder():
    from cluster.cluster import compute_clustering

    store = Datastore(":memory:")
    for i, (dim, emb) in enumerate([(8, "a"), (16, "b")]):
        store.upsert_document(
            doc_id=f"d{i}",
            diet_id="self",
            source_id="s",
            stratum_id=None,
            url=None,
            title="t",
            published_utc=None,
            fetched_utc="2026-07-23T00:00:00+00:00",
            word_count=10,
            minhash=None,
        )
        store.upsert_embedding(document_id=f"d{i}", vector=[0.0] * dim, embedder=emb)
    # no meta['embedder'] -> no filter -> mixed dims -> clear error, not a numpy crash
    import pytest

    with pytest.raises(ValueError, match="mixed dimensions"):
        compute_clustering(store, min_cluster_size=2)
    store.close()


def test_too_few_docs_is_all_noise():
    store = Datastore(":memory:")
    emb = HashingEmbedder(dim=32)
    store.upsert_document(
        doc_id="d0",
        diet_id="self",
        source_id="s",
        stratum_id=None,
        url=None,
        title="t",
        published_utc=None,
        fetched_utc="2026-07-23T00:00:00+00:00",
        word_count=10,
        minhash=None,
    )
    store.upsert_embedding(document_id="d0", vector=emb.embed("hello"), embedder=emb.name)
    outcome = run_clustering(store, MEMBERS)
    assert outcome.n_clusters == 0 and outcome.blindspots == []
    store.close()


# -- collapsed near-duplicates keep their outlet's coverage ------------------


def test_a_wire_story_both_sides_ran_stops_being_a_blindspot():
    """The bug, stated as an A/B. Near-duplicate detection is global, so when the
    same wire copy runs on both sides only the first-fetched copy survives — and
    only it is embedded and clustered. Read from the surviving document's own source
    the cluster looks 100% one-sided; read from every outlet that carried it, it is
    a story both sides ran."""
    doc_ids, sources, titles, labels = (
        ["d0", "d1", "d2"],
        ["src_self"] * 3,
        ["t"] * 3,
        [0, 0, 0],
    )

    # What the old code saw: coverage is the surviving document's own source.
    before = detect_blindspots(
        _result(doc_ids, sources, titles, labels), MEMBERS, dominance=0.8, min_size=2
    )
    assert [b.dominant_share for b in before] == [1.0]

    # What it is: d0's collapsed duplicate came from the other side's outlet.
    after = detect_blindspots(
        _result(
            doc_ids,
            sources,
            titles,
            labels,
            coverage=[
                frozenset({"src_self", "src_modeled_ce"}),
                frozenset({"src_self"}),
                frozenset({"src_self"}),
            ],
        ),
        MEMBERS,
        dominance=0.8,
        min_size=2,
    )
    assert after == [], "a story the other side also ran is not a blindspot"


def test_a_cluster_that_is_only_a_shared_wire_story_is_not_a_blindspot():
    result = _result(
        ["d0", "d1"],
        ["src_self", "src_self"],
        ["t", "t"],
        [0, 0],
        coverage=[frozenset({"src_self", "src_modeled_ce"})] * 2,
    )
    assert detect_blindspots(result, MEMBERS, dominance=0.8, min_size=2) == []


def test_the_store_reads_coverage_back_from_the_duplicate_rows():
    """Same correction on the persisted path, which is what the dashboard and the
    email read. `duplicate_of` recorded the collapsed copy's outlet all along; it
    was simply never looked at."""
    from cluster.blindspot import blindspots_from_store

    store = Datastore(":memory:")
    _add = lambda doc_id, source_id, dup_of=None: store.upsert_document(  # noqa: E731
        doc_id=doc_id,
        source_id=source_id,
        stratum_id=None,
        url=None,
        title="t",
        published_utc=None,
        fetched_utc="2026-08-10T00:00:00+00:00",
        word_count=200,
        minhash=None,
        is_duplicate=dup_of is not None,
        duplicate_of=dup_of,
    )
    _add("canon", "src_self")
    _add("dup", "src_modeled_ce", dup_of="canon")  # same wire story, other side
    _add("own", "src_self")
    store.replace_clustering(
        clusters=[(0, "a story", 2)],
        assignments=[("canon", 0), ("own", 0)],
    )

    assert store.duplicate_coverage() == {"canon": {"src_modeled_ce"}}
    # Two `self` documents and one collapsed copy from the other side: 2/3 of the
    # pair's coverage, under the 0.8 threshold, so not a blindspot. Reading only the
    # surviving documents' own sources it was 2/2 and reported as one.
    assert blindspots_from_store(store, MEMBERS, dominance=0.8, min_size=2) == []
    assert blindspots_from_store(store, MEMBERS, dominance=0.6, min_size=2)[0].counts == {
        "self": 2,
        "modeled_ce": 1,
    }
    store.close()


def test_a_collapsed_copy_is_listed_among_the_outlets_that_carried_a_story():
    """The outlet list is what makes a blindspot card checkable. A card naming one
    masthead because deduplication hid the other two claims less than the corpus
    supports."""
    from cluster.blindspot import articles_from_store, blindspots_from_store

    store = Datastore(":memory:")
    for doc_id, source_id, dup_of, title in [
        ("canon", "src_modeled_ce", None, "Senate advances the funding bill"),
        ("dup", "src_ce_two", "canon", "Senate advances the funding bill"),
        ("solo", "src_modeled_ce", None, "A second story on the same subject"),
    ]:
        store.upsert_document(
            doc_id=doc_id,
            source_id=source_id,
            stratum_id=None,
            url=f"https://example.test/{doc_id}",
            title=title,
            published_utc=None,
            fetched_utc="2026-08-10T00:00:00+00:00",
            word_count=200,
            minhash=None,
            is_duplicate=dup_of is not None,
            duplicate_of=dup_of,
        )
    store.set_source_label("src_modeled_ce", "The First Outlet")
    store.set_source_label("src_ce_two", "The Second Outlet")
    store.replace_clustering(clusters=[(0, "funding", 2)], assignments=[("canon", 0), ("solo", 0)])

    members = {"self": {"src_self"}, "modeled_ce": {"src_modeled_ce", "src_ce_two"}}
    spots = blindspots_from_store(store, members, dominance=0.8, min_size=2)
    articles = articles_from_store(store, spots, members)
    outlets = {a.outlet for a in articles[0]}
    assert outlets == {"The First Outlet", "The Second Outlet"}
    store.close()


def test_agenda_counts_credit_a_collapsed_copy_to_its_own_outlet():
    """Attention shares had the same flaw: a wire story counted for whichever
    outlet was fetched first and as zero coverage for every other."""
    store = Datastore(":memory:")
    for doc_id, source_id, dup_of in [
        ("canon", "src_self", None),
        ("dup", "src_modeled_ce", "canon"),
    ]:
        store.upsert_document(
            doc_id=doc_id,
            source_id=source_id,
            stratum_id=None,
            url=None,
            title="t",
            published_utc=None,
            fetched_utc="2026-08-10T00:00:00+00:00",
            word_count=200,
            minhash=None,
            is_duplicate=dup_of is not None,
            duplicate_of=dup_of,
        )
    store.replace_clustering(clusters=[(0, "a story", 1)], assignments=[("canon", 0)])

    counts = {(r["cluster_id"], r["source_id"]): r["n"] for r in store.cluster_source_counts()}
    assert counts == {(0, "src_self"): 1, (0, "src_modeled_ce"): 1}
    store.close()


def test_a_duplicate_that_somehow_got_clustered_is_not_counted_twice():
    """`document_clusters` holds only canonicals today, because only canonicals are
    embedded. The query asserts it anyway: a duplicate landing there would be counted
    on both legs of the union, and a double-counted outlet is the failure
    cluster_source_counts exists to fix."""
    store = Datastore(":memory:")
    for doc_id, source_id, dup_of in [
        ("canon", "src_self", None),
        ("dup", "src_modeled_ce", "canon"),
    ]:
        store.upsert_document(
            doc_id=doc_id,
            source_id=source_id,
            stratum_id=None,
            url=None,
            title="t",
            published_utc=None,
            fetched_utc="2026-08-10T00:00:00+00:00",
            word_count=200,
            minhash=None,
            is_duplicate=dup_of is not None,
            duplicate_of=dup_of,
        )
    # Assign both, including the duplicate, which the pipeline would never do.
    store.replace_clustering(clusters=[(0, "a story", 2)], assignments=[("canon", 0), ("dup", 0)])

    counts = {(r["cluster_id"], r["source_id"]): r["n"] for r in store.cluster_source_counts()}
    assert counts == {(0, "src_self"): 1, (0, "src_modeled_ce"): 1}
    store.close()


def test_an_empty_source_id_is_not_an_outlet():
    """'' would bucket as a nameless share in the attention numbers. Guarded on both
    legs of the union, matching duplicate_coverage."""
    store = Datastore(":memory:")
    for doc_id, source_id, dup_of in [
        ("canon", "src_self", None),
        ("nameless", "", "canon"),
    ]:
        store.upsert_document(
            doc_id=doc_id,
            source_id=source_id,
            stratum_id=None,
            url=None,
            title="t",
            published_utc=None,
            fetched_utc="2026-08-10T00:00:00+00:00",
            word_count=200,
            minhash=None,
            is_duplicate=dup_of is not None,
            duplicate_of=dup_of,
        )
    store.replace_clustering(clusters=[(0, "a story", 1)], assignments=[("canon", 0)])

    sources = {r["source_id"] for r in store.cluster_source_counts()}
    assert sources == {"src_self"}
    assert store.duplicate_coverage() == {}
    store.close()
