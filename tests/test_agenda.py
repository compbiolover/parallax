"""Attention divergence: the second number, over story clusters.

The foundation divergence is small on a real corpus because averages converge.
These tests pin the claim this module makes instead — that two diets reading
about different events diverge on *attention* even when their moral vocabulary
matches — and the honesty properties around it: shares not counts, noise
excluded, thin coverage flagged, absence rendered as absence.
"""

from __future__ import annotations

from compare.agenda import THIN_ARTICLES, attention_shares, compare_agendas
from ingestion.datastore import Datastore


def _store() -> Datastore:
    return Datastore(":memory:")


def _seed(store: Datastore, layout: dict[str, dict[int, int]]) -> None:
    """``{diet: {cluster_id: how many articles}}`` -> documents + a clustering.

    Cluster ``-1`` is HDBSCAN's noise label and is seeded like any other, so a
    test can assert it is excluded rather than assume it.
    """
    assignments: list[tuple[str, int]] = []
    sizes: dict[int, int] = {}
    for diet, per_cluster in layout.items():
        for cluster_id, n in per_cluster.items():
            sizes[cluster_id] = sizes.get(cluster_id, 0) + n
            for i in range(n):
                doc_id = f"{diet}-{cluster_id}-{i}"
                store.upsert_document(
                    doc_id=doc_id, diet_id=diet, source_id="s", stratum_id=None,
                    url=None, title=f"{diet} story {cluster_id} article {i}",
                    published_utc=None, fetched_utc="2026-08-01T00:00:00+00:00",
                    word_count=400, minhash=None,
                )
                assignments.append((doc_id, cluster_id))
    clusters = [(c, f"cluster {c}", n) for c, n in sorted(sizes.items())]
    store.replace_clustering(clusters=clusters, assignments=assignments)


def test_disjoint_agendas_diverge_completely():
    """Neither diet touched a story the other did — the ceiling of the scale."""
    store = _store()
    _seed(store, {"self": {0: 20, 1: 20}, "modeled_ce": {2: 20, 3: 20}})
    a = compare_agendas(store)
    assert a is not None
    assert a.divergence == 1.0
    assert a.exclusive == {"self": 1.0, "modeled_ce": 1.0}
    assert a.shared_stories == 0
    assert a.overlap == 0.0
    store.close()


def test_identical_agendas_do_not_diverge():
    store = _store()
    _seed(store, {"self": {0: 20, 1: 20}, "modeled_ce": {0: 20, 1: 20}})
    a = compare_agendas(store)
    assert a.divergence < 1e-9
    assert a.exclusive == {"self": 0.0, "modeled_ce": 0.0}
    assert a.overlap == 1.0
    store.close()


def test_attention_is_measured_in_shares_not_counts():
    """One diet ingesting twice as much is a fact about the source registry,
    not about the agenda. Same proportions, same attention."""
    store = _store()
    _seed(store, {"self": {0: 30, 1: 10}, "modeled_ce": {0: 60, 1: 20}})
    shares = attention_shares(store)
    assert shares["self"] == shares["modeled_ce"] == {0: 0.75, 1: 0.25}
    assert compare_agendas(store).divergence < 1e-9
    store.close()


def test_a_story_only_one_diet_touched_is_where_they_differ():
    """The union of clusters is the key set, so an exclusive story counts as a
    difference rather than being dropped for having no counterpart."""
    store = _store()
    _seed(store, {"self": {0: 30, 1: 10}, "modeled_ce": {0: 30}})
    a = compare_agendas(store)
    assert a.divergence > 0.0
    assert a.exclusive["self"] == 0.25          # 10 of 40 articles
    assert a.exclusive["modeled_ce"] == 0.0
    assert a.exclusive_stories == {"self": 1, "modeled_ce": 0}
    assert a.shared_stories == 1
    assert a.total_stories == 2
    assert a.overlap == 0.5
    store.close()


def test_unclustered_documents_are_not_a_story_either_diet_chose():
    """Noise is what the engine could not place. Counting it as a shared story
    would make a bad clustering day look like agreement."""
    store = _store()
    _seed(store, {"self": {0: 20, -1: 40}, "modeled_ce": {1: 20, -1: 40}})
    a = compare_agendas(store)
    assert a.total_stories == 2
    assert a.articles == {"self": 20, "modeled_ce": 20}
    assert a.divergence == 1.0
    store.close()


def test_thin_coverage_is_flagged_rather_than_hidden():
    store = _store()
    _seed(store, {"self": {0: THIN_ARTICLES}, "modeled_ce": {1: THIN_ARTICLES - 1}})
    assert compare_agendas(store).thin is True
    store.close()


def test_ample_coverage_is_not_flagged_thin():
    store = _store()
    _seed(store, {"self": {0: THIN_ARTICLES}, "modeled_ce": {1: THIN_ARTICLES}})
    assert compare_agendas(store).thin is False
    store.close()


def test_nothing_clustered_is_absence_not_agreement():
    """`None`, not 0.0 — "no clustering" and "identical agendas" are different
    statements and the surfaces render an absent metric as silence."""
    store = _store()
    assert compare_agendas(store) is None
    assert attention_shares(store) == {}
    store.close()


def test_one_diet_alone_cannot_be_compared():
    store = _store()
    _seed(store, {"self": {0: 20, 1: 20}})
    assert compare_agendas(store) is None
    store.close()


def test_the_comparison_serializes_flat():
    store = _store()
    _seed(store, {"self": {0: 30, 1: 10}, "modeled_ce": {0: 30}})
    d = compare_agendas(store).to_dict()
    assert d["pair"] == ["modeled_ce", "self"]
    assert set(d) == {
        "pair", "divergence", "exclusive", "exclusive_stories", "articles",
        "shared_stories", "total_stories", "overlap", "thin",
    }
    assert 0.0 <= d["divergence"] <= 1.0
    store.close()
