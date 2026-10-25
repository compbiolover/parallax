"""Story threads: the identity a story keeps once the clusters under it change.

The point of these tests is the thing clusters cannot do. A cluster id is a
position in one HDBSCAN run, so the only interesting assertions here are about
what survives a *second* run: the same thread id, a timeline that grew by one
point, and a first_seen that did not move.
"""

from __future__ import annotations

import json

from cluster.cluster import ClusterResult
from cluster.threads import (
    MAX_DORMANT_DAYS,
    SIMILARITY_THRESHOLD,
    ThreadMatch,
    centroid,
    cluster_centroids,
    cosine,
    match,
    normalize,
    slugify,
    track,
)
from ingestion.datastore import Datastore


def _result(labels, vectors, titles=None, sources=None, doc_ids=None):
    n = len(labels)
    doc_ids = doc_ids or [f"d{i}" for i in range(n)]
    sources = sources or [f"s{i}" for i in range(n)]
    titles = titles or [f"title {i}" for i in range(n)]
    return ClusterResult(
        doc_ids=doc_ids,
        sources=sources,
        coverage=[frozenset({s}) for s in sources],
        titles=titles,
        labels=labels,
        vectors=vectors,
    )


def _store(tmp_path, embedder="test-embedder"):
    store = Datastore(str(tmp_path / "t.sqlite"))
    store.set_meta("embedder", embedder)
    return store


def _seed(store, result):
    """thread_documents references documents(id), so the rows have to exist."""
    for doc_id, source_id, title in zip(result.doc_ids, result.sources, result.titles, strict=True):
        store.upsert_document(
            doc_id=doc_id,
            source_id=source_id,
            stratum_id=None,
            url=f"https://example.test/{doc_id}",
            title=title,
            published_utc="2026-10-20T00:00:00+00:00",
            fetched_utc="2026-10-20T00:00:00+00:00",
            word_count=400,
            minhash=None,
        )


def _run(store, result, **kwargs):
    _seed(store, result)
    return track(store, result, **kwargs)


# -- vector helpers ---------------------------------------------------------


def test_normalize_makes_a_unit_vector():
    v = normalize([3.0, 4.0])
    assert abs(sum(x * x for x in v) - 1.0) < 1e-9


def test_normalize_survives_a_zero_vector():
    """A cluster whose vectors cancel out must not divide by zero."""
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


def test_cosine_of_mismatched_dimensions_is_zero_not_an_error():
    """A datastore that has seen two embedders holds centroids of two widths."""
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_centroid_is_the_normalized_mean():
    c = centroid([[1.0, 0.0], [0.0, 1.0]])
    assert abs(c[0] - c[1]) < 1e-9
    assert abs(sum(x * x for x in c) - 1.0) < 1e-9


def test_cluster_centroids_skips_noise():
    result = _result([0, 0, -1], [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    centroids = cluster_centroids(result)
    assert set(centroids) == {0}


def test_cluster_centroids_is_empty_without_vectors():
    """Older callers build a ClusterResult with no vectors; that is not a crash."""
    assert cluster_centroids(_result([0, 0], None)) == {}


# -- slugs ------------------------------------------------------------------


def test_slug_is_readable_and_unique():
    assert slugify("France protests over pensions", set()) == "france-protests-over-pensions"
    assert slugify("France protests", {"france-protests"}) == "france-protests-2"


def test_slug_handles_a_title_with_nothing_usable_in_it():
    assert slugify("!!! ???", set()) == "thread"


def test_slug_strips_accents_rather_than_dropping_the_word():
    assert slugify("Grève générale", set()) == "greve-generale"


# -- matching ---------------------------------------------------------------


def _thread(tid, vec, last_seen="2026-10-25", title="t"):
    return {"thread_id": tid, "title": title, "centroid": vec, "last_seen": last_seen}


def test_a_similar_cluster_joins_the_existing_thread():
    out = match({0: [1.0, 0.0]}, [_thread("a", [0.99, 0.14])], "2026-10-25")
    assert out[0].thread_id == "a"
    assert not out[0].is_new


def test_an_unrelated_cluster_matches_nothing():
    out = match({0: [1.0, 0.0]}, [_thread("a", [0.0, 1.0])], "2026-10-25")
    assert out == {}


def test_a_cluster_joins_its_most_similar_thread():
    """Not whichever thread happened to come first in the list above threshold."""
    threads = [_thread("loose", [0.80, 0.60]), _thread("tight", [1.0, 0.0])]
    out = match({0: [0.999, 0.045]}, threads, "2026-10-25")
    assert out[0].thread_id == "tight"


def test_two_clusters_may_share_one_thread():
    """A fragmented story (the protests, the pension vote) is one thread by
    design; `track` sums them into a single timeline point."""
    out = match({0: [0.80, 0.60], 1: [0.999, 0.045]}, [_thread("a", [1.0, 0.0])], "2026-10-25")
    assert {m.thread_id for m in out.values()} == {"a"}


def test_a_long_dormant_thread_is_not_resurrected():
    """Matching onto it would put a two-week hole in the middle of a timeline."""
    stale = _thread("a", [1.0, 0.0], last_seen="2026-10-01")
    assert match({0: [1.0, 0.0]}, [stale], "2026-10-25") == {}


def test_a_thread_inside_the_dormancy_window_still_matches():
    from datetime import date, timedelta

    day = date(2026, 10, 25)
    recent = (day - timedelta(days=MAX_DORMANT_DAYS)).isoformat()
    out = match({0: [1.0, 0.0]}, [_thread("a", [1.0, 0.0], last_seen=recent)], day.isoformat())
    assert out[0].thread_id == "a"


def test_threshold_is_the_documented_one():
    """It is a judgement about the embedder, so a silent change should fail."""
    assert SIMILARITY_THRESHOLD == 0.62


# -- tracking across runs ---------------------------------------------------


def test_track_creates_a_thread_and_one_timeline_point(tmp_path):
    store = _store(tmp_path)
    result = _result([0, 0], [[1.0, 0.0], [0.9, 0.1]], titles=["Pension strike", "Strike grows"])
    out = _run(store, result, day="2026-10-20", labels={0: "Pension strike"})
    assert len(out) == 1 and out[0].is_new
    rows = store.story_thread_rows()
    assert len(rows) == 1
    assert rows[0]["first_seen"] == "2026-10-20"
    timeline = store.thread_timeline(rows[0]["thread_id"])
    assert [r["doc_count"] for r in timeline] == [2]


def test_the_same_story_tomorrow_extends_the_thread(tmp_path):
    """The whole point: a new clustering, the same thread, two timeline points."""
    store = _store(tmp_path)
    day_one = _result([0, 0], [[1.0, 0.0], [0.98, 0.2]])
    _run(store, day_one, day="2026-10-20", labels={0: "Pension strike"})
    thread_id = store.story_thread_rows()[0]["thread_id"]

    # Tomorrow the story is cluster 7, with different documents.
    day_two = _result(
        [7, 7, 7],
        [[0.97, 0.24], [1.0, 0.0], [0.99, 0.1]],
        doc_ids=["x0", "x1", "x2"],
    )
    out = _run(store, day_two, day="2026-10-21", labels={7: "Pension strike widens"})
    assert [m.thread_id for m in out] == [thread_id]
    assert not out[0].is_new

    rows = store.story_thread_rows()
    assert len(rows) == 1, "a second run must not fork the thread"
    assert rows[0]["first_seen"] == "2026-10-20", "first_seen is what dates a timeline"
    assert rows[0]["last_seen"] == "2026-10-21"
    assert rows[0]["title"] == "Pension strike widens", "the title follows the coverage"
    timeline = store.thread_timeline(thread_id)
    assert [(r["day"], r["doc_count"]) for r in timeline] == [
        ("2026-10-20", 2),
        ("2026-10-21", 3),
    ]


def test_an_unrelated_story_starts_its_own_thread(tmp_path):
    store = _store(tmp_path)
    _run(store, _result([0], [[1.0, 0.0]]), day="2026-10-20", labels={0: "Pensions"})
    _run(store, _result([0], [[0.0, 1.0]]), day="2026-10-21", labels={0: "Hurricane"})
    assert len(store.story_thread_rows()) == 2


def test_rerunning_the_same_day_overwrites_its_point(tmp_path):
    """The daily job can run twice; the timeline has one point per day."""
    store = _store(tmp_path)
    _run(store, _result([0, 0], [[1.0, 0.0]] * 2), day="2026-10-20", labels={0: "Pensions"})
    _run(store, _result([0, 0, 0], [[1.0, 0.0]] * 3), day="2026-10-20", labels={0: "Pensions"})
    thread_id = store.story_thread_rows()[0]["thread_id"]
    timeline = store.thread_timeline(thread_id)
    assert [r["doc_count"] for r in timeline] == [3]


def test_a_note_survives_a_second_cluster_run_the_same_day(tmp_path):
    """The note costs a model call, so a re-cluster must not wipe it."""
    store = _store(tmp_path)
    _run(store, _result([0], [[1.0, 0.0]]), day="2026-10-20", labels={0: "Pensions"})
    thread_id = store.story_thread_rows()[0]["thread_id"]
    store.set_thread_note(thread_id, "2026-10-20", "The vote was delayed.")
    _run(store, _result([0], [[1.0, 0.0]]), day="2026-10-20", labels={0: "Pensions"})
    assert store.thread_timeline(thread_id)[0]["note"] == "The vote was delayed."


def test_centroids_from_another_embedder_are_ignored(tmp_path):
    """Two embedders live in incompatible spaces, so the old thread is not a match."""
    store = _store(tmp_path, embedder="old-embedder")
    _run(store, _result([0], [[1.0, 0.0]]), day="2026-10-20", labels={0: "Pensions"})
    store.set_meta("embedder", "new-embedder")
    _run(store, _result([0], [[1.0, 0.0]]), day="2026-10-21", labels={0: "Pensions"})
    rows = store.story_thread_rows()
    assert len(rows) == 2
    assert {r["embedder"] for r in rows} == {"old-embedder", "new-embedder"}


def test_per_diet_counts_come_from_collapsed_coverage(tmp_path):
    """A wire story is clustered once; both diets that ran it must be counted."""
    store = _store(tmp_path)
    result = ClusterResult(
        doc_ids=["d0", "d1"],
        sources=["guardian", "fox"],
        coverage=[frozenset({"guardian", "fox"}), frozenset({"fox"})],
        titles=["Shared story", "Their angle"],
        labels=[0, 0],
        vectors=[[1.0, 0.0], [0.99, 0.1]],
    )
    _run(
        store,
        result,
        day="2026-10-20",
        labels={0: "Shared story"},
        members={"self": {"guardian"}, "modeled_ce": {"fox"}},
    )
    row = store.thread_timeline(store.story_thread_rows()[0]["thread_id"])[0]
    assert json.loads(row["diet_counts"]) == {"self": 1, "modeled_ce": 2}


def test_the_motif_is_a_vote_not_the_first_document(tmp_path):
    """A cluster is not always one subject, so the theme is the commonest one."""
    store = _store(tmp_path)
    result = _result([0, 0, 0], [[1.0, 0.0]] * 3)
    themes = {
        "d0": ("sports", "Sport"),
        "d1": ("faith", "Faith & the church"),
        "d2": ("faith", "Faith & the church"),
    }
    _run(store, result, day="2026-10-20", labels={0: "Church league"}, themes=themes)
    assert store.story_thread_rows()[0]["theme_title"] == "Faith & the church"


def test_tracking_is_a_no_op_without_vectors(tmp_path):
    store = _store(tmp_path)
    assert _run(store, _result([0, 0], None), day="2026-10-20", labels={0: "x"}) == []
    assert store.story_thread_rows() == []


def test_threads_active_on_ranks_by_coverage(tmp_path):
    store = _store(tmp_path)
    result = _result([0, 0, 0, 1], [[1.0, 0.0]] * 3 + [[0.0, 1.0]])
    _run(store, result, day="2026-10-20", labels={0: "Big story", 1: "Small story"})
    active = store.threads_active_on("2026-10-20")
    assert [r["title"] for r in active] == ["Big story", "Small story"]
    assert store.threads_active_on("2026-10-21") == []


def test_match_returns_thread_match_objects():
    out = match({0: [1.0, 0.0]}, [_thread("a", [1.0, 0.0])], "2026-10-25")
    assert isinstance(out[0], ThreadMatch)
    assert out[0].similarity > 0.99
