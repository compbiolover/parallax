"""Rebuilding every vector when the embedder changes.

Vectors from two embedders live in incompatible spaces, so a switch is not a
setting you flip — the corpus has to be rebuilt. What makes that possible at
all is that embeddings come from the *title* (`_cluster_text`), and titles are
stored permanently; raw article text is discarded at ingestion, so this is the
one stage of the pipeline that can be redone without re-fetching anything.

The hazards these tests pin are all about the transition rather than the
embedding: a half-finished switch that reads as a finished one, a repeat run
that destroys history it had no reason to touch, and a single bad batch taking
the whole corpus down with it.
"""

from __future__ import annotations

import json

import pytest

from ingestion.__main__ import REEMBED_BATCH, _reembed
from ingestion.datastore import Datastore


class FakeEmbedder:
    """Deterministic vectors keyed on the text, so a re-embed is checkable."""

    name = "fake/v2"

    def __init__(self, fail_on: str | None = None) -> None:
        self.seen: list[str] = []
        self.batches: list[int] = []
        self.fail_on = fail_on

    def embed(self, text: str) -> list[float]:
        self.seen.append(text)
        return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise RuntimeError("model blew up")
        return [self.embed(t) for t in texts]


def _settings():
    return {"cluster": {"embedder": {"kind": "fake"}}}


def _store(tmp_path, titles, embedder="old/v1"):
    store = Datastore(str(tmp_path / "r.sqlite"))
    store.set_meta("embedder", embedder)
    for i, title in enumerate(titles):
        store.upsert_document(
            doc_id=f"d{i}",
            source_id="src",
            stratum_id=None,
            url=None,
            title=title,
            published_utc=None,
            fetched_utc="2026-10-25T00:00:00+00:00",
            word_count=400,
            minhash=None,
        )
        store.upsert_embedding(document_id=f"d{i}", vector=[0.0, 1.0], embedder=embedder)
    return store


def _run(store, monkeypatch, embedder=None):
    embedder = embedder or FakeEmbedder()
    monkeypatch.setattr(
        "ingestion.__main__.build_embedder", lambda settings: (embedder, embedder.name)
    )
    code = _reembed(store, _settings())
    return embedder, code


# -- the rebuild ------------------------------------------------------------


def test_every_document_is_re_embedded_from_its_title(tmp_path, monkeypatch):
    store = _store(tmp_path, ["A protest in Lyon", "A second headline here"])
    embedder, code = _run(store, monkeypatch)
    assert code == 0
    assert sorted(embedder.seen) == ["A protest in Lyon", "A second headline here"]
    rows = list(store.conn.execute("SELECT document_id, vector, embedder, dim FROM embeddings"))
    assert {r["embedder"] for r in rows} == {"fake/v2"}
    assert {r["dim"] for r in rows} == {3}


def test_titles_are_encoded_in_batches(tmp_path, monkeypatch):
    """One encode call per batch, not per title: that is the 8.8x."""
    store = _store(tmp_path, [f"Headline number {i}" for i in range(REEMBED_BATCH + 5)])
    embedder, _ = _run(store, monkeypatch)
    assert embedder.batches == [REEMBED_BATCH, 5]


def test_a_batch_boundary_does_not_drop_documents(tmp_path, monkeypatch):
    titles = [f"Headline number {i}" for i in range(REEMBED_BATCH + 1)]
    store = _store(tmp_path, titles)
    _run(store, monkeypatch)
    assert store.embedding_count() == len(titles)


def test_each_vector_lands_on_its_own_document(tmp_path, monkeypatch):
    """A transposed zip would still give the right row count."""
    store = _store(tmp_path, ["Alpha story here", "Beta", "Gamma story about things"])
    embedder, _ = _run(store, monkeypatch)
    for row in store.conn.execute("SELECT document_id, vector FROM embeddings"):
        title = store.conn.execute(
            "SELECT title FROM documents WHERE id = ?", (row["document_id"],)
        ).fetchone()["title"]
        assert json.loads(row["vector"]) == embedder.embed(title)


# -- the switch is atomic ---------------------------------------------------


def test_the_active_embedder_is_recorded(tmp_path, monkeypatch):
    store = _store(tmp_path, ["A headline about something"])
    _run(store, monkeypatch)
    assert store.get_meta("embedder") == "fake/v2"


def test_vectors_from_the_old_embedder_are_gone(tmp_path, monkeypatch):
    """A document the re-embed skips still holds a vector in the old space."""
    store = _store(tmp_path, ["A headline about something"])
    store.upsert_document(
        doc_id="skipped",
        source_id="src",
        stratum_id=None,
        url=None,
        title=None,
        published_utc=None,
        fetched_utc="2026-10-25T00:00:00+00:00",
        word_count=400,
        minhash=None,
    )
    store.upsert_embedding(document_id="skipped", vector=[0.0, 1.0], embedder="old/v1")
    _run(store, monkeypatch)
    assert store.embedder_names() == ["fake/v2"]


def test_the_clustering_is_cleared_too(tmp_path, monkeypatch):
    """It was built in the old space, so it is as unusable as the threads —
    and unlike them it is what the dashboard and the brief actually show."""
    store = _store(tmp_path, ["A headline about something"])
    store.replace_clustering([(0, "old cluster", 1)], [("d0", 0)])
    store.set_meta("cluster_since", "2026-10-01")
    _run(store, monkeypatch)
    assert store.cluster_rows() == []
    assert store.get_meta("cluster_since") is None


def test_story_threads_are_forgotten(tmp_path, monkeypatch):
    store = _store(tmp_path, ["A headline about something"])
    store.upsert_story_thread(
        thread_id="t1",
        title="Old story",
        centroid="[1.0, 0.0]",
        dim=2,
        embedder="old/v1",
        day="2026-10-24",
    )
    store.record_thread_day(
        thread_id="t1", day="2026-10-24", doc_count=3, diet_counts="{}", headline="h"
    )
    store.replace_thread_documents([("d0", "t1")])
    _run(store, monkeypatch)
    assert store.story_thread_rows() == []
    assert store.latest_thread_day() is None
    assert store.conn.execute("SELECT COUNT(*) c FROM thread_documents").fetchone()["c"] == 0


# -- the guard --------------------------------------------------------------


def test_running_it_again_changes_nothing_and_keeps_the_timelines(tmp_path, monkeypatch):
    """The destructive no-op: an unguarded second run re-embedded everything
    identically and still wiped every thread, including notes that cost a
    model call and cannot be rebuilt."""
    store = _store(tmp_path, ["A headline about something"])
    _run(store, monkeypatch)
    store.upsert_story_thread(
        thread_id="t1",
        title="A story",
        centroid="[1.0, 0.0]",
        dim=2,
        embedder="fake/v2",
        day="2026-10-25",
    )
    store.record_thread_day(
        thread_id="t1", day="2026-10-25", doc_count=2, diet_counts="{}", headline="h"
    )
    store.set_thread_note("t1", "2026-10-25", "The vote was delayed.")

    second, code = _run(store, monkeypatch)
    assert code == 0
    assert second.seen == [], "nothing should be re-encoded"
    assert len(store.story_thread_rows()) == 1
    assert store.thread_timeline("t1")[0]["note"] == "The vote was delayed."


# -- partial failure --------------------------------------------------------


def test_a_failing_batch_is_skipped_rather_than_ending_the_run(tmp_path, monkeypatch):
    titles = [f"Headline number {i}" for i in range(REEMBED_BATCH + 3)]
    titles[-1] = "poison headline"
    store = _store(tmp_path, titles)
    embedder, code = _run(store, monkeypatch, FakeEmbedder(fail_on="poison"))
    assert code == 0
    assert store.get_meta("embedder") == "fake/v2", "the first batch still converted"
    kinds = {r["embedder"] for r in store.conn.execute("SELECT embedder FROM embeddings")}
    assert kinds == {"fake/v2"}, "stale vectors must not survive the switch"


def test_a_total_failure_leaves_the_store_on_its_old_embedder(tmp_path, monkeypatch):
    """Half-switching is the state that made the brief republish stale data."""
    store = _store(tmp_path, ["poison headline here"])
    _, code = _run(store, monkeypatch, FakeEmbedder(fail_on="poison"))
    assert code == 1
    assert store.get_meta("embedder") == "old/v1"
    assert store.embedder_names() == ["old/v1"]


# -- what it works from -----------------------------------------------------


def test_documents_without_a_title_are_skipped(tmp_path, monkeypatch):
    store = _store(tmp_path, ["A real headline"])
    store.upsert_document(
        doc_id="untitled",
        source_id="src",
        stratum_id=None,
        url=None,
        title=None,
        published_utc=None,
        fetched_utc="2026-10-25T00:00:00+00:00",
        word_count=400,
        minhash=None,
    )
    embedder, _ = _run(store, monkeypatch)
    assert embedder.seen == ["A real headline"]


def test_a_whitespace_only_title_is_skipped(tmp_path, monkeypatch):
    store = _store(tmp_path, ["A real headline", "   "])
    embedder, _ = _run(store, monkeypatch)
    assert embedder.seen == ["A real headline"]


def test_duplicates_are_skipped(tmp_path, monkeypatch):
    """Only canonicals carry embeddings; a wire story is clustered once."""
    store = _store(tmp_path, ["A real headline"])
    store.upsert_document(
        doc_id="dup",
        source_id="src",
        stratum_id=None,
        url=None,
        title="A near copy of the headline",
        published_utc=None,
        fetched_utc="2026-10-25T00:00:00+00:00",
        word_count=400,
        minhash=None,
        is_duplicate=True,
        duplicate_of="d0",
    )
    embedder, _ = _run(store, monkeypatch)
    assert embedder.seen == ["A real headline"]


def test_an_empty_store_does_not_declare_a_new_embedder(tmp_path, monkeypatch):
    """What the guard actually prevents: declaring the switch done, and
    wiping every thread, on a store with nothing to embed."""
    store = Datastore(str(tmp_path / "empty.sqlite"))
    store.set_meta("embedder", "old/v1")
    store.upsert_story_thread(
        thread_id="t1",
        title="A story",
        centroid="[1.0]",
        dim=1,
        embedder="old/v1",
        day="2026-10-25",
    )
    store.record_thread_day(
        thread_id="t1", day="2026-10-25", doc_count=1, diet_counts="{}", headline="h"
    )
    _, code = _run(store, monkeypatch)
    assert code == 0
    assert store.get_meta("embedder") == "old/v1"
    assert len(store.story_thread_rows()) == 1


@pytest.mark.parametrize("kind", ["body", "title_only"])
def test_provenance_does_not_affect_what_gets_re_embedded(tmp_path, monkeypatch, kind):
    """Headlines are excluded from clustering, not from the store: they still
    need a current-space vector in case cluster_stubs is turned on."""
    store = Datastore(str(tmp_path / f"{kind}.sqlite"))
    store.set_meta("embedder", "old/v1")
    store.upsert_document(
        doc_id="d0",
        source_id="src",
        stratum_id=None,
        url=None,
        title="A headline about something",
        published_utc=None,
        fetched_utc="2026-10-25T00:00:00+00:00",
        word_count=12,
        minhash=None,
        ingest_kind=kind,
    )
    embedder, _ = _run(store, monkeypatch)
    assert embedder.seen == ["A headline about something"]
