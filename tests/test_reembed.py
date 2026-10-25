"""Rebuilding every vector when the embedder changes.

Vectors from two embedders live in incompatible spaces, so a switch is not a
setting you flip — the corpus has to be rebuilt. What makes that possible at
all is that embeddings come from the *title* (`_cluster_text`), and titles are
stored permanently; raw article text is discarded at ingestion, so this is the
one stage of the pipeline that can be redone without re-fetching anything.
"""

from __future__ import annotations

import json

from ingestion.__main__ import _reembed
from ingestion.datastore import Datastore


class FakeEmbedder:
    """Deterministic 2-d vectors, so a re-embed is checkable by value."""

    name = "fake/v2"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.seen.append(text)
        return [float(len(text)), 1.0]


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
    assert _reembed(store, _settings()) == 0
    return embedder


# -- the rebuild ------------------------------------------------------------


def test_every_document_is_re_embedded_from_its_title(tmp_path, monkeypatch):
    store = _store(tmp_path, ["A protest in Lyon", "A second headline here"])
    embedder = _run(store, monkeypatch)
    assert sorted(embedder.seen) == ["A protest in Lyon", "A second headline here"]
    rows = list(store.conn.execute("SELECT document_id, vector, embedder, dim FROM embeddings"))
    assert {r["embedder"] for r in rows} == {"fake/v2"}
    assert {r["dim"] for r in rows} == {2}
    assert json.loads(rows[0]["vector"])[0] == float(len("A protest in Lyon"))


def test_the_active_embedder_is_recorded(tmp_path, monkeypatch):
    """`iter_embeddings` filters on it, so a stale value clusters nothing."""
    store = _store(tmp_path, ["A headline about something"])
    _run(store, monkeypatch)
    assert store.get_meta("embedder") == "fake/v2"


def test_vectors_from_the_old_embedder_are_gone(tmp_path, monkeypatch):
    """A document the re-embed skips still holds a vector in the old space.

    Left behind it is dead weight every later read has to filter around, and
    on the live corpus there were 318 of them: titleless rows and duplicates
    that had been embedded under the previous embedder.
    """
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
    assert sorted(store.embedder_names()) == ["old/v1"]
    _run(store, monkeypatch)
    assert store.embedder_names() == ["fake/v2"]


def test_story_threads_are_forgotten(tmp_path, monkeypatch):
    """A centroid is in its embedder's space; after a switch nothing in the
    table can be matched against again."""
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
    _run(store, monkeypatch)
    assert store.story_thread_rows() == []
    assert store.latest_thread_day() is None


def test_documents_without_a_title_are_skipped(tmp_path, monkeypatch):
    """A title is all that survives ingestion, so there is nothing to embed."""
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
    embedder = _run(store, monkeypatch)
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
    embedder = _run(store, monkeypatch)
    assert embedder.seen == ["A real headline"]


def test_an_empty_store_is_not_an_error(tmp_path, monkeypatch):
    store = Datastore(str(tmp_path / "empty.sqlite"))
    monkeypatch.setattr(
        "ingestion.__main__.build_embedder", lambda settings: (FakeEmbedder(), "fake/v2")
    )
    assert _reembed(store, _settings()) == 0


def test_a_batch_boundary_does_not_drop_documents(tmp_path, monkeypatch):
    """The loop writes on a full batch and again at the end; an exact multiple
    of the batch size is where a double-write or a lost tail would show."""
    from ingestion.__main__ import REEMBED_BATCH

    titles = [f"Headline number {i}" for i in range(REEMBED_BATCH + 1)]
    store = _store(tmp_path, titles)
    _run(store, monkeypatch)
    assert store.embedding_count() == len(titles)
