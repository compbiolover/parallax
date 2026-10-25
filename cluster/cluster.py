"""Cluster documents into "stories" from their stored embeddings.

Pipeline: load persisted per-document embeddings -> optional TruncatedSVD
dimensionality reduction (UMAP is a documented upgrade but not required) ->
HDBSCAN density clustering. Each resulting cluster is a story; noise points get
label -1.

scikit-learn is imported lazily so the ``cluster`` package stays importable
(and ingestion, which only needs the embedder, stays light); it is a core
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ingestion.datastore import Datastore


@dataclass
class ClusterResult:
    doc_ids: list[str]
    # The source each document came from. Persona membership is resolved from
    # this, because a source several personas read puts its documents in all of
    # their clusters — which one diet id per document could not express.
    sources: list[str]
    # Every source that carried this story: its own, plus the outlets whose
    # near-duplicate copies were collapsed into it. Coverage questions read this
    # rather than `sources` — a wire story is clustered once, and crediting it to
    # the one outlet fetched first is what made a shared story look one-sided.
    coverage: list[frozenset[str]]
    titles: list[str | None]
    labels: list[int]  # cluster id per doc, -1 = noise
    # The raw embedding per document, in the embedder's own space. Thread
    # matching needs these and cannot use the reduced coordinates the
    # clustering runs in: `_reduce` refits the SVD on each run's documents, so
    # two days' reduced vectors are in different bases and a cosine between
    # them is meaningless. Optional because older callers build a result
    # without them, and nothing but thread matching reads it.
    vectors: list[list[float]] | None = None

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def cluster_ids(self) -> list[int]:
        return sorted({lbl for lbl in self.labels if lbl != -1})


def compute_clustering(
    store: Datastore,
    min_cluster_size: int = 2,
    svd_components: int = 30,
    random_state: int = 42,
    since: str | None = None,
    min_words: int = 0,
) -> ClusterResult:
    """Load embeddings and cluster them. Returns per-document labels.

    Only embeddings from the active embedder (``meta['embedder']``) are
    clustered — vectors from different embedders live in incompatible spaces.
    If the datastore mixes embedders and none is recorded as active, a clear
    error is raised rather than crashing deep in numpy on ragged vectors.

    ``since`` (``YYYY-MM-DD``) clusters only documents dated on or after it.

    ``min_words`` excludes headline-only documents, which are what defines a
    *story* here. It defaults to off at this level and is applied by
    :func:`cluster.blindspot.run_clustering`, which is where a policy about
    real corpora belongs; callers clustering a hand-built set get every
    document they put in. They still count everywhere coverage is counted — a headline
    is real evidence that an outlet ran something — so this narrows what can
    form a cluster, not what the corpus contains. ``0`` clusters everything,
    which is the pre-2026-10 behaviour.
    """
    active = store.get_meta("embedder")
    # Only canonical documents carry embeddings, so a wire story that ran in six
    # outlets is clustered once. This restores the other five for coverage.
    # Under a window, only copies dated inside it count, the canonical included:
    # an outlet that ran a story weeks ago did not cover it this week.
    collapsed = store.duplicate_coverage(since)
    ids, sources, coverage, titles, vectors = [], [], [], [], []
    rows = store.iter_embeddings(embedder=active, since=since, min_words=min_words)
    for doc_id, source_id, title, dated, vec in rows:
        ids.append(doc_id)
        sources.append(source_id)
        own = {source_id} if since is None or dated >= since else set()
        coverage.append(frozenset(own | collapsed.get(doc_id, set())))
        titles.append(title)
        vectors.append(vec)

    if len(ids) < max(2, min_cluster_size):
        # Too few documents to cluster meaningfully — everything is noise.
        return ClusterResult(ids, sources, coverage, titles, [-1] * len(ids), vectors)

    dims = {len(v) for v in vectors}
    if len(dims) > 1:
        raise ValueError(
            f"Embeddings have mixed dimensions {sorted(dims)} from embedders "
            f"{store.embedder_names()} — the datastore mixes multiple embedders. "
            "Re-ingest after changing the embedder, or use a fresh datastore."
        )

    X = np.asarray(vectors, dtype=np.float32)
    X = _reduce(X, svd_components, random_state)
    labels = _hdbscan_labels(X, min_cluster_size)
    return ClusterResult(ids, sources, coverage, titles, [int(x) for x in labels], vectors)


def _reduce(X: np.ndarray, components: int, random_state: int) -> np.ndarray:
    n_samples, n_features = X.shape
    k = min(components, n_features - 1, n_samples - 1)
    if k < 2:
        return X
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    reduced = TruncatedSVD(n_components=k, random_state=random_state).fit_transform(X)
    # Re-normalize so Euclidean distance tracks cosine similarity for HDBSCAN.
    return normalize(reduced)


def _hdbscan_labels(X: np.ndarray, min_cluster_size: int) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    mcs = max(2, min(min_cluster_size, X.shape[0] // 2))
    # 'leaf' selection yields finer, more coherent story clusters than the
    # default 'eom', which tends to merge diverse news into one blob.
    model = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="leaf",
        copy=False,
    )
    return model.fit_predict(X)
