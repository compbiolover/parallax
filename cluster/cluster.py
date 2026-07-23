"""Cluster documents into "stories" from their stored embeddings.

Pipeline: load persisted per-document embeddings -> optional TruncatedSVD
dimensionality reduction (UMAP is a documented upgrade but not required) ->
HDBSCAN density clustering. Each resulting cluster is a story; noise points get
label -1.

scikit-learn is imported lazily so the ``cluster`` package stays importable
(and ingestion, which only needs the embedder, stays light) when sklearn is not
installed. Install it with ``pip install parallax[cluster]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ingestion.datastore import Datastore


@dataclass
class ClusterResult:
    doc_ids: list[str]
    diets: list[str]
    titles: list[str | None]
    labels: list[int]         # cluster id per doc, -1 = noise

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
) -> ClusterResult:
    """Load embeddings and cluster them. Returns per-document labels."""
    ids, diets, titles, vectors = [], [], [], []
    for doc_id, diet_id, title, _unused, vec in store.iter_embeddings():
        ids.append(doc_id)
        diets.append(diet_id)
        titles.append(title)
        vectors.append(vec)

    if len(ids) < max(2, min_cluster_size):
        # Too few documents to cluster meaningfully — everything is noise.
        return ClusterResult(ids, diets, titles, [-1] * len(ids))

    X = np.asarray(vectors, dtype=np.float32)
    X = _reduce(X, svd_components, random_state)
    labels = _hdbscan_labels(X, min_cluster_size)
    return ClusterResult(ids, diets, titles, [int(x) for x in labels])


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
