"""Coverage-asymmetry (blindspot) detection over clusters.

A **blindspot** is a story cluster that one diet covers heavily and the other
barely covers at all. We emit both directions — what the modeled diet sees that
the author's diet doesn't, and vice versa — so the author's own blindspots are
surfaced with equal prominence (the symmetry requirement, ``CLAUDE.md`` §0).

Cluster labels and representative headlines come from the persisted document
titles (raw bodies are long gone). This module also persists the clustering and
returns everything the exporter needs.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ingestion.datastore import Datastore

from .cluster import ClusterResult, compute_clustering

_WORD_RE = re.compile(r"[a-z][a-z'-]+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "it",
    "its", "this", "that", "these", "those", "he", "she", "they", "we", "you",
    "his", "her", "their", "our", "your", "will", "would", "can", "could", "has",
    "have", "had", "not", "no", "new", "says", "say", "said", "after", "over",
    "how", "why", "what", "who", "amid", "into", "out", "up", "down", "about",
}


@dataclass
class Blindspot:
    cluster_id: int
    label: str
    counts: dict[str, int]                 # diet_id -> member count
    dominant_diet: str
    other_diet: str
    dominant_share: float
    size: int
    representative_titles: list[str] = field(default_factory=list)


@dataclass
class ClusteringOutcome:
    n_docs: int
    n_clusters: int
    n_noise: int
    blindspots: list[Blindspot]
    diets: list[str]


def label_cluster(titles: list[str | None], top: int = 4) -> str:
    counter: Counter[str] = Counter()
    for t in titles:
        if not t:
            continue
        for w in _WORD_RE.findall(t.lower()):
            if w not in _STOP and len(w) > 2:
                counter[w] += 1
    terms = [w for w, _ in counter.most_common(top)]
    return " · ".join(terms) if terms else "(untitled cluster)"


def _members_by_cluster(result: ClusterResult) -> dict[int, list[int]]:
    by: dict[int, list[int]] = {}
    for i, lbl in enumerate(result.labels):
        if lbl != -1:
            by.setdefault(lbl, []).append(i)
    return by


def detect_blindspots(
    result: ClusterResult,
    dominance: float = 0.8,
    min_size: int = 3,
) -> list[Blindspot]:
    """A cluster is a blindspot when one diet holds >= ``dominance`` of its
    members (and the cluster has at least ``min_size`` members)."""
    diets = sorted(set(result.diets))
    out: list[Blindspot] = []
    for cid, idxs in _members_by_cluster(result).items():
        if len(idxs) < min_size:
            continue
        counts = Counter(result.diets[i] for i in idxs)
        size = len(idxs)
        dominant_diet, dominant_count = counts.most_common(1)[0]
        share = dominant_count / size
        if share < dominance:
            continue
        other = next((d for d in diets if d != dominant_diet), "—")
        titles = [result.titles[i] for i in idxs]
        rep = [result.titles[i] for i in idxs if result.diets[i] == dominant_diet and result.titles[i]]
        out.append(
            Blindspot(
                cluster_id=cid,
                label=label_cluster(titles),
                counts=dict(counts),
                dominant_diet=dominant_diet,
                other_diet=other,
                dominant_share=share,
                size=size,
                representative_titles=rep[:3],
            )
        )
    out.sort(key=lambda b: (b.size, b.dominant_share), reverse=True)
    return out


def blindspots_from_store(
    store: Datastore, dominance: float = 0.75, min_size: int = 2
) -> list[Blindspot]:
    """Rebuild blindspots from the persisted clustering (no re-embedding).

    Lets the dashboard exporter surface blindspots without importing sklearn or
    recomputing embeddings — it just reads the stored assignment.
    """
    diets = store.diet_ids()
    out: list[Blindspot] = []
    for row in store.cluster_rows():
        cid = row["cluster_id"]
        if cid == -1:
            continue
        members = store.cluster_members(cid)
        size = len(members)
        if size < min_size:
            continue
        counts = Counter(m["diet_id"] for m in members)
        dominant_diet, dominant_count = counts.most_common(1)[0]
        share = dominant_count / size
        if share < dominance:
            continue
        other = next((d for d in diets if d != dominant_diet), "—")
        rep = [m["title"] for m in members if m["diet_id"] == dominant_diet and m["title"]]
        out.append(
            Blindspot(
                cluster_id=cid,
                label=row["label"] or "(untitled cluster)",
                counts=dict(counts),
                dominant_diet=dominant_diet,
                other_diet=other,
                dominant_share=share,
                size=size,
                representative_titles=rep[:3],
            )
        )
    out.sort(key=lambda b: (b.size, b.dominant_share), reverse=True)
    return out


def run_clustering(
    store: Datastore,
    min_cluster_size: int = 2,
    dominance: float = 0.75,
    min_blindspot_size: int = 2,
) -> ClusteringOutcome:
    """Cluster, persist the assignment, and detect blindspots."""
    result = compute_clustering(store, min_cluster_size=min_cluster_size)

    members = _members_by_cluster(result)
    cluster_rows = [
        (cid, label_cluster([result.titles[i] for i in idxs]), len(idxs))
        for cid, idxs in members.items()
    ]
    assignments = [
        (result.doc_ids[i], result.labels[i])
        for i in range(result.n_docs)
        if result.labels[i] != -1
    ]
    store.replace_clustering(cluster_rows, assignments)

    blindspots = detect_blindspots(result, dominance, min_blindspot_size)
    n_noise = sum(1 for lbl in result.labels if lbl == -1)
    return ClusteringOutcome(
        n_docs=result.n_docs,
        n_clusters=len(members),
        n_noise=n_noise,
        blindspots=blindspots,
        diets=sorted(set(result.diets)),
    )
