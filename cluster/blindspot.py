"""Coverage-asymmetry (blindspot) detection over clusters.

A **blindspot** is a story cluster that one diet covers heavily and the other
barely covers at all. We emit both directions — what the modeled diet sees that
the author's diet doesn't, and vice versa — so the author's own blindspots are
surfaced with equal prominence (the symmetry requirement, ``CLAUDE.md`` §0).

Cluster labels and representative headlines come from the persisted document
titles (raw bodies are long gone), cleaned by :mod:`cluster.titles` first —
labels built from raw GDELT strings inherited their tokenization and their
outlet stamps, which is how a cluster came to be called "christianity today ·
christianity · today".

A cluster is the unit the asymmetry is *measured* on. The unit it is *read* on
is a theme (:mod:`cluster.themes`), assigned here so the naming happens once per
run rather than once per surface.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ingestion.datastore import Datastore

from .cluster import ClusterResult, compute_clustering
from .themes import Article, Theme, ThemeAssignment, assign_themes, group_blindspots
from .titles import clean_title, clean_titles

_WORD_RE = re.compile(r"[a-z][a-z'-]+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "it",
    "its", "this", "that", "these", "those", "he", "she", "they", "we", "you",
    "his", "her", "their", "our", "your", "will", "would", "can", "could", "has",
    "have", "had", "not", "no", "new", "says", "say", "said", "after", "over",
    "how", "why", "what", "who", "amid", "into", "out", "up", "down", "about",
}


# How many headlines to keep per blindspot. Wider than any one surface shows,
# because a theme pools the headlines of every cluster under it and then picks —
# keeping three per cluster capped a six-cluster theme at material from two.
REPRESENTATIVE_LIMIT = 6


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
    themes: list[Theme] = field(default_factory=list)


def label_cluster(titles: list[str | None], top: int = 4) -> str:
    """Simple frequency label (no sklearn) — the fallback labeler."""
    counter: Counter[str] = Counter()
    for t in clean_titles(titles):
        for w in _WORD_RE.findall(t.lower()):
            if w not in _STOP and len(w) > 2:
                counter[w] += 1
    terms = [w for w, _ in counter.most_common(top)]
    return " · ".join(terms) if terms else "(untitled cluster)"


def label_clusters(cluster_titles: dict[int, list[str | None]], top: int = 3) -> dict[int, str]:
    """Corpus-aware c-TF-IDF labels: terms *distinctive* to each cluster.

    Treats each cluster's concatenated titles as one document and scores terms by
    TF-IDF across clusters, so generic words shared by every cluster ("trump",
    "new") are down-weighted in favor of what sets a cluster apart. Falls back to
    the frequency labeler when sklearn is unavailable or the vocabulary is empty.

    These labels are a technical readout — the theme (:mod:`cluster.themes`) is
    what a reader is given. They still run on cleaned titles: an outlet stamp
    repeated across every headline in a cluster is exactly the kind of term
    c-TF-IDF finds distinctive, and it names the publisher, not the story.
    """
    cleaned = {c: clean_titles(ts) for c, ts in cluster_titles.items()}
    cids = [c for c, ts in cleaned.items() if ts]
    if not cids:
        return {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return {c: label_cluster(cluster_titles[c]) for c in cids}

    docs = [" ".join(cleaned[c]) for c in cids]
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"[A-Za-z][A-Za-z]+",
            min_df=1,
        )
        matrix = vec.fit_transform(docs)
    except ValueError:  # empty vocabulary
        return {c: label_cluster(cluster_titles[c]) for c in cids}

    terms = vec.get_feature_names_out()
    labels: dict[int, str] = {}
    for i, c in enumerate(cids):
        row = matrix[i].toarray().ravel()
        ranked = row.argsort()[::-1][:top]
        chosen = [terms[j] for j in ranked if row[j] > 0]
        labels[c] = " · ".join(chosen) if chosen else label_cluster(cluster_titles[c])
    return labels


def _representative(titles: list[str | None]) -> list[str]:
    """Most descriptive headlines first, cleaned, index pages dropped.

    Longer titles carry more story detail, so they lead — but length is measured
    after cleaning, or an outlet stamp counts as detail and a headline wins on
    the strength of the name attached to it.
    """
    return clean_titles(sorted((t for t in titles if t), key=len, reverse=True))


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
    labels: dict[int, str] | None = None,
) -> list[Blindspot]:
    """A cluster is a blindspot when one diet holds >= ``dominance`` of its
    members (and the cluster has at least ``min_size`` members).

    ``labels`` supplies precomputed (c-TF-IDF) labels; without it, each cluster
    is labeled with the simple frequency fallback.
    """
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
        rep = _representative(
            [result.titles[i] for i in idxs if result.diets[i] == dominant_diet]
        )
        label = labels.get(cid) if labels else label_cluster(titles)
        out.append(
            Blindspot(
                cluster_id=cid,
                label=label or "(untitled cluster)",
                counts=dict(counts),
                dominant_diet=dominant_diet,
                other_diet=other,
                dominant_share=share,
                size=size,
                representative_titles=rep[:REPRESENTATIVE_LIMIT],
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
        rep = _representative([m["title"] for m in members if m["diet_id"] == dominant_diet])
        out.append(
            Blindspot(
                cluster_id=cid,
                label=row["label"] or "(untitled cluster)",
                counts=dict(counts),
                dominant_diet=dominant_diet,
                other_diet=other,
                dominant_share=share,
                size=size,
                representative_titles=rep[:REPRESENTATIVE_LIMIT],
            )
        )
    out.sort(key=lambda b: (b.size, b.dominant_share), reverse=True)
    return out


def articles_from_store(
    store: Datastore, blindspots: list[Blindspot]
) -> dict[int, list[Article]]:
    """The dominant diet's articles per blindspot cluster, with outlet and link.

    Only the dominant diet's: the card's claim is "these outlets carried it and
    yours did not", so listing the one or two articles from the other side
    under that heading would contradict the sentence above them.
    """
    outlets = store.source_labels()
    out: dict[int, list[Article]] = {}
    for spot in blindspots:
        articles: list[Article] = []
        for member in store.cluster_members(spot.cluster_id):
            if member["diet_id"] != spot.dominant_diet:
                continue
            title = clean_title(member["title"])
            if not title:
                continue
            articles.append(Article(
                doc_id=member["id"],
                title=title,
                url=member["url"],
                outlet=outlets.get(member["source_id"], ""),
            ))
        if articles:
            out[spot.cluster_id] = articles
    return out


def themes_from_store(store: Datastore, blindspots: list[Blindspot] | None = None) -> list[Theme]:
    """Themed blindspots from the datastore, for a surface that only reads.

    Uses the assignments the cluster run persisted. When there are none — an
    older datastore, or a run that predates theming — it themes from the stored
    headlines with the taxonomy instead of returning nothing, so the dashboard
    and the email never fall back to the unreadable cluster labels.
    """
    spots = blindspots if blindspots is not None else blindspots_from_store(store)
    stored = {
        row["document_id"]: ThemeAssignment(
            row["theme_key"], row["theme_title"], row["method"]
        )
        for row in store.blindspot_theme_rows()
    }
    return group_blindspots(spots, stored, articles_from_store(store, spots))


def run_clustering(
    store: Datastore,
    min_cluster_size: int = 2,
    dominance: float = 0.75,
    min_blindspot_size: int = 2,
    theme_model: str | None = None,
    theme_client: object | None = None,
    claude_themes: bool = True,
) -> ClusteringOutcome:
    """Cluster, persist the assignment, detect blindspots, and theme them."""
    result = compute_clustering(store, min_cluster_size=min_cluster_size)

    members = _members_by_cluster(result)
    cluster_titles = {cid: [result.titles[i] for i in idxs] for cid, idxs in members.items()}
    labels = label_clusters(cluster_titles)
    cluster_rows = [
        (cid, labels.get(cid) or "(untitled cluster)", len(idxs))
        for cid, idxs in members.items()
    ]
    assignments = [
        (result.doc_ids[i], result.labels[i])
        for i in range(result.n_docs)
        if result.labels[i] != -1
    ]
    store.replace_clustering(cluster_rows, assignments)

    blindspots = detect_blindspots(result, dominance, min_blindspot_size, labels=labels)
    themes = _theme_blindspots(store, blindspots, theme_model, theme_client, claude_themes)
    n_noise = sum(1 for lbl in result.labels if lbl == -1)
    return ClusteringOutcome(
        n_docs=result.n_docs,
        n_clusters=len(members),
        n_noise=n_noise,
        blindspots=blindspots,
        diets=sorted(set(result.diets)),
        themes=themes,
    )


def _theme_blindspots(
    store: Datastore,
    blindspots: list[Blindspot],
    theme_model: str | None,
    theme_client: object | None,
    claude_themes: bool,
) -> list[Theme]:
    """Name the themes once, here, and persist them.

    The cluster run is the only step that is allowed to spend a model call on
    this. Doing it at export time instead would bill a call per surface and let
    the email and the dashboard disagree about what a theme is called on the
    same day's data.
    """
    if not blindspots:
        store.replace_blindspot_themes([])
        return []
    articles = articles_from_store(store, blindspots)
    # One entry per article, because the subject is a property of the headline
    # and not of the cluster it landed in.
    entries = [(a.doc_id, [a.title]) for group in articles.values() for a in group]
    kwargs = {"use_claude": claude_themes}
    if theme_model:
        kwargs["model"] = theme_model
    assignments = assign_themes(entries, client=theme_client, **kwargs)
    store.replace_blindspot_themes(
        [(doc_id, a.key, a.title, a.method) for doc_id, a in assignments.items()]
    )
    return group_blindspots(blindspots, assignments, articles)
