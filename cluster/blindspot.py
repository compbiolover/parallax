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
from .themes import (
    UNSET,
    Article,
    Theme,
    ThemeAssignment,
    _Unset,
    assign_themes,
    group_blindspots,
)
from .titles import clean_title, clean_titles

_WORD_RE = re.compile(r"[a-z][a-z'-]+")
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "they",
    "we",
    "you",
    "his",
    "her",
    "their",
    "our",
    "your",
    "will",
    "would",
    "can",
    "could",
    "has",
    "have",
    "had",
    "not",
    "no",
    "new",
    "says",
    "say",
    "said",
    "after",
    "over",
    "how",
    "why",
    "what",
    "who",
    "amid",
    "into",
    "out",
    "up",
    "down",
    "about",
}


# How many headlines to keep per blindspot. Wider than any one surface shows,
# because a theme pools the headlines of every cluster under it and then picks —
# keeping three per cluster capped a six-cluster theme at material from two.
REPRESENTATIVE_LIMIT = 6


@dataclass
class Blindspot:
    cluster_id: int
    label: str
    counts: dict[str, int]  # persona_id -> member count, pair only
    dominant_diet: str
    other_diet: str
    dominant_share: float
    # Members the reference pair accounts for. The dominance share is measured
    # against this, not against the whole cluster.
    size: int
    # The whole cluster, including documents only personas outside the pair read.
    # Shown next to `size` so a cluster where most members come from elsewhere is
    # visibly that rather than looking like a small story.
    cluster_size: int = 0
    representative_titles: list[str] = field(default_factory=list)


@dataclass
class ClusteringOutcome:
    n_docs: int
    n_clusters: int
    n_noise: int
    blindspots: list[Blindspot]
    personas: list[str]
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
    members: dict[str, set[str]],
    dominance: float = 0.8,
    min_size: int = 3,
    labels: dict[int, str] | None = None,
) -> list[Blindspot]:
    """A cluster is a blindspot when one of the reference pair holds
    >= ``dominance`` of the members *the pair accounts for*.

    ``members`` is ``{persona_id: {source_id, ...}}`` for exactly the two
    personas being compared. Restricting to the pair is the correction that
    matters: the share used to be measured against *every* member of the cluster,
    so once three personas existed, two of them on the same side would split a
    cluster's coverage and push the leader below the threshold — dropping a real
    cross-family blindspot for an arithmetic reason.

    A document from a source both personas read counts toward both, pulling the
    share toward 0.5. That is right: it means both saw it. So does a story whose
    near-duplicate copies were collapsed: coverage is read from every outlet that
    carried it, not just the one whose copy survived deduplication. Crediting a
    wire story to whichever outlet was fetched first is what let a story both sides
    ran register as one side's blindspot.

    Counting is unweighted. A source weighted 0.05 counts the same as one at 1.0,
    because the question here is "did this reach them at all", which is binary; a
    fractional "partly saw it" would not be interpretable.

    ``labels`` supplies precomputed (c-TF-IDF) labels; without it, each cluster
    is labeled with the simple frequency fallback.
    """
    pair = list(members)
    if len(pair) != 2:
        raise ValueError(f"blindspots need exactly two personas, got {pair}")
    a, b = pair
    out: list[Blindspot] = []
    for cid, idxs in _members_by_cluster(result).items():
        in_a = [i for i in idxs if result.coverage[i] & members[a]]
        in_b = [i for i in idxs if result.coverage[i] & members[b]]
        counts = {a: len(in_a), b: len(in_b)}
        size = len(in_a) + len(in_b)
        if size < min_size:
            continue
        dominant, dominant_idxs = (a, in_a) if len(in_a) >= len(in_b) else (b, in_b)
        share = len(dominant_idxs) / size
        if share < dominance:
            continue
        titles = [result.titles[i] for i in idxs]
        rep = _representative([result.titles[i] for i in dominant_idxs])
        label = labels.get(cid) if labels else label_cluster(titles)
        out.append(
            Blindspot(
                cluster_id=cid,
                label=label or "(untitled cluster)",
                counts=counts,
                dominant_diet=dominant,
                other_diet=b if dominant == a else a,
                dominant_share=share,
                size=size,
                cluster_size=len(idxs),
                representative_titles=rep[:REPRESENTATIVE_LIMIT],
            )
        )
    out.sort(key=lambda b: (b.size, b.dominant_share), reverse=True)
    return out


def blindspots_from_store(
    store: Datastore,
    members: dict[str, set[str]],
    dominance: float = 0.75,
    min_size: int = 2,
) -> list[Blindspot]:
    """Rebuild blindspots from the persisted clustering (no re-embedding).

    Lets the dashboard exporter surface blindspots without importing sklearn or
    recomputing embeddings — it just reads the stored assignment. Same
    pair-restricted arithmetic as :func:`detect_blindspots`, and the same
    coverage-not-source rule for collapsed near-duplicates; see its docstring for
    both.
    """
    pair = list(members)
    if len(pair) != 2:
        raise ValueError(f"blindspots need exactly two personas, got {pair}")
    a, b = pair
    collapsed = store.duplicate_coverage()
    out: list[Blindspot] = []
    for row in store.cluster_rows():
        cid = row["cluster_id"]
        if cid == -1:
            continue
        cluster = store.cluster_members(cid)
        carried = {m["id"]: {m["source_id"]} | collapsed.get(m["id"], set()) for m in cluster}
        in_a = [m for m in cluster if carried[m["id"]] & members[a]]
        in_b = [m for m in cluster if carried[m["id"]] & members[b]]
        counts = {a: len(in_a), b: len(in_b)}
        size = len(in_a) + len(in_b)
        if size < min_size:
            continue
        dominant, dominant_members = (a, in_a) if len(in_a) >= len(in_b) else (b, in_b)
        share = len(dominant_members) / size
        if share < dominance:
            continue
        rep = _representative([m["title"] for m in dominant_members])
        out.append(
            Blindspot(
                cluster_id=cid,
                label=row["label"] or "(untitled cluster)",
                counts=counts,
                dominant_diet=dominant,
                other_diet=b if dominant == a else a,
                dominant_share=share,
                size=size,
                cluster_size=len(cluster),
                representative_titles=rep[:REPRESENTATIVE_LIMIT],
            )
        )
    out.sort(key=lambda b: (b.size, b.dominant_share), reverse=True)
    return out


def articles_from_store(
    store: Datastore, blindspots: list[Blindspot], members: dict[str, set[str]]
) -> dict[int, list[Article]]:
    """The dominant persona's articles per blindspot cluster, with outlet and link.

    Only the dominant persona's: the card's claim is "these outlets carried it and
    yours did not", so listing the one or two articles from the other side
    under that heading would contradict the sentence above them.

    Collapsed near-duplicates are listed alongside the canonical copy. They are
    real articles at real URLs, and the outlet list is the part of a blindspot card
    that makes it checkable — a card claiming three outlets carried a story is
    worth more than the same card naming one because deduplication hid the rest.
    """
    outlets = store.source_labels()
    out: dict[int, list[Article]] = {}
    for spot in blindspots:
        sources = members.get(spot.dominant_diet, set())
        cluster = store.cluster_members(spot.cluster_id)
        collapsed = store.duplicates_of(m["id"] for m in cluster)
        articles: list[Article] = []
        for member in cluster:
            for row in (member, *collapsed.get(member["id"], ())):
                if row["source_id"] not in sources:
                    continue
                title = clean_title(row["title"])
                if not title:
                    continue
                articles.append(
                    Article(
                        doc_id=row["id"],
                        title=title,
                        url=row["url"],
                        outlet=outlets.get(row["source_id"], ""),
                        # Carried so a story is still attributable in a store ingested
                        # before outlet names were recorded: the key de-slugs into a
                        # recognizable masthead, and no outlet at all is the one thing
                        # that would make the story uncheckable.
                        source_id=row["source_id"] or "",
                    )
                )
        if articles:
            out[spot.cluster_id] = articles
    return out


def themes_from_store(
    store: Datastore,
    members: dict[str, set[str]],
    blindspots: list[Blindspot] | None = None,
) -> list[Theme]:
    """Themed blindspots from the datastore, for a surface that only reads.

    Uses the assignments the cluster run persisted. When there are none — an
    older datastore, or a run that predates theming — it themes from the stored
    headlines with the taxonomy instead of returning nothing, so the dashboard
    and the email never fall back to the unreadable cluster labels.
    """
    spots = blindspots if blindspots is not None else blindspots_from_store(store, members)
    stored = {
        row["document_id"]: ThemeAssignment(row["theme_key"], row["theme_title"], row["method"])
        for row in store.blindspot_theme_rows()
    }
    return group_blindspots(spots, stored, articles_from_store(store, spots, members))


def run_clustering(
    store: Datastore,
    members: dict[str, set[str]],
    min_cluster_size: int = 2,
    dominance: float = 0.75,
    min_blindspot_size: int = 2,
    theme_model: str | None = None,
    theme_client: object | None = None,
    claude_themes: bool = True,
    theme_effort: str | None | _Unset = UNSET,
) -> ClusteringOutcome:
    """Cluster, persist the assignment, detect blindspots, and theme them.

    ``members`` is ``{persona_id: {source_id, ...}}`` for the reference pair.
    Clustering itself is persona-agnostic — every document is embedded once and
    clustered once — but the asymmetry the blindspots measure is a statement
    about two specific personas, so the pair has to be named here.
    """
    result = compute_clustering(store, min_cluster_size=min_cluster_size)

    # `by_cluster`, not `members`: the parameter above is persona membership, and
    # a local called `members` shadowed it here — so the blindspot detector was
    # handed a map of cluster ids and reported "needs exactly two personas".
    by_cluster = _members_by_cluster(result)
    cluster_titles = {cid: [result.titles[i] for i in idxs] for cid, idxs in by_cluster.items()}
    labels = label_clusters(cluster_titles)
    cluster_rows = [
        (cid, labels.get(cid) or "(untitled cluster)", len(idxs))
        for cid, idxs in by_cluster.items()
    ]
    assignments = [
        (result.doc_ids[i], result.labels[i])
        for i in range(result.n_docs)
        if result.labels[i] != -1
    ]
    store.replace_clustering(cluster_rows, assignments)

    blindspots = detect_blindspots(result, members, dominance, min_blindspot_size, labels=labels)
    themes = _theme_blindspots(
        store, blindspots, members, theme_model, theme_client, claude_themes, theme_effort
    )
    n_noise = sum(1 for lbl in result.labels if lbl == -1)
    return ClusteringOutcome(
        n_docs=result.n_docs,
        n_clusters=len(by_cluster),
        n_noise=n_noise,
        blindspots=blindspots,
        personas=sorted(members),
        themes=themes,
    )


def _theme_blindspots(
    store: Datastore,
    blindspots: list[Blindspot],
    members: dict[str, set[str]],
    theme_model: str | None,
    theme_client: object | None,
    claude_themes: bool,
    theme_effort: str | None | _Unset = UNSET,
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
    articles = articles_from_store(store, blindspots, members)
    # One entry per article, because the subject is a property of the headline
    # and not of the cluster it landed in.
    entries = [(a.doc_id, [a.title]) for group in articles.values() for a in group]
    kwargs = {"use_claude": claude_themes}
    if theme_model:
        kwargs["model"] = theme_model
    # `is not UNSET`, not truthiness: an unconfigured effort has to leave
    # `assign_themes`' own default in place, while a configured `None` has to
    # reach it, because that is what sends no effort at all. Truthiness maps
    # both to "leave it alone" and loses the second.
    if theme_effort is not UNSET:
        kwargs["effort"] = theme_effort
    assignments = assign_themes(entries, client=theme_client, **kwargs)
    store.replace_blindspot_themes(
        [(doc_id, a.key, a.title, a.method) for doc_id, a in assignments.items()]
    )
    return group_blindspots(blindspots, assignments, articles)
