"""Story threads: the identity a running news story keeps across days.

A cluster is not a story thread. ``clusters.cluster_id`` is a position in one
HDBSCAN run, so cluster 3 is about something different every morning, and
``replace_clustering`` deletes the lot on the next run. That is fine for
"what are today's stories" and useless for "what has happened in the French
protests since the 14th", which is the question a timeline answers.

So each run matches today's clusters onto persisted threads by embedding
centroid, in the embedder's own space (see ``ClusterResult.vectors`` for why
not the reduced one). Above :data:`SIMILARITY_THRESHOLD` a cluster joins the
thread it is closest to; below it, the cluster starts a thread of its own.

Two deliberate simplifications, both of which can bite:

* **Several clusters may join one thread.** A story that fragments into "the
  protests" and "the pension vote" lands in one thread rather than two, which
  is usually what a reader wants and is occasionally a merge they would not
  have made.
* **Threads are never merged or split retroactively.** Two threads that
  converge on the same story stay two rows, each with its own timeline. The
  alternative is rewriting history on every run, and a timeline that changes
  shape overnight is worse than one that is occasionally duplicated.

shortcut: no merge/split of existing threads, and one global similarity
threshold rather than one tuned per theme. Upgrade when the thread list shows
visible duplicates on a 20-item spot check, which is the same bar CLAUDE.md
§4 sets for the clustering itself.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Cosine similarity above which a cluster is "the same story" as a thread.
# 0.62 on `thenlper/gte-small`: at 0.5 unrelated politics stories merged into
# one thread, and at 0.75 the same protest story started a new thread every
# second day as the coverage moved on to a new angle. It is a judgement about
# the embedder, so re-check it if the embedder changes.
SIMILARITY_THRESHOLD = 0.62

# How much of the old centroid survives an update. A thread has to be allowed
# to drift — a story's coverage on day ten is not about what day one was about
# — but a centroid that simply becomes today's cluster forgets the story and
# will match anything tomorrow's coverage drifts into.
CENTROID_MOMENTUM = 0.7

# Cap on a thread's dormancy. A story nothing has covered for this long is over;
# matching a new cluster onto it would resurrect a finished thread and put a
# two-week hole in the middle of its timeline.
MAX_DORMANT_DAYS = 10


@dataclass(frozen=True)
class ThreadMatch:
    thread_id: str
    title: str
    similarity: float
    is_new: bool


def normalize(vec: list[float]) -> list[float]:
    """L2-normalize, so a dot product is a cosine."""
    norm = math.sqrt(sum(x * x for x in vec))
    if not norm:
        return list(vec)
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine of two vectors. 0.0 when either is degenerate or shaped wrong.

    Mismatched dimensions return 0 rather than raising: a datastore that has
    seen two embedders holds centroids of two widths, and one stale thread
    should not abort the run. ``track`` filters by embedder for the same
    reason, so this is the second line of defence.
    """
    if len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def centroid(vectors: list[list[float]]) -> list[float]:
    """Mean of the vectors, L2-normalized. ``[]`` for no input."""
    if not vectors:
        return []
    dim = len(vectors[0])
    totals = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:  # defensive: compute_clustering already rejects this
            continue
        for i, x in enumerate(vec):
            totals[i] += x
    return normalize([x / len(vectors) for x in totals])


def cluster_centroids(result) -> dict[int, list[float]]:
    """One centroid per non-noise cluster, in the embedder's own space."""
    if not result.vectors:
        return {}
    grouped: dict[int, list[list[float]]] = {}
    for i, label in enumerate(result.labels):
        if label == -1:
            continue
        grouped.setdefault(label, []).append(result.vectors[i])
    return {cid: centroid(vecs) for cid, vecs in grouped.items() if vecs}


def slugify(title: str, taken: set[str]) -> str:
    """A readable, stable thread id from a cluster label.

    Readable because these ids end up in a prompt, in parsed headings, and in
    anything debugging a timeline by hand; a uuid would make all three harder
    to follow for no gain.
    """
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    base = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")[:48] or "thread"
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def _days_between(a: str, b: str) -> int:
    """Whole days from ISO date ``a`` to ISO date ``b``; 0 if either is unusable."""
    from datetime import date

    try:
        return (date.fromisoformat(b) - date.fromisoformat(a)).days
    except ValueError:
        return 0


def match(
    centroids: dict[int, list[float]],
    threads: list[dict],
    day: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict[int, ThreadMatch]:
    """Map each cluster id onto a thread, existing or new.

    ``threads`` is ``[{thread_id, title, centroid, last_seen}, ...]`` with the
    centroid already decoded. A cluster joins the thread it is *most* similar
    to, which is why the candidates are sorted: scanning in cluster order would
    attach a cluster to whichever thread happened to come first in the list
    above the threshold. Two clusters may both join one thread, by design (see
    the module docstring); ``track`` sums them into one timeline point.
    """
    live = [
        t
        for t in threads
        if t.get("centroid") and _days_between(t.get("last_seen") or day, day) <= MAX_DORMANT_DAYS
    ]
    scored: list[tuple[float, int, str]] = []
    for cid, vec in centroids.items():
        for thread in live:
            sim = cosine(vec, thread["centroid"])
            if sim >= threshold:
                scored.append((sim, cid, thread["thread_id"]))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))

    out: dict[int, ThreadMatch] = {}
    titles = {t["thread_id"]: t.get("title") or "" for t in live}
    for sim, cid, thread_id in scored:
        if cid in out:
            continue
        out[cid] = ThreadMatch(thread_id, titles.get(thread_id, ""), sim, is_new=False)
    return out


def track(
    store,
    result,
    *,
    day: str,
    labels: dict[int, str],
    themes: dict[str, tuple[str, str]] | None = None,
    members: dict[str, set[str]] | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[ThreadMatch]:
    """Match today's clusters onto threads and persist one timeline point each.

    ``labels`` is ``{cluster_id: label}`` from the clustering, ``themes`` is
    ``{document_id: (theme_key, theme_title)}`` so a thread can report the
    motif it reads under, and ``members`` is ``{persona_id: {source_id}}`` for
    the per-diet counts on each timeline point.
    """
    centroids = cluster_centroids(result)
    if not centroids:
        return []
    embedder = store.get_meta("embedder") or ""

    existing = []
    # Every id in the table, not just the comparable ones: a slug is unique per
    # table, not per embedder. Taking `taken` from the filtered list let a new
    # thread pick a slug an older embedder's thread already held, and the
    # upsert then rewrote that row instead of adding one.
    all_ids = set()
    for row in store.story_thread_rows():
        all_ids.add(row["thread_id"])
        if row["embedder"] != embedder:
            continue  # a centroid from another embedder is not comparable
        try:
            vec = json.loads(row["centroid"])
        except (TypeError, ValueError):
            continue
        existing.append(
            {
                "thread_id": row["thread_id"],
                "title": row["title"],
                "centroid": vec,
                "last_seen": row["last_seen"],
            }
        )

    matched = match(centroids, existing, day, threshold)
    taken = all_ids | {m.thread_id for m in matched.values()}

    by_cluster: dict[int, list[int]] = {}
    for i, label in enumerate(result.labels):
        if label != -1:
            by_cluster.setdefault(label, []).append(i)

    # Cluster ids grouped by the thread they land in, because two clusters may
    # share one. Persisting per cluster instead wrote one timeline point per
    # cluster and the last one won, so a story covered in two clusters reported
    # the smaller count. Largest cluster first, so the biggest story gets the
    # readable slug when two would produce the same one, and so the thread's
    # title comes from its dominant cluster.
    order = sorted(by_cluster, key=lambda c: (-len(by_cluster[c]), c))
    per_thread: dict[str, list[int]] = {}
    is_new: dict[str, bool] = {}
    titles: dict[str, str] = {}
    sims: dict[str, float] = {}
    for cid in order:
        title = labels.get(cid) or "(untitled story)"
        hit = matched.get(cid)
        if hit is None:
            thread_id = slugify(title, taken)
            taken.add(thread_id)
            is_new.setdefault(thread_id, True)
            sims.setdefault(thread_id, 1.0)
        else:
            thread_id = hit.thread_id
            is_new.setdefault(thread_id, False)
            sims[thread_id] = max(sims.get(thread_id, 0.0), hit.similarity)
        per_thread.setdefault(thread_id, []).extend(by_cluster[cid])
        titles.setdefault(thread_id, title)

    assignments: list[tuple[str, str]] = []
    out: list[ThreadMatch] = []
    for thread_id, idxs in per_thread.items():
        fresh = centroid([result.vectors[i] for i in idxs])
        if is_new[thread_id]:
            new_centroid = fresh
        else:
            prior = next((t["centroid"] for t in existing if t["thread_id"] == thread_id), [])
            new_centroid = _blend(prior, fresh)

        theme_key = theme_title = None
        if themes:
            seen = [themes[result.doc_ids[i]] for i in idxs if result.doc_ids[i] in themes]
            if seen:
                # The commonest theme among the thread's documents, not the
                # first: a cluster is not always one subject (which is why
                # blindspot_themes is keyed per document), so the motif is a
                # vote rather than a lookup.
                counts: dict[tuple[str, str], int] = {}
                for pair in seen:
                    counts[pair] = counts.get(pair, 0) + 1
                theme_key, theme_title = max(counts, key=lambda k: (counts[k], k))

        diet_counts: dict[str, int] = {}
        if members:
            for persona_id, sources in members.items():
                n = sum(1 for i in idxs if result.coverage[i] & sources)
                if n:
                    diet_counts[persona_id] = n

        headline = next((result.titles[i] for i in idxs if result.titles[i]), None)
        store.upsert_story_thread(
            thread_id=thread_id,
            title=titles[thread_id],
            centroid=json.dumps([round(x, 6) for x in new_centroid]),
            dim=len(new_centroid),
            embedder=embedder,
            day=day,
            theme_key=theme_key,
            theme_title=theme_title,
        )
        store.record_thread_day(
            thread_id=thread_id,
            day=day,
            doc_count=len(idxs),
            diet_counts=json.dumps(diet_counts),
            headline=headline,
        )
        assignments.extend((result.doc_ids[i], thread_id) for i in idxs)
        out.append(ThreadMatch(thread_id, titles[thread_id], sims[thread_id], is_new[thread_id]))

    store.replace_thread_documents(assignments)
    carried = sum(1 for m in out if not m.is_new)
    logger.info("story threads: %d carried forward, %d new", carried, len(out) - carried)
    return out


def _blend(prior: list[float], fresh: list[float]) -> list[float]:
    """Move a thread's centroid toward today's coverage without forgetting it."""
    if not prior or len(prior) != len(fresh):
        return fresh
    m = CENTROID_MOMENTUM
    return normalize([m * a + (1.0 - m) * b for a, b in zip(prior, fresh, strict=True)])
