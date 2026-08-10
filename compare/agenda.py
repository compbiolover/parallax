"""How differently the two diets *allocate attention* — the agenda, not the tone.

The headline Jensen-Shannon divergence in ``compare/divergence.py`` compares
two five-number foundation compositions. Those numbers come from averaging
hundreds of documents, and averages of hundreds of documents converge: on a
real corpus the two diets land within a few thousandths of each other, and the
number reads as "these diets are nearly identical" when the reader can see with
their own eyes that they are not.

Both things are true, and they are answers to different questions:

* **Foundation divergence** asks *what moral vocabulary does each diet speak?*
  A small number there is a real finding — care and fairness lead in both, and
  the binding foundations are present in both. It is evidence against the
  strong form of the liberal/conservative asymmetry hypothesis, which
  ``CLAUDE.md`` §5 says this tool is supposed to test rather than assume.
* **Agenda divergence** asks *what did each diet spend the day on?* That is
  where two readers end up in different worlds: not by moralizing the same
  events differently, but by reading about different events.

This module measures the second. Its unit is the story cluster the blindspot
engine already produces — each diet has a distribution of attention across
clusters, and the same Jensen-Shannon machinery applies to those distributions.
Nothing here re-reads text or calls a model; it is arithmetic over the stored
clustering.

Three numbers, in increasing order of how blunt they are:

``divergence``
    JSD between the two attention distributions, base 2, squared to a
    divergence on [0, 1] — the same convention as the foundation number, so the
    two are directly comparable and the contrast is the point.
``exclusive``
    Per diet, the share of its clustered articles that sit in stories the other
    diet never touched at all. The number a person feels: "two thirds of what I
    read had no counterpart on the other side."
``overlap``
    The share of stories both diets covered. Small by construction on a corpus
    this size, and worth printing precisely because it is small.

Noise (cluster -1) is excluded throughout: an unclustered document is one the
engine could not place, not a story either diet chose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compare.divergence import jensen_shannon_divergence
from ingestion.datastore import Datastore

# Below this many clustered articles a diet's attention distribution is a
# handful of stories and the divergence is mostly sampling noise. The rest of
# the codebase reports thin coverage rather than hiding it, so this does too.
THIN_ARTICLES = 30


@dataclass
class AgendaComparison:
    """Attention divergence between two diets, over story clusters."""

    pair: tuple[str, str]
    divergence: float
    exclusive: dict[str, float] = field(default_factory=dict)
    exclusive_stories: dict[str, int] = field(default_factory=dict)
    articles: dict[str, int] = field(default_factory=dict)
    shared_stories: int = 0
    total_stories: int = 0
    thin: bool = False

    @property
    def overlap(self) -> float:
        """Share of stories both diets covered."""
        return self.shared_stories / self.total_stories if self.total_stories else 0.0

    def to_dict(self) -> dict:
        return {
            "pair": list(self.pair),
            "divergence": self.divergence,
            "exclusive": dict(self.exclusive),
            "exclusive_stories": dict(self.exclusive_stories),
            "articles": dict(self.articles),
            "shared_stories": self.shared_stories,
            "total_stories": self.total_stories,
            "overlap": self.overlap,
            "thin": self.thin,
        }


def persona_cluster_counts(store: Datastore, registry) -> dict[str, dict[int, int]]:
    """``{persona: {cluster_id: article count}}``.

    Built by crossing the per-source cluster counts with each persona's
    membership, because a source several personas read contributes its articles to
    every one of them. A ``GROUP BY diet_id`` cannot express that, which is why
    the store now groups by source.
    """
    per_source: dict[str, dict[int, int]] = {}
    for row in store.cluster_source_counts():
        per_source.setdefault(row["source_id"], {})[row["cluster_id"]] = row["n"]
    counts: dict[str, dict[int, int]] = {}
    for persona_id in registry.persona_ids():
        totals: dict[int, int] = {}
        for source_id in registry.weights_for(persona_id):
            for cluster_id, n in per_source.get(source_id, {}).items():
                totals[cluster_id] = totals.get(cluster_id, 0) + n
        if totals:
            counts[persona_id] = totals
    return counts


def attention_shares(store: Datastore, registry) -> dict[str, dict[int, float]]:
    """``{persona: {cluster_id: share of that persona's clustered articles}}``.

    Shares rather than counts, because personas do not consume the same volume
    and a raw-count comparison would measure the source registry rather than the
    agenda.
    """
    counts = persona_cluster_counts(store, registry)
    shares: dict[str, dict[int, float]] = {}
    for persona_id, per_cluster in counts.items():
        total = sum(per_cluster.values())
        if total:
            shares[persona_id] = {c: n / total for c, n in per_cluster.items()}
    return shares


def compare_agendas(store: Datastore, registry, pair) -> AgendaComparison | None:
    """Agenda divergence across the reference pair, or ``None`` if unclusterable.

    ``None`` rather than a zero: no clustering and "identical agendas" are not
    the same statement, and the surfaces render an absent metric as silence.

    A story carried by a source *both* personas read counts for both, so it is
    shared rather than exclusive. That is the correct reading and a consequence of
    the shared catalog worth stating: overlap between two personas is partly a
    fact about how much of their source lists coincide.
    """
    counts = persona_cluster_counts(store, registry)
    a, b = pair.mine, pair.theirs
    if a not in counts or b not in counts:
        return None

    shares = attention_shares(store, registry)
    clusters = sorted(set(shares[a]) | set(shares[b]))
    # The same JSD the foundation number uses, over cluster ids instead of
    # foundation names — so the two divergences are on one scale and the
    # contrast between them is readable. The key list is the *union*, so a
    # story only one diet touched counts as a place they differ rather than
    # being dropped from the comparison.
    keys = [str(c) for c in clusters]
    left = {str(c): shares[a].get(c, 0.0) for c in clusters}
    right = {str(c): shares[b].get(c, 0.0) for c in clusters}

    exclusive: dict[str, float] = {}
    exclusive_stories: dict[str, int] = {}
    for persona_id, other_id in ((a, b), (b, a)):
        ours, others = counts[persona_id], counts[other_id]
        alone = [c for c in ours if not others.get(c)]
        exclusive_stories[persona_id] = len(alone)
        total = sum(ours.values())
        exclusive[persona_id] = sum(ours[c] for c in alone) / total if total else 0.0

    shared = len(set(counts[a]) & set(counts[b]))
    articles = {d: sum(counts[d].values()) for d in (a, b)}
    return AgendaComparison(
        pair=(a, b),
        divergence=jensen_shannon_divergence(left, right, foundations=keys),
        exclusive=exclusive,
        exclusive_stories=exclusive_stories,
        articles=articles,
        shared_stories=shared,
        total_stories=len(clusters),
        thin=min(articles.values()) < THIN_ARTICLES,
    )
