"""Compare how two diets divide fairness between equality and proportionality.

The headline divergence treats fairness as one quantity, which can make two
diets look similar precisely where they disagree most. Both may talk about
fairness constantly and mean incompatible things by it: equal treatment and
equal outcomes on one side, reward proportional to merit and contribution on the
other. This module reports that division.

**Within-fairness shares, not overall emphasis.** The numbers here answer "of
the fairness this diet expresses, how much is equality and how much is
proportionality?" — not "how much does this diet care about fairness?", which
the radar already shows. Keeping the question inside fairness means a diet that
simply writes more does not read as more concerned with either half.

**Coverage is part of the result, not a footnote.** Only documents carrying
enough lexical evidence are partitioned at all, so every profile reports what
fraction of the diet's documents it rests on. A 70/30 split over 4% of documents
is not a finding, and the dashboard is expected to show the coverage next to the
shares rather than beneath them.

**This tests a hypothesis; it does not assume one.** The theory predicts the
author's diet over-indexes on equality and the modeled conservative-evangelical
diet on proportionality. That prediction is exactly what the numbers are for.
Two reasons for caution beyond the usual ones: the liberal/conservative
foundation asymmetry replicates poorly when measured in *language* specifically,
and proportionality is reported to bridge the political divide rather than
divide it — both sides value merit and effort. A null result here is a real
result, and a result matching the prediction is weak evidence given an
unvalidated partition (see ``scoring/fairness_split.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Below this share of documents, a split is reported but flagged as too thin to
# read. Chosen as a round number, not a validated threshold.
LOW_COVERAGE = 0.10


@dataclass(frozen=True)
class FairnessProfile:
    """How one diet divides its fairness mass."""

    equality: float           # share of fairness that is equality-framed, [0, 1]
    proportionality: float    # share that is proportionality-framed, [0, 1]
    docs_split: int           # documents that carried enough evidence to partition
    docs_total: int           # documents the scorer saw at all

    @property
    def coverage(self) -> float:
        """Fraction of the diet's documents behind these shares."""
        return self.docs_split / self.docs_total if self.docs_total else 0.0

    @property
    def thin(self) -> bool:
        """True when too few documents were partitioned to read the shares."""
        return self.coverage < LOW_COVERAGE or self.docs_split == 0

    @property
    def leans(self) -> str:
        if self.equality > self.proportionality:
            return "equality"
        if self.proportionality > self.equality:
            return "proportionality"
        return "balanced"


def persona_fairness_profile(
    store, weights: dict[str, float], scorer: str = "dictionary"
) -> FairnessProfile:
    """Reach-weighted equality/proportionality shares for one persona.

    ``weights`` is ``{source_id: weight}`` from ``Registry.weights_for`` — the
    persona's membership list and its weighting in one map."""
    rows, total = store.fairness_split_for_sources(weights, scorer)
    eq_sum = pr_sum = 0.0
    for source_id, eq, pr in rows:
        weight = weights.get(source_id, 0.0)
        eq_sum += weight * eq
        pr_sum += weight * pr

    mass = eq_sum + pr_sum
    if mass <= 0:
        # Documents were partitioned but carried no fairness mass to divide.
        return FairnessProfile(0.0, 0.0, len(rows), total)
    return FairnessProfile(eq_sum / mass, pr_sum / mass, len(rows), total)


def all_persona_fairness(store, registry, scorer: str = "dictionary") -> dict[str, FairnessProfile]:
    """Profiles for every persona that has any document scored by ``scorer``."""
    out: dict[str, FairnessProfile] = {}
    for persona_id in registry.persona_ids():
        profile = persona_fairness_profile(store, registry.weights_for(persona_id), scorer)
        if profile.docs_total:
            out[persona_id] = profile
    return out


def gap(profiles: dict[str, FairnessProfile], pair) -> dict | None:
    """The equality-share difference across the reference pair, or ``None``.

    Positive ``equality_gap`` means ``pair.mine`` leans more on equality than
    ``pair.theirs``. Oriented rather than alphabetical: the previous version took
    the first two ids in sorted order, which made the sign an accident of
    spelling. ``thin`` propagates: a gap computed from a thin profile is itself
    thin.
    """
    mine, theirs = pair.mine, pair.theirs
    if mine not in profiles or theirs not in profiles:
        return None
    return {
        "pair": [mine, theirs],
        "equality_gap": profiles[mine].equality - profiles[theirs].equality,
        "thin": profiles[mine].thin or profiles[theirs].thin,
    }
