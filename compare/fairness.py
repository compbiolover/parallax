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


def diet_fairness_profile(
    store, diet_id: str, scorer: str = "dictionary"
) -> FairnessProfile:
    """Reach-weighted equality/proportionality shares for one diet."""
    rows, total = store.fairness_split_for_diet(diet_id, scorer)
    eq_sum = pr_sum = 0.0
    for weight, eq, pr in rows:
        eq_sum += weight * eq
        pr_sum += weight * pr

    mass = eq_sum + pr_sum
    if mass <= 0:
        # Documents were partitioned but carried no fairness mass to divide.
        return FairnessProfile(0.0, 0.0, len(rows), total)
    return FairnessProfile(eq_sum / mass, pr_sum / mass, len(rows), total)


def all_diet_fairness(store, scorer: str = "dictionary") -> dict[str, FairnessProfile]:
    """Profiles for every diet that has any document scored by ``scorer``."""
    out: dict[str, FairnessProfile] = {}
    for diet_id in store.diet_ids():
        profile = diet_fairness_profile(store, diet_id, scorer)
        if profile.docs_total:
            out[diet_id] = profile
    return out


def gap(profiles: dict[str, FairnessProfile]) -> dict | None:
    """The equality-share difference between the first two diets, or ``None``.

    Positive ``equality_gap`` means the first diet (alphabetically, matching the
    rest of the dashboard's pairing) leans more on equality than the second.
    ``thin`` propagates: a gap computed from a thin profile is itself thin.
    """
    ids = sorted(profiles)
    if len(ids) < 2:
        return None
    a, b = ids[:2]
    return {
        "pair": [a, b],
        "equality_gap": profiles[a].equality - profiles[b].equality,
        "thin": profiles[a].thin or profiles[b].thin,
    }
