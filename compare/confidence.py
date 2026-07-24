"""Per-diet, per-foundation confidence bands from tagger disagreement.

The ensemble's §5 signal — *when the dictionary and the transformer disagree,
trust the number less* — surfaced at the diet-aggregate level the dashboard
shows. For each diet and foundation we compute:

- two independent composition estimates, one from the dictionary and one from
  the transformer (each normalized to sum to 1 across the five foundations);
- a **point** estimate (their mean) and a **band** spanning ``[min, max]`` of the
  two — a wide band means the two methods place very different emphasis there;
- a **disagreement share**: the fraction of the diet's documents where the two
  taggers split on that foundation's *presence*, using the exact vote convention
  the validated ensemble uses (dictionary present iff its rate > 0; transformer
  present iff P(present) > 0.5). This is the calibrated confidence signal
  (`scoring/ensemble.py`), now aggregated per foundation.

The band is honest about method disagreement, not a statistical CI — see
``LIMITATIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from scoring.aggregate import to_composition
from scoring.foundations import CLASSIC_FOUNDATIONS


@dataclass(frozen=True)
class FoundationBand:
    point: float          # ensemble emphasis (mean of the two compositions)
    low: float            # min(dictionary, transformer)
    high: float           # max(dictionary, transformer)
    dictionary: float     # dictionary composition share
    transformer: float    # transformer composition share
    disagreement: float   # fraction of docs where the taggers split on presence


def _mean_profile(rows: list, foundations) -> dict[str, float]:
    """Mean per-foundation value across score rows (weighted by document weight).

    Rows come from ``scores_for_diet``, which always aliases ``d.weight``.
    """
    totals = dict.fromkeys(foundations, 0.0)
    weight_sum = 0.0
    for row in rows:
        w = float(row["weight"] or 1.0)
        weight_sum += w
        for f in foundations:
            totals[f] += w * float(row[f] or 0.0)
    if weight_sum == 0:
        return dict.fromkeys(foundations, 0.0)
    return {f: totals[f] / weight_sum for f in foundations}


def _disagreement_shares(pairs, foundations) -> dict[str, float]:
    """Per foundation, the fraction of documents where the dictionary's presence
    vote (rate > 0) differs from the transformer's (P(present) > 0.5)."""
    if not pairs:
        return dict.fromkeys(foundations, 0.0)
    split = dict.fromkeys(foundations, 0)
    for dict_scores, trans_scores in pairs:
        for f in foundations:
            dict_vote = 1 if dict_scores.get(f, 0.0) > 0 else 0
            trans_vote = 1 if trans_scores.get(f, 0.0) > 0.5 else 0
            if dict_vote != trans_vote:
                split[f] += 1
    n = len(pairs)
    return {f: split[f] / n for f in foundations}


def diet_band(
    store,
    diet_id: str,
    transformer_scorer: str,
    dict_scorer: str = "dictionary",
    foundations=CLASSIC_FOUNDATIONS,
) -> dict[str, FoundationBand] | None:
    """Confidence band per foundation for one diet, or ``None`` if the diet has
    no documents scored by both taggers."""
    dict_rows = store.scores_for_diet(diet_id, dict_scorer)
    trans_rows = store.scores_for_diet(diet_id, transformer_scorer)
    if not dict_rows or not trans_rows:
        return None

    dict_comp = to_composition(_mean_profile(dict_rows, foundations), foundations)
    trans_comp = to_composition(_mean_profile(trans_rows, foundations), foundations)
    pairs = store.paired_scores_for_diet(
        diet_id, dict_scorer, transformer_scorer, list(foundations)
    )
    disagreement = _disagreement_shares(pairs, foundations)

    bands: dict[str, FoundationBand] = {}
    for f in foundations:
        d, t = dict_comp[f], trans_comp[f]
        bands[f] = FoundationBand(
            point=(d + t) / 2, low=min(d, t), high=max(d, t),
            dictionary=d, transformer=t, disagreement=disagreement[f],
        )
    return bands


def all_diet_bands(
    store, transformer_scorer: str, dict_scorer: str = "dictionary"
) -> dict[str, dict]:
    """Bands for every diet that has both taggers' scores. Empty when the
    transformer never ran (dashboard then falls back to the dictionary profile)."""
    out: dict[str, dict] = {}
    for diet_id in store.diet_ids():
        band = diet_band(store, diet_id, transformer_scorer, dict_scorer)
        if band is not None:
            out[diet_id] = band
    return out
