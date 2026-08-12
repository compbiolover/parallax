"""Per-diet, per-foundation confidence bands from tagger disagreement.

The ensemble's §5 signal — *when the dictionary and the transformer disagree,
trust the number less* — surfaced at the diet-aggregate level the dashboard
shows. For each diet and foundation we compute:

- two composition estimates, one from the dictionary and one from the transformer
  (each normalized to sum to 1 across the five foundations);
- a **point** estimate (their mean) and a **band** spanning ``[min, max]`` of the
  two — a wide band means the two methods place very different emphasis there;
- a **disagreement share**: the fraction of the diet's documents where the two
  taggers split on that foundation's *presence*, using the exact vote convention
  the validated ensemble uses (dictionary present iff its rate > 0; transformer
  present iff P(present) > 0.5). This is the calibrated confidence signal
  (`scoring/ensemble.py`), now aggregated per foundation.

Both compositions are built over the **same** document set — only the documents
scored by *both* taggers. That matters once GDELT backfill is mixed in: backfill
writes dictionary rows but no transformer rows, so aggregating each tagger over
"all its rows" would compare different populations and the band would conflate
population differences with method disagreement (the very thing it claims to
isolate). Pairing keeps the band an apples-to-apples method comparison.

The band is honest about method disagreement, not a statistical CI — see
``LIMITATIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from scoring.aggregate import to_composition
from scoring.foundations import CLASSIC_FOUNDATIONS


@dataclass(frozen=True)
class FoundationBand:
    point: float  # ensemble emphasis (mean of the two compositions)
    low: float  # min(dictionary, transformer)
    high: float  # max(dictionary, transformer)
    dictionary: float  # dictionary composition share
    transformer: float  # transformer composition share
    disagreement: float  # fraction of docs where the taggers split on presence


def _weighted_composition(pairs, index: int, foundations) -> dict[str, float]:
    """Reach-weighted mean of one tagger's per-foundation scores across the paired
    documents (``index`` 0 = dictionary map, 1 = transformer map), normalized to a
    composition summing to 1."""
    totals = dict.fromkeys(foundations, 0.0)
    weight_sum = 0.0
    for pair in pairs:
        w, scores = pair[0], pair[1 + index]
        weight_sum += w
        for f in foundations:
            totals[f] += w * float(scores.get(f, 0.0) or 0.0)
    profile = totals if weight_sum == 0 else {f: totals[f] / weight_sum for f in foundations}
    return to_composition(profile, foundations)


def _disagreement_shares(pairs, foundations) -> dict[str, float]:
    """Per foundation, the fraction of documents where the dictionary's presence
    vote (rate > 0) differs from the transformer's (P(present) > 0.5)."""
    if not pairs:
        return dict.fromkeys(foundations, 0.0)
    split = dict.fromkeys(foundations, 0)
    for _w, dict_scores, trans_scores in pairs:
        for f in foundations:
            dict_vote = 1 if dict_scores.get(f, 0.0) > 0 else 0
            trans_vote = 1 if trans_scores.get(f, 0.0) > 0.5 else 0
            if dict_vote != trans_vote:
                split[f] += 1
    n = len(pairs)
    return {f: split[f] / n for f in foundations}


def persona_band(
    store,
    weights: dict[str, float],
    transformer_scorer: str,
    dict_scorer: str = "dictionary",
    foundations=CLASSIC_FOUNDATIONS,
) -> dict[str, FoundationBand] | None:
    """Confidence band per foundation for one persona, or ``None`` if it has
    no documents scored by both taggers.

    Both compositions and the disagreement share are computed over the *same*
    paired document set (docs scored by both taggers), so the band reflects method
    disagreement rather than a difference in which documents each tagger saw.
    """
    rows = store.paired_scores_for_sources(
        weights, dict_scorer, transformer_scorer, list(foundations)
    )
    pairs = [(weights.get(source_id, 0.0), a, b) for source_id, a, b in rows]
    if not pairs:
        return None

    dict_comp = _weighted_composition(pairs, 0, foundations)
    trans_comp = _weighted_composition(pairs, 1, foundations)
    disagreement = _disagreement_shares(pairs, foundations)

    bands: dict[str, FoundationBand] = {}
    for f in foundations:
        d, t = dict_comp[f], trans_comp[f]
        bands[f] = FoundationBand(
            point=(d + t) / 2,
            low=min(d, t),
            high=max(d, t),
            dictionary=d,
            transformer=t,
            disagreement=disagreement[f],
        )
    return bands


def all_persona_bands(
    store, registry, transformer_scorer: str, dict_scorer: str = "dictionary"
) -> dict[str, dict]:
    """Bands for every persona that has both taggers' scores. Empty when the
    transformer never ran (dashboard then falls back to the dictionary profile)."""
    out: dict[str, dict] = {}
    for persona_id in registry.persona_ids():
        band = persona_band(
            store, registry.weights_for(persona_id), transformer_scorer, dict_scorer
        )
        if band is not None:
            out[persona_id] = band
    return out
