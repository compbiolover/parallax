"""Aggregate per-document scores into a diet-level foundation profile.

Follows ``CLAUDE.md`` §3(b): documents are already length-normalized by the
scorer, so aggregation is a (optionally reach-weighted) mean per foundation,
followed by normalization to a composition that sums to 1 — the form the
divergence metrics in ``compare/`` consume.
"""

from __future__ import annotations

from collections.abc import Sequence

from .dictionary import DocumentScore
from .foundations import CLASSIC_FOUNDATIONS


def aggregate_profile(
    scores: Sequence[DocumentScore],
    weights: Sequence[float] | None = None,
    foundations: Sequence[str] = CLASSIC_FOUNDATIONS,
) -> dict[str, float]:
    """Weighted mean of per-document foundation rates.

    ``weights`` (e.g. source reach/consumption weights) must be parallel to
    ``scores``; when omitted every document is weighted equally. Returns the
    mean per-token rate per foundation — not yet normalized to a composition.
    """
    if weights is not None and len(weights) != len(scores):
        raise ValueError("weights must be parallel to scores")

    totals = dict.fromkeys(foundations, 0.0)
    weight_sum = 0.0
    for i, score in enumerate(scores):
        w = 1.0 if weights is None else float(weights[i])
        if w <= 0:
            continue
        weight_sum += w
        for f in foundations:
            totals[f] += w * score.foundations.get(f, 0.0)

    if weight_sum == 0:
        return dict.fromkeys(foundations, 0.0)
    return {f: totals[f] / weight_sum for f in foundations}


def to_composition(
    profile: dict[str, float],
    foundations: Sequence[str] = CLASSIC_FOUNDATIONS,
) -> dict[str, float]:
    """Normalize a profile to a composition summing to 1.

    An all-zero profile (no moral signal found) maps to a uniform distribution
    rather than dividing by zero — a defensible neutral prior for the divergence
    metrics downstream.
    """
    values = {f: max(0.0, profile.get(f, 0.0)) for f in foundations}
    total = sum(values.values())
    if total == 0:
        uniform = 1.0 / len(foundations)
        return {f: uniform for f in foundations}
    return {f: values[f] / total for f in foundations}
