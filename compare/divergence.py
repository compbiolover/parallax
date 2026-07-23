"""Divergence between two diet-level foundation profiles.

The headline metric is Jensen-Shannon divergence (``CLAUDE.md`` §3): base-2, so
bounded [0, 1], symmetric, and decomposable per foundation. ``scipy``'s
``jensenshannon`` returns the *distance* (the square root) — this module squares
it back to the divergence and also returns the interpretable per-foundation
log-ratios that carry the human-readable story.

Only the five classic foundations are compared here: the dictionary baseline
does not score liberty/oppression, so including it would compare a real number
against a structural zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.spatial.distance import jensenshannon

from scoring.foundations import CLASSIC_FOUNDATIONS


def jensen_shannon_divergence(
    p: dict[str, float],
    q: dict[str, float],
    foundations: Sequence[str] = CLASSIC_FOUNDATIONS,
) -> float:
    """Squared JS distance (base 2) between two compositions. Bounded [0, 1]."""
    pv = [p.get(f, 0.0) for f in foundations]
    qv = [q.get(f, 0.0) for f in foundations]
    dist = jensenshannon(pv, qv, base=2)
    if math.isnan(dist):  # both all-zero
        return 0.0
    return float(dist**2)


def log_ratios(
    p: dict[str, float],
    q: dict[str, float],
    foundations: Sequence[str] = CLASSIC_FOUNDATIONS,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Per-foundation ln(P_i / Q_i). Positive = P over-indexes vs Q.

    A small epsilon replaces zeros so the ratio stays finite (see the smoothed
    sanity-check note in ``CLAUDE.md`` §3).
    """
    return {
        f: math.log((p.get(f, 0.0) + epsilon) / (q.get(f, 0.0) + epsilon))
        for f in foundations
    }


def index_form(
    p: dict[str, float],
    q: dict[str, float],
    foundations: Sequence[str] = CLASSIC_FOUNDATIONS,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Per-foundation 100 x (P_i / Q_i); 100 = parity. Reads well on a dashboard."""
    return {
        f: 100.0 * (p.get(f, 0.0) + epsilon) / (q.get(f, 0.0) + epsilon)
        for f in foundations
    }
