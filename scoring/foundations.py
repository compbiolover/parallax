"""Canonical moral-foundations vocabulary shared across the pipeline.

The six foundations Parallax models. The dictionary tagger covers only the five
CLASSIC foundations — it has no signal for liberty/oppression (that is supplied
by the Claude tagger in a later phase). Keeping the two sets explicit here means
downstream code never silently treats a missing liberty score as a zero.
"""

from __future__ import annotations

# Full modeled set, in canonical order. Split-fairness (equality vs
# proportionality) is a later refinement and is not encoded here yet.
FOUNDATIONS: tuple[str, ...] = (
    "care",
    "fairness",
    "loyalty",
    "authority",
    "sanctity",
    "liberty",
)

# What the dictionary baseline can actually score (eMFD covers 5, no liberty).
CLASSIC_FOUNDATIONS: tuple[str, ...] = (
    "care",
    "fairness",
    "loyalty",
    "authority",
    "sanctity",
)

# Vice pole is tracked via sentiment sign rather than as separate labels here;
# the virtue/vice split (up to 12 labels) is a Phase 3 concern.
