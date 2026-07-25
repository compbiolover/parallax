"""Canonical moral-foundations vocabulary shared across the pipeline.

Parallax models six foundations, with fairness optionally split in two. Which
set a given piece of code should use depends on what can actually produce it:

- :data:`CLASSIC_FOUNDATIONS` (5) — what the dictionary baseline scores. eMFD,
  MFD, and MFD 2.0 all carry a single ``fairness`` dimension and no liberty.
  These are the datastore's core score columns and the basis of the headline
  divergence number.
- :data:`FOUNDATIONS` (6) — the full modeled set, adding liberty/oppression.
  Liberty is supplied by the Claude tagger; it stays ``None`` elsewhere so
  downstream code never reads "unscored" as "zero".
- :data:`MFQ2_FOUNDATIONS` (6) — the post-2023 taxonomy (Atari, Haidt, Graham
  et al.), which splits fairness into :data:`FAIRNESS_SUBFOUNDATIONS`. Note this
  set has no liberty: MFQ-2 does not include it.
- :data:`EXTENDED_FOUNDATIONS` (7) — MFQ-2 plus liberty. Parallax's widest view,
  and not something any single tagger produces on its own.

**On the split.** Equality is about equal treatment and equal outcomes;
proportionality is about people being rewarded in proportion to their merit or
contribution. The distinction matters here because it is exactly where the two
modeled diets are theorized to diverge inside a foundation that a five-way
tagger reports as one number — a diet can be intensely concerned with fairness
in a sense the other diet does not recognize as fairness at all.

The split is a refinement layer, not a replacement. The headline metrics stay on
:data:`CLASSIC_FOUNDATIONS` so that recorded history remains comparable and so
an unvalidated partition never silently becomes the number on the front page.
"""

from __future__ import annotations

# What the dictionary baseline can actually score (eMFD covers 5, no liberty).
CLASSIC_FOUNDATIONS: tuple[str, ...] = (
    "care",
    "fairness",
    "loyalty",
    "authority",
    "sanctity",
)

# Full modeled set, in canonical order — the classic five plus liberty.
FOUNDATIONS: tuple[str, ...] = (
    "care",
    "fairness",
    "loyalty",
    "authority",
    "sanctity",
    "liberty",
)

# The two halves fairness divides into under MFQ-2.
FAIRNESS_SUBFOUNDATIONS: tuple[str, ...] = ("equality", "proportionality")

# The post-2023 taxonomy: fairness replaced by its two halves. No liberty —
# MFQ-2 does not measure it.
MFQ2_FOUNDATIONS: tuple[str, ...] = (
    "care",
    "equality",
    "proportionality",
    "loyalty",
    "authority",
    "sanctity",
)

# Everything Parallax can model at once: MFQ-2 plus liberty.
EXTENDED_FOUNDATIONS: tuple[str, ...] = (*MFQ2_FOUNDATIONS, "liberty")

# One-line glosses, kept next to the vocabulary so the dashboard and any
# generated copy describe a foundation the same way the theory does.
FOUNDATION_GLOSS: dict[str, str] = {
    "care": "care for others; aversion to suffering and harm",
    "fairness": "fairness in general — equality and proportionality combined",
    "equality": "equal treatment and equal outcomes for individuals",
    "proportionality": "reward in proportion to merit, effort, or contribution",
    "loyalty": "commitment to one's group; aversion to betrayal",
    "authority": "respect for legitimate authority and tradition",
    "sanctity": "purity and the sacred; aversion to degradation",
    "liberty": "freedom from domination and coercion",
}

# Vice pole is tracked via sentiment sign rather than as separate labels here;
# the virtue/vice split (up to 12 labels) is a Phase 3 concern.
