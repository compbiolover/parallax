"""Compare how much each diet engages the liberty/oppression foundation.

Liberty is reported on its own rather than as a sixth spoke on the radar, for a
measurement reason rather than a presentational one. The radar shows a
*composition* — five shares summing to 1 over documents every tagger saw. Liberty
is scored by Claude on feed-ingested documents only, and only when an API key is
set, so its document population is a subset of the radar's. Folding a
partial-coverage foundation into a composition would change every other share as
a side effect of coverage, and would silently move the headline divergence number
that ``compare/history.py`` has been recording since the series began.

So the classic five stay the composition and the headline, and liberty is
reported as an absolute mean over the documents that carry it, with coverage
attached. That keeps the recorded history comparable and keeps a coverage
artifact from reading as a finding.

**Reading the numbers.** Liberty presence is scored 0-1 per document, and the
rubric instructs conservatism — most news does not substantially engage liberty,
so diet-level means in the low tenths are expected. What matters is the
difference between diets, and whether it survives the coverage caveat.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# Below this share of a diet's documents, the mean is reported but flagged as
# too thin to read. A round number, not a validated threshold.
LOW_COVERAGE = 0.10

# The rubric treats this as the line between incidental and live moral framing.
SALIENT = 0.5


@dataclass(frozen=True)
class LibertyProfile:
    """One diet's liberty engagement over the documents that were tagged."""

    mean: float  # reach-weighted mean presence, [0, 1]
    salient_share: float  # fraction of tagged docs scoring above SALIENT
    docs_scored: int
    docs_total: int

    @property
    def coverage(self) -> float:
        return self.docs_scored / self.docs_total if self.docs_total else 0.0

    @property
    def thin(self) -> bool:
        return self.docs_scored == 0 or self.coverage < LOW_COVERAGE


def persona_liberty_profile(store, weights: dict[str, float], scorer: str) -> LibertyProfile:
    """Reach-weighted liberty engagement for one persona.

    ``weights`` is ``{source_id: weight}`` from ``Registry.weights_for``."""
    scored, total = store.liberty_for_sources(weights, scorer)
    rows = [(weights.get(source_id, 0.0), value) for source_id, value in scored]
    if not rows:
        return LibertyProfile(0.0, 0.0, 0, total)
    weight_sum = sum(w for w, _ in rows)
    mean = (
        sum(w * v for w, v in rows) / weight_sum
        if weight_sum
        else statistics.mean(v for _, v in rows)
    )
    salient = sum(1 for _w, v in rows if v > SALIENT) / len(rows)
    return LibertyProfile(mean, salient, len(rows), total)


def all_persona_liberty(store, registry, scorer: str) -> dict[str, LibertyProfile]:
    """Profiles for every persona with any document in the corpus."""
    out: dict[str, LibertyProfile] = {}
    for persona_id in registry.persona_ids():
        profile = persona_liberty_profile(store, registry.weights_for(persona_id), scorer)
        if profile.docs_total:
            out[persona_id] = profile
    return out


def gap(profiles: dict[str, LibertyProfile], pair) -> dict | None:
    """Difference in mean liberty across the reference pair, or ``None``.

    Positive means ``pair.mine`` engages liberty more. Oriented rather than
    alphabetical, so the sign means something. ``thin`` propagates — a gap drawn
    from a thin profile is itself thin.
    """
    mine, theirs = pair.mine, pair.theirs
    if mine not in profiles or theirs not in profiles:
        return None
    return {
        "pair": [mine, theirs],
        "mean_gap": profiles[mine].mean - profiles[theirs].mean,
        "thin": profiles[mine].thin or profiles[theirs].thin,
    }
