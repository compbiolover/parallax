"""The reference pair: which two personas every headline number is about.

With two diets, "the other diet" needed no definition and the code did not give
it one — it took ``sorted(ids)[:2]`` in seven places. That was already fragile
(the pair was an alphabetical accident, and ``modeled_ce`` sorts before ``self``,
so the per-foundation log-ratios came out with the sign inverted relative to
``CLAUDE.md`` §3(5), which asks for "positive = your diet over-indexes"). With a
library of personas it stops working entirely: adding one would silently change
the headline number and the recorded series.

So the pair is named, and it is oriented: ``mine`` first, ``theirs`` second. The
orientation is not cosmetic. It fixes the log-ratio sign to what the spec asks
for, and it lets every surface colour by role in the comparison instead of by
position in a list — which is the bug ``digest/render.py`` documents, where one
diet came out blue in one panel and orange in another.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The two personas Parallax has always compared. Defaulting to them is what keeps
# the recorded divergence series continuous across the persona migration: the ids
# are unchanged, so every snapshot ever recorded is still about this pair.
DEFAULT_MINE = "self"
DEFAULT_THEIRS = "modeled_ce"


@dataclass(frozen=True)
class ReferencePair:
    """The oriented pair. ``mine`` is the reader's own side."""

    mine: str
    theirs: str

    def __iter__(self):
        yield self.mine
        yield self.theirs

    @property
    def ids(self) -> tuple[str, str]:
        return (self.mine, self.theirs)

    def other(self, persona_id: str) -> str:
        """The opposite member. Raises for a persona outside the pair, because
        "the other diet" is only meaningful within it — the old
        ``next(d for d in diets if d != dominant)`` returned an arbitrary third
        party as soon as a third persona existed."""
        if persona_id == self.mine:
            return self.theirs
        if persona_id == self.theirs:
            return self.mine
        raise KeyError(f"{persona_id} is not in the reference pair {self.ids}")

    def as_list(self) -> list[str]:
        return [self.mine, self.theirs]


def resolve(
    settings: dict[str, Any] | None = None,
    mine: str | None = None,
    theirs: str | None = None,
    available: list[str] | None = None,
    families: dict[str, list[str]] | None = None,
) -> ReferencePair:
    """Resolve the pair: explicit argument, then settings, then the default.

    An unknown persona degrades with a logged reason rather than failing the run,
    in the same style as the rest of the pipeline: it falls back to the first
    persona of each family when families are known, and to the first two
    available ids otherwise. A run that produces a slightly differently-scoped
    number is more useful at 6am than a run that produced nothing.
    """
    cfg = ((settings or {}).get("compare") or {}).get("reference_pair") or {}
    chosen_mine = mine or cfg.get("mine") or DEFAULT_MINE
    chosen_theirs = theirs or cfg.get("theirs") or DEFAULT_THEIRS

    if available is None:
        return ReferencePair(chosen_mine, chosen_theirs)

    known = set(available)
    missing = [p for p in (chosen_mine, chosen_theirs) if p not in known]
    if not missing:
        return ReferencePair(chosen_mine, chosen_theirs)

    fallback = _fallback(available, families)
    logger.warning(
        "reference pair names unknown persona(s) %s; falling back to %s vs %s",
        ", ".join(missing), fallback.mine, fallback.theirs,
    )
    return fallback


def _fallback(available: list[str], families: dict[str, list[str]] | None) -> ReferencePair:
    if families:
        # One persona from each of the first two families, so the fallback still
        # compares across sides rather than two variants of the same one.
        sides = [ids for ids in families.values() if ids]
        if len(sides) >= 2:
            return ReferencePair(sides[0][0], sides[1][0])
    if len(available) >= 2:
        return ReferencePair(available[0], available[1])
    only = available[0] if available else DEFAULT_MINE
    return ReferencePair(only, only)
