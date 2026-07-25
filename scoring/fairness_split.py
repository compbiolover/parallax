"""Split the fairness foundation into equality and proportionality (MFQ-2).

In 2023 Atari, Haidt, Graham and colleagues split Fairness in two, on evidence
from 25 populations: **equality** (equal treatment and equal outcomes for
individuals) and **proportionality** (people rewarded in proportion to their
merit or contribution). The MFQ-2 measures six foundations on that basis — care,
equality, proportionality, loyalty, authority, purity — and drops fairness as a
single construct.

The distinction earns its place in this project. Both modeled diets talk about
fairness constantly, and a five-way tagger reports that as one number, which
hides the more interesting fact: they frequently mean different things by it.
The historical record makes the point — Haidt originally proposed Liberty partly
because economic conservatives objected that the five-foundation model captured
equality but not their notion of fairness, which is proportional.

**Why this module is a heuristic, and says so.**

No validated dictionary implements the split. The eMFD, the MFD, and MFD 2.0 all
carry a single ``fairness`` dimension, and the Moral Foundations Reddit Corpus
that Mformer is fine-tuned on does not split it either. The split is, so far, a
questionnaire-level construct. So there is nothing to load here — this module
ships a hand-built term list, in the same spirit as ``seed_lexicon.py`` and with
the same warning attached: it is a starting point for a measurement, not a
measurement.

**What it does, and deliberately does not do.**

It does not generate new fairness signal. The dictionary scorer decides how much
fairness a document contains; this module only decides how that existing mass
divides, from separate lexical evidence about *which kind* of fairness is under
discussion. When a document offers no such evidence, the split is ``None`` —
not 50/50, and not zero. Unsplit and evenly-split are different claims, and only
one of them is honest about ignorance.

Both halves carry virtue and vice terms. A list that coded one diet's fairness
vocabulary as virtuous and the other's as suspect would violate the project's
symmetry requirement while looking like measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_PREFIX_LEN = 4

# Hand-built, unvalidated. Stems of length >= MIN_PREFIX_LEN match by prefix, so
# "merit" catches "merits", "meritocracy", "meritocratic".
#
# EQUALITY — equal treatment, equal outcomes, and the vices of their absence:
# exclusion, disparity, unearned advantage.
EQUALITY_TERMS: tuple[str, ...] = (
    "equal", "equality", "equalit", "equitab", "equity", "egalitarian",
    "inequal", "unequal", "inequit", "disparit", "imbalance",
    "discriminat", "marginaliz", "underrepresent", "exclusion", "excluded",
    "segregat", "disenfranchis", "systemic", "structural",
    "inclusive", "inclusion", "access", "accessib", "universal",
    "civil right", "same right", "level playing", "fair share",
    "privileg", "advantag", "favoritism", "nepotism",
)

# PROPORTIONALITY — desert, merit, contribution, and the vices of their absence:
# reward without contribution, contribution without reward.
PROPORTIONALITY_TERMS: tuple[str, ...] = (
    "merit", "meritocra", "deserv", "undeserv", "desert",
    "earn", "unearn", "proportion", "disproportion",
    "contribut", "effort", "hardwork", "hard work", "work ethic", "diligen",
    "reward", "incentiv", "compensat", "repay", "payback", "recompens",
    "entitle", "handout", "freeload", "freerid", "free rid", "mooch",
    "accountab", "unaccountab", "responsib", "irresponsib",
    "consequence", "impunity", "reap", "sow",
    "productiv", "self-relian", "self relian", "bootstrap",
)


@dataclass(frozen=True)
class FairnessSplit:
    """How one document's fairness mass divides. Shares sum to 1."""

    equality: float
    proportionality: float
    equality_hits: int
    proportionality_hits: int

    @property
    def evidence(self) -> int:
        """Total matched split-terms behind this partition. Low means fragile."""
        return self.equality_hits + self.proportionality_hits

    @property
    def leans(self) -> str:
        """Which half dominates: 'equality', 'proportionality', or 'balanced'."""
        if self.equality > self.proportionality:
            return "equality"
        if self.proportionality > self.equality:
            return "proportionality"
        return "balanced"


class FairnessSplitter:
    """Partition a document's fairness score into equality vs proportionality.

    ``min_evidence`` is the number of matched split-terms required before a
    partition is reported at all. The default of 2 is deliberately not 1: a
    single stray "access" or "earn" is a coin flip dressed up as a measurement,
    and an unsplit document is a more useful output than a confident wrong one.
    Raise it for stricter partitions and lower coverage.
    """

    def __init__(
        self,
        equality_terms: tuple[str, ...] = EQUALITY_TERMS,
        proportionality_terms: tuple[str, ...] = PROPORTIONALITY_TERMS,
        min_evidence: int = 2,
    ) -> None:
        if min_evidence < 1:
            raise ValueError("min_evidence must be at least 1")
        self.min_evidence = min_evidence
        # (stem, side), longest first so "equalit" wins over "equal".
        self._stems: list[tuple[str, str]] = sorted(
            [(t.lower(), "equality") for t in equality_terms]
            + [(t.lower(), "proportionality") for t in proportionality_terms],
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        self._exact = {s: side for s, side in self._stems if len(s) < MIN_PREFIX_LEN}

    def _side(self, token: str) -> str | None:
        side = self._exact.get(token)
        if side is not None:
            return side
        for stem, stem_side in self._stems:
            if len(stem) >= MIN_PREFIX_LEN and token.startswith(stem):
                return stem_side
        return None

    def split(self, tokens: list[str]) -> FairnessSplit | None:
        """Partition from a document's tokens, or ``None`` on thin evidence.

        Multi-word stems in the term lists (e.g. "hard work") never match a
        single token; they are kept because they document intent and would match
        if this ever moved to n-grams.
        """
        eq = pr = 0
        for token in tokens:
            side = self._side(token)
            if side == "equality":
                eq += 1
            elif side == "proportionality":
                pr += 1

        total = eq + pr
        if total < self.min_evidence:
            return None
        return FairnessSplit(
            equality=eq / total,
            proportionality=pr / total,
            equality_hits=eq,
            proportionality_hits=pr,
        )


def apply_split(fairness_rate: float, split: FairnessSplit | None) -> dict[str, float | None]:
    """Divide a fairness rate into its two halves.

    Returns ``None`` for both when there is no partition — the caller must keep
    that distinct from zero, exactly as it does for an unscored liberty.
    """
    if split is None:
        return {"equality": None, "proportionality": None}
    return {
        "equality": fairness_rate * split.equality,
        "proportionality": fairness_rate * split.proportionality,
    }
