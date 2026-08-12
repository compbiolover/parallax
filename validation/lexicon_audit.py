"""Audit a lexicon for asymmetry between the two halves of fairness.

Built after the seed lexicon was caught with a structural bias: its fairness
vocabulary contained equality terms and no proportionality terms whatsoever, so
a merit-framed argument scored as containing *no fairness at all*. That
systematically under-measured whichever diet frames fairness as proportion, and
nothing surfaced it until the MFQ-2 split forced the two senses apart. A word
list can encode a political asymmetry while looking like a neutral instrument,
so the check belongs in the repo rather than in one session's notes.

The audit answers two questions about any lexicon:

1. **Is either side missing entirely?** That is the categorical failure the seed
   had, and the one worth an alarm rather than a number.
2. **Per occurrence, how much fairness does each side's vocabulary actually
   contribute?** Under ``argmax`` (the default and the mode the eMFD requires) a
   word contributes its fairness weight only when fairness is its dominant
   foundation. Merit vocabulary that the lexicon assigns to authority or care
   yields nothing to fairness, however clearly it belongs there conceptually.

**What this does not measure.** It sees only the intersection of
``scoring/fairness_split.py``'s hand-built term lists with the lexicon's
vocabulary, which for the eMFD is a couple of dozen words. It therefore measures
*that intersection*, not "the lexicon's fairness vocabulary" in any absolute
sense, and both term lists are themselves unvalidated. Treat the yield ratio as
a description of a known, small sample, and re-run it whenever the term lists
change.

    python -m validation.lexicon_audit                             # built-in seed
    python -m validation.lexicon_audit --lexicon data/emfd_scoring.csv
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field

from scoring.fairness_split import FairnessSplitter
from scoring.foundations import FAIRNESS_SUBFOUNDATIONS

# A yield ratio outside this band is a lean worth stating out loud. It is a
# reporting threshold chosen for readability, not a validated cutoff.
BALANCED_BAND = (0.75, 1.33)


@dataclass
class SideAudit:
    """One half of fairness, as the lexicon actually represents it."""

    side: str
    words: list[str] = field(default_factory=list)
    to_fairness: int = 0  # words whose dominant foundation is fairness
    yields: list[float] = field(default_factory=list)  # fairness added per occurrence

    @property
    def n(self) -> int:
        return len(self.words)

    @property
    def mean_yield(self) -> float:
        return statistics.mean(self.yields) if self.yields else 0.0

    @property
    def median_yield(self) -> float:
        return statistics.median(self.yields) if self.yields else 0.0

    @property
    def zero_yield(self) -> int:
        """Words present in the lexicon that contribute nothing to fairness."""
        return sum(1 for y in self.yields if y == 0.0)


@dataclass
class FairnessAudit:
    lexicon_name: str
    assignment: str
    vocabulary_size: int
    sides: dict[str, SideAudit]

    @property
    def missing_sides(self) -> list[str]:
        """Sides with no vocabulary at all. This is the seed's original bug."""
        return [s for s, a in self.sides.items() if a.n == 0]

    @property
    def yield_ratio(self) -> float | None:
        """Proportionality yield over equality yield. 1.0 means a merit word and
        an equality word contribute equally. ``None`` if equality yields nothing."""
        eq = self.sides["equality"].mean_yield
        pr = self.sides["proportionality"].mean_yield
        return (pr / eq) if eq else None

    @property
    def balanced(self) -> bool:
        r = self.yield_ratio
        return r is not None and BALANCED_BAND[0] <= r <= BALANCED_BAND[1]


def audit_fairness(
    lexicon,
    lexicon_name: str = "lexicon",
    splitter: FairnessSplitter | None = None,
    assignment: str = "argmax",
) -> FairnessAudit:
    """Measure how a lexicon represents each half of fairness.

    ``assignment`` must match how the scorer is configured: under ``argmax`` a
    word contributes to its dominant foundation only, so a merit word assigned
    elsewhere yields zero fairness. Under ``probability`` every word contributes
    its fairness weight regardless.
    """
    if assignment not in ("argmax", "probability"):
        raise ValueError(f"unknown assignment mode: {assignment!r}")
    splitter = splitter or FairnessSplitter()
    sides = {s: SideAudit(side=s) for s in FAIRNESS_SUBFOUNDATIONS}

    vocabulary = 0
    for term, entry in lexicon.items():
        vocabulary += 1
        side = splitter._side(term)
        if side not in sides:
            continue
        weights = entry.foundations
        fairness = float(weights.get("fairness", 0.0) or 0.0)
        if assignment == "argmax":
            dominant = max(weights, key=weights.get) if weights else None
            contributes = dominant == "fairness"
        else:
            contributes = fairness > 0
        sides[side].words.append(term)
        sides[side].to_fairness += int(contributes)
        sides[side].yields.append(fairness if contributes else 0.0)

    return FairnessAudit(lexicon_name, assignment, vocabulary, sides)


def format_report(audit: FairnessAudit) -> str:
    lines = [
        f"Fairness-split audit: {audit.lexicon_name}",
        f"  vocabulary {audit.vocabulary_size} terms, assignment={audit.assignment}",
        "",
    ]
    for side in FAIRNESS_SUBFOUNDATIONS:
        a = audit.sides[side]
        lines.append(f"  {side}")
        if not a.n:
            lines.append("    NO VOCABULARY — this side cannot register at all.")
            continue
        lines.append(
            f"    {a.n} term(s) present; {a.to_fairness} assigned to fairness, "
            f"{a.zero_yield} yielding nothing"
        )
        lines.append(
            f"    fairness per occurrence: mean {a.mean_yield:.4f}, median {a.median_yield:.4f}"
        )
        lines.append(f"    terms: {', '.join(sorted(a.words))}")
    lines.append("")

    if audit.missing_sides:
        lines.append(
            f"  ALARM: no vocabulary for {', '.join(audit.missing_sides)}. Text framing "
            "fairness that way scores as containing no fairness at all, which "
            "under-measures whichever diet argues in those terms."
        )
        return "\n".join(lines)

    ratio = audit.yield_ratio
    if ratio is None:
        lines.append("  Equality vocabulary yields no fairness; ratio undefined.")
        return "\n".join(lines)

    lines.append(f"  proportionality / equality yield ratio: {ratio:.2f}")
    if audit.balanced:
        lines.append("  Within the reporting band: neither half is structurally favoured.")
    else:
        favoured = "equality" if ratio < 1 else "proportionality"
        lines.append(
            f"  Outside the reporting band: this lexicon yields more fairness for "
            f"{favoured}-framed language. That is a property of the dictionary, not "
            "of the diets being measured."
        )
    lines.append(
        "  Sample is the intersection of the split term lists with this lexicon, "
        "and both lists are unvalidated. See LIMITATIONS.md."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from scoring.lexicon import build_lexicon

    p = argparse.ArgumentParser(
        prog="validation.lexicon_audit",
        description="Audit a lexicon for equality/proportionality asymmetry",
    )
    p.add_argument("--lexicon", help="eMFD-format CSV; omit for the built-in demo seed")
    p.add_argument(
        "--assignment",
        default="argmax",
        choices=["argmax", "probability"],
        help="must match the scorer's configuration (default argmax)",
    )
    p.add_argument(
        "--min-evidence",
        type=int,
        default=2,
        help="splitter min_evidence (does not affect this audit's counts)",
    )
    args = p.parse_args(argv)

    lexicon, name = build_lexicon(args.lexicon)
    audit = audit_fairness(
        lexicon, name, FairnessSplitter(min_evidence=args.min_evidence), args.assignment
    )
    print(format_report(audit))
    # Non-zero only for the categorical failure, so this is usable as a check.
    return 1 if audit.missing_sides else 0


if __name__ == "__main__":
    raise SystemExit(main())
