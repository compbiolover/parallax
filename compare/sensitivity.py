"""Does the finding survive a different weighting?

``CLAUDE.md`` §5 asks for conclusions to be sensitivity-tested against source
weighting, and ``LIMITATIONS.md`` says plainly that the weights are the
least-evidenced and most load-bearing part of the tool: no survey gives the
devotional-versus-cable split of a devout evangelical's week, or the policy-journal
share of a wonk's. Every number on the dashboard sits downstream of numbers that
were reasoned about and written down.

That requirement was expensive to satisfy while a document's weight was baked in at
ingestion — checking a different weighting meant re-ingesting the corpus. Weights
resolve at aggregation now, so this module can re-run the arithmetic over score rows
already in the store. One scan per persona, then every perturbation is in memory.

**What it perturbs.** One stratum weight at a time, scaled up and down, for each
persona in the reference pair. One-at-a-time rather than random draws: the answer
"the headline moves most when you change how much cable counts" is actionable, and
"the headline varies by ±0.02 under random resampling" is not. It is also
deterministic, so two runs on one corpus agree.

**What to read.** The JSD range matters less than the sign table. A divergence that
moves from 0.033 to 0.041 is the same finding; a per-foundation log-ratio that
changes sign is a different one, because the sign is the human-readable claim —
which diet over-indexes on care. A flip means the claim was an artifact of the
weighting rather than a fact about the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from scoring.aggregate import aggregate_profile, to_composition
from scoring.dictionary import DocumentScore
from scoring.foundations import CLASSIC_FOUNDATIONS

from .divergence import jensen_shannon_divergence, log_ratios

# How far to move a stratum weight. ±50% is large enough that surviving it means
# something and small enough to stay a plausible alternative reading of the same
# diet, rather than a different diet wearing its name.
DEFAULT_FACTOR = 0.5

# Below this a JSD difference is not worth reporting as movement: the snapshot
# series itself is rounded to six decimals, so anything smaller is noise the tool
# does not even record.
NEGLIGIBLE = 1e-6


@dataclass(frozen=True)
class Perturbation:
    """One stratum's weight moved, and what that did to the headline."""

    persona: str
    stratum: str
    factor: float             # 1.5 = counted 50% more, 0.5 = 50% less
    jsd: float
    delta: float              # signed change from the baseline JSD
    flipped: tuple[str, ...] = ()   # foundations whose log-ratio changed sign

    @property
    def label(self) -> str:
        direction = "up" if self.factor > 1 else "down"
        return f"{self.persona}/{self.stratum} {direction}"


@dataclass
class SensitivityReport:
    pair: tuple[str, str]
    baseline_jsd: float
    baseline_log_ratios: dict[str, float]
    perturbations: list[Perturbation] = field(default_factory=list)
    factor: float = DEFAULT_FACTOR
    docs: dict[str, int] = field(default_factory=dict)

    @property
    def jsd_range(self) -> tuple[float, float]:
        values = [p.jsd for p in self.perturbations] or [self.baseline_jsd]
        return min(values), max(values)

    @property
    def worst(self) -> list[Perturbation]:
        """Perturbations that moved the headline most, largest first."""
        return sorted(self.perturbations, key=lambda p: abs(p.delta), reverse=True)

    @property
    def flipped_foundations(self) -> dict[str, list[str]]:
        """``{foundation: [perturbation labels that flipped its sign]}``.

        The part of the report that answers "does the finding survive".
        """
        out: dict[str, list[str]] = {}
        for p in self.perturbations:
            for foundation in p.flipped:
                out.setdefault(foundation, []).append(p.label)
        return out

    @property
    def stable(self) -> bool:
        return not self.flipped_foundations

    def to_dict(self) -> dict:
        return {
            "pair": list(self.pair),
            "factor": self.factor,
            "baseline_jsd": self.baseline_jsd,
            "jsd_range": list(self.jsd_range),
            "docs": dict(self.docs),
            "stable": self.stable,
            "flipped": {k: list(v) for k, v in self.flipped_foundations.items()},
            "perturbations": [
                {
                    "persona": p.persona, "stratum": p.stratum, "factor": p.factor,
                    "jsd": p.jsd, "delta": p.delta, "flipped": list(p.flipped),
                }
                for p in self.worst
            ],
        }


def _rows(store, weights: dict[str, float], scorer: str) -> list:
    return store.scores_for_sources(weights, scorer)


def _profile(rows: list, weights: dict[str, float]) -> dict[str, float] | None:
    """Re-aggregate cached score rows under a (possibly perturbed) weighting.

    The rows are fetched once and reused for every perturbation — that is the whole
    reason this is cheap. Scores do not depend on weights; only the aggregation does.
    """
    if not rows:
        return None
    scores = [
        DocumentScore(
            foundations={f: (row[f] or 0.0) for f in CLASSIC_FOUNDATIONS},
            sentiment=0.0, moral_word_ratio=0.0, word_count=1, matched_words=0,
        )
        for row in rows
    ]
    applied = [weights.get(row["source_id"], 0.0) for row in rows]
    if not any(w > 0 for w in applied):
        return None
    return to_composition(aggregate_profile(scores, applied))


def _sign_flips(baseline: dict[str, float], candidate: dict[str, float]) -> tuple[str, ...]:
    """Foundations whose log-ratio changed sign, ignoring ones sitting on zero.

    A ratio within ``NEGLIGIBLE`` of parity has no sign worth defending, and
    counting its wobble as a flip would report noise as a lost finding.
    """
    flipped = []
    for foundation, value in baseline.items():
        other = candidate.get(foundation, 0.0)
        if abs(value) < NEGLIGIBLE or abs(other) < NEGLIGIBLE:
            continue
        if (value > 0) != (other > 0):
            flipped.append(foundation)
    return tuple(flipped)


def analyze(
    store,
    registry,
    pair,
    factor: float = DEFAULT_FACTOR,
    scorer: str = "dictionary",
) -> SensitivityReport | None:
    """Perturb each stratum weight of each persona in ``pair`` and report.

    ``None`` when either persona has no scored documents — "the weighting does not
    matter" and "there is nothing to weight" are different statements, and a report
    full of zeros would read as the first.
    """
    personas = {}
    rows = {}
    weights = {}
    for persona_id in pair:
        persona = registry.persona(persona_id)
        if persona is None:
            return None
        personas[persona_id] = persona
        weights[persona_id] = registry.weights_of(persona)
        rows[persona_id] = _rows(store, weights[persona_id], scorer)

    base = {p: _profile(rows[p], weights[p]) for p in pair}
    if any(profile is None for profile in base.values()):
        return None

    mine, theirs = pair.mine, pair.theirs
    baseline_jsd = jensen_shannon_divergence(base[mine], base[theirs])
    baseline_lr = log_ratios(base[mine], base[theirs])

    report = SensitivityReport(
        pair=(mine, theirs),
        baseline_jsd=baseline_jsd,
        baseline_log_ratios=baseline_lr,
        factor=factor,
        docs={p: len(rows[p]) for p in pair},
    )

    for persona_id in pair:
        persona = personas[persona_id]
        for stratum, weight in sorted(persona.stratum_weights.items()):
            for multiplier in (1.0 + factor, 1.0 - factor):
                moved = replace(
                    persona,
                    stratum_weights={**persona.stratum_weights, stratum: weight * multiplier},
                )
                profile = _profile(rows[persona_id], registry.weights_of(moved))
                if profile is None:
                    continue
                candidate = dict(base)
                candidate[persona_id] = profile
                jsd = jensen_shannon_divergence(candidate[mine], candidate[theirs])
                report.perturbations.append(Perturbation(
                    persona=persona_id,
                    stratum=stratum,
                    factor=multiplier,
                    jsd=jsd,
                    delta=jsd - baseline_jsd,
                    flipped=_sign_flips(
                        baseline_lr, log_ratios(candidate[mine], candidate[theirs])
                    ),
                ))
    return report


def format_report(report: SensitivityReport | None, limit: int = 10) -> str:
    """The report as text, worst movers first."""
    if report is None:
        return ("Not enough scored documents to test a weighting. Run "
                "`python -m ingestion run` first.")
    mine, theirs = report.pair
    low, high = report.jsd_range
    lines = [
        f"Weighting sensitivity — {mine} vs {theirs}",
        f"  each stratum weight moved +/-{report.factor:.0%}, one at a time",
        f"  documents: {mine} {report.docs.get(mine, 0)}, "
        f"{theirs} {report.docs.get(theirs, 0)}",
        "",
        f"  headline divergence  {report.baseline_jsd:.4f}"
        f"   range {low:.4f} .. {high:.4f}",
        "",
    ]

    if report.stable:
        lines += [
            "  Every per-foundation log-ratio kept its sign under every "
            "perturbation.",
            "  Which diet over-indexes on what does not depend on these weights.",
        ]
    else:
        lines.append("  SIGN FLIPS — these claims do not survive a re-weighting:")
        for foundation, causes in sorted(report.flipped_foundations.items()):
            shown = ", ".join(causes[:3])
            more = f" (+{len(causes) - 3} more)" if len(causes) > 3 else ""
            lines.append(f"    {foundation:10} flips when {shown}{more}")
        lines.append("  A flipped sign means the direction was an artifact of the "
                     "weighting.")

    movers = [p for p in report.worst if abs(p.delta) > NEGLIGIBLE][:limit]
    if movers:
        lines += ["", "  Largest movers:"]
        for p in movers:
            flag = "  <- sign flip" if p.flipped else ""
            lines.append(
                f"    {p.label:38} {p.jsd:.4f}  ({p.delta:+.4f}){flag}"
            )
    else:
        lines += ["", "  No perturbation moved the divergence measurably."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from compare.reference import resolve
    from ingestion.config import load_registry, load_settings
    from ingestion.datastore import Datastore

    parser = argparse.ArgumentParser(
        prog="compare.sensitivity",
        description="Does the finding survive a different source weighting?",
    )
    parser.add_argument("--db", help="SQLite path (default from settings)")
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--factor", type=float, default=DEFAULT_FACTOR,
                        help="fraction to move each stratum weight (default 0.5)")
    parser.add_argument("--scorer", default="dictionary")
    parser.add_argument("--mine", help="persona id for your side of the pair")
    parser.add_argument("--theirs", help="persona id for the other side")
    parser.add_argument("--limit", type=int, default=10,
                        help="how many movers to list (default 10)")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    db = args.db or (settings.get("datastore", {}) or {}).get("path", "data/parallax.sqlite")
    registry = load_registry(settings=settings)
    pair = resolve(settings, args.mine, args.theirs,
                   available=registry.persona_ids(), families=registry.families())
    store = Datastore(db)
    try:
        report = analyze(store, registry, pair, args.factor, args.scorer)
        print(format_report(report, args.limit))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
