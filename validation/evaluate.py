"""Evaluate a foundation scorer against the gold set.

Runs a scorer over every gold item, then reports per-foundation agreement and
checks the Phase 3 trigger from §5: if a *binding* foundation (loyalty,
authority, sanctity) scores below ~0.7 AUC, the dictionary alone is not
trustworthy there and the transformer/Claude taggers are warranted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from scoring.foundations import CLASSIC_FOUNDATIONS

from .gold import GoldSet
from .metrics import foundation_agreement

# The foundations MFT literature (and §5) expects a dictionary to handle worst.
BINDING = ("loyalty", "authority", "sanctity")
TRIGGER_AUC = 0.7

# text -> {foundation: continuous score}
ScoreFn = Callable[[str], dict[str, float]]


def evaluate(
    goldset: GoldSet,
    score_fn: ScoreFn,
    foundations: Sequence[str] = CLASSIC_FOUNDATIONS,
    threshold: float = 0.0,
) -> dict[str, dict]:
    """Per-foundation agreement between ``score_fn`` and the gold labels."""
    scored = [score_fn(item.text) for item in goldset.items]
    results: dict[str, dict] = {}
    for f in foundations:
        gold = goldset.gold_column(f)
        scores = [float(sc.get(f, 0.0) or 0.0) for sc in scored]
        results[f] = foundation_agreement(gold, scores, threshold)
    return results


def binding_trigger(results: dict[str, dict], trigger_auc: float = TRIGGER_AUC) -> list[str]:
    """Binding foundations whose AUC falls below the trigger (or is undefined)."""
    flagged = []
    for f in BINDING:
        auc = results.get(f, {}).get("auc")
        if auc is None or auc < trigger_auc:
            flagged.append(f)
    return flagged


def format_report(results: dict[str, dict], label: str) -> str:
    lines = [
        f"Validation: {label} vs gold set",
        f"  {'foundation':11} {'n':>3} {'pos':>3} {'AUC':>6} {'F1':>6} {'kappa':>6}",
    ]
    aucs = []
    for f, m in results.items():
        auc = m["auc"]
        if auc is not None:
            aucs.append(auc)
        binding = " *binding" if f in BINDING else ""
        lines.append(
            f"  {f:11} {m['n']:>3} {m['positives']:>3} "
            f"{_fmt(m['auc']):>6} {m['f1']:>6.2f} {_fmt(m['kappa']):>6}{binding}"
        )
    if aucs:
        lines.append(f"  macro-AUC: {sum(aucs) / len(aucs):.3f}")
    flagged = binding_trigger(results)
    if flagged:
        lines.append(
            f"  ⚠ TRIGGER (§5): binding foundation(s) below {TRIGGER_AUC} AUC: "
            f"{', '.join(flagged)} → add the transformer/Claude taggers."
        )
    else:
        lines.append(f"  binding foundations all ≥ {TRIGGER_AUC} AUC.")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"
