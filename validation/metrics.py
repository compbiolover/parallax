"""Agreement metrics for validating the foundation scorers against a gold set.

Two kinds:

- **Method vs. gold** (:func:`foundation_agreement`) — how well an automated
  scorer's continuous per-foundation output matches hand-coded binary labels:
  AUC (threshold-free), plus F1 / precision / recall / Cohen's kappa at a
  decision threshold. This is what drives the Phase 3 trigger (§5): if a
  foundation's AUC is below ~0.7 — expected for the binding foundations under a
  dictionary method — the dictionary alone is not trustworthy there.
- **Inter-coder** (:func:`krippendorff_alpha`) — reliability across multiple
  human coders, for when the gold set grows beyond a single annotator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def foundation_agreement(
    gold: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> dict:
    """Agreement between binary ``gold`` labels and continuous ``scores``.

    ``auc``/``kappa`` are ``None`` when undefined (a single class present in the
    gold labels or in the thresholded predictions), so callers can report "n/a"
    rather than a misleading number.
    """
    gold = list(gold)
    preds = [1 if s > threshold else 0 for s in scores]
    n = len(gold)
    positives = sum(gold)
    both_classes = 0 < positives < n

    return {
        "n": n,
        "positives": positives,
        "auc": roc_auc_score(gold, scores) if both_classes else None,
        "f1": f1_score(gold, preds, zero_division=0),
        "precision": precision_score(gold, preds, zero_division=0),
        "recall": recall_score(gold, preds, zero_division=0),
        "kappa": (
            cohen_kappa_score(gold, preds)
            if len(set(gold)) > 1 and len(set(preds)) > 1
            else None
        ),
    }


def krippendorff_alpha(coder_values: Sequence[Sequence]) -> float | None:
    """Krippendorff's alpha for nominal data across coders.

    ``coder_values`` is one sequence per coder, all the same length (one entry
    per unit); ``None`` marks a missing judgement. Returns ``None`` if there is
    no pairable data. Verified against Krippendorff's canonical reliability
    example (alpha ≈ 0.743).
    """
    n_units = len(coder_values[0]) if coder_values else 0
    coincidence: Counter = Counter()
    total = 0.0
    for u in range(n_units):
        vals = [coder[u] for coder in coder_values if coder[u] is not None]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    coincidence[(vals[i], vals[j])] += 1.0 / (m - 1)
        total += m

    if total == 0:
        return None

    marginals: Counter = Counter()
    for (v, _w), c in coincidence.items():
        marginals[v] += c
    n = sum(marginals.values())
    if n <= 1:
        return None

    observed = sum(c for (v, w), c in coincidence.items() if v != w)
    expected = sum(
        marginals[v] * marginals[w] / (n - 1)
        for v in marginals
        for w in marginals
        if v != w
    )
    if expected == 0:
        return 1.0
    return 1.0 - observed / expected
