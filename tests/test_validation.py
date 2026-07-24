"""Validation harness: metrics, gold loader, evaluate, transformer stub."""

from __future__ import annotations

from scoring.transformer import TransformerScorer
from validation.evaluate import binding_trigger, evaluate
from validation.gold import GOLD_DIR, GoldItem, GoldSet, load_gold
from validation.metrics import foundation_agreement, krippendorff_alpha


# -- metrics ---------------------------------------------------------------

def test_foundation_agreement_perfect_separation():
    gold = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.1, 0.2]
    m = foundation_agreement(gold, scores, threshold=0.5)
    assert m["auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["kappa"] == 1.0
    assert m["positives"] == 2 and m["n"] == 4


def test_foundation_agreement_single_class_gives_none():
    m = foundation_agreement([0, 0, 0], [0.1, 0.2, 0.0], threshold=0.5)
    assert m["auc"] is None      # AUC undefined with one class
    assert m["kappa"] is None


def test_krippendorff_alpha_canonical_example():
    # Krippendorff's canonical nominal reliability-data example -> alpha ~ 0.743
    N = None
    coders = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
        [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N],
    ]
    alpha = krippendorff_alpha(coders)
    assert abs(alpha - 0.743) < 0.01


def test_krippendorff_perfect_and_none():
    assert krippendorff_alpha([[1, 2, 1], [1, 2, 1]]) == 1.0
    assert krippendorff_alpha([[None, None], [None, None]]) is None


# -- gold loader -----------------------------------------------------------

def test_seed_gold_loads_and_is_balanced():
    gold = load_gold(GOLD_DIR / "seed.json")
    assert len(gold.items) >= 40
    assert gold.coders == ["author"]
    for foundation in ("care", "fairness", "loyalty", "authority", "sanctity"):
        col = gold.gold_column(foundation)
        assert 0 < sum(col) < len(col)  # both classes present -> AUC computable


# -- evaluate + trigger ----------------------------------------------------

def _toy_gold():
    return GoldSet(version=1, coders=["t"], items=[
        GoldItem("a", "x", {"care": 1, "fairness": 0, "loyalty": 1, "authority": 0, "sanctity": 0}),
        GoldItem("b", "y", {"care": 0, "fairness": 1, "loyalty": 0, "authority": 1, "sanctity": 1}),
        GoldItem("c", "z", {"care": 1, "fairness": 0, "loyalty": 1, "authority": 0, "sanctity": 0}),
        GoldItem("d", "w", {"care": 0, "fairness": 1, "loyalty": 0, "authority": 1, "sanctity": 1}),
    ])


def test_evaluate_and_binding_trigger():
    gold = _toy_gold()
    # a scorer that perfectly separates loyalty but is random on sanctity
    def score_fn(text):
        return {"care": 0.0, "fairness": 0.0,
                "loyalty": 0.9 if text in ("x", "z") else 0.1,
                "authority": 0.0, "sanctity": 0.5}
    results = evaluate(gold, score_fn, threshold=0.5)
    assert results["loyalty"]["auc"] == 1.0
    flagged = binding_trigger(results, trigger_auc=0.7)
    assert "loyalty" not in flagged          # perfect -> not flagged
    assert "authority" in flagged            # constant 0.0 -> below trigger


# -- transformer scorer (stubbed, no model download) -----------------------

def test_transformer_scorer_stub_shape():
    ts = TransformerScorer(predict_fn=lambda foundation, text: 0.8 if foundation == "care" else 0.1)
    out = ts.score("a caring headline")
    assert set(out) == {"care", "fairness", "loyalty", "authority", "sanctity"}
    assert out["care"] == 0.8 and out["loyalty"] == 0.1
    assert all(0.0 <= v <= 1.0 for v in out.values())
