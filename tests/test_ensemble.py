"""Ensemble scorer: combination, confidence signal, and calibration report."""

from __future__ import annotations

import pytest

from scoring.ensemble import EnsembleScorer, Tagger, dictionary_prob
from scoring.foundations import CLASSIC_FOUNDATIONS
from validation.evaluate import confidence_calibration
from validation.gold import GoldItem, GoldSet


def _const(value: float):
    return lambda text: {f: value for f in CLASSIC_FOUNDATIONS}


def _map(mapping: dict):
    return lambda text: {f: mapping.get(f, 0.0) for f in CLASSIC_FOUNDATIONS}


def test_agreement_is_high_confidence():
    ens = EnsembleScorer([Tagger("a", _const(0.9)), Tagger("b", _const(0.8))])
    es = ens.score("x")["care"]
    assert es.label == 1
    assert es.low_confidence is False
    assert es.confidence == 1.0
    assert es.score == pytest.approx(0.85)


def test_disagreement_is_low_confidence():
    ens = EnsembleScorer([Tagger("a", _const(0.9)), Tagger("b", _const(0.1))])
    es = ens.score("x")["care"]
    assert es.low_confidence is True
    assert es.confidence == pytest.approx(0.5)   # equal weight, one each way
    assert es.score == pytest.approx(0.5)
    assert set(es.votes.values()) == {0, 1}


def test_weighting_shifts_score_but_split_still_flagged():
    ens = EnsembleScorer([Tagger("weak", _const(0.9), 1.0), Tagger("strong", _const(0.1), 3.0)])
    es = ens.score("x")["care"]
    assert es.score == pytest.approx((0.9 * 1 + 0.1 * 3) / 4)  # 0.3 -> label 0
    assert es.label == 0
    assert es.low_confidence is True             # still a split -> flagged


def test_scores_returns_continuous_per_foundation():
    ens = EnsembleScorer([Tagger("a", _const(0.9)), Tagger("b", _const(0.7))])
    scores = ens.scores("x")
    assert set(scores) == set(CLASSIC_FOUNDATIONS)
    assert scores["care"] == pytest.approx(0.8)


def test_label_tie_break_matches_metrics_strict_threshold():
    # An exact 0.5 must label 0, matching validation.metrics (`s > threshold`),
    # so the calibration report and the F1/kappa report never disagree on it.
    ens = EnsembleScorer([Tagger("a", _const(1.0)), Tagger("b", _const(0.0))])
    es = ens.score("x")["care"]
    assert es.score == pytest.approx(0.5)
    assert es.label == 0


def test_dictionary_prob_adapter_is_presence_binary():
    class _DS:
        def score(self, text):
            class R:
                foundations = {"care": 0.2, "fairness": 0.0, "loyalty": 0.0,
                               "authority": 0.0, "sanctity": 0.0}
            return R()

    prob = dictionary_prob(_DS())("anything")
    assert prob["care"] == 1.0 and prob["fairness"] == 0.0


def test_empty_ensemble_rejected():
    with pytest.raises(ValueError):
        EnsembleScorer([])


def test_confidence_calibration_rewards_agreement():
    # tagger A is always right; tagger B is right only on care -> care is
    # high-confidence & correct, others are low-confidence.
    all_present = {"care": 1, "fairness": 1, "loyalty": 1, "authority": 1, "sanctity": 1}
    gold = GoldSet(version=1, coders=["t"], items=[GoldItem("a", "t1", all_present)])
    a = _const(0.9)                              # says present everywhere (matches gold)
    b = _map({"care": 0.9})                      # present only on care
    ens = EnsembleScorer([Tagger("a", a), Tagger("b", b)])
    cal = confidence_calibration(gold, ens)
    # care: both present -> high-confidence, correct
    assert cal["high_confidence"]["n"] == 1
    assert cal["high_confidence"]["accuracy"] == 1.0
    # the other four: split -> low-confidence
    assert cal["low_confidence"]["n"] == 4
