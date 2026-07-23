"""Aggregation to diet profiles and the divergence metrics."""

from __future__ import annotations

import math

from compare.divergence import index_form, jensen_shannon_divergence, log_ratios
from scoring.aggregate import aggregate_profile, to_composition
from scoring.dictionary import DocumentScore


def _doc(**foundations):
    base = {"care": 0.0, "fairness": 0.0, "loyalty": 0.0, "authority": 0.0, "sanctity": 0.0}
    base.update(foundations)
    return DocumentScore(foundations=base, sentiment=0.0, moral_word_ratio=0.0,
                         word_count=1, matched_words=0)


def test_composition_sums_to_one():
    prof = aggregate_profile([_doc(care=0.2, fairness=0.1), _doc(loyalty=0.3)])
    comp = to_composition(prof)
    assert abs(sum(comp.values()) - 1.0) < 1e-9


def test_all_zero_profile_is_uniform():
    comp = to_composition(aggregate_profile([_doc(), _doc()]))
    assert all(abs(v - 0.2) < 1e-9 for v in comp.values())


def test_weighted_mean():
    prof = aggregate_profile([_doc(care=1.0), _doc(care=0.0)], weights=[3.0, 1.0])
    assert abs(prof["care"] - 0.75) < 1e-9


def test_weights_must_be_parallel():
    try:
        aggregate_profile([_doc(care=1.0)], weights=[1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched weights")


def test_jsd_identical_is_zero_and_symmetric():
    p = to_composition(aggregate_profile([_doc(care=0.3, loyalty=0.1)]))
    assert jensen_shannon_divergence(p, p) == 0.0


def test_jsd_bounded_and_symmetric():
    p = {"care": 1.0, "fairness": 0.0, "loyalty": 0.0, "authority": 0.0, "sanctity": 0.0}
    q = {"care": 0.0, "fairness": 0.0, "loyalty": 0.0, "authority": 0.0, "sanctity": 1.0}
    jsd_pq = jensen_shannon_divergence(p, q)
    jsd_qp = jensen_shannon_divergence(q, p)
    assert abs(jsd_pq - jsd_qp) < 1e-12
    assert 0.99 <= jsd_pq <= 1.0  # disjoint distributions -> ~1


def test_log_ratio_sign_and_parity():
    p = {"care": 0.6, "fairness": 0.4, "loyalty": 0.0, "authority": 0.0, "sanctity": 0.0}
    q = {"care": 0.3, "fairness": 0.4, "loyalty": 0.0, "authority": 0.0, "sanctity": 0.0}
    lr = log_ratios(p, q)
    assert lr["care"] > 0            # p over-indexes on care
    assert abs(lr["fairness"]) < 1e-3  # parity
    idx = index_form(p, q)
    assert idx["care"] > 100 and abs(idx["fairness"] - 100) < 1.0


def test_log_ratio_finite_with_zeros():
    p = {"care": 0.0, "fairness": 1.0, "loyalty": 0.0, "authority": 0.0, "sanctity": 0.0}
    q = {"care": 1.0, "fairness": 0.0, "loyalty": 0.0, "authority": 0.0, "sanctity": 0.0}
    assert all(math.isfinite(v) for v in log_ratios(p, q).values())
