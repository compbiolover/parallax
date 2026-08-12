"""Lexicon fairness-split audit: the check that would have caught the seed's bias."""

from __future__ import annotations

from scoring.lexicon import Entry, Lexicon, load_seed
from validation.lexicon_audit import audit_fairness, format_report, main


def _lexicon(terms: dict[str, dict[str, float]]) -> Lexicon:
    lex = Lexicon()
    for term, weights in terms.items():
        lex.add(term, Entry(foundations=dict(weights), pole=1), wildcard=False)
    return lex


# -- the categorical failure ----------------------------------------------


def test_missing_side_is_detected():
    """The seed's original bug: equality vocabulary, no proportionality at all."""
    audit = audit_fairness(_lexicon({"equal": {"fairness": 1.0}}), "equality-only")
    assert audit.missing_sides == ["proportionality"]
    assert audit.sides["equality"].n == 1


def test_missing_side_is_alarming_in_the_report():
    audit = audit_fairness(_lexicon({"equal": {"fairness": 1.0}}), "equality-only")
    report = format_report(audit)
    assert "ALARM" in report
    assert "no fairness at all" in report


def test_cli_exits_non_zero_only_for_a_missing_side(capsys, tmp_path):
    csv_path = tmp_path / "lex.csv"
    csv_path.write_text(
        "word,care_p,fairness_p,loyalty_p,authority_p,sanctity_p\nequal,0.0,1.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    assert main(["--lexicon", str(csv_path)]) == 1
    capsys.readouterr()
    assert main([]) == 0  # built-in seed carries both sides


# -- yield accounting ------------------------------------------------------


def test_argmax_zeroes_words_the_lexicon_assigns_elsewhere():
    """A merit word the lexicon calls 'authority' contributes nothing to fairness,
    however clearly it belongs there conceptually."""
    lex = _lexicon(
        {
            "equal": {"fairness": 0.5, "care": 0.1},
            "merit": {"fairness": 0.4, "authority": 0.6},  # argmax is authority
        }
    )
    audit = audit_fairness(lex, "t", assignment="argmax")
    assert audit.sides["equality"].mean_yield == 0.5
    assert audit.sides["proportionality"].mean_yield == 0.0
    assert audit.sides["proportionality"].zero_yield == 1
    assert audit.yield_ratio == 0.0


def test_probability_mode_counts_the_weight_regardless_of_argmax():
    lex = _lexicon(
        {
            "equal": {"fairness": 0.5, "care": 0.1},
            "merit": {"fairness": 0.4, "authority": 0.6},
        }
    )
    audit = audit_fairness(lex, "t", assignment="probability")
    assert audit.sides["proportionality"].mean_yield == 0.4
    assert abs(audit.yield_ratio - 0.8) < 1e-9


def test_balanced_lexicon_reports_balanced():
    lex = _lexicon(
        {
            "equal": {"fairness": 0.5},
            "merit": {"fairness": 0.5},
        }
    )
    audit = audit_fairness(lex, "t")
    assert audit.yield_ratio == 1.0
    assert audit.balanced
    assert "neither half is structurally favoured" in format_report(audit)


def test_lean_is_named_and_attributed_to_the_dictionary():
    lex = _lexicon(
        {
            "equal": {"fairness": 1.0},
            "merit": {"fairness": 0.2},
        }
    )
    audit = audit_fairness(lex, "t")
    assert not audit.balanced
    report = format_report(audit)
    assert "equality-framed" in report
    assert "property of the dictionary" in report


def test_unknown_assignment_mode_is_rejected():
    try:
        audit_fairness(_lexicon({"equal": {"fairness": 1.0}}), "t", assignment="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# -- the shipped seed ------------------------------------------------------


def test_shipped_seed_carries_both_sides():
    """Regression guard on the fix: the seed must never lose a side again."""
    audit = audit_fairness(load_seed(), "seed")
    assert audit.missing_sides == []
    assert audit.sides["proportionality"].n >= 3
    assert audit.balanced


def test_lexicon_items_covers_exact_and_wildcard_terms():
    lex = Lexicon()
    lex.add("alpha", Entry(foundations={"care": 1.0}, pole=1), wildcard=False)
    lex.add("beta", Entry(foundations={"care": 1.0}, pole=1), wildcard=True)
    assert {t for t, _ in lex.items()} == {"alpha", "beta"}
    assert len(list(lex.items())) == len(lex)
