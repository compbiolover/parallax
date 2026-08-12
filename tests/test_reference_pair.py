"""Which two personas the headline numbers are about, and how that is decided.

Before this existed, seven places took ``sorted(ids)[:2]``. That was already
wrong in a way nobody could see: ``modeled_ce`` sorts before ``self``, so the
per-foundation log-ratios reported "positive = the modeled diet over-indexes",
the opposite of what ``CLAUDE.md`` §3(5) asks for. With a library of personas it
stops working outright — adding one would change the headline number and the
recorded series without anyone touching a weight.
"""

from __future__ import annotations

import logging

from compare.reference import DEFAULT_MINE, DEFAULT_THEIRS, ReferencePair, resolve


def test_an_unconfigured_pair_is_the_two_personas_the_series_was_recorded_on():
    """Every snapshot ever written compares `self` and `modeled_ce`. Defaulting to
    anything else would silently break the continuity of the divergence chart."""
    assert resolve({}) == ReferencePair(DEFAULT_MINE, DEFAULT_THEIRS)
    assert resolve(None) == ReferencePair("self", "modeled_ce")


def test_settings_name_the_pair():
    settings = {"compare": {"reference_pair": {"mine": "wonk", "theirs": "devout"}}}
    assert resolve(settings) == ReferencePair("wonk", "devout")


def test_an_explicit_argument_beats_settings():
    """The layering contract the rest of the CLI uses: a flag wins, and an absent
    flag leaves settings alone rather than overriding them with a default."""
    settings = {"compare": {"reference_pair": {"mine": "wonk", "theirs": "devout"}}}
    assert resolve(settings, mine="activist") == ReferencePair("activist", "devout")
    assert resolve(settings, theirs="cable") == ReferencePair("wonk", "cable")


def test_the_pair_is_oriented_not_a_set():
    pair = ReferencePair("mine", "theirs")
    assert list(pair) == ["mine", "theirs"]
    assert pair.as_list() == ["mine", "theirs"]
    assert pair.ids == ("mine", "theirs")


def test_the_other_side_is_the_pairs_other_member():
    pair = ReferencePair("a", "b")
    assert pair.other("a") == "b"
    assert pair.other("b") == "a"


def test_asking_for_the_other_side_of_an_outsider_raises():
    """`next(d for d in diets if d != dominant)` used to answer this, and with
    three personas it returned an arbitrary third party as "the other diet"."""
    pair = ReferencePair("a", "b")
    try:
        pair.other("c")
    except KeyError:
        return
    raise AssertionError("expected a KeyError for a persona outside the pair")


def test_an_unknown_persona_degrades_with_a_logged_reason(caplog):
    """A run that produces a slightly differently-scoped number is more useful at
    6am than a run that produced nothing — the same posture every other step takes
    when a dependency is missing."""
    settings = {"compare": {"reference_pair": {"mine": "typo", "theirs": "modeled_ce"}}}
    families = {"left": ["self"], "right": ["modeled_ce"]}
    with caplog.at_level(logging.WARNING):
        pair = resolve(settings, available=["self", "modeled_ce"], families=families)
    assert pair == ReferencePair("self", "modeled_ce")
    assert "typo" in caplog.text


def test_the_fallback_still_compares_across_families(caplog):
    """Falling back to the first two ids alphabetically could pick two variants of
    the same side, and a comparison of one family with itself is not the metric."""
    families = {"left": ["wonk", "activist"], "right": ["cable", "devout"]}
    with caplog.at_level(logging.WARNING):
        pair = resolve(
            {"compare": {"reference_pair": {"mine": "gone", "theirs": "also_gone"}}},
            available=["activist", "cable", "devout", "wonk"],
            families=families,
        )
    assert pair == ReferencePair("wonk", "cable")


def test_a_known_pair_is_not_second_guessed():
    families = {"left": ["self"], "right": ["modeled_ce"]}
    settings = {"compare": {"reference_pair": {"mine": "self", "theirs": "modeled_ce"}}}
    assert resolve(settings, available=["self", "modeled_ce"], families=families) == ReferencePair(
        "self", "modeled_ce"
    )
