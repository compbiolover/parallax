"""The register probe: matched-by-construction pairs, and honest arithmetic.

The probe exists to catch a tilt in the liberty rubric. These tests exist to
catch a tilt in the probe — a pair whose two sides differ by more than the
actor, or a report that reads noise as a finding.
"""

from __future__ import annotations

import re

from scoring.liberty import LibertyScore
from validation.register_probe import (
    PAIRS,
    PRIVATE,
    STATE,
    Cell,
    ProbePair,
    ProbeResult,
    format_report,
    run_probe,
)


def _score(presence: float, register: str = STATE) -> LibertyScore:
    return LibertyScore(
        presence=presence,
        pole="vice",
        register=register,
        quote="no choice but to comply",
        rationale="coercion",
        model="stub",
    )


class _StubTagger:
    """Returns a canned score per text, cycling through a list when given one."""

    model = "stub-model"

    def __init__(self, by_actor=None, sequence=None):
        self.by_actor = by_actor or {}
        self.sequence = list(sequence or [])
        self.seen: list[str] = []

    def score(self, text: str):
        self.seen.append(text)
        if self.sequence:
            return self.sequence.pop(0)
        for needle, score in self.by_actor.items():
            if needle in text:
                return score
        return _score(0.5)


# -- matched by construction ------------------------------------------------


def test_every_pair_has_exactly_one_actor_slot():
    """Two slots would let the two renderings differ in more than one place;
    zero would make the pair identical and the probe vacuous."""
    for pair in PAIRS:
        assert pair.template.count("{actor}") == 1, pair.topic
        assert "{" not in pair.template.replace("{actor}", ""), pair.topic


def test_the_two_renderings_differ_only_in_the_actor_phrase():
    """The guarantee the whole probe rests on. If the state sentence is written
    more vividly than the private one, the probe measures my prose, not the
    rubric — which is the exact failure it was built to detect."""
    for pair in PAIRS:
        state, private = pair.state_text(), pair.private_text()
        assert state.replace(pair.state_actor, "\x00") == private.replace(
            pair.private_actor, "\x00"
        ), pair.topic


def test_topics_are_distinct():
    """Topics key the result dicts — a duplicate would silently drop a cell."""
    topics = [p.topic for p in PAIRS]
    assert len(topics) == len(set(topics))


def test_no_pair_names_its_own_register():
    """A template containing 'state' or 'private' would hand the model the
    answer through the shared text rather than through the actor."""
    for pair in PAIRS:
        body = pair.template.replace("{actor}", "").lower()
        assert not re.search(r"\b(state|government|private|corporate)\b", body), pair.topic


# -- cell and result arithmetic ---------------------------------------------


def test_empty_cell_reports_zero_rather_than_raising():
    cell = Cell()
    assert cell.mean == 0.0
    assert cell.spread == 0.0


def test_single_sample_has_no_spread():
    """One sample cannot establish a noise floor, and pretending otherwise
    would make any gap look significant."""
    cell = Cell(presences=[0.7])
    assert cell.spread == 0.0


def test_gap_is_state_minus_private():
    result = ProbeResult(model="m", repeats=2)
    result.state["a"] = Cell(presences=[0.8, 0.6])
    result.private["a"] = Cell(presences=[0.5, 0.5])
    assert result.state_mean == 0.7
    assert result.private_mean == 0.5
    assert abs(result.gap - 0.2) < 1e-9


def test_noise_pools_only_cells_that_were_repeated():
    """A cell with one usable sample contributes no spread information; folding
    its zero into the mean would understate the noise floor and turn variance
    into a finding."""
    result = ProbeResult(model="m", repeats=2)
    result.state["a"] = Cell(presences=[0.9, 0.5])      # stdev ~0.283
    result.private["a"] = Cell(presences=[0.7])          # no spread to pool
    assert abs(result.noise - result.state["a"].spread) < 1e-9


# -- register classification ------------------------------------------------


def test_register_accuracy_counts_both_as_correct():
    """A single-actor sentence can defensibly read as either register; the
    failure under test is calling private power state coercion, not hedging."""
    result = ProbeResult(model="m", repeats=3)
    result.state["a"] = Cell(registers=[STATE, "both", PRIVATE])
    assert abs(result.register_accuracy(result.state, STATE) - 2 / 3) < 1e-9


def test_register_accuracy_of_nothing_is_zero_not_a_crash():
    result = ProbeResult(model="m", repeats=1)
    result.state["a"] = Cell()
    assert result.register_accuracy(result.state, STATE) == 0.0


# -- the run ----------------------------------------------------------------


def test_run_probe_scores_both_sides_of_every_pair():
    tagger = _StubTagger()
    pairs = PAIRS[:2]
    result = run_probe(tagger, pairs=pairs, repeats=2)

    assert len(tagger.seen) == len(pairs) * 2 * 2
    for pair in pairs:
        assert result.state[pair.topic].presences == [0.5, 0.5]
        assert result.private[pair.topic].presences == [0.5, 0.5]


def test_run_probe_records_the_model_from_the_tagger():
    assert run_probe(_StubTagger(), pairs=PAIRS[:1], repeats=1).model == "stub-model"


def test_failed_calls_are_counted_not_averaged_in():
    """A dropped call is an unknown, not a zero. Averaging it in would drag the
    mean toward the register that happened to fail more often."""
    pair = PAIRS[0]
    tagger = _StubTagger(sequence=[_score(0.8), None, _score(0.6), _score(0.4)])
    result = run_probe(tagger, pairs=(pair,), repeats=2)

    assert result.state[pair.topic].presences == [0.8, 0.6]
    assert result.private[pair.topic].presences == [0.4]
    assert result.private[pair.topic].failures == 1
    assert result.failures == 1
    assert result.private_mean == 0.4        # not (0.4 + 0) / 2


def test_a_one_sided_tilt_shows_up_as_a_gap():
    """The probe's reason for existing: state framing scored higher than an
    otherwise identical private-power sentence."""
    tagger = _StubTagger(
        by_actor={
            PAIRS[0].state_actor: _score(0.9, STATE),
            PAIRS[0].private_actor: _score(0.3, PRIVATE),
        }
    )
    result = run_probe(tagger, pairs=PAIRS[:1], repeats=2)
    assert result.gap > 0.5
    assert result.register_accuracy(result.state, STATE) == 1.0
    assert result.register_accuracy(result.private, PRIVATE) == 1.0


# -- the report reads its own numbers honestly ------------------------------


def _flat(result: ProbeResult) -> str:
    """The report with line breaks flattened — the prose is wrapped to terminal
    width, so a phrase can straddle two lines without being any less present."""
    return " ".join(format_report(result).split())


def _report(state, private, repeats):
    result = ProbeResult(model="m", repeats=repeats)
    result.state["compulsion"] = Cell(presences=list(state), registers=[STATE] * len(state))
    result.private["compulsion"] = Cell(
        presences=list(private), registers=[PRIVATE] * len(private)
    )
    return _flat(result)


def test_one_repeat_refuses_to_interpret_the_magnitude():
    """With no within-cell variance measured, a large gap and a rounding error
    look the same. Say so rather than picking one."""
    text = _report([0.9], [0.2], repeats=1)
    assert "No noise floor" in text
    assert "--repeats" in text


def test_a_gap_inside_the_noise_is_reported_as_no_evidence_either_way():
    text = _report([0.9, 0.1], [0.8, 0.2], repeats=2)
    assert "absence of evidence" in text


def test_a_gap_above_the_noise_names_the_favoured_side():
    text = _report([0.9, 0.9], [0.2, 0.2], repeats=2)
    assert "leaning toward state framing" in text
    assert "Descriptive only" in text


def test_a_private_leaning_gap_names_private_power():
    """The symmetric case. A probe that could only report a tilt in one
    direction would be the bias it is looking for."""
    text = _report([0.2, 0.2], [0.9, 0.9], repeats=2)
    assert "leaning toward private-power framing" in text


def test_dropped_calls_are_disclosed_in_the_report():
    result = ProbeResult(model="m", repeats=2)
    result.state["compulsion"] = Cell(presences=[0.5, 0.5], failures=1)
    result.private["compulsion"] = Cell(presences=[0.5, 0.5])
    assert "1 call(s) returned no verdict" in _flat(result)


def test_report_always_states_the_matched_by_construction_guarantee():
    """A reader deciding whether to believe a gap needs to know the two sides
    were generated from one template, not written by hand."""
    assert "identical by construction" in _report([0.5, 0.5], [0.5, 0.5], repeats=2)


def test_report_lists_every_topic_with_both_means():
    result = run_probe(_StubTagger(), pairs=PAIRS, repeats=2)
    text = format_report(result)
    for pair in PAIRS:
        assert pair.topic in text


def test_report_lines_stay_within_terminal_width():
    """An unwrapped 300-character caveat is the easiest thing in a report to
    skim past, and the caveats are the part that matters most here."""
    result = run_probe(_StubTagger(), pairs=PAIRS, repeats=2)
    assert all(len(line) <= 80 for line in format_report(result).splitlines())


# -- rendering --------------------------------------------------------------


def test_probe_pair_renders_the_actor_into_the_slot():
    pair = ProbePair("t", "{actor} did the thing.", "The board", "The firm")
    assert pair.state_text() == "The board did the thing."
    assert pair.private_text() == "The firm did the thing."
