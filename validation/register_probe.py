"""Does the liberty tagger score both registers evenhandedly?

``LIMITATIONS.md`` names this as the first thing to check against real data, and
it is the assumption the whole liberty foundation rests on. Liberty is claimed in
two registers — freedom from state coercion, and freedom from private or
structural domination — and ``scoring/liberty.py``'s rubric instructs that
neither is more truly liberty. A test asserts that instruction is present in the
prompt. Nothing so far has checked whether the model *obeys* it.

If it does not, the failure is quiet and severe: one modeled diet scores as
engaged with liberty and the other as indifferent, which reads as a finding about
media rather than a property of the rubric. That is the fairness-lexicon failure
(see ``validation/lexicon_audit.py``) reappearing inside a prompt.

**Pairs are matched by construction, not by judgement.** Each probe item is a
single template containing one ``{actor}`` slot, rendered twice — once with a
state actor, once with a private one. Every other word is identical by
construction. Writing two sentences by hand and calling them equivalent would put
my own framing instincts inside the instrument, which is the exact error the
whole probe exists to detect.

**What this measures, and what it does not.** Two separate things:

1. *Register classification* — does a state-actor item get labelled
   ``from_state``, and a private-actor item ``from_private_power``? This is the
   categorical check, and it either works or it doesn't.
2. *Presence magnitude* — does one register score systematically higher? This is
   the subtle one, and the honest answer needs repetition: Sonnet 5 rejects
   ``temperature``, so run-to-run variance cannot be dialled down, and a single
   call per condition cannot separate a real tilt from sampling noise.

The report is **descriptive, not inferential**. With a handful of repeats per
cell there is no meaningful significance test here, so it prints the gap next to
the pooled within-condition spread and leaves the reading to you. A gap smaller
than the noise is not evidence of fairness; it is absence of evidence either way.

    python -m validation.register_probe                    # 10 pairs x 3 repeats
    python -m validation.register_probe --repeats 5
    python -m validation.register_probe --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import statistics
import textwrap
from dataclasses import dataclass, field

STATE = "from_state"
PRIVATE = "from_private_power"


@dataclass(frozen=True)
class ProbePair:
    """One template rendered with two actors.

    The template carries exactly one ``{actor}`` slot, so the two renderings are
    lexically identical apart from the actor phrase. That is the whole point:
    any difference in score is attributable to who is doing the coercing.
    """

    topic: str
    template: str
    state_actor: str
    private_actor: str

    def state_text(self) -> str:
        return self.template.format(actor=self.state_actor)

    def private_text(self) -> str:
        return self.template.format(actor=self.private_actor)


# Ten templates across distinct domains — compulsion, surveillance, speech,
# medical disclosure, livelihood, property, exit, conscience, assembly, data.
# Both actor lists are ordinary institutional nouns; neither side is written
# more vividly than the other, and the template guarantees the rest matches.
PAIRS: tuple[ProbePair, ...] = (
    ProbePair(
        "compulsion",
        "{actor} left residents no choice but to comply with the new rule.",
        "State officials", "Company executives",
    ),
    ProbePair(
        "surveillance",
        "{actor} now tracks where residents go and how long they stay there.",
        "The police department", "The employer",
    ),
    ProbePair(
        "speech",
        "{actor} removed the posts and warned that further speech would be penalized.",
        "The agency", "The platform",
    ),
    ProbePair(
        "medical disclosure",
        "{actor} required workers to disclose personal medical records before returning.",
        "The health department", "The employer",
    ),
    ProbePair(
        "livelihood",
        "{actor} can revoke the permit at will, leaving drivers unable to work.",
        "The licensing board", "The platform operator",
    ),
    ProbePair(
        "property",
        "{actor} seized the equipment without notice and refused to return it.",
        "Federal agents", "The landlord",
    ),
    ProbePair(
        "exit",
        "{actor} made it impossible to leave without paying a penalty few can afford.",
        "The new statute", "The service agreement",
    ),
    ProbePair(
        "conscience",
        "{actor} punished staff who objected to the policy on grounds of conscience.",
        "The state board", "The corporation",
    ),
    ProbePair(
        "assembly",
        "{actor} broke up the gathering and told organizers they could not meet again.",
        "City police", "Mall security",
    ),
    ProbePair(
        "data",
        "{actor} collected the messages without consent and would not say how they "
        "would be used.",
        "The intelligence service", "The data broker",
    ),
)


@dataclass
class Cell:
    """All samples for one (pair, register) condition."""

    presences: list[float] = field(default_factory=list)
    registers: list[str] = field(default_factory=list)
    failures: int = 0

    @property
    def mean(self) -> float:
        return statistics.mean(self.presences) if self.presences else 0.0

    @property
    def spread(self) -> float:
        """Sample standard deviation — the run-to-run noise floor for this cell."""
        return statistics.stdev(self.presences) if len(self.presences) > 1 else 0.0


@dataclass
class ProbeResult:
    model: str
    repeats: int
    state: dict[str, Cell] = field(default_factory=dict)     # topic -> cell
    private: dict[str, Cell] = field(default_factory=dict)

    def _all(self, side: dict[str, Cell]) -> list[float]:
        return [p for cell in side.values() for p in cell.presences]

    @property
    def state_mean(self) -> float:
        values = self._all(self.state)
        return statistics.mean(values) if values else 0.0

    @property
    def private_mean(self) -> float:
        values = self._all(self.private)
        return statistics.mean(values) if values else 0.0

    @property
    def gap(self) -> float:
        """Positive means state-actor items scored higher."""
        return self.state_mean - self.private_mean

    @property
    def noise(self) -> float:
        """Pooled within-condition spread — how much a cell moves on repetition.

        The comparison that matters: a gap below this is indistinguishable from
        the model's own sampling variance, which cannot be reduced because
        ``temperature`` is rejected on current models.
        """
        spreads = [c.spread for c in (*self.state.values(), *self.private.values())
                   if len(c.presences) > 1]
        return statistics.mean(spreads) if spreads else 0.0

    def concentration(self, factor: float = 2.0) -> tuple[list[str], list[str]]:
        """Split topics into those carrying the gap and those within noise.

        Returns ``(carrying, evenhanded)``, thresholded at ``factor`` x the noise
        floor. This exists because the aggregate gap can be a badly misleading
        summary: a mean of +0.06 reads as "everything tilts slightly" when the
        real shape was four topics at +0.115 and six indistinguishable from
        zero. Those two worlds call for different responses — a uniform tilt
        invites a calibration offset, a concentrated one means a flat offset
        would overcorrect most of the set.

        Guarded on ``repeats``, not on the noise floor. A pooled noise of exactly
        0.0 is legitimate — the presence values are coarse, so a model can return
        the same number on every repeat of a cell — and it means the opposite of
        "cannot classify": with no sampling variance to attribute a difference
        to, a reproducible gap is as certain as this probe gets. Guarding on
        ``noise <= 0`` skipped the analysis in exactly the case it was most
        confident about, while ``format_report`` went on interpreting the
        magnitude regardless.

        With a zero threshold the comparison has to be strict: ``>=`` would
        classify a topic whose gap is exactly 0.0 as carrying, which is
        backwards.
        """
        if self.repeats < 2:
            return [], []
        threshold = factor * self.noise
        carrying, evenhanded = [], []
        for topic in self.state:
            gap = abs(self.topic_gap(topic))
            (carrying if gap > 0 and gap >= threshold else evenhanded).append(topic)
        return carrying, evenhanded

    def topic_gap(self, topic: str) -> float:
        return self.state[topic].mean - self.private[topic].mean

    def register_accuracy(self, side: dict[str, Cell], expected: str) -> float:
        """Fraction of samples labelled with the register their actor implies."""
        labels = [r for cell in side.values() for r in cell.registers]
        if not labels:
            return 0.0
        # "both" is a defensible reading of a single-actor sentence, so it counts
        # as correct; the failure being tested for is labelling a private-power
        # item as state coercion or vice versa.
        return sum(1 for r in labels if r in (expected, "both")) / len(labels)

    @property
    def failures(self) -> int:
        return sum(c.failures for c in (*self.state.values(), *self.private.values()))


def run_probe(tagger, pairs=PAIRS, repeats: int = 3, progress=None) -> ProbeResult:
    """Score every pair on both sides, ``repeats`` times each."""
    result = ProbeResult(model=getattr(tagger, "model", "unknown"), repeats=repeats)
    for pair in pairs:
        state_cell, private_cell = Cell(), Cell()
        for _ in range(repeats):
            for text, cell in ((pair.state_text(), state_cell),
                               (pair.private_text(), private_cell)):
                score = tagger.score(text)
                if score is None:
                    cell.failures += 1
                    continue
                cell.presences.append(score.presence)
                cell.registers.append(score.register)
        result.state[pair.topic] = state_cell
        result.private[pair.topic] = private_cell
        if progress:
            progress(pair)
    return result


def _wrap(text: str) -> list[str]:
    """Indent and wrap a paragraph to terminal width.

    The interpretive notes are the part most likely to be skimmed, and an
    unwrapped 300-character line is the easiest thing in a report to skip.
    """
    return textwrap.wrap(" ".join(text.split()), width=78,
                         initial_indent="  ", subsequent_indent="  ")


def format_report(result: ProbeResult) -> str:
    lines = [
        f"Liberty register probe — {result.model}, {result.repeats} repeat(s) per cell",
        "",
        f"  {'topic':<20}{'state':>8}{'private':>10}{'gap':>8}",
        "  " + "-" * 46,
    ]
    for topic in result.state:
        s, p = result.state[topic], result.private[topic]
        lines.append(f"  {topic:<20}{s.mean:>8.2f}{p.mean:>10.2f}{s.mean - p.mean:>+8.2f}")

    lines += [
        "",
        "1. Register classification (the categorical check)",
        f"     state-actor items labelled {STATE}: "
        f"{result.register_accuracy(result.state, STATE):.0%}",
        f"     private-actor items labelled {PRIVATE}: "
        f"{result.register_accuracy(result.private, PRIVATE):.0%}",
        "",
        "2. Presence magnitude (the subtle check)",
        f"     state mean   {result.state_mean:.3f}",
        f"     private mean {result.private_mean:.3f}",
        f"     gap          {result.gap:+.3f}   (positive = state scores higher)",
        f"     noise floor  {result.noise:.3f}   (mean within-cell spread on repetition)",
        "",
    ]

    carrying, evenhanded = result.concentration()
    if carrying and evenhanded:
        hot = statistics.mean(result.topic_gap(t) for t in carrying)
        cool = statistics.mean(result.topic_gap(t) for t in evenhanded)
        lines += _wrap(
            f"Concentrated, not uniform: {len(carrying)} of "
            f"{len(carrying) + len(evenhanded)} topics carry the gap "
            f"(mean {hot:+.3f}), while {len(evenhanded)} sit within noise "
            f"({cool:+.3f}). Carrying: {', '.join(sorted(carrying))}. The "
            f"aggregate above averages those together, so read it as a summary "
            f"of two different behaviours rather than one small tilt — and note "
            f"that a single calibration offset would overcorrect the "
            f"{len(evenhanded)} that are already even."
        )
        lines.append("")

    if result.failures:
        lines.append(f"  {result.failures} call(s) returned no verdict and were dropped.")
        lines.append("")

    if result.repeats < 2:
        lines += _wrap(
            "No noise floor: with one repeat per cell there is nothing to compare the "
            "gap against. Re-run with --repeats 3 or more before reading the magnitude."
        )
    elif abs(result.gap) <= result.noise:
        lines += _wrap(
            "The gap is within the model's own run-to-run variance. That is not "
            "evidence of evenhandedness — it is absence of evidence either way, at "
            "this sample size."
        )
    else:
        favoured = "state" if result.gap > 0 else "private-power"
        if result.noise == 0:
            # "Exceeds the noise floor" is true but reads as uninformative when
            # the floor is zero, and a floor of exactly zero across every cell is
            # itself worth remarking on.
            lines += _wrap(
                "Every cell returned the same value on every repeat, so the noise "
                "floor is exactly zero. Any gap below is therefore reproducible "
                "rather than sampling variance — but check the model is not "
                "returning canned values before reading much into the size."
            )
            lines.append("")
        lines += _wrap(
            f"The gap exceeds the noise floor, leaning toward {favoured} framing. "
            "Descriptive only — this is a handful of samples per cell, not a "
            "significance test. Re-run with more repeats before acting on it, and if "
            "it holds, treat it as a property of the rubric rather than of the diets."
        )
    lines.append("")
    lines += _wrap(
        "Pairs differ only in the actor phrase; everything else is identical by "
        "construction (see PAIRS in this module)."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from scoring.liberty import DEFAULT_MODEL, build_tagger

    parser = argparse.ArgumentParser(
        prog="validation.register_probe",
        description="Check whether the liberty rubric scores both registers evenhandedly",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id")
    parser.add_argument("--repeats", type=int, default=3,
                        help="samples per condition (default 3; 1 disables the noise floor)")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args(argv)

    calls = len(PAIRS) * 2 * max(1, args.repeats)
    print(f"{len(PAIRS)} pairs x 2 registers x {args.repeats} repeats = {calls} calls "
          f"to {args.model}.")
    if not args.yes:
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Cancelled.")
            return 1

    tagger = build_tagger(model=args.model, use_batch=False)
    if tagger is None:
        return 1        # build_tagger already warned with the reason

    done = [0]

    def progress(pair: ProbePair) -> None:
        done[0] += 1
        print(f"  [{done[0]}/{len(PAIRS)}] {pair.topic}", flush=True)

    print()
    result = run_probe(tagger, repeats=args.repeats, progress=progress)
    print()
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
