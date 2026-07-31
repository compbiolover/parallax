"""Prompt construction for the daily diet summaries.

Kept separate from the API call so the rubric is unit-testable without a network
round-trip. The system prompt encodes the project's non-negotiable tone rules
(``CLAUDE.md`` §0): charitable understanding, steelman each side, symmetry, no
pathologizing, and explicit uncertainty about the (currently demo-grade) scores.

It also encodes how the prose should read, which is not decoration: the
executive summary is the part of the brief that gets read on a phone before
anything else, and a wall of clause-stacked, em-dash-jointed text is one that
gets skimmed. The style rules below are the concrete tells from the humanizer
guidance, written as instructions rather than as a preference.

They are instructions and nothing else. Nothing scrubs the generated text
afterwards, deliberately: swapping an em dash for a comma moves where the
sentence breaks, and a second pass rewriting prose whose claims were never
verified is one more place for the summary to stop matching the numbers it
describes. If a rule stops holding, the fix is here.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = """You are the summarization voice of Parallax, a tool that \
compares two media diets through Moral Foundations Theory.

Non-negotiable rules:
- CHARITABLE UNDERSTANDING. Steelman each diet's framing. The binding \
foundations (loyalty, authority, sanctity) are sincere moral commitments, not \
deficits. Never mock, pathologize, or "dunk on" either diet.
- SYMMETRY. Treat both diets identically. The author's own diet ("self") gets \
the same scrutiny and its blindspots the same prominence as the modeled diet.
- UNCERTAINTY IS FIRST-CLASS. The foundation numbers are noisy estimates from a \
dictionary method over the configured lexicon (stated in the data below). \
Describe tendencies, never certainties. Do not overclaim.
- GROUND CLAIMS in the supplied headlines and numbers. Do not invent stories, \
quotes, or figures that are not in the input.
- NAME EACH DIET by the label given below. The machine ids ("self", \
"modeled_ce") are database keys and must never appear in the prose.

How to write it:
- Calm, plain prose. No bullet lists of grievances, no partisan adjectives.
- Lead with the finding. The first sentence of each section says what is true; \
the sentences after it say why you think so.
- Short paragraphs, two to four sentences each, separated by a blank line. One \
long block is the thing people stop reading.
- Vary sentence length. Several long clause-stacked sentences in a row read as \
machine output no matter how accurate they are.
- No em dashes or en dashes. Use a period, a comma, a colon, or parentheses.
- Do not force ideas into groups of three. Two examples are usually enough, and \
four is fine when there are four.
- Plain verbs. Prefer "is" and "has" over "serves as", "stands as", \
"represents". Avoid: underscores, highlights, reflects a broader, testament, \
landscape, tapestry, interplay, pivotal, crucial, delve, vibrant, showcase.
- No boldface, no headings beyond the ones asked for, no closing flourish. End \
on the last real observation."""


@dataclass(frozen=True)
class DietContext:
    diet_id: str
    label: str
    doc_count: int
    profile: dict[str, float]        # composition, sums to 1
    headlines: list[str]


@dataclass(frozen=True)
class ComparisonContext:
    diet_a: str
    diet_b: str
    jsd: float
    log_ratios: dict[str, float]


def _fmt_profile(profile: dict[str, float]) -> str:
    ranked = sorted(profile.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{f}={v:.2f}" for f, v in ranked)


def build_user_prompt(
    contexts: list[DietContext],
    comparison: ComparisonContext | None,
    max_headlines: int = 20,
    lexicon: str | None = None,
) -> str:
    """Assemble the data block Claude summarizes."""
    parts: list[str] = []
    if lexicon:
        note = f"Scores were produced by the '{lexicon}' lexicon."
        if lexicon == "built-in demo seed":
            note += " This is a DEMO lexicon — treat differences as illustrative."
        parts.append(note + "\n")
    parts += [
        "Summarize today's coverage for each media diet below, then write a "
        "cross-diet executive summary.\n",
        "For EACH diet, write one short paragraph headed exactly "
        "`## <label>`, using the diet's label verbatim as the heading. Say what "
        "that diet morally emphasized today and why a thoughtful person holding "
        "those foundations would see it that way.\n",
        "Then write the executive summary, headed exactly `## Executive`, in "
        "two or three short paragraphs. Open with one sentence a reader could "
        "stop after: how far apart the two diets are today and on what. Then "
        "what each foregrounds that the other does not. Close on what the "
        "numbers cannot support, in plain words rather than a hedge stacked on "
        "a hedge.\n",
    ]
    for ctx in contexts:
        parts.append(f"\n### DATA — {ctx.label}")
        parts.append(f"documents today: {ctx.doc_count}")
        parts.append(f"foundation emphasis (composition): {_fmt_profile(ctx.profile)}")
        if ctx.headlines:
            shown = ctx.headlines[:max_headlines]
            parts.append("sample headlines:")
            parts.extend(f"  - {h}" for h in shown)
    if comparison is not None:
        # Labels here too, not ids. The log-ratio line is the one sentence in
        # the data block a model is most likely to paraphrase directly, so an
        # id in it comes back out in the prose.
        labels = {c.diet_id: c.label for c in contexts}
        a = labels.get(comparison.diet_a, comparison.diet_a)
        b = labels.get(comparison.diet_b, comparison.diet_b)
        parts.append(f"\n### DATA — comparison ({a} vs {b})")
        parts.append(
            f"Jensen-Shannon divergence: {comparison.jsd:.3f} "
            "(0 = identical emphasis, 1 = disjoint)"
        )
        lr = ", ".join(
            f"{f}={v:+.2f}" for f, v in sorted(comparison.log_ratios.items())
        )
        parts.append(
            f"per-foundation log-ratio (positive = {a} over-indexes vs {b}): {lr}"
        )
    return "\n".join(parts)
