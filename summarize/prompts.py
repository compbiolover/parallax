"""Prompt construction for the daily diet summaries.

Kept separate from the API call so the rubric is unit-testable without a network
round-trip. The system prompt encodes the project's non-negotiable tone rules
(``CLAUDE.md`` §0): charitable understanding, steelman each side, symmetry, no
pathologizing, and explicit uncertainty about the (currently demo-grade) scores.
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

Write in calm, plain prose. No bullet lists of grievances, no partisan adjectives."""


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
        "`## <label>` describing what that diet morally emphasized today and "
        "why a thoughtful person holding those foundations would see it that "
        "way. Then a final paragraph headed exactly `## Executive` naming what "
        "each diet foregrounds that the other does not — with explicit "
        "uncertainty language.\n",
    ]
    for ctx in contexts:
        parts.append(f"\n### DATA — {ctx.label} (id: {ctx.diet_id})")
        parts.append(f"documents today: {ctx.doc_count}")
        parts.append(f"foundation emphasis (composition): {_fmt_profile(ctx.profile)}")
        if ctx.headlines:
            shown = ctx.headlines[:max_headlines]
            parts.append("sample headlines:")
            parts.extend(f"  - {h}" for h in shown)
    if comparison is not None:
        parts.append(
            f"\n### DATA — comparison ({comparison.diet_a} vs {comparison.diet_b})"
        )
        parts.append(
            f"Jensen-Shannon divergence: {comparison.jsd:.3f} "
            "(0 = identical emphasis, 1 = disjoint)"
        )
        lr = ", ".join(
            f"{f}={v:+.2f}" for f, v in sorted(comparison.log_ratios.items())
        )
        parts.append(
            f"per-foundation log-ratio (positive = {comparison.diet_a} "
            f"over-indexes vs {comparison.diet_b}): {lr}"
        )
    return "\n".join(parts)
