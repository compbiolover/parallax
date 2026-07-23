"""Generate daily per-diet and cross-diet summaries.

Map-reduce at Phase 1 is shallow: there are no clusters yet (that's Phase 2), so
the "map" is the day's headlines + the diet's foundation profile, and the
"reduce" is a charitable paragraph per diet plus one cross-diet executive
summary. Claude does the reduce when ``ANTHROPIC_API_KEY`` is set; otherwise a
deterministic, clearly-labeled fallback composes a neutral summary from the
numbers so the pipeline and dashboard stay populated without a key.

The raw article text is long gone by this stage (scored and discarded at
ingestion), so summaries are grounded in headlines and metrics — not verbatim
body quotes. Auditable cluster-level quotes arrive with Phase 2.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from compare.divergence import jensen_shannon_divergence, log_ratios
from ingestion.datastore import Datastore
from ingestion.pipeline import diet_profiles

from .prompts import (
    SYSTEM_PROMPT,
    ComparisonContext,
    DietContext,
    build_user_prompt,
)

DEFAULT_MODEL = "claude-opus-4-8"


@dataclass
class SummaryResult:
    per_diet: dict[str, str]
    executive: str
    model: str
    method: str  # 'claude' | 'deterministic'
    generated_utc: str


def gather(store: Datastore, max_headlines: int = 50) -> tuple[list[DietContext], ComparisonContext | None]:
    """Pull per-diet contexts and a pairwise comparison from the datastore."""
    profiles = diet_profiles(store)
    contexts: list[DietContext] = []
    for diet_id in store.diet_ids():
        if diet_id not in profiles:
            continue
        contexts.append(
            DietContext(
                diet_id=diet_id,
                label=diet_id,
                doc_count=store.doc_count(diet_id),
                profile=profiles[diet_id],
                headlines=store.headlines_for_diet(diet_id, max_headlines),
            )
        )
    comparison = None
    scored = [c.diet_id for c in contexts]
    if len(scored) >= 2:
        a, b = sorted(scored)[:2]
        comparison = ComparisonContext(
            diet_a=a,
            diet_b=b,
            jsd=jensen_shannon_divergence(profiles[a], profiles[b]),
            log_ratios=log_ratios(profiles[a], profiles[b]),
        )
    return contexts, comparison


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Summarizer:
    def __init__(self, model: str = DEFAULT_MODEL, client: object | None = None) -> None:
        self.model = model
        self._client = client  # inject for testing; else built lazily from env

    def summarize(self, store: Datastore) -> SummaryResult:
        contexts, comparison = gather(store)
        lexicon = store.get_meta("lexicon")
        if not contexts:
            return SummaryResult({}, "", self.model, "deterministic", _now_iso())

        client = self._client or _build_client()
        if client is None:
            return self._deterministic(contexts, comparison, lexicon)
        try:
            text = self._call_claude(client, contexts, comparison, lexicon)
        except Exception:
            # Never let an API hiccup leave the dashboard empty.
            return self._deterministic(contexts, comparison, lexicon)
        per_diet, executive = _parse_sections(text, contexts)
        return SummaryResult(per_diet, executive, self.model, "claude", _now_iso())

    def _call_claude(self, client, contexts, comparison, lexicon) -> str:
        user = build_user_prompt(contexts, comparison, lexicon=lexicon)
        resp = client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))

    def _deterministic(self, contexts, comparison, lexicon=None) -> SummaryResult:
        per_diet = {c.diet_id: _deterministic_diet(c) for c in contexts}
        executive = _deterministic_executive(contexts, comparison, lexicon)
        return SummaryResult(per_diet, executive, self.model, "deterministic", _now_iso())

    def persist(self, store: Datastore, result: SummaryResult) -> None:
        for diet_id, text in result.per_diet.items():
            store.upsert_summary(
                scope=diet_id, generated_utc=result.generated_utc,
                model=result.model, method=result.method, text=text,
            )
        store.upsert_summary(
            scope="executive", generated_utc=result.generated_utc,
            model=result.model, method=result.method, text=result.executive,
        )


def _build_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic()


def _parse_sections(text: str, contexts) -> tuple[dict[str, str], str]:
    """Split Claude's ``## <label>`` / ``## Executive`` sections back apart."""
    label_to_id = {c.label.lower(): c.diet_id for c in contexts}
    label_to_id.update({c.diet_id.lower(): c.diet_id for c in contexts})
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is not None:
            sections[current] = "\n".join(buf).strip()

    for line in text.splitlines():
        m = re.match(r"^\s*#{1,3}\s*(.+?)\s*$", line)
        if m:
            flush()
            buf = []
            head = m.group(1).strip().lower()
            current = "executive" if head.startswith("exec") else label_to_id.get(head, head)
        else:
            buf.append(line)
    flush()

    executive = sections.pop("executive", "")
    per_diet = {c.diet_id: sections.get(c.diet_id, "") for c in contexts}
    if not executive and not any(per_diet.values()):
        executive = text.strip()  # unparseable -> keep whole thing
    return per_diet, executive


# -- deterministic fallback -------------------------------------------------

_FALLBACK_NOTE = "(Generated without the LLM — no ANTHROPIC_API_KEY set. Neutral, numbers-only.)"


def _top_two(profile: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(profile.items(), key=lambda kv: kv[1], reverse=True)[:2]


def _deterministic_diet(ctx: DietContext) -> str:
    top = _top_two(ctx.profile)
    emphasis = " and ".join(f"{f} ({v:.2f})" for f, v in top)
    return (
        f"{_FALLBACK_NOTE}\n\n"
        f"Across {ctx.doc_count} stories, this diet's strongest moral-foundation "
        f"emphasis was {emphasis}. These are estimates from a dictionary method "
        f"and should be read as tendencies, not measurements."
    )


def _deterministic_executive(contexts, comparison, lexicon=None) -> str:
    from scoring.lexicon import is_demo_lexicon

    if comparison is None:
        return f"{_FALLBACK_NOTE}\n\nOnly one diet has scored documents; no comparison yet."
    over = sorted(comparison.log_ratios.items(), key=lambda kv: kv[1], reverse=True)
    a_over = over[0]
    b_over = over[-1]
    if is_demo_lexicon(lexicon):
        provenance = "Differences at this scale are provisional given the demo lexicon."
    else:
        provenance = f"Scores were produced by the {lexicon} lexicon; treat differences as estimates."
    return (
        f"{_FALLBACK_NOTE}\n\n"
        f"Jensen-Shannon divergence between {comparison.diet_a} and "
        f"{comparison.diet_b} is {comparison.jsd:.3f} (0 = identical emphasis, "
        f"1 = disjoint). Relative to {comparison.diet_b}, {comparison.diet_a} "
        f"over-indexes most on {a_over[0]} ({a_over[1]:+.2f} log-ratio) and "
        f"under-indexes most on {b_over[0]} ({b_over[1]:+.2f}). {provenance}"
    )
