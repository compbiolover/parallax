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

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from compare.divergence import jensen_shannon_divergence, log_ratios
from ingestion.datastore import Datastore
from ingestion.pipeline import diet_profiles
from scoring.claude_client import build_client

from .prompts import (
    SYSTEM_PROMPT,
    ComparisonContext,
    DietContext,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"


@dataclass
class SummaryResult:
    per_diet: dict[str, str]
    executive: str
    model: str
    method: str  # 'claude' | 'deterministic'
    generated_utc: str


def gather(
    store: Datastore, max_headlines: int = 50
) -> tuple[list[DietContext], ComparisonContext | None]:
    """Pull per-diet contexts and a pairwise comparison from the datastore.

    The label is the registry's human one ("Modeled conservative-evangelical
    diet"), recorded at ingestion. Passing the machine id as the label is what
    put "modeled_ce" and "the self diet" in prose written for a person to read.
    """
    profiles = diet_profiles(store)
    labels = store.diet_labels()
    short_labels = store.diet_short_labels()
    contexts: list[DietContext] = []
    for diet_id in store.diet_ids():
        if diet_id not in profiles:
            continue
        contexts.append(
            DietContext(
                diet_id=diet_id,
                label=labels.get(diet_id) or diet_id,
                short_label=short_labels.get(diet_id, ""),
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
    return datetime.now(UTC).isoformat()


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
    """The Claude client, or ``None`` with a warning explaining which piece of
    setup is missing.

    Falling back to the deterministic summary is a legitimate mode, but it is
    labelled on the dashboard as "numbers-only" — so an unintended fallback is
    visible in the output while its *cause* was not. The warning closes that gap.
    """
    client, reason = build_client()
    if client is None:
        logger.warning("Claude summaries disabled, using the deterministic "
                       "fallback: %s", reason)
    return client


# Words a heading can gain or lose without naming a different diet. Articles
# only: "my" is the whole of what distinguishes "My diet", and dropping it left
# that label normalized to "diet", which every other label ends with.
_HEADING_NOISE = frozenset({"the", "a", "an"})


def _heading_tokens(text: str) -> frozenset[str]:
    """A heading as a bag of words, punctuation and articles removed."""
    lowered = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return frozenset(w for w in lowered.split() if w not in _HEADING_NOISE)


def _match_heading(head: str, table: list[tuple[frozenset[str], str]]) -> str | None:
    """The diet a heading names, or ``None`` when that is not clear.

    Headings are matched against the diet labels, and the labels are now
    sentences ("Modeled conservative-evangelical diet") rather than tokens. A
    model that title-cases one, drops the trailing noun, or writes an en dash
    where the registry has a hyphen has still named the right diet, and an
    exact-match table would drop the section on the floor.

    Matching is by token subset in either direction: a heading that is part of a
    label ("Modeled conservative-evangelical") and one that is a label plus
    trimming ("My diet today") both land. Substring containment does not work
    here, and not subtly: under it the heading "The modeled diet" matched "My
    diet" reduced to "diet", and a section about the modeled diet was filed
    under the author's own — which inverts the one guarantee this tool makes.

    A heading that fits two diets equally well ("Diet") returns ``None`` rather
    than the first one scanned. Dropping a section is recoverable; attributing
    it to the wrong diet is not.
    """
    tokens = _heading_tokens(head)
    if not tokens:
        return None
    best: str | None = None
    best_score = 0
    tied = False
    for label_tokens, diet_id in table:
        if not (label_tokens <= tokens or tokens <= label_tokens):
            continue
        score = len(label_tokens & tokens)
        if score > best_score:
            best, best_score, tied = diet_id, score, False
        elif score == best_score and diet_id != best:
            tied = True
    return None if tied else best


def _heading_table(contexts) -> list[tuple[frozenset[str], str]]:
    """Every name a diet answers to: its label, its short label, its id."""
    table: list[tuple[frozenset[str], str]] = []
    for ctx in contexts:
        for name in (ctx.label, getattr(ctx, "short_label", ""), ctx.diet_id):
            tokens = _heading_tokens(name or "")
            if tokens:
                table.append((tokens, ctx.diet_id))
    return table


def _parse_sections(text: str, contexts) -> tuple[dict[str, str], str]:
    """Split Claude's ``## <label>`` / ``## Executive`` sections back apart."""
    table = _heading_table(contexts)
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
            current = ("executive" if head.startswith("exec")
                       else _match_heading(head, table) or head)
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
    # The fallback names the diets the same way the model is told to: this text
    # is read by the same person on the same morning, and "modeled_ce" is a
    # database key either way.
    labels = {c.diet_id: c.label for c in contexts}
    a = labels.get(comparison.diet_a, comparison.diet_a)
    b = labels.get(comparison.diet_b, comparison.diet_b)
    over = sorted(comparison.log_ratios.items(), key=lambda kv: kv[1], reverse=True)
    a_over = over[0]
    b_over = over[-1]
    if is_demo_lexicon(lexicon):
        provenance = "Differences at this scale are provisional given the demo lexicon."
    else:
        provenance = (
            f"Scores were produced by the {lexicon} lexicon; "
            "treat differences as estimates."
        )
    return (
        f"{_FALLBACK_NOTE}\n\n"
        f"Jensen-Shannon divergence between {a} and {b} is {comparison.jsd:.3f} "
        f"(0 = identical emphasis, 1 = disjoint).\n\n"
        f"Relative to {b}, {a} over-indexes most on {a_over[0]} "
        f"({a_over[1]:+.2f} log-ratio) and under-indexes most on {b_over[0]} "
        f"({b_over[1]:+.2f}). {provenance}"
    )
