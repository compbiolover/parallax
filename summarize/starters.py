"""Conversation starters: one question per blindspot direction, written by Claude.

The digest picks the story (``cluster.themes.lead_story``); this writes the
question so the wording varies with the story instead of repeating one sentence.
Any failure returns nothing and the digest falls back to its fixed wording.
"""

from __future__ import annotations

import json
import logging

from cluster.themes import lead_story

from .prompts import SYSTEM_PROMPT
from .summarizer import MAX_TOKENS, _response_text

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 300

INSTRUCTIONS = """Write one conversation starter per item below. Each item is a story \
one modeled media environment covered heavily and the other barely covered.

- One or two sentences, ending in a question mark, specific to the story.
- Ask about how the story looks from inside the covering environment, or what its \
coverage shows about what that environment treats as worth leading with. Vary the \
angle and the opening words across items.
- Describe coverage only. Attribute no belief, motive or feeling to anyone, and \
take no side on the story itself.
- Do not ask the reader to defend or attack either environment.

Reply with JSON only, no prose: {"starters": [{"id": 0, "question": "..."}]}\
"""


def pick(themes: list[dict]) -> list[dict]:
    """One lead story per dominant diet, from ``Theme.to_dict()`` dicts."""
    by_diet: dict[str, list[dict]] = {}
    for theme in themes:
        by_diet.setdefault(theme.get("dominant_diet") or "?", []).append(theme)
    out = []
    for dominant, group in by_diet.items():
        lead = lead_story(group)
        if lead:
            theme, story = lead
            out.append(
                {
                    "dominant": dominant,
                    "quiet": theme.get("other_diet") or "?",
                    "theme": theme.get("title") or "Other coverage",
                    "headline": story["title"],
                }
            )
    return out


def write_starters(
    client, model: str, effort: str, themes: list[dict], labels: dict[str, str]
) -> dict[str, dict]:
    """``{dominant_diet: {"headline", "question"}}``, empty on any failure."""
    items = pick(themes)
    if not items:
        return {}
    lines = [INSTRUCTIONS, ""]
    for i, item in enumerate(items):
        covered = labels.get(item["dominant"], item["dominant"])
        quiet = labels.get(item["quiet"], item["quiet"])
        lines.append(
            f"[{i}] {covered} covered heavily, {quiet} barely did. "
            f'Theme: {item["theme"]}. Headline: "{item["headline"]}"'
        )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            output_config={"effort": effort},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n".join(lines)}],
        )
        text = _response_text(resp).strip().removeprefix("```json").removesuffix("```")
        rows = json.loads(text)["starters"]
        questions = {r["id"]: r["question"] for r in rows}
    except Exception as exc:
        logger.warning("Conversation starters skipped (%s: %s)", type(exc).__name__, exc)
        return {}
    return {
        item["dominant"]: {"headline": item["headline"], "question": q.strip()}
        for i, item in enumerate(items)
        if isinstance(q := questions.get(i), str)
        and q.strip().endswith("?")
        and len(q) <= MAX_QUESTION_CHARS
    }
