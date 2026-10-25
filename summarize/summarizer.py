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

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from compare.divergence import jensen_shannon_divergence, log_ratios
from compare.weekly import MIN_WEEK_DOCS
from ingestion.datastore import Datastore
from ingestion.pipeline import persona_profiles
from scoring.claude_client import NO_TEXT, UNKNOWN, build_client, call_failed

from .prompts import (
    SYSTEM_PROMPT,
    ComparisonContext,
    DietContext,
    ThreadContext,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"


# The same week the clustering and the weekly comparison use.
SUMMARY_WINDOW_DAYS = 7


# How many story threads the review covers. Six is about what fits in an email
# before the reader stops, and the prompt orders them by coverage, so the cut
# is at the bottom of the list rather than in the middle of the day.
MAX_REVIEW_THREADS = 6

# Timeline points per thread handed to the model. Two weeks is long enough for
# "third day of coverage" and short enough that the prompt does not become
# mostly history.
THREAD_TIMELINE_DAYS = 14


@dataclass
class SummaryResult:
    per_diet: dict[str, str]
    executive: str
    model: str
    method: str  # 'claude' | 'deterministic'
    generated_utc: str
    # The story review. `stories` is {thread_id: prose}; `stories_intro` is the
    # paragraph above the per-story sections, and `themes` the motifs across
    # them. All empty on the deterministic path, which has no prose to give.
    stories: dict[str, str] = field(default_factory=dict)
    stories_intro: str = ""
    themes: str = ""
    titles: dict[str, str] = field(default_factory=dict)


def gather(
    store: Datastore,
    registry,
    pair,
    max_headlines: int = 50,
    personas: list[str] | None = None,
) -> tuple[list[DietContext], ComparisonContext | None]:
    """Pull per-diet contexts and a pairwise comparison from the datastore.

    The label is the registry's human one ("Modeled conservative-evangelical
    diet"), recorded at ingestion. Passing the machine id as the label is what
    put "modeled_ce" and "the self diet" in prose written for a person to read.
    """
    # This week, not the whole store: an all-time profile barely moves as the
    # corpus grows, so the prose it fed said the same thing every morning. A
    # week too thin for the weekly brief to compare (MIN_WEEK_DOCS a side) falls
    # back to all of it, so the prose and the numbers beside it agree on a period.
    since = (datetime.now(UTC).date() - timedelta(days=SUMMARY_WINDOW_DAYS - 1)).isoformat()
    profiles = persona_profiles(store, registry, since=since)
    thin = any(
        store.doc_count_for_sources(registry.weights_for(p), since=since) < MIN_WEEK_DOCS
        for p in pair
    )
    if thin or not all(p in profiles for p in pair):
        since = None
        profiles = persona_profiles(store, registry)
    labels = store.diet_labels()
    short_labels = store.diet_short_labels()
    # The reference pair by default, not every persona. One call carries every
    # context in its prompt and every section in its reply, and MAX_TOKENS caps
    # thinking and prose together — so persona count buys prompt size and a
    # higher chance of the empty-prose failure this file already guards against,
    # not extra calls. Widen it deliberately with `summarize.personas: all`.
    # `None` -> the pair; an explicit empty list -> every persona (that is what
    # `summarize.personas: all` resolves to); otherwise exactly what was asked for.
    if personas is None:
        wanted = list(pair.ids)
    elif not personas:
        wanted = registry.persona_ids()
    else:
        wanted = personas
    contexts: list[DietContext] = []
    for persona_id in wanted:
        if persona_id not in profiles:
            continue
        weights = registry.weights_for(persona_id)
        contexts.append(
            DietContext(
                diet_id=persona_id,
                label=labels.get(persona_id) or persona_id,
                short_label=short_labels.get(persona_id, ""),
                doc_count=store.doc_count_for_sources(weights, since=since),
                profile=profiles[persona_id],
                headlines=store.headlines_for_sources(weights, max_headlines, since=since),
                period="this week" if since else "across the whole archive",
            )
        )
    comparison = None
    scored = {c.diet_id for c in contexts}
    if pair.mine in scored and pair.theirs in scored:
        a, b = pair.mine, pair.theirs
        comparison = ComparisonContext(
            diet_a=a,
            diet_b=b,
            jsd=jensen_shannon_divergence(profiles[a], profiles[b]),
            log_ratios=log_ratios(profiles[a], profiles[b]),
        )
    return contexts, comparison


def gather_threads(store: Datastore, limit: int = MAX_REVIEW_THREADS) -> list[ThreadContext]:
    """The day's biggest story threads, with their running timelines.

    Empty is a normal answer, not a failure: a store whose cluster step has
    never run has no threads, and the brief simply has no story review in it.
    """
    day = store.latest_thread_day()
    if not day:
        return []
    out: list[ThreadContext] = []
    for row in store.threads_active_on(day, limit):
        try:
            diet_counts = json.loads(row["diet_counts"] or "{}")
        except (TypeError, ValueError):
            diet_counts = {}
        timeline = [
            (r["day"], r["doc_count"], r["headline"] or "")
            for r in store.thread_timeline(row["thread_id"], THREAD_TIMELINE_DAYS)
        ]
        headlines = [m["title"] for m in store.thread_members(row["thread_id"]) if m["title"]]
        out.append(
            ThreadContext(
                thread_id=row["thread_id"],
                title=row["title"],
                doc_count=row["doc_count"],
                diet_counts=diet_counts,
                timeline=timeline,
                headlines=headlines,
                theme_title=row["theme_title"] or "",
                first_seen=row["first_seen"],
            )
        )
    return out


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# Room for the brief *and* for thinking, which shares this budget. Opus 5
# thinks by default — a request that omits the `thinking` parameter runs
# adaptive thinking, where Opus 4.8 ran without it — and `max_tokens` caps
# thinking plus response text together. At the old 1500 the reasoning consumed
# the budget on a real corpus and the text block came back empty: a successful
# call, an empty summary, and a brief with no prose in it.
MAX_TOKENS = 16000

# Thinking depth. Not the default `high`: this is a bounded writing task over
# numbers already computed, and low is strong on Opus 5 — it is the lever for
# cost and latency here, and the daily run pays it every morning.
#
# `low` is also the reliable setting, not just the cheap one. Measured on the
# live corpus, `medium` spent ~1000 tokens thinking and then ended the turn
# with no prose in 3 of 4 calls; `low` thought for under 25 tokens and wrote
# the brief 5 times out of 5. Raise this only with that in mind, and only
# together with a check that the prose still arrives.
DEFAULT_EFFORT = "low"


class Summarizer:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: object | None = None,
        effort: str = DEFAULT_EFFORT,
    ) -> None:
        self.model = model
        self.effort = effort
        self._client = client  # inject for testing; else built lazily from env

    def summarize(
        self, store: Datastore, registry, pair, personas: list[str] | None = None
    ) -> SummaryResult:
        contexts, comparison = gather(store, registry, pair, personas=personas)
        lexicon = store.get_meta("lexicon")
        threads = gather_threads(store)
        if not contexts:
            return SummaryResult({}, "", self.model, "deterministic", _now_iso())

        # `is not None`, not truthiness: an injected client is a sentinel, and a
        # test double that defines __bool__ or __len__ would otherwise be
        # silently discarded in favour of a client built from the environment.
        injected = self._client is not None
        client, reason = (self._client, UNKNOWN) if injected else _build_client()
        if client is None:
            return self._deterministic(contexts, comparison, lexicon, reason)
        try:
            text = self._call_claude(client, contexts, comparison, lexicon, threads)
            if not text.strip():
                # Opus 5 sometimes ends the turn on a thinking block alone: a
                # 200, `end_turn`, no `max_tokens`, and no prose. It is a coin
                # flip on the same input rather than a property of the prompt,
                # so one more ask is the whole fix. Observed on the live corpus:
                # 1/4 calls at `effort: medium` returned prose, 5/5 at `low` —
                # hence `DEFAULT_EFFORT` above, with this as the guard for the
                # days the model thinks itself out of answering anyway.
                logger.warning("Claude returned no summary text — asking once more")
                text = self._call_claude(client, contexts, comparison, lexicon, threads)
        except Exception as exc:
            # Never let an API hiccup leave the dashboard empty.
            logger.warning(
                "Claude summaries failed (%s: %s) — using the deterministic fallback",
                type(exc).__name__,
                exc,
            )
            return self._deterministic(contexts, comparison, lexicon, call_failed(exc))
        if not text.strip():
            # A response with no usable text is a failure, whatever the HTTP
            # status said. Persisting it wrote empty summaries over the day's
            # brief and reported success; the fallback is what "never leave the
            # dashboard empty" meant.
            logger.warning("Claude returned no summary text — using the deterministic fallback")
            return self._deterministic(contexts, comparison, lexicon, NO_TEXT)
        per_diet, executive, review = _parse_sections(text, contexts, threads)
        missing = [c.diet_id for c in contexts if not per_diet.get(c.diet_id, "").strip()]
        if missing:
            # A partial parse is a real state — a section can be missing because
            # the model skipped it. It is also what a broken heading matcher
            # looks like, and that failure is silent: the diet's panel just
            # stops appearing. Say it out loud rather than inferring it later
            # from a brief that lost a panel.
            logger.warning(
                "No summary section parsed for %s — the model may have used "
                "headings that do not name those diets",
                ", ".join(missing),
            )
        return SummaryResult(
            per_diet,
            executive,
            self.model,
            "claude",
            _now_iso(),
            stories=review.stories,
            stories_intro=review.intro,
            themes=review.themes,
            titles=review.titles,
        )

    def _call_claude(self, client, contexts, comparison, lexicon, threads=None) -> str:
        user = build_user_prompt(contexts, comparison, lexicon=lexicon, threads=threads)
        resp = client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            output_config={"effort": self.effort},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        return _response_text(resp)

    def _deterministic(
        self, contexts, comparison, lexicon=None, reason: str = UNKNOWN
    ) -> SummaryResult:
        note = _fallback_note(reason)
        per_diet = {c.diet_id: _deterministic_diet(c, note) for c in contexts}
        executive = _deterministic_executive(contexts, comparison, lexicon, note)
        return SummaryResult(per_diet, executive, self.model, "deterministic", _now_iso())

    def persist(self, store: Datastore, result: SummaryResult) -> None:
        for diet_id, text in result.per_diet.items():
            store.upsert_summary(
                scope=diet_id,
                generated_utc=result.generated_utc,
                model=result.model,
                method=result.method,
                text=text,
            )
        store.upsert_summary(
            scope="executive",
            generated_utc=result.generated_utc,
            model=result.model,
            method=result.method,
            text=result.executive,
        )
        for scope, text in (("stories", result.stories_intro), ("themes", result.themes)):
            if text.strip():
                store.upsert_summary(
                    scope=scope,
                    generated_utc=result.generated_utc,
                    model=result.model,
                    method=result.method,
                    text=text,
                )
        # Each story's prose goes to two places on purpose. The summaries row is
        # what the dashboard and the email read today; the thread_days note is
        # what makes the timeline readable *next* week, when today is a line in
        # the middle of it and this row has been overwritten by a later day.
        day = store.latest_thread_day()
        for thread_id, text in result.stories.items():
            if not text.strip():
                continue
            store.upsert_summary(
                scope=f"thread:{thread_id}",
                generated_utc=result.generated_utc,
                model=result.model,
                method=result.method,
                text=text,
            )
            if day:
                store.set_thread_note(thread_id, day, _first_sentence(text))
            title = result.titles.get(thread_id, "")
            if title:
                store.set_thread_review_title(thread_id, title)


def _first_sentence(text: str, limit: int = 240) -> str:
    """The lead sentence, for a timeline row that has one line to spend.

    A timeline is read down the left-hand side, so the note has to carry its
    point in the first few words; the full paragraph stays in the summaries
    row for the surfaces that have room for it.
    """
    flat = " ".join(text.split())
    m = re.search(r"(?<=[.!?])\s", flat)
    sentence = flat[: m.start()].rstrip() if m else flat
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "\u2026"


def _response_text(resp) -> str:
    """The prose out of a response, or "" with a logged reason.

    Three things can leave a successful call with nothing to print, and all
    three used to be indistinguishable from a summary that happened to be
    blank:

    * the safety classifiers declined (``stop_reason: "refusal"``), which is an
      HTTP 200 with empty or partial content — so ``stop_reason`` has to be
      read *before* the content, not after;
    * thinking consumed the token budget (``max_tokens``), leaving no room for
      the answer;
    * the response carried only thinking blocks, whose text is empty by default
      on this model family.
    """
    stop = getattr(resp, "stop_reason", None)
    if stop == "refusal":
        logger.warning("Claude declined to summarize (safety classifiers)")
        return ""
    # Prose blocks only. A block that declares a type has to declare `text`;
    # the `hasattr` path is for blocks that carry no type at all. Accepting any
    # block with a `.text` attribute regardless of type — which is what the
    # first clause used to permit — would splice a future non-prose block into
    # the summary, and the failure would read as the model writing nonsense.
    text = "".join(
        getattr(block, "text", "") or ""
        for block in getattr(resp, "content", None) or []
        if getattr(block, "type", "text") == "text" and hasattr(block, "text")
    )
    if stop == "max_tokens":
        # Truncated. Partial prose still beats no prose — the brief degrades to
        # a summary that stops early rather than to silence — but the cap is
        # the thing to fix, so say so.
        logger.warning(
            "Claude hit max_tokens (%d) — the summary may stop "
            "mid-sentence; raise MAX_TOKENS or lower the effort",
            MAX_TOKENS,
        )
    return text


def _build_client():
    """``(client, reason)`` — the reason empty on success, and warned about
    otherwise so the log names which piece of setup is missing.

    Falling back to the deterministic summary is a legitimate mode, but it is
    labelled on the dashboard as "numbers-only" — so an unintended fallback is
    visible in the output while its *cause* was not. The warning closes that
    gap for anyone reading the log; the reason travels on so the page can close
    it for anyone reading only the page.
    """
    client, reason = build_client()
    if client is None:
        logger.warning("Claude summaries disabled, using the deterministic fallback: %s", reason)
    return client, reason


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


_BRACKETED = re.compile(r"\[([a-z0-9_]+)\]\s*$")


def _bracketed_id(head: str, known: set[str]) -> str | None:
    """The ``[persona_id]`` the prompt asks for, when it is there and is real.

    An id the model invented is ignored rather than trusted, so a hallucinated
    bracket degrades to the token matcher instead of creating a phantom section.
    """
    m = _BRACKETED.search(head)
    if m is None:
        return None
    candidate = m.group(1)
    return candidate if candidate in known else None


def _heading_table(contexts) -> list[tuple[frozenset[str], str]]:
    """Every name a diet answers to: its label, its short label, its id."""
    table: list[tuple[frozenset[str], str]] = []
    for ctx in contexts:
        for name in (ctx.label, getattr(ctx, "short_label", ""), ctx.diet_id):
            tokens = _heading_tokens(name or "")
            if tokens:
                table.append((tokens, ctx.diet_id))
    return table


@dataclass
class _Review:
    """The story review, split out of the model's reply."""

    intro: str = ""
    stories: dict[str, str] = field(default_factory=dict)
    themes: str = ""
    # The story name the review gave each thread, from its heading. The stored
    # cluster label is a keyword readout ("supreme · trump · court"); this is
    # the line a reader is shown.
    titles: dict[str, str] = field(default_factory=dict)


# `### [thread:france-protests] France protests` -> the id. Tolerant of the
# model dropping the brackets or the prefix, because the alternative to a
# tolerant match is a silently missing story section.
_THREAD_HEAD = re.compile(r"(?:\[\s*)?thread:\s*([a-z0-9][a-z0-9-]*)", re.IGNORECASE)


def _heading_title(head: str) -> str:
    """The story name out of `### [thread:france-protests] France protests`."""
    stripped = _THREAD_HEAD.sub("", head, count=1)
    return stripped.strip(" []:-\u2014").strip()


def _thread_id(head: str, known: set[str]) -> str | None:
    m = _THREAD_HEAD.search(head)
    if m and m.group(1).lower() in known:
        return m.group(1).lower()
    return None


def _parse_sections(text: str, contexts, threads=None) -> tuple[dict[str, str], str, _Review]:
    """Split Claude's ``## <label> [<id>]`` / ``## Executive`` sections back apart.

    The bracketed id is tried first and the token matcher is the fallback. Token
    overlap alone was a lucky parser rather than a robust one: it works for two
    diets whose labels share no vocabulary, and mis-assigns as soon as a library
    of personas has labels like "Movement-media reader" and "Party-mainstream
    reader" in it. The failure is silent — a persona's panel simply stops
    appearing — which is why the prompt now asks for the id and this reads it.
    """
    table = _heading_table(contexts)
    known = {ctx.diet_id for ctx in contexts}
    thread_ids = {t.thread_id.lower() for t in (threads or [])}
    heading_titles: dict[str, str] = {}
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
            thread = _thread_id(head, thread_ids)
            # Thread first: a story heading carries a title that can also look
            # like a diet label ("My diet's week on pensions"), and the id is
            # the unambiguous part.
            if thread:
                current = f"thread:{thread}"
                heading_titles[thread] = _heading_title(m.group(1).strip())
            elif head.startswith("exec"):
                current = "executive"
            elif head.startswith("stories") or head.startswith("the stories"):
                current = "stories"
            elif head.startswith("themes") or head.startswith("motifs"):
                current = "themes"
            else:
                current = _bracketed_id(head, known) or _match_heading(head, table) or head
        else:
            buf.append(line)
    flush()

    executive = sections.pop("executive", "")
    review = _Review(
        intro=sections.pop("stories", ""),
        themes=sections.pop("themes", ""),
        stories={
            tid: sections.pop(f"thread:{tid}")
            for tid in sorted(thread_ids)
            if sections.get(f"thread:{tid}", "").strip()
        },
        titles={tid: title for tid, title in heading_titles.items() if title},
    )
    per_diet = {c.diet_id: sections.get(c.diet_id, "") for c in contexts}
    if not executive and not any(per_diet.values()) and not review.stories:
        executive = text.strip()  # unparseable -> keep whole thing
    if thread_ids and not review.stories:
        # Same failure mode as a missing diet panel, and just as silent: the
        # brief loses its lead section and nothing says why.
        logger.warning(
            "No story sections parsed — the model may not have used the "
            "`### [thread:<id>]` headings the prompt asked for"
        )
    return per_diet, executive, review


# -- deterministic fallback -------------------------------------------------


def _fallback_note(reason: str) -> str:
    """The line on the dashboard that says why there is no LLM prose today.

    It used to name a missing key unconditionally, which is true for one of the
    ways the fallback is reached and a false lead for the rest: a missing
    `anthropic` package, a key that is set but rejected, a call that timed out,
    a response with no text in it. Reading "no ANTHROPIC_API_KEY set" sends you
    to re-export a key that was never the problem, so the cause is passed in
    rather than assumed.
    """
    short = getattr(reason, "short", None) or str(reason) or UNKNOWN.short
    return f"(Generated without the LLM — {short}. Neutral, numbers-only.)"


def _top_two(profile: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(profile.items(), key=lambda kv: kv[1], reverse=True)[:2]


def _deterministic_diet(ctx: DietContext, note: str) -> str:
    top = _top_two(ctx.profile)
    emphasis = " and ".join(f"{f} ({v:.2f})" for f, v in top)
    return (
        f"{note}\n\n"
        f"Across {ctx.doc_count} stories, this diet's strongest moral-foundation "
        f"emphasis was {emphasis}. These are estimates from a dictionary method "
        f"and should be read as tendencies, not measurements."
    )


def _deterministic_executive(contexts, comparison, lexicon=None, note: str = "") -> str:
    from scoring.lexicon import is_demo_lexicon

    note = note or _fallback_note(UNKNOWN)
    if comparison is None:
        return f"{note}\n\nOnly one diet has scored documents; no comparison yet."
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
            f"Scores were produced by the {lexicon} lexicon; treat differences as estimates."
        )
    return (
        f"{note}\n\n"
        f"Jensen-Shannon divergence between {a} and {b} is {comparison.jsd:.3f} "
        f"(0 = identical emphasis, 1 = disjoint).\n\n"
        f"Relative to {b}, {a} over-indexes most on {a_over[0]} "
        f"({a_over[1]:+.2f} log-ratio) and under-indexes most on {b_over[0]} "
        f"({b_over[1]:+.2f}). {provenance}"
    )
