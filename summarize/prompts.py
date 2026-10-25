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
compares modeled media diets through Moral Foundations Theory.

Non-negotiable rules:
- A DIET IS NOT A PERSON. Each label below names a *modeled pattern of media \
consumption* — a weighting over outlets and programs — not a reader, a group, or \
a demographic. Write about what the coverage emphasized. Never attribute beliefs, \
motives, education, intelligence, sincerity, fears, or emotional states to a \
diet, or to "people who" consume it. If a sentence would be condescending or \
defamatory said to a real person who matched the label, do not write it.
- NEVER PSYCHOLOGIZE. Do not explain a diet's emphasis by what its audience \
lacks, fears, or has been told. Explain it by what its sources covered.
- DO NOT RANK. These diets are not points on a spectrum and not milder or more \
extreme versions of one another. The numbers do not support an ordering, so do \
not supply one.
- CHARITABLE UNDERSTANDING. Steelman each diet's framing. The binding \
foundations (loyalty, authority, sanctity) are sincere moral commitments, not \
deficits. Never mock, pathologize, or "dunk on" either diet.
- SYMMETRY. Treat every diet identically. The reader's own diet gets the same \
scrutiny, and its blindspots the same prominence, as the one it is compared with.
- UNCERTAINTY IS FIRST-CLASS. The foundation numbers are noisy estimates from a \
dictionary method over the configured lexicon (stated in the data below). \
Describe tendencies, never certainties. Do not overclaim.
- GROUND CLAIMS in the supplied headlines and numbers. Do not invent stories, \
quotes, or figures that are not in the input.
- NAME EACH DIET by the label given below. Machine ids like "self" or \
"modeled_ce" are database keys and must never appear in the prose.

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
on the last real observation.

Writing the story review:
- You are writing the lead review in a daily brief, for one reader who follows \
the news closely. Assume they have seen the headlines. Tell them what the \
coverage did with them.
- A story section earns its place by saying something a headline does not: what \
moved today against what stood yesterday, which outlets picked it up and which \
stayed quiet, and where the two diets' framings actually diverge rather than \
merely differ in volume.
- Where a timeline is supplied, use it. "Third day of coverage, and the first \
on which the modeled diet picked it up" is the kind of sentence only the \
timeline can support, and it is the most useful sentence in the section.
- Name the motif a story sits in only when the story genuinely belongs to one, \
and say what the motif is doing: a recurring frame, a sustained attention gap, \
a subject one diet treats as ongoing and the other as episodic.
- Quote headlines verbatim when you characterize framing, in quotation marks. \
A claim about how something was framed is checkable only against the words.
- No invented detail. You have headlines, counts, and a timeline. You do not \
have article bodies, so do not write as though you read them. Where the \
coverage is too thin to review, say that in one sentence instead of padding."""


@dataclass(frozen=True)
class DietContext:
    diet_id: str
    label: str
    doc_count: int
    profile: dict[str, float]  # composition, sums to 1
    headlines: list[str]
    # The registry's short form, if it has one. Not shown to the model — it is
    # given the full label to write with — but a heading that comes back in the
    # short form is still this diet, and the parser needs to know that.
    short_label: str = ""
    # What the counts and headlines cover: "this week", or the whole archive
    # when the week had nothing for one side.
    period: str = "this week"


@dataclass(frozen=True)
class ThreadContext:
    """One running story, with the history that makes a review possible.

    ``timeline`` is ``[(day, doc_count, headline), ...]`` oldest first. It is
    the whole reason threads are persisted: everything else here could be read
    off a single day's clustering, and none of it supports a sentence about
    what changed.
    """

    thread_id: str
    title: str
    doc_count: int
    diet_counts: dict[str, int]
    timeline: list[tuple[str, int, str]]
    headlines: list[str]
    theme_title: str = ""
    first_seen: str = ""

    @property
    def days_running(self) -> int:
        return len(self.timeline)


@dataclass(frozen=True)
class ComparisonContext:
    diet_a: str
    diet_b: str
    jsd: float
    log_ratios: dict[str, float]


def _fmt_profile(profile: dict[str, float]) -> str:
    ranked = sorted(profile.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{f}={v:.2f}" for f, v in ranked)


def _fmt_timeline(ctx: ThreadContext) -> list[str]:
    lines = []
    for day, count, headline in ctx.timeline:
        head = f" — {headline}" if headline else ""
        lines.append(f"    {day}: {count} documents{head}")
    return lines


def build_thread_block(threads: list[ThreadContext], labels: dict[str, str]) -> list[str]:
    """The data block the story review is written from.

    Diet counts are printed with the human label, not the persona id: every
    number in this block is a candidate for direct paraphrase, and an id in the
    data comes back out in the prose.
    """
    parts: list[str] = []
    for ctx in threads:
        parts.append(f"\n### DATA — story [thread:{ctx.thread_id}]")
        parts.append(f"current title: {ctx.title}")
        if ctx.theme_title:
            parts.append(f"motif: {ctx.theme_title}")
        parts.append(f"documents today: {ctx.doc_count}")
        if ctx.first_seen:
            parts.append(f"first covered: {ctx.first_seen} ({ctx.days_running} days with coverage)")
        if ctx.diet_counts:
            shown = ", ".join(f"{labels.get(d, d)}={n}" for d, n in sorted(ctx.diet_counts.items()))
            parts.append(f"coverage by diet today: {shown}")
        if ctx.timeline:
            parts.append("timeline (oldest first):")
            parts.extend(_fmt_timeline(ctx))
        if ctx.headlines:
            parts.append("headlines in this story:")
            parts.extend(f"    - {h}" for h in ctx.headlines)
    return parts


def build_user_prompt(
    contexts: list[DietContext],
    comparison: ComparisonContext | None,
    max_headlines: int = 20,
    lexicon: str | None = None,
    threads: list[ThreadContext] | None = None,
) -> str:
    """Assemble the data block Claude summarizes."""
    period = contexts[0].period if contexts else "this week"
    labels = {c.diet_id: c.label for c in contexts}
    parts: list[str] = []
    if lexicon:
        note = f"Scores were produced by the '{lexicon}' lexicon."
        if lexicon == "built-in demo seed":
            note += " This is a DEMO lexicon — treat differences as illustrative."
        parts.append(note + "\n")
    if threads:
        parts += [
            "Review the day's biggest stories, then summarize each media diet, then "
            "write a cross-diet executive summary.\n",
            "FIRST, the story review, headed exactly `## Stories`. Open with one "
            "short paragraph naming the two or three stories that dominated and the "
            "motifs they sit in. Then one section per story below, headed exactly "
            "`### [thread:<id>] <title>` with the id verbatim from its data block — "
            "the id is how the section is matched back to the story, so it must be "
            "exact. The title after it is yours to write: a short, specific name "
            "for the story in the brief's own voice, six to nine words, not a "
            "copy of any outlet's headline and not the keyword list in the data. "
            "It is the line the reader sees above the section. Two or three short "
            "paragraphs each: what moved, how the diets "
            "covered it differently, and what the timeline shows. Review the stories "
            "in the order given; they are ordered by how much coverage they drew.\n",
            "THEN a section headed exactly `## Themes`: two short paragraphs on the "
            "motifs running across today's stories. A motif is a frame or an "
            "attention pattern that shows up in more than one story. If the day's "
            "stories share nothing, say so plainly rather than inventing a thread "
            "between them.\n",
        ]
    parts += [
        f"Summarize the coverage below ({period}) for each media diet, then write a "
        "cross-diet executive summary.\n",
        "For EACH diet, write one short paragraph headed exactly "
        "`## <label> [<id>]`, using the diet's label and id verbatim from the "
        "data blocks below. The bracketed id is how the section is matched back "
        "to the diet, so it must be exact. Say what that diet morally emphasized "
        f"{period} and why a thoughtful person holding those foundations would see it "
        "that way.\n",
        "Then write the executive summary, headed exactly `## Executive`, in "
        "two or three short paragraphs. Open with one sentence a reader could "
        f"stop after: how far apart the two compared diets are {period} and on what. Then "
        "what each foregrounds that the other does not. Close on what the "
        "numbers cannot support, in plain words rather than a hedge stacked on "
        "a hedge.\n",
    ]
    if threads:
        parts += build_thread_block(threads, labels)
    for ctx in contexts:
        parts.append(f"\n### DATA — {ctx.label} [{ctx.diet_id}]")
        parts.append(f"documents ({ctx.period}): {ctx.doc_count}")
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
        lr = ", ".join(f"{f}={v:+.2f}" for f, v in sorted(comparison.log_ratios.items()))
        parts.append(f"per-foundation log-ratio (positive = {a} over-indexes vs {b}): {lr}")
    return "\n".join(parts)
