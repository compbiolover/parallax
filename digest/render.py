"""Payload -> email. Same content as the dashboard, no JavaScript.

Reads the payload ``dashboard/export.build_payload`` produces, and in the daily
run reads the *same dict object* the dashboard was written from, so the two
cannot describe different days. That is a weaker claim than "cannot drift":
they are two renderings of one payload, and keeping them saying the same thing
about it is a matter of maintenance, not of construction. Where they diverged
once already — the page plotting ``band.point`` while this read the raw
dictionary profile — the fix was to read the same field, not to hope.

Every section degrades to nothing when its data is absent. An unscored
foundation, a missing summary, a day with no clusters, and a fairness split
with no evidence behind it all render as silence rather than as a zero, which
is the rule the rest of the codebase follows for "unscored is not the same as
none".
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Matches the dashboard's light palette so the two readings look like one tool.
# Named by position, not by whose diet they are: the payload's diet order is
# `ORDER BY diet_id`, so which diet lands first is alphabetical accident. An
# earlier version called these SELF/OTHER and assigned them by list index while
# colouring blindspots by the `own_diet` setting — under the shipped ids
# (`modeled_ce` sorts before `self`) the same diet came out blue in one panel
# and orange in another.
DIET_A = "#2f6fb0"
DIET_B = "#c46a2e"
INK = "#1a1d24"
MUTED = "#5c6270"
LINE = "#e2e5ea"
PANEL = "#ffffff"
CARD = "#fbfcfd"          # a theme card, one step off the panel it sits on
BG = "#f7f8fa"

MAX_WIDTH = 600
SPARK_DAYS = 21           # ~3 weeks reads clearly at phone width

# How much of the blindspot section a phone gets. The section it replaces ran a
# card per cluster and a couple of dozen clusters a day, which is a scroll
# nobody finishes — and an unfinished section is one where the author's own
# blindspots, whichever half they land in, go unread. Capping per *direction*
# rather than in total is what keeps the two halves comparable when one diet
# happens to have a noisier day; what the caps leave out is named, not dropped
# silently.
MAX_CARDS_PER_DIRECTION = 3
STORIES_PER_CARD = 3

# Column widths for the diverging bar chart, as percentages of a fixed-layout
# table. Percentages rather than pixels because the narrowest phone still in use
# is 320px CSS wide, and a fixed half-width wide enough to be readable on a
# desktop overflows there — an email that scrolls sideways is one you stop
# opening.
RATIO_COLS = ("28%", "26%", "26%", "20%")


@dataclass(frozen=True)
class Digest:
    """A rendered brief, ready to hand to a sender."""

    subject: str
    html: str
    text: str
    preheader: str


# -- small helpers ----------------------------------------------------------


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _label(diet_id: str) -> str:
    """``modeled_ce`` -> ``Modeled CE``. A fallback, not the first choice.

    ``build_payload`` emits a ``label`` per diet and ``sources.yaml`` can carry
    a human one; ``_diet_label`` prefers it and falls back here. Short tokens
    stay upper-cased because the ids use initialisms and plain title-casing
    renders them as words ("Modeled Ce"), which reads like a typo in the one
    place the reader is deciding whether to trust the numbers. It is a
    heuristic — ``npr_wsj`` still comes out "Npr Wsj", which is what the
    ``label`` field is for.
    """
    words = diet_id.replace("_", " ").split()
    return " ".join(w.upper() if len(w) <= 2 else w.capitalize() for w in words)


def _diet_label(diet: dict) -> str:
    """What a heading or legend calls this diet.

    Prefers the registry's short label, then its full one, then the id
    prettified. The short label exists because the full one is written to read
    as a noun phrase inside a sentence ("Modeled conservative-evangelical
    diet"), and a phrase that reads well in prose wraps to two lines in a
    legend.
    """
    for key in ("short_label", "label"):
        value = (diet.get(key) or "").strip()
        if value and value != diet.get("id"):
            return value
    return _label(diet["id"])


def _named(payload: dict, diet_id: str) -> str:
    """The same name, for the panels that hold an id rather than a diet dict.

    Without this the blindspot headings and the log-ratio legend fell back to
    the id prettifier while the panel above them used the registry's label, so
    one email called the same diet two things.
    """
    for diet in payload.get("diets") or []:
        if diet.get("id") == diet_id:
            return _diet_label(diet)
    return _label(diet_id)


def _signed(value: float, places: int = 3) -> str:
    return f"{value:+.{places}f}"


def _delta(history: list[dict]) -> tuple[float | None, str, str | None]:
    """Change in the headline divergence since the previous recorded point.

    Reads ``jsd_cumulative``, not ``jsd_window``. The headline number is
    ``comparison.jsd``, which ``dashboard/export`` computes over the whole
    corpus, and ``compare/history`` is explicit that the cumulative and the
    trailing-window series answer different questions. Subtracting one from the
    other produced a movement figure that did not belong to the number printed
    above it — including a sign that could disagree, since the cumulative basis
    is damped by design.

    Returns ``(delta, reason, since_date)``. ``delta`` is ``None`` when there is
    nothing to compare against, because a first-ever snapshot printing
    ``+0.000`` reads as "nothing moved" rather than "nothing to move from".
    ``since_date`` is the date the comparison is *against* — snapshots are one
    row per UTC date but nothing makes them consecutive, so "since yesterday"
    is a claim the data does not support and the date is printed instead.
    """
    usable = [p for p in history if p.get("jsd_cumulative") is not None]
    if not usable:
        return None, "no divergence recorded yet", None
    if len(usable) < 2:
        return None, "first recorded point — nothing to compare against", None
    previous, latest = usable[-2], usable[-1]
    return (latest["jsd_cumulative"] - previous["jsd_cumulative"],
            "", previous.get("date"))


def _headline(payload: dict) -> float | None:
    comparison = payload.get("comparison") or {}
    return comparison.get("jsd")


def _pair(payload: dict) -> tuple[str, str] | None:
    comparison = payload.get("comparison") or {}
    pair = comparison.get("pair")
    return (pair[0], pair[1]) if pair and len(pair) == 2 else None


def _check_own_diet(payload: dict, own: str | None) -> None:
    """Warn when ``own_diet`` names a diet the payload does not have.

    A typo silently disabled the ordering: `_own_first` returned the list
    unchanged and nothing said why. The house idiom everywhere else is a logged
    reason for a degradation, and the ids are right there to name.
    """
    if not own:
        return
    ids = [d["id"] for d in payload.get("diets") or []]
    if ids and own not in ids:
        logger.warning(
            "digest.own_diet=%r matches no diet in the payload (have: %s) — your "
            "own blindspots will not be ordered first", own, ", ".join(ids))


def _colours(payload: dict) -> dict[str, str]:
    """One diet -> colour map, built once and used by every panel.

    The dashboard does the same thing (`colors[di.id]`), and it is the only way
    the colours stay consistent: keying one panel off list index and another off
    the ``own_diet`` setting lets them disagree about which diet is which.
    """
    ids = [d["id"] for d in payload.get("diets") or []]
    return {diet_id: (DIET_A, DIET_B)[i % 2] for i, diet_id in enumerate(ids)}


def _own_first(blindspots: list[dict], own: str | None) -> list[dict]:
    """Order blindspots so the author's own come first.

    ``CLAUDE.md`` §5 asks for the author's blindspots at equal prominence. On a
    phone, equal styling is not enough — whatever is second gets scrolled past.
    A blindspot whose *dominant* diet is the modeled one is a story the author
    is missing, so those lead.
    """
    if not own:
        return blindspots
    mine = [b for b in blindspots if b.get("other_diet") == own]
    theirs = [b for b in blindspots if b.get("other_diet") != own]
    return mine + theirs


# -- HTML fragments ---------------------------------------------------------


def _panel(title: str, body: str) -> str:
    return (
        f'<tr><td style="padding:0 0 14px;">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:{PANEL};border:1px solid {LINE};border-radius:12px;">'
        f'<tr><td style="padding:16px 18px;">'
        f'<div style="font:600 11px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,'
        f'sans-serif;text-transform:uppercase;letter-spacing:.07em;color:{MUTED};'
        f'padding-bottom:10px;">{_esc(title)}</div>{body}'
        f"</td></tr></table></td></tr>"
    )


def _bar_row(label: str, value: float, share: float, colour: str,
             spread: tuple[float, float] | None = None) -> str:
    """One horizontal bar. ``share`` is 0..1 of the full width.

    ``spread`` is the dictionary-vs-transformer interval. It is printed rather
    than drawn: whiskers need geometry email cannot do, but the number is the
    part that matters — a wide interval means the two methods disagree, so
    trust that foundation less (``CLAUDE.md`` §5).
    """
    pct = max(0.0, min(1.0, share)) * 100
    band = ""
    if spread and spread[0] is not None and spread[1] is not None:
        band = (f'<span style="color:{MUTED};font-size:11px;"> '
                f"{spread[0]:.2f}–{spread[1]:.2f}</span>")
    return (
        f'<tr>'
        f'<td style="font:13px -apple-system,sans-serif;color:{INK};'
        f'padding:3px 8px 3px 0;white-space:nowrap;">{_esc(label)}</td>'
        f'<td style="width:100%;padding:3px 0;">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>'
        f'<td style="background:{colour};height:9px;border-radius:5px;width:{pct:.1f}%;'
        f'font-size:0;line-height:0;">&nbsp;</td>'
        f'<td style="font-size:0;line-height:0;">&nbsp;</td>'
        f"</tr></table></td>"
        f'<td style="font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:{MUTED};'
        f'padding:3px 0 3px 8px;text-align:right;white-space:nowrap;">'
        f"{value:.3f}{band}</td>"
        f"</tr>"
    )


def _foundation(diet: dict, foundation: str) -> tuple[float, tuple[float, float] | None]:
    """One foundation's value and its confidence interval, if there is one.

    Prefers the ensemble point estimate over the dictionary-only share, which
    is what the dashboard plots (``radarValue``). Reading the raw profile here
    while the page reads ``band.point`` put a different number on each surface
    for the same foundation, once the transformer had ever run.
    """
    band = (diet.get("band") or {}).get(foundation)
    if band and band.get("point") is not None:
        return band["point"], (band.get("low"), band.get("high"))
    return (diet.get("profile") or {}).get(foundation, 0.0), None


def _composition_panel(payload: dict) -> str:
    """Both diets' foundation compositions — the radar chart, unrolled.

    Two stacked bar groups rather than an overlay: a radar needs geometry email
    cannot draw, and reading two aligned lists is if anything easier on a
    narrow screen than reading a polygon.
    """
    diets = payload.get("diets") or []
    if not diets:
        return ""
    foundations = payload.get("foundations") or []
    colours = _colours(payload)
    blocks = []
    for i, diet in enumerate(diets):
        colour = colours[diet["id"]]
        values = {f: _foundation(diet, f) for f in foundations}
        peak = max([v[0] for v in values.values()] or [0.0]) or 1.0
        rows = "".join(
            _bar_row(f, values[f][0], values[f][0] / peak, colour, spread=values[f][1])
            for f in foundations
        )
        blocks.append(
            f'<div style="font:600 13px -apple-system,sans-serif;color:{colour};'
            f'padding:{"0" if i == 0 else "14px"} 0 6px;">{_esc(_diet_label(diet))}'
            f'<span style="font-weight:400;color:{MUTED};"> · '
            f'{diet.get("doc_count", 0)} docs</span></div>'
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0">'
            f"{rows}</table>"
        )
    note = (
        f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};padding-top:10px;">'
        f"Shares of each diet's moral vocabulary, summing to 1. Bars are scaled to each "
        f"diet's own largest foundation, so compare the numbers across diets, not the "
        f"bar lengths.</div>"
    )
    return _panel("Foundation composition", "".join(blocks) + note)


def _log_ratio_panel(payload: dict) -> str:
    """Per-foundation over/under-indexing — the diverging bar chart."""
    comparison = payload.get("comparison") or {}
    ratios = comparison.get("log_ratios") or {}
    pair = _pair(payload)
    if not ratios or not pair:
        return ""
    widest = max((abs(v) for v in ratios.values()), default=0.0) or 1.0
    a, b = pair
    colours = _colours(payload)

    label_w, left_w, right_w, value_w = RATIO_COLS
    rows = []
    for foundation, value in ratios.items():
        pct = min(1.0, abs(value) / widest) * 100
        colour = colours[a] if value > 0 else colours[b]
        bar = (
            f'<div style="height:9px;border-radius:5px;background:{colour};'
            f'width:{pct:.1f}%;font-size:0;line-height:0;">&nbsp;</div>'
        )
        # The left half is right-aligned so both halves grow away from the
        # centre line rather than from the page edges.
        left = (
            f'<td style="width:{left_w};" align="right">'
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0">'
            f'<tr><td style="font-size:0;">&nbsp;</td>'
            f'<td style="width:{pct:.1f}%;">{bar}</td></tr></table></td>'
            if value <= 0 else f'<td style="width:{left_w};"></td>'
        )
        right = (
            f'<td style="width:{right_w};">{bar}</td>' if value > 0
            else f'<td style="width:{right_w};"></td>'
        )
        rows.append(
            f'<tr><td style="width:{label_w};font:13px -apple-system,sans-serif;'
            f'color:{INK};padding:3px 6px 3px 0;">{_esc(foundation)}</td>'
            f"{left}{right}"
            f'<td style="width:{value_w};font:12px ui-monospace,Menlo,monospace;'
            f'color:{MUTED};padding:3px 0 3px 6px;text-align:right;">'
            f"{_signed(value, 2)}</td></tr>"
        )

    note = (
        f'<div style="font:12px/1.5 -apple-system,sans-serif;color:{MUTED};padding-top:10px;">'
        f'<span style="color:{colours[a]};">&#9632;</span> right = {_esc(_named(payload, a))} '
        f'over-indexes &nbsp; <span style="color:{colours[b]};">&#9632;</span> left = '
        f"{_esc(_named(payload, b))} does. Natural log of the ratio of shares; 0 is parity.</div>"
    )
    return _panel(
        "Who leans on which foundation",
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="table-layout:fixed;">{"".join(rows)}</table>{note}',
    )


def _sparkline_panel(payload: dict) -> str:
    """Divergence over time, as fixed-height columns."""
    history = [p for p in (payload.get("history") or []) if p.get("jsd_window") is not None]
    if len(history) < 2:
        return ""
    points = history[-SPARK_DAYS:]
    peak = max(p["jsd_window"] for p in points) or 1.0
    cells = []
    for point in points:
        height = max(2, round(48 * point["jsd_window"] / peak))
        cells.append(
            f'<td style="vertical-align:bottom;padding:0 1px;">'
            f'<div style="background:{DIET_A};height:{height}px;border-radius:2px;'
            f'font-size:0;">&nbsp;</div></td>'
        )
    window = payload.get("history_window_days")
    note = (
        f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};padding-top:8px;">'
        f"{points[0]['date']} &rarr; {points[-1]['date']}"
        f"{f', {window}-day trailing window' if window else ''}. "
        f"Peak {peak:.3f}.</div>"
    )
    return _panel(
        "Divergence over time",
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="height:48px;"><tr>{"".join(cells)}</tr></table>{note}',
    )


def _themes(payload: dict) -> list[dict]:
    """The blindspot themes, derived on the spot if the payload predates them.

    A payload exported before theming carries clusters and no themes. Grouping
    them here — with the same function the cluster run uses, not a second copy
    of the rule — keeps an old payload readable rather than making the email
    fall back to the cluster labels this redesign exists to stop showing.
    """
    themes = payload.get("blindspot_themes")
    if themes:
        return list(themes)
    spots = payload.get("blindspots") or []
    if not spots:
        return []
    try:
        from cluster.themes import group_blindspots
    except ImportError:      # renderer used standalone, without the cluster pkg
        logger.warning("cluster.themes is unavailable — blindspots go unthemed")
        return []
    return [t.to_dict() for t in group_blindspots(spots)]


def _theme_groups(themes: list[dict], own: str | None) -> list[tuple[str, list[dict]]]:
    """Themes bucketed by who covered them, the author's own blindspots first.

    One heading per direction instead of one sentence per card: with three or
    four cards under each heading, repeating "X covered this; Y barely did"
    every time costs a line per card and says nothing new.
    """
    groups: dict[str, list[dict]] = {}
    for theme in _own_first(themes, own):
        groups.setdefault(theme.get("dominant_diet") or "?", []).append(theme)
    return list(groups.items())


def _more_themes(rest: list[dict], escape: bool = True) -> str:
    """"2 more themes here: Sports, Media & speech" — named, not just counted.

    A bare "+2 more" is the one thing this section cannot say: a theme the
    reader is not shown is indistinguishable from a theme that was never found,
    which is the failure mode the whole blindspot idea exists to avoid.
    """
    titles = [t.get("title") or "" for t in rest if t.get("title")]
    named = ", ".join(_esc(t) if escape else t for t in titles)
    plural = "theme" if len(rest) == 1 else "themes"
    return f"{len(rest)} more {plural} here{f': {named}' if named else ''}"


def _theme_card(theme: dict, colour: str) -> str:
    """One theme, as a card. Stacked full width, never side by side.

    A two-across grid is the shape this wants, and it is the shape email cannot
    have: the fixed-width tables that hold a grid together in Outlook are the
    same tables a phone then scales down whole, so a 600px two-column layout
    arrives as unreadably small text rather than as two columns. Compactness
    comes from the caps above instead — fewer cards, fewer headlines each.
    """
    stories = [s for s in (theme.get("stories") or []) if s][:STORIES_PER_CARD]
    story_html = "".join(
        f'<div style="font:13px/1.4 -apple-system,sans-serif;color:{INK};'
        f'padding:2px 0 2px 2px;">· {_esc(s)}</div>' for s in stories
    )
    total = theme.get("story_count") or 0
    remainder = total - len(stories)
    more = (
        f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};padding-top:4px;">'
        f"+{remainder} more</div>" if remainder > 0 else ""
    )
    clusters = theme.get("cluster_count") or 0
    meta = (
        f"{total} {'story' if total == 1 else 'stories'} · "
        f"{clusters} {'cluster' if clusters == 1 else 'clusters'} · "
        f"{100 * (theme.get('one_sided') or 0):.0f}% one-sided"
    )
    return (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:{CARD};border:1px solid {LINE};border-left:3px solid '
        f'{colour};border-radius:10px;margin-bottom:8px;">'
        f'<tr><td style="padding:10px 12px;">'
        f'<div style="font:600 14px -apple-system,BlinkMacSystemFont,sans-serif;'
        f'color:{INK};">{_esc(theme.get("title") or "Other coverage")}</div>'
        f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};'
        f'padding:2px 0 6px;">{meta}</div>'
        f"{story_html}{more}</td></tr></table>"
    )


def _blindspot_panel(payload: dict, own: str | None) -> str:
    """Themes as cards, both directions, the author's own first.

    Blindspots are detected per cluster and read per theme. A cluster is two to
    five stories, so a day produces a couple of dozen of them; listing each one
    under a c-TF-IDF label made a section nobody finishes, titled in a
    vocabulary nobody speaks.
    """
    themes = _themes(payload)
    if not themes:
        return ""
    colours = _colours(payload)
    blocks: list[str] = []
    for dominant, group in _theme_groups(themes, own):
        # Coloured by who covered it, so the colour means the same thing here as
        # in every other panel. "Whose blindspot" is carried by the heading.
        colour = colours.get(dominant, DIET_A)
        missing = _named(payload, group[0].get("other_diet") or "?")
        shown, rest = group[:MAX_CARDS_PER_DIRECTION], group[MAX_CARDS_PER_DIRECTION:]
        # The overflow belongs to its direction, not to the end of the section:
        # a theme the modeled diet covered, listed under both halves, reads as a
        # blindspot of whichever half it was printed nearest.
        overflow = (
            f'<div style="font:12px/1.5 -apple-system,sans-serif;color:{MUTED};'
            f'padding:0 0 2px 2px;">+{_more_themes(rest)}</div>' if rest else ""
        )
        blocks.append(
            f'<div style="font:600 13px -apple-system,sans-serif;color:{colour};'
            f'padding:{"2px" if not blocks else "12px"} 0 6px;">'
            f"{_esc(_named(payload, dominant))} covered · {_esc(missing)} barely did</div>"
            + "".join(_theme_card(t, colour) for t in shown) + overflow
        )
    note = (
        f'<div style="font:12px/1.5 -apple-system,sans-serif;color:{MUTED};padding-top:10px;">'
        f"Both directions are listed. Coverage asymmetry is not a judgement about "
        f"which story mattered more. Themes group clusters by subject; the clusters "
        f"themselves are the measured unit.</div>"
    )
    return _panel("Blindspots by theme", "".join(blocks) + note)


def _paragraphs(text: str, font: str) -> str:
    """Prose as separate paragraphs with air between them.

    The previous rendering was one ``white-space:pre-wrap`` block, which put a
    twelve-sentence executive summary on the screen as a single slab. Blank
    lines in the source become real paragraph breaks; a line break inside a
    paragraph becomes a space, since a hard wrap at the model's line length is
    not a break the reader should see at the phone's.
    """
    blocks = [" ".join(b.split()) for b in re.split(r"\n\s*\n", text) if b.strip()]
    last = len(blocks) - 1
    return "".join(
        f'<div style="font:{font} -apple-system,BlinkMacSystemFont,sans-serif;'
        f'color:{INK};padding-bottom:{0 if i == last else 10}px;">{_esc(b)}</div>'
        for i, b in enumerate(blocks)
    )


def _method_note(payload: dict) -> str:
    method = payload.get("summary_method")
    if not method:
        return ""
    return (
        f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};'
        f'padding-top:12px;">Summaries: {_esc(method)}.</div>'
    )


def _executive_panel(payload: dict) -> str:
    """The cross-diet summary, on its own and one step larger than body text.

    It used to open the "What each diet said" panel, under that panel's title —
    which is a claim about what it is, and the wrong one: it is the paragraph
    about both diets at once, and it is the part most likely to be the whole
    reading on a phone.
    """
    executive = (payload.get("executive_summary") or "").strip()
    if not executive:
        return ""
    body = _paragraphs(executive, "15px/1.65")
    # The provenance note goes under the last panel carrying prose, so it reads
    # as a footnote to the summaries rather than an orphan mid-page.
    if not any((d.get("summary") or "").strip() for d in payload.get("diets") or []):
        body += _method_note(payload)
    return _panel("The short version", body)


def _summary_panel(payload: dict) -> str:
    parts = []
    colours = _colours(payload)
    for diet in payload.get("diets") or []:
        text = (diet.get("summary") or "").strip()
        if not text:
            continue
        colour = colours[diet["id"]]
        parts.append(
            f'<div style="padding-top:{14 if parts else 0}px;">'
            f'<div style="font:600 13px -apple-system,'
            f'sans-serif;color:{colour};padding-bottom:6px;">{_esc(_diet_label(diet))}</div>'
            f'{_paragraphs(text, "14px/1.6")}</div>'
        )
    if not parts:
        # The method note belongs to whichever panel actually rendered prose.
        # On its own it is a footnote to nothing.
        return ""
    return _panel("What each diet said", "".join(parts) + _method_note(payload))


def _extra_panel(payload: dict) -> str:
    """Fairness split and liberty — both partial-coverage, both kept separate.

    Neither enters the composition above, for the reason ``compare/liberty.py``
    documents: folding partial coverage into a composition moves every other
    share as a side effect of coverage. Reporting them apart, with their
    coverage attached, is the honest form.
    """
    blocks = []

    split = payload.get("fairness_split")
    if split:
        rows = []
        for diet_id, values in (split.get("diets") or {}).items():
            # Guard on the document count, not on `equality is None`.
            # `FairnessProfile` returns 0.0/0.0 when there is no fairness mass,
            # so the None check never fired and a diet with nothing to split
            # rendered as "equality 0.00 / proportionality 0.00" — the exact
            # unscored-as-zero the rest of this module refuses to do. The
            # liberty block below already guards this way.
            if not values.get("docs_split"):
                rows.append(
                    f'<div style="font:13px -apple-system,sans-serif;color:{MUTED};'
                    f'padding:2px 0;">{_esc(_named(payload, diet_id))}: not enough split-terms '
                    f"to partition</div>"
                )
                continue
            equality = values.get("equality") or 0.0
            thin = " (thin coverage)" if values.get("thin") else ""
            rows.append(
                f'<div style="font:13px -apple-system,sans-serif;color:{INK};'
                f'padding:2px 0;">{_esc(_named(payload, diet_id))}: equality {equality:.2f} / '
                f"proportionality {values.get('proportionality', 0):.2f}"
                f'<span style="color:{MUTED};"> · '
                f"{100 * values.get('coverage', 0):.0f}% of docs{thin}</span></div>"
            )
        blocks.append(
            f'<div style="font:600 13px -apple-system,sans-serif;color:{INK};'
            f'padding-bottom:4px;">Fairness: equality vs proportionality</div>'
            + "".join(rows)
        )

    liberty = payload.get("liberty")
    if liberty:
        rows = []
        for diet_id, values in (liberty.get("diets") or {}).items():
            if not values.get("docs_scored"):
                continue
            thin = " (thin coverage)" if values.get("thin") else ""
            rows.append(
                f'<div style="font:13px -apple-system,sans-serif;color:{INK};'
                f'padding:2px 0;">{_esc(_named(payload, diet_id))}: '
                f'mean {values.get("mean", 0):.2f}, '
                f"{100 * values.get('salient_share', 0):.0f}% salient"
                f'<span style="color:{MUTED};"> · '
                f"{100 * values.get('coverage', 0):.0f}% of docs scored{thin}</span></div>"
            )
        if rows:
            blocks.append(
                f'<div style="font:600 13px -apple-system,sans-serif;color:{INK};'
                f'padding:14px 0 4px;">Liberty / oppression</div>' + "".join(rows)
                + f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};'
                f'padding-top:6px;">One model, one rubric, no validation set — the '
                f"least corroborated number here.</div>"
            )

    if not blocks:
        return ""
    return _panel("Reported separately", "".join(blocks))


# -- the whole thing --------------------------------------------------------


def _caveat_for_email(payload: dict) -> str:
    """The dashboard's caveat, with its one dashboard-specific sentence rewritten.

    ``dashboard/export._caveat`` appends "Whiskers on the radar show the
    dictionary-vs-transformer range..." when bands exist. There is no radar and
    there are no whiskers here, so pointing at them tells the reader to go look
    at something that is not in front of them. The *claim* still matters, so it
    is restated for what the email actually shows.
    """
    caveat = payload.get("caveat") or ""
    whiskers = "Whiskers on the radar show"
    if whiskers in caveat:
        caveat = caveat.split(whiskers)[0].strip() + (
            " The range printed next to each foundation is the "
            "dictionary-vs-transformer spread — wider means the two methods "
            "disagree more, so trust that foundation's number less."
        )
    return caveat


def render_html(payload: dict, own_diet: str | None = None) -> str:
    jsd = _headline(payload)
    delta, reason, since = _delta(payload.get("history") or [])
    pair = _pair(payload)

    if jsd is None:
        head = (
            f'<div style="font:15px -apple-system,sans-serif;color:{MUTED};'
            f'text-align:center;padding:8px 0;">Not enough scored documents to '
            f"compare two diets yet.</div>"
        )
    else:
        if delta is None:
            movement = (
                f'<div style="font:13px -apple-system,sans-serif;color:{MUTED};">'
                f"{_esc(reason)}</div>"
            )
        else:
            arrow = "&#9650;" if delta > 0 else ("&#9660;" if delta < 0 else "&#8212;")
            movement = (
                f'<div style="font:13px -apple-system,sans-serif;color:{MUTED};">'
                f"{arrow} {_signed(delta)} since {_esc(since) if since else 'the previous point'}"
                f"</div>"
            )
        pair_note = (
            f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};padding-top:6px;">'
            f"{_esc(_named(payload, pair[0]))} vs {_esc(_named(payload, pair[1]))}</div>"
            if pair else ""
        )
        head = (
            f'<div style="text-align:center;padding:4px 0 2px;">'
            f'<div style="font:700 46px/1.05 -apple-system,BlinkMacSystemFont,sans-serif;'
            f'color:{INK};letter-spacing:-.02em;">{jsd:.3f}</div>'
            f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};'
            f'padding:4px 0 6px;">Jensen-Shannon divergence &middot; 0 identical, '
            f"1 disjoint</div>{movement}{pair_note}</div>"
        )

    panels = "".join([
        _panel("Today", head),
        # Directly under the headline number, because it is the sentence that
        # says what the number means. It used to sit below every chart, which
        # asks the reader to interpret the evidence before being told what it
        # was evidence of.
        _executive_panel(payload),
        _composition_panel(payload),
        _log_ratio_panel(payload),
        _sparkline_panel(payload),
        _blindspot_panel(payload, own_diet),
        _summary_panel(payload),
        _extra_panel(payload),
    ])

    caveat = _caveat_for_email(payload)
    generated = (payload.get("generated_utc") or "")[:10]
    footer = (
        f'<tr><td style="padding:4px 2px 0;">'
        f'<div style="font:12px/1.5 -apple-system,sans-serif;color:{MUTED};">'
        f"{_esc(caveat)}</div>"
        f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};padding-top:8px;">'
        f"Parallax &middot; generated {_esc(generated)} &middot; both diets run through "
        f"the identical pipeline.</div></td></tr>"
    )

    preheader = _preheader(payload)
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Parallax daily brief</title></head>"
        f'<body style="margin:0;padding:0;background:{BG};">'
        f'<div style="display:none;max-height:0;overflow:hidden;opacity:0;">'
        f"{_esc(preheader)}</div>"
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:{BG};"><tr><td align="center" style="padding:20px 12px 32px;">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="max-width:{MAX_WIDTH}px;">'
        f'<tr><td style="padding:0 2px 14px;">'
        f'<div style="font:700 19px -apple-system,BlinkMacSystemFont,sans-serif;'
        f'color:{INK};letter-spacing:-.01em;">Parallax</div>'
        f'<div style="font:13px -apple-system,sans-serif;color:{MUTED};">'
        f"A moral-foundations mirror for two media diets</div></td></tr>"
        f"{panels}{footer}"
        f"</table></td></tr></table></body></html>"
    )


def _preheader(payload: dict) -> str:
    """The one line iOS shows under the subject in the inbox list."""
    jsd = _headline(payload)
    if jsd is None:
        return "Not enough scored documents to compare two diets yet."
    delta, reason, since = _delta(payload.get("history") or [])
    movement = f"{_signed(delta)} since {since}" if delta is not None else reason
    # Themes, not clusters: the inbox line should count the same things the
    # section counts, and "24 blindspots" for what reads as four subjects
    # promises a section that isn't there.
    count = len(_themes(payload))
    return f"Divergence {jsd:.3f}, {movement}. {count} blindspot theme(s)."


def _extra_text(payload: dict) -> list[str]:
    """Fairness split and liberty, for the text part.

    Kept in step with ``_extra_panel``. The liberty provenance line in
    particular has to appear in both: the text part is what a screen reader
    gets, and "least corroborated number here" is not decoration.
    """
    lines: list[str] = []
    split = payload.get("fairness_split")
    if split:
        rows = []
        for diet_id, values in (split.get("diets") or {}).items():
            if not values.get("docs_split"):
                rows.append(f"  {_named(payload, diet_id)}: not enough split-terms to partition")
                continue
            thin = " (thin coverage)" if values.get("thin") else ""
            rows.append(
                f"  {_named(payload, diet_id)}: equality {values.get('equality') or 0.0:.2f} / "
                f"proportionality {values.get('proportionality') or 0.0:.2f} · "
                f"{100 * values.get('coverage', 0):.0f}% of docs{thin}"
            )
        if rows:
            lines += ["", "Fairness: equality vs proportionality", *rows]

    liberty = payload.get("liberty")
    if liberty:
        rows = []
        for diet_id, values in (liberty.get("diets") or {}).items():
            if not values.get("docs_scored"):
                continue
            thin = " (thin coverage)" if values.get("thin") else ""
            rows.append(
                f"  {_named(payload, diet_id)}: mean {values.get('mean', 0):.2f}, "
                f"{100 * values.get('salient_share', 0):.0f}% salient · "
                f"{100 * values.get('coverage', 0):.0f}% of docs scored{thin}"
            )
        if rows:
            lines += ["", "Liberty / oppression", *rows,
                      "  One model, one rubric, no validation set — the least "
                      "corroborated number here."]
    return lines


def render_text(payload: dict, own_diet: str | None = None) -> str:
    """Plain-text alternative.

    Not decoration: some clients prefer it, VoiceOver reads it more cleanly,
    and a text part is what keeps the message from looking like bulk mail.
    """
    lines = ["PARALLAX — daily brief", ""]
    jsd = _headline(payload)
    if jsd is None:
        lines.append("Not enough scored documents to compare two diets yet.")
    else:
        lines.append(f"Jensen-Shannon divergence: {jsd:.3f}  (0 identical, 1 disjoint)")
        delta, reason, since = _delta(payload.get("history") or [])
        lines.append(f"  {_signed(delta)} since {since}" if delta is not None
                     else f"  {reason}")
        pair = _pair(payload)
        if pair:
            lines.append(f"  {_named(payload, pair[0])} vs {_named(payload, pair[1])}")

    # Directly under the number, as in the HTML. The two parts are one brief
    # and a screen reader should not meet the sections in a different order.
    executive = (payload.get("executive_summary") or "").strip()
    if executive:
        lines += ["", "THE SHORT VERSION", "", executive]

    foundations = payload.get("foundations") or []
    for diet in payload.get("diets") or []:
        lines += ["", f"{_diet_label(diet)} ({diet.get('doc_count', 0)} docs)"]
        for f in foundations:
            value, spread = _foundation(diet, f)
            band = (f"   [{spread[0]:.2f}-{spread[1]:.2f}]"
                    if spread and spread[0] is not None and spread[1] is not None else "")
            lines.append(f"  {f:<10} {value:.3f}{band}")

    ratios = (payload.get("comparison") or {}).get("log_ratios") or {}
    pair = _pair(payload)
    if ratios:
        who = (f"positive = {_named(payload, pair[0])} over-indexes, "
               f"negative = {_named(payload, pair[1])}"
               if pair else "positive = the first diet leans harder")
        lines += ["", f"Over/under-indexing (log-ratio; {who})"]
        lines += [f"  {f:<10} {_signed(ratios[f], 2)}" for f in ratios]

    history = [p for p in (payload.get("history") or []) if p.get("jsd_cumulative") is not None]
    if len(history) >= 2:
        window = payload.get("history_window_days")
        peak = max(p["jsd_cumulative"] for p in history)
        lines += ["", "Divergence over time",
                  f"  {history[0]['date']} -> {history[-1]['date']}, "
                  f"{len(history)} point(s), peak {peak:.3f}"
                  + (f", {window}-day trailing window recorded alongside" if window else "")]

    themes = _themes(payload)
    if themes:
        lines += ["", "Blindspots by theme (both directions; not a judgement about "
                      "which story mattered more)"]
        for dominant, group in _theme_groups(themes, own_diet):
            missing = _named(payload, group[0].get("other_diet") or "?")
            lines += ["", f"  {_named(payload, dominant)} covered · {missing} barely did"]
            for theme in group[:MAX_CARDS_PER_DIRECTION]:
                total = theme.get("story_count") or 0
                lines.append(
                    f"    {theme.get('title') or 'Other coverage'} — {total} "
                    f"{'story' if total == 1 else 'stories'} in "
                    f"{theme.get('cluster_count') or 0} cluster(s), "
                    f"{100 * (theme.get('one_sided') or 0):.0f}% one-sided"
                )
                stories = [s for s in (theme.get("stories") or []) if s]
                lines += [f"        · {s}" for s in stories[:STORIES_PER_CARD]]
                remainder = total - len(stories[:STORIES_PER_CARD])
                if remainder > 0:
                    lines.append(f"        +{remainder} more")
            rest = group[MAX_CARDS_PER_DIRECTION:]
            if rest:
                lines.append(f"    +{_more_themes(rest, escape=False)}")

    said = [d for d in (payload.get("diets") or []) if (d.get("summary") or "").strip()]
    if said:
        lines += ["", "WHAT EACH DIET SAID"]
        for diet in said:
            lines += ["", _diet_label(diet), "", (diet.get("summary") or "").strip()]
    method = payload.get("summary_method")
    if method and (said or executive):
        lines += ["", f"Summaries: {method}."]

    lines += _extra_text(payload)

    caveat = _caveat_for_email(payload)
    if caveat:
        lines += ["", "-- ", caveat]
    generated = (payload.get("generated_utc") or "")[:10]
    lines.append(f"Parallax · generated {generated} · both diets run through "
                 f"the identical pipeline.")
    return "\n".join(lines)


def build_digest(payload: dict, own_diet: str | None = None) -> Digest:
    """Render both parts plus the subject line.

    The subject carries the headline number and its movement, because on a
    phone that line is often the whole reading — it has to be true on its own,
    without the body to qualify it.
    """
    _check_own_diet(payload, own_diet)
    jsd = _headline(payload)
    date = (payload.get("generated_utc") or "")[:10]
    if jsd is None:
        subject = f"Parallax {date} — not enough data to compare"
    else:
        delta, _, _ = _delta(payload.get("history") or [])
        movement = f" ({_signed(delta)})" if delta is not None else ""
        subject = f"Parallax {date} — divergence {jsd:.3f}{movement}"
    return Digest(
        subject=subject,
        html=render_html(payload, own_diet),
        text=render_text(payload, own_diet),
        preheader=_preheader(payload),
    )
