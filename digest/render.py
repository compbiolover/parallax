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
BG = "#f7f8fa"

MAX_WIDTH = 600
SPARK_DAYS = 21           # ~3 weeks reads clearly at phone width

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
    label = (diet.get("label") or "").strip()
    return label if label and label != diet.get("id") else _label(diet["id"])


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
        f'<span style="color:{colours[a]};">&#9632;</span> right = {_esc(_label(a))} '
        f'over-indexes &nbsp; <span style="color:{colours[b]};">&#9632;</span> left = '
        f"{_esc(_label(b))} does. Natural log of the ratio of shares; 0 is parity.</div>"
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


def _blindspot_panel(payload: dict, own: str | None) -> str:
    """Both directions, identical markup, the author's own first."""
    spots = payload.get("blindspots") or []
    if not spots:
        return ""
    colours = _colours(payload)
    items = []
    for spot in _own_first(spots, own):
        dominant = _label(spot.get("dominant_diet", "?"))
        missing = _label(spot.get("other_diet", "?"))
        # Coloured by who covered it, so the colour means the same thing here
        # as in every other panel. "Whose blindspot" is carried by the ordering
        # and by the sentence underneath, not by hue.
        colour = colours.get(spot.get("dominant_diet"), DIET_A)
        titles = spot.get("representative_titles") or []
        title_html = "".join(
            f'<div style="font:13px -apple-system,sans-serif;color:{INK};'
            f'padding:2px 0 2px 10px;">· {_esc(t)}</div>' for t in titles[:3]
        )
        items.append(
            f'<div style="padding:10px 0;border-top:1px solid {LINE};">'
            f'<div style="font:600 13px -apple-system,sans-serif;color:{colour};">'
            f"{_esc(spot.get('label', 'cluster'))}</div>"
            f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};padding:2px 0 4px;">'
            f"{_esc(dominant)} covered this; {_esc(missing)} barely did "
            f"({spot.get('size', 0)} stories, "
            f"{100 * spot.get('dominant_share', 0):.0f}% one-sided)</div>"
            f"{title_html}</div>"
        )
    note = (
        f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};padding-top:10px;">'
        f"Both directions are listed. Coverage asymmetry is not a judgement about "
        f"which story mattered more.</div>"
    )
    return _panel("Blindspots", "".join(items) + note)


def _summary_panel(payload: dict) -> str:
    parts = []
    executive = (payload.get("executive_summary") or "").strip()
    if executive:
        parts.append(
            f'<div style="font:14px/1.55 -apple-system,sans-serif;color:{INK};'
            f'white-space:pre-wrap;">{_esc(executive)}</div>'
        )
    colours = _colours(payload)
    for diet in payload.get("diets") or []:
        text = (diet.get("summary") or "").strip()
        if not text:
            continue
        colour = colours[diet["id"]]
        parts.append(
            f'<div style="padding-top:14px;"><div style="font:600 13px -apple-system,'
            f'sans-serif;color:{colour};padding-bottom:4px;">{_esc(_diet_label(diet))}</div>'
            f'<div style="font:14px/1.55 -apple-system,sans-serif;color:{INK};'
            f'white-space:pre-wrap;">{_esc(text)}</div></div>'
        )
    if not parts:
        return ""
    method = payload.get("summary_method")
    if method:
        parts.append(
            f'<div style="font:12px -apple-system,sans-serif;color:{MUTED};'
            f'padding-top:12px;">Summaries: {_esc(method)}.</div>'
        )
    return _panel("What each diet said", "".join(parts))


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
                    f'padding:2px 0;">{_esc(_label(diet_id))}: not enough split-terms '
                    f"to partition</div>"
                )
                continue
            equality = values.get("equality") or 0.0
            thin = " (thin coverage)" if values.get("thin") else ""
            rows.append(
                f'<div style="font:13px -apple-system,sans-serif;color:{INK};'
                f'padding:2px 0;">{_esc(_label(diet_id))}: equality {equality:.2f} / '
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
                f'padding:2px 0;">{_esc(_label(diet_id))}: mean {values.get("mean", 0):.2f}, '
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
            f"{_esc(_label(pair[0]))} vs {_esc(_label(pair[1]))}</div>" if pair else ""
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
    spots = len(payload.get("blindspots") or [])
    return f"Divergence {jsd:.3f}, {movement}. {spots} blindspot(s)."


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
                rows.append(f"  {_label(diet_id)}: not enough split-terms to partition")
                continue
            thin = " (thin coverage)" if values.get("thin") else ""
            rows.append(
                f"  {_label(diet_id)}: equality {values.get('equality') or 0.0:.2f} / "
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
                f"  {_label(diet_id)}: mean {values.get('mean', 0):.2f}, "
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
            lines.append(f"  {_label(pair[0])} vs {_label(pair[1])}")

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
        who = (f"positive = {_label(pair[0])} over-indexes, negative = {_label(pair[1])}"
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

    spots = _own_first(payload.get("blindspots") or [], own_diet)
    if spots:
        lines += ["", "Blindspots (both directions; not a judgement about which "
                      "story mattered more)"]
        for spot in spots:
            lines.append(
                f"  {spot.get('label', 'cluster')} — "
                f"{_label(spot.get('dominant_diet', '?'))} covered it, "
                f"{_label(spot.get('other_diet', '?'))} barely did "
                f"({spot.get('size', 0)} stories, "
                f"{100 * spot.get('dominant_share', 0):.0f}% one-sided)"
            )
            lines += [f"      · {t}" for t in (spot.get("representative_titles") or [])[:3]]

    executive = (payload.get("executive_summary") or "").strip()
    if executive:
        lines += ["", "Executive summary", executive]
    for diet in payload.get("diets") or []:
        text = (diet.get("summary") or "").strip()
        if text:
            lines += ["", _diet_label(diet), text]
    method = payload.get("summary_method")
    if method:
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
