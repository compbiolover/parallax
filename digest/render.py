"""Payload -> email. Same content as the dashboard, no JavaScript.

Reads exactly the payload ``dashboard/export.build_payload`` produces, so the
email and the page can never drift apart in what they claim. Every section
degrades to nothing when its data is absent — an unscored foundation, a
missing summary and a day with no clusters all render as silence rather than
as a zero, which is the same rule the rest of the codebase follows for
"unscored is not the same as none".
"""

from __future__ import annotations

import html
from dataclasses import dataclass

# Matches the dashboard's light palette so the two readings look like one tool.
SELF = "#2f6fb0"
OTHER = "#c46a2e"
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
    """``modeled_ce`` -> ``Modeled CE``. The payload carries no display names.

    Short tokens stay upper-cased because the diet ids in ``sources.yaml`` use
    initialisms, and plain title-casing renders them as words ("Modeled Ce"),
    which reads like a typo in the one place the reader is deciding whether to
    trust the numbers.
    """
    words = diet_id.replace("_", " ").split()
    return " ".join(w.upper() if len(w) <= 2 else w.capitalize() for w in words)


def _signed(value: float, places: int = 3) -> str:
    return f"{value:+.{places}f}"


def _delta(history: list[dict]) -> tuple[float | None, str]:
    """Change in windowed divergence since the previous recorded point.

    Returns ``(None, reason)`` when there is nothing to compare against. A
    first-ever snapshot has no delta, and printing ``+0.000`` there would read
    as "nothing moved" rather than "nothing to move from".
    """
    usable = [p for p in history if p.get("jsd_window") is not None]
    if not usable:
        return None, "no divergence recorded yet"
    if len(usable) < 2:
        return None, "first recorded point — no previous day to compare"
    return usable[-1]["jsd_window"] - usable[-2]["jsd_window"], ""


def _headline(payload: dict) -> float | None:
    comparison = payload.get("comparison") or {}
    return comparison.get("jsd")


def _pair(payload: dict) -> tuple[str, str] | None:
    comparison = payload.get("comparison") or {}
    pair = comparison.get("pair")
    return (pair[0], pair[1]) if pair and len(pair) == 2 else None


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


def _bar_row(label: str, value: float, share: float, colour: str) -> str:
    """One horizontal bar. ``share`` is 0..1 of the full width."""
    pct = max(0.0, min(1.0, share)) * 100
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
        f'padding:3px 0 3px 8px;text-align:right;white-space:nowrap;">{value:.3f}</td>'
        f"</tr>"
    )


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
    blocks = []
    for i, diet in enumerate(diets):
        colour = SELF if i == 0 else OTHER
        profile = diet.get("profile") or {}
        peak = max([profile.get(f, 0.0) for f in foundations] or [0.0]) or 1.0
        rows = "".join(
            _bar_row(f, profile.get(f, 0.0), profile.get(f, 0.0) / peak, colour)
            for f in foundations
        )
        blocks.append(
            f'<div style="font:600 13px -apple-system,sans-serif;color:{colour};'
            f'padding:{"0" if i == 0 else "14px"} 0 6px;">{_esc(_label(diet["id"]))}'
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

    label_w, left_w, right_w, value_w = RATIO_COLS
    rows = []
    for foundation, value in ratios.items():
        pct = min(1.0, abs(value) / widest) * 100
        colour = SELF if value > 0 else OTHER
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

    a, b = pair
    note = (
        f'<div style="font:12px/1.5 -apple-system,sans-serif;color:{MUTED};padding-top:10px;">'
        f'<span style="color:{SELF};">&#9632;</span> right = {_esc(_label(a))} '
        f'over-indexes &nbsp; <span style="color:{OTHER};">&#9632;</span> left = '
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
            f'<div style="background:{SELF};height:{height}px;border-radius:2px;'
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
    items = []
    for spot in _own_first(spots, own):
        dominant = _label(spot.get("dominant_diet", "?"))
        missing = _label(spot.get("other_diet", "?"))
        yours = spot.get("other_diet") == own
        colour = OTHER if yours else SELF
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
    for i, diet in enumerate(payload.get("diets") or []):
        text = (diet.get("summary") or "").strip()
        if not text:
            continue
        colour = SELF if i == 0 else OTHER
        parts.append(
            f'<div style="padding-top:14px;"><div style="font:600 13px -apple-system,'
            f'sans-serif;color:{colour};padding-bottom:4px;">{_esc(_label(diet["id"]))}</div>'
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
            equality = values.get("equality")
            if equality is None:
                rows.append(
                    f'<div style="font:13px -apple-system,sans-serif;color:{MUTED};'
                    f'padding:2px 0;">{_esc(_label(diet_id))}: not enough split-terms '
                    f"to partition</div>"
                )
                continue
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


def render_html(payload: dict, own_diet: str | None = None) -> str:
    jsd = _headline(payload)
    delta, reason = _delta(payload.get("history") or [])
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
                f"{arrow} {_signed(delta)} since the previous day</div>"
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

    caveat = payload.get("caveat") or ""
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
    delta, reason = _delta(payload.get("history") or [])
    movement = f"{_signed(delta)} since yesterday" if delta is not None else reason
    spots = len(payload.get("blindspots") or [])
    return f"Divergence {jsd:.3f}, {movement}. {spots} blindspot(s)."


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
        delta, reason = _delta(payload.get("history") or [])
        lines.append(f"  {_signed(delta)} since the previous day" if delta is not None
                     else f"  {reason}")
        pair = _pair(payload)
        if pair:
            lines.append(f"  {_label(pair[0])} vs {_label(pair[1])}")

    for diet in payload.get("diets") or []:
        profile = diet.get("profile") or {}
        lines += ["", f"{_label(diet['id'])} ({diet.get('doc_count', 0)} docs)"]
        lines += [f"  {f:<10} {profile.get(f, 0.0):.3f}"
                  for f in payload.get("foundations") or []]

    ratios = (payload.get("comparison") or {}).get("log_ratios") or {}
    if ratios:
        lines += ["", "Over/under-indexing (log-ratio, positive = first diet leans harder)"]
        lines += [f"  {f:<10} {_signed(v, 2)}" for f, v in ratios.items()]

    spots = _own_first(payload.get("blindspots") or [], own_diet)
    if spots:
        lines += ["", "Blindspots (both directions)"]
        for spot in spots:
            lines.append(
                f"  {spot.get('label', 'cluster')} — "
                f"{_label(spot.get('dominant_diet', '?'))} covered it, "
                f"{_label(spot.get('other_diet', '?'))} barely did "
                f"({spot.get('size', 0)} stories)"
            )
            lines += [f"      · {t}" for t in (spot.get("representative_titles") or [])[:3]]

    executive = (payload.get("executive_summary") or "").strip()
    if executive:
        lines += ["", "Executive summary", executive]
    for diet in payload.get("diets") or []:
        text = (diet.get("summary") or "").strip()
        if text:
            lines += ["", _label(diet["id"]), text]

    caveat = payload.get("caveat")
    if caveat:
        lines += ["", "-- ", caveat]
    return "\n".join(lines)


def build_digest(payload: dict, own_diet: str | None = None) -> Digest:
    """Render both parts plus the subject line.

    The subject carries the headline number and its movement, because on a
    phone that line is often the whole reading — it has to be true on its own,
    without the body to qualify it.
    """
    jsd = _headline(payload)
    date = (payload.get("generated_utc") or "")[:10]
    if jsd is None:
        subject = f"Parallax {date} — not enough data to compare"
    else:
        delta, _ = _delta(payload.get("history") or [])
        movement = f" ({_signed(delta)})" if delta is not None else ""
        subject = f"Parallax {date} — divergence {jsd:.3f}{movement}"
    return Digest(
        subject=subject,
        html=render_html(payload, own_diet),
        text=render_text(payload, own_diet),
        preheader=_preheader(payload),
    )
