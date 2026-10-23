"""This week against the weeks before it, and the stories both sides covered.

The all-time profile moves less every day the corpus grows, so a brief built on
it says the same thing every morning. This module answers the questions a
weekly brief leads with instead:

* **What moved.** Each foundation's log-ratio (mine over theirs) for the trailing
  7 days, against the same number for each of the previous ``weeks`` weeks. A
  foundation *moved* when this week sits outside that range and is at least
  ``Z_MOVED`` standard deviations from the prior mean. Both, because eight
  points are few: the range alone would flag about one foundation in five on
  chance, and the z-score alone can fire inside a narrow but noisy range.
* **One story, two framings.** For clusters both diets covered this week, each
  side's foundation mix *within that story*, ranked by how far apart they are.
  Holding the event fixed is what makes a framing difference legible
  (Mokhberian et al. 2020 measure moral framing the same way).
* **Where a gap comes from.** A diet's composition is a weighted sum over its
  documents divided by its total moral signal, so a foundation's share splits
  exactly across stories. Each story's part of the gap is its share for mine
  minus its share for theirs; the parts sum to the gap.

Weekly windows end on ``end`` (today, UTC) and run back 7 days, the same window
the daily clustering uses, so the stories here are the clustering's stories.
Only derived numbers and headlines leave this module, per the §0 content rule.
"""

from __future__ import annotations

import statistics
from datetime import UTC, date, datetime, timedelta

from cluster.titles import clean_title
from compare.divergence import jensen_shannon_divergence, log_ratios
from compare.reference import ReferencePair
from scoring.aggregate import to_composition
from scoring.foundations import CLASSIC_FOUNDATIONS

WEEKS = 8
# Prior weeks needed before "outside the usual range" means anything.
MIN_BASELINE_WEEKS = 4
# A week with fewer documents than this on either side is noise, not a point.
MIN_WEEK_DOCS = 20
Z_MOVED = 2.0
# Documents a side needs inside one story before its mix there is a framing.
MIN_STORY_DOCS = 3
MAX_FRAMINGS = 3
MAX_DRIVERS = 3


def _weights(registry, pair: ReferencePair) -> dict[str, dict[str, float]]:
    return {p: registry.weights_for(p) for p in pair}


def _mix(rows, weights: dict[str, float]) -> tuple[dict[str, float], float, int]:
    """(composition, total weighted moral signal, document count) for one side."""
    totals = dict.fromkeys(CLASSIC_FOUNDATIONS, 0.0)
    n = 0
    for row in rows:
        w = weights.get(row["source_id"], 0.0)
        if w <= 0:
            continue
        n += 1
        for f in CLASSIC_FOUNDATIONS:
            totals[f] += w * max(0.0, row[f] or 0.0)
    return to_composition(totals), sum(totals.values()), n


def _week_bounds(end: date, back: int) -> tuple[str, str]:
    """``[since, until)`` for the 7-day window ending ``back`` weeks before ``end``."""
    last = end - timedelta(days=7 * back)
    return (last - timedelta(days=6)).isoformat(), (last + timedelta(days=1)).isoformat()


def weekly_movement(
    store,
    registry,
    pair: ReferencePair,
    end: date | None = None,
    weeks: int = WEEKS,
    scorer: str = "dictionary",
) -> dict | None:
    """This week's comparison against its own recent history.

    ``None`` when this week is too thin on either side to compare at all.
    """
    end = end or datetime.now(UTC).date()
    oldest, _ = _week_bounds(end, weeks)
    _, until = _week_bounds(end, 0)
    rows = store.clustered_scores(scorer, since=oldest, until=until)
    weights = _weights(registry, pair)

    points = []  # newest first: (week_end, log_ratios, jsd, counts)
    for back in range(weeks + 1):
        lo, hi = _week_bounds(end, back)
        week = [r for r in rows if lo <= r["dated"] < hi]
        mixes = {p: _mix(week, weights[p]) for p in pair}
        counts = {p: mixes[p][2] for p in pair}
        if min(counts.values()) < MIN_WEEK_DOCS:
            if back == 0:
                return None
            continue
        mine, theirs = mixes[pair.mine][0], mixes[pair.theirs][0]
        points.append(
            (
                (end - timedelta(days=7 * back)).isoformat(),
                log_ratios(mine, theirs),
                jensen_shannon_divergence(mine, theirs),
                counts,
            )
        )

    week_end, now, jsd_now, counts = points[0]
    prior = points[1:]
    has_baseline = len(prior) >= MIN_BASELINE_WEEKS
    foundations = []
    for f in CLASSIC_FOUNDATIONS:
        entry = {"foundation": f, "now": now[f]}
        if has_baseline:
            entry.update(_against(now[f], [p[1][f] for p in prior]))
        foundations.append(entry)
    jsd = {"now": jsd_now, "series": [{"week_end": p[0], "jsd": p[2]} for p in reversed(points)]}
    if has_baseline:
        jsd.update(_against(jsd_now, [p[2] for p in prior]))

    week_rows = [r for r in rows if r["dated"] >= _week_bounds(end, 0)[0]]
    return {
        "week_start": _week_bounds(end, 0)[0],
        "week_end": week_end,
        "pair": list(pair.ids),
        "orientation": "mine_first",
        "counts": counts,
        "baseline_weeks": len(prior),
        "baseline_needed": MIN_BASELINE_WEEKS,
        "foundations": foundations,
        "jsd": jsd,
        "framings": story_framings(week_rows, pair, weights),
        "drivers": {
            e["foundation"]: gap_drivers(week_rows, pair, weights, e["foundation"])
            for e in foundations
            if e.get("moved")
        },
    }


def _against(value: float, history: list[float]) -> dict:
    """Where ``value`` sits against prior weekly values."""
    low, high = min(history), max(history)
    mean = statistics.fmean(history)
    sd = statistics.pstdev(history)
    outside = value < low or value > high
    # A perfectly flat history has no spread to measure against: z is None and
    # leaving the range alone counts as a move, rather than reading as z = 0.
    z = (value - mean) / sd if sd > 0 else None
    return {
        "low": low,
        "high": high,
        "median": statistics.median(history),
        "z": z,
        "moved": outside and (z is None or abs(z) >= Z_MOVED),
    }


def _by_story(rows) -> dict[int, list]:
    stories: dict[int, list] = {}
    for r in rows:
        if r["cluster_id"] is not None and r["cluster_id"] != -1:
            stories.setdefault(r["cluster_id"], []).append(r)
    return stories


def _headline(rows) -> dict | None:
    for r in rows:  # newest first
        title = clean_title(r["title"])
        if title:
            return {"title": title, "url": r["url"], "date": (r["dated"] or "")[:10]}
    return None


def story_framings(rows, pair: ReferencePair, weights: dict[str, dict[str, float]]) -> list[dict]:
    """Stories both sides covered this week, most differently framed first."""
    out = []
    for cid, members in _by_story(rows).items():
        sides = {}
        for p in pair:
            side_rows = [r for r in members if weights[p].get(r["source_id"], 0.0) > 0]
            mix, signal, n = _mix(side_rows, weights[p])
            if n < MIN_STORY_DOCS or signal <= 0:
                break
            top = sorted(mix.items(), key=lambda kv: -kv[1])[:3]
            sides[p] = {
                "docs": n,
                "mix": mix,
                "top": [{"foundation": f, "share": s} for f, s in top],
                "headline": _headline(side_rows),
            }
        else:
            headline = _headline(members)
            out.append(
                {
                    "cluster_id": cid,
                    "title": headline["title"] if headline else "",
                    "jsd": jensen_shannon_divergence(
                        sides[pair.mine]["mix"], sides[pair.theirs]["mix"]
                    ),
                    "sides": sides,
                }
            )
    out.sort(key=lambda s: -s["jsd"])
    return out[:MAX_FRAMINGS]


def gap_drivers(
    rows, pair: ReferencePair, weights: dict[str, dict[str, float]], foundation: str
) -> list[dict]:
    """The stories that make up this week's gap on one foundation.

    Each story's part is its share of mine's foundation mix minus its share of
    theirs. Parts over every story plus the unclustered remainder sum to the
    gap, so the shares returned are exact, not attributions by heuristic.
    """
    totals = {p: _mix(rows, weights[p])[1] for p in pair}
    if not all(totals.values()):
        return []

    def part(members) -> float:
        a = sum(
            weights[pair.mine].get(r["source_id"], 0.0) * (r[foundation] or 0.0) for r in members
        )
        b = sum(
            weights[pair.theirs].get(r["source_id"], 0.0) * (r[foundation] or 0.0) for r in members
        )
        return a / totals[pair.mine] - b / totals[pair.theirs]

    gap = part(rows)
    if abs(gap) < 1e-12:
        return []
    parts = []
    for cid, members in _by_story(rows).items():
        share = part(members) / gap
        if share > 0:  # only stories pushing the gap the way it points
            headline = _headline(members)
            parts.append(
                {"cluster_id": cid, "title": headline["title"] if headline else "", "share": share}
            )
    parts.sort(key=lambda p: -p["share"])
    top = parts[:MAX_DRIVERS]
    # Signed, and always present, so the parts still sum to the gap: when other
    # stories pull the other way the listed ones exceed 100% and this goes negative.
    rest = 1 - sum(p["share"] for p in top)
    top.append({"cluster_id": None, "title": "Everything else", "share": rest})
    return top
