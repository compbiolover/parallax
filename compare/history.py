"""Snapshot history: the foundation profiles and divergence, dated.

Every ``make daily`` run used to overwrite the last one — the dashboard showed
today and remembered nothing. This module keeps one aggregate row per UTC date
so the headline numbers become a series, which is what ``CLAUDE.md`` §2 asks the
dashboard to draw (a JSD time series) and what §4 Phase 5 needs before it can
ask whether the gap widens around elections.

Only derived metrics are retained — compositions, divergence, log-ratios, and
document counts. No text, consistent with the §0 content-handling guardrail.

**Two bases, because they answer different questions.**

*Cumulative* profiles every document in the corpus dated on or before the
snapshot date. It is the number the dashboard's radar and big JSD figure report,
so the series ends exactly where the headline sits. Being an average over an
ever-growing corpus, it is heavily damped: one week of unusual coverage barely
moves it, and it moves less the longer the project runs.

*Windowed* profiles only the trailing ``window_days`` (7 by default). It is
noisier, and on thin days it is noise — but it is the basis that can actually
respond to an event, so it is the one to read for "did the gap widen this week?"

Both are recorded for every date. The dashboard plots both and says which is
which; neither is presented as the truer number.

**Live versus reconstructed.** ``record_snapshot`` writes a *live* row: the
corpus as it stood when the pipeline ran. ``backfill_series`` reconstructs rows
for past dates from the publication dates already in the store, so the chart is
useful on day one instead of after a month of daily runs. The arithmetic is
identical, but the inputs are not equivalent: a reconstructed row for last
Tuesday includes articles that were published then and fetched since (GDELT
backfill pulls weeks of history), which a live run that day could not have seen.
Reconstructed rows therefore read as "what the corpus now says about that date",
not "what the dashboard would have shown". Rows carry a ``source`` marker so the
distinction survives into the payload rather than living only in this docstring.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta

from scoring.foundations import CLASSIC_FOUNDATIONS

logger = logging.getLogger(__name__)

# A week: long enough to survive a quiet weekend, short enough to still move.
DEFAULT_WINDOW_DAYS = 7

# How much history the dashboard payload carries by default. Rows are tiny, but
# the payload is a single file the browser parses on every load.
DEFAULT_SERIES_LIMIT = 365

LIVE = "live"
RECONSTRUCTED = "reconstructed"


# Snapshots are stored and shipped at six decimals. Nothing downstream can use
# more: §5 is explicit that these are noisy estimates, not ground truth, and full
# float repr would roughly double a year-long payload for digits that are noise.
PRECISION = 6


def _round(value):
    """Round a float, or every float in a mapping. Passes ``None`` through."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: round(float(v), PRECISION) for k, v in value.items()}
    return round(float(value), PRECISION)


@dataclass
class Basis:
    """One diet-comparison computed over a single set of documents.

    The index form (100 x P/Q) is deliberately absent: it is a pure restatement
    of ``log_ratios``, so storing it per day would inflate every payload with a
    number the dashboard can derive.
    """

    jsd: float | None  # None: fewer than two scored diets
    pair: list[str] | None
    diets: dict[str, dict] = field(default_factory=dict)  # id -> composition + doc_count
    log_ratios: dict[str, float] | None = None


@dataclass
class Snapshot:
    snapshot_date: str  # 'YYYY-MM-DD' (UTC)
    generated_utc: str
    window_days: int
    source: str  # LIVE | RECONSTRUCTED
    cumulative: Basis
    window: Basis

    def to_payload(self) -> dict:
        """The JSON blob persisted in ``snapshots.payload``."""
        return {
            "source": self.source,
            "cumulative": asdict(self.cumulative),
            "window": asdict(self.window),
        }


def _day_after(day: str) -> str:
    """Exclusive upper bound for a date, so ``[since, until)`` covers ``day``."""
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


def _window_start(day: str, window_days: int) -> str:
    """Inclusive lower bound of the trailing window ending on (and including) ``day``."""
    return (date.fromisoformat(day) - timedelta(days=max(1, window_days) - 1)).isoformat()


def today_utc() -> str:
    return datetime.now(UTC).date().isoformat()


def _basis(store, registry, pair, scorer: str, since: str | None, until: str) -> Basis:
    """Profiles, counts, and divergence over one date window.

    Every persona is reported, including those with no documents in the window —
    at count zero and with no composition. A persona going quiet is a fact about
    the week, and dropping it would silently reshape the series. That is also why
    the persona list comes from the registry rather than from the corpus.

    The divergence is computed on the named reference pair. It used to be the
    first two ids in sorted order, which made the recorded series depend on how
    the personas happened to be spelled — and would have changed the moment a
    persona was added.
    """
    # Imported here, not at module scope: ``divergence`` pulls in scipy (~0.3s),
    # and the daily runner imports this module just to read two default constants.
    from compare.divergence import jensen_shannon_divergence, log_ratios
    from ingestion.pipeline import persona_profiles

    profiles = persona_profiles(store, registry, scorer, since=since, until=until)
    diets = {
        persona_id: {
            "composition": (
                _round({f: profiles[persona_id].get(f, 0.0) for f in CLASSIC_FOUNDATIONS})
                if persona_id in profiles
                else None
            ),
            "doc_count": store.doc_count_for_sources(
                registry.weights_for(persona_id), since=since, until=until
            ),
        }
        for persona_id in registry.persona_ids()
    }

    if pair.mine not in profiles or pair.theirs not in profiles:
        return Basis(jsd=None, pair=None, diets=diets)
    a, b = pair.mine, pair.theirs
    return Basis(
        jsd=_round(jensen_shannon_divergence(profiles[a], profiles[b])),
        pair=[a, b],
        diets=diets,
        # Oriented mine-first, so positive means "my diet over-indexes" as
        # CLAUDE.md §3(5) specifies. Rows recorded before the pair was named hold
        # the opposite sign; nothing charts historical log-ratios, and the pair
        # travels with every row so the two are distinguishable.
        log_ratios=_round(log_ratios(profiles[a], profiles[b])),
    )


def build_snapshot(
    store,
    registry,
    pair,
    snapshot_date: str | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    scorer: str = "dictionary",
    source: str = LIVE,
) -> Snapshot:
    """Compute both bases for one date without persisting anything."""
    day = snapshot_date or today_utc()
    until = _day_after(day)
    return Snapshot(
        snapshot_date=day,
        generated_utc=datetime.now(UTC).isoformat(),
        window_days=window_days,
        source=source,
        cumulative=_basis(store, registry, pair, scorer, since=None, until=until),
        window=_basis(
            store,
            registry,
            pair,
            scorer,
            since=_window_start(day, window_days),
            until=until,
        ),
    )


def record_snapshot(
    store,
    registry,
    pair,
    snapshot_date: str | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    scorer: str = "dictionary",
    source: str = LIVE,
) -> Snapshot:
    """Compute and persist today's (or ``snapshot_date``'s) snapshot.

    Idempotent per date: running the pipeline three times in a morning leaves
    one row, holding the last run's numbers.

    Warns when the reference pair differs from the newest recorded row's. The
    series keeps its column names across such a change but stops meaning the same
    thing, and a divergence chart that silently splices two different comparisons
    is worse than one with a visible gap.
    """
    _warn_on_pair_change(store, pair)
    snap = build_snapshot(store, registry, pair, snapshot_date, window_days, scorer, source)
    store.upsert_snapshot(
        snapshot_date=snap.snapshot_date,
        generated_utc=snap.generated_utc,
        window_days=snap.window_days,
        jsd_cumulative=snap.cumulative.jsd,
        jsd_window=snap.window.jsd,
        payload=snap.to_payload(),
    )
    return snap


def _warn_on_pair_change(store, pair) -> None:
    rows = store.snapshot_rows()
    if not rows:
        return
    try:
        payload = json.loads(rows[-1]["payload"])
    except (ValueError, KeyError, TypeError):
        return
    recorded = (payload.get("cumulative") or {}).get("pair")
    if recorded and list(recorded) != pair.as_list():
        logger.warning(
            "reference pair changed from %s vs %s to %s vs %s — earlier points in "
            "the divergence series describe the previous comparison and are not "
            "continuous with the ones from here on",
            recorded[0],
            recorded[1],
            pair.mine,
            pair.theirs,
        )


def backfill_series(
    store,
    registry,
    pair,
    days: int = 30,
    window_days: int = DEFAULT_WINDOW_DAYS,
    scorer: str = "dictionary",
    overwrite: bool = False,
    end_date: str | None = None,
) -> list[str]:
    """Reconstruct up to ``days`` of past snapshots from stored publication dates.

    Existing rows are left alone unless ``overwrite`` — a reconstruction must not
    quietly replace a live row that recorded the corpus as it actually stood.
    Walks back no further than the oldest document, and returns the dates written.
    """
    end = end_date or today_utc()
    earliest, _ = store.document_date_range()
    if earliest is None:
        return []

    existing = {r["snapshot_date"] for r in store.snapshot_rows()} if not overwrite else set()
    written: list[str] = []
    for offset in range(days):
        day = (date.fromisoformat(end) - timedelta(days=offset)).isoformat()
        if day < earliest:
            break
        if day in existing:
            continue
        record_snapshot(store, registry, pair, day, window_days, scorer, source=RECONSTRUCTED)
        written.append(day)
    return sorted(written)


def load_series(store, limit: int | None = DEFAULT_SERIES_LIMIT) -> list[dict]:
    """The recorded history, oldest first, as plain dicts for the dashboard payload."""
    series = []
    for row in store.snapshot_rows(limit):
        payload = json.loads(row["payload"])
        series.append(
            {
                "date": row["snapshot_date"],
                "generated_utc": row["generated_utc"],
                "window_days": row["window_days"],
                "source": payload.get("source", LIVE),
                "jsd_cumulative": row["jsd_cumulative"],
                "jsd_window": row["jsd_window"],
                "cumulative": payload.get("cumulative"),
                "window": payload.get("window"),
            }
        )
    return series


# -- CLI --------------------------------------------------------------------


def _format_series(series: list[dict]) -> str:
    if not series:
        return "No snapshots recorded yet. Run `python -m daily` (or --backfill here)."
    lines = [f"{'date':<12} {'window':>8} {'all-time':>9}  docs  source", "-" * 52]
    for s in series:
        counts = (s.get("window") or {}).get("diets", {})
        docs = sum(int(d.get("doc_count") or 0) for d in counts.values())
        w = f"{s['jsd_window']:.3f}" if s["jsd_window"] is not None else "  —  "
        c = f"{s['jsd_cumulative']:.3f}" if s["jsd_cumulative"] is not None else "  —  "
        lines.append(f"{s['date']:<12} {w:>8} {c:>9} {docs:>5}  {s['source']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from ingestion.config import datastore_path, load_registry, load_settings
    from ingestion.datastore import Datastore

    from .reference import resolve

    p = argparse.ArgumentParser(
        prog="compare.history", description="Inspect or reconstruct the snapshot history"
    )
    p.add_argument("--db", help="SQLite path (default from settings)")
    p.add_argument("--settings", help="path to settings.yaml")
    p.add_argument("--record", action="store_true", help="record today's snapshot now")
    p.add_argument(
        "--backfill",
        type=int,
        metavar="DAYS",
        help="reconstruct DAYS of past snapshots from stored publication dates",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="let --backfill replace existing rows (including live ones)",
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"trailing window for the windowed basis (default {DEFAULT_WINDOW_DAYS})",
    )
    p.add_argument(
        "--limit", type=int, default=DEFAULT_SERIES_LIMIT, help="how many recent snapshots to list"
    )
    args = p.parse_args(argv)

    settings = load_settings(args.settings)
    db = datastore_path(settings, args.db)
    store = Datastore(db)
    try:
        if args.record or args.backfill:
            # Resolved only for the writers. A snapshot is a statement about two
            # named personas, so they genuinely need it — but the listing path
            # reads nothing except the SQLite file, and making it depend on a
            # loadable registry would take away the one command that still works
            # when the configuration is the thing that broke.
            registry = load_registry(settings=settings)
            pair = resolve(settings, available=registry.persona_ids(), families=registry.families())
        if args.backfill:
            written = backfill_series(
                store,
                registry,
                pair,
                days=args.backfill,
                window_days=args.window_days,
                overwrite=args.overwrite,
            )
            print(f"Reconstructed {len(written)} snapshot(s).")
        if args.record:
            snap = record_snapshot(store, registry, pair, window_days=args.window_days)
            print(f"Recorded live snapshot for {snap.snapshot_date}.")
        print(_format_series(load_series(store, args.limit)))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
