"""CLI: ``python -m daily`` runs the whole snapshot in one command.

    python -m daily                      # full: ingest -> backfill -> cluster -> summarize
                                         #       -> snapshot -> export
    python -m daily --skip backfill      # fast path: today's feeds only
    python -m daily --only cluster export
    python -m daily --only snapshot export   # re-record today's point, rebuild the payload
    python -m daily --backfill-days 3 --max-per-source 100

Exits non-zero if any step failed, so cron/systemd notices — the remaining steps
still run, so the dashboard is refreshed from whatever data did land.
"""

from __future__ import annotations

import argparse

from .runner import STEPS, DailyConfig, format_report, run_daily


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="daily", description="Parallax daily snapshot")
    p.add_argument("--db", help="SQLite path (default from settings)")
    p.add_argument("--settings", help="path to settings.yaml")
    p.add_argument("--out", help="dashboard payload path (default dashboard/public/data/latest.js)")

    sel = p.add_mutually_exclusive_group()
    sel.add_argument(
        "--only",
        nargs="+",
        choices=STEPS,
        metavar="STEP",
        help=f"run only these steps ({', '.join(STEPS)})",
    )
    sel.add_argument(
        "--skip", nargs="+", choices=STEPS, metavar="STEP", help="run everything except these steps"
    )

    p.add_argument("--max-items", type=int, help="max items per feed (ingest)")
    p.add_argument("--lexicon", help="eMFD-format CSV (overrides settings)")
    p.add_argument(
        "--transformer",
        dest="transformer",
        action="store_true",
        default=None,
        help="force the transformer tagger on (confidence bands)",
    )
    p.add_argument(
        "--no-transformer",
        dest="transformer",
        action="store_false",
        help="skip the transformer tagger (faster, no bands)",
    )

    # Backfill flags default to None so settings.yaml's `daily.backfill` block is
    # only overridden when a flag is actually given.
    p.add_argument("--backfill-days", type=int, help="GDELT window in days")
    p.add_argument("--max-per-source", type=int, help="max GDELT articles per outlet (<=250)")
    p.add_argument(
        "--backfill-extract",
        action="store_true",
        default=None,
        help="fetch bodies for backfilled articles too (much slower)",
    )
    p.add_argument(
        "--backfill-transformer",
        action="store_true",
        default=None,
        help="also transformer-score backfilled (title-only) articles",
    )

    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--dominance", type=float, default=0.75)
    p.add_argument("--min-blindspot-size", type=int, default=2)
    p.add_argument("--model", help="Claude model id for summaries")
    p.add_argument(
        "--window-days",
        type=int,
        help="trailing window for the snapshot's windowed basis (default 7)",
    )
    p.add_argument(
        "--history-limit", type=int, help="most recent N snapshots to serialize into the payload"
    )
    p.add_argument("--quiet", action="store_true", help="only print the final report")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log why individual fetches failed, not just the count",
    )
    return p.parse_args(argv)


def _steps_from(args: argparse.Namespace, base: tuple[str, ...]) -> tuple[str, ...]:
    """--only/--skip are explicit and win outright; otherwise keep whatever the
    settings block enabled (e.g. `daily.backfill.enabled: false`)."""
    if args.only:
        return tuple(s for s in STEPS if s in set(args.only))
    if args.skip:
        return tuple(s for s in base if s not in set(args.skip))
    return base


def build_config(args: argparse.Namespace, settings: dict) -> DailyConfig:
    """Settings-derived defaults, with any explicitly-given flag layered on top."""
    cfg = DailyConfig.from_settings(settings)
    cfg.db = args.db
    cfg.settings_path = args.settings
    cfg.out = args.out
    cfg.steps = _steps_from(args, cfg.steps)
    cfg.max_items_per_feed = args.max_items
    cfg.lexicon_path = args.lexicon
    cfg.transformer = args.transformer
    cfg.min_cluster_size = args.min_cluster_size
    cfg.dominance = args.dominance
    cfg.min_blindspot_size = args.min_blindspot_size
    cfg.model = args.model
    if args.backfill_days is not None:
        cfg.backfill_days = args.backfill_days
    if args.max_per_source is not None:
        cfg.backfill_max_per_source = args.max_per_source
    if args.backfill_extract is not None:
        cfg.backfill_extract_bodies = args.backfill_extract
    if args.backfill_transformer is not None:
        cfg.backfill_transformer = args.backfill_transformer
    if args.window_days is not None:
        cfg.window_days = args.window_days
    if args.history_limit is not None:
        cfg.history_limit = args.history_limit
    return cfg


def main(argv: list[str] | None = None) -> int:
    import logging

    from ingestion.__main__ import build_reporter
    from ingestion.config import load_settings

    args = _parse(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = build_config(args, load_settings(args.settings))
    progress = None if args.quiet else build_reporter()
    if progress is not None:
        print(f"Running steps: {', '.join(cfg.steps)}")
    report = run_daily(cfg, progress=progress)
    print()
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
