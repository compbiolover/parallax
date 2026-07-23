"""CLI: ``python -m ingestion`` runs the Phase 1 pipeline.

    python -m ingestion run                # fetch, dedup, score, store
    python -m ingestion compare            # print diet profiles + JSD
    python -m ingestion run --db data/parallax.sqlite --max-items 10
"""

from __future__ import annotations

import argparse
from itertools import combinations

from compare.divergence import jensen_shannon_divergence, log_ratios

from .config import load_registry, load_settings
from .datastore import Datastore
from .pipeline import PipelineConfig, RunStats, diet_profiles, run


def _db_path(args: argparse.Namespace, settings: dict) -> str:
    if args.db:
        return args.db
    return (settings.get("datastore", {}) or {}).get("path", "data/parallax.sqlite")


def _print_stats(stats: RunStats) -> None:
    print("\nIngestion complete:")
    print(f"  fetched items      : {stats.fetched}")
    print(f"  stored (unique)    : {stats.stored}")
    print(f"  exact duplicates   : {stats.exact_duplicates}")
    print(f"  near duplicates    : {stats.near_duplicates}")
    print(f"  skipped (too short): {stats.skipped_short}")
    print(f"  errors             : {stats.errors}")
    if stats.per_diet:
        print("  per diet           :")
        for diet, n in sorted(stats.per_diet.items()):
            print(f"    {diet}: {n}")


def _print_compare(store: Datastore) -> None:
    profiles = diet_profiles(store)
    if not profiles:
        print("No scored documents yet — run `python -m ingestion run` first.")
        return
    print("\nDiet foundation profiles (composition, sums to 1):")
    for diet, prof in profiles.items():
        pretty = ", ".join(f"{k}={v:.3f}" for k, v in prof.items())
        print(f"  {diet}: {pretty}")

    for a, b in combinations(sorted(profiles), 2):
        jsd = jensen_shannon_divergence(profiles[a], profiles[b])
        print(f"\nJensen-Shannon divergence  {a} vs {b}: {jsd:.4f}  (0=identical, 1=disjoint)")
        print("  per-foundation log-ratio (positive = first diet over-indexes):")
        for f, lr in log_ratios(profiles[a], profiles[b]).items():
            print(f"    {f:9}: {lr:+.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingestion", description="Parallax Phase 1 pipeline")
    parser.add_argument("command", choices=["run", "compare"], help="what to do")
    parser.add_argument("--db", help="SQLite path (default from settings)")
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--max-items", type=int, help="max items per feed")
    parser.add_argument("--min-words", type=int, help="minimum document word count")
    parser.add_argument("--lexicon", help="path to an eMFD-format CSV (overrides settings)")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    store = Datastore(_db_path(args, settings))
    try:
        if args.command == "run":
            cfg = PipelineConfig.from_settings(settings)
            if args.max_items is not None:
                cfg.max_items_per_feed = args.max_items
            if args.min_words is not None:
                cfg.min_words = args.min_words
            if args.lexicon is not None:
                cfg.lexicon_path = args.lexicon
            stats = run(store, load_registry(), cfg)
            _print_stats(stats)
            print(f"\nDatastore: {store.counts()}")
        elif args.command == "compare":
            _print_compare(store)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
