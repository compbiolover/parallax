"""CLI: ``python -m ingestion`` runs the Phase 1 pipeline.

    python -m ingestion run                # fetch, dedup, score, store
    python -m ingestion podcasts           # transcribe new episodes (slow, hours)
    python -m ingestion compare            # print diet profiles + JSD
    python -m ingestion labels             # record diet labels from the registry
    python -m ingestion run --db data/parallax.sqlite --max-items 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations

from cluster.embed import build_embedder
from compare.divergence import jensen_shannon_divergence, log_ratios

from .config import load_registry, load_settings
from .datastore import Datastore
from .pipeline import PipelineConfig, RunStats, SourceProgress, backfill, diet_profiles, run


def _db_path(args: argparse.Namespace, settings: dict) -> str:
    if args.db:
        return args.db
    return (settings.get("datastore", {}) or {}).get("path", "data/parallax.sqlite")


def format_progress(event: SourceProgress) -> str:
    """One line per source, written as it finishes.

    Carries the elapsed seconds because that is what distinguishes a slow
    network from a stuck one: sources take a couple of seconds each, and a
    double-digit number means something hit the 30s fetch timeout.
    """
    src = event.source
    head = f"  [{event.index:>2}/{event.total}] {src.id:<28} {src.diet_id:<12}"
    if event.failed:
        return f"{head} feed unreachable ({event.seconds:.1f}s)"
    detail = f"{event.stored} stored / {event.fetched} fetched"
    if event.errors:
        detail += f", {event.errors} unreadable"
    return f"{head} {detail:<28} {event.seconds:>5.1f}s"


def build_reporter(stream=None):
    """Print progress as it happens, unbuffered.

    Line-buffered output is not enough here: the gaps between lines are the
    whole signal, and a buffered pipe would deliver them in one burst at the
    end, which is the silence this exists to remove.
    """
    out = stream if stream is not None else sys.stdout

    def report(event: SourceProgress | str) -> None:
        print(event if isinstance(event, str) else format_progress(event),
              file=out, flush=True)

    return report


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


def _podcasts(store: Datastore, settings: dict, args, progress) -> None:
    """The `podcasts` command: the ledger, or a transcription run.

    Kept out of `main` because it is the one command whose *cost* needs stating
    before it starts. Everything else here is minutes; this is hours, and the
    person running it should see the budget it will honour rather than discover
    it from a log the next morning.
    """
    from .podcast import PodcastConfig
    from .podcast import run as run_podcasts

    if args.episode_status:
        counts = store.episode_counts()
        if not counts:
            print("No episodes recorded yet.")
            return
        print("Episode ledger:")
        for status, n in sorted(counts.items()):
            print(f"  {status:12} {n}")
        return

    pcfg = PodcastConfig.from_settings(settings)
    if args.max_episodes is not None:
        pcfg.max_episodes_per_source = args.max_episodes
    if args.since_days is not None:
        pcfg.since_days = args.since_days
    if args.time_budget_minutes is not None:
        pcfg.time_budget_seconds = args.time_budget_minutes * 60
    if args.whisper_model:
        pcfg.whisper_model = args.whisper_model

    cfg = PipelineConfig.from_settings(settings)
    if args.lexicon is not None:
        cfg.lexicon_path = args.lexicon
    if args.transformer is not None:
        cfg.transformer_enabled = args.transformer

    registry = load_registry()
    sources = [s for s in registry.all_sources()
               if s.ingest_type == "podcast_rss" and s.url]
    if progress is not None:
        print(f"Transcribing up to {pcfg.max_episodes_per_source} episode(s) each from "
              f"{len(sources)} podcast source(s), published within {pcfg.since_days}d, "
              f"on faster-whisper {pcfg.whisper_model} ({pcfg.compute_type}).\n"
              f"Budget: {pcfg.time_budget_seconds // 60} min — an hour of audio takes "
              f"roughly that long on CPU, so expect this to use it.\n", flush=True)

    embedder, _ = build_embedder(settings)
    stats = run_podcasts(store, registry, cfg, pcfg, embedder=embedder,
                         progress=progress)
    print(f"\n{stats.line()}")
    for source_id, n in sorted(stats.per_source.items()):
        print(f"  {source_id:32} {n}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingestion", description="Parallax Phase 1 pipeline")
    parser.add_argument("command",
                        choices=["run", "backfill", "podcasts", "compare", "labels"],
                        help="what to do")
    parser.add_argument("--db", help="SQLite path (default from settings)")
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--max-items", type=int, help="max items per feed (run)")
    parser.add_argument("--min-words", type=int, help="minimum document word count")
    parser.add_argument("--lexicon", help="path to an eMFD-format CSV (overrides settings)")
    parser.add_argument("--days", type=int, default=14, help="backfill window in days")
    parser.add_argument("--max-per-source", type=int, default=250,
                        help="backfill: max GDELT articles per outlet (<=250)")
    parser.add_argument("--extract", action="store_true",
                        help="backfill: fetch article bodies for full scoring (slow)")
    parser.add_argument("--transformer", dest="transformer", action="store_true", default=None,
                        help="run: also transformer-score every article (confidence bands)")
    parser.add_argument("--no-transformer", dest="transformer", action="store_false",
                        help="run: skip the transformer tagger (dictionary-only, no bands)")
    parser.add_argument("--max-episodes", type=int,
                        help="podcasts: episodes per source per run")
    parser.add_argument("--since-days", type=int,
                        help="podcasts: only episodes published this recently")
    parser.add_argument("--time-budget-minutes", type=int,
                        help="podcasts: stop starting new episodes after this long")
    parser.add_argument("--whisper-model",
                        help="podcasts: faster-whisper size, e.g. medium, large-v3")
    parser.add_argument("--episode-status", action="store_true",
                        help="podcasts: print the ledger and transcribe nothing")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="log why individual fetches failed, not just the count")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress the per-source progress lines")
    args = parser.parse_args(argv)

    # Progress is on by default. The run walks every source sequentially behind
    # a 30-second timeout each, so a silent terminal for several minutes is the
    # normal case, and it looks exactly like a hang.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    progress = None if args.quiet else build_reporter()

    settings = load_settings(args.settings)
    store = Datastore(_db_path(args, settings))
    try:
        if args.command in ("run", "backfill"):
            cfg = PipelineConfig.from_settings(settings)
            if args.max_items is not None:
                cfg.max_items_per_feed = args.max_items
            if args.min_words is not None:
                cfg.min_words = args.min_words
            if args.lexicon is not None:
                cfg.lexicon_path = args.lexicon
            if args.transformer is not None:
                cfg.transformer_enabled = args.transformer
            embedder, _ = build_embedder(settings)
            if args.command == "run":
                registry = load_registry()
                if progress is not None:
                    n = len(list(registry.ingestable(("rss",))))
                    print(f"Ingesting {n} RSS source(s), up to "
                          f"{cfg.max_items_per_feed} item(s) each. Sequential, with a "
                          f"{cfg.timeout}s fetch timeout and {cfg.per_host_rpm} "
                          f"requests/min per host — expect a few minutes.\n", flush=True)
                stats = run(store, registry, cfg, embedder=embedder, progress=progress)
            else:
                print(f"Backfilling {args.days}d of history from GDELT "
                      f"(≤{args.max_per_source}/source, "
                      f"{'bodies' if args.extract else 'titles only'})…")
                stats = backfill(store, load_registry(), cfg, embedder=embedder,
                                 days=args.days, max_per_source=args.max_per_source,
                                 extract_bodies=args.extract)
            _print_stats(stats)
            print(f"\nDatastore: {store.counts()}")
        elif args.command == "podcasts":
            _podcasts(store, settings, args, progress)
        elif args.command == "compare":
            _print_compare(store)
        elif args.command == "labels":
            # Ingestion records these, so a store that has not been ingested
            # since the labels were added still names its diets by machine id.
            # This writes them without re-fetching anything.
            from .pipeline import _store_diet_labels

            _store_diet_labels(store, load_registry())
            recorded = store.diet_labels()
            short = store.diet_short_labels()
            if not recorded:
                print("No diet labels in the registry to record.")
            for diet_id, label in sorted(recorded.items()):
                print(f"  {diet_id}: {label}"
                      + (f"  (short: {short[diet_id]})" if diet_id in short else ""))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
