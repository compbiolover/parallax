"""CLI: ``python -m cluster`` clusters stored embeddings and finds blindspots.

    python -m cluster run
    python -m cluster run --min-cluster-size 3 --dominance 0.8

Requires scikit-learn (a core dependency) and documents that
were ingested after embeddings were added (older rows have no vector).
"""

from __future__ import annotations

import argparse

from ingestion.config import load_settings
from ingestion.datastore import Datastore

from .blindspot import run_clustering


def _db_path(args: argparse.Namespace, settings: dict) -> str:
    if args.db:
        return args.db
    return (settings.get("datastore", {}) or {}).get("path", "data/parallax.sqlite")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cluster", description="Blindspot engine")
    parser.add_argument("command", choices=["run"], help="what to do")
    parser.add_argument("--db", help="SQLite path (default from settings)")
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--dominance", type=float, default=0.75,
                        help="min share of one diet for a cluster to be a blindspot")
    parser.add_argument("--min-blindspot-size", type=int, default=2)
    parser.add_argument("--no-claude-themes", action="store_true",
                        help="name blindspot themes from the built-in taxonomy "
                             "only, without an API call")
    parser.add_argument("--theme-model", help="model for theme naming")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    themes_cfg = ((settings.get("cluster", {}) or {}).get("themes", {}) or {})
    store = Datastore(_db_path(args, settings))
    try:
        if store.embedding_count() == 0:
            print("No embeddings found — run `python -m ingestion run` first.")
            return 0
        outcome = run_clustering(
            store,
            min_cluster_size=args.min_cluster_size,
            dominance=args.dominance,
            min_blindspot_size=args.min_blindspot_size,
            theme_model=args.theme_model or themes_cfg.get("model"),
            claude_themes=(not args.no_claude_themes)
                          and bool(themes_cfg.get("claude", True)),
        )
        print(
            f"Clustered {outcome.n_docs} docs -> {outcome.n_clusters} clusters "
            f"({outcome.n_noise} noise). {len(outcome.blindspots)} blindspots "
            f"in {len(outcome.themes)} themes:\n"
        )
        for t in outcome.themes:
            print(f"  [{t.dominant_diet} covers, {t.other_diet} misses] "
                  f"{t.title}  ({t.story_count} stories in {t.cluster_count} "
                  f"clusters; {t.one_sided:.0%} one-sided; named by {t.method})")
            for story in t.stories[:5]:
                print(f"      - {story}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
