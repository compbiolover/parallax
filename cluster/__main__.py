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
    args = parser.parse_args(argv)

    store = Datastore(_db_path(args, load_settings(args.settings)))
    try:
        if store.embedding_count() == 0:
            print("No embeddings found — run `python -m ingestion run` first.")
            return 0
        outcome = run_clustering(
            store,
            min_cluster_size=args.min_cluster_size,
            dominance=args.dominance,
            min_blindspot_size=args.min_blindspot_size,
        )
        print(
            f"Clustered {outcome.n_docs} docs -> {outcome.n_clusters} clusters "
            f"({outcome.n_noise} noise). {len(outcome.blindspots)} blindspots:\n"
        )
        for b in outcome.blindspots:
            counts = ", ".join(f"{d}={n}" for d, n in sorted(b.counts.items()))
            print(f"  [{b.dominant_diet} covers, {b.other_diet} misses] "
                  f"{b.label}  ({counts}; {b.dominant_share:.0%})")
            for t in b.representative_titles:
                print(f"      - {t}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
