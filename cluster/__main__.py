"""CLI: ``python -m cluster`` clusters stored embeddings and finds blindspots.

    python -m cluster run
    python -m cluster run --min-cluster-size 3 --dominance 0.8

Requires scikit-learn (a core dependency) and documents that
were ingested after embeddings were added (older rows have no vector).
"""

from __future__ import annotations

import argparse

from compare.reference import resolve
from ingestion.config import datastore_path, load_registry, load_settings
from ingestion.datastore import Datastore

from .blindspot import run_clustering


def _configured_effort(themes_cfg: dict):
    """The configured theme effort, keeping "absent" and "null" apart.

    ``effort: ~`` is a deliberate instruction — send no effort and let the model
    decide — and reading it with ``.get`` turns it back into the default.
    """
    from cluster.themes import UNSET

    # `.get` *with a default* keeps a stored None; it is `.get("effort")` and
    # `or` that collapse it, which is how this broke in the first place.
    return themes_cfg.get("effort", UNSET)


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
    parser.add_argument("--theme-effort",
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="thinking depth for theme naming (default: low)")
    parser.add_argument("--mine", help="persona id for your side of the reference pair")
    parser.add_argument("--theirs", help="persona id for the other side")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    themes_cfg = ((settings.get("cluster", {}) or {}).get("themes", {}) or {})
    registry = load_registry(settings=settings)
    pair = resolve(
        settings, args.mine, args.theirs,
        available=registry.persona_ids(), families=registry.families(),
    )
    store = Datastore(datastore_path(settings, args.db))
    try:
        if store.embedding_count() == 0:
            print("No embeddings found — run `python -m ingestion run` first.")
            return 0
        outcome = run_clustering(
            store,
            {p: set(registry.weights_for(p)) for p in pair},
            min_cluster_size=args.min_cluster_size,
            dominance=args.dominance,
            min_blindspot_size=args.min_blindspot_size,
            theme_model=args.theme_model or themes_cfg.get("model"),
            theme_effort=args.theme_effort or _configured_effort(themes_cfg),
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
