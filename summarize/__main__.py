"""CLI: ``python -m summarize`` generates and stores daily diet summaries.

    python -m summarize                 # summarize + persist to the datastore
    python -m summarize --db data/parallax.sqlite --model claude-opus-5

Uses Claude when ANTHROPIC_API_KEY is set, otherwise a deterministic fallback.
"""

from __future__ import annotations

import argparse

from compare.reference import resolve
from ingestion.config import load_registry, load_settings
from ingestion.datastore import Datastore

from .summarizer import DEFAULT_EFFORT, DEFAULT_MODEL, Summarizer


def _db_path(args: argparse.Namespace, settings: dict) -> str:
    if args.db:
        return args.db
    return (settings.get("datastore", {}) or {}).get("path", "data/parallax.sqlite")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="summarize", description="Daily diet summaries")
    parser.add_argument("--db", help="SQLite path (default from settings)")
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--model", help=f"Claude model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                        help=f"thinking depth (default: {DEFAULT_EFFORT})")
    parser.add_argument("--mine", help="persona id for your side of the reference pair")
    parser.add_argument("--theirs", help="persona id for the other side")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    registry = load_registry(settings=settings)
    pair = resolve(
        settings, args.mine, args.theirs,
        available=registry.persona_ids(), families=registry.families(),
    )
    store = Datastore(_db_path(args, settings))
    try:
        # CLI flag, then the `summarize:` block in settings, then the defaults.
        cfg = settings.get("summarize", {}) or {}
        model = args.model or cfg.get("model")
        effort = args.effort or cfg.get("effort")
        summarizer = Summarizer(model=model or DEFAULT_MODEL,
                                effort=effort or DEFAULT_EFFORT)
        result = summarizer.summarize(store, registry, pair)
        if not result.per_diet and not result.executive:
            print("No scored documents yet — run `python -m ingestion run` first.")
            return 0
        summarizer.persist(store, result)
        print(f"Summaries generated via '{result.method}' (model={result.model}).")
        for diet, text in result.per_diet.items():
            print(f"\n## {diet}\n{text}")
        print(f"\n## Executive\n{result.executive}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
