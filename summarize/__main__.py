"""CLI: ``python -m summarize`` generates and stores daily diet summaries.

    python -m summarize                 # summarize + persist to the datastore
    python -m summarize --db data/parallax.sqlite --model claude-opus-4-8

Uses Claude when ANTHROPIC_API_KEY is set, otherwise a deterministic fallback.
"""

from __future__ import annotations

import argparse

from ingestion.config import load_settings
from ingestion.datastore import Datastore

from .summarizer import DEFAULT_MODEL, Summarizer


def _db_path(args: argparse.Namespace, settings: dict) -> str:
    if args.db:
        return args.db
    return (settings.get("datastore", {}) or {}).get("path", "data/parallax.sqlite")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="summarize", description="Daily diet summaries")
    parser.add_argument("--db", help="SQLite path (default from settings)")
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id")
    args = parser.parse_args(argv)

    store = Datastore(_db_path(args, load_settings(args.settings)))
    try:
        summarizer = Summarizer(model=args.model)
        result = summarizer.summarize(store)
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
