"""Export the dashboard data payload from the datastore.

Writes a small JavaScript file (``window.PARALLAX_DATA = {...}``) rather than a
bare ``.json`` so the static page renders when opened directly from disk
(``file://``), where ``fetch`` of a sibling JSON is blocked by the browser.

The payload is aggregate-only — compositions, divergence, log-ratios, document
counts, and the generated summaries. No raw text, consistent with the §0
content-handling guardrail.

    python -m dashboard.export --db data/parallax.sqlite
    python -m dashboard.export --out dashboard/public/data/latest.js
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from compare.divergence import index_form, jensen_shannon_divergence, log_ratios
from ingestion.config import load_settings
from ingestion.datastore import Datastore
from ingestion.pipeline import diet_profiles
from scoring.foundations import CLASSIC_FOUNDATIONS

DEFAULT_OUT = Path(__file__).resolve().parent / "public" / "data" / "latest.js"

CAVEAT = (
    "Scores come from a dictionary method over a small demo lexicon and cover "
    "the five classic foundations only (no liberty). Read every number as a "
    "noisy estimate, never ground truth. See LIMITATIONS.md."
)


def build_payload(store: Datastore) -> dict:
    profiles = diet_profiles(store)
    summaries = store.all_summaries()

    diets = []
    for diet_id, profile in profiles.items():
        srow = summaries.get(diet_id)
        diets.append(
            {
                "id": diet_id,
                "label": diet_id,
                "doc_count": store.doc_count(diet_id),
                "profile": {f: profile.get(f, 0.0) for f in CLASSIC_FOUNDATIONS},
                "summary": srow["text"] if srow else "",
            }
        )

    comparison = None
    ids = sorted(profiles)
    if len(ids) >= 2:
        a, b = ids[:2]
        comparison = {
            "pair": [a, b],
            "jsd": jensen_shannon_divergence(profiles[a], profiles[b]),
            "log_ratios": log_ratios(profiles[a], profiles[b]),
            "index_form": index_form(profiles[a], profiles[b]),
        }

    exec_row = summaries.get("executive")
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "foundations": list(CLASSIC_FOUNDATIONS),
        "diets": diets,
        "comparison": comparison,
        "executive_summary": exec_row["text"] if exec_row else "",
        "summary_method": exec_row["method"] if exec_row else None,
        "caveat": CAVEAT,
    }


def write_payload(store: Datastore, out: str | Path = DEFAULT_OUT) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(store)
    body = json.dumps(payload, indent=2)
    if out.suffix == ".js":
        out.write_text(f"window.PARALLAX_DATA = {body};\n", encoding="utf-8")
    else:
        out.write_text(body + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dashboard.export", description="Export dashboard data")
    parser.add_argument("--db", help="SQLite path (default from settings)")
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output .js (or .json) path")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    db = args.db or (settings.get("datastore", {}) or {}).get("path", "data/parallax.sqlite")
    store = Datastore(db)
    try:
        out = write_payload(store, args.out)
        print(f"Wrote dashboard payload -> {out}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
