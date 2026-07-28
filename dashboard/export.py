"""Export the dashboard data payload from the datastore.

Writes a small JavaScript file (``window.PARALLAX_DATA = {...}``) rather than a
bare ``.json`` so the static page renders when opened directly from disk
(``file://``), where ``fetch`` of a sibling JSON is blocked by the browser.

The payload is aggregate-only — compositions, divergence, log-ratios, document
counts, the recorded snapshot history, and the generated summaries. No raw text,
consistent with the §0 content-handling guardrail.

Reading is all this module does. Recording a snapshot is a separate act (the
daily runner's ``snapshot`` step, or ``python -m compare.history --record``), so
exporting a payload never mutates the datastore.

    python -m dashboard.export --db data/parallax.sqlite
    python -m dashboard.export --out dashboard/public/data/latest.js
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from compare.divergence import index_form, jensen_shannon_divergence, log_ratios
from compare.history import DEFAULT_SERIES_LIMIT, load_series
from ingestion.config import load_settings
from ingestion.datastore import Datastore
from ingestion.pipeline import diet_profiles
from scoring.foundations import CLASSIC_FOUNDATIONS
from scoring.lexicon import is_demo_lexicon

DEFAULT_OUT = Path(__file__).resolve().parent / "public" / "data" / "latest.js"


def _caveat(lexicon: str | None, has_bands: bool = False) -> str:
    band_note = (
        " Whiskers on the radar show the dictionary-vs-transformer range — wider "
        "means the two methods disagree more, so trust that foundation's number less."
        if has_bands else ""
    )
    base = (
        "Scores come from a dictionary method and cover the five classic "
        "foundations only (no liberty). Read every number as a noisy estimate, "
        "never ground truth. See LIMITATIONS.md."
    )
    if is_demo_lexicon(lexicon):
        return (
            "Scores come from a dictionary method over a small DEMO lexicon "
            "(illustrative only) and cover the five classic foundations only "
            "(no liberty). Read every number as a noisy estimate, never ground "
            "truth. See LIMITATIONS.md." + band_note
        )
    return f"{base} Lexicon: {lexicon}.{band_note}"


def _band_payload(store: Datastore) -> tuple[dict, str | None]:
    """Per-diet confidence bands keyed by diet id, plus the transformer scorer
    name that produced them (``None`` if the transformer never ran)."""
    transformer_scorer = store.get_meta("transformer_scorer")
    if not transformer_scorer:
        return {}, None
    from compare.confidence import all_diet_bands

    raw = all_diet_bands(store, transformer_scorer)
    payload = {
        diet_id: {
            f: {
                "point": b.point, "low": b.low, "high": b.high,
                "dictionary": b.dictionary, "transformer": b.transformer,
                "disagreement": b.disagreement,
            }
            for f, b in bands.items()
        }
        for diet_id, bands in raw.items()
    }
    return payload, (transformer_scorer if payload else None)


def build_payload(store: Datastore, history_limit: int | None = DEFAULT_SERIES_LIMIT) -> dict:
    profiles = diet_profiles(store)
    summaries = store.all_summaries()
    bands, transformer_scorer = _band_payload(store)

    diets = []
    for diet_id, profile in profiles.items():
        srow = summaries.get(diet_id)
        diets.append(
            {
                "id": diet_id,
                "label": diet_id,
                "doc_count": store.doc_count(diet_id),
                "profile": {f: profile.get(f, 0.0) for f in CLASSIC_FOUNDATIONS},
                "band": bands.get(diet_id),  # None unless the transformer ran
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
    lexicon = store.get_meta("lexicon")
    has_bands = transformer_scorer is not None
    history = load_series(store, history_limit)
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "foundations": list(CLASSIC_FOUNDATIONS),
        "diets": diets,
        "comparison": comparison,
        # Dated history. ``history_window_days`` is the trailing window behind the
        # windowed series; it comes off the newest row, so a settings change is
        # described accurately for the points it actually applies to going forward.
        "history": history,
        "history_window_days": history[-1]["window_days"] if history else None,
        "fairness_split": _fairness_payload(store),
        "liberty": _liberty_payload(store),
        "executive_summary": exec_row["text"] if exec_row else "",
        "summary_method": exec_row["method"] if exec_row else None,
        "lexicon": lexicon,
        "has_confidence_bands": has_bands,
        "band_scorers": (
            {"dictionary": lexicon, "transformer": transformer_scorer} if has_bands else None
        ),
        "blindspots": _blindspots(store),
        "caveat": _caveat(lexicon, has_bands),
    }


def _fairness_payload(store: Datastore) -> dict | None:
    """Equality-vs-proportionality shares per diet, or ``None`` if nothing split.

    Coverage travels with the shares so the dashboard can show how much of each
    diet the partition actually rests on, rather than presenting a ratio from a
    handful of documents as if it described the whole diet.
    """
    from compare.fairness import all_diet_fairness, gap

    profiles = all_diet_fairness(store)
    split = {d: p for d, p in profiles.items() if p.docs_split}
    if not split:
        return None
    return {
        "diets": {
            diet_id: {
                "equality": p.equality,
                "proportionality": p.proportionality,
                "docs_split": p.docs_split,
                "docs_total": p.docs_total,
                "coverage": p.coverage,
                "thin": p.thin,
                "leans": p.leans,
            }
            for diet_id, p in profiles.items()
        },
        "gap": gap(profiles),
    }


def _liberty_payload(store: Datastore) -> dict | None:
    """Per-diet liberty engagement, or ``None`` if the tagger never ran.

    Reported separately from the radar composition on purpose — see
    ``compare/liberty.py`` for why partial coverage can't be folded into a
    composition without moving the headline number.
    """
    scorer = store.get_meta("liberty_scorer")
    if not scorer:
        return None
    from compare.liberty import all_diet_liberty, gap

    profiles = all_diet_liberty(store, scorer)
    if not any(p.docs_scored for p in profiles.values()):
        return None
    return {
        "scorer": scorer,
        "diets": {
            diet_id: {
                "mean": p.mean,
                "salient_share": p.salient_share,
                "docs_scored": p.docs_scored,
                "docs_total": p.docs_total,
                "coverage": p.coverage,
                "thin": p.thin,
            }
            for diet_id, p in profiles.items()
        },
        "gap": gap(profiles),
    }


def _blindspots(store: Datastore) -> list[dict]:
    """Serialize persisted blindspots (empty until `python -m cluster run`)."""
    from cluster.blindspot import blindspots_from_store

    return [
        {
            "label": b.label,
            "dominant_diet": b.dominant_diet,
            "other_diet": b.other_diet,
            "counts": b.counts,
            "size": b.size,
            "dominant_share": b.dominant_share,
            "representative_titles": b.representative_titles,
        }
        for b in blindspots_from_store(store)
    ]


def write_payload_dict(payload: dict, out: str | Path = DEFAULT_OUT) -> Path:
    """Serialize an already-built payload.

    Split out from ``write_payload`` so a caller that needs the payload for
    something else as well — the daily run renders it into an email too — can
    build it once and be sure both surfaces describe the same dict rather than
    two separately computed ones.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2)
    if out.suffix == ".js":
        out.write_text(f"window.PARALLAX_DATA = {body};\n", encoding="utf-8")
    else:
        out.write_text(body + "\n", encoding="utf-8")
    return out


def write_payload(
    store: Datastore,
    out: str | Path = DEFAULT_OUT,
    history_limit: int | None = DEFAULT_SERIES_LIMIT,
) -> Path:
    return write_payload_dict(build_payload(store, history_limit), out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dashboard.export", description="Export dashboard data")
    parser.add_argument("--db", help="SQLite path (default from settings)")
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output .js (or .json) path")
    parser.add_argument("--history-limit", type=int, default=DEFAULT_SERIES_LIMIT,
                        help="most recent N snapshots to serialize (0 = all)")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    db = args.db or (settings.get("datastore", {}) or {}).get("path", "data/parallax.sqlite")
    store = Datastore(db)
    try:
        out = write_payload(store, args.out, args.history_limit or None)
        print(f"Wrote dashboard payload -> {out}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
