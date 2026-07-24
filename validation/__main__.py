"""CLI: ``python -m validation`` scores the gold set and reports agreement.

    python -m validation                          # dictionary (built-in seed lexicon)
    python -m validation --lexicon data/emfd_scoring.csv
    python -m validation --scorer transformer     # needs parallax[scoring]
    python -m validation --gold validation/gold/seed.json
"""

from __future__ import annotations

import argparse

from scoring.dictionary import DictionaryScorer
from scoring.lexicon import build_lexicon

from .evaluate import (
    confidence_calibration_scored,
    evaluate,
    evaluate_scored,
    format_calibration,
    format_report,
)
from .gold import GOLD_DIR, load_gold


def _build_score_fn(args: argparse.Namespace):
    """Returns (score_fn, label, threshold, ensemble_or_None)."""
    if args.scorer == "transformer":
        from scoring.transformer import TransformerScorer  # lazy: heavy deps

        kwargs = {"revision": args.revision} if args.revision else {}
        if args.model:
            kwargs["model_prefix"] = args.model
        ts = TransformerScorer(**kwargs)
        return ts.score, ts.name, 0.5, None
    if args.scorer == "ensemble":
        from scoring.ensemble import build_ensemble  # lazy: heavy deps

        ens = build_ensemble(lexicon_path=args.lexicon, model_prefix=args.model, revision=args.revision)
        return ens.scores, ens.name, 0.5, ens
    lexicon, name = build_lexicon(args.lexicon)
    scorer = DictionaryScorer(lexicon)
    return (lambda text: scorer.score(text).foundations), f"dictionary [{name}]", 0.0, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validation", description="Validate scorers vs the gold set")
    parser.add_argument("--gold", default=str(GOLD_DIR / "seed.json"), help="gold set JSON")
    parser.add_argument("--scorer", choices=["dictionary", "transformer", "ensemble"], default="dictionary")
    parser.add_argument("--lexicon", help="eMFD-format CSV for the dictionary scorer")
    parser.add_argument("--model", help="transformer model prefix (Mformer by default)")
    parser.add_argument("--revision", help="pin the transformer model HF revision (commit/tag)")
    parser.add_argument("--threshold", type=float, help="presence threshold (default: scorer-specific)")
    args = parser.parse_args(argv)

    goldset = load_gold(args.gold)
    score_fn, label, default_threshold, ensemble = _build_score_fn(args)
    threshold = args.threshold if args.threshold is not None else default_threshold

    if ensemble is not None:
        # Score each item once, then derive both the agreement report and the
        # confidence calibration from it — the transformer is the expensive part.
        scored = [ensemble.score(item.text) for item in goldset.items]
        continuous = [{f: es.score for f, es in es_map.items()} for es_map in scored]
        results = evaluate_scored(goldset, continuous, threshold=threshold)
        print(format_report(results, label))
        print()
        print(format_calibration(confidence_calibration_scored(goldset, scored)))
    else:
        results = evaluate(goldset, score_fn, threshold=threshold)
        print(format_report(results, label))

    print(f"\nGold set: {len(goldset.items)} items, coders={goldset.coders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
