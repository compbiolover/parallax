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

from .evaluate import evaluate, format_report
from .gold import GOLD_DIR, load_gold


def _build_score_fn(args: argparse.Namespace):
    if args.scorer == "transformer":
        from scoring.transformer import TransformerScorer  # lazy: heavy deps

        ts = TransformerScorer(model_prefix=args.model) if args.model else TransformerScorer()
        return ts.score, ts.name, 0.5
    lexicon, name = build_lexicon(args.lexicon)
    scorer = DictionaryScorer(lexicon)
    return (lambda text: scorer.score(text).foundations), f"dictionary [{name}]", 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validation", description="Validate scorers vs the gold set")
    parser.add_argument("--gold", default=str(GOLD_DIR / "seed.json"), help="gold set JSON")
    parser.add_argument("--scorer", choices=["dictionary", "transformer"], default="dictionary")
    parser.add_argument("--lexicon", help="eMFD-format CSV for the dictionary scorer")
    parser.add_argument("--model", help="transformer model id / path (Mformer by default)")
    parser.add_argument("--threshold", type=float, help="presence threshold (default: scorer-specific)")
    args = parser.parse_args(argv)

    goldset = load_gold(args.gold)
    score_fn, label, default_threshold = _build_score_fn(args)
    threshold = args.threshold if args.threshold is not None else default_threshold
    results = evaluate(goldset, score_fn, threshold=threshold)
    print(format_report(results, label))
    print(f"\nGold set: {len(goldset.items)} items, coders={goldset.coders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
