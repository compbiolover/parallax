"""CLI: ``python -m validation`` scores the gold set and reports agreement.

    python -m validation                          # dictionary (built-in seed lexicon)
    python -m validation --lexicon data/emfd_scoring.csv
    python -m validation --scorer transformer     # needs pip install -e ".[scoring]"
    python -m validation --scorer liberty         # Claude; needs ANTHROPIC_API_KEY
    python -m validation --gold validation/gold/seed.json

The ``liberty`` scorer is the one that costs money per run. ``--model`` picks the
Claude model and ``--limit`` caps how many gold items are scored, so comparing
tiers (Haiku vs Sonnet vs Opus) on a subset is cheap:

    python -m validation --scorer liberty --model claude-haiku-4-5 --limit 40
    python -m validation --scorer liberty --model claude-sonnet-5  --limit 40
"""

from __future__ import annotations

import argparse
from dataclasses import replace

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


def _build_liberty_score_fn(args: argparse.Namespace):
    """Score function for the Claude liberty tagger, plus a place to stash the
    per-item rationale and quote for eyeballing."""
    from scoring.liberty import DEFAULT_MODEL, build_tagger

    tagger = build_tagger(model=args.model or DEFAULT_MODEL, use_batch=False)
    if tagger is None:
        raise SystemExit("liberty scoring needs ANTHROPIC_API_KEY set and `anthropic` installed.")
    evidence: list[tuple[str, object]] = []

    def score_fn(text: str) -> dict[str, float]:
        result = tagger.score(text)
        evidence.append((text, result))
        # A failed or refused call is recorded as 0.0 so the run completes, which
        # understates the scorer rather than crashing the report. The evidence
        # list below shows how many came back empty.
        return {"liberty": result.presence if result is not None else 0.0}

    return score_fn, f"claude-liberty [{tagger.model}]", 0.5, evidence


def _print_liberty_evidence(evidence: list, limit: int = 5) -> None:
    """Show a few judgments with their grounding quote.

    Validation is the one place these are visible: production scores keep only
    the number, because a persisted quote is persisted article text (§0)."""
    scored = [(t, r) for t, r in evidence if r is not None]
    print(
        f"\nLiberty judgments: {len(scored)} of {len(evidence)} returned a score; "
        f"{sum(1 for _t, r in scored if r.grounded)} carried a supporting quote."
    )
    for text, result in scored[:limit]:
        print(f"\n  [{result.presence:.2f} {result.pole}/{result.register}] {text[:70].strip()}...")
        print(f"    quote: {result.quote[:100] or '(none)'}")
        print(f"    why:   {result.rationale[:140]}")


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

        ens = build_ensemble(
            lexicon_path=args.lexicon, model_prefix=args.model, revision=args.revision
        )
        return ens.scores, ens.name, 0.5, ens
    lexicon, name = build_lexicon(args.lexicon)
    scorer = DictionaryScorer(lexicon)
    return (lambda text: scorer.score(text).foundations), f"dictionary [{name}]", 0.0, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validation", description="Validate scorers vs the gold set"
    )
    parser.add_argument("--gold", default=str(GOLD_DIR / "seed.json"), help="gold set JSON")
    parser.add_argument(
        "--scorer",
        default="dictionary",
        choices=["dictionary", "transformer", "ensemble", "liberty"],
    )
    parser.add_argument("--lexicon", help="eMFD-format CSV for the dictionary scorer")
    parser.add_argument(
        "--model", help="transformer model prefix, or the Claude model id for --scorer liberty"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="score only the first N gold items (bounds cost on --scorer liberty)",
    )
    parser.add_argument("--revision", help="pin the transformer model HF revision (commit/tag)")
    parser.add_argument(
        "--threshold", type=float, help="presence threshold (default: scorer-specific)"
    )
    args = parser.parse_args(argv)

    goldset = load_gold(args.gold)
    if args.limit:
        goldset = replace(goldset, items=goldset.items[: args.limit])

    if args.scorer == "liberty":
        return _run_liberty(args, goldset)

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


def _run_liberty(args: argparse.Namespace, goldset) -> int:
    """Evaluate the Claude liberty tagger against hand-coded liberty labels."""
    if "liberty" not in goldset.labelled:
        print(
            f"This gold set ({len(goldset.items)} items) has no liberty labels, so there "
            "is nothing to score against.\n\n"
            "Liberty was added after the first gold sets were coded, and no public corpus "
            "labels it — the eMFD covers five foundations and MFRC does not label liberty "
            'either. Add a "liberty": 0/1 key to each item\'s labels in '
            f"{args.gold}, following the same MFRC-style convention as the other five "
            "(does this text invoke the foundation, virtue or vice), then re-run.\n\n"
            "Scoring it now would report an AUC against all-zero labels, which would look "
            "like a result and mean nothing."
        )
        return 1

    score_fn, label, default_threshold, evidence = _build_liberty_score_fn(args)
    threshold = args.threshold if args.threshold is not None else default_threshold
    results = evaluate(goldset, score_fn, foundations=["liberty"], threshold=threshold)
    print(format_report(results, label))
    _print_liberty_evidence(evidence)
    print(f"\nGold set: {len(goldset.items)} items, coders={goldset.coders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
