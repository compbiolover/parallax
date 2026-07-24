"""Ensemble tagger with a disagreement-based confidence signal.

§5's core idea: don't force a label when the taggers disagree — flag it. This
combines several taggers (the dictionary and the transformer today; Claude
later), each exposing a per-foundation *presence probability* in [0, 1], into:

- an **ensemble score** — the weighted mean presence probability, and
- a **confidence** — how strongly the taggers agree on the resulting label.

When taggers split (one says present, another absent) the item is marked
``low_confidence``. On the gold set, low-confidence predictions are meaningfully
less accurate than high-confidence ones (see ``validation``), so this is the
signal the dashboard uses to widen a foundation's confidence band.

Taggers report a common [0, 1] scale so different score ranges (the dictionary's
per-token rate vs the transformer's softmax probability) can be combined and
disagreement measured on equal footing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .foundations import CLASSIC_FOUNDATIONS

# text -> {foundation: presence probability in [0, 1]}
ProbFn = Callable[[str], dict[str, float]]


@dataclass(frozen=True)
class Tagger:
    name: str
    prob_fn: ProbFn
    weight: float = 1.0


@dataclass(frozen=True)
class EnsembleScore:
    score: float                 # weighted mean presence probability [0, 1]
    label: int                   # 1 if score > 0.5 else 0 (matches metrics' threshold)
    confidence: float            # weighted fraction of taggers agreeing with label
    low_confidence: bool         # taggers split on presence
    votes: dict[str, int] = field(default_factory=dict)   # per-tagger 0/1


class EnsembleScorer:
    def __init__(self, taggers: list[Tagger]) -> None:
        if not taggers:
            raise ValueError("ensemble needs at least one tagger")
        self.taggers = taggers
        self._weight_total = sum(t.weight for t in taggers)

    @property
    def name(self) -> str:
        return "ensemble[" + "+".join(t.name for t in self.taggers) + "]"

    def score(self, text: str) -> dict[str, EnsembleScore]:
        per_tagger = {t.name: t.prob_fn(text) for t in self.taggers}
        out: dict[str, EnsembleScore] = {}
        for f in CLASSIC_FOUNDATIONS:
            probs = {t.name: float(per_tagger[t.name].get(f, 0.0) or 0.0) for t in self.taggers}
            votes = {name: (1 if p > 0.5 else 0) for name, p in probs.items()}
            score = sum(t.weight * probs[t.name] for t in self.taggers) / self._weight_total
            # Strict ``>`` matches validation.metrics.foundation_agreement, so the
            # calibration report's labels and format_report's F1/kappa never disagree
            # on an item that lands exactly on 0.5.
            label = 1 if score > 0.5 else 0
            agreeing = sum(t.weight for t in self.taggers if votes[t.name] == label)
            agree = agreeing / self._weight_total
            out[f] = EnsembleScore(
                score=score,
                label=label,
                confidence=agree,
                low_confidence=len(set(votes.values())) > 1,
                votes=votes,
            )
        return out

    def scores(self, text: str) -> dict[str, float]:
        """Continuous ensemble score per foundation (for AUC / ranking)."""
        return {f: es.score for f, es in self.score(text).items()}


def dictionary_prob(scorer) -> ProbFn:
    """Adapt a DictionaryScorer to a [0, 1] presence prob: present (1.0) iff any
    moral word of that foundation is matched (the dictionary's natural presence
    signal), else 0.0."""

    def prob(text: str) -> dict[str, float]:
        foundations = scorer.score(text).foundations
        return {f: (1.0 if foundations.get(f, 0.0) > 0 else 0.0) for f in CLASSIC_FOUNDATIONS}

    return prob


def build_ensemble(
    lexicon_path: str | None = None,
    model_prefix: str | None = None,
    revision: str | None = None,
    dict_weight: float = 1.0,
    transformer_weight: float = 1.0,
) -> EnsembleScorer:
    """Build the default dictionary + transformer ensemble.

    Loads the transformer lazily (heavy). Equal weights by default — the score
    leans on both, while the confidence signal weighs disagreement regardless of
    weight.
    """
    from .dictionary import DictionaryScorer
    from .lexicon import build_lexicon
    from .transformer import TransformerScorer

    lexicon, _ = build_lexicon(lexicon_path)
    dict_scorer = DictionaryScorer(lexicon)
    ts_kwargs = {}
    if model_prefix:
        ts_kwargs["model_prefix"] = model_prefix
    if revision:
        ts_kwargs["revision"] = revision
    transformer = TransformerScorer(**ts_kwargs)
    return EnsembleScorer([
        Tagger("dictionary", dictionary_prob(dict_scorer), dict_weight),
        Tagger("transformer", transformer.score, transformer_weight),
    ])
