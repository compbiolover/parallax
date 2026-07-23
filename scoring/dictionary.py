"""Dictionary-baseline moral-foundations scorer.

Tokenizes a document, matches tokens against a :class:`~scoring.lexicon.Lexicon`,
and returns a length-normalized score per classic foundation plus a sentiment
signal and a moral-word ratio.

**Length normalization is not optional.** Raw dictionary counts correlate with
document length (r up to ~0.98 for the eMFD); every foundation value returned
here is a per-token rate (sum of matched weights / total tokens), never a raw
count. This is the single most important guard against garbage aggregates
(see ``CLAUDE.md`` §3 and ``LIMITATIONS.md``).

Coverage: the dictionary baseline scores the five CLASSIC foundations only — it
has no signal for liberty/oppression, which stays ``None`` here and is supplied
by the Claude tagger in a later phase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .foundations import CLASSIC_FOUNDATIONS
from .lexicon import Lexicon, load_seed

_TOKEN_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class DocumentScore:
    """A single document's length-normalized foundation scores."""

    foundations: dict[str, float]  # classic foundation -> per-token rate
    sentiment: float               # net (virtue-vice) signal, length-normalized
    moral_word_ratio: float        # matched moral tokens / total tokens
    word_count: int
    matched_words: int
    scorer: str = "dictionary"
    # liberty is not covered by the dictionary; kept explicit so downstream code
    # never mistakes "unscored" for "zero".
    liberty: None = field(default=None)


class DictionaryScorer:
    """Score documents against a moral-foundations lexicon."""

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon if lexicon is not None else load_seed()

    def score(self, text: str) -> DocumentScore:
        tokens = _TOKEN_RE.findall(text.lower())
        word_count = len(tokens)
        sums = dict.fromkeys(CLASSIC_FOUNDATIONS, 0.0)
        sentiment_sum = 0.0
        matched = 0

        for token in tokens:
            entry = self.lexicon.lookup(token)
            if entry is None:
                continue
            matched += 1
            for foundation, weight in entry.foundations.items():
                sums[foundation] += weight
            sentiment_sum += entry.pole

        if word_count == 0:
            return DocumentScore(
                foundations=dict.fromkeys(CLASSIC_FOUNDATIONS, 0.0),
                sentiment=0.0,
                moral_word_ratio=0.0,
                word_count=0,
                matched_words=0,
            )

        foundations = {f: sums[f] / word_count for f in CLASSIC_FOUNDATIONS}
        return DocumentScore(
            foundations=foundations,
            sentiment=sentiment_sum / word_count,
            moral_word_ratio=matched / word_count,
            word_count=word_count,
            matched_words=matched,
        )
