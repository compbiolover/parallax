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

Optionally, the fairness score is further partitioned into equality and
proportionality (MFQ-2) by :mod:`scoring.fairness_split`. That partition divides
the fairness mass this scorer already found; it never creates more of it, and it
reports ``None`` rather than guessing when a document gives it nothing to go on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .fairness_split import FairnessSplit, FairnessSplitter, apply_split
from .foundations import CLASSIC_FOUNDATIONS
from .lexicon import Lexicon, load_seed

_TOKEN_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class DocumentScore:
    """A single document's length-normalized foundation scores."""

    foundations: dict[str, float]  # classic foundation -> per-token rate
    sentiment: float  # net (virtue-vice) signal, length-normalized
    moral_word_ratio: float  # matched moral tokens / total tokens
    word_count: int
    matched_words: int
    scorer: str = "dictionary"
    # liberty is not covered by the dictionary; kept explicit so downstream code
    # never mistakes "unscored" for "zero".
    liberty: None = field(default=None)
    # The MFQ-2 halves of `fairness`, present only when the splitter was enabled
    # *and* the document carried enough evidence. None means "not partitioned",
    # which is a different claim from an even split.
    equality: float | None = field(default=None)
    proportionality: float | None = field(default=None)
    fairness_split: FairnessSplit | None = field(default=None, repr=False)


class DictionaryScorer:
    """Score documents against a moral-foundations lexicon.

    ``assignment`` controls how a word's foundation weights are aggregated:

    - ``"argmax"`` (default) — each matched word contributes to its single
      dominant foundation only. This is the standard reduction for a
      *probabilistic* lexicon like the eMFD, where every word carries mass on
      all five foundations: summing the raw probabilities makes every document
      collapse toward the lexicon's base-rate distribution, so profiles barely
      discriminate between corpora. Argmax restores discrimination.
    - ``"probability"`` — sum every foundation's weight for each word (the raw
      bag-of-words probability aggregate). Kept for comparison/analysis.

    For a single-foundation lexicon (the built-in seed, classic MFD word lists)
    the two modes are identical.

    ``splitter`` optionally partitions the fairness score into equality and
    proportionality. Pass ``FairnessSplitter()`` to enable it, or leave it
    ``None`` to score the classic five only. Enabling it never changes any of
    the five foundation values — it only adds the two derived ones.
    """

    def __init__(
        self,
        lexicon: Lexicon | None = None,
        assignment: str = "argmax",
        splitter: FairnessSplitter | None = None,
    ) -> None:
        self.lexicon = lexicon if lexicon is not None else load_seed()
        if assignment not in ("argmax", "probability"):
            raise ValueError(f"unknown assignment mode: {assignment!r}")
        self.assignment = assignment
        self.splitter = splitter

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
            if self.assignment == "argmax":
                # Assign the word to its dominant foundation only. Ties break by
                # canonical order via max()'s first-max behavior over the dict.
                foundation = max(entry.foundations, key=entry.foundations.get)
                sums[foundation] += entry.foundations[foundation]
            else:
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
        # The split reads the raw tokens, not the lexicon matches: whether a
        # document argues about equality or about desert is carried by words the
        # foundation lexicon may never have flagged as moral at all.
        split = self.splitter.split(tokens) if self.splitter is not None else None
        halves = apply_split(foundations["fairness"], split)
        return DocumentScore(
            foundations=foundations,
            sentiment=sentiment_sum / word_count,
            moral_word_ratio=matched / word_count,
            word_count=word_count,
            matched_words=matched,
            equality=halves["equality"],
            proportionality=halves["proportionality"],
            fairness_split=split,
        )
