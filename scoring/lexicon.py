"""Moral-foundations lexicon: loading and term lookup.

A ``Lexicon`` maps terms to per-foundation weights and a sentiment pole. Two
loaders are provided:

- :func:`load_seed` — the small built-in demo lexicon (``seed_lexicon.py``).
- :func:`load_emfd_csv` — the eMFD in CSV form, the intended production lexicon.
  The eMFD ships a row per word with continuous foundation probabilities; point
  this at that file (kept outside git — data is gitignored) for real scoring.

Matching supports MFD-style wildcards: a stem may match by exact token or by
prefix. Seed stems of length >= ``MIN_PREFIX_LEN`` match by prefix so common
inflections are caught without an explicit lemmatizer; shorter stems and eMFD
rows match exactly unless they carry a trailing ``*``.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .foundations import CLASSIC_FOUNDATIONS
from .seed_lexicon import SEED_LEXICON

MIN_PREFIX_LEN = 4


@dataclass(frozen=True)
class Entry:
    """A scored lexicon term: foundation weights plus a sentiment pole (+1/-1)."""

    foundations: dict[str, float]
    pole: int


class Lexicon:
    """Term lookup with exact and longest-prefix (wildcard) matching."""

    def __init__(self) -> None:
        self._exact: dict[str, Entry] = {}
        # (stem, entry), kept sorted by descending stem length for longest match.
        self._prefixes: list[tuple[str, Entry]] = []

    def add(self, term: str, entry: Entry, *, wildcard: bool) -> None:
        term = term.lower()
        if wildcard:
            self._prefixes.append((term, entry))
            self._prefixes.sort(key=lambda pair: len(pair[0]), reverse=True)
        else:
            self._exact[term] = entry

    def lookup(self, token: str) -> Entry | None:
        """Return the entry for a token: exact match first, then longest prefix."""
        entry = self._exact.get(token)
        if entry is not None:
            return entry
        for stem, prefix_entry in self._prefixes:
            if token.startswith(stem):
                return prefix_entry
        return None

    def __len__(self) -> int:
        return len(self._exact) + len(self._prefixes)

    def items(self) -> Iterator[tuple[str, Entry]]:
        """Every (term, entry) pair, exact matches first then wildcard stems.

        Lets tooling inspect the vocabulary that was actually loaded rather than
        re-parsing the source file, so an audit sees the same lexicon the scorer
        does (see ``validation/lexicon_audit.py``).
        """
        yield from self._exact.items()
        yield from self._prefixes


logger = logging.getLogger(__name__)

SEED_NAME = "built-in demo seed"


def build_lexicon(path: str | Path | None = None) -> tuple[Lexicon, str]:
    """Return ``(lexicon, provenance_name)`` for the pipeline.

    With a readable ``path``, loads the eMFD CSV and names it after the file;
    otherwise returns the built-in demo seed. The name is recorded in the
    datastore so summaries and the dashboard can state which lexicon produced
    the scores (and soften the demo caveat once the real eMFD is in use).

    A configured path that does not resolve is **warned about**, not silently
    ignored. ``config/settings.example.yaml`` ships pointing at
    ``data/emfd_scoring.csv``, which is gitignored and absent until you download
    it — so the default configuration lands in exactly this branch, and the only
    signal used to be a caveat at the bottom of a rendered page. A whole corpus
    can get scored by a demo lexicon before anyone notices.
    """
    if path:
        p = Path(path)
        if p.exists():
            return load_emfd_csv(p), f"eMFD ({p.name})"
        logger.warning(
            "lexicon_path is set to %s but no such file exists — scoring with the "
            "%s instead, which is illustrative only and not a validated instrument. "
            "Download emfd_scoring.csv from the eMFDscore repo and put it there, or "
            "clear scoring.taggers.dictionary.lexicon_path to silence this.",
            p,
            SEED_NAME,
        )
    return load_seed(), SEED_NAME


def is_demo_lexicon(name: str | None) -> bool:
    """True when the active lexicon is the built-in demo seed (or unknown)."""
    return not name or name == SEED_NAME


def load_seed() -> Lexicon:
    """Build the built-in demo lexicon."""
    lex = Lexicon()
    for term, (foundations, pole) in SEED_LEXICON.items():
        wildcard = term.endswith("*") or len(term) >= MIN_PREFIX_LEN
        stem = term.rstrip("*")
        lex.add(stem, Entry(foundations=dict(foundations), pole=pole), wildcard=wildcard)
    return lex


def load_emfd_csv(path: str | Path) -> Lexicon:
    """Load the eMFD from CSV.

    Expected columns: a ``word`` column plus one probability column per classic
    foundation named ``<foundation>_p`` (e.g. ``care_p``) and, optionally,
    matching ``<foundation>_sent`` sentiment columns. Words ending in ``*`` are
    treated as wildcard stems. Unknown columns are ignored, so the loader
    tolerates the eMFD's extra fields.
    """
    path = Path(path)
    lex = Lexicon()
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        word_col = _find_word_column(reader.fieldnames)
        for row in reader:
            term = (row.get(word_col) or "").strip()
            if not term:
                continue
            foundations: dict[str, float] = {}
            sentiment = 0.0
            for foundation in CLASSIC_FOUNDATIONS:
                prob = _to_float(row.get(f"{foundation}_p"))
                if prob and prob > 0:
                    foundations[foundation] = prob
                sentiment += _to_float(row.get(f"{foundation}_sent"))
            if not foundations:
                continue
            pole = 1 if sentiment >= 0 else -1
            wildcard = term.endswith("*")
            lex.add(term.rstrip("*"), Entry(foundations=foundations, pole=pole), wildcard=wildcard)
    return lex


def _find_word_column(fieldnames: list[str]) -> str:
    for candidate in ("word", "term", "lemma", "token"):
        for name in fieldnames:
            if name.strip().lower() == candidate:
                return name
    return fieldnames[0]


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
