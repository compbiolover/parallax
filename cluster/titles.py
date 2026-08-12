"""Headline hygiene for the strings a reader actually sees.

Titles land in the datastore exactly as the feed or GDELT handed them over, and
GDELT hands back a tokenized form — ``U . S . Senate``, ``14 %``, ``Century ?``
— with the outlet's name stamped on the end. Everything downstream read those
strings verbatim: the c-TF-IDF cluster labels were built from them, and so were
the story lists in the email. That is how a blindspot came to be titled
"kidney stone · bret · institutional".

Two jobs, both display-only — nothing here touches scoring, embedding, or
clustering, all of which ran on the raw text before it was ever stored:

1. **Detokenize.** Put the punctuation back where a person would write it.
2. **Recognize what is not a headline.** Feed index pages ("Palestine Articles
   - Christianity Today") carry no story. They also cluster *well*, on the
   outlet name they all share, and then lend that name to the cluster — so
   dropping them removes a whole class of nonsense label at the root.

Both are heuristics over titles alone, and both are deliberately conservative:
a mangled real headline is much cheaper than a dropped one.
"""

from __future__ import annotations

import re

# Whitespace before punctuation that never takes a space before it.
_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%\)\]’'])")
_AFTER_OPEN = re.compile(r"([(\[])\s+")
# "U. S. Senate" -> "U.S. Senate". Run twice for three-letter initialisms.
_INITIALS = re.compile(r"(?<=\b[A-Za-z]\.)\s+(?=[A-Za-z]\.)")
# "Trump ' s plan" -> "Trump's plan". Before _BEFORE_PUNCT, which would
# otherwise glue the apostrophe to the wrong side.
_CONTRACTION = re.compile(r"\s*([’'])\s*(s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_WS = re.compile(r"\s+")

# Separators an outlet stamp hides behind, longest first so " — " is not
# mistaken for " - ".
_SEPARATORS = (" — ", " – ", " - ", " | ", " · ", " :: ")

# Lowercase words allowed inside an outlet name ("Christianity Today", "The
# Dispatch", "Voice of America").
_NAME_WORDS = {"the", "of", "and", "for", "in", "on", "at", "a", "an", "com", "org"}

# What a feed's section/index page is called. Matched only at the end of a
# short title, so "Articles of Impeachment Filed Against the Governor" survives.
_INDEX_TAIL = re.compile(
    r"\b(articles?|archives?|topics?|tags?|categor(?:y|ies)|sections?|"
    r"newsletters?|podcasts?|videos?|photos?|galleries|feeds?|rss)\s*$",
    re.IGNORECASE,
)
_SECTION_PAGE = re.compile(
    r"^(home|homepage|latest|latest news|top stories|news|opinion|editorials?|"
    r"world|u\.?s\.?|politics|business|sports|entertainment|more|about|"
    r"subscribe|sign in|log in|page \d+)$",
    re.IGNORECASE,
)

MIN_HEADLINE_WORDS = 3


def clean_title(raw: str | None) -> str | None:
    """Detokenize one stored title and strip its outlet stamp.

    Returns ``None`` for anything that is empty once cleaned — the caller
    decides what an absent title means, which is not always "skip".
    """
    if not raw:
        return None
    text = _WS.sub(" ", raw).strip()
    text = _CONTRACTION.sub(r"\1\2", text)
    text = _BEFORE_PUNCT.sub(r"\1", text)
    text = _AFTER_OPEN.sub(r"\1", text)
    for _ in range(2):
        text = _INITIALS.sub("", text)
    text = _strip_outlet(text)
    text = _join_compound(text)
    text = _WS.sub(" ", text.strip(" -–—|·:")).strip()
    return text or None


def is_boilerplate(title: str) -> bool:
    """True when a cleaned title is a section or index page, not a story.

    Length is part of the test on purpose: "Palestine Articles" is an index,
    "Articles of Confederation Rediscovered in a Maryland Attic" is a story, and
    the only thing separating them at the level of a title is how much else is
    in it.

    The check also looks *before* the first separator, because an index page's
    outlet stamp is what stops :func:`_strip_outlet` from firing on it: "U.S.
    Senate Articles" alone is too short to be a headline the stamp hangs off, so
    the stamp stays, and the whole string is then long enough to pass a naive
    length test.
    """
    if _is_index(title):
        return True
    cut = _first_separator(title)
    return cut is not None and _is_index(title[:cut].strip(), by_length=False)


def _is_index(text: str, by_length: bool = True) -> bool:
    words = text.split()
    if by_length and len(words) < MIN_HEADLINE_WORDS:
        return True
    if _SECTION_PAGE.match(text.strip()):
        return True
    return bool(_INDEX_TAIL.search(text)) and len(words) <= 5


def clean_titles(raw_titles: list[str | None]) -> list[str]:
    """Clean a list of stored titles, drop the non-headlines, de-duplicate.

    Duplicates are compared case-insensitively after cleaning, because stripping
    two outlets off the same syndicated wire headline is exactly how a "story"
    ends up listed three times under one blindspot.

    Cleaning never empties a cluster. When every title in the list reads as
    boilerplate — a cluster of index pages, or short titles from a fixture — the
    cleaned titles come back anyway. The boilerplate rule exists to keep index
    pages from *out-shouting* real headlines in a mixed cluster; applied
    absolutely it would instead hide the cluster, and an empty card says less
    than an ugly one.
    """
    kept: list[str] = []
    relaxed: list[str] = []
    seen: set[str] = set()
    for raw in raw_titles:
        cleaned = clean_title(raw)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        relaxed.append(cleaned)
        if not is_boilerplate(cleaned):
            kept.append(cleaned)
    return kept or relaxed


def _strip_outlet(title: str) -> str:
    """Drop a trailing " - Outlet Name". Once, never twice.

    A second pass looks harmless and is not: GDELT writes a hyphenated compound
    with spaces, so "How to Revitalize a 400 - Year - Old Church - Christianity
    Today" offers three candidate stamps. One pass takes the publisher; two take
    "Old Church" as well and leave a headline about nothing.
    """
    cut = _last_separator(title)
    if cut is None:
        return title
    index, sep = cut
    head, tail = title[:index].strip(), title[index + len(sep) :].strip()
    return head if _looks_like_outlet(head, tail) else title


def _join_compound(title: str) -> str:
    """Re-hyphenate "400 - Year - Old" once the outlet is out of the way.

    Only when two or more spaced hyphens remain, which is what a tokenized
    compound looks like and what a headline using a dash for punctuation almost
    never does. One remaining dash is left alone — it is far more likely to be
    real punctuation than half of a compound.
    """
    if title.count(" - ") < 2:
        return title
    return re.sub(r"(?<=[\w)]) - (?=[\w(])", "-", title)


def _first_separator(title: str) -> int | None:
    positions = [title.find(sep) for sep in _SEPARATORS]
    found = [p for p in positions if p > 0]
    return min(found) if found else None


def _last_separator(title: str) -> tuple[int, str] | None:
    best: tuple[int, str] | None = None
    for sep in _SEPARATORS:
        index = title.rfind(sep)
        if index > 0 and (best is None or index > best[0]):
            best = (index, sep)
    return best


def _looks_like_outlet(head: str, tail: str) -> bool:
    """Is the trailing segment a publisher's stamp rather than headline text?

    Four signals, all of which have to hold: it is short, it is not a sentence,
    what precedes it stands on its own as a headline, and it is capitalized the
    way a name is. A headline whose real second half happens to be title-cased
    ("Trump Meets Xi - Live Updates") loses that half; the trade is deliberate,
    since the half that survives is the one carrying the story.
    """
    words = tail.split()
    if not 1 <= len(words) <= 4:
        return False
    if tail[-1] in ".?!":
        return False
    if len(head.split()) < 4:
        return False
    return all(
        not word[0].isalpha() or word[0].isupper() or word.lower() in _NAME_WORDS for word in words
    )
