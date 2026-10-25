"""How much one document counts, by how much of it we actually have.

Ingestion records *how* a document's text was obtained; this decides what that
is worth. Keeping the two apart is the point: the recording step is the only
one that knows the answer, and the weighting rule is a scoring judgement that
several aggregation paths have to share.

A headline and a 1,100-word feature are not one document each. GDELT backfill
stores titles without bodies, and store-wide they run 9,692 against 13,223
full articles. Counted equally they dominate the profile, and a 12-word
headline is a bad place to measure moral emphasis: length normalization fixes
the arithmetic, not the fact that there is almost no moral vocabulary in a
headline to find.

So a headline counts, at a quarter. Not zero: it is real evidence that an
outlet ran something, and dropping it would bias the corpus toward whoever
publishes in a format the extractor happens to handle.

**Keyed on provenance, not on length.** An earlier version used a 120-word
floor and demoted 1,297 documents that had real text — 421 from PBS NewsHour
and 338 from The Hill, outlets whose article extraction fails and whose RSS
summary is what gets stored. That is exactly the bias this weighting exists to
remove, pointed the wrong way: extraction depth is not random, so a length
test measures which outlets the scraper can read.
"""

from __future__ import annotations

# How a document's text was obtained, recorded on `documents.ingest_kind`.
BODY = "body"  # the article itself, extracted from the page
SUMMARY = "summary"  # the feed's own blurb: extraction failed or was refused
TITLE_ONLY = "title_only"  # GDELT backfill, which fetches no page at all

# What a headline counts for, against 1.0 for a document with text.
STUB_WEIGHT = 0.25

# Fallback threshold for rows written before `ingest_kind` existed. The same
# proxy the migration backfills with, and the same known error; nothing
# ingested since relies on it.
LEGACY_STUB_WORDS = 30


def is_stub(ingest_kind: str | None, word_count: int | None = None) -> bool:
    """Is this a headline rather than a story?

    ``ingest_kind`` is authoritative when present. A ``SUMMARY`` is not a stub:
    it is shorter than an article and it is still the publisher's own prose
    about the story.
    """
    if ingest_kind:
        return ingest_kind == TITLE_ONLY
    return (word_count or 0) < LEGACY_STUB_WORDS


def depth_weight(ingest_kind: str | None, word_count: int | None = None) -> float:
    """What one document contributes to an aggregate."""
    return STUB_WEIGHT if is_stub(ingest_kind, word_count) else 1.0


def row_weights(rows, weights_by_source: dict[str, float]) -> list[float]:
    """Per-row aggregation weight: the persona's source weight times depth.

    One function because there are four places that turn stored score rows
    into a weighted composition — ``persona_profiles``, ``compare.sensitivity``,
    ``compare.weekly`` and ``compare.confidence``. When the depth factor lived
    in only one of them, ``make sensitivity`` stress-tested a baseline the
    dashboard never showed and flipped the sign on an authority claim.

    Rows must carry ``source_id`` and ``ingest_kind``; ``word_count`` is read
    when present, for the legacy fallback.
    """
    return [
        weights_by_source.get(row["source_id"], 0.0)
        * depth_weight(_get(row, "ingest_kind"), _get(row, "word_count"))
        for row in rows
    ]


def _get(row, key: str):
    """``row[key]`` for a sqlite3.Row that may predate the column."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return None
