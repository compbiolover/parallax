"""Dedup: exact hashing and MinHash near-duplicate detection."""

from __future__ import annotations

from ingestion.dedup import (
    NearDuplicateIndex,
    content_hash,
    minhash_signature,
    normalize_text,
    signature_from_list,
    signature_list,
)


def test_content_hash_ignores_whitespace_and_case():
    a = content_hash("Hello   World\n")
    b = content_hash("hello world")
    assert a == b


def test_content_hash_distinguishes_content():
    assert content_hash("the quick brown fox") != content_hash("the slow brown fox")


def test_normalize_text():
    assert normalize_text("  A\t B\nC ") == "a b c"


def test_minhash_near_duplicate_detected():
    # A syndicated wire story reprinted with one edited word — the common case
    # dedup must catch. Long, varied body so a single change barely moves Jaccard.
    base = (
        "washington lawmakers passed a sweeping funding bill on thursday after weeks "
        "of negotiation over defense spending and domestic programs the measure now "
        "heads to the senate where its future remains uncertain amid partisan divisions "
        "analysts said the compromise reflected pressure from both parties to avoid a "
        "government shutdown before the end of the fiscal year according to officials"
    )
    edited = base.replace("uncertain", "unclear")  # single-word reprint edit
    index = NearDuplicateIndex(threshold=0.7)
    index.add("doc1", minhash_signature(base, k=5))
    assert index.find_duplicate(minhash_signature(edited, k=5)) == "doc1"


def test_minhash_distinct_not_flagged():
    a = "climate summit opens with pledges on emissions and renewable energy targets " * 3
    b = "the local baseball team won the championship in extra innings last night " * 3
    index = NearDuplicateIndex(threshold=0.7)
    index.add("a", minhash_signature(a, k=5))
    assert index.find_duplicate(minhash_signature(b, k=5)) is None


def test_signature_roundtrip():
    text = "moral foundations theory compares media diets across many outlets " * 4
    mh = minhash_signature(text, k=5)
    restored = signature_from_list(signature_list(mh))
    assert mh.jaccard(restored) == 1.0
