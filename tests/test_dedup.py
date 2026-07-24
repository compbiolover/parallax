"""Dedup: exact hashing and MinHash near-duplicate detection."""

from __future__ import annotations

from ingestion.dedup import (
    NearDuplicateIndex,
    content_hash,
    document_id,
    minhash_signature,
    normalize_text,
    normalize_url,
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


def test_normalize_url_canonicalizes_equivalent_urls():
    base = "https://foxnews.com/politics/story"
    # scheme, trailing slash, fragment, utm/tracking params, param order all ignored
    variants = [
        "http://foxnews.com/politics/story/",
        "https://foxnews.com/politics/story?utm_source=twitter&utm_medium=social",
        "https://foxnews.com/politics/story#comments",
        "https://foxnews.com/politics/story?fbclid=abc",
    ]
    canon = normalize_url(base)
    assert canon == "foxnews.com/politics/story"
    for v in variants:
        assert normalize_url(v) == canon


def test_normalize_url_keeps_content_query_and_distinguishes_articles():
    a = normalize_url("https://site.com/article?id=1&utm_source=x")
    b = normalize_url("https://site.com/article?id=2")
    assert a == "site.com/article?id=1"      # content param kept, tracking dropped
    assert a != b                            # different articles stay distinct


def test_document_id_matches_across_feed_and_gdelt_forms():
    # feed link (utm-tagged) and GDELT url (clean) -> same document id
    feed = "https://foxnews.com/a/story?utm_campaign=rss"
    gdelt = "http://foxnews.com/a/story/"
    assert document_id(feed, "feed title and body") == document_id(gdelt, "just title")
    # no link -> falls back to content hash of the text
    assert document_id(None, "abc") == content_hash("abc")


def test_signature_roundtrip():
    text = "moral foundations theory compares media diets across many outlets " * 4
    mh = minhash_signature(text, k=5)
    restored = signature_from_list(signature_list(mh))
    assert mh.jaccard(restored) == 1.0
