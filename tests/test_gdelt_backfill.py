"""GDELT client parsing/throttling, domain resolution, and backfill wiring."""

from __future__ import annotations

import json

import pytest

from ingestion.config import _derive_domain, load_registry
from ingestion.datastore import Datastore
from ingestion.gdelt import GdeltClient, RateLimited, parse_articles
from ingestion.pipeline import PipelineConfig, backfill

SAMPLE = json.dumps({
    "articles": [
        {"url": "https://foxnews.com/a", "title": "Story A",
         "seendate": "20260724T014500Z", "domain": "foxnews.com"},
        {"url": "https://foxnews.com/b", "title": "Story B",
         "seendate": "20260723T120000Z", "domain": "foxnews.com"},
        {"url": "", "title": "no url — skipped", "seendate": "", "domain": "foxnews.com"},
    ]
})


# -- parsing ---------------------------------------------------------------

def test_parse_articles_extracts_and_skips_incomplete():
    arts = parse_articles(SAMPLE)
    assert [a.title for a in arts] == ["Story A", "Story B"]
    assert arts[0].url == "https://foxnews.com/a"
    assert arts[0].published_utc.startswith("2026-07-24T01:45:00")
    assert arts[0].domain == "foxnews.com"


def test_parse_articles_rate_limit_notice_raises():
    with pytest.raises(RateLimited):
        parse_articles("Please limit requests to one every 5 sec")


def test_parse_articles_empty_and_nonjson():
    assert parse_articles("") == []
    assert parse_articles("<html>garbage</html>") == []


def test_query_error_is_not_treated_as_rate_limit():
    # A GDELT query-syntax error is plain text but not the throttle notice —
    # it should parse to [] (no retry), not raise RateLimited.
    assert parse_articles("Your query was not valid. Please check the syntax.") == []


# -- client throttle + retry ----------------------------------------------

def test_client_throttles_between_calls():
    slept: list[float] = []
    t = {"now": 0.0}

    def clock():
        return t["now"]

    def sleep(s):
        slept.append(s); t["now"] += s

    client = GdeltClient(min_interval=8.0, fetch=lambda url: SAMPLE, sleep=sleep, clock=clock)
    client.search_domain("foxnews.com")
    client.search_domain("foxnews.com")  # second call must wait ~8s
    assert any(abs(s - 8.0) < 1e-6 for s in slept)


def test_client_retries_on_rate_limit_then_succeeds():
    calls = {"n": 0}

    def fetch(url):
        calls["n"] += 1
        return "Please limit requests" if calls["n"] == 1 else SAMPLE

    client = GdeltClient(min_interval=0.0, fetch=fetch, sleep=lambda s: None, clock=lambda: 0.0)
    arts = client.search_domain("foxnews.com")
    assert calls["n"] == 2 and len(arts) == 2


# -- domain resolution -----------------------------------------------------

def test_derive_domain_strips_feed_subdomains():
    assert _derive_domain("https://moxie.foxnews.com/x.xml") == "foxnews.com"
    assert _derive_domain("https://rss.nytimes.com/s.xml") == "nytimes.com"
    assert _derive_domain("https://feeds.bbci.co.uk/news/world/rss.xml") == "bbci.co.uk"
    assert _derive_domain(None) is None


def test_registry_backfillable_includes_ap_excludes_podcasts():
    reg = load_registry()
    domains = {s.id: s.domain for s in reg.backfillable()}
    assert domains.get("self_ap_topnews") == "apnews.com"   # null RSS, still backfillable
    assert domains.get("self_bbc_world") == "bbc.com"       # explicit override
    assert "ce_relatable_stuckey" not in domains            # podcast, no domain


# -- backfill wiring (fake GDELT client, no network) -----------------------

class _FakeGdelt:
    def __init__(self, per_domain):
        self.per_domain = per_domain
        self.queried = []

    def search_domain(self, domain, timespan="14d", max_records=250, language="english"):
        self.queried.append(domain)
        return self.per_domain.get(domain, [])


def test_feed_and_backfill_of_same_url_collapse_to_one_doc():
    from ingestion.dedup import NearDuplicateIndex
    from ingestion.pipeline import RunStats, _ingest_one
    from scoring.dictionary import DictionaryScorer
    from cluster.embed import HashingEmbedder

    reg = load_registry()
    source = reg.backfillable()[0]
    store = Datastore(":memory:")
    scorer, embedder, index = DictionaryScorer(), HashingEmbedder(dim=32), NearDuplicateIndex()
    stats = RunStats()
    # feed form: utm-tagged; backfill form: clean, http, trailing slash
    _ingest_one(store, source, scorer, embedder, index, stats,
                title="Nuclear deal signed with Saudi Arabia today", link="https://x.com/a?utm_source=rss",
                published_utc=None, text="body one two three four five six", cluster_text="Nuclear deal", min_words=3)
    _ingest_one(store, source, scorer, embedder, index, stats,
                title="Nuclear deal signed with Saudi Arabia today", link="http://x.com/a/",
                published_utc=None, text="only a title", cluster_text="Nuclear deal", min_words=3)
    assert stats.stored == 1
    assert stats.exact_duplicates == 1  # second collapsed onto the first via URL identity
    store.close()


def test_backfill_stores_titles_and_dedups_domains():
    from ingestion.gdelt import GdeltArticle
    reg = load_registry()
    # give every backfillable domain one article
    per = {}
    for s in reg.backfillable():
        per.setdefault(s.domain, [
            GdeltArticle(url=f"https://{s.domain}/x", title=f"Headline about {s.domain} policy news today",
                         published_utc="2026-07-20T00:00:00+00:00", domain=s.domain)
        ])
    fake = _FakeGdelt(per)
    store = Datastore(":memory:")
    stats = backfill(store, reg, PipelineConfig(), gdelt=fake, days=14)
    # npr.org appears twice in the registry but must be queried once per diet
    assert len(fake.queried) == len(set(fake.queried))
    assert stats.stored > 0
    assert store.embedding_count() == stats.stored  # titles embedded for clustering
    store.close()
