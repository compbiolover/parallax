"""Fetch feeds and extract article bodies.

- Feeds are parsed with ``feedparser`` (RSS/Atom).
- Article bodies are extracted with ``trafilatura`` from the linked page; if that
  fails we fall back to the feed's own summary (HTML stripped).
- ``robots.txt`` is honored per host, and requests are rate-limited per host.

Raw HTML/text never leaves this module as anything persisted — callers score it
and discard it (``CLAUDE.md`` §0).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import feedparser
import trafilatura

DEFAULT_UA = "parallax-research-bot/0.1 (+https://github.com/compbiolover/parallax)"
DEFAULT_TIMEOUT = 30


@dataclass(frozen=True)
class FeedItem:
    title: str
    link: str | None
    published_utc: str | None
    summary: str


def parse_feed(url: str, user_agent: str = DEFAULT_UA) -> list[FeedItem]:
    """Fetch and parse an RSS/Atom feed into items."""
    parsed = feedparser.parse(url, agent=user_agent)
    items: list[FeedItem] = []
    for e in parsed.entries:
        items.append(
            FeedItem(
                title=e.get("title", "").strip(),
                link=e.get("link"),
                published_utc=_entry_published(e),
                summary=e.get("summary", ""),
            )
        )
    return items


def _entry_published(entry) -> str | None:
    # feedparser entries are dict-like (FeedParserDict); both fields are UTC
    # struct_time tuples when present.
    pp = entry.get("published_parsed") or entry.get("updated_parsed")
    if not pp:
        return None
    try:
        return datetime(*pp[:6], tzinfo=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


class RobotsCache:
    """Per-host robots.txt cache."""

    def __init__(self, user_agent: str = DEFAULT_UA, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: dict[str, RobotFileParser | None] = {}

    def can_fetch(self, url: str) -> bool:
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._cache:
            self._cache[host] = self._load(host)
        rp = self._cache[host]
        if rp is None:  # robots.txt unreachable -> do not block ingestion
            return True
        return rp.can_fetch(self.user_agent, url)

    def _load(self, host: str) -> RobotFileParser | None:
        rp = RobotFileParser()
        try:
            req = urllib.request.Request(
                f"{host}/robots.txt", headers={"User-Agent": self.user_agent}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                rp.parse(resp.read().decode("utf-8", "ignore").splitlines())
            return rp
        except (urllib.error.URLError, ValueError, TimeoutError):
            return None


class RateLimiter:
    """Simple per-host minimum-interval limiter."""

    def __init__(self, per_host_rpm: int = 20) -> None:
        self.min_interval = 60.0 / per_host_rpm if per_host_rpm > 0 else 0.0
        self._last: dict[str, float] = {}

    def wait(self, url: str, *, sleep=time.sleep, clock=time.monotonic) -> None:
        if self.min_interval <= 0:
            return
        host = urlsplit(url).netloc
        now = clock()
        elapsed = now - self._last.get(host, 0.0)
        if elapsed < self.min_interval:
            sleep(self.min_interval - elapsed)
        self._last[host] = clock()


def extract_article(
    url: str,
    *,
    user_agent: str = DEFAULT_UA,
    timeout: int = DEFAULT_TIMEOUT,
    robots: RobotsCache | None = None,
    rate_limiter: RateLimiter | None = None,
) -> str | None:
    """Fetch ``url`` and extract the main article text, or None on failure."""
    if robots is not None and not robots.can_fetch(url):
        return None
    if rate_limiter is not None:
        rate_limiter.wait(url)
    html = _fetch(url, user_agent=user_agent, timeout=timeout)
    if html is None:
        return None
    return trafilatura.extract(html, url=url, favor_precision=True)


def _fetch(url: str, *, user_agent: str, timeout: int) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, "ignore")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def strip_html(html: str) -> str:
    """Best-effort text from an HTML fragment (feed-summary fallback)."""
    extracted = trafilatura.extract(f"<html><body>{html}</body></html>")
    if extracted:
        return extracted
    # Last resort: crude tag strip.
    import re

    return re.sub(r"<[^>]+>", " ", html)
