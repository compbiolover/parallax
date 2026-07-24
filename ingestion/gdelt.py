"""GDELT DOC 2.0 client for historical article discovery.

RSS feeds are a rolling window of the latest items; GDELT's DOC 2.0 API indexes
worldwide online news for the trailing ~3 months and can be filtered by outlet
domain, so it is how Parallax backfills *weeks* of coverage per source (the
volume that makes blindspot detection trustworthy).

GDELT returns article metadata only — url, title, publish time, domain — not
body text. That's exactly what the (title-based) clustering needs; bodies are
fetched separately by the backfill only when full moral-foundation scoring is
wanted.

**Rate limits are strict**: the free endpoint allows roughly one request every
five seconds and answers a too-fast request with a plain-text "Please limit
requests" body (not JSON). :class:`GdeltClient` enforces a conservative global
interval and retries on that signal.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_UA = "parallax-research-bot/0.1 (+https://github.com/compbiolover/parallax)"
# GDELT's throttle notice is specifically "Please limit requests to one every 5
# sec". Keep this narrow so genuine query-syntax errors (also plain text) aren't
# misread as rate limits and pointlessly retried.
_RATE_LIMIT_MARKERS = ("please limit",)


@dataclass(frozen=True)
class GdeltArticle:
    url: str
    title: str
    published_utc: str | None
    domain: str


class RateLimited(Exception):
    """GDELT answered with its rate-limit notice instead of results."""


def _default_fetch(url: str, timeout: int, user_agent: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def _parse_seendate(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def parse_articles(body: str) -> list[GdeltArticle]:
    """Parse a GDELT artlist JSON body. Raises :class:`RateLimited` on the
    plain-text rate-limit notice."""
    stripped = body.strip()
    if not stripped:
        return []
    if not stripped.startswith("{"):
        if any(m in stripped.lower() for m in _RATE_LIMIT_MARKERS):
            raise RateLimited(stripped[:120])
        return []
    data = json.loads(stripped)
    out: list[GdeltArticle] = []
    for a in data.get("articles", []):
        url = a.get("url")
        title = (a.get("title") or "").strip()
        if not url or not title:
            continue
        out.append(
            GdeltArticle(
                url=url,
                title=title,
                published_utc=_parse_seendate(a.get("seendate")),
                domain=a.get("domain", ""),
            )
        )
    return out


class GdeltClient:
    """Throttled GDELT DOC 2.0 client.

    ``fetch`` is injectable for testing; ``clock``/``sleep`` are injectable so the
    rate limiter can be tested without real delays.
    """

    def __init__(
        self,
        min_interval: float = 10.0,
        timeout: int = 40,
        user_agent: str = DEFAULT_UA,
        max_retries: int = 3,
        fetch=None,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.min_interval = min_interval
        self.timeout = timeout
        self.user_agent = user_agent
        self.max_retries = max_retries
        self._fetch = fetch or (lambda url: _default_fetch(url, timeout, user_agent))
        self._sleep = sleep
        self._clock = clock
        self._last = 0.0

    def _throttle(self) -> None:
        elapsed = self._clock() - self._last
        if elapsed < self.min_interval:
            self._sleep(self.min_interval - elapsed)
        self._last = self._clock()

    def search_domain(
        self,
        domain: str,
        timespan: str = "14d",
        max_records: int = 250,
        language: str = "english",
    ) -> list[GdeltArticle]:
        """Return recent articles from ``domain`` within ``timespan`` (e.g. "14d")."""
        query = f"domainis:{domain}"
        if language:
            query += f" sourcelang:{language}"
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "sort": "datedesc",
            "maxrecords": str(max(1, min(max_records, 250))),
            "timespan": timespan,
        }
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                body = self._fetch(url)
                return parse_articles(body)
            except RateLimited:
                self._sleep(self.min_interval * (attempt + 1))  # linear backoff
            except (urllib.error.URLError, TimeoutError, ValueError):
                self._sleep(self.min_interval)
        return []
