"""End-to-end Phase 1 pipeline: fetch -> extract -> dedup -> score -> store.

One pass per document. Text is fetched, scored, hashed, and signed for
near-duplicate detection in memory, then only the derived metrics are persisted
— the raw body is discarded (``CLAUDE.md`` §0). Exact duplicates collapse on the
content-hash primary key; near-duplicates are flagged via an LSH seeded from
previously-stored signatures so dedup holds across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from scoring.aggregate import aggregate_profile, to_composition
from scoring.dictionary import DictionaryScorer, DocumentScore

from .config import Registry, Source, load_registry, load_settings
from .datastore import Datastore
from .dedup import (
    NearDuplicateIndex,
    content_hash,
    minhash_signature,
    signature_from_list,
    signature_list,
)
from .extract import (
    DEFAULT_UA,
    FeedItem,
    RateLimiter,
    RobotsCache,
    extract_article,
    parse_feed,
    strip_html,
)


@dataclass
class PipelineConfig:
    max_items_per_feed: int = 25
    min_words: int = 50
    near_dup_threshold: float = 0.85
    user_agent: str = DEFAULT_UA
    timeout: int = 30
    per_host_rpm: int = 20
    respect_robots: bool = True

    @classmethod
    def from_settings(cls, settings: dict) -> PipelineConfig:
        ing = settings.get("ingestion", {}) or {}
        dedup = (settings.get("dedup", {}) or {}).get("near_duplicate", {}) or {}
        rate = (ing.get("rate_limit", {}) or {})
        return cls(
            user_agent=ing.get("user_agent", DEFAULT_UA),
            timeout=int(ing.get("request_timeout_seconds", 30)),
            per_host_rpm=int(rate.get("per_host_requests_per_minute", 20)),
            respect_robots=bool(ing.get("respect_robots_txt", True)),
            near_dup_threshold=float(dedup.get("minhash_threshold", 0.85)),
        )


@dataclass
class RunStats:
    fetched: int = 0
    stored: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    skipped_short: int = 0
    errors: int = 0
    per_diet: dict[str, int] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _document_text(item: FeedItem, cfg: PipelineConfig, robots, limiter) -> str | None:
    body: str | None = None
    if item.link:
        try:
            body = extract_article(
                item.link,
                user_agent=cfg.user_agent,
                timeout=cfg.timeout,
                robots=robots,
                rate_limiter=limiter,
            )
        except Exception:
            body = None
    if not body and item.summary:
        body = strip_html(item.summary)
    if not body:
        return None
    return f"{item.title}\n\n{body}".strip()


def run(
    store: Datastore,
    registry: Registry | None = None,
    config: PipelineConfig | None = None,
    scorer: DictionaryScorer | None = None,
) -> RunStats:
    """Ingest every RSS source with a URL, scoring and deduping into ``store``."""
    registry = registry or load_registry()
    cfg = config or PipelineConfig()
    scorer = scorer or DictionaryScorer()
    robots = RobotsCache(cfg.user_agent, cfg.timeout) if cfg.respect_robots else None
    limiter = RateLimiter(cfg.per_host_rpm)
    index = _seed_index(store, cfg.near_dup_threshold)
    stats = RunStats()

    for source in registry.ingestable(("rss",)):
        _ingest_source(source, store, cfg, scorer, robots, limiter, index, stats)
    return stats


def _seed_index(store: Datastore, threshold: float) -> NearDuplicateIndex:
    index = NearDuplicateIndex(threshold=threshold)
    for doc_id, sig in store.iter_minhash_signatures():
        try:
            index.add(doc_id, signature_from_list(sig))
        except Exception:
            continue
    return index


def _ingest_source(
    source: Source,
    store: Datastore,
    cfg: PipelineConfig,
    scorer: DictionaryScorer,
    robots: RobotsCache | None,
    limiter: RateLimiter,
    index: NearDuplicateIndex,
    stats: RunStats,
) -> None:
    try:
        items = parse_feed(source.url, cfg.user_agent)  # type: ignore[arg-type]
    except Exception:
        stats.errors += 1
        return

    for item in items[: cfg.max_items_per_feed]:
        stats.fetched += 1
        text = _document_text(item, cfg, robots, limiter)
        if text is None:
            stats.errors += 1
            continue

        doc_id = content_hash(text)
        if store.has_document(doc_id):
            stats.exact_duplicates += 1
            continue

        score: DocumentScore = scorer.score(text)
        if score.word_count < cfg.min_words:
            stats.skipped_short += 1
            continue

        mh = minhash_signature(text, k=5)
        dup_of = index.find_duplicate(mh)
        is_dup = dup_of is not None

        store.upsert_document(
            doc_id=doc_id,
            diet_id=source.diet_id,
            source_id=source.id,
            stratum_id=source.stratum_id,
            url=item.link,
            title=item.title,
            published_utc=item.published_utc,
            fetched_utc=_now_iso(),
            word_count=score.word_count,
            minhash=signature_list(mh),
            weight=source.diet_weight,
            is_duplicate=is_dup,
            duplicate_of=dup_of,
        )
        store.upsert_scores(
            document_id=doc_id,
            scorer=score.scorer,
            foundations=score.foundations,
            sentiment=score.sentiment,
            moral_word_ratio=score.moral_word_ratio,
            matched_words=score.matched_words,
            liberty=score.liberty,
        )

        if is_dup:
            stats.near_duplicates += 1
        else:
            index.add(doc_id, mh)
            stats.stored += 1
            stats.per_diet[source.diet_id] = stats.per_diet.get(source.diet_id, 0) + 1


def diet_profiles(store: Datastore, scorer_name: str = "dictionary") -> dict[str, dict[str, float]]:
    """Build a normalized foundation composition per diet from stored scores."""
    from scoring.foundations import CLASSIC_FOUNDATIONS

    profiles: dict[str, dict[str, float]] = {}
    for diet_id in store.diet_ids():
        rows = store.scores_for_diet(diet_id, scorer_name)
        scores = [
            DocumentScore(
                foundations={f: (row[f] or 0.0) for f in CLASSIC_FOUNDATIONS},
                sentiment=row["sentiment"] or 0.0,
                moral_word_ratio=row["moral_word_ratio"] or 0.0,
                word_count=1,
                matched_words=row["matched_words"] or 0,
            )
            for row in rows
        ]
        weights = [row["weight"] or 1.0 for row in rows]
        if scores:
            profiles[diet_id] = to_composition(aggregate_profile(scores, weights))
    return profiles


def load_config() -> tuple[Registry, PipelineConfig]:
    """Convenience: load registry + pipeline config from disk."""
    return load_registry(), PipelineConfig.from_settings(load_settings())
