"""End-to-end Phase 1 pipeline: fetch -> extract -> dedup -> score -> store.

One pass per document. Text is fetched, scored, hashed, and signed for
near-duplicate detection in memory, then only the derived metrics are persisted
— the raw body is discarded (``CLAUDE.md`` §0). Exact duplicates collapse on the
content-hash primary key; near-duplicates are flagged via an LSH seeded from
previously-stored signatures so dedup holds across runs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cluster.embed import Embedder, HashingEmbedder
from scoring.aggregate import aggregate_profile, to_composition
from scoring.dictionary import DictionaryScorer, DocumentScore
from scoring.lexicon import build_lexicon

from .config import Registry, Source, load_registry, load_settings
from .datastore import Datastore
from .dedup import (
    NearDuplicateIndex,
    document_id,
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

if TYPE_CHECKING:  # import cost stays off the hot path; annotations are lazy
    from .gdelt import GdeltClient

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    max_items_per_feed: int = 25
    min_words: int = 50
    near_dup_threshold: float = 0.85
    user_agent: str = DEFAULT_UA
    timeout: int = 30
    per_host_rpm: int = 20
    respect_robots: bool = True
    lexicon_path: str | None = None  # eMFD CSV; None -> built-in demo seed
    assignment: str = "argmax"       # 'argmax' | 'probability' (see DictionaryScorer)
    # Partition fairness into equality vs proportionality (MFQ-2). Cheap — it is
    # a second pass over tokens already in memory — but unvalidated, so it is
    # opt-out rather than load-bearing: the classic five are unaffected either way.
    split_fairness: bool = True
    fairness_min_evidence: int = 2
    # Transformer tagger (Mformer) run alongside the dictionary at ingestion, so
    # every article carries both estimates and the dashboard can show a
    # dictionary-vs-transformer confidence band. Requires parallax[scoring]; when
    # the deps are missing the pipeline logs once and continues dictionary-only.
    transformer_enabled: bool = True
    transformer_model: str | None = None    # 'mformer' alias or a HF prefix
    transformer_revision: str | None = None  # pin the HF revision (recommended)
    transformer_max_length: int = 256
    # Liberty/oppression via Claude — the sixth foundation, which no dictionary
    # and no available transformer covers. Costs money per document, so it is
    # off unless a key is present; the run completes without it either way.
    liberty_enabled: bool = True
    liberty_model: str | None = None      # None -> scoring.liberty.DEFAULT_MODEL
    liberty_effort: str | None = None     # None -> DEFAULT_EFFORT ('low')
    liberty_batch: bool = True            # Batch API: half price, overnight-friendly

    @classmethod
    def from_settings(cls, settings: dict) -> PipelineConfig:
        ing = settings.get("ingestion", {}) or {}
        dedup = (settings.get("dedup", {}) or {}).get("near_duplicate", {}) or {}
        rate = (ing.get("rate_limit", {}) or {})
        taggers = ((settings.get("scoring", {}) or {}).get("taggers", {}) or {})
        dict_cfg = taggers.get("dictionary", {}) or {}
        tr_cfg = taggers.get("transformer", {}) or {}
        lib_cfg = taggers.get("liberty", {}) or {}
        return cls(
            user_agent=ing.get("user_agent", DEFAULT_UA),
            timeout=int(ing.get("request_timeout_seconds", 30)),
            per_host_rpm=int(rate.get("per_host_requests_per_minute", 20)),
            respect_robots=bool(ing.get("respect_robots_txt", True)),
            near_dup_threshold=float(dedup.get("minhash_threshold", 0.85)),
            lexicon_path=dict_cfg.get("lexicon_path"),
            assignment=dict_cfg.get("assignment", "argmax"),
            split_fairness=bool(dict_cfg.get("split_fairness", True)),
            fairness_min_evidence=int(dict_cfg.get("fairness_min_evidence", 2)),
            transformer_enabled=bool(tr_cfg.get("enabled", True)),
            transformer_model=tr_cfg.get("model"),
            transformer_revision=tr_cfg.get("revision"),
            transformer_max_length=int(tr_cfg.get("max_length", 256)),
            liberty_enabled=bool(lib_cfg.get("enabled", True)),
            liberty_model=lib_cfg.get("model"),
            liberty_effort=lib_cfg.get("effort"),
            liberty_batch=bool(lib_cfg.get("batch", True)),
        )


@dataclass
class RunStats:
    fetched: int = 0
    stored: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    skipped_short: int = 0
    errors: int = 0
    liberty_scored: int = 0
    per_diet: dict[str, int] = field(default_factory=dict)


@dataclass
class SourceProgress:
    """What one source did, reported the moment it finishes.

    A run walks every source sequentially, each waiting on the network behind a
    30-second timeout and a per-host rate limit. Without this the terminal is
    blank for minutes and a slow source is indistinguishable from a hang — the
    first thing that went wrong on the first real run of this pipeline.
    """

    index: int              # 1-based position in the run
    total: int
    source: Source
    stored: int             # documents new to the datastore
    fetched: int            # feed items seen
    errors: int
    seconds: float
    failed: bool = False    # the feed itself could not be parsed


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _cluster_text(item: FeedItem, text: str) -> str:
    """Text used for the story embedding.

    Headlines carry the story signal; full bodies share boilerplate and generic
    vocabulary that washes out topical structure (empirically, body embeddings
    collapse into one undifferentiated blob). Embed the title, backfilling with
    the body lead only when the title is too short to embed reliably.
    """
    title = (item.title or "").strip()
    if len(title.split()) >= 4:
        return title
    return f"{title} {text[:300]}".strip()


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
    embedder: Embedder | None = None,
    transformer: object | None = None,
    liberty_tagger: object | None = None,
    progress: Callable[[SourceProgress | str], None] | None = None,
) -> RunStats:
    """Ingest every RSS source with a URL, scoring and deduping into ``store``.

    ``progress`` is called after each source with a :class:`SourceProgress`, and
    with a plain string for the occasional free-text note (the Batch API wait).
    Omit it and the run is silent, which is what the library callers and the
    tests want; the CLI passes a reporter.
    """
    registry = registry or load_registry()
    cfg = config or PipelineConfig()
    if scorer is None:
        lexicon, lexicon_name = build_lexicon(cfg.lexicon_path)
        scorer = DictionaryScorer(lexicon, assignment=cfg.assignment,
                                  splitter=_build_splitter(cfg))
    else:
        lexicon_name = "injected"
    if embedder is None:
        embedder = HashingEmbedder()
    if transformer is None:
        transformer = _build_transformer(cfg)
    store.set_meta("lexicon", lexicon_name)
    store.set_meta("embedder", getattr(embedder, "name", type(embedder).__name__))
    _store_transformer_meta(store, transformer)
    robots = RobotsCache(cfg.user_agent, cfg.timeout) if cfg.respect_robots else None
    limiter = RateLimiter(cfg.per_host_rpm)
    index = _seed_index(store, cfg.near_dup_threshold)
    stats = RunStats()

    # Only allocate the buffer when something will consume it — otherwise every
    # run holds the day's article bodies in memory for no reason.
    liberty = liberty_tagger if liberty_tagger is not None else _build_liberty(cfg)
    liberty_texts: dict[str, str] | None = {} if liberty is not None else None

    sources = list(registry.ingestable(("rss",)))
    for position, source in enumerate(sources, start=1):
        before = (stats.stored, stats.fetched, stats.errors)
        started = time.monotonic()
        failed = _ingest_source(
            source, store, cfg, scorer, embedder, robots, limiter, index, stats,
            transformer, liberty_texts,
        )
        if progress is not None:
            progress(SourceProgress(
                index=position, total=len(sources), source=source,
                stored=stats.stored - before[0], fetched=stats.fetched - before[1],
                errors=stats.errors - before[2],
                seconds=time.monotonic() - started, failed=failed,
            ))

    if liberty is not None and liberty_texts:
        if progress is not None:
            _announce_liberty(liberty, liberty_texts, progress)
        _score_liberty(store, liberty, liberty_texts, stats)
    return stats


def _announce_liberty(tagger, texts: dict[str, str], progress) -> None:
    """Warn before the pipeline's one genuinely long silent wait.

    Above ``BATCH_MIN_ITEMS`` the tagger submits a Batch API job and polls it,
    which is half price and appropriate for an overnight run but can sit quiet
    for minutes after ingestion has visibly finished. Saying so is the
    difference between "still working" and "hung".
    """
    from scoring.liberty import BATCH_MIN_ITEMS

    batched = getattr(tagger, "use_batch", False) and len(texts) >= BATCH_MIN_ITEMS
    progress(f"\nLiberty tagging {len(texts)} document(s) via {tagger.name}…")
    if batched:
        progress("  Submitted as a Batch API job (half price). This polls every "
                 "20s and usually lands in a few minutes; it can take longer. "
                 "Nothing is wrong if this line sits alone for a while.")


def _score_liberty(store: Datastore, tagger, texts: dict[str, str], stats: RunStats) -> None:
    """Tag this run's documents on liberty and persist the scores.

    Runs after the source loop rather than per document so the whole day goes
    out as one Batch API submission. It has to happen before ``run`` returns:
    the texts live only in ``texts``, and nothing persists them.
    """
    try:
        scores = tagger.score_many(texts)
    except Exception as exc:
        logger.warning("liberty scoring failed for the whole run (%s: %s)",
                       type(exc).__name__, exc)
        return
    for doc_id, score in scores.items():
        store.upsert_scores(
            document_id=doc_id, scorer=tagger.name, foundations={},
            sentiment=0.0, moral_word_ratio=0.0, matched_words=0,
            liberty=score.presence,
        )
    store.set_meta("liberty_scorer", tagger.name)
    stats.liberty_scored = len(scores)
    if len(scores) < len(texts):
        logger.info("liberty: scored %d of %d documents this run",
                    len(scores), len(texts))


def backfill(
    store: Datastore,
    registry: Registry | None = None,
    config: PipelineConfig | None = None,
    scorer: DictionaryScorer | None = None,
    embedder: Embedder | None = None,
    gdelt: GdeltClient | None = None,
    days: int = 14,
    max_per_source: int = 250,
    extract_bodies: bool = False,
    transformer: object | None = None,
) -> RunStats:
    """Backfill weeks of coverage per source from GDELT (title-based by default).

    For each outlet with a resolvable domain, pull up to ``max_per_source``
    articles over the trailing ``days`` and run them through the same
    score→dedup→embed→store path. Title-only unless ``extract_bodies`` is set
    (fetching bodies for a big historical set is slow); titles are what the
    clustering/blindspot engine needs, and the point of backfill is that volume.

    The transformer tagger is **not** auto-run here: this is a bulk historical,
    title-only job where five RoBERTas per headline would be very slow for little
    gain (bands are body-scored from feed ``run``). Pass ``transformer`` to opt in.
    """
    from .gdelt import GdeltClient

    registry = registry or load_registry()
    cfg = config or PipelineConfig()
    if scorer is None:
        lexicon, lexicon_name = build_lexicon(cfg.lexicon_path)
        scorer = DictionaryScorer(lexicon, assignment=cfg.assignment,
                                  splitter=_build_splitter(cfg))
    else:
        lexicon_name = "injected"
    if embedder is None:
        embedder = HashingEmbedder()
    client = gdelt or GdeltClient()
    store.set_meta("lexicon", lexicon_name)
    store.set_meta("embedder", getattr(embedder, "name", type(embedder).__name__))
    robots = (
        RobotsCache(cfg.user_agent, cfg.timeout)
        if (extract_bodies and cfg.respect_robots) else None
    )
    limiter = RateLimiter(cfg.per_host_rpm)
    index = _seed_index(store, cfg.near_dup_threshold)
    stats = RunStats()
    # One GDELT query per unique (diet, domain) — several sources share a domain.
    seen_domains: set[tuple[str, str]] = set()

    for source in registry.backfillable():
        key = (source.diet_id, source.domain or "")
        if key in seen_domains:
            continue
        seen_domains.add(key)
        try:
            articles = client.search_domain(
                source.domain, timespan=f"{days}d", max_records=max_per_source
            )
        except Exception:
            stats.errors += 1
            continue
        for art in articles:
            stats.fetched += 1
            if extract_bodies:
                body = extract_article(
                    art.url, user_agent=cfg.user_agent, timeout=cfg.timeout,
                    robots=robots, rate_limiter=limiter,
                )
                text = f"{art.title}\n\n{body}".strip() if body else art.title
                min_words = cfg.min_words
            else:
                text = art.title
                min_words = 3  # title-only: don't skip on the short-doc guard
            _ingest_one(
                store, source, scorer, embedder, index, stats,
                title=art.title, link=art.url, published_utc=art.published_utc,
                text=text, cluster_text=art.title, min_words=min_words,
                transformer=transformer,
            )
    return stats


def _build_splitter(cfg: PipelineConfig):
    """The fairness equality/proportionality partitioner, or ``None`` if disabled."""
    if not cfg.split_fairness:
        return None
    from scoring.fairness_split import FairnessSplitter

    return FairnessSplitter(min_evidence=cfg.fairness_min_evidence)


def _build_liberty(cfg: PipelineConfig):
    """The Claude liberty tagger, or ``None`` when it can't or shouldn't run."""
    from scoring.liberty import DEFAULT_EFFORT, DEFAULT_MODEL, build_tagger

    return build_tagger(
        model=cfg.liberty_model or DEFAULT_MODEL,
        effort=cfg.liberty_effort or DEFAULT_EFFORT,
        use_batch=cfg.liberty_batch,
        enabled=cfg.liberty_enabled,
    )


def _build_transformer(cfg: PipelineConfig):
    """Construct the Mformer transformer scorer, or return ``None`` (with a
    single warning) when it's disabled or its heavy deps aren't installed.

    Keeping the failure non-fatal preserves the zero-dependency ``ingestion run``
    path: without ``parallax[scoring]`` you still get dictionary scores, just no
    confidence band. Install the extra to have every article transformer-scored.
    """
    if not cfg.transformer_enabled:
        return None
    from scoring.transformer import TransformerScorer, resolve_model_prefix

    try:
        return TransformerScorer(
            model_prefix=resolve_model_prefix(cfg.transformer_model),
            max_length=cfg.transformer_max_length,
            revision=cfg.transformer_revision,
        )
    except Exception as exc:  # missing torch/transformers, offline, etc.
        logger.warning(
            "transformer tagger unavailable (%s: %s) — scoring dictionary-only, "
            "no confidence band. Install parallax[scoring] to enable it.",
            type(exc).__name__, exc,
        )
        return None


def _store_transformer_meta(store: Datastore, transformer) -> None:
    """Record which scorer produced the transformer rows, so the exporter can
    pair it against the dictionary for the confidence band (and clear stale
    provenance when the transformer is off)."""
    store.set_meta("transformer_scorer", transformer.name if transformer is not None else "")


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
    embedder: Embedder,
    robots: RobotsCache | None,
    limiter: RateLimiter,
    index: NearDuplicateIndex,
    stats: RunStats,
    transformer: object | None = None,
    liberty_texts: dict[str, str] | None = None,
) -> bool:
    """Ingest one source. Returns True if the feed itself could not be read.

    A dead feed and a feed of unreachable articles both raise the error count,
    but only the first means the source is gone — worth distinguishing in the
    progress line, since it is the one that needs a fix in ``sources.yaml``.
    """
    try:
        items = parse_feed(source.url, cfg.user_agent)  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning("feed unreadable for %s (%s: %s)", source.id, type(exc).__name__, exc)
        stats.errors += 1
        return True

    for item in items[: cfg.max_items_per_feed]:
        stats.fetched += 1
        text = _document_text(item, cfg, robots, limiter)
        if text is None:
            stats.errors += 1
            continue
        _ingest_one(
            store, source, scorer, embedder, index, stats,
            title=item.title, link=item.link, published_utc=item.published_utc,
            text=text, cluster_text=_cluster_text(item, text), min_words=cfg.min_words,
            transformer=transformer, liberty_texts=liberty_texts,
        )
    return False


def _ingest_one(
    store: Datastore,
    source: Source,
    scorer: DictionaryScorer,
    embedder: Embedder,
    index: NearDuplicateIndex,
    stats: RunStats,
    *,
    title: str,
    link: str | None,
    published_utc: str | None,
    text: str,
    cluster_text: str,
    min_words: int,
    transformer: object | None = None,
    liberty_texts: dict[str, str] | None = None,
) -> None:
    """Score, dedup, embed and store one document. Shared by feed ingest and
    GDELT backfill. Identity is the canonical article URL when present, so the
    same story reached via a feed (often utm-tagged) and via GDELT (clean url)
    collapses to one document.

    When a ``transformer`` scorer is supplied, every stored document also gets a
    second ``foundation_scores`` row (its ``P(present)`` per foundation), so the
    dashboard can show the dictionary-vs-transformer confidence band."""
    doc_id = document_id(link, text)
    if store.has_document(doc_id):
        stats.exact_duplicates += 1
        return

    score: DocumentScore = scorer.score(text)
    if score.word_count < min_words:
        stats.skipped_short += 1
        return

    mh = minhash_signature(text, k=5)
    dup_of = index.find_duplicate(mh)
    is_dup = dup_of is not None

    store.upsert_document(
        doc_id=doc_id, diet_id=source.diet_id, source_id=source.id,
        stratum_id=source.stratum_id, url=link, title=title,
        published_utc=published_utc, fetched_utc=_now_iso(),
        word_count=score.word_count, minhash=signature_list(mh),
        weight=source.diet_weight, is_duplicate=is_dup, duplicate_of=dup_of,
    )
    store.upsert_scores(
        document_id=doc_id, scorer=score.scorer, foundations=score.foundations,
        sentiment=score.sentiment, moral_word_ratio=score.moral_word_ratio,
        matched_words=score.matched_words, liberty=score.liberty,
        equality=score.equality, proportionality=score.proportionality,
    )
    if transformer is not None:
        # A single flaky document (encoding, length edge, transient model error)
        # must not abort the batch or leave the doc half-ingested — the dictionary
        # row and embedding still land; that doc simply gets no confidence band.
        try:
            probs = transformer.score(text)
            store.upsert_scores(
                document_id=doc_id, scorer=transformer.name, foundations=probs,
                sentiment=0.0, moral_word_ratio=0.0, matched_words=0,
            )
        except Exception as exc:
            logger.warning("transformer scoring failed for %s (%s: %s) — dictionary-only",
                           doc_id, type(exc).__name__, exc)
    store.upsert_embedding(
        document_id=doc_id,
        vector=embedder.embed(cluster_text),
        embedder=getattr(embedder, "name", type(embedder).__name__),
    )

    if is_dup:
        stats.near_duplicates += 1
    else:
        index.add(doc_id, mh)
        stats.stored += 1
        if liberty_texts is not None:
            # Buffered, not persisted: raw text is discarded at the end of this
            # run (§0), so the liberty tagger has to see it before then.
            liberty_texts[doc_id] = text
        stats.per_diet[source.diet_id] = stats.per_diet.get(source.diet_id, 0) + 1


def diet_profiles(
    store: Datastore,
    scorer_name: str = "dictionary",
    since: str | None = None,
    until: str | None = None,
) -> dict[str, dict[str, float]]:
    """Build a normalized foundation composition per diet from stored scores.

    ``since``/``until`` (``YYYY-MM-DD``, half-open) restrict the profile to
    documents dated in that window, which is how ``compare.history`` builds a
    trailing-window series. Unbounded — the default — profiles the whole corpus,
    which is what the dashboard's headline numbers report.
    """
    from scoring.foundations import CLASSIC_FOUNDATIONS

    profiles: dict[str, dict[str, float]] = {}
    for diet_id in store.diet_ids():
        rows = store.scores_for_diet(diet_id, scorer_name, since=since, until=until)
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
