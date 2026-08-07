"""Podcast and talk-radio ingestion: feed -> enclosure -> transcript -> document.

This is the step that makes the modeled diet honest. Talk radio and podcasts are
most of what that diet actually is, and a text-only pipeline scores none of it —
so every number the tool produced about that diet was really a number about the
minority of it that publishes articles.

The shape is the same as article ingestion, with one difference that drives the
whole design: an article is cheap to fetch and re-fetch, so the pipeline dedups
*after* the fact on document id. An episode costs a download and minutes of
transcription, so it has to be recognised *before* any of that, from the feed
alone. That is what ``podcast_episodes`` is for — the guid ledger is checked
first, and every outcome is written back to it, failures included. An episode
that 404s or exceeds the size cap is recorded as failed rather than retried
every morning until someone notices.

Everything downstream is shared with articles: the same dictionary and
transformer scoring, the same MinHash near-duplicate index (a syndicated
segment can appear on two shows), the same embedding, the same store, via
``pipeline._ingest_one``.

**Audio is transient.** It is streamed to a temporary file, transcribed, and
deleted in a ``finally`` — including when transcription raises. Transcripts are
held in memory only, exactly as article bodies are (§0). Nothing but the derived
metrics survives the run.

Requires ``parallax[media]`` for faster-whisper. Without it this step logs the
reason once and does nothing, which is how the transformer tagger and the
liberty tagger already behave: a missing optional dependency degrades the run
rather than failing it.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import feedparser

logger = logging.getLogger(__name__)

# Audio is big and the disk is not ours to fill. 500 MB is a ~4-hour show at a
# typical spoken-word bitrate; past that it is a video file or a mistake, and
# either way it is not worth an hour of CPU to find out.
DEFAULT_MAX_BYTES = 500 * 1024 * 1024

# A wall-clock ceiling for the whole step, because this is the one part of the
# pipeline that can run for hours. The daily job has a brief to deliver in the
# morning; an unbounded transcription queue is how it silently misses it. The
# budget is checked between episodes, never mid-episode — a half-transcribed
# episode is worth nothing, so stopping inside one only wastes what it spent.
DEFAULT_TIME_BUDGET_SECONDS = 3 * 60 * 60

# Per source, per run. Most shows publish daily; a backlog of hundreds on first
# run would otherwise all arrive at once.
DEFAULT_MAX_EPISODES = 3

# Only recent episodes. A first run against a feed with 3000 back-episodes
# should ingest this week, not the archive.
DEFAULT_SINCE_DAYS = 7

_AUDIO_TYPES = ("audio/", "video/")   # some feeds ship video enclosures


@dataclass(frozen=True)
class Episode:
    guid: str
    title: str
    audio_url: str
    published_utc: str | None = None
    link: str | None = None            # episode page, for the dashboard to link
    duration_seconds: int | None = None


@dataclass
class PodcastStats:
    considered: int = 0
    already_seen: int = 0
    transcribed: int = 0
    stored: int = 0
    failed: int = 0
    skipped: int = 0
    seconds_spent: float = 0.0
    per_source: dict[str, int] = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"{self.stored} stored from {self.transcribed} transcribed "
            f"({self.already_seen} already seen, {self.skipped} skipped, "
            f"{self.failed} failed) in {self.seconds_spent / 60:.1f} min"
        )


# -- feed parsing -----------------------------------------------------------


def _duration_seconds(entry) -> int | None:
    """``itunes:duration``, which is seconds, or MM:SS, or HH:MM:SS."""
    raw = (entry.get("itunes_duration") or "").strip()
    if not raw:
        return None
    try:
        parts = [int(p) for p in raw.split(":")]
    except ValueError:
        return None
    seconds = 0
    for part in parts:            # works for [s], [m, s] and [h, m, s]
        seconds = seconds * 60 + part
    return seconds


def _audio_enclosure(entry) -> str | None:
    for enc in entry.get("enclosures", []) or []:
        href = enc.get("href") or enc.get("url")
        if not href:
            continue
        # Type is advisory and some feeds omit it; a bare href on an enclosure
        # in a podcast feed is the audio far more often than it is anything
        # else, so an absent type is accepted rather than dropped.
        etype = (enc.get("type") or "").lower()
        if not etype or etype.startswith(_AUDIO_TYPES):
            return href
    return None


def parse_podcast_feed(url: str, user_agent: str) -> list[Episode]:
    """Episodes with playable audio, newest first as the feed ordered them.

    Distinct from ``extract.parse_feed`` because the two want different things
    from the same XML: an article feed needs the link, a podcast feed needs the
    enclosure and the guid. Entries without audio are dropped here rather than
    downstream — a show that also publishes text posts to the same feed should
    not book a transcription slot for them.
    """
    parsed = feedparser.parse(url, agent=user_agent)
    episodes: list[Episode] = []
    for entry in parsed.entries:
        audio_url = _audio_enclosure(entry)
        if not audio_url:
            continue
        # `id` is the feed's guid. Falling back to the audio url keeps a feed
        # without guids usable, and it is stable for as long as the file is.
        guid = (entry.get("id") or "").strip() or audio_url
        episodes.append(
            Episode(
                guid=guid,
                title=(entry.get("title") or "").strip(),
                audio_url=audio_url,
                published_utc=_published(entry),
                link=entry.get("link"),
                duration_seconds=_duration_seconds(entry),
            )
        )
    return episodes


def _published(entry) -> str | None:
    pp = entry.get("published_parsed") or entry.get("updated_parsed")
    if not pp:
        return None
    try:
        return datetime(*pp[:6], tzinfo=UTC).isoformat()
    except (TypeError, ValueError):
        return None


def recent(episodes: list[Episode], since_days: int) -> list[Episode]:
    """Episodes published within the window, plus any with no date at all.

    An undated episode is kept rather than dropped: feeds that omit dates are
    rare, and silently ingesting nothing from one is worse than transcribing an
    old episode once — the ledger stops it happening twice.
    """
    if since_days <= 0:
        return episodes
    cutoff = datetime.now(UTC) - timedelta(days=since_days)
    keep = []
    for ep in episodes:
        if ep.published_utc is None:
            keep.append(ep)
            continue
        try:
            if datetime.fromisoformat(ep.published_utc) >= cutoff:
                keep.append(ep)
        except ValueError:
            keep.append(ep)
    return keep


# -- audio ------------------------------------------------------------------


def download_audio(
    url: str,
    user_agent: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: int = 60,
) -> str:
    """Stream an enclosure to a temp file and return its path.

    Streamed rather than read into memory, and capped: an episode is tens of
    megabytes and the caller may be a laptop. Raises on anything wrong, and
    leaves no file behind when it does — the caller's ``finally`` handles the
    success path.
    """
    fd, path = tempfile.mkstemp(suffix=".audio", prefix="parallax-")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    written = 0
    try:
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(request, timeout=timeout) as resp:
            # Checked against the header first where there is one, so an
            # oversized file costs one round trip rather than a full download.
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise ValueError(
                    f"enclosure declares {int(declared) // (1024 * 1024)} MB, over the "
                    f"{max_bytes // (1024 * 1024)} MB cap — not downloading it"
                )
            while chunk := resp.read(1024 * 256):
                written += len(chunk)
                # And again while reading: Content-Length is optional, and on a
                # chunked response it is absent exactly when the body is large.
                if written > max_bytes:
                    raise ValueError(
                        f"enclosure exceeds the {max_bytes // (1024 * 1024)} MB cap "
                        "— not transcribing it"
                    )
                fh.write(chunk)
    except BaseException:
        _unlink(path)
        raise
    return path


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError as exc:          # already gone, or a permissions problem
        logger.debug("could not remove %s: %s", path, exc)


# -- transcription ----------------------------------------------------------


class WhisperTranscriber:
    """faster-whisper, loaded once and reused for every episode in the run.

    The model is several hundred megabytes and takes real time to load, which
    is worth paying once per run and not once per episode — the same reason the
    Mformer tagger is built once in the daily runner.
    """

    def __init__(self, model, name: str, vad_filter: bool = True) -> None:
        self._model = model
        self.name = name
        self.vad_filter = vad_filter

    def transcribe(self, path: str) -> str:
        # `vad_filter` drops long silences before they reach the model. Whisper
        # hallucinates fluent text over silence — it is trained to produce
        # speech — and a hallucinated paragraph is not merely noise here: it
        # gets scored, and it scores as whatever the model happened to invent.
        segments, _info = self._model.transcribe(path, vad_filter=self.vad_filter)
        return " ".join(segment.text.strip() for segment in segments).strip()


def build_transcriber(
    model_size: str = "medium",
    compute_type: str = "int8",
    device: str = "auto",
    vad_filter: bool = True,
) -> tuple[WhisperTranscriber | None, str]:
    """``(transcriber, reason)`` — the reason empty on success.

    A return value rather than an exception, and the same contract as
    ``scoring.claude_client.build_client``: a missing optional dependency should
    cost the run its audio, not its articles.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None, (
            "faster-whisper is not installed. It lives in the `media` extra — "
            'install it with `pip install -e ".[media]"`.'
        )
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:
        # A bad model name, a compute type this machine cannot do, a failed
        # download. All of them are setup problems and none should abort a run
        # that has already ingested and scored the day's articles.
        return None, f"could not load the {model_size} model ({type(exc).__name__}: {exc})"
    return WhisperTranscriber(model, f"faster-whisper/{model_size}", vad_filter), ""


# -- the run ----------------------------------------------------------------


@dataclass
class PodcastConfig:
    """Budgets. Every one of these exists because this step can run for hours."""

    max_episodes_per_source: int = DEFAULT_MAX_EPISODES
    since_days: int = DEFAULT_SINCE_DAYS
    time_budget_seconds: int = DEFAULT_TIME_BUDGET_SECONDS
    max_bytes: int = DEFAULT_MAX_BYTES
    whisper_model: str = "medium"
    compute_type: str = "int8"
    device: str = "auto"
    vad_filter: bool = True

    @classmethod
    def from_settings(cls, settings: dict) -> PodcastConfig:
        audio = ((settings.get("ingestion", {}) or {}).get("audio", {}) or {})
        return cls(
            max_episodes_per_source=int(audio.get("max_episodes_per_source",
                                                  DEFAULT_MAX_EPISODES)),
            since_days=int(audio.get("since_days", DEFAULT_SINCE_DAYS)),
            time_budget_seconds=int(audio.get("time_budget_seconds",
                                              DEFAULT_TIME_BUDGET_SECONDS)),
            max_bytes=int(audio.get("max_megabytes", DEFAULT_MAX_BYTES // (1024 * 1024)))
            * 1024 * 1024,
            whisper_model=audio.get("whisper_model", "medium"),
            compute_type=audio.get("compute_type", "int8"),
            device=audio.get("device", "auto"),
            vad_filter=bool(audio.get("filter_silence", True)),
        )


def run(
    store,
    registry=None,
    config=None,
    podcast_config: PodcastConfig | None = None,
    transcriber=None,
    scorer=None,
    embedder=None,
    transformer: object | None = None,
    progress=None,
) -> PodcastStats:
    """Transcribe new episodes from every ``podcast_rss`` source into ``store``.

    Deliberately a separate entry point from ``pipeline.run`` rather than
    another branch inside it. The two differ in every operational way that
    matters — minutes per item against milliseconds, a hard time budget, an
    optional heavyweight model, a ledger that has to be consulted before any
    work — and folding that into the article loop would put an hours-long tail
    on the step that has to finish before the morning brief.
    """
    from cluster.embed import HashingEmbedder
    from scoring.dictionary import DictionaryScorer
    from scoring.lexicon import build_lexicon

    from .config import load_registry
    from .pipeline import PipelineConfig, _build_transformer, _seed_index

    registry = registry or load_registry()
    cfg = config or PipelineConfig()
    pcfg = podcast_config or PodcastConfig()
    stats = PodcastStats()

    sources = [s for s in registry.all_sources()
               if s.ingest_type == "podcast_rss" and s.url]
    if not sources:
        _note(progress, "no podcast sources with a URL — nothing to do")
        return stats

    if transcriber is None:
        transcriber, reason = build_transcriber(
            pcfg.whisper_model, pcfg.compute_type, pcfg.device, pcfg.vad_filter)
        if transcriber is None:
            # The one place this step gives up entirely, and it says why at
            # WARNING: an absent transcript is indistinguishable from a diet
            # that simply did not talk about anything.
            logger.warning("podcast ingestion skipped: %s", reason)
            _note(progress, f"skipped: {reason}")
            return stats

    if scorer is None:
        lexicon, _name = build_lexicon(cfg.lexicon_path)
        scorer = DictionaryScorer(lexicon, assignment=cfg.assignment)
    embedder = embedder or HashingEmbedder()
    if transformer is None:
        transformer = _build_transformer(cfg)
    index = _seed_index(store, cfg.near_dup_threshold)

    deadline = time.monotonic() + pcfg.time_budget_seconds
    started_all = time.monotonic()
    for source in sources:
        if time.monotonic() >= deadline:
            # Said out loud, never silent: a truncated run that reports success
            # is how a diet quietly loses half its audio.
            _note(progress, f"time budget spent — {source.id} and later sources "
                            "were not reached this run")
            logger.warning("podcast time budget (%ds) spent before %s",
                           pcfg.time_budget_seconds, source.id)
            break
        _ingest_source(source, store, cfg, pcfg, transcriber, scorer, embedder,
                       transformer, index, stats, deadline, progress)
    stats.seconds_spent = time.monotonic() - started_all
    return stats


def _note(progress, message: str) -> None:
    if progress is not None:
        progress(message)


def _ingest_source(
    source, store, cfg, pcfg, transcriber, scorer, embedder, transformer,
    index, stats: PodcastStats, deadline: float, progress,
) -> None:
    try:
        episodes = parse_podcast_feed(source.url, cfg.user_agent)
    except Exception as exc:
        logger.warning("podcast feed unreadable for %s (%s: %s)",
                       source.id, type(exc).__name__, exc)
        stats.failed += 1
        return

    seen = store.seen_episode_keys(source.id)
    # Either key is enough to recognise an episode; see `seen_episode_keys`.
    fresh = [ep for ep in recent(episodes, pcfg.since_days)
             if ep.guid not in seen and ep.audio_url not in seen]
    stats.already_seen += len(episodes) - len(fresh)
    todo = fresh[: pcfg.max_episodes_per_source]

    for episode in todo:
        if time.monotonic() >= deadline:
            _note(progress, f"time budget spent inside {source.id} — "
                            f"{len(todo)} of its episodes were queued")
            return
        stats.considered += 1
        _ingest_episode(episode, source, store, cfg, pcfg, transcriber, scorer,
                        embedder, transformer, index, stats, progress)


def _ingest_episode(
    episode: Episode, source, store, cfg, pcfg, transcriber, scorer, embedder,
    transformer, index, stats: PodcastStats, progress,
) -> None:
    """Download, transcribe, score, store — and record the outcome either way.

    Every exit from this function writes to the ledger. An episode that leaves
    no row is an episode the next run pays for again.
    """
    from .dedup import document_id
    from .pipeline import _ingest_one

    record = {
        "guid": episode.guid, "source_id": source.id, "title": episode.title,
        "published_utc": episode.published_utc,
        "duration_seconds": episode.duration_seconds,
        "audio_url": episode.audio_url,
    }
    path = None
    try:
        path = download_audio(episode.audio_url, cfg.user_agent, pcfg.max_bytes,
                              cfg.timeout)
        _note(progress, f"transcribing {source.id}: {episode.title[:60]}")
        text = transcriber.transcribe(path)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        logger.warning("episode failed for %s (%s)", source.id, detail)
        stats.failed += 1
        store.record_episode(status="failed", detail=detail[:500], **record)
        return
    finally:
        # Audio never outlives the episode that needed it, transcription
        # failures included (§0: raw material is a processing artifact).
        if path is not None:
            _unlink(path)

    stats.transcribed += 1
    if not text.strip():
        # A silent or music-only episode. Not a failure worth retrying —
        # transcribing it again produces the same nothing.
        stats.skipped += 1
        store.record_episode(status="skipped", detail="empty transcript", **record)
        return

    # The episode page, not the audio file: the dashboard links this, and a
    # citation that starts playing a 90-minute mp3 is not a citation.
    link = episode.link or episode.audio_url
    counters = _IngestCounters()
    _ingest_one(
        store, source, scorer, embedder, index, counters,
        title=episode.title, link=link, published_utc=episode.published_utc,
        text=text, cluster_text=f"{episode.title}\n\n{text[:2000]}",
        min_words=cfg.min_words, transformer=transformer, liberty_texts=None,
    )

    if counters.stored:
        stats.stored += 1
        stats.per_source[source.id] = stats.per_source.get(source.id, 0) + 1
        store.record_episode(status="transcribed", document_id=document_id(link, text),
                             **record)
        return
    # Transcribed but not stored: too short to score, or the same segment
    # already arrived from another show. Recorded as handled either way — the
    # transcript would come out identical next time, and the point of the
    # ledger is to not pay for that twice.
    stats.skipped += 1
    store.record_episode(status="skipped", detail=counters.reason(), **record)


class _IngestCounters:
    """The counters ``_ingest_one`` increments, and why each one fired.

    ``_ingest_one`` is shared with article ingestion and speaks ``RunStats``.
    Rather than widen that for one caller, this stands in for it and lets the
    episode path read back what happened — which it needs, because the ledger
    entry differs between "stored" and "dropped as a duplicate"."""

    def __init__(self) -> None:
        self.exact_duplicates = 0
        self.skipped_short = 0
        self.near_duplicates = 0
        self.stored = 0
        self.per_diet: dict[str, int] = {}

    def reason(self) -> str:
        if self.skipped_short:
            return "transcript too short to score"
        if self.near_duplicates:
            return "near-duplicate of an already-stored document"
        if self.exact_duplicates:
            return "already stored under the same document id"
        return "not stored"
