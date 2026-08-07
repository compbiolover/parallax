"""The daily snapshot: one call that refreshes everything the dashboard reads.

Eight steps, in dependency order:

1. **ingest**    — fetch every RSS source, extract, dedup, score (dictionary +
                   transformer), embed.
2. **backfill**  — pull weeks of per-outlet history from GDELT so clusters rest
                   on real volume. Slow and rate-limited; on by default because
                   blindspots are only trustworthy with it.
3. **podcasts**  — podcast/talk-radio enclosures -> faster-whisper -> transcripts,
                   scored like articles. Outside the defaults: it needs
                   parallax[media] and runs in hours rather than minutes.
4. **cluster**   — embeddings -> clusters -> coverage-asymmetry blindspots.
5. **summarize** — per-diet + cross-diet summaries (Claude, or the deterministic
                   fallback when no API key is set).
6. **snapshot**  — record today's compositions and divergence so the run leaves
                   a dated point behind instead of only overwriting yesterday.
7. **export**    — write the dashboard payload.
8. **digest**    — render that payload into an email and send it. The only step
                   outside the defaults: it needs SMTP credentials, so it is
                   opted into rather than failing every morning until set up.

Two properties matter for something meant to run unattended every morning:

**Steps are isolated.** A failure in one step is recorded and the run continues.
A GDELT outage or a missing API key should still leave you with a refreshed
dashboard built from whatever data did land, not a half-updated datastore and a
stack trace. The report says exactly what failed, and the CLI exits non-zero so
cron notices.

**The transformer is loaded once.** Mformer is five RoBERTa models; loading it
per step would dominate the runtime. It is built once here and handed to the
steps that use it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cluster.themes import UNSET as THEME_EFFORT_UNSET
from compare.history import DEFAULT_SERIES_LIMIT, DEFAULT_WINDOW_DAYS
from ingestion.datastore import Datastore

# Step names, in execution order. ``snapshot`` runs before ``export`` so the
# payload carries today's point rather than lagging a day behind.
STEPS = ("ingest", "backfill", "podcasts", "cluster", "summarize", "snapshot",
         "export", "digest")

# What a run does unless told otherwise. ``digest`` is the one step that is not
# in here: it needs SMTP credentials nobody has on a first run, and a step that
# fails every morning until configured trains you to ignore the report that is
# supposed to tell you when something actually broke. Settings opt it in;
# ``--only digest`` still runs it on demand.
# ``podcasts`` is out for the same reason as ``digest``, from the other
# direction: it needs parallax[media] and hours of CPU, so a default run would
# either no-op with a warning every morning or quietly turn a five-minute job
# into an overnight one. Opt in with `podcasts: true` under `daily:` in
# settings, or run it on demand with `--only podcasts`.
DEFAULT_STEPS = tuple(s for s in STEPS if s not in ("digest", "podcasts"))


@dataclass
class DailyConfig:
    """What the daily run does. Defaults are the full snapshot."""

    db: str | None = None                 # None -> from settings
    settings_path: str | None = None
    out: str | None = None                # dashboard payload path; None -> default
    steps: tuple[str, ...] = DEFAULT_STEPS

    # ingest
    max_items_per_feed: int | None = None
    lexicon_path: str | None = None
    transformer: bool | None = None       # None -> whatever settings say

    # backfill (GDELT)
    backfill_days: int = 14
    backfill_max_per_source: int = 250
    backfill_extract_bodies: bool = False
    backfill_transformer: bool = False    # title-only bulk; bands come from feeds

    # cluster
    min_cluster_size: int = 2
    dominance: float = 0.75
    min_blindspot_size: int = 2
    claude_themes: bool = True            # name blindspot themes with Claude
    theme_model: str | None = None        # None -> cluster.themes default
    # UNSET -> cluster.themes default; None -> send no effort at all
    theme_effort: object = THEME_EFFORT_UNSET

    # summarize
    model: str | None = None              # None -> summarizer default
    summary_effort: str | None = None     # None -> summarizer default

    # snapshot
    window_days: int = DEFAULT_WINDOW_DAYS
    history_limit: int = DEFAULT_SERIES_LIMIT

    # podcasts (parallax[media]); None -> defaults from ingestion.podcast
    podcast_config: object = None

    # digest
    own_diet: str | None = None           # whose blindspots lead the email

    # Built once per run by `_payload`, shared by export and digest so the two
    # cannot describe different dicts. Not configuration — excluded from repr
    # and comparison so it stays out of the report and out of test assertions.
    _payload_cache: dict | None = field(default=None, repr=False, compare=False)

    def enabled(self, step: str) -> bool:
        return step in self.steps

    @classmethod
    def from_settings(cls, settings: dict) -> DailyConfig:
        """Defaults from the ``daily:`` block of settings.yaml. CLI flags layer
        on top of this (see ``daily.__main__``)."""
        daily = (settings.get("daily", {}) or {})
        bf = (daily.get("backfill", {}) or {})
        snap = (daily.get("snapshot", {}) or {})
        dig = (settings.get("digest", {}) or {})
        themes = ((settings.get("cluster", {}) or {}).get("themes", {}) or {})
        cfg = cls(
            backfill_days=int(bf.get("days", 14)),
            backfill_max_per_source=int(bf.get("max_per_source", 250)),
            backfill_extract_bodies=bool(bf.get("extract_bodies", False)),
            backfill_transformer=bool(bf.get("transformer", False)),
            window_days=int(snap.get("window_days", DEFAULT_WINDOW_DAYS)),
            history_limit=int(snap.get("history_limit", DEFAULT_SERIES_LIMIT)),
            claude_themes=bool(themes.get("claude", True)),
            theme_model=themes.get("model"),
            # `.get` with a default keeps a stored None: `effort: ~` means
            # "send no effort", which is not the same instruction as leaving
            # the key out. Plain `.get("effort")` collapses the two.
            theme_effort=themes.get("effort", THEME_EFFORT_UNSET),
            # `summarize.model` was documented in settings.example.yaml and read
            # by nothing: the daily run always took the summarizer's default.
            model=(settings.get("summarize", {}) or {}).get("model"),
            summary_effort=(settings.get("summarize", {}) or {}).get("effort"),
            own_diet=dig.get("own_diet"),
        )
        from ingestion.podcast import PodcastConfig

        cfg.podcast_config = PodcastConfig.from_settings(settings)
        if daily.get("podcasts", False):
            cfg.steps = tuple(s for s in STEPS if s in {*cfg.steps, "podcasts"})
        if not bf.get("enabled", True):
            cfg.steps = tuple(s for s in cfg.steps if s != "backfill")
        if dig.get("enabled", False):
            cfg.steps = tuple(s for s in STEPS if s in {*cfg.steps, "digest"})
        return cfg


@dataclass
class StepResult:
    name: str
    status: str                # "ok" | "failed" | "skipped"
    seconds: float = 0.0
    detail: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "skipped")


@dataclass
class DailyReport:
    started_utc: str
    steps: list[StepResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def failed(self) -> list[StepResult]:
        return [s for s in self.steps if s.status == "failed"]

    def get(self, name: str) -> StepResult | None:
        return next((s for s in self.steps if s.name == name), None)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_db(cfg: DailyConfig, settings: dict) -> str:
    if cfg.db:
        return cfg.db
    return (settings.get("datastore", {}) or {}).get("path", "data/parallax.sqlite")


# -- individual steps -------------------------------------------------------
# Each returns a human-readable detail string. Exceptions propagate to the
# isolation wrapper in run_daily, which records them as a failed step.


def _step_ingest(store, cfg: DailyConfig, pcfg, registry, embedder, transformer,
                 progress=None) -> str:
    from ingestion.pipeline import run

    stats = run(store, registry, pcfg, embedder=embedder, transformer=transformer,
                progress=progress)
    return (
        f"{stats.stored} stored, {stats.exact_duplicates} exact-dup, "
        f"{stats.near_duplicates} near-dup, {stats.errors} errors"
    )


def _step_backfill(store, cfg: DailyConfig, pcfg, registry, embedder, transformer) -> str:
    from ingestion.pipeline import backfill

    stats = backfill(
        store, registry, pcfg, embedder=embedder,
        days=cfg.backfill_days, max_per_source=cfg.backfill_max_per_source,
        extract_bodies=cfg.backfill_extract_bodies,
        transformer=transformer if cfg.backfill_transformer else None,
    )
    return (
        f"{stats.stored} stored from {cfg.backfill_days}d window, "
        f"{stats.exact_duplicates} already known, {stats.errors} source errors"
    )


def _step_podcasts(store, cfg: DailyConfig, pcfg, registry, embedder, transformer,
                   progress=None) -> str:
    """Transcribe new episodes. The one step measured in hours, not minutes.

    It carries its own budgets rather than borrowing the run's, because the
    thing being bounded is different: every other step is bounded by how much
    there is to fetch, and this one by how long you are willing to let the
    machine think.
    """
    from ingestion.podcast import run as run_podcasts

    stats = run_podcasts(
        store, registry, pcfg, cfg.podcast_config,
        embedder=embedder, transformer=transformer, progress=progress,
    )
    return stats.line()


def _step_cluster(store, cfg: DailyConfig) -> str:
    from cluster.blindspot import run_clustering

    if store.embedding_count() == 0:
        return "no embeddings yet — nothing to cluster"
    outcome = run_clustering(
        store,
        min_cluster_size=cfg.min_cluster_size,
        dominance=cfg.dominance,
        min_blindspot_size=cfg.min_blindspot_size,
        theme_model=cfg.theme_model,
        theme_effort=cfg.theme_effort,
        claude_themes=cfg.claude_themes,
    )
    return (
        f"{outcome.n_docs} docs -> {outcome.n_clusters} clusters "
        f"({outcome.n_noise} noise), {len(outcome.blindspots)} blindspots "
        f"in {len(outcome.themes)} themes"
    )


def _step_summarize(store, cfg: DailyConfig) -> str:
    from summarize.summarizer import DEFAULT_EFFORT, DEFAULT_MODEL, Summarizer

    summarizer = Summarizer(
        model=cfg.model or DEFAULT_MODEL,
        effort=cfg.summary_effort or DEFAULT_EFFORT,
    )
    result = summarizer.summarize(store)
    # Test the *text*, not the dict. `{"self": "", "modeled_ce": ""}` is a
    # truthy dict of empty summaries, so this guard passed straight through an
    # empty result: the run persisted blank prose over the brief and reported
    # "2 diets" while the email showed no summary at all.
    diets = sum(1 for t in result.per_diet.values() if t.strip())
    if not diets and not result.executive.strip():
        return "no summary text produced — nothing persisted"
    summarizer.persist(store, result)
    # An empty per-diet summary is persisted rather than skipped: the row is
    # this run's answer for that diet, and leaving the previous run's text in
    # place would put yesterday's prose under today's date. The report says
    # which sections came back so a partial result is visible here rather than
    # only as a panel that quietly stopped appearing.
    detail = f"via {result.method} (model={result.model}), {diets} diets"
    if diets < len(result.per_diet):
        detail += f" of {len(result.per_diet)}"
    return detail + ("" if result.executive.strip() else ", no executive")


def _step_snapshot(store, cfg: DailyConfig) -> str:
    from compare.history import record_snapshot

    snap = record_snapshot(store, window_days=cfg.window_days)
    jsd = snap.window.jsd
    moved = f"window JSD {jsd:.3f}" if jsd is not None else "window JSD n/a (one diet)"
    return f"recorded {snap.snapshot_date} ({moved}), {store.snapshot_count()} in history"


def _payload(store, cfg: DailyConfig) -> dict:
    """Today's payload, built once and shared by export and digest.

    Both steps used to call ``build_payload`` independently, which re-ran the
    whole aggregate layer — every diet's score scan, bands, fairness, liberty,
    blindspots, and a JSON parse of up to a year of snapshots — a second time.
    Worse than the cost: they were two separately computed dicts with different
    ``generated_utc`` values, so the email could describe a payload the
    dashboard never got, including when export had failed outright. That is
    exactly the drift ``digest/render`` claims cannot happen.
    """
    from dashboard.export import build_payload

    if cfg._payload_cache is None:
        cfg._payload_cache = build_payload(store, cfg.history_limit)
    return cfg._payload_cache


def _step_export(store, cfg: DailyConfig) -> str:
    from dashboard.export import DEFAULT_OUT, write_payload_dict

    out = write_payload_dict(_payload(store, cfg), cfg.out or DEFAULT_OUT)
    return f"wrote {out}"


def _step_digest(store, cfg: DailyConfig) -> str:
    """Render the brief and mail it.

    Runs last, after export, so the email describes the same payload the
    dashboard does rather than a state one step behind.
    """
    from digest.render import build_digest
    from digest.send import send

    digest = build_digest(_payload(store, cfg), own_diet=cfg.own_diet)
    reason = send(digest)
    if reason is None:
        return f"sent — {digest.subject}"
    # A step failure, not a silent skip: the digest was switched on deliberately,
    # so not sending is the thing you need to be told about — and it says which
    # of the four failures it was, since the report is what actually gets read.
    raise RuntimeError(reason)


def run_daily(cfg: DailyConfig | None = None, store: Datastore | None = None,
              progress=None) -> DailyReport:
    """Run the daily snapshot. Never raises for a step failure — see the report.

    ``progress`` is an optional callback that receives step announcements and
    the ingest step's per-source lines. A full run is minutes long and prints
    nothing until the end without it, which makes a slow network and a hung
    process look identical.
    """
    from ingestion.config import load_registry, load_settings
    from ingestion.pipeline import PipelineConfig

    cfg = cfg or DailyConfig()
    settings = load_settings(cfg.settings_path)
    report = DailyReport(started_utc=_now_iso())
    started = time.monotonic()

    owns_store = store is None
    store = store or Datastore(_resolve_db(cfg, settings))

    try:
        pcfg = PipelineConfig.from_settings(settings)
        if cfg.max_items_per_feed is not None:
            pcfg.max_items_per_feed = cfg.max_items_per_feed
        if cfg.lexicon_path is not None:
            pcfg.lexicon_path = cfg.lexicon_path
        if cfg.transformer is not None:
            pcfg.transformer_enabled = cfg.transformer

        registry = load_registry()
        embedder = _build_embedder(settings)
        # Load Mformer once and share it — it is five RoBERTa models.
        transformer = _build_transformer(pcfg) if _needs_transformer(cfg) else None
        if transformer is None:
            # ``run()`` builds its own transformer when passed None. Tell it not to
            # bother: either we deliberately skipped it, or the build already failed
            # (and retrying would repeat a slow failure and a duplicate warning).
            pcfg.transformer_enabled = False

        step_args = {
            "ingest": lambda: _step_ingest(store, cfg, pcfg, registry, embedder,
                                           transformer, progress),
            "backfill": lambda: _step_backfill(store, cfg, pcfg, registry, embedder, transformer),
            "podcasts": lambda: _step_podcasts(store, cfg, pcfg, registry, embedder,
                                               transformer, progress),
            "cluster": lambda: _step_cluster(store, cfg),
            "summarize": lambda: _step_summarize(store, cfg),
            "snapshot": lambda: _step_snapshot(store, cfg),
            "export": lambda: _step_export(store, cfg),
            "digest": lambda: _step_digest(store, cfg),
        }
        for name in STEPS:
            report.steps.append(_run_step(name, step_args[name], cfg.enabled(name), progress))
    finally:
        if owns_store:
            store.close()

    report.seconds = time.monotonic() - started
    return report


def _needs_transformer(cfg: DailyConfig) -> bool:
    """Only pay the model-load cost if a step that uses it will actually run."""
    return (cfg.enabled("ingest") or cfg.enabled("podcasts")
            or (cfg.enabled("backfill") and cfg.backfill_transformer))


def _build_embedder(settings: dict):
    from cluster.embed import build_embedder

    embedder, _ = build_embedder(settings)
    return embedder


def _build_transformer(pcfg):
    from ingestion.pipeline import _build_transformer as build

    return build(pcfg)


def _run_step(name: str, fn, enabled: bool, progress=None) -> StepResult:
    """Run one step, converting any failure into a recorded result.

    Announces the step before running it, not after: the announcement exists to
    say what the process is currently blocked on, and one printed afterwards
    would arrive exactly when it stopped being useful.
    """
    if not enabled:
        return StepResult(name=name, status="skipped", detail="disabled")
    if progress is not None:
        progress(f"\n→ {name}")
    started = time.monotonic()
    try:
        detail = fn() or ""
        result = StepResult(name, "ok", time.monotonic() - started, detail=detail)
    except Exception as exc:
        result = StepResult(
            name, "failed", time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    if progress is not None:
        note = result.error or result.detail
        progress(f"  {name} {result.status} in {result.seconds:.1f}s"
                 f"{' — ' + note if note else ''}")
    return result


def format_report(report: DailyReport) -> str:
    """Human-readable run summary — what ran, how long, what broke."""
    icon = {"ok": "✓", "failed": "✗", "skipped": "–"}
    lines = [f"Parallax daily snapshot — started {report.started_utc}", ""]
    for s in report.steps:
        line = f"  {icon.get(s.status, '?')} {s.name:10} {s.seconds:6.1f}s  "
        line += s.error if s.status == "failed" else s.detail
        lines.append(line.rstrip())
    lines.append("")
    if report.ok:
        lines.append(f"All steps completed in {report.seconds:.1f}s.")
    else:
        names = ", ".join(s.name for s in report.failed)
        lines.append(
            f"Completed in {report.seconds:.1f}s with {len(report.failed)} "
            f"failed step(s): {names}. The dashboard was still refreshed from "
            "the data that did land."
        )
    return "\n".join(lines)


def default_out() -> Path:
    from dashboard.export import DEFAULT_OUT

    return DEFAULT_OUT
