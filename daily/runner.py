"""The daily snapshot: one call that refreshes everything the dashboard reads.

Six steps, in dependency order:

1. **ingest**    — fetch every RSS source, extract, dedup, score (dictionary +
                   transformer), embed.
2. **backfill**  — pull weeks of per-outlet history from GDELT so clusters rest
                   on real volume. Slow and rate-limited; on by default because
                   blindspots are only trustworthy with it.
3. **cluster**   — embeddings -> clusters -> coverage-asymmetry blindspots.
4. **summarize** — per-diet + cross-diet summaries (Claude, or the deterministic
                   fallback when no API key is set).
5. **snapshot**  — record today's compositions and divergence so the run leaves
                   a dated point behind instead of only overwriting yesterday.
6. **export**    — write the dashboard payload.

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

from compare.history import DEFAULT_SERIES_LIMIT, DEFAULT_WINDOW_DAYS
from ingestion.datastore import Datastore

# Step names, in execution order. ``snapshot`` runs before ``export`` so the
# payload carries today's point rather than lagging a day behind.
STEPS = ("ingest", "backfill", "cluster", "summarize", "snapshot", "export")


@dataclass
class DailyConfig:
    """What the daily run does. Defaults are the full snapshot."""

    db: str | None = None                 # None -> from settings
    settings_path: str | None = None
    out: str | None = None                # dashboard payload path; None -> default
    steps: tuple[str, ...] = STEPS

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

    # summarize
    model: str | None = None              # None -> summarizer default

    # snapshot
    window_days: int = DEFAULT_WINDOW_DAYS
    history_limit: int = DEFAULT_SERIES_LIMIT

    def enabled(self, step: str) -> bool:
        return step in self.steps

    @classmethod
    def from_settings(cls, settings: dict) -> DailyConfig:
        """Defaults from the ``daily:`` block of settings.yaml. CLI flags layer
        on top of this (see ``daily.__main__``)."""
        daily = (settings.get("daily", {}) or {})
        bf = (daily.get("backfill", {}) or {})
        snap = (daily.get("snapshot", {}) or {})
        cfg = cls(
            backfill_days=int(bf.get("days", 14)),
            backfill_max_per_source=int(bf.get("max_per_source", 250)),
            backfill_extract_bodies=bool(bf.get("extract_bodies", False)),
            backfill_transformer=bool(bf.get("transformer", False)),
            window_days=int(snap.get("window_days", DEFAULT_WINDOW_DAYS)),
            history_limit=int(snap.get("history_limit", DEFAULT_SERIES_LIMIT)),
        )
        if not bf.get("enabled", True):
            cfg.steps = tuple(s for s in cfg.steps if s != "backfill")
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


def _step_ingest(store, cfg: DailyConfig, pcfg, registry, embedder, transformer) -> str:
    from ingestion.pipeline import run

    stats = run(store, registry, pcfg, embedder=embedder, transformer=transformer)
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


def _step_cluster(store, cfg: DailyConfig) -> str:
    from cluster.blindspot import run_clustering

    if store.embedding_count() == 0:
        return "no embeddings yet — nothing to cluster"
    outcome = run_clustering(
        store,
        min_cluster_size=cfg.min_cluster_size,
        dominance=cfg.dominance,
        min_blindspot_size=cfg.min_blindspot_size,
    )
    return (
        f"{outcome.n_docs} docs -> {outcome.n_clusters} clusters "
        f"({outcome.n_noise} noise), {len(outcome.blindspots)} blindspots"
    )


def _step_summarize(store, cfg: DailyConfig) -> str:
    from summarize.summarizer import DEFAULT_MODEL, Summarizer

    summarizer = Summarizer(model=cfg.model or DEFAULT_MODEL)
    result = summarizer.summarize(store)
    if not result.per_diet and not result.executive:
        return "no scored documents — nothing to summarize"
    summarizer.persist(store, result)
    return f"via {result.method} (model={result.model}), {len(result.per_diet)} diets"


def _step_snapshot(store, cfg: DailyConfig) -> str:
    from compare.history import record_snapshot

    snap = record_snapshot(store, window_days=cfg.window_days)
    jsd = snap.window.jsd
    moved = f"window JSD {jsd:.3f}" if jsd is not None else "window JSD n/a (one diet)"
    return f"recorded {snap.snapshot_date} ({moved}), {store.snapshot_count()} in history"


def _step_export(store, cfg: DailyConfig) -> str:
    from dashboard.export import DEFAULT_OUT, write_payload

    out = write_payload(store, cfg.out or DEFAULT_OUT, cfg.history_limit)
    return f"wrote {out}"


def run_daily(cfg: DailyConfig | None = None, store: Datastore | None = None) -> DailyReport:
    """Run the daily snapshot. Never raises for a step failure — see the report."""
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
            "ingest": lambda: _step_ingest(store, cfg, pcfg, registry, embedder, transformer),
            "backfill": lambda: _step_backfill(store, cfg, pcfg, registry, embedder, transformer),
            "cluster": lambda: _step_cluster(store, cfg),
            "summarize": lambda: _step_summarize(store, cfg),
            "snapshot": lambda: _step_snapshot(store, cfg),
            "export": lambda: _step_export(store, cfg),
        }
        for name in STEPS:
            report.steps.append(_run_step(name, step_args[name], cfg.enabled(name)))
    finally:
        if owns_store:
            store.close()

    report.seconds = time.monotonic() - started
    return report


def _needs_transformer(cfg: DailyConfig) -> bool:
    """Only pay the model-load cost if a step that uses it will actually run."""
    return cfg.enabled("ingest") or (cfg.enabled("backfill") and cfg.backfill_transformer)


def _build_embedder(settings: dict):
    from cluster.embed import build_embedder

    embedder, _ = build_embedder(settings)
    return embedder


def _build_transformer(pcfg):
    from ingestion.pipeline import _build_transformer as build

    return build(pcfg)


def _run_step(name: str, fn, enabled: bool) -> StepResult:
    """Run one step, converting any failure into a recorded result."""
    if not enabled:
        return StepResult(name=name, status="skipped", detail="disabled")
    started = time.monotonic()
    try:
        detail = fn() or ""
        return StepResult(name, "ok", time.monotonic() - started, detail=detail)
    except Exception as exc:
        return StepResult(
            name, "failed", time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


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
