"""Progress output: the difference between a slow run and an apparent hang.

A run walks every source sequentially behind a 30s fetch timeout and a per-host
rate limit, so several minutes of silence is the *normal* case. Until this
existed the first real run of the pipeline looked identical to a deadlock.
"""

from __future__ import annotations

import io

from daily import runner
from daily.runner import DailyConfig, run_daily
from ingestion.__main__ import build_reporter, format_progress
from ingestion.config import Source
from ingestion.datastore import Datastore
from ingestion.pipeline import PipelineConfig, SourceProgress, run


def _source(sid="fox_news", stratum="national_cable") -> Source:
    return Source(
        id=sid,
        name=sid,
        medium="digital",
        role="national",
        ingest_type="rss",
        url=f"https://example.com/{sid}/feed",
        stratum_id=stratum,
        domain="example.com",
    )


def _event(**kw) -> SourceProgress:
    base = dict(
        index=1,
        total=16,
        source=_source(),
        stored=3,
        fetched=3,
        errors=0,
        seconds=2.1,
        failed=False,
    )
    return SourceProgress(**{**base, **kw})


# -- the line itself --------------------------------------------------------


def test_line_carries_position_source_and_stratum():
    line = format_progress(_event(index=4, total=16))
    assert "[ 4/16]" in line
    assert "fox_news" in line
    # The stratum, not a diet: a source belongs to one stratum and to as many
    # personas as list it, so there is no single diet to print here any more.
    assert "national_cable" in line


def test_line_carries_elapsed_seconds():
    """The number that distinguishes slow from stuck. Sources take a couple of
    seconds; anything near 30 means a fetch hit the timeout."""
    assert "30.4s" in format_progress(_event(seconds=30.4))


def test_an_unreachable_feed_says_so_rather_than_reporting_zero_stored():
    """A dead feed and a feed of unreachable articles both raise the error
    count, but only the first needs a fix in sources.yaml. '0 stored' would
    read as 'nothing new today'."""
    line = format_progress(_event(stored=0, fetched=0, errors=1, failed=True))
    assert "feed unreachable" in line
    assert "stored" not in line


def test_unreadable_articles_are_counted_without_failing_the_source():
    line = format_progress(_event(stored=2, fetched=3, errors=1))
    assert "2 stored / 3 fetched" in line
    assert "1 unreadable" in line
    assert "feed unreachable" not in line


def test_a_clean_source_does_not_mention_errors():
    assert "unreadable" not in format_progress(_event(errors=0))


# -- the reporter -----------------------------------------------------------


def test_reporter_writes_source_lines_and_passes_strings_through():
    out = io.StringIO()
    report = build_reporter(out)
    report(_event(index=2))
    report("\nLiberty tagging 12 document(s)…")

    text = out.getvalue()
    assert "[ 2/16]" in text
    assert "Liberty tagging 12 document(s)" in text


# -- the pipeline calls it --------------------------------------------------


class _Registry:
    def __init__(self, sources):
        self._sources = sources

    def ingestable(self, _kinds, _source_ids=None):
        return list(self._sources)


def _run_with(monkeypatch, sources, parse_feed):
    """Run the pipeline against stubbed feeds, capturing progress events."""
    import ingestion.pipeline as pipeline

    monkeypatch.setattr(pipeline, "parse_feed", parse_feed)
    events: list[object] = []
    store = Datastore(":memory:")
    try:
        run(store, _Registry(sources), PipelineConfig(), progress=events.append)
    finally:
        store.close()
    return events


def test_every_source_reports_once_in_order(monkeypatch):
    sources = [_source("a"), _source("b"), _source("c")]
    events = _run_with(monkeypatch, sources, lambda *a, **k: [])

    assert [e.source.id for e in events] == ["a", "b", "c"]
    assert [e.index for e in events] == [1, 2, 3]
    assert all(e.total == 3 for e in events)


def test_a_source_that_raises_is_reported_as_failed_not_skipped(monkeypatch):
    """The whole point of reporting per source: a broken feed has to be visible
    while the run continues, not swallowed into a final error count."""

    def parse_feed(url, *a, **k):
        if "/b/" in url:
            raise OSError("connection reset")
        return []

    events = _run_with(monkeypatch, [_source("a"), _source("b")], parse_feed)
    assert [e.failed for e in events] == [False, True]


def test_counts_are_per_source_deltas_not_running_totals(monkeypatch):
    """Cumulative numbers would make every later source look productive even
    when it stored nothing."""

    class _Item:
        title = "A headline about something"
        link = None
        published_utc = None
        summary = " ".join(["word"] * 80)

    monkeypatch.setattr(
        "ingestion.pipeline._document_text", lambda item, cfg, robots, limiter: item.summary
    )
    events = _run_with(monkeypatch, [_source("a"), _source("b")], lambda *a, **k: [_Item()])

    assert all(e.fetched == 1 for e in events)


def test_omitting_progress_keeps_the_pipeline_silent(monkeypatch):
    """Library callers and the test suite must not have to absorb printing."""
    import ingestion.pipeline as pipeline

    monkeypatch.setattr(pipeline, "parse_feed", lambda *a, **k: [])
    store = Datastore(":memory:")
    try:
        stats = run(store, _Registry([_source("a")]), PipelineConfig())
    finally:
        store.close()
    assert stats.fetched == 0  # ran, reported nothing, raised nothing


# -- the one genuinely long silent wait -------------------------------------


class _Tagger:
    name = "claude-liberty/claude-sonnet-5"

    def __init__(self, use_batch=True):
        self.use_batch = use_batch

    def score_many(self, texts):
        return {}


def _announce(n_docs, use_batch):
    from ingestion.pipeline import _announce_liberty

    said: list[str] = []
    _announce_liberty(_Tagger(use_batch), {str(i): "text" for i in range(n_docs)}, said.append)
    return " ".join(said)


def test_a_batched_liberty_job_warns_about_the_wait():
    """Ingestion finishes visibly, then the Batch API polls every 20s with no
    output. Without a warning that reads as a hang at the very last step."""
    from scoring.liberty import BATCH_MIN_ITEMS

    said = _announce(BATCH_MIN_ITEMS, use_batch=True)
    assert "Batch API" in said
    assert "20s" in said
    assert "Nothing is wrong" in said


def test_an_inline_liberty_job_does_not_warn_about_batching():
    """Below the batch threshold the calls are inline and finish promptly —
    a batch warning there would be noise that teaches the user to ignore it."""
    from scoring.liberty import BATCH_MIN_ITEMS

    said = _announce(BATCH_MIN_ITEMS - 1, use_batch=True)
    assert "Liberty tagging" in said
    assert "Batch API" not in said


def test_batching_turned_off_does_not_warn_about_batching():
    assert "Batch API" not in _announce(50, use_batch=False)


def test_the_announcement_names_the_tagger():
    assert "claude-sonnet-5" in _announce(3, use_batch=False)


# -- the daily runner announces steps ---------------------------------------


def test_daily_announces_each_step_before_running_it(monkeypatch):
    """Announced before, not after: the line exists to say what the process is
    currently blocked on, and one printed afterwards arrives exactly when it
    stops being useful."""
    order: list[str] = []

    def announce(msg):
        order.append(f"said:{msg.strip()}")

    def step(name):
        def fn(*a, **k):
            order.append(f"ran:{name}")
            return ""

        return fn

    for name in ("ingest", "backfill", "cluster", "summarize", "snapshot", "export"):
        monkeypatch.setattr(runner, f"_step_{name}", step(name))
    monkeypatch.setattr(runner, "_build_embedder", lambda settings: object())
    monkeypatch.setattr(runner, "_build_transformer", lambda pcfg: None)

    store = Datastore(":memory:")
    try:
        run_daily(DailyConfig(), store=store, progress=announce)
    finally:
        store.close()

    assert order.index("said:→ ingest") < order.index("ran:ingest")


def test_daily_reports_a_failed_step_immediately(monkeypatch):
    """`_run_step` converts exceptions into recorded results, so without this
    line a failing step is invisible until the final report — and the run keeps
    going for minutes afterwards."""
    said: list[str] = []

    def boom(*a, **k):
        raise RuntimeError("GDELT is down")

    monkeypatch.setattr(runner, "_step_ingest", boom)
    for name in ("backfill", "cluster", "summarize", "snapshot", "export"):
        monkeypatch.setattr(runner, f"_step_{name}", lambda *a, **k: "")
    monkeypatch.setattr(runner, "_build_embedder", lambda settings: object())
    monkeypatch.setattr(runner, "_build_transformer", lambda pcfg: None)

    store = Datastore(":memory:")
    try:
        report = run_daily(DailyConfig(), store=store, progress=said.append)
    finally:
        store.close()

    assert any("ingest failed" in s and "GDELT is down" in s for s in said)
    assert not report.ok


def test_daily_stays_silent_without_a_progress_callback(monkeypatch):
    for name in ("ingest", "backfill", "cluster", "summarize", "snapshot", "export"):
        monkeypatch.setattr(runner, f"_step_{name}", lambda *a, **k: "")
    monkeypatch.setattr(runner, "_build_embedder", lambda settings: object())
    monkeypatch.setattr(runner, "_build_transformer", lambda pcfg: None)

    store = Datastore(":memory:")
    try:
        assert run_daily(DailyConfig(), store=store).ok
    finally:
        store.close()
