"""The one-command daily snapshot: step selection, isolation, and config layering."""

from __future__ import annotations

import daily.runner as runner
from daily.__main__ import _parse, build_config
from daily.runner import STEPS, DailyConfig, format_report, run_daily
from ingestion.datastore import Datastore


def _stub_all(monkeypatch, record: list, failing: str | None = None):
    """Replace every step with a recorder so the runner is tested in isolation
    (no network, no models, no clustering)."""

    def make(name):
        def step(*args, **kwargs):
            record.append(name)
            if name == failing:
                raise RuntimeError(f"{name} exploded")
            return f"{name} detail"
        return step

    for name in STEPS:
        monkeypatch.setattr(runner, f"_step_{name}", make(name))
    monkeypatch.setattr(runner, "_build_embedder", lambda settings: object())
    monkeypatch.setattr(runner, "_build_transformer", lambda pcfg: None)


def _run(monkeypatch, cfg, record, failing=None):
    _stub_all(monkeypatch, record, failing)
    store = Datastore(":memory:")
    try:
        return run_daily(cfg, store=store)
    finally:
        store.close()


# -- ordering and selection ------------------------------------------------

def test_runs_all_steps_in_dependency_order(monkeypatch):
    record: list[str] = []
    report = _run(monkeypatch, DailyConfig(), record)
    assert record == list(STEPS)
    assert report.ok
    assert all(s.status == "ok" for s in report.steps)


def test_backfill_included_by_default():
    # The daily snapshot is the *full* one — backfill is not opt-in.
    assert "backfill" in DailyConfig().steps


def test_skip_marks_step_skipped_not_missing(monkeypatch):
    record: list[str] = []
    cfg = DailyConfig(steps=tuple(s for s in STEPS if s != "backfill"))
    report = _run(monkeypatch, cfg, record)
    assert "backfill" not in record
    assert report.get("backfill").status == "skipped"
    assert report.ok  # skipped is not a failure


def test_only_runs_selected_steps(monkeypatch):
    record: list[str] = []
    report = _run(monkeypatch, DailyConfig(steps=("cluster", "export")), record)
    assert record == ["cluster", "export"]
    assert report.get("ingest").status == "skipped"


# -- isolation: one failure must not stop the run --------------------------

def test_failed_step_does_not_abort_the_rest(monkeypatch):
    record: list[str] = []
    report = _run(monkeypatch, DailyConfig(), record, failing="backfill")
    # everything still ran...
    assert record == list(STEPS)
    # ...and the dashboard export in particular still happened
    assert report.get("export").status == "ok"
    assert report.get("backfill").status == "failed"
    assert "RuntimeError" in report.get("backfill").error
    assert not report.ok
    assert [s.name for s in report.failed] == ["backfill"]


def test_report_formats_failure_visibly(monkeypatch):
    record: list[str] = []
    report = _run(monkeypatch, DailyConfig(), record, failing="summarize")
    text = format_report(report)
    assert "summarize" in text and "RuntimeError" in text
    assert "still refreshed" in text


# -- transformer is loaded once, and only when needed ----------------------

def test_transformer_built_once_and_shared(monkeypatch):
    calls: list[object] = []
    for name in STEPS:
        monkeypatch.setattr(runner, f"_step_{name}", lambda *a, **k: "")
    monkeypatch.setattr(runner, "_build_embedder", lambda settings: object())
    monkeypatch.setattr(runner, "_build_transformer",
                        lambda pcfg: calls.append(pcfg) or "TR")
    store = Datastore(":memory:")
    try:
        run_daily(DailyConfig(), store=store)
    finally:
        store.close()
    assert len(calls) == 1  # five RoBERTas loaded once, not per step


def test_transformer_not_built_when_no_step_needs_it(monkeypatch):
    calls: list[object] = []
    for name in STEPS:
        monkeypatch.setattr(runner, f"_step_{name}", lambda *a, **k: "")
    monkeypatch.setattr(runner, "_build_embedder", lambda settings: object())
    monkeypatch.setattr(runner, "_build_transformer", lambda pcfg: calls.append(pcfg))
    store = Datastore(":memory:")
    try:
        # export-only: nothing scores documents, so don't pay the model load
        run_daily(DailyConfig(steps=("export",)), store=store)
    finally:
        store.close()
    assert calls == []


# -- settings <-> flag layering --------------------------------------------

def test_settings_block_drives_backfill_defaults():
    cfg = DailyConfig.from_settings(
        {"daily": {"backfill": {"days": 3, "max_per_source": 50, "transformer": True}}}
    )
    assert cfg.backfill_days == 3
    assert cfg.backfill_max_per_source == 50
    assert cfg.backfill_transformer is True
    assert "backfill" in cfg.steps


def test_settings_can_disable_backfill():
    cfg = DailyConfig.from_settings({"daily": {"backfill": {"enabled": False}}})
    assert "backfill" not in cfg.steps


def test_flags_override_settings_but_absent_flags_do_not():
    settings = {"daily": {"backfill": {"days": 3, "max_per_source": 50}}}
    cfg = build_config(_parse(["--backfill-days", "21"]), settings)
    assert cfg.backfill_days == 21     # flag wins
    assert cfg.backfill_max_per_source == 50  # untouched by an absent flag


def test_skip_flag_composes_with_disabled_settings_step():
    settings = {"daily": {"backfill": {"enabled": False}}}
    cfg = build_config(_parse(["--skip", "summarize"]), settings)
    assert "backfill" not in cfg.steps   # from settings
    assert "summarize" not in cfg.steps  # from the flag
    assert "ingest" in cfg.steps


def test_failed_transformer_build_is_not_retried_by_ingest(monkeypatch):
    # daily builds the transformer once; if that fails, the ingest step must not
    # re-attempt it (a slow failure would be paid twice and warn twice).
    seen = {}

    def fake_ingest(store, cfg, pcfg, registry, embedder, transformer, progress=None):
        seen["enabled"] = pcfg.transformer_enabled
        seen["transformer"] = transformer
        return ""

    monkeypatch.setattr(runner, "_step_ingest", fake_ingest)
    for name in ("backfill", "cluster", "summarize", "snapshot", "export"):
        monkeypatch.setattr(runner, f"_step_{name}", lambda *a, **k: "")
    monkeypatch.setattr(runner, "_build_embedder", lambda settings: object())
    monkeypatch.setattr(runner, "_build_transformer", lambda pcfg: None)  # build fails

    store = Datastore(":memory:")
    try:
        run_daily(DailyConfig(), store=store)
    finally:
        store.close()
    assert seen["transformer"] is None
    assert seen["enabled"] is False   # run() won't rebuild


# -- snapshot step ---------------------------------------------------------

def test_snapshot_runs_before_export():
    # The payload has to carry today's point, not lag a day behind it.
    assert STEPS.index("snapshot") < STEPS.index("export")


def test_snapshot_step_records_a_dated_row():
    """Unstubbed: the real step, against a real store."""
    store = Datastore(":memory:")
    try:
        for diet, care in (("self", 0.5), ("modeled_ce", 0.1)):
            store.upsert_document(
                doc_id=f"{diet}-d", diet_id=diet, source_id="s", stratum_id=None,
                url=None, title="t", published_utc=None,
                fetched_utc="2026-07-25T00:00:00+00:00", word_count=200, minhash=None,
            )
            store.upsert_scores(
                document_id=f"{diet}-d", scorer="dictionary",
                foundations={"care": care, "fairness": 0.1, "loyalty": 0.2,
                             "authority": 0.1, "sanctity": 0.1},
                sentiment=0.0, moral_word_ratio=0.2, matched_words=20,
            )
        detail = runner._step_snapshot(store, DailyConfig())
        assert store.snapshot_count() == 1
        assert "1 in history" in detail
    finally:
        store.close()


def test_snapshot_window_days_layer_from_settings_and_flags():
    settings = {"daily": {"snapshot": {"window_days": 14}}}
    assert build_config(_parse([]), settings).window_days == 14
    assert build_config(_parse(["--window-days", "3"]), settings).window_days == 3
