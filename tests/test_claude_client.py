"""Claude client construction: the two ordinary setup failures, said out loud."""

from __future__ import annotations

import builtins
import logging

import scoring.claude_client as cc
from scoring.claude_client import NO_KEY, NO_PACKAGE, build_client, describe_availability
from scoring.liberty import build_tagger


def _hide_anthropic(monkeypatch):
    """Simulate the `llm` extra not being installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_key_is_reported_with_the_env_caveat(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, reason = build_client()
    assert client is None
    assert reason == NO_KEY
    # The two things that actually bite: .env isn't read, and cron isn't your shell.
    assert ".env is NOT read automatically" in reason
    assert "crontab" in reason


def test_missing_package_names_the_extra(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    _hide_anthropic(monkeypatch)
    client, reason = build_client()
    assert client is None
    assert reason == NO_PACKAGE
    # The whole point: `[dev]` alone doesn't install it, and the message says so.
    assert '".[dev,llm]"' in reason


def test_key_missing_takes_precedence_over_package(monkeypatch):
    """Both broken at once should report the key — it's the first thing to fix
    and the one that stays broken after a reinstall."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _hide_anthropic(monkeypatch)
    assert build_client()[1] == NO_KEY


def test_success_returns_a_client_and_no_reason(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    sentinel = object()
    monkeypatch.setattr(cc, "build_client", lambda: (sentinel, ""))
    assert cc.build_client() == (sentinel, "")


def test_describe_availability_is_a_one_liner(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert describe_availability() == NO_KEY


# -- the callers warn rather than failing silently -------------------------

def test_build_tagger_warns_with_the_reason(monkeypatch, caplog):
    """The regression this exists to prevent: returning None at INFO level, so a
    misconfigured run produced no liberty scores and no visible explanation."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        assert build_tagger() is None
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert NO_KEY in caplog.text


def test_disabling_deliberately_stays_quiet(monkeypatch, caplog):
    """Opting out in settings is a choice, not a misconfiguration — no warning."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        assert build_tagger(enabled=False) is None
    assert caplog.text == ""


def test_summarizer_warns_before_falling_back(monkeypatch, caplog):
    from summarize.summarizer import _build_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        assert _build_client() is None
    assert "deterministic fallback" in caplog.text
    assert NO_KEY in caplog.text
