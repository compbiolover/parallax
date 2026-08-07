"""Structural checks on config/sources.yaml.

These guard the source registry's invariants (schema shape, unique ids, both
diets present, sane weights) so a bad edit fails fast rather than silently
skewing every downstream aggregate.
"""

from __future__ import annotations

import pathlib

import yaml

REGISTRY_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "sources.yaml"

VALID_MEDIA = {"news", "cable", "talk_radio", "podcast", "youtube", "newsletter"}
VALID_INGEST_TYPES = {"rss", "gdelt", "mediacloud", "podcast_rss", "youtube"}


def _load() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_registry_parses():
    assert _load() is not None


def test_top_level_shape():
    reg = _load()
    assert isinstance(reg["version"], int)
    assert reg["updated"]  # ISO date parsed by pyyaml into a date object
    assert isinstance(reg["diets"], list) and reg["diets"]


def test_both_diets_present():
    """Guardrail: the pipeline is symmetric — self and modeled diet both exist."""
    ids = {d["id"] for d in _load()["diets"]}
    assert "self" in ids
    assert "modeled_ce" in ids


def test_source_ids_unique_across_file():
    reg = _load()
    seen: list[str] = []
    for diet in reg["diets"]:
        for stratum in diet["strata"]:
            for src in stratum["sources"]:
                seen.append(src["id"])
    assert len(seen) == len(set(seen)), "duplicate source ids in registry"


def test_source_fields_valid():
    reg = _load()
    for diet in reg["diets"]:
        for stratum in diet["strata"]:
            assert 0.0 < stratum["stratum_weight"] <= 1.0
            for src in stratum["sources"]:
                assert src["medium"] in VALID_MEDIA, src["id"]
                assert src["ingest"]["type"] in VALID_INGEST_TYPES, src["id"]
                assert src["weight"] > 0, src["id"]
                assert src["rationale"].strip(), src["id"]


# -- subscriber feeds -------------------------------------------------------


def test_url_env_takes_precedence_when_the_variable_is_set(monkeypatch, tmp_path):
    """A subscriber feed URL is a credential, so it is named in the registry
    and read from the environment — never written into a public file."""
    from ingestion.config import load_registry

    monkeypatch.setenv("PARALLAX_TEST_FEED", "https://example.com/private?id=tok")
    path = tmp_path / "sources.yaml"
    path.write_text(_registry_with_ingest(
        {"type": "podcast_rss", "url_env": "PARALLAX_TEST_FEED",
         "url": "https://example.com/public.xml"}), encoding="utf-8")
    assert load_registry(path).all_sources()[0].url == "https://example.com/private?id=tok"


def test_url_is_the_fallback_when_the_variable_is_unset(monkeypatch, tmp_path):
    """Not holding the subscription is a supported state: the public feed is a
    truncated version of the same show, not nothing."""
    from ingestion.config import load_registry

    monkeypatch.delenv("PARALLAX_TEST_FEED", raising=False)
    path = tmp_path / "sources.yaml"
    path.write_text(_registry_with_ingest(
        {"type": "podcast_rss", "url_env": "PARALLAX_TEST_FEED",
         "url": "https://example.com/public.xml"}), encoding="utf-8")
    assert load_registry(path).all_sources()[0].url == "https://example.com/public.xml"


def test_an_empty_variable_is_treated_as_unset(monkeypatch, tmp_path):
    """`export PARALLAX_FEED_X=` is how a secret that failed to resolve arrives.
    Using it would request the empty string and fail somewhere less obvious."""
    from ingestion.config import load_registry

    monkeypatch.setenv("PARALLAX_TEST_FEED", "   ")
    path = tmp_path / "sources.yaml"
    path.write_text(_registry_with_ingest(
        {"type": "podcast_rss", "url_env": "PARALLAX_TEST_FEED",
         "url": "https://example.com/public.xml"}), encoding="utf-8")
    assert load_registry(path).all_sources()[0].url == "https://example.com/public.xml"


def test_a_secret_url_is_never_logged(monkeypatch, tmp_path, caplog):
    """The point of keeping it out of git is lost if it lands in the log file."""
    import logging

    from ingestion.config import load_registry

    secret = "https://example.com/private?id=SECRETTOKEN"
    monkeypatch.setenv("PARALLAX_TEST_FEED", secret)
    path = tmp_path / "sources.yaml"
    path.write_text(_registry_with_ingest(
        {"type": "podcast_rss", "url_env": "PARALLAX_TEST_FEED"}), encoding="utf-8")
    with caplog.at_level(logging.DEBUG):
        load_registry(path)
    assert "SECRETTOKEN" not in caplog.text


def test_the_registry_never_hardcodes_a_subscriber_url():
    """A token pasted into the registry instead of named there is the mistake
    this whole mechanism exists to prevent, and it is invisible on review."""
    for diet in _load()["diets"]:
        for stratum in diet["strata"]:
            for src in stratum["sources"]:
                url = src["ingest"].get("url") or ""
                assert "subscriber-rss" not in url, src["id"]


def _registry_with_ingest(ingest: dict) -> str:
    return yaml.safe_dump({
        "version": 1,
        "diets": [{
            "id": "self",
            "label": "My diet",
            "strata": [{
                "id": "s",
                "stratum_weight": 1.0,
                "sources": [{
                    "id": "src", "name": "Src", "medium": "podcast",
                    "ingest": ingest, "weight": 1.0, "rationale": "test",
                }],
            }],
        }],
    })
