"""Structural checks on config/sources.yaml.

These guard the source registry's invariants (schema shape, unique ids, both
families represented, sane weights) so a bad edit fails fast rather than silently
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
    assert isinstance(reg["strata"], list) and reg["strata"]
    assert isinstance(reg["catalog"], list) and reg["catalog"]
    assert isinstance(reg["personas"], list) and reg["personas"]


def test_the_default_reference_pair_exists():
    """Guardrail: `self` and `modeled_ce` are the pair every recorded snapshot was
    computed on, so they have to keep existing under those ids or the divergence
    series stops being continuous."""
    ids = {p["id"] for p in _load()["personas"]}
    assert "self" in ids
    assert "modeled_ce" in ids


def test_both_families_are_represented():
    """Guardrail: the pipeline is symmetric — a comparison needs two sides, and
    each side is modeled with the same machinery."""
    families = {p["family"] for p in _load()["personas"]}
    assert "left" in families
    assert "conservative_evangelical" in families


def test_source_ids_unique_across_file():
    seen = [src["id"] for src in _load()["catalog"]]
    assert len(seen) == len(set(seen)), "duplicate source ids in registry"


def test_stratum_ids_unique_across_file():
    seen = [s["id"] for s in _load()["strata"]]
    assert len(seen) == len(set(seen)), "duplicate stratum ids in registry"


def test_source_fields_valid():
    strata = {s["id"] for s in _load()["strata"]}
    for src in _load()["catalog"]:
        assert src["medium"] in VALID_MEDIA, src["id"]
        assert src["ingest"]["type"] in VALID_INGEST_TYPES, src["id"]
        assert src["stratum"] in strata, src["id"]
        assert src["rationale"].strip(), src["id"]


def test_every_persona_is_documented_and_internally_consistent():
    """A persona is a stronger modeling claim than a source list — it asserts
    someone consumes this mix — so an undocumented one is not reviewable."""
    reg = _load()
    strata = {s["id"] for s in reg["strata"]}
    catalog = {src["id"]: src["stratum"] for src in reg["catalog"]}
    for persona in reg["personas"]:
        pid = persona["id"]
        assert persona["label"].strip(), pid
        assert persona["family"].strip(), pid
        assert persona["description"].strip(), pid
        assert persona["strata"], pid
        assert persona["sources"], pid
        for stratum_id, weight in persona["strata"].items():
            assert stratum_id in strata, f"{pid}: unknown stratum {stratum_id}"
            assert weight > 0, f"{pid}: {stratum_id} weighted {weight}"
        for source_id, weight in persona["sources"].items():
            assert source_id in catalog, f"{pid}: unknown source {source_id}"
            assert weight > 0, f"{pid}: {source_id} weighted {weight}"
            # A listed source whose stratum carries no weight would be silently
            # zeroed, which is the failure mode the loader raises on.
            assert catalog[source_id] in persona["strata"], (
                f"{pid} lists {source_id} but gives its stratum no weight"
            )


def test_every_catalog_source_is_read_by_someone():
    """A source no persona lists is fetched by nothing and measured in nothing —
    a registry entry that looks like coverage and is not."""
    reg = _load()
    listed = {sid for p in reg["personas"] for sid in p["sources"]}
    orphans = [src["id"] for src in reg["catalog"] if src["id"] not in listed]
    assert not orphans, f"catalog sources no persona reads: {orphans}"


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
    for src in _load()["catalog"]:
        url = src["ingest"].get("url") or ""
        assert "subscriber-rss" not in url, src["id"]


def _registry_with_ingest(ingest: dict) -> str:
    return yaml.safe_dump({
        "version": 3,
        "strata": [{"id": "s", "description": "test"}],
        "catalog": [{
            "id": "src", "name": "Src", "medium": "podcast", "stratum": "s",
            "ingest": ingest, "rationale": "test",
        }],
        "personas": [{
            "id": "self", "label": "My diet", "family": "left",
            "description": "test persona",
            "strata": {"s": 1.0},
            "sources": {"src": 1.0},
        }],
    })
