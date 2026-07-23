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
