"""Load the source registry and settings into typed objects.

Reads ``config/sources.yaml`` (the versioned source model) and, optionally,
``config/settings.yaml`` (non-secret operational config; falls back to the
committed ``settings.example.yaml``). Secrets are never read from here — they
live in ``.env`` (see ``.env.example``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = REPO_ROOT / "config" / "sources.yaml"
DEFAULT_SETTINGS = REPO_ROOT / "config" / "settings.yaml"
EXAMPLE_SETTINGS = REPO_ROOT / "config" / "settings.example.yaml"


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    medium: str
    role: str
    ingest_type: str
    url: str | None
    weight: float
    diet_id: str
    stratum_id: str
    # weight of this source within the whole diet = stratum_weight * source weight
    diet_weight: float


@dataclass(frozen=True)
class Diet:
    id: str
    label: str
    sources: list[Source] = field(default_factory=list)


@dataclass(frozen=True)
class Registry:
    version: int
    diets: list[Diet]

    def all_sources(self) -> list[Source]:
        return [s for d in self.diets for s in d.sources]

    def ingestable(self, ingest_types: tuple[str, ...] = ("rss",)) -> list[Source]:
        """Sources with a non-null URL and an ingest type we can process now."""
        return [
            s
            for s in self.all_sources()
            if s.url and s.ingest_type in ingest_types
        ]


def load_registry(path: str | Path = DEFAULT_SOURCES) -> Registry:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    diets: list[Diet] = []
    for d in data["diets"]:
        sources: list[Source] = []
        for stratum in d["strata"]:
            sw = float(stratum.get("stratum_weight", 1.0))
            for s in stratum["sources"]:
                ingest = s["ingest"]
                weight = float(s.get("weight", 1.0))
                sources.append(
                    Source(
                        id=s["id"],
                        name=s["name"],
                        medium=s["medium"],
                        role=s.get("role", ""),
                        ingest_type=ingest["type"],
                        url=ingest.get("url"),
                        weight=weight,
                        diet_id=d["id"],
                        stratum_id=stratum["id"],
                        diet_weight=sw * weight,
                    )
                )
        diets.append(Diet(id=d["id"], label=d.get("label", d["id"]), sources=sources))
    return Registry(version=int(data.get("version", 0)), diets=diets)


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Load settings.yaml, falling back to the committed example."""
    if path is not None:
        chosen = Path(path)
    elif DEFAULT_SETTINGS.exists():
        chosen = DEFAULT_SETTINGS
    else:
        chosen = EXAMPLE_SETTINGS
    return yaml.safe_load(chosen.read_text(encoding="utf-8")) or {}
