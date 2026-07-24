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
from urllib.parse import urlsplit

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
    # outlet domain for GDELT historical backfill (explicit, or derived from url)
    domain: str | None = None


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

    def backfillable(self) -> list[Source]:
        """Sources with a resolvable outlet domain for GDELT historical backfill.

        Includes text outlets even when their RSS url is null (e.g. AP), so long
        as an explicit domain is set — GDELT can reach them by domain."""
        return [s for s in self.all_sources() if s.domain]


# Multi-part public suffixes so "feeds.bbci.co.uk" -> "bbci.co.uk", not "co.uk".
_MULTI_TLDS = ("co.uk", "com.au", "co.nz", "org.uk", "co.za", "com.br")


def _derive_domain(url: str | None) -> str | None:
    if not url:
        return None
    host = urlsplit(url).netloc.lower().split(":")[0]
    if not host:
        return None
    for suffix in _MULTI_TLDS:
        if host.endswith("." + suffix) or host == suffix:
            labels = host.split(".")
            return ".".join(labels[-3:]) if len(labels) >= 3 else host
    labels = host.split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


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
                url = ingest.get("url")
                # Only text/rss outlets get a GDELT domain; podcast/youtube feeds
                # point at hosting infra (megaphone, youtube), not the outlet.
                explicit = s.get("domain")
                derived = _derive_domain(url) if ingest["type"] == "rss" else None
                sources.append(
                    Source(
                        id=s["id"],
                        name=s["name"],
                        medium=s["medium"],
                        role=s.get("role", ""),
                        ingest_type=ingest["type"],
                        url=url,
                        weight=weight,
                        diet_id=d["id"],
                        stratum_id=stratum["id"],
                        diet_weight=sw * weight,
                        domain=explicit or derived,
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
