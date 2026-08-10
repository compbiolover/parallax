"""Load the source registry and settings into typed objects.

Reads ``config/sources.yaml`` (the versioned source model) and, optionally,
``config/settings.yaml`` (non-secret operational config; falls back to the
committed ``settings.example.yaml``). Secrets are never read from here — they
live in the environment (see ``.env.example``).

The registry separates *sources* from *personas*. A source is a fact about a
publisher; a persona is a fact about a reader. Several personas can therefore
share one source — a passive cable viewer and a devout evangelical both watch
Fox — and the catalog holds it once, so it is fetched, scored, and embedded once
however many personas include it. That is what makes adding a persona free at
runtime, and it is why ``Source`` carries no diet identity at all.

A user's own diet does not belong in the committed registry: it describes a real
person's real consumption, which is the one thing ``CLAUDE.md`` §0 promises this
project does not collect. It goes in a gitignored overlay file instead, merged
over the public one by ``load_registry``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = REPO_ROOT / "config" / "sources.yaml"
DEFAULT_OVERLAY = REPO_ROOT / "config" / "personas.local.yaml"
DEFAULT_SETTINGS = REPO_ROOT / "config" / "settings.yaml"
EXAMPLE_SETTINGS = REPO_ROOT / "config" / "settings.example.yaml"

# The registry schema this loader understands. Version 2 nested sources inside
# diets, which made a source diet-private and a shared source impossible.
MIN_REGISTRY_VERSION = 3


@dataclass(frozen=True)
class Stratum:
    """A medium/role grouping. Global, because which stratum a source belongs to
    is a property of the source rather than of whoever consumes it."""

    id: str
    description: str = ""


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    medium: str
    role: str
    ingest_type: str
    url: str | None
    stratum_id: str
    # outlet domain for GDELT historical backfill (explicit, or derived from url)
    domain: str | None = None


@dataclass(frozen=True)
class Persona:
    """A weight profile over the source catalog.

    ``source_weights`` is the membership list as well as the weighting: a persona
    consumes exactly the sources it names. Membership is not inherited from the
    stratum, because strata are shared — ``podcasts`` holds both NPR's Up First
    and Relatable — so inheriting would hand every persona every other persona's
    audio. The alternative, an exclusion list per persona, would have to grow
    every time anyone added a source, and a missed entry would silently move a
    profile.
    """

    id: str
    label: str
    short_label: str = ""
    family: str = ""
    description: str = ""
    stratum_weights: dict[str, float] = field(default_factory=dict)
    source_weights: dict[str, float] = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        return self.short_label or self.label

    @property
    def source_ids(self) -> frozenset[str]:
        return frozenset(self.source_weights)

    def weight_for(self, source: Source) -> float:
        """This persona's effective weight for one source, or 0.0 if it does not
        consume it.

        Only *relative* weights matter downstream — ``aggregate_profile``
        divides by the total — so the absolute scale carries no meaning.
        """
        within = self.source_weights.get(source.id)
        if within is None:
            return 0.0
        return self.stratum_weights.get(source.stratum_id, 0.0) * within


@dataclass(frozen=True)
class Registry:
    version: int
    strata: list[Stratum]
    sources: list[Source]
    personas: list[Persona]

    def all_sources(self) -> list[Source]:
        """Every catalog source, once. Personas share these rather than owning
        them, so ingestion over this list never fetches the same feed twice."""
        return list(self.sources)

    def ingestable(
        self,
        ingest_types: tuple[str, ...] = ("rss",),
        source_ids: Collection[str] | None = None,
    ) -> list[Source]:
        """Sources with a non-null URL and an ingest type we can process now.

        ``source_ids`` restricts the run to a subset — see :meth:`scope`. Adding a
        persona is free, but adding the *sources* a new persona needs is not: they
        are fetched, extracted, embedded, and liberty-tagged like any other.
        """
        return [
            s for s in self.sources
            if s.url and s.ingest_type in ingest_types
            and (source_ids is None or s.id in source_ids)
        ]

    def backfillable(self, source_ids: Collection[str] | None = None) -> list[Source]:
        """Sources with a resolvable outlet domain for GDELT historical backfill.

        Includes text outlets even when their RSS url is null (e.g. AP), so long
        as an explicit domain is set — GDELT can reach them by domain."""
        return [
            s for s in self.sources
            if s.domain and (source_ids is None or s.id in source_ids)
        ]

    def scope(self, personas: Collection[str] | None) -> set[str] | None:
        """The union of those personas' sources, or ``None`` for the whole catalog.

        The lever for the one cost personas genuinely add. Ingestion, scoring,
        embedding and liberty tagging are per *document*, so they scale with the
        catalog rather than with the persona count — and a library covering four
        variants a side needs several times the sources two diets did.

        Narrowing it is not free either: a persona whose sources are never fetched
        has an empty profile and renders as a blank column, so the default is the
        whole catalog and this exists for someone who has looked at the bill.
        """
        if personas is None:
            return None
        wanted: set[str] = set()
        for persona_id in personas:
            wanted |= set(self.weights_for(persona_id))
        return wanted

    def source(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def persona(self, persona_id: str) -> Persona | None:
        return next((p for p in self.personas if p.id == persona_id), None)

    def persona_ids(self) -> list[str]:
        """The personas, in registry order.

        The registry is the authority on which diets exist — not the datastore,
        which only knows which sources happened to yield documents. A persona
        whose feeds were all unreachable today still exists, and dropping it
        would silently reshape every comparison.
        """
        return [p.id for p in self.personas]

    def weights_for(self, persona_id: str) -> dict[str, float]:
        """``{source_id: weight}`` for one persona; empty if it is unknown.

        This is the whole of the old ``Source.diet_weight`` arithmetic, moved to
        where it belongs: a weight is a fact about a reader, so it cannot live on
        a source row shared by many readers.
        """
        persona = self.persona(persona_id)
        if persona is None:
            return {}
        by_id = {s.id: s for s in self.sources}
        weights = {}
        for source_id in persona.source_weights:
            source = by_id.get(source_id)
            if source is not None:
                weights[source_id] = persona.weight_for(source)
        return weights

    def families(self) -> dict[str, list[str]]:
        """``{family: [persona_id, ...]}`` in registry order."""
        out: dict[str, list[str]] = {}
        for persona in self.personas:
            out.setdefault(persona.family or "unassigned", []).append(persona.id)
        return out


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


def _resolve_url(source_id: str, ingest: dict) -> str | None:
    """The feed URL, taking a secret one from the environment when there is one.

    A subscriber feed is itself the credential: Sam Harris hands out a per-account
    URL with a token in it, and publishers that do this say plainly not to share
    it. That makes it a secret in a file whose whole point is being public, so it
    is named rather than written — ``url_env: PARALLAX_FEED_X`` reads
    ``$PARALLAX_FEED_X``, which is already how the scheduled run gets everything
    else (``scripts/parallax-daily.sh`` exports from Bitwarden Secrets Manager
    before handing off to the job).

    ``url`` stays as the fallback, so an entry can carry the public feed for
    anyone without the subscription and quietly upgrade for anyone with it. That
    ordering is the point: the private feed is a superset, and the public one is
    the degraded mode.

    Never log the resolved value — that would put the token in the log file the
    secret was kept out of git to avoid.
    """
    env_name = ingest.get("url_env")
    if env_name:
        secret = os.environ.get(env_name, "").strip()
        if secret:
            return secret
        logger.info(
            "%s: %s is not set, falling back to the public feed", source_id, env_name
        )
    return ingest.get("url")


def _merge_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge a local registry over the committed one, by id.

    An entry whose id already exists is replaced wholesale rather than merged
    field by field: a half-overridden persona, carrying some weights from the
    public file and some from yours, is not something anyone could reason about.
    New ids are appended.
    """
    merged = dict(base)
    for section in ("strata", "catalog", "personas"):
        incoming = overlay.get(section) or []
        if not incoming:
            continue
        existing = list(merged.get(section) or [])
        by_id = {entry["id"]: i for i, entry in enumerate(existing)}
        for entry in incoming:
            entry_id = entry.get("id")
            if entry_id is None:
                raise ValueError(f"overlay {section} entry has no id: {entry!r}")
            if entry_id in by_id:
                existing[by_id[entry_id]] = entry
            else:
                by_id[entry_id] = len(existing)
                existing.append(entry)
        merged[section] = existing
    return merged


def _registry_path(explicit: str | Path | None, settings: dict[str, Any] | None) -> Path:
    """Which registry file to read: argument, then settings, then the committed one.

    A missing configured path is an error rather than a silent fall back to the
    default. Pointing at a registry that is not there and getting the public one
    anyway is how you spend an afternoon wondering why your personas are absent.
    """
    if explicit is not None:
        return Path(explicit)
    configured = ((settings or {}).get("sources") or {}).get("registry")
    if configured:
        path = Path(configured)
        if not path.exists():
            raise FileNotFoundError(
                f"sources.registry points at {path}, which does not exist"
            )
        return path
    return DEFAULT_SOURCES


def _overlay_path(
    explicit: str | Path | None, settings: dict[str, Any] | None
) -> tuple[Path | None, bool]:
    """``(path, required)`` for the local persona overlay.

    Three tiers, and they differ in what a missing file means:

    - an explicit argument is **required** — someone named this file, and quietly
      running against the public registry instead would compare a diet they think
      they replaced;
    - a path from ``sources.local_registry`` is optional, because
      ``settings.example.yaml`` ships it pointing at a file that does not exist
      until you write one, exactly as it documents ``config/settings.yaml``;
    - the conventional path is used only if it happens to be there.
    """
    if explicit is not None:
        return Path(explicit), True
    configured = ((settings or {}).get("sources") or {}).get("local_registry")
    if configured:
        return Path(configured), False
    return (DEFAULT_OVERLAY if DEFAULT_OVERLAY.exists() else None), False


def _build_sources(catalog: list[dict], strata_ids: set[str]) -> list[Source]:
    sources: list[Source] = []
    seen: set[str] = set()
    for entry in catalog:
        source_id = entry["id"]
        if source_id in seen:
            raise ValueError(f"duplicate source id in registry: {source_id}")
        seen.add(source_id)
        stratum_id = entry.get("stratum")
        if stratum_id not in strata_ids:
            raise ValueError(
                f"source {source_id} names stratum {stratum_id!r}, which is not declared"
            )
        ingest = entry["ingest"]
        url = _resolve_url(source_id, ingest)
        # Only text/rss outlets get a GDELT domain; podcast/youtube feeds
        # point at hosting infra (megaphone, youtube), not the outlet.
        explicit = entry.get("domain")
        derived = _derive_domain(url) if ingest["type"] == "rss" else None
        sources.append(
            Source(
                id=source_id,
                name=entry["name"],
                medium=entry["medium"],
                role=entry.get("role", ""),
                ingest_type=ingest["type"],
                url=url,
                stratum_id=stratum_id,
                domain=explicit or derived,
            )
        )
    return sources


def _build_personas(raw: list[dict], sources: list[Source]) -> list[Persona]:
    stratum_of = {s.id: s.stratum_id for s in sources}
    personas: list[Persona] = []
    seen: set[str] = set()
    for entry in raw:
        persona_id = entry["id"]
        if persona_id in seen:
            raise ValueError(f"duplicate persona id in registry: {persona_id}")
        seen.add(persona_id)
        stratum_weights = {k: float(v) for k, v in (entry.get("strata") or {}).items()}
        source_weights = {k: float(v) for k, v in (entry.get("sources") or {}).items()}
        for source_id in source_weights:
            if source_id not in stratum_of:
                raise ValueError(
                    f"persona {persona_id} lists source {source_id!r}, which is not in the catalog"
                )
            stratum_id = stratum_of[source_id]
            if stratum_id not in stratum_weights:
                # Silently weighting this at zero would drop a source the
                # persona plainly means to consume, and the profile would just
                # come out slightly wrong with nothing to point at.
                raise ValueError(
                    f"persona {persona_id} lists source {source_id!r} but gives its "
                    f"stratum {stratum_id!r} no weight"
                )
        personas.append(
            Persona(
                id=persona_id,
                label=entry.get("label", persona_id),
                short_label=entry.get("short_label", ""),
                family=entry.get("family", ""),
                description=(entry.get("description") or "").strip(),
                stratum_weights=stratum_weights,
                source_weights=source_weights,
            )
        )
    return personas


def load_registry(
    path: str | Path | None = None,
    overlay: str | Path | None = None,
    settings: dict[str, Any] | None = None,
) -> Registry:
    """Load the registry, merging the local persona overlay when there is one.

    ``path`` wins over ``settings["sources"]["registry"]``, which wins over
    ``config/sources.yaml``. ``path`` defaults to ``None`` rather than to the
    committed registry so that "no path given" stays distinguishable from "the
    default path given explicitly" — otherwise the settings key can never take
    effect, which is precisely how it sat documented and dead.

    ``overlay`` wins over ``settings["sources"]["local_registry"]``, which wins
    over ``config/personas.local.yaml`` if it exists. Most installations have no
    overlay at all, which is why only an explicitly-passed one is required to
    exist — see :func:`_overlay_path`.
    """
    chosen = _registry_path(path, settings)
    data = yaml.safe_load(Path(chosen).read_text(encoding="utf-8"))
    version = int(data.get("version", 0))
    if version < MIN_REGISTRY_VERSION:
        raise ValueError(
            f"{chosen}: registry version {version} nests sources inside diets, which "
            f"cannot express a source shared by several personas. Version "
            f"{MIN_REGISTRY_VERSION} splits it into `catalog` and `personas` — see the "
            f"schema comment at the top of config/sources.yaml."
        )

    overlay_path, required = _overlay_path(overlay, settings)
    if overlay_path is not None and not overlay_path.exists() and required:
        raise FileNotFoundError(f"persona overlay not found: {overlay_path}")
    if overlay_path is not None and overlay_path.exists():
        local = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
        data = _merge_overlay(data, local)
        # Name the file but never its contents: it describes a real person's
        # media consumption, and a log is a file too.
        logger.info("merged persona overlay from %s", overlay_path)

    strata = [
        Stratum(id=s["id"], description=(s.get("description") or "").strip())
        for s in (data.get("strata") or [])
    ]
    if len({s.id for s in strata}) != len(strata):
        raise ValueError("duplicate stratum id in registry")
    sources = _build_sources(data.get("catalog") or [], {s.id for s in strata})
    personas = _build_personas(data.get("personas") or [], sources)
    return Registry(version=version, strata=strata, sources=sources, personas=personas)


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Load settings.yaml, falling back to the committed example."""
    if path is not None:
        chosen = Path(path)
    elif DEFAULT_SETTINGS.exists():
        chosen = DEFAULT_SETTINGS
    else:
        chosen = EXAMPLE_SETTINGS
    return yaml.safe_load(chosen.read_text(encoding="utf-8")) or {}
