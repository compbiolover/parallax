"""Export the dashboard data payload from the datastore.

Writes a small JavaScript file (``window.PARALLAX_DATA = {...}``) rather than a
bare ``.json`` so the static page renders when opened directly from disk
(``file://``), where ``fetch`` of a sibling JSON is blocked by the browser.

The payload is aggregate-only — compositions, divergence, log-ratios, document
counts, the recorded snapshot history, and the generated summaries. No raw text,
consistent with the §0 content-handling guardrail.

Reading is all this module does. Recording a snapshot is a separate act (the
daily runner's ``snapshot`` step, or ``python -m compare.history --record``), so
exporting a payload never mutates the datastore.

    python -m dashboard.export --db data/parallax.sqlite
    python -m dashboard.export --out dashboard/public/data/latest.js
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from compare.divergence import index_form, jensen_shannon_divergence, log_ratios
from compare.history import DEFAULT_SERIES_LIMIT, load_series
from compare.reference import ReferencePair, resolve
from ingestion.config import Registry, datastore_path, load_registry, load_settings
from ingestion.datastore import Datastore
from ingestion.pipeline import persona_profiles
from scoring.foundations import CLASSIC_FOUNDATIONS
from scoring.lexicon import is_demo_lexicon

DEFAULT_OUT = Path(__file__).resolve().parent / "public" / "data" / "latest.js"
DEFAULT_CATALOG_OUT = Path(__file__).resolve().parent / "public" / "data" / "catalog.js"


def _caveat(lexicon: str | None, has_bands: bool = False) -> str:
    band_note = (
        " Whiskers on the radar show the dictionary-vs-transformer range — wider "
        "means the two methods disagree more, so trust that foundation's number less."
        if has_bands else ""
    )
    base = (
        "Scores come from a dictionary method and cover the five classic "
        "foundations only (no liberty). Read every number as a noisy estimate, "
        "never ground truth. See LIMITATIONS.md."
    )
    if is_demo_lexicon(lexicon):
        return (
            "Scores come from a dictionary method over a small DEMO lexicon "
            "(illustrative only) and cover the five classic foundations only "
            "(no liberty). Read every number as a noisy estimate, never ground "
            "truth. See LIMITATIONS.md." + band_note
        )
    return f"{base} Lexicon: {lexicon}.{band_note}"


def _band_payload(store: Datastore, registry: Registry) -> tuple[dict, str | None]:
    """Per-persona confidence bands keyed by persona id, plus the transformer
    scorer name that produced them (``None`` if the transformer never ran)."""
    transformer_scorer = store.get_meta("transformer_scorer")
    if not transformer_scorer:
        return {}, None
    from compare.confidence import all_persona_bands

    raw = all_persona_bands(store, registry, transformer_scorer)
    payload = {
        diet_id: {
            f: {
                "point": b.point, "low": b.low, "high": b.high,
                "dictionary": b.dictionary, "transformer": b.transformer,
                "disagreement": b.disagreement,
            }
            for f, b in bands.items()
        }
        for diet_id, bands in raw.items()
    }
    return payload, (transformer_scorer if payload else None)


def _matrix_payload(profiles: dict[str, dict[str, float]], order: list[str]) -> dict:
    """Pairwise foundation divergence over every persona, and their source overlap.

    The overlap matrix is not decoration. Personas are weight profiles over one
    shared catalog, so two of them that read mostly the same outlets have a small
    JSD *by construction*. A grid of small numbers on its own reads as "the
    personas are interchangeable", which is the opposite of what it shows. Sitting
    the overlap beside it is what makes the divergence legible: low divergence plus
    high overlap is an artifact, low divergence plus low overlap is a finding.
    """
    present = [p for p in order if p in profiles]
    jsd = [
        [jensen_shannon_divergence(profiles[a], profiles[b]) for b in present]
        for a in present
    ]
    return {"personas": present, "jsd": jsd}


def _overlap_matrix(registry: Registry, order: list[str]) -> dict:
    """Weighted-cosine overlap between personas' source weightings.

    Registry-only, so it costs nothing and is available with an empty corpus.
    """
    weights = {p: registry.weights_for(p) for p in order}

    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        shared = set(a) & set(b)
        dot = sum(a[s] * b[s] for s in shared)
        na = sum(v * v for v in a.values()) ** 0.5
        nb = sum(v * v for v in b.values()) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    return {
        "personas": list(order),
        "cosine": [[cosine(weights[a], weights[b]) for b in order] for a in order],
    }


def build_payload(
    store: Datastore,
    registry: Registry,
    pair: ReferencePair,
    history_limit: int | None = DEFAULT_SERIES_LIMIT,
) -> dict:
    profiles = persona_profiles(store, registry)
    summaries = store.all_summaries()
    bands, transformer_scorer = _band_payload(store, registry)
    members = {p: set(registry.weights_for(p)) for p in pair}

    # The registry's human labels, recorded at ingestion. Emitting the id as the
    # label made every surface print "modeled_ce" at the reader. `label` reads
    # as a noun phrase in prose; `short_label` is what a heading or legend uses.
    labels = store.diet_labels()
    short_labels = store.diet_short_labels()

    # Reference pair first, then registry order. Ordering used to be
    # `ORDER BY diet_id`, which made "which one is first" an alphabetical accident
    # — and every surface that coloured by position inherited that accident.
    rest = [p for p in registry.persona_ids() if p not in pair.ids]
    order = [*pair.ids, *rest]

    diets = []
    for persona_id in order:
        if persona_id not in profiles:
            continue
        persona = registry.persona(persona_id)
        srow = summaries.get(persona_id)
        label = labels.get(persona_id) or (persona.label if persona else persona_id)
        weights = registry.weights_for(persona_id)
        diets.append(
            {
                "id": persona_id,
                "label": label,
                "short_label": short_labels.get(persona_id) or label,
                # The side of the comparison. Surfaces colour by this rather than
                # by list position, which is the bug digest/render.py documents.
                "family": persona.family if persona else "",
                "description": persona.description if persona else "",
                "role": (
                    "mine" if persona_id == pair.mine
                    else "theirs" if persona_id == pair.theirs
                    else ""
                ),
                "doc_count": store.doc_count_for_sources(weights),
                "source_count": len(weights),
                "profile": {f: profile_of.get(f, 0.0) for f in CLASSIC_FOUNDATIONS}
                if (profile_of := profiles[persona_id]) else {},
                "band": bands.get(persona_id),  # None unless the transformer ran
                "summary": srow["text"] if srow else "",
            }
        )

    comparison = None
    if pair.mine in profiles and pair.theirs in profiles:
        a, b = pair.mine, pair.theirs
        comparison = {
            "pair": [a, b],
            # Oriented mine-first, so a positive log-ratio means "my diet
            # over-indexes", which is what CLAUDE.md §3(5) asks for. The previous
            # `sorted(ids)[:2]` put `modeled_ce` first and inverted the sign.
            "orientation": "mine_first",
            "jsd": jensen_shannon_divergence(profiles[a], profiles[b]),
            "log_ratios": log_ratios(profiles[a], profiles[b]),
            "index_form": index_form(profiles[a], profiles[b]),
        }

    exec_row = summaries.get("executive")
    lexicon = store.get_meta("lexicon")
    has_bands = transformer_scorer is not None
    history = load_series(store, history_limit)
    blindspots, blindspot_themes = _blindspot_payload(store, members)
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "foundations": list(CLASSIC_FOUNDATIONS),
        # Still called `diets` deliberately: renaming it would break every
        # latest.js already on disk and the page that reads it.
        "diets": diets,
        "reference": {"mine": pair.mine, "theirs": pair.theirs},
        "families": registry.families(),
        "registry_version": registry.version,
        # Every persona against every other, with the source overlap that explains
        # how much of each number is shared vocabulary rather than agreement.
        "matrix": _matrix_payload(profiles, order),
        "overlap": _overlap_matrix(registry, [d["id"] for d in diets]),
        "comparison": comparison,
        # Dated history. ``history_window_days`` is the trailing window behind the
        # windowed series; it comes off the newest row, so a settings change is
        # described accurately for the points it actually applies to going forward.
        "history": history,
        "history_window_days": history[-1]["window_days"] if history else None,
        "fairness_split": _fairness_payload(store, registry, pair),
        "liberty": _liberty_payload(store, registry, pair),
        "executive_summary": exec_row["text"] if exec_row else "",
        "summary_method": exec_row["method"] if exec_row else None,
        "lexicon": lexicon,
        "has_confidence_bands": has_bands,
        "band_scorers": (
            {"dictionary": lexicon, "transformer": transformer_scorer} if has_bands else None
        ),
        # Attention divergence over story clusters. The headline number above
        # compares moral vocabularies and comes out small on a real corpus,
        # because averages of hundreds of documents converge. This one compares
        # what the two diets spent the day on, and is where the reader's
        # experience of "different worlds" actually lives.
        "agenda": _agenda_payload(store, registry, pair),
        "blindspots": blindspots,
        # The reading unit: blindspot clusters grouped by subject and direction.
        # `blindspots` stays alongside it as the measured unit, for a surface
        # that wants to drill into one cluster.
        "blindspot_themes": blindspot_themes,
        "caveat": _caveat(lexicon, has_bands),
    }


def _fairness_payload(store: Datastore, registry: Registry, pair: ReferencePair) -> dict | None:
    """Equality-vs-proportionality shares per persona, or ``None`` if nothing split.

    Coverage travels with the shares so the dashboard can show how much of each
    persona the partition actually rests on, rather than presenting a ratio from a
    handful of documents as if it described the whole diet.
    """
    from compare.fairness import all_persona_fairness, gap

    profiles = all_persona_fairness(store, registry)
    split = {d: p for d, p in profiles.items() if p.docs_split}
    if not split:
        return None
    return {
        "diets": {
            diet_id: {
                "equality": p.equality,
                "proportionality": p.proportionality,
                "docs_split": p.docs_split,
                "docs_total": p.docs_total,
                "coverage": p.coverage,
                "thin": p.thin,
                "leans": p.leans,
            }
            for diet_id, p in profiles.items()
        },
        "gap": gap(profiles, pair),
    }


def _liberty_payload(store: Datastore, registry: Registry, pair: ReferencePair) -> dict | None:
    """Per-persona liberty engagement, or ``None`` if the tagger never ran.

    Reported separately from the radar composition on purpose — see
    ``compare/liberty.py`` for why partial coverage can't be folded into a
    composition without moving the headline number.
    """
    scorer = store.get_meta("liberty_scorer")
    if not scorer:
        return None
    from compare.liberty import all_persona_liberty, gap

    profiles = all_persona_liberty(store, registry, scorer)
    if not any(p.docs_scored for p in profiles.values()):
        return None
    return {
        "scorer": scorer,
        "diets": {
            diet_id: {
                "mean": p.mean,
                "salient_share": p.salient_share,
                "docs_scored": p.docs_scored,
                "docs_total": p.docs_total,
                "coverage": p.coverage,
                "thin": p.thin,
            }
            for diet_id, p in profiles.items()
        },
        "gap": gap(profiles, pair),
    }


def _agenda_payload(
    store: Datastore, registry: Registry, pair: ReferencePair
) -> dict | None:
    """Attention divergence, or ``None`` before anything has been clustered."""
    from compare.agenda import compare_agendas

    comparison = compare_agendas(store, registry, pair)
    return comparison.to_dict() if comparison else None


def _blindspot_payload(
    store: Datastore, members: dict[str, set[str]]
) -> tuple[list[dict], list[dict]]:
    """Blindspots and their themes (both empty until `python -m cluster run`).

    Built together from one read so the two cannot describe different clusters:
    the themes are a grouping *of* these blindspots, and re-deriving them from a
    second read is how a card ends up counting stories that are not in the list
    underneath it.
    """
    from cluster.blindspot import blindspots_from_store, themes_from_store

    spots = blindspots_from_store(store, members)
    serialized = [
        {
            "label": b.label,
            "dominant_diet": b.dominant_diet,
            "other_diet": b.other_diet,
            "counts": b.counts,
            "size": b.size,
            "cluster_size": b.cluster_size,
            "dominant_share": b.dominant_share,
            "representative_titles": b.representative_titles,
        }
        for b in spots
    ]
    return serialized, [t.to_dict() for t in themes_from_store(store, members, spots)]


def write_payload_dict(payload: dict, out: str | Path = DEFAULT_OUT) -> Path:
    """Serialize an already-built payload.

    Split out from ``write_payload`` so a caller that needs the payload for
    something else as well — the daily run renders it into an email too — can
    build it once and be sure both surfaces describe the same dict rather than
    two separately computed ones.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2)
    if out.suffix == ".js":
        out.write_text(f"window.PARALLAX_DATA = {body};\n", encoding="utf-8")
    else:
        out.write_text(body + "\n", encoding="utf-8")
    return out


def write_payload(
    store: Datastore,
    registry: Registry,
    pair: ReferencePair,
    out: str | Path = DEFAULT_OUT,
    history_limit: int | None = DEFAULT_SERIES_LIMIT,
) -> Path:
    return write_payload_dict(build_payload(store, registry, pair, history_limit), out)


def build_catalog(registry: Registry, pair: ReferencePair) -> dict:
    """The source catalog and every persona's weighting, corpus-free.

    Separate from the data payload because it describes the *model* rather than a
    day's measurements: it is meaningful with an empty datastore, which is what a
    persona-authoring surface needs.

    **No resolved URL is ever written here.** ``Source.url`` may hold a subscriber
    feed token pulled from the environment (see ``ingestion/config._resolve_url``),
    and this is the first time registry data leaves the process as a file. Only the
    *name* of the environment variable travels, never its value.
    """
    ingest_env = _ingest_env_names(registry)
    return {
        "version": registry.version,
        "reference": {"mine": pair.mine, "theirs": pair.theirs},
        "families": registry.families(),
        "strata": [
            {"id": s.id, "description": s.description} for s in registry.strata
        ],
        "sources": [
            {
                "id": s.id,
                "name": s.name,
                "medium": s.medium,
                "stratum": s.stratum_id,
                "role": s.role,
                "ingest_type": s.ingest_type,
                # Whether it is fetchable at all — not the URL itself.
                "has_url": bool(s.url),
                "url_env": ingest_env.get(s.id, ""),
                "domain": s.domain or "",
            }
            for s in registry.sources
        ],
        "personas": [
            {
                "id": p.id,
                "label": p.label,
                "short_label": p.display_label,
                "family": p.family,
                "description": p.description,
                "strata": dict(p.stratum_weights),
                "sources": dict(p.source_weights),
                "weights": registry.weights_for(p.id),
            }
            for p in registry.personas
        ],
    }


def _ingest_env_names(registry: Registry) -> dict[str, str]:
    """``{source_id: url_env}`` read back from the registry file on disk.

    The loader resolves ``url_env`` into ``Source.url`` and deliberately keeps no
    record of which variable it came from, so the name is re-read from the YAML
    rather than reconstructed — reconstructing it would mean carrying the secret
    alongside it.
    """
    import yaml

    from ingestion.config import DEFAULT_SOURCES

    try:
        raw = yaml.safe_load(Path(DEFAULT_SOURCES).read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    return {
        entry["id"]: (entry.get("ingest") or {}).get("url_env", "")
        for entry in (raw.get("catalog") or [])
        if (entry.get("ingest") or {}).get("url_env")
    }


def write_catalog(
    registry: Registry, pair: ReferencePair, out: str | Path = DEFAULT_CATALOG_OUT
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(build_catalog(registry, pair), indent=2)
    if out.suffix == ".js":
        out.write_text(f"window.PARALLAX_CATALOG = {body};\n", encoding="utf-8")
    else:
        out.write_text(body + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dashboard.export", description="Export dashboard data")
    parser.add_argument("--db", help="SQLite path (default from settings)")
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output .js (or .json) path")
    parser.add_argument("--catalog-out", default=str(DEFAULT_CATALOG_OUT),
                        help="output path for the persona/source catalog")
    parser.add_argument("--mine", help="persona id for your side of the reference pair")
    parser.add_argument("--theirs", help="persona id for the other side")
    parser.add_argument("--history-limit", type=int, default=DEFAULT_SERIES_LIMIT,
                        help="most recent N snapshots to serialize (0 = all)")
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    db = datastore_path(settings, args.db)
    registry = load_registry(settings=settings)
    pair = resolve(
        settings, args.mine, args.theirs,
        available=registry.persona_ids(), families=registry.families(),
    )
    store = Datastore(db)
    try:
        out = write_payload(store, registry, pair, args.out, args.history_limit or None)
        print(f"Wrote dashboard payload -> {out}")
        catalog = write_catalog(registry, pair, args.catalog_out)
        print(f"Wrote persona catalog   -> {catalog}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
