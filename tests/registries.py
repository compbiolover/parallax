"""Small synthetic registries for tests.

An explicit helper module rather than a ``conftest.py``: there is no pytest magic
here and nothing is injected: a test that wants a registry imports and calls this,
which keeps the wiring visible at the call site the way the rest of this suite
does.

Persona weights are resolved from the registry now instead of being baked onto
document rows, so almost every test that builds a store also needs a registry
saying who reads which source. That is the one piece of shared scaffolding the
persona model adds.
"""

from __future__ import annotations

from compare.reference import ReferencePair
from ingestion.config import Persona, Registry, Source, Stratum

# Sources here all sit in one stratum weighted 1.0, so a persona's per-source
# weight *is* its effective weight — which keeps the arithmetic in a test about
# something else from needing to be worked out.
STRATUM = "all"


def registry(**personas: dict[str, float]) -> Registry:
    """A registry from ``persona_id={source_id: weight}`` keyword arguments.

    Every named source is created in the catalog exactly once however many
    personas read it, which is the property most of these tests are checking.
    """
    source_ids = sorted({sid for weights in personas.values() for sid in weights})
    return Registry(
        version=3,
        strata=[Stratum(id=STRATUM, description="test stratum")],
        sources=[
            Source(
                id=sid, name=sid.replace("_", " ").title(), medium="news", role="",
                ingest_type="rss", url=f"https://example.test/{sid}", stratum_id=STRATUM,
            )
            for sid in source_ids
        ],
        personas=[
            Persona(
                id=persona_id,
                label=f"The {persona_id} diet",
                family="left" if persona_id.startswith(("self", "left")) else "right",
                stratum_weights={STRATUM: 1.0},
                source_weights=dict(weights),
            )
            for persona_id, weights in personas.items()
        ],
    )


def two_personas(mine: str = "self", theirs: str = "modeled_ce") -> tuple[Registry, ReferencePair]:
    """The common case: two personas over one source each, plus their pair."""
    return (
        registry(**{mine: {"s_mine": 1.0}, theirs: {"s_theirs": 1.0}}),
        ReferencePair(mine, theirs),
    )


def pair(mine: str = "self", theirs: str = "modeled_ce") -> ReferencePair:
    return ReferencePair(mine, theirs)


def members(reg: Registry, ref: ReferencePair) -> dict[str, set[str]]:
    """``{persona_id: {source_id}}`` for the pair, as the blindspot engine wants."""
    return {p: set(reg.weights_for(p)) for p in ref}
