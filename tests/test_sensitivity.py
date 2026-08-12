"""Weighting sensitivity: does the finding survive a different weighting?

`CLAUDE.md` §5 asks for this and `LIMITATIONS.md` says the weights are the
least-evidenced part of the tool. What these tests protect is the *interpretation*:
the report has to distinguish a divergence that merely moved from a per-foundation
sign that flipped, because the sign is the claim a reader takes away.
"""

from __future__ import annotations

import pytest

from compare.reference import ReferencePair
from compare.sensitivity import DEFAULT_FACTOR, analyze, format_report
from ingestion.config import Persona, Registry, Source, Stratum
from ingestion.datastore import Datastore

CARE = {"care": 0.7, "fairness": 0.1, "loyalty": 0.1, "authority": 0.05, "sanctity": 0.05}
BINDING = {"care": 0.05, "fairness": 0.05, "loyalty": 0.5, "authority": 0.2, "sanctity": 0.2}


def _registry(mine_strata: dict[str, float], theirs_strata: dict[str, float]) -> Registry:
    """Two personas, each reading one source per stratum it weights."""
    strata = sorted(set(mine_strata) | set(theirs_strata))
    sources = [
        Source(
            id=f"src_{s}",
            name=s,
            medium="news",
            role="",
            ingest_type="rss",
            url=f"https://example.test/{s}",
            stratum_id=s,
        )
        for s in strata
    ]
    return Registry(
        version=3,
        strata=[Stratum(id=s) for s in strata],
        sources=sources,
        personas=[
            Persona(
                id="self",
                label="Mine",
                family="left",
                stratum_weights=dict(mine_strata),
                source_weights={f"src_{s}": 1.0 for s in mine_strata},
            ),
            Persona(
                id="modeled_ce",
                label="Theirs",
                family="right",
                stratum_weights=dict(theirs_strata),
                source_weights={f"src_{s}": 1.0 for s in theirs_strata},
            ),
        ],
    )


def _store(docs: list[tuple[str, dict[str, float]]]) -> Datastore:
    """``[(source_id, foundations)]`` -> a store with one scored document each."""
    store = Datastore(":memory:")
    for i, (source_id, foundations) in enumerate(docs):
        doc_id = f"{source_id}-{i}"
        store.upsert_document(
            doc_id=doc_id,
            source_id=source_id,
            stratum_id=None,
            url=None,
            title="t",
            published_utc=None,
            fetched_utc="2026-08-10T00:00:00+00:00",
            word_count=400,
            minhash=None,
        )
        store.upsert_scores(
            document_id=doc_id,
            scorer="dictionary",
            foundations=foundations,
            sentiment=0.0,
            moral_word_ratio=0.2,
            matched_words=30,
        )
    return store


def test_a_weighting_that_changes_nothing_is_reported_as_stable():
    """Both of `self`'s strata carry the same coverage, so how much either counts
    cannot move the composition. Nothing to report is a real answer."""
    registry = _registry({"a": 0.5, "b": 0.5}, {"c": 1.0})
    store = _store([("src_a", CARE), ("src_b", CARE), ("src_c", BINDING)])

    report = analyze(store, registry, ReferencePair("self", "modeled_ce"))
    assert report is not None
    assert report.stable
    low, high = report.jsd_range
    assert high - low == pytest.approx(0.0, abs=1e-12)
    store.close()


def test_a_sign_flip_is_surfaced_as_the_finding_not_surviving():
    """The load-bearing case. `self` reads one care-heavy source and one
    binding-heavy one; which way it over-indexes on loyalty depends entirely on
    which of the two you decide it reads more of. That is a claim about the
    weighting, not about the corpus, and the report has to say so."""
    registry = _registry({"a": 1.0, "b": 1.0}, {"c": 1.0})
    store = _store(
        [
            ("src_a", CARE),
            ("src_b", BINDING),
            # The other side sits between them, so `self` can land on either side of it.
            (
                "src_c",
                {
                    "care": 0.35,
                    "fairness": 0.1,
                    "loyalty": 0.3,
                    "authority": 0.125,
                    "sanctity": 0.125,
                },
            ),
        ]
    )

    report = analyze(store, registry, ReferencePair("self", "modeled_ce"), factor=0.5)
    assert report is not None
    assert not report.stable
    flipped = report.flipped_foundations
    assert flipped, "expected at least one per-foundation sign to flip"
    # And the report names which perturbation did it, not just that something did.
    for causes in flipped.values():
        assert all("/" in cause for cause in causes)
    store.close()


def test_the_largest_mover_names_the_stratum_the_result_depends_on():
    registry = _registry({"a": 1.0, "b": 0.05}, {"c": 1.0})
    store = _store([("src_a", CARE), ("src_b", BINDING), ("src_c", BINDING)])

    report = analyze(store, registry, ReferencePair("self", "modeled_ce"))
    worst = report.worst[0]
    assert worst.persona == "self"
    assert worst.stratum in {"a", "b"}
    assert abs(worst.delta) > 0
    store.close()


def test_every_stratum_is_moved_in_both_directions():
    registry = _registry({"a": 1.0, "b": 1.0}, {"c": 1.0})
    store = _store([("src_a", CARE), ("src_b", BINDING), ("src_c", BINDING)])

    report = analyze(store, registry, ReferencePair("self", "modeled_ce"), factor=0.5)
    seen = {(p.persona, p.stratum, p.factor > 1) for p in report.perturbations}
    assert seen == {
        ("self", "a", True),
        ("self", "a", False),
        ("self", "b", True),
        ("self", "b", False),
        ("modeled_ce", "c", True),
        ("modeled_ce", "c", False),
    }
    store.close()


def test_an_empty_corpus_is_absence_not_agreement():
    """`None`, not a stable report: "the weighting does not matter" and "there is
    nothing to weight" are different statements, and a page full of zeros reads as
    the first."""
    registry = _registry({"a": 1.0}, {"c": 1.0})
    store = Datastore(":memory:")
    assert analyze(store, registry, ReferencePair("self", "modeled_ce")) is None
    assert "Not enough scored documents" in format_report(None)
    store.close()


def test_a_persona_outside_the_registry_is_not_analyzed():
    registry = _registry({"a": 1.0}, {"c": 1.0})
    store = _store([("src_a", CARE), ("src_c", BINDING)])
    assert analyze(store, registry, ReferencePair("self", "ghost")) is None
    store.close()


def test_perturbing_weights_never_writes_to_the_store_or_the_registry():
    """The whole point of resolving weights at aggregation: a hypothetical
    weighting costs a re-aggregation and leaves no trace."""
    registry = _registry({"a": 1.0, "b": 0.5}, {"c": 1.0})
    store = _store([("src_a", CARE), ("src_b", BINDING), ("src_c", BINDING)])
    before = registry.weights_for("self")

    analyze(store, registry, ReferencePair("self", "modeled_ce"))

    assert registry.weights_for("self") == before
    assert registry.persona("self").stratum_weights == {"a": 1.0, "b": 0.5}
    store.close()


def test_the_report_reads_as_prose_and_names_the_pair():
    registry = _registry({"a": 1.0, "b": 1.0}, {"c": 1.0})
    store = _store([("src_a", CARE), ("src_b", BINDING), ("src_c", BINDING)])
    text = format_report(analyze(store, registry, ReferencePair("self", "modeled_ce")))
    assert "self vs modeled_ce" in text
    assert "headline divergence" in text
    assert f"{DEFAULT_FACTOR:.0%}" in text
    store.close()


def test_the_report_serializes_flat():
    registry = _registry({"a": 1.0}, {"c": 1.0})
    store = _store([("src_a", CARE), ("src_c", BINDING)])
    d = analyze(store, registry, ReferencePair("self", "modeled_ce")).to_dict()
    assert d["pair"] == ["self", "modeled_ce"]
    assert set(d) == {
        "pair",
        "factor",
        "baseline_jsd",
        "jsd_range",
        "docs",
        "stable",
        "flipped",
        "perturbations",
    }
    store.close()


# -- the factor has to be a re-weighting, not a deletion ---------------------


@pytest.mark.parametrize("bad", [1.0, 1.5, 0.0, -0.5])
def test_a_factor_outside_zero_to_one_is_refused(bad):
    """At 1.0 the downward multiplier is 0, which deletes a stratum rather than
    re-weighting it. Above 1.0 it goes negative, and `aggregate_profile` skips
    non-positive weights — so the run silently becomes that same deletion while the
    report still says "down 150%". Both are wrong answers to "would a different
    weighting change the finding", so neither is reported."""
    registry = _registry({"a": 1.0}, {"c": 1.0})
    store = _store([("src_a", CARE), ("src_c", BINDING)])
    with pytest.raises(ValueError, match="between 0 and 1"):
        analyze(store, registry, ReferencePair("self", "modeled_ce"), factor=bad)
    store.close()


def test_a_bad_factor_on_the_command_line_is_a_usage_error():
    """argparse, so it arrives as usage rather than as a traceback."""
    import pytest as _pytest

    from compare.sensitivity import main

    with _pytest.raises(SystemExit) as excinfo:
        main(["--factor", "2"])
    assert excinfo.value.code == 2
