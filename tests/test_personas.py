"""Personas: weight resolution, the version-2 migration, and the local overlay.

The load-bearing test here is the migration one. Registry version 2 nested sources
inside diets, so a source belonged to exactly one diet and its weight could be
baked onto every document row at ingestion. Version 3 separates the catalog from
the personas that read it, which is what lets several personas share a source —
and means the weight has to be resolved per persona at aggregation instead.

That refactor is only safe if it changes no number. The divergence series has been
accumulating one row a day since the project started, and every one of those rows
is a comparison between `self` and `modeled_ce` under version 2's weights.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from ingestion.config import load_registry

# Every source's effective weight under registry version 2, generated from the
# v2 file itself (`stratum_weight * source weight`) rather than retyped. Version 3
# has to reproduce these up to one positive scale factor per persona: aggregation
# divides by the weight sum, so a uniform rescale of one persona cancels exactly
# and only the ratios within a persona are observable.
V2_WEIGHTS = {
    "self": {
        "self_nyt_home": 0.36,
        "self_wapo_national": 0.324,
        "self_guardian_us": 0.288,
        "self_politico_picks": 0.252,
        "self_ap_topnews": 0.252,
        "self_npr_news": 0.31,
        "self_npr_politics": 0.217,
        "self_pbs_newshour": 0.217,
        "self_bbc_world": 0.248,
        "self_atlantic": 0.23,
        "self_vox": 0.18400000000000002,
        "self_goodgood": 0.115,
        "self_npr_upfirst": 0.1,
        "self_pbs_newshour_show": 0.06999999999999999,
        "self_nyt_daily": 0.09000000000000001,
        "self_radio_atlantic": 0.06999999999999999,
        "self_economist_intelligence": 0.06999999999999999,
        "self_jacobin_radio": 0.05,
        "self_makingsense_harris": 0.04000000000000001,
        "self_yt_alex_oconnor": 0.06999999999999999,
        "self_yt_paulogia": 0.06,
        "self_yt_frontline": 0.05,
    },
    "modeled_ce": {
        "ce_foxnews_latest": 0.3,
        "ce_foxnews_opinion": 0.18,
        "ce_dailywire_mattwalsh": 0.2,
        "ce_dailywire_news": 0.2,
        "ce_thefp": 0.13999999999999999,
        "ce_christianitytoday": 0.2,
        "ce_worldnews": 0.16000000000000003,
        "ce_relatable_stuckey": 0.1,
    },
}


def _normalized(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


# -- the migration ----------------------------------------------------------


@pytest.mark.parametrize("persona_id", sorted(V2_WEIGHTS))
def test_the_migrated_personas_reproduce_the_version_2_diet_weights(persona_id):
    """Every recorded snapshot is a comparison under version 2's weights, so a
    migration that moved any source's weight relative to its neighbours would
    silently break the continuity of the whole divergence series."""
    resolved = load_registry().weights_for(persona_id)
    expected = V2_WEIGHTS[persona_id]
    assert set(resolved) == set(expected)
    assert _normalized(resolved) == pytest.approx(_normalized(expected), rel=1e-12)


def test_splitting_the_podcast_stratum_did_not_halve_its_sources():
    """`podcasts_youtube` was one stratum weighted 0.10 in *both* diets, which a
    flat catalog cannot express. Splitting it into `podcasts` and `youtube` had to
    give each the same 0.10 — two weights summing to 0.10 would have quietly
    halved every audio and video source."""
    registry = load_registry()
    self_persona = registry.persona("self")
    assert self_persona.stratum_weights["podcasts"] == 0.10
    assert self_persona.stratum_weights["youtube"] == 0.10
    # And the effect: Up First (a podcast) and Alex O'Connor (video) keep the
    # ratio to each other and to a newspaper that they had under version 2.
    weights = registry.weights_for("self")
    assert weights["self_npr_upfirst"] == pytest.approx(0.10)
    assert weights["self_yt_alex_oconnor"] == pytest.approx(0.07)


def test_a_version_2_registry_is_refused_with_the_reason(tmp_path):
    """Loading it as if the schema matched would produce a registry with no
    personas and no sources, and the run would report an empty corpus."""
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"version": 2, "diets": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="version 2"):
        load_registry(path)


# -- weight resolution ------------------------------------------------------


def _registry_yaml(**overrides) -> dict:
    base = {
        "version": 3,
        "strata": [{"id": "papers"}, {"id": "audio"}],
        "catalog": [
            {"id": "times", "name": "The Times", "medium": "news", "stratum": "papers",
             "ingest": {"type": "rss", "url": "https://t.test/f"}, "rationale": "r"},
            {"id": "show", "name": "A Show", "medium": "podcast", "stratum": "audio",
             "ingest": {"type": "podcast_rss", "url": "https://s.test/f"}, "rationale": "r"},
        ],
        "personas": [
            {"id": "reader", "label": "A reader", "family": "left",
             "description": "d", "strata": {"papers": 0.8, "audio": 0.2},
             "sources": {"times": 1.0, "show": 0.5}},
        ],
    }
    base.update(overrides)
    return base


def _write(tmp_path, data) -> str:
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_a_weight_is_the_stratum_weight_times_the_source_weight(tmp_path):
    registry = load_registry(_write(tmp_path, _registry_yaml()))
    weights = registry.weights_for("reader")
    assert weights == {"times": pytest.approx(0.8), "show": pytest.approx(0.1)}


def test_a_source_two_personas_read_appears_in_the_catalog_once(tmp_path):
    """The whole point of separating the catalog from the personas: a shared
    source is fetched, extracted, scored, embedded and tagged once however many
    personas read it, so adding a persona costs nothing at runtime."""
    data = _registry_yaml()
    data["personas"].append({
        "id": "listener", "label": "A listener", "family": "left", "description": "d",
        "strata": {"papers": 0.1, "audio": 0.9}, "sources": {"times": 1.0, "show": 1.0},
    })
    registry = load_registry(_write(tmp_path, data))

    assert [s.id for s in registry.ingestable(("rss", "podcast_rss"))] == ["times", "show"]
    # Same sources, different weights — which is what a persona is.
    assert registry.weights_for("reader")["show"] == pytest.approx(0.1)
    assert registry.weights_for("listener")["show"] == pytest.approx(0.9)


def test_membership_is_opt_in_so_a_shared_stratum_is_not_a_shared_diet(tmp_path):
    """Strata are global, so `audio` holds every persona's podcasts. Inheriting a
    stratum's sources would hand each persona all the others' listening; the
    alternative to opt-in is an exclusion list that has to grow whenever anyone
    adds a source."""
    data = _registry_yaml()
    data["catalog"].append(
        {"id": "other_show", "name": "Another Show", "medium": "podcast",
         "stratum": "audio", "ingest": {"type": "podcast_rss", "url": "https://o.test/f"},
         "rationale": "r"},
    )
    data["personas"].append({
        "id": "listener", "label": "A listener", "family": "right", "description": "d",
        "strata": {"audio": 1.0}, "sources": {"other_show": 1.0},
    })
    registry = load_registry(_write(tmp_path, data))

    assert set(registry.weights_for("reader")) == {"times", "show"}
    assert set(registry.weights_for("listener")) == {"other_show"}


def test_a_persona_listing_an_unknown_source_fails_at_load(tmp_path):
    """A typo'd source id would silently drop that source from the persona, and
    the profile would come out slightly wrong with nothing to point at."""
    data = _registry_yaml()
    data["personas"][0]["sources"]["tiems"] = 1.0
    with pytest.raises(ValueError, match="tiems"):
        load_registry(_write(tmp_path, data))


def test_a_source_whose_stratum_carries_no_weight_fails_at_load(tmp_path):
    """Zero-weighting it silently would drop a source the persona plainly means to
    consume — the same failure as a typo, arrived at from the other direction."""
    data = _registry_yaml()
    data["personas"][0]["strata"] = {"papers": 1.0}      # `audio` dropped
    with pytest.raises(ValueError, match="audio"):
        load_registry(_write(tmp_path, data))


def test_a_source_in_an_undeclared_stratum_fails_at_load(tmp_path):
    data = _registry_yaml()
    data["catalog"][0]["stratum"] = "newspapers"          # declared as `papers`
    with pytest.raises(ValueError, match="newspapers"):
        load_registry(_write(tmp_path, data))


def test_duplicate_ids_fail_at_load(tmp_path):
    data = _registry_yaml()
    data["catalog"].append(dict(data["catalog"][0]))
    with pytest.raises(ValueError, match="duplicate source id"):
        load_registry(_write(tmp_path, data))


# -- the local overlay ------------------------------------------------------
#
# A user's own diet describes a real person's real media consumption, which is the
# one thing CLAUDE.md §0 promises this project does not collect. It lives in a
# gitignored file rather than the public registry, merged over it at load.


def test_a_local_overlay_adds_a_persona_the_public_registry_does_not_have(tmp_path):
    base = _write(tmp_path, _registry_yaml())
    overlay = tmp_path / "local.yaml"
    overlay.write_text(yaml.safe_dump({"personas": [{
        "id": "me", "label": "My actual diet", "family": "left", "description": "d",
        "strata": {"papers": 1.0}, "sources": {"times": 1.0},
    }]}), encoding="utf-8")

    registry = load_registry(base, overlay=overlay)
    assert registry.persona_ids() == ["reader", "me"]
    assert registry.weights_for("me") == {"times": pytest.approx(1.0)}


def test_a_local_persona_replaces_a_public_one_of_the_same_id_wholesale(tmp_path):
    """Field-by-field merging would leave a persona carrying some weights from the
    public file and some from yours, which is not a weighting anyone could reason
    about."""
    base = _write(tmp_path, _registry_yaml())
    overlay = tmp_path / "local.yaml"
    overlay.write_text(yaml.safe_dump({"personas": [{
        "id": "reader", "label": "My tuning", "family": "left", "description": "d",
        "strata": {"audio": 1.0}, "sources": {"show": 1.0},
    }]}), encoding="utf-8")

    registry = load_registry(base, overlay=overlay)
    assert registry.persona_ids() == ["reader"]
    assert set(registry.weights_for("reader")) == {"show"}      # not {"times", "show"}
    assert registry.persona("reader").label == "My tuning"


def test_a_local_overlay_can_add_sources_and_strata(tmp_path):
    base = _write(tmp_path, _registry_yaml())
    overlay = tmp_path / "local.yaml"
    overlay.write_text(yaml.safe_dump({
        "strata": [{"id": "newsletters"}],
        "catalog": [{"id": "letter", "name": "A Letter", "medium": "newsletter",
                     "stratum": "newsletters",
                     "ingest": {"type": "rss", "url": "https://l.test/f"},
                     "rationale": "r"}],
        "personas": [{"id": "me", "label": "Me", "family": "left", "description": "d",
                      "strata": {"newsletters": 1.0}, "sources": {"letter": 1.0}}],
    }), encoding="utf-8")

    registry = load_registry(base, overlay=overlay)
    assert registry.source("letter") is not None
    assert registry.weights_for("me") == {"letter": pytest.approx(1.0)}


def test_no_overlay_is_not_an_error(tmp_path):
    """Most installations have none, the same way `config/settings.yaml` is absent
    until you copy the example."""
    registry = load_registry(_write(tmp_path, _registry_yaml()))
    assert registry.persona_ids() == ["reader"]


def test_a_named_overlay_that_is_missing_is_an_error(tmp_path):
    """Silently ignoring it would run the whole pipeline against the public
    registry while the user believed their own diet was in the comparison."""
    base = _write(tmp_path, _registry_yaml())
    with pytest.raises(FileNotFoundError):
        load_registry(base, overlay=tmp_path / "not-there.yaml")


def test_the_overlay_is_named_in_the_log_but_never_its_contents(tmp_path, caplog):
    """The file records what one person reads. A log is a file too."""
    base = _write(tmp_path, _registry_yaml())
    overlay = tmp_path / "local.yaml"
    overlay.write_text(yaml.safe_dump({"personas": [{
        "id": "me", "label": "SENSITIVE LABEL", "family": "left", "description": "d",
        "strata": {"papers": 1.0}, "sources": {"times": 1.0},
    }]}), encoding="utf-8")

    with caplog.at_level(logging.DEBUG):
        load_registry(base, overlay=overlay)
    assert "local.yaml" in caplog.text
    assert "SENSITIVE LABEL" not in caplog.text


def test_the_overlay_path_comes_from_settings_when_not_passed(tmp_path):
    base = _write(tmp_path, _registry_yaml())
    overlay = tmp_path / "mine.yaml"
    overlay.write_text(yaml.safe_dump({"personas": [{
        "id": "me", "label": "Me", "family": "left", "description": "d",
        "strata": {"papers": 1.0}, "sources": {"times": 1.0},
    }]}), encoding="utf-8")

    registry = load_registry(
        base, settings={"sources": {"local_registry": str(overlay)}}
    )
    assert "me" in registry.persona_ids()


def test_a_configured_overlay_that_does_not_exist_yet_is_not_an_error(tmp_path):
    """`settings.example.yaml` ships `sources.local_registry` pointing at a file
    you have not written, exactly as it does for `config/settings.yaml`. Treating
    that as fatal made the shipped example break every default run."""
    base = _write(tmp_path, _registry_yaml())
    registry = load_registry(
        base, settings={"sources": {"local_registry": str(tmp_path / "absent.yaml")}}
    )
    assert registry.persona_ids() == ["reader"]


def test_the_shipped_example_settings_load_the_registry():
    """The example is the fallback when `config/settings.yaml` is absent, so a
    path in it that cannot be resolved breaks a fresh checkout on the first run."""
    from ingestion.config import load_settings

    registry = load_registry(settings=load_settings())
    assert "self" in registry.persona_ids()
