"""Dashboard exporter: payload shape and file writing."""

from __future__ import annotations

import json

from dashboard.export import build_payload, write_payload
from ingestion.datastore import Datastore
from scoring.foundations import CLASSIC_FOUNDATIONS

from .registries import pair, registry


def _reg(*personas):
    """A registry over `src_<persona>` sources, one per persona."""
    names = personas or ("self", "modeled_ce")
    return registry(**{name: {f"src_{name}": 1.0} for name in names})


def _store_with_two_diets():
    store = Datastore(":memory:")
    for diet, care in [("self", 0.3), ("modeled_ce", 0.1)]:
        did = f"{diet}-doc"
        store.upsert_document(
            doc_id=did, source_id=f"src_{diet}", stratum_id=None, url=None,
            title="t", published_utc=None, fetched_utc="2026-07-23T00:00:00+00:00",
            word_count=90, minhash=None,
        )
        store.upsert_scores(
            document_id=did, scorer="dictionary",
            foundations={"care": care, "fairness": 0.1, "loyalty": 0.2,
                         "authority": 0.1, "sanctity": 0.1},
            sentiment=0.0, moral_word_ratio=0.2, matched_words=18,
        )
    store.upsert_summary(scope="self", generated_utc="t", model="m",
                         method="deterministic", text="self summary")
    store.upsert_summary(scope="executive", generated_utc="t", model="m",
                         method="deterministic", text="exec summary")
    return store


def test_payload_shape():
    store = _store_with_two_diets()
    p = build_payload(store, _reg(), pair())
    assert p["foundations"] == list(CLASSIC_FOUNDATIONS)
    assert len(p["diets"]) == 2
    for d in p["diets"]:
        assert set(d["profile"]) == set(CLASSIC_FOUNDATIONS)
        assert abs(sum(d["profile"].values()) - 1.0) < 1e-6
    # Oriented mine-first rather than alphabetically, so a positive log-ratio
    # means "my diet over-indexes" as CLAUDE.md §3(5) specifies.
    assert p["comparison"]["pair"] == ["self", "modeled_ce"]
    assert p["comparison"]["orientation"] == "mine_first"
    assert p["reference"] == {"mine": "self", "theirs": "modeled_ce"}
    assert 0.0 <= p["comparison"]["jsd"] <= 1.0
    assert p["executive_summary"] == "exec summary"
    # No lexicon recorded -> treated as demo, strong caveat.
    assert p["lexicon"] is None
    assert "DEMO lexicon" in p["caveat"]
    store.close()


def test_the_payload_carries_both_of_a_diets_names():
    """`label` reads as a noun phrase inside a sentence, `short_label` fits a
    legend. Emitting the id as the label put "modeled_ce" on every surface."""
    store = _store_with_two_diets()
    store.set_diet_label("modeled_ce", "Modeled conservative-evangelical diet",
                         "The modeled diet")
    by_id = {d["id"]: d for d in build_payload(store, _reg(), pair())["diets"]}
    assert by_id["modeled_ce"]["label"] == "Modeled conservative-evangelical diet"
    assert by_id["modeled_ce"]["short_label"] == "The modeled diet"
    # A persona with no label recorded in the store falls back to the registry's,
    # and only to the machine id if that is missing too. The id was never fit to
    # print at a reader.
    assert by_id["self"]["label"] == "The self diet"
    assert by_id["self"]["short_label"] == "The self diet"
    store.close()


def test_a_label_with_no_short_form_supplies_itself():
    store = _store_with_two_diets()
    store.set_diet_label("self", "My diet")
    by_id = {d["id"]: d for d in build_payload(store, _reg(), pair())["diets"]}
    assert by_id["self"]["short_label"] == "My diet"
    store.close()


def test_caveat_softens_for_real_lexicon():
    store = _store_with_two_diets()
    store.set_meta("lexicon", "eMFD (emfd_scoring.csv)")
    p = build_payload(store, _reg(), pair())
    assert p["lexicon"] == "eMFD (emfd_scoring.csv)"
    assert "DEMO" not in p["caveat"]
    assert "eMFD (emfd_scoring.csv)" in p["caveat"]
    store.close()


def test_write_js_payload(tmp_path):
    store = _store_with_two_diets()
    out = write_payload(store, _reg(), pair(), tmp_path / "latest.js")
    text = out.read_text()
    assert text.startswith("window.PARALLAX_DATA = ")
    data = json.loads(text[len("window.PARALLAX_DATA = "):].rstrip().rstrip(";"))
    assert data["diets"]
    store.close()


def test_blindspots_in_payload():
    store = _store_with_two_diets()
    # add two more modeled_ce docs so a cluster can be one-sided
    for i in range(3):
        did = f"ce-extra-{i}"
        store.upsert_document(doc_id=did, source_id="src_modeled_ce", stratum_id=None,
            url=None, title=f"faith story {i}", published_utc=None,
            fetched_utc="2026-07-23T00:00:00+00:00", word_count=80, minhash=None)
        store.upsert_scores(document_id=did, scorer="dictionary", foundations={"sanctity": 0.5},
                            sentiment=0.0, moral_word_ratio=0.1, matched_words=5)
    # seed a persisted clustering directly (cluster 5 = modeled_ce-only)
    store.replace_clustering(
        clusters=[(5, "faith · story", 3)],
        assignments=[(f"ce-extra-{i}", 5) for i in range(3)],
    )
    p = build_payload(store, _reg(), pair())
    assert p["blindspots"], "expected a blindspot"
    b = p["blindspots"][0]
    assert b["dominant_diet"] == "modeled_ce"
    assert b["other_diet"] == "self"
    assert b["label"] == "faith · story"
    assert b["representative_titles"]
    store.close()


def test_blindspot_themes_travel_with_the_clusters():
    """The payload carries both units: clusters are what the asymmetry is
    measured on, themes are what it is read as."""
    store = _store_with_two_diets()
    store.set_source_label("ct", "Christianity Today")
    store.set_source_label("cp", "The Christian Post")
    sources = ["ct", "cp", "ct"]
    for i in range(3):
        did = f"ce-extra-{i}"
        store.upsert_document(doc_id=did, source_id=sources[i], stratum_id=None,
            url=f"https://example.org/pastor-{i}",
            title=f"Church congregation welcomes a new pastor number {i}",
            published_utc=None, fetched_utc="2026-07-23T00:00:00+00:00",
            word_count=80, minhash=None)
        store.upsert_scores(document_id=did, scorer="dictionary",
                            foundations={"sanctity": 0.5}, sentiment=0.0,
                            moral_word_ratio=0.1, matched_words=5)
    store.replace_clustering(clusters=[(5, "faith · story", 3)],
                             assignments=[(f"ce-extra-{i}", 5) for i in range(3)])
    store.replace_blindspot_themes(
        [(f"ce-extra-{i}", "faith", "Faith & the church", "taxonomy")
         for i in range(3)]
    )

    # `modeled_ce` reads the two mastheads as well as its own seed source.
    reg = registry(
        self={"src_self": 1.0},
        modeled_ce={"src_modeled_ce": 1.0, "ct": 1.0, "cp": 1.0},
    )
    themes = build_payload(store, reg, pair())["blindspot_themes"]
    assert [t["title"] for t in themes] == ["Faith & the church"]
    assert themes[0]["dominant_diet"] == "modeled_ce"
    # One cluster, so one story — and the three articles are its coverage, not
    # three separate stories.
    assert themes[0]["story_count"] == 1
    assert themes[0]["article_count"] == 3
    story = themes[0]["stories"][0]
    assert story["articles"] == 3
    # The mastheads that carried it, de-duplicated, each with a link.
    assert [o["label"] for o in story["outlets"]] == [
        "Christianity Today", "The Christian Post"
    ]
    assert story["outlets"][0]["url"] == "https://example.org/pastor-0"
    store.close()


def test_the_agenda_divergence_travels_with_the_payload():
    """The second number the page prints. `None` before anything is clustered,
    because "no clustering" is not "identical agendas"."""
    store = _store_with_two_diets()
    assert build_payload(store, _reg(), pair())["agenda"] is None

    for diet in ("self", "modeled_ce"):
        for i in range(3):
            did = f"{diet}-agenda-{i}"
            store.upsert_document(doc_id=did, source_id=f"src_{diet}",
                stratum_id=None, url=None, title=f"{diet} story {i}",
                published_utc=None, fetched_utc="2026-07-23T00:00:00+00:00",
                word_count=80, minhash=None)
            store.upsert_scores(document_id=did, scorer="dictionary",
                                foundations={"care": 0.5}, sentiment=0.0,
                                moral_word_ratio=0.1, matched_words=5)
    store.replace_clustering(
        clusters=[(7, "mine", 3), (8, "theirs", 3)],
        assignments=[(f"self-agenda-{i}", 7) for i in range(3)]
                    + [(f"modeled_ce-agenda-{i}", 8) for i in range(3)],
    )
    agenda = build_payload(store, _reg(), pair())["agenda"]
    assert agenda["divergence"] == 1.0      # wholly different stories
    assert agenda["shared_stories"] == 0
    assert agenda["thin"] is True           # three articles each
    store.close()


def test_no_blindspots_when_unclustered():
    store = _store_with_two_diets()
    payload = build_payload(store, _reg(), pair())
    assert payload["blindspots"] == []
    assert payload["blindspot_themes"] == []
    store.close()


def test_single_diet_has_no_comparison():
    store = Datastore(":memory:")
    store.upsert_document(
        doc_id="d", diet_id="self", source_id="s", stratum_id=None, url=None,
        title="t", published_utc=None, fetched_utc="2026-07-23T00:00:00+00:00",
        word_count=90, minhash=None,
    )
    store.upsert_scores(document_id="d", scorer="dictionary",
                        foundations={"care": 0.5}, sentiment=0.0,
                        moral_word_ratio=0.1, matched_words=5)
    assert build_payload(store, _reg(), pair())["comparison"] is None
    store.close()


# -- snapshot history in the payload ---------------------------------------

def test_history_is_empty_until_a_snapshot_is_recorded():
    store = _store_with_two_diets()
    p = build_payload(store, _reg(), pair())
    assert p["history"] == []
    assert p["history_window_days"] is None
    store.close()


def test_exporting_never_records_a_snapshot(tmp_path):
    """Export is a read. Recording is the daily runner's `snapshot` step, so
    rebuilding the payload can't invent history."""
    store = _store_with_two_diets()
    build_payload(store, _reg(), pair())
    write_payload(store, _reg(), pair(), tmp_path / "latest.js")
    assert store.snapshot_count() == 0
    store.close()


def test_history_carries_dated_points_and_the_window():
    from compare.history import record_snapshot

    store = _store_with_two_diets()
    for day in ("2026-07-23", "2026-07-24"):
        record_snapshot(store, _reg(), pair(), day, window_days=5)
    p = build_payload(store, _reg(), pair())
    assert [s["date"] for s in p["history"]] == ["2026-07-23", "2026-07-24"]
    assert p["history_window_days"] == 5
    # The series' all-time basis is the headline number, up to the 6-decimal
    # rounding snapshots are stored at.
    assert p["history"][-1]["jsd_cumulative"] == round(p["comparison"]["jsd"], 6)
    store.close()


def test_history_limit_caps_what_is_serialized():
    from compare.history import record_snapshot

    store = _store_with_two_diets()
    for day in ("2026-07-21", "2026-07-22", "2026-07-23"):
        record_snapshot(store, _reg(), pair(), day)
    assert len(build_payload(store, _reg(), pair(), history_limit=2)["history"]) == 2
    store.close()


# -- fairness split in the payload -----------------------------------------

def test_fairness_split_absent_when_nothing_was_partitioned():
    store = _store_with_two_diets()
    assert build_payload(store, _reg(), pair())["fairness_split"] is None
    store.close()


def test_fairness_split_carries_shares_and_coverage():
    store = _store_with_two_diets()
    store.upsert_scores(
        document_id="self-doc", scorer="dictionary",
        foundations={"care": 0.3, "fairness": 0.2, "loyalty": 0.2,
                     "authority": 0.1, "sanctity": 0.1},
        sentiment=0.0, moral_word_ratio=0.2, matched_words=18,
        equality=0.15, proportionality=0.05,
    )
    fs = build_payload(store, _reg(), pair())["fairness_split"]
    assert fs["diets"]["self"]["leans"] == "equality"
    assert fs["diets"]["self"]["docs_split"] == 1
    assert fs["diets"]["self"]["coverage"] == 1.0
    # The other diet was never split — reported, but with nothing behind it.
    assert fs["diets"]["modeled_ce"]["docs_split"] == 0
    assert fs["diets"]["modeled_ce"]["thin"]
    store.close()


# -- liberty in the payload ------------------------------------------------

def test_liberty_absent_until_the_tagger_runs():
    store = _store_with_two_diets()
    assert build_payload(store, _reg(), pair())["liberty"] is None
    store.close()


def test_liberty_reports_mean_and_coverage():
    store = _store_with_two_diets()
    scorer = "claude-liberty/claude-sonnet-5"
    store.upsert_scores(
        document_id="self-doc", scorer=scorer, foundations={},
        sentiment=0.0, moral_word_ratio=0.0, matched_words=0, liberty=0.75,
    )
    store.set_meta("liberty_scorer", scorer)
    lib = build_payload(store, _reg(), pair())["liberty"]
    assert lib["scorer"] == scorer
    assert lib["diets"]["self"]["mean"] == 0.75
    assert lib["diets"]["self"]["docs_scored"] == 1
    # The other diet exists but was never tagged — reported, with nothing behind it.
    assert lib["diets"]["modeled_ce"]["docs_scored"] == 0
    assert lib["diets"]["modeled_ce"]["thin"]


def test_liberty_does_not_change_the_headline_composition():
    """Partial-coverage liberty must not move the five-way profile or the JSD
    that compare/history.py has been recording."""
    store = _store_with_two_diets()
    before = build_payload(store, _reg(), pair())
    store.upsert_scores(
        document_id="self-doc", scorer="claude-liberty/claude-sonnet-5", foundations={},
        sentiment=0.0, moral_word_ratio=0.0, matched_words=0, liberty=0.9,
    )
    store.set_meta("liberty_scorer", "claude-liberty/claude-sonnet-5")
    after = build_payload(store, _reg(), pair())
    assert after["comparison"]["jsd"] == before["comparison"]["jsd"]
    assert [d["profile"] for d in after["diets"]] == [d["profile"] for d in before["diets"]]
    assert after["foundations"] == before["foundations"]   # still the classic five
    store.close()


# -- N personas, the matrix, and the catalog --------------------------------


def test_the_comparison_is_the_named_pair_not_the_first_two_ids():
    """With more than two personas, `sorted(ids)[:2]` would compare whichever two
    happened to sort first — so adding a persona could change the headline number
    without anyone touching a weight."""
    store = _store_with_two_diets()
    reg = registry(
        aardvark={"src_aardvark": 1.0},
        self={"src_self": 1.0},
        modeled_ce={"src_modeled_ce": 1.0},
    )
    p = build_payload(store, reg, pair())
    assert p["comparison"]["pair"] == ["self", "modeled_ce"]
    store.close()


def test_the_reference_pair_leads_the_persona_list():
    """Ordering used to be `ORDER BY diet_id`, so which persona came first was an
    alphabetical accident — and every surface that coloured by list position
    inherited it."""
    store = _store_with_two_diets()
    reg = registry(
        aardvark={"src_aardvark": 1.0},
        self={"src_self": 1.0},
        modeled_ce={"src_modeled_ce": 1.0},
    )
    p = build_payload(store, reg, pair())
    assert [d["id"] for d in p["diets"]][:2] == ["self", "modeled_ce"]
    assert [d["role"] for d in p["diets"]][:2] == ["mine", "theirs"]
    store.close()


def test_every_persona_carries_its_family_so_colour_can_mean_a_side():
    store = _store_with_two_diets()
    by_id = {d["id"]: d for d in build_payload(store, _reg(), pair())["diets"]}
    assert by_id["self"]["family"] == "left"
    assert by_id["modeled_ce"]["family"] == "right"
    store.close()


def test_the_divergence_matrix_is_symmetric_with_a_zero_diagonal():
    store = _store_with_two_diets()
    p = build_payload(store, _reg(), pair())
    m = p["matrix"]
    n = len(m["personas"])
    assert n == 2
    for i in range(n):
        assert m["jsd"][i][i] == 0.0
        for j in range(n):
            assert m["jsd"][i][j] == m["jsd"][j][i]
    store.close()


def test_the_overlap_matrix_ships_beside_the_divergence_one():
    """Personas are weight profiles over one shared catalog, so two that read
    mostly the same outlets have a small divergence *by construction*. A grid of
    small numbers alone reads as "the personas are interchangeable", which is the
    opposite of what it shows."""
    store = _store_with_two_diets()
    reg = registry(
        self={"src_self": 1.0},
        modeled_ce={"src_modeled_ce": 1.0},
    )
    overlap = build_payload(store, reg, pair())["overlap"]
    order = overlap["personas"]
    i, j = order.index("self"), order.index("modeled_ce")
    assert overlap["cosine"][i][i] == 1.0        # a persona overlaps itself entirely
    assert overlap["cosine"][i][j] == 0.0        # these two share no source
    store.close()


def test_two_personas_sharing_a_source_overlap_more_than_none():
    store = _store_with_two_diets()
    reg = registry(
        self={"src_self": 1.0, "src_modeled_ce": 1.0},
        modeled_ce={"src_modeled_ce": 1.0},
    )
    overlap = build_payload(store, reg, pair())["overlap"]
    order = overlap["personas"]
    i, j = order.index("self"), order.index("modeled_ce")
    assert overlap["cosine"][i][j] > 0.0
    store.close()


def test_the_catalog_lists_every_source_once_with_its_personas_weights():
    from dashboard.export import build_catalog

    reg = registry(
        self={"src_shared": 1.0, "src_self": 0.5},
        modeled_ce={"src_shared": 0.25},
    )
    catalog = build_catalog(reg, pair())
    assert sorted(s["id"] for s in catalog["sources"]) == ["src_self", "src_shared"]
    by_id = {p["id"]: p for p in catalog["personas"]}
    assert by_id["self"]["weights"]["src_shared"] == 1.0
    assert by_id["modeled_ce"]["weights"]["src_shared"] == 0.25
    assert catalog["reference"] == {"mine": "self", "theirs": "modeled_ce"}


def test_the_catalog_never_carries_a_resolved_subscriber_url(monkeypatch, tmp_path):
    """A subscriber feed URL *is* the credential, and this is the first time
    registry data leaves the process as a file. Only the name of the environment
    variable travels, never its value."""
    import yaml

    from dashboard.export import build_catalog
    from ingestion.config import load_registry

    secret = "https://example.com/private?id=SECRETTOKEN"
    monkeypatch.setenv("PARALLAX_TEST_FEED", secret)
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({
        "version": 3,
        "strata": [{"id": "audio"}],
        "catalog": [{
            "id": "show", "name": "A Show", "medium": "podcast", "stratum": "audio",
            "ingest": {"type": "podcast_rss", "url_env": "PARALLAX_TEST_FEED"},
            "rationale": "r",
        }],
        "personas": [{
            "id": "self", "label": "Me", "family": "left", "description": "d",
            "strata": {"audio": 1.0}, "sources": {"show": 1.0},
        }],
    }), encoding="utf-8")

    reg = load_registry(path)
    assert reg.source("show").url == secret          # resolved in memory
    body = json.dumps(build_catalog(reg, pair()))
    assert "SECRETTOKEN" not in body
    assert secret not in body


def test_the_catalog_writes_a_window_assignment_so_file_urls_work(tmp_path):
    """Same reason `latest.js` is a `.js`: a static page opened from disk cannot
    `fetch` a sibling JSON."""
    from dashboard.export import write_catalog

    out = write_catalog(_reg(), pair(), tmp_path / "catalog.js")
    assert out.read_text(encoding="utf-8").startswith("window.PARALLAX_CATALOG = {")
