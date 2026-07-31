"""Title hygiene and blindspot themes — the reading layer over the clusters.

The failure these guard against is not a crash. It is a section of the brief
that is technically correct and unreadable: headlines carrying their outlet's
name and GDELT's tokenization, clusters titled in c-TF-IDF terms, and one card
per cluster for as many clusters as the day produced.
"""

from __future__ import annotations

from cluster.blindspot import Blindspot, run_clustering, themes_from_store
from cluster.themes import (
    OTHER_KEY,
    Theme,
    ThemeAssignment,
    assign_themes,
    claude_assignments,
    group_blindspots,
    taxonomy_theme,
)
from cluster.titles import clean_title, clean_titles, is_boilerplate
from ingestion.datastore import Datastore

# -- titles -----------------------------------------------------------------


def test_gdelt_tokenization_is_put_back_together():
    assert clean_title("U . S . Senate rejects the measure") == (
        "U.S. Senate rejects the measure")
    assert clean_title("Soft exosuit cuts walking energy use by 14 %") == (
        "Soft exosuit cuts walking energy use by 14%")
    assert clean_title("Where Do Mission Hospitals Fit in the 21st Century ?") == (
        "Where Do Mission Hospitals Fit in the 21st Century?")
    assert clean_title("California mom welcomes baby after little girl couldn ' t wait") == (
        "California mom welcomes baby after little girl couldn't wait")


def test_the_outlet_stamp_comes_off():
    assert clean_title("Iran Shutters Country Last Presbyterian Church - Christianity Today") == (
        "Iran Shutters Country Last Presbyterian Church")


def test_only_one_stamp_comes_off_and_the_compound_survives():
    """Two passes look harmless and are not: a tokenized hyphenated compound
    offers a second false stamp, and taking it leaves a headline about nothing.
    """
    assert clean_title("How to Revitalize a 400 - Year - Old Church - Christianity Today") == (
        "How to Revitalize a 400-Year-Old Church")


def test_a_headline_is_not_mistaken_for_an_outlet():
    """The head has to stand on its own before the tail can be called a stamp."""
    assert clean_title("Trump - Harris") == "Trump - Harris"
    assert clean_title("Manhunt ends - the sheriff said so.") == (
        "Manhunt ends - the sheriff said so.")


def test_index_pages_are_recognized_through_their_outlet_stamp():
    """"U.S. Senate Articles" is too short to be a headline the stamp hangs off,
    so the stamp stays and the whole string is long enough to look like a story.
    """
    assert is_boilerplate("U.S. Senate Articles - Christianity Today")
    assert is_boilerplate("Palestine Articles - Christianity Today")
    assert not is_boilerplate("Articles of Confederation Rediscovered in an Attic")


def test_cleaning_drops_index_pages_but_never_empties_a_cluster():
    titles = ["Palestine Articles - Christianity Today",
              "Iran Shutters Country Last Presbyterian Church - Christianity Today"]
    assert clean_titles(titles) == ["Iran Shutters Country Last Presbyterian Church"]
    # every title boilerplate -> the cleaned titles come back rather than nothing
    assert clean_titles(["Palestine Articles", "Opinion"]) == [
        "Palestine Articles", "Opinion"]


def test_the_same_wire_story_under_two_outlets_is_listed_once():
    assert clean_titles([
        "Senate advances the funding bill - The Dispatch",
        "Senate advances the funding bill - Christianity Today",
    ]) == ["Senate advances the funding bill"]


# -- taxonomy ---------------------------------------------------------------


def test_taxonomy_names_a_subject_in_words_a_person_uses():
    assignment = taxonomy_theme([
        "Iran Shutters Country Last Presbyterian Church",
        "How to Revitalize a 400-Year-Old Church",
    ])
    assert assignment.key == "faith"
    assert assignment.title == "Faith & the church"
    assert assignment.method == "taxonomy"


def test_an_unrecognized_subject_says_so_instead_of_guessing():
    assignment = taxonomy_theme(["Barefoot bandit takes a cutout from a truck"])
    assert assignment.key == OTHER_KEY


def test_specific_subjects_win_over_the_catch_all_ones():
    """Nearly every political story mentions a chamber or a party; the taxonomy
    order is what keeps "Politics & government" from swallowing the day."""
    assignment = taxonomy_theme([
        "Senate Republicans weigh an abortion bill",
        "Lawmakers press the administration on abortion policy",
    ])
    assert assignment.key == "life"


# -- grouping ---------------------------------------------------------------


def _spot(cluster_id, dominant, other, titles, size=4, dominant_size=None):
    dominant_size = size if dominant_size is None else dominant_size
    return Blindspot(
        cluster_id=cluster_id,
        label="whatever c-tf-idf said",
        counts={dominant: dominant_size, other: size - dominant_size},
        dominant_diet=dominant,
        other_diet=other,
        dominant_share=dominant_size / size,
        size=size,
        representative_titles=titles,
    )


def test_clusters_on_one_subject_become_one_card():
    themes = group_blindspots([
        _spot(0, "modeled_ce", "self", ["Church plants a congregation downtown"]),
        _spot(1, "modeled_ce", "self", ["Pastor resigns after a long ministry"]),
    ])
    assert len(themes) == 1
    assert themes[0].title == "Faith & the church"
    assert themes[0].cluster_count == 2
    assert themes[0].story_count == 8


def test_direction_is_never_averaged_away():
    """One subject, both diets, is two findings — a card merging them reports
    neither, and the symmetry requirement is exactly about keeping them apart."""
    themes = group_blindspots([
        _spot(0, "modeled_ce", "self", ["Church plants a congregation downtown"]),
        _spot(1, "self", "modeled_ce", ["Pastor resigns after a long ministry"]),
    ])
    assert {t.dominant_diet for t in themes} == {"modeled_ce", "self"}
    assert all(t.key == "faith" for t in themes)


def test_a_theme_counts_the_stories_the_dominant_diet_actually_ran():
    """A cluster that is 90% one diet still holds a story or two from the other,
    and those are what makes "barely" the honest word rather than "never"."""
    theme = group_blindspots([
        _spot(0, "modeled_ce", "self", ["Pastor resigns after a long ministry"],
              size=10, dominant_size=9)
    ])[0]
    assert theme.story_count == 9
    assert round(theme.one_sided, 2) == 0.9


def test_unnamed_coverage_sorts_last():
    themes = group_blindspots([
        _spot(0, "modeled_ce", "self", ["A cutout goes missing from a truck"], size=9),
        _spot(1, "modeled_ce", "self", ["Pastor resigns after a long ministry"], size=2),
    ])
    assert [t.key for t in themes] == ["faith", OTHER_KEY]


def test_grouping_reads_exported_dicts_as_well_as_objects():
    """The digest groups the dicts out of a payload; the cluster run groups
    ``Blindspot`` objects. One implementation, so the two cannot drift."""
    as_dict = {
        "cluster_id": 0, "dominant_diet": "self", "other_diet": "modeled_ce",
        "size": 4, "dominant_share": 1.0, "counts": {"self": 4},
        "representative_titles": ["Pastor resigns after a long ministry"],
    }
    from_dict = group_blindspots([as_dict])
    from_object = group_blindspots(
        [_spot(0, "self", "modeled_ce", ["Pastor resigns after a long ministry"])])
    assert from_dict[0].to_dict() == from_object[0].to_dict()


# -- Claude naming ----------------------------------------------------------


class _FakeClient:
    """Minimal stand-in for the Anthropic client."""

    def __init__(self, text: str = "", error: Exception | None = None):
        self._text, self._error = text, error
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return type("R", (), {"content": [type("B", (), {"text": self._text})()]})()


ENTRIES = [(0, ["Pastor resigns after a long ministry"]),
           (1, ["Senate advances the funding bill"])]


def test_claude_names_the_themes_when_it_answers():
    client = _FakeClient(
        '{"assignments": [{"cluster_id": 0, "key": "faith", "title": "Life of the church"},'
        ' {"cluster_id": 1, "key": "politics", "title": "Congress this week"}]}'
    )
    out = assign_themes(ENTRIES, client=client)
    assert out[0].title == "Life of the church"
    assert out[0].method == "claude"
    assert out[1].title == "Congress this week"


def test_a_cluster_claude_skips_falls_back_on_its_own():
    client = _FakeClient(
        '{"assignments": [{"cluster_id": 0, "key": "faith", "title": "Life of the church"}]}'
    )
    out = assign_themes(ENTRIES, client=client)
    assert out[0].method == "claude"
    assert out[1].method == "taxonomy"
    assert out[1].title == "Politics & government"


def test_a_key_claude_used_keeps_claudes_wording_everywhere():
    """Otherwise one subject arrives as two cards under two spellings."""
    client = _FakeClient(
        '{"assignments": [{"cluster_id": 0, "key": "faith", "title": "Life of the church"}]}'
    )
    out = assign_themes([(0, ["Pastor resigns after a long ministry"]),
                         (1, ["Church plants a congregation downtown"])], client=client)
    assert out[1].title == "Life of the church"   # not the taxonomy's wording


def test_unusable_answers_are_dropped_per_cluster():
    client = _FakeClient(
        '{"assignments": ['
        '{"cluster_id": 0, "key": "faith", "title": "<b>Church</b>"},'          # markup
        '{"cluster_id": 1, "key": "Politics!", "title": "Congress this week"},'  # bad key
        '{"cluster_id": 9, "key": "faith", "title": "Not in the batch"}]}'       # unknown id
    )
    assert claude_assignments(ENTRIES, client=client) == {}


def test_the_prompts_own_examples_pass_the_validator():
    """A rule the prompt states and the validator does not enforce is not a
    rule; a rule the validator enforces without stating drops good titles for a
    reason the model was never told. The prompt's examples are the seam."""
    from cluster.themes import _MAX_TITLE_CHARS, _SYSTEM, _TITLE_OK

    for example in ("Israel-Hamas war", "Faith, family & work", "Faith & the church"):
        assert example in _SYSTEM               # the prompt holds it out as good
        assert _TITLE_OK.match(example)         # so the validator has to accept it
        assert len(example) <= _MAX_TITLE_CHARS
    assert not _TITLE_OK.match('"><b>x</b>')
    assert str(_MAX_TITLE_CHARS) in _SYSTEM   # the length limit is stated, not implied


def test_a_title_too_long_for_a_card_is_not_used():
    client = _FakeClient(
        '{"assignments": [{"cluster_id": 0, "key": "faith", "title": '
        f'"{"Very long theme name " * 4}"}}]}}'
    )
    assert claude_assignments(ENTRIES, client=client) == {}


def test_an_api_failure_is_a_quieter_brief_not_a_failed_run():
    client = _FakeClient(error=RuntimeError("503"))
    out = assign_themes(ENTRIES, client=client)
    assert all(a.method == "taxonomy" for a in out.values())


def test_claude_is_not_called_when_it_is_switched_off():
    client = _FakeClient('{"assignments": []}')
    assign_themes(ENTRIES, client=client, use_claude=False)
    assert client.calls == []


def test_no_api_key_means_the_taxonomy_not_an_empty_section(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = assign_themes(ENTRIES)
    assert [a.method for a in out.values()] == ["taxonomy", "taxonomy"]


# -- persistence ------------------------------------------------------------


def _seed(store):
    from cluster.embed import HashingEmbedder

    emb = HashingEmbedder(dim=256)
    rows = [
        ("modeled_ce", "Pastor resigns after a long ministry at the church",
         "church pastor congregation ministry faith worship scripture"),
        ("modeled_ce", "Church plants a congregation downtown - Christianity Today",
         "church pastor congregation ministry faith worship scripture"),
        ("modeled_ce", "Seminary trains a new generation of pastors",
         "church pastor congregation ministry faith worship scripture"),
        ("self", "Climate scientists report record ocean warming",
         "climate emissions carbon warming renewable solar energy"),
        ("self", "Solar and wind now supply a third of the grid",
         "climate emissions carbon warming renewable solar energy"),
        ("self", "Carbon targets slip further out of reach",
         "climate emissions carbon warming renewable solar energy"),
    ]
    for i, (diet, title, text) in enumerate(rows):
        store.upsert_document(
            doc_id=f"d{i}", diet_id=diet, source_id="s", stratum_id=None, url=None,
            title=title, published_utc=None, fetched_utc="2026-07-31T00:00:00+00:00",
            word_count=40, minhash=None)
        store.upsert_embedding(document_id=f"d{i}", vector=emb.embed(text),
                               embedder=emb.name)


def test_a_cluster_run_persists_themes_for_the_surfaces_to_read(monkeypatch):
    """The model call belongs to the cluster run. Naming at export time instead
    would bill once per surface and let the email and the page disagree."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = Datastore(":memory:")
    _seed(store)
    outcome = run_clustering(store, min_cluster_size=3, dominance=0.8,
                             min_blindspot_size=3)
    assert outcome.themes
    assert {t.title for t in outcome.themes} == {
        t.title for t in themes_from_store(store)}
    assert {row["theme_key"] for row in store.blindspot_theme_rows()}
    store.close()


def test_a_reclustering_does_not_inherit_yesterdays_names(monkeypatch):
    """Cluster ids are positions in a fresh run, not identities."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = Datastore(":memory:")
    _seed(store)
    run_clustering(store, min_cluster_size=3, dominance=0.8, min_blindspot_size=3)
    store.replace_clustering([(0, "label", 3)], [("d0", 0)])
    assert store.blindspot_theme_rows() == []
    store.close()


def test_a_store_with_no_persisted_themes_still_reads_by_theme(monkeypatch):
    """An older datastore, or one clustered before theming existed. Falling back
    to the taxonomy beats falling back to the cluster labels."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = Datastore(":memory:")
    _seed(store)
    run_clustering(store, min_cluster_size=3, dominance=0.8, min_blindspot_size=3)
    store.replace_blindspot_themes([])
    themes = themes_from_store(store)
    assert themes and all(isinstance(t, Theme) for t in themes)
    assert all(t.method == "taxonomy" for t in themes)
    store.close()


def test_persisted_names_are_the_ones_that_get_read():
    store = Datastore(":memory:")
    _seed(store)
    run_clustering(store, min_cluster_size=3, claude_themes=False)
    rows = store.blindspot_theme_rows()
    store.replace_blindspot_themes(
        [(r["cluster_id"], "renamed", "A Name From The Run", "claude") for r in rows])
    assert {t.title for t in themes_from_store(store)} == {"A Name From The Run"}
    store.close()


def test_assignment_records_who_named_the_theme():
    assignment = ThemeAssignment("faith", "Faith & the church", "taxonomy")
    theme = group_blindspots(
        [_spot(0, "self", "modeled_ce", ["Pastor resigns"])],
        {0: assignment},
    )[0]
    assert theme.method == "taxonomy"
    assert theme.to_dict()["method"] == "taxonomy"
