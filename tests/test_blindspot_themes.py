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
    Article,
    Theme,
    ThemeAssignment,
    assign_themes,
    claude_assignments,
    group_blindspots,
    taxonomy_theme,
)
from cluster.titles import clean_title, clean_titles, is_boilerplate
from ingestion.datastore import Datastore

# The reference pair, each persona reading its own source.
MEMBERS = {"self": {"src_self"}, "modeled_ce": {"src_modeled_ce"}}

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
    assert themes[0].story_count == 2          # one story per cluster
    assert themes[0].article_count == 2


def test_an_impure_cluster_splits_across_the_themes_it_actually_holds():
    """The bug this replaces: the theme was decided for the whole cluster, so
    the plurality won and every headline inherited it — which put "Cam
    Skattebo's backflipping at Fanatics Fest" under "Faith & the church"."""
    articles = {0: [
        Article("a", "Church plants a congregation downtown"),
        Article("b", "Pastor resigns after a long ministry at the church"),
        Article("c", "Giants quarterback earns a talking-to after a Fanatics Fest backflip"),
    ]}
    themes = group_blindspots(
        [_spot(0, "modeled_ce", "self", [])], articles=articles)
    by_key = {t.key: t for t in themes}
    assert set(by_key) == {"faith", "sports"}
    assert by_key["faith"].article_count == 2
    assert by_key["sports"].stories[0].title.startswith("Giants quarterback")


def test_a_headline_the_taxonomy_cannot_name_is_not_absorbed_by_its_neighbours():
    """"Other coverage" is the honest answer for a headline with no keyword the
    taxonomy knows. What it must not do is inherit the subject of the headlines
    filed beside it, which is what cluster-level assignment did."""
    articles = {0: [
        Article("a", "Church plants a congregation downtown"),
        Article("b", "Pastor resigns after a long ministry at the church"),
        Article("c", "Skattebo backflipping at Fanatics Fest earned a talking-to"),
    ]}
    by_key = {t.key: t for t in group_blindspots(
        [_spot(0, "modeled_ce", "self", [])], articles=articles)}
    assert by_key[OTHER_KEY].stories[0].title.startswith("Skattebo")
    assert all("Skattebo" not in s.title for s in by_key["faith"].stories)


def test_direction_is_never_averaged_away():
    """One subject, both diets, is two findings — a card merging them reports
    neither, and the symmetry requirement is exactly about keeping them apart."""
    themes = group_blindspots([
        _spot(0, "modeled_ce", "self", ["Church plants a congregation downtown"]),
        _spot(1, "self", "modeled_ce", ["Pastor resigns after a long ministry"]),
    ])
    assert {t.dominant_diet for t in themes} == {"modeled_ce", "self"}
    assert all(t.key == "faith" for t in themes)


def test_a_story_carries_the_outlets_that_ran_it():
    """"Three mastheads carried this and none of yours did" is the concrete
    form of the finding; three loose headlines are not."""
    articles = {0: [
        Article("a", "Iran shutters the country's last Presbyterian church",
                url="https://christianitytoday.com/a", outlet="Christianity Today"),
        Article("b", "Iran closes last Presbyterian church in the country",
                url="https://christianpost.com/b", outlet="The Christian Post"),
        Article("c", "Iran's last Presbyterian church closes its doors",
                url="https://christianitytoday.com/c", outlet="Christianity Today"),
    ]}
    story = group_blindspots(
        [_spot(0, "modeled_ce", "self", [], size=10, dominant_size=9)],
        articles=articles,
    )[0].stories[0]
    assert story.articles == 3
    # de-duplicated by masthead: two pieces from one outlet is one outlet
    assert [label for label, _ in story.outlets] == [
        "Christianity Today", "The Christian Post"]
    assert story.outlets[0][1] == "https://christianitytoday.com/a"
    assert round(story.one_sided, 2) == 0.9


def test_an_outlet_with_no_recorded_name_falls_back_to_its_host():
    articles = {0: [Article("a", "Pastor resigns after a long ministry",
                            url="https://www.example.org/story")]}
    story = group_blindspots(
        [_spot(0, "self", "modeled_ce", [])], articles=articles)[0].stories[0]
    assert story.outlets == [("example.org", "https://www.example.org/story")]


def test_an_outlet_with_neither_a_name_nor_a_link_falls_back_to_its_key():
    """A store ingested before outlet names were recorded still has the
    registry key. De-slugged it is a recognizable masthead, and dropping it
    would cost the reader the one thing that makes a story checkable."""
    articles = {0: [Article("a", "Pastor resigns after a long ministry",
                            source_id="christianity_today"),
                    Article("b", "Congregation votes on the pastor's successor",
                            source_id="npr")]}
    story = group_blindspots(
        [_spot(0, "self", "modeled_ce", [])], articles=articles)[0].stories[0]
    # Initialisms stay upper-case, since the registry's ids use them.
    assert story.outlets == [("Christianity Today", None), ("NPR", None)]


def test_a_recorded_name_beats_the_key_and_the_host():
    articles = {0: [Article("a", "Pastor resigns after a long ministry",
                            url="https://www.example.org/s",
                            outlet="Christianity Today",
                            source_id="christianity_today")]}
    story = group_blindspots(
        [_spot(0, "self", "modeled_ce", [])], articles=articles)[0].stories[0]
    assert story.outlets == [("Christianity Today", "https://www.example.org/s")]


def test_a_theme_credits_claude_when_claude_named_it():
    """`assign_themes` unifies a key's title, so one Claude assignment in a
    bucket means the words on the card are Claude's. Reporting the first
    story's method printed a footnote crediting the wrong author."""
    from cluster.themes import ThemeAssignment

    articles = {
        0: [Article("a", "Pastor resigns after a long ministry")],
        1: [Article("b", "Congregation votes on the pastor's successor")],
    }
    theme = group_blindspots(
        [_spot(0, "self", "modeled_ce", []), _spot(1, "self", "modeled_ce", [])],
        assignments={"b": ThemeAssignment("faith", "Church life", "claude")},
        articles=articles,
    )[0]
    assert theme.method == "claude"
    assert theme.title == "Church life"      # Claude's wording, not the taxonomy's
    assert theme.story_count == 2


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
    # a payload that predates stories still yields a card, minus the outlets
    assert from_dict[0].stories[0].title == "Pastor resigns after a long ministry"
    assert from_dict[0].stories[0].outlets == []


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


# Keyed by document id; the prompt refers to them by position.
ENTRIES = [("doc-a", ["Pastor resigns after a long ministry"]),
           ("doc-b", ["Senate advances the funding bill"])]


def test_claude_names_the_themes_when_it_answers():
    client = _FakeClient(
        '{"themes": [{"key": "faith", "title": "Life of the church", "stories": [0]},'
        ' {"key": "politics", "title": "Congress this week", "stories": [1]}]}'
    )
    out = assign_themes(ENTRIES, client=client)
    assert out["doc-a"].title == "Life of the church"
    assert out["doc-a"].method == "claude"
    assert out["doc-b"].title == "Congress this week"


def test_a_cluster_claude_skips_falls_back_on_its_own():
    client = _FakeClient(
        '{"themes": [{"key": "faith", "title": "Life of the church", "stories": [0]}]}'
    )
    out = assign_themes(ENTRIES, client=client)
    assert out["doc-a"].method == "claude"
    assert out["doc-b"].method == "taxonomy"
    assert out["doc-b"].title == "Politics & government"


def test_a_key_claude_used_keeps_claudes_wording_everywhere():
    """Otherwise one subject arrives as two cards under two spellings."""
    client = _FakeClient(
        '{"themes": [{"key": "faith", "title": "Life of the church", "stories": [0]}]}'
    )
    out = assign_themes([("doc-a", ["Pastor resigns after a long ministry"]),
                         ("doc-b", ["Church plants a congregation downtown"])],
                        client=client)
    assert out["doc-b"].title == "Life of the church"   # not the taxonomy's wording


def test_a_theme_missing_its_stories_is_dropped_whole():
    """The grouped reply fans out here, so a malformed group would otherwise
    expand into records that each fail validation for a different reason. Its
    stories fall back on the taxonomy, which is where a story Claude never
    mentioned ends up anyway."""
    client = _FakeClient(
        '{"themes": [{"key": "faith", "title": "Life of the church"},'
        ' {"key": "politics", "title": "Congress this week", "stories": [1]}]}'
    )
    out = assign_themes(ENTRIES, client=client)
    assert out["doc-a"].method == "taxonomy"
    assert out["doc-b"].method == "claude"


def test_a_big_days_answer_fits_the_budget():
    """The bug that ran silently from the day themes landed: the reply cost
    ~30 tokens a story, `max_tokens` was clamped to 8000, and a 717-story day
    therefore asked for an answer several times larger than the budget could
    hold. It could not have succeeded at any effort, and every cluster run
    dropped the whole batch to the taxonomy. Both halves of the fix are pinned
    here: the reply is grouped, so a story costs an integer rather than an
    object, and the budget scales past what that costs."""
    entries = [(f"doc-{i}", [f"Headline {i}"]) for i in range(717)]
    client = _FakeClient('{"themes": []}')
    assign_themes(entries, client=client)
    call = client.calls[0]

    prompt = call["messages"][0]["content"]
    assert '"stories": [0, 3, 7]' in prompt      # grouped, not one object per story
    assert '"story": 0' not in prompt            # the shape that could not fit

    # A story number and its separator run ~2 tokens; leave the same again for
    # the theme objects around them, and thinking shares this budget too.
    assert call["max_tokens"] > 4 * len(entries)


def test_unusable_answers_are_dropped_per_cluster():
    client = _FakeClient(
        '{"themes": ['
        '{"key": "faith", "title": "<b>Church</b>", "stories": [0]},'          # markup
        '{"key": "Politics!", "title": "Congress this week", "stories": [1]},'  # bad key
        '{"key": "faith", "title": "Not in the batch", "stories": [9]}]}'       # index out of range
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
        '{"themes": [{"key": "faith", "stories": [0], "title": '
        f'"{"Very long theme name " * 4}"}}]}}'
    )
    assert claude_assignments(ENTRIES, client=client) == {}


def test_an_api_failure_is_a_quieter_brief_not_a_failed_run():
    client = _FakeClient(error=RuntimeError("503"))
    out = assign_themes(ENTRIES, client=client)
    assert all(a.method == "taxonomy" for a in out.values())


def test_theming_sends_an_effort_and_it_is_settable():
    """The knob the summary and liberty already had. Without it this call took
    the model's own default, which on Sonnet 5 is adaptive thinking at `high` —
    several times the output tokens to place a headline in a fixed vocabulary,
    billed on every cluster run."""
    from cluster.themes import DEFAULT_THEME_EFFORT

    client = _FakeClient('{"themes": []}')
    assign_themes(ENTRIES, client=client)
    assert client.calls[0]["output_config"]["effort"] == DEFAULT_THEME_EFFORT

    louder = _FakeClient('{"themes": []}')
    assign_themes(ENTRIES, client=louder, effort="high")
    assert louder.calls[0]["output_config"]["effort"] == "high"


def test_an_explicit_none_effort_sends_no_effort():
    """A `None` effort means "let the model decide" — a real third option, not
    the same call as the default one. `test_settings_null_effort_survives_to_the
    _call` is what checks that YAML's `effort: ~` arrives here as this."""
    client = _FakeClient('{"themes": []}')
    assign_themes(ENTRIES, client=client, effort=None)
    assert "effort" not in client.calls[0]["output_config"]


def test_the_reply_is_constrained_to_a_schema():
    """Structured outputs, so a malformed reply is not a failure mode the
    taxonomy has to absorb. The schema travels in the same `output_config` as
    the effort, and unlike the effort it is sent unconditionally."""
    client = _FakeClient('{"themes": []}')
    assign_themes(ENTRIES, client=client, effort=None)
    fmt = client.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    item = fmt["schema"]["properties"]["themes"]["items"]
    assert item["properties"]["stories"] == {"type": "array",
                                             "items": {"type": "integer"}}
    assert sorted(item["required"]) == ["key", "stories", "title"]


def test_settings_effort_reaches_the_call(monkeypatch):
    """The setting is only worth having if it survives the walk from
    settings.yaml down to `messages.create`."""
    import cluster.blindspot as bs

    seen: dict = {}
    monkeypatch.setattr(bs, "articles_from_store", lambda store, spots, members: {})
    monkeypatch.setattr(bs, "assign_themes",
                        lambda entries, client=None, **kw: seen.update(kw) or {})
    monkeypatch.setattr(bs, "group_blindspots", lambda *a, **k: [])

    class _Store:
        def replace_blindspot_themes(self, rows): pass

    bs._theme_blindspots(_Store(), [object()], MEMBERS, None, None, True, "medium")
    assert seen["effort"] == "medium"

    # Not configured at all: `assign_themes` applies its own default.
    seen.clear()
    bs._theme_blindspots(_Store(), [object()], MEMBERS, None, None, True)
    assert "effort" not in seen

    # Configured as null: that is an instruction, and it has to survive. This is
    # the distinction a `None` default cannot carry, which is why UNSET exists.
    seen.clear()
    bs._theme_blindspots(_Store(), [object()], MEMBERS, None, None, True, None)
    assert seen["effort"] is None


def test_settings_null_effort_survives_to_the_call():
    """`effort: ~` end to end, which is where this broke: every layer read the
    key with `.get`, so a configured null came back as "unconfigured" and the
    default was reapplied. The knob documented as restoring the model's own
    behaviour did nothing at all."""
    from cluster.themes import UNSET
    from daily.runner import DailyConfig

    absent = DailyConfig.from_settings({"cluster": {"themes": {}}})
    assert absent.theme_effort is UNSET

    null = DailyConfig.from_settings({"cluster": {"themes": {"effort": None}}})
    assert null.theme_effort is None

    named = DailyConfig.from_settings({"cluster": {"themes": {"effort": "high"}}})
    assert named.theme_effort == "high"


def test_claude_is_not_called_when_it_is_switched_off():
    client = _FakeClient('{"themes": []}')
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
            doc_id=f"d{i}", source_id=f"src_{diet}", stratum_id=None, url=None,
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
    outcome = run_clustering(store, MEMBERS, min_cluster_size=3, dominance=0.8,
                             min_blindspot_size=3)
    assert outcome.themes
    assert {t.title for t in outcome.themes} == {
        t.title for t in themes_from_store(store, MEMBERS)}
    assert {row["theme_key"] for row in store.blindspot_theme_rows()}
    store.close()


def test_a_reclustering_does_not_inherit_yesterdays_names(monkeypatch):
    """Cluster ids are positions in a fresh run, not identities."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = Datastore(":memory:")
    _seed(store)
    run_clustering(store, MEMBERS, min_cluster_size=3, dominance=0.8, min_blindspot_size=3)
    store.replace_clustering([(0, "label", 3)], [("d0", 0)])
    assert store.blindspot_theme_rows() == []
    store.close()


def test_a_store_with_no_persisted_themes_still_reads_by_theme(monkeypatch):
    """An older datastore, or one clustered before theming existed. Falling back
    to the taxonomy beats falling back to the cluster labels."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = Datastore(":memory:")
    _seed(store)
    run_clustering(store, MEMBERS, min_cluster_size=3, dominance=0.8, min_blindspot_size=3)
    store.replace_blindspot_themes([])
    themes = themes_from_store(store, MEMBERS)
    assert themes and all(isinstance(t, Theme) for t in themes)
    assert all(t.method == "taxonomy" for t in themes)
    store.close()


def test_persisted_names_are_the_ones_that_get_read():
    store = Datastore(":memory:")
    _seed(store)
    run_clustering(store, MEMBERS, min_cluster_size=3, claude_themes=False)
    rows = store.blindspot_theme_rows()
    store.replace_blindspot_themes(
        [(r["document_id"], "renamed", "A Name From The Run", "claude") for r in rows])
    assert {t.title for t in themes_from_store(store, MEMBERS)} == {"A Name From The Run"}
    store.close()


def test_assignment_records_who_named_the_theme():
    assignment = ThemeAssignment("faith", "Faith & the church", "taxonomy")
    theme = group_blindspots(
        [_spot(0, "self", "modeled_ce", ["Pastor resigns"])],
        {0: assignment},
    )[0]
    assert theme.method == "taxonomy"
    assert theme.to_dict()["method"] == "taxonomy"
