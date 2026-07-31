"""Summarizer: prompt building, section parsing, deterministic fallback, and
the injected-client path (no network)."""

from __future__ import annotations

import re

from ingestion.datastore import Datastore
from summarize.prompts import SYSTEM_PROMPT, ComparisonContext, DietContext, build_user_prompt
from summarize.summarizer import DEFAULT_MODEL, Summarizer, _parse_sections, gather


def _seed_store():
    store = Datastore(":memory:")
    for i, diet in enumerate(["self", "modeled_ce"]):
        doc_id = f"{diet}-{i}"
        store.upsert_document(
            doc_id=doc_id, diet_id=diet, source_id="s", stratum_id=None,
            url="http://x", title=f"{diet} headline", published_utc=None,
            fetched_utc="2026-07-23T00:00:00+00:00", word_count=100, minhash=None,
        )
        store.upsert_scores(
            document_id=doc_id, scorer="dictionary",
            foundations={"care": 0.3 if diet == "self" else 0.1,
                         "fairness": 0.1, "loyalty": 0.1 if diet == "self" else 0.4,
                         "authority": 0.2, "sanctity": 0.1},
            sentiment=0.0, moral_word_ratio=0.2, matched_words=20,
        )
    return store


def test_prompt_contains_rules_data_and_headlines():
    ctx = [DietContext("self", "self", 3, {"care": 0.5, "loyalty": 0.5},
                       ["a headline", "b headline"])]
    cmp = ComparisonContext("self", "modeled_ce", 0.12, {"care": 0.4, "loyalty": -0.4})
    prompt = build_user_prompt(ctx, cmp)
    assert "## <label>" in prompt
    assert "a headline" in prompt
    assert "Jensen-Shannon divergence: 0.120" in prompt
    assert "care=0.50" in prompt


def test_max_headlines_respected():
    ctx = [DietContext("self", "self", 30, {"care": 1.0}, [f"h{i}" for i in range(30)])]
    prompt = build_user_prompt(ctx, None, max_headlines=5)
    assert "h4" in prompt and "h5" not in prompt


def test_parse_sections_splits_by_headers():
    contexts = [DietContext("self", "self", 1, {}, []),
                DietContext("modeled_ce", "modeled_ce", 1, {}, [])]
    text = "## self\nSelf paragraph.\n## modeled_ce\nOther paragraph.\n## Executive\nThe exec."
    per_diet, executive = _parse_sections(text, contexts)
    assert per_diet["self"] == "Self paragraph."
    assert per_diet["modeled_ce"] == "Other paragraph."
    assert executive == "The exec."


def test_unparseable_response_falls_back_to_whole_text():
    contexts = [DietContext("self", "self", 1, {}, [])]
    per_diet, executive = _parse_sections("just a blob with no headers", contexts)
    assert executive == "just a blob with no headers"


def test_deterministic_fallback_without_key():
    store = _seed_store()
    result = Summarizer().summarize(store)  # no ANTHROPIC_API_KEY in tests
    assert result.method == "deterministic"
    assert set(result.per_diet) == {"self", "modeled_ce"}
    assert "Jensen-Shannon" in result.executive
    store.close()


class _FakeBlock:
    def __init__(self, text): self.text = text


class _FakeMessages:
    def create(self, **kwargs):
        class R:
            content = [_FakeBlock("## self\nS.\n## modeled_ce\nO.\n## Executive\nE.")]
        return R()


class _FakeClient:
    messages = _FakeMessages()


# -- the diets are named, not keyed ----------------------------------------


def _seed_labels(store):
    store.set_diet_label("self", "My diet", "My diet")
    store.set_diet_label("modeled_ce", "Modeled conservative-evangelical diet",
                         "The modeled diet")


def test_the_summarizer_is_given_the_diets_human_labels():
    """The registry has written labels for exactly this. Passing the machine id
    as the label is what put "modeled_ce" in prose meant for a person."""
    store = _seed_store()
    _seed_labels(store)
    contexts, _ = gather(store)
    assert {c.label for c in contexts} == {
        "My diet", "Modeled conservative-evangelical diet"}
    store.close()


def test_the_prompt_carries_labels_and_forbids_the_ids():
    ctx = [DietContext("self", "My diet", 3, {"care": 1.0}, ["a headline"]),
           DietContext("modeled_ce", "Modeled conservative-evangelical diet", 2,
                       {"loyalty": 1.0}, ["b headline"])]
    cmp = ComparisonContext("self", "modeled_ce", 0.12, {"care": 0.4})
    prompt = build_user_prompt(ctx, cmp)
    assert "Modeled conservative-evangelical diet" in prompt
    # the ids are database keys; nothing in the data block should invite them
    assert "modeled_ce" not in prompt
    assert "machine ids" in SYSTEM_PROMPT


def test_the_style_rules_are_in_the_system_prompt():
    """The executive summary is read on a phone before anything else. These are
    the tells that make a paragraph unreadable there, not stylistic taste."""
    for rule in ("em dashes", "Short paragraphs", "groups of three",
                 "Lead with the finding", "Vary sentence length"):
        assert rule in SYSTEM_PROMPT


def _labelled_contexts():
    return [DietContext("self", "My diet", 1, {}, [], short_label="My diet"),
            DietContext("modeled_ce", "Modeled conservative-evangelical diet", 1, {}, [],
                        short_label="The modeled diet")]


def test_a_heading_that_does_not_match_the_label_exactly_still_lands():
    """Labels are sentences now. A model that title-cases one or drops the
    trailing noun has still named the right diet."""
    text = ("## My Diet\nMine.\n## Modeled Conservative-Evangelical\nTheirs.\n"
            "## Executive\nThe exec.")
    per_diet, executive = _parse_sections(text, _labelled_contexts())
    assert per_diet["self"] == "Mine."
    assert per_diet["modeled_ce"] == "Theirs."
    assert executive == "The exec."


def test_a_heading_in_the_short_form_lands_on_the_right_diet():
    """The bug this replaces: substring matching reduced "My diet" to "diet",
    which every label ends with, so "The modeled diet" matched the author's own
    and a section about the modeled diet was filed under `self`. That inverts
    the one guarantee the tool makes."""
    per_diet, _ = _parse_sections(
        "## The modeled diet\nTheirs.\n## Executive\nE.", _labelled_contexts())
    assert per_diet["modeled_ce"] == "Theirs."
    assert per_diet["self"] == ""


def test_a_heading_that_fits_both_diets_is_not_guessed_at():
    """Dropping a section is recoverable. Attributing it to the wrong diet is
    not, and it is the failure nobody would notice."""
    per_diet, _ = _parse_sections("## Diet\nWhose?\n", _labelled_contexts())
    assert per_diet == {"self": "", "modeled_ce": ""}


def test_a_heading_carrying_extra_words_still_lands():
    per_diet, _ = _parse_sections(
        "## My diet today\nMine.\n", _labelled_contexts())
    assert per_diet["self"] == "Mine."


def test_the_numbers_only_fallback_names_the_diets_too():
    """Same reader, same morning. The fallback has no excuse the model doesn't."""
    store = _seed_store()
    _seed_labels(store)
    result = Summarizer().summarize(store)
    assert "Modeled conservative-evangelical diet" in result.executive
    assert not re.search(r"\bmodeled_ce\b", result.executive)
    store.close()


def test_the_default_model_is_the_current_opus():
    assert DEFAULT_MODEL == "claude-opus-5"


def test_the_daily_run_honours_the_configured_summary_model():
    """`summarize.model` was documented in settings.example.yaml and read by
    nothing: the daily run always took the summarizer's default."""
    from daily.runner import DailyConfig

    cfg = DailyConfig.from_settings({"summarize": {"model": "claude-sonnet-5"}})
    assert cfg.model == "claude-sonnet-5"
    assert DailyConfig.from_settings({}).model is None   # -> the summarizer default


def test_the_labels_command_records_them_without_re_ingesting():
    """Labels are written at ingestion, so a store last ingested before they
    existed names its diets by machine id until something records them."""
    from types import SimpleNamespace

    from ingestion.pipeline import _store_diet_labels

    store = Datastore(":memory:")
    registry = SimpleNamespace(diets=[
        SimpleNamespace(id="self", label="My diet", short_label="My diet"),
        SimpleNamespace(id="modeled_ce", label="Modeled conservative-evangelical diet",
                        short_label="The modeled diet"),
    ])
    _store_diet_labels(store, registry)
    assert store.diet_labels()["modeled_ce"] == "Modeled conservative-evangelical diet"
    assert store.diet_short_labels()["modeled_ce"] == "The modeled diet"
    store.close()


def test_injected_client_path_and_persist():
    store = _seed_store()
    result = Summarizer(client=_FakeClient()).summarize(store)
    assert result.method == "claude"
    assert result.per_diet["self"] == "S."
    assert result.executive == "E."
    Summarizer(client=_FakeClient()).persist(store, result)
    assert store.all_summaries()["executive"]["text"] == "E."
    store.close()
