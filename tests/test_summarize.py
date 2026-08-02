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


def test_deterministic_fallback_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = _seed_store()
    result = Summarizer().summarize(store)
    assert result.method == "deterministic"
    assert set(result.per_diet) == {"self", "modeled_ce"}
    assert "Jensen-Shannon" in result.executive
    assert "no ANTHROPIC_API_KEY set" in result.executive
    store.close()


def test_fallback_note_names_the_actual_cause(monkeypatch):
    """The regression this exists to prevent: every fallback, whatever caused
    it, told the reader to go set a key. Four of the five causes are something
    else, and for two of them the key is already set and already working."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    store = _seed_store()

    class _Boom:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise TimeoutError("upstream took too long")

    result = Summarizer(client=_Boom()).summarize(store)
    assert result.method == "deterministic"
    for text in [result.executive, *result.per_diet.values()]:
        assert "the Claude call failed (TimeoutError)" in text
        assert "ANTHROPIC_API_KEY" not in text
    store.close()


def test_fallback_note_names_an_empty_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    store = _seed_store()
    result = Summarizer(client=_FakeClient(blocks=[])).summarize(store)
    assert result.method == "deterministic"
    assert "Claude returned no summary text" in result.executive
    store.close()


def test_a_falsy_injected_client_is_still_the_client(monkeypatch):
    """The injected client is a sentinel, not a truth value: a double that
    defines __bool__ (or __len__, which a Mock-alike may) would otherwise be
    discarded in favour of one built from the environment — the opposite of
    what injecting it asked for."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = _FakeClient()
    client.__class__.__bool__ = lambda self: False
    try:
        store = _seed_store()
        result = Summarizer(client=client).summarize(store)
        assert result.method == "claude"
        assert client.messages.calls, "the injected client was never called"
        store.close()
    finally:
        del client.__class__.__bool__


def test_missing_package_is_not_reported_as_a_missing_key(monkeypatch):
    """A key that is set and an `llm` extra that is not — the one people hit
    right after a fresh checkout and a `pip install -e ".[dev]"`."""
    import summarize.summarizer as mod
    from scoring.claude_client import NO_PACKAGE

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(mod, "build_client", lambda: (None, NO_PACKAGE))
    store = _seed_store()
    result = Summarizer().summarize(store)
    assert "the `anthropic` package is not installed" in result.executive
    assert "no ANTHROPIC_API_KEY set" not in result.executive
    store.close()


class _FakeBlock:
    def __init__(self, text): self.text = text


class _FakeMessages:
    def __init__(self, blocks=None, stop_reason="end_turn"):
        self._blocks = blocks
        self._stop_reason = stop_reason
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        blocks = self._blocks
        if blocks is None:
            blocks = [_FakeBlock("## self\nS.\n## modeled_ce\nO.\n## Executive\nE.")]
        return type("R", (), {"content": blocks, "stop_reason": self._stop_reason})()


class _FakeClient:
    def __init__(self, blocks=None, stop_reason="end_turn"):
        self.messages = _FakeMessages(blocks, stop_reason)


class _ThinkingBlock:
    """What a response carries when reasoning ate the whole token budget.

    Thinking blocks have `.thinking`, not `.text`, and on this model family
    their text is empty by default — so a response can be well-formed, HTTP
    200, and carry nothing to print.
    """

    type = "thinking"

    def __init__(self, thinking=""):
        self.thinking = thinking


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
    existed names its diets by machine id until something records them. Outlet
    names come along, because a blindspot story lists which outlets ran it and
    `christianity_today` is a key rather than a masthead."""
    from types import SimpleNamespace

    from ingestion.pipeline import _store_diet_labels

    store = Datastore(":memory:")
    sources = [SimpleNamespace(id="christianity_today", name="Christianity Today"),
               SimpleNamespace(id="unnamed", name="")]
    registry = SimpleNamespace(
        diets=[
            SimpleNamespace(id="self", label="My diet", short_label="My diet"),
            SimpleNamespace(id="modeled_ce", label="Modeled conservative-evangelical diet",
                            short_label="The modeled diet"),
        ],
        all_sources=lambda: sources,
    )
    _store_diet_labels(store, registry)
    assert store.diet_labels()["modeled_ce"] == "Modeled conservative-evangelical diet"
    assert store.diet_short_labels()["modeled_ce"] == "The modeled diet"
    assert store.source_labels() == {"christianity_today": "Christianity Today"}
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


# -- a successful call that says nothing ------------------------------------


def test_a_response_with_no_text_falls_back_instead_of_writing_nothing():
    """The bug this replaces: Opus 5 thinks by default and `max_tokens` caps
    thinking and prose together, so at the old 1500 the reasoning consumed the
    budget and the text block came back empty. The run persisted empty
    summaries over the brief and reported success; the email had no prose in
    it and nothing said why."""
    store = _seed_store()
    client = _FakeClient(blocks=[_ThinkingBlock()], stop_reason="max_tokens")
    result = Summarizer(client=client).summarize(store)
    assert result.method == "deterministic"
    assert result.executive.strip()
    assert all(t.strip() for t in result.per_diet.values())
    store.close()


def test_a_refusal_is_read_before_the_content():
    """Safety classifiers decline with an HTTP 200 and empty-or-partial
    content, so `stop_reason` has to be checked first."""
    store = _seed_store()
    client = _FakeClient(blocks=[], stop_reason="refusal")
    result = Summarizer(client=client).summarize(store)
    assert result.method == "deterministic"
    assert result.executive.strip()
    store.close()


def test_the_token_budget_leaves_room_for_thinking_and_prose():
    """`max_tokens` is one budget for both on this model family."""
    from summarize.summarizer import MAX_TOKENS

    store = _seed_store()
    client = _FakeClient()
    Summarizer(client=client).summarize(store)
    call = client.messages.calls[0]
    assert call["max_tokens"] == MAX_TOKENS >= 16000
    assert call["output_config"] == {"effort": "medium"}
    store.close()


def test_the_effort_level_is_configurable():
    store = _seed_store()
    client = _FakeClient()
    Summarizer(client=client, effort="high").summarize(store)
    assert client.messages.calls[0]["output_config"] == {"effort": "high"}
    store.close()


class _TypedBlock:
    """A block that declares a type other than `text` but carries `.text`.

    No shipped block does this today; the point of the test is that a future
    one cannot silently splice itself into the prose.
    """

    type = "fallback"

    def __init__(self, text):
        self.text = text


def test_only_prose_blocks_reach_the_summary():
    store = _seed_store()
    client = _FakeClient(blocks=[_TypedBlock("NOT PROSE"),
                                 _FakeBlock("## Executive\nE.")])
    result = Summarizer(client=client).summarize(store)
    assert result.executive == "E."
    assert "NOT PROSE" not in result.executive
    store.close()


def test_a_missing_diet_section_is_reported_not_silently_blank(caplog):
    """A per-diet section can be missing because the model skipped it — and it
    is also what a broken heading matcher looks like, which is otherwise
    silent: the panel just stops appearing."""
    store = _seed_store()
    _seed_labels(store)
    client = _FakeClient(blocks=[_FakeBlock("## My diet\nMine.\n## Executive\nE.")])
    with caplog.at_level("WARNING"):
        result = Summarizer(client=client).summarize(store)
    assert result.per_diet["self"] == "Mine."
    assert result.per_diet["modeled_ce"] == ""
    assert "modeled_ce" in caplog.text
    store.close()


def test_a_partial_result_is_persisted_and_the_report_names_the_shortfall(monkeypatch):
    """The empty row is written, not skipped: it is this run's answer for that
    diet, and leaving the previous run's text in place would put yesterday's
    prose under today's date."""
    from summarize.summarizer import Summarizer, SummaryResult

    partial = SummaryResult({"self": "Mine.", "modeled_ce": ""}, "E.", "m", "claude", "t")
    monkeypatch.setattr(Summarizer, "summarize", lambda self, store: partial)

    store = _seed_store()
    detail = _step_summarize_detail(store)
    assert "1 diets of 2" in detail
    assert store.all_summaries()["modeled_ce"]["text"] == ""   # written, not skipped
    assert store.all_summaries()["executive"]["text"] == "E."
    store.close()


def test_an_executive_only_result_says_so(monkeypatch):
    from summarize.summarizer import Summarizer, SummaryResult

    only_diets = SummaryResult({"self": "Mine."}, "", "m", "claude", "t")
    monkeypatch.setattr(Summarizer, "summarize", lambda self, store: only_diets)

    store = _seed_store()
    assert "no executive" in _step_summarize_detail(store)
    store.close()


def test_a_truncated_but_usable_summary_is_kept():
    """Partial prose beats none — the brief degrades to a summary that stops
    early rather than to silence."""
    store = _seed_store()
    client = _FakeClient(blocks=[_FakeBlock("## Executive\nHalf a sen")],
                         stop_reason="max_tokens")
    result = Summarizer(client=client).summarize(store)
    assert result.method == "claude"
    assert result.executive == "Half a sen"
    store.close()


def test_the_daily_step_does_not_report_success_on_empty_summaries(monkeypatch):
    """`{"self": ""}` is a truthy dict of empty strings — the guard tested the
    dict, so an all-empty result was persisted over the day's brief and
    reported as "2 diets"."""
    from daily.runner import DailyConfig, _step_summarize
    from summarize.summarizer import Summarizer, SummaryResult

    empty = SummaryResult({"self": "", "modeled_ce": ""}, "", "m", "claude", "t")
    monkeypatch.setattr(Summarizer, "summarize", lambda self, store: empty)

    store = _seed_store()
    detail = _step_summarize(store, DailyConfig())
    assert "nothing persisted" in detail
    assert "executive" not in store.all_summaries()
    store.close()


def test_the_daily_step_reports_the_diets_it_actually_wrote():
    store = _seed_store()
    detail = _step_summarize_detail(store)
    assert "2 diets" in detail
    assert store.all_summaries()["executive"]["text"].strip()
    store.close()


def _step_summarize_detail(store):
    from daily.runner import DailyConfig, _step_summarize

    return _step_summarize(store, DailyConfig())


def test_the_daily_run_reads_the_configured_effort():
    from daily.runner import DailyConfig

    cfg = DailyConfig.from_settings({"summarize": {"effort": "low"}})
    assert cfg.summary_effort == "low"
    assert DailyConfig.from_settings({}).summary_effort is None
