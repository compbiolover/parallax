"""Summarizer: prompt building, section parsing, deterministic fallback, and
the injected-client path (no network)."""

from __future__ import annotations

from ingestion.datastore import Datastore
from summarize.prompts import ComparisonContext, DietContext, build_user_prompt
from summarize.summarizer import Summarizer, _parse_sections


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
    ctx = [DietContext("self", "self", 3, {"care": 0.5, "loyalty": 0.5}, ["a headline", "b headline"])]
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
        class R: content = [_FakeBlock("## self\nS.\n## modeled_ce\nO.\n## Executive\nE.")]
        return R()


class _FakeClient:
    messages = _FakeMessages()


def test_injected_client_path_and_persist():
    store = _seed_store()
    result = Summarizer(client=_FakeClient()).summarize(store)
    assert result.method == "claude"
    assert result.per_diet["self"] == "S."
    assert result.executive == "E."
    Summarizer(client=_FakeClient()).persist(store, result)
    assert store.all_summaries()["executive"]["text"] == "E."
    store.close()
