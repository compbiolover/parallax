"""The Claude liberty tagger: parsing, batching, aggregation, and rubric symmetry."""

from __future__ import annotations

import json
from types import SimpleNamespace

from compare.liberty import (
    LOW_COVERAGE,
    all_persona_liberty,
    gap,
    persona_liberty_profile,
)
from ingestion.datastore import Datastore
from scoring.liberty import (
    BATCH_MIN_ITEMS,
    SYSTEM_PROMPT,
    LibertyScore,
    LibertyTagger,
    _parse,
    build_tagger,
)

from .registries import pair, registry

VERDICT = {
    "presence": 0.8,
    "pole": "vice",
    "register": "from_state",
    "quote": "the mandate leaves families no choice",
    "rationale": "Frames the rule as coercion of families by the state.",
}


def _response(payload: dict | str, stop_reason: str = "end_turn"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )


class _StubMessages:
    """Records calls and replays canned responses."""

    def __init__(self, response=None, batches=None):
        self._response = response or _response(VERDICT)
        self.calls: list[dict] = []
        self.batches = batches

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _StubBatches:
    def __init__(self, results, status="ended"):
        self._results = results
        self._status = status
        self.submitted: list[list[dict]] = []

    def create(self, requests):
        self.submitted.append(requests)
        return SimpleNamespace(id="batch_1")

    def retrieve(self, _batch_id):
        return SimpleNamespace(processing_status=self._status)

    def results(self, _batch_id):
        return iter(self._results)


def _batch_result(custom_id: str, payload: dict, kind: str = "succeeded"):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type=kind, message=_response(payload)),
    )


def _client(response=None, batches=None):
    return SimpleNamespace(messages=_StubMessages(response, batches))


# -- the rubric ------------------------------------------------------------


def _rubric() -> str:
    """The rubric with wrapping collapsed — line breaks are formatting, not content."""
    return " ".join(SYSTEM_PROMPT.lower().split())


def test_rubric_names_both_liberty_registers():
    """Regression guard on the design. A rubric that recognized only coercion by
    the state would score one diet as caring about liberty and the other as
    silent — the same one-sided-instrument failure the fairness lexicon had."""
    prompt = _rubric()
    for state_side in ("government overreach", "mandates", "censorship"):
        assert state_side in prompt
    for private_side in ("corporate", "bodily autonomy", "employer control"):
        assert private_side in prompt
    assert "both registers count" in prompt
    assert "neither is more truly liberty" in prompt


def test_rubric_excludes_the_neighbouring_foundations():
    prompt = _rubric()
    assert "that is fairness, not liberty" in prompt
    assert "that is care" in prompt


# -- parsing ---------------------------------------------------------------


def test_parses_a_well_formed_verdict():
    score = _parse(json.dumps(VERDICT), "m")
    assert score.presence == 0.8
    assert score.pole == "vice"
    assert score.register == "from_state"
    assert score.grounded


def test_presence_is_clamped_not_rejected():
    """A schema-valid response with a stray 1.4 is still a usable judgment;
    discarding it would silently thin coverage."""
    assert _parse(json.dumps({**VERDICT, "presence": 1.4}), "m").presence == 1.0
    assert _parse(json.dumps({**VERDICT, "presence": -0.2}), "m").presence == 0.0


def test_unparseable_payloads_yield_none():
    assert _parse("not json", "m") is None
    assert _parse(json.dumps({"pole": "virtue"}), "m") is None
    assert _parse(json.dumps({**VERDICT, "presence": "high"}), "m") is None


def test_signed_uses_the_pole():
    assert LibertyScore(0.6, "virtue", "both", "q", "r", "m").signed == 0.6
    assert LibertyScore(0.6, "vice", "both", "q", "r", "m").signed == -0.6
    assert LibertyScore(0.6, "mixed", "both", "q", "r", "m").signed == 0.0


def test_a_verdict_without_a_quote_is_flagged_ungrounded():
    assert not _parse(json.dumps({**VERDICT, "quote": "  "}), "m").grounded


# -- single-document scoring ----------------------------------------------


def test_score_sends_the_cached_rubric_and_json_schema():
    client = _client()
    LibertyTagger(client, model="claude-sonnet-5").score("some article text")
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "low"


def test_refusal_is_handled_before_reading_content():
    client = _client(_response("", stop_reason="refusal"))
    assert LibertyTagger(client).score("text") is None


def test_api_errors_degrade_to_none():
    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("network")

    assert LibertyTagger(SimpleNamespace(messages=Boom())).score("text") is None


def test_empty_text_is_not_sent():
    client = _client()
    assert LibertyTagger(client).score("   ") is None
    assert client.messages.calls == []


def test_scorer_name_carries_the_model():
    assert LibertyTagger(_client(), model="claude-sonnet-5").name == (
        "claude-liberty/claude-sonnet-5"
    )


# -- batching --------------------------------------------------------------


def test_small_sets_skip_the_batch_api():
    client = _client()
    texts = {f"d{i}": "text" for i in range(BATCH_MIN_ITEMS - 1)}
    scored = LibertyTagger(client).score_many(texts)
    assert len(scored) == len(texts)
    assert len(client.messages.calls) == len(texts)  # all synchronous


def test_batch_results_are_keyed_by_custom_id_not_position():
    texts = {f"d{i}": "text" for i in range(BATCH_MIN_ITEMS)}
    # Deliberately reversed: results arrive in any order.
    results = [
        _batch_result(doc_id, {**VERDICT, "presence": 0.1 * i})
        for i, doc_id in reversed(list(enumerate(texts)))
    ]
    batches = _StubBatches(results)
    client = SimpleNamespace(messages=_StubMessages(batches=batches))
    client.messages.batches = batches

    scored = LibertyTagger(client, poll_interval_s=0).score_many(texts)
    assert len(scored) == len(texts)
    for i, doc_id in enumerate(texts):
        assert abs(scored[doc_id].presence - 0.1 * i) < 1e-9
    assert len(batches.submitted[0]) == len(texts)


def test_documents_the_batch_drops_are_retried_inline():
    """The text only exists in memory for this run — a document that leaves
    score_many unscored can never be scored later."""
    texts = {f"d{i}": "text" for i in range(BATCH_MIN_ITEMS)}
    keys = list(texts)
    results = [_batch_result(k, VERDICT) for k in keys[:-2]]
    results.append(_batch_result(keys[-2], VERDICT, kind="errored"))
    batches = _StubBatches(results)
    client = SimpleNamespace(messages=_StubMessages(batches=batches))
    client.messages.batches = batches

    scored = LibertyTagger(client, poll_interval_s=0).score_many(texts)
    assert set(scored) == set(texts)  # the errored + missing ones were retried
    assert len(client.messages.calls) == 2  # exactly the two the batch didn't return


def test_batch_timeout_falls_back_rather_than_losing_the_run():
    texts = {f"d{i}": "text" for i in range(BATCH_MIN_ITEMS)}
    batches = _StubBatches([], status="in_progress")
    client = SimpleNamespace(messages=_StubMessages(batches=batches))
    client.messages.batches = batches

    tagger = LibertyTagger(client, batch_timeout_s=0, poll_interval_s=0)
    assert len(tagger.score_many(texts)) == len(texts)
    assert len(client.messages.calls) == len(texts)


def test_blank_texts_are_dropped_before_submission():
    client = _client()
    assert LibertyTagger(client).score_many({"a": "", "b": "   "}) == {}
    assert client.messages.calls == []


# -- build guard -----------------------------------------------------------


def test_build_tagger_needs_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert build_tagger() is None


def test_build_tagger_respects_the_disable_switch(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert build_tagger(enabled=False) is None


# -- persistence and aggregation ------------------------------------------

SCORER = "claude-liberty/claude-sonnet-5"


def _reg():
    """Both personas, each reading its own source."""
    return registry(self={"src_self": 1.0}, modeled_ce={"src_modeled_ce": 1.0})


def _store_with_liberty(values: dict[str, list[float]], untagged: int = 0):
    store = Datastore(":memory:")
    for diet, scores in values.items():
        for i, value in enumerate(scores):
            did = f"{diet}-{i}"
            store.upsert_document(
                doc_id=did,
                source_id=f"src_{diet}",
                stratum_id=None,
                url=None,
                title="t",
                published_utc=None,
                fetched_utc="2026-07-25T00:00:00+00:00",
                word_count=300,
                minhash=None,
            )
            store.upsert_scores(
                document_id=did,
                scorer=SCORER,
                foundations={},
                sentiment=0.0,
                moral_word_ratio=0.0,
                matched_words=0,
                liberty=value,
            )
        for j in range(untagged):
            did = f"{diet}-untagged-{j}"
            store.upsert_document(
                doc_id=did,
                source_id=f"src_{diet}",
                stratum_id=None,
                url=None,
                title="t",
                published_utc=None,
                fetched_utc="2026-07-25T00:00:00+00:00",
                word_count=300,
                minhash=None,
            )
    return store


def test_untagged_documents_are_excluded_not_zeroed():
    store = _store_with_liberty({"self": [0.8, 0.6]}, untagged=8)
    rows, total = store.liberty_for_sources(["src_self"], SCORER)
    assert len(rows) == 2 and total == 10
    profile = persona_liberty_profile(store, {"src_self": 1.0}, SCORER)
    assert abs(profile.mean - 0.7) < 1e-9  # not 0.14, which zero-filling would give
    assert profile.coverage == 0.2
    store.close()


def test_salient_share_counts_only_live_framing():
    store = _store_with_liberty({"self": [0.9, 0.6, 0.2, 0.0]})
    profile = persona_liberty_profile(store, {"src_self": 1.0}, SCORER)
    assert profile.salient_share == 0.5  # 0.9 and 0.6 clear the 0.5 line
    store.close()


def test_thin_coverage_is_flagged():
    store = _store_with_liberty({"self": [0.8]}, untagged=40)
    profile = persona_liberty_profile(store, {"src_self": 1.0}, SCORER)
    assert profile.coverage < LOW_COVERAGE
    assert profile.thin
    store.close()


def test_a_diet_with_nothing_tagged_reports_zero_not_a_crash():
    store = _store_with_liberty({"self": [0.8], "modeled_ce": []}, untagged=1)
    profiles = all_persona_liberty(store, _reg(), SCORER)
    assert profiles["modeled_ce"].docs_scored == 0
    assert profiles["modeled_ce"].thin
    store.close()


def test_gap_is_oriented_mine_first_not_alphabetically():
    """The gap took the first two ids in sorted order, so its sign depended on how
    the personas were spelled rather than on which one is the reader's own."""
    store = _store_with_liberty({"self": [0.2, 0.2], "modeled_ce": [0.8, 0.8]})
    g = gap(all_persona_liberty(store, _reg(), SCORER), pair())
    assert g["pair"] == ["self", "modeled_ce"]
    # `self` engages liberty less here, so mine-first makes the gap negative.
    assert abs(g["mean_gap"] + 0.6) < 1e-9
    store.close()


def test_gap_is_none_when_the_pair_is_not_both_profiled():
    assert gap({}, pair()) is None


def test_liberty_rows_do_not_pollute_the_five_way_profile():
    """The headline composition must keep running on the dictionary scorer only —
    otherwise partial liberty coverage would move the recorded JSD."""
    from ingestion.pipeline import persona_profiles

    store = _store_with_liberty({"self": [0.9]})
    assert persona_profiles(store, _reg()) == {}  # no dictionary rows exist
    store.close()
