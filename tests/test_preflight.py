"""The preflight check: every configured model, one call each, no network."""

from __future__ import annotations

from scoring.preflight import Step, main, planned_steps, run

SETTINGS = {
    "summarize": {"model": "claude-opus-5", "effort": "medium"},
    "scoring": {"taggers": {"liberty": {"model": "claude-sonnet-5", "effort": "low"}}},
    "cluster": {"themes": {"model": "claude-sonnet-5"}},
}


class _FakeMessages:
    def __init__(self, fail_on=None, exc=None):
        self.fail_on = fail_on or set()
        self.exc = exc or RuntimeError("nope")
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["model"] in self.fail_on:
            raise self.exc
        return object()


class _FakeClient:
    def __init__(self, fail_on=None, exc=None):
        self.messages = _FakeMessages(fail_on, exc)


def test_every_configured_step_is_planned():
    steps = planned_steps(SETTINGS)
    assert [s.name for s in steps] == ["summary", "liberty tagging", "blindspot themes"]
    assert [s.model for s in steps] == ["claude-opus-5"] + ["claude-sonnet-5"] * 2
    assert steps[0].effort == "medium"


def test_defaults_come_from_the_modules_not_from_here():
    """An empty settings file has to plan the same calls the pipeline makes."""
    from cluster.themes import DEFAULT_THEME_MODEL
    from scoring.liberty import DEFAULT_MODEL as LIBERTY_MODEL
    from summarize.summarizer import DEFAULT_MODEL as SUMMARY_MODEL

    models = [s.model for s in planned_steps({})]
    assert models == [SUMMARY_MODEL, LIBERTY_MODEL, DEFAULT_THEME_MODEL]


def test_each_step_is_probed_the_way_it_calls():
    """The point of the check: an SDK or a key that can't do `output_config` on
    Opus fails here rather than in tomorrow's brief."""
    client = _FakeClient()
    run(SETTINGS, client=client)
    sent = client.messages.calls
    assert sent[0]["model"] == "claude-opus-5"
    assert sent[0]["output_config"] == {"effort": "medium"}
    assert sent[1]["output_config"] == {"effort": "low"}


def test_two_steps_on_one_model_share_a_call():
    client = _FakeClient()
    results = run(SETTINGS, client=client)
    # themes has no effort, so it is a distinct call from liberty's `low`;
    # what must not happen is paying twice for the identical one.
    assert len(client.messages.calls) == len({(c["model"], str(c.get("output_config")))
                                              for c in client.messages.calls})
    assert all(r.ok for r in results)


def test_a_shared_model_reports_once_but_marks_both_steps():
    steps = [Step("a", "claude-sonnet-5", "low"), Step("b", "claude-sonnet-5", "low")]
    client = _FakeClient()
    results = run({}, client=client, steps=steps)
    assert len(client.messages.calls) == 1
    assert [r.ok for r in results] == [True, True]
    assert results[1].shared


def test_one_unreachable_model_does_not_hide_the_others():
    client = _FakeClient(fail_on={"claude-opus-5"})
    results = run(SETTINGS, client=client)
    assert not results[0].ok
    assert "RuntimeError" in results[0].detail
    assert all(r.ok for r in results[1:])


def test_an_old_sdk_is_named_as_an_upgrade_not_a_model_problem():
    client = _FakeClient(
        fail_on={"claude-opus-5"},
        exc=TypeError("create() got an unexpected keyword argument 'output_config'"),
    )
    result = run(SETTINGS, client=client)[0]
    assert not result.ok
    assert "pip install -U anthropic" in result.detail


def test_a_disabled_step_is_listed_not_probed():
    settings = {
        **SETTINGS,
        "scoring": {"taggers": {"liberty": {"enabled": False}}},
        "cluster": {"themes": {"claude": False}},
    }
    client = _FakeClient()
    results = run(settings, client=client)
    assert [c["model"] for c in client.messages.calls] == ["claude-opus-5"]
    # Disabled is a choice, so it is not a failure — but it is said out loud.
    assert all(r.ok for r in results)
    assert "enabled: false" in results[1].detail
    assert "claude: false" in results[2].detail


def test_no_client_fails_every_step_with_the_same_reason(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    results = run(SETTINGS)
    assert not any(r.ok for r in results)
    assert all("ANTHROPIC_API_KEY is not set" in r.detail for r in results)


def test_cli_reports_one_cause_per_line_and_exits_nonzero(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main([]) == 1
    out = capsys.readouterr().out
    # Three failing steps, one underlying cause: say it once.
    assert out.count("ANTHROPIC_API_KEY is not set") == 1
    assert "FAILED" in out


def test_cli_probes_an_explicit_model_instead_of_the_configured_set(monkeypatch, capsys):
    import scoring.preflight as pf

    client = _FakeClient()
    monkeypatch.setattr(pf, "build_client", lambda: (client, ""))
    assert main(["--model", "claude-haiku-4-5"]) == 0
    assert [c["model"] for c in client.messages.calls] == ["claude-haiku-4-5"]
    assert "Every configured model is reachable" in capsys.readouterr().out
