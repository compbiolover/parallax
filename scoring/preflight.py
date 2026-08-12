"""Check every Claude model this project will actually call, in one command.

``python -m scoring.liberty`` proves the key, the SDK and the rubric work —
against the liberty model. The daily run calls three: Opus for the summary,
Sonnet for liberty tagging, Sonnet for blindspot theming, each configurable
separately in ``settings.yaml``. So a probe that greenlights one of them is a
probe that can pass while the morning brief still comes out numbers-only,
which is the exact confusion this exists to end.

Each step is probed the way it *calls* — same model id, same effort, and the
same parameter shape, down to liberty's block-list system prompt and its JSON
schema — because the failures that matter here are per-model and per-parameter.
A key with no access to Opus, a model id that no longer resolves, a model that
won't take a schema, an ``anthropic`` too old to accept ``output_config``: none
of those show up in a probe of a different model or a simpler call, and all of
them end the same way, in a fallback that used to blame the key.

What it checks is reachability and parameter shape, not judgment: a stand-in
system prompt and one word back, not the rubric and a scored document.
``python -m scoring.liberty`` is still the end-to-end check of the rubric and
the parser, and it is what to run once this one is green.

Costs a few hundred tokens for the lot — the prompts are one line each, and the
only real weight is the schema on the one call that sends one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .claude_client import Reason, build_client, call_failed

# Short enough that thinking models can't spend the budget before answering,
# and short enough that the whole check costs a rounding error.
_PROMPT = "Reply with the single word: ok"
_MAX_TOKENS = 16

# A schema-bearing call has to have room to satisfy the schema; 16 tokens would
# truncate mid-object and report a model failure that is really a probe bug.
_SCHEMA_MAX_TOKENS = 400

# Stands in for the step's real system prompt. The *shape* is what a probe can
# usefully check — a string where the step sends a string, a block list where it
# sends blocks — and sending the genuine rubric would cost real input tokens
# every time to test nothing extra.
_SYSTEM = "You are a reachability probe. Answer as briefly as possible."


@dataclass(frozen=True)
class Step:
    """A place the pipeline calls Claude, and what it calls with.

    The flags describe the *shape* of the call, which is a distinct thing to get
    wrong from the model id. Structured output and block-list system prompts are
    per-model capabilities and per-version SDK features, so a probe that sent a
    bare message to every step could go green while the step that asks for JSON
    fails on the same model.
    """

    name: str
    model: str
    effort: str | None = None
    # Liberty sends its rubric as a block list (it cache-controls the prefix);
    # the summary and themes send a plain string.
    system_blocks: bool = False
    # Liberty constrains the response with `output_config.format` and a JSON
    # schema. Nothing else does.
    structured: bool = False
    # Set when the step is configured off. It is still listed — "liberty is
    # disabled" is a different answer from "liberty is broken", and only one of
    # them is worth acting on.
    disabled: str = ""

    @property
    def key(self) -> tuple:
        """What makes two steps the same call. Shape included: the same model
        can accept a plain message and reject a schema."""
        return (self.model, self.effort, self.system_blocks, self.structured)


@dataclass(frozen=True)
class Result:
    step: Step
    ok: bool
    # The failure, or "" when it worked. A ``Reason`` where one is available,
    # which keeps the actionable sentence for the fix list and the short phrase
    # for the table — a paragraph per row makes a three-row table unreadable.
    detail: str = ""
    millis: int = 0
    shared: bool = False  # answered by another step's call, not its own

    @property
    def short(self) -> str:
        return getattr(self.detail, "short", None) or self.detail


def planned_steps(settings: dict) -> list[Step]:
    """The Claude calls the configured pipeline would make.

    Defaults come from the modules themselves rather than being restated here:
    a model default that drifts in one place and not the other would make this
    check confidently wrong, which is worse than not having it.
    """
    from cluster.themes import DEFAULT_THEME_EFFORT, DEFAULT_THEME_MODEL
    from scoring.liberty import DEFAULT_EFFORT as LIBERTY_EFFORT
    from scoring.liberty import DEFAULT_MODEL as LIBERTY_MODEL
    from summarize.summarizer import DEFAULT_EFFORT as SUMMARY_EFFORT
    from summarize.summarizer import DEFAULT_MODEL as SUMMARY_MODEL

    summarize = settings.get("summarize", {}) or {}
    taggers = (settings.get("scoring", {}) or {}).get("taggers", {}) or {}
    liberty = taggers.get("liberty", {}) or {}
    themes = (settings.get("cluster", {}) or {}).get("themes", {}) or {}

    return [
        Step(
            "summary",
            summarize.get("model") or SUMMARY_MODEL,
            summarize.get("effort") or SUMMARY_EFFORT,
        ),
        Step(
            "liberty tagging",
            liberty.get("model") or LIBERTY_MODEL,
            liberty.get("effort") or LIBERTY_EFFORT,
            system_blocks=True,
            structured=True,
            disabled="" if liberty.get("enabled", True) else "enabled: false",
        ),
        Step(
            "blindspot themes",
            themes.get("model") or DEFAULT_THEME_MODEL,
            # `.get` with a default, not `or`: theming is the one step that can
            # be configured to send no effort at all (`effort: ~`), and a probe
            # that quietly substituted the default would be checking a call the
            # run doesn't make — the one thing this command must not do.
            themes.get("effort", DEFAULT_THEME_EFFORT),
            disabled="" if themes.get("claude", True) else "claude: false",
        ),
    ]


def probe(client, step: Step) -> Result:
    """One minimal call in the step's own shape, timed, failures named.

    The schema is liberty's real one rather than a toy: a schema the model or
    the API rejects is exactly the failure this is here to surface, and a
    simplified stand-in would be accepted where the real one might not.
    """
    kwargs = {
        "model": step.model,
        "max_tokens": _SCHEMA_MAX_TOKENS if step.structured else _MAX_TOKENS,
        "system": ([{"type": "text", "text": _SYSTEM}] if step.system_blocks else _SYSTEM),
        "messages": [{"role": "user", "content": _PROMPT}],
    }
    output_config = {}
    if step.effort:
        output_config["effort"] = step.effort
    if step.structured:
        from .liberty import _SCHEMA

        output_config["format"] = {"type": "json_schema", "schema": _SCHEMA}
    if output_config:
        kwargs["output_config"] = output_config
    started = time.monotonic()
    try:
        client.messages.create(**kwargs)
    except TypeError as exc:
        # An SDK that doesn't know a parameter the pipeline sends. Worth its own
        # sentence: it reads as a model problem, it is fixed by an upgrade, and
        # it will fail identically every morning until someone runs one.
        return Result(
            step,
            False,
            Reason(
                "the installed `anthropic` SDK is too old",
                f"the installed `anthropic` SDK rejected a parameter this step "
                f"sends ({exc}) — upgrade it with `pip install -U anthropic`.",
            ),
        )
    except Exception as exc:
        return Result(step, False, call_failed(exc))
    return Result(step, True, millis=int((time.monotonic() - started) * 1000))


def _skipped(step: Step) -> Result:
    return Result(step, True, f"skipped ({step.disabled})")


def run(settings: dict, client=None, steps: list[Step] | None = None) -> list[Result]:
    """Probe each configured step, calling once per distinct call shape.

    Two steps that would make the same call share one. Reachability is a
    property of the model and the shape, not of the step, and paying twice to
    learn the same fact is the kind of thing a daily habit shouldn't do.
    """
    steps = steps if steps is not None else planned_steps(settings)
    if client is None:
        client, reason = build_client()
        if client is None:
            # A step that is switched off is still skipped, not failed. Without
            # a key that is *most* of what this prints, and calling a deliberate
            # `enabled: false` a failure sends someone to fix a setting they
            # chose on purpose.
            return [_skipped(s) if s.disabled else Result(s, False, reason) for s in steps]

    seen: dict[tuple, Result] = {}
    results: list[Result] = []
    for step in steps:
        if step.disabled:
            results.append(_skipped(step))
            continue
        prior = seen.get(step.key)
        if prior is not None:
            results.append(Result(step, prior.ok, prior.detail, shared=True))
            continue
        result = probe(client, step)
        seen[step.key] = result
        results.append(result)
    return results


def _format(results: list[Result]) -> str:
    width = max((len(r.step.name) for r in results), default=0)
    lines = []
    for r in results:
        effort = f"effort={r.step.effort}" if r.step.effort else ""
        if r.step.disabled:
            status = f"— {r.short}"
        elif not r.ok:
            status = f"FAILED — {r.short}"
        elif r.shared:
            status = "ok (same model as above, one call)"
        else:
            status = f"ok ({r.millis} ms)"
        lines.append(f"  {r.step.name:<{width}}  {r.step.model:<18} {effort:<14} {status}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m scoring.preflight`` — every configured model, one command."""
    import argparse

    from ingestion.config import load_settings

    parser = argparse.ArgumentParser(
        prog="scoring.preflight",
        description="Check every Claude model the configured pipeline will call",
    )
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="ID",
        help="probe this model instead of the configured set (repeatable)",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.settings)
    steps = (
        [Step(f"--model {m}", m) for m in args.models] if args.models else planned_steps(settings)
    )
    distinct = len({s.key for s in steps if not s.disabled})
    print(f"Checking {len(steps)} step(s), {distinct} distinct model call(s)...\n")

    results = run(settings, steps=steps)
    print(_format(results))

    failed = [r for r in results if not r.ok]
    if not failed:
        print(
            "\nEvery configured model is reachable. That is the plumbing, not "
            "the judgment —\n`python3 -m scoring.liberty` scores a real probe "
            "through the rubric and the parser."
        )
        return 0
    # One line per distinct cause: three steps failing on one missing key is one
    # problem, and printing it three times reads like three.
    print(f"\n{len(failed)} step(s) cannot run. What to fix:\n")
    for detail in dict.fromkeys(r.detail for r in failed):
        print(f"  - {detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
