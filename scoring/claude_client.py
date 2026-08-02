"""Building the Anthropic client, and saying why when it can't be built.

Two things stop a Claude-backed tagger from running, and both are ordinary
setup mistakes rather than exceptional conditions: the key is not in the
environment, or the ``anthropic`` package is not installed (it lives in the
``llm`` extra, which ``pip install -e ".[dev]"`` does not pull in).

Both used to fail the same way — return ``None``, log at INFO, and leave the
caller with a pipeline that quietly produced no scores. INFO is below the
default threshold, so in practice the reason was invisible: you got a working
run with a missing foundation and nothing to explain it.

So this module returns the *reason* alongside the client, and callers log it at
WARNING. The distinction matters more here than for most optional dependencies,
because a silently-absent liberty score looks exactly like a genuine absence of
liberty content in the corpus.
"""

from __future__ import annotations

import os


class Reason(str):
    """Why Claude can't run: a sentence for the log, a phrase for the page.

    The long form is what someone reading a log needs — which piece of setup is
    missing and the command that fixes it. It is far too long for a card on the
    dashboard, which is where a fallback is actually noticed, so each reason
    also carries a `short` phrase that fits inline in one.

    A ``str`` subclass so the long form stays the default in every existing
    use — ``%s`` in a log call, an ``in`` check, equality against ``NO_KEY``.
    """

    short: str

    def __new__(cls, short: str, detail: str) -> Reason:
        obj = super().__new__(cls, detail)
        obj.short = short
        return obj


NO_KEY = Reason(
    "no ANTHROPIC_API_KEY set",
    "ANTHROPIC_API_KEY is not set in this environment. Export it in your shell "
    "(and note that .env is NOT read automatically). For scheduled runs, set it "
    "inside the crontab — cron does not inherit your shell environment.",
)

NO_PACKAGE = Reason(
    "the `anthropic` package is not installed",
    "the `anthropic` package is not installed. It lives in the `llm` extra, "
    'which `pip install -e ".[dev]"` does not include — install it with '
    '`pip install -e ".[dev,llm]"`.',
)


def call_failed(exc: BaseException) -> Reason:
    """The reason for a key that *is* set and a call that failed anyway.

    Naming the exception type is the whole value: an ``AuthenticationError``
    means the key is wrong or revoked, a ``NotFoundError`` means the model name
    is, and a timeout means neither — three different mornings that otherwise
    produce the same numbers-only page.
    """
    return Reason(
        f"the Claude call failed ({type(exc).__name__})",
        f"the Claude call failed ({type(exc).__name__}: {exc})",
    )


NO_TEXT = Reason(
    "Claude returned no summary text",
    "Claude returned no usable text — the response was empty, refused, or "
    "spent its whole token budget on thinking.",
)

# What to say when a caller has no reason to hand — a client that was injected
# and turned out to be None, say. Better than an empty parenthetical.
UNKNOWN = Reason("the LLM step did not run", "the LLM step did not run")


def build_client() -> tuple[object | None, str]:
    """Return ``(client, reason)``.

    ``reason`` is empty on success, and an actionable sentence otherwise. It is
    deliberately a return value rather than an exception: a missing key should
    degrade the run to fewer foundations, not abort the ingest that already
    fetched and scored everything else.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, NO_KEY
    try:
        import anthropic
    except ImportError:
        return None, NO_PACKAGE
    return anthropic.Anthropic(), ""


def describe_availability() -> str:
    """One line on whether Claude-backed scoring can run here, for diagnostics."""
    client, reason = build_client()
    return "ready" if client is not None else reason
