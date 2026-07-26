"""Liberty/oppression tagging via Claude.

The sixth foundation, and the one no other tagger in this project can supply.
The eMFD, the MFD, and MFD 2.0 all cover five foundations; Mformer is fine-tuned
on a corpus that does not label liberty either. ``CLAUDE.md`` §3(a) assigns it to
the LLM tagger for exactly that reason, and this module is that tagger.

**The rubric is the whole design.** Liberty is claimed by both modeled diets, in
different registers. Freedom from state coercion, mandates, and regulatory
overreach is liberty language; so is freedom from corporate domination,
surveillance, and constraints on bodily autonomy. A rubric that recognized only
one register would score one diet as caring about liberty and the other as
silent — a measurement artifact indistinguishable from a finding, and precisely
the failure the fairness lexicon audit caught in ``scoring/fairness_split.py``.
So the rubric names both registers explicitly, and instructs that neither is more
truly liberty than the other.

**Grounding.** The model must return a verbatim quote alongside its judgment. The
quote is not persisted (see below) — its job is to force the judgment to point at
something in the text rather than at a vibe. Scores where the model cannot
produce a quote are more likely to be projection.

**What is not stored.** Only the derived score reaches the datastore. Rationales
and quotes are surfaced during validation, where the gold-set texts are on disk
anyway, but production scores are deliberately not individually auditable: a
persisted quote is persisted article text, which §0 forbids. That is a real
tradeoff against ``CLAUDE.md`` §3(a)'s auditability goal, and it is recorded in
``LIMITATIONS.md`` rather than quietly resolved.

Cost shape: the rubric is a stable prefix, so it is cached; the daily path uses
the Batch API (half price, and the daily run is an overnight batch job by
design). Effort is ``low`` — this is judgment against a fixed rubric, not
open-ended reasoning, and Sonnet 5 runs adaptive thinking by default at ``high``
effort, which would bill several times the output tokens for no gain here.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_EFFORT = "low"

# Enough article to judge framing without paying for the whole piece. Liberty
# framing, when present, is rarely withheld until the last paragraph.
DEFAULT_MAX_CHARS = 6000

# Below this, the Batch API's overhead (submit, poll, collect) is not worth the
# 50% saving — just call synchronously.
BATCH_MIN_ITEMS = 10

# Batches usually finish well inside an hour; this is the ceiling before the
# remainder falls back to synchronous calls.
DEFAULT_BATCH_TIMEOUT_S = 3600
DEFAULT_POLL_INTERVAL_S = 20

SYSTEM_PROMPT = """\
You are a careful annotator applying Moral Foundations Theory to news and \
commentary. You judge one foundation only: LIBERTY / OPPRESSION.

DEFINITION
Liberty/oppression covers intuitions about freedom from domination, coercion, \
and illegitimate control over one's own life, body, property, speech, or \
conscience. Its virtue pole is freedom, autonomy, and self-determination. Its \
vice pole is oppression: domination, coercion, tyranny, and the concentration \
of power over others.

BOTH REGISTERS COUNT — THIS IS THE MOST IMPORTANT INSTRUCTION
Liberty language appears across the political spectrum in different registers. \
Both of the following are fully liberty-framed, and neither is more truly \
liberty than the other:

- Freedom from state or collective coercion: government overreach, mandates, \
  lockdowns, regulatory burden, censorship, confiscation, surveillance by the \
  state, conscience and religious-liberty claims, parental authority against \
  official direction, the right to refuse.
- Freedom from private or structural domination: corporate or monopoly power \
  over individuals, employer control over workers, bodily autonomy and \
  reproductive self-determination, civil liberties and voting access, \
  surveillance by companies, debt or contract terms that trap people, \
  incarceration and policing as domination.

If you find yourself scoring one of these registers systematically higher than \
the other, you are measuring your own priors rather than the text. Judge the \
framing the author actually uses.

WHAT DOES NOT COUNT
- Fairness claims about equal treatment or proportional desert (that is \
  fairness, not liberty).
- Harm or suffering with no element of domination or control (that is care).
- Mere mention of the words "freedom" or "rights" in a slogan, brand, proper \
  noun, or boilerplate, with no substantive liberty claim in the argument.
- Purely procedural or descriptive reporting that recounts a dispute without \
  framing anything as coercion or as freedom.

HOW TO SCORE
- presence: 0.0 to 1.0 — how strongly the text engages the liberty/oppression \
  foundation. 0.0 means absent. Below 0.5 means incidental or ambiguous. Above \
  0.5 means liberty is a live moral frame in the piece. Reserve values above \
  0.85 for text substantially organized around a liberty claim.
- pole: "virtue" when the text valorizes freedom or autonomy; "vice" when it \
  condemns oppression, domination, or coercion; "mixed" when it does both or \
  when the framing is genuinely balanced; "none" when presence is 0.
- register: "from_state", "from_private_power", "both", or "none" — which of \
  the two registers above the text uses. This is descriptive, not evaluative.
- quote: a verbatim span from the text, at most 25 words, that best evidences \
  your judgment. If you cannot find one, return an empty string and score \
  presence at or near 0 — a liberty reading you cannot point at is a reading \
  you are supplying yourself.
- rationale: one sentence, in your own words, explaining the judgment.

Be conservative. Most news text does not substantially engage liberty. A high \
false-positive rate makes the foundation useless as a signal.
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "presence": {"type": "number"},
        "pole": {"type": "string", "enum": ["virtue", "vice", "mixed", "none"]},
        "register": {
            "type": "string",
            "enum": ["from_state", "from_private_power", "both", "none"],
        },
        "quote": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["presence", "pole", "register", "quote", "rationale"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LibertyScore:
    """One document's liberty judgment.

    ``quote`` and ``rationale`` are carried here for validation and debugging;
    the persistence layer stores only ``presence`` and ``pole``.
    """

    presence: float           # [0, 1]
    pole: str                 # virtue | vice | mixed | none
    register: str             # from_state | from_private_power | both | none
    quote: str
    rationale: str
    model: str

    @property
    def grounded(self) -> bool:
        """Did the model point at text? An ungrounded high score is suspect."""
        return bool(self.quote.strip())

    @property
    def signed(self) -> float:
        """Presence signed by pole, for callers that want a net virtue-vice value."""
        if self.pole == "virtue":
            return self.presence
        if self.pole == "vice":
            return -self.presence
        return 0.0


def _user_prompt(text: str, max_chars: int) -> str:
    body = text.strip()[:max_chars]
    return f"Judge the liberty/oppression foundation in this text.\n\n<text>\n{body}\n</text>"


def _parse(payload: str, model: str) -> LibertyScore | None:
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return None
    try:
        presence = float(data["presence"])
    except (KeyError, TypeError, ValueError):
        return None
    # Clamp rather than reject: a schema-valid response with a stray 1.2 is a
    # usable judgment, and discarding it would silently thin coverage.
    presence = min(1.0, max(0.0, presence))
    return LibertyScore(
        presence=presence,
        pole=str(data.get("pole", "none")),
        register=str(data.get("register", "none")),
        quote=str(data.get("quote", "")),
        rationale=str(data.get("rationale", "")),
        model=model,
    )


class LibertyTagger:
    """Score documents on liberty/oppression with Claude.

    ``score`` handles one document synchronously. ``score_many`` prefers the
    Batch API — half price, and the caller is an overnight job — falling back to
    synchronous calls for anything the batch does not return.
    """

    def __init__(
        self,
        client,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_tokens: int = 2000,
        use_batch: bool = True,
        batch_timeout_s: int = DEFAULT_BATCH_TIMEOUT_S,
        poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.use_batch = use_batch
        self.batch_timeout_s = batch_timeout_s
        self.poll_interval_s = poll_interval_s

    @property
    def name(self) -> str:
        """Scorer name recorded alongside the scores, so a model change is visible."""
        return f"claude-liberty/{self.model}"

    def _params(self, text: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # The rubric is a stable prefix across every call — cache it.
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": _user_prompt(text, self.max_chars)}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": _SCHEMA},
            },
        }

    def score(self, text: str) -> LibertyScore | None:
        """Score one document, or ``None`` if the call failed or was declined."""
        if not text or not text.strip():
            return None
        try:
            response = self.client.messages.create(**self._params(text))
        except Exception as exc:
            logger.warning("liberty tagging failed (%s: %s)", type(exc).__name__, exc)
            return None
        return self._from_response(response)

    def _from_response(self, response) -> LibertyScore | None:
        # Safety classifiers can decline; content is then empty or partial, so
        # this has to be checked before indexing into it.
        if getattr(response, "stop_reason", None) == "refusal":
            logger.warning("liberty tagging refused by safety classifiers")
            return None
        payload = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return _parse(payload, self.model)

    def score_many(self, texts: dict[str, str]) -> dict[str, LibertyScore]:
        """Score a batch of ``{document_id: text}``.

        Returns only the documents that scored. Missing keys are unscored, which
        callers must keep distinct from a zero — an untagged document is not a
        document with no liberty content.
        """
        usable = {k: v for k, v in texts.items() if v and v.strip()}
        if not usable:
            return {}
        if not self.use_batch or len(usable) < BATCH_MIN_ITEMS:
            return {k: s for k, v in usable.items() if (s := self.score(v)) is not None}

        try:
            scored = self._score_batch(usable)
        except Exception as exc:
            logger.warning(
                "liberty batch failed (%s: %s) — falling back to synchronous calls",
                type(exc).__name__, exc,
            )
            scored = {}

        # Anything the batch didn't return (errored, expired, or timed out) is
        # retried synchronously while the text is still in memory. It cannot be
        # retried later: raw text is never persisted (§0), so a document that
        # leaves this function unscored stays unscored forever.
        missing = {k: v for k, v in usable.items() if k not in scored}
        if missing:
            logger.info("liberty: %d document(s) not returned by batch, retrying inline",
                        len(missing))
            for k, v in missing.items():
                score = self.score(v)
                if score is not None:
                    scored[k] = score
        return scored

    def _score_batch(self, texts: dict[str, str]) -> dict[str, LibertyScore]:
        requests = [
            {"custom_id": doc_id, "params": self._params(text)}
            for doc_id, text in texts.items()
        ]
        batch = self.client.messages.batches.create(requests=requests)
        logger.info("liberty: submitted batch %s (%d documents)", batch.id, len(requests))

        deadline = time.monotonic() + self.batch_timeout_s
        while True:
            current = self.client.messages.batches.retrieve(batch.id)
            if current.processing_status == "ended":
                break
            if time.monotonic() >= deadline:
                logger.warning("liberty: batch %s still running after %ds — giving up on it",
                               batch.id, self.batch_timeout_s)
                return {}
            time.sleep(self.poll_interval_s)

        out: dict[str, LibertyScore] = {}
        for result in self.client.messages.batches.results(batch.id):
            # Results arrive in any order — key by custom_id, never by position.
            if result.result.type != "succeeded":
                continue
            score = self._from_response(result.result.message)
            if score is not None:
                out[result.custom_id] = score
        return out


def build_tagger(
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    use_batch: bool = True,
    enabled: bool = True,
    **kwargs,
) -> LibertyTagger | None:
    """The tagger, or ``None`` when it can't run.

    Mirrors ``_build_transformer``: a missing key or package degrades to "no
    liberty scores" rather than failing the run, because the rest of the
    pipeline is still worth completing without the sixth foundation.
    """
    if not enabled:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("liberty tagging skipped — no ANTHROPIC_API_KEY set")
        return None
    try:
        import anthropic
    except ImportError:
        logger.info("liberty tagging skipped — `anthropic` not installed")
        return None
    return LibertyTagger(
        anthropic.Anthropic(), model=model, effort=effort, use_batch=use_batch, **kwargs
    )
