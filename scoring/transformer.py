"""Transformer moral-foundations tagger (Mformer).

The dictionary baseline has poor convergent validity on the binding foundations
(the validation harness fires the §5 trigger on the real eMFD). Mformer is a set
of five fine-tuned RoBERTa binary classifiers — one per classic foundation —
that do meaningfully better in-domain. This wraps them behind the same
``text -> {foundation: score}`` interface the dictionary and evaluation use.

Heavy: requires ``transformers`` + ``torch`` (``pip install parallax[scoring]``)
and downloads ~5 RoBERTa models on first use, so it is imported lazily and only
constructed when actually selected. ``predict_fn`` lets tests inject a stub so
the aggregation is testable without the models.

Supply chain: models are fetched from the Hugging Face hub (a third-party
account by default), so **pin ``revision``** (a commit hash or tag) for
reproducibility and safety, and prefer models that ship ``safetensors``. This
code never sets ``trust_remote_code``, so a model cannot execute bundled code on
load.

Coverage note: like the dictionary, Mformer covers the five CLASSIC foundations
only — liberty/oppression is left to the Claude tagger.
"""

from __future__ import annotations

from collections.abc import Callable

from .foundations import CLASSIC_FOUNDATIONS

DEFAULT_PREFIX = "joshnguyen/mformer-"


def _positive_index(id2label: dict, foundation: str) -> int:
    """Index of the 'foundation present' class in a binary classifier's labels.

    Robust across labelling conventions: prefer a label naming the foundation
    (``care``), else a label not prefixed ``not`` (``not care`` -> the other),
    else the conventional positive index ``1`` — never silently defaulting to 0,
    which would invert scores on a model with generic labels (``LABEL_0/1``).
    """
    items = [(int(i), str(lbl).lower()) for i, lbl in id2label.items()]
    # 1. a label naming the foundation, not negated ("care").
    for i, lbl in items:
        if foundation in lbl and not lbl.startswith("not"):
            return i
    # 2. only if some label is explicitly negated ("not care") is the non-negated
    #    one meaningfully the positive class.
    if any(lbl.startswith("not") for _, lbl in items):
        for i, lbl in items:
            if not lbl.startswith("not"):
                return i
    # 3. generic labels (LABEL_0/1) — conventional positive index 1, never 0.
    return 1 if any(i == 1 for i, _ in items) else items[0][0]


class TransformerScorer:
    """Per-foundation Mformer scorer. Returns P(foundation present) in [0, 1]."""

    def __init__(
        self,
        model_prefix: str = DEFAULT_PREFIX,
        max_length: int = 256,
        revision: str | None = None,
        predict_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.model_prefix = model_prefix
        self.max_length = max_length
        self.revision = revision
        self._predict_fn = predict_fn
        if predict_fn is not None:
            return
        # Lazy heavy load — one RoBERTa per foundation, shared tokenizer.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            f"{model_prefix}{CLASSIC_FOUNDATIONS[0]}", revision=revision
        )
        self._models: dict[str, object] = {}
        self._pos_index: dict[str, int] = {}
        for foundation in CLASSIC_FOUNDATIONS:
            model = AutoModelForSequenceClassification.from_pretrained(
                f"{model_prefix}{foundation}", revision=revision
            )
            model.eval()
            self._models[foundation] = model
            self._pos_index[foundation] = _positive_index(model.config.id2label, foundation)

    @property
    def name(self) -> str:
        suffix = f"@{self.revision}" if self.revision else ""
        return f"transformer/{self.model_prefix.rstrip('-/')}{suffix}"

    def score(self, text: str) -> dict[str, float]:
        """Return {foundation: P(present)} over the five classic foundations."""
        if self._predict_fn is not None:
            return {f: float(self._predict_fn(f, text)) for f in CLASSIC_FOUNDATIONS}

        torch = self._torch
        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        out: dict[str, float] = {}
        with torch.no_grad():
            for foundation, model in self._models.items():
                probs = torch.softmax(model(**enc).logits, dim=-1)[0]
                out[foundation] = float(probs[self._pos_index[foundation]])
        return out
