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

Coverage note: like the dictionary, Mformer covers the five CLASSIC foundations
only — liberty/oppression is left to the Claude tagger.
"""

from __future__ import annotations

from collections.abc import Callable

from .foundations import CLASSIC_FOUNDATIONS

DEFAULT_PREFIX = "joshnguyen/mformer-"


class TransformerScorer:
    """Per-foundation Mformer scorer. Returns P(foundation present) in [0, 1]."""

    def __init__(
        self,
        model_prefix: str = DEFAULT_PREFIX,
        max_length: int = 256,
        predict_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.model_prefix = model_prefix
        self.max_length = max_length
        self._predict_fn = predict_fn
        if predict_fn is not None:
            return
        # Lazy heavy load — one RoBERTa per foundation, shared tokenizer.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(f"{model_prefix}{CLASSIC_FOUNDATIONS[0]}")
        self._models: dict[str, object] = {}
        self._pos_index: dict[str, int] = {}
        for foundation in CLASSIC_FOUNDATIONS:
            model = AutoModelForSequenceClassification.from_pretrained(f"{model_prefix}{foundation}")
            model.eval()
            self._models[foundation] = model
            # Positive class = the label not prefixed "not" (id2label e.g. {0:'not care',1:'care'}).
            id2label = model.config.id2label
            self._pos_index[foundation] = next(
                i for i, lbl in id2label.items() if not str(lbl).lower().startswith("not")
            )

    @property
    def name(self) -> str:
        return f"transformer/{self.model_prefix.rstrip('-/')}"

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
