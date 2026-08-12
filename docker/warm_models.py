"""Populate the Hugging Face cache at image build time.

Mformer is one binary classifier per foundation — five RoBERTa models plus a
shared tokenizer (``scoring/transformer.py``). Left to the run, they download on
first use, which puts a Hugging Face outage on the critical path of a scheduled
job and needs a token in the task's environment to raise the rate limit.

Baking them in trades ~500 MB of image for a task start that touches no network
and needs no ``HF_TOKEN``. The image sets ``HF_HUB_OFFLINE=1`` afterwards, so a
model this script failed to fetch fails loudly at build time rather than
silently degrading a run later — the transformer tagger is optional and
``ingestion/pipeline.py`` logs and continues without it, which is the right
behaviour for a workstation and the wrong one for an image built to include it.

Deliberately reads the same settings the scorer does — model prefix and
revision — rather than repeating them, so the cache and the runtime cannot
disagree about what to load. Baking one revision while the run asks for another
would download the second at task start, which quietly undoes the point of
baking anything, and ``HF_HUB_OFFLINE=1`` would then turn that into a failure
instead.

An unpinned revision resolves to whatever the hub's default branch points at
*on the day the image is built*, so two builds of identical source can carry
different weights and score the same corpus differently. Pin
``scoring.taggers.transformer.revision`` in settings — ``settings.example.yaml``
already recommends it — and both this and the run will honour it.
"""

from __future__ import annotations

import sys

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ingestion.config import load_settings
from scoring.foundations import CLASSIC_FOUNDATIONS
from scoring.transformer import resolve_model_prefix


def main() -> int:
    settings = load_settings()
    transformer = ((settings.get("scoring") or {}).get("taggers") or {}).get("transformer") or {}
    prefix = resolve_model_prefix(transformer.get("model"))
    revision = transformer.get("revision")

    if revision:
        print(f"revision   {revision}", flush=True)
    else:
        print(
            "revision   UNPINNED — baking whatever the hub's default branch points at "
            "today. Set scoring.taggers.transformer.revision for a reproducible image.",
            flush=True,
        )

    tokenizer_id = f"{prefix}{CLASSIC_FOUNDATIONS[0]}"
    print(f"tokenizer  {tokenizer_id}", flush=True)
    AutoTokenizer.from_pretrained(tokenizer_id, revision=revision)

    for foundation in CLASSIC_FOUNDATIONS:
        model_id = f"{prefix}{foundation}"
        print(f"model      {model_id}", flush=True)
        AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)

    print(f"cached {len(CLASSIC_FOUNDATIONS)} models + 1 tokenizer", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
