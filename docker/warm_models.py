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

Deliberately imports the same constants the scorer uses rather than repeating
the model names, so a change to either cannot leave the cache and the runtime
disagreeing about what to load.
"""

from __future__ import annotations

import sys

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scoring.foundations import CLASSIC_FOUNDATIONS
from scoring.transformer import DEFAULT_PREFIX


def main() -> int:
    tokenizer_id = f"{DEFAULT_PREFIX}{CLASSIC_FOUNDATIONS[0]}"
    print(f"tokenizer  {tokenizer_id}", flush=True)
    AutoTokenizer.from_pretrained(tokenizer_id)

    for foundation in CLASSIC_FOUNDATIONS:
        model_id = f"{DEFAULT_PREFIX}{foundation}"
        print(f"model      {model_id}", flush=True)
        AutoModelForSequenceClassification.from_pretrained(model_id)

    print(f"cached {len(CLASSIC_FOUNDATIONS)} models + 1 tokenizer", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
