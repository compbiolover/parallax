"""Hand-coded gold set: schema and loader.

The gold set is committed as JSON (small, auditable, and — unlike the gitignored
``*.jsonl``/``*.csv`` — trackable). Each item carries a short text and binary
presence labels over the five classic foundations, following MFRC-style
annotation (does this text *invoke* the foundation, virtue or vice). Only
derived labels and links are stored — never scraped article bodies at scale; the
gold texts are short excerpts coded by hand.

Schema (``validation/gold/*.json``)::

    {
      "version": 1,
      "guidelines": "one-line pointer to the coding rubric",
      "coders": ["author"],
      "items": [
        {"id": "g001", "text": "…", "source": "hand",
         "labels": {"care": 1, "fairness": 0, "loyalty": 0,
                    "authority": 0, "sanctity": 0}}
      ]
    }

``labels_by_coder`` (optional, per item) enables inter-coder reliability once the
set grows past a single annotator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from scoring.foundations import CLASSIC_FOUNDATIONS

GOLD_DIR = Path(__file__).resolve().parent / "gold"


@dataclass(frozen=True)
class GoldItem:
    id: str
    text: str
    labels: dict[str, int]                      # foundation -> 0/1
    source: str = "hand"
    labels_by_coder: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldSet:
    version: int
    coders: list[str]
    items: list[GoldItem]
    guidelines: str = ""

    def gold_column(self, foundation: str) -> list[int]:
        """Binary gold labels for one foundation, in item order."""
        return [int(it.labels.get(foundation, 0)) for it in self.items]


def load_gold(path: str | Path) -> GoldSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[GoldItem] = []
    for raw in data["items"]:
        labels = {f: int(raw["labels"].get(f, 0)) for f in CLASSIC_FOUNDATIONS}
        items.append(
            GoldItem(
                id=raw["id"],
                text=raw["text"],
                labels=labels,
                source=raw.get("source", "hand"),
                labels_by_coder=raw.get("labels_by_coder", {}),
            )
        )
    return GoldSet(
        version=int(data.get("version", 1)),
        coders=list(data.get("coders", [])),
        items=items,
        guidelines=data.get("guidelines", ""),
    )
