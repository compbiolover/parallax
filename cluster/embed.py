"""Per-document embeddings, computed at ingestion.

Parallax discards raw text at ingestion, so embeddings must be produced in the
same pass that scores a document and then persisted as a derived metric — the
clustering stage never sees the text again. That forces a **corpus-independent,
per-document** embedder (no global IDF fit).

Two implementations:

- :class:`HashingEmbedder` (default) — a dependency-free, deterministic feature
  hasher over word unigrams+bigrams (numpy only). Lower quality than a neural
  encoder, but it always runs and needs nothing external. Deterministic across
  runs because it hashes with blake2b, not Python's salted ``hash()``.
- :class:`SentenceTransformerEmbedder` (optional) — the quality path
  (``all-MiniLM-L6-v2`` etc.), loaded lazily so ``sentence-transformers`` /
  ``torch`` are only needed when actually selected.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _features(text: str) -> list[str]:
    toks = _tokens(text)
    feats = list(toks)
    feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]  # bigrams
    return feats


def _h(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")


class HashingEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams), L2-normalized."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hashing(d={self.dim})"

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature in _features(text):
            hv = _h(feature)
            idx = hv % self.dim
            sign = 1.0 if (hv >> 63) & 1 else -1.0  # signed hashing reduces collision bias
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()


class SentenceTransformerEmbedder:
    """Neural sentence embeddings (quality path). Requires sentence-transformers.

    Default is ``thenlper/gte-small`` — the best simple performer in the embedder
    benchmark (top-tier quality, no prompt needed, MiniLM-sized; see
    LIMITATIONS.md). ``query_prefix`` prepends an instruction to every text:
    instruction-tuned families need it to score well (bge: "Represent this
    sentence for searching relevant passages: "; e5: "query: "). Plain models
    (gte, MiniLM, mpnet) take no prefix.
    """

    def __init__(self, model: str = "thenlper/gte-small", query_prefix: str = "") -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self.model_name = model
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self.model_name}"

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(self.query_prefix + text, normalize_embeddings=True)
        return [float(x) for x in vec]


def build_embedder(settings: dict | None = None) -> tuple[Embedder, str]:
    """Build the configured embedder and its provenance name.

    settings.cluster.embedder:
      kind: hashing | sentence-transformers   (default: hashing)
      dim:  hashing dimensionality            (default: 512)
      model: sentence-transformers model id   (default: all-MiniLM-L6-v2)
    """
    cfg = ((settings or {}).get("cluster", {}) or {}).get("embedder", {}) or {}
    kind = cfg.get("kind", "hashing")
    if kind == "sentence-transformers":
        emb = SentenceTransformerEmbedder(
            cfg.get("model", "thenlper/gte-small"),
            query_prefix=cfg.get("query_prefix", ""),
        )
        return emb, emb.name
    emb = HashingEmbedder(int(cfg.get("dim", 512)))
    return emb, emb.name
