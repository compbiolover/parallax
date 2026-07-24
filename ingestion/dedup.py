"""Deduplication: exact content hashing + MinHash near-duplicate detection.

Syndicated wire copy repeats heavily across outlets; without dedup it skews
every aggregate (``CLAUDE.md``). Two layers:

- **Exact**: a sha256 over normalized text. Identical bodies collapse to one id.
- **Near-duplicate**: a MinHash signature over word shingles, indexed in an LSH
  so lightly-edited reprints (different headers, a changed sentence) are caught
  at a configurable Jaccard threshold.

The content hash doubles as the document's primary key, so exact duplicates are
detected simply by a primary-key collision in the datastore.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit

from datasketch import MinHash, MinHashLSH

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_NUM_PERM = 128
DEFAULT_SHINGLE = 5

# Query params that identify a share/campaign, not the article — dropped so the
# same article reached via a feed (often utm-tagged) and via GDELT (usually
# clean) canonicalizes to one identity.
_TRACKING_KEYS = {
    "fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref", "ref_src",
    "cmpid", "cmp", "spm", "src", "smid", "smtyp", "_hsenc", "_hsmi",
}


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace so trivial formatting differs don't
    defeat exact hashing."""
    return _WS_RE.sub(" ", text.lower()).strip()


def content_hash(text: str) -> str:
    """Stable sha256 hex digest of the normalized text — the document id."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    """Canonicalize a URL to a stable identity key.

    Drops scheme (http/https treated alike), fragment, tracking query params,
    and any trailing slash; lowercases the host; sorts the surviving query so
    param order doesn't matter. Content-bearing query params (e.g. ``?id=123``)
    are kept. Used so the same article via a feed and via GDELT hashes to one id.
    """
    parts = urlsplit(url.strip())
    host = parts.netloc.rsplit("@", 1)[-1].split(":")[0].lower()
    path = parts.path.rstrip("/") or "/"
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_KEYS
    ]
    query = urlencode(sorted(kept))
    return f"{host}{path}" + (f"?{query}" if query else "")


def document_id(link: str | None, text: str) -> str:
    """Stable id for a document: canonical URL when present, else content hash."""
    return content_hash(normalize_url(link)) if link else content_hash(text)


def _shingles(text: str, k: int) -> set[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < k:
        # Short docs: fall back to individual tokens so they still get a signature.
        return set(tokens)
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def minhash_signature(text: str, num_perm: int = DEFAULT_NUM_PERM, k: int = DEFAULT_SHINGLE) -> MinHash:
    """Build a MinHash over word k-shingles of the text."""
    mh = MinHash(num_perm=num_perm)
    for shingle in _shingles(text, k):
        mh.update(shingle.encode("utf-8"))
    return mh


def signature_list(mh: MinHash) -> list[int]:
    """Serialize a MinHash to a plain list of ints for JSON storage."""
    return [int(x) for x in mh.hashvalues]


def signature_from_list(values: list[int], num_perm: int = DEFAULT_NUM_PERM) -> MinHash:
    """Rebuild a MinHash from a stored signature list."""
    import numpy as np

    mh = MinHash(num_perm=num_perm)
    mh.hashvalues = np.array(values, dtype=np.uint64)
    return mh


class NearDuplicateIndex:
    """LSH index over MinHash signatures for near-duplicate lookup.

    Seed it with previously-stored signatures (:meth:`add`) so dedup persists
    across runs, then use :meth:`query` before inserting each new document.
    """

    def __init__(self, threshold: float = 0.85, num_perm: int = DEFAULT_NUM_PERM) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    def add(self, doc_id: str, mh: MinHash) -> None:
        if doc_id not in self.lsh:
            self.lsh.insert(doc_id, mh)

    def query(self, mh: MinHash) -> list[str]:
        """Return ids of indexed documents within the Jaccard threshold."""
        return list(self.lsh.query(mh))

    def find_duplicate(self, mh: MinHash) -> str | None:
        """Return one existing near-duplicate id, or None."""
        matches = self.query(mh)
        return matches[0] if matches else None
