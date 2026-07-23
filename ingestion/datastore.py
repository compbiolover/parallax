"""SQLite datastore for Parallax (MVP).

Persists **derived metrics only** — document metadata, a content hash, a MinHash
signature, and foundation scores. Raw article text is a transient processing
artifact and is never written here (``CLAUDE.md`` §0). The pipeline scores a
document in memory during ingestion, then stores the results and discards the
text.

Schema:
  documents        one row per ingested item (metadata + dedup signals)
  foundation_scores one row per (document, scorer)

Move to Postgres + pgvector when co-located embeddings become worthwhile; the
access helpers here are deliberately thin so that swap stays localized.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,      -- content hash (sha256 hex)
    diet_id       TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    stratum_id    TEXT,
    url           TEXT,
    title         TEXT,
    published_utc TEXT,                  -- ISO-8601, may be NULL
    fetched_utc   TEXT NOT NULL,
    word_count    INTEGER NOT NULL,
    minhash       TEXT,                  -- JSON array of the signature ints
    weight        REAL NOT NULL DEFAULT 1.0,
    is_duplicate  INTEGER NOT NULL DEFAULT 0,
    duplicate_of  TEXT                   -- id of the canonical document, if dup
);

CREATE INDEX IF NOT EXISTS idx_documents_diet ON documents(diet_id);
CREATE INDEX IF NOT EXISTS idx_documents_dup  ON documents(is_duplicate);

CREATE TABLE IF NOT EXISTS foundation_scores (
    document_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    scorer           TEXT NOT NULL,
    care             REAL,
    fairness         REAL,
    loyalty          REAL,
    authority        REAL,
    sanctity         REAL,
    liberty          REAL,               -- NULL for the dictionary scorer
    sentiment        REAL,
    moral_word_ratio REAL,
    matched_words    INTEGER,
    PRIMARY KEY (document_id, scorer)
);
"""


@dataclass(frozen=True)
class StoredDocument:
    id: str
    diet_id: str
    source_id: str
    stratum_id: str | None
    url: str | None
    title: str | None
    published_utc: str | None
    fetched_utc: str
    word_count: int
    weight: float
    is_duplicate: bool
    duplicate_of: str | None


class Datastore:
    """Thin SQLite wrapper. Use as a context manager or call ``close()``."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Datastore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- writes ----------------------------------------------------------
    def has_document(self, doc_id: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,))
        return cur.fetchone() is not None

    def upsert_document(
        self,
        *,
        doc_id: str,
        diet_id: str,
        source_id: str,
        stratum_id: str | None,
        url: str | None,
        title: str | None,
        published_utc: str | None,
        fetched_utc: str,
        word_count: int,
        minhash: list[int] | None,
        weight: float = 1.0,
        is_duplicate: bool = False,
        duplicate_of: str | None = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, diet_id, source_id, stratum_id, url,
                    title, published_utc, fetched_utc, word_count, minhash,
                    weight, is_duplicate, duplicate_of)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    is_duplicate=excluded.is_duplicate,
                    duplicate_of=excluded.duplicate_of
                """,
                (
                    doc_id, diet_id, source_id, stratum_id, url, title,
                    published_utc, fetched_utc, word_count,
                    json.dumps(minhash) if minhash is not None else None,
                    weight, int(is_duplicate), duplicate_of,
                ),
            )

    def upsert_scores(
        self,
        *,
        document_id: str,
        scorer: str,
        foundations: dict[str, float],
        sentiment: float,
        moral_word_ratio: float,
        matched_words: int,
        liberty: float | None = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO foundation_scores (document_id, scorer, care,
                    fairness, loyalty, authority, sanctity, liberty, sentiment,
                    moral_word_ratio, matched_words)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(document_id, scorer) DO UPDATE SET
                    care=excluded.care, fairness=excluded.fairness,
                    loyalty=excluded.loyalty, authority=excluded.authority,
                    sanctity=excluded.sanctity, liberty=excluded.liberty,
                    sentiment=excluded.sentiment,
                    moral_word_ratio=excluded.moral_word_ratio,
                    matched_words=excluded.matched_words
                """,
                (
                    document_id, scorer,
                    foundations.get("care"), foundations.get("fairness"),
                    foundations.get("loyalty"), foundations.get("authority"),
                    foundations.get("sanctity"), liberty,
                    sentiment, moral_word_ratio, matched_words,
                ),
            )

    # -- reads -----------------------------------------------------------
    def iter_minhash_signatures(self, diet_id: str | None = None) -> Iterator[tuple[str, list[int]]]:
        """Yield (document_id, signature) for non-duplicate docs with a signature."""
        sql = "SELECT id, minhash FROM documents WHERE minhash IS NOT NULL AND is_duplicate = 0"
        params: tuple[str, ...] = ()
        if diet_id is not None:
            sql += " AND diet_id = ?"
            params = (diet_id,)
        for row in self.conn.execute(sql, params):
            yield row["id"], json.loads(row["minhash"])

    def scores_for_diet(self, diet_id: str, scorer: str = "dictionary") -> list[sqlite3.Row]:
        """All non-duplicate document scores for a diet."""
        return list(
            self.conn.execute(
                """
                SELECT d.weight AS weight, s.*
                FROM foundation_scores s
                JOIN documents d ON d.id = s.document_id
                WHERE d.diet_id = ? AND d.is_duplicate = 0 AND s.scorer = ?
                """,
                (diet_id, scorer),
            )
        )

    def diet_ids(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT diet_id FROM documents ORDER BY diet_id")
        return [r["diet_id"] for r in rows]

    def counts(self) -> dict[str, int]:
        total = self.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        dups = self.conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE is_duplicate = 1"
        ).fetchone()["n"]
        return {"documents": total, "duplicates": dups, "unique": total - dups}
