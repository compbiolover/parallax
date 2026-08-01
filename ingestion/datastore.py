"""SQLite datastore for Parallax (MVP).

Persists **derived metrics only** — document metadata, a content hash, a MinHash
signature, and foundation scores. Raw article text is a transient processing
artifact and is never written here (``CLAUDE.md`` §0). The pipeline scores a
document in memory during ingestion, then stores the results and discards the
text.

Schema:
  documents        one row per ingested item (metadata + dedup signals)
  foundation_scores one row per (document, scorer)
  snapshots        one aggregate row per UTC date (the JSD/composition history)

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
    -- MFQ-2 halves of `fairness`. NULL means "not partitioned" (no splitter, or
    -- too little evidence) — never confuse that with an even split or a zero.
    equality         REAL,
    proportionality  REAL,
    sentiment        REAL,
    moral_word_ratio REAL,
    matched_words    INTEGER,
    PRIMARY KEY (document_id, scorer)
);

CREATE TABLE IF NOT EXISTS summaries (
    scope         TEXT PRIMARY KEY,      -- diet id, or 'executive'
    generated_utc TEXT NOT NULL,
    model         TEXT NOT NULL,
    method        TEXT NOT NULL,         -- 'claude' | 'deterministic'
    text          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,              -- provenance, e.g. 'lexicon'
    value TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    dim         INTEGER NOT NULL,
    vector      TEXT NOT NULL,           -- JSON array of floats
    embedder    TEXT NOT NULL
);

-- A single latest clustering (rewritten each cluster run).
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id  INTEGER PRIMARY KEY,     -- -1 reserved for noise
    label       TEXT,
    size        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS document_clusters (
    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    cluster_id  INTEGER NOT NULL
);

-- Which theme each blindspot *story* reads under. Keyed per document, not per
-- cluster: a cluster is not always one subject, and labeling the whole cluster
-- filed a sports headline under "Faith & the church" because the other four
-- headlines beside it were about church. Written by the cluster run
-- (which is where a model call belongs) and read by the exporter, so naming a
-- theme costs one API call a day rather than one per surface that shows it.
-- Rewritten whole each run, like the clustering it describes.
CREATE TABLE IF NOT EXISTS blindspot_themes (
    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    theme_key   TEXT NOT NULL,
    theme_title TEXT NOT NULL,
    method      TEXT NOT NULL            -- 'taxonomy' | 'claude'
);

-- One aggregate snapshot per UTC date. Re-running the daily job on the same
-- day overwrites that day's row rather than appending, so the series has one
-- point per day regardless of how many times the pipeline ran.
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_date TEXT PRIMARY KEY,      -- 'YYYY-MM-DD' (UTC)
    generated_utc TEXT NOT NULL,
    window_days   INTEGER NOT NULL,      -- trailing window used for the windowed basis
    jsd_cumulative REAL,                 -- NULL when fewer than two diets are scored
    jsd_window     REAL,
    payload       TEXT NOT NULL          -- JSON: compositions, counts, log-ratios
);
"""


# How a document is dated: by publication where the source supplied one, by
# fetch time otherwise. Both are stored as UTC ISO-8601.
DOC_DATE_SQL = "COALESCE(d.published_utc, d.fetched_utc)"


def _date_window(since: str | None, until: str | None) -> tuple[str, tuple[str, ...]]:
    """SQL fragment + params restricting documents to the half-open ``[since, until)``.

    Bounds are plain ``YYYY-MM-DD`` strings compared directly against ISO-8601
    timestamps. That is exact rather than a shortcut: every timestamp this store
    holds is normalized to UTC at ingestion, so the strings sort chronologically
    and a bare date compares against them the way the calendar does. It would
    stop being exact the moment a non-UTC offset were persisted.
    """
    clause = ""
    params: list[str] = []
    if since is not None:
        clause += f" AND {DOC_DATE_SQL} >= ?"
        params.append(since)
    if until is not None:
        clause += f" AND {DOC_DATE_SQL} < ?"
        params.append(until)
    return clause, tuple(params)


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
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for stores created by an earlier schema.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op on a table that already
        exists, so columns added to :data:`SCHEMA` after a store was first
        created never appear in it. Each entry below is applied only when its
        column is missing, which makes opening an old database upgrade it in
        place instead of failing on the first write that mentions a new column.
        """
        # `blindspot_themes` was keyed by cluster_id before themes were assigned
        # per story. It is a per-run cache rewritten by every cluster run, so
        # the old table is dropped rather than migrated — there is nothing in it
        # worth carrying across, and a changed primary key is not something
        # `CREATE TABLE IF NOT EXISTS` can fix on its own.
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(blindspot_themes)")}
        if cols and "document_id" not in cols:
            self.conn.execute("DROP TABLE blindspot_themes")
            self.conn.executescript(SCHEMA)

        added: list[tuple[str, str, str]] = [
            ("foundation_scores", "equality", "REAL"),
            ("foundation_scores", "proportionality", "REAL"),
        ]
        for table, column, decl in added:
            cols = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if cols and column not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

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
        equality: float | None = None,
        proportionality: float | None = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO foundation_scores (document_id, scorer, care,
                    fairness, loyalty, authority, sanctity, liberty,
                    equality, proportionality, sentiment,
                    moral_word_ratio, matched_words)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(document_id, scorer) DO UPDATE SET
                    care=excluded.care, fairness=excluded.fairness,
                    loyalty=excluded.loyalty, authority=excluded.authority,
                    sanctity=excluded.sanctity, liberty=excluded.liberty,
                    equality=excluded.equality,
                    proportionality=excluded.proportionality,
                    sentiment=excluded.sentiment,
                    moral_word_ratio=excluded.moral_word_ratio,
                    matched_words=excluded.matched_words
                """,
                (
                    document_id, scorer,
                    foundations.get("care"), foundations.get("fairness"),
                    foundations.get("loyalty"), foundations.get("authority"),
                    foundations.get("sanctity"), liberty,
                    equality, proportionality,
                    sentiment, moral_word_ratio, matched_words,
                ),
            )

    # -- reads -----------------------------------------------------------
    def iter_minhash_signatures(
        self, diet_id: str | None = None
    ) -> Iterator[tuple[str, list[int]]]:
        """Yield (document_id, signature) for non-duplicate docs with a signature."""
        sql = "SELECT id, minhash FROM documents WHERE minhash IS NOT NULL AND is_duplicate = 0"
        params: tuple[str, ...] = ()
        if diet_id is not None:
            sql += " AND diet_id = ?"
            params = (diet_id,)
        for row in self.conn.execute(sql, params):
            yield row["id"], json.loads(row["minhash"])

    def scores_for_diet(
        self,
        diet_id: str,
        scorer: str = "dictionary",
        since: str | None = None,
        until: str | None = None,
    ) -> list[sqlite3.Row]:
        """All non-duplicate document scores for a diet.

        ``since``/``until`` restrict to the half-open date window
        ``[since, until)`` — see :func:`_date_window` for why plain string
        comparison is exact here. Omitting both gives every document ever
        ingested, which is what the headline dashboard numbers use.
        """
        clause, params = _date_window(since, until)
        return list(
            self.conn.execute(
                f"""
                SELECT d.weight AS weight, s.*
                FROM foundation_scores s
                JOIN documents d ON d.id = s.document_id
                WHERE d.diet_id = ? AND d.is_duplicate = 0 AND s.scorer = ?{clause}
                """,
                (diet_id, scorer, *params),
            )
        )

    def fairness_split_for_diet(
        self, diet_id: str, scorer: str = "dictionary"
    ) -> tuple[list[tuple[float, float, float]], int]:
        """``([(weight, equality, proportionality), ...], n_scored)`` for a diet.

        Only rows that were actually partitioned are returned; ``n_scored`` is
        the number of documents the scorer saw at all, so the caller can report
        what fraction carried enough evidence to split. Treating the unsplit
        remainder as zeros would manufacture an equality/proportionality reading
        for every document that never discussed either.
        """
        rows = self.conn.execute(
            """
            SELECT d.weight AS weight, s.equality AS eq, s.proportionality AS pr
            FROM foundation_scores s
            JOIN documents d ON d.id = s.document_id
            WHERE d.diet_id = ? AND d.is_duplicate = 0 AND s.scorer = ?
              AND s.equality IS NOT NULL AND s.proportionality IS NOT NULL
            """,
            (diet_id, scorer),
        )
        split = [(float(r["weight"] or 1.0), float(r["eq"]), float(r["pr"])) for r in rows]
        total = self.conn.execute(
            """
            SELECT COUNT(*) AS n FROM foundation_scores s
            JOIN documents d ON d.id = s.document_id
            WHERE d.diet_id = ? AND d.is_duplicate = 0 AND s.scorer = ?
            """,
            (diet_id, scorer),
        ).fetchone()["n"]
        return split, total

    def liberty_for_diet(
        self, diet_id: str, scorer: str
    ) -> tuple[list[tuple[float, float]], int]:
        """``([(weight, liberty), ...], n_documents)`` for one diet.

        Only rows carrying a liberty score are returned; ``n_documents`` is the
        diet's whole non-duplicate corpus, so the caller can report what fraction
        was actually tagged. Liberty coverage is always partial — the tagger runs
        on feed-ingested documents only, and only when an API key is set — so a
        mean over the tagged subset is not a mean over the diet.
        """
        rows = self.conn.execute(
            """
            SELECT d.weight AS weight, s.liberty AS liberty
            FROM foundation_scores s
            JOIN documents d ON d.id = s.document_id
            WHERE d.diet_id = ? AND d.is_duplicate = 0 AND s.scorer = ?
              AND s.liberty IS NOT NULL
            """,
            (diet_id, scorer),
        )
        scored = [(float(r["weight"] or 1.0), float(r["liberty"])) for r in rows]
        return scored, self.doc_count(diet_id)

    def scorer_names(self) -> list[str]:
        """Distinct scorer names present in foundation_scores (e.g. 'dictionary',
        'transformer/...'). Lets the exporter detect whether bands are possible."""
        rows = self.conn.execute("SELECT DISTINCT scorer FROM foundation_scores ORDER BY scorer")
        return [r["scorer"] for r in rows]

    def paired_scores_for_diet(
        self, diet_id: str, scorer_a: str, scorer_b: str,
        foundations: list[str] | None = None,
    ) -> list[tuple[float, dict[str, float], dict[str, float]]]:
        """Per non-duplicate document in a diet, ``(weight, scorer_a_map, scorer_b_map)``
        — only for documents scored by *both*. Pairing keeps the two taggers on the
        same document population (backfill writes dictionary rows but no transformer
        rows), which both the confidence band and the disagreement share rely on.

        ``founds`` is interpolated into the SELECT (SQLite can't bind identifiers),
        so it is whitelisted against the known foundation columns to keep this from
        becoming an injection point if a future caller passes a non-constant list.
        """
        from scoring.foundations import CLASSIC_FOUNDATIONS

        allowed = set(CLASSIC_FOUNDATIONS)
        founds = [f for f in (foundations or list(CLASSIC_FOUNDATIONS)) if f in allowed]
        if not founds:
            founds = list(CLASSIC_FOUNDATIONS)
        rows = self.conn.execute(
            f"""
            SELECT d.weight AS weight,
                   {', '.join(f'a.{c} AS a_{c}' for c in founds)},
                   {', '.join(f'b.{c} AS b_{c}' for c in founds)}
            FROM foundation_scores a
            JOIN foundation_scores b ON a.document_id = b.document_id
            JOIN documents d ON d.id = a.document_id
            WHERE d.diet_id = ? AND d.is_duplicate = 0
              AND a.scorer = ? AND b.scorer = ?
            """,
            (diet_id, scorer_a, scorer_b),
        )
        out = []
        for r in rows:
            a = {c: (r[f"a_{c}"] if r[f"a_{c}"] is not None else 0.0) for c in founds}
            b = {c: (r[f"b_{c}"] if r[f"b_{c}"] is not None else 0.0) for c in founds}
            out.append((float(r["weight"] or 1.0), a, b))
        return out

    def headlines_for_diet(self, diet_id: str, limit: int = 50) -> list[str]:
        """Titles of non-duplicate documents for a diet, most recent first."""
        rows = self.conn.execute(
            """
            SELECT title FROM documents
            WHERE diet_id = ? AND is_duplicate = 0 AND title IS NOT NULL AND title != ''
            ORDER BY COALESCE(published_utc, fetched_utc) DESC
            LIMIT ?
            """,
            (diet_id, limit),
        )
        return [r["title"] for r in rows]

    def doc_count(
        self, diet_id: str, since: str | None = None, until: str | None = None
    ) -> int:
        clause, params = _date_window(since, until)
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM documents d "
            f"WHERE d.diet_id = ? AND d.is_duplicate = 0{clause}",
            (diet_id, *params),
        ).fetchone()["n"]

    def document_date_range(self) -> tuple[str | None, str | None]:
        """Earliest and latest document date (``YYYY-MM-DD``) over non-duplicates.

        Bounds the retroactive history reconstruction so it does not walk back
        through days the corpus never covered.
        """
        row = self.conn.execute(
            f"SELECT MIN({DOC_DATE_SQL}) AS lo, MAX({DOC_DATE_SQL}) AS hi "
            "FROM documents d WHERE d.is_duplicate = 0"
        ).fetchone()
        lo, hi = row["lo"], row["hi"]
        return (lo[:10] if lo else None, hi[:10] if hi else None)

    # -- summaries -------------------------------------------------------
    def upsert_summary(
        self, *, scope: str, generated_utc: str, model: str, method: str, text: str
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO summaries (scope, generated_utc, model, method, text)
                VALUES (?,?,?,?,?)
                ON CONFLICT(scope) DO UPDATE SET
                    generated_utc=excluded.generated_utc, model=excluded.model,
                    method=excluded.method, text=excluded.text
                """,
                (scope, generated_utc, model, method, text),
            )

    def all_summaries(self) -> dict[str, sqlite3.Row]:
        return {r["scope"]: r for r in self.conn.execute("SELECT * FROM summaries")}

    # -- snapshot history ------------------------------------------------
    def upsert_snapshot(
        self,
        *,
        snapshot_date: str,
        generated_utc: str,
        window_days: int,
        jsd_cumulative: float | None,
        jsd_window: float | None,
        payload: dict,
    ) -> None:
        """Record (or replace) the aggregate snapshot for one UTC date."""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO snapshots (snapshot_date, generated_utc, window_days,
                    jsd_cumulative, jsd_window, payload)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(snapshot_date) DO UPDATE SET
                    generated_utc=excluded.generated_utc,
                    window_days=excluded.window_days,
                    jsd_cumulative=excluded.jsd_cumulative,
                    jsd_window=excluded.jsd_window,
                    payload=excluded.payload
                """,
                (
                    snapshot_date, generated_utc, int(window_days),
                    jsd_cumulative, jsd_window, json.dumps(payload),
                ),
            )

    def snapshot_rows(self, limit: int | None = None) -> list[sqlite3.Row]:
        """Snapshots in chronological order. ``limit`` keeps the *most recent*
        N, still oldest-first — the shape a time-series chart wants."""
        if limit is None:
            return list(self.conn.execute("SELECT * FROM snapshots ORDER BY snapshot_date"))
        rows = self.conn.execute(
            "SELECT * FROM snapshots ORDER BY snapshot_date DESC LIMIT ?", (limit,)
        )
        return list(reversed(list(rows)))

    def snapshot_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()["n"]

    # -- provenance metadata --------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    # A diet's human labels, recorded at ingestion from the source registry.
    # They live here rather than being read from `sources.yaml` at render time
    # because every surface downstream already has the store and none of them
    # has the registry path — and because a label read from today's registry
    # would silently rename a diet in yesterday's exported payload.
    #
    # Two of them: the full label reads as a noun phrase in prose ("Modeled
    # conservative-evangelical diet"), the short one fits a chart legend.
    DIET_LABEL_PREFIX = "diet_label:"
    DIET_SHORT_LABEL_PREFIX = "diet_short_label:"
    # And the outlets', for the same reason: `source_id` is `christianity_today`,
    # which is not what a masthead is called.
    SOURCE_LABEL_PREFIX = "source_label:"

    def set_diet_label(self, diet_id: str, label: str, short_label: str = "") -> None:
        self.set_meta(f"{self.DIET_LABEL_PREFIX}{diet_id}", label)
        if short_label:
            self.set_meta(f"{self.DIET_SHORT_LABEL_PREFIX}{diet_id}", short_label)

    def diet_labels(self) -> dict[str, str]:
        """``{diet_id: label}`` for every diet whose label has been recorded."""
        return self._labels(self.DIET_LABEL_PREFIX)

    def diet_short_labels(self) -> dict[str, str]:
        """``{diet_id: short_label}``, for diets that supplied a short one."""
        return self._labels(self.DIET_SHORT_LABEL_PREFIX)

    def set_source_label(self, source_id: str, name: str) -> None:
        self.set_meta(f"{self.SOURCE_LABEL_PREFIX}{source_id}", name)

    def source_labels(self) -> dict[str, str]:
        """``{source_id: display name}`` as the registry writes them."""
        return self._labels(self.SOURCE_LABEL_PREFIX)

    def _labels(self, prefix: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM meta WHERE key LIKE ?", (f"{prefix}%",)
        )
        return {r["key"][len(prefix):]: r["value"] for r in rows if r["value"]}

    # -- embeddings ------------------------------------------------------
    def upsert_embedding(self, *, document_id: str, vector: list[float], embedder: str) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO embeddings (document_id, dim, vector, embedder)
                VALUES (?,?,?,?)
                ON CONFLICT(document_id) DO UPDATE SET
                    dim=excluded.dim, vector=excluded.vector, embedder=excluded.embedder
                """,
                (document_id, len(vector), json.dumps(vector), embedder),
            )

    def iter_embeddings(
        self, embedder: str | None = None
    ) -> Iterator[tuple[str, str, str | None, list[int], list[float]]]:
        """Yield (document_id, diet_id, title, _, vector) for non-duplicate docs.

        When ``embedder`` is given, only rows produced by that embedder are
        yielded — embeddings from different embedders live in incompatible vector
        spaces (and often different dimensions), so clustering must not mix them.
        """
        sql = """
            SELECT e.document_id AS id, d.diet_id AS diet_id, d.title AS title,
                   e.vector AS vector
            FROM embeddings e JOIN documents d ON d.id = e.document_id
            WHERE d.is_duplicate = 0
        """
        params: tuple[str, ...] = ()
        if embedder is not None:
            sql += " AND e.embedder = ?"
            params = (embedder,)
        for r in self.conn.execute(sql, params):
            yield r["id"], r["diet_id"], r["title"], [], json.loads(r["vector"])

    def embedder_names(self) -> list[str]:
        """Distinct embedder names present in the embeddings table."""
        rows = self.conn.execute("SELECT DISTINCT embedder FROM embeddings ORDER BY embedder")
        return [r["embedder"] for r in rows]

    def embedding_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"]

    # -- clusters --------------------------------------------------------
    def replace_clustering(
        self,
        clusters: list[tuple[int, str | None, int]],       # (cluster_id, label, size)
        assignments: list[tuple[str, int]],                 # (document_id, cluster_id)
    ) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM clusters")
            conn.execute("DELETE FROM document_clusters")
            # Cluster ids are positions in a fresh HDBSCAN run, not stable
            # identities. Leaving yesterday's themes behind would file today's
            # cluster 3 under whatever cluster 3 was about yesterday.
            conn.execute("DELETE FROM blindspot_themes")
            conn.executemany(
                "INSERT INTO clusters (cluster_id, label, size) VALUES (?,?,?)", clusters
            )
            conn.executemany(
                "INSERT INTO document_clusters (document_id, cluster_id) VALUES (?,?)",
                assignments,
            )

    def replace_blindspot_themes(
        self, themes: list[tuple[str, str, str, str]]   # (document_id, key, title, method)
    ) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM blindspot_themes")
            conn.executemany(
                "INSERT INTO blindspot_themes (document_id, theme_key, theme_title, "
                "method) VALUES (?,?,?,?)",
                themes,
            )

    def blindspot_theme_rows(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM blindspot_themes"))

    def cluster_rows(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM clusters ORDER BY size DESC"))

    def cluster_members(self, cluster_id: int) -> list[sqlite3.Row]:
        """Members of a cluster with diet, title, outlet, and link.

        The link is the point: a story is only checkable if the reader can go
        read it, and ``CLAUDE.md`` §0 permits exactly that — summarize and
        link, never republish.
        """
        return list(
            self.conn.execute(
                """
                SELECT d.id AS id, d.diet_id AS diet_id, d.title AS title,
                       d.url AS url, d.source_id AS source_id
                FROM document_clusters dc JOIN documents d ON d.id = dc.document_id
                WHERE dc.cluster_id = ?
                ORDER BY COALESCE(d.published_utc, d.fetched_utc) DESC
                """,
                (cluster_id,),
            )
        )

    def cluster_diet_counts(self) -> list[sqlite3.Row]:
        """``(cluster_id, diet_id, n)`` for the whole clustering, in one query.

        What the agenda metrics run on. Asking per cluster instead would be one
        query per story on a corpus that has hundreds of them.
        """
        return list(
            self.conn.execute(
                """
                SELECT dc.cluster_id AS cluster_id, d.diet_id AS diet_id,
                       COUNT(*) AS n
                FROM document_clusters dc JOIN documents d ON d.id = dc.document_id
                WHERE dc.cluster_id != -1
                GROUP BY dc.cluster_id, d.diet_id
                """
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
