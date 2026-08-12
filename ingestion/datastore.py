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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


PODCAST_EPISODES_DDL = """
-- Episodes seen, whether or not they produced a document.
--
-- The ledger is what makes podcast ingestion re-runnable. Every other source
-- type is cheap to re-fetch, so the pipeline dedups after the fact on document
-- id; an episode costs a download and minutes of transcription, so it has to be
-- recognised *before* any of that, from the feed alone. The guid is the only
-- identifier available at that point.
--
-- Failures are recorded too, with their reason. An episode that 404s or blows
-- the size cap would otherwise be retried every morning forever, and the
-- ledger's whole purpose is to stop the run repeating expensive work.
--
-- Keyed by (source_id, guid), not guid alone. A guid is unique within a feed
-- and nowhere else: `<guid isPermaLink="false">12</guid>` is legal and two
-- shows can both use it. Keyed globally, the second show's episode would
-- overwrite the first show's row — leaving one show skipping an episode it
-- never transcribed and the other re-transcribing one it did, which is both
-- failure modes this table was built to prevent, at once.
CREATE TABLE IF NOT EXISTS podcast_episodes (
    guid          TEXT NOT NULL,         -- feed guid, or the enclosure url
    source_id     TEXT NOT NULL,
    title         TEXT,
    published_utc TEXT,
    processed_utc TEXT NOT NULL,
    status        TEXT NOT NULL,         -- 'transcribed' | 'skipped' | 'failed'
    detail        TEXT,                  -- why, for anything but 'transcribed'
    -- Set only on success, and NULL rather than a dangling id if that document
    -- is later removed: the episode was still processed, and re-transcribing it
    -- because the document went away is exactly the cost this table prevents.
    document_id   TEXT REFERENCES documents(id) ON DELETE SET NULL,
    duration_seconds INTEGER,
    -- The second identity, and not redundant. feedparser resolves a relative
    -- permalink guid against the feed's own URL, so `<guid>ep-1</guid>` becomes
    -- `https://host/ep-1` — and every guid in the feed changes if that host
    -- does. A migration, or swapping a public feed for a subscriber one, would
    -- otherwise make the whole back catalogue look new and re-transcribe it.
    -- The enclosure URL survives most of those moves.
    audio_url     TEXT,
    PRIMARY KEY (source_id, guid)
);

CREATE INDEX IF NOT EXISTS idx_podcast_episodes_source
    ON podcast_episodes(source_id, published_utc);
-- The index on `audio_url` is created in `_migrate`, not here: this script runs
-- first, and on a store predating that column the CREATE INDEX would fail
-- before the ALTER that adds it.
"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,      -- content hash (sha256 hex)
    -- LEGACY, kept for readability of pre-persona stores. It held the one diet a
    -- document was ingested under, which was well-defined only while a source
    -- belonged to exactly one diet. Nothing reads it now: membership is a
    -- property of `source_id` (a source may be read by several personas at
    -- different weights, so there is no single diet to record). New rows write ''.
    -- Not dropped, because dropping a NOT NULL column means rebuilding
    -- `documents` — the parent of five foreign keys — and this store holds the
    -- only copy of the snapshot history.
    diet_id       TEXT NOT NULL DEFAULT '',
    source_id     TEXT NOT NULL,
    stratum_id    TEXT,
    url           TEXT,
    title         TEXT,
    published_utc TEXT,                  -- ISO-8601, may be NULL
    fetched_utc   TEXT NOT NULL,
    word_count    INTEGER NOT NULL,
    minhash       TEXT,                  -- JSON array of the signature ints
    -- LEGACY, same reason: it held stratum_weight * source_weight for that one
    -- diet. Weight is now resolved per persona at aggregation time, which is what
    -- lets one document count differently for two readers of the same source.
    weight        REAL NOT NULL DEFAULT 1.0,
    is_duplicate  INTEGER NOT NULL DEFAULT 0,
    duplicate_of  TEXT                   -- id of the canonical document, if dup
);

CREATE INDEX IF NOT EXISTS idx_documents_diet ON documents(diet_id);
CREATE INDEX IF NOT EXISTS idx_documents_dup  ON documents(is_duplicate);
-- source_id is the join key for every aggregate query now that diet_id is not.
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);

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

{PODCAST_EPISODES_DDL}

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


def _source_filter(source_ids: Iterable[str]) -> tuple[str, tuple[str, ...]]:
    """SQL fragment + params restricting documents to a set of source ids.

    Documents are keyed by source, not by diet, because a source belongs to as
    many personas as list it — so "this persona's documents" is a set membership
    test rather than a column comparison.

    An empty set yields ``AND 0``, which matches nothing. That is the whole
    reason this is a helper: SQLite rejects an empty ``IN ()`` outright, and
    omitting the clause instead would quietly return *every* document in the
    store to a persona that consumes none.
    """
    ids = tuple(dict.fromkeys(source_ids))
    if not ids:
        return " AND 0", ()
    return f" AND d.source_id IN ({','.join('?' * len(ids))})", ids


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

        # `podcast_episodes` was keyed on `guid` alone before it was keyed on
        # (source_id, guid). A primary key is not something ALTER TABLE can
        # change, so the table is rebuilt and its rows copied — they are worth
        # carrying, unlike the blindspot cache above: each one stands for an
        # episode already paid for in CPU time, and dropping them would re-run
        # every transcription.
        info = list(self.conn.execute("PRAGMA table_info(podcast_episodes)"))
        if info and sum(1 for r in info if r["pk"]) == 1:
            columns = [r["name"] for r in info]
            shared = ", ".join(c for c in columns if c != "audio_url")
            self.conn.executescript(f"""
                ALTER TABLE podcast_episodes RENAME TO podcast_episodes_old;
                {PODCAST_EPISODES_DDL}
                INSERT OR IGNORE INTO podcast_episodes ({shared})
                    SELECT {shared} FROM podcast_episodes_old;
                DROP TABLE podcast_episodes_old;
            """)

        added: list[tuple[str, str, str]] = [
            ("foundation_scores", "equality", "REAL"),
            ("foundation_scores", "proportionality", "REAL"),
            ("podcast_episodes", "audio_url", "TEXT"),
        ]
        for table, column, decl in added:
            cols = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if cols and column not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

        # After the ALTERs above, so it works on an upgraded store as well as a
        # fresh one. Idempotent, so running it on both is free.
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_podcast_episodes_audio "
            "ON podcast_episodes(source_id, audio_url)"
        )

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
        source_id: str,
        stratum_id: str | None,
        url: str | None,
        title: str | None,
        published_utc: str | None,
        fetched_utc: str,
        word_count: int,
        minhash: list[int] | None,
        is_duplicate: bool = False,
        duplicate_of: str | None = None,
        # Legacy columns. Both are written blank/neutral on new rows and read by
        # nothing; see the DDL comments on `documents`.
        diet_id: str = "",
        weight: float = 1.0,
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
                    doc_id,
                    diet_id,
                    source_id,
                    stratum_id,
                    url,
                    title,
                    published_utc,
                    fetched_utc,
                    word_count,
                    json.dumps(minhash) if minhash is not None else None,
                    weight,
                    int(is_duplicate),
                    duplicate_of,
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
                    document_id,
                    scorer,
                    foundations.get("care"),
                    foundations.get("fairness"),
                    foundations.get("loyalty"),
                    foundations.get("authority"),
                    foundations.get("sanctity"),
                    liberty,
                    equality,
                    proportionality,
                    sentiment,
                    moral_word_ratio,
                    matched_words,
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

    def scores_for_sources(
        self,
        source_ids: Iterable[str],
        scorer: str = "dictionary",
        since: str | None = None,
        until: str | None = None,
    ) -> list[sqlite3.Row]:
        """All non-duplicate document scores from a set of sources.

        Returns ``source_id`` per row rather than a weight: weight is a property
        of the persona doing the reading, so the caller multiplies it in (see
        ``Registry.weights_for``). Two personas that share a source read the same
        rows here and weight them differently, which is exactly the point.

        ``since``/``until`` restrict to the half-open date window
        ``[since, until)`` — see :func:`_date_window` for why plain string
        comparison is exact here. Omitting both gives every document ever
        ingested, which is what the headline dashboard numbers use.
        """
        where, ids = _source_filter(source_ids)
        clause, params = _date_window(since, until)
        return list(
            self.conn.execute(
                f"""
                SELECT d.source_id AS source_id, s.*
                FROM foundation_scores s
                JOIN documents d ON d.id = s.document_id
                WHERE d.is_duplicate = 0 AND s.scorer = ?{where}{clause}
                ORDER BY s.document_id
                """,
                (scorer, *ids, *params),
            )
        )

    def fairness_split_for_sources(
        self, source_ids: Iterable[str], scorer: str = "dictionary"
    ) -> tuple[list[tuple[str, float, float]], int]:
        """``([(source_id, equality, proportionality), ...], n_scored)``.

        Only rows that were actually partitioned are returned; ``n_scored`` is
        the number of documents the scorer saw at all, so the caller can report
        what fraction carried enough evidence to split. Treating the unsplit
        remainder as zeros would manufacture an equality/proportionality reading
        for every document that never discussed either.
        """
        where, ids = _source_filter(source_ids)
        rows = self.conn.execute(
            f"""
            SELECT d.source_id AS source_id, s.equality AS eq, s.proportionality AS pr
            FROM foundation_scores s
            JOIN documents d ON d.id = s.document_id
            WHERE d.is_duplicate = 0 AND s.scorer = ?{where}
              AND s.equality IS NOT NULL AND s.proportionality IS NOT NULL
            """,
            (scorer, *ids),
        )
        split = [(r["source_id"], float(r["eq"]), float(r["pr"])) for r in rows]
        total = self.conn.execute(
            f"""
            SELECT COUNT(*) AS n FROM foundation_scores s
            JOIN documents d ON d.id = s.document_id
            WHERE d.is_duplicate = 0 AND s.scorer = ?{where}
            """,
            (scorer, *ids),
        ).fetchone()["n"]
        return split, total

    def liberty_for_sources(
        self, source_ids: Iterable[str], scorer: str
    ) -> tuple[list[tuple[str, float]], int]:
        """``([(source_id, liberty), ...], n_documents)``.

        Only rows carrying a liberty score are returned; ``n_documents`` is the
        whole non-duplicate corpus from those sources, so the caller can report
        what fraction was actually tagged. Liberty coverage is always partial —
        the tagger runs on feed-ingested documents only, and only when an API key
        is set — so a mean over the tagged subset is not a mean over the diet.
        """
        where, ids = _source_filter(source_ids)
        rows = self.conn.execute(
            f"""
            SELECT d.source_id AS source_id, s.liberty AS liberty
            FROM foundation_scores s
            JOIN documents d ON d.id = s.document_id
            WHERE d.is_duplicate = 0 AND s.scorer = ?{where}
              AND s.liberty IS NOT NULL
            """,
            (scorer, *ids),
        )
        scored = [(r["source_id"], float(r["liberty"])) for r in rows]
        return scored, self.doc_count_for_sources(source_ids)

    def scorer_names(self) -> list[str]:
        """Distinct scorer names present in foundation_scores (e.g. 'dictionary',
        'transformer/...'). Lets the exporter detect whether bands are possible."""
        rows = self.conn.execute("SELECT DISTINCT scorer FROM foundation_scores ORDER BY scorer")
        return [r["scorer"] for r in rows]

    def paired_scores_for_sources(
        self,
        source_ids: Iterable[str],
        scorer_a: str,
        scorer_b: str,
        foundations: list[str] | None = None,
    ) -> list[tuple[str, dict[str, float], dict[str, float]]]:
        """Per non-duplicate document, ``(source_id, scorer_a_map, scorer_b_map)``
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
        where, ids = _source_filter(source_ids)
        rows = self.conn.execute(
            f"""
            SELECT d.source_id AS source_id,
                   {", ".join(f"a.{c} AS a_{c}" for c in founds)},
                   {", ".join(f"b.{c} AS b_{c}" for c in founds)}
            FROM foundation_scores a
            JOIN foundation_scores b ON a.document_id = b.document_id
            JOIN documents d ON d.id = a.document_id
            WHERE d.is_duplicate = 0
              AND a.scorer = ? AND b.scorer = ?{where}
            ORDER BY a.document_id
            """,
            (scorer_a, scorer_b, *ids),
        )
        out = []
        for r in rows:
            a = {c: (r[f"a_{c}"] if r[f"a_{c}"] is not None else 0.0) for c in founds}
            b = {c: (r[f"b_{c}"] if r[f"b_{c}"] is not None else 0.0) for c in founds}
            out.append((r["source_id"], a, b))
        return out

    def headlines_for_sources(self, source_ids: Iterable[str], limit: int = 50) -> list[str]:
        """Titles of non-duplicate documents from a set of sources, newest first."""
        where, ids = _source_filter(source_ids)
        rows = self.conn.execute(
            f"""
            SELECT d.title AS title FROM documents d
            WHERE d.is_duplicate = 0 AND d.title IS NOT NULL AND d.title != ''{where}
            ORDER BY COALESCE(d.published_utc, d.fetched_utc) DESC
            LIMIT ?
            """,
            (*ids, limit),
        )
        return [r["title"] for r in rows]

    def doc_count_for_sources(
        self, source_ids: Iterable[str], since: str | None = None, until: str | None = None
    ) -> int:
        where, ids = _source_filter(source_ids)
        clause, params = _date_window(since, until)
        return self.conn.execute(
            f"SELECT COUNT(*) AS n FROM documents d WHERE d.is_duplicate = 0{where}{clause}",
            (*ids, *params),
        ).fetchone()["n"]

    def orphan_source_counts(self, known: Iterable[str]) -> dict[str, int]:
        """``{source_id: n}`` for stored documents no persona can reach.

        Under baked-in weights, deleting a source from the registry left its
        documents counted in their diet forever. Now that weight is resolved from
        the registry, those documents become invisible to every persona instead —
        the registry is retroactive. That is the right behaviour and a silent one,
        so the run says it out loud, the same way it says the lexicon changed.
        """
        keep = set(known)
        rows = self.conn.execute(
            "SELECT source_id, COUNT(*) AS n FROM documents "
            "WHERE is_duplicate = 0 GROUP BY source_id"
        )
        return {r["source_id"]: r["n"] for r in rows if r["source_id"] not in keep}

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
                    snapshot_date,
                    generated_utc,
                    int(window_days),
                    jsd_cumulative,
                    jsd_window,
                    json.dumps(payload),
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

    # -- podcast episodes ------------------------------------------------
    def seen_episode_keys(self, source_id: str) -> set[str]:
        """Every guid *and* enclosure URL already processed for a source.

        Both, because either can identify an episode and neither is reliable
        alone: a relative guid is rewritten when the feed moves, and an
        enclosure URL is rewritten when the host changes its CDN. Matching on
        the union costs one extra column and saves re-transcribing a back
        catalogue over a URL change.

        Read once per source and checked in memory — the alternative is a query
        per episode, and the caller is deciding what to skip before it does
        anything expensive."""
        rows = self.conn.execute(
            "SELECT guid, audio_url FROM podcast_episodes WHERE source_id = ?",
            (source_id,),
        )
        keys: set[str] = set()
        for row in rows:
            keys.add(row["guid"])
            if row["audio_url"]:
                keys.add(row["audio_url"])
        return keys

    def record_episode(
        self,
        *,
        guid: str,
        source_id: str,
        status: str,
        title: str | None = None,
        published_utc: str | None = None,
        detail: str | None = None,
        document_id: str | None = None,
        duration_seconds: int | None = None,
        audio_url: str | None = None,
    ) -> None:
        """Record an episode as handled. Idempotent on guid."""
        self.conn.execute(
            """
            INSERT INTO podcast_episodes
                (guid, source_id, title, published_utc, processed_utc, status,
                 detail, document_id, duration_seconds, audio_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, guid) DO UPDATE SET
                status = excluded.status,
                detail = excluded.detail,
                document_id = COALESCE(excluded.document_id, podcast_episodes.document_id),
                processed_utc = excluded.processed_utc
            """,
            (
                guid,
                source_id,
                title,
                published_utc,
                _now_iso(),
                status,
                detail,
                document_id,
                duration_seconds,
                audio_url,
            ),
        )
        self.conn.commit()

    def episode_counts(self, source_id: str | None = None) -> dict[str, int]:
        """Episodes by status, for the run summary and for `--status`."""
        sql = "SELECT status, COUNT(*) AS n FROM podcast_episodes"
        params: tuple = ()
        if source_id:
            sql += " WHERE source_id = ?"
            params = (source_id,)
        sql += " GROUP BY status"
        return {r["status"]: r["n"] for r in self.conn.execute(sql, params)}

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
    # Which side of the comparison a persona sits on.
    PERSONA_FAMILY_PREFIX = "persona_family:"
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

    def set_persona_family(self, persona_id: str, family: str) -> None:
        self.set_meta(f"{self.PERSONA_FAMILY_PREFIX}{persona_id}", family)

    def persona_families(self) -> dict[str, str]:
        """``{persona_id: family}``. What lets a surface colour by side rather
        than by position in a list — the bug ``digest/render.py`` documents."""
        return self._labels(self.PERSONA_FAMILY_PREFIX)

    def set_source_label(self, source_id: str, name: str) -> None:
        self.set_meta(f"{self.SOURCE_LABEL_PREFIX}{source_id}", name)

    def source_labels(self) -> dict[str, str]:
        """``{source_id: display name}`` as the registry writes them."""
        return self._labels(self.SOURCE_LABEL_PREFIX)

    def _labels(self, prefix: str) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM meta WHERE key LIKE ?", (f"{prefix}%",))
        return {r["key"][len(prefix) :]: r["value"] for r in rows if r["value"]}

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
        """Yield (document_id, source_id, title, _, vector) for non-duplicate docs.

        Keyed by source rather than diet: a document belongs to every persona that
        reads its source, so clustering resolves membership per persona afterwards
        instead of being handed one diet per document.

        When ``embedder`` is given, only rows produced by that embedder are
        yielded — embeddings from different embedders live in incompatible vector
        spaces (and often different dimensions), so clustering must not mix them.
        """
        sql = """
            SELECT e.document_id AS id, d.source_id AS source_id, d.title AS title,
                   e.vector AS vector
            FROM embeddings e JOIN documents d ON d.id = e.document_id
            WHERE d.is_duplicate = 0
        """
        params: tuple[str, ...] = ()
        if embedder is not None:
            sql += " AND e.embedder = ?"
            params = (embedder,)
        for r in self.conn.execute(sql, params):
            yield r["id"], r["source_id"], r["title"], [], json.loads(r["vector"])

    def embedder_names(self) -> list[str]:
        """Distinct embedder names present in the embeddings table."""
        rows = self.conn.execute("SELECT DISTINCT embedder FROM embeddings ORDER BY embedder")
        return [r["embedder"] for r in rows]

    def embedding_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"]

    # -- clusters --------------------------------------------------------
    def replace_clustering(
        self,
        clusters: list[tuple[int, str | None, int]],  # (cluster_id, label, size)
        assignments: list[tuple[str, int]],  # (document_id, cluster_id)
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
        self,
        themes: list[tuple[str, str, str, str]],  # (document_id, key, title, method)
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
        """Members of a cluster with source, title, outlet, and link.

        The link is the point: a story is only checkable if the reader can go
        read it, and ``CLAUDE.md`` §0 permits exactly that — summarize and
        link, never republish.
        """
        return list(
            self.conn.execute(
                """
                SELECT d.id AS id, d.title AS title,
                       d.url AS url, d.source_id AS source_id
                FROM document_clusters dc JOIN documents d ON d.id = dc.document_id
                WHERE dc.cluster_id = ?
                ORDER BY COALESCE(d.published_utc, d.fetched_utc) DESC
                """,
                (cluster_id,),
            )
        )

    def cluster_source_counts(self) -> list[sqlite3.Row]:
        """``(cluster_id, source_id, n)`` for the whole clustering, in one query.

        What the agenda metrics run on. Asking per cluster instead would be one
        query per story on a corpus that has hundreds of them.

        Grouped by source, not by diet, because with a shared catalog a document
        belongs to every persona that reads its source — which ``GROUP BY
        diet_id`` cannot express. Callers cross these counts with each persona's
        membership set.

        **Near-duplicates count for their own outlet.** Only canonical documents
        are embedded and therefore clustered, so a wire story that ran in six
        outlets appears once in ``document_clusters``. Counting only that copy
        credited the story to whichever outlet happened to be fetched first and
        recorded the other five as never having covered it — which is how a story
        both sides carried came to read as one side's blindspot. The second leg of
        the union puts each collapsed copy's outlet back.
        """
        return list(
            self.conn.execute(
                """
                SELECT cluster_id, source_id, COUNT(*) AS n FROM (
                    SELECT dc.cluster_id AS cluster_id, d.source_id AS source_id
                    FROM document_clusters dc JOIN documents d ON d.id = dc.document_id
                    -- `is_duplicate = 0` is true by construction today, since only
                    -- canonicals are embedded and therefore only canonicals are
                    -- assigned. Asserted in the query anyway: were a duplicate ever
                    -- to land here it would be counted twice, once on each leg, and
                    -- a double-counted outlet is the failure this method exists to
                    -- fix rather than to introduce.
                    WHERE dc.cluster_id != -1 AND d.is_duplicate = 0
                      AND d.source_id != ''
                    UNION ALL
                    SELECT dc.cluster_id AS cluster_id, dup.source_id AS source_id
                    FROM document_clusters dc
                    JOIN documents dup ON dup.duplicate_of = dc.document_id
                    -- Same empty-source guard as `duplicate_coverage`: '' is not an
                    -- outlet, and bucketing it would put a nameless share into the
                    -- attention numbers.
                    WHERE dc.cluster_id != -1 AND dup.is_duplicate = 1
                      AND dup.source_id != ''
                )
                GROUP BY cluster_id, source_id
                """
            )
        )

    def duplicate_coverage(self) -> dict[str, set[str]]:
        """``{canonical_document_id: {source_id, ...}}`` for collapsed near-duplicates.

        Deduplication is right for scoring and wrong for coverage, and the two need
        different answers from the same fact. Collapsing the same wire story into
        one document keeps one outlet's moral vocabulary from being counted six
        times; it must not also mean the other five outlets never ran it. The
        outlets are recorded here so "did this reach them at all" — the binary
        question blindspots ask — can be answered from what was actually fetched.

        One level deep by construction: the near-duplicate index is seeded only
        from non-duplicates (:meth:`iter_minhash_signatures` filters them), so
        ``duplicate_of`` always names a canonical and never another duplicate.
        """
        rows = self.conn.execute(
            "SELECT duplicate_of AS canonical, source_id FROM documents "
            "WHERE is_duplicate = 1 AND duplicate_of IS NOT NULL AND source_id != ''"
        )
        out: dict[str, set[str]] = {}
        for row in rows:
            out.setdefault(row["canonical"], set()).add(row["source_id"])
        return out

    def duplicates_of(self, document_ids: Iterable[str]) -> dict[str, list[sqlite3.Row]]:
        """``{canonical_id: [duplicate rows]}`` with title, url and source.

        The collapsed copies are real articles at real URLs. A blindspot card's
        claim is "these outlets carried it and yours did not", so the outlet list
        has to include them or the claim is checkable against less than it should
        be.
        """
        ids = tuple(dict.fromkeys(document_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"""
            SELECT id, duplicate_of, title, url, source_id FROM documents
            WHERE is_duplicate = 1 AND duplicate_of IN ({placeholders})
            ORDER BY COALESCE(published_utc, fetched_utc) DESC
            """,
            ids,
        )
        out: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            out.setdefault(row["duplicate_of"], []).append(row)
        return out

    def source_ids_present(self) -> list[str]:
        """Sources that actually yielded documents.

        Not the authority on which personas exist — the registry is. A persona
        whose feeds were all unreachable today still exists, and inferring the
        persona list from the corpus would drop it and silently reshape every
        comparison.
        """
        rows = self.conn.execute("SELECT DISTINCT source_id FROM documents ORDER BY source_id")
        return [r["source_id"] for r in rows]

    def counts(self) -> dict[str, int]:
        total = self.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        dups = self.conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE is_duplicate = 1"
        ).fetchone()["n"]
        return {"documents": total, "duplicates": dups, "unique": total - dups}
