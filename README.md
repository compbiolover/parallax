# Parallax

> A moral-foundations mirror for two media diets.

**Read this first.** Parallax compares the author's own media diet against a
*representative, documented model* of a conservative-evangelical media diet, using
Moral Foundations Theory (MFT) as the analytical lens. It ingests news, podcasts, and
video from two modeled information environments, scores them on moral foundations, and
reports what each one covers that the other does not.

## What this is — and is not

- **It is** a personal research tool that models and compares two *media diets*.
- **It is not** a system that tracks, surveils, or profiles any specific individual.
  The "other" diet is a versioned model of outlets and programs
  ([`config/sources.yaml`](config/sources.yaml)) — not any real person's consumption.
  No private family communications are ever ingested.

## Principles

- **Charitable understanding.** Every generated summary steelmans each side's framing.
  The binding foundations (loyalty, authority, sanctity) are sincere moral commitments
  in MFT's framework, not deficits. Parallax does not mock, pathologize, or "dunk on"
  either diet.
- **Symmetry.** The identical pipeline runs on both diets. The author's own blindspots
  and foundation skew are surfaced with equal prominence.
- **Content handling.** Summarize and link; never republish. Derived metrics (scores,
  aggregates, cluster metadata) are persisted; raw article text is a transient
  processing artifact. `robots.txt`, rate limits, and each source's terms are honored.
- **Uncertainty is first-class.** Every foundation number is an estimate with a
  confidence band, never ground truth. See [`LIMITATIONS.md`](LIMITATIONS.md).

## Architecture

Language split: **Python** owns ingestion, NLP, scoring, and comparison.
**TypeScript + D3.js** owns the dashboard. **R** is for statistical exploration.

```
  RSS / GDELT / Media Cloud / podcasts / YouTube
                    │
              [ ingestion ]      → ingestion/
                    ▼
              [ datastore ]      SQLite (MVP) → Postgres + pgvector
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    [ dedup ]   [ scoring ]  [ embeddings ]   → scoring/
                    │           │
                    │      [ clustering ]      → cluster/ (blindspot detection)
                    ▼           ▼
              [ compare ]  JSD, CLR/Aitchison  → compare/
                    │
              [ summarize ]  Claude map-reduce  → summarize/
                    │
              [ dashboard ]  static site + D3   → dashboard/
```

| Directory      | Responsibility                                                        |
| -------------- | --------------------------------------------------------------------- |
| `config/`      | The source registry (`sources.yaml`) and example settings.            |
| `ingestion/`   | RSS, GDELT, Media Cloud, podcast audio, YouTube.                       |
| `scoring/`     | Moral-foundations scoring (dictionary + transformer + Claude).        |
| `cluster/`     | Embeddings, UMAP + HDBSCAN, blindspot detection.                      |
| `compare/`     | JSD, CLR/Aitchison distance, per-foundation log-ratios.               |
| `summarize/`   | Map-reduce LLM summarization.                                         |
| `dashboard/`   | TypeScript + D3.js static site.                                      |
| `validation/`  | Hand-coded gold set, agreement metrics, notebooks.                    |
| `data/`        | Gitignored working data.                                              |

## Moral foundations modeled

care/harm, fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, and liberty/oppression — with fairness optionally split into
Equality vs Proportionality (Atari & Haidt 2023).

## Status

Phase 1 (MVP) and Phase 2 (blindspot engine) are complete — extraction, dedup, dictionary
scoring, a daily summary per diet, a static radar/JSD dashboard, and coverage-asymmetry
blindspot detection — plus **GDELT historical backfill** for weeks of per-outlet volume.
See `CLAUDE.md` for the full build spec and phased roadmap.

### Running the pipeline

```bash
# 1. Fetch every RSS source with a URL, extract bodies, dedup, score, embed, and
#    store derived metrics to SQLite (raw text is never persisted):
python -m ingestion run --max-items 25

# 1b. Backfill weeks of history per outlet from GDELT (title-based, so it's fast
#     and needs no API key) — this is the volume that makes blindspots reliable:
python -m ingestion backfill --days 14 --max-per-source 250
#     (add --extract to also fetch article bodies for full scoring; slower)

# 2. Print each diet's foundation composition, the Jensen-Shannon divergence,
#    and the per-foundation log-ratios:
python -m ingestion compare

# 3. Cluster stories from the stored embeddings and detect blindspots — the
#    clusters one diet covers heavily and the other barely touches, both
#    directions (needs scikit-learn: pip install parallax[cluster]):
python -m cluster run

# 4. Generate a charitable daily summary per diet + a cross-diet executive
#    summary (uses Claude when ANTHROPIC_API_KEY is set, else a deterministic,
#    clearly-labeled numbers-only fallback):
python -m summarize

# 5. Export the dashboard data payload:
python -m dashboard.export

# 6. View the dashboard (radar, JSD, log-ratio bars, summaries, blindspot lists):
cd dashboard && python -m http.server   # then open http://localhost:8000
```

Story clustering embeds each document at ingestion (text is discarded, so embeddings are
persisted). The default embedder is a dependency-free hashing embedder over headlines;
`cluster.embedder.kind: sentence-transformers` swaps in neural embeddings for sharper
clusters (`pip install parallax[embeddings]`). See `LIMITATIONS.md` for what the current
clusters do and don't support.

By default scoring uses a **built-in demo lexicon** so the pipeline runs with zero
external data — a placeholder, not a validated instrument. For real results, supply the
eMFD:

```bash
# 1. Drop the eMFD CSV in data/ (gitignored) — from the eMFDscore repo,
#    dictionaries/emfd_scoring.csv (columns: word, <foundation>_p, <foundation>_sent).
# 2. Either set scoring.dictionary.lexicon_path in config/settings.yaml, or:
python -m ingestion run --lexicon data/emfd_scoring.csv
```

The active lexicon is recorded with the scores, so the dashboard caveat and summaries
state which one produced the numbers. Because eMFD words carry probability across all
five foundations, the scorer defaults to `assignment: argmax` (each word counts toward
its dominant foundation); see `LIMITATIONS.md` for why, and for what the eMFD's low
aggregate discrimination does and doesn't mean. The dictionary baseline covers the five
classic foundations only; liberty/oppression arrives with the Claude tagger in Phase 3.

## Getting started

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install the package with dev tooling
pip install -e ".[dev]"

# 3. Install the pre-commit hooks (secret scanning)
pre-commit install

# 4. Copy the example configuration and fill in your own diet
cp config/settings.example.yaml config/settings.yaml
cp .env.example .env
```

## License

[AGPL-3.0](LICENSE). Derived metrics and code are shareable under its terms; raw source
content is never committed or redistributed.
