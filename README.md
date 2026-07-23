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

Phase 0 — corpus definition and repository scaffolding. See `CLAUDE.md` for the full
build spec and phased roadmap.

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
