# Parallax — Project Plan

> A moral-foundations mirror for two media diets. Ingests news, podcasts, and video
> from two modeled information environments, scores them on moral foundations, and
> reports what each one covers that the other doesn't.

This document is the build spec. It is written to be handed to a coding agent.

---

## 0. Framing and guardrails (read first)

**What this is.** A personal research tool that compares the author's own media diet
against a *representative model* of a conservative-evangelical media diet, using
Moral Foundations Theory (MFT) as the analytical lens.

**What this is not.** It does **not** track, surveil, or profile specific individuals.
The "other" diet is a documented, versioned model of outlets and programs — not any
real person's consumption. No private family communications are ingested, ever.

**Tone requirement.** The goal is charitable understanding. Any generated summary must
steelman each side's framing. Binding foundations (loyalty, authority, sanctity) are
sincere moral commitments in MFT's framework, not deficits. Do not generate copy that
mocks, pathologizes, or "dunks on" either diet.

**Symmetry requirement.** The identical pipeline runs on both diets. The author's own
blindspots and foundation skew are surfaced with equal prominence.

**Content handling.** Summarize and link; never republish. Persist derived metrics
(scores, aggregates, cluster metadata) — treat raw article text as a transient
processing artifact. Honor `robots.txt`, rate limits, and each source's terms.

---

## 1. Repository setup

- **License:** AGPL-3.0. Full license text in `LICENSE` (use GitHub's picker so it's
  detected). Verify no vendored dependency conflicts (MIT/Apache-2.0 deps are fine).
- **Visibility:** public.
- **README:** opens with the framing note in §0 — representative model, not individuals;
  charitable intent. This sentence determines how a stranger reads the whole repo.
- **`LIMITATIONS.md`:** stub it on day one. MFT text-scoring validity caveats (§5) live
  here and get updated as validation results come in.
- **Never commit:** article text, transcripts, audio, scraped bias-rating tables, model
  weights, `.env`. See `.gitignore` below.
- **Do commit:** the source registry (`config/sources.yaml`), methodology docs,
  validation notebooks and results, all code.

### Suggested layout

```
parallax/
├── config/
│   ├── sources.yaml          # THE source registry — both diets, versioned
│   └── settings.example.yaml
├── ingestion/                # Python: RSS, GDELT, Media Cloud, audio, YouTube
├── scoring/                  # Python: moral-foundations scoring service
├── cluster/                  # Python: embeddings, UMAP+HDBSCAN, blindspot detection
├── compare/                  # Python/R: JSD, CLR, log-ratios, profile math
├── summarize/                # Python: map-reduce LLM summarization
├── dashboard/                # TypeScript + D3.js static site
├── validation/               # hand-coded gold set, agreement metrics, notebooks
├── data/                     # gitignored
│   └── .gitkeep
├── LICENSE
├── LIMITATIONS.md
└── README.md
```

### `.gitignore`

```gitignore
# ---- Data: never commit ----
data/
!data/.gitkeep
corpus/
audio/
transcripts/
*.sqlite
*.db
*.parquet
*.jsonl
*.csv

# ---- Models & caches ----
models/
.cache/
*.pt
*.safetensors
.embeddings/

# ---- Secrets ----
.env
.env.*
!.env.example
*.key
credentials.json

# ---- Python ----
__pycache__/
*.py[cod]
.venv/
venv/
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/

# ---- TypeScript / dashboard ----
node_modules/
dist/
build/
.next/
.astro/

# ---- OS / editor ----
.DS_Store
.vscode/
.idea/
```

Add `pre-commit` with `gitleaks` or `detect-secrets` to catch keys before commit.

---

## 2. Architecture

Language split: **Python** owns ingestion, NLP, scoring, and comparison.
**TypeScript + D3.js** owns the dashboard. **R** is for statistical exploration.
**Rust** is an optional accelerator (PyO3) only if a hot loop demands it — not in scope
for the MVP.

```
  RSS / GDELT / Media Cloud / podcasts / YouTube
                    │
              [ ingestion ]  feedparser, httpx, trafilatura,
                    │        faster-whisper, youtube-transcript-api
                    ▼
              [ datastore ]  SQLite (MVP) → Postgres + pgvector
                    │        Parquet rollups, queried via DuckDB/Polars
        ┌───────────┼───────────┐
        ▼           ▼           ▼
  [ dedup ]   [ scoring ]  [ embeddings ]
  hash +      eMFD +       sentence-transformers
  MinHash     transformer +
              Claude
                    │           │
                    │           ▼
                    │     [ clustering ]  UMAP → HDBSCAN
                    │           │         → blindspot detection
                    ▼           ▼
              [ compare ]  JSD, CLR/Aitchison, per-foundation log-ratios
                    │
              [ summarize ]  Claude map-reduce
                    │
              [ dashboard ]  static site + D3: radar, diverging bars, JSD time series
```

### Component notes

**Ingestion**
- RSS/Atom via `feedparser` — primary, cheapest, most respectful of publisher intent.
- Article body extraction via `trafilatura` (fallback: `goose3`).
- GDELT DOC 2.0 API for discovery/metadata; Media Cloud (`pip install mediacloud`) for
  corpus construction and attention validation.
- Podcasts/talk radio: pull audio from podcast RSS enclosures → transcribe with
  **faster-whisper** (large-v3 for accuracy, medium for speed, int8 on CPU for
  overnight batch). Filter long silences — Whisper hallucinates on them.
- YouTube: `youtube-transcript-api` first; fall back to `yt-dlp` + faster-whisper when
  captions are absent.

**Datastore**
- Start with SQLite. Move to Postgres + `pgvector` when you want co-located embeddings.
- Raw audio/transcripts to local disk or object storage, outside git.
- Daily scored outputs to Parquet so weekly/monthly/yearly rollups are cheap
  aggregations, not re-scoring runs.

**Dedup**
- Exact: content hash. Near-dup: MinHash (`datasketch`) or embedding cosine threshold.
- Essential — syndicated wire copy repeats heavily across outlets and will otherwise
  skew every aggregate.

**Clustering / blindspot engine**
- `sentence-transformers` (`all-MiniLM-L6-v2` for speed; a `bge`/`e5` model for quality)
  → UMAP → HDBSCAN. `BERTopic` wraps both if you want the shortcut.
- Each cluster = a "story." Compute coverage share per diet.
- **Blindspot = a cluster with heavy coverage in one diet and near-zero in the other.**
  Emit both directions: what they see that I don't, and what I see that they don't.
- Tune `min_cluster_size` and UMAP `n_neighbors`/`n_components` — defaults over-assign
  short news items to noise.

**Summarization**
- Claude map-reduce: *map* = per-article/per-cluster summary; *reduce* = cluster →
  diet → cross-diet executive summary.
- Prefer a deterministic reduce over structured JSON where possible, to limit variance.
- Every summary includes verbatim quotes with links so it's auditable against source.
- Claude also handles: steelmanning each side's framing, tagging liberty/oppression
  (dictionaries don't cover it), and adjudicating disagreements between taggers.

**Dashboard**
- Static site (Astro/SvelteKit) reading generated JSON. No backend needed for personal use.
- D3 views: radar chart of the two 6-vectors; diverging bar chart of per-foundation
  log-ratios; JSD time series; blindspot lists with drill-down to cluster summaries.
- Surface confidence bands on every foundation score (see §5).

---

## 3. Moral foundations layer

### Foundations modeled
care/harm, fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, liberty/oppression.

Two refinements worth encoding:
- **Split fairness into Equality vs Proportionality** (Atari & Haidt 2023) — liberals
  favor equality, conservatives proportionality. Directly relevant here.
- **Sanctity/purity is especially central to evangelical discourse** — handle with care
  and don't collapse it into a generic "moralizing" signal.

### (a) Article tagging
Multi-label tags over the 6 foundations (virtue and vice poles → up to 12 labels),
following the Moral Foundations Reddit Corpus taxonomy (incl. "Thin Morality" and
implicit/explicit flags). Three complementary taggers, ensembled:

1. **Dictionary baseline** — `eMFDscore` (spaCy-based; eMFD, MFD, MFD 2.0). Returns 5
   foundation probabilities + sentiment + moral/non-moral word ratio. Its PAT mode
   extracts moral agent/patient/attribute relations — useful for "who is framed as
   harming whom." Covers only the 5 classic foundations; **no liberty/oppression**.
2. **Transformer** — Mformer (fine-tuned RoBERTa) or MoralBERT; or an off-the-shelf HF
   multi-label model trained on MFRC. `moralstrength` is a lexicon+embedding hybrid
   alternative.
3. **LLM** — Claude with a structured rubric, producing a tag plus a short rationale and
   supporting quote for auditability. Owns liberty/oppression coverage.

**Always length-normalize.** Raw dictionary counts correlate with document length
(r up to ~0.98 for eMFD). This is the single most common way to get garbage results.

### (b) Scoring and comparison pipeline

1. Score every document → 6-dim foundation vector (net virtue−vice, or salience).
2. Aggregate to a **diet-level profile** per period: length-normalize, then average
   (optionally weight by estimated reach/consumption).
3. Normalize each profile to a composition summing to 1.
4. **Headline metric: Jensen-Shannon divergence.** `scipy.spatial.distance.jensenshannon`
   returns the *distance* (square root) — square it for divergence. Base 2 → bounded
   [0,1], symmetric, linearly decomposable into per-foundation contributions.
5. **Interpretable over/under-indexing:** per-foundation log-ratio `ln(P_i / Q_i)`
   (positive = your diet over-indexes vs the modeled diet). An index form
   `100 × (P_i / Q_i)` (100 = parity) reads well on a dashboard.
6. **Compositionally correct distance:** Aitchison — CLR-transform both vectors
   (`skbio.stats.composition.clr` with multiplicative zero-replacement) then Euclidean.
   CLR/ILR also enable clean PCA over time.
7. **Sanity checks:** cosine distance and smoothed KL (add ε first). These correlate
   very highly with JSD; JSD alone is defensible, but the log-ratios carry the
   human-readable story.

---

## 4. Phased roadmap

**Phase 0 — Corpus definition (days).**
Finalize both source lists. Write `config/sources.yaml` with rationale per source.
Wire RSS + a few GDELT/Media Cloud queries. *Deliverable:* reproducible source registry
+ daily article dump to SQLite.

The conservative-evangelical model should stratify by medium and role — national cable,
talk radio, digital outlets, podcasts/YouTube, newsletters/aggregators — and represent
internal diversity rather than collapsing to one voice. Document weighting choices.

**Phase 1 — MVP (1–2 weeks).**
Extraction + dedup + eMFDscore dictionary scoring + a daily Claude summary per diet + a
static page with two radar charts and a JSD number. Text-only sources.
*Advance when:* summaries are coherent and the JSD number moves sensibly with real events.

**Phase 2 — Blindspot engine (2–4 weeks).**
Embeddings + UMAP + HDBSCAN across both diets; coverage-asymmetry detection;
"what they see / what I see" lists; cluster-level map-reduce summaries.
*Advance when:* clusters are coherent on a 20-item spot check and the asymmetry lists
match intuition.

**Phase 3 — Better morality scoring (3–6 weeks).**
Add the transformer tagger, build the hand-coded validation set (§5), ensemble all three
taggers, add liberty/oppression via Claude, add the Equality/Proportionality split.
*Trigger:* the gold set shows dictionary-only per-foundation AUC below ~0.7 on binding
foundations — which the literature suggests is likely.

**Phase 4 — Multimedia (ongoing).**
faster-whisper for podcasts/talk radio + `youtube-transcript-api`. This is where the
modeled diet becomes realistic and where the project differentiates from every text-only
tool. *Check:* compare foundation profiles with vs without multimedia — if they differ
materially, multimedia is essential rather than optional.

**Phase 5 — Cadence + polish.**
Weekly/monthly/yearly rollups, trend tracking (does the gap widen around elections?),
richer D3 dashboard, optional personal email digest.

---

## 5. Validation and known limits

**Build a gold set.** Hand-code a stratified random sample (200–400 articles across both
diets and all foundations) using MFRC annotation guidelines. Compute agreement between
you and each automated method (Krippendorff's alpha / Cohen's kappa; AUC and F1 per
foundation). Expect binding foundations (loyalty, sanctity) to be the weakest.

**Known measurement limits — put these in `LIMITATIONS.md` and on the dashboard:**
- Dictionary methods have poor convergent validity; MFD and MFD 2.0 don't reliably agree
  with each other, and dictionary measures often fail to correlate with human coders.
- eMFD word overlap with the earlier dictionaries is minimal.
- Embedding-similarity scoring performs near chance.
- Fine-tuned transformers do meaningfully better in-domain but degrade across domains.
- Therefore: **treat every foundation number as an estimate with uncertainty, never as
  ground truth.** Use ensemble *disagreement* as a confidence signal — when the
  dictionary, transformer, and Claude diverge, flag the item as low-confidence rather
  than forcing a label.

**Theory caveats to encode honestly.** The liberal/conservative foundation asymmetry
(Graham, Haidt & Nosek 2009) is real but contested in magnitude — competing work argues
the two groups rely on very similar sets of foundations. Present the asymmetry as a
hypothesis the tool *tests*, not an axiom it assumes.

**Guard against your own confirmation bias.**
- Pre-register what you expect to find each period, then check yourself.
- Run the identical pipeline on your own diet and display your blindspots with equal prominence.
- Periodically have the LLM critique your framing choices and flag where *your* diet is
  the outlier.

**Representativeness.** Version the source model; sensitivity-test conclusions against
source weighting. Validate against external benchmarks (Pew on evangelical media use;
Media Cloud attention data). Distinguish *outlet* bias from *audience* consumption — model
what is actually consumed, not merely what is published.
