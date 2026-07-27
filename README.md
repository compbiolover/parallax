# Parallax

> A moral-foundations mirror for two media diets.

**Read this first.** Parallax compares the author's own media diet against a
*representative, documented model* of a conservative-evangelical media diet, using
Moral Foundations Theory (MFT) as the analytical lens. It ingests news, podcasts, and
video from two modeled information environments, scores them on moral foundations, and
reports what each one covers that the other does not.

## What this is, and what it is not

It is a personal research tool that models and compares two *media diets*. It is not a
system that tracks, surveils, or profiles any specific individual. The "other" diet is a
versioned model of outlets and programs ([`config/sources.yaml`](config/sources.yaml)),
not any real person's consumption. No private family communications are ever ingested.

## Principles

**Charitable understanding.** Every generated summary steelmans each side's framing. The
binding foundations (loyalty, authority, sanctity) are sincere moral commitments in MFT's
framework, not deficits. Parallax does not mock, pathologize, or "dunk on" either diet.

**Symmetry.** The identical pipeline runs on both diets. The author's own blindspots and
foundation skew are surfaced with equal prominence.

**Content handling.** Summarize and link, never republish. Derived metrics (scores,
aggregates, cluster metadata) are persisted; raw article text is a transient processing
artifact. `robots.txt`, rate limits, and each source's terms are honored.

**Uncertainty is first-class.** Every foundation number is an estimate with a confidence
band, never ground truth. See [`LIMITATIONS.md`](LIMITATIONS.md).

## Architecture

Python owns ingestion, NLP, scoring, and comparison. TypeScript and D3.js own the
dashboard. R is there for statistical exploration.

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
| `compare/`     | JSD, CLR/Aitchison distance, log-ratios, dated snapshot history.      |
| `summarize/`   | Map-reduce LLM summarization.                                         |
| `dashboard/`   | TypeScript + D3.js static site.                                      |
| `daily/`       | One-command snapshot orchestrator (`make daily`).                     |
| `validation/`  | Hand-coded gold set, agreement metrics, notebooks.                    |
| `data/`        | Gitignored working data.                                              |

## Moral foundations modeled

care/harm, fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, and liberty/oppression, with fairness split into Equality and
Proportionality (Atari & Haidt 2023).

### Fairness, split two ways

MFQ-2 divides Fairness into equality (equal treatment, equal outcomes) and
proportionality (reward tracking merit and contribution). The distinction earns its place
here. Both diets argue about fairness constantly, and a five-way tagger reports that as
one number, which hides the more interesting fact: they often mean different things by it.
One asks who is being left out. The other asks whether reward is tracking contribution.
Each is a coherent account of fairness, and neither reduces to the other.

The dashboard shows the division per diet, with coverage next to it. Read it with more
suspicion than anything else on the page. No validated dictionary implements this split,
so Parallax partitions fairness using a hand-built term list, and the prediction it tests
(equality on the left, proportionality on the right) replicates poorly when measured in
language. See `LIMITATIONS.md`.

```yaml
scoring:
  taggers:
    dictionary:
      split_fairness: true
      fairness_min_evidence: 2   # split-terms required before partitioning at all
```

Building this turned up a bug worth mentioning, because it is the kind that hides well.
The built-in seed lexicon's fairness vocabulary contained no proportionality terms at all,
so a merit-framed argument scored as containing *no fairness*. That systematically
under-measured whichever diet frames fairness as proportion: the symmetry requirement
failing quietly inside a word list.

Since a word list can carry a political asymmetry while looking like a neutral instrument,
that check is now a command rather than a memory:

```bash
make audit-lexicon                                            # the built-in seed
python -m validation.lexicon_audit --lexicon data/emfd_scoring.csv
```

It reports whether either half of fairness is missing outright, which is the seed's
original failure, and how much fairness each half's vocabulary contributes per occurrence.
Run against the real eMFD, the categorical bug does not reproduce: merit vocabulary is
there and merit-framed text does score fairness. A milder tilt in the same direction does
show up, because half the merit terms the eMFD contains are assigned to other foundations
(*accountable* to authority, *contribution* to loyalty, *effort* to care) and contribute
nothing to fairness. `LIMITATIONS.md` has the numbers and what they do and don't support.

### Liberty, the foundation nothing else can score

The eMFD, the MFD, and MFD 2.0 all cover five foundations. Mformer's training corpus does
not label liberty either. So liberty/oppression falls to Claude, which is what `CLAUDE.md`
§3(a) assigns it to and what `scoring/liberty.py` implements.

The rubric is the part worth reading. Liberty is claimed in two registers: freedom from
state coercion (mandates, overreach, censorship, conscience claims) and freedom from
private or structural domination (corporate power, surveillance, bodily autonomy, employer
control). A rubric that recognized only one would score one diet as caring about liberty
and the other as silent — a coverage artifact wearing the clothes of a finding, and the
same failure the fairness lexicon audit caught. Both registers are named explicitly, and a
test asserts they stay named.

A test can only check that the words are in the prompt, though — not that the model obeys
them. So there is a probe for that too:

```bash
make register-probe                 # 10 pairs x 2 registers x 3 repeats = 60 calls
```

Ten sentences, each with a single `{actor}` slot, rendered once with a state actor and once
with a private one. The two sides are identical word for word because they come from the
same string — matched by construction rather than by my judgement, since hand-writing two
"equivalent" sentences would put my own framing instincts inside the instrument. It reports
whether the registers get classified correctly, and whether one of them scores
systematically higher, with the run-to-run noise floor printed next to the gap so a few
samples don't get read as a result. A gap under the noise is absence of evidence, not a
clean bill of health, and the report says so in those words.

```yaml
scoring:
  taggers:
    liberty:
      enabled: true
      model: claude-sonnet-5
      effort: low       # this matters: adaptive thinking at default `high` bills far more
      batch: true       # Batch API — half price, and the daily run is overnight anyway
```

It needs `ANTHROPIC_API_KEY`. Without one the run completes with five foundations rather
than failing. At roughly 200 documents a day the batched cost is about $17/month on
Sonnet 5, $8 on Haiku 4.5, $42 on Opus 5 — but don't take the tier on faith:

```bash
python -m validation --scorer liberty --model claude-haiku-4-5 --limit 40
python -m validation --scorer liberty --model claude-sonnet-5  --limit 40
```

That reports AUC/F1/kappa against hand-coded liberty labels, the same as every other
scorer. It needs those labels to exist first — no public corpus supplies them, so the gold
set has to be coded for liberty by hand, and the CLI says so rather than scoring against
all-zeros and printing a number that means nothing.

Liberty appears as its own panel rather than a sixth spoke on the radar. The radar is a
composition over documents every tagger saw; liberty is scored on feed-ingested documents
only, and only when a key is set. Folding partial coverage into a composition would move
every other share, and would shift the headline divergence that the snapshot history has
been recording since the series started.

## Status

Phase 1 (MVP) and Phase 2 (blindspot engine) are complete: extraction, dedup, dictionary
scoring, a daily summary per diet, a static radar/JSD dashboard, and coverage-asymmetry
blindspot detection, plus GDELT historical backfill for weeks of per-outlet volume.

Phase 3 adds the transformer tagger (Mformer), a validation gold set, and an ensemble
confidence signal wired to the dashboard. With the transformer running at ingestion, each
foundation on the radar carries a dictionary-vs-transformer band, where wider means more
disagreement and lower confidence. Every run also leaves a dated snapshot behind, so the
dashboard plots divergence over time rather than only ever showing today. That is the
first piece of Phase 5's cadence work. See `CLAUDE.md` for the full build spec and
roadmap.

### The daily snapshot (one command)

```bash
make daily          # or: python -m daily
```

That runs the whole chain, ingest through GDELT backfill, cluster, summarize, snapshot,
and export, and leaves a refreshed dashboard. Then `make dashboard` and open
<http://localhost:8000>.

Steps are isolated. If one fails (GDELT throttling, no API key), the rest still run so the
dashboard reflects whatever data landed, the report names what broke, and the exit code is
non-zero so cron notices. The transformer is loaded once and shared across steps.

```bash
make daily-fast                      # today's feeds only; skips the slow backfill
python -m daily --skip backfill
python -m daily --only cluster export
python -m daily --backfill-days 3 --max-per-source 100   # cheaper daily window
```

The backfill is included by default because blindspots are only trustworthy with real
per-outlet volume behind them. Tune it in `config/settings.yaml` under `daily.backfill`,
or disable it there. Re-runs are idempotent thanks to URL-canonical dedup, so overlapping
windows cost time, not correctness.

Expect the full run to take a while. It is a batch job, not an interactive command. Two
things dominate: transformer scoring runs five RoBERTa models over every ingested article
(seconds per article on CPU), and GDELT's free endpoint throttles to roughly one request
every five seconds across the whole registry. That is the cost of confidence bands and
trustworthy blindspots. If you want it quicker, run it overnight from cron, use `make
daily-fast` to skip backfill, shorten the window with `--backfill-days 3`, or pass
`--no-transformer` for a fast dictionary-only refresh with no confidence bands.

Run it every morning with cron:

```cron
0 6 * * *  cd /path/to/parallax && /path/to/.venv/bin/python -m daily >> data/daily.log 2>&1
```

### Divergence over time

Each run records one dated snapshot of compositions, JSD, log-ratios, and document counts,
so the numbers accumulate into a series instead of overwriting yesterday. Re-running on the
same day replaces that day's row rather than adding one, so the series stays at one point
per day however often the pipeline runs.

Every snapshot is computed on two bases, because they answer different questions:

| Basis | What it profiles | How to read it |
| ----- | ---------------- | -------------- |
| Trailing window (7 days by default) | Only documents dated in the window | The basis that can actually respond to an event. Noisier, and on thin days it *is* noise. |
| All-time | Every document dated on or before that day | The headline number the radar reports. Heavily damped, and it moves less the longer the project runs. |

Neither is the truer number, and the dashboard labels both rather than picking one.

```bash
python -m compare.history                      # print the recorded series
python -m compare.history --backfill 30        # reconstruct 30 past days
python -m daily --window-days 14               # widen the trailing window
python -m dashboard.export --history-limit 90  # ship 90 days to the dashboard
```

`--backfill` reconstructs past days from publication dates already in the store, so the
chart is useful on day one rather than after a month of runs. It computes the same
arithmetic on a different corpus, though. A reconstructed row for last Tuesday includes
articles published then but fetched since (GDELT backfill pulls weeks of history), which a
live run that day could not have seen. Reconstructed rows are therefore *what the corpus
now says about that date*, not what the dashboard would have shown. The chart shades them,
and reconstruction will not overwrite a live row unless you pass `--overwrite`.

### Running the steps individually

```bash
# 1. Fetch every RSS source with a URL, extract bodies, dedup, score, embed, and
#    store derived metrics to SQLite (raw text is never persisted). Every article
#    is scored by BOTH the dictionary and the transformer (Mformer) so the
#    dashboard can show a dictionary-vs-transformer confidence band; add
#    --no-transformer for the fast dictionary-only path (parallax[scoring] needed
#    for the transformer, else it degrades to dictionary-only automatically):
python -m ingestion run --max-items 25

# 1b. Backfill weeks of history per outlet from GDELT (title-based, so it's fast
#     and needs no API key). This is the volume that makes blindspots reliable:
python -m ingestion backfill --days 14 --max-per-source 250
#     (add --extract to also fetch article bodies for full scoring; slower)

# 2. Print each diet's foundation composition, the Jensen-Shannon divergence,
#    and the per-foundation log-ratios:
python -m ingestion compare

# 3. Cluster stories from the stored embeddings and detect blindspots: the
#    clusters one diet covers heavily and the other barely touches, both
#    directions (scikit-learn is a core dependency):
python -m cluster run

# 4. Generate a charitable daily summary per diet + a cross-diet executive
#    summary (uses Claude when ANTHROPIC_API_KEY is set, else a deterministic,
#    clearly-labeled numbers-only fallback):
python -m summarize

# 5. Export the dashboard data payload:
python -m dashboard.export

# 6. View the dashboard (radar, JSD, log-ratio bars, summaries, blindspot lists):
cd dashboard && python -m http.server   # then open http://localhost:8000

# Validate a scorer against the hand-coded gold set: per-foundation AUC/F1/kappa
# and the §5 trigger (binding foundations below 0.7 AUC warrant the transformer):
python -m validation --lexicon data/emfd_scoring.csv
python -m validation --scorer transformer   # Mformer; needs parallax[scoring]
python -m validation --scorer ensemble      # dictionary + transformer, with a
                                            # disagreement-based confidence report
```

Story clustering embeds each document at ingestion (text is discarded, so embeddings are
persisted). The default embedder is a dependency-free hashing embedder over headlines.
Setting `cluster.embedder.kind: sentence-transformers` swaps in neural embeddings for
sharper clusters (`pip install parallax[embeddings]`). See `LIMITATIONS.md` for what the
current clusters do and don't support.

By default scoring uses a built-in demo lexicon so the pipeline runs with zero external
data. It is a placeholder, not a validated instrument. For real results, supply the eMFD:

```bash
# 1. Drop the eMFD CSV in data/ (gitignored), from the eMFDscore repo:
#    dictionaries/emfd_scoring.csv (columns: word, <foundation>_p, <foundation>_sent).
# 2. Either set scoring.dictionary.lexicon_path in config/settings.yaml, or:
python -m ingestion run --lexicon data/emfd_scoring.csv
```

The active lexicon is recorded with the scores, so the dashboard caveat and summaries
state which one produced the numbers. Because eMFD words carry probability across all five
foundations, the scorer defaults to `assignment: argmax`, where each word counts toward
its dominant foundation. See `LIMITATIONS.md` for why, and for what the eMFD's low
aggregate discrimination does and doesn't mean. The dictionary baseline covers the five
classic foundations only; liberty/oppression arrives with the Claude tagger in Phase 3.

## Getting started

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install the package. `llm` brings in the anthropic SDK, which the liberty
#    tagger and the Claude summaries both need — without it they degrade to
#    "not scored" and "numbers-only" respectively.
pip install -e ".[dev,llm]"

# 3. Install the pre-commit hooks (secret scanning)
pre-commit install

# 4. Copy the example configuration and fill in your own diet
cp config/settings.example.yaml config/settings.yaml
```

### The API key

Anything Claude-backed reads `ANTHROPIC_API_KEY` **from the environment**.
`.env.example` documents which keys exist, but nothing in this repo loads a `.env`
file — export the key in your shell instead, and put it in the crontab for scheduled
runs, since cron does not inherit your shell environment.

```bash
# in ~/.zshrc (or ~/.bashrc) — editing the file beats typing `export`,
# which would leave the key in your shell history
export ANTHROPIC_API_KEY=sk-ant-...
```

Then check the whole path — key, SDK, rubric, structured output, parser — with one
call costing a fraction of a cent:

```bash
python -m scoring.liberty
```

It prints a scored probe if everything is wired up, and names the missing piece if not.
Without a key the pipeline still runs; you get five foundations instead of six, and a
warning saying why.

## License

[AGPL-3.0](LICENSE). Derived metrics and code are shareable under its terms; raw source
content is never committed or redistributed.
