# Limitations

Parallax measures a genuinely hard, contested construct — the moral framing of text —
with imperfect tools. This document records what those tools can and cannot support. It
is a living document: measurement caveats are stubbed here on day one (§5 of `CLAUDE.md`)
and updated as validation results come in. **Every foundation number the tool reports is
an estimate with uncertainty, never ground truth.**

## Measurement limits

- **Dictionary methods have poor convergent validity.** MFD and MFD 2.0 do not reliably
  agree with each other, and dictionary measures often fail to correlate with human
  coders. The eMFD's word overlap with the earlier dictionaries is minimal.
- **Length confound.** Raw dictionary counts correlate strongly with document length
  (r up to ~0.98 for eMFD). All scores are length-normalized before aggregation; skipping
  this is the single most common way to get garbage results.
- **Embedding-similarity scoring performs near chance** and is not used as a primary
  scorer.
- **Fine-tuned transformers do meaningfully better in-domain but degrade across
  domains.** In-domain gains do not transfer cleanly to out-of-domain outlets.
- **Coverage gaps.** The dictionary baseline (`eMFDscore`) covers only the five classic
  foundations — it has **no liberty/oppression**. That foundation is supplied by the
  Claude tagger.
- **Ensemble disagreement is the confidence signal.** When the dictionary, transformer,
  and Claude taggers diverge on an item, it is flagged low-confidence rather than forced
  into a label. Confidence bands are surfaced on every foundation score in the dashboard.

## Implementation status (Phase 1)

- **The bundled lexicon is a demo, not an instrument.** The Phase 1 dictionary scorer
  ships a small hand-built seed lexicon (`scoring/seed_lexicon.py`) so the pipeline runs
  out of the box. It is a few dozen stems with unit weights — it must **not** be read as
  a validated measurement. Real scoring requires the full eMFD (continuous per-word
  probabilities over ~10k words), loaded via `scoring.lexicon.load_emfd_csv`. Until then,
  every foundation number the pipeline emits is illustrative only.
- **Dictionary-only coverage.** The current scorer covers the five classic foundations;
  liberty/oppression is unscored (`None`, never `0`) until the Claude tagger lands.
- **Ensemble confidence signal exists and is calibrated; dashboard bands are not wired
  yet.** The dictionary + transformer ensemble (`scoring/ensemble.py`) flags an item
  low-confidence when the taggers split. On the seed gold set this signal is strongly
  meaningful: predictions where the taggers **agree** are 86% accurate; where they **split**,
  27% (a +0.59 gap). Two honest caveats: (1) the ensemble's *point estimate* AUC (macro
  0.86) sits **below the transformer alone** (0.95) — its contribution is the confidence
  flag, not a better score, so the transformer remains the best single scorer; (2) the
  confidence band is computed at evaluation time but is not yet surfaced on the dashboard,
  which would require running the transformer at ingestion (a performance decision) — that
  wiring is the remaining §5 step.
- **The validation gold set is a starter.** `validation/gold/seed.json` is 42 hand-coded
  items by a single coder — enough to run the harness and fire the §5 trigger, but far
  short of the 200–400 multi-coder items §5 targets. Agreement numbers below are indicative
  and will move as the gold set grows. With one coder, inter-coder reliability
  (Krippendorff's alpha) is not yet meaningful.
- **Measured: the dictionary is weak on the binding foundations; the transformer fixes
  it.** On the seed gold set, the real eMFD dictionary scores AUC 0.63 on both **loyalty**
  and **sanctity** — firing the §5 trigger — while the Mformer transformer tagger clears
  every binding foundation at ≥ 0.94 (macro-AUC 0.71 → 0.95). This is exactly the §5
  prediction, now empirical: dictionary numbers on the binding foundations should be
  treated with particular caution, and the transformer tagger (`--scorer transformer`,
  `pip install parallax[scoring]`) is the more trustworthy source there.
- **The eMFD is supported, and reveals its own low discriminative power.** Point the
  scorer at the real eMFD via `scoring.dictionary.lexicon_path`. Note two things the eMFD
  forces:
  - *Aggregation matters.* Every eMFD word carries probability mass on all five
    foundations, so **summing raw probabilities makes every document collapse toward the
    eMFD's base-rate distribution** and profiles stop discriminating between corpora.
    Parallax defaults to `assignment: argmax` (each word counts toward its dominant
    foundation only), which restores discrimination when diets genuinely differ.
  - *Real news barely differs in aggregate.* On a sample of live mainstream vs.
    conservative feeds, the two diets' aggregate eMFD profiles come out nearly identical
    (JSD ≈ 0) — both emphasize care > fairness > authority > loyalty > sanctity in similar
    proportions. A near-zero JSD does **not** mean the diets are identical; it means the
    aggregate dictionary signal cannot tell them apart. Sharper signal is expected from
    topic-level blindspot analysis (Phase 2) and the transformer/Claude taggers (Phase 3),
    not from pushing on the dictionary aggregate.

## Blindspot engine (Phase 2)

- **Clusters are only as good as the embedder.** The default embedder is a dependency-free
  feature hasher over **headlines** (bodies share boilerplate that washes out topic
  signal, so titles cluster far better). It captures obvious topical structure but produces
  loose or spurious clusters on subtler stories. `sentence-transformers` (config:
  `cluster.embedder.kind: sentence-transformers`) is the quality upgrade. Benchmarked on 70
  live stories (identical SVD→HDBSCAN pipeline, coherence = mean intra-cluster cosine
  similarity in sentence-transformer space, an independent semantic yardstick):

  | embedder | coherence | lift over random pairs | noise | one-sided clusters |
  | --- | --- | --- | --- | --- |
  | hashing (default) | 0.121 | +0.048 | 39% | 7 (some spurious) |
  | all-MiniLM-L6-v2 | **0.366** | **+0.293** | **26%** | 8 (cleaner; nuclear-deal story 3→4) |

  Sentence-transformers roughly **tripled cluster coherence and cut noise by a third**,
  and recovered a story the hashing embedder missed — worth the heavier `torch` dependency
  for real use. The residual loose size-2 clusters are a *data-volume* problem (more sources
  + accumulation lets you raise the min-cluster/min-blindspot thresholds), not an embedder
  one.

- **Which sentence-transformer?** `all-MiniLM-L6-v2` is the fast classic default, not the
  best available. Models were benchmarked with a *model-agnostic* metric: a hand-labeled gold
  set of same-story pairs scored by average precision (how well a model ranks same-story pairs
  above all others). The benchmark was run twice — and the bigger run **overturned** the
  first, a useful lesson about small samples. On **396 stories / 151 gold pairs across 28
  stories** (US–Saudi nuclear deal, Trump's 80-country tariffs, Houthi Red Sea attacks,
  Nolan's *Odyssey*, …), with each model given its proper prompt:

  | model (prompt) | dim | params | avg. precision |
  | --- | --- | --- | --- |
  | **bge-small-en-v1.5** (instruction) | 384 | 33M | **0.727** |
  | **thenlper/gte-small** (none) | 384 | 33M | 0.716 |
  | bge-base-en-v1.5 (instruction) | 768 | 109M | 0.710 |
  | all-MiniLM-L6-v2 (none) | 384 | 23M | 0.709 |
  | e5-small-v2 ("query:") | 384 | 33M | 0.688 |
  | all-mpnet-base-v2 (none) | 768 | 109M | 0.677 |
  | bge-small-en-v1.5 (raw, no prompt) | 384 | 33M | 0.670 |
  | e5-small-v2 (raw) | 384 | 33M | 0.643 |

  Lessons: (1) **the small gold set lied** — on 70 stories / 10 pairs, `all-mpnet-base-v2`
  looked best (0.895); on the 6× corpus it falls to 6th, and the small-sample caveat we flagged
  is exactly what bit. (2) **The instruction prompt is worth ~0.05 AP** for bge/e5 — omitting it
  (as the first run did) understated them; `query_prefix` now supports it. (3) **Bigger ≠
  better** (bge-base < bge-small, mpnet mid-pack). **`thenlper/gte-small` is the recommended
  default** — within noise of the top score, needs no prompt, MiniLM-sized; `bge-small-en-v1.5`
  with its instruction edges it if you configure the prompt. Still caveats: one corpus, one
  day's window, and every model leaves ~30–37% of docs as cluster noise — that residue is a
  data-volume lever, not an embedder one. Re-run as the corpus grows.
- **Thin daily samples yield thin overlap.** A single day across a handful of feeds rarely
  has many stories covered by multiple outlets in one diet and none in the other, so
  blindspot lists can be short and some entries rest on 2 stories. Treat them as candidates,
  not verdicts; they strengthen as coverage accumulates over time and across more sources.
  The **GDELT backfill** (`python -m ingestion backfill`) is the fix — it pulls weeks of
  per-outlet history so clusters rest on real volume. Caveats specific to GDELT:
  - **Titles only.** GDELT returns article metadata, not bodies, so backfilled documents are
    title-based by default. That is what the clustering needs, but their moral-foundation
    scores (computed on the title) are weaker than body-scored feed documents — mixing the
    two in one aggregate slightly muddies the MFT profile. Use `--extract` for full bodies
    when the MFT numbers matter, or keep backfill for the blindspot engine and feed ingestion
    for scoring.
  - **~3-month window, imperfect coverage.** GDELT indexes the trailing ~3 months and does
    not include every outlet's every article; some domains return little.
  - **Rate limits.** The free endpoint throttles hard (≈1 request/5s) and throttles shared
    IPs harder — a full-registry backfill is slow and can return partial results on a hot IP.
    The client backs off and retries; if a source comes back empty, re-run later.
- **A blindspot is a coverage signal, not a moral judgment.** "One diet covers X, the other
  doesn't" is descriptive. The tool reports both directions with equal prominence
  (including the author's own blindspots) and never editorializes about which absence is
  worse.

## Theory caveats

- **The liberal/conservative foundation asymmetry** (Graham, Haidt & Nosek 2009) — that
  liberals draw mainly on care and fairness while conservatives draw more evenly on all
  foundations — is real but **contested in magnitude**. Competing work argues the two
  groups rely on very similar sets of foundations. Parallax presents this asymmetry as a
  **hypothesis the tool tests**, not an axiom it assumes.
- **Sanctity/purity is especially central to evangelical discourse** and is handled as a
  distinct signal, not collapsed into a generic "moralizing" measure.
- **Fairness is split** into Equality vs Proportionality (Atari & Haidt 2023), because the
  two poles map differently across the political spectrum.

## Representativeness limits

- The "other" diet is a **model**, not a measurement of any individual's consumption. It
  is versioned in `config/sources.yaml` and its conclusions should be sensitivity-tested
  against source weighting.
- **Outlet bias ≠ audience consumption.** The model targets what is plausibly *consumed*,
  not merely what is *published*, but this mapping is approximate.
- Validate against external benchmarks (Pew on evangelical media use; Media Cloud
  attention data) rather than treating the source list as self-evidently representative.

## Guarding against confirmation bias

- Pre-register what you expect to find each period, then check yourself against it.
- The identical pipeline runs on the author's own diet; the author's blindspots are
  displayed with equal prominence.
- Periodically have the LLM critique the framing choices and flag where the author's own
  diet is the outlier.

## Expected weak spots (to be quantified by validation)

Per-foundation agreement with human coders is expected to be weakest on the **binding
foundations** (loyalty, authority, sanctity). The validation gold set (`validation/`)
will report Krippendorff's alpha / Cohen's kappa and per-foundation AUC/F1 as they are
computed; this section will be updated with actual numbers.
