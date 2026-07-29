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
  into a label. When the transformer runs at ingestion, the dashboard draws a
  dictionary-vs-transformer band (whiskers) on each foundation of the radar — wider means
  the two methods disagree more, so trust that number less. The band shows *method
  disagreement*, not a statistical confidence interval.

## Liberty is scored by one model against one rubric

Liberty/oppression is supplied by Claude, because nothing else in the pipeline can supply
it — the eMFD and MFD cover five foundations and the Moral Foundations Reddit Corpus behind
Mformer does not label liberty. That makes it the least corroborated number here.

- **There is no second opinion.** Every other foundation gets at least two independent
  estimates (dictionary and transformer), and the disagreement between them is the
  confidence signal. Liberty has one estimate from one model, so it carries no band and no
  ensemble flag. Treat a liberty score as a single annotator's judgment, because that is
  what it is.
- **The rubric is the instrument, and it is hand-written.** Liberty is claimed in two
  registers — freedom from state coercion, and freedom from private or structural
  domination. The rubric names both explicitly and instructs that neither is more truly
  liberty, precisely because a one-sided rubric would produce a clean, wrong finding: one
  diet engaged with liberty, the other silent. That symmetry is asserted by a test, but a
  test only checks that the words are present in the prompt — not that the model obeys
  them. `python3 -m validation.register_probe` measures the obedience: ten templates, each
  rendered once with a state actor and once with a private one and otherwise identical
  word for word, scored several times per condition. It reports register-classification
  accuracy, the mean presence gap between the two registers, and the pooled within-cell
  spread to compare that gap against. Run it whenever the rubric or the model changes.
  **The reading is descriptive, not inferential** — a handful of samples per cell cannot
  support a significance test, and a gap smaller than the noise floor is absence of
  evidence rather than evidence of evenhandedness.
- **Not validated at all yet.** No public corpus labels liberty, so the gold set does not
  either. `python3 -m validation --scorer liberty` refuses to run until liberty labels are
  hand-coded, rather than reporting an AUC against all-zero labels. Until that coding
  happens, every liberty number on the dashboard is unvalidated in the strict sense — not
  "weakly validated", but never checked against a human coder.
- **Coverage is partial and uneven.** The tagger runs on feed-ingested documents only —
  GDELT backfill is title-only, and a title is too little to judge liberty framing from.
  It also runs only when an API key is set, so a corpus can contain long stretches with no
  liberty scores at all. Coverage is reported next to every mean for that reason.
- **It is deliberately kept out of the composition.** Liberty is not a sixth spoke on the
  radar and does not enter the Jensen-Shannon divergence. Mixing a partial-coverage
  foundation into a composition would change every other share as a side effect of
  coverage, and would move the headline number that the snapshot history has been recording
  since it began. The cost of that choice is that the dashboard's headline still describes
  five foundations while the project models six.
- **Production scores are not individually auditable.** The rubric requires a verbatim
  supporting quote, which is what forces the judgment to point at the text rather than at a
  vibe — but the quote is not persisted, because a persisted quote is persisted article
  text, which §0 forbids. Quotes and rationales are visible during validation, where the
  gold texts are on disk anyway. This is a real tension with `CLAUDE.md` §3(a)'s
  auditability goal, resolved in favour of the content-handling rule rather than reconciled.
- **Model changes are silent breaks.** The scorer name records the model, but scores from
  different models sit in the same column. Swapping the tier mid-corpus makes earlier and
  later documents incomparable, and nothing detects it.

## The equality/proportionality split is exploratory

MFQ-2 (Atari, Haidt, Graham et al. 2023) divides Fairness into **equality** (equal treatment
and equal outcomes) and **proportionality** (reward tracking merit or contribution). Parallax
reports that division, with heavier caveats than anything else on the dashboard.

- **No validated dictionary implements the split.** The eMFD, the MFD, and MFD 2.0 all carry
  a single `fairness` dimension, and the Moral Foundations Reddit Corpus behind Mformer does
  not split it either. So far the split exists at the questionnaire level. Parallax's
  partition runs off a **hand-built term list** (`scoring/fairness_split.py`) with no
  validation behind it — the same status as the demo seed lexicon, and the same warning.
- **It divides existing signal; it does not detect fairness.** The dictionary decides how
  much fairness a document contains. The split only decides how that mass divides. If the
  dictionary misses a fairness argument, the split cannot recover it.
- **The seed lexicon's own fairness vocabulary was equality-skewed.** Before this was
  corrected, the built-in seed contained no proportionality terms at all, so a merit-framed
  argument scored as containing *no fairness*. That is worth stating plainly because of what
  it implies: a word list can encode a political asymmetry while looking like a neutral
  instrument, and the failure is invisible unless you go looking. The correction added
  merit/desert terms, and `validation/lexicon_audit.py` now checks any lexicon for the
  same failure so it cannot come back unnoticed.
- **The eMFD does not repeat the seed's bug, but it does lean the same way.** Audited
  against the real eMFD (3,270 terms, `assignment: argmax`): merit vocabulary is present,
  and merit-framed text does produce fairness signal, so the categorical failure does not
  reproduce. What the audit does find is a structural tilt. Per occurrence, proportionality
  vocabulary yields about **0.64x** the fairness that equality vocabulary does, because
  half of the merit terms the eMFD contains are assigned to some other foundation and
  contribute nothing to fairness: *accountable* to authority, *contribution* to loyalty,
  *effort* to care. Equality terms mostly land on fairness (7 of 10) where merit terms
  split evenly (8 of 16). Median yield differs by roughly eightfold (0.275 against 0.033).
  This is a property of the dictionary, not of the diets it measures.
- **That lexicon-level tilt has not been shown to propagate to documents.** Four matched
  text pairs scored with the real eMFD gave proportionality/equality fairness ratios of
  0.83, 2.51, 0.13, and 0.70. The median (0.77) points the same way as the lexicon
  measurement, but the spread is far wider than the effect, and one pair reversed it
  entirely. The texts were also written by the same person running the test, which is a
  confound rather than a control. So: the tilt in the dictionary is real and measurable;
  its consequence for real coverage is not established either way, and should not be
  asserted from these numbers.
- **The audit's sample is small and depends on an unvalidated list.** It sees only where
  `scoring/fairness_split.py`'s hand-built terms intersect the lexicon: 10 equality and 16
  proportionality terms for the eMFD. It measures that intersection, not "the eMFD's
  fairness vocabulary" in any absolute sense. Re-run it whenever the term lists change:
  `python3 -m validation.lexicon_audit --lexicon data/emfd_scoring.csv`.
- **Unsplit is recorded as unsplit.** Documents without enough evidence get NULL, never an
  even split, and coverage is reported next to every share. A 70/30 split over 4% of
  documents is not a finding.
- **The prediction it tests is contested.** Theory expects equality on the left and
  proportionality on the right. Two reasons to hold that loosely: the liberal/conservative
  foundation asymmetry replicates poorly when measured in *language* specifically (one
  review reports ~30% replication success), and proportionality is reported to **bridge**
  the divide rather than mark it — both sides value merit and effort. There is also
  longitudinal evidence that ideology predicts foundation endorsement rather than the
  reverse, which complicates reading any of this as a window into moral psychology.
- **Historical note worth keeping in view.** Liberty was originally proposed partly because
  economic conservatives objected that the five-foundation model captured equality but not
  their proportional notion of fairness. The split and the liberty foundation address
  overlapping complaints, so adding both risks double-counting the same disagreement.

## What the divergence time series does and does not show

The dashboard plots recorded snapshots on two bases. Both inherit every caveat above —
they are the same noisy estimates, dated. Read movement, not decimals.

- **The all-time series is heavily damped, by construction.** It averages every document
  ever ingested, so it moves less the longer the project runs. A flat all-time line is
  weak evidence that nothing changed; it is mostly evidence that the corpus is large.
- **The trailing-window series is noisy, and on thin days it is only noise.** A week with
  few documents can swing it hard. Check the document count in the hover readout before
  reading anything into a spike.
- **No confidence interval is drawn on the series.** Day-to-day wobble of the same
  magnitude as the radar's method-disagreement bands should be treated as indistinguishable
  from zero. The series has no error bars because we do not have a defensible way to put
  them there yet.
- **Reconstructed points are not observations.** `--backfill` recomputes past days from
  publication dates in the current store, which includes articles fetched long after the
  fact. They are what the corpus now says about that date, not what a run that day would
  have produced. The chart shades them; do not read the shaded region as a record of
  what was actually measured at the time.
- **Snapshots are not re-derived when scoring changes.** A row records the numbers as
  scored on the day it was written. Swapping the lexicon or the transformer makes older
  rows incomparable with newer ones, and nothing currently detects that. After changing a
  scorer, treat the series as starting over — or rebuild it with
  `python3 -m compare.history --backfill N --overwrite`.
- **The corpus is not a stable panel.** Sources are added and feeds break. A move in
  either series can reflect a change in what was collected rather than a change in how
  either diet framed anything. The source registry is versioned so this is at least
  auditable after the fact.

## Implementation status (Phase 1)

- **The bundled lexicon is a demo, not an instrument.** The Phase 1 dictionary scorer
  ships a small hand-built seed lexicon (`scoring/seed_lexicon.py`) so the pipeline runs
  out of the box. It is a few dozen stems with unit weights — it must **not** be read as
  a validated measurement. Real scoring requires the full eMFD (continuous per-word
  probabilities over ~10k words), loaded via `scoring.lexicon.load_emfd_csv`. Until then,
  every foundation number the pipeline emits is illustrative only.
- **Dictionary-only coverage.** The current scorer covers the five classic foundations;
  liberty/oppression is unscored (`None`, never `0`) until the Claude tagger lands.
- **Ensemble confidence signal exists, is calibrated, and is now on the dashboard.** The
  dictionary + transformer ensemble (`scoring/ensemble.py`) flags an item low-confidence
  when the taggers split. On the seed gold set this signal is strongly meaningful:
  predictions where the taggers **agree** are 86% accurate; where they **split**, 27% (a
  +0.59 gap). The transformer now runs on every article at ingestion (`ingestion run`,
  toggle with `--transformer/--no-transformer`), and the exporter aggregates the
  disagreement into a per-diet, per-foundation **band** drawn as whiskers on the radar
  (`compare/confidence.py`). The band and both its compositions are computed over the same
  paired document set — only documents scored by *both* taggers — so it never conflates a
  method difference with the fact that GDELT backfill writes dictionary rows but no
  transformer rows. Two honest caveats remain: (1) the ensemble's *point estimate*
  AUC (macro 0.86) sits **below the transformer alone** (0.95) — its contribution is the
  confidence flag, not a better score, so the transformer remains the best single scorer;
  (2) the band shows dictionary-vs-transformer **method disagreement**, not a statistical
  confidence interval, and the two compositions are on the same simplex but built
  differently (dictionary argmax rates vs transformer presence probabilities), so read the
  band as "how much the methods argue here," not as error bars. Transformer scoring needs
  `pip install -e ".[scoring]"`; without it, ingestion degrades to dictionary-only and no band shows.
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
  `pip install -e ".[scoring]"`) is the more trustworthy source there.
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
  The **GDELT backfill** (`python3 -m ingestion backfill`, and included in `make daily`)
  is the fix — it pulls weeks of per-outlet history so clusters rest on real volume.
  Caveats specific to GDELT:
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

- **A daily snapshot is a batch job, not an interactive command.** `make daily` runs
  ingest → backfill → cluster → summarize → export. Two costs dominate and neither is a
  bug: transformer scoring runs five RoBERTa models over every ingested article (seconds
  per article on CPU — the price of confidence bands), and the GDELT throttle paces the
  backfill across the registry. Run it from cron rather than waiting on it. Steps are
  isolated, so a GDELT outage or a missing `ANTHROPIC_API_KEY` degrades that step only —
  the dashboard still refreshes from the data that landed, the report names what failed,
  and the exit code is non-zero. A partial snapshot is therefore normal and visible rather
  than silent: check the step report before reading a day's numbers as complete.
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
