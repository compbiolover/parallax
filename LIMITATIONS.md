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
- **The probe has now been run, and it found a tilt.** On `claude-sonnet-5` (two runs, 60
  and 140 calls, 2026-07-29): register *classification* is clean — 100% correct across 200
  samples, not one crossed label. Presence *magnitude* is not. State-actor sentences score
  **~0.06 higher** than word-for-word identical private-actor ones, 1.7× the noise floor,
  with 9 of 10 topics leaning the same way (sign test p ≈ 0.021). The gap is
  **concentrated, not uniform**: four topics (surveillance, speech, property, data) carry
  +0.115 while the other six sit within noise at +0.020, and that per-topic ordering
  reproduced in 38 of 40 pairwise comparisons across the two runs.

  **What it means for any cross-diet liberty comparison.** The modeled diet claims liberty
  largely in the state register (mandates, overreach, censorship, conscience); the author's
  diet leans on the private and structural register (corporate power, surveillance,
  employer control). A rubric that scores state framing higher therefore inflates the
  modeled diet's liberty engagement relative to the author's — a property of the prompt
  wearing the clothes of a finding, which is precisely what this probe exists to catch.
  Treat any liberty gap between the diets as containing roughly this much instrument
  before it contains anything about media. It also interacts with a threshold:
  `salient_share` counts documents above 0.5, so a systematic shift moves borderline items
  asymmetrically and the effect there can exceed 0.06.

  **The rubric was deliberately not retuned**, because editing it until the gap closes
  would overfit to the ten templates that measure it, and a flat correction would
  overcorrect the six domains already even. `validation/README.md` records the numbers and
  what a defensible revision would require.
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
- **Model changes are silent breaks, and nothing detects them.** The scorer name records
  the model, but scores from different models sit in the same column, so swapping the tier
  mid-corpus makes earlier and later documents incomparable. The dictionary has a warning
  for the equivalent hazard (see the next section); **liberty does not** — `liberty_scorer`
  is recorded in the store and read back for display, but never compared against the model
  a new run is about to use. Changing `scoring.taggers.liberty.model` on an existing corpus
  is currently undetectable from the inside.

## Changing an instrument mid-corpus blends two of them

This applies to every scorer, and it is a property of the design rather than a bug that
can be patched away.

Dictionary scores are all written under the single scorer key `dictionary`, with the
lexicon name recorded separately as metadata. Ingestion skips documents it already holds,
and **raw text is never persisted** (`CLAUDE.md` §0), so nothing can be re-scored after
the fact. Swapping the lexicon therefore appends a second instrument's numbers into the
same column: every aggregate blends the two, while the metadata reports only the newer
name. The dashboard caveat would stop saying DEMO while most of the corpus was still
demo-scored — wrong in the direction of confidence, which is the worst direction.

Two guards exist. Both cover the **dictionary lexicon only**, and neither is a fix:

- A configured `lexicon_path` that does not resolve now **warns** instead of silently
  falling back to the demo seed. `config/settings.example.yaml` ships pointing at
  `data/emfd_scoring.csv`, which is gitignored, so the default configuration lands in
  exactly that branch until you download the file.
- A run whose lexicon differs from the one already recorded in the store **warns** and
  says what it means.

The same hazard applies to the transformer model and the Claude liberty model, and
**neither is checked**. Both record their scorer name in the store, so the comparison is
available and simply not implemented. Until it is, changing either mid-corpus is silent.

**The only clean answer is a fresh datastore.** Nothing of value is lost — the store holds
derived metrics, not text — but the snapshot history restarts with it, so change
instruments deliberately rather than incidentally, and preferably not mid-period.

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

## Attention divergence measures the agenda, not the framing

`compare/agenda.py` reports a second Jensen-Shannon number: the divergence between the
two diets' distributions of attention across story clusters. It exists because the
headline foundation divergence is *small* on a real corpus — a few thousandths — and that
small number, printed alone, tells a reader "these diets are nearly identical" when they
can see that they are not. Both numbers are honest answers to different questions: the
foundation one asks what moral vocabulary each diet speaks, the agenda one asks what each
diet spent the day on. Its limits:

- **It inherits every clustering caveat above.** Its unit is the HDBSCAN cluster, so a
  loose clustering day inflates the story count and a coarse one merges two stories into
  one. The number moves with `min_cluster_size` and with the embedder. Compare it across
  days only when the clustering configuration has not changed.
- **Noise is excluded.** Documents HDBSCAN could not place (cluster −1) are dropped
  entirely. On the default hashing embedder that is 30–40% of the corpus, and nothing
  guarantees the dropped share is the same for both diets.
- **It measures the registry as much as the diets.** Two diets ingested from different
  numbers of feeds will diverge partly because they were sampled differently. Shares
  rather than counts remove the volume difference, not the selection difference.
- **A large number is the expected result, not a finding.** With a thin corpus most
  clusters are touched by one diet only, which drives the divergence toward 1 almost
  mechanically. It becomes informative as coverage accumulates and the shared-story count
  becomes non-trivial; below `THIN_ARTICLES` clustered articles it is flagged `thin` on
  every surface.
- **It cannot separate agenda from framing.** A story both diets covered contributes to
  overlap regardless of how differently they covered it. The metric that would isolate
  framing — foundation vectors computed *within* the clusters both diets touched — is not
  built. Until it is, "they moralize alike" and "they moralize the same events alike" are
  not the same claim, and only the first is supported.

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
- **The theme over a story is presentation, not measurement.** Asymmetry is computed per
  cluster; a theme is a grouping by subject so the result can be read. The grouping comes
  from a keyword taxonomy in `cluster/themes.py`, or from Claude given the headlines when a
  key is set — neither is validated, and neither changes a number. Two consequences worth
  holding onto: a mis-grouped story is a mis-filed card over correct arithmetic, and the
  taxonomy's vocabulary is a choice, so a subject it has no words for lands under "Other
  coverage" rather than under a name that fits. Which method named a theme travels with it
  (`method` in the payload) and is printed on the dashboard.
- **The theme is assigned per article, not per cluster.** Clusters are not pure — HDBSCAN
  groups by headline similarity, and a cluster that is mostly one subject can hold an
  article about another. Labelling the whole cluster propagated its plurality subject onto
  every headline inside it, which is what a reader sees as a story filed under the wrong
  theme. Each article now carries its own assignment and a cluster splits across the themes
  its articles actually hold. This makes a story's *theme* accurate at the cost of letting
  one cluster appear under two cards; the cluster remains the measured unit either way, so
  no count double-counts an article.
- **The outlets under a story are the ones ingested, not the ones that ran it.** A story
  lists the mastheads from the dominant diet's registry that carried an article in that
  cluster, with a link each. Absence from the list means the source registry did not
  surface it — not that the outlet ignored the story — and the list is capped for
  display. Only the dominant diet's articles are shown, because the sentence above them
  claims the other diet did not cover it.
- **What the email shows is a sample of what was found.** The brief caps the cards per
  direction and the headlines per card; the dashboard carries the full set. The cap is
  per direction rather than per section, so a diet with a noisy clustering day cannot
  crowd the other diet out — but the counts on a card describe the theme, not the three
  headlines under it.
- **Headline cleaning is a heuristic over titles alone.** `cluster/titles.py` detokenizes
  GDELT's spacing, strips a trailing outlet stamp, and drops feed index pages. It is
  deliberately conservative — a mangled real headline costs more than a surviving ugly one
  — but it can still take a title-cased second clause for a publisher's name, and it never
  removes the last headline in a cluster even when every candidate looks like boilerplate.
  Nothing here touches scoring, embedding, or clustering, all of which ran before it.

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

- The modeled diets are **models**, not measurements of any individual's consumption. They
  are versioned in `config/sources.yaml` and their conclusions should be sensitivity-tested
  against source weighting.
- **Outlet bias ≠ audience consumption.** The model targets what is plausibly *consumed*,
  not merely what is *published*, but this mapping is approximate.
- Validate against external benchmarks (Pew on evangelical media use; Media Cloud
  attention data) rather than treating the source list as self-evidently representative.

## A persona claims more than an outlet list does

Parallax compares **personas**: named weightings over a shared catalog of sources. Four
per side ship in `config/sources.yaml`. This is a stronger claim than the source list it
replaced, and the strength is easy to miss because it reads as a label rather than as an
assertion.

- **An outlet list says "these outlets published this." A persona says "someone plausibly
  consumes these, in roughly this proportion."** The second is a behavioural claim about a
  reader, and nothing here validates it. No audience panel, no diary study, no survey.
- **The weights are the least-evidenced and most load-bearing part of the whole tool.** No
  source gives the devotional-versus-cable split of a devout evangelical's week, or the
  policy-journal share of a wonk's. Every number on the dashboard is downstream of numbers
  that were reasoned about and written down, not measured.

  `make sensitivity` is the check, and it is cheap because weights resolve at aggregation:
  it moves each stratum weight ±50% in turn and re-aggregates score rows already in the
  store, with no re-ingestion and nothing written back. Read the **sign table** rather than
  the divergence range. A headline that moves from 0.033 to 0.041 is the same finding; a
  per-foundation log-ratio that changes sign is a different one, because the sign *is* the
  claim — which diet over-indexes on care. A flip means that claim was an artifact of a
  weighting nobody measured.
- **Persona count is not persona coverage.** Four a side does not cover the space of
  American media diets. It means four points were written down.
- **A named archetype invites over-reading.** "Devotionally-heavy reader" is a weighting of
  feeds. It is not a person, not a demographic, and not a claim about anyone's faith,
  intelligence or sincerity. The summarizer is instructed at the system-prompt level not to
  write as though it were, but that is an instruction to a model and not a guarantee.
- **`ce_devout` will show high sanctity, and that is mostly the genre.** Devotional and
  teaching content is saturated with purity, authority and loyalty vocabulary regardless of
  anyone's politics. It is the most confident-looking number this tool will produce and
  among the least meaningful — the same class of error as the fairness-lexicon asymmetry
  documented above, arrived at from the other direction.
- **A devotional-heavy persona will also show few blindspots**, because devotionals carry
  almost no news agenda and therefore land in few story clusters. That is a fact about
  measurement, not about what that diet misses.
- **Personas over a shared catalog are correlated by construction.** Two that read mostly
  the same outlets have a small divergence for a mechanical reason. The dashboard prints a
  source-overlap matrix beside the divergence matrix for exactly this: low divergence with
  high overlap is an artifact, low divergence with low overlap is a finding. The shipped
  activist and progressive personas overlap 0.83.
- **Blindspot counting is unweighted.** A source weighted 0.05 counts the same as one at
  1.0 for "did this reach them at all", because the question is binary. A fractional
  "partly saw it" would not be interpretable, but the consequence is that a persona's
  blindspot list is insensitive to how much of its attention a source actually holds.

## The reference pair is a choice, and the registry is retroactive

- **Every headline number is about two personas**, named in `compare.reference_pair`. The
  JSD, the log-ratios, the agenda divergence, the blindspots, the email and the dated
  snapshot series are all about that pair. Choosing a different pair changes all of them
  while the column names stay the same, so the pair travels with every recorded snapshot
  and the run warns when it changes.
- **Per-foundation log-ratios are oriented `mine` first**, so positive means the reader's
  own diet over-indexes, as §3(5) of `CLAUDE.md` specifies. Rows recorded before the pair
  was named carry the opposite sign: the previous code took the first two ids in sorted
  order, and `modeled_ce` sorts before `self`. Nothing charts historical log-ratios, so the
  practical discontinuity is zero, but the stored values are not comparable across that
  change.
- **Weights now resolve at aggregation rather than being stored per document**, which makes
  the registry retroactive: re-weighting a persona changes what the entire corpus says
  about it, including for dates already past. Recorded snapshots are immutable and
  unaffected; the live all-time basis is not. That is the price of being able to
  sensitivity-test weighting at all, and it is the right trade, but a number that moved
  because a weight was edited looks exactly like a number that moved because the news did.
- **Documents from a source dropped from the catalog become invisible to every persona**
  rather than staying counted as they used to. The run reports the orphaned counts rather
  than letting the corpus quietly shrink.

## Deduplication, coverage, and what is still counted once

Near-duplicate detection is global: the index is seeded from every stored signature with no
persona filter. When the same wire story runs in outlets on both sides, one copy is
canonical and the rest are flagged `is_duplicate = 1`. Only canonicals are embedded, so a
wire story that ran in six outlets is clustered once.

**For coverage this used to erase five outlets, and no longer does.** Blindspot detection
and attention shares now read every source that carried a story — the canonical document's
own, plus the outlets whose copies were collapsed into it (`duplicate_of` recorded them all
along; nothing read it). Before that fix, a story both sides genuinely ran was credited to
whichever outlet was fetched first and recorded as never covered by any other, which
manufactured blindspots out of syndication. Collapsed copies are also listed in a blindspot
card's outlet list, since that list is the part a reader can check.

**Two things remain true and are worth knowing:**

- **A collapsed copy still does not contribute to any foundation profile.** Scoring counts
  each story once, globally. That is right within a persona — nobody's moral vocabulary
  should be counted six times for one wire story — but across personas it means the *second*
  diet to be fetched loses that story's vocabulary from its composition entirely. Dedup
  ought to be per-persona for scoring and is currently global. The effect is small (one
  document among hundreds, and compositions converge) and fixing it would move the headline
  divergence and break comparability with the recorded series, so it is written down here
  rather than changed quietly.
- **Coverage counting is binary and unweighted**, so a source weighted 0.05 counts the same
  as one at 1.0 for "did this reach them at all". A fractional "partly saw it" would not be
  interpretable, but a persona's blindspot list is therefore insensitive to how much of its
  attention a source actually holds.

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
