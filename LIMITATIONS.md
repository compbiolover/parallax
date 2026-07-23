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
- **Single tagger, no ensemble yet.** The ensemble-disagreement confidence signal
  described below is a Phase 3 deliverable; Phase 1 has no confidence band.
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
