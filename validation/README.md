# validation/

The gold set and agreement metrics that keep every foundation number honest
(see `CLAUDE.md` §5 and `LIMITATIONS.md`).

## Plan

1. **Hand-coded gold set.** A stratified random sample of **200–400 articles**
   across both diets and all foundations, coded using the Moral Foundations
   Reddit Corpus (MFRC) annotation guidelines (incl. "Thin Morality" and
   implicit/explicit flags).
2. **Agreement metrics.** Krippendorff's alpha / Cohen's kappa between the human
   coder and each automated method; per-foundation AUC and F1.
3. **Expectation.** Binding foundations (loyalty, authority, sanctity) are
   expected to be the weakest — quantify by how much.

## Trigger for Phase 3

If the gold set shows dictionary-only per-foundation AUC **below ~0.7** on the
binding foundations, add the transformer tagger and ensemble all three methods.

## Confirmation-bias guards

- **Pre-register** what you expect to find each period before scoring, then
  check yourself.
- Run the identical pipeline on your own diet; display your blindspots with
  equal prominence.
- Periodically have the LLM critique the framing choices and flag where *your*
  diet is the outlier.

Committed here: the gold-set schema, agreement notebooks, and results. **Not**
committed: raw article text or transcripts (gitignored) — store only IDs,
links, and hand-coded labels.
