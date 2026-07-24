# validation/

The gold set and agreement metrics that keep every foundation number honest
(`CLAUDE.md` §5). This is where a scorer earns trust — or is shown not to have it.

## Run it

```bash
python -m validation                                  # dictionary, built-in seed lexicon
python -m validation --lexicon data/emfd_scoring.csv  # dictionary, real eMFD
python -m validation --scorer transformer             # Mformer (needs parallax[scoring])
```

Reports per-foundation **AUC / F1 / Cohen's kappa** against the hand-coded gold
labels, a macro-AUC, and the **§5 trigger**: if a *binding* foundation (loyalty,
authority, sanctity) scores below 0.7 AUC, the dictionary alone is not
trustworthy there and the transformer/Claude taggers are warranted.

## Files

- `gold/seed.json` — the hand-coded gold set: short texts with binary presence
  labels over the five classic foundations (virtue **or** vice counts as
  present), MFRC-style. A **starter** set (§5 targets 200–400 items across both
  diets and all foundations); expand it and add coders over time.
- `gold.py` — schema + loader. `metrics.py` — agreement metrics, incl.
  Krippendorff's alpha for inter-coder reliability (verified against his
  canonical example). `evaluate.py` — scores the gold set and applies the trigger.

## The result that justified Phase 3

On the seed gold set (42 items, single coder), the real eMFD dictionary vs the
Mformer transformer tagger:

| foundation | eMFD AUC | Mformer AUC |
| --- | --- | --- |
| care | 0.67 | 0.92 |
| fairness | 0.78 | 0.98 |
| loyalty *(binding)* | **0.63** | 0.94 |
| authority *(binding)* | 0.84 | 0.96 |
| sanctity *(binding)* | **0.63** | 0.96 |
| **macro-AUC** | **0.71** | **0.95** |

The dictionary fires the §5 trigger on **loyalty and sanctity** — exactly the
binding foundations the literature (and §5) predict it handles worst. The
transformer clears every binding foundation at ≥ 0.94. This is the empirical
case for the transformer tagger, and the harness will re-check it as the gold
set grows.

## The ensemble confidence signal (`--scorer ensemble`)

The dictionary + transformer ensemble flags an item **low-confidence** when the
taggers split. That flag is the §5 payoff — and on the seed gold set it is
strongly calibrated:

| bucket | predictions | label-accuracy |
| --- | --- | --- |
| taggers **agree** (high-confidence) | 125 | **0.86** |
| taggers **split** (low-confidence) | 85 | **0.27** |

A **+0.59** accuracy gap: when the two methods agree, trust the label; when they
disagree, don't. Note the ensemble's *point-estimate* AUC (macro 0.86) is below
the transformer alone (0.95) — the ensemble's job is the confidence flag, not a
better score.

## Guarding against confirmation bias

- Pre-register what you expect each period, then check yourself.
- The identical pipeline runs on the author's own diet; blindspots shown equally.
- Committed here: the gold-set labels, schema, and metrics — **not** raw article
  text or transcripts (gitignored). Gold texts are short hand-coded excerpts.
