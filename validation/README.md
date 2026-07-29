# validation/

The gold set and agreement metrics that keep every foundation number honest
(`CLAUDE.md` §5). This is where a scorer earns trust — or is shown not to have it.

## Run it

```bash
python3 -m validation                                  # dictionary, built-in seed lexicon
python3 -m validation --lexicon data/emfd_scoring.csv  # dictionary, real eMFD
python3 -m validation --scorer transformer             # Mformer (needs `pip install -e ".[scoring]"`)
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

## The liberty register probe (`python3 -m validation.register_probe`)

The gold set cannot validate liberty yet — no public corpus labels it. But one
specific liberty failure can be tested without any hand-coding, and it is the
one that would do the most damage: the rubric scoring *state coercion* more
readily than *private or structural domination*. That asymmetry would show up on
the dashboard as one diet caring about freedom and the other not — a clean,
wrong finding about media rather than a property of the prompt.

```bash
make register-probe               # 10 pairs x 2 registers x 3 repeats = 60 calls
make register-probe REPEATS=5
python3 -m validation.register_probe --model claude-haiku-4-5 --yes
```

Ten templates across compulsion, surveillance, speech, medical disclosure,
livelihood, property, exit, conscience, assembly, and data. Each is a single
sentence with one `{actor}` slot, rendered twice — once with a state actor, once
with a private one. **The two sides are matched by construction, not by
judgement**: every other word is identical because it is literally the same
string. Writing two sentences by hand and calling them equivalent would put the
author's framing instincts inside the instrument, which is the exact error the
probe exists to detect. A test asserts the one-slot property, and asserts that
no template body contains the words *state*, *government*, *private*, or
*corporate* — which would hand the model the answer through the shared text.

It reports two separable things:

1. **Register classification** — do state-actor items get labelled `from_state`
   and private-actor items `from_private_power`? Categorical; `both` counts as
   correct, since a single-actor sentence can defensibly read either way. The
   failure under test is a crossed label, not a hedge.
2. **Presence magnitude** — does one register score systematically higher? This
   is the subtle one. Current models reject `temperature`, so run-to-run
   variance cannot be dialled down; the probe repeats each condition and prints
   the gap next to the **pooled within-cell spread**.

The magnitude reading is **descriptive, not inferential**. A few samples per
cell is not a significance test, and a gap below the noise floor is *absence of
evidence either way*, not evidence of fairness — the report says so in those
words rather than letting a reader infer a clean bill of health. With
`--repeats 1` it refuses to interpret the magnitude at all. Failed calls are
counted and dropped, never averaged in as zeros, so a register that errors more
often doesn't get dragged toward the middle.

This costs real API calls; the CLI prints the call count and asks before
spending (`--yes` skips it). Re-run it whenever the rubric text or the model
changes — both are the instrument, and both drift.

## Guarding against confirmation bias

- Pre-register what you expect each period, then check yourself.
- The identical pipeline runs on the author's own diet; blindspots shown equally.
- Committed here: the gold-set labels, schema, and metrics — **not** raw article
  text or transcripts (gitignored). Gold texts are short hand-coded excerpts.
