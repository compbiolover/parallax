"""Scoring: score documents on the six moral foundations.

Three complementary, ensembled taggers:
  1. Dictionary baseline (eMFDscore) — five classic foundations, no liberty.
  2. Transformer (Mformer/MoralBERT) — in-domain accuracy (Phase 3).
  3. Claude with a structured rubric — owns liberty/oppression; emits a
     rationale + supporting quote for auditability.

ALWAYS length-normalize before aggregation (raw dictionary counts correlate
with document length up to r~0.98). Ensemble disagreement is the confidence
signal: divergence flags an item low-confidence rather than forcing a label.
"""
