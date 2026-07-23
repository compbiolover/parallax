"""Summarize: Claude map-reduce summarization of clusters and diets.

map   = per-article / per-cluster summary.
reduce = cluster -> diet -> cross-diet executive summary. Prefer a
         deterministic reduce over structured JSON to limit variance.

Every summary steelmans each side's framing and includes verbatim quotes with
links so it is auditable against the source. Claude also tags liberty/oppression
(dictionaries miss it) and adjudicates disagreements between taggers.
"""
