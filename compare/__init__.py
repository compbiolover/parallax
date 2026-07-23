"""Compare: quantify the divergence between two diet-level foundation profiles.

Headline metric: Jensen-Shannon divergence (base 2, the squared distance from
``scipy.spatial.distance.jensenshannon``) — bounded [0, 1], symmetric, and
linearly decomposable into per-foundation contributions.

Interpretable companions:
  - per-foundation log-ratio ln(P_i / Q_i); index form 100 x (P_i / Q_i).
  - Aitchison distance via CLR transform (compositionally correct).
  - cosine distance and smoothed KL as sanity checks.

Profiles are length-normalized, aggregated per period, then normalized to a
composition summing to 1 before any distance is computed.
"""
