"""Cluster: embed, cluster, and detect blindspots across both diets.

sentence-transformers -> UMAP -> HDBSCAN. Each cluster is a "story"; coverage
share is computed per diet.

Blindspot = a cluster with heavy coverage in one diet and near-zero in the
other. Emit both directions: what they cover that I don't, and what I cover
that they don't. Tune min_cluster_size and UMAP n_neighbors/n_components —
defaults over-assign short news items to noise.

A cluster is the unit the asymmetry is *measured* on. It is not the unit anyone
can read: clusters are small by design, a day makes dozens, and their c-TF-IDF
labels name whatever terms happened to be distinctive. ``themes.py`` groups
them by subject and direction under a name written for a person, and
``titles.py`` cleans the headlines both surfaces show.
"""

# What the daily run and the CLI cluster unless told otherwise: about how long a
# story stays news. Settings: `cluster.window_days`; 0 clusters the whole store.
# Here rather than in blindspot.py so the daily runner can read it without
# importing numpy at load time.
DEFAULT_WINDOW_DAYS = 7


def window_days(settings: dict) -> int:
    """`cluster.window_days`; a key with no value means the default, 0 means all."""
    value = (settings.get("cluster", {}) or {}).get("window_days")
    return DEFAULT_WINDOW_DAYS if value is None else int(value)


# Whether headline-only documents may form clusters. They cannot usefully: a
# cluster built from titles groups stories by shared keywords rather than by
# subject, which the story review reported in its own titles ("a sports-tagged
# cluster mixing WNBA, voting, and revenue"). They keep their quarter
# weight in every corpus-derived number and count for nothing in the
# clustering-derived ones (blindspot counts, per-thread diet counts,
# attention shares). LIMITATIONS.md names them, because the difference is
# not visible on the surfaces that print them.
#
# A boolean, not a word-count floor. The floor it replaces excluded 1,297
# documents that had real text, because "short" and "headline" are different
# questions and only `ingest_kind` answers the second one.
DEFAULT_CLUSTER_STUBS = False


def cluster_stubs(settings: dict) -> bool:
    """`cluster.cluster_stubs`; a key with no value means the default."""
    value = (settings.get("cluster", {}) or {}).get("cluster_stubs")
    return DEFAULT_CLUSTER_STUBS if value is None else bool(value)
