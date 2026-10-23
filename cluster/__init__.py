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
