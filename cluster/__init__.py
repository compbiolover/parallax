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
