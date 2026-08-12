"""Dashboard: the static site, and the export step that feeds it.

Two artifacts, written by :mod:`dashboard.export` and read by ``index.html``:

``latest.js`` is the measurement payload — compositions, divergence, log-ratios,
document counts, confidence bands, blindspots, summaries. Aggregates only; no
article text reaches it, per ``CLAUDE.md`` §0.

``catalog.js`` is the source registry, corpus-free: it is meaningful against an
empty datastore, which is what a persona-authoring surface needs. It carries no
resolved feed URL — only ``has_url`` and the name of the environment variable a
subscriber URL comes from — because a catalog entry may hold a credential.

Both are ``.js`` assignments rather than ``.json`` so the page works over
``file://``, where fetching a sibling JSON file is blocked. That is the whole
reason there is no build step here: the dashboard is a single HTML file you can
open from disk, with no server and no toolchain.
"""
