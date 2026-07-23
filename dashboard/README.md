# dashboard/

Static, dependency-light dashboard reading a generated data payload — no backend
needed for personal use.

## Files

- `index.html` — the page. Renders with D3 (loaded from CDN): a radar chart
  overlaying both diets' foundation compositions, the Jensen-Shannon divergence
  as a headline number, a diverging bar chart of per-foundation log-ratios,
  per-diet summary cards, the cross-diet executive summary, and a standing
  limitations banner. Theme-aware (light/dark).
- `export.py` — builds the data payload from the datastore and writes
  `public/data/latest.js` (a `window.PARALLAX_DATA = {…}` assignment, so the
  page also works when opened directly from disk). Aggregates only — no raw text.

## Generate and view

```bash
python -m ingestion run          # ingest + score
python -m summarize              # daily summaries
python -m dashboard.export       # -> dashboard/public/data/latest.js
cd dashboard && python -m http.server   # open http://localhost:8000
```

The generated `public/data/` payload is gitignored (regenerate it locally). Every
number carries a confidence caveat: the Phase 1 scorer is a demo lexicon (see
`LIMITATIONS.md`).

## Later

The Phase 1 page is intentionally a single self-contained file. A fuller
Astro/SvelteKit build with JSD time series and blindspot drill-downs arrives with
Phase 2.
