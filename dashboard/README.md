# dashboard/

Static, dependency-light dashboard reading a generated data payload — no backend
needed for personal use.

## Files

- `index.html` — the page. Renders with D3 (loaded from CDN): a radar chart
  overlaying both diets' foundation compositions, the Jensen-Shannon divergence
  as a headline number, the attention divergence beside it, a diverging bar
  chart of per-foundation log-ratios, blindspot themes with their stories and
  the outlets that ran them, per-diet summary cards, the cross-diet executive
  summary, and a standing limitations banner. Theme-aware (light/dark).
- `export.py` — builds the data payload from the datastore and writes
  `public/data/latest.js` (a `window.PARALLAX_DATA = {…}` assignment, so the
  page also works when opened directly from disk). Aggregates only — no raw text.

## Generate and view

```bash
python3 -m ingestion run          # ingest + score
python3 -m summarize              # daily summaries
python3 -m dashboard.export       # -> dashboard/public/data/latest.js
cd dashboard && python3 -m http.server   # open http://localhost:8000
```

The generated `public/data/` payload is gitignored (regenerate it locally). Every
number carries a confidence caveat: the Phase 1 scorer is a demo lexicon (see
`LIMITATIONS.md`).

## Later

The Phase 1 page is intentionally a single self-contained file. A fuller
Astro/SvelteKit build with JSD time series and blindspot drill-downs arrives with
Phase 2.

## builder.html

`CLAUDE.md` Phase 7: a page for describing your own diet, which is the one thing
this repository deliberately has no file for. Open it from disk — it needs no
server — pick sources, set weights, download a `personas.local.yaml`, and drop it
at `config/personas.local.yaml`.

Its only input is `catalog.js`, which `python -m dashboard.export` writes and
which is corpus-free by construction: `build_catalog` is meaningful against an
empty datastore, and never carries a resolved feed URL — only `has_url` and the
*name* of the environment variable a subscriber URL comes from.

**Nothing leaves the browser.** No `fetch`, no form, no CDN — not even D3, which
the dashboard pulls and the builder does not need. State lives in `localStorage`
and the file is assembled in the page and handed to the download machinery. That
is not a nicety: a diet you author is your own consumption, the thing `CLAUDE.md`
§0 keeps out of this project, so the page is built to have nowhere to send it.
`tests/test_builder_page.py` asserts all of that rather than trusting it.

Two things it shows that are easy to get wrong:

**Realized share, next to the slider.** A stratum weight is a *per-source
multiplier*, not that stratum's share of the diet — `config/sources.yaml:104`
gives the worked example, where `self`'s nominal 0.10 audio strata are really
about 18% of its weight. The grey bar is what you set; the blue bar is what the
comparison actually uses, and the blue one is the one to watch.

**Overlap, and no divergence.** The page can honestly compute weighted-cosine
source overlap against the shipped personas, because that needs only the
registry. It cannot compute a JSD — the catalog carries no scores — so it does
not show one. A divergence number here would carry the authority of the
dashboard's headline metric with nothing behind it.
