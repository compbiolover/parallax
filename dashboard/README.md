# dashboard/

TypeScript + D3.js static site. Reads generated JSON (no backend needed for
personal use) written to `dashboard/public/data/` by the `compare` and
`summarize` stages.

Planned views:

- **Radar chart** of the two six-dimension foundation vectors.
- **Diverging bar chart** of per-foundation log-ratios (100 = parity).
- **JSD time series** — how the divergence moves with real events.
- **Blindspot lists** — "what they cover / what I cover" — with drill-down to
  cluster summaries.

Every foundation score is rendered with a **confidence band** (see
`LIMITATIONS.md`): ensemble disagreement widens the band. Summaries steelman
both diets and link to source; the dashboard never republishes article text.

Scaffolding (Astro or SvelteKit) is added in Phase 1. This directory is a stub
until then.
