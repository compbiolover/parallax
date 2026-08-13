"""The contract between `build_catalog` and the page that authors a persona.

``dashboard/builder.html`` is one file of vanilla JS with no build step and no
test runner, in the same shape as ``index.html`` and for the same reasons. The
failure it invites is the same one: a catalog key renamed on the Python side and
a panel that quietly renders empty.

It invites two more of its own, and those are what most of this file is about.

The page **emits a file another program has to load**. If its idea of a valid
overlay drifts from ``ingestion/config.py``'s, the download is a file that fails
at ``load_registry`` with a message about the registry — pointing at the loader
rather than at the page that wrote it. So the round trip is tested for real: run
the page's own emitter, hand the result to the real loader, and check the weights
come back as the page predicted.

And the page **promises that nothing leaves the browser**. That promise is the
whole reason it is a static file rather than a form, so it is asserted rather
than trusted.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import tempfile

import pytest

from compare.reference import ReferencePair
from dashboard.export import build_catalog
from ingestion.config import load_registry, load_settings

from .registries import registry

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "dashboard" / "builder.html"

# Keys the page reads off the catalog root as `C.<name>`.
_ROOT_READ = re.compile(r"\bC\.([a-z_][a-z0-9_]*)\b")


def _catalog() -> dict:
    reg = registry(self={"s_mine": 1.0}, modeled_ce={"s_theirs": 1.0})
    return build_catalog(reg, ReferencePair(mine="self", theirs="modeled_ce"))


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


# -- the same contract test index.html has -----------------------------------


def test_every_catalog_key_the_page_reads_is_one_the_exporter_emits():
    page = _page()
    catalog = _catalog()
    read = set(_ROOT_READ.findall(page))
    missing = read - set(catalog)
    assert not missing, (
        f"builder.html reads catalog keys the exporter does not emit: {sorted(missing)}"
    )


def test_the_page_reads_something_at_all():
    """Guards the regex: a pattern that matched nothing would pass silently."""
    assert len(set(_ROOT_READ.findall(_page()))) >= 3


# -- the promise that nothing leaves the browser -----------------------------


def test_the_page_cannot_send_anything_anywhere():
    """The header says "nothing leaves this page". This is that claim, checked.

    A form post or a stray analytics tag would make the page a place a real
    person's real media diet gets uploaded — the one thing CLAUDE.md §0 exists
    to prevent, and the reason this is a downloadable file rather than a save
    button.
    """
    page = _page()
    for forbidden in ("fetch(", "XMLHttpRequest", "navigator.sendBeacon", "<form"):
        assert forbidden not in page, f"builder.html contains {forbidden!r}"


def test_the_page_loads_nothing_from_the_network():
    """No CDN either — unlike index.html, which pulls D3.

    The builder draws no charts, so it needs no library, and every external
    reference is another party who learns that this page was opened.
    """
    page = _page()
    assert "//cdn." not in page
    assert not re.search(r'src\s*=\s*"https?://', page)
    # The one script it does load is the generated catalog, relative.
    assert 'src="./public/data/catalog.js"' in page


def test_a_missing_catalog_is_a_message_rather_than_a_blank_page():
    """A 404 on a script tag does not throw, so absence needs handling."""
    page = _page()
    assert 'onerror="window.__noCatalog=true"' in page
    assert "python -m dashboard.export" in page


# -- the rules the loader enforces, mirrored on the page ---------------------


def test_the_page_warns_about_replacing_a_shipped_persona():
    """`_merge_overlay` replaces wholesale by id.

    Reusing a shipped id does not add a persona — it silently replaces that one,
    and the comparison the author meant to make stops existing with nothing to
    point at.
    """
    page = _page()
    assert "C.personas.some(p => p.id === d.id)" in page


def test_the_page_auto_weights_a_stratum_when_a_source_needs_one():
    """The commonest first action — pick a source, download — must not produce a
    file the loader refuses.

    `_build_personas` raises when a persona lists a source whose stratum carries
    no weight, and membership is what it tests: a weight of 0.0 passes.
    """
    page = _page()
    assert "if (!(src.stratum in d.strata)) d.strata[src.stratum] = 0.25" in page


def test_the_page_shows_realized_share_not_just_the_slider():
    """The trap the registry's own schema comment warns about.

    A stratum weight is a per-source multiplier, not that stratum's share, so a
    page showing only the slider teaches the reader something false about the
    persona they are building.
    """
    page = _page()
    assert "function realizedShares" in page
    assert "share of the diet" in page


def test_the_page_does_not_invent_a_divergence_score():
    """The catalog carries no scores, so JSD is not computable here.

    Showing one would be the worst thing this page could do: a number with the
    authority of the dashboard's headline metric and nothing behind it.
    """
    page = _page()
    # No divergence number anywhere — not even a variable named for one. The
    # `or` form this was first written with short-circuits, so it would have
    # passed while checking nothing.
    assert not re.search(r"\bjsd\b", page, re.IGNORECASE)
    # And the overlap panel says out loud what it is not.
    assert "Source overlap only" in page
    assert "cannot tell you how differently" in page


# -- the emitted file cannot carry a feed URL --------------------------------


def test_the_emitted_yaml_can_never_contain_a_resolved_feed_url():
    """The mirror of the catalog's own guarantee.

    `build_catalog` emits `has_url` and the *name* of an env var, never a
    resolved URL, because a subscriber feed's URL is the credential. The page
    cannot leak what it was never given — but that is worth pinning, since a
    future edit that adds `url` to the catalog would break it silently.
    """
    catalog = _catalog()
    assert not any("url" in s and s.get("url") for s in catalog["sources"])
    # And the emitter writes only ids and weights out of the source records.
    page = _page()
    emitter = page[page.index("function toYaml") : page.index("function sourceById")]
    assert ".url" not in emitter


# -- the round trip, which is the one that matters ---------------------------

HARNESS = r"""
import fs from "node:fs";
const html = fs.readFileSync(process.argv[2], "utf8");
const m = html.match(/<script>\n([\s\S]*?)\n<\/script>/);
if (!m) { console.error("no inline script"); process.exit(1); }
const catalog = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const draft = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const stub = `
globalThis.window = { PARALLAX_CATALOG: ${JSON.stringify(catalog)} };
globalThis.localStorage = { getItem: () => null, setItem: () => {} };
globalThis.document = {
  getElementById: () => ({ innerHTML: "" }),
  createElement: () => ({ set textContent(v) { this._t = v; },
                          get innerHTML() { return String(this._t); } }),
};
`;
const body = m[1].replace(/^if \(!C \|\| window\.__noCatalog\)[\s\S]*?^}$/m, "");
await import("data:text/javascript," + encodeURIComponent(stub + body + `
globalThis.__api = { toYaml, realizedShares, effectiveWeights, problems };
`));
const api = globalThis.__api;
process.stdout.write(JSON.stringify({
  problems: api.problems(draft),
  yaml: api.problems(draft).length ? null : api.toYaml(draft),
  effective: api.effectiveWeights(draft),
  shares: api.realizedShares(draft),
}));
"""

needs_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="needs node to execute the page's own emitter",
)


def _run_page(draft: dict, catalog: dict) -> dict:
    """Run the page's real functions over a draft, outside a browser."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "harness.mjs").write_text(HARNESS, encoding="utf-8")
        (tmp / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        (tmp / "draft.json").write_text(json.dumps(draft), encoding="utf-8")
        out = subprocess.run(
            [
                "node",
                str(tmp / "harness.mjs"),
                str(PAGE),
                str(tmp / "catalog.json"),
                str(tmp / "draft.json"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return json.loads(out.stdout)


@needs_node
def test_what_the_page_emits_is_what_the_loader_accepts(tmp_path):
    """The whole point. Emitter and loader must agree, or the download is a trap.

    Uses the real shipped registry rather than a synthetic one, because the
    thing being checked is that a file authored against the *catalog people will
    actually see* loads against the *registry it was derived from*.
    """
    reg = load_registry(settings=load_settings())
    catalog = build_catalog(reg, ReferencePair(mine="self", theirs="modeled_ce"))

    news = next(s for s in catalog["sources"] if s["stratum"] == "national_dailies")
    pods = next(s for s in catalog["sources"] if s["stratum"] == "podcasts")
    draft = {
        "id": "mine_roundtrip",
        "label": "Round trip",
        "short_label": "RT",
        "family": "left",
        "description": "Built by the page, loaded by the loader.",
        "strata": {news["stratum"]: 0.4, pods["stratum"]: 0.1},
        "sources": {news["id"]: 1.0, pods["id"]: 0.8},
    }

    result = _run_page(draft, catalog)
    assert result["problems"] == [], result["problems"]

    overlay = tmp_path / "personas.local.yaml"
    overlay.write_text(result["yaml"], encoding="utf-8")

    loaded = load_registry(settings=load_settings(), overlay=str(overlay))
    assert "mine_roundtrip" in loaded.persona_ids()

    got = loaded.weights_for("mine_roundtrip")
    predicted = result["effective"]
    assert set(got) == set(predicted)
    for source_id, weight in got.items():
        assert weight == pytest.approx(predicted[source_id]), source_id


@needs_node
def test_the_page_refuses_a_draft_the_loader_would_refuse(tmp_path):
    """Every problem the page reports is a file the loader would reject.

    Checked in the direction that matters: take a draft the page calls invalid,
    force the file out anyway, and confirm the loader agrees it is bad. A page
    that refused *valid* drafts would merely be annoying; one that permitted
    invalid ones hands you a download that fails somewhere else.
    """
    reg = load_registry(settings=load_settings())
    catalog = build_catalog(reg, ReferencePair(mine="self", theirs="modeled_ce"))
    source = catalog["sources"][0]

    # A source selected, but its stratum given no weight — the rule users trip.
    draft = {
        "id": "mine_broken",
        "label": "",
        "short_label": "",
        "family": "",
        "description": "",
        "strata": {},
        "sources": {source["id"]: 1.0},
    }
    result = _run_page(draft, catalog)
    assert any("stratum" in p for p in result["problems"])
    assert result["yaml"] is None, "the page must not offer a file it knows is bad"

    # And the loader really does refuse that shape.
    overlay = tmp_path / "personas.local.yaml"
    overlay.write_text(
        "personas:\n"
        "  - id: mine_broken\n"
        "    strata: {}\n"
        f"    sources:\n      {source['id']}: 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no weight"):
        load_registry(settings=load_settings(), overlay=str(overlay))


@needs_node
def test_the_realized_share_the_page_draws_is_the_one_the_loader_computes(tmp_path):
    """The bars are the page's main claim; they have to be true.

    A stratum at 0.1 across many sources really can outweigh one at 0.4 across
    few, and that is the whole lesson — so the arithmetic behind the blue bar is
    checked against the registry rather than eyeballed.
    """
    reg = load_registry(settings=load_settings())
    catalog = build_catalog(reg, ReferencePair(mine="self", theirs="modeled_ce"))

    dailies = [s for s in catalog["sources"] if s["stratum"] == "national_dailies"][:1]
    pods = [s for s in catalog["sources"] if s["stratum"] == "podcasts"][:4]
    assert dailies and len(pods) >= 2, "fixture needs both strata populated"

    draft = {
        "id": "mine_shares",
        "label": "",
        "short_label": "",
        "family": "",
        "description": "",
        # The inversion, deliberately: the *lower* slider wins on realized share.
        "strata": {"national_dailies": 0.9, "podcasts": 0.3},
        "sources": {**{s["id"]: 1.0 for s in dailies}, **{s["id"]: 1.0 for s in pods}},
    }
    result = _run_page(draft, catalog)
    assert result["problems"] == [], result["problems"]

    overlay = tmp_path / "personas.local.yaml"
    overlay.write_text(result["yaml"], encoding="utf-8")
    loaded = load_registry(settings=load_settings(), overlay=str(overlay))

    weights = loaded.weights_for("mine_shares")
    total = sum(weights.values())
    by_stratum: dict[str, float] = {}
    for source_id, weight in weights.items():
        stratum = loaded.source(source_id).stratum_id
        by_stratum[stratum] = by_stratum.get(stratum, 0.0) + weight

    for stratum, weight in by_stratum.items():
        assert result["shares"][stratum] == pytest.approx(weight / total), stratum

    # And the lesson actually holds for this fixture: four sources at the lower
    # slider outweigh one at the higher.
    assert by_stratum["podcasts"] > by_stratum["national_dailies"]
