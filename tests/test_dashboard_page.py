"""The contract between the exporter and the hand-written page.

``dashboard/index.html`` is one file of vanilla JS with no build step and no test
runner, so nothing catches a payload key renamed on the Python side: the page just
renders a panel short, or blank, and looks like it is working. These tests read the
page as text and check that every top-level key it reaches for is one the exporter
actually emits.
"""

from __future__ import annotations

import pathlib
import re

from compare.reference import ReferencePair
from dashboard.export import build_payload
from ingestion.datastore import Datastore

from .registries import registry

PAGE = pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "index.html"

# Keys the page reads off the payload root as `d.<name>` / `d.<name>?.` etc.
_ROOT_READ = re.compile(r"\bd\.([a-z_][a-z0-9_]*)\b")

# Reads that are not payload keys: locals and DOM objects also called `d`, plus
# d3's conventional datum argument.
_NOT_PAYLOAD = {
    "diets",
    "forEach",
    "map",
    "filter",
    "length",
    "id",
    "then",
    "push",
    "toFixed",
    "value",
    "key",
    "title",
    "label",
    "date",
    "jsd",
    "profile",
}


def _payload() -> dict:
    store = Datastore(":memory:")
    for persona, care in (("self", 0.3), ("modeled_ce", 0.1)):
        doc = f"{persona}-doc"
        store.upsert_document(
            doc_id=doc,
            source_id=f"src_{persona}",
            stratum_id=None,
            url=None,
            title="t",
            published_utc=None,
            fetched_utc="2026-08-10T00:00:00+00:00",
            word_count=90,
            minhash=None,
        )
        store.upsert_scores(
            document_id=doc,
            scorer="dictionary",
            foundations={
                "care": care,
                "fairness": 0.1,
                "loyalty": 0.2,
                "authority": 0.1,
                "sanctity": 0.1,
            },
            sentiment=0.0,
            moral_word_ratio=0.2,
            matched_words=18,
        )
    reg = registry(self={"src_self": 1.0}, modeled_ce={"src_modeled_ce": 1.0})
    payload = build_payload(store, reg, ReferencePair("self", "modeled_ce"))
    store.close()
    return payload


def test_every_payload_key_the_page_reads_is_one_the_exporter_emits():
    """A key renamed on the Python side leaves the page rendering a panel blank,
    which reads as "no data today" rather than as a bug."""
    page = PAGE.read_text(encoding="utf-8")
    payload = _payload()
    read = {m for m in _ROOT_READ.findall(page)} - _NOT_PAYLOAD
    missing = sorted(k for k in read if k not in payload)
    assert not missing, f"page reads payload keys the exporter does not emit: {missing}"


def test_the_page_reads_the_keys_the_persona_work_added():
    """Guards the other direction: the picker, the matrix, the overlap grid and the
    persona panel are only useful if the page is actually wired to them."""
    page = PAGE.read_text(encoding="utf-8")
    for key in ("d.reference", "d.matrix", "d.overlap", "d.registry_version"):
        assert key in page, key


def test_the_page_colours_by_role_rather_than_by_list_position():
    """The bug digest/render.py documents: colour assigned by index in one panel
    and by a setting in another put the same diet in two colours."""
    page = PAGE.read_text(encoding="utf-8")
    assert "function colourMap" in page
    # The old index-into-a-list-of-css-vars form must be gone.
    assert '["--self", "--other", "--muted"]' not in page


def test_the_page_labels_panels_it_cannot_recompute_for_a_new_pair():
    """Agenda, blindspots, fairness, liberty and the summaries come from the
    pipeline for one pair. Silently redrawing them under a different pair's
    heading would attribute one comparison's numbers to another."""
    page = PAGE.read_text(encoding="utf-8")
    assert "function pinnedNote" in page
    # Every panel that cannot be recomputed client-side has to call it.
    assert page.count("pinnedNote(p, d, onRef)") >= 4
