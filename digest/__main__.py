"""CLI: ``python -m digest`` renders the brief, and optionally mails it.

    python -m digest --dry-run              # write it to a file and open it
    python -m digest --dry-run --text       # the plain-text part instead
    python -m digest                        # render and send

``--dry-run`` is the default posture while you are still tuning the layout:
it needs no SMTP settings, costs nothing, and the file opens in any browser.
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

DEFAULT_PREVIEW = "data/digest-preview.html"


def main(argv: list[str] | None = None) -> int:
    from dashboard.export import build_payload
    from ingestion.config import load_settings
    from ingestion.datastore import Datastore

    from .render import build_digest
    from .send import NO_CONFIG, MailConfig, send

    parser = argparse.ArgumentParser(prog="digest", description="Parallax daily brief")
    parser.add_argument("--db", help="SQLite path (default from settings)")
    parser.add_argument("--settings", help="path to settings.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the rendered brief to a file instead of sending")
    parser.add_argument("--out", default=DEFAULT_PREVIEW,
                        help=f"where --dry-run writes (default {DEFAULT_PREVIEW})")
    parser.add_argument("--text", action="store_true",
                        help="--dry-run: write the plain-text part instead of the HTML")
    parser.add_argument("--open", action="store_true",
                        help="--dry-run: open the rendered file in a browser")
    parser.add_argument("--own-diet",
                        help="diet id that is yours — puts your own blindspots first")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    settings = load_settings(args.settings)
    db = args.db or (settings.get("datastore", {}) or {}).get("path", "data/parallax.sqlite")
    own = args.own_diet or (settings.get("digest", {}) or {}).get("own_diet")

    store = Datastore(db)
    try:
        digest = build_digest(build_payload(store), own_diet=own)
    finally:
        store.close()

    print(f"Subject: {digest.subject}")
    print(f"Preview: {digest.preheader}")

    if args.dry_run:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(digest.text if args.text else digest.html, encoding="utf-8")
        print(f"Wrote {out}")
        if args.open:
            webbrowser.open(out.resolve().as_uri())
        return 0

    if MailConfig.from_env() is None:
        # Not an error worth a traceback — it is the ordinary first-run state.
        print(f"\n{NO_CONFIG}\n\nRender it without sending: python -m digest --dry-run",
              file=sys.stderr)
        return 1
    return 0 if send(digest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
