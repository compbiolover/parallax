"""The daily brief: the whole dashboard, rendered into an email.

The D3 dashboard is a page you have to remember to open, on a host that would
have to exist somewhere. This renders the same content into a self-contained
email instead, so the brief arrives on its own and nothing about the author's
media diet is published anywhere.

That is a deliberate trade. A dated, running record of one person's news
consumption is not something to put at a public URL for the sake of
convenience, and the interactive charts are not worth the exposure. Everything
the dashboard shows is here as static markup.

Three constraints shape the rendering:

**Email cannot run JavaScript.** No D3, no external images, no web fonts.
Every chart is a table cell with an inline background colour — the one layout
primitive that renders the same in Apple Mail, Gmail and Outlook.

**Symmetry is a content requirement, not a styling one** (``CLAUDE.md`` §0).
Both directions of blindspot get identical markup and equal space, and the
author's own blindspots come first: that is the direction most likely to be
skipped, and on a phone "below the fold" means "unread".

**Uncertainty travels with the numbers.** The caveat that every foundation
score is a noisy estimate is part of the email, not a footnote trimmed to save
height.
"""

from .render import Digest, build_digest, render_html, render_text

__all__ = ["Digest", "build_digest", "render_html", "render_text"]
