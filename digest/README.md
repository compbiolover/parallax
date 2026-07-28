# digest/

The dashboard, rendered into an email that arrives on its own.

## Why an email and not a hosted page

A dashboard you have to remember to open is one you check twice and then
forget. The obvious fix is to host the static page and send yourself a link —
but that puts a dated, running record of one person's news consumption at a URL,
permanently, for the sake of convenience. Everything the D3 page shows fits in
an email instead, so nothing is published anywhere and the brief still turns up
every morning.

The trade is real and worth naming: no interactivity, no drill-down, no hover.
The radar becomes two aligned bar lists, the JSD time series becomes a column
chart of fixed-height divs, and the blindspot lists lose their expansion. In
exchange the whole thing is a single self-contained message with no host, no
account, and no link to anything.

## Try it

```bash
make digest                     # render to data/digest-preview.html and open it
python -m digest --dry-run --text   # the plain-text part instead
python -m digest --dry-run --out /tmp/brief.html
```

`--dry-run` needs no SMTP settings and costs nothing, which is the right way to
tune the layout. `data/` is gitignored, so previews never end up committed.

## Sending it

Four environment variables, all required (see `.env.example`):

```bash
export PARALLAX_SMTP_HOST=smtp.gmail.com
export PARALLAX_SMTP_USER=you@gmail.com
export PARALLAX_SMTP_PASSWORD=...      # Gmail: an app password, not your login
export PARALLAX_DIGEST_TO=you@gmail.com
python -m digest                       # or: make digest-send
```

All-or-nothing on purpose. A half-configured mailer is precisely the thing that
fails at 6am on a machine nobody is watching, so an incomplete setup declines to
send and prints exactly which variable is missing rather than raising in cron.

Two provider quirks account for most first-attempt failures: Gmail rejects your
account password outright once 2FA is on (you need an app password), and most
providers silently drop a message whose `From` doesn't match the authenticated
user — which is why `PARALLAX_DIGEST_FROM` defaults to `PARALLAX_SMTP_USER`.

## In the daily run

```yaml
digest:
  enabled: true
  own_diet: self
```

`digest` is the only step of the seven that is **off by default**. Every other
step works with no credentials; this one can't, and a step that fails every
morning until configured teaches you to ignore the report that exists to tell
you when something actually broke. Once enabled it runs last, after `export`, so
the email describes the same payload the dashboard does.

`python -m daily --only digest` sends one on demand without re-running anything.

## What the rendering commits to

- **Symmetry is content, not styling.** Both directions of blindspot get
  identical markup, and `own_diet` puts *your* blindspots first — equal styling
  isn't equal prominence on a phone, where second means scrolled past. A test
  asserts the ordering, and another asserts the copy never editorialises about
  either diet.
- **Unscored is never zero.** A first-ever snapshot reports "no previous day to
  compare" rather than `+0.000`; a fairness split with no evidence says so
  rather than printing a ratio; liberty carries its coverage next to its mean.
- **Uncertainty ships with the numbers.** The caveat is in the email body, not
  trimmed to save height. Liberty is labelled as the least corroborated number
  in the brief, because it is.
- **The subject line has to be true alone.** It carries the divergence and its
  movement, since on a phone that line is frequently the entire reading.

## Email constraints worth knowing before editing

No JavaScript, no external images, no web fonts, no linked stylesheets — mail
clients strip or block all of it, and a remote image reads as a tracking pixel.
Every chart here is a table cell with an inline background colour, which is the
one layout primitive that renders the same in Apple Mail, Gmail and Outlook.

Widths are percentages inside `table-layout: fixed`, not pixels. The narrowest
phone still in common use is 320 CSS px, and a bar half-width wide enough to
read on a desktop overflows there. Verified at 320, 390, and 600.
