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
python3 -m digest --dry-run --text   # the plain-text part instead
python3 -m digest --dry-run --out /tmp/brief.html
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
python3 -m digest                       # or: make digest-send
```

All-or-nothing on purpose. A half-configured mailer is precisely the thing that
fails at 6am on a machine nobody is watching, so an incomplete setup declines to
send and prints exactly which variable is missing rather than raising in cron.

Two provider quirks account for most first-attempt failures: Gmail rejects your
account password outright once 2FA is on (you need an app password), and most
providers silently drop a message whose `From` doesn't match the authenticated
user — which is why `PARALLAX_DIGEST_FROM` defaults to `PARALLAX_SMTP_USER`.

### The password crosses this connection

So the TLS has to be *authenticated* TLS, not merely encrypted. `smtplib`'s
`starttls()` defaults to `ssl._create_stdlib_context()` when handed no context,
which is `check_hostname=False` and `verify_mode=CERT_NONE` — it will accept a
self-signed certificate from anyone who can get on the path, with no error to
tell you it happened. Every Python in range (3.11 through 3.14) still takes that
default, so this is not a version to grow out of. `send()` passes
`ssl.create_default_context()` explicitly, and a test pins the premise so the
explicit context stops being load-bearing only when the stdlib default changes.

Consequences worth knowing before you hit them:

- **Certificate verification failures refuse the send.** The password is not
  transmitted. That failure looks identical whether the server is misconfigured
  or the connection is being intercepted, so it is logged as its own message
  rather than folded into a generic "send failed" — fix the certificate rather
  than working around the check.
- **Port 465 is treated as implicit TLS** and wrapped at connect time; 587 and
  everything else upgrade via STARTTLS.
- **`PARALLAX_SMTP_STARTTLS=0` against a remote host is refused outright**, before
  the connection opens. Turning encryption off is a deliberate setting; sending a
  credential across a network in the clear is a different thing. It remains
  allowed for `localhost`, which is the local-relay case the option exists for.

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

`python3 -m daily --only digest` sends one on demand without re-running anything.

## What the rendering commits to

- **Symmetry is content, not styling.** Both directions of blindspot get
  identical markup, and `own_diet` puts *your* blindspots first — equal styling
  isn't equal prominence on a phone, where second means scrolled past. A test
  asserts the ordering, and another asserts the copy never editorialises about
  either diet.
- **Blindspots are read by theme, not by cluster.** A cluster is two to five
  stories and a day makes a couple of dozen of them, each labelled with whatever
  terms c-TF-IDF found distinctive ("kidney stone · bret · institutional").
  `cluster/themes.py` groups them by subject *and* direction, and the email
  renders each group as a card: a name a person would use, the counts, and three
  headlines. Three cards per direction, capped per direction rather than per
  section so a diet having a noisy day cannot push the other diet's blindspots
  off the page. Whatever the cap leaves out is named on the line beneath, since
  a theme you are not shown is otherwise indistinguishable from one that was
  never found.
- **Headlines are cleaned before they are shown.** GDELT stores them tokenized
  and outlet-stamped ("U . S . Senate Articles - Christianity Today");
  `cluster/titles.py` puts the punctuation back, drops the stamp, and recognizes
  feed index pages, which are not stories and which used to lend the outlet's
  name to whatever cluster they landed in.
- **Unscored is never zero.** A first-ever snapshot reports "no previous day to
  compare" rather than `+0.000`; a fairness split with no evidence says so
  rather than printing a ratio; liberty carries its coverage next to its mean.
- **Uncertainty ships with the numbers.** The caveat is in the email body, not
  trimmed to save height. Liberty is labelled as the least corroborated number
  in the brief, because it is. Where the transformer has run, each foundation
  carries its dictionary-vs-transformer range next to the value — printed
  rather than drawn, since whiskers need geometry email cannot do, but the
  number is the part that matters (`CLAUDE.md` §5).
- **The value shown is the ensemble point where one exists**, matching what the
  dashboard plots. Reading the raw dictionary profile here put a different
  number on each surface for the same foundation once the transformer had run.
- **The text part is not a stub.** It carries every section the HTML does —
  including the sparkline summary, the fairness split, and liberty's
  provenance line. It is what a screen reader gets, so "uncertainty travels
  with the numbers" has to hold there too. A test walks a shared list of
  sections and asserts each appears in both.
- **Colour means "which diet", nothing else.** One diet→colour map is built
  once and used by every panel. Keying the composition off list index while
  keying blindspots off the `own_diet` setting let the same diet come out blue
  in one panel and orange in another, because `modeled_ce` sorts before `self`.
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
