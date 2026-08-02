# Parallax

> A moral-foundations mirror for two media diets.

**Read this first.** Parallax compares the author's own media diet against a
*representative, documented model* of a conservative-evangelical media diet, using
Moral Foundations Theory (MFT) as the analytical lens. It ingests news, podcasts, and
video from two modeled information environments, scores them on moral foundations, and
reports what each one covers that the other does not.

**It is not a system that tracks, surveils, or profiles any specific individual.** The
"other" diet is a versioned model of outlets and programs
([`config/sources.yaml`](config/sources.yaml)), not any real person's consumption. No
private family communications are ever ingested.

Four commitments shape every part of it:

| Commitment | What it requires |
| --- | --- |
| **Charitable understanding** | Summaries steelman each side. The binding foundations (loyalty, authority, sanctity) are sincere moral commitments in MFT's framework, not deficits. Parallax does not mock, pathologize, or "dunk on" either diet. |
| **Symmetry** | The identical pipeline runs on both diets. The author's own blindspots and foundation skew are surfaced with equal prominence. |
| **Content handling** | Summarize and link, never republish. Derived metrics persist; raw article text is a transient processing artifact. `robots.txt`, rate limits, and each source's terms are honored. |
| **Uncertainty is first-class** | Every foundation number is an estimate with a confidence band, never ground truth. See [`LIMITATIONS.md`](LIMITATIONS.md) — it is not an appendix, it is the reading instructions. |

**Foundations modeled:** care/harm, fairness/cheating, loyalty/betrayal,
authority/subversion, sanctity/degradation, and liberty/oppression — with fairness further
split into Equality and Proportionality (MFQ-2, Atari & Haidt 2023). The last two are the
interesting ones and the least certain; both have their own section below.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip                 # the bundled pip is usually out of date
pip install -e ".[dev,llm,scoring]"       # ~a few hundred MB: torch lives in `scoring`
pre-commit install                        # secret scanning
cp config/settings.example.yaml config/settings.yaml
```

Then:

```bash
make daily          # build today's snapshot  (first run: see the note below)
make dashboard      # serve it at http://localhost:8000
```

> **The first `make daily` is slower than the rest.** With `[scoring]` installed it
> downloads Mformer — five RoBERTa models, a couple of GB — from the Hugging Face Hub
> before it can score anything. That happens once and is cached. You will also see
> `You are sending unauthenticated requests to the HF Hub`; it is a rate-limit notice,
> not an error, and setting `HF_TOKEN` silences it (see the keys table below).

<details>
<summary><b>What each install extra buys you</b> — and what you lose without it</summary>

Every extra is optional. The pipeline degrades with a logged reason rather than
failing, so an omission is quiet — which is exactly why the cost is spelled out here.

| extra | brings in | without it |
| --- | --- | --- |
| *(none)* | dictionary scoring, dedup, clustering, JSD | the pipeline runs and the dashboard renders |
| `scoring` | `torch`, `transformers` (Mformer) | `transformer tagger unavailable (No module named 'torch')` — dictionary-only scoring, and **no confidence bands anywhere**, which are the §5 payoff |
| `llm` | the `anthropic` SDK | summaries fall back to a numbers-only template; liberty is never scored |
| `embeddings` | `sentence-transformers` | clustering uses the built-in hashing embedder — workable, but blindspots are sharper with neural embeddings |
| `dev` | pytest, ruff, pre-commit | no test suite, no hooks |

Adding one later is just re-running the install: `pip install -e ".[dev,llm,scoring]"`
is idempotent.

</details>

### Keys and environment

Nothing in this repo reads a `.env` file. `.env.example` documents which variables
exist; export them in your shell profile, and set them **inside the crontab** for
scheduled runs, since cron does not inherit your shell environment.

| variable | needed for | without it |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Claude summaries, liberty tagging | five foundations instead of six, and a numbers-only summary. The run still completes. |
| `HF_TOKEN` | Mformer downloads | works fine, just unauthenticated: lower Hub rate limits, slower first download, and a warning on every run |
| `PARALLAX_SMTP_*`, `PARALLAX_DIGEST_TO` | the email brief | no email — see [the brief in your inbox](#the-brief-in-your-inbox) |
| `MEDIACLOUD_API_KEY` | Media Cloud queries | GDELT backfill still works; it needs no key |

Check every Claude model the configured pipeline will call — the summary, liberty
tagging, and blindspot themes are three separate model settings, and a key or an SDK
can be fine for one and not another:

```bash
make check-claude          # or: python3 -m scoring.preflight
```

One call per distinct call shape (a few hundred tokens for the lot), each in the shape
that step sends — including liberty's block-list system prompt and its JSON schema — so
an unreachable model, a model that won't take the schema, or an `anthropic` too old for
a parameter fails here rather than in tomorrow's brief. `--model ID` probes something
else instead, which is useful when comparing tiers before changing `settings.yaml`.

Then verify the rubric end to end for a fraction of a cent:

```bash
python3 -m scoring.liberty
```

It prints a scored probe if everything is wired up, and names the missing piece if not.

When the brief comes back numbers-only, its first line says which piece was missing —
an unset key, an `anthropic` package that isn't installed, a call that failed (with the
exception named, so a rejected key reads differently from a timeout), or a response with
no text in it. Only the first of those is fixed by exporting a key.

---

## Commands

| command | does |
| --- | --- |
| `make daily` | the full chain: ingest → backfill → cluster → summarize → snapshot → export (+ digest if enabled) |
| `make daily-fast` | today's feeds only; skips the slow GDELT backfill |
| `make dashboard` | serve the dashboard at <http://localhost:8000> |
| `make digest` | render the email to `data/digest-preview.html` and open it — no credentials needed |
| `make digest-send` | render and mail the brief |
| `make history` | print the recorded divergence series |
| `make validate` | score the gold set, report agreement, apply the §5 trigger |
| `make audit-lexicon` | check a lexicon for equality/proportionality asymmetry |
| `make register-probe` | check the liberty rubric scores both registers evenhandedly (costs API calls) |
| `make check-claude` | probe every Claude model the configured pipeline calls (a few tokens) |
| `make test` / `make lint` | the suite; ruff |

`make daily` is a batch job, not an interactive command — expect minutes, not seconds.
It narrates as it goes; `-v` explains individual fetch failures and `--quiet` turns the
narration off for cron.

---

## The brief in your inbox

A dashboard you have to remember to open is one you check twice and then forget. The
`digest` step renders the same content into a self-contained email so it arrives on its
own.

**`make daily` does not send email until you set this up.** `digest` is the seventh step
and the only one off by default — every other step works without credentials, and a step
that fails every morning until configured trains you to ignore the report that exists to
tell you when something actually broke.

**1. See it first, with no credentials at all.**

```bash
make digest         # renders to data/digest-preview.html and opens it
```

If the layout is wrong, fix it here rather than by mailing yourself tests.

**2. Export the four SMTP variables.** All four are required; a partially configured
mailer declines to send and names the missing one.

```bash
# in ~/.zshrc or ~/.bashrc — editing the file beats typing `export`, which would
# leave the password in your shell history
export PARALLAX_SMTP_HOST=smtp.gmail.com
export PARALLAX_SMTP_USER=you@gmail.com
export PARALLAX_SMTP_PASSWORD=abcd-efgh-ijkl-mnop   # Gmail: an app password
export PARALLAX_DIGEST_TO=you@gmail.com
```

For Gmail that must be an [app password](https://myaccount.google.com/apppasswords) —
with 2FA on, Google rejects your account password outright. Most providers also silently
drop mail whose `From` doesn't match the authenticated user, which is why
`PARALLAX_DIGEST_FROM` defaults to `PARALLAX_SMTP_USER`.

**3. Send one, to prove the credentials work.**

```bash
make digest-send
```

Success is silent apart from exit code 0 and mail in your inbox. Failure prints the
actual reason — unconfigured, refused connection, rejected login, or a certificate that
would not verify.

**4. Turn it on**, in `config/settings.yaml`:

```yaml
digest:
  enabled: true
  own_diet: self      # your diet's id from sources.yaml — puts your blindspots first
```

From then on `make daily` ends by mailing you the brief it just built.
`python3 -m daily --only digest` sends one without re-running the pipeline.

<details>
<summary><b>Why an email and not a hosted page</b></summary>

Hosting the static page and sending yourself a link would put a dated, running record of
one person's news consumption at a URL, permanently, to save a scroll. The email carries
everything instead and nothing is published anywhere.

The cost is real: no hover, no drill-down, no interactivity. The radar becomes two
aligned bar lists and the JSD series becomes a column chart of coloured divs, because
mail clients strip JavaScript and block remote images.

TLS certificates are verified — `smtplib`'s own default is *not* to verify them, which
would hand your mail password to anyone on the network path — and a verification failure
refuses the send rather than proceeding. [`digest/README.md`](digest/README.md) has the
detail, including why `PARALLAX_SMTP_STARTTLS=0` is refused for a remote host.

</details>

## Running it every morning

<details>
<summary><b>macOS: launchd, with secrets kept out of the plist</b></summary>

Use `launchd`, not cron. Cron does not fire while the lid is shut and simply skips
the run; a `launchd` agent runs the missed job on the next wake.

`scripts/parallax-daily.sh` is a wrapper that fetches secrets at run time instead of
storing them in the plist, so the plist holds nothing sensitive.

> **The checkout itself cannot live in `~/Documents`, `~/Desktop`, or `~/Downloads`.**
> macOS blocks background processes from reading those folders unless the specific
> process has been granted access, and a `launchd` agent has no way to request it —
> there is no one to show the permission dialog to. The job fails with an error
> that looks nothing like a permissions problem:
>
> ```
> shell-init: error retrieving current directory: getcwd: cannot access parent
> directories: Operation not permitted
> /bin/bash: .../scripts/parallax-daily.sh: Operation not permitted
> ```
>
> Terminal itself has a standing grant for those folders from when you use them
> interactively, which is why `make daily` typed by hand works fine from the same
> path — the restriction is specific to unattended processes. Clone or move the
> repo somewhere outside them first, e.g. `~/parallax` or `~/Developer/parallax`.
> If you've already been developing from inside `~/Documents`, moving it after the
> fact needs the venv rebuilt (its paths are absolute) and the plist's paths
> updated to match — both covered below.

```bash
mkdir -p ~/.config/parallax                      # cp will not create it for you
cp scripts/bitwarden.conf.example ~/.config/parallax/bitwarden.conf
chmod 600 ~/.config/parallax/bitwarden.conf      # the wrapper refuses looser modes
nano ~/.config/parallax/bitwarden.conf           # token, addresses, secret UUIDs

make check-secrets                               # resolves everything, runs nothing

mkdir -p ~/Library/LaunchAgents
cp scripts/com.parallax.daily.plist.example \
   ~/Library/LaunchAgents/com.parallax.daily.plist
nano ~/Library/LaunchAgents/com.parallax.daily.plist   # absolute paths, 4 places
launchctl load ~/Library/LaunchAgents/com.parallax.daily.plist
launchctl start com.parallax.daily                # test now; don't wait for 06:00
tail -f data/daily.log
```

**If you downloaded `bws` by hand** rather than via Homebrew, move it somewhere
stable and make it executable first — macOS also quarantines downloaded binaries:

```bash
mkdir -p ~/.local/bin
mv ~/Documents/bws ~/.local/bin/
chmod +x ~/.local/bin/bws
xattr -d com.apple.quarantine ~/.local/bin/bws 2>/dev/null   # Gatekeeper
~/.local/bin/bws --version                                    # confirm it runs
```

Then set `BWS_BIN=~/.local/bin/bws` in the config and the wrapper will find it — it
needs a *path*, not `PATH`. For interactive use (`bws secret list`) add the directory
to your shell as well:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && exec zsh
```

Those are two separate mechanisms and both are worth having: `PATH` is for you at a
prompt, `BWS_BIN` is for `launchd`, which inherits no `PATH` at all.

**Do not leave it in `~/Documents` or `~/Desktop`.** With iCloud Drive syncing those
folders — the macOS default — the binary can be evicted to a placeholder, and the 6am
run then fails intermittently with nothing obvious to point at.

`nano` because it is always present; substitute whatever you use (`vim`, `code -w`,
`open -a TextEdit`). **Do not `sudo` any of this** — everything above lives in your own
home directory and needs no elevation. Worse, `sudo` leaves the files owned by root,
which the wrapper cannot read when `launchd` runs it as you: the mode check passes at
0600 and the read fails afterwards. If you already did, `sudo chown "$(whoami)"` the
file to undo it.

`make check-secrets` prints each variable, where it came from, and its length —
never its value, since that output can land in a log. Run it before trusting a
scheduled run; a wrapper that fails at 6am on a machine nobody is watching is the
whole failure mode this is meant to avoid.

**What the secret handling does and does not buy you.** An unattended job cannot type
a master password, so something readable without a human has to exist on the disk.
This does not remove that; it changes what the readable thing unlocks:

| | readable secret | what it unlocks |
| --- | --- | --- |
| secrets in the plist | Gmail app password | the whole mailbox, read and send |
| this wrapper | Secrets Manager access token | read-only, one project, revocable on its own |

That is a genuine improvement in blast radius, and it is the entire benefit. Anyone
claiming a scheduled job can hold no secret at all is describing a job that needs a
human every morning.

It also means the Bitwarden **password manager** CLI (`bw`) is the wrong tool: `bw`
needs a master password or a live `BW_SESSION` to unlock, so making it unattended
means storing the master password — strictly worse than the app password it replaces.
**Secrets Manager** (`bws`) exists for this case, with a machine account and a scoped
token. Set `BWS_BIN` if yours isn't in the usual Homebrew or cargo location; a bare
name will not resolve under launchd, which inherits no `PATH`.

Two more things that bite:

- **launchd inherits nothing** — no `PATH`, no working directory, none of your shell
  exports. Every path in the plist and the config is absolute for that reason. It is
  the most common way a working manual run fails on a schedule.
- **launchd will not wake a sleeping Mac.** A closed lid at 06:00 means a late run,
  not a missed one — but a Mac asleep past the next scheduled time runs only once.
  Gaps show up in the divergence series; `python3 -m compare.history --backfill`
  partially reconstructs them.

If you would rather not keep any token on disk, macOS Keychain is the other reasonable
answer on a single machine: encrypted at rest, unlocked at login, and readable from a
user agent via `security find-generic-password`. It trades portability for having no
secret file at all.

</details>

<details>
<summary><b>Linux and always-on machines: systemd or cron</b></summary>

A systemd user timer is the equivalent, and the same wrapper works — point
`ExecStart` at `scripts/parallax-daily.sh`. On a machine that never sleeps, plain
cron is fine too:

```cron
0 6 * * *  cd /path/to/parallax && ./scripts/parallax-daily.sh >> data/daily.log 2>&1
```

Without the wrapper, remember that **cron does not inherit your shell environment**,
so the variables have to be set inside the crontab itself.

</details>

---

## How it works

<details>
<summary><b>Architecture and directory map</b></summary>

Python owns ingestion, NLP, scoring, and comparison. TypeScript and D3.js own the
dashboard. R is there for statistical exploration.

```
  RSS / GDELT / Media Cloud / podcasts / YouTube
                    │
              [ ingestion ]      → ingestion/
                    ▼
              [ datastore ]      SQLite (MVP) → Postgres + pgvector
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    [ dedup ]   [ scoring ]  [ embeddings ]   → scoring/
                    │           │
                    │      [ clustering ]      → cluster/ (blindspot detection)
                    ▼           ▼
              [ compare ]  JSD, CLR/Aitchison  → compare/
                    │
              [ summarize ]  Claude map-reduce  → summarize/
                    │
              [ dashboard ]  static site + D3   → dashboard/
```

| Directory | Responsibility |
| --- | --- |
| `config/` | The source registry (`sources.yaml`) and example settings |
| `ingestion/` | RSS, GDELT, Media Cloud, podcast audio, YouTube |
| `scoring/` | Moral-foundations scoring (dictionary + transformer + Claude) |
| `cluster/` | Embeddings, UMAP + HDBSCAN, blindspot detection, theme grouping |
| `compare/` | JSD, agenda divergence, CLR/Aitchison distance, log-ratios, dated snapshot history |
| `summarize/` | Map-reduce LLM summarization |
| `dashboard/` | TypeScript + D3.js static site |
| `digest/` | The dashboard rendered into a daily email |
| `daily/` | One-command snapshot orchestrator |
| `validation/` | Hand-coded gold set, agreement metrics, notebooks |
| `data/` | Gitignored working data |

</details>

<details>
<summary><b>Fairness, split two ways</b> — and a bug that hid inside a word list</summary>

MFQ-2 divides Fairness into equality (equal treatment, equal outcomes) and
proportionality (reward tracking merit and contribution). The distinction earns its place
here. Both diets argue about fairness constantly, and a five-way tagger reports that as
one number, which hides the more interesting fact: they often mean different things by it.
One asks who is being left out. The other asks whether reward is tracking contribution.
Each is a coherent account of fairness, and neither reduces to the other.

The dashboard shows the division per diet, with coverage next to it. Read it with more
suspicion than anything else on the page. No validated dictionary implements this split,
so Parallax partitions fairness using a hand-built term list, and the prediction it tests
(equality on the left, proportionality on the right) replicates poorly when measured in
language.

```yaml
scoring:
  taggers:
    dictionary:
      split_fairness: true
      fairness_min_evidence: 2   # split-terms required before partitioning at all
```

Building this turned up a bug worth mentioning, because it is the kind that hides well.
The built-in seed lexicon's fairness vocabulary contained no proportionality terms at all,
so a merit-framed argument scored as containing *no fairness*. That systematically
under-measured whichever diet frames fairness as proportion: the symmetry requirement
failing quietly inside a word list.

Since a word list can carry a political asymmetry while looking like a neutral instrument,
that check is now a command rather than a memory:

```bash
make audit-lexicon                                            # the built-in seed
python3 -m validation.lexicon_audit --lexicon data/emfd_scoring.csv
```

It reports whether either half of fairness is missing outright, and how much fairness each
half's vocabulary contributes per occurrence. Run against the real eMFD the categorical bug
does not reproduce — merit vocabulary is there, and merit-framed text does score fairness.
A milder tilt in the same direction does show up, because half the merit terms the eMFD
contains are assigned to other foundations (*accountable* to authority, *contribution* to
loyalty, *effort* to care) and contribute nothing to fairness. `LIMITATIONS.md` has the
numbers and what they do and don't support.

</details>

<details>
<summary><b>Liberty, the foundation nothing else can score</b></summary>

The eMFD, the MFD, and MFD 2.0 all cover five foundations. Mformer's training corpus does
not label liberty either. So liberty/oppression falls to Claude, which is what `CLAUDE.md`
§3(a) assigns it to and what `scoring/liberty.py` implements.

The rubric is the part worth reading. Liberty is claimed in two registers: freedom from
state coercion (mandates, overreach, censorship, conscience claims) and freedom from
private or structural domination (corporate power, surveillance, bodily autonomy, employer
control). A rubric that recognized only one would score one diet as caring about liberty
and the other as silent — a coverage artifact wearing the clothes of a finding, and the
same failure the fairness lexicon audit caught. Both registers are named explicitly, and a
test asserts they stay named.

A test can only check that the words are in the prompt, though — not that the model obeys
them. So there is a probe for that too:

```bash
make register-probe                 # 10 pairs x 2 registers x 3 repeats = 60 calls
```

Ten sentences, each with a single `{actor}` slot, rendered once with a state actor and once
with a private one. The two sides are identical word for word because they come from the
same string — matched by construction rather than by my judgement, since hand-writing two
"equivalent" sentences would put my own framing instincts inside the instrument. It reports
whether the registers get classified correctly, and whether one scores systematically
higher, with the run-to-run noise floor printed next to the gap so a few samples don't get
read as a result. A gap under the noise is absence of evidence, not a clean bill of health,
and the report says so in those words.

**It has been run, and it found something.** Classification is clean — 100% correct across
200 samples on Sonnet 5. Magnitude is not: state-actor sentences score ~0.06 higher than
word-for-word identical private-actor ones, and the gap is concentrated in four of ten
domains rather than spread evenly. So a liberty gap between the two diets contains some
instrument before it contains any media. The rubric was deliberately not retuned — that
would overfit it to the templates that measure it. `validation/README.md` has the numbers
and `LIMITATIONS.md` has what they mean for reading the dashboard.

```yaml
scoring:
  taggers:
    liberty:
      enabled: true
      model: claude-sonnet-5
      effort: low       # this matters: adaptive thinking at default `high` bills far more
      batch: true       # Batch API — half price, and the daily run is overnight anyway
```

At roughly 200 documents a day the batched cost is about $17/month on Sonnet 5, $8 on
Haiku 4.5, $42 on Opus 5 — but don't take the tier on faith:

```bash
python3 -m validation --scorer liberty --model claude-haiku-4-5 --limit 40
python3 -m validation --scorer liberty --model claude-sonnet-5  --limit 40
```

That reports AUC/F1/kappa against hand-coded liberty labels, the same as every other
scorer. It needs those labels to exist first — no public corpus supplies them, so the gold
set has to be coded for liberty by hand, and the CLI says so rather than scoring against
all-zeros and printing a number that means nothing.

Liberty appears as its own panel rather than a sixth spoke on the radar. The radar is a
composition over documents every tagger saw; liberty is scored on feed-ingested documents
only, and only when a key is set. Folding partial coverage into a composition would move
every other share, and would shift the headline divergence the snapshot history has been
recording since the series started.

</details>

<details>
<summary><b>Inside the daily run</b> — narration, timing, and tuning</summary>

Steps are isolated. If one fails (GDELT throttling, no API key), the rest still run so the
dashboard reflects whatever data landed, the report names what broke, and the exit code is
non-zero so cron notices. The transformer is loaded once and shared across steps.

```bash
make daily-fast                      # today's feeds only; skips the slow backfill
python3 -m daily --skip backfill
python3 -m daily --only cluster export
python3 -m daily --backfill-days 3 --max-per-source 100   # cheaper daily window
```

The backfill is on by default because blindspots are only trustworthy with real per-outlet
volume behind them. Tune it in `config/settings.yaml` under `daily.backfill`, or disable it
there. Re-runs are idempotent thanks to URL-canonical dedup, so overlapping windows cost
time, not correctness.

Two things dominate the runtime: transformer scoring runs five RoBERTa models over every
ingested article (seconds per article on CPU), and GDELT's free endpoint throttles to
roughly one request every five seconds across the whole registry. That is the cost of
confidence bands and trustworthy blindspots. For speed: run it overnight from cron, use
`make daily-fast`, shorten with `--backfill-days 3`, or pass `--no-transformer` for a fast
dictionary-only refresh with no confidence bands.

Because it is slow, it narrates:

```
→ ingest
  [ 1/16] self_nyt_home              self         3 stored / 3 fetched         2.1s
  [ 2/16] self_wapo_national         self         2 stored / 3 fetched, 1 unreadable  14.7s
  [ 3/16] self_guardian_us           self         feed unreachable (30.4s)
```

The elapsed seconds are the useful part: sources take a couple of seconds each, so a
double-digit number means something hit the 30-second fetch timeout, and a source that
never prints is the one you are currently waiting on. A dead feed is called out separately
from articles that failed to fetch — only the first needs a fix in `config/sources.yaml`.

Liberty tagging is the one step that goes quiet on purpose. Above ten documents it submits
a Batch API job and polls every twenty seconds, so the run can sit silent for minutes
*after* ingestion has visibly finished. It says so before it starts, because otherwise that
pause is indistinguishable from a hang at the very last step.

</details>

<details>
<summary><b>Two divergences</b> — why the headline number is small and the diets still feel unrecognisable</summary>

The headline Jensen-Shannon divergence compares two five-number *foundation compositions*.
On a real corpus it comes out small — a few thousandths — because those numbers are
averages over hundreds of documents, and averages over hundreds of documents converge.
That is a genuine finding, not a broken metric: care and fairness lead in both diets, the
binding foundations are present in both, and the strong form of the liberal/conservative
asymmetry hypothesis (§5 of `CLAUDE.md` — the thing this tool is supposed to *test*) does
not survive it.

It is also not what a reader experiences. What they experience is the agenda: two people
reading about different events. `compare/agenda.py` measures that, with the same
Jensen-Shannon machinery applied to each diet's distribution of attention across story
clusters:

| Metric | Question it answers | Typical size |
| --- | --- | --- |
| Foundation divergence | What moral vocabulary does each diet speak? | Small — they largely speak the same one |
| Attention divergence | What did each diet spend the day on? | Large — mostly different stories |
| Exclusive share | What fraction of a diet's articles were about stories the other never touched? | The number a person feels |

Both are printed together, on one scale, in the email and on the dashboard. Either alone
misleads: the first says "nearly identical", the second says "different worlds", and the
honest reading is that they moralize alike about different events.

What is *not* built yet is the metric between them — foundation vectors computed **within**
the clusters both diets covered, which would isolate framing from agenda and answer whether
the same event gets a different moral treatment. Until that exists, "they moralize alike"
and "they moralize the same events alike" are separate claims and only the first is
supported. See `LIMITATIONS.md`.

</details>

<details>
<summary><b>Divergence over time</b> — two bases, and why backfilled points are shaded</summary>

Each run records one dated snapshot of compositions, JSD, log-ratios, and document counts,
so the numbers accumulate into a series instead of overwriting yesterday. Re-running on the
same day replaces that day's row, so the series stays at one point per day however often
the pipeline runs.

Every snapshot is computed on two bases, because they answer different questions:

| Basis | What it profiles | How to read it |
| --- | --- | --- |
| Trailing window (7 days by default) | Only documents dated in the window | The basis that can actually respond to an event. Noisier, and on thin days it *is* noise. |
| All-time | Every document dated on or before that day | The headline number the radar reports. Heavily damped, and it moves less the longer the project runs. |

Neither is the truer number, and the dashboard labels both rather than picking one.

```bash
python3 -m compare.history                      # print the recorded series
python3 -m compare.history --backfill 30        # reconstruct 30 past days
python3 -m daily --window-days 14               # widen the trailing window
python3 -m dashboard.export --history-limit 90  # ship 90 days to the dashboard
```

`--backfill` reconstructs past days from publication dates already in the store, so the
chart is useful on day one rather than after a month of runs. It computes the same
arithmetic on a different corpus, though. A reconstructed row for last Tuesday includes
articles published then but fetched since (GDELT backfill pulls weeks of history), which a
live run that day could not have seen. Reconstructed rows are therefore *what the corpus
now says about that date*, not what the dashboard would have shown. The chart shades them,
and reconstruction will not overwrite a live row unless you pass `--overwrite`.

</details>

<details>
<summary><b>Running the steps individually</b></summary>

```bash
# 1. Fetch every RSS source, extract bodies, dedup, score, embed, and store derived
#    metrics to SQLite (raw text is never persisted). Every article is scored by BOTH
#    the dictionary and the transformer, so the dashboard can show a confidence band.
#    Add --no-transformer for the fast dictionary-only path. Prints a line per source;
#    -v explains individual fetch failures, -q suppresses the narration.
python3 -m ingestion run --max-items 25

# 1b. Backfill weeks of history per outlet from GDELT (title-based, fast, no key).
#     This is the volume that makes blindspots reliable. --extract also fetches
#     article bodies for full scoring, and is much slower.
python3 -m ingestion backfill --days 14 --max-per-source 250

# 2. Print each diet's composition, the JSD, and the per-foundation log-ratios:
python3 -m ingestion compare

# 3. Cluster stories from stored embeddings and detect blindspots in both directions,
#    then group them into named themes (Claude when ANTHROPIC_API_KEY is set, else a
#    built-in taxonomy; --no-claude-themes forces the taxonomy):
python3 -m cluster run

# 4. Charitable daily summary per diet + a cross-diet executive summary (Claude when
#    ANTHROPIC_API_KEY is set, else a clearly-labeled numbers-only fallback).
#    Model: --model, else `summarize.model` in settings, else claude-opus-5:
python3 -m summarize

# 5. Export the dashboard payload, then view it:
python3 -m dashboard.export
cd dashboard && python3 -m http.server      # http://localhost:8000

# Validate a scorer against the gold set: per-foundation AUC/F1/kappa and the §5
# trigger (binding foundations below 0.7 AUC warrant the transformer):
python3 -m validation --lexicon data/emfd_scoring.csv
python3 -m validation --scorer transformer   # Mformer; needs the scoring extra
python3 -m validation --scorer ensemble      # + a disagreement-based confidence report
```

Story clustering embeds each document at ingestion (text is discarded, so embeddings are
persisted). The default embedder is a dependency-free hashing embedder over headlines.
Setting `cluster.embedder.kind: sentence-transformers` swaps in neural embeddings for
sharper clusters.

</details>

<details>
<summary><b>Using the real eMFD lexicon</b> — the default is a demo, not an instrument</summary>

By default scoring uses a built-in demo lexicon so the pipeline runs with zero external
data. It is a placeholder, not a validated instrument. For real results, supply the eMFD:

```bash
# 1. Drop the eMFD CSV in data/ (gitignored), from the eMFDscore repo:
#    dictionaries/emfd_scoring.csv (columns: word, <foundation>_p, <foundation>_sent).
# 2. Either set scoring.taggers.dictionary.lexicon_path in config/settings.yaml
#    (note `taggers` — that is the key PipelineConfig.from_settings reads), or:
python3 -m ingestion run --lexicon data/emfd_scoring.csv
```

**Switch the lexicon on a fresh datastore.** Dictionary scores all land under one scorer
key, ingestion never re-scores documents it already holds, and raw text is not persisted —
so swapping the lexicon appends a second instrument's numbers into the same column and
every aggregate blends the two, while the caveat reports only the newer name. A run warns
when it notices the change, but warning is all it can do. `rm data/parallax.sqlite` (or
move it aside) before the first run on a new lexicon; the store holds derived metrics, not
text, so nothing irreplaceable is lost — though the snapshot history restarts with it.

The active lexicon is recorded with the scores, so the dashboard caveat and summaries state
which one produced the numbers. Because eMFD words carry probability across all five
foundations, the scorer defaults to `assignment: argmax`, where each word counts toward its
dominant foundation. See `LIMITATIONS.md` for why, and for what the eMFD's low aggregate
discrimination does and doesn't mean.

</details>

---

## Status

Phases 1–3 are complete: extraction, dedup, dictionary **and** transformer scoring, a
validation gold set with an ensemble confidence signal, coverage-asymmetry blindspot
detection, GDELT historical backfill, dated snapshot history, the Claude liberty tagger,
the equality/proportionality split, and the email brief.

With the transformer running at ingestion, each foundation carries a
dictionary-vs-transformer band — wider means more disagreement and lower confidence. See
[`CLAUDE.md`](CLAUDE.md) for the full build spec and roadmap; Phase 4 (podcast and video
transcription) is the next substantial piece.

## Reading the numbers honestly

Every foundation number here is an estimate produced by imperfect instruments over a
contested construct. Dictionary methods have poor convergent validity; fine-tuned
transformers do better in-domain and degrade across domains; liberty rests on a single
model against a single hand-written rubric and has never been checked against a human
coder. The tool reports ensemble *disagreement* as its confidence signal rather than
forcing a label, and the liberal/conservative foundation asymmetry is presented as a
hypothesis it tests, not an axiom it assumes.

[`LIMITATIONS.md`](LIMITATIONS.md) is where all of that lives, with numbers. Read it before
you believe anything on the dashboard.

## License

[AGPL-3.0](LICENSE). Derived metrics and code are shareable under its terms; raw source
content is never committed or redistributed.
