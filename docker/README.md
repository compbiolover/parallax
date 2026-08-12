# Containers

Two images, one Dockerfile, two targets.

```sh
docker build -f docker/Dockerfile --target core   -t parallax-core   .
docker build -f docker/Dockerfile --target scored -t parallax-scored .

docker run --rm parallax-core --only export
docker run --rm parallax-scored --only ingest,backfill
```

Everything after the image name is `python -m daily`'s own argv, so the step
names in `daily/runner.py`'s `STEPS` tuple are the scheduling vocabulary in the
container exactly as they are on the command line. Adding a step does not mean
editing a shell script somewhere to match.

## Which image runs what

| | `core` (~350 MB) | `scored` (~2.5 GB) |
| --- | --- | --- |
| Steps | cluster, summarize, snapshot, export, digest | ingest, backfill |
| Extras | `.[llm]` | `.[llm,scoring,embeddings]` + eMFDscore |
| torch | none | CPU build only |

`core` has no torch because it does not need any: the default embedder is
`hashing`, and every heavy tagger is lazy-imported behind `ImportError`
handling. The aggregate half of a run genuinely does not touch the scoring
stack, and splitting the images is what keeps that true rather than incidental.

There is no media image. Podcast transcription runs in hours rather than
minutes and is the single largest line of the running cost; it stays on the
workstation until it earns its own.

## Three things that are easy to get wrong here

**The install is editable, and has to be.** `ingestion/config.py` sets
`REPO_ROOT` from `__file__`, so a normal `pip install .` resolves it to
site-packages — where `config/sources.yaml` is not. The registry then loads
from nowhere. `pip install -e .` keeps `REPO_ROOT` pointing at `/app`, which
holds `config/`. A real fix is a `PARALLAX_CONFIG_DIR` override; until then,
editable is load-bearing rather than a convenience.

**`.dockerignore` is a privacy control, not an optimisation.**
`config/personas.local.yaml` is a real person's real media consumption — the one
thing `CLAUDE.md` §0 keeps out of this repository. It is gitignored, so it is
invisible to review and entirely visible to `docker build`, and the Dockerfile
copies the tree wholesale. An image is pushed to a registry and pulled by
anything with read access. `.github/workflows/images.yml` plants a decoy
overlay before building and fails if it survives into a layer, because a
guarantee nothing tests is a guarantee that quietly stops holding.

**The eMFD lexicon is absent unless something fetches it.** It is gitignored,
so it is not in the image. `build_lexicon` warns and falls back to the built-in
demo seed, which is right on a workstation — you read the warning and fix it.
On an unattended run nobody reads the log until something looks wrong, and what
wrong looks like here is a complete, plausible, fully-populated set of numbers
produced by an instrument that was never validated. Set
`PARALLAX_REQUIRE_LEXICON=1` and the entrypoint refuses to start without the
real file. Leave it unset for steps that never touch a lexicon.

## Model weights

`scored` bakes in the five Mformer classifiers and their shared tokenizer via
`warm_models.py`, then sets `HF_HUB_OFFLINE=1`. That trades ~500 MB of image
for a task start that touches no network and needs no `HF_TOKEN` — one fewer
secret at runtime, and a Hugging Face outage no longer sits on the critical path
of the morning run. CI loads them with `--network none` to check the weights are
really present rather than being fetched on demand.
