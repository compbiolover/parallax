# Two images from one file, so the layers they share are shared in fact and not
# just in intention.
#
#   docker build -f docker/Dockerfile --target core   -t parallax-core   .
#   docker build -f docker/Dockerfile --target scored -t parallax-scored .
#
# `core` (~350 MB) runs cluster, summarize, snapshot, export and digest. It has
# no torch at all: the default embedder is `hashing` (config/settings.example.yaml)
# and every heavy tagger is lazy-imported behind ImportError handling, so the
# aggregate half of a run genuinely does not need the scoring stack.
#
# `scored` (~2.5 GB) runs ingest and backfill, where the transformer tagger and
# neural embeddings live.
#
# Both carry the `aws` extra: the entrypoint fetches the database and the lexicon
# from S3 before the run and puts them back after (deploy/state.py), and the
# aggregate half needs that as much as the ingest half does. It stays inert when
# PARALLAX_STATE_BUCKET is unset, so the image still runs with no cloud account.
#
# The media extra (faster-whisper, yt-dlp, ffmpeg) is deliberately absent. Podcast
# transcription runs in hours rather than minutes and is the largest single line
# of the bill; it stays on the workstation until it earns its own image.
#
# What enters the build context is decided by .dockerignore, not by a list of
# COPY lines kept in step with the package layout. That matters for more than
# tidiness: config/personas.local.yaml is a real person's real media diet, it is
# gitignored and therefore invisible to review, and `COPY config/` would bake it
# into a layer that gets pushed to a registry.

# ---------------------------------------------------------------- base --------
FROM python:3.11-slim-bookworm AS base

# PYTHONUNBUFFERED: logs are the only view of a scheduled run, and a crash with a
# full stdout buffer loses exactly the lines that explain it.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ca-certificates because every fetch is HTTPS. git because eMFDscore installs
# from a repository rather than PyPI.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ------------------------------------------------------------ torch cache -----
# Its own stage so the largest download in the build does not depend on a source
# file. Installed from the CPU index: the default PyPI wheel bundles the CUDA
# runtime at ~2.5 GB, and there is no GPU anywhere in this design — that would be
# 2.3 GB pulled on every task start to be never used.
FROM base AS torch-cpu
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

# ---------------------------------------------------------------- core --------
FROM base AS core

COPY . .
RUN pip install -e ".[llm,aws]" \
 && chmod +x scripts/parallax-entrypoint.sh \
 && ln -s /app/scripts/parallax-entrypoint.sh /usr/local/bin/parallax-entrypoint \
 && mkdir -p /app/data \
 && chown -R nobody:nogroup /app

USER nobody
ENTRYPOINT ["parallax-entrypoint"]
CMD ["--help"]

# --------------------------------------------------------------- scored -------
FROM torch-cpu AS scored

COPY . .

# torch is already satisfied from the stage above, so this resolves the rest.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -e ".[llm,scoring,embeddings,aws]"

# eMFDscore is not on PyPI and is not a declared dependency. Pinned to a commit
# rather than a branch, because an unpinned scorer changes what the numbers mean
# between two builds of the same source and nothing downstream would report it.
# Override the ARG to move it deliberately; do not point it at a branch.
ARG EMFDSCORE_REF=5523d76c25afa65a59ca00e44efe45885499b116
RUN pip install "git+https://github.com/medianeuroscience/emfdscore.git@${EMFDSCORE_REF}"

# Bake the five Mformer models plus the shared tokenizer in. They are ~500 MB,
# they never change for a given revision, and downloading them at task start puts
# a Hugging Face outage on the critical path of the daily run.
ENV HF_HOME=/opt/hf
RUN python docker/warm_models.py

# With the cache populated the run needs neither the network nor a token to load
# them, which takes HF_TOKEN out of the runtime secret set entirely.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN chmod +x scripts/parallax-entrypoint.sh \
 && ln -s /app/scripts/parallax-entrypoint.sh /usr/local/bin/parallax-entrypoint \
 && mkdir -p /app/data \
 && chown -R nobody:nogroup /app /opt/hf

USER nobody
ENTRYPOINT ["parallax-entrypoint"]
CMD ["--help"]
