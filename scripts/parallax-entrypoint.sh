#!/usr/bin/env bash
# Container entrypoint: everything after it is `python -m daily`'s own argv.
#
#   docker run parallax-core --only export
#   docker run parallax-scored --only ingest,backfill
#
# Deliberately thin. The step vocabulary lives in daily/runner.py's STEPS tuple
# and the scheduler passes it through verbatim, so adding a step does not mean
# editing a shell script in a second place — which is how the two drift and how
# a step silently stops being scheduled.
#
# This is *not* where the Mac's runner lives: scripts/parallax-daily.sh keeps its
# Bitwarden secret resolution and launchd assumptions. In a container the secrets
# arrive as environment variables already, injected from Secrets Manager by the
# task definition, so there is nothing to resolve here.
set -euo pipefail

: "${PARALLAX_PYTHON:=python}"

# Ephemeral task storage. Present in the image, recreated here because a mounted
# volume can shadow it with an empty directory.
mkdir -p /app/data

# The eMFD lexicon is gitignored, so it is absent unless something put it there.
# `build_lexicon` (scoring/lexicon.py) warns and falls back to the built-in demo
# seed, which is the right call on a workstation — you see the warning and fix
# it. On a scheduled run nobody reads the log until something looks wrong, and
# what "wrong" looks like here is a complete, plausible, fully-populated set of
# numbers produced by an instrument that was never validated.
#
# So a container can demand the real thing. Opt-in rather than always-on,
# because `--only export` and `--only digest` never touch a lexicon and should
# not need one present to run.
if [ "${PARALLAX_REQUIRE_LEXICON:-0}" = "1" ]; then
  "${PARALLAX_PYTHON}" - <<'PY'
import sys
from pathlib import Path

from ingestion.config import load_settings

settings = load_settings()
taggers = (settings.get("scoring") or {}).get("taggers") or {}
configured = (taggers.get("dictionary") or {}).get("lexicon_path")

if not configured:
    sys.exit(
        "PARALLAX_REQUIRE_LEXICON=1 but scoring.taggers.dictionary.lexicon_path "
        "is unset, so the run would score with the demo seed lexicon."
    )
if not Path(configured).exists():
    sys.exit(
        f"PARALLAX_REQUIRE_LEXICON=1 and lexicon_path is {configured!r}, which "
        "does not exist. Refusing to score a corpus with the demo seed lexicon, "
        "which is illustrative only. Fetch the real emfd_scoring.csv into place."
    )
print(f"lexicon    {configured}", flush=True)
PY
fi

exec "${PARALLAX_PYTHON}" -m daily "$@"
