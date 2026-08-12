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
#
# `mkdir -p` alone proves nothing: it succeeds when the directory already exists,
# which is exactly the case when a volume is mounted over it — and a mount
# defaults to root ownership while this container runs as nobody. The failure
# then surfaces much later as sqlite's "unable to open database file", after the
# run has already fetched and scored. Cheaper to find out now, and to say why.
mkdir -p /app/data
if ! (: > /app/data/.write-test) 2>/dev/null; then
  echo "FATAL: /app/data is not writable by $(id -un)." >&2
  echo "  A volume mounted there is owned by root by default, while this image" >&2
  echo "  runs as nobody. Either chown it to $(id -u):$(id -g) on the host, or" >&2
  echo "  run with --user \$(id -u):\$(id -g)." >&2
  exit 1
fi
rm -f /app/data/.write-test

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

from ingestion.config import dictionary_lexicon_path, load_settings

settings = load_settings()
taggers = (settings.get("scoring") or {}).get("taggers") or {}
dictionary = taggers.get("dictionary") or {}

# Nothing to guard when the dictionary tagger is off: no lexicon is loaded, so
# there is no silent fallback to refuse.
if not dictionary.get("enabled", True):
    print("lexicon    dictionary tagger disabled, guard skipped", flush=True)
    raise SystemExit(0)

# Resolved by the same function the pipeline uses, so the file this checks and
# the file `build_lexicon` opens cannot be different ones. A guard that resolves
# a path its own way is a guard that can pass while the run falls back anyway.
configured = dictionary_lexicon_path(settings)

if not configured:
    sys.exit(
        "PARALLAX_REQUIRE_LEXICON=1 but scoring.taggers.dictionary.lexicon_path "
        "is unset, so the run would score with the demo seed lexicon."
    )
if not Path(configured).exists():
    sys.exit(
        f"PARALLAX_REQUIRE_LEXICON=1 and lexicon_path resolves to {configured!r}, "
        "which does not exist. Refusing to score a corpus with the demo seed "
        "lexicon, which is illustrative only. Fetch the real emfd_scoring.csv "
        "into place."
    )
print(f"lexicon    {configured}", flush=True)
PY
fi

exec "${PARALLAX_PYTHON}" -m daily "$@"
