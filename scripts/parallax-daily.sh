#!/bin/bash
# Run the daily snapshot with secrets fetched at runtime instead of stored in
# the launchd plist.
#
#   parallax-daily.sh            # fetch secrets, run `python3 -m daily`
#   parallax-daily.sh --check    # resolve every secret and report, run nothing
#
# WHAT THIS DOES AND DOES NOT BUY YOU
#
# An unattended 6am job cannot type a master password, so something readable
# without a human has to exist on this machine. This script does not eliminate
# that; it changes what the readable thing unlocks:
#
#   before  a Gmail app password in the plist  -> full mailbox, read and send
#   after   a Secrets Manager access token     -> read-only, one project, revocable
#
# That is a real improvement in blast radius, and it is the whole benefit. Anyone
# who claims a scheduled job can hold no secret at all is describing a job that
# needs a human every morning.
#
# It also means the Bitwarden *password manager* CLI (`bw`) is the wrong tool
# here: `bw` needs a master password or a live BW_SESSION to unlock, so making it
# work unattended means storing the master password — which is strictly worse
# than the app password it replaces. Secrets Manager (`bws`) exists for exactly
# this case: a machine account with a scoped, independently revocable token.

set -euo pipefail

CONFIG="${PARALLAX_BITWARDEN_CONF:-$HOME/.config/parallax/bitwarden.conf}"
PREFIX="bws:"

die() { printf 'parallax-daily: %s\n' "$1" >&2; exit 1; }

# -- locate bws -------------------------------------------------------------
# launchd inherits no PATH, so a bare `bws` resolves under an interactive shell
# and not at 6am. That asymmetry is the single most common way a working manual
# run fails as a scheduled one, so the lookup is explicit.
find_bws() {
  if [[ -n "${BWS_BIN:-}" ]]; then
    [[ -x "$BWS_BIN" ]] || die "BWS_BIN=$BWS_BIN is not executable"
    printf '%s' "$BWS_BIN"; return
  fi
  local candidate
  for candidate in /opt/homebrew/bin/bws /usr/local/bin/bws "$HOME/.cargo/bin/bws"; do
    [[ -x "$candidate" ]] && { printf '%s' "$candidate"; return; }
  done
  command -v bws 2>/dev/null && return
  die "bws not found. Install the Bitwarden Secrets Manager CLI, or set BWS_BIN
       to its absolute path in $CONFIG. A bare name will not resolve under
       launchd, which inherits no PATH."
}

# -- fetch one secret -------------------------------------------------------
# Verify the shape of `bws secret get` output once with --check before trusting
# a scheduled run: this parse is the part most likely to drift between CLI
# versions, and it is deliberately in one place so it is cheap to adjust.
fetch_secret() {
  local id="$1" json
  json="$("$BWS" secret get "$id" --output json 2>/dev/null)" \
    || die "bws could not read secret $id — is the token valid and scoped to its project?"
  printf '%s' "$json" | /usr/bin/env python3 -c '
import json, sys
doc = json.load(sys.stdin)
if isinstance(doc, list):          # some versions wrap a single secret in a list
    doc = doc[0]
value = doc.get("value")
if not value:
    sys.exit("no `value` field in the bws response")
sys.stdout.write(value)
' || die "could not parse the bws response for $id"
}

# -- load config ------------------------------------------------------------
[[ -f "$CONFIG" ]] || die "no config at $CONFIG (copy scripts/bitwarden.conf.example)"

# 0600 or tighter: this file holds the access token, which is the root secret.
# Via python rather than stat: the BSD `stat -f` that macOS wants means
# "filesystem stat" to GNU stat and *succeeds* with unrelated output, so the
# usual `bsd || gnu` fallback silently produces nonsense on Linux.
perms="$(/usr/bin/env python3 -c '
import os, stat, sys
print(oct(stat.S_IMODE(os.stat(sys.argv[1]).st_mode))[2:])' "$CONFIG")"
if [[ "$perms" != "600" && "$perms" != "400" ]]; then
  die "$CONFIG is mode $perms — it holds the access token, so it must be 600 or
       400 (owner-only). Run: chmod 600 $CONFIG"
fi

# Owner-only *and* owned by someone else is the sudo footgun: the mode check
# above passes, and then the read fails with a bare permission error further
# down. launchd runs this agent as you, not as root.
if [[ ! -r "$CONFIG" ]]; then
  die "$CONFIG is not readable by $(id -un). If you created it with sudo it is
       owned by root, which launchd cannot read when it runs this as you. Fix
       with: sudo chown $(id -un) $CONFIG"
fi

names=()
values=()
sources=()
token=""

while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"                       # strip comments
  line="${line#"${line%%[![:space:]]*}"}"  # ltrim
  line="${line%"${line##*[![:space:]]}"}"  # rtrim
  [[ -z "$line" ]] && continue
  [[ "$line" == *=* ]] || die "cannot parse line in $CONFIG: $line"
  key="${line%%=*}"
  val="${line#*=}"
  # Trim around the `=` as well as around the line. `KEY = VALUE` is ordinary
  # conf-file formatting, and without this it produced a key with a trailing
  # space (falling through the case below) and a value with a leading one — so
  # a spaced-out SMTP host would have been exported with a space in it and
  # failed at connect time, which is a long way from the cause.
  key="${key%"${key##*[![:space:]]}"}"
  val="${val#"${val%%[![:space:]]*}"}"
  [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] \
    || die "not a usable variable name in $CONFIG: '$key'"
  case "$key" in
    BWS_ACCESS_TOKEN) token="$val" ;;
                      # A leading ~/ is expanded here and nowhere else. Values
                      # read from a file are plain strings, so bash does no tilde
                      # expansion on them — `BWS_BIN=~/.local/bin/bws` would fail
                      # as "not executable" while looking perfectly correct.
                      # Confined to this key on purpose: silently rewriting a
                      # token or password that happened to start with ~/ would be
                      # a much worse surprise than a path that has to be absolute.
    BWS_BIN)          BWS_BIN="${val/#\~\//$HOME/}" ;;
    *)                names+=("$key"); values+=("$val")
                      if [[ "$val" == "$PREFIX"* ]]; then sources+=("bitwarden")
                      else sources+=("literal"); fi ;;
  esac
done < "$CONFIG"

[[ ${#names[@]} -gt 0 ]] || die "no variables defined in $CONFIG"

# Only pay for bws when something actually needs fetching.
needs_bws=0
for source in "${sources[@]}"; do [[ "$source" == "bitwarden" ]] && needs_bws=1; done
if [[ "$needs_bws" == 1 ]]; then
  [[ -n "$token" ]] || die "BWS_ACCESS_TOKEN is not set in $CONFIG, but some values use $PREFIX"
  BWS="$(find_bws)"
  export BWS_ACCESS_TOKEN="$token"
fi

# -- resolve ----------------------------------------------------------------
check_only=0
[[ "${1:-}" == "--check" ]] && check_only=1

resolved=0
for i in "${!names[@]}"; do
  name="${names[$i]}"
  raw="${values[$i]}"
  if [[ "${sources[$i]}" == "bitwarden" ]]; then
    value="$(fetch_secret "${raw#"$PREFIX"}")"
  else
    value="$raw"
  fi
  [[ -n "$value" ]] || die "$name resolved to an empty value"
  export "$name=$value"
  resolved=$((resolved + 1))
  # Length only, never the value — this output goes to a log file.
  [[ "$check_only" == 1 ]] && printf '  %-28s %-10s ok (%d chars)\n' \
    "$name" "${sources[$i]}" "${#value}"
done
unset BWS_ACCESS_TOKEN     # the job itself has no business holding the token

if [[ "$check_only" == 1 ]]; then
  printf '\n%d variable(s) resolved. Nothing was run.\n' "$resolved"
  exit 0
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PARALLAX_PYTHON:-./.venv/bin/python3}" -m daily
