"""The launchd wrapper: the one piece of this project that runs unattended.

Driven as a subprocess against a fake `bws`, because the failure mode that
matters is not a wrong number — it is 6am on a machine nobody is watching, and
"I checked it by hand once" does not survive the next edit.
"""

from __future__ import annotations

import os
import pathlib
import pwd
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def _uid_exists(uid: int) -> bool:
    try:
        pwd.getpwuid(uid)
        return True
    except KeyError:
        return False


ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "scripts" / "parallax-daily.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

# `nobody` where it exists, else any non-root uid — only used by the one test that
# needs the wrapper to run as a user who does not own the config.
UNPRIVILEGED_UID = next(
    (uid for uid in (65534, 1, 2) if _uid_exists(uid)), None
)

SMTP_ID = "aaaa0000-0000-0000-0000-000000000000"
KEY_ID = "bbbb0000-0000-0000-0000-000000000000"

# Returns a bare object for one id and a single-element list for the other —
# both shapes the bws CLI has used, so the parser is exercised on each.
FAKE_BWS = f"""#!/bin/bash
[[ "$1" == "secret" && "$2" == "get" ]] || {{ echo "unexpected: $*" >&2; exit 2; }}
[[ -n "${{BWS_ACCESS_TOKEN:-}}" ]] || {{ echo "no token" >&2; exit 1; }}
case "$3" in
  {SMTP_ID}) echo '{{"id":"x","key":"smtp","value":"app-pw-from-vault"}}' ;;
  {KEY_ID}) echo '[{{"id":"y","key":"anthropic","value":"sk-ant-from-vault"}}]' ;;
  *) echo "not found" >&2; exit 1 ;;
esac
"""

ECHO_ENV = """#!/bin/bash
echo "SMTP_USER=${PARALLAX_SMTP_USER:-UNSET}"
echo "SMTP_PASSWORD=${PARALLAX_SMTP_PASSWORD:-UNSET}"
echo "ANTHROPIC=${ANTHROPIC_API_KEY:-UNSET}"
echo "BWS_TOKEN=${BWS_ACCESS_TOKEN:-UNSET}"
echo "ARGS=$*"
"""

CONF = f"""BWS_ACCESS_TOKEN=0.test-token
PARALLAX_SMTP_HOST=smtp.gmail.com
PARALLAX_SMTP_USER=me@example.com
PARALLAX_DIGEST_TO=me@example.com
PARALLAX_SMTP_PASSWORD=bws:{SMTP_ID}
ANTHROPIC_API_KEY=bws:{KEY_ID}
"""


@pytest.fixture
def env(tmp_path):
    """A fake bws on disk plus a helper to run the wrapper against a config."""
    bws = tmp_path / "bws"
    bws.write_text(FAKE_BWS)
    bws.chmod(0o755)

    def run(conf_text: str, *args, mode: int = 0o600, python: str | None = None):
        conf = tmp_path / "bitwarden.conf"
        conf.write_text(conf_text)
        conf.chmod(mode)
        environ = {
            **os.environ,
            "BWS_BIN": str(bws),
            "PARALLAX_BITWARDEN_CONF": str(conf),
        }
        if python:
            script = tmp_path / "fake-python"
            script.write_text(python)
            script.chmod(0o755)
            environ["PARALLAX_PYTHON"] = str(script)
        return subprocess.run(
            ["bash", str(WRAPPER), *args],
            capture_output=True, text=True, env=environ, cwd=ROOT,
        )

    return run


# -- resolving ---------------------------------------------------------------


def test_check_resolves_literal_and_fetched_values(env):
    r = env(CONF, "--check")
    assert r.returncode == 0, r.stderr
    assert "5 variable(s) resolved" in r.stdout
    assert "PARALLAX_SMTP_HOST" in r.stdout and "literal" in r.stdout
    assert "PARALLAX_SMTP_PASSWORD" in r.stdout and "bitwarden" in r.stdout


def test_check_never_prints_a_secret_value(env):
    """This output goes to a log file, so it reports lengths, not values."""
    r = env(CONF, "--check")
    assert "app-pw-from-vault" not in r.stdout
    assert "sk-ant-from-vault" not in r.stdout
    assert "0.test-token" not in r.stdout
    assert "17 chars" in r.stdout


def test_both_bws_response_shapes_parse(env):
    """The CLI has returned a bare object and a single-element list; the parser
    handles both, and this is the part most likely to drift between versions."""
    r = env(CONF, "--check")
    assert r.returncode == 0
    assert r.stdout.count("bitwarden") == 2      # both fetched values resolved


def test_check_runs_nothing(env):
    r = env(CONF, "--check", python="#!/bin/bash\necho SHOULD-NOT-RUN\n")
    assert "SHOULD-NOT-RUN" not in r.stdout
    assert "Nothing was run" in r.stdout


# -- the config format -------------------------------------------------------


def test_spaces_around_the_equals_are_tolerated(env):
    """`KEY = VALUE` is ordinary conf formatting. Without trimming around the
    `=`, the key kept a trailing space and fell through the BWS_ACCESS_TOKEN
    case, while values kept a leading one — so a spaced-out SMTP host would have
    been exported with a space in it and failed at connect time, a long way from
    the cause."""
    spaced = CONF.replace("=", " = ")
    r = env(spaced, "--check")
    assert r.returncode == 0, r.stderr
    assert "5 variable(s) resolved" in r.stdout
    # and the literal value did not pick up the padding
    assert "PARALLAX_SMTP_HOST           literal    ok (14 chars)" in r.stdout


def test_comments_and_blank_lines_are_ignored(env):
    r = env("# a comment\n\n" + CONF + "\n   # indented comment\n", "--check")
    assert r.returncode == 0, r.stderr
    assert "5 variable(s) resolved" in r.stdout


def test_a_hash_inside_a_value_is_kept(env):
    """The regression: stripping from the first `#` anywhere cut
    `hunter#2` to `hunter`, exported the wrong password, and reported it as
    resolved. Generated passwords and subscriber feed URLs both carry `#`."""
    conf = ("BWS_ACCESS_TOKEN=0.t\n"
            "PARALLAX_SMTP_PASSWORD=hunter#2\n"
            "PARALLAX_FEED_MAKINGSENSE=https://example.com/rss?id=a#b\n")
    r = env(conf, "--check")
    assert r.returncode == 0, r.stderr
    # Lengths, not values — the whole value survived, so the counts are the
    # full ones rather than the truncated `hunter` (6) and `...?id=a` (28).
    assert "8 chars" in r.stdout           # len("hunter#2")
    assert "30 chars" in r.stdout          # len("https://example.com/rss?id=a#b")


def test_a_comment_after_a_value_is_still_stripped(env):
    """Whitespace then `#` is an ordinary trailing comment and has to go, or it
    ends up inside the exported value."""
    r = env("BWS_ACCESS_TOKEN=0.t\nPARALLAX_SMTP_HOST=smtp.example.com  # prod\n",
            "--check")
    assert r.returncode == 0, r.stderr
    assert "16 chars" in r.stdout          # len("smtp.example.com")


def test_a_value_that_is_only_a_hash_survives(env):
    """No whitespace before it, so it is a value, not a comment."""
    r = env("BWS_ACCESS_TOKEN=0.t\nPARALLAX_SMTP_PASSWORD=#\n", "--check")
    assert r.returncode == 0, r.stderr
    assert "1 chars" in r.stdout


def test_a_whitespace_only_line_is_skipped(env):
    r = env("BWS_ACCESS_TOKEN=0.t\n   \t \nPARALLAX_SMTP_HOST=h\n", "--check")
    assert r.returncode == 0, r.stderr
    assert "1 variable(s) resolved" in r.stdout


def test_an_unusable_variable_name_is_named(env):
    r = env("BWS_ACCESS_TOKEN=0.t\nnot a var=x\n", "--check")
    assert r.returncode == 1
    assert "not a usable variable name" in r.stderr


def test_a_line_without_an_equals_is_rejected(env):
    r = env("BWS_ACCESS_TOKEN=0.t\ngarbage\n", "--check")
    assert r.returncode == 1
    assert "cannot parse line" in r.stderr


# -- refusals ----------------------------------------------------------------


def test_a_world_readable_config_is_refused_before_fetching(env):
    r = env(CONF, "--check", mode=0o644)
    assert r.returncode == 1
    assert "mode 644" in r.stderr
    assert "600 or" in r.stderr and "400" in r.stderr   # message matches the check
    assert "resolved" not in r.stdout                   # nothing was fetched


def test_read_only_owner_mode_is_accepted(env):
    """0400 is at least as tight as 0600 and the check allows it, so the error
    message has to allow it too — otherwise triage chases a non-problem."""
    assert env(CONF, "--check", mode=0o400).returncode == 0


def test_a_bad_secret_id_fails_naming_it(env):
    r = env(CONF.replace(SMTP_ID, "cccc-nope"), "--check")
    assert r.returncode == 1
    assert "cccc-nope" in r.stderr


def test_a_bws_value_with_no_token_is_caught_up_front(env):
    r = env(CONF.replace("BWS_ACCESS_TOKEN=0.test-token\n", ""), "--check")
    assert r.returncode == 1
    assert "BWS_ACCESS_TOKEN is not set" in r.stderr


def test_a_missing_config_says_where_it_looked(tmp_path):
    r = subprocess.run(
        ["bash", str(WRAPPER), "--check"], capture_output=True, text=True,
        env={**os.environ, "PARALLAX_BITWARDEN_CONF": str(tmp_path / "nope.conf")},
        cwd=ROOT,
    )
    assert r.returncode == 1
    assert "nope.conf" in r.stderr


def test_an_empty_config_is_refused(env):
    r = env("# only comments\n", "--check")
    assert r.returncode == 1
    assert "no variables defined" in r.stderr


# -- handing off to the pipeline --------------------------------------------


def test_secrets_reach_the_child_but_the_token_does_not(env):
    """The pipeline needs the mail password; it has no business holding the
    credential that can fetch every secret in the project."""
    r = env(CONF, python=ECHO_ENV)
    assert r.returncode == 0, r.stderr
    assert "SMTP_PASSWORD=app-pw-from-vault" in r.stdout
    assert "ANTHROPIC=sk-ant-from-vault" in r.stdout
    assert "SMTP_USER=me@example.com" in r.stdout
    assert "BWS_TOKEN=UNSET" in r.stdout


def test_the_child_is_invoked_as_the_daily_module(env):
    r = env(CONF, python=ECHO_ENV)
    assert "ARGS=-m daily" in r.stdout


def test_no_bws_needed_when_nothing_is_fetched(tmp_path):
    """A config of only literal values should not require the CLI at all — worth
    keeping true so someone can adopt the wrapper before setting up a vault."""
    conf = tmp_path / "bitwarden.conf"
    conf.write_text("PARALLAX_SMTP_HOST=smtp.example.com\n")
    conf.chmod(0o600)
    script = tmp_path / "fake-python"
    script.write_text(ECHO_ENV)
    script.chmod(0o755)
    r = subprocess.run(
        ["bash", str(WRAPPER), "--check"], capture_output=True, text=True,
        env={**os.environ, "PARALLAX_BITWARDEN_CONF": str(conf),
             "BWS_BIN": "/nonexistent/bws", "PARALLAX_PYTHON": str(script)},
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert "1 variable(s) resolved" in r.stdout


@pytest.mark.skipif(os.geteuid() != 0,
                    reason="needs root to create a file owned by another user")
def test_an_unreadable_config_names_the_sudo_cause():
    """0600 owned by someone else passes the mode check and then fails the read.

    Constructing it needs two uids: the file stays 0600 — so the mode check is
    satisfied — while the wrapper runs as a user who does not own it. `chmod 000`
    does not stand in for this, because mode 0 trips the mode check first and
    never reaches the branch; an earlier version of this test made exactly that
    mistake and was skipped everywhere it would have failed.
    """
    # Not tmp_path: pytest's ancestors are owner-only, so an unprivileged process
    # cannot even stat through them and the wrapper dies at "no config at" — the
    # earlier check, on a file that does exist. The whole chain has to be
    # traversable for the readability branch to be the one under test.
    home = pathlib.Path(tempfile.mkdtemp(prefix="parallax-wrapper-", dir="/tmp"))
    home.chmod(0o755)
    conf = home / "bitwarden.conf"
    conf.write_text("PARALLAX_SMTP_HOST=smtp.example.com\n")
    conf.chmod(0o600)                       # owned by root, owner-only

    try:
        r = subprocess.run(
            ["bash", str(WRAPPER), "--check"], capture_output=True, text=True,
            env={**os.environ, "PARALLAX_BITWARDEN_CONF": str(conf)}, cwd=ROOT,
            user=UNPRIVILEGED_UID, group=UNPRIVILEGED_UID,
        )
    finally:
        shutil.rmtree(home, ignore_errors=True)
    assert r.returncode == 1
    assert "not readable by" in r.stderr
    assert "sudo chown" in r.stderr
    assert "created with sudo" in r.stderr      # the likely cause, named as likely
    assert "ACL" in r.stderr                    # but not the only one


def test_a_tilde_in_bws_bin_is_expanded(tmp_path, monkeypatch):
    """Values read from a file are plain strings — bash does no tilde expansion on
    them, so `BWS_BIN=~/.local/bin/bws` failed as "not executable" while looking
    perfectly correct. The most likely thing for someone to write."""
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    bws = home / ".local" / "bin" / "bws"
    bws.write_text(FAKE_BWS)
    bws.chmod(0o755)

    conf = tmp_path / "bitwarden.conf"
    conf.write_text(f"BWS_BIN=~/.local/bin/bws\n"
                    f"BWS_ACCESS_TOKEN=0.t\n"
                    f"PARALLAX_SMTP_PASSWORD=bws:{SMTP_ID}\n")
    conf.chmod(0o600)

    r = subprocess.run(
        ["bash", str(WRAPPER), "--check"], capture_output=True, text=True,
        env={**os.environ, "HOME": str(home), "PARALLAX_BITWARDEN_CONF": str(conf)},
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert "1 variable(s) resolved" in r.stdout


def test_a_tilde_in_a_secret_value_is_left_alone(env):
    """Expansion is confined to BWS_BIN. Silently rewriting a password that
    happened to start with ~/ would be a far worse surprise than a path that has
    to be absolute."""
    r = env("PARALLAX_SMTP_PASSWORD=~/not-a-path\n", python=ECHO_ENV)
    assert r.returncode == 0, r.stderr
    assert "SMTP_PASSWORD=~/not-a-path" in r.stdout
