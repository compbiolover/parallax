"""Every Make target has to be declared phony.

`make digest` silently did nothing — it printed "`digest' is up to date" and
exited 0, because `digest/` is a directory and Make treats a target that names
an existing path as already built. `daily/` and `dashboard/` have the same
collision and were already declared; `digest` was the one that got missed.

The failure mode is the bad kind: exit code 0, no error, no output, and the
target it silently skipped is the one that sends you email. So this checks the
whole list rather than the three known collisions, since the next directory
someone adds is the one nobody will think to check.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAKEFILE = ROOT / "Makefile"


def _targets() -> set[str]:
    """Rule targets, i.e. lines like `digest:  ## help text`."""
    return set(re.findall(r"^([a-zA-Z][\w-]*):", MAKEFILE.read_text(), re.MULTILINE))


def _phony() -> set[str]:
    """The .PHONY list, following backslash continuations."""
    text = MAKEFILE.read_text().replace("\\\n", " ")
    declared: set[str] = set()
    for line in text.splitlines():
        if line.startswith(".PHONY:"):
            declared.update(line.removeprefix(".PHONY:").split())
    return declared


def test_every_target_is_phony():
    missing = _targets() - _phony()
    assert not missing, (
        f"Make targets not in .PHONY: {sorted(missing)}. Any target sharing a name "
        f"with a file or directory will silently no-op."
    )


def test_targets_colliding_with_real_paths_are_covered():
    """The subset that actually breaks today, named explicitly so a failure
    points straight at the cause rather than at a list."""
    colliding = {t for t in _targets() if (ROOT / t).exists()}
    assert colliding, "expected daily/, dashboard/ and digest/ to exist"
    assert colliding <= _phony(), (
        f"these targets share a name with a real path and are not phony: "
        f"{sorted(colliding - _phony())}"
    )


def test_phony_does_not_list_targets_that_do_not_exist():
    """A stale .PHONY entry is harmless but means a target was renamed or
    removed and the declaration was left behind."""
    stale = _phony() - _targets()
    assert not stale, f"declared phony but no such target: {sorted(stale)}"
