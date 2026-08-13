"""Every `docker run` in the docs has to be a command that actually parses.

`docker/README.md` and the entrypoint both documented
``--only ingest,backfill``. It exits 2: ``--only`` is ``nargs="+"`` with
``choices=STEPS``, so the comma-joined token is one value and not a valid step.

Ordinarily a wrong example is a documentation bug. This one is not, because the
audience for those lines is a container scheduler: an ECS task definition copied
from the README fails on every invocation, and a retry policy keyed on failure
retries a permanently broken command forever. Worse, exit 2 looks exactly like
exit 1 from a genuinely failed run, so the thing that finally surfaces is a
missing snapshot rather than an error anyone can act on.

Extracts the commands rather than restating them, so the test is about the
documentation rather than a copy of it that can drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from daily.__main__ import _parse

ROOT = Path(__file__).resolve().parent.parent
DOCS = (ROOT / "docker" / "README.md", ROOT / "scripts" / "parallax-entrypoint.sh")

# `docker run [flags] <image> [argv...]` — argv is what reaches `python -m daily`.
_RUN = re.compile(r"docker run\s+(?P<rest>.+)$")
_IMAGE = re.compile(r"^parallax-[a-z]+")


def _documented_argv() -> list[tuple[str, str, list[str]]]:
    """``(file, line, argv)`` for every documented `docker run` of our images."""
    found: list[tuple[str, str, list[str]]] = []
    for doc in DOCS:
        for raw in doc.read_text(encoding="utf-8").splitlines():
            line = raw.lstrip("# ").strip()
            match = _RUN.search(line)
            if not match:
                continue
            tokens = match.group("rest").split()
            # Skip `docker run` flags to find the image, then take the rest.
            for i, token in enumerate(tokens):
                if _IMAGE.match(token):
                    found.append((doc.name, line, tokens[i + 1 :]))
                    break
    return found


def test_the_docs_actually_show_some_commands():
    """Guards the extractor: a regex that matches nothing would pass silently."""
    documented = _documented_argv()
    assert len(documented) >= 2, f"expected to find documented commands, got {documented}"


@pytest.mark.parametrize("doc,line,argv", _documented_argv(), ids=lambda v: str(v)[:60])
def test_documented_command_parses(doc, line, argv):
    try:
        _parse(argv)
    except SystemExit as exc:  # argparse exits rather than raising
        pytest.fail(
            f"{doc} documents a command that exits {exc.code}:\n"
            f"    {line}\n"
            f"  argv after the image: {argv}"
        )


def test_comma_joined_steps_are_still_rejected():
    """The shape that was wrong, pinned — so nobody 'fixes' the parser instead."""
    with pytest.raises(SystemExit) as exc:
        _parse(["--only", "ingest,backfill"])
    assert exc.value.code == 2
