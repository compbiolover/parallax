"""Every local package a run imports has to be in the wheel.

`packages` listed the six the pipeline is named after and omitted `dashboard`,
`digest` and `validation`. Nothing caught it, because every way the project is
normally run — `make daily`, `python -m daily`, pytest — puts the repo root on
`sys.path`, where an unpackaged directory imports exactly like a packaged one.
The omission only surfaces somewhere the repo root is *not* on the path, which
is every container: `daily.runner` imports `dashboard.export` and
`digest.render` at step time, so the run would ingest and score normally and
then fail on export and digest, at the end of a long job.

So this checks the whole set rather than the three that were missing, since the
next package someone adds is the one nobody will think to check.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

# Packaged for distribution; `tests` ships with the repo, not the wheel.
NOT_SHIPPED = {"tests"}


def _declared() -> set[str]:
    config = tomllib.loads(PYPROJECT.read_text())
    wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]
    return set(wheel["packages"])


def _local_packages() -> set[str]:
    """Top-level directories that are importable Python packages."""
    return {
        path.name for path in ROOT.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    } - NOT_SHIPPED


def test_every_local_package_is_in_the_wheel():
    missing = _local_packages() - _declared()
    assert not missing, (
        f"packages not declared in [tool.hatch.build.targets.wheel]: {sorted(missing)}. "
        f"An unpackaged module imports fine from the repo root and raises ImportError "
        f"once installed."
    )


def test_declared_packages_all_exist():
    """A stale entry would fail the build rather than the run, but name it here."""
    stale = {name for name in _declared() if not (ROOT / name / "__init__.py").is_file()}
    assert not stale, f"declared packages with no __init__.py: {sorted(stale)}"
