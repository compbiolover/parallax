# Every target is phony. Three of them — daily, dashboard, digest — share a name
# with a directory in the repo, and without this Make treats that directory as
# the already-built artifact and refuses to run the recipe ("`digest' is up to
# date"). Add new targets here as well as below; tests/test_makefile.py checks.
.PHONY: help daily daily-fast dashboard history validate audit-lexicon \
        digest digest-send podcasts register-probe check-claude check-secrets \
        test lint
.DEFAULT_GOAL := help

PY ?= python3

help:  ## Show these targets
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

daily:  ## Full snapshot: ingest -> backfill -> cluster -> summarize -> snapshot -> export (+ digest if enabled)
	$(PY) -m daily

daily-fast:  ## Today's feeds only — skips the slow GDELT backfill
	$(PY) -m daily --skip backfill

podcasts:  ## Transcribe new podcast episodes (needs parallax[media]; hours, not minutes)
	$(PY) -m ingestion podcasts $(if $(MAX_EPISODES),--max-episodes $(MAX_EPISODES))

dashboard:  ## Serve the dashboard at http://localhost:8000
	cd dashboard && $(PY) -m http.server

history:  ## Print the recorded snapshot series (JSD over time)
	$(PY) -m compare.history

validate:  ## Score the gold set and report agreement (§5 trigger)
	$(PY) -m validation --scorer ensemble

audit-lexicon:  ## Check a lexicon for equality/proportionality asymmetry
	$(PY) -m validation.lexicon_audit $(if $(LEXICON),--lexicon $(LEXICON))

digest:  ## Render the daily brief to data/digest-preview.html (no email needed)
	$(PY) -m digest --dry-run --open

digest-send:  ## Render and email the daily brief (needs SMTP settings)
	$(PY) -m digest

register-probe:  ## Check the liberty rubric scores both registers evenhandedly (costs API calls)
	$(PY) -m validation.register_probe $(if $(REPEATS),--repeats $(REPEATS))

check-claude:  ## Probe every Claude model the configured pipeline calls (a few tokens)
	$(PY) -m scoring.preflight

check-secrets:  ## Resolve every scheduled-run secret and report (fetches nothing else)
	./scripts/parallax-daily.sh --check

test:  ## Run the test suite
	$(PY) -m pytest -q

lint:  ## Lint with ruff
	ruff check .
