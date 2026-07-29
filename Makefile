.PHONY: daily daily-fast dashboard history test lint validate help
.DEFAULT_GOAL := help

PY ?= python3

help:  ## Show these targets
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

daily:  ## Full snapshot: ingest -> backfill -> cluster -> summarize -> snapshot -> export (+ digest if enabled)
	$(PY) -m daily

daily-fast:  ## Today's feeds only — skips the slow GDELT backfill
	$(PY) -m daily --skip backfill

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

test:  ## Run the test suite
	$(PY) -m pytest -q

lint:  ## Lint with ruff
	ruff check .
