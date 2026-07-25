.PHONY: daily daily-fast dashboard history test lint validate help
.DEFAULT_GOAL := help

PY ?= python

help:  ## Show these targets
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

daily:  ## Full snapshot: ingest + backfill + cluster + summarize + snapshot + export
	$(PY) -m daily

daily-fast:  ## Today's feeds only — skips the slow GDELT backfill
	$(PY) -m daily --skip backfill

dashboard:  ## Serve the dashboard at http://localhost:8000
	cd dashboard && $(PY) -m http.server

history:  ## Print the recorded snapshot series (JSD over time)
	$(PY) -m compare.history

validate:  ## Score the gold set and report agreement (§5 trigger)
	$(PY) -m validation --scorer ensemble

test:  ## Run the test suite
	$(PY) -m pytest -q

lint:  ## Lint with ruff
	ruff check .
