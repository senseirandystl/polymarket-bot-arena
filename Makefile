# Polymarket Bot Arena — test & quality targets.
# Always runs through the project venv (same interpreter as the launchd services).

PY := .venv/bin/python3

.PHONY: test test-unit test-integration coverage typecheck

test:            ## full suite (unit + integration)
	$(PY) -m pytest

test-unit:       ## fast isolated unit tests only
	$(PY) -m pytest -m unit

test-integration:## cross-module flows (dashboard, paper cycle, pipelines)
	$(PY) -m pytest -m integration

coverage:        ## full suite with line coverage report
	$(PY) -m pytest --cov=. --cov-report=term-missing

typecheck:       ## mypy baseline (see mypy.ini)
	$(PY) -m mypy .
