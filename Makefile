# Polymarket Bot Arena — test & quality targets.
# Always runs through the project venv (same interpreter as the launchd services).

PY := .venv/bin/python3

.PHONY: test test-unit test-integration coverage typecheck \
	docker-up docker-down docker-logs docker-ps docker-build

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

# --- Docker (24/7 stack; see docs/docker.md) ---

docker-build:    ## build arena+dashboard image
	docker compose build

docker-up:       ## build + start arena + dashboard detached
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example — set DASHBOARD_PASS before public expose")
	docker compose up -d --build

docker-down:     ## stop containers (keeps ./data)
	docker compose down

docker-logs:     ## follow arena + dashboard logs
	docker compose logs -f

docker-ps:       ## status + health
	docker compose ps
