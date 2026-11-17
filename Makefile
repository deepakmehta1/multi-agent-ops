# All Python tooling runs from a project-local virtualenv, so nothing touches the system Python.
VENV := .venv
PY := $(VENV)/bin/python

.DEFAULT_GOAL := help

# `make` with no target lists every target. It greps the `## description` after each one, so
# documenting a new target is just adding that comment.
.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# The virtualenv is a file target that depends on pyproject.toml, so editing the dependencies there
# triggers a reinstall; the `touch` at the end records when it last ran.
$(VENV): pyproject.toml
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	@touch $(VENV)

.PHONY: install
install: $(VENV) ## Create the virtualenv and install everything

.PHONY: run
run: $(VENV) ## Start the API on :8000 with reload
	$(VENV)/bin/uvicorn agentops.main:app --reload --port 8000

.PHONY: demo
demo: $(VENV) ## Scripted end-to-end run, no API key needed
	$(PY) scripts/demo.py --offline

.PHONY: test
test: $(VENV) ## Run the test suite
	$(PY) -m pytest

.PHONY: cov
cov: $(VENV) ## Run the tests with a coverage report
	$(PY) -m pytest --cov=agentops --cov-report=term-missing

.PHONY: lint
lint: $(VENV) ## Lint and type-check
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests
	$(VENV)/bin/mypy

.PHONY: fmt
fmt: $(VENV) ## Format and auto-fix
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

# Close to what CI runs, not identical: CI additionally lints scripts/, parses the console script
# with node, runs the offline demo and builds and boots the Docker image.
.PHONY: check
check: lint test ## Everything CI runs

.PHONY: docker
docker: ## Build and start the container
	docker compose up --build

.PHONY: clean
clean: ## Remove caches and the virtualenv
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
