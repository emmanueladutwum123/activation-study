# The whole study, start to finish:  make setup && make data && make all
.PHONY: setup data study test lint all clean

PY := .venv/bin/python

setup:                     ## create the venv and install pinned dependencies
	python3 -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt

data:                      ## download MovieLens 10M and convert it to parquet (~2 min)
	PATH=".venv/bin:$$PATH" bash scripts/fetch_data.sh

study: ## re-run the analysis: results/findings.json and figures/
	PYTHONPATH=src $(PY) analysis/run_study.py

test:                      ## unit tests (no dataset required -- they use fixtures)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src analysis tests

all: lint test study

clean:                     ## drop derived data; `make data` rebuilds it
	rm -rf data/raw data/processed .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
