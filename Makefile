PYTHON ?= .venv/bin/python

.PHONY: install test lint run eval benchmark build

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -c requirements.lock -e '.[dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check harness tests evals/run_evals.py

run:
	$(PYTHON) -m harness --config config.yaml

benchmark:
	$(PYTHON) evals/run_evals.py --offline

eval:
	$(PYTHON) evals/run_evals.py --live --config config.yaml

build:
	$(PYTHON) -m build
