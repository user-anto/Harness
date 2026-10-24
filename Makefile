PYTHON ?= .venv/bin/python

.PHONY: install test lint run eval benchmark security-eval build commit

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
	$(PYTHON) evals/run_evals.py --offline --suite all

security-eval:
	$(PYTHON) evals/run_evals.py --offline --suite security

eval:
	$(PYTHON) evals/run_evals.py --live --config config.yaml

build:
	$(PYTHON) -m build

commit:
	git add .
	git commit -m "$(m)"
	git push origin main
