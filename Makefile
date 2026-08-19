export PATH := $(PWD)/.venv/bin:$(PATH)

.PHONY: test eval clean run commit install plot

test:
	pip install -e .
	python3 evals/run_evals.py --dry-run

eval:
	pip install -e .
	rm evals/results.csv
	python3 evals/run_evals.py
	$(MAKE) plot

clean:
	rm -rf evals/eval_env/*
	rmdir scratch 2>/dev/null || true

run:
	pip install -e .
	python3 -m src.graph

commit: clean
	git add .
	git commit -m "$(m)"
	git push -u origin main

plot:
	python3 evals/spider.py

install:
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment in .venv..."; \
		python3 -m venv .venv; \
	fi
	@echo "Installing dependencies..."
	@.venv/bin/pip install -e .
	@mkdir -p ~/.local/bin
	@echo '#!/bin/bash' > ~/.local/bin/harness
	@echo 'export PYTHONPATH="$(PWD):$$PYTHONPATH"' >> ~/.local/bin/harness
	@echo 'if [ -f "$(PWD)/.venv/bin/python" ]; then' >> ~/.local/bin/harness
	@echo '    exec "$(PWD)/.venv/bin/python" -m src.graph "$$@"' >> ~/.local/bin/harness
	@echo 'else' >> ~/.local/bin/harness
	@echo '    exec python3 -m src.graph "$$@"' >> ~/.local/bin/harness
	@echo 'fi' >> ~/.local/bin/harness
	@chmod +x ~/.local/bin/harness
	@echo "'harness' command installed to ~/.local/bin/harness"
	@command -v llama-server >/dev/null 2>&1 || echo "WARNING: 'llama-server' not found in PATH."
	@command -v ollama >/dev/null 2>&1 || echo "WARNING: 'ollama' not found in PATH."