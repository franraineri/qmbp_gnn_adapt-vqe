# GNN-HVA v6.0 — Project Makefile
#
# This is the SINGLE entry point for all project operations.
# Run `make help` to see all available targets.
#
# Structure:
#   scripts/              — All executable scripts
#     smoke_test.py       — Quick end-to-end validation (~7s)
#     benchmark_v6.py     — Multi-run benchmark with configurable params
#     run_notebooks.py    — Notebook executor with validation
#     hooks/              — Pre-commit hook scripts
#   tests/                — Pytest test suite
#   src/poc/v6/           — Source modules (do NOT put scripts here)

PYTHON := python
VENV := source .venv/bin/activate

.PHONY: help lint format test smoke-test benchmark check check-full \
        hooks-install strip-notebooks freeze run-notebooks run-nb-12 run-nb-34

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Code quality ─────────────────────────────────────────────

lint:  ## Run ruff linter
	ruff check src/ tests/ scripts/ --exclude "scripts/hooks/*"

format:  ## Auto-format with ruff
	ruff format src/ tests/ scripts/

# ── Testing ──────────────────────────────────────────────────

test:  ## Run fast tests only, excluding slow (FakeTorino) tests (~8s)
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not slow"

test-full:  ## Run ALL tests including slow FakeTorino tests (~60s)
	$(PYTHON) -m pytest tests/ -v --tb=short

smoke-test:  ## Run end-to-end smoke test (~7s)
	$(PYTHON) scripts/smoke_test.py

# ── Benchmarking ─────────────────────────────────────────────

benchmark:  ## Run benchmark (3 runs, N=6). Use ARGS for options.
	$(PYTHON) scripts/benchmark_v6.py $(ARGS)

benchmark-n10:  ## Run benchmark with N=10 chain
	$(PYTHON) scripts/benchmark_v6.py --n-qubits 10 --n-restarts 5 --mpnn-epochs 6000 --h-test 1.5 $(ARGS)

# ── Notebooks ────────────────────────────────────────────────

run-notebooks:  ## Execute both PoC notebooks with validation
	$(PYTHON) scripts/run_notebooks.py --phase all

run-nb-12:  ## Execute Phase 1-2 notebook only
	$(PYTHON) scripts/run_notebooks.py --phase 1-2

run-nb-34:  ## Execute Phase 3-4 notebook only
	$(PYTHON) scripts/run_notebooks.py --phase 3-4

# ── Pre-commit ───────────────────────────────────────────────

hooks-install:  ## Install pre-commit hooks (including commit-msg)
	pre-commit install
	pre-commit install --hook-type commit-msg

check:  ## Run all pre-commit hooks
	pre-commit run --all-files

# ── Notebook hygiene ─────────────────────────────────────────

strip-notebooks:  ## Strip outputs from tracked notebooks
	git ls-files '*.ipynb' | xargs -I{} nbstripout {}

# ── Reproducibility ──────────────────────────────────────────

freeze:  ## Pin dependency versions to requirements.lock
	pip freeze > requirements.lock
	@echo "✅ Frozen to requirements.lock"

# ── Full validation ──────────────────────────────────────────

check-full: lint test smoke-test  ## Run lint + tests + smoke test
	@echo "✅ Full validation passed"
