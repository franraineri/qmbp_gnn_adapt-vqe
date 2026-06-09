# GNN-HVA — Project Makefile
#
# This is the SINGLE entry point for all project operations.
# Run `make help` to see all available targets.
#
# Structure:
#   scripts/              — All executable scripts
#   tests/                — Pytest test suite
#   src/qmbp_simulation/  — Source modules (do NOT put scripts here)
#   project_health/       — Analysis, diagnosis, and figure tools

PYTHON := .venv/bin/python
VENV := source .venv/bin/activate

.PHONY: help lint format test smoke-test benchmark check check-full \
        hooks-install strip-notebooks freeze run-notebooks run-nb-12 run-nb-34 \
        clean typecheck coverage health figures

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Code quality ─────────────────────────────────────────────

lint:  ## Run ruff linter
	.venv/bin/ruff check src/ tests/ scripts/ project_health/ --exclude "scripts/hooks/*"

format:  ## Auto-format with ruff
	.venv/bin/ruff format src/ tests/ scripts/ project_health/

# ── Testing ──────────────────────────────────────────────────

test:  ## Run fast tests only, excluding slow (FakeTorino) tests (~8s)
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not slow"

test-full:  ## Run ALL tests including slow FakeTorino tests (~60s)
	$(PYTHON) -m pytest tests/ -v --tb=short

test-tools:  ## Run analysis/digest/compare tool tests (~35s)
	$(PYTHON) -m pytest tests/test_diagnose.py tests/test_compare.py tests/test_analysis_tools.py tests/test_project_health.py -v --tb=short

test-slow:  ## Run only slow tests (CLI integration, VQE sweep) (~90s)
	$(PYTHON) -m pytest tests/ -v --tb=short -m "slow"

test-models:  ## Run model-specific tests (TFIM longitudinal + frustrated) (~1s)
	$(PYTHON) -m pytest tests/unit/test_tfim_longitudinal.py tests/unit/test_frustrated_tfim.py -v --tb=short

smoke-test:  ## Run end-to-end smoke test (~7s)
	$(PYTHON) tests/smoke_test.py

# ── Benchmarking ─────────────────────────────────────────────

benchmark:  ## Run benchmark (3 runs, N=6). Use ARGS for options.
	$(PYTHON) scripts/benchmark.py $(ARGS)

benchmark-n10:  ## Run benchmark with N=10 chain
	$(PYTHON) scripts/benchmark.py --n-qubits 10 --n-restarts 5 --mpnn-epochs 6000 --h-test 1.5 $(ARGS)

# ── Notebooks ────────────────────────────────────────────────

run-notebooks:  ## Execute both PoC notebooks with validation
	$(PYTHON) scripts/run_notebooks.py --phase all

run-nb-12:  ## Execute Phase 1-2 notebook only
	$(PYTHON) scripts/run_notebooks.py --phase 1-2

run-nb-34:  ## Execute Phase 3-4 notebook only
	$(PYTHON) scripts/run_notebooks.py --phase 3-4

# ── Pre-commit ───────────────────────────────────────────────

hooks-install:  ## Install pre-commit hooks (including commit-msg) and pre-push
	pre-commit install
	pre-commit install --hook-type commit-msg
	cp scripts/hooks/pre-push-tests.sh .git/hooks/pre-push
	chmod +x .git/hooks/pre-push
	@echo "✅ All hooks installed (pre-commit + commit-msg + pre-push)"

check:  ## Run all pre-commit hooks
	pre-commit run --all-files

# ── Notebook hygiene ─────────────────────────────────────────

strip-notebooks:  ## Strip outputs from tracked notebooks
	git ls-files '*.ipynb' | xargs -I{} nbstripout {}

# ── Reproducibility ──────────────────────────────────────────

freeze:  ## Pin dependency versions to requirements.lock
	pip freeze > requirements.lock
	@echo "✅ Frozen to requirements.lock"

# ── Preflight ─────────────────────────────────────────────────

preflight:  ## Run preflight checks on a variant script. Use SCRIPT=path/to/script.py
	$(PYTHON) scripts/preflight.py --from-script $(SCRIPT)

preflight-strict:  ## Preflight with strict mode (warnings = errors). Use SCRIPT=path/to/script.py
	$(PYTHON) scripts/preflight.py --from-script $(SCRIPT) --strict

# ── Full validation ──────────────────────────────────────────

check-full: lint test test-tools smoke-test  ## Run lint + fast tests + tool tests + smoke test
	@echo "✅ Full validation passed"

check-all: lint test-full smoke-test  ## Run lint + ALL tests (including slow) + smoke test
	@echo "✅ Complete validation passed (including slow tests)"


# ── Cleaning ─────────────────────────────────────────────────

clean:  ## Remove caches and build artifacts
	rm -rf .hypothesis/ .ruff_cache/ .pytest_cache/ __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "✅ Caches cleaned"

# ── Type checking ────────────────────────────────────────────

typecheck:  ## Run mypy type checking
	$(PYTHON) -m mypy src/ --ignore-missing-imports

# ── Coverage ─────────────────────────────────────────────────

coverage:  ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -m "not slow" --cov=src/qmbp_simulation --cov-report=term-missing --cov-report=html:htmlcov

# ── Project health ───────────────────────────────────────────

health:  ## Run project health report (compact)
	$(PYTHON) -m project_health --compact

health-full:  ## Run full project health report (markdown, saved)
	$(PYTHON) -m project_health --markdown -o reports/

sanity:  ## Run sanity checks (physics + data integrity)
	$(PYTHON) -m project_health.analysis.sanity_check

scaling:  ## Analyze MPS scaling results (N=40-120)
	$(PYTHON) -m project_health.analysis.scaling_analyzer

extensions:  ## Analyze E5 scaling extensions (bond-dim, HE, NLCE)
	$(PYTHON) -m project_health.analysis.scaling_extensions_analyzer --verbose --cross-check

cross-topology:  ## Analyze cross-topology transfer results
	$(PYTHON) -m project_health.digest --kind cross_topology

# ── Figures ──────────────────────────────────────────────────

figures:  ## Generate all analysis figures (PNG, default theme)
	$(PYTHON) -m project_health.figures --source both

figures-thesis:  ## Generate thesis-quality figures (PDF, 300dpi, no titles)
	$(PYTHON) -m project_health.figures --source both --theme thesis --format pdf --dpi 300 --no-titles --output-dir documentation/thesis_figures/
	$(PYTHON) -m project_health.analysis.thesis_figures --format pdf --dpi 300 --verbose

# ── Thesis Compilation ───────────────────────────────────────

validate-findings:  ## Validate all thesis findings against raw data
	$(PYTHON) -m project_health.analysis.thesis_findings_validator --verbose

validate-findings-latex:  ## Validate findings + generate LaTeX table
	$(PYTHON) -m project_health.analysis.thesis_findings_validator --verbose --latex documentation/thesis_tables/findings_validation.tex

thesis-tables:  ## Compile all thesis tables (Markdown + LaTeX)
	$(PYTHON) -m project_health.analysis.thesis_tables_compiler --verbose --markdown documentation/thesis_tables/all_tables.md --latex documentation/thesis_tables/

thesis-figures:  ## Generate thesis-level global figures (PDF)
	$(PYTHON) -m project_health.analysis.thesis_figures --format pdf --dpi 300 --verbose

thesis-all:  ## Full thesis compilation: validate + tables + figures
	@echo "═══ Validating Findings ═══"
	$(PYTHON) -m project_health.analysis.thesis_findings_validator --verbose --json documentation/thesis_tables/findings_report.json || true
	@echo "\n═══ Compiling Tables ═══"
	$(PYTHON) -m project_health.analysis.thesis_tables_compiler --verbose --markdown documentation/thesis_tables/all_tables.md --latex documentation/thesis_tables/
	@echo "\n═══ Generating Global Figures ═══"
	$(PYTHON) -m project_health.analysis.thesis_figures --format pdf --dpi 300 --verbose
	@echo "\n═══ Generating Registry Figures ═══"
	$(PYTHON) -m project_health.figures --source both --theme thesis --format pdf --dpi 300 --no-titles --output-dir documentation/thesis_figures/
	@echo "\n✅ Thesis compilation complete → documentation/thesis_tables/ + documentation/thesis_figures/"
