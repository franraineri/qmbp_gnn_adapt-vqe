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
        clean typecheck coverage health figures \
        maintain maintain-full maintain-fix maintain-all-fix maintain-ci dead-code lint-docs \
        sync-all sync-all-deep diagnose-all

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

freeze:  ## Pin resolved dependency versions to requirements.lock (reproducible installs)
	@echo "# Auto-generated lockfile — DO NOT edit by hand." > requirements.lock
	@echo "# Source of truth for dependencies is pyproject.toml." >> requirements.lock
	@echo "# Regenerate with: make freeze" >> requirements.lock
	@echo "# Reproduce env with: pip install -r requirements.lock" >> requirements.lock
	$(PYTHON) -m pip freeze --exclude-editable >> requirements.lock
	@echo "✅ Frozen to requirements.lock"

# ── Preflight ─────────────────────────────────────────────────

preflight:  ## Run preflight checks on a variant script. Use SCRIPT=path/to/script.py
	$(PYTHON) scripts/preflight.py --from-script $(SCRIPT)

preflight-strict:  ## Preflight with strict mode (warnings = errors). Use SCRIPT=path/to/script.py
	$(PYTHON) scripts/preflight.py --from-script $(SCRIPT) --strict

# ── Full validation ──────────────────────────────────────────

check-full: lint test test-tools smoke-test  ## Run lint + fast tests + tool tests + smoke test
	@echo "✅ Full validation passed"

zoo-validate:  ## Validate model zoo integrity (checksums + manifest)
	@$(PYTHON) -c "from qmbp_simulation.predictors.model_zoo import validate_zoo; r = validate_zoo(); print(f'Model Zoo: {r[\"n_entries\"]} entries, {r[\"n_valid\"]} valid, {r[\"n_missing\"]} missing (gitignored), {r[\"n_corrupted\"]} corrupted'); [print(f'  ❌ {e}') for e in r['errors']]; exit(1 if r['n_corrupted'] > 0 else 0)"

zoo-list:  ## List available pre-trained models in the zoo
	@$(PYTHON) -c "from qmbp_simulation.predictors.model_zoo import list_pretrained; entries = list_pretrained(); print(f'Model Zoo: {len(entries)} checkpoints'); [print(f'  {e.model}/{e.topology} N={e.n_qubits} p={e.p_layers} pass_rate={e.pass_rate:.0%}') for e in entries]"

quality-check:  ## Run VQE quality predictor for common configs
	@$(PYTHON) -c "from qmbp_simulation.analysis.quality_predictor import QualityPredictor; p = QualityPredictor(); configs = [('tfim','chain_1d',10,2),('tfim','heavy_hex',10,2),('tfim_longitudinal','chain_1d',10,2),('heisenberg','chain_1d',10,2)]; [print(p.predict(model=m,topology=t,n_qubits=n,p_layers=pl)) for m,t,n,pl in configs]"

check-all: lint test-full smoke-test  ## Run lint + ALL tests (including slow) + smoke test
	@echo "✅ Complete validation passed (including slow tests)"


# ── Data Sync & Diagnosis ────────────────────────────────────

sync-all:  ## Full project data/metrics sync (all stores: GT, dashboard, scoreboard, zoo, critical ranking, fidelities) — auto-corrects when possible
	@echo "═══ 1/3  Auto-fix stale e_exact in NPZ from GT cache (creates .bak backups) ═══"
	$(PYTHON) -c "from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence as v; r=v(fix=True); print('GT↔NPZ fix:', r['summary'])" || true
	@echo "\n═══ 2/3  post_experiment_sync (GT → dashboard → scoreboard → zoo pass_rate/by_n/critical_ranking → coverage → ResultIndex) ═══"
	$(PYTHON) -c "from qmbp_simulation.analysis.metrics import post_experiment_sync; post_experiment_sync(verbose=True)" || true
	@echo "\n═══ 3/3  Fill missing exact fidelities (N<=16, cached — not covered by post_experiment_sync) ═══"
	$(PYTHON) -c "from qmbp_simulation.predictors.model_zoo import backfill_missing_fidelities as b; print('fidelity backfill:', b())" || true
	@echo "\n✅ Full sync complete (auto-fix + metric consolidation + fidelity fill)"

sync-all-deep:  ## Deep sync: re-evaluate ENERGY (MPS |dE|) of every zoo model, then full sync (slow)
	@echo "═══ 0/5  Deep zoo re-evaluation with energy (MPS |dE|, updates pass_rate) ═══"
	@echo "    (this is the slow step — evaluates every multi-N model's energy, no 120s cap)"
	$(PYTHON) scripts/analysis/evaluate_zoo_models.py --update-zoo --energy-eval
	@echo "\n═══ 1-4/5  Full consolidation (post_experiment_sync + backfills) ═══"
	$(MAKE) sync-all
	@echo "\n✅ Deep sync complete (fresh energy eval + full consolidation)"

diagnose-all:  ## Full project diagnosis (data stores + consistency + GT/NPZ coherence + critical-ranking drift)
	@echo "═══ 1/4  Data stores inventory (GT, NPZ, zoo, dashboard) ═══"
	$(PYTHON) scripts/maintenance/inspect_data_stores.py --validate-dashboard || true
	@echo "\n═══ 2/4  Cross-store consistency (zoo ↔ comparison ↔ dashboard ↔ registry) ═══"
	$(PYTHON) scripts/maintenance/query_model_registry.py consistency || true
	@echo "\n═══ 3/4  GT ↔ NPZ coherence (stale e_exact detection) ═══"
	$(PYTHON) -c "from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence; print(validate_gt_npz_coherence()['summary'])" || true
	@echo "\n═══ 4/4  Critical-ranking drift (pass_rate vs empirical grade near h_c) ═══"
	$(PYTHON) -c "from qmbp_simulation.analysis.metrics import validate_data_consistency as v; issues=v().get('critical_ranking_issues',[]); print(f'{len(issues)} drift issue(s)'); [print(' DRIFT:', i['checkpoint'][:45], '->', i['issue'][:80]) for i in issues]" || true
	@echo "\n✅ Full diagnosis complete"

# ── Maintenance Checks ───────────────────────────────────────

maintain-fix:  ## Run maintenance with auto-fix (clean caches, fix steerings)
	@$(PYTHON) scripts/general_project_maintenance/run_all_checks.py --fix

maintain-all-fix:  ## Run ALL general_project_maintenance scripts with --fix where supported
	@echo "═══ Running all maintenance scripts (--fix mode) ═══"
	@echo "\n── run_all_checks.py --fix ──"
	$(PYTHON) scripts/general_project_maintenance/run_all_checks.py --fix || true
	@echo "\n── validate_test_imports.py --fix ──"
	$(PYTHON) scripts/general_project_maintenance/validate_test_imports.py --fix || true
	@echo "\n── verify_steerings.py --fix ──"
	$(PYTHON) scripts/general_project_maintenance/verify_steerings.py --fix || true
	@echo "\n── check_phantom_functions.py ──"
	$(PYTHON) scripts/general_project_maintenance/check_phantom_functions.py || true
	@echo "\n── cleanup_repo.py --execute ──"
	$(PYTHON) scripts/general_project_maintenance/cleanup_repo.py --execute || true
	@echo "\n── trim_overdocumented.py ──"
	$(PYTHON) scripts/general_project_maintenance/trim_overdocumented.py --apply || true
	@echo "\n── md_index.py ──"
	$(PYTHON) scripts/general_project_maintenance/md_index.py || true
	$(PYTHON) scripts/general_project_maintenance/generate_thesis_tables.py
	@.venv/bin/vulture src/qmbp_simulation vulture_whitelist.py --min-confidence 80 --exclude "_deprecated,.venv" || true
	@echo "\n✅ All maintenance scripts completed"


lint-docs:  ## Check docstring/signature consistency with pydoclint
	@.venv/bin/pydoclint --style=numpy --check-return-types=false --allow-init-docstring=true --skip-checking-short-docstrings=true --quiet src/qmbp_simulation


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
	@cp -f documentation/thesis_figures/fig_*.pdf tesis-figures/ 2>/dev/null || true

# ── Thesis Compilation ───────────────────────────────────────

validate-findings:  ## Validate all thesis findings against raw data
	$(PYTHON) -m project_health.analysis.thesis_findings_validator --verbose

# ── Hardware Deployment ──────────────────────────────────────

hw-cost:  ## Estimate QPU cost. Use N=10 H=3 PROFILE=kingston SPSA=default AMPLIFIER=pea
	$(PYTHON) scripts/hardware.py cost --n-qubits $(or $(N),10) --h-points $(or $(H),3) \
		--profile $(or $(PROFILE),kingston) --spsa $(or $(SPSA),default) \
		--amplifier $(or $(AMPLIFIER),pea)

hw-preflight:  ## Run preflight checks on FakeTorino. Use N=10
	$(PYTHON) scripts/hardware.py preflight --n-qubits $(or $(N),10)

hw-rehearsal:  ## Run full hardware rehearsal. Use ARGS for extra flags.
	$(PYTHON) scripts/hardware.py rehearsal $(ARGS)

hw-rehearsal-quick:  ## Rehearsal sections 8+9 only (cost + circuit audit, ~2s)
	$(PYTHON) scripts/hardware.py rehearsal --section 8 9

hw-analyze:  ## Analyze latest rehearsal results (GO/NO-GO)
	$(PYTHON) scripts/hardware.py analyze

hw-analyze-all:  ## Analyze all rehearsal runs with cross-comparison
	$(PYTHON) scripts/hardware.py analyze --all

hw-deploy-dry:  ## Dry-run deployment (preflight + cost only, no QPU)
	$(PYTHON) scripts/experiment_runners/hardware/run_ibm_deployment.py --dry-run

hw-deploy-calibrate:  ## Session 1: calibration run (Tier 0 only, measures T_one_job)
	$(PYTHON) scripts/experiment_runners/hardware/run_ibm_deployment.py --tier 0

hw-deploy:  ## Session 2: full deployment (Tier 0→3, auto-advancing, no SPSA)
	$(PYTHON) scripts/experiment_runners/hardware/run_ibm_deployment.py --no-spsa

# ── Mitigation Benchmark ─────────────────────────────────────

mitigation-bench-p0:  ## Run mitigation benchmark P0 (baseline + GF + PEA)
	@find . -path "*/__pycache__/run_mitigation*" -delete 2>/dev/null || true
	@find . -path "*/__pycache__/benchmark_configs*" -delete 2>/dev/null || true
	$(PYTHON) scripts/experiment_runners/hardware/run_mitigation_benchmark.py --priority P0 --h-values 3.25,3.5,4.0

mitigation-bench-p1:  ## Run mitigation benchmark P1 (ablation configs)
	$(PYTHON) scripts/experiment_runners/hardware/run_mitigation_benchmark.py --priority P1 --h-values 3.25,3.5,4.0

mitigation-bench-all:  ## Run ALL mitigation benchmark configs (P0-P3)
	@find . -path "*/__pycache__/run_mitigation*" -delete 2>/dev/null || true
	$(PYTHON) scripts/experiment_runners/hardware/run_mitigation_benchmark.py --h-values 3.25,3.5,4.0

mitigation-analyze:  ## Analyze mitigation benchmark results (thesis table + figures)
	$(PYTHON) -m project_health.analysis.mitigation_benchmark_analyzer --thesis-table --figures

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

# ── Flow Warmstart Pipeline ──────────────────────────────────

hw-flow-rehearsal:  ## Run V3 rehearsal with flow warmstart (generates σ_flow data)
	$(PYTHON) scripts/experiment_runners/run_hardware_rehearsal_v3.py \
		--topology heavy_hex --n-qubits 10 --p-layers 1 \
		--h-test 4.0 3.5 3.25 3.0 \
		--use-flow-warmstart --verbose

hw-flow-rehearsal-chain:  ## Run V3 rehearsal with flow + bond-resolved (chain_1d N=6 p=2)
	$(PYTHON) scripts/experiment_runners/run_hardware_rehearsal_v3.py \
		--topology chain_1d --n-qubits 6 --p-layers 2 \
		--use-flow-warmstart --use-bond-resolved --verbose

hw-flow-analyze:  ## Analyze flow warmstart results
	$(PYTHON) -m project_health.analysis.flow_warmstart_analyzer --verbose

hw-flow-deploy-dry:  ## Dry-run deployment with σ_flow safety net from latest rehearsal
	@LATEST=$$(ls -t results/experiments/exp_hw_rehearsal_v3/run_*.json 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then echo "No rehearsal results found. Run make hw-flow-rehearsal first."; exit 1; fi; \
	echo "Using σ_flow from: $$LATEST"; \
	$(PYTHON) scripts/experiment_runners/hardware/run_ibm_deployment.py \
		--dry-run --sigma-flow-results "$$LATEST" --verbose

hw-flow-deploy:  ## Full deployment with σ_flow safety net from latest rehearsal
	@LATEST=$$(ls -t results/experiments/exp_hw_rehearsal_v3/run_*.json 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then echo "No rehearsal results found. Run make hw-flow-rehearsal first."; exit 1; fi; \
	echo "Using σ_flow from: $$LATEST"; \
	$(PYTHON) scripts/experiment_runners/hardware/run_ibm_deployment.py \
		--sigma-flow-results "$$LATEST" --no-spsa

hw-flow-full:  ## Full pipeline: rehearsal → analyze → deploy (dry-run)
	@echo "═══ Step 1: V3 Rehearsal with Flow Warmstart ═══"
	$(MAKE) hw-flow-rehearsal
	@echo "\n═══ Step 2: Analyze Flow Results ═══"
	$(MAKE) hw-flow-analyze
	@echo "\n═══ Step 3: Deployment Dry-Run with σ_flow ═══"
	$(MAKE) hw-flow-deploy-dry
	@echo "\n✅ Full flow pipeline complete. Review results, then run: make hw-flow-deploy"

hw-flow-from-checkpoint:  ## Deploy using saved flow checkpoint (skip re-training)
	@CKPT=$$(ls -t results/flow_checkpoints/flow_heavy_hex_N10_p1.pt 2>/dev/null | head -1); \
	LATEST=$$(ls -t results/experiments/exp_hw_rehearsal_v3/run_*.json 2>/dev/null | head -1); \
	if [ -z "$$CKPT" ]; then echo "No flow checkpoint found. Run make hw-flow-rehearsal first."; exit 1; fi; \
	if [ -z "$$LATEST" ]; then echo "No rehearsal results found."; exit 1; fi; \
	echo "Using checkpoint: $$CKPT"; \
	echo "Using σ_flow from: $$LATEST"; \
	$(PYTHON) scripts/experiment_runners/hardware/run_ibm_deployment.py \
		--dry-run --sigma-flow-results "$$LATEST" --flow-checkpoint "$$CKPT" --verbose
