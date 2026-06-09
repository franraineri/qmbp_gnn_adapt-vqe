"""Project health analysis tools — diagnosis, coverage scanning, and validation.

Contains:
- diagnose.py: Automated failure root cause analysis
- scan_coverage.py: Coverage scanner + gap analysis + extended analytics
- verify_claims.py: Thesis claim validation against data
- verify_results.py: Pipeline result verification against specs
- validate_s_series.py: S-series experiment validation
- heisenberg_summary.py: Heisenberg XXZ cross-N comparison
- sanity_check.py: Automated sanity checks (physics + data integrity)
- scaling_analyzer.py: MPS scaling law validation (N=40-120)
- scaling_extensions_analyzer.py: E5 extensions (bond-dim, HE, NLCE)
- statistical_tests.py: Shared statistical test utilities
- thesis_findings_validator.py: Corroborate ALL thesis findings with statistics
- thesis_tables_compiler.py: Auto-generate thesis tables (Markdown + LaTeX)
- thesis_figures.py: Generate thesis-level global figures (PDF/PNG)

Usage:
    python -m project_health.analysis.diagnose --all
    python -m project_health.analysis.scan_coverage --discover --extended
    python -m project_health.analysis.verify_claims
    python -m project_health.analysis.verify_results
    python -m project_health.analysis.verify_results --json report.json --tier 1
    python -m project_health.analysis.sanity_check
    python -m project_health.analysis.scaling_analyzer
    python -m project_health.analysis.scaling_extensions_analyzer --verbose --cross-check
    python -m project_health.analysis.thesis_findings_validator --verbose
    python -m project_health.analysis.thesis_tables_compiler --latex tables/
    python -m project_health.analysis.thesis_figures --format pdf
"""
