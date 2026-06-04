"""Project health analysis tools — diagnosis, coverage scanning, and validation.

Contains:
- diagnose.py: Automated failure root cause analysis
- scan_coverage.py: Coverage scanner + gap analysis + extended analytics
- verify_claims.py: Thesis claim validation against data
- verify_results.py: Pipeline result verification against specs
- validate_s_series.py: S-series experiment validation
- heisenberg_summary.py: Heisenberg XXZ cross-N comparison

Usage:
    python -m project_health.analysis.diagnose --all
    python -m project_health.analysis.scan_coverage --discover --extended
    python -m project_health.analysis.verify_claims
    python -m project_health.analysis.verify_results
    python -m project_health.analysis.verify_results --json report.json --tier 1
"""
