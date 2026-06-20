"""Project health analysis tools — organized by domain.

Subpackages:
- validation/   — Verification, sanity checks, audit, findings validation
- hardware/     — Hardware rehearsal, mitigation, transpilation, Mitiq, layout
- scaling/      — MPS scaling law, extensions, flow warmstart
- models/       — GNN-QEM, MPNN eval, AQC-Tensor analyzers
- thesis/       — Thesis tables, figures, Heisenberg summary
- coverage/     — Coverage scanning and gap analysis

Root modules (shared utilities):
- diagnose.py          — Automated failure root cause analysis
- statistical_tests.py — Shared statistical test utilities

Usage:
    python -m project_health.analysis.validation.sanity_check
    python -m project_health.analysis.scaling.scaling_analyzer
    python -m project_health.analysis.hardware.mitigation_benchmark_analyzer
    python -m project_health.analysis.models.aqc_tensor_analyzer
    python -m project_health.analysis.thesis.thesis_figures
    python -m project_health.analysis.coverage.scan_coverage
"""
