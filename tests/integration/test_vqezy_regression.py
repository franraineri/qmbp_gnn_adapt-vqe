"""VQEzy External Benchmark — Regression test.

Runs a small subset of VQEzy TFI instances through the pipeline to verify
that refactorings don't break MPNN generalization. Uses 5 instances from
the valid regime (h >= 1.0, J ≈ 1.0) with a lightweight VQE configuration.

Prerequisites:
    VQEzy dataset must be available at data/VQEzy/ (git submodule).

Marks:
    - integration: requires full pipeline + torch + VQEzy data
    - slow: takes ~60-90s due to VQE optimizations
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VQEZY_DATA = PROJECT_ROOT / "data" / "VQEzy" / "qmanybody" / "ti_8_qubit.h5"

# Skip if VQEzy data not cloned
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not VQEZY_DATA.exists(),
        reason="VQEzy dataset not available (clone data/VQEzy submodule)",
    ),
]


# Regression thresholds — if any of these fail, a refactoring broke something
MAX_MEAN_DE_GAP = 0.15  # Mean ΔE/gap across 5 instances (generous for p=1 N=8)
MIN_VQE_BEATS_VQEZY = 0.4  # At least 40% of instances our VQE ≤ VQEzy energy


@pytest.fixture(scope="module")
def vqezy_results():
    """Run the VQEzy benchmark on 5 instances and cache the result."""
    from qmbp_simulation.predictors.external_benchmarks import (
        VQEzyBenchmarkEvaluator,
        load_vqezy_tfi,
    )

    dataset = load_vqezy_tfi(
        VQEZY_DATA,
        h_min=1.0,
        h_max=3.0,
        j_min=0.8,
        j_max=1.2,
        max_instances=5,
    )
    assert len(dataset) >= 3, f"Not enough VQEzy instances found: {len(dataset)}"

    evaluator = VQEzyBenchmarkEvaluator(
        n_qubits=8,
        p_layers=1,
        topology="square",
        n_restarts=2,
        maxiter=100,
        seed=42,
    )

    results = evaluator.evaluate(dataset, mode="sweep")
    return results


def test_vqezy_mean_de_gap(vqezy_results):
    """Mean ΔE/gap should stay below regression threshold."""
    mean_de_gap = vqezy_results.mean_de_gap
    logger.info(f"VQEzy regression: mean_de_gap = {mean_de_gap:.4f}")
    assert mean_de_gap < MAX_MEAN_DE_GAP, (
        f"MPNN regression: mean ΔE/gap = {mean_de_gap:.4f} > {MAX_MEAN_DE_GAP}. "
        f"A recent change likely broke VQE optimization or graph construction."
    )


def test_vqezy_vqe_competitive(vqezy_results):
    """Our VQE should beat or match VQEzy in a reasonable fraction of cases."""
    n_total = len(vqezy_results.instance_results)
    n_beats = sum(
        1
        for r in vqezy_results.instance_results
        if r.our_energy <= r.vqezy_energy + 1e-6
    )
    beat_rate = n_beats / n_total if n_total > 0 else 0.0
    logger.info(
        f"VQEzy regression: VQE competitive rate = {beat_rate:.1%} "
        f"({n_beats}/{n_total})"
    )
    assert beat_rate >= MIN_VQE_BEATS_VQEZY, (
        f"MPNN regression: only {beat_rate:.1%} of instances beat VQEzy "
        f"(threshold: {MIN_VQE_BEATS_VQEZY:.0%}). Check VQE warm-start logic."
    )


def test_vqezy_no_nan_energies(vqezy_results):
    """No NaN or infinite energies should appear in results."""
    import numpy as np

    for r in vqezy_results.instance_results:
        assert np.isfinite(r.our_energy), (
            f"NaN/Inf energy at h={r.h_value:.2f}: {r.our_energy}"
        )


def test_vqezy_rescale_h_by_j(vqezy_results):
    """Verify h/J rescaling doesn't crash (no accuracy requirement here)."""
    from qmbp_simulation.predictors.external_benchmarks import (
        VQEzyBenchmarkEvaluator,
        load_vqezy_tfi,
    )

    dataset = load_vqezy_tfi(
        VQEZY_DATA,
        h_min=1.5,
        h_max=2.5,
        max_instances=3,
    )
    if len(dataset) < 2:
        pytest.skip("Not enough instances for rescale test")

    evaluator = VQEzyBenchmarkEvaluator(
        n_qubits=8,
        p_layers=1,
        topology="square",
        n_restarts=1,
        maxiter=50,
        seed=42,
    )
    # Should not raise
    results = evaluator.evaluate(dataset, mode="sweep", rescale_h_by_j=True)
    assert len(results.instance_results) > 0
