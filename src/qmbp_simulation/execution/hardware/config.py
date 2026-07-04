"""Configuration dataclasses for hardware execution backend.

Defines HardwareConfig (complete execution settings), SPSAConfig (validated
SPSA parameters from V7-4A grid search), and HardwareRunResult (per-h-point
result container with full provenance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from qmbp_simulation.execution.backends import MitigationOptions


@dataclass
class HardwareConfig:
    """Complete configuration for hardware execution.

    Single source of truth — includes mitigation options as an integrated
    field to avoid ambiguity between overlapping configs.
    """

    backend_name: str = "ibm_kingston"
    mode: Literal["hardware", "fake_backend"] = "hardware"
    n_qubits: int = 10
    shots: int = 16384
    n_layouts: int = 3
    n_candidates: int = 40
    max_ces: float = 0.5
    optimization_level: int = 2
    layout_seed: int = 42
    job_timeout_s: int | None = 600
    max_retries: int = 3
    retry_delay_s: int = 30
    max_total_shots: int = 10_000_000
    spsa_enabled: bool = True
    spsa_threshold: float = 0.05
    output_dir: str = "results/hardware"
    # Layout optimizer settings (mapomatic VF2 integration)
    use_mapomatic: bool = True
    layout_max_2q_error: float = 0.01
    layout_min_t1_us: float = 50.0
    layout_call_limit: int = 100_000
    layout_exclude_qubits: list[int] = field(default_factory=list)
    layout_strategy: Literal["lowest_cost", "ces_spread", "hybrid"] = "lowest_cost"
    # Known SWAP-free layout for heavy_hex N=10 on ibm_kingston (verified 2026-06-23).
    # Used as VF2 fallback when mapomatic times out. All 9 logical edges map
    # directly to physical CZ gates (0 SWAPs, 9 CZ total).
    # Qubits: [22,23,24,25,26,27,28,16,37,17] on Kingston heavy-hex row 2.
    fallback_layout_kingston: list[int] = field(
        default_factory=lambda: [22, 23, 24, 25, 26, 27, 28, 16, 37, 17]
    )
    # P2-A: Dynamic layout escalation — if CES spread across selected layouts
    # is below min_ces_spread, escalate from n_layouts to n_layouts_max.
    min_ces_spread: float = 0.02  # Minimum CES spread for ZNE extrapolation quality
    n_layouts_max: int = 5  # Maximum layouts when spread is insufficient
    # ── QESEM Configuration (Qedma Qiskit Function, arXiv:2508.10997) ────
    # When mitigation.qesem_enabled=True, the deployment pipeline delegates
    # mitigation entirely to QESEM (bypasses PEA/GF-ZNE + affine correction).
    # QESEM handles: transpilation, characterization, ES, and unbiased EM.
    # Requires: IBM Premium/Flex/On-Prem plan + qiskit-ibm-catalog package.
    qesem_precision: float = 0.01  # Target ε for ⟨O⟩ (QESEM's default_precision)
    # QPU time cap per PUB (seconds). 300s is often insufficient for 20 observables
    # at ε=0.01 — QESEM will converge as far as possible within this budget.
    # Empirical finding (2026-06-23): 300s → σ≈0.29 for 20 obs on ibm_kingston.
    # 600s should approximately halve σ (σ ∝ 1/√T_qpu for shot-noise-limited EM).
    qesem_max_execution_time: int = 600  # QPU time cap per PUB (seconds)
    qesem_instance: str | None = None  # IBM instance for QESEM (None = use env)
    # ── QET (Quasi-probabilistic Error Tuning) — explicit noise scales ────
    # When set, enables QET mode: QESEM returns expectation values at each
    # requested noise scale, and the user performs extrapolation independently.
    # Format: {scale: target_precision, ...}
    #   - scale=0.0 → QESEM fully mitigated result
    #   - 0 < scale < 1 → partially reduced noise
    #   - scale=1.0 → physical device noise (with REM)
    #   - scale > 1.0 → amplified noise
    # Complementary pairs around 1.0 come free (e.g., request 0.5 → get 1.5).
    # Example: {0.0: 0.01, 0.5: 0.02, 1.5: 0.03} requests 3 explicit scales.
    # When None (default): standard QESEM flow (uses qesem_precision as target).
    # Reference: GitHub Qedma/QET-tutorial (qet_scales.ipynb, qesem_heuristic.ipynb)
    qesem_noise_scales: dict[float, float] | None = None
    mitigation: MitigationOptions = field(
        default_factory=lambda: MitigationOptions(
            dd_enabled=True,
            trex_enabled=True,
            twirling_enabled=True,
            zne_enabled=True,
            zne_amplifier="pea",  # Primary: PEA (+94.4% gain, R²=0.998)
            num_randomizations=32,
            shots_per_randomization=128,  # IBM LayerNoiseLearning default=128
            # HVA p=1: 1 layer of 2Q gates → shorter pair_depths suffice.
            # None = let Runtime use its default. Explicit: [0, 1, 2, 4, 8].
            layer_pair_depths=None,
            # "active-circuit" avoids twirling idle qubits (IBM recommendation).
            twirling_strategy="active-circuit",
        )
    )


@dataclass
class SPSAConfig:
    """Validated SPSA parameters from V7-4A grid search (36 configs x 10 seeds)."""

    a: float = 0.1
    c: float = 0.05
    A: float = 10.0  # noqa: N815
    n_iterations: int = 200
    alpha: float = 0.602
    gamma: float = 0.101


@dataclass
class HardwareRunResult:
    """Complete result for one h-point hardware execution."""

    h_value: float
    e_exact: float
    e_zne: float
    delta_e_gap: float
    gap: float
    phase_label: str
    expected_label: str
    zne_r2: float
    zne_gain: float
    mag_x_mean: float
    corr_zz_mean: float
    sigma: float
    total_shots: int
    job_ids: list[str] = field(default_factory=list)
    layouts_used: list[list[int]] = field(default_factory=list)
    ces_values: list[float] = field(default_factory=list)
    per_site_x: list[float] = field(default_factory=list)
    per_bond_zz: list[float] = field(default_factory=list)
    is_partial: bool = False
    spsa_applied: bool = False
    verdict: str = ""
    verdict_reason: str = ""  # Human-readable verdict explanation
    zne_amplifier_used: str = ""  # "pea", "gate_folding", "server_side", or "average"
    mitigation_strategy: str = (
        ""  # "ibm_zne_layout_avg" | "ces_zne" | "gate_folding_local" | "pea_local"
    )
    layout_std: float | None = None  # std across layouts (when using layout averaging)
    fallback_triggered: bool = False  # True if adaptive ZNE fell back from GF to PEA
    # GNN-QEM post-correction (optional, from predictors.gnn_qem)
    gnn_qem_applied: bool = False
    gnn_qem_delta_e: float | None = None  # ΔE correction predicted by GNN
    gnn_qem_confidence: float | None = None  # Model confidence [0,1]
    e_after_gnn_qem: float | None = None  # Energy after GNN correction
    # Affine correction (optional, from noisy_utils.affine_correct_energy)
    affine_correction_applied: bool = False
    e_after_affine: float | None = None
    # Statistical uncertainty on e_zne (from QESEM stds or multi-layout σ/√n)
    e_zne_std: float | None = None  # None = not available (PEA path), float = QESEM std
    # Post-QPU validation metrics (zero-cost sanity checks)
    obs_bounds_clipped: bool = False  # True if any observable was clipped to [-1,1]
    n_obs_violations: int = 0  # Count of |⟨O⟩| > 1 violations before clipping
    layout_energy_outliers: int = 0  # Count of layouts with energy > 5σ from mean
    e_obs_discrepancy: float | None = None  # |E_ZNE - E_reconstructed| from observables
    e_obs_cross_valid_passed: bool = True  # False if discrepancy > 2×gap
    n_layouts_observables: int = 0  # Number of layouts used for multi-layout obs (P1-A)
    # P2-C: Stale calibration comparison (post-sweep vs pre-sweep)
    stale_calibration_t1_drift_pct: float | None = None  # None if not measured
    stale_calibration_stable: bool | None = None  # True/False/None
    # P3: Adaptive shot budget (when κ or σ_flow triggered boost)
    effective_shots: int | None = None  # None = used base shots, int = actual shots used
    adaptive_shot_reason: str = ""  # "kappa_high", "sigma_flow", or ""
    # ── QESEM execution metadata (when mitigation.qesem_enabled=True) ─────
    qesem_used: bool = False  # True if QESEM was the mitigation strategy
    qesem_job_id: str = ""  # QESEM Qiskit Function job ID (for provenance)
    qesem_total_qpu_time: float | None = None  # Seconds reported by QESEM
    qesem_gate_fidelities: dict | None = None  # Fidelities measured by QESEM
    qesem_total_shots: int | None = None  # Total shots used by QESEM
    qesem_mitigation_shots: int | None = None  # Shots allocated to mitigation
    qesem_noisy_evs: list[float] | None = None  # Pre-mitigation raw estimates
