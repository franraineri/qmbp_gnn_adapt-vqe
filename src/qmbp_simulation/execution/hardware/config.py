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
