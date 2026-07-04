"""Benchmark configuration registry for mitigation experiments (C0-C18).

This module defines the 19 benchmark configurations used to systematically
evaluate suppression/mitigation stacks on the GNN-HVA circuit (TFIM, N=10,
p=1, heavy_hex). Each configuration specifies a unique combination of DD,
twirling, TREX, ZNE method, affine correction, GNN-QEM, and AQC options.

Design note (DD-2/DD-3 exclusion):
  DD-2 (skip_reset_qubits) and DD-3 (scheduling_method) are NOT exposed as
  fields because IBM Runtime EstimatorV2 does not accept them as user-configurable
  parameters. The DD implementation is fully controlled by the runtime when
  dd_enabled=True. See §7 of the benchmark plan for rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

from qmbp_simulation.execution import MitigationOptions

VALID_DD_SEQUENCES = ("XX", "XpXm", "XY4")
MITIQ_METHODS = ("mitiq_zne", "mitiq_cdr", "mitiq_ddd_zne")

# Valid config IDs — defined before the dataclass to break the chicken-and-egg
# problem with __post_init__ validation referencing BENCHMARK_CONFIGS.
_VALID_CONFIG_IDS: frozenset[str] = frozenset(
    {
        "C0_raw",
        "C1_dd_only",
        "C2_dd_tw",
        "C3_full_gf",
        "C4_full_pea_light",
        "C5_full_pea_balanced",
        "C6_full_pea_heavy",
        "C7_pea_no_dd",
        "C8_pea_xy4",
        "C9_gnn_qem",
        "C10_kitchen_sink",
        "C11_mitiq_zne",
        "C12_mitiq_cdr",
        "C13_mitiq_ddd_zne",
        "C14_dd_mitiq_cdr",
        "C15_pea_no_affine",
        "C16_aqc_pea",
        "C17_aqc_mitiq_cdr",
        "C18_aqc_raw",
        "C19_aqc_gf",
        "C20_aqc_dd_tw",
        "C21_qesem",
    }
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable configuration for one benchmark run (C0-C18).

    Each instance fully specifies the suppression/mitigation stack.
    Frozen to guarantee reproducibility — once created, never mutated.

    Design note (DD-2/DD-3 exclusion):
      - DD-2 (skip_reset_qubits) and DD-3 (scheduling_method) are NOT exposed as
        fields because IBM Runtime EstimatorV2 does not accept them as user-configurable
        parameters. The DD implementation is fully controlled by the runtime when
        dd_enabled=True. See §7 of the benchmark plan for rationale.

    Fields:
      config_id: Unique identifier (must be in _VALID_CONFIG_IDS).
      dd_enabled: Enable dynamical decoupling.
      dd_sequence: DD pulse sequence ("XX", "XpXm", "XY4") or None.
      twirling_num_randomizations: Number of Pauli twirling randomizations (None=disabled).
      trex_enabled: Twirled readout error extinction.
      zne_method: ZNE amplifier/method — "gf"|"pea"|"mitiq_zne"|"mitiq_cdr"|"mitiq_ddd_zne".
      zne_noise_factors: Noise amplification factors for ZNE extrapolation.
      pea_num_randomizations: PEA noise learning randomizations.
      pea_shots_per_randomization: Shots per PEA randomization.
      affine_enabled: Apply affine correction (clips to [E_ground, E_upper]).
      gnn_qem_enabled: Apply GNN-QEM error correction post-processing.
      aqc_enabled: Use AQC-Tensor compressed circuit.
      optimization_level: Qiskit transpiler optimization level (0-3).
      priority: Execution priority 0-4 (P0=critical … P4=optional).
      n_layouts: Number of layout candidates to evaluate (default=3).
      description: Human-readable description of the configuration.
    """

    config_id: str
    dd_enabled: bool = False
    dd_sequence: str | None = None
    twirling_num_randomizations: int | None = None
    trex_enabled: bool = False
    zne_method: str | None = None
    zne_noise_factors: list[float] | None = None
    pea_num_randomizations: int | None = None
    pea_shots_per_randomization: int | None = None
    affine_enabled: bool = True
    gnn_qem_enabled: bool = False
    aqc_enabled: bool = False
    qesem_enabled: bool = False  # Use QESEM (Qedma) — hardware-only, requires Premium plan
    qesem_precision: float = 0.01  # Target ε per observable (when qesem_enabled=True)
    qesem_max_execution_time: int = 300  # Max QPU time per PUB in seconds
    optimization_level: int = 2
    priority: int = 2
    n_layouts: int = 3
    description: str = ""

    def __post_init__(self) -> None:
        """Validate config_id, priority, and n_layouts."""
        if self.config_id not in _VALID_CONFIG_IDS:
            valid = ", ".join(sorted(_VALID_CONFIG_IDS))
            raise ValueError(f"Invalid config_id '{self.config_id}'. Valid: {valid}")
        if not (0 <= self.priority <= 4):
            raise ValueError(f"priority must be 0-4, got {self.priority}")
        if self.n_layouts < 1:
            raise ValueError(f"n_layouts must be >= 1, got {self.n_layouts}")

    def to_mitigation_options(self) -> MitigationOptions:
        """Convert to MitigationOptions for HardwareBackend execution.

        Routing logic:
          - Mitiq methods force optimization_level=0 (Qiskit 2.x cancels folded
            gates at level >= 1).
          - zne_enabled is True when any ZNE method is specified.
          - DD sequence defaults to "XpXm" if dd_enabled but no sequence given.
          - PEA budget uses config values or sensible defaults (48/192).
        """
        return MitigationOptions(
            zne_enabled=self.zne_method is not None,
            zne_amplifier=self._resolve_amplifier(),
            zne_noise_factors=self.zne_noise_factors,
            dd_enabled=self.dd_enabled,
            dd_sequence=self.dd_sequence or "XpXm",
            trex_enabled=self.trex_enabled,
            twirling_enabled=self.twirling_num_randomizations is not None,
            num_randomizations=self.pea_num_randomizations or 48,
            shots_per_randomization=self.pea_shots_per_randomization or 192,
        )

    @property
    def is_mitiq(self) -> bool:
        """True if this config uses a Mitiq-based ZNE method."""
        return self.zne_method in MITIQ_METHODS

    @property
    def h_test_values(self) -> list[float]:
        """h-values to test for this config.

        AQC configs extend the range to include h=3.0 (deeper into the
        ordered phase) since AQC compression is most beneficial there.
        """
        if self.aqc_enabled:
            return [3.0, 3.25, 3.5, 3.75, 4.0]
        return [3.25, 3.5, 3.75, 4.0]

    def _resolve_amplifier(self) -> str:
        """Map zne_method to the amplifier string for MitigationOptions."""
        if self.zne_method == "pea":
            return "pea"
        if self.zne_method == "gf":
            return "gate_folding"
        return "gate_folding"  # default for Mitiq methods and None


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK_CONFIGS — The 19 benchmark configurations (C0-C18)
#
# Priority levels (§7 of benchmark plan):
#   P0 = Critical baselines (must run first)
#   P1 = Important comparisons
#   P2 = Standard evaluation
#   P3 = Low priority (ablation/optional)
#   P4 = Optional (future work)
#
# n_layouts: Number of layout candidates. C0_raw uses 1 (no layout averaging
# needed for raw baseline). All others use 3 (default).
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARK_CONFIGS: dict[str, BenchmarkConfig] = {
    # ── P0: Critical baselines ──────────────────────────────────────────────
    "C0_raw": BenchmarkConfig(
        "C0_raw",
        affine_enabled=False,
        priority=0,
        n_layouts=1,
        description="Raw baseline, no mitigation",
    ),
    "C3_full_gf": BenchmarkConfig(
        "C3_full_gf",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=32,
        trex_enabled=True,
        zne_method="gf",
        zne_noise_factors=[1.0, 3.0, 5.0],  # Must be odd integers for gate-folding
        priority=0,
        description="Full stack with gate-folding ZNE",
    ),
    "C5_full_pea_balanced": BenchmarkConfig(
        "C5_full_pea_balanced",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=48,
        trex_enabled=True,
        zne_method="pea",
        pea_num_randomizations=48,
        pea_shots_per_randomization=192,
        priority=0,
        description="PEA balanced budget (recommended)",
    ),
    # ── P1: Important comparisons ─────────────────────────────────────────
    "C1_dd_only": BenchmarkConfig(
        "C1_dd_only",
        dd_enabled=True,
        dd_sequence="XpXm",
        priority=1,
        description="DD only (suppression baseline)",
    ),
    "C2_dd_tw": BenchmarkConfig(
        "C2_dd_tw",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=32,
        trex_enabled=True,
        priority=1,
        description="DD + Twirling + TREX",
    ),
    "C4_full_pea_light": BenchmarkConfig(
        "C4_full_pea_light",
        dd_enabled=True,
        dd_sequence="XpXm",
        trex_enabled=True,
        zne_method="pea",
        pea_num_randomizations=32,
        pea_shots_per_randomization=128,
        priority=1,
        description="PEA light budget",
    ),
    "C10_kitchen_sink": BenchmarkConfig(
        "C10_kitchen_sink",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=48,
        trex_enabled=True,
        zne_method="pea",
        pea_num_randomizations=48,
        pea_shots_per_randomization=192,
        gnn_qem_enabled=True,
        priority=1,
        description="PEA + GNN-QEM post-processing (kitchen sink)",
    ),
    # ── P2: Standard evaluation ──────────────────────────────────────────
    "C6_full_pea_heavy": BenchmarkConfig(
        "C6_full_pea_heavy",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=64,
        trex_enabled=True,
        zne_method="pea",
        pea_num_randomizations=64,
        pea_shots_per_randomization=256,
        priority=2,
        description="PEA heavy budget",
    ),
    "C7_pea_no_dd": BenchmarkConfig(
        "C7_pea_no_dd",
        twirling_num_randomizations=48,
        trex_enabled=True,
        zne_method="pea",
        pea_num_randomizations=48,
        pea_shots_per_randomization=192,
        priority=2,
        description="PEA without DD (ablation)",
    ),
    "C8_pea_xy4": BenchmarkConfig(
        "C8_pea_xy4",
        dd_enabled=True,
        dd_sequence="XY4",
        twirling_num_randomizations=48,
        trex_enabled=True,
        zne_method="pea",
        pea_num_randomizations=48,
        pea_shots_per_randomization=192,
        priority=2,
        description="PEA with XY4 DD sequence (vs XpXm)",
    ),
    "C9_gnn_qem": BenchmarkConfig(
        "C9_gnn_qem",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=32,
        trex_enabled=True,
        zne_method="gf",
        zne_noise_factors=[1.0, 3.0, 5.0],  # Must be odd integers for gate-folding
        gnn_qem_enabled=True,
        priority=2,
        description="Gate-folding ZNE + GNN-QEM",
    ),
    "C11_mitiq_zne": BenchmarkConfig(
        "C11_mitiq_zne",
        dd_enabled=True,
        dd_sequence="XpXm",
        trex_enabled=True,
        zne_method="mitiq_zne",
        zne_noise_factors=[1.0, 2.0, 3.0],
        optimization_level=0,
        priority=2,
        description="Mitiq ZNE (random folding)",
    ),
    "C12_mitiq_cdr": BenchmarkConfig(
        "C12_mitiq_cdr",
        dd_enabled=True,
        dd_sequence="XpXm",
        trex_enabled=True,
        zne_method="mitiq_cdr",
        optimization_level=0,
        priority=2,
        description="Mitiq CDR (Clifford Data Regression)",
    ),
    "C13_mitiq_ddd_zne": BenchmarkConfig(
        "C13_mitiq_ddd_zne",
        dd_enabled=True,
        dd_sequence="XpXm",
        trex_enabled=True,
        zne_method="mitiq_ddd_zne",
        zne_noise_factors=[1.0, 2.0, 3.0],
        optimization_level=0,
        priority=2,
        description="Mitiq DDD+ZNE composition",
    ),
    "C16_aqc_pea": BenchmarkConfig(
        "C16_aqc_pea",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=48,
        trex_enabled=True,
        zne_method="pea",
        pea_num_randomizations=48,
        pea_shots_per_randomization=192,
        aqc_enabled=True,
        priority=2,
        description="AQC-compressed circuit + PEA",
    ),
    # ── P3: Low priority (ablation / optional) ────────────────────────────
    "C14_dd_mitiq_cdr": BenchmarkConfig(
        "C14_dd_mitiq_cdr",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=32,
        trex_enabled=True,
        zne_method="mitiq_cdr",
        optimization_level=0,
        priority=3,
        description="DD + Twirling + Mitiq CDR",
    ),
    "C15_pea_no_affine": BenchmarkConfig(
        "C15_pea_no_affine",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=48,
        trex_enabled=True,
        zne_method="pea",
        pea_num_randomizations=48,
        pea_shots_per_randomization=192,
        affine_enabled=False,
        priority=3,
        description="PEA without affine correction (ablation)",
    ),
    "C17_aqc_mitiq_cdr": BenchmarkConfig(
        "C17_aqc_mitiq_cdr",
        dd_enabled=True,
        dd_sequence="XpXm",
        trex_enabled=True,
        zne_method="mitiq_cdr",
        aqc_enabled=True,
        optimization_level=0,
        priority=3,
        description="AQC-compressed circuit + Mitiq CDR",
    ),
    "C18_aqc_raw": BenchmarkConfig(
        "C18_aqc_raw",
        dd_enabled=True,
        dd_sequence="XpXm",
        aqc_enabled=True,
        priority=3,
        description="AQC-compressed circuit, no ZNE (ablation)",
    ),
    # ── New AQC variants (V2.1) ───────────────────────────────────────────
    "C19_aqc_gf": BenchmarkConfig(
        "C19_aqc_gf",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=32,
        trex_enabled=True,
        zne_method="gf",
        zne_noise_factors=[1.0, 3.0, 5.0],
        aqc_enabled=True,
        priority=2,
        description="AQC-compressed circuit + gate-folding ZNE",
    ),
    "C20_aqc_dd_tw": BenchmarkConfig(
        "C20_aqc_dd_tw",
        dd_enabled=True,
        dd_sequence="XpXm",
        twirling_num_randomizations=48,
        trex_enabled=True,
        aqc_enabled=True,
        priority=2,
        description="AQC-compressed + DD + Twirling only (no ZNE)",
    ),
    # ── P1: QESEM (Qedma) — hardware-only, unbiased mitigation ───────────
    # QESEM is a server-side Qiskit Function that handles its own
    # transpilation, characterization, and quasi-probabilistic mitigation.
    # It cannot run locally (fake_backend) — hardware mode only.
    # Ref: arXiv:2508.10997
    "C21_qesem": BenchmarkConfig(
        "C21_qesem",
        qesem_enabled=True,
        qesem_precision=0.01,
        qesem_max_execution_time=300,
        affine_enabled=False,  # QESEM output is already unbiased — no affine needed
        priority=1,
        n_layouts=3,  # Layout selection still used for provenance recording
        description="QESEM (Qedma) unbiased quasi-probabilistic mitigation",
    ),
}
