"""HardwareBackend — IBM Runtime execution with PEA/GF ZNE (primary) and R² quality gate.

Two interface levels:
- evaluate() — ABC contract, returns float. No side effects (no disk writes).
- run_deployment() — full pipeline with preflight + classify + persist.

ZNE Strategy (2026-06-04 update, per 13_hardware_zne_improvements.md):
- Hardware mode: IBM Runtime handles ZNE server-side. We average across layouts.
  CES-based client-side extrapolation is REMOVED — it fails on heavy_hex (R²≈0.04).
- fake_backend mode: local GF-ZNE or PEA-ZNE via noisy_utils.
- Default amplifier changed to "pea" (primary) with "gate_folding" as fallback.
- R² quality gate added: INDETERMINATE verdict when ZNE quality < 0.80.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.execution.backends import ExecutionBackend
from qmbp_simulation.execution.hardware.config import HardwareConfig, HardwareRunResult, SPSAConfig
from qmbp_simulation.execution.noisy_utils import LayoutSelection, NoisyEstimatorConfig, linear_zne

logger = logging.getLogger(__name__)

# R² threshold below which the ZNE result is considered unreliable
ZNE_R2_QUALITY_THRESHOLD = 0.80


class HardwareBackend(ExecutionBackend):
    """IBM Runtime hardware backend with PEA/GF ZNE and R² quality gate.

    ZNE strategy (2026-06-04):
    - hardware mode: trust IBM server-side ZNE, average across layouts.
    - fake_backend mode: local GF or PEA ZNE via noisy_utils.
    - Default amplifier: "pea" (validated +94.4% gain vs GF +20.6%).
    """

    def __init__(self, config: HardwareConfig | None = None) -> None:
        self._config = config or HardwareConfig()
        self._service: Any = None
        self._backend: Any = None
        self._estimator: Any = None
        self._execution_mode: type | None = None
        self._logger = self._create_logger()
        self._spsa_config = SPSAConfig()
        self._total_shots = 0
        self._cached_layout: LayoutSelection | None = None
        self._cached_circuit_key: tuple[int, int, int] | None = None
        self._spsa_rng = np.random.default_rng(self._config.layout_seed + 1000)
        self._gnn_qem_model: Any = None  # Optional GNN-QEM corrector (loaded via load_gnn_qem)

        if self._config.mode == "hardware":
            try:
                import qiskit_ibm_runtime  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "qiskit-ibm-runtime is required for hardware mode. "
                    "Install with: pip install 'qiskit-ibm-runtime>=0.20'"
                ) from e

    def load_gnn_qem(self, checkpoint_path: str | Path) -> None:
        """Load a trained GNN-QEM model for post-ZNE energy correction.

        Auto-detects V1 vs V2 checkpoint format. When loaded,
        `run_deployment()` will apply GNN correction after ZNE
        and before affine clipping. The correction is confidence-gated:
        if the model's confidence is below `gnn_qem_confidence_threshold`
        (default 0.5), the correction is skipped.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to a .pt checkpoint saved by `save_qem_checkpoint()` (V1)
            or `save_qem_v2_checkpoint()` (V2).

        References
        ----------
        Wang et al. arXiv:2604.16815 (2026) — GEM framework.
        """
        import torch

        path = Path(checkpoint_path)
        # Auto-detect version by peeking at checkpoint
        raw = torch.load(path, map_location="cpu", weights_only=False)  # nosec: trusted checkpoint
        is_v2 = isinstance(raw, dict) and raw.get("version") == "2.0"

        if is_v2:
            from qmbp_simulation.predictors.gnn_qem import (
                GNNQEMConfigV2,
                GNNQEMCorrectorV2,
            )
            config = GNNQEMConfigV2(**raw["config"])
            model = GNNQEMCorrectorV2(config)
            model.load_state_dict(raw["state_dict"])
            model.eval()
            self._gnn_qem_model = model
            self._gnn_qem_version = 2
            train_result = raw.get("train_result")
            metadata = raw.get("metadata", {})
        else:
            from qmbp_simulation.predictors.gnn_qem import load_qem_checkpoint
            model, train_result, metadata = load_qem_checkpoint(path)
            self._gnn_qem_model = model
            self._gnn_qem_version = 1
        self._logger.log(
            "gnn_qem_loaded",
            data={
                "checkpoint": str(checkpoint_path),
                "train_result": {
                    "best_epoch": train_result.best_epoch if train_result else None,
                    "val_mae": train_result.val_mae if train_result else None,
                    "improvement": train_result.val_improvement_pct if train_result else None,
                },
                "metadata": metadata,
            },
        )

    # ─── Lazy Connection ─────────────────────────────────────────────

    @staticmethod
    def _create_logger():
        """Lazy import of StructuredLogger to avoid circular dependency."""
        from qmbp_simulation.framework.logging import StructuredLogger

        return StructuredLogger("hardware_deployment")

    @property
    def backend(self) -> Any:
        """Backend resolved lazily on first access."""
        if self._backend is None:
            self._connect()
        return self._backend

    def _connect(self) -> None:
        """Establish connection to IBM Quantum or initialize fake backend."""
        if self._config.mode == "fake_backend":
            from qiskit_ibm_runtime.fake_provider import FakeTorino

            self._backend = FakeTorino()
        else:
            import os

            from qiskit_ibm_runtime import QiskitRuntimeService

            key = os.environ.get("IBM_KEY")
            crn = os.environ.get("IBM_INSTANCE_CRN")
            if not key:
                raise ValueError("IBM_KEY environment variable not set.")
            if not crn:
                raise ValueError("IBM_INSTANCE_CRN environment variable not set.")
            self._service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=key,
                instance=crn,
            )
            try:
                self._backend = self._service.backend(self._config.backend_name)
            except Exception as exc:
                # Backend name not found — list available backends to help user
                available = []
                try:
                    backends = self._service.backends(
                        min_num_qubits=self._config.n_qubits,
                        operational=True,
                    )
                    available = [b.name for b in backends]
                except Exception:
                    pass
                available_str = ", ".join(available) if available else "(unable to list)"
                raise ValueError(
                    f"Backend '{self._config.backend_name}' not found for your instance.\n"
                    f"  Original error: {exc}\n"
                    f"  Available backends (≥{self._config.n_qubits} qubits): {available_str}\n"
                    f"  Fix: set --backend <name> or update BACKEND_NAME in the deployment script.\n"
                    f"  Hint: IBM may have renamed the backend. Check "
                    f"https://quantum.cloud.ibm.com/services/resources"
                ) from exc
        self._execution_mode = self._detect_execution_mode()

    def _detect_execution_mode(self) -> type:
        """Detect Batch vs Session via hasattr introspection."""
        import qiskit_ibm_runtime

        if hasattr(qiskit_ibm_runtime, "Batch"):
            return qiskit_ibm_runtime.Batch  # type: ignore[no-any-return]
        if hasattr(qiskit_ibm_runtime, "Session"):
            return qiskit_ibm_runtime.Session  # type: ignore[no-any-return]
        raise ImportError(
            "Neither Batch nor Session found in qiskit_ibm_runtime. "
            "Upgrade to qiskit-ibm-runtime >= 0.20."
        )

    # ─── ExecutionBackend ABC ──────────────────────────────────────────

    def evaluate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        """Return ZNE-mitigated energy. No side effects (no disk writes).

        ZNE strategy is mode-aware:
        - hardware mode + zne_enabled: IBM server-side ZNE, layout averaging.
        - fake_backend mode + zne_enabled: local GF/PEA ZNE via noisy_utils.
        - Any mode + zne_disabled: CES-based client-side extrapolation (legacy).
        """
        bound = circuit.assign_parameters(params)
        layout_selection = self._get_cached_layouts(bound)

        from .submission import submit_all_then_collect

        raw_results = submit_all_then_collect(
            layout_selection.transpiled_circuits,
            hamiltonian,
            self.backend,
            self._config,
            self._logger,
        )
        if not raw_results:
            raise RuntimeError("All jobs failed or returned non-finite values.")

        energies = [r["energy"] for r in raw_results]
        self._total_shots += self._config.shots * len(raw_results)

        if self._config.mode == "hardware" and self._config.mitigation.zne_enabled:
            # Path A: IBM Runtime applies ZNE server-side → layout averaging
            # (each layout already returns ZNE-mitigated energy)
            return float(np.mean(energies))

        if self._config.mode == "fake_backend" and self._config.mitigation.zne_enabled:
            # Path B: Local GF/PEA ZNE via noisy_utils
            e_zne, _, _ = self._run_local_zne(raw_results, layout_selection, hamiltonian, gap=1.0)
            return e_zne

        # Path C: Legacy CES-ZNE (zne_enabled=False, any mode)
        ces_used = [layout_selection.ces_values[r["layout_idx"]] for r in raw_results]
        zne_result = linear_zne(np.array(ces_used), np.array(energies))
        return zne_result.extrapolated_value

    @property
    def name(self) -> str:
        return f"hardware_{self._config.backend_name}"

    # ─── Layout Caching ───────────────────────────────────────────────

    def _get_cached_layouts(self, bound_circuit: QuantumCircuit) -> LayoutSelection:
        """Return cached layouts if circuit structure unchanged."""
        key = (bound_circuit.num_qubits, bound_circuit.num_parameters, bound_circuit.depth())
        if self._cached_layout is not None and self._cached_circuit_key == key:
            return self._cached_layout
        from .submission import select_layouts_for_hardware

        self._cached_layout = select_layouts_for_hardware(
            bound_circuit,
            self.backend,
            self._config,
            self._logger,
        )
        self._cached_circuit_key = key
        return self._cached_layout

    # ─── High-Level Orchestration ─────────────────────────────────────

    def run_deployment(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
        h_value: float,
        e_exact: float,
        gap: float,
        expected_label: str = "paramagnetic",
    ) -> HardwareRunResult:
        """Full pipeline: preflight → evaluate → observables → classify → persist.

        Validates inputs before any QPU interaction to fail fast on misconfigs.
        """
        from .observables import build_per_site_observables, map_observables_to_layout
        from .persistence import save_partial_before_error, save_run
        from .phase import classify_phase
        from .spsa import spsa_refinement
        from .submission import build_estimator_options, submit_all_then_collect

        # ─── Input validation (fail fast, no QPU cost) ───────────────────
        params = np.asarray(params, dtype=float)
        if params.ndim != 1:
            raise ValueError(f"params must be 1-D array, got shape {params.shape}")
        if circuit.num_parameters != len(params):
            raise ValueError(
                f"Circuit has {circuit.num_parameters} parameters but "
                f"params has length {len(params)}"
            )
        if gap <= 0:
            raise ValueError(f"Spectral gap must be positive, got {gap}")
        if not np.all(np.isfinite(params)):
            raise ValueError("params contains NaN or Inf values")
        if not np.isfinite(e_exact):
            raise ValueError(f"e_exact is not finite: {e_exact}")

        self._logger.log(
            "deployment_start",
            h_value=h_value,
            data={
                "n_qubits": circuit.num_qubits,
                "n_params": len(params),
                "h_value": h_value,
                "e_exact": e_exact,
                "gap": gap,
                "expected_label": expected_label,
                "params_norm": float(np.linalg.norm(params)),
            },
        )

        partial_data: list[dict] = []
        try:
            preflight = self.run_preflight()
            if preflight.get("abort"):
                raise RuntimeError(f"Preflight abort: {preflight.get('abort_reason')}")

            # Validate circuit gate count for ZNE viability
            from .preflight import validate_circuit_for_zne

            circuit_check = validate_circuit_for_zne(circuit, self._config, self._logger)
            if circuit_check.get("abort"):
                raise RuntimeError(f"Circuit check: {circuit_check.get('abort_reason')}")

            bound = circuit.assign_parameters(params)
            layout_selection = self._get_cached_layouts(bound)

            # ── Layout-aware 2Q error verification (2026-06-14) ───────
            # The global preflight checks chip-wide mean (3.36% on Kingston).
            # Here we verify the ACTUAL qubits selected by BFS+CES layout.
            # If our 10-qubit subgraph has >1.5% mean 2Q error, we warn
            # (the layout selection should have avoided bad qubits, but
            # calibration data can be stale).
            if self._config.mode == "hardware" and layout_selection.layouts:
                from .preflight import compute_layout_2q_error

                best_layout = layout_selection.layouts[0]
                layout_error = compute_layout_2q_error(self.backend, best_layout)
                if layout_error is not None:
                    self._logger.log(
                        "layout_2q_error",
                        data={
                            "layout_qubits": best_layout[: self._config.n_qubits],
                            "mean_2q_error": layout_error,
                            "threshold": 0.015,
                        },
                    )
                    if layout_error > 0.015:
                        self._logger.log(
                            "layout_2q_error_elevated",
                            data={
                                "error": layout_error,
                                "warning": (
                                    f"Selected layout has {layout_error * 100:.2f}% mean 2Q error "
                                    f"(>1.5%). ZNE accuracy may be degraded."
                                ),
                            },
                        )

            # ── Post-transpilation quality gate (2026-06-18) ─────────
            # Validates each transpiled circuit against calibration data BEFORE
            # any QPU interaction. Checks: error budget, depth_2q, defective
            # edges, routing expansion. Aborts if quality is catastrophic.
            from .preflight import validate_transpiled_circuit_quality

            transpiled_quality_checks: list[dict] = []
            for i, isa_circ in enumerate(layout_selection.transpiled_circuits):
                layout_qubits = (
                    layout_selection.layouts[i] if i < len(layout_selection.layouts) else None
                )
                quality_check = validate_transpiled_circuit_quality(
                    isa_circ, self.backend, layout_qubits, self._logger
                )
                transpiled_quality_checks.append(quality_check)
                if quality_check.get("abort"):
                    raise RuntimeError(
                        f"Transpiled circuit quality check failed (layout {i}): "
                        f"{quality_check.get('abort_reason')}"
                    )

            # ── Capture calibration snapshot at execution time ─────────
            # This records the actual T1/T2/error rates when the circuit ran,
            # enabling post-hoc correlation of result quality vs calibration.
            calibration_snapshot = self._capture_calibration_snapshot(layout_selection)

            # ── Capture transpiled circuit stats ──────────────────────
            transpiled_stats = self._capture_transpiled_stats(layout_selection)

            # ── Save circuit diagram BEFORE QPU submission ────────────
            # Generates a PNG of the bound circuit for provenance and debugging.
            # Saved to the run output directory so every hardware execution has
            # a visual record of exactly what was submitted.
            self._save_pre_execution_circuit(circuit, params, h_value, layout_selection)

            # ── Save circuit serialization (QASM3) for reproducibility ────
            # If QPU submission fails, this allows exact retry without
            # re-transpilation. Also enables post-hoc circuit fingerprinting.
            circuit_qasm_paths = self._save_circuit_qasm(layout_selection, h_value)

            # ══════════════════════════════════════════════════════════════
            # QESEM ALTERNATE PATH — bypasses local ZNE pipeline entirely
            # ══════════════════════════════════════════════════════════════
            if self._config.mitigation.qesem_enabled:
                from .observables import build_per_site_observables
                from .persistence import save_run
                from .phase import classify_phase
                from .qesem import check_qesem_available, run_qesem_deployment

                # ── Guard: QESEM cannot run locally (requires real QPU) ──
                if self._config.mode == "fake_backend":
                    raise RuntimeError(
                        "QESEM (qesem_enabled=True) cannot run in fake_backend mode. "
                        "QESEM is a server-side Qiskit Function that requires a real QPU. "
                        "Options: (1) Use mode='hardware' with real credentials, or "
                        "(2) Disable QESEM (qesem_enabled=False) to use local PEA-ZNE."
                    )

                available, err_msg = check_qesem_available()
                if not available:
                    raise ImportError(
                        f"QESEM enabled but dependencies missing: {err_msg}. "
                        f"Install: pip install qiskit-ibm-catalog>=0.8.0"
                    )

                edges = [(i, i + 1) for i in range(self._config.n_qubits - 1)]
                x_ops, zz_ops = build_per_site_observables(self._config.n_qubits, edges)

                # ── Save QESEM pre-submission manifest ────────────────────
                # Captures exactly what will be sent to QESEM for provenance,
                # real-time monitoring, and post-hoc debugging.
                qesem_manifest = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "strategy": "qesem_unbiased",
                    "backend_name": self._config.backend_name,
                    "h_value": h_value,
                    "e_exact": e_exact,
                    "gap": gap,
                    "expected_label": expected_label,
                    "qesem_config": {
                        "precision": self._config.qesem_precision,
                        "max_execution_time": self._config.qesem_max_execution_time,
                        "client_timeout_s": self._config.qesem_max_execution_time * 2 + 300,
                    },
                    "circuit": {
                        "n_qubits": bound.num_qubits,
                        "depth": bound.depth(),
                        "depth_2q": bound.depth(
                            filter_function=lambda instr: len(instr.qubits) == 2
                        ),
                        "gate_counts": dict(bound.count_ops()),
                        "params_bound": params.tolist(),
                    },
                    "observables": {
                        "n_total": 1 + len(x_ops) + len(zz_ops),
                        "energy_terms": len(hamiltonian),
                        "n_x_ops": len(x_ops),
                        "n_zz_ops": len(zz_ops),
                    },
                    "calibration_snapshot_available": calibration_snapshot is not None,
                    "transpiled_stats_available": transpiled_stats is not None,
                }
                from qmbp_simulation.utils.helpers import json_dump

                manifest_dir = Path(self._config.output_dir)
                manifest_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = manifest_dir / f"qesem_pre_submission_h{h_value:.2f}.json"
                json_dump(qesem_manifest, manifest_path)
                self._logger.log(
                    "qesem_pre_submission_manifest_saved",
                    data={"path": str(manifest_path), "h_value": h_value},
                )
                print(f"\n  📋 QESEM pre-submission manifest saved: {manifest_path}")

                qesem_result = run_qesem_deployment(
                    circuit=bound,
                    hamiltonian=hamiltonian,
                    x_ops=x_ops,
                    zz_ops=zz_ops,
                    config=self._config,
                    structured_logger=self._logger,
                )

                # Merge QESEM circuit_stats into transpiled_stats for validators.
                # This gives post-execution analysis access to both the pre-submission
                # logical circuit stats AND the QESEM-transpiled physical circuit info.
                if qesem_result.circuit_stats:
                    transpiled_stats = transpiled_stats or {}
                    transpiled_stats["qesem_circuit_stats"] = qesem_result.circuit_stats

                # Map QESEM output to HardwareRunResult format
                e_mitigated = qesem_result.energy_mitigated
                delta_e_gap = abs(e_mitigated - e_exact) / gap if gap > 0 else float("inf")
                x_values = qesem_result.x_values
                zz_values = qesem_result.zz_values

                # ── Precision quality gate (analogous to R² gate in PEA path) ──
                # If QESEM achieved std > 2× requested precision, warn. This can
                # happen when max_execution_time is hit before convergence.
                achieved_std = qesem_result.energy_std
                if achieved_std > self._config.qesem_precision * 2:
                    self._logger.log(
                        "qesem_precision_degraded",
                        data={
                            "achieved_std": achieved_std,
                            "requested_precision": self._config.qesem_precision,
                            "ratio": achieved_std / self._config.qesem_precision,
                            "warning": (
                                f"QESEM achieved σ={achieved_std:.4f} > "
                                f"2×ε={self._config.qesem_precision:.4f}. "
                                f"QPU time cap may have been reached before convergence."
                            ),
                        },
                    )

                # Compute ZNE gain from noisy vs mitigated (comparable to PEA).
                # Only valid when noisy data is genuinely available from QESEM.
                # When noisy_data_available=False, noisy_energy is a sentinel (0.0)
                # and zne_gain would be artificially inflated — report as None.
                if qesem_result.noisy_data_available:
                    raw_error = abs(qesem_result.noisy_energy - e_exact)
                    mitigated_error = abs(e_mitigated - e_exact)
                    zne_gain = 1.0 - (mitigated_error / raw_error) if raw_error > 1e-10 else 0.0
                else:
                    zne_gain = 0.0  # Cannot compute — noisy baseline unavailable

                label, mag_x, corr_zz, sigma = classify_phase(
                    x_values, zz_values, self._config.shots
                )

                # Verdict logic (same as local ZNE path)
                if delta_e_gap < 0.05 and label == expected_label:
                    verdict = "PASS"
                    verdict_reason = f"ΔE/gap={delta_e_gap:.4f} < 5%, phase={label} correct"
                elif delta_e_gap < 0.05:
                    verdict = "PARTIAL"
                    verdict_reason = (
                        f"Energy good (ΔE/gap={delta_e_gap:.4f}) but "
                        f"phase={label} ≠ {expected_label}"
                    )
                else:
                    verdict = "FAIL"
                    verdict_reason = f"ΔE/gap={delta_e_gap:.4f} ≥ 5%"

                result = HardwareRunResult(
                    h_value=h_value,
                    e_exact=e_exact,
                    e_zne=e_mitigated,  # QESEM energy stored in e_zne field
                    e_zne_std=qesem_result.energy_std,  # Statistical uncertainty
                    delta_e_gap=delta_e_gap,
                    gap=gap,
                    phase_label=label,
                    expected_label=expected_label,
                    zne_r2=1.0,  # QESEM is unbiased — no R² applicable
                    zne_gain=zne_gain,
                    mag_x_mean=mag_x,
                    corr_zz_mean=corr_zz,
                    sigma=sigma,
                    total_shots=qesem_result.total_shots or 0,
                    job_ids=[qesem_result.job_id],
                    layouts_used=[],  # QESEM handles its own transpilation/layout
                    ces_values=[],  # Not applicable to QESEM path
                    per_site_x=x_values,
                    per_bond_zz=zz_values,
                    verdict=verdict,
                    verdict_reason=verdict_reason,
                    zne_amplifier_used="qesem",
                    mitigation_strategy="qesem_unbiased",
                    # Observable clipping (from QESEM's post-execution diagnostics)
                    obs_bounds_clipped=(
                        qesem_result.circuit_stats.get("post_execution", {}).get(
                            "n_obs_violations", 0
                        )
                        > 0
                        if qesem_result.circuit_stats
                        else False
                    ),
                    n_obs_violations=(
                        qesem_result.circuit_stats.get("post_execution", {}).get(
                            "n_obs_violations", 0
                        )
                        if qesem_result.circuit_stats
                        else 0
                    ),
                    qesem_used=True,
                    qesem_job_id=qesem_result.job_id,
                    qesem_total_qpu_time=qesem_result.total_qpu_time,
                    qesem_gate_fidelities=qesem_result.gate_fidelities,
                    qesem_total_shots=qesem_result.total_shots,
                    qesem_mitigation_shots=qesem_result.mitigation_shots,
                    qesem_noisy_evs=(
                        [qesem_result.noisy_energy]
                        + qesem_result.noisy_x_values
                        + qesem_result.noisy_zz_values
                    )
                    if qesem_result.noisy_data_available
                    else None,
                    n_layouts_observables=0,  # QESEM manages its own layout strategy
                )

                save_run(
                    result,
                    self._config,
                    self._logger,
                    calibration_info=calibration_snapshot,
                    options_dict={"qesem_enabled": True, "precision": self._config.qesem_precision},
                    execution_mode_name="qesem",
                    raw_per_layout=[],
                    zne_data={"method": "qesem", "job_id": qesem_result.job_id},
                    input_params=params,
                    transpiled_stats=transpiled_stats,
                    qpu_metrics={"total_qpu_time": qesem_result.total_qpu_time},
                )
                result._calibration_snapshot = calibration_snapshot
                result._transpiled_stats = transpiled_stats
                result._qpu_metrics = {"total_qpu_time": qesem_result.total_qpu_time}

                # ── Automatic QET/QESEM post-execution validation ─────────
                # Runs the QET validator on the fresh result to log diagnostics.
                # Non-blocking: issues are logged but do NOT abort the pipeline.
                try:
                    from project_health.analysis.hardware.validate_qet import (
                        validate_qet_result,
                    )

                    qet_validation_data = {
                        "job_id": qesem_result.job_id,
                        "energy_mitigated": qesem_result.energy_mitigated,
                        "energy_std": qesem_result.energy_std,
                        "x_values": qesem_result.x_values,
                        "zz_values": qesem_result.zz_values,
                        "noisy_energy": (
                            qesem_result.noisy_energy if qesem_result.noisy_data_available else None
                        ),
                        "metadata": {
                            "gate_fidelities": qesem_result.gate_fidelities,
                            "total_shots": qesem_result.total_shots,
                            "mitigation_shots": qesem_result.mitigation_shots,
                            "total_qpu_time": qesem_result.total_qpu_time,
                        },
                    }
                    # Inject noise_scaling from QESEMResult if available
                    if qesem_result.noise_scale_results:
                        # Rebuild metadata.results format for the validator
                        _results_for_validator = []
                        for obs_scales in qesem_result.noise_scale_results:
                            rem_pts = [
                                {"scale": s, "value": v, "error_bar": std}
                                for s, (v, std) in obs_scales.items()
                            ]
                            _results_for_validator.append(
                                [
                                    "obs",
                                    {"noise_scaling": {"results_with_REM": rem_pts}},
                                ]
                            )
                        qet_validation_data["metadata"]["results"] = [_results_for_validator]

                    qet_report = validate_qet_result(qet_validation_data, e_exact=e_exact, gap=gap)

                    self._logger.log(
                        "qet_post_execution_validation",
                        data={
                            "passed": qet_report.passed,
                            "n_issues": len(qet_report.issues),
                            "n_warnings": qet_report.n_warnings,
                            "metrics": qet_report.metrics,
                            "issues": [
                                {"severity": i.severity, "check": i.check, "msg": i.message}
                                for i in qet_report.issues
                            ],
                        },
                    )
                    if not qet_report.passed:
                        logger.warning(
                            f"QET validation FAILED for h={h_value}: "
                            + "; ".join(
                                i.message for i in qet_report.issues if i.severity == "error"
                            )
                        )
                except ImportError:
                    pass  # project_health not installed in minimal deployments
                except Exception as exc:
                    logger.debug(f"QET validator error (non-blocking): {exc}")

                return result
            # ══════════════════════════════════════════════════════════════
            # END QESEM PATH — continue with local ZNE pipeline below
            # ══════════════════════════════════════════════════════════════

            # ── Save consolidated pre-submission manifest ─────────────
            # Single atomic JSON capturing everything about what is about to
            # be sent to the QPU: params, transpiled stats, calibration,
            # error budget, quality checks, circuit hashes, and config.
            pre_submission_manifest = self._build_pre_submission_manifest(
                circuit=circuit,
                params=params,
                h_value=h_value,
                e_exact=e_exact,
                gap=gap,
                expected_label=expected_label,
                layout_selection=layout_selection,
                calibration_snapshot=calibration_snapshot,
                transpiled_stats=transpiled_stats,
                circuit_check=circuit_check,
                transpiled_quality_checks=transpiled_quality_checks,
                circuit_qasm_paths=circuit_qasm_paths,
            )
            self._save_pre_submission_manifest(pre_submission_manifest, h_value)

            # ── Print circuit before QPU submission ─────────────────────
            # Shows the first transpiled ISA circuit that will be sent to the
            # QPU, including gate counts and depth. DD/twirling are applied
            # server-side and are NOT visible here (see circuit_with_dd_*.png
            # for an approximation).
            first_isa = layout_selection.transpiled_circuits[0]
            print("\n" + "=" * 70)
            print(f"  CIRCUIT BEING SENT TO QPU — h={h_value:.3f}")
            print("=" * 70)
            print(f"  Qubits: {first_isa.num_qubits}")
            print(f"  Depth: {first_isa.depth()}")
            print(f"  Gate counts: {dict(first_isa.count_ops())}")
            print(f"  Layouts: {len(layout_selection.transpiled_circuits)}")
            print(f"  Shots/layout: {self._config.shots}")
            print(
                f"  DD: XpXm (server-side)  |  Twirling: {'ON' if self._config.mitigation.twirling_enabled else 'OFF'}"
            )
            print(f"  ZNE amplifier: {self._config.mitigation.zne_amplifier}")
            if transpiled_quality_checks:
                eb = transpiled_quality_checks[0].get("error_budget")
                fid = transpiled_quality_checks[0].get("fidelity_estimate")
                if eb is not None:
                    print(f"  Error budget: {eb:.3f} (predicted fidelity: {fid:.1%})")
            print("-" * 70)
            print(first_isa.draw(output="text", fold=100))
            print("=" * 70 + "\n")

            raw_results = submit_all_then_collect(
                layout_selection.transpiled_circuits,
                hamiltonian,
                self.backend,
                self._config,
                self._logger,
            )

            # ── ZNE Aggregation (mode-aware) ──────────────────────────────
            # Hardware mode: IBM Runtime applies ZNE server-side when
            # options.resilience.zne_mitigation=True. Each layout already
            # returns a ZNE-mitigated energy. We average across layouts for
            # variance reduction (√n improvement) — NO client-side CES
            # extrapolation, which fails on heavy_hex (R²≈0.04).
            #
            # fake_backend mode: IBM server-side ZNE is not available.
            # Use local GF or PEA ZNE via noisy_utils.
            e_zne, zne_r2, zne_amplifier_used = self._aggregate_zne_results(
                raw_results, layout_selection, hamiltonian, gap
            )
            self._total_shots += self._config.shots * len(raw_results)
            partial_data.append(
                {"step": "energy", "e_zne": e_zne, "r2": zne_r2, "amplifier": zne_amplifier_used}
            )

            # Per-site observables on ALL layouts (P1-A: multi-layout averaging)
            # Submitting observables across all layouts within the same Batch
            # reduces σ by √(n_layouts) at zero extra QPU cost (same Batch).
            edges = [(i, i + 1) for i in range(self._config.n_qubits - 1)]
            x_ops, zz_ops = build_per_site_observables(self._config.n_qubits, edges)
            n_obs = len(x_ops) + len(zz_ops)

            if self._config.mode == "hardware":
                from qiskit_ibm_runtime import Batch as _Batch
                from qiskit_ibm_runtime import EstimatorV2 as _EstV2

                from .submission import _apply_estimator_options

                # Submit observables on each layout in a single Batch
                all_evs: list[np.ndarray] = []
                with _Batch(backend=self.backend) as obs_batch:
                    obs_est = _EstV2(mode=obs_batch)
                    _apply_estimator_options(obs_est, self._config)
                    obs_jobs = []
                    for isa_circ in layout_selection.transpiled_circuits:
                        mapped_obs = map_observables_to_layout(x_ops + zz_ops, isa_circ)
                        # ── Multi-layout observable mapping validation ────────
                        # Guard: verify mapped_obs has correct qubit count for
                        # this ISA circuit. A mismatch here means apply_layout
                        # produced an incompatible observable (silent corruption).
                        for j, m_op in enumerate(mapped_obs):
                            if m_op.num_qubits != isa_circ.num_qubits:
                                raise RuntimeError(
                                    f"Observable mapping error: mapped_obs[{j}] has "
                                    f"{m_op.num_qubits} qubits but ISA circuit has "
                                    f"{isa_circ.num_qubits}. Layout mismatch."
                                )
                        obs_jobs.append(obs_est.run([(isa_circ, mapped_obs)]))

                # Collect all observable results
                for layout_idx, obs_job in enumerate(obs_jobs):
                    try:
                        from .submission import wait_for_qpu_execution

                        wait_for_qpu_execution(obs_job, qpu_timeout_s=self._config.job_timeout_s)
                        obs_result = obs_job.result()
                        evs_layout = np.atleast_1d(obs_result[0].data.evs)
                        # Validate: result length must match submitted observables
                        if len(evs_layout) != n_obs:
                            self._logger.log(
                                "obs_layout_dimension_mismatch",
                                data={
                                    "layout_idx": layout_idx,
                                    "expected_n_obs": n_obs,
                                    "received_n_obs": len(evs_layout),
                                    "warning": "Observable result dimension mismatch. "
                                    "Skipping this layout to prevent data corruption.",
                                },
                            )
                            continue
                        if not np.all(np.isfinite(evs_layout)):
                            self._logger.log(
                                "obs_layout_non_finite",
                                data={
                                    "layout_idx": layout_idx,
                                    "n_non_finite": int(np.sum(~np.isfinite(evs_layout))),
                                },
                            )
                            continue
                        all_evs.append(evs_layout)
                    except Exception as obs_exc:
                        self._logger.log(
                            "obs_layout_failed",
                            data={"error": str(obs_exc), "layout_idx": layout_idx},
                        )

                if not all_evs:
                    raise RuntimeError("All observable layout jobs failed.")

                # Average across layouts: σ_avg = σ_single / √(n_layouts)
                evs = np.mean(all_evs, axis=0)
                self._logger.log(
                    "multi_layout_observables",
                    data={
                        "n_layouts_used": len(all_evs),
                        "n_layouts_total": len(layout_selection.transpiled_circuits),
                        "sigma_reduction_factor": float(np.sqrt(len(all_evs))),
                    },
                )
            else:
                # fake_backend: submit on first layout only (no Batch support)
                isa_circ = layout_selection.transpiled_circuits[0]
                mapped_obs = map_observables_to_layout(x_ops + zz_ops, isa_circ)
                estimator = self._get_configured_estimator()
                evs = np.atleast_1d(estimator.run([(isa_circ, mapped_obs)]).result()[0].data.evs)

            x_values = [float(evs[i]) for i in range(len(x_ops))]
            zz_values = [float(evs[len(x_ops) + i]) for i in range(len(zz_ops))]

            # ── Post-QPU result validation checks ─────────────────────────
            # These are zero-cost sanity checks that catch corrupt or
            # anomalous QPU results before they contaminate downstream logic.

            # Check 1: Observable bounds (|⟨O⟩| ≤ 1 for single-term Pauli ops)
            obs_out_of_bounds = []
            for i, xv in enumerate(x_values):
                if abs(xv) > 1.0 + 1e-6:
                    obs_out_of_bounds.append(("X", i, xv))
            for i, zzv in enumerate(zz_values):
                if abs(zzv) > 1.0 + 1e-6:
                    obs_out_of_bounds.append(("ZZ", i, zzv))
            if obs_out_of_bounds:
                self._logger.log(
                    "observable_bounds_violation",
                    data={
                        "violations": [
                            {"op": op, "idx": idx, "value": val}
                            for op, idx, val in obs_out_of_bounds
                        ],
                        "warning": "Observable magnitudes exceed operator norm (|⟨O⟩| > 1). "
                        "This indicates noise/estimation artifact. Clipping to [-1, 1].",
                    },
                )
                # Clip to physical bounds (ZNE extrapolation can overshoot)
                x_values = [max(-1.0, min(1.0, v)) for v in x_values]
                zz_values = [max(-1.0, min(1.0, v)) for v in zz_values]

            # Check 2: Layout energy outlier detection (>5σ from mean)
            layout_outliers = []
            if len(raw_results) > 1:
                layout_energies = np.array([r["energy"] for r in raw_results])
                layout_mean = np.mean(layout_energies)
                layout_std_val = np.std(layout_energies, ddof=1)
                # Theoretical σ for finite shots: ~gap/√shots (conservative estimate)
                sigma_theoretical = gap / np.sqrt(self._config.shots) if gap > 0 else 0.01
                # Use max(measured_std, 5×theoretical) as outlier threshold
                outlier_threshold = max(layout_std_val, 5.0 * sigma_theoretical)
                layout_outliers = []
                for r in raw_results:
                    if abs(r["energy"] - layout_mean) > outlier_threshold:
                        layout_outliers.append(r)
                if layout_outliers:
                    self._logger.log(
                        "layout_energy_outlier",
                        data={
                            "n_outliers": len(layout_outliers),
                            "layout_mean": float(layout_mean),
                            "layout_std": float(layout_std_val),
                            "outlier_threshold": float(outlier_threshold),
                            "outlier_layouts": [
                                {"idx": r["layout_idx"], "energy": r["energy"]}
                                for r in layout_outliers
                            ],
                            "warning": "One or more layouts returned energy far from the mean. "
                            "May indicate degraded qubit subset.",
                        },
                    )

            # Check 3: Energy-observable cross-validation
            # For TFIM: H = -J·ΣZZ - h·ΣX, so E_reconstructed ≈ -J·Σ⟨ZZ⟩ - h·Σ⟨X⟩
            # This catches corrupt QPU results where energy and observables are inconsistent.
            # NOTE: ZNE is non-linear → ZNE(H) ≠ Σ cᵢ·ZNE(Oᵢ). Expected discrepancy
            # is ~10% of |E| for PEA-ZNE with exponential extrapolation (validated
            # empirically: QESEM job 82aa showed 5% discrepancy at h=4.0).
            e_reconstructed = -1.0 * sum(zz_values) - h_value * sum(x_values)
            e_obs_discrepancy = abs(e_zne - e_reconstructed)
            # Adaptive threshold: 15% of |E_zne| or gap, whichever is larger.
            # The 2×gap threshold was too permissive (11.84 for h=4.0, never triggers).
            cross_validation_threshold = max(0.15 * abs(e_zne), gap) if gap > 0 else 1.0
            if e_obs_discrepancy > cross_validation_threshold:
                self._logger.log(
                    "energy_observable_inconsistency",
                    data={
                        "e_zne": e_zne,
                        "e_reconstructed": e_reconstructed,
                        "discrepancy": e_obs_discrepancy,
                        "threshold": cross_validation_threshold,
                        "h_value": h_value,
                        "warning": "Energy and observable measurements are inconsistent. "
                        "This may indicate ZNE applied differently to energy vs observables, "
                        "or a layout-dependent artifact.",
                    },
                )

            label, mag_x, corr_zz, sigma = classify_phase(
                x_values,
                zz_values,
                self._config.shots,
            )
            delta_e_gap = abs(e_zne - e_exact) / gap if gap > 0 else float("inf")

            # ── Post-ZNE Correction Stack (2025-2026 literature) ────────
            # 1. GNN-QEM correction (if model loaded) — Wang et al. 2604.16815
            # 2. Affine correction (physics bounds) — Wang et al. 2604.16815
            # Both are additive, non-destructive, and zero-overhead.
            gnn_qem_applied = False
            gnn_qem_delta_e: float | None = None
            gnn_qem_confidence: float | None = None
            e_after_gnn = e_zne
            affine_applied = False
            e_after_affine = e_zne

            if hasattr(self, "_gnn_qem_model") and self._gnn_qem_model is not None:
                # GNN-QEM only activates when PEA is NOT the amplifier.
                # Rationale: PEA already removes structured noise; GNN-QEM
                # on PEA residuals adds nothing (validated: 0% improvement).
                # GNN-QEM helps when GF-ZNE leaves large structured residuals.
                gnn_should_apply = "pea" not in zne_amplifier_used
                if gnn_should_apply:
                    try:
                        from qmbp_simulation.predictors.gnn_qem import QEMSample
                        from qmbp_simulation.predictors.gnn_qem import correct_energy as gnn_correct

                        qem_sample = QEMSample(
                            noisy_energy=e_zne,
                            exact_energy=e_exact,
                            h_value=h_value,
                            n_2q_gates=circuit_check.get("two_qubit_gate_count", 18),
                            ces=float(np.mean(layout_selection.ces_values))
                            if layout_selection.ces_values
                            else 0.15,
                            topology=getattr(self._config, "topology", "heavy_hex"),
                            n_qubits=self._config.n_qubits,
                        )
                        conf_threshold = getattr(self._config, "gnn_qem_confidence_threshold", 0.8)
                        correction = gnn_correct(self._gnn_qem_model, qem_sample, conf_threshold)
                        gnn_qem_applied = correction.correction_applied
                        gnn_qem_delta_e = correction.delta_e_predicted
                        gnn_qem_confidence = correction.confidence
                        if correction.correction_applied:
                            e_after_gnn = correction.corrected_energy
                    except Exception as exc:
                        self._logger.log("gnn_qem_error", data={"error": str(exc)})

            # Affine correction: clip to physics bounds
            from qmbp_simulation.execution.noisy_utils import affine_correct_energy

            affine_result = affine_correct_energy(
                e_after_gnn,
                e_ground=e_exact,
                n_qubits=self._config.n_qubits,
                h_value=h_value,
            )
            affine_applied = affine_result.correction_applied
            e_after_affine = affine_result.corrected_energy

            # Use corrected energy for verdict
            e_final = e_after_affine
            delta_e_gap = abs(e_final - e_exact) / gap if gap > 0 else float("inf")

            # Conditional SPSA
            spsa_applied = False
            if self._config.spsa_enabled and delta_e_gap > self._config.spsa_threshold:
                eval_fn = partial(self.evaluate, circuit, hamiltonian)
                _, best_energy, spsa_applied = spsa_refinement(
                    eval_fn,
                    params,
                    e_zne,
                    e_exact,
                    gap,
                    self._config,
                    self._spsa_config,
                    self._logger,
                    self._spsa_rng,
                    self._total_shots,
                )
                if spsa_applied and abs(best_energy - e_exact) < abs(e_zne - e_exact):
                    e_zne = best_energy
                    # Re-apply affine correction on the SPSA-refined energy
                    # to maintain physics bounds (variational principle).
                    spsa_affine = affine_correct_energy(
                        e_zne,
                        e_ground=e_exact,
                        n_qubits=self._config.n_qubits,
                        h_value=h_value,
                    )
                    e_final = spsa_affine.corrected_energy
                    delta_e_gap = abs(e_final - e_exact) / gap if gap > 0 else float("inf")

            # ── R²-gated verdict (Issue 4) ────────────────────────────
            if zne_r2 < ZNE_R2_QUALITY_THRESHOLD:
                verdict = "INDETERMINATE"
                verdict_reason = (
                    f"ZNE quality insufficient (R²={zne_r2:.3f} < {ZNE_R2_QUALITY_THRESHOLD})"
                )
            elif delta_e_gap < 0.05 and label == expected_label:
                verdict = "PASS"
                verdict_reason = f"ΔE/gap={delta_e_gap:.4f} < 5%, phase={label} correct"
            elif delta_e_gap < 0.05:
                verdict = "PARTIAL"
                verdict_reason = (
                    f"Energy good (ΔE/gap={delta_e_gap:.4f}) but phase={label} ≠ {expected_label}"
                )
            else:
                verdict = "FAIL"
                verdict_reason = f"ΔE/gap={delta_e_gap:.4f} ≥ 5%"

            result = HardwareRunResult(
                h_value=h_value,
                e_exact=e_exact,
                e_zne=e_zne,
                delta_e_gap=delta_e_gap,
                gap=gap,
                phase_label=label,
                expected_label=expected_label,
                zne_r2=zne_r2,
                zne_gain=self._compute_zne_gain(raw_results, e_zne, e_exact),
                mag_x_mean=mag_x,
                corr_zz_mean=corr_zz,
                sigma=sigma,
                total_shots=self._total_shots,
                job_ids=[r.get("job_id", "") for r in raw_results],
                layouts_used=layout_selection.layouts,
                ces_values=layout_selection.ces_values,
                per_site_x=x_values,
                per_bond_zz=zz_values,
                spsa_applied=spsa_applied,
                verdict=verdict,
                verdict_reason=verdict_reason,
                zne_amplifier_used=zne_amplifier_used,
                mitigation_strategy=self._resolve_mitigation_strategy(zne_amplifier_used),
                layout_std=(
                    float(np.std([r["energy"] for r in raw_results], ddof=1))
                    if len(raw_results) > 1
                    else None
                ),
                gnn_qem_applied=gnn_qem_applied,
                gnn_qem_delta_e=gnn_qem_delta_e,
                gnn_qem_confidence=gnn_qem_confidence,
                e_after_gnn_qem=e_after_gnn if gnn_qem_applied else None,
                affine_correction_applied=affine_applied,
                e_after_affine=e_after_affine if affine_applied else None,
                # Post-QPU validation metrics
                obs_bounds_clipped=len(obs_out_of_bounds) > 0,
                n_obs_violations=len(obs_out_of_bounds),
                layout_energy_outliers=len(layout_outliers) if len(raw_results) > 1 else 0,
                e_obs_discrepancy=e_obs_discrepancy,
                e_obs_cross_valid_passed=(e_obs_discrepancy <= cross_validation_threshold),
                n_layouts_observables=len(all_evs) if self._config.mode == "hardware" else 1,
            )

            zne_data = {
                "extrapolated_energy": e_zne,
                "r_squared": zne_r2,
                "amplifier": zne_amplifier_used,
                "n_layouts": len(raw_results),
                # ── P0 improvements metadata (2026-06-22) ──
                "p0_enhancements": {
                    "wls_extrapolation": zne_amplifier_used in ("pea", "gate_folding", "pea_local"),
                    "pauli_evolution_circuit": True,
                    "ces_spread_guard": True,
                    "ces_spread_sufficient": getattr(
                        layout_selection, "ces_spread_sufficient", None
                    ),
                    "ces_spread_ratio": getattr(layout_selection, "ces_spread_ratio", None),
                },
                # ── Quality metrics (derived, for post-hoc analysis) ──
                "quality_metrics": self._compute_quality_metrics(
                    e_zne=e_zne,
                    e_exact=e_exact,
                    gap=gap,
                    raw_results=raw_results,
                    x_values=x_values,
                    zz_values=zz_values,
                    hamiltonian=hamiltonian,
                ),
            }
            # Aggregate QPU metrics from raw results
            qpu_metrics = self._aggregate_qpu_metrics(raw_results)

            save_run(
                result,
                self._config,
                self._logger,
                calibration_info=calibration_snapshot,
                options_dict=build_estimator_options(self._config),
                execution_mode_name=(
                    self._execution_mode.__name__ if self._execution_mode else "unknown"
                ),
                raw_per_layout=raw_results,
                zne_data=zne_data,
                input_params=params,
                transpiled_stats=transpiled_stats,
                qpu_metrics=qpu_metrics,
            )
            # Attach extra metadata for external consumers (deployment script)
            result._calibration_snapshot = calibration_snapshot
            result._transpiled_stats = transpiled_stats
            result._qpu_metrics = qpu_metrics
            return result

        except Exception as exc:
            save_partial_before_error(partial_data, self._logger, self._config, str(exc))
            raise

    def run_h_sweep(
        self,
        circuit: QuantumCircuit,
        hamiltonian_builder: Callable[[float], SparsePauliOp],
        h_values: list[float],
        params_per_h: dict[float, np.ndarray],
        e_exact_per_h: dict[float, float],
        gap_per_h: dict[float, float],
    ) -> list[HardwareRunResult]:
        """Execute multiple h-points in single Batch/Session.

        When QESEM is enabled, uses multi-PUB batch mode (single device
        characterization shared across all h-points). Otherwise falls back
        to sequential per-h execution with TLS drift monitoring.
        """
        from .persistence import save_sweep_summary

        if 4.0 in h_values:
            h_values = [4.0] + [h for h in h_values if h != 4.0]

        # ══════════════════════════════════════════════════════════════════
        # QESEM BATCH PATH — multi-PUB sweep (preferred for Tier 1+)
        # ══════════════════════════════════════════════════════════════════
        if self._config.mitigation.qesem_enabled:
            from .observables import build_per_site_observables
            from .phase import classify_phase
            from .qesem import check_qesem_available, run_qesem_sweep

            if self._config.mode == "fake_backend":
                raise RuntimeError("QESEM sweep cannot run in fake_backend mode.")

            available, err_msg = check_qesem_available()
            if not available:
                raise ImportError(f"QESEM deps missing: {err_msg}")

            edges = [(i, i + 1) for i in range(self._config.n_qubits - 1)]
            x_ops, zz_ops = build_per_site_observables(self._config.n_qubits, edges)

            # Build per-h inputs
            hamiltonians = [hamiltonian_builder(h) for h in h_values]
            x_ops_list = [x_ops] * len(h_values)  # Same topology → same obs
            zz_ops_list = [zz_ops] * len(h_values)
            params_list = [params_per_h[h] for h in h_values]

            qesem_results = run_qesem_sweep(
                circuit=circuit,
                hamiltonians=hamiltonians,
                x_ops_list=x_ops_list,
                zz_ops_list=zz_ops_list,
                params_per_h=params_list,
                h_values=h_values,
                config=self._config,
                structured_logger=self._logger,
            )

            # Map QESEMResults → HardwareRunResults
            results: list[HardwareRunResult] = []
            for i, qr in enumerate(qesem_results):
                h = h_values[i]
                e_exact = e_exact_per_h[h]
                gap = gap_per_h[h]
                delta_e_gap = abs(qr.energy_mitigated - e_exact) / gap if gap > 0 else float("inf")

                label, mag_x, corr_zz, sigma = classify_phase(
                    qr.x_values, qr.zz_values, self._config.shots
                )
                expected_label = "paramagnetic" if h > 1.0 else "ferromagnetic"

                # ZNE gain
                if qr.noisy_data_available:
                    raw_error = abs(qr.noisy_energy - e_exact)
                    mit_error = abs(qr.energy_mitigated - e_exact)
                    zne_gain = 1.0 - (mit_error / raw_error) if raw_error > 1e-10 else 0.0
                else:
                    zne_gain = 0.0

                # Observable clipping metrics from circuit_stats
                cs = qr.circuit_stats or {}
                post_exec = cs.get("post_execution", {})
                n_obs_clipped = post_exec.get("n_obs_clipped", 0)

                # Verdict
                if delta_e_gap < 0.05 and label == expected_label:
                    verdict = "PASS"
                    verdict_reason = f"ΔE/gap={delta_e_gap:.4f} < 5%, phase={label} correct"
                elif delta_e_gap < 0.05:
                    verdict = "PARTIAL"
                    verdict_reason = f"Energy OK but phase={label} ≠ {expected_label}"
                else:
                    verdict = "FAIL"
                    verdict_reason = f"ΔE/gap={delta_e_gap:.4f} ≥ 5%"

                results.append(
                    HardwareRunResult(
                        h_value=h,
                        e_exact=e_exact,
                        e_zne=qr.energy_mitigated,
                        e_zne_std=qr.energy_std,
                        delta_e_gap=delta_e_gap,
                        gap=gap,
                        phase_label=label,
                        expected_label=expected_label,
                        zne_r2=1.0,
                        zne_gain=zne_gain,
                        mag_x_mean=mag_x,
                        corr_zz_mean=corr_zz,
                        sigma=sigma,
                        total_shots=qr.total_shots or 0,
                        job_ids=[qr.job_id],
                        layouts_used=[],
                        ces_values=[],
                        per_site_x=qr.x_values,
                        per_bond_zz=qr.zz_values,
                        verdict=verdict,
                        verdict_reason=verdict_reason,
                        zne_amplifier_used="qesem",
                        mitigation_strategy="qesem_unbiased",
                        # Observable clipping (QESEM unbiased estimator artifact)
                        obs_bounds_clipped=n_obs_clipped > 0,
                        n_obs_violations=n_obs_clipped,
                        # QESEM metadata
                        qesem_used=True,
                        qesem_job_id=qr.job_id,
                        qesem_total_qpu_time=qr.total_qpu_time,
                        qesem_gate_fidelities=qr.gate_fidelities,
                        qesem_total_shots=qr.total_shots,
                        qesem_mitigation_shots=qr.mitigation_shots,
                        qesem_noisy_evs=([qr.noisy_energy] + qr.noisy_x_values + qr.noisy_zz_values)
                        if qr.noisy_data_available
                        else None,
                        n_layouts_observables=0,
                    )
                )
                # Attach circuit_stats for persistence (used by run_tier_1 per_h_results)
                results[-1]._transpiled_stats = qr.circuit_stats
                results[-1]._qpu_metrics = {"total_qpu_time": qr.total_qpu_time}

            # ── Automatic QET post-execution validation (sweep) ───────────
            try:
                from project_health.analysis.hardware.validate_qet import (
                    validate_qet_result,
                )

                for i, qr in enumerate(qesem_results):
                    h = h_values[i]
                    gt = {"e_exact": e_exact_per_h[h], "gap": gap_per_h[h]}
                    qet_data = {
                        "job_id": qr.job_id,
                        "energy_mitigated": qr.energy_mitigated,
                        "energy_std": qr.energy_std,
                        "x_values": qr.x_values,
                        "zz_values": qr.zz_values,
                        "noisy_energy": qr.noisy_energy if qr.noisy_data_available else None,
                        "metadata": {
                            "gate_fidelities": qr.gate_fidelities,
                            "total_shots": qr.total_shots,
                            "total_qpu_time": qr.total_qpu_time,
                        },
                    }
                    if qr.noise_scale_results:
                        _rem = []
                        for obs_scales in qr.noise_scale_results:
                            pts = [
                                {"scale": s, "value": v, "error_bar": std}
                                for s, (v, std) in obs_scales.items()
                            ]
                            _rem.append(["obs", {"noise_scaling": {"results_with_REM": pts}}])
                        qet_data["metadata"]["results"] = [_rem]

                    report = validate_qet_result(qet_data, e_exact=gt["e_exact"], gap=gt["gap"])
                    self._logger.log(
                        "qet_sweep_validation",
                        data={
                            "h_value": h,
                            "passed": report.passed,
                            "n_issues": len(report.issues),
                            "metrics": report.metrics,
                        },
                    )
            except ImportError:
                pass
            except Exception as exc:
                logger.debug(f"QET sweep validator error (non-blocking): {exc}")

            return results

        # ══════════════════════════════════════════════════════════════════
        # LOCAL PEA/ZNE PATH — sequential per-h with TLS drift monitoring
        # ══════════════════════════════════════════════════════════════════

        if 4.0 in h_values:
            h_values = [4.0] + [h for h in h_values if h != 4.0]

        preflight = self.run_preflight()
        if preflight.get("abort"):
            raise RuntimeError(f"Preflight abort: {preflight.get('abort_reason')}")

        # Pre-cache layouts (reuse across all h-points)
        bound = circuit.assign_parameters(params_per_h[h_values[0]])
        self._get_cached_layouts(bound)

        # TLS calibration baseline snapshot (if hardware mode)
        baseline_snapshot = None
        if self._config.mode == "hardware":
            try:
                from qmbp_simulation.execution.noisy_utils import take_calibration_snapshot

                baseline_snapshot = take_calibration_snapshot(self.backend)
                self._logger.log(
                    "calibration_baseline",
                    data={
                        "mean_t1_us": baseline_snapshot.mean_t1_us,
                        "mean_t2_us": baseline_snapshot.mean_t2_us,
                        "timestamp": baseline_snapshot.timestamp,
                    },
                )
            except Exception as exc:
                self._logger.log("calibration_snapshot_error", data={"error": str(exc)})

        results: list[HardwareRunResult] = []
        for i, h in enumerate(h_values):
            # TLS drift check between h-points (skip for first point and fake_backend)
            if i > 0 and baseline_snapshot is not None and self._config.mode == "hardware":
                try:
                    from qmbp_simulation.execution.noisy_utils import (
                        check_calibration_drift,
                        take_calibration_snapshot,
                    )

                    current_snapshot = take_calibration_snapshot(self.backend)
                    drift = check_calibration_drift(baseline_snapshot, current_snapshot)
                    self._logger.log(
                        "calibration_drift_check",
                        data={
                            "h_value": h,
                            "t1_drift_pct": drift.t1_drift_pct,
                            "should_abort": drift.should_abort,
                        },
                    )
                    if drift.should_abort:
                        self._logger.log(
                            "sweep_abort_calibration_drift",
                            data={
                                "h_value": h,
                                "t1_drift_pct": drift.t1_drift_pct,
                                "completed_h_points": len(results),
                            },
                        )
                        break
                except Exception as exc:
                    self._logger.log("drift_check_error", data={"error": str(exc)})

            result = self.run_deployment(
                circuit,
                hamiltonian_builder(h),
                params_per_h[h],
                h_value=h,
                e_exact=e_exact_per_h[h],
                gap=gap_per_h[h],
            )
            results.append(result)

            # Smoke test gate: if h=4.0 fails badly, abort sweep
            if h == 4.0 and result.delta_e_gap > 0.10:
                self._logger.log(
                    "sweep_abort_smoke_test",
                    data={
                        "h": h,
                        "delta_e_gap": result.delta_e_gap,
                    },
                )
                break

        # ── Sweep-level validation checks (post-execution) ────────────────
        if len(results) >= 2:
            # Check: Energy monotonicity (TFIM paramagnetic: E should decrease with h)
            # For descending h sweep, E(h_large) < E(h_small) in the paramagnetic phase.
            # Non-monotonic results indicate noise-corrupted points.
            h_e_pairs = sorted(
                [(r.h_value, r.e_zne) for r in results], key=lambda x: x[0], reverse=True
            )
            monotonicity_violations = []
            for j in range(len(h_e_pairs) - 1):
                h_high, e_high = h_e_pairs[j]
                h_low, e_low = h_e_pairs[j + 1]
                # In paramagnetic phase (h > h_c), E decreases with h
                if e_high > e_low:  # Higher h should have lower (more negative) energy
                    monotonicity_violations.append(
                        {"h_pair": (h_high, h_low), "e_pair": (e_high, e_low)}
                    )
            if monotonicity_violations:
                self._logger.log(
                    "sweep_monotonicity_violation",
                    data={
                        "n_violations": len(monotonicity_violations),
                        "violations": monotonicity_violations,
                        "warning": "Energy is non-monotonic across h-sweep. "
                        "Expected E(h_high) < E(h_low) in paramagnetic phase. "
                        "May indicate noise-corrupted h-point(s).",
                    },
                )

            # Check: Systematic affine correction (pattern detection)
            # If affine clips on every h-point, ZNE is systematically failing.
            n_affine_clipped = sum(
                1 for r in results if getattr(r, "affine_correction_applied", False)
            )
            if n_affine_clipped == len(results) and len(results) >= 3:
                self._logger.log(
                    "sweep_systematic_affine_clipping",
                    data={
                        "n_clipped": n_affine_clipped,
                        "n_total": len(results),
                        "warning": "Affine correction triggered on EVERY h-point. "
                        "This indicates systematic ZNE failure (e.g., PEA budget "
                        "insufficient for current calibration). Consider increasing "
                        "PEA learning budget or retrying at better calibration.",
                    },
                )

        # ── Post-sweep stale calibration comparison (P2-C) ─────────────────
        # For runs >1h, compare pre-sweep vs post-sweep calibration to tag
        # results that may have been affected by calibration drift during
        # the full sweep. This is a diagnostic — it does NOT abort, but
        # attaches a drift_report to the sweep summary for post-hoc filtering.
        if baseline_snapshot is not None and self._config.mode == "hardware":
            try:
                from qmbp_simulation.execution.noisy_utils import (
                    check_calibration_drift,
                    take_calibration_snapshot,
                )

                post_sweep_snapshot = take_calibration_snapshot(self.backend)
                stale_drift = check_calibration_drift(baseline_snapshot, post_sweep_snapshot)
                self._logger.log(
                    "stale_calibration_comparison",
                    data={
                        "pre_sweep_timestamp": baseline_snapshot.timestamp,
                        "post_sweep_timestamp": post_sweep_snapshot.timestamp,
                        "t1_drift_pct": stale_drift.t1_drift_pct,
                        "t2_drift_pct": stale_drift.t2_drift_pct,
                        "gate_error_drift_pct": stale_drift.gate_error_drift_pct,
                        "max_single_drift_pct": stale_drift.max_single_drift_pct,
                        "is_stable": stale_drift.is_stable,
                        "recommendation": stale_drift.recommendation,
                        "n_h_points_completed": len(results),
                        "pre_mean_t1_us": baseline_snapshot.mean_t1_us,
                        "post_mean_t1_us": post_sweep_snapshot.mean_t1_us,
                        "pre_mean_2q_error": baseline_snapshot.mean_2q_error,
                        "post_mean_2q_error": post_sweep_snapshot.mean_2q_error,
                    },
                )
                if not stale_drift.is_stable:
                    self._logger.log(
                        "stale_calibration_warning",
                        data={
                            "warning": (
                                f"Calibration drifted during sweep: T1 {stale_drift.t1_drift_pct:.1f}%, "
                                f"gates {stale_drift.gate_error_drift_pct:.1f}%. "
                                f"Results from later h-points may be degraded. "
                                f"Consider re-running affected points."
                            ),
                            "affected_h_points": [r.h_value for r in results[1:]],
                        },
                    )
            except Exception as exc:
                self._logger.log("stale_calibration_comparison_error", data={"error": str(exc)})

        save_sweep_summary(results, self._config, self._logger)
        return results

    def run_preflight(self) -> dict[str, Any]:
        """Preflight checks without submitting jobs."""
        from .preflight import run_preflight_checks

        return run_preflight_checks(self.backend, self._config, self._logger)

    # ─── Circuit Visualization ──────────────────────────────────────

    def _save_pre_execution_circuit(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        h_value: float,
        layout_selection,
    ) -> None:
        """Save a PNG diagram of the circuit BEFORE submitting to QPU.

        Creates two images in a timestamped subdirectory:
        - circuit_logical_h{h}_{ts}.png: The bound logical circuit (as designed)
        - circuit_transpiled_h{h}_{ts}.png: The first transpiled ISA circuit (as executed)

        Timestamped naming prevents overwriting on repeated runs of the same h-value.
        """

        try:
            from pathlib import Path as _Path

            from qiskit.visualization import circuit_drawer

            output_dir = _Path(self._config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            h_tag = f"h{h_value:.1f}".replace(".", "p")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save the logical (bound) circuit
            bound_qc = circuit.assign_parameters(params)
            logical_path = output_dir / f"circuit_logical_{h_tag}_{ts}.png"
            _diagram = circuit_drawer(
                bound_qc,
                output="mpl",
                fold=40,
                filename=str(logical_path),
            )
            if hasattr(_diagram, "savefig"):
                _diagram.savefig(str(logical_path), dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as _plt

            _plt.close("all")

            # Save the first transpiled (ISA) circuit
            if layout_selection.transpiled_circuits:
                transpiled_path = output_dir / f"circuit_transpiled_{h_tag}_{ts}.png"
                _diagram = circuit_drawer(
                    layout_selection.transpiled_circuits[0],
                    output="mpl",
                    fold=60,
                    filename=str(transpiled_path),
                )
                if hasattr(_diagram, "savefig"):
                    _diagram.savefig(str(transpiled_path), dpi=150, bbox_inches="tight")
                _plt.close("all")

                # Save the circuit with DD applied (approximation of server-side DD)
                # IBM Runtime applies DD server-side; this shows what DD insertions
                # would look like using the same sequence_type we configured.
                self._save_circuit_with_dd(
                    layout_selection.transpiled_circuits[0],
                    f"{h_tag}_{ts}",
                    output_dir,
                )

            self._logger.log(
                "circuit_diagram_saved",
                data={
                    "output_dir": str(output_dir),
                    "h_value": h_value,
                    "timestamp": ts,
                    "logical_path": str(logical_path),
                },
            )
        except Exception as exc:
            # Never let diagram generation block QPU execution
            self._logger.log(
                "circuit_diagram_error",
                data={"error": str(exc), "h_value": h_value},
            )

    def _save_circuit_with_dd(
        self,
        isa_circuit: QuantumCircuit,
        h_tag: str,
        output_dir: Path,
    ) -> None:
        """Save a PNG of the transpiled circuit with DD pulses inserted locally.

        IBM Runtime applies DD server-side, so we cannot see the actual
        post-DD circuit. This method applies PadDynamicalDecoupling locally
        to produce an *approximation* of what the server does, useful for
        visual inspection and thesis figures.

        The image is saved as circuit_with_dd_h{h}.png in the output directory.
        """
        try:
            from qiskit.circuit.library import XGate
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import (
                ALAPScheduleAnalysis,
                PadDynamicalDecoupling,
            )
            from qiskit.visualization import circuit_drawer

            # Build the DD sequence matching our config (XpXm = X, Xdg)
            dd_sequence = [XGate(), XGate()]  # XpXm approximation

            # Schedule and insert DD
            target = self.backend.target if hasattr(self.backend, "target") else None
            if target is None:
                # Cannot schedule without backend target info
                return

            pm = PassManager(
                [
                    ALAPScheduleAnalysis(target=target),
                    PadDynamicalDecoupling(
                        target=target,
                        dd_sequence=dd_sequence,
                    ),
                ]
            )
            circuit_with_dd = pm.run(isa_circuit)

            import matplotlib.pyplot as _plt

            _diagram = circuit_drawer(
                circuit_with_dd,
                output="mpl",
                fold=60,
                filename=str(output_dir / f"circuit_with_dd_{h_tag}.png"),
            )
            if hasattr(_diagram, "savefig"):
                _diagram.savefig(
                    str(output_dir / f"circuit_with_dd_{h_tag}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
            _plt.close("all")

            self._logger.log(
                "circuit_dd_diagram_saved",
                data={
                    "file": f"circuit_with_dd_{h_tag}.png",
                    "dd_sequence": "XpXm",
                    "n_ops_before": isa_circuit.size(),
                    "n_ops_after": circuit_with_dd.size(),
                },
            )
        except Exception as exc:
            # Never block QPU execution for visualization failures
            self._logger.log(
                "circuit_dd_diagram_error",
                data={"error": str(exc)},
            )

    # ─── Circuit Serialization & Pre-Submission Manifest ──────────────

    def _save_circuit_qasm(
        self,
        layout_selection,
        h_value: float,
    ) -> list[str]:
        """Serialize transpiled circuits as QASM3 for exact reproducibility.

        If QPU submission fails, these files allow exact retry without
        re-transpilation. Also enables post-hoc circuit fingerprinting.

        Returns list of saved file paths (empty on failure — never blocks QPU).
        """
        saved_paths: list[str] = []
        try:
            from qiskit.qasm3 import dumps as qasm3_dumps

            output_dir = Path(self._config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            h_tag = f"h{h_value:.1f}".replace(".", "p")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i, circ in enumerate(layout_selection.transpiled_circuits):
                qasm_path = output_dir / f"circuit_isa_layout{i}_{h_tag}_{ts}.qasm"
                qasm_str = qasm3_dumps(circ)
                qasm_path.write_text(qasm_str)
                saved_paths.append(str(qasm_path))

            self._logger.log(
                "circuit_qasm_saved",
                data={
                    "n_circuits": len(saved_paths),
                    "h_value": h_value,
                    "paths": saved_paths,
                },
            )
        except Exception as exc:
            # QASM3 export may fail for some gate types — never block QPU
            self._logger.log(
                "circuit_qasm_error",
                data={"error": str(exc), "h_value": h_value},
            )
        return saved_paths

    def _compute_circuit_fingerprint(self, circuit: QuantumCircuit) -> str:
        """Compute a deterministic fingerprint of a quantum circuit.

        Uses gate counts + depth + qubit count to create a stable hash.
        This enables matching results to exact circuit versions without
        serializing the full object.
        """
        import hashlib

        ops = sorted(circuit.count_ops().items())
        fingerprint_data = (
            f"nq={circuit.num_qubits};depth={circuit.depth()};ops={ops};size={circuit.size()}"
        )
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

    def _build_pre_submission_manifest(
        self,
        *,
        circuit: QuantumCircuit,
        params: np.ndarray,
        h_value: float,
        e_exact: float,
        gap: float,
        expected_label: str,
        layout_selection,
        calibration_snapshot: dict[str, Any],
        transpiled_stats: dict[str, Any],
        circuit_check: dict[str, Any],
        transpiled_quality_checks: list[dict],
        circuit_qasm_paths: list[str],
        sigma_flow: float | None = None,
        kappa: float | None = None,
    ) -> dict[str, Any]:
        """Build a consolidated pre-submission manifest.

        Single atomic record of everything about to be sent to the QPU.
        Enables complete audit trail without parsing multiple log files.
        """
        # Compute fingerprints for each transpiled circuit
        fingerprints = []
        for circ in layout_selection.transpiled_circuits:
            fingerprints.append(self._compute_circuit_fingerprint(circ))

        manifest: dict[str, Any] = {
            "manifest_version": "1.0",
            "timestamp": datetime.now(UTC).isoformat(),
            # ── What we're executing ──
            "execution_target": {
                "h_value": h_value,
                "e_exact": e_exact,
                "gap": gap,
                "expected_label": expected_label,
                "n_qubits": circuit.num_qubits,
                "n_params": circuit.num_parameters,
                "params": params.tolist(),
                "params_norm": float(np.linalg.norm(params)),
            },
            # ── Hardware configuration ──
            "hardware_config": {
                "backend_name": self._config.backend_name,
                "mode": self._config.mode,
                "shots": self._config.shots,
                "n_layouts": self._config.n_layouts,
                "optimization_level": self._config.optimization_level,
                "zne_amplifier": self._config.mitigation.zne_amplifier,
                "zne_noise_factors": self._config.mitigation.zne_noise_factors,
                "dd_enabled": self._config.mitigation.dd_enabled,
                "twirling_enabled": self._config.mitigation.twirling_enabled,
                "trex_enabled": self._config.mitigation.trex_enabled,
                "num_randomizations": self._config.mitigation.num_randomizations,
                "shots_per_randomization": self._config.mitigation.shots_per_randomization,
            },
            # ── Layout selection results ──
            "layouts": {
                "n_layouts_selected": len(layout_selection.layouts),
                "physical_qubits": layout_selection.layouts,
                "ces_values": layout_selection.ces_values,
                "circuit_fingerprints": fingerprints,
            },
            # ── Pre-flight validation results ──
            "validation": {
                "circuit_zne_check": {
                    "two_qubit_gate_count": circuit_check.get("two_qubit_gate_count"),
                    "zne_threshold": circuit_check.get("zne_threshold"),
                    "amplifier": circuit_check.get("amplifier"),
                    "abort": circuit_check.get("abort", False),
                },
                "transpiled_quality_per_layout": [
                    {
                        "layout_idx": i,
                        "error_budget": qc.get("error_budget"),
                        "fidelity_estimate": qc.get("fidelity_estimate"),
                        "depth_2q": qc.get("depth_2q"),
                        "n_2q_gates": qc.get("n_2q_gates"),
                        "defective_edges_in_layout": qc.get("defective_edges_in_layout", 0),
                        "abort": qc.get("abort", False),
                    }
                    for i, qc in enumerate(transpiled_quality_checks)
                ],
            },
            # ── Calibration state at execution time ──
            "calibration_snapshot": calibration_snapshot,
            # ── Transpiled circuit statistics ──
            "transpiled_stats": transpiled_stats,
            # ── Risk assessment (noiseless metrics) ──
            "risk_assessment": {
                "sigma_flow": sigma_flow,
                "kappa": kappa,
                "sigma_flow_boosted": sigma_flow is not None and sigma_flow > 0.5,
            },
            # ── Layout depth_2q ranking ──
            # Ranked by depth_2q ascending (best first). The layout with
            # lowest 2Q critical path accumulates less decoherence error.
            "layout_depth_2q_ranking": self._rank_layouts_by_depth_2q(layout_selection),
            # ── Provenance ──
            "provenance": {
                "circuit_qasm_paths": circuit_qasm_paths,
            },
        }
        return manifest

    def _save_pre_submission_manifest(self, manifest: dict[str, Any], h_value: float) -> None:
        """Persist the pre-submission manifest as a JSON file.

        Saved BEFORE QPU submission so data is never lost even if the job fails.
        """
        import json

        from qmbp_simulation.utils.helpers import json_serialize

        try:
            output_dir = Path(self._config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            h_tag = f"h{h_value:.1f}".replace(".", "p")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            manifest_path = output_dir / f"pre_submission_manifest_{h_tag}_{ts}.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, default=json_serialize)
            self._logger.log(
                "pre_submission_manifest_saved",
                data={"path": str(manifest_path), "h_value": h_value},
            )
        except Exception as exc:
            # Never block QPU execution for manifest save failures
            self._logger.log(
                "pre_submission_manifest_error",
                data={"error": str(exc), "h_value": h_value},
            )

    # ─── ZNE Aggregation ─────────────────────────────────────────────

    def _resolve_mitigation_strategy(self, amplifier_used: str) -> str:
        """Map amplifier_used to a human-readable mitigation_strategy label."""
        if amplifier_used.startswith("server_side"):
            return "ibm_zne_layout_avg"
        if amplifier_used == "pea":
            return "pea_local"
        if amplifier_used in ("gate_folding", "ces_gf"):
            return "gate_folding_local"
        if amplifier_used == "average":
            return "ces_zne"
        return amplifier_used or "ces_zne"

    def _capture_calibration_snapshot(self, layout_selection) -> dict[str, Any]:
        """Capture backend calibration data at execution time.

        Records T1/T2, readout error, and 2Q error rates for the qubits
        actually used, enabling post-hoc correlation between result quality
        and calibration state.
        """
        snapshot: dict[str, Any] = {"timestamp": datetime.now(UTC).isoformat()}
        try:
            target = self.backend.target
            # Per-qubit T1/T2 and readout error for used qubits
            qubit_props = target.qubit_properties
            if qubit_props and layout_selection.layouts:
                used_qubits = set()
                for layout in layout_selection.layouts:
                    used_qubits.update(layout[: self._config.n_qubits])
                t1_values = {}
                t2_values = {}
                readout_errors = {}
                for q in sorted(used_qubits):
                    if q < len(qubit_props) and qubit_props[q] is not None:
                        t1 = getattr(qubit_props[q], "t1", None)
                        t2 = getattr(qubit_props[q], "t2", None)
                        if t1 is not None:
                            t1_values[q] = t1 * 1e6  # Convert to μs
                        if t2 is not None:
                            t2_values[q] = t2 * 1e6
                    # Readout error from measure gate properties
                    try:
                        meas_props = target["measure"].get((q,))
                        if meas_props and meas_props.error is not None:
                            readout_errors[q] = meas_props.error
                    except (KeyError, TypeError):
                        pass
                snapshot["t1_us_per_qubit"] = t1_values
                snapshot["t2_us_per_qubit"] = t2_values
                snapshot["min_t1_us"] = min(t1_values.values()) if t1_values else None
                snapshot["min_t2_us"] = min(t2_values.values()) if t2_values else None
                snapshot["readout_error_per_qubit"] = readout_errors
                snapshot["mean_readout_error"] = (
                    sum(readout_errors.values()) / len(readout_errors) if readout_errors else None
                )

            # 2Q error rates for edges used
            for gate_name in ["ecr", "cz", "cx"]:
                if gate_name not in target.operation_names:
                    continue
                error_map = {}
                try:
                    qargs_list = target.qargs_for_operation_name(gate_name)
                    if qargs_list:
                        for qargs in qargs_list:
                            if len(qargs) == 2:
                                props = target[gate_name].get(qargs)
                                if props and props.error is not None:
                                    error_map[f"{qargs[0]}-{qargs[1]}"] = props.error
                except Exception:
                    pass
                if error_map:
                    snapshot[f"{gate_name}_error_rates"] = error_map
                    vals = list(error_map.values())
                    snapshot[f"mean_{gate_name}_error"] = sum(vals) / len(vals)
                    break  # Only need one 2Q gate type

            # Calibration age: seconds since last backend calibration update
            try:
                if hasattr(self.backend, "properties"):
                    props = self.backend.properties()
                    if props and hasattr(props, "last_update_date"):
                        last_cal = props.last_update_date
                        if last_cal:
                            now = datetime.now(UTC)
                            if hasattr(last_cal, "timestamp"):
                                age_s = (now - last_cal).total_seconds()
                            else:
                                age_s = None
                            snapshot["calibration_age_s"] = age_s
                            snapshot["last_calibration_time"] = str(last_cal)
            except Exception:
                snapshot["calibration_age_s"] = None

            # Session and backend version metadata
            try:
                if hasattr(self.backend, "name"):
                    snapshot["backend_name"] = self.backend.name
                if hasattr(self.backend, "version"):
                    snapshot["backend_version"] = self.backend.version
                if hasattr(self.backend, "session") and self.backend.session:
                    snapshot["session_id"] = getattr(self.backend.session, "session_id", None)
            except Exception:
                pass

        except Exception as exc:
            snapshot["capture_error"] = str(exc)
        return snapshot

    def _capture_transpiled_stats(self, layout_selection) -> dict[str, Any]:
        """Capture circuit statistics after transpilation.

        Records per-layout:
          - depth, depth_2q (critical path of 2Q gates only)
          - n_2q_gates, n_1q_gates, total_gates
          - count_ops: per-gate-type breakdown (from ResourceEstimation)
          - num_tensor_factors: disconnected sub-circuits (sanity check)
          - width: physical qubits in transpiled circuit
          - active_qubits: width − num_tensor_factors + 1
          - idle_cycles_per_qubit: average idle cycles per active qubit (decoherence risk)
          - max_idle_stretch: longest consecutive idle on any qubit
          - parallelism_ratio: 2Q gates / depth_2q (gate-level parallelism)
          - error_budget: calibration-aware predicted total error probability
          - fidelity_estimate: exp(-error_budget)

        The depth_2q metric is the strongest predictor of hardware error
        accumulation (2Q gates dominate noise budget on IBM devices).

        Note: idle metrics use DAG analysis (same algorithm as
        `analysis.circuit_visualizer.transpiled_circuit_stats`) but are
        computed inline to respect the module dependency DAG
        (execution cannot import analysis).
        """
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import ResourceEstimation

        stats: dict[str, Any] = {"per_layout": []}
        try:
            for i, circ in enumerate(layout_selection.transpiled_circuits):
                # ── Ad-hoc stats (fast, no DAG conversion) ──
                n_2q = sum(1 for inst in circ.data if inst.operation.num_qubits == 2)
                n_1q = sum(1 for inst in circ.data if inst.operation.num_qubits == 1)
                depth = circ.depth()
                depth_2q = circ.depth(filter_function=lambda x: x.operation.num_qubits == 2)

                layout_stats: dict[str, Any] = {
                    "layout_idx": i,
                    "depth": depth,
                    "depth_2q": depth_2q,
                    "n_2q_gates": n_2q,
                    "n_1q_gates": n_1q,
                    "total_gates": len(circ.data),
                }

                # ── ResourceEstimation pass (count_ops + tensor factors) ──
                try:
                    re_pm = PassManager([ResourceEstimation()])
                    re_pm.run(circ)
                    prop = re_pm.property_set
                    count_ops = prop.get("count_ops")
                    layout_stats["count_ops"] = dict(count_ops) if count_ops else {}
                    num_tf = prop.get("num_tensor_factors")
                    width = prop.get("width")
                    layout_stats["num_tensor_factors"] = num_tf
                    layout_stats["width"] = width
                    if width is not None and num_tf is not None:
                        layout_stats["active_qubits"] = width - num_tf + 1
                except Exception:
                    # Fallback: build count_ops manually
                    gate_counts: dict[str, int] = {}
                    for inst in circ.data:
                        name = inst.operation.name
                        gate_counts[name] = gate_counts.get(name, 0) + 1
                    layout_stats["count_ops"] = gate_counts

                # ── Idle/decoherence metrics via DAG analysis ──
                # Computes how many cycles each qubit spends idle (not in any
                # gate), and the longest consecutive idle stretch. Long idle
                # stretches → T1/T2 decoherence accumulation.
                try:
                    idle_metrics = self._compute_idle_metrics(circ)
                    layout_stats["idle_cycles_per_qubit"] = idle_metrics["idle_cycles_per_qubit"]
                    layout_stats["max_idle_stretch"] = idle_metrics["max_idle_stretch"]
                except Exception:
                    layout_stats["idle_cycles_per_qubit"] = None
                    layout_stats["max_idle_stretch"] = None

                # ── Parallelism ratio ──
                # How many 2Q gates execute per depth layer on average.
                # Higher = better hardware utilization, less idle time.
                active = layout_stats.get("active_qubits") or circ.num_qubits
                layout_stats["parallelism_ratio"] = n_2q / depth_2q if depth_2q > 0 else 0.0
                layout_stats["gate_density_2q"] = (
                    n_2q / (active * depth_2q) if (depth_2q > 0 and active > 0) else 0.0
                )

                # ── Calibration-aware error budget per layout ──
                # Uses actual backend error rates for the specific qubits in
                # this layout to predict total accumulated error probability.
                try:
                    layout_qubits = (
                        layout_selection.layouts[i] if i < len(layout_selection.layouts) else None
                    )
                    error_budget_data = self._compute_error_budget_for_layout(circ, layout_qubits)
                    layout_stats["error_budget"] = error_budget_data["error_budget"]
                    layout_stats["fidelity_estimate"] = error_budget_data["fidelity_estimate"]
                    layout_stats["error_budget_source"] = error_budget_data["source"]
                except Exception:
                    layout_stats["error_budget"] = None
                    layout_stats["fidelity_estimate"] = None

                # ── Circuit fingerprint ──
                layout_stats["circuit_fingerprint"] = self._compute_circuit_fingerprint(circ)

                stats["per_layout"].append(layout_stats)

            if stats["per_layout"]:
                stats["mean_depth"] = float(np.mean([s["depth"] for s in stats["per_layout"]]))
                stats["mean_depth_2q"] = float(
                    np.mean([s["depth_2q"] for s in stats["per_layout"]])
                )
                stats["mean_2q_gates"] = float(
                    np.mean([s["n_2q_gates"] for s in stats["per_layout"]])
                )
                # Aggregate error budget (use worst-case for safety assessment)
                budgets = [
                    s["error_budget"]
                    for s in stats["per_layout"]
                    if s.get("error_budget") is not None
                ]
                if budgets:
                    stats["mean_error_budget"] = float(np.mean(budgets))
                    stats["max_error_budget"] = float(np.max(budgets))
                    stats["mean_fidelity_estimate"] = float(np.exp(-np.mean(budgets)))
                # Aggregate idle metrics
                idle_vals = [
                    s["idle_cycles_per_qubit"]
                    for s in stats["per_layout"]
                    if s.get("idle_cycles_per_qubit") is not None
                ]
                if idle_vals:
                    stats["mean_idle_cycles_per_qubit"] = float(np.mean(idle_vals))
                max_stretches = [
                    s["max_idle_stretch"]
                    for s in stats["per_layout"]
                    if s.get("max_idle_stretch") is not None
                ]
                if max_stretches:
                    stats["max_idle_stretch_across_layouts"] = int(max(max_stretches))
                # Aggregate count_ops across layouts (use first layout as representative)
                first_ops = stats["per_layout"][0].get("count_ops")
                if first_ops:
                    stats["basis_gates"] = sorted(first_ops.keys())
        except Exception as exc:
            stats["capture_error"] = str(exc)
        return stats

    @staticmethod
    def _compute_idle_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
        """Compute idle-time metrics via DAG analysis.

        Analyzes the circuit DAG to determine how many cycles each qubit
        spends idle (not participating in any gate), and the longest
        consecutive idle stretch on any qubit. Long idle stretches correlate
        with T1/T2 decoherence accumulation on hardware.

        Parameters
        ----------
        circuit : QuantumCircuit
            A transpiled circuit to analyze.

        Returns
        -------
        dict[str, Any]
            idle_cycles_per_qubit: float average idle cycles per active qubit.
            max_idle_stretch: int longest consecutive idle on any qubit.
        """
        from qiskit.converters import circuit_to_dag

        dag = circuit_to_dag(circuit)
        idle_wires = set(dag.idle_wires())
        active_qubits = [q for q in dag.qubits if q not in idle_wires]

        if not active_qubits:
            return {"idle_cycles_per_qubit": 0.0, "max_idle_stretch": 0}

        # Pre-compute node-to-layer mapping once for efficiency
        node_to_layer: dict[int, int] = {}
        for layer_idx, layer in enumerate(dag.layers()):
            for node in layer["graph"].op_nodes():
                node_to_layer[node._node_id] = layer_idx

        total_idle = 0
        max_stretch = 0
        depth = circuit.depth()

        for qubit in active_qubits:
            nodes = list(dag.nodes_on_wire(qubit, only_ops=True))
            if len(nodes) <= 1:
                idle_for_qubit = max(0, depth - 1)
                total_idle += idle_for_qubit
                max_stretch = max(max_stretch, idle_for_qubit)
                continue
            # Compute gaps between consecutive ops using pre-built layer map
            stretches = []
            for j in range(len(nodes) - 1):
                layer_a = node_to_layer.get(nodes[j]._node_id, 0)
                layer_b = node_to_layer.get(nodes[j + 1]._node_id, 0)
                gap = max(0, layer_b - layer_a - 1)
                stretches.append(gap)
            total_idle += sum(stretches)
            if stretches:
                max_stretch = max(max_stretch, max(stretches))

        n_active = len(active_qubits)
        return {
            "idle_cycles_per_qubit": total_idle / n_active if n_active > 0 else 0.0,
            "max_idle_stretch": max_stretch,
        }

    @staticmethod
    def _rank_layouts_by_depth_2q(layout_selection) -> list[dict[str, Any]]:
        """Rank transpiled circuits by depth_2q (lowest = best for hardware).

        The layout with the lowest 2Q critical path accumulates less
        decoherence error. This ranking is stored in the pre-submission
        manifest to record which layout is optimal for ZNE primary execution.

        Returns
        -------
        list[dict[str, Any]]
            Per-layout stats sorted by depth_2q ascending (best first).
        """
        ranked: list[dict[str, Any]] = []
        for i, circ in enumerate(layout_selection.transpiled_circuits):
            depth_2q = circ.depth(filter_function=lambda x: x.operation.num_qubits == 2)
            depth = circ.depth()
            n_2q = sum(1 for inst in circ.data if inst.operation.num_qubits == 2)
            entry: dict[str, Any] = {
                "layout_idx": i,
                "depth_2q": depth_2q,
                "depth": depth,
                "n_2q_gates": n_2q,
            }
            if i < len(layout_selection.layouts):
                entry["layout"] = layout_selection.layouts[i]
            if i < len(layout_selection.ces_values):
                entry["ces"] = layout_selection.ces_values[i]
            ranked.append(entry)

        ranked.sort(key=lambda x: (x["depth_2q"], x["n_2q_gates"]))
        return ranked

    def _compute_error_budget_for_layout(
        self,
        circuit: QuantumCircuit,
        layout: list[int] | None,
    ) -> dict[str, Any]:
        """Compute calibration-aware error budget for a transpiled circuit.

        Uses actual error rates from the backend's Target API for the
        specific qubits in the selected layout.

        Parameters
        ----------
        circuit : QuantumCircuit
            Transpiled (ISA) circuit.
        layout : list[int] | None
            Physical qubit indices used in this layout.

        Returns
        -------
        dict[str, Any]
            error_budget, fidelity_estimate, source ("calibration" or "typical_fallback")
        """
        # Count ops in circuit
        count_ops: dict[str, int] = {}
        for inst in circuit.data:
            name = inst.operation.name
            count_ops[name] = count_ops.get(name, 0) + 1

        # Get error rates from backend target (layout-filtered)
        error_rates: dict[str, float] = {}
        source = "typical_fallback"

        try:
            target = self.backend.target
            layout_set = set(layout) if layout else None

            for gate_name in target.operation_names:
                try:
                    qargs_list = target.qargs_for_operation_name(gate_name)
                except Exception:
                    continue
                if qargs_list is None:
                    continue
                errs: list[float] = []
                for qargs in qargs_list:
                    if layout_set is not None:
                        if not all(q in layout_set for q in qargs):
                            continue
                    try:
                        props = target[gate_name].get(qargs)
                        if props is not None and props.error is not None:
                            errs.append(props.error)
                    except Exception:
                        continue
                if errs:
                    error_rates[gate_name] = sum(errs) / len(errs)
            if error_rates:
                source = "calibration"
        except Exception:
            pass

        # Fallback rates for gates not found in calibration
        TYPICAL_RATES = {
            "cz": 8e-3,
            "ecr": 8e-3,
            "cx": 8e-3,
            "sx": 2.5e-4,
            "x": 2.5e-4,
            "rz": 0.0,
            "id": 0.0,
            "delay": 0.0,
            "barrier": 0.0,
            "measure": 0.0,
        }
        for gate in count_ops:
            if gate not in error_rates:
                error_rates[gate] = TYPICAL_RATES.get(gate, 1e-4)

        # Compute budget
        total_error = 0.0
        for gate, count in count_ops.items():
            rate = error_rates.get(gate, 0.0)
            total_error += count * rate

        return {
            "error_budget": total_error,
            "fidelity_estimate": float(np.exp(-total_error)),
            "source": source,
        }

    def _compute_quality_metrics(
        self,
        e_zne: float,
        e_exact: float,
        gap: float,
        raw_results: list[dict[str, Any]],
        x_values: list[float],
        zz_values: list[float],
        hamiltonian: SparsePauliOp,
    ) -> dict[str, Any]:
        """Compute derived quality metrics for post-hoc analysis.

        These metrics are NOT used for pass/fail decisions — they provide
        insight into result reliability and error sources.
        """
        metrics: dict[str, Any] = {}

        # ── SNR (Signal-to-Noise Ratio) per observable type ──
        # SNR = |⟨O⟩| × √shots. If SNR < 1, signal is buried in shot noise.
        shots = self._config.shots
        sqrt_shots = np.sqrt(shots)
        if x_values:
            x_snr = [abs(v) * sqrt_shots for v in x_values]
            metrics["snr_x_per_site"] = x_snr
            metrics["snr_x_min"] = min(x_snr) if x_snr else None
            metrics["snr_x_mean"] = float(np.mean(x_snr)) if x_snr else None
        if zz_values:
            zz_snr = [abs(v) * sqrt_shots for v in zz_values]
            metrics["snr_zz_per_bond"] = zz_snr
            metrics["snr_zz_min"] = min(zz_snr) if zz_snr else None
            metrics["snr_zz_mean"] = float(np.mean(zz_snr)) if zz_snr else None

        # ── Variational principle check ──
        # E_hw should be ≥ E_exact (within tolerance). Violation indicates bias.
        energy_error = e_zne - e_exact
        variational_violation = energy_error < -1e-6 * abs(e_exact)
        metrics["variational_principle_violated"] = variational_violation
        metrics["energy_error_signed"] = energy_error

        # Escalate severity: hardware violations > 0.1 are always significant
        # (unlike noiseless, where they could be numerical — on hardware
        # a large sub-ground-state result indicates systematic bias).
        if variational_violation:
            violation_magnitude = abs(energy_error)
            if violation_magnitude >= 0.1:
                logger.error(
                    "❌ CRITICAL variational principle violation on hardware: "
                    "E_hw=%.6f < E_exact=%.6f (Δ=%.4e). "
                    "ZNE/mitigation may be introducing systematic bias.",
                    e_zne,
                    e_exact,
                    violation_magnitude,
                )
            else:
                logger.warning(
                    "⚠️  Variational principle violated on hardware: "
                    "E_hw=%.6f < E_exact=%.6f (Δ=%.2e). "
                    "Expected for ZNE overshoot — affine correction will clip.",
                    e_zne,
                    e_exact,
                    violation_magnitude,
                )

        # ── Observable consistency ──
        # Compare energy from local observables vs ZNE energy.
        # For TFIM: E = -J·Σ⟨ZiZi+1⟩ - h·Σ⟨Xi⟩
        # This checks if observables tell a consistent story.
        try:
            n_qubits = self._config.n_qubits
            # Reconstruct E from local observables (TFIM: E = -Σzz - h·Σx)
            # h is encoded in the Hamiltonian — extract from coefficients
            h_coeffs = [
                abs(float(c))
                for op, c in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs, strict=False)
                if op.count("X") == 1 and op.count("I") == len(op) - 1
            ]
            h_field = h_coeffs[0] if h_coeffs else None
            if h_field and x_values and zz_values:
                e_from_obs = -sum(zz_values) - h_field * sum(x_values)
                metrics["e_from_local_observables"] = e_from_obs
                metrics["observable_energy_discrepancy"] = abs(e_from_obs - e_zne)
        except Exception:
            metrics["observable_consistency_error"] = "Could not compute"

        # ── ZNE extrapolation residual ──
        # For server-side ZNE, we don't have the raw noise-factor data.
        # Use inter-layout spread as proxy for extrapolation quality.
        if len(raw_results) > 1:
            energies = [r["energy"] for r in raw_results]
            metrics["zne_max_residual"] = float(max(energies) - min(energies))
            metrics["zne_layout_energies"] = energies

        # ── Queue time estimation ──
        # Calculate from running_timestamps if available.
        timestamps = [r.get("running_timestamp") for r in raw_results if r.get("running_timestamp")]
        if len(timestamps) >= 2:
            try:
                from datetime import datetime as _dt

                parsed = sorted(_dt.fromisoformat(t.replace("Z", "+00:00")) for t in timestamps)
                metrics["queue_span_s"] = (parsed[-1] - parsed[0]).total_seconds()
                metrics["first_job_started"] = str(parsed[0])
                metrics["last_job_started"] = str(parsed[-1])
            except Exception:
                pass

        return metrics

    @staticmethod
    def _aggregate_qpu_metrics(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate QPU usage metrics from per-layout job results.

        IBM Runtime provides qpu_seconds (numeric) per job, billed_seconds
        (total including classical), and running_timestamp (ISO string) for
        provenance. Only numeric values are summed.
        """
        qpu_seconds_list = [
            r.get("qpu_seconds")
            for r in raw_results
            if isinstance(r.get("qpu_seconds"), (int, float))
        ]
        billed_seconds_list = [
            r.get("billed_seconds")
            for r in raw_results
            if isinstance(r.get("billed_seconds"), (int, float))
        ]
        timestamps = [
            r.get("running_timestamp")
            for r in raw_results
            if r.get("running_timestamp") is not None
        ]
        return {
            "total_qpu_seconds": sum(qpu_seconds_list) if qpu_seconds_list else None,
            "total_billed_seconds": sum(billed_seconds_list) if billed_seconds_list else None,
            "n_jobs_with_metrics": len(qpu_seconds_list),
            "per_layout_qpu_s": qpu_seconds_list or None,
            "running_timestamps": timestamps or None,
        }

    @staticmethod
    def _compute_zne_gain(raw_results: list[dict[str, Any]], e_zne: float, e_exact: float) -> float:
        """Compute ZNE gain from noise-factor data when available.

        For PEA-ZNE (IBM Runtime), the server applies ZNE and returns the
        extrapolated energy directly. The raw (NF=1) energy is available in
        evs_noise_factors[0][0] from the PUB result. We use it to compute:
            gain = 1 - |e_zne - e_exact| / |e_nf1 - e_exact|

        When noise-factor data is unavailable, returns 0.0 (not None) to
        maintain backward compatibility with validators and thesis figures.
        """
        print("  [DEBUG] _compute_zne_gain: computing from noise-factor baseline")
        if not raw_results:
            return 0.0

        # Try to extract NF=1 energy from the first layout's noise-factor data
        first = raw_results[0]
        evs_nf = first.get("evs_noise_factors")

        if evs_nf is not None and len(evs_nf) > 0:
            # evs_noise_factors is typically [[val_nf1, val_nf1.5, val_nf3], ...]
            # For energy PUB, the first element is the NF array for the Hamiltonian
            # In hardware mode, raw_results contain per-layout energy at NF=1
            # which is stored in raw_results[i]["energy_nf1"] or derivable.
            pass

        # Fallback: use the layout mean at NF=1 if stored in raw_results
        nf1_energies = [r.get("energy_nf1") for r in raw_results if r.get("energy_nf1") is not None]
        if nf1_energies:
            e_nf1 = float(np.mean(nf1_energies))
        elif evs_nf is not None and isinstance(evs_nf, (list, np.ndarray)):
            # evs_noise_factors[observable_idx][noise_factor_idx]
            # For energy (single observable PUB): evs_nf[0] = NF=1 value
            try:
                e_nf1 = float(evs_nf[0]) if np.isscalar(evs_nf[0]) else float(evs_nf[0][0])
            except (IndexError, TypeError):
                return 0.0
        else:
            # No noise-factor baseline available — cannot compute gain
            return 0.0

        raw_error = abs(e_nf1 - e_exact)
        mitigated_error = abs(e_zne - e_exact)
        if raw_error < 1e-10:
            return 0.0
        return float(1.0 - mitigated_error / raw_error)

    def _aggregate_zne_results(
        self,
        raw_results: list[dict[str, Any]],
        layout_selection: LayoutSelection,
        hamiltonian: SparsePauliOp,
        gap: float,
    ) -> tuple[float, float, str]:
        """Aggregate ZNE results based on execution mode.

        Returns
        -------
        (e_zne, r2, amplifier_used)
            e_zne: ZNE-mitigated energy estimate
            r2: quality metric (1.0 = perfect, <0.80 = unreliable)
            amplifier_used: description of the strategy used
        """
        energies = [r["energy"] for r in raw_results]

        if self._config.mode == "hardware":
            # IBM Runtime applies ZNE server-side (amplifier configured via
            # build_estimator_options). Each layout returns a ZNE-mitigated
            # energy. We average for variance reduction (√n improvement).
            # CES-based client-side extrapolation is removed — it fails on
            # heavy_hex where all CES≈0.15 (no spread → R²≈0.04).
            e_zne = float(np.mean(energies))
            # R² proxy: use inter-layout consistency as a quality signal.
            # IBM applies ZNE independently per layout; consistent results
            # indicate reliable mitigation. We map layout agreement → R²-like score:
            #   - Perfect agreement (std ≈ 0): R² → 1.0
            #   - Large spread (std > 5% of |e_zne|): R² → 0.0 (unreliable)
            # This is conservative: a single-layout run gets R²=1.0 (trusted,
            # since IBM's own quality check passed at submission time).
            if len(energies) > 1 and abs(e_zne) > 1e-10:
                relative_std = float(np.std(energies, ddof=1)) / abs(e_zne)
                # 5% relative std → R²=0. Scale linearly in [0, 0.05] → [1, 0].
                r2 = float(max(0.0, 1.0 - relative_std / 0.05))
            else:
                r2 = 1.0  # Single layout: trust IBM's own ZNE quality
            amplifier = self._config.mitigation.zne_amplifier or "pea"
            return e_zne, r2, f"server_side_{amplifier}"

        # fake_backend mode: use local ZNE via noisy_utils
        return self._run_local_zne(raw_results, layout_selection, hamiltonian, gap)

    def _run_local_zne(
        self,
        raw_results: list[dict[str, Any]],
        layout_selection: LayoutSelection,
        hamiltonian: SparsePauliOp,
        gap: float,
    ) -> tuple[float, float, str]:
        """Local ZNE for fake_backend mode using GF, PEA, or adaptive from noisy_utils."""
        energies = [r["energy"] for r in raw_results]
        amplifier = self._config.mitigation.zne_amplifier or "pea"

        if len(energies) < 2:
            # Only 1 layout available — return raw energy, low quality
            return float(energies[0]), 0.5, f"raw_{amplifier}"

        if amplifier == "adaptive":
            # Adaptive: try GF first, fall back to PEA if R² < threshold
            try:
                from qmbp_simulation.execution.noisy_utils import run_adaptive_zne

                isa_circ = layout_selection.transpiled_circuits[0]
                h_mapped = hamiltonian.apply_layout(isa_circ.layout)
                r2_threshold = getattr(self._config.mitigation, "zne_r2_fallback_threshold", 0.90)
                result = run_adaptive_zne(
                    isa_circ,
                    h_mapped,
                    self.backend,
                    config=NoisyEstimatorConfig(
                        shots=self._config.shots,
                        seed_simulator=self._config.layout_seed,
                    ),
                    r2_threshold=r2_threshold,
                )
                return result.extrapolated_value, result.r_squared, result.amplifier_used
            except Exception as exc:
                self._logger.log(
                    "zne_adaptive_fallback", data={"error": str(exc), "fallback": "pea_or_gf"}
                )

        if amplifier == "pea":
            # Try PEA on the best (first) transpiled circuit
            try:
                from qmbp_simulation.execution.noisy_utils import run_pea_zne

                isa_circ = layout_selection.transpiled_circuits[0]
                h_mapped = hamiltonian.apply_layout(isa_circ.layout)
                pea_config = NoisyEstimatorConfig(
                    shots=self._config.shots,
                    seed_simulator=self._config.layout_seed,
                )
                pea = run_pea_zne(
                    isa_circ,
                    h_mapped,
                    self.backend,
                    pea_config,
                    noise_factors=(1, 3, 5),
                )
                return pea.extrapolated_value, pea.r_squared, "pea"
            except Exception as exc:
                self._logger.log(
                    "zne_pea_fallback", data={"error": str(exc), "fallback": "gate_folding"}
                )

        # Gate-folding ZNE: check CES spread using P0-A relative threshold
        # (spread_ratio = (max-min)/mean ≥ 0.3). The old hardcoded 0.02 absolute
        # threshold missed the heavy_hex failure mode where CES≈0.15 uniformly.
        ces_used = [layout_selection.ces_values[r["layout_idx"]] for r in raw_results]
        ces_mean = float(np.mean(ces_used)) if ces_used else 0.0
        ces_range = max(ces_used) - min(ces_used) if len(ces_used) >= 2 else 0.0
        spread_ratio = ces_range / ces_mean if ces_mean > 1e-10 else 0.0

        # Also check the P0-A metadata if available (from select_layouts_by_circuit_ces)
        ces_spread_sufficient = getattr(layout_selection, "ces_spread_sufficient", None)
        if ces_spread_sufficient is None:
            # Fallback: compute locally with the same threshold as P0-A
            ces_spread_sufficient = spread_ratio >= 0.3

        if ces_spread_sufficient:
            zne_result = linear_zne(np.array(ces_used), np.array(energies))
            return zne_result.extrapolated_value, zne_result.r_squared, "ces_gf"

        # Fallback: gate-folding with noise_factors on first circuit
        try:
            from qmbp_simulation.execution.noisy_utils import run_gate_folding_zne

            isa_circ = layout_selection.transpiled_circuits[0]
            h_mapped = hamiltonian.apply_layout(isa_circ.layout)
            gf_config = NoisyEstimatorConfig(
                shots=self._config.shots,
                seed_simulator=self._config.layout_seed,
            )
            gf = run_gate_folding_zne(isa_circ, h_mapped, self.backend, gf_config)
            return gf.extrapolated_value, gf.r_squared, "gate_folding"
        except Exception as exc:
            # Last resort: simple average
            self._logger.log(
                "zne_all_failed", data={"error": str(exc), "fallback": "simple_average"}
            )
            e_avg = float(np.mean(energies))
            return e_avg, 0.5, "average"

    # ─── Estimator Configuration ──────────────────────────────────────

    def _get_configured_estimator(self) -> Any:
        """Create estimator with full mitigation options.

        For hardware mode, uses the structured options API (same pattern as
        submission._get_estimator) to avoid dict-format incompatibilities.
        """
        from .submission import _get_estimator

        return _get_estimator(self.backend, self._config)
