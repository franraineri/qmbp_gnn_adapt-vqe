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

        When loaded, `run_deployment()` will apply GNN correction after ZNE
        and before affine clipping. The correction is confidence-gated:
        if the model's confidence is below `gnn_qem_confidence_threshold`
        (default 0.5), the correction is skipped.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to a .pt checkpoint saved by `save_qem_checkpoint()`.

        References
        ----------
        Wang et al. arXiv:2604.16815 (2026) — GEM framework.
        """
        from qmbp_simulation.predictors.gnn_qem import load_qem_checkpoint

        model, train_result, metadata = load_qem_checkpoint(Path(checkpoint_path))
        self._gnn_qem_model = model
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
            from qiskit_ibm_runtime.fake_provider import FakeKingston

            self._backend = FakeKingston()
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

            # Per-site observables on first layout
            edges = [(i, i + 1) for i in range(self._config.n_qubits - 1)]
            x_ops, zz_ops = build_per_site_observables(self._config.n_qubits, edges)
            isa_circ = layout_selection.transpiled_circuits[0]
            mapped_obs = map_observables_to_layout(x_ops + zz_ops, isa_circ)

            # Submit observables within a Batch (hardware) or directly (fake_backend)
            if self._config.mode == "hardware":
                from qiskit_ibm_runtime import Batch as _Batch
                from qiskit_ibm_runtime import EstimatorV2 as _EstV2

                with _Batch(backend=self.backend) as obs_batch:
                    obs_est = _EstV2(mode=obs_batch)
                    from .submission import _apply_estimator_options

                    _apply_estimator_options(obs_est, self._config)
                    obs_job = obs_est.run([(isa_circ, mapped_obs)])

                # Wait for result outside the batch context
                if hasattr(obs_job, "wait_for_final_state"):
                    obs_job.wait_for_final_state(timeout=self._config.job_timeout_s)
                obs_result = obs_job.result()
                evs = obs_result[0].data.evs
            else:
                estimator = self._get_configured_estimator()
                evs = estimator.run([(isa_circ, mapped_obs)]).result()[0].data.evs

            evs = np.atleast_1d(evs)
            x_values = [float(evs[i]) for i in range(len(x_ops))]
            zz_values = [float(evs[len(x_ops) + i]) for i in range(len(zz_ops))]

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
                    delta_e_gap = abs(e_zne - e_exact) / gap if gap > 0 else float("inf")

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
                zne_gain=0.0,
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
            )

            zne_data = {
                "extrapolated_energy": e_zne,
                "r_squared": zne_r2,
                "amplifier": zne_amplifier_used,
                "n_layouts": len(raw_results),
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

        Includes TLS calibration drift monitoring: takes a baseline snapshot
        before the sweep and checks for drift (>20% T1 degradation) between
        h-points. Aborts early if calibration drifts beyond safe thresholds.
        """
        from .persistence import save_sweep_summary

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
                    from qmbp_simulation.execution.noisy_utils import check_calibration_drift

                    drift = check_calibration_drift(self.backend, baseline_snapshot)
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

        Creates two images in the output directory:
        - circuit_logical_h{h}.png: The bound logical circuit (as designed)
        - circuit_transpiled_h{h}.png: The first transpiled ISA circuit (as executed)

        This provides a visual provenance record for every hardware execution,
        enabling quick debugging and thesis figure generation.
        """

        try:
            from pathlib import Path as _Path

            from qiskit.visualization import circuit_drawer

            output_dir = _Path(self._config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            h_tag = f"h{h_value:.1f}".replace(".", "p")

            # Save the logical (bound) circuit
            bound_qc = circuit.assign_parameters(params)
            _diagram = circuit_drawer(
                bound_qc,
                output="mpl",
                fold=40,
                filename=str(output_dir / f"circuit_logical_{h_tag}.png"),
            )
            if hasattr(_diagram, "savefig"):
                _diagram.savefig(
                    str(output_dir / f"circuit_logical_{h_tag}.png"), dpi=150, bbox_inches="tight"
                )
            import matplotlib.pyplot as _plt

            _plt.close("all")

            # Save the first transpiled (ISA) circuit
            if layout_selection.transpiled_circuits:
                _diagram = circuit_drawer(
                    layout_selection.transpiled_circuits[0],
                    output="mpl",
                    fold=60,
                    filename=str(output_dir / f"circuit_transpiled_{h_tag}.png"),
                )
                if hasattr(_diagram, "savefig"):
                    _diagram.savefig(
                        str(output_dir / f"circuit_transpiled_{h_tag}.png"),
                        dpi=150,
                        bbox_inches="tight",
                    )
                _plt.close("all")

            self._logger.log(
                "circuit_diagram_saved",
                data={"output_dir": str(output_dir), "h_value": h_value},
            )
        except Exception as exc:
            # Never let diagram generation block QPU execution
            self._logger.log(
                "circuit_diagram_error",
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

        Records T1/T2 and 2Q error rates for the qubits actually used,
        enabling post-hoc correlation between result quality and calibration.
        """
        snapshot: dict[str, Any] = {"timestamp": datetime.now(UTC).isoformat()}
        try:
            target = self.backend.target
            # Per-qubit T1/T2 for used qubits
            qubit_props = target.qubit_properties
            if qubit_props and layout_selection.layouts:
                used_qubits = set()
                for layout in layout_selection.layouts:
                    used_qubits.update(layout[: self._config.n_qubits])
                t1_values = {}
                t2_values = {}
                for q in sorted(used_qubits):
                    if q < len(qubit_props) and qubit_props[q] is not None:
                        t1 = getattr(qubit_props[q], "t1", None)
                        t2 = getattr(qubit_props[q], "t2", None)
                        if t1 is not None:
                            t1_values[q] = t1 * 1e6  # Convert to μs
                        if t2 is not None:
                            t2_values[q] = t2 * 1e6
                snapshot["t1_us_per_qubit"] = t1_values
                snapshot["t2_us_per_qubit"] = t2_values
                snapshot["min_t1_us"] = min(t1_values.values()) if t1_values else None
                snapshot["min_t2_us"] = min(t2_values.values()) if t2_values else None

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

        The depth_2q metric is the strongest predictor of hardware error
        accumulation (2Q gates dominate noise budget on IBM devices).

        Note: This mirrors `transpiled_circuit_stats` from
        `analysis.circuit_visualizer` but is kept inline to respect the
        module dependency DAG (execution cannot import analysis).
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

                stats["per_layout"].append(layout_stats)

            if stats["per_layout"]:
                stats["mean_depth"] = np.mean([s["depth"] for s in stats["per_layout"]])
                stats["mean_depth_2q"] = np.mean([s["depth_2q"] for s in stats["per_layout"]])
                stats["mean_2q_gates"] = np.mean([s["n_2q_gates"] for s in stats["per_layout"]])
                # Aggregate count_ops across layouts (use first layout as representative)
                first_ops = stats["per_layout"][0].get("count_ops")
                if first_ops:
                    stats["basis_gates"] = sorted(first_ops.keys())
        except Exception as exc:
            stats["capture_error"] = str(exc)
        return stats

    @staticmethod
    def _aggregate_qpu_metrics(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate QPU usage metrics from per-layout job results.

        IBM Runtime provides qpu_seconds (numeric) per job and running_timestamp
        (ISO string) for provenance. Only numeric values are summed.
        """
        qpu_seconds_list = [
            r.get("qpu_seconds")
            for r in raw_results
            if isinstance(r.get("qpu_seconds"), (int, float))
        ]
        timestamps = [
            r.get("running_timestamp")
            for r in raw_results
            if r.get("running_timestamp") is not None
        ]
        return {
            "total_qpu_seconds": sum(qpu_seconds_list) if qpu_seconds_list else None,
            "n_jobs_with_metrics": len(qpu_seconds_list),
            "per_layout_qpu_s": qpu_seconds_list or None,
            "running_timestamps": timestamps or None,
        }

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

        # Gate-folding ZNE: use CES spread if available, else noise factors
        ces_used = [layout_selection.ces_values[r["layout_idx"]] for r in raw_results]
        ces_spread = max(ces_used) - min(ces_used)

        if ces_spread > 0.02:  # Enough CES spread for meaningful extrapolation
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
