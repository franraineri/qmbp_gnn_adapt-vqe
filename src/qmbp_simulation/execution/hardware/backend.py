"""HardwareBackend — IBM Runtime execution with inhomogeneous ZNE.

Two interface levels:
- evaluate() — ABC contract, returns float. No side effects (no disk writes).
- run_deployment() — full pipeline with preflight + classify + persist.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.execution.backends import ExecutionBackend
from qmbp_simulation.execution.hardware.config import HardwareConfig, HardwareRunResult, SPSAConfig
from qmbp_simulation.execution.noisy_utils import LayoutSelection, linear_zne


class HardwareBackend(ExecutionBackend):
    """IBM Runtime hardware backend with inhomogeneous ZNE and full mitigation."""

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

        if self._config.mode == "hardware":
            try:
                import qiskit_ibm_runtime  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "qiskit-ibm-runtime is required for hardware mode. "
                    "Install with: pip install 'qiskit-ibm-runtime>=0.20'"
                ) from e

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
        """Establish connection to IBM Quantum or initialize FakeTorino."""
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
            self._service = QiskitRuntimeService(channel="ibm_quantum_platform")
            self._backend = self._service.backend(self._config.backend_name)
        self._execution_mode = self._detect_execution_mode()

    def _detect_execution_mode(self) -> type:
        """Detect Batch vs Session via hasattr introspection."""
        import qiskit_ibm_runtime

        if hasattr(qiskit_ibm_runtime, "Batch"):
            return qiskit_ibm_runtime.Batch
        if hasattr(qiskit_ibm_runtime, "Session"):
            return qiskit_ibm_runtime.Session
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
        """Return ZNE-extrapolated energy. No side effects (no disk writes)."""
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
        ces_used = [layout_selection.ces_values[r["layout_idx"]] for r in raw_results]

        zne_result = linear_zne(np.array(ces_used), np.array(energies))
        self._total_shots += self._config.shots * len(raw_results)
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
            raw_results = submit_all_then_collect(
                layout_selection.transpiled_circuits,
                hamiltonian,
                self.backend,
                self._config,
                self._logger,
            )
            energies = [r["energy"] for r in raw_results]
            ces_used = [layout_selection.ces_values[r["layout_idx"]] for r in raw_results]
            zne_result = linear_zne(np.array(ces_used), np.array(energies))
            e_zne = zne_result.extrapolated_value
            self._total_shots += self._config.shots * len(raw_results)
            partial_data.append({"step": "energy", "e_zne": e_zne})

            # Per-site observables on first layout
            edges = [(i, i + 1) for i in range(self._config.n_qubits - 1)]
            x_ops, zz_ops = build_per_site_observables(self._config.n_qubits, edges)
            isa_circ = layout_selection.transpiled_circuits[0]
            mapped_obs = map_observables_to_layout(x_ops + zz_ops, isa_circ)
            estimator = self._get_configured_estimator()
            evs = estimator.run([(isa_circ, mapped_obs)]).result()[0].data.evs
            x_values = [float(evs[i]) for i in range(len(x_ops))]
            zz_values = [float(evs[len(x_ops) + i]) for i in range(len(zz_ops))]

            label, mag_x, corr_zz, sigma = classify_phase(
                x_values,
                zz_values,
                self._config.shots,
            )
            delta_e_gap = abs(e_zne - e_exact) / gap if gap > 0 else float("inf")

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

            verdict = "PASS" if (delta_e_gap < 0.05 and label == expected_label) else "FAIL"
            result = HardwareRunResult(
                h_value=h_value,
                e_exact=e_exact,
                e_zne=e_zne,
                delta_e_gap=delta_e_gap,
                gap=gap,
                phase_label=label,
                expected_label=expected_label,
                zne_r2=zne_result.r_squared,
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
            )

            zne_data = {
                "extrapolated_energy": zne_result.extrapolated_value,
                "r_squared": zne_result.r_squared,
                "slope": zne_result.slope,
                "ces_values": zne_result.ces_values.tolist(),
                "measured_values": zne_result.measured_values.tolist(),
            }
            save_run(
                result,
                self._config,
                self._logger,
                calibration_info={},
                options_dict=build_estimator_options(self._config),
                execution_mode_name=(
                    self._execution_mode.__name__ if self._execution_mode else "unknown"
                ),
                raw_per_layout=raw_results,
                zne_data=zne_data,
                input_params=params,
            )
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
        """Execute multiple h-points in single Batch/Session."""
        from .persistence import save_sweep_summary

        if 4.0 in h_values:
            h_values = [4.0] + [h for h in h_values if h != 4.0]

        preflight = self.run_preflight()
        if preflight.get("abort"):
            raise RuntimeError(f"Preflight abort: {preflight.get('abort_reason')}")

        # Pre-cache layouts (reuse across all h-points)
        bound = circuit.assign_parameters(params_per_h[h_values[0]])
        self._get_cached_layouts(bound)

        results: list[HardwareRunResult] = []
        for h in h_values:
            result = self.run_deployment(
                circuit,
                hamiltonian_builder(h),
                params_per_h[h],
                h_value=h,
                e_exact=e_exact_per_h[h],
                gap=gap_per_h[h],
            )
            results.append(result)
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

    # ─── Estimator Configuration ──────────────────────────────────────

    def _get_configured_estimator(self) -> Any:
        """Create estimator with full mitigation options."""
        from .submission import build_estimator_options

        options = build_estimator_options(self._config)
        if self._config.mode == "fake_backend":
            from qiskit.primitives import BackendEstimatorV2

            return BackendEstimatorV2(backend=self.backend, options=options)
        from qiskit_ibm_runtime import EstimatorV2

        return EstimatorV2(backend=self.backend, options=options)
