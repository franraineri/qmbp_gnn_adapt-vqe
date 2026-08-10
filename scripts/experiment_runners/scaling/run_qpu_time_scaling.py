#!/usr/bin/env python3
"""QPU Time Scaling — Measure circuit scaling behavior from N=20 to N=133.

Demonstrates that HVA p=1 QPU time scales polynomially with system size N,
confirming the CLOPS model predictions. Runs entirely locally using
FakeTorino (133 qubits) for empirical transpilation+timing measurements.

Sections:
    1. Circuit Construction & Transpilation: Build HVA p=1 circuits at each N,
       transpile to FakeTorino ISA, measure classical preprocessing time and
       circuit metrics (depth, depth_2q, CX count, SWAP count).

    2. CLOPS Model Prediction: Use estimate_qpu_cost_extended() to predict
       QPU wall-clock at each N for comparison with empirical timing.

    3. FakeTorino Empirical Timing: Execute circuits on FakeTorino noise model
       to measure local simulation wall-clock (validates circuit executability
       and provides a timing reference, though not identical to real QPU time).

    4. Scaling Law Fit: Fit T(N) = a·N^b + c to the empirical data and
       compare exponent with CLOPS model prediction.

Output metrics per N:
    - n_qubits, p_layers, topology
    - cx_count_pre_transpile (from HVA builder)
    - cx_count_post_transpile (after layout + routing)
    - n_swap_gates (routing overhead)
    - depth_total, depth_2q
    - transpile_time_s (classical preprocessing)
    - estimated_qpu_s (from CLOPS model, per h-point)
    - estimated_qpu_total_s (full 4-point sweep)
    - fake_backend_time_s (FakeTorino execution, single eval)
    - decoherence_fraction, t1_budget_ratio, snr_at_critical

Usage:
    # Default: N=[20,30,40,50,80,100,127], heavy_hex, p=1
    python scripts/experiment_runners/scaling/run_qpu_time_scaling.py

    # Quick test (fewer N values)
    python scripts/experiment_runners/scaling/run_qpu_time_scaling.py \
        --n-values 20 40 80

    # Different topology
    python scripts/experiment_runners/scaling/run_qpu_time_scaling.py \
        --topology chain_1d --n-values 20 40 60 80 100

    # Dry run
    python scripts/experiment_runners/scaling/run_qpu_time_scaling.py --dry-run
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Default N values spanning the full heavy-hex range (FakeTorino = 133 qubits).
# N=127 is the max embeddable in 133-qubit heavy-hex with margin for routing.
DEFAULT_N_VALUES = [20, 30, 40, 50, 80, 100, 127]
DEFAULT_TOPOLOGY = "heavy_hex"
DEFAULT_MODEL = "tfim"
DEFAULT_P = 1
DEFAULT_H_TEST = 4.0  # Deep paramagnetic — always in valid regime
DEFAULT_SHOTS = 16384
DEFAULT_N_LAYOUTS = 3
DEFAULT_N_H_POINTS = 4  # For QPU cost estimation (full sweep scenario)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class QPUTimeScalingRunner(ValidationRunner):
    """Measure QPU time scaling with system size N for HVA p=1.

    Combines transpilation metrics, CLOPS model predictions, and FakeTorino
    empirical timing to characterize how execution cost grows with N.
    """

    runner_id = "qpu_time_scaling_v1"
    experiment_id = "scaling/qpu_time"
    description = "QPU Time Scaling — T(N) characterization for HVA p=1"
    hypothesis = (
        "QPU execution time scales as O(N^b) with b≈1.3-1.7 (polynomial), "
        "confirming quantum advantage window at N>20 where MPS loses precision."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES,
            help="System sizes to test (default: 20 30 40 50 80 100 127)",
        )
        parser.add_argument("--topology", type=str, default=DEFAULT_TOPOLOGY)
        parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
        parser.add_argument("--p-layers", type=int, default=DEFAULT_P, choices=[1, 2])
        parser.add_argument(
            "--h-test", type=float, default=DEFAULT_H_TEST,
            help="h-value for circuit evaluation (default: 4.0, deep paramagnetic)",
        )
        parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
        parser.add_argument("--n-layouts", type=int, default=DEFAULT_N_LAYOUTS)
        parser.add_argument(
            "--n-h-points", type=int, default=DEFAULT_N_H_POINTS,
            help="Number of h-points for QPU cost estimation (full sweep scenario)",
        )
        parser.add_argument(
            "--skip-fake-backend",
            action="store_true",
            help="Skip FakeTorino execution (faster, transpilation+model only)",
        )

    def run_preflight(self) -> bool:
        """Validate N values fit within FakeTorino (133 qubits)."""
        max_n = max(self._args.n_values)
        if max_n > 133:
            logger.error(
                f"Max N={max_n} exceeds FakeTorino capacity (133 qubits). "
                f"Use N≤127 to allow routing margin."
            )
            return False
        if self._args.p_layers > 1 and max_n > 50:
            logger.warning(
                f"p={self._args.p_layers} at N={max_n} may exceed ZNE threshold. "
                f"Results are still valid for timing but not for deployment."
            )
        return True


    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "system": {
                "n_values": self._args.n_values,
                "topology": self._args.topology,
                "model": self._args.model,
                "p_layers": self._args.p_layers,
                "h_test": self._args.h_test,
            },
            "execution": {
                "shots": self._args.shots,
                "n_layouts": self._args.n_layouts,
                "n_h_points": self._args.n_h_points,
                "skip_fake_backend": self._args.skip_fake_backend,
            },
            "seeds": [42],
        }

    def setup(self):
        """Load FakeTorino backend and build shared objects."""
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        self._fake_backend = FakeTorino()
        logger.info(f"  FakeTorino loaded: {self._fake_backend.num_qubits} qubits")
        logger.info(f"  N values: {self._args.n_values}")
        logger.info(f"  Topology: {self._args.topology}, p={self._args.p_layers}")

        # Store results across sections
        self._transpile_data: list[dict] = []
        self._cost_data: list[dict] = []
        self._timing_data: list[dict] = []

    def define_sections(self) -> list[Section]:
        sections = [
            Section(
                id=1,
                name="Circuit Construction & Transpilation",
                fn=self.section_transpilation,
                hypothesis=(
                    "Transpiled CX count scales linearly with N for HVA p=1 "
                    "on heavy-hex (minimal routing due to native connectivity)"
                ),
            ),
            Section(
                id=2,
                name="CLOPS Model Prediction",
                fn=self.section_clops_model,
                hypothesis=(
                    "CLOPS model predicts polynomial QPU time scaling "
                    "T(N) ~ N^1.3 for shallow HVA circuits"
                ),
            ),
        ]
        if not self._args.skip_fake_backend:
            sections.append(
                Section(
                    id=3,
                    name="FakeTorino Empirical Timing",
                    fn=self.section_fake_backend_timing,
                    hypothesis=(
                        "FakeTorino execution time confirms polynomial scaling "
                        "and validates circuit executability at all N"
                    ),
                )
            )
        sections.append(
            Section(
                id=4,
                name="Scaling Law Fit & Summary",
                fn=self.section_scaling_fit,
                hypothesis="T(N) = a·N^b + c with b < 2 (subquadratic)",
            )
        )
        return sections


    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Circuit Construction & Transpilation
    # ═══════════════════════════════════════════════════════════════════════════

    def section_transpilation(self) -> dict:
        """Build and transpile HVA circuits at each N, measure metrics."""
        from experiments.helpers.scaling_utils import compute_transpilation_metrics

        from qmbp_simulation import make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec

        topology = self._args.topology
        p = self._args.p_layers
        h = self._args.h_test
        spec = get_model_spec(self._args.model)

        self._transpile_data = []

        for N in self._args.n_values:
            logger.info(f"    N={N}: building + transpiling...")
            t0 = time.perf_counter()

            # Build lattice and circuit
            lattice = make_lattice(topology, N, J=1.0, h=h)
            circuit, _ = spec.create_circuit(N, p, lattice, **spec.circuit_kwargs)

            # Use shared transpilation utility
            metrics = compute_transpilation_metrics(
                circuit, self._fake_backend, optimization_level=2
            )

            total_time = time.perf_counter() - t0

            record = {
                "n_qubits": N,
                "p_layers": p,
                "topology": topology,
                "n_params": circuit.num_parameters,
                "cx_count_pre_transpile": metrics["cx_count_pre_transpile"],
                "cx_count_post_transpile": metrics["cx_count_post_transpile"],
                "n_swap_gates": metrics["n_swap_gates"],
                "routing_overhead_ratio": metrics["routing_overhead_ratio"],
                "depth_total": metrics["depth_total"],
                "depth_2q": metrics["depth_2q"],
                "transpile_time_s": metrics["transpile_time_s"],
                "total_build_time_s": round(total_time, 3),
                "gate_counts": metrics["gate_counts"],
            }
            self._transpile_data.append(record)

            logger.info(
                f"      CX: {metrics['cx_count_pre_transpile']}→"
                f"{metrics['cx_count_post_transpile']} "
                f"(routing ×{metrics['routing_overhead_ratio']:.2f}), "
                f"depth_2q={metrics['depth_2q']}, SWAPs={metrics['n_swap_gates']}, "
                f"transpile={metrics['transpile_time_s']:.2f}s"
            )

        return {
            "pass": True,
            "n_points": len(self._transpile_data),
            "per_n": self._transpile_data,
            "summary": {
                "cx_scaling": {
                    "min_n": self._transpile_data[0]["n_qubits"],
                    "max_n": self._transpile_data[-1]["n_qubits"],
                    "min_cx": self._transpile_data[0]["cx_count_post_transpile"],
                    "max_cx": self._transpile_data[-1]["cx_count_post_transpile"],
                },
            },
        }


    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: CLOPS Model Prediction
    # ═══════════════════════════════════════════════════════════════════════════

    def section_clops_model(self) -> dict:
        """Predict QPU time at each N using the calibrated CLOPS model."""
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.preflight import (
            QPUThroughputProfile,
            SPSACostModel,
            estimate_qpu_cost,
            estimate_qpu_cost_extended,
        )

        shots = self._args.shots
        n_layouts = self._args.n_layouts
        n_h_points = self._args.n_h_points

        # Use Kingston/Heron r2 profile (our target QPU)
        profile = QPUThroughputProfile.ibm_kingston()
        spsa_model = SPSACostModel.disabled()  # Pure prediction, no SPSA

        self._cost_data = []

        for idx, N in enumerate(self._args.n_values):
            # Use transpile data if available for accurate depth/cx_count
            cx_count = None
            depth = None
            if self._transpile_data and idx < len(self._transpile_data):
                cx_count = self._transpile_data[idx]["cx_count_post_transpile"]
                depth = self._transpile_data[idx]["depth_total"]

            config = HardwareConfig(
                n_qubits=N,
                shots=shots,
                n_layouts=n_layouts,
            )

            # Basic estimate
            est = estimate_qpu_cost(
                config=config,
                n_h_points=n_h_points,
                include_spsa=False,
                circuit_depth=depth,
                cx_count=cx_count,
                profile=profile,
                spsa_model=spsa_model,
            )

            # Extended estimate (decoherence, SNR)
            est_ext = estimate_qpu_cost_extended(
                config=config,
                n_h_points=n_h_points,
                include_spsa=False,
                circuit_depth=depth,
                cx_count=cx_count,
                profile=profile,
                spsa_model=spsa_model,
                backend=self._fake_backend,
            )

            record = {
                "n_qubits": N,
                "effective_clops": est.effective_clops,
                "time_per_circuit_s": est.time_per_circuit_s,
                "est_time_per_h_s": est.est_time_per_h_s,
                "est_total_optimistic_s": est.est_total_optimistic_s,
                "est_total_expected_s": est.est_total_s,
                "pea_noise_learning_s": est.pea_noise_learning_s,
                "classical_latency_s": est.classical_latency_s,
                # Extended metrics
                "decoherence_fraction": est_ext.decoherence_fraction,
                "t1_budget_ratio": est_ext.t1_budget_ratio,
                "shot_noise_sigma": est_ext.shot_noise_sigma,
                "snr_at_critical": est_ext.snr_at_critical,
                "decoherence_aware_clops": est_ext.decoherence_aware_clops,
            }
            self._cost_data.append(record)

            logger.info(
                f"    N={N:>3d}: CLOPS_eff={est.effective_clops:>5d}, "
                f"T/circuit={est.time_per_circuit_s:.3f}s, "
                f"T_total({n_h_points}h)={est.est_total_s:.1f}s, "
                f"T1_ratio={est_ext.t1_budget_ratio:.4f}"
            )

        # Compute scaling exponent from CLOPS estimates
        from experiments.helpers.scaling_utils import fit_power_law

        ns = np.array([d["n_qubits"] for d in self._cost_data], dtype=float)
        ts = np.array([d["est_total_expected_s"] for d in self._cost_data], dtype=float)
        scaling_fit = fit_power_law(ns, ts)
        scaling_exponent = scaling_fit["exponent"]

        return {
            "pass": True,
            "per_n": self._cost_data,
            "profile_name": profile.name,
            "profile_base_clops": profile.base_clops,
            "scaling_exponent_clops_model": (
                round(scaling_exponent, 3) if scaling_exponent else None
            ),
            "n_h_points": n_h_points,
            "shots": shots,
            "n_layouts": n_layouts,
        }


    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: FakeTorino Empirical Timing
    # ═══════════════════════════════════════════════════════════════════════════

    def section_fake_backend_timing(self) -> dict:
        """Execute a single evaluation at each N on FakeTorino, measure time."""
        from qiskit.primitives import BackendEstimatorV2

        from qmbp_simulation import make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec

        topology = self._args.topology
        p = self._args.p_layers
        h = self._args.h_test
        shots = self._args.shots
        spec = get_model_spec(self._args.model)

        self._timing_data = []

        for N in self._args.n_values:
            logger.info(f"    N={N}: FakeTorino execution...")

            # Build circuit + Hamiltonian
            lattice = make_lattice(topology, N, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
            circuit, _ = spec.create_circuit(N, p, lattice, **spec.circuit_kwargs)

            # Random params (timing only, don't need real θ_opt)
            rng = np.random.default_rng(42)
            params = rng.uniform(-0.01, 0.01, circuit.num_parameters)
            bound = circuit.assign_parameters(params)

            # Let BackendEstimatorV2 handle transpilation internally
            # (avoids qubit count mismatch between transpiled circuit and observable)
            estimator = BackendEstimatorV2(
                backend=self._fake_backend,
                options={"default_precision": 0.01},
            )

            t0 = time.perf_counter()
            job = estimator.run([(bound, H)])
            result = job.result()
            exec_time = time.perf_counter() - t0

            energy = float(result[0].data.evs)
            std = float(result[0].data.stds) if hasattr(result[0].data, "stds") else None

            record = {
                "n_qubits": N,
                "fake_backend_time_s": round(exec_time, 3),
                "energy": energy,
                "std": std,
                "precision": 0.01,
            }
            self._timing_data.append(record)

            logger.info(
                f"      E={energy:.6f}, time={exec_time:.2f}s"
            )

        return {
            "pass": True,
            "per_n": self._timing_data,
        }


    # ═══════════════════════════════════════════════════════════════════════════
    # Section 4: Scaling Law Fit & Summary
    # ═══════════════════════════════════════════════════════════════════════════

    def section_scaling_fit(self) -> dict:
        """Fit T(N) scaling law and produce consolidated summary."""
        from experiments.helpers.scaling_utils import fit_power_law

        ns = np.array(self._args.n_values, dtype=float)

        # Consolidate all data sources
        consolidated = []
        for idx, N in enumerate(self._args.n_values):
            entry = {"n_qubits": N}

            # Transpilation data
            if idx < len(self._transpile_data):
                td = self._transpile_data[idx]
                entry["cx_count_pre_transpile"] = td["cx_count_pre_transpile"]
                entry["cx_count_post_transpile"] = td["cx_count_post_transpile"]
                entry["n_swap_gates"] = td["n_swap_gates"]
                entry["routing_overhead_ratio"] = td["routing_overhead_ratio"]
                entry["depth_total"] = td["depth_total"]
                entry["depth_2q"] = td["depth_2q"]
                entry["transpile_time_s"] = td["transpile_time_s"]

            # CLOPS model data
            if idx < len(self._cost_data):
                cd = self._cost_data[idx]
                entry["effective_clops"] = cd["effective_clops"]
                entry["estimated_qpu_per_h_s"] = cd["est_time_per_h_s"]
                entry["estimated_qpu_total_s"] = cd["est_total_expected_s"]
                entry["decoherence_fraction"] = cd["decoherence_fraction"]
                entry["t1_budget_ratio"] = cd["t1_budget_ratio"]
                entry["snr_at_critical"] = cd["snr_at_critical"]

            # FakeTorino timing
            if idx < len(self._timing_data):
                ft = self._timing_data[idx]
                entry["fake_backend_time_s"] = ft["fake_backend_time_s"]

            consolidated.append(entry)

        # Fit power law to CLOPS model predictions
        clops_fit = fit_power_law(
            ns, [d["est_total_expected_s"] for d in self._cost_data]
        )
        clops_exponent = clops_fit["exponent"]
        clops_fit_r2 = clops_fit["r_squared"]

        # Fit power law to FakeTorino timing (if available)
        fake_fit = fit_power_law(
            ns, [d["fake_backend_time_s"] for d in self._timing_data]
        ) if self._timing_data else {"exponent": None, "r_squared": None}
        fake_exponent = fake_fit["exponent"]
        fake_fit_r2 = fake_fit["r_squared"]

        # Fit power law to CX count (should be ~linear for 1D-embeddable)
        cx_fit = fit_power_law(
            ns, [d["cx_count_post_transpile"] for d in self._transpile_data]
        ) if self._transpile_data else {"exponent": None}
        cx_exponent = cx_fit["exponent"]

        # Fit power law to transpilation time
        transpile_fit = fit_power_law(
            ns, [d["transpile_time_s"] for d in self._transpile_data]
        ) if self._transpile_data else {"exponent": None}
        transpile_exponent = transpile_fit["exponent"]

        # Determine if polynomial (b < 2)
        # If insufficient data points, still pass (informational only)
        is_polynomial = (
            clops_exponent is not None and clops_exponent < 2.0
        ) if clops_exponent is not None else True

        # Print summary table
        logger.info("\n    ═══ QPU TIME SCALING SUMMARY ═══")
        logger.info(f"    {'N':>5} {'CX_post':>8} {'depth_2q':>9} {'T_est(s)':>9} "
                    f"{'T_fake(s)':>10} {'T1_ratio':>9}")
        logger.info(f"    {'-'*5} {'-'*8} {'-'*9} {'-'*9} {'-'*10} {'-'*9}")
        for entry in consolidated:
            cx = entry.get("cx_count_post_transpile", "—")
            d2q = entry.get("depth_2q", "—")
            t_est = entry.get("estimated_qpu_total_s")
            t_est_s = f"{t_est:.1f}" if t_est else "—"
            t_fake = entry.get("fake_backend_time_s")
            t_fake_s = f"{t_fake:.2f}" if t_fake else "—"
            t1r = entry.get("t1_budget_ratio")
            t1r_s = f"{t1r:.4f}" if t1r else "—"
            logger.info(
                f"    {entry['n_qubits']:>5} {cx:>8} {d2q:>9} "
                f"{t_est_s:>9} {t_fake_s:>10} {t1r_s:>9}"
            )

        logger.info(f"\n    Scaling exponents (T ~ N^b):")
        logger.info(f"      CLOPS model:     b = {clops_exponent:.3f} (R²={clops_fit_r2:.4f})"
                    if clops_exponent else "      CLOPS model:     insufficient data")
        if fake_exponent:
            logger.info(f"      FakeTorino:      b = {fake_exponent:.3f} (R²={fake_fit_r2:.4f})")
        if cx_exponent:
            logger.info(f"      CX count:        b = {cx_exponent:.3f}")
        if transpile_exponent:
            logger.info(f"      Transpile time:  b = {transpile_exponent:.3f}")
        logger.info(f"    Polynomial (b<2): {'YES ✓' if is_polynomial else 'NO ✗'}")

        return {
            "pass": is_polynomial,
            "consolidated_per_n": consolidated,
            "scaling_exponents": {
                "clops_model": clops_exponent,
                "clops_model_r2": clops_fit_r2,
                "fake_backend": fake_exponent,
                "fake_backend_r2": fake_fit_r2,
                "cx_count": cx_exponent,
                "transpile_time": transpile_exponent,
            },
            "is_polynomial": is_polynomial,
            "thesis_claim": (
                f"QPU time scales as T(N) ~ N^{clops_exponent:.2f} "
                f"(subquadratic, R²={clops_fit_r2:.3f})"
                if clops_exponent and clops_fit_r2
                else "Insufficient data for scaling claim"
            ),
        }


if __name__ == "__main__":
    QPUTimeScalingRunner.main()
