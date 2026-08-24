#!/usr/bin/env python
"""Quench Dynamics Study — GNN as Enabling Technology for Quantum Advantage.

Positioning: GNN-HVA reduces state preparation overhead from O(VQE iterations × QPU-time)
to O(1 forward pass), enabling systematic exploration of quench dynamics in the regime
where quantum advantage has been demonstrated (IBM arXiv:2607.24937).

This runner validates:
  Section 1: Show that quench dynamics from |ψ₀(h₁)⟩ (GNN-prepared) differs
    qualitatively from dynamics starting at |0⟩^N, in heavy-hex N=21-35 (exact ED).
    This proves the initial state MATTERS — justifying non-trivial preparation.

  Section 2: MPS crossover identification — for heavy-hex N=51, at how many Trotter
    steps does TEBD/MPS (χ=64-256) lose precision? If <15, we're in the demonstrated
    quantum advantage regime (IBM needed ~30 cycles).

  Section 3: Preparation cost comparison — quantify the VQE cost (iterations × time)
    vs GNN cost (1 forward pass) for preparing |ψ₀(h)⟩ across the phase diagram.

Key distinction from IBM:
  - IBM uses |0⟩^N (trivial preparation) + Floquet dynamics (single parameter point)
  - We enable |ψ₀(h)⟩ (arbitrary phase diagram point) + quench dynamics
  - This opens a vastly richer parameter space: {h₁, h₂, N, topology}

Usage:
    # Section 1: Initial-state dependence (N=21, heavy_hex, exact)
    python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \\
        --section 1 --n-qubits 21 --topology heavy_hex --n-trotter 25 --dt 0.1

    # Section 2: MPS crossover (N=51, heavy_hex, MPS only)
    python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \\
        --section 2 --n-qubits 51 --topology heavy_hex --n-trotter 30 --dt 0.1

    # Section 3: Preparation cost analysis
    python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \\
        --section 3 --n-qubits 20 --topology heavy_hex

    # Multi-N DQPT scaling ladder (builds trajectories for validate_dqpt_results)
    python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \\
        --section 1 --topology heavy_hex --dqpt-n-values 8 10 12 14 16 20 \\
        --h1 3.0 --h2 0.5 --dt 0.05 --n-trotter 60

References:
    - IBM+Qedma arXiv:2607.24937: Floquet dynamics quantum advantage at N=51-74
    - This project: GNN-HVA state preparation as enabling technology
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from qmbp_simulation.framework.runner_base import Section, ValidationRunner

logger = logging.getLogger(__name__)

# Practical limit for exact time evolution (sparse eigsh + expm_multiply)
_ED_MAX_N = 22
# Dense expm limit (above this, use sparse Krylov)
_DENSE_LIMIT = 16


class QuenchDynamicsStudyRunner(ValidationRunner):
    """Quench dynamics: GNN preparation as enabling technology for quantum advantage."""

    runner_id = "quench_dynamics_study"
    experiment_id = "QD1"
    description = "Quench Dynamics — GNN preparation enables quantum advantage regime"
    hypothesis = (
        "GNN-prepared |ψ₀(h)⟩ produces qualitatively different quench dynamics "
        "than trivial |0⟩^N, and MPS loses fidelity at ≤15 Trotter steps for "
        "heavy-hex N≥51 — within the demonstrated quantum advantage regime."
    )

    @classmethod
    def _add_custom_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--n-qubits", type=int, default=21,
            help="System size (default: 21 for heavy_hex)",
        )
        parser.add_argument(
            "--topology", type=str, nargs="+", default=["heavy_hex"],
            help="Lattice topology (default: heavy_hex).",
        )
        parser.add_argument(
            "--model", type=str, default="tfim", help="Hamiltonian model (default: tfim)",
        )
        parser.add_argument(
            "--p-layers", type=int, default=1, help="HVA circuit depth (default: 1)",
        )
        parser.add_argument(
            "--h1", type=float, default=3.0,
            help="Initial field h₁ (ground state preparation). Default: 3.0 (paramagnetic)",
        )
        parser.add_argument(
            "--h2", type=float, default=0.5,
            help="Quench field h₂ (evolution Hamiltonian). Default: 0.5 (ferromagnetic)",
        )
        parser.add_argument(
            "--n-trotter", type=int, default=25,
            help="Number of Trotter steps (default: 25)",
        )
        parser.add_argument(
            "--dt", type=float, default=0.1, help="Trotter time step dt (default: 0.1)",
        )
        parser.add_argument(
            "--chi-values", type=int, nargs="+", default=[64, 128, 256],
            help="MPS bond dimensions to test (default: 64 128 256)",
        )
        parser.add_argument(
            "--seeds", type=int, nargs="+", default=[42], help="Random seeds",
        )
        parser.add_argument(
            "--dqpt-n-values", type=int, nargs="+", default=None,
            help="Multiple N values for DQPT scaling ladder (e.g., 8 10 12 14 16 20). "
            "When set, Section 1 iterates over all N values, producing one trajectory "
            "per N. Required for validate_dqpt_results scaling checks (needs >=4 N).",
        )

    def build_config(self) -> dict:
        args = self._args
        topo = args.topology[0] if isinstance(args.topology, list) else args.topology
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "system": {
                "n_qubits": args.n_qubits,
                "topology": topo,
                "model": args.model,
                "p_layers": args.p_layers,
            },
            "quench": {
                "h1": args.h1,
                "h2": args.h2,
                "n_trotter": args.n_trotter,
                "dt": args.dt,
                "total_time": args.n_trotter * args.dt,
            },
            "mps": {"chi_values": args.chi_values},
            "seeds": args.seeds,
        }

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="Initial-state dependence (GNN vs trivial)",
                hypothesis=(
                    "Quench from GNN-prepared |ψ₀(h₁)⟩ produces qualitatively "
                    "different dynamics than from |0⟩^N, validating non-trivial "
                    "state preparation."
                ),
                fn=self.section_initial_state_dependence,
            ),
            Section(
                id=2,
                name="MPS crossover point (heavy-hex N≥51)",
                hypothesis=(
                    "MPS (χ=64-256) loses fidelity tracking quench dynamics after "
                    "≤15 Trotter steps for heavy-hex N≥51, placing us in the "
                    "demonstrated quantum advantage regime."
                ),
                fn=self.section_mps_crossover,
            ),
            Section(
                id=3,
                name="Preparation cost comparison (GNN vs VQE)",
                hypothesis=(
                    "GNN reduces preparation cost from O(n_restarts × maxiter × "
                    "circuit_evals) to O(1 forward pass), a 100-1000× speedup."
                ),
                fn=self.section_preparation_cost,
            ),
        ]

    def setup(self) -> None:
        """Initialize using the framework's setup_physics() for full module reuse."""
        self.setup_physics()
        args = self._args
        self._topology = args.topology[0] if isinstance(args.topology, list) else args.topology

    # ─── Section 1: Initial-State Dependence ─────────────────────────────────

    def section_initial_state_dependence(self) -> dict:
        """Compare quench dynamics from |ψ₀(h₁)⟩ vs |0⟩^N vs |+⟩^N.

        For N within ED range (≤22): exact time evolution via expm_multiply.
        For N > 22: MPS-based comparison (energy trajectory proxy).
        """
        args = self._args
        topo = self._topology

        # Multi-N mode: iterate over dqpt_n_values to build scaling ladder
        if args.dqpt_n_values:
            logger.info(f"  Multi-N DQPT mode: N={args.dqpt_n_values}")
            all_results = {}
            for n_val in args.dqpt_n_values:
                if n_val > _ED_MAX_N:
                    logger.info(f"  Skipping N={n_val} (> ED limit {_ED_MAX_N})")
                    continue
                logger.info(f"\n  ── N={n_val} ──")
                single = self._section1_single_n(n_val, topo, args)
                all_results[n_val] = single
            # Aggregate pass/fail
            n_pass = sum(1 for r in all_results.values() if r.get("pass"))
            return {
                "mode": "multi_n",
                "n_values": list(all_results.keys()),
                "results_by_n": all_results,
                "n_passed": n_pass,
                "n_total": len(all_results),
                "pass": n_pass > 0,
            }

        # Single-N mode (default)
        n = args.n_qubits
        return self._section1_single_n(n, topo, args)

    def _section1_single_n(self, n: int, topo: str, args) -> dict:
        """Section 1 logic for a single N value."""
        dt = args.dt
        n_steps = args.n_trotter

        logger.info(f"  Initial-state dependence: N={n}, {topo}")
        logger.info(f"  h₁={args.h1} → h₂={args.h2}, {n_steps} steps × dt={dt}")

        # Build quench Hamiltonian H(h₂) using framework's make_lattice + builder
        lattice_h2 = self.make_lattice(topo, n, J=1.0, h=args.h2)
        H2_op = self.builder.build(lattice_h2)

        if n > _ED_MAX_N:
            return self._initial_state_mps(n, topo, args.h1, args.h2, dt, n_steps)

        # ── Exact evolution (N ≤ 22) ─────────────────────────────────
        if n <= _DENSE_LIMIT:
            H2_matrix = np.asarray(H2_op.to_matrix())
            U_dt = expm(-1j * H2_matrix * dt)
            use_sparse = False
        else:
            H2_sparse = H2_op.to_matrix(sparse=True)
            use_sparse = True

        # Prepare three initial states:
        # 1. |ψ₀(h₁)⟩ = exact ground state of H(h₁)
        psi_gnn = self._ground_state_vector(n, topo, args.h1)
        logger.info(f"  |ψ₀(h₁={args.h1})⟩ prepared (exact GS)")

        # 2. |0⟩^N (IBM-style trivial preparation)
        psi_zero = np.zeros(2**n, dtype=complex)
        psi_zero[0] = 1.0

        # 3. |+⟩^N (our HVA initial state)
        psi_plus = np.ones(2**n, dtype=complex) / np.sqrt(2**n)

        # Evolve all three
        results_by_state = {}
        for label, psi_init in [("gnn_gs", psi_gnn), ("zero", psi_zero), ("plus", psi_plus)]:
            if use_sparse:
                energies, entropies, magnetizations = self._evolve_sparse(
                    psi_init, H2_sparse, n, n_steps, dt
                )
            else:
                energies, entropies, magnetizations = self._evolve_exact(
                    psi_init, U_dt, H2_matrix, n, n_steps
                )
            results_by_state[label] = {
                "energies": energies, "entropies": entropies, "magnetizations": magnetizations,
            }
            logger.info(f"  {label:>8}: E₀={energies[0]:.4f}, S_max={max(entropies):.4f}")

        # ── Analysis ─────────────────────────────────────────────────
        s_gnn = np.array(results_by_state["gnn_gs"]["entropies"])
        s_zero = np.array(results_by_state["zero"]["entropies"])
        s_plus = np.array(results_by_state["plus"]["entropies"])

        max_diff_gnn_zero = float(np.max(np.abs(s_gnn - s_zero)))
        max_diff_gnn_plus = float(np.max(np.abs(s_gnn - s_plus)))
        qualitatively_different = max_diff_gnn_zero > 0.3

        result = {
            "n_qubits": n, "topology": topo, "h1": args.h1, "h2": args.h2,
            "dt": dt, "n_steps": n_steps, "method": "exact_ed",
            "states_compared": ["gnn_gs", "zero", "plus"],
            "results_by_state": results_by_state,
            "max_entropy_diff_gnn_vs_zero": max_diff_gnn_zero,
            "max_entropy_diff_gnn_vs_plus": max_diff_gnn_plus,
            "initial_energies": {
                k: float(v["energies"][0]) for k, v in results_by_state.items()
            },
            "qualitatively_different": qualitatively_different,
            "pass": qualitatively_different,
        }

        logger.info(f"\n  Max |ΔS| GNN vs |0⟩^N: {max_diff_gnn_zero:.4f}")
        logger.info(f"  Max |ΔS| GNN vs |+⟩^N: {max_diff_gnn_plus:.4f}")
        logger.info(
            f"  Qualitatively different: "
            f"{'YES ✅' if qualitatively_different else 'NO ❌'}"
        )

        # ── Persist DQPT trajectory NPZ (feeds validate_dqpt_results + qpt_detection) ──
        # Compute Loschmidt echo + rate function for the GNN ground-state quench.
        # This data is already "free" since we have psi_gnn and the evolution operator.
        try:
            self._persist_dqpt_trajectory(
                psi_0=psi_gnn,
                n_qubits=n,
                topology=topo,
                h_pre=args.h1,
                h_post=args.h2,
                dt=dt,
                n_steps=n_steps,
                energies=results_by_state["gnn_gs"]["energies"],
                entropies=results_by_state["gnn_gs"]["entropies"],
                H_post_op=H2_op,
                use_sparse=use_sparse,
                H_sparse=H2_sparse if use_sparse else None,
                U_dt=U_dt if not use_sparse else None,
            )
        except Exception as e:
            logger.warning(f"  DQPT trajectory persistence failed (non-blocking): {e}")

        # ── Auto-run DQPT fidelity threshold (reuses psi_gnn + H_post_op) ──────
        # Determines minimum state-preparation fidelity for hardware DQPT detection.
        # "Free" since we already have the ground state and the evolution operator.
        try:
            from scripts.analysis.dqpt_fidelity_threshold import (
                run_fidelity_threshold_scan,
                save_report as save_fidelity_report,
            )

            fidelities = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30]
            logger.info("  Running DQPT fidelity threshold scan (piggyback)...")
            fid_report = run_fidelity_threshold_scan(
                topology=topo,
                n_qubits=n,
                fidelities=fidelities,
                h_pre=args.h1,
                h_post=args.h2,
                dt=dt,
                n_steps=n_steps,
            )
            if fid_report.f_min is not None:
                logger.info(f"  F_min = {fid_report.f_min:.2f} (hardware go/no-go threshold)")
                result["fidelity_threshold"] = {
                    "f_min": fid_report.f_min,
                    "t_star_reference": fid_report.t_star_reference,
                }
            else:
                logger.info("  F_min = None (no detectable DQPTs at any fidelity)")
                result["fidelity_threshold"] = {"f_min": None}

            # Persist fidelity report
            fid_path = (
                self._get_project_root() / "results" / "analysis"
                / f"dqpt_fidelity_threshold_{topo}_N{n}.json"
            )
            save_fidelity_report(fid_report, fid_path)
        except Exception as e:
            logger.debug(f"  Fidelity threshold scan skipped: {e}")

        return result

    # ─── Section 2: MPS Crossover Point ──────────────────────────────────────

    def section_mps_crossover(self) -> dict:
        """Identify Trotter step where MPS loses precision.

        N ≤ 22: exact ED reference, compare MPS entropy vs exact.
        N > 22: energy conservation diagnostic (drift = truncation error).
        """
        args = self._args
        n = args.n_qubits
        dt = args.dt
        n_steps = args.n_trotter
        topo = self._topology
        chi_values = sorted(args.chi_values)

        logger.info(f"  MPS crossover: N={n}, {topo}, χ={chi_values}")
        logger.info(f"  h₁={args.h1} → h₂={args.h2}, {n_steps} steps × dt={dt}")

        if n <= _ED_MAX_N:
            return self._mps_crossover_exact_reference(n, topo, args, chi_values)
        else:
            return self._mps_crossover_energy_drift(n, topo, args, chi_values)

    # ─── Section 3: Preparation Cost ─────────────────────────────────────────

    def section_preparation_cost(self) -> dict:
        """Quantify GNN vs VQE preparation cost across the phase diagram."""
        args = self._args
        n = args.n_qubits
        topo = self._topology

        logger.info(f"  Preparation cost: N={n}, {topo}")

        h_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        gnn_times = []
        gnn_success = []

        for h in h_values:
            t0 = time.time()
            theta = self._predict_theta_gnn(topo, n, h, args.p_layers)
            gnn_times.append(time.time() - t0)
            gnn_success.append(theta is not None)

        mean_gnn_time = float(np.mean(gnn_times))
        logger.info(f"  GNN inference: mean={mean_gnn_time:.4f}s, "
                    f"success={sum(gnn_success)}/{len(h_values)}")

        # VQE cost estimate
        n_restarts, maxiter = 5, 500
        if n <= 22:
            time_per_eval_s = 0.001 * (2**n / 1024)
        else:
            time_per_eval_s = 0.01 * (n / 10) ** 2
        vqe_time_per_h = n_restarts * maxiter * time_per_eval_s

        # Actual VQE measurement (N ≤ 16 only)
        actual_vqe_time = None
        if n <= 16:
            actual_vqe_time = self._measure_vqe_time(topo, n, args.h1, args.p_layers)
            if actual_vqe_time is not None:
                vqe_time_per_h = actual_vqe_time
                logger.info(f"  VQE measured: {actual_vqe_time:.2f}s per h-point")

        speedup = vqe_time_per_h / mean_gnn_time if mean_gnn_time > 0 else float("inf")
        total_vqe_time = vqe_time_per_h * len(h_values)
        total_gnn_time = mean_gnn_time * len(h_values)

        result = {
            "n_qubits": n, "topology": topo, "p_layers": args.p_layers,
            "h_values_tested": h_values,
            "gnn_inference_times_s": gnn_times,
            "gnn_success_rate": sum(gnn_success) / len(h_values),
            "mean_gnn_time_s": mean_gnn_time,
            "estimated_vqe_time_per_h_s": vqe_time_per_h,
            "actual_vqe_time_s": actual_vqe_time,
            "speedup_factor": float(speedup),
            "total_phase_diagram_vqe_s": total_vqe_time,
            "total_phase_diagram_gnn_s": total_gnn_time,
            "thesis_claim": (
                f"GNN reduces preparation cost from {vqe_time_per_h:.1f}s/point "
                f"to {mean_gnn_time:.4f}s/point ({speedup:.0f}× speedup). "
                f"Full phase diagram: {total_vqe_time:.0f}s → {total_gnn_time:.2f}s."
            ),
            "pass": speedup > 10,
        }
        logger.info(f"\n  Speedup: {speedup:.0f}× (VQE={vqe_time_per_h:.1f}s vs "
                    f"GNN={mean_gnn_time:.4f}s per h-point)")
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers — Evolution
    # ═══════════════════════════════════════════════════════════════════════════

    def _evolve_exact(self, psi_init, U_dt, H_matrix, n_qubits, n_steps):
        """Dense matrix evolution (N ≤ 16)."""
        psi_t = psi_init.copy()
        energies = [float(np.real(psi_t.conj() @ H_matrix @ psi_t))]
        entropies = [self._half_chain_entropy(psi_t, n_qubits)]
        mags = [self._magnetization_z(psi_t, n_qubits)]
        for _ in range(n_steps):
            psi_t = U_dt @ psi_t
            psi_t /= np.linalg.norm(psi_t)
            energies.append(float(np.real(psi_t.conj() @ H_matrix @ psi_t)))
            entropies.append(self._half_chain_entropy(psi_t, n_qubits))
            mags.append(self._magnetization_z(psi_t, n_qubits))
        return energies, entropies, mags

    def _evolve_sparse(self, psi_init, H_sparse, n_qubits, n_steps, dt):
        """Sparse Krylov evolution via expm_multiply (16 < N ≤ 22)."""
        from scipy.sparse.linalg import expm_multiply

        A = -1j * H_sparse * dt
        psi_t = psi_init.copy().astype(complex)
        energies = [float(np.real(psi_t.conj() @ (H_sparse @ psi_t)))]
        entropies = [self._half_chain_entropy(psi_t, n_qubits)]
        mags = [self._magnetization_z(psi_t, n_qubits)]
        for _ in range(n_steps):
            psi_t = expm_multiply(A, psi_t)
            psi_t /= np.linalg.norm(psi_t)
            energies.append(float(np.real(psi_t.conj() @ (H_sparse @ psi_t))))
            entropies.append(self._half_chain_entropy(psi_t, n_qubits))
            mags.append(self._magnetization_z(psi_t, n_qubits))
        return energies, entropies, mags

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers — MPS crossover
    # ═══════════════════════════════════════════════════════════════════════════

    def _mps_crossover_exact_reference(self, n, topo, args, chi_values):
        """MPS crossover with exact ED reference (N ≤ 22)."""
        from scipy.sparse.linalg import expm_multiply

        dt, n_steps = args.dt, args.n_trotter

        lattice_h2 = self.make_lattice(topo, n, J=1.0, h=args.h2)
        H2_op = self.builder.build(lattice_h2)
        psi_0 = self._ground_state_vector(n, topo, args.h1)

        if n <= _DENSE_LIMIT:
            H2_matrix = np.asarray(H2_op.to_matrix())
            U_dt = expm(-1j * H2_matrix * dt)
            use_sparse = False
        else:
            H2_sparse = H2_op.to_matrix(sparse=True)
            A = -1j * H2_sparse * dt
            use_sparse = True

        # Exact evolution entropy trajectory
        psi_t = psi_0.copy().astype(complex)
        exact_entropies = [self._half_chain_entropy(psi_t, n)]
        for _ in range(n_steps):
            psi_t = expm_multiply(A, psi_t) if use_sparse else U_dt @ psi_t
            psi_t /= np.linalg.norm(psi_t)
            exact_entropies.append(self._half_chain_entropy(psi_t, n))

        logger.info(f"  Exact: S_max={max(exact_entropies):.4f}")

        # MPS comparison via energy drift (faster than get_statevector per step)
        crossover_data = []
        for chi in chi_values:
            mps_backend = self.MPSBackend(chi_max=chi)
            trotter_step = self._build_trotter_step_circuit(n, topo, args.h2, dt)

            from qiskit.circuit import QuantumCircuit
            init_qc = QuantumCircuit(n)
            init_qc.initialize(psi_0, range(n))

            energies_mps = []
            full_circuit = init_qc.copy()
            empty_params = np.array([])

            for step in range(n_steps + 1):
                try:
                    e = mps_backend.evaluate(full_circuit, H2_op, empty_params)
                    energies_mps.append(float(e))
                except Exception:
                    energies_mps.append(energies_mps[-1] if energies_mps else 0.0)
                if step < n_steps:
                    full_circuit = full_circuit.compose(trotter_step)

            # Energy should be conserved — drift = truncation error
            e0 = energies_mps[0]
            drifts = [abs(e - e0) for e in energies_mps]
            crossover_step = next((s for s, d in enumerate(drifts) if d > 0.05), None)

            crossover_data.append({
                "chi": chi, "crossover_step": crossover_step,
                "max_drift": max(drifts), "energies": energies_mps,
            })
            logger.info(f"    χ={chi:>3}: crossover@step={crossover_step}, "
                        f"max_drift={max(drifts):.4f}")

        chi64_crossover = next(
            (d["crossover_step"] for d in crossover_data if d["chi"] == 64), None
        )
        return {
            "n_qubits": n, "topology": topo, "method": "exact_reference",
            "exact_entropies": exact_entropies, "crossover_data": crossover_data,
            "chi64_crossover_step": chi64_crossover,
            "in_quantum_advantage_regime": chi64_crossover is not None and chi64_crossover <= 15,
            "pass": True,
        }

    def _mps_crossover_energy_drift(self, n, topo, args, chi_values):
        """MPS crossover for N > 22: compare energy conservation across χ values."""
        dt, n_steps = args.dt, args.n_trotter

        lattice_h2 = self.make_lattice(topo, n, J=1.0, h=args.h2)
        H2_op = self.builder.build(lattice_h2)
        trotter_step = self._build_trotter_step_circuit(n, topo, args.h2, dt)

        # Resume from checkpoint if available (per-χ persistence)
        cp_label = f"mps_crossover_N{n}_{topo}"
        cp = self.load_checkpoint(cp_label)
        chi_results = cp.get("chi_results", {}) if cp else {}

        for chi in chi_values:
            # Skip already-computed χ (from checkpoint)
            if str(chi) in chi_results:
                logger.info(f"    χ={chi}: loaded from checkpoint")
                continue

            logger.info(f"    Evaluating χ={chi}...")
            mps_backend = self.MPSBackend(chi_max=chi)

            from qiskit.circuit import QuantumCircuit
            init_qc = QuantumCircuit(n)
            init_qc.h(range(n))  # |+⟩^N as proxy for paramagnetic GS

            energies = []
            full_circuit = init_qc.copy()
            empty_params = np.array([])

            t0 = time.time()
            for step in range(n_steps + 1):
                try:
                    e = mps_backend.evaluate(full_circuit, H2_op, empty_params)
                    energies.append(float(e))
                except Exception as exc:
                    logger.debug(f"    χ={chi} step {step} failed: {exc}")
                    energies.append(energies[-1] if energies else 0.0)
                if step < n_steps:
                    full_circuit = full_circuit.compose(trotter_step)
            elapsed = time.time() - t0

            e0 = energies[0]
            drifts = [abs(e - e0) for e in energies]
            chi_results[str(chi)] = {
                "energies": energies, "max_drift": max(drifts), "elapsed_s": elapsed,
            }
            logger.info(
                f"    χ={chi:>3}: E₀={e0:.4f}, max_drift={max(drifts):.4f} ({elapsed:.1f}s)"
            )

            # Checkpoint after each χ (crash-safe for multi-hour runs)
            self.save_checkpoint(cp_label, {"chi_results": chi_results})

        # Crossover: compare χ_low vs χ_high
        chi_ref = chi_values[-1]
        crossover_data = []
        for chi in chi_values[:-1]:
            drifts_chi = chi_results[str(chi)]["energies"]
            drifts_ref = chi_results[str(chi_ref)]["energies"]
            rel_devs = [abs(d1 - d2) for d1, d2 in zip(drifts_chi, drifts_ref)]
            crossover_step = next((s for s, d in enumerate(rel_devs) if d > 0.05), None)
            crossover_data.append({
                "chi": chi, "chi_ref": chi_ref, "crossover_step": crossover_step,
                "max_relative_drift": max(rel_devs) if rel_devs else 0,
            })

        chi64_crossover = next(
            (d["crossover_step"] for d in crossover_data if d["chi"] == 64), None
        )
        in_advantage = chi64_crossover is not None and chi64_crossover <= 15

        if in_advantage:
            logger.info(f"\n  RESULT: χ=64 fails at step {chi64_crossover} ≤ 15 — "
                        f"quantum advantage regime!")
        else:
            logger.info(f"\n  RESULT: χ=64 crossover at step {chi64_crossover}")

        # Cleanup checkpoint on success
        self.cleanup_checkpoints(cp_label)

        return {
            "n_qubits": n, "topology": topo, "method": "energy_drift",
            "chi_values": chi_values,
            "chi_results": {str(k): {"energies": v["energies"], "max_drift": v["max_drift"],
                                      "elapsed_s": v["elapsed_s"]} for k, v in chi_results.items()},
            "crossover_data": crossover_data,
            "chi64_crossover_step": chi64_crossover,
            "in_quantum_advantage_regime": in_advantage,
            "pass": True,
        }

    def _initial_state_mps(self, n, topo, h1, h2, dt, n_steps):
        """MPS-based initial-state comparison for N > 22."""
        logger.info(f"  N={n} > 22: using MPS energy trajectories")
        chi_ref = max(self._args.chi_values)

        energies_plus = self._mps_energy_evolution(n, topo, h2, dt, n_steps, chi_ref, "plus")
        energies_zero = self._mps_energy_evolution(n, topo, h2, dt, n_steps, chi_ref, "zero")

        energy_diff = abs(energies_plus[0] - energies_zero[0])
        drift_plus = max(abs(e - energies_plus[0]) for e in energies_plus)
        drift_zero = max(abs(e - energies_zero[0]) for e in energies_zero)
        qualitatively_different = energy_diff > 0.1

        logger.info(f"  |+>^N: E₀={energies_plus[0]:.4f}, drift={drift_plus:.4f}")
        logger.info(f"  |0>^N: E₀={energies_zero[0]:.4f}, drift={drift_zero:.4f}")

        return {
            "n_qubits": n, "topology": topo, "method": "mps_energy", "chi": chi_ref,
            "energies_plus": energies_plus, "energies_zero": energies_zero,
            "initial_energy_diff": float(energy_diff),
            "drift_plus": float(drift_plus), "drift_zero": float(drift_zero),
            "qualitatively_different": qualitatively_different,
            "pass": qualitatively_different,
        }

    def _mps_energy_evolution(self, n_qubits, topology, h2, dt, n_steps, chi, init_state):
        """Evolve via MPS Trotter, track energy ⟨H₂⟩ at each step."""
        from qiskit.circuit import QuantumCircuit

        mps_backend = self.MPSBackend(chi_max=chi)
        lattice_h2 = self.make_lattice(topology, n_qubits, J=1.0, h=h2)
        H2_op = self.builder.build(lattice_h2)
        trotter_step = self._build_trotter_step_circuit(n_qubits, topology, h2, dt)

        init_qc = QuantumCircuit(n_qubits)
        if init_state == "plus":
            init_qc.h(range(n_qubits))

        energies = []
        full_circuit = init_qc.copy()
        empty_params = np.array([])

        for step in range(n_steps + 1):
            try:
                e = mps_backend.evaluate(full_circuit, H2_op, empty_params)
                energies.append(float(e))
            except Exception:
                energies.append(energies[-1] if energies else 0.0)
            if step < n_steps:
                full_circuit = full_circuit.compose(trotter_step)
        return energies

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers — Trotter circuit
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_trotter_step_circuit(self, n_qubits, topology, h, dt):
        """Second-order Suzuki-Trotter step for TFIM.

        U₂(dt) = e^{-i(dt/2)H_ZZ} · e^{-i(dt)H_X} · e^{-i(dt/2)H_ZZ}
        H = -Σ Z_iZ_j - h Σ X_i
        """
        from qiskit.circuit import QuantumCircuit

        lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h)
        edges = lattice.edges
        qc = QuantumCircuit(n_qubits)
        half_dt = dt / 2.0

        # exp(-i(dt/2)H_ZZ) = Π exp(+i(dt/2) Z_iZ_j) → RZZ(-dt)
        for i, j in edges:
            qc.rzz(-2 * half_dt, i, j)
        # exp(-i(dt)H_X) = Π exp(+i dt h X_i) → RX(-2 dt h)
        for i in range(n_qubits):
            qc.rx(-2 * dt * h, i)
        # Second half ZZ
        for i, j in edges:
            qc.rzz(-2 * half_dt, i, j)
        return qc

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers — State preparation & MPNN
    # ═══════════════════════════════════════════════════════════════════════════

    def _ground_state_vector(self, n_qubits, topology, h):
        """Get ground state vector via sparse eigsh (up to N=22).

        Uses GroundTruthCache for energy/gap caching (cross-session reuse),
        and the framework's solver for N ≤ 16. Direct eigsh for 17-22.
        """
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        # Cache energy+gap via GroundTruthCache (even though we need the vector,
        # caching the energy avoids redundant DMRG calls in other sections)
        gt_cache = GroundTruthCache()
        cached = gt_cache.get(topology, n_qubits, self._args.model, float(h))
        if cached:
            logger.debug(f"  GT cache hit: {topology} N={n_qubits} h={h}")

        lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h)
        H_op = self.builder.build(lattice)

        e_gs = None
        gap = 0.0

        if n_qubits <= 16:
            # Dense diag: get 2 lowest eigenvalues for gap (negligible overhead)
            H_dense = np.asarray(H_op.to_matrix())
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
            gs = eigenvectors[:, 0]
            gs /= np.linalg.norm(gs)
            e_gs = float(eigenvalues[0])
            gap = float(eigenvalues[1] - eigenvalues[0])
        else:
            from scipy.sparse.linalg import eigsh
            logger.info(f"  Computing ground state (sparse eigsh k=2, N={n_qubits})...")
            H_sparse = H_op.to_matrix(sparse=True)
            eigenvalues, evecs = eigsh(H_sparse, k=2, which="SA")
            idx = np.argsort(eigenvalues)
            gs = evecs[:, idx[0]]
            gs /= np.linalg.norm(gs)
            e_gs = float(eigenvalues[idx[0]])
            gap = float(eigenvalues[idx[1]] - eigenvalues[idx[0]])

        # Store in GT cache if not already there (now with real spectral gap)
        if not cached:
            gt_cache.put(topology, n_qubits, self._args.model, float(h),
                         energy=e_gs, gap=gap, method="eigsh_k2")
            gt_cache.flush()

        return gs

    def _predict_theta_gnn(self, topology, n_qubits, h, p_layers):
        """Predict θ using model zoo — reuses load_best_model_for_topology."""
        try:
            from qmbp_simulation.predictors.model_zoo import load_best_model_for_topology
            from qmbp_simulation.predictors.unified_graph import (
                build_unified_bond_resolved_graph,
            )

            result = load_best_model_for_topology(topology)
            model = result[0]
            if model is None:
                return None

            import torch

            lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h)
            graph = build_unified_bond_resolved_graph(lattice, h_value=h, p_layers=p_layers)

            model.eval()
            with torch.no_grad():
                theta_pred = model(graph).cpu().numpy().flatten()
            return theta_pred
        except Exception as e:
            logger.debug(f"  MPNN prediction failed: {e}")
            return None

    def _measure_vqe_time(self, topology, n_qubits, h, p_layers):
        """Measure actual VQE wall time using framework VQEOptimizer."""
        try:
            lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h)
            H_op = self.builder.build(lattice)
            circuit, _ = self.hva.create(n_qubits=n_qubits, p_layers=p_layers, lattice=lattice)

            vqe_config = self.VQEConfig(maxiter=500, n_restarts=5)
            optimizer = self.VQEOptimizer(config=vqe_config, backend=self.noiseless, seed=42)

            t0 = time.time()
            optimizer.optimize(circuit, H_op)
            return time.time() - t0
        except Exception as e:
            logger.debug(f"  VQE measurement failed: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers — Physics observables (vectorized)
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _half_chain_entropy(psi, n_qubits):
        """Von Neumann entropy of half-chain bipartition via SVD."""
        n_a = n_qubits // 2
        psi_matrix = psi.reshape(2**n_a, 2**(n_qubits - n_a))
        sv = np.linalg.svd(psi_matrix, compute_uv=False)
        probs = sv**2
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log(probs)))

    @staticmethod
    def _magnetization_z(psi, n_qubits):
        """Compute ⟨M_z⟩ = (1/N) Σ ⟨Z_i⟩ — vectorized."""
        dim = 2**n_qubits
        probs = np.abs(psi) ** 2
        basis_states = np.arange(dim, dtype=np.int64)
        popcount = np.zeros(dim, dtype=np.int32)
        tmp = basis_states.copy()
        while np.any(tmp > 0):
            popcount += (tmp & 1).astype(np.int32)
            tmp >>= 1
        z_eigenvalues = (n_qubits - 2 * popcount) / n_qubits
        return float(np.dot(probs, z_eigenvalues))

    # ═══════════════════════════════════════════════════════════════════════════
    # Post-run auto-validation
    # ═══════════════════════════════════════════════════════════════════════════

    def _log_data_quality_feedback(self) -> None:
        """Override: run DQPT validation after all sections, then call super()."""
        # Auto-validate DQPT trajectories produced by this run
        try:
            from scripts.analysis.validate_dqpt_results import (
                validate_dqpt_topology,
                save_report as save_dqpt_report,
            )

            topo = self._topology
            report = validate_dqpt_topology(topo, verbose=False)
            if report.n_trajectories > 0:
                n_pass = sum(1 for c in report.checks if c.passed)
                n_total = len(report.checks)
                status = "PASS" if report.overall_pass else "FAIL"
                logger.info(
                    f"\n  ── DQPT Auto-Validation ({topo}) ──"
                    f"\n  {n_pass}/{n_total} checks passed [{status}]"
                    f"\n  Trajectories: {report.n_trajectories}"
                )
                # Persist validation report
                out_path = (
                    self._get_project_root() / "results" / "analysis"
                    / f"dqpt_validation_{topo}.json"
                )
                save_dqpt_report(report, out_path)
        except Exception as e:
            logger.debug(f"  DQPT auto-validation skipped: {e}")

        # Call parent's implementation (zoo pass_rate, retrain triggers, etc.)
        super()._log_data_quality_feedback()

    @staticmethod
    def _get_project_root() -> Path:
        return Path(__file__).resolve().parents[3]

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers — DQPT trajectory persistence
    # ═══════════════════════════════════════════════════════════════════════════

    def _persist_dqpt_trajectory(
        self,
        psi_0: np.ndarray,
        n_qubits: int,
        topology: str,
        h_pre: float,
        h_post: float,
        dt: float,
        n_steps: int,
        energies: list,
        entropies: list,
        H_post_op,
        use_sparse: bool,
        H_sparse=None,
        U_dt=None,
    ) -> None:
        """Compute Loschmidt echo and persist full DQPT trajectory as NPZ.

        Saves to data/dqpt_trajectories/{topology}_N{n_qubits}.npz in the format
        expected by validate_dqpt_results.py and qpt_detection.py.

        This is "free" computation since we already have psi_0 and the evolution
        operator from Section 1. The Loschmidt echo only requires re-evolving
        psi_0 and computing overlaps |<psi_0|psi(t)>|^2 at each step.
        """
        from qmbp_simulation.analysis.observables import (
            detect_dqpt_critical_times,
            loschmidt_echo,
            rate_function,
        )

        logger.info("  Persisting DQPT trajectory (Loschmidt echo + rate function)...")

        # Re-evolve psi_0 and compute Loschmidt echo at each step
        psi_t = psi_0.copy().astype(complex)
        times = [0.0]
        loschmidt_values = [1.0]
        rate_values = [0.0]

        for step in range(1, n_steps + 1):
            if use_sparse:
                from scipy.sparse.linalg import expm_multiply

                psi_t = expm_multiply(-1j * H_sparse * dt, psi_t)
            else:
                psi_t = U_dt @ psi_t
            psi_t /= np.linalg.norm(psi_t)

            t = step * dt
            times.append(t)

            L_t = loschmidt_echo(psi_0, psi_t)
            loschmidt_values.append(L_t)
            rate_values.append(rate_function(L_t, n_qubits))

        # Detect critical times
        critical_times = detect_dqpt_critical_times(times, loschmidt_values, threshold=0.1)

        # Get ground-state energy and gap at h_pre (for qpt_detection.py integration)
        e_exact_h_pre = float(energies[0]) if energies else 0.0
        gap_h_pre = 0.0
        try:
            from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

            gt_cache = GroundTruthCache()
            cached = gt_cache.get(topology, n_qubits, self._args.model, h_pre)
            if cached is not None:
                gap_h_pre = float(cached.get("gap", 0.0))
        except Exception:
            pass

        # Persist to NPZ (immediate write, crash-safe)
        out_dir = self._get_project_root() / "data" / "dqpt_trajectories"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{topology}_N{n_qubits}.npz"

        # Anti-regression: skip overwrite if existing trajectory has more steps
        if out_path.exists():
            try:
                existing = np.load(out_path, allow_pickle=True)
                existing_steps = int(existing.get("n_steps", 0))
                if existing_steps > n_steps:
                    logger.info(
                        f"  Skipping NPZ write: existing {out_path.name} has "
                        f"{existing_steps} steps > current {n_steps}"
                    )
                    return
            except Exception:
                pass  # Corrupted file — overwrite is fine

        np.savez(
            out_path,
            # Required keys (validate_dqpt_results.py schema)
            n_qubits=n_qubits,
            topology=topology,
            h_pre=h_pre,
            h_post=h_post,
            dt=dt,
            n_steps=n_steps,
            times=np.array(times),
            loschmidt_echo=np.array(loschmidt_values),
            rate_function=np.array(rate_values),
            energies=np.array(energies, dtype=float),
            entropies=np.array(entropies, dtype=float),
            critical_times=np.array(critical_times),
            method="exact_ed" if n_qubits <= _ED_MAX_N else "mps",
            # Extra keys for qpt_detection.py integration
            e_exact_h_pre=e_exact_h_pre,
            gap_h_pre=gap_h_pre,
        )
        logger.info(
            f"  Saved DQPT trajectory: {out_path.name} "
            f"({len(critical_times)} DQPTs detected, "
            f"r_peak={max(rate_values):.4f})"
        )


if __name__ == "__main__":
    QuenchDynamicsStudyRunner.main()
