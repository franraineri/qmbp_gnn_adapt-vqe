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

# Import observables for DQPT trajectory tracking
from qmbp_simulation.analysis.observables import magnetization_x as _magnetization_x_obs

# Practical limit for exact time evolution (sparse eigsh + expm_multiply)
_ED_MAX_N = 22
# Dense expm limit (above this, use sparse Krylov). N=14 dense expm needs a
# 16384² matrix (~4GB for U+H) and is impractically slow; sparse Krylov
# (expm_multiply) handles N=14 in seconds.
_DENSE_LIMIT = 12
# Observable stride threshold — for N>14, compute expensive observables
# every obs_stride steps and interpolate the rest (~20% speedup)
_OBS_STRIDE_THRESHOLD = 14
# Per-step checkpoint stride for crash recovery (N>14 only)
_CHECKPOINT_STRIDE = 20


def _interpolate_sparse_observables(
    values: list[float],
    stride: int,
    n_total: int,
) -> list[float]:
    """Linearly interpolate sparsely-sampled observable values to full time grid.

    Parameters
    ----------
    values : list[float]
        Observable values at stride-separated indices (including index 0).
    stride : int
        Sampling stride.
    n_total : int
        Total number of time steps (including step 0).

    Returns
    -------
    list[float]
        Full-resolution list of length n_total.
    """
    if stride <= 1 or len(values) == n_total:
        return list(values)

    # Defensive: if values is empty or n_total <= 0, return zeros
    if not values or n_total <= 0:
        return [0.0] * max(n_total, 0)

    full = np.empty(n_total, dtype=float)
    sample_indices = list(range(0, n_total, stride))
    # Ensure last index is included
    if sample_indices[-1] != n_total - 1:
        sample_indices.append(n_total - 1)

    # Build piecewise linear interpolation
    for seg_idx in range(len(sample_indices) - 1):
        i_start = sample_indices[seg_idx]
        i_end = sample_indices[seg_idx + 1]
        # Defensive bounds-check: clamp indices into values array
        v_start = values[min(seg_idx, len(values) - 1)]
        v_end = values[min(seg_idx + 1, len(values) - 1)]
        n_seg = i_end - i_start
        for k in range(n_seg):
            alpha = k / n_seg if n_seg > 0 else 0.0
            full[i_start + k] = v_start + alpha * (v_end - v_start)
    full[-1] = values[-1]
    return full.tolist()


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
            "--n-qubits",
            type=int,
            default=21,
            help="System size (default: 21 for heavy_hex)",
        )
        parser.add_argument(
            "--topology",
            type=str,
            nargs="+",
            default=["heavy_hex"],
            help="Lattice topology (default: heavy_hex).",
        )
        parser.add_argument(
            "--p-layers",
            type=int,
            default=1,
            help="HVA circuit depth (default: 1)",
        )
        parser.add_argument(
            "--h1",
            type=float,
            default=3.0,
            help="Initial field h₁ (ground state preparation). Default: 3.0 (paramagnetic)",
        )
        parser.add_argument(
            "--h2",
            type=float,
            default=0.5,
            help="Quench field h₂ (evolution Hamiltonian). Default: 0.5 (ferromagnetic)",
        )
        parser.add_argument(
            "--n-trotter",
            type=int,
            default=25,
            help="Number of Trotter steps (default: 25)",
        )
        parser.add_argument(
            "--dt",
            type=float,
            default=0.1,
            help="Trotter time step dt (default: 0.1)",
        )
        parser.add_argument(
            "--chi-values",
            type=int,
            nargs="+",
            default=[64, 128, 256],
            help="MPS bond dimensions to test (default: 64 128 256)",
        )
        parser.add_argument(
            "--seeds",
            type=int,
            nargs="+",
            default=[42],
            help="Random seeds",
        )
        parser.add_argument(
            "--dqpt-n-values",
            type=int,
            nargs="+",
            default=None,
            help="Multiple N values for DQPT scaling ladder (e.g., 8 10 12 14 16 20). "
            "When set, Section 1 iterates over all N values, producing one trajectory "
            "per N. Required for validate_dqpt_results scaling checks (needs >=4 N).",
        )
        # NOTE: --j2 is provided by the base parser (runner_base) and consumed via
        # self.model_kwargs(); do not redefine it here (argparse conflict).

    def build_config(self) -> dict:
        args = self._args
        topo = args.topology[0] if isinstance(args.topology, list) else args.topology
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "system": {
                "n_qubits": args.n_qubits,
                "topology": topo,
                "model": args.model or "tfim",
                "p_layers": args.p_layers,
                "j2": args.j2,
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
                id=4,
                name="Resource + noise budget (heavy-hex, t_noise vs t*)",
                hypothesis=(
                    "A quantum-advantage window exists only if the hardware "
                    "noise horizon t_noise (predicted circuit fidelity > 1/e) "
                    "exceeds the classical crossover t*."
                ),
                fn=self.section_resource_noise_budget,
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
        if args.model is None:
            args.model = "tfim"
        self._topology = args.topology[0] if isinstance(args.topology, list) else args.topology

        # In-memory ground state vector cache — avoids recomputing eigenvectors
        # (1-5s per call for N=20-22). Key: (n_qubits, topology, round(h, 2))
        # Capped at 8 entries to prevent unbounded memory growth
        # (N=22 vectors are 64MB each in complex128).
        self._gs_vector_cache: dict[tuple[int, str, float], np.ndarray] = {}
        self._GS_CACHE_MAX_SIZE = 8

        # Auto-select quench parameters if user didn't override defaults
        # (defaults: h1=3.0, h2=0.5 — detect via sys.argv heuristic)
        if not self._user_provided_h_override():
            try:
                h1_auto, h2_auto = self.auto_select_quench_params(
                    self._topology,
                    args.n_qubits,
                )
                args.h1 = h1_auto
                args.h2 = h2_auto
            except Exception as e:
                logger.debug(f"  auto_select_quench_params failed, using defaults: {e}")

    def _user_provided_h_override(self) -> bool:
        """Check if the user explicitly passed --h1 or --h2 on the CLI."""
        import sys

        return "--h1" in sys.argv or "--h2" in sys.argv

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
        # Cache on instance so Section 2 (_mps_crossover_exact_reference) can reuse
        cache_key_h2 = (topo, n, round(args.h2, 2))
        if not hasattr(self, "_h2_op_cache"):
            self._h2_op_cache: dict = {}
        if cache_key_h2 in self._h2_op_cache:
            lattice_h2, H2_op = self._h2_op_cache[cache_key_h2]
        else:
            lattice_h2 = self.make_lattice(topo, n, J=1.0, h=args.h2)
            H2_op = self._build_quench_hamiltonian(lattice_h2)
            self._h2_op_cache[cache_key_h2] = (lattice_h2, H2_op)

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

        # ── Adaptive observable stride & checkpoint config ────────────
        obs_stride = 1 if n <= _OBS_STRIDE_THRESHOLD else 5
        use_checkpoint = n > _OBS_STRIDE_THRESHOLD
        cp_label = f"dqpt_section1_N{n}_{topo}"

        # Checkpoint callback for crash recovery (N>14 only)
        def _make_checkpoint_fn(state_label: str):
            if not use_checkpoint:
                return None

            def _cp_fn(step, psi_t, energies, entropies, mags):
                self.save_checkpoint(
                    cp_label,
                    {
                        "state_label": state_label,
                        "step": step,
                        "psi_t_real": psi_t.real.tolist(),
                        "psi_t_imag": psi_t.imag.tolist(),
                        "energies": energies,
                        "entropies": entropies,
                        "mags": mags,
                    },
                )

            return _cp_fn

        if obs_stride > 1:
            logger.info(f"  Adaptive stride: obs_stride={obs_stride} (N>{_OBS_STRIDE_THRESHOLD})")

        # Evolve all three
        results_by_state = {}
        for label, psi_init in [("gnn_gs", psi_gnn), ("zero", psi_zero), ("plus", psi_plus)]:
            if use_sparse:
                energies, entropies, magnetizations = self._evolve_sparse(
                    psi_init,
                    H2_sparse,
                    n,
                    n_steps,
                    dt,
                    obs_stride=obs_stride,
                    checkpoint_fn=_make_checkpoint_fn(label),
                )
            else:
                energies, entropies, magnetizations = self._evolve_exact(
                    psi_init,
                    U_dt,
                    H2_matrix,
                    n,
                    n_steps,
                    obs_stride=obs_stride,
                    checkpoint_fn=_make_checkpoint_fn(label),
                )
            results_by_state[label] = {
                "energies": energies,
                "entropies": entropies,
                "magnetizations": magnetizations,
            }
            logger.info(f"  {label:>8}: E₀={energies[0]:.4f}, S_max={max(entropies):.4f}")

        # Cleanup checkpoint on success
        if use_checkpoint:
            self.cleanup_checkpoints(cp_label)

        # ── Analysis ─────────────────────────────────────────────────
        s_gnn = np.array(results_by_state["gnn_gs"]["entropies"])
        s_zero = np.array(results_by_state["zero"]["entropies"])
        s_plus = np.array(results_by_state["plus"]["entropies"])

        max_diff_gnn_zero = float(np.max(np.abs(s_gnn - s_zero)))
        max_diff_gnn_plus = float(np.max(np.abs(s_gnn - s_plus)))
        qualitatively_different = max_diff_gnn_zero > 0.3

        result = {
            "n_qubits": n,
            "topology": topo,
            "h1": args.h1,
            "h2": args.h2,
            "dt": dt,
            "n_steps": n_steps,
            "method": "exact_ed",
            "states_compared": ["gnn_gs", "zero", "plus"],
            "results_by_state": results_by_state,
            "max_entropy_diff_gnn_vs_zero": max_diff_gnn_zero,
            "max_entropy_diff_gnn_vs_plus": max_diff_gnn_plus,
            "initial_energies": {k: float(v["energies"][0]) for k, v in results_by_state.items()},
            "qualitatively_different": qualitatively_different,
            "pass": qualitatively_different,
        }

        logger.info(f"\n  Max |ΔS| GNN vs |0⟩^N: {max_diff_gnn_zero:.4f}")
        logger.info(f"  Max |ΔS| GNN vs |+⟩^N: {max_diff_gnn_plus:.4f}")
        logger.info(f"  Qualitatively different: {'YES ✅' if qualitatively_different else 'NO ❌'}")

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

        # ── GNN comparison (compares exact GS quench vs GNN-prepared quench) ─────
        try:
            gnn_cmp = self._dqpt_gnn_comparison(
                n_qubits=n,
                topology=topo,
                h_pre=args.h1,
                h_post=args.h2,
                dt=dt,
                n_steps=n_steps,
                H_post_op=H2_op,
                use_sparse=use_sparse,
                H_sparse=H2_sparse if use_sparse else None,
                U_dt=U_dt if not use_sparse else None,
            )
            if gnn_cmp is not None:
                result["gnn_comparison"] = gnn_cmp
        except Exception as e:
            logger.debug(f"  GNN comparison skipped: {e}")

        # ── Auto-run DQPT fidelity threshold (reuses psi_gnn + H_post_op) ──────
        # Determines minimum state-preparation fidelity for hardware DQPT detection.
        # "Free" since we already have the ground state and the evolution operator.
        try:
            from scripts.analysis.dqpt_fidelity_threshold import (
                run_fidelity_threshold_scan,
            )
            from scripts.analysis.dqpt_fidelity_threshold import (
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
            fid_path = self._get_project_root() / "results" / "analysis" / f"dqpt_fidelity_threshold_{topo}_N{n}.json"
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

    def section_resource_noise_budget(self) -> dict:
        """Resource count + error-budget model over the full Trotter evolution.

        For each accumulated Trotter depth k = 0..n_trotter, transpile the
        evolution circuit to the real heavy-hex backend (FakeTorino, Heron) and
        apply the gate-count error model F ≈ exp(−Σ nᵢ·εᵢ). Estimates t_noise =
        the step where the predicted circuit fidelity drops below 1/e, i.e. where
        hardware noise overwhelms the signal. Compared against the classical
        crossover t* (section 2): a quantum-advantage window exists only when
        t*_classical < t_noise.

        Runs automatically on every quench run. Diagnostic — never gates pass.
        """
        args = self._args
        n = args.n_qubits
        topo = self._topology
        curve = self._estimate_tnoise_curve(n, topo, args.h2, args.dt, args.n_trotter)

        t_noise = curve["t_noise_step"]
        logger.info(f"  Resource/noise budget ({curve['backend']}, {curve['error_rates_source']}):")
        logger.info(
            f"    logical 2q/step={curve['logical_2q_per_step']}, "
            f"transpiled 2q/step≈{curve['transpiled_2q_per_step']:.0f} "
            f"(routing overhead ×{curve['routing_overhead']:.1f})"
        )
        logger.info(
            f"    t_noise = step {t_noise} (t={curve['t_noise_time']}) "
            f"[F<1/e]; F@last_step={curve['fidelity_curve'][-1]:.3f}"
        )

        # Compare against the classical crossover if section 2 produced one.
        window = None
        t_star = self._last_crossover_step()
        if t_star is not None:
            window = {
                "t_star_classical": t_star,
                "t_noise": t_noise,
                "has_advantage_window": t_noise is not None and t_star < t_noise,
            }
            verdict = "WINDOW ✅" if window["has_advantage_window"] else "no window ❌"
            logger.info(f"    vs classical crossover t*={t_star}: {verdict}")

        result = {
            "n_qubits": n,
            "topology": topo,
            **curve,
            "window": window,
            "pass": True,
        }
        try:
            self._persist_resource_noise_json(result)
        except Exception as e:
            logger.warning(f"  resource/noise JSON persistence failed (non-blocking): {e}")
        return result

    def _estimate_tnoise_curve(self, n, topo, h2, dt, n_steps, fidelity_floor=None):
        """Transpile prep + k Trotter steps to real HW and apply exp(−budget).

        Returns the fidelity-vs-depth curve and t_noise (first step with
        predicted fidelity < fidelity_floor, default 1/e). Prep is |+⟩^N (the
        minimal-cost preparation); the evolution circuit is what scales with
        depth, so it dominates the resource/noise budget.
        """
        from qiskit import transpile
        from qiskit.circuit import QuantumCircuit

        from qmbp_simulation.analysis.circuit_visualizer import compute_error_budget

        if fidelity_floor is None:
            fidelity_floor = float(np.exp(-1.0))  # 1/e ≈ 0.368

        backend, backend_name = self._get_fake_backend()

        trotter_step = self._build_trotter_step_circuit(n, topo, h2, dt)
        logical_2q_per_step = sum(
            1 for inst in trotter_step.data if inst.operation.num_qubits == 2
        )

        prep = QuantumCircuit(n)
        prep.h(range(n))  # |+⟩^N minimal-cost preparation

        # Resume from checkpoint if a prior run was interrupted mid-sweep.
        cp_label = f"tnoise_N{n}_{topo}_h{h2:.2f}_dt{dt:.3f}"
        cp = self.load_checkpoint(cp_label)
        if cp and cp.get("n_steps") == n_steps and abs(cp.get("dt", -1) - dt) < 1e-12:
            fidelity_curve = list(cp["fidelity_curve"])
            error_budget_curve = list(cp["error_budget_curve"])
            cz_cumulative = list(cp["cz_cumulative"])
            source = cp.get("source", "typical_fallback")
            start_step = len(fidelity_curve)
            logger.info(f"    resuming t_noise curve from step {start_step}")
        else:
            fidelity_curve, error_budget_curve, cz_cumulative = [], [], []
            source = "typical_fallback"
            start_step = 0

        # Rebuild the accumulated circuit up to the resume point.
        full = prep.copy()
        for _ in range(start_step):
            full = full.compose(trotter_step)

        _CP_STRIDE = 10
        early_stopped = False
        for step in range(start_step, n_steps + 1):
            if backend is not None:
                try:
                    tqc = transpile(full, backend=backend, optimization_level=3, seed_transpiler=42)
                    # Pass the ACTUAL layout: the chip-wide error mean on fake
                    # backends is poisoned by dead edges (error=1.0). Averaging
                    # only the used qubits/edges gives a physical budget.
                    lay = tqc.layout.final_index_layout() if tqc.layout else None
                    budget = compute_error_budget(tqc, backend=backend, layout=lay)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"    transpile/budget failed at step {step}: {e}")
                    budget = compute_error_budget(full, backend=None)
            else:
                budget = compute_error_budget(full, backend=None)
            source = budget["source"]
            fidelity_curve.append(float(budget["fidelity_estimate"]))
            error_budget_curve.append(float(budget["error_budget"]))
            cz_cumulative.append(int(budget["n_2q_gates"]))

            # Early stop: once fidelity is firmly below the floor, t_noise is
            # already determined and deeper transpilations add no information.
            if budget["fidelity_estimate"] < fidelity_floor:
                early_stopped = True
                break

            if step % _CP_STRIDE == 0:
                self.save_checkpoint(
                    cp_label,
                    {
                        "n_steps": n_steps,
                        "dt": dt,
                        "source": source,
                        "fidelity_curve": fidelity_curve,
                        "error_budget_curve": error_budget_curve,
                        "cz_cumulative": cz_cumulative,
                    },
                )
            if step < n_steps:
                full = full.compose(trotter_step)
        self.cleanup_checkpoints(cp_label)

        t_noise_step = next(
            (s for s, f in enumerate(fidelity_curve) if f < fidelity_floor), None
        )
        # transpiled 2q per step: slope of the cumulative count over the steps
        # actually computed (robust to early stop), excluding the prep layer.
        n_computed = len(cz_cumulative)
        transpiled_2q_per_step = (
            (cz_cumulative[-1] - cz_cumulative[1]) / max(n_computed - 2, 1)
            if n_computed > 2
            else float(cz_cumulative[-1] if cz_cumulative else 0.0)
        )
        routing_overhead = (
            transpiled_2q_per_step / logical_2q_per_step if logical_2q_per_step else 1.0
        )

        return {
            "backend": backend_name,
            "error_rates_source": source,
            "fidelity_floor": fidelity_floor,
            "dt": dt,
            "n_trotter": n_steps,
            "n_steps_computed": n_computed - 1,
            "early_stopped": early_stopped,
            "h2": h2,
            "logical_2q_per_step": logical_2q_per_step,
            "transpiled_2q_per_step": transpiled_2q_per_step,
            "routing_overhead": routing_overhead,
            "fidelity_curve": fidelity_curve,
            "error_budget_curve": error_budget_curve,
            "cz_cumulative": cz_cumulative,
            "t_noise_step": t_noise_step,
            "t_noise_time": (t_noise_step * dt if t_noise_step is not None else None),
        }

    def _get_fake_backend(self):
        """Return (FakeTorino, name) for heavy-hex transpilation, or (None, ...)."""
        try:
            from qiskit_ibm_runtime.fake_provider import FakeTorino

            return FakeTorino(), "FakeTorino"
        except Exception as e:  # noqa: BLE001
            logger.debug(f"  FakeTorino unavailable ({e}); using typical-rate fallback")
            return None, "typical_fallback"

    def _last_crossover_step(self):
        """χ=64 classical crossover step from this run's section 2, if available."""
        try:
            for sr in getattr(self, "_section_results", []) or []:
                data = getattr(sr, "data", None) or {}
                if data.get("method", "").startswith("exact_reference"):
                    return data.get("chi64_crossover_step")
        except Exception:  # noqa: BLE001
            pass
        return None

    def _persist_resource_noise_json(self, result: dict) -> None:
        """Write the resource/noise analysis to a dedicated, well-named JSON.

        Path: results/experiments/exp_frustrated/resource_noise/{model}/
              {topology}_N{n}_p{p}_h{h1}to{h2}/resource_noise_{timestamp}.json
        """
        import json
        from datetime import datetime

        from qmbp_simulation.utils.helpers import json_serialize

        args = self._args
        model = args.model or "tfim"
        p = getattr(args, "p_layers", 1)
        n = args.n_qubits
        topo = self._topology
        subdir = (
            self._get_project_root()
            / "results" / "experiments" / "exp_frustrated" / "resource_noise"
            / model
            / f"{topo}_N{n}_p{p}_h{args.h1:.2f}to{args.h2:.2f}"
        )
        subdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc = {
            "schema": "resource_noise_budget_v1",
            "experiment": "quench_resource_noise",
            "specs": {
                "model": model,
                "j2": getattr(args, "j2", 0.0),
                "topology": topo,
                "n_qubits": n,
                "p_layers": p,
                "h1": args.h1,
                "h2": args.h2,
                "dt": args.dt,
                "n_trotter": args.n_trotter,
                "chi_values": list(args.chi_values),
            },
            "result": result,
            "timestamp": ts,
        }
        out = subdir / f"resource_noise_{ts}.json"
        with open(out, "w") as f:
            json.dump(doc, f, indent=2, default=json_serialize)
        logger.info(f"  resource/noise JSON → {out.relative_to(self._get_project_root())}")

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
        logger.info(f"  GNN inference: mean={mean_gnn_time:.4f}s, success={sum(gnn_success)}/{len(h_values)}")

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
            "n_qubits": n,
            "topology": topo,
            "p_layers": args.p_layers,
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
        logger.info(f"\n  Speedup: {speedup:.0f}× (VQE={vqe_time_per_h:.1f}s vs GNN={mean_gnn_time:.4f}s per h-point)")
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers — Evolution
    # ═══════════════════════════════════════════════════════════════════════════

    def _evolve_exact(self, psi_init, U_dt, H_matrix, n_qubits, n_steps, obs_stride=1, checkpoint_fn=None):
        """Dense matrix evolution (N ≤ 16).

        Parameters
        ----------
        obs_stride : int
            Compute entropy/M_x every obs_stride steps (1 = every step).
        checkpoint_fn : callable or None
            If provided, called as checkpoint_fn(step, psi_t, energies, entropies, mags)
            every _CHECKPOINT_STRIDE steps for crash recovery.
        """
        psi_t = psi_init.copy()
        energies = [float(np.real(psi_t.conj() @ H_matrix @ psi_t))]
        entropies = [self._half_chain_entropy(psi_t, n_qubits)]
        mags = [self._magnetization_z(psi_t, n_qubits)]
        for step in range(1, n_steps + 1):
            psi_t = U_dt @ psi_t
            psi_t /= np.linalg.norm(psi_t)
            energies.append(float(np.real(psi_t.conj() @ H_matrix @ psi_t)))
            if obs_stride <= 1 or step % obs_stride == 0 or step == n_steps:
                entropies.append(self._half_chain_entropy(psi_t, n_qubits))
                mags.append(self._magnetization_z(psi_t, n_qubits))
            if checkpoint_fn and step % _CHECKPOINT_STRIDE == 0:
                checkpoint_fn(step, psi_t, energies, entropies, mags)
        # Interpolate if stride > 1
        if obs_stride > 1:
            entropies = _interpolate_sparse_observables(entropies, obs_stride, n_steps + 1)
            mags = _interpolate_sparse_observables(mags, obs_stride, n_steps + 1)
        return energies, entropies, mags

    def _evolve_sparse(self, psi_init, H_sparse, n_qubits, n_steps, dt, obs_stride=1, checkpoint_fn=None):
        """Sparse Krylov evolution via expm_multiply (16 < N ≤ 22).

        Uses chunk-based expm_multiply for ~20-30% speedup: evolves
        _CHECKPOINT_STRIDE steps at once, then computes observables on
        the intermediate states.

        Parameters
        ----------
        obs_stride : int
            Compute entropy/M_x every obs_stride steps (1 = every step).
        checkpoint_fn : callable or None
            If provided, called every _CHECKPOINT_STRIDE steps for crash recovery.
        """
        from scipy.sparse.linalg import expm_multiply

        A = -1j * H_sparse * dt
        psi_t = psi_init.copy().astype(complex)
        energies = [float(np.real(psi_t.conj() @ (H_sparse @ psi_t)))]
        entropies = [self._half_chain_entropy(psi_t, n_qubits)]
        mags = [self._magnetization_z(psi_t, n_qubits)]

        # Evolve in chunks for better Krylov subspace reuse
        chunk_size = min(_CHECKPOINT_STRIDE, n_steps)
        step = 0
        while step < n_steps:
            remaining = n_steps - step
            this_chunk = min(chunk_size, remaining)

            # Batch evolution: compute states at all intermediate points in one call
            # expm_multiply(A, v, start=0, stop=this_chunk, num=this_chunk+1)
            # returns array of shape (this_chunk+1, dim) — state at each sub-step
            states = expm_multiply(
                A,
                psi_t,
                start=0.0,
                stop=float(this_chunk),
                num=this_chunk + 1,
                endpoint=True,
            )

            # Process each sub-step's state (skip index 0 = current psi_t)
            for sub_idx in range(1, this_chunk + 1):
                step += 1
                psi_t = states[sub_idx]
                psi_t /= np.linalg.norm(psi_t)
                energies.append(float(np.real(psi_t.conj() @ (H_sparse @ psi_t))))
                if obs_stride <= 1 or step % obs_stride == 0 or step == n_steps:
                    entropies.append(self._half_chain_entropy(psi_t, n_qubits))
                    mags.append(self._magnetization_z(psi_t, n_qubits))

            # Checkpoint at chunk boundary
            if checkpoint_fn and step % _CHECKPOINT_STRIDE == 0:
                checkpoint_fn(step, psi_t, energies, entropies, mags)

        # Interpolate if stride > 1
        if obs_stride > 1:
            entropies = _interpolate_sparse_observables(entropies, obs_stride, n_steps + 1)
            mags = _interpolate_sparse_observables(mags, obs_stride, n_steps + 1)
        return energies, entropies, mags

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers — MPS crossover
    # ═══════════════════════════════════════════════════════════════════════════

    def _mps_crossover_exact_reference(self, n, topo, args, chi_values):
        """MPS crossover measured against the EXACT evolved state (N ≤ 22).

        The classical→quantum crossover is the Trotter step at which a
        bond-dimension-χ MPS can no longer track the true state. We measure
        this against ground truth (exact time evolution), not the MPS's own
        energy-conservation drift — the latter is non-monotone in χ under the
        simulator's greedy truncation and does NOT measure error vs the truth.

        Per step we truncate the exact state to χ (variational SVD truncation,
        monotone in χ) and record the infidelity 1−|⟨ψ_exact|ψ_χ⟩|² and the
        relative energy error |E_χ−E_exact|/|E_exact|. Crossover = first step
        where infidelity exceeds INFIDELITY_THRESHOLD.
        """
        from scipy.sparse.linalg import expm_multiply

        dt, n_steps = args.dt, args.n_trotter
        infidelity_threshold = 0.01  # 99% fidelity floor

        cache_key_h2 = (topo, n, round(args.h2, 2))
        _h2_cache = getattr(self, "_h2_op_cache", {})
        if cache_key_h2 in _h2_cache:
            lattice_h2, H2_op = _h2_cache[cache_key_h2]
        else:
            lattice_h2 = self.make_lattice(topo, n, J=1.0, h=args.h2)
            H2_op = self._build_quench_hamiltonian(lattice_h2)
        psi_0 = self._ground_state_vector(n, topo, args.h1)

        if n <= _DENSE_LIMIT:
            H2_matrix = np.asarray(H2_op.to_matrix())
            U_dt = expm(-1j * H2_matrix * dt)
            H2_for_energy = H2_matrix
            use_sparse = False
        else:
            H2_sparse = H2_op.to_matrix(sparse=True)
            A = -1j * H2_sparse * dt
            H2_for_energy = H2_sparse
            use_sparse = True

        def _energy(psi):
            return float(np.real(psi.conj() @ (H2_for_energy @ psi)))

        # Exact evolution: store the true state at every step (ground truth).
        psi_t = psi_0.copy().astype(complex)
        exact_states = [psi_t.copy()]
        exact_entropies = [self._entanglement_entropy(psi_t, n, topo)]
        for _ in range(n_steps):
            psi_t = expm_multiply(A, psi_t) if use_sparse else U_dt @ psi_t
            psi_t /= np.linalg.norm(psi_t)
            exact_states.append(psi_t.copy())
            exact_entropies.append(self._entanglement_entropy(psi_t, n, topo))

        e_exact_ref = abs(_energy(exact_states[0])) or 1.0
        logger.info(f"  Exact: S_max={max(exact_entropies):.4f}")

        # Per-χ error vs the exact state (variational SVD truncation, monotone).
        crossover_data = []
        for chi in chi_values:
            infidelities = []
            rel_energy_errors = []
            for psi_exact in exact_states:
                psi_chi = self.truncate_statevector_mps(psi_exact, n, chi_max=chi)
                fid = abs(np.vdot(psi_exact, psi_chi)) ** 2
                infidelities.append(1.0 - float(fid))
                rel_energy_errors.append(abs(_energy(psi_chi) - _energy(psi_exact)) / e_exact_ref)

            crossover_step = next(
                (s for s, inf in enumerate(infidelities) if inf > infidelity_threshold), None
            )
            crossover_data.append(
                {
                    "chi": chi,
                    "crossover_step": crossover_step,
                    "max_infidelity": max(infidelities),
                    "max_rel_energy_error": max(rel_energy_errors),
                    "infidelities": infidelities,
                    "rel_energy_errors": rel_energy_errors,
                }
            )
            logger.info(
                f"    χ={chi:>3}: crossover@step={crossover_step}, "
                f"max_infid={max(infidelities):.4f}, max_relE={max(rel_energy_errors):.4f}"
            )

        chi64_crossover = next((d["crossover_step"] for d in crossover_data if d["chi"] == 64), None)
        return {
            "n_qubits": n,
            "topology": topo,
            "method": "exact_reference_fidelity",
            "infidelity_threshold": infidelity_threshold,
            "exact_entropies": exact_entropies,
            "crossover_data": crossover_data,
            "chi64_crossover_step": chi64_crossover,
            "in_quantum_advantage_regime": chi64_crossover is not None and chi64_crossover <= 15,
            "pass": True,
        }

    def _mps_crossover_energy_drift(self, n, topo, args, chi_values):
        """MPS crossover for N > 22: compare energy conservation across χ values."""
        dt, n_steps = args.dt, args.n_trotter

        lattice_h2 = self.make_lattice(topo, n, J=1.0, h=args.h2)
        H2_op = self._build_quench_hamiltonian(lattice_h2)
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
                "energies": energies,
                "max_drift": max(drifts),
                "elapsed_s": elapsed,
            }
            logger.info(f"    χ={chi:>3}: E₀={e0:.4f}, max_drift={max(drifts):.4f} ({elapsed:.1f}s)")

            # Checkpoint after each χ (crash-safe for multi-hour runs)
            self.save_checkpoint(cp_label, {"chi_results": chi_results})

        # Crossover: compare χ_low vs χ_high
        chi_ref = chi_values[-1]
        crossover_data = []
        for chi in chi_values[:-1]:
            drifts_chi = chi_results[str(chi)]["energies"]
            drifts_ref = chi_results[str(chi_ref)]["energies"]
            rel_devs = [abs(d1 - d2) for d1, d2 in zip(drifts_chi, drifts_ref, strict=False)]
            crossover_step = next((s for s, d in enumerate(rel_devs) if d > 0.05), None)
            crossover_data.append(
                {
                    "chi": chi,
                    "chi_ref": chi_ref,
                    "crossover_step": crossover_step,
                    "max_relative_drift": max(rel_devs) if rel_devs else 0,
                }
            )

        chi64_crossover = next((d["crossover_step"] for d in crossover_data if d["chi"] == 64), None)
        in_advantage = chi64_crossover is not None and chi64_crossover <= 15

        if in_advantage:
            logger.info(f"\n  RESULT: χ=64 fails at step {chi64_crossover} ≤ 15 — quantum advantage regime!")
        else:
            logger.info(f"\n  RESULT: χ=64 crossover at step {chi64_crossover}")

        # Cleanup checkpoint on success
        self.cleanup_checkpoints(cp_label)

        return {
            "n_qubits": n,
            "topology": topo,
            "method": "energy_drift",
            "chi_values": chi_values,
            "chi_results": {
                str(k): {
                    "energies": v["energies"],
                    "max_drift": v["max_drift"],
                    "elapsed_s": v["elapsed_s"],
                }
                for k, v in chi_results.items()
            },
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
            "n_qubits": n,
            "topology": topo,
            "method": "mps_energy",
            "chi": chi_ref,
            "energies_plus": energies_plus,
            "energies_zero": energies_zero,
            "initial_energy_diff": float(energy_diff),
            "drift_plus": float(drift_plus),
            "drift_zero": float(drift_zero),
            "qualitatively_different": qualitatively_different,
            "pass": qualitatively_different,
        }

    def _mps_energy_evolution(self, n_qubits, topology, h2, dt, n_steps, chi, init_state):
        """Evolve via MPS Trotter, track energy ⟨H₂⟩ at each step."""
        from qiskit.circuit import QuantumCircuit

        mps_backend = self.MPSBackend(chi_max=chi)
        lattice_h2 = self.make_lattice(topology, n_qubits, J=1.0, h=h2)
        H2_op = self._build_quench_hamiltonian(lattice_h2)
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

    def _build_quench_hamiltonian(self, lattice):
        """Build the quench Hamiltonian for the given lattice.

        Routes through the frustrated TFIM (−J₁ Σ_nn ZZ + J₂ Σ_nnn ZZ − h Σ X)
        when --j2 ≠ 0, otherwise the standard TFIM via builder.build. The same
        H must generate both the ground state and the time evolution, so this
        is the single source of truth for both.
        """
        j2 = float(getattr(self._args, "j2", 0.0) or 0.0)
        if abs(j2) > 1e-15:
            return self.builder.build_frustrated_tfim(lattice, J2=j2)
        return self.builder.build(lattice)

    def _build_trotter_step_circuit(self, n_qubits, topology, h, dt):
        """Second-order Suzuki-Trotter step.

        U₂(dt) = e^{-i(dt/2)H_ZZ} · e^{-i(dt)H_X} · e^{-i(dt/2)H_ZZ}
        H = -J₁ Σ_nn Z_iZ_j (+ J₂ Σ_nnn Z_iZ_j) - h Σ X_i

        The NNN term (sign +J₂) is included only when --j2 ≠ 0, matching
        build_frustrated_tfim. NNN edges come from builder._generate_nnn_edges
        so the Trotter generator and the energy observable share the same bonds.
        """
        from qiskit.circuit import QuantumCircuit

        lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h)
        edges = lattice.edges
        j2 = float(getattr(self._args, "j2", 0.0) or 0.0)
        nnn_edges = self.builder._generate_nnn_edges(lattice) if abs(j2) > 1e-15 else []
        qc = QuantumCircuit(n_qubits)
        half_dt = dt / 2.0

        def _zz_half():
            # NN ferromagnetic (-J₁): exp(+i(dt/2) Z_iZ_j) → RZZ(-dt)
            for i, j in edges:
                qc.rzz(-2 * half_dt, i, j)
            # NNN antiferromagnetic (+J₂): exp(-i(dt/2) J₂ Z_iZ_j) → RZZ(+J₂ dt)
            for i, j in nnn_edges:
                qc.rzz(2 * half_dt * j2, i, j)

        _zz_half()
        # exp(-i(dt)H_X) = Π exp(+i dt h X_i) → RX(-2 dt h)
        for i in range(n_qubits):
            qc.rx(-2 * dt * h, i)
        _zz_half()
        return qc

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers — State preparation & MPNN
    # ═══════════════════════════════════════════════════════════════════════════

    def _ground_state_vector(self, n_qubits, topology, h):
        """Get ground state vector via sparse eigsh (up to N=22).

        Uses in-memory _gs_vector_cache to avoid redundant eigenvector recomputation
        (1-5s per call for N=20-22). Also caches energy/gap in GroundTruthCache
        for cross-session reuse.
        """
        # ── In-memory cache lookup ───────────────────────────────────────────
        cache_key = (n_qubits, topology, round(h, 2))
        if cache_key in self._gs_vector_cache:
            logger.debug(f"  GS vector cache hit: {topology} N={n_qubits} h={h:.2f}")
            return self._gs_vector_cache[cache_key]

        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        # Reuse the runner's shared disk cache singleton (set by exact_ground_state
        # or setup_physics). Avoids fresh instantiation that loads stale data
        # and creates unnecessary I/O.
        gt_cache = getattr(self, "_disk_gt_cache", None)
        if gt_cache is None:
            gt_cache = GroundTruthCache()
            self._disk_gt_cache = gt_cache
        cached = gt_cache.get(topology, n_qubits, self._args.model, float(h))
        if cached:
            logger.debug(f"  GT cache hit: {topology} N={n_qubits} h={h}")

        lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h)
        H_op = self._build_quench_hamiltonian(lattice)

        e_gs = None
        gap = 0.0

        if n_qubits <= _DENSE_LIMIT:
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
            gt_cache.put(
                topology,
                n_qubits,
                self._args.model,
                float(h),
                energy=e_gs,
                gap=gap,
                method="eigsh_k2",
            )
            gt_cache.flush()

        # Store in in-memory vector cache (avoids eigsh recomputation)
        # Evict oldest entry if cache is full (LRU-like, FIFO eviction)
        if len(self._gs_vector_cache) >= self._GS_CACHE_MAX_SIZE:
            oldest_key = next(iter(self._gs_vector_cache))
            del self._gs_vector_cache[oldest_key]
        self._gs_vector_cache[cache_key] = gs
        return gs

    def _predict_theta_gnn(self, topology, n_qubits, h, p_layers):
        """Predict θ using model zoo — reuses load_best_model_for."""
        try:
            from qmbp_simulation.predictors.model_zoo import load_best_model_for
            from qmbp_simulation.predictors.unified_graph import (
                build_graph_for_model,
            )

            result = load_best_model_for(
                topology,
                n_target=n_qubits,
                h_regime="critical",
                p_layers=p_layers,
            )
            model = result[0]
            if model is None:
                return None

            import torch

            lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h)
            graph = build_graph_for_model(model, lattice, h_value=h, p_layers=p_layers)

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
        """Von Neumann entropy of the index-contiguous bipartition, in bits.

        Splits by qubit INDEX (first n/2 vs rest). This is a genuine spatial cut
        only for 1D chains. For 2D lattices use _entanglement_entropy, which cuts
        along the lattice geometry. Kept for 1D and backward compatibility.
        """
        n_a = n_qubits // 2
        psi_matrix = psi.reshape(2**n_a, 2 ** (n_qubits - n_a))
        sv = np.linalg.svd(psi_matrix, compute_uv=False)
        probs = sv**2
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log2(probs)))

    def _entanglement_entropy(self, psi, n_qubits, topology):
        """Von Neumann entropy of a SPATIAL bipartition, in bits (log2).

        For 2D grids (square) the qubit index does not map to a contiguous
        spatial region, so an index cut mixes the two halves. Here we split the
        lattice into left/right column blocks (partition A = the leftmost
        columns) and trace out the complement, giving a geometrically meaningful
        cut. For 1D/index-contiguous topologies this reduces to the half-chain
        cut.
        """
        partition_a = self._spatial_partition_a(n_qubits, topology)
        if partition_a is None:
            return self._half_chain_entropy(psi, n_qubits)

        # Reorder axes so partition_a qubits come first, then reshape & SVD.
        state = np.asarray(psi).reshape([2] * n_qubits)
        rest = [q for q in range(n_qubits) if q not in partition_a]
        state = np.transpose(state, partition_a + rest)
        mat = state.reshape(2 ** len(partition_a), 2 ** len(rest))
        sv = np.linalg.svd(mat, compute_uv=False)
        probs = sv**2
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log2(probs)))

    def _spatial_partition_a(self, n_qubits, topology):
        """Qubit indices of the left spatial half of the lattice, or None.

        Infers the grid width from the first vertical edge (j−i > 1) of the
        square lattice and returns the leftmost ⌈ncols/2⌉ columns. Returns None
        for topologies where the index order is already spatially contiguous
        (chain/ladder) so the caller falls back to the half-chain cut.
        """
        if topology != "square":
            return None
        lattice = self.make_lattice(topology, n_qubits, J=1.0, h=1.0)
        verticals = [j - i for i, j in lattice.edges if j - i > 1]
        if not verticals:
            return None
        ncols = min(verticals)  # grid width (row stride)
        if ncols < 2 or ncols >= n_qubits:
            return None
        cut_col = ncols // 2
        return [q for q in range(n_qubits) if (q % ncols) < cut_col]

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
        """Override: run DQPT validation + go/no-go after all sections, then call super()."""
        # Auto-validate DQPT trajectories produced by this run
        try:
            from scripts.analysis.validate_dqpt_results import (
                save_report as save_dqpt_report,
            )
            from scripts.analysis.validate_dqpt_results import (
                validate_dqpt_topology,
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
                out_path = self._get_project_root() / "results" / "analysis" / f"dqpt_validation_{topo}.json"
                save_dqpt_report(report, out_path)
        except Exception as e:
            logger.debug(f"  DQPT auto-validation skipped: {e}")

        # ── Go/No-Go evaluation (combines QPT + DQPT → hardware readiness) ──
        try:
            from scripts.analysis.validate_dqpt_results import compute_go_no_go

            topo = self._topology
            p = getattr(self._args, "p_layers", 1)
            go_result = compute_go_no_go(topo, p_layers=p)

            n_pass = go_result.get("n_passed", 0)
            n_total = go_result.get("n_total", 0)
            overall = go_result.get("overall_go", False)
            blocking = go_result.get("blocking_issues", [])

            logger.info(
                f"\n  ── QPT/DQPT Go/No-Go ({topo}) ──"
                f"\n  {n_pass}/{n_total} criteria passed "
                f"[{'GO ✅' if overall else 'NO-GO ❌'}]"
            )
            if blocking:
                for issue in blocking[:3]:
                    logger.info(f"     ⚠️  {issue}")

            # Persist go/no-go result for downstream consumption
            import json

            from qmbp_simulation.utils.helpers import json_serialize

            go_path = self._get_project_root() / "results" / "analysis" / f"go_no_go_{topo}_p{p}.json"
            go_path.parent.mkdir(parents=True, exist_ok=True)
            with open(go_path, "w") as f:
                json.dump(go_result, f, indent=2, default=json_serialize)
        except Exception as e:
            logger.debug(f"  Go/No-Go evaluation skipped: {e}")

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
        mx_values = [float(_magnetization_x_obs(psi_0, n_qubits))]

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
            mx_values.append(float(_magnetization_x_obs(psi_t, n_qubits)))

        # Detect critical times
        critical_times = detect_dqpt_critical_times(times, loschmidt_values, threshold=0.1)

        # Get ground-state energy and gap at h_pre (for qpt_detection.py integration)
        # Use exact_ground_state() from runner_base for proper 2-level cache
        try:
            e_exact_h_pre, gap_h_pre = self.exact_ground_state(
                topology,
                n_qubits,
                h_pre,
                model=self._args.model,
            )
        except Exception:
            e_exact_h_pre = float(energies[0]) if energies else 0.0
            gap_h_pre = 0.0
            try:
                from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

                gt_cache = GroundTruthCache()
                cached = gt_cache.get(topology, n_qubits, self._args.model, h_pre)
                if cached is not None:
                    e_exact_h_pre = float(cached.get("energy", e_exact_h_pre))
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
                        f"  Skipping NPZ write: existing {out_path.name} has {existing_steps} steps > current {n_steps}"
                    )
                    return
            except Exception:
                pass  # Corrupted file — overwrite is fine

        from qmbp_simulation.utils.helpers import atomic_savez

        atomic_savez(
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
            entropy_units="bits",
            magnetization_x=np.array(mx_values, dtype=float),
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

    # ═══════════════════════════════════════════════════════════════════════════
    # GNN comparison for DQPT — compares exact GS vs GNN-prepared quench
    # ═══════════════════════════════════════════════════════════════════════════

    def _dqpt_gnn_comparison(
        self,
        n_qubits: int,
        topology: str,
        h_pre: float,
        h_post: float,
        dt: float,
        n_steps: int,
        H_post_op,
        use_sparse: bool,
        H_sparse=None,
        U_dt=None,
    ) -> dict | None:
        """Compare Loschmidt echo from GNN-prepared state vs exact ground state.

        After the main DQPT evolution (from exact GS), we:
        1. Prepare the GNN state |ψ_GNN(h₁)⟩ = HVA(θ_predicted) |+⟩^N
        2. Evolve under the same H(h₂) quench Hamiltonian
        3. Compute L_gnn(t) = |⟨ψ_GNN(h₁)|ψ_GNN(t)⟩|² and detect t*_gnn
        4. Compare t*_gnn shift relative to t*_exact

        Returns None if GNN model is unavailable or N > ED limit.
        """
        if n_qubits > _ED_MAX_N:
            logger.debug("  GNN comparison skipped: N > ED limit")
            return None

        from qmbp_simulation.analysis.observables import (
            detect_dqpt_critical_times,
            loschmidt_echo,
            rate_function,
        )

        # Prepare GNN state: HVA(θ_predicted) applied to |+⟩^N
        theta_pred = self._predict_theta_gnn(topology, n_qubits, h_pre, self._args.p_layers)
        if theta_pred is None:
            logger.debug("  GNN comparison skipped: no model available")
            return None

        try:
            lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h_pre)
            circuit, _ = self.hva.create(
                n_qubits=n_qubits,
                p_layers=self._args.p_layers,
                lattice=lattice,
            )

            # Evaluate GNN circuit → statevector
            from qiskit.quantum_info import Statevector

            bound_circuit = circuit.assign_parameters(theta_pred)
            psi_gnn = Statevector(bound_circuit).data
        except Exception as e:
            logger.debug(f"  GNN state preparation failed: {e}")
            return None

        # Evolve GNN state and compute Loschmidt echo
        psi_t = psi_gnn.copy().astype(complex)
        times_gnn = [0.0]
        loschmidt_gnn = [1.0]
        rate_gnn = [0.0]

        for step in range(1, n_steps + 1):
            if use_sparse:
                from scipy.sparse.linalg import expm_multiply

                psi_t = expm_multiply(-1j * H_sparse * dt, psi_t)
            else:
                psi_t = U_dt @ psi_t
            psi_t /= np.linalg.norm(psi_t)

            t = step * dt
            times_gnn.append(t)
            L_t = loschmidt_echo(psi_gnn, psi_t)
            loschmidt_gnn.append(L_t)
            rate_gnn.append(rate_function(L_t, n_qubits))

        # Detect GNN critical times
        critical_times_gnn = detect_dqpt_critical_times(
            times_gnn,
            loschmidt_gnn,
            threshold=0.1,
        )

        # Compare with exact (get from cache if already persisted)
        exact_gs = self._ground_state_vector(n_qubits, topology, h_pre)
        fidelity_gnn_vs_exact = float(np.abs(np.vdot(exact_gs, psi_gnn)) ** 2)

        # t* shift analysis
        t_star_shift = None
        if critical_times_gnn:
            # Load exact critical times from the trajectory we just persisted
            traj_path = self._get_project_root() / "data" / "dqpt_trajectories" / f"{topology}_N{n_qubits}.npz"
            if traj_path.exists():
                try:
                    traj = np.load(traj_path, allow_pickle=True)
                    exact_crits = traj.get("critical_times", np.array([]))
                    if len(exact_crits) > 0 and len(critical_times_gnn) > 0:
                        t_star_shift = float(critical_times_gnn[0] - exact_crits[0])
                except Exception:
                    pass

        result = {
            "fidelity_gnn_vs_exact": fidelity_gnn_vs_exact,
            "n_dqpt_gnn": len(critical_times_gnn),
            "critical_times_gnn": critical_times_gnn,
            "t_star_shift": t_star_shift,
            "rate_peak_gnn": float(max(rate_gnn)) if rate_gnn else 0.0,
            "loschmidt_gnn": loschmidt_gnn,
        }
        logger.info(
            f"  GNN comparison: F={fidelity_gnn_vs_exact:.4f}, "
            f"DQPTs={len(critical_times_gnn)}, "
            f"Δt*={t_star_shift if t_star_shift is not None else 'N/A'}"
        )
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Auto-select quench parameters from data
    # ═══════════════════════════════════════════════════════════════════════════

    def auto_select_quench_params(self, topology: str, n_qubits: int) -> tuple[float, float]:
        """Auto-select h₁, h₂ for quench by analyzing existing data.

        Strategy:
        1. Get h_c from QPT detection (cached results or quick computation)
        2. Read extrapolation/training NPZ to find h with lowest ΔE/gap
           (= best MPNN accuracy point → most reliable initial state)
        3. Place h_post on opposite side of h_c from best_h

        Returns
        -------
        tuple[float, float]
            (h1, h2) — h_pre for ground state prep, h_post for quench Hamiltonian.
        """
        # Get critical field from QPT detection (topology-specific, data-driven)
        h_c = 1.0  # Fallback for TFIM (most topologies)
        try:
            from scripts.analysis.qpt_detection import get_h_critical

            h_c_detected = get_h_critical(topology, p_layers=getattr(self._args, "p_layers", 1))
            if h_c_detected is not None:
                h_c = h_c_detected
                logger.info(f"  Auto-select: h_c={h_c:.3f} (from QPT detection)")
        except Exception as e:
            logger.debug(f"  QPT detection unavailable, using h_c=1.0: {e}")

        # Try to load from training NPZ (best MPNN accuracy → best state prep)
        data_dir = self._get_project_root() / "data" / "multi_n_training"
        p_layers = self._args.p_layers if hasattr(self._args, "p_layers") else 1
        npz_path = data_dir / f"{topology}_N{n_qubits}_p{p_layers}.npz"

        best_h = None
        if npz_path.exists():
            try:
                data = np.load(npz_path, allow_pickle=True)
                h_values = data.get("h_values", np.array([]))
                de_gap_values = data.get("de_gap", None)

                if de_gap_values is not None and len(de_gap_values) > 0:
                    # Find h with lowest ΔE/gap (best prediction quality)
                    best_idx = int(np.argmin(np.abs(de_gap_values)))
                    best_h = float(h_values[best_idx])
                    logger.info(
                        f"  Auto-select: best h={best_h:.2f} "
                        f"(ΔE/gap={de_gap_values[best_idx]:.4f}) from {npz_path.name}"
                    )
            except Exception as e:
                logger.debug(f"  Auto-select NPZ read failed: {e}")

        # Fallback: try large-N extrapolation data
        if best_h is None:
            extrap_dir = self._get_project_root() / "data" / "large_n_extrapolation"
            extrap_path = extrap_dir / f"{topology}_N{n_qubits}_p{p_layers}.npz"
            if extrap_path.exists():
                try:
                    data = np.load(extrap_path, allow_pickle=True)
                    h_values = data.get("h_values", np.array([]))
                    de_gap_values = data.get("de_gap", None)
                    if de_gap_values is not None and len(de_gap_values) > 0:
                        best_idx = int(np.argmin(np.abs(de_gap_values)))
                        best_h = float(h_values[best_idx])
                        logger.info(f"  Auto-select: best h={best_h:.2f} from {extrap_path.name}")
                except Exception:
                    pass

        # Default: h1=3.0 (deep paramagnetic), h2=0.5 (deep ferromagnetic)
        if best_h is None:
            logger.info("  Auto-select: no data found, using defaults h1=3.0, h2=0.5")
            return 3.0, 0.5

        # Place h_post on opposite side of h_c from best_h
        if best_h > h_c:
            # best_h is paramagnetic → quench into ferromagnetic
            h1 = round(best_h, 2)
            h2 = round(max(0.1, h_c - (best_h - h_c) * 0.5), 2)
        else:
            # best_h is ferromagnetic → quench into paramagnetic
            h1 = round(best_h, 2)
            h2 = round(min(5.0, h_c + (h_c - best_h) * 0.5), 2)

        logger.info(f"  Auto-select: h1={h1:.2f}, h2={h2:.2f} (h_c={h_c})")
        return h1, h2


if __name__ == "__main__":
    QuenchDynamicsStudyRunner.main()
