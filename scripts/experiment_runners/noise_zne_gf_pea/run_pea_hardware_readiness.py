#!/usr/bin/env python3
"""PEA-ZNE Hardware Readiness — Scalability & Realism Assessment.

Tests PEA-ZNE under conditions that match the actual IBM Torino deployment:
  - heavy_hex topology (N=10, p=1) — the real target
  - Full FakeTorino noise at factor=1 (realistic baseline)
  - Measures absolute ΔE/gap (not just relative gain)
  - Compares PEA extrapolation against exact noiseless energy
  - Tests if PEA can achieve ΔE/gap < 5% (the hardware success criterion)
  - Applies affine correction post-ZNE (zero cost, always beneficial)

Key question: Can PEA-ZNE bring the noisy energy close enough to pass
the hardware deployment criterion (ΔE/gap < 5%) on heavy_hex?

Sections:
  1. Noiseless baseline — VQE θ_opt on heavy_hex N=10 p=1
  2. Full-noise baseline — FakeTorino native noise (factor=1)
  3. GF-ZNE mitigation — gate folding [1,3,5] + affine correction
  4. PEA-ZNE mitigation — learned noise amplification [1,3,5] + affine correction
  5. Hardware readiness verdict — ΔE/gap < 5% check

Usage:
    python scripts/experiment_runners/noise_zne_gf_pea/run_pea_hardware_readiness.py
    python scripts/experiment_runners/noise_zne_gf_pea/run_pea_hardware_readiness.py --topology chain_1d --n-qubits 6
    python scripts/experiment_runners/noise_zne_gf_pea/run_pea_hardware_readiness.py --dry-run
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import numpy as np

from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)
from qmbp_simulation.models.constants import (
    DE_GAP_THRESHOLD,
    ZNE_CES_PERTURBATIVE_THRESHOLD,
    ZNE_DEFAULT_N_CANDIDATE_LAYOUTS,
    ZNE_DEFAULT_NOISE_FACTORS,
    ZNE_DEFAULT_SHOTS,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants — hardware deployment spec values (not redeclaring framework defaults)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TOPOLOGY = "heavy_hex"
DEFAULT_N_QUBITS = 10
DEFAULT_P_LAYERS = 1

# h-values from deployment spec (in-regime for heavy_hex p=1)
H_TEST_HARDWARE = [4.0, 3.25, 3.0]

VQE_MAXITER = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class PEAHardwareReadinessRunner(ValidationRunner):
    """PEA-ZNE hardware readiness: can it achieve ΔE/gap < 5% on heavy_hex?"""

    runner_id = "pea_hardware_readiness"
    experiment_id = "noisy/tfim/heavy_hex"
    description = "PEA-ZNE Hardware Readiness (heavy_hex N=10 p=1)"
    hypothesis = (
        "PEA-ZNE on heavy_hex N=10 p=1 achieves ΔE/gap closer to noiseless "
        "than GF-ZNE, with R²>0.9 and consistent gain across h-values."
    )

    @classmethod
    def _add_custom_args(cls, parser) -> None:
        cls._add_standard_physics_args(
            parser,
            n_qubits=DEFAULT_N_QUBITS,
            p_layers=DEFAULT_P_LAYERS,
            topology=DEFAULT_TOPOLOGY,
            model="tfim",
            h_min=3.0,
            h_max=4.0,
            h_points=4,
            seeds=[42],
            maxiter=VQE_MAXITER,
            n_restarts=1,
        )
        parser.add_argument(
            "--h-values",
            type=float,
            nargs="+",
            default=None,
            help="Explicit h-values (overrides --h-min/--h-max/--h-points)",
        )
        parser.add_argument(
            "--shots",
            type=int,
            default=ZNE_DEFAULT_SHOTS,
            help=f"Shots per estimation (default: {ZNE_DEFAULT_SHOTS})",
        )

    def run_preflight(self) -> bool:
        """Extended preflight: validate dependencies before setup()."""
        if not super().run_preflight():
            return False
        try:
            from qiskit_ibm_runtime.fake_provider import FakeTorino  # noqa: F401
        except ImportError:
            logger.error(
                "  Preflight ERROR: qiskit-ibm-runtime is required for FakeTorino noise model. "
                "Install with: pip install qiskit-ibm-runtime"
            )
            return False
        return True

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="Noiseless Baseline",
                fn=self._section_noiseless,
                hypothesis="VQE converges with ΔE/gap < 20% at h≥3.0 (p=1 limit)",
            ),
            Section(
                id=2,
                name="Full-Noise Baseline",
                fn=self._section_noisy,
                hypothesis="FakeTorino noise increases ΔE/gap to >50% (noise-dominated)",
            ),
            Section(
                id=3,
                name="Gate-Folding ZNE",
                fn=self._section_gf_zne,
                hypothesis="GF-ZNE reduces ΔE/gap relative to noisy with R²>0.5",
            ),
            Section(
                id=4,
                name="PEA-ZNE",
                fn=self._section_pea_zne,
                hypothesis="PEA-ZNE gain > GF-ZNE gain with R²>0.9",
            ),
            Section(
                id=5,
                name="Hardware Readiness Verdict",
                fn=self._section_verdict,
                hypothesis="PEA-ZNE extrapolation approaches noiseless ΔE/gap",
            ),
        ]

    def build_config(self) -> dict:
        topo = (
            self._args.topology[0] if isinstance(self._args.topology, list) else self._args.topology
        )
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "category": "ZNE",
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "n_qubits": self._args.n_qubits,
                "p_layers": self._args.p_layers,
                "topology": topo,
                "topologies": [topo],
                "model": "tfim",
            },
            "zne": {
                "noise_factors": list(ZNE_DEFAULT_NOISE_FACTORS),
                "shots": getattr(self._args, "shots", ZNE_DEFAULT_SHOTS),
                "n_candidate_layouts": ZNE_DEFAULT_N_CANDIDATE_LAYOUTS,
                "method": "PEA + GF comparison",
                "affine_correction": True,
            },
            "seeds": self._args.seeds if hasattr(self._args, "seeds") else [],
        }

    def setup(self) -> None:
        """Initialize physics + noisy estimation infrastructure."""
        self.setup_physics()

        self.topology = (
            self._args.topology[0] if isinstance(self._args.topology, list) else self._args.topology
        )
        self.n_qubits = self._args.n_qubits
        self.p_layers = self._args.p_layers
        self._vqe_restarts = self.compute_vqe_restarts(self.p_layers, self.n_qubits)

        # h-values: explicit list > h-min/h-max/h-points > auto from p/topo
        if self._args.h_values is not None:
            self._h_test = sorted(self._args.h_values, reverse=True)
        elif self._args.h_min != 3.0 or self._args.h_max != 4.0 or self._args.h_points != 4:
            self._h_test = list(
                np.linspace(self._args.h_max, self._args.h_min, self._args.h_points)
            )
        else:
            self._h_test = self.default_h_test_values(self.p_layers, self.topology)

        # Noisy estimation setup (FakeTorino, config, candidates, utility functions)
        self._shots = getattr(self._args, "shots", ZNE_DEFAULT_SHOTS)
        self.setup_noisy_estimation(self.n_qubits, shots=self._shots, seed_simulator=42)

        logger.info(
            f"[setup] {self.topology} N={self.n_qubits} p={self.p_layers}, "
            f"{len(self.candidates)} candidates, restarts={self._vqe_restarts}, "
            f"h_test={self._h_test}, shots={self._shots}"
        )
        if self.n_qubits >= 16:
            logger.info(
                "  ⚠️  N≥16: noisy sections will be SLOW (FakeTorino transpilation "
                "~2-5 min per h-point). This is expected, not a hang."
            )

        # Shared mutable state across sections
        self._data: dict[str, list[dict]] = {}
        self._transpiled_cache: dict[float, tuple] = {}
        self._circuit = None

    def restore_section_state(
        self, resumed_data: dict[str, Any], resumed_sections: set[int]
    ) -> None:
        """Restore internal state from a resumed run for downstream sections."""
        results = resumed_data.get("results", {})

        if 1 in resumed_sections:
            s1_data = results.get("section_1", {}).get("data", {})
            noiseless_results = s1_data.get("results", [])
            if noiseless_results:
                self._data["noiseless"] = noiseless_results
                # Rebuild circuit from model spec
                from qmbp_simulation.models.model_registry import get_model_spec

                spec = get_model_spec("tfim")
                lattice_ref = self.make_lattice(
                    self.topology, self.n_qubits, J=1.0, h=max(self._h_test)
                )
                circuit, _ = spec.create_circuit(
                    self.n_qubits, self.p_layers, lattice_ref, **spec.circuit_kwargs
                )
                self._circuit = circuit
                logger.info(
                    "  ♻️  Restored Section 1 state: %d noiseless results", len(noiseless_results)
                )

        if 2 in resumed_sections:
            s2_data = results.get("section_2", {}).get("data", {})
            noisy_results = s2_data.get("results", [])
            if noisy_results:
                self._data["noisy"] = noisy_results
                # Note: _transpiled_cache cannot be serialized/restored
                # Sections 3-4 will need to re-transpile if section 2 is skipped
                logger.info("  ♻️  Restored Section 2 state: %d noisy results", len(noisy_results))

    def _section_noiseless(self) -> dict:
        """Section 1: VQE noiseless baseline using base class vqe_adaptive_sweep."""
        from qmbp_simulation.models.model_registry import get_model_spec

        spec = get_model_spec("tfim")
        lattice_ref = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=max(self._h_test))
        circuit, _ = spec.create_circuit(
            self.n_qubits, self.p_layers, lattice_ref, **spec.circuit_kwargs
        )
        self._circuit = circuit
        n_params = circuit.num_parameters

        # Use base class vqe_adaptive_sweep (includes bidirectional + adaptive restarts)
        use_ascending = self.should_use_bidirectional(self.n_qubits)

        vqe_results = self.vqe_adaptive_sweep(
            topology=self.topology,
            n_qubits=self.n_qubits,
            h_values=self._h_test,
            seed=42,
            p_layers=self.p_layers,
            n_restarts=self._vqe_restarts,
            maxiter=VQE_MAXITER,
            model="tfim",
            ascending_pass=use_ascending,
            compute_fidelity=False,
        )

        # Build results with exact energies and variational principle check
        results = []
        for vqe_pt in vqe_results:
            h = vqe_pt["h"]
            e_exact, gap = self.exact_ground_state(self.topology, self.n_qubits, h)
            theta_opt = np.array(vqe_pt["theta_opt"])

            lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
            H = self.builder.build(lattice)
            e_vqe = float(self.noiseless.evaluate(circuit, H, theta_opt))
            de_gap = abs(e_vqe - e_exact) / max(gap, 1e-10)

            # Variational principle check
            if e_vqe < e_exact - 1e-6:
                logger.warning(
                    f"  h={h:.2f}: VARIATIONAL PRINCIPLE VIOLATED! "
                    f"E_vqe={e_vqe:.6f} < E_exact={e_exact:.6f}"
                )

            results.append(
                {
                    "h": h,
                    "e_exact": e_exact,
                    "gap": gap,
                    "e_noiseless": e_vqe,
                    "de_gap_noiseless": de_gap,
                    "theta_opt": theta_opt.tolist(),
                }
            )
            logger.info(f"  h={h:.2f}: E_exact={e_exact:.4f}, ΔE/gap={de_gap:.4f}")

        # Register circuit artifact
        self.artifacts.register(
            "circuit",
            circuit,
            format="qpy",
            metadata={
                "n_qubits": self.n_qubits,
                "p_layers": self.p_layers,
                "topology": self.topology,
                "n_params": n_params,
            },
        )

        self._data["noiseless"] = results
        return {"results": results, "n_params": n_params}

    def _section_noisy(self) -> dict:
        """Section 2: Full FakeTorino noise — unmitigated baseline with checkpointing."""
        if "noiseless" not in self._data:
            raise RuntimeError("Run Section 1 first (no noiseless data)")
        if self._circuit is None:
            raise RuntimeError("Run Section 1 first (no circuit cached)")

        # Check for resumed checkpoint
        cp = self.load_checkpoint("noisy_baseline")
        if cp:
            self._transpiled_cache = {float(k): v for k, v in cp.get("transpiled_meta", {}).items()}
            completed_results = cp.get("results", [])
            done_h = {r["h"] for r in completed_results}
        else:
            self._transpiled_cache = {}
            completed_results = []
            done_h = set()

        def _estimate_noisy(h: float) -> dict | None:
            if h in done_h:
                return None  # Already done via checkpoint

            pt = next((p for p in self._data["noiseless"] if p["h"] == h), None)
            if pt is None:
                return None
            theta_opt = np.array(pt["theta_opt"])
            e_exact, gap = pt["e_exact"], pt["gap"]

            logger.info(f"  h={h:.2f}: transpiling + layout selection...")
            lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
            H = self.builder.build(lattice)
            bound = self._circuit.assign_parameters(theta_opt)

            layout_sel = self.select_low_ces(
                bound,
                self.fake_backend,
                self.candidates,
                n_select=1,
                optimization_level=2,
                max_ces=0.5,
            )
            transpiled = layout_sel.transpiled_circuits[0]
            ces = float(layout_sel.ces_values[0])
            H_mapped = H.apply_layout(transpiled.layout)
            self._transpiled_cache[h] = (transpiled, H_mapped, ces)

            # Warn if CES exceeds perturbative threshold
            if ces > ZNE_CES_PERTURBATIVE_THRESHOLD:
                logger.warning(
                    f"  h={h:.2f}: CES={ces:.4f} > {ZNE_CES_PERTURBATIVE_THRESHOLD} "
                    f"— outside perturbative regime for GF-ZNE"
                )

            e_noisy = self.noisy_estimate(
                transpiled, H_mapped, self.fake_backend, self.noisy_config
            )
            if not np.isfinite(e_noisy):
                return None

            de_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)
            logger.info(
                f"  h={h:.2f}: E_noisy={e_noisy:.4f}, ΔE/gap={de_noisy:.4f}, "
                f"CES={ces:.4f}, depth={transpiled.depth()}"
            )
            return {
                "h": h,
                "e_noisy": float(e_noisy),
                "de_gap_noisy": float(de_noisy),
                "layout_ces": ces,
                "circuit_depth": transpiled.depth(),
            }

        h_values = [pt["h"] for pt in self._data["noiseless"]]
        new_results = self.safe_per_h_loop(
            [h for h in h_values if h not in done_h], _estimate_noisy, "noisy_estimate"
        )

        # Merge checkpoint results with new results
        all_results = completed_results + new_results

        # Save checkpoint after completing this section
        if new_results:
            self.save_checkpoint(
                "noisy_baseline",
                {
                    "results": all_results,
                    "transpiled_meta": {
                        str(h): {"ces": t[2], "depth": t[0].depth()}
                        for h, t in self._transpiled_cache.items()
                        if isinstance(t, tuple)
                    },
                },
            )

        if not all_results:
            return {"pass": False, "error": "All noisy estimations failed"}

        self._data["noisy"] = all_results
        return {"results": all_results}

    def _section_gf_zne(self) -> dict:
        """Section 3: Gate-Folding ZNE + affine correction — comparison baseline."""
        if "noiseless" not in self._data or not self._transpiled_cache:
            raise RuntimeError("Run Sections 1 and 2 first")

        def _run_gf(h: float) -> dict | None:
            pt = next((p for p in self._data["noiseless"] if p["h"] == h), None)
            if pt is None or h not in self._transpiled_cache:
                return None
            e_exact, gap = pt["e_exact"], pt["gap"]
            transpiled, H_mapped, _ = self._transpiled_cache[h]

            t0 = time.time()
            gf = self.run_gf_zne(
                transpiled,
                H_mapped,
                self.fake_backend,
                self.noisy_config,
                noise_factors=ZNE_DEFAULT_NOISE_FACTORS,
                extrapolator="linear",
                seed_offset=500,
            )
            elapsed = time.time() - t0

            if not np.isfinite(gf.extrapolated_value):
                return None

            # Apply affine correction (zero cost, always beneficial)
            corrected = self.affine_correct_energy(
                gf.extrapolated_value,
                e_exact,
                n_qubits=self.n_qubits,
                h_value=h,
            )

            de_gf_raw = abs(gf.extrapolated_value - e_exact) / max(gap, 1e-10)
            de_gf_corrected = abs(corrected.corrected_energy - e_exact) / max(gap, 1e-10)

            logger.info(
                f"  h={h:.2f}: E_gf={gf.extrapolated_value:.4f} → "
                f"E_corrected={corrected.corrected_energy:.4f}, "
                f"ΔE/gap={de_gf_raw:.4f}→{de_gf_corrected:.4f}, "
                f"R²={gf.r_squared:.4f}, {elapsed:.1f}s"
            )
            return {
                "h": h,
                "e_gf_zne_raw": float(gf.extrapolated_value),
                "e_gf_zne": float(corrected.corrected_energy),
                "de_gap_gf_raw": float(de_gf_raw),
                "de_gap_gf": float(de_gf_corrected),
                "gf_r2": float(gf.r_squared),
                "gf_slope": float(gf.slope),
                "elapsed_s": round(elapsed, 2),
                "measured": gf.measured_values,
                "affine_corrected": corrected.correction_applied,
            }

        h_values = list(self._transpiled_cache.keys())
        results = self.safe_per_h_loop(h_values, _run_gf, "GF-ZNE")

        if not results:
            return {"pass": False, "error": "All GF-ZNE extrapolations failed"}
        self._data["gf_zne"] = results
        return {"results": results}

    def _section_pea_zne(self) -> dict:
        """Section 4: PEA-ZNE + affine correction — probabilistic error amplification."""
        if "noiseless" not in self._data or not self._transpiled_cache:
            raise RuntimeError("Run Sections 1 and 2 first")

        def _run_pea(h: float) -> dict | None:
            pt = next((p for p in self._data["noiseless"] if p["h"] == h), None)
            if pt is None or h not in self._transpiled_cache:
                return None
            e_exact, gap = pt["e_exact"], pt["gap"]
            transpiled, H_mapped, _ = self._transpiled_cache[h]

            t0 = time.time()
            pea = self.run_pea_zne(
                transpiled,
                H_mapped,
                self.fake_backend,
                self.noisy_config,
                noise_factors=ZNE_DEFAULT_NOISE_FACTORS,
                extrapolator="linear",
                seed_offset=2000,
            )
            elapsed = time.time() - t0

            if not np.isfinite(pea.extrapolated_value):
                return None

            # Apply affine correction (zero cost, always beneficial)
            corrected = self.affine_correct_energy(
                pea.extrapolated_value,
                e_exact,
                n_qubits=self.n_qubits,
                h_value=h,
            )

            de_pea_raw = abs(pea.extrapolated_value - e_exact) / max(gap, 1e-10)
            de_pea_corrected = abs(corrected.corrected_energy - e_exact) / max(gap, 1e-10)

            logger.info(
                f"  h={h:.2f}: E_pea={pea.extrapolated_value:.4f} → "
                f"E_corrected={corrected.corrected_energy:.4f}, "
                f"ΔE/gap={de_pea_raw:.4f}→{de_pea_corrected:.4f}, "
                f"R²={pea.r_squared:.4f}, {elapsed:.1f}s"
            )
            return {
                "h": h,
                "e_pea_zne_raw": float(pea.extrapolated_value),
                "e_pea_zne": float(corrected.corrected_energy),
                "de_gap_pea_raw": float(de_pea_raw),
                "de_gap_pea": float(de_pea_corrected),
                "pea_r2": float(pea.r_squared),
                "pea_slope": float(pea.slope),
                "elapsed_s": round(elapsed, 2),
                "measured": pea.measured_values,
                "learned_rates": pea.learned_error_rates,
                "affine_corrected": corrected.correction_applied,
            }

        h_values = list(self._transpiled_cache.keys())
        results = self.safe_per_h_loop(h_values, _run_pea, "PEA-ZNE")

        if not results:
            return {"pass": False, "error": "All PEA-ZNE extrapolations failed"}
        self._data["pea_zne"] = results
        return {"results": results}

    def _section_verdict(self) -> dict:
        """Section 5: Hardware readiness — comprehensive comparison."""
        nl = self._data.get("noiseless", [])
        ny = self._data.get("noisy", [])
        gf = self._data.get("gf_zne", [])
        pea = self._data.get("pea_zne", [])

        if not nl:
            return {"pass": False, "error": "No noiseless data"}
        if not ny:
            return {"pass": False, "error": "No noisy data"}
        if not gf:
            return {"pass": False, "error": "No GF-ZNE data (all failed)"}
        if not pea:
            return {"pass": False, "error": "No PEA-ZNE data (all failed)"}

        # Build comparison using h-value matching (handles partial failures)
        nl_by_h = {pt["h"]: pt for pt in nl}
        ny_by_h = {pt["h"]: pt for pt in ny}
        gf_by_h = {pt["h"]: pt for pt in gf}
        pea_by_h = {pt["h"]: pt for pt in pea}

        # Only compare h-points present in ALL four datasets
        common_h = sorted(
            set(nl_by_h) & set(ny_by_h) & set(gf_by_h) & set(pea_by_h),
            reverse=True,
        )

        if not common_h:
            return {"pass": False, "error": "No h-points with data in all 4 modes"}

        comparison = []
        for h in common_h:
            row = {
                "h": h,
                "e_exact": nl_by_h[h]["e_exact"],
                "gap": nl_by_h[h]["gap"],
                "de_noiseless": nl_by_h[h]["de_gap_noiseless"],
                "de_noisy": ny_by_h[h]["de_gap_noisy"],
                "de_gf": gf_by_h[h]["de_gap_gf"],
                "de_pea": pea_by_h[h]["de_gap_pea"],
                "gf_r2": gf_by_h[h]["gf_r2"],
                "pea_r2": pea_by_h[h]["pea_r2"],
                "gf_gain": (
                    (ny_by_h[h]["de_gap_noisy"] - gf_by_h[h]["de_gap_gf"])
                    / max(ny_by_h[h]["de_gap_noisy"], 1e-10)
                ),
                "pea_gain": (
                    (ny_by_h[h]["de_gap_noisy"] - pea_by_h[h]["de_gap_pea"])
                    / max(ny_by_h[h]["de_gap_noisy"], 1e-10)
                ),
                "passes_hw_criterion": pea_by_h[h]["de_gap_pea"] < DE_GAP_THRESHOLD,
                "gf_passes": gf_by_h[h]["de_gap_gf"] < DE_GAP_THRESHOLD,
            }
            comparison.append(row)

        # Print comparison table
        logger.info("")
        logger.info(
            f"  {'h':>5} | {'Noiseless':>10} | {'Noisy':>10} | "
            f"{'GF-ZNE':>10} {'R²':>5} | {'PEA-ZNE':>10} {'R²':>5} | HW Pass?"
        )
        logger.info("  " + "-" * 80)
        for row in comparison:
            hw_icon = "✅" if row["passes_hw_criterion"] else "❌"
            logger.info(
                f"  {row['h']:5.2f} | {row['de_noiseless']:10.4f} | {row['de_noisy']:10.4f} | "
                f"{row['de_gf']:10.4f} {row['gf_r2']:5.3f} | "
                f"{row['de_pea']:10.4f} {row['pea_r2']:5.3f} | {hw_icon}"
            )

        # Aggregate metrics
        mean_gf_gain = float(np.mean([r["gf_gain"] for r in comparison]))
        mean_pea_gain = float(np.mean([r["pea_gain"] for r in comparison]))
        mean_pea_r2 = float(np.mean([r["pea_r2"] for r in comparison]))
        mean_gf_r2 = float(np.mean([r["gf_r2"] for r in comparison]))
        n_hw_pass_pea = sum(1 for r in comparison if r["passes_hw_criterion"])
        n_hw_pass_gf = sum(1 for r in comparison if r["gf_passes"])

        # Key metrics for hardware
        mean_de_pea = float(np.mean([r["de_pea"] for r in comparison]))
        mean_de_gf = float(np.mean([r["de_gf"] for r in comparison]))
        mean_de_noisy = float(np.mean([r["de_noisy"] for r in comparison]))
        mean_de_noiseless = float(np.mean([r["de_noiseless"] for r in comparison]))

        logger.info("")
        logger.info("  ─── HARDWARE READINESS VERDICT ───")
        logger.info(f"  Topology: {self.topology}, N={self.n_qubits}, p={self.p_layers}")
        logger.info(
            f"  HW criterion (ΔE/gap < {DE_GAP_THRESHOLD * 100:.0f}%): "
            f"PEA passes {n_hw_pass_pea}/{len(comparison)}, "
            f"GF passes {n_hw_pass_gf}/{len(comparison)}"
        )
        logger.info(
            f"  Mean ΔE/gap:  noiseless={mean_de_noiseless:.4f}, "
            f"noisy={mean_de_noisy:.4f}, GF={mean_de_gf:.4f}, PEA={mean_de_pea:.4f}"
        )
        logger.info(f"  Mean gain:    GF={mean_gf_gain:+.1%}, PEA={mean_pea_gain:+.1%}")
        logger.info(f"  Mean R²:      GF={mean_gf_r2:.4f}, PEA={mean_pea_r2:.4f}")
        logger.info(f"  PEA > GF:     {mean_pea_gain > mean_gf_gain}")

        # Note: p=1 expressibility limit means ΔE/gap >> 5% even noiseless
        # The real hardware criterion applies to warm-start MPNN predictions
        # not raw VQE at arbitrary h. But for ZNE comparison, relative gain matters.
        if mean_de_noiseless > DE_GAP_THRESHOLD:
            logger.info(
                f"\n  NOTE: Even noiseless ΔE/gap={mean_de_noiseless:.4f} > "
                f"{DE_GAP_THRESHOLD * 100:.0f}% threshold."
                f"\n  This is the p=1 expressibility limit, NOT a ZNE failure."
                f"\n  ZNE effectiveness is measured by RELATIVE gain, not absolute ΔE/gap."
            )

        summary = {
            "topology": self.topology,
            "n_qubits": self.n_qubits,
            "comparison": comparison,
            "mean_de_noiseless": mean_de_noiseless,
            "mean_de_noisy": mean_de_noisy,
            "mean_de_gf": mean_de_gf,
            "mean_de_pea": mean_de_pea,
            "mean_gf_gain": mean_gf_gain,
            "mean_pea_gain": mean_pea_gain,
            "mean_gf_r2": mean_gf_r2,
            "mean_pea_r2": mean_pea_r2,
            "pea_better_than_gf": mean_pea_gain > mean_gf_gain,
            "n_hw_pass_pea": n_hw_pass_pea,
            "n_hw_pass_gf": n_hw_pass_gf,
            "affine_correction_applied": True,
        }

        # Pass = PEA is better than GF and has good R²
        passed = mean_pea_gain > mean_gf_gain and mean_pea_r2 > 0.9

        logger.info("")
        logger.info(
            "  ⚠ Note: Results are from depolarizing PEA approximation — "
            "real hardware may differ by ±10%."
        )

        # Cleanup checkpoints on success
        if passed:
            self.cleanup_checkpoints("noisy_*")

        return {"pass": passed, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    PEAHardwareReadinessRunner.main()
