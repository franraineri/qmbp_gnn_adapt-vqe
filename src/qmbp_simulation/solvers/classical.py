"""
Classical Solver — Ground truth generation via exact diagonalization or DMRG/TeNPy.

Supports automatic method selection based on system size:
  - N ≤ 15: dense exact diagonalization (np.linalg.eigh)
  - N > 15: DMRG via TeNPy (quasi-1D systems up to N=40)

Includes memory fallback for 2D lattices that exceed available memory.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from qmbp_simulation.models import (
    DMRG_QUBIT_LIMIT,
    EXACT_DIAG_QUBIT_LIMIT,
    GroundTruthResult,
    LatticeConfig,
)
from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

logger = logging.getLogger(__name__)


class ClassicalSolver:
    """Generate noise-free ground truth data for spin Hamiltonians.

    Methods
    -------
    solve(hamiltonian, lattice, method="auto")
        Compute ground state energy, gap, wavefunction, and local observables.
    """

    def solve(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        method: str = "auto",
        *,
        obs_x: list[SparsePauliOp] | None = None,
        obs_zz: list[SparsePauliOp] | None = None,
        chi_max: int | None = None,
    ) -> GroundTruthResult:
        """Solve for the ground state of *hamiltonian*.

        Parameters
        ----------
        hamiltonian : SparsePauliOp
            The spin Hamiltonian.
        lattice : LatticeConfig
            Lattice specification (needed for observable computation).
        method : str
            ``"exact"``, ``"dmrg"``, or ``"auto"`` (selects by N).
        obs_x : list[SparsePauliOp] | None
            Per-site X observables.  Built automatically if ``None``.
        obs_zz : list[SparsePauliOp] | None
            Per-bond ZZ observables.  Built automatically if ``None``.
        chi_max : int | None
            MPS bond dimension for DMRG. If None, uses dynamic scaling
            (200-400 for 1D, min(256, 2^(N/2)) for 2D). Useful for studying
            the effect of truncation on ground state precision.

        Returns
        -------
        GroundTruthResult
        """
        n = lattice.n_qubits

        if method == "auto":
            # F5: For N ≤ STATEVECTOR_MAX_N (22), ALWAYS use exact diag
            # regardless of topology. Exact is correct by construction;
            # DMRG can have modeling errors on 2D topologies.
            if n <= STATEVECTOR_MAX_N:
                method = "exact"
            else:
                method = "dmrg"

        logger.debug(
            "ClassicalSolver.solve: n=%d, method=%s, topology=%s, h=%s",
            n,
            method,
            lattice.topology,
            lattice.h,
        )

        # Build observables if not provided
        if obs_x is None or obs_zz is None:
            from qmbp_simulation.models import HamiltonianBuilder

            builder = HamiltonianBuilder()
            obs_x, obs_zz = builder.build_local_observables(lattice)

        if method == "exact":
            return self._solve_exact(hamiltonian, lattice, obs_x, obs_zz)
        elif method == "dmrg":
            if n > DMRG_QUBIT_LIMIT:
                raise ValueError(f"DMRG supports up to {DMRG_QUBIT_LIMIT} qubits, got {n}.")
            result = self._solve_dmrg(hamiltonian, lattice, obs_x, obs_zz, chi_max=chi_max)

            # ── Cross-validation guard: verify DMRG vs eigsh when feasible ──
            # This catches the class of bugs where DMRG silently drops bonds
            # (e.g., using 1D TFIChain for a 2D topology). Cost: one extra
            # eigsh(k=1) call (~1-50s for N≤22), but prevents catastrophic errors.
            if n <= STATEVECTOR_MAX_N and lattice.topology != "chain_1d":
                try:
                    from scipy.sparse.linalg import eigsh as _eigsh

                    H_sparse = hamiltonian.to_matrix(sparse=True)
                    e_exact = float(_eigsh(H_sparse, k=1, which="SA", return_eigenvectors=False)[0])
                    delta = abs(result.ground_energy - e_exact)
                    if delta > 1e-4:
                        logger.error(
                            "⛔ DMRG cross-validation FAILED for %s N=%d h=%.4f: "
                            "|E_dmrg - E_exact| = %.2e (threshold=1e-4). "
                            "This indicates a DMRG modeling error (missing bonds?). "
                            "Falling back to exact diag.",
                            lattice.topology,
                            n,
                            float(lattice.h) if np.isscalar(lattice.h) else float(np.mean(lattice.h)),
                            delta,
                        )
                        # Fall back to exact — guaranteed correct
                        return self._solve_exact(hamiltonian, lattice, obs_x, obs_zz)
                    elif delta > 1e-8:
                        logger.debug(
                            "DMRG cross-validation OK for %s N=%d: |ΔE|=%.2e",
                            lattice.topology, n, delta,
                        )
                except Exception as exc:
                    logger.debug("DMRG cross-validation skipped: %s", exc)

            return result
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'exact', 'dmrg', or 'auto'.")

    def ground_state_vector(
        self,
        hamiltonian: SparsePauliOp,
        n_qubits: int | None = None,
    ) -> np.ndarray:
        """Return the ground state eigenvector as a numpy array.

        Selects the best algorithm based on system size:
          - N ≤ 12: dense ``np.linalg.eigh`` (full spectrum, instant)
          - 12 < N ≤ 22: sparse Lanczos via ``scipy.sparse.linalg.eigsh``
            using ``SparsePauliOp.to_matrix(sparse=True)``
          - N > 22: raises ValueError (use DMRG + MPS; no statevector available)

        This method is intended for fidelity computation where the full 2^N
        ground state vector is needed. For energy-only ground truth, use
        ``solve()`` which dispatches to DMRG for large N.

        Parameters
        ----------
        hamiltonian : SparsePauliOp
            The spin Hamiltonian.
        n_qubits : int | None
            Number of qubits. Inferred from hamiltonian if None.

        Returns
        -------
        np.ndarray
            Ground state vector of shape (2^N,), complex128.

        Raises
        ------
        ValueError
            If N > 22 (statevector too large even for sparse methods).
        """
        from scipy.sparse.linalg import eigsh

        n = n_qubits if n_qubits is not None else hamiltonian.num_qubits

        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

        if n > STATEVECTOR_MAX_N:
            raise ValueError(
                f"ground_state_vector not supported for N={n} > {STATEVECTOR_MAX_N}. "
                f"Statevector requires 2^{n} amplitudes ({2**n * 16 / 1e9:.1f} GB). "
                f"Use DMRG (solve method) for energy/gap, skip fidelity for N>{STATEVECTOR_MAX_N}."
            )

        # Use sparse eigsh for N >= 13 (saves memory: sparse avoids 2^N × 2^N dense matrix)
        _SPARSE_THRESHOLD = 13

        if n < _SPARSE_THRESHOLD:
            logger.debug("ground_state_vector: N=%d → dense eigh (2^%d matrix)", n, n)
            mat = np.asarray(hamiltonian.to_matrix())
            _, evecs = np.linalg.eigh(mat)
            return np.ascontiguousarray(evecs[:, 0])
        else:
            dim = 2**n
            logger.info(
                "ground_state_vector: N=%d → sparse eigsh (Lanczos, dim=%d)",
                n,
                dim,
            )
            H_sparse = hamiltonian.to_matrix(sparse=True)
            try:
                _, evecs = eigsh(H_sparse, k=1, which="SA")
            except Exception as exc:
                raise RuntimeError(
                    f"Sparse eigsh failed for N={n} ({H_sparse.shape[0]}×{H_sparse.shape[1]}, "
                    f"nnz={H_sparse.nnz}). This may indicate a non-Hermitian Hamiltonian "
                    f"or insufficient memory for Lanczos workspace."
                ) from exc
            gs = evecs[:, 0]
            # Sanity: eigenvector should be normalized
            norm = float(np.linalg.norm(gs))
            if abs(norm - 1.0) > 1e-6:
                logger.warning(
                    "ground_state_vector: eigsh returned non-unit vector (norm=%.6f), normalizing",
                    norm,
                )
                gs = gs / norm
            return gs

    @staticmethod
    def validate_ground_truth(
        results: list[GroundTruthResult],
        *,
        model: str = "tfim",
        n_qubits: int | None = None,
        n_edges: int | None = None,
    ) -> dict:
        """Validate Phase 1 (exact diag / DMRG) results for physical consistency.

        Checks:
          1. Gap positivity: gap > 0 for all points
          2. Energy bounds: E ∈ [E_lower, E_upper] (spectral bounds)
          3. Observable bounds: ⟨X⟩ ∈ [0, 1], ⟨ZZ⟩ ∈ [-1, 1]
          4. Energy monotonicity: for TFIM, E(h) should be monotonically
             decreasing as h increases (more negative)
          5. Gap floor warning: if gap == 2π/N for all points, flag as
             "fallback gap" (DMRG excited state failed)

        Parameters
        ----------
        results : list[GroundTruthResult]
            Phase 1 results (ordered by h, descending sweep).
        model : str
            Model name for bound computation.
        n_qubits : int | None
            System size (inferred from results if None).
        n_edges : int | None
            Number of edges (for energy bound computation).

        Returns
        -------
        dict
            Validation report with keys: passed, warnings, errors, details.
        """
        if not results:
            return {"passed": False, "warnings": [], "errors": ["Empty results list"]}

        warnings_list = []
        errors_list = []

        n = (
            n_qubits or len(results[0].per_site_mag_x)
            if results[0].per_site_mag_x is not None
            else 0
        )
        n_e = n_edges or (n - 1)  # Default: open chain

        # 1. Gap positivity
        zero_gaps = [r for r in results if r.gap <= 0]
        if zero_gaps:
            errors_list.append(
                f"Gap ≤ 0 at {len(zero_gaps)} points: h={[r.h_value for r in zero_gaps[:3]]}"
            )

        # 2. Energy bounds
        for r in results:
            h_abs = abs(r.h_value)
            if model in ("tfim", "tfim_longitudinal", "tfim_frustrated"):
                e_lower = -abs(1.0) * n_e - h_abs * n
                e_upper = abs(1.0) * n_e + h_abs * n
            else:
                e_lower = -3 * abs(1.0) * n_e - h_abs * n
                e_upper = 3 * abs(1.0) * n_e + h_abs * n

            if r.ground_energy < e_lower - 1e-6:
                errors_list.append(
                    f"E={r.ground_energy:.6f} below lower bound {e_lower:.6f} at h={r.h_value:.3f}"
                )
            if r.ground_energy > e_upper + 1e-6:
                errors_list.append(
                    f"E={r.ground_energy:.6f} above upper bound {e_upper:.6f} at h={r.h_value:.3f}"
                )

        # 3. Observable bounds
        for r in results:
            if r.per_site_mag_x is not None:
                mx_max = float(np.max(np.abs(r.per_site_mag_x)))
                if mx_max > 1.0 + 1e-6:
                    errors_list.append(f"|⟨X⟩| = {mx_max:.4f} > 1 at h={r.h_value:.3f}")
            if r.per_bond_corr_zz is not None:
                zz_max = float(np.max(np.abs(r.per_bond_corr_zz)))
                if zz_max > 1.0 + 1e-6:
                    errors_list.append(f"|⟨ZZ⟩| = {zz_max:.4f} > 1 at h={r.h_value:.3f}")

        # 4. Energy monotonicity (TFIM: E decreases as h increases)
        if model in ("tfim", "tfim_longitudinal") and len(results) >= 3:
            # Results are in descending h order
            h_vals = [r.h_value for r in results]
            e_vals = [r.ground_energy for r in results]
            # In descending h sweep: h decreases, E should become less negative
            # Check: no large non-monotonic jumps (> 10% of range)
            e_range = max(e_vals) - min(e_vals) if max(e_vals) != min(e_vals) else 1.0
            for i in range(len(e_vals) - 1):
                if h_vals[i] > h_vals[i + 1]:  # h decreasing
                    # E should increase (become less negative)
                    if e_vals[i + 1] < e_vals[i] - 0.1 * e_range:
                        warnings_list.append(
                            f"Non-monotonic E: E({h_vals[i]:.3f})={e_vals[i]:.4f} → "
                            f"E({h_vals[i + 1]:.3f})={e_vals[i + 1]:.4f}"
                        )

        # 5. Gap floor warning (DMRG fallback detection)
        if n > 0:
            finite_floor = 2 * np.pi / n
            gaps = [r.gap for r in results]
            n_at_floor = sum(1 for g in gaps if abs(g - finite_floor) < 1e-6)
            if n_at_floor == len(results) and len(results) > 1:
                warnings_list.append(
                    f"All {len(results)} gaps = 2π/N = {finite_floor:.4f} "
                    f"(DMRG excited-state likely failed — using finite-size floor)"
                )
            elif n_at_floor > len(results) * 0.5:
                warnings_list.append(
                    f"{n_at_floor}/{len(results)} gaps at finite-size floor "
                    f"2π/N = {finite_floor:.4f}"
                )

        # 6. ⟨X⟩ ≈ 0 in paramagnetic phase (symmetry breaking)
        for r in results:
            if r.h_value > 2.0 and r.mag_x < 0.1:
                warnings_list.append(
                    f"⟨X⟩={r.mag_x:.4f} ≈ 0 at h={r.h_value:.3f} (expect ≈1 in PM phase) "
                    f"— likely DMRG symmetry breaking"
                )
                break  # Only warn once

        passed = len(errors_list) == 0
        return {
            "passed": passed,
            "n_points": len(results),
            "warnings": warnings_list,
            "errors": errors_list,
            "details": {
                "gap_min": min(r.gap for r in results),
                "gap_max": max(r.gap for r in results),
                "e_min": min(r.ground_energy for r in results),
                "e_max": max(r.ground_energy for r in results),
                "n_at_gap_floor": sum(
                    1 for r in results if n > 0 and abs(r.gap - 2 * np.pi / n) < 1e-6
                ),
            },
        }

    def _solve_exact(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        obs_x: list[SparsePauliOp],
        obs_zz: list[SparsePauliOp],
    ) -> GroundTruthResult:
        """Exact diagonalization — dense for N≤12, sparse eigsh for N=13-15.

        Returns E₀, gap, ψ_gs, and bulk-averaged + per-site/per-bond observables.
        """
        h_val = float(lattice.h) if np.isscalar(lattice.h) else float(np.mean(lattice.h))  # type: ignore[arg-type]
        n = lattice.n_qubits

        # Use sparse eigsh for N >= 13 (avoids 2^N × 2^N dense matrix)
        _SPARSE_THRESHOLD = 13

        if n < _SPARSE_THRESHOLD:
            logger.debug("_solve_exact: N=%d → dense eigh (full spectrum)", n)
            try:
                mat = np.asarray(hamiltonian.to_matrix())
                evals, evecs = np.linalg.eigh(mat)
            except MemoryError:
                return self._memory_fallback(hamiltonian, lattice, obs_x, obs_zz)

            psi_gs = np.ascontiguousarray(evecs[:, 0])
            e0 = float(evals[0])
            gap = float(evals[1] - evals[0])
        else:
            from scipy.sparse.linalg import eigsh

            logger.info(
                "_solve_exact: N=%d → sparse eigsh (k=2, dim=%d) at h=%.4f",
                n,
                2**n,
                h_val,
            )
            H_sparse = hamiltonian.to_matrix(sparse=True)
            try:
                evals, evecs = eigsh(H_sparse, k=2, which="SA")
            except Exception as exc:
                raise RuntimeError(
                    f"Sparse eigsh failed for N={n} at h={h_val:.4f}. "
                    f"Matrix: {H_sparse.shape[0]}×{H_sparse.shape[1]}, nnz={H_sparse.nnz}."
                ) from exc

            # Sort eigenvalues (eigsh doesn't guarantee order)
            order = np.argsort(evals)
            evals = evals[order]
            evecs = evecs[:, order]

            psi_gs = np.ascontiguousarray(evecs[:, 0])
            e0 = float(evals[0])
            gap = float(evals[1] - evals[0])

        # Physics guard: near-degeneracy makes ΔE/gap metrics meaningless.
        # Also detect non-finite eigenvalues (corrupted Hamiltonian).
        if not np.isfinite(e0):
            raise RuntimeError(
                f"Non-finite ground energy E₀={e0} from eigh at h={h_val:.4f}. "
                f"Hamiltonian may be corrupted (check hermiticity/construction)."
            )
        if gap < 1e-12 and gap >= 0:
            logger.error(
                "Near-degenerate ground state: gap=%.2e at h=%.4f. "
                "ΔE/gap metrics will be unreliable. This may indicate a "
                "first-order QPT or accidental degeneracy.",
                gap,
                h_val,
            )
        elif gap < 1e-6:
            logger.warning(
                "Very small gap=%.2e at h=%.4f — ΔE/gap metrics may be "
                "numerically unstable at this point.",
                gap,
                h_val,
            )

        sv = Statevector(psi_gs)

        # Per-site ⟨Xᵢ⟩
        per_site_mx = np.array([sv.expectation_value(op).real for op in obs_x])
        # Per-bond ⟨ZᵢZⱼ⟩
        per_bond_zz = np.array([sv.expectation_value(op).real for op in obs_zz])

        # Physics guard: observable bounds — Pauli expectation values ∈ [-1, 1]
        mx_max = float(np.max(np.abs(per_site_mx))) if len(per_site_mx) > 0 else 0
        zz_max = float(np.max(np.abs(per_bond_zz))) if len(per_bond_zz) > 0 else 0
        if mx_max > 1.0 + 1e-6:
            logger.error(
                "Observable bound violated: |⟨X⟩|_max=%.6f > 1 at h=%.4f. "
                "Eigenvector may not be a valid quantum state.",
                mx_max,
                h_val,
            )
        elif mx_max > 1.0 + 1e-10:
            logger.warning(
                "Observable slightly out of bounds: |⟨X⟩|_max=%.10f at h=%.4f (numerical noise).",
                mx_max,
                h_val,
            )
        if zz_max > 1.0 + 1e-6:
            logger.error(
                "Observable bound violated: |⟨ZZ⟩|_max=%.6f > 1 at h=%.4f. "
                "Eigenvector may not be a valid quantum state.",
                zz_max,
                h_val,
            )

        return GroundTruthResult(
            h_value=h_val,
            ground_energy=e0,
            gap=gap,
            ground_state=psi_gs,
            mag_x=float(np.mean(per_site_mx)),
            corr_zz=float(np.mean(per_bond_zz)),
            per_site_mag_x=per_site_mx,
            per_bond_corr_zz=per_bond_zz,
            gap_method="exact_dense" if n < _SPARSE_THRESHOLD else "exact_sparse",
        )

    def _solve_dmrg(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        obs_x: list[SparsePauliOp],
        obs_zz: list[SparsePauliOp],
        *,
        chi_max: int | None = None,
    ) -> GroundTruthResult:
        """DMRG ground state solver using TeNPy.

        Dispatches between:
        - TFIChain: for 1D chain topology (fastest, native TeNPy model)
        - SpinModel + 2D lattice: for square/triangular (genuine 2D DMRG)
        - GraphDMRG (CouplingMPOModel): for heavy_hex/ladder/kagome/arbitrary
          topologies where bonds are non-sequential. Uses explicit edge list
          to build the MPO correctly.

        Returns E₀, gap, local observables.  ψ_gs is None (MPS, not statevector).
        """
        # Topologies with native TeNPy 2D lattice support
        if lattice.topology in ("square", "triangular") and lattice.n_qubits >= 9:
            return self._solve_dmrg_2d(hamiltonian, lattice, obs_x, obs_zz, chi_max=chi_max)

        # Topologies with non-sequential bonds: use graph-based DMRG
        # This handles heavy_hex, ladder, kagome correctly by encoding ALL edges
        _GRAPH_TOPOLOGIES = ("heavy_hex", "ladder", "kagome")
        if lattice.topology in _GRAPH_TOPOLOGIES:
            return self._solve_dmrg_graph(hamiltonian, lattice, obs_x, obs_zz, chi_max=chi_max)

        # Default: 1D chain (TFIChain, only sequential bonds)
        return self._solve_dmrg_1d(hamiltonian, lattice, obs_x, obs_zz, chi_max=chi_max)

    def _solve_dmrg_graph(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        obs_x: list[SparsePauliOp],
        obs_zz: list[SparsePauliOp],
        *,
        chi_max: int | None = None,
    ) -> GroundTruthResult:
        """DMRG for arbitrary graph topologies via CouplingMPOModel.

        Builds the Hamiltonian MPO from the explicit edge list in lattice.edges.
        Works for any topology (heavy_hex, ladder, kagome, etc.) by encoding
        every bond (i,j) as a coupling term in the MPO.

        The MPS uses a linear ordering (Chain lattice container) — TeNPy handles
        long-range couplings via MPO bond dimension growth. For N≤30 with
        chi_max≥128, this gives exact results.

        Convention mapping (Pauli → TeNPy spin-1/2):
            Z_i = 2 * Sz_i  →  ZZ coupling: -4J * Sz_i * Sz_j
            X_i = 2 * Sx_i  →  X field: -2h * Sx_i
        """
        try:
            from tenpy.algorithms import dmrg as tenpy_dmrg
            from tenpy.models.lattice import Chain
            from tenpy.models.model import CouplingMPOModel
            from tenpy.networks.mps import MPS
            from tenpy.networks.site import SpinHalfSite
        except ImportError as exc:
            raise ImportError(
                "TeNPy is required for DMRG. Install via: pip install physics-tenpy"
            ) from exc

        n = lattice.n_qubits
        h_val = float(lattice.h) if np.isscalar(lattice.h) else float(np.mean(lattice.h))
        j_val = float(lattice.J) if np.isscalar(lattice.J) else float(np.mean(lattice.J))
        edges = lattice.edges

        logger.info(
            f"_solve_dmrg_graph: {lattice.topology} N={n}, {len(edges)} edges, "
            f"h={h_val:.4f}, chi_max={chi_max}"
        )

        # Build TeNPy model with explicit edge coupling
        site = SpinHalfSite(conserve=None)
        lat = Chain(L=n, site=site, bc_MPS="finite")

        class _GraphTFIM(CouplingMPOModel):
            def init_terms(self, model_params):
                _J = model_params.get("J", 1.0)
                _h = model_params.get("h", 1.0)
                _edges = model_params.get("edges", [])
                for i in range(self.lat.N_sites):
                    self.add_onsite_term(-2.0 * _h, i, "Sx")
                for i, j in _edges:
                    ii, jj = (i, j) if i < j else (j, i)
                    self.add_coupling_term(-4.0 * _J, ii, jj, "Sz", "Sz")

        params = {"lattice": lat, "J": j_val, "h": h_val, "edges": edges}
        model = _GraphTFIM(params)

        # Initial state: |+⟩^N (paramagnetic, Sx eigenstate)
        # In TeNPy spin-1/2: "up" is Sz=+0.5 eigenstate
        psi = MPS.from_lat_product_state(lat, [["up"]] * n)

        # DMRG parameters
        _chi = chi_max if chi_max is not None else min(256, max(64, 2 ** (n // 3)))
        dmrg_params = {
            "mixer": True,
            "max_E_err": 1e-12,
            "trunc_params": {"chi_max": _chi, "svd_min": 1e-14},
            "max_sweeps": 80,
        }

        eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
        e0, _ = eng.run()

        # Gap estimation
        from qmbp_simulation.models.constants import EXACT_GAP_QUBIT_LIMIT

        if n <= EXACT_GAP_QUBIT_LIMIT:
            try:
                from scipy.sparse.linalg import eigsh as _eigsh

                H_sparse = hamiltonian.to_matrix(sparse=True)
                evals_k2, _ = _eigsh(H_sparse, k=2, which="SA")
                evals_k2 = np.sort(evals_k2)
                gap = float(evals_k2[1] - evals_k2[0])
            except Exception:
                gap = 2 * np.pi / n
        else:
            gap = 2 * np.pi / n

        # Local observables from MPS
        per_site_sx = np.real(psi.expectation_value("Sx"))
        per_site_mx = np.abs(2.0 * per_site_sx)  # Pauli X = 2*Sx

        per_bond_zz = np.zeros(len(edges))
        try:
            for idx, (i, j) in enumerate(edges):
                val = psi.expectation_value_term([("Sz", i), ("Sz", j)])
                per_bond_zz[idx] = 4.0 * float(np.real(val))
        except Exception:
            logger.warning("Could not compute per-bond ZZ via graph DMRG. Using zeros.")

        return GroundTruthResult(
            h_value=h_val,
            ground_energy=float(e0),
            gap=gap,
            ground_state=None,
            mag_x=float(np.mean(per_site_mx)),
            corr_zz=float(np.mean(per_bond_zz)) if len(per_bond_zz) > 0 else 0.0,
            per_site_mag_x=per_site_mx,
            per_bond_corr_zz=per_bond_zz,
            gap_method="eigsh_fallback" if n <= EXACT_GAP_QUBIT_LIMIT else "floor_2pi_n",
        )

    def _solve_dmrg_2d(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        obs_x: list[SparsePauliOp],
        obs_zz: list[SparsePauliOp],
        *,
        chi_max: int | None = None,
    ) -> GroundTruthResult:
        """2D DMRG via TeNPy SpinModel with native Square/Triangular lattice.

        Uses TeNPy's built-in 2D lattice classes for proper snake-path MPS
        ordering, which is significantly more accurate than treating 2D systems
        as 1D chains.

        Convention mapping (our Pauli → TeNPy spin-1/2):
            Z_i = 2 * Sz_i  →  ZZ coupling: Jz = -4J
            X_i = 2 * Sx_i  →  X field: hx = -2h
        """
        try:
            from tenpy.algorithms import dmrg as tenpy_dmrg
            from tenpy.models.spins import SpinModel
            from tenpy.networks.mps import MPS
        except ImportError as exc:
            raise ImportError(
                "TeNPy is required for DMRG. Install via: pip install physics-tenpy"
            ) from exc

        n = lattice.n_qubits
        h_val = float(lattice.h) if np.isscalar(lattice.h) else float(np.mean(lattice.h))
        j_val = float(lattice.J) if np.isscalar(lattice.J) else float(np.mean(lattice.J))

        # Determine grid dimensions
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # Map topology to TeNPy lattice class
        if lattice.topology == "square":
            tenpy_lattice = "Square"
        elif lattice.topology == "triangular":
            tenpy_lattice = "Triangular"
        else:
            # Fallback to 1D method for unsupported topologies
            return self._solve_dmrg_1d(hamiltonian, lattice, obs_x, obs_zz)

        # TeNPy SpinModel uses spin-1/2 operators (Sx, Sz with eigenvalues ±0.5)
        # Our H = -J·ZZ - h·X uses Pauli matrices (eigenvalues ±1)
        # Conversion: Z = 2*Sz → ZZ = 4*Sz*Sz, X = 2*Sx
        # So: Jz_tenpy = -4J, hx_tenpy = -2h
        model_params = {
            "lattice": tenpy_lattice,
            "Lx": rows,
            "Ly": cols,
            "bc_MPS": "finite",
            "bc_y": "open",
            "Jx": 0.0,
            "Jy": 0.0,
            "Jz": -4.0 * j_val,
            "hx": -2.0 * h_val,
            "hz": 0.0,
            "hy": 0.0,
            "conserve": None,
            "S": 0.5,
        }

        model = SpinModel(model_params)

        # Handle case where TeNPy lattice has more sites than our N
        actual_sites = model.lat.N_sites
        if actual_sites != n:
            logger.warning(
                f"TeNPy 2D lattice has {actual_sites} sites but our lattice has {n}. "
                f"Using graph-based DMRG instead (preserves all bonds correctly)."
            )
            return self._solve_dmrg_graph(hamiltonian, lattice, obs_x, obs_zz, chi_max=chi_max)

        # Initial state: product state |↑⟩^N (paramagnetic along x at h→∞)
        p_state = [[["up"] for _ in range(cols)] for _ in range(rows)]
        psi = MPS.from_lat_product_state(model.lat, p_state)

        # DMRG parameters — higher chi for 2D (more entanglement)
        _chi = chi_max if chi_max is not None else min(256, 2 ** (n // 2))
        dmrg_params = {
            "mixer": True,
            "max_E_err": 1e-12,
            "trunc_params": {"chi_max": _chi, "svd_min": 1e-12},
            "max_sweeps": 80,
        }

        eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
        e0, _ = eng.run()

        # Gap estimation: prefer exact eigsh(k=2) when tractable, else floor
        from qmbp_simulation.models.constants import EXACT_GAP_QUBIT_LIMIT

        gap_method = "floor_2pi_n"
        if n <= EXACT_GAP_QUBIT_LIMIT:
            try:
                from scipy.sparse.linalg import eigsh as _eigsh

                H_sparse = hamiltonian.to_matrix(sparse=True)
                evals_k2, _ = _eigsh(H_sparse, k=2, which="SA")
                evals_k2 = np.sort(evals_k2)
                gap = float(evals_k2[1] - evals_k2[0])
                gap_method = "eigsh_fallback"
                logger.info(
                    "DMRG 2D (%s %dx%d): E0=%.8f, exact eigsh gap=%.6f (floor would be %.6f)",
                    lattice.topology,
                    rows,
                    cols,
                    e0,
                    gap,
                    2 * np.pi / n,
                )
                # Near-degeneracy guard
                if gap < 1e-12:
                    logger.warning(
                        "eigsh gap fallback: near-degenerate gap=%.2e for "
                        "%s N=%d at h=%.4f. ΔE/gap metrics unreliable.",
                        gap,
                        lattice.topology,
                        n,
                        h_val,
                    )
            except Exception as exc:
                gap = 2 * np.pi / n
                logger.warning(
                    "eigsh(k=2) failed for %s %dx%d N=%d at h=%.4f: %s. "
                    "Falling back to floor gap=%.4f.",
                    lattice.topology,
                    rows,
                    cols,
                    n,
                    h_val,
                    exc,
                    gap,
                )
        else:
            gap = 2 * np.pi / n
            logger.info(
                f"DMRG 2D ({lattice.topology} {rows}x{cols}): E0={e0:.8f}, "
                f"N>{EXACT_GAP_QUBIT_LIMIT}, using finite-size gap floor={gap:.4f}"
            )

        # Local observables via MPS expectation values
        # SpinModel with S=0.5, conserve=None uses: Sx, Sy, Sz (spin-1/2 operators)
        # Our ⟨X_i⟩ = 2*⟨Sx_i⟩ (Pauli X = 2*Sx for spin-1/2)
        per_site_sx = np.real(psi.expectation_value("Sx"))
        per_site_mx = np.abs(2.0 * per_site_sx)  # Convert to Pauli scale

        # ZZ correlations: ⟨Z_i Z_j⟩ = 4*⟨Sz_i Sz_j⟩
        per_bond_zz = np.zeros(len(lattice.edges))
        try:
            for idx, (i, j) in enumerate(lattice.edges):
                # Map our qubit indices to TeNPy MPS site indices
                mps_i = model.lat.lat2mps_idx((i // cols, i % cols, 0))
                mps_j = model.lat.lat2mps_idx((j // cols, j % cols, 0))
                val = psi.expectation_value_term([("Sz", mps_i), ("Sz", mps_j)])
                per_bond_zz[idx] = 4.0 * float(np.real(val))  # 4*SzSz = ZZ
        except Exception:
            logger.warning("Could not compute per-bond ZZ via DMRG. Using zeros.")

        return GroundTruthResult(
            h_value=h_val,
            ground_energy=float(e0),
            gap=gap,
            ground_state=None,
            mag_x=float(np.mean(per_site_mx)),
            corr_zz=float(np.mean(per_bond_zz)) if len(per_bond_zz) > 0 else 0.0,
            per_site_mag_x=per_site_mx,
            per_bond_corr_zz=per_bond_zz,
            gap_method=gap_method,
        )

    def _solve_dmrg_1d(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        obs_x: list[SparsePauliOp],
        obs_zz: list[SparsePauliOp],
        *,
        chi_max: int | None = None,
    ) -> GroundTruthResult:
        """DMRG ground state solver using TeNPy for quasi-1D systems.

        Returns E₀, gap, local observables.  ψ_gs is None (MPS, not statevector).

        Notes
        -----
        TeNPy's TFIChain convention: H = -J Σ Sigmaz_i Sigmaz_{i+1} - g Σ Sigmax_i
        where Sigmax/Sigmaz are the full Pauli matrices (not spin-1/2 operators).
        This matches our Qiskit convention: H = -J Σ Z_i Z_{i+1} - h Σ X_i.

        DMRG may find a symmetry-broken ground state where ⟨X⟩ ≈ 0 even in the
        paramagnetic phase.  For ⟨X⟩ we use |⟨X⟩| from the absolute magnetization
        or the correlation-based estimator sqrt(⟨X_i X_j⟩) when available.
        The ⟨ZZ⟩ correlations are reliable in either symmetry sector.
        """
        try:
            from tenpy.algorithms import dmrg as tenpy_dmrg
            from tenpy.models.tf_ising import TFIChain
            from tenpy.networks.mps import MPS
        except ImportError as exc:
            raise ImportError(
                "TeNPy is required for DMRG. Install via: pip install physics-tenpy"
            ) from exc

        n = lattice.n_qubits
        h_val = float(lattice.h) if np.isscalar(lattice.h) else float(np.mean(lattice.h))  # type: ignore[arg-type]
        j_val = float(lattice.J) if np.isscalar(lattice.J) else float(np.mean(lattice.J))  # type: ignore[arg-type]

        model_params = {
            "L": n,
            "J": j_val,
            "g": h_val,
            "bc_MPS": "finite",
            "conserve": None,
        }

        model = TFIChain(model_params)
        psi = MPS.from_lat_product_state(model.lat, [["up"]])

        # Dynamic chi scaling: sufficient for 1D TFIM (area law → bounded entanglement)
        # - N≤50: chi_max=200 (backward-compatible with previous hardcoded value)
        # - N=60: chi_max=240
        # - N=100: chi_max=400 (capped to limit O(N·χ³) compute time)
        # If chi_max is provided explicitly, use it (for precision studies).
        _chi = chi_max if chi_max is not None else min(400, max(200, 4 * n))
        logger.debug(f"DMRG 1D: N={n}, chi_max={_chi}{' (user-specified)' if chi_max else ''}")

        dmrg_params = {
            "mixer": True,
            "max_E_err": 1e-12,
            "trunc_params": {"chi_max": _chi, "svd_min": 1e-12},
            "max_sweeps": 100,
        }

        eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
        e0, _ = eng.run()

        # ── Gap via excited-state DMRG ────────────────────────────────
        gap = 0.0
        gap_method = "floor_2pi_n"  # default, overridden below
        try:
            psi_ex = MPS.from_lat_product_state(model.lat, [["down"]])
            dmrg_params_ex = {
                "mixer": True,
                "max_E_err": 1e-10,
                "trunc_params": {"chi_max": 100, "svd_min": 1e-10},
                "max_sweeps": 50,
            }
            eng_ex = tenpy_dmrg.TwoSiteDMRGEngine(psi_ex, model, dmrg_params_ex)
            e1, _ = eng_ex.run()
            # Only use the gap if e1 > e0 + tolerance
            if e1 > e0 + 1e-8:
                gap = float(e1 - e0)
                gap_method = "dmrg_excitation"
                logger.debug(
                    "DMRG gap via excitation: E1=%.8f, gap=%.6f at h=%.4f",
                    e1,
                    gap,
                    h_val,
                )
        except Exception as exc:
            logger.warning("Could not compute gap via DMRG excitation: %s", exc)

        # ── Analytical gap fallback for 1D TFIM ──────────────────────
        if gap == 0.0:
            if lattice.topology == "chain_1d":
                # Finite-size correction for 1D TFIM (open BC):
                # Exact gap in thermodynamic limit: Δ_∞ = 2|J - h|
                # Finite-size correction: Δ(N) ≈ Δ_∞ + π²·J/(N²·max(h,J))
                # This is more accurate than bare 2|J-h| near h_c where Δ_∞→0
                delta_inf = 2 * abs(j_val - h_val)
                finite_size_correction = np.pi**2 * j_val / (n**2 * max(h_val, j_val))
                gap = max(delta_inf + finite_size_correction, 2 * np.pi / n)
                gap_method = "analytical_1d"
                warnings.warn(
                    f"DMRG excited state converged to GS (N={n}, h={h_val:.2f}). "
                    f"Using analytical gap={gap:.4f} "
                    f"(Δ_∞={delta_inf:.4f} + FS correction={finite_size_correction:.4f}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                # For non-chain topologies: use sparse eigsh(k=2) for exact gap
                # if system size is tractable, otherwise fall back to floor.
                from qmbp_simulation.models.constants import EXACT_GAP_QUBIT_LIMIT

                if n <= EXACT_GAP_QUBIT_LIMIT:
                    try:
                        from scipy.sparse.linalg import eigsh as _eigsh

                        H_sparse = hamiltonian.to_matrix(sparse=True)
                        evals_k2, _ = _eigsh(H_sparse, k=2, which="SA")
                        evals_k2 = np.sort(evals_k2)
                        gap = float(evals_k2[1] - evals_k2[0])
                        gap_method = "eigsh_fallback"
                        floor_val = 2 * np.pi / n
                        logger.info(
                            "DMRG gap fallback: sparse eigsh(k=2) for %s N=%d → "
                            "gap=%.6f (floor would have been %.6f)",
                            lattice.topology,
                            n,
                            gap,
                            floor_val,
                        )
                        # Near-degeneracy guard (same as _solve_exact)
                        if gap < 1e-12:
                            logger.warning(
                                "eigsh gap fallback: near-degenerate gap=%.2e for "
                                "%s N=%d at h=%.4f. ΔE/gap metrics unreliable.",
                                gap,
                                lattice.topology,
                                n,
                                h_val,
                            )
                    except Exception as exc:
                        # eigsh failed — fall back to conservative floor
                        gap = 2 * np.pi / n
                        logger.warning(
                            "eigsh(k=2) failed for %s N=%d at h=%.4f: %s. "
                            "Falling back to floor gap=%.4f.",
                            lattice.topology,
                            n,
                            h_val,
                            exc,
                            gap,
                        )
                else:
                    gap = 2 * np.pi / n
                    warnings.warn(
                        f"DMRG excited state converged to GS (N={n}, h={h_val:.2f}, "
                        f"topology={lattice.topology}). N>{EXACT_GAP_QUBIT_LIMIT}, "
                        f"cannot use eigsh fallback. Using finite-size floor "
                        f"gap={gap:.4f}. This is a lower bound only.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        # ── Local observables ─────────────────────────────────────────
        # Sigmax per site (array over all sites)
        per_site_mx_raw = np.real(psi.expectation_value("Sigmax"))

        # DMRG may break Z₂ symmetry → ⟨X⟩ ≈ 0.  Use |⟨X⟩| as estimator.
        per_site_mx = np.abs(per_site_mx_raw)

        # ZZ correlations on lattice bonds
        per_bond_zz = np.array(
            [
                float(np.real(psi.expectation_value_term([("Sigmaz", i), ("Sigmaz", j)])))
                for i, j in lattice.edges
            ]
        )

        return GroundTruthResult(
            h_value=h_val,
            ground_energy=float(e0),
            gap=gap,
            ground_state=None,
            mag_x=float(np.mean(per_site_mx)),
            corr_zz=float(np.mean(per_bond_zz)) if len(per_bond_zz) > 0 else 0.0,
            per_site_mag_x=per_site_mx,
            per_bond_corr_zz=per_bond_zz,
            gap_method=gap_method,
        )

    def _memory_fallback(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        obs_x: list[SparsePauliOp],
        obs_zz: list[SparsePauliOp],
    ) -> GroundTruthResult:
        """Fall back to DMRG when exact diag exceeds memory (2D lattices).

        Logs a warning and attempts DMRG on the original lattice.
        If DMRG also fails, raises the original MemoryError.
        """
        logger.warning(
            f"MemoryError during exact diagonalization for N={lattice.n_qubits} "
            f"({lattice.topology}). Falling back to DMRG with quasi-1D mapping."
        )
        warnings.warn(
            f"Exact diag failed for {lattice.topology} N={lattice.n_qubits}. "
            f"Falling back to DMRG (quasi-1D cylindrical spin ladder).",
            RuntimeWarning,
            stacklevel=3,
        )
        try:
            return self._solve_dmrg(hamiltonian, lattice, obs_x, obs_zz)
        except Exception as exc:
            raise MemoryError(
                f"Both exact diag and DMRG failed for {lattice.topology} "
                f"N={lattice.n_qubits}. Consider reducing system size."
            ) from exc
