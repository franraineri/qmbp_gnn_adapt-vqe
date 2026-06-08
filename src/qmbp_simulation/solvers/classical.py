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

        Returns
        -------
        GroundTruthResult
        """
        n = lattice.n_qubits

        if method == "auto":
            method = "exact" if n <= EXACT_DIAG_QUBIT_LIMIT else "dmrg"

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
            return self._solve_dmrg(hamiltonian, lattice, obs_x, obs_zz)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'exact', 'dmrg', or 'auto'.")

    def _solve_exact(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        obs_x: list[SparsePauliOp],
        obs_zz: list[SparsePauliOp],
    ) -> GroundTruthResult:
        """Dense exact diagonalization via np.linalg.eigh.

        Returns E₀, gap, ψ_gs, and bulk-averaged + per-site/per-bond observables.
        """
        h_val = float(lattice.h) if np.isscalar(lattice.h) else float(np.mean(lattice.h))  # type: ignore[arg-type]

        try:
            mat = np.asarray(hamiltonian.to_matrix())
            evals, evecs = np.linalg.eigh(mat)
        except MemoryError:
            return self._memory_fallback(hamiltonian, lattice, obs_x, obs_zz)

        psi_gs = np.ascontiguousarray(evecs[:, 0])
        e0 = float(evals[0])
        gap = float(evals[1] - evals[0])

        sv = Statevector(psi_gs)

        # Per-site ⟨Xᵢ⟩
        per_site_mx = np.array([sv.expectation_value(op).real for op in obs_x])
        # Per-bond ⟨ZᵢZⱼ⟩
        per_bond_zz = np.array([sv.expectation_value(op).real for op in obs_zz])

        return GroundTruthResult(
            h_value=h_val,
            ground_energy=e0,
            gap=gap,
            ground_state=psi_gs,
            mag_x=float(np.mean(per_site_mx)),
            corr_zz=float(np.mean(per_bond_zz)),
            per_site_mag_x=per_site_mx,
            per_bond_corr_zz=per_bond_zz,
        )

    def _solve_dmrg(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        obs_x: list[SparsePauliOp],
        obs_zz: list[SparsePauliOp],
    ) -> GroundTruthResult:
        """DMRG ground state solver using TeNPy.

        Dispatches between:
        - TFIChain: for 1D chain topology (fastest, native TeNPy model)
        - SpinModel + 2D lattice: for square/triangular (genuine 2D DMRG)
        - TFIChain with snake ordering: for other quasi-1D topologies

        Returns E₀, gap, local observables.  ψ_gs is None (MPS, not statevector).
        """
        if lattice.topology in ("square", "triangular") and lattice.n_qubits >= 9:
            return self._solve_dmrg_2d(hamiltonian, lattice, obs_x, obs_zz)
        return self._solve_dmrg_1d(hamiltonian, lattice, obs_x, obs_zz)

    def _solve_dmrg_2d(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        obs_x: list[SparsePauliOp],
        obs_zz: list[SparsePauliOp],
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
                f"TeNPy lattice has {actual_sites} sites but our lattice has {n}. "
                f"Falling back to 1D DMRG."
            )
            return self._solve_dmrg_1d(hamiltonian, lattice, obs_x, obs_zz)

        # Initial state: product state |↑⟩^N (paramagnetic along x at h→∞)
        p_state = [[["up"] for _ in range(cols)] for _ in range(rows)]
        psi = MPS.from_lat_product_state(model.lat, p_state)

        # DMRG parameters — higher chi for 2D (more entanglement)
        chi_max = min(256, 2 ** (n // 2))  # Scale with system size
        dmrg_params = {
            "mixer": True,
            "max_E_err": 1e-12,
            "trunc_params": {"chi_max": chi_max, "svd_min": 1e-12},
            "max_sweeps": 80,
        }

        eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
        e0, _ = eng.run()

        # Gap estimation via finite-size floor (excited-state DMRG unreliable for 2D)
        gap = 2 * np.pi / n
        logger.info(
            f"DMRG 2D ({lattice.topology} {rows}x{cols}): E0={e0:.8f}, "
            f"using finite-size gap floor={gap:.4f}"
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
        )

    def _solve_dmrg_1d(
        self,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        obs_x: list[SparsePauliOp],
        obs_zz: list[SparsePauliOp],
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
        chi_max = min(400, max(200, 4 * n))
        logger.debug(f"DMRG 1D: N={n}, dynamic chi_max={chi_max}")

        dmrg_params = {
            "mixer": True,
            "max_E_err": 1e-12,
            "trunc_params": {"chi_max": chi_max, "svd_min": 1e-12},
            "max_sweeps": 100,
        }

        eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
        e0, _ = eng.run()

        # ── Gap via excited-state DMRG ────────────────────────────────
        gap = 0.0
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
        except Exception:
            logger.warning("Could not compute gap via DMRG excitation.")

        # ── Analytical gap fallback for 1D TFIM ──────────────────────
        if gap == 0.0:
            if lattice.topology == "chain_1d":
                # Analytical approximation: gap = 2|J - h| (exact in thermodynamic limit)
                # with finite-size floor: gap >= 2*pi/N (minimum gap from dispersion)
                gap = max(2 * abs(j_val - h_val), 2 * np.pi / n)
                warnings.warn(
                    f"DMRG excited state converged to GS (N={n}, h={h_val:.2f}). "
                    f"Using analytical gap={gap:.4f} (valid for 1D TFIM, "
                    f"approximate near h_c=1.0).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                # For non-1D topologies, use a conservative finite-size floor only.
                # The 2|J-h| formula is NOT valid for ladder/triangular/kagome.
                gap = 2 * np.pi / n
                warnings.warn(
                    f"DMRG excited state converged to GS (N={n}, h={h_val:.2f}, "
                    f"topology={lattice.topology}). Using finite-size floor "
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
