"""VQEzy Dataset Loader — Load and filter VQEzy HDF5 instances.

VQEzy (Zhang et al., 2025, arXiv:2509.17322) provides 12,110 pre-computed
VQE instances across 3 domains. We focus on:

- **2D TFI** (ti_8_qubit.h5): 2D Transverse Field Ising on 4×2 grid, 8 qubits,
  1000 instances with random (j, h) ∈ [0, 5].
- **1D XYZ** (xyz_4_qubit.h5, xyz_12_qubit.h5): Heisenberg XYZ chain,
  4/12 qubits, 2000 instances with random (J1, J2, J3) ∈ [-3, 3].

Each HDF5 file stores instances as groups with keys:
- coupling constants (j, h for TFI; J array for XYZ)
- n_qubits, n_cells (grid dimensions)
- loss_history (optimization trajectory)
- best_params (optimal parameters for their CZRXRY ansatz)

Our benchmark approach: we reconstruct the SAME Hamiltonian using our
HamiltonianBuilder, then evaluate whether our MPNN (trained on our Phase 2
data) can predict good θ for our HVA ansatz.

Usage:
    from qmbp_simulation.predictors.external_benchmarks import (
        load_vqezy_tfi, VQEzyDataset
    )

    dataset = load_vqezy_tfi("path/to/VQEzy/qmanybody/ti_8_qubit.h5")
    print(f"Loaded {len(dataset)} TFI instances")
    for inst in dataset[:5]:
        print(f"  j={inst.j:.2f}, h={inst.h:.2f}, E_opt={inst.e_optimal:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

    from qmbp_simulation.models.data_models import LatticeConfig

logger = logging.getLogger(__name__)


@dataclass
class VQEzyInstance:
    """A single VQEzy instance with Hamiltonian parameters and VQE results.

    Attributes
    ----------
    instance_id : str
        Unique identifier within the dataset.
    n_qubits : int
        Number of qubits.
    model : str
        Model type ('tfi' or 'xyz').
    h : float
        Transverse field strength (TFI) or external field (XYZ).
    j : float
        Coupling constant (TFI). For XYZ, this is J1.
    coupling_params : dict
        Full coupling parameters (model-specific).
    grid_shape : tuple[int, ...] | None
        Grid dimensions for 2D models (e.g., (4, 2) for 4×2 TFI).
    e_optimal : float
        Best energy found by VQEzy's optimizer (last entry of loss_history).
    loss_history : np.ndarray
        Full optimization trajectory (energy vs iteration).
    n_vqe_iterations : int
        Number of VQE iterations used by VQEzy.
    best_params : np.ndarray
        Optimal parameters for VQEzy's ansatz (CZRXRY).
    """

    instance_id: str
    n_qubits: int
    model: str
    h: float
    j: float
    coupling_params: dict = field(default_factory=dict)
    grid_shape: tuple[int, ...] | None = None
    e_optimal: float = 0.0
    loss_history: np.ndarray = field(default_factory=lambda: np.array([]))
    n_vqe_iterations: int = 0
    best_params: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class VQEzyDataset:
    """Collection of VQEzy instances with metadata.

    Attributes
    ----------
    instances : list[VQEzyInstance]
        All loaded instances.
    model : str
        Model type ('tfi' or 'xyz').
    source_file : str
        Path to the source HDF5 file.
    n_qubits : int
        Number of qubits (uniform within a file).
    """

    instances: list[VQEzyInstance]
    model: str
    source_file: str
    n_qubits: int

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def filter_h_range(self, h_min: float, h_max: float) -> "VQEzyDataset":
        """Filter instances to h ∈ [h_min, h_max]."""
        filtered = [inst for inst in self.instances if h_min <= inst.h <= h_max]
        return VQEzyDataset(
            instances=filtered,
            model=self.model,
            source_file=self.source_file,
            n_qubits=self.n_qubits,
        )

    def filter_j_range(self, j_min: float, j_max: float) -> "VQEzyDataset":
        """Filter instances to j ∈ [j_min, j_max]."""
        filtered = [inst for inst in self.instances if j_min <= inst.j <= j_max]
        return VQEzyDataset(
            instances=filtered,
            model=self.model,
            source_file=self.source_file,
            n_qubits=self.n_qubits,
        )

    def get_h_values(self) -> np.ndarray:
        """Get all h-values as sorted array."""
        return np.sort(np.array([inst.h for inst in self.instances]))

    def get_energies(self) -> np.ndarray:
        """Get all optimal energies."""
        return np.array([inst.e_optimal for inst in self.instances])

    def filter_converged(self, tail_fraction: float = 0.1) -> "VQEzyDataset":
        """Filter out instances where VQEzy optimization didn't converge.

        Checks the tail of the loss_history: if it's still decreasing
        significantly, the instance likely didn't converge.

        Parameters
        ----------
        tail_fraction : float
            Fraction of trajectory tail to check (default 0.1 = last 10%).

        Returns
        -------
        VQEzyDataset
            Filtered dataset with only converged instances.
        """
        filtered = []
        for inst in self.instances:
            if len(inst.loss_history) < 10:
                filtered.append(inst)
                continue
            tail_start = int(len(inst.loss_history) * (1 - tail_fraction))
            tail = inst.loss_history[tail_start:]
            # Check if energy is still decreasing significantly in the tail
            # A "converged" instance should have relative change < 1% in tail
            if len(tail) >= 2:
                relative_change = abs(tail[-1] - tail[0]) / (abs(tail[0]) + 1e-10)
                if relative_change < 0.01:
                    filtered.append(inst)
            else:
                filtered.append(inst)

        n_removed = len(self.instances) - len(filtered)
        if n_removed > 0:
            logger.info(
                f"filter_converged: removed {n_removed}/{len(self.instances)} "
                f"unconverged instances"
            )

        return VQEzyDataset(
            instances=filtered,
            model=self.model,
            source_file=self.source_file,
            n_qubits=self.n_qubits,
        )

    def summary(self) -> str:
        """Print dataset summary."""
        h_vals = np.array([inst.h for inst in self.instances])
        j_vals = np.array([inst.j for inst in self.instances])
        e_vals = np.array([inst.e_optimal for inst in self.instances])
        return (
            f"VQEzyDataset({self.model}, N={self.n_qubits}, "
            f"n_instances={len(self)}, "
            f"h∈[{h_vals.min():.2f}, {h_vals.max():.2f}], "
            f"j∈[{j_vals.min():.2f}, {j_vals.max():.2f}], "
            f"E∈[{e_vals.min():.4f}, {e_vals.max():.4f}])"
        )



def _require_h5py():
    """Lazy import h5py with helpful error message."""
    try:
        import h5py
        return h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for VQEzy dataset loading. "
            "Install with: pip install h5py"
        ) from e


def load_vqezy_tfi(
    path: str | Path,
    *,
    h_min: float | None = None,
    h_max: float | None = None,
    j_min: float | None = None,
    j_max: float | None = None,
    max_instances: int | None = None,
) -> VQEzyDataset:
    """Load VQEzy 2D Transverse Field Ising instances from HDF5.

    VQEzy's TFI Hamiltonian is:
        H = -j * Σ_{⟨i,j⟩} Z_i Z_j - μ * Σ_i X_i

    where the sum runs over nearest neighbors on a 2D rectangular grid.
    Their 'h' parameter corresponds to μ (transverse field).
    Their 'j' parameter is the ZZ coupling.

    HDF5 structure:
        j_h/sample_X          → [j, h] (float64, shape (2,))
        loss_history/sample_X  → optimization trajectory (float64, shape (2000,))
        n_cells/sample_X       → [rows, cols] grid dims (int64, shape (2,))
        n_qubits/sample_X      → scalar int
        opt_params/sample_X    → [1, 2, N_qubits] CZRXRY params (float32)

    Parameters
    ----------
    path : str | Path
        Path to HDF5 file (e.g., 'VQEzy/qmanybody/ti_8_qubit.h5').
    h_min, h_max : float | None
        Filter transverse field range.
    j_min, j_max : float | None
        Filter coupling range.
    max_instances : int | None
        Maximum number of instances to load (for quick testing).

    Returns
    -------
    VQEzyDataset
        Loaded and optionally filtered dataset.
    """
    h5py = _require_h5py()
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"VQEzy HDF5 file not found: {path}\n"
            f"Clone the dataset: git clone https://github.com/chizhang24/VQEzy.git"
        )

    instances: list[VQEzyInstance] = []
    n_qubits = 0

    with h5py.File(path, "r") as f:
        # VQEzy stores data in top-level groups by field type
        j_h_grp = f["j_h"]
        loss_grp = f["loss_history"]
        n_cells_grp = f["n_cells"]
        n_qubits_grp = f["n_qubits"]
        opt_params_grp = f["opt_params"]

        keys = sorted(j_h_grp.keys(), key=lambda x: int(x.split("_")[1]))
        if max_instances is not None:
            keys = keys[:max_instances]

        for key in keys:
            j_h_val = np.array(j_h_grp[key])  # [j, h]
            inst_j = float(j_h_val[0])
            inst_h = float(j_h_val[1])

            # Apply filters early (skip loading heavy data)
            if h_min is not None and inst_h < h_min:
                continue
            if h_max is not None and inst_h > h_max:
                continue
            if j_min is not None and inst_j < j_min:
                continue
            if j_max is not None and inst_j > j_max:
                continue

            n_cells = np.array(n_cells_grp[key])
            inst_n_qubits = int(np.array(n_qubits_grp[key]))
            loss_history = np.array(loss_grp[key])
            best_params = np.array(opt_params_grp[key])

            n_qubits = inst_n_qubits

            # Best energy is the last entry in loss history
            e_optimal = float(loss_history[-1])

            instances.append(VQEzyInstance(
                instance_id=key,
                n_qubits=inst_n_qubits,
                model="tfi",
                h=inst_h,
                j=inst_j,
                coupling_params={"j": inst_j, "h": inst_h},
                grid_shape=tuple(n_cells.tolist()),
                e_optimal=e_optimal,
                loss_history=loss_history,
                n_vqe_iterations=len(loss_history),
                best_params=best_params,
            ))

    logger.info(
        f"Loaded {len(instances)} TFI instances from {path.name} "
        f"(N={n_qubits}, filters: h∈[{h_min}, {h_max}], j∈[{j_min}, {j_max}])"
    )

    return VQEzyDataset(
        instances=instances,
        model="tfi",
        source_file=str(path),
        n_qubits=n_qubits,
    )



def load_vqezy_xyz(
    path: str | Path,
    *,
    max_instances: int | None = None,
) -> VQEzyDataset:
    """Load VQEzy Heisenberg XYZ instances from HDF5.

    VQEzy's XYZ Hamiltonian is:
        H = Σ_i (J1·X_iX_{i+1} + J2·Y_iY_{i+1} + J3·Z_iZ_{i+1})

    HDF5 structure:
        coupling_const/sample_X → [J1, J2, J3] (float64, shape (3,))
        loss_history/sample_X   → optimization trajectory (float64, shape (2000,))
        n_qubits/sample_X       → scalar int
        opt_params/sample_X     → [1, 2, N_qubits] CZRXRY params (float32)

    Parameters
    ----------
    path : str | Path
        Path to HDF5 file (e.g., 'VQEzy/qmanybody/xyz_4_qubit.h5').
    max_instances : int | None
        Maximum number of instances to load.

    Returns
    -------
    VQEzyDataset
        Loaded dataset with Heisenberg XYZ instances.
    """
    h5py = _require_h5py()
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"VQEzy HDF5 file not found: {path}\n"
            f"Clone the dataset: git clone https://github.com/chizhang24/VQEzy.git"
        )

    instances: list[VQEzyInstance] = []
    n_qubits = 0

    with h5py.File(path, "r") as f:
        coupling_grp = f["coupling_const"]
        loss_grp = f["loss_history"]
        n_qubits_grp = f["n_qubits"]
        opt_params_grp = f["opt_params"]

        keys = sorted(coupling_grp.keys(), key=lambda x: int(x.split("_")[1]))
        if max_instances is not None:
            keys = keys[:max_instances]

        for key in keys:
            J = np.array(coupling_grp[key])  # [J1, J2, J3]
            inst_n_qubits = int(np.array(n_qubits_grp[key]))
            loss_history = np.array(loss_grp[key])
            best_params = np.array(opt_params_grp[key])

            n_qubits = inst_n_qubits
            e_optimal = float(loss_history[-1])

            # For XYZ: j = J1 (XX coupling), h = 0 (no external field)
            j_val = float(J[0]) if len(J) > 0 else 0.0

            instances.append(VQEzyInstance(
                instance_id=key,
                n_qubits=inst_n_qubits,
                model="xyz",
                h=0.0,  # No external field in XYZ
                j=j_val,
                coupling_params={"J1": float(J[0]), "J2": float(J[1]), "J3": float(J[2])},
                grid_shape=None,  # 1D chain
                e_optimal=e_optimal,
                loss_history=loss_history,
                n_vqe_iterations=len(loss_history),
                best_params=best_params,
            ))

    logger.info(
        f"Loaded {len(instances)} XYZ instances from {path.name} (N={n_qubits})"
    )

    return VQEzyDataset(
        instances=instances,
        model="xyz",
        source_file=str(path),
        n_qubits=n_qubits,
    )


def reconstruct_tfi_hamiltonian(instance: VQEzyInstance) -> tuple[SparsePauliOp, LatticeConfig]:
    """Reconstruct the TFI Hamiltonian using our HamiltonianBuilder.

    VQEzy's 2D TFI uses a rectangular grid with ZZ coupling on nearest
    neighbors and X-field on all sites. We map this to our framework's
    HamiltonianBuilder which supports arbitrary lattice topologies.

    For VQEzy's 4×2 grid (8 qubits), our generate_square(8) produces a
    3×3 grid (different topology). We construct LatticeConfig directly
    with the correct edges matching VQEzy's PennyLane convention.

    Parameters
    ----------
    instance : VQEzyInstance
        A TFI instance from VQEzy.

    Returns
    -------
    tuple[SparsePauliOp, LatticeConfig]
        (hamiltonian, lattice) — reconstructed using our framework.
    """
    from qmbp_simulation import HamiltonianBuilder
    from qmbp_simulation.models.data_models import LatticeConfig

    if instance.model != "tfi":
        raise ValueError(f"Expected TFI instance, got model={instance.model}")

    n_qubits = instance.n_qubits
    j = instance.j
    h = instance.h

    # Build edges matching VQEzy's rectangular grid convention
    if instance.grid_shape is not None:
        rows, cols = instance.grid_shape[0], instance.grid_shape[1]
        edges = _build_rectangular_edges(rows, cols)
    else:
        # Fallback: chain_1d
        edges = [(i, i + 1) for i in range(n_qubits - 1)]

    # Compute coordination numbers from edges
    coord = np.zeros(n_qubits)
    for i, j_site in edges:
        coord[i] += 1
        coord[j_site] += 1

    lattice = LatticeConfig(
        topology="square",
        n_qubits=n_qubits,
        J=j,
        h=h,
        edges=edges,
        coordination_numbers=coord,
        periodic=False,
    )

    builder = HamiltonianBuilder()
    H = builder.build(lattice)
    return H, lattice


def _build_rectangular_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    """Build nearest-neighbor edges for a rectangular grid.

    Qubit indexing: row-major (qubit i = row * cols + col).
    Matching PennyLane's `qml.spin.transverse_ising('rectangle', [rows, cols])`.

    Parameters
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.

    Returns
    -------
    list[tuple[int, int]]
        List of (i, j) edges for nearest neighbors.
    """
    edges = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            # Right neighbor
            if c + 1 < cols:
                edges.append((idx, idx + 1))
            # Down neighbor
            if r + 1 < rows:
                edges.append((idx, idx + cols))
    return edges


def verify_hamiltonian_equivalence(
    instance: VQEzyInstance,
    *,
    atol: float = 1e-6,
) -> dict:
    """Verify our reconstructed Hamiltonian matches VQEzy's energy.

    Checks: given VQEzy's optimal energy E_opt, our exact diagonalization
    should produce E_exact ≤ E_opt (since E_opt is variational, not exact).
    Also verifies the energy scale is consistent (not off by a factor).

    Parameters
    ----------
    instance : VQEzyInstance
        A VQEzy instance to verify.
    atol : float
        Absolute tolerance for energy comparison.

    Returns
    -------
    dict
        Verification report with keys: consistent, e_exact, e_vqezy,
        delta, message.
    """
    from qmbp_simulation import ClassicalSolver

    H, lattice = reconstruct_tfi_hamiltonian(instance)
    solver = ClassicalSolver()
    gt = solver.solve(H, lattice)
    e_exact = gt.ground_energy
    e_vqezy = instance.e_optimal
    delta = e_vqezy - e_exact

    # VQEzy's energy should be ≥ E_exact (variational principle)
    # Allow small tolerance for numerics
    consistent = delta >= -atol

    # Also check energy scale consistency (not off by factor of N or 2)
    scale_ok = True
    if abs(e_exact) > 0.1:
        ratio = abs(e_vqezy / e_exact)
        scale_ok = 0.5 < ratio < 2.0

    if not consistent:
        message = (
            f"INCONSISTENCY: E_vqezy={e_vqezy:.6f} < E_exact={e_exact:.6f} "
            f"(Δ={delta:.2e}). VQEzy should be variational (≥ E_exact). "
            f"Possible sign convention mismatch in Hamiltonian reconstruction."
        )
    elif not scale_ok:
        message = (
            f"SCALE MISMATCH: E_vqezy={e_vqezy:.6f}, E_exact={e_exact:.6f}, "
            f"ratio={abs(e_vqezy / e_exact):.2f}. Energy scale differs by >2×."
        )
    else:
        message = (
            f"OK: E_exact={e_exact:.6f}, E_vqezy={e_vqezy:.6f}, "
            f"ΔE/gap={(delta / gt.gap):.4f}" if gt.gap > 1e-10
            else f"OK: E_exact={e_exact:.6f}, E_vqezy={e_vqezy:.6f}"
        )

    return {
        "consistent": consistent and scale_ok,
        "e_exact": e_exact,
        "e_vqezy": e_vqezy,
        "delta": delta,
        "spectral_gap": gt.gap,
        "message": message,
    }
