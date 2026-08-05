"""Multi-N Data Aggregator — Combine cross-N results for model improvement.

Scans all available experiment results across different N values and
aggregates them into a unified training dataset. Used to retrain the
UnifiedMPNN with data from multiple system sizes for better cross-N
generalization.

The aggregator:
1. Scans results/experiments/ for bond-resolved runs at different N
2. Extracts (h, θ_opt, e_exact, fidelity) per point
3. Filters by quality (ΔE/gap < threshold)
4. Builds a combined PyG dataset for UnifiedMPNN training

Usage:
    from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

    agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
    dataset = agg.build_combined_dataset(min_fidelity=0.90)
    # → PyG dataset with graphs from N=6, 10, 20 (whatever is available)

    # Retrain with combined data
    from qmbp_simulation.predictors.unified_mpnn import train_unified_mpnn
    train_unified_mpnn(model, dataset, n_epochs=3000)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_DIR = _PROJECT_ROOT / "results" / "experiments"


class MultiNAggregator:
    """Aggregates bond-resolved VQE data across multiple N values.

    Scans available results and builds a combined training dataset
    suitable for UnifiedMPNN retraining with multi-N data.

    Parameters
    ----------
    topology : str
        Lattice topology (must be same across all N).
    model : str
        Hamiltonian model name.
    results_dir : Path | None
        Results directory to scan. Default: results/experiments/.
    """

    def __init__(
        self,
        topology: str = "chain_1d",
        model: str = "tfim_bond_resolved",
        results_dir: Path | None = None,
    ) -> None:
        self.topology = topology
        self.model = model
        self._results_dir = results_dir or _RESULTS_DIR
        self._data_by_n: dict[int, list[dict[str, Any]]] = {}

    def scan(self) -> dict[int, int]:
        """Scan results for available N values and per-point data.

        Scans two sources:
        1. data/multi_n_training/*.npz — saved by AcceleratedCrossNRunner
        2. results/experiments/ — JSON results with per-point data

        Returns dict mapping N → number of usable data points found.
        """
        self._data_by_n = {}

        # Source 1: NPZ files in data/multi_n_training/ (primary, high quality)
        npz_dir = _PROJECT_ROOT / "data" / "multi_n_training"
        if npz_dir.exists():
            for npz_file in sorted(npz_dir.glob(f"{self.topology}_N*_p1.npz")):
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    h_values = data["h_values"]
                    theta_opt = data["theta_opt"]
                    e_exact = data["e_exact"]

                    # Extract N from filename: topology_N10_p1.npz
                    # NOTE: must parse N BEFORE any GroundTruthCache lookup that
                    # needs `n`. Previous code used `n` in the cache lookup block
                    # above before assigning it here (NameError in runtime).
                    fname = npz_file.stem
                    n_str = fname.split("_N")[1].split("_")[0]
                    n = int(n_str)

                    # Compute de_gaps on-the-fly if missing from NPZ
                    if "de_gaps" in data:
                        de_gaps = data["de_gaps"]
                    else:
                        # Fallback: compute from e_vqe/energies and e_exact + gaps
                        e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
                        gaps_arr = data["gaps"] if "gaps" in data else None

                        if e_key and gaps_arr is not None:
                            e_vqe = data[e_key]
                            de_gaps = np.abs(e_vqe - e_exact) / np.maximum(gaps_arr, 1e-10)
                        elif e_key:
                            # No gaps in NPZ: try GroundTruthCache lookup
                            # n is now defined above, safe to use here
                            e_vqe = data[e_key]
                            try:
                                from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache
                                gt_cache = GroundTruthCache()
                                gaps_from_cache = []
                                for h_val in h_values:
                                    cached = gt_cache.get(self.topology, n, self.model, float(h_val))
                                    gaps_from_cache.append(
                                        cached["gap"] if cached else 0.0
                                    )
                                gaps_from_cache = np.array(gaps_from_cache)
                                if np.any(gaps_from_cache > 0):
                                    de_gaps = np.abs(e_vqe - e_exact) / np.maximum(gaps_from_cache, 1e-10)
                                    logger.debug(
                                        "  MultiNAggregator: %s gaps from GroundTruthCache (%d/%d found)",
                                        npz_file.name, int(np.sum(gaps_from_cache > 0)), len(gaps_from_cache),
                                    )
                                else:
                                    # No gaps available anywhere: use absolute error as proxy
                                    # For TFIM h>2, gap ≈ 2h - 2J ≈ 2*(h-1). Conservative: assume gap=1.
                                    de_gaps = np.abs(e_vqe - e_exact)
                                    logger.debug(
                                        "  MultiNAggregator: %s no gaps found, using |ΔE| as proxy",
                                        npz_file.name,
                                    )
                            except Exception:
                                # GroundTruthCache unavailable: use absolute error
                                de_gaps = np.abs(e_vqe - e_exact)
                        else:
                            de_gaps = np.zeros(len(h_values))

                    points = []
                    for i in range(len(h_values)):
                        points.append({
                            "h": float(h_values[i]),
                            "theta": theta_opt[i],
                            "e_exact": float(e_exact[i]),
                            "de_gap": float(de_gaps[i]) if i < len(de_gaps) else 0.0,
                            "n_qubits": n,
                            "source": "npz",
                        })

                    if n not in self._data_by_n:
                        self._data_by_n[n] = []
                    self._data_by_n[n].extend(points)
                    logger.info(f"  NPZ: {npz_file.name} -> N={n}, {len(points)} points")
                except Exception as e:
                    logger.debug(f"  NPZ load failed: {npz_file.name}: {e}")

        # Source 2: ResultIndex JSON files (fallback if no NPZ found)
        if not self._data_by_n:
            self._scan_json_results()

        summary = {n: len(pts) for n, pts in self._data_by_n.items()}
        logger.info(
            "MultiNAggregator: scanned %d N values, %d total points",
            len(summary), sum(summary.values()),
        )
        return summary

    def _scan_json_results(self) -> None:
        """Fallback scan from JSON result files."""
        from qmbp_simulation.framework.result_index import ResultIndex

        try:
            idx = ResultIndex()
            runs = idx.query(model=self.model, topology=self.topology)
        except Exception:
            return

        for run in runs:
            n = run.get("n_qubits", 0)
            p = run.get("p_layers", 0)
            if n == 0 or p != 1:
                continue
            file_path = run.get("_file", "")
            if not file_path:
                continue
            full_path = idx.root / file_path
            if not full_path.exists():
                continue
            try:
                with open(full_path) as f:
                    data = json.load(f)
                points = self._extract_per_point(data, n)
                if points:
                    if n not in self._data_by_n:
                        self._data_by_n[n] = []
                    self._data_by_n[n].extend(points)
            except (json.JSONDecodeError, OSError, KeyError):
                continue

    def _extract_per_point(self, data: dict, n: int) -> list[dict]:
        """Extract per-h-point data from a result JSON."""
        points = []
        results = data.get("results", {})

        # Try different section structures
        for key, section in results.items():
            if not isinstance(section, dict):
                continue
            section_data = section.get("data", {})

            # Look for per_point data with theta/energy
            per_point = section_data.get("per_point", [])
            for pt in per_point:
                if not isinstance(pt, dict):
                    continue
                h = pt.get("h")
                theta = pt.get("theta_opt") or pt.get("theta_pred")
                e_exact = pt.get("e_exact") or pt.get("energy_exact")
                de_gap = pt.get("de_gap", 1.0)

                if h is not None and theta is not None and e_exact is not None:
                    points.append({
                        "h": float(h),
                        "theta": np.array(theta),
                        "e_exact": float(e_exact),
                        "de_gap": float(de_gap),
                        "n_qubits": n,
                    })

        return points

    def build_combined_dataset(
        self,
        max_de_gap: float = 0.10,
        min_n_values: int = 1,
    ) -> list:
        """Build a PyG dataset from aggregated multi-N data.

        Parameters
        ----------
        max_de_gap : float
            Only include points with ΔE/gap below this (quality filter).
        min_n_values : int
            Minimum number of distinct N values required.

        Returns
        -------
        list[Data]
            PyG graph dataset for UnifiedMPNN training.
        """
        if not self._data_by_n:
            self.scan()

        if len(self._data_by_n) < min_n_values:
            logger.warning(
                "Only %d N values available (need %d). Dataset may be insufficient.",
                len(self._data_by_n), min_n_values,
            )

        from qmbp_simulation import make_lattice
        from qmbp_simulation.analysis.metrics import is_point_failure
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

        import gc as _gc
        import torch

        # Disable GC during batch graph construction: creating many PyTorch
        # Data objects + Qiskit HamiltonianBuilder calls can trigger the
        # mimalloc GC deadlock on macOS ARM64 (same root cause as VQE freeze).
        _gc_was_enabled = _gc.isenabled()
        _gc.disable()

        try:
            dataset = self._build_dataset_inner(
                make_lattice, is_point_failure, build_unified_bond_resolved_graph,
                torch, max_de_gap,
            )
        finally:
            if _gc_was_enabled:
                _gc.enable()

        logger.info(f"Combined dataset: {len(dataset)} total training graphs")
        return dataset

    def _build_dataset_inner(self, make_lattice, is_point_failure,
                             build_unified_bond_resolved_graph, torch, max_de_gap):
        """Inner loop for building dataset (runs with GC disabled)."""
        dataset = []
        for n, points in sorted(self._data_by_n.items()):
            # Quality filter: use dual criterion (ΔE/gap + |ΔE| + fidelity)
            filtered = [
                p for p in points
                if not is_point_failure(
                    de_gap=p["de_gap"],
                    abs_error=p.get("abs_error"),
                    fidelity=p.get("fidelity"),
                    de_gap_threshold=max_de_gap,
                )
            ]
            if not filtered:
                continue

            lattice = make_lattice(self.topology, n, J=1.0, h=2.0)

            for pt in filtered:
                g = build_unified_bond_resolved_graph(
                    lattice, h_value=pt["h"], p_layers=1,
                    include_circuit_nodes=True,
                )
                g.y = torch.tensor(pt["theta"], dtype=torch.float32)
                dataset.append(g)

            logger.info(
                f"  N={n}: {len(filtered)}/{len(points)} points passed quality filter"
            )

        return dataset

    def available_n_values(self) -> list[int]:
        """Return sorted list of N values with available data."""
        if not self._data_by_n:
            self.scan()
        return sorted(self._data_by_n.keys())

    def summary(self) -> dict[str, Any]:
        """Return aggregation summary."""
        if not self._data_by_n:
            self.scan()
        return {
            "topology": self.topology,
            "model": self.model,
            "n_values": sorted(self._data_by_n.keys()),
            "points_per_n": {n: len(pts) for n, pts in self._data_by_n.items()},
            "total_points": sum(len(pts) for pts in self._data_by_n.values()),
        }
