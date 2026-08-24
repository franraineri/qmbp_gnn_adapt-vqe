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
    max_n : int | None
        If set, exclude N > max_n from training data.
    p_layers : int
        Number of HVA layers (determines NPZ file suffix and graph structure).
        Default 1 for backward compatibility.
    """

    def __init__(
        self,
        topology: str = "chain_1d",
        model: str = "tfim_bond_resolved",
        results_dir: Path | None = None,
        max_n: int | None = None,
        p_layers: int = 1,
    ) -> None:
        self.topology = topology
        self.model = model
        self.p_layers = p_layers
        self._results_dir = results_dir or _RESULTS_DIR
        self._data_by_n: dict[int, list[dict[str, Any]]] = {}
        self.max_n = max_n  # If set, exclude N > max_n from training data

        if p_layers < 1:
            raise ValueError(f"p_layers must be >= 1, got {p_layers}")

    def scan(self) -> dict[int, int]:
        """Scan results for available N values and per-point data.

        Scans two sources:
        1. data/multi_n_training/*.npz — saved by AcceleratedCrossNRunner
        2. results/experiments/ — JSON results with per-point data

        Skips NPZ files marked as 'not_useful' in the model quality dashboard
        (if available). This prevents contaminating the training dataset with
        data the MPNN cannot learn from.

        Returns dict mapping N → number of usable data points found.
        """
        self._data_by_n = {}

        # ── Pre-filter: load dashboard 'not_useful' + exclusion registry ─
        not_useful_files = self._load_not_useful_files()
        excluded_files = self._load_exclusion_registry()
        skip_files = not_useful_files | excluded_files

        # Source 1: NPZ files in data/multi_n_training/ (primary, high quality)
        npz_dir = _PROJECT_ROOT / "data" / "multi_n_training"
        if npz_dir.exists():
            for npz_file in sorted(npz_dir.glob(f"{self.topology}_N*_p{self.p_layers}.npz")):
                # Skip NPZ files excluded from training
                # Check both dir-qualified path and bare filename (legacy compat)
                qualified = f"multi_n_training/{npz_file.name}"
                if qualified in skip_files or npz_file.name in skip_files:
                    logger.info(
                        f"  MultiNAggregator: SKIPPING {npz_file.name} (excluded from training)"
                    )
                    continue

                try:
                    data = np.load(npz_file, allow_pickle=True)
                    h_values = np.asarray(data["h_values"], dtype=np.float64)
                    theta_opt = data["theta_opt"]
                    e_exact = np.asarray(data["e_exact"], dtype=np.float64)

                    # Extract N from filename: topology_N10_p1.npz
                    # NOTE: must parse N BEFORE any GroundTruthCache lookup that
                    # needs `n`. Previous code used `n` in the cache lookup block
                    # above before assigning it here (NameError in runtime).
                    fname = npz_file.stem
                    n_str = fname.split("_N")[1].split("_")[0]
                    n = int(n_str)

                    # Skip if beyond max_n (prevents contamination from extrapolation data)
                    if self.max_n is not None and n > self.max_n:
                        logger.info(
                            f"  MultiNAggregator: SKIPPING {npz_file.name} "
                            f"(N={n} > max_n={self.max_n})"
                        )
                        continue

                    # Compute de_gaps on-the-fly if missing from NPZ
                    if "de_gaps" in data:
                        de_gaps = np.asarray(data["de_gaps"], dtype=np.float64)
                    else:
                        # Fallback: compute from e_vqe/energies and e_exact + gaps
                        e_key = (
                            "e_vqe"
                            if "e_vqe" in data
                            else ("energies" if "energies" in data else None)
                        )
                        gaps_arr = (
                            np.asarray(data["gaps"], dtype=np.float64) if "gaps" in data else None
                        )

                        if e_key and gaps_arr is not None:
                            e_vqe = np.asarray(data[e_key], dtype=np.float64)
                            de_gaps = np.abs(e_vqe - e_exact) / np.maximum(gaps_arr, 1e-10)
                        elif e_key:
                            # No gaps in NPZ: try GroundTruthCache lookup
                            # n is now defined above, safe to use here
                            e_vqe = np.asarray(data[e_key], dtype=np.float64)
                            try:
                                from qmbp_simulation.solvers.ground_truth_cache import (
                                    GroundTruthCache,
                                )

                                gt_cache = GroundTruthCache()
                                gaps_from_cache = []
                                for h_val in h_values:
                                    cached = gt_cache.get(
                                        self.topology, n, self.model, float(h_val)
                                    )
                                    gaps_from_cache.append(cached["gap"] if cached else 0.0)
                                gaps_from_cache = np.array(gaps_from_cache)
                                if np.any(gaps_from_cache > 0):
                                    de_gaps = np.abs(e_vqe - e_exact) / np.maximum(
                                        gaps_from_cache, 1e-10
                                    )
                                    logger.debug(
                                        "  MultiNAggregator: %s gaps from GroundTruthCache (%d/%d found)",
                                        npz_file.name,
                                        int(np.sum(gaps_from_cache > 0)),
                                        len(gaps_from_cache),
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

                    # Compute abs_error for dual criterion filtering
                    e_key = (
                        "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
                    )
                    abs_errors = None
                    if e_key is not None:
                        e_vqe_arr = np.asarray(data[e_key], dtype=np.float64)
                        abs_errors = np.abs(e_vqe_arr - e_exact)

                    # Load quality tier (backward compat: default "unverified")
                    tier_arr = data["quality_tier"].tolist() if "quality_tier" in data else None
                    method_arr = data["method"].tolist() if "method" in data else None

                    points = []
                    for i in range(len(h_values)):
                        # Ensure theta is always float64 (handles legacy dtype=object NPZs)
                        theta_i = np.asarray(theta_opt[i], dtype=np.float64)
                        pt = {
                            "h": float(h_values[i]),
                            "theta": theta_i,
                            "e_exact": float(e_exact[i]),
                            "de_gap": float(de_gaps[i]) if i < len(de_gaps) else 0.0,
                            "n_qubits": n,
                            "source": "npz",
                            "quality_tier": tier_arr[i] if tier_arr else "unverified",
                            "method": str(method_arr[i]) if method_arr else "unknown",
                        }
                        if abs_errors is not None:
                            pt["abs_error"] = float(abs_errors[i])
                        points.append(pt)

                    if n not in self._data_by_n:
                        self._data_by_n[n] = []
                    self._data_by_n[n].extend(points)
                    logger.info(f"  NPZ: {npz_file.name} -> N={n}, {len(points)} points")
                except Exception as e:
                    logger.debug(f"  NPZ load failed: {npz_file.name}: {e}")

        # Source 2: Large-N extrapolation data (approximate tier, bootstrapping cycle)
        # These are MPNN predictions that passed dual criterion but haven't been
        # VQE-verified. They're included with relaxed threshold to enable the
        # iterative improvement cycle: predict(N=30) → train → predict(N=40) → ...
        extrap_dir = _PROJECT_ROOT / "data" / "large_n_extrapolation"
        if extrap_dir.exists():
            for npz_file in sorted(extrap_dir.glob(f"{self.topology}_N*_p{self.p_layers}.npz")):
                # Check both dir-qualified path and bare filename (legacy compat)
                qualified = f"large_n_extrapolation/{npz_file.name}"
                if qualified in skip_files or npz_file.name in skip_files:
                    continue
                try:
                    data = np.load(str(npz_file), allow_pickle=True)
                    h_values = np.asarray(data["h_values"], dtype=np.float64)
                    theta_opt = data["theta_opt"]
                    e_exact = np.asarray(data["e_exact"], dtype=np.float64)

                    fname = npz_file.stem
                    n_str = fname.split("_N")[1].split("_")[0]
                    n = int(n_str)

                    # Skip if beyond max_n
                    if self.max_n is not None and n > self.max_n:
                        logger.info(
                            f"  MultiNAggregator: SKIPPING extrap {npz_file.name} "
                            f"(N={n} > max_n={self.max_n})"
                        )
                        continue

                    de_gaps = (
                        np.asarray(data["de_gaps"], dtype=np.float64)
                        if "de_gaps" in data
                        else np.zeros(len(h_values))
                    )
                    e_key = (
                        "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
                    )
                    abs_errors = (
                        np.abs(np.asarray(data[e_key], dtype=np.float64) - e_exact)
                        if e_key
                        else None
                    )

                    points = []
                    for i in range(len(h_values)):
                        theta_i = np.asarray(theta_opt[i], dtype=np.float64)
                        points.append(
                            {
                                "h": float(h_values[i]),
                                "theta": theta_i,
                                "e_exact": float(e_exact[i]),
                                "de_gap": float(de_gaps[i]) if i < len(de_gaps) else 0.0,
                                "abs_error": float(abs_errors[i])
                                if abs_errors is not None
                                else None,
                                "n_qubits": n,
                                "source": "large_n_extrapolation",
                                "quality_tier": "approximate",
                            }
                        )

                    if n not in self._data_by_n:
                        self._data_by_n[n] = []
                    self._data_by_n[n].extend(points)
                    logger.info(
                        f"  LargeN NPZ: {npz_file.name} -> N={n}, "
                        f"{len(points)} points (approximate tier)"
                    )
                except Exception as e:
                    logger.debug(f"  LargeN NPZ load failed: {npz_file.name}: {e}")

        # Source 3: ResultIndex JSON files (fallback if no NPZ found)
        if not self._data_by_n:
            self._scan_json_results()

        summary = {n: len(pts) for n, pts in self._data_by_n.items()}
        logger.info(
            "MultiNAggregator: scanned %d N values, %d total points",
            len(summary),
            sum(summary.values()),
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
            if n == 0 or p != self.p_layers:
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
                    points.append(
                        {
                            "h": float(h),
                            "theta": np.asarray(theta, dtype=np.float64),
                            "e_exact": float(e_exact),
                            "de_gap": float(de_gap),
                            "n_qubits": n,
                        }
                    )

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
                len(self._data_by_n),
                min_n_values,
            )

        import gc as _gc

        import torch

        from qmbp_simulation import make_lattice
        from qmbp_simulation.analysis.metrics import MAX_ABS_ERROR, is_point_failure
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

        # Disable GC during batch graph construction: creating many PyTorch
        # Data objects + Qiskit HamiltonianBuilder calls can trigger the
        # mimalloc GC deadlock on macOS ARM64 (same root cause as VQE freeze).
        _gc_was_enabled = _gc.isenabled()
        _gc.disable()

        try:
            dataset = self._build_dataset_inner(
                make_lattice,
                is_point_failure,
                build_unified_bond_resolved_graph,
                torch,
                max_de_gap,
                MAX_ABS_ERROR,
            )
        finally:
            if _gc_was_enabled:
                _gc.enable()

        logger.info(f"Combined dataset: {len(dataset)} total training graphs")

        # ── Consistency validation: all graphs must have same node_features dim ──
        if len(dataset) > 1:
            feat_dims = set(g.x.shape[1] for g in dataset if hasattr(g, "x"))
            if len(feat_dims) > 1:
                logger.error(
                    "MultiNAggregator: inconsistent node_features dimensions "
                    "across graphs: %s. This will crash during training. "
                    "Check that all NPZ data was generated with the same "
                    "build_unified_bond_resolved_graph version.",
                    feat_dims,
                )
                raise ValueError(
                    f"Inconsistent node_features: {feat_dims}. "
                    "Cannot mix graphs with different feature dimensions."
                )

        return dataset

    def _build_dataset_inner(
        self,
        make_lattice,
        is_point_failure,
        build_unified_bond_resolved_graph,
        torch,
        max_de_gap,
        max_abs_error,
    ):
        """Inner loop for building dataset (runs with GC disabled).

        Quality tier logic:
        - "verified" (VQE-converged): always included (trusted source)
        - "approximate" (MPNN passing dual criterion): included with relaxed threshold
        - "unverified" (legacy/unknown): strict dual criterion filter

        Data augmentation:
        - Only verified points with small datasets get Z₂ + noise augmentation
        - Augmented variants have lower sample_weight than originals
        """
        from qmbp_simulation.analysis.metrics import (
            AUGMENTATION_MAX_FILTERED_POINTS,
            AUGMENTATION_MAX_VARIANTS_PER_POINT,
            QUALITY_TIER_WEIGHT_APPROXIMATE,
            QUALITY_TIER_WEIGHT_AUGMENTED,
            QUALITY_TIER_WEIGHT_UNVERIFIED,
            QUALITY_TIER_WEIGHT_VERIFIED,
        )
        from qmbp_simulation.models.constants import AUGMENTATION_NOISE_SIGMA

        dataset = []
        for n, points in sorted(self._data_by_n.items()):
            # Tier-aware quality filter
            filtered = []
            for p in points:
                tier = p.get("quality_tier", "unverified")

                if tier == "verified":
                    # VQE-converged data is always trusted
                    filtered.append(p)
                elif tier == "approximate":
                    # MPNN predictions that passed dual criterion at persist time.
                    # Re-verify with slightly relaxed threshold (1.5× of strict)
                    if not is_point_failure(
                        de_gap=p["de_gap"],
                        abs_error=p.get("abs_error"),
                        de_gap_threshold=max_de_gap * 1.5,
                        max_abs_error=max_abs_error * 1.5,
                    ):
                        filtered.append(p)
                else:
                    # Unverified/legacy: strict dual criterion
                    if not is_point_failure(
                        de_gap=p["de_gap"],
                        abs_error=p.get("abs_error"),
                        fidelity=p.get("fidelity"),
                        de_gap_threshold=max_de_gap,
                    ):
                        filtered.append(p)

            if not filtered:
                logger.warning(
                    "MultiNAggregator: N=%d has 0 points passing quality filter "
                    "(0 verified, 0 approximate, 0 unverified pass). "
                    "Skipping — this config is not useful for MPNN training.",
                    n,
                )
                continue

            lattice = make_lattice(self.topology, n, J=1.0, h=2.0)

            # Count tiers for logging
            n_verified = sum(1 for p in filtered if p.get("quality_tier") == "verified")
            n_approx = sum(1 for p in filtered if p.get("quality_tier") == "approximate")
            n_unverified = len(filtered) - n_verified - n_approx
            n_augmented = 0

            # ── Z₂ sweep canonicalization ─────────────────────────────────
            if len(filtered) >= 3:
                from qmbp_simulation.utils.helpers import canonicalize_sweep_z2

                h_arr = np.array([p["h"] for p in filtered])
                theta_mat = np.array(
                    [np.asarray(p["theta"], dtype=np.float64) for p in filtered]
                )
                theta_canon = canonicalize_sweep_z2(theta_mat, h_arr)
                for i, pt in enumerate(filtered):
                    pt["theta"] = theta_canon[i]

            for pt in filtered:
                g = build_unified_bond_resolved_graph(
                    lattice,
                    h_value=pt["h"],
                    p_layers=self.p_layers,
                    include_circuit_nodes=True,
                )
                # Ensure theta is float before torch conversion (safety against object arrays)
                theta_arr = np.asarray(pt["theta"], dtype=np.float64)
                g.y = torch.tensor(theta_arr, dtype=torch.float32)

                # Store quality weight in graph for optional weighted training
                tier = pt.get("quality_tier", "unverified")
                weight = {
                    "verified": QUALITY_TIER_WEIGHT_VERIFIED,
                    "approximate": QUALITY_TIER_WEIGHT_APPROXIMATE,
                    "unverified": QUALITY_TIER_WEIGHT_UNVERIFIED,
                }.get(tier, QUALITY_TIER_WEIGHT_UNVERIFIED)
                g.sample_weight = torch.tensor([weight], dtype=torch.float32)

                dataset.append(g)

                # ── Data augmentation: Z₂ symmetry for verified points ────
                # Only augment verified data (high-quality VQE-converged θ).
                # Augmenting approximate/unverified would amplify noise.
                if tier == "verified" and len(filtered) < AUGMENTATION_MAX_FILTERED_POINTS:
                    try:
                        from qmbp_simulation.utils.helpers import augment_theta_symmetries

                        # Guard: only augment finite theta
                        if np.all(np.isfinite(theta_arr)) and theta_arr.size > 0:
                            # More variants for very small datasets
                            n_noise = 2 if len(filtered) < 20 else 1
                            max_variants = (
                                3 if len(filtered) < 20 else AUGMENTATION_MAX_VARIANTS_PER_POINT
                            )
                            variants = augment_theta_symmetries(
                                theta_arr,
                                include_z2=True,
                                noise_std=AUGMENTATION_NOISE_SIGMA,
                                n_noise_variants=n_noise,
                                seed=hash(pt["h"]) % 2**31,
                            )
                            for var_theta in variants[:max_variants]:
                                # Guard: verify augmented theta is finite
                                if not np.all(np.isfinite(var_theta)):
                                    continue
                                g_aug = build_unified_bond_resolved_graph(
                                    lattice,
                                    h_value=pt["h"],
                                    p_layers=self.p_layers,
                                    include_circuit_nodes=True,
                                )
                                g_aug.y = torch.tensor(
                                    var_theta.astype(np.float32), dtype=torch.float32
                                )
                                g_aug.sample_weight = torch.tensor(
                                    [QUALITY_TIER_WEIGHT_AUGMENTED], dtype=torch.float32
                                )
                                dataset.append(g_aug)
                                n_augmented += 1
                    except Exception as e:
                        # Augmentation failure is non-fatal — continue without it
                        logger.debug(f"  Augmentation failed for h={pt['h']:.3f}: {e}")

            logger.info(
                f"  N={n}: {len(filtered)}/{len(points)} points pass "
                f"(verified={n_verified}, approx={n_approx}, legacy={n_unverified})"
                f"{f', +{n_augmented} augmented' if n_augmented > 0 else ''}"
            )

        return dataset

    def available_n_values(self) -> list[int]:
        """Return sorted list of N values with available data."""
        if not self._data_by_n:
            self.scan()
        return sorted(self._data_by_n.keys())

    def _load_not_useful_files(self) -> set[str]:
        """Load NPZ filenames classified as 'not_useful' from the dashboard.

        Reads `data/model_quality_dashboard.json` and returns the set of
        NPZ filenames that have training_utility='not_useful'.

        Returns bare filenames (dashboard applies to multi_n_training/ only).
        The scan() method checks both bare filenames and dir-qualified paths
        to handle the exclusion registry format consistently.

        Returns empty set if dashboard doesn't exist or has no utility field.
        """
        dashboard_path = _PROJECT_ROOT / "data" / "model_quality_dashboard.json"
        if not dashboard_path.exists():
            return set()
        try:
            import json

            with open(dashboard_path) as f:
                dashboard = json.load(f)
            # Dashboard filenames are bare (no dir prefix) and only refer to
            # multi_n_training/ files. Return both bare and qualified versions
            # for consistent matching in scan().
            result = set()
            for c in dashboard.get("configs", []):
                if c.get("training_utility") == "not_useful":
                    fname = c["file"]
                    result.add(fname)
                    result.add(f"multi_n_training/{fname}")
            return result
        except (json.JSONDecodeError, OSError, KeyError):
            return set()

    def _load_exclusion_registry(self) -> set[str]:
        """Load NPZ filenames from the persistent exclusion registry.

        Reads `data/training_exclusions.json` — the persistent record of
        NPZ files excluded from training (both auto-detected and manual).

        Returns empty set if registry doesn't exist.
        """
        try:
            from qmbp_simulation.analysis.metrics import get_excluded_files

            return get_excluded_files()
        except (ImportError, Exception):
            return set()

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


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Topology Aggregator
# ═══════════════════════════════════════════════════════════════════════════════

# Default topologies to include (those with sufficient verified data)
MULTI_TOPOLOGY_DEFAULTS = ["chain_1d", "heavy_hex", "ladder", "square", "triangular"]


class MultiTopologyAggregator:
    """Aggregates bond-resolved data across ALL topologies for universal model training.

    Wraps MultiNAggregator per-topology, combines into a single dataset.
    The resulting model can predict θ for any topology it was trained on,
    and potentially generalize to unseen topologies via shared GNN representations.

    Parameters
    ----------
    topologies : list[str] | None
        Topologies to include. None = all available in data/multi_n_training/.
    model : str
        Hamiltonian model name (default: tfim_bond_resolved).
    max_n : int | None
        Maximum N to include per topology.
    min_verified_points : int
        Minimum verified points per topology to include it (quality gate).
    p_layers : int
        Number of HVA layers. Default 1.
    """

    def __init__(
        self,
        topologies: list[str] | None = None,
        model: str = "tfim_bond_resolved",
        max_n: int | None = None,
        min_verified_points: int = 10,
        p_layers: int = 1,
    ) -> None:
        self.model = model
        self.max_n = max_n
        self.min_verified_points = min_verified_points
        self.p_layers = p_layers
        self._topologies = topologies
        self._aggregators: dict[str, MultiNAggregator] = {}
        self._summary: dict[str, dict] = {}

        if p_layers < 1:
            raise ValueError(f"p_layers must be >= 1, got {p_layers}")

    @property
    def topologies(self) -> list[str]:
        """Resolve topologies from parameter or auto-detect from disk."""
        if self._topologies is not None:
            return self._topologies
        # Auto-detect: find all topologies with NPZ data
        npz_dir = _PROJECT_ROOT / "data" / "multi_n_training"
        if not npz_dir.exists():
            return MULTI_TOPOLOGY_DEFAULTS
        found = set()
        for f in npz_dir.glob(f"*_N*_p{self.p_layers}.npz"):
            topo = f.stem.rsplit("_N", 1)[0]
            found.add(topo)
        return sorted(found) if found else MULTI_TOPOLOGY_DEFAULTS

    def scan(self) -> dict[str, dict[int, int]]:
        """Scan all topologies and return per-topology N→points summary.

        Returns
        -------
        dict[str, dict[int, int]]
            {topology: {N: n_points}} for each included topology.
        """
        self._aggregators = {}
        self._summary = {}

        for topo in self.topologies:
            agg = MultiNAggregator(
                topology=topo,
                model=self.model,
                max_n=self.max_n,
                p_layers=self.p_layers,
            )
            topo_summary = agg.scan()

            if not topo_summary:
                logger.info(f"  MultiTopo: {topo} — no data, skipping")
                continue

            self._aggregators[topo] = agg
            self._summary[topo] = topo_summary

        total = sum(sum(s.values()) for s in self._summary.values())
        logger.info(
            f"MultiTopologyAggregator: {len(self._aggregators)} topologies, "
            f"{total} total points across {sum(len(s) for s in self._summary.values())} configs"
        )
        return self._summary

    def build_combined_dataset(
        self,
        max_de_gap: float = 0.10,
        min_n_values: int = 1,
    ) -> list:
        """Build a combined PyG dataset from all topologies.

        Reuses each per-topology MultiNAggregator.build_combined_dataset() and
        concatenates results. The resulting graphs already encode topology
        structure implicitly (different edge connectivity patterns).

        Quality gate: topologies with fewer than min_verified_points verified
        data points are excluded to prevent contamination.

        Parameters
        ----------
        max_de_gap : float
            Quality filter per point.
        min_n_values : int
            Minimum N values per topology.

        Returns
        -------
        list[Data]
            Combined PyG dataset for UnifiedMPNN training.
        """
        if not self._aggregators:
            self.scan()

        combined_dataset = []
        topology_stats = {}

        for topo, agg in self._aggregators.items():
            try:
                topo_dataset = agg.build_combined_dataset(
                    max_de_gap=max_de_gap,
                    min_n_values=min_n_values,
                )
            except (ValueError, RuntimeError) as e:
                logger.warning(f"  MultiTopo: {topo} dataset build failed: {e}")
                continue

            if not topo_dataset:
                logger.info(f"  MultiTopo: {topo} — 0 graphs after filtering, skipping")
                continue

            # Quality gate: check verified count
            n_verified = sum(
                1
                for g in topo_dataset
                if hasattr(g, "sample_weight") and g.sample_weight.item() >= 0.95
            )
            if n_verified < self.min_verified_points:
                logger.warning(
                    f"  MultiTopo: {topo} has only {n_verified} verified points "
                    f"(need {self.min_verified_points}). Excluding from training."
                )
                continue

            # Tag each graph with topology info (for analysis, not used in forward)
            for g in topo_dataset:
                g.topology = topo

            combined_dataset.extend(topo_dataset)
            topology_stats[topo] = {
                "n_graphs": len(topo_dataset),
                "n_verified": n_verified,
            }
            logger.info(f"  MultiTopo: {topo} → {len(topo_dataset)} graphs ({n_verified} verified)")

        # Validate feature dimension consistency across topologies
        if len(combined_dataset) > 1:
            feat_dims = set(g.x.shape[1] for g in combined_dataset)
            if len(feat_dims) > 1:
                raise ValueError(
                    f"Inconsistent node features across topologies: {feat_dims}. "
                    "All graphs must use the same build_unified_bond_resolved_graph version."
                )

        logger.info(
            f"MultiTopologyAggregator: combined dataset = {len(combined_dataset)} graphs "
            f"from {len(topology_stats)} topologies"
        )
        return combined_dataset

    def summary(self) -> dict[str, Any]:
        """Return aggregation summary."""
        if not self._aggregators:
            self.scan()
        return {
            "model": self.model,
            "topologies": list(self._aggregators.keys()),
            "per_topology": {
                topo: {
                    "n_values": sorted(self._summary.get(topo, {}).keys()),
                    "total_points": sum(self._summary.get(topo, {}).values()),
                }
                for topo in self._aggregators
            },
            "total_points": sum(sum(s.values()) for s in self._summary.values()),
            "max_n": self.max_n,
        }
