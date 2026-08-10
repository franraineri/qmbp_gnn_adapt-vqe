"""Ground Truth Cache — Persist DMRG/ExactDiag results for reuse.

Computing ground truth (especially DMRG for N>22) is expensive. This
module caches results keyed by (model, topology, N, h, method) so that
repeated evaluations at the same points are instant.

Storage: JSON file at ``data/ground_truth_cache.json`` with structure:
    { "version": "2.0",
      "entries": { "chain_1d|30|tfim|3.500000": {"energy": -X, "gap": Y, ...}, ... } }

Usage:
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    cache = GroundTruthCache()

    # Check before computing
    cached = cache.get(topology="chain_1d", n_qubits=30, model="tfim", h=3.5)
    if cached:
        e_exact, gap = cached["energy"], cached["gap"]
    else:
        gt = solver.solve(H, lattice)
        cache.put_from_result("chain_1d", 30, "tfim", 3.5, gt)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CACHE_PATH = _PROJECT_ROOT / "data" / "ground_truth_cache.json"


class GroundTruthCache:
    """Persistent cache for ground truth energies and gaps.

    Stores results indexed by (topology, N, model, h) to avoid
    recomputing expensive DMRG solves across sessions.

    Validates on write: rejects NaN/Inf, non-physical gaps (<0),
    and suspiciously large energies. Batches disk writes to avoid
    excessive I/O during sweeps.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _CACHE_PATH
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._write_count = 0
        self._load()

    def _make_key(self, topology: str, n_qubits: int, model: str, h: float) -> str:
        """Create cache key with 6-decimal h for float stability."""
        return f"{topology}|{n_qubits}|{model}|{h:.6f}"

    def _load(self) -> None:
        """Load cache from disk. Handles both legacy and v2 formats."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    raw = json.load(f)
                # v2 format: {"version": "2.0", "entries": {...}}
                if isinstance(raw, dict) and "entries" in raw:
                    self._data = raw["entries"]
                else:
                    # Legacy format: flat dict
                    self._data = raw
                logger.debug("GroundTruthCache: loaded %d entries", len(self._data))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        """Persist cache to disk in compact format."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "2.0",
            "n_entries": len(self._data),
            "entries": self._data,
        }
        with open(self._path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        self._dirty = False

    def get(
        self, topology: str, n_qubits: int, model: str, h: float
    ) -> dict[str, Any] | None:
        """Look up cached ground truth.

        Returns dict with {energy, gap, method, mag_x, corr_zz} or None.
        """
        key = self._make_key(topology, n_qubits, model, h)
        return self._data.get(key)

    def put(
        self,
        topology: str,
        n_qubits: int,
        model: str,
        h: float,
        *,
        energy: float,
        gap: float,
        method: str = "unknown",
        mag_x: float | None = None,
        corr_zz: float | None = None,
    ) -> None:
        """Store a ground truth result in the cache.

        Validates before storing:
        - Energy must be finite
        - Gap must be finite and >= 0
        - |Energy| must be < 1e6 (sanity bound for spin systems)
        """
        # Validation: reject non-physical values
        if not np.isfinite(energy):
            logger.warning("GroundTruthCache: rejecting non-finite energy for %s|%d|%s|%.4f",
                           topology, n_qubits, model, h)
            return
        if not np.isfinite(gap) or gap < 0:
            logger.warning("GroundTruthCache: rejecting invalid gap=%.4e for %s|%d|%s|%.4f",
                           gap, topology, n_qubits, model, h)
            return
        if abs(energy) > 1e6:
            logger.warning("GroundTruthCache: rejecting |energy|=%.2e > 1e6", abs(energy))
            return

        key = self._make_key(topology, n_qubits, model, h)
        self._data[key] = {
            "energy": energy,
            "gap": gap,
            "method": method,
            "mag_x": mag_x,
            "corr_zz": corr_zz,
        }
        self._dirty = True
        self._write_count += 1
        # Batch writes: flush every 10 puts (avoids 14 writes per sweep)
        if self._write_count % 10 == 0:
            self._save()

    def put_from_result(
        self, topology: str, n_qubits: int, model: str, h: float, gt: Any
    ) -> None:
        """Store from a GroundTruthResult object."""
        self.put(
            topology=topology, n_qubits=n_qubits, model=model, h=h,
            energy=gt.ground_energy, gap=gt.gap,
            method=gt.gap_method if hasattr(gt, "gap_method") else "unknown",
            mag_x=gt.mag_x if hasattr(gt, "mag_x") else None,
            corr_zz=gt.corr_zz if hasattr(gt, "corr_zz") else None,
        )

    def flush(self) -> None:
        """Force save to disk."""
        self._save()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: tuple) -> bool:
        topology, n_qubits, model, h = key
        return self._make_key(topology, n_qubits, model, h) in self._data

    def __del__(self) -> None:
        """Auto-flush on garbage collection."""
        try:
            self._save()
        except Exception:
            pass

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        if not self._data:
            return {"n_entries": 0}
        topologies = set()
        n_values = set()
        models = set()
        for key in self._data:
            parts = key.split("|")
            if len(parts) >= 3:
                topologies.add(parts[0])
                try:
                    n_values.add(int(parts[1]))
                except ValueError:
                    pass
                models.add(parts[2])
        return {
            "n_entries": len(self._data),
            "topologies": sorted(topologies),
            "n_values": sorted(n_values),
            "models": sorted(models),
        }
