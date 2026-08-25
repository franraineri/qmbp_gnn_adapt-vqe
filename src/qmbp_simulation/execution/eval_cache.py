"""Circuit Evaluation Cache — Persist energy evaluations across sessions.

Caches the result of backend.evaluate(circuit, H, theta) keyed by a hash
of (topology, N, model, h, p_layers, theta). Avoids re-running expensive
MPS/statevector simulations when the same parameters are evaluated again.

Two usage patterns:

1. Direct (manual key management):
    from qmbp_simulation.execution.eval_cache import EvalCache

    cache = EvalCache()
    key = cache.make_key(topology="chain_1d", n_qubits=20, h=3.0, theta=theta)
    cached = cache.get(key)
    if cached is not None:
        energy = cached
    else:
        energy = backend.evaluate(circui jt, H, theta)
        cache.put(key, energy)

2. Transparent (wrap any backend — recommended):
    from qmbp_simulation.execution.eval_cache import CachedBackend

    backend = NoiselessBackend()
    cached_backend = CachedBackend(backend, topology="chain_1d", n_qubits=10)
    # Use cached_backend.evaluate() — identical API, automatic caching
    energy = cached_backend.evaluate(circuit, H, theta)

Integration with runners:
    - PipelineRunner: pass CachedBackend as the backend parameter
    - AcceleratedVQE: wraps internal backend automatically when cache=True
    - NoiselessPipelineRunner: --no-eval-cache flag to disable

Storage: JSON at data/eval_cache.json (default)
Max size: 50,000 entries (LRU eviction beyond this)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_PATH = _PROJECT_ROOT / "data" / "eval_cache.json"
_MAX_ENTRIES = 50_000


class EvalCache:
    """Persistent cache for circuit energy evaluations.

    Parameters
    ----------
    path : Path | None
        Path to JSON cache file. Defaults to data/eval_cache_p{p_layers}.json.
    enabled : bool
        If False, all operations are no-ops (zero overhead).
    max_entries : int
        Maximum cache size. Oldest entries evicted when exceeded.
    p_layers : int | None
        If provided and path is None, uses data/eval_cache_p{p_layers}.json.
        Falls back to data/eval_cache.json for backward compatibility when
        p_layers is None or 0.
    """

    def __init__(
        self,
        path: Path | None = None,
        enabled: bool = True,
        max_entries: int = _MAX_ENTRIES,
        p_layers: int | None = None,
    ) -> None:
        if path is not None:
            self._path = path
        elif p_layers and p_layers > 1:
            # Partition cache by p_layers for scalability
            self._path = _PROJECT_ROOT / "data" / f"eval_cache_p{p_layers}.json"
        else:
            # Default: backward compatible single file (covers p=1 and legacy)
            self._path = _DEFAULT_CACHE_PATH
        self._enabled = enabled
        self._max_entries = max_entries
        self._data: dict[str, float] = {}
        self._access_order: list[str] = []  # LRU tracking
        self._hits = 0
        self._misses = 0
        self._dirty = False
        if enabled:
            self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    raw = json.load(f)
                # Support both flat dict and structured format
                if isinstance(raw, dict) and "entries" in raw:
                    self._data = raw["entries"]
                else:
                    self._data = raw
                self._access_order = list(self._data.keys())
                logger.debug("EvalCache: loaded %d entries from %s", len(self._data), self._path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("EvalCache: failed to load %s: %s", self._path, e)
                self._data = {}
                self._access_order = []

    def _save(self) -> None:
        if not self._enabled or not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "2.0",
            "n_entries": len(self._data),
            "entries": self._data,
        }
        # Atomic write: write to tmp then rename (prevents corruption on crash)
        # Use PID in tmp name to avoid collisions between parallel processes
        tmp_path = self._path.with_suffix(f".tmp.{os.getpid()}")
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, separators=(",", ":"))
            tmp_path.rename(self._path)
            self._dirty = False
        except OSError as e:
            logger.warning("EvalCache: atomic save failed: %s", e)
            # Clean up orphaned tmp file
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            # Fallback: direct write (less safe but better than losing data)
            try:
                with open(self._path, "w") as f:
                    json.dump(payload, f, separators=(",", ":"))
                self._dirty = False
            except OSError:
                pass

    def _evict_if_needed(self) -> None:
        """Remove oldest entries if cache exceeds max_entries."""
        while len(self._data) > self._max_entries and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._data.pop(oldest_key, None)

    def make_key(
        self,
        topology: str,
        n_qubits: int,
        h: float,
        theta: np.ndarray,
        *,
        model: str = "tfim",
        p_layers: int = 0,
        J: float = 1.0,
    ) -> str:
        """Create a deterministic cache key from evaluation parameters."""
        # h rounded to 2 decimals (our grid never uses finer resolution).
        # theta_hash ensures unique key even at same h.
        theta_bytes = np.asarray(theta, dtype=np.float64).tobytes()
        theta_hash = hashlib.sha256(theta_bytes).hexdigest()[:32]
        return f"{model}|{topology}|{n_qubits}|{p_layers}|J{J:.4f}|{h:.2f}|{theta_hash}"

    def get(self, key: str) -> float | None:
        """Look up cached energy. Returns None on miss."""
        if not self._enabled:
            return None
        result = self._data.get(key)
        if result is not None:
            self._hits += 1
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        else:
            self._misses += 1
        return result

    def put(self, key: str, energy: float) -> None:
        """Store an energy evaluation result.

        Validates before caching:
        - Rejects NaN/Inf (non-physical)
        - Rejects unreasonably large values (|E| > 1e6 per qubit heuristic)
        """
        if not self._enabled:
            return
        if not np.isfinite(energy):
            logger.debug("EvalCache: rejecting non-finite energy for key %s", key[:30])
            return
        # Sanity bound: for spin-1/2 Hamiltonians, |E| < N * max_coupling * z_max
        # Heuristic: reject obviously wrong values (likely bug, not physics)
        if abs(energy) > 1e6:
            logger.warning(
                "EvalCache: rejecting suspiciously large energy %.2e for key %s",
                energy,
                key[:40],
            )
            return
        self._data[key] = energy
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        self._dirty = True
        self._evict_if_needed()
        # Auto-flush every 50 new entries
        if (self._hits + self._misses) % 50 == 0:
            self._save()

    def get_ground_truth(
        self, topology: str, n_qubits: int, h: float, model: str = "tfim"
    ) -> float | None:
        """Look up cached ground truth energy (from ClassicalSolver)."""
        key = f"GT|{model}|{topology}|{n_qubits}|{h:.2f}"
        return self.get(key)

    def put_ground_truth(
        self,
        topology: str,
        n_qubits: int,
        h: float,
        energy: float,
        model: str = "tfim",
    ) -> None:
        """Cache a ground truth energy computation."""
        key = f"GT|{model}|{topology}|{n_qubits}|{h:.2f}"
        self.put(key, energy)

    def flush(self) -> None:
        """Force save to disk."""
        self._save()

    def clear(self) -> None:
        """Remove all cached entries."""
        self._data.clear()
        self._access_order.clear()
        self._dirty = True
        self._save()

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict[str, Any]:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        return {
            "n_entries": len(self._data),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
            "path": str(self._path),
            "enabled": self._enabled,
        }

    def count_entries_for_config(
        self,
        topology: str,
        n_qubits: int,
        model: str = "tfim",
        p_layers: int = 0,
    ) -> int:
        """Count cached entries matching a (model, topology, N) prefix."""
        if not self._enabled:
            return 0
        prefix = f"{model}|{topology}|{n_qubits}|"
        return sum(1 for k in self._data if k.startswith(prefix))

    def __del__(self) -> None:
        """Auto-flush on garbage collection."""
        try:
            self._save()
        except Exception:
            pass

    def validate_entry(
        self, key: str, backend: Any, circuit: Any, hamiltonian: Any, theta: np.ndarray
    ) -> bool:
        """Spot-check a cached entry against a fresh computation.

        Useful for detecting cache corruption or stale entries after code changes.
        Returns True if cached value matches fresh computation within tolerance.
        """
        cached = self.get(key)
        if cached is None:
            return True  # Nothing to validate
        fresh = backend.evaluate(circuit, hamiltonian, theta)
        # Noiseless backends should be deterministic to machine epsilon
        tolerance = 1e-10
        matches = abs(cached - fresh) < tolerance
        if not matches:
            logger.warning(
                "EvalCache MISMATCH: key=%s, cached=%.10f, fresh=%.10f, diff=%.2e. "
                "Removing stale entry.",
                key[:40],
                cached,
                fresh,
                abs(cached - fresh),
            )
            self._data.pop(key, None)
            self._dirty = True
        return matches


class CachedBackend:
    """Transparent caching wrapper around any ExecutionBackend.

    Intercepts evaluate() calls and returns cached results when available.
    Falls through to the underlying backend on cache miss. Fully compatible
    with ExecutionBackend interface (duck-typing).

    Parameters
    ----------
    backend : ExecutionBackend
        The real backend to wrap (NoiselessBackend, MPSBackend, etc.).
    topology : str
        Lattice topology (used in cache key).
    n_qubits : int
        System size (used in cache key).
    model : str
        Hamiltonian model name (used in cache key).
    p_layers : int
        HVA depth (used in cache key).
    cache : EvalCache | None
        Shared cache instance. If None, creates a new one.
    h_resolver : callable | None
        Function that extracts h-value from hamiltonian. If None, uses
        a hash-based approach (less human-readable keys but always works).

    Usage
    -----
    backend = NoiselessBackend()
    cached = CachedBackend(backend, topology="chain_1d", n_qubits=10)
    # Now use `cached` anywhere you'd use `backend`:
    energy = cached.evaluate(circuit, H, theta)  # cached transparently
    """

    def __init__(
        self,
        backend: Any,
        topology: str = "chain_1d",
        n_qubits: int = 10,
        model: str = "tfim",
        p_layers: int = 2,
        J: float = 1.0,
        cache: EvalCache | None = None,
        h_resolver: Any = None,
    ) -> None:
        # Safety: refuse to cache stochastic backends (noisy/hardware)
        # These produce different results each call due to shot noise.
        backend_name = getattr(backend, "name", type(backend).__name__).lower()
        if any(tag in backend_name for tag in ("noisy", "hardware", "fake")):
            logger.warning(
                "CachedBackend: wrapping a stochastic backend (%s). "
                "Cache disabled — noisy evaluations are non-deterministic.",
                backend_name,
            )
            # Create a passthrough (cache disabled)
            object.__setattr__(self, "_backend", backend)
            object.__setattr__(self, "_topology", topology)
            object.__setattr__(self, "_n_qubits", n_qubits)
            object.__setattr__(self, "_model", model)
            object.__setattr__(self, "_p_layers", p_layers)
            object.__setattr__(self, "_J", J)
            object.__setattr__(self, "_cache", EvalCache(enabled=False))
            object.__setattr__(self, "_h_resolver", None)
            object.__setattr__(self, "_h_current", 0.0)
            return

        # Set all instance attributes BEFORE __getattr__ could be triggered
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_topology", topology)
        object.__setattr__(self, "_n_qubits", n_qubits)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_p_layers", p_layers)
        object.__setattr__(self, "_J", J)
        object.__setattr__(self, "_cache", cache if cache is not None else EvalCache(p_layers=p_layers))
        object.__setattr__(self, "_h_resolver", h_resolver)
        object.__setattr__(self, "_h_current", 0.0)

    @property
    def name(self) -> str:
        """Backend name with cache indicator."""
        return f"Cached({self._backend.name})"

    def set_h(self, h: float) -> None:
        """Set current h-value for cache key generation.

        Call this before evaluate() when sweeping through h-values.
        If not called, uses h=0.0 (still caches, but key is less precise).
        """
        self._h_current = h

    def evaluate(
        self,
        circuit: Any,
        hamiltonian: Any,
        params: np.ndarray,
    ) -> float:
        """Evaluate with caching. Falls through to backend on miss."""
        h = self._h_current
        key = self._cache.make_key(
            topology=self._topology,
            n_qubits=self._n_qubits,
            h=h,
            theta=params,
            model=self._model,
            p_layers=self._p_layers,
            J=self._J,
        )

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        energy = self._backend.evaluate(circuit, hamiltonian, params)
        self._cache.put(key, float(energy))
        return energy

    def compute_fidelity(
        self,
        circuit: Any,
        params: np.ndarray,
        exact_state: np.ndarray,
    ) -> float:
        """Delegate fidelity computation (not cached — cheap enough)."""
        return self._backend.compute_fidelity(circuit, params, exact_state)

    def compute_energy_variance(
        self,
        circuit: Any,
        hamiltonian: Any,
        params: np.ndarray,
    ) -> float:
        """Delegate variance computation."""
        return self._backend.compute_energy_variance(circuit, hamiltonian, params)

    def get_statevector(self, circuit: Any, params: np.ndarray) -> np.ndarray:
        """Delegate statevector extraction."""
        return self._backend.get_statevector(circuit, params)

    @property
    def cache(self) -> EvalCache:
        """Access the underlying cache for stats/flush."""
        return self._cache

    def flush(self) -> None:
        """Force-save cache to disk. Delegates to EvalCache.flush()."""
        self._cache.flush()

    def stats(self) -> dict[str, Any]:
        """Return cache performance statistics. Delegates to EvalCache.stats()."""
        return self._cache.stats()

    def validate_entry(
        self,
        key: str,
        circuit: Any,
        hamiltonian: Any,
        theta: np.ndarray,
    ) -> bool:
        """Spot-check a cached entry against a fresh computation.


        Delegates to EvalCache.validate_entry() using the wrapped backend
        for the fresh evaluation. Useful for detecting cache corruption.
        """
        return self._cache.validate_entry(key, self._backend, circuit, hamiltonian, theta)

    def make_key(
        self,
        h: float,
        theta: np.ndarray,
    ) -> str:
        """Create a cache key for the current configuration + given h and theta.

        Convenience method that uses the topology/n_qubits/model/p_layers
        set at construction time, so callers don't need to repeat them.

        Parameters
        ----------
        h : float
            Transverse field value.
        theta : np.ndarray
            Parameter vector.

        Returns
        -------
        str
            Deterministic cache key.
        """
        return self._cache.make_key(
            topology=self._topology,
            n_qubits=self._n_qubits,
            h=h,
            theta=theta,
            model=self._model,
            p_layers=self._p_layers,
        )

    def __getattr__(self, name: str) -> Any:
        """Forward any other attributes to the underlying backend.

        Note: __getattr__ is only called when normal attribute lookup fails,
        so it won't intercept _cache, _backend, or other instance attributes.
        """
        return getattr(self._backend, name)

    # ── Context manager protocol ─────────────────────────────────────────────

    def __enter__(self) -> CachedBackend:
        """Context manager entry — returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit — flush cache regardless of exception."""
        self.flush()
        return False
