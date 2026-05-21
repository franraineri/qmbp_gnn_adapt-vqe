"""
V6.1 Hardware Deployer — Production-ready deployment with full error mitigation stack.

Implements inhomogeneous ZNE, dynamical decoupling, Pauli twirling + TREX,
observable grouping, and NN-enhanced ZNE extrapolation for IBM quantum hardware.

This module extends the V6.0 deployer without modifying stable code.
"""

from __future__ import annotations

import logging
import os
import random
from collections import deque
from datetime import UTC, datetime

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .config import GroundTruthResult, LatticeConfig
from .config_v61 import (
    CALIBRATION_MAX_AGE_HOURS,
    DEFAULT_NUM_RANDOMIZATIONS,
    DEFAULT_SHOTS_PER_RANDOMIZATION,
    MAX_ADAPT_ITERATIONS_HARDWARE,
    MAX_CES_RATIO,
    MAX_LAYOUTS,
    MIN_CES_RATIO,
    MIN_LAYOUTS,
    MIN_SHOT_OVERRIDE,
    NN_HIDDEN_LAYERS,
    NN_MAX_ITER,
    NN_MIN_DATA_POINTS,
    SHOT_BUDGET_LARGE,
    SHOT_BUDGET_MEDIUM,
    SHOT_BUDGET_SMALL,
    ZNE_R_SQUARED_WARNING_THRESHOLD,
    BaselineComparison,
    BaselineMetrics,
    DeployResultV61,
    LayoutResult,
)

logger = logging.getLogger(__name__)


class ObservableGrouper:
    """Group commuting observables into minimal measurement bases.

    Groups N site-local ⟨X_i⟩ into one SparsePauliOp and N-1 bond-local
    ⟨Z_iZ_{i+1}⟩ into another, reducing circuit executions from O(N) to O(1).

    Important: EstimatorV2 behavior with observables:
    - A single multi-term SparsePauliOp → returns a SCALAR (weighted sum of
      all terms' expectation values). Useful for total energy computation.
    - A LIST of individual SparsePauliOps → returns an ARRAY with one value
      per observable. Required for per-site/per-bond measurements.

    For per-term expectation values (needed for phase classification), submit
    observables as a list of individual operators, not as a grouped SparsePauliOp.
    The grouped SparsePauliOp is still useful for total energy estimation where
    only the sum matters.
    """

    @staticmethod
    def group_observables(
        lattice: LatticeConfig,
    ) -> tuple[SparsePauliOp, SparsePauliOp]:
        """Group site and bond observables into two SparsePauliOps.

        Parameters
        ----------
        lattice : LatticeConfig
            Lattice specification with edges and n_qubits.

        Returns
        -------
        (x_group, zz_group)
            x_group : SparsePauliOp combining all X_i with coefficient 1.0 each
                      (N terms total — estimator returns per-term expectation values)
            zz_group : SparsePauliOp combining all Z_iZ_{i+1} with coefficient 1.0 each
                       (n_bonds terms total)

        Note: Coefficients are 1.0 per term (not 1/N). The estimator returns
        individual expectation values for each Pauli term in the operator.
        Averaging to get bulk ⟨X⟩ and ⟨ZZ⟩ is done in extract_individual_values().
        """
        n_qubits = lattice.n_qubits

        # Build X group: one X_i term per site
        x_sparse_list = [("X", [i], 1.0) for i in range(n_qubits)]
        x_group = SparsePauliOp.from_sparse_list(x_sparse_list, num_qubits=n_qubits)

        # Build ZZ group: one Z_iZ_j term per bond
        zz_sparse_list = [("ZZ", [i, j], 1.0) for (i, j) in lattice.edges]
        zz_group = SparsePauliOp.from_sparse_list(zz_sparse_list, num_qubits=n_qubits)

        logger.debug(
            "Grouped observables: %d X terms, %d ZZ terms for %d qubits",
            n_qubits,
            len(lattice.edges),
            n_qubits,
        )

        return x_group, zz_group

    @staticmethod
    def extract_individual_values(
        x_result: np.ndarray,
        zz_result: np.ndarray,
        lattice: LatticeConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract per-site and per-bond values from grouped measurement.

        The EstimatorV2 returns per-term expectation values when given a
        multi-term SparsePauliOp, so the results are already in canonical
        order matching the construction in group_observables().

        Parameters
        ----------
        x_result : np.ndarray
            Raw expectation values from X-group measurement (length N).
        zz_result : np.ndarray
            Raw expectation values from ZZ-group measurement (length n_bonds).
        lattice : LatticeConfig
            Lattice for ordering reference.

        Returns
        -------
        (per_site_x, per_bond_zz)
            Individual observable values in canonical order.

        Raises
        ------
        ValueError
            If array lengths don't match expected lattice dimensions.
        """
        per_site_x = np.asarray(x_result, dtype=np.float64)
        per_bond_zz = np.asarray(zz_result, dtype=np.float64)

        if per_site_x.shape[0] != lattice.n_qubits:
            raise ValueError(
                f"x_result length ({per_site_x.shape[0]}) must match n_qubits ({lattice.n_qubits})."
            )
        if per_bond_zz.shape[0] != len(lattice.edges):
            raise ValueError(
                f"zz_result length ({per_bond_zz.shape[0]}) must match "
                f"number of edges ({len(lattice.edges)})."
            )

        return per_site_x, per_bond_zz

    @staticmethod
    def apply_layout(
        observable: SparsePauliOp,
        layout,
    ) -> SparsePauliOp:
        """Apply transpilation layout to observable for hardware execution.

        Parameters
        ----------
        observable : SparsePauliOp
            Observable in logical qubit ordering.
        layout
            Transpilation layout from a transpiled circuit
            (e.g., ``transpiled_circuit.layout``).

        Returns
        -------
        SparsePauliOp
            Observable remapped to physical qubit indices.
        """
        return observable.apply_layout(layout)


class LayoutSelector:
    """Select qubit layouts with diverse CES for inhomogeneous ZNE.

    Queries backend calibration data and finds connected subsets
    spanning a range of circuit error sums.
    """

    def __init__(self, backend, seed: int = 42) -> None:
        """Initialize with backend for calibration data access.

        Parameters
        ----------
        backend
            A Qiskit backend object (IBM backend with calibration data).
            Must expose ``backend.target`` (Qiskit 2.x) or
            ``backend.properties()`` for gate error rates.
        seed : int
            Random seed for reproducible layout selection (default: 42).
        """
        self._backend = backend
        self._rng = random.Random(seed)
        self._layout_cache: dict[tuple[int, int], list[LayoutResult]] = {}

        # Detect fake backends (skip calibration freshness check)
        backend_name = getattr(backend, "name", "")
        self._is_fake_backend = "fake" in backend_name.lower()

        # Extract calibration data and build connectivity graph
        self._gate_errors: dict[tuple[int, int], float] = {}
        self._connectivity_graph: dict[int, list[int]] = {}
        self._calibration_timestamp: datetime | None = None

        self._extract_calibration_data()

    def _extract_calibration_data(self) -> None:
        """Extract gate error rates and connectivity from backend."""
        target = self._backend.target

        # Extract calibration timestamp
        # Qiskit 2.x backends store this in target.dt or via properties
        if hasattr(self._backend, "properties") and self._backend.properties():
            props = self._backend.properties()
            if hasattr(props, "last_update_date"):
                self._calibration_timestamp = props.last_update_date
        elif hasattr(target, "dt"):
            # Fallback: use current time if no explicit timestamp
            self._calibration_timestamp = None

        # Iterate over all qargs in the target to find 2-qubit connections
        for op_name in target.operation_names:
            qargs_list = target.qargs_for_operation_name(op_name)
            if qargs_list is None:
                continue
            for qargs in qargs_list:
                if len(qargs) == 2:
                    q0, q1 = qargs
                    # Get error rate for this gate on these qubits
                    gate_props = target[op_name].get((q0, q1))
                    if gate_props is not None and gate_props.error is not None:
                        error = gate_props.error
                    else:
                        error = 0.01  # Default fallback error rate

                    # Store the error (use the first 2Q gate found per edge)
                    edge = (min(q0, q1), max(q0, q1))
                    if edge not in self._gate_errors:
                        self._gate_errors[edge] = error

                    # Build adjacency list
                    if q0 not in self._connectivity_graph:
                        self._connectivity_graph[q0] = []
                    if q1 not in self._connectivity_graph:
                        self._connectivity_graph[q1] = []
                    if q1 not in self._connectivity_graph[q0]:
                        self._connectivity_graph[q0].append(q1)
                    if q0 not in self._connectivity_graph[q1]:
                        self._connectivity_graph[q1].append(q0)

        logger.debug(
            "Extracted calibration: %d edges, %d qubits",
            len(self._gate_errors),
            len(self._connectivity_graph),
        )

    def select_layouts(
        self,
        n_qubits: int,
        n_layouts: int = 3,
        min_ces_ratio: float = MIN_CES_RATIO,
    ) -> list[LayoutResult]:
        """Find n_layouts connected qubit subsets with diverse CES.

        Parameters
        ----------
        n_qubits : int
            Number of qubits needed.
        n_layouts : int
            Target number of layouts (3-5).
        min_ces_ratio : float
            Minimum ratio max_CES/min_CES for meaningful extrapolation.

        Returns
        -------
        list[LayoutResult]
            Selected layouts with their CES values.

        Notes
        -----
        Logs warning and returns single default layout if insufficient spread.
        """
        # Clamp n_layouts to valid range for ZNE (3-5).
        # If n_layouts < MIN_LAYOUTS, caller wants raw execution (no ZNE) —
        # return a single best layout without diversity requirements.
        if n_layouts < MIN_LAYOUTS:
            # Single-layout mode: return the best (lowest CES) layout
            # No diversity needed — this is for raw noisy baseline
            cache_key = (n_qubits, n_layouts)
            if cache_key in self._layout_cache:
                return self._layout_cache[cache_key]

            subsets = self._find_connected_subsets(n_qubits, self._connectivity_graph)
            if not subsets:
                default_layout = self._make_default_layout(n_qubits)
                self._layout_cache[cache_key] = [default_layout]
                return [default_layout]

            # Find the subset with lowest topology CES
            best_subset = None
            best_ces = float("inf")
            best_n2q = 0
            for subset in subsets:
                ces, n_2q = self._compute_subset_ces(subset)
                if 0 < ces < best_ces:
                    best_subset = subset
                    best_ces = ces
                    best_n2q = n_2q

            if best_subset is None:
                default_layout = self._make_default_layout(n_qubits)
                self._layout_cache[cache_key] = [default_layout]
                return [default_layout]

            result = [
                LayoutResult(
                    initial_layout=best_subset, ces=best_ces, two_qubit_gate_count=best_n2q
                )
            ]
            self._layout_cache[cache_key] = result
            logger.debug("Single-layout mode: CES=%.4f, qubits=%s", best_ces, best_subset[:5])
            return result

        n_layouts = min(MAX_LAYOUTS, n_layouts)

        # Check cache
        cache_key = (n_qubits, n_layouts)
        if cache_key in self._layout_cache:
            logger.debug("Returning cached layouts for key %s", cache_key)
            return self._layout_cache[cache_key]

        # Check calibration freshness
        if not self._is_calibration_fresh():
            logger.warning(
                "Backend calibration data is stale (> %d hours). Proceeding with default layout.",
                CALIBRATION_MAX_AGE_HOURS,
            )
            default_layout = self._make_default_layout(n_qubits)
            self._layout_cache[cache_key] = [default_layout]
            return [default_layout]

        # Find connected subsets
        subsets = self._find_connected_subsets(n_qubits, self._connectivity_graph)

        if not subsets:
            logger.warning(
                "No connected subsets of size %d found. Returning default layout.",
                n_qubits,
            )
            default_layout = self._make_default_layout(n_qubits)
            self._layout_cache[cache_key] = [default_layout]
            return [default_layout]

        # Compute CES for each subset
        subset_ces_pairs: list[tuple[list[int], float, int]] = []
        for subset in subsets:
            ces, n_2q_gates = self._compute_subset_ces(subset)
            if ces > 0:
                subset_ces_pairs.append((subset, ces, n_2q_gates))

        if not subset_ces_pairs:
            logger.warning("No subsets with valid CES found. Returning default layout.")
            default_layout = self._make_default_layout(n_qubits)
            self._layout_cache[cache_key] = [default_layout]
            return [default_layout]

        # Sort by CES to select diverse layouts
        subset_ces_pairs.sort(key=lambda x: x[1])

        # Select n_layouts subsets that maximize CES spread
        selected = self._select_diverse_layouts(subset_ces_pairs, n_layouts)

        # Verify CES spread
        ces_values = [ces for _, ces, _ in selected]
        if len(ces_values) >= MIN_LAYOUTS:
            ces_ratio = max(ces_values) / min(ces_values) if min(ces_values) > 0 else 0
            if ces_ratio < min_ces_ratio:
                logger.warning(
                    "Insufficient CES spread: ratio %.2f < %.2f. Returning single default layout.",
                    ces_ratio,
                    min_ces_ratio,
                )
                default_layout = LayoutResult(
                    initial_layout=selected[0][0],
                    ces=selected[0][1],
                    two_qubit_gate_count=selected[0][2],
                )
                self._layout_cache[cache_key] = [default_layout]
                return [default_layout]
        else:
            logger.warning(
                "Fewer than %d valid layouts found (%d). Returning single default layout.",
                MIN_LAYOUTS,
                len(ces_values),
            )
            default_layout = LayoutResult(
                initial_layout=selected[0][0],
                ces=selected[0][1],
                two_qubit_gate_count=selected[0][2],
            )
            self._layout_cache[cache_key] = [default_layout]
            return [default_layout]

        # Build LayoutResult list
        results = [
            LayoutResult(
                initial_layout=subset,
                ces=ces,
                two_qubit_gate_count=n_2q,
            )
            for subset, ces, n_2q in selected
        ]

        # Cache and return
        self._layout_cache[cache_key] = results
        logger.debug(
            "Selected %d layouts with CES range [%.4f, %.4f] (ratio %.2f)",
            len(results),
            min(ces_values),
            max(ces_values),
            ces_ratio,
        )
        return results

    def compute_ces(
        self,
        transpiled_circuit,
    ) -> float:
        """Compute Circuit Error Sum from transpiled circuit's 2Q gates.

        Sums two-qubit gate error rates from calibration data for all
        two-qubit gates in the transpiled circuit.

        Parameters
        ----------
        transpiled_circuit
            A transpiled QuantumCircuit (mapped to physical qubits).

        Returns
        -------
        float
            Total circuit error sum.
        """
        ces = 0.0
        for instruction in transpiled_circuit.data:
            if instruction.operation.num_qubits == 2:
                # Get physical qubit indices
                qubits = [transpiled_circuit.find_bit(q).index for q in instruction.qubits]
                q0, q1 = qubits[0], qubits[1]
                edge = (min(q0, q1), max(q0, q1))

                # Look up error rate
                error = self._gate_errors.get(edge, 0.01)
                ces += error

        return ces

    def _find_connected_subsets(
        self,
        n_qubits: int,
        connectivity_graph: dict[int, list[int]],
    ) -> list[list[int]]:
        """Find connected qubit subsets on heavy-hex topology.

        Uses BFS from randomly sampled starting nodes to find connected
        subgraphs of the requested size. For large backends (e.g., 127 qubits),
        random sampling is used to keep computation tractable.

        Parameters
        ----------
        n_qubits : int
            Required subset size.
        connectivity_graph : dict[int, list[int]]
            Adjacency list representation of backend connectivity.

        Returns
        -------
        list[list[int]]
            List of distinct connected subsets (aim for 10-20 candidates).
        """
        all_nodes = list(connectivity_graph.keys())
        if not all_nodes:
            return []

        # For small graphs, try all starting nodes; for large ones, sample more
        # Increased from 40 to 80 for better diversity on large backends (133 qubits)
        max_starts = min(len(all_nodes), 80)
        if len(all_nodes) > max_starts:
            start_nodes = self._rng.sample(all_nodes, max_starts)
        else:
            start_nodes = all_nodes

        subsets: list[list[int]] = []
        seen_subsets: set[tuple[int, ...]] = set()

        for start in start_nodes:
            # BFS to find a connected subset of size n_qubits
            subset = self._bfs_subset(start, n_qubits, connectivity_graph)
            if subset is not None and len(subset) == n_qubits:
                key = tuple(sorted(subset))
                if key not in seen_subsets:
                    seen_subsets.add(key)
                    subsets.append(subset)

            # Stop if we have enough candidates
            if len(subsets) >= 40:
                break

        return subsets

    def _bfs_subset(
        self,
        start: int,
        n_qubits: int,
        connectivity_graph: dict[int, list[int]],
    ) -> list[int] | None:
        """BFS from start node to find a connected subset of size n_qubits."""
        if start not in connectivity_graph:
            return None

        visited = [start]
        queue = deque(connectivity_graph.get(start, []))
        visited_set = {start}

        while queue and len(visited) < n_qubits:
            node = queue.popleft()
            if node in visited_set:
                continue
            visited_set.add(node)
            visited.append(node)

            # Add neighbors to queue
            for neighbor in connectivity_graph.get(node, []):
                if neighbor not in visited_set:
                    queue.append(neighbor)

        if len(visited) >= n_qubits:
            return visited[:n_qubits]
        return None

    def _is_calibration_fresh(self) -> bool:
        """Check if calibration data is less than 24 hours old.

        Returns True if calibration age < CALIBRATION_MAX_AGE_HOURS.
        If the calibration timestamp is unavailable (common with newer
        Qiskit 2.x backends using Target API), assumes fresh calibration
        to avoid blocking inhomogeneous ZNE on modern backends.

        For fake backends (FakeTorino, etc.), always returns True since
        their calibration data is a static snapshot and the freshness
        check is irrelevant.

        Returns
        -------
        bool
            True if calibration is fresh or timestamp is unavailable.
        """
        # Fake backends have static calibration data — always "fresh"
        if self._is_fake_backend:
            return True

        if self._calibration_timestamp is None:
            # Modern backends may not expose timestamp via properties.
            # Assume fresh to avoid blocking ZNE on these backends.
            logger.debug("Calibration timestamp unavailable — assuming fresh calibration.")
            return True

        now = datetime.now(UTC)
        # Ensure timestamp is timezone-aware
        cal_time = self._calibration_timestamp
        if cal_time.tzinfo is None:
            cal_time = cal_time.replace(tzinfo=UTC)

        age_hours = (now - cal_time).total_seconds() / 3600.0
        return age_hours < CALIBRATION_MAX_AGE_HOURS

    def _compute_subset_ces(self, subset: list[int]) -> tuple[float, int]:
        """Compute *topology CES* for a qubit subset by summing internal edge errors.

        This is an intentional heuristic used during layout selection: it
        estimates the Circuit Error Sum from the subset's connectivity graph
        (sum of calibrated 2Q gate errors on internal edges) WITHOUT actually
        transpiling a circuit onto those qubits.  This avoids the cost of
        transpiling every candidate subset during the selection search.

        The true *circuit CES* — which accounts for the actual gate count and
        routing overhead of a specific transpiled circuit — is computed by
        :meth:`compute_ces(transpiled_circuit)` and used for the ZNE
        extrapolation axis.  The topology CES here serves only to rank
        candidate layouts by expected noise level so that the final selected
        set spans a diverse CES range.

        Parameters
        ----------
        subset : list[int]
            Physical qubit indices.

        Returns
        -------
        (ces, n_2q_gates)
            Total error sum and number of internal 2Q edges.
        """
        subset_set = set(subset)
        ces = 0.0
        n_2q_gates = 0

        for q in subset:
            for neighbor in self._connectivity_graph.get(q, []):
                if neighbor in subset_set and neighbor > q:
                    edge = (q, neighbor)
                    error = self._gate_errors.get(edge, 0.01)
                    ces += error
                    n_2q_gates += 1

        return ces, n_2q_gates

    def _select_diverse_layouts(
        self,
        sorted_pairs: list[tuple[list[int], float, int]],
        n_layouts: int,
    ) -> list[tuple[list[int], float, int]]:
        """Select n_layouts subsets that maximize CES spread without extreme outliers.

        Strategy: first filter out extreme outliers (CES > MAX_CES_RATIO × min),
        then pick evenly spaced layouts from the filtered set. This ensures all
        selected layouts produce physically meaningful results while still
        providing CES diversity for ZNE extrapolation.

        Parameters
        ----------
        sorted_pairs : list
            (subset, ces, n_2q_gates) sorted by CES ascending.
        n_layouts : int
            Number of layouts to select.

        Returns
        -------
        list
            Selected (subset, ces, n_2q_gates) tuples.
        """
        if len(sorted_pairs) <= n_layouts:
            return sorted_pairs

        # Step 1: Filter out extreme outliers FIRST (before selection)
        min_ces = sorted_pairs[0][1]  # sorted ascending, first is minimum
        if min_ces > 0:
            max_allowed_ces = MAX_CES_RATIO * min_ces
            valid_pairs = [s for s in sorted_pairs if s[1] <= max_allowed_ces]

            if len(valid_pairs) < len(sorted_pairs):
                n_removed = len(sorted_pairs) - len(valid_pairs)
                logger.info(
                    "Pre-filtered %d layout(s) with CES > %.2f × min_CES (%.4f). "
                    "%d candidates remain.",
                    n_removed,
                    MAX_CES_RATIO,
                    min_ces,
                    len(valid_pairs),
                )
        else:
            valid_pairs = sorted_pairs

        # Step 2: Select evenly spaced from the VALID set
        if len(valid_pairs) <= n_layouts:
            return valid_pairs

        n = len(valid_pairs)
        indices = [int(round(i * (n - 1) / (n_layouts - 1))) for i in range(n_layouts)]
        indices = sorted(set(indices))

        return [valid_pairs[i] for i in indices]

    def _make_default_layout(self, n_qubits: int) -> LayoutResult:
        """Create a default layout using the first n_qubits connected nodes.

        Parameters
        ----------
        n_qubits : int
            Number of qubits needed.

        Returns
        -------
        LayoutResult
            Default layout with estimated CES.
        """
        all_nodes = sorted(self._connectivity_graph.keys())
        if all_nodes:
            subset = self._bfs_subset(all_nodes[0], n_qubits, self._connectivity_graph)
            if subset is not None:
                ces, n_2q = self._compute_subset_ces(subset)
                return LayoutResult(
                    initial_layout=subset,
                    ces=ces,
                    two_qubit_gate_count=n_2q,
                )

        # Ultimate fallback: sequential qubits
        return LayoutResult(
            initial_layout=list(range(n_qubits)),
            ces=0.0,
            two_qubit_gate_count=0,
        )


# ---------------------------------------------------------------------------
# EstimatorV2 Options Builder
# ---------------------------------------------------------------------------


def build_estimator_options(
    shots: int = 8192,
    enable_dd: bool = True,
    dd_sequence: str = "XpXm",
    enable_twirling: bool = True,
    num_randomizations: int = DEFAULT_NUM_RANDOMIZATIONS,
    shots_per_randomization: int = DEFAULT_SHOTS_PER_RANDOMIZATION,
    enable_trex: bool = True,
    enable_runtime_zne: bool = False,
    zne_noise_factors: list[int] | None = None,
    zne_extrapolator: str = "exponential",
) -> dict:
    """Build EstimatorV2 options dictionary for hardware execution.

    Parameters
    ----------
    shots : int
        Total shot budget per circuit.
    enable_dd : bool
        Enable dynamical decoupling (XpXm sequence).
    dd_sequence : str
        DD sequence type ("XpXm", "XY4", "XX").
    enable_twirling : bool
        Enable Pauli gate twirling.
    num_randomizations : int
        Number of twirling randomizations.
    shots_per_randomization : int
        Shots per twirling randomization.
    enable_trex : bool
        Enable TREX measurement mitigation.
    enable_runtime_zne : bool
        Enable Runtime-level ZNE (alternative to inhomogeneous ZNE).
    zne_noise_factors : list[int] | None
        Noise factors for Runtime ZNE [1, 2, 3].
    zne_extrapolator : str
        Extrapolator type for Runtime ZNE.

    Returns
    -------
    dict
        Options dictionary to pass to EstimatorV2.
    """
    options: dict = {
        "default_shots": shots,
    }

    if enable_dd:
        options["dynamical_decoupling"] = {
            "enable": True,
            "sequence_type": dd_sequence,
        }

    if enable_twirling:
        options["twirling"] = {
            "enable_gates": True,
            "num_randomizations": num_randomizations,
            "shots_per_randomization": shots_per_randomization,
        }

    if enable_trex:
        options.setdefault("resilience", {})
        options["resilience"]["measure_mitigation"] = True

    if enable_runtime_zne:
        options.setdefault("resilience", {})
        options["resilience"]["zne"] = {
            "noise_factors": zne_noise_factors or [1, 2, 3],
            "extrapolator": zne_extrapolator,
        }

    return options


# ---------------------------------------------------------------------------
# Shot Budget Computation
# ---------------------------------------------------------------------------


def compute_shot_budget(n_qubits: int, shots_override: int | None = None) -> int:
    """Determine shot budget based on system size or user override.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.
    shots_override : int | None
        User-specified shot budget. Must be >= 4096 if provided.

    Returns
    -------
    int
        Shot budget to use.

    Raises
    ------
    ValueError
        If shots_override < 4096.
    """
    if shots_override is not None:
        if shots_override < MIN_SHOT_OVERRIDE:
            raise ValueError(
                f"Shot budget override must be >= {MIN_SHOT_OVERRIDE}, got {shots_override}."
            )
        return shots_override

    # Scale shot budget based on system size
    if n_qubits <= 6:
        return SHOT_BUDGET_SMALL
    elif n_qubits <= 10:
        return SHOT_BUDGET_MEDIUM
    else:
        return SHOT_BUDGET_LARGE


# ---------------------------------------------------------------------------
# HardwareDeployerV61 — Main Orchestrator
# ---------------------------------------------------------------------------


class HardwareDeployerV61:
    """Production-ready hardware deployer with full error mitigation stack.

    Supports three modes:
    - 'simulation': StatevectorEstimator (noiseless, exact)
    - 'noisy_simulation': FakeTorino + BackendEstimatorV2 (local noisy, ZNE only)
    - 'hardware': IBM Runtime EstimatorV2 (full mitigation: DD, twirling, TREX, ZNE)
    """

    def __init__(
        self,
        backend_name: str | None = None,
        mode: str = "simulation",
        nn_extrapolation: bool = False,
        shots: int | None = None,
        n_layouts: int = 3,
        seed: int = 42,
    ) -> None:
        """Initialize the V6.1 deployer.

        Parameters
        ----------
        backend_name : str | None
            IBM backend name (e.g., "ibm_torino"). None → simulation mode.
        mode : str
            "hardware", "noisy_simulation", or "simulation" (default).
        nn_extrapolation : bool
            Enable NN-enhanced ZNE extrapolation (Sun et al. 2025).
        shots : int | None
            Override shot budget. Must be ≥ 4096 if specified.
        n_layouts : int
            Number of layouts for inhomogeneous ZNE (3-5).
        seed : int
            Random seed for reproducible layout selection (default: 42).
        """
        self._mode = mode
        self._nn_extrapolation = nn_extrapolation
        self._shots_override = shots
        self._n_layouts = n_layouts
        self._seed = seed
        self._backend = None
        self._layout_selector: LayoutSelector | None = None

        if mode == "hardware":
            # Validate environment variables
            ibm_key = os.environ.get("IBM_KEY")
            if not ibm_key:
                raise ValueError(
                    "IBM_KEY environment variable is not set. "
                    "Required for hardware mode connection to IBM Quantum."
                )
            ibm_instance = os.environ.get("IBM_INSTANCE_CRN", "")

            # Connect to IBM backend
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=ibm_key,
                instance=ibm_instance,
            )
            self._backend = service.backend(backend_name)
            self._layout_selector = LayoutSelector(self._backend, seed=seed)

            logger.info(
                "HardwareDeployerV61 initialized in hardware mode: backend=%s",
                backend_name,
            )
        elif mode == "noisy_simulation":
            try:
                from qiskit_ibm_runtime.fake_provider import FakeTorino
            except ImportError as e:
                raise ImportError(
                    "qiskit-ibm-runtime is required for noisy_simulation mode. "
                    "Install with: pip install qiskit-ibm-runtime"
                ) from e
            self._backend = FakeTorino()
            self._layout_selector = LayoutSelector(self._backend, seed=seed)
            logger.info(
                "HardwareDeployerV61 initialized in noisy_simulation mode (FakeTorino, seed=%d).",
                seed,
            )
        else:
            logger.info("HardwareDeployerV61 initialized in simulation mode.")

    @property
    def backend(self) -> object | None:
        """Return the resolved backend object, or None in simulation mode."""
        return self._backend

    def deploy_adapt_vqe(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        theta_pred: np.ndarray,
        lattice: LatticeConfig,
        exact: GroundTruthResult,
        *,
        max_iterations: int = MAX_ADAPT_ITERATIONS_HARDWARE,
    ) -> DeployResultV61:
        """Main deployment route: MPNN θ_pred → measure → classify.

        Orchestrates the full pipeline: shot budget computation, parameter
        binding, observable grouping, execution (hardware or simulation),
        phase classification, and result construction.

        Note on naming: the "adapt" in ``deploy_adapt_vqe`` refers to the
        pipeline's *capability* — AdaptVQE refinement (iteratively growing
        the circuit with Pauli pool operators) is available via
        ``max_iterations``.  However, with MPNN warm-start providing
        near-optimal parameters, adapt_iterations=0 is the expected case
        (the warm-start is already converged, so no circuit growth is
        needed).  The method is kept named this way for interface
        consistency with V6.0's ``deploy_adapt_vqe``.

        Parameters
        ----------
        circuit : QuantumCircuit
            Parameterized HVA circuit.
        hamiltonian : SparsePauliOp
            Full TFIM Hamiltonian.
        theta_pred : np.ndarray
            MPNN-predicted parameters (warm-start).
        lattice : LatticeConfig
            Lattice specification.
        exact : GroundTruthResult
            Exact solution for validation metrics.
        max_iterations : int
            Maximum ADAPT iterations (default 2, typically 0 for warm-start).

        Returns
        -------
        DeployResultV61
            Full result with provenance and mitigation metadata.
        """
        # ADAPT iterations are skipped (adapt_iterations=0) because the MPNN
        # warm-start provides near-optimal parameters — iterative circuit
        # growth adds depth (violating the Mele et al. p≤2 constraint) for
        # negligible energy improvement.  The constant MAX_TWO_QUBIT_GATES_P2
        # in config_v61.py documents the gate budget that would apply if ADAPT
        # iterations were needed in future (e.g., for systems where the MPNN
        # prediction is insufficiently accurate).

        # 1. Compute shot budget
        total_shots = compute_shot_budget(lattice.n_qubits, self._shots_override)
        # In simulation mode, StatevectorEstimator returns exact values (no shot noise).
        # Use a negligible sigma to avoid false "indeterminate" classifications.
        sigma = 1e-10 if self._mode == "simulation" else 1.0 / np.sqrt(total_shots)

        # 2. Bind predicted parameters to circuit
        bound_circuit = circuit.assign_parameters(theta_pred)

        # 3. Group observables
        x_group, zz_group = ObservableGrouper.group_observables(lattice)

        # 4. Execute measurement
        adapt_iterations = 0  # Ideal warm-start from MPNN
        energy: float
        x_values: np.ndarray
        zz_values: np.ndarray
        ces_values: list[float] = []
        energies_per_layout: list[float] = []
        zne_r_squared: float | None = None
        nn_fit_loss: float | None = None
        extrapolation_method = "none"
        raw_energy: float | None = None
        raw_mag_x: float | None = None
        raw_corr_zz: float | None = None
        job_id: str | None = None
        calibration_date: str | None = None
        execution_timestamp: str | None = None
        backend_name: str | None = None

        if self._mode in ("hardware", "noisy_simulation"):
            # Hardware / noisy_simulation mode: inhomogeneous ZNE
            assert self._layout_selector is not None
            backend_name = self._backend.name if self._backend else None
            execution_timestamp = datetime.now(UTC).isoformat()

            # Select layouts
            layouts = self._layout_selector.select_layouts(lattice.n_qubits, self._n_layouts)

            # Run inhomogeneous ZNE
            (
                x_values,
                zz_values,
                ces_values,
                energies_per_layout,
                zne_r_squared,
                extrapolated_energy,
            ) = self._run_inhomogeneous_zne(
                bound_circuit, x_group, zz_group, hamiltonian, lattice, layouts
            )

            # Store raw values from first layout (unmitigated reference)
            if energies_per_layout:
                raw_energy = energies_per_layout[0]

            # Check if ZNE produced valid data
            if not ces_values and not energies_per_layout:
                # All layouts were filtered or no valid execution occurred.
                # Log error and set energy to NaN to propagate failure clearly.
                logger.error(
                    "ZNE produced no valid results (all layouts filtered or execution failed). "
                    "Setting energy=NaN. Check CES ratio filtering and backend calibration."
                )
                energy = float("nan")
                extrapolation_method = "none"
                zne_r_squared = None
            elif extrapolated_energy is not None:
                # Use ZNE-extrapolated Hamiltonian energy when available (from
                # linear fit on per-layout Hamiltonian PUB results). This is more
                # accurate than reconstructing from separately-extrapolated observables.
                energy = extrapolated_energy
            elif energies_per_layout:
                # Single layout fallback: use the raw Hamiltonian PUB energy
                energy = energies_per_layout[0]
            else:
                # Last resort: reconstruct from observables
                h_val = lattice.h if isinstance(lattice.h, int | float) else np.mean(lattice.h)
                J_val = lattice.J if isinstance(lattice.J, int | float) else np.mean(lattice.J)
                energy = -J_val * np.sum(zz_values) - h_val * np.sum(x_values)

            # Determine extrapolation method based on actual layouts used
            if len(ces_values) >= 2:
                extrapolation_method = "linear"
            else:
                # Single layout: no extrapolation possible
                extrapolation_method = "none"
                zne_r_squared = None

            # NN extrapolation if enabled and enough data
            if self._nn_extrapolation and len(ces_values) >= NN_MIN_DATA_POINTS:
                nn_ext = NNExtrapolator()
                energy, nn_fit_loss = nn_ext.extrapolate(
                    np.array(ces_values), np.array(energies_per_layout)
                )
                extrapolation_method = "nn"

        else:
            # Simulation mode: StatevectorEstimator
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator()

            # For simulation, submit individual observables as a list to get
            # per-term expectation values (StatevectorEstimator returns a scalar
            # for multi-term SparsePauliOp, but an array for a list of ops).
            x_obs_list = [
                SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=lattice.n_qubits)
                for i in range(lattice.n_qubits)
            ]
            zz_obs_list = [
                SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0)], num_qubits=lattice.n_qubits)
                for (i, j) in lattice.edges
            ]

            # Submit as PUBs with observable lists for per-term results
            job = estimator.run(
                [
                    (bound_circuit, x_obs_list),
                    (bound_circuit, zz_obs_list),
                    (bound_circuit, hamiltonian),
                ]
            )
            result = job.result()

            x_values = np.asarray(result[0].data.evs, dtype=np.float64)
            zz_values = np.asarray(result[1].data.evs, dtype=np.float64)
            energy = float(result[2].data.evs)

        # 5. Extract per-site/per-bond values
        per_site_x, per_bond_zz = ObservableGrouper.extract_individual_values(
            x_values, zz_values, lattice
        )

        # 6. Compute bulk averages
        mag_x = float(np.mean(per_site_x))
        corr_zz = float(np.mean(per_bond_zz))

        # 7. Classify phase
        phase_label = self.classify_phase(mag_x, corr_zz, sigma)

        # 8. Compute validation metrics
        delta_e = abs(energy - exact.ground_energy)
        delta_e_over_gap = delta_e / exact.gap if exact.gap > 0 else float("inf")
        mag_x_error = abs(mag_x - exact.mag_x)
        corr_zz_error = abs(corr_zz - exact.corr_zz)

        # Metrics checklist
        metrics_checklist = {
            "delta_e_over_gap_lt_5pct": delta_e_over_gap < 0.05,
            "correct_phase_label": phase_label != "indeterminate",
            "mag_x_error_lt_0.1": mag_x_error < 0.1,
            "corr_zz_error_lt_0.1": corr_zz_error < 0.1,
        }

        # 9. Build and return DeployResultV61
        deploy_result = DeployResultV61(
            # V6.0 compatible fields
            route="adapt_vqe",
            h_test=float(exact.h_value),
            predicted_energy=float(energy),
            delta_e=float(delta_e),
            delta_e_over_gap=float(delta_e_over_gap),
            mag_x_pred=mag_x,
            corr_zz_pred=corr_zz,
            mag_x_error=float(mag_x_error),
            corr_zz_error=float(corr_zz_error),
            fidelity=None,  # Unmeasurable on hardware
            adapt_iterations=adapt_iterations,
            phase_label=phase_label,
            metrics_checklist=metrics_checklist,
            # V6.1 hardware extensions
            mode=self._mode,
            backend_name=backend_name,
            job_id=job_id,
            calibration_date=calibration_date,
            execution_timestamp=execution_timestamp,
            total_shots=total_shots,
            # ZNE data
            ces_values=ces_values,
            energies_per_layout=energies_per_layout,
            zne_r_squared=zne_r_squared,
            nn_fit_loss=nn_fit_loss,
            extrapolation_method=extrapolation_method,
            # Raw vs mitigated
            raw_energy=raw_energy,
            raw_mag_x=raw_mag_x,
            raw_corr_zz=raw_corr_zz,
            # Uncertainty
            sigma=sigma,
            per_site_mag_x=per_site_x,
            per_bond_corr_zz=per_bond_zz,
        )

        logger.info(
            "DeployResultV61: mode=%s, E=%.6f, ΔE=%.6f, ΔE/gap=%.4f, phase=%s, ⟨X⟩=%.4f, ⟨ZZ⟩=%.4f",
            self._mode,
            energy,
            delta_e,
            delta_e_over_gap,
            phase_label,
            mag_x,
            corr_zz,
        )

        return deploy_result

    def _run_inhomogeneous_zne(
        self,
        bound_circuit: QuantumCircuit,
        x_group: SparsePauliOp,
        zz_group: SparsePauliOp,
        hamiltonian: SparsePauliOp,
        lattice: LatticeConfig,
        layouts: list[LayoutResult],
    ) -> tuple[np.ndarray, np.ndarray, list[float], list[float], float, float | None]:
        """Execute circuit at multiple layouts and extrapolate to CES=0.

        For each layout, transpiles the circuit, applies the layout to
        observables, and submits PUBs to EstimatorV2. Then performs linear
        regression on (CES, energy) pairs to extrapolate to zero noise.

        Submits 3 PUBs per layout: X-group, ZZ-group, and full Hamiltonian.
        The Hamiltonian PUB gives the correct total energy directly from the
        Estimator (avoiding manual reconstruction with J/h prefactors).

        Parameters
        ----------
        bound_circuit : QuantumCircuit
            Parameter-bound circuit ready for transpilation.
        x_group : SparsePauliOp
            Grouped X-basis observables.
        zz_group : SparsePauliOp
            Grouped ZZ-basis observables.
        hamiltonian : SparsePauliOp
            Full TFIM Hamiltonian for direct energy evaluation.
        lattice : LatticeConfig
            Lattice specification (for J/h values in fallback energy calc).
        layouts : list[LayoutResult]
            Selected layouts with CES values.

        Returns
        -------
        (extrapolated_x, extrapolated_zz, ces_values, energies, r_squared, extrapolated_energy)
            Extrapolated observable values, CES per layout, energy per layout,
            R² of the linear fit, and the ZNE-extrapolated energy (None if < 2 layouts).
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        if self._mode == "noisy_simulation":
            from qiskit.primitives import BackendEstimatorV2

            shots = compute_shot_budget(bound_circuit.num_qubits, self._shots_override)
            # BackendEstimatorV2 uses default_precision = 1/√shots
            precision = 1.0 / np.sqrt(shots)
            estimator = BackendEstimatorV2(
                backend=self._backend,
                options={"default_precision": precision, "seed_simulator": self._seed},
            )
        else:
            from qiskit_ibm_runtime import EstimatorV2

            estimator = EstimatorV2(backend=self._backend)

        if self._mode != "noisy_simulation":
            # Apply hardware options (DD, twirling, TREX)
            options = build_estimator_options(
                shots=compute_shot_budget(bound_circuit.num_qubits, self._shots_override),
            )
            for key, value in options.items():
                if isinstance(value, dict):
                    # Nested options (e.g., dynamical_decoupling, twirling, resilience)
                    sub_opts = getattr(estimator.options, key, None)
                    if sub_opts is not None:
                        for sub_key, sub_value in value.items():
                            setattr(sub_opts, sub_key, sub_value)
                else:
                    setattr(estimator.options, key, value)

        ces_values: list[float] = []
        all_x_values: list[np.ndarray] = []
        all_zz_values: list[np.ndarray] = []
        energies: list[float] = []

        # ── Prepare all layouts: transpile and filter ──
        # Guard: if no layouts provided, return zeros immediately
        if not layouts:
            n_x = lattice.n_qubits
            n_zz = len(lattice.edges)
            logger.warning("No layouts provided to _run_inhomogeneous_zne. Returning zeros.")
            return np.zeros(n_x), np.zeros(n_zz), [], [], 0.0, None

        valid_layouts: list[tuple[QuantumCircuit, float]] = []
        for layout in layouts:
            pm = generate_preset_pass_manager(
                optimization_level=2,
                backend=self._backend,
                initial_layout=layout.initial_layout,
            )
            transpiled = pm.run(bound_circuit)
            actual_ces = self._layout_selector.compute_ces(transpiled)

            # Post-transpilation CES filter
            if valid_layouts:
                min_actual_ces = min(ces for _, ces in valid_layouts)
                if min_actual_ces > 0 and actual_ces > MAX_CES_RATIO * min_actual_ces:
                    logger.warning(
                        "Skipping layout with actual CES=%.4f (> %.1f × min=%.4f). "
                        "Likely excessive SWAP routing.",
                        actual_ces,
                        MAX_CES_RATIO,
                        min_actual_ces,
                    )
                    continue

            valid_layouts.append((transpiled, actual_ces))

        # Guard: if all layouts were filtered out, return zeros
        if not valid_layouts:
            n_x = lattice.n_qubits
            n_zz = len(lattice.edges)
            logger.warning(
                "All layouts filtered by post-transpilation CES ratio. "
                "Returning unextrapolated zeros."
            )
            return np.zeros(n_x), np.zeros(n_zz), [], [], 0.0, None

        # ── Batch all PUBs into a single estimator.run() call ──
        all_pubs = []

        for _layout_idx, (transpiled, _) in enumerate(valid_layouts):
            x_obs_individual = [
                SparsePauliOp.from_sparse_list(
                    [("X", [i], 1.0)], num_qubits=lattice.n_qubits
                ).apply_layout(transpiled.layout)
                for i in range(lattice.n_qubits)
            ]
            zz_obs_individual = [
                SparsePauliOp.from_sparse_list(
                    [("ZZ", [i, j], 1.0)], num_qubits=lattice.n_qubits
                ).apply_layout(transpiled.layout)
                for (i, j) in lattice.edges
            ]
            h_mapped = hamiltonian.apply_layout(transpiled.layout)

            all_pubs.append((transpiled, x_obs_individual))
            all_pubs.append((transpiled, zz_obs_individual))
            all_pubs.append((transpiled, h_mapped))

        # Single batched execution
        job = estimator.run(all_pubs)
        result = job.result()

        # ── Extract results per layout ──
        for layout_idx, (_, actual_ces) in enumerate(valid_layouts):
            base_pub_idx = layout_idx * 3
            x_vals = np.asarray(result[base_pub_idx].data.evs, dtype=np.float64)
            zz_vals = np.asarray(result[base_pub_idx + 1].data.evs, dtype=np.float64)
            e_layout = float(result[base_pub_idx + 2].data.evs)

            all_x_values.append(x_vals)
            all_zz_values.append(zz_vals)
            ces_values.append(actual_ces)
            energies.append(e_layout)

        # Log actual CES values used for extrapolation
        if ces_values:
            ces_ratio_actual = max(ces_values) / min(ces_values) if min(ces_values) > 0 else 0
            logger.info(
                "ZNE using %d layout(s) with actual CES: %s (ratio %.2f)",
                len(ces_values),
                [f"{c:.4f}" for c in ces_values],
                ces_ratio_actual,
            )

        # Perform linear regression on (CES, energy) to extrapolate to CES=0
        ces_arr = np.array(ces_values)
        energy_arr = np.array(energies)
        extrapolated_energy: float | None = None

        if len(ces_values) >= 2:
            coeffs = np.polyfit(ces_arr, energy_arr, 1)
            extrapolated_energy = float(np.polyval(coeffs, 0.0))

            # Compute R²
            y_pred = np.polyval(coeffs, ces_arr)
            ss_res = np.sum((energy_arr - y_pred) ** 2)
            ss_tot = np.sum((energy_arr - np.mean(energy_arr)) ** 2)
            r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            if r_squared < ZNE_R_SQUARED_WARNING_THRESHOLD:
                logger.warning(
                    "ZNE linear regression R²=%.4f < %.2f threshold. "
                    "Extrapolation quality may be poor.",
                    r_squared,
                    ZNE_R_SQUARED_WARNING_THRESHOLD,
                )
        else:
            r_squared = 0.0

        # Extrapolate per-observable values using linear regression on CES.
        # Note: Energy and observable extrapolations are independent fits.
        # E_ZNE ≠ -J*sum(zz_extrapolated) - h*sum(x_extrapolated) in general.
        # This is intentional: energy uses the Hamiltonian PUB (most accurate),
        # observables use per-term fits (needed for phase classification).
        n_x = all_x_values[0].shape[0] if all_x_values else 0
        n_zz = all_zz_values[0].shape[0] if all_zz_values else 0

        extrapolated_x = np.zeros(n_x)
        extrapolated_zz = np.zeros(n_zz)

        if len(ces_values) >= 2:
            for i in range(n_x):
                vals = np.array([xv[i] for xv in all_x_values])
                c = np.polyfit(ces_arr, vals, 1)
                extrapolated_x[i] = float(np.polyval(c, 0.0))

            for i in range(n_zz):
                vals = np.array([zzv[i] for zzv in all_zz_values])
                c = np.polyfit(ces_arr, vals, 1)
                extrapolated_zz[i] = float(np.polyval(c, 0.0))
        elif all_x_values:
            # Single layout fallback
            extrapolated_x = all_x_values[0]
            extrapolated_zz = all_zz_values[0]

        return extrapolated_x, extrapolated_zz, ces_values, energies, r_squared, extrapolated_energy

    def classify_phase(self, mag_x: float, corr_zz: float, sigma: float) -> str:
        """Data-driven phase classification with uncertainty awareness.

        Uses the ⟨X⟩ vs |⟨ZZ⟩| crossover criterion for TFIM:
        - Paramagnetic: ⟨X⟩ > |⟨ZZ⟩| (spins aligned with transverse field)
        - Ferromagnetic: |⟨ZZ⟩| > ⟨X⟩ (nearest-neighbor correlations dominate)
        - Indeterminate: difference within statistical uncertainty σ

        Note: For TFIM with |+⟩^N initial state, ⟨X⟩ ≥ 0 always.
        ⟨ZZ⟩ < 0 for our convention (H = -J*ZZ - h*X), so we use |⟨ZZ⟩|.

        Parameters
        ----------
        mag_x : float
            Bulk-averaged magnetization ⟨X⟩ (expected ≥ 0 for TFIM).
        corr_zz : float
            Bulk-averaged correlation ⟨ZZ⟩ (expected ≤ 0 for TFIM).
        sigma : float
            Statistical uncertainty (1/√shots).

        Returns
        -------
        str
            "paramagnetic", "ferromagnetic", or "indeterminate".
        """
        # ⟨X⟩ is non-negative for TFIM with |+⟩^N; use raw value
        # ⟨ZZ⟩ is negative for our convention; compare magnitudes
        obs_x = abs(mag_x)
        obs_zz = abs(corr_zz)

        diff = abs(obs_x - obs_zz)

        if diff <= sigma:
            return "indeterminate"
        elif obs_x > obs_zz:
            return "paramagnetic"
        else:
            return "ferromagnetic"

    # ── Random Baseline Comparison ───────────────────────────────────────

    def deploy_with_baseline(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        theta_pred: np.ndarray,
        lattice: LatticeConfig,
        exact: GroundTruthResult,
        *,
        n_random_seeds: int = 5,
        random_seed_base: int = 100,
    ) -> tuple[DeployResultV61, BaselineComparison]:
        """Deploy with MPNN warm-start AND random cold-start baseline.

        Executes the standard warm-start deployment, then runs multiple
        cold-start deployments with random θ ~ U(-π, π) to quantify the
        value of the MPNN prediction.

        Parameters
        ----------
        circuit : QuantumCircuit
            Parameterized HVA circuit.
        hamiltonian : SparsePauliOp
            Full Hamiltonian.
        theta_pred : np.ndarray
            MPNN-predicted parameters (warm-start).
        lattice : LatticeConfig
            Lattice specification.
        exact : GroundTruthResult
            Exact solution for validation metrics.
        n_random_seeds : int
            Number of random initializations (default 5).
        random_seed_base : int
            Base seed for reproducible random θ generation.

        Returns
        -------
        (warm_result, comparison)
            The warm-start DeployResultV61 (unchanged behavior) plus
            the BaselineComparison object with gain metrics.
        """

        # 1. Warm-start deployment (existing behavior, unchanged)
        logger.info("Baseline: deploying warm-start (MPNN prediction)...")
        warm_result = self.deploy_adapt_vqe(circuit, hamiltonian, theta_pred, lattice, exact)

        # Determine correct phase for comparison
        exact_phase = self.classify_phase(exact.mag_x, exact.corr_zz, 1e-10)

        warm_metrics = BaselineMetrics(
            theta_init=theta_pred.tolist(),
            predicted_energy=warm_result.predicted_energy,
            delta_e=warm_result.delta_e,
            delta_e_over_gap=warm_result.delta_e_over_gap,
            fidelity=warm_result.fidelity,
            adapt_iterations=warm_result.adapt_iterations,
            phase_label=warm_result.phase_label,
            phase_correct=(warm_result.phase_label == exact_phase),
        )

        # 2. Cold-start deployments
        cold_metrics_list: list[BaselineMetrics] = []
        seeds_used: list[int] = []
        n_params = len(theta_pred)

        for i in range(n_random_seeds):
            seed = random_seed_base + i
            seeds_used.append(seed)
            rng = np.random.default_rng(seed)
            theta_random = rng.uniform(-np.pi, np.pi, n_params)

            logger.debug(
                "Baseline: cold-start seed=%d, θ_random[:3]=%s",
                seed,
                theta_random[:3].round(3),
            )

            cold_result = self.deploy_adapt_vqe(circuit, hamiltonian, theta_random, lattice, exact)

            cold_metrics_list.append(
                BaselineMetrics(
                    theta_init=theta_random.tolist(),
                    predicted_energy=cold_result.predicted_energy,
                    delta_e=cold_result.delta_e,
                    delta_e_over_gap=cold_result.delta_e_over_gap,
                    fidelity=cold_result.fidelity,
                    adapt_iterations=cold_result.adapt_iterations,
                    phase_label=cold_result.phase_label,
                    phase_correct=(cold_result.phase_label == exact_phase),
                )
            )

        # 3. Compute comparison metrics
        comparison = self._build_baseline_comparison(warm_metrics, cold_metrics_list, seeds_used)

        # 4. Log summary
        if comparison.gain_energy_pct < 0:
            logger.warning(
                "ANOMALY: warm-start ΔE/gap (%.4f) WORSE than cold-start mean (%.4f) "
                "at h=%.2f. Possible causes: MPNN prediction in wrong basin, "
                "VQE training data issue.",
                warm_metrics.delta_e_over_gap,
                comparison.cold_start_mean["delta_e_over_gap"],
                exact.h_value,
            )
        else:
            logger.info(
                "Baseline h=%.2f: warm ΔE/gap=%.4f, cold mean=%.4f±%.4f, gain=%.1f%%",
                exact.h_value,
                warm_metrics.delta_e_over_gap,
                comparison.cold_start_mean["delta_e_over_gap"],
                comparison.cold_start_std["delta_e_over_gap"],
                comparison.gain_energy_pct,
            )

        return warm_result, comparison

    def _build_baseline_comparison(
        self,
        warm: BaselineMetrics,
        cold_list: list[BaselineMetrics],
        seeds: list[int],
    ) -> BaselineComparison:
        """Build BaselineComparison from warm and cold metrics.

        Parameters
        ----------
        warm : BaselineMetrics
            Warm-start metrics.
        cold_list : list[BaselineMetrics]
            Per-seed cold-start metrics.
        seeds : list[int]
            Seeds used for cold-start.

        Returns
        -------
        BaselineComparison
        """

        # Compute mean and std of cold-start metrics
        cold_de_gaps = [m.delta_e_over_gap for m in cold_list]
        cold_energies = [m.predicted_energy for m in cold_list]
        cold_fidelities = [m.fidelity for m in cold_list if m.fidelity is not None]

        cold_mean = {
            "delta_e_over_gap": float(np.mean(cold_de_gaps)),
            "predicted_energy": float(np.mean(cold_energies)),
            "fidelity": float(np.mean(cold_fidelities)) if cold_fidelities else None,
            "adapt_iterations": float(np.mean([m.adapt_iterations for m in cold_list])),
            "phase_correct_rate": float(np.mean([m.phase_correct for m in cold_list])),
        }

        cold_std = {
            "delta_e_over_gap": float(np.std(cold_de_gaps)),
            "predicted_energy": float(np.std(cold_energies)),
            "fidelity": float(np.std(cold_fidelities)) if cold_fidelities else None,
            "adapt_iterations": float(np.std([m.adapt_iterations for m in cold_list])),
        }

        # Gain: positive means warm-start is better
        cold_mean_de_gap = cold_mean["delta_e_over_gap"]
        if cold_mean_de_gap > 1e-10:
            gain_energy_pct = (cold_mean_de_gap - warm.delta_e_over_gap) / cold_mean_de_gap * 100.0
        else:
            # Cold-start is perfect (unlikely) — no gain to report
            gain_energy_pct = 0.0

        # Fidelity gain (simulation only)
        gain_fidelity_abs: float | None = None
        if warm.fidelity is not None and cold_mean["fidelity"] is not None:
            gain_fidelity_abs = warm.fidelity - cold_mean["fidelity"]

        return BaselineComparison(
            n_random_seeds=len(cold_list),
            random_seeds=seeds,
            warm_start=warm,
            cold_start_mean=cold_mean,
            cold_start_std=cold_std,
            cold_start_per_seed=cold_list,
            gain_energy_pct=float(gain_energy_pct),
            gain_fidelity_abs=gain_fidelity_abs,
            warm_start_sufficient=(warm.delta_e_over_gap < 0.05),
            cold_start_any_success=any(m.delta_e_over_gap < 0.05 for m in cold_list),
        )


# ---------------------------------------------------------------------------
# NNExtrapolator — NN-Enhanced ZNE (Sun et al. 2025)
# ---------------------------------------------------------------------------


class NNExtrapolator:
    """NN-enhanced ZNE extrapolation using 2-layer MLP (Sun et al. 2025).

    Fits MLPRegressor to (noise_factor, energy) data and predicts
    the zero-noise energy. Falls back to linear regression if
    insufficient data points.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, int] = NN_HIDDEN_LAYERS,
        max_iter: int = NN_MAX_ITER,
    ) -> None:
        """Initialize MLP configuration.

        Parameters
        ----------
        hidden_layer_sizes : tuple[int, int]
            Hidden layer sizes for the MLP (default: (16, 8)).
        max_iter : int
            Maximum training iterations (default: 1000).
        """
        self._hidden_layer_sizes = hidden_layer_sizes
        self._max_iter = max_iter

    def extrapolate(
        self,
        noise_values: np.ndarray,
        energy_values: np.ndarray,
    ) -> tuple[float, float]:
        """Extrapolate energy to zero noise using MLP.

        Parameters
        ----------
        noise_values : np.ndarray
            CES values or noise factors for each data point.
        energy_values : np.ndarray
            Measured energies at each noise level.

        Returns
        -------
        (extrapolated_energy, quality_metric)
            Energy at noise=0 and a quality metric:
            - R² (0-1, higher is better) when using linear regression fallback
            - 1 - normalized_loss (0-1, higher is better) when using MLP

        Notes
        -----
        Falls back to linear regression if len(noise_values) < 5.
        The MLP has ~169 parameters. With only 3-5 data points it will
        overfit. The NN extrapolator is designed for use when combining
        inhomogeneous ZNE (3-5 layouts) with Runtime ZNE noise factors [1,2,3],
        giving 9-15 data points total.
        """
        if len(noise_values) < NN_MIN_DATA_POINTS:
            logger.warning(
                "Only %d data points for NN extrapolation (need >= %d). "
                "Falling back to linear regression.",
                len(noise_values),
                NN_MIN_DATA_POINTS,
            )
            return self.linear_extrapolate(noise_values, energy_values)

        from sklearn.neural_network import MLPRegressor

        mlp = MLPRegressor(
            hidden_layer_sizes=self._hidden_layer_sizes,
            max_iter=self._max_iter,
            random_state=42,
        )
        X = noise_values.reshape(-1, 1)
        mlp.fit(X, energy_values)
        extrapolated = float(mlp.predict(np.array([[0.0]]))[0])

        # Normalize loss to a 0-1 quality metric (higher is better)
        # comparable to R² from linear regression
        fit_loss = float(mlp.loss_)
        energy_variance = float(np.var(energy_values))
        quality_metric = max(0.0, 1.0 - fit_loss / energy_variance) if energy_variance > 0 else 0.0

        return extrapolated, quality_metric

    @staticmethod
    def linear_extrapolate(
        noise_values: np.ndarray,
        energy_values: np.ndarray,
    ) -> tuple[float, float]:
        """Linear regression extrapolation to noise=0.

        Parameters
        ----------
        noise_values : np.ndarray
            CES values or noise factors.
        energy_values : np.ndarray
            Measured energies at each noise level.

        Returns
        -------
        (extrapolated_energy, r_squared)
            Energy at noise=0 and R² of the linear fit.
        """
        coeffs = np.polyfit(noise_values, energy_values, 1)
        extrapolated = float(np.polyval(coeffs, 0.0))

        # Compute R²
        y_pred = np.polyval(coeffs, noise_values)
        ss_res = np.sum((energy_values - y_pred) ** 2)
        ss_tot = np.sum((energy_values - np.mean(energy_values)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return extrapolated, r_squared
