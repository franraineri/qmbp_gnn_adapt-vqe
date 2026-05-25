"""
LEGACY: QRC Pipeline — Quantum Reservoir Computing fallback route (no quantum training).

╔══════════════════════════════════════════════════════════════════════════╗
║  ABANDONED in V6.1 — the MPNN warm-start route proved sufficient.      ║
║  Kept for backward compatibility with:                                  ║
║    - scripts/smoke_test.py (V6.0 smoke test)                           ║
║    - scripts/benchmark_v6.py (V6.0 benchmark)                          ║
║  DO NOT use for new development.                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

Uses a fixed (un-optimized) HVA circuit as a quantum reservoir.  The reservoir
parameters are NEVER updated — only a classical linear readout is trained on
the reservoir's output features.  This eliminates barren plateaus by design.

Architecture:
  1. build_reservoir(): Fixed random HVA circuit (parameters frozen)
  2. encode_and_measure(): Rx(h) encoding → measure local observables
  3. train_readout(): Classical linear regression on reservoir features
  4. predict(): Encode unseen h → reservoir features → linear readout

References
----------
- Design doc: Component 6 (QRC Pipeline).
- Correctness Property 6: QRC No-Training Invariant.
"""

from __future__ import annotations

import logging

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.linear_model import LinearRegression

from .config import LatticeConfig

logger = logging.getLogger(__name__)


class QRCPipeline:
    """Quantum Reservoir Computing fallback for phase classification."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._reservoir_params: np.ndarray | None = None
        self._reservoir_circuit: QuantumCircuit | None = None
        self._readout_model: LinearRegression | None = None
        self._obs_x: list[SparsePauliOp] | None = None
        self._obs_zz: list[SparsePauliOp] | None = None

    # ── Task 7.1: build_reservoir() ──────────────────────────────────

    def build_reservoir(
        self,
        n_qubits: int,
        p_layers: int,
        lattice: LatticeConfig,
    ) -> QuantumCircuit:
        """Create an HVA circuit with fixed random parameters (never updated).

        Parameters
        ----------
        n_qubits : int
        p_layers : int
        lattice : LatticeConfig

        Returns
        -------
        QuantumCircuit — reservoir circuit with bound (fixed) parameters.
        """
        from .hamiltonian_builder import HamiltonianBuilder
        from .hva_builder import HVACircuitBuilder

        hva_builder = HVACircuitBuilder()
        qc, theta = hva_builder.create(n_qubits, p_layers, lattice)

        # Fixed random parameters — NEVER updated
        rng = np.random.RandomState(self._seed)
        self._reservoir_params = rng.uniform(-np.pi, np.pi, len(theta))

        # Bind parameters to create a fixed circuit
        self._reservoir_circuit = qc.assign_parameters(self._reservoir_params)

        # Build observables for feature extraction
        ham_builder = HamiltonianBuilder()
        self._obs_x, self._obs_zz = ham_builder.build_local_observables(lattice)

        logger.info(
            f"QRC reservoir built: {n_qubits} qubits, p={p_layers}, "
            f"{len(self._reservoir_params)} fixed params"
        )
        return self._reservoir_circuit

    # ── Task 7.2: encode_and_measure() ───────────────────────────────

    def encode_and_measure(
        self,
        h_value: float,
        reservoir: QuantumCircuit | None = None,
    ) -> np.ndarray:
        """Encode h via Rx(h) gates and measure local observables as features.

        Parameters
        ----------
        h_value : float — transverse field value to encode.
        reservoir : QuantumCircuit | None — uses stored reservoir if None.

        Returns
        -------
        np.ndarray — feature vector [⟨X₀⟩, ..., ⟨Xₙ⟩, ⟨ZZ₀₁⟩, ...].
        """
        res = reservoir if reservoir is not None else self._reservoir_circuit
        if res is None:
            raise RuntimeError("Reservoir not built. Call build_reservoir() first.")

        n = res.num_qubits

        # Encoding: append Rx(h) on each qubit
        encoded = res.copy()
        for i in range(n):
            encoded.rx(h_value, i)

        # Measure local observables via statevector simulation
        sv = Statevector(encoded)

        features = []
        for op in self._obs_x:
            features.append(float(sv.expectation_value(op).real))
        for op in self._obs_zz:
            features.append(float(sv.expectation_value(op).real))

        return np.array(features)

    # ── Task 7.3: train_readout() ────────────────────────────────────

    def train_readout(
        self,
        h_values: np.ndarray,
        exact_mag_x: np.ndarray,
        exact_corr_zz: np.ndarray,
    ) -> LinearRegression:
        """Train classical linear regression on reservoir features.

        Parameters
        ----------
        h_values : np.ndarray — training h-values.
        exact_mag_x : np.ndarray — exact ⟨X⟩ per h-point.
        exact_corr_zz : np.ndarray — exact ⟨ZZ⟩ per h-point.

        Returns
        -------
        LinearRegression — trained readout model.
        """
        # Build feature matrix
        features = np.array([self.encode_and_measure(h) for h in h_values])

        # Targets: [mag_x, corr_zz] per point
        targets = np.column_stack([exact_mag_x, exact_corr_zz])

        self._readout_model = LinearRegression()
        self._readout_model.fit(features, targets)

        # Task 7.4: verify reservoir params unchanged
        self._assert_reservoir_unchanged()

        r2 = self._readout_model.score(features, targets)
        logger.info(f"QRC readout trained: R²={r2:.4f} on {len(h_values)} points")

        return self._readout_model

    def predict(self, h_value: float) -> tuple[float, float]:
        """Predict observables for an unseen h value.

        Returns
        -------
        (mag_x_pred, corr_zz_pred)
        """
        if self._readout_model is None:
            raise RuntimeError("Readout not trained. Call train_readout() first.")

        features = self.encode_and_measure(h_value).reshape(1, -1)
        pred = self._readout_model.predict(features)[0]

        # Task 7.4: verify reservoir params unchanged after prediction
        self._assert_reservoir_unchanged()

        return float(pred[0]), float(pred[1])

    # ── Task 7.4: no-training invariant assertion ────────────────────

    def _assert_reservoir_unchanged(self) -> None:
        """Verify that reservoir parameters remain identical to initial values.

        Raises
        ------
        AssertionError
            If reservoir parameters have been modified.
        """
        if self._reservoir_circuit is None or self._reservoir_params is None:
            return

        # The reservoir circuit has no free parameters (all bound)
        assert self._reservoir_circuit.num_parameters == 0, (
            "QRC invariant violated: reservoir circuit has unbound parameters."
        )
        # Verify the stored params haven't been mutated
        rng = np.random.RandomState(self._seed)
        expected = rng.uniform(-np.pi, np.pi, len(self._reservoir_params))
        assert np.allclose(self._reservoir_params, expected), (
            "QRC invariant violated: reservoir parameters have been modified."
        )
