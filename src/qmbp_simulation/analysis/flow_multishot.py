"""Flow Multi-Shot Predictor — Energy-ranked multi-candidate θ selection.

Instead of predicting a single deterministic θ (MPNN) or filtering by
log-probability (FlowWarmstartManager.sample_topk), this module samples K
candidates from the trained normalizing flow and ranks them by actual
energy evaluation. This eliminates failures near the expressibility
boundary where the MPNN picks the wrong local minimum.

Architecture:
    FlowWarmstartManager.sample(graph, K)
        → K candidate θ vectors from learned P(θ|h)
    backend.evaluate(circuit, H, θ_k) for each k
        → K energy values
    Select argmin → best θ

Cost: K energy evaluations per h-point (vs 1 for MPNN, vs 50-200 for VQE).
Typical K=5 gives near-100% pass rate at 5× cost of MPNN (still 50-100×
faster than VQE).

Usage:
    from qmbp_simulation.analysis.flow_multishot import FlowMultiShotPredictor

    predictor = FlowMultiShotPredictor(flow_manager, K=5)
    best_theta, info = predictor.predict(graph, hamiltonian, circuit, backend)
    # info contains: all energies, sigma_flow, selected index

Integration with PipelineRunner:
    predictor = FlowMultiShotPredictor.from_checkpoint(
        flow_path="model_flow.pt", mpnn_path="model_mpnn.pt", K=5
    )
    # Or from existing manager:
    predictor = FlowMultiShotPredictor(runner.flow_manager, K=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from torch_geometric.data import Data

from qmbp_simulation.analysis.flow_warmstart import FlowWarmstartManager
from qmbp_simulation.execution import ExecutionBackend

logger = logging.getLogger(__name__)


@dataclass
class MultiShotResult:
    """Result from a multi-shot prediction at a single h-point.

    Attributes
    ----------
    best_theta : np.ndarray
        Parameter vector with lowest energy among K candidates.
    best_energy : float
        Energy of the best candidate.
    best_index : int
        Index of the selected candidate (0-based).
    all_energies : list[float]
        Energies of all K candidates.
    sigma_flow : float
        Standard deviation of the sampled distribution (uncertainty proxy).
    K : int
        Number of candidates sampled.
    energy_spread : float
        max(energies) - min(energies) — landscape ruggedness indicator.
    """

    best_theta: np.ndarray
    best_energy: float
    best_index: int
    all_energies: list[float] = field(default_factory=list)
    sigma_flow: float = 0.0
    K: int = 0
    energy_spread: float = 0.0


class FlowMultiShotPredictor:
    """Energy-ranked multi-candidate θ predictor using normalizing flows.

    Samples K candidates from a trained FlowWarmstartManager, evaluates
    each via the quantum backend, and returns the lowest-energy θ.

    Parameters
    ----------
    flow_manager : FlowWarmstartManager
        Trained flow model (contains frozen MPNN encoder + MAF flow).
    K : int
        Number of candidates to sample per h-point (default 5).
        Higher K → better energy but more evaluations.
        K=1 is equivalent to single-sample flow (stochastic MPNN).
        K=10+ gives diminishing returns for smooth landscapes.
    n_oversample : int
        Draw n_oversample from the flow, pre-filter by log-prob to K,
        then evaluate energies. Reduces wasted evaluations on clearly
        bad samples. Set to K for no pre-filtering (default: 2*K).
    fallback_to_mpnn : bool
        If True and the flow is not trained, fall back to deterministic
        MPNN prediction (K=1, no sampling). Default True.
    """

    def __init__(
        self,
        flow_manager: FlowWarmstartManager,
        K: int = 5,
        n_oversample: int | None = None,
        fallback_to_mpnn: bool = True,
    ) -> None:
        self.flow_manager = flow_manager
        self.K = K
        self.n_oversample = n_oversample or 2 * K
        self.fallback_to_mpnn = fallback_to_mpnn

    def predict(
        self,
        graph: Data,
        hamiltonian: SparsePauliOp,
        circuit: QuantumCircuit,
        backend: ExecutionBackend,
    ) -> MultiShotResult:
        """Predict best θ via K-shot energy-ranked sampling.

        Parameters
        ----------
        graph : Data
            PyG graph for this h-point (node features include h-value).
        hamiltonian : SparsePauliOp
            Hamiltonian at this h-point.
        circuit : QuantumCircuit
            Parameterized HVA circuit.
        backend : ExecutionBackend
            Backend for energy evaluation (NoiselessBackend or NoisyBackend).

        Returns
        -------
        MultiShotResult
            Contains best_theta, best_energy, and diagnostics.
        """
        # Fallback: if flow is not trained, use deterministic MPNN (K=1)
        if not self.flow_manager.is_trained and self.fallback_to_mpnn:
            logger.info("FlowMultiShot: flow not trained, falling back to MPNN (K=1)")
            from qmbp_simulation.analysis.flow_warmstart import _extract_embedding
            # Get MPNN prediction directly (no sampling)
            encoder = self.flow_manager._encoder
            if encoder is None:
                raise RuntimeError(
                    "FlowMultiShotPredictor: neither flow nor encoder available. "
                    "Train the FlowWarmstartManager first or provide a fallback model."
                )
            encoder.eval()
            with torch.no_grad():
                theta_pred = encoder(graph).numpy().flatten()
            energy = backend.evaluate(circuit, hamiltonian, theta_pred)
            return MultiShotResult(
                best_theta=theta_pred,
                best_energy=float(energy),
                best_index=0,
                all_energies=[float(energy)],
                sigma_flow=0.0,
                K=1,
                energy_spread=0.0,
            )

        # Sample candidates from flow
        if self.n_oversample > self.K:
            # Pre-filter: sample more, keep top-K by log-prob, then evaluate
            candidates, sigma_flow = self.flow_manager.sample_topk(
                graph, n_samples=self.n_oversample, k=self.K
            )
        else:
            candidates, sigma_flow = self.flow_manager.sample(graph, n_samples=self.K)

        # Guard: empty candidates
        if candidates is None or len(candidates) == 0:
            raise RuntimeError(
                "FlowMultiShot: sample returned 0 candidates. "
                "Check FlowWarmstartManager training status."
            )

        # Evaluate energy for each candidate
        candidates_np = candidates.detach().cpu().numpy()
        energies: list[float] = []

        for i in range(len(candidates_np)):
            theta_k = candidates_np[i]
            energy = backend.evaluate(circuit, hamiltonian, theta_k)
            energies.append(float(energy))

        # Select best (lowest energy)
        best_idx = int(np.argmin(energies))
        best_theta = candidates_np[best_idx]
        best_energy = energies[best_idx]
        energy_spread = max(energies) - min(energies)

        logger.debug(
            "FlowMultiShot: K=%d, best_E=%.6f, spread=%.4f, σ_flow=%.4f",
            len(energies), best_energy, energy_spread, sigma_flow,
        )

        return MultiShotResult(
            best_theta=best_theta,
            best_energy=best_energy,
            best_index=best_idx,
            all_energies=energies,
            sigma_flow=sigma_flow,
            K=len(energies),
            energy_spread=energy_spread,
        )


    def predict_sweep(
        self,
        graphs: list[Data],
        hamiltonians: list[SparsePauliOp],
        circuit: QuantumCircuit,
        backend: ExecutionBackend,
    ) -> list[MultiShotResult]:
        """Predict best θ for a sweep of h-points.

        Parameters
        ----------
        graphs : list[Data]
            One graph per h-point.
        hamiltonians : list[SparsePauliOp]
            One Hamiltonian per h-point.
        circuit : QuantumCircuit
            Shared parameterized circuit.
        backend : ExecutionBackend
            Backend for energy evaluation.

        Returns
        -------
        list[MultiShotResult]
            Results for each h-point in order.
        """
        results = []
        for i, (graph, H) in enumerate(zip(graphs, hamiltonians, strict=False)):
            result = self.predict(graph, H, circuit, backend)
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(
                    f"  FlowMultiShot sweep: {i + 1}/{len(graphs)} points, "
                    f"last E={result.best_energy:.6f}"
                )
        return results

    @classmethod
    def from_checkpoint(
        cls,
        flow_path: str,
        mpnn_path: str,
        K: int = 5,
        n_oversample: int | None = None,
    ) -> "FlowMultiShotPredictor":
        """Create predictor from saved checkpoints.

        Parameters
        ----------
        flow_path : str
            Path to saved FlowWarmstartManager (.pt file).
        mpnn_path : str
            Path to saved MPNNPredictor checkpoint.
        K : int
            Number of candidates per point.
        n_oversample : int | None
            Oversampling factor for pre-filtering.

        Returns
        -------
        FlowMultiShotPredictor
            Ready-to-use predictor.
        """
        from qmbp_simulation.predictors import load_mpnn_checkpoint

        mpnn = load_mpnn_checkpoint(mpnn_path)
        mpnn.eval()
        manager = FlowWarmstartManager.load(flow_path, model=mpnn)
        return cls(manager, K=K, n_oversample=n_oversample)

    def summary(self) -> str:
        """Human-readable configuration summary."""
        n_params = self.flow_manager.trainable_param_count()
        return (
            f"FlowMultiShotPredictor(K={self.K}, oversample={self.n_oversample}, "
            f"flow_params={n_params})"
        )
