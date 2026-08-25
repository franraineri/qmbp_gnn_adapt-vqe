"""Flow-based warmstart manager for VQE parameter initialisation.

Wraps EmbeddingMAF training and inference for opt-in theta warmstart
in the hardware deployment pipeline.  The MPNN encoder is always kept
frozen (torch.no_grad throughout); only the ~584 EmbeddingMAF
parameters are trained.

Requirements: 1.5, 1.7, 6.3
"""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


def _extract_embedding(model: Any, data: Data) -> torch.Tensor:
    """Extract frozen GNN embedding from MPNNPredictor (no head).

    Runs the GNN message-passing layers + global_mean_pool under
    torch.no_grad(), producing a [1, hidden_dim] embedding vector.
    The MPNN encoder parameters are never modified.

    Parameters
    ----------
    model : MPNNPredictor
        Trained MPNN predictor (must be in eval mode).
    data : Data
        Single graph (must have x, edge_index).

    Returns
    -------
    torch.Tensor
        Embedding of shape [1, hidden_dim].
    """
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if model.use_edge_features:
            edge_attr = data.edge_attr
            for conv, norm in zip(model.convs, model.norms, strict=False):
                x = conv(x, edge_index, edge_attr)
                x = norm(x)
                x = torch.relu(x)
        else:
            for conv, norm in zip(model.convs, model.norms, strict=False):
                x = conv(x, edge_index)
                x = norm(x)
                x = torch.relu(x)

        # Global mean pooling → [1, hidden_dim]
        z = global_mean_pool(x, batch)
    return z


class FlowWarmstartManager:
    """Manages EmbeddingMAF training and inference for theta warmstart.

    The MPNN encoder is always frozen. Only the EmbeddingMAF parameters
    (~584 for default architecture) are trained via NLL minimization.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        theta_dim: int = 4,
        n_flow_layers: int = 2,
        hidden_dim: int = 32,
        n_epochs: int = 500,
        lr: float = 1e-3,
        patience: int = 0,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.theta_dim = theta_dim
        self.n_flow_layers = n_flow_layers
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.patience = patience

        self.flow_model: Any | None = None
        self.is_trained: bool = False
        self._encoder: Any | None = None

    def train(self, model: Any, dataset: list[Data]) -> dict[str, Any]:
        """Train EmbeddingMAF on frozen GNN embeddings.

        Parameters
        ----------
        model : MPNNPredictor
            Trained MPNN predictor (kept frozen throughout).
        dataset : list[Data]
            Training graphs with y tensors as theta targets.

        Returns
        -------
        dict with keys:
            nll_history: list[float] — per-epoch mean NLL
            final_nll: float — last epoch NLL

        Raises
        ------
        ValueError
            If dataset is empty.
        """
        if len(dataset) == 0:
            raise ValueError("dataset is empty")

        from qmbp_simulation.analysis.normalizing_flow import EmbeddingMAF

        # Extract embeddings (frozen) for all training graphs
        model.eval()
        embeddings: list[torch.Tensor] = []
        thetas: list[torch.Tensor] = []
        for data in dataset:
            z = _extract_embedding(model, data)
            embeddings.append(z)
            thetas.append(data.y.unsqueeze(0) if data.y.dim() == 1 else data.y)

        z_train = torch.cat(embeddings, dim=0)  # [N, embedding_dim]
        theta_train = torch.cat(thetas, dim=0)  # [N, theta_dim]

        # Build flow model
        flow = EmbeddingMAF(
            embedding_dim=self.embedding_dim,
            theta_dim=self.theta_dim,
            n_flow_layers=self.n_flow_layers,
            hidden_dim=self.hidden_dim,
        )
        optimizer = torch.optim.Adam(flow.parameters(), lr=self.lr)

        nll_history: list[float] = []
        best_nll = float("inf")
        patience_counter = 0

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            log_probs = flow.log_prob(theta_train, z_train)
            nll = -log_probs.mean()
            nll.backward()
            optimizer.step()

            epoch_nll = float(nll.item())
            nll_history.append(epoch_nll)

            # Early stopping (patience > 0 enables it)
            if self.patience > 0:
                if epoch_nll < best_nll - 1e-6:
                    best_nll = epoch_nll
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        self.flow_model = flow
        self.is_trained = True
        self._encoder = model  # store reference for sample()

        return {
            "nll_history": nll_history,
            "final_nll": nll_history[-1],
        }

    def sample(self, graph: Data, n_samples: int = 50) -> tuple[torch.Tensor, float]:
        """Sample theta vectors from the trained flow conditioned on graph.

        Parameters
        ----------
        graph : Data
            Input graph to condition on (embedding extracted frozen).
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        (theta_samples, sigma_flow)
            theta_samples: Tensor [n_samples, theta_dim], clamped to [-π, π].
            sigma_flow: float — mean per-dimension std of the samples.

        Raises
        ------
        RuntimeError
            If the manager has not been trained yet.
        """
        if not self.is_trained and self.flow_model is None:
            raise RuntimeError("FlowWarmstartManager has not been trained. Call train() first.")

        if not hasattr(self, "_encoder") or self._encoder is None:
            raise RuntimeError("FlowWarmstartManager has not been trained. Call train() first.")

        z = _extract_embedding(self._encoder, graph)
        theta_samples = self.flow_model.sample(z, n_samples=n_samples)
        sigma_flow = float(theta_samples.std(dim=0).mean().item())
        return theta_samples, sigma_flow

    def sample_topk(
        self, graph: Data, n_samples: int = 50, k: int = 5
    ) -> tuple[torch.Tensor, float]:
        """Sample and return top-k by log-probability.

        Parameters
        ----------
        graph : Data
            Input graph to condition on.
        n_samples : int
            Total samples to draw before filtering.
        k : int
            Number of top samples to return.

        Returns
        -------
        (top_samples, sigma_flow)
            top_samples: Tensor [min(k, n_samples), theta_dim].
            sigma_flow: float — std of ALL samples (before filtering).
        """
        theta_samples, sigma_flow = self.sample(graph, n_samples=n_samples)

        # Rank by log_prob and take top-k
        z = _extract_embedding(self._encoder, graph)
        with torch.no_grad():
            log_probs = self.flow_model.log_prob(theta_samples, z.expand(n_samples, -1))

        actual_k = min(k, n_samples)
        _, top_indices = torch.topk(log_probs, actual_k)
        return theta_samples[top_indices], sigma_flow

    def trainable_param_count(self) -> int:
        """Return trainable parameter count of the flow model.

        Raises
        ------
        RuntimeError
            If no flow_model is set.
        """
        if self.flow_model is None:
            raise RuntimeError(
                "FlowWarmstartManager has not been trained. "
                "Call train() or assign flow_model first."
            )
        return self.flow_model.trainable_param_count()

    def save(self, path: str) -> None:
        """Save trained flow model to disk.

        Parameters
        ----------
        path : str
            File path for the checkpoint (.pt).

        Raises
        ------
        RuntimeError
            If the manager has not been trained.
        """
        if not self.is_trained or self.flow_model is None:
            raise RuntimeError("FlowWarmstartManager has not been trained. Call train() first.")
        checkpoint = {
            "flow_state_dict": self.flow_model.state_dict(),
            "config": {
                "embedding_dim": self.embedding_dim,
                "theta_dim": self.theta_dim,
                "n_flow_layers": self.n_flow_layers,
                "hidden_dim": self.hidden_dim,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "patience": self.patience,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, model: Any) -> FlowWarmstartManager:
        """Load a trained FlowWarmstartManager from checkpoint.

        Parameters
        ----------
        path : str
            Path to saved .pt checkpoint.
        model : MPNNPredictor
            Encoder model (used for embedding extraction in sample()).

        Returns
        -------
        FlowWarmstartManager
            Loaded and ready-to-sample manager.
        """
        from qmbp_simulation.analysis.normalizing_flow import EmbeddingMAF

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # nosec: trusted checkpoint
        config = checkpoint["config"]

        mgr = cls(
            embedding_dim=config["embedding_dim"],
            theta_dim=config["theta_dim"],
            n_flow_layers=config["n_flow_layers"],
            hidden_dim=config["hidden_dim"],
            n_epochs=config["n_epochs"],
            lr=config["lr"],
            patience=config.get("patience", 0),
        )

        flow = EmbeddingMAF(
            embedding_dim=config["embedding_dim"],
            theta_dim=config["theta_dim"],
            n_flow_layers=config["n_flow_layers"],
            hidden_dim=config["hidden_dim"],
        )
        flow.load_state_dict(checkpoint["flow_state_dict"])
        flow.eval()

        mgr.flow_model = flow
        mgr.is_trained = True
        mgr._encoder = model
        return mgr
