"""Flow-based warmstart manager for VQE parameter initialisation.

Wraps EmbeddingMAF training and inference for opt-in theta warmstart
in the hardware deployment pipeline.  The MPNN encoder is always
kept frozen (torch.no_grad throughout); only the ~584 EmbeddingMAF
parameters are trained.

Requirements: 1.5, 1.7, 6.3
"""

from __future__ import annotations

import logging

import torch

from qmbp_simulation.analysis.normalizing_flow import EmbeddingMAF

logger = logging.getLogger(__name__)


def _extract_embedding(model: torch.nn.Module, data) -> torch.Tensor:
    """Extract GNN embedding (global_mean_pool output) from MPNNPredictor.

    Runs model.convs + model.norms + global_mean_pool under
    torch.no_grad().  This mirrors MPNNPredictor.forward() up to but
    not including the final MLP head, so the MPNN encoder parameters
    are never touched.

    Parameters
    ----------
    model : torch.nn.Module
        Trained MPNNPredictor instance.  Must expose public attributes
        convs (nn.ModuleList) and norms (nn.ModuleList).
    data : torch_geometric.data.Data
        Single graph with attributes x (node features) and edge_index.

    Returns
    -------
    torch.Tensor
        Shape [1, hidden_dim] -- the graph-level embedding.

    Requirements: 1.1, 1.6
    """
    # Guard: validate model has required attributes for embedding extraction
    if not hasattr(model, "convs") or not hasattr(model, "norms"):
        raise AttributeError(
            f"_extract_embedding requires model with 'convs' and 'norms' attributes "
            f"(MPNNPredictor). Got {type(model).__name__} with attributes: "
            f"{[a for a in dir(model) if not a.startswith('_')][:10]}..."
        )
    # Guard: validate data has required graph structure
    if not hasattr(data, "x") or data.x is None:
        raise ValueError(
            "_extract_embedding: graph data is missing 'x' (node features). "
            "Ensure the graph has been properly constructed with node features."
        )
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError(
            "_extract_embedding: graph data is missing 'edge_index'. "
            "Ensure the graph has been properly constructed with edge connectivity."
        )
    x, edge_index = data.x, data.edge_index
    # Ensure tensors are on the same device as the model
    device = next(model.parameters()).device
    x = x.to(device)
    edge_index = edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        for conv, norm in zip(model.convs, model.norms, strict=False):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
        from torch_geometric.nn import global_mean_pool

        z = global_mean_pool(x, batch)  # [1, hidden_dim]
    return z


class FlowWarmstartManager:
    """Wraps EmbeddingMAF training and inference for opt-in warmstart.

    The MPNN encoder is always frozen (torch.no_grad throughout).
    Only EmbeddingMAF parameters (~584) are trained.

    Parameters
    ----------
    embedding_dim : int
        GNN hidden dimension (must match MPNNPredictor.hidden_dim).
    theta_dim : int
        Number of VQE parameters (2 * p_layers).
    n_flow_layers : int
        MAF layers (default 2, matching EmbeddingMAF default).
    hidden_dim : int
        MAF hidden units (default 32).
    n_epochs : int
        NLL training epochs (default 500).
    lr : float
        Adam learning rate (default 1e-3).
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
        self.patience = patience  # 0 = disabled, >0 = early stop if no NLL improvement
        self.flow_model = None  # EmbeddingMAF | None
        self.is_trained: bool = False

    def train(
        self,
        mpnn_model: torch.nn.Module,
        dataset: list,
        seed: int | None = None,
    ) -> dict:
        """Extract frozen GNN embeddings; train EmbeddingMAF via NLL.

        Parameters
        ----------
        mpnn_model : torch.nn.Module
            Trained MPNNPredictor instance.  Parameters are never modified.
        dataset : list[torch_geometric.data.Data]
            Graph dataset. Each item must have .y (theta_opt) and graph structure.
        seed : int | None
            If provided, calls torch.manual_seed(seed) before instantiating
            EmbeddingMAF and starting the training loop. Enables reproducibility
            and multi-seed comparison.

        Returns
        -------
        dict
            Keys: ``"nll_history"`` (list[float]) and ``"final_nll"`` (float).

        Side effects
        ------------
        Sets ``self.flow_model`` (EmbeddingMAF) and ``self.is_trained = True``.

        Raises
        ------
        ValueError
            If ``dataset`` is empty.

        Requirements: 1.1, 1.2, 1.6
        """
        if len(dataset) == 0:
            raise ValueError("dataset is empty")

        # Store mpnn_model reference so sample() can re-extract embeddings later
        self._mpnn_model = mpnn_model

        # --- Guard: ensure mpnn_model is in eval mode (frozen encoder) ---
        if mpnn_model.training:
            logger.warning(
                "FlowWarmstartManager.train(): mpnn_model is in training mode. "
                "Switching to eval() to ensure frozen encoder (no dropout/BN drift)."
            )
            mpnn_model.eval()

        # --- Guard: validate embedding_dim matches model hidden_dim ---
        if hasattr(mpnn_model, "hidden_dim"):
            actual_hidden = mpnn_model.hidden_dim
            if actual_hidden != self.embedding_dim:
                raise ValueError(
                    f"FlowWarmstartManager embedding_dim={self.embedding_dim} does not "
                    f"match MPNNPredictor.hidden_dim={actual_hidden}. "
                    f"Use FlowWarmstartManager(embedding_dim={actual_hidden}, ...)"
                )

        # --- Extract frozen embeddings (encoder never touched) ---
        embeddings: list[torch.Tensor] = []
        thetas: list[torch.Tensor] = []
        with torch.no_grad():
            for i, data in enumerate(dataset):
                # Guard: check data.y exists (theta_opt target)
                if not hasattr(data, "y") or data.y is None:
                    raise ValueError(
                        f"dataset[{i}] is missing .y attribute (theta_opt). "
                        f"Each graph in the dataset must have data.y set to the "
                        f"VQE-optimized parameters for flow training."
                    )
                z = _extract_embedding(mpnn_model, data)  # [1, embedding_dim]
                embeddings.append(z)
                thetas.append(data.y.unsqueeze(0))  # [1, theta_dim]

        Z = torch.cat(embeddings, dim=0)  # [N, embedding_dim]
        T = torch.cat(thetas, dim=0)  # [N, theta_dim]

        # --- Set seed for reproducibility (if provided) ---
        if seed is not None:
            torch.manual_seed(seed)

        # --- Instantiate flow model ---
        self.flow_model = EmbeddingMAF(
            embedding_dim=self.embedding_dim,
            theta_dim=self.theta_dim,
            n_flow_layers=self.n_flow_layers,
            hidden_dim=self.hidden_dim,
        )

        # --- NLL training loop (with optional early stopping) ---
        optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.lr)
        nll_history: list[float] = []
        best_nll = float("inf")
        epochs_no_improve = 0

        for _epoch in range(self.n_epochs):
            log_prob = self.flow_model.log_prob(T, Z)  # [N]
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nll_val = loss.item()
            nll_history.append(nll_val)

            # Early stopping check
            if self.patience > 0:
                if nll_val < best_nll - 1e-4:
                    best_nll = nll_val
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        logger.info(
                            "FlowWarmstartManager early stop at epoch %d "
                            "(patience=%d, best NLL=%.4f)",
                            _epoch + 1,
                            self.patience,
                            best_nll,
                        )
                        break

        self.is_trained = True
        logger.info(
            "FlowWarmstartManager trained for %d epochs; final NLL=%.4f",
            self.n_epochs,
            nll_history[-1],
        )
        return {"nll_history": nll_history, "final_nll": float(nll_history[-1])}

    def sample(
        self,
        graph_data,
        n_samples: int = 50,
    ) -> tuple[torch.Tensor, float]:
        """Sample theta ~ p(theta | z_frozen) for one graph.

        Parameters
        ----------
        graph_data : torch_geometric.data.Data
            Single graph. Embedding extracted with torch.no_grad().
        n_samples : int
            Number of flow samples to draw (default 50).

        Returns
        -------
        tuple[torch.Tensor, float]
            theta_samples : Tensor of shape [n_samples, theta_dim],
                            all values clamped to [-pi, pi].
            sigma_flow : float
                samples.std(dim=0).mean().item() — flow uncertainty scalar.

        Raises
        ------
        RuntimeError
            If called before train().

        Requirements: 1.3, 1.4, 7.2, 7.4
        """
        if not self.is_trained:
            raise RuntimeError("FlowWarmstartManager has not been trained. Call train() first.")
        # Ensure MPNN is in eval mode for deterministic embeddings
        if self._mpnn_model.training:
            logger.warning(
                "FlowWarmstartManager.sample(): mpnn_model was in training mode. "
                "Switching to eval() for deterministic embedding extraction."
            )
            self._mpnn_model.eval()
        z_frozen = _extract_embedding(self._mpnn_model, graph_data)  # [1, embedding_dim]
        self._last_z = z_frozen
        samples = self.flow_model.sample(z_frozen, n_samples)  # [n_samples, theta_dim]
        sigma_flow: float = samples.std(dim=0).mean().item()
        return (samples, sigma_flow)

    def trainable_param_count(self) -> int:
        """Delegate to EmbeddingMAF.trainable_param_count().

        Raises
        ------
        RuntimeError
            If called before train() has been called.
        """
        if self.flow_model is None:
            raise RuntimeError("FlowWarmstartManager has not been trained. Call train() first.")
        return self.flow_model.trainable_param_count()

    def sample_topk(
        self,
        graph_data,
        n_samples: int = 50,
        k: int = 1,
    ) -> tuple[torch.Tensor, float]:
        """Sample and return the top-k samples by log-probability.

        Convenience method that combines sample() + best-selection logic
        used by the §10 runner. Returns the k samples with highest
        log p(θ|z) and the associated σ_flow.

        Parameters
        ----------
        graph_data : torch_geometric.data.Data
            Single graph for embedding extraction.
        n_samples : int
            Total samples to draw (default 50).
        k : int
            Number of top samples to return (default 1).

        Returns
        -------
        tuple[torch.Tensor, float]
            top_samples : Tensor of shape [k, theta_dim].
            sigma_flow : float (computed from all n_samples).
        """
        samples, sigma_flow = self.sample(graph_data, n_samples=n_samples)
        log_probs = self.flow_model.log_prob(
            samples,
            self._last_z.expand(samples.shape[0], -1),
        )
        topk_idx = log_probs.topk(min(k, n_samples)).indices
        return samples[topk_idx], sigma_flow

    def save(self, path: str | Path) -> None:
        """Save trained flow model and config to disk.

        Parameters
        ----------
        path : str or Path
            File path to save the checkpoint (.pt format).

        Raises
        ------
        RuntimeError
            If called before train().
        """
        from pathlib import Path as _Path

        if not self.is_trained or self.flow_model is None:
            raise RuntimeError("FlowWarmstartManager has not been trained. Call train() first.")
        save_path = _Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "flow_model_state_dict": self.flow_model.state_dict(),
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
        torch.save(checkpoint, save_path)
        logger.info("FlowWarmstartManager saved to %s", save_path)

    def train_multi_seed(
        self,
        mpnn_model: torch.nn.Module,
        dataset: list,
        seeds: list[int] | None = None,
    ) -> dict:
        """Train with multiple seeds, keep best (lowest final NLL).

        Since ``train()`` always reinitializes the flow model, we can call
        it repeatedly with different ``torch.manual_seed(seed)`` and track
        which seed produces the lowest final NLL. After all seeds are tried,
        the manager retrains with the best seed so ``self.flow_model`` holds
        the optimal weights.

        Parameters
        ----------
        mpnn_model : torch.nn.Module
            Trained MPNNPredictor (frozen during training).
        dataset : list
            Graph dataset with ``.y`` attributes (theta_opt targets).
        seeds : list[int] | None
            Seeds to try. Defaults to [42, 43, 44].

        Returns
        -------
        dict
            Keys: ``"best_seed"``, ``"best_nll"``, ``"all_results"``
            (list of {seed, final_nll}), and the full ``train()`` return
            dict for the best run (``"nll_history"``, ``"final_nll"``).
        """
        if seeds is None:
            seeds = [42, 43, 44]

        all_results: list[dict] = []
        best_seed = seeds[0]
        best_nll = float("inf")

        for seed in seeds:
            torch.manual_seed(seed)
            info = self.train(mpnn_model, dataset, seed=seed)
            nll = info["final_nll"]
            all_results.append({"seed": seed, "final_nll": nll})
            logger.info("  [multi-seed] seed=%d → final_nll=%.4f", seed, nll)
            if nll < best_nll:
                best_nll = nll
                best_seed = seed

        # Retrain with best seed to ensure self.flow_model holds optimal weights
        torch.manual_seed(best_seed)
        best_info = self.train(mpnn_model, dataset, seed=best_seed)
        logger.info(
            "  [multi-seed] Best seed=%d (NLL=%.4f), retrained.",
            best_seed,
            best_info["final_nll"],
        )

        return {
            "best_seed": best_seed,
            "best_nll": best_info["final_nll"],
            "all_results": all_results,
            "nll_history": best_info["nll_history"],
            "final_nll": best_info["final_nll"],
        }

    @classmethod
    def load(cls, path: str | Path, mpnn_model: torch.nn.Module) -> FlowWarmstartManager:
        """Load a trained FlowWarmstartManager from a checkpoint.

        Parameters
        ----------
        path : str or Path
            Path to the saved checkpoint (.pt file).
        mpnn_model : torch.nn.Module
            The MPNNPredictor instance to use for embedding extraction.

        Returns
        -------
        FlowWarmstartManager
            A ready-to-sample instance with is_trained=True.
        """
        from pathlib import Path as _Path

        checkpoint = torch.load(_Path(path), weights_only=False)
        config = checkpoint["config"]
        manager = cls(
            embedding_dim=config["embedding_dim"],
            theta_dim=config["theta_dim"],
            n_flow_layers=config["n_flow_layers"],
            hidden_dim=config["hidden_dim"],
            n_epochs=config["n_epochs"],
            lr=config["lr"],
            patience=config.get("patience", 0),
        )
        manager.flow_model = EmbeddingMAF(
            embedding_dim=config["embedding_dim"],
            theta_dim=config["theta_dim"],
            n_flow_layers=config["n_flow_layers"],
            hidden_dim=config["hidden_dim"],
        )
        manager.flow_model.load_state_dict(checkpoint["flow_model_state_dict"])
        manager.flow_model.eval()
        manager._mpnn_model = mpnn_model
        manager.is_trained = True
        logger.info("FlowWarmstartManager loaded from %s", path)
        return manager
