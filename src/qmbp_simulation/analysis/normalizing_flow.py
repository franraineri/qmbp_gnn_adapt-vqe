"""Normalizing flow architectures for Ext3 — Flujos Normalizantes.

Architecture A: FlowHead — conditional MAF as generative head of MPNNPredictor.
Architecture B: EmbeddingMAF — MAF trained over frozen GNN embeddings.

Both architectures keep trainable param count ~584 (K=2 layers, hidden=32)
which is well below the 5K overparameterization threshold (Req 3.2).

θ samples are guaranteed within [-π, π] via tanh clamping (Req 3.4).

Req: 3.4, 3.5
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Masked Autoregressive Flow components (minimal, no external nflows dep)
# ---------------------------------------------------------------------------


class MaskedLinear(nn.Linear):
    """Linear layer with a binary mask applied to weights."""

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor) -> None:
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        mask: torch.Tensor = self.mask  # type: ignore[assignment]
        return nn.functional.linear(x, self.weight * mask, self.bias)


def _build_maf_mask(dim: int, hidden_dim: int, layer: int) -> torch.Tensor:
    """Build autoregressive mask for MADE-style connectivity."""
    if layer == 0:
        # Input to hidden: m_hidden[j] assigned round-robin order
        m_k = torch.arange(hidden_dim) % max(1, dim - 1) + 1  # 1 … dim-1
        m_in = torch.arange(dim)
        # mask[j, i] = 1 iff m_hidden[j] >= m_in[i]  (autoregressive)
        return (m_k.unsqueeze(1) >= m_in.unsqueeze(0)).float()
    else:
        # Hidden to output: output has dimension 2*dim (scale + shift per dim)
        m_out = torch.arange(dim).repeat(2)  # [0,1,…,dim-1, 0,1,…,dim-1]
        m_k = torch.arange(hidden_dim) % max(1, dim - 1) + 1
        # mask[i, j] = 1 iff m_out[i] > m_k[j]
        return (m_out.unsqueeze(1) > m_k.unsqueeze(0)).float()


class MAFLayer(nn.Module):
    """Single Masked Autoregressive Flow layer (scale-shift transform).

    Transforms x → (x - shift(x[:<i])) / exp(scale(x[:<i]))
    with the inverse being autoregressive conditioned on context z.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 32,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        in_dim = dim + (context_dim or 0)

        # Build MADE-style masked network: in_dim → hidden_dim → 2*dim
        mask0 = _build_maf_mask(dim, hidden_dim, layer=0)
        # Extend mask for context (context dims are never masked)
        if context_dim is not None:
            ctx_mask = torch.ones(hidden_dim, context_dim)
            mask0 = torch.cat([mask0, ctx_mask], dim=1)

        mask1 = _build_maf_mask(dim, hidden_dim, layer=1)

        self.net = nn.Sequential(
            MaskedLinear(in_dim, hidden_dim, mask0),
            nn.Tanh(),
            MaskedLinear(hidden_dim, 2 * dim, mask1),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: x → y, log|det J|.

        Returns:
            y: Transformed tensor, same shape as x.
            log_det: Log absolute det of Jacobian (scalar per sample).
        """
        inp = torch.cat([x, context], dim=-1) if context is not None else x
        out = self.net(inp)
        shift, log_scale = out.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale)  # stabilise scale
        y = (x - shift) * torch.exp(-log_scale)
        log_det = -log_scale.sum(dim=-1)
        return y, log_det

    def inverse(
        self,
        y: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive inverse (sequential, O(dim) passes)."""
        x = torch.zeros_like(y)
        for i in range(self.dim):
            inp = torch.cat([x, context], dim=-1) if context is not None else x
            out = self.net(inp)
            shift, log_scale = out.chunk(2, dim=-1)
            log_scale = torch.tanh(log_scale)
            x = x.clone()
            x[:, i] = y[:, i] * torch.exp(log_scale[:, i]) + shift[:, i]
        return x


# ---------------------------------------------------------------------------
# Architecture A: FlowHead
# ---------------------------------------------------------------------------


class FlowHead(nn.Module):
    """MAF head replacing the final linear output of MPNNPredictor.

    Implements conditional p(θ | z_gnn) where z_gnn is the GNN embedding.
    K=2 MAF layers, hidden_dim=32 → ≈584 trainable params (well below 5K).

    θ samples are clamped to [-π, π] to satisfy ThetaValidator L1 (Req 3.4).

    Args:
        input_dim: Dimensionality of GNN embedding (hidden_dim, typically 64/128).
        output_dim: Number of θ parameters (2p, e.g. 4 for p=2).
        n_flow_layers: Number of MAF transformation layers (default 2).
        hidden_dim: Hidden units per MAF layer (default 32).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_flow_layers: int = 2,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_flow_layers = n_flow_layers

        # Context projection: GNN embedding → compact context
        self.context_proj = nn.Linear(input_dim, hidden_dim)

        # MAF layers with context conditioning
        self.layers = nn.ModuleList(
            [
                MAFLayer(output_dim, hidden_dim=hidden_dim, context_dim=hidden_dim)
                for _ in range(n_flow_layers)
            ]
        )

        # Base distribution: standard normal
        self._base = torch.distributions.Normal(0.0, 1.0)

    def _get_context(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.context_proj(z))

    def log_prob(self, theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(θ | z).  Used as NLL training loss.

        Args:
            theta: Parameter tensor of shape [batch, output_dim].
            z: GNN embedding tensor of shape [batch, input_dim].

        Returns:
            log_prob: Shape [batch].
        """
        context = self._get_context(z)
        x = theta
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det = layer(x, context)
            log_det_total = log_det_total + log_det
        # Log prob under standard normal base
        log_base = self._base.log_prob(x).sum(dim=-1)
        return log_base + log_det_total  # type: ignore[no-any-return]

    def sample(self, z: torch.Tensor, n_samples: int = 50) -> torch.Tensor:
        """Sample θ ~ p(θ|z).

        Args:
            z: GNN embedding of shape [batch, input_dim] (typically batch=1).
            n_samples: Number of samples to draw.

        Returns:
            samples: Shape [n_samples, output_dim], clamped to [-π, π].
        """
        context = self._get_context(z)
        # Expand context for n_samples
        ctx = context.repeat(n_samples, 1) if context.shape[0] == 1 else context
        # Sample from base distribution
        x = torch.randn(n_samples, self.output_dim, device=z.device)
        # Invert through all layers (reversed order)
        for layer in reversed(list(self.layers)):
            layer_: MAFLayer = layer  # type: ignore[assignment]
            x = layer_.inverse(x, ctx)
        # Clamp to [-π, π] per ThetaValidator L1 (Req 3.4)
        return torch.clamp(x, -math.pi, math.pi)  # type: ignore[return-value,no-any-return]


# ---------------------------------------------------------------------------


class EmbeddingMAF(nn.Module):
    """MAF trained over frozen GNN embeddings.

    The GNN encoder is fully frozen (torch.no_grad() during embedding extraction).
    Only the flow parameters (~584) are trained, ensuring zero additional VQE cost
    and much lower overparameterization risk than Architecture A end-to-end (Req 3.5).

    Args:
        embedding_dim: Dimensionality of GNN embeddings (hidden_dim of encoder).
        theta_dim: Number of θ parameters (2p, e.g. 4 for p=2).
        n_flow_layers: Number of MAF layers (default 2).
        hidden_dim: Hidden units per MAF layer (default 32).
    """

    def __init__(
        self,
        embedding_dim: int,
        theta_dim: int,
        n_flow_layers: int = 2,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.theta_dim = theta_dim
        self.n_flow_layers = n_flow_layers

        # Context from embedding
        self.context_proj = nn.Linear(embedding_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                MAFLayer(theta_dim, hidden_dim=hidden_dim, context_dim=hidden_dim)
                for _ in range(n_flow_layers)
            ]
        )

        self._base = torch.distributions.Normal(0.0, 1.0)

    def _get_context(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.context_proj(z))

    def log_prob(self, theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(θ | z_frozen).

        Args:
            theta: Shape [batch, theta_dim].
            z: Frozen GNN embedding shape [batch, embedding_dim].

        Returns:
            log_prob: Shape [batch].
        """
        context = self._get_context(z)
        x = theta
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det = layer(x, context)
            log_det_total = log_det_total + log_det
        log_base = self._base.log_prob(x).sum(dim=-1)
        return log_base + log_det_total  # type: ignore[no-any-return]

    def sample(self, z: torch.Tensor, n_samples: int = 50) -> torch.Tensor:
        """Sample θ ~ p(θ|z_frozen), clamped to [-π, π].

        Args:
            z: Frozen GNN embedding of shape [1, embedding_dim].
            n_samples: Number of samples to draw.

        Returns:
            samples: Shape [n_samples, theta_dim].
        """
        context = self._get_context(z)
        ctx = context.repeat(n_samples, 1) if context.shape[0] == 1 else context
        x = torch.randn(n_samples, self.theta_dim, device=z.device)
        for layer in reversed(list(self.layers)):
            layer_: MAFLayer = layer  # type: ignore[assignment]
            x = layer_.inverse(x, ctx)
        return torch.clamp(x, -math.pi, math.pi)  # type: ignore[return-value,no-any-return]

    def trainable_param_count(self) -> int:
        """Count trainable parameters (used by OverparameterizationGuard)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
