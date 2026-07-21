# Integration Plan 02: VAE+Attention for QPT Detection (Yin2025-style)

**Paper:** Yin et al. (2025) — Learning VQC Parameters with Classical AI for QPT Detection  
**arXiv:** 2506.06678  
**Code:** ❌ No public repository  
**Priority:** MEDIUM (2-3 days effort, strengthens Paper B)

## What It Does

Trains a Variational Autoencoder (VAE) with self-attention mechanism on VQE
parameter vectors θ*(h). The latent space captures nonlinear structure in the
parameter manifold, and discontinuities in the latent representation signal
quantum phase transitions without supervised labels.

## Viability Assessment

| Criterion | Status |
|-----------|--------|
| Compatible with our pipeline? | ✅ Direct — uses same θ*(h) data we already have |
| Requires new dependencies? | ❌ PyTorch only (already available) |
| Reuses existing modules? | ✅ Phase 2 θ data, `canonicalize_theta`, VQE infrastructure |
| Novel vs our PCA? | ✅ Captures nonlinear structure (PCA is linear) |
| Publishable? | ⚠️ Only as improvement over PCA in Paper B (not standalone) |

## How To Integrate

### What It Proves

That our VQE parameter data θ*(h) contains sufficient information for
unsupervised QPT detection via more sophisticated (nonlinear) methods,
and that the detected h_c converges closer to the true value than our PCA.

### Conditions Where It Makes Sense

- **Models:** `tfim`, `tfim_longitudinal`, `tfim_bond_resolved` (have clear QPT)
- **Topologies:** chain_1d (cleanest signal), ladder, heavy_hex
- **N:** 10-100 (more data points = better latent learning)
- **p:** 1-4 (works with any param dimension)
- **Critical requirement:** h-grid must cross the critical region (h_min < h_c)

### When NOT to Use

- Heisenberg/Kitaev (HVA incompatible, no clean QPT accessible)
- Pure paramagnetic regime (h >> h_c, no transition to detect)
- N < 6 (insufficient data points for VAE training)

### Integration Architecture

```
src/qmbp_simulation/
└── analysis/
    └── vae_qpt/
        ├── __init__.py
        ├── model.py           # VAE + Attention architecture
        ├── training.py        # Training loop (ELBO loss)
        └── detection.py       # QPT detection from latent space
```

### Modules to Reuse

| Module | Usage |
|--------|-------|
| `utils.canonicalize_theta` | Mandatory pre-processing (gauge fixing) |
| `utils.filter_consistent_theta` | Remove basin outliers before training |
| `pipeline.dataset_io.load_phase12_dataset` | Load existing θ*(h) data |
| `analysis.normalizing_flow.MaskedLinear` | Reuse masked autoregressive components |
| `framework.runner_base.ValidationRunner` | Structured evaluation script |

### Architecture Design (Yin2025 adapted)

```python
class AttentionVAE(nn.Module):
    """VAE with self-attention for θ*(h) → latent QPT detection.

    Encoder: Linear(2p, 64) → MultiheadAttention(4 heads) → Linear(64, 2*latent_dim)
    Decoder: Linear(latent_dim, 64) → Linear(64, 2p)
    Latent: dim=2 (for visualization) or dim=4 (for accuracy)
    """

    def __init__(self, theta_dim: int, latent_dim: int = 2, hidden: int = 64):
        # Encoder with attention
        self.encoder_proj = nn.Linear(theta_dim, hidden)
        self.attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, theta_dim)
        )
```

### Implementation Steps

1. **Create `analysis/vae_qpt/model.py`** with `AttentionVAE` class (~60 lines)
2. **Create `analysis/vae_qpt/training.py`** with ELBO training loop:
   - Loss = MSE(θ_recon, θ_input) + β * KL(q(z|θ) || N(0,I))
   - β-annealing: β = min(1.0, epoch/warmup_epochs) for stable training
   - EarlyStopping on validation ELBO
3. **Create `analysis/vae_qpt/detection.py`** with QPT detection:
   - Method 1: Peak in reconstruction loss vs h
   - Method 2: Maximum KL divergence vs h
   - Method 3: Clustering in latent space (k-means k=2, find boundary)
4. **Create script** `scripts/analysis/run_vae_qpt.py` comparing PCA vs VAE

### Expected Output

```json
{
  "pca_h_detected": 1.25,
  "pca_delta_hc": 0.25,
  "vae_h_detected": 1.05,
  "vae_delta_hc": 0.05,
  "improvement_factor": 5.0,
  "vae_latent_dim": 2,
  "n_training_points": 35,
  "model": "tfim_longitudinal",
  "topology": "chain_1d",
  "N": 10
}
```

### Success Criterion

- VAE detects h_c with |Δh| < 0.10 (vs PCA's 0.25) → publish in Paper B
- VAE works unsupervised (no phase labels) → confirms Yin2025 finding
- Latent dim 2 is sufficient (consistent with our PCA finding: 99.96% in PC1)

### Risks

- Small dataset (17-35 points per run) may not train VAE well → use β-annealing
- Attention over 2p-dim vectors (dim 2-8) is trivial → may not add over MLP-VAE
- If PCA already works at 99.96%, VAE improvement may be marginal for TFIM
- Best case: VAE works for multi-model data where PCA fails (mixed Heisenberg+TFIM)
