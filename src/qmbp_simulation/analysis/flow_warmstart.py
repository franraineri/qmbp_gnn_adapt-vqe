"""Flow-based warmstart manager for VQE parameter initialisation.

Wraps EmbeddingMAF training and inference for opt-in theta warmstart
in the hardware deployment pipeline.  The MPNN encoder is always kept
frozen (torch.no_grad throughout); only the ~584 EmbeddingMAF
parameters are trained.

Requirements: 1.5, 1.7, 6.3
"""

from __future__ import annotations
