"""
Experimental / deprecated architectures and utilities.

These were tested during V6.0 development and rejected for the thesis pipeline:
- GATPredictor: adds instability for 1D chains (GINConv is superior)
- augment_graph_dataset: linear interpolation hurts accuracy at both N=6 and N=10

Kept for reproducibility of V6.0 benchmark results (benchmark_v6.py).
"""

from .augmentation import augment_graph_dataset
from .gat_predictor import GATPredictor

__all__ = ["GATPredictor", "augment_graph_dataset"]
