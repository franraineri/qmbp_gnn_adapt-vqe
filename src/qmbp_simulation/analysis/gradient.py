"""Weight Gradient Analysis — Unsupervised phase detection from MPNN weights.

Implements the weight gradient analyzer (Hernandes et al. 2025) that detects
phase transitions from the trained MPNN's internal weight structure at zero
QPU cost — purely classical post-training analysis.

This module has NO quantum imports (no Qiskit, no hardware dependencies).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from qmbp_simulation.analysis.data_models import GradientAnalysisResult

logger = logging.getLogger(__name__)

# Gradient analysis constants
GRADIENT_CRITICAL_REGION = (0.8, 1.4)  # h range for peak detection
GRADIENT_PEAK_PROMINENCE = 0.1  # Minimum prominence for peak detection


class WeightGradientAnalyzer:
    """Analyze MPNN weight gradients across the h-sweep for phase detection.

    Computes ∂L/∂W for each h-value by performing forward + backward passes
    on the trained model. The L2 norm of gradients per layer reveals phase
    transition signatures as peaks or discontinuities near h_c.

    This is a purely classical analysis — zero QPU cost.

    Parameters
    ----------
    model : nn.Module
        A trained MPNNPredictor (or compatible model with named_parameters).

    References
    ----------
    Hernandes et al. (2025). Adiabatic fine-tuning of neural quantum states
    enables detection of phase transitions in weight space.
    arXiv:2503.17140.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._model.eval()

    def analyze(
        self,
        dataset: list[Data],
        h_values: np.ndarray | None = None,
    ) -> GradientAnalysisResult:
        """Compute weight gradient norms across the h-sweep.

        Parameters
        ----------
        dataset : list[Data]
            Graph data objects (must have ``h_value`` attribute and ``y`` target).
        h_values : np.ndarray | None
            If provided, use these h-values. Otherwise extract from dataset.

        Returns
        -------
        GradientAnalysisResult
            Structured output with per-layer and total gradient norms,
            detected peaks, and critical region flag.
        """
        from scipy.signal import find_peaks

        # Extract h-values from dataset if not provided
        if h_values is None:
            h_values = np.array([d.h_value for d in dataset])

        # Sort by h-value for consistent analysis
        sort_idx = np.argsort(h_values)
        h_values_sorted = h_values[sort_idx]
        dataset_sorted = [dataset[i] for i in sort_idx]

        # Identify layer groups for per-layer analysis
        layer_groups = self._identify_layer_groups()

        if not layer_groups:
            logger.warning("Model has no trainable parameters. Returning zero gradient norms.")
            n_points = len(h_values_sorted)
            return GradientAnalysisResult(
                h_values=h_values_sorted,
                total_gradient_norms=np.zeros(n_points),
                per_layer_gradient_norms={},
                peak_h_values=[],
                peak_magnitudes=[],
                critical_region_detected=False,
            )

        # Compute gradient norms for each h-value
        n_points = len(dataset_sorted)
        per_layer_norms: dict[str, list[float]] = {name: [] for name in layer_groups}
        total_norms: list[float] = []

        criterion = nn.MSELoss()

        for data in dataset_sorted:
            # Forward + backward pass
            self._model.zero_grad()

            # Enable grad computation temporarily
            with torch.enable_grad():
                pred = self._model(data)
                target = data.y.view(pred.shape)
                loss = criterion(pred, target)
                loss.backward()

            # Collect per-layer gradient norms
            all_grads: list[torch.Tensor] = []
            for layer_name, param_names in layer_groups.items():
                layer_grads: list[torch.Tensor] = []
                for pname, param in self._model.named_parameters():
                    if pname in param_names and param.grad is not None:
                        layer_grads.append(param.grad.flatten())

                if layer_grads:
                    layer_grad_cat = torch.cat(layer_grads)
                    layer_norm = torch.linalg.norm(layer_grad_cat).item()
                    all_grads.append(layer_grad_cat)
                else:
                    layer_norm = 0.0

                per_layer_norms[layer_name].append(layer_norm)

            # Total gradient norm = L2 norm of concatenation of all gradients
            if all_grads:
                total_grad = torch.cat(all_grads)
                total_norm = torch.linalg.norm(total_grad).item()
            else:
                total_norm = 0.0
            total_norms.append(total_norm)

        # Convert to arrays
        total_gradient_norms = np.array(total_norms)
        per_layer_gradient_norms = {
            name: np.array(norms) for name, norms in per_layer_norms.items()
        }

        # Peak detection in the gradient norm curve
        peak_h_values: list[float] = []
        peak_magnitudes: list[float] = []
        critical_region_detected = False

        if len(total_gradient_norms) >= 3 and np.max(total_gradient_norms) > 0:
            peaks, properties = find_peaks(
                total_gradient_norms,
                prominence=GRADIENT_PEAK_PROMINENCE * np.max(total_gradient_norms),
            )

            for peak_idx in peaks:
                h_peak = float(h_values_sorted[peak_idx])
                mag = float(total_gradient_norms[peak_idx])
                peak_h_values.append(h_peak)
                peak_magnitudes.append(mag)

                # Check if peak is in critical region
                if GRADIENT_CRITICAL_REGION[0] <= h_peak <= GRADIENT_CRITICAL_REGION[1]:
                    critical_region_detected = True

        if not peak_h_values:
            logger.info(
                "No peaks detected in gradient norm curve. "
                "This may indicate the model is not sensitive to the phase transition."
            )

        if critical_region_detected:
            logger.info(
                f"Phase transition signature detected: gradient norm peaks at "
                f"h = {peak_h_values} (critical region h ∈ {GRADIENT_CRITICAL_REGION})"
            )

        return GradientAnalysisResult(
            h_values=h_values_sorted,
            total_gradient_norms=total_gradient_norms,
            per_layer_gradient_norms=per_layer_gradient_norms,
            peak_h_values=peak_h_values,
            peak_magnitudes=peak_magnitudes,
            critical_region_detected=critical_region_detected,
        )

    def _identify_layer_groups(self) -> dict[str, list[str]]:
        """Group model parameters by layer for per-layer gradient analysis.

        Returns
        -------
        dict[str, list[str]]
            Mapping from layer name (e.g. "ginconv_0", "head") to list of
            parameter names belonging to that layer.
        """
        groups: dict[str, list[str]] = {}

        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine which layer group this parameter belongs to
            if name.startswith("convs."):
                # Extract conv layer index: "convs.0.nn.0.weight" → "ginconv_0"
                parts = name.split(".")
                layer_idx = parts[1]
                group_name = f"ginconv_{layer_idx}"
            elif (
                name.startswith("head_zz.")
                or name.startswith("head_x.")
                or name.startswith("head.")
            ):
                group_name = "head"
            elif name.startswith("bns."):
                # BatchNorm params — group with corresponding conv layer
                parts = name.split(".")
                layer_idx = parts[1]
                group_name = f"ginconv_{layer_idx}"
            else:
                group_name = "other"

            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(name)

        return groups
