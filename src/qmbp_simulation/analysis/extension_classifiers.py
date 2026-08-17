"""Pure threshold-based classification engine for thesis extension analysis.

All methods are static, stateless, and deterministically testable.
No torch/qiskit imports — stdlib only.

Req: 1.3, 1.5, 1.6, 1.7, 2.2, 2.4, 2.6, 3.2, 3.6, 3.7, 3.8
"""

from __future__ import annotations

import logging
import math

from qmbp_simulation.analysis.constants import DE_GAP_THRESHOLD, MAX_ABS_ERROR
from qmbp_simulation.analysis.extension_models import ExtensionClassification

logger = logging.getLogger(__name__)

#: Maximum N for practical ExactDiag on spin-1/2 systems (H.S. = 2^18 = 262,144).
#: Exported so property tests can reference it without hard-coding 18.
EXACT_DIAG_N_CEILING: int = 18


class ClassificationEngine:
    """Pure functions for threshold-based classification.

    All methods are stateless and deterministically testable
    (Req 1.3, 1.6, 1.7, 2.4, 2.6, 3.2, 3.6, 3.7, 3.8).
    No instance state — use as a namespace of static methods.
    """

    @staticmethod
    def classify_cross_n(
        de_gap_all_sizes: list[float],
    ) -> ExtensionClassification:
        """Req 1.3: REJECTED_INSUFFICIENT_DATA if ALL sizes have ΔE/gap ≥ 5%."""
        if all(v >= DE_GAP_THRESHOLD for v in de_gap_all_sizes):
            return ExtensionClassification.REJECTED_INSUFFICIENT_DATA
        return ExtensionClassification.CONDITIONALLY_VIABLE

    @staticmethod
    def classify_intra_n(
        de_gap: float,
        n_pass: int,
        n_total: int,
    ) -> ExtensionClassification:
        """Req 1.6: CONDITIONALLY_VIABLE if ΔE/gap ≤ 1% AND ≥ 5/6 h-points pass."""
        if de_gap <= 0.01 and n_pass >= math.ceil(5 * n_total / 6):
            return ExtensionClassification.CONDITIONALLY_VIABLE
        return ExtensionClassification.REJECTED_INSUFFICIENT_DATA

    @staticmethod
    def classify_hardware(
        cx_count: int,
        threshold: int = 18,
    ) -> ExtensionClassification:
        """Req 1.7, 2.4: HARDWARE_INCOMPATIBLE if CX count exceeds ZNE threshold."""
        if cx_count > threshold:
            return ExtensionClassification.HARDWARE_INCOMPATIBLE
        return ExtensionClassification.VIABLE

    @staticmethod
    def classify_expressibility(fidelity: float) -> ExtensionClassification:
        """Req 2.6: EXPRESSIBILITY_INSUFFICIENT if fidelity < 0.60.

        Args:
            fidelity: State-overlap fidelity ∈ [0, 1].

        Returns:
            EXPRESSIBILITY_INSUFFICIENT when fidelity < 0.60; VIABLE otherwise.
        """
        if fidelity < 0.60:
            return ExtensionClassification.EXPRESSIBILITY_INSUFFICIENT
        return ExtensionClassification.VIABLE

    @staticmethod
    def classify_flow_architecture(
        calibration_improvement: float,
        de_gap: float,
        n_params: int,
        n_data: int,
    ) -> ExtensionClassification:
        """Req 3.2, 3.6, 3.7, 3.8: Classify a normalizing-flow architecture.

        Priority order (evaluated top-to-bottom, first match wins):
        1. OVERPARAMETERIZED_FOR_DATASET  — n_params > 5000 AND n_data < 50
        2. DEGRADED_VS_BASELINE           — de_gap ≥ 0.10 (2× primary threshold)
        3. VIABLE                         — calibration_improvement ≥ 0.02
                                            AND de_gap < 0.05
        4. CONDITIONALLY_VIABLE           — all other cases

        Args:
            calibration_improvement: Improvement in coverage-90 vs MC-Dropout
                                     baseline (e.g. 0.03 = +3 pp).
            de_gap:    ΔE/gap of the flow's mean prediction (fraction).
            n_params:  Number of *trainable* parameters in the flow model.
            n_data:    Number of training samples available.

        Returns:
            The highest-priority matching ExtensionClassification.
        """
        if n_params > 5000 and n_data < 50:
            return ExtensionClassification.OVERPARAMETERIZED_FOR_DATASET
        if de_gap >= MAX_ABS_ERROR:
            return ExtensionClassification.DEGRADED_VS_BASELINE
        if calibration_improvement >= 0.02 and de_gap < DE_GAP_THRESHOLD:
            return ExtensionClassification.VIABLE
        return ExtensionClassification.CONDITIONALLY_VIABLE

    @staticmethod
    def compute_n_min_data(
        n_params: int,
        ratio_threshold: int = 1000,
    ) -> int:
        """Req 1.5: Minimum training samples s.t. params/data ≤ ratio_threshold.


        N_min_data = ceil(n_params / ratio_threshold)
        """
        return math.ceil(n_params / ratio_threshold)

    @staticmethod
    def hilbert_space_dimension(n_sites: int) -> int:
        """Req 2.2: Hilbert-space dimension 2^N for spin-1/2 systems.

        Args:
            n_sites: Number of spin-1/2 sites N.

        Returns:
            dim = 2**n_sites (exact integer).

        Note:
            Emits a WARNING when N > EXACT_DIAG_N_CEILING (18) since full
            Hilbert-space diagonalization becomes impractical beyond that point.
            Use ``hilbert_space_dimension_flagged`` if you need the bool flag
            programmatically without relying on log side-effects.
        """
        if n_sites > EXACT_DIAG_N_CEILING:
            logger.warning(
                "hilbert_space_dimension: N=%d exceeds ExactDiag ceiling "
                "(N_max=%d, H.S.=2^%d=%d). Full diagonalization is impractical.",
                n_sites,
                EXACT_DIAG_N_CEILING,
                EXACT_DIAG_N_CEILING,
                2**EXACT_DIAG_N_CEILING,
            )
        return 2**n_sites

    @staticmethod
    def hilbert_space_dimension_flagged(n_sites: int) -> tuple[int, bool]:
        """Req 2.2 + Property 5: dimension plus ExactDiag ceiling flag.

        Returns:
            (dim, exceeds_ceiling) where exceeds_ceiling is True when n_sites > 18.
        """
        dim = 2**n_sites
        return dim, n_sites > 18
