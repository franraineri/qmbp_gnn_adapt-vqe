"""Pure classification functions for thesis extension analysis.

All methods are stateless and deterministically testable.
Correctness properties are validated via hypothesis property tests.

Req: 1.3, 1.5, 1.6, 1.7, 2.2, 2.4, 2.6, 3.2, 3.6, 3.7, 3.8
"""

from __future__ import annotations

import logging
import math

from qmbp_simulation.analysis.extension_models import ExtensionClassification

logger = logging.getLogger(__name__)

# Practical ceiling for ExactDiag (N ≤ 18 → H.S. ≤ 262,144)
EXACT_DIAG_N_CEILING: int = 18

# ZNE CX gate threshold: >18 gates → hardware incompatible
ZNE_CX_THRESHOLD: int = 18

# Overparameterization: trainable_params > 5000 AND n_data < 50
OVERPARAMETERIZATION_PARAM_THRESHOLD: int = 5000
OVERPARAMETERIZATION_DATA_THRESHOLD: int = 50

# Flow viability thresholds
COVERAGE_IMPROVEMENT_MIN: float = 0.02  # absolute coverage improvement
DE_GAP_VIABLE_MAX: float = 0.05  # 5 % — primary threshold
DE_GAP_DEGRADED_MIN: float = 0.10  # 10 % — 2× primary → DEGRADED

# Intra-N pass thresholds
INTRA_N_DE_GAP_MAX: float = 0.01  # ΔE/gap ≤ 1 %
INTRA_N_PASS_FRACTION: float = 5.0 / 6.0  # ≥ 5/6 h-points

# Expressibility fidelity threshold
EXPRESSIBILITY_FIDELITY_MIN: float = 0.60

# Data/param ratio requirement
DATA_PARAM_RATIO_THRESHOLD: int = 1_000


class ClassificationEngine:
    """Pure functions for threshold-based classification.

    All methods are class-level static — no state, fully deterministic.
    """

    @staticmethod
    def classify_cross_n(
        de_gap_all_sizes: list[float],
    ) -> ExtensionClassification:
        """Req 1.3: REJECTED if ALL sizes ≥ 5 %; otherwise CONDITIONALLY_VIABLE.

        Property 1: For any list where all values ≥ 0.05 → REJECTED_INSUFFICIENT_DATA.
                    For any list where at least one value < 0.05 → not REJECTED.
        """
        if not de_gap_all_sizes:
            # Empty list: no evidence of generalization → reject
            return ExtensionClassification.REJECTED_INSUFFICIENT_DATA
        if all(v >= 0.05 for v in de_gap_all_sizes):
            return ExtensionClassification.REJECTED_INSUFFICIENT_DATA
        return ExtensionClassification.CONDITIONALLY_VIABLE

    @staticmethod
    def classify_intra_n(
        de_gap: float,
        n_pass: int,
        n_total: int,
    ) -> ExtensionClassification:
        """Req 1.6: CONDITIONALLY_VIABLE if ΔE/gap ≤ 1% AND ≥ 5/6 h-points pass.

        Property 3: Both conditions must hold simultaneously for CONDITIONALLY_VIABLE.
        """
        if n_total == 0:
            return ExtensionClassification.REJECTED_INSUFFICIENT_DATA
        threshold_pass = math.floor(INTRA_N_PASS_FRACTION * n_total)
        # Use 5*n_total//6 exactly as in the design spec
        threshold_pass = (5 * n_total) // 6
        if de_gap <= INTRA_N_DE_GAP_MAX and n_pass >= threshold_pass:
            return ExtensionClassification.CONDITIONALLY_VIABLE
        return ExtensionClassification.REJECTED_INSUFFICIENT_DATA

    @staticmethod
    def classify_hardware(
        cx_count: int,
        threshold: int = ZNE_CX_THRESHOLD,
    ) -> ExtensionClassification:
        """Req 1.7, 2.4: HARDWARE_INCOMPATIBLE if CX > threshold.

        Property 4: cx_count > 18 → HARDWARE_INCOMPATIBLE; ≤ 18 → VIABLE.
        Applies equally to Ext1 and Ext2.
        """
        if cx_count > threshold:
            return ExtensionClassification.HARDWARE_INCOMPATIBLE
        return ExtensionClassification.VIABLE

    @staticmethod
    def classify_expressibility(fidelity: float) -> ExtensionClassification:
        """Req 2.6: EXPRESSIBILITY_INSUFFICIENT if fidelity < 0.60.

        Property 6: f < 0.60 → EXPRESSIBILITY_INSUFFICIENT; f ≥ 0.60 → VIABLE.
        """
        if fidelity < EXPRESSIBILITY_FIDELITY_MIN:
            return ExtensionClassification.EXPRESSIBILITY_INSUFFICIENT
        return ExtensionClassification.VIABLE

    @staticmethod
    def classify_flow_architecture(
        calibration_improvement: float,
        de_gap: float,
        n_params: int,
        n_data: int,
    ) -> ExtensionClassification:
        """Req 3.2, 3.6, 3.7, 3.8: Flow architecture classification.

        Priority order (Property 7):
          1. OVERPARAMETERIZED_FOR_DATASET   (n_params > 5000 AND n_data < 50)
          2. DEGRADED_VS_BASELINE            (de_gap ≥ 0.10)
          3. VIABLE                          (improvement ≥ 0.02 AND de_gap < 0.05)
          4. CONDITIONALLY_VIABLE            (all other cases)
        """
        if (
            n_params > OVERPARAMETERIZATION_PARAM_THRESHOLD
            and n_data < OVERPARAMETERIZATION_DATA_THRESHOLD
        ):
            return ExtensionClassification.OVERPARAMETERIZED_FOR_DATASET
        if de_gap >= DE_GAP_DEGRADED_MIN:
            return ExtensionClassification.DEGRADED_VS_BASELINE
        if calibration_improvement >= COVERAGE_IMPROVEMENT_MIN and de_gap < DE_GAP_VIABLE_MAX:
            return ExtensionClassification.VIABLE
        return ExtensionClassification.CONDITIONALLY_VIABLE

    @staticmethod
    def compute_n_min_data(
        n_params: int,
        ratio_threshold: int = DATA_PARAM_RATIO_THRESHOLD,
    ) -> int:
        """Req 1.5: N_min_data = ceil(n_params / ratio_threshold).

        Property 2: n_params / N_min_data ≤ ratio_threshold for all n_params > 0.
        """
        if n_params <= 0:
            raise ValueError(f"n_params must be positive, got {n_params}")
        return math.ceil(n_params / ratio_threshold)

    @staticmethod
    def hilbert_space_dimension(n_sites: int) -> int:
        """Req 2.2: 2^N for spin-1/2 Kagomé.

        Property 5: Exact 2**N for all positive N.
        Emits a warning for N > EXACT_DIAG_N_CEILING (practical ExactDiag ceiling).
        """
        if n_sites <= 0:
            raise ValueError(f"n_sites must be positive, got {n_sites}")
        dim = 2**n_sites
        if n_sites > EXACT_DIAG_N_CEILING:
            logger.warning(
                "N=%d exceeds ExactDiag ceiling (N≤%d). "
                "Hilbert space dimension = %d — ExactDiag not practical.",
                n_sites,
                EXACT_DIAG_N_CEILING,
                dim,
            )
        return dim
