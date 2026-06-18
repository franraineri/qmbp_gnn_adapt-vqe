"""Property-based tests for MitigationOptions dd_sequence validation.

# Feature: mitigation-benchmark, Property 11: dd_sequence validation
# **Validates: Requirements 7.3, 7.4**
#
# When dd_enabled=True, MitigationOptions SHALL accept only valid dd_sequence
# values ("XX", "XpXm", "XY4") and raise ValueError for any other string.
# When dd_enabled=False, any dd_sequence value is accepted without error.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from qmbp_simulation.execution import MitigationOptions

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

VALID_DD_SEQUENCES = ("XX", "XpXm", "XY4")


# ═══════════════════════════════════════════════════════════════════════════════
# Property 11: dd_sequence validation — invalid strings raise ValueError
# when dd_enabled=True
# ═══════════════════════════════════════════════════════════════════════════════


@given(dd_sequence=st.text(min_size=1, max_size=20))
@settings(max_examples=200)
def test_invalid_dd_sequence_raises_valueerror_when_enabled(dd_sequence: str) -> None:
    """Invalid dd_sequence values raise ValueError when dd_enabled=True.

    **Validates: Requirements 7.3, 7.4**

    For any arbitrary string that is NOT one of the valid sequences
    ("XX", "XpXm", "XY4"), constructing MitigationOptions with
    dd_enabled=True must raise ValueError.
    """
    assume(dd_sequence not in VALID_DD_SEQUENCES)

    with pytest.raises(ValueError, match="Invalid dd_sequence"):
        MitigationOptions(dd_enabled=True, dd_sequence=dd_sequence)


@given(dd_sequence=st.sampled_from(VALID_DD_SEQUENCES))
@settings(max_examples=50)
def test_valid_dd_sequence_accepted_when_enabled(dd_sequence: str) -> None:
    """Valid dd_sequence values do NOT raise when dd_enabled=True.

    **Validates: Requirements 7.3, 7.4**

    The three recognized sequences ("XX", "XpXm", "XY4") must be
    accepted without error when dd_enabled=True.
    """
    opts = MitigationOptions(dd_enabled=True, dd_sequence=dd_sequence)
    assert opts.dd_sequence == dd_sequence
    assert opts.dd_enabled is True


@given(dd_sequence=st.text(min_size=0, max_size=20))
@settings(max_examples=200)
def test_any_dd_sequence_accepted_when_disabled(dd_sequence: str) -> None:
    """Any dd_sequence value is accepted when dd_enabled=False.

    **Validates: Requirements 7.3, 7.4**

    When dd_enabled=False, the dd_sequence field is ignored and no
    validation error should be raised regardless of the value.
    """
    opts = MitigationOptions(dd_enabled=False, dd_sequence=dd_sequence)
    assert opts.dd_sequence == dd_sequence
    assert opts.dd_enabled is False
