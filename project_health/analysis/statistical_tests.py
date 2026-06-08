"""Reusable statistical tests for experiment validation.

Extracted from ad-hoc analysis scripts (analyze_gnn_qem_results.py, etc.)
into a tested, importable module. All functions return structured dicts
suitable for JSON serialization and inclusion in reports.

Usage:
    from project_health.analysis.statistical_tests import (
        paired_ttest,
        improvement_rate,
        effect_size_cohens_d,
    )

    result = paired_ttest(before=[0.15, 0.12, 0.18], after=[0.02, 0.01, 0.03])
    # {'t_stat': 8.5, 'p_value': 0.006, 'significant_005': True, ...}
"""

from __future__ import annotations

import math
from typing import Any


def paired_ttest(
    before: list[float],
    after: list[float],
    *,
    alternative: str = "greater",
) -> dict[str, Any]:
    """Paired t-test for method comparison (e.g., GNN-QEM vs baseline).

    Tests H₀: mean(before - after) ≤ 0 (no improvement)
    vs    H₁: mean(before - after) > 0 (method helps)

    Parameters
    ----------
    before : list[float]
        Metric values before treatment (e.g., error without mitigation).
    after : list[float]
        Metric values after treatment (e.g., error with mitigation).
    alternative : str
        "greater" (default, one-sided: before > after means improvement),
        "two-sided", or "less".

    Returns
    -------
    dict with keys: n, mean_diff, std_diff, t_stat, p_value,
    significant_005, significant_001, effect_size_d, ci_95_lower, ci_95_upper.

    Raises
    ------
    ValueError
        If lists have different lengths or fewer than 2 elements.
    """
    if len(before) != len(after):
        msg = f"before ({len(before)}) and after ({len(after)}) must have same length"
        raise ValueError(msg)
    n = len(before)
    if n < 2:
        msg = f"Need at least 2 paired observations, got {n}"
        raise ValueError(msg)

    diffs = [b - a for b, a in zip(before, after, strict=True)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(var_diff) if var_diff > 0 else 0.0

    if std_diff == 0:
        # All differences are identical — infinite t-stat (or 0 if mean=0)
        t_stat = float("inf") if mean_diff != 0 else 0.0
        p_value = 0.0 if mean_diff > 0 else 1.0
    else:
        se = std_diff / math.sqrt(n)
        t_stat = mean_diff / se
        # Approximate p-value using t-distribution (scipy not always available)
        p_value = _t_sf(t_stat, n - 1)

    if alternative == "two-sided":
        p_value = 2 * min(p_value, 1 - p_value)
    elif alternative == "less":
        p_value = 1 - p_value

    # 95% confidence interval for mean difference
    t_crit = _t_ppf(0.975, n - 1)  # Two-tailed 95%
    se = std_diff / math.sqrt(n) if std_diff > 0 else 0.0
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "effect_size_d": mean_diff / std_diff if std_diff > 0 else float("inf"),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
    }


def improvement_rate(
    before: list[float],
    after: list[float],
) -> dict[str, Any]:
    """Compute improvement rate and summary statistics.

    Parameters
    ----------
    before, after : list[float]
        Paired measurements (lower = better, e.g., error values).

    Returns
    -------
    dict with keys: n, n_improved, n_same, n_worsened, improvement_rate_pct,
    mean_before, mean_after, mean_reduction_pct.
    """
    if len(before) != len(after):
        msg = f"before ({len(before)}) and after ({len(after)}) must have same length"
        raise ValueError(msg)

    n = len(before)
    n_improved = sum(1 for b, a in zip(before, after, strict=True) if a < b)
    n_same = sum(1 for b, a in zip(before, after, strict=True) if math.isclose(a, b, rel_tol=1e-9))
    n_worsened = n - n_improved - n_same

    mean_before = sum(before) / n if n > 0 else 0.0
    mean_after = sum(after) / n if n > 0 else 0.0
    reduction_pct = ((mean_before - mean_after) / mean_before * 100) if mean_before > 0 else 0.0

    return {
        "n": n,
        "n_improved": n_improved,
        "n_same": n_same,
        "n_worsened": n_worsened,
        "improvement_rate_pct": n_improved / n * 100 if n > 0 else 0.0,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_reduction_pct": reduction_pct,
    }


def effect_size_cohens_d(
    before: list[float],
    after: list[float],
) -> float:
    """Cohen's d effect size for paired samples.

    Interpretation: |d| < 0.2 = small, 0.2-0.8 = medium, > 0.8 = large.
    Returns 0.0 when mean difference is zero (no effect).
    """
    if len(before) != len(after) or len(before) < 2:
        return 0.0
    diffs = [b - a for b, a in zip(before, after, strict=True)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    if mean_d == 0.0:
        return 0.0
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = math.sqrt(var_d) if var_d > 0 else 0.0
    return mean_d / std_d if std_d > 0 else float("inf")


# ═══════════════════════════════════════════════════════════════════════════════
# Internal: t-distribution approximation (no scipy dependency)
# ═══════════════════════════════════════════════════════════════════════════════


def _t_sf(t: float, df: int) -> float:
    """Survival function (1-CDF) of Student's t-distribution.

    Uses the regularized incomplete beta function approximation.
    For df >= 30, approximates with standard normal.
    """
    if df <= 0:
        return 0.5
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0

    # For large df, use normal approximation
    if df >= 30:
        return _normal_sf(t)

    # Beta function approximation: P(T > t) = 0.5 * I_x(df/2, 1/2)
    # where x = df / (df + t²)
    x = df / (df + t * t)
    p = 0.5 * _betai(df / 2.0, 0.5, x)
    return p if t > 0 else 1.0 - p


def _t_ppf(p: float, df: int) -> float:
    """Percent point function (inverse CDF) of t-distribution.

    Approximation using Abramowitz & Stegun 26.7.5 for df >= 5.
    Falls back to simple lookup for small df.
    """
    if df >= 30:
        return _normal_ppf(p)

    # Newton's method starting from normal approximation
    z = _normal_ppf(p)
    # Cornish-Fisher expansion (one iteration)
    g1 = (z**3 + z) / (4 * df)
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    return z + g1 + g2


def _normal_sf(x: float) -> float:
    """Standard normal survival function (1 - Phi(x))."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def _normal_ppf(p: float) -> float:
    """Standard normal percent point function (inverse of Phi).

    Rational approximation (Abramowitz & Stegun 26.2.23).
    """
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    if p == 0.5:
        return 0.0

    if p < 0.5:
        sign = -1.0
        p_work = p
    else:
        sign = 1.0
        p_work = 1.0 - p

    t = math.sqrt(-2.0 * math.log(p_work))
    # Coefficients for rational approximation
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    numerator = c0 + c1 * t + c2 * t * t
    denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    return sign * (t - numerator / denominator)


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Uses continued fraction expansion (Lentz's method).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use symmetry if needed for convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betai(b, a, 1.0 - x)

    # Log of the prefactor
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x)) / a

    # Continued fraction (Lentz's method)
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 201):
        # Even step
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= c * d

        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-8:
            break

    return front * f
