"""Reusable statistical tests for experiment validation.

Extracted from ad-hoc analysis scripts (analyze_gnn_qem_results.py, etc.)
into a tested, importable module. All functions return structured dicts
suitable for JSON serialization and inclusion in reports.

Usage:
    from project_health.analysis.statistical_tests import (
        paired_ttest,
        improvement_rate,
        effect_size_cohens_d,
        binomial_ci,
        spearman_correlation,
        mann_whitney_u,
        bootstrap_ci,
    )

    result = paired_ttest(before=[0.15, 0.12, 0.18], after=[0.02, 0.01, 0.03])
    # {'t_stat': 8.5, 'p_value': 0.006, 'significant_005': True, ...}

    ci = binomial_ci(17, 20)
    # {'proportion': 0.85, 'ci_lower': 0.64, 'ci_upper': 0.95, ...}

    rho = spearman_correlation(h_values, de_gaps)
    # {'rho': -0.95, 'interpretation': 'very_strong_negative', ...}
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


# ═══════════════════════════════════════════════════════════════════════════════
# Binomial confidence interval
# ═══════════════════════════════════════════════════════════════════════════════


def binomial_ci(
    n_success: int,
    n_total: int,
    *,
    confidence: float = 0.95,
    method: str = "wilson",
) -> dict[str, Any]:
    """Binomial confidence interval for pass/fail rates.

    Useful for reporting pass_rate with uncertainty bounds (e.g.,
    "pass_rate = 85% [76%, 92%] at 95% CI").

    Parameters
    ----------
    n_success : int
        Number of successes (e.g., points passing dual criterion).
    n_total : int
        Total number of trials.
    confidence : float
        Confidence level (default 0.95 = 95% CI).
    method : str
        "wilson" (default, recommended for small N) or "normal" (Wald interval).

    Returns
    -------
    dict with keys: n_success, n_total, proportion, ci_lower, ci_upper,
    confidence, method.

    Examples
    --------
    >>> binomial_ci(17, 20)
    {'proportion': 0.85, 'ci_lower': 0.62, 'ci_upper': 0.96, ...}
    """
    if n_total <= 0:
        return {
            "n_success": 0, "n_total": 0, "proportion": 0.0,
            "ci_lower": 0.0, "ci_upper": 0.0,
            "confidence": confidence, "method": method,
        }

    p_hat = n_success / n_total
    alpha = 1.0 - confidence
    z = _normal_ppf(1.0 - alpha / 2.0)

    if method == "wilson":
        # Wilson score interval (better coverage for small n)
        denom = 1 + z * z / n_total
        center = (p_hat + z * z / (2 * n_total)) / denom
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z * z / (4 * n_total * n_total)) / denom
        ci_lower = max(0.0, center - margin)
        ci_upper = min(1.0, center + margin)
    else:
        # Normal (Wald) interval
        se = math.sqrt(p_hat * (1 - p_hat) / n_total) if 0 < p_hat < 1 else 0.0
        ci_lower = max(0.0, p_hat - z * se)
        ci_upper = min(1.0, p_hat + z * se)

    return {
        "n_success": n_success,
        "n_total": n_total,
        "proportion": p_hat,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "confidence": confidence,
        "method": method,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Spearman rank correlation
# ═══════════════════════════════════════════════════════════════════════════════


def spearman_correlation(
    x: list[float],
    y: list[float],
) -> dict[str, Any]:
    """Spearman rank correlation coefficient.

    Non-parametric measure of monotonic association. Useful for:
    - Correlating ΔE/gap vs h (monotonic decrease expected)
    - Correlating training MSE vs evaluation pass_rate
    - Checking if h_frontier scales monotonically with N

    Parameters
    ----------
    x, y : list[float]
        Paired observations (same length, ≥ 3).

    Returns
    -------
    dict with keys: rho, n, t_stat, p_value, significant_005,
    interpretation.

    Examples
    --------
    >>> spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40])
    {'rho': 1.0, 'interpretation': 'perfect_positive', ...}
    """
    if len(x) != len(y):
        raise ValueError(f"x ({len(x)}) and y ({len(y)}) must have same length")
    n = len(x)
    if n < 3:
        return {
            "rho": 0.0, "n": n, "t_stat": 0.0, "p_value": 1.0,
            "significant_005": False, "interpretation": "insufficient_data",
        }

    # Compute ranks (average rank for ties)
    rank_x = _compute_ranks(x)
    rank_y = _compute_ranks(y)

    # Pearson correlation on ranks
    mean_rx = sum(rank_x) / n
    mean_ry = sum(rank_y) / n

    num = sum((rx - mean_rx) * (ry - mean_ry) for rx, ry in zip(rank_x, rank_y))
    den_x = math.sqrt(sum((rx - mean_rx) ** 2 for rx in rank_x))
    den_y = math.sqrt(sum((ry - mean_ry) ** 2 for ry in rank_y))

    if den_x * den_y == 0:
        rho = 0.0
    else:
        rho = num / (den_x * den_y)

    # t-test for significance
    if abs(rho) >= 1.0 - 1e-10:
        t_stat = float("inf") if rho > 0 else float("-inf")
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1 - rho * rho))
        p_value = 2 * _t_sf(abs(t_stat), n - 2)

    # Interpretation
    abs_rho = abs(rho)
    if abs_rho >= 0.9:
        interp = "very_strong"
    elif abs_rho >= 0.7:
        interp = "strong"
    elif abs_rho >= 0.4:
        interp = "moderate"
    elif abs_rho >= 0.2:
        interp = "weak"
    else:
        interp = "negligible"
    if rho < 0:
        interp += "_negative"
    elif rho > 0:
        interp += "_positive"

    return {
        "rho": round(rho, 4),
        "n": n,
        "t_stat": round(t_stat, 4) if math.isfinite(t_stat) else t_stat,
        "p_value": round(p_value, 6),
        "significant_005": p_value < 0.05,
        "interpretation": interp,
    }


def _compute_ranks(values: list[float]) -> list[float]:
    """Compute ranks with average rank for ties."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        # Find tied group
        j = i + 1
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        # Average rank for tied group
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based ranks
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j

    return ranks


# ═══════════════════════════════════════════════════════════════════════════════
# Mann-Whitney U test (non-parametric, unpaired)
# ═══════════════════════════════════════════════════════════════════════════════


def mann_whitney_u(
    group_a: list[float],
    group_b: list[float],
) -> dict[str, Any]:
    """Mann-Whitney U test for comparing two independent groups.

    Non-parametric alternative to independent t-test. Useful for:
    - Comparing MPNN warm-start vs random-init VQE results
    - Comparing different model architectures
    - Comparing topologies (where pairing is not possible)

    Parameters
    ----------
    group_a, group_b : list[float]
        Independent samples from two groups.

    Returns
    -------
    dict with keys: U, n_a, n_b, z_stat, p_value, significant_005,
    effect_size_r (rank-biserial correlation).
    """
    n_a = len(group_a)
    n_b = len(group_b)
    if n_a < 2 or n_b < 2:
        return {
            "U": 0, "n_a": n_a, "n_b": n_b, "z_stat": 0.0,
            "p_value": 1.0, "significant_005": False, "effect_size_r": 0.0,
        }

    # Combine and rank
    combined = [(v, "a") for v in group_a] + [(v, "b") for v in group_b]
    combined_values = [v for v, _ in combined]
    ranks = _compute_ranks(combined_values)

    # Sum of ranks for group A
    rank_sum_a = sum(ranks[i] for i in range(n_a))

    # U statistic
    U_a = rank_sum_a - n_a * (n_a + 1) / 2
    U_b = n_a * n_b - U_a
    U = min(U_a, U_b)

    # Normal approximation for large samples
    mu_U = n_a * n_b / 2
    sigma_U = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)

    if sigma_U > 0:
        z_stat = (U_a - mu_U) / sigma_U
        p_value = 2 * _normal_sf(abs(z_stat))
    else:
        z_stat = 0.0
        p_value = 1.0

    # Effect size: rank-biserial correlation
    r = 1 - (2 * U) / (n_a * n_b)

    return {
        "U": U,
        "n_a": n_a,
        "n_b": n_b,
        "z_stat": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "significant_005": p_value < 0.05,
        "effect_size_r": round(r, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap confidence interval (for small samples)
# ═══════════════════════════════════════════════════════════════════════════════


def bootstrap_ci(
    data: list[float],
    *,
    statistic: str = "mean",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap confidence interval for arbitrary statistics.

    Useful when distributional assumptions are unclear (e.g., ΔE/gap
    distributions which are often skewed).

    Parameters
    ----------
    data : list[float]
        Sample data.
    statistic : str
        "mean" or "median".
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: estimate, ci_lower, ci_upper, se_bootstrap, n, confidence.
    """
    import random

    n = len(data)
    if n < 2:
        val = data[0] if n == 1 else 0.0
        return {
            "estimate": val, "ci_lower": val, "ci_upper": val,
            "se_bootstrap": 0.0, "n": n, "confidence": confidence,
        }

    rng = random.Random(seed)

    # Compute statistic on bootstrap resamples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = [rng.choice(data) for _ in range(n)]
        if statistic == "median":
            sorted_r = sorted(resample)
            val = sorted_r[n // 2] if n % 2 == 1 else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2
        else:
            val = sum(resample) / n
        bootstrap_stats.append(val)

    bootstrap_stats.sort()
    alpha = 1.0 - confidence
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    # Original estimate
    if statistic == "median":
        sorted_data = sorted(data)
        estimate = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        estimate = sum(data) / n

    # Bootstrap SE
    mean_boot = sum(bootstrap_stats) / n_bootstrap
    se = math.sqrt(sum((s - mean_boot) ** 2 for s in bootstrap_stats) / (n_bootstrap - 1))

    return {
        "estimate": round(estimate, 6),
        "ci_lower": round(bootstrap_stats[lower_idx], 6),
        "ci_upper": round(bootstrap_stats[min(upper_idx, n_bootstrap - 1)], 6),
        "se_bootstrap": round(se, 6),
        "n": n,
        "confidence": confidence,
    }
