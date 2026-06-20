"""Temporary deep verification using ResultScanner."""

from pathlib import Path

from project_health.digest import ResultScanner

scanner = ResultScanner(Path("results"))
noiseless, noisy, experiments = scanner.scan_all()
cross_topo = []  # separate call if needed

n_with_smoothness = sum(1 for r in noiseless if r.theta_smoothness is not None)
n_with_gen_gap = sum(1 for r in noiseless if r.generalization_gap is not None)
n_with_conv = sum(1 for r in noiseless if r.convergence_rate is not None)

print(f"Noiseless total: {len(noiseless)}")
print(f"With theta_smoothness: {n_with_smoothness}")
print(f"With gen_gap: {n_with_gen_gap}")
print(f"With convergence_rate: {n_with_conv}")
print()

# Verify theta_smoothness claim
if n_with_smoothness:
    above = sum(1 for r in noiseless if r.theta_smoothness and r.theta_smoothness > 1.0)
    pct = above / n_with_smoothness * 100
    vals = [r.theta_smoothness for r in noiseless if r.theta_smoothness is not None]
    mean_s = sum(vals) / len(vals)
    max_s = max(vals)
    print(f"θ-smoothness > 1.0: {above}/{n_with_smoothness} ({pct:.0f}%)")
    print(f"  Mean: {mean_s:.4f}, Max: {max_s:.4f}")
    print("  Claimed: 96/329 (29%), mean=1.05, max=6.14")
    print()

# Verify gen_gap claim
if n_with_gen_gap:
    above = sum(1 for r in noiseless if r.generalization_gap and r.generalization_gap > 0.01)
    pct = above / n_with_gen_gap * 100
    vals = [r.generalization_gap for r in noiseless if r.generalization_gap is not None]
    mean_g = sum(vals) / len(vals)
    median_g = sorted(vals)[len(vals) // 2]
    print(f"gen_gap > 0.01: {above}/{n_with_gen_gap} ({pct:.0f}%)")
    print(f"  Mean: {mean_g:.6f}, Median: {median_g:.6f}")
    print("  Claimed: 41/279 (15%), mean=0.0049, median=0.00028")
    print()

# Verify convergence_rate claim
if n_with_conv:
    rates = [r.convergence_rate for r in noiseless if r.convergence_rate is not None]
    mean_r = sum(rates) / len(rates)
    min_r = min(rates)
    print(f"Convergence rate: n={len(rates)}, mean={mean_r:.4f}, min={min_r:.4f}")
    print("  Claimed: mean=0.9958, min=0.75")
    print()

# Verify error decomposition
n_with_ed = 0
n_circuit_zero = 0
for r in noiseless:
    if hasattr(r, "error_from_circuit") and r.error_from_circuit is not None:
        n_with_ed += 1
        if abs(r.error_from_circuit) < 1e-10:
            n_circuit_zero += 1

print(f"Error decomposition: {n_with_ed} runs have data, {n_circuit_zero} with circuit=0")
print("  Claimed: 100% from MPNN (circuit=0 everywhere)")
print()

# Verify noisy gains
if noisy:
    gains = [r.mean_gain_pct for r in noisy if r.mean_gain_pct is not None]
    r2s = [r.r_squared for r in noisy if r.r_squared is not None]
    print(f"Noisy runs: {len(noisy)}")
    if gains:
        print(f"  Mean gain: {sum(gains) / len(gains):.1f}% (claimed: +28.5%)")
    if r2s:
        print(f"  Mean R2: {sum(r2s) / len(r2s):.4f} (claimed: 0.968)")
