#!/usr/bin/env python
"""D1 with regularization: Fix overfitting problem in weight-space phase detection.

Hypothesis: Adding dropout=0.1 to the MLP makes peak detection robust and
reproducible across all seeds (not just 2/3).

D1-dense showed: overfitting (loss=0) shifts peak to h≈0.69.
Fix: dropout prevents memorization → peak stays near h_c=1.0.

Expected: ~3 min execution (5 seeds × 2 variants × 6000 epochs).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

from src.poc.v6 import ClassicalSolver, HamiltonianBuilder, HVACircuitBuilder, make_lattice

builder, solver, hva = HamiltonianBuilder(), ClassicalSolver(), HVACircuitBuilder()
estimator = StatevectorEstimator()
N, p = 6, 2
qc, _ = hva.create(N, p, make_lattice("chain_1d", N, J=1.0, h=1.0))
n_params = qc.num_parameters

print("D1 Regularized: Weight-Space Phase Detection with Dropout")
print("=" * 70)


def generate_vqe_data(h_values, seed):
    """Run VQE descending sweep."""
    np.random.seed(seed)
    theta_data = np.zeros((len(h_values), n_params))
    h_sorted = np.sort(h_values)[::-1]
    h_to_idx = {float(f"{h:.4f}"): i for i, h in enumerate(h_values)}
    prev = np.random.uniform(-0.01, 0.01, n_params)

    for h in h_sorted:
        lat = make_lattice("chain_1d", N, J=1.0, h=h)
        H = builder.build(lat)

        def cost_fn(params, _H=H):
            bound = qc.assign_parameters(params)
            return float(estimator.run([(bound, _H)]).result()[0].data.evs)

        best_e, best_t = float("inf"), prev.copy()
        for r in range(3):
            x0 = prev + np.random.normal(0, 0.1, n_params) if r > 0 else prev.copy()
            x0 = np.clip(x0, -np.pi, np.pi)
            res = minimize(
                cost_fn,
                x0,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 300, "ftol": 1e-12},
            )
            if res.fun < best_e:
                best_e, best_t = res.fun, res.x.copy()

        key = f"{h:.4f}"
        if key in h_to_idx:
            theta_data[h_to_idx[key]] = best_t.copy()
        prev = best_t.copy()

    return theta_data


def train_mlp(h_values, theta_data, seed, dropout=0.0, early_stop_loss=None):
    """Train MLP with optional dropout and early stopping."""
    torch.manual_seed(seed)
    n_out = theta_data.shape[1]

    if dropout > 0:
        model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_out),
        )
    else:
        model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_out),
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X = torch.tensor(h_values.reshape(-1, 1), dtype=torch.float32)
    Y = torch.tensor(theta_data, dtype=torch.float32)

    final_loss = 0.0
    stopped_epoch = 6000
    for epoch in range(6000):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        # Early stopping at target loss
        if early_stop_loss is not None and final_loss <= early_stop_loss:
            stopped_epoch = epoch
            break

    return model, final_loss, stopped_epoch


def compute_gradients(model, h_probe):
    """Compute ||dθ_pred/dh|| via finite differences."""
    model.eval()
    epsilon = 0.02
    grad_norms = np.zeros(len(h_probe))
    for i, h in enumerate(h_probe):
        h_p = torch.tensor([[h + epsilon]], dtype=torch.float32)
        h_m = torch.tensor([[h - epsilon]], dtype=torch.float32)
        with torch.no_grad():
            pred_p = model(h_p).numpy().flatten()
            pred_m = model(h_m).numpy().flatten()
        grad = (pred_p - pred_m) / (2 * epsilon)
        grad_norms[i] = float(np.linalg.norm(grad))
    return grad_norms


# ── Generate VQE data (full h-range) ──
h_full = np.linspace(0.5, 2.5, 40)
h_probe = np.linspace(0.5, 2.5, 50)
seeds = [42, 43, 44, 45, 46]  # 5 seeds for robustness

print("\nGenerating VQE data (40 h-points, full range [0.5, 2.5])...")
vqe_cache = {}
for seed in seeds:
    print(f"  Seed {seed}...", end=" ", flush=True)
    vqe_cache[seed] = generate_vqe_data(h_full, seed)
    print("done")

# ── Run 3 variants per seed ──
print("\nTraining models (3 variants × 5 seeds)...")
print(f"{'Seed':<6}{'Variant':<20}{'Loss':<10}{'Peak h':<10}{'|h-h_c|':<10}{'Epoch':<8}")
print("-" * 64)

peaks_no_reg = []
peaks_dropout = []
peaks_early_stop = []

for seed in seeds:
    theta_data = vqe_cache[seed]

    # Variant 1: No regularization (original D1)
    model_noreg, loss_noreg, ep_noreg = train_mlp(h_full, theta_data, seed, dropout=0.0)
    grad_noreg = compute_gradients(model_noreg, h_probe)
    peak_noreg = float(h_probe[np.argmax(grad_noreg)])
    peaks_no_reg.append(peak_noreg)
    print(
        f"{seed:<6}{'No reg':<20}{loss_noreg:<10.6f}{peak_noreg:<10.2f}{abs(peak_noreg - 1.0):<10.2f}{ep_noreg:<8}"
    )

    # Variant 2: Dropout=0.1
    model_drop, loss_drop, ep_drop = train_mlp(h_full, theta_data, seed, dropout=0.1)
    grad_drop = compute_gradients(model_drop, h_probe)
    peak_drop = float(h_probe[np.argmax(grad_drop)])
    peaks_dropout.append(peak_drop)
    print(
        f"{'':<6}{'Dropout=0.1':<20}{loss_drop:<10.6f}{peak_drop:<10.2f}{abs(peak_drop - 1.0):<10.2f}{ep_drop:<8}"
    )

    # Variant 3: Early stop at loss=0.002
    model_es, loss_es, ep_es = train_mlp(
        h_full, theta_data, seed, dropout=0.0, early_stop_loss=0.002
    )
    grad_es = compute_gradients(model_es, h_probe)
    peak_es = float(h_probe[np.argmax(grad_es)])
    peaks_early_stop.append(peak_es)
    print(
        f"{'':<6}{'EarlyStop@0.002':<20}{loss_es:<10.6f}{peak_es:<10.2f}{abs(peak_es - 1.0):<10.2f}{ep_es:<8}"
    )

# ── Summary ──
print("\n" + "=" * 70)
print("SUMMARY: Peak Detection Reliability (5 seeds)")
print("=" * 70)


def stats(peaks):
    return np.mean(peaks), np.std(peaks), np.mean(np.abs(np.array(peaks) - 1.0))


m1, s1, d1 = stats(peaks_no_reg)
m2, s2, d2 = stats(peaks_dropout)
m3, s3, d3 = stats(peaks_early_stop)

print(f"\n{'Variant':<20}{'Mean peak':<12}{'Std':<10}{'Mean |h-h_c|':<14}{'Reliable?':<10}")
print("-" * 66)
print(f"{'No reg':<20}{m1:<12.2f}{s1:<10.2f}{d1:<14.2f}{'✅' if s1 < 0.3 else '❌'}")
print(f"{'Dropout=0.1':<20}{m2:<12.2f}{s2:<10.2f}{d2:<14.2f}{'✅' if s2 < 0.3 else '❌'}")
print(f"{'EarlyStop@0.002':<20}{m3:<12.2f}{s3:<10.2f}{d3:<14.2f}{'✅' if s3 < 0.3 else '❌'}")

print("\nBest variant: ", end="")
best_std = min(s1, s2, s3)
if best_std == s2:
    print(f"Dropout=0.1 (std={s2:.2f}, mean peak={m2:.2f})")
elif best_std == s3:
    print(f"EarlyStop@0.002 (std={s3:.2f}, mean peak={m3:.2f})")
else:
    print(f"No reg (std={s1:.2f}, mean peak={m1:.2f})")

print("\nConclusion:")
if s2 < 0.3 and d2 < 0.5:
    print(f"  ✅ Dropout=0.1 makes D1 robust: peak at h≈{m2:.2f}±{s2:.2f} (h_c=1.0)")
elif s3 < 0.3 and d3 < 0.5:
    print(f"  ✅ EarlyStop@0.002 makes D1 robust: peak at h≈{m3:.2f}±{s3:.2f} (h_c=1.0)")
else:
    print("  ⚠️ Neither regularization fully stabilizes the peak")
    print("     Consider: ensemble of 5+ models, or different architecture")
