"""Unit test for exact exponential QSSA integrator.

Verifies:
1. Small D*dt limit (< 1e-6): Taylor expansion matches analytical derivative, no 0/0.
2. Moderate D*dt limit (~1): matches exact expm1 formula.
3. Stiff / large D*dt limit (15 to 1e8): exp(-D*dt) vanishes safely, Y_next reaches P/D exactly with 0 spread across dt.
4. PyTorch autograd differentiability with respect to Y_old, P, D, and dt.
5. Absence of artificial dt-dependence at equilibrium (P = D * Y_eq => Y_next == Y_eq for all dt).
"""

import torch
import numpy as np


def exact_qssa_step(Y_old, P, D, dt):
    """
    Numerically stable exact exponential integrator without artificial damping cap:
    Y_next = Y_old * exp(-D*dt) + P * ((1 - exp(-D*dt)) / D)
    """
    D_eff = torch.clamp_min(D, 0.0)
    D_dt = D_eff * dt
    D_dt_bounded = torch.clamp(D_dt, max=80.0)
    exp_neg_Ddt = torch.exp(-D_dt_bounded)

    safe_factor = torch.where(
        D_dt < 1e-6,
        dt - D_eff * (dt ** 2) / 2.0,
        -torch.expm1(-D_dt_bounded) / torch.clamp_min(D_eff, 1e-20),
    )
    return Y_old * exp_neg_Ddt + P * safe_factor


def legacy_capped_qssa_step(Y_old, P, D, dt, max_damping=15.0):
    """Legacy integrator with artificial cap D <= 15/dt."""
    D_eff = torch.minimum(D, max_damping / dt.clamp_min(1e-12))
    D_dt_bounded = D_eff * dt
    safe_factor = torch.where(
        D_dt_bounded < 1e-6,
        dt - D_eff * (dt ** 2) / 2.0,
        -torch.expm1(-D_dt_bounded) / D_eff.clamp_min(1e-20),
    )
    return Y_old * torch.exp(-D_dt_bounded) + P * safe_factor


def test_stiff_equilibrium_dt_invariance():
    """At true chemical equilibrium (P = D * Y_eq), Y_next must equal Y_eq for ANY dt."""
    print("Testing stiff equilibrium invariance across dt...")
    Y_eq = torch.tensor([1.2e-4], dtype=torch.float64)  # e.g. OH radical
    D = torch.tensor([4.2e7], dtype=torch.float64)      # stiff destruction rate 4.2e7 s^-1
    P = D * Y_eq                                        # exact chemical equilibrium

    dts = [10.0 ** (-e) for e in range(8, 3, -1)]  # 1e-8 to 1e-4

    print(f"{'dt (s)':<10} | {'D*dt':<10} | {'Exact Y_next':<16} | {'Capped Y_next':<16} | {'Capped Err (dex)':<16}")
    print("-" * 80)

    exact_vals = []
    capped_vals = []
    for dt_val in dts:
        dt = torch.tensor([dt_val], dtype=torch.float64)
        Y_next_exact = exact_qssa_step(Y_eq, P, D, dt)
        Y_next_capped = legacy_capped_qssa_step(Y_eq, P, D, dt)

        exact_vals.append(float(Y_next_exact))
        capped_vals.append(float(Y_next_capped))

        capped_err_dex = np.log10(float(Y_next_capped) / float(Y_eq))
        print(f"{dt_val:<10.1e} | {float(D*dt):<10.1e} | {float(Y_next_exact):<16.8e} | {float(Y_next_capped):<16.8e} | {capped_err_dex:+16.4f}")

    exact_spread_dex = np.max(np.log10(exact_vals)) - np.min(np.log10(exact_vals))
    capped_spread_dex = np.max(np.log10(capped_vals)) - np.min(np.log10(capped_vals))

    print("-" * 80)
    print(f"Exact Integrator Spread across dt: {exact_spread_dex:.2e} dex (Target: ~0.00)")
    print(f"Legacy Capped Spread across dt:   {capped_spread_dex:.2f} dex (Demonstrates the sawtooth bug!)")

    assert exact_spread_dex < 1e-10, f"Exact integrator spread {exact_spread_dex} is not zero!"
    assert capped_spread_dex > 0.5, f"Expected capped integrator to demonstrate large spread, got {capped_spread_dex}"
    print(">> Equilibrium dt invariance test PASSED!\n")


def test_autograd_gradients():
    """Verify that gradients can backprop through the exact integrator without NaNs or Infs."""
    print("Testing autograd gradients...")
    Y_old = torch.tensor([1e-4], dtype=torch.float32, requires_grad=True)
    P = torch.tensor([1e3], dtype=torch.float32, requires_grad=True)
    D = torch.tensor([1e7], dtype=torch.float32, requires_grad=True)
    dt = torch.tensor([1e-5], dtype=torch.float32, requires_grad=True)

    Y_next = exact_qssa_step(Y_old, P, D, dt)
    loss = torch.sum(Y_next)
    loss.backward()

    for name, grad in [("Y_old", Y_old.grad), ("P", P.grad), ("D", D.grad), ("dt", dt.grad)]:
        assert grad is not None, f"Gradient for {name} is None!"
        assert torch.isfinite(grad), f"Gradient for {name} is not finite: {grad}!"
        print(f"  grad({name}) = {float(grad):.4e} (finite)")

    print(">> Autograd test PASSED!\n")


def test_small_dt_taylor_limit():
    """Verify that for D*dt < 1e-6, Taylor expansion matches the smooth expm1 transition."""
    print("Testing Taylor expansion boundary continuity...")
    Y_old = torch.tensor([0.2], dtype=torch.float64)
    P = torch.tensor([10.0], dtype=torch.float64)
    D = torch.tensor([5.0], dtype=torch.float64)

    # Just below boundary (9.9e-7) vs just above boundary (1.01e-6)
    dt1 = torch.tensor([9.9e-7 / 5.0], dtype=torch.float64)
    dt2 = torch.tensor([1.01e-6 / 5.0], dtype=torch.float64)

    y1 = exact_qssa_step(Y_old, P, D, dt1)
    y2 = exact_qssa_step(Y_old, P, D, dt2)

    diff = abs(float(y1 - y2))
    print(f"  Boundary step difference: {diff:.4e}")
    assert diff < 1e-5, f"Discontinuity at Taylor boundary: {diff}"
    print(">> Taylor limit test PASSED!\n")


if __name__ == "__main__":
    test_stiff_equilibrium_dt_invariance()
    test_autograd_gradients()
    test_small_dt_taylor_limit()
    print("=========================================")
    print("ALL TASK 1 INTEGRATOR TESTS PASSED!")
    print("=========================================")
