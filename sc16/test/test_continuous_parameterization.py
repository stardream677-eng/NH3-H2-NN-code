"""Tests for Task 5: Continuous QSSA parameterization (dual_branch_pd_v1) and unified integrator."""

import numpy as np
import pytest
import torch

import kinetics_parameterization as kp


def test_pd_roundtrip_continuous_encoding_decoding():
    """Verify encode_pd_targets and decode_pd_outputs round-trip across 15 orders of magnitude."""
    P_vals = torch.tensor([0.0, 1e-18, 1e-12, 1e-6, 1e-3, 1.0, 1e3, 1e5], dtype=torch.float64)
    D_vals = torch.tensor([0.0, 1e-8, 1e-4, 1.0, 1e3, 1e6, 1e8, 1e9], dtype=torch.float64)

    z_P, z_D = kp.encode_pd_targets(P_vals, D_vals)
    P_rec, D_rec = kp.decode_pd_outputs(z_P, z_D)

    # Decode of non-zero values above eps matches with high precision
    significant_P = P_vals > 1e-15
    torch.testing.assert_close(P_rec[significant_P], P_vals[significant_P], rtol=1e-5, atol=1e-18)

    significant_D = D_vals > 1e-6
    torch.testing.assert_close(D_rec[significant_D], D_vals[significant_D], rtol=1e-5, atol=1e-8)

    # Zero P and D decode strictly to 0.0 without bias
    assert P_rec[0].item() == 0.0
    assert D_rec[0].item() == 0.0


def test_integrator_euler_limit_d_zero():
    """Verify that when D=0, integrator converges exactly to Y_old + P*dt with finite autograd."""
    Y_old = torch.tensor([1e-4], dtype=torch.float64, requires_grad=True)
    P = torch.tensor([5.0], dtype=torch.float64, requires_grad=True)
    D = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    dt = torch.tensor([1e-5], dtype=torch.float64, requires_grad=True)

    Y_next = kp.integrate_qssa_continuous(Y_old, P, D, dt)
    expected = Y_old + P * dt

    torch.testing.assert_close(Y_next, expected, rtol=1e-10, atol=1e-14)

    # Gradients must be finite and exact
    Y_next.backward()
    assert torch.isclose(Y_old.grad, torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(P.grad, dt.detach())
    assert torch.isclose(dt.grad, P.detach())
    assert torch.isfinite(D.grad)


def test_integrator_stiff_qssa_limit_large_d():
    """Verify that when D*dt >> 1, integrator converges to steady-state P/D exactly."""
    Y_old = torch.tensor([1e-3], dtype=torch.float64)
    P = torch.tensor([1e4], dtype=torch.float64)
    D = torch.tensor([1e8], dtype=torch.float64)
    dt = torch.tensor([1e-4], dtype=torch.float64)  # D*dt = 1e4 >> 1

    Y_next = kp.integrate_qssa_continuous(Y_old, P, D, dt)
    expected_ss = P / D  # 1e-4

    torch.testing.assert_close(Y_next, expected_ss, rtol=1e-8, atol=1e-14)


def test_integrator_smoothness_across_d_zero_boundary():
    """Verify that D scanning smoothly across 0 to positive produces no jump."""
    Y_old = torch.tensor([1e-4], dtype=torch.float64)
    P = torch.tensor([2.0], dtype=torch.float64)
    dt = torch.tensor([1e-6], dtype=torch.float64)

    # Scan D from 0 to 1e4
    D_grid = torch.logspace(-4, 4, 100, dtype=torch.float64)
    Y_d0 = kp.integrate_qssa_continuous(Y_old, P, torch.tensor([0.0], dtype=torch.float64), dt).item()

    Y_curve = [kp.integrate_qssa_continuous(Y_old, P, d.unsqueeze(0), dt).item() for d in D_grid]

    # As D -> 0, Y_curve must continuously approach Y_d0
    assert np.isclose(Y_curve[0], Y_d0, atol=1e-12)
    # The curve must be strictly monotonic decreasing as destruction increases
    diffs = np.diff(Y_curve)
    assert np.all(diffs <= 1e-14), "Integrator response was not monotonic with increasing destruction!"


def test_absence_of_legacy_sentinel_discontinuity():
    """Demonstrate that dual_branch_pd_v1 has NO 29-order-of-magnitude jump near z2=-40."""
    # In legacy qssa, when z2 crosses -40 from -39.999 to -40.001:
    # z1 was interpreted as log10(P/D) for z2 > -40, but log10(P) for z2 <= -40, causing a 29-dex jump.
    z_P = torch.tensor([-10.0, -10.0], dtype=torch.float64)
    # Head outputs near the legacy boundary
    z_D_above = torch.tensor([-39.999], dtype=torch.float64)
    z_D_below = torch.tensor([-40.001], dtype=torch.float64)

    P_a, D_a = kp.decode_pd_outputs(z_P[:1], z_D_above)
    P_b, D_b = kp.decode_pd_outputs(z_P[:1], z_D_below)

    # Decoded P and D are continuous and virtually identical
    assert torch.isclose(P_a, P_b, atol=1e-15)
    assert torch.isclose(D_a, D_b, atol=1e-15)

    # Integrator outputs are smooth and virtually identical
    Y_old = torch.tensor([1e-6], dtype=torch.float64)
    dt = torch.tensor([1e-6], dtype=torch.float64)
    Y_next_a = kp.integrate_qssa_continuous(Y_old, P_a, D_a, dt)
    Y_next_b = kp.integrate_qssa_continuous(Y_old, P_b, D_b, dt)

    torch.testing.assert_close(Y_next_a, Y_next_b, atol=1e-15, rtol=1e-6)
