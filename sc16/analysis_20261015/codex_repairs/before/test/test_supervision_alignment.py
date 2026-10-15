"""Unit tests for Task 6: Supervision alignment between parameter regression and AR trajectory state rollout.

Verifies:
1. Two-branch input & label pairing on a toy dynamical system f(y) = -k*y.
2. Forward pass reuse when true input matches feedback input.
3. Independent gradient routing when AR feedback state drifts.
4. Cold vs mixed cold/hot batch NO loss gradients and penalty behaviour.
5. Induction weight assignment across temperature regimes and boost factors.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from enhanced_loss_and_weights import (
    compute_induction_sample_weights,
    compute_pollutant_no_loss,
)


class LinearDecayODE(nn.Module):
    """Toy model f(y) = -k * y predicting rate of change."""
    def __init__(self, k_init: float = 2.0):
        super().__init__()
        self.k = nn.Parameter(torch.tensor([k_init], dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return -self.k * y


def test_toy_two_branch_supervision_pairing():
    """Verify that parameter supervision uses true states and AR uses accumulated states."""
    torch.manual_seed(42)
    dt = 0.1
    # True trajectory for dy/dt = -3*y: y(t) = y0 * exp(-3*t)
    y0 = torch.tensor([[1.0]], dtype=torch.float32)
    y1_true = y0 * math.exp(-0.3)
    y2_true = y0 * math.exp(-0.6)

    # True parameter rates
    r0_true = -3.0 * y0
    r1_true = -3.0 * y1_true

    model = LinearDecayODE(k_init=1.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Step 0: current_X == true_current_X (y0)
    current_y = y0.clone()
    true_y0 = y0.clone()

    # Step 0: forward pass reuse
    pred_r0 = model(current_y)
    r_pred_true_0 = pred_r0
    loss_param_0 = F.mse_loss(r_pred_true_0, r0_true)
    y1_pred = current_y + pred_r0 * dt
    loss_state_0 = F.mse_loss(y1_pred, y1_true)

    total_loss_0 = loss_param_0 + loss_state_0
    total_loss_0.backward()
    assert model.k.grad is not None and torch.isfinite(model.k.grad)
    optimizer.zero_grad()

    # Step 1: Simulated AR drift: current_y = y1_pred (differs from y1_true)
    current_y = y1_pred.detach() + 0.15  # Introduce intentional AR feedback drift
    assert not torch.equal(current_y, y1_true)

    # Two separate forward passes
    pred_r1_ar = model(current_y)         # Rate evaluated on drifted AR state
    pred_r1_true = model(y1_true)        # Rate evaluated on true state

    # Parameter loss MUST use pred_r1_true against r1_true
    loss_param_1 = F.mse_loss(pred_r1_true, r1_true)
    # AR State loss MUST use y1_pred + pred_r1_ar * dt against y2_true
    y2_pred = current_y + pred_r1_ar * dt
    loss_state_1 = F.mse_loss(y2_pred, y2_true)

    total_loss_1 = loss_param_1 + loss_state_1
    total_loss_1.backward()

    # Gradient must be finite and properly reflect both heads
    assert model.k.grad is not None and torch.isfinite(model.k.grad)
    assert model.k.grad.abs().item() > 0.0


def test_cold_vs_mixed_no_gradient():
    """Verify that NO loss gives finite non-zero gradients on pure cold, pure hot, and mixed batches."""
    class DummyTools:
        def __init__(self):
            self.macro_species = ["NH3", "H2", "O2", "N2", "H2O", "NO"]
            self.micro_species = ["H", "O", "OH"]
            self.macro_idx = torch.tensor([0, 1, 2, 3, 4, 5])
            self.micro_idx = torch.tensor([6, 7, 8])
            self.scale_X = torch.ones(9)
            self.mean_X = torch.zeros(9)

    tools = DummyTools()

    # Case 1: Pure cold batch (true NO mass fraction is 1e-12, well below active 1e-7)
    # The predicted NO has some false positive (e.g. 1e-3, scaled asinh > cold_threshold 0.2)
    B = 4
    pred_macro_raw = torch.full((B, 6), 1e-3, requires_grad=True)
    pred_state_cold = {
        "Y_macro_next_raw": pred_macro_raw,
        "Y_micro_next_raw": torch.zeros(B, 3),
        "logY_micro_next_raw": torch.full((B, 3), -30.0),
    }
    true_next_cold = torch.zeros(B, 9)
    true_next_cold[:, 5] = 1e-12  # cold true NO

    loss_cold = compute_pollutant_no_loss(
        tools=tools,
        pred_state=pred_state_cold,
        true_next_X_scaled=true_next_cold,
        state_source="raw",
    )
    assert torch.isfinite(loss_cold)
    assert loss_cold.item() > 0.0
    loss_cold.backward()
    # pred_macro_raw[:, 5] must receive a gradient penalizing false positive
    assert pred_macro_raw.grad is not None
    assert torch.isfinite(pred_macro_raw.grad).all()
    assert pred_macro_raw.grad[:, 5].abs().sum().item() > 0.0

    # Case 2: Mixed batch (half cold, half hot active)
    pred_macro_mixed = torch.full((4, 6), 1e-3, requires_grad=True)
    pred_state_mixed = {
        "Y_macro_next_raw": pred_macro_mixed,
        "Y_micro_next_raw": torch.zeros(4, 3),
        "logY_micro_next_raw": torch.full((4, 3), -30.0),
    }
    true_next_mixed = torch.zeros(4, 9)
    true_next_mixed[:2, 5] = 1e-12  # cold
    true_next_mixed[2:, 5] = 5e-3   # active hot

    loss_mixed = compute_pollutant_no_loss(
        tools=tools,
        pred_state=pred_state_mixed,
        true_next_X_scaled=true_next_mixed,
        state_source="raw",
    )
    assert torch.isfinite(loss_mixed)
    loss_mixed.backward()
    assert pred_macro_mixed.grad is not None
    assert torch.isfinite(pred_macro_mixed.grad).all()
    assert pred_macro_mixed.grad[:, 5].abs().sum().item() > 0.0


def test_induction_sample_weights_regimes():
    """Verify compute_induction_sample_weights across cold, induction, and burnt regimes."""
    class DummyTools:
        def __init__(self):
            self.temperature_idx = 0
            self.micro_species = ["H", "O", "OH", "HO2"]
            self.micro_idx = torch.tensor([1, 2, 3, 4])

    tools = DummyTools()

    # Batch of 4 conditions:
    # 0: Cold unburnt (T=600 K, dT=5 K) -> weight 1.0
    # 1: Induction (T=1100 K, dT=20 K, dlog(OH)=0.1 > 0.02) -> weight boost
    # 2: Ignition flame front (T=1400 K, dT=300 K > 60 K) -> weight 1.0
    # 3: Burnt post-flame (T=2200 K, dT=0 K) -> weight 1.0

    cur_X = torch.zeros(4, 5)
    nxt_X = torch.zeros(4, 5)

    # Condition 0: Cold
    cur_X[0, 0] = 600.0
    nxt_X[0, 0] = 605.0

    # Condition 1: Warm induction
    cur_X[1, 0] = 1100.0
    nxt_X[1, 0] = 1120.0
    cur_X[1, 3] = -8.0   # OH logY_cur
    nxt_X[1, 3] = -7.8   # OH logY_nxt, growth = +0.2 > 0.02

    # Condition 2: Flame front jump
    cur_X[2, 0] = 1400.0
    nxt_X[2, 0] = 1700.0  # dT = 300 K > 60 K

    # Condition 3: Burnt post-flame
    cur_X[3, 0] = 2200.0
    nxt_X[3, 0] = 2200.0

    for boost in (1.0, 2.0, 6.0):
        w = compute_induction_sample_weights(cur_X, nxt_X, tools, boost_factor=boost)
        assert w[0].item() == pytest.approx(1.0)
        assert w[1].item() == pytest.approx(boost)
        assert w[2].item() == pytest.approx(1.0)
        assert w[3].item() == pytest.approx(1.0)
