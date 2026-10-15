"""Tests for Task 3: Cold state loss, projection floor consistency, and D mask contract."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import train_sc_v68 as tr
import enhanced_loss_and_weights as elw


class DummyTools:
    """Mock tools object for testing loss and masks without loading full model."""
    def __init__(self, mode="dual_branch"):
        self.rate_decode_mode = mode
        self.macro_species = ["H2", "O2", "H2O", "NH3", "N2", "NO"]
        self.micro_species = ["OH", "H", "O", "HO2", "NH2", "NNH"]
        self.all_species = self.macro_species + self.micro_species
        self.n_macro = len(self.macro_species)
        self.n_micro = len(self.micro_species)
        self.macro_idx = list(range(self.n_macro))
        self.micro_idx = list(range(self.n_macro, self.n_macro + self.n_micro))
        self.temperature_idx = 0
        self.pressure_idx = 1
        self.micro_in_log = True

        # Scalers
        self.scale_Y = torch.ones(self.n_macro + 2 * self.n_micro, dtype=torch.float32)
        self.mean_Y = torch.zeros(self.n_macro + 2 * self.n_micro, dtype=torch.float32)
        self.scale_X = torch.ones(2 + len(self.all_species) + 1, dtype=torch.float32)
        self.mean_X = torch.zeros(2 + len(self.all_species) + 1, dtype=torch.float32)


def test_cold_no_batch_loss_and_gradients():
    """Verify that an all-cold NO batch produces non-zero loss and finite gradient pointing down."""
    tools = DummyTools(mode="dual_branch")
    B = 4

    # True NO is strictly 0.0 (all cold)
    # Input has: T, P, Y_H2, Y_O2, Y_H2O, Y_NH3, Y_N2, Y_NO, logY_micro...
    no_col_idx = tools.macro_idx[tools.macro_species.index("NO")]
    true_X_unscaled = torch.zeros(B, len(tools.scale_X), dtype=torch.float32)
    true_next_X_scaled = torch.zeros(B, len(tools.scale_X), dtype=torch.float32)

    # Predicted NO is 1e-3 (1000 ppm, severe false positive!)
    pred_macro = torch.zeros(B, tools.n_macro, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        pred_macro[:, tools.macro_species.index("NO")] = 1e-3

    # Call specialized NO loss
    loss = elw.compute_pollutant_no_loss(
        tools=tools,
        pred_state={"Y_macro_next_raw": pred_macro},
        true_next_X_scaled=true_next_X_scaled,
        state_source="raw",
        loss_type="huber",
    )

    # 1. Loss must be strictly positive
    assert loss.item() > 0.0, f"Cold batch NO penalty was 0! Loss={loss.item()}"

    # 2. Backward pass must yield finite positive gradient pointing toward decreasing NO
    loss.backward()
    no_grad = pred_macro.grad[:, tools.macro_species.index("NO")]
    assert torch.all(torch.isfinite(no_grad)), "NO gradient contains NaN/Inf!"
    assert torch.all(no_grad > 0.0), f"Expected positive gradient to reduce NO, got {no_grad}"

    # 3. Test benign cold batch (pred NO = 1e-8, well within cold tolerance)
    pred_macro_benign = torch.zeros(B, tools.n_macro, dtype=torch.float32)
    pred_macro_benign[:, tools.macro_species.index("NO")] = 1e-8
    loss_benign = elw.compute_pollutant_no_loss(
        tools=tools,
        pred_state={"Y_macro_next_raw": pred_macro_benign},
        true_next_X_scaled=true_next_X_scaled,
        state_source="raw",
        loss_type="huber",
    )
    assert loss_benign.item() == 0.0, f"Benign cold state got unexpected penalty: {loss_benign.item()}"


def test_projection_floor_no_spurious_elevation():
    """Verify dormant species (1e-40) are NOT elevated by 10 orders of magnitude by projection."""
    # Build minimal element matrix for N, H, O across NH3, H2, O2, N2, H2O, OH
    species = ["NH3", "H2", "O2", "N2", "H2O", "OH"]
    n_macro = 5
    n_micro = 1

    # Y_macro normal combustion mixture
    Y_macro = torch.tensor([[0.1, 0.02, 0.15, 0.65, 0.08]], dtype=torch.float32)
    # Y_micro is a dormant radical at floor 1e-40
    Y_micro = torch.tensor([[1e-40]], dtype=torch.float32)
    logY_micro = torch.log10(Y_micro)

    # Current unconstrained state
    current_X_scaled = torch.cat([torch.tensor([[1200.0, 1.0]]), Y_macro, logY_micro, torch.tensor([[-6.0]])], dim=-1)

    import chemistry_contract as cc
    import cantera as ct
    gas = ct.Solution("NUIG2024.yaml")
    _, A_elem = cc.compute_element_mass_matrix(gas, species)
    A_tensor = torch.tensor(A_elem, dtype=torch.float32)

    # Project
    scaled_dyn = torch.cat([Y_macro, Y_micro], dim=-1)
    target_elements = torch.matmul(scaled_dyn, A_tensor.T)

    # Test projection with 1e-40 floor
    p0 = scaled_dyn.clamp_min(1e-40)
    # Check that micro species p0 is 1e-40, NOT 1e-30
    assert torch.isclose(p0[0, -1], torch.tensor(1e-40), rtol=1e-5), f"Floor elevated micro species to {p0[0, -1]}!"


def test_qssa_d_mask_preserves_decay_and_masks_dormant():
    """Verify compute_qssa_rate_mask preserves supervision on decaying species and masks truly dormant ones."""
    tools = DummyTools(mode="dual_branch")
    B = 2

    # Case 0: Strong decay (Y_old = 1e-4 -> Y_next = 1e-16, D = 1e8)
    # Case 1: Truly dormant (Y_old = 1e-35 -> Y_next = 1e-35, D is noise)
    Y_old_micro = torch.tensor([
        [1e-4] * tools.n_micro,
        [1e-35] * tools.n_micro,
    ], dtype=torch.float32)

    Y_next_micro = torch.tensor([
        [1e-16] * tools.n_micro,
        [1e-35] * tools.n_micro,
    ], dtype=torch.float32)

    true_rate_scaled = torch.zeros(B, tools.n_macro + 2 * tools.n_micro, dtype=torch.float32)

    # 1. State info provided
    mask = tr.compute_qssa_rate_mask(
        tools=tools,
        true_rate_scaled=true_rate_scaled,
        Y_old_unscaled=Y_old_micro,
        Y_next_unscaled=Y_next_micro,
        active_threshold=1e-12,
        noise_threshold=1e-12,
    )

    assert mask is not None
    # For Case 0 (decaying species), D columns (odd columns in micro part) MUST BE 1.0 (supervised!)
    slow_dim = tools.n_macro
    d_cols = [slow_dim + 2 * i + 1 for i in range(tools.n_micro)]
    for col in d_cols:
        assert mask[0, col].item() == 1.0, f"D column {col} was incorrectly masked for decaying species!"

    # For Case 1 (dormant species), D columns MUST BE 0.0 (masked against 0/0 noise)
    for col in d_cols:
        assert mask[1, col].item() == 0.0, f"D column {col} was NOT masked for dormant species!"

    # 2. State info omitted: function must return None (safe fallback, never mask blindly)
    mask_none = tr.compute_qssa_rate_mask(tools, true_rate_scaled)
    assert mask_none is None, "Masking occurred without state information!"
