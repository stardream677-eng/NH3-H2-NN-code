"""Unit test for Phase 2:
1. Task 4: Manifold projection integration in predict_next_state (Simplex mass sum and Dykstra elemental conservation).
2. Task 5: Trust region bounding and dead-zone masking for dormant trace radicals.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

import train_sc_v68 as tr
tr.DEVICE = torch.device("cpu")


def test_qssa_manifold_projection():
    print("Testing QSSA manifold projection in predict_next_state...")
    fi = tr.build_feature_info(
        species=["N2", "H2", "O2", "OH"],
        inert_species=["N2"],
        macro_species=["H2", "O2"],
        micro_species=["OH"],
        output_parameterization="qssa_asymptotic",
        rate_decode_mode="qssa_asymptotic",
    )
    
    n_in = len(fi["input_cols"])
    n_out = len(fi["output_cols"])
    
    class MockScaler:
        def __init__(self, n):
            self.mean_ = np.zeros(n, dtype=np.float32)
            self.scale_ = np.ones(n, dtype=np.float32)
            self.feature_names_in_ = np.array([f"f_{i}" for i in range(n)])
            
    sx = MockScaler(n_in)
    sy = MockScaler(n_out)
    
    tools = tr.CoarseOperatorTools(
        feature_info=fi,
        scaler_X=sx,
        scaler_Y=sy,
        rate_decode_mode="qssa_asymptotic",
    )
    
    # Simulate an unconstrained state prediction that would drift mass to 1.15
    # Rate prediction asking for high concentrations
    rate_pred_scaled = torch.tensor(
        [[-0.5, 4.0, -0.5, 4.0, -1.0, 4.0]], # High Y_qssa for H2, O2, OH
        dtype=torch.float32,
        device=tr.DEVICE,
    )
    
    # Input state with N2 = 0.70
    current_row = [1500.0, 40.0, 0.70, 0.10, 0.20, -4.0, np.log10(1e-4)]
    current_X_scaled = torch.tensor([current_row], dtype=torch.float32, device=tr.DEVICE)
    
    # 1. Test with safe_for_rollout=True (Feedback mode: must project to exact mass 1.0)
    pred_safe = tools.predict_next_state(
        current_X_scaled=current_X_scaled,
        rate_pred_scaled=rate_pred_scaled,
        safe_for_rollout=True,
        state_update_mode="qssa_asymptotic",
    )
    
    Y_next_constrained = pred_safe["Y_next_constrained_full"][0]
    inert_sum = 0.70
    dyn_sum = Y_next_constrained.sum().item()
    total_mass = dyn_sum + inert_sum
    print(f"  Constrained dynamic species sum: {dyn_sum:.8f}")
    print(f"  Total mass (dyn + inert): {total_mass:.8f}")
    assert abs(total_mass - 1.0) < 1e-5, f"Mass conservation violated in constrained state: {total_mass}"
    
    # 2. Test that raw unconstrained candidate is also preserved in dict for state_loss_source='raw'
    assert "Y_macro_next_raw" in pred_safe
    assert "Y_next_raw_full" in pred_safe
    raw_dyn_sum = pred_safe["Y_next_raw_full"][0].sum().item()
    print(f"  Raw dynamic species sum: {raw_dyn_sum:.8f} (Demonstrates raw is preserved)")
    
    print(">> Manifold projection test PASSED!\n")


def test_trust_region_bounding():
    print("Testing trust_region_fraction bounding in predict_next_state...")
    fi = tr.build_feature_info(
        species=["N2", "H2", "O2", "OH"],
        inert_species=["N2"],
        macro_species=["H2", "O2"],
        micro_species=["OH"],
        output_parameterization="qssa_asymptotic",
        rate_decode_mode="qssa_asymptotic",
    )
    
    class MockScaler:
        def __init__(self, n):
            self.mean_ = np.zeros(n, dtype=np.float32)
            self.scale_ = np.ones(n, dtype=np.float32)
            self.feature_names_in_ = np.array([f"f_{i}" for i in range(n)])
            
    sx = MockScaler(len(fi["input_cols"]))
    sy = MockScaler(len(fi["output_cols"]))
    tools = tr.CoarseOperatorTools(fi, sx, sy, rate_decode_mode="qssa_asymptotic")
    
    # Initial H2 = 0.10, rate predicts jump to 0.80
    current_row = [1500.0, 40.0, 0.70, 0.10, 0.20, -5.0, np.log10(1e-4)]
    current_X_scaled = torch.tensor([current_row], dtype=torch.float32, device=tr.DEVICE)
    rate_pred_scaled = torch.tensor([[-0.1, 6.0, -0.7, 6.0, -2.0, 6.0]], dtype=torch.float32, device=tr.DEVICE)
    
    # With trust_region_fraction = 0.25, max change in macro is 25% of current
    pred = tools.predict_next_state(
        current_X_scaled=current_X_scaled,
        rate_pred_scaled=rate_pred_scaled,
        safe_for_rollout=True,
        trust_region_fraction=0.25,
        state_update_mode="qssa_asymptotic",
    )
    
    Y_macro_raw = pred["Y_macro_next_raw"][0, 0].item() # H2
    print(f"  Initial H2: 0.10, unconstrained would be ~0.80")
    print(f"  With trust_region=0.25, H2 raw is bounded to: {Y_macro_raw:.4f}")
    assert abs(Y_macro_raw - 0.10) <= 0.10 * 0.25 + 1e-5, f"Trust region failed to bound step: {Y_macro_raw}"
    print(">> Trust region test PASSED!\n")


def test_dead_zone_mask():
    print("Testing dead-zone loss masking for dormant trace radicals...")
    fi = tr.build_feature_info(
        species=["N2", "H2", "O2", "OH"],
        inert_species=["N2"],
        macro_species=["H2", "O2"],
        micro_species=["OH"],
        output_parameterization="qssa_asymptotic",
        rate_decode_mode="qssa_asymptotic",
    )
    class MockScaler:
        def __init__(self, n):
            self.mean_ = np.zeros(n, dtype=np.float32)
            self.scale_ = np.ones(n, dtype=np.float32)
            self.feature_names_in_ = np.array([f"f_{i}" for i in range(n)])
            
    sx = MockScaler(len(fi["input_cols"]))
    sy = MockScaler(len(fi["output_cols"]))
    tools = tr.CoarseOperatorTools(fi, sx, sy, rate_decode_mode="qssa_asymptotic")
    
    # Batch with 2 samples:
    # Sample 0: Active flame (log10(Y_qssa) = -2.0, log10(D) = 5.0 for all 3 dynamic species)
    # Sample 1: Dormant cold radical (log10(Y_qssa) = -18.0, log10(D) = 2.0)
    true_rate_scaled = torch.tensor(
        [
            [-2.0, 5.0, -3.0, 4.0, -4.0, 5.0],
            [-18.0, 2.0, -18.0, 3.0, -18.0, 4.0],
        ],
        dtype=torch.float32,
        device=tr.DEVICE,
    )
    
    mask = tr.compute_qssa_rate_mask(tools, true_rate_scaled)
    assert mask is not None
    assert torch.all(mask[0] == 1.0), "Active sample should have full 1.0 mask"
    assert torch.all(mask[1, 0::2] == 1.0), "Dormant creation heads must remain supervised (1.0)"
    assert torch.all(mask[1, 1::2] == 0.0), "Dormant destruction heads must be masked (0.0)"
    
    # Test that masked loss completely ignores arbitrary predictions on masked columns
    pred1 = true_rate_scaled.clone()
    pred2 = true_rate_scaled.clone()
    pred2[1, 1::2] = 999.0  # Massive outlier on dormant destruction heads
    
    l1 = tr.weighted_mse_loss(pred1, true_rate_scaled, mask=mask)
    l2 = tr.weighted_mse_loss(pred2, true_rate_scaled, mask=mask)
    assert torch.allclose(l1, l2), "Masked elements must not contribute to loss"
    print(">> Dead-zone mask test PASSED!\n")


def test_element_conservation_loss_supervises_raw():
    print("Testing compute_element_conservation_loss supervises raw predictions...")
    fi = tr.build_feature_info(
        species=["N2", "H2", "O2", "OH"],
        inert_species=["N2"],
        macro_species=["H2", "O2"],
        micro_species=["OH"],
        output_parameterization="qssa_asymptotic",
        rate_decode_mode="qssa_asymptotic",
    )
    class MockScaler:
        def __init__(self, n):
            self.mean_ = np.zeros(n, dtype=np.float32)
            self.scale_ = np.ones(n, dtype=np.float32)
            self.feature_names_in_ = np.array([f"f_{i}" for i in range(n)])
            
    sx = MockScaler(len(fi["input_cols"]))
    sy = MockScaler(len(fi["output_cols"]))
    tools = tr.CoarseOperatorTools(fi, sx, sy, rate_decode_mode="qssa_asymptotic")
    
    # Construct a pred_state with drifted raw output and projected constrained output
    pred_state = {
        "Y_macro_old": torch.tensor([[0.10, 0.20]], dtype=torch.float32, device=tr.DEVICE),
        "logY_micro_old": torch.tensor([[-4.0]], dtype=torch.float32, device=tr.DEVICE),
        # Raw prediction drifts mass to 1.50 (severe violation)
        "Y_next_raw_full": torch.tensor([[0.50, 0.50, 0.01]], dtype=torch.float32, device=tr.DEVICE),
        # Constrained prediction perfectly projected to 0.30
        "Y_next_constrained_full": torch.tensor([[0.10, 0.20, 1e-4]], dtype=torch.float32, device=tr.DEVICE),
        "Y_next_safe_full": torch.tensor([[0.10, 0.20, 1e-4]], dtype=torch.float32, device=tr.DEVICE),
    }
    
    # In raw mode: loss_element and loss_mass must be non-zero (penalizing raw error)
    loss_elem_raw, loss_mass_raw = tr.compute_element_conservation_loss(
        tools, pred_state, state_loss_source="raw"
    )
    assert loss_mass_raw.item() > 0.05, f"Raw mass loss should be high, got {loss_mass_raw.item()}"
    
    # In constrained mode: mass loss should be near zero
    loss_elem_c, loss_mass_c = tr.compute_element_conservation_loss(
        tools, pred_state, state_loss_source="constrained"
    )
    assert loss_mass_c.item() < 1e-4, f"Constrained mass loss should be near zero, got {loss_mass_c.item()}"
    print(f"  Raw mass loss: {loss_mass_raw.item():.4f} (Strict penalty on model error)")
    print(f"  Constrained mass loss: {loss_mass_c.item():.8f} (Projection satisfies physics)")
    print(">> Element conservation dual-track test PASSED!\n")


def test_concentration_weighted_projection_preserves_radicals():
    print("Testing concentration-weighted projection preserves trace radicals...")
    species = ["N2", "H2", "O2", "H2O", "H", "O", "OH", "HO2"]
    macro = ["H2", "O2", "H2O"]
    micro = ["H", "O", "OH", "HO2"]
    dyn_species = macro + micro
    A_dyn = tr.build_element_matrix(dyn_species)

    # Realistic state with trace radicals
    Y_macro = torch.tensor([[0.05, 0.20, 0.05]], dtype=torch.float32, device=tr.DEVICE)
    Y_micro = torch.tensor([[1e-6, 1e-7, 1e-5, 1e-8]], dtype=torch.float32, device=tr.DEVICE)
    target_dyn = torch.cat([Y_macro, Y_micro], dim=-1)
    inert_sum = 1.0 - target_dyn.sum().item()
    target_elements = torch.matmul(target_dyn, A_dyn.T)

    # Corrupt macro species by 1% relative perturbation
    Y_macro_noisy = Y_macro * (1.0 + 0.01 * torch.tensor([[1.0, -1.0, 0.5]], device=tr.DEVICE))

    # Perform concentration-weighted closed-form projection
    macro_proj, micro_proj, diag = tr.project_physical_state(
        Y_macro_raw=Y_macro_noisy,
        Y_micro_raw=Y_micro,
        inert_sum=inert_sum,
        element_matrix=A_dyn,
        target_elements=target_elements,
        iterations=2,
    )

    # 1. Total mass must be conserved to float32 precision
    total_mass = (macro_proj.sum() + micro_proj.sum() + inert_sum).item()
    assert abs(total_mass - 1.0) < 1e-6, f"Mass conservation violated: {total_mass}"

    # 2. Elemental conservation residual must be tiny
    proj_dyn = torch.cat([macro_proj, micro_proj], dim=-1)
    elem_res = (torch.matmul(proj_dyn, A_dyn.T) - target_elements).abs().max().item()
    assert elem_res < 1e-5, f"Elemental residual too high: {elem_res}"

    # 3. Radicals must NOT be wiped out to zero or negative!
    for sp_name, y_orig, y_p in zip(micro, Y_micro[0], micro_proj[0]):
        rel_diff = abs((y_p.item() - y_orig.item()) / y_orig.item())
        print(f"  Radical {sp_name}: orig={y_orig.item():.2e}, proj={y_p.item():.2e}, rel_diff={rel_diff:.4f}")
        assert y_p.item() > 0.0, f"Radical {sp_name} was wiped out to zero!"
        assert rel_diff < 0.20, f"Radical {sp_name} experienced excessive shift: {rel_diff}"

    # 4. Severe perturbation test (50% extreme noise, answering GPT review)
    Y_macro_severe = Y_macro * (1.0 + 0.50 * torch.tensor([[1.0, -1.0, 0.5]], device=tr.DEVICE))
    macro_proj_s, micro_proj_s, diag_s = tr.project_physical_state(
        Y_macro_raw=Y_macro_severe,
        Y_micro_raw=Y_micro,
        inert_sum=inert_sum,
        element_matrix=A_dyn,
        target_elements=target_elements,
        iterations=2,
    )
    assert not diag_s["projection_failed"], "Severe perturbation caused projection failure!"
    assert diag_s["element_residual_max"].item() < 1e-5, f"Severe perturbation element residual: {diag_s['element_residual_max'].item()}"
    assert (micro_proj_s > 0.0).all(), "Radicals were wiped out under severe perturbation!"
    print(">> Severe 50% perturbation test PASSED (residual < 1e-5, zero radical wipeout)!\n")

    print(">> Concentration-weighted projection test PASSED!\n")


def test_kl_projection_uses_bounded_iterations_and_preserves_elements(monkeypatch):
    macro = ["H2", "O2", "H2O"]
    micro = ["H", "O", "OH", "HO2"]
    element_matrix = tr.build_element_matrix(macro + micro).to(
        dtype=torch.float32, device=tr.DEVICE
    )
    y_macro = torch.tensor([[0.05, 0.20, 0.05]], dtype=torch.float32, device=tr.DEVICE)
    y_micro = torch.tensor(
        [[1e-6, 1e-7, 1e-5, 1e-8]], dtype=torch.float32, device=tr.DEVICE
    )
    inert_sum = 1.0 - (y_macro.sum() + y_micro.sum()).item()
    target_elements = torch.matmul(torch.cat([y_macro, y_micro], dim=1), element_matrix.T)
    y_macro_raw = y_macro * torch.tensor([[1.01, 0.99, 1.005]], device=tr.DEVICE)

    original_solve = torch.linalg.solve
    solve_calls = 0

    def counted_solve(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve", counted_solve)
    macro_proj, micro_proj, diagnostics = tr.project_physical_state(
        Y_macro_raw=y_macro_raw,
        Y_micro_raw=y_micro,
        inert_sum=inert_sum,
        element_matrix=element_matrix,
        target_elements=target_elements,
    )

    projected = torch.cat([macro_proj, micro_proj], dim=1)
    residual = torch.matmul(projected, element_matrix.T) - target_elements
    assert solve_calls <= 8, f"KL projection used {solve_calls} Newton solves"
    assert residual.abs().max().item() < 1e-6
    assert not diagnostics["projection_failed"]
    assert (projected > 0.0).all()


if __name__ == "__main__":
    test_qssa_manifold_projection()
    test_trust_region_bounding()
    test_dead_zone_mask()
    test_element_conservation_loss_supervises_raw()
    test_concentration_weighted_projection_preserves_radicals()
    print("=========================================")
    print("ALL PHASE 2 UNIT TESTS PASSED!")
    print("=========================================")

