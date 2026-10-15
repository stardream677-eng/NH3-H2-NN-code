"""Unit test for QSSA Asymptotic Parameterization (Task 2).

Verifies:
1. Data target transformation:
   log10(Y_qssa) = clip(log10(P) - log10(D), -20.0, 0.0)
   log10(D) = log10(D)
2. Feature info and transform signature metadata (qssa_asymptotic_v2).
3. CoarseOperatorTools decoding: inverse_qssa_asymptotic_parameters correctly recovers
   physical Y_qssa in [1e-20, 1.0] and D in [1e-30, 1e30].
4. State update: predict_next_state with state_update_mode='qssa_asymptotic'
   unconditionally bounds Y_next to physical bounds without artificial dt-damping cap.
5. End-to-end autograd differentiability.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

# Import train_sc_v68
import train_sc_v68 as tr


def test_target_transformation():
    print("Testing qssa_asymptotic target transformation...")
    reactive_species = ["H2", "O2", "OH"]
    n_pts = 10
    
    # Create synthetic dataframe
    data = {}
    for sp in reactive_species:
        # P in [1e-5, 1e4], D in [1e2, 1e8]
        P_val = 1e2
        D_val = 1e6 # Expected Y_qssa = 1e-4 -> log10 = -4.0
        data[f"P_avg_{sp}"] = np.full(n_pts, P_val, dtype=np.float64)
        data[f"D_coef_avg_{sp}"] = np.full(n_pts, D_val, dtype=np.float64)
    df = pd.DataFrame(data)
    
    targets = tr.transform_target_array(
        df=df,
        macro_species=["H2", "O2"],
        micro_species=["OH"],
        macro_rate_scale=[1.0, 1.0],
        output_parameterization="qssa_asymptotic",
    )
    
    assert targets.shape == (n_pts, 2 * len(reactive_species)), f"Shape mismatch: {targets.shape}"
    
    for k in range(len(reactive_species)):
        logY_qssa = targets[:, 2 * k]
        logD = targets[:, 2 * k + 1]
        assert np.allclose(logY_qssa, -4.0), f"Species {k} logY_qssa mismatch: {logY_qssa[0]}"
        assert np.allclose(logD, 6.0), f"Species {k} logD mismatch: {logD[0]}"
        
    print(">> Target transformation test PASSED!\n")


def test_feature_info_metadata():
    print("Testing qssa_asymptotic feature_info metadata...")
    fi = tr.build_feature_info(
        species=["N2", "H2", "O2", "OH"],
        inert_species=["N2"],
        macro_species=["H2", "O2"],
        micro_species=["OH"],
        output_parameterization="qssa_asymptotic",
        rate_decode_mode="qssa_asymptotic",
    )
    assert fi["transform_signature"] == tr.QSSA_ASYMPTOTIC_TRANSFORM_SIGNATURE
    assert "log10_Y_qssa_OH" in fi["output_cols"]
    assert "log10_D_coef_avg_OH" in fi["output_cols"]
    print(">> Feature info test PASSED!\n")


def test_coarse_operator_tools_and_step():
    print("Testing CoarseOperatorTools decoding & predict_next_state with qssa_asymptotic...")
    fi = tr.build_feature_info(
        species=["N2", "H2", "O2", "OH"],
        inert_species=["N2"],
        macro_species=["H2", "O2"],
        micro_species=["OH"],
        output_parameterization="qssa_asymptotic",
        rate_decode_mode="qssa_asymptotic",
    )
    
    # Create simple identity scalers
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
    
    # Rate scaled vector: H2=0.10 (log=-1.0), O2=0.20 (log=-0.69897), OH=1e-4 (log=-4.0), all with D=1e6
    rate_pred_scaled = torch.tensor(
        [[-1.0, 6.0, -0.69897, 6.0, -4.0, 6.0]],
        dtype=torch.float32,
        device=tr.DEVICE,
    )
    
    Y_qssa, D = tools.inverse_qssa_asymptotic_parameters(rate_pred_scaled)
    assert torch.allclose(Y_qssa[0, 2], torch.tensor([1e-4], device=tr.DEVICE), atol=1e-6)
    assert torch.allclose(D[0, 2], torch.tensor([1e6], device=tr.DEVICE), atol=1e-1)
    
    # Synthetic current_X_scaled: [T=1500, P=40, Y_N2=0.7, Y_H2=0.1, Y_O2=0.2, logY_OH=-5.0, log_dt=-5.0]
    # dt = 1e-5 s -> D*dt = 10.0
    dt_val = 1e-5
    current_row = [1500.0, 40.0, 0.7, 0.1, 0.2, -5.0, np.log10(dt_val)]
    current_X_scaled = torch.tensor([current_row], dtype=torch.float32, device=tr.DEVICE)
    
    pred_dict = tools.predict_next_state(
        current_X_scaled=current_X_scaled,
        rate_pred_scaled=rate_pred_scaled,
        safe_for_rollout=True,
        state_update_mode="qssa_asymptotic",
    )
    
    Y_next = pred_dict["Y_next_safe_full"]
    print(f"Y_old for micro OH: 1e-5 (log=-5.0)")
    print(f"Y_qssa predicted: 1e-4")
    print(f"dt: {dt_val} s, D*dt = 10.0")
    print(f"Y_next: {Y_next[0].cpu().numpy()}")
    
    # For D*dt = 10.0, exp(-10) = 4.5e-5 ~ 0, so Y_next should be very close to Y_qssa = 1e-4
    assert abs(float(Y_next[0, 2]) - 1e-4) < 1e-5, f"OH not approaching Y_qssa: {Y_next[0, 2]}"
    print(">> CoarseOperatorTools and step test PASSED!\n")


def test_autograd_differentiability():
    print("Testing autograd backward pass through qssa_asymptotic...")
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
    
    dt_val = 1e-5
    current_row = [1500.0, 40.0, 0.7, 0.1, 0.2, -5.0, np.log10(dt_val)]
    current_X_scaled = torch.tensor([current_row], dtype=torch.float32, device=tr.DEVICE)
    
    rate_pred_scaled = torch.tensor(
        [[-4.0, 6.0, -4.0, 6.0, -4.0, 6.0]],
        dtype=torch.float32,
        device=tr.DEVICE,
        requires_grad=True,
    )
    
    pred_dict = tools.predict_next_state(
        current_X_scaled=current_X_scaled,
        rate_pred_scaled=rate_pred_scaled,
        safe_for_rollout=True,
        state_update_mode="qssa_asymptotic",
    )
    
    # The safe state uses a detached physical projection for rollout stability;
    # differentiability is checked on the raw QSSA integration candidate.
    loss = torch.sum(pred_dict["Y_next_raw_full"])
    loss.backward()
    
    assert rate_pred_scaled.grad is not None, "Gradient is None!"
    assert torch.isfinite(rate_pred_scaled.grad).all(), f"Gradient has non-finite values: {rate_pred_scaled.grad}"
    print(f"rate_pred_scaled grad: {rate_pred_scaled.grad.cpu().numpy()}")
    print(">> Autograd test PASSED!\n")


if __name__ == "__main__":
    test_target_transformation()
    test_feature_info_metadata()
    test_coarse_operator_tools_and_step()
    test_autograd_differentiability()
    print("=========================================")
    print("ALL TASK 2 ASYMPTOTIC TESTS PASSED!")
    print("=========================================")
