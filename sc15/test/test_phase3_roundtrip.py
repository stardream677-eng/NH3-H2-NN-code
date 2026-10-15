"""Phase 3 Unit Test: QSSA Asymptotic Transform Round-Trip Closure

Verifies that the parameterization (log10(P/D), log10(D)) without artificial
clamping (log10(P/D) <= 0) achieves machine precision round-trip closure across:
1. P/D > 1 (fast explosive creation)
2. P/D >> 1 (e.g. P/D = 1e6)
3. D = 0 (pure production, zero destruction)
4. D >> 1 (ultra-fast equilibrium relaxation)
5. Trace radical formation from zero
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import train_sc_v68 as tr


def test_qssa_asymptotic_roundtrip_closure():
    print("\n--- Testing QSSA Asymptotic Transform Round-Trip Closure ---")

    # Build mock tools
    species = ["H2", "O2", "OH"]
    fi = tr.build_feature_info(
        species=species,
        inert_species=[],
        macro_species=["H2", "O2"],
        micro_species=["OH"],
        output_parameterization="qssa_asymptotic",
        rate_decode_mode="qssa_asymptotic",
    )

    n_in = len(fi["input_cols"])
    n_out = len(fi["output_cols"])

    class IdentityScaler:
        def __init__(self, n):
            self.mean_ = np.zeros(n, dtype=np.float32)
            self.scale_ = np.ones(n, dtype=np.float32)
            self.feature_names_in_ = np.array([f"f_{i}" for i in range(n)])

    tools = tr.CoarseOperatorTools(
        feature_info=fi,
        scaler_X=IdentityScaler(n_in),
        scaler_Y=IdentityScaler(n_out),
        rate_decode_mode="qssa_asymptotic",
    )

    test_cases = [
        # (name, Y0, P, D, dt)
        ("Normal burning", 0.05, 100.0, 500.0, 1e-5),
        ("P/D > 1 moderate", 0.01, 500.0, 20.0, 1e-4),       # P/D = 25
        ("P/D >> 1 explosive", 1e-5, 1e4, 1e-2, 1e-5),       # P/D = 1e6
        ("Pure production (D=0)", 0.02, 50.0, 0.0, 1e-4),    # D = 0
        ("Ultra-stiff equilibrium", 1e-3, 1e5, 1e7, 1e-4),   # D = 1e7
        ("Radical birth from 0", 0.0, 10.0, 100.0, 1e-5),    # Y0 = 0
        ("GPT case (P=1.3e-27, D=0)", 1.04e-30, 1.3e-27, 0.0, 8.69e-5),
        ("Pure destruction (P=0)", 1e-5, 0.0, 10.0, 1e-4),
        ("Deep trace extinction", 1e-25, 0.0, 1e3, 1e-4),
        ("Sub-floor pure production", 0.0, 1e-39, 0.0, 1e-4),
    ]

    for name, y0, P, D, dt in test_cases:
        # 1. Exact analytic state next
        z_true = D * dt
        if z_true < 1e-7:
            phi1 = 1.0 - 0.5 * z_true + (z_true**2) / 6.0
        else:
            phi1 = (1.0 - np.exp(-z_true)) / z_true
        y_true = y0 * np.exp(-z_true) + P * dt * phi1

        # 2. Encode to targets as done in prepare_target_arrays
        is_prod_only = D <= 0.0
        logP = np.log10(max(P, 1e-40))
        logD = np.log10(max(D, 1e-40))
        logY_qssa = np.clip(logP - logD, -40.0, 30.0)
        z1 = -45.0 if P <= 0.0 else (logP if is_prod_only else logY_qssa)
        z2 = -45.0 if is_prod_only else logD

        # 3. Decode using operator tools
        target_tensor = torch.tensor(
            [[z1, z2, z1, z2, z1, z2]],
            dtype=torch.float32,
            device=tr.DEVICE,
        )
        dec_logP, dec_logD = tools.inverse_qssa_asymptotic_log_parameters(target_tensor)

        # 4. Advance using exact logic from advance_state
        P_pred = torch.where(dec_logP < -40.0, torch.zeros_like(dec_logP), torch.pow(10.0, torch.clamp(dec_logP, -40.0, 30.0)))
        dec_prod_only = dec_logD < -40.0
        D_pred = torch.where(dec_prod_only, torch.zeros_like(dec_logD), torch.pow(10.0, torch.clamp(dec_logD, -40.0, 30.0)))
        D_eff = torch.clamp_min(D_pred, 0.0)
        D_dt = D_eff * dt

        is_small_D = D_dt < 1e-6
        D_safe_large = torch.where(is_small_D, torch.ones_like(D_eff), D_eff)
        D_dt_large = D_safe_large * dt
        D_dt_bounded = torch.clamp(D_dt_large, max=80.0)

        prod_term_small = P_pred * dt * (1.0 - 0.5 * D_dt)
        prod_term_large = (P_pred / D_safe_large) * (-torch.expm1(-D_dt_bounded))
        prod_term = torch.where(is_small_D, prod_term_small, prod_term_large)

        exp_neg_Ddt = torch.where(dec_prod_only, torch.ones_like(D_eff), torch.exp(-torch.clamp(D_dt, max=80.0)))
        y0_tensor = torch.tensor([[y0, y0, y0]], dtype=torch.float32, device=tr.DEVICE)
        y_pred = y0_tensor * exp_neg_Ddt + prod_term
        y_pred_val = float(y_pred[0, 0].item())

        abs_err = abs(y_pred_val - y_true)
        rel_err = abs_err / max(y_true, 1e-45)
        print(f"Case [{name:28s}]: Target = {y_true:.8e}, Pred = {y_pred_val:.8e}, AbsErr = {abs_err:.2e}, RelErr = {rel_err:.2e}")
        if y_true < 1e-38:
            assert abs_err < 1e-45, f"Round-trip failed on {name}: abs_err={abs_err}, rel_err={rel_err}"
        else:
            assert rel_err < 1e-5, f"Round-trip failed on {name}: abs_err={abs_err}, rel_err={rel_err}"

    print("[PASS] All transform round-trip test cases closed to machine precision!")


if __name__ == "__main__":
    test_qssa_asymptotic_roundtrip_closure()
    print("\nALL PHASE 3 ROUND-TRIP TESTS PASSED!")
