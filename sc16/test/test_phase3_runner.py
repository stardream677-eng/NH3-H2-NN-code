"""Phase 3 Unit Test: Transient Chemistry Database Generator (runner_sc_v21_qssa.py)

Verifies:
1. Pressure unit conversion and range sanity assertions (preventing accidental 101325 atm errors).
2. Bounded monotonic root-finding on extreme inputs (solving GPT counterexample with machine precision).
3. Exact QSSA algebraic parameter inversion across diverse mixtures with Cantera's Y_next (< 1e-12 abs err).
4. Equilibrium stopping guard during slow induction under the EXACT default dt distribution ([1e-8, 1e-4] s).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cantera as ct
import runner_sc_v21_qssa as runner_mod


def test_pressure_validation_and_conversion():
    print("\n--- Testing Runner: Pressure Validation & Unit Conversions ---")
    
    # 1. Valid atm conversion
    p_pa = runner_mod._pressure_to_pa(1.0, "atm")
    assert abs(p_pa - 101325.0) < 1.0, f"Expected 101325 Pa, got {p_pa}"

    p_pa_40 = runner_mod._pressure_to_pa(40.0, "atm")
    assert abs(p_pa_40 - 40.0 * ct.one_atm) < 1.0

    # 2. Valid Pa conversion
    p_pa = runner_mod._pressure_to_pa(101325.0, "pa")
    assert abs(p_pa - 101325.0) < 1e-6

    # 3. Auto detection
    assert abs(runner_mod._pressure_to_pa(1.0, "auto") - ct.one_atm) < 1.0
    assert abs(runner_mod._pressure_to_pa(101325.0, "auto") - 101325.0) < 1e-6

    # 4. Out of range sanity checks
    # Old bug: 101325 Pa mistakenly passed to atm
    caught_large_atm = False
    try:
        runner_mod._pressure_to_pa(101325.0, "atm")
    except ValueError as exc:
        caught_large_atm = True
        print(f"Correctly caught out-of-range atm error: {exc}")
    assert caught_large_atm, "Failed to catch 101325 atm sanity violation!"

    # Atm mistakenly passed to Pa
    caught_small_pa = False
    try:
        runner_mod._pressure_to_pa(2.0, "pa")
    except ValueError as exc:
        caught_small_pa = True
        print(f"Correctly caught out-of-range pa error: {exc}")
    assert caught_small_pa, "Failed to catch 2.0 Pa sanity violation!"

    print("[PASS] Pressure validation & conversion checks passed.")


def test_gpt_counterexample_closure():
    print("\n--- Testing Runner: Bounded Root-Finding on Extreme Input ---")
    # GPT noted that an extreme initial guess on an unbracketed Newton method produced:
    # target Y_next = 0.1182376, reconstructed = 2.50e-35 (error = 0.1182376)
    y0 = np.array([0.20])
    y_next = np.array([0.1182376])
    dt = 1e-4
    p_raw = np.array([50.0])
    d_init = np.array([1e8]) # extreme destruction initial guess

    p_eff, d_eff = runner_mod._solve_exact_qssa_params(
        Y0=y0,
        Y_next=y_next,
        dt=dt,
        P_raw=p_raw,
        D_raw_init=d_init,
    )

    z = d_eff[0] * dt
    phi1 = (1.0 - np.exp(-z)) / z
    y_rec = y0[0] * np.exp(-z) + p_eff[0] * dt * phi1
    abs_err = abs(y_rec - y_next[0])
    print(f"GPT Extreme Input Test: Target = {y_next[0]}, Reconstructed = {y_rec:.10f}, Abs Err = {abs_err:.3e}")
    assert abs_err < 1e-12, f"Failed bounded root solve on extreme input! Err: {abs_err}"
    print("[PASS] Bounded root solver resolved extreme counterexample to machine precision (< 1e-12).")


def test_trace_species_relative_closure():
    print("\n--- Testing Runner: Trace Species Relative Closure Across 32 Orders of Mag ---")
    # Test cases identified by GPT:
    test_cases = [
        # (y0, y_tgt, dt, p_raw, d_raw)
        (5.366e-06, 1.841e-22, 5.573e-08, 9.519e-20, 31.61), # 16 orders of decay
        (1.0e-20, 5.0e-21, 1.0e-04, 0.0, 100.0),             # trace radical decay
        (1.0e-25, 2.0e-25, 1.0e-05, 1.5e-20, 10.0),          # trace production
        (1.0e-32, 1.0e-32, 1.0e-04, 0.0, 0.0),               # deep unchanged trace
        (1.0e-20, 1.0e-35, 1.0e-04, 1.0e-25, 10.0),          # 15 orders of decay down to 1e-35 (GPT case)
        (1.0e-25, 1.0e-38, 1.0e-05, 0.0, 50.0),              # pure destruction down to 1e-38
    ]

    for y0, y_tgt, dt, p_raw, d_raw in test_cases:
        p_eff, d_eff = runner_mod._solve_exact_qssa_params(
            Y0=np.array([y0]),
            Y_next=np.array([y_tgt]),
            dt=dt,
            P_raw=np.array([p_raw]),
            D_raw_init=np.array([d_raw]),
        )
        z = d_eff[0] * dt
        safe_z = max(z, 1e-12)
        phi1 = 1.0 - 0.5 * z + (z**2) / 6.0 if z < 1e-7 else (1.0 - np.exp(-min(z, 700.0))) / safe_z
        exp_z = 0.0 if z > 700.0 else np.exp(-z)
        y_rec = y0 * exp_z + p_eff[0] * dt * phi1
        abs_err = abs(y_rec - y_tgt)
        rel_err = abs_err / y_tgt
        print(f"Trace target: {y_tgt:.3e}, Rec: {y_rec:.3e}, AbsErr: {abs_err:.3e}, RelErr: {rel_err:.3e}")
        assert rel_err < 1e-8, f"Relative error too high: {rel_err} for target {y_tgt}"

    # Random stress test: 2,000 cases across 40 orders of magnitude
    np.random.seed(42)
    max_rel = 0.0
    for _ in range(2000):
        y0 = 10.0 ** np.random.uniform(-40, 0)
        y_tgt = 10.0 ** np.random.uniform(-40, 0)
        dt = 10.0 ** np.random.uniform(-8, -4)
        p_raw = 10.0 ** np.random.uniform(-25, 4)
        d_raw = 10.0 ** np.random.uniform(-5, 6)

        p_eff, d_eff = runner_mod._solve_exact_qssa_params(
            Y0=np.array([y0]),
            Y_next=np.array([y_tgt]),
            dt=dt,
            P_raw=np.array([p_raw]),
            D_raw_init=np.array([d_raw]),
        )
        z = d_eff[0] * dt
        safe_z = max(z, 1e-12)
        phi1 = 1.0 - 0.5 * z + (z**2) / 6.0 if z < 1e-7 else (1.0 - np.exp(-min(z, 700.0))) / safe_z
        exp_z = 0.0 if z > 700.0 else np.exp(-z)
        y_rec = y0 * exp_z + p_eff[0] * dt * phi1
        rel_err = abs(y_rec - y_tgt) / y_tgt
        max_rel = max(max_rel, rel_err)

    print(f"2000 Random Trace Stress Cases Max RelErr: {max_rel:.3e}")
    assert max_rel < 1e-8, f"Max relative error exceeded 1e-8: {max_rel}"
    print("[PASS] Trace species relative closure verified to machine precision (< 1e-8).")


def test_exact_qssa_algebraic_closure():
    print("\n--- Testing Runner: Exact QSSA Algebraic Closure with Cantera ---")
    mech_path = "NUIG2024.yaml"
    gas = ct.Solution(mech_path)
    
    test_conditions = [
        (1000.0, 10.0 * ct.one_atm, "NH3:0.10, H2:0.05, O2:0.10, N2:0.75"),
        (1400.0, 1.0 * ct.one_atm, "NH3:0.08, H2:0.12, O2:0.12, N2:0.68"),
        (850.0, 50.0 * ct.one_atm, "NH3:0.15, H2:0.02, O2:0.1225, N2:0.7075"),
    ]

    for T_val, P_val, comp_str in test_conditions:
        gas.TPX = T_val, P_val, comp_str
        r = ct.IdealGasReactor(gas)
        sim = ct.ReactorNet([r])
        W_k = gas.molecular_weights

        for step in range(5):
            t_curr = sim.time
            y0 = gas.Y.copy()
            rho0 = gas.density
            P0 = (gas.creation_rates * W_k) / rho0
            D0_prime = (gas.destruction_rates * W_k) / rho0
            d0 = D0_prime / np.maximum(y0, runner_mod.QSSA_STATE_FLOOR)

            dt = 2e-5
            sim.advance(t_curr + dt)

            y_next = gas.Y.copy()
            rho1 = gas.density
            P1 = (gas.creation_rates * W_k) / rho1
            D1_prime = (gas.destruction_rates * W_k) / rho1
            d1 = D1_prime / np.maximum(y_next, runner_mod.QSSA_STATE_FLOOR)

            P_candidate = 0.5 * (P0 + P1)
            D_candidate = 0.5 * (d0 + d1)

            P_exact, D_exact = runner_mod._solve_exact_qssa_params(
                Y0=y0,
                Y_next=y_next,
                dt=dt,
                P_raw=P_candidate,
                D_raw_init=D_candidate,
            )

            # Reconstruct next state using the analytical exponential integrator
            z = D_exact * dt
            safe_z = np.maximum(z, 1e-12)
            phi1 = np.where(z < 1e-7, 1.0 - 0.5 * z + (z**2) / 6.0, (1.0 - np.exp(-z)) / safe_z)
            y_reconstructed = y0 * np.exp(-z) + P_exact * dt * phi1

            abs_err = np.max(np.abs(y_reconstructed - y_next))
            delta_y = np.abs(y_next - y0)
            active = delta_y > 1e-7
            rel_err = np.max(np.abs(y_reconstructed[active] - y_next[active]) / delta_y[active]) if np.any(active) else 0.0

            assert abs_err < 1e-12, f"Reconstruction error too large: {abs_err}"
            assert rel_err < 1e-6, f"Relative reconstruction error too large: {rel_err}"

    print("[PASS] Exact QSSA algebraic parameter closure verified (< 1e-12 abs err).")


def test_slow_induction_ignition_guard():
    print("\n--- Testing Runner: Slow Induction Ignition Guard Under Default dt ---")
    mech_path = "NUIG2024.yaml"
    
    # 850 K, 80 atm case with exact stoichiometric air
    sample = {
        "sample_id": "test_slow_induction",
        "T": 850.0,
        "P": 80.0,
        "phi": 1.0,
        "phi_actual": 1.0,
        "X_NH3": 0.10,
        "X_H2": 0.05,
        "X_O2": 0.10,
        "X_N2": 0.75,
    }

    # Test under EXACT default production dt distribution: [1e-8, 1e-4] s
    cfg = runner_mod.Config(
        dt_min=1e-8,                  # EXACT default production setting
        dt_max=1e-4,                  # EXACT default production setting
        max_time=0.20,                # 200 ms, enough to cover 95 ms ignition
        max_unignited_time=0.30,
        max_steps_per_trajectory=30000, # Raised from 4000 to prevent step truncation
        equilibrium_check_start=2e-3, # 2 ms
        min_ignition_temp_rise=200.0, # Must rise >= 200 K before checking equilibrium
        pressure_unit="atm",
    )

    df = runner_mod._simulate_trajectory(
        mech_path=mech_path,
        mech_name="nuig_2024",
        sample=sample,
        cfg=cfg,
    )

    assert not df.empty, "Simulation returned empty DataFrame"
    final_time = df["time"].iloc[-1]
    final_T = df["T_next"].iloc[-1]
    n_steps = len(df)
    term_reason = df["termination_reason"].iloc[-1]

    print(f"Total steps: {n_steps}")
    print(f"Final time: {final_time * 1e3:.2f} ms")
    print(f"Final T: {final_T:.1f} K (initial: 850.0 K)")
    print(f"Termination reason: {term_reason}")

    # Verify that it DID NOT exit at 2 ms or 42 ms (step 4000)
    assert final_time > 0.08, f"Simulation exited too early: {final_time * 1e3:.2f} ms (expected > 80 ms)"
    # Verify that ignition actually occurred
    assert final_T > 1050.0, f"Expected ignition temperature > 1050 K, got {final_T} K"
    # Verify termination reason is equilibrium (after ignition) or max_time
    assert term_reason in {"equilibrium", "max_time"}, f"Unexpected termination reason: {term_reason}"

    print("[PASS] Production default dt distribution simulated through slow induction into ignition and equilibrium!")


if __name__ == "__main__":
    test_pressure_validation_and_conversion()
    test_gpt_counterexample_closure()
    test_trace_species_relative_closure()
    test_exact_qssa_algebraic_closure()
    test_slow_induction_ignition_guard()
    print("\nALL PHASE 3 RUNNER TESTS PASSED!")
