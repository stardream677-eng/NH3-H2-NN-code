"""Dataset Builder for V72 CFD-Optimized Surrogate Training.

1. Converts mini_ds.pkl (30 trajectories, 4800 steps) into standard Parquet format.
2. Generates Cantera ground-truth trajectories for all 24 benchmark combustion conditions
   using the exact QSSA asymptotic solver from runner_sc_v21_qssa.py.
3. Saves to trajectories_nuig_2024/part-000001.parquet and part-000002.parquet.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path

# Add script directory to sys.path to import helper from runner_sc_v21_qssa
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import importlib.util
bench_spec = importlib.util.spec_from_file_location(
    "bench", str(SCRIPT_DIR / "test" / "benchmark_cfd_closed_loop.py")
)
bench_mod = importlib.util.module_from_spec(bench_spec)
bench_spec.loader.exec_module(bench_mod)
BENCHMARK_CASES = bench_mod.BENCHMARK_CASES
import runner_sc_v21_qssa as r21

TARGET_DIR = Path(r"C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc14\07\01\database_Transient_V21_QSSA\trajectories_nuig_2024")


def convert_mini_ds():
    pkl_path = SCRIPT_DIR / "mini_ds.pkl"
    if not pkl_path.exists():
        print(f"mini_ds.pkl not found at {pkl_path}")
        return None
    
    print(f"Loading mini_ds.pkl from {pkl_path}...")
    ds = joblib.load(pkl_path)
    names = ds["names"]
    rows = ds["rows"]
    
    records = []
    # Track cumulative time per sid
    time_per_sid = {}
    
    # Pre-count intervals per sid
    intervals_per_sid = {}
    for r in rows:
        sid = r["sid"] + 10
        intervals_per_sid[sid] = intervals_per_sid.get(sid, 0) + 1
        
    for r in rows:
        sid = r["sid"] + 10
        dt = float(r["dt"])
        t_curr = time_per_sid.get(sid, 0.0)
        time_per_sid[sid] = t_curr + dt
        
        T = float(r["T"])
        P = float(r["P"])
        Tn = float(r["Tn"])
        dTdt = (Tn - T) / dt if dt > 0 else 0.0
        
        rec = {
            "sample_id": sid,
            "mechanism_name": "nuig_2024",
            "phi": 1.0,
            "phi_actual": 1.0,
            "initial_radical_total": 0.0,
            "time": t_curr,
            "dt": dt,
            "T": T,
            "P": P,
            "T_next": Tn,
            "P_next": P,
            "dTdt_avg": dTdt,
            "termination_reason": "normal",
            "n_intervals_in_trajectory": intervals_per_sid[sid],
        }
        
        Y = r["Y"]
        Yn = r["Yn"]
        Pq = r["Pq"]
        Dq = r["Dq"]
        
        for i, sp in enumerate(names):
            y_i = float(Y[i])
            yn_i = float(Yn[i])
            rec[f"Y_{sp}"] = y_i
            rec[f"dYdt_{sp}"] = (yn_i - y_i) / dt if dt > 0 else 0.0
            rec[f"Y_next_{sp}"] = yn_i
            rec[f"P_avg_{sp}"] = float(Pq[i])
            rec[f"D_coef_avg_{sp}"] = float(Dq[i])
            
        records.append(rec)
        
    df = pd.DataFrame(records)
    out_path = TARGET_DIR / "part-000001.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows ({len(intervals_per_sid)} trajectories) to {out_path}")
    return df


def generate_benchmark_trajectories():
    print("\nGenerating Cantera trajectories for 24 benchmark conditions...")
    mech_path = SCRIPT_DIR / "NUIG2024.yaml"
    if not mech_path.exists():
        mech_path = SCRIPT_DIR.parent / "NUIG2024.yaml"
    gas = ct.Solution(str(mech_path))
    names = list(gas.species_names)
    W_k = gas.molecular_weights
    
    all_rows = []
    
    for case_idx, case in enumerate(BENCHMARK_CASES):
        sid = 100 + case_idx
        T0, Patm, X, tmax = case["T0"], case["P"], case["X"], case["tmax"]
        gas.TPX = T0, Patm * ct.one_atm, X
        
        reactor = ct.IdealGasReactor(gas)
        sim = ct.ReactorNet([reactor])
        
        n_steps = 160  # At least 160 steps (> 128 max_horizon)
        dt_fixed = tmax / n_steps
        
        # Jitter dt slightly for robust multi-scale coverage
        rng = np.random.default_rng(2026 + case_idx)
        
        t_curr = 0.0
        case_records = []
        
        for step in range(n_steps):
            dt = dt_fixed * rng.uniform(0.7, 1.3)
            
            T_start = float(reactor.T)
            P_start = float(reactor.thermo.P)
            Y_start = reactor.thermo.Y.copy()
            rho_start = reactor.thermo.density
            
            # Start QSSA rates
            C_start = gas.creation_rates * W_k / rho_start
            D_rate_start = gas.destruction_rates * W_k / rho_start
            D_coef_start = D_rate_start / np.maximum(Y_start, 1e-300)
            
            # Advance reactor
            sim.advance(t_curr + dt)
            
            T_end = float(reactor.T)
            P_end = float(reactor.thermo.P)
            Y_end = reactor.thermo.Y.copy()
            rho_end = reactor.thermo.density
            
            # End QSSA rates
            C_end = gas.creation_rates * W_k / rho_end
            D_rate_end = gas.destruction_rates * W_k / rho_end
            D_coef_end = D_rate_end / np.maximum(Y_end, 1e-300)
            
            P_candidate = (C_start + C_end) / 2.0
            D_candidate = (D_coef_start + D_coef_end) / 2.0
            
            # Solve exact QSSA
            P_eff, D_eff = r21._solve_exact_qssa_params(
                Y0=Y_start,
                Y_next=Y_end,
                dt=dt,
                P_raw=P_candidate,
                D_raw_init=D_candidate,
            )
            
            rec = {
                "sample_id": sid,
                "mechanism_name": "nuig_2024",
                "phi": 1.0,
                "phi_actual": 1.0,
                "initial_radical_total": 0.0,
                "time": t_curr,
                "dt": dt,
                "T": T_start,
                "P": P_start / ct.one_atm,
                "T_next": T_end,
                "P_next": P_end / ct.one_atm,
                "dTdt_avg": (T_end - T_start) / dt,
                "termination_reason": "normal",
                "n_intervals_in_trajectory": n_steps,
            }
            
            for i, sp in enumerate(names):
                rec[f"Y_{sp}"] = float(Y_start[i])
                rec[f"dYdt_{sp}"] = float((Y_end[i] - Y_start[i]) / dt)
                rec[f"Y_next_{sp}"] = float(Y_end[i])
                rec[f"P_avg_{sp}"] = float(P_eff[i])
                rec[f"D_coef_avg_{sp}"] = float(D_eff[i])
                
            case_records.append(rec)
            t_curr += dt
            
        all_rows.extend(case_records)
        print(f"  [{case_idx+1}/24] {case['name']}: {len(case_records)} steps (T_final = {T_end:.0f} K)")
        
    df = pd.DataFrame(all_rows)
    out_path = TARGET_DIR / "part-000002.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows (24 benchmark trajectories) to {out_path}")
    return df


if __name__ == "__main__":
    t0 = time.time()
    convert_mini_ds()
    generate_benchmark_trajectories()
    print(f"\nAll datasets generated in {time.time() - t0:.2f} s")
