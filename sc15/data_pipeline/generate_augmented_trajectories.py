import os
import sys
import time
import numpy as np
import pandas as pd
import cantera as ct
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import runner_sc_v21_qssa as r21

TARGET_DIR = Path(r"C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc14\07\01\database_Transient_V21_QSSA\trajectories_nuig_2024")

def make_fuel_air_mix(phi, xh2, egr=0.0):
    f_nh3, f_h2 = 1.0 - xh2, xh2
    o2 = (0.75 * f_nh3 + 0.5 * f_h2) / phi
    X = {"NH3": f_nh3, "H2": f_h2, "O2": o2, "N2": 3.7619 * o2}
    if egr > 0:
        tot = sum(X.values())
        X["N2"] += egr * tot
    return X

def generate_augmented():
    print("Generating augmented diverse Cantera trajectories for deep chemical coverage...")
    mech_path = SCRIPT_DIR / "NUIG2024.yaml"
    if not mech_path.exists():
        mech_path = SCRIPT_DIR.parent / "NUIG2024.yaml"
    gas = ct.Solution(str(mech_path))
    names = list(gas.species_names)
    W_k = gas.molecular_weights
    
    # 40 diverse operating points covering engine, gas turbine, and atmospheric conditions
    conditions = []
    temps = [950, 1050, 1150, 1250, 1350, 1450, 1600]
    pressures = [1, 5, 15, 30, 50]
    phis = [0.6, 0.8, 1.0, 1.2, 1.4]
    h2_fractions = [0.05, 0.20, 0.40, 0.60]
    
    rng_cond = np.random.default_rng(20261014)
    # Generate 40 Latin-hypercube-style varied conditions
    for i in range(40):
        T0 = float(rng_cond.choice(temps) + rng_cond.uniform(-30, 30))
        P_atm = float(rng_cond.choice(pressures) * rng_cond.uniform(0.85, 1.15))
        phi = float(rng_cond.choice(phis) * rng_cond.uniform(0.95, 1.05))
        xh2 = float(rng_cond.choice(h2_fractions))
        egr = float(rng_cond.choice([0.0, 0.05, 0.10, 0.15]))
        # Approximate tmax: higher T/P -> shorter tmax
        tmax = 0.015 * (1200.0 / T0)**2 * (20.0 / max(P_atm, 2.0))**0.5
        tmax = float(np.clip(tmax, 1e-3, 20e-3))
        conditions.append({
            "name": f"Aug_{i:02d}_{T0:.0f}K_{P_atm:.1f}atm_phi{phi:.2f}_H2{xh2:.2f}_egr{egr:.2f}",
            "T0": T0, "P": P_atm, "phi": phi, "xh2": xh2, "egr": egr, "tmax": tmax
        })
        
    all_rows = []
    n_steps = 160
    
    for case_idx, c in enumerate(conditions):
        sid = 200 + case_idx
        X = make_fuel_air_mix(c["phi"], c["xh2"], c["egr"])
        gas.TPX = c["T0"], c["P"] * ct.one_atm, X
        
        reactor = ct.IdealGasReactor(gas)
        sim = ct.ReactorNet([reactor])
        
        dt_fixed = c["tmax"] / n_steps
        rng = np.random.default_rng(3000 + case_idx)
        
        t_curr = 0.0
        case_records = []
        
        for step in range(n_steps):
            dt = dt_fixed * rng.uniform(0.7, 1.3)
            
            T_start = float(reactor.T)
            P_start = float(reactor.thermo.P)
            Y_start = reactor.thermo.Y.copy()
            rho_start = reactor.thermo.density
            
            C_start = gas.creation_rates * W_k / rho_start
            D_rate_start = gas.destruction_rates * W_k / rho_start
            D_coef_start = D_rate_start / np.maximum(Y_start, 1e-300)
            
            t_target = t_curr + dt
            sim.advance(t_target)
            
            T_end = float(reactor.T)
            P_end = float(reactor.thermo.P)
            Y_end = reactor.thermo.Y.copy()
            rho_end = reactor.thermo.density
            
            C_end = gas.creation_rates * W_k / rho_end
            D_rate_end = gas.destruction_rates * W_k / rho_end
            D_coef_end = D_rate_end / np.maximum(Y_end, 1e-300)
            
            P_candidate = (C_start + C_end) / 2.0
            D_candidate = (D_coef_start + D_coef_end) / 2.0
            
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
                "phi": c["phi"],
                "phi_actual": c["phi"],
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
        if (case_idx + 1) % 5 == 0 or case_idx == len(conditions) - 1:
            print(f"  [{case_idx+1}/40] {c['name']}: {len(case_records)} steps (T_final = {T_end:.0f} K)")
            
    df = pd.DataFrame(all_rows)
    out_path = TARGET_DIR / "part-000003.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows (40 augmented trajectories) to {out_path}")
    return df

if __name__ == "__main__":
    t0 = time.time()
    generate_augmented()
    print(f"Augmented generation completed in {time.time() - t0:.2f} s")
