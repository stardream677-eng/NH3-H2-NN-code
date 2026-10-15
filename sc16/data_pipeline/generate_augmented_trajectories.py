"""Generate augmented diverse Cantera trajectories for deep chemical coverage.

Atomic write to part-000003.parquet with condition_group_id tagging and project-relative paths.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import cantera as ct
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import runner_sc_v21_qssa as r21

DEFAULT_TARGET_DIR = PROJECT_ROOT / "trajectories_nuig_2024"


def make_fuel_air_mix(phi: float, xh2: float, egr: float = 0.0) -> dict[str, float]:
    f_nh3, f_h2 = 1.0 - xh2, xh2
    o2 = (0.75 * f_nh3 + 0.5 * f_h2) / phi
    X = {"NH3": f_nh3, "H2": f_h2, "O2": o2, "N2": 3.7619 * o2}
    if egr > 0:
        tot = sum(X.values())
        X["N2"] += egr * tot
    return X


def generate_augmented(
    target_dir: str | Path | None = None,
    mech_path: str | Path | None = None,
    n_cases: int = 40,
) -> pd.DataFrame:
    print("Generating augmented diverse Cantera trajectories for deep chemical coverage...")
    if target_dir is None:
        target_dir = DEFAULT_TARGET_DIR
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if mech_path is None:
        mech_path = PROJECT_ROOT / "NUIG2024.yaml"
        if not mech_path.exists():
            mech_path = SCRIPT_DIR / "NUIG2024.yaml"
    gas = ct.Solution(str(mech_path))
    names = list(gas.species_names)
    W_k = gas.molecular_weights

    temps = [950, 1050, 1150, 1250, 1350, 1450, 1600]
    pressures = [1, 5, 15, 30, 50]
    phis = [0.6, 0.8, 1.0, 1.2, 1.4]
    h2_fractions = [0.05, 0.20, 0.40, 0.60]

    rng_cond = np.random.default_rng(20261014)
    conditions = []
    for i in range(n_cases):
        T0 = float(rng_cond.choice(temps) + rng_cond.uniform(-30, 30))
        P_atm = float(rng_cond.choice(pressures) * rng_cond.uniform(0.85, 1.15))
        phi = float(rng_cond.choice(phis) * rng_cond.uniform(0.95, 1.05))
        xh2 = float(rng_cond.choice(h2_fractions))
        egr = float(rng_cond.choice([0.0, 0.05, 0.10, 0.15]))
        tmax = 0.015 * (1200.0 / T0) ** 2 * (20.0 / max(P_atm, 2.0)) ** 0.5
        tmax = float(np.clip(tmax, 1e-3, 20e-3))
        conditions.append({
            "name": f"Aug_{i:02d}_{T0:.0f}K_{P_atm:.1f}atm_phi{phi:.2f}_H2{xh2:.2f}_egr{egr:.2f}",
            "T0": T0,
            "P": P_atm,
            "phi": phi,
            "xh2": xh2,
            "egr": egr,
            "tmax": tmax,
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
                "condition_group_id": f"aug_cond_{case_idx}",
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
            print(f"  [{case_idx+1}/{len(conditions)}] {c['name']}: {len(case_records)} steps (T_final = {T_end:.0f} K)")

    df = pd.DataFrame(all_rows)
    out_path = target_dir / "part-000003.parquet"
    tmp_path = out_path.with_suffix(".tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    print(f"Saved {len(df)} rows ({len(conditions)} augmented trajectories) to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented trajectories")
    parser.add_argument("--target_dir", type=str, default=str(DEFAULT_TARGET_DIR))
    parser.add_argument("--cases", type=int, default=40)
    args = parser.parse_args()

    t0 = time.time()
    generate_augmented(target_dir=args.target_dir, n_cases=args.cases)
    print(f"Augmented generation completed in {time.time() - t0:.2f} s")
