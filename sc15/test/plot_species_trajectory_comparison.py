"""Plot detailed species concentration trajectory comparison across representative CFD closed-loop cases.
Compares:
1. Cantera Ground Truth (black solid)
2. Baseline Model v71 (blue dashed)
3. Retrained Model v72 (red solid)
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import cantera as ct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports work
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_cfd_closed_loop import BenchmarkEngine, make_fuel_air_mix

# 4 Key Representative Cases
SELECTED_CASES = [
    {
        "id": "case1_1100K_60atm_H2-40",
        "title": "Case 1: 1100 K, 60 atm, phi=1.0, 40% H2 (Mid-T High-P Engine)",
        "T0": 1100.0,
        "P": 60.0,
        "X": make_fuel_air_mix(1.0, 0.4),
        "tmax": 2.5e-3,
        "dt": 2e-6,
    },
    {
        "id": "case2_1200K_30atm_H2-20_EGR15",
        "title": "Case 2: 1200 K, 30 atm, phi=1.0, 20% H2, 15% EGR (Exhaust Gas Recirculation)",
        "T0": 1200.0,
        "P": 30.0,
        "X": make_fuel_air_mix(1.0, 0.2, egr=0.15),
        "tmax": 3.0e-3,
        "dt": 2e-6,
    },
    {
        "id": "case3_1400K_30atm_NH3-pure",
        "title": "Case 3: 1400 K, 30 atm, phi=1.0, Pure NH3 (Zero Carbon Fuel)",
        "T0": 1400.0,
        "P": 30.0,
        "X": make_fuel_air_mix(1.0, 0.0),
        "tmax": 2.5e-3,
        "dt": 2e-6,
    },
    {
        "id": "case4_1250K_10atm_phi0.5_H2-30",
        "title": "Case 4: 1250 K, 10 atm, phi=0.5, 30% H2 (Lean Burn Gas Turbine)",
        "T0": 1250.0,
        "P": 10.0,
        "X": make_fuel_air_mix(0.5, 0.3),
        "tmax": 2.5e-3,
        "dt": 2e-6,
    },
]

# Species and physical quantities layout for 4x4 subplots
PLOT_LAYOUT = [
    # Row 1: State & Fuels
    ("T", "Temperature [K]", False, 800, 3400),
    ("P", "Pressure [atm]", False, 0, 150),
    ("NH3", "Mass Frac NH3", True, 1e-6, 1.0),
    ("H2", "Mass Frac H2", True, 1e-6, 1.0),
    # Row 2: Oxidizer, Product, NOx
    ("O2", "Mass Frac O2", True, 1e-6, 1.0),
    ("H2O", "Mass Frac H2O", True, 1e-6, 1.0),
    ("NO", "Mass Frac NO", True, 1e-8, 1e-1),
    ("N2O", "Mass Frac N2O", True, 1e-9, 1e-2),
    # Row 3: Key Radicals
    ("OH", "Radical OH", True, 1e-12, 1e-1),
    ("H", "Radical H", True, 1e-12, 1e-2),
    ("O", "Radical O", True, 1e-12, 1e-2),
    ("HO2", "Radical HO2", True, 1e-12, 1e-3),
    # Row 4: Intermediates
    ("NH2", "Intermediate NH2", True, 1e-12, 1e-2),
    ("NH", "Intermediate NH", True, 1e-12, 1e-3),
    ("HNO", "Intermediate HNO", True, 1e-12, 1e-3),
    ("H2O2", "Intermediate H2O2", True, 1e-12, 1e-3),
]


def simulate_case_full_trajectory(engine, case, projection_mode="kl"):
    """Run closed-loop UV simulation and record full (t, T, P, Y) history."""
    gas = engine.gas
    T0, Patm, X, tmax, dt = case["T0"], case["P"], case["X"], case["tmax"], case["dt"]
    n_steps = int(round(tmax / dt))
    
    # 1. Cantera ground truth
    gas.TPX = T0, Patm * ct.one_atm, X
    Y0 = gas.Y.copy()
    rho0 = gas.density
    u0 = gas.int_energy_mass
    
    reactor = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([reactor])
    
    ref_t = [0.0]
    ref_T = [T0]
    ref_P = [Patm]
    ref_Y = [Y0.copy()]
    
    for k in range(n_steps):
        sim.advance((k + 1) * dt)
        ref_t.append(sim.time)
        ref_T.append(gas.T)
        ref_P.append(gas.P / ct.one_atm)
        ref_Y.append(gas.Y.copy())
        
    # 2. Autonomous NN rollout
    Y = Y0.copy()
    T = T0
    P = Patm
    
    nn_t = [0.0]
    nn_T = [T0]
    nn_P = [Patm]
    nn_Y = [Y0.copy()]
    
    for k in range(n_steps):
        Y = engine.step_model(T, P, Y, dt, projection_mode=projection_mode)
        try:
            gas.UVY = u0, 1.0 / rho0, Y
            T = gas.T
            P = gas.P / ct.one_atm
        except Exception:
            # If closure fails, fill remaining with nan
            for step_left in range(k + 1, n_steps + 1):
                nn_t.append(step_left * dt)
                nn_T.append(np.nan)
                nn_P.append(np.nan)
                nn_Y.append(np.full_like(Y0, np.nan))
            break
            
        nn_t.append((k + 1) * dt)
        nn_T.append(T)
        nn_P.append(P)
        nn_Y.append(Y.copy())
        
    return {
        "names": engine.names,
        "ref_t": np.array(ref_t) * 1e3,  # convert to ms
        "ref_T": np.array(ref_T),
        "ref_P": np.array(ref_P),
        "ref_Y": np.array(ref_Y),
        "nn_t": np.array(nn_t) * 1e3,  # convert to ms
        "nn_T": np.array(nn_T),
        "nn_P": np.array(nn_P),
        "nn_Y": np.array(nn_Y),
    }


def main():
    print("=" * 75)
    print("Generating Species Trajectory Validation Plots (Baseline vs Retrained)")
    print("=" * 75)
    
    output_dir = PROJECT_DIR / "plots_v72_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_dir = Path("C:/Users/123/.gemini/antigravity/brain/5d2e2966-2764-4647-86e4-4e9fcb576126")
    
    print("\n[1/3] Loading Baseline Engine (v71)...")
    base_engine = BenchmarkEngine(
        root=str(PROJECT_DIR / "output1/phase68_runs/v71_qssa_asymptotic_exact"),
        ckpt_name="best_oracle_h128_epoch27.pth"
    )
    
    print("[2/3] Loading Retrained Engine (v72_cfd_optimized)...")
    v72_engine = BenchmarkEngine(
        root=str(PROJECT_DIR / "output1/phase68_runs/v72_cfd_optimized"),
        ckpt_name="best_oracle_h128_epoch11.pth"
    )
    
    print("\n[3/3] Simulating and plotting cases...")
    for idx, case in enumerate(SELECTED_CASES, start=1):
        print(f"  -> Simulating {case['id']}...")
        
        # Simulate baseline
        base_res = simulate_case_full_trajectory(base_engine, case, projection_mode="kl")
        # Simulate retrained
        v72_res = simulate_case_full_trajectory(v72_engine, case, projection_mode="kl")
        
        # Plot figure
        fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True)
        fig.suptitle(f"{case['title']} [dt=2 us UV Autonomous Closed Loop]", fontsize=14, fontweight="bold", y=0.99)
        
        t_ref = base_res["ref_t"]
        t_base = base_res["nn_t"]
        t_v72 = v72_res["nn_t"]
        
        names = base_res["names"]
        
        for p_idx, (var_name, var_label, is_log, ymin, ymax) in enumerate(PLOT_LAYOUT):
            r = p_idx // 4
            c = p_idx % 4
            ax = axes[r, c]
            
            if var_name == "T":
                y_ref = base_res["ref_T"]
                y_base = base_res["nn_T"]
                y_v72 = v72_res["nn_T"]
            elif var_name == "P":
                y_ref = base_res["ref_P"]
                y_base = base_res["nn_P"]
                y_v72 = v72_res["nn_P"]
            else:
                sp_idx = names.index(var_name)
                y_ref = base_res["ref_Y"][:, sp_idx]
                y_base = base_res["nn_Y"][:, sp_idx]
                y_v72 = v72_res["nn_Y"][:, sp_idx]
                
            if is_log:
                y_ref = np.clip(y_ref, 1e-30, None)
                y_base = np.clip(y_base, 1e-30, None)
                y_v72 = np.clip(y_v72, 1e-30, None)
                ax.set_yscale("log")
                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)
            else:
                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)
                    
            # Curves
            ax.plot(t_ref, y_ref, color="black", lw=2.0, label="Cantera (Ref)" if p_idx == 0 else "")
            ax.plot(t_base, y_base, color="tab:blue", ls="--", lw=1.5, alpha=0.85, label="Baseline (v71)" if p_idx == 0 else "")
            ax.plot(t_v72, y_v72, color="tab:red", ls="-", lw=1.6, label="Retrained (v72)" if p_idx == 0 else "")
            
            ax.set_ylabel(var_label, fontsize=9)
            ax.grid(True, which="both", alpha=0.25)
            
            if r == 3:
                ax.set_xlabel("Time [ms]", fontsize=10)
                
        # Legend on top-left plot
        axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.85)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = output_dir / f"{case['id']}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"     Saved: {save_path}")
        
        # Also copy to artifact directory if it exists
        if artifact_dir.exists():
            art_save_path = artifact_dir / f"{case['id']}.png"
            shutil.copy2(save_path, art_save_path)
            print(f"     Artifact copied: {art_save_path}")
            
    print("\nAll 4 case trajectory comparison plots generated successfully!")


if __name__ == "__main__":
    main()
