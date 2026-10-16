"""LEGACY Cantera-assisted CFD benchmark (not a fully neural autonomous benchmark).

This historical exploratory script calls Cantera UVY/HPY to re-close the NN
temperature and uses separate NumPy projection routines. It therefore does
NOT measure independent production NASA7/EOS network rollout accuracy.
For verified H=128 T/P/Y evaluation use test/benchmark_autonomous_h128.py.

Original title: CFD-grade Closed-loop Autonomous Benchmark Suite for NH3/H2 Combustion Surrogate.

Evaluates neural network surrogate against Cantera ground truth across realistic
combustion conditions with:
  1. Autonomous thermodynamic closure:
     - Const-Volume ('UV'): u(T^{n+1}, Y^{n+1}) = u_0  (Internal combustion engine / compressible CFD)
     - Const-Pressure ('HP'): h(T^{n+1}, Y^{n+1}) = h_0 (Gas turbine / reactingFoam)
  2. Fixed CFD-like time steps (e.g. 1e-6 s, 5e-6 s)
  3. Pluggable projection methods:
     - 'kl': Multiplicative element-potential (KL divergence) projection (strictly positive, 0 wipes)
     - 'linear': Current baseline weighted-linear projection + clamp
     - 'none': Raw unconstrained network output
"""

import os
import sys
import glob
import time
import argparse
import importlib.util
import json
import numpy as np
import torch
import joblib
import cantera as ct

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load training module for classes and helpers
import train_sc_v68 as tr
import experiment_manifest as em
sys.modules['tr'] = tr
sys.modules['train_sc_v68'] = tr
sys.modules['__main__'].FeatureScaler = tr.FeatureScaler

DEFAULT_ROOT = "output1/phase68_runs/v71_qssa_asymptotic_exact"
DEFAULT_CKPT = "best_oracle_h128_epoch27.pth"


class BenchmarkEngine:
    def __init__(self, root=DEFAULT_ROOT, ckpt_name=DEFAULT_CKPT, device=tr.DEVICE):
        self.device = device
        self.root = root
        self.ckpt_name = ckpt_name
        
        # Load scalers and feature info (preferring desentinelized versions if available)
        fi_matches = glob.glob(os.path.join(root, "scalers_v68/feature_info_*desentinelized*.pkl")) or glob.glob(os.path.join(root, "scalers_v68/feature_info_*.pkl"))
        sx_matches = glob.glob(os.path.join(root, "scalers_v68/scaler_X_*desentinelized*.pkl")) or glob.glob(os.path.join(root, "scalers_v68/scaler_X_*.pkl"))
        sy_matches = glob.glob(os.path.join(root, "scalers_v68/scaler_Y_*desentinelized*.pkl")) or glob.glob(os.path.join(root, "scalers_v68/scaler_Y_*.pkl"))
        self.fi = joblib.load(fi_matches[0])
        self.sx = joblib.load(sx_matches[0])
        self.sy = joblib.load(sy_matches[0])
        
        # Cantera mechanism & species indices
        self.gas = ct.Solution("NUIG2024.yaml")
        self.names = self.gas.species_names
        self.inert = self.fi["inert_species"]
        self.macro = self.fi["macro_species"]
        self.micro = self.fi["micro_species"]
        self.react = self.macro + self.micro
        
        # Load model
        ckpt_path = os.path.join(root, "weights_v68", ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        arch = ck.get("training_config", {}).get("network_arch", "mlp")
        in_dim = len(self.fi["input_cols"])
        out_dim = len(self.fi["output_cols"])
        if arch == "dual_branch":
            from dual_branch_network import DualBranchKineticsNet
            slow_species = self.fi["macro_species"]
            fast_species = self.fi["micro_species"]
            active_elements = [e for e in ["N", "H", "O"] if any(self.gas.n_atoms(s, e) for s in slow_species)]
            A_slow_np = np.zeros((len(active_elements), len(slow_species)), dtype=np.float32)
            for j, s in enumerate(slow_species):
                mw = self.gas.molecular_weights[self.names.index(s)]
                for i, e in enumerate(active_elements):
                    A_slow_np[i, j] = self.gas.n_atoms(s, e) * self.gas.atomic_weight(e) / mw
            A_slow_tensor = torch.from_numpy(A_slow_np).to(self.device)

            self.model = DualBranchKineticsNet(
                input_dim=in_dim,
                slow_dim=len(slow_species),
                fast_dim=len(fast_species),
                hidden_dim=512,
                num_blocks=4,
                output_parameterization=self.fi.get("output_parameterization", "dual_branch"),
                use_null_space=False,
                element_matrix_slow=None,
            ).to(self.device)
            print(f"Loaded DualBranchKineticsNet ({in_dim} -> 512x4 -> slow={len(slow_species)}+fast={2*len(fast_species)}={out_dim})")
        elif arch == "resmlp":
            from dual_branch_network import CoupledKineticsResMLP
            self.model = CoupledKineticsResMLP(
                in_dim,
                out_dim,
                hidden_dim=512,
                num_blocks=4,
                output_parameterization=self.fi.get("output_parameterization", "qssa_asymptotic"),
            ).to(self.device)
            print(f"Loaded ResMLP architecture model ({in_dim} -> 512x4 -> {out_dim})")
        else:
            self.model = tr.CoupledKineticsNet(
                in_dim,
                out_dim,
                output_parameterization="qssa_asymptotic"
            ).to(self.device)
            print(f"Loaded baseline CoupledKineticsNet model ({in_dim} -> {out_dim})")
        if arch == "dual_branch":
            from dual_branch_network import load_dual_branch_state_dict
            load_dual_branch_state_dict(self.model, ck["model_state_dict"])
        else:
            self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        
        eff_decode_mode = self.fi.get("rate_decode_mode", "qssa_asymptotic")
        self.eff_decode_mode = eff_decode_mode
        self.tools = tr.CoarseOperatorTools(
            self.fi, self.sx, self.sy,
            rate_decode_mode=eff_decode_mode,
            rate_z_limit=10.0,
            device=self.device,
        )
        
        self.ridx = np.array([self.names.index(s) for s in self.react])
        self.iidx = [self.names.index(s) for s in self.inert]
        
        # Build element matrix A for reactive species
        elements = ['N', 'H', 'O', 'Ar', 'He']
        self.elem_list = [e for e in elements if any(self.gas.n_atoms(s, e) for s in self.react)]
        A_np = np.zeros((len(self.elem_list), len(self.react)), dtype=np.float64)
        for j, s in enumerate(self.react):
            idx = self.names.index(s)
            mw = self.gas.molecular_weights[idx]
            for i, e in enumerate(self.elem_list):
                A_np[i, j] = self.gas.n_atoms(s, e) * self.gas.atomic_weight(e) / mw
        self.A_np = A_np
        self.A_tensor = torch.tensor(A_np, dtype=torch.float64, device=self.device)

    def build_X(self, T, P_atm, Y, dt):
        row = [T, P_atm] + [Y[self.names.index(s)] for s in self.inert] + \
              [Y[self.names.index(s)] for s in self.macro] + \
              [np.log10(max(Y[self.names.index(s)], 1e-40)) for s in self.micro] + \
              [np.log10(dt)]
        arr = np.array(row, dtype=np.float64)[None, :]
        return torch.tensor(self.sx.transform(arr).astype(np.float32), device=self.device)

    def project_kl(self, Y_react_raw, target_elements, max_iters=25, tol=1e-12):
        """Multiplicative KL (Element-potential) projection in NumPy for zero wipeout."""
        A = self.A_np
        b = target_elements
        # Strict lower guard to ensure positivity without destroying relative profile
        p0 = np.maximum(Y_react_raw, 1e-40)
        lam = np.zeros(len(b), dtype=np.float64)
        p = p0.copy()
        reg = 1e-15 * np.eye(len(b))
        
        for it in range(max_iters):
            F = A @ p - b
            if np.abs(F).max() < tol:
                break
            # Jacobian J = A @ diag(p) @ A.T
            J = (A * p) @ A.T
            try:
                dlam = -np.linalg.solve(J + reg, F)
            except np.linalg.LinAlgError:
                dlam = -np.linalg.pinv(J + reg) @ F
            # Damped step if large
            step_norm = np.abs(dlam).max()
            if step_norm > 10.0:
                dlam *= (10.0 / step_norm)
            lam += dlam
            p = p0 * np.exp(np.clip(A.T @ lam, -40.0, 40.0))
            
        p = np.clip(p, 0.0, None)
        tot = p.sum()
        if tot > 0:
            target_sum = b.sum()
            p = p * (target_sum / tot)
        return p

    def step_model(self, T, P_atm, Y, dt, projection_mode="kl"):
        """Single-step model advance with selectable projection mode."""
        with torch.no_grad():
            Xs = self.build_X(T, P_atm, Y, dt)
            out = self.model(Xs)
            ps = self.tools.predict_next_state(
                Xs, out, safe_for_rollout=True, state_update_mode=self.eff_decode_mode
            )
            
            if projection_mode == "linear":
                # Baseline: weighted linear projection with clamp
                Yn = ps["Y_next_constrained_full"][0].double().cpu().numpy()
            elif projection_mode == "none":
                # Raw unconstrained
                Y_macro = ps["Y_macro_next_raw"][0].double().cpu().numpy()
                Y_micro = ps["Y_micro_next_raw"][0].double().cpu().numpy()
                Yn = np.concatenate([Y_macro, Y_micro])
            elif projection_mode == "kl":
                # Multiplicative KL projection
                Y_macro = ps["Y_macro_next_raw"][0].double().cpu().numpy()
                Y_micro = ps["Y_micro_next_raw"][0].double().cpu().numpy()
                Y_raw_react = np.concatenate([Y_macro, Y_micro])
                
                # Target elements from previous state
                target_elem = self.A_np @ Y[self.ridx]
                Yn = self.project_kl(Y_raw_react, target_elem)
            else:
                raise ValueError(f"Unknown projection mode: {projection_mode}")
                
        full = Y.copy()
        full[self.ridx] = Yn
        full = np.clip(full, 0.0, None)
        s = full.sum()
        if s > 0:
            full /= s
        return full

    @staticmethod
    def calc_idt(times, temps, T0):
        """Calculates ignition delay time via 50% temperature rise."""
        times = np.asarray(times)
        temps = np.asarray(temps)
        dT_max = temps.max() - T0
        if dT_max < 200.0:
            return np.nan
        thr = T0 + 0.5 * dT_max
        idx = np.argmax(temps >= thr)
        return times[idx]


def make_fuel_air_mix(phi, xh2, egr=0.0):
    f_nh3, f_h2 = 1.0 - xh2, xh2
    o2 = (0.75 * f_nh3 + 0.5 * f_h2) / phi
    X = {"NH3": f_nh3, "H2": f_h2, "O2": o2, "N2": 3.7619 * o2}
    if egr > 0:
        tot = sum(X.values())
        X["N2"] += egr * tot
    return X


# Standardized 24 Benchmark Cases covering the key NH3/H2 operational domain
BENCHMARK_CASES = [
    # Core Engine / High Pressure (Cases 1-6)
    {"name": "1200K_40atm_phi1.0_H2-20", "T0": 1200, "P": 40, "X": make_fuel_air_mix(1.0, 0.2), "tmax": 4e-3},
    {"name": "1100K_60atm_phi1.0_H2-40", "T0": 1100, "P": 60, "X": make_fuel_air_mix(1.0, 0.4), "tmax": 4e-3},
    {"name": "1000K_80atm_phi0.8_H2-30", "T0": 1000, "P": 80, "X": make_fuel_air_mix(0.8, 0.3), "tmax": 8e-3},
    {"name": "1300K_20atm_phi0.6_H2-10", "T0": 1300, "P": 20, "X": make_fuel_air_mix(0.6, 0.1), "tmax": 4e-3},
    {"name": "1400K_10atm_phi1.4_H2-20", "T0": 1400, "P": 10, "X": make_fuel_air_mix(1.4, 0.2), "tmax": 3e-3},
    {"name": "1600K_5atm_phi1.0_NH3-pure", "T0": 1600, "P": 5, "X": make_fuel_air_mix(1.0, 0.0), "tmax": 2e-3},

    # Low-Temperature Compression Limit (Cases 7-10)
    {"name": "950K_40atm_phi1.0_H2-40", "T0": 950, "P": 40, "X": make_fuel_air_mix(1.0, 0.4), "tmax": 10e-3},
    {"name": "900K_60atm_phi1.0_H2-50", "T0": 900, "P": 60, "X": make_fuel_air_mix(1.0, 0.5), "tmax": 12e-3},
    {"name": "1000K_40atm_phi1.0_H2-20", "T0": 1000, "P": 40, "X": make_fuel_air_mix(1.0, 0.2), "tmax": 10e-3},
    {"name": "1050K_30atm_phi0.8_H2-30", "T0": 1050, "P": 30, "X": make_fuel_air_mix(0.8, 0.3), "tmax": 8e-3},

    # Lean / Rich Boundaries (Cases 11-14)
    {"name": "1100K_20atm_phi1.2_H2-20", "T0": 1100, "P": 20, "X": make_fuel_air_mix(1.2, 0.2), "tmax": 6e-3},
    {"name": "1150K_40atm_phi1.0_H2-10", "T0": 1150, "P": 40, "X": make_fuel_air_mix(1.0, 0.1), "tmax": 6e-3},
    {"name": "1250K_10atm_phi0.5_H2-30", "T0": 1250, "P": 10, "X": make_fuel_air_mix(0.5, 0.3), "tmax": 5e-3},
    {"name": "1350K_20atm_phi1.5_H2-20", "T0": 1350, "P": 20, "X": make_fuel_air_mix(1.5, 0.2), "tmax": 4e-3},

    # Gas Turbine & Atmospheric Burner (Cases 15-18)
    {"name": "1200K_1atm_phi1.0_H2-40", "T0": 1200, "P": 1, "X": make_fuel_air_mix(1.0, 0.4), "tmax": 8e-3},
    {"name": "1300K_5atm_phi1.0_H2-20", "T0": 1300, "P": 5, "X": make_fuel_air_mix(1.0, 0.2), "tmax": 5e-3},
    {"name": "1400K_30atm_phi1.0_NH3-pure", "T0": 1400, "P": 30, "X": make_fuel_air_mix(1.0, 0.0), "tmax": 4e-3},
    {"name": "1450K_1atm_phi1.0_NH3-pure", "T0": 1450, "P": 1, "X": make_fuel_air_mix(1.0, 0.0), "tmax": 6e-3},

    # High Temperature / Dilution & Extremes (Cases 19-24)
    {"name": "1500K_10atm_phi0.8_H2-10", "T0": 1500, "P": 10, "X": make_fuel_air_mix(0.8, 0.1), "tmax": 3e-3},
    {"name": "1550K_20atm_phi1.0_H2-30", "T0": 1550, "P": 20, "X": make_fuel_air_mix(1.0, 0.3), "tmax": 2e-3},
    {"name": "1000K_30atm_phi1.0_H2-pure", "T0": 1000, "P": 30, "X": make_fuel_air_mix(1.0, 1.0), "tmax": 3e-3},
    {"name": "1100K_40atm_phi1.0_H2-20_EGR10", "T0": 1100, "P": 40, "X": make_fuel_air_mix(1.0, 0.2, egr=0.10), "tmax": 8e-3},
    {"name": "1200K_30atm_phi1.0_H2-20_EGR15", "T0": 1200, "P": 30, "X": make_fuel_air_mix(1.0, 0.2, egr=0.15), "tmax": 6e-3},
    {"name": "1300K_40atm_phi1.0_H2-20_EGR20", "T0": 1300, "P": 40, "X": make_fuel_air_mix(1.0, 0.2, egr=0.20), "tmax": 5e-3},
]


def run_single_case(engine, case, dt, closure="UV", projection_mode="kl"):
    gas = engine.gas
    T0, Patm, X, tmax = case["T0"], case["P"], case["X"], case["tmax"]
    n_steps = int(round(tmax / dt))
    iNO = engine.names.index("NO")
    iH2 = engine.names.index("H2")
    iOH = engine.names.index("OH")
    
    # 1. Cantera ground truth reference
    gas.TPX = T0, Patm * ct.one_atm, X
    Y0 = gas.Y.copy()
    rho0 = gas.density
    u0 = gas.int_energy_mass
    h0 = gas.enthalpy_mass
    
    if closure == "UV":
        reactor = ct.IdealGasReactor(gas)
    else:  # HP
        reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])
    
    ref_t, ref_T = [0.0], [T0]
    for k in range(n_steps):
        sim.advance((k + 1) * dt)
        ref_t.append(sim.time)
        ref_T.append(gas.T)
    ref_Tend = ref_T[-1]
    ref_Tmax = max(ref_T)
    ref_NO = gas.Y[iNO]
    ref_idt = engine.calc_idt(ref_t, ref_T, T0)
    
    # 2. Autonomous NN rollout
    Y = Y0.copy()
    T = T0
    P = Patm
    nn_t, nn_T = [0.0], [T0]
    closure_ok = True
    fail_step = None
    wiped_species_count = 0
    
    for k in range(n_steps):
        Y = engine.step_model(T, P, Y, dt, projection_mode=projection_mode)
        
        # Check if active species were wiped to near zero
        for sp_name in ["H2", "OH", "H", "O"]:
            idx = engine.names.index(sp_name)
            if Y[idx] < 1e-25 and Y0[idx] > 1e-8:
                wiped_species_count += 1
                
        # Thermodynamic closure
        try:
            if closure == "UV":
                gas.UVY = u0, 1.0 / rho0, Y
                T = gas.T
                P = gas.P / ct.one_atm
            else:  # HP
                gas.HPY = h0, Patm * ct.one_atm, Y
                T = gas.T
                P = Patm
        except ct.CanteraError:
            closure_ok = False
            fail_step = k + 1
            break
            
        nn_t.append((k + 1) * dt)
        nn_T.append(T)
        
    nn_Tend = nn_T[-1]
    nn_Tmax = max(nn_T)
    nn_NO = Y[iNO]
    nn_idt = engine.calc_idt(nn_t, nn_T, T0)
    
    idt_ratio = nn_idt / ref_idt if (np.isfinite(nn_idt) and np.isfinite(ref_idt) and ref_idt > 0) else np.nan
    delta_Tend = nn_Tend - ref_Tend
    delta_Tmax = nn_Tmax - ref_Tmax
    no_rel_err = abs(nn_NO - ref_NO) / max(ref_NO, 1e-12)
    
    return {
        "case": case["name"],
        "T0": T0,
        "P": Patm,
        "closure": closure,
        "projection": projection_mode,
        "dt": dt,
        "ref_idt_ms": float(ref_idt * 1e3) if np.isfinite(ref_idt) else None,
        "nn_idt_ms": float(nn_idt * 1e3) if np.isfinite(nn_idt) else None,
        "idt_ratio": float(idt_ratio) if np.isfinite(idt_ratio) else None,
        "ref_Tend": float(ref_Tend),
        "nn_Tend": float(nn_Tend),
        "delta_Tend": float(delta_Tend),
        "ref_Tmax": float(ref_Tmax),
        "nn_Tmax": float(nn_Tmax),
        "ref_NO": float(ref_NO),
        "nn_NO": float(nn_NO),
        "no_rel_err": float(no_rel_err),
        "closure_ok": bool(closure_ok),
        "fail_step": fail_step,
        "wiped_species_count": int(wiped_species_count),
    }


def run_benchmark(root=DEFAULT_ROOT, ckpt_name=DEFAULT_CKPT, dt=1e-6, closure="UV", projection_mode="kl", cases=None, save_json=None):
    if cases is None:
        cases = BENCHMARK_CASES
        
    print(f"\n{'='*95}")
    print(f"CFD CLOSED-LOOP BENCHMARK: {ckpt_name}")
    print(f"Root: {root}")
    print(f"Settings: Closure={closure}, Projection={projection_mode}, dt={dt:.1e} s, Cases={len(cases)}")
    print(f"{'='*95}")
    
    engine = BenchmarkEngine(root=root, ckpt_name=ckpt_name)
    results = []
    
    header = f"{'Case Name':<28s} | {'IDT ref':>7s} {'NN':>7s} {'Ratio':>6s} | {'Tend ref':>8s} {'NN':>8s} {'dT':>6s} | {'NO err':>7s} | {'Status':>8s}"
    print(header)
    print("-" * len(header))
    
    for c in cases:
        t0 = time.time()
        res = run_single_case(engine, c, dt, closure=closure, projection_mode=projection_mode)
        results.append(res)
        
        ratio_str = f"{res['idt_ratio']:6.2f}" if res['idt_ratio'] is not None else "   nan"
        ref_idt_str = f"{res['ref_idt_ms']:7.3f}" if res['ref_idt_ms'] is not None else "    nan"
        nn_idt_str = f"{res['nn_idt_ms']:7.3f}" if res['nn_idt_ms'] is not None else "    nan"
        status_str = "OK" if res["closure_ok"] else f"FAIL@{res['fail_step']}"
        
        row = f"{res['case']:<28s} | {ref_idt_str} {nn_idt_str} {ratio_str} | {res['ref_Tend']:8.0f} {res['nn_Tend']:8.0f} {res['delta_Tend']:+6.0f} | {res['no_rel_err']:7.1%} | {status_str:>8s}"
        print(row, flush=True)
        
    print(f"{'='*95}")
    
    # Calculate summary metrics strictly via experiment_manifest
    summary_metrics = em.recompute_benchmark_summary(results)
    mean_ratio = summary_metrics["mean_idt_ratio"]
    median_ratio = summary_metrics["median_idt_ratio"]
    mean_dT = summary_metrics["mean_abs_delta_Tend"]
    mean_no_err = summary_metrics["mean_no_rel_err"]
    success_count = int(summary_metrics["closure_success_rate"] * len(results))
    total_wipes = summary_metrics["total_wiped_species"]
    ignited_count = summary_metrics["ignited_count"]

    ratio_mean_str = f"{mean_ratio:.2f}" if mean_ratio is not None else "nan"
    ratio_median_str = f"{median_ratio:.2f}" if median_ratio is not None else "nan"
    mean_dT_str = f"{mean_dT:.1f} K" if mean_dT is not None else "nan"
    mean_no_str = f"{mean_no_err:.1%}" if mean_no_err is not None else "nan"

    print("BENCHMARK SUMMARY:")
    print(f"  Cases Evaluated: {len(results)}")
    print(f"  Closure Success: {success_count}/{len(results)} ({summary_metrics['closure_success_rate']:.1%})")
    print(f"  Ignited Cases:   {ignited_count}/{len(results)}")
    print(f"  IDT Ratio Mean:  {ratio_mean_str}")
    print(f"  IDT Ratio Median:{ratio_median_str}")
    print(f"  Mean |Tend Err|: {mean_dT_str}")
    print(f"  Mean NO Rel Err: {mean_no_str}")
    print(f"  Total Wiped Sp:  {total_wipes}")
    print(f"{'='*95}\n")
    
    summary_data = {
        "checkpoint": ckpt_name,
        "root": root,
        "closure": closure,
        "projection": projection_mode,
        "dt": dt,
        **summary_metrics,
        "cases": results,
    }
    
    if save_json:
        json_path = os.path.abspath(save_json)
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"Benchmark results saved to: {json_path}")
        
    return summary_data


if __name__ == "__main__":
    print(
        "[LEGACY] Cantera corrects neural temperature after each step here. "
        "Use benchmark_autonomous_h128.py for independent neural T/P/Y rollouts.",
        file=sys.stderr, flush=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--dt", type=float, default=2e-6)
    parser.add_argument("--closure", default="UV", choices=["UV", "HP"])
    parser.add_argument("--projection", default="kl", choices=["kl", "linear", "none"])
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()
    
    run_benchmark(
        root=args.root,
        ckpt_name=args.ckpt,
        dt=args.dt,
        closure=args.closure,
        projection_mode=args.projection,
        save_json=args.save_json,
    )
