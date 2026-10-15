"""Fully closed-loop a-posteriori probe for the v71 checkpoint.

For each test case a fresh mixture is integrated three ways with the same random dt sequence:
  ref  : Cantera IdealGasReactor (const volume, adiabatic) -- ground truth
  ar_T : network species AR, T and P taken from the *reference* trajectory (= what validate_sc_v68 measures)
  auto : network species AR, T from exact internal-energy closure (Cantera UV), P from ideal gas
         (= what a CFD chemistry substep actually needs)
Outputs ignition delay (max dT/dt), final T, final NO, and the one-step oracle error by regime.
"""
import os, sys, glob, importlib.util, json, numpy as np, torch, joblib, cantera as ct
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
spec = importlib.util.spec_from_file_location('tr', 'train_sc_v68.py'); tr = importlib.util.module_from_spec(spec); sys.modules['tr'] = tr; spec.loader.exec_module(tr)
sys.modules['__main__'].FeatureScaler = tr.FeatureScaler
root = "output1/phase68_runs/v71_qssa_asymptotic_exact"
fi = joblib.load(glob.glob(root + "/scalers_v68/feature_info_*.pkl")[0])
sx = joblib.load(glob.glob(root + "/scalers_v68/scaler_X_*.pkl")[0]); sy = joblib.load(glob.glob(root + "/scalers_v68/scaler_Y_*.pkl")[0])
CKPT = os.environ.get("CKPT", "best_oracle_h128_epoch27.pth")
ck = torch.load(root + "/weights_v68/" + CKPT, map_location=tr.DEVICE, weights_only=False)
model = tr.CoupledKineticsNet(len(fi["input_cols"]), len(fi["output_cols"]), output_parameterization="qssa_asymptotic").to(tr.DEVICE)
model.load_state_dict(ck["model_state_dict"]); model.eval()
tools = tr.CoarseOperatorTools(fi, sx, sy, rate_decode_mode="qssa_asymptotic", rate_z_limit=10.0)
inert, macro, micro = fi["inert_species"], fi["macro_species"], fi["micro_species"]; react = macro + micro
gas = ct.Solution("NUIG2024.yaml"); names = gas.species_names
ridx = np.array([names.index(s) for s in react]); iidx = [names.index(s) for s in inert]

def build_X(T, P_atm, Y, dt):
    row = [T, P_atm] + [Y[names.index(s)] for s in inert] + [Y[names.index(s)] for s in macro] \
          + [np.log10(max(Y[names.index(s)], 1e-40)) for s in micro] + [np.log10(dt)]
    return torch.tensor(sx.transform(np.array(row, dtype=np.float64)[None, :]).astype(np.float32), device=tr.DEVICE)

def step_model(T, P_atm, Y, dt):
    with torch.no_grad():
        Xs = build_X(T, P_atm, Y, dt); out = model(Xs)
        ps = tools.predict_next_state(Xs, out, safe_for_rollout=True, state_update_mode="qssa_asymptotic")
        Yn = ps["Y_next_constrained_full"][0].double().cpu().numpy()
    full = Y.copy(); full[ridx] = Yn
    full = np.clip(full, 0, None); full /= full.sum()
    return full

def run_case(T0, Patm, X, tmax, seed, nmax):
    rng = np.random.default_rng(seed)
    dts = 10 ** rng.uniform(-8, -4, size=nmax)
    gas.TPX = T0, Patm * ct.one_atm, X
    Y0 = gas.Y.copy(); rho = gas.density
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r]); t = 0.0
    ref = [(0.0, gas.T, gas.P / ct.one_atm, gas.Y.copy())]
    for k in range(nmax):
        if t >= tmax: break
        sim.advance(t + dts[k]); t = sim.time
        ref.append((t, gas.T, gas.P / ct.one_atm, gas.Y.copy()))
    n = len(ref) - 1
    # (a) species AR with recorded T/P (validator setting)
    Y = Y0.copy(); arT = [Y.copy()]
    for k in range(n):
        dt = ref[k + 1][0] - ref[k][0]
        Y = step_model(ref[k][1], ref[k][2], Y, dt); arT.append(Y.copy())
    # (b) fully autonomous: T from internal energy closure at fixed (u, v)
    gas.TPY = T0, Patm * ct.one_atm, Y0; u0 = gas.int_energy_mass
    Y = Y0.copy(); T = T0; P = Patm; auto = [(T, P, Y.copy())]
    for k in range(n):
        dt = ref[k + 1][0] - ref[k][0]
        Y = step_model(T, P, Y, dt)
        try:
            gas.UVY = u0, 1.0 / rho, Y
            T = gas.T; P = gas.P / ct.one_atm
        except ct.CanteraError:
            break
        auto.append((T, P, Y.copy()))
    return ref, arT, auto

def idt(times, temps, T0):
    times = np.asarray(times); temps = np.asarray(temps)
    if temps.max() - T0 < 200: return np.nan
    thr = T0 + 0.5 * (temps.max() - T0)
    i = np.argmax(temps >= thr)
    return times[i]

def mix(phi, xh2, egr=0.0):
    # NH3/H2 fuel with air; xh2 = H2 mole fraction in fuel
    f_nh3, f_h2 = 1 - xh2, xh2
    o2 = (0.75 * f_nh3 + 0.5 * f_h2) / phi
    X = {"NH3": f_nh3, "H2": f_h2, "O2": o2, "N2": 3.76 * o2}
    if egr > 0:
        tot = sum(X.values()); X["N2"] += egr * tot
    return X

cases = [
    dict(name="engine_1200K_40atm_phi1_H2-20", T0=1200, P=40, X=mix(1.0, 0.2), tmax=5e-3),
    dict(name="engine_1100K_60atm_phi1_H2-40", T0=1100, P=60, X=mix(1.0, 0.4), tmax=5e-3),
    dict(name="engine_1000K_80atm_phi0.8_H2-30", T0=1000, P=80, X=mix(0.8, 0.3), tmax=1e-2),
    dict(name="lean_1300K_20atm_phi0.6_H2-10", T0=1300, P=20, X=mix(0.6, 0.1), tmax=5e-3),
    dict(name="rich_1400K_10atm_phi1.5_H2-20", T0=1400, P=10, X=mix(1.5, 0.2), tmax=5e-3),
    dict(name="hot_1600K_5atm_phi1_H2-0", T0=1600, P=5, X=mix(1.0, 0.0), tmax=3e-3),
]
out = []
for i, c in enumerate(cases):
    ref, arT, auto = run_case(c["T0"], c["P"], c["X"], c["tmax"], seed=100 + i, nmax=int(os.environ.get("NMAX", "1500")))
    tr_t = [x[0] for x in ref]; tr_T = [x[1] for x in ref]
    au_T = [x[0] for x in auto]; au_t = tr_t[:len(auto)]
    iNO = names.index("NO"); iNH3 = names.index("NH3"); iOH = names.index("OH")
    res = dict(case=c["name"], steps=len(ref) - 1,
               idt_ref=idt(tr_t, tr_T, c["T0"]), idt_auto=idt(au_t, au_T, c["T0"]),
               Tend_ref=tr_T[-1], Tend_auto=au_T[-1], Tmax_auto=max(au_T),
               NO_end_ref=ref[-1][3][iNO], NO_end_arT=arT[-1][iNO], NO_end_auto=auto[-1][2][iNO],
               NH3_end_ref=ref[-1][3][iNH3], NH3_end_arT=arT[-1][iNH3], NH3_end_auto=auto[-1][2][iNH3],
               OHmax_ref=max(x[3][iOH] for x in ref), OHmax_arT=max(y[iOH] for y in arT), OHmax_auto=max(x[2][iOH] for x in auto),
               auto_steps=len(auto) - 1)
    out.append(res)
    print(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
json.dump(out, open("test/diagnostics_v72/closed_loop_" + CKPT.replace(".pth", "") + ".json", "w"), indent=1, default=float)
