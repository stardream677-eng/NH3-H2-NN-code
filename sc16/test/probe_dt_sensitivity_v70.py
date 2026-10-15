"""Probe dt-sensitivity and cap compensation of a trained V68/V69 QSSA checkpoint.

Run from the sc15 directory:
    set KMP_DUPLICATE_LIB_OK=TRUE
    python probe_dt_sensitivity_v70.py

Findings on v70 best_composite_epoch20 (2026-10-09): at a true post-flame
equilibrium state the predicted log10(Y_next/Y) of radicals varies by ~1 dex
across dt in [1e-8, 1e-4]; oracle sawtooth std is 0.3-0.6 dex with random dt
and exactly 0.00 with constant dt.  Root cause: max_damping=15 cap in
predict_next_state forces the P head to learn P_pred ~ P_true*(15/(D*dt))^0.8.
"""
import sys, os, importlib.util, glob, numpy as np, torch, joblib, cantera as ct
spec=importlib.util.spec_from_file_location('tr','train_sc_v68.py'); tr=importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
sys.modules['__main__'].FeatureScaler = tr.FeatureScaler
root=os.environ.get("V70_RUN_ROOT", "../sc15-0/v70/phase68_runs/v69_qssa_stabilized")
fi=joblib.load(glob.glob(root+"/scalers_v68/feature_info_*.pkl")[0]); sx=joblib.load(glob.glob(root+"/scalers_v68/scaler_X_*.pkl")[0]); sy=joblib.load(glob.glob(root+"/scalers_v68/scaler_Y_*.pkl")[0])
ck=torch.load(root+"/weights_v68/best_composite_epoch20.pth",map_location=tr.DEVICE,weights_only=False)
model=tr.CoupledKineticsNet(len(fi["input_cols"]),len(fi["output_cols"]),output_parameterization="qssa").to(tr.DEVICE); model.load_state_dict(ck["model_state_dict"]); model.eval()
tools=tr.CoarseOperatorTools(fi,sx,sy,rate_decode_mode="qssa",rate_z_limit=10.0)
inert,macro,micro=fi["inert_species"],fi["macro_species"],fi["micro_species"]; react=macro+micro
gas=ct.Solution("NUIG2024.yaml"); W=gas.molecular_weights; names=gas.species_names
def build_X(T,P_atm,Y,dt):
    row=[T,P_atm]+[Y[names.index(s)] for s in inert]+[Y[names.index(s)] for s in macro]+[np.log10(max(Y[names.index(s)],1e-20)) for s in micro]+[np.log10(dt)]
    return torch.tensor(sx.transform(np.array(row,dtype=np.float32)[None,:]),dtype=torch.float32,device=tr.DEVICE)
def true_PD(gas):
    C=gas.creation_rates*W/gas.density; Dd=gas.destruction_rates*W/gas.density; Y=gas.Y
    return C, Dd/np.maximum(Y,1e-20)
def step_model(T,P_atm,Y,dt):
    with torch.no_grad():
        Xs=build_X(T,P_atm,Y,dt); out=model(Xs)
        ps=tools.predict_next_state(Xs,out,safe_for_rollout=False,state_update_mode="qssa")
        Yn=ps["Y_next_raw_full"][0].cpu().numpy()
        Pp,Dp=tools.inverse_qssa_parameters(out)
    full=Y.copy()
    for k,s in enumerate(react): full[names.index(s)]=Yn[k]
    return full, Pp[0].cpu().numpy(), Dp[0].cpu().numpy()

if __name__ == "__main__":
    X = {"NH3": 0.1035, "H2": 0.0259, "O2": 0.0906, "N2": 0.78}
    gas.TPX = 1300.0, 40 * ct.one_atm, X
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r]); sim.advance(0.02)
    Teq, Yeq, Peq = gas.T, gas.Y.copy(), gas.P / ct.one_atm
    Pt, Dt = true_PD(gas)
    key = ["OH", "H", "O", "HO2", "NH2", "NO", "N2O", "H2O2", "NH", "NH3", "O2", "H2", "H2O"]
    dts = [10.0 ** e for e in range(-8, -3)]
    print(f"post-flame equilibrium: T={Teq:.0f} K P={Peq:.1f} atm")
    print("species   D_true(1/s) | " + " ".join(f" dt=1e{e:+d}" for e in range(-8, -3)) + "   spread(dex)")
    for s in key:
        j = names.index(s); vals = []
        for dt in dts:
            Yn, _, _ = step_model(Teq, Peq, Yeq, dt)
            vals.append(np.log10(max(Yn[j], 1e-30) / max(Yeq[j], 1e-30)))
        print(f"{s:8s} {Dt[j]:10.2e} | " + " ".join(f"{v:+8.2f}" for v in vals) + f"   {max(vals)-min(vals):5.2f}")
