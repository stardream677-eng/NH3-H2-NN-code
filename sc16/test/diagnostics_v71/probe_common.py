import sys, os, importlib.util, glob, numpy as np, torch, joblib, cantera as ct
spec=importlib.util.spec_from_file_location('tr','train_sc_v68.py'); tr=importlib.util.module_from_spec(spec); sys.modules['tr']=tr; spec.loader.exec_module(tr)
sys.modules['__main__'].FeatureScaler = tr.FeatureScaler
spec2=importlib.util.spec_from_file_location('rn','runner_sc_v21_qssa.py'); rn=importlib.util.module_from_spec(spec2); sys.modules['rn']=rn; spec2.loader.exec_module(rn)
root="phase68_runs/v71_qssa_asymptotic_exact"
fi=joblib.load(glob.glob(root+"/scalers_v68/feature_info_*.pkl")[0]); sx=joblib.load(glob.glob(root+"/scalers_v68/scaler_X_*.pkl")[0]); sy=joblib.load(glob.glob(root+"/scalers_v68/scaler_Y_*.pkl")[0])
CKPT=os.environ.get("CKPT","best_oracle_h128_epoch16.pth")
ck=torch.load(root+"/weights_v68/"+CKPT,map_location=tr.DEVICE,weights_only=False)
model=tr.CoupledKineticsNet(len(fi["input_cols"]),len(fi["output_cols"]),output_parameterization="qssa_asymptotic").to(tr.DEVICE); model.load_state_dict(ck["model_state_dict"]); model.eval()
tools=tr.CoarseOperatorTools(fi,sx,sy,rate_decode_mode="qssa_asymptotic",rate_z_limit=10.0)
inert,macro,micro=fi["inert_species"],fi["macro_species"],fi["micro_species"]; react=macro+micro
gas=ct.Solution("NUIG2024.yaml"); W=gas.molecular_weights; names=gas.species_names
ridx=[names.index(s) for s in react]
def build_X(T,P_atm,Y,dt):
    row=[T,P_atm]+[Y[names.index(s)] for s in inert]+[Y[names.index(s)] for s in macro]+[np.log10(max(Y[names.index(s)],1e-40)) for s in micro]+[np.log10(dt)]
    return torch.tensor(sx.transform(np.array(row,dtype=np.float64)[None,:]).astype(np.float32),dtype=torch.float32,device=tr.DEVICE)
def step_model(T,P_atm,Y,dt,safe=True):
    with torch.no_grad():
        Xs=build_X(T,P_atm,Y,dt); out=model(Xs)
        ps=tools.predict_next_state(Xs,out,safe_for_rollout=safe,state_update_mode="qssa_asymptotic")
        Yn=(ps["Y_next_constrained_full"] if safe else ps["Y_next_raw_full"])[0].double().cpu().numpy()
        z=(out*tools.scale_Y+tools.mean_Y)[0].double().cpu().numpy()
    full=Y.copy(); full[ridx]=Yn
    return full, z
def cantera_traj(T0,Patm,X,tmax,dt_fn,nmax=4000):
    gas.TPX=T0,Patm*ct.one_atm,X; r=ct.IdealGasReactor(gas); sim=ct.ReactorNet([r]); t=0.0; rec=[]
    while t<tmax and len(rec)<nmax:
        dt=dt_fn(len(rec)); Y0=gas.Y.copy(); T_0=gas.T; P0=gas.P/ct.one_atm
        Cs=gas.creation_rates*W/gas.density; Ds=gas.destruction_rates*W/gas.density/np.maximum(Y0,1e-300)
        sim.advance(t+dt); dtr=sim.time-t; t=sim.time; Y1=gas.Y.copy()
        Ce=gas.creation_rates*W/gas.density; De=gas.destruction_rates*W/gas.density/np.maximum(Y1,1e-300)
        Pq,Dq=rn._solve_exact_qssa_params(Y0,Y1,dtr,(Cs+Ce)/2,(Ds+De)/2)
        rec.append(dict(t=t-dtr,dt=dtr,T=T_0,P=P0,Tn=gas.T,Y=Y0,Yn=Y1,Pq=Pq,Dq=Dq))
    return rec
def targets(P,D):
    logP=np.log10(np.clip(P,1e-40,None)); logD=np.log10(np.clip(D,1e-40,None))
    z1=np.where(P<=0,-45.0,np.where(D<=0,logP,np.clip(logP-logD,-40,30))); z2=np.where(D<=0,-45.0,logD)
    return z1,z2
L=lambda a: np.log10(np.maximum(a,1e-40))
