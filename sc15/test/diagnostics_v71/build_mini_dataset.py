# Build a small but representative Cantera dataset with the runner's exact inversion,
# using the NEW lhs sampler strata, so that alternative target definitions can be compared.
import sys, importlib.util, numpy as np, cantera as ct, time, pickle, os
spec=importlib.util.spec_from_file_location('rn','runner_sc_v21_qssa.py'); rn=importlib.util.module_from_spec(spec); sys.modules['rn']=rn; spec.loader.exec_module(rn)
spec2=importlib.util.spec_from_file_location('lhs','lhs_sampler_sc_v11.py'); lhs=importlib.util.module_from_spec(spec2); sys.modules['lhs']=lhs; spec2.loader.exec_module(lhs)
gas=ct.Solution("NUIG2024.yaml"); W=gas.molecular_weights; names=gas.species_names
s=lhs.UniversalLHSSampler_Species({"nuig_2024":{}})
df=s.generate_species_database(total_pts=int(os.environ.get("NPTS","200")))
xcols=[c for c in df.columns if c.startswith("X_") and c!="X_H2_fuel"]
rows=[]; t0=time.time(); ntraj=0
for _,smp in df.iterrows():
    comp={c[2:]:float(smp[c]) for c in xcols if float(smp[c])>1e-12 and c[2:] in names}
    try:
        gas.TPX=float(smp["T"]),float(smp["P"])*ct.one_atm,comp
    except Exception: continue
    r=ct.IdealGasReactor(gas); sim=ct.ReactorNet([r]); rng=np.random.default_rng(int(smp["sample_id"])); t=0.0; n=0
    try:
        while t<5e-3 and n<160:
            dt=10**rng.uniform(-8,-4); Y0=gas.Y.copy(); T0=gas.T; P0=gas.P/ct.one_atm
            Cs=gas.creation_rates*W/gas.density; Ds=gas.destruction_rates*W/gas.density/np.maximum(Y0,1e-300)
            sim.advance(t+dt); dtr=sim.time-t; t=sim.time; Y1=gas.Y.copy()
            Ce=gas.creation_rates*W/gas.density; De=gas.destruction_rates*W/gas.density/np.maximum(Y1,1e-300)
            Pq,Dq=rn._solve_exact_qssa_params(Y0,Y1,dtr,(Cs+Ce)/2,(Ds+De)/2)
            rows.append(dict(sid=int(smp["sample_id"]),stratum=smp["stratum_name"],dt=dtr,T=T0,P=P0,Tn=gas.T,Y=Y0,Yn=Y1,Pq=Pq,Dq=Dq,Praw=(Cs+Ce)/2,Draw=(Ds+De)/2))
            n+=1
        ntraj+=1
    except Exception as e:
        pass
print(f"trajectories={ntraj} rows={len(rows)} time={time.time()-t0:.0f}s")
pickle.dump(dict(rows=rows,names=names),open(os.environ.get("OUT","mini_ds.pkl"),"wb"))
