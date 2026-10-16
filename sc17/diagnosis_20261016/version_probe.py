"""Identical read-only probes for sc15/sc16/sc17, run in separate processes.

Uses the auxiliary 4800-row mini data and sc15 scalers to keep comparisons
controlled. These are mathematical/compatibility probes, not trained sc17 scores.
Original code/data/weights are never written. All output stays in this folder.
"""
from pathlib import Path
import sys, json, argparse, warnings, subprocess, hashlib, inspect
sys.stdout.reconfigure(encoding='utf-8')
parser = argparse.ArgumentParser()
parser.add_argument('version', choices=['sc15','sc16','sc17'])
args = parser.parse_args()
OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE / args.version
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import cantera as ct
import joblib
import train_sc_v68 as tr
from thermo_nasa7_layer import ThermoNASA7Layer

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
torch.set_num_threads(4)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FeatureScaler = tr.FeatureScaler
setattr(sys.modules['__main__'],'FeatureScaler',FeatureScaler)
scalers = BASE/'sc15/phase68_runs/v72_dual_branch/scalers_v68'
sx = joblib.load(next(scalers.glob('scaler_X*')))
sy = joblib.load(next(scalers.glob('scaler_Y*')))
fi = joblib.load(next(scalers.glob('feature_info*')))
fi.update(mechanism='nuig_2024',mechanism_path=str(ROOT/'NUIG2024.yaml'))
tools = tr.CoarseOperatorTools(fi,sx,sy,device=DEV)
mini = joblib.load(ROOT/'data_pipeline/mini_ds.pkl')
rows = mini['rows']; names = mini['names']
df = pd.DataFrame({k:[r[k] for r in rows] for k in ['T','P','dt']})
df['T_next'] = [r['Tn'] for r in rows]
for i,sp in enumerate(names):
    for key,col in [('Y','Y_'),('Yn','Y_next_'),('Pq','P_avg_'),('Dq','D_coef_avg_')]:
        df[col+sp]=[r[key][i] for r in rows]
    df['dYdt_'+sp]=(df['Y_next_'+sp]-df['Y_'+sp])/df.dt
order = fi['macro_species']+fi['micro_species']
full_order = fi['inert_species']+order
truth = df[['Y_next_'+sp for sp in order]].to_numpy()
xo = tr.transform_input_array(df,fi['inert_species'],fi['macro_species'],fi['micro_species'])
yt = tr.transform_target_array(df,fi['macro_species'],fi['micro_species'],fi['macro_rate_scale'],'dual_branch')
x = torch.tensor(sx.transform(xo),device=DEV,dtype=torch.float32)
y = torch.tensor(sy.transform(yt),device=DEV,dtype=torch.float32)

def stats(a,b):
    e = np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    return dict(rmse=float(np.sqrt(np.mean(e*e))),max_abs=float(np.max(abs(e))))

def error(exc):
    return dict(type=type(exc).__name__,message=str(exc),reason=getattr(exc,'reason',None))

result = dict(version=args.version,root=str(ROOT),rows=len(rows),
    environment=dict(python=sys.executable,cantera=ct.__version__,torch=torch.__version__,device=str(DEV)),
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and OUT not in p.parents},
    mini_sha256=hashlib.sha256((ROOT/'data_pipeline/mini_ds.pkl').read_bytes()).hexdigest())
inert_original=df[['Y_'+sp for sp in fi['inert_species']]].to_numpy()
inert_inverse=tools.inverse_state(x)[:,tools.inert_idx]
result['inert_input']=dict(original_minimum=float(inert_original.min()),
    original_negative_rows=int((inert_original<0).any(1).sum()),
    reconstructed_negative_rows=int((inert_inverse<0).any(1).sum()),
    reconstructed_minimum=float(inert_inverse.min()))
try:
    with torch.no_grad():
        tools.predict_next_state(x,y,experiment_mode='qssa',state_update_mode='dual_branch')
    result['default_input_decode']='returned'
except Exception as exc:result['default_input_decode']=error(exc)
# For the isolated decoder comparison only, pass the same explicitly nonnegative
# inert sum to all versions. This does not change any original input/source.
controlled_inert=torch.tensor(np.maximum(inert_original,0).sum(1,keepdims=True),device=DEV,dtype=torch.float32)
with torch.no_grad():
    pred = tools.predict_next_state(x,y,experiment_mode='qssa',state_update_mode='dual_branch',inert_sum=controlled_inert)
    yp = pred['Y_next_raw_full'].cpu().numpy()
    yc = pred['Y_next_constrained_full'].cpu().numpy()
result['perfect_targets'] = dict(
    NH3_raw=stats(yp[:,order.index('NH3')],truth[:,order.index('NH3')]),
    NH3_projected=stats(yc[:,order.index('NH3')],truth[:,order.index('NH3')]),
    macro_clipped_counts=dict(zip(fi['macro_species'],(np.abs(yt[:,:tools.n_macro])>20).sum(0).tolist())),
    control='explicit_nonnegative_inert_sum_same_for_all_versions')
yn = torch.tensor(truth,device=DEV,dtype=torch.float32)
inert = torch.tensor(df[['Y_next_'+sp for sp in fi['inert_species']]].to_numpy(),device=DEV,dtype=torch.float32).clamp_min(0).sum(1,keepdim=True)
with torch.no_grad():
    pm,pu,diag = tr.project_physical_state(yn[:,:tools.n_macro],yn[:,tools.n_macro:],inert,
        element_matrix=tools.A,target_elements=yn@tools.A.T)
result['exact_endpoint_projection'] = dict(
    micro_log_error=stats(torch.log10(pu.clamp_min(1e-40)).cpu().numpy(),torch.log10(yn[:,tools.n_macro:].clamp_min(1e-40)).cpu().numpy()),
    lifted_above1e32=int(((yn[:,tools.n_macro:]<1e-32)&(pu>1e-32)).sum()))

# Test conservation recovery on the same difficult but feasible target/candidate.
gas = ct.Solution(str(ROOT/'NUIG2024.yaml'))
A = np.array([[gas.n_atoms(sp,e)*gas.atomic_weight(e)/gas.molecular_weights[gas.species_index(sp)]
               for sp in order] for e in ['N','H','O']])
A_t = torch.tensor(A,device=DEV,dtype=torch.float32)
rng = np.random.default_rng(1729)
raw = torch.tensor(rng.normal(0,8,(32,len(order))),device=DEV,dtype=torch.float32).softmax(1)
target_y = yn[:32].clamp_min(0)
target_y = target_y/target_y.sum(1,keepdim=True)
target_e = target_y@A_t.T
stress = dict(returned=0,rejected=0,fallback_used=0,invalid_returned=0,max_mass_error=0.,max_element_error=0.,
              trace_values_zeroed=0,errors=[])
for i in range(32):
    try:
        ma,mi,dg = tr.project_physical_state(raw[i:i+1,:tools.n_macro],raw[i:i+1,tools.n_macro:],0.,
            element_matrix=A_t,target_elements=target_e[i:i+1])
        ret=torch.cat([ma,mi],1).double()
        mass_err=float((ret.sum(1)-1).abs().max())
        elem_err=float((ret@A_t.double().T-target_e[i:i+1].double()).abs().max())
        stress['returned']+=1
        stress['fallback_used']+=int(bool(torch.as_tensor(dg.get('projection_fallback_used',False)).any()))
        stress['invalid_returned']+=int(mass_err>5e-6 or elem_err>1e-6 or bool((ret<0).any()))
        stress['max_mass_error']=max(stress['max_mass_error'],mass_err)
        stress['max_element_error']=max(stress['max_element_error'],elem_err)
        stress['trace_values_zeroed']+=int(((raw[i:i+1]>1e-30)&(ret<=1e-40)).sum())
    except Exception as exc:
        stress['rejected']+=1
        if len(stress['errors'])<2:stress['errors'].append(error(exc))
result['projection_stress_32'] = stress

# Same finite saturated raw candidate for each version, bypassing projection
# only to isolate the exact conservation-loss derivative.
physical,macro,log_micro,micro = tools.get_current_state(x[:8])
bad_micro=torch.full_like(micro,10.,requires_grad=True)
raw_state=dict(Y_macro_old=macro,logY_micro_old=log_micro,
    Y_next_raw_full=torch.cat([macro,bad_micro],1),Y_next_safe_full=torch.cat([macro,bad_micro],1))
el,ms=tr.compute_element_conservation_loss(tools,raw_state,state_loss_source='raw')
g=torch.autograd.grad(el+ms,bad_micro)[0]
result['saturated_conservation_loss']=dict(element=float(el.detach()),mass=float(ms.detach()),max_abs_gradient=float(g.abs().max()))

# Ground-truth energy update. sc17 rejects a too-large interval instead of
# returning the silently capped state used by previous versions.
layer=ThermoNASA7Layer(ROOT/'NUIG2024.yaml',full_order,device=DEV)
t=torch.tensor(df['T'].to_numpy(),device=DEV,dtype=torch.float64).reshape(-1,1)
tn=df['T_next'].to_numpy().reshape(-1,1)
yo_full=torch.tensor(df[['Y_'+sp for sp in full_order]].to_numpy(),device=DEV,dtype=torch.float64)
yn_full=torch.tensor(df[['Y_next_'+sp for sp in full_order]].to_numpy(),device=DEV,dtype=torch.float64)
closure=dict(nasa7_layer=layer,reactor_type='IdealGasReactor',species_order=full_order,
             heat_capacity_coeff=np.ones(len(full_order)),energy_coeff=np.zeros(len(full_order)))
with torch.no_grad():
    solved=layer.solve_temperature_uv(layer.compute_mixture_properties(t,yo_full)[3],yn_full,t)
result['thermal']=dict(uncapped=stats(solved.cpu().numpy(),tn),real_steps_over500=int((np.abs(t.cpu().numpy()-tn)>500).sum()))
for limit in [500.,3000.]:
    try:
        with torch.no_grad():
            ts=tr.update_temperature_autonomous(t,yn_full-yo_full,max_delta_temperature=limit,
                 thermo_closure=closure,Y_old_full=yo_full,Y_next_full=yn_full)
        result['thermal'][str(int(limit))]=dict(returned=True,error=stats(ts.cpu().numpy(),tn))
    except Exception as exc:result['thermal'][str(int(limit))]=dict(returned=False,**error(exc))

# Float32 standardization can reconstruct a trace major species as slightly
# negative. Supply a nonnegative next composition to isolate current-state EOS.
physical,macro,log_micro,micro=tools.get_current_state(x)
neg = (macro<0).any(1)
indices=(neg & (inert_inverse>=0).all(1)).nonzero(as_tuple=False).reshape(-1)
result['input_roundoff']=dict(negative_macro_rows=int(neg.sum()),negative_macro_values=int((macro<0).sum()),
                             minimum_macro=float(macro.min()))
if len(indices):
    i=int(indices[0]); next_macro=macro[i:i+1].clamp_min(0)
    try:
        with torch.no_grad():
            tools.state_to_input_scaled(x[i:i+1],next_macro,log_micro[i:i+1],temperature_mode='autonomous',
                thermo_closure=closure,current_X_scaled=x[i:i+1],max_delta_temperature=3000.)
        result['input_roundoff']['near_identity_feedback']='returned'
    except Exception as exc: result['input_roundoff']['near_identity_feedback']=error(exc)

# Preset guard is reached before filesystem setup or optimization.
import train_all_v68 as ta
launch=ta.parse_args_from(['--plan','v72_dual_branch','--mode','full','--no-confirm'])
keys=['output_parameterization','rate_decode_mode','state_update_mode','temperature_mode',
      'feedback_noise_std','feedback_state_source','max_delta_temperature']
result['preset']={k:getattr(launch,k) for k in keys}
if args.version=='sc17':
    try:
        tr.train_model_v68(data_dir=str(OUT/'nonexistent_dataset_guard'),mechanism='nuig_2024',
            **{k:getattr(launch,k) for k in keys})
        result['preset_start_guard']='unexpectedly passed'
    except Exception as exc:result['preset_start_guard']=error(exc)

# The old allowed scaled-space feedback noise is incompatible with strict
# element+unit-mass targets when constrained feedback is selected.
species_idx=torch.cat([tools.macro_idx,tools.micro_idx])
torch.manual_seed(1729)
noisy=x[:128].clone()
noisy[:,species_idx]=tr.add_feedback_state_noise(noisy[:,species_idx],0.00475,
    torch.ones(128,1,dtype=torch.bool,device=DEV))
_,nm,nl,nu=tools.get_current_state(noisy)
prior_mass=nm.sum(1)+nu.clamp(max=1).sum(1)+tools.inverse_state(noisy)[:,tools.inert_idx].sum(1)
result['scaled_noise']=dict(rows=128,off_mass_tolerance_rows=int(((prior_mass-1).abs()>5e-6).sum()),
                          max_mass_error=float((prior_mass-1).abs().max()))
try:
    with torch.no_grad(): tools.predict_next_state(noisy,y[:128],experiment_mode='qssa',state_update_mode='dual_branch')
    result['scaled_noise']['decoder']='returned'
except Exception as exc:result['scaled_noise']['decoder']=error(exc)

if args.version=='sc17':
    import autonomous_pressure as ap
    y0=yo_full.clamp_min(0);y0=y0/y0.sum(1,keepdim=True)
    y1=yn_full.clamp_min(0);y1=y1/y1.sum(1,keepdim=True)
    p0=torch.tensor(df['P'].to_numpy(),device=DEV,dtype=torch.float64).reshape(-1,1)
    t1=torch.tensor(tn,device=DEV,dtype=torch.float64)
    eos=ap.autonomous_pressure_next(p0,t,t1,y0,y1,layer.W,'IdealGasReactor')
    pref=[]
    for i in range(len(rows)):
        gas.TPY=float(t[i]),float(p0[i])*ct.one_atm,y0[i].cpu().numpy()[[full_order.index(sp) for sp in gas.species_names]]
        density=gas.density
        gas.TDY=float(t1[i]),density,y1[i].cpu().numpy()[[full_order.index(sp) for sp in gas.species_names]]
        pref.append(gas.P/ct.one_atm)
    result['eos_vs_cantera']=stats(eos.cpu().numpy().reshape(-1),np.array(pref))

    # Poison only the future reference P: autonomous feedback must ignore it.
    good=((macro>=0).all(1)&((yn[:,:tools.n_macro]>=0).all(1))&(inert_inverse>=0).all(1)).nonzero(as_tuple=False).reshape(-1)
    i=int(good[0]); base=x[i:i+1].clone(); other=base.clone()
    other[:,tools.pressure_idx]+=100./tools.scale_X[tools.pressure_idx]
    nxt_macro=yn[i:i+1,:tools.n_macro]
    nxt_log=torch.log10(yn[i:i+1,tools.n_macro:].clamp_min(1e-40))
    with torch.no_grad():
        a=tools.state_to_input_scaled(base,nxt_macro,nxt_log,temperature_mode='autonomous',thermo_closure=closure,
             current_X_scaled=x[i:i+1],max_delta_temperature=3000.)
        b=tools.state_to_input_scaled(other,nxt_macro,nxt_log,temperature_mode='autonomous',thermo_closure=closure,
             current_X_scaled=x[i:i+1],max_delta_temperature=3000.)
    result['future_reference_pressure_invariance']=dict(max_scaled_difference=float((a-b).abs().max()))

    import importlib.util
    spec=importlib.util.spec_from_file_location('split_probe',ROOT/'data_pipeline/split_trajectories_v21.py')
    spl=importlib.util.module_from_spec(spec);spec.loader.exec_module(spl)
    ids=np.array([str(i) for i in range(20)],dtype=object)
    metadata=pd.DataFrame({'condition_group_id':[f'g{i//2}' for i in range(20)]},index=ids)
    splits=spl._split_ids(ids,np.random.default_rng(1729),.6,.2,metadata=metadata,stratify=False)
    groups={k:set(metadata.loc[list(v),'condition_group_id']) for k,v in splits.items()}
    result['unstratified_group_split']=dict(group_overlap=any(groups[a]&groups[b] for a in groups for b in groups if a!=b),
                                           sizes={k:len(v) for k,v in splits.items()})
    import autonomous_benchmark_protocol as bp
    tt=np.array([0.,.001,.002])
    yr=np.array([[.1,.0],[.05,.05],[0.,.1]])
    met=bp.compare_complete_trajectories(tt,np.array([1000.,1500.,2000.]),np.array([1000.,1050.,1100.]),
          np.ones(3),np.ones(3),yr,yr,['NH3','H2O'])
    result['benchmark_missing_ignition']= {k:met[k] for k in ['ref_ignited','nn_ignited','idt_nn_s','idt_abs_error_s','ignition_agreement']}

(OUT/(args.version+'_probes.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2,ensure_ascii=False))
