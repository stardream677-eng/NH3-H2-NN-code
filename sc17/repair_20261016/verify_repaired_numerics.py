"""Record reproducible probes using the diagnosed 4,800 real transitions."""
from pathlib import Path
import sys,importlib.util,json
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('probe_fixture',ROOT/'tests/conftest.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
import train_sc_v68 as tr
import physical_projection_contract as ppc
from thermo_nasa7_layer import ThermoNASA7Layer

tools,x,y,df,fi=fixture.chemistry.__wrapped__()
with torch.no_grad():
    pred=tools.predict_next_state(x,y,experiment_mode='qssa',state_update_mode='dual_branch')
    truth=torch.tensor(df[['Y_next_'+s for s in fi['macro_species']]].values,dtype=torch.float32)
    denorm=y[:,:tools.n_macro]*tools.scale_Y[:tools.n_macro]+tools.mean_Y[:tools.n_macro]
    _,old_macro,_,_=tools.get_current_state(x)
    old_capped=old_macro+tools.macro_rate_scale*torch.sinh(denorm.clamp(-20,20))*tools.get_current_dt(x)
    order=fi['inert_species']+fi['macro_species']+fi['micro_species']
    old=torch.tensor(df[['Y_'+s for s in order]].values,dtype=torch.float64)
    nxt=torch.tensor(df[['Y_next_'+s for s in order]].values,dtype=torch.float64)
    layer=ThermoNASA7Layer(ROOT/'NUIG2024.yaml',order,device='cpu')
    temp=torch.tensor(df['T'].values,dtype=torch.float64)[:,None]
    result=tr.update_temperature_autonomous(temp,nxt-old,max_delta_temperature=500.,
        thermo_closure=dict(nasa7_layer=layer,reactor_type='IdealGasReactor',species_order=order),
        Y_old_full=old,Y_next_full=nxt)
    t_truth=torch.tensor(df.T_next.values,dtype=torch.float64)[:,None]
    dtemp=(df.T_next-df['T']).abs().values
    record=dict(rows=len(df),macro_species=fi['macro_species'],
        macro_asinh_abs_max=float(denorm.abs().max()),
        macro_outputs_exceeding_old_cap=int((denorm.abs()>20).sum()),
        old_capped_perfect_label_macro_max_error=float((old_capped-truth).abs().max()),
        repaired_perfect_label_macro_raw_max_error=float((pred['Y_macro_next_raw']-truth).abs().max()),
        repaired_perfect_label_macro_projected_max_error=float((pred['Y_macro_next_constrained']-truth).abs().max()),
        real_temperature_jumps_over_500K=int((dtemp>500).sum()),
        repaired_temperature_all_rows_max_error_K=float((result-t_truth).abs().max()),
        repaired_temperature_ignition_max_error_K=float((result-t_truth).abs()[torch.tensor(dtemp>500)].max()))
(ROOT/'repair_20261016/repaired_probe_results.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
