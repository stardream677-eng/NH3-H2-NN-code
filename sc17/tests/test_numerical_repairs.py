from pathlib import Path
import subprocess,sys
import numpy as np
import pytest
import torch
import cantera as ct
import train_sc_v68 as tr
import kinetics_parameterization as kp
import physical_projection_contract as ppc
from thermo_nasa7_layer import ThermoNASA7Layer

ROOT=Path(__file__).resolve().parents[1]

def test_true_labels_reconstruct_macro_endpoint(chemistry):
    tools,x,y,df,fi=chemistry
    p=tools.predict_next_state(x,y,experiment_mode='qssa',state_update_mode='dual_branch',
                               inert_sum=tools.inverse_state(x)[:,tools.inert_idx].clamp_min(0).sum(1,keepdim=True))
    truth=torch.tensor(df[['Y_next_'+s for s in fi['macro_species']]].values,dtype=torch.float32)
    assert (p['Y_next_raw_full'][:,:tools.n_macro]-truth).abs().max()<2e-6

def test_real_inputs_do_not_fail_on_quantization(chemistry):
    tools,x,y,_,_=chemistry
    current=tools.inverse_state(x)
    assert (current[:,torch.cat([tools.inert_idx,tools.macro_idx])] >= 0).all()
    p=tools.predict_next_state(x,y,experiment_mode='qssa',state_update_mode='dual_branch')
    assert torch.isfinite(p['Y_next_constrained_full']).all()

def test_roundoff_cleanup_preserves_material_negative_for_rejection(chemistry):
    tools,x,_,_,_=chemistry
    bad=x[:1].clone();idx=tools.inert_idx[0]
    bad[:,idx]=(-0.01-tools.mean_X[idx])/tools.scale_X[idx]
    assert tools.inverse_state(bad)[0,idx]<-0.009
    with pytest.raises(ppc.ProjectionConstraintError):
        tools.predict_next_state(bad,torch.zeros(1,tools.n_macro+2*tools.n_micro),state_update_mode='dual_branch')

def test_bad_finite_state_keeps_restoring_gradients(chemistry):
    tools,x,_,_,_=chemistry
    _,ma,lm,mi=tools.get_current_state(x[:8])
    bad=torch.full_like(mi,10.,requires_grad=True)
    state=dict(Y_macro_old=ma,logY_micro_old=lm,Y_next_raw_full=torch.cat([ma,bad],1),Y_next_safe_full=torch.cat([ma,bad],1))
    a,b=tr.compute_element_conservation_loss(tools,state,state_loss_source='raw')
    grad=torch.autograd.grad(a+b,bad)[0]
    assert torch.isfinite(grad).all() and (grad.abs()>0).all()

def test_raw_micro_above_one_has_nonzero_gradient(chemistry):
    tools,x,y,_,_=chemistry
    target=y[:1].clone().requires_grad_()
    # Pure production sentinel in old dual_branch, deliberately unphysical Y>1.
    z=(torch.tensor(10.)-tools.mean_Y[tools.n_macro])/tools.scale_Y[tools.n_macro]
    with torch.no_grad():
        target[:,tools.n_macro]=z
        target[:,tools.n_macro+1]=(-45.-tools.mean_Y[tools.n_macro+1])/tools.scale_Y[tools.n_macro+1]
    p=tools.predict_next_state(x[:1],target,experiment_mode='qssa',state_update_mode='dual_branch')
    g=torch.autograd.grad(p['Y_next_raw_full'][:,tools.n_macro].sum(),target)[0]
    assert torch.isfinite(g).all() and g[0,tools.n_macro].abs()>0

def test_difficult_feasible_projection(chemistry):
    tools,_,_,df,fi=chemistry
    order=fi['macro_species']+fi['micro_species'];gas=ct.Solution(str(ROOT/'NUIG2024.yaml'))
    A=torch.tensor([[gas.n_atoms(s,e)*gas.atomic_weight(e)/gas.molecular_weights[gas.species_index(s)] for s in order] for e in ['N','H','O']],dtype=torch.float32)
    truth=torch.tensor(df[['Y_next_'+s for s in order]].values[:32],dtype=torch.float32).clamp_min(0)
    truth=truth/truth.sum(1,keepdim=True)
    raw=torch.tensor(np.random.default_rng(1729).normal(0,8,(32,len(order))),dtype=torch.float32).softmax(1)
    ma,mi,_=tr.project_physical_state(raw[:,:tools.n_macro],raw[:,tools.n_macro:],0.,element_matrix=A,target_elements=truth@A.T)
    ppc.require_projection_valid(torch.cat([ma,mi],1),torch.zeros(32,1),A,truth@A.T)
    residual=ppc.projection_residuals(torch.cat([ma,mi],1),torch.zeros(32,1),A,truth@A.T)
    assert residual['element'].max()<1e-7,'Local projection errors must leave headroom for long rollouts'

@pytest.mark.parametrize('zp,zd',[(40.,40.),(40.,-10.),(-20.,40.)])
def test_pd_extreme_finite_predictions_have_finite_gradients(zp,zd):
    a=torch.tensor([[zp]],requires_grad=True);b=torch.tensor([[zd]],requires_grad=True)
    yy,ll=kp.pd_v1_micro_log_update(torch.tensor([[-15.]]),a,b,torch.tensor([[1e-4]]))
    assert torch.isfinite(yy).all() and torch.isfinite(ll).all()
    g=torch.autograd.grad(ll.sum(),[a,b])
    assert all(torch.isfinite(t).all() for t in g)

def test_temperature_real_ignition_is_not_artificially_capped(chemistry):
    _,_,_,df,fi=chemistry
    order=fi['inert_species']+fi['macro_species']+fi['micro_species']
    i=int(np.argmax(abs(df.T_next-df['T'])))
    t=torch.tensor([[df['T'].iloc[i]]],dtype=torch.float64)
    old=torch.tensor(df[['Y_'+s for s in order]].iloc[[i]].values,dtype=torch.float64)
    nxt=torch.tensor(df[['Y_next_'+s for s in order]].iloc[[i]].values,dtype=torch.float64)
    layer=ThermoNASA7Layer(ROOT/'NUIG2024.yaml',order,device='cpu')
    closure=dict(nasa7_layer=layer,reactor_type='IdealGasReactor',species_order=order)
    result=tr.update_temperature_autonomous(t,nxt-old,max_delta_temperature=500.,thermo_closure=closure,Y_old_full=old,Y_next_full=nxt)
    assert abs(float(result)-df.T_next.iloc[i])<1e-3

def test_autonomous_preset_disables_unphysical_noise():
    import train_all_v68 as ta
    for name,c in ta.PLAN_PRESETS.items():
        if c.get('temperature_mode')=='autonomous':assert c.get('feedback_noise_std',0)==0,name

@pytest.mark.parametrize('script,args',[
    ('data_pipeline/build_training_dataset_v72.py',['--help']),
    ('validate_all_best_models_v68.py',['--plan','v72_dual_branch','--help']),
])
def test_entrypoint_help(script,args):
    p=subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stderr

def test_v72_validation_plan_is_selectable():
    import validate_all_best_models_v68 as val
    assert 'v72_dual_branch' in val.MODEL_VERSIONS


def test_v72_checkpoints_are_discovered_with_their_update_rules(tmp_path):
    import validate_all_best_models_v68 as val
    for plan in ('v72_dual_branch','v72_cfd_optimized'):
        weights=tmp_path/'phase68_runs'/plan/'weights_v68'
        weights.mkdir(parents=True)
        torch.save({'epoch':1},weights/'latest_v68_nuig_2024.pth')
    models=val.scan_available_models(tmp_path,'nuig_2024',checkpoint_types=['latest'])
    modes={m['version']:m['state_update_mode'] for m in models}
    assert modes['v72_dual_branch']=='dual_branch'
    assert modes['v72_cfd_optimized']=='qssa_asymptotic'

def test_pd_pure_decay_is_continuous_across_stiff_threshold():
    for x in [79.99,80.01,100.]:
        z=torch.tensor([[np.log10(x/1e-4)]],requires_grad=True)
        _,log=kp.pd_v1_micro_log_update(torch.tensor([[-2.]]),torch.tensor([[-20.]]),z,torch.tensor([[1e-4]]))
        assert abs(float(log.detach())-(-2.-x/np.log(10.)))<1e-4
        assert torch.autograd.grad(log.sum(),z)[0].abs()>0

def test_large_raw_macro_does_not_corrupt_projected_ste(chemistry):
    tools,x,y,_,_=chemistry
    pred=y[:1].clone().requires_grad_()
    with torch.no_grad():pred[:,0]=(40.-tools.mean_Y[0])/tools.scale_Y[0]
    p=tools.predict_next_state(x[:1],pred,state_update_mode='dual_branch')
    old=torch.cat([p['Y_macro_old'],torch.pow(10.,p['logY_micro_old'].clamp(-40.,0.))],1)
    ppc.require_projection_valid(p['Y_next_constrained_full'],tools.inverse_state(x[:1])[:,tools.inert_idx].sum(1,keepdim=True),tools.A,old@tools.A.T)
    assert torch.isfinite(torch.autograd.grad(p['Y_next_constrained_full'].sum(),pred)[0]).all()
