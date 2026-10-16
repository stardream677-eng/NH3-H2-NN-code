from pathlib import Path
import importlib.util
import cantera as ct
import torch
import train_sc_v68 as tr
import physical_projection_contract as ppc

ROOT=Path(__file__).resolve().parents[1]

def test_retry_reevaluates_model_at_half_dt_and_preserves_outer_dt(chemistry):
    path=ROOT/'autonomous_integrator.py'
    assert path.exists(),'Bounded adaptive model re-evaluation is missing'
    spec=importlib.util.spec_from_file_location('adaptive',path)
    adaptive=importlib.util.module_from_spec(spec);spec.loader.exec_module(adaptive)
    from complete_validation import runner_module
    runner=runner_module();tools,*_=chemistry
    gas=ct.Solution(str(ROOT/'NUIG2024.yaml'));gas.TPX=300.,ct.one_atm,runner.fuel_air_mole_fractions(1.,.2)
    x=runner.physical_input(tools,gas,300.,1.,gas.Y,1.6e-5)
    names=tools.feature_info['inert_species']+tools.macro_species+tools.micro_species
    closure=tr.build_temperature_closure({'mechanism_path':str(ROOT/'NUIG2024.yaml'),'reactor_type':'IdealGasReactor'},names)
    z=torch.full_like(tools.mean_Y,-45.);z[:tools.n_macro]=0.
    calls=[]
    class StepLimitedModel(torch.nn.Module):
        def forward(self,state):
            dt=float(tools.get_current_dt(state)[0,0]);calls.append(dt)
            if dt>8.01e-6:raise ppc.ProjectionConstraintError('not_converged',[0],'step too large')
            return ((z-tools.mean_Y)/tools.scale_Y).expand(len(state),-1)
    nxt,info=adaptive.advance_autonomous(StepLimitedModel(),tools,x,closure,'dual_branch')
    assert info['accepted_substeps']==2 and info['model_evaluations']==3
    assert calls[0]>1.5e-5 and all(v<8.01e-6 for v in calls[1:])
    assert abs(float(tools.get_current_dt(nxt)[0,0])-1.6e-5)<1e-10


def test_nested_subdivision_advances_the_entire_prescribed_interval():
    import autonomous_integrator as ai
    class ToyTools:
        dt_idx=0
        scale_X=torch.ones(2,dtype=torch.float64)
        def predict_next_state(self,state,predicted,**kwargs):
            return dict(Y_macro_next_constrained=predicted,logY_micro_next_constrained=predicted)
        def state_to_input_scaled(self,state,macro,micro,**kwargs):
            out=state.clone();out[:,1]=macro[:,0];return out
    class StepLimited(torch.nn.Module):
        def forward(self,state):
            dt=10.**state[:,:1]
            if (dt>4.01e-6).any():raise ppc.ProjectionConstraintError('not_converged',[0],'step limit')
            return state[:,1:2]+dt
    state=torch.tensor([[__import__('math').log10(1.6e-5),0.]],dtype=torch.float64)
    out,info=ai.advance_autonomous(StepLimited(),ToyTools(),state,{},'dual_branch')
    assert info['accepted_substeps']==4 and info['model_evaluations']==7
    assert abs(float(out[0,1])-1.6e-5)<1e-16
    assert out[0,0]==state[0,0]
