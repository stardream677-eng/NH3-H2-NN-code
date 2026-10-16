from pathlib import Path
import importlib.util
import torch

ROOT=Path(__file__).resolve().parents[1]

def test_complete_runner_keeps_cold_identity_trajectory_physical(chemistry):
    path=ROOT/'test/benchmark_autonomous_h128.py'
    assert path.exists(),'Independent complete-trajectory runner is missing'
    spec=importlib.util.spec_from_file_location('complete_runner',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    tools,*_=chemistry
    tools._benchmark_state_update_mode='dual_branch'
    tools._benchmark_trust_region_fraction=0.
    tools._benchmark_max_delta_temperature=500.
    z=torch.full_like(tools.mean_Y,-45.);z[:tools.n_macro]=0.
    scaled=(z-tools.mean_Y)/tools.scale_Y
    class FrozenChemistry(torch.nn.Module):
        def forward(self,x):return scaled.expand(x.shape[0],-1)
    case=dict(name='cold_identity',T0_K=300.,P0_atm=1.,phi=1.,h2_fraction=.2,
              dt_s=1e-6,steps=128,reactor_type='IdealGasReactor')
    record,traces=runner.run_case(case,FrozenChemistry(),tools,ROOT/'NUIG2024.yaml')
    assert record['status']=='completed',record
    assert len(traces['nn_T_K'])==129
    assert record['metrics']['ignition_agreement']

def test_missing_ignition_cannot_win_complete_selection():
    import autonomous_benchmark_protocol as protocol
    assert hasattr(protocol,'selection_score'),'Complete selection has not been connected'
    bad=dict(status='completed',metrics={'ignition_agreement':False})
    assert protocol.selection_score([bad])==float('inf')


def test_long_rollout_does_not_accumulate_local_projection_error(chemistry):
    from complete_validation import runner_module
    tools,*_=chemistry
    tools._benchmark_state_update_mode='dual_branch'
    tools._benchmark_trust_region_fraction=0.
    tools._benchmark_max_delta_temperature=500.
    class MeanParameters(torch.nn.Module):
        def forward(self,x):return torch.zeros(len(x),len(tools.mean_Y),device=x.device)
    case=dict(name='conservation_stress',T0_K=1473.,P0_atm=14.,phi=1.11,h2_fraction=.16,
              dt_s=8e-6,steps=128,reactor_type='IdealGasReactor')
    record,_=runner_module().run_case(case,MeanParameters(),tools,ROOT/'NUIG2024.yaml')
    assert record['status']=='completed',record
    assert record['physics']['max_element_residual'] <= 1e-6
