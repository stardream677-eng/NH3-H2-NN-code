"""Review-only regressions: assert intended contracts, so current broken paths fail.

These tests are deliberately outside test/; no training source is edited.
Run this file explicitly after reading review.md.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import train_sc_v68 as tr
import species_validation_protocol as svp
import experiment_manifest as em
from data_pipeline import split_trajectories_v21 as splitter

def make_tools(mode='dual_branch'):
    fi=tr.build_feature_info(['H2','H'],[],['H2'],['H'],mode,mode)
    sx=tr.FeatureScaler(np.zeros(5,np.float32),np.ones(5,np.float32),1)
    n=len(fi['output_cols'])
    sy=tr.FeatureScaler(np.zeros(n,np.float32),np.ones(n,np.float32),1)
    return tr.CoarseOperatorTools(fi,sx,sy,device='cpu')

def test_reported_training_cli_accepts_new_mode():
    args=tr.get_training_arg_parser().parse_args(['--mechanism','nuig_2024',
        '--output_parameterization','dual_branch_pd_v1','--rate_decode_mode','dual_branch_pd_v1'])
    assert args.output_parameterization=='dual_branch_pd_v1'

def test_real_target_pipeline_encodes_new_mode():
    df=pd.DataFrame({'dYdt_H2':[0.], 'P_avg_H':[1e-6], 'D_coef_avg_H':[1e3]})
    result=tr.transform_target_array(df,['H2'],['H'],[1.],output_parameterization='dual_branch_pd_v1')
    assert result.shape==(1,3)

def test_real_tools_accept_new_feature_signature():
    assert make_tools('dual_branch_pd_v1').rate_decode_mode=='dual_branch_pd_v1'

def test_new_decoder_float32_cold_log_gradient_is_finite():
    op=make_tools()
    # Directly test the new numerical branch, independently of mode-registration failures.
    x=torch.tensor([[1200.,40.,1.,-39.,-6.]],requires_grad=True)
    z=torch.tensor([[0.,-20.,-10.]],requires_grad=True)
    pred=op.predict_next_state(x,z,state_update_mode='dual_branch_pd_v1',safe_for_rollout=False)
    pred['logY_micro_next_raw'].sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(z.grad).all()

def test_original_condition_groups_do_not_cross_splits(tmp_path):
    df=pd.DataFrame([{'sample_id':f'traj_{g}_{t}','condition_group_id':f'g_{g}',
                     'time':0.,'T':1000.+50*g,'P':1.+g,'phi_actual':1.,'Y_H2':.05}
                    for g in range(10) for t in range(3)])
    df.to_parquet(tmp_path/'part-00000.parquet',index=False)
    counts,metadata=splitter._collect_trajectory_metadata(tmp_path)
    parts=splitter._split_ids(np.asarray(sorted(counts),dtype=object),np.random.default_rng(42),.6,.2,metadata=metadata)
    # Use original group IDs as the independent oracle, not already corrupted metadata.
    truth=df.set_index('sample_id').condition_group_id
    groups={name:set(truth.loc[ids]) for name,ids in parts.items()}
    assert groups['train'].isdisjoint(groups['validation'])
    assert groups['train'].isdisjoint(groups['test'])
    assert groups['validation'].isdisjoint(groups['test'])

def test_zero_prediction_of_active_radical_is_not_rewarded():
    true=np.array([[.9999,1e-4]])
    zero=svp.compute_species_trajectory_metrics(true,np.array([[1.,0.]]),['H2','H'])
    low=svp.compute_species_trajectory_metrics(true,np.array([[.99999,1e-5]]),['H2','H'])
    assert zero['mean_radical_log_rmse']>low['mean_radical_log_rmse']

def test_invalid_physical_state_is_not_valid_candidate():
    result=svp.compute_species_trajectory_metrics(np.array([[.5,.5]]),np.array([[-.1,2.]]),
                                                  ['H2','O2'],element_matrix=np.eye(2))
    assert result['is_valid_candidate'] is False

def test_missing_states_do_not_mask_destruction():
    mask=tr.compute_qssa_rate_mask(make_tools(),torch.tensor([[0.,-16.,8.]]))
    assert mask is None or bool((mask==1.).all())

def test_qssa_mask_accepts_actual_full_input_layout():
    op=make_tools('qssa')
    x=torch.tensor([[1200.,40.,.9999,-4.,-6.]])
    mask=tr.compute_qssa_rate_mask(op,torch.zeros(1,4),Y_old_unscaled=x,Y_next_unscaled=x)
    assert mask is None or mask.shape==(1,4)

def test_manifest_rejects_wrong_transform_and_missing_scaler_hashes():
    expected={'mechanism_sha256':'abc','species_order':['H2'],'output_parameterization':'dual_branch_pd_v1',
              'transform_signature':'new','scaler_hashes':{'X':'xhash','Y':'yhash'}}
    wrong={**expected,'transform_signature':'wrong','scaler_hashes':{}}
    compatible,errors=em.verify_checkpoint_compatibility({'manifest':wrong},expected,strict=True)
    assert not compatible and errors
