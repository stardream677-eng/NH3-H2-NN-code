"""Regression contracts from the third review; all fixtures are small and local."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import train_sc_v68 as tr
import species_validation_protocol as svp
import training_diagnostics as td
import experiment_manifest as em


def tiny_dataset(tmp_path, **kwargs):
    fi = tr.build_feature_info(['H2','H'], [], ['H2'], ['H'], 'dual_branch_pd_v1','dual_branch_pd_v1')
    sx = tr.FeatureScaler(np.zeros(5), np.ones(5), 1)
    sy = tr.FeatureScaler(np.zeros(3), np.ones(3), 1)
    files = []
    for n in range(3):
        frame = pd.DataFrame([dict(sample_id=n, time=i*1e-6, T=1000.+n*100+i,
            P=1., dt=1e-6, T_next=1001.+n*100+i, P_next=1., Y_H2=1., Y_H=1e-20,
            Y_next_H2=1., Y_next_H=1e-20, dYdt_H2=0., P_avg_H=0.,D_coef_avg_H=0.) for i in range(4)])
        p=tmp_path/f'part-{n:05d}.parquet'; frame.to_parquet(p,index=False); files.append(p)
    return tr.CoarseOperatorDataset(files,fi,sx,sy,max_horizon=1,stride=1,
                                   parts_per_epoch=2,shuffle_parts=False,**kwargs)


def test_fixed_validation_preserves_even_a_truncated_prefix_across_epochs(tmp_path):
    dataset=tiny_dataset(tmp_path)
    dataset.set_epoch(0); a=list(dataset)[:3]
    dataset.set_epoch(1); b=list(dataset)[:3]
    assert all(torch.equal(x[0],y[0]) for x,y in zip(a,b))


def test_explicit_missing_mechanism_cannot_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        tr.build_element_matrix(['NH3','H2O'],device='cpu',mechanism_path=tmp_path/'MISSING.yaml')


def test_stable_clipping_preserves_finite_large_gradient_direction():
    model=torch.nn.Linear(2,1,bias=False)
    model.weight.grad=torch.full_like(model.weight,1e20)
    td.clip_gradients_stable(model, max_norm=1.)
    assert torch.allclose(model.weight.grad,torch.full_like(model.weight,.5**.5),atol=1e-6)


def test_snapshot_replays_the_failed_stochastic_forward_exactly(tmp_path):
    torch.manual_seed(123)
    model=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.Dropout(.5))
    model.train(); x=torch.ones(4,8)
    tracker=td.TrainingDiagnosticTracker(tmp_path)
    tracker.step_attempted(model=model)
    expected=model(x).detach()
    snap=tracker.record_failure_and_check_budget(model,x,error_msg='after forward')
    model.eval()  # Replay must restore the captured training mode as well.
    seen=[]
    def forward(m,bx,by):
        y=m(bx); seen.append(y.detach()); return y.sum()
    td.replay_snapshot_with_anomaly_detection(snap,model,forward)
    assert torch.equal(expected,seen[0])


def test_sentinel_restores_best_and_rejects_a_worse_resume_candidate():
    first=svp.HorizonSentinelTracker([1,8])
    assert first.consider_candidate(8,{'is_valid_candidate':True,'composite_score':.1},3)
    restored=svp.HorizonSentinelTracker([1,8])
    restored.load_state_dict(first.as_dict())
    assert not restored.consider_candidate(8,{'is_valid_candidate':True,'composite_score':.2},4)
    assert restored.best_epochs[8]==3


def test_manifest_rejects_a_changed_validation_population():
    manifest=dict(mechanism_sha256='a'*64,species_order=['H2'],output_parameterization='dual_branch_pd_v1',
                  transform_signature='dual_branch_pd_v1',scaler_hashes={'X':'x'},split_hash='group_a',
                  evaluation_protocol_id=svp.PROTOCOL_ID,sentinel_hash='windows_a')
    old={**manifest,'sentinel_hash':'windows_b'}
    good,errors=em.verify_checkpoint_compatibility({'manifest':old},manifest,strict=True)
    assert not good and errors


def test_manifest_missing_mechanism_hash_has_a_clear_rejection():
    expected=dict(mechanism_sha256='a'*64,species_order=['H2'])
    good,errors=em.verify_checkpoint_compatibility({'manifest':{'species_order':['H2']}},expected,strict=True)
    assert not good and any('mechanism' in s.lower() for s in errors)


def test_initial_phi_does_not_change_with_burning_composition(tmp_path):
    from data_pipeline import build_training_dataset_v72 as builder
    frame=builder.convert_mini_ds(tmp_path)
    assert (frame.groupby('sample_id').phi.nunique()==1).all()
    assert (frame.groupby('sample_id').phi_actual.nunique()==1).all()


def test_dataset_diagnostic_metadata_identifies_the_window(tmp_path):
    dataset=tiny_dataset(tmp_path,include_metadata=True)
    x,y,meta=next(iter(dataset))
    assert meta['sample_id']=='0'
    assert meta['window_start']==0
    assert meta['part'].endswith('part-00000.parquet')
