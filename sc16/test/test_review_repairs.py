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


def artifact_fixture(tmp_path):
    import chemistry_contract as cc
    fi=tr.build_feature_info(['H2','H'],[],['H2'],['H'],'dual_branch_pd_v1','dual_branch_pd_v1')
    sx=tr.FeatureScaler(np.zeros(5,dtype=np.float32),np.ones(5,dtype=np.float32),1)
    sy=tr.FeatureScaler(np.zeros(3,dtype=np.float32),np.ones(3,dtype=np.float32),1)
    mech=tmp_path/'NUIG2024.yaml'
    mech.write_bytes((Path(tr.__file__).parent/'NUIG2024.yaml').read_bytes())
    fi.update(mechanism='nuig_2024',mechanism_path=str(mech))
    contract=cc.get_mechanism_contract(mech,fi['all_species'],device='cpu')
    manifest=em.create_experiment_manifest(contract,'dual_branch_pd_v1','dual_branch_pd_v1',
        {'validation':'finite_positive','clipping':False},{'floor':1e-40},svp.PROTOCOL_ID,
        scaler_hashes=em.hash_scalers(sx,sy))
    fi['manifest']=manifest
    checkpoint=dict(model_state_dict={},feature_info=fi,version=tr.VERSION,mechanism='nuig_2024',
        transform_signature='dual_branch_pd_v1',rollout_signature=tr.ROLLOUT_SIGNATURE,manifest=manifest)
    return checkpoint,sx,sy


def test_real_checkpoint_loader_rejects_inconsistent_transform(tmp_path):
    import validate_sc_v68 as val
    ckpt,_,_=artifact_fixture(tmp_path)
    ckpt['transform_signature']=tr.LEGACY_TRANSFORM_SIGNATURE
    p=tmp_path/'weights.pth'; torch.save(ckpt,p)
    with pytest.raises(ValueError,match='[Tt]ransform'):
        val.load_checkpoint(p,'nuig_2024')


def test_real_checkpoint_loader_detects_changed_mechanism_bytes(tmp_path):
    import validate_sc_v68 as val
    ckpt,_,_=artifact_fixture(tmp_path)
    p=tmp_path/'weights.pth'; torch.save(ckpt,p)
    mech=Path(ckpt['feature_info']['mechanism_path'])
    mech.write_bytes(mech.read_bytes()+b'\n# changed mechanism file\n')
    with pytest.raises(ValueError,match='[Mm]echanism'):
        val.load_checkpoint(p,'nuig_2024')


def test_real_scaler_loader_rejects_changed_statistics(tmp_path):
    import joblib
    import validate_sc_v68 as val
    ckpt,sx,sy=artifact_fixture(tmp_path)
    scaler_dir=tmp_path/tr.artifact_subdir('scalers'); scaler_dir.mkdir()
    fi=ckpt['feature_info']; sig=fi['transform_signature']
    sx.mean_[0]+=1.
    for name,value in [('scaler_X',sx),('scaler_Y',sy),('feature_info',fi)]:
        joblib.dump(value,scaler_dir/f'{name}_nuig_2024_{sig}.pkl')
    with pytest.raises(ValueError,match='[Ss]caler hash'):
        val.load_scalers(tmp_path,'nuig_2024',fi)


def test_trajectory_diagnostics_use_each_window_physical_time():
    acc = svp.TrajectoryDiagnosticAccumulator(['OH', 'NO'])
    true = np.array([[0., 2., 1.], [0., 3., 1.]])
    pred = np.array([[0., 1., 2.], [0., 1., 3.]])
    times = np.array([[0., 1., 4.], [0., 2., 8.]]) * 1e-6
    acc.update(np.stack([true, true * 1e-3], axis=-1),
               np.stack([pred, pred * 1e-3], axis=-1), times)
    metrics = acc.finalize()
    assert metrics['OH']['n_windows'] == 2
    assert metrics['OH']['peak_time_mae_s'] == pytest.approx(4.5e-6)
    assert metrics['OH']['peak_amplitude_mae'] == 0.
    assert metrics['NO']['endpoint_mae'] == pytest.approx(1.5e-3)


def test_full_input_log_zero_means_unit_concentration_not_dormancy():
    class Tools:
        rate_decode_mode = 'dual_branch_pd_v1'
        macro_species = ['H2']
        micro_species = ['H']
        macro_idx = [0]
        micro_idx = [1]
        mean_Y = torch.zeros(3)
        scale_Y = torch.ones(3)
    state = torch.tensor([[0., 0.]])  # log10(Y_H) = 0 => Y_H = 1
    assert tr.compute_qssa_rate_mask(Tools(), torch.zeros(1, 3), state, state) is None


def test_rebuilt_scalers_must_match_checkpoint_before_being_written(tmp_path, monkeypatch):
    import validate_sc_v68 as val
    ckpt, sx, sy = artifact_fixture(tmp_path)
    # Isolate the rebuild path. Only the expensive scaler estimation is stubbed;
    # the production loader, actual mechanism, hash checks and writes are real.
    sx.mean_[0] += 1.
    monkeypatch.setattr(val, 'SCRIPT_DIR', tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tr, 'get_schema', lambda *args: (None, None, None, []))
    monkeypatch.setattr(tr, 'compute_streaming_scalers', lambda *args, **kwargs: (sx, sy))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='[Ss]caler hash'):
        val.load_scalers(output, 'nuig_2024', ckpt['feature_info'],
                         data_dir=tmp_path, rebuild_scalers=True)
    assert not (output / tr.artifact_subdir('scalers')).exists()


def test_tools_reject_saved_element_matrix_that_disagrees_with_mechanism(tmp_path):
    ckpt, sx, sy = artifact_fixture(tmp_path)
    fi = ckpt['feature_info']
    fi['element_matrix'] = [[0.9, 1.]]
    with pytest.raises(ValueError, match='[Ee]lement matrix'):
        tr.CoarseOperatorTools(fi, sx, sy, device='cpu', rate_decode_mode='dual_branch_pd_v1')


def test_snapshot_context_replays_after_the_saved_tail_draw(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.5))
    tracker = td.TrainingDiagnosticTracker(tmp_path)
    tracker.step_attempted(model)
    tail = torch.randint(1, 4, (2,))
    context = {'tail_lengths': tail, 'rng_before_forward': td.capture_rng_state()}
    x = torch.ones(2, 4)
    expected = model(x).detach()
    path = tracker.record_failure_and_check_budget(model, x, replay_context=context)
    seen = []
    def forward(m, bx, by, saved_context):
        assert torch.equal(saved_context['tail_lengths'], tail)
        result = m(bx); seen.append(result.detach()); return result.sum()
    result = td.replay_snapshot_with_anomaly_detection(path, model, forward, use_saved_context=True)
    assert not result['anomaly_detected']
    assert torch.equal(expected, seen[0])
