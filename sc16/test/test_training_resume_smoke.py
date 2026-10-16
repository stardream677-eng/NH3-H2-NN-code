"""Bounded integration test: 64 rows, two CPU epochs, no production artifacts."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import train_sc_v68 as tr
import validate_sc_v68 as val
from data_pipeline import build_training_dataset_v72 as builder


@pytest.mark.filterwarnings('ignore:.*pin_memory.*')
def test_pd_training_resume_and_independent_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(tr,'DEVICE',torch.device('cpu'))
    threads=torch.get_num_threads(); torch.set_num_threads(2)
    try:
        data=tmp_path/'data'/'trajectories_nuig_2024'
        frame=builder.convert_mini_ds(data)
        ids=sorted(frame.sample_id.unique())[:4]
        small=frame[frame.sample_id.isin(ids)].groupby('sample_id',sort=False).head(16)
        small.to_parquet(data/'part-000001.parquet',index=False)
        (data/'metadata.json').write_text(json.dumps({'training_rows':len(small),'success':4,'failed':0,'parquet_parts':1}))
        split=tmp_path/'splits'; split.mkdir()
        for name,values in [('train',ids[:3]),('validation',ids[3:])]:
            pd.DataFrame({'sample_id':values}).to_csv(split/f'nuig_2024_{name}_sample_ids.csv',index=False)
        output=tmp_path/'output'
        kwargs=dict(data_dir=str(tmp_path/'data'),split_dir=str(split),output_root=str(output),
                    epochs=1,batch_size=8,val_batch_size=8,max_horizon=8,
                    curriculum_horizons=[1],curriculum_start_epochs=[0],eval_horizons=[1,8],
                    train_stride=8,val_stride=8,parts_per_epoch=1,val_parts_per_epoch=1,
                    scaler_parts=1,scaler_sample_rate=1.,scaler_max_rows=512,
                    full_val_every=1,checkpoint_every=1,backward_chunk_steps=1,num_workers=0,
                    output_parameterization='dual_branch_pd_v1',rate_decode_mode='dual_branch_pd_v1',
                    state_update_mode='dual_branch_pd_v1',resume='none',lr=1e-5)
        tr.train_model_v68(**kwargs)
        weights=output/tr.artifact_subdir('weights')
        latest=weights/'latest_v68_nuig_2024.pth'
        first=torch.load(latest,map_location='cpu',weights_only=False)
        assert first['epoch']==0
        assert first['transform_signature']==first['feature_info']['transform_signature']=='dual_branch_pd_v1'
        assert len(first['manifest']['mechanism_sha256'])==64
        assert first['manifest']['dataset_audit']['actual_total_rows']==64
        before=first['sentinel_tracker']['best_scores']
        kwargs.update(epochs=2,resume='auto')
        tr.train_model_v68(**kwargs)
        resumed=torch.load(latest,map_location='cpu',weights_only=False)
        assert resumed['epoch']==1 and len(resumed['history'])==2
        assert resumed['manifest']['sentinel_hash']==first['manifest']['sentinel_hash']
        assert all(resumed['sentinel_tracker']['best_scores'][h]<=s for h,s in before.items())
        selected=weights/'best_v68_net_nuig_2024.pth'
        assert selected.read_bytes()==(weights/'best_v68_net_nuig_2024_H8.pth').read_bytes()
        checkpoint=val.load_checkpoint(selected,'nuig_2024')
        assert checkpoint['sentinel_metrics']['is_valid_candidate']
        assert checkpoint['sentinel_metrics']['metric_scope'] == 'window_endpoints'
        assert checkpoint['sentinel_metrics']['trajectory_diagnostics']['NO']['n_windows'] > 0
        assert checkpoint['sentinel_metrics']['timing_metrics']['OH_peak_time_mae_s'] >= 0.
        sx,sy,fi=val.load_scalers(output,'nuig_2024',checkpoint['feature_info'],split_dir=split)
        model=val._build_checkpoint_model(checkpoint,fi,device=torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            assert torch.isfinite(model(torch.zeros(1,len(fi['input_cols'])))).all()
        # An incompatible resume must reject before doing another optimization step.
        resumed['manifest']['mechanism_sha256']='0'*64
        torch.save(resumed,latest)
        kwargs['epochs']=3
        with pytest.raises(RuntimeError,match='[Mm]echanism'):
            tr.train_model_v68(**kwargs)
    finally:
        torch.set_num_threads(threads)
