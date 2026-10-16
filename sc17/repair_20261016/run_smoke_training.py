"""One epoch integration check; artifacts are not accuracy-qualified models."""
from pathlib import Path
import sys
import inspect
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import train_sc_v68 as tr
from train_all_v68 import PLAN_PRESETS

torch.set_num_threads(4)
active_horizon=int(sys.argv[1]) if len(sys.argv)>1 else 1
run_name='smoke_training_verified' if active_horizon==1 else f'smoke_training_h{active_horizon}'
config={k:v for k,v in PLAN_PRESETS['v72_dual_branch'].items()
        if k in inspect.signature(tr.train_model_v68).parameters}
config.update(data_dir=str(ROOT/'repair_20261016/mini_database'),
    split_dir=str(ROOT/'repair_20261016/mini_splits'),
    output_root=str(ROOT/'repair_20261016'/run_name),
    mechanism='nuig_2024',epochs=1,batch_size=8,val_batch_size=3,
    max_horizon=128,curriculum_horizons=[active_horizon],curriculum_start_epochs=[0],
    eval_horizons=[1,8,32,128],train_stride=128,val_stride=128,
    parts_per_epoch=1,val_parts_per_epoch=1,scaler_parts=1,scaler_sample_rate=1.,
    scaler_max_rows=4800,num_workers=0,max_val_batches=1,
    checkpoint_every=1,full_val_every=1,resume='none')
(ROOT/'repair_20261016'/f'{run_name}_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
tr.train_model_v68(**config)
