"""Read-only third review. Small training runs use temporary data/output directories.

Run from sc16 with the user-specified cantera Python. No original sources,
datasets, scalers, tests, or checkpoints are modified.
"""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CANTERA_DATA', str(ROOT))
import numpy as np
import pandas as pd
import torch
import cantera as ct
import joblib
import train_sc_v68 as tr
import species_validation_protocol as svp
from data_pipeline import build_training_dataset_v72 as builder

torch.set_num_threads(2)
OUT = {'python': sys.executable, 'python_version': platform.python_version(),
       'torch': torch.__version__, 'cantera': ct.__version__, 'device': str(tr.DEVICE)}
sources = [*ROOT.glob('*.py'), *(ROOT/'data_pipeline').glob('*.py')]
OUT['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
for p in sources:
    dest = HERE/'source_snapshot'/p.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dest)

def save():
    (HERE/'evidence.json').write_text(json.dumps(OUT, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def probe(name, fn):
    print('PROBE', name, flush=True)
    try:
        OUT[name] = fn()
    except Exception as exc:
        OUT[name] = {'error': type(exc).__name__ + ': ' + str(exc), 'traceback': traceback.format_exc()}
    save()

def make_tools(mode='dual_branch_pd_v1'):
    fi = tr.build_feature_info(['H2', 'H'], [], ['H2'], ['H'], mode, mode)
    sx = tr.FeatureScaler(np.zeros(5, np.float32), np.ones(5, np.float32), 1)
    sy = tr.FeatureScaler(np.zeros(3, np.float32), np.ones(3, np.float32), 1)
    return tr.CoarseOperatorTools(fi, sx, sy, device='cpu')

def mass_gate():
    true = np.full((1, 3), 1/3)
    pred = true - 0.04
    # Pure-element species: independent elemental errors are 0.04, but mass error is 0.12.
    return svp.compute_species_trajectory_metrics(true, pred, ['H2','N2','O2'], element_matrix=np.eye(3))
probe('mass_gate_12_percent_defect', mass_gate)

def mask_probe():
    answer = []
    for mode, z1 in [('dual_branch', -16.), ('dual_branch_pd_v1', -20.)]:
        op = make_tools(mode)
        target = torch.tensor([[0., z1, 4.]])
        old = torch.tensor([[1200., 40., .9999, -4., -4.]])
        new = old.clone()
        new[:,3] = -4. - 1/np.log(10.)
        missing = tr.compute_qssa_rate_mask(op, target)
        supplied = tr.compute_qssa_rate_mask(op, target, Y_old_unscaled=old, Y_next_unscaled=new)
        answer.append({'mode': mode, 'D': 1e4, 'dt': 1e-4, 'Y_old': 1e-4, 'Y_next': 1e-4*np.exp(-1),
                       'without_states': None if missing is None else missing.tolist(),
                       'with_states': None if supplied is None else supplied.tolist()})
    return answer
probe('active_decay_missing_states', mask_probe)

def cli_probe():
    args = ['--mechanism','nuig_2024','--output_parameterization','dual_branch_pd_v1',
            '--rate_decode_mode','dual_branch_pd_v1','--state_update_mode','dual_branch_pd_v1',
            '--max_horizon','128','--curriculum_warmup_epochs','5','--epochs','120',
            '--batch_size','512','--lr','1e-3','--weight_decay','1e-5','--induction_boost','6.0',
            '--lambda_radical','0.1','--lambda_no','0.1','--lambda_mass','1.0','--lambda_element','1.0',
            '--data_dir','/path/to/workstation/database_Transient_V21_QSSA']
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        try:
            tr.get_training_arg_parser().parse_args(args)
            return {'accepted': True}
        except SystemExit as exc:
            return {'accepted': False, 'exit_code': exc.code, 'last_line': buf.getvalue().splitlines()[-1]}
probe('gemini_recommended_command', cli_probe)

def callsites():
    names = {'verify_checkpoint_compatibility','create_experiment_manifest','audit_parquet_dataset',
             'compute_qssa_rate_mask','compute_species_trajectory_metrics'}
    result = {name: [] for name in names}
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Call):
                n = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
                if n in names:
                    result[n].append({'file': str(path.relative_to(ROOT)), 'line': node.lineno,
                                      'keywords': [kw.arg for kw in node.keywords]})
    return result
probe('production_callsites', callsites)

def smoke_training():
    result = {}
    with tempfile.TemporaryDirectory(prefix='sc16_review_') as temp:
        base = Path(temp)
        data = base/'data'/'trajectories_nuig_2024'
        output = base/'output'
        splits = base/'splits'
        splits.mkdir()
        with (HERE/'smoke_train.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            df = builder.convert_mini_ds(data)
            ids = sorted(df.sample_id.unique())[:4]
            small = df[df.sample_id.isin(ids)].groupby('sample_id', sort=False).head(16).copy()
            small.to_parquet(data/'part-000001.parquet', index=False)
            (data/'metadata.json').write_text(json.dumps({'training_rows':len(small),'success':4,'failed':0,'parquet_parts':1}), encoding='utf-8')
            for name, values in [('train', ids[:3]), ('validation', ids[3:])]:
                pd.DataFrame({'sample_id': values}).to_csv(splits/f'nuig_2024_{name}_sample_ids.csv', index=False)
            result['rows'] = len(small)
            result['trajectory_ids'] = [int(i) for i in ids]
            kwargs = dict(data_dir=str(base/'data'), split_dir=str(splits), output_root=str(output),
                          epochs=1, batch_size=8, val_batch_size=8, max_horizon=8,
                          curriculum_horizons=[1], curriculum_start_epochs=[0], eval_horizons=[1,8],
                          train_stride=8, val_stride=8, parts_per_epoch=1, val_parts_per_epoch=1,
                          scaler_parts=1, scaler_sample_rate=1., scaler_max_rows=512,
                          full_val_every=1, checkpoint_every=1, backward_chunk_steps=1, num_workers=0,
                          output_parameterization='dual_branch_pd_v1', rate_decode_mode='dual_branch_pd_v1',
                          state_update_mode='dual_branch_pd_v1', resume='none', lr=1e-5)
            result['kwargs'] = kwargs.copy()
            try:
                tr.train_model_v68(**kwargs)
                result['first_run'] = 'success'
            except Exception as exc:
                result['first_run'] = {'error': str(exc), 'traceback':traceback.format_exc()}
                traceback.print_exc()
            result['artifact_names'] = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
            fi_files = list(output.rglob('feature_info*.pkl'))
            if fi_files:
                fi = joblib.load(fi_files[0])
                result['feature_info_signature'] = fi.get('transform_signature')
            result['checkpoints'] = {}
            for p in output.rglob('*.pth'):
                ckpt = torch.load(p, map_location='cpu', weights_only=False)
                nested_manifest = ckpt.get('training_config',{}).get('manifest')
                result['checkpoints'][p.name] = {
                    'top_keys':list(ckpt), 'transform_signature':ckpt.get('transform_signature'),
                    'feature_info_signature':ckpt.get('feature_info',{}).get('transform_signature'),
                    'manifest':ckpt.get('manifest',nested_manifest),
                    'sentinel_metrics':ckpt.get('sentinel_metrics'),
                    'history':ckpt.get('history')}
            kwargs.update(epochs=2, resume='auto')
            try:
                tr.train_model_v68(**kwargs)
                result['resume'] = 'success'
            except Exception as exc:
                result['resume'] = {'error':type(exc).__name__+': '+str(exc)}
                traceback.print_exc()
    return result
probe('actual_new_mode_training_and_resume', smoke_training)
print(json.dumps({k:v for k,v in OUT.items() if k not in ('source_sha256','actual_new_mode_training_and_resume')}, indent=2, ensure_ascii=False))
print('Training result:', OUT.get('actual_new_mode_training_and_resume',{}).get('first_run'))
print('Resume result:', OUT.get('actual_new_mode_training_and_resume',{}).get('resume'))
