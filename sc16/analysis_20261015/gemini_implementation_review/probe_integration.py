"""Read-only integration review. Original training sources and datasets are not modified.

Outputs evidence.json beside this file. Temporary datasets/snapshots use TemporaryDirectory.
This probe deliberately reaches the new decoder through an old valid fixture to inspect
its numerical behavior independently of the separately demonstrated registration failure.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import contextlib
import io
import json
import sys
import tempfile
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import pandas as pd
import cantera as ct
import joblib
import train_sc_v68 as tr
import species_validation_protocol as svp
import training_diagnostics as td
import experiment_manifest as em
from data_pipeline import split_trajectories_v21 as split
from data_pipeline import build_training_dataset_v72 as builder

torch.set_num_threads(2)
OUT = {'python': sys.executable, 'versions': {'torch': torch.__version__, 'cantera': ct.__version__},
       'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [*ROOT.glob('*.py'), *(ROOT/'data_pipeline').glob('*.py')]}}

def probe(name, fn):
    try:
        OUT[name] = fn()
    except Exception as exc:
        OUT[name] = {'error': type(exc).__name__ + ': ' + str(exc)}

def parser_probe():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        try:
            tr.get_training_arg_parser().parse_args([
                '--mechanism', 'nuig_2024', '--output_parameterization', 'dual_branch_pd_v1',
                '--rate_decode_mode', 'dual_branch_pd_v1'])
            return {'accepted': True}
        except SystemExit as exc:
            return {'accepted': False, 'exit_code': exc.code, 'message': buf.getvalue().splitlines()[-1]}
probe('new_mode_cli', parser_probe)
probe('new_mode_target_transform', lambda: tr.transform_target_array(
    pd.DataFrame({'dYdt_H2': [0.], 'P_avg_H': [0.], 'D_coef_avg_H': [0.]}),
    ['H2'], ['H'], [1.], output_parameterization='dual_branch_pd_v1'))

def make_tools(mode='dual_branch', macro=None, micro=None):
    macro = ['H2'] if macro is None else macro
    micro = ['H'] if micro is None else micro
    fi = tr.build_feature_info(macro+micro, [], macro, micro, mode, mode)
    sx = tr.FeatureScaler(np.zeros(len(fi['input_cols']),np.float32), np.ones(len(fi['input_cols']),np.float32),1)
    sy = tr.FeatureScaler(np.zeros(len(fi['output_cols']),np.float32), np.ones(len(fi['output_cols']),np.float32),1)
    return tr.CoarseOperatorTools(fi,sx,sy,device='cpu')
probe('new_mode_tools_constructor', lambda: str(make_tools('dual_branch_pd_v1')))
probe('new_mode_mlp_constructor', lambda: str(tr.CoupledKineticsNet(5,3,output_parameterization='dual_branch_pd_v1')))

def cold_decoder_probe():
    op = make_tools()
    # Isolate the new decoder without modifying source or concealing constructor rejection.
    op.rate_decode_mode = 'dual_branch_pd_v1'
    results=[]
    for log_y in [-40.,-39.5,-39.,-38.,-30.]:
        x = torch.tensor([[1200.,40.,1.,log_y,-6.]], requires_grad=True)
        z = torch.tensor([[0.,-20.,-10.]], requires_grad=True)
        pred = op.predict_next_state(x,z,state_update_mode='legacy',safe_for_rollout=False)
        loss = pred['logY_micro_next_raw'].sum()
        loss.backward()
        results.append({'initial_logY':log_y,'raw_logY': pred['logY_micro_next_raw'].detach().tolist(),
                       'input_gradient': x.grad.tolist(), 'output_gradient':z.grad.tolist(),
                       'all_gradients_finite':bool(torch.isfinite(x.grad).all() & torch.isfinite(z.grad).all())})
    return results
probe('new_decoder_cold_float32_backward', cold_decoder_probe)

def grouping_probe():
    rows = [{'sample_id':f'traj_{g}_{t}','condition_group_id':f'g_{g}',
             'time':0.,'T':1000.+50*g,'P':1.+g,'phi_actual':1.,'Y_H2':0.05}
            for g in range(10) for t in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp); frame=pd.DataFrame(rows);frame.to_parquet(path/'part-00000.parquet',index=False)
        selected_columns=split._available_condition_columns(path/'part-00000.parquet')
        counts, meta=split._collect_trajectory_metadata(path)
        result=split._split_ids(np.asarray(sorted(counts),dtype=object),np.random.default_rng(42),0.6,0.2,metadata=meta)
        truth=frame.set_index('sample_id')['condition_group_id']
        groups={k:set(truth.loc[v]) for k,v in result.items()}
        overlap={a+'_'+b:sorted(groups[a]&groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
        return {'selected_columns':selected_columns,'true_groups':int(truth.nunique()),
                'loaded_groups':int(meta.condition_group_id.nunique()),'split_sizes':{k:len(v) for k,v in result.items()},
                'true_group_overlap':overlap}
probe('condition_group_ingestion_and_split',grouping_probe)

def metric_zero_probe():
    truth=np.array([[.9999,1e-4],[.9999,1e-4]])
    zero=np.array([[1.,0.],[1.,0.]])
    weak=np.array([[.99999,1e-5],[.99999,1e-5]])
    return {'zero_H':svp.compute_species_trajectory_metrics(truth,zero,['H2','H']),
            'tenfold_low_H':svp.compute_species_trajectory_metrics(truth,weak,['H2','H'])}
probe('metric_zero_radical',metric_zero_probe)
probe('metric_invalid_state_gate', lambda: svp.compute_species_trajectory_metrics(
    np.array([[.5,.5]]),np.array([[-.1,2.]]),['H2','O2'],element_matrix=np.eye(2)))

def mask_missing_probe():
    op=make_tools()
    mask=tr.compute_qssa_rate_mask(op,torch.tensor([[0.,-16.,8.]]))
    return {'mask': None if mask is None else mask.tolist(), 'should_not_mask_without_states':True}
probe('mask_missing_states',mask_missing_probe)
def qssa_mask_probe():
    op=make_tools('qssa')
    x=torch.tensor([[1200.,40.,.9999,-4.,-6.]])
    return tr.compute_qssa_rate_mask(op,torch.zeros(1,4),Y_old_unscaled=x,Y_next_unscaled=x).tolist()
probe('qssa_mask_full_input_layout',qssa_mask_probe)

def f64_then_clip_probe():
    model=torch.nn.Linear(2,1,bias=False)
    model.weight.grad=torch.full_like(model.weight,1e20)
    good,norm,diag=td.check_gradient_finite_and_norm(model)
    norm32=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
    return {'finite_check_passed':bool(good),'float64_norm':norm,'clip_returned_norm':float(norm32),
            'gradient_after_clip':model.weight.grad.tolist()}
probe('actual_gradient_clipping_after_diagnostic',f64_then_clip_probe)

def replay_rng_probe():
    torch.manual_seed(123)
    model=torch.nn.Sequential(torch.nn.Linear(8,8,bias=False),torch.nn.Dropout(.5))
    model.train();x=torch.ones(4,8)
    expected=model(x).detach()
    actual=[]
    def forward(m,bx,by):
        y=m(bx);actual.append(y.detach());return y.sum()
    with tempfile.TemporaryDirectory() as tmp:
        tracker=td.TrainingDiagnosticTracker(tmp)
        tracker.step_attempted()
        snap=tracker.record_failure_and_check_budget(model,x,error_msg='synthetic failure after stochastic forward')
        replay=td.replay_snapshot_with_anomaly_detection(snap,model,forward)
        payload=torch.load(snap,weights_only=False)
        return {'exact_replay':bool(torch.equal(expected,actual[0])),
                'max_abs_difference':float((expected-actual[0]).abs().max()),'snapshot_keys':sorted(payload),
                'snapshot_sample_ids':payload['sample_ids'],'snapshot_window_start':payload['window_start'],'replay':replay}
probe('failure_snapshot_rng_replay',replay_rng_probe)

def callsite_probe():
    functions={'compute_species_trajectory_metrics','consider_candidate','create_experiment_manifest',
               'verify_checkpoint_compatibility','audit_parquet_dataset','encode_pd_targets','get_mechanism_contract'}
    result={k:[] for k in functions}
    for path in list(ROOT.glob('*.py'))+list((ROOT/'data_pipeline').glob('*.py')):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                if name in result:result[name].append(f'{path.relative_to(ROOT)}:{node.lineno}')
    return result
probe('production_callsites',callsite_probe)

def missing_mechanism_probe():
    with tempfile.TemporaryDirectory() as tmp:
        A=tr.build_element_matrix(['NH3','H2O'],device='cpu',mechanism_path=Path(tmp)/'MISSING.yaml')
        return {'missing_explicit_path_accepted':True,'matrix':A.tolist()}
probe('missing_mechanism_silent_fallback',missing_mechanism_probe)

def mini_phi_probe():
    with tempfile.TemporaryDirectory() as tmp:
        df=builder.convert_mini_ds(tmp)
        ranges=df.groupby('sample_id').phi.agg(['min','max','first','last'])
        worst=(ranges['max']-ranges['min']).idxmax()
        return {'trajectories':len(ranges),'trajectories_phi_range_gt_0_1':int(((ranges['max']-ranges['min'])>.1).sum()),
                'worst_sample_id':int(worst),'worst_phi':ranges.loc[worst].to_dict(),
                'mechanism_present':(ROOT/'NUIG2024.yaml').is_file()}
probe('mini_phi_not_initial_condition',mini_phi_probe)

def manifest_probe():
    expected={'mechanism_sha256':'abc','species_order':['H2'],'output_parameterization':'dual_branch_pd_v1',
              'transform_signature':'new','scaler_hashes':{'X':'xhash','Y':'yhash'}}
    wrong={**expected,'transform_signature':'wrong','scaler_hashes':{}}
    good,errors=em.verify_checkpoint_compatibility({'manifest':wrong},expected)
    return {'wrong_transform_missing_scalers_accepted':good,'errors':errors}
probe('manifest_contract_missing_checks',manifest_probe)

def corrected_contracts_probe():
    op=make_tools()
    x=torch.tensor([[1200.,40.,1.,-40.,np.log10(1.625e-4)]],dtype=torch.float32)
    from enhanced_loss_and_weights import compute_pollutant_no_loss
    no_tools=SimpleNamespace(micro_species=[],macro_species=['NO'],macro_idx=[0],mean_X=torch.zeros(1),scale_X=torch.ones(1))
    pred=torch.tensor([[1e-3]],requires_grad=True)
    loss=compute_pollutant_no_loss(no_tools,{'Y_macro_next_raw':pred},torch.zeros(1,1));loss.backward()
    _,micro,_=tr.project_physical_state(torch.tensor([[1.]],dtype=torch.float64),torch.tensor([[1e-40]],dtype=torch.float64),
                    element_matrix=torch.tensor([[1.,1.]],dtype=torch.float64),target_elements=torch.tensor([[1.]],dtype=torch.float64))
    return {'actual_decoded_dt':float(op.get_current_dt(x)), 'NO_cold_loss':float(loss.detach()),'NO_cold_grad':float(pred.grad),
            'projected_dormant_Y':float(micro)}
probe('verified_improvements',corrected_contracts_probe)

def real_label_roundtrip_probe():
    ds=joblib.load(ROOT/'data_pipeline/mini_ds.pkl'); names=ds['names']; rows=ds['rows']
    Y=np.stack([r['Y'] for r in rows]);Yn=np.stack([r['Yn'] for r in rows])
    Pq=np.stack([r['Pq'] for r in rows]);Dq=np.stack([r['Dq'] for r in rows])
    dt=np.array([r['dt'] for r in rows])
    columns={'T':[r['T'] for r in rows],'P':[r['P'] for r in rows],'dt':dt}
    for i,s in enumerate(names):
        columns.update({f'Y_{s}':Y[:,i],f'Y_next_{s}':Yn[:,i],f'dYdt_{s}':(Yn[:,i]-Y[:,i])/dt,
                        f'P_avg_{s}':Pq[:,i],f'D_coef_avg_{s}':Dq[:,i]})
    frame=pd.DataFrame(columns)
    inert,macro,micro=tr.classify_species(names,include_no_in_macro=True)
    info=tr.build_feature_info(names,inert,macro,micro,'dual_branch','dual_branch')
    info['macro_rate_scale']=tr.compute_macro_rate_scale(frame[[f'dYdt_{s}' for s in macro]].to_numpy()).astype(np.float32)
    sx=tr.FeatureScaler(np.zeros(len(info['input_cols']),np.float32),np.ones(len(info['input_cols']),np.float32),1)
    sy=tr.FeatureScaler(np.zeros(len(info['output_cols']),np.float32),np.ones(len(info['output_cols']),np.float32),1)
    op=tr.CoarseOperatorTools(info,sx,sy,device='cpu')
    X=tr.transform_input_array(frame,inert,macro,micro)
    Z=tr.transform_target_array(frame,macro,micro,info['macro_rate_scale'],'dual_branch')
    outputs=[]
    for i in range(0,len(frame),256):
        with torch.no_grad():
            pred=op.predict_next_state(torch.from_numpy(X[i:i+256]),torch.from_numpy(Z[i:i+256]),state_update_mode='dual_branch')
        outputs.append(pred['Y_next_constrained_full'].numpy())
    true=Yn[:,[names.index(s) for s in macro+micro]]
    old=json.loads((ROOT/'analysis_20261015/results.json').read_text(encoding='utf-8'))
    import chemistry_contract as cc
    gas=ct.Solution(str(ROOT/'NUIG2024.yaml'))
    _,expected_A=cc.compute_element_mass_matrix(gas,macro+micro)
    return {'rows':len(frame),'mode':'dual_branch (existing mode, not new PD mode)',
            'max_absolute_projected_error':float(np.abs(np.concatenate(outputs)-true).max()),
            'previous_audit_max_absolute_projected_error':old['exact_label_roundtrip_fp32']['dual_branch']['max_abs_constrained_error'],
            'matrix_max_difference_vs_mechanism':float(np.abs(op.A.numpy()-expected_A).max()),
            'has_saved_element_matrix': 'element_matrix' in info,
            'has_saved_mechanism_hash':'mechanism_sha256' in info}
probe('actual_real_label_roundtrip',real_label_roundtrip_probe)

def strict_json_value(value):
    if isinstance(value,dict):return {k:strict_json_value(v) for k,v in value.items()}
    if isinstance(value,list):return [strict_json_value(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    return value
OUT=strict_json_value(OUT)
(Path(__file__).parent/'evidence.json').write_text(json.dumps(OUT,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(OUT,ensure_ascii=False,indent=2,allow_nan=False))
