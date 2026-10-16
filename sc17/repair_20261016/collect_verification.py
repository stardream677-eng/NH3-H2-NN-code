from pathlib import Path
import sys,json,hashlib,difflib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import experiment_manifest as em
import validate_sc_v68 as val

OUT=ROOT/'repair_20261016'
original=json.loads((OUT/'original_manifest.json').read_text(encoding='utf-8'))
changes=[];patch=[]
for relative,old_hash in original.items():
    before=OUT/'original'/relative;after=ROOT/relative
    assert hashlib.sha256(before.read_bytes()).hexdigest()==old_hash
    digest=hashlib.sha256(after.read_bytes()).hexdigest()
    if digest!=old_hash:
        changes.append(dict(file=relative,old_sha256=old_hash,new_sha256=digest))
        patch.extend(difflib.unified_diff(before.read_text(encoding='utf-8-sig').splitlines(True),
            after.read_text(encoding='utf-8-sig').splitlines(True),fromfile='a/'+relative,tofile='b/'+relative))
new_files=['numerical_contract.py','benchmark_cases.py','complete_validation.py','autonomous_integrator.py',
           'test/benchmark_autonomous_h128.py','test/benchmark_autonomous_cases.example.json']
for relative in new_files:
    target=ROOT/relative
    changes.append(dict(file=relative,new_sha256=em.compute_file_sha256(target),status='added'))
    patch.extend(difflib.unified_diff([],target.read_text(encoding='utf-8-sig').splitlines(True),
        fromfile='/dev/null',tofile='b/'+relative))
(OUT/'changes.patch').write_text(''.join(patch),encoding='utf-8')
(OUT/'changed_files.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')

run=OUT/'smoke_training_verified'
checkpoint=val.load_checkpoint(run/'weights_v68/latest_v68_nuig_2024.pth','nuig_2024')
assert checkpoint['manifest']['source_hashes']==em.compute_source_hashes()
history=json.loads((run/'history_v68/history_nuig_2024.json').read_text(encoding='utf-8'))
assert len(history)==1
assert not list((run/'weights_v68').glob('best*.pth'))
complete=json.loads((run/'complete_validation_epoch1.json').read_text(encoding='utf-8'))
assert complete['summary']['completed']==2 and not complete['is_valid_candidate']
window=json.loads((OUT/'window_validation_verified/metrics.json').read_text(encoding='utf-8'))
assert window['rollout_status']['pure_ar']['completed_all_windows']
assert window['rollout_failures'][0]['mode']=='pure_ar_raw'
full=json.loads((OUT/'complete_cli_verified/benchmark_report.json').read_text(encoding='utf-8'))
assert full['summary']['completed']==4 and full['summary']['failed']==0
assert full['selection_score'] is None and full['diagnostic_checkpoint']
record=dict(regression_tests=26,backup_files_verified=len(original),modified_original_files=sum('status' not in c for c in changes),
    runtime_source_hashes_match=True,mini_data_rows=4800,train_validation_test_trajectories=[24,3,3],
    smoke_epochs=1,smoke_active_training_horizon=1,smoke_evaluation_horizons=[1,8,32,128],
    smoke_training_loss=history[0]['train']['total'],best_files_generated=0,
    selection_complete_summary=complete['summary'],
    standalone_window_status=window['rollout_status'],standalone_raw_failures=window['rollout_failures'],
    standalone_complete_summary=full['summary'],
    standalone_complete_expected_exit=2,
    accuracy_qualified=False)
h8=OUT/'smoke_training_h8'
h8_history=json.loads((h8/'history_v68/history_nuig_2024.json').read_text(encoding='utf-8'))[0]
assert h8_history['horizon']==8 and h8_history['train']['rollout']>0
assert not list((h8/'weights_v68').glob('best*.pth'))
record['multistep_smoke']=dict(active_training_horizon=8,epochs=1,
    train_total=h8_history['train']['total'],train_rollout=h8_history['train']['rollout'],
    complete_summary=json.loads((h8/'complete_validation_epoch1.json').read_text(encoding='utf-8'))['summary'])
(OUT/'verification_summary.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
