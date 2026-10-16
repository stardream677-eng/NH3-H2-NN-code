"""Read-only source/entry-point audit; only writes this diagnosis folder."""
from pathlib import Path
import csv, hashlib, json, subprocess, sys

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
VERSIONS=['sc15','sc16','sc17']
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

files=sorted(set().union(*[
    {str(p.relative_to(BASE/v)).replace('\\','/') for p in (BASE/v).rglob('*')
     if p.is_file() and (p.suffix in ['.py','.yaml'] or p.name=='mini_ds.pkl')
     and not any(part.startswith('diagnosis_') or part in ['__pycache__','.pytest_cache'] for part in p.relative_to(BASE/v).parts)}
    for v in VERSIONS]))
table=[]
for name in files:
    vals={v:digest(BASE/v/name) for v in VERSIONS}
    table.append({'file':name,**vals,'sc16_vs_sc17':'identical' if vals['sc16']==vals['sc17'] else ('new' if vals['sc16'] is None else 'changed')})
with (OUT/'source_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['file',*VERSIONS,'sc16_vs_sc17'])
    writer.writeheader();writer.writerows(table)

entry=[]
for script,args in [
    ('data_pipeline/build_training_dataset_v72.py',['--help']),
    ('validate_all_best_models_v68.py',['--plan','v72_dual_branch']),
]:
    p=subprocess.run([sys.executable,str(BASE/'sc17'/script),*args],cwd=BASE/'sc17',
                     capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=40)
    entry.append(dict(script=script,args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))

integrity={}
for v in VERSIONS:
    before=json.loads((OUT/(v+'_probes.json')).read_text(encoding='utf-8'))['source_hashes']
    changes=[]
    for name,old in before.items():
        if any(part.startswith('diagnosis_') for part in Path(name).parts):continue
        if digest(BASE/v/name)!=old:changes.append(name)
    integrity[v]=dict(original_python_changes_since_probe_start=changes)

result=dict(source_comparison=table,entry_probes=entry,integrity=integrity,
            sc17_checkpoint_files=[str(p.relative_to(BASE/'sc17')) for p in (BASE/'sc17').rglob('*')
                                  if p.suffix in ['.pt','.pth','.ckpt'] and not any(x.startswith('diagnosis_') for x in p.parts)])
(OUT/'source_and_entry_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(sc16_vs_sc17={k:[r['file'] for r in table if r['sc16_vs_sc17']==k] for k in ['changed','new']},
                     entry_probes=entry,integrity=integrity,sc17_checkpoint_files=result['sc17_checkpoint_files']),ensure_ascii=False,indent=2))
