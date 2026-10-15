"""Independent read-only checks for unresolved review items and sentinel membership."""
from pathlib import Path
import sys
import json
import tempfile
import itertools
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import train_sc_v68 as tr
import training_diagnostics as td
from test_followup_regressions import inert_validation_metrics, tools_for

torch.set_num_threads(2)
out = {}
files = [f'part-{i:05d}.parquet' for i in range(12)]
out['validation_file_membership_default_shuffle'] = {
    str(epoch):tr.select_epoch_files(files,9,tr.SEED,epoch,shuffle=True) for epoch in [0,1,2]}
out['perfect_prediction_with_inert'] = inert_validation_metrics()
grid = list(itertools.product([-40.,-39.5,-39.,-38.,-30.,-20.,-6.],[-20.,-19.,-6.],[-10.,0.,4.,10.],[1e-8,1e-4]))
op = tools_for()
x = torch.tensor([[1200.,40.,1.,ly,np.log10(dt)] for ly,zp,zd,dt in grid], dtype=torch.float32, requires_grad=True)
z = torch.tensor([[0.,zp,zd] for ly,zp,zd,dt in grid], dtype=torch.float32, requires_grad=True)
pred = op.predict_next_state(x,z,state_update_mode='dual_branch_pd_v1',safe_for_rollout=False)
pred['logY_micro_next_raw'].sum().backward()
bad = ~(torch.isfinite(x.grad).all(1) & torch.isfinite(z.grad).all(1))
out['float32_full_decoder_gradient_sweep'] = {'cases':len(grid),'nonfinite_gradient_cases':int(bad.sum()),
    'bad_inputs':[grid[i] for i in torch.nonzero(bad).flatten().tolist()]}
model = torch.nn.Linear(2,1,bias=False)
model.weight.grad = torch.full_like(model.weight,1e20)
good,norm,diag = td.check_gradient_finite_and_norm(model)
clipped = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
out['finite_gradient_clipping'] = {'finite_check_passed':bool(good),'float64_norm':norm,
    'clipping_norm':str(float(clipped)),'after_clip':model.weight.grad.tolist()}

torch.manual_seed(123)
model = torch.nn.Sequential(torch.nn.Linear(8,8,bias=False),torch.nn.Dropout(.5))
model.train()
x = torch.ones(4,8)
expected = model(x).detach()
actual = []
def forward(m,bx,by):
    y = m(bx)
    actual.append(y.detach())
    return y.sum()
with tempfile.TemporaryDirectory() as tmp:
    tracker = td.TrainingDiagnosticTracker(tmp)
    tracker.step_attempted()
    snap = tracker.record_failure_and_check_budget(model,x,error_msg='synthetic failure after stochastic forward')
    replay = td.replay_snapshot_with_anomaly_detection(snap,model,forward)
    out['failure_snapshot_rng_replay'] = {'exact_replay':bool(torch.equal(expected,actual[0])),
        'max_absolute_difference':float((expected-actual[0]).abs().max()),'replay':replay}
    A = tr.build_element_matrix(['NH3','H2O'],device='cpu',mechanism_path=Path(tmp)/'MISSING.yaml')
    out['missing_explicit_mechanism_path_accepted'] = {'matrix':A.tolist()}

old = json.loads((ROOT/'analysis_20261015/gemini_implementation_review/evidence.json').read_text(encoding='utf8'))
new = json.loads((HERE/'evidence.json').read_text(encoding='utf8'))
out['unchanged_since_previous_review'] = [k for k,v in new['source_sha256'].items() if old['source_sha256'].get(k)==v]
(HERE/'supplement_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
