"""Read-only numerical audit of sc16. Writes only this directory's results.json.

Run with the cantera environment. No training or original-data mutation.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import json
import hashlib
import torch
import numpy as np
import pandas as pd
import cantera as ct
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_sc_v68 as tr
from enhanced_loss_and_weights import compute_pollutant_no_loss, MaskedOnlineMoments
from thermo_nasa7_layer import ThermoNASA7Layer

torch.set_num_threads(2)
OUT = {"source_root": str(ROOT), "python": sys.executable,
       "versions": {"torch": torch.__version__, "cantera": ct.__version__},
       "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in list(ROOT.glob('*.py')) + list((ROOT/'data_pipeline').glob('*.py'))}}

# 1. The same continuous head changes its physical meaning at the D sentinel.
z2 = torch.tensor([[-40.0001], [-39.9999]], dtype=torch.float64)
log_next = tr.dual_branch_micro_log_update(
    torch.full_like(z2, -40.), torch.full_like(z2, -5.), z2, torch.full_like(z2, 1e-6))
OUT['sentinel_boundary'] = {'z2': z2.flatten().tolist(),
                           'log10_Y_next': log_next.flatten().tolist(),
                           'jump_dex': float(abs(log_next[0]-log_next[1]))}

# 2. Low equilibrium concentration does not imply negligible destruction.
mask_tools = SimpleNamespace(rate_decode_mode='dual_branch', macro_species=['H2'],
                             mean_Y=torch.zeros(3), scale_Y=torch.ones(3))
targets = torch.tensor([[0., -16., 8.]])
OUT['destruction_mask'] = {'target_macro_z_logYqssa_logD': targets.tolist(),
    'mask': tr.compute_qssa_rate_mask(mask_tools, targets).tolist(),
    'Y_old': 1e-4, 'dt': 1e-6,
    'Y_next_D_1e8': float(10**tr.dual_branch_micro_log_update(
        torch.tensor([[-4.]], dtype=torch.float64), torch.tensor([[-16.]], dtype=torch.float64),
        torch.tensor([[8.]], dtype=torch.float64), torch.tensor([[1e-6]], dtype=torch.float64))),
    'Y_next_D_1': float(10**tr.dual_branch_micro_log_update(
        torch.tensor([[-4.]], dtype=torch.float64), torch.tensor([[-16.]], dtype=torch.float64),
        torch.tensor([[0.]], dtype=torch.float64), torch.tensor([[1e-6]], dtype=torch.float64)))}

# 3. NO's cold false-positive penalty is bypassed when the whole batch is cold.
no_tools = SimpleNamespace(micro_species=[], macro_species=['NO'], macro_idx=[0],
                           mean_X=torch.zeros(1), scale_X=torch.ones(1))
pred = {'Y_macro_next_raw': torch.tensor([[1e-3]], requires_grad=True)}
loss = compute_pollutant_no_loss(no_tools, pred, torch.zeros((1, 1)))
OUT['no_cold_batch'] = {'true_Y_NO': 0., 'pred_Y_NO': 1e-3,
                      'dedicated_NO_loss': float(loss), 'has_gradient': loss.requires_grad}

# 4. Valid nearly constant channels can collapse the masked standard deviation.
mom = MaskedOnlineMoments(1)
mom.update(np.array([[-5.], [-5.], [-5.], [-45.]]))
mean, std, count = mom.get_stats()
OUT['masked_scaler_constant_active'] = {'mean': mean.tolist(), 'std': std.tolist(),
    'n_active': count.tolist(), 'standardized_sentinel': ((-45-mean)/std).tolist()}

# 5. dt transformation silently clips intervals from augmentation.
frame = pd.DataFrame({'T': [1200.], 'P': [40.], 'dt': [1.5e-4], 'Y_H2': [1.]})
transformed = tr.transform_input_array(frame, [], ['H2'], [])
OUT['dt_clipping'] = {'actual_dt': float(frame.dt.iloc[0]),
                     'encoded_dt': float(10**transformed[0, -1]),
                     'augmented_possible_dt_max': .020/160*1.3}

# 6. Verify pressure behavior and thermal clipping using real mechanism thermodynamics.
mech = ROOT.parent/'sc15'/'NUIG2024.yaml'
OUT['mechanism_path'] = str(mech)
gas = ct.Solution(str(mech))
names = gas.species_names
inert, macro, micro = tr.classify_species(names, include_no_in_macro=True)
fi = tr.build_feature_info(names, inert, macro, micro, 'dual_branch', 'dual_branch')
sx = tr.FeatureScaler(np.zeros(len(fi['input_cols']), dtype=np.float32), np.ones(len(fi['input_cols']), dtype=np.float32), 1)
sy = tr.FeatureScaler(np.zeros(len(fi['output_cols']), dtype=np.float32), np.ones(len(fi['output_cols']), dtype=np.float32), 1)
op = tr.CoarseOperatorTools(fi, sx, sy, device='cpu')
order = inert+macro+micro
closure = tr.build_temperature_closure({'mechanism_path': str(mech), 'reactor_type': 'IdealGasReactor'}, order)
gas.TPX = 1200., 40*ct.one_atm, {'NH3': .8, 'H2': .2, 'O2': .7, 'N2': .7*3.76}
y0 = gas.Y.copy(); u0 = gas.int_energy_mass
row = {'T': 1200., 'P': 40., 'dt': 1e-6, **{f'Y_{s}': float(y0[i]) for i, s in enumerate(names)}}
x0 = torch.tensor(tr.transform_input_array(pd.DataFrame([row]), inert, macro, micro))
base = x0.clone(); base[:, 0] = 2500.; base[:, 1] = 123.
xn = op.state_to_input_scaled(base, x0[:, op.macro_idx], x0[:, op.micro_idx],
    temperature_mode='autonomous', thermo_closure=closure, current_X_scaled=x0)
OUT['autonomous_pressure'] = {'original_P_atm': 40., 'provided_reference_next_P_atm': 123.,
    'returned_P_atm': float(xn[0, 1]), 'returned_T_K': float(xn[0, 0]),
    'expected_P_for_unchanged_T_Y_constV_atm': 40.}
gas.equilibrate('UV'); eq_t = gas.T; yeq = gas.Y.copy()
idx = [names.index(s) for s in order]
yt0 = torch.tensor(y0[idx][None], dtype=torch.float64)
yte = torch.tensor(yeq[idx][None], dtype=torch.float64)
t0 = torch.tensor([[1200.]], dtype=torch.float64)
tn = tr.update_temperature_autonomous(t0, yte-yt0, max_delta_temperature=500.,
    thermo_closure=closure, Y_old_full=yt0, Y_next_full=yte)
layer = closure['nasa7_layer']
_, _, _, u_after = layer.compute_mixture_properties(tn, yte)
OUT['temperature_clipping'] = {'true_UV_equilibrium_T_K': eq_t, 'returned_T_K': float(tn),
    'energy_residual_J_per_kg': float(u_after)-u0,
    'note': 'Large-jump counterexample, not a measured frequency in training.'}

# 7. Even a physically unchanged dormant species is lifted by the projection floor.
ym = torch.tensor([[1.]], dtype=torch.float64)
yf = torch.tensor([[1e-40]], dtype=torch.float64)
_, projected, diag = tr.project_physical_state(ym, yf,
    element_matrix=torch.tensor([[1., 1.]], dtype=torch.float64),
    target_elements=torch.tensor([[1.]], dtype=torch.float64))
OUT['projection_floor'] = {'raw_Y': float(yf), 'projected_Y': float(projected),
                          'factor': float(projected/yf)}

# 8. Audit only the small data artifact supplied in sc16, not a full training corpus.
ds = joblib.load(ROOT/'data_pipeline'/'mini_ds.pkl')
rows, ds_names = ds['rows'], ds['names']
Y = np.stack([r['Y'] for r in rows]); Yn = np.stack([r['Yn'] for r in rows])
Pq = np.stack([r['Pq'] for r in rows]); Dq = np.stack([r['Dq'] for r in rows])
P = np.asarray([r['P'] for r in rows]); dt = np.asarray([r['dt'] for r in rows])
T = np.asarray([r['T'] for r in rows]); Tn = np.asarray([r['Tn'] for r in rows])
sids = np.asarray([r['sid'] for r in rows]); unique, counts = np.unique(sids, return_counts=True)
qssa_log = np.log10(np.maximum(Pq, 1e-40))-np.log10(np.maximum(Dq, 1e-40))
d_masked = (qssa_log <= -15.) & (Dq > 0.)
decay_active = (Y > 1e-20) & (Yn < Y*.5)
by_sp = {}
for s in ['H', 'O', 'OH', 'HO2', 'NH2', 'HNO', 'NO', 'N2O']:
    j = ds_names.index(s)
    by_sp[s] = {'mask_fraction': float(d_masked[:,j].mean()),
                'masked_with_halving_above_1e_20': int((d_masked[:,j] & decay_active[:,j]).sum()),
                'exact_D_zero_fraction': float((Dq[:,j] == 0).mean())}
first = [int(np.flatnonzero(sids==sid)[0]) for sid in unique]
mw = np.array([gas.molecular_weights[gas.species_index(s)] for s in ds_names])
jnh3, jh2, jo2 = [ds_names.index(s) for s in ['NH3', 'H2', 'O2']]
phi = (.75*Y[first,jnh3]/mw[jnh3] + .5*Y[first,jh2]/mw[jh2]) / np.maximum(Y[first,jo2]/mw[jo2],1e-300)
Pnext_eos = P * (Tn/T) * ((Yn/mw).sum(1)/(Y/mw).sum(1))
OUT['mini_dataset'] = {'rows': len(rows), 'trajectories': len(unique), 'species': len(ds_names),
    'rows_per_trajectory_min_max': [int(counts.min()), int(counts.max())],
    'initial_T_range': [float(T[first].min()), float(T[first].max())],
    'initial_phi_simple_fuel_O2_range': [float(phi.min()), float(phi.max())],
    'dt_range': [float(dt.min()), float(dt.max())],
    'micro_mask_summary': by_sp,
    'converter_Pnext_equal_P_error_max_atm': float(abs(Pnext_eos-P).max()),
    'warning': 'mini_ds is a small diagnostic sample; its statistics do not describe the unavailable full current training corpus.'}
OUT['builder_import_target_exists'] = (ROOT/'data_pipeline'/'test'/'benchmark_cfd_closed_loop.py').exists()
OUT['N2O_classified_as'] = 'macro' if 'N2O' in macro else 'micro'

# 9. End-to-end target roundtrip: remove NN error and measure just representation/projection.
frame_columns = {'T': T, 'P': P, 'dt': dt}
for j, s in enumerate(ds_names):
    frame_columns[f'Y_{s}'] = Y[:, j]
    frame_columns[f'Y_next_{s}'] = Yn[:, j]
    frame_columns[f'dYdt_{s}'] = (Yn[:, j]-Y[:, j])/dt
    frame_columns[f'P_avg_{s}'] = Pq[:, j]
    frame_columns[f'D_coef_avg_{s}'] = Dq[:, j]
frame = pd.DataFrame(frame_columns)
roundtrip = {}
for mode in ['dual_branch', 'qssa_asymptotic']:
    ii, ma, mi = tr.classify_species(ds_names, include_no_in_macro=(mode=='dual_branch'))
    info = tr.build_feature_info(ds_names, ii, ma, mi, mode, mode)
    raw_macro_rates = np.column_stack([frame[f'dYdt_{s}'] for s in ma])
    info['macro_rate_scale'] = tr.compute_macro_rate_scale(raw_macro_rates).astype(np.float32)
    dx, dy = len(info['input_cols']), len(info['output_cols'])
    tx = tr.FeatureScaler(np.zeros(dx, dtype=np.float32), np.ones(dx, dtype=np.float32), 1)
    ty = tr.FeatureScaler(np.zeros(dy, dtype=np.float32), np.ones(dy, dtype=np.float32), 1)
    tool = tr.CoarseOperatorTools(info, tx, ty, device='cpu')
    xs = tr.transform_input_array(frame, ii, ma, mi)
    target = tr.transform_target_array(frame, ma, mi, info['macro_rate_scale'], mode)
    raw_all, constrained_all, residuals = [], [], []
    for start in range(0, len(frame), 256):
        with torch.no_grad():
            ps = tool.predict_next_state(torch.from_numpy(xs[start:start+256]),
                torch.from_numpy(target[start:start+256]), state_update_mode=mode)
        raw_all.append(ps['Y_next_raw_full'].numpy())
        constrained_all.append(ps['Y_next_constrained_full'].numpy())
        residuals.append(ps['diagnostic']['element_residual_max'].numpy())
    raw_all = np.concatenate(raw_all); constrained_all = np.concatenate(constrained_all)
    true = Yn[:, [ds_names.index(s) for s in ma+mi]]
    active = true > 1e-15
    log_err = abs(np.log10(np.maximum(raw_all,1e-40))-np.log10(np.maximum(true,1e-40)))
    log_proj_err = abs(np.log10(np.maximum(constrained_all,1e-40))-np.log10(np.maximum(true,1e-40)))
    elem_res = np.concatenate(residuals)
    worst = np.unravel_index(np.argmax(np.where(active,log_err,-1)),log_err.shape)
    roundtrip[mode] = {'max_abs_raw_error': float(abs(raw_all-true).max()),
        'max_abs_constrained_error': float(abs(constrained_all-true).max()),
        'raw_active_log_error_p95_max_dex': np.quantile(log_err[active], [.95,1.]).tolist(),
        'constrained_active_log_error_p95_max_dex': np.quantile(log_proj_err[active], [.95,1.]).tolist(),
        'projection_element_residual_p95_max': np.quantile(elem_res,[.95,1.]).tolist(),
        'projection_failure_fraction_above_1e_6': float((elem_res>1e-6).mean()),
        'worst_active_log_error_example': {'row': int(worst[0]), 'species': (ma+mi)[worst[1]],
            'Y_old': float(Y[worst[0],ds_names.index((ma+mi)[worst[1]])]),
            'true_next': float(true[worst]), 'decoded_next': float(raw_all[worst]),
            'dt': float(dt[worst[0]])}}
    # Change just the element matrix in memory and repeat the projection.
    tool.A = torch.tensor([[gas.n_atoms(s,e)*gas.atomic_weight(e)/gas.molecular_weights[gas.species_index(s)]
                           for s in ma+mi] for e in ['N','H','O']], dtype=torch.float32)
    corrected_all = []
    for start in range(0, len(frame), 256):
        with torch.no_grad():
            ps = tool.predict_next_state(torch.from_numpy(xs[start:start+256]),
                torch.from_numpy(target[start:start+256]), state_update_mode=mode)
        corrected_all.append(ps['Y_next_constrained_full'].numpy())
    corrected_all = np.concatenate(corrected_all)
    roundtrip[mode]['max_abs_constrained_error_with_mechanism_A'] = float(abs(corrected_all-true).max())
OUT['exact_label_roundtrip_fp32'] = roundtrip
OUT['mini_dataset']['durations_min_median_max_s'] = np.quantile(
    [dt[sids==s].sum() for s in unique], [0,.5,1]).tolist()
OUT['mini_dataset']['trajectories_with_200K_rise'] = int(sum(
    max(Tn[sids==s])-T[sids==s][0]>=200 for s in unique))

elements = ['N','H','O']
A_ct = np.array([[gas.n_atoms(s,e)*gas.atomic_weight(e)/gas.molecular_weights[gas.species_index(s)]
                  for s in macro+micro] for e in elements])
A_code = tr.build_element_matrix(macro+micro, device='cpu').numpy()
OUT['element_matrix_max_abs_difference_from_mechanism'] = float(abs(A_ct-A_code).max())

from dual_branch_network import DualBranchKineticsNet, CoupledKineticsResMLP
OUT['parameter_counts_for_39_species'] = {
    'mlp_qssa_asymptotic': sum(p.numel() for p in tr.CoupledKineticsNet(42,74,'qssa_asymptotic').parameters()),
    'resmlp_qssa_asymptotic': sum(p.numel() for p in CoupledKineticsResMLP(42,74).parameters()),
    'dual_branch_actual_partition': sum(p.numel() for p in DualBranchKineticsNet(42,len(macro),len(micro)).parameters())}

path = Path(__file__).with_name('results.json')
path.write_text(json.dumps(OUT, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in OUT.items() if k!='source_sha256'}, ensure_ascii=False, indent=2))
print('Saved:', path)
