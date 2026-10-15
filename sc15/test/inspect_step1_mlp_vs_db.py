import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'test')
import glob, torch, numpy as np, cantera as ct
import train_sc_v68 as tr
sys.modules['__main__'].FeatureScaler = tr.FeatureScaler
import benchmark_cfd_closed_loop as bch

eng_mlp = bch.BenchmarkEngine('output1/phase68_runs/v72_cfd_optimized_mlp', 'best_composite_epoch4.pth', device='cuda')
eng_db = bch.BenchmarkEngine('output1/phase68_runs/v72_dual_branch', 'best_v68_net_nuig_2024.pth', device='cuda')
case = bch.BENCHMARK_CASES[0] # 1200K_40atm_phi1.0_H2-20

gas = eng_mlp.gas
gas.TPX = case['T0'], case['P'] * ct.one_atm, case['X']
Y0 = gas.Y.copy()
dt = 2e-6

print("=== INITIAL RECTANT STATE ===")
print("T0 =", case['T0'], "P =", case['P'])
print("Y_H2 =", Y0[gas.species_names.index('H2')])
print("Y_O2 =", Y0[gas.species_names.index('O2')])
print("Y_OH =", Y0[gas.species_names.index('OH')])

# Step 1 MLP
with torch.no_grad():
    Xs_mlp = eng_mlp.build_X(case['T0'], case['P'], Y0, dt)
    out_mlp = eng_mlp.model(Xs_mlp)
    ps_mlp = eng_mlp.tools.predict_next_state(Xs_mlp, out_mlp, safe_for_rollout=True, state_update_mode='qssa_asymptotic')

# Step 1 DB
with torch.no_grad():
    Xs_db = eng_db.build_X(case['T0'], case['P'], Y0, dt)
    out_db = eng_db.model(Xs_db)
    ps_db = eng_db.tools.predict_next_state(Xs_db, out_db, safe_for_rollout=True, state_update_mode='dual_branch')

print("\n=== STEP 1 RAW OUTPUTS ===")
print("MLP out shape:", out_mlp.shape)
print("DB out shape:", out_db.shape)

# In MLP: all 37 species are QSSA asymptotic (pairs of logP, logD)
P_mlp = ps_mlp['P_pred'][0].cpu().numpy()
D_mlp = ps_mlp['D_pred'][0].cpu().numpy()
print("\n--- MLP QSSA terms for radicals & H2 ---")
for sp in ['H2', 'OH', 'H', 'O', 'HO2', 'NH2']:
    if sp in eng_mlp.react:
        idx = eng_mlp.react.index(sp)
        print(f"MLP {sp:<5}: P = {P_mlp[idx]:.3e}, D = {D_mlp[idx]:.3e}")

print("\n--- DB terms ---")
print("DB Macro rates (dY/dt):")
for i, sp in enumerate(eng_db.macro):
    print(f"DB {sp:<5}: rate = {ps_db['rate_macro'][0, i].item():.3e} 1/s")

print("DB Micro QSSA terms:")
P_db = ps_db['P_pred'][0].cpu().numpy()
D_db = ps_db['D_pred'][0].cpu().numpy()
for sp in ['OH', 'H', 'O', 'HO2', 'NH2']:
    if sp in eng_db.micro:
        idx = eng_db.micro.index(sp)
        print(f"DB {sp:<5}: P = {P_db[idx]:.3e}, D = {D_db[idx]:.3e}")

# Check next mass fractions after 1 step:
Y1_mlp = eng_mlp.step_model(case['T0'], case['P'], Y0, dt, projection_mode='kl')
Y1_db = eng_db.step_model(case['T0'], case['P'], Y0, dt, projection_mode='kl')

print("\n=== AFTER 1 STEP (dt = 2 us) ===")
for sp in ['H2', 'O2', 'H2O', 'OH', 'H', 'O']:
    idx = gas.species_names.index(sp)
    print(f"{sp:<5} | Y0: {Y0[idx]:.3e} | MLP Y1: {Y1_mlp[idx]:.3e} | DB Y1: {Y1_db[idx]:.3e}")
