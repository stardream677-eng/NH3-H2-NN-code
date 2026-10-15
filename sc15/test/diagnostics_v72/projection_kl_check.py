"""Replay test-set oracle_raw states through (a) the current weighted-linear projection and
(b) a multiplicative (KL / element-potential) projection y = r*exp(A^T lam), and compare."""
import sys, importlib.util, numpy as np, pandas as pd, torch, cantera as ct
spec = importlib.util.spec_from_file_location('tr', 'train_sc_v68.py'); tr = importlib.util.module_from_spec(spec); sys.modules['tr'] = tr; spec.loader.exec_module(tr)
df = pd.read_csv(sys.argv[1] + "/v71_qssa_best_oracle_h128/trajectory_states.csv")
gas = ct.Solution("NUIG2024.yaml")
sp = list(df[df.trajectory_id == 1][df.step == 0].species)
macro = sp[:5]; micro = sp[5:]
els = [e for e in gas.element_names if any(gas.n_atoms(s, e) for s in sp)]
A = np.array([[gas.n_atoms(s, e) * gas.atomic_weight(e) / gas.molecular_weights[gas.species_index(s)] for s in sp] for e in els])

def kl_project(r, b, iters=30):
    r = np.maximum(r, 1e-40); lam = np.zeros(len(b))
    for _ in range(iters):
        y = r * np.exp(np.clip(A.T @ lam, -50, 50)); F = A @ y - b
        if np.abs(F).max() < 1e-13: break
        J = (A * y) @ A.T
        lam -= np.linalg.solve(J + 1e-30 * np.eye(len(b)), F)
    return r * np.exp(np.clip(A.T @ lam, -50, 50))

L = lambda a: np.log10(np.maximum(a, 1e-30))
stats = {'cur': [], 'kl': []}; killed = {'cur': 0, 'kl': 0}; n = 0
for tid, g in df.groupby('trajectory_id'):
    p_true = g.pivot(index='step', columns='species', values='true')[sp]
    p_raw = g.pivot(index='step', columns='species', values='oracle_raw')[sp]
    p_cur = g.pivot(index='step', columns='species', values='oracle')[sp]
    for k in range(1, len(p_true)):
        b = A @ p_true.iloc[k - 1].values
        raw = p_raw.iloc[k].values; tru = p_true.iloc[k].values
        y_kl = kl_project(raw, b)
        alive = (tru > 1e-12) & (raw > 1e-12)
        for key, y in (('cur', p_cur.iloc[k].values), ('kl', y_kl)):
            stats[key].append(np.abs(L(y[alive]) - L(tru[alive])).mean())
            killed[key] += int(((y < 1e-20) & (raw > 1e-8)).sum())
        n += 1
print(f"steps={n}")
for key in stats:
    print(f"{key:4s}: mean |log10 err| over alive species = {np.mean(stats[key]):.3f}, p95={np.percentile(stats[key],95):.3f}; species wiped (<1e-20 while raw>1e-8): {killed[key]}")
g = df[(df.trajectory_id == 2)]
pt = g.pivot(index='step', columns='species', values='true')[sp]; pr = g.pivot(index='step', columns='species', values='oracle_raw')[sp]; pc = g.pivot(index='step', columns='species', values='oracle')[sp]
yk = kl_project(pr.loc[9].values, A @ pt.loc[8].values)
i = sp.index('H2')
print(f"traj2 step9 H2: true={pt.loc[9].values[i]:.3e} raw={pr.loc[9].values[i]:.3e} current={pc.loc[9].values[i]:.3e} KL={yk[i]:.3e}; elem residual KL={np.abs(A@yk-A@pt.loc[8].values).max():.1e}")
