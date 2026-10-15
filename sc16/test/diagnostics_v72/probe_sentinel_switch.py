"""Does the discrete (logP,-45) <-> (log(P/D),logD) target switch hurt growing species?
For induction+flame steps: compare true branch with predicted z2, and signed state error per group."""
import os, sys, importlib.util, numpy as np, torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
src = open('test/diagnostics_v72/probe_closed_loop.py', encoding='utf-8').read().split("cases = [")[0]
g = {}; exec(compile(src, 'head', 'exec'), g)
spec2 = importlib.util.spec_from_file_location('rn', 'runner_sc_v21_qssa.py'); rn = importlib.util.module_from_spec(spec2); sys.modules['rn'] = rn; spec2.loader.exec_module(rn)
ct, gas, names, react, ridx, mix = g['ct'], g['gas'], g['names'], g['react'], g['ridx'], g['mix']
model, tools, build_X, tr = g['model'], g['tools'], g['build_X'], g['tr']
W = gas.molecular_weights
SP = ['OH', 'H', 'O', 'HO2', 'NO', 'N2O', 'H2O', 'NH2', 'HNO']
spi = [react.index(s) for s in SP]
rows = []
for T0, Patm, X, tmax, seed in [(1200, 40, mix(1.0, 0.2), 6e-4, 3), (1000, 80, mix(0.8, 0.3), 2.8e-3, 4), (1300, 20, mix(0.6, 0.1), 9e-4, 5), (1100, 60, mix(1.0, 0.4), 8e-4, 6)]:
    rng = np.random.default_rng(seed)
    gas.TPX = T0, Patm * ct.one_atm, X
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r]); t = 0.0
    while t < tmax:
        dt = 10 ** rng.uniform(-8, -5.5)
        Y0 = gas.Y.copy(); Tb = gas.T; Pb = gas.P / ct.one_atm
        Cs = gas.creation_rates * W / gas.density; Ds = gas.destruction_rates * W / gas.density / np.maximum(Y0, 1e-300)
        sim.advance(t + dt); dtr = sim.time - t; t = sim.time; Y1 = gas.Y.copy()
        Ce = gas.creation_rates * W / gas.density; De = gas.destruction_rates * W / gas.density / np.maximum(Y1, 1e-300)
        Pq, Dq = rn._solve_exact_qssa_params(Y0, Y1, dtr, (Cs + Ce) / 2, (Ds + De) / 2)
        if gas.T > 2400: break
        with torch.no_grad():
            Xs = build_X(Tb, Pb, Y0, dtr); out = model(Xs)
            z = (out * tools.scale_Y + tools.mean_Y)[0].double().cpu().numpy()
            ps = tools.predict_next_state(Xs, out, safe_for_rollout=False, state_update_mode="qssa_asymptotic")
            Yraw = ps["Y_next_raw_full"][0].double().cpu().numpy()
        for s, j in zip(SP, spi):
            P, D = Pq[ridx][j], Dq[ridx][j]
            prod_only = D <= 0
            z2p = z[2 * j + 1]
            e = np.log10(max(Yraw[j], 1e-30)) - np.log10(max(Y1[ridx][j], 1e-30))
            rows.append((s, prod_only, z2p, e, Y1[ridx][j] > Y0[ridx][j]))
rows = np.array(rows, dtype=object)
prod = rows[:, 1].astype(bool); z2p = rows[:, 2].astype(float); err = rows[:, 3].astype(float); grow = rows[:, 4].astype(bool)
print(f"rows={len(rows)}  true prod-only (D=0, z2=-45) fraction={prod.mean():.1%}; growing species fraction={grow.mean():.1%}")
amb = (z2p > -40) & (z2p < -8)
for name, m in [("true prod-only & pred z2<-40 (branch right)", prod & (z2p <= -40)),
                ("true prod-only & pred z2 in (-40,-8) (wrong branch, P≈10^(z1+z2))", prod & amb),
                ("true prod-only & pred z2>=-8 (decay branch)", prod & (z2p >= -8)),
                ("true decay & pred z2<-40 (wrong branch)", (~prod) & (z2p <= -40)),
                ("true decay & pred z2>=-40", (~prod) & (z2p > -40))]:
    if m.sum():
        print(f"  {name:70s} n={m.sum():5d}  median signed log10 err={np.median(err[m]):+.2f}  frac under-pred={np.mean(err[m] < 0):.0%}  median|err|={np.median(np.abs(err[m])):.2f}")
for s in SP:
    m = rows[:, 0] == s
    print(f"  {s:4s} prod-only frac={prod[m].mean():.0%}  ambiguous-z2 frac (among prod-only)={amb[m & prod].mean() if (m & prod).any() else 0:.0%}  median signed err={np.median(err[m]):+.2f}")
