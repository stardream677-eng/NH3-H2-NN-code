"""One-step oracle error by combustion regime, plus how often the D-head mask hides a decaying species."""
import os, sys, importlib.util, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location('cl', 'test/diagnostics_v72/probe_closed_loop.py')
src = open('test/diagnostics_v72/probe_closed_loop.py', encoding='utf-8').read().split("cases = [")[0]
g = {}; exec(compile(src, 'probe_closed_loop_head', 'exec'), g)
spec2 = importlib.util.spec_from_file_location('rn', 'runner_sc_v21_qssa.py'); rn = importlib.util.module_from_spec(spec2); sys.modules['rn'] = rn; spec2.loader.exec_module(rn)
ct = g['ct']; gas = g['gas']; names = g['names']; react = g['react']; ridx = g['ridx']; step_model = g['step_model']; mix = g['mix']
W = gas.molecular_weights
SP = ['NH3', 'O2', 'H2O', 'H2', 'OH', 'H', 'O', 'HO2', 'NH2', 'NO', 'N2O']
cases = [("1200K_40atm", 1200, 40, mix(1.0, 0.2), 2e-3), ("1000K_80atm", 1000, 80, mix(0.8, 0.3), 4e-3), ("1300K_20atm_lean", 1300, 20, mix(0.6, 0.1), 2e-3)]
for name, T0, Patm, X, tmax in cases:
    rng = np.random.default_rng(7)
    gas.TPX = T0, Patm * ct.one_atm, X
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r]); t = 0.0
    recs = []
    while t < tmax and len(recs) < 1500:
        dt = 10 ** rng.uniform(-8, -4)
        Y0 = gas.Y.copy(); Tb = gas.T; Pb = gas.P / ct.one_atm
        Cs = gas.creation_rates * W / gas.density; Ds = gas.destruction_rates * W / gas.density / np.maximum(Y0, 1e-300)
        sim.advance(t + dt); dtr = sim.time - t; t = sim.time; Y1 = gas.Y.copy()
        Ce = gas.creation_rates * W / gas.density; De = gas.destruction_rates * W / gas.density / np.maximum(Y1, 1e-300)
        Pq, Dq = rn._solve_exact_qssa_params(Y0, Y1, dtr, (Cs + Ce) / 2, (Ds + De) / 2)
        recs.append((t - dtr, dtr, Tb, Pb, gas.T, Y0, Y1, Pq, Dq))
    Ts = np.array([x[2] for x in recs]); Tn = np.array([x[4] for x in recs])
    Tad = Tn[-1]
    regime = np.where(Ts < T0 + 30, 'induction', np.where(np.abs(Tn - Ts) > 5, 'flame', 'post'))
    err = {k: [] for k in ['induction', 'flame', 'post']}
    masked_live = 0; masked_total = 0
    for (tt, dt, Tb, Pb, Tnx, Y0, Y1, Pq, Dq), reg in zip(recs, regime):
        Yp = step_model(Tb, Pb, Y0, dt)
        e = [np.log10(max(Yp[names.index(s)], 1e-30)) - np.log10(max(Y1[names.index(s)], 1e-30)) for s in SP]
        err[reg].append(e)
        # D-head mask: z1=log10(P/D) <= -15 -> D head unsupervised. Count cases where species is still alive (Y0 > 1e-12)
        Pr = Pq[ridx]; Dr = Dq[ridx]
        with np.errstate(divide='ignore'):
            z1 = np.where(Pr <= 0, -45, np.where(Dr <= 0, np.log10(np.clip(Pr, 1e-40, None)), np.clip(np.log10(np.clip(Pr, 1e-40, None)) - np.log10(np.clip(Dr, 1e-40, None)), -40, 30)))
        m = (z1 <= -15) & (Dr > 0)
        masked_total += m.sum(); masked_live += (m & (Y0[ridx] > 1e-12) & (Dr * dt > 0.1)).sum()
    print(f"=== {name}: steps={len(recs)} Tad={Tad:.0f}K  n(induction/flame/post)={[len(err[k]) for k in err]}")
    print("   median |log10 err| per species (oracle one step):")
    print("   regime     " + " ".join(f"{s:>6s}" for s in SP))
    for k in err:
        if err[k]:
            a = np.abs(np.array(err[k]))
            print(f"   {k:10s} " + " ".join(f"{v:6.2f}" for v in np.median(a, axis=0)))
            print(f"   {k+'-p90':10s} " + " ".join(f"{v:6.2f}" for v in np.percentile(a, 90, axis=0)))
    print(f"   D-head masked entries: {masked_total}, of which species still alive (Y>1e-12) and decaying (D*dt>0.1): {masked_live}")
