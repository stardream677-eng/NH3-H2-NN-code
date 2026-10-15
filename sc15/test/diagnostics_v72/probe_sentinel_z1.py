"""Post-flame wipe-outs with oracle inputs: is it the z1<-40 'zero creation' decode branch, or the projection?"""
import os, sys, importlib.util, numpy as np, torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
src = open('test/diagnostics_v72/probe_closed_loop.py', encoding='utf-8').read().split("cases = [")[0]
g = {}; exec(compile(src, 'head', 'exec'), g)
ct, gas, names, react, ridx, mix = g['ct'], g['gas'], g['names'], g['react'], g['ridx'], g['mix']
model, tools, build_X = g['model'], g['tools'], g['build_X']
SP = ['NH3', 'H2', 'H', 'NH2', 'O', 'OH']; j = [react.index(s) for s in SP]
for name, T0, Patm, X, tmax in [("1000K_80atm", 1000, 80, mix(0.8, 0.3), 4e-3), ("1300K_20atm_lean", 1300, 20, mix(0.6, 0.1), 2e-3)]:
    rng = np.random.default_rng(7); gas.TPX = T0, Patm * ct.one_atm, X
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r]); t = 0.0
    cnt = {s: [0, 0, 0] for s in SP}; n = 0
    while t < tmax and n < 1500:
        dt = 10 ** rng.uniform(-8, -4); Y0 = gas.Y.copy(); Tb = gas.T; Pb = gas.P / ct.one_atm
        sim.advance(t + dt); dtr = sim.time - t; t = sim.time; Y1 = gas.Y.copy(); n += 1
        if Tb < T0 + 300: continue
        with torch.no_grad():
            Xs = build_X(Tb, Pb, Y0, dtr); out = model(Xs)
            z = (out * tools.scale_Y + tools.mean_Y)[0].double().cpu().numpy()
            ps = tools.predict_next_state(Xs, out, safe_for_rollout=True, state_update_mode="qssa_asymptotic")
            raw = ps["Y_next_raw_full"][0].double().cpu().numpy(); prj = ps["Y_next_constrained_full"][0].double().cpu().numpy()
        for s, k in zip(SP, j):
            tru = Y1[ridx][k]
            if tru < 1e-12: continue
            if prj[k] < 1e-20:
                cnt[s][0] += 1
                if z[2 * k] < -40: cnt[s][1] += 1       # zero-creation sentinel branch fired
                elif raw[k] > 1e-12: cnt[s][2] += 1     # raw was fine, projection wiped it
    print(f"{name}: hot steps; wiped (<1e-20 while true>1e-12) counts per species -> [total, via z1<-40 sentinel, via projection]")
    print("   " + ", ".join(f"{s}: {v}" for s, v in cnt.items()))
