"""Signed one-step bias of radicals during induction (oracle inputs), and the closed-loop radical pool vs Cantera."""
import os, sys, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
src = open('test/diagnostics_v72/probe_closed_loop.py', encoding='utf-8').read().split("cases = [")[0]
g = {}; exec(compile(src, 'head', 'exec'), g)
ct, gas, names, step_model, mix = g['ct'], g['gas'], g['names'], g['step_model'], g['mix']
SP = ['OH', 'H', 'O', 'HO2', 'NH2', 'H2O2', 'N2H4', 'HNO']
for name, T0, Patm, X, tmax in [("1200K_40atm", 1200, 40, mix(1.0, 0.2), 3.0e-4), ("1000K_80atm", 1000, 80, mix(0.8, 0.3), 2.3e-3)]:
    rng = np.random.default_rng(11)
    gas.TPX = T0, Patm * ct.one_atm, X
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r]); t = 0.0; bias = []
    while t < tmax:
        dt = 10 ** rng.uniform(-8, -4); dt = min(dt, tmax - t) if tmax - t > 1e-9 else dt
        Y0 = gas.Y.copy(); Tb = gas.T; Pb = gas.P / ct.one_atm
        sim.advance(t + dt); t = sim.time; Y1 = gas.Y.copy()
        if gas.T > T0 + 30: break
        Yp = step_model(Tb, Pb, Y0, dt)
        bias.append([np.log10(max(Yp[names.index(s)], 1e-30)) - np.log10(max(Y1[names.index(s)], 1e-30)) for s in SP])
    b = np.array(bias)
    print(f"{name}: induction steps={len(b)}  signed median log10(pred/true): " + ", ".join(f"{s}={v:+.2f}" for s, v in zip(SP, np.median(b, 0))))
    print(f"{'':12s} fraction of steps under-predicted: " + ", ".join(f"{s}={v:.0%}" for s, v in zip(SP, (b < 0).mean(0))))
