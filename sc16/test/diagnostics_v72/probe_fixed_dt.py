"""Closed-loop (autonomous T via UV closure) with a CFD-like constant dt."""
import os, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
src = open('test/diagnostics_v72/probe_closed_loop.py', encoding='utf-8').read().split("cases = [")[0]
g = {}; exec(compile(src, 'head', 'exec'), g)
ct, gas, names, step_model, mix, idt = g['ct'], g['gas'], g['names'], g['step_model'], g['mix'], g['idt']
iNO = names.index("NO")
for dt in [1e-6, 1e-5]:
    for name, T0, Patm, X, tmax in [("1200K_40atm", 1200, 40, mix(1.0, 0.2), 3e-3), ("1000K_80atm", 1000, 80, mix(0.8, 0.3), 6e-3), ("1300K_20atm_lean", 1300, 20, mix(0.6, 0.1), 3e-3), ("1600K_5atm_NH3", 1600, 5, mix(1.0, 0.0), 2e-3)]:
        n = int(round(tmax / dt))
        gas.TPX = T0, Patm * ct.one_atm, X; Y0 = gas.Y.copy(); rho = gas.density; u0 = gas.int_energy_mass
        r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r]); ts = [0.0]; Ts = [T0]
        for k in range(n):
            sim.advance((k + 1) * dt); ts.append(sim.time); Ts.append(gas.T)
        NOref = gas.Y[iNO]
        Y = Y0.copy(); T = T0; P = Patm; Ta = [T0]; ok = True
        for k in range(n):
            Y = step_model(T, P, Y, dt)
            try:
                gas.UVY = u0, 1.0 / rho, Y; T = gas.T; P = gas.P / ct.one_atm
            except ct.CanteraError:
                ok = False; break
            Ta.append(T)
        a, b = idt(ts, Ts, T0), idt(ts[:len(Ta)], Ta, T0)
        print(f"dt={dt:.0e} {name:18s} IDT ref={a*1e3:7.3f} ms  NN={b*1e3:7.3f} ms  ratio={b/a if np.isfinite(b) else float('nan'):5.2f} | Tend ref={Ts[-1]:6.0f} NN={Ta[-1]:6.0f} | NO ref={NOref:.2e} NN={Y[iNO]:.2e} | {'ok' if ok else 'UV closure failed at step %d' % len(Ta)}", flush=True)
