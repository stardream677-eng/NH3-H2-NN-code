import os, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
src = open('test/diagnostics_v72/probe_closed_loop.py', encoding='utf-8').read().split("cases = [")[0]
g = {}; exec(compile(src, 'head', 'exec'), g)
ct, gas, names, step_model, mix = g['ct'], g['gas'], g['names'], g['step_model'], g['mix']
dt = 1e-6; T0, Patm, X = 1600, 5, mix(1.0, 0.0)
gas.TPX = T0, Patm * ct.one_atm, X; Y0 = gas.Y.copy(); rho = gas.density; u0 = gas.int_energy_mass
r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r]); ref = {}
for k in range(2000):
    sim.advance((k + 1) * dt)
    if (k + 1) in (50, 100, 200, 400, 1000, 2000): ref[k + 1] = (gas.T, gas.Y.copy())
S = ['NH3', 'O2', 'H2', 'H2O', 'N2', 'NO', 'OH']
Y = Y0.copy(); T = T0; P = Patm
for k in range(2000):
    Y = step_model(T, P, Y, dt); gas.UVY = u0, 1.0 / rho, Y; T = gas.T; P = gas.P / ct.one_atm
    if (k + 1) in ref:
        Tr, Yr = ref[k + 1]
        print(f"t={(k+1)*dt*1e3:5.2f} ms  T ref={Tr:6.0f} NN={T:6.0f} | " + " ".join(f"{s}: {Yr[names.index(s)]:.2e}/{Y[names.index(s)]:.2e}" for s in S))
