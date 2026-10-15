import sys
import os
import torch
import numpy as np
import cantera as ct
sys.path.insert(0, 'test')
import benchmark_cfd_closed_loop as bm

def inspect():
    engine = bm.BenchmarkEngine('output1/phase68_runs/v72_cfd_optimized', 'best_pure_ar_h128_epoch4.pth')
    case = bm.BENCHMARK_CASES[0]  # 1200K_40atm_phi1.0_H2-20
    dt = 2e-6
    gas = engine.gas
    gas.TPX = case['T0'], case['P'] * ct.one_atm, case['X']
    u0 = gas.int_energy_mass
    rho0 = gas.density
    Y = gas.Y.copy()
    
    iNO = engine.names.index('NO')
    iOH = engine.names.index('OH')
    iH2 = engine.names.index('H2')
    iNH3 = engine.names.index('NH3')
    iO2 = engine.names.index('O2')
    iH2O = engine.names.index('H2O')
    iN2 = engine.names.index('N2')
    
    print(f"Initial: T={gas.T:.1f} K, Y_H2={Y[iH2]:.4e}, Y_NH3={Y[iNH3]:.4e}, Y_O2={Y[iO2]:.4e}")
    T = gas.T
    P = case['P']
    
    for step in range(1, 1501):
        Y = engine.step_model(T, P, Y, dt, projection_mode='kl')
        gas.UVY = u0, 1.0 / rho0, Y
        T = gas.T
        P = gas.P / ct.one_atm
        
        if step in [10, 50, 100, 200, 500, 800, 1000, 1028, 1050, 1100, 1200, 1500]:
            print(f"Step {step:4d} (t={step*dt*1000:.3f}ms): T={T:6.1f} K, P={P:5.1f} atm | "
                  f"Y_H2={Y[iH2]:.3e}, Y_NH3={Y[iNH3]:.3e}, Y_OH={Y[iOH]:.3e}, Y_NO={Y[iNO]:.3e}")

if __name__ == '__main__':
    inspect()
