"""Shared exploratory cases restored from sc17-0/test; not held-out test data."""

def make_fuel_air_mix(phi, xh2, egr=0.0):
    f_nh3, f_h2 = 1.0 - xh2, xh2
    o2 = (0.75 * f_nh3 + 0.5 * f_h2) / phi
    X = {"NH3": f_nh3, "H2": f_h2, "O2": o2, "N2": 3.7619 * o2}
    if egr > 0:
        tot = sum(X.values())
        X["N2"] += egr * tot
    return X

BENCHMARK_CASES = [
    # Core Engine / High Pressure (Cases 1-6)
    {"name": "1200K_40atm_phi1.0_H2-20", "T0": 1200, "P": 40, "X": make_fuel_air_mix(1.0, 0.2), "tmax": 4e-3},
    {"name": "1100K_60atm_phi1.0_H2-40", "T0": 1100, "P": 60, "X": make_fuel_air_mix(1.0, 0.4), "tmax": 4e-3},
    {"name": "1000K_80atm_phi0.8_H2-30", "T0": 1000, "P": 80, "X": make_fuel_air_mix(0.8, 0.3), "tmax": 8e-3},
    {"name": "1300K_20atm_phi0.6_H2-10", "T0": 1300, "P": 20, "X": make_fuel_air_mix(0.6, 0.1), "tmax": 4e-3},
    {"name": "1400K_10atm_phi1.4_H2-20", "T0": 1400, "P": 10, "X": make_fuel_air_mix(1.4, 0.2), "tmax": 3e-3},
    {"name": "1600K_5atm_phi1.0_NH3-pure", "T0": 1600, "P": 5, "X": make_fuel_air_mix(1.0, 0.0), "tmax": 2e-3},

    # Low-Temperature Compression Limit (Cases 7-10)
    {"name": "950K_40atm_phi1.0_H2-40", "T0": 950, "P": 40, "X": make_fuel_air_mix(1.0, 0.4), "tmax": 10e-3},
    {"name": "900K_60atm_phi1.0_H2-50", "T0": 900, "P": 60, "X": make_fuel_air_mix(1.0, 0.5), "tmax": 12e-3},
    {"name": "1000K_40atm_phi1.0_H2-20", "T0": 1000, "P": 40, "X": make_fuel_air_mix(1.0, 0.2), "tmax": 10e-3},
    {"name": "1050K_30atm_phi0.8_H2-30", "T0": 1050, "P": 30, "X": make_fuel_air_mix(0.8, 0.3), "tmax": 8e-3},

    # Lean / Rich Boundaries (Cases 11-14)
    {"name": "1100K_20atm_phi1.2_H2-20", "T0": 1100, "P": 20, "X": make_fuel_air_mix(1.2, 0.2), "tmax": 6e-3},
    {"name": "1150K_40atm_phi1.0_H2-10", "T0": 1150, "P": 40, "X": make_fuel_air_mix(1.0, 0.1), "tmax": 6e-3},
    {"name": "1250K_10atm_phi0.5_H2-30", "T0": 1250, "P": 10, "X": make_fuel_air_mix(0.5, 0.3), "tmax": 5e-3},
    {"name": "1350K_20atm_phi1.5_H2-20", "T0": 1350, "P": 20, "X": make_fuel_air_mix(1.5, 0.2), "tmax": 4e-3},

    # Gas Turbine & Atmospheric Burner (Cases 15-18)
    {"name": "1200K_1atm_phi1.0_H2-40", "T0": 1200, "P": 1, "X": make_fuel_air_mix(1.0, 0.4), "tmax": 8e-3},
    {"name": "1300K_5atm_phi1.0_H2-20", "T0": 1300, "P": 5, "X": make_fuel_air_mix(1.0, 0.2), "tmax": 5e-3},
    {"name": "1400K_30atm_phi1.0_NH3-pure", "T0": 1400, "P": 30, "X": make_fuel_air_mix(1.0, 0.0), "tmax": 4e-3},
    {"name": "1450K_1atm_phi1.0_NH3-pure", "T0": 1450, "P": 1, "X": make_fuel_air_mix(1.0, 0.0), "tmax": 6e-3},

    # High Temperature / Dilution & Extremes (Cases 19-24)
    {"name": "1500K_10atm_phi0.8_H2-10", "T0": 1500, "P": 10, "X": make_fuel_air_mix(0.8, 0.1), "tmax": 3e-3},
    {"name": "1550K_20atm_phi1.0_H2-30", "T0": 1550, "P": 20, "X": make_fuel_air_mix(1.0, 0.3), "tmax": 2e-3},
    {"name": "1000K_30atm_phi1.0_H2-pure", "T0": 1000, "P": 30, "X": make_fuel_air_mix(1.0, 1.0), "tmax": 3e-3},
    {"name": "1100K_40atm_phi1.0_H2-20_EGR10", "T0": 1100, "P": 40, "X": make_fuel_air_mix(1.0, 0.2, egr=0.10), "tmax": 8e-3},
    {"name": "1200K_30atm_phi1.0_H2-20_EGR15", "T0": 1200, "P": 30, "X": make_fuel_air_mix(1.0, 0.2, egr=0.15), "tmax": 6e-3},
    {"name": "1300K_40atm_phi1.0_H2-20_EGR20", "T0": 1300, "P": 40, "X": make_fuel_air_mix(1.0, 0.2, egr=0.20), "tmax": 5e-3},
]
