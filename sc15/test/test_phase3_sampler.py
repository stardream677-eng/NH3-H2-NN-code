"""Phase 3 Unit Test: Universal Latin Hypercube Sampler (lhs_sampler_sc_v11.py)

Verifies:
1. Strict Air-Oxidizer baseline (N2 / O2 == 3.76190476) when Z_dil == 0.
2. Exact stoichiometry conservation: phi_actual == phi (relative error < 1e-12).
3. Exact sum of mole fractions: sum(X_i) == 1.0 (to IEEE-754 precision).
4. Four physical strata coverage with auditable dilution metadata and zero-radical baseline in Cold Oxidation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lhs_sampler_sc_v11 as sampler_mod


def test_sampler_air_baseline_and_stoichiometry():
    print("\n--- Testing LHS Sampler: Air Baseline & Stoichiometric Consistency ---")
    sampler = sampler_mod.UniversalLHSSampler_Species({
        "nuig_2024": "NUIG2024.yaml"
    })

    # Test boundary where Z_dil = 0 (pure air)
    bounds_pure_air = {
        "T": [1000.0, 1000.0],
        "P": [10.0, 10.0],
        "phi": [1.0, 1.0],
        "X_H2_fuel": [0.5, 0.5],
        "Z_dil": [0.0, 0.0],  # 0% diluent -> 100% standard air oxidizer
        "f_N2": [1.0, 1.0],
        "f_AR": [0.0, 0.0],
        "f_HE": [0.0, 0.0],
        "f_H2O": [0.0, 0.0],
        "X_OH": [0.0, 0.0],
        "X_H": [0.0, 0.0],
        "X_O": [0.0, 0.0],
        "X_NO": [0.0, 0.0],
    }

    df = sampler._generate_composite(n_pts=50, bounds=bounds_pure_air, seed=42, include_radicals=False, stratum_name="Spec_Pure_Air_Test")

    # 1. Sum of mole fractions == 1.0
    # Note: metadata columns are not species mole fractions
    meta_cols = {"stratum_name", "T", "P", "phi", "phi_actual", "radical_total", "X_H2_fuel", "Z_dil", "f_N2", "f_AR", "f_HE", "f_H2O"}
    species_cols = [c for c in df.columns if c not in meta_cols]
    total_x = df[species_cols].sum(axis=1)
    max_sum_err = np.max(np.abs(total_x - 1.0))
    print(f"Max sum(species) - 1.0 error: {max_sum_err:.3e}")
    assert max_sum_err < 1e-14, f"Sum of mole fractions deviates from 1.0: {max_sum_err}"

    # 2. Strict standard air ratio: N2 / O2 == 3.76190476
    n2_o2_ratio = df["X_N2"] / df["X_O2"]
    expected_ratio = sampler_mod.AIR_N2_O2_RATIO
    max_air_ratio_err = np.max(np.abs(n2_o2_ratio - expected_ratio))
    print(f"Max N2/O2 ratio error vs standard air ({expected_ratio:.6f}): {max_air_ratio_err:.3e}")
    assert max_air_ratio_err < 1e-12, f"Air oxidizer ratio mismatch: {max_air_ratio_err}"

    # 3. Exact equivalence ratio: phi_actual == phi
    phi_err = np.max(np.abs(df["phi_actual"] - df["phi"]))
    print(f"Max |phi_actual - phi| error: {phi_err:.3e}")
    assert phi_err < 1e-14, f"Stoichiometric phi mismatch: {phi_err}"
    print("[PASS] Air baseline and stoichiometric consistency verified.")


def test_four_strata_generation():
    print("\n--- Testing LHS Sampler: 4 Physical Strata Generation & Auditing ---")
    sampler = sampler_mod.UniversalLHSSampler_Species({
        "nuig_2024": "NUIG2024.yaml"
    })

    # Generate small dataset covering all 4 layers
    total_pts = 400
    df = sampler.generate_species_database(total_pts=total_pts, include_radicals=True)
    assert len(df) == total_pts, f"Expected {total_pts} rows, got {len(df)}"

    # Check that metadata columns are properly exported
    for col in ["stratum_name", "Z_dil", "f_N2", "f_AR", "f_HE", "f_H2O"]:
        assert col in df.columns, f"Expected column {col} in output CSV"

    # Check non-negativity and bounds
    assert np.all(df["T"] >= 300.0) and np.all(df["T"] <= 2400.0), "T out of range"
    assert np.all(df["P"] >= 0.5) and np.all(df["P"] <= 150.0), "P out of range"
    assert np.all(df["phi"] >= 0.05) and np.all(df["phi"] <= 4.5), "phi out of range"

    # Check composition sum
    meta_cols = {"stratum_name", "sample_id", "mechanism_name", "T", "P", "phi", "phi_actual", "radical_total", "X_H2_fuel", "Z_dil", "f_N2", "f_AR", "f_HE", "f_H2O"}
    species_cols = [c for c in df.columns if c not in meta_cols]
    sum_x = df[species_cols].sum(axis=1)
    max_sum_err = np.max(np.abs(sum_x - 1.0))
    print(f"Dataset-wide max sum(species) - 1.0: {max_sum_err:.3e}")
    assert max_sum_err < 1e-12, f"Total mole fraction sum error: {max_sum_err}"

    # Verify that Cold Oxidation layer is pure and free of artificial radicals
    cold_df = df[df["stratum_name"] == "Spec_Cold_Oxidation"]
    assert len(cold_df) > 0
    assert np.all(cold_df["X_OH"] == 0.0), "Cold oxidation must not have seeded OH"
    assert np.all(cold_df["X_H"] == 0.0), "Cold oxidation must not have seeded H"
    assert np.all(cold_df["X_O"] == 0.0), "Cold oxidation must not have seeded O"
    print(f"Verified {len(cold_df)} cold oxidation samples have exactly 0.0 initial radicals.")

    # Verify that Radical Burst stratum contains elevated radicals
    burst_df = df[df["stratum_name"] == "Spec_Radical_Burst"]
    assert len(burst_df) > 0
    assert np.min(burst_df["X_OH"]) > 1e-6, "Radical burst must contain active OH"
    print(f"Verified {len(burst_df)} radical burst samples have active radicals.")

    print(f"[PASS] 4 Strata generation and auditability successfully verified ({len(df)} samples).")


if __name__ == "__main__":
    test_sampler_air_baseline_and_stoichiometry()
    test_four_strata_generation()
    print("\nALL PHASE 3 SAMPLER TESTS PASSED!")
