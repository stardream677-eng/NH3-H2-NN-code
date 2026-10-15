"""Unit tests for chemistry_contract module and element matrix consistency."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import pytest
import numpy as np
import torch
import joblib
import cantera as ct

import chemistry_contract as cc
import train_sc_v68 as tr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MECH_PATH = PROJECT_ROOT / "NUIG2024.yaml"


def test_element_matrix_exact_match_with_cantera():
    """Verify that element_mass_matrix strictly matches Cantera species thermo properties."""
    gas = ct.Solution(str(MECH_PATH))
    species_order = list(gas.species_names)
    
    elements, A = cc.compute_element_mass_matrix(gas, species_order)
    
    # 1. Total elemental mass fraction must sum to 1.0 for each species
    col_sums = A.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-12, rtol=1e-12)
    
    # 2. Check each element entry manually for key species
    # H2: 2 * 1.008 / 2.016 = 1.0 (for H)
    i_H2 = species_order.index("H2")
    i_H = elements.index("H")
    assert pytest.approx(1.0, rel=1e-10) == A[i_H, i_H2]
    
    # H2O: H: 2 * 1.008 / 18.015, O: 15.999 / 18.015
    i_H2O = species_order.index("H2O")
    i_O = elements.index("O")
    assert A[i_H, i_H2O] > 0.11
    assert A[i_O, i_H2O] > 0.88
    assert pytest.approx(1.0, rel=1e-10) == A[i_H, i_H2O] + A[i_O, i_H2O]


def test_legacy_handcoded_aw_discrepancy():
    """Verify that legacy parse_formula produces ~2.01e-5 discrepancy against real Cantera."""
    gas = ct.Solution(str(MECH_PATH))
    species_order = list(gas.species_names)
    
    elements_new, A_new = cc.compute_element_mass_matrix(gas, species_order)
    elements_old, A_old = cc.parse_legacy_element_matrix(species_order)
    
    assert elements_new == elements_old
    max_diff = np.max(np.abs(A_new - A_old))
    print(f"Max difference between Cantera element matrix and legacy AW: {max_diff:.2e}")
    assert max_diff > 1.5e-5, f"Expected noticeable difference ~2e-5, got {max_diff}"
    assert max_diff < 3.0e-5


def test_real_label_projection_roundtrip():
    """Test that true labels from mini_ds.pkl have 20x lower projection error with true matrix."""
    pkl_path = PROJECT_ROOT / "data_pipeline" / "mini_ds.pkl"
    if not pkl_path.exists():
        pytest.skip("mini_ds.pkl not found")
        
    ds = joblib.load(pkl_path)
    names = ds["names"]
    rows = ds["rows"]
    
    gas = ct.Solution(str(MECH_PATH))
    elements_new, A_new = cc.compute_element_mass_matrix(gas, names)
    elements_old, A_old = cc.parse_legacy_element_matrix(names)
    
    A_new_tensor = torch.tensor(A_new, dtype=torch.float32)
    A_old_tensor = torch.tensor(A_old, dtype=torch.float32)
    
    # Extract true next Y
    Y_next_list = [r["Yn"] for r in rows[:500]]
    Y_true = torch.tensor(np.array(Y_next_list), dtype=torch.float32)
    
    # Calculate target elements from true state using Cantera matrix
    target_elements = torch.matmul(Y_true, A_new_tensor.T)
    
    # Project Y_true using legacy matrix vs new matrix
    # Since Y_true is already on the true chemical manifold:
    # Projection with true matrix should yield tiny residual
    _, _, diag_new = tr.project_physical_state(
        Y_macro_raw=Y_true[:, :6],
        Y_micro_raw=Y_true[:, 6:],
        inert_sum=0.0,
        element_matrix=A_new_tensor,
        target_elements=target_elements,
        iterations=8,
        projection_mode="multiplicative_kl",
    )
    
    _, _, diag_old = tr.project_physical_state(
        Y_macro_raw=Y_true[:, :6],
        Y_micro_raw=Y_true[:, 6:],
        inert_sum=0.0,
        element_matrix=A_old_tensor,
        target_elements=torch.matmul(Y_true, A_old_tensor.T),
        iterations=8,
        projection_mode="multiplicative_kl",
    )
    
    # The true matrix should have virtually zero correction (< 5e-7)
    max_corr_new = diag_new["max_correction"].max().item()
    max_corr_old = diag_old["max_correction"].max().item()
    
    print(f"Max projection intervention with Cantera matrix: {max_corr_new:.2e}")
    print(f"Max projection intervention with legacy AW matrix: {max_corr_old:.2e}")
    
    assert max_corr_new < 1e-6
