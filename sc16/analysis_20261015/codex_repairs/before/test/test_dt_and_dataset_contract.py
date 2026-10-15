"""Tests for Task 2: Real dt contract, dataset helpers, and condition group isolation."""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

import train_sc_v68 as tr
import data_pipeline.split_trajectories_v21 as splitter
import data_pipeline.build_training_dataset_v72 as builder


def test_dt_preservation_and_rejection():
    """Verify that valid dt values (including dt > 1e-4) are preserved and invalid ones rejected."""
    dt_values = np.array([1e-8, 1e-6, 1e-4, 1.5e-4, 1.625e-4], dtype=np.float64)
    
    # 1. NumPy preservation
    sanitized_np = tr.sanitize_dt_numpy(dt_values)
    np.testing.assert_allclose(sanitized_np, dt_values, rtol=1e-12)
    assert sanitized_np[-1] == 1.625e-4, "dt > 1e-4 was silently truncated!"

    # 2. PyTorch preservation
    dt_tensor = torch.tensor(dt_values, dtype=torch.float64)
    sanitized_t = tr.sanitize_dt_tensor(dt_tensor)
    torch.testing.assert_close(sanitized_t, dt_tensor)
    assert sanitized_t[-1].item() == 1.625e-4, "dt > 1e-4 was silently truncated in torch!"

    # 3. Round-trip log10 encoding/decoding
    log_dt = np.log10(dt_values)
    decoded_dt = 10.0 ** log_dt
    np.testing.assert_allclose(decoded_dt, dt_values, rtol=1e-12)

    # 4. Strict rejection of 0, negative, NaN, Inf
    invalid_cases = [0.0, -1e-6, np.nan, np.inf, -np.inf]
    for inv in invalid_cases:
        with pytest.raises(ValueError, match="dt must be finite and strictly positive"):
            tr.require_positive_dt(np.array([1e-6, inv]))
        with pytest.raises(ValueError, match="dt must be finite and strictly positive"):
            tr.require_positive_dt_tensor(torch.tensor([1e-6, inv], dtype=torch.float64))


def test_compute_phi_calculation():
    """Verify stoichiometric equivalence ratio calculation across lean, stoich, and rich mixtures."""
    # Stoichiometric NH3: NH3 + 0.75 O2 -> 0.5 N2 + 1.5 H2O
    w_nh3, w_o2, w_h2 = 17.03052, 31.9988, 2.01588
    # For 1 mole NH3 (17.03052 g), need 0.75 mole O2 (23.9991 g)
    y_nh3 = 17.03052 / (17.03052 + 23.9991)
    y_o2 = 23.9991 / (17.03052 + 23.9991)
    phi_stoich = builder.compute_phi_from_species_dict({"NH3": y_nh3, "O2": y_o2})
    assert np.isclose(phi_stoich, 1.0, atol=1e-4), f"Expected phi=1.0, got {phi_stoich}"

    # Rich mixture (excess fuel by 1.5x)
    phi_rich = builder.compute_phi_from_species_dict({"NH3": y_nh3 * 1.5, "O2": y_o2})
    assert np.isclose(phi_rich, 1.5, atol=1e-4), f"Expected phi=1.5, got {phi_rich}"

    # Lean mixture (half fuel)
    phi_lean = builder.compute_phi_from_species_dict({"NH3": y_nh3 * 0.5, "O2": y_o2})
    assert np.isclose(phi_lean, 0.5, atol=1e-4), f"Expected phi=0.5, got {phi_lean}"


def test_convert_mini_ds_eos_and_atomic_write(tmp_path):
    """Verify convert_mini_ds writes atomically to target_dir and computes P_next via EOS for last row."""
    df = builder.convert_mini_ds(target_dir=tmp_path)
    assert df is not None
    out_file = tmp_path / "part-000001.parquet"
    assert out_file.is_file()
    assert not (tmp_path / "part-000001.tmp").exists()

    # Verify P_next is not blindly identical to P across entire combustion trajectories
    has_p_change = (df["P_next"] != df["P"]).any()
    assert has_p_change, "P_next was blindly copied from P without dynamic evolution!"

    # Verify condition_group_id is present
    assert "condition_group_id" in df.columns
    assert (df["condition_group_id"].str.len() > 0).all()

    # Verify phi is positive and not blindly 1.0 everywhere if mixture varies
    assert (df["phi"] > 0).all()


def test_condition_group_isolation_in_splits(tmp_path):
    """Verify that trajectories sharing condition_group_id are never split across train/val/test."""
    rows = []
    # 10 condition groups, each with 3 trajectories (e.g. different dt jitters)
    for g_idx in range(10):
        group_id = f"cond_group_{g_idx}"
        for t_idx in range(3):
            sid = f"traj_{g_idx}_{t_idx}"
            rows.append({
                "sample_id": sid,
                "condition_group_id": group_id,
                "time": 0.0,
                "T": 1000.0 + g_idx * 50.0,
                "P": 1.0 + g_idx,
                "phi_actual": 1.0,
                "Y_H2": 0.05,
            })
    
    df = pd.DataFrame(rows)
    df.to_parquet(tmp_path / "part-00000.parquet", index=False)

    counts, metadata = splitter._collect_trajectory_metadata(tmp_path)
    assert "condition_group_id" in metadata.columns

    split_res = splitter._split_ids(
        np.asarray(sorted(counts), dtype=object),
        np.random.default_rng(42),
        train=0.6,
        val=0.2,
        metadata=metadata,
    )

    train_groups = set(metadata.loc[split_res["train"], "condition_group_id"])
    val_groups = set(metadata.loc[split_res["validation"], "condition_group_id"])
    test_groups = set(metadata.loc[split_res["test"], "condition_group_id"])

    # Verify strict isolation between sets
    assert train_groups.isdisjoint(val_groups), f"Train and Val share groups: {train_groups & val_groups}"
    assert train_groups.isdisjoint(test_groups), f"Train and Test share groups: {train_groups & test_groups}"
    assert val_groups.isdisjoint(test_groups), f"Val and Test share groups: {val_groups & test_groups}"
