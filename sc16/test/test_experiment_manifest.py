"""Unit tests for experiment_manifest provenance, dataset audit, and benchmark audit."""

import json
from pathlib import Path
import pytest
import numpy as np

import experiment_manifest as em
import chemistry_contract as cc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MECH_PATH = PROJECT_ROOT / "NUIG2024.yaml"


def test_source_hashing():
    """Verify source files are hashed into hex SHA256."""
    hashes = em.compute_source_hashes(PROJECT_ROOT)
    assert "train_sc_v68.py" in hashes
    assert "chemistry_contract.py" in hashes
    assert len(hashes["train_sc_v68.py"]) == 64


def test_checkpoint_compatibility_verification():
    """Verify strict compatibility rejection on mechanism or parameterization mismatch."""
    dummy_contract = {
        "mechanism_name": "nuig_2024",
        "mechanism_sha256": "a" * 64,
        "species_order": ["H2", "O2", "H2O"],
        "element_names": ["H", "O"],
    }
    expected_manifest = em.create_experiment_manifest(
        mechanism_contract=dummy_contract,
        output_parameterization="dual_branch",
        transform_signature="dual_branch_v1",
        dt_policy={"min": 1e-8, "max": 2e-4},
        concentration_policy={"floor": 1e-40},
        evaluation_protocol_id="eval_v1",
        root_dir=PROJECT_ROOT,
    )

    # 1. Matching checkpoint should pass
    compatible_ckpt = {
        "manifest": expected_manifest,
        "training_config": {"output_parameterization": "dual_branch"},
    }
    is_ok, errs = em.verify_checkpoint_compatibility(compatible_ckpt, expected_manifest, strict=True)
    assert is_ok
    assert len(errs) == 0

    # 2. Checkpoint with different mechanism hash should fail
    bad_mech_ckpt = {
        "manifest": {**expected_manifest, "mechanism_sha256": "b" * 64},
        "training_config": {"output_parameterization": "dual_branch"},
    }
    is_ok, errs = em.verify_checkpoint_compatibility(bad_mech_ckpt, expected_manifest, strict=True)
    assert not is_ok
    assert any("Mechanism SHA256 mismatch" in e for e in errs)

    # 3. Checkpoint lacking manifest in strict mode should fail
    legacy_ckpt = {
        "training_config": {"output_parameterization": "dual_branch"},
    }
    is_ok, errs = em.verify_checkpoint_compatibility(legacy_ckpt, expected_manifest, strict=True)
    assert not is_ok
    assert any("lacks 'manifest'" in e for e in errs)


def test_benchmark_audit_detects_cases_discrepancy():
    """Verify that recompute_benchmark_summary accurately recomputes metrics from cases."""
    # Simulate a benchmark output where summary claimed 23 ignitions, but cases actually had 24 valid IDTs
    cases = []
    for i in range(24):
        cases.append({
            "case": f"case_{i}",
            "closure_ok": True,
            "fail_step": None,
            "ref_idt_ms": 1.0,
            "nn_idt_ms": 1.2 if i < 23 else 0.8, # All 24 cases actually ignited!
            "idt_ratio": 1.2 if i < 23 else 0.8,
            "delta_Tend": 50.0,
            "no_rel_err": 0.05,
            "wiped_species_count": 0,
        })
        
    summary = em.recompute_benchmark_summary(cases)
    assert summary["num_cases"] == 24
    assert summary["ignited_count"] == 24  # Not 23!
    assert summary["closure_success_rate"] == 1.0
    assert summary["mean_abs_delta_Tend"] == pytest.approx(50.0)


def test_audit_parquet_dataset_detects_inconsistency(tmp_path):
    """Verify that parquet audit compares actual footer rows with metadata.json."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Create dummy parquet table with 100 rows
    table = pa.table({"a": np.arange(100)})
    pq.write_table(table, tmp_path / "part-000001.parquet")

    # Write metadata.json claiming only 50 rows (inconsistent)
    metadata = [{"training_rows": 50}]
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    audit = em.audit_parquet_dataset(tmp_path)
    assert audit["actual_total_rows"] == 100
    assert audit["recorded_total_rows"] == 50
    assert not audit["is_consistent"]
    assert "metadata.json records 50 rows, but actual Parquet files contain 100 rows" in audit["inconsistency_warning"]
