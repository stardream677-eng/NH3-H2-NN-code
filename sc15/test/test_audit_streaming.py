"""Unit tests for audit_generated_database.py streaming audit."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from audit_generated_database import audit_database


def test_audit_database_streaming(tmp_path: Path):
    # Create a small simulated database
    db_dir = tmp_path / "database_Transient_V21_QSSA"
    mech_dir = db_dir / "trajectories_nuig_2024"
    mech_dir.mkdir(parents=True)

    # 1. Create a simulated parquet part file with QSSA columns
    n_rows = 100
    dt = 1e-5
    # Species H2, O2, H2O
    y_h2 = np.full(n_rows, 0.1)
    p_h2 = np.full(n_rows, 0.0)
    d_h2 = np.full(n_rows, 10.0)
    # y_next = y0 * exp(-D*dt) + P*dt*phi1
    z = d_h2 * dt
    y_next_h2 = y_h2 * np.exp(-z)

    df = pd.DataFrame(
        {
            "sample_id": ["sample_0"] * n_rows,
            "mechanism_name": ["nuig_2024"] * n_rows,
            "stratum_name": ["Spec_Cold_Oxidation"] * n_rows,
            "time": np.linspace(0.0, 1e-3, n_rows),
            "dt": np.full(n_rows, dt),
            "P": np.full(n_rows, 1.0),
            "initial_radical_total": np.zeros(n_rows),
            "termination_reason": [""] * (n_rows - 1) + ["equilibrium"],
            "Y_H2": y_h2,
            "Y_next_H2": y_next_h2,
            "P_avg_H2": p_h2,
            "D_coef_avg_H2": d_h2,
        }
    )
    df.to_parquet(mech_dir / "part-00000.parquet")

    # 2. Summary file
    summary = {
        "database_version": "V21",
        "total_input_samples": 100,
        "success_trajectories": 98,
        "failed_trajectories": 2,
        "total_training_rows": n_rows,
    }
    pd.DataFrame([summary]).to_json(
        db_dir / "database_summary.json", orient="records", indent=2
    )

    metadata = {
        "mechanism_name": "nuig_2024",
        "success": 98,
        "failed": 2,
        "training_rows": n_rows,
    }
    pd.DataFrame([metadata]).to_json(
        mech_dir / "metadata.json", orient="records", indent=2
    )

    # Run audit and assert exit 0 (passes all gates)
    with pytest.raises(SystemExit) as exc_info:
        audit_database(db_dir, check_all_rows=True, min_success_rate=90.0)
    assert exc_info.value.code == 0
