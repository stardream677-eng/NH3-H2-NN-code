"""Regression tests for trace-state and database integrity fixes."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runner_sc_v21_qssa as runner
import train_sc_v68 as train
from compute_physical_metrics import PhysicalMetricAccumulator


def test_trace_input_transform_keeps_1e_minus_35_state():
    frame = pd.DataFrame(
        {
            "T": [850.0],
            "P": [80.0],
            "Y_H2": [0.1],
            "Y_OH": [1e-35],
            "dt": [1e-5],
        }
    )
    transformed = train.transform_input_array(frame, [], ["H2"], ["OH"])
    assert transformed[0, 3] == pytest.approx(-35.0, abs=1e-5)


def test_tiny_negative_mass_fraction_is_sanitized_and_large_negative_rejected():
    cfg = runner.Config(negative_mass_fraction_tolerance=1e-12)
    sanitized = runner._sanitize_mass_fractions(
        np.array([0.7, 0.3 + 2e-27, -2e-27]), cfg
    )
    assert np.all(sanitized >= 0.0)
    assert np.sum(sanitized) == pytest.approx(1.0, abs=1e-15)

    with pytest.raises(ValueError, match="negative_mass_fraction"):
        runner._sanitize_mass_fractions(np.array([1.0, -1e-6]), cfg)


def test_audit_trajectory_key_keeps_mechanisms_distinct():
    assert runner.trajectory_key("nuig_2024", 7) != runner.trajectory_key("ucf_2024", 7)


def test_qssa_asymptotic_preserves_tiny_destruction_and_zero_creation():
    assert train.QSSA_ASYMPTOTIC_TRANSFORM_SIGNATURE == "qssa_asymptotic_v2"
    frame = pd.DataFrame(
        {
            "P_avg_OH": [0.0, 1e-27],
            "D_coef_avg_OH": [1e3, 1e-25],
        }
    )
    targets = train.transform_target_array(
        frame,
        macro_species=[],
        micro_species=["OH"],
        macro_rate_scale=[],
        output_parameterization="qssa_asymptotic",
    )

    # Zero creation has a dedicated sentinel; tiny but nonzero D remains a
    # normal logarithmic target instead of being silently treated as D=0.
    assert targets[0, 0] < -40.0
    assert targets[0, 1] == pytest.approx(3.0)
    assert targets[1, 1] == pytest.approx(-25.0)


def test_floor_metrics_follow_trace_floor_not_legacy_1e_minus_20():
    acc = PhysicalMetricAccumulator(["OH"])
    acc.update(np.array([[1e-35], [1e-40]]), np.array([[1e-35], [1e-40]]))
    result = acc.as_dict()["per_species"]["OH"]
    assert result["floor_hit_count"] == 1


def test_resume_rejects_a_different_rate_transform():
    """QSSA and asymptotic-QSSA checkpoints are not interchangeable."""
    checkpoint = {
        "version": train.VERSION,
        "transform_signature": train.QSSA_TRANSFORM_SIGNATURE,
        "rollout_signature": train.ROLLOUT_SIGNATURE,
    }
    assert not train.checkpoint_compatible(
        checkpoint,
        train.VERSION,
        train.QSSA_ASYMPTOTIC_TRANSFORM_SIGNATURE,
        train.ROLLOUT_SIGNATURE,
    )


def test_resume_rejects_changed_optimization_configuration():
    """A checkpoint from the pre-fix raw-loss run must not resume silently."""
    checkpoint = {
        "version": train.VERSION,
        "transform_signature": train.QSSA_ASYMPTOTIC_TRANSFORM_SIGNATURE,
        "rollout_signature": train.ROLLOUT_SIGNATURE,
        "training_config": {
            "output_parameterization": "qssa_asymptotic",
            "rate_decode_mode": "qssa_asymptotic",
            "state_update_mode": "qssa_asymptotic",
            "state_loss_source": "raw",
            "feedback_state_source": "constrained",
            "lambda_element": 0.05,
            "lambda_mass": 0.05,
            "lambda_positive": 0.01,
        },
    }
    expected = {
        "output_parameterization": "qssa_asymptotic",
        "rate_decode_mode": "qssa_asymptotic",
        "state_update_mode": "qssa_asymptotic",
        "state_loss_source": "constrained",
        "feedback_state_source": "constrained",
        "lambda_element": 0.0,
        "lambda_mass": 0.0,
        "lambda_positive": 0.0,
    }
    assert not train.checkpoint_compatible(
        checkpoint,
        train.VERSION,
        train.QSSA_ASYMPTOTIC_TRANSFORM_SIGNATURE,
        train.ROLLOUT_SIGNATURE,
        expected_training_config=expected,
    )
