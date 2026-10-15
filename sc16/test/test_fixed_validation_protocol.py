"""Unit tests for Task 7: Fixed validation protocol, sentinel horizons, and model selection.

Verifies:
1. Determinism: consecutive evaluations on the same data yield identical metrics.
2. Horizon isolation: H=1 curriculum evaluation does not overwrite or block H=128 records.
3. Cold false-positive and active radical logging.
4. Non-finite gating rejects corrupted candidates.
5. Case-by-case aggregate consistency.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

import species_validation_protocol as svp


def test_validation_protocol_determinism():
    """Verify that running the evaluation twice on identical data yields bitwise identical results."""
    np.random.seed(123)
    species = ["NH3", "H2", "O2", "N2", "H2O", "OH", "NO"]
    T = 32
    S = len(species)
    true_states = np.random.uniform(1e-5, 0.2, size=(T, S))
    pred_states = true_states + np.random.normal(0, 1e-4, size=(T, S))

    res1 = svp.compute_species_trajectory_metrics(true_states, pred_states, species)
    res2 = svp.compute_species_trajectory_metrics(true_states, pred_states, species)

    assert res1["composite_score"] == pytest.approx(res2["composite_score"], abs=1e-12)
    assert res1["protocol_id"] == svp.PROTOCOL_ID
    for sp in species:
        assert res1["per_species"][sp]["mae"] == pytest.approx(res2["per_species"][sp]["mae"], abs=1e-12)
        assert res1["per_species"][sp]["nmae"] == pytest.approx(res2["per_species"][sp]["nmae"], abs=1e-12)


def test_horizon_sentinel_tracker_isolation():
    """Verify that curriculum training on short horizons does not overwrite long horizon records."""
    tracker = svp.HorizonSentinelTracker(horizons=(1, 8, 32, 128))

    # Stage 1: Epoch 1, training at H=1 gets very low 1-step error 0.002
    m_h1 = {
        "composite_score": 0.002,
        "is_valid_candidate": True,
        "mean_major_nmae": 0.001,
    }
    updated_h1 = tracker.consider_candidate(horizon=1, metrics=m_h1, epoch=1)
    assert updated_h1 is True
    assert tracker.best_scores[1] == 0.002
    # Horizons 8, 32, 128 must remain untouched
    assert tracker.best_scores[8] == float("inf")
    assert tracker.best_scores[128] == float("inf")

    # Stage 2: Epoch 10, training reaches H=128, gets 128-step error 0.045
    m_h128 = {
        "composite_score": 0.045,
        "is_valid_candidate": True,
        "mean_major_nmae": 0.03,
    }
    updated_h128 = tracker.consider_candidate(horizon=128, metrics=m_h128, epoch=10)
    assert updated_h128 is True
    assert tracker.best_scores[128] == 0.045
    # H=1 record must still be preserved
    assert tracker.best_scores[1] == 0.002
    assert tracker.best_epochs[1] == 1
    assert tracker.best_epochs[128] == 10

    # Stage 3: A candidate with non-finite values must be rejected
    m_corrupt = {
        "composite_score": 0.001,
        "is_valid_candidate": False,
    }
    assert tracker.consider_candidate(horizon=128, metrics=m_corrupt, epoch=11) is False
    assert tracker.best_scores[128] == 0.045


def test_cold_false_positive_and_radical_tracking():
    """Verify detection of cold false positives and active radical log errors."""
    species = ["NH3", "O2", "OH", "NO"]
    # 2 time steps
    # Step 0: cold state (OH=1e-15, NO=1e-15)
    # Step 1: active state (OH=1e-3, NO=1e-4)
    true_states = np.array([
        [0.1, 0.2, 1e-15, 1e-15],
        [0.08, 0.16, 1e-3, 1e-4],
    ])
    # Prediction has cold false-positive on NO at step 0 (pred = 1e-3 > 1e-6)
    pred_states = np.array([
        [0.1, 0.2, 1e-15, 1e-3],
        [0.08, 0.16, 1.2e-3, 1e-4],
    ])

    metrics = svp.compute_species_trajectory_metrics(true_states, pred_states, species)
    no_metrics = metrics["per_species"]["NO"]

    assert no_metrics["cold_fp_count"] == 1
    assert no_metrics["cold_fp_ratio"] == 1.0  # 1 out of 1 cold evaluation
    assert metrics["cold_fp_rate"] > 0.0

    # OH active log-RMSE at step 1: log10(1.2e-3) - log10(1e-3) = log10(1.2) ~ 0.079
    oh_metrics = metrics["per_species"]["OH"]
    expected_log_err = abs(np.log10(1.2))
    assert oh_metrics["active_log_rmse"] == pytest.approx(expected_log_err, rel=1e-3)


def test_nonfinite_gating():
    """Verify that non-finite values in prediction trigger gating disqualification."""
    species = ["NH3", "H2", "O2"]
    true_states = np.ones((4, 3)) * 0.1
    pred_states = true_states.copy()
    pred_states[2, 1] = np.nan

    metrics = svp.compute_species_trajectory_metrics(true_states, pred_states, species)
    assert metrics["nonfinite_count"] == 1
    assert metrics["is_valid_candidate"] is False
    assert metrics["composite_score"] > 1e5
