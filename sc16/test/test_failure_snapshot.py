"""Tests for Task 4: Failure snapshot, gradient finiteness diagnostics, and anomaly replay."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path

import training_diagnostics as td


def test_gradient_finite_and_float64_norm():
    """Verify check_gradient_finite_and_norm identifies non-finite gradients and uses float64."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    # 1. Normal finite gradients
    x = torch.randn(5, 4)
    loss = model(x).sum()
    loss.backward()

    is_finite, norm_f64, diag = td.check_gradient_finite_and_norm(model)
    assert is_finite
    assert norm_f64 > 0.0
    assert np.isfinite(norm_f64)
    assert len(diag["non_finite_params"]) == 0

    # 2. Inject NaN gradient into a specific layer
    with torch.no_grad():
        model[0].weight.grad[0, 0] = float("nan")

    is_finite, norm_f64, diag = td.check_gradient_finite_and_norm(model)
    assert not is_finite
    assert "0.weight" in diag["non_finite_params"]
    assert "0.weight" in diag["large_grad_params"]


def test_tracker_monotonic_idx_and_budget_enforcement(tmp_path):
    """Verify monotonic batch index and skip budget enforcement."""
    tracker = td.TrainingDiagnosticTracker(snapshot_dir=tmp_path, max_skip_budget=2, debug_mode=False)
    model = nn.Linear(3, 2)

    # Step 1: Success
    idx1 = tracker.step_attempted()
    tracker.record_success()
    assert idx1 == 1

    # Step 2: Failure 1 (within budget)
    idx2 = tracker.step_attempted()
    assert idx2 == 2
    x = torch.randn(2, 3)
    p1 = tracker.record_failure_and_check_budget(
        model=model,
        batch_X=x,
        sample_ids=[101, 102],
        window_start=16,
        epoch=1,
        error_msg="simulated NaN",
    )
    assert p1.is_file()
    assert tracker.skipped_batches == 1

    # Step 3: Failure 2 (at budget)
    idx3 = tracker.step_attempted()
    assert idx3 == 3
    p2 = tracker.record_failure_and_check_budget(
        model=model,
        batch_X=x,
        sample_ids=[103, 104],
        window_start=32,
        epoch=1,
        error_msg="simulated NaN 2",
    )
    assert p2.is_file()
    assert tracker.skipped_batches == 2

    # Step 4: Failure 3 (exceeds budget -> raises RuntimeError)
    tracker.step_attempted()
    with pytest.raises(RuntimeError, match="Exceeded maximum skip budget"):
        tracker.record_failure_and_check_budget(
            model=model,
            batch_X=x,
            sample_ids=[105, 106],
            window_start=48,
            epoch=1,
            error_msg="simulated NaN 3",
        )


def test_debug_mode_immediate_halt(tmp_path):
    """Verify debug mode halts immediately on the very first failure."""
    tracker = td.TrainingDiagnosticTracker(snapshot_dir=tmp_path, debug_mode=True)
    model = nn.Linear(2, 2)
    tracker.step_attempted()

    with pytest.raises(RuntimeError, match="Debug mode halt"):
        tracker.record_failure_and_check_budget(
            model=model,
            batch_X=torch.randn(2, 2),
            epoch=0,
            error_msg="halt test",
        )


def test_replay_with_anomaly_detection(tmp_path):
    """Verify replay_snapshot_with_anomaly_detection isolates backward NaN operations."""
    model = nn.Linear(2, 1)
    x = torch.tensor([[-5.0, -10.0]], dtype=torch.float32)

    # Function that produces NaN in autograd (sqrt of negative)
    def forward_fn(m, bx, by=None):
        out = m(bx)
        # sqrt of negative creates NaN gradient in backward!
        return torch.sqrt(out.clamp_max(-1.0).abs() * -1.0)

    tracker = td.TrainingDiagnosticTracker(snapshot_dir=tmp_path)
    tracker.step_attempted()
    snap_path = tracker.record_failure_and_check_budget(
        model=model,
        batch_X=x,
        epoch=0,
        error_msg="test",
    )

    result = td.replay_snapshot_with_anomaly_detection(
        snapshot_path=snap_path,
        model=model,
        forward_fn=forward_fn,
    )
    assert result["anomaly_detected"]
    assert result["anomaly_msg"] is not None
