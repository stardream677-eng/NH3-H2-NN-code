"""Failure snapshots must retain the failed predicted state, not only teacher batch data.

These tests intentionally need only PyTorch, NumPy and the diagnostics module.
"""
import torch

import training_diagnostics as td


class _FakeTools:
    def __init__(self):
        # Physical columns: T, P, inert, macro, micro in log10, dt in log10.
        self.feature_info = {
            "input_cols": ["T", "P", "Y_N2", "Y_H2", "logY_OH", "log_dt"]
        }
        self.macro_idx = torch.tensor([3], dtype=torch.long)
        self.micro_idx = torch.tensor([4], dtype=torch.long)
        self.inert_idx = torch.tensor([2], dtype=torch.long)
        self.mean = torch.tensor([1000.0, 1e5, 0.0, 0.0, 0.0, 0.0])
        self.scale = torch.tensor([100.0, 1e5, 1.0, 1.0, 1.0, 1.0])

    def inverse_state(self, state):
        return state * self.scale.to(state.device) + self.mean.to(state.device)


def test_failed_rollout_capture_preserves_predicted_physical_state_and_dt():
    tools = _FakeTools()
    # Contains feedback from an earlier predicted step, not the reference batch.
    current_scaled = torch.tensor(
        [[1.5, 0.2, 0.70, 0.08, -7.0, -5.0],
         [2.0, 0.5, 0.65, 0.10, -9.0, -6.0]],
        requires_grad=True,
    )
    predicted_rate = torch.tensor([[0.1], [0.2]], requires_grad=True)
    predicted = {
        "Y_macro_next_raw": torch.tensor([[0.07], [0.09]], requires_grad=True),
        "logY_micro_next_raw": torch.tensor([[-6.5], [-8.5]], requires_grad=True),
    }

    state = td.capture_rollout_failure_state(
        tools, current_scaled, step=17,
        phase="autonomous_feedback_state",
        predicted_rate_scaled=predicted_rate, predicted_state=predicted,
    )
    assert state["phase"] == "autonomous_feedback_state"
    assert state["rollout_step_zero_based"] == 17
    assert state["batch_size"] == 2
    torch.testing.assert_close(state["temperature_K"], torch.tensor([1150., 1200.]))
    torch.testing.assert_close(state["pressure_in_dataset_units"], torch.tensor([120000., 150000.]))
    torch.testing.assert_close(state["dt_seconds"], torch.tensor([1e-5, 1e-6], dtype=torch.float64), rtol=1e-6, atol=0)
    torch.testing.assert_close(state["current_inert_mass_fractions"], torch.tensor([[0.70], [0.65]]))
    torch.testing.assert_close(state["current_macro_mass_fractions"], torch.tensor([[0.08], [0.10]]))
    torch.testing.assert_close(state["current_micro_log10_mass_fractions"], torch.tensor([[-7.], [-9.]]))
    torch.testing.assert_close(state["Y_macro_next_raw"], torch.tensor([[0.07], [0.09]]))
    assert state["current_input_scaled"].device.type == "cpu"
    assert not state["current_input_scaled"].requires_grad
    assert not state["predicted_rate_scaled"].requires_grad
    assert not state["logY_micro_next_raw"].requires_grad


def test_failure_snapshot_roundtrips_step_and_predicted_state(tmp_path):
    tools = _FakeTools()
    model = torch.nn.Linear(6, 1)
    tracker = td.TrainingDiagnosticTracker(snapshot_dir=tmp_path, max_skip_budget=1)
    tracker.step_attempted(model)
    teacher = torch.zeros((1, 3, 6), dtype=torch.float32)
    updated = torch.tensor([[1.0, 0.1, 0.7, 0.08, -7.0, -5.0]])
    failure_context = td.capture_rollout_failure_state(
        tools, updated, step=1, phase="autonomous_feedback_state",
    )
    path = tracker.record_failure_and_check_budget(
        model, teacher, epoch=3, sample_ids=["trajectory-A"], window_start=[32],
        error_msg="NASA7 uv energy closure did not converge",
        diagnostics={"failed_rollout_state": failure_context},
        replay_context={"actual_horizon": 2, "part": ["part-00003.parquet"]},
    )
    snap = torch.load(path, map_location="cpu", weights_only=False)
    assert snap["epoch"] == 3
    assert snap["sample_ids"] == ["trajectory-A"]
    assert snap["window_start"] == [32]
    assert snap["diagnostics"]["failed_rollout_state"]["rollout_step_zero_based"] == 1
    assert snap["replay_context"]["actual_horizon"] == 2
    torch.testing.assert_close(
        snap["diagnostics"]["failed_rollout_state"]["current_input_scaled"], updated
    )
    assert "NASA7" in snap["error_msg"]


def test_failed_rollout_capture_identifies_invalid_contract():
    tools = _FakeTools()
    tools.feature_info["input_cols"].remove("log_dt")
    with __import__("pytest").raises(ValueError, match="feature columns"):
        td.capture_rollout_failure_state(
            tools, torch.zeros(1, 6), step=0, phase="state_update"
        )
