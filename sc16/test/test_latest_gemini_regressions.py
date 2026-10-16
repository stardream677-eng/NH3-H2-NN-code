import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import dual_branch_network as dbn
import enhanced_loss_and_weights as elw
import run_clean_benchmark as clean_benchmark
import validate_sc_v68 as validator
import audit_initial_conditions_v67 as initial_audit
import train_sc_v68 as tr


def test_legacy_null_space_checkpoint_keeps_identical_forward_output():
    element_matrix = torch.tensor([[1.0, 0.0, 1.0]])
    old_model = dbn.DualBranchKineticsNet(
        input_dim=4,
        slow_dim=3,
        fast_dim=2,
        hidden_dim=8,
        num_blocks=1,
        use_null_space=True,
        element_matrix_slow=element_matrix,
    ).eval()
    new_model = dbn.DualBranchKineticsNet(
        input_dim=4,
        slow_dim=3,
        fast_dim=2,
        hidden_dim=8,
        num_blocks=1,
        use_null_space=False,
    ).eval()
    x = torch.randn(5, 4)

    with torch.no_grad():
        expected = old_model(x)
        dbn.load_dual_branch_state_dict(new_model, old_model.state_dict())
        actual = new_model(x)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert "P_null" not in new_model.state_dict()


def test_clean_benchmark_module_call_handles_missing_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights_dir = Path("output1/phase68_runs/v72_dual_branch/weights_v68")
    weights_dir.mkdir(parents=True)
    (weights_dir / "dummy.pth").touch()

    monkeypatch.setattr(clean_benchmark.bch, "BENCHMARK_CASES", [{"name": "case"}])
    monkeypatch.setattr(clean_benchmark.bch, "BenchmarkEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        clean_benchmark.bch,
        "run_single_case",
        lambda *args, **kwargs: {
            "case": "case",
            "ref_Tend": 1500.0,
            "nn_Tend": 1200.0,
            "delta_Tend": -300.0,
            "ref_idt_ms": 1.0,
            "nn_idt_ms": None,
            "idt_ratio": None,
            "no_rel_err": None,
        },
    )

    clean_benchmark.run_benchmark("dummy.pth")

    output = json.loads(
        Path("output1/phase68_runs/v72_dual_branch/clean_benchmark_24cases.json").read_text()
    )
    assert output["mean_idt_ratio"] is None
    assert output["mean_no_rel_err"] is None
    assert output["mean_abs_delta_Tend"] == pytest.approx(300.0)


def test_radical_loss_supports_radical_in_macro_species():
    class Tools:
        macro_species = ["OH"]
        micro_species = []
        macro_idx = [0]
        micro_idx = []
        mean_X = torch.zeros(1)
        scale_X = torch.ones(1)

    pred_macro = torch.tensor([[2e-6], [3e-6]], requires_grad=True)
    loss = elw.compute_physical_radical_loss(
        tools=Tools(),
        pred_state={"Y_macro_next_raw": pred_macro},
        true_next_X_scaled=torch.tensor([[1e-6], [2e-6]]),
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred_macro.grad).all()


def test_coarse_operator_tools_honors_requested_device():
    feature_info = tr.build_feature_info(
        species=["N2", "H2", "O2", "OH"],
        inert_species=["N2"],
        macro_species=["H2", "O2"],
        micro_species=["OH"],
        output_parameterization="qssa_asymptotic",
        rate_decode_mode="qssa_asymptotic",
    )

    class Scaler:
        def __init__(self, size):
            self.mean_ = torch.zeros(size).numpy()
            self.scale_ = torch.ones(size).numpy()
            self.feature_names_in_ = [f"f_{i}" for i in range(size)]

    tools = tr.CoarseOperatorTools(
        feature_info,
        Scaler(len(feature_info["input_cols"])),
        Scaler(len(feature_info["output_cols"])),
        rate_decode_mode="qssa_asymptotic",
        device="cpu",
    )

    assert tools.mean_X.device.type == "cpu"
    assert tools.mean_Y.device.type == "cpu"
    assert tools.A.device.type == "cpu"


def test_ste_gradient_flows_through_constrained_rollout():
    feature_info = tr.build_feature_info(
        species=["N2", "H2", "O2", "OH"],
        inert_species=["N2"],
        macro_species=["H2", "O2"],
        micro_species=["OH"],
        output_parameterization="direct_rate",
        rate_decode_mode="direct_rate",
    )

    class Scaler:
        def __init__(self, size):
            self.mean_ = torch.zeros(size).numpy()
            self.scale_ = torch.ones(size).numpy()

    tools = tr.CoarseOperatorTools(
        feature_info,
        Scaler(len(feature_info["input_cols"])),
        Scaler(len(feature_info["output_cols"])),
        rate_decode_mode="direct_rate",
        device="cpu",
    )
    # T, P, Y_N2, Y_H2, Y_O2, logY_OH, log_dt
    current_X_scaled = torch.tensor([[1000.0, 1.0, 0.7, 0.05, 0.25, -5.0, -5.0]])
    rate_pred_scaled = torch.tensor([[0.01, -0.01, 0.0]], requires_grad=True)

    result = tools.predict_next_state(
        current_X_scaled=current_X_scaled,
        rate_pred_scaled=rate_pred_scaled,
        safe_for_rollout=True,
        state_update_mode="legacy",
    )
    y_next = result["Y_macro_next"]
    loss = (y_next ** 2).sum()
    loss.backward()

    assert rate_pred_scaled.grad is not None
    assert torch.isfinite(rate_pred_scaled.grad).all()
    assert (rate_pred_scaled.grad.abs() > 0.0).any()


def test_compute_qssa_rate_mask_dual_branch():
    class DummyTools:
        rate_decode_mode = "dual_branch"
        macro_species = ["NH3", "O2"]
        micro_species = ["OH", "H"]
        macro_idx = [0, 1]
        micro_idx = [2, 3]
        mean_Y = torch.zeros(6)
        scale_Y = torch.ones(6)

    # 2 macro (unmasked) + 2 micro pairs:
    # Radical 1: z1=-20 (dormant), z2=-10 -> destruction head (col 3) masked
    # Radical 2: z1=-5 (active), z2=-5 -> unmasked
    true_rate = torch.tensor([
        [1.0, -1.0, -20.0, -10.0, -5.0, -5.0]
    ])
    # Parameters alone cannot establish dormancy; use both physical endpoints.
    assert tr.compute_qssa_rate_mask(DummyTools(), true_rate) is None
    state = torch.tensor([[0.1, 0.9, -20.0, -5.0]])
    mask = tr.compute_qssa_rate_mask(DummyTools(), true_rate, state, state)
    assert mask is not None
    # Macro heads (0, 1): 1.0
    assert (mask[0, :2] == 1.0).all()
    # Micro radical 1: z1 is col 2 (1.0), z2 is col 3 (0.0 because dormant)
    assert mask[0, 2] == 1.0
    assert mask[0, 3] == 0.0
    # Micro radical 2: both 1.0
    assert mask[0, 4] == 1.0
    assert mask[0, 5] == 1.0


def test_validation_true_rate_supports_dual_branch():
    class DummyTools:
        rate_decode_mode = "dual_branch"
        macro_idx = [3, 4]
        micro_idx = [5]
        dt_idx = 6

        def inverse_state(self, x):
            return x

        def get_current_dt(self, x):
            return torch.tensor([[1e-6]])

    # current: [T, P, Y_Ar, Y_H2, Y_O2, logY_OH, dt]
    x_curr = torch.tensor([[1000.0, 1.0, 0.7, 0.05, 0.25, -5.0, -6.0]])
    x_next = torch.tensor([[1010.0, 1.0, 0.7, 0.04, 0.24, -4.5, -6.0]])
    target_scaled = torch.zeros(1, 4)

    rate = validator.validation_true_rate(DummyTools(), target_scaled, x_curr, x_next)
    assert rate.shape == (1, 3)  # 2 macro rate + 1 micro log-rate
    assert torch.isfinite(rate).all()


def _evaluate_until_checkpoint_config(monkeypatch, tmp_path, checkpoint, feature_info):
    class ReachedCheckpointConfig(Exception):
        pass

    monkeypatch.setattr(validator.tr, "CoarseOperatorTools", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        validator,
        "checkpoint_rollout_config",
        lambda _checkpoint: (_ for _ in ()).throw(ReachedCheckpointConfig()),
    )
    args = SimpleNamespace(seed=1, experiment_mode="baseline", state_update_mode="legacy")
    with pytest.raises(ReachedCheckpointConfig):
        validator.evaluate(
            args,
            checkpoint,
            feature_info,
            scaler_x=None,
            scaler_y=None,
            parts=[],
            output_dir=tmp_path,
            data_dir=tmp_path,
        )


def test_validator_loads_resmlp_checkpoint(monkeypatch, tmp_path):
    model = dbn.CoupledKineticsResMLP(
        input_dim=4,
        output_dim=3,
        hidden_dim=8,
        num_blocks=1,
        output_parameterization="qssa_asymptotic",
    )
    checkpoint = {
        "training_config": {
            "network_arch": "resmlp",
            "hidden_dim": 8,
            "num_blocks": 1,
            "output_parameterization": "qssa_asymptotic",
        },
        "model_state_dict": model.state_dict(),
    }
    feature_info = {"input_cols": list(range(4)), "output_cols": list(range(3))}

    _evaluate_until_checkpoint_config(monkeypatch, tmp_path, checkpoint, feature_info)


def test_validator_loads_legacy_dual_branch_checkpoint(monkeypatch, tmp_path):
    element_matrix = torch.tensor([[1.0, 0.0, 1.0]])
    old_model = dbn.DualBranchKineticsNet(
        input_dim=4,
        slow_dim=3,
        fast_dim=2,
        hidden_dim=8,
        num_blocks=1,
        output_parameterization="dual_branch",
        use_null_space=True,
        element_matrix_slow=element_matrix,
    )
    checkpoint = {
        "training_config": {
            "network_arch": "dual_branch",
            "hidden_dim": 8,
            "num_blocks": 1,
            "output_parameterization": "dual_branch",
        },
        "model_state_dict": old_model.state_dict(),
    }
    feature_info = {
        "input_cols": list(range(4)),
        "output_cols": list(range(7)),
        "macro_species": ["A", "B", "C"],
        "micro_species": ["D", "E"],
    }

    _evaluate_until_checkpoint_config(monkeypatch, tmp_path, checkpoint, feature_info)


def test_initial_condition_audit_uses_last_transition_endpoint(tmp_path):
    part = tmp_path / "part-00000.parquet"
    pd.DataFrame(
        {
            "sample_id": ["trajectory", "trajectory"],
            "time": [0.0, 1.0],
            "Y_A": [0.4, 0.3],
            "Y_B": [0.6, 0.5],
            "Y_next_A": [0.35, 0.2],
            "Y_next_B": [0.65, 0.8],
        }
    ).to_parquet(part, index=False)

    result = initial_audit._audit_mechanism("test", [(part, tmp_path)])

    assert result["final_mass_sum_max_abs_error"] == pytest.approx(0.0)
    assert result["final_state_source"] == "Y_next"

