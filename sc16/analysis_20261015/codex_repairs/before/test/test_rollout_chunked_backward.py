import torch

import train_sc_v68 as tr


class LinearRolloutTools:
    def __init__(self):
        self.macro_idx = torch.tensor([0])
        self.micro_idx = torch.tensor([1])
        self.mean_X = torch.zeros(2)
        self.scale_X = torch.ones(2)
        self.rate_decode_mode = "direct_rate"

    def inverse_qssa_asymptotic_log_parameters(self, scaled_rate):
        return scaled_rate[:, :1], scaled_rate[:, 1:2]

    def predict_next_state(self, current_X, rate_pred, **_kwargs):
        macro = current_X[:, :1] + 0.05 * rate_pred[:, :1]
        log_micro = current_X[:, 1:2] + 0.03 * rate_pred[:, 1:2]
        return {
            "Y_macro_next": macro,
            "Y_macro_next_raw": macro,
            "Y_macro_next_constrained": macro,
            "logY_micro_next": log_micro,
            "logY_micro_next_raw": log_micro,
            "logY_micro_next_constrained": log_micro,
        }

    @staticmethod
    def state_to_input_scaled(_base, macro, log_micro, **_kwargs):
        return torch.cat([macro, log_micro], dim=1)


def _run(model, tools, batch_X, batch_Y=None, mode="direct_rate", backward_callback=None):
    tools.rate_decode_mode = mode
    return tr.compute_training_rollout_loss(
        model=model,
        tools=tools,
        batch_X=batch_X,
        active_horizon=8,
        experiment_mode="baseline",
        batch_Y=batch_Y,
        horizons=(4, 8),
        detach_interval=4,
        lambda_stability=0.3,
        state_update_mode=mode,
        state_loss_source="raw",
        feedback_state_source="raw",
        backward_callback=backward_callback,
    )


@torch.no_grad()
def _make_data():
    torch.manual_seed(123)
    batch_X = torch.randn(3, 9, 2) * 0.1
    batch_Y = torch.randn(3, 8, 2) * 0.1
    return batch_X, batch_Y


def test_chunked_backward_matches_full_rollout_loss_and_gradients():
    for mode in ("direct_rate", "qssa_asymptotic"):
        torch.manual_seed(123)
        tools = LinearRolloutTools()
        batch_X, batch_Y = _make_data()
        model = torch.nn.Linear(2, 2, bias=True)
        model_reference = torch.nn.Linear(2, 2, bias=True)
        model_reference.load_state_dict(model.state_dict())

        full_loss, _ = _run(
            model_reference,
            tools,
            batch_X,
            batch_Y=batch_Y,
            mode=mode,
        )
        full_loss.backward()
        expected_grads = [
            parameter.grad.detach().clone()
            for parameter in model_reference.parameters()
        ]

        chunked_loss, _ = _run(
            model,
            tools,
            batch_X,
            batch_Y=batch_Y,
            mode=mode,
            backward_callback=lambda chunk_loss: chunk_loss.backward(),
        )

        assert not chunked_loss.requires_grad
        torch.testing.assert_close(chunked_loss, full_loss.detach(), rtol=1e-6, atol=1e-8)
        for parameter, expected in zip(model.parameters(), expected_grads):
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-7)
