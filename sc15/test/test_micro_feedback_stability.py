import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_sc_v68 as tr


def test_micro_projection_feedback_uses_log_space_ste():
    """A tiny projected mass fraction must not create a 1/Y gradient."""
    raw_log = torch.tensor([[-40.0]], dtype=torch.float32, requires_grad=True)
    projected = torch.tensor([[1.0e-30]], dtype=torch.float32, requires_grad=True)

    constrained_log, constrained_y = tr.constrained_micro_feedback(raw_log, projected)

    torch.testing.assert_close(constrained_log, torch.tensor([[-30.0]]), atol=1e-5, rtol=0)
    torch.testing.assert_close(constrained_y, projected.detach(), atol=1e-36, rtol=1e-5)

    # The backward path is in log coordinates, so it remains finite and bounded.
    (constrained_log + constrained_y).sum().backward()
    assert raw_log.grad is not None
    assert torch.isfinite(raw_log.grad).all()
    assert raw_log.grad.abs().item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert projected.grad is None


def test_linear_space_ste_would_overflow_at_float32_micro_floor():
    """Regression guard documenting the failure mode fixed by the helper above."""
    raw = torch.tensor([[1.01e-40]], dtype=torch.float32, requires_grad=True)
    projected = torch.tensor([[1.01e-40]], dtype=torch.float32)
    bad = raw + (projected - raw).detach()
    torch.log10(bad).sum().backward()

    assert not torch.isfinite(raw.grad).all()


def test_log_space_ste_does_not_reintroduce_linear_log_singularity():
    """The full dual-branch log update stays finite at dormant-radical levels."""
    log_y_old = torch.tensor([[-40.0]], dtype=torch.float32, requires_grad=True)
    z1 = torch.tensor([[-40.0]], dtype=torch.float32, requires_grad=True)
    z2 = torch.tensor([[-30.0]], dtype=torch.float32, requires_grad=True)
    dt = torch.tensor([[1.0e-6]], dtype=torch.float32)
    raw_log = tr.dual_branch_micro_log_update(log_y_old, z1, z2, dt)
    projected = torch.tensor([[1.0e-30]], dtype=torch.float32)

    constrained_log, _ = tr.constrained_micro_feedback(raw_log, projected)
    constrained_log.sum().backward()

    for value in (log_y_old.grad, z1.grad, z2.grad):
        assert value is not None
        assert torch.isfinite(value).all()


def test_dual_branch_prod_only_decodes_z1_as_log_production():
    log_y_old = torch.tensor([[-35.0]], dtype=torch.float32)
    log_production = torch.tensor([[-20.0]], dtype=torch.float32)
    prod_only_sentinel = torch.tensor([[-45.0]], dtype=torch.float32)
    dt = torch.tensor([[1.0e-4]], dtype=torch.float32)

    actual = tr.dual_branch_micro_log_update(
        log_y_old, log_production, prod_only_sentinel, dt
    )
    expected = torch.log10(torch.tensor([[1.0e-35 + 1.0e-24]]))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
