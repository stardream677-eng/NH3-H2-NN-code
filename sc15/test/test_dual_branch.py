"""Unit tests for Phase 5: ResMLP Backbone and DualBranchKineticsNet."""

import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from dual_branch_network import (
    ResMLPBlock,
    ResMLPBackbone,
    CoupledKineticsResMLP,
    DualBranchKineticsNet,
)


def test_resmlp_block():
    print("Testing ResMLPBlock forward pass and residual connection...")
    dim = 128
    block = ResMLPBlock(dim)
    x = torch.randn(8, dim)
    out = block(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x), "Block output should differ from input"
    # Gradient test
    loss = out.sum()
    loss.backward()
    for p in block.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    print(">> ResMLPBlock test PASSED!\n")


def test_coupled_kinetics_resmlp():
    print("Testing CoupledKineticsResMLP drop-in replacement...")
    in_dim = 36
    out_dim = 60
    model = CoupledKineticsResMLP(in_dim, out_dim, hidden_dim=256, num_blocks=3)
    
    # 2D batch [batch, in_dim]
    x2d = torch.randn(16, in_dim)
    out2d = model(x2d)
    assert out2d.shape == (16, out_dim)

    # 3D rollout batch [batch, horizon, in_dim]
    x3d = torch.randn(4, 32, in_dim)
    out3d = model(x3d)
    assert out3d.shape == (4, 32, out_dim)
    print(">> CoupledKineticsResMLP test PASSED!\n")


def test_dual_branch_net():
    print("Testing DualBranchKineticsNet multi-timescale separation...")
    in_dim = 36
    slow_dim = 7  # H2, H2O, N2, NH3, O2, NO, N2O
    fast_dim = 25 # radicals
    
    # Mock elemental matrix for slow species [N, H, O] x 7 species
    A_slow = torch.tensor([
        # H2, H2O, N2, NH3, O2, NO, N2O
        [0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0],  # N
        [2.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0],  # H
        [0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0],  # O
    ], dtype=torch.float32)

    model = DualBranchKineticsNet(
        input_dim=in_dim,
        slow_dim=slow_dim,
        fast_dim=fast_dim,
        hidden_dim=256,
        num_blocks=3,
        use_null_space=True,
        element_matrix_slow=A_slow,
    )

    x = torch.randn(8, in_dim)
    
    # 1. Test dictionary output mode
    out_dict = model(x, return_dict=True)
    assert "slow_delta" in out_dict
    assert "fast_qssa" in out_dict
    assert out_dict["slow_delta"].shape == (8, slow_dim)
    assert out_dict["fast_qssa"].shape == (8, 2 * fast_dim)

    # 2. Test null-space elemental conservation on ΔY_slow
    delta_slow = out_dict["slow_delta"]
    elem_residual = torch.matmul(delta_slow, A_slow.T)
    max_elem_err = elem_residual.abs().max().item()
    print(f"  Slow branch null-space elemental residual: {max_elem_err:.2e}")
    assert max_elem_err < 1e-5, f"Null-space conservation violated: {max_elem_err}"

    # 3. Test concatenated output mode for standard training pipelines
    out_cat = model(x, return_dict=False)
    expected_dim = slow_dim + 2 * fast_dim
    assert out_cat.shape == (8, expected_dim)

    # 4. Backward gradient flow through both heads
    loss = out_dict["slow_delta"].pow(2).sum() + out_dict["fast_qssa"].pow(2).sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"NaN gradient in {name}"

    print(">> DualBranchKineticsNet test PASSED!\n")


if __name__ == "__main__":
    test_resmlp_block()
    test_coupled_kinetics_resmlp()
    test_dual_branch_net()
    print("=========================================")
    print("ALL PHASE 5 DUAL-BRANCH TESTS PASSED!")
    print("=========================================")
